import Foundation
import AIChatCore

/// A declared tool reduced to what routing needs: its name and enough of its parameter schema to
/// validate and lightly coerce the arguments a small router model produced.
struct RoutableTool: Sendable, Equatable {
    struct Property: Sendable, Equatable {
        var type: String?
        var enumValues: [String]?
    }

    let name: String
    let required: [String]
    let properties: [String: Property]
    /// FunctionGemma's canonical `declaration:…` text for this tool (see ``FunctionGemmaDeclaration``).
    let declaration: String

    /// Extracts the declared tools from request options, preferring `tools` over
    /// `nativeToolSpecs` — the same precedence `MLXProvider` uses when it builds the template's
    /// `tools=` array.
    static func declared(in options: ChatRequestOptions) -> [RoutableTool] {
        if let tools = options.tools, !tools.isEmpty {
            let encoder = JSONEncoder()
            return tools.map { tool in
                var params: [String: Any] = [:]
                if let schema = tool.parameters,
                   let data = try? encoder.encode(schema),
                   let dict = try? JSONSerialization.jsonObject(with: data) as? [String: Any] {
                    params = dict
                }
                return RoutableTool(name: tool.name, description: tool.description, parameters: params)
            }
        }
        return (options.nativeToolSpecs ?? []).compactMap { spec in
            guard let function = spec["function"] as? [String: Any],
                  let name = function["name"] as? String
            else { return nil }
            return RoutableTool(
                name: name, description: function["description"] as? String,
                parameters: function["parameters"] as? [String: Any] ?? [:]
            )
        }
    }

    init(name: String, description: String? = nil, parameters: [String: Any]) {
        self.name = name
        self.declaration = FunctionGemmaDeclaration.render(name: name, description: description, parameters: parameters)
        self.required = parameters["required"] as? [String] ?? []
        var props: [String: Property] = [:]
        for (key, value) in parameters["properties"] as? [String: Any] ?? [:] {
            guard let dict = value as? [String: Any] else { continue }
            props[key] = Property(
                type: (dict["type"] as? String)?.lowercased(),
                enumValues: dict["enum"] as? [String]
            )
        }
        self.properties = props
    }

    enum ValidationFailure: Error, Equatable {
        case unknownTool(String)
        case invalidArguments(String)
    }

    /// Validates a router-produced call against `tools` and returns arguments JSON that matches
    /// the declared schema.
    ///
    /// Coercions are deliberately obvious ones only: numeric/boolean strings to numbers/booleans,
    /// scalars to strings, a lone scalar to a one-element array, and case-insensitive enum
    /// matches. Keys the schema does not declare are dropped. A missing required key, a value
    /// that cannot be coerced, or an enum value outside the declared set is a failure — the caller
    /// then falls back rather than executing a guess.
    static func validate(
        name: String, argumentsJSON: String, against tools: [RoutableTool]
    ) -> Result<String, ValidationFailure> {
        guard let tool = tools.first(where: { $0.name == name }) else {
            return .failure(.unknownTool(name))
        }
        let trimmed = argumentsJSON.trimmingCharacters(in: .whitespacesAndNewlines)
        var args: [String: Any] = [:]
        if !trimmed.isEmpty {
            guard let data = trimmed.data(using: .utf8),
                  let object = try? JSONSerialization.jsonObject(with: data) as? [String: Any]
            else { return .failure(.invalidArguments("arguments are not a JSON object")) }
            args = object
        }

        if !tool.properties.isEmpty {
            args = args.filter { tool.properties[$0.key] != nil }
        }
        for (key, property) in tool.properties {
            guard let raw = args[key] else { continue }
            if raw is NSNull { args[key] = nil; continue }
            guard let coerced = coerce(raw, to: property) else {
                return .failure(.invalidArguments("'\(key)' is not a valid \(property.type ?? "value")"))
            }
            args[key] = coerced
        }
        for key in tool.required {
            let value = args[key]
            let isBlank = (value as? String)?.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty == true
            if value == nil || isBlank {
                return .failure(.invalidArguments("missing required '\(key)'"))
            }
        }
        guard let out = try? JSONSerialization.data(withJSONObject: args, options: [.sortedKeys]),
              let json = String(data: out, encoding: .utf8)
        else { return .failure(.invalidArguments("arguments could not be re-encoded")) }
        return .success(json)
    }

    private static func isBool(_ value: Any) -> Bool {
        guard let number = value as? NSNumber else { return false }
        return CFGetTypeID(number) == CFBooleanGetTypeID()
    }

    private static func coerce(_ value: Any, to property: Property) -> Any? {
        if let allowed = property.enumValues {
            guard let s = value as? String,
                  let match = allowed.first(where: { $0.caseInsensitiveCompare(s.trimmingCharacters(in: .whitespaces)) == .orderedSame })
            else { return nil }
            return match
        }
        switch property.type {
        case "integer":
            if isBool(value) { return nil }
            if let n = value as? NSNumber { return n.doubleValue == n.doubleValue.rounded() ? n.intValue : nil }
            if let s = value as? String {
                let t = s.trimmingCharacters(in: .whitespaces)
                if let i = Int(t) { return i }
                if let d = Double(t), d == d.rounded() { return Int(d) }
            }
            return nil
        case "number":
            if isBool(value) { return nil }
            if let n = value as? NSNumber { return n }
            if let s = value as? String, let d = Double(s.trimmingCharacters(in: .whitespaces)) { return d }
            return nil
        case "boolean":
            if isBool(value) { return value }
            if let s = (value as? String)?.lowercased() {
                if ["true", "yes"].contains(s) { return true }
                if ["false", "no"].contains(s) { return false }
            }
            return nil
        case "string":
            if let s = value as? String { return s }
            if isBool(value) { return (value as? NSNumber)?.boolValue == true ? "true" : "false" }
            if let n = value as? NSNumber { return n.stringValue }
            return nil
        case "array":
            if value is [Any] { return value }
            if value is [String: Any] { return nil }
            return [value]
        case "object":
            return value is [String: Any] ? value : nil
        default:
            return value
        }
    }
}

/// Builds the minimal prompt the router sees: the last few conversational turns, nothing else.
enum RouterPrompt {
    /// Text of a message's text blocks, or `nil` when it has none.
    private static func text(of message: ChatMessage) -> String? {
        let parts = message.content.compactMap { block -> String? in
            if case .text(let t) = block { return t }
            return nil
        }
        let joined = parts.joined(separator: "\n").trimmingCharacters(in: .whitespacesAndNewlines)
        return joined.isEmpty ? nil : joined
    }

    /// Head + tail of an over-long turn, so a pasted document neither bloats the router prompt nor
    /// hides the request at its end.
    static func clip(_ text: String, limit: Int) -> String {
        guard limit > 0, text.count > limit else { return text }
        let half = limit / 2
        return String(text.prefix(half)) + " … " + String(text.suffix(half))
    }

    /// The last `turns` user turns (with the assistant replies between them), text only.
    static func recentTurns(_ messages: [ChatMessage], turns: Int, characterLimit: Int) -> [ChatMessage] {
        let plain = messages.filter {
            ($0.role == .user || $0.role == .assistant) && $0.toolCalls?.isEmpty != false
        }
        let userIndices = plain.indices.filter { plain[$0].role == .user }
        guard let start = userIndices.suffix(max(1, turns)).first else { return [] }
        return plain[start...].compactMap { message in
            text(of: message).map {
                ChatMessage(role: message.role, content: clip($0, limit: characterLimit))
            }
        }
    }

    /// Context for re-routing after tool results: the originating user request plus each result
    /// as plain text (the router's chat template needs a tool `name` on tool messages, which the
    /// provider-agnostic history does not carry).
    static func withToolResults(_ messages: [ChatMessage], characterLimit: Int) -> [ChatMessage] {
        guard let userIndex = messages.lastIndex(where: { $0.role == .user }),
              let request = text(of: messages[userIndex])
        else { return [] }
        var names: [String: String] = [:]
        var lines: [String] = []
        for message in messages[(userIndex + 1)...] {
            for call in message.toolCalls ?? [] { names[call.id] = call.name }
            if message.role == .tool, let result = text(of: message) {
                let name = message.toolCallId.flatMap { names[$0] } ?? "tool"
                lines.append("Result of \(name): \(clip(result, limit: characterLimit))")
            }
        }
        let body = ([clip(request, limit: characterLimit)] + lines).joined(separator: "\n")
        return [ChatMessage(role: .user, content: body)]
    }
}

/// Renders tool declarations in FunctionGemma's canonical prompt syntax, byte for byte what the
/// model's Jinja chat template produces under a reference (Python) Jinja engine:
///
/// `<start_function_declaration>declaration:name{description:<escape>…<escape>,parameters:{properties:{…},required:[…],type:<escape>OBJECT<escape>}}<end_function_declaration>`
///
/// Why this exists: rendering through `swift-jinja` (what `MLXProvider` uses for `tools=`) leaves
/// stray spaces after `parameters:{` and `properties:{` that the template's `{{-` markers strip in
/// the reference engine. A 270M model fine-tuned on the exact format is sensitive to that, so the
/// router can be fed this text inline instead (``ToolRoutingProvider/RouterToolFormat``).
enum FunctionGemmaDeclaration {
    private static let standardKeys: Set<String> = ["description", "type", "properties", "required", "nullable"]

    static func wrap(_ declarations: [RoutableTool]) -> String {
        declarations.map { "<start_function_declaration>\($0.declaration)<end_function_declaration>" }.joined()
    }

    static func render(name: String, description: String?, parameters: [String: Any]) -> String {
        var out = "declaration:\(name){description:\(esc(description ?? ""))"
        if !parameters.isEmpty {
            out += ",parameters:{"
            if let props = parameters["properties"] as? [String: Any], !props.isEmpty {
                out += "properties:{\(properties(props))},"
            }
            if let required = parameters["required"] as? [String], !required.isEmpty {
                out += "required:[\(required.map(esc).joined(separator: ","))],"
            }
            out += "type:\(esc(((parameters["type"] as? String) ?? "object").uppercased()))}"
        }
        return out + "}"
    }

    private static func esc(_ s: String) -> String { "<escape>\(s)<escape>" }

    private static func properties(_ props: [String: Any]) -> String {
        props.keys.sorted().filter { !standardKeys.contains($0) }.compactMap { key -> String? in
            guard let value = props[key] as? [String: Any] else { return nil }
            let type = ((value["type"] as? String) ?? "string").uppercased()
            var out = "\(key):{description:\(esc(value["description"] as? String ?? ""))"
            switch type {
            case "STRING":
                if let e = value["enum"] as? [String], !e.isEmpty { out += ",enum:[\(e.map(esc).joined(separator: ","))]" }
            case "OBJECT":
                if let nested = value["properties"] as? [String: Any] { out += ",properties:{\(properties(nested))}" }
                if let r = value["required"] as? [String], !r.isEmpty { out += ",required:[\(r.map(esc).joined(separator: ","))]" }
            case "ARRAY":
                if let items = value["items"] as? [String: Any], !items.isEmpty {
                    let parts = items.keys.sorted().map { k -> String in
                        if k == "type" { return "type:\(esc(((items[k] as? String) ?? "string").uppercased()))" }
                        return "\(k):\(argument(items[k] as Any))"
                    }
                    out += ",items:{\(parts.joined(separator: ","))}"
                }
            default: break
            }
            return out + ",type:\(esc(type))}"
        }.joined(separator: ",")
    }

    /// The template's `format_argument`: strings escaped, booleans bare, containers recursive.
    private static func argument(_ value: Any) -> String {
        if let s = value as? String { return esc(s) }
        if let n = value as? NSNumber {
            return CFGetTypeID(n) == CFBooleanGetTypeID() ? (n.boolValue ? "true" : "false") : "\(n)"
        }
        if let d = value as? [String: Any] {
            return "{" + d.keys.sorted().map { "\(esc($0)):\(argument(d[$0] as Any))" }.joined(separator: ",") + "}"
        }
        if let a = value as? [Any] { return "[" + a.map(argument).joined(separator: ",") + "]" }
        return "\(value)"
    }
}
