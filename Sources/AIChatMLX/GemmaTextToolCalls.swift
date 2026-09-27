import Foundation

/// Tool calls a Gemma model wrote as plain text instead of native tool tokens — the formats a
/// LoRA or the Jinja fallback template produce: `<tool_call>{json}</tool_call>` blocks and
/// ` ```tool_code ` fenced Python calls. `Gemma4StreamProcessor` uses these to turn them into
/// tool-call events at stream level, so a host never has to scrape assistant text.
enum GemmaTextToolCalls {

    struct Call: Equatable {
        let name: String
        let argumentsJSON: String
    }

    /// Parses the JSON object between `<tool_call>` and `</tool_call>`:
    /// `{"name": "...", "arguments": {...}}`. Returns `nil` when it is not such an object.
    static func parseXMLBody(_ body: String) -> Call? {
        guard let data = body.trimmingCharacters(in: .whitespacesAndNewlines).data(using: .utf8),
              let obj = try? JSONSerialization.jsonObject(with: data) as? [String: Any],
              let name = obj["name"] as? String else { return nil }
        var args = "{}"
        if let argObj = obj["arguments"],
           let argData = try? JSONSerialization.data(withJSONObject: argObj),
           let str = String(data: argData, encoding: .utf8) {
            args = str
        }
        return Call(name: name, argumentsJSON: args)
    }

    /// Parses one or more Python-style `funcName(...)` calls from a block of text.
    static func parsePythonCalls(
        from text: String,
        schemas: [[String: any Sendable]]?
    ) -> [Call] {
        // Match: identifier followed by ( ... )
        let pattern = #"(\w+)\s*\(([^)]*)\)"#
        guard let regex = try? NSRegularExpression(pattern: pattern, options: []) else { return [] }
        let ns = text as NSString
        let matches = regex.matches(in: text, range: NSRange(location: 0, length: ns.length))

        return matches.compactMap { match -> Call? in
            guard match.numberOfRanges > 2 else { return nil }
            let name = ns.substring(with: match.range(at: 1))
            let argsRaw = ns.substring(with: match.range(at: 2)).trimmingCharacters(in: .whitespaces)
            guard !name.isEmpty else { return nil }

            let argsJSON = pythonArgsToJSON(argsRaw, toolName: name, schemas: schemas)
            return Call(name: name, argumentsJSON: argsJSON)
        }
    }

    /// Converts a Python arg string to JSON. Handles:
    ///   - Keyword args:  `query="value", key2="v2"` → `{"query":"value","key2":"v2"}`
    ///   - Positional str: `"value"` → first required param name from schema, else `"query"`
    private static func pythonArgsToJSON(
        _ argsRaw: String,
        toolName: String,
        schemas: [[String: any Sendable]]?
    ) -> String {
        let trimmed = argsRaw.trimmingCharacters(in: .whitespaces)
        guard !trimmed.isEmpty else { return "{}" }

        // Try keyword args first: key="value" or key='value' pairs.
        let kwPattern = #"(\w+)\s*=\s*(?:"([^"\\]*(\\.[^"\\]*)*)"|'([^'\\]*(\\.[^'\\]*)*)')"#
        if let kwRegex = try? NSRegularExpression(pattern: kwPattern) {
            let ns = trimmed as NSString
            let matches = kwRegex.matches(in: trimmed, range: NSRange(location: 0, length: ns.length))
            if !matches.isEmpty {
                var dict: [String: String] = [:]
                for m in matches {
                    let key = ns.substring(with: m.range(at: 1))
                    // Group 2 = double-quoted value, group 4 = single-quoted value
                    let val = m.range(at: 2).location != NSNotFound
                        ? ns.substring(with: m.range(at: 2))
                        : (m.range(at: 4).location != NSNotFound ? ns.substring(with: m.range(at: 4)) : "")
                    dict[key] = val
                }
                if let data = try? JSONSerialization.data(withJSONObject: dict),
                   let str = String(data: data, encoding: .utf8) {
                    return str
                }
            }
        }

        // Positional single string arg: "value" or 'value'.
        let posPattern = #"^(?:"([^"\\]*(\\.[^"\\]*)*)"|'([^'\\]*(\\.[^'\\]*)*)')\s*$"#
        if let posRegex = try? NSRegularExpression(pattern: posPattern),
           let match = posRegex.firstMatch(in: trimmed, range: NSRange(location: 0, length: (trimmed as NSString).length)) {
            let ns = trimmed as NSString
            let value = match.range(at: 1).location != NSNotFound
                ? ns.substring(with: match.range(at: 1))
                : (match.range(at: 3).location != NSNotFound ? ns.substring(with: match.range(at: 3)) : trimmed)

            // Look up the first required parameter name from the tool schema.
            let paramName = firstParamName(for: toolName, schemas: schemas) ?? "query"
            if let data = try? JSONSerialization.data(withJSONObject: [paramName: value]),
               let str = String(data: data, encoding: .utf8) {
                return str
            }
        }

        return "{}"
    }

    /// Returns the first required (or first defined) parameter name for a tool from its schema.
    private static func firstParamName(
        for toolName: String,
        schemas: [[String: any Sendable]]?
    ) -> String? {
        guard let schemas else { return nil }
        for spec in schemas {
            guard let fn = spec["function"] as? [String: any Sendable],
                  fn["name"] as? String == toolName,
                  let params = fn["parameters"] as? [String: any Sendable],
                  let props = params["properties"] as? [String: any Sendable]
            else { continue }

            // Prefer the first required param, then any param.
            let required = (params["required"] as? [String]) ?? []
            if let first = required.first { return first }
            return props.keys.first
        }
        return nil
    }
}
