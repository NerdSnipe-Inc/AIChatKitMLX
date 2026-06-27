import Foundation
import MLXLMCommon

/// Splits Gemma 4 streamed output into reasoning (thought channel), user-visible text,
/// and native `call:name{...}` / `<|tool_call>` tool calls.
public struct Gemma4StreamProcessor: Sendable {

    /// Streaming events emitted by `Gemma4StreamProcessor`.
    public enum Event: Sendable, Equatable {
        /// A model "thought" channel delta.
        case reasoning(String)
        /// User-visible assistant text delta.
        case text(String)
        /// A fully parsed inline or native tool call.
        ///
        /// - Parameters:
        ///   - name: Tool function name.
        ///   - argumentsJSON: Serialized JSON arguments object.
        case toolCall(name: String, argumentsJSON: String)
    }

    private enum Phase {
        case preamble
        case thought
        case response
    }

    private static let thoughtStart = "<|channel>thought"
    private static let channelEnd = "<channel|>"
    private static let maxMarkerHold = 32

    private var phase: Phase = .preamble
    private var buffer = ""
    private let toolProcessor: ToolCallProcessor
    private var emittedToolCount = 0
    private var inlineCallBuffer = ""

    /// Creates a processor for a specific streamed response.
    ///
    /// - Parameter tools: Native tool schema payload forwarded to `ToolCallProcessor`.
    public init(tools: [[String: any Sendable]]?) {
        self.toolProcessor = ToolCallProcessor(format: .gemma, tools: tools)
    }

    /// Ingests the next streamed token chunk and emits any parsed events.
    ///
    /// - Parameter chunk: Raw text chunk from MLX generation output.
    /// - Returns: Zero or more parsed events that became complete after appending `chunk`.
    public mutating func processChunk(_ chunk: String) -> [Event] {
        buffer += chunk
        return drain()
    }

    /// Flushes any buffered partial state when the stream ends.
    ///
    /// Call this once after the model stream finishes to emit remaining text/tool calls.
    ///
    /// - Returns: Final events derived from buffered content and native tool processor state.
    public mutating func finish() -> [Event] {
        if !buffer.isEmpty {
            switch phase {
            case .thought:
                return emitReasoning(buffer) + flushTools()
            case .preamble, .response:
                return emitResponse(buffer, flushRemainder: true) + flushTools()
            }
        }
        return flushTools()
    }

    // MARK: - Drain loop

    private mutating func drain() -> [Event] {
        var events: [Event] = []
        while true {
            switch phase {
            case .preamble:
                if let range = buffer.range(of: Self.thoughtStart) {
                    let before = String(buffer[..<range.lowerBound])
                    if !before.isEmpty {
                        events += emitResponse(before, flushRemainder: false)
                    }
                    buffer = String(buffer[range.upperBound...])
                    phase = .thought
                    continue
                }
                if buffer.hasPrefix("<|channel>") || couldBePartialMarker(buffer, marker: Self.thoughtStart) {
                    return events
                }
                events += emitResponse(buffer, flushRemainder: false)
                buffer = ""
                phase = .response
                return events

            case .thought:
                if let range = buffer.range(of: Self.channelEnd) {
                    let thought = String(buffer[..<range.lowerBound])
                    events += emitReasoning(thought)
                    buffer = String(buffer[range.upperBound...])
                    phase = .response
                    continue
                }
                if couldBePartialMarker(buffer, marker: Self.channelEnd) {
                    let split = holdSuffix(buffer)
                    if !split.safe.isEmpty {
                        events += emitReasoning(split.safe)
                        buffer = split.keep
                    }
                    return events
                }
                events += emitReasoning(buffer)
                buffer = ""
                return events

            case .response:
                if let range = buffer.range(of: Self.thoughtStart) {
                    let before = String(buffer[..<range.lowerBound])
                    if !before.isEmpty {
                        events += emitResponse(before, flushRemainder: false)
                    }
                    buffer = String(buffer[range.upperBound...])
                    phase = .thought
                    continue
                }
                if couldBePartialMarker(buffer, marker: Self.thoughtStart) {
                    let split = holdSuffix(buffer)
                    if !split.safe.isEmpty {
                        events += emitResponse(split.safe, flushRemainder: false)
                        buffer = split.keep
                    }
                    return events
                }
                events += emitResponse(buffer, flushRemainder: false)
                buffer = ""
                return events
            }
        }
    }

    // MARK: - Emit helpers

    private mutating func emitReasoning(_ text: String) -> [Event] {
        guard !text.isEmpty else { return [] }
        return [.reasoning(text)]
    }

    private mutating func emitResponse(_ text: String, flushRemainder: Bool) -> [Event] {
        guard !text.isEmpty else { return [] }
        var events: [Event] = []
        var remainder = text

        if !inlineCallBuffer.isEmpty {
            remainder = inlineCallBuffer + remainder
            inlineCallBuffer = ""
        }

        while !remainder.isEmpty {
            if let tagged = toolProcessor.processChunk(remainder) {
                events += emitInlineOrText(tagged, flushRemainder: flushRemainder)
                remainder = ""
            } else if let callRange = remainder.range(of: "call:") {
                let leading = String(remainder[..<callRange.lowerBound])
                if !leading.isEmpty {
                    events.append(.text(leading))
                }
                remainder = String(remainder[callRange.lowerBound...])
                if let (call, rest) = GemmaInlineCallParser.extractFirst(from: remainder, allowPartial: !flushRemainder) {
                    events.append(.toolCall(name: call.name, argumentsJSON: call.argumentsJSON))
                    remainder = rest
                } else {
                    if flushRemainder {
                        events.append(.text(remainder))
                        remainder = ""
                    } else {
                        inlineCallBuffer = remainder
                        remainder = ""
                    }
                    break
                }
            } else if flushRemainder {
                events.append(.text(remainder))
                remainder = ""
            } else {
                let split = holdSuffix(remainder, extraMarkers: ["call:"])
                if !split.safe.isEmpty {
                    events += emitInlineOrText(split.safe, flushRemainder: false)
                }
                inlineCallBuffer = split.keep
                remainder = ""
            }
        }

        return events
    }

    private mutating func emitInlineOrText(_ text: String, flushRemainder: Bool) -> [Event] {
        var events: [Event] = []
        var remainder = text
        while !remainder.isEmpty {
            if let callRange = remainder.range(of: "call:") {
                let leading = String(remainder[..<callRange.lowerBound])
                if !leading.isEmpty { events.append(.text(leading)) }
                remainder = String(remainder[callRange.lowerBound...])
                if let (call, rest) = GemmaInlineCallParser.extractFirst(from: remainder, allowPartial: !flushRemainder) {
                    events.append(.toolCall(name: call.name, argumentsJSON: call.argumentsJSON))
                    remainder = rest
                } else if flushRemainder {
                    events.append(.text(remainder))
                    remainder = ""
                } else {
                    inlineCallBuffer = remainder
                    remainder = ""
                }
            } else {
                events.append(.text(remainder))
                remainder = ""
            }
        }
        return events
    }

    private mutating func flushTools() -> [Event] {
        toolProcessor.processEOS()
        var events: [Event] = []
        let newCalls = toolProcessor.toolCalls.dropFirst(emittedToolCount)
        for tc in newCalls {
            let argsJSON = Self.serializeArguments(tc.function.arguments)
            events.append(.toolCall(name: tc.function.name, argumentsJSON: argsJSON))
        }
        emittedToolCount = toolProcessor.toolCalls.count
        return events
    }

    private static func serializeArguments(_ args: [String: JSONValue]) -> String {
        let plain = args.mapValues { $0.anyValue }
        guard let data = try? JSONSerialization.data(withJSONObject: plain),
              let str = String(data: data, encoding: .utf8) else { return "{}" }
        return str
    }

    // MARK: - Marker helpers

    private func couldBePartialMarker(_ text: String, marker: String) -> Bool {
        guard !text.isEmpty, text.count < marker.count else { return false }
        return marker.hasPrefix(text) || text.hasSuffix("<") || text.contains("<|")
    }

    private func holdSuffix(_ text: String, extraMarkers: [String] = []) -> (safe: String, keep: String) {
        let markers = [Self.thoughtStart, Self.channelEnd, "call:"] + extraMarkers
        let hold = min(Self.maxMarkerHold, text.count)
        guard hold > 0, text.count > hold else { return ("", text) }
        let safeEnd = text.index(text.endIndex, offsetBy: -hold)
        let safe = String(text[..<safeEnd])
        let keep = String(text[safeEnd...])
        for marker in markers where keep.count < marker.count && marker.hasPrefix(keep) {
            return (safe, keep)
        }
        return (text, "")
    }
}

// MARK: - Inline call: parser

enum GemmaInlineCallParser {

    struct ParsedCall: Sendable {
        let name: String
        let argumentsJSON: String
    }

    static func extractFirst(from text: String, allowPartial: Bool) -> (ParsedCall, String)? {
        guard text.hasPrefix("call:") else { return nil }
        guard let braceStart = text.firstIndex(of: "{") else { return nil }
        guard let braceEnd = balancedBraceEnd(in: text, from: braceStart) else {
            return allowPartial ? nil : nil
        }
        let nameStart = text.index(text.startIndex, offsetBy: 5)
        let name = String(text[nameStart..<braceStart])
        guard !name.isEmpty else { return nil }

        let argsBody = String(text[text.index(after: braceStart)..<braceEnd])
        let json = gemma4ArgsToJSON(argsBody)
        guard let data = json.data(using: .utf8),
              let _ = try? JSONSerialization.jsonObject(with: data) else { return nil }

        let consumedEnd = text.index(after: braceEnd)
        let rest = String(text[consumedEnd...])
        return (ParsedCall(name: name, argumentsJSON: json), rest)
    }

    static func parseAll(from text: String) -> (cleaned: String, calls: [ParsedCall]) {
        var remainder = text
        var calls: [ParsedCall] = []
        while let (call, rest) = extractFirst(from: remainder, allowPartial: false) {
            calls.append(call)
            remainder = rest
        }
        return (remainder.trimmingCharacters(in: .whitespacesAndNewlines), calls)
    }

    private static func balancedBraceEnd(in text: String, from start: String.Index) -> String.Index? {
        guard text[start] == "{" else { return nil }
        var depth = 0
        var i = start
        while i < text.endIndex {
            let ch = text[i]
            if ch == "{" { depth += 1 }
            else if ch == "}" {
                depth -= 1
                if depth == 0 { return i }
            }
            i = text.index(after: i)
        }
        return nil
    }

    private static func gemma4ArgsToJSON(_ body: String) -> String {
        var strings: [String] = []
        var working = body

        while let start = working.range(of: #"<|"|>"#) {
            guard let end = working.range(of: #"<|"|>"#, range: start.upperBound..<working.endIndex) else { break }
            let value = String(working[start.upperBound..<end.lowerBound])
            strings.append(value)
            let placeholder = "\u{0000}\(strings.count - 1)\u{0000}"
            working.replaceSubrange(start.lowerBound..<end.upperBound, with: placeholder)
        }

        var json = working.replacingOccurrences(
            of: #"(^|[{,]\s*)(\w+)\s*:"#,
            with: "$1\"$2\":",
            options: .regularExpression
        )

        for (idx, value) in strings.enumerated() {
            let escaped = value
                .replacingOccurrences(of: "\\", with: "\\\\")
                .replacingOccurrences(of: "\"", with: "\\\"")
            json = json.replacingOccurrences(of: "\u{0000}\(idx)\u{0000}", with: "\"\(escaped)\"")
        }

        if !json.hasPrefix("{") { json = "{\(json)}" }
        return json
    }
}
