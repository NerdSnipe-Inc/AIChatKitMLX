import Foundation

/// Splits Gemma 4 streamed output into reasoning (thought channel), user-visible text,
/// and native `call:name{...}` / `<|tool_call>` tool calls.
public struct Gemma4StreamProcessor {

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
        case response
        case thought
        case toolBlock
    }

    private static let thoughtStart = "<|channel>thought"
    private static let channelEnd = "<channel|>"
    private static let toolStart = "<|tool_call>"
    private static let toolEnd = "<tool_call|>"
    /// Markers that are recognised (and hidden from output) while in the response phase.
    private static let responseMarkers = [thoughtStart, toolStart, channelEnd, toolEnd]

    private var phase: Phase = .response
    private var buffer = ""

    /// Creates a processor for a specific streamed response.
    ///
    /// - Parameter tools: Retained for API compatibility; argument typing is derived from the
    ///   call syntax itself (quoted strings vs bare numbers/booleans).
    public init(tools: [[String: any Sendable]]?) {}

    /// Ingests the next streamed token chunk and emits any parsed events.
    ///
    /// Markers and calls may be split across chunks at any character boundary; anything that could
    /// still turn out to be part of a marker is held back until the next chunk (or `finish()`).
    ///
    /// - Parameter chunk: Raw text chunk from MLX generation output.
    /// - Returns: Zero or more parsed events that became complete after appending `chunk`.
    public mutating func processChunk(_ chunk: String) -> [Event] {
        buffer += chunk
        return drain(final: false)
    }

    /// Flushes any buffered partial state when the stream ends. Unfinished constructs are never
    /// dropped: an unterminated thought is emitted as reasoning, and an unparseable or truncated
    /// call is emitted as visible text so the user can see what the model produced.
    ///
    /// - Returns: Remaining events.
    public mutating func finish() -> [Event] {
        let events = drain(final: true)
        buffer = ""
        return events
    }

    // MARK: - Drain loop

    private mutating func drain(final: Bool) -> [Event] {
        var events: [Event] = []
        while true {
            switch phase {
            case .response:
                guard drainResponse(&events, final: final) else { return events }
            case .thought:
                guard drainThought(&events, final: final) else { return events }
            case .toolBlock:
                guard drainToolBlock(&events, final: final) else { return events }
            }
        }
    }

    /// Returns `true` when the loop should continue (phase changed / more to consume).
    private mutating func drainResponse(_ events: inout [Event], final: Bool) -> Bool {
        if buffer.isEmpty { return false }

        // Earliest structural marker.
        var marker: (range: Range<String.Index>, text: String)?
        for m in Self.responseMarkers {
            if let r = buffer.range(of: m), marker == nil || r.lowerBound < marker!.range.lowerBound {
                marker = (r, m)
            }
        }
        // A bare `call:` before that marker.
        if let cand = GemmaCallSyntax.nextCallCandidate(in: buffer, before: marker?.range.lowerBound) {
            switch GemmaCallSyntax.scan(buffer, at: cand) {
            case .complete(let call, let range):
                appendText(String(buffer[..<cand]), to: &events)
                events.append(.toolCall(name: call.name, argumentsJSON: call.argumentsJSON))
                buffer = String(buffer[range.upperBound...])
                return true
            case .incomplete where !final:
                appendText(String(buffer[..<cand]), to: &events)
                buffer = String(buffer[cand...])
                return false
            case .incomplete, .malformed:
                // Truncated or unparseable: show it as text, but keep scanning after it.
                let after = buffer.index(cand, offsetBy: 5)
                appendText(String(buffer[..<after]), to: &events)
                buffer = String(buffer[after...])
                return true
            case .notACall:
                // Ordinary prose containing "call:"; emit through it and keep scanning.
                let after = buffer.index(cand, offsetBy: 5)
                appendText(String(buffer[..<after]), to: &events)
                buffer = String(buffer[after...])
                return true
            }
        }

        if let (range, text) = marker {
            appendText(String(buffer[..<range.lowerBound]), to: &events)
            buffer = String(buffer[range.upperBound...])
            switch text {
            case Self.thoughtStart: phase = .thought
            case Self.toolStart: phase = .toolBlock
            default: break   // stray close markers are dropped
            }
            return true
        }

        // No marker or call in view: emit everything that cannot be the start of one.
        let keep = final ? 0 : Self.heldSuffixLength(buffer, markers: Self.responseMarkers + ["call:"])
        let cut = buffer.index(buffer.endIndex, offsetBy: -keep)
        appendText(String(buffer[..<cut]), to: &events)
        buffer = String(buffer[cut...])
        return false
    }

    private mutating func drainThought(_ events: inout [Event], final: Bool) -> Bool {
        if let r = buffer.range(of: Self.channelEnd) {
            let thought = String(buffer[..<r.lowerBound])
            if !thought.isEmpty { events.append(.reasoning(thought)) }
            buffer = String(buffer[r.upperBound...])
            phase = .response
            return true
        }
        let keep = final ? 0 : Self.heldSuffixLength(buffer, markers: [Self.channelEnd])
        let cut = buffer.index(buffer.endIndex, offsetBy: -keep)
        let safe = String(buffer[..<cut])
        if !safe.isEmpty { events.append(.reasoning(safe)) }
        buffer = String(buffer[cut...])
        return false
    }

    private mutating func drainToolBlock(_ events: inout [Event], final: Bool) -> Bool {
        if let r = buffer.range(of: Self.toolEnd) {
            let body = String(buffer[..<r.lowerBound])
            buffer = String(buffer[r.upperBound...])
            phase = .response
            emitToolBlock(body, to: &events)
            return true
        }
        if final {
            let body = buffer
            buffer = ""
            phase = .response
            emitToolBlock(body, to: &events)
        }
        return false
    }

    private func emitToolBlock(_ body: String, to events: inout [Event]) {
        if let calls = GemmaCallSyntax.parseCalls(in: body) {
            for c in calls { events.append(.toolCall(name: c.name, argumentsJSON: c.argumentsJSON)) }
        } else if !body.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty {
            // Unparseable tool block: surface the raw text rather than silently swallowing it.
            events.append(.text(body))
        }
    }

    private func appendText(_ text: String, to events: inout [Event]) {
        guard !text.isEmpty else { return }
        events.append(.text(text))
    }

    /// Length (in Characters) of the longest suffix of `text` that is a proper prefix of a marker.
    private static func heldSuffixLength(_ text: String, markers: [String]) -> Int {
        var best = 0
        for m in markers {
            let maxLen = min(m.count - 1, text.count)
            guard maxLen > best else { continue }
            for len in stride(from: maxLen, to: best, by: -1) {
                if m.hasPrefix(String(text.suffix(len))) { best = len; break }
            }
        }
        return best
    }
}

// MARK: - Inline call: parser (used by FunctionGemma and legacy callers)

enum GemmaInlineCallParser {

    struct ParsedCall: Sendable {
        let name: String
        let argumentsJSON: String
    }

    /// Parses a call that starts exactly at the beginning of `text`.
    static func extractFirst(from text: String, allowPartial: Bool) -> (ParsedCall, String)? {
        guard text.hasPrefix("call:") else { return nil }
        guard case .complete(let call, let range) = GemmaCallSyntax.scan(text, at: text.startIndex) else { return nil }
        return (ParsedCall(name: call.name, argumentsJSON: call.argumentsJSON), String(text[range.upperBound...]))
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
}
