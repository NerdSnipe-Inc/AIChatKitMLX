import Foundation
import MLXLMCommon

/// Normalizes model-family-specific generated text into the event shape consumed by
/// `MLXProvider`. FunctionGemma intentionally buffers its very small routing response until EOS:
/// this prevents partial `<start_function_call>` markers from leaking into user-visible text.
enum ModelStreamProcessor {
    case gemma4(Gemma4StreamProcessor)
    case functionGemma(FunctionGemmaStreamProcessor)

    init(syntax: MLXProvider.StreamSyntax, tools: [[String: any Sendable]]?) {
        switch syntax {
        case .gemma4:
            self = .gemma4(Gemma4StreamProcessor(tools: tools))
        case .functionGemma:
            self = .functionGemma(FunctionGemmaStreamProcessor())
        }
    }

    mutating func processChunk(_ chunk: String) -> [Gemma4StreamProcessor.Event] {
        switch self {
        case .gemma4(var processor):
            let events = processor.processChunk(chunk)
            self = .gemma4(processor)
            return events
        case .functionGemma(var processor):
            let events = processor.processChunk(chunk)
            self = .functionGemma(processor)
            return events
        }
    }

    mutating func finish() -> [Gemma4StreamProcessor.Event] {
        switch self {
        case .gemma4(var processor):
            let events = processor.finish()
            self = .gemma4(processor)
            return events
        case .functionGemma(var processor):
            let events = processor.finish()
            self = .functionGemma(processor)
            return events
        }
    }
}

/// Parses FunctionGemma's native textual function-call envelope:
/// `<start_function_call>call:name{...}<end_function_call>`.
struct FunctionGemmaStreamProcessor {
    private static let startMarker = "<start_function_call>"
    private static let endMarker = "<end_function_call>"

    private var buffer = ""

    mutating func processChunk(_ chunk: String) -> [Gemma4StreamProcessor.Event] {
        buffer += chunk
        return []
    }

    mutating func finish() -> [Gemma4StreamProcessor.Event] {
        var remainder = buffer
        buffer = ""
        var events: [Gemma4StreamProcessor.Event] = []

        while let start = remainder.range(of: Self.startMarker) {
            let leading = String(remainder[..<start.lowerBound])
                .trimmingCharacters(in: .whitespacesAndNewlines)
            if !leading.isEmpty {
                events.append(.text(leading))
            }

            let afterStart = remainder[start.upperBound...]
            guard let end = afterStart.range(of: Self.endMarker) else {
                let unresolved = String(remainder[start.lowerBound...])
                    .trimmingCharacters(in: .whitespacesAndNewlines)
                if !unresolved.isEmpty {
                    events.append(.text(unresolved))
                }
                remainder = ""
                break
            }

            let payload = String(afterStart[..<end.lowerBound])
                .trimmingCharacters(in: .whitespacesAndNewlines)
            if let (call, trailing) = GemmaInlineCallParser.extractFirst(
                from: payload,
                allowPartial: false
            ), trailing.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty {
                events.append(.toolCall(
                    name: call.name,
                    argumentsJSON: call.argumentsJSON
                ))
            } else if !payload.isEmpty {
                events.append(.text(payload))
            }

            remainder = String(afterStart[end.upperBound...])
        }

        let trailing = remainder.trimmingCharacters(in: .whitespacesAndNewlines)
        if !trailing.isEmpty {
            if let (call, rest) = GemmaInlineCallParser.extractFirst(
                from: trailing,
                allowPartial: false
            ), rest.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty {
                events.append(.toolCall(
                    name: call.name,
                    argumentsJSON: call.argumentsJSON
                ))
            } else {
                events.append(.text(trailing))
            }
        }
        return events
    }
}
