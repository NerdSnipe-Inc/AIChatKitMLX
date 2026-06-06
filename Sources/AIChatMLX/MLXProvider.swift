import Foundation
import AIChatCore
import MLXLLM
import MLXLMCommon
import MLXHuggingFace
import HuggingFace
import Tokenizers

// MARK: - MLXProvider

/// ChatProvider that runs inference on-device via Apple MLX.
///
/// Models are downloaded from Hugging Face Hub on first use and cached in the system's
/// standard caches directory. Subsequent runs load from cache with no network round-trip.
///
/// Requires Apple Silicon (M-series Mac or A-series iPhone/iPad). MLX will not run
/// meaningfully on Intel or Simulator targets.
///
/// Tool use uses Gemma 4's native format — tools are declared via the chat template's
/// `tools=` parameter and the model outputs `<|tool_call>call:name{...}<tool_call|>` tokens.
/// Pass tools via `ChatRequestOptions.nativeToolSpecs`.
public actor MLXProvider: ChatProvider {

    // MARK: - Default model

    public static let defaultModelId = "mlx-community/gemma-4-e4b-it-4bit"

    // MARK: - ChatProvider identity

    public nonisolated let id   = "mlx"
    public nonisolated let name = "MLX"
    public nonisolated var zeroResponseMessage: String {
        "No response from on-device model — try resending, or reload the model if it seems stuck"
    }

    // MARK: - Configuration

    nonisolated private let configuration:      ModelConfiguration
    nonisolated private let generateParameters: GenerateParameters
    private var container:                      ModelContainer?
    private var loadedAdapter:                  LoRAContainer?

    // MARK: - Init

    public init(
        modelId: String = MLXProvider.defaultModelId,
        maxTokens: Int? = nil,
        temperature: Float = 0.6,
        topP: Float = 1.0,
        repetitionPenalty: Float? = nil
    ) {
        self.configuration = ModelConfiguration(id: modelId)
        self.generateParameters = GenerateParameters(
            maxTokens: maxTokens,
            temperature: temperature,
            topP: topP,
            repetitionPenalty: repetitionPenalty
        )
    }

    public init(
        modelPath: URL,
        maxTokens: Int? = nil,
        temperature: Float = 0.6,
        topP: Float = 1.0,
        repetitionPenalty: Float? = nil
    ) {
        self.configuration = ModelConfiguration(directory: modelPath)
        self.generateParameters = GenerateParameters(
            maxTokens: maxTokens,
            temperature: temperature,
            topP: topP,
            repetitionPenalty: repetitionPenalty
        )
    }

    // MARK: - Model management

    public func loadModel(
        progressHandler: (@Sendable (Progress) -> Void)? = nil
    ) async throws {
        guard container == nil else { return }
        let handler = progressHandler ?? { _ in }
        container = try await LLMModelFactory.shared.loadContainer(
            from: #hubDownloader(),
            using: #huggingFaceTokenizerLoader(),
            configuration: configuration,
            progressHandler: handler
        )
    }

    // MARK: - ChatProvider

    public nonisolated func stream(
        messages: [ChatMessage],
        model: String,
        options: ChatRequestOptions
    ) -> AsyncThrowingStream<ChatStreamEvent, Error> {
        AsyncThrowingStream { continuation in
            Task {
                do {
                    try await self.ensureLoaded()

                    guard let container = await self.container else {
                        throw ChatError.invalidConfiguration("MLX model container failed to initialise.")
                    }

                    let params      = self.generateParameters
                    let toolSpecs   = Self.toToolSpecs(options.tools)
                    let mlxMessages = Self.toMLXMessages(messages: messages, systemPrompt: options.systemPrompt)

                    let completionInfo: GenerateCompletionInfo? = try await container.perform { @Sendable ctx in
                        let userInput = UserInput(messages: mlxMessages, tools: toolSpecs)
                        let lmInput   = try await ctx.processor.prepare(input: userInput)
                        let stream    = try MLXLMCommon.generate(
                            input: lmInput,
                            cache: nil,
                            parameters: params,
                            context: ctx
                        )

                        var processor = Gemma4StreamProcessor(tools: toolSpecs)
                        var info: GenerateCompletionInfo?

                        for await generation in stream {
                            guard !Task.isCancelled else { break }

                            if let nativeCall = generation.toolCall {
                                let argsJSON = Self.serializeArguments(nativeCall.function.arguments)
                                continuation.yield(.toolCallComplete(
                                    id: UUID().uuidString,
                                    name: nativeCall.function.name,
                                    arguments: argsJSON
                                ))
                            }

                            if let chunk = generation.chunk, !chunk.isEmpty {
                                for event in processor.processChunk(chunk) {
                                    switch event {
                                    case .reasoning(let delta):
                                        if !delta.isEmpty { continuation.yield(.reasoning(delta)) }
                                    case .text(let text):
                                        if !text.isEmpty { continuation.yield(.text(text)) }
                                    case .toolCall(let name, let argsJSON):
                                        continuation.yield(.toolCallComplete(
                                            id: UUID().uuidString,
                                            name: name,
                                            arguments: argsJSON
                                        ))
                                    }
                                }
                            }

                            if let genInfo = generation.info { info = genInfo }
                        }

                        for event in processor.finish() {
                            switch event {
                            case .reasoning(let delta):
                                if !delta.isEmpty { continuation.yield(.reasoning(delta)) }
                            case .text(let text):
                                if !text.isEmpty { continuation.yield(.text(text)) }
                            case .toolCall(let name, let argsJSON):
                                continuation.yield(.toolCallComplete(
                                    id: UUID().uuidString,
                                    name: name,
                                    arguments: argsJSON
                                ))
                            }
                        }

                        return info
                    }

                    if let info = completionInfo {
                        continuation.yield(.usage(TokenUsage(
                            promptTokens:     info.promptTokenCount,
                            completionTokens: info.generationTokenCount,
                            totalTokens:      info.promptTokenCount + info.generationTokenCount
                        )))
                    }

                    continuation.yield(.done)
                    continuation.finish()
                } catch is CancellationError {
                    continuation.finish(throwing: ChatError.cancelled)
                } catch {
                    continuation.finish(throwing: error)
                }
            }
        }
    }

    public nonisolated func complete(
        messages: [ChatMessage],
        model: String,
        options: ChatRequestOptions
    ) async throws -> ChatCompletionResult {
        var fullText = ""
        var usage: TokenUsage?
        for try await event in stream(messages: messages, model: model, options: options) {
            switch event {
            case .text(let delta): fullText += delta
            case .usage(let u):    usage = u
            default:               break
            }
        }
        return ChatCompletionResult(
            id: nil,
            model: id,
            message: ChatMessage(role: .assistant, content: fullText),
            usage: usage,
            finishReason: .stop
        )
    }

    // MARK: - Persona adapter management

    public func loadAdapter(at path: URL) async throws {
        guard let container else {
            throw ChatError.invalidConfiguration("Model must be loaded before loading an adapter.")
        }
        if loadedAdapter != nil { try await unloadAdapter() }
        let adapter = try LoRAContainer.from(directory: path)
        _ = try await container.perform { ctx in
            try adapter.load(into: ctx.model)
        }
        loadedAdapter = adapter
        print("[MLXProvider] Adapter loaded from \(path.lastPathComponent)")
    }

    public func unloadAdapter() async throws {
        guard let adapter = loadedAdapter, let container else { return }
        _ = await container.perform { ctx in
            adapter.unload(from: ctx.model)
        }
        loadedAdapter = nil
        print("[MLXProvider] Adapter unloaded")
    }

    // MARK: - Private helpers

    private func ensureLoaded() async throws {
        guard container == nil else { return }
        try await loadModel()
    }

    /// Serialise `[String: JSONValue]` tool call arguments back to a JSON string.
    private static func serializeArguments(_ args: [String: JSONValue]) -> String {
        let plain = args.mapValues { $0.anyValue }
        guard let data = try? JSONSerialization.data(withJSONObject: plain),
              let str  = String(data: data, encoding: .utf8)
        else { return "{}" }
        return str
    }

    /// Convert `[ChatMessage]` to the `[[String: any Sendable]]` format MLX expects.
    ///
    /// - Tool-result messages (`role == .tool`) are passed with `tool_call_id` so the
    ///   chat template's forward-scan picks them up from the preceding assistant message.
    /// - Assistant messages with `toolCalls` use a `tool_calls` array (not plain text),
    ///   so the template renders `<|tool_call>...<tool_call|>` and inlines the response.
    private static func toMLXMessages(
        messages: [ChatMessage],
        systemPrompt: String?
    ) -> [[String: any Sendable]] {
        var result: [[String: any Sendable]] = []

        if let sys = systemPrompt {
            result.append(["role": "system" as any Sendable, "content": sys as any Sendable])
        }

        for message in messages {

            // --- Tool-result messages ---
            if message.role == .tool {
                let text = message.content.compactMap {
                    if case .text(let t) = $0 { return t } else { return nil }
                }.joined(separator: "\n")

                var msg: [String: any Sendable] = [
                    "role":    "tool"   as any Sendable,
                    "content": text     as any Sendable
                ]
                if let tid = message.toolCallId {
                    msg["tool_call_id"] = tid as any Sendable
                }
                result.append(msg)
                continue
            }

            // --- Assistant messages with tool calls (native format) ---
            if let toolCalls = message.toolCalls, !toolCalls.isEmpty {
                let nativeCalls: [any Sendable] = toolCalls.compactMap { tc -> [String: any Sendable]? in
                    guard let argData = tc.arguments.data(using: .utf8),
                          let argObj  = try? JSONSerialization.jsonObject(with: argData) as? [String: Any]
                    else { return nil }
                    return [
                        "id":   tc.id   as any Sendable,
                        "type": "function" as any Sendable,
                        "function": [
                            "name":      tc.name as any Sendable,
                            "arguments": Self.toSendable(argObj) as any Sendable
                        ] as [String: any Sendable] as any Sendable
                    ]
                }
                guard !nativeCalls.isEmpty else { continue }
                result.append([
                    "role":       "assistant" as any Sendable,
                    "content":    ""          as any Sendable,
                    "tool_calls": nativeCalls as any Sendable
                ])
                continue
            }

            // --- All other messages (plain text) ---
            let parts = message.content.compactMap {
                if case .text(let t) = $0 { return t } else { return nil }
            }
            guard !parts.isEmpty else { continue }
            result.append([
                "role":    message.role.rawValue as any Sendable,
                "content": parts.joined(separator: "\n") as any Sendable
            ])
        }

        return result
    }

    /// Convert `ChatRequestOptions.ToolDefinition` array to the `[ToolSpec]` format expected by MLXLMCommon.
    private static func toToolSpecs(_ tools: [ChatRequestOptions.ToolDefinition]?) -> [ToolSpec]? {
        guard let tools, !tools.isEmpty else { return nil }
        let encoder = JSONEncoder()
        return tools.compactMap { tool -> ToolSpec? in
            var function: [String: any Sendable] = ["name": tool.name]
            if let desc = tool.description { function["description"] = desc }
            if let params = tool.parameters,
               let data = try? encoder.encode(params),
               let dict = try? JSONSerialization.jsonObject(with: data) as? [String: Any] {
                let safe = GemmaJinjaToolSchema.sanitizeParameters(dict)
                function["parameters"] = toSendable(safe)
            }
            return ["type": "function", "function": function]
        }
    }

    /// Recursively convert `[String: Any]` (from JSONSerialization) to `[String: any Sendable]`.
    private static func toSendable(_ dict: [String: Any]) -> [String: any Sendable] {
        dict.compactMapValues { value -> (any Sendable)? in
            if let d = value as? [String: Any] { return toSendable(d) }
            if let a = value as? [Any]         { return a.compactMap { v -> (any Sendable)? in
                if let d = v as? [String: Any] { return toSendable(d) }
                if let s = v as? String { return s }
                if let b = v as? Bool { return b }
                if let n = v as? Double { return n }
                if let n = v as? Int { return n }
                return nil
            }}
            if let s = value as? String        { return s }
            if let b = value as? Bool          { return b }
            if let n = value as? Double        { return n }
            if let n = value as? Int           { return n }
            return nil
        }
    }
}
