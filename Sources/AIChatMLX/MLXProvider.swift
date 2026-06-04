import Foundation
import AIChatCore
import MLXLLM
import MLXLMCommon
import MLXHuggingFace
import HuggingFace
import Tokenizers

/// ChatProvider that runs inference on-device via Apple MLX.
///
/// Models are downloaded from Hugging Face Hub on first use and cached in the system's
/// standard caches directory. Subsequent runs load from cache with no network round-trip.
///
/// Requires Apple Silicon (M-series Mac or A-series iPhone/iPad). MLX will not run
/// meaningfully on Intel or Simulator targets.
///
/// ### Quick start
/// ```swift
/// let provider = MLXProvider()   // uses mlx-community/gemma-4-e4b-it-4bit
/// let session  = ChatSession(provider: provider, model: "")
/// session.send("Hello")
/// ```
///
/// ### Custom model
/// ```swift
/// let provider = MLXProvider(modelId: "mlx-community/Qwen3-1.7B-4bit")
/// ```
///
/// ### Pre-downloaded model
/// ```swift
/// let provider = MLXProvider(modelPath: URL(fileURLWithPath: "/path/to/model-dir"))
/// ```
///
/// The `model` parameter on `ChatSession` is not used by `MLXProvider`; the model is
/// determined at init time. Pass any non-empty string (e.g. `""`).
public actor MLXProvider: ChatProvider {

    // MARK: - Default model

    /// Default model: Gemma 4 E4B instruction-tuned, 4-bit quantized.
    ///
    /// This is the smallest Gemma 4 variant with vision support stripped (text-only via
    /// mlx-community). It runs comfortably on MacBook Air M-series and A-series iPhones.
    /// The model is approximately 2.5 GB on disk after download.
    ///
    /// Available quantisation variants on mlx-community if you need a different size/quality:
    /// - `mlx-community/gemma-4-e4b-it-4bit`  (default, ~2.5 GB)
    /// - `mlx-community/gemma-4-e4b-it-8bit`  (~4.5 GB)
    /// - `mlx-community/gemma-4-e4b-it-bf16`  (~8 GB, full precision)
    public static let defaultModelId = "mlx-community/gemma-4-e4b-it-4bit"

    // MARK: - ChatProvider identity

    public nonisolated let id   = "mlx"
    public nonisolated let name = "MLX"

    // MARK: - Configuration

    // nonisolated: immutable constants, safe to read without an actor hop
    nonisolated private let configuration:      ModelConfiguration
    nonisolated private let generateParameters: GenerateParameters
    private var container:                      ModelContainer?

    // MARK: - Init

    /// Create an MLXProvider with a Hugging Face Hub model ID.
    ///
    /// The model is downloaded to the system caches directory on first use.
    ///
    /// - Parameters:
    ///   - modelId: Hugging Face repo ID. Defaults to `mlx-community/gemma-4-e4b-it-4bit`.
    ///   - maxTokens: Maximum tokens to generate. `nil` = unlimited.
    ///   - temperature: Sampling temperature (default 0.6).
    ///   - topP: Top-p nucleus sampling threshold (default 1.0).
    ///   - repetitionPenalty: Penalty for repeating recent tokens. `nil` = disabled.
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

    /// Create an MLXProvider from a pre-downloaded local model directory.
    ///
    /// The directory must contain `config.json`, model weight shards, and tokenizer files
    /// (the standard layout produced by `mlx_lm.convert` or downloaded from mlx-community).
    ///
    /// - Parameter modelPath: URL of the directory containing the model files.
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

    /// Load (and if needed, download) the model.
    ///
    /// Called automatically on the first `stream()` or `complete()` call.
    /// Call this explicitly with a progress handler to show download UI before starting a chat.
    ///
    /// - Parameter progressHandler: Receives a `Progress` object during Hub download.
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
                    let mlxMessages: [[String: any Sendable]] = Self.toMLXMessages(
                        messages: messages, systemPrompt: options.systemPrompt)

                    // ModelContainer.perform provides thread-safe access to the underlying
                    // ModelContext. All heavy work — prompt preparation and the generation
                    // loop — happens inside this closure.
                    let completionInfo: GenerateCompletionInfo? = try await container.perform { @Sendable ctx in
                        let userInput = UserInput(messages: mlxMessages)
                        let lmInput   = try await ctx.processor.prepare(input: userInput)
                        // cache: nil — let MLX manage the KV cache internally.
                        let stream    = try MLXLMCommon.generate(
                            input: lmInput,
                            cache: nil,
                            parameters: params,
                            context: ctx
                        )

                        var info: GenerateCompletionInfo? = nil
                        for await generation in stream {
                            guard !Task.isCancelled else { break }
                            if let chunk = generation.chunk, !chunk.isEmpty {
                                continuation.yield(.text(chunk))
                            }
                            if let genInfo = generation.info {
                                info = genInfo
                            }
                        }
                        return info
                    }

                    if let info = completionInfo {
                        continuation.yield(.usage(TokenUsage(
                            promptTokens:      info.promptTokenCount,
                            completionTokens:  info.generationTokenCount,
                            totalTokens:       info.promptTokenCount + info.generationTokenCount
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
        var usage: TokenUsage? = nil
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

    // MARK: - Helpers

    private func ensureLoaded() async throws {
        guard container == nil else { return }
        try await loadModel()
    }

    /// Converts `[ChatMessage]` to the `[UserInput.Message]` format MLX expects.
    ///
    /// `UserInput.Message` is `[String: any Sendable]`. The chat template is applied by
    /// `context.processor.prepare(input:)` using the model's built-in tokenizer template,
    /// so no manual template formatting is needed here.
    // MLXLMCommon.Message = [String: any Sendable] (module-level typealias, not UserInput.Message)
    private static func toMLXMessages(
        messages: [ChatMessage],
        systemPrompt: String?
    ) -> [[String: any Sendable]] {
        var result: [[String: any Sendable]] = []

        if let sys = systemPrompt {
            result.append(["role": "system" as any Sendable, "content": sys as any Sendable])
        }

        for message in messages {
            let parts = message.content.compactMap { block -> String? in
                guard case .text(let t) = block else { return nil }
                return t
            }
            guard !parts.isEmpty else { continue }
            let content = parts.joined(separator: "\n")
            result.append(["role": message.role.rawValue as any Sendable, "content": content as any Sendable])
        }

        return result
    }
}
