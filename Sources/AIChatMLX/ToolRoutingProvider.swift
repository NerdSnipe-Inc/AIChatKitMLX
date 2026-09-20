import Foundation
import AIChatCore

// MARK: - RoutingDecision

/// What ``ToolRoutingProvider`` did with one `stream`/`complete` call, and why.
///
/// Delivered through ``ToolRoutingProvider/onDecision`` before the responder (if any) starts, and
/// logged at `ChatLog` debug level (category `tools`).
public struct RoutingDecision: Sendable, Equatable {
    /// The coarse result.
    public enum Outcome: Sendable, Equatable {
        /// The router chose tool call(s); they were emitted and the responder was not invoked.
        case routedToTool
        /// The router was not needed or declined: the responder answered (see ``responderHasTools``).
        case passthrough
        /// The router was consulted but its output was unusable (failure, timeout, unknown tool,
        /// invalid arguments, repeated call); the responder answered per the fallback policy.
        case fallback
    }

    /// The precise cause.
    public enum Reason: Sendable, Equatable {
        /// The request declared no tools.
        case noTools
        /// `toolChoice == .none`.
        case toolsDisabled
        /// ``ToolRoutingProvider/Configuration/isRouterEnabled`` is `false`.
        case routerDisabled
        /// The last message was neither a user turn nor a tool result.
        case notUserTurn
        /// The last message was a tool result and the policy sent it to the responder.
        case afterToolResult
        /// The router produced text or nothing: it judged that no tool is needed.
        case routerDeclined
        /// The router chose valid tool call(s).
        case routedCall
        /// The router named a tool that is not declared.
        case unknownTool(String)
        /// The router's arguments failed validation.
        case invalidArguments(String)
        /// The router repeated a call already made this turn, or the per-turn call cap was hit.
        case repeatedCall
        /// The router threw (load error, template error, ...).
        case routerFailed(String)
        /// The router exceeded ``ToolRoutingProvider/Configuration/routerTimeout``.
        case routerTimedOut
    }

    public let outcome: Outcome
    public let reason: Reason
    /// Names of the tools that were emitted (`.routedToTool`) — empty otherwise.
    public let toolNames: [String]
    /// Wall time spent in the router stage; `nil` when the router was not run.
    public let routerLatency: Duration?
    /// Prompt tokens the router reported, when its provider emits usage.
    public let routerPromptTokens: Int?
    /// Tokens the router generated, when its provider emits usage. With ``routerLatency`` this
    /// shows whether a slow router is slow per token or generating too much.
    public let routerCompletionTokens: Int?
    /// Raw router text when it declined (useful to see what a router said instead of a call).
    public let routerText: String?
    /// Whether the responder was (or will be) given the tool declarations. `false` for
    /// `.routedToTool` (no responder call).
    public let responderHasTools: Bool
}

// MARK: - ToolRoutingProvider

/// A two-stage provider: a tiny **router** (normally FunctionGemma-270M) decides whether the
/// user's turn needs a tool and with what arguments; a **responder** (normally Gemma 4) writes the
/// answer.
///
/// ```
///   user turn ──▶ router (minimal prompt + tool declarations)
///                    ├─ valid call(s) ─▶ .toolCallComplete … .done          (responder skipped)
///                    └─ text / nothing / invalid / timeout / error
///                                     ─▶ responder, tools REMOVED           (fast path)
///                                        (or WITH tools per ``FallbackPolicy``)
///   tool result ──▶ ``AfterToolResultPolicy`` (default: responder, no tools, composes the answer)
/// ```
///
/// Chat turns therefore never pay for tool-schema tokens or the big model's tool decision, and
/// tool turns skip the big model until the result has to be phrased.
///
/// ## Failure behaviour
/// A router that is missing, fails to load, throws, times out or emits garbage never fails the
/// turn — but it never degrades silently either: every such case is logged with
/// `ChatLog.warning` (category `tools`) and reported as a `.fallback` ``RoutingDecision``. Only a
/// responder failure surfaces to the caller (as a specific `ChatError`).
///
/// ## Known limitation: missed mandatory tools
/// A small router can decide that no tool is needed when one is required (e.g. "draft a reply to
/// conversation 8f3a…"); the tool-less responder then answers from nothing, plausibly and wrongly.
/// If a request must use a tool, set ``Configuration/requiresTool`` (return `true` when the user
/// text carries an identifier/entity that only a tool can resolve) so a non-routed turn goes to the
/// responder *with* tools, or disable routing for that call site with
/// ``Configuration/isRouterEnabled``.
///
/// `toolChoice == .none` is honoured; other `toolChoice` values are not interpreted.
///
/// See `docs/TOOL_ROUTING.md` for measured accuracy, latency and memory.
public struct ToolRoutingProvider: ChatProvider {

    /// What the responder gets when the router did not (validly) route a user turn.
    public enum FallbackPolicy: Sendable, Equatable {
        /// Tools removed from the prompt: fastest, plain chat never pays schema cost. Default.
        case respondWithoutTools
        /// The responder receives the tools (native tool calling as a safety net).
        case respondWithTools
    }

    /// What happens when the last message is a tool result (``ChatSession`` re-invokes the
    /// provider after executing a tool).
    public enum AfterToolResultPolicy: Sendable, Equatable {
        /// The responder composes the answer from the result, without tools. Default.
        case respond
        /// The router sees the request plus results and may chain another call; if it declines,
        /// the responder composes the answer.
        case routeAgain
        /// The responder gets the tools and may call further tools itself.
        case respondWithTools
    }

    /// How tool declarations reach the router model.
    public enum RouterToolFormat: Sendable, Equatable {
        /// Pass them as `tools` and let the router's chat template render them. Right for any
        /// router whose template is known to render correctly under `swift-jinja`.
        case chatTemplate
        /// Render FunctionGemma's canonical declaration syntax ourselves and send it as part of the
        /// system (developer) text with no `tools`. `swift-jinja` renders FunctionGemma's template
        /// with stray spaces the reference engine does not produce; this is byte-identical to the
        /// reference. Measured slightly more accurate (17/24 vs 15/24) — within noise, so treat it
        /// as a safe default for FunctionGemma rather than a proven win.
        case functionGemmaInline
    }

    /// Tunables. All have safe defaults.
    public struct Configuration: Sendable {
        /// Set `false` to bypass the router entirely (responder gets tools as sent).
        public var isRouterEnabled = true
        /// Number of most recent user turns (plus assistant replies between them) shown to the
        /// router. Small routers do best with `1`.
        public var routerContextTurns = 1
        /// Per-turn character cap for the router prompt (head + tail kept).
        public var routerCharacterLimit = 1_500
        /// How the router receives the tool declarations.
        public var routerToolFormat: RouterToolFormat = .chatTemplate
        /// System (developer) text for the router. `nil` sends none.
        public var routerInstruction: String? =
            "You are a model that can do function calling with the following functions"
        /// Token cap requested from the router via `ChatRequestOptions.maxTokens`. Providers that
        /// ignore per-request caps (`MLXProvider`) must be built with their own cap —
        /// ``ToolRoutingProvider/onDevice(responderModelId:routerModelId:routerMaxTokens:configuration:onDecision:)``
        /// does that.
        public var maxRouterTokens = 128
        /// Router wall-time budget, including a cold model load. Warm the router with
        /// ``ToolRoutingProvider/warmUp()`` to keep loads out of the budget.
        public var routerTimeout: Duration = .seconds(5)
        public var fallback: FallbackPolicy = .respondWithoutTools
        public var afterToolResult: AfterToolResultPolicy = .respond
        /// Upper bound on routed calls in one user turn under ``AfterToolResultPolicy/routeAgain``.
        public var maxRoutedCallsPerTurn = 3
        /// Escape hatch for missed mandatory tools: given the last user text, return `true` when a
        /// tool is required. If the router does not route such a turn, the responder receives the
        /// tools regardless of ``fallback``.
        public var requiresTool: (@Sendable (String) -> Bool)?

        public init() {}
    }

    // MARK: - ChatProvider identity

    public var id: String { "tool-routing" }
    public var name: String { "Tool routing (\(router.name) → \(responder.name))" }
    public var zeroResponseMessage: String { responder.zeroResponseMessage }

    // MARK: - Configuration

    public let router: any ChatProvider
    public let responder: any ChatProvider
    public var configuration: Configuration
    /// Model string forwarded to the router (defaults to the request's `model`).
    public let routerModel: String?
    /// Called with every decision, before the responder starts. Must be cheap and non-blocking.
    public let onDecision: (@Sendable (RoutingDecision) -> Void)?

    /// - Parameters:
    ///   - router: Small tool-selecting provider.
    ///   - responder: Provider that writes answers.
    ///   - routerModel: Model string for the router; `nil` reuses the request's `model`.
    ///   - configuration: Policies and budgets.
    ///   - onDecision: Diagnostics callback.
    public init(
        router: any ChatProvider,
        responder: any ChatProvider,
        routerModel: String? = nil,
        configuration: Configuration = Configuration(),
        onDecision: (@Sendable (RoutingDecision) -> Void)? = nil
    ) {
        self.router = router
        self.responder = responder
        self.routerModel = routerModel
        self.configuration = configuration
        self.onDecision = onDecision
    }

    /// Copy with `configuration` modified — e.g. a per-call-site opt-out:
    /// `provider.with { $0.isRouterEnabled = false }`.
    public func with(_ change: (inout Configuration) -> Void) -> ToolRoutingProvider {
        var copy = self
        change(&copy.configuration)
        return copy
    }

    // MARK: - On-device convenience

    /// FunctionGemma-270M in bf16 (~0.5 GB). The 4-bit conversion (`…-it-4bit`) was measured to be
    /// unusable as a router — it emits prose and malformed calls — so it is not the default.
    public static let functionGemmaModelId = "mlx-community/functiongemma-270m-it-bf16"

    /// Builds the usual pairing: a FunctionGemma router in MLX's `.auxiliary` residency slot and
    /// an `MLXProvider` responder in `.primary`, so both stay resident and neither evicts the other.
    ///
    /// - Parameters:
    ///   - responderModelId: Conversational model (default `MLXProvider.smallModelId`).
    ///   - routerModelId: Router model; pass a fine-tuned router's id here.
    ///   - routerMaxTokens: Hard generation cap for the router (the router's own `MLXProvider`
    ///     enforces it, since `MLXProvider` ignores per-request token caps).
    public static func onDevice(
        responderModelId: String = MLXProvider.smallModelId,
        routerModelId: String = functionGemmaModelId,
        routerMaxTokens: Int = 128,
        configuration: Configuration = Configuration(),
        onDecision: (@Sendable (RoutingDecision) -> Void)? = nil
    ) -> ToolRoutingProvider {
        var configuration = configuration
        configuration.maxRouterTokens = routerMaxTokens
        return ToolRoutingProvider(
            router: MLXProvider(
                modelId: routerModelId, residency: .auxiliary,
                maxTokens: routerMaxTokens, temperature: 0, topP: 1
            ),
            responder: MLXProvider(modelId: responderModelId, residency: .primary),
            routerModel: routerModelId,
            configuration: configuration,
            onDecision: onDecision
        )
    }

    /// Loads the router (a one-token dummy request) so the first real turn does not spend its
    /// time budget on a cold model load. Failures are logged, never thrown.
    public func warmUp() async {
        var options = ChatRequestOptions()
        options.maxTokens = 1
        do {
            _ = try await router.complete(
                messages: [ChatMessage(role: .user, content: "hi")],
                model: routerModel ?? "", options: options
            )
        } catch {
            ChatLog.warning(.tools, "Router warm-up failed: \(error.localizedDescription)")
        }
    }

    // MARK: - ChatProvider

    public func stream(
        messages: [ChatMessage],
        model: String,
        options: ChatRequestOptions
    ) -> AsyncThrowingStream<ChatStreamEvent, Error> {
        AsyncThrowingStream { continuation in
            let task = Task {
                do {
                    try await self.run(messages: messages, model: model, options: options, into: continuation)
                    continuation.finish()
                } catch is CancellationError {
                    continuation.finish(throwing: ChatError.cancelled)
                } catch let error as ChatError {
                    continuation.finish(throwing: error)
                } catch {
                    continuation.finish(throwing: ChatError.classify(error, modelId: model, phase: .generate))
                }
            }
            // Cancelling the consumer cancels whichever stage is running; each stage's own
            // stream termination handler then stops its generation.
            continuation.onTermination = { _ in task.cancel() }
        }
    }

    public func complete(
        messages: [ChatMessage],
        model: String,
        options: ChatRequestOptions
    ) async throws -> ChatCompletionResult {
        var text = ""
        var usage: TokenUsage?
        var calls: [ChatMessage.ToolCallBlock] = []
        for try await event in stream(messages: messages, model: model, options: options) {
            switch event {
            case .text(let delta): text += delta
            case .usage(let u): usage = u
            case .toolCallComplete(let id, let name, let arguments):
                calls.append(.init(id: id, name: name, arguments: arguments))
            default: break
            }
        }
        return ChatCompletionResult(
            id: nil,
            model: id,
            message: ChatMessage(
                role: .assistant,
                content: text.isEmpty ? [] : [.text(text)],
                toolCalls: calls.isEmpty ? nil : calls
            ),
            usage: usage,
            finishReason: calls.isEmpty ? .stop : .toolCalls
        )
    }

    // MARK: - Orchestration

    private typealias Continuation = AsyncThrowingStream<ChatStreamEvent, Error>.Continuation

    private func run(
        messages: [ChatMessage], model: String, options: ChatRequestOptions, into continuation: Continuation
    ) async throws {
        let tools = RoutableTool.declared(in: options)
        guard !tools.isEmpty else {
            return try await respond(messages, model, options, into: continuation,
                                     decision: .init(.passthrough, .noTools, hasTools: false))
        }
        if case .some(.none) = options.toolChoice {
            return try await respond(messages, model, Self.removingTools(options), into: continuation,
                                     decision: .init(.passthrough, .toolsDisabled, hasTools: false))
        }
        guard configuration.isRouterEnabled else {
            return try await respond(messages, model, options, into: continuation,
                                     decision: .init(.passthrough, .routerDisabled, hasTools: true))
        }

        let routerMessages: [ChatMessage]
        var alreadyCalled: Set<String> = []
        switch messages.last?.role {
        case .user:
            routerMessages = RouterPrompt.recentTurns(
                messages, turns: configuration.routerContextTurns, characterLimit: configuration.routerCharacterLimit
            )
        case .tool:
            switch configuration.afterToolResult {
            case .respond:
                return try await respond(messages, model, Self.removingTools(options), into: continuation,
                                         decision: .init(.passthrough, .afterToolResult, hasTools: false))
            case .respondWithTools:
                return try await respond(messages, model, options, into: continuation,
                                         decision: .init(.passthrough, .afterToolResult, hasTools: true))
            case .routeAgain:
                let turn = Self.callsSinceLastUser(messages)
                alreadyCalled = Set(turn.map { Self.fingerprint(name: $0.name, arguments: $0.arguments) })
                if turn.count >= configuration.maxRoutedCallsPerTurn {
                    return try await respond(messages, model, Self.removingTools(options), into: continuation,
                                             decision: .init(.fallback, .repeatedCall, hasTools: false))
                }
                routerMessages = RouterPrompt.withToolResults(messages, characterLimit: configuration.routerCharacterLimit)
            }
        default:
            return try await respond(messages, model, Self.removingTools(options), into: continuation,
                                     decision: .init(.passthrough, .notUserTurn, hasTools: false))
        }

        let userText = Self.lastUserText(messages)
        let requiresTool = configuration.requiresTool?(userText) ?? false
        guard !routerMessages.isEmpty else {
            return try await fallBack(messages, model, options, into: continuation, requiresTool: requiresTool,
                                      decision: .init(.fallback, .routerFailed("no user text to route"), hasTools: false))
        }

        // Stage 1: router.
        let started = ContinuousClock.now
        let attempt: RouterResult
        do {
            attempt = try await runRouter(routerMessages, tools: tools, model: model, options: options)
        } catch is CancellationError {
            throw CancellationError()
        } catch let timeout as RouterTimeout {
            ChatLog.warning(.tools, "Tool router timed out after \(timeout.limit); falling back to the responder")
            return try await fallBack(messages, model, options, into: continuation, requiresTool: requiresTool,
                                      decision: .init(.fallback, .routerTimedOut, hasTools: false,
                                                      latency: started.duration(to: .now)))
        } catch {
            ChatLog.warning(.tools, "Tool router failed (\(error.localizedDescription)); falling back to the responder")
            ChatLog.debug(.tools, "router error chain: \(ChatLog.errorChain(error))")
            return try await fallBack(messages, model, options, into: continuation, requiresTool: requiresTool,
                                      decision: .init(.fallback, .routerFailed(error.localizedDescription), hasTools: false,
                                                      latency: started.duration(to: .now)))
        }
        // A cancelled stream ends quietly rather than throwing, so it would otherwise look like a
        // router that produced nothing and start the responder.
        try Task.checkCancellation()
        let latency = started.duration(to: .now)

        guard !attempt.calls.isEmpty else {
            let d = RoutingDecision(.passthrough, .routerDeclined, hasTools: false, latency: latency)
                .with(attempt)
            return try await fallBack(messages, model, options, into: continuation, requiresTool: requiresTool, decision: d)
        }

        // Validate every call; any bad call sends the whole turn to the fallback path.
        var validated: [(id: String, name: String, arguments: String)] = []
        for call in attempt.calls {
            switch RoutableTool.validate(name: call.name, argumentsJSON: call.arguments, against: tools) {
            case .success(let json):
                if alreadyCalled.contains(Self.fingerprint(name: call.name, arguments: json)) {
                    ChatLog.warning(.tools, "Tool router repeated \(call.name) with identical arguments; responding instead")
                    let d = RoutingDecision(.fallback, .repeatedCall, hasTools: false, latency: latency)
                        .with(attempt)
                    return try await respond(messages, model, Self.removingTools(options), into: continuation, decision: d)
                }
                validated.append((call.id, call.name, json))
            case .failure(let failure):
                let reason: RoutingDecision.Reason
                switch failure {
                case .unknownTool(let n): reason = .unknownTool(n)
                case .invalidArguments(let why): reason = .invalidArguments(why)
                }
                ChatLog.warning(.tools, "Tool router output rejected (\(reason)); falling back to the responder")
                let d = RoutingDecision(.fallback, reason, hasTools: false, latency: latency)
                    .with(attempt)
                return try await fallBack(messages, model, options, into: continuation, requiresTool: requiresTool, decision: d)
            }
        }

        try Task.checkCancellation()
        let decision = RoutingDecision(.routedToTool, .routedCall, hasTools: false, latency: latency, tools: validated.map(\.name))
            .with(attempt, includeText: false)
        publish(decision)
        for call in validated {
            continuation.yield(.toolCallComplete(id: call.id, name: call.name, arguments: call.arguments))
        }
        continuation.yield(.done)
    }

    /// Sends the turn to the responder per the fallback policy (or with tools when the turn
    /// `requiresTool`), recording `decision` with the actual `responderHasTools`.
    private func fallBack(
        _ messages: [ChatMessage], _ model: String, _ options: ChatRequestOptions, into continuation: Continuation,
        requiresTool: Bool, decision: RoutingDecision
    ) async throws {
        let withTools = requiresTool || configuration.fallback == .respondWithTools
        try await respond(
            messages, model, withTools ? options : Self.removingTools(options), into: continuation,
            decision: decision.with(hasTools: withTools)
        )
    }

    /// Publishes `decision`, then forwards the responder's stream verbatim.
    private func respond(
        _ messages: [ChatMessage], _ model: String, _ options: ChatRequestOptions, into continuation: Continuation,
        decision: RoutingDecision
    ) async throws {
        try Task.checkCancellation()
        publish(decision)
        for try await event in responder.stream(messages: messages, model: model, options: options) {
            try Task.checkCancellation()
            continuation.yield(event)
        }
        try Task.checkCancellation()
    }

    private func publish(_ decision: RoutingDecision) {
        let latency = decision.routerLatency.map { "\($0)" } ?? "-"
        ChatLog.debug(.tools, "routing: \(decision.outcome) reason=\(decision.reason) tools=\(decision.toolNames) routerLatency=\(latency) routerPromptTokens=\(decision.routerPromptTokens.map(String.init) ?? "-") responderHasTools=\(decision.responderHasTools)")
        onDecision?(decision)
    }

    // MARK: - Router stage

    private struct RouterTimeout: Error { let limit: Duration }

    struct RouterResult: Sendable {
        var calls: [(id: String, name: String, arguments: String)] = []
        var text = ""
        var promptTokens: Int?
        var completionTokens: Int?
    }

    private func runRouter(
        _ routerMessages: [ChatMessage], tools: [RoutableTool], model: String, options: ChatRequestOptions
    ) async throws -> RouterResult {
        var routerOptions = ChatRequestOptions()
        routerOptions.maxTokens = configuration.maxRouterTokens
        routerOptions.temperature = 0
        switch configuration.routerToolFormat {
        case .chatTemplate:
            routerOptions.tools = options.tools
            routerOptions.nativeToolSpecs = options.nativeToolSpecs
            routerOptions.systemPrompt = configuration.routerInstruction
        case .functionGemmaInline:
            routerOptions.systemPrompt = (configuration.routerInstruction ?? "") + FunctionGemmaDeclaration.wrap(tools)
        }
        let stream = router.stream(messages: routerMessages, model: routerModel ?? model, options: routerOptions)
        let limit = configuration.routerTimeout

        return try await withThrowingTaskGroup(of: RouterResult.self) { group in
            group.addTask {
                var result = RouterResult()
                for try await event in stream {
                    try Task.checkCancellation()
                    switch event {
                    case .toolCallComplete(let id, let name, let args): result.calls.append((id, name, args))
                    case .text(let t): result.text += t
                    case .usage(let u): result.promptTokens = u.promptTokens; result.completionTokens = u.completionTokens
                    default: break
                    }
                }
                return result
            }
            group.addTask {
                try await Task.sleep(for: limit)
                throw RouterTimeout(limit: limit)
            }
            defer { group.cancelAll() }
            guard let first = try await group.next() else { throw CancellationError() }
            return first
        }
    }

    // MARK: - Helpers

    static func removingTools(_ options: ChatRequestOptions) -> ChatRequestOptions {
        var stripped = options
        stripped.tools = nil
        stripped.nativeToolSpecs = nil
        stripped.toolChoice = nil
        return stripped
    }

    private static func lastUserText(_ messages: [ChatMessage]) -> String {
        guard let user = messages.last(where: { $0.role == .user }) else { return "" }
        return user.content.compactMap { if case .text(let t) = $0 { return t } else { return nil } }.joined(separator: "\n")
    }

    /// Tool calls issued since the most recent user message.
    private static func callsSinceLastUser(_ messages: [ChatMessage]) -> [ChatMessage.ToolCallBlock] {
        let start = messages.lastIndex(where: { $0.role == .user }).map { $0 + 1 } ?? 0
        return messages[start...].flatMap { $0.toolCalls ?? [] }
    }

    /// Canonical `name + sorted-key arguments` so equal calls compare equal regardless of spacing.
    private static func fingerprint(name: String, arguments: String) -> String {
        guard let data = arguments.data(using: .utf8),
              let object = try? JSONSerialization.jsonObject(with: data),
              let canonical = try? JSONSerialization.data(withJSONObject: object, options: [.sortedKeys]),
              let text = String(data: canonical, encoding: .utf8)
        else { return name + arguments }
        return name + text
    }
}

private extension RoutingDecision {
    init(
        _ outcome: Outcome, _ reason: Reason, hasTools: Bool,
        latency: Duration? = nil, tools: [String] = []
    ) {
        self.init(outcome: outcome, reason: reason, toolNames: tools, routerLatency: latency,
                  routerPromptTokens: nil, routerCompletionTokens: nil, routerText: nil, responderHasTools: hasTools)
    }

    func with(hasTools: Bool) -> RoutingDecision {
        RoutingDecision(outcome: outcome, reason: reason, toolNames: toolNames, routerLatency: routerLatency,
                        routerPromptTokens: routerPromptTokens, routerCompletionTokens: routerCompletionTokens,
                        routerText: routerText, responderHasTools: hasTools)
    }

    func with(_ attempt: ToolRoutingProvider.RouterResult, includeText: Bool = true) -> RoutingDecision {
        let trimmed = attempt.text.trimmingCharacters(in: .whitespacesAndNewlines)
        return RoutingDecision(outcome: outcome, reason: reason, toolNames: toolNames, routerLatency: routerLatency,
                               routerPromptTokens: attempt.promptTokens, routerCompletionTokens: attempt.completionTokens,
                               routerText: (includeText && !trimmed.isEmpty) ? trimmed : nil,
                               responderHasTools: responderHasTools)
    }
}
