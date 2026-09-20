import XCTest
import AIChatCore
@testable import AIChatMLX

// Branch coverage for ToolRoutingProvider with scripted fake providers — no model, no Metal.

/// A provider whose output is scripted per call, recording what it was asked.
private final class ScriptedProvider: ChatProvider, @unchecked Sendable {
    enum Step { case event(ChatStreamEvent), sleep(Duration), fail(Error) }
    struct Call { let messages: [ChatMessage]; let model: String; let options: ChatRequestOptions }

    let id = "scripted"
    let name: String
    private let lock = NSLock()
    private var _calls: [Call] = []
    private var _terminated = 0
    private let script: @Sendable (Int) -> [Step]

    init(name: String = "scripted", _ script: @escaping @Sendable (Int) -> [Step]) {
        self.name = name
        self.script = script
    }

    var calls: [Call] { lock.withLock { _calls } }
    var terminatedCount: Int { lock.withLock { _terminated } }

    func stream(messages: [ChatMessage], model: String, options: ChatRequestOptions)
        -> AsyncThrowingStream<ChatStreamEvent, Error> {
        let index = lock.withLock { () -> Int in
            _calls.append(Call(messages: messages, model: model, options: options))
            return _calls.count - 1
        }
        let steps = script(index)
        return AsyncThrowingStream { c in
            let task = Task {
                do {
                    for step in steps {
                        switch step {
                        case .event(let e): c.yield(e)
                        case .sleep(let d): try await Task.sleep(for: d)
                        case .fail(let e): throw e
                        }
                    }
                    c.finish()
                } catch { c.finish(throwing: error) }
            }
            c.onTermination = { [weak self] _ in
                task.cancel()
                self?.lock.withLock { self?._terminated += 1 }
            }
        }
    }

    func complete(messages: [ChatMessage], model: String, options: ChatRequestOptions) async throws -> ChatCompletionResult {
        throw ChatError.invalidConfiguration("unused")
    }

    static func replying(_ text: String) -> ScriptedProvider {
        ScriptedProvider { _ in [.event(.text(text)), .event(.done)] }
    }
    static func calling(_ name: String, _ args: String) -> ScriptedProvider {
        ScriptedProvider { _ in [.event(.toolCallComplete(id: "r1", name: name, arguments: args)), .event(.done)] }
    }
}

private final class Decisions: @unchecked Sendable {
    private let lock = NSLock()
    private var items: [RoutingDecision] = []
    var all: [RoutingDecision] { lock.withLock { items } }
    var last: RoutingDecision? { all.last }
    func record(_ d: RoutingDecision) { lock.withLock { items.append(d) } }
}

final class ToolRoutingProviderTests: XCTestCase {

    // MARK: Fixtures

    private static func spec(
        _ name: String, required: [String] = [], properties: [String: [String: any Sendable]] = [:]
    ) -> [String: any Sendable] {
        [
            "type": "function",
            "function": [
                "name": name,
                "description": "desc",
                "parameters": [
                    "type": "object",
                    "properties": properties.mapValues { $0 as [String: any Sendable] } as [String: any Sendable],
                    "required": required,
                ] as [String: any Sendable],
            ] as [String: any Sendable],
        ]
    }

    private let weather = ToolRoutingProviderTests.spec(
        "get_weather", required: ["city"],
        properties: ["city": ["type": "string"], "days": ["type": "integer"], "unit": ["type": "string", "enum": ["celsius", "fahrenheit"]]]
    )
    private let notes = ToolRoutingProviderTests.spec("create_note", required: ["title"], properties: ["title": ["type": "string"]])

    private func options(tools: Bool = true) -> ChatRequestOptions {
        var o = ChatRequestOptions(systemPrompt: "You are a persona with a very long preamble.")
        if tools { o.nativeToolSpecs = [weather, notes] }
        return o
    }

    private func make(
        router: ScriptedProvider, responder: ScriptedProvider,
        _ configure: (inout ToolRoutingProvider.Configuration) -> Void = { _ in }
    ) -> (ToolRoutingProvider, Decisions) {
        var config = ToolRoutingProvider.Configuration()
        configure(&config)
        let decisions = Decisions()
        return (ToolRoutingProvider(router: router, responder: responder, configuration: config,
                                    onDecision: { decisions.record($0) }), decisions)
    }

    private func collect(_ p: ToolRoutingProvider, _ messages: [ChatMessage], _ options: ChatRequestOptions) async throws -> [ChatStreamEvent] {
        var out: [ChatStreamEvent] = []
        for try await e in p.stream(messages: messages, model: "m", options: options) { out.append(e) }
        return out
    }

    private let userWeather = [ChatMessage(role: .user, content: "weather in Paris?")]

    private func hasTools(_ call: ScriptedProvider.Call) -> Bool {
        call.options.tools != nil || call.options.nativeToolSpecs != nil
    }

    // MARK: No tools / disabled

    func test_noTools_delegatesStraightToResponder() async throws {
        let router = ScriptedProvider.replying("x"), responder = ScriptedProvider.replying("hi")
        let (p, d) = make(router: router, responder: responder)
        let events = try await collect(p, userWeather, options(tools: false))
        XCTAssertEqual(events, [.text("hi"), .done])
        XCTAssertEqual(router.calls.count, 0)
        XCTAssertEqual(d.last?.reason, .noTools)
        XCTAssertEqual(d.last?.outcome, .passthrough)
    }

    func test_toolChoiceNone_stripsToolsAndSkipsRouter() async throws {
        let router = ScriptedProvider.replying("x"), responder = ScriptedProvider.replying("hi")
        let (p, d) = make(router: router, responder: responder)
        var o = options(); o.toolChoice = ChatRequestOptions.ToolChoiceOption.none
        _ = try await collect(p, userWeather, o)
        XCTAssertEqual(router.calls.count, 0)
        XCTAssertFalse(hasTools(responder.calls[0]))
        XCTAssertEqual(d.last?.reason, .toolsDisabled)
    }

    func test_routerDisabled_responderGetsToolsAsSent() async throws {
        let router = ScriptedProvider.replying("x"), responder = ScriptedProvider.replying("hi")
        let (p, d) = make(router: router, responder: responder) { $0.isRouterEnabled = false }
        _ = try await collect(p, userWeather, options())
        XCTAssertEqual(router.calls.count, 0)
        XCTAssertTrue(hasTools(responder.calls[0]))
        XCTAssertEqual(d.last?.reason, .routerDisabled)
    }

    // MARK: Routed

    func test_toolRouted_emitsCallAndSkipsResponder() async throws {
        let router = ScriptedProvider.calling("get_weather", #"{"city":"Paris","days":"3","unit":"Celsius","junk":1}"#)
        let responder = ScriptedProvider.replying("never")
        let (p, d) = make(router: router, responder: responder)
        let events = try await collect(p, userWeather, options())
        XCTAssertEqual(events.count, 2)
        guard case .toolCallComplete(_, let name, let args) = events[0] else { return XCTFail("expected tool call") }
        XCTAssertEqual(name, "get_weather")
        // coerced: "3" -> 3, "Celsius" -> "celsius", undeclared key dropped
        XCTAssertEqual(args, #"{"city":"Paris","days":3,"unit":"celsius"}"#)
        XCTAssertEqual(events[1], .done)
        XCTAssertEqual(responder.calls.count, 0)
        XCTAssertEqual(d.last?.outcome, .routedToTool)
        XCTAssertEqual(d.last?.toolNames, ["get_weather"])
        XCTAssertNotNil(d.last?.routerLatency)
    }

    func test_routerPromptIsMinimal() async throws {
        let router = ScriptedProvider.calling("get_weather", #"{"city":"Paris"}"#)
        let (p, _) = make(router: router, responder: .replying("x"))
        let history = [
            ChatMessage(role: .user, content: "old question"), ChatMessage(role: .assistant, content: "old answer"),
            ChatMessage(role: .user, content: "weather in Paris?"),
        ]
        _ = try await collect(p, history, options())
        let call = router.calls[0]
        XCTAssertEqual(call.messages.map(\.role), [.user])
        XCTAssertEqual(call.options.systemPrompt, ToolRoutingProvider.Configuration().routerInstruction)
        XCTAssertNotNil(call.options.nativeToolSpecs)
        XCTAssertEqual(call.options.maxTokens, 128)
    }

    func test_routerContextTurns_includesEarlierTurns() {
        let history = [
            ChatMessage(role: .user, content: "a"), ChatMessage(role: .assistant, content: "b"),
            ChatMessage(role: .user, content: "c"),
        ]
        XCTAssertEqual(RouterPrompt.recentTurns(history, turns: 2, characterLimit: 100).count, 3)
        XCTAssertEqual(RouterPrompt.recentTurns(history, turns: 1, characterLimit: 100).count, 1)
        XCTAssertEqual(RouterPrompt.clip(String(repeating: "x", count: 50), limit: 10).count, 13)
    }

    // MARK: Fall-through

    func test_invalidJSON_fallsBackWithoutTools() async throws {
        let router = ScriptedProvider.calling("get_weather", "not json{")
        let responder = ScriptedProvider.replying("plain")
        let (p, d) = make(router: router, responder: responder)
        let events = try await collect(p, userWeather, options())
        XCTAssertEqual(events, [.text("plain"), .done])
        XCTAssertFalse(hasTools(responder.calls[0]))
        XCTAssertEqual(d.last?.outcome, .fallback)
        guard case .invalidArguments? = d.last?.reason else { return XCTFail("\(String(describing: d.last))") }
    }

    func test_unknownTool_fallsBack() async throws {
        let router = ScriptedProvider.calling("delete_all", "{}")
        let responder = ScriptedProvider.replying("no")
        let (p, d) = make(router: router, responder: responder)
        let events = try await collect(p, userWeather, options())
        XCTAssertFalse(events.contains { if case .toolCallComplete = $0 { return true } else { return false } })
        XCTAssertEqual(d.last?.reason, .unknownTool("delete_all"))
        XCTAssertFalse(hasTools(responder.calls[0]))
    }

    func test_missingRequiredArgument_fallsBack() async throws {
        let router = ScriptedProvider.calling("get_weather", #"{"days":2}"#)
        let responder = ScriptedProvider.replying("ask city")
        let (p, d) = make(router: router, responder: responder)
        _ = try await collect(p, userWeather, options())
        XCTAssertEqual(d.last?.reason, .invalidArguments("missing required 'city'"))
        XCTAssertEqual(responder.calls.count, 1)
    }

    func test_badEnumValue_fallsBack() async throws {
        let router = ScriptedProvider.calling("get_weather", #"{"city":"Paris","unit":"kelvin"}"#)
        let (p, d) = make(router: router, responder: .replying("x"))
        _ = try await collect(p, userWeather, options())
        XCTAssertEqual(d.last?.outcome, .fallback)
    }

    func test_routerText_isPassthroughAndNeverLeaksToUser() async throws {
        let router = ScriptedProvider.replying("I am chatting instead")
        let responder = ScriptedProvider.replying("answer")
        let (p, d) = make(router: router, responder: responder)
        let events = try await collect(p, userWeather, options())
        XCTAssertEqual(events, [.text("answer"), .done])
        XCTAssertEqual(d.last?.outcome, .passthrough)
        XCTAssertEqual(d.last?.reason, .routerDeclined)
        XCTAssertEqual(d.last?.routerText, "I am chatting instead")
        XCTAssertFalse(hasTools(responder.calls[0]))
        // The responder keeps its full prompt (persona preamble) — only tools are removed.
        XCTAssertEqual(responder.calls[0].options.systemPrompt, "You are a persona with a very long preamble.")
        XCTAssertEqual(responder.calls[0].messages.count, 1)
    }

    func test_routerEmpty_isPassthrough() async throws {
        let router = ScriptedProvider { _ in [.event(.done)] }
        let (p, d) = make(router: router, responder: .replying("answer"))
        _ = try await collect(p, userWeather, options())
        XCTAssertEqual(d.last?.reason, .routerDeclined)
    }

    func test_fallbackPolicy_respondWithTools_givesResponderTools() async throws {
        let responder = ScriptedProvider.replying("answer")
        let (p, d) = make(router: .replying("chat"), responder: responder) { $0.fallback = .respondWithTools }
        _ = try await collect(p, userWeather, options())
        XCTAssertTrue(hasTools(responder.calls[0]))
        XCTAssertEqual(d.last?.responderHasTools, true)
    }

    func test_requiresTool_missedByRouter_responderGetsTools() async throws {
        // Lesson from production: a router can miss a mandatory-tool request. The escape hatch
        // must hand the responder the tools even under the default fast-path policy.
        let responder = ScriptedProvider.replying("answer")
        let (p, d) = make(router: .replying("no tool needed"), responder: responder) {
            $0.requiresTool = { $0.contains("conversation") }
        }
        _ = try await collect(p, [ChatMessage(role: .user, content: "draft a reply to conversation 8f3a")], options())
        XCTAssertTrue(hasTools(responder.calls[0]))
        XCTAssertEqual(d.last?.responderHasTools, true)
        // ... but plain chat still takes the fast path.
        _ = try await collect(p, [ChatMessage(role: .user, content: "hello")], options())
        XCTAssertFalse(hasTools(responder.calls[1]))
    }

    // MARK: Router failure

    func test_routerThrows_doesNotFailTheTurn() async throws {
        let router = ScriptedProvider { _ in [.fail(ChatError.modelNotFound(modelId: "router"))] }
        let responder = ScriptedProvider.replying("still answered")
        let (p, d) = make(router: router, responder: responder)
        let events = try await collect(p, userWeather, options())
        XCTAssertEqual(events, [.text("still answered"), .done])
        XCTAssertEqual(d.last?.outcome, .fallback)
        guard case .routerFailed? = d.last?.reason else { return XCTFail("\(String(describing: d.last))") }
        XCTAssertFalse(hasTools(responder.calls[0]))
    }

    func test_routerTimeout_fallsBackAndStopsRouter() async throws {
        let router = ScriptedProvider { _ in [.sleep(.seconds(30))] }
        let responder = ScriptedProvider.replying("answer")
        let (p, d) = make(router: router, responder: responder) { $0.routerTimeout = .milliseconds(150) }
        let events = try await collect(p, userWeather, options())
        XCTAssertEqual(events, [.text("answer"), .done])
        XCTAssertEqual(d.last?.reason, .routerTimedOut)
        XCTAssertEqual(router.terminatedCount, 1, "timed-out router stream must be terminated")
    }

    func test_responderError_propagatesAsChatError() async throws {
        let responder = ScriptedProvider { _ in [.fail(ChatError.outOfMemory(underlying: nil))] }
        let (p, _) = make(router: .replying("chat"), responder: responder)
        do {
            _ = try await collect(p, userWeather, options())
            XCTFail("expected throw")
        } catch let error as ChatError {
            guard case .outOfMemory = error else { return XCTFail("\(error)") }
        }
    }

    func test_responderError_whenRouterAlsoFailed_surfaces() async throws {
        let router = ScriptedProvider { _ in [.fail(ChatError.modelNotFound(modelId: "r"))] }
        let responder = ScriptedProvider { _ in [.fail(ChatError.modelNotFound(modelId: "resp"))] }
        let (p, _) = make(router: router, responder: responder)
        do {
            _ = try await collect(p, userWeather, options())
            XCTFail("expected throw")
        } catch let error as ChatError {
            guard case .modelNotFound(let id) = error else { return XCTFail("\(error)") }
            XCTAssertEqual(id, "resp")
        }
    }

    // MARK: After tool result

    private func toolHistory(call: String = #"{"city":"Paris"}"#) -> [ChatMessage] {
        [
            ChatMessage(role: .user, content: "weather in Paris?"),
            ChatMessage(role: .assistant, content: [], toolCalls: [.init(id: "c1", name: "get_weather", arguments: call)]),
            ChatMessage(toolCallId: "c1", content: "12C"),
        ]
    }

    func test_afterToolResult_respond_default() async throws {
        let router = ScriptedProvider.calling("get_weather", #"{"city":"Paris"}"#)
        let responder = ScriptedProvider.replying("It is 12C")
        let (p, d) = make(router: router, responder: responder)
        let events = try await collect(p, toolHistory(), options())
        XCTAssertEqual(events, [.text("It is 12C"), .done])
        XCTAssertEqual(router.calls.count, 0)
        XCTAssertFalse(hasTools(responder.calls[0]))
        XCTAssertEqual(responder.calls[0].messages.count, 3, "responder sees the full history incl. the tool result")
        XCTAssertEqual(d.last?.reason, .afterToolResult)
    }

    func test_afterToolResult_respondWithTools() async throws {
        let responder = ScriptedProvider.replying("x")
        let (p, _) = make(router: .replying("x"), responder: responder) { $0.afterToolResult = .respondWithTools }
        _ = try await collect(p, toolHistory(), options())
        XCTAssertTrue(hasTools(responder.calls[0]))
    }

    func test_afterToolResult_routeAgain_chainsNewCall() async throws {
        let router = ScriptedProvider.calling("create_note", #"{"title":"Paris 12C"}"#)
        let responder = ScriptedProvider.replying("x")
        let (p, d) = make(router: router, responder: responder) { $0.afterToolResult = .routeAgain }
        let events = try await collect(p, toolHistory(), options())
        guard case .toolCallComplete(_, let name, _) = events[0] else { return XCTFail("expected call") }
        XCTAssertEqual(name, "create_note")
        XCTAssertEqual(responder.calls.count, 0)
        XCTAssertEqual(d.last?.outcome, .routedToTool)
        // The router saw the request plus the result as plain text.
        let text = router.calls[0].messages.map { m in m.content.compactMap { if case .text(let t) = $0 { return t } else { return nil } }.joined() }
        XCTAssertEqual(router.calls[0].messages.count, 1)
        XCTAssertTrue(text[0].contains("Result of get_weather: 12C"))
    }

    func test_afterToolResult_routeAgain_repeatedCallIsBlocked() async throws {
        let router = ScriptedProvider.calling("get_weather", #"{ "city" : "Paris" }"#)
        let responder = ScriptedProvider.replying("It is 12C")
        let (p, d) = make(router: router, responder: responder) { $0.afterToolResult = .routeAgain }
        let events = try await collect(p, toolHistory(), options())
        XCTAssertEqual(events, [.text("It is 12C"), .done])
        XCTAssertEqual(d.last?.reason, .repeatedCall)
        XCTAssertFalse(hasTools(responder.calls[0]))
    }

    func test_afterToolResult_routeAgain_declineComposesAnswer() async throws {
        let responder = ScriptedProvider.replying("It is 12C")
        let (p, d) = make(router: .replying("done"), responder: responder) { $0.afterToolResult = .routeAgain }
        let events = try await collect(p, toolHistory(), options())
        XCTAssertEqual(events, [.text("It is 12C"), .done])
        XCTAssertEqual(d.last?.reason, .routerDeclined)
        XCTAssertFalse(hasTools(responder.calls[0]))
    }

    func test_afterToolResult_routeAgain_capsChain() async throws {
        let router = ScriptedProvider.calling("create_note", #"{"title":"x"}"#)
        let (p, d) = make(router: router, responder: .replying("done")) {
            $0.afterToolResult = .routeAgain; $0.maxRoutedCallsPerTurn = 1
        }
        _ = try await collect(p, toolHistory(), options())
        XCTAssertEqual(router.calls.count, 0)
        XCTAssertEqual(d.last?.reason, .repeatedCall)
    }

    // MARK: Cancellation

    func test_cancellingConsumer_cancelsRouterStage() async throws {
        let router = ScriptedProvider { _ in [.sleep(.seconds(60))] }
        let responder = ScriptedProvider.replying("x")
        let (p, _) = make(router: router, responder: responder) { $0.routerTimeout = .seconds(120) }
        let consumer = Task { try await self.collect(p, self.userWeather, self.options()) }
        try await Task.sleep(for: .milliseconds(200))
        XCTAssertEqual(router.calls.count, 1)
        consumer.cancel()
        _ = try? await consumer.value
        try await Task.sleep(for: .milliseconds(200))
        XCTAssertEqual(router.terminatedCount, 1)
        XCTAssertEqual(responder.calls.count, 0, "responder must not start after cancellation")
    }

    func test_cancellingConsumer_cancelsResponderStage() async throws {
        let responder = ScriptedProvider { _ in [.event(.text("a")), .sleep(.seconds(60))] }
        let (p, _) = make(router: .replying("chat"), responder: responder)
        let started = expectation(description: "first token")
        let consumer = Task {
            for try await e in p.stream(messages: self.userWeather, model: "m", options: self.options()) {
                if case .text = e { started.fulfill() }
            }
        }
        await fulfillment(of: [started], timeout: 5)
        consumer.cancel()
        _ = try? await consumer.value
        try await Task.sleep(for: .milliseconds(200))
        XCTAssertEqual(responder.terminatedCount, 1)
    }

    // MARK: complete()

    func test_complete_returnsToolCalls() async throws {
        let (p, _) = make(router: .calling("create_note", #"{"title":"hi"}"#), responder: .replying("x"))
        let result = try await p.complete(messages: [ChatMessage(role: .user, content: "note hi")], model: "m", options: options())
        XCTAssertEqual(result.message.toolCalls?.first?.name, "create_note")
        guard case .toolCalls = result.finishReason else { return XCTFail() }
    }

    func test_complete_returnsResponderText() async throws {
        let (p, _) = make(router: .replying("chat"), responder: .replying("hello there"))
        let result = try await p.complete(messages: userWeather, model: "m", options: options())
        XCTAssertEqual(result.message.content.count, 1)
        XCTAssertNil(result.message.toolCalls)
    }

    // MARK: Validation

    func test_validate_typeCoercions() {
        let tool = RoutableTool(name: "t", parameters: [
            "properties": ["n": ["type": "number"], "b": ["type": "boolean"], "s": ["type": "string"], "a": ["type": "array"]],
            "required": ["n"],
        ])
        let ok = RoutableTool.validate(name: "t", argumentsJSON: #"{"n":"2.5","b":"yes","s":7,"a":"x"}"#, against: [tool])
        XCTAssertEqual(try ok.get(), #"{"a":["x"],"b":true,"n":2.5,"s":"7"}"#)
        if case .success = RoutableTool.validate(name: "t", argumentsJSON: "{}", against: [tool]) { XCTFail("missing required must fail") }
        if case .success = RoutableTool.validate(name: "t", argumentsJSON: #"{"n":"abc"}"#, against: [tool]) { XCTFail("non-numeric must fail") }
        if case .success = RoutableTool.validate(name: "t", argumentsJSON: #"[1]"#, against: [tool]) { XCTFail("array is not an object") }
    }

    // MARK: FunctionGemma declaration rendering

    func test_functionGemmaDeclaration_matchesReferenceJinjaRender() {
        // Expected text produced by rendering the model's chat_template.jinja with the reference
        // (Python) Jinja engine for the same tool.
        let tool = RoutableTool(name: "get_weather", description: "Get the current weather or forecast for a city.", parameters: [
            "type": "object",
            "properties": [
                "city": ["type": "string", "description": "City name"],
                "days": ["type": "integer", "description": "Forecast days"],
                "unit": ["type": "string", "description": "Temperature unit", "enum": ["celsius", "fahrenheit"]],
            ],
            "required": ["city"],
        ])
        XCTAssertEqual(tool.declaration, "declaration:get_weather{description:<escape>Get the current weather or forecast for a city.<escape>,parameters:{properties:{city:{description:<escape>City name<escape>,type:<escape>STRING<escape>},days:{description:<escape>Forecast days<escape>,type:<escape>INTEGER<escape>},unit:{description:<escape>Temperature unit<escape>,enum:[<escape>celsius<escape>,<escape>fahrenheit<escape>],type:<escape>STRING<escape>}},required:[<escape>city<escape>],type:<escape>OBJECT<escape>}}")
    }

    func test_inlineFormat_sendsDeclarationsInSystemTextWithoutTools() async throws {
        let router = ScriptedProvider.replying("chat")
        let (p, _) = make(router: router, responder: .replying("x")) { $0.routerToolFormat = .functionGemmaInline }
        _ = try await collect(p, userWeather, options())
        let sent = router.calls[0].options
        XCTAssertNil(sent.tools); XCTAssertNil(sent.nativeToolSpecs)
        let system = try XCTUnwrap(sent.systemPrompt)
        XCTAssertTrue(system.hasPrefix("You are a model that can do function calling with the following functions<start_function_declaration>declaration:get_weather{"))
        XCTAssertTrue(system.contains("<end_function_declaration><start_function_declaration>declaration:create_note{"))
    }
}
