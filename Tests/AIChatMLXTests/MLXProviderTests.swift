import XCTest
import AIChatCore
@testable import AIChatMLX

// Tests for MLXProvider.toMLXMessages — the history-to-prompt conversion that feeds Gemma 4.
//
// Gemma 4's Jinja template supports native assistant tool_calls followed by
// role:"tool" messages carrying the matching tool_call_id.
//
// These tests run without a model or simulator and must stay green on every PR.

final class MLXProviderHistoryTests: XCTestCase {

    // MARK: - Helpers

    private func msgs(_ history: [ChatMessage], system: String? = nil) -> [[String: any Sendable]] {
        MLXProvider.toMLXMessages(messages: history, systemPrompt: system)
    }

    private func role(_ msg: [String: any Sendable]) -> String {
        msg["role"] as? String ?? "<missing>"
    }

    /// Assert ordinary conversation turns alternate. Native tool results do not
    /// count as turns because Gemma renders them inside the preceding model turn.
    private func assertAlternates(_ result: [[String: any Sendable]], file: StaticString = #file, line: UInt = #line) {
        let turns = result.filter {
            let currentRole = role($0)
            return currentRole != "system" && currentRole != "tool"
        }
        XCTAssertEqual(role(turns.first ?? [:]), "user", file: file, line: line)
        for index in turns.indices.dropFirst() {
            if role(turns[index]) == "user" {
                XCTAssertEqual(role(turns[index - 1]), "assistant", file: file, line: line)
            }
        }
    }

    private func assertNativeToolResults(_ result: [[String: any Sendable]], file: StaticString = #file, line: UInt = #line) {
        for index in result.indices where role(result[index]) == "tool" {
            XCTAssertGreaterThan(index, 0, file: file, line: line)
            XCTAssertEqual(role(result[index - 1]), "assistant", file: file, line: line)
            XCTAssertNotNil(result[index]["tool_call_id"] as? String, file: file, line: line)
        }
    }

    // MARK: - Baseline

    func test_emptyHistory_returnsEmpty() {
        XCTAssertTrue(msgs([]).isEmpty)
    }

    func test_systemPrompt_prependsSystemMessage() {
        let result = msgs([], system: "Be helpful.")
        XCTAssertEqual(result.count, 1)
        XCTAssertEqual(role(result[0]), "system")
        XCTAssertEqual(result[0]["content"] as? String, "Be helpful.")
    }

    func test_simpleUserMessage() {
        let result = msgs([ChatMessage(role: .user, content: "Hello")])
        XCTAssertEqual(result.count, 1)
        XCTAssertEqual(role(result[0]), "user")
    }

    func test_userAssistantPair_alternates() {
        let history: [ChatMessage] = [
            ChatMessage(role: .user,      content: "What is 2+2?"),
            ChatMessage(role: .assistant, content: "4"),
        ]
        let result = msgs(history)
        assertAlternates(result)
        XCTAssertEqual(result.count, 2)
    }

    func test_multiTurn_alternates() {
        let history: [ChatMessage] = [
            ChatMessage(role: .user,      content: "q1"),
            ChatMessage(role: .assistant, content: "a1"),
            ChatMessage(role: .user,      content: "q2"),
            ChatMessage(role: .assistant, content: "a2"),
        ]
        assertAlternates(msgs(history))
    }

    // MARK: - Tool use — single tool call (no preamble text)

    func test_toolCallOnly_noConsecutiveAssistant() {
        // Model emits only a tool call with no preamble text.
        // ChatSession writes: assistant(tool_calls), then submitToolResult writes: tool(result).
        let toolCallId = "call-001"
        let history: [ChatMessage] = [
            ChatMessage(role: .user, content: "Search for cats"),
            ChatMessage(role: .assistant, content: [], toolCalls: [
                .init(id: toolCallId, name: "webSearch", arguments: #"{"query":"cats"}"#)
            ]),
            ChatMessage(toolCallId: toolCallId, content: "Web results: 1. Cats are cool"),
        ]
        let result = msgs(history)
        assertAlternates(result)
        assertNativeToolResults(result)
        // user → assistant(tool_calls) → tool(result)
        XCTAssertEqual(result.count, 3)
        XCTAssertEqual(role(result[2]), "tool")
        XCTAssertEqual(result[2]["tool_call_id"] as? String, toolCallId)
    }

    func test_toolCallOnly_toolResultContentPreserved() {
        let history: [ChatMessage] = [
            ChatMessage(role: .user, content: "Search"),
            ChatMessage(role: .assistant, content: [], toolCalls: [
                .init(id: "id1", name: "webSearch", arguments: "{}")
            ]),
            ChatMessage(toolCallId: "id1", content: "Result text here"),
        ]
        let result = msgs(history)
        let toolResultMsg = result.last!
        XCTAssertEqual(toolResultMsg["content"] as? String, "Result text here")
    }

    // MARK: - Tool use — text preamble + tool call (the double-assistant bug)

    func test_textPreamblePlusToolCall_noConsecutiveAssistant() {
        // This is the exact scenario that triggered TemplateException in production:
        // model generates "I'll search for that.\n\ncall:webSearch{...}"
        // ChatSession writes TWO assistant entries: assistant(tool_calls) then assistant(text).
        let toolCallId = "call-002"
        let history: [ChatMessage] = [
            ChatMessage(role: .user, content: "Latest Anthropic news"),
            // appendAssistantToolCall (called first, from recoverEmbeddedToolCalls):
            ChatMessage(role: .assistant, content: [], toolCalls: [
                .init(id: toolCallId, name: "webSearch", arguments: #"{"query":"Anthropic news"}"#)
            ]),
            // captureAssistantMessage (called second):
            ChatMessage(role: .assistant, content: "I'll search for the latest news."),
            // submitToolResult:
            ChatMessage(toolCallId: toolCallId, content: "Web results: 1. Anthropic news…"),
        ]
        let result = msgs(history)
        assertAlternates(result)
        assertNativeToolResults(result)
    }

    func test_textPreamblePlusToolCall_dropsTextOnlyAssistant() {
        let history: [ChatMessage] = [
            ChatMessage(role: .user, content: "Search something"),
            ChatMessage(role: .assistant, content: [], toolCalls: [
                .init(id: "id1", name: "webSearch", arguments: "{}")
            ]),
            ChatMessage(role: .assistant, content: "I'll search for that."),
            ChatMessage(toolCallId: "id1", content: "Results here"),
        ]
        let result = msgs(history)
        // Should be: user, assistant(tool_calls), tool(result) — 3 messages
        XCTAssertEqual(result.count, 3,
            "Text-only assistant following tool_calls assistant must be dropped; got \(result.count)")
    }

    // MARK: - Full tool-use round trip with final answer

    func test_fullToolUseRoundTrip_alternates() {
        // Complete single-tool-use exchange including the model's final answer.
        let toolCallId = "call-003"
        let history: [ChatMessage] = [
            ChatMessage(role: .user, content: "What is NerdSnipe's address?"),
            ChatMessage(role: .assistant, content: [], toolCalls: [
                .init(id: toolCallId, name: "webSearch",
                      arguments: #"{"query":"NerdSnipe Inc address"}"#)
            ]),
            ChatMessage(toolCallId: toolCallId,
                        content: "Web results:\n1. NerdSnipe Inc — 1000 Innovation Drive, Ottawa"),
            ChatMessage(role: .assistant,
                        content: "NerdSnipe Inc. is located at 1000 Innovation Drive, Ottawa."),
        ]
        let result = msgs(history)
        assertAlternates(result)
        assertNativeToolResults(result)
        // user → assistant(tool_calls) → tool(result) → assistant(answer)
        XCTAssertEqual(result.count, 4)
        XCTAssertEqual(role(result[3]), "assistant")
    }

    func test_fullRoundTripWithPreamble_alternates() {
        // Same but with text preamble before the tool call.
        let toolCallId = "call-004"
        let history: [ChatMessage] = [
            ChatMessage(role: .user, content: "What is NerdSnipe's address?"),
            ChatMessage(role: .assistant, content: [], toolCalls: [
                .init(id: toolCallId, name: "webSearch",
                      arguments: #"{"query":"NerdSnipe Inc address"}"#)
            ]),
            ChatMessage(role: .assistant, content: "Let me look that up."),
            ChatMessage(toolCallId: toolCallId, content: "1000 Innovation Drive, Ottawa"),
            ChatMessage(role: .assistant, content: "The address is 1000 Innovation Drive, Ottawa."),
        ]
        let result = msgs(history)
        assertAlternates(result)
        assertNativeToolResults(result)
    }

    // MARK: - Multi-turn with multiple tool calls

    func test_twoSeparateToolCalls_alternates() {
        let id1 = "call-005", id2 = "call-006"
        let history: [ChatMessage] = [
            ChatMessage(role: .user, content: "Search A then B"),
            ChatMessage(role: .assistant, content: [], toolCalls: [
                .init(id: id1, name: "webSearch", arguments: #"{"query":"A"}"#)
            ]),
            ChatMessage(toolCallId: id1, content: "Results for A"),
            ChatMessage(role: .assistant, content: [], toolCalls: [
                .init(id: id2, name: "webSearch", arguments: #"{"query":"B"}"#)
            ]),
            ChatMessage(toolCallId: id2, content: "Results for B"),
            ChatMessage(role: .assistant, content: "Here are both results."),
        ]
        let result = msgs(history)
        assertAlternates(result)
        assertNativeToolResults(result)
    }

    // MARK: - System prompt does not affect alternation count

    func test_systemPrompt_doesNotCountInAlternation() {
        let history: [ChatMessage] = [
            ChatMessage(role: .user,      content: "Hi"),
            ChatMessage(role: .assistant, content: "Hello"),
        ]
        let result = msgs(history, system: "You are Alric.")
        // system + user + assistant = 3
        XCTAssertEqual(result.count, 3)
        XCTAssertEqual(role(result[0]), "system")
        assertAlternates(result)
    }

    // MARK: - Edge cases

    func test_assistantMessageWithoutContent_skipped() {
        let history: [ChatMessage] = [
            ChatMessage(role: .user, content: "Hi"),
            ChatMessage(role: .assistant, content: []),   // empty content, no tool_calls
        ]
        let result = msgs(history)
        // Empty assistant is skipped by the `guard !parts.isEmpty` check
        XCTAssertEqual(result.count, 1)
    }

    func test_toolResultWithoutPrecedingToolCall_isDropped() {
        // A malformed orphan cannot be associated with a preceding native tool call.
        let history: [ChatMessage] = [
            ChatMessage(role: .user, content: "Hi"),
            ChatMessage(toolCallId: "orphan", content: "Some result"),
        ]
        let result = msgs(history)
        XCTAssertEqual(result.count, 1)
        XCTAssertEqual(role(result[0]), "user")
    }
}

// Tests for MLXProvider.releaseResidency — the public API that force-releases a residency
// slot's resident model and reservation without loading a replacement.
//
// The exhaustive bookkeeping behavior (reservation ended, resident model name cleared, other
// slot untouched) is covered against an injectable `MLXModelRuntime` + fake coordinator in
// MLXResidencyReservationTests.swift, since that's the only way to exercise it without a
// compiled Metal shader library. `releaseResidency` itself is hard-wired to
// `MLXModelRuntime.shared` (by design — it's the process-wide singleton every `MLXProvider`
// resolves through), so what's verified here is that the public entry point is callable and
// safe when nothing is resident in the slot, i.e. it does not start/end a real wired-memory
// ticket and does not crash under `swift test`.
final class MLXProviderReleaseResidencyTests: XCTestCase {

    func test_releaseResidency_isSafeNoOpWhenSlotNotLoaded() async {
        // .auxiliary is not expected to hold a reservation from any other test in this process,
        // so this exercises the no-ticket-to-end path — the only path safe to run here, since
        // ending a real ticket would reach through to WiredMemoryManager's live coordinator,
        // which probes Device.defaultDevice() and aborts without a compiled MLX metallib.
        await MLXProvider.releaseResidency(.auxiliary)
        // No crash, no throw: the API is callable and inert when the slot is already empty.
    }
}
