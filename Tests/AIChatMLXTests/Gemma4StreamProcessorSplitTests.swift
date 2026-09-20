import XCTest
@testable import AIChatMLX

/// Gemma 4 output arrives in arbitrary token-sized chunks; markers such as `<|channel>thought`,
/// `<|tool_call>` and `<tool_call|>` routinely straddle chunk boundaries. Every stream here is
/// fed whole, split at every possible index, and one character at a time; the aggregate result
/// (reasoning, text, tool calls) must be identical each way.
final class Gemma4StreamProcessorSplitTests: XCTestCase {

    struct Aggregate: Equatable, CustomStringConvertible {
        var reasoning = ""
        var text = ""
        var calls: [String] = []   // "name|argsJSON" with keys sorted
        var description: String { "reasoning=\(reasoning.debugDescription) text=\(text.debugDescription) calls=\(calls)" }
    }

    private func canonical(_ json: String) -> String {
        guard let d = json.data(using: .utf8),
              let o = try? JSONSerialization.jsonObject(with: d),
              let out = try? JSONSerialization.data(withJSONObject: o, options: [.sortedKeys, .withoutEscapingSlashes]),
              let s = String(data: out, encoding: .utf8) else { return json }
        return s
    }

    private func run(_ chunks: [String]) -> Aggregate {
        var p = Gemma4StreamProcessor(tools: nil)
        var agg = Aggregate()
        func absorb(_ evs: [Gemma4StreamProcessor.Event]) {
            for e in evs {
                switch e {
                case .reasoning(let r): agg.reasoning += r
                case .text(let t): agg.text += t
                case .toolCall(let n, let a): agg.calls.append("\(n)|\(canonical(a))")
                }
            }
        }
        for c in chunks { absorb(p.processChunk(c)) }
        absorb(p.finish())
        return agg
    }

    private func assertSplitInvariant(
        _ stream: String, file: StaticString = #filePath, line: UInt = #line,
        expected: Aggregate? = nil
    ) {
        let whole = run([stream])
        if let expected { XCTAssertEqual(whole, expected, "whole-stream result", file: file, line: line) }
        let chars = stream.map(String.init)
        XCTAssertEqual(run(chars), whole, "char-by-char differs for \(stream.debugDescription)", file: file, line: line)
        for i in 1..<max(stream.count, 1) {
            let idx = stream.index(stream.startIndex, offsetBy: i)
            let two = [String(stream[..<idx]), String(stream[idx...])]
            let got = run(two)
            if got != whole {
                XCTFail("split at \(i) \(two.map { $0.debugDescription }) -> \(got)\n  whole -> \(whole)", file: file, line: line)
                return
            }
        }
    }

    // MARK: thought channel

    func test_thoughtThenAnswer() {
        assertSplitInvariant("<|channel>thought\nLet me think.\n<channel|>The answer is 391.",
            expected: .init(reasoning: "\nLet me think.\n", text: "The answer is 391.", calls: []))
    }

    func test_plainText_noMarkers() {
        assertSplitInvariant("Just a normal answer, with < angle and | pipes and <b>html</b>.",
            expected: .init(reasoning: "", text: "Just a normal answer, with < angle and | pipes and <b>html</b>.", calls: []))
    }

    func test_emptyThought() {
        assertSplitInvariant("<|channel>thought\n<channel|>Direct answer.")
    }

    func test_thoughtNeverClosed_flushesAsReasoning() {
        let r = run(["<|channel>thought\nstill thinking when tokens ran out"])
        XCTAssertEqual(r.text, "")
        XCTAssertTrue(r.reasoning.contains("still thinking"))
    }

    func test_textContainingAngleBracketsNearMarkerPrefix() {
        assertSplitInvariant("Use <|not a marker|> and <channel> tags. 5 < 6.")
    }

    // MARK: inline calls

    func test_inlineCall_gemmaQuoting() {
        assertSplitInvariant(#"call:get_weather{city:<|"|>Paris<|"|>}"#,
            expected: .init(reasoning: "", text: "", calls: [#"get_weather|{"city":"Paris"}"#]))
    }

    func test_inlineCall_withPreambleText() {
        assertSplitInvariant(#"Sure, checking. call:get_weather{city:<|"|>Paris<|"|>}"#,
            expected: .init(reasoning: "", text: "Sure, checking. ", calls: [#"get_weather|{"city":"Paris"}"#]))
    }

    func test_inlineCall_multipleArgsAndTypes() {
        assertSplitInvariant(#"call:book{title:<|"|>Dune<|"|>,qty:2,gift:true}"#,
            expected: .init(reasoning: "", text: "", calls: [#"book|{"gift":true,"qty":2,"title":"Dune"}"#]))
    }

    func test_inlineCall_multipleCallsInOneTurn() {
        assertSplitInvariant(#"call:a{x:<|"|>1<|"|>}call:b{y:<|"|>2<|"|>}"#,
            expected: .init(reasoning: "", text: "", calls: [#"a|{"x":"1"}"#, #"b|{"y":"2"}"#]))
    }

    func test_inlineCall_stringWithEmbeddedQuotesAndNewline() {
        assertSplitInvariant("call:say{text:<|\"|>He said \"hi\"\nthen left<|\"|>}",
            expected: .init(reasoning: "", text: "", calls: ["say|" + canonical("{\"text\":\"He said \\\"hi\\\"\\nthen left\"}")]))
    }

    func test_inlineCall_stringContainingBracesAndComma() {
        assertSplitInvariant(#"call:code{body:<|"|>fn a() { return 1; }, b<|"|>}"#,
            expected: .init(reasoning: "", text: "", calls: ["code|" + canonical(#"{"body":"fn a() { return 1; }, b"}"#)]))
    }

    func test_inlineCall_nestedObject() {
        assertSplitInvariant(#"call:f{obj:{a:{b:1}},n:2}"#,
            expected: .init(reasoning: "", text: "", calls: ["f|" + canonical(#"{"obj":{"a":{"b":1}},"n":2}"#)]))
    }

    func test_inlineCall_emptyArgs() {
        assertSplitInvariant("call:ping{}", expected: .init(reasoning: "", text: "", calls: ["ping|{}"]))
    }

    func test_inlineCall_thenTrailingText() {
        assertSplitInvariant(#"call:f{a:<|"|>x<|"|>} and that's it"#)
    }

    // MARK: native <|tool_call> tokens

    func test_nativeToolCallTokens() {
        assertSplitInvariant(#"<|tool_call>call:get_weather{city:<|"|>Paris<|"|>}<tool_call|>"#,
            expected: .init(reasoning: "", text: "", calls: [#"get_weather|{"city":"Paris"}"#]))
    }

    func test_nativeToolCall_afterThought() {
        assertSplitInvariant(#"<|channel>thought\#nneed weather<channel|><|tool_call>call:get_weather{city:<|"|>Tokyo<|"|>}<tool_call|>"#,
            expected: .init(reasoning: "\nneed weather", text: "", calls: [#"get_weather|{"city":"Tokyo"}"#]))
    }

    func test_nativeToolCall_multipleInOneTurn() {
        assertSplitInvariant(#"<|tool_call>call:a{x:<|"|>1<|"|>}<tool_call|><|tool_call>call:b{y:<|"|>2<|"|>}<tool_call|>"#,
            expected: .init(reasoning: "", text: "", calls: [#"a|{"x":"1"}"#, #"b|{"y":"2"}"#]))
    }

    // MARK: truncated / malformed

    func test_truncatedCall_atEndOfStream_isNotSilentlyLost() {
        let r = run([#"call:get_weather{city:<|"|>Par"#])
        // A call cut off by the token limit must not vanish: it is either a call or visible text.
        XCTAssertTrue(!r.calls.isEmpty || !r.text.isEmpty, "truncated call was dropped: \(r)")
    }

    func test_proseMentioningCall_isNotAToolCall() {
        let r = run(["Give me a call: 555-1234 when you can."])
        XCTAssertEqual(r.calls, [])
        XCTAssertEqual(r.text, "Give me a call: 555-1234 when you can.")
    }

    // MARK: more argument shapes

    func test_inlineCall_jsonStyleQuotedKeysAndValues() {
        assertSplitInvariant(#"call:webSearch{"query":"say \"hi\" {now}","n":3}"#,
            expected: .init(reasoning: "", text: "", calls: ["webSearch|" + canonical(#"{"query":"say \"hi\" {now}","n":3}"#)]))
    }

    func test_inlineCall_arrayAndUnicodeArgs() {
        assertSplitInvariant("call:tag{items:[<|\"|>a<|\"|>,<|\"|>日本語 🎉<|\"|>,3],flag:false}",
            expected: .init(reasoning: "", text: "", calls: ["tag|" + canonical(#"{"items":["a","日本語 🎉",3],"flag":false}"#)]))
    }

    func test_inlineCall_urlNotSlashEscaped() {
        let r = run([#"call:fetch{url:<|"|>https://example.com/a/b<|"|>}"#])
        XCTAssertEqual(r.calls, [#"fetch|{"url":"https://example.com/a/b"}"#])
    }

    func test_thoughtThenInlineCallThenText() {
        assertSplitInvariant(#"<|channel>thought\#nplan<channel|>Checking. call:f{a:<|"|>x<|"|>} Done."#,
            expected: .init(reasoning: "\nplan", text: "Checking.  Done.", calls: [#"f|{"a":"x"}"#]))
    }

    func test_callSyntaxInsideThought_isNotExecuted() {
        let r = run([#"<|channel>thought\#nI could call:f{a:1} maybe<channel|>No tools needed."#])
        XCTAssertEqual(r.calls, [])
        XCTAssertEqual(r.text, "No tools needed.")
    }

    func test_unterminatedNativeToolBlock_atEOS_stillParses() {
        let r = run([#"<|tool_call>call:f{a:<|"|>x<|"|>}"#])
        XCTAssertEqual(r.calls, [#"f|{"a":"x"}"#])
        XCTAssertEqual(r.text, "")
    }

    func test_unparseableNativeToolBlock_isVisibleNotSwallowed() {
        let r = run(["<|tool_call>garbage without a call<tool_call|>"])
        XCTAssertEqual(r.calls, [])
        XCTAssertTrue(r.text.contains("garbage"))
    }

    func test_recallWordIsNotACall() {
        let r = run(["I recall:{not a call}"])
        XCTAssertEqual(r.calls, [])
        XCTAssertEqual(r.text, "I recall:{not a call}")
    }

    func test_partialJSONArguments_neverEmitBeforeComplete() {
        var p = Gemma4StreamProcessor(tools: nil)
        var evs: [Gemma4StreamProcessor.Event] = []
        for chunk in ["call:f{a:<|\"|>par", "tial", "<|\"|>,b:{c:", "1}", "}"] {
            let e = p.processChunk(chunk)
            if chunk != "}" { XCTAssertTrue(e.allSatisfy { if case .toolCall = $0 { return false } else { return true } }, "early call at \(chunk)") }
            evs += e
        }
        evs += p.finish()
        XCTAssertEqual(evs.count, 1)
        XCTAssertEqual(evs.first, .toolCall(name: "f", argumentsJSON: #"{"a":"partial","b":{"c":1}}"#))
    }
}
