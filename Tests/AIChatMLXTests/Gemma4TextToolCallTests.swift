import XCTest
@testable import AIChatMLX

/// Tool calls a Gemma model writes as plain text (`call:`, `<tool_call>` XML, ```` ```tool_code ````
/// Python) become tool-call events at stream level. Every case runs whole and one character at a
/// time, since a marker can be split across chunks anywhere.
final class Gemma4TextToolCallTests: XCTestCase {

    struct Out: Equatable {
        var text = ""
        var calls: [String] = []      // "name|argsJSON"
        var names: [String] { calls.map { String($0.split(separator: "|", maxSplits: 1)[0]) } }
        func args(_ i: Int) -> [String: Any] {
            let json = String(calls[i].split(separator: "|", maxSplits: 1)[1])
            return (try? JSONSerialization.jsonObject(with: Data(json.utf8)) as? [String: Any]) ?? [:]
        }
    }

    private func run(_ input: String, chunk: Int, tools: [[String: any Sendable]]? = nil) -> Out {
        var p = Gemma4StreamProcessor(tools: tools)
        var out = Out()
        func absorb(_ events: [Gemma4StreamProcessor.Event]) {
            for e in events {
                switch e {
                case .text(let t): out.text += t
                case .reasoning: break
                case .toolCall(let n, let a): out.calls.append("\(n)|\(a)")
                }
            }
        }
        var i = input.startIndex
        while i < input.endIndex {
            let j = input.index(i, offsetBy: chunk, limitedBy: input.endIndex) ?? input.endIndex
            absorb(p.processChunk(String(input[i..<j])))
            i = j
        }
        absorb(p.finish())
        return out
    }

    /// Runs `input` whole and char-by-char, asserts both agree, returns the result.
    private func parse(_ input: String, tools: [[String: any Sendable]]? = nil,
                       file: StaticString = #filePath, line: UInt = #line) -> Out {
        let whole = run(input, chunk: max(input.count, 1), tools: tools)
        let split = run(input, chunk: 1, tools: tools)
        XCTAssertEqual(whole, split, "chunking changed the result", file: file, line: line)
        return whole
    }

    private func trimmed(_ s: String) -> String { s.trimmingCharacters(in: .whitespacesAndNewlines) }

    // MARK: inline call:

    func test_inline_nameArgsAndEmptyText() {
        let r = parse(#"call:webSearch{"query":"NerdSnipe Inc"}"#)
        XCTAssertEqual(r.names, ["webSearch"])
        XCTAssertEqual(r.args(0)["query"] as? String, "NerdSnipe Inc")
        XCTAssertTrue(trimmed(r.text).isEmpty, "raw call: syntax must not reach the visible text")
    }

    func test_inline_preservesPreamble() {
        let r = parse("I'll search for that.\n\ncall:webSearch{\"query\":\"test\"}")
        XCTAssertEqual(r.names, ["webSearch"])
        XCTAssertEqual(trimmed(r.text), "I'll search for that.")
    }

    func test_inline_nestedBraces() {
        let r = parse(#"call:remember{"body":{"key":"value"}}"#)
        XCTAssertEqual(r.names, ["remember"])
        XCTAssertEqual((r.args(0)["body"] as? [String: Any])?["key"] as? String, "value")
    }

    func test_inline_multipleCalls() {
        let r = parse(#"call:a{x:<|"|>1<|"|>} call:b{y:<|"|>2<|"|>}"#)
        XCTAssertEqual(r.names, ["a", "b"])
        XCTAssertTrue(trimmed(r.text).isEmpty)
    }

    func test_inline_stringArgContainingBraceDoesNotTruncateCall() {
        let r = parse("call:w{body:<|\"|>}{<|\"|>,k:1} tail")
        XCTAssertEqual(r.names, ["w"])
        XCTAssertEqual(r.args(0)["body"] as? String, "}{")
        XCTAssertEqual(trimmed(r.text), "tail")
    }

    func test_inline_stringWithBracesCommasAndQuotes() {
        let r = parse("call:t{code:<|\"|>if (a) { b(\"x\", 1); }<|\"|>,n:2}")
        XCTAssertEqual(r.args(0)["code"] as? String, "if (a) { b(\"x\", 1); }")
        XCTAssertEqual(r.args(0)["n"] as? Int, 2)
    }

    func test_inline_nestedObjectsAndArrays() {
        let r = parse(#"call:t{a:{b:{c:<|"|>deep<|"|>}},list:[1,2,<|"|>three<|"|>]}"#)
        XCTAssertEqual(((r.args(0)["a"] as? [String: Any])?["b"] as? [String: Any])?["c"] as? String, "deep")
        XCTAssertEqual((r.args(0)["list"] as? [Any])?.count, 3)
    }

    func test_inline_backslashesAndNewlinesInStrings() {
        let r = parse("call:t{p:<|\"|>C:\\dir\\file\nline2<|\"|>}")
        XCTAssertEqual(r.args(0)["p"] as? String, "C:\\dir\\file\nline2")
    }

    func test_inline_proseWithCallColonIsUntouched() {
        for prose in ["Give me a call: 555-1234", "I recall:{not a call}", "callback:{x:1}", "Please call: {now}"] {
            let r = parse(prose)
            XCTAssertTrue(r.calls.isEmpty, "\(prose) was treated as a tool call")
            XCTAssertEqual(r.text, prose)
        }
    }

    // MARK: native markers

    func test_nativeToolCallMarkup_isParsedAndHidden() {
        let r = parse("<|tool_call>call:webSearch{\"query\":\"q\"}<tool_call|>")
        XCTAssertEqual(r.names, ["webSearch"])
        XCTAssertTrue(trimmed(r.text).isEmpty)
    }

    func test_channelMarkup_isNotLeakedIntoText() {
        let r = parse("<|channel>thought\nsome reasoning\n<channel|>The answer is 42.")
        XCTAssertEqual(trimmed(r.text), "The answer is 42.")
        XCTAssertTrue(r.calls.isEmpty)
    }

    // MARK: <tool_call> XML

    func test_xml_parsesCallAndHidesBlock() {
        let r = parse(#"Sure. <tool_call>{"name":"lookup","arguments":{"q":"x"}}</tool_call>"#)
        XCTAssertEqual(r.names, ["lookup"])
        XCTAssertEqual(r.args(0)["q"] as? String, "x")
        XCTAssertEqual(trimmed(r.text), "Sure.")
    }

    func test_xml_escapedQuotesAndNestedJSON() {
        let r = parse(#"<tool_call>{"name":"save","arguments":{"text":"He said \"hi\"","meta":{"tags":["a","b"]}}}</tool_call>"#)
        XCTAssertEqual(r.args(0)["text"] as? String, #"He said "hi""#)
        XCTAssertEqual((r.args(0)["meta"] as? [String: Any])?["tags"] as? [String], ["a", "b"])
    }

    func test_xml_multipleCallsKeepOrder() {
        let r = parse(#"<tool_call>{"name":"a","arguments":{}}</tool_call>between<tool_call>{"name":"b","arguments":{}}</tool_call>"#)
        XCTAssertEqual(r.names, ["a", "b"])
        XCTAssertEqual(r.text, "between")
    }

    func test_xml_missingArgumentsBecomesEmptyObject() {
        let r = parse(#"<tool_call>{"name":"ping"}</tool_call>"#)
        XCTAssertEqual(r.calls, ["ping|{}"])
    }

    func test_xml_notJSON_isShownAsText() {
        let src = "<tool_call>not json</tool_call>"
        let r = parse(src)
        XCTAssertTrue(r.calls.isEmpty)
        XCTAssertEqual(r.text, src)
    }

    func test_xml_unterminated_isShownAsText() {
        let src = #"<tool_call>{"name":"a","arguments":{}}"#
        let r = parse(src)
        XCTAssertTrue(r.calls.isEmpty)
        XCTAssertEqual(r.text, src)
    }

    // MARK: ```tool_code

    func test_pythonBlock_keywordArg() {
        let r = parse("```tool_code\nwebSearch(query=\"ottawa weather\")\n```")
        XCTAssertEqual(r.names, ["webSearch"])
        XCTAssertEqual(r.args(0)["query"] as? String, "ottawa weather")
        XCTAssertTrue(trimmed(r.text).isEmpty)
    }

    func test_pythonBlock_preservesPreamble() {
        let r = parse("Let me check.\n```tool_code\nwebSearch(query=\"test\")\n```")
        XCTAssertEqual(r.names, ["webSearch"])
        XCTAssertEqual(trimmed(r.text), "Let me check.")
    }

    func test_pythonBlock_positionalArgUsesFirstRequiredParamFromSchema() {
        let spec: [String: any Sendable] = [
            "type": "function",
            "function": [
                "name": "webSearch",
                "parameters": ["type": "object", "properties": ["q": ["type": "string"]], "required": ["q"]] as [String: any Sendable],
            ] as [String: any Sendable],
        ]
        let r = parse("```tool_code\nwebSearch(\"cats\")\n```", tools: [spec])
        XCTAssertEqual(r.args(0)["q"] as? String, "cats")
    }

    func test_pythonBlock_positionalArgWithoutSchemaFallsBackToQuery() {
        let r = parse("```tool_code\nwebSearch(\"cats\")\n```")
        XCTAssertEqual(r.args(0)["query"] as? String, "cats")
    }

    func test_pythonBlock_noCall_isShownAsText() {
        let src = "```tool_code\njust some words\n```"
        let r = parse(src)
        XCTAssertTrue(r.calls.isEmpty)
        XCTAssertEqual(r.text, src)
    }

    func test_ordinaryCodeFence_isUntouched() {
        let src = "Here:\n```swift\nlet x = f(1)\n```\ndone"
        let r = parse(src)
        XCTAssertTrue(r.calls.isEmpty)
        XCTAssertEqual(r.text, src)
    }

    // MARK: no tool calls

    func test_plainText_passesThrough() {
        XCTAssertEqual(parse("The capital of France is Paris.").text, "The capital of France is Paris.")
        XCTAssertEqual(parse("").text, "")
    }

    func test_plainAnswerWithBracesAndAngleBrackets_isUntouched() {
        let text = "Use `{ key: value }` and <b>bold</b>; a < b | c."
        let r = parse(text)
        XCTAssertTrue(r.calls.isEmpty)
        XCTAssertEqual(r.text, text)
    }
}
