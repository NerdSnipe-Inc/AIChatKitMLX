import XCTest
@testable import AIChatMLX

/// Slips the fine-tuned Gemma 4 model makes in a tool call (observed in training run5: a stray string
/// delimiter written after a bare number, `limit:5<|"|>`). A call with such a slip must still run,
/// with the intended arguments, rather than being dropped or given a garbage value.
final class GemmaCallSyntaxToleranceTests: XCTestCase {

    private func args(_ body: String) -> [String: Any]? {
        guard let call = GemmaCallSyntax.parseCalls(in: body)?.first else { return nil }
        return (try? JSONSerialization.jsonObject(with: Data(call.argumentsJSON.utf8))) as? [String: Any]
    }

    func test_wellFormedCall_isUnchanged() {
        let a = args(#"call:search_contacts{query:<|"|>john<|"|>,limit:5}"#)
        XCTAssertEqual(a?["query"] as? String, "john")
        XCTAssertEqual(a?["limit"] as? Int, 5)
    }

    func test_strayDelimiterAfterBareNumber_atEnd() {
        let a = args(#"call:search_contacts{query:<|"|>john<|"|>,limit:5<|"|>}"#)
        XCTAssertEqual(a?["query"] as? String, "john")
        XCTAssertEqual(a?["limit"] as? Int, 5)
    }

    func test_strayDelimiterAfterBareNumber_beforeAnotherArgument() {
        let a = args(#"call:search_deals{limit:5<|"|>,status:<|"|>open<|"|>}"#)
        XCTAssertEqual(a?["limit"] as? Int, 5)
        XCTAssertEqual(a?["status"] as? String, "open")
    }

    func test_strayDelimiterAfterBoolean() {
        let a = args(#"call:create_task{title:<|"|>Call back<|"|>,urgent:true<|"|>}"#)
        XCTAssertEqual(a?["title"] as? String, "Call back")
        XCTAssertEqual(a?["urgent"] as? Bool, true)
    }

    func test_properStringContainingDelimiterLikeText_isStillOpaque() {
        let a = args(#"call:add_note{body:<|"|>use {braces}, commas, and 5 things<|"|>}"#)
        XCTAssertEqual(a?["body"] as? String, "use {braces}, commas, and 5 things")
    }

    func test_unrecoverableCall_isNotInvented() {
        XCTAssertNil(GemmaCallSyntax.parseCalls(in: "call:search_contacts{query:}"))
    }
}
