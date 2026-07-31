import XCTest
@testable import AIChatMLX

final class FunctionGemmaStreamProcessorTests: XCTestCase {
    func testParsesCallEnvelopeAndEscapeStrings() throws {
        var processor = FunctionGemmaStreamProcessor()
        XCTAssertTrue(processor.processChunk(
            "<start_function_call>call:add_note{contactId:<escape>ct_123"
        ).isEmpty)
        XCTAssertTrue(processor.processChunk(
            "<escape>,body:<escape>Asked for email updates<escape>}"
                + "<end_function_call>"
        ).isEmpty)

        let events = processor.finish()
        guard case .toolCall(let name, let argumentsJSON) = try XCTUnwrap(events.first)
        else {
            return XCTFail("Expected one parsed tool call")
        }
        XCTAssertEqual(name, "add_note")

        let data = try XCTUnwrap(argumentsJSON.data(using: .utf8))
        let arguments = try XCTUnwrap(
            JSONSerialization.jsonObject(with: data) as? [String: String]
        )
        XCTAssertEqual(arguments["contactId"], "ct_123")
        XCTAssertEqual(arguments["body"], "Asked for email updates")
    }

    func testNoCallTextDoesNotLeakFunctionMarkers() {
        var processor = FunctionGemmaStreamProcessor()
        _ = processor.processChunk("No tool is needed.")
        XCTAssertEqual(processor.finish(), [.text("No tool is needed.")])
    }

    func testMalformedEnvelopeIsTextInsteadOfExecuting() {
        var processor = FunctionGemmaStreamProcessor()
        _ = processor.processChunk(
            "<start_function_call>call:add_note{contactId:<escape>ct_123"
        )
        XCTAssertEqual(
            processor.finish(),
            [.text("<start_function_call>call:add_note{contactId:<escape>ct_123")]
        )
    }
}
