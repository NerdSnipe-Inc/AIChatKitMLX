import XCTest
@testable import AIChatMLX

final class Gemma4StreamProcessorTests: XCTestCase {

    func test_thoughtChannel_emitsReasoningNotText() {
        var p = Gemma4StreamProcessor(tools: nil)
        let events = p.processChunk("<|channel>thought\nNeed memory_recall.\n")
        XCTAssertTrue(events.contains(where: { if case .reasoning = $0 { return true } else { return false } }))
    }

    func test_thoughtThenResponse_splitsChannels() {
        var p = Gemma4StreamProcessor(tools: nil)
        _ = p.processChunk("<|channel>thought\nPlanning.\n")
        let end = p.processChunk("<channel|>The App Store copy says…")
        XCTAssertTrue(end.contains(where: { if case .text(let t) = $0 { return t.contains("App Store") } else { return false } }))
    }
}
