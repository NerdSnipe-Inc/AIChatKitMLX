import XCTest
import AIChatCore
@testable import AIChatMLX

/// Fault-injection tests for how MLX/Hub-shaped errors classify. No model needed.
final class MLXErrorMappingTests: XCTestCase {

    /// Mimics an untyped Hub error whose only signal is its description.
    private struct HubStub: Error, CustomStringConvertible { let description: String }

    func test_hubRepoNotFound() {
        let e = ChatError.classify(HubStub(description: "HubApi.httpStatusCode(404)"), modelId: "mlx-community/does-not-exist-xyz", phase: .load)
        guard case .modelNotFound(let id) = e else { return XCTFail("\(e)") }
        XCTAssertEqual(id, "mlx-community/does-not-exist-xyz")
        XCTAssertTrue(e.errorDescription!.contains("does-not-exist-xyz"))
    }

    func test_noModelFactory_isUnsupported() {
        let e = ChatError.classify(HubStub(description: "noModelFactoryAvailable"), modelId: "x/y", phase: .load)
        guard case .unsupportedModel = e else { return XCTFail("\(e)") }
    }

    func test_offlineDuringLoad() {
        let e = ChatError.classify(URLError(.notConnectedToInternet), modelId: "x/y", phase: .load)
        guard case .modelDownloadFailed = e else { return XCTFail("\(e)") }
        XCTAssertNotNil(e.recoverySuggestion)
    }

    func test_generationFailureDuringStream() {
        let e = ChatError.classify(HubStub(description: "shape mismatch"), modelId: "x/y", phase: .generate)
        guard case .generationFailed = e else { return XCTFail("\(e)") }
    }

    func test_unloadedProvider_streamCancellationIsNotFailure() async {
        let provider = MLXProvider(modelId: "mlx-community/does-not-exist-xyz")
        let task = Task { () -> Error? in
            do {
                for try await _ in provider.stream(messages: [ChatMessage(role: .user, content: "hi")], model: "m", options: ChatRequestOptions()) {}
                return nil
            } catch { return error }
        }
        task.cancel()
        let err = await task.value
        // Either cancelled, or (if the cancel raced after load began) a specific load error — never opaque.
        if let err { XCTAssertTrue(err is ChatError, "\(err)") }
    }
}
