import XCTest
@testable import AIChatMLX

// MLX inference requires Apple Silicon hardware and a downloaded model.
// These tests verify the provider's configuration surface without loading a model.
final class MLXProviderTests: XCTestCase {

    func test_defaultModelId() {
        XCTAssertEqual(MLXProvider.defaultModelId, "mlx-community/gemma-4-e4b-it-4bit")
    }

    func test_init_hubId_identity() async {
        let provider = MLXProvider()
        let id   = await provider.id
        let name = await provider.name
        XCTAssertEqual(id,   "mlx")
        XCTAssertEqual(name, "MLX")
    }
}
