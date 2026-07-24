import XCTest
@testable import AIChatMLX

final class MLXProviderAdapterURLTests: XCTestCase {
    func test_initWithAdapterURL_storesURL() {
        let url = URL(filePath: "/tmp/fake-adapter")
        let provider = MLXProvider(modelId: "mlx-community/gemma-4-e4b-it-4bit", adapterDirectoryURL: url)
        XCTAssertEqual(provider.adapterDirectoryURL, url)
    }

    func test_initWithoutAdapterURL_isNil() {
        let provider = MLXProvider(modelId: "mlx-community/gemma-4-e4b-it-4bit")
        XCTAssertNil(provider.adapterDirectoryURL)
    }

    func test_initWithModelPath_acceptsAdapterURL() {
        let modelURL = URL(filePath: "/tmp/fake-model")
        let adapterURL = URL(filePath: "/tmp/fake-adapter")
        let provider = MLXProvider(modelPath: modelURL, adapterDirectoryURL: adapterURL)
        XCTAssertEqual(provider.adapterDirectoryURL, adapterURL)
    }
}
