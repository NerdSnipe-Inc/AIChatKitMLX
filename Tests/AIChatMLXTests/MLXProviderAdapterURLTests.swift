import XCTest
@testable import AIChatMLX

final class MLXProviderAdapterURLTests: XCTestCase {
    func test_thinkingDefaultsOff() {
        XCTAssertFalse(MLXProvider(modelId: "mlx-community/gemma-4-e4b-it-4bit").enableThinking)
    }

    func test_thinkingCanBeExplicitlyEnabled() {
        XCTAssertTrue(MLXProvider(
            modelId: "mlx-community/gemma-4-e4b-it-4bit",
            enableThinking: true
        ).enableThinking)
    }

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

final class MLXProviderAdapterStatusTests: XCTestCase {
    func test_noAdapter_isNotConfigured() async {
        let provider = MLXProvider(modelId: "mlx-community/gemma-4-e4b-it-4bit")
        let status = await provider.adapterStatus
        XCTAssertEqual(status, .notConfigured)
    }

    func test_adapterURL_startsPending() async {
        let provider = MLXProvider(
            modelId: "mlx-community/gemma-4-e4b-it-4bit",
            adapterDirectoryURL: URL(filePath: "/tmp/fake-adapter")
        )
        let status = await provider.adapterStatus
        XCTAssertEqual(status, .pending)
    }

    func test_retryWithoutAdapter_isANoOp() async {
        let provider = MLXProvider(modelId: "mlx-community/gemma-4-e4b-it-4bit")
        await provider.retryAdapter()
        let status = await provider.adapterStatus
        XCTAssertEqual(status, .notConfigured)
    }

    func test_retryBeforeModelIsLoaded_staysPending() async {
        let provider = MLXProvider(
            modelPath: URL(filePath: "/tmp/definitely-not-a-model"),
            adapterDirectoryURL: URL(filePath: "/tmp/fake-adapter")
        )
        await provider.retryAdapter()
        let status = await provider.adapterStatus
        XCTAssertEqual(status, .pending)
    }
}
