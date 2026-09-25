import XCTest
import AIChatCore
@testable import AIChatMLX

/// Download control lives in the package, so its behaviour is tested here — no network, no model,
/// no Metal device needed.
@MainActor
final class MLXModelManagerTests: XCTestCase {

    private func makeDefaults(_ name: String = #function) -> UserDefaults {
        let d = UserDefaults(suiteName: "mlxmm.\(name)")!
        d.removePersistentDomain(forName: "mlxmm.\(name)")
        return d
    }

    private func scratchId() -> String { "mlx-community/test-scratch-model-\(UUID().uuidString)" }

    private func makeSnapshot(files: [String: String]) throws -> URL {
        let dir = FileManager.default.temporaryDirectory.appendingPathComponent("snapshot-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        for (name, contents) in files { try Data(contents.utf8).write(to: dir.appendingPathComponent(name)) }
        return dir
    }

    // MARK: - On-disk truth

    /// Regression: config/tokenizer files without the weights (an abandoned download) must not
    /// count as an installed model — it made a "Drafting Reply" spinner run ten minutes fetching ~5 GB.
    func test_snapshot_withIndexButMissingShard_isIncomplete() throws {
        let dir = try makeSnapshot(files: [
            "config.json": "{}",
            "model.safetensors.index.json": #"{"weight_map":{"a":"model.safetensors"}}"#,
        ])
        defer { try? FileManager.default.removeItem(at: dir) }
        XCTAssertFalse(MLXModelManager.snapshotHasCompleteWeights(dir))
    }

    func test_snapshot_withAllShardsPresent_isComplete() throws {
        let dir = try makeSnapshot(files: [
            "model.safetensors.index.json": #"{"weight_map":{"a":"m-1.safetensors","b":"m-2.safetensors"}}"#,
            "m-1.safetensors": "x", "m-2.safetensors": "x",
        ])
        defer { try? FileManager.default.removeItem(at: dir) }
        XCTAssertTrue(MLXModelManager.snapshotHasCompleteWeights(dir))
    }

    func test_snapshot_withOneShardMissing_isIncomplete() throws {
        let dir = try makeSnapshot(files: [
            "model.safetensors.index.json": #"{"weight_map":{"a":"m-1.safetensors","b":"m-2.safetensors"}}"#,
            "m-1.safetensors": "x",
        ])
        defer { try? FileManager.default.removeItem(at: dir) }
        XCTAssertFalse(MLXModelManager.snapshotHasCompleteWeights(dir))
    }

    func test_snapshot_singleFileModelWithoutIndex() throws {
        let complete = try makeSnapshot(files: ["config.json": "{}", "model.safetensors": "x"])
        let configOnly = try makeSnapshot(files: ["config.json": "{}", "tokenizer.json": "{}"])
        defer { try? FileManager.default.removeItem(at: complete); try? FileManager.default.removeItem(at: configOnly) }
        XCTAssertTrue(MLXModelManager.snapshotHasCompleteWeights(complete))
        XCTAssertFalse(MLXModelManager.snapshotHasCompleteWeights(configOnly))
    }

    // MARK: - Recovery policy

    func test_policy_delaysBackOffAndCap() {
        let p = MLXDownloadPolicy()
        let delays = (1...8).map { p.delay(afterAttempt: $0) }
        XCTAssertEqual(Array(delays.prefix(6)), [3, 10, 30, 60, 120, 300])
        XCTAssertEqual(delays[6], 300, "past the schedule it stays at the cap")
        XCTAssertEqual(p.delay(afterAttempt: 0), 3, "never crashes on a bad index")
        XCTAssertEqual(MLXDownloadPolicy(retryDelays: []).delay(afterAttempt: 1), 0)
    }

    func test_networkAndStallErrors_areRetryable() {
        for code in [NSURLErrorNotConnectedToInternet, NSURLErrorNetworkConnectionLost, NSURLErrorTimedOut,
                     NSURLErrorCannotConnectToHost, NSURLErrorDNSLookupFailed, NSURLErrorSecureConnectionFailed] {
            XCTAssertTrue(MLXModelManager.isRetryable(URLError(URLError.Code(rawValue: code))), "code \(code)")
        }
        XCTAssertTrue(MLXModelManager.isRetryable(MLXModelManager.StalledError()))
    }

    /// Package downloads surface as `ChatError` — the classifier must see through the wrapper.
    func test_wrappedChatErrors_areClassifiedByTheirCause() {
        let offline = URLError(.notConnectedToInternet)
        XCTAssertTrue(MLXModelManager.isRetryable(ChatError.modelDownloadFailed(modelId: "m", underlying: offline)))
        XCTAssertTrue(MLXModelManager.isRetryable(ChatError.networkError(offline)))
        XCTAssertTrue(MLXModelManager.isRetryable(ChatError.serverError(statusCode: 503, message: "")))
        XCTAssertTrue(MLXModelManager.isRetryable(ChatError.serverError(statusCode: 429, message: "")))
        XCTAssertFalse(MLXModelManager.isRetryable(ChatError.serverError(statusCode: 404, message: "")))
        XCTAssertFalse(MLXModelManager.isRetryable(ChatError.modelNotFound(modelId: "m")))
        XCTAssertFalse(MLXModelManager.isRetryable(ChatError.cancelled))
        XCTAssertFalse(MLXModelManager.isRetryable(
            ChatError.modelDownloadFailed(modelId: "m", underlying: NSError(domain: NSCocoaErrorDomain, code: NSFileWriteOutOfSpaceError))))
    }

    func test_permanentErrors_areNotRetried() {
        XCTAssertFalse(MLXModelManager.isRetryable(CancellationError()))
        XCTAssertFalse(MLXModelManager.isRetryable(URLError(.cancelled)))
        XCTAssertFalse(MLXModelManager.isRetryable(URLError(.fileDoesNotExist)))
        XCTAssertFalse(MLXModelManager.isRetryable(NSError(domain: NSCocoaErrorDomain, code: NSFileWriteOutOfSpaceError)))
        XCTAssertFalse(MLXModelManager.isRetryable(
            NSError(domain: "HF", code: 1, userInfo: [NSLocalizedDescriptionKey: "HTTP 404: repository not found"])))
    }

    // MARK: - Residency gate

    private static let gib = UInt64(1_073_741_824)

    func test_coexistence_deniedWhenCombinedWeightsExceedBudget() {
        XCTAssertFalse(MLXModelManager.canCoexistWithResidentModel(
            residentBytes: 5_200_000_000, incomingBytes: 18_400_000_000, physicalMemoryBytes: 16 * Self.gib))
        XCTAssertFalse(MLXModelManager.canCoexistWithResidentModel(
            residentBytes: 18_400_000_000, incomingBytes: 18_400_000_000, physicalMemoryBytes: 32 * Self.gib))
    }

    func test_coexistence_allowedWhenTheyFitOrNothingResident() {
        XCTAssertTrue(MLXModelManager.canCoexistWithResidentModel(
            residentBytes: 5_200_000_000, incomingBytes: 5_200_000_000, physicalMemoryBytes: 32 * Self.gib))
        XCTAssertTrue(MLXModelManager.canCoexistWithResidentModel(
            residentBytes: 0, incomingBytes: 18_400_000_000, physicalMemoryBytes: 16 * Self.gib))
    }

    // MARK: - Manager behaviour

    func test_unknownModel_readsNotDownloaded() {
        let m = MLXModelManager(defaults: makeDefaults())
        XCTAssertEqual(m.state(for: "mlx-community/never-heard-of-this-model"), .notDownloaded)
        XCTAssertEqual(m.storageUsed(for: "mlx-community/never-heard-of-this-model"), 0)
    }

    func test_track_seedsNotDownloadedForModelsWithNothingOnDisk() {
        let id = scratchId()
        let m = MLXModelManager(defaults: makeDefaults())
        m.track([id])
        XCTAssertEqual(m.state(for: id), .notDownloaded)
    }

    /// The host's policy gate runs before anything starts — a refused model never touches the network.
    func test_download_refusedByAdmission_failsWithHostsReason() async {
        let id = scratchId()
        let m = MLXModelManager(defaults: makeDefaults(), admission: { _ in "Needs a Mac with 32 GB of memory." })
        await m.download(modelId: id)
        XCTAssertEqual(m.state(for: id), .failed("Needs a Mac with 32 GB of memory."))
    }

    func test_delete_removesOnDiskDirectoryAndClearsState() throws {
        let id = scratchId()
        let dir = MLXModelManager.repoDirectory(for: id)
        try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        try Data("not a real model".utf8).write(to: dir.appendingPathComponent("fake-weights.bin"))

        let m = MLXModelManager(defaults: makeDefaults())
        m.delete(modelId: id)

        XCTAssertFalse(FileManager.default.fileExists(atPath: dir.path))
        XCTAssertEqual(m.state(for: id), .notDownloaded)
    }

    // MARK: - No hidden downloads on inference paths

    /// The regression that started this: loading for inference must not silently start a
    /// multi-GB download. With `downloadIfNeeded: false` an uninstalled model fails fast — no network.
    func test_loadModel_withoutDownload_throwsModelNotFoundForUninstalledModel() async {
        let id = scratchId()
        let provider = MLXProvider(modelId: id)
        do {
            try await provider.loadModel(downloadIfNeeded: false)
            XCTFail("expected modelNotFound")
        } catch let ChatError.modelNotFound(modelId) {
            XCTAssertEqual(modelId, id)
        } catch {
            XCTFail("unexpected error: \(error)")
        }
    }
}
