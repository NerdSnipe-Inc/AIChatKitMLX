import AIChatCore
import Foundation
import HuggingFace
import Network
import Observation

// MARK: - Public state

/// Where one on-device model stands. Observable through ``MLXModelManager``.
public enum MLXModelDownloadState: Equatable, Sendable {
    case notDownloaded
    /// `progress == nil` means "connecting" — no confirmed byte progress reported yet. Distinct
    /// from a real `0.0` so a UI can show an indeterminate state instead of a frozen percentage.
    case downloading(progress: Double?)
    /// Every weight file is on disk. Not merely "some bytes exist" — see
    /// ``MLXModelManager/hasCompleteWeights(for:)``.
    case downloaded
    case failed(String)
}

/// How ``MLXModelManager`` recovers from a failed or stalled download.
public struct MLXDownloadPolicy: Sendable, Equatable {
    /// Total attempts per download request (the first try plus retries).
    public var maxAttempts: Int
    /// Seconds to wait before attempt `n + 1` after attempt `n` fails; the last value repeats.
    /// Default: 3s, 10s, 30s, 1m, 2m, 5m — long enough to ride out a Wi-Fi blip or a CDN hiccup.
    public var retryDelays: [TimeInterval]
    /// No progress at all for this long, while online, counts as a stalled connection.
    public var stallTimeout: TimeInterval
    /// Fraction of physical RAM treated as the ceiling for "model weights resident at once".
    public var residentWeightsBudgetFraction: Double

    public init(
        maxAttempts: Int = 6,
        retryDelays: [TimeInterval] = [3, 10, 30, 60, 120, 300],
        stallTimeout: TimeInterval = 120,
        residentWeightsBudgetFraction: Double = 0.75
    ) {
        self.maxAttempts = maxAttempts
        self.retryDelays = retryDelays
        self.stallTimeout = stallTimeout
        self.residentWeightsBudgetFraction = residentWeightsBudgetFraction
    }

    public func delay(afterAttempt attempt: Int) -> TimeInterval {
        guard !retryDelays.isEmpty else { return 0 }
        return retryDelays[min(max(attempt, 1), retryDelays.count) - 1]
    }
}

// MARK: - Manager

/// The single owner of on-device model *files*: what's installed, downloading, failed — and how a
/// download recovers from trouble. Hosts (apps) hold one instance and drive their UI from it; they
/// never talk to the Hugging Face cache or call ``MLXProvider/loadModel(downloadIfNeeded:progressHandler:)``
/// to fetch weights themselves.
///
/// What it guarantees:
/// - **On-disk truth.** A model is ``MLXModelDownloadState/downloaded`` only when every weight file
///   is present (``hasCompleteWeights(for:)``) — an abandoned download that left config/tokenizer
///   files behind is *not* installed.
/// - **Automatic recovery.** A failed download retries with backoff (``MLXDownloadPolicy``),
///   waits for the network instead of burning attempts while offline, and treats a connection that
///   goes silent as a failure rather than waiting on it forever. Partial data is kept between
///   attempts of the same request.
/// - **Resume.** Requests that never finished — app quit, crash, retries exhausted — are persisted
///   and picked back up by ``resumePendingDownloads()``, with no user action.
/// - **Progress.** ``states`` and ``retryNotices`` are `@Observable`; there is no second source.
///
/// Policy that belongs to the host — which models exist, whether this Mac may run one, what to
/// show — is injected via `admission`/`estimatedSizeBytes`, not baked in.
@MainActor
@Observable
public final class MLXModelManager: @unchecked Sendable {

    // MARK: Observable state

    /// Per-model download state. Ids never tracked or requested read as ``MLXModelDownloadState/notDownloaded``.
    public private(set) var states: [String: MLXModelDownloadState] = [:]
    /// Real on-disk bytes per model, measured from the cache directory — never estimated.
    public private(set) var storageUsedBytes: [String: Int64] = [:]
    /// Short human-readable status while a download is recovering ("Connection problem — retrying
    /// in 30s (attempt 2 of 6)"); absent when everything is fine. Kept apart from
    /// ``MLXModelDownloadState`` so a retrying download still reads as `.downloading`.
    public private(set) var retryNotices: [String: String] = [:]

    // MARK: Configuration

    public let policy: MLXDownloadPolicy
    private let defaults: UserDefaults
    private let pendingKey: String
    private let downloadResidency: MLXModelResidency
    private let physicalMemoryBytes: UInt64
    private let admission: @MainActor (String) -> String?
    private let estimatedSizeBytes: @MainActor (String) -> Int64?
    private let isConnected: @MainActor () -> Bool

    private var downloadTasks: [String: Task<Void, Never>] = [:]
    private var lastProgressAt: [String: Date] = [:]
    private let pathMonitor = NetworkPathMonitor()

    /// - Parameters:
    ///   - defaults: Where unfinished-download requests are persisted (injectable for tests).
    ///   - keyPrefix: Namespaces the persisted key so two hosts/instances never collide.
    ///   - policy: Retry/stall/memory thresholds.
    ///   - downloadResidency: Slot the throwaway download load uses. `.auxiliary` (default) keeps a
    ///     download from evicting the model the user is actively using in `.primary`.
    ///   - physicalMemoryBytes: For the resident-weights safety gate.
    ///   - admission: Host policy gate, run before every download starts. Return `nil` to allow,
    ///     or a user-facing reason to refuse (unsupported model, not enough memory…).
    ///   - estimatedSizeBytes: Expected weight size, used only by the residency safety gate.
    ///   - isConnected: Override reachability (tests). Defaults to a built-in `NWPathMonitor`.
    public init(
        defaults: UserDefaults = .standard,
        keyPrefix: String = "aichatmlx",
        policy: MLXDownloadPolicy = MLXDownloadPolicy(),
        downloadResidency: MLXModelResidency = .auxiliary,
        physicalMemoryBytes: UInt64 = ProcessInfo.processInfo.physicalMemory,
        admission: @escaping @MainActor (String) -> String? = { _ in nil },
        estimatedSizeBytes: @escaping @MainActor (String) -> Int64? = { _ in nil },
        isConnected: (@MainActor () -> Bool)? = nil
    ) {
        self.defaults = defaults
        self.pendingKey = "\(keyPrefix).pendingDownloads"
        self.policy = policy
        self.downloadResidency = downloadResidency
        self.physicalMemoryBytes = physicalMemoryBytes
        self.admission = admission
        self.estimatedSizeBytes = estimatedSizeBytes
        let monitor = pathMonitor
        self.isConnected = isConnected ?? { monitor.isConnected }
    }

    // MARK: Tracking & queries

    /// Seeds `states`/`storageUsedBytes` for `modelIds` from what is *actually* on disk. Call at
    /// launch with every model the host knows about, then ``resumePendingDownloads()``.
    public func track(_ modelIds: some Sequence<String>) {
        for id in modelIds { seedState(for: id) }
    }

    public func state(for modelId: String) -> MLXModelDownloadState {
        states[modelId] ?? .notDownloaded
    }

    public func storageUsed(for modelId: String) -> Int64 {
        storageUsedBytes[modelId] ?? 0
    }

    private func seedState(for modelId: String) {
        let size = Self.directorySize(for: modelId)
        // "Has some bytes on disk" is not "is installed": an interrupted download leaves the small
        // config/tokenizer files behind while the multi-GB weights are missing. Treating that as
        // downloaded made a host load the model for inference, which then silently tried to fetch
        // the missing weights in the background — a spinner that ran for ten minutes.
        if size > 0, Self.hasCompleteWeights(for: modelId) {
            states[modelId] = .downloaded
            storageUsedBytes[modelId] = size
        } else if case .downloading = states[modelId] {
            return   // mid-download; leave the live state alone
        } else {
            states[modelId] = .notDownloaded
            storageUsedBytes[modelId] = 0
        }
    }

    // MARK: Download

    /// Downloads `modelId` (or confirms it's already complete), recovering from failures per
    /// ``policy``. Returns when the download has finished, failed for good, or was cancelled.
    ///
    /// Only `.notDownloaded`/`.failed` start a new download; a double tap while `.downloading` or a
    /// call for an already `.downloaded` model is a no-op — so two `MLXProvider`s can never race the
    /// same on-disk directory.
    public func download(modelId: String) async {
        if let refusal = admission(modelId) {
            states[modelId] = .failed(refusal)
            return
        }
        switch state(for: modelId) {
        case .downloaded, .downloading: return
        case .notDownloaded, .failed: break
        }

        states[modelId] = .downloading(progress: nil)
        setPending(true, for: modelId)

        // Owned here so `cancelDownload` has a handle; the outer `await` just re-joins it.
        let task = Task { [weak self] in
            guard let self else { return }
            await self.performDownload(modelId: modelId)
        }
        downloadTasks[modelId] = task
        await task.value
        downloadTasks[modelId] = nil
    }

    /// Picks up any download that was requested but never finished — interrupted by a quit, crash,
    /// or an exhausted retry budget — without the user coming back to tap Install again. Call once
    /// at launch, after ``track(_:)``. Models already complete on disk are just marked done.
    public func resumePendingDownloads() {
        let ids = pendingDownloadIds
        for id in ids where state(for: id) == .downloaded { setPending(false, for: id) }
        let pending = ids.filter { state(for: $0) != .downloaded }
        guard !pending.isEmpty else { return }
        Task { [weak self] in
            for id in pending.sorted() { await self?.download(modelId: id) }
        }
    }

    public func cancelDownload(modelId: String) {
        guard case .downloading = state(for: modelId) else { return }
        downloadTasks[modelId]?.cancel()
        downloadTasks[modelId] = nil
        states[modelId] = .notDownloaded
        storageUsedBytes[modelId] = 0
        setPending(false, for: modelId)
        retryNotices[modelId] = nil
        removeOnDiskDirectory(for: modelId)
        ChatLog.info(.model, "Cancelled download for \(modelId)")
        // Cancellation is cooperative — the in-flight load may still be mid-`loadModelContainer`.
        // Tear the slot down promptly rather than leaving a half-downloaded model's weights in it.
        let residency = downloadResidency
        Task { await MLXProvider.releaseResidency(residency) }
    }

    public func delete(modelId: String) {
        removeOnDiskDirectory(for: modelId)
        states[modelId] = .notDownloaded
        storageUsedBytes[modelId] = 0
        setPending(false, for: modelId)
        retryNotices[modelId] = nil
        ChatLog.info(.model, "Deleted model \(modelId)")
    }

    // MARK: Download engine

    private func performDownload(modelId: String) async {
        // Start from an empty directory *once per request*: the Hub cache would otherwise silently
        // resume from whatever partial/corrupt blobs an earlier interrupted download left behind —
        // root cause of a real MLX weight-shape-mismatch load failure. Deliberately not repeated
        // between the automatic retries below (same request): throwing away gigabytes on every
        // dropped connection would make recovery useless on a flaky network.
        removeOnDiskDirectory(for: modelId)

        var attempt = 1
        while true {
            await releaseResidentModelIfCoexistenceUnsafe(for: modelId)
            do {
                try await loadWithStallWatchdog(modelId: modelId)
                // `cancelDownload` may have reset state while this was in flight — don't clobber
                // that with a stale "downloaded".
                guard !Task.isCancelled else {
                    await MLXProvider.releaseResidency(downloadResidency)
                    return
                }
                states[modelId] = .downloaded
                storageUsedBytes[modelId] = Self.directorySize(for: modelId)
                setPending(false, for: modelId)
                retryNotices[modelId] = nil
                lastProgressAt[modelId] = nil
                ChatLog.info(.model, "Downloaded model \(modelId)")
                // The load was only a vehicle for the download; free the slot straight away.
                await MLXProvider.releaseResidency(downloadResidency)
                return
            } catch {
                await MLXProvider.releaseResidency(downloadResidency)
                guard !Task.isCancelled else { return }

                ChatLog.error(.model, "Download attempt \(attempt)/\(policy.maxAttempts) failed for \(modelId)", underlying: error)

                guard Self.isRetryable(error) else {
                    // Permanent (disk full, access denied, model gone): retrying can't help, and it
                    // must not be resumed at every launch either.
                    states[modelId] = .failed(Self.message(for: error))
                    retryNotices[modelId] = nil
                    setPending(false, for: modelId)
                    return
                }
                guard attempt < policy.maxAttempts else {
                    // Out of automatic attempts. Stays pending, so the next launch tries again.
                    states[modelId] = .failed("\(Self.message(for: error)) It will try again the next time the app launches.")
                    retryNotices[modelId] = nil
                    return
                }

                let delay = policy.delay(afterAttempt: attempt)
                retryNotices[modelId] = "Connection problem — retrying in \(Int(delay))s (attempt \(attempt + 1) of \(policy.maxAttempts))"
                states[modelId] = .downloading(progress: nil)
                // Don't burn attempts while the Mac is simply offline.
                await waitForConnectivity()
                try? await Task.sleep(for: .seconds(delay))
                guard !Task.isCancelled else { return }
                retryNotices[modelId] = "Retrying (attempt \(attempt + 1) of \(policy.maxAttempts))…"
                attempt += 1
            }
        }
    }

    /// One attempt, raced against a watchdog. A connection that goes silent never throws on its
    /// own — before this, a stalled transfer just sat there (and anything waiting on it with it).
    private func loadWithStallWatchdog(modelId: String) async throws {
        lastProgressAt[modelId] = .now
        let provider = MLXProvider(modelId: modelId, residency: downloadResidency)
        let stallTimeout = policy.stallTimeout
        // Built on the main actor, outside the child tasks: a `[weak self]` capture list inside a
        // `@Sendable` closure makes `self` a captured `var` (a Swift 6 error).
        let onProgress: @Sendable (Progress) -> Void = { [weak self] progress in
            let fraction = progress.fractionCompleted
            Task { @MainActor [weak self] in
                self?.lastProgressAt[modelId] = .now
                self?.applyProgress(modelId: modelId, fraction: fraction)
            }
        }
        try await withThrowingTaskGroup(of: Void.self) { group in
            group.addTask {
                try await provider.loadModel(downloadIfNeeded: true, progressHandler: onProgress)
            }
            group.addTask { [weak self] in
                while true {
                    try await Task.sleep(for: .seconds(10))
                    guard let self else { return }
                    let last = await self.lastProgressAt[modelId] ?? .now
                    if Date.now.timeIntervalSince(last) > stallTimeout, await self.isConnected() {
                        throw StalledError()
                    }
                }
            }
            // First to finish wins: completion, failure, or a declared stall. Cancel the other.
            _ = try await group.next()
            group.cancelAll()
        }
    }

    private func applyProgress(modelId: String, fraction: Double) {
        guard case .downloading = states[modelId] else { return }
        guard fraction > 0 else { return }
        states[modelId] = .downloading(progress: fraction)
        retryNotices[modelId] = nil   // bytes are flowing again — recovered
    }

    /// Returns once the network is reachable (or after a generous cap, so a wrong "offline"
    /// reading can't wedge a download forever).
    private func waitForConnectivity() async {
        var waited = 0
        while !isConnected(), waited < 900, !Task.isCancelled {
            try? await Task.sleep(for: .seconds(2))
            waited += 2
        }
    }

    // MARK: Residency safety gate

    /// Pure and `nonisolated` so it is unit-testable without a loaded model.
    nonisolated public static func canCoexistWithResidentModel(
        residentBytes: Int64,
        incomingBytes: Int64,
        physicalMemoryBytes: UInt64,
        budgetFraction: Double = 0.75
    ) -> Bool {
        guard residentBytes > 0, incomingBytes > 0 else { return true }
        let budget = Int64(Double(physicalMemoryBytes) * budgetFraction)
        return residentBytes + incomingBytes <= budget
    }

    /// Downloads load into `.auxiliary`, so by default the user's live `.primary` model isn't
    /// evicted. But that turns a peak of `max(existing, downloading)` into `existing +
    /// downloading` — on a Mac holding a big model, both at once is the dual-large-model pattern
    /// behind an OS RAM-pressure kill. So release `.primary` only when both-at-once is genuinely
    /// unsafe; it reloads lazily from cache on next use.
    private func releaseResidentModelIfCoexistenceUnsafe(for modelId: String) async {
        guard let incoming = estimatedSizeBytes(modelId),
              let resident = await MLXProvider.residentWeightBytes(in: .primary)
        else { return }
        guard !Self.canCoexistWithResidentModel(
            residentBytes: Int64(resident), incomingBytes: incoming,
            physicalMemoryBytes: physicalMemoryBytes, budgetFraction: policy.residentWeightsBudgetFraction
        ) else { return }
        ChatLog.info(.model, "Releasing .primary before downloading \(modelId): \(resident) resident + \(incoming) incoming bytes exceeds the safe working set")
        await MLXProvider.releaseResidency(.primary)
    }

    // MARK: Pending downloads

    private var pendingDownloadIds: Set<String> {
        Set(defaults.stringArray(forKey: pendingKey) ?? [])
    }

    private func setPending(_ pending: Bool, for modelId: String) {
        var ids = pendingDownloadIds
        if pending { ids.insert(modelId) } else { ids.remove(modelId) }
        defaults.set(Array(ids).sorted(), forKey: pendingKey)
    }

    // MARK: - On-disk truth

    /// Resolves `modelId`'s repo directory through the same `HubCache` the download path writes
    /// through — never a hand-guessed path.
    nonisolated public static func repoDirectory(for modelId: String) -> URL {
        guard let repoId = HuggingFace.Repo.ID(rawValue: modelId) else {
            return HubCache.default.cacheDirectory.appendingPathComponent(modelId)
        }
        return HubCache.default.repoDirectory(repo: repoId, kind: .model)
    }

    /// Whether some snapshot of `modelId` in the Hub cache holds every weight file it needs.
    nonisolated public static func hasCompleteWeights(for modelId: String) -> Bool {
        let snapshots = repoDirectory(for: modelId).appendingPathComponent("snapshots")
        guard let revisions = try? FileManager.default.contentsOfDirectory(
            at: snapshots, includingPropertiesForKeys: nil, options: [.skipsHiddenFiles]
        ) else { return false }
        return revisions.contains(where: snapshotHasCompleteWeights)
    }

    /// Pure check over one snapshot directory (unit-testable with a temp dir). A sharded model ships
    /// a `model.safetensors.index.json` whose `weight_map` names every shard — each must exist. A
    /// single-file model has no index, so it needs at least one `.safetensors` file. `fileExists`
    /// follows the snapshot's symlinks into `blobs/`, so a link to an unfinished or removed blob
    /// correctly counts as missing.
    nonisolated public static func snapshotHasCompleteWeights(_ snapshot: URL) -> Bool {
        let fm = FileManager.default
        let indexURL = snapshot.appendingPathComponent("model.safetensors.index.json")
        if let data = try? Data(contentsOf: indexURL),
           let object = try? JSONSerialization.jsonObject(with: data) as? [String: Any],
           let weightMap = object["weight_map"] as? [String: String], !weightMap.isEmpty {
            return Set(weightMap.values).allSatisfy {
                fm.fileExists(atPath: snapshot.appendingPathComponent($0).path)
            }
        }
        let files = (try? fm.contentsOfDirectory(atPath: snapshot.path)) ?? []
        return files.contains {
            $0.hasSuffix(".safetensors") && fm.fileExists(atPath: snapshot.appendingPathComponent($0).path)
        }
    }

    /// Sums the size of the repo directory's real (non-symlink) files — `HubCache` keeps content
    /// only in `blobs/`; `snapshots/` entries are symlinks into it, so this avoids double-counting.
    nonisolated public static func directorySize(for modelId: String) -> Int64 {
        guard let enumerator = FileManager.default.enumerator(
            at: repoDirectory(for: modelId),
            includingPropertiesForKeys: [.fileSizeKey, .isRegularFileKey],
            options: [.skipsHiddenFiles]
        ) else { return 0 }
        var total: Int64 = 0
        for case let url as URL in enumerator {
            guard let values = try? url.resourceValues(forKeys: [.fileSizeKey, .isRegularFileKey]),
                  values.isRegularFile == true, let size = values.fileSize else { continue }
            total += Int64(size)
        }
        return total
    }

    private func removeOnDiskDirectory(for modelId: String) {
        let directory = Self.repoDirectory(for: modelId)
        do {
            if FileManager.default.fileExists(atPath: directory.path) {
                try FileManager.default.removeItem(at: directory)
            }
        } catch {
            ChatLog.error(.model, "Failed to remove on-disk directory for \(modelId)", underlying: error)
        }
    }

    // MARK: - Recovery classification (pure, unit-tested)

    /// A download that stopped making progress.
    public struct StalledError: LocalizedError, Sendable {
        public init() {}
        public var errorDescription: String? { "The download stopped making progress." }
    }

    /// Whether trying again could possibly help. Network trouble, timeouts, server hiccups and
    /// stalls: yes. Cancellation, a full disk, permission problems, a model that's gone or gated,
    /// or an unsupported architecture: no — retrying those only burns time and bandwidth.
    nonisolated public static func isRetryable(_ error: Error) -> Bool {
        if error is CancellationError { return false }
        if error is StalledError { return true }
        if let chat = error as? ChatError {
            switch chat {
            case .networkError(let underlying), .modelDownloadFailed(_, let underlying):
                return isRetryable(underlying)
            case .serverError(let status, _):
                return status >= 500 || status == 408 || status == 429
            case .cancelled, .modelNotFound, .unsupportedModel, .outOfMemory, .invalidConfiguration,
                 .modelLoadFailed, .templateError, .decodingError, .streamError, .toolCallParseFailed,
                 .generationFailed:
                return false
            }
        }
        let ns = error as NSError
        switch ns.domain {
        case NSURLErrorDomain:
            let permanent: Set<Int> = [
                NSURLErrorCancelled, NSURLErrorBadURL, NSURLErrorUnsupportedURL,
                NSURLErrorUserAuthenticationRequired, NSURLErrorNoPermissionsToReadFile,
                NSURLErrorFileDoesNotExist,
            ]
            return !permanent.contains(ns.code)
        case NSCocoaErrorDomain:
            let permanent: Set<Int> = [NSFileWriteOutOfSpaceError, NSFileWriteNoPermissionError, NSFileWriteVolumeReadOnlyError]
            return !permanent.contains(ns.code)
        default:
            let text = error.localizedDescription.lowercased()
            return !["401", "403", "404", "gated", "not found", "unauthorized", "forbidden"].contains { text.contains($0) }
        }
    }

    nonisolated static func message(for error: Error) -> String {
        (error as? LocalizedError)?.errorDescription ?? error.localizedDescription
    }
}

// MARK: - Reachability

/// Tiny `NWPathMonitor` wrapper, so the package needs nothing from the host to know whether the
/// network is up.
private final class NetworkPathMonitor: @unchecked Sendable {
    private let monitor = NWPathMonitor()
    private let lock = NSLock()
    private var connected = true

    init() {
        monitor.pathUpdateHandler = { [weak self] path in
            guard let self else { return }
            self.lock.lock(); self.connected = path.status == .satisfied; self.lock.unlock()
        }
        monitor.start(queue: DispatchQueue(label: "aichatmlx.model-manager.path"))
    }

    deinit { monitor.cancel() }

    var isConnected: Bool {
        lock.lock(); defer { lock.unlock() }
        return connected
    }
}
