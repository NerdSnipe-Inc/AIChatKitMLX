// swift-tools-version: 5.10

import Foundation
import PackageDescription

// Monorepo: sibling packages when present next to this repo (Alric layout).
// SPI / standalone clone: GitHub URLs when SPI_PROCESSING is set or sibling missing.

private let packageDirectory = URL(fileURLWithPath: #filePath).deletingLastPathComponent()

private func siblingOrRemote(
    siblingRelativePath: String,
    url: String,
    from version: Version
) -> Package.Dependency {
    let siblingManifest = packageDirectory
        .appendingPathComponent(siblingRelativePath)
        .standardized
        .appendingPathComponent("Package.swift")

    // SwiftPM checks out every remote dependency as a sibling folder under one shared
    // `checkouts/` directory (Xcode: `SourcePackages/checkouts/`, plain `swift build`:
    // `.build/checkouts/`) — when THIS package is itself being resolved remotely, that
    // coincidentally makes `../AIChatKit` (or similar) resolve to a real sibling checkout that
    // happens to exist for an unrelated reason, not a genuine local monorepo dev setup. Without
    // this check, a consumer resolving this package remotely while ALSO depending on the sibling
    // package directly hits a SwiftPM "conflicting identity" resolution failure — the sibling
    // gets referenced once via this package's local `path:` and once via the consumer's own
    // `url:`, and SwiftPM refuses to treat those as the same package.
    let isSwiftPMCheckout = packageDirectory.path.contains("/checkouts/")
    let forceRemote = isSwiftPMCheckout
        || ProcessInfo.processInfo.environment["SPI_PROCESSING"] != nil
        || ProcessInfo.processInfo.environment["FORCE_REMOTE_PACKAGES"] != nil

    if !forceRemote, FileManager.default.fileExists(atPath: siblingManifest.path) {
        return .package(path: siblingRelativePath)
    }
    return .package(url: url, from: version)
}

let package = Package(
    name: "AIChatKitMLX",
    platforms: [.macOS(.v14), .iOS(.v17)],
    products: [
        .library(name: "AIChatMLX", targets: ["AIChatMLX"]),
    ],
    dependencies: [
        siblingOrRemote(
            siblingRelativePath: "../AIChatKit",
            url: "https://github.com/NerdSnipe-Inc/AIChatKit.git",
            from: "1.0.0"
        ),
        siblingOrRemote(
            siblingRelativePath: "../mlx-swift-lm",
            url: "https://github.com/ml-explore/mlx-swift-lm.git",
            from: "3.0.0"
        ),
        // Same constraint mlx-swift-lm declares — MLX is needed directly for the wired-memory
        // ticket types (`WiredMemoryTicket`, `WiredMemoryManager`) used to bound model residency.
        .package(url: "https://github.com/ml-explore/mlx-swift", .upToNextMinor(from: "0.31.3")),
        .package(url: "https://github.com/huggingface/swift-huggingface.git", .upToNextMajor(from: "0.9.0")),
        .package(url: "https://github.com/huggingface/swift-transformers.git", .upToNextMajor(from: "1.2.1")),
        // `AIChatCore.ChatRequestOptions.ToolDefinition` publicly exposes a `JSONSchema.JSONSchema`
        // value (see AIChatKit's own Package.swift), and `MLXProvider.toToolSpecs` touches that
        // type — so `AIChatMLX`'s own compiled object needs `JSONSchema`'s metadata/witness
        // tables at link time even though no file in this target ever writes `import JSONSchema`
        // itself. Plain `swift build`/`swift test` never surfaced this (SwiftPM CLI links the
        // whole dependency graph into one binary, which happens to satisfy the symbol
        // regardless), but Xcode's native package integration builds every product as its own
        // separate dynamic framework and only auto-links a target's *declared* dependencies —
        // without this, any Xcode-project consumer hits "Undefined symbols ... type metadata
        // accessor for JSONSchema.JSONSchema" linking `AIChatMLX`. Version pinned to match
        // AIChatKit's own requirement on this package exactly.
        .package(url: "https://github.com/kevinhermawan/swift-json-schema.git", .upToNextMajor(from: "2.0.1")),
    ],
    targets: [
        .target(
            name: "AIChatMLX",
            dependencies: [
                .product(name: "AIChatCore", package: "AIChatKit"),
                .product(name: "JSONSchema", package: "swift-json-schema"),
                .product(name: "MLX", package: "mlx-swift"),
                .product(name: "MLXLLM", package: "mlx-swift-lm"),
                .product(name: "MLXVLM", package: "mlx-swift-lm"),
                .product(name: "MLXLMCommon", package: "mlx-swift-lm"),
                .product(name: "MLXHuggingFace", package: "mlx-swift-lm"),
                .product(name: "HuggingFace", package: "swift-huggingface"),
                .product(name: "Tokenizers", package: "swift-transformers"),
            ],
            path: "Sources/AIChatMLX"
        ),
        .testTarget(
            name: "AIChatMLXTests",
            dependencies: [
                "AIChatMLX",
                .product(name: "MLX", package: "mlx-swift"),
            ],
            path: "Tests/AIChatMLXTests"
        ),
    ]
)
