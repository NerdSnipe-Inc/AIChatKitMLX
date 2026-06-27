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

    let forceRemote = ProcessInfo.processInfo.environment["SPI_PROCESSING"] != nil
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
        .package(url: "https://github.com/huggingface/swift-huggingface.git", .upToNextMajor(from: "0.9.0")),
        .package(url: "https://github.com/huggingface/swift-transformers.git", .upToNextMajor(from: "1.2.1")),
    ],
    targets: [
        .target(
            name: "AIChatMLX",
            dependencies: [
                .product(name: "AIChatCore", package: "AIChatKit"),
                .product(name: "MLXLLM", package: "mlx-swift-lm"),
                .product(name: "MLXVLM", package: "mlx-swift-lm"),
                .product(name: "MLXLMCommon", package: "mlx-swift-lm"),
                .product(name: "MLXHuggingFace", package: "mlx-swift-lm"),
                .product(name: "HuggingFace", package: "swift-huggingface"),
                .product(name: "Tokenizers", package: "swift-transformers"),
            ],
            path: "Sources/AIChatMLX"
        ),
        .testTarget(name: "AIChatMLXTests", dependencies: ["AIChatMLX"], path: "Tests/AIChatMLXTests"),
    ]
)
