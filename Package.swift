// swift-tools-version: 5.10

import PackageDescription

// AIChatKitMLX adds on-device Apple MLX inference (Apple Silicon only) to any app
// that already uses AIChatKit. Models are downloaded from Hugging Face Hub on first use.
//
// Usage:
//   .package(url: "https://github.com/NerdSnipe-Inc/AIChatKit",    from: "0.1.0"),
//   .package(url: "https://github.com/NerdSnipe-Inc/AIChatKitMLX", from: "0.1.0"),
//
// Requires Apple Silicon. Do not add this target to builds that must run on Intel or Simulator.
//
// mlx-swift-lm is pinned to 3.x — Gemma 4 architecture support was added in 3.0.
// The 3.x series brings in swift-syntax 600.x; this resolves cleanly on Swift 6.3+.

let package = Package(
    name: "AIChatKitMLX",
    platforms: [.macOS(.v14), .iOS(.v17)],
    products: [
        .library(name: "AIChatMLX", targets: ["AIChatMLX"]),
    ],
    dependencies: [
        .package(url: "https://github.com/NerdSnipe-Inc/AIChatKit.git", from: "0.1.0"),
        .package(url: "https://github.com/ml-explore/mlx-swift-lm.git", from: "3.0.0"),
        .package(url: "https://github.com/huggingface/swift-huggingface.git",  .upToNextMajor(from: "0.9.0")),
        .package(url: "https://github.com/huggingface/swift-transformers.git", .upToNextMajor(from: "1.2.1")),
    ],
    targets: [
        .target(
            name: "AIChatMLX",
            dependencies: [
                .product(name: "AIChatCore",     package: "AIChatKit"),
                .product(name: "MLXLLM",         package: "mlx-swift-lm"),
                .product(name: "MLXVLM",         package: "mlx-swift-lm"),
                .product(name: "MLXLMCommon",    package: "mlx-swift-lm"),
                .product(name: "MLXHuggingFace", package: "mlx-swift-lm"),
                .product(name: "HuggingFace",    package: "swift-huggingface"),
                .product(name: "Tokenizers",     package: "swift-transformers"),
            ],
            path: "Sources/AIChatMLX"
        ),
        .testTarget(name: "AIChatMLXTests", dependencies: ["AIChatMLX"], path: "Tests/AIChatMLXTests"),
    ]
)
