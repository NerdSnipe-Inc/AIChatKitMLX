# AIChatKitMLX

![Swift 5.10](https://img.shields.io/badge/Swift-5.10-orange?logo=swift)
![iOS 17+](https://img.shields.io/badge/iOS-17%2B-blue?logo=apple)
![macOS 14+](https://img.shields.io/badge/macOS-14%2B-blue?logo=apple)
![MIT License](https://img.shields.io/badge/license-MIT-green)
![SPM](https://img.shields.io/badge/SPM-compatible-brightgreen)
[![](https://img.shields.io/endpoint?url=https%3A%2F%2Fswiftpackageindex.com%2Fapi%2Fpackages%2FNerdSnipe-Inc%2FAIChatKitMLX%2Fbadge%3Ftype%3Dswift-versions)](https://swiftpackageindex.com/NerdSnipe-Inc/AIChatKitMLX)
[![](https://img.shields.io/endpoint?url=https%3A%2F%2Fswiftpackageindex.com%2Fapi%2Fpackages%2FNerdSnipe-Inc%2FAIChatKitMLX%2Fbadge%3Ftype%3Dplatforms)](https://swiftpackageindex.com/NerdSnipe-Inc/AIChatKitMLX)

Adds on-device Apple MLX inference to any app already using [AIChatKit](https://github.com/NerdSnipe-Inc/AIChatKit). Models are downloaded from Hugging Face Hub on first use and cached locally. Supports both **text-only LLMs** and **vision-language models (VLMs)**. Runs on Metal GPU and Apple Neural Engine — no network calls during inference.

**Platforms:** macOS 14+ · iOS 17+  
**Language:** Swift 5.10+  
**Requires:** Apple Silicon (M-series Mac or A-series iPhone/iPad)

> **Requires AIChatKit.** Add both packages to your target.  
> Do not add this target to builds that must run on Intel Macs or Simulator.

---

## Installation

```swift
// Package.swift
.package(url: "https://github.com/NerdSnipe-Inc/AIChatKit",    from: "1.1.2"),
.package(url: "https://github.com/NerdSnipe-Inc/AIChatKitMLX", from: "1.4.1"),

// Target dependencies
.product(name: "AIChatCore", package: "AIChatKit"),
.product(name: "AIChatUI",   package: "AIChatKit"),    // if using ChatSession / ChatView
.product(name: "AIChatMLX",  package: "AIChatKitMLX"),
```

---

## Quick start

```swift
import AIChatMLX
import AIChatUI

// Automatically picks the best model for the current device (see Model selection below)
let provider = MLXProvider()

@StateObject private var session = ChatSession(
    provider: provider,
    model: "",  // MLXProvider ignores the model string; pass anything
    options: ChatRequestOptions(systemPrompt: "You are a helpful assistant.")
)
```

`MLXProvider` is an **actor**. The model downloads and loads on the first `stream()` call, or you can pre-warm it explicitly with `loadModel(progressHandler:)`.

---

## Tool routing (FunctionGemma router + Gemma 4 responder)

`ToolRoutingProvider` puts a tiny router in front of the responder: the router decides whether a
turn needs a tool (and with which arguments); chat turns skip tool-schema prompt cost, and tool turns
skip the big model until the result has to be phrased.

```swift
let provider = ToolRoutingProvider.onDevice(          // FunctionGemma (.auxiliary slot) + Gemma 4 (.primary)
    onDecision: { print($0.outcome, $0.reason) }       // optional diagnostics
)
await provider.warmUp()                                // keep the router's cold load out of its time budget

let session = ChatSession(provider: provider, model: "", options: options) // options carries your tools
```

Read `docs/TOOL_ROUTING.md` first: the router is a small model and can miss required tool calls.

---

## Model selection

`MLXProvider.recommendedModelId()` always returns the small text-only model (`mlx-community/gemma-4-e4b-it-4bit`; the source describes it as fitting any Apple Silicon device with ≥ 8 GB RAM). The large model (`mlx-community/gemma-4-31b-it-4bit`) is opt-in: the host app must choose it and enforce its unified-memory requirement. VLMs load via `MLXVLM`, text models via `MLXLLM`.

```swift
// The model used by default (always the small one)
let modelId = MLXProvider.recommendedModelId()

// Named constants
MLXProvider.smallModelId  // gemma-4-e4b-it-4bit
MLXProvider.largeModelId  // gemma-4-31b-it-4bit
```

---

## Showing download progress

`loadModel(progressHandler:)` fetches the model itself if it's not already on disk, and reports progress as it goes. Fine for a quick start, but it has no retry, no resume, and no way to check "is this actually fully downloaded" from outside a load — see `MLXModelManager` below for that.

```swift
try await provider.loadModel { progress in
    Task { @MainActor in
        self.downloadProgress = progress.fractionCompleted  // 0.0–1.0
    }
}
// Model is now resident; session.send() responds immediately
```

---

## Model downloads: `MLXModelManager` (1.4.0+)

`MLXModelManager` is the one owner for a model's on-disk files, separate from loading it into memory. Use it when you want a real download experience — a Settings screen, a first-launch progress bar, a "Redownload" action — instead of whatever `loadModel`'s own fetch happens to do.

It gives you, on top of `loadModel`'s bare download:
- **Retry with backoff** on a failed attempt (3s, 10s, 30s, 1m, 2m, 5m by default).
- **Waits for connectivity** instead of burning retries while the Mac is offline.
- **Stall detection** — a connection that goes silent (no bytes, no error) is treated as a failure rather than hanging forever.
- **Resume after a quit or crash.** A download that never finished is picked back up automatically the next time `resumePendingDownloads()` runs.
- **On-disk truth.** A model reads as `.downloaded` only when every weight file is present (`hasCompleteWeights(for:)`); an interrupted download that left the config/tokenizer files behind does not count.

```swift
let manager = MLXModelManager()
manager.track([MLXProvider.recommendedModelId()])   // seed state from what's actually on disk, at launch
manager.resumePendingDownloads()                     // pick up anything interrupted by a previous quit

if manager.state(for: modelId) != .downloaded {
    await manager.download(modelId: modelId)         // drive a progress UI from manager.states[modelId]
}

// Only load once the weights are confirmed on disk — `downloadIfNeeded: false` means a load can
// never silently kick off a multi-gigabyte download behind a spinner; it throws .modelNotFound instead.
try await provider.loadModel(downloadIfNeeded: false)
```

`manager.retryNotices[modelId]` carries a short human-readable status while a download is recovering ("Connection problem — retrying in 30s (attempt 2 of 6)"). `cancelDownload(modelId:)` and `delete(modelId:)` are also available. See the type's doc comments for the full API, including the `admission`/`estimatedSizeBytes` hooks for host-specific policy (which models are allowed, memory checks, etc).

---

## Custom model

```swift
// Any mlx-community model by Hub ID — factory is selected automatically
let provider = MLXProvider(modelId: "mlx-community/Qwen3-4B-4bit")

// Pre-downloaded local directory
let provider = MLXProvider(modelPath: URL(fileURLWithPath: "/path/to/model-dir"))
```

The correct factory (VLM or LLM) is chosen at load time based on the model's `config.json`. No manual factory selection is needed.

---

## Vision input (VLMs)

When the large model is selected, the underlying `ModelContainer` supports image input via `UserInput`. This is available directly through the `perform { context in }` API on the container. VLM-specific features (image understanding, document analysis) are accessible when using `MLXProvider` as part of a tool-call flow or by working directly with the container.

---

## Sampling options

```swift
MLXProvider(
    modelId:           MLXProvider.recommendedModelId(),
    maxTokens:         nil,      // nil = unlimited
    temperature:       0.6,
    topP:              1.0,
    repetitionPenalty: nil       // nil = disabled
)
```

---

## Model cache location

Models are cached by the Hugging Face Swift library.

| App state | Cache path |
|---|---|
| Sandboxed (App Store / entitlements) | `~/Library/Containers/<bundle-id>/Data/Library/Caches/huggingface/hub/` |
| Not sandboxed | `~/.cache/huggingface/hub/` |

The cache is shared with the Python `huggingface_hub` library — models already downloaded via Python tools are found without re-downloading.

---

## Known limits

- **Gemma text tool calls are recovered at stream level, not the UI.** `Gemma4StreamProcessor` turns `call:name{…}`, `<tool_call>{json}</tool_call>` and ```` ```tool_code ```` blocks into `.toolCallComplete` events as they stream. As of AIChatKit 2.0.0, `AIChatUI`/`ChatSession` no longer does any of this itself — a host on AIChatKit 2.0.0+ needs AIChatKitMLX 1.4.0+ for Gemma models, or model-written-as-text tool calls just show up as plain text.
- **Requires `mlx-swift-lm` 3.31.4 or later.** On 3.31.3 and earlier `gemma-4-e4b-it-4bit` fails to load (`Key language_model.model.layers.24.self_attn.k_norm.weight not found`) and a repetition penalty crashes Gemma 4 generation. See the 1.3.1 entry in [CHANGELOG.md](CHANGELOG.md).
- **Limited live coverage.** The live tests (in the AICompleteChat repo, not this package) exercise `gemma-4-e4b-it-4bit` and, for routing, `functiongemma-270m`. The large model, other Gemma variants and other FunctionGemma variants are not covered by live tests.
- **`ToolRoutingProvider` is experimental and opt-in.** Stock FunctionGemma was less accurate and slower than Gemma 4's own tool calling in the measurements in [docs/TOOL_ROUTING.md](docs/TOOL_ROUTING.md); read it before enabling.
- **Router plus responder are both resident.** The router adds roughly 0.8 GB of weights on top of the responder (figures in the Memory section of [docs/TOOL_ROUTING.md](docs/TOOL_ROUTING.md#memory)).
- **Plain `swift test` cannot load the MLX Metal shaders.** Run the tests with `xcodebuild test -scheme AIChatKitMLX -destination 'platform=macOS' -skipMacroValidation`.

---

## License

MIT
