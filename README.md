# AIChatKitMLX

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
.package(url: "https://github.com/NerdSnipe-Inc/AIChatKit",    from: "0.1.0"),
.package(url: "https://github.com/NerdSnipe-Inc/AIChatKitMLX", from: "0.1.0"),

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

## Model selection

`MLXProvider` automatically selects a model based on the device's available RAM:

| Device | Model | Type | Download size |
|--------|-------|------|---------------|
| macOS ≥ 16 GB RAM | `mlx-community/diffusiongemma-26B-A4B-it-4bit` | VLM (text + images) | ~8–10 GB |
| macOS < 16 GB / iOS | `mlx-community/gemma-4-e4b-it-4bit` | LLM (text only) | ~2–3 GB |

The 26B model is a Mixture-of-Experts architecture with ~4B active parameters per forward pass — faster and leaner than a dense 26B model while retaining broad capability. The factory is selected automatically: VLMs load via `MLXVLM`, text models via `MLXLLM`.

```swift
// Check which model will be used on the current device
let modelId = MLXProvider.recommendedModelId()

// Named constants
MLXProvider.smallModelId  // gemma-4-e4b-it-4bit
MLXProvider.largeModelId  // diffusiongemma-26B-A4B-it-4bit
```

---

## Showing download progress

Call `loadModel(progressHandler:)` before starting a conversation to surface progress in your UI. The download only happens once — subsequent calls return immediately if the model is already in memory.

```swift
try await provider.loadModel { progress in
    Task { @MainActor in
        self.downloadProgress = progress.fractionCompleted  // 0.0–1.0
    }
}
// Model is now resident; session.send() responds immediately
```

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

## License

MIT
