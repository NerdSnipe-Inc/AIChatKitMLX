# AIChatKitMLX

Adds on-device Apple MLX inference to any app already using [AIChatKit](https://github.com/NerdSnipe-Inc/AIChatKit). Models are downloaded from Hugging Face Hub on first use and cached locally. Runs on Metal GPU and Apple Neural Engine — no network calls during inference.

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

// Default model: mlx-community/gemma-4-e4b-it-4bit (~2.5 GB, downloaded on first use)
let provider = MLXProvider()

@StateObject private var session = ChatSession(
    provider: provider,
    model: "",  // MLXProvider ignores the model string; pass anything
    options: ChatRequestOptions(systemPrompt: "You are a helpful assistant.")
)
```

`MLXProvider` is an **actor**. The model downloads and loads on the first `stream()` call.

---

## Showing download progress

Call `loadModel(progressHandler:)` before starting a conversation to display progress UI:

```swift
try await provider.loadModel { progress in
    // progress.fractionCompleted: Double (0.0–1.0)
    DispatchQueue.main.async {
        self.downloadProgress = progress.fractionCompleted
    }
}
// Model is now resident in memory; session.send() will respond immediately
```

---

## Custom model

```swift
// Any mlx-community model by Hub ID
let provider = MLXProvider(modelId: "mlx-community/Llama-3.2-3B-Instruct-4bit")

// Pre-downloaded local directory
let provider = MLXProvider(modelPath: URL(fileURLWithPath: "/path/to/model-dir"))
```

The model directory must contain `config.json`, weight shards, and tokenizer files — the standard layout produced by `mlx_lm.convert` or downloaded from [mlx-community](https://huggingface.co/mlx-community) on Hugging Face.

---

## Model cache location

Models are cached by the Hugging Face Swift library. The exact path depends on your app's sandbox state:

| App state | Cache path |
|---|---|
| Sandboxed (App Store / entitlements) | `~/Library/Containers/<bundle-id>/Data/Library/Caches/huggingface/hub/` |
| Not sandboxed | `~/.cache/huggingface/hub/` |

The cache is shared with the Python `huggingface_hub` library — if you've already downloaded a model via Python tools it will be found without re-downloading.

---

## Sampling options

```swift
MLXProvider(
    modelId:           "mlx-community/gemma-4-e4b-it-4bit",
    maxTokens:         nil,      // nil = unlimited
    temperature:       0.6,
    topP:              1.0,
    repetitionPenalty: nil       // nil = disabled
)
```

---

## License

MIT
