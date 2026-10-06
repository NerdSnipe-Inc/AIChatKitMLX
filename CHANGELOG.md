# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [1.5.0] - 2026-10-06

### Fixed
- **LoRA adapters trained with mlx-lm's default keys now load on Gemma 4.** `LoRAContainer.load(into:)` only
  adapts the modules a model's `loraLayers` exposes (Gemma 4: `self_attn`), so an adapter that also trains the
  MLP and per-layer projections was rejected with "Unhandled keys lora_a/lora_b" — and with
  `.optional` that meant a silent fall-back to the base model. `FullModelLoRA` now wraps exactly the linear
  layers named in the adapter's tensors (same `y + scale * (x @ A) @ B` maths) and `unloadAdapter()` reverts them.
- **Gemma call parser tolerates a stray string delimiter after a bare value** (`limit:5<|"|>`). The call used to be
  dropped entirely; it now runs with the intended arguments. Real strings are unaffected.

### Added
- `MLXProvider.adapterStatus` (`.notConfigured` / `.pending` / `.active` / `.unavailable(reason:)`), an
  `adapterStatusHandler:` init parameter that is called on every change, and `retryAdapter()`. A failed adapter is
  reported once per model load instead of being retried and re-logged on every request.

## [1.4.1] - 2026-09-27

### Fixed
- **`loadModel(downloadIfNeeded: false)` failed to compile in a Release build.** The `isResident` check it depends
  on (added in 1.4.0) was accidentally scoped inside an `#if DEBUG` block meant only for test seams. Any consumer
  archiving a Release build while using this method — new in 1.4.0 — hit "value of type 'MLXModelRuntime' has no
  member 'isResident'". `swift test`/`xcodebuild test` always run Debug, so this was invisible until an actual
  Release archive. `isResident` now compiles in both configurations; the remaining test-seam accessors stay
  `#if DEBUG`-only.

### Upgrading from 1.4.0
- Required if you call `loadModel(downloadIfNeeded: false)` (AICompleteChat's `AppEnvironment`, and anything
  following the "Adopt `MLXModelManager` for downloads" guidance in the 1.4.0 notes) and ship a Release/Archive
  build. No code changes.

## [1.4.0] - 2026-09-27

Model downloads get a single owner, and Gemma tool calls written as text are recovered at stream level.

### Added
- `MLXModelManager`: one owner for on-device model files — download with retry and backoff, waits
  while offline, stall detection, resume after a quit or crash, and on-disk truth
  (`hasCompleteWeights`: a model is `.downloaded` only when every weight file is present). Observable
  `states`, `retryNotices` and `storageUsedBytes`; `download`, `cancelDownload`, `delete`,
  `track`, `resumePendingDownloads`; host policy via `admission` / `estimatedSizeBytes`.
- `Gemma4StreamProcessor` turns tool calls the model wrote as plain text into tool-call events:
  `<tool_call>{json}</tool_call>` blocks and ` ```tool_code ` fenced Python calls (a positional argument
  takes its name from the tool schema). A block that does not parse, or never closes, is shown as text.
  This is the recovery `AIChatUI` used to do on the finished assistant text; it now happens at stream
  level, so the text never reaches the UI.

### Changed
- Accepts `AIChatKit` 1.1.2 up to (not including) 3.0.0, so it works with AIChatKit 2.x.

### Upgrading from 1.3.x
1. **Pair it with AIChatKit 2.0.0** if you can. AIChatKit 2.0.0 removed its own text recovery
   (see its [changelog](https://github.com/NerdSnipe-Inc/AIChatKit/blob/main/CHANGELOG.md)); this release
   is what replaces it. With AIChatKit 1.x nothing breaks — recovery just happens in both places.
2. **Adopt `MLXModelManager` for downloads (recommended).** Hold one instance, call
   `track([modelId])` at launch, then `await download(modelId:)` if `state(for:)` is not
   `.downloaded`; drive your UI from `states[modelId]` and `retryNotices[modelId]`. Then load with
   `provider.loadModel(downloadIfNeeded: false)` so a load can never start a hidden download. Hosts that
   keep calling `loadModel()` with the default still work, without retry or resume.
3. No API was removed.

## [1.3.1] - 2026-09-20

### Fixed
- Requires `mlx-swift-lm` 3.31.4 or later (was any 3.x). On 3.31.3 and earlier the default model
  `gemma-4-e4b-it-4bit` does not load at all (`Key language_model.model.layers.24.self_attn.k_norm.weight
  not found in Gemma4Attention.RMSNorm`), and Gemma 4 generation crashes whenever a repetition
  penalty is set (`[broadcast_shapes] Shapes (20) and (N+19)`: the penalty's token ring read the
  prompt length from the batch dimension of the `[1, N]` array Gemma 4 supplies). Both are fixed in
  3.31.4; the floor stops an older resolved version from reintroducing them.
- Requires `AIChatKit` 1.1.2 or later, which reports a model that is present but fails to load
  (like the case above) as a load failure with its real cause, instead of "model not found".

## [1.3.0] - 2026-09-20

### Added

- **Experimental, opt-in:** `ToolRoutingProvider`, a two-stage `ChatProvider` that puts a small tool router (FunctionGemma-270M
  via `ToolRoutingProvider.onDevice()`) in front of a responder (Gemma 4). Validated tool calls are
  emitted without invoking the responder; everything else goes to the responder with the tool schemas
  removed. Configurable fallback / after-tool-result policies, router time budget, `requiresTool`
  escape hatch, and a `RoutingDecision` diagnostics callback. See `docs/TOOL_ROUTING.md` for measured
  accuracy and latency — read it before enabling; stock FunctionGemma is markedly less reliable than
  Gemma 4's own tool calling.

## [1.2.0] - 2026-09-20

### Changed
- Requires `AIChatKit` 1.1.0 or later (uses its `ChatLog` and the new specific `ChatError` cases).
- `mlx-swift-lm` is always resolved from upstream. It is no longer picked up from a sibling
  `../mlx-swift-lm` folder, which could silently shadow it with a locally edited or half-cloned copy.

### Fixed
- `Gemma4StreamProcessor` is now a split-safe state machine with a real argument parser
  (`GemmaCallSyntax`). `call:` markers split across tokens, `<channel|>` split mid-marker,
  `<|tool_call>` blocks, arguments containing braces or quotes, and prose such as "a call: 555" are
  all handled; truncated or unparseable calls surface as text instead of vanishing.
- Cancelling a stream consumer now stops generation (the inner task previously kept running).

### Added
- `MLXProvider` load / stream / complete failures are classified into specific `ChatError` cases
  (model not found, download failed, out of memory, load failed, unsupported model, template error,
  generation failed) and logged through `ChatLog` instead of `print`. Cancellation is never logged
  or surfaced as an error.

## [1.1.3] - 2026-09-05

### Fixed

- `AIChatMLX` target now declares an explicit dependency on `JSONSchema` (`swift-json-schema`).
  `AIChatCore.ChatRequestOptions.ToolDefinition` publicly exposes a `JSONSchema.JSONSchema` value,
  and `MLXProvider.toToolSpecs` touches that type, so `AIChatMLX`'s own compiled object needs
  `JSONSchema`'s metadata/witness tables at link time even though no file in this target ever
  writes `import JSONSchema` itself. Plain `swift build`/`swift test` never surfaced this (SwiftPM
  CLI links the whole dependency graph into one binary, which happens to satisfy the symbol
  regardless), but Xcode's native package integration builds every product as its own separate
  dynamic framework and only auto-links a target's *declared* dependencies — without this fix,
  any Xcode-project consumer of `AIChatMLX` hit a link error: `Undefined symbols ... type metadata
  accessor for JSONSchema.JSONSchema`. Found and fixed while wiring `AIChatKitMLX` into an
  Xcode-project app (ultralevel-inbox) for the first time — this bug was latent since this
  package's original release, only reachable by an Xcode-native consumer, not a command-line
  SwiftPM one.

[1.1.3]: https://github.com/NerdSnipe-Inc/AIChatKitMLX/releases/tag/1.1.3
