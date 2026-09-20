# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

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
