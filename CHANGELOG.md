# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

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
