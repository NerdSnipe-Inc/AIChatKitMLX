import Foundation
import MLX
import MLXNN
import MLXLMCommon

/// Applies an mlx-lm trained LoRA adapter at the exact module paths named in its tensors.
///
/// `LoRAContainer.load(into:)` only adapts the modules a model's `loraLayers` exposes. For
/// Gemma 4 text models that is each decoder layer's `self_attn`, so an adapter trained with
/// mlx-lm's default key set (every linear layer in the last N layers: attention, MLP and the
/// per-layer-input projections) is rejected with "Unhandled keys lora_a/lora_b". The adapter's
/// tensor names (`language_model.model.layers.26.mlp.down_proj.lora_a`) already say which linear
/// layers it was trained on, so we wrap exactly those, then load the weights.
///
/// The layer maths is the library's own `LoRALinear` / `QLoRALinear`: `y + scale * (x @ A) @ B`.
enum FullModelLoRA {
    enum Failure: Error, LocalizedError, Equatable {
        case noTargets
        case missingTarget(String)
        case notLinear(String)

        var errorDescription: String? {
            switch self {
            case .noTargets: "The adapter contains no LoRA weights."
            case .missingTarget(let path): "The adapter targets \(path), which this model does not have."
            case .notLinear(let path): "The adapter targets \(path), which is not a linear layer."
            }
        }
    }

    /// Module paths the adapter has `lora_a`/`lora_b` weights for, e.g. `…layers.26.mlp.down_proj`.
    static func targetPaths(of adapter: LoRAContainer) -> [String] {
        var paths = Set<String>()
        for (key, _) in adapter.parameters.flattened() {
            guard key.hasSuffix(".lora_a") || key.hasSuffix(".lora_b") else { continue }
            paths.insert(String(key.dropLast(".lora_a".count)))
        }
        return paths.sorted()
    }

    /// Wraps every targeted linear layer and loads the adapter weights into it. Returns the paths
    /// that were wrapped so they can be reverted. Throws before touching the model if any target
    /// is missing or is not a linear layer.
    @discardableResult
    static func apply(_ adapter: LoRAContainer, to model: Module) throws -> [String] {
        let paths = targetPaths(of: adapter)
        guard !paths.isEmpty else { throw Failure.noTargets }

        let modules = Dictionary(model.namedModules().map { ($0.0, $0.1) }, uniquingKeysWith: { first, _ in first })
        var replacements: [(String, Module)] = []
        for path in paths {
            guard let module = modules[path] else { throw Failure.missingTarget(path) }
            guard let linear = module as? Linear else { throw Failure.notLinear(path) }
            replacements.append((
                path,
                LoRALinear.from(
                    linear: linear,
                    rank: adapter.configuration.loraParameters.rank,
                    scale: adapter.configuration.loraParameters.scale
                )
            ))
        }
        replace(replacements, in: modules)
        try model.update(parameters: adapter.parameters, verify: .noUnusedKeys)
        return paths
    }

    /// Restores the original linear layers at `paths`.
    static func revert(paths: [String], in model: Module) {
        let modules = Dictionary(model.namedModules().map { ($0.0, $0.1) }, uniquingKeysWith: { first, _ in first })
        var restored: [(String, Module)] = []
        for path in paths {
            if let layer = modules[path] as? LoRALayer {
                restored.append((path, layer.reverted()))
            }
        }
        replace(restored, in: modules)
    }

    /// Swaps each module into its direct parent. Updating from the root with a path such as
    /// `layers.26.mlp.down_proj` would build a sparse `layers` array (indices 0…25 missing) and
    /// MLX rejects that, so each replacement is applied to its own parent with a single-key update.
    private static func replace(_ replacements: [(String, Module)], in modules: [String: Module]) {
        for (path, module) in replacements {
            guard let dot = path.lastIndex(of: ".") else { continue }
            let parentPath = String(path[..<dot])
            let leaf = String(path[path.index(after: dot)...])
            modules[parentPath]?.update(modules: .unflattened([(leaf, module)]))
        }
    }
}
