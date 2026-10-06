import Foundation
import MLX
import MLXNN
import MLXLMCommon
import XCTest
@testable import AIChatMLX

final class FullModelLoRATests: XCTestCase {
    /// A tiny stand-in for a decoder: attention plus an MLP, the way Gemma 4 layers are laid out.
    private final class Block: Module {
        @ModuleInfo(key: "q_proj") var q: Linear
        @ModuleInfo(key: "down_proj") var down: Linear
        override init() {
            _q.wrappedValue = Linear(8, 8, bias: false)
            _down.wrappedValue = Linear(8, 8, bias: false)
            super.init()
        }
    }

    private final class Tiny: Module {
        @ModuleInfo(key: "layers") var layers: [Block]
        override init() {
            _layers.wrappedValue = [Block(), Block()]
            super.init()
        }
    }

    private func adapter(paths: [String], rank: Int = 2, scale: Float = 20) -> LoRAContainer {
        var flat: [String: MLXArray] = [:]
        for path in paths {
            flat["\(path).lora_a"] = MLXArray.ones([8, rank]) * 0.1
            flat["\(path).lora_b"] = MLXArray.ones([rank, 8]) * 0.1
        }
        return LoRAContainer(
            configuration: LoRAConfiguration(
                numLayers: 2,
                fineTuneType: .lora,
                loraParameters: .init(rank: rank, scale: scale)
            ),
            parameters: ModuleParameters.unflattened(flat)
        )
    }

    func test_targetPaths_listsEveryLoRAModule() {
        let a = adapter(paths: ["layers.1.q_proj", "layers.1.down_proj"])
        XCTAssertEqual(FullModelLoRA.targetPaths(of: a), ["layers.1.down_proj", "layers.1.q_proj"])
    }

    func test_apply_wrapsAttentionAndMLPAndChangesOutput_thenRevertRestoresIt() throws {
        let model = Tiny()
        let x = MLXArray.ones([1, 8])
        let before = model.layers[1].down(x)
        eval(before)

        let wrapped = try FullModelLoRA.apply(
            adapter(paths: ["layers.1.q_proj", "layers.1.down_proj"]), to: model
        )
        XCTAssertEqual(wrapped.count, 2)
        XCTAssertTrue(model.layers[1].down is LoRALayer)
        XCTAssertTrue(model.layers[1].q is LoRALayer)
        XCTAssertFalse(model.layers[0].down is LoRALayer, "layers the adapter does not name stay untouched")

        // y + scale * (x @ A) @ B  =  base + 20 * (8 * 0.1 * 2 * 0.1)... computed, not hard-coded
        let after = model.layers[1].down(x)
        eval(after)
        let delta = (after - before).abs().max().item(Float.self)
        XCTAssertEqual(delta, 20 * (8 * 0.1) * 2 * 0.1, accuracy: 1e-3)

        FullModelLoRA.revert(paths: wrapped, in: model)
        XCTAssertFalse(model.layers[1].down is LoRALayer)
        let restored = model.layers[1].down(x)
        eval(restored)
        XCTAssertEqual((restored - before).abs().max().item(Float.self), 0, accuracy: 1e-6)
    }

    func test_apply_missingTargetThrowsBeforeTouchingTheModel() {
        let model = Tiny()
        XCTAssertThrowsError(try FullModelLoRA.apply(adapter(paths: ["layers.1.q_proj", "layers.5.q_proj"]), to: model)) {
            XCTAssertEqual($0 as? FullModelLoRA.Failure, .missingTarget("layers.5.q_proj"))
        }
        XCTAssertFalse(model.layers[1].q is LoRALayer)
    }
}
