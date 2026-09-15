import Foundation
import MLX
import Slotstream

extension Diagnostics {
    public static func flashNeuronMask() throws -> CheckReport {
        var c = CheckBuilder("flash-neuron-mask")
        let values: [Float] = (0 ..< 6400).map { Float(($0 % 640) / 64 + 1) }
        let input = MLXArray(values, [1, 1, 10, 1, 640]).asType(.bfloat16)
        let ids = (0 ..< 10).map(Int32.init)
        let dense = try FlashNeuronMaskTransform(retaining: 10, norms: { _, _ in
            throw PlanError("dense control must never access norms")
        })
        let untouched = try dense.transformHidden(layer: 0, routerRanks: Array(0 ..< 10), expertIDs: ids, value: input)
        c.expect("dense control returns the same tensor object", untouched === input)
        var recorded = false
        let sparse = try FlashNeuronMaskTransform(retaining: 2, norms: { _, _ in Array(repeating: 1, count: 640) },
            record: { _, ranks, experts, masks in
                recorded = ranks.count == 10 && experts == ids && masks.allSatisfy { $0 == Array(repeating: false, count: 8) + [true, true] }
            })
        let masked = try sparse.transformHidden(layer: 0, routerRanks: Array(0 ..< 10), expertIDs: ids, value: input)
        c.equal("mask preserves hidden dtype", masked.dtype, input.dtype)
        c.equal("mask preserves full down-projection shape", masked.shape, input.shape)
        c.equal("only selected 64-neuron blocks survive", masked.asType(.float32).asArray(Float.self),
            values.enumerated().map { $0.offset % 640 >= 512 ? $0.element : 0 })
        c.expect("mask recorder sees every original router rank", recorded)
        do { try sparse.validateForward(tokens: 2); c.expect("multi-token mask refused", false) }
        catch { c.expect("multi-token mask refused", true) }
        let failing = try FlashNeuronMaskTransform(retaining: 2, norms: { _, _ in Array(repeating: 1, count: 640) },
            record: { _, _, _, _ in throw PlanError("mask write failed") })
        do {
            _ = try failing.transformHidden(layer: 0, routerRanks: Array(0 ..< 10), expertIDs: ids, value: input)
            c.expect("mask writer failure propagates", false)
        } catch { c.expect("mask writer failure propagates", true) }
        return c.report()
    }

    public static func flashNeuronScoring() throws -> CheckReport {
        var c = CheckBuilder("flash-neuron-scoring")
        let norms = try FlashNeuronScoring.columnNorms([3, 0, -8, 4, -12, 15], rows: 2, columns: 3)
        c.equal("row-major independent column orientation", norms, [5, 12, 17])
        var hidden = [Float](repeating: 1, count: 640)
        hidden[639] = -2
        let scores = try FlashNeuronScoring.blockScores(hidden: hidden, norms: [Float](repeating: 1, count: 640))
        c.equal("negative SwiGLU values retain contribution", scores, Array(repeating: 64, count: 9) + [65])
        let mask = try FlashNeuronScoring.selectedBlocks(scores: scores, retaining: 2)
        c.equal("largest block then ascending-index tie break", mask, [true] + Array(repeating: false, count: 8) + [true])
        c.equal("ten blocks preserves all neurons", try FlashNeuronScoring.selectedBlocks(scores: scores, retaining: 10), Array(repeating: true, count: 10))
        let zero = try FlashNeuronScoring.blockScores(hidden: Array(repeating: 0, count: 640), norms: Array(repeating: 0, count: 640))
        c.equal("zero contribution deterministic mask", try FlashNeuronScoring.selectedBlocks(scores: zero, retaining: 2), [true, true] + Array(repeating: false, count: 8))
        for invalid in [Float.nan, Float.infinity] {
            do {
                _ = try FlashNeuronScoring.columnNorms([invalid], rows: 1, columns: 1)
                c.expect("nonfinite norm source refused", false)
            } catch { c.expect("nonfinite norm source refused", true) }
        }
        do {
            _ = try FlashNeuronScoring.columnNorms([], rows: Int.max, columns: 2)
            c.expect("overflow geometry refused", false)
        } catch { c.expect("overflow geometry refused", true) }
        do {
            _ = try FlashNeuronScoring.blockScores(hidden: hidden, norms: Array(repeating: -1, count: 640))
            c.expect("negative norms refused", false)
        } catch { c.expect("negative norms refused", true) }
        do {
            _ = try FlashNeuronScoring.selectedBlocks(scores: scores, retaining: 3)
            c.expect("unfrozen retained counts refused", false)
        } catch { c.expect("unfrozen retained counts refused", true) }
        return c.report()
    }
}
