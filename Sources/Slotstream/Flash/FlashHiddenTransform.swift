import MLX

/// A diagnostic transformation is distinct from observation, which must never
/// change arithmetic. Callers own the norm reservation and per-document reset.
package protocol FlashHiddenTransform: AnyObject {
    func validateForward(tokens: Int) throws
    func transformHidden(layer: Int, routerRanks: [Int], expertIDs: [Int32], value: MLXArray) throws -> MLXArray
}

/// Dense-load oracle component. It masks intermediate neurons while still
/// reading complete experts. It is package-scoped until the charged capture
/// adapter and full-model quality gates qualify its use.
package final class FlashNeuronMaskTransform: FlashHiddenTransform {
    package let retaining: Int
    private let norms: (Int, Int) throws -> [Float]
    private let record: (Int, [Int], [Int32], [[Bool]]) throws -> Void

    package init(retaining: Int, norms: @escaping (Int, Int) throws -> [Float],
                 record: @escaping (Int, [Int], [Int32], [[Bool]]) throws -> Void = { _, _, _, _ in }) throws {
        guard [2, 4, 6, 8, 10].contains(retaining) else { throw PlanError("invalid oracle retained count") }
        self.retaining = retaining
        self.norms = norms
        self.record = record
    }

    package func validateForward(tokens: Int) throws {
        guard tokens == 1 else { throw PlanError("neuron oracle requires exactly one token") }
    }

    package func transformHidden(layer: Int, routerRanks: [Int], expertIDs: [Int32], value: MLXArray) throws -> MLXArray {
        guard (0 ..< 48).contains(layer), routerRanks == Array(0 ..< 10),
              expertIDs.count == 10, Set(expertIDs).count == 10,
              expertIDs.allSatisfy({ (0 ..< 512).contains($0) }),
              value.shape == [1, 1, 10, 1, 640],
              [.float32, .float16, .bfloat16].contains(value.dtype) else {
            throw PlanError("neuron oracle requires the unsplit one-token routed hidden tensor")
        }
        if retaining == 10 {
            // Preserve the original graph: no copy, cast, multiply or matmul.
            try record(layer, routerRanks, expertIDs, Array(repeating: Array(repeating: true, count: 10), count: 10))
            return value
        }
        let values = value.asType(.float32).asArray(Float.self)
        var masks: [[Bool]] = []
        var expanded: [Float] = []
        expanded.reserveCapacity(6400)
        for row in 0 ..< 10 {
            let hidden = Array(values[row * 640 ..< (row + 1) * 640])
            let scores = try FlashNeuronScoring.blockScores(hidden: hidden, norms: norms(layer, Int(expertIDs[row])))
            let mask = try FlashNeuronScoring.selectedBlocks(scores: scores, retaining: retaining)
            masks.append(mask)
            expanded += mask.flatMap { Array(repeating: $0 ? Float(1) : Float(0), count: 64) }
        }
        try record(layer, routerRanks, expertIDs, masks)
        return value * MLXArray(expanded, value.shape).asType(value.dtype)
    }
}
