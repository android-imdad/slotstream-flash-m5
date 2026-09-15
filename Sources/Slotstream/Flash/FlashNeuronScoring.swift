import Foundation

/// Diagnostic arithmetic shared by the column-norm exporter and neuron oracle.
/// No runtime policy selects this from normal generation or serving.
public enum FlashNeuronScoring {
    public static let algorithm = "fp32-values-fp64-sequential-column-l2-block64-v1"
    package static func acquireProcessGuard() throws { try ModelProcessGuard.acquire() }

    /// FP32 dequantized values, ascending output rows, FP64 sum and square root,
    /// then exactly one FP32 rounding. Input is row-major [outputs, neurons].
    public static func columnNorms(_ values: [Float], rows: Int, columns: Int) throws -> [Float] {
        guard rows > 0, columns > 0, rows <= Int.max / columns,
              values.count == rows * columns else { throw PlanError("invalid column-norm shape") }
        var sums = [Double](repeating: 0, count: columns)
        for row in 0 ..< rows {
            for column in 0 ..< columns {
                let value = Double(values[row * columns + column])
                guard value.isFinite else { throw PlanError("nonfinite down-projection value") }
                sums[column] += value * value
            }
        }
        return try sums.map {
            let result = Float($0.squareRoot())
            guard result.isFinite, result >= 0 else { throw PlanError("nonfinite column norm") }
            return result
        }
    }

    /// Stored FP32 inputs are promoted before multiplication. Accumulation
    /// order and index tie-breaks deliberately match the independent host code.
    public static func blockScores(hidden: [Float], norms: [Float]) throws -> [Double] {
        guard hidden.count == 640, norms.count == 640 else { throw PlanError("oracle requires 640 neurons") }
        var scores = [Double](repeating: 0, count: 10)
        for i in hidden.indices {
            guard hidden[i].isFinite, norms[i].isFinite, norms[i] >= 0 else {
                throw PlanError("invalid oracle activation or norm")
            }
            scores[i / 64] += abs(Double(hidden[i])) * Double(norms[i])
        }
        guard scores.allSatisfy(\.isFinite) else { throw PlanError("nonfinite block score") }
        return scores
    }

    public static func selectedBlocks(scores: [Double], retaining count: Int) throws -> [Bool] {
        guard [2, 4, 6, 8, 10].contains(count), scores.count == 10,
              scores.allSatisfy({ $0.isFinite && $0 >= 0 }) else {
            throw PlanError("invalid oracle block selection")
        }
        let order = scores.indices.sorted {
            scores[$0] == scores[$1] ? $0 < $1 : scores[$0] > scores[$1]
        }
        let selected = Set(order.prefix(count))
        return scores.indices.map { selected.contains($0) }
    }
}
