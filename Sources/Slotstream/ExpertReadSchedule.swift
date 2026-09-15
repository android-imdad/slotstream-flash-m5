// Experimental whole-piece scheduling. No extra reads, staging or MLX work.
import Foundation

package enum ExpertReadSchedule {
    package static func acquireProcessGuard() throws { try ModelProcessGuard.acquire() }
    /// Longest jobs first, assigned to the least loaded lane. Costs are source
    /// plus expanded destination bytes, not an estimate of SSD seconds. Ties
    /// use original job/lane order so the mapping is reproducible.
    package static func balanced(costs: [Int], lanes: Int) -> [[Int]] {
        guard !costs.isEmpty, lanes > 0 else { return [] }
        let count = min(lanes, costs.count)
        var loads = [Double](repeating: 0, count: count)
        var assignments = [[Int]](repeating: [], count: count)
        let ordered = costs.indices.sorted {
            costs[$0] == costs[$1] ? $0 < $1 : costs[$0] > costs[$1]
        }
        for job in ordered {
            var lane = 0
            for candidate in 1 ..< count where loads[candidate] < loads[lane] { lane = candidate }
            assignments[lane].append(job)
            loads[lane] += Double(max(0, costs[job]))
        }
        return assignments
    }
}
