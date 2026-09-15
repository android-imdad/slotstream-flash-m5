import CryptoKit
import Foundation
import MLX
import Slotstream

public struct PrefetchCostCase: Codable {
    public var layer: Int
    public var count: Int
    public var sourceBytes: Int
    public var seconds: [Double]
    public var outputSHA256: String
}

public struct PrefetchCostReport: Codable {
    public var format = "slotstream-prefetch-reader-costs-v1"
    public var sourceIdentity: String
    public var queueDepth = 32
    public var policy = "packed4-to6"
    public var experts: [Int]
    public var uniqueSourceBytes: Int
    public var cases: [PrefetchCostCase]
    public var readControls: [ExpertReadControlResult]
    public var conditionsBefore: ProcessMemory.OperatingConditions
    public var conditionsAfter: ProcessMemory.OperatingConditions
    public var footprint: FootprintSampler.Result
    public var check: CheckReport
    public var scope = "Eight repeated original-reader calls for every batch size 1...10 and three source classes; rotated count order. Complete staging/read/conversion/MLX wrapping timed. Output hashing excluded. Empirical cost calibration, not a cold-SSD bound or prefetch speedup."
}

extension Diagnostics {
    public static func prefetchReaderCosts(modelDir: URL) throws -> PrefetchCostReport {
        try ExpertReadSchedule.acquireProcessGuard()
        guard let vm = ProcessMemory.vmActivity(), vm.reclaimableBytes >= 4_000_000_000 else {
            throw ModelError("prefetch cost calibration needs 4 GB reclaimable memory")
        }
        MLX.Memory.cacheLimit = 64 << 20
        let sampler = FootprintSampler()
        let index = try CheckpointIndex(dir: modelDir)
        guard index.config.format == .jang6S else { throw ModelError("prefetch costs require JANG_6S") }
        let store = try ExpertStore(index: index, wideningPolicy: .packed4To6)
        let identity = try store.wideningSourceIdentity()
        let controls = store.configureWideningReadControls()
        guard controls.allSatisfy({ $0.noCacheReturnCode == 0 && $0.readAheadReturnCode == 0 }) else {
            throw ModelError("prefetch cost descriptor controls failed")
        }
        let experts = [0, 7, 19, 43, 79, 131, 211, 307, 401, 511]
        let layers = [0, 5, 22]
        let regions = layers.reduce(0) { $0 + store.sourceRecordBytes(layer: $1) * 10 }
        guard regions <= 256 << 20 else { throw ModelError("prefetch source regions exceed 256 MiB") }
        let before = ProcessMemory.operatingConditions()
        var samples: [Int: [Double]] = [:]
        var hashes: [Int: String] = [:]
        for round in 0 ..< 8 {
            for (index, layer) in layers.enumerated() {
                for step in 0 ..< 10 {
                    let count = (step + round + index) % 10 + 1
                    let key = layer * 100 + count
                    let start = RuntimeClock.now()
                    let values = try store.readBatchChecked(experts.prefix(count).map { ExpertKey(layer, $0) }, queueDepth: 32)
                    samples[key, default: []].append(RuntimeClock.seconds(since: start))
                    var hash = SHA256()
                    for value in values { hash.update(data: Data(value.reshaped([-1]).view(dtype: .uint8).asArray(UInt8.self))) }
                    let digest = hash.finalize().map { String(format: "%02x", $0) }.joined()
                    if let prior = hashes[key], prior != digest { throw ModelError("reader calibration bytes changed") }
                    hashes[key] = digest
                    guard ProcessMemory.residentBytes() <= 512_000_000 else { throw ModelError("prefetch cost footprint exceeds 512 MB") }
                }
            }
        }
        var cases: [PrefetchCostCase] = []
        for layer in layers {
            for count in 1 ... 10 {
                let key = layer * 100 + count
                cases.append(PrefetchCostCase(layer: layer, count: count, sourceBytes: store.sourceRecordBytes(layer: layer) * count,
                    seconds: samples[key]!, outputSHA256: hashes[key]!))
            }
        }
        var check = CheckBuilder("prefetch-reader-costs")
        check.equal("all thirty batch/precision cases measured", cases.count, 30)
        check.expect("eight identical-output repetitions per case", cases.allSatisfy { $0.seconds.count == 8 })
        check.equal("original source identity unchanged", try store.wideningSourceIdentity(), identity)
        let after = ProcessMemory.operatingConditions()
        let footprint = sampler.finish()
        check.expect("512 MB physical footprint bound", footprint.peakBytes <= 512_000_000)
        return PrefetchCostReport(sourceIdentity: identity, experts: experts, uniqueSourceBytes: regions,
            cases: cases, readControls: controls, conditionsBefore: before, conditionsAfter: after,
            footprint: footprint, check: check.report())
    }
}
