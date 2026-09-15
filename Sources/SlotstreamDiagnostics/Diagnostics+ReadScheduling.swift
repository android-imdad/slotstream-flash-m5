import CryptoKit
import Foundation
import MLX
import Slotstream

public struct ReadSchedulingCase: Codable {
    public var layer: Int
    public var experts: [Int]
    public var queueDepth: Int
    public var orders: [[String]]
    public var stridedSeconds: [Double]
    public var balancedSeconds: [Double]
    public var outputSHA256: String
}

public struct ReadSchedulingReport: Codable {
    public var format = "slotstream-read-scheduling-component-v1"
    public var cases: [ReadSchedulingCase]
    public var sourceIdentity: String
    public var sourceRegionBytes: Int
    public var readControls: [ExpertReadControlResult]
    public var conditionsBefore: ProcessMemory.OperatingConditions
    public var conditionsAfter: ProcessMemory.OperatingConditions
    public var footprint: FootprintSampler.Result
    public var check: CheckReport
    public var scope = "Eight alternating pairs per case; packed4-to6 in both arms; complete checked batch calls including staging and expansion; SHA verification outside timing. F_NOCACHE does not establish cold physical SSD reads. No generation speed claim."
}

extension Diagnostics {
    public static func readScheduling() -> CheckReport {
        var c = CheckBuilder("read-scheduling")
        for count in [1, 9, 27, 90, 288] {
            let costs = (0 ..< count).map { $0 % 3 == 0 ? 2_000_000 : 60_000 }
            for lanes in [1, 12, 32, 128] {
                let schedule = ExpertReadSchedule.balanced(costs: costs, lanes: lanes)
                c.equal("every job once n=\(count) lanes=\(lanes)", schedule.flatMap { $0 }.sorted(), Array(0 ..< count))
                c.equal("bounded lanes n=\(count) lanes=\(lanes)", schedule.count, min(lanes, count))
                c.equal("deterministic n=\(count) lanes=\(lanes)", schedule,
                    ExpertReadSchedule.balanced(costs: costs, lanes: lanes))
            }
        }
        c.equal("empty jobs", ExpertReadSchedule.balanced(costs: [], lanes: 32), [])
        c.equal("zero lanes", ExpertReadSchedule.balanced(costs: [1], lanes: 0), [])
        // Striding [large, small] over two lanes serializes every large job.
        c.equal("large jobs spread across lanes", ExpertReadSchedule.balanced(costs: [100, 1, 100, 1], lanes: 2), [[0, 1], [2, 3]])
        return c.report()
    }

    public static func readSchedulingComponent(modelDir: URL) throws -> ReadSchedulingReport {
        try ExpertReadSchedule.acquireProcessGuard()
        guard let vm = ProcessMemory.vmActivity(), vm.reclaimableBytes >= 4_000_000_000 else {
            throw ModelError("read scheduling component needs 4 GB reclaimable memory")
        }
        MLX.Memory.cacheLimit = 64 << 20
        let sampler = FootprintSampler()
        let conditionsBefore = ProcessMemory.operatingConditions()
        let index = try CheckpointIndex(dir: modelDir)
        guard index.config.format == .jang6S else { throw ModelError("read scheduling requires JANG_6S") }
        let store = try ExpertStore(index: index, wideningPolicy: .packed4To6)
        let identity = try store.wideningSourceIdentity()
        let controls = store.configureWideningReadControls()
        guard controls.allSatisfy({ $0.noCacheReturnCode == 0 && $0.readAheadReturnCode == 0 }) else {
            throw ModelError("read scheduling descriptor controls failed")
        }
        let layers = [0, 5, 22]
        let experts = [0, 7, 19, 43, 79, 131, 211, 307, 401, 511]
        let regions = layers.reduce(0) { $0 + store.sourceRecordBytes(layer: $1) * experts.count }
        guard regions <= 256 << 20 else { throw ModelError("read scheduling source regions exceed 256 MiB") }
        var cases: [ReadSchedulingCase] = []
        var check = CheckBuilder("read-scheduling-component")
        for layer in layers {
            for depth in [12, 32] {
                var orders: [[String]] = []
                var samples: [String: [Double]] = [:]
                var expected: String?
                for pair in 0 ..< 8 {
                    let order = pair.isMultiple(of: 2) ? ["strided", "balanced"] : ["balanced", "strided"]
                    orders.append(order)
                    for mode in order {
                        store.balancedReadScheduling = mode == "balanced"
                        let start = RuntimeClock.now()
                        let output = try store.readBatchChecked(experts.map { ExpertKey(layer, $0) }, queueDepth: depth)
                        samples[mode, default: []].append(RuntimeClock.seconds(since: start))
                        var digest = SHA256()
                        for tensor in output {
                            digest.update(data: Data(tensor.reshaped([-1]).view(dtype: .uint8).asArray(UInt8.self)))
                        }
                        let hash = digest.finalize().map { String(format: "%02x", $0) }.joined()
                        if let expected, expected != hash { throw ModelError("read scheduling output bytes changed") }
                        expected = hash
                        guard ProcessMemory.residentBytes() <= 512_000_000 else {
                            throw ModelError("read scheduling component exceeds 512 MB")
                        }
                    }
                }
                cases.append(ReadSchedulingCase(layer: layer, experts: experts, queueDepth: depth,
                    orders: orders, stridedSeconds: samples["strided"]!, balancedSeconds: samples["balanced"]!,
                    outputSHA256: expected!))
                check.expect("complete tensor bytes match layer=\(layer) depth=\(depth)", true)
            }
        }
        check.equal("source identity unchanged", try store.wideningSourceIdentity(), identity)
        let footprint = sampler.finish()
        check.expect("512 MB physical footprint", footprint.peakBytes <= 512_000_000)
        return ReadSchedulingReport(cases: cases, sourceIdentity: identity, sourceRegionBytes: regions,
            readControls: controls, conditionsBefore: conditionsBefore,
            conditionsAfter: ProcessMemory.operatingConditions(), footprint: footprint, check: check.report())
    }
}
