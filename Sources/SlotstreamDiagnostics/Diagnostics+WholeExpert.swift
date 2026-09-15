import CryptoKit
import Foundation
import MLX
import Slotstream

public struct WholeExpertCase: Codable {
    public var layer: Int
    public var count: Int
    public var orders: [[String]]
    public var rawSeconds: [Double]
    public var wholeSeconds: [Double]
    public var rawSourceBytesPerCall: Int
    public var wholeSourceBytesPerCall: Int
    public var outputSHA256: String
}

public struct WholeExpertReport: Codable {
    public var format = "slotstream-whole-expert-component-v1"
    public var sourceIdentity: String
    public var layers: [Int]
    public var experts: [Int]
    public var originalRegionBytes: Int
    public var artifactBytes: Int
    public var constructionSeconds: Double
    public var verificationSeconds: Double
    public var queueDepth = 32
    public var cases: [WholeExpertCase]
    public var rawReadControls: [ExpertReadControlResult]
    public var wholeReadControls = "PackedExpertLayout.openRead requires successful F_NOCACHE=1 and F_RDAHEAD=0"
    public var conditionsBefore: ProcessMemory.OperatingConditions
    public var conditionsAfter: ProcessMemory.OperatingConditions
    public var footprint: FootprintSampler.Result
    public var check: CheckReport
    public var scope = "Bounded expanded v1 layout, same nine output tensors; raw packed4-to6 conversion versus pre-expanded whole records. Eight alternating pairs; allocation, scheduling, I/O, conversion/copy and MLX wrapping timed. Artifact construction, verification and output hashes excluded. No cold-SSD or inference-speed claim."
}

extension Diagnostics {
    public static func wholeExpertComponent(modelDir: URL, artifact: URL) throws -> WholeExpertReport {
        try ExpertReadSchedule.acquireProcessGuard()
        guard let vm = ProcessMemory.vmActivity(), vm.reclaimableBytes >= 4_000_000_000 else {
            throw ModelError("whole-expert component requires 4 GB reclaimable memory")
        }
        MLX.Memory.cacheLimit = 64 << 20
        let sampler = FootprintSampler()
        let index = try CheckpointIndex(dir: modelDir)
        guard index.config.format == .jang6S else { throw ModelError("whole-expert component requires JANG_6S") }
        let store = try ExpertStore(index: index, wideningPolicy: .packed4To6)
        let sourceIdentity = try store.wideningSourceIdentity()
        let controls = store.configureWideningReadControls()
        guard controls.allSatisfy({ $0.noCacheReturnCode == 0 && $0.readAheadReturnCode == 0 }) else {
            throw ModelError("raw read controls failed")
        }
        let layers = [0, 5, 22]
        let experts = [0, 7, 19, 43, 79, 131, 211, 307, 401, 511]
        let regions = layers.reduce(0) { $0 + experts.count * store.sourceRecordBytes(layer: $1) }
        guard regions <= 256 << 20 else { throw ModelError("original source region bound exceeded") }
        let started = RuntimeClock.now()
        let whole = try store.buildPackedSample(at: artifact, layers: layers, experts: experts)
        let constructionSeconds = RuntimeClock.seconds(since: started) - whole.verificationSeconds
        let conditionsBefore = ProcessMemory.operatingConditions()
        var cases: [WholeExpertCase] = []
        var check = CheckBuilder("whole-expert-component")
        func hash(_ output: [MLXArray]) -> String {
            var digest = SHA256()
            for tensor in output {
                digest.update(data: Data(tensor.reshaped([-1]).view(dtype: .uint8).asArray(UInt8.self)))
            }
            return digest.finalize().map { String(format: "%02x", $0) }.joined()
        }
        for (localLayer, layer) in layers.enumerated() {
            for count in [1, 4, 10] {
                let rawKeys = experts.prefix(count).map { ExpertKey(layer, $0) }
                let localKeys = (0 ..< count).map { ExpertKey(localLayer, $0) }
                var orders: [[String]] = []
                var times: [String: [Double]] = [:]
                var expected: String?
                for pair in 0 ..< 8 {
                    let order = (pair + localLayer).isMultiple(of: 2) ? ["raw", "whole"] : ["whole", "raw"]
                    orders.append(order)
                    for arm in order {
                        let start = RuntimeClock.now()
                        let output = arm == "raw" ? try store.readBatchChecked(rawKeys, queueDepth: 32)
                            : try store.readPackedSample(whole, localKeys: localKeys, queueDepth: 32)
                        times[arm, default: []].append(RuntimeClock.seconds(since: start))
                        let digest = hash(output)
                        if let expected, expected != digest { throw ModelError("whole-expert output bytes differ") }
                        expected = digest
                        guard ProcessMemory.residentBytes() <= 512_000_000 else {
                            throw ModelError("whole-expert component exceeds 512 MB")
                        }
                    }
                }
                check.expect("all tensor bytes identical layer=\(layer) count=\(count)", true)
                cases.append(WholeExpertCase(layer: layer, count: count, orders: orders,
                    rawSeconds: times["raw"]!, wholeSeconds: times["whole"]!,
                    rawSourceBytesPerCall: count * store.sourceRecordBytes(layer: layer),
                    wholeSourceBytesPerCall: count * store.recordBytes, outputSHA256: expected!))
            }
        }
        // Verify rank/duplicates too, outside timing. Duplicate reads may not
        // silently collapse output rows or change the caller's routing order.
        let order = [9, 0, 9, 3]
        let raw = try store.readBatchChecked(order.map { ExpertKey(5, experts[$0]) }, queueDepth: 32)
        let packed = try store.readPackedSample(whole, localKeys: order.map { ExpertKey(1, $0) }, queueDepth: 32)
        check.equal("duplicate and reordered experts preserve output", hash(raw), hash(packed))
        var failureRefused = false
        whole.readFault = ReadFault(afterJobs: 0)
        do { _ = try store.readPackedSample(whole, localKeys: [ExpertKey(0, 0)], queueDepth: 32) }
        catch { failureRefused = true }
        whole.readFault = nil
        check.expect("failed sample read cannot return partial tensors", failureRefused)
        let recovered = try store.readPackedSample(whole, localKeys: [ExpertKey(0, 0)], queueDepth: 32)
        let original = try store.readBatchChecked([ExpertKey(0, experts[0])], queueDepth: 32)
        check.equal("joined read failure leaves subsequent reads intact", hash(recovered), hash(original))
        var subsetRefused = false
        let ordinaryStore = try ExpertStore(index: index)
        do { _ = try ordinaryStore.loadPackedLayout(at: artifact) }
        catch { subsetRefused = true }
        check.expect("ordinary reader rejects the subset as a complete layout", subsetRefused)
        check.equal("original source identity unchanged", try store.wideningSourceIdentity(), sourceIdentity)
        try whole.checkUnchanged()
        let conditionsAfter = ProcessMemory.operatingConditions()
        let footprint = sampler.finish()
        check.expect("physical peak below 512 MB", footprint.peakBytes <= 512_000_000)
        return WholeExpertReport(sourceIdentity: sourceIdentity, layers: layers, experts: experts,
            originalRegionBytes: regions, artifactBytes: whole.verifiedBytes,
            constructionSeconds: constructionSeconds, verificationSeconds: whole.verificationSeconds,
            cases: cases, rawReadControls: controls, conditionsBefore: conditionsBefore,
            conditionsAfter: conditionsAfter, footprint: footprint, check: check.report())
    }
}
