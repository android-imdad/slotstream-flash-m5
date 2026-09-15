import CryptoKit
import Darwin
import Foundation
import MLX
import Slotstream

public struct WideningBenchmarkResult: Codable {
    public var workers: Int
    public var policy: AffineWideningPolicy
    public var secondsPerCall: [Double]
    public var medianSecondsPerCall: Double
    public var checksum: UInt64
}

public struct WideningSyntheticReport: Codable {
    public var format = "slotstream-widening-synthetic-v1"
    public var schemaVersion = 1
    public var method: String
    public var codeCountPerCall: Int
    public var pairCount: Int
    public var allocatedBufferBytes: Int
    public var results: [WideningBenchmarkResult]
    public var exactCheck: CheckReport
    public var processFootprintEndBytes: UInt64
}

public struct WideningReaderCase: Codable {
    public var api: String
    public var queueDepth: Int
    public var layer: Int
    public var experts: [Int]
    public var expectedWideningProjections: Int
    public var observedWideningProjections: Int
    public var pairOrders: [[AffineWideningPolicy]]
    public var scalarSeconds: [Double]
    public var packedSeconds: [Double]
    public var scalarMedianSeconds: Double
    public var packedMedianSeconds: Double
    public var comparedBytes: Int
    public var exactBytes: Bool
    public var outputSHA256: String
}

public struct WideningModelReport: Codable {
    public var format = "slotstream-widening-model-component-v1"
    public var schemaVersion = 1
    public var modelPath: String
    public var checkpointFormat: CheckpointFormat
    public var policies: [AffineWideningPolicy]
    public var sourceIdentityBefore: String
    public var sourceIdentityAfter: String
    public var uniqueOriginalRegionBytes: Int
    public var uniqueOriginalRegionLimitBytes: Int
    public var logicalSourceBytesRead: Int
    public var physicalFootprintLimitBytes: UInt64
    public var footprint: FootprintSampler.Result
    public var diskBytesReadBefore: UInt64?
    public var diskBytesReadAfter: UInt64?
    public var diskObservationScope: String
    public var readControls: [ExpertReadControlResult]
    public var timingScope: String
    public var verificationScope: String
    public var cases: [WideningReaderCase]
    public var check: CheckReport
}

private final class WideningFailure: @unchecked Sendable {
    private let lock = NSLock()
    private var first: Error?
    func record(_ error: Error) { lock.withLock { if first == nil { first = error } } }
    func finish() throws { if let error = lock.withLock({ first }) { throw error } }
}

extension Diagnostics {
    /// Byte-exact checks for the opt-in packed 4-to-6-bit affine-code widening.
    /// This is pure Swift: it allocates bounded host buffers and touches no MLX
    /// device, checkpoint, or filesystem state.
    public static func exactWidening() -> CheckReport {
        var c = CheckBuilder("exact-widening")

        func oracle(_ source: [UInt8], count: Int, from sourceBits: Int, to targetBits: Int) -> [UInt8] {
            var target = [UInt8](repeating: 0, count: count * targetBits / 8)
            for codeIndex in 0 ..< count {
                var code: UInt8 = 0
                for codeBit in 0 ..< sourceBits {
                    let sourceBit = codeIndex * sourceBits + codeBit
                    let value = (source[sourceBit / 8] >> (sourceBit % 8)) & 1
                    code |= value << codeBit
                }
                for codeBit in 0 ..< targetBits {
                    let targetBit = codeIndex * targetBits + codeBit
                    let value = codeBit < sourceBits ? ((code >> codeBit) & 1) : 0
                    target[targetBit / 8] |= value << (targetBit % 8)
                }
            }
            return target
        }

        func expand(_ source: [UInt8], count: Int, from sourceBits: Int, to targetBits: Int,
                    policy: AffineWideningPolicy, fill: UInt8 = 0xff) throws -> [UInt8] {
            var target = [UInt8](repeating: fill, count: count * targetBits / 8)
            try source.withUnsafeBytes { sourceBytes in
                try target.withUnsafeMutableBytes { targetBytes in
                    try AffineCodes.widen(sourceBytes, to: targetBytes, count: count,
                        from: sourceBits, to: targetBits, policy: policy)
                }
            }
            return target
        }

        var exhaustive = true
        var firstMismatch: String?
        for word in 0 ... UInt16.max {
            let source = [UInt8(truncatingIfNeeded: word), UInt8(truncatingIfNeeded: word >> 8)]
            let expected = oracle(source, count: 4, from: 4, to: 6)
            do {
                let actual = try expand(source, count: 4, from: 4, to: 6,
                    policy: .packed4To6)
                if actual != expected {
                    exhaustive = false
                    firstMismatch = "input \(String(word, radix: 16)): \(actual) != \(expected)"
                    break
                }
            } catch {
                exhaustive = false
                firstMismatch = String(describing: error)
                break
            }
        }
        c.expect("all 65536 two-byte inputs match an independent bit oracle",
            exhaustive, firstMismatch)

        let boundarySource: [UInt8] = [0x10, 0x32, 0x54, 0x76, 0x98, 0xba]
        do {
            let packed = try expand(boundarySource, count: 12, from: 4, to: 6,
                policy: .packed4To6)
            let scalar = try expand(boundarySource, count: 12, from: 4, to: 6,
                policy: .scalar)
            c.equal("multiple groups preserve first and last codes", packed,
                oracle(boundarySource, count: 12, from: 4, to: 6))
            c.equal("packed multiple-group output equals scalar output", packed, scalar)
        } catch {
            c.expect("multiple packed groups widen", false, String(describing: error))
        }

        do {
            var sourceStorage: [UInt8] = [0xa5] + boundarySource + [0x5a]
            var targetStorage = [UInt8](repeating: 0xff, count: 11)
            try sourceStorage.withUnsafeMutableBytes { sourceStorageBytes in
                try targetStorage.withUnsafeMutableBytes { targetStorageBytes in
                    let source = UnsafeRawBufferPointer(start: sourceStorageBytes.baseAddress! + 1,
                        count: boundarySource.count)
                    let target = UnsafeMutableRawBufferPointer(start: targetStorageBytes.baseAddress! + 1,
                        count: 9)
                    try AffineCodes.widen(source, to: target, count: 12, from: 4, to: 6,
                        policy: .packed4To6)
                }
            }
            c.equal("unaligned source keeps both sentinels", [sourceStorage.first!, sourceStorage.last!],
                [0xa5, 0x5a])
            c.equal("unaligned target keeps both sentinels", [targetStorage.first!, targetStorage.last!],
                [0xff, 0xff])
            c.equal("unaligned target writes every selected byte",
                Array(targetStorage[1 ..< 10]), oracle(boundarySource, count: 12, from: 4, to: 6))
        } catch {
            c.expect("unaligned slices widen", false, String(describing: error))
        }

        func overlapping(_ policy: AffineWideningPolicy) throws -> [UInt8] {
            var storage: [UInt8] = [0x21, 0x43, 0x65, 0x87, 0xa9]
            try storage.withUnsafeMutableBytes { bytes in
                let source = UnsafeRawBufferPointer(start: bytes.baseAddress!, count: 2)
                let target = UnsafeMutableRawBufferPointer(start: bytes.baseAddress! + 1, count: 3)
                try AffineCodes.widen(source, to: target, count: 4, from: 4, to: 6,
                    policy: policy)
            }
            return storage
        }
        do {
            c.equal("overlap retains scalar clear-then-expand behavior",
                try overlapping(.packed4To6), try overlapping(.scalar))
        } catch {
            c.expect("overlapping buffers use the preserved fallback", false, String(describing: error))
        }

        var emptySource: [UInt8] = []
        var emptyTarget: [UInt8] = []
        do {
            try emptySource.withUnsafeMutableBytes { source in
                try emptyTarget.withUnsafeMutableBytes { target in
                    try AffineCodes.widen(UnsafeRawBufferPointer(source), to: target,
                        count: 0, from: 4, to: 6, policy: .packed4To6)
                }
            }
            c.expect("zero codes preserve the empty-buffer contract", true)
        } catch {
            c.expect("zero codes preserve the empty-buffer contract", false, String(describing: error))
        }

        func rejects(_ name: String, sourceCount: Int = 0, targetCount: Int = 0,
                     count: Int, from sourceBits: Int, to targetBits: Int) {
            var source = [UInt8](repeating: 0, count: sourceCount)
            var target = [UInt8](repeating: 0, count: targetCount)
            do {
                try source.withUnsafeMutableBytes { sourceBytes in
                    try target.withUnsafeMutableBytes { targetBytes in
                        try AffineCodes.widen(UnsafeRawBufferPointer(sourceBytes), to: targetBytes,
                            count: count, from: sourceBits, to: targetBits, policy: .packed4To6)
                    }
                }
                c.expect(name, false, "invalid geometry was accepted")
            } catch let error as ModelError {
                c.equal(name, error.description, "invalid lossless affine-code widening")
            } catch {
                c.expect(name, false, String(describing: error))
            }
        }
        rejects("negative count is rejected", count: -1, from: 4, to: 6)
        rejects("count multiplication overflow is rejected", count: Int.max / 8 + 1, from: 4, to: 6)
        rejects("partial output byte is rejected", sourceCount: 1, targetCount: 1,
            count: 2, from: 4, to: 6)
        rejects("unsupported source width is rejected", count: 0, from: 2, to: 6)
        rejects("unsupported target width is rejected", count: 0, from: 4, to: 5)
        rejects("narrowing is rejected", count: 0, from: 6, to: 4)
        rejects("wrong source length is rejected", sourceCount: 1, targetCount: 3,
            count: 4, from: 4, to: 6)
        rejects("wrong target length is rejected", sourceCount: 2, targetCount: 2,
            count: 4, from: 4, to: 6)

        for (sourceBits, targetBits) in [(3, 4), (3, 6), (3, 8), (4, 4),
                                         (4, 8), (6, 6), (6, 8), (8, 8)] {
            let count = 8
            let source = (0 ..< count * sourceBits / 8).map {
                UInt8(truncatingIfNeeded: $0 &* 73 &+ sourceBits &* 11 &+ targetBits)
            }
            do {
                let actual = try expand(source, count: count, from: sourceBits, to: targetBits,
                    policy: .packed4To6)
                c.equal("generic fallback \(sourceBits)-to-\(targetBits)", actual,
                    oracle(source, count: count, from: sourceBits, to: targetBits))
            } catch {
                c.expect("generic fallback \(sourceBits)-to-\(targetBits)", false,
                    String(describing: error))
            }
        }

        let realisticCount = 1_638_400
        let realisticSource = (0 ..< realisticCount / 2).map {
            UInt8(truncatingIfNeeded: $0 &* 37 &+ $0 / 97 &+ 11)
        }
        do {
            let scalar = try expand(realisticSource, count: realisticCount, from: 4, to: 6,
                policy: .scalar, fill: 0x00)
            let packed = try expand(realisticSource, count: realisticCount, from: 4, to: 6,
                policy: .packed4To6)
            c.equal("real 1638400-code projection is byte-identical", packed, scalar)
        } catch {
            c.expect("real 1638400-code projection widens", false, String(describing: error))
        }

        do {
            let source: [UInt8] = [0x10, 0x32]
            let implicit = try expand(source, count: 4, from: 4, to: 6, policy: .scalar)
            var explicit = [UInt8](repeating: 0xff, count: 3)
            try source.withUnsafeBytes { sourceBytes in
                try explicit.withUnsafeMutableBytes { targetBytes in
                    try AffineCodes.widen(sourceBytes, to: targetBytes, count: 4, from: 4, to: 6)
                }
            }
            c.equal("omitted policy remains scalar", implicit, explicit)
            c.equal("scalar policy Codable identity",
                String(data: try JSONEncoder().encode(AffineWideningPolicy.scalar), encoding: .utf8),
                "\"scalar\"")
            c.equal("packed policy Codable identity",
                String(data: try JSONEncoder().encode(AffineWideningPolicy.packed4To6), encoding: .utf8),
                "\"packed4-to6\"")
        } catch {
            c.expect("policy defaults and identities are stable", false, String(describing: error))
        }

        c.measure("exhaustive_two_byte_inputs", 65_536)
        c.measure("realistic_projection_codes", Double(realisticCount))
        return c.report()
    }

    public static func wideningSynthetic(pairCount: Int = 8) throws -> WideningSyntheticReport {
        guard pairCount > 0, pairCount <= 100 else { throw ModelError("widening pair count must be 1...100") }
        let check = exactWidening()
        guard check.passed else { throw ModelError("exact widening checks failed") }
        let codeCount = 1_638_400
        let sourceBytes = codeCount * 4 / 8
        let targetBytes = codeCount * 6 / 8
        let maximumWorkers = 16
        let source = (0 ..< maximumWorkers * sourceBytes).map {
            UInt8(truncatingIfNeeded: $0 &* 37 &+ $0 / 97 &+ 11)
        }
        var target = [UInt8](repeating: 0xa5, count: maximumWorkers * targetBytes)
        var samples: [Int: [AffineWideningPolicy: [Double]]] = [:]
        var checksums: [Int: [AffineWideningPolicy: UInt64]] = [:]

        func consume(_ workers: Int) -> UInt64 {
            var digest: UInt64 = 1_469_598_103_934_665_603
            for byte in target.prefix(workers * targetBytes) {
                digest = (digest ^ UInt64(byte)) &* 1_099_511_628_211
            }
            return digest
        }

        func measure(_ policy: AffineWideningPolicy, workers: Int) throws -> (Double, UInt64) {
            let failure = WideningFailure()
            let started = RuntimeClock.now()
            source.withUnsafeBytes { sourceBuffer in
                target.withUnsafeMutableBytes { targetBuffer in
                    DispatchQueue.concurrentPerform(iterations: workers) { worker in
                        let input = UnsafeRawBufferPointer(
                            start: sourceBuffer.baseAddress! + worker * sourceBytes, count: sourceBytes)
                        let output = UnsafeMutableRawBufferPointer(
                            start: targetBuffer.baseAddress! + worker * targetBytes, count: targetBytes)
                        do {
                            try AffineCodes.widen(input, to: output, count: codeCount,
                                from: 4, to: 6, policy: policy)
                        } catch { failure.record(error) }
                    }
                }
            }
            let seconds = RuntimeClock.seconds(since: started) / Double(workers)
            try failure.finish()
            return (seconds, consume(workers))
        }

        for workers in [1, 8, 16] {
            samples[workers] = [:]
            checksums[workers] = [:]
            for pair in 0 ..< pairCount {
                let order: [AffineWideningPolicy] = pair.isMultiple(of: 2)
                    ? [.scalar, .packed4To6] : [.packed4To6, .scalar]
                for policy in order {
                    let (seconds, checksum) = try measure(policy, workers: workers)
                    samples[workers]![policy, default: []].append(seconds)
                    if let prior = checksums[workers]![policy], prior != checksum {
                        throw ModelError("widening benchmark checksum changed between identical runs")
                    }
                    checksums[workers]![policy] = checksum
                }
            }
            guard checksums[workers]![.scalar] == checksums[workers]![.packed4To6] else {
                throw ModelError("scalar and packed widening benchmark outputs differ")
            }
        }

        func median(_ values: [Double]) -> Double {
            let ordered = values.sorted()
            let middle = ordered.count / 2
            return ordered.count.isMultiple(of: 2)
                ? (ordered[middle - 1] + ordered[middle]) / 2 : ordered[middle]
        }
        var results: [WideningBenchmarkResult] = []
        for workers in [1, 8, 16] {
            for policy in [AffineWideningPolicy.scalar, .packed4To6] {
                let durations = samples[workers]![policy]!
                results.append(WideningBenchmarkResult(workers: workers, policy: policy,
                    secondsPerCall: durations, medianSecondsPerCall: median(durations),
                    checksum: checksums[workers]![policy]!))
            }
        }
        return WideningSyntheticReport(
            method: "balanced alternating scalar/packed pairs; one fixed 1638400-code projection per worker; reusable disjoint buffers; full output FNV-1a consumed outside timing",
            codeCountPerCall: codeCount, pairCount: pairCount,
            allocatedBufferBytes: maximumWorkers * (sourceBytes + targetBytes),
            results: results, exactCheck: check,
            processFootprintEndBytes: ProcessMemory.residentBytes())
    }

    public static func wideningModelComponent(modelDir: URL) throws -> WideningModelReport {
        MLX.Memory.cacheLimit = 64 << 20
        let sampler = FootprintSampler()
        let diskBefore = diskBytesRead()
        let index = try CheckpointIndex(dir: modelDir)
        guard index.config.format == .jang6S else {
            throw ModelError("widening model component requires the pinned JANG_6S checkpoint")
        }
        let scalarStore = try ExpertStore(index: index, wideningPolicy: .scalar)
        let packedStore = try ExpertStore(index: index, wideningPolicy: .packed4To6)
        guard scalarStore.effectiveWideningPolicy == .scalar,
              packedStore.effectiveWideningPolicy == .packed4To6 else {
            throw ModelError("expert store did not retain its immutable widening policy")
        }
        var packedLayoutRejected = false
        do {
            _ = try packedStore.loadPackedLayout(at: URL(fileURLWithPath: "/nonexistent-widening-layout"))
        } catch let error as ModelError {
            packedLayoutRejected = error.description.contains("cannot be combined")
        }
        let controls = scalarStore.configureWideningReadControls()
        guard controls.allSatisfy({ $0.noCacheReturnCode == 0 && $0.readAheadReturnCode == 0 }) else {
            throw ModelError("original-reader descriptor controls could not be applied")
        }
        let identityBefore = try scalarStore.wideningSourceIdentity()
        guard identityBefore == (try packedStore.wideningSourceIdentity()) else {
            throw ModelError("scalar and packed stores do not share one source identity")
        }
        let experts = Array(0 ... 9)
        let layers = [0, 5, 22]
        let uniqueBytes = layers.reduce(0) { $0 + scalarStore.sourceRecordBytes(layer: $1) * experts.count }
        let sourceLimit = 256 << 20
        guard uniqueBytes < sourceLimit else {
            throw ModelError("widening component original source regions exceed 256 MiB")
        }
        let expectedClasses = [0: 2, 5: 1, 22: 0]
        var cases: [WideningReaderCase] = []
        var check = CheckBuilder("widening-model-component")
        check.expect("packed widening rejects a bypassing packed expert layout", packedLayoutRejected)

        enum ReaderAPI: String { case batch = "readBatchChecked", runs = "readRunsChecked" }
        func read(_ store: ExpertStore, api: ReaderAPI, layer: Int) throws -> [MLXArray] {
            let queueDepth = api == .batch ? ExpertStore.poolQueueDepth : ExpertStore.defaultQueueDepth
            switch api {
            case .batch:
                return try store.readBatchChecked(experts.map { ExpertKey(layer, $0) },
                    queueDepth: queueDepth)
            case .runs:
                return try store.readRunsChecked(layer: layer, experts: experts,
                    queueDepth: queueDepth)
            }
        }
        func timedRead(_ store: ExpertStore, api: ReaderAPI, layer: Int) throws -> ([MLXArray], Double) {
            let started = RuntimeClock.now()
            let output = try read(store, api: api, layer: layer)
            return (output, RuntimeClock.seconds(since: started))
        }
        func compare(_ scalar: [MLXArray], _ packed: [MLXArray]) -> (Bool, Int, String) {
            guard scalar.count == packed.count else { return (false, 0, "") }
            var exact = true
            var compared = 0
            var digest = SHA256()
            for piece in scalar.indices {
                guard scalar[piece].shape == packed[piece].shape,
                      scalar[piece].dtype == packed[piece].dtype else {
                    exact = false
                    continue
                }
                let scalarBytes = scalar[piece].reshaped([-1]).view(dtype: .uint8).asArray(UInt8.self)
                let packedBytes = packed[piece].reshaped([-1]).view(dtype: .uint8).asArray(UInt8.self)
                compared += scalarBytes.count
                if scalarBytes != packedBytes { exact = false }
                digest.update(data: Data(scalarBytes))
            }
            return (exact, compared, digest.finalize().map { String(format: "%02x", $0) }.joined())
        }

        func median(_ values: [Double]) -> Double {
            let ordered = values.sorted()
            let middle = ordered.count / 2
            return ordered.count.isMultiple(of: 2)
                ? (ordered[middle - 1] + ordered[middle]) / 2 : ordered[middle]
        }

        let readerPairs = 4
        var sequence = 0
        for api in [ReaderAPI.batch, .runs] {
            for layer in layers {
                var orders: [[AffineWideningPolicy]] = []
                var scalarDurations: [Double] = []
                var packedDurations: [Double] = []
                var exact = true
                var comparedBytes = 0
                var outputDigest: String?
                for pair in 0 ..< readerPairs {
                    let order: [AffineWideningPolicy] = (sequence + pair).isMultiple(of: 2)
                        ? [.scalar, .packed4To6] : [.packed4To6, .scalar]
                    orders.append(order)
                    var scalarOutput: [MLXArray] = []
                    var packedOutput: [MLXArray] = []
                    for policy in order {
                        if policy == .scalar {
                            let result = try timedRead(scalarStore, api: api, layer: layer)
                            scalarOutput = result.0
                            scalarDurations.append(result.1)
                        } else {
                            let result = try timedRead(packedStore, api: api, layer: layer)
                            packedOutput = result.0
                            packedDurations.append(result.1)
                        }
                    }
                    let comparison = compare(scalarOutput, packedOutput)
                    exact = exact && comparison.0
                    if comparedBytes == 0 { comparedBytes = comparison.1 }
                    else if comparedBytes != comparison.1 { exact = false }
                    if let outputDigest, outputDigest != comparison.2 { exact = false }
                    else { outputDigest = comparison.2 }
                    if ProcessMemory.residentBytes() > 512_000_000 {
                        throw ModelError("widening component exceeded its 512 MB physical footprint limit")
                    }
                }
                sequence += 1
                let observed = scalarStore.wideningProjectionCount(layer: layer)
                check.expect("\(api.rawValue) layer \(layer) exact full tensor bytes", exact)
                check.equal("\(api.rawValue) layer \(layer) widening class", observed,
                    expectedClasses[layer]!)
                let queueDepth = api == .batch ? ExpertStore.poolQueueDepth : ExpertStore.defaultQueueDepth
                cases.append(WideningReaderCase(api: api.rawValue, queueDepth: queueDepth, layer: layer,
                    experts: experts, expectedWideningProjections: expectedClasses[layer]!,
                    observedWideningProjections: observed, pairOrders: orders,
                    scalarSeconds: scalarDurations, packedSeconds: packedDurations,
                    scalarMedianSeconds: median(scalarDurations),
                    packedMedianSeconds: median(packedDurations),
                    comparedBytes: comparedBytes, exactBytes: exact,
                    outputSHA256: outputDigest ?? ""))
            }
        }
        let identityAfter = try scalarStore.wideningSourceIdentity()
        check.equal("checkpoint identity is unchanged", identityAfter, identityBefore)
        let footprint = sampler.finish()
        let footprintLimit: UInt64 = 512_000_000
        check.expect("component physical footprint stays within 512 MiB",
            footprint.peakBytes <= footprintLimit,
            "sampled peak \(footprint.peakBytes) bytes")
        check.expect("all descriptor controls are observed successful",
            controls.allSatisfy { $0.noCacheReturnCode == 0 && $0.readAheadReturnCode == 0 })
        check.measure("unique_original_region_bytes", Double(uniqueBytes))
        check.measure("sampled_physical_footprint_peak_bytes", Double(footprint.peakBytes))
        let logicalSourceBytes = uniqueBytes * 2 * readerPairs * 2
        check.measure("logical_source_bytes_read", Double(logicalSourceBytes))
        return WideningModelReport(modelPath: modelDir.path,
            checkpointFormat: index.config.format, policies: [.scalar, .packed4To6],
            sourceIdentityBefore: identityBefore, sourceIdentityAfter: identityAfter,
            uniqueOriginalRegionBytes: uniqueBytes, uniqueOriginalRegionLimitBytes: sourceLimit,
            logicalSourceBytesRead: logicalSourceBytes,
            physicalFootprintLimitBytes: footprintLimit, footprint: footprint,
            diskBytesReadBefore: diskBefore, diskBytesReadAfter: diskBytesRead(),
            diskObservationScope: "proc_pid_rusage RUSAGE_INFO_V4 process disk bytes; may cover only part of logical reads and does not establish a cold-SSD run",
            readControls: controls,
            timingScope: "whole ExpertStore checked reader call: staging allocation, original pread, exact code widening and metadata conversion, MLX wrapping/evaluation; full-byte verification excluded",
            verificationScope: "all nine returned tensors for experts 0...9 at layers 0, 5 and 22; both checked batch and contiguous-runs APIs",
            cases: cases, check: check.report())
    }

    private static func diskBytesRead() -> UInt64? {
        typealias ProcPIDRUsage = @convention(c) (
            pid_t, Int32, UnsafeMutableRawPointer
        ) -> Int32
        guard let library = dlopen("/usr/lib/libproc.dylib", RTLD_NOW) else { return nil }
        defer { dlclose(library) }
        guard let symbol = dlsym(library, "proc_pid_rusage") else { return nil }
        let call = unsafeBitCast(symbol, to: ProcPIDRUsage.self)
        var usage = rusage_info_v4()
        let result = withUnsafeMutablePointer(to: &usage) {
            call(getpid(), RUSAGE_INFO_V4, UnsafeMutableRawPointer($0))
        }
        return result == 0 ? usage.ri_diskio_bytesread : nil
    }
}
