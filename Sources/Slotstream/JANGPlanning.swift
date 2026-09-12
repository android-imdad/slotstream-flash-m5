import Foundation

/// Header-derived allocation geometry. A checkpoint selection never changes
/// global constants or another engine's plan.
public struct CheckpointMemory: Equatable, Sendable {
    public let format: CheckpointFormat
    public let residentBytes: Int
    public var recordBytes: Int { format.expertRecordBytes }
    public var fixedBytes: Int {
        // Conservative initial allowance for runtime, allocator, bounded read
        // staging and code widening. This is a budget, not a measured peak.
        max(PlannerCostModel.fixedBytes, ContextBytes.sum(residentBytes, 3_000_000_000))
    }
    public init(format: CheckpointFormat, residentBytes: Int) {
        self.format = format; self.residentBytes = residentBytes
    }

    /// Fixed target until each checkpoint's governor and performance envelope
    /// have been qualified. The OS-pressure and live allocation guards remain.
    public func plan(memoryGB: Double?, expertsPerLayer: Int? = nil, poolGB: Double? = nil,
                     on device: Machine = .current(), maxContext: Int = ContextPolicy.defaultTokens,
                     ramPercent: Double = 70,
                     policy: RuntimeAllocationPolicy = try! .init()) throws -> MemoryPlan {
        _ = try ContextConfiguration(maxContextTokens: maxContext)
        guard format.isJANG, residentBytes > 0, residentBytes < 32_000_000_000,
              device.ramGB.isFinite, device.workingSetGB.isFinite,
              let available = device.availableGB, available.isFinite,
              ramPercent.isFinite, ramPercent > 0, ramPercent <= 100 else {
            throw PlanError("JANG planning requires finite live memory observations")
        }
        let limit = min(device.ramGB, device.workingSetGB - 2, available - 3)
        let target = memoryGB ?? min(33, device.ramGB * ramPercent / 100, limit)
        guard target.isFinite, target > 0, target <= limit else {
            throw PlanError("JANG memory target must fit current RAM, Metal working set and reclaimable headroom")
        }
        let chunk = policy.prefillChunkOverride ?? 256
        let retention = policy.prefixCacheEnabled ? min(4096, maxContext) : 0
        let empty = ContextMemoryLedger(slots: 0, context: maxContext, chunk: chunk,
            retentionTokens: retention, mtp: false, visionResident: false, checkpoint: self)
        let availablePool = target * 1e9 - Double(empty.expectedPeakBytes) - 1e9
        guard availablePool >= Double(Geometry.floorSlots * recordBytes) else {
            throw PlanError("JANG target cannot hold its resident weights, runtime and minimum expert cache")
        }
        var slots = min(Geometry.totalRecords, Int(min(availablePool / Double(recordBytes), Double(Geometry.totalRecords))))
        var source: MemoryPlan.Source = .memoryGB
        if let n = expertsPerLayer {
            guard (1...Geometry.expertsPerLayer).contains(n) else { throw PlanError("invalid expert cache size") }
            slots = n * Geometry.layers; source = .expertsPerLayer
        } else if let gb = poolGB {
            guard gb.isFinite, gb > 0, gb <= target else { throw PlanError("invalid expert pool budget") }
            slots = min(Geometry.totalRecords, Int(gb * 1e9 / Double(recordBytes))); source = .poolGB
        }
        guard slots >= Geometry.floorSlots, Double(slots * recordBytes) <= availablePool else {
            throw PlanError("explicit JANG expert cache does not fit the checked total memory target")
        }
        return MemoryPlan(source: source, slots: slots, targetGB: target,
            ramGB: device.ramGB, workingSetGB: device.workingSetGB, ramPercent: ramPercent,
            availableGB: available, clamped: false, prefillChunk: chunk,
            prefixCacheTokens: retention, maxContextTokens: maxContext,
            notes: ["experimental JANG text inference; throughput and full-model memory envelope are unqualified",
                    "native routing and quantized values preserved; fixed cache; MTP and vision disabled"],
            simulated: device.isSimulated, runtimeAllocationPolicy: policy, checkpointMemory: self)
    }
}
