import Darwin
import Foundation
import Slotstream

public extension Diagnostics {
    static func flashIdentity() throws -> CheckReport {
        var c = CheckBuilder("flash-identity")
        let ordinary = ContextMemoryLedger(slots: 640, context: 2048, chunk: 256,
                                           retentionTokens: 2048, mtp: false, visionResident: false)
        let explicitZero = ContextMemoryLedger(slots: 640, context: 2048, chunk: 256,
                                               retentionTokens: 2048, mtp: false, visionResident: false,
                                               diagnosticReservedBytes: 0)
        c.equal("default-zero diagnostic reservation preserves ledger JSON",
                ordinary.json as NSDictionary, explicitZero.json as NSDictionary)
        c.equal("default-zero diagnostic reservation preserves expected peak",
                ordinary.expectedPeakBytes, explicitZero.expectedPeakBytes)
        let reserved = ContextMemoryLedger(slots: 640, context: 2048, chunk: 256,
                                           retentionTokens: 2048, mtp: false, visionResident: false,
                                           diagnosticReservedBytes: 128 << 20)
        c.equal("diagnostic reservation is charged exactly", reserved.expectedPeakBytes,
                ordinary.expectedPeakBytes + (128 << 20))
        c.equal("diagnostic reservation is explicit in nonzero JSON",
                reserved.json["diagnostic_reserved_bytes"] as? Int, 128 << 20)
        c.equal("negative direct ledger reservation uses refusal sentinel",
                ContextMemoryLedger(slots: 1, context: 1, chunk: 1, retentionTokens: 0,
                                    mtp: false, visionResident: false, diagnosticReservedBytes: -1).diagnosticReservedBytes,
                Int.max)
        let policy = try RuntimeAllocationPolicy(prefixCacheEnabled: false,
                                                 diagnosticReservedBytes: 128 << 20)
        let checkpoint = CheckpointMemory(format: .jang6S, residentBytes: 5_858_794_504)
        let plan = try checkpoint.plan(memoryGB: 14,
                                       on: Machine(ramGB: 48, workingSetGB: 40, availableGB: 25, isSimulated: true),
                                       maxContext: 2048, policy: policy)
        c.expect("reserved JANG capture retains the 640-slot floor", plan.slots >= 640)
        c.equal("reserved JANG plan carries ledger charge",
                plan.memoryLedger.diagnosticReservedBytes, 128 << 20)
        let unresolved = MemoryPlan(source: plan.source, slots: plan.slots, targetGB: plan.targetGB,
                                    ramGB: plan.ramGB, workingSetGB: plan.workingSetGB,
                                    ramPercent: plan.ramPercent, availableGB: plan.availableGB,
                                    clamped: plan.clamped, prefillChunk: plan.prefillChunk,
                                    prefixCacheTokens: plan.prefixCacheTokens, notes: plan.notes,
                                    checkpointMemory: checkpoint)
        do {
            _ = try Planner.applyingRuntimePolicy(unresolved, policy: policy)
            c.expect("late diagnostic reservation is refused instead of spending the margin", false)
        } catch {
            c.expect("late diagnostic reservation is refused instead of spending the margin", true)
        }
        let reapplied = try Planner.applyingRuntimePolicy(plan, policy: policy)
        c.equal("matching fresh-plan policy is accepted without double charge",
                reapplied.memoryLedger.expectedPeakBytes, plan.memoryLedger.expectedPeakBytes)
        return c.report()
    }

    static func flashObservation() throws -> CheckReport {
        var c = CheckBuilder("flash-observation")
        try c.equal("resident and missing rank split restores canonical order",
                    FlashRankMapping.inverse(parts: [[0, 3, 7], [1, 2, 4, 5, 6, 8, 9]], topK: 10),
                    [0, 3, 4, 1, 5, 6, 7, 2, 8, 9])
        do {
            _ = try FlashRankMapping.inverse(parts: [[0, 1], [1, 2]], topK: 4)
            c.expect("duplicate or missing ranks are refused", false)
        } catch { c.expect("duplicate or missing ranks are refused", true) }
        let absent = FlashStateField(name: "mtp.key", present: false)
        let data = try JSONEncoder().encode(absent)
        c.expect("absent state field is explicit and serializable",
                 try JSONDecoder().decode(FlashStateField.self, from: data) == absent)
        let root = FileManager.default.temporaryDirectory.appendingPathComponent("flash-writer-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: root, withIntermediateDirectories: false)
        defer { try? FileManager.default.removeItem(at: root) }
        var calls = 0
        let short = FlashBoundedWriter(maxBytes: 16, liveLimit: 16, writeCall: { fd, p, n in
            calls += 1; return Darwin.write(fd, p, min(1, n))
        })
        let shortPath = root.appendingPathComponent("short.bin")
        try short.write(Data([1, 2, 3, 4]), to: shortPath.path)
        let shortExact = try Data(contentsOf: shortPath) == Data([1, 2, 3, 4])
        c.expect("positive short writes drain completely", calls == 4 && shortExact)
        let quota = FlashBoundedWriter(maxBytes: 2, liveLimit: 2)
        do { try quota.write(Data([1, 2, 3]), to: root.appendingPathComponent("quota.bin").path); c.expect("quota rejects before file creation", false) }
        catch { c.expect("quota rejects before file creation",!FileManager.default.fileExists(atPath: root.appendingPathComponent("quota.bin").path)) }
        let cancelled = FlashBoundedWriter(maxBytes: 4); cancelled.cancel()
        do { try cancelled.write(Data([1]), to: root.appendingPathComponent("cancel.bin").path); c.expect("cancel rejects before file creation", false) }
        catch { c.expect("cancel rejects before file creation",!FileManager.default.fileExists(atPath: root.appendingPathComponent("cancel.bin").path)) }
        let entered = DispatchSemaphore(value: 0), cancelledDuringWrite = DispatchSemaphore(value: 0)
        var concurrent: FlashBoundedWriter!; var concurrentCalls = 0
        concurrent = FlashBoundedWriter(maxBytes: 4, liveLimit: 4, writeCall: { fd, p, n in
            concurrentCalls += 1
            if concurrentCalls == 1 { entered.signal(); cancelledDuringWrite.wait() }
            return Darwin.write(fd, p, min(1, n))
        })
        DispatchQueue.global().async { entered.wait(); concurrent.cancel(); cancelledDuringWrite.signal() }
        do { try concurrent.write(Data([1, 2, 3, 4]), to: root.appendingPathComponent("concurrent.bin").path); c.expect("concurrent cancellation stops a short-write loop", false) }
        catch { c.expect("concurrent cancellation stops a short-write loop", concurrentCalls == 1) }
        let collision = root.appendingPathComponent("collision.bin"); try Data([9]).write(to: collision)
        do { try FlashBoundedWriter(maxBytes: 4).write(Data([1]), to: collision.path); c.expect("existing output is never overwritten", false) }
        catch { let preserved = try Data(contentsOf: collision) == Data([9]); c.expect("existing output is never overwritten", preserved) }
        return c.report()
    }
}
