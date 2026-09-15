import Foundation
import Dispatch

struct ModelError: Error { let message: String; init(_ message: String) { self.message = message } }
func bf16ToFloat(_ bits: UInt16) -> Float { Float(bitPattern: UInt32(bits) << 16) }
func bf16Round(_ value: Float) -> Float { Float(bitPattern: value.bitPattern & 0xffff0000) }

@inline(never) func oldPacked(_ source: UnsafeRawBufferPointer, _ target: UnsafeMutableRawBufferPointer) throws {
    try BaselineAffineCodes.widen(source, to: target, count: source.count * 2, from: 4, to: 6, policy: .packed4To6)
}
@inline(never) func newPacked(_ source: UnsafeRawBufferPointer, _ target: UnsafeMutableRawBufferPointer) throws {
    try AffineCodes.widen(source, to: target, count: source.count * 2, from: 4, to: 6, policy: .packed4To6)
}

@main struct Bench {
    static func main() throws {
        let before = ProcessInfo.processInfo.thermalState.rawValue
        var source = (0..<819200).map { UInt8(truncatingIfNeeded: $0 &* 37 &+ $0 / 97) }
        var old = [UInt8](repeating: 0, count: source.count * 3 / 2)
        var new = old
        var oldTimes: [Double] = [], newTimes: [Double] = []
        var checksum: UInt64 = 0
        try source.withUnsafeMutableBytes { input in
            try old.withUnsafeMutableBytes { a in
                try new.withUnsafeMutableBytes { b in
                    try oldPacked(UnsafeRawBufferPointer(input), a)
                    try newPacked(UnsafeRawBufferPointer(input), b)
                    precondition(memcmp(a.baseAddress!, b.baseAddress!, a.count) == 0)
                    for round in 0..<9 {
                        for mode in (round % 2 == 0 ? [0, 1] : [1, 0]) {
                            let target = mode == 0 ? a : b
                            input[0] = 0
                            let start = DispatchTime.now().uptimeNanoseconds
                            for _ in 0..<200 {
                                input[0] &+= 1
                                if mode == 0 { try oldPacked(UnsafeRawBufferPointer(input), target) }
                                else { try newPacked(UnsafeRawBufferPointer(input), target) }
                                checksum &+= UInt64(target[0])
                            }
                            let ns = Double(DispatchTime.now().uptimeNanoseconds - start) / 200
                            if mode == 0 { oldTimes.append(ns) } else { newTimes.append(ns) }
                        }
                        precondition(memcmp(a.baseAddress!, b.baseAddress!, a.count) == 0)
                    }
                }
            }
        }
        let medianOld = oldTimes.sorted()[oldTimes.count/2]
        let medianNew = newTimes.sorted()[newTimes.count/2]
        let report: [String: Any] = ["format": "packed-swift-vs-neon-component-v1", "source_bytes": source.count,
            "codes": source.count * 2, "pairs": 9, "calls_per_arm": 200,
            "old_packed_ns_per_call": oldTimes, "neon_ns_per_call": newTimes,
            "old_median_ns": medianOld, "neon_median_ns": medianNew, "component_ratio": medianOld/medianNew,
            "checksum": checksum, "exact_bytes": true, "thermal_before": before,
            "thermal_after": ProcessInfo.processInfo.thermalState.rawValue,
            "low_power": ProcessInfo.processInfo.isLowPowerModeEnabled,
            "scope": "Standalone production widening code; baseline identifiers renamed only. Helpers for unused row-decode symbols are not exercised. No SSD, MLX or inference measurement."]
        print(String(data: try JSONSerialization.data(withJSONObject: report, options: [.prettyPrinted,.sortedKeys]), encoding: .utf8)!)
    }
}
