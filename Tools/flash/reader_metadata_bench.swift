// Standalone comparator: compile with -O and link the production affine.c.
import Foundation
@_silgen_name("slotstream_affine_fp16_to_fp32")
func native(_ source: UnsafeRawPointer, _ target: UnsafeMutableRawPointer, _ count: Int)
@inline(never)
func previous(_ bytes: UnsafeRawBufferPointer, _ output: UnsafeMutableRawPointer) {
    let floats = output.assumingMemoryBound(to: Float.self)
    for i in 0 ..< bytes.count / 2 {
        let bits = UInt16(bytes[2 * i]) | UInt16(bytes[2 * i + 1]) << 8
        floats[i] = Float(Float16(bitPattern: bits))
    }
}
let size = 65536
let input = UnsafeMutableRawPointer.allocate(byteCount: size * 2, alignment: 16)
let a = UnsafeMutableRawPointer.allocate(byteCount: size * 4, alignment: 16)
let b = UnsafeMutableRawPointer.allocate(byteCount: size * 4, alignment: 16)
defer { input.deallocate(); a.deallocate(); b.deallocate() }
for i in 0 ..< size { input.storeBytes(of: UInt16(i), toByteOffset: i * 2, as: UInt16.self) }
previous(UnsafeRawBufferPointer(start: input, count: size * 2), a)
native(input, b, size)
precondition(memcmp(a, b, size * 4) == 0, "FP16 exhaustive bits mismatch")
var samples = [[String: Any]]()
var checksum: UInt64 = 0
let count = 25600, repetitions = 2000
for pair in 0 ..< 12 {
    for mode in (pair.isMultiple(of: 2) ? ["previous", "native"] : ["native", "previous"]) {
        let start = DispatchTime.now().uptimeNanoseconds
        for _ in 0 ..< repetitions {
            if mode == "previous" { previous(UnsafeRawBufferPointer(start: input, count: count * 2), a) }
            else { native(input, a, count) }
            checksum &+= UInt64(a.load(fromByteOffset: (pair * 71) * 4, as: UInt32.self))
        }
        let elapsed = Double(DispatchTime.now().uptimeNanoseconds - start) / 1e9 / Double(repetitions)
        samples.append(["pair": pair, "mode": mode, "seconds_per_piece": elapsed])
    }
}
let report: [String: Any] = ["count": count, "repetitions": repetitions,
    "exhaustive_fp16_bits_exact": true, "checksum": checksum, "samples": samples]
print(String(data: try JSONSerialization.data(withJSONObject: report, options: [.prettyPrinted, .sortedKeys]), encoding: .utf8)!)
