import Foundation

/// CPU row decoding with the checkpoint's original scale/output precision.
/// Independent from the I/O and cache so real byte samples can be checked
/// against MLX without loading a multi-gigabyte table.
package enum AffineRow {
    package static func decode(weights: [UInt8], scales: [UInt8], biases: [UInt8],
                               columns: Int, bits: Int, group: Int, fp16: Bool) throws -> [Float] {
        guard columns > 0, columns <= 65536, [3, 4, 6, 8].contains(bits),
              [32, 64, 128].contains(group), columns % group == 0,
              weights.count == columns * bits / 8, scales.count == columns / group * 2,
              biases.count == scales.count else { throw ModelError("invalid affine row geometry") }
        return weights.withUnsafeBytes { packed in
            (0 ..< columns).map { i in
                let at = i / group * 2
                let s = UInt16(scales[at]) | UInt16(scales[at + 1]) << 8
                let b = UInt16(biases[at]) | UInt16(biases[at + 1]) << 8
                let q = Float(AffineCodes.value(packed, index: i, bits: bits))
                if fp16 { return Float(Float16(Float(Float16(bitPattern: s)) * q + Float(Float16(bitPattern: b)))) }
                return bf16Round(bf16ToFloat(s) * q + bf16ToFloat(b))
            }
        }
    }
    package static func compact(_ row: [Float], fp16: Bool) -> [UInt16] {
        fp16 ? row.map { Float16($0).bitPattern } : row.map { UInt16(truncatingIfNeeded: $0.bitPattern >> 16) }
    }
}

/// MLX affine codes form a little-endian bitstream, including codes crossing
/// byte/word boundaries at three and six bits. No floating-point quantization.
package enum AffineCodes {
    package static func value(_ bytes: UnsafeRawBufferPointer, index: Int, bits: Int) -> UInt8 {
        let bit = index * bits, byte = bit / 8, shift = bit % 8
        var word = UInt16(bytes[byte])
        if shift + bits > 8 { word |= UInt16(bytes[byte + 1]) << 8 }
        return UInt8((word >> shift) & UInt16((1 << bits) - 1))
    }

    package static func widen(_ source: UnsafeRawBufferPointer, to target: UnsafeMutableRawBufferPointer,
                              count: Int, from bits: Int, to targetBits: Int) throws {
        guard [3, 4, 6, 8].contains(bits), [4, 6, 8].contains(targetBits), targetBits >= bits,
              count >= 0, count <= Int.max / 8, count * bits % 8 == 0, count * targetBits % 8 == 0,
              source.count == count * bits / 8, target.count == count * targetBits / 8 else {
            throw ModelError("invalid lossless affine-code widening")
        }
        target.initializeMemory(as: UInt8.self, repeating: 0)
        for i in 0 ..< count {
            let q = UInt16(value(source, index: i, bits: bits))
            let bit = i * targetBits, byte = bit / 8, shift = bit % 8
            target[byte] |= UInt8(truncatingIfNeeded: q << shift)
            if shift + targetBits > 8 { target[byte + 1] |= UInt8(q >> (8 - shift)) }
        }
    }
}
