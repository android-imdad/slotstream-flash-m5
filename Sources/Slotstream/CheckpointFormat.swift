import Foundation

/// The supported checkpoint encodings, independent of the shared model graph.
public enum CheckpointFormat: String, Sendable, Codable {
    case pipe4, jang4M, jang6S
    public var isJANG: Bool { self != .pipe4 }
    public var modelName: String {
        switch self {
        case .pipe4: return PinnedModel.name
        case .jang4M: return "qwen3.8-flash-next:jang-4m"
        case .jang6S: return "qwen3.8-flash-next:jang-6s"
        }
    }
    /// JANG_6S uses a uniform six-bit cache. Four-bit source codes are widened
    /// exactly on a miss: scales, biases and represented weights are unchanged.
    public var expertBits: Int { self == .jang6S ? 6 : 4 }
    // JANG metadata is losslessly expanded from FP16 to FP32 in the cache.
    // Otherwise MLX promotes the *whole* pool's metadata on every BF16 QMM.
    public var expertRecordBytes: Int { 3 * (640 * 2560 * expertBits / 8 + 640 * 2560 / 64 * (isJANG ? 8 : 4)) }
}

public struct AffineQuantization: Equatable, Sendable {
    public let bits: Int
    public let groupSize: Int
    public init(bits: Int, groupSize: Int) { self.bits = bits; self.groupSize = groupSize }
}

/// Map JANG's MLX-VLM module paths to Slotstream's internal text graph.
/// This is also applied to quantization overrides, never just tensor names.
public enum CheckpointNames {
    public static func canonical(_ raw: String, format: CheckpointFormat) -> String {
        var name = raw
        if name.hasPrefix("language_model.") { name.removeFirst("language_model.".count) }
        guard format.isJANG else { return name }
        if name.hasPrefix("layers.") || name.hasPrefix("embed_tokens.") || name.hasPrefix("hyper_connection_mixer.") {
            name = "model." + name
        }
        if name.hasPrefix("visual.") { name = "vision_tower." + name.dropFirst("visual.".count) }
        if name.contains(".ple.") {
            name = name.replacingOccurrences(of: ".ple.conv1d_weight", with: ".ple.conv1d.weight")
            for member in ["ngram_embedding", "layer_multipliers", "ngram_heads_offsets", "ngram_heads_vocab_sizes"] {
                name = name.replacingOccurrences(of: ".ple." + member, with: ".ple.ple_embedding." + member)
            }
            name = name.replacingOccurrences(of: "ngram_embedding.shards.", with: "ngram_embedding.shard_")
        }
        return name
    }
}
