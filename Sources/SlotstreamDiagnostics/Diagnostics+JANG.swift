import Foundation
import MLX
import Slotstream

extension Diagnostics {
    private static var jangFixtures: URL {
        URL(fileURLWithPath: #filePath).deletingLastPathComponent().deletingLastPathComponent()
            .deletingLastPathComponent().appendingPathComponent("Tools/fixtures/jang")
    }

    public static func jangFormats() throws -> CheckReport {
        var c = CheckBuilder("jang-formats")
        let temporary = FileManager.default.temporaryDirectory.appendingPathComponent("slotstream-jang-" + UUID().uuidString)
        try FileManager.default.createDirectory(at: temporary, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: temporary) }
        for (tier, format) in [("4M", CheckpointFormat.jang4M), ("6S", .jang6S)] {
            let data = try Data(contentsOf: jangFixtures.appendingPathComponent("config-\(tier).json"))
            try data.write(to: temporary.appendingPathComponent("config.json"))
            let cfg = try ModelConfig.load(from: temporary)
            c.equal("\(tier): correct profile", cfg.format, format)
            c.equal("\(tier): original ten-expert routing", cfg.topK, 10)
            c.equal("\(tier): resident and expert precision independent", cfg.qBits, 8)
            c.equal("\(tier): expert pool width", cfg.expertBits, tier == "6S" ? 6 : 4)
            c.equal("\(tier): embedding override", cfg.quantization(for: "model.embed_tokens"), .init(bits: 8, groupSize: 64))
            c.equal("\(tier): n-gram override canonicalized", cfg.quantization(for: "model.layers.1.ple.ple_embedding.ngram_embedding.shard_0"), .init(bits: 3, groupSize: 32))
            c.equal("\(tier): down precision", cfg.quantization(for: "model.layers.0.mlp.switch_mlp.down_proj").bits, tier == "6S" ? 6 : 4)
            let layout = CheckpointMemory(format: format, residentBytes: 5_858_794_504)
            let device = Machine(ramGB: 48, workingSetGB: 38, availableGB: 35, isSimulated: true)
            let p = try layout.plan(memoryGB: 20, on: device)
            c.expect("\(tier): safety margin stays outside peak", p.expectedPeakGB + 1 <= 20)
            c.equal("\(tier): actual slot bytes priced", p.memoryLedger.poolBytes, p.slots * format.expertRecordBytes)
            c.expect("\(tier): measured legacy speed not reused", p.estWarmTokS.isNaN)
            c.expect("\(tier): no unqualified MTP/vision", !p.mtpEnabled && !p.visionEnabled)
            c.expect("\(tier): simulation remains non-loadable", p.simulated)
            c.expect("\(tier): JSON contains no NaN", JSONSerialization.isValidJSONObject(p.json()))
            let bounded = try p.withRequestPolicy(.init(maxContextTokens: p.maxContextTokens))
            c.equal("\(tier): request settings preserve layout", bounded.checkpointMemory, layout)
            for target in [0.0, -1, 8, 35, .nan, .infinity] {
                do { _ = try layout.plan(memoryGB: target, on: device); c.expect("\(tier): reject target \(target)", false) }
                catch { c.expect("\(tier): reject target \(target)", true) }
            }
            do { _ = try layout.plan(memoryGB: 20, expertsPerLayer: 512, on: device); c.expect("oversized explicit pool", false) }
            catch { c.expect("oversized explicit pool", true) }
            var bad = try JSONSerialization.jsonObject(with: data) as! [String: Any]
            var text = bad["text_config"] as! [String: Any]
            text["num_experts_per_tok"] = 4; bad["text_config"] = text
            try JSONSerialization.data(withJSONObject: bad).write(to: temporary.appendingPathComponent("config.json"))
            do { _ = try ModelConfig.load(from: temporary); c.expect("native routing change refused", false) }
            catch { c.expect("native routing change refused", true) }
            bad = try JSONSerialization.jsonObject(with: data) as! [String: Any]
            var jang = bad["jang_config"] as! [String: Any]
            jang["norm_convention"] = "unknown"; bad["jang_config"] = jang
            try JSONSerialization.data(withJSONObject: bad).write(to: temporary.appendingPathComponent("config.json"))
            do { _ = try ModelConfig.load(from: temporary); c.expect("unknown norm convention refused", false) }
            catch { c.expect("unknown norm convention refused", true) }
        }
        let mappings = [
            "language_model.layers.1.ple.conv1d_weight": "model.layers.1.ple.conv1d.weight",
            "language_model.layers.1.ple.layer_multipliers": "model.layers.1.ple.ple_embedding.layer_multipliers",
            "language_model.layers.1.ple.ngram_embedding.shards.127.scales": "model.layers.1.ple.ple_embedding.ngram_embedding.shard_127.scales",
            "visual.blocks.0.attn.proj.weight": "vision_tower.blocks.0.attn.proj.weight",
            "lm_head.weight": "lm_head.weight",
            "mtp.fc_hidden.weight": "mtp.fc_hidden.weight",
        ]
        for (raw, expected) in mappings { c.equal("path \(raw)", CheckpointNames.canonical(raw, format: .jang6S), expected) }
        c.equal("legacy namespace unchanged", CheckpointNames.canonical("language_model.model.layers.0.x", format: .pipe4), "model.layers.0.x")
        for model in JANGModels.all {
            c.expect("\(model.format): every download pinned", model.files.allSatisfy { $0.sha256?.count == 64 && $0.size > 0 })
            c.equal("\(model.format): alias", JANGModels.named(model.format.modelName)?.revision, model.revision)
            c.equal("\(model.format): path", ModelLocator.resolve(model.format.modelName).lastPathComponent, model.directoryName)
            let weights = WeightStore(modelDirectory: temporary.appendingPathComponent(model.directoryName), jangModel: model)
            c.equal("\(model.format): library prices selected download", weights.status().bytesToFetch, model.totalBytes)
            c.equal("\(model.format): library alias preserves selection", WeightStore.resolving(model.format.modelName).jangModel?.revision, model.revision)
        }
        c.expect("unknown model has no download", JANGModels.named("JANG_2L") == nil)
        return c.report()
    }

    public static func jangNumerics() throws -> CheckReport {
        var c = CheckBuilder("jang-numerics")
        MLX.Memory.cacheLimit = 64 << 20
        let data = try Data(contentsOf: jangFixtures.appendingPathComponent("rows.json"))
        let samples = try JSONSerialization.jsonObject(with: data) as! [[String: Any]]
        func unhex(_ value: String) -> [UInt8] {
            let chars = Array(value.utf8)
            return stride(from: 0, to: chars.count, by: 2).map { i in
                UInt8(String(decoding: chars[i ..< i + 2], as: UTF8.self), radix: 16)!
            }
        }
        for sample in samples {
            let name = sample["tensor"] as! String
            let bits = sample["bits"] as! Int, group = sample["group_size"] as! Int
            let rows = sample["rows"] as! Int, columns = sample["columns"] as! Int
            let w = unhex(sample["weight"] as! String), s = unhex(sample["scales"] as! String), b = unhex(sample["biases"] as! String)
            let wa = MLXArray(Data(w), [rows, columns * bits / 32], dtype: .uint32)
            let sa = MLXArray(Data(s), [rows, columns / group], dtype: .float16)
            let ba = MLXArray(Data(b), [rows, columns / group], dtype: .float16)
            let reference = dequantized(wa, scales: sa, biases: ba, groupSize: group, bits: bits)
            eval(reference)
            let x16 = MLXArray.ones([1, columns], dtype: .bfloat16)
            let projection = QLinear(w: wa, scales: sa, biases: ba, groupSize: group, bits: bits)
            let projected = projection(x16)
            c.equal("\(name): activation precision stays BF16", projected.dtype, .bfloat16)
            c.expect("\(name): projection follows explicit BF16 output contract",
                (projected .== quantizedMM(x16, wa, scales: sa, biases: ba, transpose: true,
                    groupSize: group, bits: bits).asType(.bfloat16)).all().item(Bool.self))
            var decoded: [Float] = []
            for row in 0 ..< rows {
                let wc = columns * bits / 8, sc = columns / group * 2
                let values = try AffineRow.decode(weights: Array(w[row * wc ..< (row + 1) * wc]),
                    scales: Array(s[row * sc ..< (row + 1) * sc]), biases: Array(b[row * sc ..< (row + 1) * sc]),
                    columns: columns, bits: bits, group: group, fp16: true)
                decoded += values
                c.equal("\(name): compact cache preserves FP16", AffineRow.compact(values, fp16: true).map { Float(Float16(bitPattern: $0)) }, values)
            }
            c.equal("\(name): CPU decoding matches native MLX", decoded, reference.asType(.float32).asArray(Float.self))
            if bits == 4 {
                var widened = Data(count: rows * columns * 6 / 8)
                try w.withUnsafeBytes { source in
                    try widened.withUnsafeMutableBytes { target in
                        try AffineCodes.widen(source, to: target, count: rows * columns, from: 4, to: 6)
                    }
                }
                let packed6 = MLXArray(widened, [rows, columns * 6 / 32], dtype: .uint32)
                let actual = dequantized(packed6, scales: sa, biases: ba, groupSize: group, bits: 6)
                eval(actual)
                c.expect("\(name): widening changes no represented weight", (actual .== reference).all().item(Bool.self))
                let x = MLXArray((0 ..< columns).map { Float(($0 % 17) - 8) / 32 }, [1, columns])
                let originalMM = quantizedMM(x, wa, scales: sa, biases: ba, transpose: true, groupSize: group, bits: 4)
                let widenedMM = quantizedMM(x, packed6, scales: sa, biases: ba, transpose: true, groupSize: group, bits: 6)
                eval(originalMM, widenedMM)
                let delta = abs(originalMM - widenedMM).max().item(Float.self)
                let scale = max(1, abs(originalMM).max().item(Float.self))
                c.expect("\(name): native QMM agrees within FP32 reduction tolerance", delta / scale < 0.0001, "relative delta \(delta / scale)")
            }
        }
        do {
            _ = try AffineRow.decode(weights: [], scales: [], biases: [], columns: 160, bits: 3, group: 32, fp16: true)
            c.expect("truncated row refused", false)
        } catch { c.expect("truncated row refused", true) }
        return c.report()
    }
    /// Exercise the production reader, contiguous sweep reads, insertion,
    /// hits and resize using a caller-supplied checkpoint. Only a few experts
    /// are read; the resident trunk and full language model are never loaded.
    public static func jangExpertStreaming(modelDir: URL) throws -> CheckReport {
        var c = CheckBuilder("jang-expert-streaming")
        MLX.Memory.cacheLimit = 64 << 20
        let index = try CheckpointIndex(dir: modelDir)
        guard index.config.format.isJANG else { throw ModelError("expected a JANG checkpoint") }
        let cfg = index.config, store = try ExpertStore(index: index)
        let keys = [ExpertKey(0, 0), ExpertKey(0, 1), ExpertKey(1, 0)]
        let batch = try store.readBatchChecked(keys)
        let runs = try store.readRunsChecked(layer: 0, experts: [0, 1])
        for p in batch.indices {
            c.expect("run/batch piece \(p)", (runs[p] .== batch[p][0..<2]).all().item(Bool.self))
        }
        for (row, key) in keys.enumerated() {
            for (projection, piece) in [("gate_proj", 0), ("up_proj", 3), ("down_proj", 6)] {
                let base = "model.layers.\(key.layer).mlp.switch_mlp.\(projection)"
                let quant = cfg.quantization(for: base)
                var originals: [MLXArray] = []
                for name in ["weight", "scales", "biases"] {
                    let ref = index.ref(base + "." + name)
                    var bytes = Data(count: ref.rowBytes)
                    try bytes.withUnsafeMutableBytes {
                        try index.preadChecked(into: $0.baseAddress!, ref, offset: key.expert * ref.rowBytes, count: ref.rowBytes)
                    }
                    originals.append(MLXArray(bytes, Array(ref.shape.dropFirst()), dtype: name == "weight" ? .uint32 : .float16))
                }
                // Native QMM promotes FP16 metadata with BF16 activations to
                // FP32. Cache those exact promoted values, not new scales.
                let expected = dequantized(originals[0], scales: originals[1].asType(.float32),
                    biases: originals[2].asType(.float32), groupSize: quant.groupSize, bits: quant.bits)
                let actual = dequantized(batch[piece][row], scales: batch[piece+1][row], biases: batch[piece+2][row], groupSize: cfg.qGroup, bits: cfg.expertBits)
                c.expect("native source values \(key.layer)/\(key.expert)/\(projection)", (expected .== actual).all().item(Bool.self))
            }
        }
        let pool = SlotPool(slots: 4, store: store)
        let slots = try pool.ensureChecked(keys)
        let sourceBytes = keys.reduce(0) { total, key in
            total + ["gate_proj", "up_proj", "down_proj"].reduce(0) { value, projection in
                value + ["weight", "scales", "biases"].reduce(0) { n, piece in
                    n + index.ref("model.layers.\(key.layer).mlp.switch_mlp.\(projection).\(piece)").rowBytes
                }
            }
        }
        c.equal("read statistics count source bytes", pool.readBytes, sourceBytes)
        c.expect("cache expansion is not disk traffic", pool.readBytes < keys.count * pool.recordBytes)
        for p in batch.indices {
            c.expect("inserted piece \(p)", (pool.pools[p][MLXArray(slots.map(Int32.init))] .== batch[p]).all().item(Bool.self))
        }
        pool.unpinAll()
        let misses = pool.misses
        _ = try pool.ensureChecked(keys)
        c.equal("cache hits perform no new reads", pool.misses, misses)
        pool.unpinAll()
        pool.resize(to: 8)
        _ = try pool.ensureChecked(keys)
        c.equal("growth preserves cached records", pool.misses, misses)
        pool.unpinAll()
        pool.resize(to: 2)
        let reloaded = try pool.ensureChecked(Array(keys.prefix(2)))
        for p in batch.indices {
            c.expect("shrink/refill piece \(p)", (pool.pools[p][MLXArray(reloaded.map(Int32.init))] .== batch[p][0..<2]).all().item(Bool.self))
        }
        c.equal("pool allocation matches priced record width", pool.poolBytes, 2 * cfg.format.expertRecordBytes)
        return c.report()
    }

}
