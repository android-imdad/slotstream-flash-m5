import CryptoKit
import Foundation
import MLX
import Slotstream

/// Diagnostic artifact writer. A single contiguous CPU copy is bounded before
/// conversion; writes and hashing consume that same copy, not a second array.
private final class PrefillCaptureWriter {
    static let copyLimit = 32 << 20
    static let quota = 1 << 30
    let directory: URL
    var written = 0
    init(_ directory: URL) { self.directory = directory }

    static func digest(_ data: Data) -> String {
        SHA256.hash(data: data).map { String(format: "%02x", $0) }.joined()
    }
    func write(_ data: Data, name: String) throws -> String {
        guard !name.contains("/"), name != ".", name != "..",
              data.count <= Self.quota - written else { throw ModelError("capture artifact quota/path refused") }
        let path = directory.appendingPathComponent(name)
        guard !FileManager.default.fileExists(atPath: path.path) else { throw ModelError("capture refuses artifact overwrite") }
        try data.write(to: path, options: .withoutOverwriting)
        written += data.count
        return Self.digest(data)
    }
    func json(_ object: Any, name: String) throws -> String {
        try write(JSONSerialization.data(withJSONObject: object, options: [.prettyPrinted, .sortedKeys]), name: name)
    }
    static func storage(_ dtype: DType) throws -> (DType, String, Int) {
        switch dtype {
        case .float16, .bfloat16, .float32: return (.float32, "float32", 4)
        case .bool, .int8, .int16, .int32, .int64, .uint8, .uint16, .uint32:
            return (.int64, "int64", 8)
        default: throw ModelError("unsupported capture dtype \(dtype)")
        }
    }
    static func checkedCopyBytes(count: Int, width: Int) throws -> Int {
        guard count >= 0, width > 0, count <= copyLimit / width else {
            throw ModelError("capture tensor exceeds 32 MiB CPU copy cap")
        }
        return count * width
    }
    func tensor(_ value: MLXArray, name: String) throws -> [String: Any] {
        let (dtype, storage, width) = try Self.storage(value.dtype)
        let byteCount = try Self.checkedCopyBytes(count: value.size, width: width)
        let data = value.asType(dtype).asData(access: .copy).data
        guard data.count == byteCount else { throw ModelError("capture byte count mismatch") }
        if dtype == .float32 {
            let finite = data.withUnsafeBytes { $0.bindMemory(to: Float.self).allSatisfy(\.isFinite) }
            guard finite else { throw ModelError("nonfinite capture tensor: \(name)") }
        }
        let digest = try write(data, name: name)
        return ["file": name, "shape": value.shape, "original_dtype": String(describing: value.dtype),
                "storage_dtype": storage, "bytes": data.count, "sha256": digest, "finite": true]
    }
}

extension Diagnostics {
    /// Refuse experimental environment controls before any model allocation.
    public static func validatePrefillCaptureEnvironment(chunk: Int, environment: [String: String] = ProcessInfo.processInfo.environment) throws {
        guard [256, 384, 512, 1024].contains(chunk) else { throw ModelError("capture chunk must be 256, 384, 512 or 1024") }
        for (key, value) in environment where key.hasPrefix("SLOTSTREAM_") {
            switch key {
            case "SLOTSTREAM_PREFIX_CACHE" where value == "0": break
            case "SLOTSTREAM_PREFILL_CHUNK" where value == String(chunk): break
            default: throw ModelError("unsupported capture environment control: \(key)")
            }
        }
        let unknownMLX = environment.keys.filter { $0.hasPrefix("MLX_") && $0 != "MLX_ENABLE_TF32" }
        guard unknownMLX.isEmpty else { throw ModelError("unsupported capture MLX controls: \(unknownMLX.sorted())") }
        if let value = environment["MLX_ENABLE_TF32"], value != "0" && value != "1" {
            throw ModelError("MLX_ENABLE_TF32 must be 0 or 1")
        }
        guard try InferenceOptimizations.environment(environment) == InferenceOptimizations.environment([:]) else {
            throw ModelError("capture requires deployed optimization controls")
        }
    }

    public static func prefillCapture(modelDir: URL, prompt: String, raw: Bool, chunk: Int,
                                      continuation: [Int]?, destination: URL,
                                      identity: [String: Any]) async throws {
        try validatePrefillCaptureEnvironment(chunk: chunk)
        guard continuation == nil || continuation?.count == 8 else { throw ModelError("continuation must contain exactly eight IDs") }
        let writer = PrefillCaptureWriter(destination)
        var report: [String: Any] = ["format": "slotstream-prefill-capture-v1", "schema_version": 1,
            "identity": identity, "actual_engine": true, "capture_path": "actualEngine", "raw": raw, "process_id": getpid(),
            "byte_order": "little", "copy_limit_bytes": PrefillCaptureWriter.copyLimit, "artifact_quota_bytes": PrefillCaptureWriter.quota]
        do {
            let index = try CheckpointIndex(dir: modelDir)
            guard index.config.format == .jang6S else { throw ModelError("capture requires JANG_6S") }
            let policy = try RuntimeAllocationPolicy(prefillChunkOverride: chunk, prefixCacheEnabled: false)
            let plan = try CheckpointMemory(index: index).plan(memoryGB: 14, maxContext: 4096, policy: policy)
            let engine = try await Engine(modelDir: modelDir, plan: plan, expertWidening: .packed4To6)
            defer { engine.generator.completePrefillObserver = nil; engine.model.routerObserver = nil }
            guard engine.model.optimizations == (try InferenceOptimizations.environment([:])),
                  engine.model.optimizations.skipUnusedFinalForward,
                  !engine.model.optimizations.terminalPrefillPruning,
                  !engine.model.optimizations.demandedPrefillOutput,
                  engine.model.mtpHead == nil, !engine.visionAllowed, !engine.prefixCache.enabled,
                  engine.generator.prefillChunk == chunk,
                  engine.effectiveWideningPolicy == .packed4To6 else { throw ModelError("capture effective configuration mismatch") }
            let control = try engine.beginRequest()
            try control.checkInputBytes(prompt.utf8.count)
            let ids: [Int]
            if raw { ids = engine.tokenizer.encode(text: prompt) }
            else { ids = try engine.encodeChat([ChatMessage(role: "user", content: prompt)], thinking: false) }
            guard !ids.isEmpty, ids.count + 8 <= 4096 else { throw ModelError("capture prompt must leave eight context tokens") }
            if let continuation, !continuation.allSatisfy({ (0 ..< engine.model.cfg.vocabSize).contains($0) }) {
                throw ModelError("continuation contains an out-of-vocabulary ID")
            }
            let options = try JSONSerialization.jsonObject(with: JSONEncoder().encode(engine.model.optimizations))
            let environment = ProcessInfo.processInfo.environment.filter { $0.key.hasPrefix("SLOTSTREAM_") || $0.key.hasPrefix("MLX_") }
            let layerTypes = engine.model.cfg.layerTypes
            var expected = ["logits", "tokens", "ngram"]
            for layer in 0 ..< engine.model.runLayers {
                expected += (layerTypes[layer] == "linear_attention" ? ["conv", "ssm"] : ["key", "value", "index"]).map { "\($0).\(layer)" }
            }
            expected += engine.model.cfg.pleLayerIndices.map { "ple.\($0)" }
            report.merge(["plan": plan.json(), "prompt_ids": ids, "top_k": engine.model.cfg.topK,
                "num_layers": engine.model.runLayers, "vocab_size": engine.model.cfg.vocabSize,
                "layer_types": layerTypes, "expected_tensor_keys": expected.sorted(),
                "optimizations": options, "runtime_environment": environment,
                "numerical_environment": ["effective_tf32": environment["MLX_ENABLE_TF32"] != "0"],
                "effective_prefill_chunk": engine.generator.prefillChunk, "effective_pool_slots": engine.model.pool.slots,
                "effective_expert_widening": engine.effectiveWideningPolicy.rawValue,
                "effective_mtp": false, "effective_vision": false, "effective_prefix_cache": false,
                "sampling": ["greedy": true, "seed": 42, "max_tokens": 1]], uniquingKeysWith: { _, new in new })
            _ = try writer.json(report, name: "started.json")
            var capturedState: Qwen4ExpModel.State?
            var capturedLogits: MLXArray?
            var observerCalls = 0
            engine.generator.completePrefillObserver = { logits, state in
                observerCalls += 1
                guard state.tokenCount == ids.count else { throw ModelError("incomplete prefill observer boundary") }
                capturedState = state; capturedLogits = logits
            }
            var phase = "prefill"
            var routes: [String: [Int: [Int32]]] = [:]
            var routeEvents: [[String: Any]] = []
            engine.model.routerObserver = { layer, values in
                routes[phase, default: [:]][layer, default: []].append(contentsOf: values)
                routeEvents.append(["phase": phase, "layer": layer, "rows": values.count / engine.model.cfg.topK])
            }
            var params = SampleParams.greedy; params.maxTokens = 1; params.seed = 42
            let result = engine.generate(promptIds: ids, params: params, request: control)
            engine.generator.completePrefillObserver = nil
            guard result.stats.runtimeError == nil, result.stats.requestFailure == nil,
                  observerCalls == 1, let state = capturedState, var logits = capturedLogits,
                  state.tokenCount == ids.count, result.stats.decodeForwardPasses == 0,
                  result.stats.prefillTokens == ids.count else {
                throw ModelError("normal one-token generation failed or consumed an unexpected token: \(result.stats.runtimeError ?? result.stats.finishReason)")
            }
            guard result.ids.count == 1, result.ids[0] == argMax(logits.reshaped([-1])).item(Int.self) else {
                throw ModelError("normal greedy output does not match captured prefill argmax")
            }
            capturedLogits = nil; capturedState = nil
            var consumed = ids
            var snapshots: [String: Any] = [:]
            func snapshot(_ name: String) throws {
                try control.check(phase: "diagnostic snapshot")
                guard ProcessMemory.residentBytes() <= 14_000_000_000 else { throw ModelError("capture footprint exceeds 14 GB") }
                var fields = state.diagnosticTensors(); fields["logits"] = logits
                guard fields.keys.sorted() == expected.sorted(), logits.size == engine.model.cfg.vocabSize else {
                    throw ModelError("capture state catalogue or full vocabulary mismatch")
                }
                var tensors: [String: Any] = [:]
                for key in fields.keys.sorted() { tensors[key] = try writer.tensor(fields[key]!, name: "\(name)-\(key).bin") }
                let value: [String: Any] = ["consumed_ids": consumed,
                    "greedy_token": argMax(logits.reshaped([-1])).item(Int.self), "tensors": tensors]
                snapshots[name] = value
                _ = try writer.json(value, name: "\(name).json")
            }
            try snapshot("prefill")
            phase = "continuation"
            var chosen: [Int] = []
            for step in 1 ... 8 {
                let token = continuation?[step - 1] ?? argMax(logits.reshaped([-1])).item(Int.self)
                chosen.append(token)
                guard state.tokenCount + 1 <= engine.maxContextTokens else { throw ModelError("continuation context exceeded") }
                let allocation = engine.model.sequenceAllocationBytes(tokens: state.tokenCount + 1, draftTokens: nil, state: state)
                try control.check(nextAllocationBytes: allocation + 1_300_000, phase: "diagnostic continuation")
                logits = try engine.model.lastLogitsChecked([token], state: state)
                eval(logits); consumed.append(token)
                guard ProcessMemory.residentBytes() <= 14_000_000_000 else { throw ModelError("continuation footprint exceeds 14 GB") }
                guard state.tokenCount == consumed.count else { throw ModelError("continuation did not commit exactly one token") }
                if step == 1 || step == 8 { try snapshot("step\(step)") }
            }
            engine.model.routerObserver = nil
            var routeArtifacts: [String: Any] = [:]
            for routePhase in ["prefill", "continuation"] {
                var byLayer: [String: Any] = [:]
                let rows = routePhase == "prefill" ? ids.count : 8
                for layer in 0 ..< engine.model.runLayers {
                    guard let values = routes[routePhase]?[layer], values.count == rows * engine.model.cfg.topK else {
                        throw ModelError("capture routing rows do not cover logical positions")
                    }
                    let data = values.withUnsafeBytes { Data($0) }
                    let name = "routes-\(routePhase)-\(layer).bin"
                    let hash = try writer.write(data, name: name)
                    byLayer[String(layer)] = ["file": name, "shape": [rows, engine.model.cfg.topK],
                        "original_dtype": "int32", "storage_dtype": "int32", "bytes": data.count,
                        "sha256": hash, "finite": true, "logical_positions": Array(0 ..< rows)]
                }
                routeArtifacts[routePhase] = byLayer
            }
            report["snapshots"] = snapshots; report["routes"] = routeArtifacts; report["route_events"] = routeEvents
            report["continuation_ids"] = chosen; report["continuation_mode"] = continuation == nil ? "baseline-raw-argmax" : "imported-baseline-ids"
            report["normal_output_ids"] = result.ids
            report["stats"] = try JSONSerialization.jsonObject(with: JSONEncoder().encode(result.stats))
            report["terminal_physical_footprint_bytes"] = ProcessMemory.residentBytes()
            report["lifetime_peak_physical_footprint_bytes"] = ProcessMemory.peakResidentBytes()
            _ = try writer.json(chosen, name: "continuation.json")
            let hash = try writer.json(report, name: "report.json")
            _ = try writer.json(["format": "slotstream-prefill-capture-completion-v1", "schema_version": 1,
                "report_sha256": hash, "passed": true], name: "completion.json")
        } catch {
            report["error"] = String(describing: error)
            _ = try? writer.json(report, name: "failure.json")
            throw error
        }
    }

    public static func prefillCaptureSelfCheck() throws -> CheckReport {
        var c = CheckBuilder("prefill-capture-serializer")
        let directory = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: false)
        defer { try? FileManager.default.removeItem(at: directory) }
        let writer = PrefillCaptureWriter(directory)
        let integers = MLXArray([Int64(1), 16_777_217, -3])
        let entry = try writer.tensor(integers, name: "integers.bin")
        let data = try Data(contentsOf: directory.appendingPathComponent("integers.bin"))
        c.equal("integer storage preserves values beyond float32 precision", data.withUnsafeBytes { Array($0.bindMemory(to: Int64.self)) }, [1, 16_777_217, -3])
        c.equal("hash matches exact artifact", entry["sha256"] as? String, PrefillCaptureWriter.digest(data))
        c.equal("original shape retained", entry["shape"] as? [Int], [3])
        let floating = try writer.tensor(MLXArray([Float(1.5), -2]), name: "floats.bin")
        c.equal("floating storage explicit", floating["storage_dtype"] as? String, "float32")
        func rejected(_ body: () throws -> Void) -> Bool { do { try body(); return false } catch { return true } }
        c.expect("copy cap checked without allocation", rejected { _ = try PrefillCaptureWriter.checkedCopyBytes(count: (32 << 20) / 4 + 1, width: 4) })
        c.expect("nonfinite refused", rejected { _ = try writer.tensor(MLXArray([Float.nan]), name: "nan.bin") })
        c.expect("overwrite refused", rejected { _ = try writer.tensor(integers, name: "integers.bin") })
        c.expect("unsafe path refused", rejected { _ = try writer.write(Data(), name: "../escape") })
        c.expect("quota checked before write", rejected { writer.written = PrefillCaptureWriter.quota; _ = try writer.write(Data([1]), name: "over.bin") })
        c.expect("experimental controls refused", rejected { try validatePrefillCaptureEnvironment(chunk: 256, environment: ["SLOTSTREAM_OPT_FINAL_FORWARD": "0"]) })
        c.expect("unknown MLX controls refused", rejected { try validatePrefillCaptureEnvironment(chunk: 256, environment: ["MLX_UNKNOWN": "1"]) })
        c.expect("mismatched chunk refused", rejected { try validatePrefillCaptureEnvironment(chunk: 512, environment: ["SLOTSTREAM_PREFILL_CHUNK": "256"]) })
        c.expect("384 control mismatched override refused", rejected { try validatePrefillCaptureEnvironment(chunk: 384, environment: ["SLOTSTREAM_PREFILL_CHUNK": "512"]) })
        try validatePrefillCaptureEnvironment(chunk: 384, environment: ["SLOTSTREAM_PREFIX_CACHE": "0", "SLOTSTREAM_PREFILL_CHUNK": "384"])
        c.expect("384 empirical control accepted", true)
        try validatePrefillCaptureEnvironment(chunk: 1024, environment: ["SLOTSTREAM_PREFIX_CACHE": "0", "SLOTSTREAM_PREFILL_CHUNK": "1024"])
        c.expect("supported controls accepted", true)
        return c.report()
    }
}
