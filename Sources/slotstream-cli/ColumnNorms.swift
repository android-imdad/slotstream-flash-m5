import ArgumentParser
import CryptoKit
import Darwin
import Foundation
import MLX
import Slotstream

/// Streams one original expert at a time. This command never creates Engine,
/// the resident trunk or a slot pool, and does not measure inference speed.
struct FlashColumnNorms: ParsableCommand {
    static let configuration = CommandConfiguration(commandName: "flash-column-norms",
        abstract: "Export original JANG_6S down-column norms for the diagnostic neuron oracle")
    @Option var model: String
    @Option var output: String
    @Flag(help: "First source precision layout samples plus final expert; unusable as a full oracle asset")
    var sample = false

    private func digest(_ url: URL) throws -> String {
        let stream = try FileHandle(forReadingFrom: url)
        defer { try? stream.close() }
        var hash = SHA256()
        while let data = try stream.read(upToCount: 1 << 20), !data.isEmpty { hash.update(data: data) }
        return hash.finalize().map { String(format: "%02x", $0) }.joined()
    }

    private func identity(_ fd: Int32, path: String) throws -> String {
        var descriptor = stat(), named = stat()
        guard fstat(fd, &descriptor) == 0, lstat(path, &named) == 0,
              descriptor.st_dev == named.st_dev, descriptor.st_ino == named.st_ino,
              descriptor.st_size == named.st_size,
              descriptor.st_mtimespec.tv_sec == named.st_mtimespec.tv_sec,
              descriptor.st_mtimespec.tv_nsec == named.st_mtimespec.tv_nsec,
              descriptor.st_ctimespec.tv_sec == named.st_ctimespec.tv_sec,
              descriptor.st_ctimespec.tv_nsec == named.st_ctimespec.tv_nsec else {
            throw PlanError("norm source descriptor/path identity changed")
        }
        return "\(descriptor.st_dev):\(descriptor.st_ino):\(descriptor.st_size):"
            + "\(descriptor.st_mtimespec.tv_sec):\(descriptor.st_mtimespec.tv_nsec):"
            + "\(descriptor.st_ctimespec.tv_sec):\(descriptor.st_ctimespec.tv_nsec)"
    }

    func run() throws {
        let fm = FileManager.default
        let root = URL(fileURLWithPath: fm.currentDirectoryPath).resolvingSymlinksInPath()
        let destination = URL(fileURLWithPath: output, relativeTo: root).standardizedFileURL
        let parent = destination.deletingLastPathComponent()
        let flash = root.appendingPathComponent(".build/flash").resolvingSymlinksInPath()
        guard parent.resolvingSymlinksInPath().path.hasPrefix(flash.path + "/"),
              fm.fileExists(atPath: parent.path),
              !fm.fileExists(atPath: destination.path),
              (try? fm.destinationOfSymbolicLink(atPath: destination.path)) == nil else {
            throw ValidationError("norm output needs a fresh directory with an existing parent under .build/flash")
        }
        try fm.createDirectory(at: destination, withIntermediateDirectories: false)
        func writeJSON(_ value: [String: Any], name: String) throws {
            try JSONSerialization.data(withJSONObject: value, options: [.prettyPrinted, .sortedKeys])
                .write(to: destination.appendingPathComponent(name), options: .atomic)
        }
        do {
            try FlashNeuronScoring.acquireProcessGuard()
            MLX.Memory.cacheLimit = 64 << 20
            let modelURL = ModelLocator.resolve(model).resolvingSymlinksInPath()
            let index = try CheckpointIndex(dir: modelURL), cfg = index.config
            guard cfg.format == .jang6S, cfg.numLayers == 48, cfg.numExperts == 512,
                  cfg.hiddenSize == 2560, cfg.moeIntermediate == 640, cfg.qGroup == 64 else {
                throw ValidationError("column norms require the qualified JANG_6S geometry")
            }
            let store = try ExpertStore(index: index)
            var sources: [String: String] = [:]
            var layouts: [[String: Any]] = []
            var selected = Set<String>()
            var keys: [ExpertKey] = []
            for layer in 0 ..< cfg.numLayers {
                let base = "model.layers.\(layer).mlp.switch_mlp.down_proj"
                let quant = cfg.quantization(for: base)
                let scale = index.ref(base + ".scales"), bias = index.ref(base + ".biases")
                let layout = "\(quant.bits):\(scale.dtype):\(bias.dtype)"
                layouts.append(["layer": layer, "bits": quant.bits, "group_size": quant.groupSize,
                    "scale_dtype": scale.dtype, "bias_dtype": bias.dtype,
                    "source_record_bytes": ["gate_proj", "up_proj", "down_proj"].reduce(0) { total, projection in
                        total + ["weight", "scales", "biases"].reduce(0) {
                            $0 + index.ref("model.layers.\(layer).mlp.switch_mlp.\(projection).\($1)").rowBytes
                        }
                    }])
                if !sample { keys += (0 ..< cfg.numExperts).map { ExpertKey(layer, $0) } }
                else if selected.insert(layout).inserted { keys.append(ExpertKey(layer, 0)) }
                for projection in ["gate_proj", "up_proj", "down_proj"] {
                    for member in ["weight", "scales", "biases"] {
                        let ref = index.ref("model.layers.\(layer).mlp.switch_mlp.\(projection).\(member)")
                        sources[ref.file.path] = try identity(index.fd(for: ref.file), path: ref.file.path)
                    }
                }
            }
            if sample { keys.append(ExpertKey(47, 511)) }
            let metadata = try ["config.json", "model.safetensors.index.json"].reduce(into: [String: String]()) {
                $0[$1] = try digest(modelURL.appendingPathComponent($1))
            }
            let writer = FlashBoundedWriter(maxBytes: 64 << 20, liveLimit: 2 << 20)
            var artifacts: [[String: Any]] = []
            for layer in Array(Set(keys.map(\.layer))).sorted() {
                let layerKeys = keys.filter { $0.layer == layer }
                var bytes = Data()
                bytes.reserveCapacity(layerKeys.count * 640 * 4)
                for key in layerKeys {
                    try autoreleasepool {
                        let pieces = try store.readBatchChecked([key], queueDepth: 1)
                        let down = dequantized(pieces[6][0], scales: pieces[7][0], biases: pieces[8][0],
                            groupSize: cfg.qGroup, bits: cfg.expertBits).asType(.float32)
                        eval(down)
                        let norms = try FlashNeuronScoring.columnNorms(down.asArray(Float.self),
                            rows: cfg.hiddenSize, columns: cfg.moeIntermediate)
                        for norm in norms {
                            var bits = norm.bitPattern.littleEndian
                            withUnsafeBytes(of: &bits) { bytes.append(contentsOf: $0) }
                        }
                        for projection in ["gate_proj", "up_proj", "down_proj"] {
                            for member in ["weight", "scales", "biases"] {
                                let ref = index.ref("model.layers.\(layer).mlp.switch_mlp.\(projection).\(member)")
                                guard sources[ref.file.path] == (try identity(index.fd(for: ref.file), path: ref.file.path)) else {
                                    throw PlanError("norm source changed during read")
                                }
                            }
                        }
                    }
                }
                let name = String(format: "layer-%02d.f32", layer)
                try writer.write(bytes, to: destination.appendingPathComponent(name).path)
                artifacts.append(["path": name, "layer": layer, "experts": layerKeys.map(\.expert),
                    "shape": [layerKeys.count, 640], "bytes": bytes.count,
                    "sha256": try digest(destination.appendingPathComponent(name))])
                print("column norms: layer \(layer) complete (\(layerKeys.count) experts)")
                fflush(stdout)
            }
            for (path, before) in sources {
                guard before == (try identity(index.fd(for: URL(fileURLWithPath: path)), path: path)) else {
                    throw PlanError("norm source changed before completion")
                }
            }
            for (name, before) in metadata {
                guard before == (try digest(modelURL.appendingPathComponent(name))) else { throw PlanError("norm metadata changed") }
            }
            let executable = URL(fileURLWithPath: CommandLine.arguments[0]).resolvingSymlinksInPath()
            let bin = executable.deletingLastPathComponent()
            let provenance = try ["slotstream", "mlx.metallib", "build-identity.json", "build-source.tar.gz"].reduce(into: [String: String]()) {
                $0[$1] = try digest(bin.appendingPathComponent($1))
            }
            try writeJSON(["format": "slotstream-column-norms-v1", "schema_version": 1,
                "algorithm": FlashNeuronScoring.algorithm, "scope": sample ? "sample" : "full",
                "dtype": "float32-le", "shape": [48, 512, 640], "expert_count": keys.count,
                "model_path": modelURL.path, "model_metadata": metadata, "source_identities": sources,
                "source_layouts": layouts, "provenance": provenance, "artifacts": artifacts,
                "model_loaded": false, "qualification": false], name: "manifest.json")
            try writeJSON(["format": "slotstream-column-norms-completion-v1",
                "manifest_sha256": try digest(destination.appendingPathComponent("manifest.json"))], name: "completion.json")
        } catch {
            try? writeJSON(["format": "slotstream-column-norms-failure-v1", "error": String(describing: error)], name: "failure.json")
            throw error
        }
    }
}
