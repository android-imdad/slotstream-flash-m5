import CryptoKit
import Darwin
import Foundation
import MLX
import Slotstream

enum OracleFiles {
    static func digest(_ url: URL) throws -> String {
        let file = try FileHandle(forReadingFrom: url)
        defer { try? file.close() }
        var hash = SHA256()
        while let part = try file.read(upToCount: 1 << 20), !part.isEmpty { hash.update(data: part) }
        return hash.finalize().map { String(format: "%02x", $0) }.joined()
    }
    static func json(_ url: URL, limit: Int) throws -> Data {
        var before = stat()
        guard lstat(url.path, &before) == 0, before.st_mode & S_IFMT == S_IFREG,
              before.st_size > 0, before.st_size <= limit else { throw PlanError("oracle JSON is missing, unsafe or too large") }
        let file = try FileHandle(forReadingFrom: url)
        defer { try? file.close() }
        let data = try file.read(upToCount: limit + 1) ?? Data()
        var after = stat()
        guard fstat(file.fileDescriptor, &after) == 0, before.st_dev == after.st_dev,
              before.st_ino == after.st_ino, before.st_size == after.st_size,
              before.st_mtimespec.tv_sec == after.st_mtimespec.tv_sec,
              before.st_mtimespec.tv_nsec == after.st_mtimespec.tv_nsec,
              data.count == before.st_size else { throw PlanError("oracle JSON changed during read") }
        return data
    }
}

struct OracleConfiguration: Decodable {
    static let extraReservation = 80 << 20
    static let totalReservation = 208 << 20
    let format: String
    let schema_version: Int
    let algorithm: String
    let retained_blocks: Int
    let split: String
    let norm_manifest: String
    let norm_manifest_sha256: String
    let study: String
    let study_sha256: String
    let suite: String
    let suite_sha256: String
    let capture_manifest_sha256: String
    let diagnostic_reserved_bytes: Int

    static func load(_ path: String, manifestHash: String) throws -> OracleConfiguration {
        let data = try OracleFiles.json(URL(fileURLWithPath: path), limit: 8192)
        let expected: Set<String> = ["format", "schema_version", "algorithm", "retained_blocks", "split",
            "norm_manifest", "norm_manifest_sha256", "study", "study_sha256", "suite", "suite_sha256",
            "capture_manifest_sha256", "diagnostic_reserved_bytes"]
        guard let object = try JSONSerialization.jsonObject(with: data) as? [String: Any], Set(object.keys) == expected else {
            throw PlanError("unknown oracle configuration fields")
        }
        let value = try JSONDecoder().decode(Self.self, from: data)
        guard value.format == "slotstream-oracle-config-v1", value.schema_version == 1,
              value.algorithm == FlashNeuronScoring.algorithm, [2, 4, 6, 8, 10].contains(value.retained_blocks),
              value.split == "development", value.capture_manifest_sha256 == manifestHash,
              value.diagnostic_reserved_bytes == totalReservation else { throw PlanError("unsupported oracle configuration") }
        for (name, hash) in [(value.norm_manifest, value.norm_manifest_sha256), (value.study, value.study_sha256), (value.suite, value.suite_sha256)] {
            guard name.hasPrefix("/"), hash.count == 64,
                  try OracleFiles.digest(URL(fileURLWithPath: name)) == hash else { throw PlanError("oracle bound asset changed") }
        }
        let studyData = try OracleFiles.json(URL(fileURLWithPath: value.study), limit: 1 << 20)
        guard let study = try JSONSerialization.jsonObject(with: studyData) as? [String: Any],
              study["format"] as? String == "slotstream-neuron-study-v1",
              study["algorithm"] as? String == value.algorithm,
              study["split"] as? String == "development",
              study["retained_order"] as? [Int] == [10, 8, 6, 4, 2],
              study["norm_manifest_sha256"] as? String == value.norm_manifest_sha256,
              study["suite_sha256"] as? String == value.suite_sha256 else { throw PlanError("oracle study bindings differ") }
        return value
    }
}

private struct NormArtifact: Decodable {
    let path: String
    let layer: Int
    let experts: [Int]
    let shape: [Int]
    let bytes: Int
    let sha256: String
}
private struct NormManifest: Decodable {
    let format: String
    let schema_version: Int
    let algorithm: String
    let scope: String
    let dtype: String
    let shape: [Int]
    let expert_count: Int
    let model_metadata: [String: String]
    let artifacts: [NormArtifact]
}

/// One CPU norm table (60 MiB), at most 1 MiB hash scratch, one position's
/// mask rows, and bounded metadata. The capture planner reserves 80 MiB extra.
final class NativeNeuronOracle: FlashHiddenTransform {
    let configuration: OracleConfiguration
    let configurationHash: String
    private let destination: URL
    private var norms: [Float]
    private let writer = FlashBoundedWriter(maxBytes: 8 << 20, liveLimit: 128 << 10)
    private var document = ""
    private var position = -1
    private var rows: [[String: Any]] = []
    private(set) var files: [[String: Any]] = []
    private lazy var mask = try! FlashNeuronMaskTransform(retaining: configuration.retained_blocks,
        norms: { [unowned self] layer, expert in
            let start = (layer * 512 + expert) * 640
            return Array(self.norms[start ..< start + 640])
        }, record: { [unowned self] layer, ranks, experts, blocks in
            try self.record(layer: layer, ranks: ranks, experts: experts, blocks: blocks)
        })

    init(configuration: OracleConfiguration, configurationPath: String, model: URL, destination: URL) throws {
        self.configuration = configuration
        configurationHash = try OracleFiles.digest(URL(fileURLWithPath: configurationPath))
        self.destination = destination
        let manifestURL = URL(fileURLWithPath: configuration.norm_manifest)
        let data = try OracleFiles.json(manifestURL, limit: 1 << 20)
        guard SHA256.hash(data: data).map({ String(format: "%02x", $0) }).joined() == configuration.norm_manifest_sha256 else {
            throw PlanError("norm manifest changed before allocation")
        }
        let manifest = try JSONDecoder().decode(NormManifest.self, from: data)
        let completionData = try OracleFiles.json(manifestURL.deletingLastPathComponent().appendingPathComponent("completion.json"), limit: 4096)
        guard let completion = try JSONSerialization.jsonObject(with: completionData) as? [String: String],
              completion == ["format": "slotstream-column-norms-completion-v1", "manifest_sha256": configuration.norm_manifest_sha256] else {
            throw PlanError("oracle norm export is incomplete")
        }
        guard manifest.format == "slotstream-column-norms-v1", manifest.schema_version == 1,
              manifest.algorithm == configuration.algorithm, manifest.scope == "full", manifest.dtype == "float32-le",
              manifest.shape == [48, 512, 640], manifest.expert_count == 24576,
              manifest.artifacts.map(\.layer) == Array(0 ..< 48), manifest.model_metadata.count == 2 else {
            throw PlanError("oracle requires a complete original norm asset")
        }
        for name in ["config.json", "model.safetensors.index.json"] {
            guard manifest.model_metadata[name] == (try OracleFiles.digest(model.appendingPathComponent(name))) else {
                throw PlanError("norms bind another checkpoint")
            }
        }
        norms = [Float](repeating: 0, count: 48 * 512 * 640)
        try norms.withUnsafeMutableBufferPointer { values in
            let target = UnsafeMutableRawBufferPointer(values)
            for artifact in manifest.artifacts {
                guard artifact.path == String(format: "layer-%02d.f32", artifact.layer),
                      artifact.experts == Array(0 ..< 512), artifact.shape == [512, 640],
                      artifact.bytes == 512 * 640 * 4 else { throw PlanError("invalid norm layer coverage") }
                let url = manifestURL.deletingLastPathComponent().appendingPathComponent(artifact.path)
                let fd = open(url.path, O_RDONLY | O_NOFOLLOW)
                guard fd >= 0 else { throw PlanError("missing or symlinked norm layer") }
                defer { close(fd) }
                var before = stat(), after = stat()
                guard fstat(fd, &before) == 0, before.st_mode & S_IFMT == S_IFREG,
                      before.st_size == artifact.bytes else { throw PlanError("norm layer has wrong size") }
                var offset = 0, hash = SHA256()
                while offset < artifact.bytes {
                    let count = min(1 << 20, artifact.bytes - offset)
                    let address = target.baseAddress! + artifact.layer * artifact.bytes + offset
                    let got = Darwin.read(fd, address, count)
                    if got < 0 && errno == EINTR { continue }
                    guard got > 0 else { throw PlanError("short norm layer read") }
                    hash.update(data: Data(bytes: address, count: got))
                    offset += got
                }
                guard fstat(fd, &after) == 0, before.st_dev == after.st_dev, before.st_ino == after.st_ino,
                      before.st_size == after.st_size,
                      before.st_mtimespec.tv_sec == after.st_mtimespec.tv_sec,
                      before.st_mtimespec.tv_nsec == after.st_mtimespec.tv_nsec,
                      before.st_ctimespec.tv_sec == after.st_ctimespec.tv_sec,
                      before.st_ctimespec.tv_nsec == after.st_ctimespec.tv_nsec,
                      hash.finalize().map({ String(format: "%02x", $0) }).joined() == artifact.sha256 else {
                    throw PlanError("norm layer changed during read")
                }
            }
            guard values.allSatisfy({ $0.isFinite && $0 >= 0 }) else { throw PlanError("nonfinite or negative norm") }
        }
    }

    func begin(document: String, position: Int) throws {
        guard rows.isEmpty, files.count < 64 else { throw PlanError("unfinished or excessive oracle position") }
        self.document = document
        self.position = position
    }
    func finish() throws {
        guard rows.isEmpty, let last = files.last,
              last["document_id"] as? String == document, last["token_position"] as? Int == position else {
            throw PlanError("incomplete oracle mask position")
        }
        document = ""; position = -1
    }
    func cancel() { writer.cancel() }
    func validateForward(tokens: Int) throws {
        guard !document.isEmpty, position >= 0 else { throw PlanError("oracle position is not initialized") }
        try mask.validateForward(tokens: tokens)
    }
    func transformHidden(layer: Int, routerRanks: [Int], expertIDs: [Int32], value: MLXArray) throws -> MLXArray {
        try mask.transformHidden(layer: layer, routerRanks: routerRanks, expertIDs: expertIDs, value: value)
    }
    private func record(layer: Int, ranks: [Int], experts: [Int32], blocks: [[Bool]]) throws {
        guard layer == rows.count, !document.isEmpty else { throw PlanError("oracle mask layer order differs") }
        rows.append(["layer": layer, "router_ranks": ranks, "expert_ids": experts, "blocks": blocks])
        if layer == 47 {
            let data = try JSONSerialization.data(withJSONObject: rows, options: [.sortedKeys])
            let name = String(format: "oracle-mask-%03d.json", files.count)
            // Write while still inside the checked forward, before state commit.
            try writer.write(data, to: destination.appendingPathComponent(name).path)
            files.append(["name": name, "document_id": document, "token_position": position,
                "bytes": data.count, "sha256": SHA256.hash(data: data).map { String(format: "%02x", $0) }.joined()])
            rows.removeAll(keepingCapacity: true)
        }
    }
    func evidence(maskFailureRefused: Bool) -> [String: Any] {
        ["format": "slotstream-oracle-evidence-v1", "config_sha256": configurationHash,
         "study_sha256": configuration.study_sha256, "suite_sha256": configuration.suite_sha256,
         "norm_manifest_sha256": configuration.norm_manifest_sha256, "algorithm": configuration.algorithm,
         "retained_blocks": configuration.retained_blocks, "diagnostic_reserved_bytes": OracleConfiguration.totalReservation,
         "norm_payload_bytes": norms.count * 4, "warmup_policy": "dense-whole-document",
         "mask_failure_state_reuse_refused": maskFailureRefused, "mask_files": files]
    }
}

final class RejectingOracleTransform: FlashHiddenTransform {
    func validateForward(tokens: Int) throws {}
    func transformHidden(layer: Int, routerRanks: [Int], expertIDs: [Int32], value: MLXArray) throws -> MLXArray {
        throw PlanError("injected oracle mask failure")
    }
}
