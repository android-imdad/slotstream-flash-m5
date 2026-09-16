import ArgumentParser
import CryptoKit
import Foundation
import Slotstream
import SlotstreamDiagnostics

/// Deliberately narrow qualification instrument; no optimization defaults change.
struct PrefillCapture: ParsableCommand {
    static let configuration = CommandConfiguration(commandName: "prefill-capture",
        abstract: "Capture matched deployed JANG prefill logits and continuation state")
    @Option var model: String?
    @Option var promptFile: String?
    @Option var chunk = 256
    @Option var output: String?
    @Option var continuationFile: String?
    @Flag var raw = false
    @Flag var selfCheck = false

    private func digest(_ url: URL) throws -> String {
        let handle = try FileHandle(forReadingFrom: url)
        defer { try? handle.close() }
        var hash = SHA256()
        while let data = try handle.read(upToCount: 1 << 20), !data.isEmpty { hash.update(data: data) }
        return hash.finalize().map { String(format: "%02x", $0) }.joined()
    }

    private func boundedRead(_ url: URL, limit: Int) throws -> Data {
        let handle = try FileHandle(forReadingFrom: url)
        defer { try? handle.close() }
        let data = try handle.read(upToCount: limit + 1) ?? Data()
        guard data.count <= limit else { throw ValidationError("capture input exceeds byte limit") }
        return data
    }

    func run() throws {
        if selfCheck {
            guard model == nil, promptFile == nil, output == nil, continuationFile == nil, !raw else {
                throw ValidationError("--self-check cannot be combined with capture options")
            }
            let report = try Diagnostics.prefillCaptureSelfCheck()
            print(String(decoding: try JSONEncoder().encode(report), as: UTF8.self))
            guard report.passed else { throw ValidationError("capture self-check failed") }
            return
        }
        guard let model, let promptFile, let output else {
            throw ValidationError("--model, --prompt-file and --output are required")
        }
        try Diagnostics.validatePrefillCaptureEnvironment(chunk: chunk)
        let fm = FileManager.default
        let destination = URL(fileURLWithPath: output).standardizedFileURL
        let parent = destination.deletingLastPathComponent()
        var isDirectory: ObjCBool = false
        guard fm.fileExists(atPath: parent.path, isDirectory: &isDirectory), isDirectory.boolValue,
              parent.resolvingSymlinksInPath().path == parent.path,
              !fm.fileExists(atPath: destination.path), destination.lastPathComponent != "." else {
            throw ValidationError("capture output requires a fresh path with an existing nonsymlink parent")
        }
        let promptURL = URL(fileURLWithPath: promptFile)
        let promptBytes = try boundedRead(promptURL, limit: 1 << 20)
        guard promptBytes.count <= 1 << 20, let prompt = String(data: promptBytes, encoding: .utf8) else {
            throw ValidationError("prompt must be UTF-8 and at most 1 MiB")
        }
        let continuation = try continuationFile.map { path -> [Int] in
            let data = try boundedRead(URL(fileURLWithPath: path), limit: 4096)
            guard data.count <= 4096 else { throw ValidationError("continuation file too large") }
            let ids = try JSONDecoder().decode([Int].self, from: data)
            guard ids.count == 8 else { throw ValidationError("continuation requires eight IDs") }
            return ids
        }
        let modelDir = ModelLocator.resolve(model).resolvingSymlinksInPath()
        let index = try CheckpointIndex(dir: modelDir)
        guard index.config.format == .jang6S else { throw ValidationError("capture requires JANG_6S") }
        for file in JANGModels.jang6S.files {
            let path = modelDir.appendingPathComponent(file.path)
            let size = (try? fm.attributesOfItem(atPath: path.path))?[.size] as? Int64
            guard size == file.size else { throw ValidationError("pinned checkpoint file missing or wrong size: \(file.path)") }
            if !file.path.hasSuffix(".safetensors") {
                guard try digest(path) == file.sha256 else { throw ValidationError("checkpoint metadata hash mismatch: \(file.path)") }
            }
        }
        let executable = URL(fileURLWithPath: CommandLine.arguments[0]).standardizedFileURL.resolvingSymlinksInPath()
        let build = executable.deletingLastPathComponent()
        var identity: [String: Any] = ["model_path": modelDir.path,
            "checkpoint_identity": "\(JANGModels.jang6S.repository)@\(JANGModels.jang6S.revision)",
            "checkpoint_identity_scope": "pinned manifest identity; config and tokenizer hashed, weight payloads not rehashed by this command",
            "binary_sha256": try digest(executable),
            "prompt_sha256": SHA256.hash(data: promptBytes).map { String(format: "%02x", $0) }.joined(),
            "model_config_sha256": try digest(modelDir.appendingPathComponent("config.json")),
            "tokenizer_sha256": try digest(modelDir.appendingPathComponent("tokenizer.json")),
            "tokenizer_config_sha256": try digest(modelDir.appendingPathComponent("tokenizer_config.json"))]
        for (field, filename) in [("metallib_sha256", "mlx.metallib"), ("build_identity_sha256", "build-identity.json"),
                                  ("source_archive_sha256", "build-source.tar.gz")] {
            identity[field] = try digest(build.appendingPathComponent(filename))
        }
        let modelIndex = modelDir.appendingPathComponent("model.safetensors.index.json")
        if fm.fileExists(atPath: modelIndex.path) { identity["model_index_sha256"] = try digest(modelIndex) }
        try fm.createDirectory(at: destination, withIntermediateDirectories: false)
        let semaphore = DispatchSemaphore(value: 0)
        var result: Result<Void, Error> = .success(())
        Task {
            do {
                try await Diagnostics.prefillCapture(modelDir: modelDir, prompt: prompt, raw: raw,
                    chunk: chunk, continuation: continuation, destination: destination, identity: identity)
            } catch { result = .failure(error) }
            semaphore.signal()
        }
        semaphore.wait()
        try result.get()
        print(destination.appendingPathComponent("report.json").path)
    }
}
