import ArgumentParser
import CryptoKit
import CoreFoundation
import Darwin
import Foundation
import Slotstream

private struct AuthoredCorpus: Decodable {
    var format: String
    var schemaVersion: Int
    var manifestSHA256: String
    var documents: [AuthoredDocument]
}

private struct AuthoredDocument: Decodable {
    var id: String
    var kind: String
    var category: String
    var split: String
    var sourceID: String
    var license: String
    var partitionKey: String
    var text: String?
    var textSHA256: String?
    var warmupTokens: Int?
    var system: String?
    var user: String?
    var reference: String?
    var contentSHA256: String?
    var scoreTokens: Int
}

enum CorpusSupport {
    static let transformerRevision = "2fa33e1f5e7131a7fc64c28e6d161dcec0d24820"
    static let validSplits = Set(["training", "development", "qualification"])

    static func digest(_ data: Data) -> String {
        SHA256.hash(data: data).map { String(format: "%02x", $0) }.joined()
    }

    static func digest(_ url: URL) throws -> String {
        digest(try Data(contentsOf: url, options: .mappedIfSafe))
    }

    static func boundedFile(_ url: URL, limit: Int) throws -> Data {
        var before = stat()
        guard lstat(url.path, &before) == 0,
              before.st_mode & S_IFMT == S_IFREG,
              before.st_size >= 0, before.st_size <= limit else {
            throw ValidationError("corpus source must be a regular bounded file")
        }
        let handle = try FileHandle(forReadingFrom: url)
        let data: Data
        do {
            data = try handle.read(upToCount: limit + 1) ?? Data()
            try handle.close()
        } catch {
            try? handle.close()
            throw error
        }
        var after = stat()
        guard lstat(url.path, &after) == 0,
              data.count <= limit, data.count == Int(before.st_size),
              before.st_dev == after.st_dev, before.st_ino == after.st_ino,
              before.st_size == after.st_size,
              before.st_mtimespec.tv_sec == after.st_mtimespec.tv_sec,
              before.st_mtimespec.tv_nsec == after.st_mtimespec.tv_nsec else {
            throw ValidationError("corpus source changed while being read")
        }
        return data
    }

    static func safeID(_ value: String) -> Bool {
        !value.isEmpty && value.utf8.count <= 64 && value.utf8.allSatisfy {
            (48 ... 57).contains($0) || (65 ... 90).contains($0) ||
                (97 ... 122).contains($0) || $0 == 45 || $0 == 95
        }
    }

    static func canonicalHash(_ raw: [String: Any], omitting key: String) throws -> String {
        var copy = raw
        copy.removeValue(forKey: key)
        return digest(Data(try canonicalJSON(copy).utf8))
    }

    static func canonicalJSON(_ value: Any) throws -> String {
        if value is NSNull { return "null" }
        if let string = value as? String {
            let encoded = try JSONSerialization.data(withJSONObject: [string],
                options: [.withoutEscapingSlashes])
            return String(decoding: encoded.dropFirst().dropLast(), as: UTF8.self)
        }
        if let number = value as? NSNumber {
            if CFGetTypeID(number) == CFBooleanGetTypeID() {
                return number.boolValue ? "true" : "false"
            }
            return String(decoding: try JSONSerialization.data(withJSONObject: [number])
                .dropFirst().dropLast(), as: UTF8.self)
        }
        if let array = value as? [Any] {
            return "[" + (try array.map(canonicalJSON)).joined(separator: ",") + "]"
        }
        if let dictionary = value as? [String: Any] {
            let fields = try dictionary.keys.sorted().map { key -> String in
                try canonicalJSON(key) + ":" + canonicalJSON(dictionary[key]!)
            }
            return "{" + fields.joined(separator: ",") + "}"
        }
        throw ValidationError("canonical JSON contains an unsupported value")
    }

    static func jsonData(_ value: Any) throws -> Data {
        try JSONSerialization.data(withJSONObject: value, options: [.prettyPrinted, .sortedKeys])
    }

    static func writeJSON(_ value: Any, to url: URL) throws {
        try jsonData(value).write(to: url, options: .atomic)
    }

    static func outputDirectory(_ value: String) throws -> URL {
        let fm = FileManager.default
        let root = URL(fileURLWithPath: fm.currentDirectoryPath).standardizedFileURL
        let flash = root.appendingPathComponent(".build/flash").standardizedFileURL
        let output = URL(fileURLWithPath: value, relativeTo: root).standardizedFileURL
        let parent = output.deletingLastPathComponent()
        guard output.path.hasPrefix(flash.path + "/"),
              !fm.fileExists(atPath: output.path), fm.fileExists(atPath: parent.path),
              parent.resolvingSymlinksInPath().path
              .hasPrefix(flash.resolvingSymlinksInPath().path + "/") else {
            throw ValidationError("tokenizer output must be fresh under .build/flash")
        }
        try fm.createDirectory(at: output, withIntermediateDirectories: false)
        return output
    }

    static func tokenizerIdentity(model: URL) throws -> [String: Any] {
        let pinned = JANGModels.jang6S
        var files: [[String: Any]] = []
        for name in ["tokenizer.json", "tokenizer_config.json", "config.json"] {
            guard let expected = pinned.files.first(where: { $0.path == name }),
                  let expectedDigest = expected.sha256 else {
                throw ValidationError("pinned tokenizer manifest is incomplete")
            }
            let path = model.appendingPathComponent(name)
            let attributes = try FileManager.default.attributesOfItem(atPath: path.path)
            guard FileManager.default.fileExists(atPath: path.path),
                  (try? path.resourceValues(forKeys: [.isRegularFileKey]).isRegularFile) == true,
                  (attributes[.size] as? NSNumber)?.int64Value == expected.size,
                  try digest(path) == expectedDigest else {
                throw ValidationError("pinned local tokenizer file is missing or changed: \(name)")
            }
            files.append(["path": name, "bytes": expected.size, "sha256": expectedDigest])
        }
        let configData = try boundedFile(model.appendingPathComponent("tokenizer_config.json"), limit: 1 << 20)
        guard let config = try JSONSerialization.jsonObject(with: configData) as? [String: Any],
              let template = config["chat_template"] as? String else {
            throw ValidationError("embedded tokenizer chat template is missing")
        }
        return [
            "model_revision": pinned.revision,
            "swift_transformers_revision": transformerRevision,
            "files": files,
            "embedded_chat_template_sha256": digest(Data(template.utf8)),
            "text_add_special_tokens": false,
            "thinking": false,
        ]
    }

    static func executableIdentity() throws -> [String: Any] {
        let executable = URL(fileURLWithPath: CommandLine.arguments[0]).standardizedFileURL
        let directory = executable.deletingLastPathComponent()
        var value: [String: Any] = [
            "path": executable.path,
            "binary_sha256": try digest(executable),
        ]
        for (key, name) in [
            ("metallib_sha256", "mlx.metallib"),
            ("build_identity_sha256", "build-identity.json"),
            ("source_archive_sha256", "build-source.tar.gz"),
        ] {
            value[key] = try digest(directory.appendingPathComponent(name))
        }
        let archiveReceipt = directory.deletingLastPathComponent().appendingPathComponent("receipt.json")
        if FileManager.default.fileExists(atPath: archiveReceipt.path) {
            value["historical"] = true
            value["archive_receipt_path"] = archiveReceipt.path
            value["archive_receipt_sha256"] = try digest(archiveReceipt)
        } else {
            value["historical"] = false
        }
        return value
    }
}

struct FlashTokenize: ParsableCommand {
    static let configuration = CommandConfiguration(
        commandName: "flash-tokenize",
        abstract: "Freeze authored text into bounded tokenizer-only capture shards")

    @Option var model: String
    @Option var source: String
    @Option var output: String

    func run() throws {
        let destination = try CorpusSupport.outputDirectory(output)
        do {
            let root = URL(fileURLWithPath: FileManager.default.currentDirectoryPath)
            let sourceURL = URL(fileURLWithPath: source, relativeTo: root).standardizedFileURL
            let sourceData = try CorpusSupport.boundedFile(sourceURL, limit: 8 << 20)
            guard let raw = try JSONSerialization.jsonObject(with: sourceData) as? [String: Any],
                  Set(raw.keys) == Set(["format", "schemaVersion", "manifestSHA256", "documents"]),
                  let rawDocuments = raw["documents"] as? [[String: Any]] else {
                throw ValidationError("authored corpus has unknown fields")
            }
            let request = try JSONDecoder().decode(AuthoredCorpus.self, from: sourceData)
            let sourceManifestHash = try CorpusSupport.canonicalHash(raw, omitting: "manifestSHA256")
            guard request.format == "slotstream-authored-corpus-v1", request.schemaVersion == 1,
                  request.manifestSHA256 == sourceManifestHash,
                  !request.documents.isEmpty, request.documents.count <= 2048,
                  request.documents.count == rawDocuments.count else {
                throw ValidationError("authored corpus identity or document count is invalid")
            }

            var ids = Set<String>()
            var partitions: [String: String] = [:]
            for (document, rawDocument) in zip(request.documents, rawDocuments) {
                let common = Set(["id", "kind", "category", "split", "sourceID", "license",
                                  "partitionKey", "scoreTokens"])
                let expected = document.kind == "raw"
                    ? common.union(["text", "textSHA256", "warmupTokens"])
                    : common.union(["system", "user", "reference", "contentSHA256"])
                guard Set(rawDocument.keys) == expected,
                      CorpusSupport.safeID(document.id), ids.insert(document.id).inserted,
                      CorpusSupport.safeID(document.category),
                      [document.license, document.partitionKey]
                      .allSatisfy({ !$0.isEmpty && $0.utf8.count <= 256 }),
                      !document.sourceID.isEmpty, document.sourceID.utf8.count <= 256,
                      document.sourceID.utf8.allSatisfy({ $0 >= 32 && $0 != 127 }),
                      CorpusSupport.validSplits.contains(document.split),
                      (1 ... 64).contains(document.scoreTokens),
                      partitions[document.partitionKey].map({ $0 == document.split }) ?? true else {
                    throw ValidationError("authored document fields, identity or split are invalid")
                }
                partitions[document.partitionKey] = document.split
                if document.kind == "raw" {
                    guard let text = document.text, let hash = document.textSHA256,
                          !text.isEmpty, text.utf8.count <= 32 << 10,
                          hash == CorpusSupport.digest(Data(text.utf8)),
                          let warmup = document.warmupTokens, (0 ... 256).contains(warmup) else {
                        throw ValidationError("raw authored document is invalid")
                    }
                } else if document.kind == "chat" {
                    guard let system = document.system,
                          let user = document.user, let reference = document.reference,
                          let hash = document.contentSHA256,
                          !user.isEmpty, !reference.isEmpty,
                          [system, user, reference]
                          .allSatisfy({ $0.utf8.count <= 32 << 10 }) else {
                        throw ValidationError("chat authored document is invalid")
                    }
                    let content = ["reference": reference, "system": system, "user": user]
                    let contentHash = try CorpusSupport.canonicalHash(content, omitting: "")
                    guard hash == contentHash else {
                        throw ValidationError("chat authored content hash changed")
                    }
                } else {
                    throw ValidationError("authored document kind is unsupported")
                }
            }

            let modelURL = ModelLocator.resolve(model).resolvingSymlinksInPath()
            let tokenizerIdentity = try CorpusSupport.tokenizerIdentity(model: modelURL)
            let tokenizer = try awaitResult { try await SlotstreamTokenizer(modelDir: modelURL) }
            var tokenized: [[String: Any]] = []
            var totalIDs = 0
            for document in request.documents {
                let full: [Int]
                let promptCount: Int?
                let warmupCount: Int
                if document.kind == "raw" {
                    full = tokenizer.encodeText(document.text!, addSpecialTokens: false)
                    promptCount = nil
                    warmupCount = document.warmupTokens!
                } else {
                    var prefix: [ChatMessage] = []
                    if let system = document.system, !system.isEmpty {
                        prefix.append(ChatMessage(role: "system", content: system))
                    }
                    prefix.append(ChatMessage(role: "user", content: document.user!))
                    let prompt = try tokenizer.encodeChat(prefix, thinking: false,
                                                          addGenerationPrompt: true)
                    let complete = prefix + [ChatMessage(role: "assistant", content: document.reference!)]
                    full = try tokenizer.encodeChat(complete, thinking: false,
                                                    addGenerationPrompt: false)
                    guard !prompt.isEmpty, full.starts(with: prompt) else {
                        throw ValidationError("completed chat tokens do not preserve the prompt prefix")
                    }
                    promptCount = prompt.count
                    warmupCount = prompt.count - 1
                }
                guard warmupCount <= 256, full.count <= 2048,
                      warmupCount >= 0,
                      warmupCount + document.scoreTokens < full.count else {
                    throw ValidationError("tokenized document cannot fit its declared scored window")
                }
                let positions: [[String: Any]] = (0 ..< document.scoreTokens).map { offset in
                    let input = warmupCount + offset
                    return ["position": input, "inputID": full[input], "nextTokenID": full[input + 1]]
                }
                totalIDs += full.count
                guard totalIDs <= 4_000_000 else {
                    throw ValidationError("tokenized corpus metadata exceeds the bounded aggregate")
                }
                var value: [String: Any] = [
                    "id": document.id,
                    "kind": document.kind,
                    "category": document.category,
                    "split": document.split,
                    "sourceID": document.sourceID,
                    "license": document.license,
                    "partitionKey": document.partitionKey,
                    "source_hash": document.textSHA256 ?? document.contentSHA256!,
                    "full_ids": full,
                    "full_count": full.count,
                    "warmup_count": warmupCount,
                    "scored_range": [warmupCount + 1, warmupCount + 1 + document.scoreTokens],
                    "unscored_range": [warmupCount + 1 + document.scoreTokens, full.count],
                    "capture": ["id": document.id, "category": document.category,
                                "sourceID": document.sourceID, "sourceHash": document.textSHA256 ?? document.contentSHA256!,
                                "split": document.split, "warmupIDs": Array(full.prefix(warmupCount)),
                                "positions": positions],
                ]
                if let promptCount { value["prompt_count"] = promptCount }
                tokenized.append(value)
            }

            let shardsURL = destination.appendingPathComponent("shards")
            try FileManager.default.createDirectory(at: shardsURL, withIntermediateDirectories: false)
            var shardEntries: [[String: Any]] = []
            var ordinal = 0
            for split in ["training", "development", "qualification"] {
                var pending: [[String: Any]] = []
                var positions = 0
                func flush() throws {
                    guard !pending.isEmpty else { return }
                    let name = String(format: "shard-%03d.json", ordinal)
                    var shard: [String: Any] = [
                        "format": "slotstream-tokenized-capture-shard-v2",
                        "schemaVersion": 2,
                        "manifestSHA256": "",
                        "tokenSource": "slotstream-auto-tokenizer-chat-template-v1",
                        "split": split,
                        "sourceCorpusSHA256": CorpusSupport.digest(sourceData),
                        "tokenizerIdentity": tokenizerIdentity,
                        "shardID": String(format: "shard-%03d", ordinal),
                        "documents": pending.map { $0["capture"]! },
                    ]
                    shard["manifestSHA256"] = try CorpusSupport.canonicalHash(shard, omitting: "manifestSHA256")
                    let path = shardsURL.appendingPathComponent(name)
                    try CorpusSupport.writeJSON(shard, to: path)
                    shardEntries.append(["path": "shards/\(name)", "split": split,
                                         "positions": positions, "sha256": try CorpusSupport.digest(path)])
                    ordinal += 1
                    pending.removeAll(keepingCapacity: true)
                    positions = 0
                }
                for document in tokenized where document["split"] as? String == split {
                    let count = ((document["capture"] as! [String: Any])["positions"] as! [[String: Any]]).count
                    if positions + count > 64 { try flush() }
                    pending.append(document)
                    positions += count
                }
                try flush()
            }

            let executable = try CorpusSupport.executableIdentity()
            let canonicalValue: [String: Any] = [
                "camelKey": "slash/a newline\n quote\" backslash\\",
                "snake_key": "é and e\u{301}",
                "Mixed_key": ["zValue": true, "a_value": 7],
            ]
            let canonicalText = try CorpusSupport.canonicalJSON(canonicalValue)
            let artifacts = Dictionary(uniqueKeysWithValues: try shardEntries.map { entry in
                let path = destination.appendingPathComponent(entry["path"] as! String)
                return (entry["path"] as! String,
                        ["bytes": try path.resourceValues(forKeys: [.fileSizeKey]).fileSize!,
                         "sha256": try CorpusSupport.digest(path)] as [String: Any])
            })
            let index: [String: Any] = [
                "format": "slotstream-tokenized-corpus-v1",
                "schema_version": 1,
                "qualification": false,
                "model_loaded": false,
                "tokenizer_only": true,
                "source_manifest_path": sourceURL.path,
                "source_manifest_sha256": CorpusSupport.digest(sourceData),
                "source_provenance_scope": "caller metadata preserved; external provenance not verified by tokenizer bridge",
                "tokenizer_identity": tokenizerIdentity,
                "executable_identity": executable,
                "canonical_test_vector": ["value": canonicalValue,
                    "canonical_json": canonicalText,
                    "sha256": CorpusSupport.digest(Data(canonicalText.utf8))],
                "documents": tokenized,
                "shards": shardEntries,
                "artifacts": artifacts,
            ]
            let indexURL = destination.appendingPathComponent("corpus.json")
            let indexData = try CorpusSupport.jsonData(index)
            guard indexData.count <= 64 << 20 else {
                throw ValidationError("tokenized corpus index exceeds metadata quota")
            }
            try indexData.write(to: indexURL, options: .atomic)
            try CorpusSupport.writeJSON([
                "format": "slotstream-tokenized-corpus-completion-v1",
                "corpus_sha256": CorpusSupport.digest(indexData),
                "qualification": false,
            ], to: destination.appendingPathComponent("completion.json"))
        } catch {
            try? CorpusSupport.writeJSON([
                "format": "slotstream-tokenized-corpus-failure-v1",
                "error": String(describing: error),
            ], to: destination.appendingPathComponent("failure.json"))
            throw error
        }
    }

    private func awaitResult<T>(_ operation: @escaping () async throws -> T) throws -> T {
        let semaphore = DispatchSemaphore(value: 0)
        var result: Result<T, Error>?
        Task {
            do { result = .success(try await operation()) }
            catch { result = .failure(error) }
            semaphore.signal()
        }
        semaphore.wait()
        return try result!.get()
    }
}
