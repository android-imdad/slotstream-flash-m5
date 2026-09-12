import ArgumentParser
import Foundation
import Slotstream

private struct PromptSource: Decodable {
    var format: String
    var schemaVersion: Int
    var manifestSHA256: String
    var documents: [PromptSourceDocument]
}

private struct PromptSourceDocument: Decodable {
    var id: String
    var kind: String
    var category: String
    var split: String
    var sourceID: String
    var license: String
    var partitionKey: String
    var text: String?
    var textSHA256: String?
    var system: String?
    var user: String?
    var contentSHA256: String?
}

struct FlashTokenizePrompts: ParsableCommand {
    static let configuration = CommandConfiguration(
        commandName: "flash-tokenize-prompts",
        abstract: "Freeze complete prompts with the pinned tokenizer, without loading the model")

    @Option var model: String
    @Option var source: String
    @Option var output: String

    func run() throws {
        let destination = try CorpusSupport.outputDirectory(output)
        do {
            try freeze(into: destination)
        } catch {
            try? CorpusSupport.writeJSON([
                "format": "slotstream-prompt-corpus-failure-v1",
                "error": String(describing: error),
            ], to: destination.appendingPathComponent("failure.json"))
            throw error
        }
    }

    private func freeze(into destination: URL) throws {
        let root = URL(fileURLWithPath: FileManager.default.currentDirectoryPath)
        let sourceURL = URL(fileURLWithPath: source, relativeTo: root).standardizedFileURL
        let sourceData = try CorpusSupport.boundedFile(sourceURL, limit: 8 << 20)
        guard let raw = try JSONSerialization.jsonObject(with: sourceData) as? [String: Any],
              Set(raw.keys) == Set(["format", "schemaVersion", "manifestSHA256", "documents"]),
              let rawDocuments = raw["documents"] as? [[String: Any]] else {
            throw ValidationError("prompt source has unknown fields or invalid types")
        }
        let request = try JSONDecoder().decode(PromptSource.self, from: sourceData)
        guard request.format == "slotstream-prompt-source-v1",
              request.schemaVersion == 1,
              request.manifestSHA256 == (try CorpusSupport.canonicalHash(raw, omitting: "manifestSHA256")),
              !request.documents.isEmpty,
              request.documents.count <= 2048,
              request.documents.count == rawDocuments.count else {
            throw ValidationError("prompt source identity or document count is invalid")
        }

        try validate(request.documents, raw: rawDocuments)
        let modelURL = ModelLocator.resolve(model).resolvingSymlinksInPath()
        let tokenizerIdentity = try CorpusSupport.tokenizerIdentity(model: modelURL)
        let tokenizer = try awaitResult { try await SlotstreamTokenizer(modelDir: modelURL) }

        let promptsURL = destination.appendingPathComponent("prompts")
        try FileManager.default.createDirectory(at: promptsURL, withIntermediateDirectories: false)
        var entries: [[String: Any]] = []
        var artifacts: [String: Any] = [:]
        var aggregateTokenCount = 0
        var aggregatePromptBytes = 0

        for (ordinal, pair) in zip(request.documents, rawDocuments).enumerated() {
            let document = pair.0
            let rawDocument = pair.1
            let promptIDs: [Int]
            if document.kind == "raw" {
                promptIDs = tokenizer.encodeText(document.text!, addSpecialTokens: false)
            } else {
                var messages: [ChatMessage] = []
                if !document.system!.isEmpty {
                    messages.append(ChatMessage(role: "system", content: document.system!))
                }
                messages.append(ChatMessage(role: "user", content: document.user!))
                promptIDs = try tokenizer.encodeChat(
                    messages, thinking: false, addGenerationPrompt: true, tools: [])
            }
            guard !promptIDs.isEmpty, promptIDs.count <= 8192,
                  promptIDs.allSatisfy({ 0 <= $0 && $0 < 248_320 }) else {
                throw ValidationError("tokenized prompt is empty, too long, or outside the pinned vocabulary")
            }
            aggregateTokenCount += promptIDs.count
            guard aggregateTokenCount <= 4_000_000 else {
                throw ValidationError("tokenized prompts exceed the aggregate token quota")
            }

            let canonicalIDs = try CorpusSupport.canonicalJSON(promptIDs)
            let idsHash = CorpusSupport.digest(Data(canonicalIDs.utf8))
            let contentHash = document.textSHA256 ?? document.contentSHA256!
            let value: [String: Any] = [
                "format": "slotstream-tokenized-prompt-v1",
                "schemaVersion": 1,
                "sourceDocument": sourceMetadata(rawDocument, kind: document.kind),
                "promptIDs": promptIDs,
                "promptCount": promptIDs.count,
                "promptIDsSHA256": idsHash,
            ]
            let data = try CorpusSupport.jsonData(value)
            aggregatePromptBytes += data.count
            guard aggregatePromptBytes <= 128 << 20 else {
                throw ValidationError("tokenized prompt artifacts exceed the output quota")
            }
            let name = String(format: "prompt-%04d.json", ordinal)
            let relative = "prompts/\(name)"
            let path = promptsURL.appendingPathComponent(name)
            try data.write(to: path, options: .atomic)
            let fileHash = CorpusSupport.digest(data)
            entries.append([
                "id": document.id,
                "path": relative,
                "prompt_count": promptIDs.count,
                "prompt_ids_sha256": idsHash,
                "source_hash": contentHash,
                "artifact_bytes": data.count,
                "artifact_sha256": fileHash,
            ])
            artifacts[relative] = ["bytes": data.count, "sha256": fileHash]
        }

        let canonicalValue: [String: Any] = [
            "camelKey": "slash/a newline\n quote\" backslash\\",
            "snake_key": "é and e\u{301}",
            "Mixed_key": ["zValue": true, "a_value": 7],
        ]
        let canonicalText = try CorpusSupport.canonicalJSON(canonicalValue)
        let index: [String: Any] = [
            "format": "slotstream-prompt-corpus-v1",
            "schema_version": 1,
            "qualification": false,
            "model_loaded": false,
            "tokenizer_only": true,
            "source_manifest_path": sourceURL.path,
            "source_manifest_sha256": CorpusSupport.digest(sourceData),
            "source_provenance_scope": "caller metadata preserved; external provenance not verified by tokenizer bridge",
            "tokenizer_identity": tokenizerIdentity,
            "executable_identity": try CorpusSupport.executableIdentity(),
            "canonical_test_vector": [
                "value": canonicalValue,
                "canonical_json": canonicalText,
                "sha256": CorpusSupport.digest(Data(canonicalText.utf8)),
            ],
            "context": [
                "artifact_token_limit": 8192,
                "inference_context_limit": 2048,
                "artifact_limit_is_inference_permission": false,
            ],
            "documents": entries,
            "aggregate": [
                "document_count": entries.count,
                "prompt_token_count": aggregateTokenCount,
                "prompt_artifact_bytes": aggregatePromptBytes,
            ],
            "artifacts": artifacts,
        ]
        let indexData = try CorpusSupport.jsonData(index)
        guard aggregatePromptBytes + indexData.count + 1024 <= 128 << 20 else {
            throw ValidationError("tokenized prompt output exceeds the aggregate byte quota")
        }
        let indexURL = destination.appendingPathComponent("prompts.json")
        try indexData.write(to: indexURL, options: .atomic)
        try CorpusSupport.writeJSON([
            "format": "slotstream-prompt-corpus-completion-v1",
            "prompts_sha256": CorpusSupport.digest(indexData),
            "qualification": false,
        ], to: destination.appendingPathComponent("completion.json"))
    }

    private func validate(_ documents: [PromptSourceDocument], raw: [[String: Any]]) throws {
        var ids = Set<String>()
        var partitions: [String: String] = [:]
        let common = Set(["id", "kind", "category", "split", "sourceID", "license", "partitionKey"])
        for (document, rawDocument) in zip(documents, raw) {
            let expected: Set<String>
            if document.kind == "raw" {
                expected = common.union(["text", "textSHA256"])
            } else if document.kind == "chat" {
                expected = common.union(["system", "user", "contentSHA256"])
            } else {
                throw ValidationError("prompt document kind is unsupported")
            }
            guard Set(rawDocument.keys) == expected,
                  CorpusSupport.safeID(document.id), ids.insert(document.id).inserted,
                  CorpusSupport.safeID(document.category),
                  CorpusSupport.validSplits.contains(document.split),
                  [document.sourceID, document.license, document.partitionKey]
                  .allSatisfy({ !$0.isEmpty && $0.utf8.count <= 256 }),
                  document.sourceID.utf8.allSatisfy({ $0 >= 32 && $0 != 127 }),
                  partitions[document.partitionKey].map({ $0 == document.split }) ?? true else {
                throw ValidationError("prompt document fields, identity, or split are invalid")
            }
            partitions[document.partitionKey] = document.split
            if document.kind == "raw" {
                guard let text = document.text, let hash = document.textSHA256,
                      !text.isEmpty, text.utf8.count <= 32 << 10,
                      hash == CorpusSupport.digest(Data(text.utf8)) else {
                    throw ValidationError("raw prompt document is invalid")
                }
            } else {
                guard let system = document.system, let user = document.user,
                      let hash = document.contentSHA256,
                      !user.isEmpty,
                      system.utf8.count <= 32 << 10, user.utf8.count <= 32 << 10,
                      hash == (try CorpusSupport.canonicalHash(
                          ["system": system, "user": user], omitting: "")) else {
                    throw ValidationError("chat prompt document is invalid")
                }
            }
        }
    }

    private func sourceMetadata(_ raw: [String: Any], kind: String) -> [String: Any] {
        var result = raw
        result.removeValue(forKey: kind == "raw" ? "text" : "system")
        if kind == "chat" { result.removeValue(forKey: "user") }
        return result
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
