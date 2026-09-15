import ArgumentParser
import CryptoKit
import Darwin
import Foundation
import MLX
import Slotstream
import SlotstreamDiagnostics

private struct M5CommandProvenance: Codable {
    var executablePath: String
    var executableSHA256: String
    var metallibSHA256: String
    var buildIdentitySHA256: String
    var sourceArchiveSHA256: String
}

private struct M5CommandReport: Codable {
    var format = "slotstream-m5-command-v1"
    var schemaVersion = 1
    var processID: Int32
    var mode: String
    var provenance: M5CommandProvenance
    var diagnostic: M5SyntheticReport
    var boundedModelCheck: CheckReport?
    var modelPath: String?
    var modelScope: String?
}

struct M5Check: ParsableCommand {
    static let configuration = CommandConfiguration(
        commandName: "m5-check",
        abstract: "Measure bounded M5 quantized expert-kernel eligibility and correctness")

    @Flag(help: "Use deterministic generated expert tensors; never load model weights")
    var synthetic = false

    @Option(help: "Use only bounded original weight-row samples from this existing model directory")
    var model: String?

    @Option(help: "Fresh evidence directory under .build/flash")
    var output: String

    @Option(name: .customLong("trace-case"), help: "Diagnostic-only selected case: grouped6 or decode6")
    var traceCase: String?

    private func digest(_ url: URL) throws -> String {
        let data = try Data(contentsOf: url, options: .mappedIfSafe)
        return SHA256.hash(data: data).map { String(format: "%02x", $0) }.joined()
    }

    private func freshOutput() throws -> URL {
        guard synthetic != (model != nil) else {
            throw ValidationError("choose exactly one of --synthetic or --model")
        }
        if traceCase != nil, !synthetic {
            throw ValidationError("--trace-case requires --synthetic")
        }
        let fm = FileManager.default
        let root = URL(fileURLWithPath: fm.currentDirectoryPath).standardizedFileURL
        let flash = root.appendingPathComponent(".build/flash").resolvingSymlinksInPath()
        let requested = URL(fileURLWithPath: output, relativeTo: root).standardizedFileURL
        let parent = requested.deletingLastPathComponent()
        guard fm.fileExists(atPath: parent.path),
              parent.resolvingSymlinksInPath().path.hasPrefix(flash.path + "/"),
              requested.path.hasPrefix(root.appendingPathComponent(".build/flash").path + "/") else {
            throw ValidationError("--output must have an existing safe parent under .build/flash")
        }
        guard !fm.fileExists(atPath: requested.path) else {
            throw ValidationError("--output already exists: \(requested.path)")
        }
        try fm.createDirectory(at: requested, withIntermediateDirectories: false)
        return requested
    }

    private func provenance() throws -> M5CommandProvenance {
        let executable = URL(fileURLWithPath: CommandLine.arguments[0]).standardizedFileURL.resolvingSymlinksInPath()
        let directory = executable.deletingLastPathComponent()
        let metallib = directory.appendingPathComponent("mlx.metallib")
        let identity = directory.appendingPathComponent("build-identity.json")
        let source = directory.appendingPathComponent("build-source.tar.gz")
        for artifact in [executable, metallib, identity, source] {
            guard FileManager.default.fileExists(atPath: artifact.path) else {
                throw ValidationError("missing build artifact beside diagnostic executable: \(artifact.lastPathComponent)")
            }
        }
        return try M5CommandProvenance(executablePath: executable.path,
            executableSHA256: digest(executable), metallibSHA256: digest(metallib),
            buildIdentitySHA256: digest(identity), sourceArchiveSHA256: digest(source))
    }

    func run() throws {
        let destination = try freshOutput()
        do {
            let mode = synthetic ? "synthetic" : "bounded-model-rows"
            let diagnostic = try traceCase.map { try Diagnostics.m5TraceCase($0) }
                ?? Diagnostics.m5Synthetic(full: true, mode: mode)
            let bounded: CheckReport?
            let modelPath: String?
            if let model {
                let url = ModelLocator.resolve(model).resolvingSymlinksInPath()
                bounded = try Diagnostics.jangExpertStreaming(modelDir: url)
                modelPath = url.path
            } else {
                bounded = nil
                modelPath = nil
            }
            guard diagnostic.check.passed, bounded?.passed != false else {
                throw ValidationError("M5 component checks failed")
            }
            let report = try M5CommandReport(processID: getpid(), mode: diagnostic.mode, provenance: provenance(),
                diagnostic: diagnostic, boundedModelCheck: bounded, modelPath: modelPath,
                modelScope: model == nil ? nil : "bounded original rows only; no full model or generation")
            let encoder = JSONEncoder()
            encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
            let reportURL = destination.appendingPathComponent("report.json")
            try encoder.encode(report).write(to: reportURL, options: .atomic)
            let completion: [String: Any] = [
                "format": "slotstream-m5-completion-v1", "schema_version": 1,
                "report_sha256": try digest(reportURL), "passed": true,
                "observation_status": diagnostic.observationStatus,
            ]
            try JSONSerialization.data(withJSONObject: completion, options: [.prettyPrinted, .sortedKeys])
                .write(to: destination.appendingPathComponent("completion.json"), options: .atomic)
            print(reportURL.path)
        } catch {
            let failure = ["format": "slotstream-m5-failure-v1", "error": String(describing: error)]
            try? JSONSerialization.data(withJSONObject: failure, options: [.prettyPrinted, .sortedKeys])
                .write(to: destination.appendingPathComponent("failure.json"), options: .atomic)
            throw error
        }
    }
}

private struct CapturePosition: Codable {
    var position: Int
    var inputID: Int
    var nextTokenID: Int
}
private struct CaptureDocument: Codable {
    var id: String
    var split: String
    var warmupIDs: [Int]
    var positions: [CapturePosition]
}
private struct CaptureManifest: Codable {
    var format: String
    var schemaVersion: Int
    var manifestSHA256: String
    var developmentOnly: Bool
    var tokenSource: String
    var documents: [CaptureDocument]
}

private struct TokenizedCaptureDocument: Codable {
    var id: String
    var category: String
    var sourceID: String
    var sourceHash: String
    var split: String
    var warmupIDs: [Int]
    var positions: [CapturePosition]
}

private struct TokenizedCaptureManifest: Codable {
    var format: String
    var schemaVersion: Int
    var manifestSHA256: String
    var tokenSource: String
    var split: String
    var sourceCorpusSHA256: String
    var tokenizerIdentity: [String: JSONValue]
    var shardID: String
    var documents: [TokenizedCaptureDocument]
}

private enum JSONValue: Codable {
    case string(String), int(Int), bool(Bool), array([JSONValue]), object([String: JSONValue]), null

    init(from decoder: Decoder) throws {
        let box = try decoder.singleValueContainer()
        if box.decodeNil() { self = .null }
        else if let value = try? box.decode(String.self) { self = .string(value) }
        else if let value = try? box.decode(Int.self) { self = .int(value) }
        else if let value = try? box.decode(Bool.self) { self = .bool(value) }
        else if let value = try? box.decode([JSONValue].self) { self = .array(value) }
        else { self = .object(try box.decode([String: JSONValue].self)) }
    }

    func encode(to encoder: Encoder) throws {
        var box = encoder.singleValueContainer()
        switch self {
        case .string(let value): try box.encode(value)
        case .int(let value): try box.encode(value)
        case .bool(let value): try box.encode(value)
        case .array(let value): try box.encode(value)
        case .object(let value): try box.encode(value)
        case .null: try box.encodeNil()
        }
    }
}

private final class NativeCaptureSink: FlashObservationSink {
    let root: URL
    let liveLimit: Int
    let activationQuota: Int64
    let logitsQuota: Int64
    var activationBytes: Int64 = 0
    var logitsBytes: Int64 = 0
    var files: [[String: Any]] = []
    private var hiddenPartCounts: [String: Int] = [:]
    private let lock = NSLock()
    private var cancelled = false
    private var documentID = ""
    private var tokenPosition = -1
    private var inputID = -1
    private var metadataBytes = 0
    private let boundedWriter: FlashBoundedWriter

    init(root: URL, liveLimit: Int, activationQuota: Int64, logitsQuota: Int64) {
        self.root = root; self.liveLimit = liveLimit
        self.activationQuota = activationQuota; self.logitsQuota = logitsQuota
        self.boundedWriter = FlashBoundedWriter(
            maxBytes: activationQuota + logitsQuota, liveLimit: liveLimit)
    }
    func validateForward(tokens: Int) throws {
        lock.lock(); let stopped = cancelled; lock.unlock()
        guard !stopped else { throw PlanError("capture cancelled") }
        guard tokens == 1 else { throw PlanError("activation capture requires exactly one token per forward") }
    }
    func begin(documentID: String, tokenPosition: Int, inputID: Int) {
        self.documentID=documentID; self.tokenPosition=tokenPosition; self.inputID=inputID
    }
    func cancel() { lock.lock(); cancelled=true; lock.unlock(); boundedWriter.cancel() }
    private func write(_ name: String, data: Data, category: String) throws {
        guard data.count <= liveLimit else { throw PlanError("capture copy exceeds live-buffer limit") }
        let next = (category == "logits" ? logitsBytes : activationBytes) + Int64(data.count)
        let limit = category == "logits" ? logitsQuota : activationQuota
        guard next <= limit else { throw PlanError("capture \(category) quota exceeded") }
        try boundedWriter.write(data,to:root.appendingPathComponent(name).path)
        if category == "logits" { logitsBytes = next } else { activationBytes = next }
        // Metadata is bounded separately from payload and the final JSON report.
    }
    private func validateCategoryQuota(bytes: Int, category: String) throws {
        guard bytes >= 0, bytes <= liveLimit else {
            throw PlanError("capture copy exceeds live-buffer limit")
        }
        let current = category == "logits" ? logitsBytes : activationBytes
        let limit = category == "logits" ? logitsQuota : activationQuota
        guard Int64(bytes) <= limit, current <= limit - Int64(bytes) else {
            throw PlanError("capture \(category) quota exceeded")
        }
    }
    func observeInput(layer: Int, value: MLXArray) throws {
        try validateCategoryQuota(bytes: value.nbytes, category: "activation")
        eval(value); let copy = value.asData(access: .copy)
        let name="activation-\(files.count).bin"; try write(name, data: copy.data, category: "activation")
        try appendMetadata(["name":name,"category":"input","document_id":documentID,"token_position":tokenPosition,
            "input_id":inputID,"layer":layer,"dtype":String(describing:copy.dType),"shape":copy.shape,
            "bytes":copy.data.count,"sha256":SHA256.hash(data:copy.data).map{String(format:"%02x",$0)}.joined()])
    }
    func observeRoute(layer: Int, expertIDs: [Int32]) throws {
        guard expertIDs.count == 10 else { throw PlanError("capture route must contain ten ranks") }
    }
    func observeHidden(layer: Int, routerRanks: [Int], expertIDs: [Int32], value: MLXArray) throws {
        guard routerRanks.count == expertIDs.count,
              Set(routerRanks).count == routerRanks.count,
              routerRanks.allSatisfy({ (0..<10).contains($0) }) else {
            throw PlanError("capture hidden rows have invalid router-rank mapping")
        }
        try validateCategoryQuota(bytes: value.nbytes, category: "activation")
        eval(value); let copy = value.asData(access: .copy)
        let key = "\(documentID):\(tokenPosition):\(layer)"
        hiddenPartCounts[key, default: 0] += 1
        let name="activation-\(files.count).bin"; try write(name, data: copy.data, category: "activation")
        try appendMetadata(["name":name,"category":"hidden","document_id":documentID,"token_position":tokenPosition,
            "input_id":inputID,"layer":layer,"dtype":String(describing:copy.dType),"shape":copy.shape,
            "router_ranks":routerRanks,"expert_ids":expertIDs,"bytes":copy.data.count,
            "sha256":SHA256.hash(data:copy.data).map{String(format:"%02x",$0)}.joined()])
    }
    func writeLogits(_ value: MLXArray, name: String) throws {
        let fp32 = value.asType(.float32)
        try validateCategoryQuota(bytes: fp32.nbytes, category: "logits")
        eval(fp32); let data=fp32.asData(access:.copy).data
        try write(name, data: data, category: "logits")
        try appendMetadata(["name":name,"category":"logits","document_id":documentID,
            "token_position":tokenPosition,"input_id":inputID,"dtype":"float32","shape":fp32.shape,
            "bytes":data.count,"sha256":SHA256.hash(data:data).map{String(format:"%02x",$0)}.joined()])
    }
    private func appendMetadata(_ entry:[String:Any]) throws {
        let size=try JSONSerialization.data(withJSONObject:entry).count
        guard metadataBytes <= liveLimit-size else { throw PlanError("capture metadata exceeds live-buffer limit") }
        metadataBytes += size; files.append(entry)
    }
    var splitLayerKeys: [String] {
        hiddenPartCounts.filter { $0.value > 1 }.map(\.key).sorted()
    }
}

private final class RejectingCaptureSink: FlashObservationSink {
    func validateForward(tokens: Int) throws {}
    func observeInput(layer: Int, value: MLXArray) throws { throw PlanError("injected capture sink failure") }
    func observeRoute(layer: Int, expertIDs: [Int32]) throws {}
    func observeHidden(layer: Int, routerRanks: [Int], expertIDs: [Int32], value: MLXArray) throws {}
}

struct FlashCapture: ParsableCommand {
    static let configuration = CommandConfiguration(commandName: "flash-capture",
        abstract: "Capture a bounded reference or explicit diagnostic neuron oracle")
    @Option var model: String
    @Option var manifest: String
    @Option var split: String
    @Option var mode: String
    @Option(name: .customLong("memory-gb")) var memoryGB: Double = 14
    @Option(name: .customLong("max-context")) var maxContext: Int = 2048
    @Option var output: String
    @Option(name: .customLong("live-limit-mb")) var liveLimitMB = 64
    @Option(name: .customLong("activation-quota-gb")) var activationQuotaGB = 32
    @Option(name: .customLong("logits-quota-gb")) var logitsQuotaGB = 8
    @Option(name: .customLong("position-limit")) var positionLimit = 64
    @Option(name: .customLong("expert-widening")) var expertWidening = AffineWideningPolicy.scalar.rawValue
    @Option(name: .customLong("oracle-config")) var oracleConfig: String?
    @Flag(name: .customLong("require-resident-split")) var requireResidentSplit = false
    private func digest(_ url:URL) throws -> String {
        SHA256.hash(data:try Data(contentsOf:url,options:.mappedIfSafe)).map{String(format:"%02x",$0)}.joined()
    }

    func run() throws {
        guard ["reference-off", "reference-on", "oracle"].contains(mode),
              (mode == "oracle") == (oracleConfig != nil) else {
            throw ValidationError("--mode must be reference-off, reference-on, or oracle with --oracle-config")
        }
        guard let wideningPolicy = AffineWideningPolicy(rawValue: expertWidening) else {
            throw ValidationError("--expert-widening must be scalar or packed4-to6")
        }
        guard mode != "oracle" || (split == "development" && liveLimitMB == 64
            && wideningPolicy == .scalar && !requireResidentSplit) else {
            throw ValidationError("oracle requires development, 64 MiB live allowance, scalar widening and no resident split")
        }
        guard ["training", "development", "qualification"].contains(split), maxContext == 2048, memoryGB == 14, (32...64).contains(liveLimitMB),
              (1...32).contains(activationQuotaGB), (1...8).contains(logitsQuotaGB),
              positionLimit == 64, !requireResidentSplit || mode == "reference-on" else {
            throw ValidationError("capture requires training/development, --memory-gb 14, --max-context 2048, position limit 64 and live limit 32...64 MiB")
        }
        let environment = ProcessInfo.processInfo.environment
        let allowedSlotstream = requireResidentSplit ? Set(["SLOTSTREAM_OPT_RESIDENT_OVERLAP"]) : Set<String>()
        let ambientSlotstream = environment.keys.filter { $0.hasPrefix("SLOTSTREAM_") && !allowedSlotstream.contains($0) }
        guard ambientSlotstream.isEmpty,
              environment["MLX_ENABLE_TF32"] == nil || environment["MLX_ENABLE_TF32"] == "0" || environment["MLX_ENABLE_TF32"] == "1" else {
            throw ValidationError("capture has unsupported Slotstream or MLX numerical environment controls")
        }
        let fm = FileManager.default, root = URL(fileURLWithPath: fm.currentDirectoryPath).standardizedFileURL
        let destination = URL(fileURLWithPath: output, relativeTo: root).standardizedFileURL
        let flash = root.appendingPathComponent(".build/flash").standardizedFileURL
        let parent=destination.deletingLastPathComponent()
        guard destination.path.hasPrefix(flash.path + "/"), !fm.fileExists(atPath: destination.path),
              fm.fileExists(atPath: parent.path), parent.resolvingSymlinksInPath().path.hasPrefix(flash.resolvingSymlinksInPath().path + "/") else {
            throw ValidationError("capture output must be fresh with an existing parent under .build/flash")
        }
        try fm.createDirectory(at: destination, withIntermediateDirectories: false)
        do {
            guard split != "qualification" else {
                throw ValidationError("qualification capture is locked until a later frozen run-set")
            }
            let manifestURL = URL(fileURLWithPath: manifest, relativeTo: root).standardizedFileURL
            var manifestBefore=stat()
            guard lstat(manifestURL.path,&manifestBefore)==0,
                  manifestBefore.st_mode & S_IFMT == S_IFREG,
                  manifestBefore.st_size >= 0, manifestBefore.st_size <= 1<<20 else {
                throw ValidationError("capture manifest must be a regular file no larger than 1 MiB")
            }
            let manifestHandle = try FileHandle(forReadingFrom: manifestURL)
            let manifestData: Data
            do { manifestData = try manifestHandle.read(upToCount:(1<<20)+1) ?? Data(); try manifestHandle.close() }
            catch { try? manifestHandle.close(); throw error }
            var manifestAfter=stat()
            guard lstat(manifestURL.path,&manifestAfter)==0,
                  manifestData.count<=1<<20,manifestData.count==Int(manifestBefore.st_size),
                  manifestBefore.st_dev==manifestAfter.st_dev,manifestBefore.st_ino==manifestAfter.st_ino,
                  manifestBefore.st_size==manifestAfter.st_size,
                  manifestBefore.st_mtimespec.tv_sec==manifestAfter.st_mtimespec.tv_sec,
                  manifestBefore.st_mtimespec.tv_nsec==manifestAfter.st_mtimespec.tv_nsec,
                  let raw=try JSONSerialization.jsonObject(with:manifestData) as? [String:Any],
                  let format=raw["format"] as? String else {
                throw ValidationError("capture manifest changed while read or has unknown fields")
            }
            var canonical = raw
            canonical.removeValue(forKey: "manifestSHA256")
            let canonicalData = try JSONSerialization.data(withJSONObject: canonical, options: [.sortedKeys])
            let canonicalHash = SHA256.hash(data: canonicalData).map { String(format: "%02x", $0) }.joined()
            let captureDocuments: [CaptureDocument]
            var tokenizedMetadata: [String: Any]? = nil
            if format == "slotstream-flash-capture-v1" {
                guard Set(raw.keys)==Set(["format","schemaVersion","manifestSHA256","developmentOnly","tokenSource","documents"]),
                      split == "development" else {
                    throw ValidationError("v1 capture remains development-only")
                }
                let request = try JSONDecoder().decode(CaptureManifest.self, from: manifestData)
                guard request.schemaVersion == 1, request.manifestSHA256 == canonicalHash,
                      request.developmentOnly,
                      request.tokenSource == "arbitrary-valid-diagnostic-ids-not-training-text" else {
                    throw ValidationError("v1 capture manifest identity is invalid")
                }
                captureDocuments = request.documents
            } else if format == "slotstream-tokenized-capture-shard-v2" {
                guard Set(raw.keys)==Set(["format","schemaVersion","manifestSHA256","tokenSource","split",
                                          "sourceCorpusSHA256","tokenizerIdentity","shardID","documents"]),
                      let rawIdentity=raw["tokenizerIdentity"] as? [String:Any] else {
                    throw ValidationError("v2 tokenized capture has unknown fields")
                }
                let request = try JSONDecoder().decode(TokenizedCaptureManifest.self, from: manifestData)
                let tokenizedHash = try CorpusSupport.canonicalHash(raw, omitting:"manifestSHA256")
                let shardSuffix = request.shardID.hasPrefix("shard-")
                    ? request.shardID.dropFirst("shard-".count) : Substring()
                guard request.schemaVersion == 2, request.manifestSHA256 == tokenizedHash,
                      request.tokenSource == "slotstream-auto-tokenizer-chat-template-v1",
                      request.split == split, ["training","development"].contains(split),
                      request.sourceCorpusSHA256.utf8.count == 64,
                      request.sourceCorpusSHA256.utf8.allSatisfy({ (48...57).contains($0) || (97...102).contains($0) }),
                      (3...4).contains(shardSuffix.count),
                      shardSuffix.utf8.allSatisfy({ (48...57).contains($0) }) else {
                    throw ValidationError("v2 tokenized capture identity or split is invalid")
                }
                let modelURL = ModelLocator.resolve(model).resolvingSymlinksInPath()
                let currentIdentity = try CorpusSupport.tokenizerIdentity(model: modelURL)
                guard NSDictionary(dictionary: rawIdentity).isEqual(to: currentIdentity) else {
                    throw ValidationError("v2 tokenizer identity differs from the local pinned model")
                }
                captureDocuments = request.documents.map {
                    CaptureDocument(id:$0.id,split:$0.split,warmupIDs:$0.warmupIDs,positions:$0.positions)
                }
                tokenizedMetadata = ["format":format,"split":split,"shard_id":request.shardID,
                                     "source_corpus_sha256":request.sourceCorpusSHA256,
                                     "tokenizer_identity":rawIdentity,
                                     "documents":request.documents.map { ["id":$0.id,"category":$0.category,
                                         "source_id":$0.sourceID,"source_hash":$0.sourceHash] }]
            } else {
                throw ValidationError("unsupported capture manifest version")
            }
            var documentIDs=Set<String>(), totalPositions=0
            let v2DocumentKeys=Set(["id","category","sourceID","sourceHash","split","warmupIDs","positions"])
            guard let rawDocuments=raw["documents"] as? [[String:Any]], rawDocuments.count==captureDocuments.count,
                  rawDocuments.allSatisfy({ Set($0.keys)==(format == "slotstream-flash-capture-v1"
                    ? Set(["id","split","warmupIDs","positions"]):v2DocumentKeys)
                    && (($0["positions"] as? [[String:Any]])?.allSatisfy({Set($0.keys)==Set(["position","inputID","nextTokenID"])}) ?? false) }) else {
                throw ValidationError("capture manifest has unknown nested fields")
            }
            for (document,rawDocument) in zip(captureDocuments,rawDocuments) {
                let safeID = !document.id.isEmpty && document.id.utf8.allSatisfy {
                    (48...57).contains($0) || (65...90).contains($0) ||
                        (97...122).contains($0) || $0 == 45 || $0 == 95
                }
                guard documentIDs.insert(document.id).inserted,
                      document.split == split,
                      document.id.utf8.count <= 64,
                      safeID,
                      document.warmupIDs.count <= 256,
                      document.warmupIDs.count + document.positions.count <= maxContext,
                      document.warmupIDs.allSatisfy({(0..<248320).contains($0)}) else {
                    throw ValidationError("capture document ID, count or context is invalid")
                }
                if format == "slotstream-tokenized-capture-shard-v2" {
                    guard let category=rawDocument["category"] as? String,CorpusSupport.safeID(category),
                          let sourceID=rawDocument["sourceID"] as? String,!sourceID.isEmpty,sourceID.utf8.count<=256,
                          sourceID.utf8.allSatisfy({$0>=32 && $0 != 127}),
                          let sourceHash=rawDocument["sourceHash"] as? String,sourceHash.utf8.count==64,
                          sourceHash.utf8.allSatisfy({(48...57).contains($0) || (97...102).contains($0)}),
                          !document.positions.isEmpty else {
                        throw ValidationError("v2 capture source metadata is invalid")
                    }
                }
                var expected=document.warmupIDs.count, prior:Int?
                for position in document.positions {
                    guard position.position==expected, (0..<248320).contains(position.inputID),
                          (0..<248320).contains(position.nextTokenID), prior == nil || prior==position.inputID else {
                        throw ValidationError("capture position or adjacent i+1 chain is invalid")
                    }
                    prior=position.nextTokenID; expected += 1; totalPositions += 1
                }
            }
            guard totalPositions>0,totalPositions<=positionLimit else { throw ValidationError("capture position limit exceeded") }
            let metadataBytesPerPosition = 48 * 3 * 1024 + 256 * 1024
            let (metadataReserve,metadataOverflow)=totalPositions.multipliedReportingOverflow(by:metadataBytesPerPosition)
            guard !metadataOverflow, metadataReserve <= liveLimitMB << 20 else {
                throw ValidationError("capture metadata and report reserve exceeds the live-buffer limit")
            }
            let (logitBytes,logitOverflow)=totalPositions.multipliedReportingOverflow(by:248320*4)
            let (activationBytes,activationOverflow)=totalPositions.multipliedReportingOverflow(by:48*(2560*2+10*640*4))
            guard !logitOverflow,!activationOverflow,logitBytes<=logitsQuotaGB*1_000_000_000,
                  activationBytes<=activationQuotaGB*1_000_000_000,248320*4<=liveLimitMB<<20 else {
                throw ValidationError("capture manifest exceeds declared quota or live-copy bounds")
            }
            let modelURL = ModelLocator.resolve(model).resolvingSymlinksInPath()
            let index = try CheckpointIndex(dir: modelURL)
            let oracleConfiguration = try oracleConfig.map { try OracleConfiguration.load($0,
                manifestHash: SHA256.hash(data: manifestData).map { String(format: "%02x", $0) }.joined()) }
            guard oracleConfiguration == nil || index.config.format == .jang6S else {
                throw PlanError("oracle requires JANG_6S")
            }
            let reservation = liveLimitMB << 20
            let policy = try RuntimeAllocationPolicy(prefixCacheEnabled: false,
                diagnosticReservedBytes: reservation * 2 + (oracleConfiguration == nil ? 0 : OracleConfiguration.extraReservation))
            let plan = try CheckpointMemory(index: index).plan(memoryGB: memoryGB,
                maxContext: maxContext, policy: policy)
            let sem = DispatchSemaphore(value: 0)
            var result: Result<Void, Error> = .success(())
            Task {
                do {
                    // The charged plan exists before allocating the sole norm table.
                    let oracle = try oracleConfiguration.map { try NativeNeuronOracle(configuration: $0,
                        configurationPath: oracleConfig!, model: modelURL, destination: destination) }
                    let engine = try await Engine(modelDir: modelURL, plan: plan,
                        expertWidening: wideningPolicy)
                    guard engine.effectiveWideningPolicy == wideningPolicy else {
                        throw PlanError("effective expert widening does not match the capture request")
                    }
                    guard engine.model.optimizations.overlapResidentExperts == requireResidentSplit else {
                        throw PlanError("effective resident-overlap control does not match the capture request")
                    }
                    let sink = NativeCaptureSink(root: destination, liveLimit: reservation,
                        activationQuota: Int64(activationQuotaGB) * 1_000_000_000,
                        logitsQuota: Int64(logitsQuotaGB) * 1_000_000_000)
                    let previousSIGINT = Darwin.signal(SIGINT, SIG_IGN)
                    let cancellation=DispatchSource.makeSignalSource(signal:SIGINT,queue:.global())
                    cancellation.setEventHandler { sink.cancel(); oracle?.cancel() }; cancellation.resume()
                    defer {
                        engine.model.flashHiddenTransform = nil
                        engine.model.flashObservationSink = nil
                        cancellation.cancel(); Darwin.signal(SIGINT, previousSIGINT)
                    }
                    let invalidState = engine.model.makeState()
                    let rejectingSink = RejectingCaptureSink()
                    engine.model.flashObservationSink = rejectingSink
                    do {
                        _ = try engine.model.lastLogitsChecked([1], state: invalidState)
                        throw PlanError("injected observation failure did not propagate")
                    } catch let error where String(describing:error).contains("injected capture sink failure") {}
                    engine.model.flashObservationSink = nil
                    var reuseRefused = false
                    do { _ = try engine.model.lastLogitsChecked([1], state: invalidState) }
                    catch { reuseRefused = String(describing:error).contains("incomplete forward") }
                    guard reuseRefused else { throw PlanError("partially advanced state was reusable after sink failure") }
                    var maskReuseRefused = false
                    if oracle != nil {
                        let maskState = engine.model.makeState()
                        engine.model.flashHiddenTransform = RejectingOracleTransform()
                        do {
                            _ = try engine.model.lastLogitsChecked([1], state: maskState)
                            throw PlanError("injected oracle failure did not propagate")
                        } catch let error where String(describing: error).contains("injected oracle mask failure") {}
                        engine.model.flashHiddenTransform = nil
                        do { _ = try engine.model.lastLogitsChecked([1], state: maskState) }
                        catch { maskReuseRefused = String(describing: error).contains("incomplete forward") }
                        guard maskReuseRefused else { throw PlanError("oracle failure left reusable state") }
                    }
                    var documents: [[String: Any]] = []
                    for document in captureDocuments where document.split == split {
                        engine.model.flashHiddenTransform = nil
                        let state = engine.model.makeState()
                        if !document.warmupIDs.isEmpty {
                            let warm = try engine.model.lastLogitsChecked(document.warmupIDs, state: state); eval(warm)
                        }
                        var positions: [[String: Any]] = []
                        for position in document.positions {
                            guard position.position == state.tokenCount,
                                  position.inputID >= 0, position.nextTokenID >= 0 else {
                                throw PlanError("capture position or i+1 target is invalid")
                            }
                            var routes: [[String: Any]] = []
                            engine.model.routerObserver = { layer, ids in routes.append(["layer": layer, "ids": ids]) }
                            engine.model.flashObservationSink = mode == "reference-off" ? nil : sink
                            sink.begin(documentID:document.id,tokenPosition:position.position,inputID:position.inputID)
                            try oracle?.begin(document: document.id, position: position.position)
                            engine.model.flashHiddenTransform = oracle
                            let logits = try engine.model.lastLogitsChecked([position.inputID], state: state)
                            eval(logits)
                            try oracle?.finish()
                            try sink.writeLogits(logits, name: "logits-\(document.id)-\(position.position).bin")
                            let stateIdentity=try state.flashStateIdentity(maxCopyBytes:reservation)
                            let stateEncoder=JSONEncoder();stateEncoder.outputFormatting=[.sortedKeys]
                            let stateData = try stateEncoder.encode(stateIdentity)
                            let stateHash = SHA256.hash(data: stateData).map { String(format: "%02x", $0) }.joined()
                            positions.append(["position": position.position, "input_id": position.inputID,
                                "next_token_id": position.nextTokenID, "routes": routes,
                                "state_sha256": stateHash,
                                "state":try JSONSerialization.jsonObject(with:stateData),
                                "continuation_id": argMax(logits).item(Int.self)])
                            engine.model.flashObservationSink = nil
                            engine.model.flashHiddenTransform = nil
                        }
                        documents.append(["id": document.id, "positions": positions])
                    }
                    let packedWidening = wideningPolicy == .packed4To6
                    var payload: [String: Any] = ["format": oracle != nil ? "slotstream-flash-oracle-output-v1" : packedWidening
                            ? "slotstream-flash-widening-output-v1" : "slotstream-flash-capture-output-v1",
                        "schema_version": 1, "qualification": false, "mode": mode,
                        "manifest_sha256": SHA256.hash(data: manifestData).map { String(format: "%02x", $0) }.joined(),
                        "binary": CommandLine.arguments[0], "model": modelURL.path,
                        "source_identity":["binary_sha256":try digest(URL(fileURLWithPath:CommandLine.arguments[0])),
                            "metallib_sha256":try digest(URL(fileURLWithPath:CommandLine.arguments[0]).deletingLastPathComponent().appendingPathComponent("mlx.metallib")),
                            "build_identity_sha256":try digest(URL(fileURLWithPath:CommandLine.arguments[0]).deletingLastPathComponent().appendingPathComponent("build-identity.json")),
                            "source_archive_sha256":try digest(URL(fileURLWithPath:CommandLine.arguments[0]).deletingLastPathComponent().appendingPathComponent("build-source.tar.gz")),
                            "model_config_sha256":try digest(modelURL.appendingPathComponent("config.json")),
                            "model_index_sha256":try digest(modelURL.appendingPathComponent("model.safetensors.index.json"))],
                        "optimizations":try JSONSerialization.jsonObject(with:JSONEncoder().encode(engine.model.optimizations)),
                        "numerical_environment":["mlx_enable_tf32_raw":ProcessInfo.processInfo.environment["MLX_ENABLE_TF32"] as Any? ?? NSNull(),
                            "effective_tf32":ProcessInfo.processInfo.environment["MLX_ENABLE_TF32"] != "0"],
                        "plan":plan.json(),
                        "memory_ledger": plan.memoryLedger.json, "documents": documents,
                        "invalid_state_reuse_refused": reuseRefused,
                        "resident_split_evidence":["required":requireResidentSplit,
                            "split_layers":sink.splitLayerKeys],
                        "files": sink.files, "activation_bytes": sink.activationBytes, "logits_bytes": sink.logitsBytes]
                    if packedWidening {
                        payload["effective_expert_widening"] = engine.effectiveWideningPolicy.rawValue
                    }
                    if let oracle { payload["oracle"] = oracle.evidence(maskFailureRefused: maskReuseRefused) }
                    if let tokenizedMetadata { payload["tokenized_corpus"] = tokenizedMetadata }
                    guard !requireResidentSplit || !sink.splitLayerKeys.isEmpty else {
                        throw PlanError("resident-overlap control did not execute both resident and miss branches")
                    }
                    let report = try JSONSerialization.data(withJSONObject: payload, options: [.prettyPrinted, .sortedKeys])
                    guard report.count<=reservation else { throw PlanError("capture report exceeds live-buffer limit") }
                    try report.write(to: destination.appendingPathComponent("report.json"), options: .atomic)
                    let hash = SHA256.hash(data: report).map { String(format: "%02x", $0) }.joined()
                    try JSONSerialization.data(withJSONObject: ["format": oracle != nil ? "slotstream-flash-oracle-completion-v1" : packedWidening
                            ? "slotstream-flash-widening-completion-v1" : "slotstream-flash-capture-completion-v1",
                        "report_sha256": hash, "qualification": false], options: [.prettyPrinted, .sortedKeys])
                        .write(to: destination.appendingPathComponent("completion.json"), options: .atomic)
                } catch { result = .failure(error) }
                sem.signal()
            }
            sem.wait(); try result.get()
        } catch {
            try? JSONSerialization.data(withJSONObject: ["format": "slotstream-flash-capture-failure-v1",
                "error": String(describing: error)], options: [.prettyPrinted, .sortedKeys])
                .write(to: destination.appendingPathComponent("failure.json"), options: .atomic)
            throw error
        }
    }
}
