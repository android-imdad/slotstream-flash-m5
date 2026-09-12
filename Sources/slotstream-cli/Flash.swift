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
        abstract: "Capture a bounded exact diagnostic reference; never enables approximate inference")
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
    @Flag(name: .customLong("require-resident-split")) var requireResidentSplit = false
    private func digest(_ url:URL) throws -> String {
        SHA256.hash(data:try Data(contentsOf:url,options:.mappedIfSafe)).map{String(format:"%02x",$0)}.joined()
    }

    func run() throws {
        guard mode == "reference-off" || mode == "reference-on" else {
            throw ValidationError("--mode must be reference-off or reference-on; approximate modes are refused")
        }
        guard split == "development", maxContext == 2048, memoryGB == 14, (32...64).contains(liveLimitMB),
              (1...32).contains(activationQuotaGB), (1...8).contains(logitsQuotaGB),
              positionLimit == 64, !requireResidentSplit || mode == "reference-on" else {
            throw ValidationError("capture requires the development split, --memory-gb 14, --max-context 2048, position limit 64 and live limit 32...64 MiB")
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
                  Set(raw.keys)==Set(["format","schemaVersion","manifestSHA256","developmentOnly","tokenSource","documents"]) else {
                throw ValidationError("capture manifest changed while read or has unknown fields")
            }
            let request = try JSONDecoder().decode(CaptureManifest.self, from: manifestData)
            var canonical = raw
            canonical.removeValue(forKey: "manifestSHA256")
            let canonicalData = try JSONSerialization.data(withJSONObject: canonical, options: [.sortedKeys])
            let canonicalHash = SHA256.hash(data: canonicalData).map { String(format: "%02x", $0) }.joined()
            guard request.format == "slotstream-flash-capture-v1", request.schemaVersion == 1,
                  request.manifestSHA256 == canonicalHash,
                  request.developmentOnly, request.tokenSource == "arbitrary-valid-diagnostic-ids-not-training-text",
                  request.documents.filter({ $0.split == split }).count > 0 else {
                throw ValidationError("capture manifest identity or split is invalid")
            }
            var documentIDs=Set<String>(), totalPositions=0
            guard let rawDocuments=raw["documents"] as? [[String:Any]], rawDocuments.count==request.documents.count,
                  rawDocuments.allSatisfy({ Set($0.keys)==Set(["id","split","warmupIDs","positions"])
                    && (($0["positions"] as? [[String:Any]])?.allSatisfy({Set($0.keys)==Set(["position","inputID","nextTokenID"])}) ?? false) }) else {
                throw ValidationError("capture manifest has unknown nested fields")
            }
            for document in request.documents {
                let safeID = !document.id.isEmpty && document.id.utf8.allSatisfy {
                    (48...57).contains($0) || (65...90).contains($0) ||
                        (97...122).contains($0) || $0 == 45 || $0 == 95
                }
                guard documentIDs.insert(document.id).inserted,
                      document.split == "development",
                      document.id.utf8.count <= 64,
                      safeID,
                      document.warmupIDs.count <= 256,
                      document.warmupIDs.count + document.positions.count <= maxContext,
                      document.warmupIDs.allSatisfy({(0..<248320).contains($0)}) else {
                    throw ValidationError("capture document ID, count or context is invalid")
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
            let reservation = liveLimitMB << 20
            let policy = try RuntimeAllocationPolicy(prefixCacheEnabled: false,
                diagnosticReservedBytes: reservation * 2)
            let plan = try CheckpointMemory(index: index).plan(memoryGB: memoryGB,
                maxContext: maxContext, policy: policy)
            let sem = DispatchSemaphore(value: 0)
            var result: Result<Void, Error> = .success(())
            Task {
                do {
                    let engine = try await Engine(modelDir: modelURL, plan: plan)
                    guard engine.model.optimizations.overlapResidentExperts == requireResidentSplit else {
                        throw PlanError("effective resident-overlap control does not match the capture request")
                    }
                    let sink = NativeCaptureSink(root: destination, liveLimit: reservation,
                        activationQuota: Int64(activationQuotaGB) * 1_000_000_000,
                        logitsQuota: Int64(logitsQuotaGB) * 1_000_000_000)
                    let previousSIGINT = Darwin.signal(SIGINT, SIG_IGN)
                    let cancellation=DispatchSource.makeSignalSource(signal:SIGINT,queue:.global())
                    cancellation.setEventHandler { sink.cancel() }; cancellation.resume()
                    defer { cancellation.cancel(); Darwin.signal(SIGINT, previousSIGINT) }
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
                    var documents: [[String: Any]] = []
                    for document in request.documents where document.split == split {
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
                            engine.model.flashObservationSink = mode == "reference-on" ? sink : nil
                            sink.begin(documentID:document.id,tokenPosition:position.position,inputID:position.inputID)
                            let logits = try engine.model.lastLogitsChecked([position.inputID], state: state)
                            eval(logits)
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
                        }
                        documents.append(["id": document.id, "positions": positions])
                    }
                    let payload: [String: Any] = ["format": "slotstream-flash-capture-output-v1",
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
                    guard !requireResidentSplit || !sink.splitLayerKeys.isEmpty else {
                        throw PlanError("resident-overlap control did not execute both resident and miss branches")
                    }
                    let report = try JSONSerialization.data(withJSONObject: payload, options: [.prettyPrinted, .sortedKeys])
                    guard report.count<=reservation else { throw PlanError("capture report exceeds live-buffer limit") }
                    try report.write(to: destination.appendingPathComponent("report.json"), options: .atomic)
                    let hash = SHA256.hash(data: report).map { String(format: "%02x", $0) }.joined()
                    try JSONSerialization.data(withJSONObject: ["format": "slotstream-flash-capture-completion-v1",
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
