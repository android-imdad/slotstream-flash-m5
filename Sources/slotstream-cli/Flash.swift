import ArgumentParser
import CryptoKit
import Foundation
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
