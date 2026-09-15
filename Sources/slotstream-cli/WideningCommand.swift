import ArgumentParser
import CryptoKit
import Foundation
import Slotstream
import SlotstreamDiagnostics

private struct WideningCommandProvenance: Codable {
    var executablePath: String
    var executableSHA256: String
    var metallibSHA256: String
    var buildIdentitySHA256: String
    var sourceArchiveSHA256: String
    var compilerVersion: String
}

private struct WideningCommandReport: Codable {
    var format = "slotstream-widening-command-v1"
    var schemaVersion = 1
    var processID: Int32
    var mode: String
    var provenance: WideningCommandProvenance
    var synthetic: WideningSyntheticReport?
    var model: WideningModelReport?
}

struct WideningCheck: ParsableCommand {
    static let configuration = CommandConfiguration(
        commandName: "widening-check",
        abstract: "Verify and measure opt-in exact 4-to-6-bit expert widening")

    @Flag(help: "Run exhaustive exactness checks and bounded host benchmarks")
    var synthetic = false

    @Option(help: "Run bounded original-reader checks against this JANG_6S checkpoint")
    var model: String?

    @Option(help: "Fresh evidence directory under .build/flash")
    var output: String

    @Option(help: "Balanced benchmark pairs in synthetic mode (default 8)")
    var pairs = 8

    private func digest(_ url: URL) throws -> String {
        let data = try Data(contentsOf: url, options: .mappedIfSafe)
        return SHA256.hash(data: data).map { String(format: "%02x", $0) }.joined()
    }

    private func freshOutput() throws -> URL {
        guard synthetic != (model != nil) else {
            throw ValidationError("choose exactly one of --synthetic or --model")
        }
        guard synthetic || pairs == 8 else {
            throw ValidationError("--pairs is available only with --synthetic")
        }
        guard pairs > 0, pairs <= 100 else {
            throw ValidationError("--pairs must be between 1 and 100")
        }
        let manager = FileManager.default
        let root = URL(fileURLWithPath: manager.currentDirectoryPath).standardizedFileURL
        let flash = root.appendingPathComponent(".build/flash").resolvingSymlinksInPath()
        let requested = URL(fileURLWithPath: output, relativeTo: root).standardizedFileURL
        let parent = requested.deletingLastPathComponent()
        guard manager.fileExists(atPath: parent.path),
              parent.resolvingSymlinksInPath().path.hasPrefix(flash.path + "/"),
              requested.path.hasPrefix(root.appendingPathComponent(".build/flash").path + "/") else {
            throw ValidationError("--output must have an existing safe parent under .build/flash")
        }
        guard !manager.fileExists(atPath: requested.path) else {
            throw ValidationError("--output already exists: \(requested.path)")
        }
        try manager.createDirectory(at: requested, withIntermediateDirectories: false)
        return requested
    }

    private func compilerVersion() throws -> String {
        let process = Process()
        let pipe = Pipe()
        process.executableURL = URL(fileURLWithPath: "/usr/bin/xcrun")
        process.arguments = ["swiftc", "--version"]
        process.standardOutput = pipe
        process.standardError = pipe
        try process.run()
        process.waitUntilExit()
        guard process.terminationStatus == 0 else {
            throw ValidationError("could not record the Swift compiler version")
        }
        return String(decoding: pipe.fileHandleForReading.readDataToEndOfFile(), as: UTF8.self)
            .trimmingCharacters(in: .whitespacesAndNewlines)
    }

    private func provenance() throws -> WideningCommandProvenance {
        let executable = URL(fileURLWithPath: CommandLine.arguments[0]).standardizedFileURL
            .resolvingSymlinksInPath()
        let directory = executable.deletingLastPathComponent()
        let metallib = directory.appendingPathComponent("mlx.metallib")
        let identity = directory.appendingPathComponent("build-identity.json")
        let source = directory.appendingPathComponent("build-source.tar.gz")
        for artifact in [executable, metallib, identity, source] {
            guard FileManager.default.fileExists(atPath: artifact.path) else {
                throw ValidationError(
                    "missing build artifact beside diagnostic executable: \(artifact.lastPathComponent)")
            }
        }
        return try WideningCommandProvenance(executablePath: executable.path,
            executableSHA256: digest(executable), metallibSHA256: digest(metallib),
            buildIdentitySHA256: digest(identity), sourceArchiveSHA256: digest(source),
            compilerVersion: compilerVersion())
    }

    func run() throws {
        let destination = try freshOutput()
        do {
            let syntheticReport = synthetic ? try Diagnostics.wideningSynthetic(pairCount: pairs) : nil
            let modelReport = try model.map {
                try Diagnostics.wideningModelComponent(
                    modelDir: ModelLocator.resolve($0).resolvingSymlinksInPath())
            }
            guard syntheticReport?.exactCheck.passed != false,
                  modelReport?.check.passed != false else {
                throw ValidationError("widening component checks failed")
            }
            let report = try WideningCommandReport(processID: getpid(),
                mode: synthetic ? "synthetic" : "bounded-model-reader",
                provenance: provenance(), synthetic: syntheticReport, model: modelReport)
            let encoder = JSONEncoder()
            encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
            let reportURL = destination.appendingPathComponent("report.json")
            try encoder.encode(report).write(to: reportURL, options: .atomic)
            let completion: [String: Any] = [
                "format": "slotstream-widening-completion-v1",
                "schema_version": 1,
                "report_sha256": try digest(reportURL),
                "passed": true,
            ]
            try JSONSerialization.data(withJSONObject: completion, options: [.prettyPrinted, .sortedKeys])
                .write(to: destination.appendingPathComponent("completion.json"), options: .atomic)
            print(reportURL.path)
        } catch {
            let failure = ["format": "slotstream-widening-failure-v1",
                           "error": String(describing: error)]
            try? JSONSerialization.data(withJSONObject: failure, options: [.prettyPrinted, .sortedKeys])
                .write(to: destination.appendingPathComponent("failure.json"), options: .atomic)
            throw error
        }
    }
}
