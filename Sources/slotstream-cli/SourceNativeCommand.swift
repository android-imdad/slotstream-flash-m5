import ArgumentParser
import CryptoKit
import Foundation
import Slotstream
import SlotstreamDiagnostics

struct SourceNativeCheck: ParsableCommand {
    static let configuration = CommandConfiguration(commandName: "source-native-check",
        abstract: "Compare source-native whole records with scattered original JANG reads")
    @Option var model: String
    @Option(help: "Fresh directory under .build/flash with an existing parent") var output: String

    func run() throws {
        let fm = FileManager.default
        let root = URL(fileURLWithPath: fm.currentDirectoryPath).appendingPathComponent(".build/flash")
            .resolvingSymlinksInPath()
        let destination = URL(fileURLWithPath: output).standardizedFileURL
        guard destination.deletingLastPathComponent().resolvingSymlinksInPath().path.hasPrefix(root.path + "/"),
              fm.fileExists(atPath: destination.deletingLastPathComponent().path),
              !fm.fileExists(atPath: destination.path) else {
            throw ValidationError("output needs a fresh path with an existing parent under .build/flash")
        }
        try fm.createDirectory(at: destination, withIntermediateDirectories: false)
        do {
            let report = try Diagnostics.sourceNativeComponent(modelDir: ModelLocator.resolve(model), artifact: destination.appendingPathComponent("sample-layout"))
            let encoder = JSONEncoder()
            encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
            let bytes = try encoder.encode(report)
            try bytes.write(to: destination.appendingPathComponent("report.json"), options: .atomic)
            guard report.check.passed else { throw ValidationError("source-native checks failed") }
            let hash = SHA256.hash(data: bytes).map { String(format: "%02x", $0) }.joined()
            try JSONSerialization.data(withJSONObject: ["complete": true, "report_sha256": hash], options: [.sortedKeys])
                .write(to: destination.appendingPathComponent("completion.json"), options: .atomic)
            print(destination.path)
        } catch {
            try? JSONSerialization.data(withJSONObject: ["error": String(describing: error)])
                .write(to: destination.appendingPathComponent("failure.json"), options: .atomic)
            throw error
        }
    }
}
