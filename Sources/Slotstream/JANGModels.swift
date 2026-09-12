import Foundation

/// Two explicitly pinned alternatives. The original default and Slotpack
/// downloader are unchanged; JANG uses the existing hash-checked raw reader.
public enum JANGModels {
    public static var all: [JANGModel] { [jang4M, jang6S] }
    public static func named(_ value: String) -> JANGModel? {
        all.first { model in
            [model.format.modelName, model.directoryName, model.repository,
             model.format == .jang4M ? "JANG_4M" : "JANG_6S"].contains(value)
        }
    }
}

public struct JANGModel: Sendable {
    public let format: CheckpointFormat
    public let directoryName: String
    public let repository: String
    public let revision: String
    public let files: [PinnedModel.File]
    public var totalBytes: Int64 { files.reduce(0) { $0 + $1.size } }

    public func verify(at directory: URL, log: WeightStore.Log = { _ in }) throws {
        for file in files {
            guard WeightStore.fileMatches(directory.appendingPathComponent(file.path), size: file.size, sha256: file.sha256) else {
                throw SlotstreamError.pull("JANG file missing or corrupt: \(file.path); run slotstream pull \(format.modelName)")
            }
        }
        log("VERIFY PASS: \(format.modelName) matches \(revision)")
    }

    public func download(to directory: URL, connections: Int? = nil,
                         cancellation: PullCancellation = .init(), sources: [String]? = nil,
                         log: @escaping WeightStore.Log = { _ in }) throws {
        let bases = sources ?? ["https://huggingface.co/\(repository)/resolve/\(revision)"]
        guard !bases.isEmpty else { throw SlotstreamError.pull("no JANG download sources configured") }
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        let lease = try DownloadDirectoryLock(directory)
        defer { withExtendedLifetime(lease) {} }
        let job = PullJob(dest: directory,
            bases: bases,
            connections: min(32, max(1, connections ?? PullTuning.connections)),
            files: files, cancellation: cancellation, log: log)
        defer { job.shutdown() }
        let remaining = try job.plan()
        let free = (try FileManager.default.attributesOfFileSystem(forPath: directory.path))[.systemFreeSize] as? Int64 ?? 0
        guard free >= remaining + 2_000_000_000 else { throw SlotstreamError.pull("not enough disk for the pinned JANG download and margin") }
        log("pulling \(repository) @ \(revision): \(remaining) bytes remaining")
        try job.run()
    }
}
