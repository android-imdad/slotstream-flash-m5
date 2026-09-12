import CryptoKit
import Darwin
import Foundation
import MLX

public protocol FlashObservationSink: AnyObject {
    func validateForward(tokens: Int) throws
    func observeInput(layer: Int, value: MLXArray) throws
    func observeRoute(layer: Int, expertIDs: [Int32]) throws
    func observeHidden(layer: Int, routerRanks: [Int], expertIDs: [Int32], value: MLXArray) throws
}

public enum FlashRankMapping {
    public static func inverse(parts: [[Int]], topK: Int) throws -> [Int32] {
        let order = parts.flatMap { $0 }
        guard topK > 0, order.count == topK, Set(order) == Set(0 ..< topK) else {
            throw PlanError("flash observation ranks must cover each router rank exactly once")
        }
        var inverse = [Int32](repeating: 0, count: topK)
        for (position, rank) in order.enumerated() {
            inverse[rank] = Int32(position)
        }
        return inverse
    }
}

private final class FlashCancellationToken {
    private let lock = NSLock()
    private var value = false
    func cancel() {
        lock.lock(); value = true; lock.unlock()
    }

    func isCancelled() -> Bool {
        lock.lock(); defer { lock.unlock() }; return value
    }
}

public final class FlashBoundedWriter {
    public typealias WriteCall = (Int32, UnsafeRawPointer, Int) -> Int
    private let maxBytes: Int64
    private let liveLimit: Int
    private let writeCall: WriteCall
    private let cancellation = FlashCancellationToken()
    private var written: Int64 = 0
    public init(maxBytes: Int64, liveLimit: Int = 64 << 20, writeCall: WriteCall? = nil) {
        self.maxBytes = maxBytes; self.liveLimit = liveLimit
        self.writeCall = writeCall ?? { Darwin.write($0, $1, $2) }
    }

    public func cancel() {
        cancellation.cancel()
    }

    public func write(_ data: Data, to path: String) throws {
        guard !cancellation.isCancelled() else { throw PlanError("capture cancelled") }
        guard maxBytes >= 0, liveLimit > 0, data.count <= liveLimit,
              Int64(data.count) <= maxBytes, written <= maxBytes - Int64(data.count)
        else {
            throw PlanError("capture write exceeds live or aggregate quota")
        }
        let fd = Darwin.open(path, O_WRONLY | O_CREAT | O_EXCL, 0o600)
        guard fd >= 0 else { throw PlanError("cannot create capture file") }
        defer { Darwin.close(fd) }
        try data.withUnsafeBytes { bytes in
            var sent = 0
            while sent < bytes.count {
                guard !cancellation.isCancelled() else { throw PlanError("capture cancelled") }
                let n = writeCall(fd, bytes.baseAddress! + sent, bytes.count - sent)
                if n > 0 { sent += n; continue }
                if n < 0, errno == EINTR { continue }
                throw PlanError("capture write failed or made no progress")
            }
        }
        var info = stat(); guard fstat(fd, &info) == 0, info.st_size == data.count else {
            throw PlanError("capture write completed with the wrong file size")
        }
        written += Int64(data.count)
    }
}

public struct FlashStateField: Codable, Equatable {
    public var name: String
    public var present: Bool
    public var dtype: String?
    public var shape: [Int]?
    public var bytes: Int?
    public var sha256: String?
    public init(name: String, present: Bool, dtype: String? = nil, shape: [Int]? = nil,
                bytes: Int? = nil, sha256: String? = nil)
    {
        self.name = name; self.present = present; self.dtype = dtype; self.shape = shape
        self.bytes = bytes; self.sha256 = sha256
    }
}

public struct FlashStateIdentity: Codable, Equatable {
    public var fields: [FlashStateField]
    public var indexerBases: [String: Int]
    public var draftIndexerBase: Int?
    public var allocatedSequenceBytes: Int
    public init(fields: [FlashStateField], indexerBases: [String: Int], draftIndexerBase: Int?, allocatedSequenceBytes: Int) {
        self.fields = fields; self.indexerBases = indexerBases; self.draftIndexerBase = draftIndexerBase
        self.allocatedSequenceBytes = allocatedSequenceBytes
    }
}

public extension Qwen4ExpModel.State {
    func flashStateIdentity(maxCopyBytes: Int = 64 << 20) throws -> FlashStateIdentity {
        let tensors = diagnosticTensors()
        var expected = Set(["ngram", "tokens", "lastMulti", "mtp.key", "mtp.value", "mtp.index", "mtp.offset"])
        for layer in linear.keys {
            expected.formUnion(["conv.\(layer)", "ssm.\(layer)", "ple.\(layer)"])
        }
        for layer in kv.keys {
            expected.formUnion(["key.\(layer)", "value.\(layer)"])
        }
        for layer in indexer.keys {
            expected.insert("index.\(layer)")
        }
        let fields = try expected.sorted().map { name -> FlashStateField in
            guard let value = tensors[name] else {
                return FlashStateField(name: name, present: false)
            }
            guard value.nbytes <= maxCopyBytes else {
                throw PlanError("defined state field \(name) exceeds bounded copy scratch")
            }
            eval(value)
            let copy = value.asData(access: .copy)
            return FlashStateField(name: name, present: true,
                                   dtype: String(describing: copy.dType), shape: copy.shape,
                                   bytes: copy.data.count,
                                   sha256: SHA256.hash(data: copy.data).map { String(format: "%02x", $0) }.joined())
        }
        return FlashStateIdentity(fields: fields, indexerBases: diagnosticIndexerBases(),
                                  draftIndexerBase: diagnosticDraftIndexerBase,
                                  allocatedSequenceBytes: allocatedSequenceBytes)
    }
}
