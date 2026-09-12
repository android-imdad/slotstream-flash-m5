import Foundation
import MLX
import Slotstream

public struct M5CaseReport: Codable, Sendable {
    public var id: String
    public var projection: String
    public var bits: Int
    public var sourceEncoding: String
    public var batch: Int
    public var expertCount: Int
    public var sortedIndices: Bool
    public var contiguousInput: Bool
    public var inputShape: [Int]
    public var inputStrides: [Int]
    public var weightShape: [Int]
    public var outputShape: [Int]
    public var inputDType: String
    public var metadataDType: String
    public var expectedDispatch: String
    public var observedDispatch: String?
    public var observationStatus: String
    public var observationReason: String
    public var maxAbsoluteError: Double
    public var maxRelativeError: Double
    public var scalarSpotError: Double
    public var scalarSpotCoverage: [String]
    public var tolerance: Double
    public var passed: Bool
}

public struct M5SyntheticReport: Codable, Sendable {
    public var format = "slotstream-m5-diagnostic-v1"
    public var schemaVersion = 1
    public var mode: String
    public var deviceArchitecture: String
    public var operatingSystem: String
    public var effectiveTF32: Bool
    public var mlxMetalNoNAX: String
    public var internalArchitectureGeneration: String
    public var observationStatus: String
    public var observationReason: String
    public var allocationLimitBytes: Int
    public var mlxPeakBytes: Int
    public var processFootprintEndBytes: UInt64
    public var cases: [M5CaseReport]
    public var check: CheckReport
}

extension Diagnostics {
    private struct M5Projection {
        let name: String
        let input: Int
        let output: Int
    }

    /// Mirrors the pinned MLX 0.31.1 GatherQMM shape predicate. Hardware and
    /// dtype eligibility are deliberately separate from this source-level path.
    public static func m5ExpectedDispatch(
        matrixRows: Int, batch: Int, experts: Int, sortedIndices: Bool,
        transpose: Bool, contiguousInput: Bool
    ) -> String {
        guard matrixRows > 0, batch > 0, experts > 0 else { return "invalid-shape" }
        if matrixRows == 1 && batch >= 16 && sortedIndices && batch / experts >= 4 {
            return transpose ? "grouped-rhs-hardware-eligible" : "grouped-rhs-non-nax"
        }
        return contiguousInput ? "non-grouped-fallback-device-dependent"
            : "non-grouped-fallback-after-contiguous-materialization"
    }

    public static func m5Eligibility() -> CheckReport {
        var c = CheckBuilder("m5-eligibility")
        c.equal("representative padded prefill reaches grouped RHS",
            m5ExpectedDispatch(matrixRows: 1, batch: 16, experts: 4,
                sortedIndices: true, transpose: true, contiguousInput: true),
            "grouped-rhs-hardware-eligible")
        c.equal("decode control remains vector fallback",
            m5ExpectedDispatch(matrixRows: 1, batch: 1, experts: 4,
                sortedIndices: false, transpose: true, contiguousInput: true),
            "non-grouped-fallback-device-dependent")
        c.equal("unsorted work cannot enter grouped RHS",
            m5ExpectedDispatch(matrixRows: 1, batch: 64, experts: 4,
                sortedIndices: false, transpose: true, contiguousInput: true),
            "non-grouped-fallback-device-dependent")
        c.equal("too few rows per expert cannot enter grouped RHS",
            m5ExpectedDispatch(matrixRows: 1, batch: 16, experts: 8,
                sortedIndices: true, transpose: true, contiguousInput: true),
            "non-grouped-fallback-device-dependent")
        c.equal("non-transposed grouped work is not NAX eligible",
            m5ExpectedDispatch(matrixRows: 1, batch: 16, experts: 4,
                sortedIndices: true, transpose: false, contiguousInput: true),
            "grouped-rhs-non-nax")
        c.equal("noncontiguous fallback is explicit",
            m5ExpectedDispatch(matrixRows: 1, batch: 4, experts: 4,
                sortedIndices: false, transpose: true, contiguousInput: false),
            "non-grouped-fallback-after-contiguous-materialization")
        c.equal("zero experts is rejected before division",
            m5ExpectedDispatch(matrixRows: 1, batch: 16, experts: 0,
                sortedIndices: true, transpose: true, contiguousInput: true),
            "invalid-shape")
        c.equal("zero batch is rejected",
            m5ExpectedDispatch(matrixRows: 1, batch: 0, experts: 4,
                sortedIndices: true, transpose: true, contiguousInput: true),
            "invalid-shape")
        return c.report()
    }

    private static func m5Weights(
        projection: M5Projection, experts: Int, bits: Int
    ) throws -> (MLXArray, MLXArray, MLXArray, String) {
        let rows = experts * projection.output
        let valueCount = rows * projection.input
        var values: [Float] = []
        values.reserveCapacity(valueCount)
        for index in 0 ..< valueCount {
            let code = (index &* 37 &+ index / 97) % 257
            values.append(Float(code - 128) / 256)
        }
        let source = MLXArray(values, [rows, projection.input])
        if bits == 4 {
            let q = quantized(source, groupSize: 64, bits: 4)
            eval(q.wq, q.scales, q.biases!)
            return (q.wq.reshaped([experts, projection.output, projection.input * 4 / 32]),
                    q.scales.asType(DType.float32).reshaped([experts, projection.output, projection.input / 64]),
                    q.biases!.asType(DType.float32).reshaped([experts, projection.output, projection.input / 64]),
                    "native-4-bit")
        }
        let q4 = quantized(source, groupSize: 64, bits: 4)
        let q6 = quantized(source, groupSize: 64, bits: 6)
        eval(q4.wq, q4.scales, q4.biases!, q6.wq, q6.scales, q6.biases!)
        let sourceBytes = q4.wq.reshaped([-1]).view(dtype: DType.uint8).asArray(UInt8.self)
        var widened = Data(count: rows * projection.input * 6 / 8)
        try sourceBytes.withUnsafeBytes { sourcePointer in
            try widened.withUnsafeMutableBytes { destinationPointer in
                try AffineCodes.widen(sourcePointer, to: destinationPointer,
                    count: rows * projection.input, from: 4, to: 6)
            }
        }
        let widenedArray = MLXArray(widened, [rows, projection.input * 6 / 32], dtype: .uint32)
        let half = rows / 2
        let weight = concatenated([widenedArray[0 ..< half], q6.wq[half ..< rows]], axis: 0)
        let scales = concatenated([q4.scales[0 ..< half], q6.scales[half ..< rows]], axis: 0).asType(.float32)
        let biases = concatenated([q4.biases![0 ..< half], q6.biases![half ..< rows]], axis: 0).asType(.float32)
        eval(weight, scales, biases)
        return (weight.reshaped([experts, projection.output, projection.input * 6 / 32]),
                scales.reshaped([experts, projection.output, projection.input / 64]),
                biases.reshaped([experts, projection.output, projection.input / 64]),
                "mixed-native-6-and-exact-widened-4")
    }

    private static func m5Case(
        projection: M5Projection, bits: Int, batch: Int, sorted: Bool,
        contiguousInput: Bool, weight: MLXArray, scales: MLXArray,
        biases: MLXArray, sourceEncoding: String, experts: Int = 4
    ) throws -> M5CaseReport {
        let inputCount = batch * projection.input
        var inputValues: [Float] = []
        inputValues.reserveCapacity(inputCount)
        for index in 0 ..< inputCount {
            let code = (index &* 17 &+ 11) % 127
            inputValues.append(Float(code - 63) / 128)
        }
        let denseInput = MLXArray(inputValues, [batch, 1, projection.input])
        let input: MLXArray
        if contiguousInput {
            input = denseInput
        } else {
            let doubled = concatenated([denseInput, denseInput], axis: -1)
            input = asStrided(doubled, [batch, 1, projection.input],
                strides: [projection.input * 2, projection.input * 2, 2])
        }
        let ids: [Int32]
        if sorted {
            ids = (0 ..< batch).map { Int32(min(experts - 1, $0 * experts / batch)) }
        } else {
            ids = (0 ..< batch).map { Int32(($0 &* 3 &+ 1) % experts) }
        }
        let indices = MLXArray(ids)
        let actual = gatherQuantizedMM(input, weight, scales: scales, biases: biases,
            rhsIndices: indices, transpose: true, groupSize: 64, bits: bits,
            sortedIndices: sorted)
        let decoded = dequantized(weight, scales: scales, biases: biases,
            groupSize: 64, bits: bits, dtype: .float32)
        let reference = gatherMM(input, decoded.transposed(axes: [0, 2, 1]),
            rhsIndices: indices, sortedIndices: sorted)
        eval(actual, reference)
        let delta = abs(actual.asType(.float32) - reference.asType(.float32))
        let maxAbsolute = delta.max().item(Float.self)
        let referenceScale = max(1, abs(reference).max().item(Float.self))
        let maxRelative = maxAbsolute / referenceScale

        var scalarError = Float(0)
        var scalarCoverage: [String] = []
        for expert in 0 ..< experts {
            guard let row = ids.firstIndex(of: Int32(expert)) else { continue }
            let cpuX = input[row][0].asType(.float32).asArray(Float.self)
            for outputColumn in [0, projection.output - 1] {
                let cpuW = decoded[expert][outputColumn].asType(.float32).asArray(Float.self)
                let scalar = zip(cpuX, cpuW).reduce(Float(0)) { $0 + $1.0 * $1.1 }
                scalarError = max(scalarError,
                    abs(actual[row][0][outputColumn].item(Float.self) - scalar))
                scalarCoverage.append("expert-\(expert)-output-\(outputColumn)")
            }
        }
        let tolerance = Float(0.05) + referenceScale * 0.01
        let expected = m5ExpectedDispatch(matrixRows: 1, batch: batch, experts: experts,
            sortedIndices: sorted, transpose: true, contiguousInput: contiguousInput)
        return M5CaseReport(
            id: "\(projection.name)-b\(bits)-n\(batch)-\(sorted ? "sorted" : "unsorted")-\(contiguousInput ? "contiguous" : "strided")",
            projection: projection.name, bits: bits, sourceEncoding: sourceEncoding,
            batch: batch, expertCount: experts, sortedIndices: sorted,
            contiguousInput: contiguousInput, inputShape: input.shape,
            inputStrides: input.asData(access: .noCopy).strides,
            weightShape: weight.shape, outputShape: actual.shape,
            inputDType: String(describing: input.dtype), metadataDType: String(describing: scales.dtype),
            expectedDispatch: expected, observedDispatch: nil, observationStatus: "unverified",
            observationReason: "MLX public API does not expose the selected Metal pipeline name",
            maxAbsoluteError: Double(maxAbsolute), maxRelativeError: Double(maxRelative),
            scalarSpotError: Double(scalarError), scalarSpotCoverage: scalarCoverage,
            tolerance: Double(tolerance),
            passed: maxAbsolute <= tolerance && scalarError <= tolerance)
    }

    private static func m5NegativeControls(
        projection: M5Projection, bits: Int, weight: MLXArray,
        scales: MLXArray, biases: MLXArray
    ) -> (padCropError: Float, wrongIndexDelta: Float) {
        let rows = 4
        var values: [Float] = []
        values.reserveCapacity(rows * projection.input)
        for index in 0 ..< rows * projection.input {
            values.append(Float((index * 13 + 5) % 97 - 48) / 96)
        }
        let input = MLXArray(values, [rows, 1, projection.input])
        let indices = MLXArray([Int32(0), 1, 2, 3])
        let unpadded = gatherQuantizedMM(input, weight, scales: scales, biases: biases,
            rhsIndices: indices, transpose: true, groupSize: 64, bits: bits,
            sortedIndices: true)
        let paddedInput = concatenated([input,
            broadcast(input[3 ..< 4], to: [12, 1, projection.input])], axis: 0)
        let paddedIndices = MLXArray([Int32(0), 1, 2] + [Int32](repeating: 3, count: 13))
        let padded = gatherQuantizedMM(paddedInput, weight, scales: scales, biases: biases,
            rhsIndices: paddedIndices, transpose: true, groupSize: 64, bits: bits,
            sortedIndices: true)[0 ..< rows]
        let wrong = gatherQuantizedMM(input, weight, scales: scales, biases: biases,
            rhsIndices: MLXArray([Int32(1), 2, 3, 0]), transpose: true,
            groupSize: 64, bits: bits, sortedIndices: false)
        eval(unpadded, padded, wrong)
        let unpadded32 = unpadded.asType(DType.float32)
        let padded32 = padded.asType(DType.float32)
        let wrong32 = wrong.asType(DType.float32)
        return (abs(unpadded32 - padded32).max().item(Float.self),
                abs(unpadded32 - wrong32).max().item(Float.self))
    }

    public static func m5Synthetic(
        full: Bool = true, mode: String = "synthetic"
    ) throws -> M5SyntheticReport {
        MLX.Memory.cacheLimit = 32 << 20
        MLX.Memory.clearCache()
        GPU.resetPeakMemory()
        let projections = [M5Projection(name: "gate", input: 2560, output: 640),
                           M5Projection(name: "up", input: 2560, output: 640),
                           M5Projection(name: "down", input: 640, output: 2560)]
        let batches = full ? [1, 4, 16, 64, 256] : [1, 16]
        var cases: [M5CaseReport] = []
        var padCropError = Float.infinity
        var wrongIndexDelta = Float(0)
        for projection in projections {
            for bits in [4, 6] {
                let (weight, scales, biases, sourceEncoding) = try m5Weights(
                    projection: projection, experts: 4, bits: bits)
                if projection.name == "gate", bits == 6 {
                    let controls = m5NegativeControls(projection: projection, bits: bits,
                        weight: weight, scales: scales, biases: biases)
                    padCropError = controls.padCropError
                    wrongIndexDelta = controls.wrongIndexDelta
                }
                for batch in batches {
                    cases.append(try m5Case(projection: projection, bits: bits,
                        batch: batch, sorted: true, contiguousInput: true,
                        weight: weight, scales: scales, biases: biases,
                        sourceEncoding: sourceEncoding))
                    cases.append(try m5Case(projection: projection, bits: bits,
                        batch: batch, sorted: false, contiguousInput: batch != 1,
                        weight: weight, scales: scales, biases: biases,
                        sourceEncoding: sourceEncoding))
                    MLX.Memory.clearCache()
                }
            }
        }
        let peak = MLX.Memory.peakMemory
        var c = CheckBuilder("m5-dispatch")
        c.expect("all three projections executed", Set(cases.map(\.projection)) == Set(["gate", "up", "down"]))
        c.expect("four-bit and mixed widened six-bit executed", Set(cases.map(\.bits)) == Set([4, 6]))
        c.expect("every numerical comparison meets its recorded tolerance", cases.allSatisfy(\.passed))
        c.expect("grouped and vector fallback cases are both present",
            cases.contains { $0.expectedDispatch == "grouped-rhs-hardware-eligible" }
                && cases.contains { $0.expectedDispatch.hasPrefix("non-grouped-fallback") })
        c.expect("noncontiguous fallback case is present",
            cases.contains { $0.expectedDispatch == "non-grouped-fallback-after-contiguous-materialization" })
        c.expect("mixed six-bit scalar spots cover native-six experts and output boundaries",
            cases.filter { $0.bits == 6 }.contains {
                $0.scalarSpotCoverage.contains("expert-2-output-0")
                    && $0.scalarSpotCoverage.contains("expert-3-output-\($0.projection == "down" ? 2559 : 639)")
            })
        c.expect("padding then cropping preserves the four real rows", padCropError <= 0.1,
            "max error \(padCropError)")
        c.expect("deliberately wrong valid expert indices change the output", wrongIndexDelta > 0.001,
            "max delta \(wrongIndexDelta)")
        c.expect("JANG metadata is FP32", cases.allSatisfy { $0.metadataDType == "float32" })
        c.expect("MLX live allocation remains at or below 256 MiB", peak <= 256 << 20, "peak \(peak)")
        c.measure("mlx_peak_bytes", Double(peak))
        c.measure("case_count", Double(cases.count))
        c.measure("pad_crop_max_error", Double(padCropError))
        c.measure("wrong_index_max_delta", Double(wrongIndexDelta))
        let info = GPU.deviceInfo()
        let tf32 = ProcessInfo.processInfo.environment["MLX_ENABLE_TF32"].map { $0 != "0" } ?? true
        return M5SyntheticReport(mode: mode, deviceArchitecture: info.architecture,
            operatingSystem: ProcessInfo.processInfo.operatingSystemVersionString,
            effectiveTF32: tf32,
            mlxMetalNoNAX: "compile-time state not exposed by the public MLX API",
            internalArchitectureGeneration: "not exposed by the public MLX API",
            observationStatus: "unverified",
            observationReason: "correctness and source eligibility were measured; no Metal encoder record was observed",
            allocationLimitBytes: 256 << 20, mlxPeakBytes: peak,
            processFootprintEndBytes: ProcessMemory.residentBytes(), cases: cases, check: c.report())
    }

    public static func m5Dispatch() throws -> CheckReport {
        try m5Synthetic(full: false).check
    }

    public static func m5TraceCase(_ kind: String) throws -> M5SyntheticReport {
        guard kind == "grouped6" || kind == "decode6" else {
            throw ModelError("--trace-case must be grouped6 or decode6")
        }
        MLX.Memory.cacheLimit = 32 << 20
        MLX.Memory.clearCache()
        GPU.resetPeakMemory()
        let projection = M5Projection(name: "gate", input: 2560, output: 640)
        let (weight, scales, biases, sourceEncoding) = try m5Weights(
            projection: projection, experts: 4, bits: 6)
        let grouped = kind == "grouped6"
        let report = try m5Case(projection: projection, bits: 6,
            batch: grouped ? 16 : 1, sorted: grouped, contiguousInput: true,
            weight: weight, scales: scales, biases: biases, sourceEncoding: sourceEncoding)
        let peak = MLX.Memory.peakMemory
        var c = CheckBuilder("m5-dispatch")
        c.expect("selected trace case numerical comparison passed", report.passed)
        c.equal("selected trace case has requested source path", report.expectedDispatch,
            grouped ? "grouped-rhs-hardware-eligible" : "non-grouped-fallback-device-dependent")
        c.expect("selected trace case stays within 256 MiB", peak <= 256 << 20, "peak \(peak)")
        c.measure("mlx_peak_bytes", Double(peak))
        let info = GPU.deviceInfo()
        let tf32 = ProcessInfo.processInfo.environment["MLX_ENABLE_TF32"].map { $0 != "0" } ?? true
        return M5SyntheticReport(mode: "trace-\(kind)", deviceArchitecture: info.architecture,
            operatingSystem: ProcessInfo.processInfo.operatingSystemVersionString,
            effectiveTF32: tf32,
            mlxMetalNoNAX: "compile-time state not exposed by the public MLX API",
            internalArchitectureGeneration: "not exposed by the public MLX API",
            observationStatus: "unverified",
            observationReason: "selected case executed; external trace must establish the pipeline name",
            allocationLimitBytes: 256 << 20, mlxPeakBytes: peak,
            processFootprintEndBytes: ProcessMemory.residentBytes(), cases: [report], check: c.report())
    }
}
