//
//  Test.swift
//  UForm
//
//  The boundary suite: what the binding itself decides - name tables, handle lifetimes, tape lowering, which
//  surfaces a call keeps, the rows it wrote, the progress trampoline, the arena views, and how blocking and
//  deferred calls complete. The checkpoint is the one `UFORM_TEST_CHECKPOINT` names, or else stand-in bytes only
//  the stub loads, so widths come from the model's description and only text, a byte per token, pins counts.
//
//  Suites run in parallel and every test gets a fresh instance, so each test spawns the executor it drives.
//

import Foundation
import Testing
import UForm

/// The error a call failed with, and nil when it succeeded.
private func failure<Value>(of body: @autoclosure () throws(UFormError) -> Value) -> UFormError? {
    do {
        _ = try body()
        return nil
    }
    catch {
        return error
    }
}

/// The status a call failed with, and nil when it succeeded.
private func failureStatus<Value>(of body: @autoclosure () throws(UFormError) -> Value) -> UForm.Status? {
    failure(of: try body())?.status
}

/// The checkpoint `UFORM_TEST_CHECKPOINT` names, or a file of stand-in bytes only the stub loads.
private func checkpoint() throws -> String {
    if let named = ProcessInfo.processInfo.environment["UFORM_TEST_CHECKPOINT"] { return named }
    let path = FileManager.default.temporaryDirectory.appendingPathComponent("uform-swift-\(UUID().uuidString)").path
    try Data(repeating: 7, count: 100).write(to: URL(fileURLWithPath: path))
    return path
}

/// A model over the checkpoint.
private func model() throws -> Model { try Model.loadRam(path: try checkpoint()) }

/// CRC-32 of `bytes`, as PNG chunks carry it.
private func crc32(_ bytes: [UInt8]) -> UInt32 {
    ~bytes.reduce(~UInt32(0)) { crc, byte in
        (0 ..< 8).reduce(crc ^ UInt32(byte)) { crc, _ in (crc >> 1) ^ (0xEDB8_8320 & (0 &- (crc & 1))) }
    }
}

/// The big-endian bytes of `value`.
private func bigEndian(_ value: UInt32) -> [UInt8] {
    (0 ..< 4).map { UInt8(truncatingIfNeeded: value >> (24 - 8 * $0)) }
}

/// A valid 8 x 8 RGB PNG of a gradient, its pixels in one stored deflate block.
private func png() -> [UInt8] {
    let raw: [UInt8] = (0 ..< 8).flatMap { row in
        [0] + (0 ..< 24).map { UInt8(truncatingIfNeeded: row * 32 + $0 * 8) }
    }
    var low: UInt32 = 1
    var high: UInt32 = 0
    for byte in raw {
        low = (low + UInt32(byte)) % 65521
        high = (high + low) % 65521
    }
    let length = UInt16(raw.count)
    let lengths = [UInt8(length & 0xFF), UInt8(length >> 8), UInt8(~length & 0xFF), UInt8(~length >> 8)]
    let stored = [0x78, 0x01, 0x01] + lengths
    let chunks: [(String, [UInt8])] = [
        ("IHDR", [0, 0, 0, 8, 0, 0, 0, 8, 8, 2, 0, 0, 0]),
        ("IDAT", stored + raw + bigEndian(high << 16 | low)),
        ("IEND", []),
    ]
    return chunks.reduce([0x89, 0x50, 0x4E, 0x47, 0x0D, 0x0A, 0x1A, 0x0A]) { png, chunk in
        let body = Array(chunk.0.utf8) + chunk.1
        return png + bigEndian(UInt32(chunk.1.count)) + body + bigEndian(crc32(body))
    }
}

/// A valid 16 kHz mono 16-bit WAV of a sawtooth.
private func wav(samples: Int = 1600) -> [UInt8] {
    func littleEndian<Value: FixedWidthInteger>(_ value: Value) -> [UInt8] {
        (0 ..< Value.bitWidth / 8).map { UInt8(truncatingIfNeeded: value >> (8 * $0)) }
    }
    let riff = Array("RIFF".utf8) + littleEndian(UInt32(36 + samples * 2)) + Array("WAVEfmt ".utf8)
    let format = littleEndian(UInt32(16)) + littleEndian(UInt16(1)) + littleEndian(UInt16(1))
    let rates = littleEndian(UInt32(16000)) + littleEndian(UInt32(32000))
    let framing = littleEndian(UInt16(2)) + littleEndian(UInt16(16))
    let data = Array("data".utf8) + littleEndian(UInt32(samples * 2))
    let sawtooth = (0 ..< samples).flatMap { littleEndian(Int16($0 % 64 * 512 - 16384)) }
    return riff + format + rates + framing + data + sawtooth
}

/// Every row of a run, widened to `Float`.
private func values(_ rows: Rows?) -> [[Float]] { rows.map { $0.map { Array($0) } } ?? [] }

/// Counts reports, which never arrive on two threads at once.
private final class Reports: @unchecked Sendable {
    var count = 0
    var completed = 0
}

@Suite("Vocabulary")
struct VocabularyTests {

    /// Every closed set spells itself through the library's own tables, and reads only those spellings back.
    @Test func namesAreTheLibrarys() {
        #expect(UForm.Status.invalidArgument.description == "invalid-argument")
        #expect(UForm.Status(name: "contended") == .contended)
        #expect(UForm.Status(name: "not-a-status") == nil)
        #expect(Scalar.bf16.description == "bf16")
        #expect(Scalar(name: "u1x8") == .u1x8)
        #expect(Scalar(name: "f128") == nil)
        #expect(Mood.spinning.description == "spinning")
        #expect(Mood(name: "asleep") == nil)
        #expect(Modality.audio.description == "audio")
        #expect(Modality(name: "video") == .video)
        #expect(Modality(name: "smell") == nil)
        #expect(Objective.joint.description == "joint")
        #expect(Objective(name: "masked") == .masked)
        #expect(Objective(name: "greedy") == nil)
        #expect(Negatives.named.description == "named")
        #expect(Negatives(name: "ambient") == .ambient)
        #expect(MultimemTier.peerRead.description == "peer-read")
        #expect(MultimemTier(name: "switch-reduced") == .switchReduced)
        #expect(MultimemTier(name: "ring") == nil)
        #expect(CPUClass.intelHaswell.description == "intel-haswell")
        #expect(CPUClass(name: "wasm64-extended") == .wasm64Extended)
        #expect(CPUClass(name: "unknown") == nil)
        #expect(GPUClass.nvidiaBlackwellRTX.description == "nvidia-blackwellrtx")
        #expect(GPUClass(name: "amd-cdna4") == .amdCDNA4)
    }

    /// Every listed device carries a vendor, and a machine without GPUs lists none rather than failing.
    @Test func gpuDevicesListWhatTheProcessDrives() throws {
        #expect(try gpuDevices().allSatisfy { $0.vendor != .unknown && $0.gpuClass != nil })
    }

    @Test func errorsRenderTheirStatusAndMessage() throws {
        let error = try #require(failure(of: try Model.loadRam(path: "/nonexistent/uform.safetensors")))
        #expect(!error.message.isEmpty && error.description == "uform: io: \(error.message)")
    }

    @Test func theLibraryNamesItsCapabilities() {
        #expect(!capabilitiesCompiled().isEmpty)
        #expect(!capabilitiesAvailable().isEmpty)
    }
}

@Suite("Executors")
struct ExecutorTests {

    @Test func factoriesMirrorTheirConstructors() throws {
        let executor = try Executor.forRam(memoryDomains: [], threads: 2, deferredLimit: 4)
        try executor.drain(mood: .spinning)
        _ = try Arena(on: executor)
        _ = try Arena()
        var pool = 0
        // The stub lends any non-null pool; the C suite borrows a real ForkUnion one.
        _ = try Executor.onPool(pool: &pool)
        #expect(failureStatus(of: try Executor.forVram(devices: [0])) == .device)
        // The streams cross behind a pointer that carries no count, so the binding counts them.
        #expect(failureStatus(of: try Executor.forVram(devices: [0, 1], streams: [nil])) == .shapeMismatch)
    }
}

@Suite("Models")
struct ModelTests {

    @Test func loadsDescribeTheirCheckpoint() throws {
        let executor = try Executor.forRam()
        let path = try checkpoint()
        let loaded = try Model.loadRam(path: path, executor: executor, quantization: .e4m3)
        let description = try loaded.describe()
        #expect(description.dimensions > 0 && description.context > 0)
        #expect(description.quantization == .e4m3)
        #expect(description.memoryUsage > 0)
        #expect(try loaded.on(executor).describe().memoryUsage == description.memoryUsage)
        #expect(try loaded.deferred(executor).describe().wait().dimensions == description.dimensions)

        let bytes = [UInt8](try Data(contentsOf: URL(fileURLWithPath: path)))
        #expect(try Model.loadRamBuffer(bytes.span.bytes).describe().dimensions == description.dimensions)
    }

    /// Serial is every build's floor, so a pin to it loads and is described, and a RAM model names no GPU class.
    @Test func aSerialPinIsDescribed() throws {
        let description = try Model.loadRam(path: try checkpoint(), cpuClass: .serial).describe()
        #expect(description.cpuClass == .serial && description.gpuClass == nil)
    }

    @Test func refusalsArriveAsStatuses() throws {
        let path = try checkpoint()
        #expect(failureStatus(of: try Model.loadRam(path: path, quantization: .f32)) == .unsupported)
        // The engine refuses an empty path as an argument, and the stub fails to open it.
        #expect([.invalidArgument, .io].contains(failureStatus(of: try Model.loadRam(path: ""))))
        #expect(failureStatus(of: try Model.loadRamBuffer([UInt8]().span.bytes)) == .invalidArgument)
        #expect(failureStatus(of: try Model.loadVram(path: path)) == .device)
        #expect(failureStatus(of: try Model.loadVramBuffer([UInt8](repeating: 1, count: 8).span.bytes)) == .device)
    }

    @Test func aLoadHearsItsTensorsAndStops() throws {
        let path = try checkpoint()
        let reports = Reports()
        _ = try Model.loadRam(path: path) { completed, _ in
            reports.count += 1
            reports.completed = completed
            return .continue
        }
        #expect(reports.count == 1 && reports.completed == 1)
        #expect(failureStatus(of: try Model.loadRam(path: path) { _, _ in .cancel }) == .cancelled)
    }
}

@Suite("Encoding")
struct EncodeTests {

    /// Both surfaces come from one call, the states one row per byte of text, and nothing past what was written.
    @Test func textLandsBothSurfaces() throws {
        let model = try model()
        let dimensions = try model.describe().dimensions
        let arena = try Arena()
        let encoded = try model.textEncode(["ab", "cde"], outputs: [.embeddings, .states], arena: arena)
        let embeddings = try #require(encoded.embeddings)
        let states = try #require(encoded.states)
        #expect(embeddings.count == 2 && embeddings.columns == dimensions && embeddings.scalar == .f32)
        #expect(embeddings.stride == dimensions * 4)
        #expect(embeddings.allSatisfy { row in row.count == dimensions && row.allSatisfy(\.isFinite) })
        #expect(states.map(\.count) == [2, 3])
        #expect(states.rows.count == 5)
        #expect(encoded.stats.truncated == 0 && encoded.stats.tokens == 5)
        #expect(embeddings.withUnsafeBytes(\.count) == 2 * dimensions * 4)
    }

    /// `outputs` decides which C pointers cross, so a surface it leaves out is nil, and naming none is refused.
    @Test func outputsChooseTheSurfaces() throws {
        let model = try model()
        let arena = try Arena()
        let embedded = try model.textEncode(["blue house"], arena: arena)
        #expect(embedded.embeddings?.count == 1 && embedded.states == nil)
        let stated = try model.textEncode(["blue house"], outputs: .states, arena: arena)
        #expect(stated.embeddings == nil && stated.states?[0].count == 10)
        #expect(failureStatus(of: try model.textEncode(["blue house"], outputs: [], arena: arena)) == .invalidArgument)
    }

    /// Token vectors share the states' offsets and have unit length, or are refused by a model without a token head.
    @Test func tokenVectorsFollowTheTokenHead() throws {
        let model = try model()
        let tokenWidth = try model.describe().tokenWidth
        let arena = try Arena()
        let texts = ["ab", "cde"]
        let outputs: Outputs = [.states, .tokenVectors]
        guard tokenWidth > 0 else {
            #expect(failureStatus(of: try model.textEncode(texts, outputs: outputs, arena: arena)) == .unsupported)
            return
        }
        let encoded = try model.textEncode(texts, outputs: outputs, arena: arena)
        let vectors = try #require(encoded.tokenVectors)
        #expect(vectors.map(\.count) == encoded.states?.map(\.count))
        #expect(vectors.rows.columns == tokenWidth)
        #expect(vectors.rows.allSatisfy { row in abs(row.reduce(0) { $0 + $1 * $1 } - 1) < 0.02 })
    }

    /// Each item's probabilities cover its choices and sum to one, or a model without the byte decoder refuses.
    @Test func choicesSumToOnePerItem() throws {
        let model = try model()
        let groups = try model.describe().choiceTemperatureGroups
        let arena = try Arena()
        let choices: [[UInt8]] = [Array("yn".utf8), Array("ABC".utf8), Array("s".utf8)]
        let texts = ["keep this candidate? ", "expand which neighbor? ", "stop? "]
        guard groups > 0 else {
            #expect(failureStatus(of: try model.textChoose(texts, choices: choices, arena: arena)) == .unsupported)
            return
        }
        let chosen = try model.textChoose(texts, choices: choices, temperatureGroups: [0, 0, 0], arena: arena)
        #expect(chosen.probabilities.map(\.count) == [2, 3, 1])
        #expect(chosen.probabilities.allSatisfy { abs($0.reduce(0, +) - 1) < 0.001 })
        let past = [0, UInt32(groups), 0]
        let refused = failureStatus(
            of: try model.textChoose(texts, choices: choices, temperatureGroups: past, arena: arena)
        )
        #expect(refused == .invalidArgument)
    }

    @Test func longItemsAreTruncatedAtTheContext() throws {
        let model = try model()
        let context = try model.describe().context
        let encoded = try model.textEncode(
            [String(repeating: "x", count: context + 88), "short"],
            outputs: .states,
            arena: try Arena()
        )
        #expect(encoded.states?.map(\.count) == [context, 5])
        #expect(encoded.stats.truncated == 1 && encoded.stats.tokens == context + 5)
    }

    /// Every verb lowers its inputs into the tape its C structure reads, and equal items span equal tokens.
    @Test func everyVerbLandsItsShapes() throws {
        let model = try model()
        let arena = try Arena()
        // Each call rewinds the arena, so every result is read before the next call.
        let images = try #require(try model.imageEncode([png(), png()], outputs: .states, arena: arena).states)
        #expect(images[0].count > 0 && images.map(\.count) == [images[0].count, images[0].count])
        let recordings = try #require(try model.audioEncode([wav(), wav()], outputs: .states, arena: arena).states)
        #expect(recordings[0].count > 0 && recordings.map(\.count) == [recordings[0].count, recordings[0].count])
        // The engine fails an undecodable file whole as malformed; the stub decodes nothing, so it encodes any bytes.
        let clip: [UInt8] = Array(repeating: 1, count: 40)
        if let status = failureStatus(of: try model.videoEncode([clip], arena: arena)) { #expect(status == .malformed) }

        let pixels = try model.imageEncodePixels(
            [Array(repeating: 9, count: 16 * 8 * 3)],
            heights: [16],
            widths: [8],
            outputs: .states,
            arena: arena
        )
        #expect(try #require(pixels.states)[0].count > 0)

        let samples = try model.audioEncodeSamples([Array(repeating: 0.5, count: 600)], sampleRate: 16000, arena: arena)
        #expect(samples.embeddings?.count == 1 && samples.stats.tokens > 0)
    }

    @Test func malformedInputsAreRefused() throws {
        let model = try model()
        let arena = try Arena()
        let pixels: [[UInt8]] = [Array(repeating: 9, count: 10)]
        #expect(
            failureStatus(of: try model.imageEncodePixels(pixels, heights: [2], widths: [2], arena: arena))
                == .shapeMismatch
        )
        // The heights and widths cross behind pointers that carry no count, so the binding counts them.
        #expect(
            failureStatus(of: try model.imageEncodePixels(pixels, heights: [], widths: [], arena: arena))
                == .shapeMismatch
        )
        #expect(failureStatus(of: try model.audioEncodeSamples([[1]], sampleRate: 0, arena: arena)) == .invalidArgument)
        let f64 = EncodeConfig(outputsScalar: .f64)
        #expect(failureStatus(of: try model.textEncode(["a"], arena: arena, config: f64)) == .unsupported)
    }

    /// Narrower outputs widen back to `Float` within their precision, from the same seeded draws.
    @Test func narrowScalarsWidenOnRead() throws {
        let model = try model()
        let dimensions = try model.describe().dimensions
        let wide = values(try model.textEncode(["scalars"], arena: try Arena()).embeddings)
        for (scalar, tolerance) in [(Scalar.bf16, Float(1) / 128), (Scalar.f16, Float(1) / 1024)] {
            let config = EncodeConfig(outputsScalar: scalar)
            let encoded = try model.textEncode(["scalars"], arena: try Arena(), config: config)
            #expect(encoded.embeddings?.scalar == scalar && encoded.embeddings?.stride == dimensions * 2)
            let narrow = values(encoded.embeddings)
            #expect(zip(wide[0], narrow[0]).allSatisfy { abs($0 - $1) <= tolerance })
        }
    }

    @Test func progressReportsAndStops() throws {
        let model = try model()
        let arena = try Arena()
        let reports = Reports()
        let texts = (0 ..< 8).map { "item \($0)" }
        let reporting = EncodeConfig(
            progress: { completed, _ in
                reports.count += 1
                reports.completed = completed
                return .continue
            },
            progressSteps: 4
        )
        _ = try model.textEncode(texts, arena: arena, config: reporting)
        #expect(reports.count == 4 && reports.completed == 8)
        let stopped = EncodeConfig(progress: { _, _ in .cancel })
        #expect(failureStatus(of: try model.textEncode(texts, arena: arena, config: stopped)) == .cancelled)
        #expect(try model.textEncode(texts, arena: arena).embeddings?.count == 8)
    }

    /// A view keeps its arena alive, so a result outlives the caller's references to the model and the arena.
    @Test func resultsOutliveTheirModelAndArena() throws {
        var model: Model? = try model()
        var arena: Arena? = try Arena()
        let encoded = try #require(model).textEncode(["kept"], arena: try #require(arena))
        let expected = values(encoded.embeddings)
        model = nil
        arena = nil
        #expect(values(encoded.embeddings) == expected)
    }

    /// The arena's next call rewinds it, so a view an earlier call answered traps on its next read rather than
    /// reading what the later call wrote there.
    #if os(macOS) || os(Linux)
        @Test func aLaterCallEndsTheEarlierViews() async {
            let exit = await #expect(processExitsWith: .failure, observing: [\.standardErrorContent]) {
                let model = try Model.loadRam(path: try checkpoint())
                let arena = try Arena()
                let first = try model.textEncode(["first"], arena: arena)
                _ = try model.textEncode(["second"], arena: arena)
                _ = first.embeddings?[0]
            }
            let reported = exit.map { String(decoding: $0.standardErrorContent, as: UTF8.self) } ?? ""
            #expect(reported.contains("the arena was rewound since this view was written"))
        }
    #endif
}

@Suite("Deferred")
struct DeferredTests {

    /// The calling thread, a blocking executor call and a deferred one land the same rows, and a deferred call
    /// answers the same every time it is waited on.
    @Test func everyReceiverLandsTheSameRows() throws {
        let model = try model()
        let executor = try Executor.forRam()
        let texts = ["one", "two words"]
        let here = values(try model.textEncode(texts, arena: try Arena()).embeddings)
        let blocking = values(try model.on(executor).textEncode(texts, arena: try Arena(on: executor)).embeddings)
        let call = try model.deferred(executor).textEncode(texts, arena: try Arena(on: executor))
        #expect(call.status == .deferred)
        let deferred = values(try call.wait().embeddings)
        #expect(call.status == .success)
        #expect(here == blocking && blocking == deferred)
        #expect(values(try call.wait().embeddings) == deferred)
    }

    /// A blocking call on the same executor runs the deferred calls submitted before it first.
    @Test func aBlockingCallRunsTheDeferredOnesBeforeIt() throws {
        let model = try model()
        let executor = try Executor.forRam()
        let call = try model.deferred(executor).imageEncode([png()], arena: try Arena(on: executor))
        #expect(call.status == .deferred)
        _ = try model.on(executor).describe()
        #expect(call.status == .success)
    }

    /// Draining runs every earlier call, and the deferred inputs were copied at submission.
    @Test func drainingFinishesTheCopiedCalls() throws {
        let model = try model()
        let executor = try Executor.forRam()
        var texts = ["copied"]
        let call = try model.deferred(executor).textEncode(texts, outputs: .states, arena: try Arena(on: executor))
        texts = []
        try executor.drain()
        #expect(call.status == .success)
        #expect(try call.wait().states?[0].count == 6)
    }

    /// Awaiting suspends rather than blocks, and reads the views back in the awaiting actor's isolation.
    @MainActor @Test func awaitingSuspendsOnTheConcurrencyRuntime() async throws {
        let model = try model()
        let executor = try Executor.forRam()
        let call = try model.deferred(executor).audioEncodeSamples(
            [Array(repeating: 0.25, count: 512)],
            sampleRate: 8000,
            outputs: [.embeddings, .states],
            arena: try Arena(on: executor)
        )
        #expect(call.status == .deferred)
        let encoded = try await call.value
        #expect(try #require(encoded.states)[0].count > 0 && encoded.embeddings?.count == 1)
    }

    /// A failure the library catches arrives through the call's status and wait.
    @Test func failuresArriveOnceFinal() throws {
        let model = try model()
        let executor = try Executor.forRam()
        let call = try model.deferred(executor).textEncode(["a"], outputs: [], arena: try Arena(on: executor))
        #expect(failureStatus(of: try call.wait()) == .invalidArgument)
        #expect(call.status == .invalidArgument)
        let stopped = EncodeConfig(progress: { _, _ in .cancel })
        let cancelled = try model.deferred(executor).textEncode(["a"], arena: try Arena(on: executor), config: stopped)
        #expect(failureStatus(of: try cancelled.wait()) == .cancelled)
    }

    /// A submission finding the deferred table full finishes at once, without waiting for an entry.
    @Test func aFullDeferredTableRefusesAtOnce() throws {
        let model = try model()
        let executor = try Executor.forRam(deferredLimit: 1)
        let held = try model.deferred(executor).textEncode(["held"], arena: try Arena(on: executor))
        let refused = try model.deferred(executor).textEncode(["refused"], arena: try Arena(on: executor))
        #expect(held.status == .deferred)
        #expect(refused.status == .contended)
        #expect(failureStatus(of: try refused.wait()) == .contended)
        #expect(try held.wait().embeddings?.count == 1)
    }

    /// A deferred call holds its arena until final, so another call naming it meanwhile is refused without ending
    /// the holder's views, and the arena serves again once the call is final.
    @Test func aDeferredCallHoldsItsArena() throws {
        let model = try model()
        let executor = try Executor.forRam()
        let arena = try Arena(on: executor)
        let call = try model.deferred(executor).textEncode(["held"], arena: arena)
        #expect(failureStatus(of: try model.textEncode(["other"], arena: arena)) == .contended)
        #expect(try model.deferred(executor).textEncode(["other"], arena: arena).status == .contended)
        let held = try call.wait()
        #expect(held.embeddings?.count == 1)
        #expect(try model.textEncode(["other", "again"], arena: arena).embeddings?.count == 2)
    }

    /// Releasing an unfinished call waits for it, so its outcome and copied inputs never dangle.
    @Test func releasingADeferredCallWaitsForIt() throws {
        let model = try model()
        let executor = try Executor.forRam()
        let reports = Reports()
        let config = EncodeConfig(progress: { completed, _ in
            reports.completed = completed
            return .continue
        })
        var call: Deferred<Encoded>? = try model.deferred(executor).textEncode(
            ["a", "b"],
            arena: try Arena(),
            config: config
        )
        #expect(call?.status == .deferred)
        call = nil
        #expect(reports.completed == 2)
    }
}

@Suite("Training")
struct TrainTests {

    /// Training is GPU-only, so a build without GPU backends refuses every constructor.
    @Test func constructorsRefuseWithoutDevices() throws {
        let executor = try Executor.forRam()
        #expect(failureStatus(of: try Mesh.bootstrap(executor: executor)) == .device)
        let refused = failureStatus(
            of: try Trainer.initVram(
                executor: executor,
                depth: 2,
                hidden: 128,
                batch: BatchShape(maxRowsPerDevice: 4, maxSegmentsPerDevice: 8, maxColumnsPerRow: 64),
                schedule: Schedule(steps: 10)
            )
        )
        #expect(refused == .device)
    }
}
