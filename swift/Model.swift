//
//  Model.swift
//  UForm
//
//  The model lifecycle, mirroring `c/uform/model.h`: load an encoder into RAM or VRAM, describe it, and the
//  receivers that choose where its verbs run.
//

import UFormC

/// What a loaded model reports about itself. Mirrors `uform_model_description_t`.
public struct ModelDescription: Sendable {
    /// Scalars per embedding row; Matryoshka prefixes of it are embeddings too.
    public var dimensions: Int
    /// Tokens one item may span before it is truncated.
    public var context: Int
    /// The scalar the transformer blocks run in.
    public var quantization: Scalar
    /// Bytes the model holds; activations are the executor's.
    public var memoryUsage: Int
    /// The class a RAM model's tower was packed for, or nil for a VRAM model.
    public var cpuClass: CPUClass?
    /// The class every replica of a VRAM model runs as, or nil for a RAM model.
    public var gpuClass: GPUClass?
    /// Scalars per token vector; zero when the checkpoint carries no token head.
    public var tokenWidth: Int
    /// Question types the choice temperatures were calibrated over: zero without the byte decoder, one when
    /// uncalibrated.
    public var choiceTemperatureGroups: Int

    init(_ raw: uform_model_description_t) {
        dimensions = raw.dimensions
        context = raw.context
        quantization = Scalar(raw: raw.quantization)
        memoryUsage = raw.memory_usage
        cpuClass = CPUClass(reported: raw.cpu_class)
        gpuClass = GPUClass(reported: raw.gpu_class)
        tokenWidth = raw.token_width
        choiceTemperatureGroups = raw.choice_temperature_groups
    }
}

/// A loaded encoder: its weights, their packed copies, and on a GPU the session its attention runs on.
///
/// One model serves every modality it was trained on and is freely shareable across threads. Every verb called
/// on the model runs on the calling thread and blocks until final; `on(_:)` runs the same verbs on an executor,
/// and `deferred(_:)` submits them there.
public final class Model: @unchecked Sendable {
    /// The native handle, released by `deinit`.
    let raw: uform_model_t

    private init(raw: uform_model_t) { self.raw = raw }

    /// Calls `uform_model_free`.
    deinit { uform_model_free(raw) }

    /// Copies a safetensors checkpoint file into host memory on `memoryDomains`, an empty list deriving every
    /// node the process may allocate on, and packs it for the CPU. `quantization` is the scalar the blocks run
    /// in: `bf16` as stored, or `e4m3` quantized at load. A `cpuClass` packs as if the CPU were of that class; nil
    /// derives its own. `progress` hears tensors placed. Calls `uform_model_load_ram`.
    public static func loadRam(
        path: String,
        executor: Executor? = nil,
        memoryDomains: [Int32] = [],
        cpuClass: CPUClass? = nil,
        quantization: Scalar = .bf16,
        progress: Progress? = nil,
        progressSteps: Int = 0
    ) throws(UFormError) -> Model {
        let landing = Landing()
        let model = landing.slots(uform_model_t?.none)
        var arguments = uform_model_load_ram_t(
            model: model,
            executor: executor?.raw,
            outcome: landing.outcome,
            completion: landing.completion,
            path: landing.lend(path),
            ram: landing.ram(memoryDomains, cpuClass),
            quantization: quantization.raw,
            progress: landing.lend(progress, steps: progressSteps)
        )
        uform_model_load_ram(&arguments)
        return try adopted(model, landing)
    }

    /// Copies a checkpoint held in memory into host memory, as `loadRam` does a path; nothing of `buffer` is kept.
    /// Calls `uform_model_load_ram_buffer`.
    public static func loadRamBuffer(
        _ buffer: RawSpan,
        executor: Executor? = nil,
        memoryDomains: [Int32] = [],
        cpuClass: CPUClass? = nil,
        quantization: Scalar = .bf16,
        progress: Progress? = nil,
        progressSteps: Int = 0
    ) throws(UFormError) -> Model {
        let landing = Landing()
        let model = landing.slots(uform_model_t?.none)
        buffer.withUnsafeBytes { bytes in
            var arguments = uform_model_load_ram_buffer_t(
                model: model,
                executor: executor?.raw,
                outcome: landing.outcome,
                completion: landing.completion,
                buffer: uform_bytes_view_t(bytes),
                ram: landing.ram(memoryDomains, cpuClass),
                quantization: quantization.raw,
                progress: landing.lend(progress, steps: progressSteps)
            )
            uform_model_load_ram_buffer(&arguments)
        }
        return try adopted(model, landing)
    }

    /// Copies a checkpoint file onto the named GPUs, replicated on each, an empty list deriving one free device,
    /// and packs it there. A `gpuClass` dispatches every device as that class; nil derives theirs. Calls
    /// `uform_model_load_vram`.
    public static func loadVram(
        path: String,
        executor: Executor? = nil,
        devices: [Int32] = [],
        gpuClass: GPUClass? = nil,
        quantization: Scalar = .bf16,
        progress: Progress? = nil,
        progressSteps: Int = 0
    ) throws(UFormError) -> Model {
        let landing = Landing()
        let model = landing.slots(uform_model_t?.none)
        var arguments = uform_model_load_vram_t(
            model: model,
            executor: executor?.raw,
            outcome: landing.outcome,
            completion: landing.completion,
            path: landing.lend(path),
            vram: landing.vram(devices, gpuClass),
            quantization: quantization.raw,
            progress: landing.lend(progress, steps: progressSteps)
        )
        uform_model_load_vram(&arguments)
        return try adopted(model, landing)
    }

    /// Copies a checkpoint held in memory onto the named GPUs, as `loadVram` does a path. Calls
    /// `uform_model_load_vram_buffer`.
    public static func loadVramBuffer(
        _ buffer: RawSpan,
        executor: Executor? = nil,
        devices: [Int32] = [],
        gpuClass: GPUClass? = nil,
        quantization: Scalar = .bf16,
        progress: Progress? = nil,
        progressSteps: Int = 0
    ) throws(UFormError) -> Model {
        let landing = Landing()
        let model = landing.slots(uform_model_t?.none)
        buffer.withUnsafeBytes { bytes in
            var arguments = uform_model_load_vram_buffer_t(
                model: model,
                executor: executor?.raw,
                outcome: landing.outcome,
                completion: landing.completion,
                buffer: uform_bytes_view_t(bytes),
                vram: landing.vram(devices, gpuClass),
                quantization: quantization.raw,
                progress: landing.lend(progress, steps: progressSteps)
            )
            uform_model_load_vram_buffer(&arguments)
        }
        return try adopted(model, landing)
    }

    /// Adopts the handle a blocking load wrote, or throws its failure.
    private static func adopted(
        _ model: UnsafeMutablePointer<uform_model_t?>,
        _ landing: Landing
    ) throws(UFormError) -> Model {
        try landing.read { () throws(UFormError) in Model(raw: try created(model)) }
    }

    /// The same verbs, blocking on `executor` after every call submitted there before them. Binding-only: names
    /// the executor every call passes.
    public func on(_ executor: Executor) -> ModelOn { ModelOn(model: self, executor: executor) }

    /// The same verbs, submitted to `executor` and answered at once as a `Deferred`. Binding-only: names the
    /// executor and the deferred completion every call passes.
    public func deferred(_ executor: Executor) -> ModelDeferredOn { ModelDeferredOn(model: self, executor: executor) }

    /// The verbs on the calling thread.
    var here: ModelOn { ModelOn(model: self, executor: nil) }

    /// Reports the model's geometry and placement. Calls `uform_model_describe`.
    public func describe() throws(UFormError) -> ModelDescription { try here.describe() }
}

// MARK: - Receivers

/// A model's verbs on an executor, each blocking until final after every call submitted there before it; see
/// `Model.on(_:)` and the model's own verbs for what each one does.
public struct ModelOn: Sendable {
    /// The model every verb addresses.
    let model: Model
    /// The executor every verb runs on, or nil for the calling thread.
    let executor: Executor?

    /// Runs one verb, blocking until it is final.
    func run<Value>(_ verb: (ModelCall) throws(UFormError) -> Submission<Value>) throws(UFormError) -> Value {
        try verb(ModelCall(model: model, executor: executor, landing: Landing())).wait()
    }

    /// Calls `uform_model_describe`.
    public func describe() throws(UFormError) -> ModelDescription {
        try run { (call) throws(UFormError) in try call.describe() }
    }
}

/// A model's verbs submitted to an executor, each answered at once as a `Deferred` that finishes there.
///
/// A deferred call copies its inputs, so the caller's arrays may go at once. The model, the executor and the
/// arena are borrowed as a blocking call borrows them, since releasing any of them waits for the calls naming it.
public struct ModelDeferredOn: Sendable {
    /// The model every verb addresses.
    let model: Model
    /// The executor every verb is submitted to.
    let executor: Executor

    /// Submits one verb, whose landing its `Deferred` keeps until final.
    func submit<Value>(
        _ verb: (ModelCall) throws(UFormError) -> Submission<Value>
    ) throws(UFormError) -> Deferred<Value> {
        Deferred(try verb(ModelCall(model: model, executor: executor, landing: Landing(uform_completion_deferred_k))))
    }

    /// Calls `uform_model_describe`.
    public func describe() throws(UFormError) -> Deferred<ModelDescription> {
        try submit { (call) throws(UFormError) in try call.describe() }
    }
}

// MARK: - Calls

/// The model, executor and landing one verb lowers into its argument structure, and the one place each verb is
/// written.
struct ModelCall {
    /// The model the call addresses.
    let model: Model
    /// The executor the call runs on, or nil for the calling thread.
    let executor: Executor?
    /// Where the call's outcome, inputs and slots live until it is final.
    let landing: Landing

    func describe() throws(UFormError) -> Submission<ModelDescription> {
        let description = landing.slots(uform_model_description_t())
        var arguments = uform_model_describe_t(
            model: model.raw,
            executor: executor?.raw,
            outcome: landing.outcome,
            completion: landing.completion,
            description: description
        )
        uform_model_describe(&arguments)
        return try landing.submitted { ModelDescription(description.pointee) }
    }
}

extension uform_bytes_view_t {
    /// Views a byte buffer as the pointer-and-length pair every buffer argument takes.
    init(_ bytes: UnsafeRawBufferPointer) {
        self.init(data: bytes.baseAddress?.assumingMemoryBound(to: UInt8.self), length: bytes.count)
    }
}
