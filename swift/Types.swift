//
//  Types.swift
//  UForm
//
//  Shared vocabulary of the UForm Swift SDK, mirroring `c/uform/types.h`: statuses, scalars, moods, progress,
//  deferred calls, executors, arenas, and the rows calls write into arenas.
//

import Dispatch
import UFormC

// MARK: - Errors

/// What went wrong, in the vocabulary the library and every binding share - one constant per `uform_status_*_k`,
/// carrying the C enumerator itself.
///
/// A `UFormError` never carries `success` or `deferred`; a `Deferred` call reports both.
public struct Status: Hashable, Sendable, CustomStringConvertible {
    /// The C enumerator.
    let raw: uform_status_t

    /// The call succeeded and carries no message.
    public static let success = Status(raw: uform_status_success_k)
    /// The call was submitted deferred and is not final yet.
    public static let deferred = Status(raw: uform_status_deferred_k)
    /// Device, pinned or host memory could not be allocated.
    public static let outOfMemory = Status(raw: uform_status_out_of_memory_k)
    /// Item counts, a pixel shape or a batch envelope disagree with the call or the model.
    public static let shapeMismatch = Status(raw: uform_status_shape_mismatch_k)
    /// A stride or packing the model cannot read.
    public static let unsupportedLayout = Status(raw: uform_status_unsupported_layout_k)
    /// A scalar, modality or name this build does not know.
    public static let unknownKind = Status(raw: uform_status_unknown_kind_k)
    /// A checkpoint file could not be opened, read or written.
    public static let io = Status(raw: uform_status_io_k)
    /// The driver refused or the device faulted.
    public static let device = Status(raw: uform_status_device_k)
    /// Supplied bytes could not be decoded.
    public static let malformed = Status(raw: uform_status_malformed_k)
    /// The library reported neither success nor a failure.
    public static let `internal` = Status(raw: uform_status_internal_k)
    /// The request contradicts a frozen configuration.
    public static let configMismatch = Status(raw: uform_status_config_mismatch_k)
    /// A call was refused rather than waited for - a full deferred table, or an arena serving another call.
    public static let contended = Status(raw: uform_status_contended_k)
    /// A progress handler asked the call to stop; it unwound and published nothing.
    public static let cancelled = Status(raw: uform_status_cancelled_k)
    /// The verb is declared, but this residency or model does not implement it.
    public static let unsupported = Status(raw: uform_status_unsupported_k)
    /// An argument breaks the verb's contract.
    public static let invalidArgument = Status(raw: uform_status_invalid_argument_k)

    /// The status as the library spells it, like `invalid-argument`. Calls `uform_status_name`.
    public var description: String { String(cString: uform_status_name(raw)) }

    /// Reads a spelling back, and nil for one no status carries. Calls `uform_status_from_name`.
    public init?(name: String) {
        guard let raw = parsed(name, uform_status_from_name, uform_status_name) else { return nil }
        self.raw = raw
    }

    init(raw: uform_status_t) { self.raw = raw }
}

/// One failed call: a status every caller can branch on, and the detail the status cannot carry.
public struct UFormError: Swift.Error, CustomStringConvertible, Equatable, Sendable {
    /// The failure identity, shared with every other SDK.
    public let status: Status
    /// The library's static detail, or this binding's own when it refused the call.
    public let message: String

    public var description: String { "uform: \(status): \(message)" }

    init(status: Status, message: String) {
        self.status = status
        self.message = message
    }

    /// Lifts a final outcome; messages are static library storage, so the copy is the only cost.
    init(_ outcome: uform_outcome_t) {
        status = Status(raw: outcome.status)
        message = outcome.message.map { String(cString: $0) } ?? ""
    }
}

/// Reads `name` through a C `from_name` table, keeping only a value whose own spelling echoes it, since every
/// table answers some default for a name it does not know.
func parsed<Raw>(
    _ name: String,
    _ read: (uform_bytes_view_t) -> Raw,
    _ spell: (Raw) -> UnsafePointer<CChar>?
) -> Raw? {
    let raw = Array(name.utf8).withUnsafeBufferPointer {
        read(uform_bytes_view_t(data: $0.baseAddress, length: $0.count))
    }
    return spell(raw).map { String(cString: $0) } == name ? raw : nil
}

// MARK: - Scalars & Moods

/// A scalar type a model computes in and its outputs land as, numbered as USearch numbers it, carrying the C
/// enumerator. Sub-byte kinds pack several dimensions per byte, as their `xN` suffixes spell.
public struct Scalar: Hashable, Sendable, CustomStringConvertible {
    /// The C enumerator.
    let raw: uform_scalar_t

    /// No scalar named - an unset field.
    public static let unknown = Scalar(raw: uform_scalar_unknown_k)
    /// IEEE-754 binary64.
    public static let f64 = Scalar(raw: uform_scalar_f64_k)
    /// IEEE-754 binary32.
    public static let f32 = Scalar(raw: uform_scalar_f32_k)
    /// The brain floating-point format.
    public static let bf16 = Scalar(raw: uform_scalar_bf16_k)
    /// IEEE-754 binary16.
    public static let f16 = Scalar(raw: uform_scalar_f16_k)
    /// 8-bit floating point with five exponent bits.
    public static let e5m2 = Scalar(raw: uform_scalar_e5m2_k)
    /// 8-bit floating point with four exponent bits.
    public static let e4m3 = Scalar(raw: uform_scalar_e4m3_k)
    /// 6-bit floating point with three exponent bits.
    public static let e3m2 = Scalar(raw: uform_scalar_e3m2_k)
    /// 6-bit floating point with two exponent bits.
    public static let e2m3 = Scalar(raw: uform_scalar_e2m3_k)
    /// An 8-bit signed integer.
    public static let i8 = Scalar(raw: uform_scalar_i8_k)
    /// Two 4-bit signed integers per byte.
    public static let i4x2 = Scalar(raw: uform_scalar_i4x2_k)
    /// A 32-bit signed integer.
    public static let i32 = Scalar(raw: uform_scalar_i32_k)
    /// An 8-bit unsigned integer, the element of UTF-8 text and of encoded files.
    public static let u8 = Scalar(raw: uform_scalar_u8_k)
    /// Two 4-bit unsigned integers per byte.
    public static let u4x2 = Scalar(raw: uform_scalar_u4x2_k)
    /// Eight single-bit dimensions per byte.
    public static let u1x8 = Scalar(raw: uform_scalar_u1x8_k)

    /// The kind as the library spells it. Calls `uform_scalar_name`.
    public var description: String { String(cString: uform_scalar_name(raw)) }

    /// Reads a spelling back, and nil for one no kind carries. Calls `uform_scalar_from_name`.
    public init?(name: String) {
        guard let raw = parsed(name, uform_scalar_from_name, uform_scalar_name) else { return nil }
        self.raw = raw
    }

    init(raw: uform_scalar_t) { self.raw = raw }
}

/// What an executor does with its threads or streams once it goes idle, carrying the C enumerator.
public struct Mood: Hashable, Sendable, CustomStringConvertible {
    /// The C enumerator.
    let raw: uform_mood_t

    /// Idle threads and streams sleep, freeing the core.
    public static let parked = Mood(raw: uform_mood_parked_k)
    /// Idle threads and streams poll, trading a core for the lowest wake-up latency.
    public static let spinning = Mood(raw: uform_mood_spinning_k)

    /// The mood as the library spells it. Calls `uform_mood_name`.
    public var description: String { String(cString: uform_mood_name(raw)) }

    /// Reads a spelling back, and nil for one no mood carries. Calls `uform_mood_from_name`.
    public init?(name: String) {
        guard let raw = parsed(name, uform_mood_from_name, uform_mood_name) else { return nil }
        self.raw = raw
    }

    init(raw: uform_mood_t) { self.raw = raw }
}

// MARK: - Classes & Devices

/// A CPU family UForm ships tuned kernels for, carrying the C enumerator; dispatch reports its pick in this
/// vocabulary, and a RAM placement may pin one.
public struct CPUClass: Hashable, Sendable, CustomStringConvertible {
    /// The C enumerator.
    let raw: uform_cpu_class_t

    /// The portable floor, with no SIMD assumptions.
    public static let serial = CPUClass(raw: uform_cpu_class_serial_k)
    /// Haswell-class x86: AVX2 and FMA.
    public static let intelHaswell = CPUClass(raw: uform_cpu_class_intel_haswell_k)
    /// Sierra Forest-class x86: AVX-VNNI without AVX-512.
    public static let intelSierra = CPUClass(raw: uform_cpu_class_intel_sierra_k)
    /// Skylake-class x86: the AVX-512 foundation.
    public static let intelSkylake = CPUClass(raw: uform_cpu_class_intel_skylake_k)
    /// Icelake-class x86: AVX-512 VNNI and VPOPCNT.
    public static let intelIcelake = CPUClass(raw: uform_cpu_class_intel_icelake_k)
    /// Genoa-class x86: native bf16 dot products over AVX-512.
    public static let amdGenoa = CPUClass(raw: uform_cpu_class_amd_genoa_k)
    /// Sapphire Rapids-class x86: AVX-512 FP16 and the AMX tiles.
    public static let intelSapphire = CPUClass(raw: uform_cpu_class_intel_sapphire_k)
    /// Granite Rapids-class x86: the AMX-FP16 tiles.
    public static let intelGranite = CPUClass(raw: uform_cpu_class_intel_granite_k)
    /// Diamond Rapids-class x86: AVX10.2's FP8 conversions.
    public static let intelDiamond = CPUClass(raw: uform_cpu_class_intel_diamond_k)
    /// Graviton2-class Arm: the bare NEON floor.
    public static let awsG2 = CPUClass(raw: uform_cpu_class_aws_g2_k)
    /// Armv8.6 NEON with dot products and bf16.
    public static let appleM2 = CPUClass(raw: uform_cpu_class_apple_m2_k)
    /// Apple M4-class Arm: the SME outer-product tiles.
    public static let appleM4 = CPUClass(raw: uform_cpu_class_apple_m4_k)
    /// Graviton5-class Arm: Armv9.2.
    public static let awsG5 = CPUClass(raw: uform_cpu_class_aws_g5_k)
    /// Vera-class Arm: FP8 dot products.
    public static let nvidiaVera = CPUClass(raw: uform_cpu_class_nvidia_vera_k)
    /// Alibaba C930-class RISC-V: RVA23 vectors.
    public static let alibabaC930 = CPUClass(raw: uform_cpu_class_alibaba_c930_k)
    /// WebAssembly with fixed-width simd128.
    public static let wasm32V128 = CPUClass(raw: uform_cpu_class_wasm32_v128_k)
    /// WebAssembly with relaxed SIMD, threads and 64-bit addressing.
    public static let wasm64Extended = CPUClass(raw: uform_cpu_class_wasm64_extended_k)

    /// The class as the library spells it, like `intel-haswell`. Calls `uform_cpu_class_name`.
    public var description: String { String(cString: uform_cpu_class_name(raw)) }

    /// Reads a spelling back, and nil for one no class carries. Calls `uform_cpu_class_from_name`.
    public init?(name: String) {
        guard let raw = parsed(name, uform_cpu_class_from_name, uform_cpu_class_name),
            raw != uform_cpu_class_unknown_k
        else { return nil }
        self.raw = raw
    }

    /// Lifts a reported class, and nil for none.
    init?(reported raw: uform_cpu_class_t) {
        guard raw != uform_cpu_class_unknown_k else { return nil }
        self.raw = raw
    }

    init(raw: uform_cpu_class_t) { self.raw = raw }
}

/// A GPU family UForm ships tuned kernels for, carrying the C enumerator; dispatch reports its pick in this
/// vocabulary, and a VRAM placement may pin one.
public struct GPUClass: Hashable, Sendable, CustomStringConvertible {
    /// The C enumerator.
    let raw: uform_gpu_class_t

    /// Ampere-class NVIDIA, and every NVIDIA GPU without a class of its own.
    public static let nvidiaAmpere = GPUClass(raw: uform_gpu_class_nvidia_ampere_k)
    /// Ada-class NVIDIA: the fp8 conversions.
    public static let nvidiaAda = GPUClass(raw: uform_gpu_class_nvidia_ada_k)
    /// Hopper-class NVIDIA.
    public static let nvidiaHopper = GPUClass(raw: uform_gpu_class_nvidia_hopper_k)
    /// Blackwell-class NVIDIA.
    public static let nvidiaBlackwell = GPUClass(raw: uform_gpu_class_nvidia_blackwell_k)
    /// Consumer Blackwell NVIDIA.
    public static let nvidiaBlackwellRTX = GPUClass(raw: uform_gpu_class_nvidia_blackwellrtx_k)
    /// Rubin-class NVIDIA.
    public static let nvidiaRubin = GPUClass(raw: uform_gpu_class_nvidia_rubin_k)
    /// CDNA3 AMD: gfx942.
    public static let amdCDNA3 = GPUClass(raw: uform_gpu_class_amd_cdna3_k)
    /// CDNA4 AMD: gfx950.
    public static let amdCDNA4 = GPUClass(raw: uform_gpu_class_amd_cdna4_k)
    /// CDNA5 AMD: gfx1250.
    public static let amdCDNA5 = GPUClass(raw: uform_gpu_class_amd_cdna5_k)
    /// Apple7-family GPUs, and every Apple GPU without a class of its own.
    public static let appleMetal7 = GPUClass(raw: uform_gpu_class_apple_metal7_k)
    /// Apple9-family GPUs.
    public static let appleMetal9 = GPUClass(raw: uform_gpu_class_apple_metal9_k)
    /// Apple10-family GPUs.
    public static let appleMetal10 = GPUClass(raw: uform_gpu_class_apple_metal10_k)

    /// The class as the library spells it, like `nvidia-hopper`. Calls `uform_gpu_class_name`.
    public var description: String { String(cString: uform_gpu_class_name(raw)) }

    /// Reads a spelling back, and nil for one no class carries. Calls `uform_gpu_class_from_name`.
    public init?(name: String) {
        guard let raw = parsed(name, uform_gpu_class_from_name, uform_gpu_class_name),
            raw != uform_gpu_class_unknown_k
        else { return nil }
        self.raw = raw
    }

    /// Lifts a reported class, and nil for none.
    init?(reported raw: uform_gpu_class_t) {
        guard raw != uform_gpu_class_unknown_k else { return nil }
        self.raw = raw
    }

    init(raw: uform_gpu_class_t) { self.raw = raw }
}

/// The vendor whose runtime drives a GPU, carrying the C enumerator.
public struct GPUVendor: Hashable, Sendable {
    /// The C enumerator.
    let raw: uform_gpu_vendor_t

    /// A vendor this binding has no name for.
    public static let unknown = GPUVendor(raw: uform_gpu_vendor_unknown_k)
    /// NVIDIA, through the CUDA driver.
    public static let nvidia = GPUVendor(raw: uform_gpu_vendor_nvidia_k)
    /// AMD, through the HIP runtime.
    public static let amd = GPUVendor(raw: uform_gpu_vendor_amd_k)
    /// Apple, through Metal.
    public static let apple = GPUVendor(raw: uform_gpu_vendor_apple_k)
}

/// One GPU this process can drive; its position in `gpuDevices()` is the ordinal every VRAM placement names it by.
/// Mirrors `uform_gpu_device_t`.
public struct GPUDevice: Sendable {
    /// The vendor whose runtime drives the device.
    public var vendor: GPUVendor
    /// The device's ordinal in its vendor's runtime.
    public var vendorOrdinal: UInt32
    /// The class whose kernels run on it, or nil for none.
    public var gpuClass: GPUClass?
    /// Streaming multiprocessors or compute units.
    public var computeUnits: UInt32
    /// Device memory in bytes.
    public var memoryBytes: UInt64
    /// The vendor's own name for the device.
    public var name: String

    init(_ raw: uform_gpu_device_t) {
        vendor = GPUVendor(raw: raw.vendor)
        vendorOrdinal = raw.vendor_ordinal
        gpuClass = GPUClass(reported: raw.gpu_class)
        computeUnits = raw.compute_units
        memoryBytes = raw.memory_bytes
        name = withUnsafeBytes(of: raw.name) { bytes in
            String(decoding: bytes.prefix { $0 != 0 }, as: UTF8.self)
        }
    }
}

/// Every GPU this process can drive, NVIDIA devices first, then AMD; empty on a machine without GPUs. Calls
/// `uform_gpu_devices`.
public func gpuDevices() throws(UFormError) -> [GPUDevice] {
    let landing = Landing()
    let count = landing.slots(0)
    var counting = uform_gpu_devices_t(
        outcome: landing.outcome,
        devices: nil,
        devices_capacity: 0,
        devices_count: count
    )
    uform_gpu_devices(&counting)
    try landing.read {}
    let capacity = count.pointee
    let devices = landing.slots(uform_gpu_device_t(), count: capacity)
    var listing = uform_gpu_devices_t(
        outcome: landing.outcome,
        devices: devices,
        devices_capacity: capacity,
        devices_count: count
    )
    uform_gpu_devices(&listing)
    return try landing.read {
        (0 ..< Swift.min(capacity, count.pointee)).map { GPUDevice(devices[$0]) }
    }
}

// MARK: - Progress

/// Whether the call a progress handler is watching carries on or unwinds.
public enum Continuation: Sendable {
    /// Carry on to the next unit of work.
    case `continue`
    /// Stop, and report `cancelled`.
    case cancel
}

/// A periodic report of a long call's advance - units finished and units in total, items for an encoding and
/// tensors for a load - and the caller's chance to stop it.
///
/// Never invoked concurrently, so it needs no synchronization of its own, but it runs on whichever thread
/// advanced the work, hence `@Sendable`.
public typealias Progress = @Sendable (_ completed: Int, _ total: Int) -> Continuation

/// A progress handler placed in a landing, since a Swift closure has no stable C representation.
private final class ProgressState {
    let progress: Progress
    init(_ progress: @escaping Progress) { self.progress = progress }
}

/// The trampoline every reporting verb hands the library.
private let progressTrampoline: uform_progress_callback_t = { completed, total, state in
    guard let state else { return uform_cancel_k }
    let handler = Unmanaged<ProgressState>.fromOpaque(state).takeUnretainedValue()
    return handler.progress(completed, total) == .cancel ? uform_cancel_k : uform_continue_k
}

// MARK: - Landings

/// Everything one call points the library at, held where it cannot move until the call is final.
///
/// Submission copies the argument structure, so only its pointees live here: the outcome, the slots the call
/// writes, the tapes it reads, and the closures it calls back. A blocking call releases it on return, and a
/// deferred one is kept by its `Deferred` until final.
final class Landing {
    /// Whether the call blocks or returns once submitted.
    let completion: uform_completion_t
    /// The outcome the call stores its final status in.
    let outcome = UnsafeMutablePointer<uform_outcome_t>.allocate(capacity: 1)
    /// The trivial values the call reads or writes through, freed with the landing.
    private var slots: [UnsafeMutableRawPointer] = []
    /// The callback states the call reads.
    private var anchors: [AnyObject] = []
    /// The arena lent to the call, until `rewind` learns whether the call took it.
    private var lent: ArenaWrite?

    init(_ completion: uform_completion_t = uform_completion_blocking_k) {
        self.completion = completion
        outcome.initialize(to: uform_outcome_t())
    }

    /// Runs only once the call is final: it returned blocking, its `Deferred` waited, or it never crossed.
    deinit {
        for slot in slots { slot.deallocate() }
        outcome.deallocate()
    }

    /// Whether the call returns once submitted, so its result is read on a later wait.
    var isDeferred: Bool { completion == uform_completion_deferred_k }

    /// Allocates `count` trivial values, initialized to `initial`, which the call reads or writes through.
    func slots<Value>(_ initial: Value, count: Int = 1) -> UnsafeMutablePointer<Value> {
        let values = UnsafeMutablePointer<Value>.allocate(capacity: Swift.max(count, 1))
        values.initialize(repeating: initial, count: Swift.max(count, 1))
        slots.append(UnsafeMutableRawPointer(values))
        return values
    }

    /// Copies trivial `values` into the landing, answering nil for none, as the C side reads an empty list.
    func lend<Element>(_ values: [Element]) -> UnsafePointer<Element>? {
        guard let first = values.first else { return nil }
        let copy = slots(first, count: values.count)
        _ = UnsafeMutableBufferPointer(start: copy, count: values.count).update(fromContentsOf: values)
        return UnsafePointer(copy)
    }

    /// Lends the UTF-8 bytes of a path or a name, never terminated, by contract.
    func lend(_ text: String) -> uform_bytes_view_t {
        let bytes = Array(text.utf8)
        return uform_bytes_view_t(data: lend(bytes), length: bytes.count)
    }

    /// Lends the host placement every RAM load and executor takes; nil `cpuClass` derives the CPU's own.
    func ram(_ memoryDomains: [Int32], _ cpuClass: CPUClass?) -> uform_ram_options_t {
        uform_ram_options_t(
            memory_domains: lend(memoryDomains),
            memory_domains_count: memoryDomains.count,
            cpu_class: cpuClass?.raw ?? uform_cpu_class_unknown_k
        )
    }

    /// Lends the device placement every VRAM load and executor takes; nil `gpuClass` derives the devices' own.
    func vram(_ devices: [Int32], _ gpuClass: GPUClass?) -> uform_vram_options_t {
        uform_vram_options_t(
            devices: lend(devices),
            devices_count: devices.count,
            gpu_class: gpuClass?.raw ?? uform_gpu_class_unknown_k
        )
    }

    /// Lays `items` end to end with Apache Arrow offsets around them, as every ragged input crosses.
    func tape<Item: Collection>(_ items: [Item], _ scalar: Scalar) -> uform_tape_view_t
    where Item.Element: BitwiseCopyable {
        let offsets = slots(0, count: items.count + 1)
        for (position, item) in items.enumerated() { offsets[position + 1] = offsets[position] + item.count }
        let total = offsets[items.count]
        let data = UnsafeMutableRawPointer.allocate(
            byteCount: Swift.max(total, 1) * MemoryLayout<Item.Element>.stride,
            alignment: MemoryLayout<Item.Element>.alignment
        )
        slots.append(data)
        let elements = data.bindMemory(to: Item.Element.self, capacity: Swift.max(total, 1))
        for (position, item) in items.enumerated() {
            _ = UnsafeMutableBufferPointer(start: elements + offsets[position], count: item.count)
                .initialize(fromContentsOf: item)
        }
        return uform_tape_view_t(data: UnsafeRawPointer(data), offsets: UnsafePointer(offsets), scalar: scalar.raw)
    }

    /// Lends an optional handler and its report count as the progress every reporting verb takes; nil lends a
    /// null callback.
    func lend(_ progress: Progress?, steps: Int) -> uform_progress_t {
        guard let progress else { return uform_progress_t(callback: nil, state: nil, steps: steps) }
        let state = ProgressState(progress)
        anchors.append(state)
        return uform_progress_t(
            callback: progressTrampoline,
            state: Unmanaged.passUnretained(state).toOpaque(),
            steps: steps
        )
    }

    /// Lends `arena` to the call, answering the stamp its views carry once `rewind` advances the arena to it.
    func lend(_ arena: Arena) -> ArenaWrite {
        let write = ArenaWrite(arena: arena, generation: arena.generation &+ 1)
        lent = write
        return write
    }

    /// Advances the lent arena once the library took the call, after a blocking one returns or right after a
    /// deferred one is submitted, so the views earlier calls wrote there stop reading. Failed calls advance it
    /// too, and only a `contended` one, which never touched the arena, does not.
    func rewind() {
        guard let write = lent else { return }
        lent = nil
        if uform_outcome_status(outcome) != uform_status_contended_k { write.arena.generation &+= 1 }
    }

    /// Waits until the call is final, then answers its result or throws its failure.
    func read<Value>(_ result: () throws(UFormError) -> Value) throws(UFormError) -> Value {
        let status = uform_outcome_wait(outcome)
        rewind()
        guard status == uform_status_success_k else { throw UFormError(outcome.pointee) }
        return try result()
    }

    /// The submission of a verb whose result `read` builds once final: at once for a blocking call, and kept
    /// only by a deferred one, so the blocking path forms no escaping closure.
    func submitted<Value>(_ read: @escaping () throws(UFormError) -> Value) throws(UFormError) -> Submission<Value> {
        try isDeferred ? .pending(self, read) : .final(self.read(read))
    }
}

/// A call the library accepted, and how to read what it produced.
enum Submission<Value> {
    /// A blocking call's result, read once the call returned final.
    case final(Value)
    /// A deferred call's landing, and how to read its result once final.
    case pending(Landing, () throws(UFormError) -> Value)

    /// Blocks until the call is final, then answers its result or throws its failure.
    func wait() throws(UFormError) -> Value {
        switch self {
        case .final(let value): return value
        case .pending(let landing, let result): return try landing.read(result)
        }
    }
}

/// The handle a lifecycle call wrote, which its success promises.
func created<Handle>(_ slot: UnsafeMutablePointer<Handle?>) throws(UFormError) -> Handle {
    guard let handle = slot.pointee else {
        throw UFormError(status: .internal, message: "the library reported success without a handle")
    }
    return handle
}

// MARK: - Deferred Calls

/// A call submitted to an executor, which finishes there while the caller does other work.
///
/// It owns the call's outcome, outputs and copied inputs until the call is final. The handles it names, the
/// arena among them, it borrows as a blocking call does. Releasing an unfinished one waits for it first.
public final class Deferred<Value> {
    /// The landing and the result reader.
    private let submission: Submission<Value>

    init(_ submission: Submission<Value>) {
        self.submission = submission
        if case .pending(let landing, _) = submission { landing.rewind() }
    }

    /// Calls `uform_outcome_wait`.
    deinit {
        if case .pending(let landing, _) = submission { _ = uform_outcome_wait(landing.outcome) }
    }

    /// `deferred` until the call is final, then `success` or its failure, read without blocking. Calls
    /// `uform_outcome_status`.
    public var status: Status {
        guard case .pending(let landing, _) = submission else { return .success }
        return Status(raw: uform_outcome_status(landing.outcome))
    }

    /// Blocks until the call is final, then answers its result or throws its failure, the same every time it is
    /// asked. Calls `uform_outcome_wait`.
    public func wait() throws(UFormError) -> Value { try submission.wait() }
}

// A call reading arena views stays on the arena's thread, so only a Sendable result may travel.
extension Deferred: @unchecked Sendable where Value: Sendable {}

extension Deferred {
    /// Waits for the call on a Dispatch thread, so an `async` caller suspends rather than blocks and the
    /// cooperative pool keeps every thread; the result is read back in the caller's isolation. Binding-only:
    /// runs `wait`.
    nonisolated(nonsending) public var value: Value {
        get async throws(UFormError) {
            if case .pending(let landing, _) = submission {
                nonisolated(unsafe) let outcome = landing.outcome
                await withCheckedContinuation { continuation in
                    DispatchQueue.global().async {
                        _ = uform_outcome_wait(outcome)
                        continuation.resume()
                    }
                }
            }
            return try wait()
        }
    }
}

// MARK: - Executors

/// The host threads or device streams a call runs on, with the activation scratch and the staging they use.
///
/// One executor serves any number of models of its residency, so several models share one machine's cores or
/// devices; a mismatch reports `device`. Calls on one executor run in submission order.
public final class Executor: @unchecked Sendable {
    /// Where an executor computes, which the factory that spawned it named.
    enum Placement {
        case host
        case devices
    }

    /// The native handle, released by `deinit`.
    let raw: uform_executor_t
    /// Where it computes, and with it where its arenas keep their memory.
    let placement: Placement

    private init(raw: uform_executor_t, placement: Placement) {
        self.raw = raw
        self.placement = placement
    }

    /// Spawns host threads for RAM-resident models, interleaved across the compute domains local to
    /// `memoryDomains`; an empty list derives every node the process may allocate on. Zero `threads` takes 7/8 of
    /// the cores those domains carry, and zero `deferredLimit` sizes the deferred table by default. A `cpuClass`
    /// dispatches as if the CPU were of that class; nil derives its own. Calls `uform_executor_for_ram`.
    public static func forRam(
        memoryDomains: [Int32] = [],
        cpuClass: CPUClass? = nil,
        threads: Int = 0,
        deferredLimit: Int = 0
    ) throws(UFormError) -> Executor {
        let landing = Landing()
        let executor = landing.slots(uform_executor_t?.none)
        var arguments = uform_executor_for_ram_t(
            executor: executor,
            outcome: landing.outcome,
            ram: landing.ram(memoryDomains, cpuClass),
            threads: threads,
            deferred_limit: deferredLimit
        )
        uform_executor_for_ram(&arguments)
        return try landing.read { () throws(UFormError) in Executor(raw: try created(executor), placement: .host) }
    }

    /// Creates a RAM executor over a borrowed ForkUnion `fu_pool_t`, which must outlive it. Reports `unsupported`
    /// until the library wires borrowed pools. A `cpuClass` pins as `forRam` does. Calls `uform_executor_on_pool`.
    public static func onPool(
        memoryDomains: [Int32] = [],
        cpuClass: CPUClass? = nil,
        pool: UnsafeMutableRawPointer,
        deferredLimit: Int = 0
    ) throws(UFormError) -> Executor {
        let landing = Landing()
        let executor = landing.slots(uform_executor_t?.none)
        var arguments = uform_executor_on_pool_t(
            executor: executor,
            outcome: landing.outcome,
            ram: landing.ram(memoryDomains, cpuClass),
            pool: pool,
            deferred_limit: deferredLimit
        )
        uform_executor_on_pool(&arguments)
        return try landing.read { () throws(UFormError) in Executor(raw: try created(executor), placement: .host) }
    }

    /// Opens one submission stream on every named device, or borrows the caller's `streams`, one per device in
    /// `devices` order; an empty list derives one free device. A zero `scratchLimit` lets per-device scratch grow
    /// as batches ask. A `gpuClass` dispatches every device as that class; nil derives theirs. Calls
    /// `uform_executor_for_vram`.
    public static func forVram(
        devices: [Int32] = [],
        gpuClass: GPUClass? = nil,
        streams: [UnsafeMutableRawPointer?]? = nil,
        scratchLimit: Int = 0,
        deferredLimit: Int = 0
    ) throws(UFormError) -> Executor {
        // The C side reads one stream per device behind a pointer that carries no count.
        if let streams, streams.count != devices.count {
            throw UFormError(status: .shapeMismatch, message: "the streams must name one per device")
        }
        let landing = Landing()
        let executor = landing.slots(uform_executor_t?.none)
        let lentStreams = streams.flatMap { landing.lend($0) }
        var arguments = uform_executor_for_vram_t(
            executor: executor,
            outcome: landing.outcome,
            vram: landing.vram(devices, gpuClass),
            streams: lentStreams,
            scratch_limit: scratchLimit,
            deferred_limit: deferredLimit
        )
        uform_executor_for_vram(&arguments)
        return try landing.read { () throws(UFormError) in Executor(raw: try created(executor), placement: .devices) }
    }

    /// Calls `uform_executor_free`.
    deinit { uform_executor_free(raw) }

    /// Waits for every call submitted before it, then adopts `mood` whenever the executor next goes idle; a
    /// failure is the executor's own, never one of a drained call. Calls `uform_executor_drain`.
    public func drain(mood: Mood = .parked) throws(UFormError) {
        let landing = Landing()
        var arguments = uform_executor_drain_t(executor: raw, outcome: landing.outcome, mood: mood.raw)
        uform_executor_drain(&arguments)
        try landing.read {}
    }
}

// MARK: - Arenas

/// Memory the library writes every output whose length scales with a call into, reused across calls so a steady
/// loop allocates once.
///
/// Every output verb requires one and answers views reading it in place. Each call given the arena rewinds it,
/// so the views an earlier call answered stop reading: any read of them traps, as an out-of-bounds index does.
/// To keep a result past the next call, give that call another arena. An arena serves one call at a time,
/// deferred ones included, and a call naming it while another holds it is refused with `contended`.
public final class Arena {
    /// The native handle, released by `deinit`.
    let raw: uform_arena_t
    /// The device executor that made it, held so the arena is freed first; nil for host pages.
    let device: Executor?
    /// Advanced by every call given the arena, so a view remembers which call wrote it.
    var generation: UInt64 = 0

    /// Makes an arena where `executor` computes, host pages or its GPU's memory, decided once; nil makes host
    /// pages for calls on the calling thread. Calls `uform_arena_init`.
    public init(on executor: Executor? = nil) throws(UFormError) {
        let landing = Landing()
        let arena = landing.slots(uform_arena_t?.none)
        var arguments = uform_arena_init_t(arena: arena, executor: executor?.raw, outcome: landing.outcome)
        uform_arena_init(&arguments)
        raw = try landing.read { () throws(UFormError) in try created(arena) }
        device = executor?.placement == .devices ? executor : nil
    }

    /// Calls `uform_arena_free`.
    deinit { uform_arena_free(raw) }
}

/// The arena a view reads, and which of the calls given it wrote the view.
struct ArenaWrite {
    let arena: Arena
    let generation: UInt64

    /// Refuses a read the arena no longer backs, or one reaching device memory.
    func check() {
        precondition(arena.generation == generation, "the arena was rewound since this view was written")
        precondition(arena.device == nil, "the view is device memory, which no host read reaches")
    }

    /// Refuses as `check()` does, then a `position` past `count`, as a collection does.
    func check(_ position: Int, below count: Int) {
        check()
        precondition(position >= 0 && position < count, "the position is out of bounds")
    }
}

// MARK: - Rows

/// A run of rows a call wrote packed into its arena, one per item, read in place. Mirrors `uform_rows_view_t`.
///
/// Rows of `f32`, `bf16` and `f16` read as `Float`; `withUnsafeBytes` hands any scalar on as stored, such as
/// to a USearch index.
public struct Rows: RandomAccessCollection {
    /// The arena the rows live in, and which call wrote them.
    let write: ArenaWrite
    /// The first row.
    let base: UnsafeRawPointer?
    /// Bytes between the starts of consecutive rows.
    public let stride: Int
    /// Values per row, the model's embedding dimensions.
    public let columns: Int
    /// The scalar type the rows hold.
    public let scalar: Scalar
    /// One past the last row.
    public let endIndex: Int

    /// The first row's position, which a collection walk starts from.
    public var startIndex: Int { 0 }

    init(write: ArenaWrite, base: UnsafeRawPointer?, stride: Int, columns: Int, scalar: Scalar, count: Int) {
        self.write = write
        self.base = base
        self.stride = stride
        self.columns = columns
        self.scalar = scalar
        endIndex = count
    }

    init(write: ArenaWrite, view: uform_rows_view_t, count: Int) {
        let scalar = Scalar(raw: view.scalar)
        self.init(
            write: write,
            base: view.data,
            stride: view.stride,
            columns: view.columns,
            scalar: scalar,
            count: count
        )
    }

    /// One row's values.
    public subscript(row: Int) -> Row {
        write.check(row, below: endIndex)
        return Row(write: write, base: base! + row * stride, scalar: scalar, endIndex: columns)
    }

    /// Lends the rows' bytes as stored to `body`, which must not keep the buffer past its return.
    public func withUnsafeBytes<Result, Failure: Swift.Error>(
        _ body: (UnsafeRawBufferPointer) throws(Failure) -> Result
    ) throws(Failure) -> Result {
        write.check()
        return try body(UnsafeRawBufferPointer(start: endIndex == 0 ? nil : base, count: endIndex * stride))
    }

    /// One row of an arena's values, read in place and widened to `Float`.
    public struct Row: RandomAccessCollection {
        /// The arena the row lives in, and which call wrote it.
        let write: ArenaWrite
        /// The row's first value.
        let base: UnsafeRawPointer
        /// The scalar type the row holds.
        let scalar: Scalar
        /// One past the last value in the row.
        public let endIndex: Int

        /// The first value's position, which a collection walk starts from.
        public var startIndex: Int { 0 }

        /// The value at a position, widened from the row's scalar.
        public subscript(position: Int) -> Float {
            write.check(position, below: endIndex)
            switch scalar {
            case .f32: return base.loadUnaligned(fromByteOffset: position * 4, as: Float.self)
            case .bf16:
                let bits = base.loadUnaligned(fromByteOffset: position * 2, as: UInt16.self)
                return Float(bitPattern: UInt32(bits) << 16)
            case .f16: return widened(half: base.loadUnaligned(fromByteOffset: position * 2, as: UInt16.self))
            default: preconditionFailure("\(scalar) rows read through withUnsafeBytes")
            }
        }
    }
}

/// Widens IEEE half precision bits to a single, by hand, since `Float16` is missing on Intel macOS.
private func widened(half bits: UInt16) -> Float {
    let sign: Float = bits & 0x8000 == 0 ? 1 : -1
    let exponent = UInt32(bits >> 10) & 0x1f
    let mantissa = UInt32(bits & 0x3ff)
    switch exponent {
    case 0: return sign * Float(mantissa) * 0x1p-24
    case 0x1f: return Float(bitPattern: UInt32(bits & 0x8000) << 16 | 0x7f80_0000 | mantissa << 13)
    default: return Float(bitPattern: UInt32(bits & 0x8000) << 16 | (exponent + 112) << 23 | mantissa << 13)
    }
}

/// A ragged run of rows a call wrote into its arena, as many per item as the item had tokens left after
/// truncation, read in place. Mirrors `uform_rows_tape_view_t`.
public struct RowsTape: RandomAccessCollection {
    /// Every item's rows, end to end.
    public let rows: Rows
    /// One more offset than items, in rows.
    let offsets: UnsafePointer<Int>?
    /// One past the last item.
    public let endIndex: Int

    /// The first item's position, which a collection walk starts from.
    public var startIndex: Int { 0 }

    init(write: ArenaWrite, view: uform_rows_tape_view_t, count: Int) {
        offsets = view.offsets
        endIndex = count
        let total = count == 0 ? 0 : view.offsets?[count] ?? 0
        rows = Rows(write: write, view: view.rows, count: total)
    }

    /// One item's rows, one per token it kept.
    public subscript(item: Int) -> Rows {
        rows.write.check(item, below: endIndex)
        let first = offsets![item]
        return Rows(
            write: rows.write,
            base: rows.base.map { $0 + first * rows.stride },
            stride: rows.stride,
            columns: rows.columns,
            scalar: rows.scalar,
            count: offsets![item + 1] - first
        )
    }
}

/// A ragged run of `f32` values a call wrote into its arena, as many per item as the item asked for, read in
/// place. Mirrors `uform_tape_view_t`.
public struct Tape: RandomAccessCollection {
    /// The arena the values live in, and which call wrote them.
    let write: ArenaWrite
    /// The first value; nil when the call wrote none.
    let base: UnsafeRawPointer?
    /// One more offset than items, in values.
    let offsets: UnsafePointer<Int>?
    /// One past the last item.
    public let endIndex: Int

    /// The first item's position, which a collection walk starts from.
    public var startIndex: Int { 0 }

    init(write: ArenaWrite, view: uform_tape_view_t, count: Int) {
        self.write = write
        base = view.data
        offsets = view.offsets
        endIndex = count
    }

    /// One item's values.
    public subscript(item: Int) -> Rows.Row {
        write.check(item, below: endIndex)
        let first = offsets![item]
        let length = offsets![item + 1] - first
        // An item asking for nothing may sit on a null base, which its empty row never reads.
        let start = base.map { $0 + first * MemoryLayout<Float>.stride } ?? UnsafeRawPointer(bitPattern: 4)!
        return Rows.Row(write: write, base: start, scalar: .f32, endIndex: length)
    }
}

// MARK: - Introspection

/// The version of the linked library, like `4.0.0`. Calls `uform_version_name`.
public func version() -> String { String(cString: uform_version_name()) }

/// The hardware capabilities the linked binary was compiled with, comma separated. Calls
/// `uform_capabilities_compiled`.
public func capabilitiesCompiled() -> String { String(cString: uform_capabilities_compiled()) }

/// The hardware capabilities the current machine offers at runtime, comma separated. Calls
/// `uform_capabilities_available`.
public func capabilitiesAvailable() -> String { String(cString: uform_capabilities_available()) }
