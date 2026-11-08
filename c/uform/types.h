/**
 *  @file c/uform/types.h
 *  @author Ash Vardanian
 *  @date October 8, 2026
 *  @brief Shared vocabulary of the UForm C interface: scalar types, shapes, outcomes, executors, arenas,
 *      and library introspection.
 *
 *  Everything @c model.h, @c encode.h and @c train.h have in common, and nothing they don't. Include
 *  @c uform.h for all of them, or one alone, since each pulls this one in. Statuses, scalars, rows, tapes,
 *  progress, arenas and the executor contract mirror USearch's @c types.h value for value, so embeddings and
 *  failures cross between the two libraries unchanged.
 *
 *  @section types_verbs Verbs
 *
 *  Every verb that takes options, outputs, or callbacks takes one argument structure named after it -
 *  @c uform_text_encode takes a @c uform_text_encode_t - so optional arguments are skipped by name and their
 *  order never matters. Name lookups and the @c _free family stay positional. Every structure opens in the
 *  same order: the handle, @c executor, @c outcome, then @c completion. Inputs follow, then outputs and the
 *  @c arena they land in, then knobs, and last @c progress and @c stats.
 *
 *  Structures carry no size or version field, so they grow source-compatibly only: initialize them by
 *  designator, which zeroes every field left out, and expect the major version to change whenever one does.
 *  The library never logs.
 *
 *  @section types_asynchrony Asynchrony
 *
 *  Every call names its @c completion, and synchronous and deferred calls share one shape. A blocking call,
 *  the default, returns with its final status stored and every output readable. A deferred call returns once
 *  submitted and finishes on its executor. A call with a @c NULL executor runs on the calling thread and is
 *  final when it returns. Submission copies the structure; everything it points to stays borrowed until final.
 *
 *  - @c outcome->status reads @c uform_status_deferred_k until final, even for a call refused at submission.
 *  - The message, then the final status, are stored with release ordering and a wake.
 *  - The caller reads the status with acquire ordering, polling through @c uform_outcome_status or waiting
 *    through @c uform_outcome_wait.
 *  - A GPU executor stores the final status from a host callback that still fires after a device fault.
 *  - Executors own their streams, so a caller orders dependent calls by waiting on the first.
 *  - A submission finding the deferred table full, sized by @c deferred_limit, finishes
 *    @c uform_status_contended_k.
 *  - A call stops early only when its progress handler returns @c uform_cancel_k.
 *
 *  @code{.c}
 *  uform_outcome_t outcomes[4] = {0};
 *  for (size_t shard = 0; shard != 4; ++shard) {
 *      encodes[shard].outcome = &outcomes[shard];
 *      encodes[shard].completion = uform_completion_deferred_k;
 *      uform_image_encode(&encodes[shard]);
 *  }
 *  for (size_t shard = 0; shard != 4; ++shard) uform_outcome_wait(&outcomes[shard]);
 *  @endcode
 *
 *  @section types_memory Memory
 *
 *  Every byte has one owner, and each owner picks where its bytes live once, when it is created:
 *
 *  - Callers lend memory, never a way to allocate it. No allocator callback crosses the interface, and
 *    every pointer a caller passes is an input, classified through the driver once per call.
 *  - Weights and their packed copies belong to the model, placed by its load's residency, as
 *    @c uform_model_load_vram places them on the named devices. Weights, gradients and optimizer moments
 *    under training belong to the trainer, placed on its executor's devices.
 *  - Activations, scratch and staging belong to the executor, placed at its creation and grown to a
 *    high-water mark.
 *  - Every output whose length scales with the call lands in an arena: pooled embeddings, per-token states
 *    and their offsets alike. Only fixed-size reports, such as stats and descriptions, stay caller fields.
 *  - Executors make arenas. @c uform_arena_init places one where its executor computes, host pages on its
 *    memory domains or the memory of its GPU, decided once.
 *  - A call rewinds the arena it is given, invalidating the previous call's outputs, and writes views into
 *    it. Any verb may reuse any arena of its residency, one call at a time.
 *  - Blocking calls borrow the arena; deferred calls hold it until final.
 *
 *  Outputs are views the call writes, so none carries a capacity or a stride for the caller to pick, and an
 *  arena keeps its high-water mark, so a steady loop allocates once. On a VRAM executor the outputs are
 *  device memory, which the caller copies out itself when it wants them elsewhere, or hands to a USearch
 *  VRAM index as they are.
 *
 *  @code{.c}
 *  uform_arena_t arena = NULL;
 *  uform_arena_init(&(uform_arena_init_t){.arena = &arena, .executor = executor, .outcome = &outcome});
 *  for (size_t batch = 0; batch != batches_count; ++batch) {
 *      encode.texts = texts[batch], encode.arena = arena, encode.embeddings = &embeddings;
 *      uform_text_encode(&encode);
 *      if (encode.outcome->status != uform_status_success_k) break; // blocking, so already final
 *      consume(embeddings, encode.count); // valid until the next call given `arena`
 *  }
 *  uform_arena_free(arena);
 *  @endcode
 *
 *  @section types_shapes Shapes
 *
 *  A verb's @c count is its batch axis and nothing else. Inputs are views, where a zero stride repeats the
 *  first element as BLAS does; outputs are views too, written packed into the arena, so none carries a
 *  capacity. Ragged inputs and outputs are tapes with Apache Arrow offsets: one more of them than entries,
 *  around the elements laid end to end.
 */
#ifndef UNUM_UFORM_TYPES_H
#define UNUM_UFORM_TYPES_H

#include <stddef.h> // `size_t`
#include <stdint.h> // `uint8_t`, `int32_t`

/* Inferring target OS: Windows, macOS, Linux, FreeBSD, or WASI */
#if defined(WIN32) || defined(_WIN32) || defined(__WIN32__) || defined(__NT__)
#define UFORM_OS_WINDOWS_ 1
#else
#define UFORM_OS_WINDOWS_ 0
#endif
#if defined(__APPLE__) && defined(__MACH__)
#define UFORM_OS_APPLE_ 1
#else
#define UFORM_OS_APPLE_ 0
#endif
#if defined(__linux__)
#define UFORM_OS_LINUX_ 1
#else
#define UFORM_OS_LINUX_ 0
#endif
#if defined(__FreeBSD__)
#define UFORM_OS_FREEBSD_ 1
#else
#define UFORM_OS_FREEBSD_ 0
#endif
#if defined(__wasi__)
#define UFORM_OS_WASI_ 1
#else
#define UFORM_OS_WASI_ 0
#endif
#define UFORM_OS_POSIX_ (UFORM_OS_LINUX_ || UFORM_OS_APPLE_ || UFORM_OS_FREEBSD_)

#ifdef __cplusplus
extern "C" {
#endif

/** Marks exported functions: exported while building the shared library, imported when consuming it on
 *  Windows, and of default visibility on ELF and Mach-O. Define it to override. */
#ifndef UFORM_EXPORT
#if (defined(_WIN32) || defined(__CYGWIN__)) && defined(UFORM_EXPORTING)
#define UFORM_EXPORT __declspec(dllexport)
#elif defined(_WIN32) || defined(__CYGWIN__)
#define UFORM_EXPORT __declspec(dllimport)
#elif defined(__GNUC__) || defined(__clang__)
#define UFORM_EXPORT __attribute__((visibility("default")))
#else
#define UFORM_EXPORT
#endif
#endif // UFORM_EXPORT

/** The library's version, which the release tooling rewrites in place. */
#define UFORM_VERSION_MAJOR 4
#define UFORM_VERSION_MINOR 0
#define UFORM_VERSION_PATCH 0

/** A borrowed run of bytes and its length, never null-terminated by contract - the shape every path,
 *  checkpoint buffer and name arrives in. */
typedef struct uform_bytes_view_t {

    /** First byte of the run. */
    uint8_t const* data;

    /** Bytes the run spans, with no terminator assumed or written. */
    size_t length;
} uform_bytes_view_t;

/**
 *  @brief What went wrong, numbered as USearch numbers it, so one failure carries one identity across the
 *      two libraries and every binding that validates before calling them.
 *
 *  Values USearch spends on index states UForm has no counterpart for - a read-only handle, an exhausted ID
 *  space, an uninitialized reservation - stay unused here rather than renumbered.
 *
 *  Statuses fall into five categories. Resources: @c uform_status_out_of_memory_k,
 *  @c uform_status_io_k and @c uform_status_device_k. Input: @c uform_status_shape_mismatch_k,
 *  @c uform_status_unsupported_layout_k, @c uform_status_unknown_kind_k,
 *  @c uform_status_malformed_k and @c uform_status_invalid_argument_k. State:
 *  @c uform_status_config_mismatch_k, @c uform_status_contended_k and
 *  @c uform_status_unsupported_k. Interruptions: @c uform_status_cancelled_k and
 *  @c uform_status_deferred_k. Defects: @c uform_status_internal_k. Any verb that allocates, binds
 *  a device or touches a file may also return a Resources status; any verb taking a progress
 *  callback may return @c uform_status_cancelled_k. A verb's own block lists only what its contract
 *  adds.
 */
typedef enum uform_status_t {

    /** The call completed; @c message carries nothing. */
    uform_status_success_k = 0,

    /** The call was submitted deferred and is not final yet; the one non-final status. */
    uform_status_deferred_k = 1,

    /** Device, pinned, or host memory could not be allocated, or the executor's scratch ceiling is too low. */
    uform_status_out_of_memory_k = 2,

    /** Item counts, a pixel shape, or a batch envelope disagree with the call or the model. */
    uform_status_shape_mismatch_k = 3,

    /** A stride or packing the model cannot read. */
    uform_status_unsupported_layout_k = 4,

    /** A scalar, modality, or name this build does not know. */
    uform_status_unknown_kind_k = 5,

    /** A checkpoint file could not be opened, read, or written. */
    uform_status_io_k = 7,

    /** The driver refused or the device faulted; after a fault the executor refuses every later call. */
    uform_status_device_k = 8,

    /** Supplied bytes could not be decoded: a checkpoint, an image, a clip, or a recording. */
    uform_status_malformed_k = 9,

    /** The library reported neither success nor a failure - a contract violation, so it names a defect. */
    uform_status_internal_k = 10,

    /** The request contradicts a frozen configuration, like resuming a checkpoint of another geometry. */
    uform_status_config_mismatch_k = 13,

    /** The call was refused rather than waited for, like a submission to a full deferred table, or a
     *  blocking call from inside a callback of the executor it would wait on. */
    uform_status_contended_k = 14,

    /** A progress handler asked the call to stop; it unwound and published nothing. */
    uform_status_cancelled_k = 15,

    /** The verb is declared for every residency or model, but this one does not implement it, like a
     *  modality the model was not trained on; no retry or smaller request can help. */
    uform_status_unsupported_k = 16,

    /** An argument breaks the verb's contract - a required pointer left @c NULL, inputs inside the arena the
     *  call rewinds, a name no table spells. */
    uform_status_invalid_argument_k = 17,
} uform_status_t;

/** Outcome of one call: a status every binding can branch on, and a static message carrying the detail the
 *  status cannot. Messages are static and @b don't need to be deallocated, and are @c NULL exactly when the
 *  status is @c uform_status_success_k. */
typedef struct uform_outcome_t {

    /** The status bindings branch on; @c uform_status_deferred_k until a deferred call settles. */
    uform_status_t status;

    /** The static detail, @c NULL exactly when @c status is @c uform_status_success_k. */
    char const* message;
} uform_outcome_t;

/** Scalar types a model computes in and its outputs land as, numbered as USearch's @c usearch_scalar_t, so a
 *  run of embeddings is a run USearch indexes without conversion. Sub-byte types pack several dimensions per
 *  byte, as their @c xN suffixes spell. */
typedef enum uform_scalar_t {

    /** No scalar named - an unset field, refused rather than written at the wrong width. */
    uform_scalar_unknown_k = 0,

    /** IEEE-754 binary64. */
    uform_scalar_f64_k = 1,

    /** IEEE-754 binary32. */
    uform_scalar_f32_k = 2,

    /** Truncated f32 with an 8-bit exponent and a 7-bit mantissa. */
    uform_scalar_bf16_k = 3,

    /** IEEE-754 binary16. */
    uform_scalar_f16_k = 4,

    /** FP8 with a 5-bit exponent: ±57344, with infinities and NaNs. */
    uform_scalar_e5m2_k = 5,

    /** FP8 with a 4-bit exponent: ±448, no infinities, and two NaN codes. */
    uform_scalar_e4m3_k = 6,

    /** FP6 with a 3-bit exponent, byte-aligned with two padding bits. */
    uform_scalar_e3m2_k = 7,

    /** FP6 with a 2-bit exponent, byte-aligned with two padding bits. */
    uform_scalar_e2m3_k = 8,

    /** 8-bit signed integer. */
    uform_scalar_i8_k = 10,

    /** Two signed nibbles per byte, even dimensions in the low nibble. */
    uform_scalar_i4x2_k = 11,

    /** 32-bit signed integer. */
    uform_scalar_i32_k = 12,

    /** 8-bit unsigned integer - the element of UTF-8 text and of encoded files. */
    uform_scalar_u8_k = 20,

    /** Two unsigned nibbles per byte, even dimensions in the low nibble. */
    uform_scalar_u4x2_k = 21,

    /** Eight single-bit dimensions per byte, the sign of each embedding dimension - the form USearch's
     *  Hamming and Jaccard metrics score. */
    uform_scalar_u1x8_k = 22,
} uform_scalar_t;

/**
 *  @brief A run of rows along a call's batch axis - written packed into an arena as an output, one row per
 *      item, field for field USearch's @c usearch_rows_view_t, so pooled embeddings feed an index as they are.
 *
 *  The call writes one row's bytes as @c stride, and @c columns and @c scalar as it wrote them.
 */
typedef struct uform_rows_view_t {

    /** Base pointer of the first row. */
    void const* data;

    /** Bytes between the starts of consecutive rows. */
    size_t stride;

    /** Values per row, the model's embedding @c dimensions. */
    size_t columns;

    /** The scalar type the rows hold. */
    uform_scalar_t scalar;
} uform_rows_view_t;

/**
 *  @brief A borrowed ragged batch as Apache Arrow lays one out: elements end to end, and @c count + 1 offsets
 *      around them.
 *
 *  Entry @c i spans elements @c offsets[i] up to @c offsets[i+1]. What an element is belongs to the verb: a
 *  byte of an encoded file or of UTF-8 text, a byte of an RGB pixel row, a sample of a recording. The data may
 *  live on the executor's device or in host memory; the offsets always live in host memory, since they drive
 *  host-side dispatch.
 */
typedef struct uform_tape_view_t {

    /** First element of the tape, which every offset counts from. */
    void const* data;

    /** The call's @c count + 1 non-decreasing offsets, in elements. */
    size_t const* offsets;

    /** The scalar type the elements hold: @c u8 for text and encoded files, @c f32 for decoded samples. */
    uform_scalar_t scalar;
} uform_tape_view_t;

/**
 *  @brief A ragged run of rows a call wrote into its arena, for outputs with as many rows per item as the
 *      item had tokens left after truncation: rows end to end, and one more offset than items.
 *
 *  Item @c i spans rows @c offsets[i] up to @c offsets[i+1], so the last offset is the total.
 */
typedef struct uform_rows_tape_view_t {

    /** Every item's rows, end to end, packed. */
    uform_rows_view_t rows;

    /** The call's @c count + 1 non-decreasing offsets into @c rows, in rows. */
    size_t const* offsets;
} uform_rows_tape_view_t;

/** Whether the call a handler is watching carries on or unwinds. */
typedef enum uform_continuation_t {

    /** Carry on to the next unit of work. */
    uform_continue_k = 0,

    /** Stop and report @c uform_status_cancelled_k. */
    uform_cancel_k = 1,
} uform_continuation_t;

/**
 *  @brief Periodic report of a long call's advance, and the caller's chance to stop it. Counts are in the
 *      units the verb consumes - items for an encoding, tensors for a load.
 *
 *  Never invoked concurrently, and never after the call it was passed to stores its final status.
 *
 *  Arguments, in order:
 *  - @c completed: units finished so far, never decreasing across one call's reports.
 *  - @c total: units the call will finish if it runs to completion.
 *  - @c state: the opaque pointer registered alongside the handler, forwarded verbatim.
 */
typedef uform_continuation_t (*uform_progress_callback_t)( //
    size_t completed, size_t total, void* state);

/**
 *  @brief How a long call reports its advance: a handler, its state, and how many reports it hears.
 *
 *  A batch verb splits its items into @c steps nearly equal slices, hands each to its engine whole and
 *  reports after each; a load, whose engine counts its own tensors, forwards only the ticks that cross into
 *  the next slice. Without a handler the call runs as one slice and reports nothing.
 */
typedef struct uform_progress_t {

    /** The @b optional handler; @c NULL skips every report. */
    uform_progress_callback_t callback;

    /** Opaque pointer forwarded into every @c callback call. */
    void* state;

    /** Reports a call makes at most, the last at completion; zero derives to ten. */
    size_t steps;
} uform_progress_t;

/**
 *  @brief Blocks on an atomic wait until @p outcome is final, with acquire ordering.
 *  @param[in] outcome An outcome passed to a call, read until its status leaves @c uform_status_deferred_k.
 *  @return The final status.
 *
 *  Never wait on a call's own outcome from inside its progress handler: the call finishes only after it
 *  returns.
 */
UFORM_EXPORT uform_status_t uform_outcome_wait(uform_outcome_t* outcome);

/**
 *  @brief Reads the status of @p outcome without waiting, with the acquire ordering its final store pairs
 *      with, so a final status also publishes every output of the call.
 *  @param[in] outcome An outcome passed to a call.
 *  @return @c uform_status_deferred_k while the call is not final, then its final status.
 *
 *  The one way to poll: the status word is atomic only by this contract, so a caller reading the field
 *  directly needs an atomic load its language may not offer on foreign memory.
 */
UFORM_EXPORT uform_status_t uform_outcome_status(uform_outcome_t* outcome);

/** Whether a call returns with its final status, or once submitted to its executor. */
typedef enum uform_completion_t {

    /** Return with the final status stored and every output readable; the default. */
    uform_completion_blocking_k = 0,

    /** Return once submitted, the status deferred until final; the caller waits or polls. */
    uform_completion_deferred_k = 1,
} uform_completion_t;

/** The CPU families UForm ships tuned variants for - dispatch reports its pick in this vocabulary,
 *  and callers may pin one. The enumerators and their ordinals are USearch's. */
typedef enum uform_cpu_class_t {
    uform_cpu_class_unknown_k = 0,

    /** The portable floor: no SIMD assumptions. */
    uform_cpu_class_serial_k,

    /** Haswell-class x86: AVX2 & FMA, the 256-bit floor. */
    uform_cpu_class_intel_haswell_k,

    /** Sierra Forest-class x86 - the E-core Xeons: AVX-VNNI and its successors without AVX-512. */
    uform_cpu_class_intel_sierra_k,

    /** Skylake-class x86: the AVX-512 foundation tier. */
    uform_cpu_class_intel_skylake_k,

    /** Icelake-class x86: AVX-512 VNNI & VPOPCNT. */
    uform_cpu_class_intel_icelake_k,

    /** Genoa-class x86 - Zen 4 and newer: native bf16 dot products over AVX-512. */
    uform_cpu_class_amd_genoa_k,

    /** Sapphire Rapids-class x86: the Genoa bundle plus AVX-512 FP16 and the AMX-BF16 & AMX-INT8
     *  tiles. */
    uform_cpu_class_intel_sapphire_k,

    /** Granite Rapids-class x86: the Sapphire Rapids bundle plus the AMX-FP16 tiles. */
    uform_cpu_class_intel_granite_k,

    /** Diamond Rapids-class x86: the Granite Rapids bundle plus AVX10.2's FP8 conversions. */
    uform_cpu_class_intel_diamond_k,

    /** Graviton2-class Arm - Neoverse N1, the bare NEON floor; phones predating Armv8.6 land here
     *  too. */
    uform_cpu_class_aws_g2_k,

    /** Armv8.6+ NEON: dot products, bf16 & f16 fused multiplies - Apple M2, Cortex-A710 & X2
     *  class silicon. */
    uform_cpu_class_apple_m2_k,

    /** Apple M4-class Arm: the Armv8.6 NEON bundle plus the SME outer-product tiles. */
    uform_cpu_class_apple_m4_k,

    /** Graviton5-class Arm - Neoverse V3, Armv9.2; reserved, as its kits equal @c apple_m2's. */
    uform_cpu_class_aws_g5_k,

    /** Vera-class Arm - NVIDIA's Armv9.2-generation cores with FP8 dot products. */
    uform_cpu_class_nvidia_vera_k,

    /** Alibaba C930-class RISC-V - T-Head's XuanTie core, RVA23 vectors. */
    uform_cpu_class_alibaba_c930_k,

    /** WebAssembly every engine loads: 32-bit, single-threaded, the fixed-width simd128 tier. */
    uform_cpu_class_wasm32_v128_k,

    /** WebAssembly with relaxed SIMD, shared memory for real thread pools, and 64-bit
     *  addressing. */
    uform_cpu_class_wasm64_extended_k,

    /** The one class of a monoversioned build: every capability its floor implies or its preset
     *  vouches for, as the kernel libraries compiled them. */
    uform_cpu_class_monoversioned_k,
} uform_cpu_class_t;

/** The GPU families UForm ships tuned variants for, the device twin of @c uform_cpu_class_t:
 *  one class per generation of kernel kits. The enumerators and their ordinals are USearch's. */
typedef enum uform_gpu_class_t {
    uform_gpu_class_unknown_k = 0,

    /** Ampere-class NVIDIA and every GPU without a class of its own - A100, A10 & RTX 30. */
    uform_gpu_class_nvidia_ampere_k,

    /** Ada-class NVIDIA - L4, L40S & RTX 40: the fp8 conversions. */
    uform_gpu_class_nvidia_ada_k,

    /** Hopper-class NVIDIA - H100 & H200. */
    uform_gpu_class_nvidia_hopper_k,

    /** Blackwell-class NVIDIA - B200, B300 & GB300. */
    uform_gpu_class_nvidia_blackwell_k,

    /** Consumer Blackwell NVIDIA - RTX 50, RTX PRO 6000 & DGX Spark. */
    uform_gpu_class_nvidia_blackwellrtx_k,

    /** Rubin-class NVIDIA - the Vera-Rubin platform's device half. */
    uform_gpu_class_nvidia_rubin_k,

    /** CDNA3 AMD - MI300X & MI325X: gfx942, FNUZ fp8. */
    uform_gpu_class_amd_cdna3_k,

    /** CDNA4 AMD - MI350X & MI355X: gfx950, OCP fp8. */
    uform_gpu_class_amd_cdna4_k,

    /** CDNA5 AMD - MI400: gfx1250. */
    uform_gpu_class_amd_cdna5_k,

    /** Apple7-family GPUs and every Apple GPU without a class of its own - M1 & M2. */
    uform_gpu_class_apple_metal7_k,

    /** Apple9-family GPUs - M3 & M4. */
    uform_gpu_class_apple_metal9_k,

    /** Apple10-family GPUs - M5. */
    uform_gpu_class_apple_metal10_k,
} uform_gpu_class_t;

/** The GPU vendors one binary drives at once. */
typedef enum uform_gpu_vendor_t {
    uform_gpu_vendor_unknown_k = 0,

    /** NVIDIA, through the CUDA driver. */
    uform_gpu_vendor_nvidia_k,

    /** AMD, through the HIP runtime. */
    uform_gpu_vendor_amd_k,

    /** Apple, through Metal. */
    uform_gpu_vendor_apple_k,
} uform_gpu_vendor_t;

/** One GPU as UForm numbers it: its position in @c uform_gpu_devices is the UForm device ordinal
 *  every VRAM call names. */
typedef struct uform_gpu_device_t {

    /** The vendor whose runtime drives the device. */
    uform_gpu_vendor_t vendor;

    /** The device's ordinal in its vendor's runtime, as @c nvidia-smi or @c rocm-smi list it. */
    uint32_t vendor_ordinal;

    /** The class whose kernels run on it. */
    uform_gpu_class_t gpu_class;

    /** Streaming multiprocessors or compute units. */
    uint32_t compute_units;

    /** Device memory in bytes. */
    uint64_t memory_bytes;

    /** The vendor's own name for the device, NUL-terminated and truncated to fit. */
    char name[64];
} uform_gpu_device_t;

/** The arguments of @c uform_gpu_devices. */
typedef struct uform_gpu_devices_t {

    /** Output for the outcome, final when the call returns. */
    uform_outcome_t* outcome;

    /** Output for up to @c devices_capacity devices, in UForm ordinal order; may be @c NULL. */
    uform_gpu_device_t* devices;

    /** Entries @c devices holds. */
    size_t devices_capacity;

    /** Output for every device this process can drive, which may exceed @c devices_capacity. */
    size_t* devices_count;
} uform_gpu_devices_t;

/**
 *  @brief Lists every GPU this process can drive, NVIDIA devices first, then AMD, so the position
 *      of a device is the UForm ordinal @c uform_vram_options_t names it by.
 *
 *  A vendor this build does not compile, or whose runtime is absent, lists no devices, so the call
 *  succeeds with a zero count on a machine without GPUs.
 */
UFORM_EXPORT void uform_gpu_devices(uform_gpu_devices_t* arguments);

/** The host memory domains a RAM-resident model and executor place their bytes in - one NUMA node each. */
typedef struct uform_ram_options_t {

    /** NUMA node IDs, borrowed for the duration of the call; @c NULL derives every node the process may
     *  allocate on. */
    int32_t const* memory_domains;

    /** Number of entries in @c memory_domains; zero derives. */
    size_t memory_domains_count;

    /** The class to dispatch as if the CPU were of it, admitting every rung whose features it
     *  holds; @c uform_cpu_class_unknown_k derives the CPU's own. A class this CPU cannot run
     *  reports @c uform_status_unsupported_k. */
    uform_cpu_class_t cpu_class;
} uform_ram_options_t;

/** The devices a VRAM-resident model, trainer and executor place their bytes in. */
typedef struct uform_vram_options_t {

    /** UForm device ordinals, positions in the listing @c uform_gpu_devices reports, borrowed
     *  for the duration of the call; @c NULL derives one free device. */
    int32_t const* devices;

    /** Number of entries in @c devices; zero derives. */
    size_t devices_count;

    /** The class every named device dispatches as; @c uform_gpu_class_unknown_k derives the
     *  devices' own, which must agree, or the call reports @c uform_status_config_mismatch_k. A
     *  class a named device cannot run, or no compiled variant takes, reports
     *  @c uform_status_unsupported_k. */
    uform_gpu_class_t gpu_class;
} uform_vram_options_t;

/**
 *  @brief Everything a call needs to run but a model does not own: the host threads or device streams that
 *      advance the work, the activation scratch, and the pinned bounce buffer staging passes through.
 *
 *  One executor serves any number of models of its residency, so several models share one machine's cores or
 *  devices instead of each claiming a private set; a mismatch reports @c uform_status_device_k. Scratch grows
 *  to the largest batch asked of it, is rewound per call, and is kept between calls.
 *
 *  Calls on one executor run in submission order. Every executor tracks its deferred calls in a fixed table
 *  sized at creation, so submitting never allocates.
 *
 *  The library stages only to change layout on the side that computes, or to make pageable memory DMA-able:
 *
 *      Case                                          Action
 *      Device inputs on the executor's device        read in place
 *      Pinned or managed host inputs                 a direct asynchronous upload
 *      Pageable host inputs                          the pinned bounce buffer, then an upload
 *      Encoded media needing a host decoder          decoded on the executor's host threads, then uploaded
 *      Outputs in an arena on the executor's device  the kernel writes them directly
 */
typedef struct uform_executor_handle_t* uform_executor_t;

/** The arguments of @c uform_executor_for_ram. */
typedef struct uform_executor_for_ram_t {

    /** Output for the new executor. */
    uform_executor_t* executor;

    /** Output for the outcome, final once its status leaves @c uform_status_deferred_k. */
    uform_outcome_t* outcome;

    /** The memory domains of the models it will drive; copied, not borrowed. */
    uform_ram_options_t ram;

    /** Threads to spawn; zero takes 7/8 of the cores those domains carry. */
    size_t threads;

    /** Deferred calls it tracks at once, sizing its deferred table; zero takes a default. */
    size_t deferred_limit;
} uform_executor_for_ram_t;

/** Spawns host threads for RAM-resident models, interleaved across the compute domains local to the named
 *  memory domains. */
UFORM_EXPORT void uform_executor_for_ram(uform_executor_for_ram_t* arguments);

/** The arguments of @c uform_executor_on_pool. */
typedef struct uform_executor_on_pool_t {

    /** Output for the new executor. */
    uform_executor_t* executor;

    /** Output for the outcome, final once its status leaves @c uform_status_deferred_k. */
    uform_outcome_t* outcome;

    /** The memory domains of the models it will drive; copied, not borrowed. */
    uform_ram_options_t ram;

    /** The ForkUnion @c fu_pool_t to borrow; it must outlive the executor, and keeps its lender's mood. */
    void* pool;

    /** Deferred calls it tracks at once, sizing its deferred table; zero takes a default. */
    size_t deferred_limit;
} uform_executor_on_pool_t;

/** Creates a RAM executor over a borrowed ForkUnion pool, so UForm, USearch and UStore in one process share
 *  one set of threads while each keeps its own scratch. */
UFORM_EXPORT void uform_executor_on_pool(uform_executor_on_pool_t* arguments);

/** The arguments of @c uform_executor_for_vram. */
typedef struct uform_executor_for_vram_t {

    /** Output for the new executor. */
    uform_executor_t* executor;

    /** Output for the outcome, final once its status leaves @c uform_status_deferred_k. */
    uform_outcome_t* outcome;

    /** The devices of the models it will drive; copied, not borrowed. */
    uform_vram_options_t vram;

    /** The memory domains and the CPU class of its host decoders; copied, not borrowed. */
    uform_ram_options_t ram;

    /** The @b optional borrowed CUDA streams, one per device in @c vram order; @c NULL creates its
     *  own. Borrowing is not wired yet, so any non-null value reports
     *  @c uform_status_unsupported_k. */
    void* const* streams;

    /** The @b optional ceiling on scratch bytes per device; zero lets scratch grow as batches ask. A batch
     *  larger than the ceiling fits runs in several passes. */
    size_t scratch_limit;

    /** Deferred calls it tracks at once, sizing its deferred table; zero takes a default. */
    size_t deferred_limit;
} uform_executor_for_vram_t;

/**
 *  @brief Opens one submission stream on every named device, all of them driven by whichever
 *      host thread submits; borrowing the caller's streams reports @c uform_status_unsupported_k
 *      for now.
 *
 *  Text and JPEG images never leave the device; the rest of decoding - container demux, PNG and
 *  WebP, audio, and every modality on AMD - runs on host threads the executor spawns beside its
 *  streams, 7/8 of the cores local to the memory domains @c ram names, dispatching as its
 *  @c cpu_class . A GPU executor never changes process-wide device flags.
 */
UFORM_EXPORT void uform_executor_for_vram(uform_executor_for_vram_t* arguments);

/** What an executor does with its threads or streams once it goes idle. */
typedef enum uform_mood_t {

    /** Sleep on futexes or events created with blocking sync, freeing the core. */
    uform_mood_parked_k = 0,

    /** Poll them, trading a core for the lowest wake-up latency. */
    uform_mood_spinning_k = 1,
} uform_mood_t;

/** The arguments of @c uform_executor_drain. */
typedef struct uform_executor_drain_t {

    /** The executor whose submitted calls are awaited. */
    uform_executor_t executor;

    /** Output for an executor-level failure, such as a device fault; never one of a drained call. */
    uform_outcome_t* outcome;

    /** What the executor does whenever it next goes idle; the last drain wins. */
    uform_mood_t mood;
} uform_executor_drain_t;

/** Waits for every call submitted before the drain began, so later submissions can't hold it up, then sets
 *  the mood the executor adopts whenever it next goes idle. Reports only executor-level failures. */
UFORM_EXPORT void uform_executor_drain(uform_executor_drain_t* arguments);

/** Drains every call, releases the scratch and the streams it owns, returns borrowed streams and pools, and
 *  frees the executor; @c NULL is safe, and it cannot fail, so takes no outcome. */
UFORM_EXPORT void uform_executor_free(uform_executor_t executor);

/**
 *  @brief Memory the library allocates for the outputs only a call can size, reused across calls.
 *      @ref types_memory
 *
 *  Made by @c uform_arena_init where its executor computes: host pages on its memory domains, or the memory of
 *  its GPU, and a GPU that runs out reports @c uform_status_out_of_memory_k rather than falling back to the
 *  host. Each call given the arena rewinds it, which invalidates every output of the previous call, and
 *  refills it, growing only when the new outputs outsize everything it held, so a steady loop allocates once.
 *  To keep one call's outputs alive past the next call, pass the next call another arena.
 *
 *  - A @c NULL arena is refused with @c uform_status_invalid_argument_k.
 *  - A call whose inputs point into the arena it is about to rewind reports
 *    @c uform_status_invalid_argument_k.
 *  - An arena serves one call at a time; a second call naming it finishes @c uform_status_contended_k, and a
 *    deferred one does so at submission.
 *  - A call on another residency or device than the arena's, or a call naming a GPU arena with another
 *    executor than the one that made it, @c NULL included, reports @c uform_status_device_k.
 *  - A blocking call borrows the arena until it returns, and a deferred one holds it until final.
 *  - A call that fails or is cancelled leaves its outputs @c NULL.
 *
 *  @note A GPU arena grows, fills and frees on the first stream of the executor that made it, so it serves
 *      only calls on that executor, and must be freed before that executor is.
 */
typedef struct uform_arena_handle_t* uform_arena_t;

/** The arguments of @c uform_arena_init. */
typedef struct uform_arena_init_t {

    /** Output for the new arena. */
    uform_arena_t* arena;

    /** The executor whose calls the arena serves, which decides where it lives; @c NULL serves calls on the
     *  calling thread from host pages. */
    uform_executor_t executor;

    /** Output for the outcome, final once its status leaves @c uform_status_deferred_k. */
    uform_outcome_t* outcome;
} uform_arena_init_t;

/**
 *  @brief Creates an empty arena where @c executor computes, so its placement is decided once and GPU code
 *      keeps its outputs on the device without asking.
 *
 *  Nothing is allocated until a call writes into it. An arena of host pages may outlive the executor that
 *  made it, keeping its own memory until it is freed; a GPU arena keeps the rule @ref uform_arena_t notes.
 */
UFORM_EXPORT void uform_arena_init(uform_arena_init_t* arguments);

/**
 *  @brief Frees the arena and every output in it; @c NULL is safe, and it cannot fail, so takes no outcome.
 *  @param[in,out] arena The arena to be freed; invalid afterwards.
 *
 *  Waits until every deferred call naming the handle is final.
 */
UFORM_EXPORT void uform_arena_free(uform_arena_t arena);

/** Where an arena's outputs live, as the executor that made it decided. */
typedef enum uform_residency_t {

    /** No residency named. */
    uform_residency_unknown_k = 0,

    /** Host pages, which the host reads directly. */
    uform_residency_ram_k = 1,

    /** One GPU's memory, which only device code reads. */
    uform_residency_vram_k = 2,
} uform_residency_t;

/** Where an arena's outputs live. */
typedef struct uform_arena_description_t {

    /** RAM for host pages, VRAM for one device's memory. */
    uform_residency_t residency;

    /** The UForm ordinal of the device, as @c uform_gpu_devices lists it, or -1 for host pages. */
    int32_t device;
} uform_arena_description_t;

/** The arguments of @c uform_arena_describe. */
typedef struct uform_arena_describe_t {

    /** The arena to describe. */
    uform_arena_t arena;

    /** Output for the outcome, final once its status leaves @c uform_status_deferred_k. */
    uform_outcome_t *outcome;

    /** Output for where the arena's outputs live. */
    uform_arena_description_t *description;
} uform_arena_describe_t;

/**
 *  @brief Reports where an arena's outputs live, as the executor that made it decided, so a caller
 *      reads them as host memory or hands them to device code without remembering that executor.
 *
 *  @par Statuses:
 *  - @c uform_status_invalid_argument_k A @c NULL arena or description.
 */
UFORM_EXPORT void uform_arena_describe(uform_arena_describe_t *arguments);

/**
 *  @brief The canonical lowercase, dash-separated spelling of each closed set's enumerators, and the inverse
 *      that reads one back.
 *
 *  The names are static storage and are never deallocated. A value no status carries spells "unrecognized".
 */
UFORM_EXPORT char const* uform_status_name(uform_status_t status);

/** Reads a status back; @c uform_status_invalid_argument_k when unrecognized. */
UFORM_EXPORT uform_status_t uform_status_from_name(uform_bytes_view_t name);

/** The canonical spelling of a scalar kind, as @c uform_scalar_f32_k spells "f32". */
UFORM_EXPORT char const* uform_scalar_name(uform_scalar_t scalar);

/** Reads a scalar kind back; @c uform_scalar_unknown_k when unrecognized. */
UFORM_EXPORT uform_scalar_t uform_scalar_from_name(uform_bytes_view_t name);

/** The canonical spelling of a mood, as @c uform_mood_parked_k spells "parked". */
UFORM_EXPORT char const* uform_mood_name(uform_mood_t mood);

/** Reads a mood back; @c uform_mood_parked_k when unrecognized. */
UFORM_EXPORT uform_mood_t uform_mood_from_name(uform_bytes_view_t name);

/** The canonical spelling of a CPU class, as @c uform_cpu_class_intel_haswell_k spells
 *  "intel-haswell". */
UFORM_EXPORT char const* uform_cpu_class_name(uform_cpu_class_t cpu_class);

/** Reads a CPU class back; @c uform_cpu_class_unknown_k when unrecognized. */
UFORM_EXPORT uform_cpu_class_t uform_cpu_class_from_name(uform_bytes_view_t name);

/** The canonical spelling of a GPU class, as @c uform_gpu_class_nvidia_hopper_k spells
 *  "nvidia-hopper". */
UFORM_EXPORT char const* uform_gpu_class_name(uform_gpu_class_t gpu_class);

/** Reads a GPU class back; @c uform_gpu_class_unknown_k when unrecognized. */
UFORM_EXPORT uform_gpu_class_t uform_gpu_class_from_name(uform_bytes_view_t name);

/**
 *  @brief Reports the version of the library this process linked, which may differ from the
 *      @c UFORM_VERSION_MAJOR family of macros it was compiled against.
 *  @param[out] major The major version, which changes whenever an argument structure does.
 *  @param[out] minor The minor version.
 *  @param[out] patch The patch version.
 */
UFORM_EXPORT void uform_version(int* major, int* minor, int* patch);

/** The version of the library as a static null-terminated string. */
UFORM_EXPORT char const* uform_version_name(void);

/** The hardware capabilities this binary was compiled with, as a comma-separated static string. */
UFORM_EXPORT char const* uform_capabilities_compiled(void);

/** The hardware capabilities the current machine offers at runtime, as a comma-separated static string. */
UFORM_EXPORT char const* uform_capabilities_available(void);

#ifdef __cplusplus
}
#endif

#endif // UNUM_UFORM_TYPES_H
