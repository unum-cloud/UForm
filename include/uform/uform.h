/**
 *  @brief The C binary interface of UForm: load an encoder onto a GPU, then turn images, video, audio and
 *      text into embeddings, each call reporting through one outcome that a caller waits on or polls.
 *  @file include/uform/uform.h
 *  @author Ash Vardanian
 *  @date October 1, 2026
 *
 *  Two handles split the bytes by owner. A @c uform_model_t owns the weights, their packed copies and the cuDNN
 *  session on one device. A @c uform_executor_t owns the streams and the activation scratch, grown to its
 *  high-water mark and rewound every call. Embeddings, whose size @c count bounds, land in the caller's own
 *  memory, so no call allocates them, and no allocator, alignment or memory kind crosses this interface.
 *
 *  Every verb with options takes one structure of named arguments, so callers use designated initializers
 *  and skip what they don't need. The structures carry no size or version field: growing one changes the
 *  major version. The library never logs.
 *
 *  Synchronous and asynchronous calls share one shape. Every call on an executor carries a
 *  @c uform_completion_t : zero blocks until the outcome is final, and @c uform_completion_deferred_k returns
 *  once the call is queued, so a thread pool, a GPU stream or a remote client can serve the same verbs without
 *  changing a call site. The outcome's status is the one word both sides synchronize on:
 *
 *  - For a deferred call the library stores @c uform_status_pending_k , then the message and the final status
 *    with release ordering, from a host callback that still fires after a device fault, and touches nothing
 *    the structure names after that store. A blocking call synchronizes the stream instead.
 *  - The caller reads the status with acquire ordering, by waiting or by polling, and may reuse the
 *    structure or free its buffers only once the status is final.
 *  - Calls on one executor run in submission order, and a deferred submission that finds the in-flight table
 *    full finishes at once with @c uform_status_contended_k instead of queueing.
 *
 *  A call sequence, encoding images on one GPU:
 *
 *      uform_model_t model;
 *      uform_executor_t executor;
 *      uform_outcome_t outcome;
 *      uform_model_load_vram(&(uform_model_load_vram_t){
 *          .model = &model, .outcome = &outcome, .path = {path, path_length}, .device = 0});
 *      uform_executor_for_vram(&(uform_executor_for_vram_t){.executor = &executor, .outcome = &outcome, .device = 0});
 *
 *      uform_model_description_t description;
 *      uform_model_describe(&(uform_model_describe_t){
 *          .model = model, .executor = executor, .outcome = &outcome, .description = &description});
 *
 *      uform_outcome_t encoded;
 *      uform_encode(&(uform_encode_t){.model = model, .executor = executor, .outcome = &encoded,
 *          .completion = uform_completion_deferred_k, .count = n, .inputs = {uform_modality_image_k, blobs, offsets},
 *          .outputs = {embeddings, description.dimensions * 2, n, uform_scalar_bf16_k}});
 *      // ... other work, while the images encode ...
 *      uform_outcome_wait(&encoded);
 *
 *      uform_executor_free(executor);
 *      uform_model_free(model);
 *
 *  Training stays in C++ ( @c cuda_core/encoder_tower.hpp ): a backward reuses the stashes its paired
 *  forward left inside the tower, and a step also needs the loss, the optimizer and the collectives, none of
 *  which this interface can state yet.
 */
#pragma once

#include <stdbool.h> // `bool`
#include <stddef.h>  // `size_t`
#include <stdint.h>  // `uint64_t`, `int32_t`

#ifdef __cplusplus
extern "C" {
#endif

/** @brief Marks the functions UForm exports: imported from the DLL on Windows unless building it, and given
 *      default visibility on ELF and Mach-O. Define it empty before including this header to link statically. */
#ifndef UFORM_EXPORT
#if defined(_WIN32) || defined(__CYGWIN__)
#if defined(UFORM_BUILDING_LIBRARY)
#define UFORM_EXPORT __declspec(dllexport)
#else
#define UFORM_EXPORT __declspec(dllimport)
#endif
#elif defined(__GNUC__)
#define UFORM_EXPORT __attribute__((visibility("default")))
#else
#define UFORM_EXPORT
#endif
#endif

/** @brief A model on one GPU: its weights, their packed copies and the cuDNN session that runs its attention.
 *      Shared by every executor on its device; calls through different executors take turns on its session. */
typedef struct uform_model_handle_t *uform_model_t;

/** @brief The streams and activation scratch that calls run on, in submission order, on one GPU. */
typedef struct uform_executor_handle_t *uform_executor_t;

/**
 *  @brief What went wrong, in UForm's own numbering. Each failure maps from the C++ @c status_t code
 *      named beside it:
 *
 *      C ABI                              C++ status_t
 *      uform_status_out_of_memory_k       bad_alloc_k            -10
 *      uform_status_invalid_argument_k    invalid_input_k        -11
 *      uform_status_unsupported_k         shape_unsupported_k    -12
 *      uform_status_device_k              cuda_error_k           -13
 *      uform_status_io_k                  io_error_k             -19
 *      uform_status_internal_k            other_k                -1
 *
 *  @c uform_status_malformed_k has no C++ twin yet: today a loader pads an undecodable item instead.
 *  @c uform_status_contended_k comes from the executor's in-flight table, which only this interface has.
 */
typedef enum uform_status_t {
    /** @brief The call completed; the message is @c NULL . */
    uform_status_success_k = 0,
    /** @brief The call was submitted and has not finished; never a final status. */
    uform_status_pending_k = 1,
    /** @brief Device or pinned memory could not be allocated, or the executor's scratch ceiling is too low. */
    uform_status_out_of_memory_k = 2,
    /** @brief An argument is null, out of range, or contradicts another, like a zero output stride. */
    uform_status_invalid_argument_k = 3,
    /** @brief Supplied bytes could not be decoded: a checkpoint, an image, or a clip. */
    uform_status_malformed_k = 4,
    /** @brief The model or this build lacks what the call asks for, like a modality it wasn't trained on. */
    uform_status_unsupported_k = 5,
    /** @brief A checkpoint file could not be opened or read. */
    uform_status_io_k = 6,
    /** @brief The driver refused or the device faulted; after a fault the executor refuses every later call. */
    uform_status_device_k = 7,
    /** @brief A failure no other status names, which points at a defect. */
    uform_status_internal_k = 8,
    /** @brief Every slot of the executor's in-flight table is taken, or a blocking call came from inside a
     *      callback of the executor it would wait on; the call was refused, not queued. */
    uform_status_contended_k = 9,
} uform_status_t;

/** @brief Outcome of one call: a status to branch on, and a static message that is never freed and is @c NULL
 *      exactly when the status is @c uform_status_success_k . */
typedef struct uform_outcome_t {
    uform_status_t status;
    char const *message;
} uform_outcome_t;

/**
 *  @brief Blocks until @p outcome is final, without spinning, and with acquire ordering.
 *  @param[in] outcome An outcome passed to a call, read until its status leaves @c uform_status_pending_k .
 *  @return The final status; @c uform_status_pending_k at once when called from inside a callback of the executor
 *      running that call, rather than waiting on itself.
 */
UFORM_EXPORT uform_status_t uform_outcome_wait(uform_outcome_t const *outcome);

/** @brief Whether a call returns once its work is done, or once it is queued on the executor. */
typedef enum uform_completion_t {
    /** @brief Return with the final status stored and every output readable; the default. */
    uform_completion_blocking_k = 0,
    /** @brief Return once queued, with the status pending until the call ends; the caller waits or polls. */
    uform_completion_deferred_k = 1,
} uform_completion_t;

/** @brief A borrowed run of bytes, never null-terminated by contract. */
typedef struct uform_bytes_view_t {
    void const *data;
    size_t length;
} uform_bytes_view_t;

/** @brief How an executor waits whenever it next goes idle. */
typedef enum uform_mood_t {
    /** @brief Sleep on events created with blocking sync, freeing the core. */
    uform_mood_parked_k = 0,
    /** @brief Poll the events, trading a core for the lowest wake-up latency. */
    uform_mood_spinning_k = 1,
} uform_mood_t;

/** @brief What the bytes of an input tape hold, one item per tape row. */
typedef enum uform_modality_t {
    /** @brief UTF-8 text, tokenized byte by byte; text longer than the model's context is truncated. */
    uform_modality_text_k = 0,
    /** @brief Encoded images: JPEG decodes on the device, PNG and WebP on the host. */
    uform_modality_image_k = 1,
    /** @brief Video containers, MP4 or WebM, demuxed on the host and decoded by NVDEC. */
    uform_modality_video_k = 2,
    /** @brief Encoded audio clips, decoded and resampled to mono on the host. */
    uform_modality_audio_k = 3,
} uform_modality_t;

/** @brief The element type of an output span. Zero is no type, so a span left unset is refused rather than
 *      written at the wrong width. */
typedef enum uform_scalar_t {
    uform_scalar_f32_k = 1,
    uform_scalar_f16_k = 2,
    uform_scalar_bf16_k = 3,
} uform_scalar_t;

/** @brief The precision the transformer blocks run in. */
typedef enum uform_precision_t {
    /** @brief BFloat16 throughout, as the checkpoint stores it. */
    uform_precision_bf16_k = 0,
    /** @brief E4M3 blocks quantized at load, with BFloat16 entries, attention pool and head. */
    uform_precision_e4m3_k = 1,
} uform_precision_t;

/**
 *  @brief Items of one modality as an Arrow tape: item @c i is the bytes @c [offsets[i],offsets[i+1]) of
 *      @c data .
 *
 *  Device bytes on the executor's device are read in place; host bytes are classified from the pointer and
 *  uploaded, through the pinned bounce buffer when pageable.
 */
typedef struct uform_tape_view_t {
    uform_modality_t modality;
    void const *data;
    /** @brief @c count+1 offsets, always in host memory, since they drive host-side dispatch. */
    uint64_t const *offsets;
} uform_tape_view_t;

/**
 *  @brief Rows the call writes, one embedding of @c uform_model_description_t::dimensions scalars per input
 *      item, as the projection head emits them, without normalization.
 *
 *  The memory is classified from the pointer. Device memory on the executor's device, pinned and managed
 *  memory are written directly; pageable memory is written from the bounce buffer before the call's final
 *  status is stored.
 */
typedef struct uform_embeddings_span_t {
    void *data;
    /** @brief Bytes between consecutive rows; zero is refused, as every row would land in one place. */
    size_t stride;
    /** @brief Rows @c data holds; fewer than the call's @c count is refused. */
    size_t capacity;
    uform_scalar_t scalar;
} uform_embeddings_span_t;

/** @brief Arguments of @c uform_model_load_vram . */
typedef struct uform_model_load_vram_t {
    /** @brief Output for the new model. */
    uform_model_t *model;
    /** @brief Output for the outcome, final once its status leaves @c uform_status_pending_k . */
    uform_outcome_t *outcome;
    /** @brief Path to a safetensors checkpoint, read in full and closed before the load returns. */
    uform_bytes_view_t path;
    /** @brief The CUDA ordinal the weights and sessions live on. */
    int32_t device;
    uform_precision_t precision;
} uform_model_load_vram_t;

/** @brief Copies a checkpoint file onto one GPU and plans its products there, synchronously. The checkpoint
 *      carries the model's geometry. */
UFORM_EXPORT void uform_model_load_vram(uform_model_load_vram_t *arguments);

/** @brief Arguments of @c uform_model_load_vram_buffer . */
typedef struct uform_model_load_vram_buffer_t {
    /** @brief Output for the new model. */
    uform_model_t *model;
    /** @brief Output for the outcome, final once its status leaves @c uform_status_pending_k . */
    uform_outcome_t *outcome;
    /** @brief The safetensors checkpoint itself, in host or device memory, copied before the load returns. */
    uform_bytes_view_t buffer;
    /** @brief The CUDA ordinal the weights and sessions live on. */
    int32_t device;
    uform_precision_t precision;
} uform_model_load_vram_buffer_t;

/** @brief Copies a checkpoint held in memory onto one GPU and plans its products there, synchronously;
 *      nothing of @c buffer is kept once the load returns. */
UFORM_EXPORT void uform_model_load_vram_buffer(uform_model_load_vram_buffer_t *arguments);

/** @brief Arguments of @c uform_executor_for_vram . */
typedef struct uform_executor_for_vram_t {
    /** @brief Output for the new executor. */
    uform_executor_t *executor;
    /** @brief Output for the outcome, final once its status leaves @c uform_status_pending_k . */
    uform_outcome_t *outcome;
    /** @brief The CUDA ordinal the streams and scratch live on; borrowed streams must belong to it. */
    int32_t device;
    /** @brief Optional @c CUstream s to borrow until the executor is freed; without any, it creates its own. */
    void *const *streams;
    size_t streams_count;
    /** @brief Optional ceiling on the activation scratch, rewound every call; zero lets it grow as batches ask.
     *      A batch larger than the ceiling fits runs in several passes. */
    size_t scratch_limit;
    /** @brief Slots of the fixed in-flight table, one per deferred call not yet final. A deferred call finding
     *      it full stores @c uform_status_contended_k as its final status. Zero means a default. */
    size_t in_flight_limit;
    /** @brief Optional ForkUnion @c fu_pool_t to borrow for the host share of decoding, keeping its lender's
     *      mood; without one, that share runs on the executor's own host thread alone. */
    void *pool;
} uform_executor_for_vram_t;

/**
 *  @brief Creates an executor on one GPU, synchronously.
 *
 *  Besides its streams, an executor keeps one host thread for the share of decoding that needs the host:
 *  container demux, PNG and WebP, and audio. Text and JPEG never leave the device.
 */
UFORM_EXPORT void uform_executor_for_vram(uform_executor_for_vram_t *arguments);

/** @brief Arguments of @c uform_executor_drain . */
typedef struct uform_executor_drain_t {
    uform_executor_t executor;
    /** @brief Output for an executor-level failure, such as a device fault; never one of a drained call. */
    uform_outcome_t *outcome;
    /** @brief Applied whenever the executor next goes idle; the last drain wins. Never changes the
     *      process-wide device flags. */
    uform_mood_t mood;
} uform_executor_drain_t;

/** @brief Waits for every call submitted before the drain began, so later submissions can't hold it up.
 *      Reports only executor-level failures, like a device fault; each call reports its own. */
UFORM_EXPORT void uform_executor_drain(uform_executor_drain_t *arguments);

/** @brief What a loaded model reports about itself. */
typedef struct uform_model_description_t {
    /** @brief Scalars per embedding row. */
    size_t dimensions;
    /** @brief The precision the transformer blocks run in. */
    uform_precision_t precision;
    /** @brief The CUDA ordinal the weights live on. */
    int32_t device;
} uform_model_description_t;

/** @brief Arguments of @c uform_model_describe . */
typedef struct uform_model_describe_t {
    uform_model_t model;
    /** @brief The executor the report runs on. */
    uform_executor_t executor;
    /** @brief Output for the outcome, final once its status leaves @c uform_status_pending_k . */
    uform_outcome_t *outcome;
    /** @brief Whether the call blocks or returns once queued; zero blocks. */
    uform_completion_t completion;
    /** @brief Output, filled whole. */
    uform_model_description_t *description;
} uform_model_describe_t;

/** @brief Reports a model's geometry in one call. */
UFORM_EXPORT void uform_model_describe(uform_model_describe_t *arguments);

/**
 *  @brief Arguments of @c uform_encode . A deferred call borrows this structure, and everything it names,
 *      until its outcome is final, so it must stay alive until then.
 */
typedef struct uform_encode_t {
    uform_model_t model;
    /** @brief The executor whose streams run the call; it must sit on the model's device. */
    uform_executor_t executor;
    /** @brief Output for the outcome, final once its status leaves @c uform_status_pending_k . */
    uform_outcome_t *outcome;
    /** @brief Whether the call blocks or returns once queued; zero blocks. */
    uform_completion_t completion;
    /** @brief Items in @c inputs , and rows written to @c outputs . */
    size_t count;
    uform_tape_view_t inputs;
    uform_embeddings_span_t outputs;
} uform_encode_t;

/** @brief Encodes @c count items of one modality into embeddings. A call refused at submission stores its
 *      final status before it returns. */
UFORM_EXPORT void uform_encode(uform_encode_t *arguments);

/** @brief Frees a model, safe on @c NULL . Every call using it must have completed. */
UFORM_EXPORT void uform_model_free(uform_model_t model);

/** @brief Drains, then frees an executor and returns its borrowed streams and pool, safe on @c NULL . */
UFORM_EXPORT void uform_executor_free(uform_executor_t executor);

/** @brief The canonical spelling of @p status , like @c "out_of_memory" , as a static string; @c NULL for a
 *      value outside the enumeration. */
UFORM_EXPORT char const *uform_status_name(uform_status_t status);

/**
 *  @brief Reads a status back from its canonical spelling.
 *  @param[in] name The spelling, not null-terminated.
 *  @param[out] status The status it names, left untouched when it names none.
 *  @return Whether @p name names a status.
 */
UFORM_EXPORT bool uform_status_from_name(uform_bytes_view_t name, uform_status_t *status);

/** @brief Reports the version of the linked library. */
UFORM_EXPORT void uform_version(int *major, int *minor, int *patch);

#ifdef __cplusplus
} // extern "C"
#endif
