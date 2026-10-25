/**
 *  @file c/uform/model.h
 *  @author Ash Vardanian
 *  @date October 8, 2026
 *  @brief The model lifecycle of the UForm C interface: load an encoder into RAM or VRAM, describe it, and
 *      free it.
 *
 *  A @c uform_model_t owns the weights and their packed copies, and on a GPU the session its attention runs
 *  on. Residency is in the verb name, as in USearch and UStore - @c uform_model_load_vram places the weights
 *  on the devices its @c vram options name, and @c uform_model_load_ram in host memory - so a residency a
 *  build lacks has no verb to call. One model serves every modality it was trained on; every verb of
 *  @c encode.h takes it. A model only infers: training runs on a @c uform_trainer_t of @c train.h, which
 *  saves the checkpoints these verbs load.
 *
 *  A call sequence, encoding images on one GPU:
 *
 *  @code{.c}
 *  int32_t const device = 0;
 *  uform_vram_options_t const vram = {.devices = &device, .devices_count = 1};
 *  uform_model_t model = NULL;
 *  uform_executor_t executor = NULL;
 *  uform_arena_t arena = NULL;
 *  uform_outcome_t outcome = {0};
 *  uform_executor_for_vram(&(uform_executor_for_vram_t){.executor = &executor, .outcome = &outcome, .vram = vram});
 *  uform_arena_init(&(uform_arena_init_t){.arena = &arena, .executor = executor, .outcome = &outcome});
 *  uform_model_load_vram(&(uform_model_load_vram_t){
 *      .model = &model, .executor = executor, .outcome = &outcome, .path = {path, path_length}, .vram = vram,
 *      .quantization = uform_scalar_bf16_k});
 *
 *  uform_outcome_t encoded = {0};
 *  uform_rows_view_t embeddings = {0};
 *  uform_image_encode(&(uform_image_encode_t){
 *      .model = model, .executor = executor, .outcome = &encoded, .completion = uform_completion_deferred_k,
 *      .count = n, .images = {files, offsets, uform_scalar_u8_k}, .embeddings = &embeddings, .arena = arena,
 *      .outputs_scalar = uform_scalar_bf16_k});
 *  // ... other work, while the images encode ...
 *  uform_outcome_wait(&encoded);
 *
 *  uform_arena_free(arena);
 *  uform_model_free(model);
 *  uform_executor_free(executor);
 *  @endcode
 */
#ifndef UNUM_UFORM_MODEL_H
#define UNUM_UFORM_MODEL_H

#include "uform/types.h"

#ifdef __cplusplus
extern "C" {
#endif

/** A loaded encoder: its weights, their packed copies, and on a GPU the session its attention runs on.
 *  Shared by every executor of its residency; calls through different executors take turns on its session. */
typedef struct uform_model_handle_t* uform_model_t;

/** The arguments of @c uform_model_load_ram. */
typedef struct uform_model_load_ram_t {

    /** Output for the new model, written once the call completes. */
    uform_model_t* model;

    /** The executor the load runs on. */
    uform_executor_t executor;

    /** Output for the outcome, final once its status leaves @c uform_status_deferred_k. */
    uform_outcome_t* outcome;

    /** Whether the call blocks or returns once queued; zero blocks. */
    uform_completion_t completion;

    /** Path to a safetensors checkpoint, read in full and closed before the load completes. */
    uform_bytes_view_t path;

    /** The memory domains to place the weights in. */
    uform_ram_options_t ram;

    /** The scalar the transformer blocks run in: @c uform_scalar_bf16_k as the checkpoint stores it, or
     *  @c uform_scalar_e4m3_k quantized at load, with BFloat16 entries, attention pool and head. */
    uform_scalar_t quantization;

    /** The @b optional handler, called with tensors placed. */
    uform_progress_t progress;
} uform_model_load_ram_t;

/** Copies a checkpoint file into host memory and packs its products for the CPU. The checkpoint carries the
 *  model's geometry. */
UFORM_EXPORT void uform_model_load_ram(uform_model_load_ram_t* arguments);

/** The arguments of @c uform_model_load_ram_buffer. */
typedef struct uform_model_load_ram_buffer_t {

    /** Output for the new model, written once the call completes. */
    uform_model_t* model;

    /** The executor the load runs on. */
    uform_executor_t executor;

    /** Output for the outcome, final once its status leaves @c uform_status_deferred_k. */
    uform_outcome_t* outcome;

    /** Whether the call blocks or returns once queued; zero blocks. */
    uform_completion_t completion;

    /** The safetensors checkpoint itself, copied before the load completes; nothing of it is kept. */
    uform_bytes_view_t buffer;

    /** The memory domains to place the weights in. */
    uform_ram_options_t ram;

    /** The scalar the transformer blocks run in, as for @c uform_model_load_ram. */
    uform_scalar_t quantization;

    /** The @b optional handler, called with tensors placed. */
    uform_progress_t progress;
} uform_model_load_ram_buffer_t;

/** Copies a checkpoint held in memory into host memory and packs its products for the CPU. */
UFORM_EXPORT void uform_model_load_ram_buffer(uform_model_load_ram_buffer_t* arguments);

/** The arguments of @c uform_model_load_vram. */
typedef struct uform_model_load_vram_t {

    /** Output for the new model, written once the call completes. */
    uform_model_t* model;

    /** The executor the upload runs on, one stream per device. */
    uform_executor_t executor;

    /** Output for the outcome, final once its status leaves @c uform_status_deferred_k. */
    uform_outcome_t* outcome;

    /** Whether the call blocks or returns once queued; zero blocks. */
    uform_completion_t completion;

    /** Path to a safetensors checkpoint, read in full and closed before the load completes. */
    uform_bytes_view_t path;

    /** The devices to place the weights on, replicated on each, since a model is small next to the
     *  activations it serves. */
    uform_vram_options_t vram;

    /** The scalar the transformer blocks run in, as for @c uform_model_load_ram. */
    uform_scalar_t quantization;

    /** The @b optional handler, called with tensors placed. */
    uform_progress_t progress;
} uform_model_load_vram_t;

/** Copies a checkpoint file onto the named GPUs and packs its products there. */
UFORM_EXPORT void uform_model_load_vram(uform_model_load_vram_t* arguments);

/** The arguments of @c uform_model_load_vram_buffer. */
typedef struct uform_model_load_vram_buffer_t {

    /** Output for the new model, written once the call completes. */
    uform_model_t* model;

    /** The executor the upload runs on, one stream per device. */
    uform_executor_t executor;

    /** Output for the outcome, final once its status leaves @c uform_status_deferred_k. */
    uform_outcome_t* outcome;

    /** Whether the call blocks or returns once queued; zero blocks. */
    uform_completion_t completion;

    /** The safetensors checkpoint itself, in host or device memory, copied before the load completes. */
    uform_bytes_view_t buffer;

    /** The devices to place the weights on, replicated on each. */
    uform_vram_options_t vram;

    /** The scalar the transformer blocks run in, as for @c uform_model_load_ram. */
    uform_scalar_t quantization;

    /** The @b optional handler, called with tensors placed. */
    uform_progress_t progress;
} uform_model_load_vram_buffer_t;

/** Copies a checkpoint held in memory onto the named GPUs and packs its products there. */
UFORM_EXPORT void uform_model_load_vram_buffer(uform_model_load_vram_buffer_t* arguments);

/** What a loaded model reports about itself. */
typedef struct uform_model_description_t {

    /** Scalars per embedding row; Matryoshka prefixes of it are embeddings too. */
    size_t dimensions;

    /** Tokens one item may span before it is truncated: bytes of text, patches of an image, tubes of a
     *  clip, frames of a recording. */
    size_t context;

    /** The scalar the transformer blocks run in. */
    uform_scalar_t quantization;

    /** Bytes the model holds; activations are the executor's. */
    size_t memory_usage;

    /** Static codename of the kernels the model dispatched to on this machine. */
    char const* capabilities;

    /** The class a RAM model's tower was packed for; @c uform_cpu_class_unknown_k for a VRAM
     *  model. */
    uform_cpu_class_t cpu_class;

    /** The class every replica of a VRAM model runs as; @c uform_gpu_class_unknown_k for a RAM
     *  model. */
    uform_gpu_class_t gpu_class;

    /** Scalars per token vector; zero when the checkpoint carries no token head. */
    size_t token_width;

    /** Question types the choice temperatures were calibrated over: zero without the byte
     *  decoder, one when uncalibrated. */
    size_t choice_temperature_groups;
} uform_model_description_t;

/** The arguments of @c uform_model_describe. */
typedef struct uform_model_describe_t {

    /** The model to be described. */
    uform_model_t model;

    /** The executor the report runs on. */
    uform_executor_t executor;

    /** Output for the outcome, final once its status leaves @c uform_status_deferred_k. */
    uform_outcome_t* outcome;

    /** Whether the call blocks or returns once queued; zero blocks. */
    uform_completion_t completion;

    /** Output, filled whole. */
    uform_model_description_t* description;
} uform_model_describe_t;

/** Reports a model's geometry and placement in one call. */
UFORM_EXPORT void uform_model_describe(uform_model_describe_t* arguments);

/** Frees a model; @c NULL is safe, and it cannot fail, so takes no outcome. Every call using it must be
 *  final. */
UFORM_EXPORT void uform_model_free(uform_model_t model);

#ifdef __cplusplus
}
#endif

#endif // UNUM_UFORM_MODEL_H
