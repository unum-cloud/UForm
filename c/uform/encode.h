/**
 *  @file c/uform/encode.h
 *  @author Ash Vardanian
 *  @date October 17, 2026
 *  @brief The forward passes of the UForm C interface: text, images, video and audio in, pooled
 *      embeddings and per-token states out, one verb per input form.
 *
 *  Every verb runs the same tower - the modality's entry, the transformer blocks, and the final
 *  norm - and keeps whichever surfaces the caller names, from one forward:
 *
 *      Output        Rows                           Read past the final norm
 *      embeddings    one per item                   the attention pool and the projection head
 *      states        one per token left per item    nothing; the surface rerankers and masked heads read
 *
 *  A @c NULL output skips its surface, and naming neither is refused with
 *  @c uform_status_invalid_argument_k. Both land in the call's arena, written as @c outputs_scalar,
 *  and the embeddings come without normalization. An item longer than the model's @c context is
 *  truncated, and the call's stats count how many were.
 *
 *  Encoded files and decoded signals are separate verbs rather than one verb with a layout tag, so
 *  each takes only the fields its input needs. Models are tokenizer-free: text arrives as UTF-8
 *  bytes and every byte is a token. JPEG decodes on the device; PNG, WebP, container demux and
 *  audio on the executor's host threads.
 *
 *  @code{.c}
 *  uform_rows_view_t embeddings = {0};
 *  uform_rows_tape_view_t states = {0};
 *  uform_text_encode(&(uform_text_encode_t){
 *      .model = model, .executor = executor, .outcome = &outcome, .count = n,
 *      .texts = {bytes, offsets, uform_scalar_u8_k}, .embeddings = &embeddings, .states = &states,
 *      .arena = arena, .outputs_scalar = uform_scalar_bf16_k});
 *  // `embeddings.data` holds `n` rows, and item `i` of `states` spans its rows `states.offsets[i]` up to
 *  // `states.offsets[i + 1]`, both valid until the next call given `arena`.
 *  @endcode
 */
#ifndef UNUM_UFORM_ENCODE_H
#define UNUM_UFORM_ENCODE_H

#include "uform/model.h"

#ifdef __cplusplus
extern "C" {
#endif

/** What a text call reports beside its outputs. */
typedef struct uform_text_encode_stats_t {

    /** Items cut at the model's @c context. */
    size_t truncated;

    /** Tokens the call encoded, after truncation. */
    size_t tokens;
} uform_text_encode_stats_t;

/** The arguments of @c uform_text_encode. */
typedef struct uform_text_encode_t {

    /** The model to encode with. */
    uform_model_t model;

    /** The executor the call runs on; it must share the model's residency. */
    uform_executor_t executor;

    /** Output for the outcome, final once its status leaves @c uform_status_deferred_k. */
    uform_outcome_t* outcome;

    /** Whether the call blocks or returns once queued; zero blocks. */
    uform_completion_t completion;

    /** Items in @c texts, and the batch axis of every output. */
    size_t count;

    /** UTF-8 texts, one item per entry, in bytes. */
    uform_tape_view_t texts;

    /** The @b optional output for one pooled embedding of the model's @c dimensions per item,
     *  inside @c arena; @c NULL skips the pool and the head. */
    uform_rows_view_t* embeddings;

    /** The @b optional output for the final-norm states, one row of the model's @c dimensions per
     *  token left after truncation, inside @c arena; @c NULL skips them. */
    uform_rows_tape_view_t* states;

    /** The @b optional output for the token head's unit vectors, one row of the model's
     *  @c token_width per token left after truncation, the prefix included, inside @c arena; a
     *  model without a token head reports @c uform_status_unsupported_k. */
    uform_rows_tape_view_t* token_vectors;

    /** Memory for every output, rewound by the call. */
    uform_arena_t arena;

    /** The scalar type every output lands as. */
    uform_scalar_t outputs_scalar;

    /** The @b optional handler, called with items encoded. */
    uform_progress_t progress;

    /** The @b optional report; @c NULL skips it. */
    uform_text_encode_stats_t* stats;
} uform_text_encode_t;

/** Encodes @c count texts through one forward, keeping the surfaces the caller names. */
UFORM_EXPORT void uform_text_encode(uform_text_encode_t* arguments);

/** The arguments of @c uform_text_choose. */
typedef struct uform_text_choose_t {

    /** The model to choose with; it must carry the byte decoder, its
     *  @c choice_temperature_groups nonzero. */
    uform_model_t model;

    /** The executor the call runs on; it must share the model's residency. */
    uform_executor_t executor;

    /** Output for the outcome, final once its status leaves @c uform_status_deferred_k. */
    uform_outcome_t* outcome;

    /** Whether the call blocks or returns once queued; zero blocks. */
    uform_completion_t completion;

    /** Items in @c texts, and the batch axis of every input and output. */
    size_t count;

    /** UTF-8 states, one per item, each ending where the choice goes; the call appends the mask
     *  byte, and a state longer than the model's context loses its head, never its end. */
    uform_tape_view_t texts;

    /** Each item's legal choices, one byte each, @c uform_scalar_u8_k; an item with
     *  none chooses nothing. */
    uform_tape_view_t choices;

    /** The @b optional question type of each item, @c count of them, below the model's
     *  @c choice_temperature_groups; @c NULL puts every item in group zero. */
    uint32_t const* temperature_groups;

    /** Output for one probability per choice in @c choices order, @c uform_scalar_f32_k, inside
     *  @c arena: the shape a training step reads its @c choice_targets in. */
    uform_tape_view_t* probabilities;

    /** Memory for every output, rewound by the call. */
    uform_arena_t arena;

    /** The @b optional handler, called with items scored. */
    uform_progress_t progress;

    /** The @b optional report; @c NULL skips it. */
    uform_text_encode_stats_t* stats;
} uform_text_choose_t;

/** Scores each item's legal choices through one forward, a softmax over them alone at the item's
 *  group's calibrated inverse temperature; a model without the byte decoder reports
 *  @c uform_status_unsupported_k. */
UFORM_EXPORT void uform_text_choose(uform_text_choose_t* arguments);

/** What an image call reports beside its outputs. */
typedef struct uform_image_encode_stats_t {

    /** Items whose patch grid was cut at the model's @c context. */
    size_t truncated;

    /** Patches the call encoded, after truncation. */
    size_t tokens;
} uform_image_encode_stats_t;

/** The arguments of @c uform_image_encode. */
typedef struct uform_image_encode_t {

    /** The model to encode with. */
    uform_model_t model;

    /** The executor the call runs on; it must share the model's residency. */
    uform_executor_t executor;

    /** Output for the outcome, final once its status leaves @c uform_status_deferred_k. */
    uform_outcome_t* outcome;

    /** Whether the call blocks or returns once queued; zero blocks. */
    uform_completion_t completion;

    /** Items in @c images, and the batch axis of every output. */
    size_t count;

    /** JPEG, PNG or WebP files, one item per entry, in bytes. */
    uform_tape_view_t images;

    /** The @b optional output for one pooled embedding per item, inside @c arena. */
    uform_rows_view_t* embeddings;

    /** The @b optional output for the final-norm states, one row per patch, inside @c arena. */
    uform_rows_tape_view_t* states;

    /** The @b optional output for the token head's unit vectors, one row per patch,
     *  inside @c arena. */
    uform_rows_tape_view_t* token_vectors;

    /** Memory for every output, rewound by the call. */
    uform_arena_t arena;

    /** The scalar type every output lands as. */
    uform_scalar_t outputs_scalar;

    /** The @b optional handler, called with items encoded. */
    uform_progress_t progress;

    /** The @b optional report; @c NULL skips it. */
    uform_image_encode_stats_t* stats;
} uform_image_encode_t;

/** Decodes and encodes @c count image files; an undecodable file fails the call with
 *  @c uform_status_malformed_k. */
UFORM_EXPORT void uform_image_encode(uform_image_encode_t* arguments);

/** The arguments of @c uform_image_encode_pixels. */
typedef struct uform_image_encode_pixels_t {

    /** The model to encode with. */
    uform_model_t model;

    /** The executor the call runs on; it must share the model's residency. */
    uform_executor_t executor;

    /** Output for the outcome, final once its status leaves @c uform_status_deferred_k. */
    uform_outcome_t* outcome;

    /** Whether the call blocks or returns once queued; zero blocks. */
    uform_completion_t completion;

    /** Items in @c pixels, and the batch axis of every output. */
    size_t count;

    /** Interleaved 8-bit RGB, one item per entry, in bytes: item @c i holds @c heights[i] rows of
     *  @c widths[i] pixels, three bytes each, rows end to end. */
    uform_tape_view_t pixels;

    /** Rows of each item, @c count of them in host memory. */
    uint32_t const* heights;

    /** Pixels per row of each item, @c count of them in host memory. */
    uint32_t const* widths;

    /** The @b optional output for one pooled embedding per item, inside @c arena. */
    uform_rows_view_t* embeddings;

    /** The @b optional output for the final-norm states, one row per patch, inside @c arena. */
    uform_rows_tape_view_t* states;

    /** The @b optional output for the token head's unit vectors, one row per patch,
     *  inside @c arena. */
    uform_rows_tape_view_t* token_vectors;

    /** Memory for every output, rewound by the call. */
    uform_arena_t arena;

    /** The scalar type every output lands as. */
    uform_scalar_t outputs_scalar;

    /** The @b optional handler, called with items encoded. */
    uform_progress_t progress;

    /** The @b optional report; @c NULL skips it. */
    uform_image_encode_stats_t* stats;
} uform_image_encode_pixels_t;

/** Encodes @c count decoded images, resizing on the executor's side; a tape entry whose length
 *  isn't @c heights[i] × @c widths[i] × 3 fails with @c uform_status_shape_mismatch_k. */
UFORM_EXPORT void uform_image_encode_pixels(uform_image_encode_pixels_t* arguments);

/** What a video call reports beside its outputs. */
typedef struct uform_video_encode_stats_t {

    /** Items cut at the model's @c context in tubes. */
    size_t truncated;

    /** Tubes the call encoded, after truncation. */
    size_t tokens;
} uform_video_encode_stats_t;

/** The arguments of @c uform_video_encode. */
typedef struct uform_video_encode_t {

    /** The model to encode with. */
    uform_model_t model;

    /** The executor the call runs on; it must share the model's residency. */
    uform_executor_t executor;

    /** Output for the outcome, final once its status leaves @c uform_status_deferred_k. */
    uform_outcome_t* outcome;

    /** Whether the call blocks or returns once queued; zero blocks. */
    uform_completion_t completion;

    /** Items in @c videos, and the batch axis of every output. */
    size_t count;

    /** MP4 or WebM files, one item per entry, in bytes, demuxed on the host and
     *  decoded by NVDEC. */
    uform_tape_view_t videos;

    /** The @b optional output for one pooled embedding per item, inside @c arena. */
    uform_rows_view_t* embeddings;

    /** The @b optional output for the final-norm states, one row per tube, inside @c arena. */
    uform_rows_tape_view_t* states;

    /** The @b optional output for the token head's unit vectors, one row per tube,
     *  inside @c arena. */
    uform_rows_tape_view_t* token_vectors;

    /** Memory for every output, rewound by the call. */
    uform_arena_t arena;

    /** The scalar type every output lands as. */
    uform_scalar_t outputs_scalar;

    /** The @b optional handler, called with items encoded. */
    uform_progress_t progress;

    /** The @b optional report; @c NULL skips it. */
    uform_video_encode_stats_t* stats;
} uform_video_encode_t;

/** Decodes and encodes @c count clips. */
UFORM_EXPORT void uform_video_encode(uform_video_encode_t* arguments);

/** What an audio call reports beside its outputs. */
typedef struct uform_audio_encode_stats_t {

    /** Items cut at the model's @c context in frames. */
    size_t truncated;

    /** Frames the call encoded, after truncation. */
    size_t tokens;
} uform_audio_encode_stats_t;

/** The arguments of @c uform_audio_encode. */
typedef struct uform_audio_encode_t {

    /** The model to encode with. */
    uform_model_t model;

    /** The executor the call runs on; it must share the model's residency. */
    uform_executor_t executor;

    /** Output for the outcome, final once its status leaves @c uform_status_deferred_k. */
    uform_outcome_t* outcome;

    /** Whether the call blocks or returns once queued; zero blocks. */
    uform_completion_t completion;

    /** Items in @c recordings, and the batch axis of every output. */
    size_t count;

    /** WAV, FLAC, MP3 or Opus files, one item per entry, in bytes, decoded and resampled to mono
     *  on the host. */
    uform_tape_view_t recordings;

    /** The @b optional output for one pooled embedding per item, inside @c arena. */
    uform_rows_view_t* embeddings;

    /** The @b optional output for the final-norm states, one row per frame, inside @c arena. */
    uform_rows_tape_view_t* states;

    /** The @b optional output for the token head's unit vectors, one row per frame,
     *  inside @c arena. */
    uform_rows_tape_view_t* token_vectors;

    /** Memory for every output, rewound by the call. */
    uform_arena_t arena;

    /** The scalar type every output lands as. */
    uform_scalar_t outputs_scalar;

    /** The @b optional handler, called with items encoded. */
    uform_progress_t progress;

    /** The @b optional report; @c NULL skips it. */
    uform_audio_encode_stats_t* stats;
} uform_audio_encode_t;

/** Decodes and encodes @c count recordings. */
UFORM_EXPORT void uform_audio_encode(uform_audio_encode_t* arguments);

/** The arguments of @c uform_audio_encode_samples. */
typedef struct uform_audio_encode_samples_t {

    /** The model to encode with. */
    uform_model_t model;

    /** The executor the call runs on; it must share the model's residency. */
    uform_executor_t executor;

    /** Output for the outcome, final once its status leaves @c uform_status_deferred_k. */
    uform_outcome_t* outcome;

    /** Whether the call blocks or returns once queued; zero blocks. */
    uform_completion_t completion;

    /** Items in @c samples, and the batch axis of every output. */
    size_t count;

    /** Mono samples, one item per entry, in samples, of @c uform_scalar_f32_k. */
    uform_tape_view_t samples;

    /** Samples per second shared by every item; resampled to the model's rate where it differs. */
    uint32_t sample_rate;

    /** The @b optional output for one pooled embedding per item, inside @c arena. */
    uform_rows_view_t* embeddings;

    /** The @b optional output for the final-norm states, one row per frame, inside @c arena. */
    uform_rows_tape_view_t* states;

    /** The @b optional output for the token head's unit vectors, one row per frame,
     *  inside @c arena. */
    uform_rows_tape_view_t* token_vectors;

    /** Memory for every output, rewound by the call. */
    uform_arena_t arena;

    /** The scalar type every output lands as. */
    uform_scalar_t outputs_scalar;

    /** The @b optional handler, called with items encoded. */
    uform_progress_t progress;

    /** The @b optional report; @c NULL skips it. */
    uform_audio_encode_stats_t* stats;
} uform_audio_encode_samples_t;

/** Encodes @c count decoded recordings. */
UFORM_EXPORT void uform_audio_encode_samples(uform_audio_encode_samples_t* arguments);

#ifdef __cplusplus
}
#endif

#endif // UNUM_UFORM_ENCODE_H
