/**
 *  @file javascript/encode.c
 *  @author Ash Vardanian
 *  @date October 17, 2026
 *  @brief The Node-API forward passes: text, images, video and audio in, pooled embeddings and per-token states
 *      viewed inside the caller's arena out.
 *
 *  Every verb takes the executor and the completion first, so one native function serves the calling thread, a
 *  blocking receiver and a deferred one alike, then its tape, then the arena, the surfaces it keeps, the scalar
 *  they land as, and the progress handler a call on the calling thread may take.
 */
#include "lib.h"

#pragma region Calls

/** An encode call: the shared header, then every output an encode verb lands. */
typedef struct uform_js_encode_call_t {

    /** The outcome, the answer builder and the held values. */
    uform_js_call_t call;

    /** The pooled embeddings, one row per item. */
    uform_rows_view_t embeddings;

    /** The final-norm states, one row per token kept. */
    uform_rows_tape_view_t states;

    /** The report every encode verb fills, whose four types share one layout. */
    union {
        uform_text_encode_stats_t text;
        uform_image_encode_stats_t image;
        uform_video_encode_stats_t video;
        uform_audio_encode_stats_t audio;
    } stats;

    /** Items on the batch axis. */
    size_t count;

    /** The surfaces the caller asked for, as @c uform_js_outputs_t flags. */
    uint32_t outputs;
} uform_js_encode_call_t;

/** What every encode verb reads before its own inputs. */
typedef struct uform_js_encode_arguments_t {
    uform_executor_t executor;
    uform_completion_t completion;
    uform_tape_view_t tape;
    uform_arena_t arena;
    uform_scalar_t outputs_scalar;
    uform_progress_t progress;
    uform_rows_view_t *embeddings;
    uform_rows_tape_view_t *states;
} uform_js_encode_arguments_t;

static napi_value uform_js_finish_encode(napi_env env, uform_js_call_t *call, napi_value const *slots) {
    uform_js_encode_call_t const *const encode = (uform_js_encode_call_t const *)call;
    napi_value const arena = slots[uform_js_slot_arena_k];
    napi_value object = NULL, stats = NULL, embeddings = NULL, states = NULL;
    static char const *const stats_names[] = {"truncated", "tokens"};
    napi_value const stats_values[] = {uform_js_number(env, encode->stats.text.truncated),
                                       uform_js_number(env, encode->stats.text.tokens)};
    if (!(stats = uform_js_object(env, 2, stats_names, stats_values)) || napi_create_object(env, &object) != napi_ok ||
        napi_set_named_property(env, object, "stats", stats) != napi_ok) {
        uform_js_throw(env, uform_status_internal_k, "Could not publish the outputs");
        return NULL;
    }
    if (encode->outputs & uform_js_outputs_embeddings_k) {
        embeddings = uform_js_rows_object(env, arena, &encode->embeddings, encode->count);
        if (!embeddings || napi_set_named_property(env, object, "embeddings", embeddings) != napi_ok) return NULL;
    }
    if (encode->outputs & uform_js_outputs_states_k) {
        size_t const rows = encode->states.offsets ? encode->states.offsets[encode->count] : 0;
        bool const wide = sizeof(size_t) == 8;
        napi_value const offsets = uform_js_arena_view(env, arena, wide ? napi_biguint64_array : napi_uint32_array,
                                                       encode->states.offsets, (encode->count + 1) * sizeof(size_t),
                                                       encode->count + 1);
        states = offsets ? uform_js_rows_object(env, arena, &encode->states.rows, rows) : NULL;
        if (!states || napi_set_named_property(env, states, "offsets", offsets) != napi_ok ||
            napi_set_named_property(env, object, "states", states) != napi_ok)
            return NULL;
    }
    return object;
}

/**
 *  @brief Opens an encode call, reading the executor, the completion, the tape in arguments two and three, and the
 *      arena, the surfaces, the scalar and the progress handler from argument @p tail on.
 *  @return The call, or @c NULL with an exception pending.
 */
static uform_js_encode_call_t *uform_js_encode_open(napi_env env, napi_callback_info info, size_t wanted,
                                                    napi_value *arguments, napi_value *self,
                                                    uform_js_encode_call_t *local, napi_typedarray_type element_type,
                                                    size_t tail, uform_model_t *model,
                                                    uform_js_encode_arguments_t *parsed) {
    uform_js_model_t *const wrapped = (uform_js_model_t *)uform_js_self(env, info, wanted, arguments, self);
    if (wrapped && !wrapped->handle) uform_js_throw(env, uform_status_invalid_argument_k, "The model was freed");
    if (!wrapped || !wrapped->handle) return NULL;
    *model = wrapped->handle;
    uform_js_encode_call_t *const call = (uform_js_encode_call_t *)uform_js_call_open_on(
        env, arguments, &local->call, sizeof(*local), &parsed->executor, &parsed->completion);
    if (!call) return NULL;
    size_t outputs = 0;
    if (!uform_js_tape_from_value(env, arguments[2], arguments[3], "inputs", element_type, &call->call, &call->count,
                                  &parsed->tape) ||
        !uform_js_size_from_value(env, arguments[tail + 1], "outputs", &outputs) ||
        !uform_js_scalar_from_value(env, arguments[tail + 2], "outputsScalar", &parsed->outputs_scalar) ||
        !uform_js_progress_from_value(env, arguments[tail + 3], parsed->executor, &call->call, &parsed->progress) ||
        !uform_js_arena_from_value(env, arguments[tail], &call->call, &parsed->arena)) {
        uform_js_call_abandon(&call->call, &local->call);
        return NULL;
    }
    call->outputs = (uint32_t)outputs;
    call->call.finish = uform_js_finish_encode;
    parsed->embeddings = outputs & uform_js_outputs_embeddings_k ? &call->embeddings : NULL;
    parsed->states = outputs & uform_js_outputs_states_k ? &call->states : NULL;
    return call;
}

#pragma endregion Calls

#pragma region Verbs

/** Which encoded-file verb a shared body serves. */
typedef enum uform_js_files_t {
    uform_js_files_text_k,
    uform_js_files_image_k,
    uform_js_files_video_k,
    uform_js_files_audio_k,
} uform_js_files_t;

/** @c textEncode, @c imageEncode, @c videoEncode and @c audioEncode(executor, completion, bytes, offsets, arena,
 *  outputs, outputsScalar, progress) → @c {embeddings, states, stats}, viewed inside @p arena. */
static napi_value uform_js_encode_files(napi_env env, napi_callback_info info, uform_js_files_t verb) {
    napi_value arguments[8], self = NULL, deferred = NULL;
    uform_js_encode_call_t local;
    uform_js_encode_arguments_t parsed;
    uform_model_t model = NULL;
    uform_js_encode_call_t *const call = uform_js_encode_open(env, info, 8, arguments, &self, &local, napi_uint8_array,
                                                              4, &model, &parsed);
    if (!call) return NULL;
    napi_value const slots[uform_js_slots_k] = {self, arguments[0], arguments[4], arguments[2]};
    if (!uform_js_call_hold(env, &call->call, &local.call, slots, &deferred)) return NULL;
    uform_outcome_t *const outcome = &call->call.outcome;
    switch (verb) {
    case uform_js_files_text_k: {
        uform_text_encode_t encode = {.model = model,
                                      .executor = parsed.executor,
                                      .outcome = outcome,
                                      .completion = parsed.completion,
                                      .count = call->count,
                                      .texts = parsed.tape,
                                      .embeddings = parsed.embeddings,
                                      .states = parsed.states,
                                      .arena = parsed.arena,
                                      .outputs_scalar = parsed.outputs_scalar,
                                      .progress = parsed.progress,
                                      .stats = &call->stats.text};
        uform_text_encode(&encode);
    } break;
    case uform_js_files_image_k: {
        uform_image_encode_t encode = {.model = model,
                                       .executor = parsed.executor,
                                       .outcome = outcome,
                                       .completion = parsed.completion,
                                       .count = call->count,
                                       .images = parsed.tape,
                                       .embeddings = parsed.embeddings,
                                       .states = parsed.states,
                                       .arena = parsed.arena,
                                       .outputs_scalar = parsed.outputs_scalar,
                                       .progress = parsed.progress,
                                       .stats = &call->stats.image};
        uform_image_encode(&encode);
    } break;
    case uform_js_files_video_k: {
        uform_video_encode_t encode = {.model = model,
                                       .executor = parsed.executor,
                                       .outcome = outcome,
                                       .completion = parsed.completion,
                                       .count = call->count,
                                       .videos = parsed.tape,
                                       .embeddings = parsed.embeddings,
                                       .states = parsed.states,
                                       .arena = parsed.arena,
                                       .outputs_scalar = parsed.outputs_scalar,
                                       .progress = parsed.progress,
                                       .stats = &call->stats.video};
        uform_video_encode(&encode);
    } break;
    case uform_js_files_audio_k: {
        uform_audio_encode_t encode = {.model = model,
                                       .executor = parsed.executor,
                                       .outcome = outcome,
                                       .completion = parsed.completion,
                                       .count = call->count,
                                       .recordings = parsed.tape,
                                       .embeddings = parsed.embeddings,
                                       .states = parsed.states,
                                       .arena = parsed.arena,
                                       .outputs_scalar = parsed.outputs_scalar,
                                       .progress = parsed.progress,
                                       .stats = &call->stats.audio};
        uform_audio_encode(&encode);
    } break;
    }
    return uform_js_call_answer(env, &call->call, &local.call, slots, deferred);
}

static napi_value uform_js_text_encode(napi_env env, napi_callback_info info) {
    return uform_js_encode_files(env, info, uform_js_files_text_k);
}

static napi_value uform_js_image_encode(napi_env env, napi_callback_info info) {
    return uform_js_encode_files(env, info, uform_js_files_image_k);
}

static napi_value uform_js_video_encode(napi_env env, napi_callback_info info) {
    return uform_js_encode_files(env, info, uform_js_files_video_k);
}

static napi_value uform_js_audio_encode(napi_env env, napi_callback_info info) {
    return uform_js_encode_files(env, info, uform_js_files_audio_k);
}

/** @c imageEncodePixels(executor, completion, rgb, offsets, heights, widths, arena, outputs, outputsScalar,
 *  progress) → @c {embeddings, states, stats}, viewed inside @p arena. */
static napi_value uform_js_image_encode_pixels(napi_env env, napi_callback_info info) {
    napi_value arguments[10], self = NULL, deferred = NULL;
    uform_js_encode_call_t local;
    uform_js_encode_arguments_t parsed;
    uform_model_t model = NULL;
    uform_js_encode_call_t *const call = uform_js_encode_open(env, info, 10, arguments, &self, &local, napi_uint8_array,
                                                              6, &model, &parsed);
    if (!call) return NULL;
    void *heights = NULL, *widths = NULL;
    size_t heights_count = 0, widths_count = 0;
    if (!uform_js_array_from_value(env, arguments[4], "heights", napi_uint32_array, &heights, &heights_count) ||
        !uform_js_array_from_value(env, arguments[5], "widths", napi_uint32_array, &widths, &widths_count)) {
        uform_js_call_abandon(&call->call, &local.call);
        return NULL;
    }
    if (heights_count != call->count || widths_count != call->count) {
        uform_js_call_abandon(&call->call, &local.call);
        uform_js_throw(env, uform_status_shape_mismatch_k, "Every image needs one height and one width");
        return NULL;
    }
    napi_value const slots[uform_js_slots_k] = {self,         arguments[0], arguments[6],
                                                arguments[2], arguments[4], arguments[5]};
    if (!uform_js_call_hold(env, &call->call, &local.call, slots, &deferred)) return NULL;
    uform_image_encode_pixels_t encode = {.model = model,
                                          .executor = parsed.executor,
                                          .outcome = &call->call.outcome,
                                          .completion = parsed.completion,
                                          .count = call->count,
                                          .pixels = parsed.tape,
                                          .heights = (uint32_t const *)heights,
                                          .widths = (uint32_t const *)widths,
                                          .embeddings = parsed.embeddings,
                                          .states = parsed.states,
                                          .arena = parsed.arena,
                                          .outputs_scalar = parsed.outputs_scalar,
                                          .progress = parsed.progress,
                                          .stats = &call->stats.image};
    uform_image_encode_pixels(&encode);
    return uform_js_call_answer(env, &call->call, &local.call, slots, deferred);
}

/** @c audioEncodeSamples(executor, completion, samples, offsets, sampleRate, arena, outputs, outputsScalar,
 *  progress) → @c {embeddings, states, stats}, viewed inside @p arena. */
static napi_value uform_js_audio_encode_samples(napi_env env, napi_callback_info info) {
    napi_value arguments[9], self = NULL, deferred = NULL;
    uform_js_encode_call_t local;
    uform_js_encode_arguments_t parsed;
    uform_model_t model = NULL;
    uform_js_encode_call_t *const call = uform_js_encode_open(env, info, 9, arguments, &self, &local,
                                                              napi_float32_array, 5, &model, &parsed);
    if (!call) return NULL;
    size_t sample_rate = 0;
    bool const read = uform_js_size_from_value(env, arguments[4], "sampleRate", &sample_rate) &&
                      (sample_rate <= UINT32_MAX || uform_js_throw_field(env, "sampleRate", "must fit 32 bits"));
    if (!read) {
        uform_js_call_abandon(&call->call, &local.call);
        return NULL;
    }
    napi_value const slots[uform_js_slots_k] = {self, arguments[0], arguments[5], arguments[2]};
    if (!uform_js_call_hold(env, &call->call, &local.call, slots, &deferred)) return NULL;
    uform_audio_encode_samples_t encode = {.model = model,
                                           .executor = parsed.executor,
                                           .outcome = &call->call.outcome,
                                           .completion = parsed.completion,
                                           .count = call->count,
                                           .samples = parsed.tape,
                                           .sample_rate = (uint32_t)sample_rate,
                                           .embeddings = parsed.embeddings,
                                           .states = parsed.states,
                                           .arena = parsed.arena,
                                           .outputs_scalar = parsed.outputs_scalar,
                                           .progress = parsed.progress,
                                           .stats = &call->stats.audio};
    uform_audio_encode_samples(&encode);
    return uform_js_call_answer(env, &call->call, &local.call, slots, deferred);
}

#pragma endregion Verbs

#pragma region Registration

napi_property_descriptor const *uform_js_encode_properties(size_t *count) {
    static napi_property_descriptor const properties[] = {
        {"textEncode", NULL, uform_js_text_encode, NULL, NULL, NULL, napi_default_method, NULL},
        {"imageEncode", NULL, uform_js_image_encode, NULL, NULL, NULL, napi_default_method, NULL},
        {"imageEncodePixels", NULL, uform_js_image_encode_pixels, NULL, NULL, NULL, napi_default_method, NULL},
        {"videoEncode", NULL, uform_js_video_encode, NULL, NULL, NULL, napi_default_method, NULL},
        {"audioEncode", NULL, uform_js_audio_encode, NULL, NULL, NULL, napi_default_method, NULL},
        {"audioEncodeSamples", NULL, uform_js_audio_encode_samples, NULL, NULL, NULL, napi_default_method, NULL},
    };
    *count = sizeof(properties) / sizeof(properties[0]);
    return properties;
}

#pragma endregion Registration
