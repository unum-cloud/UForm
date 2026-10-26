/**
 *  @file javascript/train.c
 *  @author Ash Vardanian
 *  @date October 17, 2026
 *  @brief The Node-API training verbs: the mesh a run's processes join, and the trainer that steps and saves.
 *
 *  Training is GPU-only, so both constructors take a VRAM executor and block on it; the trainer's step and save
 *  take the executor and the completion first, as every verb does.
 */
#include <stdlib.h> // `calloc`, `malloc`, `free`

#include "lib.h"

#pragma region Mesh

static void uform_js_mesh_finalize(napi_env env, void *data, void *hint) {
    uform_js_mesh_t *const mesh = (uform_js_mesh_t *)data;
    (void)env;
    (void)hint;
    if (!mesh) return;
    uform_mesh_free(mesh->handle);
    free(mesh);
}

/** @c bootstrap(executor, {hostDomainIndex, hostDomainCount, gpusPerMultimemDomain, directory, job,
 *  timeoutSeconds}) → this process's mesh, once every process has joined. */
static napi_value uform_js_mesh_bootstrap(napi_env env, napi_callback_info info) {
    napi_value arguments[2];
    uform_outcome_t outcome = {uform_status_success_k, NULL};
    uform_mesh_t handle = NULL;
    uform_mesh_bootstrap_t bootstrap = {.mesh = &handle, .outcome = &outcome};
    uform_mesh_options_t *const options = &bootstrap.options;
    char *directory = NULL, *job = NULL;
    if (!uform_js_arguments(env, info, 2, arguments, NULL) ||
        !uform_js_executor_from_value(env, arguments[0], &bootstrap.executor) ||
        !uform_js_size_from_value(env, uform_js_property(env, arguments[1], "hostDomainIndex"), "hostDomainIndex",
                                  &options->host_domain_index) ||
        !uform_js_size_from_value(env, uform_js_property(env, arguments[1], "hostDomainCount"), "hostDomainCount",
                                  &options->host_domain_count) ||
        !uform_js_size_from_value(env, uform_js_property(env, arguments[1], "gpusPerMultimemDomain"),
                                  "gpusPerMultimemDomain", &options->gpus_per_multimem_domain) ||
        !uform_js_double_from_value(env, uform_js_property(env, arguments[1], "timeoutSeconds"), "timeoutSeconds",
                                    &options->timeout_seconds))
        return NULL;
    napi_value const directory_name = uform_js_property(env, arguments[1], "directory");
    napi_value const job_name = uform_js_property(env, arguments[1], "job");
    if ((!uform_js_value_absent(env, directory_name) &&
         !(directory = uform_js_string_from_value(env, directory_name, "directory", &options->directory))) ||
        (!uform_js_value_absent(env, job_name) &&
         !(job = uform_js_string_from_value(env, job_name, "job", &options->job)))) {
        free(directory);
        return NULL;
    }
    uform_mesh_bootstrap(&bootstrap);
    free(directory);
    free(job);
    if (!uform_js_succeeded(env, &outcome)) return NULL;

    uform_js_addon_t *const addon = uform_js_addon(env);
    napi_value const instance = addon ? uform_js_instantiate(env, addon->mesh_class) : NULL;
    uform_js_mesh_t *const mesh = (uform_js_mesh_t *)calloc(1, sizeof(uform_js_mesh_t));
    if (!instance || !mesh || napi_wrap(env, instance, mesh, uform_js_mesh_finalize, NULL, NULL) != napi_ok) {
        free(mesh);
        uform_mesh_free(handle);
        uform_js_throw(env, uform_status_internal_k, "Could not instantiate the CompiledMesh");
        return NULL;
    }
    mesh->handle = handle;
    return instance;
}

/** @c describe() → the mesh's shape and this process's place in it. */
static napi_value uform_js_mesh_describe(napi_env env, napi_callback_info info) {
    uform_js_mesh_t *const mesh = (uform_js_mesh_t *)uform_js_self(env, info, 0, NULL, NULL);
    if (!mesh) return NULL;
    uform_outcome_t outcome = {uform_status_success_k, NULL};
    uform_mesh_description_t description = {0};
    uform_mesh_describe_t describe = {.mesh = mesh->handle, .outcome = &outcome, .description = &description};
    uform_mesh_describe(&describe);
    if (!uform_js_succeeded(env, &outcome)) return NULL;
    static char const *const names[] = {"gpusPerHostDomain", "gpusPerMultimemDomain", "gpusPerColocatedDomain",
                                        "hostDomainIndex",   "firstGpuIndex",         "multimemTier"};
    napi_value const values[] = {uform_js_number(env, description.gpus_per_host_domain),
                                 uform_js_number(env, description.gpus_per_multimem_domain),
                                 uform_js_number(env, description.gpus_per_colocated_domain),
                                 uform_js_number(env, description.host_domain_index),
                                 uform_js_number(env, description.first_gpu_index),
                                 uform_js_static_string(env, uform_multimem_tier_name(description.multimem_tier))};
    return uform_js_object(env, sizeof(names) / sizeof(names[0]), names, values);
}

/** @c free() - leaves the mesh now; every trainer borrowing it must be freed first; idempotent. */
static napi_value uform_js_mesh_release(napi_env env, napi_callback_info info) {
    uform_js_mesh_t *const mesh = (uform_js_mesh_t *)uform_js_self(env, info, 0, NULL, NULL);
    if (!mesh) return NULL;
    uform_mesh_free(mesh->handle);
    mesh->handle = NULL;
    return uform_js_undefined(env);
}

napi_value uform_js_mesh_define(napi_env env) {
    static napi_property_descriptor const properties[] = {
        {"free", NULL, uform_js_mesh_release, NULL, NULL, NULL, napi_default_method, NULL},
        {"describe", NULL, uform_js_mesh_describe, NULL, NULL, NULL, napi_default_method, NULL},
        {"bootstrap", NULL, uform_js_mesh_bootstrap, NULL, NULL, NULL, napi_static, NULL},
    };
    return uform_js_class_define(env, "CompiledMesh", sizeof(properties) / sizeof(properties[0]), properties);
}

#pragma endregion Mesh

#pragma region Trainer

/** A trainer call: the shared header, then the outputs a step lands. */
typedef struct uform_js_trainer_call_t {

    /** The outcome, the answer builder and the held values. */
    uform_js_call_t call;

    /** The pooled embeddings the contrastive loss scored, one row per item. */
    uform_rows_view_t embeddings;

    /** The step's report. */
    uform_trainer_step_stats_t stats;

    /** Items in the step. */
    size_t count;
} uform_js_trainer_call_t;

static void uform_js_trainer_finalize(napi_env env, void *data, void *hint) {
    uform_js_trainer_t *const trainer = (uform_js_trainer_t *)data;
    (void)hint;
    if (!trainer) return;
    uform_trainer_free(trainer->handle);
    if (trainer->mesh_reference) napi_delete_reference(env, trainer->mesh_reference);
    free(trainer);
}

/** Reads a named size of an options object. */
static bool uform_js_size_field(napi_env env, napi_value object, char const *name, size_t *result) {
    return uform_js_size_from_value(env, uform_js_property(env, object, name), name, result);
}

/** Reads a named real of an options object into a single-precision field. */
static bool uform_js_float_field(napi_env env, napi_value object, char const *name, float *result) {
    double wide = 0;
    if (!uform_js_double_from_value(env, uform_js_property(env, object, name), name, &wide)) return false;
    *result = (float)wide;
    return true;
}

/** Reads a batch envelope: the row, segment, column, bag and choice capacities, and the image,
 *  video and audio slices. */
static bool uform_js_batch_from_value(napi_env env, napi_value value, uform_batch_shape_t *batch) {
    napi_value const image = uform_js_property(env, value, "image");
    napi_value const video = uform_js_property(env, value, "video");
    napi_value const audio = uform_js_property(env, value, "audio");
    return uform_js_size_field(env, value, "maxRowsPerDevice", &batch->max_rows_per_device) &&
           uform_js_size_field(env, value, "maxSegmentsPerDevice", &batch->max_segments_per_device) &&
           uform_js_size_field(env, value, "maxColumnsPerRow", &batch->max_columns_per_row) &&
           uform_js_size_field(env, value, "maxBagsPerDevice", &batch->max_bags_per_device) &&
           uform_js_size_field(env, value, "maxChoicesPerItem", &batch->max_choices_per_item) &&
           uform_js_size_field(env, image, "patchSide", &batch->image.patch_side) &&
           uform_js_size_field(env, image, "channels", &batch->image.channels) &&
           uform_js_size_field(env, video, "patchSide", &batch->video.patch_side) &&
           uform_js_size_field(env, video, "tubeFrames", &batch->video.tube_frames) &&
           uform_js_size_field(env, video, "frameStride", &batch->video.frame_stride) &&
           uform_js_size_field(env, video, "channels", &batch->video.channels) &&
           uform_js_size_field(env, audio, "chunkSamples", &batch->audio.chunk_samples) &&
           uform_js_size_field(env, audio, "sampleRate", &batch->audio.sample_rate);
}

/** Reads the masked objective's corruption and head weights. */
static bool uform_js_masking_from_value(napi_env env, napi_value value, uform_masking_t *masking) {
    return uform_js_float_field(env, value, "tokenMaskProbability", &masking->token_mask_probability) &&
           uform_js_float_field(env, value, "tokenMeanSpan", &masking->token_mean_span) &&
           uform_js_float_field(env, value, "replaceFraction", &masking->replace_fraction) &&
           uform_js_float_field(env, value, "randomFraction", &masking->random_fraction) &&
           uform_js_float_field(env, value, "imageMaskRatio", &masking->image_mask_ratio) &&
           uform_js_float_field(env, value, "videoMaskRatio", &masking->video_mask_ratio) &&
           uform_js_float_field(env, value, "audioMaskRatio", &masking->audio_mask_ratio) &&
           uform_js_float_field(env, value, "textWeight", &masking->text_weight) &&
           uform_js_float_field(env, value, "imageWeight", &masking->image_weight) &&
           uform_js_float_field(env, value, "videoWeight", &masking->video_weight) &&
           uform_js_float_field(env, value, "audioWeight", &masking->audio_weight);
}

/** Reads the contrastive loss's scoring; the Matryoshka widths land in a block the caller frees. */
static bool uform_js_contrastive_from_value(napi_env env, napi_value value, uform_contrastive_t *contrastive,
                                            size_t **widths) {
    size_t negatives = 0, pair_loss = 0;
    if (!uform_js_size_field(env, value, "temperatureGroups", &contrastive->temperature_groups) ||
        !uform_js_float_field(env, value, "inverseTemperature", &contrastive->inverse_temperature) ||
        !uform_js_float_field(env, value, "bias", &contrastive->bias) ||
        !uform_js_size_field(env, value, "negatives", &negatives) ||
        !uform_js_float_field(env, value, "ambientRepulsionWeight", &contrastive->ambient_repulsion_weight) ||
        !uform_js_size_field(env, value, "maxEdgesPerDevice", &contrastive->max_edges_per_device) ||
        !uform_js_size_field(env, value, "pairLoss", &pair_loss))
        return false;
    contrastive->negatives = (uform_negatives_t)negatives;
    contrastive->pair_loss = (uform_pair_loss_form_t)pair_loss;
    int32_t *ordinals = NULL;
    size_t count = 0;
    if (!uform_js_ordinals_from_value(env, uform_js_property(env, value, "matryoshkaDimensions"),
                                      "matryoshkaDimensions", &ordinals, &count))
        return false;
    *widths = count ? (size_t *)malloc(count * sizeof(size_t)) : NULL;
    if (count && !*widths) {
        free(ordinals);
        return uform_js_throw(env, uform_status_out_of_memory_k, "Could not copy the Matryoshka widths");
    }
    for (size_t position = 0; position != count; ++position) (*widths)[position] = (size_t)ordinals[position];
    free(ordinals);
    contrastive->matryoshka_dimensions = *widths;
    contrastive->matryoshka_dimensions_count = count;
    return true;
}

/** Reads the late-interaction loss's scoring of the per-token vectors. */
static bool uform_js_late_interaction_from_value(napi_env env, napi_value value,
                                                 uform_late_interaction_t *late_interaction) {
    size_t pair_loss = 0;
    if (!uform_js_float_field(env, value, "weight", &late_interaction->weight) ||
        !uform_js_size_field(env, value, "width", &late_interaction->width) ||
        !uform_js_float_field(env, value, "inverseTemperature", &late_interaction->inverse_temperature) ||
        !uform_js_float_field(env, value, "bias", &late_interaction->bias) ||
        !uform_js_size_field(env, value, "maxQuerySpan", &late_interaction->max_query_span) ||
        !uform_js_size_field(env, value, "maxQueryTokensPerDevice", &late_interaction->max_query_tokens_per_device) ||
        !uform_js_size_field(env, value, "maxDocumentsPerDevice", &late_interaction->max_documents_per_device) ||
        !uform_js_size_field(env, value, "maxDocumentTokensPerDevice",
                             &late_interaction->max_document_tokens_per_device) ||
        !uform_js_size_field(env, value, "pairLoss", &pair_loss))
        return false;
    late_interaction->pair_loss = (uform_pair_loss_form_t)pair_loss;
    return true;
}

/** @c initVram(executor, mesh, {resume, depth, hidden, batch, objective, masking, contrastive,
 *  lateInteraction, choices, schedule, seed}) → a trainer, its creation blocking on @p executor. */
static napi_value uform_js_trainer_init_vram(napi_env env, napi_callback_info info) {
    napi_value arguments[3];
    uform_outcome_t outcome = {uform_status_success_k, NULL};
    uform_trainer_t handle = NULL;
    uform_trainer_init_vram_t init = {.trainer = &handle, .outcome = &outcome};
    uform_js_addon_t *const addon = uform_js_addon(env);
    if (!addon || !uform_js_arguments(env, info, 3, arguments, NULL) ||
        !uform_js_executor_from_value(env, arguments[0], &init.executor))
        return NULL;
    if (!uform_js_value_absent(env, arguments[1])) {
        uform_js_mesh_t const *const mesh = (uform_js_mesh_t const *)uform_js_unwrap_instance(
            env, arguments[1], addon->mesh_class, "mesh", "must be a Mesh");
        if (!mesh) return NULL;
        init.mesh = mesh->handle;
    }
    napi_value const config = arguments[2];
    napi_value const schedule = uform_js_property(env, config, "schedule");
    napi_value const choices = uform_js_property(env, config, "choices");
    napi_value const resume = uform_js_property(env, config, "resume");
    size_t objective = 0, seed = 0;
    size_t *widths = NULL;
    char *resume_path = NULL;
    if (!uform_js_size_field(env, config, "depth", &init.depth) ||
        !uform_js_size_field(env, config, "hidden", &init.hidden) ||
        !uform_js_size_field(env, config, "objective", &objective) ||
        !uform_js_size_field(env, config, "seed", &seed) ||
        !uform_js_batch_from_value(env, uform_js_property(env, config, "batch"), &init.batch) ||
        !uform_js_masking_from_value(env, uform_js_property(env, config, "masking"), &init.masking) ||
        !uform_js_late_interaction_from_value(env, uform_js_property(env, config, "lateInteraction"),
                                              &init.late_interaction) ||
        !uform_js_float_field(env, choices, "weight", &init.choices.weight) ||
        !uform_js_float_field(env, choices, "brierWeight", &init.choices.brier_weight) ||
        !uform_js_float_field(env, schedule, "learningRate", &init.schedule.learning_rate) ||
        !uform_js_float_field(env, schedule, "minLearningRate", &init.schedule.min_learning_rate) ||
        !uform_js_size_field(env, schedule, "warmupSteps", &init.schedule.warmup_steps) ||
        !uform_js_size_field(env, schedule, "steps", &init.schedule.steps) ||
        !uform_js_contrastive_from_value(env, uform_js_property(env, config, "contrastive"), &init.contrastive,
                                         &widths))
        return NULL;
    if (!uform_js_value_absent(env, resume) &&
        !(resume_path = uform_js_string_from_value(env, resume, "resume", &init.resume))) {
        free(widths);
        return NULL;
    }
    init.objective = (uform_objective_t)objective;
    init.seed = seed;
    uform_trainer_init_vram(&init);
    free(widths);
    free(resume_path);
    if (!uform_js_succeeded(env, &outcome)) return NULL;

    napi_value const instance = uform_js_instantiate(env, addon->trainer_class);
    uform_js_trainer_t *const trainer = (uform_js_trainer_t *)calloc(1, sizeof(uform_js_trainer_t));
    bool const held = !init.mesh ||
                      (trainer && napi_create_reference(env, arguments[1], 1, &trainer->mesh_reference) == napi_ok);
    if (!instance || !trainer || !held ||
        napi_wrap(env, instance, trainer, uform_js_trainer_finalize, NULL, NULL) != napi_ok) {
        if (trainer && trainer->mesh_reference) napi_delete_reference(env, trainer->mesh_reference);
        free(trainer);
        uform_trainer_free(handle);
        uform_js_throw(env, uform_status_internal_k, "Could not instantiate the CompiledTrainer");
        return NULL;
    }
    trainer->handle = handle;
    return instance;
}

/** Reads the receiver of a trainer verb, refusing a freed one. */
static uform_js_trainer_t *uform_js_trainer_live(napi_env env, napi_callback_info info, size_t wanted,
                                                 napi_value *arguments, napi_value *self) {
    uform_js_trainer_t *const trainer = (uform_js_trainer_t *)uform_js_self(env, info, wanted, arguments, self);
    if (trainer && !trainer->handle) {
        uform_js_throw(env, uform_status_invalid_argument_k, "The trainer was freed");
        return NULL;
    }
    return trainer;
}

static napi_value uform_js_finish_step(napi_env env, uform_js_call_t *call, napi_value const *slots) {
    uform_js_trainer_call_t const *const step = (uform_js_trainer_call_t const *)call;
    uform_trainer_step_stats_t const *const stats = &step->stats;
    static char const *const names[] = {"step",
                                        "learningRate",
                                        "loss",
                                        "contrastiveLoss",
                                        "maskedTextLoss",
                                        "maskedImageLoss",
                                        "maskedVideoLoss",
                                        "maskedAudioLoss",
                                        "lateInteractionLoss",
                                        "choicesLoss",
                                        "truncated"};
    napi_value values[sizeof(names) / sizeof(names[0])];
    float const reals[] = {stats->learning_rate,     stats->loss,
                           stats->contrastive_loss,  stats->masked_text_loss,
                           stats->masked_image_loss, stats->masked_video_loss,
                           stats->masked_audio_loss, stats->late_interaction_loss,
                           stats->choices_loss};
    size_t const reals_count = sizeof(reals) / sizeof(reals[0]);
    values[0] = uform_js_number(env, stats->step);
    for (size_t position = 0; position != reals_count; ++position)
        napi_create_double(env, reals[position], &values[position + 1]);
    values[reals_count + 1] = uform_js_number(env, stats->truncated);
    napi_value const report = uform_js_object(env, sizeof(names) / sizeof(names[0]), names, values);
    napi_value object = NULL;
    if (!report || napi_create_object(env, &object) != napi_ok ||
        napi_set_named_property(env, object, "stats", report) != napi_ok)
        return NULL;
    if (!uform_js_value_absent(env, slots[uform_js_slot_arena_k])) {
        napi_value const embeddings = uform_js_rows_object(env, slots[uform_js_slot_arena_k], &step->embeddings,
                                                           step->count);
        if (!embeddings || napi_set_named_property(env, object, "embeddings", embeddings) != napi_ok) return NULL;
    }
    return object;
}

/** Reads one edge set: its widened offsets, its borrowed ids, and its @b optional borrowed weights. */
static bool uform_js_edges_from_value(napi_env env, napi_value const *arguments, uform_js_call_t *call,
                                      uform_edges_view_t *edges) {
    size_t count = 0, ids_count = 0, weights_count = 0;
    void *ids = NULL, *weights = NULL;
    if (!uform_js_offsets_from_value(env, arguments[0], "edges", call, &edges->offsets, &count)) return false;
    if (!edges->offsets) return true;
    if (!uform_js_array_from_value(env, arguments[1], "ids", napi_uint32_array, &ids, &ids_count) ||
        (!uform_js_value_absent(env, arguments[2]) &&
         !uform_js_array_from_value(env, arguments[2], "weights", napi_float32_array, &weights, &weights_count)))
        return false;
    if (edges->offsets[count] > ids_count || (weights && weights_count != ids_count))
        return uform_js_throw(env, uform_status_shape_mismatch_k, "The edges reach past their ids or weights");
    edges->ids = (uint32_t const *)ids;
    edges->weights = (float const *)weights;
    return true;
}

/** Reads the @b optional choices: widened offsets, borrowed bytes, and one target per byte. */
static bool uform_js_choices_from_value(napi_env env, napi_value const *arguments, size_t count, uform_js_call_t *call,
                                        uform_choices_view_t *choices) {
    size_t offsets_count = 0, bytes_count = 0, targets_count = 0;
    void *bytes = NULL, *targets = NULL;
    if (!uform_js_offsets_from_value(env, arguments[0], "choices", call, &choices->offsets, &offsets_count))
        return false;
    if (!choices->offsets) return true;
    if (offsets_count != count) return uform_js_throw_field(env, "choices", "must hold one more offset than items");
    if (!uform_js_array_from_value(env, arguments[1], "bytes", napi_uint8_array, &bytes, &bytes_count) ||
        !uform_js_array_from_value(env, arguments[2], "targets", napi_float32_array, &targets, &targets_count))
        return false;
    if (choices->offsets[count] > bytes_count || targets_count != bytes_count)
        return uform_js_throw(env, uform_status_shape_mismatch_k, "The choices reach past their bytes or targets");
    choices->bytes = (uint8_t const *)bytes;
    choices->targets = (float const *)targets;
    return true;
}

/** @c step(executor, completion, modalities, contents, offsets, bags, temperatureGroups, sides,
 *  choiceOffsets, choiceBytes, choiceTargets, attractOffsets, attractIds, attractWeights,
 *  repelOffsets, repelIds, repelWeights, ignoreOffsets, ignoreIds, arena, outputsScalar) →
 *  @c {stats, embeddings}, the embeddings present when an arena was given. */
static napi_value uform_js_trainer_step(napi_env env, napi_callback_info info) {
    napi_value arguments[21], self = NULL, deferred = NULL;
    uform_js_trainer_t *const trainer = uform_js_trainer_live(env, info, 21, arguments, &self);
    if (!trainer) return NULL;
    uform_js_trainer_call_t local;
    uform_trainer_step_t step = {.trainer = trainer->handle};
    uform_js_trainer_call_t *const call = (uform_js_trainer_call_t *)uform_js_call_open_on(
        env, arguments, &local.call, sizeof(local), &step.executor, &step.completion);
    if (!call) return NULL;
    void *modalities = NULL, *sides = NULL;
    size_t modalities_count = 0, sides_count = 0;
    napi_value const ignore[3] = {arguments[17], arguments[18], NULL};
    bool const with_arena = !uform_js_value_absent(env, arguments[19]);
    bool const with_sides = !uform_js_value_absent(env, arguments[7]);
    bool parsed =
        uform_js_tape_from_value(env, arguments[3], arguments[4], "contents", napi_uint8_array, &call->call,
                                 &step.count, &step.contents) &&
        uform_js_array_from_value(env, arguments[2], "modalities", napi_int32_array, &modalities, &modalities_count) &&
        (modalities_count == step.count || uform_js_throw_field(env, "modalities", "must hold one entry per item")) &&
        uform_js_per_item_from_value(env, arguments[5], "bags", step.count, &step.bags) &&
        uform_js_per_item_from_value(env, arguments[6], "temperatureGroups", step.count, &step.temperature_groups) &&
        (!with_sides ||
         (uform_js_array_from_value(env, arguments[7], "sides", napi_int32_array, &sides, &sides_count) &&
          (sides_count == step.count || uform_js_throw_field(env, "sides", "must hold one entry per item")))) &&
        uform_js_choices_from_value(env, arguments + 8, step.count, &call->call, &step.choices) &&
        uform_js_edges_from_value(env, arguments + 11, &call->call, &step.relations.attract) &&
        uform_js_edges_from_value(env, arguments + 14, &call->call, &step.relations.repel) &&
        uform_js_edges_from_value(env, ignore, &call->call, &step.relations.ignore);
    if (parsed && with_arena)
        parsed = uform_js_scalar_from_value(env, arguments[20], "outputsScalar", &step.outputs_scalar) &&
                 uform_js_arena_from_value(env, arguments[19], &call->call, &step.arena);
    if (!parsed) {
        uform_js_call_abandon(&call->call, &local.call);
        return NULL;
    }
    napi_value const slots[uform_js_slots_k] = {self,          arguments[0],  arguments[19], arguments[3],
                                                arguments[2],  arguments[5],  arguments[6],  arguments[7],
                                                arguments[9],  arguments[10], arguments[12], arguments[13],
                                                arguments[15], arguments[16], arguments[18]};
    call->call.finish = uform_js_finish_step;
    call->count = step.count;
    step.modalities = (uform_modality_t const *)modalities;
    step.sides = (uform_late_interaction_side_t const *)sides;
    step.outcome = &call->call.outcome;
    step.embeddings = with_arena ? &call->embeddings : NULL;
    step.stats = &call->stats;
    if (!uform_js_call_hold(env, &call->call, &local.call, slots, &deferred)) return NULL;
    uform_trainer_step(&step);
    return uform_js_call_answer(env, &call->call, &local.call, slots, deferred);
}

static napi_value uform_js_finish_nothing(napi_env env, uform_js_call_t *call, napi_value const *slots) {
    (void)call;
    (void)slots;
    return uform_js_undefined(env);
}

/** @c save(executor, completion, path) - writes the checkpoint, renamed into place once whole. */
static napi_value uform_js_trainer_save(napi_env env, napi_callback_info info) {
    napi_value arguments[3], self = NULL, deferred = NULL;
    uform_js_trainer_t *const trainer = uform_js_trainer_live(env, info, 3, arguments, &self);
    if (!trainer) return NULL;
    uform_js_call_t local;
    uform_trainer_save_t save = {.trainer = trainer->handle};
    uform_js_call_t *const call = uform_js_call_open_on(env, arguments, &local, sizeof(local), &save.executor,
                                                        &save.completion);
    if (!call) return NULL;
    if (!(call->owned[0] = uform_js_string_from_value(env, arguments[2], "path", &save.path))) {
        uform_js_call_abandon(call, &local);
        return NULL;
    }
    napi_value const slots[uform_js_slots_k] = {self, arguments[0]};
    call->finish = uform_js_finish_nothing;
    save.outcome = &call->outcome;
    if (!uform_js_call_hold(env, call, &local, slots, &deferred)) return NULL;
    uform_trainer_save(&save);
    return uform_js_call_answer(env, call, &local, slots, deferred);
}

/** @c free() - releases the trainer and lets go of its mesh; idempotent. */
static napi_value uform_js_trainer_release(napi_env env, napi_callback_info info) {
    uform_js_trainer_t *const trainer = (uform_js_trainer_t *)uform_js_self(env, info, 0, NULL, NULL);
    if (!trainer) return NULL;
    uform_trainer_free(trainer->handle);
    trainer->handle = NULL;
    if (trainer->mesh_reference) napi_delete_reference(env, trainer->mesh_reference);
    trainer->mesh_reference = NULL;
    return uform_js_undefined(env);
}

napi_value uform_js_trainer_define(napi_env env) {
    static napi_property_descriptor const properties[] = {
        {"free", NULL, uform_js_trainer_release, NULL, NULL, NULL, napi_default_method, NULL},
        {"step", NULL, uform_js_trainer_step, NULL, NULL, NULL, napi_default_method, NULL},
        {"save", NULL, uform_js_trainer_save, NULL, NULL, NULL, napi_default_method, NULL},
        {"initVram", NULL, uform_js_trainer_init_vram, NULL, NULL, NULL, napi_static, NULL},
    };
    return uform_js_class_define(env, "CompiledTrainer", sizeof(properties) / sizeof(properties[0]), properties);
}

#pragma endregion Trainer
