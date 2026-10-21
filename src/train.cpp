/**
 *  @file src/train.cpp
 *  @author Ash Vardanian
 *  @date October 18, 2026
 *  @brief The verbs of @c train.h over @c train::mesh_t and @c train::trainer_t : the mesh bootstrap, a trainer's
 *      construction from the C structs, its steps over caller items, and its checkpoints.
 *
 *  A mesh is made once per process over a VRAM executor, whose devices are its members, and borrowed by every
 *  trainer made over the same executor; a trainer without one makes a one-process mesh of its own. Every option is
 *  explicit: the library never reads the environment.
 */
#include <memory> // `std::unique_ptr` - a handle under construction
#include <new>    // `std::nothrow`

#include <cuda.h>      // `cuMemcpyHtoD`
#include <cuda_fp16.h> // `__float2half` - embeddings landing as Float16

#include "lib.hpp"
#include "train/trainer.hpp"

using namespace unum::uform;
namespace ut = unum::uform::train;

using core::bf16_t;
using core::f32_t;
using core::modality_t;
using core::u32_t;

/** What @c uform_mesh_t owns: the bootstrapped mesh over its executor's devices, and how many trainers borrow it. */
struct uform_mesh_handle_t {
    uform_executor_t executor = nullptr;

    /** Trainers borrowing the mesh; a free waits for zero. */
    std::atomic<std::size_t> trainers {0};

    std::unique_ptr<ut::mesh_t> mesh;
};

/** What @c uform_trainer_t owns: its run, the items source feeding it, and the host columns a step converts into;
 *  the mesh is the caller's borrowed one or a one-process mesh of its own. */
struct uform_trainer_handle_t {
    uform_executor_t executor = nullptr;
    std::variant<uform_mesh_t, std::unique_ptr<uform_mesh_handle_t>> mesh;
    std::unique_ptr<ut::trainer_t> trainer;
    std::unique_ptr<ut::items_source_t> items;

    /** One step's modalities in the engine's spelling, and its embeddings before they land in the arena. */
    array<modality_t> modalities;
    array<b8_t> embeddings;
};

/** The C status a training failure crosses into: an objective or a modality the run left out is unsupported. */
[[nodiscard]] static uform_status_t train_status_of(core::expected_t const &failure) noexcept {
    return status_of(failure.status_);
}

/** Whether @p executor drives GPUs, the only residency training runs on. */
[[nodiscard]] static bool drives_vram(uform_executor_t executor) noexcept {
    return executor && std::holds_alternative<vram_executor_t>(executor->executor);
}

/** The device ordinals of a VRAM @p executor , the members of its mesh. */
[[nodiscard]] static bool devices_of(uform_executor_t executor, array<i32_t> &devices) noexcept {
    for (vram_device_t const &device : std::get<vram_executor_t>(executor->executor).devices)
        if (devices.push_back(device.ordinal) != core::status_t::success_k) return false;
    return true;
}

/** The mesh a trainer runs over, borrowed or its own. */
[[nodiscard]] static ut::mesh_t &mesh_of(uform_trainer_handle_t &trainer) noexcept {
    if (uform_mesh_t const *borrowed = std::get_if<uform_mesh_t>(&trainer.mesh)) return *(*borrowed)->mesh;
    return *std::get<std::unique_ptr<uform_mesh_handle_t>>(trainer.mesh)->mesh;
}

/** Bootstrap a mesh over @p executor 's devices from @p options into a new handle. */
[[nodiscard]] static std::unique_ptr<uform_mesh_handle_t> bootstrap(uform_executor_t executor,
                                                                    uform_mesh_options_t const &options,
                                                                    uform_outcome_t *outcome) noexcept {
    array<i32_t> devices;
    if (!devices_of(executor, devices))
        return propagate_error(uform_status_out_of_memory_k, outcome, "Failed to list the devices"), nullptr;
    std::unique_ptr<uform_mesh_handle_t> handle {new (std::nothrow) uform_mesh_handle_t {}};
    if (handle) handle->mesh.reset(new (std::nothrow) ut::mesh_t {});
    if (!handle || !handle->mesh)
        return propagate_error(uform_status_out_of_memory_k, outcome, "Failed to allocate the mesh"), nullptr;
    ut::mesh_options_t engine;
    engine.host_domain_index = static_cast<u32_t>(options.host_domain_index);
    engine.host_domain_count = options.host_domain_count ? static_cast<u32_t>(options.host_domain_count) : 1u;
    engine.gpus_per_multimem_domain = static_cast<u32_t>(options.gpus_per_multimem_domain);
    engine.directory = text_of(options.directory);
    engine.job = text_of(options.job);
    engine.timeout_seconds = options.timeout_seconds > 0.0 ? options.timeout_seconds : 120.0;
    if (core::expected_t const joined = ut::mesh_t::make(*handle->mesh, engine, devices.view()); !joined)
        return propagate_error(train_status_of(joined), outcome, "The mesh bootstrap failed"), nullptr;
    handle->executor = executor;
    return handle;
}

/** Fold every field of @p arguments into the engine's run description, or answer the refusal's message. */
[[nodiscard]] static char const *configure(uform_trainer_init_vram_t const &arguments,
                                           ut::trainer_config_t &config) noexcept {
    uform_batch_shape_t const &batch = arguments.batch;
    if (!arguments.depth || !arguments.hidden || arguments.hidden % 128)
        return "The depth must be positive and the width a positive multiple of 128";
    if (!batch.max_rows_per_device || !batch.max_segments_per_device || !batch.max_columns_per_row)
        return "The batch needs rows, segments and columns";
    if (!arguments.schedule.steps) return "The schedule needs steps";
    if (arguments.objective > uform_objective_joint_k) return "The objective is not one the engine trains";
    if (arguments.contrastive.matryoshka_dimensions_count > core::max_matryoshka_prefixes_k ||
        (arguments.contrastive.matryoshka_dimensions_count && !arguments.contrastive.matryoshka_dimensions))
        return "The Matryoshka widths are at most eight, and present when counted";

    config.depth = static_cast<u32_t>(arguments.depth);
    config.hidden = static_cast<u32_t>(arguments.hidden);
    config.batch = core::multimodal_batch_shape_t {static_cast<u32_t>(batch.max_rows_per_device),
                                                   static_cast<u32_t>(batch.max_segments_per_device),
                                                   static_cast<u32_t>(batch.max_columns_per_row)};
    config.batch.max_bags_per_gpu = static_cast<u32_t>(batch.max_bags_per_device);
    if (batch.image.patch_side)
        config.batch.image = {static_cast<u32_t>(batch.image.patch_side),
                              batch.image.channels ? static_cast<u32_t>(batch.image.channels) : 4u};
    if (batch.video.patch_side)
        config.batch.video = {static_cast<u32_t>(batch.video.patch_side),
                              batch.video.channels ? static_cast<u32_t>(batch.video.channels) : 3u,
                              batch.video.tube_frames ? static_cast<u32_t>(batch.video.tube_frames) : 2u,
                              batch.video.frame_stride ? static_cast<u32_t>(batch.video.frame_stride) : 1u};
    if (batch.audio.chunk_samples)
        config.batch.audio = {static_cast<u32_t>(batch.audio.chunk_samples),
                              batch.audio.sample_rate ? static_cast<u32_t>(batch.audio.sample_rate) : 16000u};
    config.objective = static_cast<ut::objective_t>(arguments.objective);

    auto chosen = [](float value, float fallback) noexcept { return value != 0.0f ? value : fallback; };
    uform_masking_t const &masking = arguments.masking;
    config.masking.token_mask_probability = chosen(masking.token_mask_probability, 0.15f);
    config.masking.token_mean_span = chosen(masking.token_mean_span, 16.0f);
    config.masking.replace_fraction = chosen(masking.replace_fraction, 0.8f);
    config.masking.random_fraction = chosen(masking.random_fraction, 0.1f);
    config.image_mask_ratio = chosen(masking.image_mask_ratio, 0.75f);
    config.video_mask_ratio = chosen(masking.video_mask_ratio, 0.75f);
    config.audio_mask_ratio = chosen(masking.audio_mask_ratio, 0.75f);
    config.text_weight = chosen(masking.text_weight, 1.0f);
    config.image_weight = chosen(masking.image_weight, 1.0f);
    config.video_weight = chosen(masking.video_weight, 1.0f);
    config.audio_weight = chosen(masking.audio_weight, 1.0f);

    uform_contrastive_t const &contrastive = arguments.contrastive;
    config.contrastive.matryoshka_count = static_cast<u32_t>(contrastive.matryoshka_dimensions_count);
    for (std::size_t index = 0; index != contrastive.matryoshka_dimensions_count; ++index)
        config.contrastive.matryoshka_dimensions[index] = static_cast<u32_t>(contrastive.matryoshka_dimensions[index]);
    config.contrastive.temperature_groups = contrastive.temperature_groups
                                                ? static_cast<u32_t>(contrastive.temperature_groups)
                                                : 1u;
    config.contrastive.inverse_temperature = chosen(contrastive.inverse_temperature, 10.0f);
    config.contrastive.bias = chosen(contrastive.bias, -10.0f);
    config.contrastive.ambient_repulsion_weight = contrastive.negatives == uform_negatives_named_k
                                                      ? 0.0f
                                                      : chosen(contrastive.ambient_repulsion_weight, 1.0f);
    config.contrastive.max_edges_per_gpu = static_cast<u32_t>(contrastive.max_edges_per_device);

    uform_late_interaction_t const &late_interaction = arguments.late_interaction;
    config.late_interaction.weight = late_interaction.weight;
    config.late_interaction.width = late_interaction.width ? static_cast<u32_t>(late_interaction.width) : 128u;
    config.late_interaction.inverse_temperature = chosen(late_interaction.inverse_temperature, 1.0f);
    config.late_interaction.bias = late_interaction.bias;
    if (late_interaction.max_query_span)
        config.late_interaction.max_query_span = static_cast<u32_t>(late_interaction.max_query_span);
    config.late_interaction.max_query_tokens_per_gpu = static_cast<u32_t>(late_interaction.max_query_tokens_per_device);
    config.late_interaction.max_documents_per_gpu = static_cast<u32_t>(late_interaction.max_documents_per_device);
    config.late_interaction.max_document_tokens_per_gpu = static_cast<u32_t>(
        late_interaction.max_document_tokens_per_device);

    config.learning_rate = chosen(arguments.schedule.learning_rate, 5.0e-4f);
    config.min_learning_rate = chosen(arguments.schedule.min_learning_rate, 1.0e-5f);
    config.warmup_steps = static_cast<u32_t>(arguments.schedule.warmup_steps);
    config.steps = static_cast<u32_t>(arguments.schedule.steps);
    config.seed = arguments.seed;
    return nullptr;
}

/** The C status a resume failure crosses into: a checkpoint of another geometry is a mismatch. */
[[nodiscard]] static uform_status_t resume_status_of(core::expected_t const &failure) noexcept {
    switch (failure.status_) {
    case core::status_t::invalid_input_k:
    case core::status_t::shape_unsupported_k: return uform_status_config_mismatch_k;
    default: return status_of(failure.status_);
    }
}

/** Every edge set of @p view as the engine reads it. */
[[nodiscard]] static ut::item_edges_t edges_of(uform_edges_view_t const &view) noexcept {
    return {view.offsets, view.ids, view.weights};
}

/** Land the pooled vertex of every item of the last step in @p memory as @p scalar rows, zero for items that carried
 *  no tokens. False once @p outcome records why not. */
[[nodiscard]] static bool land_embeddings(uform_trainer_handle_t &handle, std::size_t count, uform_scalar_t scalar,
                                          chunked_arena_t &memory, uform_rows_view_t &rows,
                                          uform_outcome_t *outcome) noexcept {
    std::size_t const bytes_per_value = scalar == uform_scalar_f32_k ? 4 : 2;
    if (scalar != uform_scalar_f32_k && scalar != uform_scalar_f16_k && scalar != uform_scalar_bf16_k)
        return propagate_error(uform_status_unsupported_layout_k, outcome, "Embeddings land as f32, f16 or bf16");
    core::expected<core::bf16r2_view_t> pooled = handle.items->pooled(*handle.trainer);
    if (!pooled) return propagate_error(status_of(pooled.status_), outcome, "The pooled vertices could not be read");
    core::bf16r2_view_t const vertices = *pooled;
    std::size_t const hidden = vertices.extent(1), bytes = count * hidden * bytes_per_value;
    b8_t *const staging = handle.embeddings.data();
    for (std::size_t item = 0; item != count; ++item) {
        u32_t const vertex = handle.items->destination(item);
        for (std::size_t column = 0; column != hidden; ++column) {
            f32_t const value = vertex != ~u32_t {0} ? __bfloat162float(vertices[vertex, column]) : 0.0f;
            b8_t *const target = staging + (item * hidden + column) * bytes_per_value;
            if (scalar == uform_scalar_f32_k) *static_cast<f32_t *>(static_cast<void *>(target)) = value;
            else if (scalar == uform_scalar_f16_k)
                *static_cast<__half *>(static_cast<void *>(target)) = __float2half(value);
            else *static_cast<bf16_t *>(static_cast<void *>(target)) = __float2bfloat16(value);
        }
    }
    void *const landed = memory.allocate(bytes ? bytes : 1);
    if (!landed) return propagate_error(uform_status_out_of_memory_k, outcome, "Failed to allocate the embeddings");
    core::switch_to_gpu_t gpu_scope {handle.trainer->streams[0]};
    if (!gpu_scope.ok() || cuMemcpyHtoD(core::device_address(landed), staging, bytes) != CUDA_SUCCESS)
        return propagate_error(uform_status_device_k, outcome, "The embeddings could not land in the arena");
    rows = {landed, hidden * bytes_per_value, hidden, scalar};
    return true;
}

extern "C" {

UFORM_EXPORT void uform_mesh_bootstrap(uform_mesh_bootstrap_t *arguments) {
    if (submitted<uform_mesh_bootstrap>(arguments)) return;
    call_outcome_t outcome(arguments->outcome);
    *arguments->mesh = nullptr;
    if (!drives_vram(arguments->executor))
        return propagate_error(uform_status_device_k, outcome, "The verb needs a VRAM executor"), void();
    std::unique_ptr<uform_mesh_handle_t> mesh = bootstrap(arguments->executor, arguments->options, outcome);
    if (!mesh) return;
    *arguments->mesh = mesh.release();
    propagate_error(uform_status_success_k, outcome, nullptr);
}

UFORM_EXPORT void uform_mesh_describe(uform_mesh_describe_t *arguments) {
    call_outcome_t outcome(arguments->outcome);
    if (!arguments->mesh || !arguments->description)
        return propagate_error(uform_status_invalid_argument_k, outcome, "The mesh and the output are required"),
               void();
    finish_calls_using(arguments->mesh);
    core::mesh_topology_t const &topology = arguments->mesh->mesh->topology;
    core::mesh_shape_t const &shape = topology.shape;
    array<i32_t> devices;
    if (!devices_of(arguments->mesh->executor, devices))
        return propagate_error(uform_status_out_of_memory_k, outcome, "Failed to list the devices"), void();
    core::expected<core::multimem_tier_choice_t> tier = core::multimem_tier_choice_t::make(
        shape.gpus_per_multimem_domain, devices.view());
    if (!tier)
        return propagate_error(status_of(tier.status_), outcome, "The multimem tier could not be probed"), void();
    uform_mesh_description_t &description = *arguments->description;
    description.gpus_per_host_domain = shape.gpus_per_host_domain;
    description.gpus_per_multimem_domain = shape.gpus_per_multimem_domain;
    description.gpus_per_colocated_domain = shape.gpus_per_colocated_domain;
    description.host_domain_index = topology.host_domain_index;
    description.first_gpu_index = topology.first_gpu_index();
    description.multimem_tier = static_cast<uform_multimem_tier_t>(tier.value_.tier);
    propagate_error(uform_status_success_k, outcome, nullptr);
}

UFORM_EXPORT void uform_mesh_free(uform_mesh_t mesh) {
    if (!mesh) return;
    finish_calls_using(mesh);
    for (std::size_t trainers = mesh->trainers.load(std::memory_order_acquire); trainers;
         trainers = mesh->trainers.load(std::memory_order_acquire))
        mesh->trainers.wait(trainers, std::memory_order_acquire);
    delete mesh;
}

UFORM_EXPORT void uform_trainer_init_vram(uform_trainer_init_vram_t *arguments) {
    if (submitted<uform_trainer_init_vram>(arguments)) return;
    call_outcome_t outcome(arguments->outcome);
    *arguments->trainer = nullptr;
    uform_executor_t const executor = arguments->executor;
    if (!drives_vram(executor))
        return propagate_error(uform_status_device_k, outcome, "The verb needs a VRAM executor"), void();
    if (arguments->mesh && arguments->mesh->executor != executor)
        return propagate_error(uform_status_invalid_argument_k, outcome, "The mesh was made over another executor"),
               void();
    ut::trainer_config_t config;
    if (char const *refused = configure(*arguments, config))
        return propagate_error(uform_status_invalid_argument_k, outcome, refused), void();

    std::unique_ptr<uform_trainer_handle_t> handle {new (std::nothrow) uform_trainer_handle_t {}};
    if (!handle)
        return propagate_error(uform_status_out_of_memory_k, outcome, "Failed to allocate the trainer"), void();
    handle->executor = executor;
    if (arguments->mesh) handle->mesh = arguments->mesh;
    else {
        std::unique_ptr<uform_mesh_handle_t> own = bootstrap(executor, uform_mesh_options_t {}, outcome);
        if (!own) return;
        handle->mesh = std::move(own);
    }
    array<i32_t> devices;
    handle->trainer.reset(new (std::nothrow) ut::trainer_t {});
    handle->items.reset(new (std::nothrow) ut::items_source_t {});
    if (!handle->trainer || !handle->items || !devices_of(executor, devices))
        return propagate_error(uform_status_out_of_memory_k, outcome, "Failed to allocate the trainer"), void();
    if (core::expected_t const built = ut::trainer_t::make(*handle->trainer, mesh_of(*handle), config, devices.view());
        !built)
        return propagate_error(train_status_of(built), outcome, "The trainer could not be built"), void();
    if (arguments->resume.length) {
        if (core::expected_t const resumed = handle->trainer->resume(text_of(arguments->resume)); !resumed)
            return propagate_error(resume_status_of(resumed), outcome, "The checkpoint could not be resumed"), void();
        std::size_t const tensors = handle->trainer->replicas[0].weights.size();
        progress_ticks_t ticks(arguments->progress, tensors);
        if (ticks(tensors) == continuation_t::cancel_k)
            return propagate_error(uform_status_cancelled_k, outcome, "The handler cancelled the resume"), void();
    }
    if (core::expected_t const sourced = ut::items_source_t::make(*handle->items, *handle->trainer); !sourced)
        return propagate_error(train_status_of(sourced), outcome, "The items source could not be built"), void();
    std::size_t const segments = devices.size() * config.batch.max_segments_per_gpu;
    if (handle->modalities.resize(segments) != core::status_t::success_k ||
        handle->embeddings.resize(segments * config.hidden * sizeof(f32_t)) != core::status_t::success_k)
        return propagate_error(uform_status_out_of_memory_k, outcome, "Failed to allocate the step columns"), void();
    if (arguments->mesh) arguments->mesh->trainers.fetch_add(1, std::memory_order_acq_rel);
    *arguments->trainer = handle.release();
    propagate_error(uform_status_success_k, outcome, nullptr);
}

UFORM_EXPORT void uform_trainer_step(uform_trainer_step_t *arguments) {
    if (submitted<uform_trainer_step>(arguments)) return;
    uform_arena_t const requested = arguments->embeddings ? arguments->arena : nullptr;
    call_outcome_t outcome(arguments->outcome, requested);
    uform_trainer_handle_t *const handle = arguments->trainer;
    if (!handle || handle->executor != arguments->executor)
        return propagate_error(uform_status_invalid_argument_k, outcome, "The trainer and its executor are required"),
               void();
    ut::trainer_t &trainer = *handle->trainer;
    ut::trainer_config_t const &config = trainer.config;
    std::size_t const count = arguments->count;
    uform_tape_view_t const &contents = arguments->contents;
    if (count > handle->modalities.size())
        return propagate_error(uform_status_shape_mismatch_k, outcome, "More items than the devices' envelopes hold"),
               void();
    if (count && (!arguments->modalities || !contents.data || !contents.offsets))
        return propagate_error(uform_status_invalid_argument_k, outcome, "The items need modalities and contents"),
               void();
    if (count && contents.scalar != uform_scalar_u8_k)
        return propagate_error(uform_status_unsupported_layout_k, outcome, "Training items are bytes"), void();
    for (std::size_t item = 0; item != count; ++item) {
        if (contents.offsets[item + 1] < contents.offsets[item])
            return propagate_error(uform_status_malformed_k, outcome, "The tape's offsets must not decrease"), void();
        if (arguments->modalities[item] > uform_modality_audio_k)
            return propagate_error(uform_status_invalid_argument_k, outcome, "An item's modality is unrecognized"),
                   void();
        handle->modalities[item] = static_cast<modality_t>(arguments->modalities[item]);
    }
    if (arguments->bags && !config.batch.max_bags_per_gpu)
        return propagate_error(uform_status_invalid_argument_k, outcome, "Bags need the batch's bag budget"), void();
    uform_edges_view_t const &attract = arguments->relations.attract;
    if (config.contrasts() && (!attract.offsets || attract.offsets[count] == attract.offsets[0]))
        return propagate_error(uform_status_invalid_argument_k, outcome, "A contrastive step names attract edges"),
               void();
    if (arguments->embeddings && !config.contrasts())
        return propagate_error(uform_status_unsupported_k, outcome, "Only a contrastive objective pools items"), void();
    uform_arena_t const arena = requested
                                    ? claim_arena(requested, arguments->executor,
                                                  {contents.data, contents.offsets, arguments->modalities}, outcome)
                                    : nullptr;
    if (requested && !arena) return;

    ut::items_source_t &source = *handle->items;
    source.items = {{handle->modalities.data(), count},
                    static_cast<b8_t const *>(contents.data),
                    contents.offsets,
                    arguments->bags,
                    arguments->temperature_groups,
                    edges_of(attract),
                    edges_of(arguments->relations.repel),
                    edges_of(arguments->relations.ignore)};
    core::optimizer_step_t const optimizer_step {trainer.completed_steps};
    ut::step_report_t report;
    if (core::expected_t const stepped = trainer.step(source, optimizer_step, report); !stepped)
        return propagate_error(train_status_of(stepped), outcome, "The step failed"), void();
    if (arguments->stats) {
        ut::telemetry_t const telemetry = trainer.telemetry(optimizer_step);
        *arguments->stats = {optimizer_step.ordinal + 1u,
                             report.learning_rate,
                             telemetry.composite(),
                             telemetry.contrastive,
                             telemetry.token_ce / config.text_weight,
                             telemetry.image_mse / config.image_weight,
                             telemetry.video_mse / config.video_weight,
                             telemetry.audio_mse / config.audio_weight,
                             report.truncated};
    }
    if (arguments->embeddings) {
        chunked_arena_t &memory = std::visit([](chunked_arena_t &held) noexcept -> chunked_arena_t & { return held; },
                                             arena->memory);
        if (!land_embeddings(*handle, count, arguments->outputs_scalar, memory, *arguments->embeddings, outcome))
            return;
    }
    propagate_error(uform_status_success_k, outcome, nullptr);
}

UFORM_EXPORT void uform_trainer_save(uform_trainer_save_t *arguments) {
    if (submitted<uform_trainer_save>(arguments)) return;
    call_outcome_t outcome(arguments->outcome);
    uform_trainer_handle_t *const handle = arguments->trainer;
    if (!handle || handle->executor != arguments->executor || !arguments->path.length)
        return propagate_error(uform_status_invalid_argument_k, outcome, "The trainer, its executor and a path"),
               void();
    if (core::expected_t const saved = handle->trainer->save(text_of(arguments->path)); !saved)
        return propagate_error(train_status_of(saved), outcome, "The checkpoint could not be written"), void();
    std::size_t const tensors = handle->trainer->replicas[0].weights.size();
    progress_ticks_t ticks(arguments->progress, tensors);
    (void)ticks(tensors);
    propagate_error(uform_status_success_k, outcome, nullptr);
}

UFORM_EXPORT void uform_trainer_free(uform_trainer_t trainer) {
    if (!trainer) return;
    finish_calls_using(trainer);
    trainer->items.reset();
    trainer->trainer.reset();
    if (uform_mesh_t const *borrowed = std::get_if<uform_mesh_t>(&trainer->mesh)) {
        (*borrowed)->trainers.fetch_sub(1, std::memory_order_acq_rel);
        (*borrowed)->trainers.notify_all();
    }
    delete trainer;
}

} // extern "C"
