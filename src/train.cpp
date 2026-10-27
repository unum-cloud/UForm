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

#include "uform/train/trainer.hpp"

#include "lib.hpp"
#include "train/numerics.hpp" // `gpu_numerics_of`

using namespace unum::uform;
namespace ut = unum::uform::train;

using utilities::bf16_t;
using utilities::f32_t;
using encoder::modality_t;
using utilities::u32_t;

/** What @c uform_mesh_t owns: the bootstrapped mesh over its executor's devices, and how many trainers borrow it. */
struct uform_mesh_handle_t {
    uform_executor_t executor = nullptr;

    /** Trainers borrowing the mesh, touched through the standard atomic reference; a free waits for
     *  zero. */
    u64_t trainers = 0;

    std::unique_ptr<ut::mesh_t> mesh;
};

/** What @c uform_trainer_t owns: its run, the items source feeding it, and the host columns a step converts into;
 *  the mesh is the caller's borrowed one or a one-process mesh of its own. */
struct uform_trainer_handle_t {
    uform_executor_t executor = nullptr;
    std::variant<uform_mesh_t, std::unique_ptr<uform_mesh_handle_t>> mesh;
    std::unique_ptr<ut::trainer_t> trainer;
    std::unique_ptr<ut::items_source_t> items;

    /** One step's modalities and sides in the engine's spelling, and its embeddings before they
     *  land in the arena. */
    st::basic_vector<modality_t> modalities;
    st::basic_vector<encoder::late_interaction_side_t> sides;
    st::basic_vector<b8_t> embeddings;
};

/** The C status a training failure crosses into: an objective or a modality the run left out is unsupported. */
[[nodiscard]] static uform_status_t train_status_of(utilities::expected_t const &failure) noexcept {
    return status_of(failure.status_);
}

/** Whether @p executor drives GPUs, the only residency training runs on. */
[[nodiscard]] static bool drives_vram(uform_executor_t executor) noexcept {
    return executor && std::holds_alternative<vram_executor_t>(executor->executor);
}

/** The CUDA ordinals of a VRAM @p executor 's devices, the members of its mesh. */
[[nodiscard]] static bool devices_of(uform_executor_t executor, st::basic_vector<i32_t> &devices) noexcept {
    for (vram_device_t const &device : std::get<vram_executor_t>(executor->executor).devices)
        if (st::failed(devices.push_back(i32_t {device.vendor_ordinal}))) return false;
    return true;
}

/** The NumKong capabilities every GPU of a VRAM @p executor dispatches with, after its pin. */
[[nodiscard]] static nk_capability_t capabilities_of(uform_executor_t executor) noexcept {
    return std::get<vram_executor_t>(executor->executor).devices[0].capabilities;
}

/** The mesh a trainer runs over, borrowed or its own. */
[[nodiscard]] static uform_mesh_handle_t &mesh_of(uform_trainer_handle_t &trainer) noexcept {
    if (uform_mesh_t const *borrowed = std::get_if<uform_mesh_t>(&trainer.mesh)) return **borrowed;
    return *std::get<std::unique_ptr<uform_mesh_handle_t>>(trainer.mesh);
}

/** The static message an outcome carries for a fault a GPU recorded before it trapped. */
[[nodiscard]] static char const *fault_message(utilities::gpu_fault_code_t code) noexcept {
    switch (code) {
    case utilities::gpu_fault_code_t::none_k: return nullptr;
    case utilities::gpu_fault_code_t::barrier_deadline_k: return "A GPU waited past its deadline at a multimem barrier";
    case utilities::gpu_fault_code_t::flag_deadline_k: return "A GPU waited past its deadline for an InfiniBand round";
    case utilities::gpu_fault_code_t::completion_deadline_k:
        return "A GPU waited past its deadline for an InfiniBand completion";
    case utilities::gpu_fault_code_t::completion_error_k: return "An InfiniBand send completed with an error";
    case utilities::gpu_fault_code_t::nonfinite_signal_k: return "A telemetry probe sampled a non-finite value";
    }
    return nullptr;
}

/** The static messages one verb's engine failures carry, one per engine status, since the engine's
 *  own formatted description cannot outlive the call. */
struct failure_messages_t {
    char const *other;
    char const *out_of_memory;
    char const *invalid;
    char const *unsupported;
    char const *device;
    char const *io;
};

static constexpr failure_messages_t bootstrap_failures_k {
    "The mesh bootstrap failed",
    "Host memory ran out during the mesh bootstrap",
    "The mesh options are invalid: an index past the count, or peers without a directory and a job",
    "The processes disagree on the mesh's shape, GPU class or capabilities",
    "A GPU of the mesh could not be probed or pinned",
    "The exchange directory could not be written, or a peer never arrived",
};

static constexpr failure_messages_t build_failures_k {
    "The trainer could not be built",
    "Device memory ran out building the trainer",
    "The trainer's geometry does not fit its devices",
    "The trainer's geometry is unsupported on these devices",
    "A device failed while building the trainer",
    "A peer never arrived to wire the trainer's collectives",
};

static constexpr failure_messages_t resume_failures_k {
    "The checkpoint could not be resumed",
    "Memory ran out placing the checkpoint",
    "The checkpoint is malformed or holds another geometry",
    "The checkpoint holds tensors of another geometry",
    "A device failed placing the checkpoint",
    "The checkpoint could not be read",
};

static constexpr failure_messages_t source_failures_k {
    "The items source could not be built",
    "Memory ran out building the items source",
    "The items source does not fit the trainer's batch",
    "The items source does not support the run's modalities",
    "A device failed building the items source",
    "The items source could not read what it stages",
};

static constexpr failure_messages_t step_failures_k {
    "The step failed",
    "Memory ran out during the step",
    "The step's items, choices or relations are invalid",
    "The step's items do not fit the run's envelope or modalities",
    "A device failed during the step",
    "A peer missed the step's exchange",
};

static constexpr failure_messages_t save_failures_k {
    "The checkpoint could not be written",
    "Memory ran out staging the checkpoint",
    "The checkpoint path or its tensors are invalid",
    "The checkpoint holds tensors the format cannot store",
    "The weights could not be read back from the devices",
    "The checkpoint file could not be written or renamed into place",
};

/** The message of @p messages naming @p failure 's status. */
[[nodiscard]] static char const *message_of(utilities::expected_t const &failure,
                                            failure_messages_t const &messages) noexcept {
    switch (failure.status_) {
    case utilities::status_t::bad_alloc_k: return messages.out_of_memory;
    case utilities::status_t::invalid_input_k: return messages.invalid;
    case utilities::status_t::shape_unsupported_k: return messages.unsupported;
    case utilities::status_t::cuda_error_k: return messages.device;
    case utilities::status_t::io_error_k: return messages.io;
    default: return messages.other;
    }
}

/** Whether @p path holds a NUL byte, which no path can and a C string would cut it short at. */
[[nodiscard]] static bool holds_nul(uform_bytes_view_t path) noexcept { return text_of(path).contains('\0'); }

/** Publish a training @p failure with the message of @p messages naming its status, or, when
 *  the device failed, with the first fault a GPU of @p mesh recorded before it trapped; a null
 *  @p mesh , one that dies with the call, has none. */
static bool propagate_train_error(utilities::expected_t const &failure, uform_mesh_handle_t *mesh,
                                  uform_outcome_t *outcome, failure_messages_t const &messages) noexcept {
    char const *message = message_of(failure, messages);
    if (failure.status_ == utilities::status_t::cuda_error_k && mesh)
        for (utilities::gpu_fault_t const &fault : mesh->mesh->topology.faults())
            if (char const *const recorded = fault_message(fault.code)) {
                message = recorded;
                break;
            }
    return propagate_error(train_status_of(failure), outcome, message);
}

/** Bootstrap a mesh over @p executor 's devices from @p options into a new handle. */
[[nodiscard]] static std::unique_ptr<uform_mesh_handle_t> bootstrap(uform_executor_t executor,
                                                                    uform_mesh_options_t const &options,
                                                                    uform_outcome_t *outcome) noexcept {
    if (holds_nul(options.directory) || holds_nul(options.job))
        return propagate_error(uform_status_invalid_argument_k, outcome, "The directory and the job hold no NUL byte"),
               nullptr;
    st::basic_vector<i32_t> devices;
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
    engine.capabilities = capabilities_of(executor);
    if (utilities::expected_t const joined = ut::mesh_t::make(*handle->mesh, engine, devices); !joined) {
        // The engine refuses a world its processes disagree on, as unsupported.
        uform_status_t const status = joined.status_ == utilities::status_t::shape_unsupported_k
                                          ? uform_status_config_mismatch_k
                                          : train_status_of(joined);
        return propagate_error(status, outcome, message_of(joined, bootstrap_failures_k)), nullptr;
    }
    handle->executor = executor;
    return handle;
}

/** The engine's pair-loss form @p form names, or @p derived for its zero. */
[[nodiscard]] static ut::contrastive_pair_loss_form_t pair_loss_of(uform_pair_loss_form_t form,
                                                                   ut::contrastive_pair_loss_form_t derived) noexcept {
    switch (form) {
    case uform_pair_loss_form_balanced_k: return ut::contrastive_pair_loss_form_t::balanced_k;
    case uform_pair_loss_form_ranked_k: return ut::contrastive_pair_loss_form_t::ranked_k;
    case uform_pair_loss_form_sigmoid_k: return ut::contrastive_pair_loss_form_t::sigmoid_k;
    default: return derived;
    }
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
    if (arguments.contrastive.matryoshka_dimensions_count > ut::max_matryoshka_prefixes_k ||
        (arguments.contrastive.matryoshka_dimensions_count && !arguments.contrastive.matryoshka_dimensions))
        return "The Matryoshka widths are at most eight, and present when counted";
    if (arguments.contrastive.pair_loss > uform_pair_loss_form_sigmoid_k ||
        arguments.late_interaction.pair_loss > uform_pair_loss_form_sigmoid_k)
        return "The pair loss is not one the engine trains";
    if (arguments.choices.weight > 0.0f && !batch.max_choices_per_item)
        return "Choices need the batch's max_choices_per_item";

    config.depth = static_cast<u32_t>(arguments.depth);
    config.hidden = static_cast<u32_t>(arguments.hidden);
    config.batch = encoder::multimodal_batch_shape_t {static_cast<u32_t>(batch.max_rows_per_device),
                                                      static_cast<u32_t>(batch.max_segments_per_device),
                                                      static_cast<u32_t>(batch.max_columns_per_row)};
    config.batch.max_bags_per_gpu = static_cast<u32_t>(batch.max_bags_per_device);
    config.batch.max_choices_per_segment = static_cast<u32_t>(batch.max_choices_per_item);
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
    config.contrastive.pair_loss = pair_loss_of(contrastive.pair_loss, ut::contrastive_pair_loss_form_t::sigmoid_k);
    // Each form starts at its own logit scale: the sigmoid sum at SigLIP's β=10, b=−10.
    bool const sigmoid = config.contrastive.pair_loss == ut::contrastive_pair_loss_form_t::sigmoid_k;
    config.contrastive.inverse_temperature = chosen(contrastive.inverse_temperature, sigmoid ? 10.0f : 1.0f);
    config.contrastive.bias = chosen(contrastive.bias, sigmoid ? -10.0f : 0.0f);
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
    config.late_interaction.pair_loss = pair_loss_of(late_interaction.pair_loss,
                                                     ut::contrastive_pair_loss_form_t::ranked_k);
    config.choices.weight = arguments.choices.weight;
    config.choices.brier_weight = chosen(arguments.choices.brier_weight, 0.5f);

    config.learning_rate = chosen(arguments.schedule.learning_rate, 5.0e-4f);
    config.min_learning_rate = chosen(arguments.schedule.min_learning_rate, 1.0e-5f);
    config.warmup_steps = static_cast<u32_t>(arguments.schedule.warmup_steps);
    config.steps = static_cast<u32_t>(arguments.schedule.steps);
    config.seed = arguments.seed;
    return nullptr;
}

/** The C status a resume failure crosses into: a checkpoint of another geometry is a mismatch. */
[[nodiscard]] static uform_status_t resume_status_of(utilities::expected_t const &failure) noexcept {
    switch (failure.status_) {
    case utilities::status_t::invalid_input_k:
    case utilities::status_t::shape_unsupported_k: return uform_status_config_mismatch_k;
    default: return status_of(failure.status_);
    }
}

/** The edge set @p view over @p count items as the engine reads it. */
[[nodiscard]] static ut::item_edges_t edges_of(uform_edges_view_t const &view, std::size_t count) noexcept {
    if (!view.offsets) return {};
    std::size_t const edges = view.offsets[count];
    return {{{view.ids, edges}, {view.offsets, count + 1}}, {view.weights, view.weights ? edges : 0}};
}

/** Land the pooled vertex of every item of the last step in @p memory as @p scalar rows, zero for items that carried
 *  no tokens. False once @p outcome records why not. */
[[nodiscard]] static bool land_embeddings(uform_trainer_handle_t &handle, std::size_t count, uform_scalar_t scalar,
                                          chunked_arena_t &memory, uform_rows_view_t &rows,
                                          uform_outcome_t *outcome) noexcept {
    std::size_t const bytes_per_value = scalar == uform_scalar_f32_k ? 4 : 2;
    if (scalar != uform_scalar_f32_k && scalar != uform_scalar_f16_k && scalar != uform_scalar_bf16_k)
        return propagate_error(uform_status_unsupported_layout_k, outcome, "Embeddings land as f32, f16 or bf16");
    utilities::expected<utilities::bf16r2_view_t> pooled = handle.items->pooled(*handle.trainer);
    if (!pooled) return propagate_error(status_of(pooled.status_), outcome, "The pooled vertices could not be read");
    utilities::bf16r2_view_t const vertices = *pooled;
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
    utilities::switch_to_gpu_t gpu_scope {handle.trainer->streams[0]};
    if (!gpu_scope.ok() || cuMemcpyHtoD(utilities::device_address(landed), staging, bytes) != CUDA_SUCCESS)
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
    utilities::mesh_topology_t const &topology = arguments->mesh->mesh->topology;
    utilities::mesh_shape_t const &shape = topology.shape;
    st::basic_vector<i32_t> devices;
    if (!devices_of(arguments->mesh->executor, devices))
        return propagate_error(uform_status_out_of_memory_k, outcome, "Failed to list the devices"), void();
    utilities::expected<utilities::multimem_tier_choice_t> tier = utilities::multimem_tier_choice_t::make(
        shape.gpus_per_multimem_domain, devices);
    if (!tier)
        return propagate_error(status_of(tier.status_), outcome, "The multimem tier could not be probed"), void();
    uform_mesh_description_t &description = *arguments->description;
    description.gpus_per_host_domain = shape.gpus_per_host_domain;
    description.gpus_per_multimem_domain = shape.gpus_per_multimem_domain;
    description.gpus_per_colocated_domain = shape.gpus_per_colocated_domain;
    description.host_domain_index = topology.host_domain_index;
    description.first_gpu_index = topology.first_gpu_index();
    description.multimem_tier = static_cast<uform_multimem_tier_t>(tier.value_.tier);
    description.gpu_class = static_cast<uform_gpu_class_t>(gpu_class_of(capabilities_of(arguments->mesh->executor)));
    propagate_error(uform_status_success_k, outcome, nullptr);
}

UFORM_EXPORT void uform_mesh_free(uform_mesh_t mesh) {
    if (!mesh) return;
    finish_calls_using(mesh);
    fu::standard_atomic_ref<u64_t> trainers {mesh->trainers};
    for (u64_t seen = trainers.load(std::memory_order_acquire); seen; seen = trainers.load(std::memory_order_acquire))
        mesh->executor->host_verbs->wait(mesh->trainers, seen);
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
    if (holds_nul(arguments->resume))
        return propagate_error(uform_status_invalid_argument_k, outcome, "The resume path holds a NUL byte"), void();

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
    st::basic_vector<i32_t> devices;
    handle->trainer.reset(new (std::nothrow) ut::trainer_t {});
    handle->items.reset(new (std::nothrow) ut::items_source_t {});
    if (!handle->trainer || !handle->items || !devices_of(executor, devices))
        return propagate_error(uform_status_out_of_memory_k, outcome, "Failed to allocate the trainer"), void();
    if (utilities::expected_t const built = ut::trainer_t::make(
            *handle->trainer, *mesh_of(*handle).mesh, config, devices, ut::gpu_numerics_of(capabilities_of(executor)));
        !built)
        return propagate_train_error(built, arguments->mesh, outcome, build_failures_k), void();
    if (arguments->resume.length) {
        if (utilities::expected_t const resumed = handle->trainer->resume(text_of(arguments->resume)); !resumed)
            return propagate_error(resume_status_of(resumed), outcome, message_of(resumed, resume_failures_k)), void();
        std::size_t const tensors = handle->trainer->replicas[0].weights.size();
        progress_ticks_t ticks(arguments->progress, tensors);
        if (ticks(tensors) == continuation_t::cancel_k)
            return propagate_error(uform_status_cancelled_k, outcome, "The handler cancelled the resume"), void();
    }
    if (utilities::expected_t const sourced = ut::items_source_t::make(*handle->items, *handle->trainer); !sourced)
        return propagate_train_error(sourced, arguments->mesh, outcome, source_failures_k), void();
    std::size_t const segments = devices.size() * config.batch.max_segments_per_gpu;
    if (st::failed(handle->modalities.resize(segments)) || st::failed(handle->sides.resize(segments)) ||
        st::failed(handle->embeddings.resize(segments * config.hidden * sizeof(f32_t))))
        return propagate_error(uform_status_out_of_memory_k, outcome, "Failed to allocate the step columns"), void();
    if (arguments->mesh)
        fu::standard_atomic_ref<u64_t> {arguments->mesh->trainers}.fetch_add(1, std::memory_order_acq_rel);
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
    for (std::size_t item = 0; arguments->sides && item != count; ++item) {
        if (arguments->sides[item] > uform_late_interaction_side_query_and_document_k)
            return propagate_error(uform_status_invalid_argument_k, outcome, "An item's side is unrecognized"), void();
        handle->sides[item] = static_cast<encoder::late_interaction_side_t>(arguments->sides[item]);
    }
    uform_tape_view_t const &choices = arguments->choices, &targets = arguments->choice_targets;
    if (choices.offsets && !config.chooses())
        return propagate_error(uform_status_unsupported_k, outcome, "The run trains no choices"), void();
    if (choices.offsets && count && (!choices.data || !targets.data || !targets.offsets))
        return propagate_error(uform_status_invalid_argument_k, outcome, "Choices need bytes and targets"), void();
    if (choices.offsets && count && (choices.scalar != uform_scalar_u8_k || targets.scalar != uform_scalar_f32_k))
        return propagate_error(uform_status_unsupported_layout_k, outcome, "Choices are bytes, their targets f32"),
               void();
    for (std::size_t item = 0; choices.offsets && item != count; ++item) {
        if (choices.offsets[item + 1] < choices.offsets[item])
            return propagate_error(uform_status_malformed_k, outcome, "The choice offsets must not decrease"), void();
        std::size_t const offered = choices.offsets[item + 1] - choices.offsets[item];
        if (targets.offsets[item + 1] - targets.offsets[item] != offered)
            return propagate_error(uform_status_shape_mismatch_k, outcome, "Every choice needs one target"), void();
    }
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
                    tape_of<b8_t const>(contents, count),
                    {arguments->bags, arguments->bags ? count : 0},
                    {arguments->temperature_groups, arguments->temperature_groups ? count : 0},
                    edges_of(attract, count),
                    edges_of(arguments->relations.repel, count),
                    edges_of(arguments->relations.ignore, count),
                    {handle->sides.data(), arguments->sides ? count : 0},
                    {tape_of<utilities::u8_t const>(choices, count), tape_of<f32_t const>(targets, count)}};
    utilities::optimizer_step_t const optimizer_step {trainer.completed_steps};
    ut::step_report_t report;
    if (utilities::expected_t const stepped = trainer.step(ut::batch_source_of(source), optimizer_step, report);
        !stepped)
        return propagate_train_error(stepped, &mesh_of(*handle), outcome, step_failures_k), void();
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
                             telemetry.late_interaction,
                             config.chooses() ? telemetry.choices / config.choices.weight : 0.0f,
                             report.truncated,
                             telemetry.dropped};
    }
    if (arguments->embeddings) {
        chunked_arena_t &memory = chunks_of(arena->memory);
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
    if (holds_nul(arguments->path))
        return propagate_error(uform_status_invalid_argument_k, outcome, "The path holds a NUL byte"), void();
    if (utilities::expected_t const saved = handle->trainer->save(text_of(arguments->path)); !saved)
        return propagate_train_error(saved, &mesh_of(*handle), outcome, save_failures_k), void();
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
    if (uform_mesh_t const *borrowed = std::get_if<uform_mesh_t>(&trainer->mesh))
        fu::standard_atomic_ref<u64_t> {(*borrowed)->trainers}.fetch_sub(1, std::memory_order_acq_rel);
    delete trainer;
}

} // extern "C"
