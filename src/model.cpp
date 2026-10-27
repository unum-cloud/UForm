/**
 *  @file src/model.cpp
 *  @author Ash Vardanian
 *  @date October 17, 2026
 *  @brief The verbs of @c model.h: load a checkpoint into one host tower or one tower per GPU, describe it, free it.
 *
 *  A load maps the checkpoint or views the caller's buffer, reads the geometry from its metadata, and packs every
 *  tower from it, so the mapping is closed and the buffer released before the load completes. A RAM model packs once
 *  for the CPU and serves every RAM executor; a VRAM model packs once per device its load names, each copy placed by
 *  that device's capabilities and stream.
 */
#include <cstring> // `std::memchr` - a NUL inside a checkpoint path

#include <memory> // `std::unique_ptr` - a model under construction
#include <new>    // `std::nothrow`
#include <span>   // `std::span` - a checkpoint buffer

#include "encoder_punned.hpp" // `encoder_verbs`, `pick_cpu_variant`, `pick_gpu_variant`

using namespace unum::uform;
namespace ue = unum::uform::encoder;

uform_model_handle_t::~uform_model_handle_t() noexcept {
    std::visit(
        [&](auto const *verbs) noexcept {
            if (!verbs) return;
            if (void *const *const tower = std::get_if<void *>(&towers)) {
                if (*tower) verbs->destroy(*tower);
                return;
            }
            for (uf::vram_replica_t &replica : std::get<uf::per_gpu_array<uf::vram_replica_t>>(towers))
                if (replica.tower) verbs->destroy(replica.tower);
        },
        tables);
}

/** What a load was doing when the engine failed, which picks the message a malformed file gets. */
enum class load_stage_t : u8_t {
    reading_k,
    measuring_k,
    packing_k,
};

/** Records a failed checkpoint read, geometry read or tower pack at @p stage with a static
 *  message per engine @p status : what the file holds is malformed, not an invalid argument. */
static bool propagate_load_failure(utilities::status_t status, load_stage_t stage, uform_outcome_t *outcome) noexcept {
    switch (status) {
    case utilities::status_t::invalid_input_k:
        switch (stage) {
        case load_stage_t::reading_k:
            return propagate_error(uform_status_malformed_k, outcome, "The checkpoint is not a safetensors file");
        case load_stage_t::measuring_k:
            return propagate_error(uform_status_malformed_k, outcome, "The checkpoint carries no usable geometry");
        case load_stage_t::packing_k:
            return propagate_error(uform_status_malformed_k, outcome, "The checkpoint misses or misshapes a tensor");
        }
        break;
    case utilities::status_t::bad_alloc_k:
        return propagate_error(uform_status_out_of_memory_k, outcome, "Failed to allocate the model's weights");
    case utilities::status_t::io_error_k:
        return propagate_error(uform_status_io_k, outcome, "The checkpoint file could not be opened or mapped");
    case utilities::status_t::shape_unsupported_k:
        return propagate_error(uform_status_unsupported_k, outcome, "No kernel of the picked class packs a tensor");
    default: break;
    }
    return propagate_error(uform_status_internal_k, outcome, "The model or its device failed to load");
}

/** The precision @p quantization names, false once @p outcome records that the blocks run in neither. */
static bool precision_of(uform_scalar_t quantization, ue::precision_t &precision, uform_outcome_t *outcome) noexcept {
    switch (quantization) {
    case uform_scalar_bf16_k: precision = ue::precision_t::bf16_k; return true;
    case uform_scalar_e4m3_k: precision = ue::precision_t::e4m3_k; return true;
    default: return propagate_error(uform_status_unsupported_k, outcome, "Blocks run in bf16 or e4m3 only");
    }
}

/** Bytes one tower of @p geometry holds in @p precision : the block products in its dtype, the entries, pool and
 *  head in BFloat16, the norm gains and pixel statistics widened to Float32, and the rotary angle grids. */
static std::size_t tower_bytes(ue::geometry_t const &geometry, ue::precision_t precision) noexcept {
    std::size_t const hidden = geometry.hidden, ffn = geometry.ffn();
    std::size_t const product_bytes = precision == ue::precision_t::e4m3_k ? 1 : 2;
    std::size_t const block = (3 * hidden * hidden + hidden * hidden + 2 * ffn * hidden + hidden * ffn) *
                                  product_bytes +
                              2 * hidden * sizeof(float);
    std::size_t const entries = (geometry.vocabulary + geometry.image_scalars() + geometry.video_scalars() +
                                 geometry.audio_chunk_samples) *
                                    hidden * 2 +
                                2 * (geometry.image_channels + geometry.video_channels) * sizeof(float);
    std::size_t const pool = 2 * hidden * hidden * 2 + hidden * 2 + 2 * hidden * sizeof(float);
    std::size_t const angles = std::size_t {geometry.context} *
                               (geometry.rope_temporal + geometry.rope_height + geometry.rope_width) / 2 *
                               sizeof(float);
    return geometry.depth * block + entries + pool + angles;
}

/** Parses the checkpoint @p loaded holds into a model handle in @p precision , its towers still to be packed; null
 *  once @p outcome records why not. */
static std::unique_ptr<uform_model_handle_t> start_model(
    utilities::expected<utilities::checkpoint_to_load_t> const &loaded, ue::precision_t precision,
    uform_outcome_t *outcome) noexcept {
    if (!loaded) return propagate_load_failure(loaded.status_, load_stage_t::reading_k, outcome), nullptr;
    utilities::expected<ue::geometry_t> geometry = ue::geometry_t::from(*loaded);
    if (!geometry) return propagate_load_failure(geometry.status_, load_stage_t::measuring_k, outcome), nullptr;
    std::unique_ptr<uform_model_handle_t> model {new (std::nothrow) uform_model_handle_t {}};
    if (!model) return propagate_error(uform_status_out_of_memory_k, outcome, "Failed to allocate the model"), nullptr;
    model->geometry = *geometry;
    model->precision = precision;
    std::size_t const groups = geometry->choice_temperature_groups;
    nk::tensor_view<utilities::f32_t, 2> const stored = loaded->float32("choices.inverse_temperatures");
    utilities::f32_t const *const calibrated = stored.empty() ? nullptr : stored.data();
    if (utilities::st::failed(model->choice_inverse_temperatures.resize(st::assume_uninitialized, groups)))
        return propagate_error(uform_status_out_of_memory_k, outcome, "Failed to allocate the model"), nullptr;
    for (std::size_t group = 0; group != groups; ++group)
        model->choice_inverse_temperatures[group] = calibrated ? calibrated[group] : 1.0f;
    return model;
}

/** Packs one tower of @p model from @p checkpoint through @p verbs on @p stream , null on the host;
 *  null once @p outcome records why not. */
template <encoder_home_t home_>
static void *pack_tower(uform_model_handle_t &model, utilities::checkpoint_to_load_t const &checkpoint,
                        encoder_verbs<home_> const &verbs, nk_stream_t stream, uform_outcome_t *outcome) noexcept {
    utilities::expected<void *> packed = verbs.create(checkpoint, model.geometry, model.precision, stream);
    if (!packed) return propagate_load_failure(packed.status_, load_stage_t::packing_k, outcome), nullptr;
    model.memory_usage += tower_bytes(model.geometry, model.precision);
    return *packed;
}

/** The CPU capabilities a RAM load picks its class within: the executor's, or this machine's
 *  without one. */
static bool host_capabilities(uform_executor_t executor, nk_capability_t &available,
                              uform_outcome_t *outcome) noexcept {
    if (!executor) {
        if (nk_cpu_capabilities_enabled(&available) != nk_success_k)
            return propagate_error(uform_status_device_k, outcome, "The CPU capabilities could not be read");
        return true;
    }
    ram_executor_t const *const ram = std::get_if<ram_executor_t>(&executor->executor);
    if (!ram) return propagate_error(uform_status_device_k, outcome, "A RAM load needs a RAM executor or none");
    available = ram->capabilities;
    return true;
}

/** Packs a RAM model from @p loaded for the class @p ram_options pins, or the newest the CPU runs,
 *  and publishes it into @p output once the handler lets the load finish. */
static void load_ram(uform_model_t *output, uform_executor_t executor, uform_ram_options_t const &ram_options,
                     utilities::expected<utilities::checkpoint_to_load_t> loaded, ue::precision_t precision,
                     uform_progress_t progress, uform_outcome_t *outcome) noexcept {
    nk_capability_t available = 0;
    if (!host_capabilities(executor, available, outcome)) return;
    if (!cpu_pin_admitted(ram_options.cpu_class, available))
        return propagate_error(uform_status_unsupported_k, outcome, "The CPU does not admit the pinned class"), void();
    encoder_verbs<encoder_home_t::ram_k> const *const verbs = pick_cpu_variant(cpu_ladder_t {}, ram_options.cpu_class,
                                                                               available);
    std::unique_ptr<uform_model_handle_t> model = start_model(loaded, precision, outcome);
    if (!model) return;
    model->tables = verbs;
    void *&tower = model->towers.emplace<void *>(pack_tower(*model, *loaded, *verbs, nullptr, outcome));
    if (!tower) return;
    if (progress_ticks_t {progress, 1}(1) == continuation_t::cancel_k)
        return propagate_error(uform_status_cancelled_k, outcome, "The caller's handler stopped the load"), void();
    *output = model.release();
    propagate_error(uform_status_success_k, outcome, nullptr);
}

/** Packs one replica per device of @p devices that @p vram names through the newest class of
 *  @p ladder_type_ every one of them runs; false once @p outcome records why not. */
template <encoder_home_t home_, typename ladder_type_>
static bool load_replicas(uform_model_handle_t &model, utilities::checkpoint_to_load_t const &checkpoint,
                          vram_executor_t &devices, uform_vram_options_t const &vram, progress_ticks_t &ticks,
                          uform_outcome_t *outcome) noexcept {
    per_gpu_array<vram_replica_t> &replicas = std::get<per_gpu_array<vram_replica_t>>(model.towers);
    encoder_verbs<home_> const *verbs = nullptr;
    std::size_t placed = 0;
    for (vram_device_t &device : devices.devices) {
        std::size_t named = !vram.devices_count;
        for (std::size_t listed = 0; listed != vram.devices_count; ++listed)
            named += device.ordinal == vram.devices[listed];
        if (!named) continue;
        encoder_verbs<home_> const *const picked = pick_gpu_variant<home_>(ladder_type_ {}, vram.gpu_class,
                                                                           device.capabilities);
        if (!picked)
            return propagate_error(uform_status_unsupported_k, outcome,
                                   "A device runs no compiled class under the pin");
        if (verbs && picked != verbs)
            return propagate_error(uform_status_config_mismatch_k, outcome, "The devices differ in class");
        verbs = picked;
        model.tables = verbs;
        void *const tower = pack_tower(model, checkpoint, *verbs, device.stream.get(), outcome);
        if (!tower) return false;
        if (replicas.push_back(vram_replica_t {device.ordinal, tower}) != utilities::status_t::success_k)
            return verbs->destroy(tower),
                   propagate_error(uform_status_out_of_memory_k, outcome, "Failed to list the replicas");
        if (ticks(++placed, devices.devices.size()) == continuation_t::cancel_k)
            return propagate_error(uform_status_cancelled_k, outcome, "The caller's handler stopped the load");
    }
    return true;
}

/** Packs a VRAM model from @p loaded on every device of @p executor that @p vram names, all of them when it names
 *  none, and publishes it into @p output . */
static void load_vram(uform_model_t *output, uform_executor_t executor, uform_vram_options_t const &vram,
                      utilities::expected<utilities::checkpoint_to_load_t> loaded, ue::precision_t precision,
                      uform_progress_t progress, uform_outcome_t *outcome) noexcept {
    vram_executor_t *const devices = executor ? std::get_if<vram_executor_t>(&executor->executor) : nullptr;
    if (!devices) return propagate_error(uform_status_device_k, outcome, "A VRAM load needs a VRAM executor"), void();
    if (vram.devices_count && !vram.devices)
        return propagate_error(uform_status_invalid_argument_k, outcome, "The devices must not be null"), void();
    for (std::size_t listed = 0; listed != vram.devices_count; ++listed) {
        std::size_t matches = 0;
        for (vram_device_t const &device : devices->devices) matches += device.ordinal == vram.devices[listed];
        if (!matches)
            return propagate_error(uform_status_device_k, outcome, "A device the load names is not the executor's"),
                   void();
    }
    std::unique_ptr<uform_model_handle_t> model = start_model(loaded, precision, outcome);
    if (!model) return;
    model->towers.emplace<per_gpu_array<vram_replica_t>>();
    progress_ticks_t ticks {progress, devices->devices.size()};
    bool const packed = visit_vram_home(
        devices->devices[0].capabilities, [&]<encoder_home_t home_, typename ladder_type_>() noexcept {
            return load_replicas<home_, ladder_type_>(*model, *loaded, *devices, vram, ticks, outcome);
        });
    if (!packed) return;
    *output = model.release();
    propagate_error(uform_status_success_k, outcome, nullptr);
}

/** Maps the checkpoint file @p path , which the load closes before completing. */
static utilities::expected<utilities::checkpoint_to_load_t> map_checkpoint(uform_bytes_view_t path) noexcept {
    return utilities::checkpoint_to_load_t::make(text_of(path));
}

/** Views the checkpoint image @p buffer , borrowed only until the towers are packed. */
static utilities::expected<utilities::checkpoint_to_load_t> view_checkpoint(uform_bytes_view_t buffer) noexcept {
    return utilities::checkpoint_to_load_t::view(
        {static_cast<b8_t const *>(static_cast<void const *>(buffer.data)), buffer.length});
}

/** The refusals every load shares, clearing the output first so a refused load leaves no model. */
template <typename arguments_type_>
static bool rejects_load(arguments_type_ const &arguments, uform_bytes_view_t source, ue::precision_t &precision,
                         uform_outcome_t *outcome) noexcept {
    if (!arguments.model)
        return !propagate_error(uform_status_invalid_argument_k, outcome, "The model output must not be null");
    *arguments.model = nullptr;
    if (!source.data || !source.length)
        return !propagate_error(uform_status_invalid_argument_k, outcome, "The checkpoint must not be empty");
    if constexpr (requires { arguments.path; }) // a C path ends at its first NUL, so one inside names another file
        if (std::memchr(source.data, 0, source.length))
            return !propagate_error(uform_status_invalid_argument_k, outcome, "The checkpoint path holds a NUL byte");
    return !precision_of(arguments.quantization, precision, outcome);
}

extern "C" {

UFORM_EXPORT void uform_model_load_ram(uform_model_load_ram_t *arguments) {
    if (submitted<uform_model_load_ram>(arguments)) return;
    call_outcome_t outcome(arguments->outcome);
    ue::precision_t precision {};
    if (rejects_load(*arguments, arguments->path, precision, outcome)) return;
    load_ram(arguments->model, arguments->executor, arguments->ram, map_checkpoint(arguments->path), precision,
             arguments->progress, outcome);
}

UFORM_EXPORT void uform_model_load_ram_buffer(uform_model_load_ram_buffer_t *arguments) {
    if (submitted<uform_model_load_ram_buffer>(arguments)) return;
    call_outcome_t outcome(arguments->outcome);
    ue::precision_t precision {};
    if (rejects_load(*arguments, arguments->buffer, precision, outcome)) return;
    load_ram(arguments->model, arguments->executor, arguments->ram, view_checkpoint(arguments->buffer), precision,
             arguments->progress, outcome);
}

UFORM_EXPORT void uform_model_load_vram(uform_model_load_vram_t *arguments) {
    if (submitted<uform_model_load_vram>(arguments)) return;
    call_outcome_t outcome(arguments->outcome);
    ue::precision_t precision {};
    if (rejects_load(*arguments, arguments->path, precision, outcome)) return;
    load_vram(arguments->model, arguments->executor, arguments->vram, map_checkpoint(arguments->path), precision,
              arguments->progress, outcome);
}

UFORM_EXPORT void uform_model_load_vram_buffer(uform_model_load_vram_buffer_t *arguments) {
    if (submitted<uform_model_load_vram_buffer>(arguments)) return;
    call_outcome_t outcome(arguments->outcome);
    ue::precision_t precision {};
    if (rejects_load(*arguments, arguments->buffer, precision, outcome)) return;
    load_vram(arguments->model, arguments->executor, arguments->vram, view_checkpoint(arguments->buffer), precision,
              arguments->progress, outcome);
}

UFORM_EXPORT void uform_model_describe(uform_model_describe_t *arguments) {
    if (submitted<uform_model_describe>(arguments)) return;
    call_outcome_t outcome(arguments->outcome);
    if (!arguments->model || !arguments->description)
        return propagate_error(uform_status_invalid_argument_k, outcome, "The model and description must not be null"),
               void();
    uform_model_handle_t const &model = *arguments->model;
    cpu_class_t picked_cpu = cpu_class_t::unknown_k;
    gpu_class_t picked_gpu = gpu_class_t::unknown_k;
    std::visit(
        [&](auto const *verbs) noexcept {
            if (verbs) picked_cpu = verbs->cpu_class, picked_gpu = verbs->gpu_class;
        },
        model.tables);
    *arguments->description = {
        model.geometry.hidden,
        model.geometry.context,
        model.precision == ue::precision_t::e4m3_k ? uform_scalar_e4m3_k : uform_scalar_bf16_k,
        model.memory_usage,
        static_cast<uform_cpu_class_t>(picked_cpu),
        static_cast<uform_gpu_class_t>(picked_gpu),
        model.geometry.token_width,
        model.geometry.choice_temperature_groups,
    };
    propagate_error(uform_status_success_k, outcome, nullptr);
}

UFORM_EXPORT void uform_model_free(uform_model_t model) {
    finish_calls_using(model);
    delete model;
}

} // extern "C"
