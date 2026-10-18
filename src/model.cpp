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
#include <memory> // `std::unique_ptr` - a model under construction
#include <new>    // `std::nothrow`
#include <span>   // `std::span` - a checkpoint buffer

#include "lib.hpp"

using namespace unum::uform;
namespace ui = unum::uform::inference;

/** The C status a failed checkpoint read or tower pack crosses into: what the file holds is malformed, not an
 *  invalid argument. */
static uform_status_t load_status_of(core::status_t status) noexcept {
    return status == core::status_t::invalid_input_k ? uform_status_malformed_k : status_of(status);
}

/** The precision @p quantization names, false once @p outcome records that the blocks run in neither. */
static bool precision_of(uform_scalar_t quantization, ui::precision_t &precision, uform_outcome_t *outcome) noexcept {
    switch (quantization) {
    case uform_scalar_bf16_k: precision = ui::precision_t::bf16_k; return true;
    case uform_scalar_e4m3_k: precision = ui::precision_t::e4m3_k; return true;
    default: return propagate_error(uform_status_unsupported_k, outcome, "Blocks run in bf16 or e4m3 only");
    }
}

/** Bytes one tower of @p geometry holds in @p precision : the block products in its dtype, the entries, pool and
 *  head in BFloat16, the norm gains and pixel statistics widened to Float32, and the rotary angle grids. */
static std::size_t tower_bytes(ui::geometry_t const &geometry, ui::precision_t precision) noexcept {
    std::size_t const hidden = geometry.hidden, ffn = geometry.ffn();
    std::size_t const product_bytes = precision == ui::precision_t::e4m3_k ? 1 : 2;
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
static std::unique_ptr<uform_model_handle_t> start_model(core::expected<core::checkpoint_to_load_t> const &loaded,
                                                         ui::precision_t precision, uform_outcome_t *outcome) noexcept {
    if (!loaded)
        return propagate_error(load_status_of(loaded.status_), outcome, "The checkpoint could not be read"), nullptr;
    core::expected<ui::geometry_t> geometry = ui::geometry_t::from(*loaded);
    if (!geometry)
        return propagate_error(load_status_of(geometry.status_), outcome, "The checkpoint carries no usable geometry"),
               nullptr;
    std::unique_ptr<uform_model_handle_t> model {new (std::nothrow) uform_model_handle_t {}};
    if (!model) return propagate_error(uform_status_out_of_memory_k, outcome, "Failed to allocate the model"), nullptr;
    model->geometry = *geometry;
    model->precision = precision;
    return model;
}

/** Packs one tower of @p model from @p checkpoint for @p placement ; false once @p outcome records why not. */
static bool pack_tower(uform_model_handle_t &model, core::checkpoint_to_load_t const &checkpoint,
                       ui::placement_t const &placement, ui::tower_t &tower, uform_outcome_t *outcome) noexcept {
    core::expected<ui::tower_t> packed = ui::tower_t::make(checkpoint, model.geometry, model.precision, placement);
    if (!packed) return propagate_error(load_status_of(packed.status_), outcome, "The tower could not be packed");
    tower = std::move(*packed);
    model.memory_usage += tower_bytes(model.geometry, model.precision);
    return true;
}

/** The CPU capabilities a RAM load packs for: the executor's, or this machine's without one. */
static bool host_capabilities(uform_executor_t executor, nk_capability_t &capabilities,
                              uform_outcome_t *outcome) noexcept {
    if (!executor) {
        if (nk_cpu_capabilities_enabled(&capabilities) != nk_success_k)
            return propagate_error(uform_status_device_k, outcome, "The CPU capabilities could not be read");
        return true;
    }
    ram_executor_t const *const ram = std::get_if<ram_executor_t>(&executor->executor);
    if (!ram) return propagate_error(uform_status_device_k, outcome, "A RAM load needs a RAM executor or none");
    capabilities = ram->capabilities;
    return true;
}

/** Packs a RAM model from @p loaded and publishes it into @p output once the handler lets the load finish. */
static void load_ram(uform_model_t *output, uform_executor_t executor,
                     core::expected<core::checkpoint_to_load_t> loaded, ui::precision_t precision,
                     uform_progress_t progress, uform_outcome_t *outcome) noexcept {
    nk_capability_t capabilities = 0;
    if (!host_capabilities(executor, capabilities, outcome)) return;
    std::unique_ptr<uform_model_handle_t> model = start_model(loaded, precision, outcome);
    if (!model) return;
    ram_model_t &ram = model->residency.emplace<ram_model_t>();
    ram.capabilities = capabilities;
    if (!pack_tower(*model, *loaded, {capabilities, nullptr, {}}, ram.tower, outcome)) return;
    if (progress_ticks_t {progress, 1}(1) == continuation_t::cancel_k)
        return propagate_error(uform_status_cancelled_k, outcome, "The caller's handler stopped the load"), void();
    *output = model.release();
    propagate_error(uform_status_success_k, outcome, nullptr);
}

/** Packs a VRAM model from @p loaded on every device of @p executor that @p vram names, all of them when it names
 *  none, and publishes it into @p output . */
static void load_vram(uform_model_t *output, uform_executor_t executor, uform_vram_options_t const &vram,
                      core::expected<core::checkpoint_to_load_t> loaded, ui::precision_t precision,
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
    vram_model_t &replicated = model->residency.emplace<vram_model_t>();
    progress_ticks_t ticks {progress, devices->devices.size()};
    std::size_t placed = 0;
    for (vram_device_t &device : devices->devices) {
        std::size_t named = !vram.devices_count;
        for (std::size_t listed = 0; listed != vram.devices_count; ++listed)
            named += device.ordinal == vram.devices[listed];
        if (!named) continue;
        vram_replica_t replica {device.ordinal, device.capabilities, {}};
        if (!pack_tower(*model, *loaded, {device.capabilities, device.stream.get(), {}}, replica.tower, outcome))
            return;
        if (replicated.replicas.push_back(std::move(replica)) != core::status_t::success_k)
            return propagate_error(uform_status_out_of_memory_k, outcome, "Failed to list the replicas"), void();
        if (ticks(++placed, devices->devices.size()) == continuation_t::cancel_k)
            return propagate_error(uform_status_cancelled_k, outcome, "The caller's handler stopped the load"), void();
    }
    *output = model.release();
    propagate_error(uform_status_success_k, outcome, nullptr);
}

/** Maps the checkpoint file @p path , which the load closes before completing. */
static core::expected<core::checkpoint_to_load_t> map_checkpoint(uform_bytes_view_t path) noexcept {
    return core::checkpoint_to_load_t::make(text_of(path));
}

/** Views the checkpoint image @p buffer , borrowed only until the towers are packed. */
static core::expected<core::checkpoint_to_load_t> view_checkpoint(uform_bytes_view_t buffer) noexcept {
    return core::checkpoint_to_load_t::view(
        {static_cast<b8_t const *>(static_cast<void const *>(buffer.data)), buffer.length});
}

/** The refusals every load shares, clearing the output first so a refused load leaves no model. */
template <typename arguments_type_>
static bool rejects_load(arguments_type_ const &arguments, uform_bytes_view_t source, ui::precision_t &precision,
                         uform_outcome_t *outcome) noexcept {
    if (!arguments.model)
        return !propagate_error(uform_status_invalid_argument_k, outcome, "The model output must not be null");
    *arguments.model = nullptr;
    if (!source.data || !source.length)
        return !propagate_error(uform_status_invalid_argument_k, outcome, "The checkpoint must not be empty");
    return !precision_of(arguments.quantization, precision, outcome);
}

extern "C" {

UFORM_EXPORT void uform_model_load_ram(uform_model_load_ram_t *arguments) {
    if (submitted<uform_model_load_ram>(arguments)) return;
    call_outcome_t outcome(arguments->outcome);
    ui::precision_t precision {};
    if (rejects_load(*arguments, arguments->path, precision, outcome)) return;
    load_ram(arguments->model, arguments->executor, map_checkpoint(arguments->path), precision, arguments->progress,
             outcome);
}

UFORM_EXPORT void uform_model_load_ram_buffer(uform_model_load_ram_buffer_t *arguments) {
    if (submitted<uform_model_load_ram_buffer>(arguments)) return;
    call_outcome_t outcome(arguments->outcome);
    ui::precision_t precision {};
    if (rejects_load(*arguments, arguments->buffer, precision, outcome)) return;
    load_ram(arguments->model, arguments->executor, view_checkpoint(arguments->buffer), precision, arguments->progress,
             outcome);
}

UFORM_EXPORT void uform_model_load_vram(uform_model_load_vram_t *arguments) {
    if (submitted<uform_model_load_vram>(arguments)) return;
    call_outcome_t outcome(arguments->outcome);
    ui::precision_t precision {};
    if (rejects_load(*arguments, arguments->path, precision, outcome)) return;
    load_vram(arguments->model, arguments->executor, arguments->vram, map_checkpoint(arguments->path), precision,
              arguments->progress, outcome);
}

UFORM_EXPORT void uform_model_load_vram_buffer(uform_model_load_vram_buffer_t *arguments) {
    if (submitted<uform_model_load_vram_buffer>(arguments)) return;
    call_outcome_t outcome(arguments->outcome);
    ui::precision_t precision {};
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
    nk_capability_t const capabilities = std::visit(
        [](auto const &residency) noexcept -> nk_capability_t {
            if constexpr (requires { residency.replicas; })
                return residency.replicas.size() ? residency.replicas[0].capabilities : 0;
            else return residency.capabilities;
        },
        model.residency);
    char const *codename = "serial";
    for (std::size_t entry = 0; nk_capability_names_[entry].name; ++entry) // the last one held is the widest
        if (capabilities & nk_capability_names_[entry].flag) codename = nk_capability_names_[entry].name;
    *arguments->description = {
        model.geometry.hidden,
        model.geometry.context,
        model.precision == ui::precision_t::e4m3_k ? uform_scalar_e4m3_k : uform_scalar_bf16_k,
        model.memory_usage,
        codename,
    };
    propagate_error(uform_status_success_k, outcome, nullptr);
}

UFORM_EXPORT void uform_model_free(uform_model_t model) {
    finish_calls_using(model);
    delete model;
}

} // extern "C"
