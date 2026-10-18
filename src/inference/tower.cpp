/**
 *  @brief The forward-only encoder tower over NumKong — see the header for the contract.
 *  @file src/inference/tower.cpp
 *  @author Ash Vardanian
 *  @date October 17, 2026
 */
#include <cmath>   // `std::sqrt`, `std::fabs`, `std::fmax`
#include <cstring> // `std::memset`, `std::memcpy`

#include <algorithm> // `std::min`, `std::max`
#include <atomic>    // `std::atomic` — the first failure of a parallel step
#include <span>      // `std::span` — the host scalars make places

#include <numkong/attention.h> // `nk_attention_pack_*_best`, `nk_attention_packed_*_best`
#include <numkong/cast.h>      // `nk_cast_best`
#include <numkong/dots.h>      // `nk_dots_pack_*_best`, `nk_dots_packed_*_best`
#include <numkong/each.h>      // `nk_each_rmscast_bf16_best`, `nk_each_rmsnorm_bf16_best`, `nk_each_swiglu_*_best`
#include <numkong/memory.h>    // `nk_allocator_init_device_best`, `nk_allocator_init_heap`

#include "inference/tower.hpp"

namespace ui = unum::uform::inference;
using namespace ui;
using uc::divide_round_up;
using uc::status_t;

/** Rows one product tile spans, so threads share each streamed weight panel across a few dozen rows. */
static constexpr std::size_t product_tile_rows_k = 64;

/** The four block products, in @ref block_weights_t::products order. */
static constexpr std::size_t product_qkv_k = 0, product_out_k = 1, product_gate_up_k = 2, product_down_k = 3;

/** Lift a NumKong status into the tower's diagnostics, naming the step that returned it. */
[[nodiscard]] static expected_t checked_(nk_status_t status, sz::string_view_t what) noexcept {
    if (status == nk_success_k) return {};
    return expected_t::fail(status_t::other_k, "{} failed: {}", what, nk_status_name(status));
}

/** Run @p body over @p tasks in tiles of @p tile on every thread of @p pool , reporting the first NumKong failure. */
template <typename body_type_>
[[nodiscard]] static expected_t split_(fu::flat_pool_t &pool, std::size_t tasks, std::size_t tile,
                                       sz::string_view_t what, body_type_ const &body) noexcept {
    std::atomic<nk_status_t> failure {nk_success_k};
    pool.for_n(divide_round_up(tasks, tile), [&](std::size_t index, auto) noexcept {
        std::size_t const first = index * tile;
        nk_status_t const status = body(first, std::min(tile, tasks - first));
        if (status != nk_success_k) failure.store(status, std::memory_order_relaxed);
    });
    return checked_(failure.load(std::memory_order_relaxed), what);
}

/** The tile that gives every thread of @p pool one equal share of @p tasks . */
[[nodiscard]] static std::size_t share_(fu::flat_pool_t &pool, std::size_t tasks) noexcept {
    return std::max<std::size_t>(divide_round_up(tasks, std::max<std::size_t>(pool.threads_count(), 1)), 1);
}

/** A bump cursor over a call's scratch; over a null base it only measures. */
struct scratch_cursor_t {
    b8_t *base = nullptr;
    std::size_t used = 0;

    template <typename value_type_>
    [[nodiscard]] value_type_ *take(std::size_t count) noexcept {
        std::size_t const offset = used;
        used += divide_round_up(count * sizeof(value_type_), nk_default_alignment_k) * nk_default_alignment_k;
        return base ? static_cast<value_type_ *>(static_cast<void *>(base + offset)) : nullptr;
    }
};

/** Every activation one call writes, reserved in its scratch. The wide buffers carry the BFloat16 recipe's
 *  Float32 residual stream, the narrow ones the E4M3 recipe's BFloat16 stream and every operand. */
struct activations_t {
    u32_t *packed_offsets, *pool_offsets;
    i16_t *coords;
    f32_t *wide_residual, *wide_joined, *products, *wide_activated, *token_scales;
    nk_bf16_t *narrow_residual, *narrow_joined, *normalized, *qkv, *gate_up, *narrow_activated, *states, *patches;
    nk_e4m3_t *shadow;
    b8_t *keys_values;
    f32_t *wide_pooled, *wide_head;
    nk_bf16_t *pooled, *head;
};

/** Patch scalars of the widest present media modality, zero for a text-only tower. */
[[nodiscard]] static std::size_t patch_scalars_(geometry_t const &geometry) noexcept {
    std::size_t scalars = 0;
    if (geometry.image_patch_side) scalars = std::max<std::size_t>(scalars, geometry.image_scalars());
    if (geometry.video_patch_side) scalars = std::max<std::size_t>(scalars, geometry.video_scalars());
    if (geometry.audio_chunk_samples) scalars = std::max<std::size_t>(scalars, geometry.audio_chunk_samples);
    return scalars;
}

/** Reserve every activation for @p tokens rows in @p segments items behind @p cursor . */
[[nodiscard]] static activations_t reserve_(scratch_cursor_t &cursor, geometry_t const &geometry, std::size_t tokens,
                                            std::size_t segments, std::size_t keys_values_bytes) noexcept {
    std::size_t const hidden = geometry.hidden, ffn = geometry.ffn();
    std::size_t const widest = std::max<std::size_t>(3 * hidden, 2 * ffn);
    activations_t a;
    a.packed_offsets = cursor.take<u32_t>(segments + 1);
    a.pool_offsets = cursor.take<u32_t>(segments + 1);
    a.coords = cursor.take<i16_t>(tokens * 3);
    a.wide_residual = cursor.take<f32_t>(tokens * hidden);
    a.wide_joined = cursor.take<f32_t>(tokens * hidden);
    a.products = cursor.take<f32_t>(tokens * widest);
    a.wide_activated = cursor.take<f32_t>(tokens * ffn);
    a.token_scales = cursor.take<f32_t>(tokens);
    a.narrow_residual = cursor.take<nk_bf16_t>(tokens * hidden);
    a.narrow_joined = cursor.take<nk_bf16_t>(tokens * hidden);
    a.normalized = cursor.take<nk_bf16_t>(tokens * hidden);
    a.qkv = cursor.take<nk_bf16_t>(tokens * 3 * hidden);
    a.gate_up = cursor.take<nk_bf16_t>(tokens * 2 * ffn);
    a.narrow_activated = cursor.take<nk_bf16_t>(tokens * ffn);
    a.states = cursor.take<nk_bf16_t>(tokens * hidden);
    a.patches = cursor.take<nk_bf16_t>(tokens * patch_scalars_(geometry));
    a.shadow = cursor.take<nk_e4m3_t>(tokens * widest);
    a.keys_values = cursor.take<b8_t>(keys_values_bytes);
    a.wide_pooled = cursor.take<f32_t>(segments * hidden);
    a.wide_head = cursor.take<f32_t>(segments * hidden);
    a.pooled = cursor.take<nk_bf16_t>(segments * hidden);
    a.head = cursor.take<nk_bf16_t>(segments * hidden);
    return a;
}

/** The checkpoint tensor @p name , which must be @b [rows,columns] . */
[[nodiscard]] static expected<bf16r2_view_t> lookup_(uc::checkpoint_to_load_t const &checkpoint, sz::string_view_t name,
                                                     std::size_t rows, std::size_t columns) noexcept {
    bf16r2_view_t const tensor = checkpoint[name];
    if (tensor.empty() || tensor.extent(0) != rows || tensor.extent(1) != columns)
        return expected<bf16r2_view_t>::fail(status_t::invalid_input_k, "the checkpoint must hold {} as [{},{}]", name,
                                             rows, columns);
    return expected<bf16r2_view_t> {tensor};
}

/** Where @ref tower_t::make puts every weight: the tower's memory, packed for the tower's capabilities. */
struct placer_t {

    /** NumKong's aligned heap on the CPU, the device's own memory on a GPU. */
    nk::allocator<b8_t> allocator {};

    /** The vendor that copies host scalars into the device, null on the CPU. */
    device_kernels_t const *device = nullptr;
    nk_stream_t stream = nullptr;

    /** The mask every pack runs on, and the CPU mask the host-side casts run on. */
    nk_capability_t capabilities = 0, host = 0;

    /** Copy @p values into @p target , a fresh array in the tower's memory. */
    template <typename value_type_>
    [[nodiscard]] expected_t place(std::span<value_type_ const> values, placed_array<value_type_> &target,
                                   sz::string_view_t what) const noexcept {
        target = placed_array<value_type_> {nk::allocator<value_type_> {allocator}};
        if (values.empty()) return {};
        if (target.resize_uninitialized(values.size()) != status_t::success_k)
            return expected_t::fail(status_t::bad_alloc_k, "{} allocation failed ({} bytes)", what,
                                    values.size_bytes());
        if (device) return checked_(device->copy(target.data(), values.data(), values.size_bytes(), stream), what);
        std::memcpy(target.data(), values.data(), values.size_bytes());
        return {};
    }
};

/** Widen the checkpoint gain @p name of @p width scalars and place it into @p gain . */
[[nodiscard]] static expected_t widen_gain_(uc::checkpoint_to_load_t const &checkpoint, sz::string_view_t name,
                                            std::size_t width, placer_t const &placer,
                                            placed_array<f32_t> &gain) noexcept {
    expected<bf16r2_view_t> tensor = lookup_(checkpoint, name, 1, width);
    if (!tensor) return expected_t::fail(std::move(tensor));
    uc::array<f32_t> wide;
    if (wide.resize_uninitialized(width) != status_t::success_k)
        return expected_t::fail(status_t::bad_alloc_k, "{} widening failed", name);
    expected_t widened = checked_(
        nk_cast_best(tensor->data(), nk_bf16_k, wide.data(), nk_f32_k, width, placer.host, nullptr), name);
    if (!widened) return widened;
    return placer.place(std::span<f32_t const> {wide.data(), width}, gain, name);
}

/** Pack @p weight into @p packed for @p precision — E4M3 first divides by its own @c max_abs/240 descale. The
 *  operand is placed beside the pack first, as a device pack reads device memory. */
[[nodiscard]] static expected_t pack_(bf16r2_view_t weight, precision_t precision, placer_t const &placer,
                                      packed_weight_t &packed) noexcept {
    std::size_t const outputs = weight.extent(0), inputs = weight.extent(1), scalars = outputs * inputs;
    packed.outputs = outputs, packed.inputs = inputs;
    nk_size_t bytes = 0;
    nk_status_t status = precision == precision_t::bf16_k
                             ? nk_dots_pack_size_bf16_best(outputs, inputs, placer.capabilities, &bytes)
                             : nk_dots_pack_size_e4m3_best(outputs, inputs, placer.capabilities, &bytes);
    if (status != nk_success_k)
        return expected_t::fail(status_t::other_k, "pack sizing [{}x{}] failed: {}", outputs, inputs,
                                nk_status_name(status));
    packed.bytes = packed_bytes_t {placer.allocator};
    if (packed.bytes.resize_uninitialized(bytes) != status_t::success_k)
        return expected_t::fail(status_t::bad_alloc_k, "pack allocation failed ({} bytes)", bytes);
    if (precision == precision_t::bf16_k) {
        placed_array<nk_bf16_t> operand;
        expected_t placed = placer.place(std::span<nk_bf16_t const> {weight.data(), scalars}, operand, "pack operand");
        if (!placed) return placed;
        return checked_(nk_dots_pack_bf16_best(operand.data(), outputs, inputs, inputs * sizeof(nk_bf16_t),
                                               packed.bytes.data(), 0, outputs, placer.capabilities, placer.stream),
                        "packing");
    }
    uc::array<f32_t> wide;
    uc::array<nk_e4m3_t> shadow;
    if (wide.resize_uninitialized(scalars) != status_t::success_k ||
        shadow.resize_uninitialized(scalars) != status_t::success_k)
        return expected_t::fail(status_t::bad_alloc_k, "quantization staging failed ({} scalars)", scalars);
    status = nk_cast_best(weight.data(), nk_bf16_k, wide.data(), nk_f32_k, scalars, placer.host, nullptr);
    f32_t largest = 0.0f;
    for (f32_t const value : wide) largest = std::fmax(largest, std::fabs(value));
    packed.descale = std::fmax(largest / e4m3_range_k, e4m3_descale_floor_k);
    f32_t const inverse = 1.0f / packed.descale;
    for (f32_t &value : wide) value *= inverse;
    if (status == nk_success_k)
        status = nk_cast_best(wide.data(), nk_f32_k, shadow.data(), nk_e4m3_k, scalars, placer.host, nullptr);
    if (status != nk_success_k) return checked_(status, "quantization");
    placed_array<nk_e4m3_t> operand;
    expected_t placed = placer.place(std::span<nk_e4m3_t const> {shadow.data(), scalars}, operand, "pack operand");
    if (!placed) return placed;
    return checked_(nk_dots_pack_e4m3_best(operand.data(), outputs, inputs, inputs * sizeof(nk_e4m3_t),
                                           packed.bytes.data(), 0, outputs, placer.capabilities, placer.stream),
                    "packing");
}

/** Look up @p name as @b [rows,columns] and pack it. */
[[nodiscard]] static expected_t load_product_(uc::checkpoint_to_load_t const &checkpoint, sz::string_view_t name,
                                              std::size_t rows, std::size_t columns, precision_t precision,
                                              placer_t const &placer, packed_weight_t &packed) noexcept {
    expected<bf16r2_view_t> tensor = lookup_(checkpoint, name, rows, columns);
    if (!tensor) return expected_t::fail(std::move(tensor));
    return pack_(*tensor, precision, placer, packed);
}

expected<tower_t> tower_t::make(uc::checkpoint_to_load_t const &checkpoint, geometry_t const &geometry,
                                precision_t precision, placement_t const &placement) noexcept {
    using result_t = expected<tower_t>;
    placer_t placer;
    placer.capabilities = placement.capabilities;
    placer.stream = placement.stream;
    placer.host = placement.capabilities & nk_cap_cpus_k;
    nk_allocator_t policy {};
    nk_status_t initialized = nk_success_k;
    if (placement.capabilities & nk_cap_gpus_k) {
        placer.device = device_kernels_of(placement.capabilities);
        if (!placer.device)
            return result_t::fail(status_t::shape_unsupported_k, "this build compiles no kernels for that device");
        initialized = nk_allocator_init_device_best(&policy, placement.capabilities);
        if (initialized == nk_success_k) initialized = nk_cpu_capabilities_enabled(&placer.host);
    }
    else initialized = nk_allocator_init_heap(&policy, nk_default_alignment_k);
    if (initialized != nk_success_k)
        return result_t::fail(status_t::other_k, "tower memory setup failed: {}", nk_status_name(initialized));
    placer.allocator = nk::allocator<b8_t> {policy, placement.stream};

    std::size_t const hidden = geometry.hidden, ffn = geometry.ffn();
    tower_t tower;
    tower.geometry_ = geometry;
    tower.precision_ = precision;

    if (tower.blocks_.resize(geometry.depth) != status_t::success_k)
        return result_t::fail(status_t::bad_alloc_k, "block table allocation failed ({})", geometry.depth);
    sz::string_t name;
    sz::string_view_t const parts[4] = {"qkv", "out", "gate_up", "down"};
    std::size_t const extents[4][2] = {{3 * hidden, hidden}, {hidden, hidden}, {2 * ffn, hidden}, {hidden, ffn}};
    for (std::size_t index = 0; index != geometry.depth; ++index) {
        block_weights_t &block = tower.blocks_[index];
        expected_t loaded {};
        for (std::size_t product = 0; loaded && product != 4; ++product) {
            name.clear();
            if (!uc::format_into(name, "model.blocks.{}.{}", index, parts[product]))
                return result_t::fail(status_t::bad_alloc_k, "name staging failed");
            loaded = load_product_(checkpoint, name.view(), extents[product][0], extents[product][1], precision, placer,
                                   block.products[product]);
        }
        name.clear();
        if (loaded && !uc::format_into(name, "model.blocks.{}.pre_norm", index))
            return result_t::fail(status_t::bad_alloc_k, "name staging failed");
        if (loaded) loaded = widen_gain_(checkpoint, name.view(), hidden, placer, block.pre_norm);
        name.clear();
        if (loaded && !uc::format_into(name, "model.blocks.{}.post_norm", index))
            return result_t::fail(status_t::bad_alloc_k, "name staging failed");
        if (loaded) loaded = widen_gain_(checkpoint, name.view(), hidden, placer, block.post_norm);
        if (!loaded) return result_t::fail(std::move(loaded));
    }

    entry_weights_t &entries = tower.entries_;
    expected<bf16r2_view_t> embedding = lookup_(checkpoint, "model.embedding", geometry.vocabulary, hidden);
    if (!embedding) return result_t::fail(std::move(embedding));
    expected_t loaded = placer.place(std::span<nk_bf16_t const> {embedding->data(), embedding->numel()},
                                     entries.embedding, "embedding");
    if (loaded && geometry.image_patch_side)
        loaded = load_product_(checkpoint, "model.image_projection", hidden, geometry.image_scalars(),
                               precision_t::bf16_k, placer, entries.image);
    if (loaded && geometry.video_patch_side)
        loaded = load_product_(checkpoint, "model.video_projection", hidden, geometry.video_scalars(),
                               precision_t::bf16_k, placer, entries.video);
    if (loaded && geometry.audio_chunk_samples)
        loaded = load_product_(checkpoint, "model.audio_projection", hidden, geometry.audio_chunk_samples,
                               precision_t::bf16_k, placer, entries.audio);
    if (!loaded) return result_t::fail(std::move(loaded));
    f32_t inverse_stds[2][pixel_channels_max_k] = {};
    for (u32_t channel = 0; channel != geometry.image_channels; ++channel)
        inverse_stds[0][channel] = 1.0f / geometry.image_pixel_std[channel];
    for (u32_t channel = 0; channel != geometry.video_channels; ++channel)
        inverse_stds[1][channel] = 1.0f / geometry.video_pixel_std[channel];
    loaded = placer.place(std::span<f32_t const> {geometry.image_pixel_mean, geometry.image_channels},
                          entries.image_mean, "pixel statistics");
    if (loaded)
        loaded = placer.place(std::span<f32_t const> {inverse_stds[0], geometry.image_channels},
                              entries.image_inverse_std, "pixel statistics");
    if (loaded)
        loaded = placer.place(std::span<f32_t const> {geometry.video_pixel_mean, geometry.video_channels},
                              entries.video_mean, "pixel statistics");
    if (loaded)
        loaded = placer.place(std::span<f32_t const> {inverse_stds[1], geometry.video_channels},
                              entries.video_inverse_std, "pixel statistics");
    if (!loaded) return result_t::fail(std::move(loaded));

    pool_weights_t &pool = tower.pool_;
    expected<bf16r2_view_t> query = lookup_(checkpoint, "model.pool_query", 1, hidden);
    if (!query) return result_t::fail(std::move(query));
    loaded = placer.place(std::span<nk_bf16_t const> {query->data(), hidden}, pool.query, "pool query");
    if (loaded) loaded = widen_gain_(checkpoint, "model.final_norm", hidden, placer, pool.final_norm);
    if (loaded) loaded = widen_gain_(checkpoint, "model.head.norm", hidden, placer, pool.head_norm);
    if (loaded)
        loaded = load_product_(checkpoint, "model.head.fc1", hidden, hidden, precision_t::bf16_k, placer, pool.fc1);
    if (loaded)
        loaded = load_product_(checkpoint, "model.head.fc2", hidden, hidden, precision_t::bf16_k, placer, pool.fc2);
    if (!loaded) return result_t::fail(std::move(loaded));

    expected<rope_angles_t> angles = rope_angles_t::make(geometry.context, geometry.rope_temporal, geometry.rope_height,
                                                         geometry.rope_width, geometry.rope_base);
    if (!angles) return result_t::fail(std::move(angles));
    loaded = placer.place(angles->scalars(), tower.angles_, "angle grids");
    if (!loaded) return result_t::fail(std::move(loaded));
    tower.angle_grids_ = angles->grids(tower.angles_.data());

    if (placer.device) // ready once its packs land and the staged operands they read are freed on the stream
        loaded = checked_(nk_stream_synchronize_best(placement.capabilities, placement.stream), "load");
    if (!loaded) return result_t::fail(std::move(loaded));
    return result_t {std::move(tower)};
}

/** The K/V pack bytes the widest of the block attention and the pool needs. */
[[nodiscard]] static expected<std::size_t> keys_values_bytes_(geometry_t const &geometry, precision_t precision,
                                                              std::size_t tokens, std::size_t segments,
                                                              nk_capability_t capabilities) noexcept {
    nk_size_t block_bytes = 0, pool_bytes = 0;
    nk_status_t status = precision == precision_t::bf16_k
                             ? nk_attention_pack_size_bf16_best(geometry.heads(), geometry.head_width, tokens, segments,
                                                                capabilities, &block_bytes)
                             : nk_attention_pack_size_e4m3_best(geometry.heads(), geometry.head_width, tokens, segments,
                                                                capabilities, &block_bytes);
    if (status == nk_success_k)
        status = nk_attention_pack_size_bf16_best(geometry.heads(), geometry.head_width, tokens, segments, capabilities,
                                                  &pool_bytes);
    if (status != nk_success_k)
        return expected<std::size_t>::fail(status_t::other_k, "attention pack sizing failed: {}",
                                           nk_status_name(status));
    return expected<std::size_t> {std::max<std::size_t>(block_bytes, pool_bytes)};
}

expected<std::size_t> tower_t::scratch_bytes(std::size_t tokens, std::size_t segments,
                                             nk_capability_t capabilities) const noexcept {
    expected<std::size_t> keys_values = keys_values_bytes_(geometry_, precision_, tokens, segments, capabilities);
    if (!keys_values) return keys_values;
    scratch_cursor_t measure;
    (void)reserve_(measure, geometry_, tokens, segments, *keys_values);
    return expected<std::size_t> {measure.used};
}

/** The encode steps over host memory, split across a pool's threads. */
struct ram_executor_t {
    fu::flat_pool_t &pool;

    [[nodiscard]] std::size_t share(std::size_t tasks) noexcept { return share_(pool, tasks); }

    template <typename body_type_>
    [[nodiscard]] expected_t split(std::size_t tasks, std::size_t tile, sz::string_view_t what,
                                   body_type_ const &body) noexcept {
        return split_(pool, tasks, tile, what, body);
    }

    [[nodiscard]] expected_t zero(void *target, std::size_t bytes) noexcept {
        std::memset(target, 0, bytes);
        return {};
    }

    [[nodiscard]] expected_t fill_layout(u32r1_view_t lengths, u32_t *packed_offsets, u32_t *pool_offsets) noexcept {
        std::size_t const segments = lengths.extent(0);
        packed_offsets[0] = 0;
        for (std::size_t segment = 0; segment != segments; ++segment) {
            packed_offsets[segment + 1] = packed_offsets[segment] + lengths[segment];
            pool_offsets[segment] = static_cast<u32_t>(segment);
        }
        pool_offsets[segments] = static_cast<u32_t>(segments);
        return {};
    }

    /** The host @ref ui::mrope clamps each coordinate itself, so @p positions goes unused. */
    [[nodiscard]] expected_t fill_coords(segment_layout_t const &layout, nk::tensor_view<u16_t, 2> image_shapes,
                                         nk::tensor_view<u16_t, 2> video_shapes, u32_t tube_frames, u32_t frame_stride,
                                         u32_t, i16r2_span_t coords) noexcept {
        ui::fill_coords(layout, image_shapes, video_shapes, tube_frames, frame_stride, coords, pool);
        return {};
    }

    [[nodiscard]] expected_t gather_bytes(segment_layout_t const &layout, nk::tensor_view<token_id_t, 1> tokens,
                                          bf16r2_view_t table, bf16r2_span_t rows) noexcept {
        ui::gather_bytes(layout, tokens, table, rows, pool);
        return {};
    }

    [[nodiscard]] expected_t stage_patches(segment_layout_t const &layout, modality_t target,
                                           nk::tensor_view<pixel_t, 3> pixels, f32r1_view_t mean,
                                           f32r1_view_t inverse_std, bf16r2_span_t patches) noexcept {
        ui::stage_patches(layout, target, pixels, mean, inverse_std, patches, pool);
        return {};
    }

    [[nodiscard]] expected_t stage_frames(segment_layout_t const &layout, nk::tensor_view<f32_t, 2> samples,
                                          bf16r2_span_t frames) noexcept {
        ui::stage_frames(layout, samples, frames, pool);
        return {};
    }

    [[nodiscard]] expected_t mrope(bf16r2_span_t values, u32_t role_offset, u32_t heads, u32_t head_width,
                                   i16r2_view_t coords, rope_grids_t angles) noexcept {
        ui::mrope(values, role_offset, heads, head_width, coords, angles, pool);
        return {};
    }

    template <typename values_type_>
    [[nodiscard]] expected_t quantize(values_type_ values, e4m3r2_span_t shadow, f32r1_span_t token_scales) noexcept {
        ui::quantize_tokens_e4m3(values, shadow, token_scales, pool);
        return {};
    }

    [[nodiscard]] expected_t rescale(f32r2_view_t products, f32r1_view_t token_scales, f32_t weight_descale,
                                     bf16r2_view_t residual, bf16r2_span_t values) noexcept {
        ui::rescale_tokens(products, token_scales, weight_descale, residual, values, pool);
        return {};
    }
};

/** The encode steps over one device's memory, each queued once on its stream through its vendor's kernels. */
struct vram_executor_t {
    device_kernels_t const &kernels;
    nk_stream_t stream = nullptr;

    [[nodiscard]] std::size_t share(std::size_t tasks) noexcept { return std::max<std::size_t>(tasks, 1); }

    template <typename body_type_>
    [[nodiscard]] expected_t split(std::size_t tasks, std::size_t, sz::string_view_t what,
                                   body_type_ const &body) noexcept {
        return checked_(body(0, tasks), what);
    }

    [[nodiscard]] expected_t zero(void *target, std::size_t bytes) noexcept {
        return checked_(kernels.zero(target, bytes, stream), "zeroing");
    }

    [[nodiscard]] expected_t fill_layout(u32r1_view_t lengths, u32_t *packed_offsets, u32_t *pool_offsets) noexcept {
        return checked_(kernels.fill_layout(lengths, packed_offsets, pool_offsets, stream), "layout");
    }

    [[nodiscard]] expected_t fill_coords(segment_layout_t const &layout, nk::tensor_view<u16_t, 2> image_shapes,
                                         nk::tensor_view<u16_t, 2> video_shapes, u32_t tube_frames, u32_t frame_stride,
                                         u32_t positions, i16r2_span_t coords) noexcept {
        return checked_(kernels.fill_coords(layout, image_shapes, video_shapes, tube_frames, frame_stride, positions,
                                            coords, stream),
                        "coordinates");
    }

    [[nodiscard]] expected_t gather_bytes(segment_layout_t const &layout, nk::tensor_view<token_id_t, 1> tokens,
                                          bf16r2_view_t table, bf16r2_span_t rows) noexcept {
        return checked_(kernels.gather_bytes(layout, tokens, table, rows, stream), "byte gather");
    }

    [[nodiscard]] expected_t stage_patches(segment_layout_t const &layout, modality_t target,
                                           nk::tensor_view<pixel_t, 3> pixels, f32r1_view_t mean,
                                           f32r1_view_t inverse_std, bf16r2_span_t patches) noexcept {
        return checked_(kernels.stage_patches(layout, target, pixels, mean, inverse_std, patches, stream),
                        "patch staging");
    }

    [[nodiscard]] expected_t stage_frames(segment_layout_t const &layout, nk::tensor_view<f32_t, 2> samples,
                                          bf16r2_span_t frames) noexcept {
        return checked_(kernels.stage_frames(layout, samples, frames, stream), "frame staging");
    }

    [[nodiscard]] expected_t mrope(bf16r2_span_t values, u32_t role_offset, u32_t heads, u32_t head_width,
                                   i16r2_view_t coords, rope_grids_t angles) noexcept {
        return checked_(kernels.mrope(values, role_offset, heads, head_width, coords, angles, stream), "M-RoPE");
    }

    [[nodiscard]] expected_t quantize(bf16r2_view_t values, e4m3r2_span_t shadow, f32r1_span_t token_scales) noexcept {
        return checked_(kernels.quantize_bf16_tokens_e4m3(values, shadow, token_scales, stream), "quantization");
    }

    [[nodiscard]] expected_t quantize(f32r2_view_t values, e4m3r2_span_t shadow, f32r1_span_t token_scales) noexcept {
        return checked_(kernels.quantize_f32_tokens_e4m3(values, shadow, token_scales, stream), "quantization");
    }

    [[nodiscard]] expected_t rescale(f32r2_view_t products, f32r1_view_t token_scales, f32_t weight_descale,
                                     bf16r2_view_t residual, bf16r2_span_t values) noexcept {
        return checked_(kernels.rescale_tokens(products, token_scales, weight_descale, residual, values, stream),
                        "rescale");
    }
};

expected_t tower_t::encode(uc::multimodal_batch_view_t batch, outputs_t outputs,
                           placement_t const &placement) const noexcept {
    device_kernels_t const *const kernels = device_kernels_of(placement.capabilities);
    if (!kernels) return expected_t::fail(status_t::invalid_input_k, "a device encode takes a compiled device's mask");
    vram_executor_t executor {*kernels, placement.stream};
    return encode_(batch, outputs, placement, executor);
}

expected_t tower_t::encode(uc::multimodal_batch_view_t batch, outputs_t outputs, placement_t const &placement,
                           fu::flat_pool_t &pool) const noexcept {
    if (placement.capabilities & nk_cap_gpus_k)
        return expected_t::fail(status_t::invalid_input_k, "a RAM encode takes a host placement");
    ram_executor_t executor {pool};
    return encode_(batch, outputs, placement, executor);
}

template <typename executor_type_>
expected_t tower_t::encode_(uc::multimodal_batch_view_t batch, outputs_t outputs, placement_t const &placement,
                            executor_type_ &executor) const noexcept {
    std::size_t const segments = batch.segment_lengths.extent(0);
    if (segments == 0 || batch.segment_offsets.extent(0) < segments || batch.segment_modality.extent(0) != segments)
        return expected_t::fail(status_t::invalid_input_k, "a batch needs one offset, length and modality per item");
    if (outputs.embeddings == nullptr && outputs.states == nullptr)
        return expected_t::fail(status_t::invalid_input_k, "an encode keeps embeddings, states, or both");
    std::size_t tokens = 0;
    for (std::size_t segment = 0; segment != segments; ++segment) {
        if (batch.segment_lengths[segment] > geometry_.context)
            return expected_t::fail(status_t::invalid_input_k, "item {} spans {} cells, the context is {}", segment,
                                    batch.segment_lengths[segment], geometry_.context);
        tokens += batch.segment_lengths[segment];
    }
    if (tokens == 0) return expected_t::fail(status_t::invalid_input_k, "a batch needs a live cell");
    nk_capability_t const capabilities = placement.capabilities;
    expected<std::size_t> keys_values_bytes = keys_values_bytes_(geometry_, precision_, tokens, segments, capabilities);
    if (!keys_values_bytes) return expected_t::fail(std::move(keys_values_bytes));
    scratch_cursor_t cursor {placement.scratch.data()};
    activations_t const a = reserve_(cursor, geometry_, tokens, segments, *keys_values_bytes);
    if (placement.scratch.size() < cursor.used)
        return expected_t::fail(status_t::bad_alloc_k, "the scratch holds {} bytes, the call needs {}",
                                placement.scratch.size(), cursor.used);

    std::size_t const hidden = geometry_.hidden, ffn = geometry_.ffn(), heads = geometry_.heads();
    std::size_t const head_width = geometry_.head_width;
    f32_t const epsilon = geometry_.rms_epsilon;
    f32_t const scale = 1.0f / std::sqrt(static_cast<f32_t>(head_width));
    std::size_t const rows_share = executor.share(tokens);
    nk_stream_t const stream = placement.stream;

    auto cast = [&](void const *source, nk_dtype_t from, std::size_t from_bytes, void *target, nk_dtype_t to,
                    std::size_t to_bytes, std::size_t rows, std::size_t width) noexcept {
        return executor.split(rows, executor.share(rows), "cast", [&](std::size_t first, std::size_t count) noexcept {
            return nk_cast_best(static_cast<b8_t const *>(source) + first * width * from_bytes, from,
                                static_cast<b8_t *>(target) + first * width * to_bytes, to, count * width, capabilities,
                                stream);
        });
    };
    auto add = [&](f32_t *values, f32_t const *addend) noexcept {
        return executor.split(tokens, rows_share, "residual", [&](std::size_t first, std::size_t count) noexcept {
            f32_t *const slice = values + first * hidden;
            return nk_each_sum_f32_best(slice, addend + first * hidden, count * hidden, slice, capabilities, stream);
        });
    };
    auto rmscast = [&](f32_t const *source, f32_t const *gain, nk_bf16_t *target, std::size_t groups,
                       std::size_t columns, std::size_t source_width, std::size_t target_width) noexcept {
        return executor.split(tokens, rows_share, "rmscast", [&](std::size_t first, std::size_t count) noexcept {
            return nk_each_rmscast_bf16_best(source + first * source_width, gain, target + first * target_width, count,
                                             groups, columns, source_width * sizeof(f32_t),
                                             target_width * sizeof(nk_bf16_t), epsilon, capabilities, stream);
        });
    };
    auto rmsnorm = [&](nk_bf16_t const *source, f32_t const *gain, nk_bf16_t *target, std::size_t rows,
                       std::size_t groups, std::size_t columns, std::size_t width) noexcept {
        return executor.split(
            rows, executor.share(rows), "rmsnorm", [&](std::size_t first, std::size_t count) noexcept {
                return nk_each_rmsnorm_bf16_best(source + first * width, gain, target + first * width, count, groups,
                                                 columns, width * sizeof(nk_bf16_t), width * sizeof(nk_bf16_t), epsilon,
                                                 capabilities, stream);
            });
    };
    auto project = [&](packed_weight_t const &weight, precision_t precision, void const *activations, std::size_t rows,
                       f32_t *products) noexcept {
        std::size_t const outputs = weight.outputs, inputs = weight.inputs;
        return executor.split(rows, product_tile_rows_k, "product", [&](std::size_t first, std::size_t count) noexcept {
            f32_t *const tile = products + first * outputs;
            if (precision == precision_t::bf16_k)
                return nk_dots_packed_bf16_best(static_cast<nk_bf16_t const *>(activations) + first * inputs,
                                                weight.bytes.data(), tile, count, outputs, inputs,
                                                inputs * sizeof(nk_bf16_t), outputs * sizeof(f32_t), nullptr, 0,
                                                capabilities, stream);
            return nk_dots_packed_e4m3_best(
                static_cast<nk_e4m3_t const *>(activations) + first * inputs, weight.bytes.data(), tile, count, outputs,
                inputs, inputs * sizeof(nk_e4m3_t), outputs * sizeof(f32_t), nullptr, 0, capabilities, stream);
        });
    };

    expected_t staged = executor.fill_layout(batch.segment_lengths, a.packed_offsets, a.pool_offsets);
    segment_layout_t const layout {batch.segment_offsets,
                                   batch.segment_lengths,
                                   {a.packed_offsets, segments + 1},
                                   batch.segment_modality};
    i16r2_span_t const coords {a.coords, tokens, 3};
    if (staged) staged = executor.zero(a.coords, tokens * 3 * sizeof(i16_t));
    if (staged)
        staged = executor.fill_coords(layout, batch.image_shapes, batch.video_shapes, geometry_.video_tube_frames,
                                      geometry_.video_frame_stride, geometry_.context, coords);

    if (staged) staged = executor.zero(a.narrow_residual, tokens * hidden * sizeof(nk_bf16_t));
    if (staged)
        staged = executor.gather_bytes(layout, batch.tokens.flatten<1>(),
                                       {entries_.embedding.data(), geometry_.vocabulary, hidden},
                                       {a.narrow_residual, tokens, hidden});
    if (staged)
        staged = cast(a.narrow_residual, nk_bf16_k, sizeof(nk_bf16_t), a.wide_residual, nk_f32_k, sizeof(f32_t), tokens,
                      hidden);
    auto present = [&](modality_t modality, packed_weight_t const &weight, std::size_t content_scalars) noexcept {
        if (weight.bytes.empty() || content_scalars == 0) return false;
        for (std::size_t segment = 0; segment != segments; ++segment)
            if (batch.segment_modality[segment] == modality && batch.segment_lengths[segment] != 0) return true;
        return false;
    };
    auto add_projection = [&](packed_weight_t const &weight) noexcept {
        expected_t projected = project(weight, precision_t::bf16_k, a.patches, tokens, a.products);
        return projected ? add(a.wide_residual, a.products) : std::move(projected);
    };
    std::size_t media = 0;
    if (staged && present(modality_t::image_k, entries_.image, batch.images.numel())) {
        ++media;
        std::size_t const cells = batch.images.extent(0) * batch.images.extent(1);
        std::size_t const spatial = batch.images.extent(2) * batch.images.extent(3);
        staged = executor.zero(a.patches, tokens * geometry_.image_scalars() * sizeof(nk_bf16_t));
        if (staged)
            staged = executor.stage_patches(layout, modality_t::image_k,
                                            batch.images.reshape<3>({cells, spatial, batch.images.extent(4)}),
                                            {entries_.image_mean.data(), geometry_.image_channels},
                                            {entries_.image_inverse_std.data(), geometry_.image_channels},
                                            {a.patches, tokens, geometry_.image_scalars()});
        if (staged) staged = add_projection(entries_.image);
    }
    if (staged && present(modality_t::video_k, entries_.video, batch.videos.numel())) {
        ++media;
        std::size_t const cells = batch.videos.extent(0) * batch.videos.extent(1);
        std::size_t const spatial = batch.videos.extent(2) * batch.videos.extent(3) * batch.videos.extent(4);
        staged = executor.zero(a.patches, tokens * geometry_.video_scalars() * sizeof(nk_bf16_t));
        if (staged)
            staged = executor.stage_patches(layout, modality_t::video_k,
                                            batch.videos.reshape<3>({cells, spatial, batch.videos.extent(5)}),
                                            {entries_.video_mean.data(), geometry_.video_channels},
                                            {entries_.video_inverse_std.data(), geometry_.video_channels},
                                            {a.patches, tokens, geometry_.video_scalars()});
        if (staged) staged = add_projection(entries_.video);
    }
    if (staged && present(modality_t::audio_k, entries_.audio, batch.audio_samples.numel())) {
        ++media;
        std::size_t const cells = batch.audio_samples.extent(0) * batch.audio_samples.extent(1);
        std::size_t const samples = batch.audio_samples.extent(2);
        staged = executor.zero(a.patches, tokens * samples * sizeof(nk_bf16_t));
        if (staged)
            staged = executor.stage_frames(layout, batch.audio_samples.reshape<2>({cells, samples}),
                                           {a.patches, tokens, samples});
        if (staged) staged = add_projection(entries_.audio);
    }
    if (staged && media != 0 && precision_ == precision_t::e4m3_k)
        staged = cast(a.wide_residual, nk_f32_k, sizeof(f32_t), a.narrow_residual, nk_bf16_k, sizeof(nk_bf16_t), tokens,
                      hidden);
    if (!staged) return staged;

    auto attend = [&](void const *qkv, std::size_t scalar_bytes) noexcept {
        std::size_t const stride = 3 * hidden * scalar_bytes;
        b8_t const *const bytes = static_cast<b8_t const *>(qkv);
        expected_t packed = executor.split(
            segments * heads, executor.share(segments * heads), "attention pack",
            [&](std::size_t first, std::size_t count) noexcept {
                if (precision_ == precision_t::bf16_k)
                    return nk_attention_pack_bf16_best(
                        heads, head_width, a.packed_offsets, nullptr, segments,
                        static_cast<nk_bf16_t const *>(static_cast<void const *>(bytes + hidden * scalar_bytes)),
                        stride,
                        static_cast<nk_bf16_t const *>(static_cast<void const *>(bytes + 2 * hidden * scalar_bytes)),
                        stride, a.keys_values, first, first + count, capabilities, stream);
                return nk_attention_pack_e4m3_best(
                    heads, head_width, a.packed_offsets, nullptr, segments,
                    static_cast<nk_e4m3_t const *>(static_cast<void const *>(bytes + hidden * scalar_bytes)), stride,
                    static_cast<nk_e4m3_t const *>(static_cast<void const *>(bytes + 2 * hidden * scalar_bytes)),
                    stride, a.keys_values, first, first + count, capabilities, stream);
            });
        if (!packed) return packed;
        return executor.split(
            tokens * heads, executor.share(tokens * heads), "attention",
            [&](std::size_t first, std::size_t count) noexcept {
                if (precision_ == precision_t::bf16_k)
                    return nk_attention_packed_bf16_best(
                        heads, heads, head_width, a.packed_offsets, tokens, scale, NUMKONG_SIZE_MAX, NUMKONG_SIZE_MAX,
                        static_cast<nk_bf16_t const *>(qkv), stride, a.keys_values, a.products, hidden * sizeof(f32_t),
                        nullptr, first, first + count, nullptr, 0, capabilities, stream);
                return nk_attention_packed_e4m3_best(
                    heads, heads, head_width, a.packed_offsets, tokens, scale, NUMKONG_SIZE_MAX, NUMKONG_SIZE_MAX,
                    static_cast<nk_e4m3_t const *>(qkv), stride, a.keys_values, a.products, hidden * sizeof(f32_t),
                    nullptr, first, first + count, nullptr, 0, capabilities, stream);
            });
    };
    bf16r2_span_t const qkv {a.qkv, tokens, 3 * hidden};
    i16r2_view_t const coords_view {a.coords, tokens, 3};
    f32r1_span_t const token_scales {a.token_scales, tokens};
    f32r1_view_t const token_scales_view {a.token_scales, tokens};
    auto rotate = [&]() noexcept {
        expected_t rotated = executor.mrope(qkv, 0, heads, head_width, coords_view, angle_grids_);
        return rotated ? executor.mrope(qkv, hidden, heads, head_width, coords_view, angle_grids_) : std::move(rotated);
    };
    auto quantize = [&](nk_bf16_t const *values, std::size_t width) noexcept {
        return executor.quantize(bf16r2_view_t {values, tokens, width}, {a.shadow, tokens, width}, token_scales);
    };
    auto rescale = [&](block_weights_t const &block, std::size_t product, nk_bf16_t const *residual,
                       nk_bf16_t *values) noexcept {
        std::size_t const width = block.products[product].outputs;
        bf16r2_view_t const joined = residual ? bf16r2_view_t {residual, tokens, width} : bf16r2_view_t {};
        return executor.rescale({a.products, tokens, width}, token_scales_view, block.products[product].descale, joined,
                                {values, tokens, width});
    };

    for (block_weights_t const &block : blocks_) {
        packed_weight_t const *const products = block.products;
        if (precision_ == precision_t::bf16_k) {
            staged = rmscast(a.wide_residual, block.pre_norm.data(), a.normalized, 1, hidden, hidden, hidden);
            if (staged) staged = project(products[product_qkv_k], precision_, a.normalized, tokens, a.products);
            if (staged)
                staged = cast(a.products, nk_f32_k, sizeof(f32_t), a.qkv, nk_bf16_k, sizeof(nk_bf16_t), tokens,
                              3 * hidden);
            if (staged) staged = rmscast(a.products, nullptr, a.qkv, 2 * heads, head_width, 3 * hidden, 3 * hidden);
            if (staged) staged = rotate();
            if (staged) staged = attend(a.qkv, sizeof(nk_bf16_t));
            if (staged)
                staged = cast(a.products, nk_f32_k, sizeof(f32_t), a.normalized, nk_bf16_k, sizeof(nk_bf16_t), tokens,
                              hidden);
            if (staged) staged = project(products[product_out_k], precision_, a.normalized, tokens, a.wide_joined);
            if (staged) staged = add(a.wide_joined, a.wide_residual);
            if (staged)
                staged = rmscast(a.wide_joined, block.post_norm.data(), a.normalized, 1, hidden, hidden, hidden);
            if (staged) staged = project(products[product_gate_up_k], precision_, a.normalized, tokens, a.products);
            if (staged)
                staged = executor.split(
                    tokens, rows_share, "SwiGLU", [&](std::size_t first, std::size_t count) noexcept {
                        f32_t const *const gate = a.products + first * 2 * ffn;
                        return nk_each_swiglu_f32_best(gate, gate + ffn, a.wide_activated + first * ffn, count, ffn,
                                                       2 * ffn * sizeof(f32_t), 2 * ffn * sizeof(f32_t),
                                                       ffn * sizeof(f32_t), 1.0f, 1.0f, capabilities, stream);
                    });
            if (staged)
                staged = cast(a.wide_activated, nk_f32_k, sizeof(f32_t), a.narrow_activated, nk_bf16_k,
                              sizeof(nk_bf16_t), tokens, ffn);
            if (staged)
                staged = project(products[product_down_k], precision_, a.narrow_activated, tokens, a.wide_residual);
            if (staged) staged = add(a.wide_residual, a.wide_joined);
            if (!staged) return staged;
            continue;
        }
        staged = rmsnorm(a.narrow_residual, block.pre_norm.data(), a.normalized, tokens, 1, hidden, hidden);
        if (staged) staged = quantize(a.normalized, hidden);
        if (staged) staged = project(products[product_qkv_k], precision_, a.shadow, tokens, a.products);
        if (staged) staged = rescale(block, product_qkv_k, nullptr, a.qkv);
        if (staged)
            staged = executor.split(tokens, rows_share, "QK-norm", [&](std::size_t first, std::size_t count) noexcept {
                nk_bf16_t *const rows = a.qkv + first * 3 * hidden;
                return nk_each_rmsnorm_bf16_best(rows, nullptr, rows, count, 2 * heads, head_width,
                                                 3 * hidden * sizeof(nk_bf16_t), 3 * hidden * sizeof(nk_bf16_t),
                                                 epsilon, capabilities, stream);
            });
        if (staged) staged = rotate();
        if (staged)
            staged = cast(a.qkv, nk_bf16_k, sizeof(nk_bf16_t), a.shadow, nk_e4m3_k, sizeof(nk_e4m3_t), tokens,
                          3 * hidden);
        if (staged) staged = attend(a.shadow, sizeof(nk_e4m3_t));
        if (staged)
            staged = executor.quantize(f32r2_view_t {a.products, tokens, hidden}, {a.shadow, tokens, hidden},
                                       token_scales);
        if (staged) staged = project(products[product_out_k], precision_, a.shadow, tokens, a.products);
        if (staged) staged = rescale(block, product_out_k, a.narrow_residual, a.narrow_joined);
        if (staged) staged = rmsnorm(a.narrow_joined, block.post_norm.data(), a.normalized, tokens, 1, hidden, hidden);
        if (staged) staged = quantize(a.normalized, hidden);
        if (staged) staged = project(products[product_gate_up_k], precision_, a.shadow, tokens, a.products);
        if (staged) staged = rescale(block, product_gate_up_k, nullptr, a.gate_up);
        if (staged)
            staged = executor.split(tokens, rows_share, "SwiGLU", [&](std::size_t first, std::size_t count) noexcept {
                nk_bf16_t const *const gate = a.gate_up + first * 2 * ffn;
                return nk_each_swiglu_bf16_best(gate, gate + ffn, a.narrow_activated + first * ffn, count, ffn,
                                                2 * ffn * sizeof(nk_bf16_t), 2 * ffn * sizeof(nk_bf16_t),
                                                ffn * sizeof(nk_bf16_t), 1.0f, 1.0f, capabilities, stream);
            });
        if (staged) staged = quantize(a.narrow_activated, ffn);
        if (staged) staged = project(products[product_down_k], precision_, a.shadow, tokens, a.products);
        if (staged) staged = rescale(block, product_down_k, a.narrow_joined, a.narrow_residual);
        if (!staged) return staged;
    }

    staged = precision_ == precision_t::bf16_k
                 ? rmscast(a.wide_residual, pool_.final_norm.data(), a.states, 1, hidden, hidden, hidden)
                 : rmsnorm(a.narrow_residual, pool_.final_norm.data(), a.states, tokens, 1, hidden, hidden);
    if (staged && outputs.states != nullptr)
        staged = cast(a.states, nk_bf16_k, sizeof(nk_bf16_t), outputs.states, outputs.dtype,
                      nk_dtype_bits(outputs.dtype) / 8, tokens, hidden);
    if (!staged || outputs.embeddings == nullptr) return staged;

    std::size_t const segment_share = executor.share(segments * heads);
    staged = executor.split(
        segments * heads, segment_share, "pool pack", [&](std::size_t first, std::size_t count) noexcept {
            return nk_attention_pack_bf16_best(heads, head_width, a.packed_offsets, nullptr, segments, a.states,
                                               hidden * sizeof(nk_bf16_t), a.states, hidden * sizeof(nk_bf16_t),
                                               a.keys_values, first, first + count, capabilities, stream);
        });
    if (staged)
        staged = executor.split(
            segments * heads, segment_share, "pool", [&](std::size_t first, std::size_t count) noexcept {
                return nk_attention_packed_bf16_best(heads, heads, head_width, a.pool_offsets, segments, scale,
                                                     NUMKONG_SIZE_MAX, NUMKONG_SIZE_MAX, pool_.query.data(), 0,
                                                     a.keys_values, a.wide_pooled, hidden * sizeof(f32_t), nullptr,
                                                     first, first + count, nullptr, 0, capabilities, stream);
            });

    if (staged)
        staged = cast(a.wide_pooled, nk_f32_k, sizeof(f32_t), a.pooled, nk_bf16_k, sizeof(nk_bf16_t), segments, hidden);
    if (staged) staged = rmsnorm(a.pooled, pool_.head_norm.data(), a.head, segments, 1, hidden, hidden);
    if (staged) staged = project(pool_.fc1, precision_t::bf16_k, a.head, segments, a.wide_head);
    if (staged)
        staged = cast(a.wide_head, nk_f32_k, sizeof(f32_t), a.head, nk_bf16_k, sizeof(nk_bf16_t), segments, hidden);
    if (staged)
        staged = executor.split(
            segments, executor.share(segments), "SiLU", [&](std::size_t first, std::size_t count) noexcept {
                nk_bf16_t *const rows = a.head + first * hidden;
                return nk_each_swiglu_bf16_best(rows, nullptr, rows, count, hidden, hidden * sizeof(nk_bf16_t), 0,
                                                hidden * sizeof(nk_bf16_t), 1.0f, 1.0f, capabilities, stream);
            });
    if (staged) staged = project(pool_.fc2, precision_t::bf16_k, a.head, segments, a.wide_head);
    if (staged)
        staged = cast(a.wide_head, nk_f32_k, sizeof(f32_t), outputs.embeddings, outputs.dtype,
                      nk_dtype_bits(outputs.dtype) / 8, segments, hidden);
    return staged;
}
