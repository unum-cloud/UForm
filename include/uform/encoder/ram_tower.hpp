/**
 *  @file include/uform/encoder/ram_tower.hpp
 *  @author Ash Vardanian
 *  @date October 25, 2026
 *  @brief The encoder tower in host memory: weights packed by the products kit it is handed, every
 *      step of @ref encode_steps split across a thread pool.
 *
 *  Products run in 64-row tiles, attention in equal task windows, element-wise steps in row
 *  slices, and the host entries of @ref ram_kits.hpp one row or one segment per task, each in its
 *  thread's Float32 row of the call's scratch. The tower names no class and no capability: the
 *  class unit instantiating it hands it its kits and its atomic reference.
 */
#pragma once

#include <cstring> // `std::memcpy`, `std::memset`

#include <algorithm> // `std::max`, `std::min`
#include <span>      // `std::span` — the scratch a call borrows

#include "uform/encoder/ram_kits.hpp"     // the host entries
#include "uform/encoder/tower.hpp"        // `tower_weights_t`, `pack_weights`, `encode_steps`
#include "uform/utilities/ram_device.hpp" // `pool_view_t`, the threads every step splits across

namespace unum::uform::encoder {

/** The Float32 rows the host entries work in, reserved in a call's scratch before its
 *  activations. */
struct entries_rows_t {

    /** @b [threads,width] each thread's row, as wide as the widest row an entry widens. */
    f32r2_span_t workers {};

    /** @b [2·patch_scalars] the gain and bias rows of @ref fill_patch_affine , refilled per staged
     *  modality. */
    f32r1_span_t affine {};
};

/** Reserve the entries rows of @p threads threads for a tower of @p geometry behind @p cursor :
 *  each as wide as the widest block product, which also holds M-RoPE's angle, cosine and sine rows
 *  and a widened head behind them, or the widest patch. */
[[nodiscard]] inline static entries_rows_t reserve_entries(scratch_cursor_t &cursor, geometry_t const &geometry,
                                                           std::size_t threads) noexcept {
    std::size_t const patch = patch_scalars(geometry);
    std::size_t const width = std::max<std::size_t>({3 * geometry.hidden, 2 * geometry.ffn(), patch});
    return {cursor.take<f32_t>(threads, width), cursor.take<f32_t>(2 * patch)};
}

/** Places a tower's arrays in NumKong's aligned heap, copying with @c memcpy . */
struct ram_placer_t {
    nk::allocator<b8_t> allocator {};
    nk_stream_t stream = nullptr;

    /** Copy @p values into @p target , a fresh array in host memory. */
    template <typename value_type_>
    [[nodiscard]] expected_t place(std::span<value_type_ const> values, placed_array<value_type_> &target,
                                   sz::string_view_t what) const noexcept {
        target = placed_array<value_type_> {nk::allocator<value_type_> {allocator}};
        if (values.empty()) return {};
        if (st::failed(target.resize(st::assume_uninitialized, values.size())))
            return expected_t::fail(uu::status_t::bad_alloc_k, "{} allocation failed ({} bytes)", what,
                                    values.size_bytes());
        std::memcpy(target.data(), values.data(), values.size_bytes());
        return {};
    }
};

/** The steps of @ref encode_steps over host memory, split across a pool's threads, through the
 *  kits a CPU class picked; a step's first failure latches through @p atomic_ref_template_ . The
 *  host entries are @c inline @c static templates over the element-wise kit, so no two class units
 *  share a copy. */
template <typename products_kit_type_, typename attention_kit_type_, typename elementwise_kit_type_,
          template <typename> class atomic_ref_template_>
struct ram_steps {
    using products_kit_t = products_kit_type_;
    using attention_kit_t = attention_kit_type_;
    using elementwise_kit_t = elementwise_kit_type_;

    uu::pool_view_t pool;
    entries_rows_t rows;

    [[nodiscard]] nk_stream_t stream() const noexcept { return nullptr; }

    /** The tile that gives every thread one equal share of @p tasks . */
    [[nodiscard]] std::size_t share(std::size_t tasks) noexcept {
        return std::max<std::size_t>(uu::divide_round_up(tasks, std::max<std::size_t>(pool.threads_count(), 1)), 1);
    }

    /** Run @p body once per task of @p tasks , each with its thread's row of
     *  @ref entries_rows_t::workers , reporting the first NumKong failure. */
    template <typename body_type_>
    [[nodiscard]] expected_t each(std::size_t tasks, sz::string_view_t what, body_type_ const &body) noexcept {
        std::size_t const width = rows.workers.extent(1);
        nk_status_t failure = nk_success_k;
        uu::status_t const status = pool.for_n(tasks, [&](std::size_t task, std::size_t thread) noexcept {
            nk_status_t const step = body(task, f32r1_span_t {rows.workers.row(thread).data(), width});
            if (step != nk_success_k)
                atomic_ref_template_<nk_status_t> {failure}.store(step, std::memory_order_relaxed);
        });
        if (expected_t refused = dispatched(status, what); !refused) return refused;
        return checked(failure, what); // `for_n` joined every worker
    }

    /** Run @p body over @p tasks in tiles of @p tile on every thread, reporting the first NumKong
     *  failure. */
    template <typename body_type_>
    [[nodiscard]] expected_t split(std::size_t tasks, std::size_t tile, sz::string_view_t what,
                                   body_type_ const &body) noexcept {
        return each(uu::divide_round_up(tasks, tile), what, [&](std::size_t index, f32r1_span_t) noexcept {
            std::size_t const first = index * tile;
            return body(first, std::min(tile, tasks - first));
        });
    }

    [[nodiscard]] expected_t zero(uu::bytes_span_t target) noexcept {
        std::memset(target.data(), 0, target.size());
        return {};
    }

    [[nodiscard]] expected_t copy(uu::bytes_span_t target, uu::bytes_view_t source) noexcept {
        std::memcpy(target.data(), source.data(), source.size());
        return {};
    }

    [[nodiscard]] expected_t fill_layout(u32r1_view_t lengths, u32r1_span_t packed_offsets,
                                         u32r1_span_t pool_offsets) noexcept {
        std::size_t const segments = lengths.extent(0);
        packed_offsets[0] = 0;
        for (std::size_t segment = 0; segment != segments; ++segment) {
            packed_offsets[segment + 1] = packed_offsets[segment] + lengths[segment];
            pool_offsets[segment] = static_cast<u32_t>(segment);
        }
        pool_offsets[segments] = static_cast<u32_t>(segments);
        return {};
    }

    /** The host @ref encoder::mrope clamps each coordinate itself, so @p positions goes unused. */
    [[nodiscard]] expected_t fill_coords(segment_layout_t const &layout, nk::tensor_view<u16_t, 2> image_shapes,
                                         nk::tensor_view<u16_t, 2> video_shapes, u32_t tube_frames, u32_t frame_stride,
                                         u32_t, i16r2_span_t coords) noexcept {
        return dispatched(
            encoder::fill_coords(layout, image_shapes, video_shapes, tube_frames, frame_stride, coords, pool),
            "coordinate fill");
    }

    [[nodiscard]] expected_t gather_bytes(segment_layout_t const &layout, nk::tensor_view<token_id_t, 1> tokens,
                                          bf16r2_view_t table, bf16r2_span_t rows) noexcept {
        return dispatched(encoder::gather_bytes(layout, tokens, table, rows, pool), "byte gather");
    }

    /** @p status of a pool dispatch for step @p what , a failure naming the step. */
    [[nodiscard]] static expected_t dispatched(uu::status_t status, sz::string_view_t what) noexcept {
        if (status == uu::status_t::success_k) return {};
        return expected_t::fail(status, "{} dispatch was refused by the pool", what);
    }

    [[nodiscard]] expected_t stage_patches(segment_layout_t const &layout, modality_t target,
                                           nk::tensor_view<pixel_t, 3> pixels, f32r1_view_t mean,
                                           f32r1_view_t inverse_std, bf16r2_span_t patches) noexcept {
        f32r2_span_t const affine {rows.affine.data(), 2, pixels.extent(1) * pixels.extent(2)};
        fill_patch_affine(mean, inverse_std, affine);
        return each(layout.lengths.extent(0), "patch staging", [&](std::size_t segment, f32r1_span_t wide) noexcept {
            return encoder::stage_patches<elementwise_kit_t>(layout, segment, target, pixels, affine, wide, patches);
        });
    }

    [[nodiscard]] expected_t stage_frames(segment_layout_t const &layout, nk::tensor_view<f32_t, 2> samples,
                                          bf16r2_span_t frames) noexcept {
        return each(layout.lengths.extent(0), "frame staging", [&](std::size_t segment, f32r1_span_t) noexcept {
            return encoder::stage_frames<elementwise_kit_t>(layout, segment, samples, frames);
        });
    }

    [[nodiscard]] expected_t mrope(bf16r2_span_t values, u32_t role_offset, u32_t heads, u32_t head_width,
                                   i16r2_view_t coords, rope_grids_t angles) noexcept {
        return each(values.extent(0), "M-RoPE", [&](std::size_t row, f32r1_span_t wide) noexcept {
            return encoder::mrope<elementwise_kit_t>(values, row, role_offset, heads, head_width, coords, angles, wide);
        });
    }

    template <typename values_type_>
    [[nodiscard]] expected_t quantize(values_type_ values, e4m3r2_span_t shadow, f32r1_span_t token_scales) noexcept {
        return each(values.extent(0), "quantization", [&](std::size_t row, f32r1_span_t wide) noexcept {
            return encoder::quantize_tokens_e4m3<elementwise_kit_t>(values, row, shadow, token_scales, wide);
        });
    }

    [[nodiscard]] expected_t rescale(f32r2_view_t products, f32r1_view_t token_scales, f32_t weight_descale,
                                     bf16r2_view_t residual, bf16r2_span_t values) noexcept {
        return each(values.extent(0), "rescale", [&](std::size_t row, f32r1_span_t wide) noexcept {
            return encoder::rescale_tokens<elementwise_kit_t>(products, row, token_scales, weight_descale, residual,
                                                              values, wide);
        });
    }
};

/** The forward-only encoder on host memory, its products, attention and element-wise steps from
 *  the kits it is handed, and its shared words through @p atomic_ref_template_ . */
template <typename products_kit_type_, typename attention_kit_type_, typename elementwise_kit_type_,
          template <typename> class atomic_ref_template_>
struct ram_tower {
    using steps_t = ram_steps<products_kit_type_, attention_kit_type_, elementwise_kit_type_, atomic_ref_template_>;

    /**
     *  @brief Pack every weight of @p checkpoint into NumKong's aligned heap, in @p precision for
     *      the block products.
     *  @retval status_t::invalid_argument_k A tensor is missing or misshapen.
     */
    [[nodiscard]] static expected<ram_tower> make(uu::checkpoint_to_load_t const &checkpoint,
                                                  geometry_t const &geometry, precision_t precision) noexcept {
        nk_allocator_t policy {};
        if (nk_status_t const status = nk_allocator_init_heap(&policy, nk_default_alignment_k); status != nk_success_k)
            return expected<ram_tower>::fail(tower_status_of(status), "tower memory setup failed: {}",
                                             nk_status_name(status));
        ram_placer_t const placer {nk::allocator<b8_t> {policy, nullptr}, nullptr};
        expected<tower_weights_t> weights = pack_weights<products_kit_type_, elementwise_kit_type_>(
            checkpoint, geometry, precision, placer);
        if (!weights) return expected<ram_tower>::fail(std::move(weights));
        ram_tower tower;
        tower.weights_ = std::move(*weights);
        return expected<ram_tower> {std::move(tower)};
    }

    /** The scratch bytes one call over @p tokens live tokens in @p segments items needs on
     *  @p threads threads: their entries rows, then the activations. */
    [[nodiscard]] expected<std::size_t> scratch_bytes(std::size_t tokens, std::size_t segments,
                                                      std::size_t threads) const noexcept {
        scratch_cursor_t measure;
        [[maybe_unused]] entries_rows_t const unplaced = reserve_entries(measure, weights_.geometry, threads);
        expected<std::size_t> activations = scratch_bytes_of<attention_kit_type_>(weights_, tokens, segments);
        if (!activations) return activations;
        return expected<std::size_t> {measure.used + *activations};
    }

    /** Encode @p batch in host memory on @p pool 's threads. */
    [[nodiscard]] expected_t encode(multimodal_batch_view_t batch, outputs_t outputs, bytes_span_t scratch,
                                    uu::pool_view_t pool) const noexcept {
        scratch_cursor_t cursor {scratch};
        steps_t steps {pool, reserve_entries(cursor, weights_.geometry, pool.threads_count())};
        if (scratch.size() < cursor.used)
            return expected_t::fail(uu::status_t::bad_alloc_k, "the scratch holds {} bytes, the entries need {}",
                                    scratch.size(), cursor.used);
        return encode_steps(steps, weights_, batch, outputs, scratch.subspan(cursor.used));
    }

    [[nodiscard]] tower_weights_t const &weights() const noexcept { return weights_; }

  private:
    tower_weights_t weights_ {};
};

} // namespace unum::uform::encoder
