/**
 *  @file include/uform/encoder/ram_tower.hpp
 *  @author Ash Vardanian
 *  @date October 25, 2026
 *  @brief The encoder tower in host memory: weights packed by the products kit it is handed, every
 *      step of @ref encode_steps split across a thread pool.
 *
 *  Products run in 64-row tiles, attention in equal task windows, element-wise steps in row
 *  slices. The steps NumKong has no kernel for are the host loops of @ref entries.hpp . The tower
 *  names no class and no capability: the class unit instantiating it hands it its kits.
 */
#pragma once

#include <cstring> // `std::memcpy`, `std::memset`

#include <atomic> // `std::atomic` — the first failure of a parallel step
#include <span>   // `std::span` — the scratch a call borrows

#include <forkunion.hpp> // vendor-boundary: `fu::flat_pool_t`, the threads every step splits across

#include "uform/encoder/tower.hpp" // `tower_weights_t`, `pack_weights`, `encode_steps`

namespace unum::uform::encoder {

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
            return expected_t::fail(uc::status_t::bad_alloc_k, "{} allocation failed ({} bytes)", what,
                                    values.size_bytes());
        std::memcpy(target.data(), values.data(), values.size_bytes());
        return {};
    }
};

/** The steps of @ref encode_steps over host memory, split across a pool's threads, through the
 *  kits a CPU class picked. */
template <typename products_kit_type_, typename attention_kit_type_, typename elementwise_kit_type_>
struct ram_steps {
    using products_kit_t = products_kit_type_;
    using attention_kit_t = attention_kit_type_;
    using elementwise_kit_t = elementwise_kit_type_;

    fu::flat_pool_t &pool;

    [[nodiscard]] nk_stream_t stream() const noexcept { return nullptr; }

    /** The tile that gives every thread one equal share of @p tasks . */
    [[nodiscard]] std::size_t share(std::size_t tasks) noexcept {
        return std::max<std::size_t>(uc::divide_round_up(tasks, std::max<std::size_t>(pool.threads_count(), 1)), 1);
    }

    /** Run @p body over @p tasks in tiles of @p tile on every thread, reporting the first NumKong
     *  failure. */
    template <typename body_type_>
    [[nodiscard]] expected_t split(std::size_t tasks, std::size_t tile, sz::string_view_t what,
                                   body_type_ const &body) noexcept {
        std::atomic<nk_status_t> failure {nk_success_k};
        pool.for_n(uc::divide_round_up(tasks, tile), [&](std::size_t index, auto) noexcept {
            std::size_t const first = index * tile;
            nk_status_t const status = body(first, std::min(tile, tasks - first));
            if (status != nk_success_k) failure.store(status, std::memory_order_relaxed);
        });
        return checked(failure.load(std::memory_order_relaxed), what);
    }

    [[nodiscard]] expected_t zero(void *target, std::size_t bytes) noexcept {
        std::memset(target, 0, bytes);
        return {};
    }

    [[nodiscard]] expected_t copy(void *target, void const *source, std::size_t bytes) noexcept {
        std::memcpy(target, source, bytes);
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

    /** The host @ref encoder::mrope clamps each coordinate itself, so @p positions goes unused. */
    [[nodiscard]] expected_t fill_coords(segment_layout_t const &layout, nk::tensor_view<u16_t, 2> image_shapes,
                                         nk::tensor_view<u16_t, 2> video_shapes, u32_t tube_frames, u32_t frame_stride,
                                         u32_t, i16r2_span_t coords) noexcept {
        encoder::fill_coords(layout, image_shapes, video_shapes, tube_frames, frame_stride, coords, pool);
        return {};
    }

    [[nodiscard]] expected_t gather_bytes(segment_layout_t const &layout, nk::tensor_view<token_id_t, 1> tokens,
                                          bf16r2_view_t table, bf16r2_span_t rows) noexcept {
        encoder::gather_bytes(layout, tokens, table, rows, pool);
        return {};
    }

    [[nodiscard]] expected_t stage_patches(segment_layout_t const &layout, modality_t target,
                                           nk::tensor_view<pixel_t, 3> pixels, f32r1_view_t mean,
                                           f32r1_view_t inverse_std, bf16r2_span_t patches) noexcept {
        encoder::stage_patches(layout, target, pixels, mean, inverse_std, patches, pool);
        return {};
    }

    [[nodiscard]] expected_t stage_frames(segment_layout_t const &layout, nk::tensor_view<f32_t, 2> samples,
                                          bf16r2_span_t frames) noexcept {
        encoder::stage_frames(layout, samples, frames, pool);
        return {};
    }

    [[nodiscard]] expected_t mrope(bf16r2_span_t values, u32_t role_offset, u32_t heads, u32_t head_width,
                                   i16r2_view_t coords, rope_grids_t angles) noexcept {
        encoder::mrope(values, role_offset, heads, head_width, coords, angles, pool);
        return {};
    }

    template <typename values_type_>
    [[nodiscard]] expected_t quantize(values_type_ values, e4m3r2_span_t shadow, f32r1_span_t token_scales) noexcept {
        encoder::quantize_tokens_e4m3(values, shadow, token_scales, pool);
        return {};
    }

    [[nodiscard]] expected_t rescale(f32r2_view_t products, f32r1_view_t token_scales, f32_t weight_descale,
                                     bf16r2_view_t residual, bf16r2_span_t values) noexcept {
        encoder::rescale_tokens(products, token_scales, weight_descale, residual, values, pool);
        return {};
    }
};

/** The forward-only encoder on host memory, its products, attention and element-wise steps from
 *  the kits it is handed. */
template <typename products_kit_type_, typename attention_kit_type_, typename elementwise_kit_type_>
struct ram_tower {
    using steps_t = ram_steps<products_kit_type_, attention_kit_type_, elementwise_kit_type_>;

    /**
     *  @brief Pack every weight of @p checkpoint into NumKong's aligned heap, in @p precision for
     *      the block products.
     *  @retval status_t::invalid_input_k A tensor is missing or misshapen.
     */
    [[nodiscard]] static expected<ram_tower> make(uc::checkpoint_to_load_t const &checkpoint,
                                                  geometry_t const &geometry, precision_t precision) noexcept {
        nk_allocator_t policy {};
        if (nk_status_t const status = nk_allocator_init_heap(&policy, nk_default_alignment_k); status != nk_success_k)
            return expected<ram_tower>::fail(uc::status_t::other_k, "tower memory setup failed: {}",
                                             nk_status_name(status));
        ram_placer_t const placer {nk::allocator<b8_t> {policy, nullptr}, nullptr};
        expected<tower_weights_t> weights = pack_weights<products_kit_type_>(checkpoint, geometry, precision, placer);
        if (!weights) return expected<ram_tower>::fail(std::move(weights));
        ram_tower tower;
        tower.weights_ = std::move(*weights);
        return expected<ram_tower> {std::move(tower)};
    }

    /** The scratch bytes one call over @p tokens live tokens in @p segments items needs. */
    [[nodiscard]] expected<std::size_t> scratch_bytes(std::size_t tokens, std::size_t segments) const noexcept {
        return scratch_bytes_of<attention_kit_type_>(weights_, tokens, segments);
    }

    /** Encode @p batch in host memory on @p pool 's threads. */
    [[nodiscard]] expected_t encode(uc::multimodal_batch_view_t batch, outputs_t outputs, std::span<b8_t> scratch,
                                    fu::flat_pool_t &pool) const noexcept {
        steps_t steps {pool};
        return encode_steps(steps, weights_, batch, outputs, scratch);
    }

    [[nodiscard]] tower_weights_t const &weights() const noexcept { return weights_; }

  private:
    tower_weights_t weights_ {};
};

} // namespace unum::uform::encoder
