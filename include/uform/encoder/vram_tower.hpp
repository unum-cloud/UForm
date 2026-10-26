/**
 *  @file include/uform/encoder/vram_tower.hpp
 *  @author Ash Vardanian
 *  @date October 25, 2026
 *  @brief The encoder tower in one GPU's memory, shared by NVIDIA and AMD: weights packed by the
 *      products kit it is handed, every step of @ref encode_steps queued once on a stream.
 *
 *  The steps NumKong has no kernel for, and the copies weights move with, come from the vendor's
 *  entries kit, @ref vram_nvidia_entries_kit_t or @ref vram_amd_entries_kit_t ; nothing here names
 *  a vendor or a class. Every call returns once its steps are queued, and the caller synchronizes
 *  the stream before reading the outputs.
 */
#pragma once

#include <span> // `std::span` — the scratch a call borrows

#include "uform/encoder/tower.hpp" // `tower_weights_t`, `pack_weights`, `encode_steps`

namespace unum::uform::encoder {

/** Places a tower's arrays in one device's memory, copying on its stream through the vendor's
 *  entries kit. */
template <typename entries_kit_type_>
struct vram_placer {
    nk::allocator<b8_t> allocator {};
    nk_stream_t stream = nullptr;

    /** Copy @p values into @p target , a fresh array in device memory. */
    template <typename value_type_>
    [[nodiscard]] expected_t place(std::span<value_type_ const> values, placed_array<value_type_> &target,
                                   sz::string_view_t what) const noexcept {
        target = placed_array<value_type_> {nk::allocator<value_type_> {allocator}};
        if (values.empty()) return {};
        if (st::failed(target.resize(st::assume_uninitialized, values.size())))
            return expected_t::fail(uc::status_t::bad_alloc_k, "{} allocation failed ({} bytes)", what,
                                    values.size_bytes());
        return checked(entries_kit_type_::copy(std::as_writable_bytes(std::span<value_type_> {target}),
                                               std::as_bytes(values), stream),
                       what);
    }
};

/** The steps of @ref encode_steps over one device's memory, each queued once on its stream through
 *  the kits a GPU class picked and its vendor's entries kit. */
template <typename products_kit_type_, typename attention_kit_type_, typename elementwise_kit_type_,
          typename entries_kit_type_>
struct vram_steps {
    using products_kit_t = products_kit_type_;
    using attention_kit_t = attention_kit_type_;
    using elementwise_kit_t = elementwise_kit_type_;
    using entries_kit_t = entries_kit_type_;

    nk_stream_t stream_handle = nullptr;

    [[nodiscard]] nk_stream_t stream() const noexcept { return stream_handle; }

    [[nodiscard]] std::size_t share(std::size_t tasks) noexcept { return std::max<std::size_t>(tasks, 1); }

    template <typename body_type_>
    [[nodiscard]] expected_t split(std::size_t tasks, std::size_t, sz::string_view_t what,
                                   body_type_ const &body) noexcept {
        return checked(body(0, tasks), what);
    }

    [[nodiscard]] expected_t zero(uc::bytes_span_t target) noexcept {
        return checked(entries_kit_t::zero(target, stream_handle), "zeroing");
    }

    [[nodiscard]] expected_t copy(uc::bytes_span_t target, uc::bytes_view_t source) noexcept {
        return checked(entries_kit_t::copy(target, source, stream_handle), "copying");
    }

    [[nodiscard]] expected_t fill_layout(u32r1_view_t lengths, u32r1_span_t packed_offsets,
                                         u32r1_span_t pool_offsets) noexcept {
        return checked(entries_kit_t::fill_layout(lengths, packed_offsets, pool_offsets, stream_handle), "layout");
    }

    [[nodiscard]] expected_t fill_coords(segment_layout_t const &layout, nk::tensor_view<u16_t, 2> image_shapes,
                                         nk::tensor_view<u16_t, 2> video_shapes, u32_t tube_frames, u32_t frame_stride,
                                         u32_t positions, i16r2_span_t coords) noexcept {
        return checked(entries_kit_t::fill_coords(layout, image_shapes, video_shapes, tube_frames, frame_stride,
                                                  positions, coords, stream_handle),
                       "coordinates");
    }

    [[nodiscard]] expected_t gather_bytes(segment_layout_t const &layout, nk::tensor_view<token_id_t, 1> tokens,
                                          bf16r2_view_t table, bf16r2_span_t rows) noexcept {
        return checked(entries_kit_t::gather_bytes(layout, tokens, table, rows, stream_handle), "byte gather");
    }

    [[nodiscard]] expected_t stage_patches(segment_layout_t const &layout, modality_t target,
                                           nk::tensor_view<pixel_t, 3> pixels, f32r1_view_t mean,
                                           f32r1_view_t inverse_std, bf16r2_span_t patches) noexcept {
        return checked(entries_kit_t::stage_patches(layout, target, pixels, mean, inverse_std, patches, stream_handle),
                       "patch staging");
    }

    [[nodiscard]] expected_t stage_frames(segment_layout_t const &layout, nk::tensor_view<f32_t, 2> samples,
                                          bf16r2_span_t frames) noexcept {
        return checked(entries_kit_t::stage_frames(layout, samples, frames, stream_handle), "frame staging");
    }

    [[nodiscard]] expected_t mrope(bf16r2_span_t values, u32_t role_offset, u32_t heads, u32_t head_width,
                                   i16r2_view_t coords, rope_grids_t angles) noexcept {
        return checked(entries_kit_t::mrope(values, role_offset, heads, head_width, coords, angles, stream_handle),
                       "M-RoPE");
    }

    template <typename values_type_>
    [[nodiscard]] expected_t quantize(values_type_ values, e4m3r2_span_t shadow, f32r1_span_t token_scales) noexcept {
        return checked(entries_kit_t::quantize_tokens_e4m3(values, shadow, token_scales, stream_handle),
                       "quantization");
    }

    [[nodiscard]] expected_t rescale(f32r2_view_t products, f32r1_view_t token_scales, f32_t weight_descale,
                                     bf16r2_view_t residual, bf16r2_span_t values) noexcept {
        return checked(
            entries_kit_t::rescale_tokens(products, token_scales, weight_descale, residual, values, stream_handle),
            "rescale");
    }
};

/** The forward-only encoder on one GPU, its products, attention and element-wise steps from the
 *  kits it is handed, its own steps from its vendor's entries kit, and its load's host staging
 *  through @p staging_kit_type_ , the element-wise kit of the CPU class its unit compiles for. */
template <typename products_kit_type_, typename attention_kit_type_, typename elementwise_kit_type_,
          typename entries_kit_type_, typename staging_kit_type_>
struct vram_tower {
    using steps_t = vram_steps<products_kit_type_, attention_kit_type_, elementwise_kit_type_, entries_kit_type_>;

    /**
     *  @brief Pack every weight of @p checkpoint on @p stream into memory @p policy allocates, in
     *      @p precision for the block products, and return once the packs are queued; the caller
     *      synchronizes @p stream before the first encode, and frees the tower on the same stream.
     *  @retval status_t::invalid_input_k A tensor is missing or misshapen.
     */
    [[nodiscard]] static expected<vram_tower> make(uc::checkpoint_to_load_t const &checkpoint,
                                                   geometry_t const &geometry, precision_t precision,
                                                   nk_allocator_t policy, nk_stream_t stream) noexcept {
        vram_placer<entries_kit_type_> const placer {nk::allocator<b8_t> {policy, stream}, stream};
        expected<tower_weights_t> weights = pack_weights<products_kit_type_, staging_kit_type_>(checkpoint, geometry,
                                                                                                precision, placer);
        if (!weights) return expected<vram_tower>::fail(std::move(weights));
        vram_tower tower;
        tower.weights_ = std::move(*weights);
        return expected<vram_tower> {std::move(tower)};
    }

    /** The scratch bytes one call over @p tokens live tokens in @p segments items needs. */
    [[nodiscard]] expected<std::size_t> scratch_bytes(std::size_t tokens, std::size_t segments) const noexcept {
        return scratch_bytes_of<attention_kit_type_>(weights_, tokens, segments);
    }

    /**
     *  @brief Encode @p batch on @p stream and return once every step is queued.
     *
     *  The batch's tokens, pixels, samples and shapes sit in memory the device addresses, and its
     *  segment offsets, lengths and modalities in memory both the host and the device address, as
     *  the host sizes the call from them. The scratch and @p outputs are device memory.
     */
    [[nodiscard]] expected_t encode(uc::multimodal_batch_view_t batch, outputs_t outputs, std::span<b8_t> scratch,
                                    nk_stream_t stream) const noexcept {
        steps_t steps {stream};
        return encode_steps(steps, weights_, batch, outputs, scratch);
    }

    [[nodiscard]] tower_weights_t const &weights() const noexcept { return weights_; }

  private:
    tower_weights_t weights_ {};
};

} // namespace unum::uform::encoder
