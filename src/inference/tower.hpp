/**
 *  @brief The forward-only encoder tower over NumKong: entries, the transformer blocks, the final norm, the
 *      attention pool and the projection head, in BFloat16 or E4M3 chosen at load.
 *  @file src/inference/tower.hpp
 *  @author Ash Vardanian
 *  @date October 17, 2026
 *
 *  The tower reproduces the training tower's recipes step for step, so a checkpoint embeds the same on every
 *  residency. Precision is a runtime value, never a type:
 *
 *  - @c precision_t::bf16_k carries a Float32 residual stream: each norm narrows it to BFloat16 operands
 *    (rmscast), the QK-norm reads the Float32 product, and the products add back into Float32.
 *  - @c precision_t::e4m3_k carries a BFloat16 residual stream: each product quantizes its activations per token,
 *    multiplies against per-tensor-scaled E4M3 weights, and rescales with the residual folded in; Q, K and V cast
 *    unscaled into E4M3 for attention.
 *
 *  Entries, the pool and the head stay BFloat16 in both. Attention runs over ragged segments with a bidirectional
 *  band, each item one segment, on packed rows that skip the zero cells of short items.
 *
 *  The tower owns its packed weights only. Every call borrows its scratch from the caller's arena through
 *  @ref placement_t , sized by @ref tower_t::scratch_bytes , and a RAM call borrows the executor's thread pool.
 */
#pragma once

#include <cstddef> // `std::size_t`

#include <span> // `std::span` — the scratch a call borrows

#include <forkunion.hpp>          // vendor-boundary: `fu::flat_pool_t`, the RAM executor's threads
#include <numkong/capabilities.h> // `nk_capability_t`, `nk_stream_t`
#include <numkong/memory.hpp>     // `nk::aligned_allocator` — packs on NumKong's alignment

#include "core/types.hpp"            // `expected`, `expected_t`, the scalar aliases
#include "core/array.hpp"            // `array` — the owned weights
#include "core/checkpoint.hpp"       // `checkpoint_to_load_t`
#include "core/multimodal_batch.hpp" // `multimodal_batch_view_t`
#include "inference/geometry.hpp"    // `geometry_t`
#include "inference/entries.hpp"     // `rope_angles_t`, the host steps

namespace unum::uform::inference {

using uc::b8_t;
using uc::expected_t;
using uc::u8_t;

/** The dtype of the block stack's products, fixed by the load's quantization. */
enum class precision_t : u8_t { bf16_k, e4m3_k };

/** Where one call runs, built per call from the executor and never stored. */
struct placement_t {

    /** A CPU mask, or one device's mask, picking every NumKong kernel. */
    nk_capability_t capabilities = 0;

    /** Null on the CPU, or the device stream every call queues on. */
    nk_stream_t stream = nullptr;

    /** The call's activations, at least @ref tower_t::scratch_bytes long and aligned to
     *  @c nk_default_alignment_k , in the memory @ref capabilities address. */
    std::span<b8_t> scratch {};
};

/** Where one encode lands its products, both in @ref dtype ; a null pointer skips its surface. */
struct outputs_t {

    /** @b [segments,hidden] one embedding per item, PAD items zero. */
    void *embeddings = nullptr;

    /** @b [tokens,hidden] the final-norm states of every live token, items back to back in order. */
    void *states = nullptr;

    /** The scalar both surfaces are cast into. */
    nk_dtype_t dtype = nk_f32_k;
};

/** An array in the memory a tower runs from: NumKong's aligned heap on the CPU, a device's own memory on a GPU. */
template <typename value_type_>
using placed_array = uc::array<value_type_, nk::allocator<value_type_>>;

/** NumKong pack bytes, aligned for every backend. */
using packed_bytes_t = placed_array<b8_t>;

/** One weight matrix @b [outputs,inputs] packed as the B operand of its product. */
struct packed_weight_t {
    packed_bytes_t bytes {};
    std::size_t outputs = 0;
    std::size_t inputs = 0;

    /** The per-tensor scale an E4M3 pack multiplies back, @c max_abs/240 ; one for BFloat16. */
    f32_t descale = 1.0f;
};

/** One block's weights: the four products in the load's dtype, and the two norm gains widened to Float32. */
struct block_weights_t {

    /** @c qkv , @c out , @c gate_up and @c down , in that order. */
    packed_weight_t products[4] {};
    placed_array<f32_t> pre_norm {};
    placed_array<f32_t> post_norm {};
};

/** The entries, BFloat16 in every precision; an absent modality leaves its projection empty. */
struct entry_weights_t {

    /** @b [vocabulary,hidden] the byte embedding table. */
    placed_array<nk_bf16_t> embedding {};
    packed_weight_t image {}, video {}, audio {};
    placed_array<f32_t> image_mean {}, image_inverse_std {}, video_mean {}, video_inverse_std {};
};

/** The final norm, the attention pool and the head, BFloat16 in every precision. */
struct pool_weights_t {
    placed_array<f32_t> final_norm {};

    /** @b [hidden] the learned query every item's states answer. */
    placed_array<nk_bf16_t> query {};
    placed_array<f32_t> head_norm {};
    packed_weight_t fc1 {}, fc2 {};
};

/** The forward-only encoder tower: every weight packed once at @ref make , read by any number of calls. */
struct tower_t {

    /**
     *  @brief Pack every weight of @p checkpoint for @p placement 's capabilities, in @p precision for the block
     *      products — E4M3 quantizes each with its own @c max_abs/240 descale.
     *
     *  A device placement packs on its stream into memory from NumKong's device allocator and waits for the packs
     *  before returning. The tower frees that memory on the same stream, so the stream outlives the tower.
     *
     *  @retval status_t::invalid_input_k A tensor is missing or misshapen.
     *  @retval status_t::shape_unsupported_k A device placement of a vendor this build does not compile.
     */
    [[nodiscard]] static expected<tower_t> make(uc::checkpoint_to_load_t const &checkpoint, geometry_t const &geometry,
                                                precision_t precision, placement_t const &placement) noexcept;

    /** The scratch bytes one call over @p tokens live tokens in @p segments items needs on @p capabilities . */
    [[nodiscard]] expected<std::size_t> scratch_bytes(std::size_t tokens, std::size_t segments,
                                                      nk_capability_t capabilities) const noexcept;

    /** Encode @p batch in host memory on @p pool 's threads: products in 64-row tiles, attention in equal task
     *  windows, element-wise steps in row slices. */
    [[nodiscard]] expected_t encode(uc::multimodal_batch_view_t batch, outputs_t outputs, placement_t const &placement,
                                    fu::flat_pool_t &pool) const noexcept;

    /**
     *  @brief Encode @p batch on @p placement 's stream, a tower made for the same device, and return once every
     *      step is queued; the caller synchronizes the stream before reading @p outputs .
     *
     *  The batch's tokens, pixels, samples and shapes sit in memory the device addresses, and its segment offsets,
     *  lengths and modalities in memory both the host and the device address, as the host sizes the call from
     *  them. The scratch and @p outputs are device memory.
     */
    [[nodiscard]] expected_t encode(uc::multimodal_batch_view_t batch, outputs_t outputs,
                                    placement_t const &placement) const noexcept;

    [[nodiscard]] geometry_t const &geometry() const noexcept { return geometry_; }
    [[nodiscard]] precision_t precision() const noexcept { return precision_; }

  private:

    /** The one body of both encodes, its threads or its stream behind @p executor . */
    template <typename executor_type_>
    [[nodiscard]] expected_t encode_(uc::multimodal_batch_view_t batch, outputs_t outputs, placement_t const &placement,
                                     executor_type_ &executor) const noexcept;

    geometry_t geometry_ {};
    precision_t precision_ = precision_t::bf16_k;
    uc::array<block_weights_t> blocks_ {};
    entry_weights_t entries_ {};
    pool_weights_t pool_ {};

    /** The rotary grids, copied into the tower's memory, and the views over that copy. */
    placed_array<f32_t> angles_ {};
    rope_grids_t angle_grids_ {};
};

} // namespace unum::uform::inference
