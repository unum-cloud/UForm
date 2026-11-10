/**
 *  @file include/uform/encoder/tower.hpp
 *  @author Ash Vardanian
 *  @date October 17, 2026
 *  @brief What every encoder tower shares: the packed weights, the activations a call reserves,
 *      the packing of a checkpoint, and the forward recipe each tower runs through its own steps.
 *
 *  The towers reproduce the training tower's recipes step for step, so a checkpoint embeds the same
 *  on every residency. Precision is a runtime value, never a type:
 *
 *  - @c precision_t::bf16_k carries a Float32 residual stream: each norm narrows it to BFloat16
 *    operands (rmscast), the QK-norm reads the Float32 product, and the products add back into
 *    Float32.
 *  - @c precision_t::e4m3_k carries a BFloat16 residual stream: each product quantizes its
 *    activations per token, multiplies against per-tensor-scaled E4M3 weights, and rescales with
 *    the residual folded in; Q, K and V cast unscaled into E4M3 for attention.
 *
 *  Entries, the pool and the head stay BFloat16 in both. Attention runs over ragged segments
 *  with a bidirectional band, each item one segment, on packed rows that skip the zero cells
 *  of short items.
 *
 *  The towers of @ref ram_tower.hpp and @ref vram_tower.hpp each own one
 *  @ref unum::uform::encoder::tower_weights_t and run @c encode_steps through steps of their own:
 *  the host splits a step across a thread pool, a device queues it once on a stream. Every NumKong
 *  call goes through the kits a steps type names, and a load stages through the host element-wise
 *  kit it is handed, so nothing here picks a capability. Every function is @c static , so each
 *  class unit keeps the copy its own flags compiled.
 */
#pragma once

#include <cmath>   // `std::sqrt`
#include <cstddef> // `std::size_t`

#include <span> // `std::span` — the scratch a call borrows

#include <numkong/capabilities.h> // `nk_stream_t`, `nk_status_t`
#include <numkong/memory.hpp>     // `nk::allocator` — packs on NumKong's alignment

#include "uform/utilities/safetensors_format.hpp" // `checkpoint_to_load_t`
#include "uform/encoder/multimodal_batch.hpp"     // `multimodal_batch_view_t`
#include "uform/utilities/types.hpp"              // `expected`, `expected_t`, the scalar aliases, `st::basic_vector`
#include "uform/encoder/entries.hpp"              // `rope_angles_t`, `segment_layout_t`, `e4m3_range_k`
#include "uform/encoder/geometry.hpp"             // `geometry_t`
#include "uform/encoder/ram_kits.hpp" // `e4m3_products`, `quantize_tokens_e4m3` — the host staging of a load

namespace unum::uform::encoder {

using uu::b8_t;
using uu::bytes_span_t;
using uu::expected_t;
using uu::u8_t;

/** The dtype of the block stack's products, fixed by the load's quantization. */
enum class precision_t : u8_t { bf16_k, e4m3_k };

/** Rows of any scalar NumKong casts, as the bytes they span, @b [rows,row_bytes] , each row packed
 *  with @c dtype values. */
struct rows_span_t {
    nk::tensor_span<b8_t, 2> bytes {};
    nk_dtype_t dtype = nk_f32_k;

    rows_span_t() = default;
    template <typename value_type_>
    rows_span_t(nk::tensor_span<value_type_, 2> typed, nk_dtype_t dtype) noexcept
        : bytes(static_cast<b8_t *>(static_cast<void *>(typed.data())), typed.extent(0),
                typed.extent(1) * sizeof(value_type_)),
          dtype(dtype) {}

    [[nodiscard]] bool empty() const noexcept { return bytes.empty(); }

    /** Values one row holds. */
    [[nodiscard]] std::size_t columns() const noexcept { return bytes.extent(1) * 8 / nk_dtype_bits(dtype); }

    /** Rows @b [first,first+count) ; absent rows stay absent. */
    [[nodiscard]] rows_span_t slice(std::size_t first, std::size_t count) const noexcept {
        if (bytes.empty()) return *this;
        return {bytes[nk::range(first, first + count), nk::slice], dtype};
    }
};

/** Where one encode lands its products; an empty view skips its surface. */
struct outputs_t {

    /** @b [segments,hidden] one embedding per item, PAD items zero. */
    rows_span_t embeddings {};

    /** @b [tokens,hidden] the final-norm states of every live token, items back to
     *  back in order. */
    rows_span_t states {};

    /** @b [tokens,token_width] the token head's unit vectors of every live token, items back to
     *  back in order; needs a token head. */
    rows_span_t token_vectors {};

    /** @b [segments,masked_vocabulary] the byte decoder's Float32 logits at each item's last token,
     *  the position a choice masks; needs the decoder. */
    f32r2_span_t masked_logits {};
};

/** An array in the memory a tower runs from: NumKong's aligned heap on the CPU, a device's own
 *  memory on a GPU. */
template <typename value_type_>
using placed_array = st::basic_vector<value_type_, nk::allocator<value_type_>>;

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

/** One block's weights: the four products in the load's dtype, and the two norm gains
 *  widened to Float32. */
struct block_weights_t {

    /** @c qkv , @c out , @c gate_up and @c down , in that order. Unbraced, as MSVC counts a
     *  braced array member as throwing; each element still takes its own initializers. */
    packed_weight_t products[4];
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

/** The late-interaction head, BFloat16 in every precision: its norm gain, the @b [width,hidden]
 *  projection, and the gain of @c 1/sqrt(width) that turns an RMS norm into the unit length. */
struct token_head_weights_t {
    placed_array<f32_t> norm {};
    packed_weight_t projection {};
    placed_array<f32_t> unit_gain {};
};

/** The byte decoder @c masked_tokens , BFloat16 in every precision, @b [vocabulary,hidden] . */
struct masked_tokens_weights_t {
    packed_weight_t decoder {};
};

/** Everything a tower packs from a checkpoint, the same fields on every residency; how the
 *  products pack is the kit's. A head the checkpoint lacks stays empty. */
struct tower_weights_t {
    geometry_t geometry {};
    precision_t precision = precision_t::bf16_k;
    st::basic_vector<block_weights_t> blocks {};
    entry_weights_t entries {};
    pool_weights_t pool {};
    token_head_weights_t token_head {};
    masked_tokens_weights_t masked_tokens {};

    /** The rotary grids, copied into the tower's memory, and the views over that copy. */
    placed_array<f32_t> angles {};
    rope_grids_t angle_grids {};
};

/** Rows one product tile spans, so threads share each streamed weight panel across a
 *  few dozen rows. */
inline constexpr std::size_t product_tile_rows_k = 64;

/** The four block products, in @ref block_weights_t::products order. */
inline constexpr std::size_t product_qkv_k = 0, product_out_k = 1, product_gate_up_k = 2, product_down_k = 3;

/** The tower's status for a NumKong @p status : memory, a shape no kernel takes, or anything else,
 *  such as a device failing. */
[[nodiscard]] inline static uu::status_t tower_status_of(nk_status_t status) noexcept {
    switch (status) {
    case nk_success_k: return uu::status_t::success_k;
    case nk_bad_alloc_k: return uu::status_t::bad_alloc_k;
    case nk_unexpected_dimensions_k:
    case nk_missing_kernel_k:
    case nk_misaligned_k:
    case nk_pack_mismatch_k: return uu::status_t::unsupported_k;
    default: return uu::status_t::internal_k;
    }
}

/** Lift a NumKong status into the tower's diagnostics, naming the step that returned it. */
[[nodiscard]] inline static expected_t checked(nk_status_t status, sz::string_view_t what) noexcept {
    if (status == nk_success_k) return {};
    return expected_t::fail(tower_status_of(status), "{} failed: {}", what, nk_status_name(status));
}

#pragma region Activations

/** A bump cursor over a call's scratch; over an empty scratch it only measures, handing
 *  out null views. */
struct scratch_cursor_t {
    uu::bytes_span_t scratch {};
    std::size_t used = 0;

    /** The next @p count values of @p value_type_ . */
    template <typename value_type_>
    [[nodiscard]] nk::tensor_span<value_type_, 1> take(std::size_t count) noexcept {
        std::size_t const offset = used;
        used += uu::round_up_to_multiple(count * sizeof(value_type_), nk_default_alignment_k);
        if (scratch.empty()) return {static_cast<value_type_ *>(nullptr), count};
        return {static_cast<value_type_ *>(static_cast<void *>(scratch.data() + offset)), count};
    }

    /** The next @b [rows,columns] values of @p value_type_ . */
    template <typename value_type_>
    [[nodiscard]] nk::tensor_span<value_type_, 2> take(std::size_t rows, std::size_t columns) noexcept {
        return {take<value_type_>(rows * columns).data(), rows, columns};
    }
};

/** Every activation one call writes, reserved in its scratch. The wide buffers carry the
 *  BFloat16 recipe's Float32 residual stream, the narrow ones the E4M3 recipe's BFloat16 stream
 *  and every operand. The products, patches and shadow rows serve steps of several widths, so
 *  they stay flat. */
struct activations_t {
    u32r1_span_t packed_offsets {}, pool_offsets {};
    i16r2_span_t coords {};
    f32r2_span_t wide_residual {}, wide_joined {}, wide_activated {};
    f32r1_span_t products {}, token_scales {};
    bf16r2_span_t narrow_residual {}, narrow_joined {}, normalized {}, qkv {}, gate_up {}, narrow_activated {},
        states {};
    bf16r1_span_t patches {};
    e4m3r1_span_t shadow {};
    uu::b8r1_span_t keys_values {};
    f32r2_span_t wide_pooled {}, wide_head {};
    bf16r2_span_t pooled {}, head {};
};

/** Patch scalars of the widest present media modality, zero for a text-only tower. */
[[nodiscard]] inline static std::size_t patch_scalars(geometry_t const &geometry) noexcept {
    std::size_t scalars = 0;
    if (geometry.image_patch_side) scalars = fu::max_of<std::size_t>(scalars, geometry.image_scalars());
    if (geometry.video_patch_side) scalars = fu::max_of<std::size_t>(scalars, geometry.video_scalars());
    if (geometry.audio_chunk_samples) scalars = fu::max_of<std::size_t>(scalars, geometry.audio_chunk_samples);
    return scalars;
}

/** Reserve every activation for @p tokens rows in @p segments items behind @p cursor . */
[[nodiscard]] inline static activations_t reserve_activations(scratch_cursor_t &cursor, geometry_t const &geometry,
                                                              std::size_t tokens, std::size_t segments,
                                                              std::size_t keys_values_bytes) noexcept {
    std::size_t const hidden = geometry.hidden, ffn = geometry.ffn();
    std::size_t const widest = fu::max_of<std::size_t>(3 * hidden, 2 * ffn);
    activations_t a;
    a.packed_offsets = cursor.take<u32_t>(segments + 1);
    a.pool_offsets = cursor.take<u32_t>(segments + 1);
    a.coords = cursor.take<i16_t>(tokens, 3);
    a.wide_residual = cursor.take<f32_t>(tokens, hidden);
    a.wide_joined = cursor.take<f32_t>(tokens, hidden);
    a.products = cursor.take<f32_t>(tokens * widest);
    a.wide_activated = cursor.take<f32_t>(tokens, ffn);
    a.token_scales = cursor.take<f32_t>(tokens);
    a.narrow_residual = cursor.take<nk_bf16_t>(tokens, hidden);
    a.narrow_joined = cursor.take<nk_bf16_t>(tokens, hidden);
    a.normalized = cursor.take<nk_bf16_t>(tokens, hidden);
    a.qkv = cursor.take<nk_bf16_t>(tokens, 3 * hidden);
    a.gate_up = cursor.take<nk_bf16_t>(tokens, 2 * ffn);
    a.narrow_activated = cursor.take<nk_bf16_t>(tokens, ffn);
    a.states = cursor.take<nk_bf16_t>(tokens, hidden);
    a.patches = cursor.take<nk_bf16_t>(tokens * patch_scalars(geometry));
    a.shadow = cursor.take<nk_e4m3_t>(tokens * widest);
    a.keys_values = cursor.take<b8_t>(keys_values_bytes);
    a.wide_pooled = cursor.take<f32_t>(segments, hidden);
    a.wide_head = cursor.take<f32_t>(segments, hidden);
    a.pooled = cursor.take<nk_bf16_t>(segments, hidden);
    a.head = cursor.take<nk_bf16_t>(segments, hidden);
    return a;
}

/** The K/V pack bytes the widest of the block attention and the pool needs. */
template <typename attention_kit_type_>
[[nodiscard]] static expected<std::size_t> keys_values_bytes(geometry_t const &geometry, precision_t precision,
                                                             std::size_t tokens, std::size_t segments) noexcept {
    nk_size_t block_bytes = 0, pool_bytes = 0;
    nk_status_t status = precision == precision_t::bf16_k
                             ? attention_kit_type_::pack_size_bf16(geometry.heads(), geometry.head_width, tokens,
                                                                   segments, &block_bytes)
                             : attention_kit_type_::pack_size_e4m3(geometry.heads(), geometry.head_width, tokens,
                                                                   segments, &block_bytes);
    if (status == nk_success_k)
        status = attention_kit_type_::pack_size_bf16(geometry.heads(), geometry.head_width, tokens, segments,
                                                     &pool_bytes);
    if (status != nk_success_k)
        return expected<std::size_t>::fail(uu::status_t::internal_k, "attention pack sizing failed: {}",
                                           nk_status_name(status));
    return expected<std::size_t> {fu::max_of<std::size_t>(block_bytes, pool_bytes)};
}

/** The scratch bytes one call over @p tokens live tokens in @p segments items needs. */
template <typename attention_kit_type_>
[[nodiscard]] static expected<std::size_t> scratch_bytes_of(tower_weights_t const &weights, std::size_t tokens,
                                                            std::size_t segments) noexcept {
    expected<std::size_t> keys_values = keys_values_bytes<attention_kit_type_>(weights.geometry, weights.precision,
                                                                               tokens, segments);
    if (!keys_values) return keys_values;
    scratch_cursor_t measure;
    [[maybe_unused]] activations_t const unplaced = reserve_activations(measure, weights.geometry, tokens, segments,
                                                                        *keys_values);
    return expected<std::size_t> {measure.used};
}

#pragma endregion Activations

#pragma region Packing

/** Every scalar of the contiguous @p tensor , rows back to back. */
[[nodiscard]] inline static std::span<nk_bf16_t const> scalars_of(bf16r2_view_t tensor) noexcept {
    nk_bf16_t const *const first = tensor.data();
    return {first, first + tensor.numel()};
}

/** The checkpoint tensor @p name , which must be @b [rows,columns] . */
[[nodiscard]] inline static expected<bf16r2_view_t> lookup_tensor(uu::checkpoint_to_load_t const &checkpoint,
                                                                  sz::string_view_t name, std::size_t rows,
                                                                  std::size_t columns) noexcept {
    bf16r2_view_t const tensor = checkpoint[name];
    if (tensor.empty() || tensor.extent(0) != rows || tensor.extent(1) != columns)
        return expected<bf16r2_view_t>::fail(uu::status_t::invalid_argument_k, "the checkpoint must hold {} as [{},{}]",
                                             name, rows, columns);
    return expected<bf16r2_view_t> {tensor};
}

/** Widen the checkpoint gain @p name of @p width scalars on the host through @p staging_kit_type_
 *  and place it into @p gain . */
template <typename staging_kit_type_, typename placer_type_>
[[nodiscard]] static expected_t widen_gain(uu::checkpoint_to_load_t const &checkpoint, sz::string_view_t name,
                                           std::size_t width, placer_type_ const &placer,
                                           placed_array<f32_t> &gain) noexcept {
    expected<bf16r2_view_t> tensor = lookup_tensor(checkpoint, name, 1, width);
    if (!tensor) return expected_t::fail(std::move(tensor));
    st::basic_vector<f32_t> wide;
    if (st::failed(wide.resize(st::assume_uninitialized, width)))
        return expected_t::fail(uu::status_t::bad_alloc_k, "{} widening failed", name);
    expected_t widened = checked(
        staging_kit_type_::cast(tensor->data(), nk_bf16_k, wide.data(), nk_f32_k, width, nullptr), name);
    if (!widened) return widened;
    return placer.place(std::span<f32_t const> {wide.data(), width}, gain, name);
}

/** Pack @p weight into @p packed for @p precision through @p products_kit_type_ — E4M3 first
 *  divides by its own @c max_abs/240 descale on the host, through @p staging_kit_type_ , and a kit
 *  without E4M3 products refuses it. The operand is placed beside the pack first, as a device pack
 *  reads device memory. */
template <typename products_kit_type_, typename staging_kit_type_, typename placer_type_>
[[nodiscard]] static expected_t pack_product(bf16r2_view_t weight, precision_t precision, placer_type_ const &placer,
                                             packed_weight_t &packed) noexcept {
    std::size_t const outputs = weight.extent(0), inputs = weight.extent(1), scalars = outputs * inputs;
    packed.outputs = outputs, packed.inputs = inputs;
    nk_size_t bytes = 0;
    nk_status_t status = nk_success_k;
    if (precision == precision_t::bf16_k) status = products_kit_type_::pack_size_bf16(outputs, inputs, &bytes);
    else if constexpr (e4m3_products<products_kit_type_>)
        status = products_kit_type_::pack_size_e4m3(outputs, inputs, &bytes);
    else return expected_t::fail(uu::status_t::unsupported_k, "the class has no E4M3 products");
    if (status != nk_success_k)
        return expected_t::fail(uu::status_t::internal_k, "pack sizing [{}x{}] failed: {}", outputs, inputs,
                                nk_status_name(status));
    packed.bytes = packed_bytes_t {placer.allocator};
    if (st::failed(packed.bytes.resize(st::assume_uninitialized, bytes)))
        return expected_t::fail(uu::status_t::bad_alloc_k, "pack allocation failed ({} bytes)", bytes);
    if (precision == precision_t::bf16_k) {
        placed_array<nk_bf16_t> operand;
        expected_t placed = placer.place(std::span<nk_bf16_t const> {weight.data(), scalars}, operand, "pack operand");
        if (!placed) return placed;
        return checked(products_kit_type_::pack_bf16(operand.data(), outputs, inputs, inputs * sizeof(nk_bf16_t),
                                                     packed.bytes.data(), 0, outputs, placer.stream),
                       "packing");
    }
    if constexpr (e4m3_products<products_kit_type_>) {
        st::basic_vector<f32_t> wide;
        st::basic_vector<nk_e4m3_t> shadow;
        if (st::failed(wide.resize(st::assume_uninitialized, scalars)) ||
            st::failed(shadow.resize(st::assume_uninitialized, scalars)))
            return expected_t::fail(uu::status_t::bad_alloc_k, "quantization staging failed ({} scalars)", scalars);
        status = quantize_tokens_e4m3<staging_kit_type_>(
            weight.reshape<2>({1, scalars}), 0, e4m3r2_span_t {shadow.data(), 1, scalars},
            f32r1_span_t {&packed.descale, 1}, f32r1_span_t {wide.data(), scalars});
        if (status != nk_success_k) return checked(status, "quantization");
        placed_array<nk_e4m3_t> operand;
        expected_t placed = placer.place(std::span<nk_e4m3_t const> {shadow.data(), scalars}, operand, "pack operand");
        if (!placed) return placed;
        return checked(products_kit_type_::pack_e4m3(operand.data(), outputs, inputs, inputs * sizeof(nk_e4m3_t),
                                                     packed.bytes.data(), 0, outputs, placer.stream),
                       "packing");
    }
    return {}; // a kit without E4M3 products refused above
}

/** Look up @p name as @b [rows,columns] and pack it. */
template <typename products_kit_type_, typename staging_kit_type_, typename placer_type_>
[[nodiscard]] static expected_t load_product(uu::checkpoint_to_load_t const &checkpoint, sz::string_view_t name,
                                             std::size_t rows, std::size_t columns, precision_t precision,
                                             placer_type_ const &placer, packed_weight_t &packed) noexcept {
    expected<bf16r2_view_t> tensor = lookup_tensor(checkpoint, name, rows, columns);
    if (!tensor) return expected_t::fail(std::move(tensor));
    return pack_product<products_kit_type_, staging_kit_type_>(*tensor, precision, placer, packed);
}

/**
 *  @brief Pack every weight of @p checkpoint for @p products_kit_type_ into memory @p placer
 *      places, in @p precision for the block products — E4M3 quantizes each with its own
 *      @c max_abs/240 descale — widening and quantizing on the host through @p staging_kit_type_ .
 *
 *  @par Statuses:
 *  - @c status_t::invalid_argument_k A tensor is missing or misshapen.
 *  - @c status_t::unsupported_k E4M3 asked of a products kit without E4M3 kernels.
 */
template <typename products_kit_type_, typename staging_kit_type_, typename placer_type_>
[[nodiscard]] static expected<tower_weights_t> pack_weights(uu::checkpoint_to_load_t const &checkpoint,
                                                            geometry_t const &geometry, precision_t precision,
                                                            placer_type_ const &placer) noexcept {
    using result_t = expected<tower_weights_t>;
    std::size_t const hidden = geometry.hidden, ffn = geometry.ffn();
    tower_weights_t weights;
    weights.geometry = geometry;
    weights.precision = precision;

    if (st::failed(weights.blocks.resize(geometry.depth)))
        return result_t::fail(uu::status_t::bad_alloc_k, "block table allocation failed ({})", geometry.depth);
    sz::string_t name;
    sz::string_view_t const parts[4] = {"qkv", "out", "gate_up", "down"};
    std::size_t const extents[4][2] = {{3 * hidden, hidden}, {hidden, hidden}, {2 * ffn, hidden}, {hidden, ffn}};
    for (std::size_t index = 0; index != geometry.depth; ++index) {
        block_weights_t &block = weights.blocks[index];
        expected_t loaded {};
        for (std::size_t product = 0; loaded && product != 4; ++product) {
            name.clear();
            if (!uu::format_into(name, "model.blocks.{}.{}", index, parts[product]))
                return result_t::fail(uu::status_t::bad_alloc_k, "name staging failed");
            loaded = load_product<products_kit_type_, staging_kit_type_>(checkpoint, name.view(), extents[product][0],
                                                                         extents[product][1], precision, placer,
                                                                         block.products[product]);
        }
        name.clear();
        if (loaded && !uu::format_into(name, "model.blocks.{}.pre_norm", index))
            return result_t::fail(uu::status_t::bad_alloc_k, "name staging failed");
        if (loaded) loaded = widen_gain<staging_kit_type_>(checkpoint, name.view(), hidden, placer, block.pre_norm);
        name.clear();
        if (loaded && !uu::format_into(name, "model.blocks.{}.post_norm", index))
            return result_t::fail(uu::status_t::bad_alloc_k, "name staging failed");
        if (loaded) loaded = widen_gain<staging_kit_type_>(checkpoint, name.view(), hidden, placer, block.post_norm);
        if (!loaded) return result_t::fail(std::move(loaded));
    }

    entry_weights_t &entries = weights.entries;
    expected<bf16r2_view_t> embedding = lookup_tensor(checkpoint, "model.embedding", geometry.vocabulary, hidden);
    if (!embedding) return result_t::fail(std::move(embedding));
    expected_t loaded = placer.place(scalars_of(*embedding), entries.embedding, "embedding");
    if (loaded && geometry.image_patch_side)
        loaded = load_product<products_kit_type_, staging_kit_type_>(checkpoint, "model.image_projection", hidden,
                                                                     geometry.image_scalars(), precision_t::bf16_k,
                                                                     placer, entries.image);
    if (loaded && geometry.video_patch_side)
        loaded = load_product<products_kit_type_, staging_kit_type_>(checkpoint, "model.video_projection", hidden,
                                                                     geometry.video_scalars(), precision_t::bf16_k,
                                                                     placer, entries.video);
    if (loaded && geometry.audio_chunk_samples)
        loaded = load_product<products_kit_type_, staging_kit_type_>(checkpoint, "model.audio_projection", hidden,
                                                                     geometry.audio_chunk_samples, precision_t::bf16_k,
                                                                     placer, entries.audio);
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

    pool_weights_t &pool = weights.pool;
    expected<bf16r2_view_t> query = lookup_tensor(checkpoint, "model.pool_query", 1, hidden);
    if (!query) return result_t::fail(std::move(query));
    loaded = placer.place(std::span<nk_bf16_t const> {query->data(), hidden}, pool.query, "pool query");
    if (loaded) loaded = widen_gain<staging_kit_type_>(checkpoint, "model.final_norm", hidden, placer, pool.final_norm);
    if (loaded) loaded = widen_gain<staging_kit_type_>(checkpoint, "model.head.norm", hidden, placer, pool.head_norm);
    if (loaded)
        loaded = load_product<products_kit_type_, staging_kit_type_>(checkpoint, "model.head.fc1", hidden, hidden,
                                                                     precision_t::bf16_k, placer, pool.fc1);
    if (loaded)
        loaded = load_product<products_kit_type_, staging_kit_type_>(checkpoint, "model.head.fc2", hidden, hidden,
                                                                     precision_t::bf16_k, placer, pool.fc2);
    if (!loaded) return result_t::fail(std::move(loaded));

    if (std::size_t const width = geometry.token_width) {
        token_head_weights_t &head = weights.token_head;
        loaded = widen_gain<staging_kit_type_>(checkpoint, "model.token_head.norm", hidden, placer, head.norm);
        if (loaded)
            loaded = load_product<products_kit_type_, staging_kit_type_>(
                checkpoint, "model.token_head.projection", width, hidden, precision_t::bf16_k, placer, head.projection);
        st::basic_vector<f32_t> unit_gain;
        if (loaded && st::failed(unit_gain.resize(st::assume_uninitialized, width)))
            return result_t::fail(uu::status_t::bad_alloc_k, "unit gain staging failed");
        for (f32_t &gain : unit_gain) gain = 1.0f / std::sqrt(static_cast<f32_t>(width));
        if (loaded)
            loaded = placer.place(std::span<f32_t const> {unit_gain.data(), width}, head.unit_gain, "unit gain");
        if (!loaded) return result_t::fail(std::move(loaded));
    }
    if (geometry.masked_vocabulary)
        loaded = load_product<products_kit_type_, staging_kit_type_>(
            checkpoint, "model.masked_tokens", geometry.masked_vocabulary, hidden, precision_t::bf16_k, placer,
            weights.masked_tokens.decoder);
    if (!loaded) return result_t::fail(std::move(loaded));

    expected<rope_angles_t> angles = rope_angles_t::make(geometry.context, geometry.rope_temporal, geometry.rope_height,
                                                         geometry.rope_width, geometry.rope_base);
    if (!angles) return result_t::fail(std::move(angles));
    loaded = placer.place(angles->scalars(), weights.angles, "angle grids");
    if (!loaded) return result_t::fail(std::move(loaded));
    weights.angle_grids = angles->grids(weights.angles.data());
    return result_t {std::move(weights)};
}

#pragma endregion Packing

#pragma region Forward

/**
 *  @brief Encode @p batch through @p steps , landing the surfaces @p outputs names; the one recipe
 *      both towers run, each with steps that split a step across threads or queue it on a stream.
 *
 *  A @p steps_type_ names its @c products_kit_t , @c attention_kit_t and @c elementwise_kit_t ,
 *  hands their kernels its @c stream() , and serves the tower's own steps: @c share , @c split ,
 *  @c zero , @c copy , @c fill_layout , @c fill_coords , @c gather_bytes , @c stage_patches ,
 *  @c stage_frames , @c mrope , @c quantize and @c rescale .
 */
template <typename steps_type_>
[[nodiscard]] static expected_t encode_steps(steps_type_ &steps, tower_weights_t const &weights,
                                             multimodal_batch_view_t batch, outputs_t outputs,
                                             bytes_span_t scratch) noexcept {
    using products_kit_t = typename steps_type_::products_kit_t;
    using attention_kit_t = typename steps_type_::attention_kit_t;
    using elementwise_kit_t = typename steps_type_::elementwise_kit_t;
    geometry_t const &geometry = weights.geometry;
    precision_t const precision = weights.precision;

    std::size_t const segments = batch.segment_lengths.extent(0);
    if (segments == 0 || batch.segment_offsets.extent(0) < segments || batch.segment_modality.extent(0) != segments)
        return expected_t::fail(uu::status_t::invalid_argument_k,
                                "a batch needs one offset, length and modality per item");
    if (outputs.embeddings.empty() && outputs.states.empty() && outputs.token_vectors.empty() &&
        outputs.masked_logits.empty())
        return expected_t::fail(uu::status_t::invalid_argument_k, "an encode keeps at least one surface");
    if (!outputs.token_vectors.empty() && weights.token_head.projection.bytes.empty())
        return expected_t::fail(uu::status_t::invalid_argument_k, "the checkpoint carries no token head");
    if (!outputs.masked_logits.empty() && weights.masked_tokens.decoder.bytes.empty())
        return expected_t::fail(uu::status_t::invalid_argument_k, "the checkpoint carries no byte decoder");
    std::size_t tokens = 0;
    for (std::size_t segment = 0; segment != segments; ++segment) {
        if (batch.segment_lengths[segment] > geometry.context)
            return expected_t::fail(uu::status_t::invalid_argument_k, "item {} spans {} cells, the context is {}",
                                    segment, batch.segment_lengths[segment], geometry.context);
        tokens += batch.segment_lengths[segment];
    }
    if (tokens == 0) return expected_t::fail(uu::status_t::invalid_argument_k, "a batch needs a live cell");
    expected<std::size_t> keys_values = keys_values_bytes<attention_kit_t>(geometry, precision, tokens, segments);
    if (!keys_values) return expected_t::fail(std::move(keys_values));
    scratch_cursor_t cursor {scratch};
    activations_t const a = reserve_activations(cursor, geometry, tokens, segments, *keys_values);
    if (scratch.size() < cursor.used)
        return expected_t::fail(uu::status_t::bad_alloc_k, "the scratch holds {} bytes, the call needs {}",
                                scratch.size(), cursor.used);

    std::size_t const hidden = geometry.hidden, ffn = geometry.ffn(), heads = geometry.heads();
    std::size_t const head_width = geometry.head_width;
    f32_t const epsilon = geometry.rms_epsilon;
    f32_t const scale = 1.0f / std::sqrt(static_cast<f32_t>(head_width));
    std::size_t const rows_share = steps.share(tokens);
    nk_stream_t const stream = steps.stream();

    auto cast = [&](rows_span_t source, rows_span_t target) noexcept {
        std::size_t const rows = target.bytes.extent(0), width = target.columns();
        return steps.split(rows, steps.share(rows), "cast", [&](std::size_t first, std::size_t count) noexcept {
            return elementwise_kit_t::cast(source.bytes.row(first).data(), source.dtype, target.bytes.row(first).data(),
                                           target.dtype, count * width, stream);
        });
    };
    auto add = [&](f32r2_span_t values, f32r2_view_t addend) noexcept {
        return steps.split(tokens, rows_share, "residual", [&](std::size_t first, std::size_t count) noexcept {
            f32_t *const slice = values.row(first).data();
            return elementwise_kit_t::sum_f32(slice, addend.row(first).data(), count * hidden, slice, stream);
        });
    };
    auto rmscast = [&](f32r2_view_t source, std::span<f32_t const> gain, bf16r2_span_t target, std::size_t groups,
                       std::size_t columns) noexcept {
        return steps.split(tokens, rows_share, "rmscast", [&](std::size_t first, std::size_t count) noexcept {
            return elementwise_kit_t::rmscast_bf16(source.row(first).data(), gain.data(), target.row(first).data(),
                                                   count, groups, columns, source.stride_bytes(0),
                                                   target.stride_bytes(0), epsilon, stream);
        });
    };
    auto rmsnorm = [&](bf16r2_view_t source, std::span<f32_t const> gain, bf16r2_span_t target, std::size_t groups,
                       std::size_t columns) noexcept {
        std::size_t const rows = source.extent(0);
        return steps.split(rows, steps.share(rows), "rmsnorm", [&](std::size_t first, std::size_t count) noexcept {
            return elementwise_kit_t::rmsnorm_bf16(source.row(first).data(), gain.data(), target.row(first).data(),
                                                   count, groups, columns, source.stride_bytes(0),
                                                   target.stride_bytes(0), epsilon, stream);
        });
    };
    auto project = [&](packed_weight_t const &weight, rows_span_t activations, f32r2_span_t products) noexcept {
        std::size_t const rows = activations.bytes.extent(0), outputs = weight.outputs, inputs = weight.inputs;
        return steps.split(rows, product_tile_rows_k, "product", [&](std::size_t first, std::size_t count) noexcept {
            void const *const tile_inputs = activations.bytes.row(first).data();
            f32_t *const tile = products.row(first).data();
            std::size_t const stride = activations.bytes.stride_bytes(0);
            if (activations.dtype == nk_bf16_k)
                return products_kit_t::multiply_bf16(static_cast<nk_bf16_t const *>(tile_inputs), weight.bytes.data(),
                                                     tile, count, outputs, inputs, stride, products.stride_bytes(0),
                                                     nullptr, 0, stream);
            if constexpr (e4m3_products<products_kit_t>)
                return products_kit_t::multiply_e4m3(static_cast<nk_e4m3_t const *>(tile_inputs), weight.bytes.data(),
                                                     tile, count, outputs, inputs, stride, products.stride_bytes(0),
                                                     nullptr, 0, stream);
            else return nk_missing_kernel_k; // a kit without E4M3 products refused the load
        });
    };
    auto products_of = [&](std::size_t rows, std::size_t width) noexcept {
        return f32r2_span_t {a.products.data(), rows, width};
    };

    entry_weights_t const &entries = weights.entries;
    pool_weights_t const &pool = weights.pool;
    expected_t staged = steps.fill_layout(batch.segment_lengths, a.packed_offsets, a.pool_offsets);
    segment_layout_t const layout {batch.segment_offsets, batch.segment_lengths, a.packed_offsets,
                                   batch.segment_modality};
    if (staged) staged = steps.zero(uu::bytes_of(a.coords));
    if (staged)
        staged = steps.fill_coords(layout, batch.image_shapes, batch.video_shapes, geometry.video_tube_frames,
                                   geometry.video_frame_stride, geometry.context, a.coords);

    if (staged) staged = steps.zero(uu::bytes_of(a.narrow_residual));
    if (staged)
        staged = steps.gather_bytes(layout, batch.tokens.flatten<1>(),
                                    {entries.embedding.data(), geometry.vocabulary, hidden}, a.narrow_residual);
    if (staged) staged = cast({a.narrow_residual, nk_bf16_k}, {a.wide_residual, nk_f32_k});
    auto present = [&](modality_t modality, packed_weight_t const &weight, std::size_t content_scalars) noexcept {
        if (weight.bytes.empty() || content_scalars == 0) return false;
        for (std::size_t segment = 0; segment != segments; ++segment)
            if (batch.segment_modality[segment] == modality && batch.segment_lengths[segment] != 0) return true;
        return false;
    };
    auto add_projection = [&](packed_weight_t const &weight, bf16r2_span_t patches) noexcept {
        expected_t projected = project(weight, {patches, nk_bf16_k}, products_of(tokens, hidden));
        return projected ? add(a.wide_residual, products_of(tokens, hidden)) : std::move(projected);
    };
    std::size_t media = 0;
    if (staged && present(modality_t::image_k, entries.image, batch.images.numel())) {
        ++media;
        std::size_t const cells = batch.images.extent(0) * batch.images.extent(1);
        std::size_t const spatial = batch.images.extent(2) * batch.images.extent(3);
        bf16r2_span_t const patches {a.patches.data(), tokens, geometry.image_scalars()};
        staged = steps.zero(uu::bytes_of(patches));
        if (staged)
            staged = steps.stage_patches(layout, modality_t::image_k,
                                         batch.images.reshape<3>({cells, spatial, batch.images.extent(4)}),
                                         {entries.image_mean.data(), geometry.image_channels},
                                         {entries.image_inverse_std.data(), geometry.image_channels}, patches);
        if (staged) staged = add_projection(entries.image, patches);
    }
    if (staged && present(modality_t::video_k, entries.video, batch.videos.numel())) {
        ++media;
        std::size_t const cells = batch.videos.extent(0) * batch.videos.extent(1);
        std::size_t const spatial = batch.videos.extent(2) * batch.videos.extent(3) * batch.videos.extent(4);
        bf16r2_span_t const patches {a.patches.data(), tokens, geometry.video_scalars()};
        staged = steps.zero(uu::bytes_of(patches));
        if (staged)
            staged = steps.stage_patches(layout, modality_t::video_k,
                                         batch.videos.reshape<3>({cells, spatial, batch.videos.extent(5)}),
                                         {entries.video_mean.data(), geometry.video_channels},
                                         {entries.video_inverse_std.data(), geometry.video_channels}, patches);
        if (staged) staged = add_projection(entries.video, patches);
    }
    if (staged && present(modality_t::audio_k, entries.audio, batch.audio_samples.numel())) {
        ++media;
        std::size_t const cells = batch.audio_samples.extent(0) * batch.audio_samples.extent(1);
        std::size_t const samples = batch.audio_samples.extent(2);
        bf16r2_span_t const frames {a.patches.data(), tokens, samples};
        staged = steps.zero(uu::bytes_of(frames));
        if (staged) staged = steps.stage_frames(layout, batch.audio_samples.reshape<2>({cells, samples}), frames);
        if (staged) staged = add_projection(entries.audio, frames);
    }
    if (staged && media != 0 && precision == precision_t::e4m3_k)
        staged = cast({a.wide_residual, nk_f32_k}, {a.narrow_residual, nk_bf16_k});
    if (!staged) return staged;

    auto attend = [&](rows_span_t qkv) noexcept {
        std::size_t const stride = qkv.bytes.stride_bytes(0), role_bytes = qkv.bytes.extent(1) / 3;
        b8_t const *const queries = qkv.bytes.data();
        b8_t const *const keys = queries + role_bytes, *const values = queries + 2 * role_bytes;
        expected_t packed = steps.split(
            segments * heads, steps.share(segments * heads), "attention pack",
            [&](std::size_t first, std::size_t count) noexcept {
                if (qkv.dtype == nk_bf16_k)
                    return attention_kit_t::pack_bf16(heads, head_width, a.packed_offsets.data(), nullptr, segments,
                                                      static_cast<nk_bf16_t const *>(static_cast<void const *>(keys)),
                                                      stride,
                                                      static_cast<nk_bf16_t const *>(static_cast<void const *>(values)),
                                                      stride, a.keys_values.data(), first, first + count, stream);
                return attention_kit_t::pack_e4m3(heads, head_width, a.packed_offsets.data(), nullptr, segments,
                                                  static_cast<nk_e4m3_t const *>(static_cast<void const *>(keys)),
                                                  stride,
                                                  static_cast<nk_e4m3_t const *>(static_cast<void const *>(values)),
                                                  stride, a.keys_values.data(), first, first + count, stream);
            });
        if (!packed) return packed;
        f32r2_span_t const attended = products_of(tokens, hidden);
        return steps.split(tokens * heads, steps.share(tokens * heads), "attention",
                           [&](std::size_t first, std::size_t count) noexcept {
                               if (qkv.dtype == nk_bf16_k)
                                   return attention_kit_t::attend_bf16(
                                       heads, heads, head_width, a.packed_offsets.data(), tokens, scale,
                                       NUMKONG_SIZE_MAX, NUMKONG_SIZE_MAX,
                                       static_cast<nk_bf16_t const *>(static_cast<void const *>(queries)), stride,
                                       a.keys_values.data(), attended.data(), attended.stride_bytes(0), nullptr, first,
                                       first + count, nullptr, 0, stream);
                               return attention_kit_t::attend_e4m3(
                                   heads, heads, head_width, a.packed_offsets.data(), tokens, scale, NUMKONG_SIZE_MAX,
                                   NUMKONG_SIZE_MAX, static_cast<nk_e4m3_t const *>(static_cast<void const *>(queries)),
                                   stride, a.keys_values.data(), attended.data(), attended.stride_bytes(0), nullptr,
                                   first, first + count, nullptr, 0, stream);
                           });
    };
    auto rotate = [&]() noexcept { // the Q and K heads lie back to back, so one angle row turns both
        return steps.mrope(a.qkv, 0, 2 * heads, head_width, a.coords, weights.angle_grids);
    };
    auto quantize = [&](bf16r2_view_t values) noexcept {
        e4m3r2_span_t const shadow {a.shadow.data(), tokens, values.extent(1)};
        return steps.quantize(values, shadow, a.token_scales);
    };
    auto rescale = [&](block_weights_t const &block, std::size_t product, bf16r2_view_t residual,
                       bf16r2_span_t values) noexcept {
        std::size_t const width = block.products[product].outputs;
        return steps.rescale(products_of(tokens, width), a.token_scales, block.products[product].descale, residual,
                             values);
    };
    auto shadow_of = [&](std::size_t width) noexcept {
        return rows_span_t {e4m3r2_span_t {a.shadow.data(), tokens, width}, nk_e4m3_k};
    };

    for (block_weights_t const &block : weights.blocks) {
        packed_weight_t const *const products = block.products;
        if (precision == precision_t::bf16_k) {
            f32r2_span_t const qkv_products = products_of(tokens, 3 * hidden);
            f32r2_span_t const gate_up_products = products_of(tokens, 2 * ffn);
            staged = rmscast(a.wide_residual, block.pre_norm, a.normalized, 1, hidden);
            if (staged) staged = project(products[product_qkv_k], {a.normalized, nk_bf16_k}, qkv_products);
            if (staged) staged = cast({qkv_products, nk_f32_k}, {a.qkv, nk_bf16_k});
            if (staged) staged = rmscast(qkv_products, {}, a.qkv, 2 * heads, head_width);
            if (staged) staged = rotate();
            if (staged) staged = attend({a.qkv, nk_bf16_k});
            if (staged) staged = cast({products_of(tokens, hidden), nk_f32_k}, {a.normalized, nk_bf16_k});
            if (staged) staged = project(products[product_out_k], {a.normalized, nk_bf16_k}, a.wide_joined);
            if (staged) staged = add(a.wide_joined, a.wide_residual);
            if (staged) staged = rmscast(a.wide_joined, block.post_norm, a.normalized, 1, hidden);
            if (staged) staged = project(products[product_gate_up_k], {a.normalized, nk_bf16_k}, gate_up_products);
            if (staged)
                staged = steps.split(tokens, rows_share, "SwiGLU", [&](std::size_t first, std::size_t count) noexcept {
                    f32_t const *const gate = gate_up_products.row(first).data();
                    return elementwise_kit_t::swiglu_f32(gate, gate + ffn, a.wide_activated.row(first).data(), count,
                                                         ffn, gate_up_products.stride_bytes(0),
                                                         gate_up_products.stride_bytes(0),
                                                         a.wide_activated.stride_bytes(0), 1.0f, 1.0f, stream);
                });
            if (staged) staged = cast({a.wide_activated, nk_f32_k}, {a.narrow_activated, nk_bf16_k});
            if (staged) staged = project(products[product_down_k], {a.narrow_activated, nk_bf16_k}, a.wide_residual);
            if (staged) staged = add(a.wide_residual, a.wide_joined);
            if (!staged) return staged;
            continue;
        }
        staged = rmsnorm(a.narrow_residual, block.pre_norm, a.normalized, 1, hidden);
        if (staged) staged = quantize(a.normalized);
        if (staged) staged = project(products[product_qkv_k], shadow_of(hidden), products_of(tokens, 3 * hidden));
        if (staged) staged = rescale(block, product_qkv_k, {}, a.qkv);
        if (staged)
            staged = steps.split(tokens, rows_share, "QK-norm", [&](std::size_t first, std::size_t count) noexcept {
                nk_bf16_t *const rows = a.qkv.row(first).data();
                return elementwise_kit_t::rmsnorm_bf16(rows, nullptr, rows, count, 2 * heads, head_width,
                                                       a.qkv.stride_bytes(0), a.qkv.stride_bytes(0), epsilon, stream);
            });
        if (staged) staged = rotate();
        if (staged) staged = cast({a.qkv, nk_bf16_k}, shadow_of(3 * hidden));
        if (staged) staged = attend(shadow_of(3 * hidden));
        if (staged)
            staged = steps.quantize(f32r2_view_t {products_of(tokens, hidden)},
                                    e4m3r2_span_t {a.shadow.data(), tokens, hidden}, a.token_scales);
        if (staged) staged = project(products[product_out_k], shadow_of(hidden), products_of(tokens, hidden));
        if (staged) staged = rescale(block, product_out_k, a.narrow_residual, a.narrow_joined);
        if (staged) staged = rmsnorm(a.narrow_joined, block.post_norm, a.normalized, 1, hidden);
        if (staged) staged = quantize(a.normalized);
        if (staged) staged = project(products[product_gate_up_k], shadow_of(hidden), products_of(tokens, 2 * ffn));
        if (staged) staged = rescale(block, product_gate_up_k, {}, a.gate_up);
        if (staged)
            staged = steps.split(tokens, rows_share, "SwiGLU", [&](std::size_t first, std::size_t count) noexcept {
                nk_bf16_t const *const gate = a.gate_up.row(first).data();
                return elementwise_kit_t::swiglu_bf16(gate, gate + ffn, a.narrow_activated.row(first).data(), count,
                                                      ffn, a.gate_up.stride_bytes(0), a.gate_up.stride_bytes(0),
                                                      a.narrow_activated.stride_bytes(0), 1.0f, 1.0f, stream);
            });
        if (staged) staged = quantize(a.narrow_activated);
        if (staged) staged = project(products[product_down_k], shadow_of(ffn), products_of(tokens, hidden));
        if (staged) staged = rescale(block, product_down_k, a.narrow_joined, a.narrow_residual);
        if (!staged) return staged;
    }

    staged = precision == precision_t::bf16_k ? rmscast(a.wide_residual, pool.final_norm, a.states, 1, hidden)
                                              : rmsnorm(a.narrow_residual, pool.final_norm, a.states, 1, hidden);
    if (staged && !outputs.states.empty()) staged = cast({a.states, nk_bf16_k}, outputs.states);

    if (staged && !outputs.token_vectors.empty()) { // the block activations are dead, so they lend the heads room
        token_head_weights_t const &head = weights.token_head;
        std::size_t const width = head.projection.outputs;
        f32r2_span_t const projected = products_of(tokens, width);
        bf16r2_span_t const unit {a.qkv.data(), tokens, width};
        staged = rmsnorm(a.states, head.norm, a.normalized, 1, hidden);
        if (staged) staged = project(head.projection, {a.normalized, nk_bf16_k}, projected);
        if (staged) staged = rmscast(projected, head.unit_gain, unit, 1, width);
        if (staged) staged = cast({unit, nk_bf16_k}, outputs.token_vectors);
    }
    if (staged && !outputs.masked_logits.empty()) {
        std::size_t row = 0;
        for (std::size_t segment = 0; staged && segment != segments; ++segment) {
            std::size_t const length = batch.segment_lengths[segment];
            if (length)
                staged = steps.copy(uu::bytes_of(a.pooled.row(segment)), uu::bytes_of(a.states.row(row + length - 1)));
            row += length;
        }
        if (staged) staged = project(weights.masked_tokens.decoder, {a.pooled, nk_bf16_k}, outputs.masked_logits);
    }
    if (!staged || outputs.embeddings.empty()) return staged;

    std::size_t const segment_share = steps.share(segments * heads);
    staged = steps.split(segments * heads, segment_share, "pool pack",
                         [&](std::size_t first, std::size_t count) noexcept {
                             return attention_kit_t::pack_bf16(heads, head_width, a.packed_offsets.data(), nullptr,
                                                               segments, a.states.data(), a.states.stride_bytes(0),
                                                               a.states.data(), a.states.stride_bytes(0),
                                                               a.keys_values.data(), first, first + count, stream);
                         });
    if (staged)
        staged = steps.split(segments * heads, segment_share, "pool",
                             [&](std::size_t first, std::size_t count) noexcept {
                                 return attention_kit_t::attend_bf16(
                                     heads, heads, head_width, a.pool_offsets.data(), segments, scale, NUMKONG_SIZE_MAX,
                                     NUMKONG_SIZE_MAX, pool.query.data(), 0, a.keys_values.data(), a.wide_pooled.data(),
                                     a.wide_pooled.stride_bytes(0), nullptr, first, first + count, nullptr, 0, stream);
                             });

    if (staged) staged = cast({a.wide_pooled, nk_f32_k}, {a.pooled, nk_bf16_k});
    if (staged) staged = rmsnorm(a.pooled, pool.head_norm, a.head, 1, hidden);
    if (staged) staged = project(pool.fc1, {a.head, nk_bf16_k}, a.wide_head);
    if (staged) staged = cast({a.wide_head, nk_f32_k}, {a.head, nk_bf16_k});
    if (staged)
        staged = steps.split(
            segments, steps.share(segments), "SiLU", [&](std::size_t first, std::size_t count) noexcept {
                nk_bf16_t *const rows = a.head.row(first).data();
                return elementwise_kit_t::swiglu_bf16(rows, nullptr, rows, count, hidden, a.head.stride_bytes(0), 0,
                                                      a.head.stride_bytes(0), 1.0f, 1.0f, stream);
            });
    if (staged) staged = project(pool.fc2, {a.head, nk_bf16_k}, a.wide_head);
    if (staged) staged = cast({a.wide_head, nk_f32_k}, outputs.embeddings);
    return staged;
}

#pragma endregion Forward

} // namespace unum::uform::encoder
