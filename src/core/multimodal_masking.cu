/**
 *  @brief The multimodal masked-modeling objective's implementation: the corruption and reconstruction
 *      kernels and the construction bodies — compiled once into the `uform_core` library.
 *  @file src/core/multimodal_masking.cu
 *  @author Ash Vardanian
 *  @date July 3, 2026
 */
#include <algorithm> // `std::max` — the head product scratch sizing
#include <limits>    // `std::numeric_limits` — the softmax's row-maximum identity

#include <numkong/each.h> // `nk_each_sum_f32_best` — the media heads adding onto the shared gradient

#include "uform/core/multimodal_masking.hpp"

using namespace unum::uform::core;

/** All-threads-get-the-total CTA max — the fixed-order sibling of @ref block_reduce_broadcast for
 *  the softmax's row maximum: per-warp shuffle trees, lane-0 partials through static shared memory,
 *  one final warp tree, one broadcast cell. Deterministic, sized for @p block_size_ threads. */
template <unsigned block_size_>
[[nodiscard]] static __device__ f32_t block_reduce_max_broadcast_(f32_t value) noexcept {
    static_assert(block_size_ % warp_size_k == 0, "the staging array assumes whole warps");
    constexpr unsigned warps_k = block_size_ / warp_size_k;
    __shared__ f32_t warp_partials[warps_k];
    __shared__ f32_t block_total;
    unsigned const lane_index = threadIdx.x % warp_size_k;
    unsigned const warp_index = threadIdx.x / warp_size_k;
    for (unsigned delta = warp_size_k / 2u; delta != 0u; delta /= 2u)
        value = fmaxf(value, __shfl_xor_sync(0xffffffffu, value, delta));
    if (lane_index == 0u) warp_partials[warp_index] = value;
    __syncthreads();
    if (warp_index == 0u) {
        f32_t staged = lane_index < warps_k ? warp_partials[lane_index] : -std::numeric_limits<f32_t>::infinity();
        for (unsigned delta = warp_size_k / 2u; delta != 0u; delta /= 2u)
            staged = fmaxf(staged, __shfl_xor_sync(0xffffffffu, staged, delta));
        if (lane_index == 0u) block_total = staged;
    }
    __syncthreads();
    return block_total;
}

/** Cover one token at @p column of @p segment : write its keyed distortion (MASK / random / kept
 *  clean) and a loss weight of one. */
static __device__ void mask_token_(tokens_span_t distorted, u32r1_span_t loss_weights, u32_t position, u64_t seed,
                                   data_epoch_t data_epoch, u32_t segment, u32_t column, f32_t replace_fraction,
                                   f32_t random_fraction) {
    f32_t const action = masking_draw(seed, data_epoch, segment, column,
                                      static_cast<u32_t>(masking_draw_purpose_t::token_action_k));
    if (action < replace_fraction) { distorted[position] = byte_mask_k; }
    else if (action < replace_fraction + random_fraction)
        distorted[position] = static_cast<token_id_t>(
            masking_draw(seed, data_epoch, segment, column, static_cast<u32_t>(masking_draw_purpose_t::random_byte_k)) *
            256.0f);
    loss_weights[position] = 1u; // the remaining fraction keeps the clean byte, still scored
}

/**
 *  @brief Corrupt one segment in place: walk its tokens left-to-right, open geometric spans, and at
 *      each covered position write the distorted byte (MASK / random / kept clean) and a loss
 *      weight of one.
 *
 *  PAD breaks and never opens a span; uncovered positions were already copied clean with weight
 *  zero.
 *
 *  A segment the walk leaves uncovered gets one span from a keyed anchor, at its first byte that is
 *  not PAD, so every text row with bytes is scored. A non- @c text_k segment and a choice segment
 *  are skipped outright — they keep their clean tokens and zero weight.
 *
 *  __grid_size__ @c [grid_size_for(segments),1,1] , one thread per segment; no dynamic shared memory.
 */
static __global__ void masked_token_corrupt_kernel_(tokens_view_t clean, tokens_span_t distorted,
                                                    u32r1_span_t loss_weights, u32r1_view_t segment_offsets,
                                                    u32r1_view_t segment_lengths, modality_view_t segment_modality,
                                                    nk::tensor_view<token_id_t, 2> segment_choices, u64_t seed,
                                                    data_epoch_t data_epoch, f32_t open_rate, f32_t continue_rate,
                                                    f32_t replace_fraction, f32_t random_fraction) {
    u32_t const segment = blockIdx.x * blockDim.x + threadIdx.x;
    if (segment >= segment_lengths.extent(0)) return;
    if (segment_modality[segment] != modality_t::text_k) return; // media segments stay clean, weight zero
    if (!segment_choices.empty() && segment_choices[segment, 0] != byte_pad_k) return;
    u32_t const base = segment_offsets[segment];
    u32_t const length = segment_lengths[segment];
    bool in_span = false, covered = false;
    for (u32_t column = 0; column != length; ++column) {
        u32_t const position = base + column;
        if (clean[position] == byte_pad_k) {
            in_span = false;
            continue;
        }
        if (in_span)
            in_span = masking_draw(seed, data_epoch, segment, column,
                                   static_cast<u32_t>(masking_draw_purpose_t::span_continue_k)) < continue_rate;
        if (!in_span)
            in_span = masking_draw(seed, data_epoch, segment, column,
                                   static_cast<u32_t>(masking_draw_purpose_t::span_open_k)) < open_rate;
        if (!in_span) continue;
        mask_token_(distorted, loss_weights, position, seed, data_epoch, segment, column, replace_fraction,
                    random_fraction);
        covered = true;
    }
    if (covered || length == 0u) return;
    u32_t const anchor = static_cast<u32_t>(
        masking_draw(seed, data_epoch, segment, 0u, static_cast<u32_t>(masking_draw_purpose_t::span_anchor_k)) *
        static_cast<f32_t>(length));
    u32_t column = anchor;
    while (clean[base + column] == byte_pad_k) {
        column = (column + 1u) % length;
        if (column == anchor) return; // all PAD
    }
    do {
        mask_token_(distorted, loss_weights, base + column, seed, data_epoch, segment, column, replace_fraction,
                    random_fraction);
        ++column;
    } while (column != length && clean[base + column] != byte_pad_k &&
             masking_draw(seed, data_epoch, segment, column,
                          static_cast<u32_t>(masking_draw_purpose_t::span_continue_k)) < continue_rate);
}

/**
 *  @brief Count the loss-bearing positions and publish @c {λ/max(count,1), count} — one scalar the CE
 *      kernel scales gradients by, so the masked count never visits the host.
 *
 *  __grid_size__ @c [1,1,1] , one block; no dynamic shared memory.
 */
static __global__ void masked_count_kernel_(u32r1_view_t loss_weights, f32r1_span_t scale_and_count, f32_t weight) {
    f32_t live = 0.0f;
    for (std::size_t index = threadIdx.x; index < loss_weights.extent(0); index += blockDim.x)
        live += static_cast<f32_t>(loss_weights[index]);
    f32_t const total = block_reduce_broadcast<kernel_block_size_default_k>(live);
    if (threadIdx.x != 0u) return;
    scale_and_count[0] = weight / fmaxf(total, 1.0f);
    scale_and_count[1] = total;
}

/**
 *  @brief Fused cross-entropy over one token row: fixed-order block reduces for the max and the
 *      exponential sum, the per-position loss summand, and the pre-scaled softmax gradient — a zero
 *      loss weight writes zeros everywhere and scores nothing.
 *
 *  __grid_size__ @c [max_tokens,1,1] , one block per token; no dynamic shared memory.
 */
static __global__ void masked_cross_entropy_kernel_(f32r2_view_t logits, tokens_view_t targets,
                                                    u32r1_view_t loss_weights, f32r1_view_t scale_and_count,
                                                    f32r1_span_t position_loss, bf16r2_span_t logit_grad) {
    u32_t const row = blockIdx.x;
    bool const live = loss_weights[row] != 0u;
    u32_t const target = targets[row];
    u32_t const vocabulary = logits.extent(1);

    f32_t thread_max = -std::numeric_limits<f32_t>::infinity();
    for (u32_t byte = threadIdx.x; byte < vocabulary; byte += blockDim.x)
        thread_max = fmaxf(thread_max, logits[row, byte]);
    f32_t const row_max = block_reduce_max_broadcast_<kernel_block_size_default_k>(thread_max);

    f32_t thread_sum = 0.0f;
    for (u32_t byte = threadIdx.x; byte < vocabulary; byte += blockDim.x)
        thread_sum += __expf(logits[row, byte] - row_max);
    f32_t const denominator = block_reduce_broadcast<kernel_block_size_default_k>(thread_sum);

    f32_t const scale = live ? scale_and_count[0] : 0.0f;
    for (u32_t byte = threadIdx.x; byte < vocabulary; byte += blockDim.x) {
        f32_t const softmax = __expf(logits[row, byte] - row_max) / denominator;
        f32_t const indicator = live && byte == target ? 1.0f : 0.0f;
        logit_grad[row, byte] = __float2bfloat16((softmax - indicator) * scale);
    }
    if (threadIdx.x != 0u) return;
    position_loss[row] = live ? __logf(denominator) - (logits[row, target] - row_max) : 0.0f;
}

/**
 *  @brief Fold the per-position summands and add @c {λ·Σ, count} onto the rank's running loss
 *      surface.
 *
 *  __grid_size__ @c [1,1,1] , one block, fixed-order tree; no dynamic shared memory.
 */
static __global__ void masked_finalize_kernel_(f32r1_view_t position_loss, f32r1_view_t scale_and_count,
                                               f32r1_span_t loss_summands, f32_t weight) {
    f32_t thread_sum = 0.0f;
    for (std::size_t index = threadIdx.x; index < position_loss.extent(0); index += blockDim.x)
        thread_sum += position_loss[index];
    f32_t const total = block_reduce_broadcast<kernel_block_size_default_k>(thread_sum);
    if (threadIdx.x != 0u) return;
    loss_summands[0] += weight * total;
    loss_summands[1] += scale_and_count[1];
}

/** Whether segment @p segment of @p batch makes a choice: it holds tokens and its first legal
 *  byte is not the pad. */
[[nodiscard]] static __device__ bool chooses_(multimodal_batch_view_t const &batch, u32_t segment) noexcept {
    return batch.segment_lengths[segment] != 0u && batch.segment_choices[segment, 0] != byte_pad_k;
}

/**
 *  @brief Copy each choice segment's last row of @p normalized into its row of @p gathered , zeros
 *      for a segment that makes no choice, and publish @c {λ/max(count,1), count} over them.
 *
 *  __grid_size__ @c [segments,1,1] , threads stripe the hidden width; no dynamic shared memory.
 */
static __global__ void choices_gather_kernel_(bf16r2_view_t normalized, multimodal_batch_view_t batch,
                                              bf16r2_span_t gathered, f32r1_span_t scale_and_count, f32_t weight) {
    u32_t const segment = blockIdx.x, hidden = static_cast<u32_t>(gathered.extent(1));
    bool const live = chooses_(batch, segment);
    u32_t const row = live ? batch.segment_offsets[segment] + batch.segment_lengths[segment] - 1u : 0u;
    for (u32_t column = threadIdx.x; column < hidden; column += blockDim.x)
        gathered[segment, column] = live ? normalized[row, column] : __float2bfloat16(0.0f);
    if (segment != 0u) return;
    f32_t live_segments = 0.0f;
    for (u32_t index = threadIdx.x; index < gathered.extent(0); index += blockDim.x)
        live_segments += chooses_(batch, index) ? 1.0f : 0.0f;
    f32_t const total = block_reduce_broadcast<kernel_block_size_default_k>(live_segments);
    if (threadIdx.x != 0u) return;
    scale_and_count[0] = weight / std::max(total, 1.0f);
    scale_and_count[1] = total;
}

/**
 *  @brief SLCD over one choice segment: softmax restricted to its bytes, soft-target cross-entropy
 *      plus λ_B·Brier, and the pre-scaled gradient written only at its bytes' columns, zeros
 *      elsewhere.
 *
 *  With @c p_c the restricted softmax and @c t_c the targets, the logit gradient of a byte is
 *  @c (p_c−t_c)+λ_B·2·p_c·((p_c−t_c)−r) for @c r=Σ(p_v−t_v)·p_v . A pad entry is no choice, and a
 *  segment making none, a PAD among them, writes zeros and scores nothing.
 *
 *  __grid_size__ @c [segments,1,1] , one block per segment; no dynamic shared memory.
 */
static __global__ void choices_cross_entropy_kernel_(f32r2_view_t logits, multimodal_batch_view_t batch,
                                                     f32r1_view_t scale_and_count, f32_t brier_weight,
                                                     f32r1_span_t choice_loss, bf16r2_span_t logit_grad) {
    nk::tensor_view<token_id_t, 2> const choices = batch.segment_choices;
    f32r2_view_t const targets = batch.segment_choice_targets;
    u32_t const row = blockIdx.x, vocabulary = static_cast<u32_t>(logits.extent(1));
    u32_t const width = static_cast<u32_t>(choices.extent(1));
    for (u32_t byte = threadIdx.x; byte < vocabulary; byte += blockDim.x)
        logit_grad[row, byte] = __float2bfloat16(0.0f);
    if (!chooses_(batch, row)) {
        if (threadIdx.x == 0u) choice_loss[row] = 0.0f;
        return;
    }
    __syncthreads();

    f32_t thread_max = -std::numeric_limits<f32_t>::infinity();
    for (u32_t choice = threadIdx.x; choice < width; choice += blockDim.x)
        if (choices[row, choice] != byte_pad_k) thread_max = std::max(thread_max, logits[row, choices[row, choice]]);
    f32_t const row_max = block_reduce_max_broadcast_<kernel_block_size_default_k>(thread_max);

    f32_t thread_sum = 0.0f;
    for (u32_t choice = threadIdx.x; choice < width; choice += blockDim.x)
        if (choices[row, choice] != byte_pad_k) thread_sum += __expf(logits[row, choices[row, choice]] - row_max);
    f32_t const log_denominator = __logf(block_reduce_broadcast<kernel_block_size_default_k>(thread_sum));

    f32_t thread_cross_entropy = 0.0f, thread_brier = 0.0f, thread_correlation = 0.0f;
    for (u32_t choice = threadIdx.x; choice < width; choice += blockDim.x) {
        if (choices[row, choice] == byte_pad_k) continue;
        f32_t const log_probability = logits[row, choices[row, choice]] - row_max - log_denominator;
        f32_t const probability = __expf(log_probability), miss = probability - targets[row, choice];
        thread_cross_entropy -= targets[row, choice] * log_probability;
        thread_brier += miss * miss;
        thread_correlation += miss * probability;
    }
    f32_t const cross_entropy = block_reduce_broadcast<kernel_block_size_default_k>(thread_cross_entropy);
    f32_t const brier = block_reduce_broadcast<kernel_block_size_default_k>(thread_brier);
    f32_t const correlation = block_reduce_broadcast<kernel_block_size_default_k>(thread_correlation);

    f32_t const scale = scale_and_count[0];
    for (u32_t choice = threadIdx.x; choice < width; choice += blockDim.x) {
        token_id_t const byte = choices[row, choice];
        if (byte == byte_pad_k) continue;
        f32_t const probability = __expf(logits[row, byte] - row_max - log_denominator);
        f32_t const miss = probability - targets[row, choice];
        f32_t const gradient = miss + brier_weight * 2.0f * probability * (miss - correlation);
        logit_grad[row, byte] = __float2bfloat16(gradient * scale);
    }
    if (threadIdx.x == 0u) choice_loss[row] = cross_entropy + brier_weight * brier;
}

/**
 *  @brief Add each choice segment's row of @p gathered_grad onto its last row of
 *      @p normalized_grad , where the masked head's force may already sit.
 *
 *  __grid_size__ @c [segments,1,1] , threads stripe the hidden width; no dynamic shared memory.
 */
static __global__ void choices_scatter_kernel_(f32r2_view_t gathered_grad, multimodal_batch_view_t batch,
                                               bf16r2_span_t normalized_grad) {
    u32_t const segment = blockIdx.x, hidden = static_cast<u32_t>(gathered_grad.extent(1));
    if (!chooses_(batch, segment)) return;
    u32_t const row = batch.segment_offsets[segment] + batch.segment_lengths[segment] - 1u;
    for (u32_t column = threadIdx.x; column < hidden; column += blockDim.x)
        normalized_grad[row, column] = __float2bfloat16(__bfloat162float(normalized_grad[row, column]) +
                                                        gathered_grad[segment, column]);
}

/** Allocate off @p arena one linear head's NumKong packs and transients: its @b [outputs,hidden] weight packed
 *  both ways, the weight gradient's transposes over @p rows states (the input one wide enough to stage the
 *  weight's padded transpose too), the @c normalizedᵀ pack, and the Float32 products of all three products. */
static expected_t allocate_head_products_(gpu_arena_t &arena, nk_capability_t capabilities, std::size_t outputs,
                                          std::size_t hidden, std::size_t rows, b8r1_tensor_arena_t &packed_weight,
                                          b8r1_tensor_arena_t &packed_weight_transposed,
                                          bf16r1_tensor_arena_t &transposed_grad,
                                          bf16r1_tensor_arena_t &transposed_input, b8r1_tensor_arena_t &packed_input,
                                          f32r1_tensor_arena_t &products) noexcept {
    expected<b8r1_tensor_arena_t> forward = allocate_pack_bf16(outputs, hidden, capabilities, arena);
    if (!forward) return expected_t::fail(std::move(forward));
    expected<b8r1_tensor_arena_t> backward = allocate_pack_bf16(hidden, outputs, capabilities, arena);
    if (!backward) return expected_t::fail(std::move(backward));
    expected<b8r1_tensor_arena_t> input = allocate_pack_bf16(hidden, rows, capabilities, arena);
    if (!input) return expected_t::fail(std::move(input));
    packed_weight = std::move(forward.value_);
    packed_weight_transposed = std::move(backward.value_);
    packed_input = std::move(input.value_);
    std::size_t const padded_rows = round_up_to_multiple(rows, std::size_t {8});
    std::size_t const padded_outputs = round_up_to_multiple(outputs, std::size_t {8});
    gpu_arena_alloc<bf16_t> bf16_alloc {arena};
    transposed_grad = bf16r1_tensor_arena_t::uninitialized({outputs * padded_rows}, bf16_alloc).value;
    transposed_input =
        bf16r1_tensor_arena_t::uninitialized({hidden * std::max(padded_rows, padded_outputs)}, bf16_alloc).value;
    products = f32r1_tensor_arena_t::uninitialized({std::max(rows, outputs) * std::max(outputs, hidden)},
                                                   gpu_arena_alloc<f32_t> {arena})
                   .value;
    if (!transposed_grad || !transposed_input || !products)
        return expected_t::fail(status_t::bad_alloc_k, "head product scratch allocation failed (arena_used={})",
                                arena.used());
    return {};
}

expected<masked_tokens_t> masked_tokens_t::make(gpu_arena_t &arena, masked_tokens_config_t config,
                                                nk_capability_t capabilities) noexcept {
    using result_t = expected<masked_tokens_t>;
    if (config.vocabulary == 0 || config.hidden == 0 || config.max_tokens == 0)
        return result_t::fail(status_t::shape_unsupported_k,
                              "the masked head needs a full geometry (vocabulary={} hidden={} tokens={})",
                              config.vocabulary, config.hidden, config.max_tokens);
    masked_tokens_t head;
    head.config_ = config;
    head.capabilities_ = capabilities;

    gpu_arena_alloc<bf16_t> bf16_alloc {arena};
    gpu_arena_alloc<f32_t> f32_alloc {arena};
    std::size_t const padded_vocabulary = round_up_to_multiple(config.vocabulary, std::size_t {8});
    head.decoder_weight_ = bf16r2_tensor_arena_t::uninitialized({config.vocabulary, config.hidden}, bf16_alloc).value;
    head.decoder_weight_grad_ =
        bf16r2_tensor_arena_t::uninitialized({config.vocabulary, config.hidden}, bf16_alloc).value;
    head.logits_ = f32r2_tensor_arena_t::uninitialized({config.max_tokens, config.vocabulary}, f32_alloc).value;
    head.logit_grad_ = bf16r2_tensor_arena_t::uninitialized({config.max_tokens, padded_vocabulary}, bf16_alloc).value;
    head.position_loss_ = f32r1_tensor_arena_t::uninitialized({config.max_tokens}, f32_alloc).value;
    head.gradient_scale_ = f32r1_tensor_arena_t::uninitialized({2}, f32_alloc).value;
    if (!head.decoder_weight_ || !head.decoder_weight_grad_ || !head.logits_ || !head.logit_grad_ ||
        !head.position_loss_ || !head.gradient_scale_)
        return result_t::fail(status_t::bad_alloc_k, "masked head allocation failed (arena_used={})", arena.used());
    if (expected_t allocated = allocate_head_products_(arena, head.capabilities_, config.vocabulary, config.hidden,
                                                       config.max_tokens, head.packed_decoder_,
                                                       head.packed_decoder_transposed_, head.transposed_grad_,
                                                       head.transposed_input_, head.packed_input_, head.products_);
        !allocated)
        return result_t::fail(std::move(allocated));
    if (config.max_segments == 0) return result_t {std::move(head)};
    // The choices reuse the token rows' logits, which also widen the decoder gradient they add to.
    if (config.max_segments > config.max_tokens || config.max_tokens < config.hidden)
        return result_t::fail(status_t::shape_unsupported_k,
                              "choices need no more segments than tokens, and tokens past the width " //
                              "(segments={} tokens={} hidden={})",
                              config.max_segments, config.max_tokens, config.hidden);
    head.gathered_ = bf16r2_tensor_arena_t::uninitialized({config.max_segments, config.hidden}, bf16_alloc).value;
    if (!head.gathered_)
        return result_t::fail(status_t::bad_alloc_k, "choice rows allocation failed (arena_used={})", arena.used());
    return result_t {std::move(head)};
}

expected_t masked_tokens_t::pack_weights(gpu_stream_t &stream) noexcept {
    return pack_weight_bf16(decoder_weight_.view(), transposed_input_.span(), packed_decoder_.span(),
                            packed_decoder_transposed_.span(), capabilities_, stream);
}

expected_t masked_tokens_t::corrupt(tokens_view_t clean, tokens_span_t distorted, u32r1_span_t loss_weights,
                                    u32r1_view_t segment_offsets, u32r1_view_t segment_lengths,
                                    modality_view_t segment_modality, nk::tensor_view<token_id_t, 2> segment_choices,
                                    masking_recipe_t recipe, data_epoch_t data_epoch, gpu_stream_t &stream) noexcept {
    std::size_t const tokens = clean.extent(0);
    if (distorted.extent(0) != tokens || loss_weights.extent(0) != tokens)
        return expected_t::fail(status_t::shape_unsupported_k, "corrupt buffers disagree on token count ({})", tokens);
    if (recipe.token_mean_span == 0.0f) // the geometric open / continue rates divide by it below
        return expected_t::fail(status_t::shape_unsupported_k, "masking recipe needs a non-zero token_mean_span");
    if (recipe.replace_fraction + recipe.random_fraction > 1.0f) // the kept fraction is the remainder, must stay ≥ 0
        return expected_t::fail(status_t::shape_unsupported_k,
                                "masking recipe replace+random fractions exceed one ({} + {})", recipe.replace_fraction,
                                recipe.random_fraction);
    if (expected_t seeded = copy_device_to_device(clean, distorted, stream); !seeded) return seeded;
    if (expected_t reset = zero_async(loss_weights, stream); !reset) return reset;

    f32_t const open_rate = recipe.token_mask_probability / recipe.token_mean_span;
    f32_t const continue_rate = 1.0f - 1.0f / recipe.token_mean_span;
    std::size_t const segments = segment_lengths.extent(0);
    dim3 const block_shape {kernel_block_size_default_k, 1u, 1u};
    dim3 const segment_grid {grid_size_for(segments), 1u, 1u};
    return launch_kernel(&masked_token_corrupt_kernel_, segment_grid, block_shape, stream, clean, distorted,
                         loss_weights, segment_offsets, segment_lengths, segment_modality, segment_choices, recipe.seed,
                         data_epoch, open_rate, continue_rate, recipe.replace_fraction, recipe.random_fraction);
}

expected_t masked_tokens_t::operator()(bf16r2_view_t normalized, tokens_view_t clean_targets, u32r1_view_t loss_weights,
                                       f32r1_span_t loss_summands, bf16r2_span_t normalized_grad,
                                       gpu_stream_t &stream) noexcept {
    u32_t const hidden = config_.hidden;
    std::size_t const tokens = config_.max_tokens;
    if (normalized.extent(0) != tokens || normalized_grad.extent(0) != tokens || clean_targets.extent(0) != tokens ||
        loss_weights.extent(0) != tokens)
        return expected_t::fail(status_t::shape_unsupported_k, "masked head token count mismatch (expected {})",
                                tokens);
    if (normalized.extent(1) != hidden || normalized_grad.extent(1) != hidden || loss_summands.extent(0) != 2)
        return expected_t::fail(status_t::shape_unsupported_k, "masked head geometry mismatch (hidden={})", hidden);

    dim3 const one_block {1u, 1u, 1u};
    dim3 const block_shape {kernel_block_size_default_k, 1u, 1u};
    dim3 const row_grid {static_cast<u32_t>(tokens), 1u, 1u};
    bf16r2_span_t const padded_logit_grad = logit_grad_.span();
    bf16r2_span_t const logit_grad = padded_logit_grad[nk::all, nk::range(0, config_.vocabulary), nk::slice];
    f32r2_span_t const normalized_products {products_.data(), tokens, hidden};
    expected_t staged = launch_kernel(&masked_count_kernel_, one_block, block_shape, stream, loss_weights,
                                      gradient_scale_.span(), config_.weight);
    if (staged) staged = multiply_bf16(normalized, packed_decoder_.view(), logits_.span(), capabilities_, stream);
    if (staged)
        staged = launch_kernel(&masked_cross_entropy_kernel_, row_grid, block_shape, stream, logits_.view(),
                               clean_targets, loss_weights, gradient_scale_.view(), position_loss_.span(), logit_grad);
    if (staged)
        staged = launch_kernel(&masked_finalize_kernel_, one_block, block_shape, stream, position_loss_.view(),
                               gradient_scale_.view(), loss_summands, config_.weight);
    if (staged)
        staged = multiply_bf16(logit_grad, packed_decoder_transposed_.view(), normalized_products, capabilities_,
                               stream);
    if (staged)
        staged = numkong_result(nk_cast_best(normalized_products.data(), nk_f32_k, normalized_grad.data(), nk_bf16_k,
                                             tokens * hidden, capabilities_, stream.numkong()),
                                "masked token gradient narrowing");
    if (staged)
        staged = weight_gradient_bf16(logit_grad, normalized, decoder_weight_grad_.span(), transposed_grad_.span(),
                                      transposed_input_.span(), packed_input_.span(), products_.span(), b8r1_span_t {},
                                      capabilities_, stream);
    return staged;
}

expected_t masked_tokens_t::choose(bf16r2_view_t normalized, multimodal_batch_view_t const &batch,
                                   choices_config_t config, f32r1_span_t loss_summands, bf16r2_span_t normalized_grad,
                                   gpu_stream_t &stream) noexcept {
    std::size_t const hidden = config_.hidden, vocabulary = config_.vocabulary, segments = gathered_.extent(0);
    if (segments == 0 || batch.segment_choices.extent(0) != segments ||
        batch.segment_choice_targets.extent(0) != segments ||
        batch.segment_choice_targets.extent(1) != batch.segment_choices.extent(1))
        return expected_t::fail(status_t::shape_unsupported_k, "choices need [{},width] choice and target tables",
                                segments);
    if (normalized.extent(0) != config_.max_tokens || normalized_grad.extent(0) != config_.max_tokens ||
        normalized.extent(1) != hidden || normalized_grad.extent(1) != hidden || loss_summands.extent(0) != 2)
        return expected_t::fail(status_t::shape_unsupported_k, "choice head geometry mismatch (hidden={})", hidden);

    dim3 const one_block {1u, 1u, 1u};
    dim3 const block_shape {kernel_block_size_default_k, 1u, 1u};
    dim3 const segment_grid {static_cast<u32_t>(segments), 1u, 1u};
    f32r2_span_t const logits {logits_.data(), segments, vocabulary};
    bf16r2_span_t const padded_logit_grad = logit_grad_.span();
    bf16r2_span_t const logit_grad =
        padded_logit_grad[nk::range(std::size_t {0}, segments), nk::range(std::size_t {0}, vocabulary), nk::slice];
    f32r1_view_t const position_loss = position_loss_.view();
    f32r2_span_t const gathered_grad {products_.data(), segments, hidden};
    // The widened decoder gradient reuses the logits, dead once the cross-entropy read them.
    f32r1_span_t const widened_weight_grad {logits_.data(), vocabulary * hidden};
    f32r1_span_t const weight_products {products_.data(), vocabulary * hidden};
    expected_t staged = launch_kernel(&choices_gather_kernel_, segment_grid, block_shape, stream, normalized, batch,
                                      gathered_.span(), gradient_scale_.span(), config.weight);
    if (staged) staged = multiply_bf16(gathered_.view(), packed_decoder_.view(), logits, capabilities_, stream);
    if (staged)
        staged = launch_kernel(&choices_cross_entropy_kernel_, segment_grid, block_shape, stream, f32r2_view_t {logits},
                               batch, gradient_scale_.view(), config.brier_weight, position_loss_.span(), logit_grad);
    if (staged)
        staged = launch_kernel(&masked_finalize_kernel_, one_block, block_shape, stream,
                               position_loss[nk::range(std::size_t {0}, segments), nk::slice], gradient_scale_.view(),
                               loss_summands, config.weight);
    if (staged)
        staged = multiply_bf16(logit_grad, packed_decoder_transposed_.view(), gathered_grad, capabilities_, stream);
    if (staged)
        staged = launch_kernel(&choices_scatter_kernel_, segment_grid, block_shape, stream,
                               f32r2_view_t {gathered_grad}, batch, normalized_grad);
    if (staged)
        staged = numkong_result(nk_cast_best(decoder_weight_grad_.data(), nk_bf16_k, widened_weight_grad.data(),
                                             nk_f32_k, vocabulary * hidden, capabilities_, stream.numkong()),
                                "decoder gradient widening");
    if (staged)
        staged = weight_gradient_bf16(logit_grad, gathered_.view(), decoder_weight_grad_.span(),
                                      transposed_grad_.span(), transposed_input_.span(), packed_input_.span(),
                                      weight_products, b8r1_span_t {}, capabilities_, stream);
    if (staged)
        staged = numkong_result(
            nk_each_sum_f32_best(widened_weight_grad.data(), weight_products.data(), vocabulary * hidden,
                                 widened_weight_grad.data(), capabilities_, stream.numkong()),
            "decoder gradient addition");
    if (staged)
        staged = numkong_result(nk_cast_best(widened_weight_grad.data(), nk_f32_k, decoder_weight_grad_.data(),
                                             nk_bf16_k, vocabulary * hidden, capabilities_, stream.numkong()),
                                "decoder gradient narrowing");
    return staged;
}

expected_t masked_tokens_t::fill(sz::string_view_t name, st::basic_vector<differentiable_weights_t> &weights) noexcept {
    differentiable_weights_t weight;
    if (sz::failed(weight.name.assign(name)))
        return expected_t::fail(status_t::bad_alloc_k, "weight name assignment failed");
    weight.weights = decoder_weight_.span();
    weight.weights_grad = decoder_weight_grad_.view();
    if (st::failed(weights.push_back(std::move(weight))))
        return expected_t::fail(status_t::bad_alloc_k, "weight table growth failed");
    return {};
}

/** The variance floor of the per-patch target normalization, MAE's own. */
inline constexpr f32_t patch_variance_epsilon_k = 1.0e-6f;

/**
 *  @brief Sample the mask over the cells of every segment of @p modality : a cell is masked when
 *      its draw under the @p purpose salt falls under @p mask_ratio . Cells of other segments keep
 *      the zero the caller wrote.
 *
 *  Tube masking keys only the spatial index when @p spatial_cells is below the cell count, so a
 *  masked cell stays masked across time.
 *
 *  __grid_size__ @c [segments,1,1] , threads stripe the segment's cells; no dynamic shared memory.
 */
static __global__ void media_mask_sample_kernel_(u32r1_span_t mask, u32r1_view_t segment_offsets,
                                                 u32r1_view_t segment_lengths, modality_view_t segment_modality,
                                                 modality_t modality, u64_t seed, data_epoch_t data_epoch,
                                                 f32_t mask_ratio, u32_t spatial_cells,
                                                 masking_draw_purpose_t purpose) {
    u32_t const segment = blockIdx.x;
    if (segment_modality[segment] != modality) return;
    u32_t const offset = segment_offsets[segment], length = segment_lengths[segment];
    for (u32_t column = threadIdx.x; column < length; column += blockDim.x) {
        u32_t const cell = offset + column;
        f32_t const draw = masking_draw(seed, data_epoch, cell % spatial_cells, 0u, static_cast<u32_t>(purpose));
        mask[cell] = draw < mask_ratio ? 1u : 0u;
    }
}

/** Zero @p mask and sample it over the cells of every @p modality segment, the one sampler both
 *  media heads share. */
[[nodiscard]] static expected_t sample_media_mask_(u32r1_span_t mask, u32r1_view_t segment_offsets,
                                                   u32r1_view_t segment_lengths, modality_view_t segment_modality,
                                                   modality_t modality, masking_recipe_t recipe,
                                                   data_epoch_t data_epoch, f32_t mask_ratio, u32_t spatial_cells,
                                                   masking_draw_purpose_t purpose, gpu_stream_t &stream) noexcept {
    if (expected_t zeroed = zero_async(mask, stream); !zeroed) return zeroed;
    dim3 const segment_grid {static_cast<u32_t>(segment_lengths.extent(0)), 1u, 1u};
    dim3 const block_shape {kernel_block_size_default_k, 1u, 1u};
    return launch_kernel(&media_mask_sample_kernel_, segment_grid, block_shape, stream, mask, segment_offsets,
                         segment_lengths, segment_modality, modality, recipe.seed, data_epoch, mask_ratio,
                         spatial_cells, purpose);
}

/**
 *  @brief Zero the raw patch rows the mask selects, in place; visible rows keep their pixels. The driver
 *      calls this on the live batch media so the tower re-patchifies the corrupted pixels.
 *
 *  __grid_size__ @c [grid_size_for(patches),1,1] , persistent stride; no dynamic shared memory.
 */
static __global__ void patch_corrupt_raw_kernel_(nk::tensor_span<pixel_t, 2> raw, u32r1_view_t mask) {
    std::size_t const patches = mask.extent(0);
    u32_t const patch_scalars = raw.extent(1);
    for (std::size_t patch = blockIdx.x; patch < patches; patch += gridDim.x) {
        if (mask[patch] == 0u) continue;
        for (u32_t index = threadIdx.x; index < patch_scalars; index += blockDim.x) raw[patch, index] = pixel_t {0};
    }
}

/**
 *  @brief Count the masked patches and publish @c {λ·2/(patch_pixels·max(count,1)), count} — the MAE
 *      gradient's per-element constant, so the count never visits the host.
 *
 *  __grid_size__ @c [1,1,1] , one block; no dynamic shared memory.
 */
static __global__ void patch_count_kernel_(u32r1_view_t mask, f32r1_span_t scale_and_count, f32_t weight,
                                           u32_t patch_pixels) {
    f32_t live = 0.0f;
    for (std::size_t index = threadIdx.x; index < mask.extent(0); index += blockDim.x)
        live += static_cast<f32_t>(mask[index]);
    f32_t const total = block_reduce_broadcast<kernel_block_size_default_k>(live);
    if (threadIdx.x != 0u) return;
    scale_and_count[0] = weight * 2.0f / (static_cast<f32_t>(patch_pixels) * fmaxf(total, 1.0f));
    scale_and_count[1] = total;
}

/**
 *  @brief Per-patch normalized-MSE over one masked patch: z-score the target by its own mean /
 *      standard deviation, accumulate the squared difference to the raw prediction, and write the
 *      pre-scaled gradient — visible patches write zeros and score nothing.
 *
 *  __grid_size__ @c [max_patches,1,1] , one block per patch; no dynamic shared memory.
 */
static __global__ void patch_mae_loss_kernel_(bf16r2_view_t predicted, bf16r2_view_t targets, u32r1_view_t mask,
                                              f32r1_view_t scale_and_count, f32r1_span_t position_loss,
                                              bf16r2_span_t predicted_grad) {
    u32_t const row = blockIdx.x;
    bool const live = mask[row] != 0u;
    u32_t const patch_pixels = predicted.extent(1);
    f32_t const inverse_pixels = 1.0f / static_cast<f32_t>(patch_pixels);

    f32_t sum_target = 0.0f, sumsq_target = 0.0f;
    for (u32_t index = threadIdx.x; index < patch_pixels; index += blockDim.x) {
        f32_t const target = __bfloat162float(targets[row, index]);
        sum_target += target, sumsq_target += target * target;
    }
    f32_t const mean_target = block_reduce_broadcast<kernel_block_size_default_k>(sum_target) * inverse_pixels;
    f32_t const mean_sq_target = block_reduce_broadcast<kernel_block_size_default_k>(sumsq_target) * inverse_pixels;
    f32_t const inverse_std_target = rsqrtf(fmaxf(mean_sq_target - mean_target * mean_target, 0.0f) +
                                            patch_variance_epsilon_k);

    f32_t const scale = live ? scale_and_count[0] : 0.0f;
    f32_t thread_loss = 0.0f;
    for (u32_t index = threadIdx.x; index < patch_pixels; index += blockDim.x) {
        f32_t const target_norm = (__bfloat162float(targets[row, index]) - mean_target) * inverse_std_target;
        f32_t const difference = __bfloat162float(predicted[row, index]) - target_norm;
        thread_loss += difference * difference;
        predicted_grad[row, index] = __float2bfloat16(scale * difference);
    }
    f32_t const total_loss = block_reduce_broadcast<kernel_block_size_default_k>(thread_loss);
    if (threadIdx.x != 0u) return;
    position_loss[row] = live ? total_loss * inverse_pixels : 0.0f;
}

expected<masked_patches_t> masked_patches_t::make(gpu_arena_t &arena, masked_patches_config_t config,
                                                  nk_capability_t capabilities) noexcept {
    using result_t = expected<masked_patches_t>;
    if (config.hidden == 0 || config.patch_pixels == 0 || config.max_patches == 0 || config.temporal_groups == 0 ||
        config.max_patches % config.temporal_groups != 0)
        return result_t::fail(status_t::shape_unsupported_k,
                              "the patch head needs a full geometry (hidden={} pixels={} patches={} groups={})",
                              config.hidden, config.patch_pixels, config.max_patches, config.temporal_groups);
    masked_patches_t head;
    head.config_ = config;
    head.capabilities_ = capabilities;

    gpu_arena_alloc<bf16_t> bf16_alloc {arena};
    gpu_arena_alloc<f32_t> f32_alloc {arena};
    gpu_arena_alloc<u32_t> u32_alloc {arena};
    head.recon_weight_ = bf16r2_tensor_arena_t::uninitialized({config.patch_pixels, config.hidden}, bf16_alloc).value;
    head.recon_weight_grad_ =
        bf16r2_tensor_arena_t::uninitialized({config.patch_pixels, config.hidden}, bf16_alloc).value;
    head.predicted_ = bf16r2_tensor_arena_t::uninitialized({config.max_patches, config.patch_pixels}, bf16_alloc).value;
    head.predicted_grad_ =
        bf16r2_tensor_arena_t::uninitialized({config.max_patches, config.patch_pixels}, bf16_alloc).value;
    head.mask_ = u32r1_tensor_arena_t::uninitialized({config.max_patches}, u32_alloc).value;
    head.position_loss_ = f32r1_tensor_arena_t::uninitialized({config.max_patches}, f32_alloc).value;
    head.gradient_scale_ = f32r1_tensor_arena_t::uninitialized({2}, f32_alloc).value;
    if (!head.recon_weight_ || !head.recon_weight_grad_ || !head.predicted_ || !head.predicted_grad_ || !head.mask_ ||
        !head.position_loss_ || !head.gradient_scale_)
        return result_t::fail(status_t::bad_alloc_k, "patch head allocation failed (arena_used={})", arena.used());

    head.widened_grad_ = f32r1_tensor_arena_t::uninitialized({config.max_patches * config.hidden}, f32_alloc).value;
    if (!head.widened_grad_)
        return result_t::fail(status_t::bad_alloc_k, "patch head allocation failed (arena_used={})", arena.used());
    if (expected_t allocated = allocate_head_products_(arena, head.capabilities_, config.patch_pixels, config.hidden,
                                                       config.max_patches, head.packed_recon_,
                                                       head.packed_recon_transposed_, head.transposed_grad_,
                                                       head.transposed_input_, head.packed_input_, head.products_);
        !allocated)
        return result_t::fail(std::move(allocated));
    return result_t {std::move(head)};
}

expected_t masked_patches_t::pack_weights(gpu_stream_t &stream) noexcept {
    return pack_weight_bf16(recon_weight_.view(), transposed_input_.span(), packed_recon_.span(),
                            packed_recon_transposed_.span(), capabilities_, stream);
}

expected_t masked_patches_t::sample_mask(masking_recipe_t recipe, data_epoch_t data_epoch, u32r1_view_t segment_offsets,
                                         u32r1_view_t segment_lengths, modality_view_t segment_modality,
                                         gpu_stream_t &stream) noexcept {
    return sample_media_mask_(mask_.span(), segment_offsets, segment_lengths, segment_modality, config_.modality,
                              recipe, data_epoch, config_.mask_ratio, config_.max_patches / config_.temporal_groups,
                              masking_draw_purpose_t::patch_mask_k, stream);
}

expected_t masked_patches_t::corrupt_raw(nk::tensor_span<pixel_t, 2> raw_patches, gpu_stream_t &stream) noexcept {
    u32_t const patches = config_.max_patches;
    if (raw_patches.extent(0) != patches)
        return expected_t::fail(status_t::shape_unsupported_k, "corrupt_raw expects [{},patch_scalars] patch rows",
                                patches);
    dim3 const block_shape {kernel_block_size_default_k, 1u, 1u};
    dim3 const patch_grid {grid_size_for(patches), 1u, 1u};
    return launch_kernel(&patch_corrupt_raw_kernel_, patch_grid, block_shape, stream, raw_patches, mask_.view());
}

expected_t masked_patches_t::operator()(bf16r2_view_t normalized, bf16r2_view_t clean_patches,
                                        f32r1_span_t loss_summands, bf16r2_span_t normalized_grad,
                                        gpu_stream_t &stream) noexcept {
    u32_t const hidden = config_.hidden, patch_pixels = config_.patch_pixels;
    std::size_t const patches = config_.max_patches;
    if (normalized.extent(0) != patches || normalized_grad.extent(0) != patches || clean_patches.extent(0) != patches)
        return expected_t::fail(status_t::shape_unsupported_k, "patch head count mismatch (expected {})", patches);
    if (normalized.extent(1) != hidden || normalized_grad.extent(1) != hidden ||
        clean_patches.extent(1) != patch_pixels || loss_summands.extent(0) != 2)
        return expected_t::fail(status_t::shape_unsupported_k, "patch head geometry mismatch (hidden={})", hidden);

    dim3 const one_block {1u, 1u, 1u};
    dim3 const block_shape {kernel_block_size_default_k, 1u, 1u};
    dim3 const row_grid {static_cast<u32_t>(patches), 1u, 1u};
    bf16r2_view_t const predicted_view = predicted_.view();
    bf16r2_span_t const predicted_grad_span = predicted_grad_.span();
    std::size_t const rows = patches, outputs = patch_pixels;
    f32r2_span_t const predicted_products {products_.data(), rows, outputs};
    f32r2_span_t const normalized_products {products_.data(), rows, static_cast<std::size_t>(hidden)};
    expected_t staged = launch_kernel(&patch_count_kernel_, one_block, block_shape, stream, mask_.view(),
                                      gradient_scale_.span(), config_.weight, patch_pixels);
    if (staged) staged = multiply_bf16(normalized, packed_recon_.view(), predicted_products, capabilities_, stream);
    if (staged)
        staged = numkong_result(nk_cast_best(predicted_products.data(), nk_f32_k, predicted_.data(), nk_bf16_k,
                                             rows * outputs, capabilities_, stream.numkong()),
                                "prediction narrowing");
    if (staged)
        staged = launch_kernel(&patch_mae_loss_kernel_, row_grid, block_shape, stream, predicted_view, clean_patches,
                               mask_.view(), gradient_scale_.view(), position_loss_.span(), predicted_grad_span);
    if (staged)
        staged = launch_kernel(&masked_finalize_kernel_, one_block, block_shape, stream, position_loss_.view(),
                               gradient_scale_.view(), loss_summands, config_.weight);
    if (staged)
        staged = multiply_bf16(predicted_grad_.view(), packed_recon_transposed_.view(), normalized_products,
                               capabilities_, stream);
    if (staged)
        staged = numkong_result(nk_cast_best(normalized_grad.data(), nk_bf16_k, widened_grad_.data(), nk_f32_k,
                                             rows * hidden, capabilities_, stream.numkong()),
                                "normalized gradient widening");
    if (staged)
        staged = numkong_result(nk_each_sum_f32_best(widened_grad_.data(), normalized_products.data(), rows * hidden,
                                                     widened_grad_.data(), capabilities_, stream.numkong()),
                                "normalized gradient addition");
    if (staged)
        staged = numkong_result(nk_cast_best(widened_grad_.data(), nk_f32_k, normalized_grad.data(), nk_bf16_k,
                                             rows * hidden, capabilities_, stream.numkong()),
                                "normalized gradient narrowing");
    if (staged)
        staged = weight_gradient_bf16(predicted_grad_.view(), normalized, recon_weight_grad_.span(),
                                      transposed_grad_.span(), transposed_input_.span(), packed_input_.span(),
                                      products_.span(), b8r1_span_t {}, capabilities_, stream);
    return staged;
}

expected_t masked_patches_t::fill(sz::string_view_t name,
                                  st::basic_vector<differentiable_weights_t> &weights) noexcept {
    differentiable_weights_t weight;
    if (sz::failed(weight.name.assign(name)))
        return expected_t::fail(status_t::bad_alloc_k, "weight name assignment failed");
    weight.weights = recon_weight_.span();
    weight.weights_grad = recon_weight_grad_.view();
    if (st::failed(weights.push_back(std::move(weight))))
        return expected_t::fail(status_t::bad_alloc_k, "weight table growth failed");
    return {};
}

/**
 *  @brief Zero the raw waveform frames the mask selects, in place; visible frames keep their Float32
 *      samples. The driver calls this on the live batch audio so the tower re-projects the corrupted samples.
 *
 *  __grid_size__ @c [grid_size_for(frames),1,1] , persistent stride; no dynamic shared memory.
 */
static __global__ void frame_corrupt_raw_kernel_(f32r2_span_t raw, u32r1_view_t mask) {
    std::size_t const frames = mask.extent(0);
    u32_t const chunk_samples = raw.extent(1);
    for (std::size_t frame = blockIdx.x; frame < frames; frame += gridDim.x) {
        if (mask[frame] == 0u) continue;
        for (u32_t index = threadIdx.x; index < chunk_samples; index += blockDim.x) raw[frame, index] = 0.0f;
    }
}

/**
 *  @brief Count the masked frames and publish @c {λ·2/(chunk_samples·max(count,1)), count} — the MAE
 *      gradient's per-element constant, so the count never visits the host.
 *
 *  __grid_size__ @c [1,1,1] , one block; no dynamic shared memory.
 */
static __global__ void frame_count_kernel_(u32r1_view_t mask, f32r1_span_t scale_and_count, f32_t weight,
                                           u32_t chunk_samples) {
    f32_t live = 0.0f;
    for (std::size_t index = threadIdx.x; index < mask.extent(0); index += blockDim.x)
        live += static_cast<f32_t>(mask[index]);
    f32_t const total = block_reduce_broadcast<kernel_block_size_default_k>(live);
    if (threadIdx.x != 0u) return;
    scale_and_count[0] = weight * 2.0f / (static_cast<f32_t>(chunk_samples) * fmaxf(total, 1.0f));
    scale_and_count[1] = total;
}

/**
 *  @brief Per-frame normalized-MSE over one masked audio frame: z-score the raw Float32 target by
 *      its own mean / standard deviation, accumulate the squared difference to the raw prediction,
 *      and write the pre-scaled gradient — visible frames write zeros and score nothing.
 *
 *  The prediction rides BFloat16 off the decode matmul; the target is the raw waveform frame, read
 *  as Float32.
 *
 *  __grid_size__ @c [max_frames,1,1] , one block per frame; no dynamic shared memory.
 */
static __global__ void frame_mae_loss_kernel_(bf16r2_view_t predicted, f32r2_view_t targets, u32r1_view_t mask,
                                              f32r1_view_t scale_and_count, f32r1_span_t position_loss,
                                              bf16r2_span_t predicted_grad) {
    u32_t const row = blockIdx.x;
    bool const live = mask[row] != 0u;
    u32_t const chunk_samples = predicted.extent(1);
    f32_t const inverse_samples = 1.0f / static_cast<f32_t>(chunk_samples);

    f32_t sum_target = 0.0f, sumsq_target = 0.0f;
    for (u32_t index = threadIdx.x; index < chunk_samples; index += blockDim.x) {
        f32_t const target = targets[row, index];
        sum_target += target, sumsq_target += target * target;
    }
    f32_t const mean_target = block_reduce_broadcast<kernel_block_size_default_k>(sum_target) * inverse_samples;
    f32_t const mean_sq_target = block_reduce_broadcast<kernel_block_size_default_k>(sumsq_target) * inverse_samples;
    f32_t const inverse_std_target = rsqrtf(fmaxf(mean_sq_target - mean_target * mean_target, 0.0f) +
                                            patch_variance_epsilon_k);

    f32_t const scale = live ? scale_and_count[0] : 0.0f;
    f32_t thread_loss = 0.0f;
    for (u32_t index = threadIdx.x; index < chunk_samples; index += blockDim.x) {
        f32_t const target_norm = (targets[row, index] - mean_target) * inverse_std_target;
        f32_t const difference = __bfloat162float(predicted[row, index]) - target_norm;
        thread_loss += difference * difference;
        predicted_grad[row, index] = __float2bfloat16(scale * difference);
    }
    f32_t const total_loss = block_reduce_broadcast<kernel_block_size_default_k>(thread_loss);
    if (threadIdx.x != 0u) return;
    position_loss[row] = live ? total_loss * inverse_samples : 0.0f;
}

expected<masked_frames_t> masked_frames_t::make(gpu_arena_t &arena, masked_frames_config_t config,
                                                nk_capability_t capabilities) noexcept {
    using result_t = expected<masked_frames_t>;
    if (config.hidden == 0 || config.chunk_samples == 0 || config.max_frames == 0)
        return result_t::fail(status_t::shape_unsupported_k,
                              "the frame head needs a full geometry (hidden={} samples={} frames={})", config.hidden,
                              config.chunk_samples, config.max_frames);
    masked_frames_t head;
    head.config_ = config;
    head.capabilities_ = capabilities;

    gpu_arena_alloc<bf16_t> bf16_alloc {arena};
    gpu_arena_alloc<f32_t> f32_alloc {arena};
    gpu_arena_alloc<u32_t> u32_alloc {arena};
    head.recon_weight_ = bf16r2_tensor_arena_t::uninitialized({config.chunk_samples, config.hidden}, bf16_alloc).value;
    head.recon_weight_grad_ =
        bf16r2_tensor_arena_t::uninitialized({config.chunk_samples, config.hidden}, bf16_alloc).value;
    head.predicted_ = bf16r2_tensor_arena_t::uninitialized({config.max_frames, config.chunk_samples}, bf16_alloc).value;
    head.predicted_grad_ =
        bf16r2_tensor_arena_t::uninitialized({config.max_frames, config.chunk_samples}, bf16_alloc).value;
    head.mask_ = u32r1_tensor_arena_t::uninitialized({config.max_frames}, u32_alloc).value;
    head.position_loss_ = f32r1_tensor_arena_t::uninitialized({config.max_frames}, f32_alloc).value;
    head.gradient_scale_ = f32r1_tensor_arena_t::uninitialized({2}, f32_alloc).value;
    if (!head.recon_weight_ || !head.recon_weight_grad_ || !head.predicted_ || !head.predicted_grad_ || !head.mask_ ||
        !head.position_loss_ || !head.gradient_scale_)
        return result_t::fail(status_t::bad_alloc_k, "frame head allocation failed (arena_used={})", arena.used());

    head.widened_grad_ = f32r1_tensor_arena_t::uninitialized({config.max_frames * config.hidden}, f32_alloc).value;
    if (!head.widened_grad_)
        return result_t::fail(status_t::bad_alloc_k, "frame head allocation failed (arena_used={})", arena.used());
    if (expected_t allocated = allocate_head_products_(arena, head.capabilities_, config.chunk_samples, config.hidden,
                                                       config.max_frames, head.packed_recon_,
                                                       head.packed_recon_transposed_, head.transposed_grad_,
                                                       head.transposed_input_, head.packed_input_, head.products_);
        !allocated)
        return result_t::fail(std::move(allocated));
    return result_t {std::move(head)};
}

expected_t masked_frames_t::pack_weights(gpu_stream_t &stream) noexcept {
    return pack_weight_bf16(recon_weight_.view(), transposed_input_.span(), packed_recon_.span(),
                            packed_recon_transposed_.span(), capabilities_, stream);
}

expected_t masked_frames_t::sample_mask(masking_recipe_t recipe, data_epoch_t data_epoch, u32r1_view_t segment_offsets,
                                        u32r1_view_t segment_lengths, modality_view_t segment_modality,
                                        gpu_stream_t &stream) noexcept {
    return sample_media_mask_(mask_.span(), segment_offsets, segment_lengths, segment_modality, modality_t::audio_k,
                              recipe, data_epoch, config_.mask_ratio, config_.max_frames,
                              masking_draw_purpose_t::frame_mask_k, stream);
}

expected_t masked_frames_t::corrupt_raw(f32r2_span_t raw_frames, gpu_stream_t &stream) noexcept {
    u32_t const frames = config_.max_frames;
    if (raw_frames.extent(0) != frames)
        return expected_t::fail(status_t::shape_unsupported_k, "corrupt_raw expects [{},chunk_samples] frame rows",
                                frames);
    dim3 const block_shape {kernel_block_size_default_k, 1u, 1u};
    dim3 const frame_grid {grid_size_for(frames), 1u, 1u};
    return launch_kernel(&frame_corrupt_raw_kernel_, frame_grid, block_shape, stream, raw_frames, mask_.view());
}

expected_t masked_frames_t::operator()(bf16r2_view_t normalized, f32r2_view_t clean_frames, f32r1_span_t loss_summands,
                                       bf16r2_span_t normalized_grad, gpu_stream_t &stream) noexcept {
    u32_t const hidden = config_.hidden, chunk_samples = config_.chunk_samples;
    std::size_t const frames = config_.max_frames;
    if (normalized.extent(0) != frames || normalized_grad.extent(0) != frames || clean_frames.extent(0) != frames)
        return expected_t::fail(status_t::shape_unsupported_k, "frame head count mismatch (expected {})", frames);
    if (normalized.extent(1) != hidden || normalized_grad.extent(1) != hidden ||
        clean_frames.extent(1) != chunk_samples || loss_summands.extent(0) != 2)
        return expected_t::fail(status_t::shape_unsupported_k, "frame head geometry mismatch (hidden={})", hidden);

    dim3 const one_block {1u, 1u, 1u};
    dim3 const block_shape {kernel_block_size_default_k, 1u, 1u};
    dim3 const row_grid {static_cast<u32_t>(frames), 1u, 1u};
    bf16r2_view_t const predicted_view = predicted_.view();
    bf16r2_span_t const predicted_grad_span = predicted_grad_.span();
    std::size_t const rows = frames, outputs = chunk_samples;
    f32r2_span_t const predicted_products {products_.data(), rows, outputs};
    f32r2_span_t const normalized_products {products_.data(), rows, static_cast<std::size_t>(hidden)};
    expected_t staged = launch_kernel(&frame_count_kernel_, one_block, block_shape, stream, mask_.view(),
                                      gradient_scale_.span(), config_.weight, chunk_samples);
    if (staged) staged = multiply_bf16(normalized, packed_recon_.view(), predicted_products, capabilities_, stream);
    if (staged)
        staged = numkong_result(nk_cast_best(predicted_products.data(), nk_f32_k, predicted_.data(), nk_bf16_k,
                                             rows * outputs, capabilities_, stream.numkong()),
                                "prediction narrowing");
    if (staged)
        staged = launch_kernel(&frame_mae_loss_kernel_, row_grid, block_shape, stream, predicted_view, clean_frames,
                               mask_.view(), gradient_scale_.view(), position_loss_.span(), predicted_grad_span);
    if (staged)
        staged = launch_kernel(&masked_finalize_kernel_, one_block, block_shape, stream, position_loss_.view(),
                               gradient_scale_.view(), loss_summands, config_.weight);
    if (staged)
        staged = multiply_bf16(predicted_grad_.view(), packed_recon_transposed_.view(), normalized_products,
                               capabilities_, stream);
    if (staged)
        staged = numkong_result(nk_cast_best(normalized_grad.data(), nk_bf16_k, widened_grad_.data(), nk_f32_k,
                                             rows * hidden, capabilities_, stream.numkong()),
                                "normalized gradient widening");
    if (staged)
        staged = numkong_result(nk_each_sum_f32_best(widened_grad_.data(), normalized_products.data(), rows * hidden,
                                                     widened_grad_.data(), capabilities_, stream.numkong()),
                                "normalized gradient addition");
    if (staged)
        staged = numkong_result(nk_cast_best(widened_grad_.data(), nk_f32_k, normalized_grad.data(), nk_bf16_k,
                                             rows * hidden, capabilities_, stream.numkong()),
                                "normalized gradient narrowing");
    if (staged)
        staged = weight_gradient_bf16(predicted_grad_.view(), normalized, recon_weight_grad_.span(),
                                      transposed_grad_.span(), transposed_input_.span(), packed_input_.span(),
                                      products_.span(), b8r1_span_t {}, capabilities_, stream);
    return staged;
}

expected_t masked_frames_t::fill(sz::string_view_t name, st::basic_vector<differentiable_weights_t> &weights) noexcept {
    differentiable_weights_t weight;
    if (sz::failed(weight.name.assign(name)))
        return expected_t::fail(status_t::bad_alloc_k, "weight name assignment failed");
    weight.weights = recon_weight_.span();
    weight.weights_grad = recon_weight_grad_.view();
    if (st::failed(weights.push_back(std::move(weight))))
        return expected_t::fail(status_t::bad_alloc_k, "weight table growth failed");
    return {};
}
