/**
 *  @brief The multimodal masked-modeling objective's implementation: the corruption and reconstruction
 *      kernels and the construction bodies — compiled once into the `uform` library.
 *  @file cuda_core/multimodal_masking.cu
 *  @author Ash Vardanian
 *  @date July 3, 2026
 */
#include "cuda_core/multimodal_masking.hpp"

using namespace unum::uform::core;

/** @brief All-threads-get-the-total CTA max — the fixed-order sibling of @ref block_reduce_broadcast for
 *      the softmax's row maximum: per-warp shuffle trees, lane-0 partials through static shared memory,
 *      one final warp tree, one broadcast cell. Deterministic, sized for @p block_size_ threads. */
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
        f32_t staged = lane_index < warps_k ? warp_partials[lane_index] : -INFINITY;
        for (unsigned delta = warp_size_k / 2u; delta != 0u; delta /= 2u)
            staged = fmaxf(staged, __shfl_xor_sync(0xffffffffu, staged, delta));
        if (lane_index == 0u) block_total = staged;
    }
    __syncthreads();
    return block_total;
}

/**
 *  @brief Corrupt one segment in place: walk its tokens left-to-right, open geometric spans, and at each
 *      covered position write the distorted byte (MASK / random / kept clean) and a loss weight of one.
 *      PAD breaks and never opens a span; uncovered positions were already copied clean with weight zero.
 *      A non- @c text_k segment is skipped outright — media rows keep their clean tokens and zero weight.
 *
 *  __grid_size__ @c [grid_size_for(segments),1,1] , one thread per segment; no dynamic shared memory.
 */
static __global__ void masked_token_corrupt_kernel_(tokens_view_t clean, tokens_span_t distorted,
                                                    u32r1_span_t loss_weights, u32r1_view_t segment_offsets,
                                                    u32r1_view_t segment_lengths, modality_view_t segment_modality,
                                                    u64_t seed, data_epoch_t data_epoch, f32_t open_rate,
                                                    f32_t continue_rate, f32_t replace_fraction,
                                                    f32_t random_fraction) {
    u32_t const segment = blockIdx.x * blockDim.x + threadIdx.x;
    if (segment >= segment_lengths.extent(0)) return;
    if (segment_modality[segment] != modality_t::text_k) return; // media segments stay clean, weight zero
    u32_t const base = segment_offsets[segment];
    u32_t const length = segment_lengths[segment];
    bool in_span = false;
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
        f32_t const action = masking_draw(seed, data_epoch, segment, column,
                                          static_cast<u32_t>(masking_draw_purpose_t::token_action_k));
        if (action < replace_fraction) { distorted[position] = byte_mask_k; }
        else if (action < replace_fraction + random_fraction)
            distorted[position] = static_cast<token_id_t>(
                masking_draw(seed, data_epoch, segment, column,
                             static_cast<u32_t>(masking_draw_purpose_t::random_byte_k)) *
                256.0f);
        loss_weights[position] = 1u; // the remaining fraction keeps the clean byte, still scored
    }
}

/** @brief Count the loss-bearing positions and publish @c {λ/max(count,1), count} — one scalar the CE
 *      kernel scales gradients by, so the masked count never visits the host.
 *
 *  __grid_size__ @c [1,1,1] , one block; no dynamic shared memory. */
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

    f32_t thread_max = -INFINITY;
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

/** @brief Fold the per-position CE summands into @c {λ·Σ, count} — the rank's loss surface.
 *
 *  __grid_size__ @c [1,1,1] , one block, fixed-order tree; no dynamic shared memory. */
static __global__ void masked_finalize_kernel_(f32r1_view_t position_loss, f32r1_view_t scale_and_count,
                                               f32r1_span_t loss_summands, f32_t weight) {
    f32_t thread_sum = 0.0f;
    for (std::size_t index = threadIdx.x; index < position_loss.extent(0); index += blockDim.x)
        thread_sum += position_loss[index];
    f32_t const total = block_reduce_broadcast<kernel_block_size_default_k>(thread_sum);
    if (threadIdx.x != 0u) return;
    loss_summands[0] = weight * total;
    loss_summands[1] = scale_and_count[1];
}

expected<masked_tokens_t> masked_tokens_t::make(gpu_arena_t &arena, cublaslt_session_t &blas,
                                                masked_tokens_config_t config) noexcept {
    using result_t = expected<masked_tokens_t>;
    if (config.vocabulary == 0 || config.hidden == 0 || config.max_tokens == 0)
        return result_t::fail(status_t::shape_unsupported_k,
                              "the masked head needs a full geometry (vocabulary={} hidden={} tokens={})",
                              config.vocabulary, config.hidden, config.max_tokens);
    masked_tokens_t head;
    head.config_ = config;
    head.session_ = &blas;

    gpu_arena_alloc<bf16_t> bf16_alloc {arena};
    gpu_arena_alloc<f32_t> f32_alloc {arena};
    gpu_arena_alloc<b8_t> byte_alloc {arena};
    head.decoder_weight_ = bf16r2_tensor_arena_t::uninitialized({config.vocabulary, config.hidden}, bf16_alloc).value;
    head.decoder_weight_grad_ =
        bf16r2_tensor_arena_t::uninitialized({config.vocabulary, config.hidden}, bf16_alloc).value;
    head.logits_ = f32r2_tensor_arena_t::uninitialized({config.max_tokens, config.vocabulary}, f32_alloc).value;
    head.logit_grad_ = bf16r2_tensor_arena_t::uninitialized({config.max_tokens, config.vocabulary}, bf16_alloc).value;
    head.position_loss_ = f32r1_tensor_arena_t::uninitialized({config.max_tokens}, f32_alloc).value;
    head.gradient_scale_ = f32r1_tensor_arena_t::uninitialized({2}, f32_alloc).value;
    if (!head.decoder_weight_ || !head.decoder_weight_grad_ || !head.logits_ || !head.logit_grad_ ||
        !head.position_loss_ || !head.gradient_scale_)
        return result_t::fail(status_t::bad_alloc_k, "masked head allocation failed (arena_used={})", arena.used());

    cublaslt_plan_config_t logits_config; // logits[M,V] = normalized[M,H]·Wᵀ, Float32 out for the CE
    logits_config.a_rows = config.max_tokens, logits_config.a_cols = config.hidden;
    logits_config.a_row_stride = config.hidden;
    logits_config.b_rows = config.vocabulary, logits_config.b_cols = config.hidden;
    logits_config.b_row_stride = config.hidden;
    logits_config.c_rows = config.max_tokens, logits_config.c_cols = config.vocabulary;
    logits_config.c_row_stride = config.vocabulary;
    logits_config.transpose_b = true;
    cublaslt_plan_config_t data_grad_config; // d(normalized) = d(logits) · W
    data_grad_config.a_rows = config.max_tokens, data_grad_config.a_cols = config.vocabulary;
    data_grad_config.a_row_stride = config.vocabulary;
    data_grad_config.b_rows = config.vocabulary, data_grad_config.b_cols = config.hidden;
    data_grad_config.b_row_stride = config.hidden;
    data_grad_config.c_rows = config.max_tokens, data_grad_config.c_cols = config.hidden;
    data_grad_config.c_row_stride = config.hidden;
    data_grad_config.c_type = CUDA_R_16BF;
    cublaslt_plan_config_t weight_grad_config; // d(W) = d(logits)ᵀ · normalized
    weight_grad_config.a_rows = config.max_tokens, weight_grad_config.a_cols = config.vocabulary;
    weight_grad_config.a_row_stride = config.vocabulary;
    weight_grad_config.b_rows = config.max_tokens, weight_grad_config.b_cols = config.hidden;
    weight_grad_config.b_row_stride = config.hidden;
    weight_grad_config.c_rows = config.vocabulary, weight_grad_config.c_cols = config.hidden;
    weight_grad_config.c_row_stride = config.hidden;
    weight_grad_config.transpose_a = true;
    weight_grad_config.c_type = CUDA_R_16BF;
    cublaslt_plan_config_t const configs[linear_matmul_family_count_k] = {logits_config, data_grad_config,
                                                                          weight_grad_config};
    std::size_t workspace_bytes = 0;
    for (u32_t family = 0; family != linear_matmul_family_count_k; ++family) {
        expected<cublaslt_plan_t> plan = cublaslt_plan_t::make(blas, configs[family]);
        if (!plan) return result_t::fail(std::move(plan));
        if (plan.value_.workspace_bytes() > workspace_bytes) workspace_bytes = plan.value_.workspace_bytes();
        head.plans_[family] = std::move(plan.value_);
    }
    if (workspace_bytes != 0) {
        head.workspace_ = b8r1_tensor_arena_t::uninitialized({workspace_bytes}, byte_alloc).value;
        if (!head.workspace_)
            return result_t::fail(status_t::bad_alloc_k, "tensor-core workspace allocation failed ({} bytes)",
                                  workspace_bytes);
    }
    return result_t {std::move(head)};
}

expected_t masked_tokens_t::corrupt(tokens_view_t clean, tokens_span_t distorted, u32r1_span_t loss_weights,
                                    u32r1_view_t segment_offsets, u32r1_view_t segment_lengths,
                                    modality_view_t segment_modality, masking_recipe_t recipe, data_epoch_t data_epoch,
                                    gpu_stream_t &stream) noexcept {
    std::size_t const tokens = clean.extent(0);
    if (distorted.extent(0) != tokens || loss_weights.extent(0) != tokens)
        return expected_t::fail(status_t::shape_unsupported_k, "corrupt buffers disagree on token count ({})", tokens);
    if (recipe.token_mean_span == 0.0f) // the geometric open / continue rates divide by it below
        return expected_t::fail(status_t::shape_unsupported_k, "masking recipe needs a non-zero token_mean_span");
    if (recipe.replace_fraction + recipe.random_fraction > 1.0f) // the kept fraction is the remainder, must stay ≥ 0
        return expected_t::fail(status_t::shape_unsupported_k,
                                "masking recipe replace+random fractions exceed one ({} + {})", recipe.replace_fraction,
                                recipe.random_fraction);
    if (cuMemcpyDtoDAsync(reinterpret_cast<CUdeviceptr>(distorted.data()), reinterpret_cast<CUdeviceptr>(clean.data()),
                          tokens * sizeof(token_id_t), stream.get()) != CUDA_SUCCESS)
        return expected_t::fail(status_t::cuda_error_k, "distorted seed copy failed ({} tokens)", tokens);
    if (cuMemsetD8Async(reinterpret_cast<CUdeviceptr>(loss_weights.data()), 0, tokens * sizeof(u32_t), stream.get()) !=
        CUDA_SUCCESS)
        return expected_t::fail(status_t::cuda_error_k, "loss-weight reset failed ({} tokens)", tokens);

    f32_t const open_rate = recipe.token_mask_probability / recipe.token_mean_span;
    f32_t const continue_rate = 1.0f - 1.0f / recipe.token_mean_span;
    std::size_t const segments = segment_lengths.extent(0);
    dim3 const block_shape {kernel_block_size_default_k, 1u, 1u};
    dim3 const segment_grid {grid_size_for(segments), 1u, 1u};
    return launch_kernel(reinterpret_cast<void const *>(&masked_token_corrupt_kernel_), segment_grid, block_shape,
                         stream, clean, distorted, loss_weights, segment_offsets, segment_lengths, segment_modality,
                         recipe.seed, data_epoch, open_rate, continue_rate, recipe.replace_fraction,
                         recipe.random_fraction);
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
    expected_t staged = launch_kernel(reinterpret_cast<void const *>(&masked_count_kernel_), one_block, block_shape,
                                      stream, loss_weights, gradient_scale_.span(), config_.weight);
    if (staged)
        staged = plan_for(linear_matmul_family_t::forward_k)
                     .matmul(*session_, normalized, decoder_weight_.view(), logits_.span(), 0.0f, workspace_.span(),
                             stream);
    if (staged)
        staged = launch_kernel(reinterpret_cast<void const *>(&masked_cross_entropy_kernel_), row_grid, block_shape,
                               stream, logits_.view(), clean_targets, loss_weights, gradient_scale_.view(),
                               position_loss_.span(), logit_grad_.span());
    if (staged)
        staged = launch_kernel(reinterpret_cast<void const *>(&masked_finalize_kernel_), one_block, block_shape, stream,
                               position_loss_.view(), gradient_scale_.view(), loss_summands, config_.weight);
    if (staged)
        staged = plan_for(linear_matmul_family_t::data_grad_k)
                     .matmul(*session_, logit_grad_.view(), decoder_weight_.view(), normalized_grad, 0.0f,
                             workspace_.span(), stream);
    if (staged)
        staged = plan_for(linear_matmul_family_t::weight_grad_k)
                     .matmul(*session_, logit_grad_.view(), normalized, decoder_weight_grad_.span(), 0.0f,
                             workspace_.span(), stream);
    return staged;
}

expected_t masked_tokens_t::fill(sz::string_view name, array<differentiable_weights_t> &weights) noexcept {
    differentiable_weights_t weight;
    if (!weight.name.try_assign(name)) return expected_t::fail(status_t::bad_alloc_k, "weight name assignment failed");
    weight.weights = decoder_weight_.span();
    weight.weights_grad = decoder_weight_grad_.view();
    if (weights.try_push_back(std::move(weight)) != status_t::success_k)
        return expected_t::fail(status_t::bad_alloc_k, "weight table growth failed");
    return {};
}

/** @brief The variance floor of the per-patch z-scoring — the MAE loss normalizes both the prediction and
 *      the target by their own mean and standard deviation. */
inline constexpr f32_t patch_variance_epsilon_k = 1.0e-6f;

/** @brief Sample the patch mask: a patch is masked when its draw falls under @p mask_ratio . Tube masking
 *      (@p temporal_groups > 1 ) keys only the spatial index, so a masked cell stays masked across time.
 *
 *  __grid_size__ @c [grid_size_for(patches),1,1] , one thread per patch; no dynamic shared memory. */
static __global__ void patch_mask_sample_kernel_(u32r1_span_t mask, u64_t seed, data_epoch_t data_epoch,
                                                 f32_t mask_ratio, u32_t spatial_patches) {
    u32_t const patch = blockIdx.x * blockDim.x + threadIdx.x;
    if (patch >= mask.extent(0)) return;
    u32_t const key = patch % spatial_patches; // == patch when spatial_patches == max_patches (random)
    f32_t const draw = masking_draw(seed, data_epoch, key, 0u,
                                    static_cast<u32_t>(masking_draw_purpose_t::patch_mask_k));
    mask[patch] = draw < mask_ratio ? 1u : 0u;
}

/** @brief Zero the raw patch rows the mask selects, in place; visible rows keep their pixels. The driver
 *      calls this on the live batch media so the tower re-patchifies the corrupted pixels.
 *
 *  __grid_size__ @c [grid_size_for(patches),1,1] , persistent stride; no dynamic shared memory. */
static __global__ void patch_corrupt_raw_kernel_(nk::tensor_span<pixel_t, 2> raw, u32r1_view_t mask) {
    std::size_t const patches = mask.extent(0);
    u32_t const patch_scalars = raw.extent(1);
    for (std::size_t patch = blockIdx.x; patch < patches; patch += gridDim.x) {
        if (mask[patch] == 0u) continue;
        for (u32_t index = threadIdx.x; index < patch_scalars; index += blockDim.x) raw[patch, index] = pixel_t {0};
    }
}

/** @brief Count the masked patches and publish @c {λ·2/(patch_pixels·max(count,1)), count} — the MAE
 *      gradient's per-element constant, so the count never visits the host.
 *
 *  __grid_size__ @c [1,1,1] , one block; no dynamic shared memory. */
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
 *  @brief Per-patch normalized-MSE over one masked patch: z-score the prediction and the target by their
 *      own mean / standard deviation, accumulate the squared difference, and write the pre-scaled
 *      gradient — visible patches write zeros and score nothing.
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

    f32_t sum_pred = 0.0f, sumsq_pred = 0.0f, sum_target = 0.0f, sumsq_target = 0.0f;
    for (u32_t index = threadIdx.x; index < patch_pixels; index += blockDim.x) {
        f32_t const p = __bfloat162float(predicted[row, index]), t = __bfloat162float(targets[row, index]);
        sum_pred += p, sumsq_pred += p * p, sum_target += t, sumsq_target += t * t;
    }
    f32_t const mean_pred = block_reduce_broadcast<kernel_block_size_default_k>(sum_pred) * inverse_pixels;
    f32_t const mean_sq_pred = block_reduce_broadcast<kernel_block_size_default_k>(sumsq_pred) * inverse_pixels;
    f32_t const mean_target = block_reduce_broadcast<kernel_block_size_default_k>(sum_target) * inverse_pixels;
    f32_t const mean_sq_target = block_reduce_broadcast<kernel_block_size_default_k>(sumsq_target) * inverse_pixels;
    f32_t const inverse_std_pred = rsqrtf(fmaxf(mean_sq_pred - mean_pred * mean_pred, 0.0f) + patch_variance_epsilon_k);
    f32_t const inverse_std_target = rsqrtf(fmaxf(mean_sq_target - mean_target * mean_target, 0.0f) +
                                            patch_variance_epsilon_k);

    f32_t const scale = live ? scale_and_count[0] * inverse_std_pred : 0.0f;
    f32_t thread_loss = 0.0f;
    for (u32_t index = threadIdx.x; index < patch_pixels; index += blockDim.x) {
        f32_t const pred_norm = (__bfloat162float(predicted[row, index]) - mean_pred) * inverse_std_pred;
        f32_t const target_norm = (__bfloat162float(targets[row, index]) - mean_target) * inverse_std_target;
        f32_t const difference = pred_norm - target_norm;
        thread_loss += difference * difference;
        predicted_grad[row, index] = __float2bfloat16(scale * difference);
    }
    f32_t const total_loss = block_reduce_broadcast<kernel_block_size_default_k>(thread_loss);
    if (threadIdx.x != 0u) return;
    position_loss[row] = live ? total_loss * inverse_pixels : 0.0f;
}

expected<masked_patches_t> masked_patches_t::make(gpu_arena_t &arena, cublaslt_session_t &blas,
                                                  masked_patches_config_t config) noexcept {
    using result_t = expected<masked_patches_t>;
    if (config.hidden == 0 || config.patch_pixels == 0 || config.max_patches == 0 || config.temporal_groups == 0 ||
        config.max_patches % config.temporal_groups != 0)
        return result_t::fail(status_t::shape_unsupported_k,
                              "the patch head needs a full geometry (hidden={} pixels={} patches={} groups={})",
                              config.hidden, config.patch_pixels, config.max_patches, config.temporal_groups);
    masked_patches_t head;
    head.config_ = config;
    head.session_ = &blas;

    gpu_arena_alloc<bf16_t> bf16_alloc {arena};
    gpu_arena_alloc<f32_t> f32_alloc {arena};
    gpu_arena_alloc<u32_t> u32_alloc {arena};
    gpu_arena_alloc<b8_t> byte_alloc {arena};
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

    cublaslt_plan_config_t predicted_config; // predicted[P,X] = normalized[P,H]·Wᵀ, X = patch_pixels
    predicted_config.a_rows = config.max_patches, predicted_config.a_cols = config.hidden;
    predicted_config.a_row_stride = config.hidden;
    predicted_config.b_rows = config.patch_pixels, predicted_config.b_cols = config.hidden;
    predicted_config.b_row_stride = config.hidden;
    predicted_config.c_rows = config.max_patches, predicted_config.c_cols = config.patch_pixels;
    predicted_config.c_row_stride = config.patch_pixels;
    predicted_config.transpose_b = true;
    predicted_config.c_type = CUDA_R_16BF;
    cublaslt_plan_config_t data_grad_config; // d(normalized) = d(predicted) · W
    data_grad_config.a_rows = config.max_patches, data_grad_config.a_cols = config.patch_pixels;
    data_grad_config.a_row_stride = config.patch_pixels;
    data_grad_config.b_rows = config.patch_pixels, data_grad_config.b_cols = config.hidden;
    data_grad_config.b_row_stride = config.hidden;
    data_grad_config.c_rows = config.max_patches, data_grad_config.c_cols = config.hidden;
    data_grad_config.c_row_stride = config.hidden;
    data_grad_config.c_type = CUDA_R_16BF;
    cublaslt_plan_config_t weight_grad_config; // d(W) = d(predicted)ᵀ · normalized
    weight_grad_config.a_rows = config.max_patches, weight_grad_config.a_cols = config.patch_pixels;
    weight_grad_config.a_row_stride = config.patch_pixels;
    weight_grad_config.b_rows = config.max_patches, weight_grad_config.b_cols = config.hidden;
    weight_grad_config.b_row_stride = config.hidden;
    weight_grad_config.c_rows = config.patch_pixels, weight_grad_config.c_cols = config.hidden;
    weight_grad_config.c_row_stride = config.hidden;
    weight_grad_config.transpose_a = true;
    weight_grad_config.c_type = CUDA_R_16BF;
    cublaslt_plan_config_t const configs[linear_matmul_family_count_k] = {predicted_config, data_grad_config,
                                                                          weight_grad_config};
    std::size_t workspace_bytes = 0;
    for (u32_t family = 0; family != linear_matmul_family_count_k; ++family) {
        expected<cublaslt_plan_t> plan = cublaslt_plan_t::make(blas, configs[family]);
        if (!plan) return result_t::fail(std::move(plan));
        if (plan.value_.workspace_bytes() > workspace_bytes) workspace_bytes = plan.value_.workspace_bytes();
        head.plans_[family] = std::move(plan.value_);
    }
    if (workspace_bytes != 0) {
        head.workspace_ = b8r1_tensor_arena_t::uninitialized({workspace_bytes}, byte_alloc).value;
        if (!head.workspace_)
            return result_t::fail(status_t::bad_alloc_k, "tensor-core workspace allocation failed ({} bytes)",
                                  workspace_bytes);
    }
    return result_t {std::move(head)};
}

expected_t masked_patches_t::sample_mask(masking_recipe_t recipe, data_epoch_t data_epoch,
                                         gpu_stream_t &stream) noexcept {
    u32_t const patches = config_.max_patches;
    dim3 const block_shape {kernel_block_size_default_k, 1u, 1u};
    dim3 const patch_grid {grid_size_for(patches), 1u, 1u};
    u32_t const spatial_patches = patches / config_.temporal_groups;
    return launch_kernel(reinterpret_cast<void const *>(&patch_mask_sample_kernel_), patch_grid, block_shape, stream,
                         mask_.span(), recipe.seed, data_epoch, config_.mask_ratio, spatial_patches);
}

expected_t masked_patches_t::corrupt_raw(nk::tensor_span<pixel_t, 2> raw_patches, gpu_stream_t &stream) noexcept {
    u32_t const patches = config_.max_patches;
    if (raw_patches.extent(0) != patches)
        return expected_t::fail(status_t::shape_unsupported_k, "corrupt_raw expects [{},patch_scalars] patch rows",
                                patches);
    dim3 const block_shape {kernel_block_size_default_k, 1u, 1u};
    dim3 const patch_grid {grid_size_for(patches), 1u, 1u};
    return launch_kernel(reinterpret_cast<void const *>(&patch_corrupt_raw_kernel_), patch_grid, block_shape, stream,
                         raw_patches, mask_.view());
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
    expected_t staged = launch_kernel(reinterpret_cast<void const *>(&patch_count_kernel_), one_block, block_shape,
                                      stream, mask_.view(), gradient_scale_.span(), config_.weight, patch_pixels);
    if (staged)
        staged = plan_for(linear_matmul_family_t::forward_k)
                     .matmul(*session_, normalized, recon_weight_.view(), predicted_.span(), 0.0f, workspace_.span(),
                             stream);
    if (staged)
        staged = launch_kernel(reinterpret_cast<void const *>(&patch_mae_loss_kernel_), row_grid, block_shape, stream,
                               predicted_view, clean_patches, mask_.view(), gradient_scale_.view(),
                               position_loss_.span(), predicted_grad_span);
    if (staged)
        staged = launch_kernel(reinterpret_cast<void const *>(&masked_finalize_kernel_), one_block, block_shape, stream,
                               position_loss_.view(), gradient_scale_.view(), loss_summands, config_.weight);
    if (staged)
        staged = plan_for(linear_matmul_family_t::data_grad_k)
                     .matmul(*session_, predicted_grad_.view(), recon_weight_.view(), normalized_grad, 1.0f,
                             workspace_.span(), stream);
    if (staged)
        staged = plan_for(linear_matmul_family_t::weight_grad_k)
                     .matmul(*session_, predicted_grad_.view(), normalized, recon_weight_grad_.span(), 0.0f,
                             workspace_.span(), stream);
    return staged;
}

expected_t masked_patches_t::fill(sz::string_view name, array<differentiable_weights_t> &weights) noexcept {
    differentiable_weights_t weight;
    if (!weight.name.try_assign(name)) return expected_t::fail(status_t::bad_alloc_k, "weight name assignment failed");
    weight.weights = recon_weight_.span();
    weight.weights_grad = recon_weight_grad_.view();
    if (weights.try_push_back(std::move(weight)) != status_t::success_k)
        return expected_t::fail(status_t::bad_alloc_k, "weight table growth failed");
    return {};
}

/** @brief Sample the frame mask: a frame is masked when its independent per-frame draw falls under
 *      @p mask_ratio . Keyed on the frame index alone with the @c frame_mask_k salt — no tube concept, so
 *      whole waveform frames are masked i.i.d.
 *
 *  __grid_size__ @c [grid_size_for(frames),1,1] , one thread per frame; no dynamic shared memory. */
static __global__ void frame_mask_sample_kernel_(u32r1_span_t mask, u64_t seed, data_epoch_t data_epoch,
                                                 f32_t mask_ratio) {
    u32_t const frame = blockIdx.x * blockDim.x + threadIdx.x;
    if (frame >= mask.extent(0)) return;
    f32_t const draw = masking_draw(seed, data_epoch, frame, 0u,
                                    static_cast<u32_t>(masking_draw_purpose_t::frame_mask_k));
    mask[frame] = draw < mask_ratio ? 1u : 0u;
}

/** @brief Zero the raw waveform frames the mask selects, in place; visible frames keep their Float32
 *      samples. The driver calls this on the live batch audio so the tower re-projects the corrupted samples.
 *
 *  __grid_size__ @c [grid_size_for(frames),1,1] , persistent stride; no dynamic shared memory. */
static __global__ void frame_corrupt_raw_kernel_(f32r2_span_t raw, u32r1_view_t mask) {
    std::size_t const frames = mask.extent(0);
    u32_t const chunk_samples = raw.extent(1);
    for (std::size_t frame = blockIdx.x; frame < frames; frame += gridDim.x) {
        if (mask[frame] == 0u) continue;
        for (u32_t index = threadIdx.x; index < chunk_samples; index += blockDim.x) raw[frame, index] = 0.0f;
    }
}

/** @brief Count the masked frames and publish @c {λ·2/(chunk_samples·max(count,1)), count} — the MAE
 *      gradient's per-element constant, so the count never visits the host.
 *
 *  __grid_size__ @c [1,1,1] , one block; no dynamic shared memory. */
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
 *  @brief Per-frame normalized-MSE over one masked audio frame: z-score the prediction and the raw Float32
 *      target by their own mean / standard deviation, accumulate the squared difference, and write the
 *      pre-scaled gradient — visible frames write zeros and score nothing. The prediction rides BFloat16
 *      off the decode matmul; the target is the raw waveform frame, read as Float32.
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

    f32_t sum_pred = 0.0f, sumsq_pred = 0.0f, sum_target = 0.0f, sumsq_target = 0.0f;
    for (u32_t index = threadIdx.x; index < chunk_samples; index += blockDim.x) {
        f32_t const p = __bfloat162float(predicted[row, index]), t = targets[row, index];
        sum_pred += p, sumsq_pred += p * p, sum_target += t, sumsq_target += t * t;
    }
    f32_t const mean_pred = block_reduce_broadcast<kernel_block_size_default_k>(sum_pred) * inverse_samples;
    f32_t const mean_sq_pred = block_reduce_broadcast<kernel_block_size_default_k>(sumsq_pred) * inverse_samples;
    f32_t const mean_target = block_reduce_broadcast<kernel_block_size_default_k>(sum_target) * inverse_samples;
    f32_t const mean_sq_target = block_reduce_broadcast<kernel_block_size_default_k>(sumsq_target) * inverse_samples;
    f32_t const inverse_std_pred = rsqrtf(fmaxf(mean_sq_pred - mean_pred * mean_pred, 0.0f) + patch_variance_epsilon_k);
    f32_t const inverse_std_target = rsqrtf(fmaxf(mean_sq_target - mean_target * mean_target, 0.0f) +
                                            patch_variance_epsilon_k);

    f32_t const scale = live ? scale_and_count[0] * inverse_std_pred : 0.0f;
    f32_t thread_loss = 0.0f;
    for (u32_t index = threadIdx.x; index < chunk_samples; index += blockDim.x) {
        f32_t const pred_norm = (__bfloat162float(predicted[row, index]) - mean_pred) * inverse_std_pred;
        f32_t const target_norm = (targets[row, index] - mean_target) * inverse_std_target;
        f32_t const difference = pred_norm - target_norm;
        thread_loss += difference * difference;
        predicted_grad[row, index] = __float2bfloat16(scale * difference);
    }
    f32_t const total_loss = block_reduce_broadcast<kernel_block_size_default_k>(thread_loss);
    if (threadIdx.x != 0u) return;
    position_loss[row] = live ? total_loss * inverse_samples : 0.0f;
}

expected<masked_frames_t> masked_frames_t::make(gpu_arena_t &arena, cublaslt_session_t &blas,
                                                masked_frames_config_t config) noexcept {
    using result_t = expected<masked_frames_t>;
    if (config.hidden == 0 || config.chunk_samples == 0 || config.max_frames == 0)
        return result_t::fail(status_t::shape_unsupported_k,
                              "the frame head needs a full geometry (hidden={} samples={} frames={})", config.hidden,
                              config.chunk_samples, config.max_frames);
    masked_frames_t head;
    head.config_ = config;
    head.session_ = &blas;

    gpu_arena_alloc<bf16_t> bf16_alloc {arena};
    gpu_arena_alloc<f32_t> f32_alloc {arena};
    gpu_arena_alloc<u32_t> u32_alloc {arena};
    gpu_arena_alloc<b8_t> byte_alloc {arena};
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

    cublaslt_plan_config_t predicted_config; // predicted[F,S] = normalized[F,H]·Wᵀ, S = chunk_samples
    predicted_config.a_rows = config.max_frames, predicted_config.a_cols = config.hidden;
    predicted_config.a_row_stride = config.hidden;
    predicted_config.b_rows = config.chunk_samples, predicted_config.b_cols = config.hidden;
    predicted_config.b_row_stride = config.hidden;
    predicted_config.c_rows = config.max_frames, predicted_config.c_cols = config.chunk_samples;
    predicted_config.c_row_stride = config.chunk_samples;
    predicted_config.transpose_b = true;
    predicted_config.c_type = CUDA_R_16BF;
    cublaslt_plan_config_t data_grad_config; // d(normalized) = d(predicted) · W
    data_grad_config.a_rows = config.max_frames, data_grad_config.a_cols = config.chunk_samples;
    data_grad_config.a_row_stride = config.chunk_samples;
    data_grad_config.b_rows = config.chunk_samples, data_grad_config.b_cols = config.hidden;
    data_grad_config.b_row_stride = config.hidden;
    data_grad_config.c_rows = config.max_frames, data_grad_config.c_cols = config.hidden;
    data_grad_config.c_row_stride = config.hidden;
    data_grad_config.c_type = CUDA_R_16BF;
    cublaslt_plan_config_t weight_grad_config; // d(W) = d(predicted)ᵀ · normalized
    weight_grad_config.a_rows = config.max_frames, weight_grad_config.a_cols = config.chunk_samples;
    weight_grad_config.a_row_stride = config.chunk_samples;
    weight_grad_config.b_rows = config.max_frames, weight_grad_config.b_cols = config.hidden;
    weight_grad_config.b_row_stride = config.hidden;
    weight_grad_config.c_rows = config.chunk_samples, weight_grad_config.c_cols = config.hidden;
    weight_grad_config.c_row_stride = config.hidden;
    weight_grad_config.transpose_a = true;
    weight_grad_config.c_type = CUDA_R_16BF;
    cublaslt_plan_config_t const configs[linear_matmul_family_count_k] = {predicted_config, data_grad_config,
                                                                          weight_grad_config};
    std::size_t workspace_bytes = 0;
    for (u32_t family = 0; family != linear_matmul_family_count_k; ++family) {
        expected<cublaslt_plan_t> plan = cublaslt_plan_t::make(blas, configs[family]);
        if (!plan) return result_t::fail(std::move(plan));
        if (plan.value_.workspace_bytes() > workspace_bytes) workspace_bytes = plan.value_.workspace_bytes();
        head.plans_[family] = std::move(plan.value_);
    }
    if (workspace_bytes != 0) {
        head.workspace_ = b8r1_tensor_arena_t::uninitialized({workspace_bytes}, byte_alloc).value;
        if (!head.workspace_)
            return result_t::fail(status_t::bad_alloc_k, "tensor-core workspace allocation failed ({} bytes)",
                                  workspace_bytes);
    }
    return result_t {std::move(head)};
}

expected_t masked_frames_t::sample_mask(masking_recipe_t recipe, data_epoch_t data_epoch,
                                        gpu_stream_t &stream) noexcept {
    u32_t const frames = config_.max_frames;
    dim3 const block_shape {kernel_block_size_default_k, 1u, 1u};
    dim3 const frame_grid {grid_size_for(frames), 1u, 1u};
    return launch_kernel(reinterpret_cast<void const *>(&frame_mask_sample_kernel_), frame_grid, block_shape, stream,
                         mask_.span(), recipe.seed, data_epoch, config_.mask_ratio);
}

expected_t masked_frames_t::corrupt_raw(f32r2_span_t raw_frames, gpu_stream_t &stream) noexcept {
    u32_t const frames = config_.max_frames;
    if (raw_frames.extent(0) != frames)
        return expected_t::fail(status_t::shape_unsupported_k, "corrupt_raw expects [{},chunk_samples] frame rows",
                                frames);
    dim3 const block_shape {kernel_block_size_default_k, 1u, 1u};
    dim3 const frame_grid {grid_size_for(frames), 1u, 1u};
    return launch_kernel(reinterpret_cast<void const *>(&frame_corrupt_raw_kernel_), frame_grid, block_shape, stream,
                         raw_frames, mask_.view());
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
    expected_t staged = launch_kernel(reinterpret_cast<void const *>(&frame_count_kernel_), one_block, block_shape,
                                      stream, mask_.view(), gradient_scale_.span(), config_.weight, chunk_samples);
    if (staged)
        staged = plan_for(linear_matmul_family_t::forward_k)
                     .matmul(*session_, normalized, recon_weight_.view(), predicted_.span(), 0.0f, workspace_.span(),
                             stream);
    if (staged)
        staged = launch_kernel(reinterpret_cast<void const *>(&frame_mae_loss_kernel_), row_grid, block_shape, stream,
                               predicted_view, clean_frames, mask_.view(), gradient_scale_.view(),
                               position_loss_.span(), predicted_grad_span);
    if (staged)
        staged = launch_kernel(reinterpret_cast<void const *>(&masked_finalize_kernel_), one_block, block_shape, stream,
                               position_loss_.view(), gradient_scale_.view(), loss_summands, config_.weight);
    if (staged)
        staged = plan_for(linear_matmul_family_t::data_grad_k)
                     .matmul(*session_, predicted_grad_.view(), recon_weight_.view(), normalized_grad, 1.0f,
                             workspace_.span(), stream);
    if (staged)
        staged = plan_for(linear_matmul_family_t::weight_grad_k)
                     .matmul(*session_, predicted_grad_.view(), normalized, recon_weight_grad_.span(), 0.0f,
                             workspace_.span(), stream);
    return staged;
}

expected_t masked_frames_t::fill(sz::string_view name, array<differentiable_weights_t> &weights) noexcept {
    differentiable_weights_t weight;
    if (!weight.name.try_assign(name)) return expected_t::fail(status_t::bad_alloc_k, "weight name assignment failed");
    weight.weights = recon_weight_.span();
    weight.weights_grad = recon_weight_grad_.view();
    if (weights.try_push_back(std::move(weight)) != status_t::success_k)
        return expected_t::fail(status_t::bad_alloc_k, "weight table growth failed");
    return {};
}
