/**
 *  @brief Device kernels and heavy method bodies of the encoder tower declared in
 *      @ref encoder_tower.cuh — the translation unit that joins the `uform_attention` target beside
 *      @ref transformer_block.cu .
 *  @file cuda_core/encoder_tower.cu
 *  @author Ash Vardanian
 *  @date July 4, 2026
 */
#include <algorithm> // `std::max` — the running plan-workspace maximum

#include <cub/device/device_radix_sort.cuh> // the deterministic embedding-gradient scatter

#include "cuda_core/encoder_tower.hpp"

using namespace unum::uform::core;

/**
 *  @brief Token-embedding lookup over one segment: for each @b text segment, @c x[offset+k,:] =
 *      table[tokens[offset+k],:] . Segments of other modalities exit at once, so only text cells are
 *      written and the packed state's media / PAD cells keep whatever the caller pre-zeroed.
 *
 *  __grid_size__ @c [segments,1,1] , threads stripe the hidden channels for each of the segment's cells;
 *      no dynamic shared memory.
 */
static __global__ void tower_text_entry_kernel_(tokens_view_t tokens, bf16r2_span_t table, u32r1_view_t offsets,
                                                u32r1_view_t lengths, modality_view_t modality,
                                                bf16r2_span_t destination) {
    u32_t const segment = blockIdx.x;
    if (modality[segment] != modality_t::text_k) return;
    u32_t const length = lengths[segment];
    if (length == 0u) return;
    u32_t const offset = offsets[segment];
    u32_t const hidden = static_cast<u32_t>(destination.extent(1));
    for (u32_t k = 0; k != length; ++k) {
        std::size_t const row = static_cast<std::size_t>(offset) + k;
        std::size_t const id = static_cast<std::size_t>(tokens[row]);
        for (u32_t channel = threadIdx.x; channel < hidden; channel += blockDim.x)
            destination[row, channel] = table[id, channel];
    }
}

/**
 *  @brief Stage the embedding-scatter sort keys from one @b text segment: @c keys[offset+k] =
 *      tokens[offset+k] . Non-text segments exit, leaving their cells at the sentinel the caller pre-filled
 *      so they sort into one skipped run — the packed grid interleaves modalities, so the text token ids are
 *      not a leading slice any more.
 *
 *  __grid_size__ @c [segments,1,1] , threads stripe the segment's length; no dynamic shared memory.
 */
static __global__ void tower_text_keys_kernel_(tokens_view_t tokens, u32r1_view_t offsets, u32r1_view_t lengths,
                                               modality_view_t modality, tokens_span_t keys) {
    u32_t const segment = blockIdx.x;
    if (modality[segment] != modality_t::text_k) return;
    u32_t const length = lengths[segment];
    if (length == 0u) return;
    u32_t const offset = offsets[segment];
    for (u32_t k = threadIdx.x; k < length; k += blockDim.x) keys[offset + k] = tokens[offset + k];
}

/**
 *  @brief Deterministic embedding-gradient scatter over the pre-sorted (token, position) pairs: one CTA
 *      per position range start walks its whole duplicate run serially — fixed order, no atomics.
 *
 *  __grid_size__ @c [tokens,1,1] ; a CTA whose row is not a run head exits immediately.
 */
static __global__ void tower_embedding_scatter_kernel_(tokens_view_t sorted_tokens, u32r1_view_t sorted_positions,
                                                       bf16r2_view_t token_grad, bf16r2_span_t table_grad) {
    std::size_t const row = blockIdx.x;
    std::size_t const tokens = sorted_tokens.extent(0);
    token_id_t const token = sorted_tokens[row];
    // Non-text cells carry the sentinel id (>= vocabulary); they sort into one trailing run with no table
    // row to fold into, so the whole run is skipped and no media / PAD gradient leaks into the table.
    if (static_cast<std::size_t>(token) >= table_grad.extent(0)) return;
    if (row != 0 && sorted_tokens[row - 1] == token) return; // Not this run's head.
    u32_t const hidden = static_cast<u32_t>(table_grad.extent(1));
    for (u32_t channel = threadIdx.x; channel < hidden; channel += blockDim.x) {
        f32_t total = 0.0f;
        for (std::size_t run = row; run < tokens && sorted_tokens[run] == token; ++run)
            total += __bfloat162float(token_grad[static_cast<std::size_t>(sorted_positions[run]), channel]);
        table_grad[static_cast<std::size_t>(token), channel] = __float2bfloat16_rn(total);
    }
}

/** @brief Fill the position payload 0..tokens-1 for the embedding sort.
 *  __grid_size__ grid-stride; no dynamic shared memory. */
static __global__ void tower_iota_kernel_(u32r1_span_t positions) {
    std::size_t const total = positions.extent(0);
    std::size_t const stride = static_cast<std::size_t>(gridDim.x) * blockDim.x;
    for (std::size_t index = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x; index < total;
         index += stride)
        positions[index] = static_cast<u32_t>(index);
}

/**
 *  @brief Elementwise SiLU forward: @c y = x·σ(x) ; the backward recomputes σ from the stashed input.
 *
 *  __grid_size__ grid-stride over the flattened extent; no dynamic shared memory.
 */
static __global__ void tower_silu_kernel_(bf16r1_span_t input, bf16r1_span_t output) {
    std::size_t const total = input.extent(0);
    std::size_t const stride = static_cast<std::size_t>(gridDim.x) * blockDim.x;
    for (std::size_t index = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x; index < total;
         index += stride) {
        f32_t const value = __bfloat162float(input[index]);
        output[index] = __float2bfloat16_rn(value / (1.0f + expf(-value)));
    }
}

/**
 *  @brief Elementwise SiLU backward: @c dx = dy·σ(x)·(1+x·(1−σ(x))) from the stashed input.
 *
 *  __grid_size__ grid-stride; no dynamic shared memory.
 */
static __global__ void tower_silu_backward_kernel_(bf16r1_span_t input, bf16r1_span_t output_grad,
                                                   bf16r1_span_t input_grad) {
    std::size_t const total = input.extent(0);
    std::size_t const stride = static_cast<std::size_t>(gridDim.x) * blockDim.x;
    for (std::size_t index = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x; index < total;
         index += stride) {
        f32_t const value = __bfloat162float(input[index]);
        f32_t const sigmoid = 1.0f / (1.0f + expf(-value));
        input_grad[index] = __float2bfloat16_rn(__bfloat162float(output_grad[index]) * sigmoid *
                                                (1.0f + value * (1.0f - sigmoid)));
    }
}

/**
 *  @brief The attention pool's backward — the house replacement for cuDNN's @c sdpa_backward , which
 *      emits non-finite gradients at single-query shapes. CTA @c (s,h) owns segment @c s 's head @c h ,
 *      thread @c c its channel: four streaming passes recompute the softmax the forward ran (max, then
 *      the denominator, then the @c Σp·dp projection, then the fold), so no attention weights are ever
 *      stashed. K=V, so one write per token row carries @c dV+dK ; the per-segment query-gradient
 *      partials land in @p query_partials for @ref tower_pool_query_grad_kernel_ 's fixed-order fold. A
 *      zero-length (PAD) segment writes only its zero partial.
 *
 *  __grid_size__ @c [segments,heads,1] , one @ref transformer_head_dim_k -wide block; no dynamic shared
 *      memory.
 */
static __global__ void tower_pool_backward_kernel_(bf16r2_view_t states, bf16r1_view_t query, bf16r2_view_t pooled_grad,
                                                   u32r1_view_t segment_offsets, u32r1_view_t segment_lengths,
                                                   bf16r2_span_t states_grad, bf16r2_span_t query_partials,
                                                   f32_t scale) {
    u32_t const segment = blockIdx.x, head = blockIdx.y, channel = threadIdx.x;
    u32_t const base_channel = head * transformer_head_dim_k + channel;
    u32_t const length = segment_lengths[segment];
    if (length == 0u) {
        // Sound because the loss zeroes PAD-vertex forces by relation precedence (PAD outranks every
        // edge in contrastive_pair_relation), so nothing real ever flows back through a PAD vertex.
        query_partials[segment, base_channel] = __float2bfloat16(0.0f);
        return;
    }
    u32_t const offset = segment_offsets[segment];
    f32_t const query_value = __bfloat162float(query[base_channel]);
    f32_t const output_grad = __bfloat162float(pooled_grad[segment, base_channel]);

    f32_t score_max = -INFINITY;
    for (u32_t token = 0; token != length; ++token) {
        f32_t const score = block_reduce_broadcast<transformer_head_dim_k>(
                                query_value * __bfloat162float(states[offset + token, base_channel])) *
                            scale;
        score_max = fmaxf(score_max, score);
    }
    f32_t denominator = 0.0f;
    for (u32_t token = 0; token != length; ++token) {
        f32_t const score = block_reduce_broadcast<transformer_head_dim_k>(
                                query_value * __bfloat162float(states[offset + token, base_channel])) *
                            scale;
        denominator += expf(score - score_max);
    }
    f32_t projection = 0.0f; // Σ_t p_t·dp_t — the softmax Jacobian's rank-1 term.
    for (u32_t token = 0; token != length; ++token) {
        f32_t const value = __bfloat162float(states[offset + token, base_channel]);
        f32_t const score = block_reduce_broadcast<transformer_head_dim_k>(query_value * value) * scale;
        f32_t const probability = expf(score - score_max) / denominator;
        f32_t const value_projection = block_reduce_broadcast<transformer_head_dim_k>(output_grad * value);
        projection += probability * value_projection;
    }
    f32_t query_grad = 0.0f;
    for (u32_t token = 0; token != length; ++token) {
        f32_t const value = __bfloat162float(states[offset + token, base_channel]);
        f32_t const score = block_reduce_broadcast<transformer_head_dim_k>(query_value * value) * scale;
        f32_t const probability = expf(score - score_max) / denominator;
        f32_t const value_projection = block_reduce_broadcast<transformer_head_dim_k>(output_grad * value);
        f32_t const score_grad = probability * (value_projection - projection);
        states_grad[offset + token, base_channel] = __float2bfloat16_rn(probability * output_grad +
                                                                        score_grad * scale * query_value);
        query_grad += score_grad * scale * value;
    }
    query_partials[segment, base_channel] = __float2bfloat16_rn(query_grad);
}

/**
 *  @brief Fold the pool's per-segment query-gradient partials into the one @b [hidden] gradient — a
 *      fixed-order serial sum per channel, deterministic by construction.
 *
 *  __grid_size__ grid-stride over @c hidden channels; no dynamic shared memory.
 */
static __global__ void tower_pool_query_grad_kernel_(bf16r2_view_t query_partials, bf16r1_span_t query_grad) {
    std::size_t const segments = query_partials.extent(0), hidden = query_partials.extent(1);
    std::size_t const stride = static_cast<std::size_t>(gridDim.x) * blockDim.x;
    for (std::size_t channel = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x; channel < hidden;
         channel += stride) {
        f32_t total = 0.0f;
        for (std::size_t segment = 0; segment != segments; ++segment)
            total += __bfloat162float(query_partials[segment, channel]);
        query_grad[channel] = __float2bfloat16_rn(total);
    }
}

/**
 *  @brief Elementwise add of the masked force onto the pooled force at the final-norm states — the
 *      co-trained join @c dL_pooled + dL_masked .
 *
 *  __grid_size__ grid-stride; no dynamic shared memory.
 */
static __global__ void tower_grad_add_kernel_(bf16r1_span_t accumulator, bf16r1_view_t addend) {
    std::size_t const total = accumulator.extent(0);
    std::size_t const stride = static_cast<std::size_t>(gridDim.x) * blockDim.x;
    for (std::size_t index = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x; index < total;
         index += stride)
        accumulator[index] = __float2bfloat16_rn(__bfloat162float(accumulator[index]) +
                                                 __bfloat162float(addend[index]));
}

/**
 *  @brief Normalize one @b pixel segment's cells in place into the projection's BFloat16 fan-in. CTA @c s
 *      owns segment @c s , exits unless @p target matches its @ref modality_t , and otherwise walks its
 *      @c length cells at flat grid rows @c [offset,offset+length) — a cell's raw pixels live at the same
 *      flat position as its state, so the peel is a zero-copy slice — writing each channel as
 *      @c (byte/255 − mean[channel])·inverse_std[channel] over the rank-3 peel @c [rows,spatial,channels] ,
 *      @c channels innermost so @p mean / @p inverse_std index by the channel coordinate directly. The scratch
 *      is pre-zeroed, so every cell this pass skips stays zero and the projection's non-modality rows vanish.
 *
 *  __grid_size__ @c [segments,1,1] , threads stripe the segment's @c length·spatial cells; no dynamic shared memory.
 */
static __global__ void tower_normalize_segment_kernel_(nk::tensor_view<pixel_t, 3> content, u32r1_view_t offsets,
                                                       u32r1_view_t lengths, modality_view_t modality,
                                                       modality_t target, f32r1_view_t mean, f32r1_view_t inverse_std,
                                                       nk::tensor_span<bf16_t, 3> patches) {
    u32_t const segment = blockIdx.x;
    if (modality[segment] != target) return;
    u32_t const length = lengths[segment];
    if (length == 0u) return;
    u32_t const offset = offsets[segment];
    u32_t const spatial = static_cast<u32_t>(content.extent(1));
    u32_t const channels = static_cast<u32_t>(content.extent(2));
    std::size_t const cells = static_cast<std::size_t>(length) * spatial;
    f32_t const inverse_byte_max = 1.0f / 255.0f;
    for (std::size_t cell = threadIdx.x; cell < cells; cell += blockDim.x) {
        u32_t const spatial_position = static_cast<u32_t>(cell % spatial);
        std::size_t const row = static_cast<std::size_t>(offset) + cell / spatial;
        for (u32_t channel = 0; channel != channels; ++channel) {
            f32_t const value = (static_cast<f32_t>(content[row, spatial_position, channel]) * inverse_byte_max -
                                 mean[channel]) *
                                inverse_std[channel];
            patches[row, spatial_position, channel] = __float2bfloat16(value);
        }
    }
}

/**
 *  @brief Stage one @b audio segment's raw waveform frames in place into the projection's BFloat16 fan-in. Like
 *      @ref tower_normalize_segment_kernel_ but raw samples carry no per-channel statistics to subtract, so the
 *      cast is the whole staging ( @c f32 → BFloat16 ) over the flat @c [rows,scalars] peel (a frame is
 *      @c chunk_samples raw samples); non-audio segments and the pre-zeroed remainder stay zero.
 *
 *  __grid_size__ @c [segments,1,1] , threads stripe the segment's @c length·scalars cells; no dynamic shared memory.
 */
static __global__ void tower_audio_segment_kernel_(nk::tensor_view<f32_t, 2> content, u32r1_view_t offsets,
                                                   u32r1_view_t lengths, modality_view_t modality,
                                                   nk::tensor_span<bf16_t, 2> patches) {
    u32_t const segment = blockIdx.x;
    if (modality[segment] != modality_t::audio_k) return;
    u32_t const length = lengths[segment];
    if (length == 0u) return;
    u32_t const offset = offsets[segment];
    u32_t const scalars = static_cast<u32_t>(content.extent(1));
    std::size_t const cells = static_cast<std::size_t>(length) * scalars;
    for (std::size_t cell = threadIdx.x; cell < cells; cell += blockDim.x) {
        u32_t const scalar = static_cast<u32_t>(cell % scalars);
        std::size_t const row = static_cast<std::size_t>(offset) + cell / scalars;
        patches[row, scalar] = __float2bfloat16(content[row, scalar]);
    }
}

/** @brief The attention pool's softmax scale @c 1/√head_dim — one derivation for the device upload and
 *      the backward kernel argument. */
[[nodiscard]] static f32_t tower_pool_scale_() noexcept {
    return 1.0f / sqrtf(static_cast<f32_t>(transformer_head_dim_k));
}

/** @brief File-scope mirror of the towers' @c plan_absent_k sentinel, so the shared media helpers below can
 *      skip an absent modality without reaching into either tower's private constants. */
static constexpr std::size_t plan_absent_k = ~std::size_t {0};

/**
 *  @brief Upload one modality's per-channel pixel mean and inverse-std to the device, allocating the two
 *      @b [channels] arenas off @p f32_alloc — the shared upload @ref encoder_tower_bf16_t::make runs for
 *      image and video (audio frames are raw waveform with no per-channel statistics, so they never call this).
 */
static expected_t upload_pixel_constants(f32r1_tensor_arena_t &mean_arena, f32r1_tensor_arena_t &inverse_std_arena,
                                         f32_t const *mean_source, f32_t const *std_source, u32_t channels,
                                         gpu_arena_alloc<f32_t> &f32_alloc, gpu_arena_t &arena) noexcept {
    mean_arena = f32r1_tensor_arena_t::uninitialized({channels}, f32_alloc).value;
    inverse_std_arena = f32r1_tensor_arena_t::uninitialized({channels}, f32_alloc).value;
    if (!mean_arena || !inverse_std_arena)
        return expected_t::fail(status_t::bad_alloc_k, "pixel constant allocation failed (arena_used={})",
                                arena.used());
    f32_t mean[4] {}, inverse_std[4] {};
    for (u32_t channel = 0; channel != channels; ++channel) {
        mean[channel] = mean_source[channel];
        inverse_std[channel] = 1.0f / std_source[channel];
    }
    if (CUresult result = cuMemcpyHtoD(reinterpret_cast<CUdeviceptr>(mean_arena.data()), mean,
                                       channels * sizeof(f32_t));
        result != CUDA_SUCCESS)
        return expected_t::fail(result, "pixel mean upload failed");
    if (CUresult result = cuMemcpyHtoD(reinterpret_cast<CUdeviceptr>(inverse_std_arena.data()), inverse_std,
                                       channels * sizeof(f32_t));
        result != CUDA_SUCCESS)
        return expected_t::fail(result, "pixel inverse-std upload failed");
    return {};
}

/**
 *  @brief Stage one pixel modality's raw cells into per-channel-normalized BFloat16 patch vectors over the
 *      caller's @b [rows,scalars] @p patches — zero the whole span, then run @ref tower_normalize_segment_kernel_
 *      so only @p target 's cells are written ( @c (byte/255 − mean)·inverse_std , @c channels innermost) and every
 *      non-modality row stays zero. The one normalize launch @ref project_media_entries folds behind its
 *      projection and @ref encoder_tower_bf16_t::patchify_media_targets hands out as the masked-autoencoder
 *      reconstruction target — the two differ only in where @p patches lands and whether a projection follows,
 *      never in the staging itself.
 */
static expected_t stage_pixel_patches(nk::tensor_view<pixel_t, 3> raw, modality_t target, f32r1_view_t mean,
                                      f32r1_view_t inverse_std, u32r1_view_t offsets, u32r1_view_t lengths,
                                      modality_view_t modality, dim3 segment_grid, bf16r2_span_t patches,
                                      gpu_stream_t &stream) noexcept {
    std::size_t const rows = patches.extent(0);
    u32_t const spatial = static_cast<u32_t>(raw.extent(1));
    u32_t const channels = static_cast<u32_t>(raw.extent(2));
    dim3 const block_shape {kernel_block_size_default_k, 1u, 1u};
    if (CUresult result = cuMemsetD8Async(reinterpret_cast<CUdeviceptr>(patches.data()), 0,
                                          rows * static_cast<std::size_t>(spatial) * channels * sizeof(bf16_t),
                                          stream.get());
        result != CUDA_SUCCESS)
        return expected_t::fail(result, "pixel patch reset failed");
    return launch_kernel(reinterpret_cast<void const *>(&tower_normalize_segment_kernel_), segment_grid, block_shape,
                         stream, raw, offsets, lengths, modality, target, mean, inverse_std,
                         patches.reshape<3>({rows, spatial, channels}));
}

/**
 *  @brief Project every present media modality's raw signal into the packed entry state: for each present
 *      modality zero @p patch_scratch , stage only that modality's cells in place (image / video pixels
 *      normalized by their per-channel mean / inverse-std, audio raw frames cast as-is — a cell's content
 *      shares the flat grid position of its state), and accumulate the pinned projection ( @c β=1 ) so the
 *      zero non-modality rows leave every other cell untouched — no scatter, no atomics. Shared verbatim by
 *      both towers' forwards; the text entry (a kernel) is the caller's.
 */
static expected_t project_media_entries(
    multimodal_batch_on_gpu_t const &batch, cublaslt_plan_t *plans, std::size_t image_plan_base,
    std::size_t video_plan_base, std::size_t audio_plan_base, bf16r2_tensor_arena_t &patch_scratch,
    b8r1_tensor_arena_t &workspace, bf16r2_tensor_arena_t &entry_output, f32r1_tensor_arena_t &image_mean,
    f32r1_tensor_arena_t &image_inverse_std, bf16r2_tensor_arena_t &image_projection, f32r1_tensor_arena_t &video_mean,
    f32r1_tensor_arena_t &video_inverse_std, bf16r2_tensor_arena_t &video_projection,
    bf16r2_tensor_arena_t &audio_projection, cublaslt_session_t &blas, gpu_stream_t &stream) noexcept {
    std::size_t const columns = batch.tokens.extent(1);
    std::size_t const live_tokens = static_cast<std::size_t>(batch.tokens.extent(0)) * columns;
    dim3 const block_shape {kernel_block_size_default_k, 1u, 1u};
    dim3 const segment_grid {static_cast<u32_t>(batch.live_segments()), 1u, 1u};
    u32r1_view_t const segment_offsets = batch.segment_offsets.view();
    u32r1_view_t const segment_lengths = batch.segment_lengths.view();
    modality_view_t const segment_modality = batch.segment_modality.view();

    // One pixel modality's forward: zero the scratch, normalize its cells in place, accumulate the projection.
    auto project_pixels = [&](std::size_t plan_base, nk::tensor_view<pixel_t, 3> raw, modality_t target,
                              f32r1_view_t mean, f32r1_view_t inverse_std,
                              bf16r2_view_t projection) noexcept -> expected_t {
        u32_t const scalars = static_cast<u32_t>(raw.extent(1)) * static_cast<u32_t>(raw.extent(2));
        bf16r2_span_t const patches {patch_scratch.data(), live_tokens, scalars};
        expected_t staged = stage_pixel_patches(raw, target, mean, inverse_std, segment_offsets, segment_lengths,
                                                segment_modality, segment_grid, patches, stream);
        if (!staged) return staged;
        return plans[plan_base].matmul(blas, bf16r2_view_t {patches}, projection, entry_output.span(), 1.0f,
                                       workspace.span(), stream);
    };

    expected_t staged {};
    if (image_plan_base != plan_absent_k && batch.images.data() != nullptr) {
        std::size_t const spatial = static_cast<std::size_t>(batch.images.extent(2)) * batch.images.extent(3);
        staged = project_pixels(
            image_plan_base, batch.images.view().reshape<3>({live_tokens, spatial, batch.images.extent(4)}),
            modality_t::image_k, image_mean.view(), image_inverse_std.view(), image_projection.view());
        if (!staged) return staged;
    }
    if (video_plan_base != plan_absent_k && batch.videos.data() != nullptr) {
        std::size_t const spatial = static_cast<std::size_t>(batch.videos.extent(2)) * batch.videos.extent(3) *
                                    batch.videos.extent(4);
        staged = project_pixels(
            video_plan_base, batch.videos.view().reshape<3>({live_tokens, spatial, batch.videos.extent(5)}),
            modality_t::video_k, video_mean.view(), video_inverse_std.view(), video_projection.view());
        if (!staged) return staged;
    }
    if (audio_plan_base != plan_absent_k && batch.audio_samples.data() != nullptr) {
        u32_t const scalars = static_cast<u32_t>(batch.audio_samples.extent(2));
        if (CUresult result = cuMemsetD8Async(reinterpret_cast<CUdeviceptr>(patch_scratch.data()), 0,
                                              live_tokens * scalars * sizeof(bf16_t), stream.get());
            result != CUDA_SUCCESS)
            return expected_t::fail(result, "audio patch scratch reset failed");
        nk::tensor_view<f32_t, 2> const raw = batch.audio_samples.view().reshape<2>({live_tokens, scalars});
        bf16r2_span_t const patches {patch_scratch.data(), live_tokens, scalars};
        staged = launch_kernel(reinterpret_cast<void const *>(&tower_audio_segment_kernel_), segment_grid, block_shape,
                               stream, raw, segment_offsets, segment_lengths, segment_modality, patches);
        if (!staged) return staged;
        staged = plans[audio_plan_base].matmul(blas, bf16r2_view_t {patches}, audio_projection.view(),
                                               entry_output.span(), 1.0f, workspace.span(), stream);
        if (!staged) return staged;
    }
    return staged;
}

/**
 *  @brief Fold the packed entry gradient against each present media modality's re-staged patches into that
 *      modality's projection weight gradient: re-stage exactly as @ref project_media_entries (segment-driven,
 *      so non-modality rows stay zero), then run each modality's weight-grad product ( @c plan_base+1 , @c β=0 )
 *      — the backward twin of @ref project_media_entries , differing only in the matmul direction and offset.
 */
static expected_t project_media_weight_grads(multimodal_batch_on_gpu_t const &batch, cublaslt_plan_t *plans,
                                             std::size_t image_plan_base, std::size_t video_plan_base,
                                             std::size_t audio_plan_base, bf16r2_tensor_arena_t &patch_scratch,
                                             b8r1_tensor_arena_t &workspace, bf16r2_view_t entry_grad,
                                             f32r1_tensor_arena_t &image_mean, f32r1_tensor_arena_t &image_inverse_std,
                                             bf16r2_tensor_arena_t &image_projection_grad,
                                             f32r1_tensor_arena_t &video_mean, f32r1_tensor_arena_t &video_inverse_std,
                                             bf16r2_tensor_arena_t &video_projection_grad,
                                             bf16r2_tensor_arena_t &audio_projection_grad, cublaslt_session_t &blas,
                                             gpu_stream_t &stream) noexcept {
    std::size_t const columns = batch.tokens.extent(1);
    std::size_t const live_tokens = static_cast<std::size_t>(batch.tokens.extent(0)) * columns;
    dim3 const block_shape {kernel_block_size_default_k, 1u, 1u};
    dim3 const segment_grid {static_cast<u32_t>(batch.live_segments()), 1u, 1u};
    u32r1_view_t const segment_offsets = batch.segment_offsets.view();
    u32r1_view_t const segment_lengths = batch.segment_lengths.view();
    modality_view_t const segment_modality = batch.segment_modality.view();

    // One pixel modality's weight grad: re-stage its cells, then fold the whole entry gradient against them.
    auto project_pixels_grad = [&](std::size_t plan_base, nk::tensor_view<pixel_t, 3> raw, modality_t target,
                                   f32r1_view_t mean, f32r1_view_t inverse_std,
                                   bf16r2_span_t projection_grad) noexcept -> expected_t {
        u32_t const spatial = static_cast<u32_t>(raw.extent(1));
        u32_t const channels = static_cast<u32_t>(raw.extent(2));
        u32_t const scalars = spatial * channels;
        if (CUresult result = cuMemsetD8Async(reinterpret_cast<CUdeviceptr>(patch_scratch.data()), 0,
                                              live_tokens * scalars * sizeof(bf16_t), stream.get());
            result != CUDA_SUCCESS)
            return expected_t::fail(result, "pixel patch scratch reset failed");
        bf16r2_span_t const patches {patch_scratch.data(), live_tokens, scalars};
        expected_t normalized = launch_kernel(reinterpret_cast<void const *>(&tower_normalize_segment_kernel_),
                                              segment_grid, block_shape, stream, raw, segment_offsets, segment_lengths,
                                              segment_modality, target, mean, inverse_std,
                                              patches.reshape<3>({live_tokens, spatial, channels}));
        if (!normalized) return normalized;
        return plans[plan_base + 1].matmul(blas, entry_grad, bf16r2_view_t {patches}, projection_grad, 0.0f,
                                           workspace.span(), stream);
    };

    expected_t staged {};
    if (image_plan_base != plan_absent_k && batch.images.data() != nullptr) {
        std::size_t const spatial = static_cast<std::size_t>(batch.images.extent(2)) * batch.images.extent(3);
        staged = project_pixels_grad(
            image_plan_base, batch.images.view().reshape<3>({live_tokens, spatial, batch.images.extent(4)}),
            modality_t::image_k, image_mean.view(), image_inverse_std.view(), image_projection_grad.span());
        if (!staged) return staged;
    }
    if (video_plan_base != plan_absent_k && batch.videos.data() != nullptr) {
        std::size_t const spatial = static_cast<std::size_t>(batch.videos.extent(2)) * batch.videos.extent(3) *
                                    batch.videos.extent(4);
        staged = project_pixels_grad(
            video_plan_base, batch.videos.view().reshape<3>({live_tokens, spatial, batch.videos.extent(5)}),
            modality_t::video_k, video_mean.view(), video_inverse_std.view(), video_projection_grad.span());
        if (!staged) return staged;
    }
    if (audio_plan_base != plan_absent_k && batch.audio_samples.data() != nullptr) {
        u32_t const scalars = static_cast<u32_t>(batch.audio_samples.extent(2));
        if (CUresult result = cuMemsetD8Async(reinterpret_cast<CUdeviceptr>(patch_scratch.data()), 0,
                                              live_tokens * scalars * sizeof(bf16_t), stream.get());
            result != CUDA_SUCCESS)
            return expected_t::fail(result, "audio patch scratch reset failed");
        nk::tensor_view<f32_t, 2> const raw = batch.audio_samples.view().reshape<2>({live_tokens, scalars});
        bf16r2_span_t const patches {patch_scratch.data(), live_tokens, scalars};
        staged = launch_kernel(reinterpret_cast<void const *>(&tower_audio_segment_kernel_), segment_grid, block_shape,
                               stream, raw, segment_offsets, segment_lengths, segment_modality, patches);
        if (!staged) return staged;
        staged = plans[audio_plan_base + 1].matmul(blas, entry_grad, bf16r2_view_t {patches},
                                                   audio_projection_grad.span(), 0.0f, workspace.span(), stream);
        if (!staged) return staged;
    }
    return staged;
}

expected_t encoder_tower_bf16_t::pin_plan(cublaslt_plan_config_t plan_config, std::size_t &role,
                                          std::size_t &workspace_bytes) noexcept {
    plan_config.c_type = CUDA_R_16BF;
    expected<cublaslt_plan_t> plan = cublaslt_plan_t::make(*blas_, plan_config);
    if (!plan) return expected_t::fail(std::move(plan));
    workspace_bytes = std::max(workspace_bytes, plan.value_.workspace_bytes());
    plans_[role++] = std::move(plan.value_);
    return {};
}

expected_t encoder_tower_bf16_t::pin_media_plans(std::size_t &plan_base, u32_t fan_in, u32_t max_tokens, u32_t hidden,
                                                 std::size_t &role, std::size_t &workspace_bytes) noexcept {
    plan_base = role;
    expected_t pinned = pin_plan(
        {max_tokens, fan_in, fan_in, hidden, fan_in, fan_in, max_tokens, hidden, hidden, false, true}, role,
        workspace_bytes);
    if (pinned)
        pinned = pin_plan({max_tokens, hidden, hidden, max_tokens, fan_in, fan_in, hidden, fan_in, fan_in, true, false},
                          role, workspace_bytes);
    return pinned;
}

expected<encoder_tower_bf16_t> encoder_tower_bf16_t::make(gpu_arena_t &arena, cublaslt_session_t &blas,
                                                          cudnn_session_t &cudnn, encoder_tower_config_t config,
                                                          multimodal_batch_shape_t shape) noexcept {
    using result_t = expected<encoder_tower_bf16_t>;
    if (config.depth == 0) return result_t::fail(status_t::invalid_input_k, "a tower needs at least one block");
    if (config.vocabulary == 0)
        return result_t::fail(status_t::invalid_input_k,
                              "the token entry is the only entry wired so far; " "vocabulary must be non-zero");
    if (config.rope_max_positions == 0)
        return result_t::fail(status_t::invalid_input_k, "rope_max_positions must cover the longest sequence");

    encoder_tower_bf16_t tower;
    tower.config_ = config;
    tower.blas_ = &blas;
    if (nk_status_t const status = nk_cuda_capabilities_enabled(static_cast<nk_size_t>(arena.device()),
                                                                &tower.capabilities_);
        status != nk_success_k)
        return result_t::fail(status_t::cuda_error_k, "NumKong cannot query device {}: {}", arena.device(),
                              nk_status_name(status));

    std::size_t const hidden = config.block.hidden;
    std::size_t const tokens = config.block.max_tokens;
    std::size_t const segments = config.block.max_segments;
    std::size_t const vocabulary = config.vocabulary;
    gpu_arena_alloc<bf16_t> bf16_alloc {arena};
    gpu_arena_alloc<f32_t> f32_alloc {arena};
    gpu_arena_alloc<u32_t> u32_alloc {arena};
    gpu_arena_alloc<b8_t> b8_alloc {arena};
    gpu_arena_alloc<i16_t> i16_alloc {arena};

    // The token embedding table + its gradient, and the deterministic-scatter double buffers.
    tower.entry_weight_ = bf16r2_tensor_arena_t::uninitialized({vocabulary, hidden}, bf16_alloc).value;
    tower.entry_weight_grad_ = bf16r2_tensor_arena_t::uninitialized({vocabulary, hidden}, bf16_alloc).value;
    gpu_arena_alloc<token_id_t> token_alloc {arena};
    tower.sort_keys_ = tokens_tensor_arena_t::uninitialized({2 * tokens}, token_alloc).value;
    tower.sort_values_ = u32r1_tensor_arena_t::uninitialized({2 * tokens}, u32_alloc).value;
    if (tower.sort_keys_ && tower.sort_values_) {
        std::size_t sort_bytes = 0;
        cub::DeviceRadixSort::SortPairs(nullptr, sort_bytes, tower.sort_keys_.data(), tower.sort_keys_.data() + tokens,
                                        tower.sort_values_.data(), tower.sort_values_.data() + tokens,
                                        static_cast<int>(tokens));
        tower.sort_workspace_ = b8r1_tensor_arena_t::uninitialized({sort_bytes}, b8_alloc).value;
    }
    if (!tower.entry_weight_ || !tower.entry_weight_grad_ || !tower.sort_keys_ || !tower.sort_values_ ||
        !tower.sort_workspace_)
        return result_t::fail(status_t::bad_alloc_k, "token entry allocation failed (arena_used={})", arena.used());

    // Final norm, attention-pool query, and the head weights, each with their gradient.
    tower.final_norm_scale_ = bf16r1_tensor_arena_t::uninitialized({hidden}, bf16_alloc).value;
    tower.final_norm_scale_grad_ = bf16r1_tensor_arena_t::uninitialized({hidden}, bf16_alloc).value;
    tower.widened_final_norm_scale_ = f32r1_tensor_arena_t::uninitialized({hidden}, f32_alloc).value;
    tower.pool_query_ = bf16r1_tensor_arena_t::uninitialized({hidden}, bf16_alloc).value;
    tower.pool_query_grad_ = bf16r1_tensor_arena_t::uninitialized({hidden}, bf16_alloc).value;
    tower.head_norm_scale_ = bf16r1_tensor_arena_t::uninitialized({hidden}, bf16_alloc).value;
    tower.head_norm_scale_grad_ = bf16r1_tensor_arena_t::uninitialized({hidden}, bf16_alloc).value;
    tower.head_fc1_ = bf16r2_tensor_arena_t::uninitialized({hidden, hidden}, bf16_alloc).value;
    tower.head_fc1_grad_ = bf16r2_tensor_arena_t::uninitialized({hidden, hidden}, bf16_alloc).value;
    tower.head_fc2_ = bf16r2_tensor_arena_t::uninitialized({hidden, hidden}, bf16_alloc).value;
    tower.head_fc2_grad_ = bf16r2_tensor_arena_t::uninitialized({hidden, hidden}, bf16_alloc).value;
    if (!tower.final_norm_scale_ || !tower.final_norm_scale_grad_ || !tower.widened_final_norm_scale_ ||
        !tower.pool_query_ || !tower.pool_query_grad_ || !tower.head_norm_scale_ || !tower.head_norm_scale_grad_ ||
        !tower.head_fc1_ || !tower.head_fc1_grad_ || !tower.head_fc2_ || !tower.head_fc2_grad_)
        return result_t::fail(status_t::bad_alloc_k, "head weights allocation failed (arena_used={})", arena.used());

    // Forward stashes and the two inverse-rms stashes.
    tower.entry_output_ = bf16r2_tensor_arena_t::uninitialized({tokens, hidden}, bf16_alloc).value;
    tower.normalized_ = bf16r2_tensor_arena_t::uninitialized({tokens, hidden}, bf16_alloc).value;
    tower.pooled_ = bf16r2_tensor_arena_t::uninitialized({segments, hidden}, bf16_alloc).value;
    tower.head_normalized_ = bf16r2_tensor_arena_t::uninitialized({segments, hidden}, bf16_alloc).value;
    tower.head_hidden_ = bf16r2_tensor_arena_t::uninitialized({segments, hidden}, bf16_alloc).value;
    tower.head_activated_ = bf16r2_tensor_arena_t::uninitialized({segments, hidden}, bf16_alloc).value;
    tower.final_inverse_rms_ = f32r1_tensor_arena_t::uninitialized({tokens}, f32_alloc).value;
    tower.head_inverse_rms_ = f32r1_tensor_arena_t::uninitialized({segments}, f32_alloc).value;
    tower.coords_ = i16r2_tensor_arena_t::uninitialized({tokens, 3}, i16_alloc).value;
    if (!tower.entry_output_ || !tower.normalized_ || !tower.pooled_ || !tower.head_normalized_ ||
        !tower.head_hidden_ || !tower.head_activated_ || !tower.final_inverse_rms_ || !tower.head_inverse_rms_ ||
        !tower.coords_)
        return result_t::fail(status_t::bad_alloc_k, "stashes allocation failed (arena_used={})", arena.used());

    // The activation and gradient ping-pong buffers.
    tower.activations_a_ = f32r2_tensor_arena_t::uninitialized({tokens, hidden}, f32_alloc).value;
    tower.activations_b_ = f32r2_tensor_arena_t::uninitialized({tokens, hidden}, f32_alloc).value;
    tower.normalized_grad_ = bf16r2_tensor_arena_t::uninitialized({tokens, hidden}, bf16_alloc).value;
    tower.token_grad_a_ = f32r2_tensor_arena_t::uninitialized({tokens, hidden}, f32_alloc).value;
    tower.token_grad_b_ = f32r2_tensor_arena_t::uninitialized({tokens, hidden}, f32_alloc).value;
    tower.segment_grad_a_ = bf16r2_tensor_arena_t::uninitialized({segments, hidden}, bf16_alloc).value;
    tower.segment_grad_b_ = bf16r2_tensor_arena_t::uninitialized({segments, hidden}, bf16_alloc).value;
    if (!tower.activations_a_ || !tower.activations_b_ || !tower.normalized_grad_ || !tower.token_grad_a_ ||
        !tower.token_grad_b_ || !tower.segment_grad_a_ || !tower.segment_grad_b_)
        return result_t::fail(status_t::bad_alloc_k, "ping-pong allocation failed (arena_used={})", arena.used());

    // The RoPE angle tables (model state; the frontend derives the per-token coordinates).

    expected<rope_angles_t> angles = rope_angles_t::make(arena, config.rope_max_positions, config.rope,
                                                         config.rope_base);
    if (!angles) return result_t::fail(std::move(angles));
    tower.angles_ = std::move(angles.value_);

    // The block stack.
    if (tower.blocks_.try_resize(config.depth) != status_t::success_k)
        return result_t::fail(status_t::bad_alloc_k, "block table allocation failed ({})", config.depth);
    expected<transformer_scratch_t> scratch = transformer_scratch_t::make(arena, config.block);
    if (!scratch) return result_t::fail(std::move(scratch));
    tower.scratch_ = std::make_unique<transformer_scratch_t>(std::move(scratch.value_));
    for (u32_t layer_index = 0; layer_index != config.depth; ++layer_index) {
        expected<transformer_block_bf16_t> block =
            transformer_block_bf16_t::make(arena, blas, cudnn, *tower.scratch_, config.block);
        if (!block) return result_t::fail(std::move(block));
        tower.blocks_[layer_index] = std::move(block.value_);
    }

    // The head's pinned products: forward (fc1 and fc2 share the [H,H] shape), data grad, weight grad.

    std::size_t workspace_bytes = 0;
    std::size_t plan_role = 0;
    u32_t const hidden32 = config.block.hidden;
    u32_t const segments32 = config.block.max_segments;
    expected_t pinned = tower.pin_plan(
        {segments32, hidden32, hidden32, hidden32, hidden32, hidden32, segments32, hidden32, hidden32, false, true},
        plan_role, workspace_bytes);
    if (pinned)
        pinned = tower.pin_plan({segments32, hidden32, hidden32, hidden32, hidden32, hidden32, segments32, hidden32,
                                 hidden32, false, false},
                                plan_role, workspace_bytes);
    if (pinned)
        pinned = tower.pin_plan(
            {segments32, hidden32, hidden32, segments32, hidden32, hidden32, hidden32, hidden32, hidden32, true, false},
            plan_role, workspace_bytes);
    if (!pinned) return result_t::fail(std::move(pinned));

    expected<tower_pool_t> pool = tower_pool_t::make(arena, config.block);
    if (!pool) return result_t::fail(std::move(pool));
    tower.pool_ = std::move(pool.value_);

    // Media entries: each present modality patch-cuts its raw signal and projects it through pinned cuBLASLt
    // plans. A modality is present iff its geometry is fully specified — zero extents leave its projection,
    // plan, and scratch untouched. Fan-ins are per-modality: image is ph·pw·image_channels, video folds
    // video_tube_frames ahead of ph·pw·video_channels, and audio — a 1-D signal, unpatched — is just
    // chunk_samples raw samples per frame (its projection is the learnable spectral frontend). Image and video
    // pixels carry an ImageNet-style per-channel normalization; audio raw frames upload no constants.

    bool const image_present = config.image_patch_height != 0 && config.image_patch_width != 0;
    bool const video_present = config.video_tube_frames != 0 && config.video_patch_height != 0 &&
                               config.video_patch_width != 0;
    bool const audio_present = config.audio_chunk_samples != 0;
    if (image_present && config.image_patch_height != config.image_patch_width)
        return result_t::fail(status_t::invalid_input_k, "image patches must be square ({}x{})",
                              config.image_patch_height, config.image_patch_width);
    if (image_present && (config.image_channels == 0 || config.image_channels > 4))
        return result_t::fail(status_t::invalid_input_k, "image_channels must be 1..4 ({})", config.image_channels);
    if (video_present && (config.video_channels == 0 || config.video_channels > 3))
        return result_t::fail(status_t::invalid_input_k, "video_channels must be 1..3 ({})", config.video_channels);

    u32_t const image_patch_scalars = image_present ? patch_scalars(config.image_patch_height, config.image_channels)
                                                    : 0u;
    u32_t const video_tube_scalars = video_present ? config.video_tube_frames * config.video_channels *
                                                         config.video_patch_height * config.video_patch_width
                                                   : 0u;
    u32_t const audio_frame_scalars = audio_present ? config.audio_chunk_samples : 0u;
    u32_t const max_tokens32 = config.block.max_tokens;

    if (image_present) {
        if (expected_t uploaded = upload_pixel_constants(tower.image_pixel_mean_, tower.image_pixel_inverse_std_,
                                                         config.image_pixel_mean, config.image_pixel_std,
                                                         config.image_channels, f32_alloc, arena);
            !uploaded)
            return result_t::fail(std::move(uploaded));
        tower.image_projection_ = bf16r2_tensor_arena_t::uninitialized({hidden, image_patch_scalars}, bf16_alloc).value;
        tower.image_projection_grad_ =
            bf16r2_tensor_arena_t::uninitialized({hidden, image_patch_scalars}, bf16_alloc).value;
        if (!tower.image_projection_ || !tower.image_projection_grad_)
            return result_t::fail(status_t::bad_alloc_k, "image projection allocation failed (arena_used={})",
                                  arena.used());
    }
    if (video_present) {
        if (expected_t uploaded = upload_pixel_constants(tower.video_pixel_mean_, tower.video_pixel_inverse_std_,
                                                         config.video_pixel_mean, config.video_pixel_std,
                                                         config.video_channels, f32_alloc, arena);
            !uploaded)
            return result_t::fail(std::move(uploaded));
        tower.video_projection_ = bf16r2_tensor_arena_t::uninitialized({hidden, video_tube_scalars}, bf16_alloc).value;
        tower.video_projection_grad_ =
            bf16r2_tensor_arena_t::uninitialized({hidden, video_tube_scalars}, bf16_alloc).value;
        if (!tower.video_projection_ || !tower.video_projection_grad_)
            return result_t::fail(status_t::bad_alloc_k, "video projection allocation failed (arena_used={})",
                                  arena.used());
    }
    if (audio_present) {
        tower.audio_projection_ = bf16r2_tensor_arena_t::uninitialized({hidden, audio_frame_scalars}, bf16_alloc).value;
        tower.audio_projection_grad_ =
            bf16r2_tensor_arena_t::uninitialized({hidden, audio_frame_scalars}, bf16_alloc).value;
        if (!tower.audio_projection_ || !tower.audio_projection_grad_)
            return result_t::fail(status_t::bad_alloc_k, "audio projection allocation failed (arena_used={})",
                                  arena.used());
    }

    // Media projection plans, pinned from plan_media_base_k in modality order (image, video, audio): a forward
    // product (patches·Wᵀ → entry) and a weight-grad product (d_entryᵀ·patches → dW) each — no data-grad plan
    // exists, since raw pixels are never differentiated. Batches are heterogeneous — any mix of modalities
    // fills the one row budget, so each plan's row extent is pinned at the WORST case (max_tokens) purely to
    // size the heuristic + workspace; the view-driven @ref cublaslt_plan_t::matmul then takes the live per-step
    // tile count from the operand extents, the same algorithm serving any M.

    if (pinned && image_present)
        pinned = tower.pin_media_plans(tower.image_plan_base_, image_patch_scalars, max_tokens32, hidden32, plan_role,
                                       workspace_bytes);
    if (pinned && video_present)
        pinned = tower.pin_media_plans(tower.video_plan_base_, video_tube_scalars, max_tokens32, hidden32, plan_role,
                                       workspace_bytes);
    if (pinned && audio_present)
        pinned = tower.pin_media_plans(tower.audio_plan_base_, audio_frame_scalars, max_tokens32, hidden32, plan_role,
                                       workspace_bytes);
    if (!pinned) return result_t::fail(std::move(pinned));

    if (image_present || video_present || audio_present) {
        u32_t const max_entry_scalars = std::max({image_patch_scalars, video_tube_scalars, audio_frame_scalars});
        tower.patch_scratch_ =
            bf16r2_tensor_arena_t::uninitialized({max_tokens32, max_entry_scalars}, bf16_alloc).value;
        if (!tower.patch_scratch_)
            return result_t::fail(status_t::bad_alloc_k, "patch scratch allocation failed (arena_used={})",
                                  arena.used());
    }

    if (workspace_bytes != 0) {
        tower.workspace_ = b8r1_tensor_arena_t::uninitialized({workspace_bytes}, b8_alloc).value;
        if (!tower.workspace_)
            return result_t::fail(status_t::bad_alloc_k, "tower workspace allocation failed ({} bytes)",
                                  workspace_bytes);
    }
    return result_t {std::move(tower)};
}

expected<tower_pool_t> tower_pool_t::make(gpu_arena_t &arena, transformer_block_config_t const &block) noexcept {
    using result_t = expected<tower_pool_t>;
    tower_pool_t pool;
    if (nk_status_t const status = nk_cuda_capabilities_enabled(static_cast<nk_size_t>(arena.device()),
                                                                &pool.capabilities_);
        status != nk_success_k)
        return result_t::fail(status_t::cuda_error_k, "NumKong cannot query device {}: {}", arena.device(),
                              nk_status_name(status));
    std::size_t const segments = block.max_segments;
    std::size_t const heads = block.hidden / transformer_head_dim_k;

    nk_size_t packed_bytes = 0;
    if (nk_status_t const status = nk_attention_pack_size_bf16_best(heads, transformer_head_dim_k,
                                                                    segments * block.max_segment_tokens, segments,
                                                                    pool.capabilities_, &packed_bytes);
        status != nk_success_k)
        return result_t::fail(status_t::cuda_error_k, "pool pack sizing failed: {}", nk_status_name(status));
    array<u32_t> host;
    if (host.try_resize(segments + 1u) != status_t::success_k)
        return result_t::fail(status_t::bad_alloc_k, "pool staging allocation failed ({} segments)", segments);
    for (std::size_t segment = 0; segment <= segments; ++segment) host[segment] = static_cast<u32_t>(segment);

    pool.query_offsets_ = u32r1_tensor_arena_t::uninitialized({segments + 1u}, gpu_arena_alloc<u32_t> {arena}).value;
    pool.packed_ = b8r1_tensor_arena_t::uninitialized({static_cast<std::size_t>(packed_bytes)},
                                                      gpu_arena_alloc<b8_t> {arena})
                       .value;
    pool.pooled_ = f32r2_tensor_arena_t::uninitialized({segments, block.hidden}, gpu_arena_alloc<f32_t> {arena}).value;
    if (!pool.query_offsets_ || !pool.packed_ || !pool.pooled_)
        return result_t::fail(status_t::bad_alloc_k, "pool allocation failed (arena_used={})", arena.used());
    if (CUresult result = cuMemcpyHtoD(reinterpret_cast<CUdeviceptr>(pool.query_offsets_.data()), host.data(),
                                       (segments + 1u) * sizeof(u32_t));
        result != CUDA_SUCCESS)
        return result_t::fail(result, "pool query-offset upload failed");
    return result_t {std::move(pool)};
}

expected_t tower_pool_t::forward(bf16r1_view_t query, bf16r2_view_t states, u32r1_view_t segment_offsets,
                                 u32r1_view_t segment_lengths, bf16r2_span_t pooled, gpu_stream_t &stream) noexcept {
    std::size_t const segments = pooled_.extent(0), hidden = pooled_.extent(1);
    std::size_t const heads = hidden / transformer_head_dim_k;
    auto const *const states_data = reinterpret_cast<nk_bf16_t const *>(states.data());
    nk_status_t status = nk_attention_pack_bf16_best(
        states_data, states_data, heads, transformer_head_dim_k, segment_offsets.data(), segment_lengths.data(),
        segments, hidden * sizeof(bf16_t), hidden * sizeof(bf16_t), packed_.data(), 0, segments * heads,
        capabilities_, stream.get());
    if (status == nk_success_k)
        status = nk_attention_bidirectional_packed_bf16_best(
            reinterpret_cast<nk_bf16_t const *>(query.data()), packed_.data(), pooled_.data(), NUMKONG_NULL, heads,
            heads, transformer_head_dim_k, query_offsets_.data(), 0, hidden * sizeof(f32_t), tower_pool_scale_(),
            NUMKONG_NULL, 0, NUMKONG_SIZE_MAX, capabilities_, stream.get());
    if (status == nk_success_k)
        status = nk_cast_best(pooled_.data(), nk_f32_k, pooled.data(), nk_bf16_k, segments * hidden, capabilities_,
                              stream.get());
    if (status != nk_success_k)
        return expected_t::fail(status_t::cuda_error_k, "attention pool failed: {}", nk_status_name(status));
    return {};
}

expected_t encoder_tower_bf16_t::forward(multimodal_batch_on_gpu_t const &batch, bf16r2_span_t vertices,
                                         gpu_stream_t &stream) noexcept {
    u32_t const hidden = config_.block.hidden;
    std::size_t const columns = batch.tokens.extent(1);
    std::size_t const live_tokens = static_cast<std::size_t>(batch.tokens.extent(0)) * columns;
    std::size_t const segments = batch.live_segments();
    if (live_tokens == 0 || live_tokens > config_.block.max_tokens || segments != config_.block.max_segments)
        return expected_t::fail(status_t::shape_unsupported_k,
                                "batch disagrees with the baked capacities (tokens={} segments={})", live_tokens,
                                segments);

    // The step's live extent is the whole packed grid — modalities interleave, so there is no per-modality
    // row band to trim; PAD and inter-segment gap cells ride through as zeros. Resized within each arena's
    // reserved capacity (the check above makes this infallible; the bool is still verified, never assumed).

    bool const resized = nk::succeeded(entry_output_.resize({live_tokens, hidden})) &&
                         nk::succeeded(normalized_.resize({live_tokens, hidden})) &&
                         nk::succeeded(activations_a_.resize({live_tokens, hidden})) &&
                         nk::succeeded(activations_b_.resize({live_tokens, hidden})) &&
                         nk::succeeded(coords_.resize({live_tokens, 3}));
    if (!resized)
        return expected_t::fail(status_t::shape_unsupported_k, "tower scratch cannot hold {} live tokens (capacity {})",
                                live_tokens, entry_output_.capacity() / hidden);

    // Zero the coordinates (PAD / gap cells stay (0,0,0)) and the packed entry state (cells no segment writes
    // stay 0, and the media projections accumulate onto that zero with β=1).
    if (CUresult result = cuMemsetD8Async(reinterpret_cast<CUdeviceptr>(coords_.data()), 0,
                                          live_tokens * 3u * sizeof(i16_t), stream.get());
        result != CUDA_SUCCESS)
        return expected_t::fail(result, "coordinate reset failed");
    if (CUresult result = cuMemsetD8Async(reinterpret_cast<CUdeviceptr>(entry_output_.data()), 0,
                                          live_tokens * hidden * sizeof(bf16_t), stream.get());
        result != CUDA_SUCCESS)
        return expected_t::fail(result, "entry state reset failed");

    dim3 const block_shape {kernel_block_size_default_k, 1u, 1u};
    dim3 const segment_grid {static_cast<u32_t>(segments), 1u, 1u};
    u32r1_view_t const segment_offsets = batch.segment_offsets.view();
    u32r1_view_t const segment_lengths = batch.segment_lengths.view();
    modality_view_t const segment_modality = batch.segment_modality.view();

    // Coordinates: one segment-driven fill dispatching on modality replaces the three per-modality adapters.
    // fill_coords reads its descriptors through span-typed parameters, so the const batch's read views are
    // bridged to spans (the kernel only reads them; nothing here writes the batch).
    u32r1_span_t const offsets_span {const_cast<u32_t *>(segment_offsets.data()), segment_offsets.extent(0)};
    u32r1_span_t const lengths_span {const_cast<u32_t *>(segment_lengths.data()), segment_lengths.extent(0)};
    modality_span_t const modality_span {const_cast<modality_t *>(segment_modality.data()), segment_modality.extent(0)};
    expected_t staged = fill_coords(offsets_span, lengths_span, modality_span, batch.image_shapes.view(),
                                    batch.video_shapes.view(), config_.video_tube_frames, coords_.span(), stream);
    if (!staged) return staged;

    // Entry, per-segment gather: text segments look up the embedding table into their cells; each present
    // media modality zeroes the scratch, stages only its own cells in place (image / video pixels normalized,
    // audio raw frames cast as-is; content shares the flat grid position of its state), and accumulates its
    // projection (β=1) so the zero non-modality rows leave every other cell untouched — no scatter, no atomics.

    live_token_ids_ = batch.tokens.flatten<1>();
    staged = launch_kernel(reinterpret_cast<void const *>(&tower_text_entry_kernel_), segment_grid, block_shape, stream,
                           live_token_ids_, entry_weight_.span(), segment_offsets, segment_lengths, segment_modality,
                           entry_output_.span());
    if (!staged) return staged;

    // One β=1 projection accumulate per present media modality (zero non-modality rows compose); the shared
    // helper stages each modality's cells and folds its projection, identical to the E4M3 twin's forward.
    staged = project_media_entries(batch, plans_.data(), image_plan_base_, video_plan_base_, audio_plan_base_,
                                   patch_scratch_, workspace_, entry_output_, image_pixel_mean_,
                                   image_pixel_inverse_std_, image_projection_, video_pixel_mean_,
                                   video_pixel_inverse_std_, video_projection_, audio_projection_, *blas_, stream);
    if (!staged) return staged;

    // The block stack carries the Float32 residual stream: the entry state widens into `activations_b_`, layer n
    // reads layer n-1's output, ping-ponging the two buffers, and the last output is a stash the backward reads.

    if (nk_status_t const status = nk_cast_best(entry_output_.data(), nk_bf16_k, activations_b_.data(), nk_f32_k,
                                                live_tokens * hidden, capabilities_, stream.get());
        status != nk_success_k)
        return expected_t::fail(status_t::cuda_error_k, "entry widening failed: {}", nk_status_name(status));
    f32r2_view_t layer_input = activations_b_.view();
    for (u32_t layer_index = 0; layer_index != config_.depth; ++layer_index) {
        f32r2_tensor_arena_t &layer_output = layer_index % 2 == 0 ? activations_a_ : activations_b_;
        staged = blocks_[layer_index].forward(layer_input, layer_output.span(), segment_offsets, segment_lengths,
                                              angles_.view(), coords_.view(), stream);
        if (!staged) return staged;
        layer_input = layer_output.view();
    }
    stack_output_ = layer_input;

    // Final norm narrows the stack output into the per-token states, under the scale widened for rmscast.
    nk_status_t status = nk_cast_best(final_norm_scale_.data(), nk_bf16_k, widened_final_norm_scale_.data(), nk_f32_k,
                                      hidden, capabilities_, stream.get());
    if (status == nk_success_k)
        status = nk_each_rmscast_bf16_best(stack_output_.data(), widened_final_norm_scale_.data(),
                                           reinterpret_cast<nk_bf16_t *>(normalized_.data()), live_tokens, 1, hidden,
                                           hidden * sizeof(f32_t), hidden * sizeof(bf16_t), config_.block.rms_eps,
                                           capabilities_, stream.get());
    if (status != nk_success_k)
        return expected_t::fail(status_t::cuda_error_k, "final norm failed: {}", nk_status_name(status));

    // Masked path stops here — the masked head reads normalized(); the contrastive path pools + heads.
    if (vertices.data() == nullptr) return {};
    if (vertices.extent(0) != segments || vertices.extent(1) != hidden)
        return expected_t::fail(status_t::shape_unsupported_k, "vertices must be [{},{}] (got [{},{}])", segments,
                                hidden, vertices.extent(0), vertices.extent(1));

    // The attention pool: every segment's live tokens under the one learned query into pooled_.
    staged = pool_.forward(pool_query_.view(), normalized_.view(), segment_offsets, segment_lengths, pooled_.span(),
                           stream);
    if (!staged) return staged;

    // The head: learnable-γ RMSNorm, then the biasless fc1 → SiLU → fc2 into the caller's vertices —
    // raw, never L2-normalized here (the loss owns the cosine geometry).
    staged = launch_kernel(reinterpret_cast<void const *>(&rmsnorm_forward_kernel), segment_grid, block_shape, stream,
                           pooled_.view(), head_norm_scale_.view(), head_normalized_.span(), head_inverse_rms_.span(),
                           config_.block.rms_eps);
    if (!staged) return staged;
    staged = plans_[plan_head_forward_k].matmul(*blas_, head_normalized_.view(), head_fc1_.view(), head_hidden_.span(),
                                                0.0f, workspace_.span(), stream);
    if (!staged) return staged;
    dim3 const head_cell_grid {static_cast<u32_t>(divide_round_up(segments * hidden, kernel_block_size_default_k)), 1u,
                               1u};
    staged = launch_kernel(reinterpret_cast<void const *>(&tower_silu_kernel_), head_cell_grid, block_shape, stream,
                           head_hidden_.span().flatten<1>(), head_activated_.span().flatten<1>());
    if (!staged) return staged;
    return plans_[plan_head_forward_k].matmul(*blas_, head_activated_.view(), head_fc2_.view(), vertices, 0.0f,
                                              workspace_.span(), stream);
}

expected_t encoder_tower_bf16_t::patchify_media_targets(multimodal_batch_on_gpu_t const &batch,
                                                        bf16r2_span_t image_patches, bf16r2_span_t video_patches,
                                                        gpu_stream_t &stream) noexcept {
    // The masked-autoencoder reconstruction target: the same per-channel-normalized patch vectors the entry
    // stages before its projection, but written into the caller's clean-batch spans with no projection and no
    // forward — one shared stage per present pixel modality, non-modality and gap rows left zero. Audio's target
    // is the raw waveform frames, so the caller reshapes batch.audio_samples directly and passes nothing here.
    std::size_t const columns = batch.tokens.extent(1);
    std::size_t const live_tokens = static_cast<std::size_t>(batch.tokens.extent(0)) * columns;
    dim3 const segment_grid {static_cast<u32_t>(batch.live_segments()), 1u, 1u};
    u32r1_view_t const segment_offsets = batch.segment_offsets.view();
    u32r1_view_t const segment_lengths = batch.segment_lengths.view();
    modality_view_t const segment_modality = batch.segment_modality.view();

    expected_t staged {};
    if (image_plan_base_ != plan_absent_k && batch.images.data() != nullptr && image_patches.data() != nullptr &&
        image_patches.extent(0) != 0) {
        std::size_t const spatial = static_cast<std::size_t>(batch.images.extent(2)) * batch.images.extent(3);
        staged = stage_pixel_patches(batch.images.view().reshape<3>({live_tokens, spatial, batch.images.extent(4)}),
                                     modality_t::image_k, image_pixel_mean_.view(), image_pixel_inverse_std_.view(),
                                     segment_offsets, segment_lengths, segment_modality, segment_grid, image_patches,
                                     stream);
        if (!staged) return staged;
    }
    if (video_plan_base_ != plan_absent_k && batch.videos.data() != nullptr && video_patches.data() != nullptr &&
        video_patches.extent(0) != 0) {
        std::size_t const spatial = static_cast<std::size_t>(batch.videos.extent(2)) * batch.videos.extent(3) *
                                    batch.videos.extent(4);
        staged = stage_pixel_patches(batch.videos.view().reshape<3>({live_tokens, spatial, batch.videos.extent(5)}),
                                     modality_t::video_k, video_pixel_mean_.view(), video_pixel_inverse_std_.view(),
                                     segment_offsets, segment_lengths, segment_modality, segment_grid, video_patches,
                                     stream);
        if (!staged) return staged;
    }
    return staged;
}

expected_t encoder_tower_bf16_t::backward(bf16r2_view_t vertices_grad, bf16r2_view_t hidden_grad,
                                          multimodal_batch_on_gpu_t const &batch, gpu_stream_t &stream) noexcept {
    u32_t const hidden = config_.block.hidden;
    std::size_t const columns = batch.tokens.extent(1);
    std::size_t const live_tokens = static_cast<std::size_t>(batch.tokens.extent(0)) * columns;
    bool const pooled_present = vertices_grad.data() != nullptr;
    bool const masked_present = hidden_grad.data() != nullptr;
    if (!pooled_present && !masked_present)
        return expected_t::fail(status_t::invalid_input_k, "backward needs one of vertices_grad / hidden_grad");
    std::size_t const segments = batch.live_segments();
    if (pooled_present && (vertices_grad.extent(0) != segments || vertices_grad.extent(1) != hidden))
        return expected_t::fail(status_t::shape_unsupported_k, "vertices_grad must be [{},{}] (got [{},{}])", segments,
                                hidden, vertices_grad.extent(0), vertices_grad.extent(1));

    // The step's live extent, resized within each arena's reserved capacity — no address moves.

    bool const resized = nk::succeeded(normalized_grad_.resize({live_tokens, hidden})) &&
                         nk::succeeded(token_grad_a_.resize({live_tokens, hidden})) &&
                         nk::succeeded(token_grad_b_.resize({live_tokens, hidden}));
    if (!resized)
        return expected_t::fail(status_t::shape_unsupported_k,
                                "gradient scratch cannot hold {} live tokens (capacity {})", live_tokens,
                                token_grad_a_.capacity() / hidden);

    dim3 const block_shape {kernel_block_size_default_k, 1u, 1u};
    dim3 const row_grid {static_cast<u32_t>(live_tokens), 1u, 1u};
    dim3 const segment_grid {static_cast<u32_t>(batch.live_segments()), 1u, 1u};
    dim3 const channel_grid {static_cast<u32_t>(divide_round_up(hidden, kernel_block_size_default_k)), 1u, 1u};
    u32r1_view_t const segment_offsets = batch.segment_offsets.view();
    u32r1_view_t const segment_lengths = batch.segment_lengths.view();
    modality_view_t const segment_modality = batch.segment_modality.view();

    expected_t staged {};
    if (pooled_present) {
        // The head backward, mirroring the forward in reverse through the segment ping-pong: fc2's two
        // grads, the SiLU derivative on the stashed pre-activation, fc1's two grads, then the head norm.
        dim3 const head_cell_grid {static_cast<u32_t>(divide_round_up(segments * hidden, kernel_block_size_default_k)),
                                   1u, 1u};
        staged = plans_[plan_head_weight_grad_k].matmul(*blas_, vertices_grad, head_activated_.view(),
                                                        head_fc2_grad_.span(), 0.0f, workspace_.span(), stream);
        if (staged)
            staged = plans_[plan_head_data_grad_k].matmul(*blas_, vertices_grad, head_fc2_.view(),
                                                          segment_grad_a_.span(), 0.0f, workspace_.span(), stream);
        if (staged)
            staged = launch_kernel(reinterpret_cast<void const *>(&tower_silu_backward_kernel_), head_cell_grid,
                                   block_shape, stream, head_hidden_.span().flatten<1>(),
                                   segment_grad_a_.span().flatten<1>(), segment_grad_b_.span().flatten<1>());
        if (staged)
            staged = plans_[plan_head_weight_grad_k].matmul(*blas_, segment_grad_b_.view(), head_normalized_.view(),
                                                            head_fc1_grad_.span(), 0.0f, workspace_.span(), stream);
        if (staged)
            staged = plans_[plan_head_data_grad_k].matmul(*blas_, segment_grad_b_.view(), head_fc1_.view(),
                                                          segment_grad_a_.span(), 0.0f, workspace_.span(), stream);
        if (staged)
            staged = launch_kernel(reinterpret_cast<void const *>(&rmsnorm_backward_kernel), segment_grid, block_shape,
                                   stream, pooled_.view(), head_norm_scale_.view(), head_inverse_rms_.view(),
                                   segment_grad_a_.view(), segment_grad_b_.span(), /*accumulate=*/false);
        if (staged)
            staged = launch_kernel(reinterpret_cast<void const *>(&rmsnorm_scale_grad_kernel), channel_grid,
                                   block_shape, stream, pooled_.view(), head_inverse_rms_.view(),
                                   segment_grad_a_.view(), head_norm_scale_grad_.span());
        if (!staged) return staged;

        // The pool backward scatters d(normalized) onto each segment's live token rows (uncovered rows
        // stay the zeros this reset writes) and folds the query gradient; head_activated_ is free after
        // fc2's weight grad, so it carries the per-segment query partials.
        if (CUresult result = cuMemsetD8Async(reinterpret_cast<CUdeviceptr>(normalized_grad_.data()), 0,
                                              live_tokens * hidden * sizeof(bf16_t), stream.get());
            result != CUDA_SUCCESS)
            return expected_t::fail(result, "pool gradient reset failed");
        f32_t const attention_scale = tower_pool_scale_();
        dim3 const pool_grid {static_cast<u32_t>(segments), hidden / transformer_head_dim_k, 1u};
        dim3 const pool_block {transformer_head_dim_k, 1u, 1u};
        staged = launch_kernel(reinterpret_cast<void const *>(&tower_pool_backward_kernel_), pool_grid, pool_block,
                               stream, normalized_.view(), pool_query_.view(), segment_grad_b_.view(), segment_offsets,
                               segment_lengths, normalized_grad_.span(), head_activated_.span(), attention_scale);
        if (staged)
            staged = launch_kernel(reinterpret_cast<void const *>(&tower_pool_query_grad_kernel_), channel_grid,
                                   block_shape, stream, head_activated_.view(), pool_query_grad_.span());
        if (!staged) return staged;

        // The co-trained join: the masked force adds onto the pooled force at the final-norm states.
        if (masked_present) {
            dim3 const cell_grid {
                static_cast<u32_t>(divide_round_up(live_tokens * hidden, kernel_block_size_default_k)), 1u, 1u};
            staged = launch_kernel(reinterpret_cast<void const *>(&tower_grad_add_kernel_), cell_grid, block_shape,
                                   stream, normalized_grad_.span().flatten<1>(),
                                   hidden_grad(nk::range(0, live_tokens), nk::slice).flatten<1>());
            if (!staged) return staged;
        }
    }

    // Final-norm backward: the total d(normalized) — the pool's scatter (plus the co-trained add) or the
    // caller's masked force alone — widens into token_grad_a_, then → the Float32 d(stack output) into
    // token_grad_b_, plus the scale grad. The caller's hidden_grad may carry a wider capacity than the live
    // extent, so it is narrowed to match.
    bf16r2_view_t const normalized_grad_total = pooled_present ? normalized_grad_.view()
                                                               : hidden_grad(nk::range(0, live_tokens), nk::slice);
    if (nk_status_t const status = nk_cast_best(normalized_grad_total.data(), nk_bf16_k, token_grad_a_.data(), nk_f32_k,
                                                live_tokens * hidden, capabilities_, stream.get());
        status != nk_success_k)
        return expected_t::fail(status_t::cuda_error_k, "final-norm gradient widening failed: {}",
                                nk_status_name(status));
    staged = launch_kernel(reinterpret_cast<void const *>(&rmscast_backward_kernel), row_grid, block_shape, stream,
                           stack_output_, final_norm_scale_.view(), token_grad_a_.view(), f32r2_view_t {},
                           token_grad_b_.span(), final_inverse_rms_.span(), config_.block.rms_eps);
    if (staged)
        staged = launch_kernel(reinterpret_cast<void const *>(&rmscast_scale_grad_kernel), channel_grid, block_shape,
                               stream, stack_output_, final_inverse_rms_.view(), token_grad_a_.view(),
                               final_norm_scale_grad_.span());
    if (!staged) return staged;

    // Blocks in reverse; the flowing gradient ping-pongs the two token-gradient buffers. `coords_` still
    // holds this step's coordinates from the paired forward.
    f32r2_tensor_arena_t *grad_in = &token_grad_b_;
    f32r2_tensor_arena_t *grad_out = &token_grad_a_;
    f32r2_view_t flowing = grad_in->view();
    for (u32_t layer_index = config_.depth; layer_index != 0; --layer_index) {
        staged = blocks_[layer_index - 1].backward(flowing, grad_out->span(), segment_offsets, segment_lengths,
                                                   angles_.view(), coords_.view(), stream);
        if (!staged) return staged;
        flowing = grad_out->view();
        std::swap(grad_in, grad_out);
    }
    // The entry backward reads BFloat16, so the stack's gradient narrows into the free normalized-grad buffer.
    if (nk_status_t const status = nk_cast_best(flowing.data(), nk_f32_k, normalized_grad_.data(), nk_bf16_k,
                                                live_tokens * hidden, capabilities_, stream.get());
        status != nk_success_k)
        return expected_t::fail(status_t::cuda_error_k, "entry gradient narrowing failed: {}", nk_status_name(status));
    bf16r2_view_t const entry_grad = normalized_grad_.view();

    // Entry backward, text cells: deterministic embedding scatter over the whole packed grid. Keys start at
    // the sentinel (max token id) so every non-text cell folds into one trailing run the scatter skips; the
    // text segments overwrite their keys, positions are the flat cell indices, and the sort + serial per-run
    // fold stay deterministic with no atomics.

    std::size_t const table_scalars = static_cast<std::size_t>(config_.vocabulary) * hidden;
    if (CUresult result = cuMemsetD8Async(reinterpret_cast<CUdeviceptr>(entry_weight_grad_.data()), 0,
                                          table_scalars * sizeof(bf16_t), stream.get());
        result != CUDA_SUCCESS)
        return expected_t::fail(result, "embedding gradient reset failed");
    // The sort scratch is one arena split into two halves, max_tokens apart: [0,live) and [max,max+live).

    std::size_t const max_tokens = config_.block.max_tokens;
    tokens_span_t const keys_in = sort_keys_(nk::range(0, live_tokens), nk::slice);
    tokens_span_t const keys_out = sort_keys_(nk::range(max_tokens, max_tokens + live_tokens), nk::slice);
    u32r1_span_t const values_in = sort_values_(nk::range(0, live_tokens), nk::slice);
    u32r1_span_t const values_out = sort_values_(nk::range(max_tokens, max_tokens + live_tokens), nk::slice);
    if (CUresult result = cuMemsetD8Async(reinterpret_cast<CUdeviceptr>(keys_in.data()), 0xFF,
                                          live_tokens * sizeof(token_id_t), stream.get());
        result != CUDA_SUCCESS)
        return expected_t::fail(result, "sentinel key reset failed");
    staged = launch_kernel(reinterpret_cast<void const *>(&tower_text_keys_kernel_), segment_grid, block_shape, stream,
                           live_token_ids_, segment_offsets, segment_lengths, segment_modality, keys_in);
    if (staged)
        staged = launch_kernel(reinterpret_cast<void const *>(&tower_iota_kernel_),
                               dim3 {grid_size_for(live_tokens), 1u, 1u}, dim3 {warp_size_k, 1u, 1u}, stream,
                               values_in);
    if (!staged) return staged;
    std::size_t sort_bytes = sort_workspace_.extent(0);
    if (cudaError_t sorted = cub::DeviceRadixSort::SortPairs(
            sort_workspace_.data(), sort_bytes, keys_in.data(), keys_out.data(), values_in.data(), values_out.data(),
            static_cast<int>(live_tokens), 0, sizeof(token_id_t) * 8, reinterpret_cast<cudaStream_t>(stream.get()));
        sorted != cudaSuccess)
        return expected_t::fail(status_t::cuda_error_k, "embedding sort failed (code={})", static_cast<int>(sorted));
    staged = launch_kernel(reinterpret_cast<void const *>(&tower_embedding_scatter_kernel_), row_grid, block_shape,
                           stream, tokens_view_t {keys_out}, u32r1_view_t {values_out}, entry_grad,
                           entry_weight_grad_.span());
    if (!staged) return staged;

    // Media weight grads: re-stage each present modality's cells (segment-driven, so non-modality rows stay
    // zero) and fold the whole entry gradient against them — the zero patch rows drop every non-modality term,
    // so dW gathers exactly its modality's cells. The backward twin of project_media_entries; coords_ was
    // filled by the paired forward, so the backward derives no coordinates.
    staged = project_media_weight_grads(
        batch, plans_.data(), image_plan_base_, video_plan_base_, audio_plan_base_, patch_scratch_, workspace_,
        entry_grad, image_pixel_mean_, image_pixel_inverse_std_, image_projection_grad_, video_pixel_mean_,
        video_pixel_inverse_std_, video_projection_grad_, audio_projection_grad_, *blas_, stream);
    if (!staged) return staged;

    return staged;
}

expected_t encoder_tower_bf16_t::pack_weights(gpu_stream_t &stream) noexcept {
    for (transformer_block_bf16_t &layer : blocks_)
        if (expected_t packed = layer.pack_weights(stream); !packed) return packed;
    return {};
}

expected_t encoder_tower_bf16_t::fill(array<differentiable_weights_t> &weights) noexcept {
    u32_t const hidden = config_.block.hidden;
    u32_t const ffn = config_.block.ffn_expansion * hidden;
    // Every weight joins under the checkpoint name the tower spells for it — the composition root adds only
    // its `model.` prefix. A block weight arrives as a rank-1 accessor; it joins as its logical [rows,cols]
    // rectangle (a tower matrix is already rank-2 and joins through its own span / view).

    auto push = [&weights](sz::string_view name, bf16r2_span_t weight, bf16r2_view_t grad) noexcept {
        differentiable_weights_t entry {{}, weight, grad};
        if (!entry.name.try_assign(name)) return status_t::bad_alloc_k;
        return weights.try_push_back(std::move(entry));
    };
    auto push_flat = [&push](sz::string_view name, bf16r1_span_t weight, bf16r1_view_t grad, std::size_t rows,
                             std::size_t cols) noexcept {
        return push(name, bf16r2_span_t {weight.data(), rows, cols}, bf16r2_view_t {grad.data(), rows, cols});
    };
    status_t pushed = push("embedding", entry_weight_.span(), entry_weight_grad_.view());
    // Media projections next, in modality order — each present iff its plan base was pinned. These rank-2
    // [hidden, fan_in] matrices join directly (like the head products), replica-identical with the grads
    // backward() produces, so the optimizer trains what was frozen before.
    if (pushed == status_t::success_k && image_plan_base_ != plan_absent_k)
        pushed = push("image_projection", image_projection_.span(), image_projection_grad_.view());
    if (pushed == status_t::success_k && video_plan_base_ != plan_absent_k)
        pushed = push("video_projection", video_projection_.span(), video_projection_grad_.view());
    if (pushed == status_t::success_k && audio_plan_base_ != plan_absent_k)
        pushed = push("audio_projection", audio_projection_.span(), audio_projection_grad_.view());
    sz::string name_text;
    for (u32_t layer_index = 0; pushed == status_t::success_k && layer_index != config_.depth; ++layer_index) {
        transformer_block_bf16_t &layer = blocks_[layer_index];
        struct block_part_t {
            sz::string_view suffix;
            bf16r1_span_t weight;
            bf16r1_view_t grad;
            std::size_t rows, cols;
        };
        block_part_t const parts[] = {
            {"qkv", layer.qkv_weight(), layer.qkv_weight_grad(), 3u * hidden, hidden},
            {"out", layer.out_weight(), layer.out_weight_grad(), hidden, hidden},
            {"gate_up", layer.gate_up_weight(), layer.gate_up_weight_grad(), 2u * ffn, hidden},
            {"down", layer.down_weight(), layer.down_weight_grad(), hidden, ffn},
            {"pre_norm", layer.pre_norm_scale(), layer.pre_norm_scale_grad(), 1u, hidden},
            {"post_norm", layer.post_norm_scale(), layer.post_norm_scale_grad(), 1u, hidden},
        };
        for (block_part_t const &part : parts) {
            name_text.clear();
            if (!format_into(name_text, "blocks.{}.{}", layer_index, part.suffix))
                return expected_t::fail(status_t::bad_alloc_k, "weight name format failed");
            pushed = push_flat(name_text.view(), part.weight, part.grad, part.rows, part.cols);
            if (pushed != status_t::success_k) break;
        }
    }
    if (pushed == status_t::success_k)
        pushed = push_flat("final_norm", final_norm_scale_.span(), final_norm_scale_grad_.view(), 1u, hidden);
    if (pushed == status_t::success_k)
        pushed = push_flat("pool_query", pool_query_.span(), pool_query_grad_.view(), 1u, hidden);
    if (pushed == status_t::success_k)
        pushed = push_flat("head.norm", head_norm_scale_.span(), head_norm_scale_grad_.view(), 1u, hidden);
    if (pushed == status_t::success_k) pushed = push("head.fc1", head_fc1_.span(), head_fc1_grad_.view());
    if (pushed == status_t::success_k) pushed = push("head.fc2", head_fc2_.span(), head_fc2_grad_.view());
    if (pushed != status_t::success_k) return expected_t::fail(status_t::bad_alloc_k, "weight table overflowed");
    return {};
}

/** @brief Snapshot one BFloat16 matrix into a fresh arena tensor — the twin's tail tiers (entries,
 *      norms, pool, head) stay BFloat16, so their quantization is a plain device-to-device copy.
 *      @p name tags the failure diagnostic with the weight's role. */
[[nodiscard]] static expected<bf16r2_tensor_arena_t> snapshot_weight_(bf16r2_view_t source, gpu_arena_t &arena,
                                                                      gpu_stream_t &stream,
                                                                      sz::string_view name) noexcept {
    using result_t = expected<bf16r2_tensor_arena_t>;
    gpu_arena_alloc<bf16_t> bf16_alloc {arena};
    bf16r2_tensor_arena_t copy =
        bf16r2_tensor_arena_t::uninitialized({source.extent(0), source.extent(1)}, bf16_alloc).value;
    if (!copy) return result_t::fail(status_t::bad_alloc_k, "twin {} allocation failed", name);
    // vendor-boundary: device→device weight snapshot — no typed DtoD helper exists, so the driver copy
    // is spelled by hand over the views' raw `data()` pointers.
    if (CUresult result = cuMemcpyDtoDAsync(reinterpret_cast<CUdeviceptr>(copy.data()),
                                            reinterpret_cast<CUdeviceptr>(source.data()),
                                            source.extent(0) * source.extent(1) * sizeof(bf16_t), stream.get());
        result != CUDA_SUCCESS)
        return result_t::fail(result, "twin {} snapshot failed", name);
    return result_t {std::move(copy)};
}

/** @brief The rank-1 overload of @ref snapshot_weight_ for the norm scales and the pool query. */
[[nodiscard]] static expected<bf16r1_tensor_arena_t> snapshot_weight_(bf16r1_view_t source, gpu_arena_t &arena,
                                                                      gpu_stream_t &stream,
                                                                      sz::string_view name) noexcept {
    using result_t = expected<bf16r1_tensor_arena_t>;
    gpu_arena_alloc<bf16_t> bf16_alloc {arena};
    bf16r1_tensor_arena_t copy = bf16r1_tensor_arena_t::uninitialized({source.extent(0)}, bf16_alloc).value;
    if (!copy) return result_t::fail(status_t::bad_alloc_k, "twin {} allocation failed", name);
    // vendor-boundary: device→device weight snapshot, as in the rank-2 overload above.
    if (CUresult result = cuMemcpyDtoDAsync(reinterpret_cast<CUdeviceptr>(copy.data()),
                                            reinterpret_cast<CUdeviceptr>(source.data()),
                                            source.extent(0) * sizeof(bf16_t), stream.get());
        result != CUDA_SUCCESS)
        return result_t::fail(result, "twin {} snapshot failed", name);
    return result_t {std::move(copy)};
}

expected_t encoder_tower_e4m3_t::pin_plan(cublaslt_plan_config_t plan_config, std::size_t &role,
                                          std::size_t &workspace_bytes) noexcept {
    plan_config.c_type = CUDA_R_16BF;
    expected<cublaslt_plan_t> plan = cublaslt_plan_t::make(*blas_, plan_config);
    if (!plan) return expected_t::fail(std::move(plan));
    workspace_bytes = std::max(workspace_bytes, plan.value_.workspace_bytes());
    plans_[role++] = std::move(plan.value_);
    return {};
}

expected_t encoder_tower_e4m3_t::pin_media_plan(std::size_t &plan_base, u32_t fan_in, u32_t max_tokens, u32_t hidden,
                                                std::size_t &role, std::size_t &workspace_bytes) noexcept {
    plan_base = role;
    return pin_plan({max_tokens, fan_in, fan_in, hidden, fan_in, fan_in, max_tokens, hidden, hidden, false, true}, role,
                    workspace_bytes);
}

expected<encoder_tower_e4m3_t> encoder_tower_bf16_t::quantize_weights(gpu_arena_t &arena,
                                                                      gpu_stream_t &stream) noexcept {
    using result_t = expected<encoder_tower_e4m3_t>;
    encoder_tower_e4m3_t twin;
    twin.config_ = config_;
    twin.blas_ = blas_;

    std::size_t const hidden = config_.block.hidden;
    std::size_t const tokens = config_.block.max_tokens;
    std::size_t const segments = config_.block.max_segments;
    u32_t const hidden32 = config_.block.hidden;
    u32_t const max_tokens32 = config_.block.max_tokens;
    u32_t const segments32 = config_.block.max_segments;
    gpu_arena_alloc<bf16_t> bf16_alloc {arena};
    gpu_arena_alloc<f32_t> f32_alloc {arena};
    gpu_arena_alloc<i16_t> i16_alloc {arena};
    gpu_arena_alloc<b8_t> b8_alloc {arena};

    // The block stack quantizes per layer; everything else below is a BFloat16 snapshot.
    if (twin.blocks_.try_resize(config_.depth) != status_t::success_k)
        return result_t::fail(status_t::bad_alloc_k, "twin block table allocation failed ({})", config_.depth);
    for (u32_t layer_index = 0; layer_index != config_.depth; ++layer_index) {
        expected<transformer_block_e4m3_t> block = blocks_[layer_index].quantize_weights(arena, stream);
        if (!block) return result_t::fail(std::move(block));
        twin.blocks_[layer_index] = std::move(block.value_);
    }

    // The BFloat16 tail weights, snapshotted from the trained tower.
    expected<bf16r2_tensor_arena_t> entry_copy = snapshot_weight_(entry_weight_.view(), arena, stream, "embedding");
    if (!entry_copy) return result_t::fail(std::move(entry_copy));
    twin.entry_weight_ = std::move(entry_copy.value_);
    expected<bf16r1_tensor_arena_t> final_norm_copy = snapshot_weight_(final_norm_scale_.view(), arena, stream,
                                                                       "final norm");
    if (!final_norm_copy) return result_t::fail(std::move(final_norm_copy));
    twin.final_norm_scale_ = std::move(final_norm_copy.value_);
    expected<bf16r1_tensor_arena_t> query_copy = snapshot_weight_(pool_query_.view(), arena, stream, "pool query");
    if (!query_copy) return result_t::fail(std::move(query_copy));
    twin.pool_query_ = std::move(query_copy.value_);
    expected<bf16r1_tensor_arena_t> head_norm_copy = snapshot_weight_(head_norm_scale_.view(), arena, stream,
                                                                      "head norm");
    if (!head_norm_copy) return result_t::fail(std::move(head_norm_copy));
    twin.head_norm_scale_ = std::move(head_norm_copy.value_);
    expected<bf16r2_tensor_arena_t> fc1_copy = snapshot_weight_(head_fc1_.view(), arena, stream, "head fc1");
    if (!fc1_copy) return result_t::fail(std::move(fc1_copy));
    twin.head_fc1_ = std::move(fc1_copy.value_);
    expected<bf16r2_tensor_arena_t> fc2_copy = snapshot_weight_(head_fc2_.view(), arena, stream, "head fc2");
    if (!fc2_copy) return result_t::fail(std::move(fc2_copy));
    twin.head_fc2_ = std::move(fc2_copy.value_);

    // The fused head product first, then — per present media modality — the projection snapshot and its one
    // forward plan, in modality order, exactly as the training tower's make pins its own.
    std::size_t workspace_bytes = 0;
    std::size_t plan_role = 0;
    expected_t pinned = twin.pin_plan(
        {segments32, hidden32, hidden32, hidden32, hidden32, hidden32, segments32, hidden32, hidden32, false, true},
        plan_role, workspace_bytes);
    if (!pinned) return result_t::fail(std::move(pinned));

    u32_t max_entry_scalars = 0;
    if (image_projection_.data() != nullptr) {
        u32_t const fan_in = static_cast<u32_t>(image_projection_.view().extent(1));
        expected<bf16r2_tensor_arena_t> projection = snapshot_weight_(image_projection_.view(), arena, stream,
                                                                      "image projection");
        if (!projection) return result_t::fail(std::move(projection));
        twin.image_projection_ = std::move(projection.value_);
        pinned = twin.pin_media_plan(twin.image_plan_base_, fan_in, max_tokens32, hidden32, plan_role, workspace_bytes);
        if (!pinned) return result_t::fail(std::move(pinned));
        max_entry_scalars = std::max(max_entry_scalars, fan_in);
    }
    if (video_projection_.data() != nullptr) {
        u32_t const fan_in = static_cast<u32_t>(video_projection_.view().extent(1));
        expected<bf16r2_tensor_arena_t> projection = snapshot_weight_(video_projection_.view(), arena, stream,
                                                                      "video projection");
        if (!projection) return result_t::fail(std::move(projection));
        twin.video_projection_ = std::move(projection.value_);
        pinned = twin.pin_media_plan(twin.video_plan_base_, fan_in, max_tokens32, hidden32, plan_role, workspace_bytes);
        if (!pinned) return result_t::fail(std::move(pinned));
        max_entry_scalars = std::max(max_entry_scalars, fan_in);
    }
    if (audio_projection_.data() != nullptr) {
        u32_t const fan_in = static_cast<u32_t>(audio_projection_.view().extent(1));
        expected<bf16r2_tensor_arena_t> projection = snapshot_weight_(audio_projection_.view(), arena, stream,
                                                                      "audio projection");
        if (!projection) return result_t::fail(std::move(projection));
        twin.audio_projection_ = std::move(projection.value_);
        pinned = twin.pin_media_plan(twin.audio_plan_base_, fan_in, max_tokens32, hidden32, plan_role, workspace_bytes);
        if (!pinned) return result_t::fail(std::move(pinned));
        max_entry_scalars = std::max(max_entry_scalars, fan_in);
    }
    if (twin.image_plan_base_ != encoder_tower_e4m3_t::plan_absent_k) {
        expected_t uploaded = upload_pixel_constants(twin.image_pixel_mean_, twin.image_pixel_inverse_std_,
                                                     config_.image_pixel_mean, config_.image_pixel_std,
                                                     config_.image_channels, f32_alloc, arena);
        if (!uploaded) return result_t::fail(std::move(uploaded));
    }
    if (twin.video_plan_base_ != encoder_tower_e4m3_t::plan_absent_k) {
        expected_t uploaded = upload_pixel_constants(twin.video_pixel_mean_, twin.video_pixel_inverse_std_,
                                                     config_.video_pixel_mean, config_.video_pixel_std,
                                                     config_.video_channels, f32_alloc, arena);
        if (!uploaded) return result_t::fail(std::move(uploaded));
    }

    // The rotary tables rebuild from the same config (their content is a pure function of it), and the
    // forward activations allocate at the tower's capacities.
    expected<rope_angles_t> angles = rope_angles_t::make(arena, config_.rope_max_positions, config_.rope,
                                                         config_.rope_base);
    if (!angles) return result_t::fail(std::move(angles));
    twin.angles_ = std::move(angles.value_);
    twin.coords_ = i16r2_tensor_arena_t::uninitialized({tokens, 3}, i16_alloc).value;
    twin.entry_output_ = bf16r2_tensor_arena_t::uninitialized({tokens, hidden}, bf16_alloc).value;
    twin.normalized_ = bf16r2_tensor_arena_t::uninitialized({tokens, hidden}, bf16_alloc).value;
    twin.pooled_ = bf16r2_tensor_arena_t::uninitialized({segments, hidden}, bf16_alloc).value;
    twin.head_normalized_ = bf16r2_tensor_arena_t::uninitialized({segments, hidden}, bf16_alloc).value;
    twin.head_hidden_ = bf16r2_tensor_arena_t::uninitialized({segments, hidden}, bf16_alloc).value;
    twin.head_activated_ = bf16r2_tensor_arena_t::uninitialized({segments, hidden}, bf16_alloc).value;
    twin.final_inverse_rms_ = f32r1_tensor_arena_t::uninitialized({tokens}, f32_alloc).value;
    twin.head_inverse_rms_ = f32r1_tensor_arena_t::uninitialized({segments}, f32_alloc).value;
    twin.activations_a_ = bf16r2_tensor_arena_t::uninitialized({tokens, hidden}, bf16_alloc).value;
    twin.activations_b_ = bf16r2_tensor_arena_t::uninitialized({tokens, hidden}, bf16_alloc).value;
    if (!twin.coords_ || !twin.entry_output_ || !twin.normalized_ || !twin.pooled_ || !twin.head_normalized_ ||
        !twin.head_hidden_ || !twin.head_activated_ || !twin.final_inverse_rms_ || !twin.head_inverse_rms_ ||
        !twin.activations_a_ || !twin.activations_b_)
        return result_t::fail(status_t::bad_alloc_k, "twin activation allocation failed (arena_used={})", arena.used());
    if (max_entry_scalars != 0) {
        twin.patch_scratch_ = bf16r2_tensor_arena_t::uninitialized({tokens, max_entry_scalars}, bf16_alloc).value;
        if (!twin.patch_scratch_)
            return result_t::fail(status_t::bad_alloc_k, "twin patch scratch allocation failed (arena_used={})",
                                  arena.used());
    }
    if (workspace_bytes != 0) {
        twin.workspace_ = b8r1_tensor_arena_t::uninitialized({workspace_bytes}, b8_alloc).value;
        if (!twin.workspace_)
            return result_t::fail(status_t::bad_alloc_k, "twin workspace allocation failed ({} bytes)",
                                  workspace_bytes);
    }

    expected<tower_pool_t> pool = tower_pool_t::make(arena, twin.config_.block);
    if (!pool) return result_t::fail(std::move(pool));
    twin.pool_ = std::move(pool.value_);

    // The snapshots above are stream-ordered; drain them so the twin owns settled weights.
    if (CUresult result = cuStreamSynchronize(stream.get()); result != CUDA_SUCCESS)
        return result_t::fail(result, "twin snapshot drain failed");
    return result_t {std::move(twin)};
}

expected_t encoder_tower_e4m3_t::forward(multimodal_batch_on_gpu_t const &batch, bf16r2_span_t vertices,
                                         gpu_stream_t &stream) noexcept {
    u32_t const hidden = config_.block.hidden;
    std::size_t const columns = batch.tokens.extent(1);
    std::size_t const live_tokens = static_cast<std::size_t>(batch.tokens.extent(0)) * columns;
    std::size_t const segments = batch.live_segments();
    if (live_tokens == 0 || live_tokens > config_.block.max_tokens || segments != config_.block.max_segments)
        return expected_t::fail(status_t::shape_unsupported_k,
                                "batch disagrees with the baked capacities (tokens={} segments={})", live_tokens,
                                segments);

    bool const resized = nk::succeeded(entry_output_.resize({live_tokens, hidden})) &&
                         nk::succeeded(activations_a_.resize({live_tokens, hidden})) &&
                         nk::succeeded(activations_b_.resize({live_tokens, hidden})) &&
                         nk::succeeded(coords_.resize({live_tokens, 3}));
    if (!resized)
        return expected_t::fail(status_t::shape_unsupported_k, "tower scratch cannot hold {} live tokens (capacity {})",
                                live_tokens, entry_output_.capacity() / hidden);

    if (CUresult result = cuMemsetD8Async(reinterpret_cast<CUdeviceptr>(coords_.data()), 0,
                                          live_tokens * 3u * sizeof(i16_t), stream.get());
        result != CUDA_SUCCESS)
        return expected_t::fail(result, "coordinate reset failed");
    if (CUresult result = cuMemsetD8Async(reinterpret_cast<CUdeviceptr>(entry_output_.data()), 0,
                                          live_tokens * hidden * sizeof(bf16_t), stream.get());
        result != CUDA_SUCCESS)
        return expected_t::fail(result, "entry state reset failed");

    dim3 const block_shape {kernel_block_size_default_k, 1u, 1u};
    dim3 const segment_grid {static_cast<u32_t>(segments), 1u, 1u};
    u32r1_view_t const segment_offsets = batch.segment_offsets.view();
    u32r1_view_t const segment_lengths = batch.segment_lengths.view();
    modality_view_t const segment_modality = batch.segment_modality.view();

    // Coordinates + per-segment entry, mirroring the training twin: one fill_coords, a text-only embedding
    // gather, and a β=1 projection accumulate per present media modality (zero non-modality rows compose).
    u32r1_span_t const offsets_span {const_cast<u32_t *>(segment_offsets.data()), segment_offsets.extent(0)};
    u32r1_span_t const lengths_span {const_cast<u32_t *>(segment_lengths.data()), segment_lengths.extent(0)};
    modality_span_t const modality_span {const_cast<modality_t *>(segment_modality.data()), segment_modality.extent(0)};
    expected_t staged = fill_coords(offsets_span, lengths_span, modality_span, batch.image_shapes.view(),
                                    batch.video_shapes.view(), config_.video_tube_frames, coords_.span(), stream);
    if (!staged) return staged;

    tokens_view_t const token_ids = batch.tokens.flatten<1>();
    staged = launch_kernel(reinterpret_cast<void const *>(&tower_text_entry_kernel_), segment_grid, block_shape, stream,
                           token_ids, entry_weight_.span(), segment_offsets, segment_lengths, segment_modality,
                           entry_output_.span());
    if (!staged) return staged;

    // One β=1 projection accumulate per present media modality — the same shared helper the training twin runs.
    staged = project_media_entries(batch, plans_.data(), image_plan_base_, video_plan_base_, audio_plan_base_,
                                   patch_scratch_, workspace_, entry_output_, image_pixel_mean_,
                                   image_pixel_inverse_std_, image_projection_, video_pixel_mean_,
                                   video_pixel_inverse_std_, video_projection_, audio_projection_, *blas_, stream);
    if (!staged) return staged;

    // The E4M3 block stack: layer n reads layer n-1's output, ping-ponging two dedicated buffers.
    bf16r2_view_t layer_input = entry_output_.view();
    for (u32_t layer_index = 0; layer_index != config_.depth; ++layer_index) {
        bf16r2_tensor_arena_t &layer_output = layer_index % 2 == 0 ? activations_a_ : activations_b_;
        staged = blocks_[layer_index].forward(layer_input, layer_output.span(), segment_offsets, segment_lengths,
                                              angles_.view(), coords_.view(), stream);
        if (!staged) return staged;
        layer_input = layer_output.view();
    }

    // Final norm over the packed token states — the block stack already emits BFloat16, so the tail
    // runs the training twin's own kernels with no dtype boundary.
    if (nk::failed(normalized_.resize({live_tokens, hidden})))
        return expected_t::fail(status_t::shape_unsupported_k, "twin norm scratch cannot hold {} live tokens",
                                live_tokens);
    dim3 const token_grid {static_cast<u32_t>(live_tokens), 1u, 1u};
    staged = launch_kernel(reinterpret_cast<void const *>(&rmsnorm_forward_kernel), token_grid, block_shape, stream,
                           layer_input, final_norm_scale_.view(), normalized_.span(), final_inverse_rms_.span(),
                           config_.block.rms_eps);
    if (!staged) return staged;

    // Masked path stops here; the contrastive path pools + heads, exactly as the training twin.
    if (vertices.data() == nullptr) return {};
    if (vertices.extent(0) != segments || vertices.extent(1) != hidden)
        return expected_t::fail(status_t::shape_unsupported_k, "vertices must be [{},{}] (got [{},{}])", segments,
                                hidden, vertices.extent(0), vertices.extent(1));
    staged = pool_.forward(pool_query_.view(), normalized_.view(), segment_offsets, segment_lengths, pooled_.span(),
                           stream);
    if (!staged) return staged;
    staged = launch_kernel(reinterpret_cast<void const *>(&rmsnorm_forward_kernel), segment_grid, block_shape, stream,
                           pooled_.view(), head_norm_scale_.view(), head_normalized_.span(), head_inverse_rms_.span(),
                           config_.block.rms_eps);
    if (!staged) return staged;
    staged = plans_[plan_head_forward_k].matmul(*blas_, head_normalized_.view(), head_fc1_.view(), head_hidden_.span(),
                                                0.0f, workspace_.span(), stream);
    if (!staged) return staged;
    dim3 const head_cell_grid {static_cast<u32_t>(divide_round_up(segments * hidden, kernel_block_size_default_k)), 1u,
                               1u};
    staged = launch_kernel(reinterpret_cast<void const *>(&tower_silu_kernel_), head_cell_grid, block_shape, stream,
                           head_hidden_.span().flatten<1>(), head_activated_.span().flatten<1>());
    if (!staged) return staged;
    return plans_[plan_head_forward_k].matmul(*blas_, head_activated_.view(), head_fc2_.view(), vertices, 0.0f,
                                              workspace_.span(), stream);
}
