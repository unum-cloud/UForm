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
    mean_arena = f32r1_tensor_arena_t::try_empty({channels}, f32_alloc);
    inverse_std_arena = f32r1_tensor_arena_t::try_empty({channels}, f32_alloc);
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
    tower.cudnn_ = &cudnn;

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
    tower.entry_weight_ = bf16r2_tensor_arena_t::try_empty({vocabulary, hidden}, bf16_alloc);
    tower.entry_weight_grad_ = bf16r2_tensor_arena_t::try_empty({vocabulary, hidden}, bf16_alloc);
    gpu_arena_alloc<token_id_t> token_alloc {arena};
    tower.sort_keys_ = tokens_tensor_arena_t::try_empty({2 * tokens}, token_alloc);
    tower.sort_values_ = u32r1_tensor_arena_t::try_empty({2 * tokens}, u32_alloc);
    if (tower.sort_keys_ && tower.sort_values_) {
        std::size_t sort_bytes = 0;
        cub::DeviceRadixSort::SortPairs(nullptr, sort_bytes, tower.sort_keys_.data(), tower.sort_keys_.data() + tokens,
                                        tower.sort_values_.data(), tower.sort_values_.data() + tokens,
                                        static_cast<int>(tokens));
        tower.sort_workspace_ = b8r1_tensor_arena_t::try_empty({sort_bytes}, b8_alloc);
    }
    if (!tower.entry_weight_ || !tower.entry_weight_grad_ || !tower.sort_keys_ || !tower.sort_values_ ||
        !tower.sort_workspace_)
        return result_t::fail(status_t::bad_alloc_k, "token entry allocation failed (arena_used={})", arena.used());

    // Final norm, attention-pool query, and the head weights, each with their gradient.
    tower.final_norm_scale_ = bf16r1_tensor_arena_t::try_empty({hidden}, bf16_alloc);
    tower.final_norm_scale_grad_ = bf16r1_tensor_arena_t::try_empty({hidden}, bf16_alloc);
    tower.pool_query_ = bf16r1_tensor_arena_t::try_empty({hidden}, bf16_alloc);
    tower.pool_query_grad_ = bf16r1_tensor_arena_t::try_empty({hidden}, bf16_alloc);
    tower.head_norm_scale_ = bf16r1_tensor_arena_t::try_empty({hidden}, bf16_alloc);
    tower.head_norm_scale_grad_ = bf16r1_tensor_arena_t::try_empty({hidden}, bf16_alloc);
    tower.head_fc1_ = bf16r2_tensor_arena_t::try_empty({hidden, hidden}, bf16_alloc);
    tower.head_fc1_grad_ = bf16r2_tensor_arena_t::try_empty({hidden, hidden}, bf16_alloc);
    tower.head_fc2_ = bf16r2_tensor_arena_t::try_empty({hidden, hidden}, bf16_alloc);
    tower.head_fc2_grad_ = bf16r2_tensor_arena_t::try_empty({hidden, hidden}, bf16_alloc);
    if (!tower.final_norm_scale_ || !tower.final_norm_scale_grad_ || !tower.pool_query_ || !tower.pool_query_grad_ ||
        !tower.head_norm_scale_ || !tower.head_norm_scale_grad_ || !tower.head_fc1_ || !tower.head_fc1_grad_ ||
        !tower.head_fc2_ || !tower.head_fc2_grad_)
        return result_t::fail(status_t::bad_alloc_k, "head weights allocation failed (arena_used={})", arena.used());

    // Forward stashes and the two inverse-rms stashes.
    tower.entry_output_ = bf16r2_tensor_arena_t::try_empty({tokens, hidden}, bf16_alloc);
    tower.normalized_ = bf16r2_tensor_arena_t::try_empty({tokens, hidden}, bf16_alloc);
    tower.pooled_ = bf16r2_tensor_arena_t::try_empty({segments, hidden}, bf16_alloc);
    tower.head_normalized_ = bf16r2_tensor_arena_t::try_empty({segments, hidden}, bf16_alloc);
    tower.head_hidden_ = bf16r2_tensor_arena_t::try_empty({segments, hidden}, bf16_alloc);
    tower.head_activated_ = bf16r2_tensor_arena_t::try_empty({segments, hidden}, bf16_alloc);
    tower.final_inverse_rms_ = f32r1_tensor_arena_t::try_empty({tokens}, f32_alloc);
    tower.head_inverse_rms_ = f32r1_tensor_arena_t::try_empty({segments}, f32_alloc);
    tower.coords_ = i16r2_tensor_arena_t::try_empty({tokens, 3}, i16_alloc);
    if (!tower.entry_output_ || !tower.normalized_ || !tower.pooled_ || !tower.head_normalized_ ||
        !tower.head_hidden_ || !tower.head_activated_ || !tower.final_inverse_rms_ || !tower.head_inverse_rms_ ||
        !tower.coords_)
        return result_t::fail(status_t::bad_alloc_k, "stashes allocation failed (arena_used={})", arena.used());

    // The activation and gradient ping-pong buffers.
    tower.activations_a_ = bf16r2_tensor_arena_t::try_empty({tokens, hidden}, bf16_alloc);
    tower.activations_b_ = bf16r2_tensor_arena_t::try_empty({tokens, hidden}, bf16_alloc);
    tower.token_grad_a_ = bf16r2_tensor_arena_t::try_empty({tokens, hidden}, bf16_alloc);
    tower.token_grad_b_ = bf16r2_tensor_arena_t::try_empty({tokens, hidden}, bf16_alloc);
    tower.segment_grad_a_ = bf16r2_tensor_arena_t::try_empty({segments, hidden}, bf16_alloc);
    tower.segment_grad_b_ = bf16r2_tensor_arena_t::try_empty({segments, hidden}, bf16_alloc);
    if (!tower.activations_a_ || !tower.activations_b_ || !tower.token_grad_a_ || !tower.token_grad_b_ ||
        !tower.segment_grad_a_ || !tower.segment_grad_b_)
        return result_t::fail(status_t::bad_alloc_k, "ping-pong allocation failed (arena_used={})", arena.used());

    // The RoPE angle tables (model state; the frontend derives the per-token coordinates).

    expected<rope_angles_t> angles = rope_angles_t::make(arena, config.rope_max_positions, config.rope,
                                                         config.rope_base);
    if (!angles) return result_t::fail(std::move(angles));
    tower.angles_ = std::move(angles.value_);

    // The block stack.
    if (tower.blocks_.try_resize(config.depth) != status_t::success_k)
        return result_t::fail(status_t::bad_alloc_k, "block table allocation failed ({})", config.depth);
    for (u32_t layer_index = 0; layer_index != config.depth; ++layer_index) {
        expected<transformer_block_bf16_t> block = transformer_block_bf16_t::make(arena, blas, cudnn, config.block);
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
        tower.image_projection_ = bf16r2_tensor_arena_t::try_empty({hidden, image_patch_scalars}, bf16_alloc);
        tower.image_projection_grad_ = bf16r2_tensor_arena_t::try_empty({hidden, image_patch_scalars}, bf16_alloc);
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
        tower.video_projection_ = bf16r2_tensor_arena_t::try_empty({hidden, video_tube_scalars}, bf16_alloc);
        tower.video_projection_grad_ = bf16r2_tensor_arena_t::try_empty({hidden, video_tube_scalars}, bf16_alloc);
        if (!tower.video_projection_ || !tower.video_projection_grad_)
            return result_t::fail(status_t::bad_alloc_k, "video projection allocation failed (arena_used={})",
                                  arena.used());
    }
    if (audio_present) {
        tower.audio_projection_ = bf16r2_tensor_arena_t::try_empty({hidden, audio_frame_scalars}, bf16_alloc);
        tower.audio_projection_grad_ = bf16r2_tensor_arena_t::try_empty({hidden, audio_frame_scalars}, bf16_alloc);
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
        tower.patch_scratch_ = bf16r2_tensor_arena_t::try_empty({max_tokens32, max_entry_scalars}, bf16_alloc);
        if (!tower.patch_scratch_)
            return result_t::fail(status_t::bad_alloc_k, "patch scratch allocation failed (arena_used={})",
                                  arena.used());
    }

    if (workspace_bytes != 0) {
        tower.workspace_ = b8r1_tensor_arena_t::try_empty({workspace_bytes}, b8_alloc);
        if (!tower.workspace_)
            return result_t::fail(status_t::bad_alloc_k, "tower workspace allocation failed ({} bytes)",
                                  workspace_bytes);
    }
    return result_t {std::move(tower)};
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

    bool const resized = entry_output_.try_resize({live_tokens, hidden}) &&
                         normalized_.try_resize({live_tokens, hidden}) &&
                         activations_a_.try_resize({live_tokens, hidden}) &&
                         activations_b_.try_resize({live_tokens, hidden}) && coords_.try_resize({live_tokens, 3});
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
    dim3 const token_grid {static_cast<u32_t>(live_tokens), 1u, 1u};
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

    // The block stack: layer n reads layer n-1's output, ping-ponging two dedicated buffers; the last
    // output is a stash the backward reads.

    bf16r2_view_t layer_input = entry_output_.view();
    for (u32_t layer_index = 0; layer_index != config_.depth; ++layer_index) {
        bf16r2_tensor_arena_t &layer_output = layer_index % 2 == 0 ? activations_a_ : activations_b_;
        staged = blocks_[layer_index].forward(layer_input, layer_output.span(), segment_offsets, segment_lengths,
                                              angles_.view(), coords_.view(), stream);
        if (!staged) return staged;
        layer_input = layer_output.view();
    }
    stack_output_ = layer_input;

    // Final norm over the packed token states.
    staged = launch_kernel(reinterpret_cast<void const *>(&rmsnorm_forward_kernel), token_grid, block_shape, stream,
                           stack_output_, final_norm_scale_.view(), normalized_.span(), final_inverse_rms_.span(),
                           config_.block.rms_eps);
    if (!staged) return staged;

    // Masked path stops here — the masked head reads normalized(); the contrastive path pools + heads.
    if (vertices.data() == nullptr) return {};
    return expected_t::fail(status_t::shape_unsupported_k,
                            "the contrastive attention pool + head are not yet built (masked path only)");
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
    if (vertices_grad.data() != nullptr)
        return expected_t::fail(status_t::shape_unsupported_k,
                                "the contrastive pool + head backward are not yet built (masked path only)");
    if (hidden_grad.data() == nullptr)
        return expected_t::fail(status_t::invalid_input_k, "backward needs one of vertices_grad / hidden_grad");

    // The step's live extent, resized within each arena's reserved capacity — no address moves.

    bool const resized = token_grad_a_.try_resize({live_tokens, hidden}) &&
                         token_grad_b_.try_resize({live_tokens, hidden});
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

    // Final-norm backward: d(normalized) → d(stack output) into token_grad_b_, plus the scale grad. The
    // caller's hidden_grad may carry a wider capacity than the live extent, so it is narrowed to match.

    bool const overwrite = false;
    bf16r2_view_t const hidden_grad_live = hidden_grad(nk::range(0, live_tokens), nk::slice);
    expected_t staged = launch_kernel(reinterpret_cast<void const *>(&rmsnorm_backward_kernel), row_grid, block_shape,
                                      stream, stack_output_, final_norm_scale_.view(), final_inverse_rms_.view(),
                                      hidden_grad_live, token_grad_b_.span(), overwrite);
    if (staged)
        staged = launch_kernel(reinterpret_cast<void const *>(&rmsnorm_scale_grad_kernel), channel_grid, block_shape,
                               stream, stack_output_, final_inverse_rms_.view(), hidden_grad_live,
                               final_norm_scale_grad_.span());
    if (!staged) return staged;

    // Blocks in reverse; the flowing gradient ping-pongs the two token-gradient buffers. `coords_` still
    // holds this step's coordinates from the paired forward.
    bf16r2_tensor_arena_t *grad_in = &token_grad_b_;
    bf16r2_tensor_arena_t *grad_out = &token_grad_a_;
    bf16r2_view_t flowing = grad_in->view();
    for (u32_t layer_index = config_.depth; layer_index != 0; --layer_index) {
        staged = blocks_[layer_index - 1].backward(flowing, grad_out->span(), segment_offsets, segment_lengths,
                                                   angles_.view(), coords_.view(), stream);
        if (!staged) return staged;
        flowing = grad_out->view();
        std::swap(grad_in, grad_out);
    }
    bf16r2_view_t const entry_grad = flowing;

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

expected<encoder_tower_e4m3_t> encoder_tower_bf16_t::quantize_weights(gpu_arena_t &, gpu_stream_t &) noexcept {
    return expected<encoder_tower_e4m3_t>::fail(status_t::shape_unsupported_k,
                                                "the E4M3 inference twin is not yet built");
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

    bool const resized = entry_output_.try_resize({live_tokens, hidden}) &&
                         activations_a_.try_resize({live_tokens, hidden}) &&
                         activations_b_.try_resize({live_tokens, hidden}) && coords_.try_resize({live_tokens, 3});
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

    (void)vertices;
    return expected_t::fail(status_t::shape_unsupported_k,
                            "the E4M3 final norm + attention pool + head are not yet built");
}
