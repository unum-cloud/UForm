/**
 *  @brief Device kernels and heavy method bodies of the encoder tower declared in @ref encoder_tower.hpp .
 *  @file src/core/encoder_tower.cu
 *  @author Ash Vardanian
 *  @date July 4, 2026
 */
#include <algorithm> // `std::max` for the widest media entry, `std::min` for the pool pack

#include <cub/block/block_scan.cuh>         // `cub::BlockScan` — the token head's compact offsets
#include <cub/device/device_radix_sort.cuh> // the deterministic embedding-gradient scatter
#include <cub/thread/thread_search.cuh>     // `cub::UpperBound` — a compact row's segment

#include "core/encoder_tower.hpp"

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
    gather_bytes_segment(tokens, table, offset, offset, length, destination, threadIdx.x, blockDim.x);
}

/**
 *  @brief Stage the embedding-scatter sort keys from one @b text segment: @c keys[offset+k] =
 *      tokens[offset+k] .
 *
 *  Non-text segments exit, leaving their cells at the sentinel the caller pre-filled so they sort into one
 *  skipped run — the packed grid interleaves modalities, so the text token ids are not a leading slice
 *  any more.
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

/** Fill the position payload 0..tokens-1 for the embedding sort.
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
 *  @brief Fold the pool's per-segment query-gradient partials into the one @b [hidden] gradient — a
 *      fixed-order serial sum per channel, deterministic by construction.
 *
 *  __grid_size__ grid-stride over @c hidden channels; no dynamic shared memory.
 */
static __global__ void tower_pool_query_grad_kernel_(f32r2_view_t query_partials, bf16r1_span_t query_grad) {
    std::size_t const segments = query_partials.extent(0), hidden = query_partials.extent(1);
    std::size_t const stride = static_cast<std::size_t>(gridDim.x) * blockDim.x;
    for (std::size_t channel = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x; channel < hidden;
         channel += stride) {
        f32_t total = 0.0f;
        for (std::size_t segment = 0; segment != segments; ++segment) total += query_partials[segment, channel];
        query_grad[channel] = __float2bfloat16_rn(total);
    }
}

/**
 *  @brief Normalize one @b pixel segment's cells in place into the projection's BFloat16 fan-in.
 *
 *  CTA @c s owns segment @c s , exits unless @p target matches its @ref modality_t , and otherwise walks its
 *  @c length cells at flat grid rows @c [offset,offset+length) — a cell's raw pixels live at the same
 *  flat position as its state, so the peel is a zero-copy slice — writing each channel as
 *  @c (byte/255 − mean[channel])·inverse_std[channel] over the rank-3 peel @c [rows,spatial,channels] ,
 *  @c channels innermost so @p mean / @p inverse_std index by the channel coordinate directly. The scratch
 *  is pre-zeroed, so every cell this pass skips stays zero and the projection's non-modality rows vanish.
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
    stage_patches_segment(content, offset, offset, length, mean, inverse_std, patches, threadIdx.x, blockDim.x,
                          tower_scalars_t {});
}

/**
 *  @brief Stage one @b audio segment's raw waveform frames in place into the projection's BFloat16 fan-in.
 *
 *  Like @ref tower_normalize_segment_kernel_ but raw samples carry no per-channel statistics to subtract, so
 *  the cast is the whole staging ( @c f32 → BFloat16 ) over the flat @c [rows,scalars] peel (a frame is
 *  @c chunk_samples raw samples); non-audio segments and the pre-zeroed remainder stay zero.
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
    stage_frames_segment(content, offset, offset, length, patches, threadIdx.x, blockDim.x, tower_scalars_t {});
}

/** The head's biasless fc1 → SiLU → fc2 over the normalized pooled rows into @p vertices — raw, never
 *  L2-normalized here, since the loss owns the cosine geometry. Each product lands in the Float32 @p products
 *  and narrows into its BFloat16 destination. Shared verbatim by both towers' forwards. */
static expected_t head_forward_(bf16r2_view_t normalized, b8r1_tensor_arena_t const &packed_fc1,
                                b8r1_tensor_arena_t const &packed_fc2, bf16r2_span_t hidden_state,
                                bf16r2_span_t activated, f32r2_span_t products, bf16r2_span_t vertices,
                                nk_capability_t capabilities, gpu_stream_t &stream) noexcept {
    std::size_t const cells = normalized.extent(0) * normalized.extent(1);
    dim3 const cell_grid {static_cast<u32_t>(divide_round_up(cells, kernel_block_size_default_k)), 1u, 1u};
    expected_t staged = multiply_bf16(normalized, packed_fc1.view(), products, capabilities, stream);
    if (staged)
        staged = numkong_result(nk_cast_best(products.data(), nk_f32_k, hidden_state.data(), nk_bf16_k, cells,
                                             capabilities, stream.numkong()),
                                "head fc1 narrowing");
    if (staged)
        staged = launch_kernel(&tower_silu_kernel_, cell_grid, dim3 {kernel_block_size_default_k, 1u, 1u}, stream,
                               hidden_state.flatten<1>(), activated.flatten<1>());
    if (staged) staged = multiply_bf16(activated, packed_fc2.view(), products, capabilities, stream);
    if (staged)
        staged = numkong_result(
            nk_cast_best(products.data(), nk_f32_k, vertices.data(), nk_bf16_k, cells, capabilities, stream.numkong()),
            "head fc2 narrowing");
    return staged;
}

/** Upload one modality's per-channel pixel mean and inverse-std to the device, allocating the two
 *  @b [channels] arenas off @p f32_alloc — the shared upload @ref encoder_tower_bf16_t::make runs for
 *  image and video (audio frames are raw waveform with no per-channel statistics, so they never call this). */
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
    if (CUresult result = cuMemcpyHtoD(device_address(mean_arena.data()), mean, channels * sizeof(f32_t));
        result != CUDA_SUCCESS)
        return cuda_failure(result, "pixel mean upload failed");
    if (CUresult result = cuMemcpyHtoD(device_address(inverse_std_arena.data()), inverse_std, channels * sizeof(f32_t));
        result != CUDA_SUCCESS)
        return cuda_failure(result, "pixel inverse-std upload failed");
    return {};
}

/**
 *  @brief Stage one pixel modality's raw cells into per-channel-normalized BFloat16 patch vectors over the
 *      caller's @b [rows,scalars] @p patches .
 *
 *  Zero the whole span, then run @ref tower_normalize_segment_kernel_ so only @p target 's cells are
 *  written ( @c (byte/255 − mean)·inverse_std , @c channels innermost) and every non-modality row stays
 *  zero. The one normalize launch @ref project_media_entries folds behind its projection and
 *  @ref encoder_tower_bf16_t::patchify_media_targets hands out as the masked-autoencoder reconstruction
 *  target — the two differ only in where @p patches lands and whether a projection follows, never in the
 *  staging itself.
 */
static expected_t stage_pixel_patches(nk::tensor_view<pixel_t, 3> raw, modality_t target, f32r1_view_t mean,
                                      f32r1_view_t inverse_std, u32r1_view_t offsets, u32r1_view_t lengths,
                                      modality_view_t modality, dim3 segment_grid, bf16r2_span_t patches,
                                      gpu_stream_t &stream) noexcept {
    std::size_t const rows = patches.extent(0);
    u32_t const spatial = static_cast<u32_t>(raw.extent(1));
    u32_t const channels = static_cast<u32_t>(raw.extent(2));
    dim3 const block_shape {kernel_block_size_default_k, 1u, 1u};
    if (expected_t zeroed = zero_async(patches, stream); !zeroed) return zeroed;
    return launch_kernel(&tower_normalize_segment_kernel_, segment_grid, block_shape, stream, raw, offsets, lengths,
                         modality, target, mean, inverse_std, patches.reshape<3>({rows, spatial, channels}));
}

/**
 *  @brief Project every present media modality's raw signal into the Float32 entry state.
 *
 *  For each present modality zero @p patch_scratch , stage only that modality's cells in place (image / video
 *  pixels normalized by their per-channel mean / inverse-std, audio raw frames cast as-is — a cell's content
 *  shares the flat grid position of its state), multiply by the packed projection, and add the product into
 *  @p entry_state , so the zero non-modality rows leave every other cell untouched — no scatter, no atomics.
 *  A modality is present iff its projection is packed. Shared verbatim by both towers' forwards; the text
 *  entry (a kernel) is the caller's.
 */
static expected_t project_media_entries(multimodal_batch_on_gpu_t const &batch, bf16r2_tensor_arena_t &patch_scratch,
                                        f32r1_tensor_arena_t &image_mean, f32r1_tensor_arena_t &image_inverse_std,
                                        b8r1_tensor_arena_t &packed_image, f32r1_tensor_arena_t &video_mean,
                                        f32r1_tensor_arena_t &video_inverse_std, b8r1_tensor_arena_t &packed_video,
                                        b8r1_tensor_arena_t &packed_audio, f32r2_span_t entry_state,
                                        f32r1_span_t products, nk_capability_t capabilities,
                                        gpu_stream_t &stream) noexcept {
    std::size_t const columns = batch.tokens.extent(1);
    std::size_t const live_tokens = static_cast<std::size_t>(batch.tokens.extent(0)) * columns;
    std::size_t const hidden = entry_state.extent(1);
    dim3 const block_shape {kernel_block_size_default_k, 1u, 1u};
    dim3 const segment_grid {static_cast<u32_t>(batch.live_segments()), 1u, 1u};
    u32r1_view_t const segment_offsets = batch.segment_offsets.view();
    u32r1_view_t const segment_lengths = batch.segment_lengths.view();
    modality_view_t const segment_modality = batch.segment_modality.view();
    f32r2_span_t const projected {products.data(), live_tokens, hidden};

    auto add_projection = [&](bf16r2_view_t patches, b8r1_tensor_arena_t const &packed) noexcept -> expected_t {
        expected_t staged = multiply_bf16(patches, packed.view(), projected, capabilities, stream);
        if (staged)
            staged = numkong_result(nk_each_sum_f32_best(entry_state.data(), projected.data(), live_tokens * hidden,
                                                         entry_state.data(), capabilities, stream.numkong()),
                                    "entry projection");
        return staged;
    };
    // One pixel modality's forward: zero the scratch, normalize its cells in place, add the projection.
    auto project_pixels = [&](nk::tensor_view<pixel_t, 3> raw, modality_t target, f32r1_view_t mean,
                              f32r1_view_t inverse_std, b8r1_tensor_arena_t const &packed) noexcept -> expected_t {
        u32_t const scalars = static_cast<u32_t>(raw.extent(1)) * static_cast<u32_t>(raw.extent(2));
        bf16r2_span_t const patches {patch_scratch.data(), live_tokens, scalars};
        expected_t staged = stage_pixel_patches(raw, target, mean, inverse_std, segment_offsets, segment_lengths,
                                                segment_modality, segment_grid, patches, stream);
        if (!staged) return staged;
        return add_projection(patches, packed);
    };

    expected_t staged {};
    if (packed_image.data() != nullptr && batch.images.data() != nullptr) {
        std::size_t const spatial = static_cast<std::size_t>(batch.images.extent(2)) * batch.images.extent(3);
        staged = project_pixels(batch.images.view().reshape<3>({live_tokens, spatial, batch.images.extent(4)}),
                                modality_t::image_k, image_mean.view(), image_inverse_std.view(), packed_image);
        if (!staged) return staged;
    }
    if (packed_video.data() != nullptr && batch.videos.data() != nullptr) {
        std::size_t const spatial = static_cast<std::size_t>(batch.videos.extent(2)) * batch.videos.extent(3) *
                                    batch.videos.extent(4);
        staged = project_pixels(batch.videos.view().reshape<3>({live_tokens, spatial, batch.videos.extent(5)}),
                                modality_t::video_k, video_mean.view(), video_inverse_std.view(), packed_video);
        if (!staged) return staged;
    }
    if (packed_audio.data() != nullptr && batch.audio_samples.data() != nullptr) {
        u32_t const scalars = static_cast<u32_t>(batch.audio_samples.extent(2));
        bf16r2_span_t const patches {patch_scratch.data(), live_tokens, scalars};
        if (expected_t zeroed = zero_async(patches, stream); !zeroed) return zeroed;
        nk::tensor_view<f32_t, 2> const raw = batch.audio_samples.view().reshape<2>({live_tokens, scalars});
        staged = launch_kernel(&tower_audio_segment_kernel_, segment_grid, block_shape, stream, raw, segment_offsets,
                               segment_lengths, segment_modality, patches);
        if (staged) staged = add_projection(patches, packed_audio);
    }
    return staged;
}

/**
 *  @brief Fold the packed entry gradient against each present media modality's re-staged patches into that
 *      modality's projection weight gradient.
 *
 *  Re-stage exactly as @ref project_media_entries (segment-driven, so non-modality rows stay zero), then
 *  take the weight gradient through the blocks' shared @p scratch — the backward twin of
 *  @ref project_media_entries . A modality is present iff its gradient is allocated.
 */
static expected_t project_media_weight_grads(multimodal_batch_on_gpu_t const &batch,
                                             bf16r2_tensor_arena_t &patch_scratch, bf16r2_view_t entry_grad,
                                             f32r1_tensor_arena_t &image_mean, f32r1_tensor_arena_t &image_inverse_std,
                                             bf16r2_tensor_arena_t &image_projection_grad,
                                             f32r1_tensor_arena_t &video_mean, f32r1_tensor_arena_t &video_inverse_std,
                                             bf16r2_tensor_arena_t &video_projection_grad,
                                             bf16r2_tensor_arena_t &audio_projection_grad,
                                             transformer_scratch_t &scratch, nk_capability_t capabilities,
                                             gpu_stream_t &stream) noexcept {
    std::size_t const columns = batch.tokens.extent(1);
    std::size_t const live_tokens = static_cast<std::size_t>(batch.tokens.extent(0)) * columns;
    dim3 const block_shape {kernel_block_size_default_k, 1u, 1u};
    dim3 const segment_grid {static_cast<u32_t>(batch.live_segments()), 1u, 1u};
    u32r1_view_t const segment_offsets = batch.segment_offsets.view();
    u32r1_view_t const segment_lengths = batch.segment_lengths.view();
    modality_view_t const segment_modality = batch.segment_modality.view();

    auto weight_gradient = [&](bf16r2_view_t patches, bf16r2_span_t projection_grad) noexcept {
        return weight_gradient_bf16(entry_grad, patches, projection_grad, scratch.transposed_grad.span(),
                                    scratch.transposed_input.span(), scratch.packed_input.span(),
                                    scratch.weight_grad_products.span(), scratch.weight_gradient_workspace.span(),
                                    capabilities, stream);
    };
    // One pixel modality's weight grad: re-stage its cells, then fold the whole entry gradient against them.
    auto project_pixels_grad = [&](nk::tensor_view<pixel_t, 3> raw, modality_t target, f32r1_view_t mean,
                                   f32r1_view_t inverse_std, bf16r2_span_t projection_grad) noexcept -> expected_t {
        u32_t const scalars = static_cast<u32_t>(raw.extent(1)) * static_cast<u32_t>(raw.extent(2));
        bf16r2_span_t const patches {patch_scratch.data(), live_tokens, scalars};
        expected_t staged = stage_pixel_patches(raw, target, mean, inverse_std, segment_offsets, segment_lengths,
                                                segment_modality, segment_grid, patches, stream);
        if (!staged) return staged;
        return weight_gradient(patches, projection_grad);
    };

    expected_t staged {};
    if (image_projection_grad.data() != nullptr && batch.images.data() != nullptr) {
        std::size_t const spatial = static_cast<std::size_t>(batch.images.extent(2)) * batch.images.extent(3);
        staged = project_pixels_grad(batch.images.view().reshape<3>({live_tokens, spatial, batch.images.extent(4)}),
                                     modality_t::image_k, image_mean.view(), image_inverse_std.view(),
                                     image_projection_grad.span());
        if (!staged) return staged;
    }
    if (video_projection_grad.data() != nullptr && batch.videos.data() != nullptr) {
        std::size_t const spatial = static_cast<std::size_t>(batch.videos.extent(2)) * batch.videos.extent(3) *
                                    batch.videos.extent(4);
        staged = project_pixels_grad(batch.videos.view().reshape<3>({live_tokens, spatial, batch.videos.extent(5)}),
                                     modality_t::video_k, video_mean.view(), video_inverse_std.view(),
                                     video_projection_grad.span());
        if (!staged) return staged;
    }
    if (audio_projection_grad.data() != nullptr && batch.audio_samples.data() != nullptr) {
        u32_t const scalars = static_cast<u32_t>(batch.audio_samples.extent(2));
        bf16r2_span_t const patches {patch_scratch.data(), live_tokens, scalars};
        if (expected_t zeroed = zero_async(patches, stream); !zeroed) return zeroed;
        nk::tensor_view<f32_t, 2> const raw = batch.audio_samples.view().reshape<2>({live_tokens, scalars});
        staged = launch_kernel(&tower_audio_segment_kernel_, segment_grid, block_shape, stream, raw, segment_offsets,
                               segment_lengths, segment_modality, patches);
        if (staged) staged = weight_gradient(patches, audio_projection_grad.span());
    }
    return staged;
}

expected<encoder_tower_bf16_t> encoder_tower_bf16_t::make(gpu_arena_t &arena, encoder_tower_config_t config,
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

    // Forward stashes and the head norm's inverse-rms stash.
    tower.entry_output_ = bf16r2_tensor_arena_t::uninitialized({tokens, hidden}, bf16_alloc).value;
    tower.normalized_ = bf16r2_tensor_arena_t::uninitialized({tokens, hidden}, bf16_alloc).value;
    tower.pooled_ = bf16r2_tensor_arena_t::uninitialized({segments, hidden}, bf16_alloc).value;
    tower.head_normalized_ = bf16r2_tensor_arena_t::uninitialized({segments, hidden}, bf16_alloc).value;
    tower.head_hidden_ = bf16r2_tensor_arena_t::uninitialized({segments, hidden}, bf16_alloc).value;
    tower.head_activated_ = bf16r2_tensor_arena_t::uninitialized({segments, hidden}, bf16_alloc).value;
    tower.head_inverse_rms_ = f32r1_tensor_arena_t::uninitialized({segments}, f32_alloc).value;
    tower.coords_ = i16r2_tensor_arena_t::uninitialized({tokens, 3}, i16_alloc).value;
    if (!tower.entry_output_ || !tower.normalized_ || !tower.pooled_ || !tower.head_normalized_ ||
        !tower.head_hidden_ || !tower.head_activated_ || !tower.head_inverse_rms_ || !tower.coords_)
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

    expected<tower_pool_t> pool = tower_pool_t::make(arena, config.block);
    if (!pool) return result_t::fail(std::move(pool));
    tower.pool_ = std::move(pool.value_);
    if (config.token_width != 0) {
        expected<tower_token_head_t> token_head = tower_token_head_t::make(arena, config.block, config.token_width);
        if (!token_head) return result_t::fail(std::move(token_head));
        tower.token_head_ = std::move(token_head.value_);
    }

    // Media entries: each present modality patch-cuts its raw signal and projects it through a NumKong pack.
    // A modality is present iff its geometry is fully specified — zero extents leave its projection, pack, and
    // scratch untouched. Fan-ins are per-modality: image is ph·pw·image_channels, video folds
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

    auto allocate_pack = [&](b8r1_tensor_arena_t &packed, std::size_t depth) noexcept -> expected_t {
        expected<b8r1_tensor_arena_t> allocated = allocate_pack_bf16(hidden, depth, tower.capabilities_, arena);
        if (!allocated) return expected_t::fail(std::move(allocated));
        packed = std::move(allocated.value_);
        return {};
    };
    expected_t packs = allocate_pack(tower.packed_head_fc1_, hidden);
    if (packs) packs = allocate_pack(tower.packed_head_fc2_, hidden);
    if (packs) packs = allocate_pack(tower.packed_head_fc1_transposed_, hidden);
    if (packs) packs = allocate_pack(tower.packed_head_fc2_transposed_, hidden);
    if (packs && image_present) // raw pixels are never differentiated, so no transposed pack
        packs = allocate_pack(tower.packed_image_projection_, image_patch_scalars);
    if (packs && video_present) packs = allocate_pack(tower.packed_video_projection_, video_tube_scalars);
    if (packs && audio_present) packs = allocate_pack(tower.packed_audio_projection_, audio_frame_scalars);
    if (!packs) return result_t::fail(std::move(packs));

    u32_t const max_entry_scalars = std::max({image_patch_scalars, video_tube_scalars, audio_frame_scalars});
    if (max_entry_scalars != 0) {
        tower.patch_scratch_ =
            bf16r2_tensor_arena_t::uninitialized({max_tokens32, max_entry_scalars}, bf16_alloc).value;
        if (!tower.patch_scratch_)
            return result_t::fail(status_t::bad_alloc_k, "patch scratch allocation failed (arena_used={})",
                                  arena.used());
    }

    // The block stack, over one scratch wide enough for the entries' and the head's weight gradients too.
    if (tower.blocks_.resize(config.depth) != status_t::success_k)
        return result_t::fail(status_t::bad_alloc_k, "block table allocation failed ({})", config.depth);
    u32_t const entry_scalars[3] = {image_patch_scalars, video_tube_scalars, audio_frame_scalars};
    expected<transformer_scratch_t> scratch = transformer_scratch_t::make(arena, config.block, entry_scalars,
                                                                          config.token_width);
    if (!scratch) return result_t::fail(std::move(scratch));
    tower.scratch_ = std::make_unique<transformer_scratch_t>(std::move(scratch.value_));
    for (u32_t layer_index = 0; layer_index != config.depth; ++layer_index) {
        expected<transformer_block_bf16_t> block = transformer_block_bf16_t::make(arena, *tower.scratch_, config.block);
        if (!block) return result_t::fail(std::move(block));
        tower.blocks_[layer_index] = std::move(block.value_);
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

    // Live tokens of all segments together never exceed the batch, however they split.
    std::size_t const tokens = std::min<std::size_t>(block.max_tokens, segments * block.max_segment_tokens);
    nk_size_t packed_bytes = 0;
    if (nk_status_t const status = nk_attention_pack_size_bf16_best(heads, transformer_head_dim_k, tokens, segments,
                                                                    pool.capabilities_, &packed_bytes);
        status != nk_success_k)
        return result_t::fail(status_t::cuda_error_k, "pool pack sizing failed: {}", nk_status_name(status));
    array<u32_t> host;
    if (host.resize(segments + 1u) != status_t::success_k)
        return result_t::fail(status_t::bad_alloc_k, "pool staging allocation failed ({} segments)", segments);
    for (std::size_t segment = 0; segment <= segments; ++segment) host[segment] = static_cast<u32_t>(segment);

    pool.query_offsets_ = u32r1_tensor_arena_t::uninitialized({segments + 1u}, gpu_arena_alloc<u32_t> {arena}).value;
    pool.packed_ = b8r1_tensor_arena_t::uninitialized({static_cast<std::size_t>(packed_bytes)},
                                                      gpu_arena_alloc<b8_t> {arena})
                       .value;
    gpu_arena_alloc<f32_t> const f32_alloc {arena};
    pool.pooled_ = f32r2_tensor_arena_t::uninitialized({segments, block.hidden}, f32_alloc).value;
    pool.log_sum_exp_ = f32r2_tensor_arena_t::uninitialized({segments, heads}, f32_alloc).value;
    pool.pooled_grad_ = f32r2_tensor_arena_t::uninitialized({segments, block.hidden}, f32_alloc).value;
    pool.query_partials_ = f32r2_tensor_arena_t::uninitialized({segments, block.hidden}, f32_alloc).value;
    if (!pool.query_offsets_ || !pool.packed_ || !pool.pooled_ || !pool.log_sum_exp_ || !pool.pooled_grad_ ||
        !pool.query_partials_)
        return result_t::fail(status_t::bad_alloc_k, "pool allocation failed (arena_used={})", arena.used());
    if (CUresult result = cuMemcpyHtoD(device_address(pool.query_offsets_.data()), host.data(),
                                       (segments + 1u) * sizeof(u32_t));
        result != CUDA_SUCCESS)
        return cuda_failure<tower_pool_t>(result, "pool query-offset upload failed");
    return result_t {std::move(pool)};
}

expected_t tower_pool_t::forward(bf16r1_view_t query, bf16r2_view_t states, u32r1_view_t segment_offsets,
                                 u32r1_view_t segment_lengths, bf16r2_span_t pooled, gpu_stream_t &stream) noexcept {
    std::size_t const segments = pooled_.extent(0), hidden = pooled_.extent(1);
    std::size_t const heads = hidden / transformer_head_dim_k;
    nk_bf16_t const *const states_data = numkong_scalars(states.data());
    nk_status_t status = nk_attention_pack_bf16_best(
        heads, transformer_head_dim_k, segment_offsets.data(), segment_lengths.data(), segments, states_data,
        hidden * sizeof(bf16_t), states_data, hidden * sizeof(bf16_t), packed_.data(), 0, segments * heads,
        capabilities_, stream.numkong());
    if (status == nk_success_k)
        status = nk_attention_packed_bf16_best(heads, heads, transformer_head_dim_k, query_offsets_.data(), segments,
                                               transformer_attention_scale(), NUMKONG_SIZE_MAX, NUMKONG_SIZE_MAX,
                                               numkong_scalars(query.data()), 0, packed_.data(), pooled_.data(),
                                               hidden * sizeof(f32_t), log_sum_exp_.data(), 0, NUMKONG_SIZE_MAX,
                                               NUMKONG_NULL, 0, capabilities_, stream.numkong());
    if (status == nk_success_k)
        status = nk_cast_best(pooled_.data(), nk_f32_k, pooled.data(), nk_bf16_k, segments * hidden, capabilities_,
                              stream.numkong());
    if (status != nk_success_k)
        return expected_t::fail(status_t::cuda_error_k, "attention pool failed: {}", nk_status_name(status));
    return {};
}

expected_t tower_pool_t::backward(bf16r1_view_t query, bf16r2_view_t pooled_grad, f32r2_span_t key_grad,
                                  f32r2_span_t value_grad, bf16r1_span_t query_grad, gpu_stream_t &stream) noexcept {
    std::size_t const segments = pooled_.extent(0), hidden = pooled_.extent(1);
    std::size_t const heads = hidden / transformer_head_dim_k;
    std::size_t const tokens = key_grad.extent(0);
    // NumKong never writes rows past a segment's live keys, nor the query rows of key-less PAD segments.
    for (f32_t *const gradient : {key_grad.data(), value_grad.data()})
        if (expected_t zeroed = zero_async(f32r1_span_t {gradient, tokens * hidden}, stream); !zeroed) return zeroed;
    if (expected_t zeroed = zero_async(f32r1_span_t {query_partials_.data(), segments * hidden}, stream); !zeroed)
        return zeroed;
    nk_status_t status = nk_cast_best(pooled_grad.data(), nk_bf16_k, pooled_grad_.data(), nk_f32_k, segments * hidden,
                                      capabilities_, stream.numkong());
    if (status == nk_success_k)
        status = nk_attention_packed_gradients_bf16_best(
            heads, heads, transformer_head_dim_k, query_offsets_.data(), segments, transformer_attention_scale(),
            NUMKONG_SIZE_MAX, NUMKONG_SIZE_MAX, numkong_scalars(query.data()), 0, packed_.data(), pooled_.data(),
            pooled_grad_.data(), hidden * sizeof(f32_t), log_sum_exp_.data(), query_partials_.data(),
            hidden * sizeof(f32_t), key_grad.data(), value_grad.data(), key_grad.stride_bytes(0), 0, NUMKONG_SIZE_MAX,
            NUMKONG_NULL, 0, capabilities_, stream.numkong());
    if (status == nk_success_k)
        status = nk_each_sum_f32_best(key_grad.data(), value_grad.data(), tokens * hidden, key_grad.data(),
                                      capabilities_, stream.numkong());
    if (status != nk_success_k)
        return expected_t::fail(status_t::cuda_error_k, "attention pool backward failed: {}", nk_status_name(status));
    dim3 const channel_grid {static_cast<u32_t>(divide_round_up(hidden, kernel_block_size_default_k)), 1u, 1u};
    return launch_kernel(&tower_pool_query_grad_kernel_, channel_grid, dim3 {kernel_block_size_default_k, 1u, 1u},
                         stream, query_partials_.view(), query_grad);
}

/** The rows segment @p segment lends the token head: its live tokens past the prefix, or none when
 *  it plays no late-interaction side. */
[[nodiscard]] static __device__ inline u32_t tower_token_span_(multimodal_batch_view_t const &batch,
                                                               u32_t segment) noexcept {
    u32_t const length = batch.segment_lengths[segment], prefix = batch.segment_prefix_length[segment];
    bool const sided = batch.segment_side[segment] != late_interaction_side_t::none_k;
    return sided && length > prefix ? length - prefix : 0u;
}

/** The flat state row compact row @p row of the token head reads, or @c ~0u past the live total. */
[[nodiscard]] static __device__ inline u32_t tower_token_source_(multimodal_batch_view_t const &batch,
                                                                 u32r1_view_t token_offsets, u32_t row) noexcept {
    u32_t const segments = static_cast<u32_t>(token_offsets.extent(0) - 1u);
    if (row >= token_offsets[segments]) return ~0u;
    u32_t const segment = static_cast<u32_t>(cub::UpperBound(token_offsets.data(), segments + 1u, row)) - 1u;
    return batch.segment_offsets[segment] + batch.segment_prefix_length[segment] + (row - token_offsets[segment]);
}

/**
 *  @brief The token head's compact offsets: the exclusive scan of every segment's span, in segment
 *      order, with the live total last.
 *
 *  __grid_size__ @c [1,1,1] ; one CTA scans the segments in chunks of its width; no dynamic shared
 *  memory.
 */
static __global__ void tower_token_offsets_kernel_(multimodal_batch_view_t batch, u32r1_span_t token_offsets) {
    using block_scan_t = cub::BlockScan<u32_t, kernel_block_size_default_k>;
    __shared__ typename block_scan_t::TempStorage scan_storage;
    __shared__ u32_t carried;
    u32_t const segments = static_cast<u32_t>(token_offsets.extent(0) - 1u);
    if (threadIdx.x == 0u) carried = 0u;
    __syncthreads();
    for (u32_t chunk = 0; chunk < segments; chunk += blockDim.x) {
        u32_t const segment = chunk + threadIdx.x;
        u32_t const span = segment < segments ? tower_token_span_(batch, segment) : 0u;
        u32_t exclusive = 0, chunk_total = 0;
        block_scan_t(scan_storage).ExclusiveSum(span, exclusive, chunk_total);
        if (segment < segments) token_offsets[segment] = carried + exclusive;
        __syncthreads();
        if (threadIdx.x == 0u) carried += chunk_total;
        __syncthreads();
    }
    if (threadIdx.x == 0u) token_offsets[segments] = carried;
}

/**
 *  @brief Gather every compact row of the token head from its state row, zeroing the rows past the
 *      live total.
 *
 *  __grid_size__ @c [rows,1,1] ; one CTA per compact row; threads stripe the channels; no dynamic
 *  shared memory.
 */
static __global__ void tower_token_gather_kernel_(bf16r2_view_t states, multimodal_batch_view_t batch,
                                                  u32r1_view_t token_offsets, bf16r2_span_t gathered) {
    u32_t const row = blockIdx.x;
    u32_t const source = tower_token_source_(batch, token_offsets, row);
    for (std::size_t channel = threadIdx.x; channel < gathered.extent(1); channel += blockDim.x)
        gathered[row, channel] = source != ~0u ? states[source, channel] : __float2bfloat16(0.0f);
}

/**
 *  @brief Add every live compact row's gradient onto its state row; each state row receives at most
 *      one, so the fold is order-free.
 *
 *  __grid_size__ @c [rows,1,1] ; one CTA per compact row; threads stripe the channels; no dynamic
 *  shared memory.
 */
static __global__ void tower_token_scatter_kernel_(bf16r2_view_t gathered_grad, multimodal_batch_view_t batch,
                                                   u32r1_view_t token_offsets, f32r2_span_t states_grad) {
    u32_t const row = blockIdx.x;
    u32_t const source = tower_token_source_(batch, token_offsets, row);
    if (source == ~0u) return;
    for (std::size_t channel = threadIdx.x; channel < gathered_grad.extent(1); channel += blockDim.x)
        states_grad[source, channel] += __bfloat162float(gathered_grad[row, channel]);
}

/**
 *  @brief Per-row L2 normalization @c v = p/max(‖p‖,√ε) , stashing the inverse norm for the
 *      backward.
 *
 *  __grid_size__ @c [rows,1,1] ; one CTA per row; threads stripe the width; no dynamic shared
 *  memory.
 */
static __global__ void tower_token_normalize_kernel_(f32r2_view_t projected, f32r1_span_t inverse_norms,
                                                     bf16r2_span_t vectors, f32_t epsilon) {
    u32_t const row = blockIdx.x;
    std::size_t const width = projected.extent(1);
    f32_t thread_squares = 0.0f;
    for (std::size_t channel = threadIdx.x; channel < width; channel += blockDim.x)
        thread_squares += projected[row, channel] * projected[row, channel];
    f32_t const inverse_norm = rsqrtf(
        std::fmax(block_reduce_broadcast<kernel_block_size_default_k>(thread_squares), epsilon));
    for (std::size_t channel = threadIdx.x; channel < width; channel += blockDim.x)
        vectors[row, channel] = __float2bfloat16(projected[row, channel] * inverse_norm);
    if (threadIdx.x == 0u) inverse_norms[row] = inverse_norm;
}

/**
 *  @brief The L2 normalization's backward @c dp = (dv − (u·dv)·u)/‖p‖ with @c u = p/‖p‖ .
 *
 *  __grid_size__ @c [rows,1,1] ; one CTA per row; threads stripe the width; no dynamic shared
 *  memory.
 */
static __global__ void tower_token_normalize_backward_kernel_(f32r2_view_t projected, f32r1_view_t inverse_norms,
                                                              bf16r2_view_t vectors_grad,
                                                              bf16r2_span_t projected_grad) {
    u32_t const row = blockIdx.x;
    std::size_t const width = projected.extent(1);
    f32_t const inverse_norm = inverse_norms[row];
    f32_t thread_dot = 0.0f;
    for (std::size_t channel = threadIdx.x; channel < width; channel += blockDim.x)
        thread_dot += projected[row, channel] * inverse_norm * __bfloat162float(vectors_grad[row, channel]);
    f32_t const dot = block_reduce_broadcast<kernel_block_size_default_k>(thread_dot);
    for (std::size_t channel = threadIdx.x; channel < width; channel += blockDim.x) {
        f32_t const unit = projected[row, channel] * inverse_norm;
        projected_grad[row, channel] = __float2bfloat16((__bfloat162float(vectors_grad[row, channel]) - dot * unit) *
                                                        inverse_norm);
    }
}

expected<tower_token_head_t> tower_token_head_t::make(gpu_arena_t &arena, transformer_block_config_t const &block,
                                                      u32_t width) noexcept {
    using result_t = expected<tower_token_head_t>;
    if (width == 0 || width % 8u != 0u)
        return result_t::fail(status_t::shape_unsupported_k, "the token width must be a positive multiple of 8 ({})",
                              width);
    tower_token_head_t head;
    head.epsilon_ = block.rms_eps;
    if (nk_status_t const status = nk_cuda_capabilities_enabled(static_cast<nk_size_t>(arena.device()),
                                                                &head.capabilities_);
        status != nk_success_k)
        return result_t::fail(status_t::cuda_error_k, "NumKong cannot query device {}: {}", arena.device(),
                              nk_status_name(status));
    std::size_t const hidden = block.hidden, tokens = block.max_tokens;
    gpu_arena_alloc<bf16_t> bf16_alloc {arena};
    gpu_arena_alloc<f32_t> f32_alloc {arena};
    head.norm_scale_ = bf16r1_tensor_arena_t::uninitialized({hidden}, bf16_alloc).value;
    head.norm_scale_grad_ = bf16r1_tensor_arena_t::uninitialized({hidden}, bf16_alloc).value;
    head.projection_ = bf16r2_tensor_arena_t::uninitialized({width, hidden}, bf16_alloc).value;
    head.projection_grad_ = bf16r2_tensor_arena_t::uninitialized({width, hidden}, bf16_alloc).value;
    head.token_offsets_ =
        u32r1_tensor_arena_t::uninitialized({block.max_segments + 1u}, gpu_arena_alloc<u32_t> {arena}).value;
    head.gathered_ = bf16r2_tensor_arena_t::uninitialized({tokens, hidden}, bf16_alloc).value;
    head.normalized_ = bf16r2_tensor_arena_t::uninitialized({tokens, hidden}, bf16_alloc).value;
    head.gathered_grad_ = bf16r2_tensor_arena_t::uninitialized({tokens, hidden}, bf16_alloc).value;
    head.inverse_rms_ = f32r1_tensor_arena_t::uninitialized({tokens}, f32_alloc).value;
    head.inverse_norms_ = f32r1_tensor_arena_t::uninitialized({tokens}, f32_alloc).value;
    head.projected_ = f32r2_tensor_arena_t::uninitialized({tokens, width}, f32_alloc).value;
    head.vectors_ = bf16r2_tensor_arena_t::uninitialized({tokens, width}, bf16_alloc).value;
    head.projected_grad_ = bf16r2_tensor_arena_t::uninitialized({tokens, width}, bf16_alloc).value;
    if (!head.norm_scale_ || !head.norm_scale_grad_ || !head.projection_ || !head.projection_grad_ ||
        !head.token_offsets_ || !head.gathered_ || !head.normalized_ || !head.gathered_grad_ || !head.inverse_rms_ ||
        !head.inverse_norms_ || !head.projected_ || !head.vectors_ || !head.projected_grad_)
        return result_t::fail(status_t::bad_alloc_k, "token head allocation failed (arena_used={})", arena.used());
    expected<b8r1_tensor_arena_t> packed = allocate_pack_bf16(width, hidden, head.capabilities_, arena);
    if (!packed) return result_t::fail(std::move(packed));
    head.packed_projection_ = std::move(packed.value_);
    expected<b8r1_tensor_arena_t> packed_transposed = allocate_pack_bf16(hidden, width, head.capabilities_, arena);
    if (!packed_transposed) return result_t::fail(std::move(packed_transposed));
    head.packed_projection_transposed_ = std::move(packed_transposed.value_);
    return result_t {std::move(head)};
}

expected_t tower_token_head_t::forward(bf16r2_view_t states, multimodal_batch_view_t const &batch,
                                       gpu_stream_t &stream) noexcept {
    std::size_t const tokens = gathered_.extent(0);
    dim3 const block_shape {kernel_block_size_default_k, 1u, 1u};
    dim3 const row_grid {static_cast<u32_t>(tokens), 1u, 1u};
    expected_t staged = launch_kernel(&tower_token_offsets_kernel_, dim3 {1u, 1u, 1u}, block_shape, stream, batch,
                                      token_offsets_.span());
    if (staged)
        staged = launch_kernel(&tower_token_gather_kernel_, row_grid, block_shape, stream, states, batch,
                               token_offsets_.view(), gathered_.span());
    if (staged)
        staged = launch_kernel(&rmsnorm_forward_kernel, row_grid, block_shape, stream, gathered_.view(),
                               norm_scale_.view(), normalized_.span(), inverse_rms_.span(), epsilon_);
    if (staged)
        staged = multiply_bf16(normalized_.view(), packed_projection_.view(), projected_.span(), capabilities_, stream);
    if (staged)
        staged = launch_kernel(&tower_token_normalize_kernel_, row_grid, block_shape, stream, projected_.view(),
                               inverse_norms_.span(), vectors_.span(), epsilon_);
    return staged;
}

expected_t tower_token_head_t::backward(bf16r2_view_t vectors_grad, multimodal_batch_view_t const &batch,
                                        f32r2_span_t states_grad, transformer_scratch_t &scratch,
                                        gpu_stream_t &stream) noexcept {
    std::size_t const tokens = gathered_.extent(0), hidden = gathered_.extent(1);
    if (vectors_grad.extent(0) != tokens || vectors_grad.extent(1) != vectors_.extent(1))
        return expected_t::fail(status_t::shape_unsupported_k, "token vectors_grad must be [{},{}] (got [{},{}])",
                                tokens, vectors_.extent(1), vectors_grad.extent(0), vectors_grad.extent(1));
    dim3 const block_shape {kernel_block_size_default_k, 1u, 1u};
    dim3 const row_grid {static_cast<u32_t>(tokens), 1u, 1u};
    std::size_t const token_bands = divide_round_up(tokens, std::size_t {kernel_block_size_default_k});
    f32r2_span_t const scale_grad_partials {scratch.qkv_grad.data(), token_bands, hidden};
    f32r2_span_t const products {scratch.products.data(), tokens, hidden};
    expected_t staged = launch_kernel(&tower_token_normalize_backward_kernel_, row_grid, block_shape, stream,
                                      projected_.view(), inverse_norms_.view(), vectors_grad, projected_grad_.span());
    if (staged)
        staged = weight_gradient_bf16(projected_grad_.view(), normalized_.view(), projection_grad_.span(),
                                      scratch.transposed_grad.span(), scratch.transposed_input.span(),
                                      scratch.packed_input.span(), scratch.weight_grad_products.span(),
                                      scratch.weight_gradient_workspace.span(), capabilities_, stream);
    if (staged)
        staged = multiply_bf16(projected_grad_.view(), packed_projection_transposed_.view(), products, capabilities_,
                               stream);
    // The normalized stash is dead once the projection's weight gradient read it, so it carries the
    // narrowed data gradient into the norm's backward.
    if (staged)
        staged = numkong_result(nk_cast_best(products.data(), nk_f32_k, normalized_.data(), nk_bf16_k, tokens * hidden,
                                             capabilities_, stream.numkong()),
                                "token head gradient narrowing");
    if (staged)
        staged = launch_kernel(&rmsnorm_backward_kernel, dim3 {static_cast<u32_t>(token_bands), 1u, 1u}, block_shape,
                               stream, gathered_.view(), norm_scale_.view(), inverse_rms_.view(), normalized_.view(),
                               gathered_grad_.span(), scale_grad_partials);
    if (staged)
        staged = launch_kernel(&rmsnorm_scale_grad_kernel,
                               dim3 {static_cast<u32_t>(divide_round_up(hidden, warp_size_k)), 1u, 1u}, block_shape,
                               stream, f32r2_view_t {scale_grad_partials}, norm_scale_grad_.span());
    if (staged)
        staged = launch_kernel(&tower_token_scatter_kernel_, row_grid, block_shape, stream, gathered_grad_.view(),
                               batch, token_offsets_.view(), states_grad);
    return staged;
}

expected_t tower_token_head_t::fill(array<differentiable_weights_t> &weights) noexcept {
    std::size_t const hidden = norm_scale_.extent(0);
    differentiable_weights_t norm {{},
                                   bf16r2_span_t {norm_scale_.data(), 1u, hidden},
                                   bf16r2_view_t {norm_scale_grad_.data(), 1u, hidden}};
    differentiable_weights_t projection {{}, projection_.span(), projection_grad_.view()};
    if (sz::failed(norm.name.assign("token_head.norm")) ||
        sz::failed(projection.name.assign("token_head.projection")) ||
        weights.push_back(std::move(norm)) != status_t::success_k ||
        weights.push_back(std::move(projection)) != status_t::success_k)
        return expected_t::fail(status_t::bad_alloc_k, "weight table overflowed");
    return {};
}

expected_t tower_token_head_t::pack_weights(bf16r1_span_t transposed, gpu_stream_t &stream) noexcept {
    return pack_weight_bf16(projection_.view(), transposed, packed_projection_.span(),
                            packed_projection_transposed_.span(), capabilities_, stream);
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
    // stay 0, and the media projections add onto that zero).
    if (expected_t zeroed = zero_async(coords_.span(), stream); !zeroed) return zeroed;
    if (expected_t zeroed = zero_async(entry_output_.span(), stream); !zeroed) return zeroed;

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
                                    batch.video_shapes.view(), config_.video_tube_frames, config_.video_frame_stride,
                                    coords_.span(), stream);
    if (!staged) return staged;

    // Entry, per-segment gather: text segments look up the embedding table into their cells, which widen into
    // the Float32 state `activations_b_`; each present media modality zeroes the scratch, stages only its own
    // cells in place (image / video pixels normalized, audio raw frames cast as-is; content shares the flat grid
    // position of its state), and adds its projection, so the zero non-modality rows leave every other cell
    // untouched — no scatter, no atomics.

    live_token_ids_ = batch.tokens.flatten<1>();
    staged = launch_kernel(&tower_text_entry_kernel_, segment_grid, block_shape, stream, live_token_ids_,
                           entry_weight_.span(), segment_offsets, segment_lengths, segment_modality,
                           entry_output_.span());
    if (staged)
        staged = numkong_result(nk_cast_best(entry_output_.data(), nk_bf16_k, activations_b_.data(), nk_f32_k,
                                             live_tokens * hidden, capabilities_, stream.numkong()),
                                "entry widening");
    if (staged)
        staged = project_media_entries(batch, patch_scratch_, image_pixel_mean_, image_pixel_inverse_std_,
                                       packed_image_projection_, video_pixel_mean_, video_pixel_inverse_std_,
                                       packed_video_projection_, packed_audio_projection_, activations_b_.span(),
                                       scratch_->products.span(), capabilities_, stream);
    if (!staged) return staged;

    // The block stack carries the Float32 residual stream: layer n reads layer n-1's output, ping-ponging the two
    // buffers, and the last output is a stash the backward reads.
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
                                      hidden, capabilities_, stream.numkong());
    if (status == nk_success_k)
        status = nk_each_rmscast_bf16_best(stack_output_.data(), widened_final_norm_scale_.data(),
                                           numkong_scalars(normalized_.data()), live_tokens, 1, hidden,
                                           hidden * sizeof(f32_t), hidden * sizeof(bf16_t), config_.block.rms_eps,
                                           capabilities_, stream.numkong());
    if (status != nk_success_k)
        return expected_t::fail(status_t::cuda_error_k, "final norm failed: {}", nk_status_name(status));
    if (config_.token_width != 0)
        if (expected_t projected = token_head_.forward(normalized_.view(), batch.view(), stream); !projected)
            return projected;

    // Masked path stops here — the masked head reads normalized(); the contrastive path pools + heads.
    if (vertices.empty()) return {};
    if (vertices.extent(0) != segments || vertices.extent(1) != hidden)
        return expected_t::fail(status_t::shape_unsupported_k, "vertices must be [{},{}] (got [{},{}])", segments,
                                hidden, vertices.extent(0), vertices.extent(1));

    // The attention pool: every segment's live tokens under the one learned query into pooled_.
    staged = pool_.forward(pool_query_.view(), normalized_.view(), segment_offsets, segment_lengths, pooled_.span(),
                           stream);
    if (!staged) return staged;

    // The head: learnable-γ RMSNorm, then the biasless fc1 → SiLU → fc2 into the caller's vertices —
    // raw, never L2-normalized here (the loss owns the cosine geometry).
    staged = launch_kernel(&rmsnorm_forward_kernel, segment_grid, block_shape, stream, pooled_.view(),
                           head_norm_scale_.view(), head_normalized_.span(), head_inverse_rms_.span(),
                           config_.block.rms_eps);
    if (!staged) return staged;
    return head_forward_(head_normalized_.view(), packed_head_fc1_, packed_head_fc2_, head_hidden_.span(),
                         head_activated_.span(), f32r2_span_t {scratch_->products.data(), segments, hidden}, vertices,
                         capabilities_, stream);
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
    if (image_projection_.data() != nullptr && batch.images.data() != nullptr && image_patches.data() != nullptr &&
        image_patches.extent(0) != 0) {
        std::size_t const spatial = static_cast<std::size_t>(batch.images.extent(2)) * batch.images.extent(3);
        staged = stage_pixel_patches(batch.images.view().reshape<3>({live_tokens, spatial, batch.images.extent(4)}),
                                     modality_t::image_k, image_pixel_mean_.view(), image_pixel_inverse_std_.view(),
                                     segment_offsets, segment_lengths, segment_modality, segment_grid, image_patches,
                                     stream);
        if (!staged) return staged;
    }
    if (video_projection_.data() != nullptr && batch.videos.data() != nullptr && video_patches.data() != nullptr &&
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
                                          multimodal_batch_on_gpu_t const &batch, gpu_stream_t &stream,
                                          bf16r2_view_t token_vectors_grad) noexcept {
    u32_t const hidden = config_.block.hidden;
    std::size_t const columns = batch.tokens.extent(1);
    std::size_t const live_tokens = static_cast<std::size_t>(batch.tokens.extent(0)) * columns;
    bool const pooled_present = vertices_grad.data() != nullptr;
    bool const masked_present = hidden_grad.data() != nullptr;
    bool const tokens_present = token_vectors_grad.data() != nullptr;
    if (!pooled_present && !masked_present && !tokens_present)
        return expected_t::fail(status_t::invalid_input_k,
                                "backward needs one of vertices_grad / hidden_grad / token_vectors_grad");
    if (tokens_present && config_.token_width == 0)
        return expected_t::fail(status_t::invalid_input_k, "token_vectors_grad given to a tower without a token head");
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
    dim3 const channel_grid {static_cast<u32_t>(divide_round_up(hidden, warp_size_k)), 1u, 1u};
    std::size_t const token_bands = divide_round_up(live_tokens, std::size_t {kernel_block_size_default_k});
    std::size_t const segment_bands = divide_round_up(segments, std::size_t {kernel_block_size_default_k});
    f32r2_span_t const token_scale_grad_partials {scratch_->qkv_grad.data(), token_bands, hidden};
    f32r2_span_t const segment_scale_grad_partials {scratch_->qkv_grad.data(), segment_bands, hidden};
    u32r1_view_t const segment_offsets = batch.segment_offsets.view();
    u32r1_view_t const segment_lengths = batch.segment_lengths.view();
    modality_view_t const segment_modality = batch.segment_modality.view();

    expected_t staged {};
    if (pooled_present) {
        // The head backward, mirroring the forward in reverse through the segment ping-pong: fc2's two
        // grads, the SiLU derivative on the stashed pre-activation, fc1's two grads, then the head norm.
        std::size_t const head_cells = segments * hidden;
        dim3 const head_cell_grid {static_cast<u32_t>(divide_round_up(head_cells, kernel_block_size_default_k)), 1u,
                                   1u};
        transformer_scratch_t &scratch = *scratch_;
        f32r2_span_t const head_products {scratch.products.data(), segments, hidden};
        auto weight_gradient = [&](bf16r2_view_t product_grad, bf16r2_view_t product_input,
                                   bf16r2_span_t weight_grad) noexcept {
            return weight_gradient_bf16(product_grad, product_input, weight_grad, scratch.transposed_grad.span(),
                                        scratch.transposed_input.span(), scratch.packed_input.span(),
                                        scratch.weight_grad_products.span(), scratch.weight_gradient_workspace.span(),
                                        capabilities_, stream);
        };
        auto data_gradient = [&](bf16r2_view_t product_grad, b8r1_tensor_arena_t const &packed_transposed) noexcept {
            expected_t multiplied = multiply_bf16(product_grad, packed_transposed.view(), head_products, capabilities_,
                                                  stream);
            if (multiplied)
                multiplied = numkong_result(nk_cast_best(head_products.data(), nk_f32_k, segment_grad_a_.data(),
                                                         nk_bf16_k, head_cells, capabilities_, stream.numkong()),
                                            "head gradient narrowing");
            return multiplied;
        };
        staged = weight_gradient(vertices_grad, head_activated_.view(), head_fc2_grad_.span());
        if (staged) staged = data_gradient(vertices_grad, packed_head_fc2_transposed_);
        if (staged)
            staged = launch_kernel(&tower_silu_backward_kernel_, head_cell_grid, block_shape, stream,
                                   head_hidden_.span().flatten<1>(), segment_grad_a_.span().flatten<1>(),
                                   segment_grad_b_.span().flatten<1>());
        if (staged) staged = weight_gradient(segment_grad_b_.view(), head_normalized_.view(), head_fc1_grad_.span());
        if (staged) staged = data_gradient(segment_grad_b_.view(), packed_head_fc1_transposed_);
        if (staged)
            staged = launch_kernel(&rmsnorm_backward_kernel, dim3 {static_cast<u32_t>(segment_bands), 1u, 1u},
                                   block_shape, stream, pooled_.view(), head_norm_scale_.view(),
                                   head_inverse_rms_.view(), segment_grad_a_.view(), segment_grad_b_.span(),
                                   segment_scale_grad_partials);
        if (staged)
            staged = launch_kernel(&rmsnorm_scale_grad_kernel, channel_grid, block_shape, stream,
                                   f32r2_view_t {segment_scale_grad_partials}, head_norm_scale_grad_.span());
        if (!staged) return staged;

        // The pool backward leaves the Float32 d(normalized) of every live token row in token_grad_a_.
        staged = pool_.backward(pool_query_.view(), segment_grad_b_.view(), token_grad_a_.span(), token_grad_b_.span(),
                                pool_query_grad_.span(), stream);
        if (!staged) return staged;
    }

    // The caller's BFloat16 masked force widens into token_grad_a_ alone, or beside the pooled force and adds
    // onto it: the co-trained join. The caller's hidden_grad may carry a wider capacity than the live extent.
    if (masked_present) {
        bf16r2_view_t const masked_grad = hidden_grad[nk::range(0, live_tokens), nk::slice];
        f32_t *const widened = pooled_present ? token_grad_b_.data() : token_grad_a_.data();
        nk_status_t status = nk_cast_best(masked_grad.data(), nk_bf16_k, widened, nk_f32_k, live_tokens * hidden,
                                          capabilities_, stream.numkong());
        if (status == nk_success_k && pooled_present)
            status = nk_each_sum_f32_best(token_grad_a_.data(), token_grad_b_.data(), live_tokens * hidden,
                                          token_grad_a_.data(), capabilities_, stream.numkong());
        if (status != nk_success_k)
            return expected_t::fail(status_t::cuda_error_k, "masked force join failed: {}", nk_status_name(status));
    }
    if (tokens_present) {
        if (!pooled_present && !masked_present)
            if (expected_t zeroed = zero_async(token_grad_a_.span(), stream); !zeroed) return zeroed;
        staged = token_head_.backward(token_vectors_grad, batch.view(), token_grad_a_.span(), *scratch_, stream);
        if (!staged) return staged;
    }

    // Final-norm backward: the Float32 d(normalized) in token_grad_a_ → the Float32 d(stack output) into
    // token_grad_b_, plus the scale grad.
    staged = launch_kernel(&rmscast_backward_kernel, dim3 {static_cast<u32_t>(token_bands), 1u, 1u}, block_shape,
                           stream, stack_output_, final_norm_scale_.view(), token_grad_a_.view(), f32r2_view_t {},
                           token_grad_b_.span(), token_scale_grad_partials, config_.block.rms_eps);
    if (staged)
        staged = launch_kernel(&rmsnorm_scale_grad_kernel, channel_grid, block_shape, stream,
                               f32r2_view_t {token_scale_grad_partials}, final_norm_scale_grad_.span());
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
                                                live_tokens * hidden, capabilities_, stream.numkong());
        status != nk_success_k)
        return expected_t::fail(status_t::cuda_error_k, "entry gradient narrowing failed: {}", nk_status_name(status));
    bf16r2_view_t const entry_grad = normalized_grad_.view();

    // Entry backward, text cells: deterministic embedding scatter over the whole packed grid. Keys start at
    // the sentinel (max token id) so every non-text cell folds into one trailing run the scatter skips; the
    // text segments overwrite their keys, positions are the flat cell indices, and the sort + serial per-run
    // fold stay deterministic with no atomics.

    if (expected_t zeroed = zero_async(entry_weight_grad_.span(), stream); !zeroed) return zeroed;
    // The sort scratch is one arena split into two halves, max_tokens apart: [0,live) and [max,max+live).

    std::size_t const max_tokens = config_.block.max_tokens;
    tokens_span_t const keys_in = sort_keys_[nk::range(0, live_tokens), nk::slice];
    tokens_span_t const keys_out = sort_keys_[nk::range(max_tokens, max_tokens + live_tokens), nk::slice];
    u32r1_span_t const values_in = sort_values_[nk::range(0, live_tokens), nk::slice];
    u32r1_span_t const values_out = sort_values_[nk::range(max_tokens, max_tokens + live_tokens), nk::slice];
    if (CUresult result = cuMemsetD8Async(device_address(keys_in.data()), 0xFF, live_tokens * sizeof(token_id_t),
                                          stream.get());
        result != CUDA_SUCCESS)
        return cuda_failure(result, "sentinel key reset failed");
    staged = launch_kernel(&tower_text_keys_kernel_, segment_grid, block_shape, stream, live_token_ids_,
                           segment_offsets, segment_lengths, segment_modality, keys_in);
    if (staged)
        staged = launch_kernel(&tower_iota_kernel_, dim3 {grid_size_for(live_tokens), 1u, 1u},
                               dim3 {warp_size_k, 1u, 1u}, stream, values_in);
    if (!staged) return staged;
    std::size_t sort_bytes = sort_workspace_.extent(0);
    if (cudaError_t sorted = cub::DeviceRadixSort::SortPairs(
            sort_workspace_.data(), sort_bytes, keys_in.data(), keys_out.data(), values_in.data(), values_out.data(),
            static_cast<int>(live_tokens), 0, sizeof(token_id_t) * 8, stream.get());
        sorted != cudaSuccess)
        return expected_t::fail(status_t::cuda_error_k, "embedding sort failed (code={})", static_cast<int>(sorted));
    staged = launch_kernel(&tower_embedding_scatter_kernel_, row_grid, block_shape, stream, tokens_view_t {keys_out},
                           u32r1_view_t {values_out}, entry_grad, entry_weight_grad_.span());
    if (!staged) return staged;

    // Media weight grads: re-stage each present modality's cells (segment-driven, so non-modality rows stay
    // zero) and fold the whole entry gradient against them — the zero patch rows drop every non-modality term,
    // so dW gathers exactly its modality's cells. The backward twin of project_media_entries; coords_ was
    // filled by the paired forward, so the backward derives no coordinates.
    return project_media_weight_grads(batch, patch_scratch_, entry_grad, image_pixel_mean_, image_pixel_inverse_std_,
                                      image_projection_grad_, video_pixel_mean_, video_pixel_inverse_std_,
                                      video_projection_grad_, audio_projection_grad_, *scratch_, capabilities_, stream);
}

expected_t encoder_tower_bf16_t::pack_weights(gpu_stream_t &stream) noexcept {
    for (transformer_block_bf16_t &layer : blocks_)
        if (expected_t packed = layer.pack_weights(stream); !packed) return packed;
    bf16r1_span_t const transposed = scratch_->transposed_weight.span();
    expected_t packed = pack_weight_bf16(head_fc1_.view(), transposed, packed_head_fc1_.span(),
                                         packed_head_fc1_transposed_.span(), capabilities_, stream);
    if (packed)
        packed = pack_weight_bf16(head_fc2_.view(), transposed, packed_head_fc2_.span(),
                                  packed_head_fc2_transposed_.span(), capabilities_, stream);
    bf16r2_tensor_arena_t const *const projections[] = {&image_projection_, &video_projection_, &audio_projection_};
    b8r1_tensor_arena_t *const packs[] = {&packed_image_projection_, &packed_video_projection_,
                                          &packed_audio_projection_};
    for (std::size_t modality = 0; packed && modality != 3; ++modality)
        if (projections[modality]->data() != nullptr)
            packed = pack_bf16(projections[modality]->view(), packs[modality]->span(), capabilities_, stream);
    if (packed && config_.token_width != 0) packed = token_head_.pack_weights(transposed, stream);
    return packed;
}

expected_t encoder_tower_bf16_t::fill(array<differentiable_weights_t> &weights) noexcept {
    u32_t const hidden = config_.block.hidden;
    u32_t const ffn = config_.block.ffn_expansion * hidden;
    // Every weight joins under the checkpoint name the tower spells for it — the composition root adds only
    // its `model.` prefix. A block weight arrives as a rank-1 accessor; it joins as its logical [rows,cols]
    // rectangle (a tower matrix is already rank-2 and joins through its own span / view).

    auto push = [&weights](sz::string_view_t name, bf16r2_span_t weight, bf16r2_view_t grad) noexcept {
        differentiable_weights_t entry {{}, weight, grad};
        if (sz::failed(entry.name.assign(name))) return status_t::bad_alloc_k;
        return weights.push_back(std::move(entry));
    };
    auto push_flat = [&push](sz::string_view_t name, bf16r1_span_t weight, bf16r1_view_t grad, std::size_t rows,
                             std::size_t cols) noexcept {
        return push(name, bf16r2_span_t {weight.data(), rows, cols}, bf16r2_view_t {grad.data(), rows, cols});
    };
    status_t pushed = push("embedding", entry_weight_.span(), entry_weight_grad_.view());
    // Media projections next, in modality order — each present iff it was allocated. These rank-2
    // [hidden, fan_in] matrices join directly (like the head products), replica-identical with the grads
    // backward() produces, so the optimizer trains what was frozen before.
    if (pushed == status_t::success_k && image_projection_.data() != nullptr)
        pushed = push("image_projection", image_projection_.span(), image_projection_grad_.view());
    if (pushed == status_t::success_k && video_projection_.data() != nullptr)
        pushed = push("video_projection", video_projection_.span(), video_projection_grad_.view());
    if (pushed == status_t::success_k && audio_projection_.data() != nullptr)
        pushed = push("audio_projection", audio_projection_.span(), audio_projection_grad_.view());
    sz::string_t name_text;
    for (u32_t layer_index = 0; pushed == status_t::success_k && layer_index != config_.depth; ++layer_index) {
        transformer_block_bf16_t &layer = blocks_[layer_index];
        struct block_part_t {
            sz::string_view_t suffix;
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
    if (config_.token_width != 0) return token_head_.fill(weights);
    return {};
}

/** Snapshot one BFloat16 matrix into a fresh arena tensor — the twin's tail tiers (entries,
 *  norms, pool, head) stay BFloat16, so their quantization is a plain device-to-device copy.
 *  @p name tags the failure diagnostic with the weight's role. */
[[nodiscard]] static expected<bf16r2_tensor_arena_t> snapshot_weight_(bf16r2_view_t source, gpu_arena_t &arena,
                                                                      gpu_stream_t &stream,
                                                                      sz::string_view_t name) noexcept {
    using result_t = expected<bf16r2_tensor_arena_t>;
    gpu_arena_alloc<bf16_t> bf16_alloc {arena};
    bf16r2_tensor_arena_t copy =
        bf16r2_tensor_arena_t::uninitialized({source.extent(0), source.extent(1)}, bf16_alloc).value;
    if (!copy) return result_t::fail(status_t::bad_alloc_k, "twin {} allocation failed", name);
    std::size_t const scalars = source.extent(0) * source.extent(1);
    if (expected_t copied = copy_device_to_device(bf16r1_view_t {source.data(), scalars},
                                                  bf16r1_span_t {copy.data(), scalars}, stream);
        !copied)
        return result_t::fail(std::move(copied));
    return result_t {std::move(copy)};
}

/** The rank-1 overload of @ref snapshot_weight_ for the norm scales and the pool query. */
[[nodiscard]] static expected<bf16r1_tensor_arena_t> snapshot_weight_(bf16r1_view_t source, gpu_arena_t &arena,
                                                                      gpu_stream_t &stream,
                                                                      sz::string_view_t name) noexcept {
    using result_t = expected<bf16r1_tensor_arena_t>;
    gpu_arena_alloc<bf16_t> bf16_alloc {arena};
    bf16r1_tensor_arena_t copy = bf16r1_tensor_arena_t::uninitialized({source.extent(0)}, bf16_alloc).value;
    if (!copy) return result_t::fail(status_t::bad_alloc_k, "twin {} allocation failed", name);
    if (expected_t copied = copy_device_to_device(source, copy.span(), stream); !copied)
        return result_t::fail(std::move(copied));
    return result_t {std::move(copy)};
}

expected<encoder_tower_e4m3_t> encoder_tower_bf16_t::quantize_weights(gpu_arena_t &arena,
                                                                      gpu_stream_t &stream) noexcept {
    using result_t = expected<encoder_tower_e4m3_t>;
    encoder_tower_e4m3_t twin;
    twin.config_ = config_;
    twin.capabilities_ = capabilities_;

    std::size_t const hidden = config_.block.hidden;
    std::size_t const tokens = config_.block.max_tokens;
    std::size_t const segments = config_.block.max_segments;
    gpu_arena_alloc<bf16_t> bf16_alloc {arena};
    gpu_arena_alloc<f32_t> f32_alloc {arena};
    gpu_arena_alloc<i16_t> i16_alloc {arena};

    // The block stack quantizes per layer; everything else below is a BFloat16 snapshot.
    if (twin.blocks_.resize(config_.depth) != status_t::success_k)
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

    // Every product's B operand packs once here, since the twin's weights never change: the two head projections,
    // then — per present media modality — the projection snapshot, in modality order.
    auto snapshot_and_pack = [&](bf16r2_view_t weight, bf16r2_tensor_arena_t &copy, b8r1_tensor_arena_t &packed,
                                 sz::string_view_t name) noexcept -> expected_t {
        expected<bf16r2_tensor_arena_t> snapshot = snapshot_weight_(weight, arena, stream, name);
        if (!snapshot) return expected_t::fail(std::move(snapshot));
        copy = std::move(snapshot.value_);
        expected<b8r1_tensor_arena_t> allocated = allocate_pack_bf16(weight.extent(0), weight.extent(1),
                                                                     twin.capabilities_, arena);
        if (!allocated) return expected_t::fail(std::move(allocated));
        packed = std::move(allocated.value_);
        return pack_bf16(copy.view(), packed.span(), twin.capabilities_, stream);
    };
    expected_t packed = snapshot_and_pack(head_fc1_.view(), twin.head_fc1_, twin.packed_head_fc1_, "head fc1");
    if (packed) packed = snapshot_and_pack(head_fc2_.view(), twin.head_fc2_, twin.packed_head_fc2_, "head fc2");
    if (packed && image_projection_.data() != nullptr)
        packed = snapshot_and_pack(image_projection_.view(), twin.image_projection_, twin.packed_image_projection_,
                                   "image projection");
    if (packed && video_projection_.data() != nullptr)
        packed = snapshot_and_pack(video_projection_.view(), twin.video_projection_, twin.packed_video_projection_,
                                   "video projection");
    if (packed && audio_projection_.data() != nullptr)
        packed = snapshot_and_pack(audio_projection_.view(), twin.audio_projection_, twin.packed_audio_projection_,
                                   "audio projection");
    if (!packed) return result_t::fail(std::move(packed));
    if (image_projection_.data() != nullptr) {
        expected_t uploaded = upload_pixel_constants(twin.image_pixel_mean_, twin.image_pixel_inverse_std_,
                                                     config_.image_pixel_mean, config_.image_pixel_std,
                                                     config_.image_channels, f32_alloc, arena);
        if (!uploaded) return result_t::fail(std::move(uploaded));
    }
    if (video_projection_.data() != nullptr) {
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
    twin.products_ = f32r1_tensor_arena_t::uninitialized({tokens * hidden}, f32_alloc).value;
    if (!twin.coords_ || !twin.entry_output_ || !twin.normalized_ || !twin.pooled_ || !twin.head_normalized_ ||
        !twin.head_hidden_ || !twin.head_activated_ || !twin.final_inverse_rms_ || !twin.head_inverse_rms_ ||
        !twin.activations_a_ || !twin.activations_b_ || !twin.products_)
        return result_t::fail(status_t::bad_alloc_k, "twin activation allocation failed (arena_used={})", arena.used());
    if (patch_scratch_.data() != nullptr) {
        twin.patch_scratch_ =
            bf16r2_tensor_arena_t::uninitialized({tokens, patch_scratch_.extent(1)}, bf16_alloc).value;
        twin.entry_state_ = f32r2_tensor_arena_t::uninitialized({tokens, hidden}, f32_alloc).value;
        if (!twin.patch_scratch_ || !twin.entry_state_)
            return result_t::fail(status_t::bad_alloc_k, "twin media scratch allocation failed (arena_used={})",
                                  arena.used());
    }

    expected<tower_pool_t> pool = tower_pool_t::make(arena, twin.config_.block);
    if (!pool) return result_t::fail(std::move(pool));
    twin.pool_ = std::move(pool.value_);

    // The snapshots above are stream-ordered; drain them so the twin owns settled weights.
    if (CUresult result = cuStreamSynchronize(stream.get()); result != CUDA_SUCCESS)
        return cuda_failure<encoder_tower_e4m3_t>(result, "twin snapshot drain failed");
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

    if (expected_t zeroed = zero_async(coords_.span(), stream); !zeroed) return zeroed;
    if (expected_t zeroed = zero_async(entry_output_.span(), stream); !zeroed) return zeroed;

    dim3 const block_shape {kernel_block_size_default_k, 1u, 1u};
    dim3 const segment_grid {static_cast<u32_t>(segments), 1u, 1u};
    u32r1_view_t const segment_offsets = batch.segment_offsets.view();
    u32r1_view_t const segment_lengths = batch.segment_lengths.view();
    modality_view_t const segment_modality = batch.segment_modality.view();

    // Coordinates + per-segment entry, mirroring the training twin: one fill_coords, a text-only embedding
    // gather, and a projection added per present media modality (zero non-modality rows compose).
    u32r1_span_t const offsets_span {const_cast<u32_t *>(segment_offsets.data()), segment_offsets.extent(0)};
    u32r1_span_t const lengths_span {const_cast<u32_t *>(segment_lengths.data()), segment_lengths.extent(0)};
    modality_span_t const modality_span {const_cast<modality_t *>(segment_modality.data()), segment_modality.extent(0)};
    expected_t staged = fill_coords(offsets_span, lengths_span, modality_span, batch.image_shapes.view(),
                                    batch.video_shapes.view(), config_.video_tube_frames, config_.video_frame_stride,
                                    coords_.span(), stream);
    if (!staged) return staged;

    tokens_view_t const token_ids = batch.tokens.flatten<1>();
    staged = launch_kernel(&tower_text_entry_kernel_, segment_grid, block_shape, stream, token_ids,
                           entry_weight_.span(), segment_offsets, segment_lengths, segment_modality,
                           entry_output_.span());
    if (!staged) return staged;

    // The media projections add in Float32, the same shared helper the training twin runs, then narrow once.
    if (entry_state_.data() != nullptr) {
        std::size_t const cells = live_tokens * hidden;
        f32r2_span_t const entry_state {entry_state_.data(), live_tokens, hidden};
        staged = numkong_result(nk_cast_best(entry_output_.data(), nk_bf16_k, entry_state.data(), nk_f32_k, cells,
                                             capabilities_, stream.numkong()),
                                "entry widening");
        if (staged)
            staged = project_media_entries(batch, patch_scratch_, image_pixel_mean_, image_pixel_inverse_std_,
                                           packed_image_projection_, video_pixel_mean_, video_pixel_inverse_std_,
                                           packed_video_projection_, packed_audio_projection_, entry_state,
                                           products_.span(), capabilities_, stream);
        if (staged)
            staged = numkong_result(nk_cast_best(entry_state.data(), nk_f32_k, entry_output_.data(), nk_bf16_k, cells,
                                                 capabilities_, stream.numkong()),
                                    "entry narrowing");
        if (!staged) return staged;
    }

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
    staged = launch_kernel(&rmsnorm_forward_kernel, token_grid, block_shape, stream, layer_input,
                           final_norm_scale_.view(), normalized_.span(), final_inverse_rms_.span(),
                           config_.block.rms_eps);
    if (!staged) return staged;

    // Masked path stops here; the contrastive path pools + heads, exactly as the training twin.
    if (vertices.empty()) return {};
    if (vertices.extent(0) != segments || vertices.extent(1) != hidden)
        return expected_t::fail(status_t::shape_unsupported_k, "vertices must be [{},{}] (got [{},{}])", segments,
                                hidden, vertices.extent(0), vertices.extent(1));
    staged = pool_.forward(pool_query_.view(), normalized_.view(), segment_offsets, segment_lengths, pooled_.span(),
                           stream);
    if (!staged) return staged;
    staged = launch_kernel(&rmsnorm_forward_kernel, segment_grid, block_shape, stream, pooled_.view(),
                           head_norm_scale_.view(), head_normalized_.span(), head_inverse_rms_.span(),
                           config_.block.rms_eps);
    if (!staged) return staged;
    return head_forward_(head_normalized_.view(), packed_head_fc1_, packed_head_fc2_, head_hidden_.span(),
                         head_activated_.span(), f32r2_span_t {products_.data(), segments, hidden}, vertices,
                         capabilities_, stream);
}
