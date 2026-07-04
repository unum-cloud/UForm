/**
 *  @brief Device kernels and heavy method bodies of the encoder tower declared in
 *      @ref encoder_tower.cuh — the translation unit that joins the `uform_attention` target beside
 *      @ref transformer_block.cu .
 *  @file cuda_core/encoder_tower.cu
 *  @author Ash Vardanian
 *  @date July 4, 2026
 */
#include "encoder_tower.cuh"

#include <algorithm> // `std::max` — the running plan-workspace maximum

#include <cub/device/device_radix_sort.cuh> // the deterministic embedding-gradient scatter

using namespace unum::uform::core;

/**
 *  @brief Token-embedding lookup: @c x[t,:] = table[token[t],:] .
 *
 *  __grid_size__ @c [tokens,1,1] , threads stripe the hidden channels; no dynamic shared memory.
 */
static __global__ void tower_embedding_lookup_kernel_(tokens_view_t tokens, bf16r2_span_t table,
                                                      bf16r2_span_t destination) {
    std::size_t const token = blockIdx.x;
    u32_t const hidden = static_cast<u32_t>(destination.extent(1));
    std::size_t const id = static_cast<std::size_t>(tokens[token]);
    for (u32_t channel = threadIdx.x; channel < hidden; channel += blockDim.x)
        destination[token, channel] = table[id, channel];
}

/**
 *  @brief Deterministic embedding-gradient scatter over the pre-sorted (token, position) pairs: one CTA
 *      per position range start walks its whole duplicate run serially — fixed order, no atomics.
 *
 *  __grid_size__ @c [tokens,1,1] ; a CTA whose slot is not a run head exits immediately.
 */
static __global__ void tower_embedding_scatter_kernel_(tokens_view_t sorted_tokens, u32r1_view_t sorted_positions,
                                                       bf16r2_view_t token_grad, bf16r2_span_t table_grad) {
    std::size_t const slot = blockIdx.x;
    std::size_t const tokens = sorted_tokens.extent(0);
    token_id_t const token = sorted_tokens[slot];
    if (slot != 0 && sorted_tokens[slot - 1] == token) return; // Not this run's head.
    u32_t const hidden = static_cast<u32_t>(table_grad.extent(1));
    for (u32_t channel = threadIdx.x; channel < hidden; channel += blockDim.x) {
        f32_t total = 0.0f;
        for (std::size_t run = slot; run < tokens && sorted_tokens[run] == token; ++run)
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

expected<encoder_tower_bf16_t> encoder_tower_bf16_t::make(gpu_arena_t &arena, cublaslt_session_t const &blas,
                                                          cudnn_session_t const &cudnn, encoder_tower_config_t config,
                                                          [[maybe_unused]] multimodal_batch_shape_t shape) noexcept {
    // Media-entry validation reads @p shape once the image_frontend patch-cut handoff lands.

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
    if (!tower.entry_output_ || !tower.normalized_ || !tower.pooled_ || !tower.head_normalized_ ||
        !tower.head_hidden_ || !tower.head_activated_ || !tower.final_inverse_rms_ || !tower.head_inverse_rms_)
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
    std::size_t plan_slot = 0;
    auto pin_plan = [&](cublaslt_plan_config_t plan_config) noexcept -> expected_t {
        plan_config.c_type = CUDA_R_16BF;
        expected<cublaslt_plan_t> plan = cublaslt_plan_t::make(blas, plan_config);
        if (!plan) return expected_t::fail(std::move(plan));
        workspace_bytes = std::max(workspace_bytes, plan.value_.workspace_bytes());
        tower.plans_[plan_slot++] = std::move(plan.value_);
        return {};
    };
    u32_t const hidden32 = config.block.hidden;
    u32_t const segments32 = config.block.max_segments;
    expected_t pinned = pin_plan(
        {segments32, hidden32, hidden32, hidden32, hidden32, hidden32, segments32, hidden32, hidden32, false, true});
    if (pinned)
        pinned = pin_plan({segments32, hidden32, hidden32, hidden32, hidden32, hidden32, segments32, hidden32, hidden32,
                           false, false});
    if (pinned)
        pinned = pin_plan({segments32, hidden32, hidden32, segments32, hidden32, hidden32, hidden32, hidden32, hidden32,
                           true, false});
    if (!pinned) return result_t::fail(std::move(pinned));
    if (workspace_bytes != 0) {
        tower.workspace_ = b8r1_tensor_arena_t::try_empty({workspace_bytes}, b8_alloc);
        if (!tower.workspace_)
            return result_t::fail(status_t::bad_alloc_k, "tower workspace allocation failed ({} bytes)",
                                  workspace_bytes);
    }
    return result_t {std::move(tower)};
}

expected_t encoder_tower_bf16_t::forward(multimodal_batch_on_gpu_t const &batch, i16r2_view_t coords,
                                         bf16r2_span_t vertices, gpu_stream_t const &stream) noexcept {
    u32_t const hidden = config_.block.hidden;
    std::size_t const live_tokens = batch.tokens.extent(0) * batch.tokens.extent(1);
    std::size_t const segments = batch.segment_lengths.extent(0);
    if (live_tokens == 0 || live_tokens > config_.block.max_tokens || segments != config_.block.max_segments)
        return expected_t::fail(status_t::shape_unsupported_k,
                                "batch disagrees with the baked capacities (tokens={} segments={})", live_tokens,
                                segments);
    if (coords.extent(0) < live_tokens)
        return expected_t::fail(status_t::invalid_input_k, "coords cover {} of {} tokens", coords.extent(0),
                                live_tokens);

    // The step's live extent, resized within each arena's reserved capacity — no address moves, and the
    // capacity check above makes this resize infallible; the bool is still checked, never assumed.

    bool const resized = entry_output_.try_resize({live_tokens, hidden}) &&
                         normalized_.try_resize({live_tokens, hidden}) &&
                         activations_a_.try_resize({live_tokens, hidden}) &&
                         activations_b_.try_resize({live_tokens, hidden});
    if (!resized)
        return expected_t::fail(status_t::shape_unsupported_k, "tower scratch cannot hold {} live tokens (capacity {})",
                                live_tokens, entry_output_.capacity() / hidden);

    dim3 const block_shape {kernel_block_size_default_k, 1u, 1u};
    dim3 const token_grid {static_cast<u32_t>(live_tokens), 1u, 1u};
    u32r1_view_t const segment_offsets = batch.segment_offsets.view();
    u32r1_view_t const segment_lengths = batch.segment_lengths.view();

    // Entry: token embedding lookup into the block-stack input. (Media projections land with the
    // image_frontend patch-cut handoff.)
    live_token_ids_ = batch.tokens.flatten<1>();
    expected_t staged = launch_kernel(reinterpret_cast<void const *>(&tower_embedding_lookup_kernel_), token_grid,
                                      block_shape, stream, live_token_ids_, entry_weight_.span(), entry_output_.span());
    if (!staged) return staged;

    // The block stack: layer n reads layer n-1's output, ping-ponging two dedicated buffers; the last
    // output is a stash the backward reads.

    bf16r2_view_t layer_input = entry_output_.view();
    for (u32_t layer_index = 0; layer_index != config_.depth; ++layer_index) {
        bf16r2_tensor_arena_t &layer_output = layer_index % 2 == 0 ? activations_a_ : activations_b_;
        staged = blocks_[layer_index].forward(layer_input, layer_output.span(), segment_offsets, segment_lengths,
                                              angles_.view(), coords, stream);
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

expected_t encoder_tower_bf16_t::backward(bf16r2_view_t vertices_grad, bf16r2_view_t hidden_grad,
                                          multimodal_batch_on_gpu_t const &batch, i16r2_view_t coords,
                                          gpu_stream_t const &stream) noexcept {
    u32_t const hidden = config_.block.hidden;
    std::size_t const live_tokens = batch.tokens.extent(0) * batch.tokens.extent(1);
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
    dim3 const token_grid {static_cast<u32_t>(live_tokens), 1u, 1u};
    dim3 const channel_grid {static_cast<u32_t>(divide_round_up(hidden, kernel_block_size_default_k)), 1u, 1u};
    u32r1_view_t const segment_offsets = batch.segment_offsets.view();
    u32r1_view_t const segment_lengths = batch.segment_lengths.view();

    // Final-norm backward: d(normalized) → d(stack output) into token_grad_b_, plus the scale grad. The
    // caller's hidden_grad may carry a wider capacity than the live extent, so it is narrowed to match.

    bool const overwrite = false;
    bf16r2_view_t const hidden_grad_live = hidden_grad(nk::range(0, live_tokens), nk::slice);
    expected_t staged = launch_kernel(reinterpret_cast<void const *>(&rmsnorm_backward_kernel), token_grid, block_shape,
                                      stream, stack_output_, final_norm_scale_.view(), final_inverse_rms_.view(),
                                      hidden_grad_live, token_grad_b_.span(), overwrite);
    if (staged)
        staged = launch_kernel(reinterpret_cast<void const *>(&rmsnorm_scale_grad_kernel), channel_grid, block_shape,
                               stream, stack_output_, final_inverse_rms_.view(), hidden_grad_live,
                               final_norm_scale_grad_.span());
    if (!staged) return staged;

    // Blocks in reverse; the flowing gradient ping-pongs the two token-gradient buffers.
    bf16r2_tensor_arena_t *grad_in = &token_grad_b_;
    bf16r2_tensor_arena_t *grad_out = &token_grad_a_;
    bf16r2_view_t flowing = grad_in->view();
    for (u32_t layer_index = config_.depth; layer_index != 0; --layer_index) {
        staged = blocks_[layer_index - 1].backward(flowing, grad_out->span(), segment_offsets, segment_lengths,
                                                   angles_.view(), coords, stream);
        if (!staged) return staged;
        flowing = grad_out->view();
        std::swap(grad_in, grad_out);
    }

    // Entry backward: deterministic embedding scatter (sort (token, position), then serial per-run fold).

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
    if (CUresult result = cuMemcpyDtoDAsync(reinterpret_cast<CUdeviceptr>(keys_in.data()),
                                            reinterpret_cast<CUdeviceptr>(live_token_ids_.data()),
                                            live_tokens * sizeof(token_id_t), stream.get());
        result != CUDA_SUCCESS)
        return expected_t::fail(result, "token id staging failed");
    staged = launch_kernel(reinterpret_cast<void const *>(&tower_iota_kernel_), token_grid, dim3 {warp_size_k, 1u, 1u},
                           stream, values_in);
    if (!staged) return staged;
    std::size_t sort_bytes = sort_workspace_.extent(0);
    if (cudaError_t sorted = cub::DeviceRadixSort::SortPairs(
            sort_workspace_.data(), sort_bytes, keys_in.data(), keys_out.data(), values_in.data(), values_out.data(),
            static_cast<int>(live_tokens), 0, sizeof(token_id_t) * 8, reinterpret_cast<cudaStream_t>(stream.get()));
        sorted != cudaSuccess)
        return expected_t::fail(status_t::cuda_error_k, "embedding sort failed (code={})", static_cast<int>(sorted));
    return launch_kernel(reinterpret_cast<void const *>(&tower_embedding_scatter_kernel_), token_grid, block_shape,
                         stream, tokens_view_t {keys_out}, u32r1_view_t {values_out}, flowing,
                         entry_weight_grad_.span());
}

expected_t encoder_tower_bf16_t::fill(array<differentiable_weights_t> &slots) noexcept {
    u32_t const hidden = config_.block.hidden;
    u32_t const ffn = config_.block.ffn_expansion * hidden;
    // A block weight arrives as a rank-1 accessor; it joins as its logical [rows,cols] rectangle (a tower
    // matrix is already rank-2 and joins through its own span / view).

    auto make_slot = [](bf16r1_span_t weight, bf16r1_view_t grad, std::size_t rows, std::size_t cols) noexcept {
        return differentiable_weights_t {bf16r2_span_t {weight.data(), rows, cols},
                                         bf16r2_view_t {grad.data(), rows, cols}};
    };
    status_t pushed = slots.try_push_back({entry_weight_.span(), entry_weight_grad_.view()});
    for (u32_t layer_index = 0; pushed == status_t::success_k && layer_index != config_.depth; ++layer_index) {
        transformer_block_bf16_t &layer = blocks_[layer_index];
        pushed = slots.try_push_back(make_slot(layer.qkv_weight(), layer.qkv_weight_grad(), 3u * hidden, hidden));
        if (pushed == status_t::success_k)
            pushed = slots.try_push_back(make_slot(layer.out_weight(), layer.out_weight_grad(), hidden, hidden));
        if (pushed == status_t::success_k)
            pushed = slots.try_push_back(
                make_slot(layer.gate_up_weight(), layer.gate_up_weight_grad(), 2u * ffn, hidden));
        if (pushed == status_t::success_k)
            pushed = slots.try_push_back(make_slot(layer.down_weight(), layer.down_weight_grad(), hidden, ffn));
        if (pushed == status_t::success_k)
            pushed = slots.try_push_back(make_slot(layer.pre_norm_scale(), layer.pre_norm_scale_grad(), 1u, hidden));
        if (pushed == status_t::success_k)
            pushed = slots.try_push_back(make_slot(layer.post_norm_scale(), layer.post_norm_scale_grad(), 1u, hidden));
    }
    if (pushed == status_t::success_k)
        pushed = slots.try_push_back(make_slot(final_norm_scale_.span(), final_norm_scale_grad_.view(), 1u, hidden));
    if (pushed == status_t::success_k)
        pushed = slots.try_push_back(make_slot(pool_query_.span(), pool_query_grad_.view(), 1u, hidden));
    if (pushed == status_t::success_k)
        pushed = slots.try_push_back(make_slot(head_norm_scale_.span(), head_norm_scale_grad_.view(), 1u, hidden));
    if (pushed == status_t::success_k) pushed = slots.try_push_back({head_fc1_.span(), head_fc1_grad_.view()});
    if (pushed == status_t::success_k) pushed = slots.try_push_back({head_fc2_.span(), head_fc2_grad_.view()});
    if (pushed != status_t::success_k) return expected_t::fail(status_t::bad_alloc_k, "weight slot table overflowed");
    return {};
}

expected<encoder_tower_e4m3_t> encoder_tower_bf16_t::quantize_weights(gpu_arena_t &, gpu_stream_t const &) noexcept {
    return expected<encoder_tower_e4m3_t>::fail(status_t::shape_unsupported_k,
                                                "the E4M3 inference twin is not yet built");
}

expected_t encoder_tower_e4m3_t::forward(multimodal_batch_on_gpu_t const &, i16r2_view_t, bf16r2_span_t,
                                         gpu_stream_t const &) noexcept {
    return expected_t::fail(status_t::shape_unsupported_k, "the E4M3 inference forward is not yet built");
}
