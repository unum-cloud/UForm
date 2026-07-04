/**
 *  @brief Device kernels and heavy method bodies of the sharded sigmoid-pairwise segments contrastive loss
 *      declared in @ref contrastive_segments.cuh — the one translation unit behind its consumers.
 *  @file cuda_core/contrastive_segments.cu
 *  @author Ash Vardanian
 *  @date July 2, 2026
 */
#include "contrastive_segments.cuh"

using namespace unum::uform::core;

expected<matryoshka_prefixes_t> matryoshka_prefixes_t::make(u32r1_view_t prefix_dims, std::size_t hidden) noexcept {
    std::size_t const count = prefix_dims.extent(0);
    if (count == 0 || count > max_matryoshka_prefixes_k)
        return expected<matryoshka_prefixes_t>::fail(status_t::shape_unsupported_k,
                                                     "need 1..{} Matryoshka prefixes (got {})",
                                                     max_matryoshka_prefixes_k, count);
    matryoshka_prefixes_t prefixes;
    prefixes.count_ = static_cast<u32_t>(count);
    for (std::size_t prefix_index = 0; prefix_index != count; ++prefix_index) {
        u32_t const dim = prefix_dims[prefix_index];
        u32_t const previous = prefix_index != 0 ? prefix_dims[prefix_index - 1] : 0u;
        if (dim <= previous || dim > hidden)
            return expected<matryoshka_prefixes_t>::fail( //
                status_t::shape_unsupported_k,
                "prefix dims must ascend strictly within hidden (prefix={} dim={} previous={} hidden={})",
                prefix_index, dim, previous, hidden);
        prefixes.dims_[prefix_index] = dim;
    }
    return expected<matryoshka_prefixes_t> {std::move(prefixes)};
}

/**
 *  @brief Per-row L2 norms of every Matryoshka prefix in @b one pass over the hidden axis: the running
 *      sum-of-squares is block-reduced at each prefix boundary, so @c norms[prefix,row] = ‖row[:D_p]‖ .
 *
 *  __grid_size__ @c [rows,1,1] ; one CTA per row; @ref kernel_block_size_default_k threads stripe each
 *  prefix segment; no dynamic shared memory.
 */
static __global__ void contrastive_prefix_norms_kernel_(bf16r2_view_t embeddings, f32r2_span_t norms,
                                                        matryoshka_prefixes_t prefixes) {
    std::size_t const rows = embeddings.extent(0);
    std::size_t const row_index = blockIdx.x;
    if (row_index >= rows) return;
    f32_t running_sum_squares = 0.0f;
    for (u32_t prefix_index = 0u; prefix_index != prefixes.size(); ++prefix_index) {
        f32_t thread_partial = 0.0f;
        for (unsigned channel_index = prefixes.segment_begin(prefix_index) + threadIdx.x;
             channel_index < prefixes[prefix_index]; channel_index += blockDim.x) {
            f32_t const value = __bfloat162float(embeddings[row_index, channel_index]);
            thread_partial += value * value;
        }
        running_sum_squares += block_reduce_broadcast<kernel_block_size_default_k>(thread_partial);
        if (threadIdx.x == 0u) norms[prefix_index, row_index] = sqrtf(running_sum_squares);
    }
}

/**
 *  @brief The similarity scaling of one subgraph block: @c similarity[p,v,n] = β_v·raw/(‖v‖·‖n‖) with the
 *      floored denominator — the epilogue of the tensor-core dot products.
 *
 *  __grid_size__ @c [ceil(K·R·Rc/block),1,1] ; grid-stride over cells; no dynamic shared memory.
 */
static __global__ void contrastive_similarity_kernel_(f32r3_view_t raw_dot, f32r2_view_t vertex_norms,
                                                      f32r2_view_t neighbor_norms, contrastive_recipe_t recipe,
                                                      f32r3_span_t similarity) {
    std::size_t const num_vertices = raw_dot.extent(1);
    std::size_t const num_neighbors = raw_dot.extent(2);
    std::size_t const total = raw_dot.extent(0) * num_vertices * num_neighbors;
    std::size_t const stride = static_cast<std::size_t>(gridDim.x) * blockDim.x;
    for (std::size_t cell = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x; cell < total;
         cell += stride) {
        std::size_t const prefix_index = cell / (num_vertices * num_neighbors);
        std::size_t const vertex_index = (cell / num_neighbors) % num_vertices;
        std::size_t const neighbor_index = cell % num_neighbors;
        f32_t const denominator =
            vertex_norms[prefix_index, vertex_index] * neighbor_norms[prefix_index, neighbor_index];
        f32_t const cosine = denominator > cosine_denominator_floor_k
                                 ? raw_dot[prefix_index, vertex_index, neighbor_index] / denominator
                                 : 0.0f;
        similarity[prefix_index, vertex_index, neighbor_index] =
            contrastive_vertex_beta_(recipe, static_cast<u32_t>(vertex_index)) * cosine;
    }
}

/**
 *  @brief Signed per-vertex weight totals @c Σw⁺(i) / @c Σw⁻(i) over one subgraph — the similarity-free
 *      pre-pass supplying the balanced-reduction denominators, resolved pair by pair through the relation
 *      graph. Run once per subgraph; the row totals accumulate serially on the stream — no atomics.
 *
 *  __grid_size__ @c [num_vertices,1,1] ; one CTA per vertex row; threads stripe the subgraph's rows; no
 *  dynamic shared memory.
 */
static __global__ void contrastive_signed_weight_sum_kernel_(f32r1_span_t pos_weight_row, f32r1_span_t neg_weight_row,
                                                             contrastive_relations_view_t relations,
                                                             std::size_t num_neighbors, std::size_t neighbor_base) {
    std::size_t const num_vertices = pos_weight_row.extent(0);
    std::size_t const vertex_index = blockIdx.x;
    if (vertex_index >= num_vertices) return;
    f32_t thread_pos = 0.0f, thread_neg = 0.0f;
    for (std::size_t neighbor_local = threadIdx.x; neighbor_local < num_neighbors; neighbor_local += blockDim.x) {
        contrastive_pair_relation_t const relation = contrastive_pair_relation(
            relations, static_cast<u32_t>(vertex_index), static_cast<u32_t>(neighbor_base + neighbor_local));
        if (relation.target > 0.0f) thread_pos += relation.weight;
        else if (relation.target < 0.0f) thread_neg += relation.weight;
    }
    f32_t const block_pos = block_reduce_broadcast<kernel_block_size_default_k>(thread_pos);
    f32_t const block_neg = block_reduce_broadcast<kernel_block_size_default_k>(thread_neg);
    if (threadIdx.x == 0u) {
        pos_weight_row[vertex_index] += block_pos;
        neg_weight_row[vertex_index] += block_neg;
    }
}

/**
 *  @brief NaN-safe balanced weighted sigmoid-BCE over one subgraph block, all prefixes fused: writes the
 *      @b final per-vertex-scaled @c dL/d(logit) and accumulates the four per-vertex rows (softplus
 *      masses, bias and log-β gradient summands). A non-finite @c t·logit leaves the pair's gradient
 *      zero; a @c w==0 pair is skipped in place.
 *
 *  __grid_size__ @c [num_vertices,1,1] ; one CTA per vertex row; the prefix loop lives inside the CTA and
 *  the threads stripe the subgraph's rows, so the row accumulators stay race-free; no dynamic shared
 *  memory.
 */
static __global__ void contrastive_bce_balanced_kernel_(
    f32r3_view_t similarity, f32r3_span_t logit_grad, f32r1_view_t pos_weight_row, f32r1_view_t neg_weight_row,
    f32r1_span_t pos_loss_row, f32r1_span_t neg_loss_row, f32r1_span_t bias_grad_row,
    f32r1_span_t log_inverse_temperature_grad_row, contrastive_relations_view_t relations, contrastive_recipe_t recipe,
    f32_t vertices_per_mesh, std::size_t neighbor_base) {
    std::size_t const num_vertices = pos_loss_row.extent(0);
    std::size_t const num_neighbors = similarity.extent(2);
    std::size_t const prefix_count = similarity.extent(0);
    std::size_t const vertex_index = blockIdx.x;
    if (vertex_index >= num_vertices) return;
    f32_t const pos_scale = 1.0f / (fmaxf(pos_weight_row[vertex_index], weight_sum_floor_k) * vertices_per_mesh);
    f32_t const neg_scale = 1.0f / (fmaxf(neg_weight_row[vertex_index], weight_sum_floor_k) * vertices_per_mesh);
    f32_t const bias = contrastive_vertex_bias_(recipe, static_cast<u32_t>(vertex_index));
    f32_t thread_pos_loss = 0.0f, thread_neg_loss = 0.0f;
    f32_t thread_bias_grad = 0.0f, thread_log_beta_grad = 0.0f;
    for (std::size_t prefix_index = 0; prefix_index != prefix_count; ++prefix_index) {
        for (std::size_t neighbor_local = threadIdx.x; neighbor_local < num_neighbors; neighbor_local += blockDim.x) {
            contrastive_pair_relation_t const relation = contrastive_pair_relation(
                relations, static_cast<u32_t>(vertex_index), static_cast<u32_t>(neighbor_base + neighbor_local));
            if (relation.weight == 0.0f) {
                logit_grad[prefix_index, vertex_index, neighbor_local] = 0.0f;
                continue;
            }
            f32_t const logit = similarity[prefix_index, vertex_index, neighbor_local] + bias;
            f32_t const signed_logit = relation.target * logit;
            f32_t cell_grad = 0.0f;
            if (isfinite(signed_logit)) {
                // Numerically-stable sigmoid + softplus, weighted by the per-cell weight and scaled by
                // the sign's own balanced denominator.

                f32_t sigmoid_value;
                if (signed_logit >= 0.0f) sigmoid_value = 1.0f / (1.0f + expf(-signed_logit));
                else {
                    f32_t const exp_positive = expf(signed_logit);
                    sigmoid_value = exp_positive / (1.0f + exp_positive);
                }
                f32_t const negated = -signed_logit;
                f32_t const softplus = fmaxf(negated, 0.0f) + log1pf(expf(-fabsf(negated)));
                bool const is_positive = relation.target > 0.0f;
                if (is_positive) thread_pos_loss += relation.weight * softplus;
                else thread_neg_loss += relation.weight * softplus;
                cell_grad = relation.weight * relation.target * (sigmoid_value - 1.0f) *
                            (is_positive ? pos_scale : neg_scale);
                thread_bias_grad += cell_grad;
                thread_log_beta_grad += cell_grad * similarity[prefix_index, vertex_index, neighbor_local];
            }
            logit_grad[prefix_index, vertex_index, neighbor_local] = cell_grad;
        }
    }
    f32_t const block_pos_loss = block_reduce_broadcast<kernel_block_size_default_k>(thread_pos_loss);
    f32_t const block_neg_loss = block_reduce_broadcast<kernel_block_size_default_k>(thread_neg_loss);
    f32_t const block_bias_grad = block_reduce_broadcast<kernel_block_size_default_k>(thread_bias_grad);
    f32_t const block_log_beta_grad = block_reduce_broadcast<kernel_block_size_default_k>(thread_log_beta_grad);
    if (threadIdx.x == 0u) {
        pos_loss_row[vertex_index] += block_pos_loss;
        neg_loss_row[vertex_index] += block_neg_loss;
        bias_grad_row[vertex_index] += block_bias_grad;
        log_inverse_temperature_grad_row[vertex_index] += block_log_beta_grad;
    }
}

/**
 *  @brief The raw-dot gradient of one subgraph block: @c raw_grad[p,v,n] = β_v·logit_grad/max(‖v‖·‖n‖,floor)
 *      — the operand the gradient matmuls and the self-coefficient folds both consume.
 *
 *  __grid_size__ @c [ceil(K·R·Rc/block),1,1] ; grid-stride over cells; no dynamic shared memory.
 */
static __global__ void contrastive_raw_grad_kernel_(f32r3_view_t logit_grad, f32r2_view_t vertex_norms,
                                                    f32r2_view_t neighbor_norms, contrastive_recipe_t recipe,
                                                    f32r3_span_t raw_grad) {
    std::size_t const num_vertices = logit_grad.extent(1);
    std::size_t const num_neighbors = logit_grad.extent(2);
    std::size_t const total = logit_grad.extent(0) * num_vertices * num_neighbors;
    std::size_t const stride = static_cast<std::size_t>(gridDim.x) * blockDim.x;
    for (std::size_t cell = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x; cell < total;
         cell += stride) {
        std::size_t const prefix_index = cell / (num_vertices * num_neighbors);
        std::size_t const vertex_index = (cell / num_neighbors) % num_vertices;
        std::size_t const neighbor_index = cell % num_neighbors;
        f32_t const denominator =
            vertex_norms[prefix_index, vertex_index] * neighbor_norms[prefix_index, neighbor_index];
        raw_grad[prefix_index, vertex_index, neighbor_index] =
            contrastive_vertex_beta_(recipe, static_cast<u32_t>(vertex_index)) *
            logit_grad[prefix_index, vertex_index, neighbor_index] / fmaxf(denominator, cosine_denominator_floor_k);
    }
}

/**
 *  @brief Deterministic per-vertex cosine self-coefficients of one subgraph block:
 *      @c coefficients[prefix,vertex] = Σ_j raw_grad·raw_dot — a fixed-order block reduce per row.
 *
 *  __grid_size__ @c [num_vertices,num_prefixes,1] ; threads stripe the subgraph's rows; no dynamic shared
 *  memory.
 */
static __global__ void contrastive_vertex_self_coefficients_kernel_(f32r3_view_t raw_grad, f32r3_view_t raw_dot,
                                                                    f32r2_span_t vertex_self_coefficients) {
    std::size_t const num_vertices = raw_grad.extent(1);
    std::size_t const num_neighbors = raw_grad.extent(2);
    std::size_t const vertex_index = blockIdx.x;
    std::size_t const prefix_index = blockIdx.y;
    if (vertex_index >= num_vertices) return;
    f32_t thread_partial = 0.0f;
    for (std::size_t neighbor_local = threadIdx.x; neighbor_local < num_neighbors; neighbor_local += blockDim.x)
        thread_partial += raw_grad[prefix_index, vertex_index, neighbor_local] *
                          raw_dot[prefix_index, vertex_index, neighbor_local];
    f32_t const total = block_reduce_broadcast<kernel_block_size_default_k>(thread_partial);
    if (threadIdx.x == 0u) vertex_self_coefficients[prefix_index, vertex_index] = total;
}

/**
 *  @brief Deterministic per-neighbor cosine self-coefficients — the column-sum sibling of
 *      @ref contrastive_vertex_self_coefficients_kernel_ .
 *
 *  __grid_size__ @c [count,num_prefixes,1] ; threads stripe the vertex rows; no dynamic shared memory.
 */
static __global__ void contrastive_neighbor_self_coefficients_kernel_(f32r3_view_t raw_grad, f32r3_view_t raw_dot,
                                                                      f32r2_span_t neighbor_self_coefficients) {
    std::size_t const num_vertices = raw_grad.extent(1);
    std::size_t const num_neighbors = raw_grad.extent(2);
    std::size_t const neighbor_local = blockIdx.x;
    std::size_t const prefix_index = blockIdx.y;
    if (neighbor_local >= num_neighbors) return;
    f32_t thread_partial = 0.0f;
    for (std::size_t vertex_index = threadIdx.x; vertex_index < num_vertices; vertex_index += blockDim.x)
        thread_partial += raw_grad[prefix_index, vertex_index, neighbor_local] *
                          raw_dot[prefix_index, vertex_index, neighbor_local];
    f32_t const total = block_reduce_broadcast<kernel_block_size_default_k>(thread_partial);
    if (threadIdx.x == 0u) neighbor_self_coefficients[prefix_index, neighbor_local] = total;
}

/**
 *  @brief Suffix-accumulate the per-prefix raw-dot gradients @b in @b place, turning plane @c p into
 *      @c G_p = Σ_{p'>=p} raw_grad_{p'} — the per-segment matmul operand (segment @c s is contained in
 *      exactly the prefixes @c p>=s ).
 *
 *  __grid_size__ @c [ceil(R·Rn/block),1,1] ; each thread owns one (vertex, neighbor) pair and walks the
 *  prefixes high-to-low serially — fixed order, no atomics; no dynamic shared memory.
 */
static __global__ void contrastive_suffix_accumulate_kernel_(f32r3_span_t raw_grad) {
    std::size_t const prefix_count = raw_grad.extent(0);
    std::size_t const num_neighbors = raw_grad.extent(2);
    std::size_t const cells_per_prefix = raw_grad.extent(1) * num_neighbors;
    std::size_t const stride = static_cast<std::size_t>(gridDim.x) * blockDim.x;
    for (std::size_t cell = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x; cell < cells_per_prefix;
         cell += stride) {
        std::size_t const vertex_index = cell / num_neighbors;
        std::size_t const neighbor_index = cell % num_neighbors;
        for (std::size_t prefix_index = prefix_count - 1u; prefix_index != 0u; --prefix_index)
            raw_grad[prefix_index - 1u, vertex_index, neighbor_index] +=
                raw_grad[prefix_index, vertex_index, neighbor_index];
    }
}

/**
 *  @brief The owned-vertex cosine self-fold of one subgraph:
 *      @c vertices_grad_f32(q,ch) -= Σ_p 1{ch<D_p}·coefficients[p,q]/max(‖q‖²,floor)·q(q,ch) — the term the
 *      cross-product matmuls cannot express.
 *
 *  __grid_size__ @c [num_vertices,1,1] ; threads stripe the hidden channels, prefixes loop inside; no
 *  dynamic shared memory.
 */
static __global__ void contrastive_vertex_self_fold_kernel_(bf16r2_view_t owned_vertices, f32r2_view_t vertex_norms,
                                                            f32r2_view_t vertex_self_coefficients,
                                                            f32r2_span_t vertices_grad_f32,
                                                            matryoshka_prefixes_t prefixes) {
    std::size_t const num_vertices = owned_vertices.extent(0);
    std::size_t const hidden = owned_vertices.extent(1);
    std::size_t const vertex_index = blockIdx.x;
    if (vertex_index >= num_vertices) return;
    for (unsigned channel_index = threadIdx.x; channel_index < hidden; channel_index += blockDim.x) {
        f32_t fold = 0.0f;
        for (u32_t prefix_index = 0u; prefix_index != prefixes.size(); ++prefix_index) {
            if (channel_index >= prefixes[prefix_index]) continue;
            f32_t const vertex_norm = vertex_norms[prefix_index, vertex_index];
            fold += vertex_self_coefficients[prefix_index, vertex_index] /
                    fmaxf(vertex_norm * vertex_norm, cosine_denominator_floor_k);
        }
        vertices_grad_f32[vertex_index, channel_index] -=
            fold * __bfloat162float(owned_vertices[vertex_index, channel_index]);
    }
}

/**
 *  @brief The neighbor-side cosine self-fold — the column sibling of
 *      @ref contrastive_vertex_self_fold_kernel_ , folded into the subgraph's @c vertices_grad .
 *
 *  __grid_size__ @c [count,1,1] ; threads stripe the hidden channels, prefixes loop inside; no dynamic
 *  shared memory.
 */
static __global__ void contrastive_neighbor_self_fold_kernel_(bf16r2_view_t neighbors, f32r2_view_t neighbor_norms,
                                                              f32r2_view_t neighbor_self_coefficients,
                                                              f32r2_span_t vertices_grad,
                                                              matryoshka_prefixes_t prefixes) {
    std::size_t const count = neighbors.extent(0);
    std::size_t const hidden = neighbors.extent(1);
    std::size_t const neighbor_local = blockIdx.x;
    if (neighbor_local >= count) return;
    for (unsigned channel_index = threadIdx.x; channel_index < hidden; channel_index += blockDim.x) {
        f32_t fold = 0.0f;
        for (u32_t prefix_index = 0u; prefix_index != prefixes.size(); ++prefix_index) {
            if (channel_index >= prefixes[prefix_index]) continue;
            f32_t const neighbor_norm = neighbor_norms[prefix_index, neighbor_local];
            fold += neighbor_self_coefficients[prefix_index, neighbor_local] /
                    fmaxf(neighbor_norm * neighbor_norm, cosine_denominator_floor_k);
        }
        vertices_grad[neighbor_local, channel_index] -=
            fold * __bfloat162float(neighbors[neighbor_local, channel_index]);
    }
}

/**
 *  @brief Balanced finalise: the rank-local loss summand plus the optional learnable-temperature / bias
 *      gradient summands scattered @b per pair-type slot — fixed-order block reduces over the per-vertex
 *      rows.
 *
 *  __grid_size__ @c [1,1,1] ; one CTA, threads stripe the vertex rows once per slot; no dynamic shared
 *  memory.
 */
static __global__ void contrastive_finalize_kernel_(
    f32r1_view_t pos_loss_row, f32r1_view_t neg_loss_row, f32r1_view_t pos_weight_row, f32r1_view_t neg_weight_row,
    f32r1_view_t bias_grad_row, f32r1_view_t log_inverse_temperature_grad_row, contrastive_recipe_t recipe,
    f32r1_span_t loss_out, f32r1_span_t bias_grad, f32r1_span_t log_inverse_temperature_grad, f32_t vertices_per_mesh) {
    std::size_t const num_vertices = pos_loss_row.extent(0);
    f32_t thread_loss = 0.0f;
    for (std::size_t vertex_index = threadIdx.x; vertex_index < num_vertices; vertex_index += blockDim.x)
        thread_loss += (pos_loss_row[vertex_index] / fmaxf(pos_weight_row[vertex_index], weight_sum_floor_k) +
                        neg_loss_row[vertex_index] / fmaxf(neg_weight_row[vertex_index], weight_sum_floor_k)) /
                       vertices_per_mesh;
    f32_t const total_loss = block_reduce_broadcast<kernel_block_size_default_k>(thread_loss);
    if (threadIdx.x == 0u) loss_out[0] = total_loss;

    bool const wants_bias = !bias_grad.empty();
    bool const wants_log_beta = !log_inverse_temperature_grad.empty();
    if (!wants_bias && !wants_log_beta) return;
    for (u32_t slot = 0; slot != recipe.pair_type_count(); ++slot) {
        f32_t thread_bias_grad = 0.0f, thread_log_beta_grad = 0.0f;
        for (std::size_t vertex_index = threadIdx.x; vertex_index < num_vertices; vertex_index += blockDim.x) {
            u32_t const vertex_slot = recipe.vertex_pair_type.empty()
                                          ? 0u
                                          : recipe.vertex_pair_type[static_cast<u32_t>(vertex_index)];
            if (vertex_slot != slot) continue;
            thread_bias_grad += bias_grad_row[vertex_index];
            thread_log_beta_grad += log_inverse_temperature_grad_row[vertex_index];
        }
        f32_t const slot_bias_grad = block_reduce_broadcast<kernel_block_size_default_k>(thread_bias_grad);
        f32_t const slot_log_beta_grad = block_reduce_broadcast<kernel_block_size_default_k>(thread_log_beta_grad);
        if (threadIdx.x == 0u) {
            if (wants_bias) bias_grad[slot] = slot_bias_grad;
            if (wants_log_beta) log_inverse_temperature_grad[slot] = slot_log_beta_grad;
        }
    }
}

/**
 *  @brief Elementwise Float32 → BFloat16 cast over flattened towers.
 *
 *  __grid_size__ @c [ceil(count/block),1,1] ; grid-stride; no dynamic shared memory.
 */
static __global__ void contrastive_cast_f32_to_bf16_kernel_(f32r1_view_t source, bf16r1_span_t destination) {
    std::size_t const count = source.extent(0);
    std::size_t const stride = static_cast<std::size_t>(gridDim.x) * blockDim.x;
    for (std::size_t index = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x; index < count;
         index += stride)
        destination[index] = __float2bfloat16(source[index]);
}

expected<contrastive_segments_t> contrastive_segments_t::make(gpu_arena_t &arena, cublaslt_session_t const &blas,
                                                              contrastive_segments_config_t config,
                                                              u32r1_view_t prefix_dims) noexcept {
    if (config.vertices_per_gpu == 0 || config.hidden == 0)
        return expected<contrastive_segments_t>::fail(
            status_t::shape_unsupported_k, "the segments loss scores pooled rows (vertices={} hidden={})",
            config.vertices_per_gpu, config.hidden);
    if (config.neighbors_per_gpu == 0 || config.gpu_subgraphs_per_step == 0)
        return expected<contrastive_segments_t>::fail(status_t::invalid_input_k,
                                                      "need at least one subgraph of at least one vertex");
    expected<matryoshka_prefixes_t> prefixes = matryoshka_prefixes_t::make(prefix_dims, config.hidden);
    if (!prefixes) return expected<contrastive_segments_t>::fail(std::move(prefixes));

    contrastive_segments_t loss;
    loss.prefixes_ = prefixes.value_;
    loss.config_ = config;

    // Scratch tensors — the former "workspace", owned like the relation columns are: plain
    // `try_empty` allocations, checked in named groups so a failure names the arena pressure.

    std::size_t const rows = config.vertices_per_gpu, hidden = config.hidden;
    u32_t const neighbors_per_gpu = config.neighbors_per_gpu;
    std::size_t const prefix_count = loss.prefixes_.size();
    std::size_t const block_cells = prefix_count * rows * neighbors_per_gpu;
    gpu_arena_alloc<f32_t> f32_alloc {arena};
    gpu_arena_alloc<bf16_t> bf16_alloc {arena};
    gpu_arena_alloc<b8_t> byte_alloc {arena};
    loss.vertex_norms_ = f32r2_tensor_arena_t::try_empty({prefix_count, rows}, f32_alloc);
    loss.neighbor_norms_ = f32r2_tensor_arena_t::try_empty({prefix_count, neighbors_per_gpu}, f32_alloc);
    loss.running_dots_ = f32r2_tensor_arena_t::try_empty({rows, neighbors_per_gpu}, f32_alloc);
    loss.raw_dot_ = f32r3_tensor_arena_t::try_empty({prefix_count, rows, neighbors_per_gpu}, f32_alloc);
    loss.similarity_ = f32r3_tensor_arena_t::try_empty({prefix_count, rows, neighbors_per_gpu}, f32_alloc);
    loss.logit_grad_ = f32r3_tensor_arena_t::try_empty({prefix_count, rows, neighbors_per_gpu}, f32_alloc);
    loss.raw_grad_ = f32r3_tensor_arena_t::try_empty({prefix_count, rows, neighbors_per_gpu}, f32_alloc);
    loss.raw_grad_bf16_ = bf16r3_tensor_arena_t::try_empty({prefix_count, rows, neighbors_per_gpu}, bf16_alloc);
    loss.vertex_accumulator_rows_ = f32r2_tensor_arena_t::try_empty({6, rows}, f32_alloc);
    loss.vertex_self_coefficients_ = f32r2_tensor_arena_t::try_empty({prefix_count, rows}, f32_alloc);
    loss.neighbor_self_coefficients_ = f32r2_tensor_arena_t::try_empty({prefix_count, neighbors_per_gpu}, f32_alloc);
    loss.vertices_grad_f32_ = f32r2_tensor_arena_t::try_empty({rows, hidden}, f32_alloc);
    if (!loss.vertex_norms_ || !loss.neighbor_norms_ || !loss.running_dots_ || !loss.raw_dot_ ||
        !loss.similarity_ || !loss.logit_grad_ || !loss.raw_grad_ || !loss.raw_grad_bf16_ ||
        !loss.vertex_accumulator_rows_ || !loss.vertex_self_coefficients_ || !loss.neighbor_self_coefficients_ ||
        !loss.vertices_grad_f32_)
        return expected<contrastive_segments_t>::fail(status_t::bad_alloc_k,
                                                      "loss scratch allocation failed ({} cells, arena_used={})",
                                                      block_cells, arena.used());

    // One pinned plan per (segment, product family), against the borrowed session.
    loss.session_ = &blas;
    std::size_t workspace_bytes = 0;
    for (u32_t segment = 0; segment != loss.prefixes_.size(); ++segment) {
        u32_t const width = loss.prefixes_.segment_width(segment);
        u32_t const rows32 = static_cast<u32_t>(rows), hidden32 = static_cast<u32_t>(hidden);
        // dots: running[R,Rn] += V_seg[R,W]·(N_seg[Rn,W])ᵀ

        cublaslt_plan_config_t dots_config {rows32,   width,  hidden32,          neighbors_per_gpu, width,
                                            hidden32, rows32, neighbors_per_gpu, neighbors_per_gpu, false,
                                            true};
        // owned-vertex grad: vertices_grad_f32[:,seg][R,W] += G_seg[R,Rn]·N_seg[Rn,W]

        cublaslt_plan_config_t vertex_grad_config {rows32,
                                                   neighbors_per_gpu,
                                                   neighbors_per_gpu,
                                                   neighbors_per_gpu,
                                                   width,
                                                   hidden32,
                                                   rows32,
                                                   width,
                                                   hidden32,
                                                   false,
                                                   false};
        // subgraph-vertex grad: subgraph.vertices_grad[:,seg][Rn,W] += (G_seg[R,Rn])ᵀ·V_seg[R,W]

        cublaslt_plan_config_t neighbor_grad_config {rows32,
                                                     neighbors_per_gpu,
                                                     neighbors_per_gpu,
                                                     rows32,
                                                     width,
                                                     hidden32,
                                                     neighbors_per_gpu,
                                                     width,
                                                     hidden32,
                                                     true,
                                                     false};
        cublaslt_plan_config_t const configs[3] = {dots_config, vertex_grad_config, neighbor_grad_config};
        for (u32_t family = 0; family != 3u; ++family) {
            expected<cublaslt_plan_t> plan = cublaslt_plan_t::make(*loss.session_, configs[family]);
            if (!plan) return expected<contrastive_segments_t>::fail(std::move(plan));
            if (plan.value_.workspace_bytes() > workspace_bytes) workspace_bytes = plan.value_.workspace_bytes();
            loss.plans_[3u * segment + family] = std::move(plan.value_);
        }
    }
    if (workspace_bytes != 0) {
        loss.cublaslt_workspace_ = b8r1_tensor_arena_t::try_empty({workspace_bytes}, byte_alloc);
        if (!loss.cublaslt_workspace_)
            return expected<contrastive_segments_t>::fail(
                status_t::bad_alloc_k, "tensor-core workspace allocation failed ({} bytes)", workspace_bytes);
    }
    return expected<contrastive_segments_t> {std::move(loss)};
}

expected_t contrastive_segments_t::operator()(bf16r2_view_t vertices, nk::tensor_view<gpu_subgraph_t, 1> gpu_subgraphs,
                                              contrastive_relations_view_t relations, contrastive_recipe_t recipe,
                                              std::size_t vertices_per_mesh, f32r1_span_t loss_out,
                                              bf16r2_span_t vertices_grad, gpu_stream_t const &stream,
                                              f32r1_span_t log_inverse_temperature_grad,
                                              f32r1_span_t bias_grad) noexcept {

    if (vertices.extent(0) != config_.vertices_per_gpu || vertices.extent(1) != config_.hidden)
        return expected_t::fail(status_t::shape_unsupported_k,
                                "vertex tower [{}x{}] disagrees with the baked [{}x{}]", vertices.extent(0),
                                vertices.extent(1), config_.vertices_per_gpu, config_.hidden);
    if (gpu_subgraphs.extent(0) != config_.gpu_subgraphs_per_step)
        return expected_t::fail(status_t::shape_unsupported_k, "{} gpu_subgraphs staged, {} baked",
                                gpu_subgraphs.extent(0), config_.gpu_subgraphs_per_step);
    if (loss_out.empty())
        return expected_t::fail(status_t::invalid_input_k, "a one-element loss_out span is required");
    if (vertices_grad.extent(0) != config_.vertices_per_gpu || vertices_grad.extent(1) != config_.hidden)
        return expected_t::fail(status_t::shape_unsupported_k,
                                "vertices_grad shape disagrees with the baked [{}x{}]", config_.vertices_per_gpu,
                                config_.hidden);
    if (vertices_per_mesh == 0)
        return expected_t::fail(status_t::invalid_input_k, "vertices_per_mesh must be non-zero");
    if (relations.vertices_per_gpu() != config_.vertices_per_gpu)
        return expected_t::fail(status_t::shape_unsupported_k,
                                "the relation graph describes {} vertices, the loss was baked for {}",
                                relations.vertices_per_gpu(), config_.vertices_per_gpu);
    if (!bias_grad.empty() && bias_grad.extent(0) < recipe.pair_type_count())
        return expected_t::fail(status_t::shape_unsupported_k, "bias_grad must span the {} pair-type slots",
                                recipe.pair_type_count());
    if (!log_inverse_temperature_grad.empty() && log_inverse_temperature_grad.extent(0) < recipe.pair_type_count())
        return expected_t::fail(status_t::shape_unsupported_k,
                                "log_inverse_temperature_grad must span the {} pair-type slots",
                                recipe.pair_type_count());
    std::size_t const neighbors_per_mesh = static_cast<std::size_t>(config_.neighbors_per_gpu) *
                                           config_.gpu_subgraphs_per_step;
    for (std::size_t subgraph_index = 0; subgraph_index != gpu_subgraphs.extent(0); ++subgraph_index) {
        gpu_subgraph_t const subgraph = gpu_subgraphs[subgraph_index];
        if (subgraph.vertices.extent(0) != config_.neighbors_per_gpu ||
            subgraph.vertices.extent(1) != config_.hidden ||
            subgraph.vertices_grad.extent(0) != config_.neighbors_per_gpu ||
            subgraph.vertices_grad.extent(1) != config_.hidden)
            return expected_t::fail(status_t::shape_unsupported_k, "subgraph {} disagrees with the baked [{}x{}]",
                                    subgraph_index, config_.neighbors_per_gpu, config_.hidden);
        if (subgraph.global_offset != subgraph_index * static_cast<std::size_t>(config_.neighbors_per_gpu))
            return expected_t::fail(status_t::shape_unsupported_k,
                                    "subgraphs must tile the global vertex axis in order (offset={} expected={})",
                                    subgraph.global_offset,
                                    subgraph_index * static_cast<std::size_t>(config_.neighbors_per_gpu));
    }
    if (!relations.has_explicit_edges() && neighbors_per_mesh < relations.self_edge_base + config_.vertices_per_gpu)
        return expected_t::fail( //
            status_t::shape_unsupported_k,
            "self-edges need neighbors_per_mesh>=self_edge_base+vertices_per_gpu " //
            "(neighbors={} base={} vertices={})",
            neighbors_per_mesh, relations.self_edge_base, config_.vertices_per_gpu);

    // Zero the accumulators: the six contiguous per-vertex rows and the Float32 owned-vertex force.
    if (CUresult result =
            cuMemsetD8Async(reinterpret_cast<CUdeviceptr>(vertex_accumulator_rows_.data()), 0,
                            vertex_accumulator_rows_.numel() * sizeof(f32_t), stream.get());
        result != CUDA_SUCCESS)
        return expected_t::fail(result, "accumulator-row reset failed");
    if (CUresult result = cuMemsetD8Async(reinterpret_cast<CUdeviceptr>(vertices_grad_f32_.data()), 0,
                                          vertices_grad_f32_.numel() * sizeof(f32_t), stream.get());
        result != CUDA_SUCCESS)
        return expected_t::fail(result, "vertex-force accumulator reset failed");

    f32_t const vertices_per_mesh_float = static_cast<f32_t>(vertices_per_mesh);
    u32_t const prefix_count = prefixes_.size();
    std::size_t const cells_per_prefix = static_cast<std::size_t>(config_.vertices_per_gpu) *
                                         config_.neighbors_per_gpu;
    std::size_t const block_cells = static_cast<std::size_t>(prefix_count) * cells_per_prefix;
    dim3 const block_shape {kernel_block_size_default_k, 1u, 1u};
    dim3 const vertex_grid {config_.vertices_per_gpu, 1u, 1u};
    dim3 const neighbor_grid {config_.neighbors_per_gpu, 1u, 1u};
    dim3 const block_grid {grid_size_for(block_cells), 1u, 1u};
    b8r1_span_t const cublaslt_workspace = cublaslt_workspace_.span();

    // The scratch views the read-only kernel parameters bind to; the launcher copies raw bytes, so
    // spans convert to views here, explicitly. The six accumulator rows are consecutive rows of one
    // [6,vertices_per_gpu] tensor, so a single memset covers them all.

    std::size_t const row_width = vertex_accumulator_rows_.extent(1);
    f32r1_span_t const pos_weight_row {vertex_accumulator_rows_.row(0).data(), row_width};
    f32r1_span_t const neg_weight_row {vertex_accumulator_rows_.row(1).data(), row_width};
    f32r1_span_t const pos_loss_row {vertex_accumulator_rows_.row(2).data(), row_width};
    f32r1_span_t const neg_loss_row {vertex_accumulator_rows_.row(3).data(), row_width};
    f32r1_span_t const bias_grad_row {vertex_accumulator_rows_.row(4).data(), row_width};
    f32r1_span_t const log_beta_grad_row {vertex_accumulator_rows_.row(5).data(), row_width};
    f32r2_view_t const vertex_norms_view = vertex_norms_.view();
    f32r2_view_t const neighbor_norms_view = neighbor_norms_.view();
    f32r3_view_t const raw_dot_view = raw_dot_.view();
    f32r3_view_t const similarity_view = similarity_.view();
    f32r3_view_t const logit_grad_view = logit_grad_.view();
    f32r3_view_t const raw_grad_view = raw_grad_.view();
    f32r2_view_t const vertex_self_view = vertex_self_coefficients_.view();
    f32r2_view_t const neighbor_self_view = neighbor_self_coefficients_.view();
    std::size_t const num_neighbors = config_.neighbors_per_gpu;

    // Per-prefix owned-vertex norms, once; each subgraph recomputes only its own vertex norms.

    expected_t launched = launch_kernel(reinterpret_cast<void const *>(&contrastive_prefix_norms_kernel_),
                                        vertex_grid, block_shape, stream, vertices, vertex_norms_.span(),
                                        prefixes_);
    if (!launched) return launched;

    // Signed Σw pre-pass over every subgraph: the per-vertex weight totals must span the whole
    // global vertex axis before any BCE block runs.
    for (std::size_t subgraph_index = 0; subgraph_index != gpu_subgraphs.extent(0); ++subgraph_index) {
        gpu_subgraph_t const subgraph = gpu_subgraphs[subgraph_index];
        launched = launch_kernel(reinterpret_cast<void const *>(&contrastive_signed_weight_sum_kernel_),
                                 vertex_grid, block_shape, stream, pos_weight_row, neg_weight_row, relations,
                                 num_neighbors, subgraph.global_offset);
        if (!launched) return launched;
    }

    for (std::size_t subgraph_index = 0; subgraph_index != gpu_subgraphs.extent(0); ++subgraph_index) {
        gpu_subgraph_t const subgraph = gpu_subgraphs[subgraph_index];

        launched = launch_kernel(reinterpret_cast<void const *>(&contrastive_prefix_norms_kernel_), neighbor_grid,
                                 block_shape, stream, subgraph.vertices, neighbor_norms_.span(), prefixes_);
        if (!launched) return launched;

        // Tensor-core dots per segment: the running block after segment p IS prefix p's raw dot. The
        // segment column offsets feed cuBLASLt raw pointers directly (vendor boundary), so they stay
        // pointer arithmetic; the destination plane is the arena's leading (prefix) axis, so it's a row.
        for (u32_t segment = 0; segment != prefix_count; ++segment) {
            __nv_bfloat16 const *vertex_segment = vertices.data() + prefixes_.segment_begin(segment);
            __nv_bfloat16 const *neighbor_segment = subgraph.vertices.data() + prefixes_.segment_begin(segment);
            if (expected_t product = plans_[3 * segment].matmul(*session_, vertex_segment, neighbor_segment,
                                                                running_dots_.data(), segment == 0 ? 0.0f : 1.0f,
                                                                cublaslt_workspace, stream);
                !product)
                return product;
            if (CUresult result =
                    cuMemcpyDtoDAsync(reinterpret_cast<CUdeviceptr>(raw_dot_.row(segment).data()),
                                      reinterpret_cast<CUdeviceptr>(running_dots_.data()),
                                      cells_per_prefix * sizeof(f32_t), stream.get());
                result != CUDA_SUCCESS)
                return expected_t::fail(result, "prefix dot copy failed (segment={})", segment);
        }

        launched = launch_kernel(reinterpret_cast<void const *>(&contrastive_similarity_kernel_), block_grid,
                                 block_shape, stream, raw_dot_view, vertex_norms_view, neighbor_norms_view, recipe,
                                 similarity_.span());
        if (launched)
            launched = launch_kernel(reinterpret_cast<void const *>(&contrastive_bce_balanced_kernel_), vertex_grid,
                                     block_shape, stream, similarity_view, logit_grad_.span(),
                                     f32r1_view_t {pos_weight_row}, f32r1_view_t {neg_weight_row}, pos_loss_row,
                                     neg_loss_row, bias_grad_row, log_beta_grad_row, relations, recipe,
                                     vertices_per_mesh_float, subgraph.global_offset);
        if (launched)
            launched = launch_kernel(reinterpret_cast<void const *>(&contrastive_raw_grad_kernel_), block_grid,
                                     block_shape, stream, logit_grad_view, vertex_norms_view, neighbor_norms_view,
                                     recipe, raw_grad_.span());
        if (launched)
            launched = launch_kernel(reinterpret_cast<void const *>(&contrastive_vertex_self_coefficients_kernel_),
                                     dim3 {config_.vertices_per_gpu, prefix_count, 1u}, block_shape, stream,
                                     raw_grad_view, raw_dot_view, vertex_self_coefficients_.span());
        if (launched)
            launched = launch_kernel(reinterpret_cast<void const *>(&contrastive_neighbor_self_coefficients_kernel_),
                                     dim3 {config_.neighbors_per_gpu, prefix_count, 1u}, block_shape, stream,
                                     raw_grad_view, raw_dot_view, neighbor_self_coefficients_.span());
        if (launched && prefix_count > 1u)
            launched = launch_kernel(reinterpret_cast<void const *>(&contrastive_suffix_accumulate_kernel_),
                                     dim3 {grid_size_for(cells_per_prefix), 1u, 1u}, block_shape, stream,
                                     raw_grad_.span());
        if (launched)
            launched = launch_kernel(reinterpret_cast<void const *>(&contrastive_cast_f32_to_bf16_kernel_),
                                     block_grid, block_shape, stream, raw_grad_view.flatten<1>(),
                                     raw_grad_bf16_.span().flatten<1>());
        if (!launched) return launched;

        // Zero the subgraph's whole force so the segment matmuls can accumulate (beta=1) and the
        // channels beyond the last prefix stay exactly zero.
        if (CUresult result = cuMemsetD8Async(
                reinterpret_cast<CUdeviceptr>(subgraph.vertices_grad.data()), 0,
                static_cast<std::size_t>(config_.neighbors_per_gpu) * config_.hidden * sizeof(f32_t), stream.get());
            result != CUDA_SUCCESS)
            return expected_t::fail(result, "subgraph force reset failed (subgraph={})", subgraph_index);

        // Tensor-core gradients per segment: G_seg lives at the suffix-accumulated plane s, so its
        // pointer is now the arena's leading (prefix) row; the column offsets stay pointer arithmetic
        // (vendor boundary / external subgraph memory, same as the dots pass above).
        for (u32_t segment = 0; segment != prefix_count; ++segment) {
            __nv_bfloat16 const *gradient_segment = raw_grad_bf16_.row(segment).data();
            __nv_bfloat16 const *vertex_segment = vertices.data() + prefixes_.segment_begin(segment);
            __nv_bfloat16 const *neighbor_segment = subgraph.vertices.data() + prefixes_.segment_begin(segment);
            if (expected_t product =
                    plans_[3 * segment + 1].matmul(*session_, gradient_segment, neighbor_segment,
                                                   vertices_grad_f32_.data() + prefixes_.segment_begin(segment),
                                                   1.0f, cublaslt_workspace, stream);
                !product)
                return product;
            if (expected_t product = plans_[3 * segment + 2].matmul(
                    *session_, gradient_segment, vertex_segment,
                    subgraph.vertices_grad.data() + prefixes_.segment_begin(segment), 1.0f, cublaslt_workspace,
                    stream);
                !product)
                return product;
        }

        launched = launch_kernel(reinterpret_cast<void const *>(&contrastive_vertex_self_fold_kernel_), vertex_grid,
                                 block_shape, stream, vertices, vertex_norms_view, vertex_self_view,
                                 vertices_grad_f32_.span(), prefixes_);
        if (launched)
            launched = launch_kernel(reinterpret_cast<void const *>(&contrastive_neighbor_self_fold_kernel_),
                                     neighbor_grid, block_shape, stream, subgraph.vertices, neighbor_norms_view,
                                     neighbor_self_view, subgraph.vertices_grad, prefixes_);
        if (!launched) return launched;
    }

    launched = launch_kernel(reinterpret_cast<void const *>(&contrastive_finalize_kernel_), dim3 {1u, 1u, 1u},
                             block_shape, stream, f32r1_view_t {pos_loss_row}, f32r1_view_t {neg_loss_row},
                             f32r1_view_t {pos_weight_row}, f32r1_view_t {neg_weight_row},
                             f32r1_view_t {bias_grad_row}, f32r1_view_t {log_beta_grad_row}, recipe, loss_out,
                             bias_grad, log_inverse_temperature_grad, vertices_per_mesh_float);
    if (!launched) return launched;

    // Cast the owned-vertex force to BFloat16 — every vertex is owned by exactly one rank.

    std::size_t const vertices_grad_scalars = vertices_grad.numel();
    f32r1_view_t const vertices_grad_flat = vertices_grad_f32_.view().flatten<1>();
    bf16r1_span_t const vertices_grad_out = vertices_grad.flatten<1>();
    return launch_kernel(reinterpret_cast<void const *>(&contrastive_cast_f32_to_bf16_kernel_),
                         dim3 {grid_size_for(vertices_grad_scalars), 1u, 1u}, block_shape, stream,
                         vertices_grad_flat, vertices_grad_out);
}

/**
 *  @brief Fold a reduced Float32 column force into the caller's BFloat16 gradient in place:
 *      @c destination[i] = bf16(f32(destination[i]) + column[i]) — the row force already lives there.
 *
 *  __grid_size__ grid-stride over the flattened tower; no dynamic shared memory.
 */
static __global__ void contrastive_column_fold_kernel_(bf16r1_span_t destination, f32r1_view_t column_force) {
    std::size_t const scalars = destination.extent(0);
    std::size_t const stride = static_cast<std::size_t>(gridDim.x) * blockDim.x;
    for (std::size_t index = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x; index < scalars;
         index += stride)
        destination[index] = __float2bfloat16_rn(__fadd_rn(__bfloat162float(destination[index]), column_force[index]));
}

expected<distributed_contrastive_segments_t> distributed_contrastive_segments_t::make(
    multimem_domain_t &domain, colocated_domain_t &colocated, std::span<gpu_arena_t> arenas,
    std::span<gpu_stream_t const> streams, std::span<cublaslt_session_t const> sessions,
    std::span<bf16r2_view_t const> towers, std::span<contrastive_relations_view_t const> relations,
    std::span<bf16r2_span_t const> vertices_grads, contrastive_segments_config_t config, u32r1_view_t prefix_dims,
    f64_t timeout_seconds) noexcept {
    using result_t = expected<distributed_contrastive_segments_t>;
    std::size_t const gpus = arenas.size();
    u32_t const domains = colocated.multimem_domain_count();
    if (gpus == 0 || streams.size() != gpus || domain.member_count() != gpus || sessions.size() != gpus ||
        towers.size() != gpus || relations.size() != gpus || vertices_grads.size() != gpus)
        return result_t::fail(status_t::invalid_input_k,
                              "every rank needs an arena, stream, session, tower, relations, gradient, " //
                              "and domain member ({} ranks, {} members)",
                              gpus, domain.member_count());
    if (config.gpu_subgraphs_per_step != gpus * domains || config.neighbors_per_gpu != config.vertices_per_gpu)
        return result_t::fail(status_t::shape_unsupported_k,
                              "the self-scoring world scores one subgraph per GPU " //
                              "(subgraphs={} gpus={} domains={} neighbors={} vertices={})",
                              config.gpu_subgraphs_per_step, gpus, domains, config.neighbors_per_gpu,
                              config.vertices_per_gpu);
    mesh_shape_t const &shape = colocated.topology().shape;
    if (domains > 1 && shape.gpus_per_host_domain != shape.gpus_per_multimem_domain)
        return result_t::fail(status_t::shape_unsupported_k,
                              "one process must drive one multimem_domain (host={} multimem={})",
                              shape.gpus_per_host_domain, shape.gpus_per_multimem_domain);

    distributed_contrastive_segments_t distributed;
    distributed.config_ = config;
    distributed.domain_ = &domain;
    distributed.world_domains_ = domains;
    distributed.multimem_domain_index_ = domains > 1 ? colocated.topology().multimem_domain_index() : 0;
    std::size_t const subgraphs = config.gpu_subgraphs_per_step;
    std::size_t const tower_scalars = static_cast<std::size_t>(config.vertices_per_gpu) * config.hidden;
    if (distributed.losses_.try_resize(gpus) != status_t::success_k ||
        distributed.reduce_members_.try_resize(gpus) != status_t::success_k ||
        distributed.subgraph_tables_.try_resize(gpus * subgraphs) != status_t::success_k ||
        distributed.relations_.try_assign({relations.data(), gpus}) != status_t::success_k ||
        distributed.vertices_grads_.try_assign({vertices_grads.data(), gpus}) != status_t::success_k)
        return result_t::fail(status_t::bad_alloc_k, "rank table allocation failed ({} ranks)", gpus);
    // The caller's tower views are stored once: in the subgraph tables in a single-domain world,
    // in the gateway's gather-staging list across domains — never both.
    if (domains > 1u && distributed.gather_sources_.try_assign({towers.data(), gpus}) != status_t::success_k)
        return result_t::fail(status_t::bad_alloc_k, "gather source table allocation failed ({} ranks)", gpus);

    // Level-1 precondition: every arena readable by every device — visibility is the arenas'
    // construction-time configuration, validated here, never granted here.
    for (std::size_t rank = 0; rank != gpus; ++rank)
        for (std::size_t peer = 0; peer != gpus; ++peer)
            if (!arenas[rank].is_visible_to(static_cast<int>(peer)))
                return result_t::fail(status_t::invalid_input_k,
                                      "rank {}'s arena is not visible to device {} — " //
                                      "construct it with `visible_to`",
                                      rank, peer);
    expected<multimem_window_t> window = domain.reserve(payload_scalars_needed(config));
    if (!window) return result_t::fail(std::move(window));
    distributed.window_ = window.value_;

    // Every rank: the loss object and the cross-domain reduce member, both on its own device —
    // entered through the context the rank's stream already owns.
    for (std::size_t rank = 0; rank != gpus; ++rank) {
        switch_to_gpu_t gpu_scope {streams[rank]};
        if (!gpu_scope.ok()) return result_t::fail(status_t::cuda_error_k, "context entry failed (rank={})", rank);
        expected<contrastive_segments_t> loss = contrastive_segments_t::make(arenas[rank], sessions[rank], config,
                                                                             prefix_dims);
        if (!loss) return result_t::fail(std::move(loss));
        distributed.losses_[rank] = std::move(loss.value_);
        sz::string key;
        if (!format_into(key, "csegments.reduce.{}", rank))
            return result_t::fail(status_t::bad_alloc_k, "exchange key formatting failed");
        expected<infiniband_member_t> reduce_member = colocated.wire_allreduce(
            static_cast<u32_t>(rank), arenas[rank], domains * tower_scalars * sizeof(f32_t), key.view(),
            timeout_seconds);
        if (!reduce_member) return result_t::fail(std::move(reduce_member));
        distributed.reduce_members_[rank] = std::move(reduce_member.value_);
    }

    // The gateway's gather member, the per-rank loss summands, and the gather-ready ordering token
    // all live with the lead device.
    {
        switch_to_gpu_t gpu_scope {streams[0]};
        if (!gpu_scope.ok()) return result_t::fail(status_t::cuda_error_k, "context entry failed (rank=0)");
        expected<infiniband_member_t> gather_member = colocated.wire_allgather(
            0, arenas[0], gpus * tower_scalars * sizeof(bf16_t), "csegments.gather", timeout_seconds);
        if (!gather_member) return result_t::fail(std::move(gather_member));
        distributed.gather_member_ = std::move(gather_member.value_);
        gpu_arena_alloc<f32_t> f32_alloc {arenas[0]};
        distributed.loss_outs_ = f32r1_tensor_arena_t::try_empty({gpus}, f32_alloc);
        if (!distributed.loss_outs_)
            return result_t::fail(status_t::bad_alloc_k, "scalar bookkeeping allocation failed");
        expected<gpu_event_t> event = gpu_event_t::make();
        if (!event) return result_t::fail(std::move(event));
        distributed.gather_ready_ = std::move(event.value_);
    }

    // Bake every rank's subgraph table once — all the addresses are capture-stable by contract:
    // caller towers/grads, the domain window, and the gather payload never move after construction.
    bf16_t const *world_tower = domains > 1u
                                    ? reinterpret_cast<bf16_t const *>(distributed.gather_member_.payload().data())
                                    : nullptr;
    for (std::size_t rank = 0; rank != gpus; ++rank) {
        f32_t *const window_base = domain.payload(rank).data() + distributed.window_.offset;
        for (std::size_t subgraph = 0; subgraph != subgraphs; ++subgraph) {
            gpu_subgraph_t &entry = distributed.subgraph_tables_[rank * subgraphs + subgraph];
            entry.vertices = domains > 1u ? bf16r2_view_t {world_tower + subgraph * tower_scalars,
                                                           config.vertices_per_gpu, config.hidden}
                                          : towers[subgraph];
            entry.global_offset = subgraph * static_cast<std::size_t>(config.vertices_per_gpu);
            entry.vertices_grad = f32r2_span_t {window_base + subgraph * tower_scalars, config.vertices_per_gpu,
                                                config.hidden};
        }
    }
    return result_t {std::move(distributed)};
}

expected_t distributed_contrastive_segments_t::operator()(std::size_t rank, contrastive_recipe_t recipe,
                                                          gpu_stream_t const &stream) noexcept {
    std::size_t const gpus = losses_.size();
    std::size_t const subgraphs = config_.gpu_subgraphs_per_step;
    std::size_t const tower_scalars = static_cast<std::size_t>(config_.vertices_per_gpu) * config_.hidden;
    std::size_t const vertices_per_mesh = static_cast<std::size_t>(config_.vertices_per_gpu) * subgraphs;
    if (rank >= gpus) return expected_t::fail(status_t::invalid_input_k, "rank {} of {}", rank, gpus);

    switch_to_gpu_t gpu_scope {stream};
    if (!gpu_scope.ok()) return expected_t::fail(status_t::cuda_error_k, "context entry failed ({})", rank);

    // The shared phase rides the lead rank's slice — see the ordering contract.
    if (rank == 0) {
        if (expected_t advanced = domain_->advance(stream); !advanced) return advanced;
        if (world_domains_ > 1u)
            if (expected_t gathered = gather_world_tower_(stream); !gathered) return gathered;
    }
    if (world_domains_ > 1u)
        if (CUresult result = cuStreamWaitEvent(stream.get(), gather_ready_.get(), 0); result != CUDA_SUCCESS)
            return expected_t::fail(result, "gather wait failed (rank={})", rank);

    nk::tensor_view<gpu_subgraph_t, 1> const table {subgraph_tables_.view().data() + rank * subgraphs, subgraphs};
    f32r1_span_t const loss_out {loss_outs_.data() + rank, 1};
    bf16r1_span_t const grad_flat {vertices_grads_[rank].data(), tower_scalars};
    f32_t *const window_base = domain_->payload(rank).data() + window_.offset;
    // The rank's own tower is stored once: across domains in the gather-staging list, in one
    // domain as its own subgraph's table entry (subgraph index == rank in a one-domain world).

    bf16r2_view_t const own_tower = world_domains_ > 1u ? gather_sources_[rank]
                                                        : subgraph_tables_[rank * subgraphs + rank].vertices;

    expected_t staged = losses_[rank](own_tower, table, relations_[rank], recipe, vertices_per_mesh, loss_out,
                                      vertices_grads_[rank], stream);
    if (staged) staged = domain_->barrier(rank, stream);
    // Level-1 reduce: in one domain, in place into this rank's own window slot (slots are
    // disjoint across ranks); across domains, into the rank's reduce member — one switch-reduced
    // window per domain, its own global slot in each — then the level-2 recursive-doubling sum.

    f32r1_view_t column_force {};
    if (world_domains_ == 1u) {
        f32r1_span_t const own_slot {window_base + rank * tower_scalars, tower_scalars};
        if (staged) staged = domain_->reduce_scatter(rank, own_slot, window_.offset + rank * tower_scalars, stream);
        column_force = f32r1_view_t {own_slot};
    }
    else {
        if (staged) staged = reduce_world_columns_(rank, stream, column_force);
    }
    if (staged)
        staged = launch_kernel(reinterpret_cast<void const *>(&contrastive_column_fold_kernel_),
                               dim3 {grid_size_for(tower_scalars), 1u, 1u},
                               dim3 {kernel_block_size_default_k, 1u, 1u}, stream, grad_flat, column_force);
    return staged;
}

expected_t distributed_contrastive_segments_t::operator()(contrastive_recipe_t recipe,
                                                          std::span<gpu_stream_t const> streams) noexcept {
    std::size_t const gpus = losses_.size();
    if (streams.size() != gpus)
        return expected_t::fail(status_t::invalid_input_k, "every rank needs its stream ({} ranks)", gpus);
    for (std::size_t rank = 0; rank != gpus; ++rank)
        if (expected_t staged = (*this)(rank, recipe, streams[rank]); !staged) return staged;
    return {};
}

expected_t distributed_contrastive_segments_t::gather_world_tower_(gpu_stream_t const &stream) noexcept {
    std::size_t const gpus = losses_.size();
    std::size_t const tower_scalars = static_cast<std::size_t>(config_.vertices_per_gpu) * config_.hidden;
    expected_t staged {};
    CUdeviceptr const own_block = reinterpret_cast<CUdeviceptr>(gather_member_.own_block().data());
    for (std::size_t rank = 0; rank != gpus && staged; ++rank)
        if (CUresult result = cuMemcpyDtoDAsync(own_block + rank * tower_scalars * sizeof(bf16_t),
                                                reinterpret_cast<CUdeviceptr>(gather_sources_[rank].data()),
                                                tower_scalars * sizeof(bf16_t), stream.get());
            result != CUDA_SUCCESS)
            staged = expected_t::fail(result, "tower staging copy failed (rank={})", rank);
    if (staged) staged = gather_member_.advance(stream);
    if (staged) staged = gather_member_.allgather(stream);
    if (staged)
        if (CUresult result = cuEventRecord(gather_ready_.get(), stream.get()); result != CUDA_SUCCESS)
            staged = expected_t::fail(result, "gather event record failed");
    return staged;
}

expected_t distributed_contrastive_segments_t::reduce_world_columns_(std::size_t rank, gpu_stream_t const &stream,
                                                                     f32r1_view_t &column_force) noexcept {
    std::size_t const gpus = losses_.size();
    std::size_t const tower_scalars = static_cast<std::size_t>(config_.vertices_per_gpu) * config_.hidden;
    f32_t *const shard = reinterpret_cast<f32_t *>(reduce_members_[rank].payload().data());
    for (u32_t domain = 0; domain != world_domains_; ++domain) {
        f32r1_span_t const domain_slot {shard + domain * tower_scalars, tower_scalars};
        std::size_t const grid_slot = window_.offset +
                                      (static_cast<std::size_t>(domain) * gpus + rank) * tower_scalars;
        if (expected_t reduced = domain_->reduce_scatter(rank, domain_slot, grid_slot, stream); !reduced)
            return reduced;
    }
    if (expected_t advanced = reduce_members_[rank].advance(stream); !advanced) return advanced;
    if (expected_t summed = reduce_members_[rank].allreduce_sum(stream); !summed) return summed;
    column_force = f32r1_view_t {shard + static_cast<std::size_t>(multimem_domain_index_) * tower_scalars,
                                 tower_scalars};
    return {};
}
