/**
 *  @brief Device kernels and heavy method bodies of the sharded sigmoid-pairwise segments contrastive loss
 *      declared in @ref contrastive_segments.cuh — the one translation unit behind its consumers.
 *  @file src/core/contrastive_segments.cu
 *  @author Ash Vardanian
 *  @date July 2, 2026
 */
#include <cmath> // `std::sqrt`, `std::fmax` — the prefix norms and their floored squares

#include <algorithm> // `std::max` — the largest product pack
#include <iterator>  // `std::size` — the retrieval sums
#include <limits>    // `std::numeric_limits` — the empty maximum

#include <cub/block/block_scan.cuh> // `cub::BlockScan` — the ranked rows' positive ranks

#include "uform/core/contrastive_segments.cuh"

using namespace unum::uform::core;

expected<matryoshka_prefixes_t> matryoshka_prefixes_t::make(u32r1_view_t prefix_dims, std::size_t hidden) noexcept {
    std::size_t const count = prefix_dims.extent(0);
    if (count == 0 || count > max_matryoshka_prefixes_k)
        return expected<matryoshka_prefixes_t>::fail(
            status_t::shape_unsupported_k, "need 1..{} Matryoshka prefixes (got {})", max_matryoshka_prefixes_k, count);
    matryoshka_prefixes_t prefixes;
    prefixes.count_ = static_cast<u32_t>(count);
    for (std::size_t prefix_index = 0; prefix_index != count; ++prefix_index) {
        u32_t const dim = prefix_dims[prefix_index];
        u32_t const previous = prefix_index != 0 ? prefix_dims[prefix_index - 1] : 0u;
        if (dim <= previous || dim > hidden)
            return expected<matryoshka_prefixes_t>::fail( //
                status_t::shape_unsupported_k,
                "prefix dims must ascend strictly within hidden (prefix={} dim={} previous={} hidden={})", prefix_index,
                dim, previous, hidden);
        prefixes.dims_[prefix_index] = dim;
    }
    return expected<matryoshka_prefixes_t> {std::move(prefixes)};
}

/**
 *  @brief Count the live vertices of the global axis, the mean divisor: the nonzero entries of the
 *      relations' to-vertex lengths, or the whole @p to_vertices_per_mesh when every vertex is
 *      live, floored at one so an all-PAD step divides its zero sums by one.
 *
 *  __grid_size__ @c [1,1,1] ; one CTA stripes the axis; no dynamic shared memory.
 */
static __global__ void contrastive_live_vertices_kernel_(contrastive_relations_view_t relations,
                                                         std::size_t to_vertices_per_mesh, f32r1_span_t live_vertices) {
    u32r1_view_t const lengths = relations.to_vertex_effective_length;
    f32_t thread_live = 0.0f;
    for (std::size_t vertex = threadIdx.x; vertex < to_vertices_per_mesh; vertex += blockDim.x)
        thread_live += lengths.empty() || lengths[vertex] != 0u ? 1.0f : 0.0f;
    f32_t const live = block_reduce_broadcast<kernel_block_size_default_k>(thread_live);
    if (threadIdx.x == 0u) live_vertices[0] = std::fmax(live, 1.0f);
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
        if (threadIdx.x == 0u) norms[prefix_index, row_index] = std::sqrt(running_sum_squares);
    }
}

/**
 *  @brief The scaled cosines of every subgraph: @c logits[p,s,v,n] = β_v·raw/(‖v‖·‖n‖) , that is
 *      β·cos with @b no bias, with the floored denominator — the epilogue of the tensor-core dot
 *      products.
 *
 *  __grid_size__ @c [K·S·R·Rc/block,1,1] rounded up; grid-stride over cells; no dynamic shared
 *  memory.
 */
static __global__ void contrastive_scaled_cosine_kernel_(f32r4_view_t raw_dot, f32r2_view_t from_vertex_norms,
                                                         f32r3_view_t to_vertex_norms, contrastive_recipe_t recipe,
                                                         f32r4_span_t logits) {
    std::size_t const subgraphs = raw_dot.extent(1), num_vertices = raw_dot.extent(2);
    std::size_t const num_neighbors = raw_dot.extent(3);
    std::size_t const total = raw_dot.extent(0) * subgraphs * num_vertices * num_neighbors;
    std::size_t const stride = static_cast<std::size_t>(gridDim.x) * blockDim.x;
    for (std::size_t cell = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x; cell < total;
         cell += stride) {
        std::size_t const prefix_index = cell / (subgraphs * num_vertices * num_neighbors);
        std::size_t const subgraph_index = (cell / (num_vertices * num_neighbors)) % subgraphs;
        std::size_t const vertex_index = (cell / num_neighbors) % num_vertices;
        std::size_t const neighbor_index = cell % num_neighbors;
        f32_t const denominator = from_vertex_norms[prefix_index, vertex_index] *
                                  to_vertex_norms[subgraph_index, prefix_index, neighbor_index];
        f32_t const cosine = denominator > cosine_denominator_floor_k
                                 ? raw_dot[prefix_index, subgraph_index, vertex_index, neighbor_index] / denominator
                                 : 0.0f;
        logits[prefix_index, subgraph_index, vertex_index, neighbor_index] =
            contrastive_from_vertex_beta_(recipe, static_cast<u32_t>(vertex_index)) * cosine;
    }
}

/**
 *  @brief The raw-dot gradient of one subgraph block: @c raw_dot_grad[p,v,n] = β_v·logit_grad/max(‖v‖·‖n‖,floor)
 *      — the operand the gradient matmuls and the self-coefficient folds both consume. A vertex's
 *      cell against its own copy on the self-edge default is zero.
 *
 *  __grid_size__ @c [ceil(K·R·Rc/block),1,1] ; grid-stride over cells; no dynamic shared memory.
 */
static __global__ void contrastive_raw_dot_grad_kernel_(f32r3_view_t logit_grad, f32r2_view_t from_vertex_norms,
                                                        f32r2_view_t to_vertex_norms, contrastive_recipe_t recipe,
                                                        contrastive_relations_view_t relations,
                                                        std::size_t global_offset, f32r3_span_t raw_dot_grad) {
    std::size_t const num_vertices = logit_grad.extent(1);
    std::size_t const num_neighbors = logit_grad.extent(2);
    std::size_t const total = logit_grad.extent(0) * num_vertices * num_neighbors;
    std::size_t const stride = static_cast<std::size_t>(gridDim.x) * blockDim.x;
    for (std::size_t cell = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x; cell < total;
         cell += stride) {
        std::size_t const prefix_index = cell / (num_vertices * num_neighbors);
        std::size_t const vertex_index = (cell / num_neighbors) % num_vertices;
        std::size_t const neighbor_index = cell % num_neighbors;
        f32_t const denominator = from_vertex_norms[prefix_index, vertex_index] *
                                  to_vertex_norms[prefix_index, neighbor_index];
        // Cosine 1 against its own copy has a zero gradient; skip the cancelling BF16 product.
        bool const own_copy = !relations.has_explicit_attract_edges() &&
                              global_offset + neighbor_index == relations.self_edge_base + vertex_index;
        raw_dot_grad[prefix_index, vertex_index, neighbor_index] =
            own_copy ? 0.0f
                     : contrastive_from_vertex_beta_(recipe, static_cast<u32_t>(vertex_index)) *
                           logit_grad[prefix_index, vertex_index, neighbor_index] /
                           std::fmax(denominator, cosine_denominator_floor_k);
    }
}

/**
 *  @brief Deterministic per-vertex cosine self-coefficients of one subgraph block:
 *      @c coefficients[prefix,vertex] = Σ_j raw_dot_grad·raw_dot — a fixed-order block reduce per row.
 *
 *  __grid_size__ @c [num_vertices,num_prefixes,1] ; threads stripe the subgraph's rows; no dynamic shared
 *  memory.
 */
static __global__ void contrastive_from_vertex_self_coefficients_kernel_(f32r3_view_t raw_dot_grad,
                                                                         f32r3_view_t raw_dot,
                                                                         f32r2_span_t from_vertex_self_coefficients) {
    std::size_t const num_vertices = raw_dot_grad.extent(1);
    std::size_t const num_neighbors = raw_dot_grad.extent(2);
    std::size_t const vertex_index = blockIdx.x;
    std::size_t const prefix_index = blockIdx.y;
    if (vertex_index >= num_vertices) return;
    f32_t thread_partial = 0.0f;
    for (std::size_t neighbor_local = threadIdx.x; neighbor_local < num_neighbors; neighbor_local += blockDim.x)
        thread_partial += raw_dot_grad[prefix_index, vertex_index, neighbor_local] *
                          raw_dot[prefix_index, vertex_index, neighbor_local];
    f32_t const total = block_reduce_broadcast<kernel_block_size_default_k>(thread_partial);
    if (threadIdx.x == 0u) from_vertex_self_coefficients[prefix_index, vertex_index] = total;
}

/**
 *  @brief Deterministic per-neighbor cosine self-coefficients — the column-sum sibling of
 *      @ref contrastive_from_vertex_self_coefficients_kernel_ .
 *
 *  __grid_size__ @c [count,num_prefixes,1] ; threads stripe the vertex rows; no dynamic shared memory.
 */
static __global__ void contrastive_to_vertex_self_coefficients_kernel_(f32r3_view_t raw_dot_grad, f32r3_view_t raw_dot,
                                                                       f32r2_span_t to_vertex_self_coefficients) {
    std::size_t const num_vertices = raw_dot_grad.extent(1);
    std::size_t const num_neighbors = raw_dot_grad.extent(2);
    std::size_t const neighbor_local = blockIdx.x;
    std::size_t const prefix_index = blockIdx.y;
    if (neighbor_local >= num_neighbors) return;
    f32_t thread_partial = 0.0f;
    for (std::size_t vertex_index = threadIdx.x; vertex_index < num_vertices; vertex_index += blockDim.x)
        thread_partial += raw_dot_grad[prefix_index, vertex_index, neighbor_local] *
                          raw_dot[prefix_index, vertex_index, neighbor_local];
    f32_t const total = block_reduce_broadcast<kernel_block_size_default_k>(thread_partial);
    if (threadIdx.x == 0u) to_vertex_self_coefficients[prefix_index, neighbor_local] = total;
}

/**
 *  @brief Suffix-accumulate the per-prefix raw-dot gradients @b in @b place, turning plane @c p into
 *      @c G_p = Σ_{p'>=p} raw_dot_grad_{p'} — the per-segment matmul operand (segment @c s is contained in
 *      exactly the prefixes @c p>=s ).
 *
 *  __grid_size__ @c [ceil(R·Rn/block),1,1] ; each thread owns one (vertex, neighbor) pair and walks the
 *  prefixes high-to-low serially — fixed order, no atomics; no dynamic shared memory.
 */
static __global__ void contrastive_suffix_accumulate_kernel_(f32r3_span_t raw_dot_grad) {
    std::size_t const prefix_count = raw_dot_grad.extent(0);
    std::size_t const num_neighbors = raw_dot_grad.extent(2);
    std::size_t const cells_per_prefix = raw_dot_grad.extent(1) * num_neighbors;
    std::size_t const stride = static_cast<std::size_t>(gridDim.x) * blockDim.x;
    for (std::size_t cell = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x; cell < cells_per_prefix;
         cell += stride) {
        std::size_t const vertex_index = cell / num_neighbors;
        std::size_t const neighbor_index = cell % num_neighbors;
        for (std::size_t prefix_index = prefix_count - 1u; prefix_index != 0u; --prefix_index)
            raw_dot_grad[prefix_index - 1u, vertex_index, neighbor_index] +=
                raw_dot_grad[prefix_index, vertex_index, neighbor_index];
    }
}

/**
 *  @brief The owned-vertex cosine self-fold of one subgraph:
 *      @c from_vertices_grad_f32(q,ch) -= Σ_p 1{ch<D_p}·coefficients[p,q]/max(‖q‖²,floor)·q(q,ch) — the term the
 *      cross-product matmuls cannot express.
 *
 *  __grid_size__ @c [num_vertices,1,1] ; threads stripe the hidden channels, prefixes loop inside; no
 *  dynamic shared memory.
 */
static __global__ void contrastive_from_vertex_self_fold_kernel_(bf16r2_view_t owned_vertices,
                                                                 f32r2_view_t from_vertex_norms,
                                                                 f32r2_view_t from_vertex_self_coefficients,
                                                                 f32r2_span_t from_vertices_grad_f32,
                                                                 matryoshka_prefixes_t prefixes) {
    std::size_t const num_vertices = owned_vertices.extent(0);
    std::size_t const hidden = owned_vertices.extent(1);
    std::size_t const vertex_index = blockIdx.x;
    if (vertex_index >= num_vertices) return;
    for (unsigned channel_index = threadIdx.x; channel_index < hidden; channel_index += blockDim.x) {
        f32_t fold = 0.0f;
        for (u32_t prefix_index = 0u; prefix_index != prefixes.size(); ++prefix_index) {
            if (channel_index >= prefixes[prefix_index]) continue;
            f32_t const vertex_norm = from_vertex_norms[prefix_index, vertex_index];
            fold += from_vertex_self_coefficients[prefix_index, vertex_index] /
                    std::fmax(vertex_norm * vertex_norm, cosine_denominator_floor_k);
        }
        from_vertices_grad_f32[vertex_index, channel_index] -= fold * __bfloat162float(
                                                                          owned_vertices[vertex_index, channel_index]);
    }
}

/**
 *  @brief The neighbor-side cosine self-fold — the column sibling of
 *      @ref contrastive_from_vertex_self_fold_kernel_ , folded into the subgraph's @c to_vertices_grad .
 *
 *  __grid_size__ @c [count,1,1] ; threads stripe the hidden channels, prefixes loop inside; no dynamic
 *  shared memory.
 */
static __global__ void contrastive_to_vertex_self_fold_kernel_(bf16r2_view_t neighbors, f32r2_view_t to_vertex_norms,
                                                               f32r2_view_t to_vertex_self_coefficients,
                                                               f32r2_span_t to_vertices_grad,
                                                               matryoshka_prefixes_t prefixes) {
    std::size_t const count = neighbors.extent(0);
    std::size_t const hidden = neighbors.extent(1);
    std::size_t const neighbor_local = blockIdx.x;
    if (neighbor_local >= count) return;
    for (unsigned channel_index = threadIdx.x; channel_index < hidden; channel_index += blockDim.x) {
        f32_t fold = 0.0f;
        for (u32_t prefix_index = 0u; prefix_index != prefixes.size(); ++prefix_index) {
            if (channel_index >= prefixes[prefix_index]) continue;
            f32_t const neighbor_norm = to_vertex_norms[prefix_index, neighbor_local];
            fold += to_vertex_self_coefficients[prefix_index, neighbor_local] /
                    std::fmax(neighbor_norm * neighbor_norm, cosine_denominator_floor_k);
        }
        to_vertices_grad[neighbor_local, channel_index] -= fold *
                                                           __bfloat162float(neighbors[neighbor_local, channel_index]);
    }
}

/**
 *  @brief Elementwise Float32 → BFloat16 cast of a matrix into a same-shaped, possibly row-padded
 *      destination.
 *
 *  __grid_size__ @c [rows·columns/block,1,1] rounded up; grid-stride; no dynamic shared memory.
 */
static __global__ void contrastive_cast_f32_to_bf16_kernel_(f32r2_view_t source, bf16r2_span_t destination) {
    std::size_t const columns = source.extent(1), count = source.extent(0) * columns;
    std::size_t const stride = static_cast<std::size_t>(gridDim.x) * blockDim.x;
    for (std::size_t index = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x; index < count;
         index += stride)
        destination[index / columns, index % columns] = __float2bfloat16(source[index / columns, index % columns]);
}

/** The numerically stable logistic @c σ(value) . */
[[nodiscard]] static __device__ f32_t contrastive_sigmoid_(f32_t value) noexcept {
    if (value >= 0.0f) return 1.0f / (1.0f + std::exp(-value));
    f32_t const exp_value = std::exp(value);
    return exp_value / (1.0f + exp_value);
}

/** The numerically stable @c softplus(value)=log(1+e^value) . */
[[nodiscard]] static __device__ f32_t contrastive_softplus_(f32_t value) noexcept {
    return std::fmax(value, 0.0f) + std::log1p(std::exp(-std::fabs(value)));
}

/** Cell @c (prefix,row,candidate) of a @b [prefixes,subgraphs,rows,per_subgraph] stash by global
 *  candidate. */
[[nodiscard]] static __device__ f32_t &contrastive_stash_cell_(f32r4_span_t stash, std::size_t prefix, std::size_t row,
                                                               std::size_t candidate) noexcept {
    std::size_t const per_subgraph = stash.extent(3);
    return stash[prefix, candidate / per_subgraph, row, candidate % per_subgraph];
}

/**
 *  @brief The sigmoid BCE over whole anchor rows, all prefixes fused, in place: the signed weight
 *      totals over the row first, then every cell's @c d(loss)/d(logit) over its scaled score, and
 *      the row's loss, bias and log-β summands.
 *
 *  The balanced @p form normalizes the attract and repel halves apart; the sigmoid form weighs
 *  every repel on the attracts' scale.
 *
 *  A non-finite @c t·logit leaves the pair's gradient zero; a @c w==0 pair is skipped in place.
 *
 *  __grid_size__ @c [rows,1,1] ; one CTA per row, the threads stripe the candidates; no dynamic
 *  shared memory.
 */
static __global__ void contrastive_sigmoid_bce_kernel_(f32r4_span_t logits, f32r2_span_t accumulator,
                                                       contrastive_relations_view_t relations,
                                                       contrastive_recipe_t recipe, f32r1_view_t live_anchors,
                                                       contrastive_pair_loss_form_t form) {
    std::size_t const prefixes = logits.extent(0), rows = logits.extent(2);
    std::size_t const candidates = logits.extent(1) * logits.extent(3);
    u32_t const row = blockIdx.x;
    if (row >= rows) return;
    f32_t thread_positive = 0.0f, thread_negative = 0.0f;
    for (std::size_t candidate = threadIdx.x; candidate < candidates; candidate += blockDim.x) {
        contrastive_pair_relation_t const relation = contrastive_pair_relation(relations, row, u32_t(candidate));
        if (relation.target > 0.0f) thread_positive += relation.weight;
        else if (relation.target < 0.0f) thread_negative += relation.weight;
    }
    f32_t const live = std::fmax(live_anchors[0], 1.0f);
    f32_t const positive_weight = block_reduce_broadcast<kernel_block_size_default_k>(thread_positive);
    f32_t const negative_weight = block_reduce_broadcast<kernel_block_size_default_k>(thread_negative);
    // A sigmoid row weighs cells like SigLIP's one positive, so a row of repels alone stays finite.
    bool const sigmoid = form == contrastive_pair_loss_form_t::sigmoid_k;
    f32_t const positive_scale = 1.0f / (std::fmax(positive_weight, sigmoid ? 1.0f : weight_sum_floor_k) * live);
    f32_t const negative_scale = sigmoid ? positive_scale
                                         : 1.0f / (std::fmax(negative_weight, weight_sum_floor_k) * live);
    f32_t const bias = contrastive_from_vertex_bias_(recipe, row);
    f32_t thread_loss = 0.0f, thread_bias_grad = 0.0f, thread_log_inverse_temperature_grad = 0.0f;
    for (std::size_t prefix = 0; prefix != prefixes; ++prefix)
        for (std::size_t candidate = threadIdx.x; candidate < candidates; candidate += blockDim.x) {
            f32_t &cell = contrastive_stash_cell_(logits, prefix, row, candidate);
            contrastive_pair_relation_t const relation = contrastive_pair_relation(relations, row, u32_t(candidate));
            f32_t const scaled = cell;
            f32_t const signed_logit = relation.target * (scaled + bias);
            f32_t cell_grad = 0.0f;
            if (relation.weight != 0.0f && isfinite(signed_logit)) {
                f32_t const scale = relation.target > 0.0f ? positive_scale : negative_scale;
                thread_loss += relation.weight * contrastive_softplus_(-signed_logit) * scale;
                cell_grad = relation.weight * relation.target * (contrastive_sigmoid_(signed_logit) - 1.0f) * scale;
                thread_bias_grad += cell_grad;
                thread_log_inverse_temperature_grad += cell_grad * scaled;
            }
            cell = cell_grad;
        }
    f32_t const loss = block_reduce_broadcast<kernel_block_size_default_k>(thread_loss);
    f32_t const bias_grad = block_reduce_broadcast<kernel_block_size_default_k>(thread_bias_grad);
    f32_t const log_inverse_temperature_grad = block_reduce_broadcast<kernel_block_size_default_k>(
        thread_log_inverse_temperature_grad);
    if (threadIdx.x != 0u) return;
    accumulator[static_cast<u32_t>(pair_loss_row_t::loss_k), row] = loss;
    accumulator[static_cast<u32_t>(pair_loss_row_t::bias_grad_k), row] = bias_grad;
    accumulator[static_cast<u32_t>(pair_loss_row_t::log_inverse_temperature_grad_k), row] =
        log_inverse_temperature_grad;
}

/**
 *  @brief The ranked BCE of one prefix over whole anchor rows: every positive's scaled score
 *      against every negative's, written into @p gradients while @p logits keeps the scores the
 *      other cells read.
 *
 *  Row @c i 's positives are the cells the resolver attracts, by edge, by shared key or as its own
 *  copy on the self-edge default, ranked in a fixed order by a block scan. Each pass over the
 *  negatives sums @ref ranked_positives_per_pass_k positives at once: a negative's gradient
 *  accumulates across the passes in its own cell, and each positive's total folds once at the
 *  pass's end in a fixed order. The first prefix assigns the row's summands and every later one
 *  adds; the bias cancels, so its summand stays zero.
 *
 *  __grid_size__ @c [rows,1,1] ; one CTA per row, the threads stripe the candidates; no dynamic
 *  shared memory.
 */
static __global__ void contrastive_ranked_bce_kernel_(f32r4_view_t logits, f32r2_span_t gradients,
                                                      f32r2_span_t accumulator, contrastive_relations_view_t relations,
                                                      f32r1_view_t live_anchors, u32_t prefix) {
    std::size_t const rows = logits.extent(2), per_subgraph = logits.extent(3);
    std::size_t const candidates = logits.extent(1) * per_subgraph;
    u32_t const row = blockIdx.x;
    if (row >= rows) return;
    auto const score = [&](std::size_t candidate) noexcept -> f32_t {
        return logits[prefix, candidate / per_subgraph, row, candidate % per_subgraph];
    };
    f32_t thread_positive = 0.0f, thread_negative = 0.0f;
    u32_t thread_positives = 0;
    for (std::size_t candidate = threadIdx.x; candidate < candidates; candidate += blockDim.x) {
        contrastive_pair_relation_t const relation = contrastive_pair_relation(relations, row, u32_t(candidate));
        if (relation.target > 0.0f && relation.weight != 0.0f) thread_positive += relation.weight, ++thread_positives;
        else if (relation.target < 0.0f) thread_negative += relation.weight;
    }
    using positive_scan_t = cub::BlockScan<u32_t, kernel_block_size_default_k>;
    __shared__ typename positive_scan_t::TempStorage positive_scan;
    __shared__ u32_t pass_ids[ranked_positives_per_pass_k];
    __shared__ f32_t pass_weights[ranked_positives_per_pass_k];
    u32_t first_rank = 0, positives = 0;
    positive_scan_t {positive_scan}.ExclusiveSum(thread_positives, first_rank, positives);
    f32_t const positive_weight = block_reduce_broadcast<kernel_block_size_default_k>(thread_positive);
    f32_t const negative_weight = block_reduce_broadcast<kernel_block_size_default_k>(thread_negative);
    f32_t const scale = 1.0f / (std::fmax(positive_weight * negative_weight, weight_sum_floor_k) *
                                std::fmax(live_anchors[0], 1.0f));

    f32_t thread_loss = 0.0f, thread_log_inverse_temperature_grad = 0.0f;
    u32_t const passes = positives != 0u ? divide_round_up(positives, ranked_positives_per_pass_k) : 1u;
    for (u32_t pass = 0; pass != passes; ++pass) {
        // The pass takes the positives ranked [pass·k, pass·k+k), each thread's in its stride
        // order.
        u32_t const first_of_pass = pass * ranked_positives_per_pass_k;
        u32_t const end_of_pass = first_of_pass + ranked_positives_per_pass_k;
        __syncthreads();
        if (threadIdx.x < ranked_positives_per_pass_k) pass_weights[threadIdx.x] = 0.0f;
        __syncthreads();
        u32_t rank = first_rank;
        for (std::size_t candidate = threadIdx.x;
             candidate < candidates && rank < end_of_pass && first_rank + thread_positives > first_of_pass;
             candidate += blockDim.x) {
            contrastive_pair_relation_t const relation = contrastive_pair_relation(relations, row, u32_t(candidate));
            if (relation.target <= 0.0f || relation.weight == 0.0f) continue;
            if (rank >= first_of_pass) {
                pass_ids[rank - first_of_pass] = u32_t(candidate);
                pass_weights[rank - first_of_pass] = relation.weight;
            }
            ++rank;
        }
        __syncthreads();
        f32_t positive_scores[ranked_positives_per_pass_k] = {}, positive_weights[ranked_positives_per_pass_k] = {};
        u32_t positive_ids[ranked_positives_per_pass_k] = {};
        for (u32_t entry = 0; entry != ranked_positives_per_pass_k; ++entry) {
            positive_weights[entry] = pass_weights[entry];
            if (positive_weights[entry] == 0.0f) continue;
            positive_ids[entry] = pass_ids[entry];
            positive_scores[entry] = score(positive_ids[entry]);
        }
        f32_t thread_pulls[ranked_positives_per_pass_k] = {};
        for (std::size_t candidate = threadIdx.x; candidate < candidates; candidate += blockDim.x) {
            contrastive_pair_relation_t const relation = contrastive_pair_relation(relations, row, u32_t(candidate));
            bool const negative = relation.target < 0.0f && relation.weight != 0.0f;
            f32_t cell_grad = pass != 0u && negative ? gradients[row, candidate] : 0.0f;
            if (negative) {
                f32_t const negative_score = score(candidate);
                f32_t pushed = 0.0f;
                for (u32_t entry = 0; entry != ranked_positives_per_pass_k; ++entry) {
                    f32_t const margin = negative_score - positive_scores[entry];
                    if (positive_weights[entry] == 0.0f || !isfinite(margin)) continue;
                    f32_t const sigmoid = contrastive_sigmoid_(margin);
                    pushed += positive_weights[entry] * sigmoid;
                    thread_pulls[entry] += relation.weight * sigmoid;
                    thread_loss += positive_weights[entry] * relation.weight * contrastive_softplus_(margin);
                }
                f32_t const pass_grad = relation.weight * pushed * scale;
                cell_grad += pass_grad;
                thread_log_inverse_temperature_grad += pass_grad * negative_score;
            }
            if (pass == 0u || negative) gradients[row, candidate] = cell_grad;
        }
        for (u32_t entry = 0; entry != ranked_positives_per_pass_k; ++entry) {
            f32_t const pull = block_reduce_broadcast<kernel_block_size_default_k>(thread_pulls[entry]);
            if (threadIdx.x != 0u || positive_weights[entry] == 0.0f) continue;
            f32_t const positive_grad = -positive_weights[entry] * pull * scale;
            gradients[row, positive_ids[entry]] = positive_grad;
            thread_log_inverse_temperature_grad += positive_grad * positive_scores[entry];
        }
    }
    f32_t const loss = block_reduce_broadcast<kernel_block_size_default_k>(thread_loss) * scale;
    f32_t const log_inverse_temperature_grad = block_reduce_broadcast<kernel_block_size_default_k>(
        thread_log_inverse_temperature_grad);
    if (threadIdx.x != 0u) return;
    u32_t const loss_row = static_cast<u32_t>(pair_loss_row_t::loss_k);
    u32_t const bias_row = static_cast<u32_t>(pair_loss_row_t::bias_grad_k);
    u32_t const beta_row = static_cast<u32_t>(pair_loss_row_t::log_inverse_temperature_grad_k);
    accumulator[loss_row, row] = (prefix != 0u ? accumulator[loss_row, row] : 0.0f) + loss;
    accumulator[bias_row, row] = 0.0f;
    accumulator[beta_row, row] = (prefix != 0u ? accumulator[beta_row, row] : 0.0f) + log_inverse_temperature_grad;
}

/**
 *  @brief Copy one prefix's ranked @p gradients back into its @b [subgraphs,rows,per_subgraph]
 *      plane of the stash.
 *
 *  __grid_size__ grid-stride over the plane; no dynamic shared memory.
 */
static __global__ void contrastive_ranked_copy_kernel_(f32r2_view_t gradients, f32r3_span_t plane) {
    std::size_t const rows = plane.extent(1), per_subgraph = plane.extent(2);
    std::size_t const cells = plane.extent(0) * rows * per_subgraph;
    f32r3_view_t const row_planes = gradients.reshape<3>({rows, plane.extent(0), per_subgraph});
    std::size_t const stride = static_cast<std::size_t>(gridDim.x) * blockDim.x;
    for (std::size_t cell = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x; cell < cells;
         cell += stride) {
        std::size_t const subgraph = cell / (rows * per_subgraph), row = (cell / per_subgraph) % rows;
        std::size_t const column = cell % per_subgraph;
        plane[subgraph, row, column] = row_planes[row, subgraph, column];
    }
}

/**
 *  @brief The pair loss's finalise: the rank-local loss summand and the optional temperature
 *      gradient summands of every group, in fixed-order block reduces over the per-row accumulator.
 *
 *  __grid_size__ @c [1,1,1] ; one CTA, threads stripe the rows once per group; no dynamic shared
 *  memory.
 */
static __global__ void contrastive_pair_loss_finalize_kernel_(f32r2_view_t accumulator, contrastive_recipe_t recipe,
                                                              f32r1_span_t loss_out, f32r1_span_t bias_grad,
                                                              f32r1_span_t log_inverse_temperature_grad) {
    std::size_t const rows = accumulator.extent(1);
    u32_t const loss_row = static_cast<u32_t>(pair_loss_row_t::loss_k);
    u32_t const bias_row = static_cast<u32_t>(pair_loss_row_t::bias_grad_k);
    u32_t const beta_row = static_cast<u32_t>(pair_loss_row_t::log_inverse_temperature_grad_k);
    f32_t thread_loss = 0.0f;
    for (std::size_t row = threadIdx.x; row < rows; row += blockDim.x) thread_loss += accumulator[loss_row, row];
    f32_t const total_loss = block_reduce_broadcast<kernel_block_size_default_k>(thread_loss);
    if (threadIdx.x == 0u) loss_out[0] = total_loss;

    bool const wants_bias = !bias_grad.empty();
    bool const wants_log_inverse_temperature = !log_inverse_temperature_grad.empty();
    if (!wants_bias && !wants_log_inverse_temperature) return;
    for (u32_t temperature_group = 0; temperature_group != recipe.temperature_group_count(); ++temperature_group) {
        f32_t thread_bias_grad = 0.0f, thread_log_inverse_temperature_grad = 0.0f;
        for (std::size_t row = threadIdx.x; row < rows; row += blockDim.x) {
            u32_t const resolved_temperature_group =
                recipe.from_vertex_temperature_group.empty()
                    ? 0u
                    : static_cast<u32_t>(recipe.from_vertex_temperature_group[static_cast<u32_t>(row)]);
            if (resolved_temperature_group != temperature_group) continue;
            thread_bias_grad += accumulator[bias_row, row];
            thread_log_inverse_temperature_grad += accumulator[beta_row, row];
        }
        f32_t const group_bias_grad = block_reduce_broadcast<kernel_block_size_default_k>(thread_bias_grad);
        f32_t const group_log_inverse_temperature_grad = block_reduce_broadcast<kernel_block_size_default_k>(
            thread_log_inverse_temperature_grad);
        if (threadIdx.x == 0u) {
            if (wants_bias) bias_grad[temperature_group] = group_bias_grad;
            if (wants_log_inverse_temperature)
                log_inverse_temperature_grad[temperature_group] = group_log_inverse_temperature_grad;
        }
    }
}

/** The CTA-wide maximum of @p value , broadcast to every thread; the identity is negative
 *  infinity. */
[[nodiscard]] static __device__ f32_t contrastive_block_max_(f32_t value) noexcept {
    __shared__ f32_t warp_maxima[kernel_block_size_default_k / warp_size_k];
    __shared__ f32_t block_maximum;
    for (unsigned delta = warp_size_k / 2u; delta != 0u; delta /= 2u)
        value = std::fmax(value, __shfl_xor_sync(0xffffffffu, value, delta));
    if (threadIdx.x % warp_size_k == 0u) warp_maxima[threadIdx.x / warp_size_k] = value;
    __syncthreads();
    if (threadIdx.x == 0u) {
        f32_t maximum = warp_maxima[0];
        for (unsigned warp = 1; warp != kernel_block_size_default_k / warp_size_k; ++warp)
            maximum = std::fmax(maximum, warp_maxima[warp]);
        block_maximum = maximum;
    }
    __syncthreads();
    return block_maximum;
}

/**
 *  @brief Add one subgraph's live unit vertices into the world's centre sum over its width,
 *      @c centre[k]+=Σ_j v[j,k]/‖v_j‖ , each thread walking the rows of its channel in order.
 *
 *  __grid_size__ @c [ceil(width/block),1,1] ; no dynamic shared memory.
 */
static __global__ void contrastive_centre_add_kernel_(bf16r2_view_t vertices, f32r1_view_t norms, u32r1_view_t lengths,
                                                      f32r1_span_t centre) {
    std::size_t const channel = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (channel >= centre.extent(0)) return;
    f32_t sum = centre[channel];
    for (std::size_t row = 0; row != vertices.extent(0); ++row)
        if ((lengths.empty() || lengths[row] != 0u) && norms[row] != 0.0f)
            sum += __bfloat162float(vertices[row, channel]) / norms[row];
    centre[channel] = sum;
}

/**
 *  @brief Every row's unit vertex against the world's centre sum,
 *      @c alignment[j]=(v_j/‖v_j‖)·centre , over the centre's width.
 *
 *  __grid_size__ @c [rows,1,1] ; one CTA per row; no dynamic shared memory.
 */
static __global__ void contrastive_centre_alignment_kernel_(bf16r2_view_t vertices, f32r1_view_t norms,
                                                            f32r1_view_t centre, f32r1_span_t alignment) {
    std::size_t const row = blockIdx.x;
    if (row >= vertices.extent(0)) return;
    f32_t thread_partial = 0.0f;
    for (std::size_t channel = threadIdx.x; channel < centre.extent(0); channel += blockDim.x)
        thread_partial += __bfloat162float(vertices[row, channel]) * centre[channel];
    f32_t const total = block_reduce_broadcast<kernel_block_size_default_k>(thread_partial);
    if (threadIdx.x == 0u) alignment[row] = norms[row] != 0.0f ? total / norms[row] : 0.0f;
}

/**
 *  @brief One row's retrieval over the full-width scaled cosines: the rank of its best positive
 *      among the scored cells and the mean cosines of its attract and ambient cells.
 *
 *  It also counts the row's hard negatives above that positive, and forms its products with every
 *  live vertex once each unit vertex has the world's mean unit vector @c m subtracted,
 *  @c cos−a_i−a_j+‖m‖² with @c a_j=û_j·m , per cell so no sum of near-unit cosines cancels.
 *
 *  __grid_size__ @c [rows,1,1] ; one CTA per row, the threads stripe the candidates; no dynamic
 *  shared memory.
 */
static __global__ void contrastive_retrieval_kernel_(f32r4_view_t logits, f32r2_span_t table,
                                                     contrastive_relations_view_t relations,
                                                     contrastive_recipe_t recipe, f32r1_view_t centre,
                                                     f32r1_view_t from_vertex_alignment,
                                                     f32r2_view_t to_vertex_alignment, f32r1_view_t live_vertices) {
    std::size_t const prefix = logits.extent(0) - 1u, rows = logits.extent(2), per_subgraph = logits.extent(3);
    std::size_t const candidates = logits.extent(1) * per_subgraph;
    u32_t const row = blockIdx.x;
    if (row >= rows) return;
    u32r1_view_t const row_lengths = relations.from_vertex_effective_length;
    u32r1_view_t const lengths = relations.to_vertex_effective_length;
    bool const live = row_lengths.empty() || row_lengths[row] != 0u;
    f32_t const beta = contrastive_from_vertex_beta_(recipe, row);
    u32_t const repel_begin = relations.has_explicit_attract_edges() ? relations.repel_offsets[row] : 0u;
    u32_t const repel_end = relations.has_explicit_attract_edges() ? relations.repel_offsets[row + 1u] : 0u;
    auto const score = [&](std::size_t candidate) noexcept -> f32_t {
        return logits[prefix, candidate / per_subgraph, row, candidate % per_subgraph];
    };
    f32_t const world = live_vertices[0];
    f32_t thread_centre = 0.0f;
    for (std::size_t channel = threadIdx.x; channel < centre.extent(0); channel += blockDim.x)
        thread_centre += centre[channel] * centre[channel];
    f32_t const centre_norm = block_reduce_broadcast<kernel_block_size_default_k>(thread_centre) / (world * world);
    f32_t const row_alignment = from_vertex_alignment[row] / world;
    f32_t best_positive = -std::numeric_limits<f32_t>::infinity();
    f32_t positives = 0.0f, attract_cosine = 0.0f, centred_products = 0.0f;
    for (std::size_t candidate = threadIdx.x; live && candidate < candidates; candidate += blockDim.x) {
        if (!lengths.empty() && lengths[candidate] == 0u) continue;
        f32_t const cosine = score(candidate) / beta;
        f32_t const candidate_alignment = to_vertex_alignment[candidate / per_subgraph, candidate % per_subgraph] /
                                          world;
        f32_t const centred = cosine - row_alignment - candidate_alignment + centre_norm;
        centred_products += centred * centred;
        contrastive_pair_relation_t const relation = contrastive_pair_relation(relations, row, u32_t(candidate));
        if (relation.target <= 0.0f || relation.weight == 0.0f) continue;
        best_positive = std::fmax(best_positive, score(candidate));
        positives += 1.0f, attract_cosine += cosine;
    }
    best_positive = contrastive_block_max_(best_positive);
    f32_t negatives = 0.0f, above = 0.0f, ambient_cosine = 0.0f, ambients = 0.0f, hard_above = 0.0f, hards = 0.0f;
    for (std::size_t candidate = threadIdx.x; live && candidate < candidates; candidate += blockDim.x) {
        contrastive_pair_relation_t const relation = contrastive_pair_relation(relations, row, u32_t(candidate));
        if (relation.target >= 0.0f || relation.weight == 0.0f) continue;
        f32_t const scored = score(candidate);
        bool const outranks = scored > best_positive;
        negatives += 1.0f, above += outranks ? 1.0f : 0.0f;
        bool const hard = contrastive_run_find_(relations.repel_ids, repel_begin, repel_end, u32_t(candidate)) !=
                          repel_end;
        if (hard) hards += 1.0f, hard_above += outranks ? 1.0f : 0.0f;
        else ambients += 1.0f, ambient_cosine += scored / beta;
    }
    f32_t const sums[] = {positives, negatives,  above, attract_cosine,  ambient_cosine,
                          ambients,  hard_above, hards, centred_products};
    f32_t totals[std::size(sums)];
    for (std::size_t index = 0; index != std::size(sums); ++index)
        totals[index] = block_reduce_broadcast<kernel_block_size_default_k>(sums[index]);
    if (threadIdx.x != 0u) return;
    auto const column = [&](retrieval_column_t which) noexcept -> f32_t & { return table[row, std::size_t(which)]; };
    bool const ranked = totals[0] != 0.0f && totals[1] != 0.0f;
    column(retrieval_column_t::ranked_k) = ranked ? 1.0f : 0.0f;
    column(retrieval_column_t::hit_at_1_k) = ranked && totals[2] == 0.0f ? 1.0f : 0.0f;
    column(retrieval_column_t::hit_at_5_k) = ranked && totals[2] < 5.0f ? 1.0f : 0.0f;
    column(retrieval_column_t::chance_k) = ranked ? totals[0] / (totals[0] + totals[1]) : 0.0f;
    column(retrieval_column_t::attract_cosine_k) = totals[3];
    column(retrieval_column_t::attracts_k) = totals[0];
    column(retrieval_column_t::ambient_cosine_k) = totals[4];
    column(retrieval_column_t::ambients_k) = totals[5];
    column(retrieval_column_t::hard_above_k) = totals[6];
    column(retrieval_column_t::hards_k) = totals[7];
    column(retrieval_column_t::centred_norm_k) = live ? 1.0f - 2.0f * row_alignment + centre_norm : 0.0f;
    column(retrieval_column_t::centred_products_k) = totals[8];
}

expected<contrastive_pair_loss_t> contrastive_pair_loss_t::make(gpu_arena_t &arena, u32_t rows,
                                                                u32_t candidates_per_row,
                                                                contrastive_pair_loss_form_t form) noexcept {
    using result_t = expected<contrastive_pair_loss_t>;
    if (rows == 0 || candidates_per_row == 0)
        return result_t::fail(status_t::shape_unsupported_k, "a pair loss needs rows and candidates ({}x{})", rows,
                              candidates_per_row);
    contrastive_pair_loss_t loss;
    loss.form_ = form;
    gpu_arena_alloc<f32_t> f32_alloc {arena};
    loss.from_vertex_accumulator_ = f32r2_tensor_arena_t::uninitialized({3u, rows}, f32_alloc).value;
    if (form == contrastive_pair_loss_form_t::ranked_k)
        loss.ranked_grad_ = f32r2_tensor_arena_t::uninitialized({rows, candidates_per_row}, f32_alloc).value;
    if (!loss.from_vertex_accumulator_ || (form == contrastive_pair_loss_form_t::ranked_k && !loss.ranked_grad_))
        return result_t::fail(status_t::bad_alloc_k, "pair loss scratch allocation failed ({}x{}, arena_used={})", rows,
                              candidates_per_row, arena.used());
    return result_t {std::move(loss)};
}

expected_t contrastive_pair_loss_t::operator()(f32r4_span_t logits, contrastive_relations_view_t relations,
                                               contrastive_recipe_t recipe, f32r1_view_t live_anchors,
                                               f32r1_span_t loss_out, gpu_stream_t &stream,
                                               f32r1_span_t log_inverse_temperature_grad,
                                               f32r1_span_t bias_grad) noexcept {
    std::size_t const rows = from_vertex_accumulator_.extent(1);
    std::size_t const candidates = logits.extent(1) * logits.extent(3);
    if (logits.extent(2) != rows || relations.from_vertices_per_gpu() != rows)
        return expected_t::fail(status_t::shape_unsupported_k, "{} logit rows and {} relation rows, {} baked",
                                logits.extent(2), relations.from_vertices_per_gpu(), rows);
    if (form_ == contrastive_pair_loss_form_t::ranked_k && candidates != ranked_grad_.extent(1))
        return expected_t::fail(status_t::shape_unsupported_k, "{} candidates per row, {} baked", candidates,
                                ranked_grad_.extent(1));
    if (loss_out.empty() || live_anchors.empty())
        return expected_t::fail(status_t::invalid_input_k, "the loss and the live count need one element each");
    u32_t const groups = recipe.temperature_group_count();
    if ((!bias_grad.empty() && bias_grad.extent(0) < groups) ||
        (!log_inverse_temperature_grad.empty() && log_inverse_temperature_grad.extent(0) < groups))
        return expected_t::fail(status_t::shape_unsupported_k, "temperature gradients must span the {} groups", groups);
    dim3 const block_shape {kernel_block_size_default_k, 1u, 1u};
    dim3 const row_grid {static_cast<u32_t>(rows), 1u, 1u};
    expected_t launched {};
    if (form_ != contrastive_pair_loss_form_t::ranked_k)
        launched = launch_kernel(&contrastive_sigmoid_bce_kernel_, row_grid, block_shape, stream, logits,
                                 from_vertex_accumulator_.span(), relations, recipe, live_anchors, form_);
    for (u32_t prefix = 0; form_ == contrastive_pair_loss_form_t::ranked_k && launched && prefix != logits.extent(0);
         ++prefix) {
        launched = launch_kernel(&contrastive_ranked_bce_kernel_, row_grid, block_shape, stream, f32r4_view_t {logits},
                                 ranked_grad_.span(), from_vertex_accumulator_.span(), relations, live_anchors, prefix);
        f32r3_span_t const plane = logits[prefix, nk::slice];
        if (launched)
            launched = launch_kernel(&contrastive_ranked_copy_kernel_, dim3 {grid_size_for(plane.numel()), 1u, 1u},
                                     block_shape, stream, ranked_grad_.view(), plane);
    }
    if (launched)
        launched = launch_kernel(&contrastive_pair_loss_finalize_kernel_, dim3 {1u, 1u, 1u}, block_shape, stream,
                                 from_vertex_accumulator_.view(), recipe, loss_out, bias_grad,
                                 log_inverse_temperature_grad);
    return launched;
}

expected<contrastive_segments_t> contrastive_segments_t::make(gpu_arena_t &arena, contrastive_segments_config_t config,
                                                              u32r1_view_t prefix_dims,
                                                              gpu_numerics_t const &numerics) noexcept {
    if (config.from_vertices_per_gpu == 0 || config.hidden == 0)
        return expected<contrastive_segments_t>::fail(status_t::shape_unsupported_k,
                                                      "the segments loss scores pooled rows (vertices={} hidden={})",
                                                      config.from_vertices_per_gpu, config.hidden);
    if (config.to_vertices_per_gpu == 0 || config.gpu_subgraphs_per_step == 0)
        return expected<contrastive_segments_t>::fail(status_t::invalid_input_k,
                                                      "need at least one subgraph of at least one vertex");
    expected<matryoshka_prefixes_t> prefixes = matryoshka_prefixes_t::make(prefix_dims, config.hidden);
    if (!prefixes) return expected<contrastive_segments_t>::fail(std::move(prefixes));
    bool aligned = config.hidden % 8u == 0u;
    for (u32_t segment = 0; segment != prefixes.value_.size(); ++segment)
        aligned = aligned && prefixes.value_.segment_begin(segment) % 8u == 0u;
    if (!aligned)
        return expected<contrastive_segments_t>::fail(
            status_t::shape_unsupported_k, "hidden and every prefix must be multiples of 8 (hidden={})", config.hidden);

    contrastive_segments_t loss;
    loss.prefixes_ = prefixes.value_;
    loss.config_ = config;
    loss.numerics_ = numerics;

    // Scratch tensors — the former "workspace", owned like the relation columns are: plain
    // `uninitialized` allocations, checked in named groups so a failure names the arena pressure.

    std::size_t const rows = config.from_vertices_per_gpu, hidden = config.hidden;
    u32_t const to_vertices_per_gpu = config.to_vertices_per_gpu;
    std::size_t const prefix_count = loss.prefixes_.size(), subgraphs = config.gpu_subgraphs_per_step;
    std::size_t const block_cells = prefix_count * rows * to_vertices_per_gpu;
    std::size_t const stash_cells = block_cells * subgraphs;
    std::size_t const padded_rows = round_up_to_multiple(rows, std::size_t {8});
    std::size_t const padded_neighbors = round_up_to_multiple(to_vertices_per_gpu, std::size_t {8});
    gpu_arena_alloc<f32_t> f32_alloc {arena};
    gpu_arena_alloc<bf16_t> bf16_alloc {arena};
    gpu_arena_alloc<b8_t> byte_alloc {arena};
    loss.from_vertex_norms_ = f32r2_tensor_arena_t::uninitialized({prefix_count, rows}, f32_alloc).value;
    loss.to_vertex_norms_ =
        f32r3_tensor_arena_t::uninitialized({subgraphs, prefix_count, to_vertices_per_gpu}, f32_alloc).value;
    loss.raw_dot_ =
        f32r4_tensor_arena_t::uninitialized({prefix_count, subgraphs, rows, to_vertices_per_gpu}, f32_alloc).value;
    loss.logits_ =
        f32r4_tensor_arena_t::uninitialized({prefix_count, subgraphs, rows, to_vertices_per_gpu}, f32_alloc).value;
    loss.raw_dot_grad_ =
        f32r3_tensor_arena_t::uninitialized({prefix_count, rows, to_vertices_per_gpu}, f32_alloc).value;
    loss.raw_dot_grad_bf16_ =
        bf16r3_tensor_arena_t::uninitialized({prefix_count, rows, padded_neighbors}, bf16_alloc).value;
    loss.live_vertices_ = f32r1_tensor_arena_t::uninitialized({1}, f32_alloc).value;
    loss.from_vertex_self_coefficients_ = f32r2_tensor_arena_t::uninitialized({prefix_count, rows}, f32_alloc).value;
    loss.to_vertex_self_coefficients_ =
        f32r2_tensor_arena_t::uninitialized({prefix_count, to_vertices_per_gpu}, f32_alloc).value;
    loss.from_vertices_grad_f32_ = f32r2_tensor_arena_t::uninitialized({rows, hidden}, f32_alloc).value;
    loss.from_vertices_grad_products_ = f32r2_tensor_arena_t::uninitialized({rows, hidden}, f32_alloc).value;
    loss.vertices_transposed_ = bf16r1_tensor_arena_t::uninitialized({hidden * padded_rows}, bf16_alloc).value;
    loss.to_vertices_transposed_ = bf16r1_tensor_arena_t::uninitialized({hidden * padded_neighbors}, bf16_alloc).value;
    loss.gradient_transposed_ =
        bf16r1_tensor_arena_t::uninitialized({to_vertices_per_gpu * padded_rows}, bf16_alloc).value;
    if (!loss.from_vertex_norms_ || !loss.to_vertex_norms_ || !loss.raw_dot_ || !loss.logits_ || !loss.raw_dot_grad_ ||
        !loss.raw_dot_grad_bf16_ || !loss.live_vertices_ || !loss.from_vertex_self_coefficients_ ||
        !loss.to_vertex_self_coefficients_ || !loss.from_vertices_grad_f32_ || !loss.from_vertices_grad_products_ ||
        !loss.vertices_transposed_ || !loss.to_vertices_transposed_ || !loss.gradient_transposed_)
        return expected<contrastive_segments_t>::fail(status_t::bad_alloc_k,
                                                      "loss scratch allocation failed ({} cells, arena_used={})",
                                                      stash_cells, arena.used());
    loss.retrieval_ =
        f32r2_tensor_arena_t::uninitialized({rows, std::size_t(retrieval_column_t::count_k)}, f32_alloc).value;
    loss.world_centre_ = f32r1_tensor_arena_t::uninitialized({hidden}, f32_alloc).value;
    loss.from_vertex_alignment_ = f32r1_tensor_arena_t::uninitialized({rows}, f32_alloc).value;
    loss.to_vertex_alignment_ = f32r2_tensor_arena_t::uninitialized({subgraphs, to_vertices_per_gpu}, f32_alloc).value;
    if (!loss.retrieval_ || !loss.world_centre_ || !loss.from_vertex_alignment_ || !loss.to_vertex_alignment_)
        return expected<contrastive_segments_t>::fail(status_t::bad_alloc_k, "retrieval table allocation failed");
    expected<contrastive_pair_loss_t> pair_loss = contrastive_pair_loss_t::make(
        arena, config.from_vertices_per_gpu, static_cast<u32_t>(subgraphs * to_vertices_per_gpu), config.pair_loss);
    if (!pair_loss) return expected<contrastive_segments_t>::fail(std::move(pair_loss));
    loss.pair_loss_ = std::move(pair_loss.value_);

    std::size_t packed_bytes = 0;
    for (u32_t segment = 0; segment != prefix_count; ++segment) {
        std::size_t const width = loss.prefixes_.segment_width(segment);
        std::size_t const pack_shapes[][2] = {{to_vertices_per_gpu, width},
                                              {width, to_vertices_per_gpu},
                                              {width, rows}};
        for (auto const &[columns, depth] : pack_shapes) {
            expected<std::size_t> shape_bytes = pack_bytes_bf16(columns, depth, loss.numerics_);
            if (!shape_bytes) return expected<contrastive_segments_t>::fail(std::move(shape_bytes));
            packed_bytes = std::max(packed_bytes, shape_bytes.value_);
        }
    }
    loss.packed_ = b8r1_tensor_arena_t::uninitialized({packed_bytes}, byte_alloc).value;
    if (!loss.packed_)
        return expected<contrastive_segments_t>::fail(status_t::bad_alloc_k,
                                                      "product pack allocation failed ({} bytes)", packed_bytes);
    return expected<contrastive_segments_t> {std::move(loss)};
}

expected_t contrastive_segments_t::score_(bf16r2_view_t vertices, nk::tensor_view<gpu_subgraph_t, 1> gpu_subgraphs,
                                          contrastive_relations_view_t relations, contrastive_recipe_t recipe,
                                          gpu_stream_t &stream) noexcept {
    if (vertices.extent(0) != config_.from_vertices_per_gpu || vertices.extent(1) != config_.hidden)
        return expected_t::fail(status_t::shape_unsupported_k, "vertex tower [{}x{}] disagrees with the baked [{}x{}]",
                                vertices.extent(0), vertices.extent(1), config_.from_vertices_per_gpu, config_.hidden);
    if (gpu_subgraphs.extent(0) != config_.gpu_subgraphs_per_step)
        return expected_t::fail(status_t::shape_unsupported_k, "{} gpu_subgraphs staged, {} baked",
                                gpu_subgraphs.extent(0), config_.gpu_subgraphs_per_step);
    if (relations.from_vertices_per_gpu() != config_.from_vertices_per_gpu)
        return expected_t::fail(status_t::shape_unsupported_k,
                                "the relation graph describes {} vertices, the loss was baked for {}",
                                relations.from_vertices_per_gpu(), config_.from_vertices_per_gpu);
    std::size_t const to_vertices_per_mesh = static_cast<std::size_t>(config_.to_vertices_per_gpu) *
                                             config_.gpu_subgraphs_per_step;
    for (std::size_t subgraph_index = 0; subgraph_index != gpu_subgraphs.extent(0); ++subgraph_index) {
        gpu_subgraph_t const subgraph = gpu_subgraphs[subgraph_index];
        if (subgraph.to_vertices.extent(0) != config_.to_vertices_per_gpu ||
            subgraph.to_vertices.extent(1) != config_.hidden ||
            subgraph.to_vertices_grad.extent(0) != config_.to_vertices_per_gpu ||
            subgraph.to_vertices_grad.extent(1) != config_.hidden)
            return expected_t::fail(status_t::shape_unsupported_k, "subgraph {} disagrees with the baked [{}x{}]",
                                    subgraph_index, config_.to_vertices_per_gpu, config_.hidden);
        if (subgraph.global_offset != subgraph_index * static_cast<std::size_t>(config_.to_vertices_per_gpu))
            return expected_t::fail(status_t::shape_unsupported_k,
                                    "subgraphs must tile the global vertex axis in order (offset={} expected={})",
                                    subgraph.global_offset,
                                    subgraph_index * static_cast<std::size_t>(config_.to_vertices_per_gpu));
    }
    if (!relations.has_explicit_attract_edges() &&
        to_vertices_per_mesh < relations.self_edge_base + config_.from_vertices_per_gpu)
        return expected_t::fail( //
            status_t::shape_unsupported_k,
            "self-edges need to_vertices_per_mesh>=self_edge_base+from_vertices_per_gpu " //
            "(neighbors={} base={} vertices={})",
            to_vertices_per_mesh, relations.self_edge_base, config_.from_vertices_per_gpu);

    u32_t const prefix_count = prefixes_.size();
    std::size_t const cells_per_prefix = static_cast<std::size_t>(config_.from_vertices_per_gpu) *
                                         config_.to_vertices_per_gpu;
    dim3 const block_shape {kernel_block_size_default_k, 1u, 1u};
    dim3 const vertex_grid {config_.from_vertices_per_gpu, 1u, 1u};
    dim3 const neighbor_grid {config_.to_vertices_per_gpu, 1u, 1u};
    f32r2_view_t const from_vertex_norms_view = from_vertex_norms_.view();
    std::size_t const rows = config_.from_vertices_per_gpu, hidden = config_.hidden;
    bf16r2_span_t const vertices_padded {vertices_transposed_.data(), hidden,
                                         round_up_to_multiple(rows, std::size_t {8})};
    bf16r2_span_t const vertices_transposed = vertices_padded[nk::all, nk::range(0, rows), nk::slice];
    auto multiply = [&](bf16r2_view_t a, bf16r2_view_t b, f32r2_span_t c) noexcept {
        expected_t multiplied = pack_bf16(b, packed_.span(), numerics_, stream);
        if (multiplied) multiplied = multiply_bf16(a, packed_.view(), c, numerics_, stream);
        return multiplied;
    };

    // Per-prefix owned-vertex norms, once; each subgraph computes its own vertex norms.

    expected_t launched = launch_kernel(&contrastive_live_vertices_kernel_, dim3 {1u, 1u, 1u}, block_shape, stream,
                                        relations, to_vertices_per_mesh, live_vertices_.span());
    if (launched)
        launched = launch_kernel(&contrastive_prefix_norms_kernel_, vertex_grid, block_shape, stream, vertices,
                                 from_vertex_norms_.span(), prefixes_);
    if (launched) launched = transpose_bf16(vertices, vertices_transposed, stream);
    if (!launched) return launched;

    // Score: every subgraph's raw dots per segment over the column slices of the full `[·,hidden]`
    // tiles in that segment's prefix band; each plane adds the previous one, so plane p IS prefix
    // p's raw dot.
    for (std::size_t subgraph_index = 0; subgraph_index != gpu_subgraphs.extent(0); ++subgraph_index) {
        gpu_subgraph_t const subgraph = gpu_subgraphs[subgraph_index];
        launched = launch_kernel(&contrastive_prefix_norms_kernel_, neighbor_grid, block_shape, stream,
                                 subgraph.to_vertices, f32r2_span_t {to_vertex_norms_[subgraph_index, nk::slice]},
                                 prefixes_);
        if (!launched) return launched;
        for (u32_t segment = 0; segment != prefix_count; ++segment) {
            u32_t const segment_begin = prefixes_.segment_begin(segment);
            u32_t const segment_width = prefixes_.segment_width(segment);
            auto const band = nk::range(segment_begin, segment_begin + segment_width);
            f32r2_span_t const raw_dot_plane = raw_dot_[segment, subgraph_index, nk::slice];
            expected_t product = multiply(vertices[nk::all, band, nk::slice],
                                          subgraph.to_vertices[nk::all, band, nk::slice], raw_dot_plane);
            if (product && segment != 0) {
                f32r2_view_t const previous_plane = raw_dot_[segment - 1u, subgraph_index, nk::slice];
                product = numkong_result(numerics_.sum_f32(raw_dot_plane.data(), previous_plane.data(),
                                                           cells_per_prefix, raw_dot_plane.data(), stream.numkong()),
                                         "prefix dot nesting");
            }
            if (!product) return product;
        }
    }

    // The scaled cosines of every subgraph in one pass, which the pair loss or the evaluation
    // reads.
    return launch_kernel(&contrastive_scaled_cosine_kernel_, dim3 {grid_size_for(logits_.numel()), 1u, 1u}, block_shape,
                         stream, raw_dot_.view(), from_vertex_norms_view, to_vertex_norms_.view(), recipe,
                         logits_.span());
}

expected_t contrastive_segments_t::evaluate(bf16r2_view_t vertices, nk::tensor_view<gpu_subgraph_t, 1> gpu_subgraphs,
                                            contrastive_relations_view_t relations, contrastive_recipe_t recipe,
                                            gpu_stream_t &stream) noexcept {
    if (expected_t scored = score_(vertices, gpu_subgraphs, relations, recipe, stream); !scored) return scored;
    // The world's centre over the full-width prefix, summed subgraph by subgraph in order, and
    // every unit vertex's alignment with it.
    u32_t const prefix = prefixes_.size() - 1u;
    f32r1_span_t const centre = world_centre_[nk::range(0, prefixes_[prefix]), nk::slice];
    dim3 const block_shape {kernel_block_size_default_k, 1u, 1u};
    expected_t launched = zero_async(centre, stream);
    for (std::size_t subgraph_index = 0; launched && subgraph_index != gpu_subgraphs.extent(0); ++subgraph_index) {
        gpu_subgraph_t const subgraph = gpu_subgraphs[subgraph_index];
        std::size_t const neighbors = subgraph.to_vertices.extent(0);
        u32r1_view_t const lengths =
            relations.to_vertex_effective_length.empty()
                ? u32r1_view_t {}
                : relations
                      .to_vertex_effective_length[nk::range(subgraph.global_offset, subgraph.global_offset + neighbors),
                                                  nk::slice];
        launched = launch_kernel(&contrastive_centre_add_kernel_, dim3 {grid_size_for(centre.extent(0)), 1u, 1u},
                                 block_shape, stream, subgraph.to_vertices,
                                 f32r1_view_t {to_vertex_norms_[subgraph_index, prefix, nk::slice]}, lengths, centre);
    }
    for (std::size_t subgraph_index = 0; launched && subgraph_index != gpu_subgraphs.extent(0); ++subgraph_index)
        launched = launch_kernel(&contrastive_centre_alignment_kernel_,
                                 dim3 {static_cast<u32_t>(gpu_subgraphs[subgraph_index].to_vertices.extent(0)), 1u, 1u},
                                 block_shape, stream, gpu_subgraphs[subgraph_index].to_vertices,
                                 f32r1_view_t {to_vertex_norms_[subgraph_index, prefix, nk::slice]},
                                 f32r1_view_t {centre}, to_vertex_alignment_[subgraph_index, nk::slice]);
    if (launched)
        launched = launch_kernel(&contrastive_centre_alignment_kernel_, dim3 {config_.from_vertices_per_gpu, 1u, 1u},
                                 block_shape, stream, vertices, f32r1_view_t {from_vertex_norms_[prefix, nk::slice]},
                                 f32r1_view_t {centre}, from_vertex_alignment_.span());
    if (!launched) return launched;
    return launch_kernel(&contrastive_retrieval_kernel_, dim3 {config_.from_vertices_per_gpu, 1u, 1u}, block_shape,
                         stream, logits_.view(), retrieval_.span(), relations, recipe, f32r1_view_t {centre},
                         from_vertex_alignment_.view(), to_vertex_alignment_.view(), live_vertices_.view());
}

expected_t contrastive_segments_t::operator()(bf16r2_view_t vertices, nk::tensor_view<gpu_subgraph_t, 1> gpu_subgraphs,
                                              contrastive_relations_view_t relations, contrastive_recipe_t recipe,
                                              f32r1_span_t loss_out, bf16r2_span_t from_vertices_grad,
                                              gpu_stream_t &stream, f32r1_span_t log_inverse_temperature_grad,
                                              f32r1_span_t bias_grad) noexcept {

    if (loss_out.empty()) return expected_t::fail(status_t::invalid_input_k, "a one-element loss_out span is required");
    if (from_vertices_grad.extent(0) != config_.from_vertices_per_gpu || from_vertices_grad.extent(1) != config_.hidden)
        return expected_t::fail(status_t::shape_unsupported_k,
                                "from_vertices_grad shape disagrees with the baked [{}x{}]",
                                config_.from_vertices_per_gpu, config_.hidden);
    if (!bias_grad.empty() && bias_grad.extent(0) < recipe.temperature_group_count())
        return expected_t::fail(status_t::shape_unsupported_k, "bias_grad must span the {} pair types",
                                recipe.temperature_group_count());
    if (!log_inverse_temperature_grad.empty() &&
        log_inverse_temperature_grad.extent(0) < recipe.temperature_group_count())
        return expected_t::fail(status_t::shape_unsupported_k,
                                "log_inverse_temperature_grad must span the {} pair types",
                                recipe.temperature_group_count());
    if (expected_t scored = score_(vertices, gpu_subgraphs, relations, recipe, stream); !scored) return scored;
    // Zero the Float32 owned-vertex force; the pair loss writes every row of its own accumulator.
    for (f32r2_tensor_arena_t *const zeroed : {&from_vertices_grad_f32_, &from_vertices_grad_products_})
        if (expected_t reset = zero_async(zeroed->span(), stream); !reset) return reset;

    u32_t const prefix_count = prefixes_.size();
    std::size_t const cells_per_prefix = static_cast<std::size_t>(config_.from_vertices_per_gpu) *
                                         config_.to_vertices_per_gpu;
    std::size_t const block_cells = static_cast<std::size_t>(prefix_count) * cells_per_prefix;
    dim3 const block_shape {kernel_block_size_default_k, 1u, 1u};
    dim3 const vertex_grid {config_.from_vertices_per_gpu, 1u, 1u};
    dim3 const neighbor_grid {config_.to_vertices_per_gpu, 1u, 1u};
    dim3 const block_grid {grid_size_for(block_cells), 1u, 1u};

    // The scratch views the read-only kernel parameters bind to; the launcher copies raw bytes, so
    // spans convert to views here, explicitly.

    f32r2_view_t const from_vertex_norms_view = from_vertex_norms_.view();
    f32r2_view_t const from_vertex_self_view = from_vertex_self_coefficients_.view();
    f32r2_view_t const to_vertex_self_view = to_vertex_self_coefficients_.view();
    f32r3_view_t const raw_dot_grad_view = raw_dot_grad_.view();
    std::size_t const num_neighbors = config_.to_vertices_per_gpu;
    std::size_t const rows = config_.from_vertices_per_gpu, hidden = config_.hidden;
    std::size_t const padded_rows = round_up_to_multiple(rows, std::size_t {8});
    std::size_t const padded_neighbors = round_up_to_multiple(num_neighbors, std::size_t {8});
    bf16r2_span_t const vertices_padded {vertices_transposed_.data(), hidden, padded_rows};
    bf16r2_span_t const to_vertices_padded {to_vertices_transposed_.data(), hidden, padded_neighbors};
    bf16r2_span_t const gradient_padded {gradient_transposed_.data(), num_neighbors, padded_rows};
    bf16r2_span_t const raw_dot_grad_padded = raw_dot_grad_bf16_.reshape<2>({prefix_count * rows, padded_neighbors});
    bf16r2_span_t const vertices_transposed = vertices_padded[nk::all, nk::range(0, rows), nk::slice];
    bf16r2_span_t const to_vertices_transposed = to_vertices_padded[nk::all, nk::range(0, num_neighbors), nk::slice];
    bf16r2_span_t const gradient_transposed = gradient_padded[nk::all, nk::range(0, rows), nk::slice];
    bf16r2_span_t const raw_dot_grad_bf16 = raw_dot_grad_padded[nk::all, nk::range(0, num_neighbors), nk::slice];
    bf16r3_span_t const raw_dot_grad_planes =
        raw_dot_grad_bf16_[nk::all, nk::all, nk::range(0, num_neighbors), nk::slice];
    auto multiply = [&](bf16r2_view_t a, bf16r2_view_t b, f32r2_span_t c) noexcept {
        expected_t multiplied = pack_bf16(b, packed_.span(), numerics_, stream);
        if (multiplied) multiplied = multiply_bf16(a, packed_.view(), c, numerics_, stream);
        return multiplied;
    };

    // Loss: the pair loss over whole rows rewrites the scaled cosines into logit gradients and
    // writes the loss and the temperature summands.
    expected_t launched = pair_loss_(logits_.span(), relations, recipe, live_vertices_.view(), loss_out, stream,
                                     log_inverse_temperature_grad, bias_grad);
    if (!launched) return launched;

    // Force: every subgraph's logit gradients into the owned and the subgraph vertices.
    for (std::size_t subgraph_index = 0; subgraph_index != gpu_subgraphs.extent(0); ++subgraph_index) {
        gpu_subgraph_t const subgraph = gpu_subgraphs[subgraph_index];
        f32r3_view_t const raw_dot_view = raw_dot_[nk::all, subgraph_index, nk::slice];
        f32r3_view_t const logit_grad_view = logits_[nk::all, subgraph_index, nk::slice];
        f32r2_view_t const to_vertex_norms_view = to_vertex_norms_[subgraph_index, nk::slice];

        launched = transpose_bf16(subgraph.to_vertices, to_vertices_transposed, stream);
        if (launched)
            launched = launch_kernel(&contrastive_raw_dot_grad_kernel_, block_grid, block_shape, stream,
                                     logit_grad_view, from_vertex_norms_view, to_vertex_norms_view, recipe, relations,
                                     subgraph.global_offset, raw_dot_grad_.span());
        if (launched)
            launched = launch_kernel(&contrastive_from_vertex_self_coefficients_kernel_,
                                     dim3 {config_.from_vertices_per_gpu, prefix_count, 1u}, block_shape, stream,
                                     raw_dot_grad_view, raw_dot_view, from_vertex_self_coefficients_.span());
        if (launched)
            launched = launch_kernel(&contrastive_to_vertex_self_coefficients_kernel_,
                                     dim3 {config_.to_vertices_per_gpu, prefix_count, 1u}, block_shape, stream,
                                     raw_dot_grad_view, raw_dot_view, to_vertex_self_coefficients_.span());
        if (launched && prefix_count > 1u)
            launched = launch_kernel(&contrastive_suffix_accumulate_kernel_,
                                     dim3 {grid_size_for(cells_per_prefix), 1u, 1u}, block_shape, stream,
                                     raw_dot_grad_.span());
        if (launched)
            launched = launch_kernel(&contrastive_cast_f32_to_bf16_kernel_, block_grid, block_shape, stream,
                                     raw_dot_grad_.view().reshape<2>({prefix_count * rows, num_neighbors}),
                                     raw_dot_grad_bf16);
        if (!launched) return launched;

        // Zero the subgraph's whole force so the segment products fill disjoint bands and the channels beyond
        // the last prefix stay exactly zero.
        std::size_t const force_scalars = static_cast<std::size_t>(config_.to_vertices_per_gpu) * config_.hidden;
        if (expected_t reset = zero_async(f32r1_span_t {subgraph.to_vertices_grad.data(), force_scalars}, stream);
            !reset)
            return reset;

        // Gradients per segment: G_seg lives at the suffix-accumulated plane s, and each force fills its band of a
        // `[·,hidden]` tile — the owned one in the products scratch, added onto the accumulator below.
        f32r2_span_t const from_vertices_grad_products = from_vertices_grad_products_.span();
        for (u32_t segment = 0; segment != prefix_count; ++segment) {
            u32_t const segment_begin = prefixes_.segment_begin(segment);
            u32_t const segment_width = prefixes_.segment_width(segment);
            auto const band = nk::range(segment_begin, segment_begin + segment_width);
            bf16r2_view_t const gradient = raw_dot_grad_planes[segment, nk::slice];
            expected_t product = transpose_bf16(gradient, gradient_transposed, stream);
            if (product)
                product = multiply(gradient, to_vertices_transposed[band, nk::slice],
                                   from_vertices_grad_products[nk::all, band, nk::slice]);
            if (product)
                product = multiply(gradient_transposed, vertices_transposed[band, nk::slice],
                                   subgraph.to_vertices_grad[nk::all, band, nk::slice]);
            if (!product) return product;
        }
        launched = numkong_result(numerics_.sum_f32(from_vertices_grad_f32_.data(), from_vertices_grad_products_.data(),
                                                    rows * hidden, from_vertices_grad_f32_.data(), stream.numkong()),
                                  "vertex force addition");
        if (!launched) return launched;

        launched = launch_kernel(&contrastive_from_vertex_self_fold_kernel_, vertex_grid, block_shape, stream, vertices,
                                 from_vertex_norms_view, from_vertex_self_view, from_vertices_grad_f32_.span(),
                                 prefixes_);
        if (launched)
            launched = launch_kernel(&contrastive_to_vertex_self_fold_kernel_, neighbor_grid, block_shape, stream,
                                     subgraph.to_vertices, to_vertex_norms_view, to_vertex_self_view,
                                     subgraph.to_vertices_grad, prefixes_);
        if (!launched) return launched;
    }

    // Cast the owned-vertex force to BFloat16 — every vertex is owned by exactly one rank.
    if (from_vertices_grad.is_contiguous())
        return numkong_result(numerics_.cast(from_vertices_grad_f32_.data(), nk_f32_k, from_vertices_grad.data(),
                                             nk_bf16_k, rows * hidden, stream.numkong()),
                              "vertex force narrowing");
    return launch_kernel(&contrastive_cast_f32_to_bf16_kernel_, dim3 {grid_size_for(rows * hidden), 1u, 1u},
                         block_shape, stream, from_vertices_grad_f32_.view(), from_vertices_grad);
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
    std::span<gpu_stream_t> streams, std::span<bf16r2_view_t const> towers,
    std::span<contrastive_relations_view_t const> relations, std::span<bf16r2_span_t const> from_vertices_grads,
    contrastive_segments_config_t config, u32r1_view_t prefix_dims, gpu_numerics_t const &numerics,
    f64_t timeout_seconds) noexcept {
    using result_t = expected<distributed_contrastive_segments_t>;
    std::size_t const gpus = arenas.size();
    std::size_t const members = domain.member_count();
    u32_t const domains = colocated.multimem_domain_count();
    if (gpus == 0 || streams.size() != gpus || domain.local_member_count() != gpus || towers.size() != gpus ||
        relations.size() != gpus || from_vertices_grads.size() != gpus)
        return result_t::fail(status_t::invalid_input_k,
                              "every rank needs an arena, stream, tower, relations, gradient, " //
                              "and local domain member ({} ranks, {} local members)",
                              gpus, domain.local_member_count());
    if (config.gpu_subgraphs_per_step != members * domains ||
        config.to_vertices_per_gpu != config.from_vertices_per_gpu)
        return result_t::fail(status_t::shape_unsupported_k,
                              "the self-scoring world scores one subgraph per GPU " //
                              "(subgraphs={} members={} domains={} neighbors={} vertices={})",
                              config.gpu_subgraphs_per_step, members, domains, config.to_vertices_per_gpu,
                              config.from_vertices_per_gpu);

    distributed_contrastive_segments_t distributed;
    distributed.config_ = config;
    distributed.domain_ = &domain;
    distributed.colocated_ = &colocated;
    distributed.home_domain_index_ = domains > 1 ? colocated.topology().multimem_domain_index() : 0;
    std::size_t const subgraphs = config.gpu_subgraphs_per_step;
    std::size_t const tower_scalars = static_cast<std::size_t>(config.from_vertices_per_gpu) * config.hidden;
    if (distributed.losses_.resize(gpus) != status_t::success_k ||
        distributed.reduce_buffers_.resize(gpus) != status_t::success_k ||
        distributed.subgraph_tables_.resize(gpus) != status_t::success_k ||
        distributed.relations_.resize(gpus) != status_t::success_k ||
        distributed.from_vertices_grads_.resize(gpus) != status_t::success_k ||
        distributed.world_scalars_.resize(gpus) != status_t::success_k)
        return result_t::fail(status_t::bad_alloc_k, "rank table allocation failed ({} ranks)", gpus);
    for (std::size_t rank = 0; rank != gpus; ++rank) {
        distributed.relations_[rank] = relations[rank];
        distributed.from_vertices_grads_[rank] = from_vertices_grads[rank];
    }
    // The caller's tower views are stored once: in the subgraph tables when this process holds the whole world,
    // in the staging list otherwise — never both.
    bool const whole_world = gpus == subgraphs;
    if (!whole_world) {
        if (distributed.gather_sources_.resize(gpus) != status_t::success_k)
            return result_t::fail(status_t::bad_alloc_k, "gather source table allocation failed ({} ranks)", gpus);
        for (std::size_t rank = 0; rank != gpus; ++rank) distributed.gather_sources_[rank] = towers[rank];
    }

    // Level-1 precondition: every arena readable by every device — visibility is the arenas'
    // construction-time configuration, validated here, never granted here.
    for (std::size_t rank = 0; rank != gpus; ++rank)
        for (std::size_t peer = 0; peer != gpus; ++peer)
            if (!arenas[rank].is_visible_to(static_cast<int>(peer)))
                return result_t::fail(status_t::invalid_input_k,
                                      "rank {}'s arena is not visible to device {} — " //
                                      "construct it with `visible_to`",
                                      rank, peer);
    std::size_t const force_scalars = subgraphs * tower_scalars;
    std::size_t const world_scalars = world_scalar_count_(config);
    expected<multimem_window_t> window = domain.reserve(force_scalars + world_scalars);
    if (!window) return result_t::fail(std::move(window));
    distributed.window_ = window.value_;
    if (!whole_world) {
        expected<multimem_window_t> tower_window = domain.reserve(force_scalars * sizeof(bf16_t) / sizeof(f32_t));
        if (!tower_window) return result_t::fail(std::move(tower_window));
        distributed.world_tower_window_ = tower_window.value_;
    }

    // Every rank: the loss object and the cross-domain reduce member, both on its own device —
    // entered through the context the rank's stream already owns.
    for (std::size_t rank = 0; rank != gpus; ++rank) {
        switch_to_gpu_t gpu_scope {streams[rank]};
        if (!gpu_scope.ok()) return result_t::fail(status_t::cuda_error_k, "context entry failed (rank={})", rank);
        expected<contrastive_segments_t> loss = contrastive_segments_t::make(arenas[rank], config, prefix_dims,
                                                                             numerics);
        if (!loss) return result_t::fail(std::move(loss));
        distributed.losses_[rank] = std::move(loss.value_);
        distributed.world_scalars_[rank] =
            f32r1_tensor_arena_t::uninitialized({world_scalars}, gpu_arena_alloc<f32_t> {arenas[rank]}).value;
        if (!distributed.world_scalars_[rank])
            return result_t::fail(status_t::bad_alloc_k, "world scalars allocation failed (rank={})", rank);
        u32_t const member = static_cast<u32_t>(domain.member_offset() + rank);
        expected<infiniband_all_reduce<f32_t>> reduce_buffer = infiniband_all_reduce<f32_t>::make(
            colocated, member, arenas[rank], domains * tower_scalars + world_scalars, streams[rank], timeout_seconds);
        if (!reduce_buffer) return result_t::fail(std::move(reduce_buffer));
        distributed.reduce_buffers_[rank] = std::move(reduce_buffer.value_);
    }

    // The gather member, wired only where member 0 lives, and the per-rank loss summands on the lead device.
    {
        switch_to_gpu_t gpu_scope {streams[0]};
        if (!gpu_scope.ok()) return result_t::fail(status_t::cuda_error_k, "context entry failed (rank=0)");
        expected<infiniband_all_gather<bf16_t>> gather_buffer = infiniband_all_gather<bf16_t>::make(
            colocated, 0, arenas[0], members * tower_scalars, streams[0], timeout_seconds);
        if (!gather_buffer) return result_t::fail(std::move(gather_buffer));
        distributed.gather_buffer_ = std::move(gather_buffer.value_);
        gpu_arena_alloc<f32_t> f32_alloc {arenas[0]};
        distributed.loss_outs_ = f32r1_tensor_arena_t::uninitialized({gpus}, f32_alloc).value;
        if (!distributed.loss_outs_)
            return result_t::fail(status_t::bad_alloc_k, "scalar bookkeeping allocation failed");
    }

    // Bake every rank's subgraph table once — all the addresses are capture-stable by contract:
    // caller towers/grads and the domain windows never move after construction.
    bf16r3_span_t const world_tower = whole_world ? bf16r3_span_t {} : distributed.world_tower_();
    for (std::size_t rank = 0; rank != gpus; ++rank) {
        st::basic_vector<gpu_subgraph_t> &table = distributed.subgraph_tables_[rank];
        if (st::failed(table.resize(subgraphs)))
            return result_t::fail(status_t::bad_alloc_k, "subgraph table allocation failed (rank={})", rank);
        f32r3_span_t const forces = distributed.window_forces_(domain.member_offset() + rank);
        for (std::size_t subgraph = 0; subgraph != subgraphs; ++subgraph) {
            table[subgraph].to_vertices = whole_world ? towers[subgraph]
                                                      : bf16r2_view_t {world_tower[subgraph, nk::slice]};
            table[subgraph].global_offset = subgraph * static_cast<std::size_t>(config.from_vertices_per_gpu);
            table[subgraph].to_vertices_grad = forces[subgraph, nk::slice];
        }
    }
    return result_t {std::move(distributed)};
}

expected_t distributed_contrastive_segments_t::stage_world_tower_(std::size_t rank, gpu_stream_t &stream) noexcept {
    std::size_t const member = domain_->member_offset() + rank;
    // A whole world is read in place from the peers' towers, so every peer's forward must have
    // landed.
    if (world_tower_window_.scalars == 0) return domain_->barrier(member, stream);
    // Overwrites member 0's window, whose last readers passed the force barrier.
    std::size_t const world_gpu = home_domain_index_ * domain_->member_count() + member;
    bf16r3_span_t const world_tower = world_tower_();
    expected_t staged = copy_device_to_device(gather_sources_[rank].flatten<1>(),
                                              world_tower[world_gpu, nk::slice].flatten<1>(), stream);
    if (staged) staged = domain_->barrier(member, stream);
    if (staged && member == 0 && world_domains() > 1u) staged = gather_world_tower_(stream);
    if (staged && world_domains() > 1u) staged = domain_->barrier(member, stream);
    return staged;
}

expected_t distributed_contrastive_segments_t::evaluate(std::size_t rank, contrastive_recipe_t recipe,
                                                        gpu_stream_t &stream) noexcept {
    if (rank >= losses_.size())
        return expected_t::fail(status_t::invalid_input_k, "rank {} of {}", rank, losses_.size());
    switch_to_gpu_t gpu_scope {stream};
    if (!gpu_scope.ok()) return expected_t::fail(status_t::cuda_error_k, "context entry failed ({})", rank);
    std::size_t const member = domain_->member_offset() + rank;
    bf16r2_view_t const own_tower = world_tower_window_.scalars != 0 ? gather_sources_[rank]
                                                                     : subgraph_tables_[rank][rank].to_vertices;
    expected_t evaluated = stage_world_tower_(rank, stream);
    if (evaluated)
        evaluated = losses_[rank].evaluate(own_tower, std::span<gpu_subgraph_t const> {subgraph_tables_[rank]},
                                           relations_[rank], recipe, stream);
    // Every member has read the world tower before any member's next call stages it again.
    if (evaluated) evaluated = domain_->barrier(member, stream);
    return evaluated;
}

expected_t distributed_contrastive_segments_t::operator()(std::size_t rank, contrastive_recipe_t recipe,
                                                          gpu_stream_t &stream) noexcept {
    std::size_t const gpus = losses_.size();
    std::size_t const subgraphs = config_.gpu_subgraphs_per_step;
    std::size_t const tower_scalars = static_cast<std::size_t>(config_.from_vertices_per_gpu) * config_.hidden;
    std::size_t const force_scalars = subgraphs * tower_scalars, groups = config_.temperature_groups;
    if (rank >= gpus) return expected_t::fail(status_t::invalid_input_k, "rank {} of {}", rank, gpus);
    if (recipe.temperature_group_count() > groups)
        return expected_t::fail(status_t::shape_unsupported_k, "the recipe selects {} temperature groups of {}",
                                recipe.temperature_group_count(), groups);

    switch_to_gpu_t gpu_scope {stream};
    if (!gpu_scope.ok()) return expected_t::fail(status_t::cuda_error_k, "context entry failed ({})", rank);

    std::size_t const member = domain_->member_offset() + rank;
    expected_t staged = stage_world_tower_(rank, stream);
    nk::tensor_view<gpu_subgraph_t, 1> const table = std::span<gpu_subgraph_t const> {subgraph_tables_[rank]};
    f32r1_span_t const loss_out = loss_outs_[nk::range(rank, rank + 1), nk::slice];
    bf16r1_span_t const grad_flat = from_vertices_grads_[rank].flatten<1>();
    // The rank's own tower is stored once: in the staging list when the world is staged, otherwise as its
    // own subgraph's table entry (subgraph index == rank when this process holds the whole world).
    bf16r2_view_t const own_tower = world_tower_window_.scalars != 0 ? gather_sources_[rank]
                                                                     : subgraph_tables_[rank][rank].to_vertices;

    // The window's tail past the column forces carries this rank's log-β, bias and loss summands,
    // so one reduce sums them across the world beside the forces.
    f32r1_span_t const tail = window_tail_(member);
    std::size_t const gradients = 2u * groups;
    f32r1_span_t const log_inverse_temperature_grad = tail[nk::range(0, groups), nk::slice];
    f32r1_span_t const bias_grad = tail[nk::range(groups, gradients), nk::slice];
    f32r1_span_t const window_loss = tail[nk::range(gradients, gradients + 1), nk::slice];
    if (staged)
        staged = losses_[rank](own_tower, table, relations_[rank], recipe, window_loss, from_vertices_grads_[rank],
                               stream, log_inverse_temperature_grad, bias_grad);
    if (staged) staged = copy_device_to_device(window_loss, loss_out, stream);
    if (staged) staged = domain_->barrier(member, stream);
    // Level-1 reduce: in one domain, in place into this rank's own window shard (shards are
    // disjoint across ranks); across domains, into the rank's reduce member — one switch-reduced
    // window per domain, its own global shard in each — then the level-2 recursive-doubling sum.

    f32r1_view_t column_force {};
    if (world_domains() == 1u) {
        f32r3_span_t const forces = window_forces_(member);
        f32r1_span_t const own_shard = forces[member, nk::slice].flatten<1>();
        if (staged)
            staged = domain_->reduce_scatter(member, own_shard, window_.offset + member * tower_scalars, stream);
        if (staged)
            staged = domain_->reduce_scatter(member, world_scalars_[rank].span(), window_.offset + force_scalars,
                                             stream);
        column_force = f32r1_view_t {own_shard};
    }
    else {
        if (staged) staged = reduce_world_columns_(rank, stream, column_force);
    }
    // Every member has read every peer's window before any member's next call writes its own again.
    if (staged) staged = domain_->barrier(member, stream);
    if (staged)
        staged = launch_kernel(&contrastive_column_fold_kernel_, dim3 {grid_size_for(tower_scalars), 1u, 1u},
                               dim3 {kernel_block_size_default_k, 1u, 1u}, stream, grad_flat, column_force);
    return staged;
}

expected_t distributed_contrastive_segments_t::operator()(contrastive_recipe_t recipe,
                                                          std::span<gpu_stream_t> streams) noexcept {
    std::size_t const gpus = losses_.size();
    if (streams.size() != gpus)
        return expected_t::fail(status_t::invalid_input_k, "every rank needs its stream ({} ranks)", gpus);
    for (std::size_t rank = 0; rank != gpus; ++rank)
        if (expected_t staged = (*this)(rank, recipe, streams[rank]); !staged) return staged;
    return {};
}

expected_t distributed_contrastive_segments_t::gather_world_tower_(gpu_stream_t &stream) noexcept {
    bf16r1_span_t const own_block = gather_buffer_.own_block();
    bf16r1_span_t const world_tower = world_tower_().flatten<1>();
    bf16r2_span_t const world_tower_rows = world_tower.reshape<2>({std::size_t {world_domains()}, own_block.extent(0)});
    expected_t staged = copy_device_to_device(world_tower_rows[home_domain_index_, nk::slice], own_block, stream);
    if (staged) staged = gather_buffer_(stream);
    if (staged) staged = copy_device_to_device(gather_buffer_.payload(), world_tower, stream);
    return staged;
}

expected_t distributed_contrastive_segments_t::reduce_world_columns_(std::size_t rank, gpu_stream_t &stream,
                                                                     f32r1_view_t &column_force) noexcept {
    std::size_t const members = domain_->member_count();
    std::size_t const member = domain_->member_offset() + rank;
    std::size_t const tower_scalars = static_cast<std::size_t>(config_.from_vertices_per_gpu) * config_.hidden;
    f32r1_span_t const shard = reduce_buffers_[rank].payload();
    std::size_t const world_columns = world_domains() * tower_scalars;
    f32r2_span_t const shard_rows = shard[nk::range(0, world_columns), nk::slice].reshape<2>(
        {std::size_t {world_domains()}, tower_scalars});
    for (u32_t domain = 0; domain != world_domains(); ++domain) {
        std::size_t const grid_offset = window_.offset +
                                        (static_cast<std::size_t>(domain) * members + member) * tower_scalars;
        if (expected_t reduced = domain_->reduce_scatter(member, shard_rows[domain, nk::slice], grid_offset, stream);
            !reduced)
            return reduced;
    }
    std::size_t const force_scalars = static_cast<std::size_t>(config_.gpu_subgraphs_per_step) * tower_scalars;
    f32r1_span_t const scalars_shard =
        shard[nk::range(world_columns, world_columns + world_scalar_count_(config_)), nk::slice];
    if (expected_t reduced = domain_->reduce_scatter(member, scalars_shard, window_.offset + force_scalars, stream);
        !reduced)
        return reduced;
    if (expected_t summed = reduce_buffers_[rank](stream); !summed) return summed;
    if (expected_t copied = copy_device_to_device(scalars_shard, world_scalars_[rank].span(), stream); !copied)
        return copied;
    column_force = shard_rows[home_domain_index_, nk::slice];
    return {};
}
