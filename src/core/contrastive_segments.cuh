/**
 *  @file src/core/contrastive_segments.cuh
 *  @author Ash Vardanian
 *  @date July 2, 2026
 *  @brief Sharded sigmoid-pairwise (SigLIP-BCE) contrastive loss over pooled @b segment embeddings
 *      — the Embedder objective, as a function object with owned scratch and tensor-core matmuls.
 *
 *  The loss is best read as a @b force-directed @b graph @b layout relaxing in embedding space: every
 *  pooled embedding is a vertex, the relation graph's attract edges are springs pulling matched pairs
 *  together on the cosine sphere, repel edges and the ambient repulsion push everything else apart, and
 *  one call evaluates the potential (the weighted sigmoid-BCE on the Matryoshka cosine) and its forces
 *  (the gradients). Because the potential is a plain @b sum over independent vertex pairs (no softmax /
 *  InfoNCE normalisation), it distributes trivially: the global vertex axis partitions into
 *  @b per-GPU @b subgraphs and the call loops them, scoring each against the rank-owned vertices and
 *  accumulating. No @b [global_batch,hidden] embedding block ever materialises.
 *
 *  @section gemms The heavy passes are tensor-core matmuls
 *
 *  The nested prefixes decompose into disjoint @b segments @c [D_{p-1},D_p) , and every O(R²·H) pass is a
 *  BFloat16 → Float32 NumKong product over segments, its B operand packed at call time:
 *
 *      - scores: @c raw_p = V_seg·N_segᵀ + raw_{p-1} per segment (owned vertices × subgraph vertices), so
 *        prefix @c p 's raw dot builds on prefix @c p-1 's — the nesting costs one full-width product, not
 *        @c K ;
 *      - from-vertex forces: @c from_vertices_grad[:,seg] += G_seg·N_seg , with @c G_seg the suffix sum of
 *        the per-prefix @c d(raw) blocks over the prefixes containing the segment;
 *      - to-vertex forces: @c subgraph.to_vertices_grad[:,seg] += G_segᵀ·V_seg .
 *
 *  Hand-rolled kernels keep only the O(R²·K) and O(R·H·K) epilogues: norms, the scaled-cosine
 *  scaling, the pair loss with the relation resolution, the @c d(raw) build and its suffix
 *  accumulation, the cosine self-coefficient folds, the finalise, and the casts. Determinism holds
 *  through the products: NumKong has no split-K, so every run and every replay sums in the same
 *  order; the packs and transposes are owned arena tensors, so no allocation hides inside a call —
 *  the capture guard of @ref gpu_replay_graph_t::make would name it if it did.
 *
 *  @section gpu_subgraphs The force grid as a per-GPU subgraph loop
 *
 *  The global force grid is @b [from_vertices_per_gpu,to_vertices_per_mesh] . Each GPU's subgraph owns a
 *  contiguous run of the global vertex axis starting at its @c global_offset ; the subgraphs tile the
 *  axis in order, without gap or overlap. The mesh is homogeneous, so @b every subgraph is
 *  @b [to_vertices_per_gpu,hidden] — the shape is baked at construction and ragged reality is PAD in the
 *  relation view, never a shape change. A subgraph's @c to_vertices view may point at:
 *
 *      - this rank's own vertices — the single-GPU case, or several local tiles of one bag;
 *      - a peer GPU's vertices inside this host_domain, reachable over NVLink peer mappings;
 *      - a @b landed copy of a remote multimem_domain's vertices, delivered into a gather handle's
 *        symmetric heap by @ref infiniband_all_gather — the cross-cluster case. The math is
 *        placement-blind: the transport fills subgraphs, the loss never knows the difference.
 *
 *  @section labels Edges come from the relation graph
 *
 *  Every pair's target and weight resolve through @ref contrastive_pair_relation against the
 *  @ref contrastive_relations_view_t taken by value at launch: PAD > ignore > self > key > attract
 *  > repel > ambient repulsion, with the self-edge default when no explicit edges or keys are
 *  staged.
 *
 *  @section balanced Score, loss and force passes
 *
 *  A call scores every subgraph into a @b [prefixes,subgraphs,rows,neighbors] stash of scaled
 *  cosines, hands the whole rows to @ref contrastive_pair_loss_t , which rewrites them into logit
 *  gradients, and then turns each subgraph's gradients into forces. Under the balanced form,
 *  attractive and repulsive contributions are normalised @b separately by their own per-vertex
 *  weight totals @c Σw⁺(i) / @c Σw⁻(i) and averaged over the @b live global vertex count, PAD
 *  excluded: @c loss = Σ_i (pos_i/Σw⁺(i) + neg_i/Σw⁻(i)) / live_vertices_per_mesh . The ranked form
 *  couples each row's positives with its negatives instead, as @ref contrastive_pair_loss_t states.
 *
 *  @section distributed Distributed contract — rank-local object, collectives at the boundaries
 *
 *  A call computes the rank-local contribution: @c loss_out is this rank's per-vertex-mean potential
 *  summand (one all-reduce-sum yields the global mean), @c from_vertices_grad is the full force on the
 *  rank-owned vertices, and each subgraph's @c to_vertices_grad is this rank's Float32 force contribution
 *  to that subgraph's vertices (reduce-scatter-summed across ranks by the caller —
 *  @ref infiniband_all_reduce across domains — then cast). Ordering is the caller's; no
 *  rank writes memory another rank reads during the loss.
 *
 *  @section traffic What the distributed case actually moves
 *
 *  With @c D multimem_domains, @c R vertex rows per domain and @c H hidden, one loss step exchanges per
 *  domain, each way:
 *
 *      - vertex all-gather: @c (D-1)·R·H·2 bytes — recursive doubling sends @c 2^r subgraphs in round
 *        @c r , summing to every other domain's BFloat16 vertices exactly once;
 *      - force all-reduce-sum: @c log2(D)·R·H·4 bytes — the Float32 contribution to each remote
 *        subgraph, once per round;
 *      - loss / β / bias reduces: @c O(D) scalars — noise.
 *
 *  Total ≈ @c (2·(D-1)+4·log2(D))·R·H bytes per domain per step — @b linear in the domain's own vertex
 *  count and independent of the global batch beyond it: doubling the world doubles the repulsors every
 *  vertex feels while each domain's traffic grows by one more subgraph, not by the whole world.
 *
 *  @section ownership Owned scratch, baked shapes
 *
 *  @ref contrastive_segments_t owns every scratch tensor (arena-backed NumKong tensors, like
 *  @ref contrastive_relations_t owns its columns) and every product's packing buffer. Construction is the
 *  validation and the sizing; a call is a pure enqueue. A different shape is a different object — exactly the
 *  capture contract of @ref multimodal_batch_shape_t .
 */
#pragma once

#include <cstddef> // `std::size_t`, `std::byte`
#include <cmath>   // `sqrtf`, `expf`, `log1pf`, `fabsf`, `fmaxf`, `isfinite`

#include <span> // `std::span` — per-rank object lists

#include <cuda.h>      // `cuMemsetD8Async`, `cuMemcpyDtoDAsync`
#include <cuda_bf16.h> // `__nv_bfloat16`

#include <numkong/tensor.hpp> // `nk::tensor_view`, `nk::tensor_span`, `nk::tensor`

#include "core/types.hpp"                  // `expected_t`, `status_t`, `warp_size_k`, `kernel_block_size_default_k`
#include "core/schedule.cuh"               // `gpu_stream_t`, `launch_kernel`, the NumKong products
#include "core/multimem_collectives.cuh"   // `multimem_domain_t` — the main-owned level-1 force surface
#include "core/infiniband_collectives.cuh" // `colocated_domain_t`, `infiniband_all_reduce/all_gather` — level-2
#include "core/contrastive_relations.cuh"  // `contrastive_relations_view_t`, `contrastive_recipe_t`

namespace unum::uform::core {

/** Most Matryoshka prefix widths one loss supports; the dims travel to every kernel by value
 *  inside @ref matryoshka_prefixes_t , so the bound keeps the kernel-argument footprint fixed. */
inline constexpr u32_t max_matryoshka_prefixes_k = 8u;

/**
 *  @brief Denominator floor for cosine normalisation @c ‖q‖·‖c‖ .
 *
 *  This is the @b stability floor of the contrastive backward, not a mere divide-by-zero guard. The
 *  cosine gradient carries a @c 1/(‖q‖·‖c‖) factor and the self-fold a @c 1/‖·‖² factor; when pooled
 *  embeddings collapse to a small norm, those factors amplify the gradient without bound. Flooring caps
 *  the backward gain at @c β/floor while healthy unit-ish embeddings sit far above it.
 */
inline constexpr f32_t cosine_denominator_floor_k = 1.0e-2f;

/** Which pair loss turns an anchor row's scores into logit gradients. */
enum class contrastive_pair_loss_form_t : u32_t {

    /** The sigmoid BCE of every cell, its attract and repel halves normalized apart. */
    balanced_k = 0,

    /** The sigmoid BCE of every positive's score over every negative's, where the bias cancels. */
    ranked_k = 1,

    /** SigLIP's sigmoid BCE of every cell, each repel weighed like an attract, so the negatives sum
     *  instead of sharing one half of the row's loss. */
    sigmoid_k = 2,
};

/** Floor on a per-vertex @c Σw denominator so a fully-masked row yields a finite zero loss and
 *  zero gradient instead of a divide-by-zero. */
inline constexpr f32_t weight_sum_floor_k = 1.0e-12f;

/** The three rows of the pair loss's @b [3,rows] accumulator, in storage order. */
enum class pair_loss_row_t : u32_t {
    loss_k = 0,
    bias_grad_k = 1,
    log_inverse_temperature_grad_k = 2,
};

/** The per-row sums of a retrieval evaluation, in the columns of its table. Ranked rows hold a
 *  positive and a negative; a hit ranks its best positive first, or within the top five; chance is
 *  the share of positives among the scored cells; the cosine and hard-negative columns count the
 *  scored cells; the last two hold the row's squared norm and its squared products with every live
 *  vertex once the world's mean unit vector is subtracted, for the effective rank. */
enum class retrieval_column_t : u32_t {
    ranked_k = 0,
    hit_at_1_k,
    hit_at_5_k,
    chance_k,
    attract_cosine_k,
    attracts_k,
    ambient_cosine_k,
    ambients_k,
    hard_above_k,
    hards_k,
    centred_norm_k,
    centred_products_k,
    count_k,
};

/** The positives one pass over a ranked row's negatives sums at once: the register tile of their
 *  per-positive totals. */
inline constexpr u32_t ranked_positives_per_pass_k = 8u;

/** One GPU's subgraph of the force layout: its contiguous run of the global vertex axis — always
 *  @b [to_vertices_per_gpu,hidden] , wherever the transport put it (local tile, NVLink peer memory, or
 *  a landed remote copy). Vertex existence (PAD) is state of the relation view, not of the subgraph. */
struct gpu_subgraph_t {

    /** @b [to_vertices_per_gpu,hidden] BFloat16 to-vertex (column) tower of this block, row-major. */
    bf16r2_view_t to_vertices {};

    /** Global vertex id of this subgraph's row 0 — shifts its local rows onto the global vertex
     *  axis. */
    std::size_t global_offset = 0;

    /** @b [to_vertices_per_gpu,hidden] Float32 output — this rank's force contribution to the
     *  subgraph's to-vertices; Float32 so the caller can reduce-scatter-sum without per-rank
     *  rounding. */
    f32r2_span_t to_vertices_grad {};
};

/** The validated Matryoshka prefix widths — a fixed-capacity value type every kernel takes by
 *  value. Construction is the validation: @ref make rejects an empty, oversized, non-ascending, or
 *  over-wide ladder, so an instance is always well-formed. */
struct matryoshka_prefixes_t {

    /** Validate @p prefix_dims (strictly ascending, each @c >=1 and @c <=hidden , at most
     *  @ref max_matryoshka_prefixes_k entries) and adopt them. */
    [[nodiscard]] static expected<matryoshka_prefixes_t> make(u32r1_view_t prefix_dims, std::size_t hidden) noexcept;

    /** Number of nested prefixes. */
    [[nodiscard]] constexpr u32_t size() const noexcept { return count_; }

    /** Width of prefix @p index — strictly ascending in @p index . */
    [[nodiscard]] constexpr u32_t operator[](u32_t index) const noexcept { return dims_[index]; }

    /** First channel of segment @p index — the previous prefix's width, 0 for the first. */
    [[nodiscard]] constexpr u32_t segment_begin(u32_t index) const noexcept {
        return index != 0 ? dims_[index - 1] : 0u;
    }

    /** Channel count of segment @p index . */
    [[nodiscard]] constexpr u32_t segment_width(u32_t index) const noexcept {
        return dims_[index] - segment_begin(index);
    }

  private:
    u32_t dims_[max_matryoshka_prefixes_k] {};
    u32_t count_ = 0;
};

#if defined(__CUDACC__)

/** The per-vertex logit scale β: the device entry selected by the from-vertex's temperature group,
 *  or the host fallback when no device view is bound. */
[[nodiscard]] __device__ inline f32_t contrastive_from_vertex_beta_(contrastive_recipe_t const &recipe,
                                                                    u32_t vertex_index) noexcept {
    if (recipe.inverse_temperature_device.empty()) return recipe.inverse_temperature;
    temperature_group_t const temperature_group = recipe.from_vertex_temperature_group.empty()
                                                      ? temperature_group_t {0}
                                                      : recipe.from_vertex_temperature_group[vertex_index];
    return recipe.inverse_temperature_device[static_cast<u32_t>(temperature_group)];
}

/** The per-vertex additive logit bias — the sibling of @ref contrastive_from_vertex_beta_ . */
[[nodiscard]] __device__ inline f32_t contrastive_from_vertex_bias_(contrastive_recipe_t const &recipe,
                                                                    u32_t vertex_index) noexcept {
    if (recipe.bias_device.empty()) return recipe.bias;
    temperature_group_t const temperature_group = recipe.from_vertex_temperature_group.empty()
                                                      ? temperature_group_t {0}
                                                      : recipe.from_vertex_temperature_group[vertex_index];
    return recipe.bias_device[static_cast<u32_t>(temperature_group)];
}

/**
 *  @brief The pair loss over whole anchor rows, the component every contrastive loss owns: it turns
 *      a stash of scaled scores into logit gradients in place, then folds the loss and the
 *      temperature summands.
 *
 *  Row @c r is local vertex @c r , and candidate @c s·per_subgraph+n of the stash is global vertex
 *  @c s·per_subgraph+n , so the subgraphs tile the global axis in order. Under @c balanced_k every
 *  cell is a sigmoid BCE on @c β·score+bias , its attract and repel halves normalized apart, and
 *  under @c sigmoid_k every repel is weighed on the attracts' scale instead. Under @c ranked_k ,
 *  row @c i scores @c Σ_p Σ_n w_p·w_n·softplus(x_n−x_p)/(Σw_p·Σw_n) over the attract set @c P_i
 *  and the repel and ambient set @c N_i , where the bias cancels and its gradient is zero. All
 *  three average over the live anchors, and every fold runs in a fixed order.
 */
struct contrastive_pair_loss_t {

    /** Allocate the @b [3,rows] accumulator, and under @c ranked_k the @b [rows,candidates_per_row]
     *  gradient stash the positives' scores must outlive. */
    [[nodiscard]] static expected<contrastive_pair_loss_t> make(gpu_arena_t &arena, u32_t rows,
                                                                u32_t candidates_per_row,
                                                                contrastive_pair_loss_form_t form) noexcept;

    /**
     *  @brief Rewrite @p logits from scaled scores into @c d(loss)/d(logit) and write the loss and
     *      the temperature summands, enqueue-only and capture-safe.
     *
     *  @param[inout] logits @b [prefixes,subgraphs,rows,per_subgraph] Float32: @c β·score without
     *      the bias in, the logit gradient out.
     *  @param[in] relations The relation graph over the rows and the global candidates.
     *  @param[in] recipe β and bias, host scalars or per-temperature-group device entries.
     *  @param[in] live_anchors @b [1] the mean divisor, floored at one here.
     *  @param[out] loss_out @b [1] Float32 summand of the mean. Overwritten.
     *  @param[in] stream The stream every kernel is enqueued on.
     *  @param[out] log_inverse_temperature_grad Optional @b [temperature_group_count] summands;
     *      empty disables.
     *  @param[out] bias_grad Optional @b [temperature_group_count] summands, zero under the ranked
     *      form; empty disables.
     */
    [[nodiscard]] expected_t operator()(f32r4_span_t logits, contrastive_relations_view_t relations,
                                        contrastive_recipe_t recipe, f32r1_view_t live_anchors, f32r1_span_t loss_out,
                                        gpu_stream_t &stream, f32r1_span_t log_inverse_temperature_grad = {},
                                        f32r1_span_t bias_grad = {}) noexcept;

  private:

    /** @b [3,rows] per-row loss, bias and log-β gradient summands, in @ref pair_loss_row_t
     *  order. */
    f32r2_tensor_arena_t from_vertex_accumulator_ {};

    /** @b [rows,candidates_per_row] one prefix's ranked gradients, written while its scores are
     *  still read; empty unless @c ranked_k . */
    f32r2_tensor_arena_t ranked_grad_ {};

    /** The pair loss baked at @ref make . */
    contrastive_pair_loss_form_t form_ = contrastive_pair_loss_form_t::balanced_k;
};

/** The baked geometry of one segments loss — every capacity a call may not exceed; a different
 *  config is a different object, exactly the capture contract of @ref multimodal_batch_shape_t . */
struct contrastive_segments_config_t {

    /** Owned (pooled) from-vertex rows this rank scores — the force grid's row count. */
    u32_t from_vertices_per_gpu = 0;

    /** Embedding width shared by both towers. */
    u32_t hidden = 0;

    /** To-vertex rows in every GPU's subgraph — world-uniform, the homogeneous-mesh guarantee. */
    u32_t to_vertices_per_gpu = 0;

    /** Subgraphs scored per step — the world's GPU count when fully gathered. */
    u32_t gpu_subgraphs_per_step = 0;

    /** Learnable β and bias entries the recipe selects by group, sizing the temperature grads. */
    u32_t temperature_groups = 1;

    /** The pair loss over each anchor row. */
    contrastive_pair_loss_form_t pair_loss = contrastive_pair_loss_form_t::balanced_k;
};

/** The segments contrastive loss as a function object: shapes, Matryoshka prefixes, and every
 *  scratch tensor are baked at @ref make — a call is a pure enqueue, capture-safe and bit-reproducible
 *  (see the file header). */
struct contrastive_segments_t {

    /**
     *  @brief Bake everything static: validate the prefix ladder, and allocate every scratch tensor and the
     *      products' packing buffer off @p arena — so a constructed loss never sizes and never allocates.
     *
     *  @param[in] config The baked geometry (see @ref contrastive_segments_config_t ); @c hidden a multiple of 8.
     *  @param[in] prefix_dims Matryoshka prefix widths, validated through @ref matryoshka_prefixes_t , each
     *      a multiple of 8 so every segment's column band starts on NumKong's 16-byte boundary.
     */
    [[nodiscard]] static expected<contrastive_segments_t> make(gpu_arena_t &arena, contrastive_segments_config_t config,
                                                               u32r1_view_t prefix_dims) noexcept;

    /**
     *  @brief Rank-local forward + backward for one step — enqueue-only, capture-safe. Inputs and outputs
     *      are the only per-call state; every intermediate lives inside the object.
     *
     *  @param[in] from_vertices This rank's owned @b [rows,hidden] BFloat16 from-vertex tower.
     *  @param[in] gpu_subgraphs @b [gpu_subgraphs_per_step] descriptors, each @b [to_vertices_per_gpu,hidden] ,
     *      tiling the global vertex axis in order.
     *  @param[in] relations The staged relation graph view; @c from_vertices_per_gpu() must equal the baked
     *      rows.
     *  @param[in] recipe β / bias / ambient-repulsion weight, host scalars or per-temperature-group device
     *      entries.
     *  @param[out] loss_out @b [1] Float32 rank-local summand of the mean over the live vertices of
     *      the global axis, which the relations' to-vertex lengths count. Overwritten.
     *  @param[out] from_vertices_grad @b [rows,hidden] BFloat16 force on the owned from-vertices. Overwritten.
     *  @param[in] stream The stream every kernel and matmul is enqueued on.
     *  @param[out] log_inverse_temperature_grad Optional @b [temperature_group_count] summands; empty disables.
     *  @param[out] bias_grad Optional @b [temperature_group_count] summands; empty disables.
     */
    [[nodiscard]] expected_t operator()(bf16r2_view_t from_vertices, nk::tensor_view<gpu_subgraph_t, 1> gpu_subgraphs,
                                        contrastive_relations_view_t relations, contrastive_recipe_t recipe,
                                        f32r1_span_t loss_out, bf16r2_span_t from_vertices_grad, gpu_stream_t &stream,
                                        f32r1_span_t log_inverse_temperature_grad = {},
                                        f32r1_span_t bias_grad = {}) noexcept;

    /** Score the vertices as a call does, at the full width and without the loss, then fold each
     *  row's retrieval into the @b [rows,retrieval_columns] table @ref retrieval reads. */
    [[nodiscard]] expected_t evaluate(bf16r2_view_t from_vertices, nk::tensor_view<gpu_subgraph_t, 1> gpu_subgraphs,
                                      contrastive_relations_view_t relations, contrastive_recipe_t recipe,
                                      gpu_stream_t &stream) noexcept;

    /** The @b [rows,retrieval_columns] table of the last @ref evaluate , in @ref retrieval_column_t
     *  order. */
    [[nodiscard]] f32r2_view_t retrieval() const noexcept { return retrieval_.view(); }

  private:

    /** Validate a call's views and write the scaled cosines of every subgraph into
     *  @ref logits_ . */
    [[nodiscard]] expected_t score_(bf16r2_view_t from_vertices, nk::tensor_view<gpu_subgraph_t, 1> gpu_subgraphs,
                                    contrastive_relations_view_t relations, contrastive_recipe_t recipe,
                                    gpu_stream_t &stream) noexcept;

    /** The baked geometry and the validated prefix ladder — every capacity a call is checked
     *  against. */
    matryoshka_prefixes_t prefixes_ {};
    contrastive_segments_config_t config_ {};

    /** @b [rows,retrieval_columns] per-row retrieval sums of the last @ref evaluate . */
    f32r2_tensor_arena_t retrieval_ {};

    /** @b [hidden] sum of the world's live unit vertices, which @ref evaluate centres on, and the
     *  @b [rows] / @b [subgraphs,neighbors] products of every unit vertex with that sum. */
    f32r1_tensor_arena_t world_centre_ {}, from_vertex_alignment_ {};
    f32r2_tensor_arena_t to_vertex_alignment_ {};

    /** @b [prefixes,rows] per-row L2 norms of the from-vertex tower, and
     *  @b [subgraphs,prefixes,neighbors] those of every subgraph's to-vertex tower. */
    f32r2_tensor_arena_t from_vertex_norms_ {};
    f32r3_tensor_arena_t to_vertex_norms_ {};

    /** @b [prefixes,subgraphs,rows,neighbors] per-prefix raw dot products of every subgraph, and
     *  the logits over the same cells: the scaled cosines (β·cos, no bias) the pair loss rewrites
     *  into logit gradients. */
    f32r4_tensor_arena_t raw_dot_ {}, logits_ {};

    /** @b [prefixes,rows,neighbors] raw-dot gradient of one subgraph block. */
    f32r3_tensor_arena_t raw_dot_grad_ {};

    /** @b [prefixes,rows,neighbors] BFloat16 cast of @ref raw_dot_grad_ — the gradient products'
     *  operand, its rows padded to 8 for NumKong's 16-byte stride rule. */
    bf16r3_tensor_arena_t raw_dot_grad_bf16_ {};

    /** The pair loss over whole rows, baked from the config's form. */
    contrastive_pair_loss_t pair_loss_ {};

    /** @b [1] live vertices of the global axis, counted every call — the mean divisor. */
    f32r1_tensor_arena_t live_vertices_ {};

    /** @b [prefixes,rows] / @b [prefixes,neighbors] deterministic cosine self-coefficients. */
    f32r2_tensor_arena_t from_vertex_self_coefficients_ {}, to_vertex_self_coefficients_ {};

    /** @b [rows,hidden] Float32 owned-vertex force accumulator, cast to BFloat16 at the end, and one
     *  subgraph's segment products before they add onto it — zeroed every call, since no prefix writes the
     *  channels past the last one. */
    f32r2_tensor_arena_t from_vertices_grad_f32_ {}, from_vertices_grad_products_ {};

    /** @b [hidden,rows] , @b [hidden,neighbors] and @b [neighbors,rows] BFloat16 transposes of the owned
     *  vertices, a subgraph's vertices and one segment's gradient, their rows padded to 8 — the products' B
     *  operands come from the first two, the to-vertex force's A operand is the third. */
    bf16r1_tensor_arena_t vertices_transposed_ {}, to_vertices_transposed_ {}, gradient_transposed_ {};

    /** The one NumKong pack every product's B operand passes through, sized for the largest. */
    b8r1_tensor_arena_t packed_ {};

    /** The NumKong capabilities of the arena's device, picking every product. */
    nk_capability_t capabilities_ = 0;
};

/**
 *  @brief The distributed sibling of @ref contrastive_segments_t : one object owning a whole world's
 *      force evaluation.
 *
 *  It holds the per-rank loss objects, a reserved window of the main-owned @ref multimem_domain_t whose
 *  replicas @b are the column-force surfaces, and the cross-domain members wired through the main-owned
 *  @ref colocated_domain_t .
 *
 *  This signature expresses the self-scoring configuration: @c towers serves as both the owned rows and
 *  the subgraph columns of one global vertex axis, so the row-side and column-side forces of every
 *  vertex always fold — a bipartite mesh is a different signature, not a flag. One call runs: the
 *  world-tower staging and gather (when no process holds the whole world), then per rank the loss (row force straight into the
 *  caller's gradient, column forces into the domain window), the round barrier, the switch
 *  reduce-scatter, the cross-domain force all-reduce, and the column fold — enqueue-only,
 *  deterministic, capture-friendly.
 */
struct distributed_contrastive_segments_t {

    /** How many Float32 payload scalars this consumer will @c reserve from the shared
     *  @ref multimem_domain_t over @p shape — for main's domain sizing. The column forces and the
     *  temperature gradients share one window, and a world no single process holds whole also needs
     *  the BFloat16 world tower, staged in member 0's window. */
    [[nodiscard]] static constexpr std::size_t payload_scalars_needed(contrastive_segments_config_t config,
                                                                      mesh_shape_t const &shape) noexcept {
        std::size_t const forces = static_cast<std::size_t>(config.gpu_subgraphs_per_step) *
                                   config.from_vertices_per_gpu * config.hidden;
        std::size_t const window = forces + world_scalar_count_(config);
        if (shape.gpus_per_host_domain == shape.gpus_per_colocated_domain) return window;
        return multimem_domain_t::window_scalars(window) + forces * sizeof(bf16_t) / sizeof(f32_t);
    }

    /**
     *  @brief One construction absorbs the whole world: validates every arena's visibility, builds
     *      every rank's loss object, reserves this consumer's window of the shared @p domain , and wires
     *      its cross-domain members through @p colocated .
     *
     *  The caller (main) owns the infrastructure, this object only borrows it.
     *
     *  @param[inout] domain The main-owned level-1 surface; one force window is reserved from it, and the world
     *      tower window when this process does not hold the whole world.
     *  @param[inout] colocated The main-owned level-2 surface; reduce members are wired per rank and
     *      one gather member on the gateway — zero-round identities in single-domain worlds.
     *  @param[in] arenas @b [gpus] per-rank arenas, one per member this process drives
     *      ( @c [domain.member_offset(),+gpus) ); loss scratch and member heaps allocate from each rank's own.
     *  @param[in] streams @b [gpus] per-rank streams; their contexts scope every per-device setup.
     *  @param[in] towers @b [gpus] per-rank pooled towers — capture-stable device views, baked here so
     *      the per-step slices carry no host arguments.
     *  @param[in] relations @b [gpus] per-rank relation views — addresses baked, contents restaged by
     *      the caller per step.
     *  @param[in] from_vertices_grads @b [gpus] per-rank from-vertex-force outputs — baked like the towers.
     *  @param[in] config The per-rank geometry; @c gpu_subgraphs_per_step must equal the @b world GPU
     *      count and both row counts must agree (the self-scoring shape).
     *  @param[in] prefix_dims Matryoshka prefix widths, shared by every rank.
     */
    [[nodiscard]] static expected<distributed_contrastive_segments_t> make(
        multimem_domain_t &domain, colocated_domain_t &colocated, std::span<gpu_arena_t> arenas,
        std::span<gpu_stream_t> streams, std::span<bf16r2_view_t const> towers,
        std::span<contrastive_relations_view_t const> relations, std::span<bf16r2_span_t const> from_vertices_grads,
        contrastive_segments_config_t config, u32r1_view_t prefix_dims, f64_t timeout_seconds = 120.0) noexcept;

    /** The @b [gpus] rank-local loss summands of the last call — sum for the global mean. */
    [[nodiscard]] f32r1_view_t loss_outs() const noexcept { return loss_outs_.view(); }

    /** Local rank @p rank 's world-summed temperature gradients of the last call, on its device:
     *  @b [2·temperature_groups] , the log-β gradient of every group, then its bias gradient.
     *  Identical on every rank, so every replica takes the identical temperature step. */
    [[nodiscard]] f32r1_view_t temperature_grads(std::size_t rank) const noexcept {
        return world_scalars_[rank][nk::range(0, 2u * std::size_t {config_.temperature_groups}), nk::slice];
    }

    /** Local rank @p rank 's @b [1] world loss of the last call, every process's summands
     *  added, on its own device. */
    [[nodiscard]] f32r1_view_t world_loss(std::size_t rank) const noexcept {
        std::size_t const gradients = 2u * std::size_t {config_.temperature_groups};
        return world_scalars_[rank][nk::range(gradients, gradients + 1u), nk::slice];
    }

    /** How many multimem_domains this object spans — 1 means the members are identities. Derived
     *  from the retained @ref colocated_domain_t topology, never a stored copy. */
    [[nodiscard]] u32_t world_domains() const noexcept { return colocated_->multimem_domain_count(); }

    /**
     *  @brief Local rank @p rank 's slice of one distributed evaluation — pure enqueue onto @p stream , the
     *      shape a fork/join capture body composes directly.
     *
     *  When no process holds the whole world, every member first stages its tower into member 0's window and
     *  barriers; across domains member 0 then gathers the remote towers into that window and a second barrier
     *  releases the readers. All views were baked at @ref make ; the only per-call argument is the
     *  recipe. Ranks must be enqueued in order @c 0..gpus-1 within a step.
     */
    [[nodiscard]] expected_t operator()(std::size_t rank, contrastive_recipe_t recipe, gpu_stream_t &stream) noexcept;

    /** Local rank @p rank 's evaluation of the world tower, staged as a call stages it, without the
     *  loss and its forces; ranks are enqueued in order within one evaluation. */
    [[nodiscard]] expected_t evaluate(std::size_t rank, contrastive_recipe_t recipe, gpu_stream_t &stream) noexcept;

    /** Local rank @p rank 's retrieval table of the last evaluation, on its device. */
    [[nodiscard]] f32r2_view_t retrieval(std::size_t rank) const noexcept { return losses_[rank].retrieval(); }

    /** The eager convenience: every rank's slice in order over the per-rank streams. */
    [[nodiscard]] expected_t operator()(contrastive_recipe_t recipe, std::span<gpu_stream_t> streams) noexcept;

    /** Rebind rank @p rank 's relation view after its owner restaged edge sets of new extents — outside any
     *  capture, before the next call. */
    void rebind_relations(std::size_t rank, contrastive_relations_view_t relations) noexcept {
        relations_[rank] = relations;
    }

  private:

    /** The scalars riding the force window's tail into the world reduce: the log-β and bias
     *  gradients of every temperature group, then the loss summand. */
    [[nodiscard]] static constexpr std::size_t world_scalar_count_(contrastive_segments_config_t config) noexcept {
        return 2u * std::size_t {config.temperature_groups} + 1u;
    }

    /** Make the world tower readable for rank @p rank : a whole world waits for every peer's tower
     *  in place, otherwise the rank stages its tower into member 0's window and gathers the remote
     *  domains. */
    [[nodiscard]] expected_t stage_world_tower_(std::size_t rank, gpu_stream_t &stream) noexcept;

    /** Copy this domain's staged towers into the gather member's own block, all-gather the remote
     *  domains' blocks, and copy the world tower back into member 0's window. Gateway stream only,
     *  caller already inside its context. */
    [[nodiscard]] expected_t gather_world_tower_(gpu_stream_t &stream) noexcept;

    /** Member 0's window as the @b [subgraphs,rows,hidden] BFloat16 world tower every member
     *  reads. */
    [[nodiscard]] bf16r3_span_t world_tower_() noexcept {
        std::size_t const subgraphs = config_.gpu_subgraphs_per_step, rows = config_.from_vertices_per_gpu;
        std::size_t const hidden = config_.hidden;
        bf16r1_span_t const window = typed_span<bf16_t>(domain_->window_bytes(0, world_tower_window_));
        return window[nk::range(0, subgraphs * rows * hidden), nk::slice].reshape<3>({subgraphs, rows, hidden});
    }

    /** Member @p member 's replica of this consumer's window up to its scalar tail: the
     *  @b [subgraphs,rows,hidden] column forces. */
    [[nodiscard]] f32r3_span_t window_forces_(std::size_t member) noexcept {
        std::size_t const subgraphs = config_.gpu_subgraphs_per_step, rows = config_.from_vertices_per_gpu;
        std::size_t const hidden = config_.hidden;
        f32r1_span_t const payload = domain_->payload(member);
        f32r1_span_t const window = payload[nk::range(window_.offset, window_.offset + window_.scalars), nk::slice];
        return window[nk::range(0, subgraphs * rows * hidden), nk::slice].reshape<3>({subgraphs, rows, hidden});
    }

    /** The @ref world_scalar_count_ tail past @ref window_forces_ : every temperature group's log-β
     *  summand, then its bias summand, then the loss. */
    [[nodiscard]] f32r1_span_t window_tail_(std::size_t member) noexcept {
        std::size_t const tail_begin = window_.offset + window_forces_(member).numel();
        f32r1_span_t const payload = domain_->payload(member);
        return payload[nk::range(tail_begin, window_.offset + window_.scalars), nk::slice];
    }

    /** Level-1 reduce of @p rank 's world columns — one switch-reduced window per domain into
     *  the rank's reduce member — then the level-2 recursive-doubling sum; @p column_force receives
     *  the rank's own reduced window. Caller holds the rank's context. */
    [[nodiscard]] expected_t reduce_world_columns_(std::size_t rank, gpu_stream_t &stream,
                                                   f32r1_view_t &column_force) noexcept;

    /** The baked world geometry every capacity check keys on. */
    contrastive_segments_config_t config_ {};

    /** One loss object per local GPU — per-device by nature. */
    per_gpu_array<contrastive_segments_t> losses_ {};

    /** The main-owned level-1 surface — borrowed; its replicas carry the column forces. */
    multimem_domain_t *domain_ = nullptr;

    /** The main-owned level-2 surface — borrowed; retained so @ref world_domains derives the
     *  multimem_domain span from its topology rather than caching it as a fact. */
    colocated_domain_t *colocated_ = nullptr;

    /** This consumer's disjoint window of the domain payload, reserved at @ref make . */
    multimem_window_t window_ {};

    /** The world tower's window, read only in member 0's replica; empty when this process holds the whole
     *  world and reads the towers in place. */
    multimem_window_t world_tower_window_ {};

    /** @b [gpus] rank-local loss summands on the lead device, peer-written by each rank. */
    f32r1_tensor_arena_t loss_outs_ {};

    /** Per-rank world-summed window tails, @b [2·temperature_groups+1] on each rank's arena. */
    per_gpu_array<f32r1_tensor_arena_t> world_scalars_ {};

    /** This process's own multimem_domain coordinate — the recursive-doubling home index. Genuinely
     *  baked: not recoverable from the retained topology. */
    u32_t home_domain_index_ = 0;

    /** One cross-domain all-reduce buffer per local GPU — its typed payload is the rank's world-column
     *  shard. */
    per_gpu_array<infiniband_all_reduce<f32_t>> reduce_buffers_ {};

    /** The gateway's cross-domain all-gather buffer — its typed payload is the world tower. */
    infiniband_all_gather<bf16_t> gather_buffer_ {};

    /** Every rank's @b [subgraphs] subgraph table, baked at @ref make . */
    per_gpu_array<array<gpu_subgraph_t>> subgraph_tables_ {};

    /** @b [gpus] tower views each rank stages into the world tower — populated only with
     *  @ref world_tower_window_ ; otherwise the tables read the same views in place. */
    per_gpu_array<bf16r2_view_t> gather_sources_ {};

    /** @b [gpus] per-local-GPU relation views — addresses baked, contents restaged per step. */
    per_gpu_array<contrastive_relations_view_t> relations_ {};

    /** @b [gpus] per-local-GPU from-vertex-force outputs, baked at @ref make . */
    per_gpu_array<bf16r2_span_t> from_vertices_grads_ {};
};

#endif // defined(__CUDACC__)

} // namespace unum::uform::core
