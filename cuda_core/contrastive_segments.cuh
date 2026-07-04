/**
 *  @brief Sharded sigmoid-pairwise (SigLIP-BCE) contrastive loss over pooled @b segment embeddings — the
 *      Embedder objective, as a function object with owned scratch and tensor-core matmuls. The
 *      token-level (late-interaction / Reranker) sibling lives in @c contrastive_tokens.cuh ; the two
 *      share the relation-graph label boundary of @c contrastive_relations.cuh and the balanced-BCE epilogue.
 *  @file cuda_core/contrastive_segments.cuh
 *  @author Ash Vardanian
 *  @date July 2, 2026
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
 *  BFloat16→Float32 product over segments through @ref cublaslt_plan_t — pinned at construction, enqueued
 *  at call time:
 *
 *      - scores: @c running += V_seg·N_segᵀ per segment (owned vertices × subgraph vertices); after
 *        segment @c p the running block @b is prefix @c p 's raw dot, copied out — the nesting costs one
 *        full-width product, not @c K ;
 *      - owned-vertex forces: @c vertices_grad[:,seg] += G_seg·N_seg , with @c G_seg the suffix sum of
 *        the per-prefix @c d(raw) blocks over the prefixes containing the segment;
 *      - subgraph-vertex forces: @c subgraph.vertices_grad[:,seg] += G_segᵀ·V_seg .
 *
 *  Hand-rolled kernels keep only the O(R²·K) and O(R·H·K) epilogues: norms, the similarity scaling, the
 *  balanced BCE with the relation resolution, the @c d(raw) build and its suffix accumulation, the cosine
 *  self-coefficient folds, the finalise, and the casts. Determinism holds through the vendor path: the
 *  algorithm is heuristic-chosen @b once and pinned, so every run and every replay executes the same
 *  kernel; the workspace is an owned arena tensor, so no allocation hides inside a call — the capture
 *  guard of @ref gpu_replay_graph_t::make would name it if it did.
 *
 *  @section gpu_subgraphs The force grid as a per-GPU subgraph loop
 *
 *  The global force grid is @b [vertices_per_gpu,neighbors_per_mesh] . Each GPU's subgraph owns a
 *  contiguous run of the global vertex axis starting at its @c global_offset ; the subgraphs tile the
 *  axis in order, without gap or overlap. The mesh is homogeneous, so @b every subgraph is
 *  @b [neighbors_per_gpu,hidden] — the shape is baked at construction and ragged reality is PAD in the
 *  relation view, never a shape change. A subgraph's @c vertices view may point at:
 *
 *      - this rank's own vertices — the single-GPU case, or several local tiles of one bag;
 *      - a peer GPU's vertices inside this host_domain, reachable over NVLink peer mappings;
 *      - a @b landed copy of a remote multimem_domain's vertices, delivered into a gather member's
 *        symmetric heap by @ref infiniband_member_t::allgather — the cross-cluster case. The math is
 *        placement-blind: the transport fills subgraphs, the loss never knows the difference.
 *
 *  @section labels Edges come from the relation graph
 *
 *  Every pair's target and weight resolve through @ref contrastive_pair_relation against the
 *  @ref contrastive_relations_view_t taken by value at launch: PAD > ignore > attract > repel > ambient
 *  repulsion, with the self-edge default when no explicit edges are staged.
 *
 *  @section balanced Balanced per-vertex reduction
 *
 *  Attractive and repulsive contributions are normalised @b separately by their own per-vertex weight
 *  totals @c Σw⁺(i) / @c Σw⁻(i) and averaged over the @b global vertex count:
 *  @c loss = Σ_i (pos_i/Σw⁺(i) + neg_i/Σw⁻(i)) / vertices_per_mesh .
 *
 *  @section distributed Distributed contract — rank-local object, collectives at the boundaries
 *
 *  A call computes the rank-local contribution: @c loss_out is this rank's per-vertex-mean potential
 *  summand (one all-reduce-sum yields the global mean), @c vertices_grad is the full force on the
 *  rank-owned vertices, and each subgraph's @c vertices_grad is this rank's Float32 force contribution
 *  to that subgraph's vertices (reduce-scatter-summed across ranks by the caller —
 *  @ref infiniband_member_t::allreduce_sum across domains — then cast). Ordering is the caller's; no
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
 *  @ref contrastive_relations_t owns its columns) and every matmul plan. Construction is the validation
 *  and the planning; a call is a pure enqueue. A different shape is a different object — exactly the
 *  capture contract of @ref multimodal_batch_shape_t .
 */
#pragma once

#include <array>   // `std::array` — the fixed-capacity pinned-plan table
#include <cmath>   // `sqrtf`, `expf`, `log1pf`, `fabsf`, `fmaxf`, `isfinite`
#include <cstddef> // `std::size_t`, `std::byte`
#include <span>    // `std::span` — per-rank object lists

#include <cuda.h>      // `cuMemsetD8Async`, `cuMemcpyDtoDAsync`
#include <cuda_bf16.h> // `__nv_bfloat16`

#include <numkong/tensor.hpp> // `nk::tensor_view`, `nk::tensor_span`, `nk::tensor`

#include "types.cuh"                  // `expected_t`, `status_t`, `warp_size_k`, `kernel_block_size_default_k`
#include "schedule.cuh"               // `gpu_stream_t`, `launch_kernel`, `cublaslt_session_t`, `cublaslt_plan_t`
#include "multimem_collectives.cuh"   // `multimem_domain_t` — the main-owned level-1 force surface
#include "infiniband_collectives.cuh" // `colocated_domain_t`, `infiniband_member_t` — the level-2 surface
#include "contrastive_relations.cuh"  // `contrastive_relations_view_t`, `contrastive_recipe_t`, the cell resolver

namespace unum::uform::core {

/** @brief Rank-2 BFloat16 device / host view, shaped @b [rows,hidden] — an embedding tower. */
using bf16r2_view_t = nk::tensor_view<__nv_bfloat16, 2>;
/** @brief Rank-2 BFloat16 mutable device / host span, shaped @b [rows,hidden] — a gradient tower. */
using bf16r2_span_t = nk::tensor_span<__nv_bfloat16, 2>;
/** @brief Rank-1 BFloat16 mutable device span — a flattened tower or gradient block. */
using bf16r1_span_t = nk::tensor_span<__nv_bfloat16, 1>;
/** @brief Rank-1 BFloat16 device-only buffer owned through a @ref gpu_arena_t . */
using bf16r1_tensor_arena_t = nk::tensor<__nv_bfloat16, gpu_arena_alloc<__nv_bfloat16>, 1>;

/** @brief Rank-3 Float32 device / host view, shaped @b [prefixes,rows,columns] — the per-prefix block-cell
 *      tensors (raw dots, similarity, logit and raw-dot gradients) shared by the segment matmuls and their
 *      epilogue kernels. */
using f32r3_view_t = nk::tensor_view<f32_t, 3>;
/** @brief Rank-3 Float32 mutable device / host span, the sibling of @ref f32r3_view_t . */
using f32r3_span_t = nk::tensor_span<f32_t, 3>;
/** @brief Rank-2 Float32 device-only buffer owned through a @ref gpu_arena_t . */
using f32r2_tensor_arena_t = nk::tensor<f32_t, gpu_arena_alloc<f32_t>, 2>;
/** @brief Rank-3 Float32 device-only buffer owned through a @ref gpu_arena_t . */
using f32r3_tensor_arena_t = nk::tensor<f32_t, gpu_arena_alloc<f32_t>, 3>;
/** @brief Rank-3 BFloat16 device-only buffer owned through a @ref gpu_arena_t — the cast mirror of a
 *      @ref f32r3_tensor_arena_t used as a tensor-core matmul operand. */
using bf16r3_tensor_arena_t = nk::tensor<__nv_bfloat16, gpu_arena_alloc<__nv_bfloat16>, 3>;

/** @brief Most Matryoshka prefix widths one loss supports; the dims travel to every kernel by value
 *      inside @ref matryoshka_prefixes_t , so the bound keeps the kernel-argument footprint fixed. */
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

/** @brief Floor on a per-vertex @c Σw denominator so a fully-masked row yields a finite zero loss and
 *      zero gradient instead of a divide-by-zero. */
inline constexpr f32_t weight_sum_floor_k = 1.0e-12f;

/**
 *  @brief One GPU's subgraph of the force layout: its contiguous run of the global vertex axis — always
 *      @b [neighbors_per_gpu,hidden] , wherever the transport put it (local tile, NVLink peer memory, or
 *      a landed remote copy). Vertex existence (PAD) is state of the relation view, not of the subgraph.
 */
struct gpu_subgraph_t {
    /** @brief @b [neighbors_per_gpu,hidden] BFloat16 vertex tower of this block, row-major. */
    bf16r2_view_t vertices {};
    /** @brief Global vertex id of this subgraph's row 0 — shifts its local rows onto the global vertex
     *      axis. */
    std::size_t global_offset = 0;
    /** @brief @b [neighbors_per_gpu,hidden] Float32 output — this rank's force contribution to the
     *      subgraph's vertices; Float32 so the caller can reduce-scatter-@b sum without per-rank
     *      rounding. */
    f32r2_span_t vertices_grad {};
};

/**
 *  @brief The validated Matryoshka prefix widths — a fixed-capacity value type every kernel takes by
 *      value. Construction is the validation: @ref make rejects an empty, oversized, non-ascending, or
 *      over-wide ladder, so an instance is always well-formed.
 */
struct matryoshka_prefixes_t {

    /** @brief Validate @p prefix_dims (strictly ascending, each @c >=1 and @c <=hidden , at most
     *      @ref max_matryoshka_prefixes_k entries) and adopt them. */
    [[nodiscard]] static expected<matryoshka_prefixes_t> make(u32r1_view_t prefix_dims, std::size_t hidden) noexcept;

    /** @brief Number of nested prefixes. */
    [[nodiscard]] constexpr u32_t size() const noexcept { return count_; }
    /** @brief Width of prefix @p index — strictly ascending in @p index . */
    [[nodiscard]] constexpr u32_t operator[](u32_t index) const noexcept { return dims_[index]; }
    /** @brief First channel of segment @p index — the previous prefix's width, 0 for the first. */
    [[nodiscard]] constexpr u32_t segment_begin(u32_t index) const noexcept {
        return index != 0 ? dims_[index - 1] : 0u;
    }
    /** @brief Channel count of segment @p index . */
    [[nodiscard]] constexpr u32_t segment_width(u32_t index) const noexcept {
        return dims_[index] - segment_begin(index);
    }

  private:
    u32_t dims_[max_matryoshka_prefixes_k] {};
    u32_t count_ = 0;
};

#if defined(__CUDACC__)

/** @brief The per-vertex logit scale β: the device slot selected by the vertex's pair type, or the host
 *      fallback when no device view is bound. */
[[nodiscard]] __device__ inline f32_t contrastive_vertex_beta_(contrastive_recipe_t const &recipe,
                                                               u32_t vertex_index) noexcept {
    if (recipe.inverse_temperature_device.empty()) return recipe.inverse_temperature;
    u32_t const slot = recipe.vertex_pair_type.empty() ? 0u : recipe.vertex_pair_type[vertex_index];
    return recipe.inverse_temperature_device[slot];
}

/** @brief The per-vertex additive logit bias — the sibling of @ref contrastive_vertex_beta_ . */
[[nodiscard]] __device__ inline f32_t contrastive_vertex_bias_(contrastive_recipe_t const &recipe,
                                                               u32_t vertex_index) noexcept {
    if (recipe.bias_device.empty()) return recipe.bias;
    u32_t const slot = recipe.vertex_pair_type.empty() ? 0u : recipe.vertex_pair_type[vertex_index];
    return recipe.bias_device[slot];
}

/**
 *  @brief The baked geometry of one segments loss — every capacity a call may not exceed; a different
 *      config is a different object, exactly the capture contract of @ref multimodal_batch_shape_t .
 */
struct contrastive_segments_config_t {
    /** @brief Owned (pooled) vertex rows this rank scores — the force grid's row count. */
    u32_t vertices_per_gpu = 0;
    /** @brief Embedding width shared by both towers. */
    u32_t hidden = 0;
    /** @brief Vertex rows in every GPU's subgraph — world-uniform, the homogeneous-mesh guarantee. */
    u32_t neighbors_per_gpu = 0;
    /** @brief Subgraphs scored per step — the world's GPU count when fully gathered. */
    u32_t gpu_subgraphs_per_step = 0;
};

/**
 *  @brief The segments contrastive loss as a function object: shapes, Matryoshka prefixes, tensor-core
 *      matmul plans, and every scratch tensor are baked at @ref make — a call is a pure enqueue,
 *      capture-safe and bit-reproducible (see the file header).
 */
struct contrastive_segments_t {

    /**
     *  @brief Bake everything static: validate the prefix ladder, allocate every scratch tensor off
     *      @p arena , and pin one algorithm per segment matmul family against the caller's tensor-core
     *      session — so a constructed loss never plans and never allocates.
     *
     *  @param[in] blas The device's one cuBLASLt session, main-owned and borrowed for this object's
     *      lifetime — one handle per device, never per consumer.
     *  @param[in] config The baked geometry (see @ref contrastive_segments_config_t ).
     *  @param[in] prefix_dims Matryoshka prefix widths, validated through @ref matryoshka_prefixes_t .
     */
    [[nodiscard]] static expected<contrastive_segments_t> make(gpu_arena_t &arena, cublaslt_session_t const &blas,
                                                               contrastive_segments_config_t config,
                                                               u32r1_view_t prefix_dims) noexcept;

    /**
     *  @brief Rank-local forward + backward for one step — enqueue-only, capture-safe. Inputs and outputs
     *      are the only per-call state; every intermediate lives inside the object.
     *
     *  @param[in] vertices This rank's owned @b [rows,hidden] BFloat16 vertex tower.
     *  @param[in] gpu_subgraphs @b [gpu_subgraphs_per_step] descriptors, each @b [neighbors_per_gpu,hidden] ,
     *      tiling the global vertex axis in order.
     *  @param[in] relations The staged relation graph view; @c vertices_per_gpu() must equal the baked rows.
     *  @param[in] recipe β / bias / ambient-repulsion weight, host scalars or per-pair-type device slots.
     *  @param[in] vertices_per_mesh Global vertex-row count across the world — the mean divisor.
     *  @param[out] loss_out @b [1] Float32 rank-local per-vertex-mean potential summand. Overwritten.
     *  @param[out] vertices_grad @b [rows,hidden] BFloat16 force on the owned vertices. Overwritten.
     *  @param[in] stream The stream every kernel and matmul is enqueued on.
     *  @param[out] log_inverse_temperature_grad Optional @b [pair_type_count] summands; empty disables.
     *  @param[out] bias_grad Optional @b [pair_type_count] summands; empty disables.
     */
    [[nodiscard]] expected_t operator()(bf16r2_view_t vertices, nk::tensor_view<gpu_subgraph_t, 1> gpu_subgraphs,
                                        contrastive_relations_view_t relations, contrastive_recipe_t recipe,
                                        std::size_t vertices_per_mesh, f32r1_span_t loss_out,
                                        bf16r2_span_t vertices_grad, gpu_stream_t const &stream,
                                        f32r1_span_t log_inverse_temperature_grad = {},
                                        f32r1_span_t bias_grad = {}) noexcept;

  private:
    /** @brief The baked geometry and the validated prefix ladder — every capacity a call is checked
     *      against. */
    matryoshka_prefixes_t prefixes_ {};
    contrastive_segments_config_t config_ {};

    /** @brief @b [prefixes,rows] / @b [prefixes,neighbors] per-row L2 norms of the owned and subgraph
     *      towers. */
    f32r2_tensor_arena_t vertex_norms_ {}, neighbor_norms_ {};
    /** @brief @b [rows,neighbors] running tensor-core dot accumulator — one segment's raw product,
     *      reused every segment and copied out into @ref raw_dot_ 's plane. */
    f32r2_tensor_arena_t running_dots_ {};
    /** @brief @b [prefixes,rows,neighbors] per-prefix raw dot products, cosine similarity, logit
     *      gradient, and raw-dot gradient of one subgraph block. */
    f32r3_tensor_arena_t raw_dot_ {}, similarity_ {}, logit_grad_ {}, raw_grad_ {};
    /** @brief @b [prefixes,rows,neighbors] BFloat16 cast of @ref raw_grad_ — the tensor-core gradient
     *      matmul operand. */
    bf16r3_tensor_arena_t raw_grad_bf16_ {};
    /** @brief The six @b [rows] accumulator rows: Σw⁺, Σw⁻, positive and negative softplus mass, bias
     *      and log-β gradient summands — one @b [6,rows] tensor, one memset covers them all. */
    f32r2_tensor_arena_t vertex_accumulator_rows_ {};
    /** @brief @b [prefixes,rows] / @b [prefixes,neighbors] deterministic cosine self-coefficients. */
    f32r2_tensor_arena_t vertex_self_coefficients_ {}, neighbor_self_coefficients_ {};
    /** @brief @b [rows,hidden] Float32 owned-vertex force accumulator, cast to BFloat16 at the end. */
    f32r2_tensor_arena_t vertices_grad_f32_ {};
    b8r1_tensor_arena_t cublaslt_workspace_ {};

    /** @brief The device's one main-owned tensor-core session — borrowed, never created here. */
    cublaslt_session_t const *session_ = nullptr;
    /** @brief Three pinned plans per segment — dots, owned-vertex force, subgraph force — at slots
     *      @c [3s,3s+2] ; capacity is the prefix bound, the live count is @c 3·prefixes_.size() . */
    std::array<cublaslt_plan_t, 3 * max_matryoshka_prefixes_k> plans_ {};
};

/**
 *  @brief The distributed sibling of @ref contrastive_segments_t : one object owning a whole world's
 *      force evaluation — the per-rank loss objects, a reserved window of the main-owned
 *      @ref multimem_domain_t whose replicas @b are the column-force surfaces, and the cross-domain
 *      members wired through the main-owned @ref colocated_domain_t .
 *
 *  This signature expresses the self-scoring configuration: @c towers serves as both the owned rows and
 *  the subgraph columns of one global vertex axis, so the row-side and column-side forces of every
 *  vertex always fold — a bipartite mesh is a different signature, not a flag. One call runs: the
 *  world-tower gather (multi-domain worlds), then per rank the loss (row force straight into the
 *  caller's gradient, column forces into the domain window), the epoch barrier, the switch
 *  reduce-scatter, the cross-domain force all-reduce, and the column fold — enqueue-only,
 *  deterministic, capture-friendly.
 */
struct distributed_contrastive_segments_t {

    /** @brief How many Float32 payload scalars this consumer will @c reserve from the shared
     *      @ref multimem_domain_t — for main's domain sizing. */
    [[nodiscard]] static constexpr std::size_t payload_scalars_needed(contrastive_segments_config_t config) noexcept {
        return static_cast<std::size_t>(config.gpu_subgraphs_per_step) * config.vertices_per_gpu * config.hidden;
    }

    /**
     *  @brief One construction absorbs the whole world: validates every arena's visibility, builds
     *      every rank's loss object, reserves this consumer's window of the shared @p domain , and wires
     *      its cross-domain members through @p colocated — the caller (main) owns the infrastructure,
     *      this object only borrows it.
     *
     *  @param[in,out] domain The main-owned level-1 surface; one force window is reserved from it.
     *  @param[in,out] colocated The main-owned level-2 surface; reduce members are wired per rank and
     *      one gather member on the gateway — zero-round identities in single-domain worlds.
     *  @param[in] arenas @b [gpus] per-rank arenas; loss scratch and member heaps allocate from each
     *      rank's own.
     *  @param[in] streams @b [gpus] per-rank streams; their contexts scope every per-device setup.
     *  @param[in] sessions @b [gpus] per-device tensor-core sessions — main-owned, borrowed by every
     *      rank's loss object.
     *  @param[in] towers @b [gpus] per-rank pooled towers — capture-stable device views, baked here so
     *      the per-step slices carry no host arguments.
     *  @param[in] relations @b [gpus] per-rank relation views — addresses baked, contents restaged by
     *      the caller per step.
     *  @param[in] vertices_grads @b [gpus] per-rank row-force outputs — baked like the towers.
     *  @param[in] config The per-rank geometry; @c gpu_subgraphs_per_step must equal the @b world GPU
     *      count and both row counts must agree (the self-scoring shape).
     *  @param[in] prefix_dims Matryoshka prefix widths, shared by every rank.
     */
    [[nodiscard]] static expected<distributed_contrastive_segments_t> make(
        multimem_domain_t &domain, colocated_domain_t &colocated, std::span<gpu_arena_t> arenas,
        std::span<gpu_stream_t const> streams, std::span<cublaslt_session_t const> sessions,
        std::span<bf16r2_view_t const> towers, std::span<contrastive_relations_view_t const> relations,
        std::span<bf16r2_span_t const> vertices_grads, contrastive_segments_config_t config, u32r1_view_t prefix_dims,
        f64_t timeout_seconds = 120.0) noexcept;

    /** @brief The @b [gpus] rank-local loss summands of the last call — sum for the global mean. */
    [[nodiscard]] f32r1_view_t loss_outs() const noexcept { return loss_outs_.view(); }
    /** @brief How many multimem_domains this object spans — 1 means the members are identities. */
    [[nodiscard]] u32_t world_domains() const noexcept { return world_domains_; }

    /**
     *  @brief Rank @p rank 's slice of one distributed evaluation — pure enqueue onto @p stream , the
     *      shape a fork/join capture body composes directly. Rank 0's slice carries the shared phase:
     *      the domain's epoch advance and (across domains) the world-tower gather every other rank's
     *      stream waits on. All views were baked at @ref make ; the only per-call argument is the
     *      recipe. Ranks must be enqueued in order @c 0..gpus-1 within a step.
     */
    [[nodiscard]] expected_t operator()(std::size_t rank, contrastive_recipe_t recipe,
                                        gpu_stream_t const &stream) noexcept;

    /** @brief The eager convenience: every rank's slice in order over the per-rank streams. */
    [[nodiscard]] expected_t operator()(contrastive_recipe_t recipe, std::span<gpu_stream_t const> streams) noexcept;

    /** @brief The replay-time bumps of every counter this consumer advances inside a captured step —
     *      once before every @c cuGraphLaunch , mirroring one evaluation. */
    void advance_host() noexcept {
        domain_->advance_host();
        if (world_domains_ > 1u) {
            gather_member_.advance_host();
            for (std::size_t rank = 0; rank != reduce_members_.size(); ++rank) reduce_members_[rank].advance_host();
        }
    }

  private:
    /** @brief Stage this domain's towers into the gather member's own block, all-gather the remote
     *      domains' blocks, and record the gather event every rank waits on. Gateway stream only,
     *      caller already inside its context. */
    [[nodiscard]] expected_t gather_world_tower_(gpu_stream_t const &stream) noexcept;

    /** @brief Level-1 reduce of @p rank 's world columns — one switch-reduced window per domain into
     *      the rank's reduce member — then the level-2 recursive-doubling sum; @p column_force receives
     *      the rank's own reduced window. Caller holds the rank's context. */
    [[nodiscard]] expected_t reduce_world_columns_(std::size_t rank, gpu_stream_t const &stream,
                                                   f32r1_view_t &column_force) noexcept;

    /** @brief The baked world geometry every capacity check keys on. */
    contrastive_segments_config_t config_ {};
    /** @brief One loss object per rank — per-device by nature. */
    array<contrastive_segments_t> losses_ {};
    /** @brief The main-owned level-1 surface — borrowed; its replicas carry the column forces. */
    multimem_domain_t *domain_ = nullptr;
    /** @brief This consumer's disjoint window of the domain payload, reserved at @ref make . */
    multimem_window_t window_ {};
    /** @brief @b [gpus] rank-local loss summands on the lead device, peer-written by each rank. */
    f32r1_tensor_arena_t loss_outs_ {};
    /** @brief How many multimem_domains the world spans; 1 makes the members below identities. */
    u32_t world_domains_ = 1;
    /** @brief This rank's multimem_domain index — the recursive-doubling coordinate. */
    u32_t multimem_domain_index_ = 0;
    /** @brief The gateway's gather-complete ordering token every rank's stream waits on. */
    gpu_event_t gather_ready_ {};
    /** @brief One cross-domain reduce member per rank — its payload is the rank's world-column shard. */
    array<infiniband_member_t> reduce_members_ {};
    /** @brief The gateway's cross-domain gather member — its payload is the world tower. */
    infiniband_member_t gather_member_ {};
    /** @brief All ranks' subgraph tables in one rank-major table, baked at @ref make . */
    array<gpu_subgraph_t> subgraph_tables_ {};
    /** @brief @b [gpus] tower views the gateway stages into the gather member's own block — populated
     *      only across domains; the single-domain path reads the same views through the tables. */
    array<bf16r2_view_t> gather_sources_ {};
    /** @brief @b [gpus] per-rank relation views — addresses baked, contents restaged per step. */
    array<contrastive_relations_view_t> relations_ {};
    /** @brief @b [gpus] per-rank row-force outputs, baked at @ref make . */
    array<bf16r2_span_t> vertices_grads_ {};
};

#endif // defined(__CUDACC__)

} // namespace unum::uform::core
