/**
 *  @brief The contrastive relation graph: per-pair attract / repel / ignore semantics over the global
 *      vertex axis, as compressed sparse rows — the edge sets of the force layout the losses relax.
 *  @file cuda/primitives/contrastive_relations.cuh
 *  @author Ash Vardanian
 *  @date July 2, 2026
 *
 *  Both contrastive losses (segments and tokens) score a @b [vertices_per_gpu,neighbors_per_mesh] cell grid
 *  and need one answer per cell: the target sign and the weight. This file is that answer's single source
 *  of truth — a device-readable view, the per-cell resolver, and the host owner that stages it
 *  capture-safely.
 *
 *  @section relations_semantics Per-cell contract
 *
 *  For local vertex @c i and @b global vertex @c j , resolved by @ref contrastive_pair_relation in this
 *  exact precedence order (first match wins):
 *
 *      1. Either side is PAD (effective length 0) — weight 0, the cell does not exist.
 *      2. @c j is in @c ignore_ids[i] — weight 0: a known false negative or a self-pairing to skip.
 *      3. @c j is in @c attract_ids[i] — target +1, weight from @c attract_scores (default 1).
 *      4. @c j is in @c repel_ids[i] — target −1, weight from @c repel_scores (default 1).
 *      5. Every other cell — target −1, weight @c ambient_repulsion_weight .
 *
 *  The explicit-vs-ambient balance is @b weights-only — there is no policy flag, and
 *  @c ambient_repulsion_weight==0 removes the background repulsion through the same code path. When no
 *  explicit attract edges exist the graph degenerates to the @b self-edge default: vertex @c i attracts
 *  its own global copy at @c self_edge_base+i with unit weight, and every other pair repels with
 *  @c ambient_repulsion_weight — the classic in-batch contract, bit-for-bit.
 *
 *  Padding is @b state of the graph, not a parameter threaded through kernels: the view carries optional
 *  effective-length columns for the local vertices and the global axis, and the resolver reads them
 *  itself. Empty columns mean everything is live.
 *
 *  @section relations_staging Staging and capture
 *
 *  Ids index the @b global (post-all-gather) vertex axis and are staged global — no rank-local rebasing
 *  scratch downstream. Within each vertex's run the ids are @b sorted ascending: the in-kernel
 *  membership test is a per-row binary search, so an unsorted run silently mis-scores — which is exactly
 *  why @ref contrastive_relations_t::upload validates every invariant host-side, where the data still is,
 *  and fails @b loudly (capacity overflow included: an overflowing graph is an error, never a silent
 *  fallback to the self-edge default).
 *
 *  The device columns are NumKong rank-1 tensors owning memory through @ref gpu_arena_alloc — allocated
 *  @b once at fixed capacity, so device addresses never change; only per-step @b contents are
 *  re-uploaded, and a captured graph replays against fresh edges. One loud exception: the view a kernel
 *  took @b by value at capture bakes in the explicit-vs-self-edge mode — flipping between them requires a
 *  @b recapture; contents-only changes never do.
 */
#pragma once

#include <cstdint> // `std::uint32_t`

#include <cuda.h> // `cuMemcpyHtoDAsync`, `cuMemsetD8Async`
#if defined(__CUDACC__)
#include <cub/thread/thread_search.cuh> // `cub::LowerBound` — the per-run membership search
#endif

#include <numkong/tensor.hpp> // `nk::tensor`, `nk::tensor_view` — owning columns and borrowed views

#include "cuda_core/types.cuh"          // `expected`, `expected_t`, `status_t`
#include "cuda_core/gpu_allocators.cuh" // `gpu_arena_t`, `gpu_arena_alloc` backing the owning tensors
#include "cuda_core/schedule.cuh"       // `gpu_stream_t` ordering the uploads

namespace unum::uform::core {

namespace nk = ashvardanian::numkong;

/**
 *  @brief Scaling and labeling knobs of a contrastive loss — host fallbacks plus optional device views of
 *      the learnable scalars, so a captured step reads per-step-updated values without a host readback.
 *
 *  When @c inverse_temperature_device / @c bias_device are non-empty they override the host scalars;
 *  @c vertex_pair_type (when non-empty, @b [vertices_per_gpu] ) selects each vertex's slot — e.g. separate
 *  temperatures for text-text vs text-image pairs. An empty selector means every vertex uses slot 0,
 *  bit-identical to the single-slot path. Views instead of raw pointers: every device column carries its
 *  own extent, so the slot count is never a separately-carried integer that can disagree.
 */
struct contrastive_recipe_t {
    /** @brief Logit scale β applied to every cosine; overridden by @ref inverse_temperature_device . */
    f32_t inverse_temperature = 1.0f;
    /** @brief Additive logit bias; overridden by @ref bias_device . */
    f32_t bias = 0.0f;
    /** @brief Weight of every pair the graph names no edge for — the ambient all-pairs repulsion. */
    f32_t ambient_repulsion_weight = 1.0f;
    /** @brief Optional @b [pair_type_count] Float32 device slots of the learnable β; empty falls back to
     *      the host scalar. */
    f32r1_view_t inverse_temperature_device {};
    /** @brief Optional @b [pair_type_count] Float32 device slots of the learnable bias; empty falls back
     *      to the host scalar. */
    f32r1_view_t bias_device {};
    /** @brief Optional @b [vertices_per_gpu] UInt32 per-vertex slot selector; empty means slot 0 for all. */
    u32r1_view_t vertex_pair_type {};

    /** @brief Number of (β, bias) slots — carried by the device views; 1 on the host-scalar fallback. */
    [[nodiscard]] constexpr u32_t pair_type_count() const noexcept {
        return inverse_temperature_device.empty() ? 1u : static_cast<u32_t>(inverse_temperature_device.extent(0));
    }
};

/** @brief One resolved cell: the target sign (+1 attract, −1 repel) and the weight; @c weight==0 means
 *      the cell is skipped and the target carries no meaning. */
struct contrastive_pair_relation_t {
    f32_t target;
    f32_t weight;
};

/**
 *  @brief The device-readable relation graph — NumKong views borrowing the staged columns, passed to
 *      kernels @b by value.
 *
 *  Three CSR edge sets over the local vertex rows: attract and repel with per-edge scores, ignore
 *  without. Every
 *  extent is carried by its view: @ref vertices_per_gpu derives from the offsets row, and empty
 *  @ref ignore_offsets means no ignores. All ids are @b global vertex ids, sorted ascending within
 *  each row's run.
 */
struct contrastive_relations_view_t {
    /** @brief Attract @b [vertices_per_gpu+1] row pointers — always present; all-zero on the self-edge
     *      default path. */
    u32r1_view_t attract_offsets {};
    /** @brief Attract @b [attract_edge_count] global ids, sorted ascending per run; empty selects the
     *      self-edge default parity path. */
    u32r1_view_t attract_ids {};
    /** @brief Attract @b [attract_edge_count] per-edge weights. */
    f32r1_view_t attract_scores {};
    /** @brief Repel @b [vertices_per_gpu+1] row pointers. */
    u32r1_view_t repel_offsets {};
    /** @brief Repel global ids, sorted ascending per run. */
    u32r1_view_t repel_ids {};
    /** @brief Repel per-edge weights. */
    f32r1_view_t repel_scores {};
    /** @brief Ignore @b [vertices_per_gpu+1] row pointers; empty when there are no ignores. */
    u32r1_view_t ignore_offsets {};
    /** @brief Ignore global ids, sorted ascending per run. */
    u32r1_view_t ignore_ids {};
    /** @brief Optional @b [vertices_per_gpu] UInt32 device column of local-vertex effective lengths; 0
     *      marks a PAD row no pair resolves against. Empty means every vertex is live. */
    u32r1_view_t vertex_effective_lengths {};
    /** @brief Optional @b [neighbors_per_mesh] UInt32 device column of effective lengths over the whole
     *      global vertex axis, indexed by @b global id; 0 marks a PAD slot. Empty means all live. */
    u32r1_view_t neighbor_effective_lengths {};

    /** @brief Global id of local vertex 0's own copy — the base of the implicit self-edges. */
    std::size_t self_edge_base = 0;
    /** @brief Weight of every unnamed cell; 0 removes in-batch negatives. */
    f32_t ambient_repulsion_weight = 1.0f;

    /** @brief Number of local vertex rows the graph describes — one less than the offsets row. Like the views'
     *      own accessors, @c constexpr and therefore device-callable. */
    [[nodiscard]] constexpr u32_t vertices_per_gpu() const noexcept {
        return attract_offsets.empty() ? 0u : static_cast<u32_t>(attract_offsets.extent(0) - 1u);
    }
    /** @brief Whether explicit attract edges drive the labels (vs the self-edge default). */
    [[nodiscard]] constexpr bool has_explicit_edges() const noexcept { return attract_ids.extent(0) != 0u; }
};

#if defined(__CUDACC__)

/** @brief Membership search inside one sorted CSR run — CUB's thread-level lower bound plus the equality
 *      check; the slot index of @p id , or @p end when absent. */
[[nodiscard]] __device__ inline u32_t contrastive_run_find_(u32r1_view_t ids, u32_t begin, u32_t end,
                                                            u32_t id) noexcept {
    // vendor-boundary: cub needs a raw pointer + count, not a view, into the CSR `ids` run.
    u32_t const slot = begin + cub::LowerBound(ids.data() + begin, end - begin, id);
    return slot < end && ids[slot] == id ? slot : end;
}

/** @brief Resolve one cell of the @b [vertices_per_gpu,neighbors_per_mesh] grid to its target and weight,
 *      in the file-header precedence: PAD > ignore > attract > repel > implicit negative. Padding is read
 *      from the view's own effective-length columns — the label grid carries its whole state.
 *
 *  @param[in] relations The graph views, taken by value at kernel launch.
 *  @param[in] vertex_index Row in @c [0,vertices_per_gpu) .
 *  @param[in] neighbor_global Slot on the @b global vertex axis. */
[[nodiscard]] __device__ inline contrastive_pair_relation_t contrastive_pair_relation(
    contrastive_relations_view_t const &relations, u32_t vertex_index, u32_t neighbor_global) noexcept {

    if (!relations.vertex_effective_lengths.empty() && relations.vertex_effective_lengths[vertex_index] == 0u)
        return {0.0f, 0.0f};
    if (!relations.neighbor_effective_lengths.empty() && relations.neighbor_effective_lengths[neighbor_global] == 0u)
        return {0.0f, 0.0f};
    if (!relations.ignore_offsets.empty()) {
        u32_t const begin = relations.ignore_offsets[vertex_index];
        u32_t const end = relations.ignore_offsets[vertex_index + 1];
        if (contrastive_run_find_(relations.ignore_ids, begin, end, neighbor_global) != end) return {0.0f, 0.0f};
    }
    if (!relations.has_explicit_edges()) {
        bool const is_self_edge = static_cast<std::size_t>(neighbor_global) ==
                                  relations.self_edge_base + static_cast<std::size_t>(vertex_index);
        return is_self_edge ? contrastive_pair_relation_t {1.0f, 1.0f}
                            : contrastive_pair_relation_t {-1.0f, relations.ambient_repulsion_weight};
    }
    // Attract lookup: an explicit positive edge carrying its own weight.
    {
        u32_t const begin = relations.attract_offsets[vertex_index];
        u32_t const end = relations.attract_offsets[vertex_index + 1];
        u32_t const slot = contrastive_run_find_(relations.attract_ids, begin, end, neighbor_global);
        if (slot != end) return {1.0f, relations.attract_scores[slot]};
    }
    // Repel lookup: an explicit hard negative carrying its own weight.
    {
        u32_t const begin = relations.repel_offsets[vertex_index];
        u32_t const end = relations.repel_offsets[vertex_index + 1];
        u32_t const slot = contrastive_run_find_(relations.repel_ids, begin, end, neighbor_global);
        if (slot != end) return {-1.0f, relations.repel_scores[slot]};
    }
    return {-1.0f, relations.ambient_repulsion_weight};
}

#endif // defined(__CUDACC__)

/**
 *  @brief The host owner of the relation graph — one memory-owning NumKong tensor per column, allocated
 *      through @ref gpu_arena_alloc at fixed capacity, re-uploaded per step, and viewed by kernels
 *      through @ref contrastive_relations_view_t windows over the live extents.
 *
 *  Eight columns: three offset rows @b [max_vertices+1] , three id runs @b [max_edges] , two score runs
 *  @b [max_edges] . Addresses never change after @ref make — the capture-stability contract — so a
 *  replayed graph reads whatever @ref upload staged last. Uploads run @b outside any capture.
 *
 *  Ownership is deliberately split: the CSR columns are @b owned (they are staged copies of host data
 *  with no other home), while the effective-length columns are @b borrowed device views — the batch
 *  pipeline owns those lengths and re-fills them per step at capture-stable addresses; copying them here
 *  would duplicate per-step state. Both arrive through @ref upload / @ref upload_self_edges , the single
 *  staging entry points.
 */
struct contrastive_relations_t {

    /** @brief Allocate the eight columns off @p arena at their worst-case capacity; every upload
     *      @c try_resize s them to the step's live extents without moving an address — the columns' own
     *      shapes are the single source of the live counts.
     *  @param[in] max_vertices Most local vertex rows any step will carry.
     *  @param[in] max_edges Most explicit edges any step will carry, per edge set. */
    [[nodiscard]] static expected<contrastive_relations_t> make(gpu_arena_t &arena, u32_t max_vertices,
                                                                u32_t max_edges) noexcept {
        if (max_vertices == 0)
            return expected<contrastive_relations_t>::fail(status_t::invalid_input_k,
                                                           "relations need at least one vertex row");
        contrastive_relations_t relations;
        gpu_arena_alloc<u32_t> index_alloc {arena};
        gpu_arena_alloc<f32_t> score_alloc {arena};
        std::size_t const offsets_count = max_vertices + 1u;
        relations.attract_offsets_ = u32r1_tensor_arena_t::try_empty({offsets_count}, index_alloc);
        relations.repel_offsets_ = u32r1_tensor_arena_t::try_empty({offsets_count}, index_alloc);
        relations.ignore_offsets_ = u32r1_tensor_arena_t::try_empty({offsets_count}, index_alloc);
        bool allocated = relations.attract_offsets_ && relations.repel_offsets_ && relations.ignore_offsets_;
        if (max_edges != 0) {
            relations.attract_ids_ = u32r1_tensor_arena_t::try_empty({max_edges}, index_alloc);
            relations.repel_ids_ = u32r1_tensor_arena_t::try_empty({max_edges}, index_alloc);
            relations.ignore_ids_ = u32r1_tensor_arena_t::try_empty({max_edges}, index_alloc);
            relations.attract_scores_ = f32r1_tensor_arena_t::try_empty({max_edges}, score_alloc);
            relations.repel_scores_ = f32r1_tensor_arena_t::try_empty({max_edges}, score_alloc);
            allocated = allocated && relations.attract_ids_ && relations.repel_ids_ && relations.ignore_ids_ &&
                        relations.attract_scores_ && relations.repel_scores_;
        }
        if (!allocated)
            return expected<contrastive_relations_t>::fail(
                status_t::bad_alloc_k,
                "relations column allocation failed (max_vertices={} max_edges={} arena_used={})", max_vertices,
                max_edges, arena.used());
        return expected<contrastive_relations_t> {std::move(relations)};
    }

    /**
     *  @brief Validate a host CSR and stage it into the device columns on @p stream — @b outside capture.
     *
     *  Every invariant is checked here, where the data is still host-resident, and every violation is a
     *  named failure: capacity overflow, offset rows of the wrong length, non-monotone or non-zero-based
     *  offsets, id/score length mismatches, unsorted or duplicate ids within a run, ids outside
     *  @c [0,neighbors_per_mesh) , negative scores. There is no silent fallback.
     *
     *  @param[in] ignore_offsets / @p ignore_ids Empty views mean no ignores.
     *  @param[in] vertex_effective_lengths / @p neighbor_effective_lengths The per-step PAD columns the
     *      resolver reads: @b [vertices_per_gpu] and @b [neighbors_per_mesh] (by global id) device views, 0
     *      marking PAD, empty meaning all-live. @b Borrowed, never copied — the batch pipeline owns them
     *      and keeps the addresses capture-stable.
     */
    [[nodiscard]] expected_t upload(                                                                //
        u32r1_view_t attract_offsets, u32r1_view_t attract_ids,                                     //
        f32r1_view_t attract_scores,                                                                //
        u32r1_view_t repel_offsets, u32r1_view_t repel_ids,                                         //
        f32r1_view_t repel_scores,                                                                  //
        u32r1_view_t ignore_offsets, u32r1_view_t ignore_ids,                                       //
        u32r1_view_t vertex_effective_lengths, u32r1_view_t neighbor_effective_lengths,             //
        std::size_t neighbors_per_mesh, std::size_t self_edge_base, f32_t ambient_repulsion_weight, //
        gpu_stream_t &stream) noexcept {

        if (attract_offsets.extent(0) < 2 || attract_offsets.extent(0) != repel_offsets.extent(0))
            return expected_t::fail( //
                status_t::invalid_input_k,
                "attract and repel offset rows must both span vertices_per_gpu+1 (got {} and {})",
                attract_offsets.extent(0), repel_offsets.extent(0));
        u32_t const vertices_per_gpu = static_cast<u32_t>(attract_offsets.extent(0) - 1u);
        bool const has_ignores = !ignore_offsets.empty();
        if (has_ignores && ignore_offsets.extent(0) != attract_offsets.extent(0))
            return expected_t::fail( //
                status_t::invalid_input_k,
                "ignore offsets must span vertices_per_gpu+1 like the others (got {}, expected {})",
                ignore_offsets.extent(0), attract_offsets.extent(0));

        if (expected_t checked = validate_edges_("attract", attract_offsets, attract_ids, attract_scores,
                                                 neighbors_per_mesh);
            !checked)
            return checked;
        if (expected_t checked = validate_edges_("repel", repel_offsets, repel_ids, repel_scores, neighbors_per_mesh);
            !checked)
            return checked;
        if (has_ignores)
            if (expected_t checked = validate_edges_("ignore", ignore_offsets, ignore_ids, {}, neighbors_per_mesh);
                !checked)
                return checked;

        // Resize every column to the step's live extents — no address moves, and an overflow beyond the
        // reserved capacity fails here, loudly, before anything is staged.
        bool resized = attract_offsets_.try_resize({vertices_per_gpu + 1u});
        resized = resized && repel_offsets_.try_resize({vertices_per_gpu + 1u});
        resized = resized && attract_ids_.try_resize({attract_ids.extent(0)});
        resized = resized && attract_scores_.try_resize({attract_scores.extent(0)});
        resized = resized && repel_ids_.try_resize({repel_ids.extent(0)});
        resized = resized && repel_scores_.try_resize({repel_scores.extent(0)});
        resized = resized && ignore_offsets_.try_resize({has_ignores ? vertices_per_gpu + 1u : 0u});
        resized = resized && ignore_ids_.try_resize({has_ignores ? ignore_ids.extent(0) : 0u});
        if (!resized)
            return expected_t::fail( //
                status_t::shape_unsupported_k,
                "relations exceed the reserved capacities (vertices={}/{} attract={}/{} repel={}/{} ignore={}/{})",
                vertices_per_gpu + 1u, attract_offsets_.capacity(), attract_ids.extent(0), attract_ids_.capacity(),
                repel_ids.extent(0), repel_ids_.capacity(), has_ignores ? ignore_ids.extent(0) : 0u,
                ignore_ids_.capacity());

        // Stage every validated column; the typed helper carries extents and element sizes itself.
        expected_t copied = copy_host_to_device(attract_offsets, attract_offsets_.span(), stream);
        if (copied) copied = copy_host_to_device(attract_ids, attract_ids_.span(), stream);
        if (copied) copied = copy_host_to_device(attract_scores, attract_scores_.span(), stream);
        if (copied) copied = copy_host_to_device(repel_offsets, repel_offsets_.span(), stream);
        if (copied) copied = copy_host_to_device(repel_ids, repel_ids_.span(), stream);
        if (copied) copied = copy_host_to_device(repel_scores, repel_scores_.span(), stream);
        if (copied && has_ignores) copied = copy_host_to_device(ignore_offsets, ignore_offsets_.span(), stream);
        if (copied && has_ignores) copied = copy_host_to_device(ignore_ids, ignore_ids_.span(), stream);
        if (!copied) return copied;

        vertex_effective_lengths_ = vertex_effective_lengths;
        neighbor_effective_lengths_ = neighbor_effective_lengths;
        self_edge_base_ = self_edge_base;
        ambient_repulsion_weight_ = ambient_repulsion_weight;
        return {};
    }

    /** @brief Stage the zero-edge self-edge default: the offsets row is zeroed (every run empty), both
     *      edge sets collapse to empty extents, and the borrowed effective-length columns are adopted as
     *      in @ref upload . */
    [[nodiscard]] expected_t upload_self_edges(u32_t vertices_per_gpu, u32r1_view_t vertex_effective_lengths,
                                               u32r1_view_t neighbor_effective_lengths, std::size_t self_edge_base,
                                               f32_t ambient_repulsion_weight, gpu_stream_t &stream) noexcept {
        bool resized = vertices_per_gpu != 0 && attract_offsets_.try_resize({vertices_per_gpu + 1u});
        resized = resized && repel_offsets_.try_resize({vertices_per_gpu + 1u});
        resized = resized && attract_ids_.try_resize({0}) && attract_scores_.try_resize({0});
        resized = resized && repel_ids_.try_resize({0}) && repel_scores_.try_resize({0});
        resized = resized && ignore_offsets_.try_resize({0}) && ignore_ids_.try_resize({0});
        if (!resized)
            return expected_t::fail(status_t::shape_unsupported_k,
                                    "self-edge relations for {} vertices exceed the column capacity of {}",
                                    vertices_per_gpu, attract_offsets_.capacity());
        if (CUresult result = cuMemsetD8Async(reinterpret_cast<CUdeviceptr>(attract_offsets_.data()), 0,
                                              (vertices_per_gpu + 1u) * sizeof(u32_t), stream.get());
            result != CUDA_SUCCESS)
            return expected_t::fail(result, "self-edge offsets reset failed (vertices_per_gpu={})", vertices_per_gpu);
        vertex_effective_lengths_ = vertex_effective_lengths;
        neighbor_effective_lengths_ = neighbor_effective_lengths;
        self_edge_base_ = self_edge_base;
        ambient_repulsion_weight_ = ambient_repulsion_weight;
        return {};
    }

    /** @brief Views over the owned columns' live extents — the shapes the last successful upload resized
     *      them to; kernels take this by value. An ignore set staged with zero edges is indistinguishable
     *      from no ignores, so its views collapse to empty. */
    [[nodiscard]] contrastive_relations_view_t view() const noexcept {
        contrastive_relations_view_t view;
        view.attract_offsets = attract_offsets_.view();
        view.attract_ids = attract_ids_.view();
        view.attract_scores = attract_scores_.view();
        view.repel_offsets = repel_offsets_.view();
        view.repel_ids = repel_ids_.view();
        view.repel_scores = repel_scores_.view();
        view.ignore_offsets = ignore_offsets_.view();
        view.ignore_ids = ignore_ids_.view();
        view.vertex_effective_lengths = vertex_effective_lengths_;
        view.neighbor_effective_lengths = neighbor_effective_lengths_;
        view.self_edge_base = self_edge_base_;
        view.ambient_repulsion_weight = ambient_repulsion_weight_;
        return view;
    }

  private:
    /** @brief All host-side invariants of one edge set (attract / repel / ignore); every violation names
     *      the @p edge_kind and the row. */
    [[nodiscard]] expected_t validate_edges_(char const *edge_kind, u32r1_view_t offsets, u32r1_view_t ids,
                                             f32r1_view_t scores, std::size_t neighbors_per_mesh) const noexcept {
        std::size_t const rows = offsets.extent(0) - 1u;
        std::size_t const edge_count = ids.extent(0);
        if (!scores.empty() && scores.extent(0) != edge_count)
            return expected_t::fail(status_t::invalid_input_k, "{} scores ({}) must match the ids ({})", edge_kind,
                                    scores.extent(0), edge_count);
        if (offsets[0] != 0u)
            return expected_t::fail(status_t::invalid_input_k, "{} offsets must start at 0 (got {})", edge_kind,
                                    offsets[0]);
        if (offsets[rows] != edge_count)
            return expected_t::fail(status_t::invalid_input_k, "{} offsets must end at the edge count {} (got {})",
                                    edge_kind, edge_count, offsets[rows]);
        for (std::size_t row = 0; row != rows; ++row) {
            if (offsets[row] > offsets[row + 1])
                return expected_t::fail(status_t::invalid_input_k, "{} offsets decrease at row {} ({} > {})", edge_kind,
                                        row, offsets[row], offsets[row + 1]);
            for (u32_t slot = offsets[row]; slot < offsets[row + 1]; ++slot) {
                u32_t const id = ids[slot];
                if (id >= neighbors_per_mesh)
                    return expected_t::fail(status_t::invalid_input_k,
                                            "{} id {} at row {} is outside [0,{}) — ids are global", edge_kind, id, row,
                                            neighbors_per_mesh);
                if (slot > offsets[row] && ids[slot - 1] >= id)
                    return expected_t::fail( //
                        status_t::invalid_input_k,
                        "{} run of row {} is not strictly ascending ({} then {}) — the kernel's binary search " //
                        "would silently mis-score",
                        edge_kind, row, ids[slot - 1], id);
                if (!scores.empty() && scores[slot] < 0.0f)
                    return expected_t::fail(status_t::invalid_input_k, "{} score {} at row {} is negative", edge_kind,
                                            scores[slot], row);
            }
        }
        return {};
    }

    // The owned CSR columns, allocated once at their worst-case capacity and resized per upload: each
    // tensor's own shape IS the live extent, and its `capacity()` the staging ceiling — no shadow counts.
    u32r1_tensor_arena_t attract_offsets_ {}, repel_offsets_ {}, ignore_offsets_ {};
    u32r1_tensor_arena_t attract_ids_ {}, repel_ids_ {}, ignore_ids_ {};
    f32r1_tensor_arena_t attract_scores_ {}, repel_scores_ {};
    // Borrowed per-step PAD state — the batch pipeline owns these at capture-stable addresses.
    u32r1_view_t vertex_effective_lengths_ {};
    u32r1_view_t neighbor_effective_lengths_ {};
    // The self-edge default's anchor slot and the ambient repulsion — plain label scalars, not extents.
    std::size_t self_edge_base_ = 0;
    f32_t ambient_repulsion_weight_ = 1.0f;
};

} // namespace unum::uform::core
