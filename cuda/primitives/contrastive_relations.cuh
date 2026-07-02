/**
 *  @brief The contrastive relation graph: per-cell attract / repel / ignore semantics over the global
 *      candidate axis, as compressed sparse rows.
 *  @file cuda/primitives/contrastive_relations.cuh
 *  @author Ash Vardanian
 *  @date July 2, 2026
 *
 *  Both contrastive losses (segments and tokens) score a @b [num_anchors,num_candidates_global] cell grid
 *  and need one answer per cell: the target sign and the weight. This file is that answer's single source
 *  of truth — a device-readable view, the per-cell resolver, and the host owner that stages it
 *  capture-safely.
 *
 *  @section relations_semantics Per-cell contract
 *
 *  For query @c i and @b global candidate @c j , resolved by @ref contrastive_cell_target_weight in this
 *  exact precedence order (first match wins):
 *
 *      1. Either side is PAD (effective length 0) — weight 0, the cell does not exist.
 *      2. @c j is in @c ignore_ids[i] — weight 0: a known false negative or a self-pairing to skip.
 *      3. @c j is in @c attract_ids[i] — target +1, weight from @c attract_scores (default 1).
 *      4. @c j is in @c repel_ids[i] — target −1, weight from @c repel_scores (default 1).
 *      5. Every other cell — target −1, weight @c implicit_negative_weight .
 *
 *  The explicit-vs-implicit balance is @b weights-only — there is no policy flag, and
 *  @c implicit_negative_weight==0 removes in-batch negatives through the same code path. When no explicit
 *  attract edges exist the graph degenerates to the diagonal default: query @c i attracts global
 *  candidate @c diagonal_base+i with unit weight, everything else repels with
 *  @c implicit_negative_weight — the classic in-batch contract, bit-for-bit.
 *
 *  Padding is @b state of the label grid, not a parameter threaded through kernels: the view carries
 *  optional per-query and per-candidate effective-length columns, and the resolver reads them itself.
 *  Empty columns mean everything is live.
 *
 *  @section relations_staging Staging and capture
 *
 *  Ids index the @b global (post-all-gather) candidate axis and are staged global — no rank-local
 *  rebasing scratch downstream. Within each query's run the ids are @b sorted ascending: the in-kernel
 *  membership test is a per-row binary search, so an unsorted run silently mis-scores — which is exactly
 *  why @ref contrastive_relations_t::upload validates every invariant host-side, where the data still is,
 *  and fails @b loudly (capacity overflow included: an overflowing graph is an error, never a silent
 *  fallback to the diagonal).
 *
 *  The device columns are NumKong rank-1 tensors owning memory through @ref gpu_arena_alloc — allocated
 *  @b once at fixed capacity, so device addresses never change; only per-step @b contents are
 *  re-uploaded, and a captured graph replays against fresh edges. One loud exception: the view a kernel
 *  took @b by value at capture bakes in the explicit-vs-diagonal mode — flipping between them requires a
 *  @b recapture; contents-only changes never do.
 */
#pragma once

#include <cstdint> // `std::uint32_t`

#include <cuda.h> // `cuMemcpyHtoDAsync`, `cuMemsetD8Async`
#if defined(__CUDACC__)
#include <cub/thread/thread_search.cuh> // `cub::LowerBound` — the per-run membership search
#endif

#include <numkong/tensor.hpp> // `nk::tensor`, `nk::tensor_view` — owning columns and borrowed views

#include "types.cuh"      // `expected`, `expected_t`, `status_t`
#include "allocators.cuh" // `gpu_arena_t`, `gpu_arena_alloc` backing the owning tensors
#include "schedule.cuh"   // `gpu_stream_t` ordering the uploads

namespace unum::uform::primitives {

namespace nk = ashvardanian::numkong;

/**
 *  @brief Scaling and labeling knobs of a contrastive loss — host fallbacks plus optional device views of
 *      the learnable scalars, so a captured step reads per-step-updated values without a host readback.
 *
 *  When @c inverse_temperature_device / @c bias_device are non-empty they override the host scalars;
 *  @c query_pair_type (when non-empty, @b [num_anchors] ) selects each query's slot — e.g. separate
 *  temperatures for text-text vs text-image pairs. An empty selector means every query uses slot 0,
 *  bit-identical to the single-slot path. Views instead of raw pointers: every device column carries its
 *  own extent, so the slot count is never a separately-carried integer that can disagree.
 */
struct contrastive_recipe_t {
    /** @brief Logit scale β applied to every cosine; overridden by @ref inverse_temperature_device . */
    f32_t inverse_temperature = 1.0f;
    /** @brief Additive logit bias; overridden by @ref bias_device . */
    f32_t bias = 0.0f;
    /** @brief Weight of every cell not named by the graph — the in-batch negatives. */
    f32_t implicit_negative_weight = 1.0f;
    /** @brief Optional @b [pair_type_count] Float32 device slots of the learnable β; empty falls back to
     *      the host scalar. */
    f32r1_view_t inverse_temperature_device {};
    /** @brief Optional @b [pair_type_count] Float32 device slots of the learnable bias; empty falls back
     *      to the host scalar. */
    f32r1_view_t bias_device {};
    /** @brief Optional @b [num_anchors] UInt32 per-query slot selector; empty means slot 0 for all. */
    u32r1_view_t query_pair_type {};

    /** @brief Number of (β, bias) slots — carried by the device views; 1 on the host-scalar fallback. */
    [[nodiscard]] constexpr u32_t pair_type_count() const noexcept {
        return inverse_temperature_device.empty() ? 1u : static_cast<u32_t>(inverse_temperature_device.extent(0));
    }
};

/** @brief One resolved cell: the target sign (+1 attract, −1 repel) and the weight; @c weight==0 means
 *      the cell is skipped and the target carries no meaning. */
struct contrastive_cell_relation_t {
    f32_t target;
    f32_t weight;
};

/**
 *  @brief The device-readable relation graph — NumKong views borrowing the staged columns, passed to
 *      kernels @b by value.
 *
 *  Three CSR edge sets over the anchor rows: attract and repel with per-edge scores, ignore without. Every
 *  extent is carried by its view: @ref num_anchors derives from the offsets row, and empty
 *  @ref ignore_offsets means no ignores. All ids are @b global candidate ids, sorted ascending within
 *  each row's run.
 */
struct contrastive_relations_view_t {
    /** @brief Attract @b [num_anchors+1] row pointers — always present; all-zero on the diagonal path. */
    u32r1_view_t attract_offsets {};
    /** @brief Attract @b [attract_edge_count] global ids, sorted ascending per run; empty selects the
     *      diagonal-default parity path. */
    u32r1_view_t attract_ids {};
    /** @brief Attract @b [attract_edge_count] per-edge weights. */
    f32r1_view_t attract_scores {};
    /** @brief Repel @b [num_anchors+1] row pointers. */
    u32r1_view_t repel_offsets {};
    /** @brief Repel global ids, sorted ascending per run. */
    u32r1_view_t repel_ids {};
    /** @brief Repel per-edge weights. */
    f32r1_view_t repel_scores {};
    /** @brief Ignore @b [num_anchors+1] row pointers; empty when there are no ignores. */
    u32r1_view_t ignore_offsets {};
    /** @brief Ignore global ids, sorted ascending per run. */
    u32r1_view_t ignore_ids {};
    /** @brief Optional @b [num_anchors] UInt32 device column of per-query effective lengths; 0 marks a
     *      PAD row the cell grid skips. Empty means every query is live. */
    u32r1_view_t query_effective_lengths {};
    /** @brief Optional @b [num_candidates_global] UInt32 device column of per-candidate effective
     *      lengths, indexed by @b global candidate id; 0 marks a PAD column. Empty means all live. */
    u32r1_view_t candidate_effective_lengths {};

    /** @brief Global candidate id of query 0's diagonal partner on the default path. */
    std::size_t diagonal_base = 0;
    /** @brief Weight of every unnamed cell; 0 removes in-batch negatives. */
    f32_t implicit_negative_weight = 1.0f;

    /** @brief Number of query rows the graph describes — one less than the offsets row. Like the views'
     *      own accessors, @c constexpr and therefore device-callable. */
    [[nodiscard]] constexpr u32_t num_anchors() const noexcept {
        return attract_offsets.empty() ? 0u : static_cast<u32_t>(attract_offsets.extent(0) - 1u);
    }
    /** @brief Whether explicit attract edges drive the labels (vs the diagonal default). */
    [[nodiscard]] constexpr bool has_explicit_edges() const noexcept { return attract_ids.extent(0) != 0u; }
};

#if defined(__CUDACC__)

/** @brief Membership search inside one sorted CSR run — CUB's thread-level lower bound plus the equality
 *      check; the slot index of @p id , or @p end when absent. */
[[nodiscard]] __device__ inline u32_t contrastive_run_find_(u32r1_view_t ids, u32_t begin, u32_t end,
                                                            u32_t id) noexcept {
    u32_t const slot = begin + cub::LowerBound(ids.data() + begin, end - begin, id);
    return slot < end && ids[slot] == id ? slot : end;
}

/** @brief Resolve one cell of the @b [num_anchors,num_candidates_global] grid to its target and weight,
 *      in the file-header precedence: PAD > ignore > attract > repel > implicit negative. Padding is read
 *      from the view's own effective-length columns — the label grid carries its whole state.
 *
 *  @param[in] relations The graph views, taken by value at kernel launch.
 *  @param[in] query_index Row in @c [0,num_anchors) .
 *  @param[in] candidate_global Column on the @b global candidate axis. */
[[nodiscard]] __device__ inline contrastive_cell_relation_t contrastive_cell_target_weight(
    contrastive_relations_view_t const &relations, u32_t query_index, u32_t candidate_global) noexcept {

    if (!relations.query_effective_lengths.empty() && relations.query_effective_lengths[query_index] == 0u)
        return {0.0f, 0.0f};
    if (!relations.candidate_effective_lengths.empty() && relations.candidate_effective_lengths[candidate_global] == 0u)
        return {0.0f, 0.0f};
    if (!relations.ignore_offsets.empty()) {
        u32_t const begin = relations.ignore_offsets[query_index];
        u32_t const end = relations.ignore_offsets[query_index + 1];
        if (contrastive_run_find_(relations.ignore_ids, begin, end, candidate_global) != end) return {0.0f, 0.0f};
    }
    if (!relations.has_explicit_edges()) {
        bool const is_diagonal = static_cast<std::size_t>(candidate_global) ==
                                 relations.diagonal_base + static_cast<std::size_t>(query_index);
        return is_diagonal ? contrastive_cell_relation_t {1.0f, 1.0f}
                           : contrastive_cell_relation_t {-1.0f, relations.implicit_negative_weight};
    }
    // Attract lookup: an explicit positive edge carrying its own weight.
    {
        u32_t const begin = relations.attract_offsets[query_index];
        u32_t const end = relations.attract_offsets[query_index + 1];
        u32_t const slot = contrastive_run_find_(relations.attract_ids, begin, end, candidate_global);
        if (slot != end) return {1.0f, relations.attract_scores[slot]};
    }
    // Repel lookup: an explicit hard negative carrying its own weight.
    {
        u32_t const begin = relations.repel_offsets[query_index];
        u32_t const end = relations.repel_offsets[query_index + 1];
        u32_t const slot = contrastive_run_find_(relations.repel_ids, begin, end, candidate_global);
        if (slot != end) return {-1.0f, relations.repel_scores[slot]};
    }
    return {-1.0f, relations.implicit_negative_weight};
}

#endif // defined(__CUDACC__)

/**
 *  @brief The host owner of the relation graph — one memory-owning NumKong tensor per column, allocated
 *      through @ref gpu_arena_alloc at fixed capacity, re-uploaded per step, and viewed by kernels
 *      through @ref contrastive_relations_view_t windows over the live extents.
 *
 *  Eight columns: three offset rows @b [max_anchors+1] , three id runs @b [max_edges] , two score runs
 *  @b [max_edges] . Addresses never change after @ref make — the capture-stability contract — so a
 *  replayed graph reads whatever @ref upload staged last. Uploads run @b outside any capture.
 *
 *  Ownership is deliberately split: the CSR columns are @b owned (they are staged copies of host data
 *  with no other home), while the effective-length columns are @b borrowed device views — the batch
 *  pipeline owns those lengths and re-fills them per step at capture-stable addresses; copying them here
 *  would duplicate per-step state. Both arrive through @ref upload / @ref upload_diagonal , the single
 *  staging entry points.
 */
struct contrastive_relations_t {

    /** @brief Carve the eight fixed-capacity columns off @p arena .
     *  @param[in] max_anchors Most queries any step will carry.
     *  @param[in] max_edges Most explicit edges any step will carry, per edge set. */
    [[nodiscard]] static expected<contrastive_relations_t> make(gpu_arena_t &arena, u32_t max_anchors,
                                                                u32_t max_edges) noexcept {
        if (max_anchors == 0)
            return expected<contrastive_relations_t>::fail(status_t::invalid_input_k,
                                                           "relations need at least one anchor");
        contrastive_relations_t relations;
        relations.max_anchors_ = max_anchors;
        relations.max_edges_ = max_edges;
        gpu_arena_alloc<u32_t> index_alloc {arena};
        gpu_arena_alloc<f32_t> score_alloc {arena};
        std::size_t const offsets_count = max_anchors + 1u;
        relations.attract_offsets_ = u32r1_tensor_arena_t::try_empty({offsets_count}, index_alloc);
        relations.repel_offsets_ = u32r1_tensor_arena_t::try_empty({offsets_count}, index_alloc);
        relations.ignore_offsets_ = u32r1_tensor_arena_t::try_empty({offsets_count}, index_alloc);
        bool carved = relations.attract_offsets_ && relations.repel_offsets_ && relations.ignore_offsets_;
        if (max_edges != 0) {
            relations.attract_ids_ = u32r1_tensor_arena_t::try_empty({max_edges}, index_alloc);
            relations.repel_ids_ = u32r1_tensor_arena_t::try_empty({max_edges}, index_alloc);
            relations.ignore_ids_ = u32r1_tensor_arena_t::try_empty({max_edges}, index_alloc);
            relations.attract_scores_ = f32r1_tensor_arena_t::try_empty({max_edges}, score_alloc);
            relations.repel_scores_ = f32r1_tensor_arena_t::try_empty({max_edges}, score_alloc);
            carved = carved && relations.attract_ids_ && relations.repel_ids_ && relations.ignore_ids_ &&
                     relations.attract_scores_ && relations.repel_scores_;
        }
        if (!carved)
            return expected<contrastive_relations_t>::fail(
                status_t::bad_alloc_k, "relations column allocation failed (max_anchors={} max_edges={} arena_used={})",
                max_anchors, max_edges, arena.used());
        return expected<contrastive_relations_t> {std::move(relations)};
    }

    /**
     *  @brief Validate a host CSR and stage it into the device columns on @p stream — @b outside capture.
     *
     *  Every invariant is checked here, where the data is still host-resident, and every violation is a
     *  named failure: capacity overflow, offset rows of the wrong length, non-monotone or non-zero-based
     *  offsets, id/score length mismatches, unsorted or duplicate ids within a run, ids outside
     *  @c [0,num_candidates_global) , negative scores. There is no silent fallback.
     *
     *  @param[in] ignore_offsets / @p ignore_ids Empty views mean no ignores.
     *  @param[in] query_effective_lengths / @p candidate_effective_lengths The per-step PAD columns the
     *      resolver reads: @b [num_anchors] and @b [num_candidates_global] (by global id) device views, 0
     *      marking PAD, empty meaning all-live. @b Borrowed, never copied — the batch pipeline owns them
     *      and keeps the addresses capture-stable.
     */
    [[nodiscard]] expected_t upload(                                                                  //
        u32r1_view_t attract_offsets, u32r1_view_t attract_ids,                                       //
        f32r1_view_t attract_scores,                                                                  //
        u32r1_view_t repel_offsets, u32r1_view_t repel_ids,                                           //
        f32r1_view_t repel_scores,                                                                    //
        u32r1_view_t ignore_offsets, u32r1_view_t ignore_ids,                                         //
        u32r1_view_t query_effective_lengths, u32r1_view_t candidate_effective_lengths,               //
        std::size_t num_candidates_global, std::size_t diagonal_base, f32_t implicit_negative_weight, //
        gpu_stream_t const &stream) noexcept {

        if (attract_offsets.extent(0) < 2 || attract_offsets.extent(0) != repel_offsets.extent(0))
            return expected_t::fail( //
                status_t::invalid_input_k, "attract and repel offset rows must both span num_anchors+1 (got {} and {})",
                attract_offsets.extent(0), repel_offsets.extent(0));
        u32_t const num_anchors = static_cast<u32_t>(attract_offsets.extent(0) - 1u);
        if (num_anchors > max_anchors_)
            return expected_t::fail( //
                status_t::shape_unsupported_k, "relations staged for {} anchors exceed the column capacity of {}",
                num_anchors, max_anchors_);
        bool const has_ignores = !ignore_offsets.empty();
        if (has_ignores && ignore_offsets.extent(0) != attract_offsets.extent(0))
            return expected_t::fail( //
                status_t::invalid_input_k,
                "ignore offsets must span num_anchors+1 like the others (got {}, expected {})",
                ignore_offsets.extent(0), attract_offsets.extent(0));

        if (expected_t checked = validate_edges_("attract", attract_offsets, attract_ids, attract_scores,
                                                 num_candidates_global);
            !checked)
            return checked;
        if (expected_t checked = validate_edges_("repel", repel_offsets, repel_ids, repel_scores,
                                                 num_candidates_global);
            !checked)
            return checked;
        if (has_ignores)
            if (expected_t checked = validate_edges_("ignore", ignore_offsets, ignore_ids, {}, num_candidates_global);
                !checked)
                return checked;

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

        num_anchors_ = num_anchors;
        attract_edge_count_ = static_cast<u32_t>(attract_ids.extent(0));
        repel_edge_count_ = static_cast<u32_t>(repel_ids.extent(0));
        ignore_edge_count_ = has_ignores ? static_cast<u32_t>(ignore_ids.extent(0)) : 0u;
        query_effective_lengths_ = query_effective_lengths;
        candidate_effective_lengths_ = candidate_effective_lengths;
        diagonal_base_ = diagonal_base;
        implicit_negative_weight_ = implicit_negative_weight;
        return {};
    }

    /** @brief Stage the zero-edge diagonal default: the offsets row is zeroed (every run empty), both
     *      edge sets collapse to empty views, and the borrowed effective-length columns are adopted as in
     *      @ref upload . */
    [[nodiscard]] expected_t upload_diagonal(u32_t num_anchors, u32r1_view_t query_effective_lengths,
                                             u32r1_view_t candidate_effective_lengths, std::size_t diagonal_base,
                                             f32_t implicit_negative_weight, gpu_stream_t const &stream) noexcept {
        if (num_anchors == 0 || num_anchors > max_anchors_)
            return expected_t::fail(status_t::shape_unsupported_k,
                                    "diagonal relations for {} anchors exceed the column capacity of {}", num_anchors,
                                    max_anchors_);
        if (CUresult result = cuMemsetD8Async(reinterpret_cast<CUdeviceptr>(attract_offsets_.data()), 0,
                                              (num_anchors + 1u) * sizeof(u32_t), stream.get());
            result != CUDA_SUCCESS)
            return expected_t::fail(result, "diagonal offsets reset failed (num_anchors={})", num_anchors);
        num_anchors_ = num_anchors;
        attract_edge_count_ = 0u;
        repel_edge_count_ = 0u;
        ignore_edge_count_ = 0u;
        query_effective_lengths_ = query_effective_lengths;
        candidate_effective_lengths_ = candidate_effective_lengths;
        diagonal_base_ = diagonal_base;
        implicit_negative_weight_ = implicit_negative_weight;
        return {};
    }

    /** @brief Views over the live extents of the owned columns — what kernels take by value. Valid after
     *      the last successful upload. An ignore set staged with zero edges is indistinguishable from no
     *      ignores, so its views collapse to empty. */
    [[nodiscard]] contrastive_relations_view_t view() const noexcept {
        contrastive_relations_view_t view;
        view.attract_offsets = {attract_offsets_.data(), num_anchors_ + 1u};
        view.attract_ids = {attract_ids_.data(), attract_edge_count_};
        view.attract_scores = {attract_scores_.data(), attract_edge_count_};
        if (attract_edge_count_ != 0 || repel_edge_count_ != 0) {
            view.repel_offsets = {repel_offsets_.data(), num_anchors_ + 1u};
            view.repel_ids = {repel_ids_.data(), repel_edge_count_};
            view.repel_scores = {repel_scores_.data(), repel_edge_count_};
        }
        if (ignore_edge_count_ != 0) {
            view.ignore_offsets = {ignore_offsets_.data(), num_anchors_ + 1u};
            view.ignore_ids = {ignore_ids_.data(), ignore_edge_count_};
        }
        view.query_effective_lengths = query_effective_lengths_;
        view.candidate_effective_lengths = candidate_effective_lengths_;
        view.diagonal_base = diagonal_base_;
        view.implicit_negative_weight = implicit_negative_weight_;
        return view;
    }

  private:
    /** @brief All host-side invariants of one edge set (attract / repel / ignore); every violation names
     *      the @p edge_kind and the row. */
    [[nodiscard]] expected_t validate_edges_(char const *edge_kind, u32r1_view_t offsets, u32r1_view_t ids,
                                             f32r1_view_t scores, std::size_t num_candidates_global) const noexcept {
        std::size_t const rows = offsets.extent(0) - 1u;
        std::size_t const edge_count = ids.extent(0);
        if (edge_count > max_edges_)
            return expected_t::fail(status_t::bad_alloc_k, "{} edges ({}) exceed the column capacity of {}", edge_kind,
                                    edge_count, max_edges_);
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
                if (id >= num_candidates_global)
                    return expected_t::fail(status_t::invalid_input_k,
                                            "{} id {} at row {} is outside [0,{}) — ids are global", edge_kind, id, row,
                                            num_candidates_global);
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

    u32r1_tensor_arena_t attract_offsets_ {}, repel_offsets_ {}, ignore_offsets_ {};
    u32r1_tensor_arena_t attract_ids_ {}, repel_ids_ {}, ignore_ids_ {};
    f32r1_tensor_arena_t attract_scores_ {}, repel_scores_ {};
    u32r1_view_t query_effective_lengths_ {};
    u32r1_view_t candidate_effective_lengths_ {};
    u32_t max_anchors_ = 0;
    u32_t max_edges_ = 0;
    u32_t num_anchors_ = 0;
    u32_t attract_edge_count_ = 0;
    u32_t repel_edge_count_ = 0;
    u32_t ignore_edge_count_ = 0;
    std::size_t diagonal_base_ = 0;
    f32_t implicit_negative_weight_ = 1.0f;
};

} // namespace unum::uform::primitives
