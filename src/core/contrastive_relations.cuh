/**
 *  @file src/core/contrastive_relations.cuh
 *  @author Ash Vardanian
 *  @date July 2, 2026
 *  @brief The contrastive relation graph: per-pair attract / repel / ignore semantics over the
 *      global vertex axis, as compressed sparse rows — the edge sets of the force layout the losses
 *      relax.
 *
 *  Both contrastive losses (segments and tokens) score a @b [from_vertices_per_gpu,to_vertices_per_mesh] cell grid
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
 *      2. @c j is in @c ignore_ids[i] — weight 0: a known false negative to skip.
 *      3. @c j is @c i 's own copy while explicit attract edges exist — weight 0.
 *      4. @c j is in @c attract_ids[i] — target +1, weight from @c attract_scores (default 1).
 *      5. @c j is in @c repel_ids[i] — target −1, weight from @c repel_scores (default 1).
 *      6. Every other cell — target −1, weight @c ambient_repulsion_weight .
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

#include <memory> // `std::allocator_traits`, the edge owner's per-type rebinding

#include <cuda.h> // `cuMemcpyHtoDAsync`, `cuMemsetD8Async`
#if defined(__CUDACC__)
#include <cub/thread/thread_search.cuh> // `cub::LowerBound` — the per-run membership search
#endif

#include <numkong/tensor.hpp> // `nk::tensor`, `nk::tensor_view` — owning columns and borrowed views

#include "core/types.hpp"            // `expected`, `expected_t`, `status_t`
#include "core/gpu_allocators.hpp"   // `gpu_arena_t`, `gpu_arena_alloc` backing the owning tensors
#include "core/schedule.cuh"         // `gpu_stream_t` ordering the uploads
#include "core/multimodal_batch.hpp" // `segment_role_t`, the bag roles the builder reads

namespace unum::uform::core {

namespace nk = ashvardanian::numkong;

/** A per-anchor temperature-group id — a runtime category selecting one learnable (β, bias) entry
 *  from the recipe's device tables. Open by design: the group set is counted at runtime through
 *  @ref contrastive_recipe_t::temperature_group_count , never a fixed compile-time enumeration. */
enum class temperature_group_t : u32_t {};

/**
 *  @brief Scaling and labeling knobs of a contrastive loss — host fallbacks plus optional device views of
 *      the learnable scalars, so a captured step reads per-step-updated values without a host readback.
 *
 *  When @c inverse_temperature_device / @c bias_device are non-empty they override the host scalars;
 *  @c from_vertex_temperature_group (when non-empty, @b [from_vertices_per_gpu] ) selects each vertex's
 *  entry — e.g. a per-modality or per-dataset temperature group, one learnable (β, bias) per group,
 *  selected per anchor vertex. An empty selector means every vertex uses entry 0, bit-identical to the
 *  single-entry path. Views instead of raw pointers: every device column carries its own extent, so the
 *  entry count is never a separately-carried integer that can disagree.
 */
struct contrastive_recipe_t {

    /** Logit scale β applied to every cosine; overridden by @ref inverse_temperature_device . */
    f32_t inverse_temperature = 1.0f;

    /** Additive logit bias; overridden by @ref bias_device . */
    f32_t bias = 0.0f;

    /** Weight of every pair the graph names no edge for — the ambient all-pairs repulsion. The
     *  kernels read the copy on @ref contrastive_relations_view_t (staged through @c upload ); this one
     *  is inert after upload. */
    f32_t ambient_repulsion_weight = 1.0f;

    /** Optional @b [temperature_group_count] Float32 device entries of the learnable β; empty falls
     *  back to the host scalar. */
    f32r1_view_t inverse_temperature_device {};

    /** Optional @b [temperature_group_count] Float32 device entries of the learnable bias; empty
     *  falls back to the host scalar. */
    f32r1_view_t bias_device {};

    /** Optional @b [from_vertices_per_gpu] per-anchor temperature-group selector; empty means group
     *  0 for every vertex. */
    nk::tensor_view<temperature_group_t, 1> from_vertex_temperature_group {};

    /** Number of (β, bias) entries — carried by the device views; 1 on the host-scalar fallback. */
    [[nodiscard]] constexpr u32_t temperature_group_count() const noexcept {
        return inverse_temperature_device.empty() ? 1u : static_cast<u32_t>(inverse_temperature_device.extent(0));
    }
};

/** One resolved cell: the loss-sign multiplier and the weight; @c weight==0 means the cell is
 *  skipped and the multiplier carries no meaning. */
struct contrastive_pair_relation_t {

    /** Loss-sign multiplier (+1 attract / −1 repel) — multiplies the logit as @c target·logit . */
    f32_t target;

    /** Per-cell weight; 0 skips the cell entirely. */
    f32_t weight;
};

/** One CSR edge set over a GPU's vertex rows, borrowed: @b [rows+1] offsets, the global vertex IDs
 *  ascending per row, and per-edge weights, which an ignore set carries none of. Empty offsets name
 *  no set. */
struct contrastive_edges_view_t {
    u32r1_view_t offsets {};
    u32r1_view_t ids {};
    f32r1_view_t weights {};
};

/**
 *  @brief The device-readable relation graph — NumKong views borrowing the staged columns, passed to
 *      kernels @b by value.
 *
 *  Three CSR edge sets over the local vertex rows: attract and repel with per-edge scores, ignore
 *  without. Every extent is carried by its view: @ref from_vertices_per_gpu derives from the offsets
 *  row, and empty @ref ignore_offsets means no ignores. All ids are @b global vertex ids, sorted
 *  ascending within each row's run.
 */
struct contrastive_relations_view_t {

    /** Attract @b [from_vertices_per_gpu+1] row pointers — always present; all-zero on the
     *  self-edge default path. */
    u32r1_view_t attract_offsets {};

    /** Attract @b [attract_edge_count] global ids, sorted ascending per run; empty selects the
     *  self-edge default parity path. */
    u32r1_view_t attract_ids {};

    /** Attract @b [attract_edge_count] per-edge weights. */
    f32r1_view_t attract_scores {};

    /** Repel @b [from_vertices_per_gpu+1] row pointers. */
    u32r1_view_t repel_offsets {};

    /** Repel global ids, sorted ascending per run. */
    u32r1_view_t repel_ids {};

    /** Repel per-edge weights. */
    f32r1_view_t repel_scores {};

    /** Ignore @b [from_vertices_per_gpu+1] row pointers; empty when there are no ignores. */
    u32r1_view_t ignore_offsets {};

    /** Ignore global ids, sorted ascending per run. */
    u32r1_view_t ignore_ids {};

    /** Optional @b [from_vertices_per_gpu] UInt32 device column of from-vertex token counts; 0 == PAD
     *  (magnitude unused), a row no pair resolves against. Empty means every vertex is live. */
    u32r1_view_t from_vertex_effective_length {};

    /** Optional @b [to_vertices_per_mesh] UInt32 device column of to-vertex token counts over the
     *  whole global vertex axis, indexed by @b global id; 0 == PAD (magnitude unused). Empty means all
     *  live. */
    u32r1_view_t to_vertex_effective_length {};

    /** Global id of local vertex 0's own copy — the base of the implicit self-edges. */
    std::size_t self_edge_base = 0;

    /** Weight of every unnamed cell; 0 removes in-batch negatives. */
    f32_t ambient_repulsion_weight = 1.0f;

    /** Number of from-vertex rows the graph describes — one less than the offsets row. Like the
     *  views' own accessors, @c constexpr and therefore device-callable. */
    [[nodiscard]] constexpr u32_t from_vertices_per_gpu() const noexcept {
        return attract_offsets.empty() ? 0u : static_cast<u32_t>(attract_offsets.extent(0) - 1u);
    }

    /** Whether explicit attract edges drive the labels (vs the self-edge default). */
    [[nodiscard]] constexpr bool has_explicit_attract_edges() const noexcept { return attract_ids.extent(0) != 0u; }
};

#if defined(__CUDACC__)

/** Membership search inside one sorted CSR run — CUB's thread-level lower bound plus the equality
 *  check; the position of @p id , or @p end when absent. Vendor boundary: CUB takes a raw pointer
 *  and a count, not a view, into the CSR `ids` run. */
[[nodiscard]] __device__ inline u32_t contrastive_run_find_(u32r1_view_t ids, u32_t begin, u32_t end,
                                                            u32_t id) noexcept {
    u32_t const position = begin + cub::LowerBound(ids.data() + begin, end - begin, id);
    return position < end && ids[position] == id ? position : end;
}

/**
 *  @brief Resolve one cell of the @b [from_vertices_per_gpu,to_vertices_per_mesh] grid to its target and weight,
 *      in the file-header precedence: PAD > ignore > self > attract > repel > implicit negative.
 *      Padding is read from the view's own effective-length columns — the label grid carries its
 *      whole state.
 *
 *  @param[in] relations The graph views, taken by value at kernel launch.
 *  @param[in] vertex_index Row in @c [0,from_vertices_per_gpu) .
 *  @param[in] neighbor_global Position on the @b global vertex axis.
 */
[[nodiscard]] __device__ inline contrastive_pair_relation_t contrastive_pair_relation(
    contrastive_relations_view_t const &relations, u32_t vertex_index, u32_t neighbor_global) noexcept {

    if (!relations.from_vertex_effective_length.empty() && relations.from_vertex_effective_length[vertex_index] == 0u)
        return {0.0f, 0.0f};
    if (!relations.to_vertex_effective_length.empty() && relations.to_vertex_effective_length[neighbor_global] == 0u)
        return {0.0f, 0.0f};
    if (!relations.ignore_offsets.empty()) {
        u32_t const begin = relations.ignore_offsets[vertex_index];
        u32_t const end = relations.ignore_offsets[vertex_index + 1];
        if (contrastive_run_find_(relations.ignore_ids, begin, end, neighbor_global) != end) return {0.0f, 0.0f};
    }
    bool const is_self_edge = static_cast<std::size_t>(neighbor_global) ==
                              relations.self_edge_base + static_cast<std::size_t>(vertex_index);
    if (!relations.has_explicit_attract_edges())
        return is_self_edge ? contrastive_pair_relation_t {1.0f, 1.0f}
                            : contrastive_pair_relation_t {-1.0f, relations.ambient_repulsion_weight};
    if (is_self_edge) return {0.0f, 0.0f}; // a vertex is never its own negative once bags name positives
    // Attract lookup: an explicit positive edge carrying its own weight.
    {
        u32_t const begin = relations.attract_offsets[vertex_index];
        u32_t const end = relations.attract_offsets[vertex_index + 1];
        u32_t const position = contrastive_run_find_(relations.attract_ids, begin, end, neighbor_global);
        if (position != end) return {1.0f, relations.attract_scores[position]};
    }
    // Repel lookup: an explicit hard negative carrying its own weight.
    {
        u32_t const begin = relations.repel_offsets[vertex_index];
        u32_t const end = relations.repel_offsets[vertex_index + 1];
        u32_t const position = contrastive_run_find_(relations.repel_ids, begin, end, neighbor_global);
        if (position != end) return {-1.0f, relations.repel_scores[position]};
    }
    return {-1.0f, relations.ambient_repulsion_weight};
}

#endif // defined(__CUDACC__)

/** The three edge sets of a relation graph, in their storage order. */
enum class contrastive_edge_set_t : u32_t {
    attract_k = 0,
    repel_k = 1,
    ignore_k = 2,
};

/** The live edges of @p set over @b [3,rows+1] offsets, @b [3,capacity] IDs and @b [2,capacity]
 *  weights. A plain function, as nvcc's front end miscompiles a multidimensional subscript inside a
 *  class template. */
[[nodiscard]] inline contrastive_edges_view_t contrastive_edges_live_(u32r2_view_t offsets, u32r2_view_t ids,
                                                                      f32r2_view_t weights,
                                                                      contrastive_edge_set_t set) noexcept {
    std::size_t const index = static_cast<std::size_t>(set);
    u32r1_view_t const row_offsets = offsets[index, nk::slice];
    u32_t const count = row_offsets[row_offsets.extent(0) - 1u];
    contrastive_edges_view_t view;
    view.offsets = row_offsets;
    view.ids = ids[index, nk::range(0, count), nk::slice];
    if (set != contrastive_edge_set_t::ignore_k) view.weights = weights[index, nk::range(0, count), nk::slice];
    return view;
}

/**
 *  @brief The host owner of one GPU's attract, repel and ignore sets, sized once at @ref make and
 *      refilled every step by @ref relate_roles or by an explicit edge list.
 *
 *  Each set's live edge count is the last entry of its offsets row, so nothing beside the columns
 *  counts it.
 *
 *  @tparam alloc_type_ A byte-typed host allocator handle, @c pinned_alloc_t for staging and
 *      @c nk::aligned_allocator for host-only use, rebound per element type inside.
 */
template <typename alloc_type_>
struct contrastive_edges {

    /** @p alloc_type_ rebound to @p value_type_ , every column's allocator spelling. */
    template <typename value_type_>
    using rebound = typename std::allocator_traits<alloc_type_>::template rebind_alloc<value_type_>;

    /** Allocate @b [3,rows+1] offsets, @b [3,max_edges] IDs and @b [2,max_edges] weights, every set
     *  empty. */
    [[nodiscard]] static expected<contrastive_edges> make(u32_t rows, u32_t max_edges,
                                                          alloc_type_ allocator = {}) noexcept {
        using result_t = expected<contrastive_edges>;
        contrastive_edges edges;
        edges.offsets_ = nk::tensor<u32_t, rebound<u32_t>, 2>::zeros({3u, rows + 1u}, rebound<u32_t> {allocator}).value;
        edges.ids_ =
            nk::tensor<u32_t, rebound<u32_t>, 2>::uninitialized({3u, max_edges}, rebound<u32_t> {allocator}).value;
        edges.weights_ =
            nk::tensor<f32_t, rebound<f32_t>, 2>::uninitialized({2u, max_edges}, rebound<f32_t> {allocator}).value;
        if (!edges.offsets_ || (max_edges != 0u && (!edges.ids_ || !edges.weights_)))
            return result_t::fail(status_t::bad_alloc_k, "edge sets of {} rows and {} edges failed to allocate", rows,
                                  max_edges);
        return result_t {std::move(edges)};
    }

    /** Vertex rows every set describes. */
    [[nodiscard]] u32_t rows() const noexcept { return static_cast<u32_t>(offsets_.extent(1) - 1u); }

    /** Edges each set holds at most. */
    [[nodiscard]] u32_t capacity() const noexcept { return static_cast<u32_t>(ids_.extent(1)); }

    /** The @b [rows+1] offsets row of @p set , to write. */
    [[nodiscard]] u32r1_span_t offsets(contrastive_edge_set_t set) noexcept {
        return offsets_.row(static_cast<std::size_t>(set)).template flatten<1>();
    }

    /** The @b [capacity] IDs of @p set , to write. */
    [[nodiscard]] u32r1_span_t ids(contrastive_edge_set_t set) noexcept {
        return ids_.row(static_cast<std::size_t>(set)).template flatten<1>();
    }

    /** The @b [capacity] weights of @p set , to write; empty for the ignore set. */
    [[nodiscard]] f32r1_span_t weights(contrastive_edge_set_t set) noexcept {
        if (set == contrastive_edge_set_t::ignore_k) return {};
        return weights_.row(static_cast<std::size_t>(set)).template flatten<1>();
    }

    /** The attract set over its live edges. */
    [[nodiscard]] contrastive_edges_view_t attract() const noexcept { return live_(contrastive_edge_set_t::attract_k); }

    /** The repel set over its live edges. */
    [[nodiscard]] contrastive_edges_view_t repel() const noexcept { return live_(contrastive_edge_set_t::repel_k); }

    /** The ignore set over its live edges. */
    [[nodiscard]] contrastive_edges_view_t ignore() const noexcept { return live_(contrastive_edge_set_t::ignore_k); }

  private:

    [[nodiscard]] contrastive_edges_view_t live_(contrastive_edge_set_t set) const noexcept {
        return contrastive_edges_live_(offsets_.view(), ids_.view(), weights_.view(), set);
    }

    /** @b [3,rows+1] , @b [3,capacity] and @b [2,capacity] columns in @ref contrastive_edge_set_t
     *  order. */
    nk::tensor<u32_t, rebound<u32_t>, 2> offsets_ {}, ids_ {};
    nk::tensor<f32_t, rebound<f32_t>, 2> weights_ {};
};

/** The pinned owner the trainer stages from. */
using contrastive_edges_pinned_t = contrastive_edges<pinned_alloc_t>;

/**
 *  @brief One GPU's relation graph from its bags, a pure host function over the mixed columns,
 *      which the mixer lands whole and contiguous per bag on one GPU.
 *
 *  Inside a bag, the anchor and its positives attract each other at unit weight, the anchor and a
 *  negative repel each other at the negative's weight, and a positive and a negative, or two
 *  negatives, ignore each other. Across bags, two live vertices with one nonzero key attract, as
 *  the same media in another language does. Every other pair stays ambient, and no vertex names
 *  itself.
 *
 *  @param[in] bags The GPU's @b [segments] mixed bag column.
 *  @param[in] roles The GPU's @b [segments] mixed role column.
 *  @param[in] lengths The GPU's @b [segments] mixed length column; a zero length is PAD.
 *  @param[in] weights The GPU's @b [segments] repel weight of every negative; empty means unit
 *      weights.
 *  @param[in] world_keys The @b [world_segments] keys of every vertex by global ID, zero for PAD
 *      and for negatives; empty attracts no key.
 *  @param[in] vertex_base The global vertex of the GPU's segment 0, its world coordinate times the
 *      segments.
 *  @param[out] edges The three sets, refilled over the @b [segments] rows.
 *  @retval status_t::invalid_input_k Columns of other extents, a keyed negative, or more edges than
 *      a set holds.
 */
template <typename alloc_type_>
[[nodiscard]] expected_t relate_roles(u32r1_view_t bags, nk::tensor_view<segment_role_t, 1> roles, u32r1_view_t lengths,
                                      f32r1_view_t weights, u64r1_view_t world_keys, u32_t vertex_base,
                                      contrastive_edges<alloc_type_> &edges) noexcept {
    using set_t = contrastive_edge_set_t;
    u32_t const segments = static_cast<u32_t>(lengths.extent(0));
    u32_t const world = static_cast<u32_t>(world_keys.extent(0));
    if (bags.extent(0) != segments || roles.extent(0) != segments || edges.rows() != segments ||
        (!weights.empty() && weights.extent(0) != segments) || (world != 0u && world < vertex_base + segments))
        return expected_t::fail(status_t::invalid_input_k,
                                "{} bags, {} roles, {} lengths, {} weights, {} rows and {} world keys disagree",
                                bags.extent(0), roles.extent(0), segments, weights.extent(0), edges.rows(), world);
    u32_t written[3] = {};
    for (set_t const set : {set_t::attract_k, set_t::repel_k, set_t::ignore_k}) {
        u32r1_span_t const offsets = edges.offsets(set);
        offsets[0] = 0u;
    }
    for (u32_t segment = 0; segment != segments; ++segment) {
        bool const live = lengths[segment] != 0u;
        u32_t begin = segment, end = segment + 1u;
        if (live) {
            while (begin != 0u && bags[begin - 1u] == bags[segment] && lengths[begin - 1u] != 0u) --begin;
            while (end != segments && bags[end] == bags[segment] && lengths[end] != 0u) ++end;
        }
        segment_role_t const role = roles[segment];
        u64_t const key = live && world != 0u ? world_keys[vertex_base + segment] : 0u;
        if (key != 0u && role == segment_role_t::negative_k)
            return expected_t::fail(status_t::invalid_input_k, "negative segment {} carries key {}", segment, key);
        u32_t const bag_begin = vertex_base + begin, bag_end = vertex_base + end, self = vertex_base + segment;
        u32_t const scan_begin = key != 0u ? 0u : bag_begin, scan_end = key != 0u ? world : bag_end;
        for (u32_t partner = scan_begin; live && partner != scan_end; ++partner) {
            if (partner == self) continue;
            set_t set = set_t::attract_k;
            f32_t weight = 1.0f;
            if (partner >= bag_begin && partner < bag_end) {
                u32_t const partner_segment = partner - vertex_base;
                segment_role_t const partner_role = roles[partner_segment];
                bool const negative = role == segment_role_t::negative_k;
                bool const partner_negative = partner_role == segment_role_t::negative_k;
                if (negative != partner_negative &&
                    (role == segment_role_t::anchor_k || partner_role == segment_role_t::anchor_k)) {
                    set = set_t::repel_k;
                    u32_t const negative_segment = negative ? segment : partner_segment;
                    weight = weights.empty() ? 1.0f : weights[negative_segment];
                }
                else if (negative || partner_negative) set = set_t::ignore_k;
            }
            else if (world_keys[partner] != key) continue;
            u32_t &count = written[static_cast<u32_t>(set)];
            if (count == edges.capacity())
                return expected_t::fail(status_t::invalid_input_k, "a GPU's bags name over {} edges in one set",
                                        edges.capacity());
            u32r1_span_t const ids = edges.ids(set);
            ids[count] = partner;
            if (set != set_t::ignore_k) {
                f32r1_span_t const weights_of_set = edges.weights(set);
                weights_of_set[count] = weight;
            }
            ++count;
        }
        for (set_t const set : {set_t::attract_k, set_t::repel_k, set_t::ignore_k}) {
            u32r1_span_t const offsets = edges.offsets(set);
            offsets[segment + 1u] = written[static_cast<u32_t>(set)];
        }
    }
    return {};
}

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

    /**
     *  @brief Allocate the eight columns off @p arena at their worst-case capacity; every upload
     *      @c resize s them to the step's live extents without moving an address — the columns' own
     *      shapes are the single source of the live counts.
     *  @param[in] max_vertices Most local vertex rows any step will carry.
     *  @param[in] max_edges Most explicit edges any step will carry, per edge set.
     */
    [[nodiscard]] static expected<contrastive_relations_t> make(gpu_arena_t &arena, u32_t max_vertices,
                                                                u32_t max_edges) noexcept {
        if (max_vertices == 0)
            return expected<contrastive_relations_t>::fail(status_t::invalid_input_k,
                                                           "relations need at least one vertex row");
        contrastive_relations_t relations;
        gpu_arena_alloc<u32_t> index_alloc {arena};
        gpu_arena_alloc<f32_t> score_alloc {arena};
        std::size_t const offsets_count = max_vertices + 1u;
        relations.attract_offsets_ = u32r1_tensor_arena_t::uninitialized({offsets_count}, index_alloc).value;
        relations.repel_offsets_ = u32r1_tensor_arena_t::uninitialized({offsets_count}, index_alloc).value;
        relations.ignore_offsets_ = u32r1_tensor_arena_t::uninitialized({offsets_count}, index_alloc).value;
        bool allocated = relations.attract_offsets_ && relations.repel_offsets_ && relations.ignore_offsets_;
        if (max_edges != 0) {
            relations.attract_ids_ = u32r1_tensor_arena_t::uninitialized({max_edges}, index_alloc).value;
            relations.repel_ids_ = u32r1_tensor_arena_t::uninitialized({max_edges}, index_alloc).value;
            relations.ignore_ids_ = u32r1_tensor_arena_t::uninitialized({max_edges}, index_alloc).value;
            relations.attract_scores_ = f32r1_tensor_arena_t::uninitialized({max_edges}, score_alloc).value;
            relations.repel_scores_ = f32r1_tensor_arena_t::uninitialized({max_edges}, score_alloc).value;
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
     *  @c [0,to_vertices_per_mesh) , negative scores. There is no silent fallback.
     *
     *  @param[in] attract The attract set; its offsets span @c from_vertices_per_gpu+1 .
     *  @param[in] repel The repel set over the same rows.
     *  @param[in] ignore The ignore set over the same rows, its weights unread; empty offsets mean
     *      no ignores.
     *  @param[in] from_vertex_effective_length / @p to_vertex_effective_length The per-step PAD columns the
     *      resolver reads: @b [from_vertices_per_gpu] and @b [to_vertices_per_mesh] (by global id) device
     *      views, 0 marking PAD, empty meaning all-live. @b Borrowed, never copied — the batch pipeline owns
     *      them and keeps the addresses capture-stable.
     */
    [[nodiscard]] expected_t upload(contrastive_edges_view_t attract, contrastive_edges_view_t repel,
                                    contrastive_edges_view_t ignore, u32r1_view_t from_vertex_effective_length,
                                    u32r1_view_t to_vertex_effective_length, std::size_t to_vertices_per_mesh,
                                    std::size_t self_edge_base, f32_t ambient_repulsion_weight,
                                    gpu_stream_t &stream) noexcept {

        if (attract.offsets.extent(0) < 2 || attract.offsets.extent(0) != repel.offsets.extent(0))
            return expected_t::fail( //
                status_t::invalid_input_k,
                "attract and repel offset rows must both span from_vertices_per_gpu+1 (got {} and {})",
                attract.offsets.extent(0), repel.offsets.extent(0));
        u32_t const from_vertices_per_gpu = static_cast<u32_t>(attract.offsets.extent(0) - 1u);
        bool const has_ignores = !ignore.offsets.empty();
        if (has_ignores && ignore.offsets.extent(0) != attract.offsets.extent(0))
            return expected_t::fail( //
                status_t::invalid_input_k,
                "ignore offsets must span from_vertices_per_gpu+1 like the others (got {}, expected {})",
                ignore.offsets.extent(0), attract.offsets.extent(0));

        if (expected_t checked = validate_edges_("attract", attract.offsets, attract.ids, attract.weights,
                                                 to_vertices_per_mesh);
            !checked)
            return checked;
        if (expected_t checked = validate_edges_("repel", repel.offsets, repel.ids, repel.weights,
                                                 to_vertices_per_mesh);
            !checked)
            return checked;
        if (has_ignores)
            if (expected_t checked = validate_edges_("ignore", ignore.offsets, ignore.ids, {}, to_vertices_per_mesh);
                !checked)
                return checked;

        bool resized = nk::succeeded(attract_offsets_.resize({from_vertices_per_gpu + 1u}));
        resized = resized && nk::succeeded(repel_offsets_.resize({from_vertices_per_gpu + 1u}));
        resized = resized && nk::succeeded(attract_ids_.resize({attract.ids.extent(0)}));
        resized = resized && nk::succeeded(attract_scores_.resize({attract.weights.extent(0)}));
        resized = resized && nk::succeeded(repel_ids_.resize({repel.ids.extent(0)}));
        resized = resized && nk::succeeded(repel_scores_.resize({repel.weights.extent(0)}));
        resized = resized && nk::succeeded(ignore_offsets_.resize({has_ignores ? from_vertices_per_gpu + 1u : 0u}));
        resized = resized && nk::succeeded(ignore_ids_.resize({has_ignores ? ignore.ids.extent(0) : 0u}));
        if (!resized)
            return expected_t::fail( //
                status_t::shape_unsupported_k,
                "relations exceed the reserved capacities (vertices={}/{} attract={}/{} repel={}/{} ignore={}/{})",
                from_vertices_per_gpu + 1u, attract_offsets_.capacity(), attract.ids.extent(0), attract_ids_.capacity(),
                repel.ids.extent(0), repel_ids_.capacity(), has_ignores ? ignore.ids.extent(0) : 0u,
                ignore_ids_.capacity());

        expected_t copied = copy_host_to_device(attract.offsets, attract_offsets_.span(), stream);
        if (copied) copied = copy_host_to_device(attract.ids, attract_ids_.span(), stream);
        if (copied) copied = copy_host_to_device(attract.weights, attract_scores_.span(), stream);
        if (copied) copied = copy_host_to_device(repel.offsets, repel_offsets_.span(), stream);
        if (copied) copied = copy_host_to_device(repel.ids, repel_ids_.span(), stream);
        if (copied) copied = copy_host_to_device(repel.weights, repel_scores_.span(), stream);
        if (copied && has_ignores) copied = copy_host_to_device(ignore.offsets, ignore_offsets_.span(), stream);
        if (copied && has_ignores) copied = copy_host_to_device(ignore.ids, ignore_ids_.span(), stream);
        if (!copied) return copied;

        from_vertex_effective_length_ = from_vertex_effective_length;
        to_vertex_effective_length_ = to_vertex_effective_length;
        self_edge_base_ = self_edge_base;
        ambient_repulsion_weight_ = ambient_repulsion_weight;
        return {};
    }

    /** Stage the zero-edge self-edge default: the offsets row is zeroed (every run empty), both
     *  edge sets collapse to empty extents, and the borrowed effective-length columns are adopted as
     *  in @ref upload . */
    [[nodiscard]] expected_t upload_self_edges(u32_t from_vertices_per_gpu, u32r1_view_t from_vertex_effective_length,
                                               u32r1_view_t to_vertex_effective_length, std::size_t self_edge_base,
                                               f32_t ambient_repulsion_weight, gpu_stream_t &stream) noexcept {
        bool resized = from_vertices_per_gpu != 0 &&
                       nk::succeeded(attract_offsets_.resize({from_vertices_per_gpu + 1u}));
        resized = resized && nk::succeeded(repel_offsets_.resize({from_vertices_per_gpu + 1u}));
        resized = resized && nk::succeeded(attract_ids_.resize({0})) && nk::succeeded(attract_scores_.resize({0}));
        resized = resized && nk::succeeded(repel_ids_.resize({0})) && nk::succeeded(repel_scores_.resize({0}));
        resized = resized && nk::succeeded(ignore_offsets_.resize({0})) && nk::succeeded(ignore_ids_.resize({0}));
        if (!resized)
            return expected_t::fail(status_t::shape_unsupported_k,
                                    "self-edge relations for {} vertices exceed the column capacity of {}",
                                    from_vertices_per_gpu, attract_offsets_.capacity());
        if (expected_t reset = zero_async(attract_offsets_.span(), stream); !reset) return reset;
        from_vertex_effective_length_ = from_vertex_effective_length;
        to_vertex_effective_length_ = to_vertex_effective_length;
        self_edge_base_ = self_edge_base;
        ambient_repulsion_weight_ = ambient_repulsion_weight;
        return {};
    }

    /** Views over the owned columns' live extents — the shapes the last successful upload resized
     *  them to; kernels take this by value. An ignore set staged with zero edges is indistinguishable
     *  from no ignores, so its views collapse to empty. */
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
        view.from_vertex_effective_length = from_vertex_effective_length_;
        view.to_vertex_effective_length = to_vertex_effective_length_;
        view.self_edge_base = self_edge_base_;
        view.ambient_repulsion_weight = ambient_repulsion_weight_;
        return view;
    }

  private:

    /** All host-side invariants of one edge set (attract / repel / ignore); every violation names
     *  the @p edge_kind and the row. */
    [[nodiscard]] expected_t validate_edges_(sz::string_view_t edge_kind, u32r1_view_t offsets, u32r1_view_t ids,
                                             f32r1_view_t scores, std::size_t to_vertices_per_mesh) const noexcept {
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
            for (u32_t edge = offsets[row]; edge < offsets[row + 1]; ++edge) {
                u32_t const id = ids[edge];
                if (id >= to_vertices_per_mesh)
                    return expected_t::fail(status_t::invalid_input_k,
                                            "{} id {} at row {} is outside [0,{}) — ids are global", edge_kind, id, row,
                                            to_vertices_per_mesh);
                if (edge > offsets[row] && ids[edge - 1] >= id)
                    return expected_t::fail( //
                        status_t::invalid_input_k,
                        "{} run of row {} is not strictly ascending ({} then {}) — the kernel's binary search " //
                        "would silently mis-score",
                        edge_kind, row, ids[edge - 1], id);
                if (!scores.empty() && scores[edge] < 0.0f)
                    return expected_t::fail(status_t::invalid_input_k, "{} score {} at row {} is negative", edge_kind,
                                            scores[edge], row);
            }
        }
        return {};
    }

    /** The owned CSR columns, allocated once at their worst-case capacity and resized per upload: each
     *  tensor's own shape IS the live extent, and its `capacity()` the staging ceiling — no shadow counts. */
    u32r1_tensor_arena_t attract_offsets_ {}, repel_offsets_ {}, ignore_offsets_ {};
    u32r1_tensor_arena_t attract_ids_ {}, repel_ids_ {}, ignore_ids_ {};
    f32r1_tensor_arena_t attract_scores_ {}, repel_scores_ {};

    /** Borrowed per-step PAD state — the batch pipeline owns these at capture-stable addresses. */
    u32r1_view_t from_vertex_effective_length_ {};
    u32r1_view_t to_vertex_effective_length_ {};

    /** The self-edge default's anchor base and the ambient repulsion — plain label scalars, not extents. */
    std::size_t self_edge_base_ = 0;
    f32_t ambient_repulsion_weight_ = 1.0f;
};

} // namespace unum::uform::core
