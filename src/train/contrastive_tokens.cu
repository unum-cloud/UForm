/**
 *  @file src/train/contrastive_tokens.cu
 *  @author Ash Vardanian
 *  @date October 21, 2026
 *  @brief Device kernels and method bodies of the late-interaction tokens loss declared in
 *      @ref contrastive_tokens.cuh .
 */
#include <cmath> // `std::fmax`

#include <limits> // `std::numeric_limits` — the empty maximum

#include <cub/block/block_scan.cuh>     // `cub::BlockScan` — the tile layouts
#include <cub/thread/thread_search.cuh> // `cub::UpperBound` — a tile row's segment

#include "uform/train/contrastive_tokens.cuh"

namespace ue = unum::uform::encoder;
using ue::late_interaction_side_t;
using ue::multimodal_batch_view_t;
using namespace unum::uform::train;
namespace util = unum::uform::utilities;

/** Whether @p side queries in late interaction. */
[[nodiscard]] static __device__ inline bool contrastive_tokens_queries_(late_interaction_side_t side) noexcept {
    return side == late_interaction_side_t::query_k || side == late_interaction_side_t::query_and_document_k;
}

/** Whether @p side is queried in late interaction. */
[[nodiscard]] static __device__ inline bool contrastive_tokens_documents_(late_interaction_side_t side) noexcept {
    return side == late_interaction_side_t::document_k || side == late_interaction_side_t::query_and_document_k;
}

/** The entry of the ascending @p offsets run whose range holds @p row , for a @p row below
 *  its last entry. */
[[nodiscard]] static __device__ inline util::u32_t contrastive_tokens_owner_(util::u32r1_view_t offsets,
                                                                             util::u32_t row) noexcept {
    return static_cast<util::u32_t>(cub::UpperBound(offsets.data(), static_cast<util::u32_t>(offsets.extent(0)), row)) -
           1u;
}

/**
 *  @brief Lay out one GPU's query tile and documents from the token head's spans: the query offsets
 *      and spans by segment, the documents' first rows and segments by document, each segment's
 *      document, the live query count, and the overflow mark when a tile outgrows its capacity.
 *
 *  A document never crosses a band: one that would is moved to the next band's first row, leaving
 *  the rows between no document's, so every band's epilogue sees whole documents.
 *
 *  __grid_size__ @c [1,1,1] ; one CTA scans the segments in chunks of its width, then one thread
 *  places the documents in order; no dynamic shared memory.
 */
static __global__ void contrastive_tokens_layout_kernel_( //
    multimodal_batch_view_t batch, util::u32r1_view_t token_offsets, util::u32r1_span_t query_offsets,
    util::u32r1_span_t query_lengths, util::u32r1_span_t document_offsets, util::u32r1_span_t document_segments,
    util::u32r1_span_t document_lengths, util::u32r1_span_t document_of_segment, util::f32r1_span_t live_queries,
    util::u32r1_span_t overflow, util::u32_t query_capacity, util::u32_t document_capacity, util::u32_t band_tokens,
    util::u32_t query_span_tokens) {
    using block_scan_t = cub::BlockScan<util::u32_t, util::kernel_block_size_default_k>;
    __shared__ typename block_scan_t::TempStorage scan_storage;
    __shared__ util::u32_t query_rows, documents, queries;
    util::u32_t const segments = static_cast<util::u32_t>(query_lengths.extent(0));
    util::u32_t const documents_capacity = static_cast<util::u32_t>(document_segments.extent(0));
    if (threadIdx.x == 0u) query_rows = documents = queries = 0u;
    __syncthreads();
    for (util::u32_t chunk = 0; chunk < segments; chunk += blockDim.x) {
        util::u32_t const segment = chunk + threadIdx.x;
        bool const live = segment < segments;
        util::u32_t const span = live ? token_offsets[segment + 1u] - token_offsets[segment] : 0u;
        late_interaction_side_t const side = live ? batch.segment_side[segment] : late_interaction_side_t::none_k;
        util::u32_t const query_span = contrastive_tokens_queries_(side) ? fu::min_of(span, query_span_tokens) : 0u;
        util::u32_t const is_document = contrastive_tokens_documents_(side) && span != 0u ? 1u : 0u;
        util::u32_t const document_span = is_document ? span : 0u;
        util::u32_t query_row = 0, document = 0, is_query = 0, totals[3] = {};
        block_scan_t(scan_storage).ExclusiveSum(query_span, query_row, totals[0]);
        __syncthreads();
        block_scan_t(scan_storage).ExclusiveSum(is_document, document, totals[1]);
        __syncthreads();
        block_scan_t(scan_storage).ExclusiveSum(query_span != 0u ? 1u : 0u, is_query, totals[2]);
        if (live) {
            query_offsets[segment] = query_rows + query_row;
            query_lengths[segment] = query_span;
            document_lengths[segment] = document_span;
            util::u32_t const ordinal = documents + document;
            document_of_segment[segment] = is_document && ordinal < documents_capacity ? ordinal : ~0u;
            if (is_document && ordinal < documents_capacity) document_segments[ordinal] = segment;
        }
        __syncthreads();
        if (threadIdx.x == 0u) {
            query_rows += totals[0];
            documents += totals[1];
            queries += totals[2];
        }
        __syncthreads();
    }
    for (util::u32_t ordinal = documents + threadIdx.x; ordinal < documents_capacity; ordinal += blockDim.x)
        document_segments[ordinal] = ~0u;
    __syncthreads();
    if (threadIdx.x != 0u) return;
    util::u32_t cursor = 0, placed = fu::min_of(documents, documents_capacity);
    bool oversized = false;
    for (util::u32_t ordinal = 0; ordinal != placed; ++ordinal) {
        util::u32_t const span = document_lengths[document_segments[ordinal]];
        oversized = oversized || span > band_tokens;
        if (cursor % band_tokens + span > band_tokens) cursor += band_tokens - cursor % band_tokens;
        document_offsets[ordinal] = cursor;
        cursor += span;
    }
    for (util::u32_t ordinal = placed; ordinal <= documents_capacity; ++ordinal) document_offsets[ordinal] = cursor;
    query_offsets[segments] = query_rows;
    live_queries[0] = static_cast<util::f32_t>(queries);
    overflow[0] = oversized || query_rows > query_capacity || cursor > document_capacity ||
                          documents > documents_capacity
                      ? 1u
                      : 0u;
}

/**
 *  @brief Gather the query tile's rows from the token head's, with each row's segment, zeroing the
 *      rows past the live total.
 *
 *  __grid_size__ @c [query_tokens,1,1] ; one CTA per row; threads stripe the width; no
 *  dynamic shared memory.
 */
static __global__ void contrastive_tokens_gather_queries_kernel_(util::bf16r2_view_t token_vectors,
                                                                 util::u32r1_view_t token_offsets,
                                                                 util::u32r1_view_t query_offsets,
                                                                 util::bf16r2_span_t query_tile,
                                                                 util::u32r1_span_t query_segments) {
    util::u32_t const row = blockIdx.x;
    std::size_t const segments = query_offsets.extent(0) - 1u;
    bool const live = row < query_offsets[segments];
    util::u32_t const segment = live ? contrastive_tokens_owner_(query_offsets, row) : ~0u;
    util::u32_t const source = live ? token_offsets[segment] + (row - query_offsets[segment]) : 0u;
    for (std::size_t channel = threadIdx.x; channel < query_tile.extent(1); channel += blockDim.x)
        query_tile[row, channel] = live ? token_vectors[source, channel] : __float2bfloat16(0.0f);
    if (threadIdx.x == 0u) query_segments[row] = segment;
}

/**
 *  @brief Gather the documents' rows from the token head's, with each row's document, zeroing the
 *      rows past the live total.
 *
 *  __grid_size__ @c [document_tokens,1,1] ; one CTA per row; threads stripe the width; no
 *  dynamic shared memory.
 */
static __global__ void contrastive_tokens_gather_documents_kernel_(
    util::bf16r2_view_t token_vectors, util::u32r1_view_t token_offsets, util::u32r1_view_t document_offsets,
    util::u32r1_view_t document_segments, util::bf16r2_span_t document_tile, util::u32r1_span_t document_of_row) {
    util::u32_t const row = blockIdx.x;
    util::u32_t const owner = row < document_offsets[document_offsets.extent(0) - 1u]
                                  ? contrastive_tokens_owner_(document_offsets, row)
                                  : ~0u;
    util::u32_t const segment = owner != ~0u ? document_segments[owner] : ~0u;
    bool const live = segment != ~0u &&
                      row < document_offsets[owner] + token_offsets[segment + 1u] - token_offsets[segment];
    util::u32_t const document = live ? owner : ~0u;
    util::u32_t const source = live ? token_offsets[segment] + (row - document_offsets[owner]) : 0u;
    for (std::size_t channel = threadIdx.x; channel < document_tile.extent(1); channel += blockDim.x)
        document_tile[row, channel] = live ? token_vectors[source, channel] : __float2bfloat16(0.0f);
    if (threadIdx.x == 0u) document_of_row[row] = document;
}

/** One warp's view of a band chunk: the document of each lane's row, @c ~0u past the documents, the
 *  first lane of its document's run within the chunk, and whether the lane closes that run. */
struct contrastive_tokens_lane_t {
    util::u32_t document;
    util::u32_t run_begin_lane;
    bool closes_run;
};

/** Classify lane @p lane holding @p document within its chunk; the documents tile
 *  the rows contiguously. */
[[nodiscard]] static __device__ inline contrastive_tokens_lane_t contrastive_tokens_lane_(util::u32_t document,
                                                                                          util::u32_t lane) noexcept {
    util::u32_t const previous = __shfl_up_sync(0xffffffffu, document, 1u);
    util::u32_t const next = __shfl_down_sync(0xffffffffu, document, 1u);
    util::u32_t run_begin_lane = lane == 0u || previous != document ? lane : 0u;
    for (util::u32_t delta = 1u; delta != util::warp_size_k; delta *= 2u) {
        util::u32_t const earlier = __shfl_up_sync(0xffffffffu, run_begin_lane, delta);
        if (lane >= delta) run_begin_lane = fu::max_of(run_begin_lane, earlier);
    }
    return {document, run_begin_lane, document != ~0u && (lane + 1u == util::warp_size_k || next != document)};
}

/** The warp-wide maximum of @p value , on every lane. */
[[nodiscard]] static __device__ inline util::f32_t contrastive_tokens_warp_max_(util::f32_t value) noexcept {
    for (util::u32_t delta = util::warp_size_k / 2u; delta != 0u; delta /= 2u)
        value = std::fmax(value, __shfl_xor_sync(0xffffffffu, value, delta));
    return value;
}

/** Rows a warp covers per wide step, eight per lane through two 16-byte loads and
 *  one 16-byte store. */
inline constexpr std::size_t contrastive_tokens_wide_rows_k = 8u * util::warp_size_k;

/** The four products at @p column and after of a 16-byte aligned row. */
[[nodiscard]] static __device__ inline float4 contrastive_tokens_load_(util::f32_t const* row_start,
                                                                       std::size_t column) noexcept {
    float4 vector;
    asm("ld.global.v4.f32 {%0, %1, %2, %3}, [%4];"
        : "=f"(vector.x), "=f"(vector.y), "=f"(vector.z), "=f"(vector.w)
        : "l"(row_start + column));
    return vector;
}

/** The BFloat16 pair @p low , @p high packed into one word, @p low first in memory. */
[[nodiscard]] static __device__ inline util::u32_t contrastive_tokens_pack_(util::f32_t low,
                                                                            util::f32_t high) noexcept {
    return static_cast<util::u32_t>(__bfloat16_as_ushort(__float2bfloat16(low))) |
           static_cast<util::u32_t>(__bfloat16_as_ushort(__float2bfloat16(high))) << 16u;
}

/** Store the eight gradients @p low then @p high at @p column and after of a 16-byte
 *  aligned BFloat16 row. */
static __device__ inline void contrastive_tokens_store_(                             //
    util::bf16_t* row_start, std::size_t column, float4 low, float4 high) noexcept { // device-boundary
    asm volatile("st.global.v4.u32 [%0], {%1, %2, %3, %4};" ::"l"(row_start + column),
                 "r"(contrastive_tokens_pack_(low.x, low.y)), "r"(contrastive_tokens_pack_(low.z, low.w)),
                 "r"(contrastive_tokens_pack_(high.x, high.y)), "r"(contrastive_tokens_pack_(high.z, high.w))
                 : "memory");
}

/** The document of band row @p row , or @c ~0u past the live rows. */
[[nodiscard]] static __device__ inline util::u32_t contrastive_tokens_document_(util::u32r1_view_t document_of_row,
                                                                                std::size_t row,
                                                                                std::size_t live_rows) noexcept {
    return row < live_rows ? document_of_row[row] : ~0u;
}

/**
 *  @brief Fold one band's products into every live query token's maximum per document, written once
 *      per document since a document never crosses a band.
 *
 *  A warp walks its row. A wide step of 256 rows inside the open document folds eight rows per lane
 *  into a register through two 16-byte loads; any other step walks its 32-row chunks, where a chunk
 *  holding only the open document folds per lane, and a chunk where documents change closes the
 *  open one with one warp reduce, reduces the documents it holds whole with a segmented scan, and
 *  opens its last.
 *
 *  __grid_size__ @c [ceil(query_tokens/8),1,1] ; one warp per query row, 8 per CTA; no
 *  dynamic shared memory.
 */
static __global__ void contrastive_tokens_band_max_kernel_(util::f32r2_view_t band_products,
                                                           util::u32r1_view_t query_segments,
                                                           util::u32r1_view_t document_of_row,
                                                           util::u32r1_view_t document_offsets, std::size_t band_begin,
                                                           util::f32r2_span_t maxima) {
    util::u32_t const lane = threadIdx.x % util::warp_size_k;
    std::size_t const query_row = static_cast<std::size_t>(blockIdx.x) * (blockDim.x / util::warp_size_k) +
                                  threadIdx.x / util::warp_size_k;
    if (query_row >= band_products.extent(0) || query_segments[query_row] == ~0u) return;
    std::size_t const live_rows = document_offsets[document_offsets.extent(0) - 1u];
    std::size_t const band_rows = band_products.extent(1);
    util::f32_t const* const products = &band_products[query_row, 0];
    util::u32_t open_document = ~0u;
    util::f32_t open_maximum = -std::numeric_limits<util::f32_t>::infinity();
    auto const fold_chunk = [&](std::size_t chunk) noexcept {
        std::size_t const row = band_begin + chunk + lane;
        util::u32_t const document = contrastive_tokens_document_(document_of_row, row, live_rows);
        util::f32_t const value = products[chunk + lane];
        if (__all_sync(0xffffffffu, document == open_document)) {
            open_maximum = std::fmax(open_maximum, value);
            return;
        }
        util::f32_t const closing = contrastive_tokens_warp_max_(
            std::fmax(open_maximum, document == open_document ? value : -std::numeric_limits<util::f32_t>::infinity()));
        if (lane == 0u && open_document != ~0u) maxima[query_row, open_document] = closing;
        contrastive_tokens_lane_t const position = contrastive_tokens_lane_(document, lane);
        util::f32_t reduced = value;
        for (util::u32_t delta = 1u; delta != util::warp_size_k; delta *= 2u) {
            util::f32_t const earlier = __shfl_up_sync(0xffffffffu, reduced, delta);
            if (lane >= delta && lane - delta >= position.run_begin_lane) reduced = std::fmax(reduced, earlier);
        }
        util::u32_t const last_document = __shfl_sync(0xffffffffu, document, util::warp_size_k - 1u);
        if (position.closes_run && document != open_document && document != last_document)
            maxima[query_row, document] = reduced;
        open_document = last_document;
        open_maximum = document == last_document ? value : -std::numeric_limits<util::f32_t>::infinity();
    };
    for (std::size_t step = 0; step < band_rows; step += contrastive_tokens_wide_rows_k) {
        std::size_t const step_end = step + contrastive_tokens_wide_rows_k;
        bool const inside_open = step_end <= band_rows && open_document != ~0u &&
                                 contrastive_tokens_document_(document_of_row, band_begin + step_end - 1u, live_rows) ==
                                     open_document;
        if (inside_open) {
            float4 const low = contrastive_tokens_load_(products, step + 8u * lane);
            float4 const high = contrastive_tokens_load_(products, step + 8u * lane + 4u);
            open_maximum = std::fmax(open_maximum,
                                     std::fmax(std::fmax(std::fmax(low.x, low.y), std::fmax(low.z, low.w)),
                                               std::fmax(std::fmax(high.x, high.y), std::fmax(high.z, high.w))));
            continue;
        }
        for (std::size_t chunk = step; chunk < step_end && chunk < band_rows; chunk += util::warp_size_k)
            fold_chunk(chunk);
    }
    util::f32_t const closing = contrastive_tokens_warp_max_(open_maximum);
    if (lane == 0u && open_document != ~0u) maxima[query_row, open_document] = closing;
}

/**
 *  @brief The raw MaxSim scores of one query tile: @c raw[q,d] = Σ_t maxima[t,d]/|q| over the
 *      query's rows, in row order, for every live query segment and document; the caller
 *      zeroed the rest.
 *
 *  __grid_size__ @c [segments,1,1] ; one CTA per query segment; threads stripe the documents; no
 *  dynamic shared memory.
 */
static __global__ void contrastive_tokens_scores_kernel_(util::f32r2_view_t maxima, util::u32r1_view_t query_offsets,
                                                         util::u32r1_view_t document_segments,
                                                         util::f32r2_span_t raw_scores) {
    util::u32_t const segment = blockIdx.x;
    std::size_t const rows = maxima.extent(0);
    std::size_t const begin = query_offsets[segment], end = query_offsets[segment + 1u];
    if (begin == end || end > rows) return;
    util::f32_t const inverse_length = 1.0f / static_cast<util::f32_t>(end - begin);
    for (std::size_t document = threadIdx.x; document < document_segments.extent(0); document += blockDim.x) {
        if (document_segments[document] == ~0u) continue;
        util::f32_t sum = 0.0f;
        for (std::size_t row = begin; row != end; ++row) sum += maxima[row, document];
        raw_scores[segment, document_segments[document]] = sum * inverse_length;
    }
}

/**
 *  @brief One band's credited gradient: for every live query token and document, the document's
 *      lowest row whose recomputed product equals the stored maximum takes @c d(raw)/|q| , every
 *      other cell zero.
 *
 *  Ties resolve to the lowest row: a ballot over each step finds it, and the claim passes into
 *  the document's later rows, since a document never crosses a band. Wide steps inside the open
 *  document move eight rows per lane, as in @ref contrastive_tokens_band_max_kernel_ .
 *
 *  __grid_size__ @c [ceil(query_tokens/8),1,1] ; one warp per query row, 8 per CTA; no
 *  dynamic shared memory.
 */
static __global__ void contrastive_tokens_band_grad_kernel_(
    util::f32r2_view_t band_products, util::u32r1_view_t query_segments, util::u32r1_view_t query_offsets,
    util::u32r1_view_t document_of_row, util::u32r1_view_t document_offsets, util::u32r1_view_t document_segments,
    util::f32r2_view_t score_grad, std::size_t band_begin, util::f32r2_view_t maxima, util::bf16r2_span_t band_grad) {
    util::u32_t const lane = threadIdx.x % util::warp_size_k;
    std::size_t const query_row = static_cast<std::size_t>(blockIdx.x) * (blockDim.x / util::warp_size_k) +
                                  threadIdx.x / util::warp_size_k;
    if (query_row >= band_products.extent(0)) return;
    util::u32_t const query = query_segments[query_row];
    std::size_t const live_rows = query != ~0u ? document_offsets[document_offsets.extent(0) - 1u] : 0u;
    std::size_t const band_rows = band_products.extent(1);
    util::f32_t const inverse_length =
        query != ~0u ? 1.0f / static_cast<util::f32_t>(query_offsets[query + 1u] - query_offsets[query]) : 0.0f;
    util::f32_t const* const products = &band_products[query_row, 0];
    util::bf16_t* const gradients = &band_grad[query_row, 0];
    util::u32_t open_document = ~0u;
    util::f32_t open_maximum = 0.0f, open_gradient = 0.0f;
    bool open_claimed = false;
    auto const open = [&](util::u32_t document) noexcept {
        open_document = document;
        open_maximum = document != ~0u ? maxima[query_row, document] : 0.0f;
        open_gradient = document != ~0u ? score_grad[query, document_segments[document]] * inverse_length : 0.0f;
    };
    auto const credit_chunk = [&](std::size_t chunk) noexcept {
        util::u32_t const document = contrastive_tokens_document_(document_of_row, band_begin + chunk + lane,
                                                                  live_rows);
        util::f32_t const value = products[chunk + lane];
        if (__all_sync(0xffffffffu, document == open_document)) {
            util::u32_t const matching_lanes = __ballot_sync(0xffffffffu, document != ~0u && value == open_maximum);
            bool const claims = !open_claimed && matching_lanes != 0u &&
                                lane == util::cuda_warp_t::first(matching_lanes);
            open_claimed = open_claimed || matching_lanes != 0u;
            gradients[chunk + lane] = __float2bfloat16(claims ? open_gradient : 0.0f);
            return;
        }
        bool const matches = document != ~0u && value == maxima[query_row, document];
        util::u32_t const matching_lanes = __ballot_sync(0xffffffffu, matches);
        contrastive_tokens_lane_t const position = contrastive_tokens_lane_(document, lane);
        util::u32_t const run_lanes = ~((1u << position.run_begin_lane) - 1u);
        bool const claimed_before = open_claimed && document == open_document;
        bool const claims = matches && (matching_lanes & ((1u << lane) - 1u) & run_lanes) == 0u && !claimed_before;
        util::u32_t const last_document = __shfl_sync(0xffffffffu, document, util::warp_size_k - 1u);
        util::u32_t const last_run_lanes = __shfl_sync(0xffffffffu, run_lanes, util::warp_size_k - 1u);
        open_claimed = (__ballot_sync(0xffffffffu, claims) & last_run_lanes) != 0u;
        open(last_document);
        gradients[chunk + lane] = __float2bfloat16(
            claims ? score_grad[query, document_segments[document]] * inverse_length : 0.0f);
    };
    for (std::size_t step = 0; step < band_rows; step += contrastive_tokens_wide_rows_k) {
        std::size_t const step_end = step + contrastive_tokens_wide_rows_k;
        bool const inside_open = step_end <= band_rows && open_document != ~0u &&
                                 contrastive_tokens_document_(document_of_row, band_begin + step_end - 1u, live_rows) ==
                                     open_document;
        if (inside_open) {
            float4 const low = contrastive_tokens_load_(products, step + 8u * lane);
            float4 const high = contrastive_tokens_load_(products, step + 8u * lane + 4u);
            util::f32_t const values[8] = {low.x, low.y, low.z, low.w, high.x, high.y, high.z, high.w};
            util::u32_t first = 8u;
            for (util::u32_t index = 8u; index != 0u; --index)
                if (values[index - 1u] == open_maximum) first = index - 1u;
            util::u32_t const matching_lanes = __ballot_sync(0xffffffffu, first != 8u);
            bool const claims = !open_claimed && matching_lanes != 0u &&
                                lane == util::cuda_warp_t::first(matching_lanes);
            open_claimed = open_claimed || matching_lanes != 0u;
            auto const credit = [&](util::u32_t index) noexcept {
                return claims && first == index ? open_gradient : 0.0f;
            };
            contrastive_tokens_store_(gradients, step + 8u * lane, float4 {credit(0), credit(1), credit(2), credit(3)},
                                      float4 {credit(4), credit(5), credit(6), credit(7)});
            continue;
        }
        for (std::size_t chunk = step; chunk < step_end && chunk < band_rows; chunk += util::warp_size_k)
            credit_chunk(chunk);
    }
}

/**
 *  @brief The token head's gradient: each row's query force and document force by its segment's
 *      sides, in that order, and zero past the live rows.
 *
 *  __grid_size__ @c [head_tokens,1,1] ; one CTA per row; threads stripe the width; no
 *  dynamic shared memory.
 */
static __global__ void contrastive_tokens_finish_kernel_(
    multimodal_batch_view_t batch, util::u32r1_view_t token_offsets, util::u32r1_view_t query_offsets,
    util::f32r2_view_t query_grad, util::u32r1_view_t document_of_segment, util::u32r1_view_t document_offsets,
    util::f32r2_view_t document_grad, util::bf16r2_span_t token_vectors_grad) {
    util::u32_t const row = blockIdx.x;
    std::size_t const segments = token_offsets.extent(0) - 1u;
    bool const live = row < token_offsets[segments];
    util::u32_t const segment = live ? contrastive_tokens_owner_(token_offsets, row) : 0u;
    util::u32_t const offset = live ? row - token_offsets[segment] : 0u;
    late_interaction_side_t const side = live ? batch.segment_side[segment] : late_interaction_side_t::none_k;
    bool const queried = contrastive_tokens_queries_(side) &&
                         offset < query_offsets[segment + 1u] - query_offsets[segment];
    std::size_t const query_row = queried ? query_offsets[segment] + offset : ~std::size_t {0};
    util::u32_t const document = contrastive_tokens_documents_(side) ? document_of_segment[segment] : ~0u;
    std::size_t const document_row = document != ~0u ? document_offsets[document] + offset : ~std::size_t {0};
    for (std::size_t channel = threadIdx.x; channel < token_vectors_grad.extent(1); channel += blockDim.x) {
        util::f32_t gradient = 0.0f;
        if (query_row < query_grad.extent(0)) gradient += query_grad[query_row, channel];
        if (document_row < document_grad.extent(0)) gradient += document_grad[document_row, channel];
        token_vectors_grad[row, channel] = __float2bfloat16(gradient);
    }
}

/**
 *  @brief Scale the home score rows by @p weight times their query's β in place: raw scores
 *      into the pair loss's logits before it, and its logit gradient into the weighted score
 *      gradient after it.
 *
 *  __grid_size__ grid-stride over the @b [subgraphs,segments,segments] cells; no
 *  dynamic shared memory.
 */
static __global__ void contrastive_tokens_scale_kernel_(util::f32r3_span_t scores, contrastive_recipe_t recipe,
                                                        util::f32_t weight) {
    std::size_t const rows = scores.extent(1), columns = scores.extent(2);
    std::size_t const cells = scores.extent(0) * rows * columns;
    std::size_t const stride = static_cast<std::size_t>(gridDim.x) * blockDim.x;
    for (std::size_t cell = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x; cell < cells;
         cell += stride) {
        std::size_t const row = (cell / columns) % rows;
        scores[cell / (rows * columns), row, cell % columns] *= weight * contrastive_from_vertex_beta_(
                                                                             recipe, static_cast<util::u32_t>(row));
    }
}

/** Scale the loss summand and the temperature-gradient summands by the loss weight @p weight .
 *
 *  __grid_size__ @c [1,1,1] ; threads stripe the temperature groups; no dynamic shared memory. */
static __global__ void contrastive_tokens_weigh_kernel_(util::f32r1_span_t loss_out,
                                                        util::f32r1_span_t log_inverse_temperature_grad,
                                                        util::f32r1_span_t bias_grad, util::f32_t weight) {
    if (threadIdx.x == 0u) loss_out[0] *= weight;
    for (std::size_t group = threadIdx.x; group < log_inverse_temperature_grad.extent(0); group += blockDim.x)
        log_inverse_temperature_grad[group] *= weight;
    for (std::size_t group = threadIdx.x; group < bias_grad.extent(0); group += blockDim.x) bias_grad[group] *= weight;
}

util::expected<contrastive_tokens_t> contrastive_tokens_t::make(util::gpu_arena_t& arena,
                                                                contrastive_tokens_config_t config,
                                                                util::gpu_numerics_t const& numerics) noexcept {
    using result_t = util::expected<contrastive_tokens_t>;
    if (config.segments_per_gpu == 0 || config.head_tokens_per_gpu == 0 || config.query_tokens_per_gpu == 0 ||
        config.documents_per_gpu == 0 || config.band_tokens == 0 || config.gpu_subgraphs_per_step == 0)
        return result_t::fail(util::status_code_t::invalid_argument_k, "Tokens loss needs every size nonzero");
    if (config.width % 8u != 0u || config.query_tokens_per_gpu % 8u != 0u ||
        config.band_tokens % util::warp_size_k != 0u)
        return result_t::fail(util::status_code_t::unsupported_k,
                              "width and query rows must be multiples of 8 and the band of {} (width={} queries={} " //
                              "band={})",
                              util::warp_size_k, config.width, config.query_tokens_per_gpu, config.band_tokens);
    config.document_tokens_per_gpu = static_cast<util::u32_t>(
        util::round_up_to_multiple(fu::max_of(config.document_tokens_per_gpu, config.band_tokens), config.band_tokens));

    contrastive_tokens_t loss;
    loss.config_ = config;
    loss.numerics_ = numerics;
    auto [pair_loss, make_status] = contrastive_pair_loss_t::make(
        arena, config.segments_per_gpu, config.segments_per_gpu * config.gpu_subgraphs_per_step, config.pair_loss);
    if (!make_status) return result_t::fail(std::move(make_status));
    loss.pair_loss_ = std::move(pair_loss);

    std::size_t const segments = config.segments_per_gpu, width = config.width;
    std::size_t const queries = config.query_tokens_per_gpu, documents = config.documents_per_gpu;
    std::size_t const document_rows = config.document_tokens_per_gpu, band = config.band_tokens;
    std::size_t const bands = document_rows / band;
    util::gpu_arena_alloc<util::bf16_t> bf16_alloc {arena};
    util::gpu_arena_alloc<util::f32_t> f32_alloc {arena};
    util::gpu_arena_alloc<util::u32_t> u32_alloc {arena};
    util::gpu_arena_alloc<util::b8_t> byte_alloc {arena};
    bool const tiles_allocated =
        util::allocate_uninitialized(loss.query_tile_, {queries, width}, bf16_alloc) &&
        util::allocate_uninitialized(loss.query_segments_, {queries}, u32_alloc) &&
        util::allocate_uninitialized(loss.query_offsets_, {segments + 1u}, u32_alloc) &&
        util::allocate_uninitialized(loss.query_lengths_, {segments}, u32_alloc) &&
        util::allocate_uninitialized(loss.document_tile_, {document_rows, width}, bf16_alloc) &&
        util::allocate_uninitialized(loss.document_of_row_, {document_rows}, u32_alloc) &&
        util::allocate_uninitialized(loss.document_offsets_, {documents + 1u}, u32_alloc) &&
        util::allocate_uninitialized(loss.document_segments_, {documents}, u32_alloc) &&
        util::allocate_uninitialized(loss.document_lengths_, {segments}, u32_alloc) &&
        util::allocate_uninitialized(loss.document_of_segment_, {segments}, u32_alloc) &&
        util::allocate_uninitialized(loss.live_queries_, {1}, f32_alloc) &&
        util::allocate_uninitialized(loss.overflow_, {1}, u32_alloc) &&
        util::allocate_uninitialized(loss.query_tile_transposed_, {width, queries}, bf16_alloc) &&
        util::allocate_uninitialized(loss.document_tile_transposed_, {width, document_rows}, bf16_alloc) &&
        util::allocate_uninitialized(loss.maxima_, {config.gpu_subgraphs_per_step, queries, documents}, f32_alloc) &&
        util::allocate_uninitialized(loss.band_products_, {queries, band}, f32_alloc) &&
        util::allocate_uninitialized(loss.band_grad_, {queries, band}, bf16_alloc) &&
        util::allocate_uninitialized(loss.band_grad_transposed_, {band, queries}, bf16_alloc) &&
        util::allocate_uninitialized(loss.query_grad_products_, {queries, width}, f32_alloc) &&
        util::allocate_uninitialized(loss.document_grad_products_, {band, width}, f32_alloc) &&
        util::allocate_uninitialized(loss.document_grad_, {document_rows, width}, f32_alloc) &&
        util::allocate_uninitialized(loss.query_grad_, {queries, width}, f32_alloc) &&
        util::allocate_uninitialized(loss.scores_, {segments * segments}, f32_alloc);
    if (!tiles_allocated)
        return result_t::fail(util::status_code_t::out_of_memory_k, "tokens loss allocation failed (arena_used={})",
                              arena.used());

    auto [band_bytes, measure_band_bytes_status] = util::pack_bytes_bf16(band, width, loss.numerics_);
    if (!measure_band_bytes_status) return result_t::fail(std::move(measure_band_bytes_status));
    auto [band_transposed_bytes, measure_band_transposed_bytes_status] = util::pack_bytes_bf16(width, band,
                                                                                               loss.numerics_);
    if (!measure_band_transposed_bytes_status) return result_t::fail(std::move(measure_band_transposed_bytes_status));
    auto [queries_transposed_bytes, measure_queries_transposed_bytes_status] = util::pack_bytes_bf16(width, queries,
                                                                                                     loss.numerics_);
    if (!measure_queries_transposed_bytes_status)
        return result_t::fail(std::move(measure_queries_transposed_bytes_status));
    loss.band_pack_bytes_ = band_bytes;
    loss.band_pack_transposed_bytes_ = band_transposed_bytes;
    bool const packs_allocated =
        util::allocate_uninitialized(loss.packed_documents_, {bands * loss.band_pack_bytes_}, byte_alloc) &&
        util::allocate_uninitialized(loss.packed_documents_transposed_, {bands * loss.band_pack_transposed_bytes_},
                                     byte_alloc) &&
        util::allocate_uninitialized(loss.packed_queries_transposed_, {queries_transposed_bytes}, byte_alloc);
    if (!packs_allocated)
        return result_t::fail(util::status_code_t::out_of_memory_k, "tokens loss pack allocation failed ({} bands)",
                              bands);
    return result_t {std::move(loss)};
}

util::expected_t contrastive_tokens_t::stage(util::bf16r2_view_t token_vectors, util::u32r1_view_t token_offsets,
                                             multimodal_batch_view_t const& batch,
                                             util::executor_stream_t& stream) noexcept {
    if (token_vectors.extent(0) != config_.head_tokens_per_gpu || token_vectors.extent(1) != config_.width ||
        token_offsets.extent(0) != config_.segments_per_gpu + 1u)
        return util::expected_t::fail(
            util::status_code_t::unsupported_k,
            "token head output [{}x{}] over {} offsets disagrees with the baked [{}x{}] over {}",
            token_vectors.extent(0), token_vectors.extent(1), token_offsets.extent(0), config_.head_tokens_per_gpu,
            config_.width, config_.segments_per_gpu + 1u);
    dim3 const block_shape {util::kernel_block_size_default_k, 1u, 1u};
    std::size_t const band = config_.band_tokens;
    util::expected_t staged = util::launch_kernel(
        &contrastive_tokens_layout_kernel_, dim3 {1u, 1u, 1u}, block_shape, stream, batch, token_offsets,
        query_offsets_.span(), query_lengths_.span(), document_offsets_.span(), document_segments_.span(),
        document_lengths_.span(), document_of_segment_.span(), live_queries_.span(), overflow_.span(),
        config_.query_tokens_per_gpu, config_.document_tokens_per_gpu, config_.band_tokens, config_.query_span_tokens);
    if (staged)
        staged = util::launch_kernel(&contrastive_tokens_gather_queries_kernel_,
                                     dim3 {config_.query_tokens_per_gpu, 1u, 1u}, block_shape, stream, token_vectors,
                                     token_offsets, query_offsets_.view(), query_tile_.span(), query_segments_.span());
    if (staged)
        staged = util::launch_kernel(&contrastive_tokens_gather_documents_kernel_,
                                     dim3 {config_.document_tokens_per_gpu, 1u, 1u}, block_shape, stream, token_vectors,
                                     token_offsets, document_offsets_.view(), document_segments_.view(),
                                     document_tile_.span(), document_of_row_.span());
    if (staged) staged = util::transpose_bf16(document_tile_.view(), document_tile_transposed_.span(), stream);
    for (std::size_t band_index = 0; staged && band_index != bands_(); ++band_index) {
        auto const rows = nk::range(band_index * band, (band_index + 1u) * band);
        staged = util::pack_bf16(document_tile_[rows, nk::slice],
                                 packed_band_(packed_documents_, band_pack_bytes_, band_index), numerics_, stream);
        if (staged)
            staged = util::pack_bf16(
                document_tile_transposed_[nk::all, rows, nk::slice],
                packed_band_(packed_documents_transposed_, band_pack_transposed_bytes_, band_index), numerics_, stream);
    }
    if (staged) staged = util::zero_async(document_grad_.span(), stream);
    return staged;
}

util::expected_t contrastive_tokens_t::score(std::size_t subgraph_index, gpu_query_subgraph_t const& subgraph,
                                             util::f32r2_span_t raw_scores, util::executor_stream_t& stream) noexcept {
    if (subgraph_index >= config_.gpu_subgraphs_per_step ||
        subgraph.query_vectors.extent(0) != config_.query_tokens_per_gpu)
        return util::expected_t::fail(util::status_code_t::unsupported_k,
                                      "query subgraph {} disagrees with the baked shape", subgraph_index);
    std::size_t const band = config_.band_tokens;
    util::f32r2_span_t const maxima = maxima_[subgraph_index, nk::slice];
    dim3 const block_shape {util::kernel_block_size_default_k, 1u, 1u};
    dim3 const row_warps_grid {static_cast<util::u32_t>(util::divide_round_up(
                                   std::size_t {config_.query_tokens_per_gpu},
                                   std::size_t {util::kernel_block_size_default_k / util::warp_size_k})),
                               1u, 1u};
    util::expected_t staged = util::zero_async(raw_scores, stream);
    for (std::size_t band_index = 0; staged && band_index != bands_(); ++band_index) {
        staged = util::multiply_bf16(subgraph.query_vectors,
                                     packed_band_(packed_documents_, band_pack_bytes_, band_index),
                                     band_products_.span(), numerics_, stream);
        if (staged)
            staged = util::launch_kernel(&contrastive_tokens_band_max_kernel_, row_warps_grid, block_shape, stream,
                                         band_products_.view(), subgraph.query_segments, document_of_row_.view(),
                                         document_offsets_.view(), band_index * band, maxima);
    }
    if (staged)
        staged = util::launch_kernel(&contrastive_tokens_scores_kernel_, dim3 {config_.segments_per_gpu, 1u, 1u},
                                     block_shape, stream, util::f32r2_view_t {maxima}, subgraph.query_offsets,
                                     document_segments_.view(), raw_scores);
    return staged;
}

util::expected_t contrastive_tokens_t::force(std::size_t subgraph_index, gpu_query_subgraph_t const& subgraph,
                                             util::f32r2_view_t score_grad, util::f32r2_span_t query_grad,
                                             util::executor_stream_t& stream) noexcept {
    std::size_t const band = config_.band_tokens, queries = config_.query_tokens_per_gpu, width = config_.width;
    util::f32r2_span_t const maxima = maxima_[subgraph_index, nk::slice];
    dim3 const block_shape {util::kernel_block_size_default_k, 1u, 1u};
    dim3 const row_warps_grid {static_cast<util::u32_t>(util::divide_round_up(
                                   queries, std::size_t {util::kernel_block_size_default_k / util::warp_size_k})),
                               1u, 1u};
    util::expected_t staged = util::transpose_bf16(subgraph.query_vectors, query_tile_transposed_.span(), stream);
    if (staged)
        staged = util::pack_bf16(query_tile_transposed_.view(), packed_queries_transposed_.span(), numerics_, stream);
    if (staged) staged = util::zero_async(query_grad, stream);
    for (std::size_t band_index = 0; staged && band_index != bands_(); ++band_index) {
        staged = util::multiply_bf16(subgraph.query_vectors,
                                     packed_band_(packed_documents_, band_pack_bytes_, band_index),
                                     band_products_.span(), numerics_, stream);
        if (staged)
            staged = util::launch_kernel(&contrastive_tokens_band_grad_kernel_, row_warps_grid, block_shape, stream,
                                         band_products_.view(), subgraph.query_segments, subgraph.query_offsets,
                                         document_of_row_.view(), document_offsets_.view(), document_segments_.view(),
                                         score_grad, band_index * band, util::f32r2_view_t {maxima}, band_grad_.span());
        if (staged)
            staged = util::multiply_bf16(
                band_grad_.view(), packed_band_(packed_documents_transposed_, band_pack_transposed_bytes_, band_index),
                query_grad_products_.span(), numerics_, stream);
        if (staged)
            staged = util::numkong_result(
                numerics_.sum_f32(query_grad.data(), query_grad_products_.data(), queries * width, query_grad.data(),
                                  util::numkong_stream(stream)),
                "query force addition");
        if (staged) staged = util::transpose_bf16(band_grad_.view(), band_grad_transposed_.span(), stream);
        if (staged)
            staged = util::multiply_bf16(band_grad_transposed_.view(), packed_queries_transposed_.view(),
                                         document_grad_products_.span(), numerics_, stream);
        if (staged) {
            util::f32_t* const band_force = document_grad_.data() + band_index * band * width;
            staged = util::numkong_result(numerics_.sum_f32(band_force, document_grad_products_.data(), band * width,
                                                            band_force, util::numkong_stream(stream)),
                                          "document force addition");
        }
    }
    return staged;
}

util::expected_t contrastive_tokens_t::relax(util::f32r4_span_t scores, contrastive_relations_view_t relations,
                                             contrastive_recipe_t recipe, util::u32r1_view_t world_document_lengths,
                                             util::f32r1_view_t live_queries, util::f32r1_span_t loss_out,
                                             util::executor_stream_t& stream,
                                             util::f32r1_span_t log_inverse_temperature_grad,
                                             util::f32r1_span_t bias_grad) noexcept {
    relations.from_vertex_effective_length = query_lengths_.view();
    relations.to_vertex_effective_length = world_document_lengths;
    util::f32r3_span_t const planes = scores[0, nk::slice];
    dim3 const grid {util::grid_size_for(planes.numel()), 1u, 1u};
    dim3 const block_shape {util::kernel_block_size_default_k, 1u, 1u};
    util::expected_t staged = util::launch_kernel(&contrastive_tokens_scale_kernel_, grid, block_shape, stream, planes,
                                                  recipe, 1.0f);
    if (staged)
        staged = pair_loss_(scores, relations, recipe, live_queries, loss_out, stream, log_inverse_temperature_grad,
                            bias_grad);
    if (staged)
        staged = util::launch_kernel(&contrastive_tokens_scale_kernel_, grid, block_shape, stream, planes, recipe,
                                     config_.weight);
    if (staged && config_.weight != 1.0f)
        staged = util::launch_kernel(&contrastive_tokens_weigh_kernel_, dim3 {1u, 1u, 1u}, block_shape, stream,
                                     loss_out, log_inverse_temperature_grad, bias_grad, config_.weight);
    return staged;
}

util::expected_t contrastive_tokens_t::finish(util::f32r2_view_t query_grad, util::u32r1_view_t token_offsets,
                                              multimodal_batch_view_t const& batch,
                                              util::bf16r2_span_t token_vectors_grad,
                                              util::executor_stream_t& stream) noexcept {
    if (token_vectors_grad.extent(0) != config_.head_tokens_per_gpu || token_vectors_grad.extent(1) != config_.width)
        return util::expected_t::fail(util::status_code_t::unsupported_k, "token_vectors_grad must be [{}x{}]",
                                      config_.head_tokens_per_gpu, config_.width);
    return util::launch_kernel(&contrastive_tokens_finish_kernel_, dim3 {config_.head_tokens_per_gpu, 1u, 1u},
                               dim3 {util::kernel_block_size_default_k, 1u, 1u}, stream, batch, token_offsets,
                               query_offsets_.view(), query_grad, document_of_segment_.view(), document_offsets_.view(),
                               document_grad_.view(), token_vectors_grad);
}

util::expected_t contrastive_tokens_t::operator()(util::bf16r2_view_t token_vectors, util::u32r1_view_t token_offsets,
                                                  multimodal_batch_view_t const& batch,
                                                  contrastive_relations_view_t relations, contrastive_recipe_t recipe,
                                                  util::f32r1_span_t loss_out, util::bf16r2_span_t token_vectors_grad,
                                                  util::executor_stream_t& stream,
                                                  util::f32r1_span_t log_inverse_temperature_grad,
                                                  util::f32r1_span_t bias_grad) noexcept {
    if (config_.gpu_subgraphs_per_step != 1)
        return util::expected_t::fail(util::status_code_t::invalid_argument_k,
                                      "a {}-subgraph tokens loss runs distributed", config_.gpu_subgraphs_per_step);
    std::size_t const segments = config_.segments_per_gpu;
    util::f32r2_span_t const raw_scores = scores_.span().reshape<2>({segments, segments});
    util::expected_t staged = stage(token_vectors, token_offsets, batch, stream);
    if (staged) staged = score(0, query_subgraph(), raw_scores, stream);
    if (staged)
        staged = relax(scores_.span().reshape<4>({1u, 1u, segments, segments}), relations, recipe,
                       document_lengths_.view(), live_queries_.view(), loss_out, stream, log_inverse_temperature_grad,
                       bias_grad);
    if (staged) staged = force(0, query_subgraph(), util::f32r2_view_t {raw_scores}, query_grad_.span(), stream);
    if (staged) staged = finish(query_grad_.view(), token_offsets, batch, token_vectors_grad, stream);
    return staged;
}

util::expected<distributed_contrastive_tokens_t> distributed_contrastive_tokens_t::make(
    util::multimem_domain_t& domain, util::colocated_domain_t& colocated, std::span<util::gpu_arena_t> arenas,
    util::cuda_executor_t& streams, contrastive_tokens_config_t config, util::gpu_numerics_t const& numerics,
    util::f64_t timeout_seconds) noexcept {
    using result_t = util::expected<distributed_contrastive_tokens_t>;
    std::size_t const gpus = arenas.size();
    util::u32_t const domains = colocated.multimem_domain_count();
    if (gpus == 0 || streams.size() != gpus || domain.local_member_count() != gpus)
        return result_t::fail(util::status_code_t::invalid_argument_k, "Every rank needs arena, stream, local member");
    if (domain.member_count() != gpus || config.gpu_subgraphs_per_step != gpus * domains)
        return result_t::fail(util::status_code_t::unsupported_k,
                              "the tokens loss scores one query tile per GPU, every domain held whole by one " //
                              "process (subgraphs={} members={} local={} domains={})",
                              config.gpu_subgraphs_per_step, domain.member_count(), gpus, domains);
    for (std::size_t rank = 0; rank != gpus; ++rank)
        for (std::size_t peer = 0; peer != gpus; ++peer)
            if (!arenas[rank].is_visible_to(static_cast<int>(peer)))
                return result_t::fail(util::status_code_t::invalid_argument_k,
                                      "rank {}'s arena is not visible to device {}, construct it with `visible_to`",
                                      rank, peer);

    distributed_contrastive_tokens_t distributed;
    distributed.config_ = config;
    distributed.domain_ = &domain;
    distributed.colocated_ = &colocated;
    distributed.home_domain_index_ = domains > 1u ? colocated.topology().multimem_domain_index() : 0u;
    auto [window, reserve_status] = domain.reserve(payload_scalars_needed(config));
    if (!reserve_status) return result_t::fail(std::move(reserve_status));
    distributed.window_ = window;
    if (distributed.losses_.resize(gpus) != util::status_code_t::success_k ||
        distributed.home_scores_.resize(gpus) != util::status_code_t::success_k ||
        distributed.world_document_lengths_.resize(gpus) != util::status_code_t::success_k ||
        distributed.query_grads_.resize(gpus) != util::status_code_t::success_k ||
        distributed.world_scalars_.resize(gpus) != util::status_code_t::success_k)
        return result_t::fail(util::status_code_t::out_of_memory_k, "rank table allocation failed ({} ranks)", gpus);
    if (domains > 1u && (distributed.tile_gathers_.resize(gpus) != util::status_code_t::success_k ||
                         distributed.description_gathers_.resize(gpus) != util::status_code_t::success_k ||
                         distributed.score_gathers_.resize(gpus) != util::status_code_t::success_k ||
                         distributed.gradient_gathers_.resize(gpus) != util::status_code_t::success_k ||
                         distributed.force_reduces_.resize(gpus) != util::status_code_t::success_k ||
                         distributed.count_reduces_.resize(gpus) != util::status_code_t::success_k))
        return result_t::fail(util::status_code_t::out_of_memory_k,
                              "cross-domain member table allocation failed ({} ranks)", gpus);
    std::size_t const subgraphs = config.gpu_subgraphs_per_step, segments = config.segments_per_gpu;
    std::size_t const tile_scalars = std::size_t {config.query_tokens_per_gpu} * config.width;
    std::size_t const score_scalars = subgraphs * segments * segments;
    for (std::size_t rank = 0; rank != gpus; ++rank) {
        [[maybe_unused]] util::status_t const gpu_scope = (streams.stream(rank)).bind();
        if (gpu_scope.failed())
            return result_t::fail(util::status_code_t::device_k, "context entry failed (rank={})", rank);
        auto [loss, make_loss_status] = contrastive_tokens_t::make(arenas[rank], config, numerics);
        if (!make_loss_status) return result_t::fail(std::move(make_loss_status));
        distributed.losses_[rank] = std::move(loss);
        util::gpu_arena_alloc<util::f32_t> f32_alloc {arenas[rank]};
        bool const scratch_allocated =
            util::allocate_uninitialized(distributed.home_scores_[rank], {score_scalars}, f32_alloc) &&
            util::allocate_uninitialized(distributed.world_document_lengths_[rank], {subgraphs * segments},
                                         util::gpu_arena_alloc<util::u32_t> {arenas[rank]}) &&
            util::allocate_uninitialized(distributed.query_grads_[rank], {config.query_tokens_per_gpu, config.width},
                                         f32_alloc) &&
            util::allocate_uninitialized(distributed.world_scalars_[rank], {world_scalar_count_(config)}, f32_alloc);
        if (!scratch_allocated)
            return result_t::fail(util::status_code_t::out_of_memory_k,
                                  "tokens rank scratch allocation failed (rank={})", rank);
        if (domains == 1u) continue;
        // Every process wires its members in the same order, as each handle keys
        // its own rendezvous.
        util::u32_t const member = static_cast<util::u32_t>(domain.member_offset() + rank);
        auto [tiles, make_tiles_status] = util::infiniband_all_gather<util::bf16_t>::make(
            colocated, member, arenas[rank], tile_scalars, streams.stream(rank), timeout_seconds);
        if (!make_tiles_status) return result_t::fail(std::move(make_tiles_status));
        distributed.tile_gathers_[rank] = std::move(tiles);
        auto [descriptions, make_descriptions_status] = util::infiniband_all_gather<util::u32_t>::make(
            colocated, member, arenas[rank], distributed.description_scalars_(), streams.stream(rank), timeout_seconds);
        if (!make_descriptions_status) return result_t::fail(std::move(make_descriptions_status));
        distributed.description_gathers_[rank] = std::move(descriptions);
        auto [scores, make_scores_status] = util::infiniband_all_gather<util::f32_t>::make(
            colocated, member, arenas[rank], score_scalars, streams.stream(rank), timeout_seconds);
        if (!make_scores_status) return result_t::fail(std::move(make_scores_status));
        distributed.score_gathers_[rank] = std::move(scores);
        auto [gradients, make_gradients_status] = util::infiniband_all_gather<util::f32_t>::make(
            colocated, member, arenas[rank], score_scalars, streams.stream(rank), timeout_seconds);
        if (!make_gradients_status) return result_t::fail(std::move(make_gradients_status));
        distributed.gradient_gathers_[rank] = std::move(gradients);
        auto [forces, make_forces_status] = util::infiniband_all_reduce<util::f32_t>::make(
            colocated, member, arenas[rank], domains * tile_scalars + world_scalar_count_(config), streams.stream(rank),
            timeout_seconds);
        if (!make_forces_status) return result_t::fail(std::move(make_forces_status));
        distributed.force_reduces_[rank] = std::move(forces);
        auto [count, make_count_status] = util::infiniband_all_reduce<util::f32_t>::make(
            colocated, member, arenas[rank], 1u, streams.stream(rank), timeout_seconds);
        if (!make_count_status) return result_t::fail(std::move(make_count_status));
        distributed.count_reduces_[rank] = std::move(count);
    }
    return result_t {std::move(distributed)};
}

util::f32r3_span_t distributed_contrastive_tokens_t::window_scores_(std::size_t member, std::size_t block) noexcept {
    std::size_t const subgraphs = config_.gpu_subgraphs_per_step, segments = config_.segments_per_gpu;
    std::size_t const scalars = subgraphs * segments * segments;
    util::f32r1_span_t const window = window_payload_(member);
    util::f32r2_span_t const blocks = window[nk::range(0, 2u * scalars), nk::slice].reshape<2>({2u, scalars});
    return blocks[block, nk::slice].reshape<3>({subgraphs, segments, segments});
}

util::f32r3_span_t distributed_contrastive_tokens_t::window_query_forces_(std::size_t member) noexcept {
    std::size_t const subgraphs = config_.gpu_subgraphs_per_step, segments = config_.segments_per_gpu;
    std::size_t const begin = 2u * subgraphs * segments * segments;
    std::size_t const scalars = subgraphs * std::size_t {config_.query_tokens_per_gpu} * config_.width;
    util::f32r1_span_t const window = window_payload_(member);
    return window[nk::range(begin, begin + scalars), nk::slice].reshape<3>(
        {subgraphs, std::size_t {config_.query_tokens_per_gpu}, std::size_t {config_.width}});
}

util::f32r1_span_t distributed_contrastive_tokens_t::window_tail_(std::size_t member) noexcept {
    std::size_t const tail_begin = window_.scalars - world_scalar_count_(config_);
    util::f32r1_span_t const window = window_payload_(member);
    return window[nk::range(tail_begin, window_.scalars), nk::slice];
}

gpu_query_subgraph_t distributed_contrastive_tokens_t::world_subgraph_(std::size_t world_gpu) noexcept {
    std::size_t const members = losses_.size(), member = world_gpu % members, domain = world_gpu / members;
    if (domain == home_domain_index_) return losses_[member].query_subgraph();
    std::size_t const queries = config_.query_tokens_per_gpu, segments = config_.segments_per_gpu;
    std::size_t const tile_scalars = queries * config_.width, description = description_scalars_();
    util::bf16r2_span_t const tiles = tile_gathers_[member].payload().reshape<2>({world_domains_(), tile_scalars});
    util::u32r2_span_t const descriptions = description_gathers_[member].payload().reshape<2>(
        {world_domains_(), description});
    util::u32r1_span_t const described = descriptions[domain, nk::slice];
    gpu_query_subgraph_t subgraph;
    subgraph.query_vectors = tiles[domain, nk::slice].reshape<2>({queries, std::size_t {config_.width}});
    subgraph.query_segments = described[nk::range(0, queries), nk::slice];
    subgraph.query_offsets = described[nk::range(queries, queries + segments + 1u), nk::slice];
    return subgraph;
}

util::u32r1_view_t distributed_contrastive_tokens_t::world_document_lengths_of_(std::size_t world_gpu) noexcept {
    std::size_t const members = losses_.size(), member = world_gpu % members, domain = world_gpu / members;
    if (domain == home_domain_index_) return losses_[member].document_lengths();
    std::size_t const segments = config_.segments_per_gpu;
    std::size_t const lengths_begin = std::size_t {config_.query_tokens_per_gpu} + segments + 1u;
    util::u32r2_span_t const descriptions = description_gathers_[member].payload().reshape<2>(
        {world_domains_(), description_scalars_()});
    util::u32r1_span_t const described = descriptions[domain, nk::slice];
    return described[nk::range(lengths_begin, lengths_begin + segments), nk::slice];
}

util::f32r3_view_t distributed_contrastive_tokens_t::world_scores_(std::size_t world_gpu, std::size_t block) noexcept {
    std::size_t const members = losses_.size(), member = world_gpu % members, domain = world_gpu / members;
    if (domain == home_domain_index_) return window_scores_(domain_->member_offset() + member, block);
    std::size_t const subgraphs = config_.gpu_subgraphs_per_step, segments = config_.segments_per_gpu;
    util::infiniband_all_gather<util::f32_t>& gathered = block == 0u ? score_gathers_[member]
                                                                     : gradient_gathers_[member];
    util::f32r2_span_t const blocks = gathered.payload().reshape<2>(
        {world_domains_(), subgraphs * segments * segments});
    return blocks[domain, nk::slice].reshape<3>({subgraphs, segments, segments});
}

util::expected_t distributed_contrastive_tokens_t::operator()(std::span<gpu_tokens_inputs_t const> inputs,
                                                              util::cuda_executor_t& streams) noexcept {
    std::size_t const gpus = losses_.size();
    if (inputs.size() != gpus || streams.size() != gpus)
        return util::expected_t::fail(util::status_code_t::invalid_argument_k,
                                      "every rank needs its inputs and stream ({} ranks)", gpus);
    for (gpu_tokens_inputs_t const& rank_inputs : inputs)
        if (rank_inputs.recipe.temperature_group_count() > config_.temperature_groups)
            return util::expected_t::fail(util::status_code_t::unsupported_k,
                                          "a recipe selects {} temperature groups of {}",
                                          rank_inputs.recipe.temperature_group_count(), config_.temperature_groups);
    // Phase-major: every rank's phase is enqueued before any rank's next one, so a rank spinning at
    // a barrier never holds the host while its peers' arrivals wait unenqueued behind it.
    using phase_t = util::expected_t (distributed_contrastive_tokens_t::*)(std::size_t, gpu_tokens_inputs_t const&,
                                                                           util::executor_stream_t&) noexcept;
    phase_t const phases[] = {&distributed_contrastive_tokens_t::stage_, &distributed_contrastive_tokens_t::score_,
                              &distributed_contrastive_tokens_t::relax_, &distributed_contrastive_tokens_t::force_,
                              &distributed_contrastive_tokens_t::reduce_};
    for (phase_t const phase : phases)
        for (std::size_t rank = 0; rank != gpus; ++rank) {
            [[maybe_unused]] util::status_t const gpu_scope = (streams.stream(rank)).bind();
            if (gpu_scope.failed())
                return util::expected_t::fail(util::status_code_t::device_k, "context entry failed ({})", rank);
            std::size_t const member = domain_->member_offset() + rank;
            if (util::expected_t staged = (this->*phase)(rank, inputs[rank], streams.stream(rank)); !staged)
                return staged;
            if (util::expected_t met = domain_->barrier(member, streams.stream(rank)); !met) return met;
        }
    return {};
}

util::expected_t distributed_contrastive_tokens_t::stage_(std::size_t rank, gpu_tokens_inputs_t const& inputs,
                                                          util::executor_stream_t& stream) noexcept {
    std::size_t const gradients = 2u * std::size_t {config_.temperature_groups};
    util::f32r1_span_t const tail = window_tail_(domain_->member_offset() + rank);
    contrastive_tokens_t& loss = losses_[rank];
    util::expected_t staged = loss.stage(inputs.token_vectors, inputs.token_offsets, inputs.batch, stream);
    if (staged)
        staged = util::copy_device_to_device(loss.live_queries(),
                                             tail[nk::range(gradients + 1u, gradients + 2u), nk::slice], stream);
    if (!staged || world_domains_() == 1u) return staged;
    std::size_t const queries = config_.query_tokens_per_gpu, segments = config_.segments_per_gpu;
    std::size_t const lengths_begin = queries + segments + 1u;
    gpu_query_subgraph_t const own = loss.query_subgraph();
    util::u32r1_span_t const description = description_gathers_[rank].own_block();
    staged = util::copy_device_to_device(own.query_vectors.flatten<1>(), tile_gathers_[rank].own_block(), stream);
    if (staged)
        staged = util::copy_device_to_device(own.query_segments, description[nk::range(0, queries), nk::slice], stream);
    if (staged)
        staged = util::copy_device_to_device(
            own.query_offsets, description[nk::range(queries, queries + segments + 1u), nk::slice], stream);
    if (staged)
        staged = util::copy_device_to_device(loss.document_lengths(),
                                             description[nk::range(lengths_begin, lengths_begin + segments), nk::slice],
                                             stream);
    if (staged) staged = tile_gathers_[rank](stream);
    if (staged) staged = description_gathers_[rank](stream);
    return staged;
}

util::expected_t distributed_contrastive_tokens_t::score_(std::size_t rank, gpu_tokens_inputs_t const&,
                                                          util::executor_stream_t& stream) noexcept {
    std::size_t const subgraphs = config_.gpu_subgraphs_per_step, segments = config_.segments_per_gpu;
    std::size_t const gradients = 2u * std::size_t {config_.temperature_groups};
    std::size_t const member = domain_->member_offset() + rank;
    std::size_t const tail_offset = window_.offset + window_.scalars - world_scalar_count_(config_);
    contrastive_tokens_t& loss = losses_[rank];
    util::f32r3_span_t const outbound = window_scores_(member, 0);
    util::expected_t staged {};
    for (std::size_t home = 0; staged && home != subgraphs; ++home)
        staged = loss.score(home, world_subgraph_(home), outbound[home, nk::slice], stream);
    util::u32r2_span_t const world_document_lengths = world_document_lengths_[rank].span().reshape<2>(
        {subgraphs, segments});
    for (std::size_t world_gpu = 0; staged && world_gpu != subgraphs; ++world_gpu)
        staged = util::copy_device_to_device(world_document_lengths_of_(world_gpu),
                                             world_document_lengths[world_gpu, nk::slice], stream);
    util::f32r1_span_t const world_scalars = world_scalars_[rank].span();
    util::f32r1_span_t const world_live_queries = world_scalars[nk::range(gradients + 1u, gradients + 2u), nk::slice];
    if (staged) staged = domain_->reduce_scatter(member, world_live_queries, tail_offset + gradients + 1u, stream);
    if (!staged || world_domains_() == 1u) return staged;
    staged = util::copy_device_to_device(util::f32r1_view_t {world_live_queries}, count_reduces_[rank].payload(),
                                         stream);
    if (staged) staged = count_reduces_[rank](stream);
    if (staged)
        staged = util::copy_device_to_device(util::f32r1_view_t {count_reduces_[rank].payload()}, world_live_queries,
                                             stream);
    if (staged)
        staged = util::copy_device_to_device(util::f32r1_view_t {outbound.flatten<1>()},
                                             score_gathers_[rank].own_block(), stream);
    if (staged) staged = score_gathers_[rank](stream);
    return staged;
}

util::expected_t distributed_contrastive_tokens_t::relax_(std::size_t rank, gpu_tokens_inputs_t const& inputs,
                                                          util::executor_stream_t& stream) noexcept {
    std::size_t const subgraphs = config_.gpu_subgraphs_per_step, segments = config_.segments_per_gpu;
    std::size_t const groups = config_.temperature_groups, gradients = 2u * groups;
    std::size_t const member = domain_->member_offset() + rank;
    std::size_t const own_world_gpu = home_domain_index_ * losses_.size() + rank;
    util::f32r1_span_t const tail = window_tail_(member);
    util::f32r1_span_t const world_scalars = world_scalars_[rank].span();
    util::f32r3_span_t const home_scores = home_scores_[rank].span().reshape<3>({subgraphs, segments, segments});
    util::expected_t staged {};
    for (std::size_t document_gpu = 0; staged && document_gpu != subgraphs; ++document_gpu) {
        util::f32r3_view_t const outbound = world_scores_(document_gpu, 0);
        staged = util::copy_device_to_device(outbound[own_world_gpu, nk::slice].flatten<1>(),
                                             home_scores[document_gpu, nk::slice].flatten<1>(), stream);
    }
    if (staged)
        staged = losses_[rank].relax(
            home_scores_[rank].span().reshape<4>({1u, subgraphs, segments, segments}), inputs.relations, inputs.recipe,
            world_document_lengths_[rank].view(),
            util::f32r1_view_t {world_scalars[nk::range(gradients + 1u, gradients + 2u), nk::slice]},
            tail[nk::range(gradients, gradients + 1u), nk::slice], stream, tail[nk::range(0, groups), nk::slice],
            tail[nk::range(groups, gradients), nk::slice]);
    util::f32r3_span_t const inbound = window_scores_(member, 1);
    if (staged) staged = util::copy_device_to_device(home_scores_[rank].view(), inbound.flatten<1>(), stream);
    if (!staged || world_domains_() == 1u) return staged;
    staged = util::copy_device_to_device(util::f32r1_view_t {inbound.flatten<1>()}, gradient_gathers_[rank].own_block(),
                                         stream);
    if (staged) staged = gradient_gathers_[rank](stream);
    return staged;
}

util::expected_t distributed_contrastive_tokens_t::force_(std::size_t rank, gpu_tokens_inputs_t const&,
                                                          util::executor_stream_t& stream) noexcept {
    std::size_t const subgraphs = config_.gpu_subgraphs_per_step;
    std::size_t const own_world_gpu = home_domain_index_ * losses_.size() + rank;
    util::f32r3_span_t const query_forces = window_query_forces_(domain_->member_offset() + rank);
    util::expected_t staged {};
    for (std::size_t home = 0; staged && home != subgraphs; ++home) {
        util::f32r3_view_t const home_gradients = world_scores_(home, 1);
        staged = losses_[rank].force(home, world_subgraph_(home), home_gradients[own_world_gpu, nk::slice],
                                     query_forces[home, nk::slice], stream);
    }
    return staged;
}

util::expected_t distributed_contrastive_tokens_t::reduce_(std::size_t rank, gpu_tokens_inputs_t const& inputs,
                                                           util::executor_stream_t& stream) noexcept {
    std::size_t const members = losses_.size(), segments = config_.segments_per_gpu;
    std::size_t const subgraphs = config_.gpu_subgraphs_per_step;
    std::size_t const gradients = 2u * std::size_t {config_.temperature_groups};
    std::size_t const member = domain_->member_offset() + rank;
    std::size_t const forces_offset = window_.offset + 2u * subgraphs * segments * segments;
    std::size_t const force_scalars = std::size_t {config_.query_tokens_per_gpu} * config_.width;
    std::size_t const tail_offset = window_.offset + window_.scalars - world_scalar_count_(config_);
    util::f32r1_span_t const world_scalars = world_scalars_[rank].span();
    util::f32r1_span_t const query_grad = query_grads_[rank].span().flatten<1>();
    util::expected_t staged {};
    if (world_domains_() == 1u) {
        staged = domain_->reduce_scatter(member, query_grad, forces_offset + rank * force_scalars, stream);
        if (staged)
            staged = domain_->reduce_scatter(member, world_scalars[nk::range(0, gradients + 1u), nk::slice],
                                             tail_offset, stream);
    }
    else {
        // Level 1 sums this member's row of every domain's query tiles, level 2 sums
        // them across domains.
        util::f32r1_span_t const shard = force_reduces_[rank].payload();
        std::size_t const world_forces = world_domains_() * force_scalars;
        util::f32r2_span_t const shard_rows = shard[nk::range(0, world_forces), nk::slice].reshape<2>(
            {std::size_t {world_domains_()}, force_scalars});
        for (util::u32_t domain = 0; staged && domain != world_domains_(); ++domain)
            staged = domain_->reduce_scatter(member, shard_rows[domain, nk::slice],
                                             forces_offset + (domain * members + rank) * force_scalars, stream);
        util::f32r1_span_t const scalars_shard =
            shard[nk::range(world_forces, world_forces + gradients + 1u), nk::slice];
        if (staged) staged = domain_->reduce_scatter(member, scalars_shard, tail_offset, stream);
        if (staged) staged = force_reduces_[rank](stream);
        if (staged)
            staged = util::copy_device_to_device(util::f32r1_view_t {shard_rows[home_domain_index_, nk::slice]},
                                                 query_grad, stream);
        if (staged)
            staged = util::copy_device_to_device(util::f32r1_view_t {scalars_shard},
                                                 world_scalars[nk::range(0, gradients + 1u), nk::slice], stream);
    }
    if (staged)
        staged = losses_[rank].finish(query_grads_[rank].view(), inputs.token_offsets, inputs.batch,
                                      inputs.token_vectors_grad, stream);
    return staged;
}
