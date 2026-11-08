/**
 *  @file src/train/contrastive_tokens.cu
 *  @author Ash Vardanian
 *  @date October 21, 2026
 *  @brief Device kernels and method bodies of the late-interaction tokens loss declared in
 *      @ref contrastive_tokens.cuh .
 */
#include <cmath> // `std::fmax`

#include <algorithm> // `std::max` — the document rows cover at least one band
#include <limits>    // `std::numeric_limits` — the empty maximum

#include <cub/block/block_scan.cuh>     // `cub::BlockScan` — the tile layouts
#include <cub/thread/thread_search.cuh> // `cub::UpperBound` — a tile row's segment

#include "uform/train/contrastive_tokens.cuh"

namespace ue = unum::uform::encoder;
using ue::late_interaction_side_t;
using ue::multimodal_batch_view_t;
using namespace unum::uform::train;
using namespace unum::uform::utilities;

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
[[nodiscard]] static __device__ inline u32_t contrastive_tokens_owner_(u32r1_view_t offsets, u32_t row) noexcept {
    return static_cast<u32_t>(cub::UpperBound(offsets.data(), static_cast<u32_t>(offsets.extent(0)), row)) - 1u;
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
    multimodal_batch_view_t batch, u32r1_view_t token_offsets, u32r1_span_t query_offsets, u32r1_span_t query_lengths,
    u32r1_span_t document_offsets, u32r1_span_t document_segments, u32r1_span_t document_lengths,
    u32r1_span_t document_of_segment, f32r1_span_t live_queries, u32r1_span_t overflow, u32_t query_capacity,
    u32_t document_capacity, u32_t band_tokens, u32_t query_span_tokens) {
    using block_scan_t = cub::BlockScan<u32_t, kernel_block_size_default_k>;
    __shared__ typename block_scan_t::TempStorage scan_storage;
    __shared__ u32_t query_rows, documents, queries;
    u32_t const segments = static_cast<u32_t>(query_lengths.extent(0));
    u32_t const documents_capacity = static_cast<u32_t>(document_segments.extent(0));
    if (threadIdx.x == 0u) query_rows = documents = queries = 0u;
    __syncthreads();
    for (u32_t chunk = 0; chunk < segments; chunk += blockDim.x) {
        u32_t const segment = chunk + threadIdx.x;
        bool const live = segment < segments;
        u32_t const span = live ? token_offsets[segment + 1u] - token_offsets[segment] : 0u;
        late_interaction_side_t const side = live ? batch.segment_side[segment] : late_interaction_side_t::none_k;
        u32_t const query_span = contrastive_tokens_queries_(side) ? min(span, query_span_tokens) : 0u;
        u32_t const is_document = contrastive_tokens_documents_(side) && span != 0u ? 1u : 0u;
        u32_t const document_span = is_document ? span : 0u;
        u32_t query_row = 0, document = 0, is_query = 0, totals[3] = {};
        block_scan_t(scan_storage).ExclusiveSum(query_span, query_row, totals[0]);
        __syncthreads();
        block_scan_t(scan_storage).ExclusiveSum(is_document, document, totals[1]);
        __syncthreads();
        block_scan_t(scan_storage).ExclusiveSum(query_span != 0u ? 1u : 0u, is_query, totals[2]);
        if (live) {
            query_offsets[segment] = query_rows + query_row;
            query_lengths[segment] = query_span;
            document_lengths[segment] = document_span;
            u32_t const ordinal = documents + document;
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
    for (u32_t ordinal = documents + threadIdx.x; ordinal < documents_capacity; ordinal += blockDim.x)
        document_segments[ordinal] = ~0u;
    __syncthreads();
    if (threadIdx.x != 0u) return;
    u32_t cursor = 0, placed = min(documents, documents_capacity);
    bool oversized = false;
    for (u32_t ordinal = 0; ordinal != placed; ++ordinal) {
        u32_t const span = document_lengths[document_segments[ordinal]];
        oversized = oversized || span > band_tokens;
        if (cursor % band_tokens + span > band_tokens) cursor += band_tokens - cursor % band_tokens;
        document_offsets[ordinal] = cursor;
        cursor += span;
    }
    for (u32_t ordinal = placed; ordinal <= documents_capacity; ++ordinal) document_offsets[ordinal] = cursor;
    query_offsets[segments] = query_rows;
    live_queries[0] = static_cast<f32_t>(queries);
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
static __global__ void contrastive_tokens_gather_queries_kernel_(bf16r2_view_t token_vectors,
                                                                 u32r1_view_t token_offsets, u32r1_view_t query_offsets,
                                                                 bf16r2_span_t query_tile,
                                                                 u32r1_span_t query_segments) {
    u32_t const row = blockIdx.x;
    std::size_t const segments = query_offsets.extent(0) - 1u;
    bool const live = row < query_offsets[segments];
    u32_t const segment = live ? contrastive_tokens_owner_(query_offsets, row) : ~0u;
    u32_t const source = live ? token_offsets[segment] + (row - query_offsets[segment]) : 0u;
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
    bf16r2_view_t token_vectors, u32r1_view_t token_offsets, u32r1_view_t document_offsets,
    u32r1_view_t document_segments, bf16r2_span_t document_tile, u32r1_span_t document_of_row) {
    u32_t const row = blockIdx.x;
    u32_t const owner = row < document_offsets[document_offsets.extent(0) - 1u]
                            ? contrastive_tokens_owner_(document_offsets, row)
                            : ~0u;
    u32_t const segment = owner != ~0u ? document_segments[owner] : ~0u;
    bool const live = segment != ~0u &&
                      row < document_offsets[owner] + token_offsets[segment + 1u] - token_offsets[segment];
    u32_t const document = live ? owner : ~0u;
    u32_t const source = live ? token_offsets[segment] + (row - document_offsets[owner]) : 0u;
    for (std::size_t channel = threadIdx.x; channel < document_tile.extent(1); channel += blockDim.x)
        document_tile[row, channel] = live ? token_vectors[source, channel] : __float2bfloat16(0.0f);
    if (threadIdx.x == 0u) document_of_row[row] = document;
}

/** One warp's view of a band chunk: the document of each lane's row, @c ~0u past the documents, the
 *  first lane of its document's run within the chunk, and whether the lane closes that run. */
struct contrastive_tokens_lane_t {
    u32_t document;
    u32_t run_begin_lane;
    bool closes_run;
};

/** Classify lane @p lane holding @p document within its chunk; the documents tile
 *  the rows contiguously. */
[[nodiscard]] static __device__ inline contrastive_tokens_lane_t contrastive_tokens_lane_(u32_t document,
                                                                                          u32_t lane) noexcept {
    u32_t const previous = __shfl_up_sync(0xffffffffu, document, 1u);
    u32_t const next = __shfl_down_sync(0xffffffffu, document, 1u);
    u32_t run_begin_lane = lane == 0u || previous != document ? lane : 0u;
    for (u32_t delta = 1u; delta != warp_size_k; delta *= 2u) {
        u32_t const earlier = __shfl_up_sync(0xffffffffu, run_begin_lane, delta);
        if (lane >= delta) run_begin_lane = max(run_begin_lane, earlier);
    }
    return {document, run_begin_lane, document != ~0u && (lane + 1u == warp_size_k || next != document)};
}

/** The warp-wide maximum of @p value , on every lane. */
[[nodiscard]] static __device__ inline f32_t contrastive_tokens_warp_max_(f32_t value) noexcept {
    for (u32_t delta = warp_size_k / 2u; delta != 0u; delta /= 2u)
        value = std::fmax(value, __shfl_xor_sync(0xffffffffu, value, delta));
    return value;
}

/** Rows a warp covers per wide step, eight per lane through two 16-byte loads and
 *  one 16-byte store. */
inline constexpr std::size_t contrastive_tokens_wide_rows_k = 8u * warp_size_k;

/** The four products at @p column and after of a 16-byte aligned row. */
[[nodiscard]] static __device__ inline float4 contrastive_tokens_load_(f32_t const *row_start,
                                                                       std::size_t column) noexcept {
    float4 vector;
    asm("ld.global.v4.f32 {%0, %1, %2, %3}, [%4];"
        : "=f"(vector.x), "=f"(vector.y), "=f"(vector.z), "=f"(vector.w)
        : "l"(row_start + column));
    return vector;
}

/** The BFloat16 pair @p low , @p high packed into one word, @p low first in memory. */
[[nodiscard]] static __device__ inline u32_t contrastive_tokens_pack_(f32_t low, f32_t high) noexcept {
    return static_cast<u32_t>(__bfloat16_as_ushort(__float2bfloat16(low))) |
           static_cast<u32_t>(__bfloat16_as_ushort(__float2bfloat16(high))) << 16u;
}

/** Store the eight gradients @p low then @p high at @p column and after of a 16-byte
 *  aligned BFloat16 row. */
static __device__ inline void contrastive_tokens_store_(bf16_t *row_start, std::size_t column, float4 low,
                                                        float4 high) noexcept {
    asm volatile("st.global.v4.u32 [%0], {%1, %2, %3, %4};" ::"l"(row_start + column),
                 "r"(contrastive_tokens_pack_(low.x, low.y)), "r"(contrastive_tokens_pack_(low.z, low.w)),
                 "r"(contrastive_tokens_pack_(high.x, high.y)), "r"(contrastive_tokens_pack_(high.z, high.w))
                 : "memory");
}

/** The document of band row @p row , or @c ~0u past the live rows. */
[[nodiscard]] static __device__ inline u32_t contrastive_tokens_document_(u32r1_view_t document_of_row, std::size_t row,
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
static __global__ void contrastive_tokens_band_max_kernel_(f32r2_view_t band_products, u32r1_view_t query_segments,
                                                           u32r1_view_t document_of_row, u32r1_view_t document_offsets,
                                                           std::size_t band_begin, f32r2_span_t maxima) {
    u32_t const lane = threadIdx.x % warp_size_k;
    std::size_t const query_row = static_cast<std::size_t>(blockIdx.x) * (blockDim.x / warp_size_k) +
                                  threadIdx.x / warp_size_k;
    if (query_row >= band_products.extent(0) || query_segments[query_row] == ~0u) return;
    std::size_t const live_rows = document_offsets[document_offsets.extent(0) - 1u];
    std::size_t const band_rows = band_products.extent(1);
    f32_t const *const products = &band_products[query_row, 0];
    u32_t open_document = ~0u;
    f32_t open_maximum = -std::numeric_limits<f32_t>::infinity();
    auto const fold_chunk = [&](std::size_t chunk) noexcept {
        std::size_t const row = band_begin + chunk + lane;
        u32_t const document = contrastive_tokens_document_(document_of_row, row, live_rows);
        f32_t const value = products[chunk + lane];
        if (__all_sync(0xffffffffu, document == open_document)) {
            open_maximum = std::fmax(open_maximum, value);
            return;
        }
        f32_t const closing = contrastive_tokens_warp_max_(
            std::fmax(open_maximum, document == open_document ? value : -std::numeric_limits<f32_t>::infinity()));
        if (lane == 0u && open_document != ~0u) maxima[query_row, open_document] = closing;
        contrastive_tokens_lane_t const position = contrastive_tokens_lane_(document, lane);
        f32_t reduced = value;
        for (u32_t delta = 1u; delta != warp_size_k; delta *= 2u) {
            f32_t const earlier = __shfl_up_sync(0xffffffffu, reduced, delta);
            if (lane >= delta && lane - delta >= position.run_begin_lane) reduced = std::fmax(reduced, earlier);
        }
        u32_t const last_document = __shfl_sync(0xffffffffu, document, warp_size_k - 1u);
        if (position.closes_run && document != open_document && document != last_document)
            maxima[query_row, document] = reduced;
        open_document = last_document;
        open_maximum = document == last_document ? value : -std::numeric_limits<f32_t>::infinity();
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
        for (std::size_t chunk = step; chunk < step_end && chunk < band_rows; chunk += warp_size_k) fold_chunk(chunk);
    }
    f32_t const closing = contrastive_tokens_warp_max_(open_maximum);
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
static __global__ void contrastive_tokens_scores_kernel_(f32r2_view_t maxima, u32r1_view_t query_offsets,
                                                         u32r1_view_t document_segments, f32r2_span_t raw_scores) {
    u32_t const segment = blockIdx.x;
    std::size_t const rows = maxima.extent(0);
    std::size_t const begin = query_offsets[segment], end = query_offsets[segment + 1u];
    if (begin == end || end > rows) return;
    f32_t const inverse_length = 1.0f / static_cast<f32_t>(end - begin);
    for (std::size_t document = threadIdx.x; document < document_segments.extent(0); document += blockDim.x) {
        if (document_segments[document] == ~0u) continue;
        f32_t sum = 0.0f;
        for (std::size_t row = begin; row != end; ++row) sum += maxima[row, document];
        raw_scores[segment, document_segments[document]] = sum * inverse_length;
    }
}

/**
 *  @brief One band's credited gradient: for every live query token and document, the document's
 *      lowest row whose recomputed product equals the stored maximum takes @c d(raw)/|q| , every
 *      other cell zero.
 *
 *  Ties resolve to the lowest row: a ballot over each step finds it, and the claim carries into
 *  the document's later rows, since a document never crosses a band. Wide steps inside the open
 *  document move eight rows per lane, as in @ref contrastive_tokens_band_max_kernel_ .
 *
 *  __grid_size__ @c [ceil(query_tokens/8),1,1] ; one warp per query row, 8 per CTA; no
 *  dynamic shared memory.
 */
static __global__ void contrastive_tokens_band_grad_kernel_(f32r2_view_t band_products, u32r1_view_t query_segments,
                                                            u32r1_view_t query_offsets, u32r1_view_t document_of_row,
                                                            u32r1_view_t document_offsets,
                                                            u32r1_view_t document_segments, f32r2_view_t score_grad,
                                                            std::size_t band_begin, f32r2_view_t maxima,
                                                            bf16r2_span_t band_grad) {
    u32_t const lane = threadIdx.x % warp_size_k;
    std::size_t const query_row = static_cast<std::size_t>(blockIdx.x) * (blockDim.x / warp_size_k) +
                                  threadIdx.x / warp_size_k;
    if (query_row >= band_products.extent(0)) return;
    u32_t const query = query_segments[query_row];
    std::size_t const live_rows = query != ~0u ? document_offsets[document_offsets.extent(0) - 1u] : 0u;
    std::size_t const band_rows = band_products.extent(1);
    f32_t const inverse_length = query != ~0u
                                     ? 1.0f / static_cast<f32_t>(query_offsets[query + 1u] - query_offsets[query])
                                     : 0.0f;
    f32_t const *const products = &band_products[query_row, 0];
    bf16_t *const gradients = &band_grad[query_row, 0];
    u32_t open_document = ~0u;
    f32_t open_maximum = 0.0f, open_gradient = 0.0f;
    bool open_claimed = false;
    auto const open = [&](u32_t document) noexcept {
        open_document = document;
        open_maximum = document != ~0u ? maxima[query_row, document] : 0.0f;
        open_gradient = document != ~0u ? score_grad[query, document_segments[document]] * inverse_length : 0.0f;
    };
    auto const credit_chunk = [&](std::size_t chunk) noexcept {
        u32_t const document = contrastive_tokens_document_(document_of_row, band_begin + chunk + lane, live_rows);
        f32_t const value = products[chunk + lane];
        if (__all_sync(0xffffffffu, document == open_document)) {
            u32_t const matching_lanes = __ballot_sync(0xffffffffu, document != ~0u && value == open_maximum);
            bool const claims = !open_claimed && matching_lanes != 0u &&
                                lane == static_cast<u32_t>(__ffs(static_cast<int>(matching_lanes)) - 1);
            open_claimed = open_claimed || matching_lanes != 0u;
            gradients[chunk + lane] = __float2bfloat16(claims ? open_gradient : 0.0f);
            return;
        }
        bool const matches = document != ~0u && value == maxima[query_row, document];
        u32_t const matching_lanes = __ballot_sync(0xffffffffu, matches);
        contrastive_tokens_lane_t const position = contrastive_tokens_lane_(document, lane);
        u32_t const run_lanes = ~((1u << position.run_begin_lane) - 1u);
        bool const carried = open_claimed && document == open_document;
        bool const claims = matches && (matching_lanes & ((1u << lane) - 1u) & run_lanes) == 0u && !carried;
        u32_t const last_document = __shfl_sync(0xffffffffu, document, warp_size_k - 1u);
        u32_t const last_run_lanes = __shfl_sync(0xffffffffu, run_lanes, warp_size_k - 1u);
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
            f32_t const values[8] = {low.x, low.y, low.z, low.w, high.x, high.y, high.z, high.w};
            u32_t first = 8u;
            for (u32_t index = 8u; index != 0u; --index)
                if (values[index - 1u] == open_maximum) first = index - 1u;
            u32_t const matching_lanes = __ballot_sync(0xffffffffu, first != 8u);
            bool const claims = !open_claimed && matching_lanes != 0u &&
                                lane == static_cast<u32_t>(__ffs(static_cast<int>(matching_lanes)) - 1);
            open_claimed = open_claimed || matching_lanes != 0u;
            auto const credit = [&](u32_t index) noexcept { return claims && first == index ? open_gradient : 0.0f; };
            contrastive_tokens_store_(gradients, step + 8u * lane, float4 {credit(0), credit(1), credit(2), credit(3)},
                                      float4 {credit(4), credit(5), credit(6), credit(7)});
            continue;
        }
        for (std::size_t chunk = step; chunk < step_end && chunk < band_rows; chunk += warp_size_k) credit_chunk(chunk);
    }
}

/**
 *  @brief The token head's gradient: each row's query force and document force by its segment's
 *      sides, in that order, and zero past the live rows.
 *
 *  __grid_size__ @c [head_tokens,1,1] ; one CTA per row; threads stripe the width; no
 *  dynamic shared memory.
 */
static __global__ void contrastive_tokens_finish_kernel_(multimodal_batch_view_t batch, u32r1_view_t token_offsets,
                                                         u32r1_view_t query_offsets, f32r2_view_t query_grad,
                                                         u32r1_view_t document_of_segment,
                                                         u32r1_view_t document_offsets, f32r2_view_t document_grad,
                                                         bf16r2_span_t token_vectors_grad) {
    u32_t const row = blockIdx.x;
    std::size_t const segments = token_offsets.extent(0) - 1u;
    bool const live = row < token_offsets[segments];
    u32_t const segment = live ? contrastive_tokens_owner_(token_offsets, row) : 0u;
    u32_t const offset = live ? row - token_offsets[segment] : 0u;
    late_interaction_side_t const side = live ? batch.segment_side[segment] : late_interaction_side_t::none_k;
    bool const queried = contrastive_tokens_queries_(side) &&
                         offset < query_offsets[segment + 1u] - query_offsets[segment];
    std::size_t const query_row = queried ? query_offsets[segment] + offset : ~std::size_t {0};
    u32_t const document = contrastive_tokens_documents_(side) ? document_of_segment[segment] : ~0u;
    std::size_t const document_row = document != ~0u ? document_offsets[document] + offset : ~std::size_t {0};
    for (std::size_t channel = threadIdx.x; channel < token_vectors_grad.extent(1); channel += blockDim.x) {
        f32_t gradient = 0.0f;
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
static __global__ void contrastive_tokens_scale_kernel_(f32r3_span_t scores, contrastive_recipe_t recipe,
                                                        f32_t weight) {
    std::size_t const rows = scores.extent(1), columns = scores.extent(2);
    std::size_t const cells = scores.extent(0) * rows * columns;
    std::size_t const stride = static_cast<std::size_t>(gridDim.x) * blockDim.x;
    for (std::size_t cell = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x; cell < cells;
         cell += stride) {
        std::size_t const row = (cell / columns) % rows;
        scores[cell / (rows * columns), row, cell % columns] *= weight * contrastive_from_vertex_beta_(
                                                                             recipe, static_cast<u32_t>(row));
    }
}

/** Scale the loss summand and the temperature-gradient summands by the loss weight @p weight .
 *
 *  __grid_size__ @c [1,1,1] ; threads stripe the temperature groups; no dynamic shared memory. */
static __global__ void contrastive_tokens_weigh_kernel_(f32r1_span_t loss_out,
                                                        f32r1_span_t log_inverse_temperature_grad,
                                                        f32r1_span_t bias_grad, f32_t weight) {
    if (threadIdx.x == 0u) loss_out[0] *= weight;
    for (std::size_t group = threadIdx.x; group < log_inverse_temperature_grad.extent(0); group += blockDim.x)
        log_inverse_temperature_grad[group] *= weight;
    for (std::size_t group = threadIdx.x; group < bias_grad.extent(0); group += blockDim.x) bias_grad[group] *= weight;
}

expected<contrastive_tokens_t> contrastive_tokens_t::make(gpu_arena_t &arena, contrastive_tokens_config_t config,
                                                          gpu_numerics_t const &numerics) noexcept {
    using result_t = expected<contrastive_tokens_t>;
    if (config.segments_per_gpu == 0 || config.head_tokens_per_gpu == 0 || config.query_tokens_per_gpu == 0 ||
        config.documents_per_gpu == 0 || config.band_tokens == 0 || config.gpu_subgraphs_per_step == 0)
        return result_t::fail(
            status_t::invalid_argument_k,
            "the tokens loss needs segments, head rows, query rows, documents, a band and a subgraph");
    if (config.width % 8u != 0u || config.query_tokens_per_gpu % 8u != 0u || config.band_tokens % warp_size_k != 0u)
        return result_t::fail(
            status_t::unsupported_k,
            "width and query rows must be multiples of 8 and the band of {} (width={} queries={} " "band={})",
            warp_size_k, config.width, config.query_tokens_per_gpu, config.band_tokens);
    config.document_tokens_per_gpu = static_cast<u32_t>(
        round_up_to_multiple(std::max(config.document_tokens_per_gpu, config.band_tokens), config.band_tokens));

    contrastive_tokens_t loss;
    loss.config_ = config;
    loss.numerics_ = numerics;
    expected<contrastive_pair_loss_t> pair_loss = contrastive_pair_loss_t::make(
        arena, config.segments_per_gpu, config.segments_per_gpu * config.gpu_subgraphs_per_step, config.pair_loss);
    if (!pair_loss) return result_t::fail(std::move(pair_loss));
    loss.pair_loss_ = std::move(pair_loss.value_);

    std::size_t const segments = config.segments_per_gpu, width = config.width;
    std::size_t const queries = config.query_tokens_per_gpu, documents = config.documents_per_gpu;
    std::size_t const document_rows = config.document_tokens_per_gpu, band = config.band_tokens;
    std::size_t const bands = document_rows / band;
    gpu_arena_alloc<bf16_t> bf16_alloc {arena};
    gpu_arena_alloc<f32_t> f32_alloc {arena};
    gpu_arena_alloc<u32_t> u32_alloc {arena};
    gpu_arena_alloc<b8_t> byte_alloc {arena};
    loss.query_tile_ = bf16r2_tensor_arena_t::uninitialized({queries, width}, bf16_alloc).value;
    loss.query_segments_ = u32r1_tensor_arena_t::uninitialized({queries}, u32_alloc).value;
    loss.query_offsets_ = u32r1_tensor_arena_t::uninitialized({segments + 1u}, u32_alloc).value;
    loss.query_lengths_ = u32r1_tensor_arena_t::uninitialized({segments}, u32_alloc).value;
    loss.document_tile_ = bf16r2_tensor_arena_t::uninitialized({document_rows, width}, bf16_alloc).value;
    loss.document_of_row_ = u32r1_tensor_arena_t::uninitialized({document_rows}, u32_alloc).value;
    loss.document_offsets_ = u32r1_tensor_arena_t::uninitialized({documents + 1u}, u32_alloc).value;
    loss.document_segments_ = u32r1_tensor_arena_t::uninitialized({documents}, u32_alloc).value;
    loss.document_lengths_ = u32r1_tensor_arena_t::uninitialized({segments}, u32_alloc).value;
    loss.document_of_segment_ = u32r1_tensor_arena_t::uninitialized({segments}, u32_alloc).value;
    loss.live_queries_ = f32r1_tensor_arena_t::uninitialized({1}, f32_alloc).value;
    loss.overflow_ = u32r1_tensor_arena_t::uninitialized({1}, u32_alloc).value;
    loss.query_tile_transposed_ = bf16r2_tensor_arena_t::uninitialized({width, queries}, bf16_alloc).value;
    loss.document_tile_transposed_ = bf16r2_tensor_arena_t::uninitialized({width, document_rows}, bf16_alloc).value;
    loss.maxima_ =
        f32r3_tensor_arena_t::uninitialized({config.gpu_subgraphs_per_step, queries, documents}, f32_alloc).value;
    loss.band_products_ = f32r2_tensor_arena_t::uninitialized({queries, band}, f32_alloc).value;
    loss.band_grad_ = bf16r2_tensor_arena_t::uninitialized({queries, band}, bf16_alloc).value;
    loss.band_grad_transposed_ = bf16r2_tensor_arena_t::uninitialized({band, queries}, bf16_alloc).value;
    loss.query_grad_products_ = f32r2_tensor_arena_t::uninitialized({queries, width}, f32_alloc).value;
    loss.document_grad_products_ = f32r2_tensor_arena_t::uninitialized({band, width}, f32_alloc).value;
    loss.document_grad_ = f32r2_tensor_arena_t::uninitialized({document_rows, width}, f32_alloc).value;
    loss.query_grad_ = f32r2_tensor_arena_t::uninitialized({queries, width}, f32_alloc).value;
    loss.scores_ = f32r1_tensor_arena_t::uninitialized({segments * segments}, f32_alloc).value;
    if (!loss.query_tile_ || !loss.query_segments_ || !loss.query_offsets_ || !loss.query_lengths_ ||
        !loss.document_tile_ || !loss.document_of_row_ || !loss.document_offsets_ || !loss.document_segments_ ||
        !loss.document_lengths_ || !loss.document_of_segment_ || !loss.live_queries_ || !loss.overflow_ ||
        !loss.query_tile_transposed_ || !loss.document_tile_transposed_ || !loss.maxima_ || !loss.band_products_ ||
        !loss.band_grad_ || !loss.band_grad_transposed_ || !loss.query_grad_products_ ||
        !loss.document_grad_products_ || !loss.document_grad_ || !loss.query_grad_ || !loss.scores_)
        return result_t::fail(status_t::bad_alloc_k, "tokens loss allocation failed (arena_used={})", arena.used());

    expected<std::size_t> band_bytes = pack_bytes_bf16(band, width, loss.numerics_);
    if (!band_bytes) return result_t::fail(std::move(band_bytes));
    expected<std::size_t> band_transposed_bytes = pack_bytes_bf16(width, band, loss.numerics_);
    if (!band_transposed_bytes) return result_t::fail(std::move(band_transposed_bytes));
    expected<std::size_t> queries_transposed_bytes = pack_bytes_bf16(width, queries, loss.numerics_);
    if (!queries_transposed_bytes) return result_t::fail(std::move(queries_transposed_bytes));
    loss.band_pack_bytes_ = band_bytes.value_;
    loss.band_pack_transposed_bytes_ = band_transposed_bytes.value_;
    loss.packed_documents_ = b8r1_tensor_arena_t::uninitialized({bands * loss.band_pack_bytes_}, byte_alloc).value;
    loss.packed_documents_transposed_ =
        b8r1_tensor_arena_t::uninitialized({bands * loss.band_pack_transposed_bytes_}, byte_alloc).value;
    loss.packed_queries_transposed_ =
        b8r1_tensor_arena_t::uninitialized({queries_transposed_bytes.value_}, byte_alloc).value;
    if (!loss.packed_documents_ || !loss.packed_documents_transposed_ || !loss.packed_queries_transposed_)
        return result_t::fail(status_t::bad_alloc_k, "tokens loss pack allocation failed ({} bands)", bands);
    return result_t {std::move(loss)};
}

expected_t contrastive_tokens_t::stage(bf16r2_view_t token_vectors, u32r1_view_t token_offsets,
                                       multimodal_batch_view_t const &batch, gpu_stream_t &stream) noexcept {
    if (token_vectors.extent(0) != config_.head_tokens_per_gpu || token_vectors.extent(1) != config_.width ||
        token_offsets.extent(0) != config_.segments_per_gpu + 1u)
        return expected_t::fail(status_t::unsupported_k,
                                "token head output [{}x{}] over {} offsets disagrees with the baked [{}x{}] over {}",
                                token_vectors.extent(0), token_vectors.extent(1), token_offsets.extent(0),
                                config_.head_tokens_per_gpu, config_.width, config_.segments_per_gpu + 1u);
    dim3 const block_shape {kernel_block_size_default_k, 1u, 1u};
    std::size_t const band = config_.band_tokens;
    expected_t staged = launch_kernel(
        &contrastive_tokens_layout_kernel_, dim3 {1u, 1u, 1u}, block_shape, stream, batch, token_offsets,
        query_offsets_.span(), query_lengths_.span(), document_offsets_.span(), document_segments_.span(),
        document_lengths_.span(), document_of_segment_.span(), live_queries_.span(), overflow_.span(),
        config_.query_tokens_per_gpu, config_.document_tokens_per_gpu, config_.band_tokens, config_.query_span_tokens);
    if (staged)
        staged = launch_kernel(&contrastive_tokens_gather_queries_kernel_, dim3 {config_.query_tokens_per_gpu, 1u, 1u},
                               block_shape, stream, token_vectors, token_offsets, query_offsets_.view(),
                               query_tile_.span(), query_segments_.span());
    if (staged)
        staged = launch_kernel(&contrastive_tokens_gather_documents_kernel_,
                               dim3 {config_.document_tokens_per_gpu, 1u, 1u}, block_shape, stream, token_vectors,
                               token_offsets, document_offsets_.view(), document_segments_.view(),
                               document_tile_.span(), document_of_row_.span());
    if (staged) staged = transpose_bf16(document_tile_.view(), document_tile_transposed_.span(), stream);
    for (std::size_t band_index = 0; staged && band_index != bands_(); ++band_index) {
        auto const rows = nk::range(band_index * band, (band_index + 1u) * band);
        staged = pack_bf16(document_tile_[rows, nk::slice],
                           packed_band_(packed_documents_, band_pack_bytes_, band_index), numerics_, stream);
        if (staged)
            staged = pack_bf16(document_tile_transposed_[nk::all, rows, nk::slice],
                               packed_band_(packed_documents_transposed_, band_pack_transposed_bytes_, band_index),
                               numerics_, stream);
    }
    if (staged) staged = zero_async(document_grad_.span(), stream);
    return staged;
}

expected_t contrastive_tokens_t::score(std::size_t subgraph_index, gpu_query_subgraph_t const &subgraph,
                                       f32r2_span_t raw_scores, gpu_stream_t &stream) noexcept {
    if (subgraph_index >= config_.gpu_subgraphs_per_step ||
        subgraph.query_vectors.extent(0) != config_.query_tokens_per_gpu)
        return expected_t::fail(status_t::unsupported_k, "query subgraph {} disagrees with the baked shape",
                                subgraph_index);
    std::size_t const band = config_.band_tokens;
    f32r2_span_t const maxima = maxima_[subgraph_index, nk::slice];
    dim3 const block_shape {kernel_block_size_default_k, 1u, 1u};
    dim3 const row_warps_grid {
        static_cast<u32_t>(divide_round_up(std::size_t {config_.query_tokens_per_gpu},
                                           std::size_t {kernel_block_size_default_k / warp_size_k})),
        1u, 1u};
    expected_t staged = zero_async(raw_scores, stream);
    for (std::size_t band_index = 0; staged && band_index != bands_(); ++band_index) {
        staged = multiply_bf16(subgraph.query_vectors, packed_band_(packed_documents_, band_pack_bytes_, band_index),
                               band_products_.span(), numerics_, stream);
        if (staged)
            staged = launch_kernel(&contrastive_tokens_band_max_kernel_, row_warps_grid, block_shape, stream,
                                   band_products_.view(), subgraph.query_segments, document_of_row_.view(),
                                   document_offsets_.view(), band_index * band, maxima);
    }
    if (staged)
        staged = launch_kernel(&contrastive_tokens_scores_kernel_, dim3 {config_.segments_per_gpu, 1u, 1u}, block_shape,
                               stream, f32r2_view_t {maxima}, subgraph.query_offsets, document_segments_.view(),
                               raw_scores);
    return staged;
}

expected_t contrastive_tokens_t::force(std::size_t subgraph_index, gpu_query_subgraph_t const &subgraph,
                                       f32r2_view_t score_grad, f32r2_span_t query_grad,
                                       gpu_stream_t &stream) noexcept {
    std::size_t const band = config_.band_tokens, queries = config_.query_tokens_per_gpu, width = config_.width;
    f32r2_span_t const maxima = maxima_[subgraph_index, nk::slice];
    dim3 const block_shape {kernel_block_size_default_k, 1u, 1u};
    dim3 const row_warps_grid {
        static_cast<u32_t>(divide_round_up(queries, std::size_t {kernel_block_size_default_k / warp_size_k})), 1u, 1u};
    expected_t staged = transpose_bf16(subgraph.query_vectors, query_tile_transposed_.span(), stream);
    if (staged) staged = pack_bf16(query_tile_transposed_.view(), packed_queries_transposed_.span(), numerics_, stream);
    if (staged) staged = zero_async(query_grad, stream);
    for (std::size_t band_index = 0; staged && band_index != bands_(); ++band_index) {
        staged = multiply_bf16(subgraph.query_vectors, packed_band_(packed_documents_, band_pack_bytes_, band_index),
                               band_products_.span(), numerics_, stream);
        if (staged)
            staged = launch_kernel(&contrastive_tokens_band_grad_kernel_, row_warps_grid, block_shape, stream,
                                   band_products_.view(), subgraph.query_segments, subgraph.query_offsets,
                                   document_of_row_.view(), document_offsets_.view(), document_segments_.view(),
                                   score_grad, band_index * band, f32r2_view_t {maxima}, band_grad_.span());
        if (staged)
            staged = multiply_bf16(band_grad_.view(),
                                   packed_band_(packed_documents_transposed_, band_pack_transposed_bytes_, band_index),
                                   query_grad_products_.span(), numerics_, stream);
        if (staged)
            staged = numkong_result(numerics_.sum_f32(query_grad.data(), query_grad_products_.data(), queries * width,
                                                      query_grad.data(), stream.numkong()),
                                    "query force addition");
        if (staged) staged = transpose_bf16(band_grad_.view(), band_grad_transposed_.span(), stream);
        if (staged)
            staged = multiply_bf16(band_grad_transposed_.view(), packed_queries_transposed_.view(),
                                   document_grad_products_.span(), numerics_, stream);
        if (staged) {
            f32_t *const band_force = document_grad_.data() + band_index * band * width;
            staged = numkong_result(numerics_.sum_f32(band_force, document_grad_products_.data(), band * width,
                                                      band_force, stream.numkong()),
                                    "document force addition");
        }
    }
    return staged;
}

expected_t contrastive_tokens_t::relax(f32r4_span_t scores, contrastive_relations_view_t relations,
                                       contrastive_recipe_t recipe, u32r1_view_t world_document_lengths,
                                       f32r1_view_t live_queries, f32r1_span_t loss_out, gpu_stream_t &stream,
                                       f32r1_span_t log_inverse_temperature_grad, f32r1_span_t bias_grad) noexcept {
    relations.from_vertex_effective_length = query_lengths_.view();
    relations.to_vertex_effective_length = world_document_lengths;
    f32r3_span_t const planes = scores[0, nk::slice];
    dim3 const grid {grid_size_for(planes.numel()), 1u, 1u};
    dim3 const block_shape {kernel_block_size_default_k, 1u, 1u};
    expected_t staged = launch_kernel(&contrastive_tokens_scale_kernel_, grid, block_shape, stream, planes, recipe,
                                      1.0f);
    if (staged)
        staged = pair_loss_(scores, relations, recipe, live_queries, loss_out, stream, log_inverse_temperature_grad,
                            bias_grad);
    if (staged)
        staged = launch_kernel(&contrastive_tokens_scale_kernel_, grid, block_shape, stream, planes, recipe,
                               config_.weight);
    if (staged && config_.weight != 1.0f)
        staged = launch_kernel(&contrastive_tokens_weigh_kernel_, dim3 {1u, 1u, 1u}, block_shape, stream, loss_out,
                               log_inverse_temperature_grad, bias_grad, config_.weight);
    return staged;
}

expected_t contrastive_tokens_t::finish(f32r2_view_t query_grad, u32r1_view_t token_offsets,
                                        multimodal_batch_view_t const &batch, bf16r2_span_t token_vectors_grad,
                                        gpu_stream_t &stream) noexcept {
    if (token_vectors_grad.extent(0) != config_.head_tokens_per_gpu || token_vectors_grad.extent(1) != config_.width)
        return expected_t::fail(status_t::unsupported_k, "token_vectors_grad must be [{}x{}]",
                                config_.head_tokens_per_gpu, config_.width);
    return launch_kernel(&contrastive_tokens_finish_kernel_, dim3 {config_.head_tokens_per_gpu, 1u, 1u},
                         dim3 {kernel_block_size_default_k, 1u, 1u}, stream, batch, token_offsets,
                         query_offsets_.view(), query_grad, document_of_segment_.view(), document_offsets_.view(),
                         document_grad_.view(), token_vectors_grad);
}

expected_t contrastive_tokens_t::operator()(bf16r2_view_t token_vectors, u32r1_view_t token_offsets,
                                            multimodal_batch_view_t const &batch,
                                            contrastive_relations_view_t relations, contrastive_recipe_t recipe,
                                            f32r1_span_t loss_out, bf16r2_span_t token_vectors_grad,
                                            gpu_stream_t &stream, f32r1_span_t log_inverse_temperature_grad,
                                            f32r1_span_t bias_grad) noexcept {
    if (config_.gpu_subgraphs_per_step != 1)
        return expected_t::fail(status_t::invalid_argument_k, "a {}-subgraph tokens loss runs distributed",
                                config_.gpu_subgraphs_per_step);
    std::size_t const segments = config_.segments_per_gpu;
    f32r2_span_t const raw_scores = scores_.span().reshape<2>({segments, segments});
    expected_t staged = stage(token_vectors, token_offsets, batch, stream);
    if (staged) staged = score(0, query_subgraph(), raw_scores, stream);
    if (staged)
        staged = relax(scores_.span().reshape<4>({1u, 1u, segments, segments}), relations, recipe,
                       document_lengths_.view(), live_queries_.view(), loss_out, stream, log_inverse_temperature_grad,
                       bias_grad);
    if (staged) staged = force(0, query_subgraph(), f32r2_view_t {raw_scores}, query_grad_.span(), stream);
    if (staged) staged = finish(query_grad_.view(), token_offsets, batch, token_vectors_grad, stream);
    return staged;
}

expected<distributed_contrastive_tokens_t> distributed_contrastive_tokens_t::make(
    multimem_domain_t &domain, colocated_domain_t &colocated, std::span<gpu_arena_t> arenas,
    std::span<gpu_stream_t> streams, contrastive_tokens_config_t config, gpu_numerics_t const &numerics,
    f64_t timeout_seconds) noexcept {
    using result_t = expected<distributed_contrastive_tokens_t>;
    std::size_t const gpus = arenas.size();
    u32_t const domains = colocated.multimem_domain_count();
    if (gpus == 0 || streams.size() != gpus || domain.local_member_count() != gpus)
        return result_t::fail(status_t::invalid_argument_k, "every rank needs an arena, a stream and a local member");
    if (domain.member_count() != gpus || config.gpu_subgraphs_per_step != gpus * domains)
        return result_t::fail(status_t::unsupported_k,
                              "the tokens loss scores one query tile per GPU, every domain held whole by one " //
                              "process (subgraphs={} members={} local={} domains={})",
                              config.gpu_subgraphs_per_step, domain.member_count(), gpus, domains);
    for (std::size_t rank = 0; rank != gpus; ++rank)
        for (std::size_t peer = 0; peer != gpus; ++peer)
            if (!arenas[rank].is_visible_to(static_cast<int>(peer)))
                return result_t::fail(status_t::invalid_argument_k,
                                      "rank {}'s arena is not visible to device {}, construct it with `visible_to`",
                                      rank, peer);

    distributed_contrastive_tokens_t distributed;
    distributed.config_ = config;
    distributed.domain_ = &domain;
    distributed.colocated_ = &colocated;
    distributed.home_domain_index_ = domains > 1u ? colocated.topology().multimem_domain_index() : 0u;
    expected<multimem_window_t> window = domain.reserve(payload_scalars_needed(config));
    if (!window) return result_t::fail(std::move(window));
    distributed.window_ = window.value_;
    if (distributed.losses_.resize(gpus) != status_t::success_k ||
        distributed.home_scores_.resize(gpus) != status_t::success_k ||
        distributed.world_document_lengths_.resize(gpus) != status_t::success_k ||
        distributed.query_grads_.resize(gpus) != status_t::success_k ||
        distributed.world_scalars_.resize(gpus) != status_t::success_k)
        return result_t::fail(status_t::bad_alloc_k, "rank table allocation failed ({} ranks)", gpus);
    if (domains > 1u && (distributed.tile_gathers_.resize(gpus) != status_t::success_k ||
                         distributed.description_gathers_.resize(gpus) != status_t::success_k ||
                         distributed.score_gathers_.resize(gpus) != status_t::success_k ||
                         distributed.gradient_gathers_.resize(gpus) != status_t::success_k ||
                         distributed.force_reduces_.resize(gpus) != status_t::success_k ||
                         distributed.count_reduces_.resize(gpus) != status_t::success_k))
        return result_t::fail(status_t::bad_alloc_k, "cross-domain member table allocation failed ({} ranks)", gpus);
    std::size_t const subgraphs = config.gpu_subgraphs_per_step, segments = config.segments_per_gpu;
    std::size_t const tile_scalars = std::size_t {config.query_tokens_per_gpu} * config.width;
    std::size_t const score_scalars = subgraphs * segments * segments;
    for (std::size_t rank = 0; rank != gpus; ++rank) {
        switch_to_gpu_t gpu_scope {streams[rank]};
        if (!gpu_scope.ok()) return result_t::fail(status_t::device_error_k, "context entry failed (rank={})", rank);
        expected<contrastive_tokens_t> loss = contrastive_tokens_t::make(arenas[rank], config, numerics);
        if (!loss) return result_t::fail(std::move(loss));
        distributed.losses_[rank] = std::move(loss.value_);
        gpu_arena_alloc<f32_t> f32_alloc {arenas[rank]};
        distributed.home_scores_[rank] = f32r1_tensor_arena_t::uninitialized({score_scalars}, f32_alloc).value;
        distributed.world_document_lengths_[rank] =
            u32r1_tensor_arena_t::uninitialized({subgraphs * segments}, gpu_arena_alloc<u32_t> {arenas[rank]}).value;
        distributed.query_grads_[rank] =
            f32r2_tensor_arena_t::uninitialized({config.query_tokens_per_gpu, config.width}, f32_alloc).value;
        distributed.world_scalars_[rank] =
            f32r1_tensor_arena_t::uninitialized({world_scalar_count_(config)}, f32_alloc).value;
        if (!distributed.home_scores_[rank] || !distributed.world_document_lengths_[rank] ||
            !distributed.query_grads_[rank] || !distributed.world_scalars_[rank])
            return result_t::fail(status_t::bad_alloc_k, "tokens rank scratch allocation failed (rank={})", rank);
        if (domains == 1u) continue;
        // Every process wires its members in the same order, as each handle keys
        // its own rendezvous.
        u32_t const member = static_cast<u32_t>(domain.member_offset() + rank);
        auto tiles = infiniband_all_gather<bf16_t>::make(colocated, member, arenas[rank], tile_scalars, streams[rank],
                                                         timeout_seconds);
        if (!tiles) return result_t::fail(std::move(tiles));
        distributed.tile_gathers_[rank] = std::move(tiles.value_);
        auto descriptions = infiniband_all_gather<u32_t>::make(
            colocated, member, arenas[rank], distributed.description_scalars_(), streams[rank], timeout_seconds);
        if (!descriptions) return result_t::fail(std::move(descriptions));
        distributed.description_gathers_[rank] = std::move(descriptions.value_);
        auto scores = infiniband_all_gather<f32_t>::make(colocated, member, arenas[rank], score_scalars, streams[rank],
                                                         timeout_seconds);
        if (!scores) return result_t::fail(std::move(scores));
        distributed.score_gathers_[rank] = std::move(scores.value_);
        auto gradients = infiniband_all_gather<f32_t>::make(colocated, member, arenas[rank], score_scalars,
                                                            streams[rank], timeout_seconds);
        if (!gradients) return result_t::fail(std::move(gradients));
        distributed.gradient_gathers_[rank] = std::move(gradients.value_);
        auto forces = infiniband_all_reduce<f32_t>::make(colocated, member, arenas[rank],
                                                         domains * tile_scalars + world_scalar_count_(config),
                                                         streams[rank], timeout_seconds);
        if (!forces) return result_t::fail(std::move(forces));
        distributed.force_reduces_[rank] = std::move(forces.value_);
        auto count = infiniband_all_reduce<f32_t>::make(colocated, member, arenas[rank], 1u, streams[rank],
                                                        timeout_seconds);
        if (!count) return result_t::fail(std::move(count));
        distributed.count_reduces_[rank] = std::move(count.value_);
    }
    return result_t {std::move(distributed)};
}

f32r3_span_t distributed_contrastive_tokens_t::window_scores_(std::size_t member, std::size_t block) noexcept {
    std::size_t const subgraphs = config_.gpu_subgraphs_per_step, segments = config_.segments_per_gpu;
    std::size_t const scalars = subgraphs * segments * segments;
    f32r1_span_t const window = window_payload_(member);
    f32r2_span_t const blocks = window[nk::range(0, 2u * scalars), nk::slice].reshape<2>({2u, scalars});
    return blocks[block, nk::slice].reshape<3>({subgraphs, segments, segments});
}

f32r3_span_t distributed_contrastive_tokens_t::window_query_forces_(std::size_t member) noexcept {
    std::size_t const subgraphs = config_.gpu_subgraphs_per_step, segments = config_.segments_per_gpu;
    std::size_t const begin = 2u * subgraphs * segments * segments;
    std::size_t const scalars = subgraphs * std::size_t {config_.query_tokens_per_gpu} * config_.width;
    f32r1_span_t const window = window_payload_(member);
    return window[nk::range(begin, begin + scalars), nk::slice].reshape<3>(
        {subgraphs, std::size_t {config_.query_tokens_per_gpu}, std::size_t {config_.width}});
}

f32r1_span_t distributed_contrastive_tokens_t::window_tail_(std::size_t member) noexcept {
    std::size_t const tail_begin = window_.scalars - world_scalar_count_(config_);
    f32r1_span_t const window = window_payload_(member);
    return window[nk::range(tail_begin, window_.scalars), nk::slice];
}

gpu_query_subgraph_t distributed_contrastive_tokens_t::world_subgraph_(std::size_t world_gpu) noexcept {
    std::size_t const members = losses_.size(), member = world_gpu % members, domain = world_gpu / members;
    if (domain == home_domain_index_) return losses_[member].query_subgraph();
    std::size_t const queries = config_.query_tokens_per_gpu, segments = config_.segments_per_gpu;
    std::size_t const tile_scalars = queries * config_.width, description = description_scalars_();
    bf16r2_span_t const tiles = tile_gathers_[member].payload().reshape<2>({world_domains_(), tile_scalars});
    u32r2_span_t const descriptions = description_gathers_[member].payload().reshape<2>(
        {world_domains_(), description});
    u32r1_span_t const described = descriptions[domain, nk::slice];
    gpu_query_subgraph_t subgraph;
    subgraph.query_vectors = tiles[domain, nk::slice].reshape<2>({queries, std::size_t {config_.width}});
    subgraph.query_segments = described[nk::range(0, queries), nk::slice];
    subgraph.query_offsets = described[nk::range(queries, queries + segments + 1u), nk::slice];
    return subgraph;
}

u32r1_view_t distributed_contrastive_tokens_t::world_document_lengths_of_(std::size_t world_gpu) noexcept {
    std::size_t const members = losses_.size(), member = world_gpu % members, domain = world_gpu / members;
    if (domain == home_domain_index_) return losses_[member].document_lengths();
    std::size_t const segments = config_.segments_per_gpu;
    std::size_t const lengths_begin = std::size_t {config_.query_tokens_per_gpu} + segments + 1u;
    u32r2_span_t const descriptions = description_gathers_[member].payload().reshape<2>(
        {world_domains_(), description_scalars_()});
    u32r1_span_t const described = descriptions[domain, nk::slice];
    return described[nk::range(lengths_begin, lengths_begin + segments), nk::slice];
}

f32r3_view_t distributed_contrastive_tokens_t::world_scores_(std::size_t world_gpu, std::size_t block) noexcept {
    std::size_t const members = losses_.size(), member = world_gpu % members, domain = world_gpu / members;
    if (domain == home_domain_index_) return window_scores_(domain_->member_offset() + member, block);
    std::size_t const subgraphs = config_.gpu_subgraphs_per_step, segments = config_.segments_per_gpu;
    infiniband_all_gather<f32_t> &gathered = block == 0u ? score_gathers_[member] : gradient_gathers_[member];
    f32r2_span_t const blocks = gathered.payload().reshape<2>({world_domains_(), subgraphs * segments * segments});
    return blocks[domain, nk::slice].reshape<3>({subgraphs, segments, segments});
}

expected_t distributed_contrastive_tokens_t::operator()(std::span<gpu_tokens_inputs_t const> inputs,
                                                        std::span<gpu_stream_t> streams) noexcept {
    std::size_t const gpus = losses_.size();
    if (inputs.size() != gpus || streams.size() != gpus)
        return expected_t::fail(status_t::invalid_argument_k, "every rank needs its inputs and stream ({} ranks)",
                                gpus);
    for (gpu_tokens_inputs_t const &rank_inputs : inputs)
        if (rank_inputs.recipe.temperature_group_count() > config_.temperature_groups)
            return expected_t::fail(status_t::unsupported_k, "a recipe selects {} temperature groups of {}",
                                    rank_inputs.recipe.temperature_group_count(), config_.temperature_groups);
    // Phase-major: every rank's phase is enqueued before any rank's next one, so a rank spinning at
    // a barrier never holds the host while its peers' arrivals wait unenqueued behind it.
    using phase_t = expected_t (distributed_contrastive_tokens_t::*)(std::size_t, gpu_tokens_inputs_t const &,
                                                                     gpu_stream_t &) noexcept;
    phase_t const phases[] = {&distributed_contrastive_tokens_t::stage_, &distributed_contrastive_tokens_t::score_,
                              &distributed_contrastive_tokens_t::relax_, &distributed_contrastive_tokens_t::force_,
                              &distributed_contrastive_tokens_t::reduce_};
    for (phase_t const phase : phases)
        for (std::size_t rank = 0; rank != gpus; ++rank) {
            switch_to_gpu_t gpu_scope {streams[rank]};
            if (!gpu_scope.ok()) return expected_t::fail(status_t::device_error_k, "context entry failed ({})", rank);
            std::size_t const member = domain_->member_offset() + rank;
            if (expected_t staged = (this->*phase)(rank, inputs[rank], streams[rank]); !staged) return staged;
            if (expected_t met = domain_->barrier(member, streams[rank]); !met) return met;
        }
    return {};
}

expected_t distributed_contrastive_tokens_t::stage_(std::size_t rank, gpu_tokens_inputs_t const &inputs,
                                                    gpu_stream_t &stream) noexcept {
    std::size_t const gradients = 2u * std::size_t {config_.temperature_groups};
    f32r1_span_t const tail = window_tail_(domain_->member_offset() + rank);
    contrastive_tokens_t &loss = losses_[rank];
    expected_t staged = loss.stage(inputs.token_vectors, inputs.token_offsets, inputs.batch, stream);
    if (staged)
        staged = copy_device_to_device(loss.live_queries(), tail[nk::range(gradients + 1u, gradients + 2u), nk::slice],
                                       stream);
    if (!staged || world_domains_() == 1u) return staged;
    std::size_t const queries = config_.query_tokens_per_gpu, segments = config_.segments_per_gpu;
    std::size_t const lengths_begin = queries + segments + 1u;
    gpu_query_subgraph_t const own = loss.query_subgraph();
    u32r1_span_t const description = description_gathers_[rank].own_block();
    staged = copy_device_to_device(own.query_vectors.flatten<1>(), tile_gathers_[rank].own_block(), stream);
    if (staged)
        staged = copy_device_to_device(own.query_segments, description[nk::range(0, queries), nk::slice], stream);
    if (staged)
        staged = copy_device_to_device(own.query_offsets,
                                       description[nk::range(queries, queries + segments + 1u), nk::slice], stream);
    if (staged)
        staged = copy_device_to_device(loss.document_lengths(),
                                       description[nk::range(lengths_begin, lengths_begin + segments), nk::slice],
                                       stream);
    if (staged) staged = tile_gathers_[rank](stream);
    if (staged) staged = description_gathers_[rank](stream);
    return staged;
}

expected_t distributed_contrastive_tokens_t::score_(std::size_t rank, gpu_tokens_inputs_t const &,
                                                    gpu_stream_t &stream) noexcept {
    std::size_t const subgraphs = config_.gpu_subgraphs_per_step, segments = config_.segments_per_gpu;
    std::size_t const gradients = 2u * std::size_t {config_.temperature_groups};
    std::size_t const member = domain_->member_offset() + rank;
    std::size_t const tail_offset = window_.offset + window_.scalars - world_scalar_count_(config_);
    contrastive_tokens_t &loss = losses_[rank];
    f32r3_span_t const outbound = window_scores_(member, 0);
    expected_t staged {};
    for (std::size_t home = 0; staged && home != subgraphs; ++home)
        staged = loss.score(home, world_subgraph_(home), outbound[home, nk::slice], stream);
    u32r2_span_t const world_document_lengths = world_document_lengths_[rank].span().reshape<2>({subgraphs, segments});
    for (std::size_t world_gpu = 0; staged && world_gpu != subgraphs; ++world_gpu)
        staged = copy_device_to_device(world_document_lengths_of_(world_gpu),
                                       world_document_lengths[world_gpu, nk::slice], stream);
    f32r1_span_t const world_scalars = world_scalars_[rank].span();
    f32r1_span_t const world_live_queries = world_scalars[nk::range(gradients + 1u, gradients + 2u), nk::slice];
    if (staged) staged = domain_->reduce_scatter(member, world_live_queries, tail_offset + gradients + 1u, stream);
    if (!staged || world_domains_() == 1u) return staged;
    staged = copy_device_to_device(f32r1_view_t {world_live_queries}, count_reduces_[rank].payload(), stream);
    if (staged) staged = count_reduces_[rank](stream);
    if (staged)
        staged = copy_device_to_device(f32r1_view_t {count_reduces_[rank].payload()}, world_live_queries, stream);
    if (staged)
        staged = copy_device_to_device(f32r1_view_t {outbound.flatten<1>()}, score_gathers_[rank].own_block(), stream);
    if (staged) staged = score_gathers_[rank](stream);
    return staged;
}

expected_t distributed_contrastive_tokens_t::relax_(std::size_t rank, gpu_tokens_inputs_t const &inputs,
                                                    gpu_stream_t &stream) noexcept {
    std::size_t const subgraphs = config_.gpu_subgraphs_per_step, segments = config_.segments_per_gpu;
    std::size_t const groups = config_.temperature_groups, gradients = 2u * groups;
    std::size_t const member = domain_->member_offset() + rank;
    std::size_t const own_world_gpu = home_domain_index_ * losses_.size() + rank;
    f32r1_span_t const tail = window_tail_(member);
    f32r1_span_t const world_scalars = world_scalars_[rank].span();
    f32r3_span_t const home_scores = home_scores_[rank].span().reshape<3>({subgraphs, segments, segments});
    expected_t staged {};
    for (std::size_t document_gpu = 0; staged && document_gpu != subgraphs; ++document_gpu) {
        f32r3_view_t const outbound = world_scores_(document_gpu, 0);
        staged = copy_device_to_device(outbound[own_world_gpu, nk::slice].flatten<1>(),
                                       home_scores[document_gpu, nk::slice].flatten<1>(), stream);
    }
    if (staged)
        staged = losses_[rank].relax(home_scores_[rank].span().reshape<4>({1u, subgraphs, segments, segments}),
                                     inputs.relations, inputs.recipe, world_document_lengths_[rank].view(),
                                     f32r1_view_t {world_scalars[nk::range(gradients + 1u, gradients + 2u), nk::slice]},
                                     tail[nk::range(gradients, gradients + 1u), nk::slice], stream,
                                     tail[nk::range(0, groups), nk::slice],
                                     tail[nk::range(groups, gradients), nk::slice]);
    f32r3_span_t const inbound = window_scores_(member, 1);
    if (staged) staged = copy_device_to_device(home_scores_[rank].view(), inbound.flatten<1>(), stream);
    if (!staged || world_domains_() == 1u) return staged;
    staged = copy_device_to_device(f32r1_view_t {inbound.flatten<1>()}, gradient_gathers_[rank].own_block(), stream);
    if (staged) staged = gradient_gathers_[rank](stream);
    return staged;
}

expected_t distributed_contrastive_tokens_t::force_(std::size_t rank, gpu_tokens_inputs_t const &,
                                                    gpu_stream_t &stream) noexcept {
    std::size_t const subgraphs = config_.gpu_subgraphs_per_step;
    std::size_t const own_world_gpu = home_domain_index_ * losses_.size() + rank;
    f32r3_span_t const query_forces = window_query_forces_(domain_->member_offset() + rank);
    expected_t staged {};
    for (std::size_t home = 0; staged && home != subgraphs; ++home) {
        f32r3_view_t const home_gradients = world_scores_(home, 1);
        staged = losses_[rank].force(home, world_subgraph_(home), home_gradients[own_world_gpu, nk::slice],
                                     query_forces[home, nk::slice], stream);
    }
    return staged;
}

expected_t distributed_contrastive_tokens_t::reduce_(std::size_t rank, gpu_tokens_inputs_t const &inputs,
                                                     gpu_stream_t &stream) noexcept {
    std::size_t const members = losses_.size(), segments = config_.segments_per_gpu;
    std::size_t const subgraphs = config_.gpu_subgraphs_per_step;
    std::size_t const gradients = 2u * std::size_t {config_.temperature_groups};
    std::size_t const member = domain_->member_offset() + rank;
    std::size_t const forces_offset = window_.offset + 2u * subgraphs * segments * segments;
    std::size_t const force_scalars = std::size_t {config_.query_tokens_per_gpu} * config_.width;
    std::size_t const tail_offset = window_.offset + window_.scalars - world_scalar_count_(config_);
    f32r1_span_t const world_scalars = world_scalars_[rank].span();
    f32r1_span_t const query_grad = query_grads_[rank].span().flatten<1>();
    expected_t staged {};
    if (world_domains_() == 1u) {
        staged = domain_->reduce_scatter(member, query_grad, forces_offset + rank * force_scalars, stream);
        if (staged)
            staged = domain_->reduce_scatter(member, world_scalars[nk::range(0, gradients + 1u), nk::slice],
                                             tail_offset, stream);
    }
    else {
        // Level 1 sums this member's row of every domain's query tiles, level 2 sums
        // them across domains.
        f32r1_span_t const shard = force_reduces_[rank].payload();
        std::size_t const world_forces = world_domains_() * force_scalars;
        f32r2_span_t const shard_rows = shard[nk::range(0, world_forces), nk::slice].reshape<2>(
            {std::size_t {world_domains_()}, force_scalars});
        for (u32_t domain = 0; staged && domain != world_domains_(); ++domain)
            staged = domain_->reduce_scatter(member, shard_rows[domain, nk::slice],
                                             forces_offset + (domain * members + rank) * force_scalars, stream);
        f32r1_span_t const scalars_shard = shard[nk::range(world_forces, world_forces + gradients + 1u), nk::slice];
        if (staged) staged = domain_->reduce_scatter(member, scalars_shard, tail_offset, stream);
        if (staged) staged = force_reduces_[rank](stream);
        if (staged)
            staged = copy_device_to_device(f32r1_view_t {shard_rows[home_domain_index_, nk::slice]}, query_grad,
                                           stream);
        if (staged)
            staged = copy_device_to_device(f32r1_view_t {scalars_shard},
                                           world_scalars[nk::range(0, gradients + 1u), nk::slice], stream);
    }
    if (staged)
        staged = losses_[rank].finish(query_grads_[rank].view(), inputs.token_offsets, inputs.batch,
                                      inputs.token_vectors_grad, stream);
    return staged;
}
