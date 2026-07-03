/**
 *  @brief The batch mixer's implementation: the routing bijection and the two exchange kernels —
 *      compiled once into the `uform` library. Everything stays on the GPU: peer addresses travel as
 *      kernel arguments, and every kernel re-derives its routes in-register from the keyed bijection,
 *      so the mixer owns no memory at all.
 *  @file cuda_core/multimodal_batch.cu
 *  @author Ash Vardanian
 *  @date July 3, 2026
 */
#include "multimodal_batch.cuh"

namespace unum::uform::core {

/** @brief One member's batch addresses — the gather kernels' view of a peer. */
struct batch_source_t_ {
    token_id_t const *tokens = nullptr;
    u32_t const *segment_offsets = nullptr;
    u32_t const *segment_lengths = nullptr;
};

/** @brief Every member's addresses, passed by value as a kernel argument — @ref max_gpus_per_host_k
 *      bounds the table, so no device-resident copy and no staging exist. */
struct batch_sources_t_ {
    batch_source_t_ entries[max_gpus_per_host_k] {};
    u32_t count = 0;
};

/** @brief One keyed mixing round of the routing bijection — the epoch shuffle's construction. */
[[nodiscard]] static constexpr __host__ __device__ u32_t mixing_round_(u32_t half, u32_t key) noexcept {
    u32_t mixed = half * 2654435761u + key;
    mixed ^= mixed >> 15;
    mixed *= 2246822519u;
    mixed ^= mixed >> 13;
    return mixed;
}

/** @brief The world segment slot that feeds destination slot @p destination at @p epoch — a keyed
 *      Feistel bijection over @c [0,count) , cycle-walked back into range; identical on every GPU of
 *      every host_domain, so routing needs no exchange. Gather-oriented: destinations look up their
 *      sources, which is equivalent to scattering by the inverse permutation. */
[[nodiscard]] static constexpr __host__ __device__ std::size_t segment_source_(u64_t seed, u32_t epoch,
                                                                               std::size_t destination,
                                                                               std::size_t count) noexcept {
    u32_t const half_bits = (ceil_log2(static_cast<u32_t>(count)) + 1u) / 2u;
    u32_t const half_mask = (1u << half_bits) - 1u;
    std::size_t value = destination;
    do {
        u32_t left = static_cast<u32_t>(value >> half_bits) & half_mask;
        u32_t right = static_cast<u32_t>(value) & half_mask;
        for (u32_t round = 0; round != 4; ++round) {
            u32_t const key = static_cast<u32_t>(seed >> (16 * (round % 4))) + epoch * 40503u + round;
            u32_t const swapped = right;
            right = left ^ (mixing_round_(right, key) & half_mask);
            left = swapped;
        }
        value = (static_cast<std::size_t>(left) << half_bits) | right;
    } while (value >= count);
    return value;
}

/**
 *  @brief Write the mixed batch's geometry: thread 0 walks the destination slots serially — the
 *      repack cursor is a strict scan — deriving each slot's world source through
 *      @ref segment_source_ , reading its length from the peer's table, and jumping the cursor past
 *      any row boundary the segment would cross. A segment that would overflow the destination
 *      rectangle is dropped loudly (zero length) — the packer's budgets make that a bug, not a case.
 *
 *  __grid_size__ @c [1,1,1] , one thread live; no dynamic shared memory.
 */
static __global__ void batch_geometry_kernel_(batch_sources_t_ sources, u32r1_span_t mixed_offsets,
                                              u32r1_span_t mixed_lengths, u32_t member_index, u32_t epoch,
                                              u64_t seed, u32_t segments_per_gpu, u32_t rows_per_gpu,
                                              u32_t columns_per_row) {
    if (threadIdx.x != 0u || blockIdx.x != 0u) return;
    std::size_t const world_segments = static_cast<std::size_t>(segments_per_gpu) * sources.count;
    u32_t const capacity = rows_per_gpu * columns_per_row;
    u32_t cursor = 0;
    for (u32_t slot = 0; slot != segments_per_gpu; ++slot) {
        std::size_t const destination = static_cast<std::size_t>(member_index) * segments_per_gpu + slot;
        std::size_t const source = segment_source_(seed, epoch, destination, world_segments);
        batch_source_t_ const &peer = sources.entries[source / segments_per_gpu];
        u32_t columns = peer.segment_lengths[source % segments_per_gpu];
        // A segment never crosses its row: jump the cursor to the next row when it would.
        u32_t const room_in_row = columns_per_row - cursor % columns_per_row;
        if (columns > room_in_row) cursor += room_in_row;
        if (cursor + columns > capacity) {
            printf("uform: mixed batch overflows at slot %u (member %u) — segment dropped\n", slot, member_index);
            columns = 0;
        }
        mixed_offsets[slot] = cursor;
        mixed_lengths[slot] = columns;
        cursor += columns;
    }
    mixed_offsets[segments_per_gpu] = cursor;
}

/**
 *  @brief Gather one routed segment: CTA @c s re-derives route @c s 's world source and copies its
 *      token span from the peer member's loaded batch into this member's mixed batch, threads
 *      striding the columns. The destination geometry is what @ref batch_geometry_kernel_ wrote.
 *
 *  __grid_size__ @c [segments_per_gpu,1,1] , threads stripe columns; no dynamic shared memory.
 */
static __global__ void batch_gather_kernel_(batch_sources_t_ sources, u32r1_view_t mixed_offsets,
                                            u32r1_view_t mixed_lengths, tokens_span_t mixed_tokens,
                                            u32_t member_index, u32_t epoch, u64_t seed,
                                            u32_t segments_per_gpu) {
    u32_t const slot = blockIdx.x;
    u32_t const columns = mixed_lengths[slot];
    if (columns == 0u) return;
    std::size_t const world_segments = static_cast<std::size_t>(segments_per_gpu) * sources.count;
    std::size_t const destination = static_cast<std::size_t>(member_index) * segments_per_gpu + slot;
    std::size_t const source = segment_source_(seed, epoch, destination, world_segments);
    batch_source_t_ const &peer = sources.entries[source / segments_per_gpu];
    u32_t const source_begin = peer.segment_offsets[source % segments_per_gpu];
    u32_t const destination_begin = mixed_offsets[slot];
    for (u32_t column = threadIdx.x; column < columns; column += blockDim.x)
        mixed_tokens[destination_begin + column] = peer.tokens[source_begin + column];
}

expected<multimodal_batch_mixer_t> multimodal_batch_mixer_t::make(multimem_domain_t &multimem,
                                                                  colocated_domain_t &colocated,
                                                                  per_gpu_array<gpu_arena_t> &arenas,
                                                                  multimodal_batch_shape_t shape,
                                                                  u64_t seed) noexcept {
    using result_t = expected<multimodal_batch_mixer_t>;
    std::size_t const members = multimem.member_count();
    if (members == 0 || arenas.size() != members)
        return result_t::fail(status_t::invalid_input_k,
                              "every multimem_domain member needs its arena ({} members, {} arenas)", members,
                              arenas.size());
    if (members > max_gpus_per_host_k)
        return result_t::fail(status_t::shape_unsupported_k, "{} members exceed max_gpus_per_host_k ({})", members,
                              max_gpus_per_host_k);
    if (colocated.multimem_domain_count() != 1u)
        return result_t::fail(status_t::shape_unsupported_k,
                              "cross-host_domain mixing lands with the two-node tier ({} multimem_domains)",
                              colocated.multimem_domain_count());
    // Level-1 precondition: every member's arena readable by every other member — visibility is the
    // arenas' construction-time configuration, validated here, never granted here.
    for (std::size_t member = 0; member != members; ++member)
        for (std::size_t peer = 0; peer != members; ++peer)
            if (!arenas[member].is_visible_to(static_cast<int>(peer)))
                return result_t::fail(status_t::invalid_input_k,
                                      "member {}'s arena is not visible to device {} — " //
                                      "construct it with `visible_to`",
                                      member, peer);
    multimodal_batch_mixer_t mixer;
    mixer.multimem_ = &multimem;
    mixer.seed_ = seed;
    mixer.shape_ = shape;
    return result_t {std::move(mixer)};
}

expected_t multimodal_batch_mixer_t::operator()(std::size_t member_index, epoch_t epoch,
                                                per_gpu_array<multimodal_batch_on_gpu_t> const &loaded,
                                                multimodal_batch_on_gpu_t &mixed,
                                                gpu_stream_t const &stream) noexcept {
    std::size_t const members = multimem_->member_count();
    if (member_index >= members)
        return expected_t::fail(status_t::invalid_input_k, "member {} of {}", member_index, members);
    if (loaded.size() != members)
        return expected_t::fail(status_t::invalid_input_k, "every member needs its loaded batch ({} of {})",
                                loaded.size(), members);
    switch_to_gpu_t gpu_scope {stream};
    if (!gpu_scope.ok())
        return expected_t::fail(status_t::cuda_error_k, "context entry failed ({})", member_index);

    // The peer table travels as a kernel argument — no device copy, no staging, nothing owned.
    batch_sources_t_ sources;
    sources.count = static_cast<u32_t>(members);
    for (std::size_t member = 0; member != members; ++member) {
        batch_source_t_ source;
        source.tokens = loaded[member].tokens.data();
        source.segment_offsets = loaded[member].segment_offsets.data();
        source.segment_lengths = loaded[member].segment_lengths.data();
        if (source.tokens == nullptr || source.segment_offsets == nullptr || source.segment_lengths == nullptr)
            return expected_t::fail(status_t::invalid_input_k, "member {}'s loaded batch is unallocated", member);
        if (source.tokens == mixed.tokens.data())
            return expected_t::fail(status_t::invalid_input_k,
                                    "loaded and mixed batches must be distinct storage (member={})", member);
        sources.entries[member] = source;
    }

    // The shared phase rides the lead member's slice — see the ordering contract: this consumer's
    // advance/barrier bracket is the step's first, ordering "everyone loaded" before any peer read.
    if (member_index == 0)
        if (expected_t advanced = multimem_->advance(stream); !advanced) return advanced;
    if (expected_t arrived = multimem_->barrier(member_index, stream); !arrived) return arrived;

    u32r1_span_t const mixed_offsets {mixed.segment_offsets.data(), mixed.segment_offsets.extent(0)};
    u32r1_span_t const mixed_lengths {mixed.segment_lengths.data(), mixed.segment_lengths.extent(0)};
    tokens_span_t const mixed_tokens {mixed.tokens.data(), static_cast<std::size_t>(shape_.max_rows_per_gpu) *
                                                               shape_.max_columns_per_row};
    dim3 const one_block {1u, 1u, 1u};
    dim3 const block_shape {kernel_block_size_default_k, 1u, 1u};
    dim3 const segment_grid {shape_.max_segments_per_gpu, 1u, 1u};
    expected_t staged = launch_kernel(reinterpret_cast<void const *>(&batch_geometry_kernel_), one_block,
                                      block_shape, stream, sources, mixed_offsets, mixed_lengths,
                                      static_cast<u32_t>(member_index), epoch.index, seed_,
                                      shape_.max_segments_per_gpu, shape_.max_rows_per_gpu,
                                      shape_.max_columns_per_row);
    if (staged)
        staged = launch_kernel(reinterpret_cast<void const *>(&batch_gather_kernel_), segment_grid, block_shape,
                               stream, sources, u32r1_view_t {mixed_offsets.data(), mixed_offsets.extent(0)},
                               u32r1_view_t {mixed_lengths.data(), mixed_lengths.extent(0)}, mixed_tokens,
                               static_cast<u32_t>(member_index), epoch.index, seed_,
                               shape_.max_segments_per_gpu);
    return staged;
}

expected_t multimodal_batch_mixer_t::operator()(epoch_t epoch,
                                                per_gpu_array<multimodal_batch_on_gpu_t> const &loaded,
                                                per_gpu_array<multimodal_batch_on_gpu_t> &mixed,
                                                per_gpu_array<gpu_stream_t> const &streams) noexcept {
    std::size_t const members = multimem_->member_count();
    if (streams.size() != members || mixed.size() != members)
        return expected_t::fail(status_t::invalid_input_k,
                                "every member needs its stream and mixed batch ({} members)", members);
    for (std::size_t member_index = 0; member_index != members; ++member_index)
        if (expected_t staged = (*this)(member_index, epoch, loaded, mixed[member_index], streams[member_index]);
            !staged)
            return staged;
    return {};
}

} // namespace unum::uform::core
