/**
 *  @brief Conformance suite for @ref multimodal_batch_mixer_t — the cross-GPU segment shuffle.
 *  @file test/mixer.cu
 *  @author Ash Vardanian
 *  @date July 6, 2026
 *
 *  Builds the composition root ( @ref mesh_topology_t::bootstrap → per-GPU arenas / streams →
 *  @ref multimem_domain_t → @ref colocated_domain_t → the mixer), fabricates every member's loaded batch
 *  with a token signature distinct per @c (gpu_index,segment,column) — the world coordinate — runs one
 *  mix, and asserts each mixed segment equals its @ref epoch_permutation source: the routing bijection,
 *  the length / modality descriptor travel, and the token content. One process per host_domain under the same
 *  @c UFORM_HOST_DOMAIN_COUNT contract as the base suite: one node exercises the NVLink multimem_domain
 *  alone, two nodes add the InfiniBand full-batch exchange to same-index landings.
 */
#include <cstddef> // `std::size_t`
#include <cstdlib> // `::setenv`, `std::getenv`, `std::atoi` — eager module loading, the host_domain contract

#include <utility>   // `std::move`
#include <algorithm> // `std::min`
#include <span>      // `std::span`

#include <cuda.h> // `cuInit`, `cuDeviceGetCount`, `cuStreamSynchronize`

#include "core/types.hpp"                  // `epoch_permutation`, `data_epoch_t`, `token_id_t`, `modality_t`
#include "core/array.hpp"                  // `array`
#include "core/gpu_allocators.hpp"         // `gpu_arena_t`, `gpu_arena_alloc_t`, `pinned_alloc`
#include "core/schedule.cuh"               // `gpu_stream_t`, `switch_to_gpu_t`, `copy_host_to_device`
#include "core/mesh.hpp"                   // `mesh_topology_t`, `host_domain_index_from_environment`
#include "core/multimem_collectives.cuh"   // `multimem_domain_t`
#include "core/infiniband_collectives.cuh" // `colocated_domain_t`
#include "core/multimodal_batch.cuh"       // `multimodal_batch_on_gpu_t`, the shape, the mixer
#include "harness.hpp"                     // `test_suite_t`

using namespace unum::uform::core;
using namespace unum::uform::test;

/** Segments per member — the routing axis; small, so the whole world permutation is easy to mirror. */
inline constexpr u32_t segments_per_member_k = 8u;

/** Tokens per segment — a short run whose per-column signature the oracle reproduces exactly. */
inline constexpr u32_t tokens_per_segment_k = 3u;

/** Members to exercise on one box — one multimem_domain's worth, capped by the visible device count. */
inline constexpr u32_t members_cap_k = 8u;

/** A rank-1 pinned host tensor of @p value_type_ — the sanctioned host staging / readback buffer. */
template <typename value_type_>
using pinned1_t = nk::tensor<value_type_, pinned_alloc<value_type_>, 1>;

/** The deterministic token a member's segment writes at a column — distinct per @c (member,segment,column)
 *  so a mislaid route mismatches. Wraps into @ref token_id_t ; the oracle wraps identically. */
[[nodiscard]] static token_id_t signature_token_(u32_t member, u32_t segment, u32_t column) noexcept {
    return static_cast<token_id_t>((member * 131u + segment) * 17u + column + 1u);
}

/**
 *  @brief Fabricate member @p member 's loaded batch: @ref segments_per_member_k text segments, each
 *      @ref tokens_per_segment_k columns, packed contiguously into row 0, tokens carrying @ref signature_token_ .
 *      Staged host → device through the pinned + @ref copy_host_to_device path, synchronized before returning.
 *  @param[inout] batch The member's loaded batch, its geometry and tokens overwritten.
 *  @param[in] member This member's world coordinate ( @c gpu_index_in_colocated_domain ), folded into every
 *      token signature.
 *  @param[inout] stream The member's stream the staging enqueues on.
 */
[[nodiscard]] static expected_t fabricate_loaded_(multimodal_batch_on_gpu_t &batch, u32_t member,
                                                  gpu_stream_t &stream) noexcept {
    u32_t const S = segments_per_member_k, L = tokens_per_segment_k;
    u32_t const rows = static_cast<u32_t>(batch.tokens.extent(0));
    u32_t const columns = static_cast<u32_t>(batch.tokens.extent(1));

    switch_to_gpu_t gpu_scope {stream};
    if (!gpu_scope.ok()) return expected_t::fail(status_t::cuda_error_k, "context entry failed for member {}", member);

    auto offsets = pinned1_t<u32_t>::zeros({static_cast<std::size_t>(S) + 1u}, pinned_alloc<u32_t> {}).value;
    auto lengths = pinned1_t<u32_t>::zeros({S}, pinned_alloc<u32_t> {}).value;
    auto modality = pinned1_t<modality_t>::zeros({S}, pinned_alloc<modality_t> {}).value;
    auto tokens =
        pinned1_t<token_id_t>::zeros({static_cast<std::size_t>(rows) * columns}, pinned_alloc<token_id_t> {}).value;
    if (!offsets || !lengths || !modality || !tokens)
        return expected_t::fail(status_t::bad_alloc_k, "loaded fabrication scratch alloc failed");

    for (u32_t s = 0; s != S; ++s) {
        offsets[s] = s * L;
        lengths[s] = L;
        modality[s] = modality_t::text_k;
        for (u32_t j = 0; j != L; ++j) tokens[s * L + j] = signature_token_(member, s, j);
    }
    offsets[S] = S * L;

    if (expected_t staged = copy_host_to_device(offsets.view(), batch.segment_offsets.span(), stream); !staged)
        return staged;
    if (expected_t staged = copy_host_to_device(lengths.view(), batch.segment_lengths.span(), stream); !staged)
        return staged;
    if (expected_t staged = copy_host_to_device(modality.view(), batch.segment_modality.span(), stream); !staged)
        return staged;
    if (expected_t staged = copy_host_to_device(tokens.view(), batch.tokens.span().flatten<1>(), stream); !staged)
        return staged;
    if (CUresult synced = cuStreamSynchronize(stream.get()); synced != CUDA_SUCCESS)
        return cuda_failure(synced, "loaded fabrication sync failed for member {}", member);
    return {};
}

/**
 *  @brief Read member @p member 's mixed batch back and assert every segment equals its @ref epoch_permutation
 *      source: destination @c member·S+s pulls world source @c src , whose member is @c src/S and segment
 *      @c src%S ; the mixed segment must carry that source's length, its @c text_k modality, and its tokens.
 *  @param[inout] suite The conformance suite the checks report into.
 *  @param[inout] mixed The member's mixed batch to read back.
 *  @param[in] member This member's world coordinate ( @c gpu_index_in_colocated_domain ) — its destination
 *      range's base.
 *  @param[in] member_count The world's member count — the permutation's segment span is @c S·member_count .
 *  @param[in] seed The routing seed the mixer was built with.
 *  @param[in] epoch The data epoch the mix ran at.
 *  @param[inout] stream The member's stream the readback enqueues on.
 */
[[nodiscard]] static expected_t verify_mixed_(test_suite_t &suite, multimodal_batch_on_gpu_t &mixed, u32_t member,
                                              u32_t member_count, u64_t seed, data_epoch_t epoch,
                                              gpu_stream_t &stream) noexcept {
    u32_t const S = segments_per_member_k, L = tokens_per_segment_k;
    u32_t const rows = static_cast<u32_t>(mixed.tokens.extent(0));
    u32_t const columns = static_cast<u32_t>(mixed.tokens.extent(1));

    switch_to_gpu_t gpu_scope {stream};
    if (!gpu_scope.ok()) return expected_t::fail(status_t::cuda_error_k, "context entry failed for member {}", member);

    auto offsets = pinned1_t<u32_t>::zeros({static_cast<std::size_t>(S) + 1u}, pinned_alloc<u32_t> {}).value;
    auto lengths = pinned1_t<u32_t>::zeros({S}, pinned_alloc<u32_t> {}).value;
    auto modality = pinned1_t<modality_t>::zeros({S}, pinned_alloc<modality_t> {}).value;
    auto tokens =
        pinned1_t<token_id_t>::zeros({static_cast<std::size_t>(rows) * columns}, pinned_alloc<token_id_t> {}).value;
    if (!offsets || !lengths || !modality || !tokens)
        return expected_t::fail(status_t::bad_alloc_k, "mixed readback scratch alloc failed");

    if (expected_t read = copy_device_to_host(mixed.segment_offsets.view(), offsets.span(), stream); !read) return read;
    if (expected_t read = copy_device_to_host(mixed.segment_lengths.view(), lengths.span(), stream); !read) return read;
    if (expected_t read = copy_device_to_host(mixed.segment_modality.view(), modality.span(), stream); !read)
        return read;
    if (expected_t read = copy_device_to_host(mixed.tokens.view().flatten<1>(), tokens.span(), stream); !read)
        return read;
    if (CUresult synced = cuStreamSynchronize(stream.get()); synced != CUDA_SUCCESS)
        return cuda_failure(synced, "mixed readback sync failed for member {}", member);

    std::size_t const world = static_cast<std::size_t>(S) * member_count;
    for (u32_t s = 0; s != S; ++s) {
        std::size_t const destination = static_cast<std::size_t>(member) * S + s;
        std::size_t const source = epoch_permutation(seed, epoch, destination, world);
        u32_t const source_member = static_cast<u32_t>(source / S);
        u32_t const source_segment = static_cast<u32_t>(source % S);
        suite.check(lengths[s] == L, "member {} segment {}: mixed length {} != source length {}", member, s, lengths[s],
                    L);
        suite.check(modality[s] == modality_t::text_k, "member {} segment {}: mixed modality is not text", member, s);
        u32_t const offset = offsets[s];
        for (u32_t j = 0; j != L; ++j) {
            token_id_t const got = tokens[offset + j];
            token_id_t const want = signature_token_(source_member, source_segment, j);
            suite.check(got == want, "member {} segment {} column {}: token {} != source ({},{}) token {}", member, s,
                        j, got, source_member, source_segment, want);
        }
    }
    return {};
}

/** Assert the routing is a bijection over the world's @c S·member_count segments — every destination maps
 *  to a distinct source — mirroring @ref epoch_permutation host-side so a broken permutation is caught even
 *  before the device gather. */
static void routing_is_a_bijection(test_suite_t &suite, u32_t member_count, u64_t seed, data_epoch_t epoch) {
    std::size_t const world = static_cast<std::size_t>(segments_per_member_k) * member_count;
    array<u8_t> seen;
    if (seen.resize(world) != status_t::success_k) return suite.check(false, "bijection scratch alloc failed");
    for (std::size_t index = 0; index != world; ++index) seen[index] = 0u;
    bool bijection = true;
    for (std::size_t destination = 0; destination != world; ++destination) {
        std::size_t const source = epoch_permutation(seed, epoch, destination, world);
        if (source >= world || seen[source] != 0u) bijection = false;
        else seen[source] = 1u;
    }
    suite.check(bijection, "the routing must be a bijection over {} world segments", world);
}

/** Bags per member in the bag-cohesion tier — the latched routing capacity. */
inline constexpr u32_t bags_per_member_k = 3u;

/** The bag id a PAD segment carries — one past the last valid bag, the mixer's no-bag @c end
 *  sentinel. */
inline constexpr u32_t pad_bag_k = bags_per_member_k;

/** Bag @p bag 's live segment count — deliberately different per bag ( @c {3,1,2} ), so the tier
 *  exercises the fixed-capacity / variable-degree law rather than a uniform special case. */
[[nodiscard]] static u32_t bag_degree_(u32_t bag) noexcept {
    u32_t const degrees[bags_per_member_k] = {3u, 1u, 2u};
    return degrees[bag];
}

/** The segment index where bag @p bag 's contiguous run starts in a fabricated bagged batch. */
[[nodiscard]] static u32_t bag_first_segment_(u32_t bag) noexcept {
    u32_t first = 0;
    for (u32_t earlier = 0; earlier != bag; ++earlier) first += bag_degree_(earlier);
    return first;
}

/**
 *  @brief Fabricate member @p member 's bagged loaded batch: @ref bags_per_member_k bags of
 *      @ref bag_degree_ segments packed contiguously in bag order, each @ref tokens_per_segment_k columns
 *      of @ref signature_token_ , the tail PAD (zero length). Mirrors @ref fabricate_loaded_ 's staging.
 *  @param[inout] batch The member's loaded batch, its geometry, bags, and tokens overwritten.
 *  @param[in] member This member's world coordinate, folded into every token signature.
 *  @param[inout] stream The member's stream the staging enqueues on.
 */
[[nodiscard]] static expected_t fabricate_bagged_(multimodal_batch_on_gpu_t &batch, u32_t member,
                                                  gpu_stream_t &stream) noexcept {
    u32_t const S = segments_per_member_k, L = tokens_per_segment_k;

    switch_to_gpu_t gpu_scope {stream};
    if (!gpu_scope.ok()) return expected_t::fail(status_t::cuda_error_k, "context entry failed for member {}", member);

    auto offsets = pinned1_t<u32_t>::zeros({static_cast<std::size_t>(S) + 1u}, pinned_alloc<u32_t> {}).value;
    auto lengths = pinned1_t<u32_t>::zeros({S}, pinned_alloc<u32_t> {}).value;
    auto modality = pinned1_t<modality_t>::zeros({S}, pinned_alloc<modality_t> {}).value;
    auto bags = pinned1_t<u32_t>::zeros({S}, pinned_alloc<u32_t> {}).value;
    auto tokens = pinned1_t<token_id_t>::zeros({batch.tokens.extent(0) * batch.tokens.extent(1)},
                                               pinned_alloc<token_id_t> {})
                      .value;
    if (!offsets || !lengths || !modality || !bags || !tokens)
        return expected_t::fail(status_t::bad_alloc_k, "bagged fabrication scratch alloc failed");

    u32_t segment = 0;
    for (u32_t bag = 0; bag != bags_per_member_k; ++bag)
        for (u32_t position = 0; position != bag_degree_(bag); ++position, ++segment) {
            offsets[segment] = segment * L;
            lengths[segment] = L;
            modality[segment] = modality_t::text_k;
            bags[segment] = bag;
            for (u32_t j = 0; j != L; ++j) tokens[segment * L + j] = signature_token_(member, segment, j);
        }
    for (u32_t pad = segment; pad != S; ++pad) {
        offsets[pad] = segment * L;
        // Keeps the fabrication self-consistent; the mixer's nonzero-length filter is what actually
        // excludes a PAD from every bag, whatever its tag.
        bags[pad] = pad_bag_k;
    }
    offsets[S] = segment * L;

    if (expected_t staged = copy_host_to_device(offsets.view(), batch.segment_offsets.span(), stream); !staged)
        return staged;
    if (expected_t staged = copy_host_to_device(lengths.view(), batch.segment_lengths.span(), stream); !staged)
        return staged;
    if (expected_t staged = copy_host_to_device(modality.view(), batch.segment_modality.span(), stream); !staged)
        return staged;
    if (expected_t staged = copy_host_to_device(bags.view(), batch.segment_bag.span(), stream); !staged) return staged;
    if (expected_t staged = copy_host_to_device(tokens.view(), batch.tokens.span().flatten<1>(), stream); !staged)
        return staged;
    if (CUresult synced = cuStreamSynchronize(stream.get()); synced != CUDA_SUCCESS)
        return cuda_failure(synced, "bagged fabrication sync failed for member {}", member);
    return {};
}

/** One host-mirrored mixed segment — what the geometry kernel must have written for it. */
struct mirrored_segment_t {

    /** The segment's flat pack position in the mixed grid. */
    u32_t offset = 0;

    /** The segment's live token count. */
    u32_t length = 0;

    /** The destination-local bag the segment was renumbered into. */
    u32_t bag = 0;

    /** The world member the segment was gathered from. */
    u32_t source_member = 0;

    /** The segment's index within its source member's batch. */
    u32_t source_segment = 0;
};

/** A member's host-mirrored bag walk — each placed segment's placement and world source,
 *  index-aligned to the mixed batch; everything past @c size() is the PAD tail. */
using mirrored_geometry_t = stack_array<mirrored_segment_t, segments_per_member_k>;

/**
 *  @brief Host-mirror the bag-cohesive geometry walk for destination @p member — the same trial-then-commit
 *      cursor scan the kernel runs, over the fabrication's known degrees. A bag that cannot land whole is
 *      skipped whole, exactly as the kernel drops it.
 *  @param[in] member The destination's world coordinate — its bag range's base.
 *  @param[in] world_member_count The world's member count — the bag permutation spans
 *      @c bags_per_member_k·world_member_count .
 *  @param[in] seed The routing seed the mixer was built with.
 *  @param[in] epoch The data epoch the mix ran at.
 *  @param[in] rows The mixed grid's row capacity.
 *  @param[in] columns_per_row The mixed grid's row width.
 */
[[nodiscard]] static mirrored_geometry_t mirror_bag_geometry_(u32_t member, u32_t world_member_count, u64_t seed,
                                                              data_epoch_t epoch, u32_t rows,
                                                              u32_t columns_per_row) noexcept {
    u32_t const L = tokens_per_segment_k;
    u32_t const capacity = rows * columns_per_row;
    std::size_t const world_bags = static_cast<std::size_t>(bags_per_member_k) * world_member_count;
    mirrored_geometry_t mirror;
    u32_t cursor = 0;
    for (u32_t bag = 0; bag != bags_per_member_k; ++bag) {
        std::size_t const destination_bag = static_cast<std::size_t>(member) * bags_per_member_k + bag;
        std::size_t const source_bag_world = epoch_permutation(seed, epoch, destination_bag, world_bags);
        u32_t const source_member = static_cast<u32_t>(source_bag_world / bags_per_member_k);
        u32_t const source_bag = static_cast<u32_t>(source_bag_world % bags_per_member_k);
        u32_t const degree = bag_degree_(source_bag);

        u32_t trial_cursor = cursor;
        bool fits = mirror.size() + degree <= segments_per_member_k;
        for (u32_t position = 0; position != degree && fits; ++position) {
            u32_t const room_in_row = columns_per_row - trial_cursor % columns_per_row;
            if (L > room_in_row) trial_cursor += room_in_row;
            fits = trial_cursor + L <= capacity;
            trial_cursor += L;
        }
        if (!fits) continue; // dropped whole, exactly as the kernel drops it

        for (u32_t position = 0; position != degree; ++position) {
            u32_t const room_in_row = columns_per_row - cursor % columns_per_row;
            if (L > room_in_row) cursor += room_in_row;
            mirrored_segment_t placed;
            placed.offset = cursor;
            placed.length = L;
            placed.bag = bag;
            placed.source_member = source_member;
            placed.source_segment = bag_first_segment_(source_bag) + position;
            (void)mirror.push_back(placed);
            cursor += L;
        }
    }
    return mirror;
}

/**
 *  @brief Read member @p member 's bagged mixed batch back and assert it equals the host mirror exactly.
 *
 *  It checks every placed segment's offset, length, destination-local bag, and tokens; contiguous bag runs
 *  in ascending order (cohesion); and a zero-length, no-bag tail — so a dropped bag is provably dropped
 *  @b whole and a surviving one provably intact.
 *
 *  @param[inout] suite The conformance suite the checks report into.
 *  @param[inout] mixed The member's mixed batch to read back.
 *  @param[in] member This member's world coordinate — its destination range's base.
 *  @param[in] world_member_count The world's member count — the bag permutation spans
 *      @c bags_per_member_k·world_member_count .
 *  @param[in] seed The routing seed the mixer was built with.
 *  @param[in] epoch The data epoch the mix ran at.
 *  @param[inout] stream The member's stream the readback enqueues on.
 */
[[nodiscard]] static expected_t verify_bagged_(test_suite_t &suite, multimodal_batch_on_gpu_t &mixed, u32_t member,
                                               u32_t world_member_count, u64_t seed, data_epoch_t epoch,
                                               gpu_stream_t &stream) noexcept {
    u32_t const S = segments_per_member_k, L = tokens_per_segment_k;
    u32_t const rows = static_cast<u32_t>(mixed.tokens.extent(0));
    u32_t const columns = static_cast<u32_t>(mixed.tokens.extent(1));

    switch_to_gpu_t gpu_scope {stream};
    if (!gpu_scope.ok()) return expected_t::fail(status_t::cuda_error_k, "context entry failed for member {}", member);

    auto offsets = pinned1_t<u32_t>::uninitialized({static_cast<std::size_t>(S) + 1u}, pinned_alloc<u32_t> {}).value;
    auto lengths = pinned1_t<u32_t>::uninitialized({S}, pinned_alloc<u32_t> {}).value;
    auto bags = pinned1_t<u32_t>::uninitialized({S}, pinned_alloc<u32_t> {}).value;
    auto tokens = pinned1_t<token_id_t>::uninitialized({static_cast<std::size_t>(rows) * columns},
                                                       pinned_alloc<token_id_t> {})
                      .value;
    if (!offsets || !lengths || !bags || !tokens)
        return expected_t::fail(status_t::bad_alloc_k, "bagged readback scratch alloc failed");

    if (expected_t read = copy_device_to_host(mixed.segment_offsets.view(), offsets.span(), stream); !read) return read;
    if (expected_t read = copy_device_to_host(mixed.segment_lengths.view(), lengths.span(), stream); !read) return read;
    if (expected_t read = copy_device_to_host(mixed.segment_bag.view(), bags.span(), stream); !read) return read;
    if (expected_t read = copy_device_to_host(mixed.tokens.view().flatten<1>(), tokens.span(), stream); !read)
        return read;
    if (CUresult synced = cuStreamSynchronize(stream.get()); synced != CUDA_SUCCESS)
        return cuda_failure(synced, "bagged readback sync failed for member {}", member);

    mirrored_geometry_t const mirror = mirror_bag_geometry_(member, world_member_count, seed, epoch, rows, columns);
    u32_t const placed = static_cast<u32_t>(mirror.size());
    for (u32_t s = 0; s != placed; ++s) {
        mirrored_segment_t const &want = mirror[s];
        suite.check(offsets[s] == want.offset && lengths[s] == want.length && bags[s] == want.bag,
                    "member {} segment {}: placed (offset={},length={},bag={}) != mirror ({},{},{})", member, s,
                    offsets[s], lengths[s], bags[s], want.offset, want.length, want.bag);
        suite.check(s == 0 || bags[s] >= bags[s - 1u], "member {} segment {}: bag {} breaks the cohesive order", member,
                    s, bags[s]);
        for (u32_t j = 0; j != L; ++j) {
            token_id_t const got = tokens[offsets[s] + j];
            token_id_t const want_token = signature_token_(want.source_member, want.source_segment, j);
            suite.check(got == want_token, "member {} segment {} column {}: token {} != source ({},{}) token {}",
                        member, s, j, got, want.source_member, want.source_segment, want_token);
        }
    }
    for (u32_t s = placed; s != S; ++s)
        suite.check(lengths[s] == 0u && bags[s] == pad_bag_k,
                    "member {} segment {}: the tail must be a no-bag PAD (length={} bag={})", member, s, lengths[s],
                    bags[s]);
    return {};
}

/** One tier's per-member surfaces — the movable half of the world stand-up both tiers share; the
 *  domains and the mixer pin references into it, so they are constructed in the tier itself and never
 *  moved. */
struct mixer_members_t {

    /** The visible device ordinals, table-shaped for the arena visibility lists. */
    array<i32_t> devices {};

    /** One stream per member — every fabrication, mix, and readback enqueues here. */
    array<gpu_stream_t> streams {};

    /** One peer-visible arena per member — the batches and the mixer's scratch live off these. */
    array<gpu_arena_t> arenas {};

    /** Each member's loaded (source) batch. */
    array<multimodal_batch_on_gpu_t> loaded {};

    /** Each member's mixed (destination) batch. */
    array<multimodal_batch_on_gpu_t> mixed {};
};

/** Stand up @p member_count members' streams, peer-visible arenas, and loaded / mixed batches at
 *  @p shape — the bring-up both mixer tiers share; fabrication and the domains stay tier-local. */
[[nodiscard]] static expected<mixer_members_t> stand_up_members_(u32_t member_count,
                                                                 multimodal_batch_shape_t shape) noexcept {
    using result_t = expected<mixer_members_t>;
    mixer_members_t members;
    if (members.devices.resize(member_count) != status_t::success_k)
        return result_t::fail(status_t::bad_alloc_k, "device table alloc failed");
    for (u32_t member = 0; member != member_count; ++member) members.devices[member] = static_cast<i32_t>(member);
    if (members.streams.reserve(member_count) != status_t::success_k ||
        members.arenas.reserve(member_count) != status_t::success_k ||
        members.loaded.reserve(member_count) != status_t::success_k ||
        members.mixed.reserve(member_count) != status_t::success_k)
        return result_t::fail(status_t::bad_alloc_k, "per-member table reservation failed");
    for (u32_t member = 0; member != member_count; ++member) {
        auto stream_made = gpu_stream_t::make(static_cast<int>(member));
        if (!stream_made) return result_t::fail(std::move(stream_made));
        (void)members.streams.push_back(std::move(stream_made.value_));
        switch_to_gpu_t gpu_scope {members.streams[member]};
        auto arena_made = gpu_arena_t::make(static_cast<int>(member), 512ull << 20, members.devices.view());
        if (!arena_made) return result_t::fail(std::move(arena_made));
        (void)members.arenas.push_back(std::move(arena_made.value_));
        auto loaded_made = multimodal_batch_on_gpu_t::make(gpu_arena_alloc_t {members.arenas[member]}, shape);
        if (!loaded_made) return result_t::fail(std::move(loaded_made));
        (void)members.loaded.push_back(std::move(loaded_made.value_));
        auto mixed_made = multimodal_batch_on_gpu_t::make(gpu_arena_alloc_t {members.arenas[member]}, shape);
        if (!mixed_made) return result_t::fail(std::move(mixed_made));
        (void)members.mixed.push_back(std::move(mixed_made.value_));
    }
    return result_t {std::move(members)};
}

/**
 *  @brief The mixer tier: stand up the domains over every visible GPU — one process per host_domain,
 *      reading the same @c UFORM_HOST_DOMAIN_COUNT contract as the base suite, so a two-node run exercises
 *      the InfiniBand full-batch exchange.
 *
 *  It fabricates each member's loaded batch under its @b global member signature, runs one mix through
 *  the eager overload, and verifies every mixed segment against its world source. Skips cleanly when the
 *  box cannot host a multicast multimem_domain (the make fails loudly).
 */
static void mixer_routes_segments(test_suite_t &suite) {
    int device_count = 0;
    if (cuDeviceGetCount(&device_count) != CUDA_SUCCESS || device_count < 1) return;
    u32_t const member_count = std::min<u32_t>(static_cast<u32_t>(device_count), members_cap_k);
    u64_t const seed = 0xA5A5C0FFEEULL;
    data_epoch_t const epoch {1u};

    u32_t const host_domain_index = host_domain_index_from_environment();
    u32_t const host_domain_count = host_domain_count_from_environment();
    auto bootstrapped = mesh_topology_t::bootstrap(host_domain_index, host_domain_count, member_count);
    if (!bootstrapped) return suite.check(false, "mesh bootstrap: {}", bootstrapped.description_);
    mesh_topology_t topology = std::move(bootstrapped.value_);
    // The routing world spans every multimem_domain's members; this domain's GPUs carry the world
    // coordinates [first, first + member_count) the signatures and the oracle both use.
    u32_t const world_member_count = member_count * host_domain_count;
    u32_t const first_gpu_index_in_colocated_domain = topology.shape.gpu_index_of_multimem_member(
        topology.multimem_domain_index(), 0u);

    multimodal_batch_shape_t shape;
    shape.max_rows_per_gpu = 2u;
    shape.max_segments_per_gpu = segments_per_member_k;
    shape.max_columns_per_row = 64u; // > segments_per_member_k·tokens_per_segment_k, so no packing row-jump

    auto members_made = stand_up_members_(member_count, shape);
    if (!members_made) return suite.check(false, "member stand-up: {}", members_made.description_);
    mixer_members_t members = std::move(members_made.value_);
    for (u32_t member = 0; member != member_count; ++member)
        if (expected_t built = fabricate_loaded_(members.loaded[member], first_gpu_index_in_colocated_domain + member,
                                                 members.streams[member]);
            !built)
            return suite.check(false, "fabricate loaded on GPU {}: {}", member, built.description_);

    routing_is_a_bijection(suite, world_member_count, seed, epoch);

    auto multimem_made = multimem_domain_t::make(
        topology, members.devices.view(),
        1024u + multimodal_batch_mixer_t::payload_scalars_needed(shape, topology.shape), members.streams.span());
    if (!multimem_made)
        return suite.check(false, "multimem_domain make (needs multicast capability): {}", multimem_made.description_);
    multimem_domain_t multimem = std::move(multimem_made.value_);
    auto colocated_made = colocated_domain_t::make(topology);
    if (!colocated_made) return suite.check(false, "colocated_domain make: {}", colocated_made.description_);
    colocated_domain_t colocated = std::move(colocated_made.value_);
    auto mixer_made = multimodal_batch_mixer_t::make(multimem, colocated, members.arenas.span(), shape, seed,
                                                     members.streams.span());
    if (!mixer_made) return suite.check(false, "mixer make: {}", mixer_made.description_);
    multimodal_batch_mixer_t mixer = std::move(mixer_made.value_);

    std::span<multimodal_batch_on_gpu_t const> const loaded_view = members.loaded.view();
    if (expected_t mixed_ok = mixer(epoch, loaded_view, members.mixed.span(), members.streams.span()); !mixed_ok)
        return suite.check(false, "the mix should run: {}", mixed_ok.description_);

    for (u32_t member = 0; member != member_count; ++member)
        if (expected_t verified = verify_mixed_(suite, members.mixed[member],
                                                first_gpu_index_in_colocated_domain + member, world_member_count, seed,
                                                epoch, members.streams[member]);
            !verified)
            suite.check(false, "verify mixed on GPU {}: {}", member, verified.description_);
}

/**
 *  @brief The bag-cohesion tier: the same bring-up as @ref mixer_routes_segments but with a nonzero bag
 *      budget, bagged fabrication, and the exact host mirror of the bag walk as the oracle.
 *
 *  Runs twice — @p rows × @p columns_per_row sized roomy (every bag lands) and tight (the grid barely holds one
 *  member's own load, so a permutation draw that concentrates heavy bags on one destination forces
 *  whole-bag drops; the mirror predicts each drop exactly, so the tier is a strict oracle either
 *  way). Every source batch stays valid — only the receive side can overflow.
 */
static void mixer_routes_bags(test_suite_t &suite, u32_t rows, u32_t columns_per_row) {
    int device_count = 0;
    if (cuDeviceGetCount(&device_count) != CUDA_SUCCESS || device_count < 1) return;
    u32_t const member_count = std::min<u32_t>(static_cast<u32_t>(device_count), members_cap_k);
    u64_t const seed = 0xBA65C0FFEEULL;
    data_epoch_t const epoch {2u};

    u32_t const host_domain_index = host_domain_index_from_environment();
    u32_t const host_domain_count = host_domain_count_from_environment();
    auto bootstrapped = mesh_topology_t::bootstrap(host_domain_index, host_domain_count, member_count);
    if (!bootstrapped) return suite.check(false, "mesh bootstrap: {}", bootstrapped.description_);
    mesh_topology_t topology = std::move(bootstrapped.value_);
    u32_t const world_member_count = member_count * host_domain_count;
    u32_t const first_gpu_index_in_colocated_domain = topology.shape.gpu_index_of_multimem_member(
        topology.multimem_domain_index(), 0u);

    multimodal_batch_shape_t shape;
    shape.max_rows_per_gpu = rows;
    shape.max_segments_per_gpu = segments_per_member_k;
    shape.max_columns_per_row = columns_per_row;
    shape.max_bags_per_gpu = bags_per_member_k;

    auto members_made = stand_up_members_(member_count, shape);
    if (!members_made) return suite.check(false, "member stand-up: {}", members_made.description_);
    mixer_members_t members = std::move(members_made.value_);
    for (u32_t member = 0; member != member_count; ++member)
        if (expected_t built = fabricate_bagged_(members.loaded[member], first_gpu_index_in_colocated_domain + member,
                                                 members.streams[member]);
            !built)
            return suite.check(false, "fabricate bagged on GPU {}: {}", member, built.description_);

    auto multimem_made = multimem_domain_t::make(
        topology, members.devices.view(),
        1024u + multimodal_batch_mixer_t::payload_scalars_needed(shape, topology.shape), members.streams.span());
    if (!multimem_made)
        return suite.check(false, "multimem_domain make (needs multicast capability): {}", multimem_made.description_);
    multimem_domain_t multimem = std::move(multimem_made.value_);
    auto colocated_made = colocated_domain_t::make(topology);
    if (!colocated_made) return suite.check(false, "colocated_domain make: {}", colocated_made.description_);
    colocated_domain_t colocated = std::move(colocated_made.value_);
    auto mixer_made = multimodal_batch_mixer_t::make(multimem, colocated, members.arenas.span(), shape, seed,
                                                     members.streams.span());
    if (!mixer_made) return suite.check(false, "mixer make: {}", mixer_made.description_);
    multimodal_batch_mixer_t mixer = std::move(mixer_made.value_);

    std::span<multimodal_batch_on_gpu_t const> const loaded_view = members.loaded.view();
    if (expected_t mixed_ok = mixer(epoch, loaded_view, members.mixed.span(), members.streams.span()); !mixed_ok)
        return suite.check(false, "the bagged mix should run: {}", mixed_ok.description_);

    for (u32_t member = 0; member != member_count; ++member)
        if (expected_t verified = verify_bagged_(suite, members.mixed[member],
                                                 first_gpu_index_in_colocated_domain + member, world_member_count, seed,
                                                 epoch, members.streams[member]);
            !verified)
            suite.check(false, "verify bagged on GPU {}: {}", member, verified.description_);
}

int main() {
    test_suite_t suite {"mixer conformance"};
    // The mixer runs eagerly (no captured graph) and its barrier is a device spin. Under CUDA's default lazy
    // module loading, the first kernel launched after the barrier blocks on its module load while the barrier
    // already spins — stranding the single host thread that still owes the other members' arrivals. Eager
    // loading resolves every module up front, so the post-barrier launch never touches the loader.
    ::setenv("CUDA_MODULE_LOADING", "EAGER", 1);
    if (cuInit(0) != CUDA_SUCCESS) return suite.finish(); // no GPU — nothing to exercise
    mixer_routes_segments(suite);
    mixer_routes_bags(suite, 2u, 64u); // roomy: every bag lands whole
    // Tight: 20 cells hold one member's own 18-token load (sources stay valid) but not every receive
    // combination — a destination drawing two heavy bags overflows and must drop whole, mirror-verified.
    mixer_routes_bags(suite, 1u, 20u);
    return suite.finish();
}
