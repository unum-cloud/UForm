/**
 *  @file test/mixer.cu
 *  @author Ash Vardanian
 *  @date July 6, 2026
 *  @brief Conformance suite for @ref multimodal_batch_mixer_t — the cross-GPU segment shuffle.
 *
 *  Builds the composition root ( @ref mesh_topology_t::bootstrap → per-GPU arenas / streams →
 *  @ref multimem_domain_t → @ref colocated_domain_t → the mixer), fabricates every member's
 *  loaded batch with a token signature distinct per @c (gpu_index,segment,column) — the world
 *  coordinate — runs one mix, and asserts each mixed segment equals its @ref epoch_permutation
 *  source: the routing bijection, the length / modality descriptor travel, and the token content.
 *  One process per host_domain under the same @c UFORM_HOST_DOMAIN_COUNT contract as the base
 *  suite: one node exercises the NVLink multimem_domain alone, two nodes add the InfiniBand
 *  full-batch exchange to same-index landings.
 */
#undef NDEBUG // ! A test's oracle must stay live in every build

#include <cstddef> // `std::size_t`
#include <cstdlib> // `::setenv`, `std::getenv`, `std::atoi` — eager module loading, the host_domain contract

#include <utility> // `std::move`
#include <span>    // `std::span`

#include <cuda.h> // `cuInit`, `cuDeviceGetCount`, `cuStreamSynchronize`

#include "uform/utilities/array.hpp"                  // `stack_array`
#include "uform/utilities/vram_nvidia_device.hpp"     // `gpu_arena_t`, `gpu_arena_alloc_t`, `pinned_alloc`
#include "uform/utilities/infiniband_collectives.cuh" // `colocated_domain_t`
#include "uform/utilities/mesh.hpp"                   // `mesh_topology_t`, `host_domain_index_from_environment`
#include "uform/utilities/multimem_collectives.cuh"   // `multimem_domain_t`
#include "uform/encoder/multimodal_batch.cuh"         // `multimodal_batch_on_gpu_t`, the shape, the mixer
#include "uform/utilities/vram_nvidia_schedule.cuh"   // `gpu_stream_t`, `switch_to_gpu_t`, `copy_host_to_device`
#include "uform/utilities/types.hpp"                  // `epoch_permutation`, `data_epoch_t`, `token_id_t`, `modality_t`

#include "harness.hpp" // `run_test_case`, `expect_true`

namespace ue = unum::uform::encoder;
using ue::modality_t;
using ue::multimodal_batch_mixer_t;
using ue::multimodal_batch_on_gpu_t;
using ue::multimodal_batch_shape_t;
namespace util = unum::uform::utilities;
using namespace unum::uform::test;

/** Segments per member — the routing axis; small, so the whole world permutation is
 *  easy to mirror. */
inline constexpr util::u32_t segments_per_member_k = 8u;

/** Tokens per segment — a short run whose per-column signature the oracle reproduces exactly. */
inline constexpr util::u32_t tokens_per_segment_k = 3u;

/** Members to exercise on one box — one multimem_domain's worth, capped by the
 *  visible device count. */
inline constexpr util::u32_t members_cap_k = 8u;

/** A rank-1 pinned host tensor of @p value_type_ — the sanctioned host staging
 *  / readback buffer. */
template <typename value_type_>
using pinned1_t = nk::tensor<value_type_, util::pinned_alloc<value_type_>, 1>;

/** The deterministic token a member's segment writes at a column — distinct per
 *  @c (member,segment,column) so a mislaid route mismatches. Wraps into @ref token_id_t ; the
 *  oracle wraps identically. */
[[nodiscard]] static util::token_id_t signature_token_(util::u32_t member, util::u32_t segment,
                                                       util::u32_t column) noexcept {
    return static_cast<util::token_id_t>((member * 131u + segment) * 17u + column + 1u);
}

/**
 *  @brief Fabricate member @p member 's loaded batch: @ref segments_per_member_k text segments,
 *      each @ref tokens_per_segment_k columns, packed contiguously into row 0, tokens carrying
 *      @ref signature_token_ . Staged host → device through the pinned + @ref copy_host_to_device
 *      path, synchronized before returning.
 *  @param[in,out] batch The member's loaded batch, its geometry and tokens overwritten.
 *  @param[in] member This member's world coordinate ( @c gpu_index_in_colocated_domain ), folded
 *      into every token signature.
 *  @param[in,out] stream The member's stream the staging enqueues on.
 */
[[nodiscard]] static util::expected_t fabricate_loaded_(multimodal_batch_on_gpu_t& batch, util::u32_t member,
                                                        util::gpu_stream_t& stream) noexcept {
    util::u32_t const S = segments_per_member_k, L = tokens_per_segment_k;
    util::u32_t const rows = static_cast<util::u32_t>(batch.tokens.extent(0));
    util::u32_t const columns = static_cast<util::u32_t>(batch.tokens.extent(1));

    util::switch_to_gpu_t gpu_scope {stream};
    if (!gpu_scope.ok())
        return util::expected_t::fail(util::status_code_t::device_k, "context entry failed for member {}", member);

    pinned1_t<util::u32_t> offsets = make_zeros<pinned1_t<util::u32_t>>({static_cast<std::size_t>(S) + 1u});
    pinned1_t<util::u32_t> lengths = make_zeros<pinned1_t<util::u32_t>>({S});
    pinned1_t<modality_t> modality = make_zeros<pinned1_t<modality_t>>({S});
    pinned1_t<util::token_id_t> tokens = make_zeros<pinned1_t<util::token_id_t>>(
        {static_cast<std::size_t>(rows) * columns});
    if (!offsets || !lengths || !modality || !tokens)
        return util::expected_t::fail(util::status_code_t::out_of_memory_k, "loaded fabrication scratch alloc failed");

    for (util::u32_t s = 0; s != S; ++s) {
        offsets[s] = s * L;
        lengths[s] = L;
        modality[s] = modality_t::text_k;
        for (util::u32_t j = 0; j != L; ++j) tokens[offsets[s] + j] = signature_token_(member, s, j);
    }
    offsets[S] = S * L;

    if (util::expected_t staged = util::copy_host_to_device(offsets.view(), batch.segment_offsets.span(), stream);
        !staged)
        return staged;
    if (util::expected_t staged = util::copy_host_to_device(lengths.view(), batch.segment_lengths.span(), stream);
        !staged)
        return staged;
    if (util::expected_t staged = util::copy_host_to_device(modality.view(), batch.segment_modality.span(), stream);
        !staged)
        return staged;
    if (util::expected_t staged = util::copy_host_to_device(tokens.view(), batch.tokens.span().flatten<1>(), stream);
        !staged)
        return staged;
    if (CUresult synced = cuStreamSynchronize(stream.get()); synced != CUDA_SUCCESS)
        return util::cuda_failure(synced, "loaded fabrication sync failed for member {}", member);
    return {};
}

/**
 *  @brief Read member @p member 's mixed batch back and assert every segment equals its
 *      @ref epoch_permutation source: destination @c member · S+s pulls world source @c src , whose
 *      member is @c src/S and segment @c src%S ; the mixed segment must have that source's length,
 *      its @c text_k modality, and its tokens.
 *  @param[in,out] mixed The member's mixed batch to read back.
 *  @param[in] member This member's world coordinate ( @c gpu_index_in_colocated_domain ) — its destination
 *      range's base.
 *  @param[in] member_count The world's member count — the permutation's segment span is @c S·member_count .
 *  @param[in] seed The routing seed the mixer was built with.
 *  @param[in] epoch The data epoch the mix ran at.
 *  @param[in,out] stream The member's stream the readback enqueues on.
 */
[[nodiscard]] static util::expected_t verify_mixed_(multimodal_batch_on_gpu_t& mixed, util::u32_t member,
                                                    util::u32_t member_count, util::u64_t seed,
                                                    util::data_epoch_t epoch, util::gpu_stream_t& stream) noexcept {
    util::u32_t const S = segments_per_member_k, L = tokens_per_segment_k;
    util::u32_t const rows = static_cast<util::u32_t>(mixed.tokens.extent(0));
    util::u32_t const columns = static_cast<util::u32_t>(mixed.tokens.extent(1));

    util::switch_to_gpu_t gpu_scope {stream};
    if (!gpu_scope.ok())
        return util::expected_t::fail(util::status_code_t::device_k, "context entry failed for member {}", member);

    pinned1_t<util::u32_t> offsets = make_zeros<pinned1_t<util::u32_t>>({static_cast<std::size_t>(S) + 1u});
    pinned1_t<util::u32_t> lengths = make_zeros<pinned1_t<util::u32_t>>({S});
    pinned1_t<modality_t> modality = make_zeros<pinned1_t<modality_t>>({S});
    pinned1_t<util::token_id_t> tokens = make_zeros<pinned1_t<util::token_id_t>>(
        {static_cast<std::size_t>(rows) * columns});
    if (!offsets || !lengths || !modality || !tokens)
        return util::expected_t::fail(util::status_code_t::out_of_memory_k, "mixed readback scratch alloc failed");

    if (util::expected_t read = util::copy_device_to_host(mixed.segment_offsets.view(), offsets.span(), stream); !read)
        return read;
    if (util::expected_t read = util::copy_device_to_host(mixed.segment_lengths.view(), lengths.span(), stream); !read)
        return read;
    if (util::expected_t read = util::copy_device_to_host(mixed.segment_modality.view(), modality.span(), stream);
        !read)
        return read;
    if (util::expected_t read = util::copy_device_to_host(mixed.tokens.view().flatten<1>(), tokens.span(), stream);
        !read)
        return read;
    if (CUresult synced = cuStreamSynchronize(stream.get()); synced != CUDA_SUCCESS)
        return util::cuda_failure(synced, "mixed readback sync failed for member {}", member);

    std::size_t const world = static_cast<std::size_t>(S) * member_count;
    for (util::u32_t s = 0; s != S; ++s) {
        std::size_t const destination = static_cast<std::size_t>(member) * S + s;
        std::size_t const source = util::epoch_permutation(seed, epoch, destination, world);
        util::u32_t const source_member = static_cast<util::u32_t>(source / S);
        util::u32_t const source_segment = static_cast<util::u32_t>(source % S);
        expect_true(lengths[s] == L, "member {} segment {}: mixed length {} != source length {}", member, s, lengths[s],
                    L);
        expect_true(modality[s] == modality_t::text_k, "member {} segment {}: mixed modality is not text", member, s);
        util::u32_t const offset = offsets[s];
        for (util::u32_t j = 0; j != L; ++j) {
            util::token_id_t const got = tokens[offset + j];
            util::token_id_t const want = signature_token_(source_member, source_segment, j);
            expect_true(got == want, "member {} segment {} column {}: token {} != source ({},{}) token {}", member, s,
                        j, got, source_member, source_segment, want);
        }
    }
    return {};
}

/** Assert the routing is a bijection over the world's @c S·member_count segments — every destination maps
 *  to a distinct source — mirroring @ref epoch_permutation host-side so a broken permutation is caught even
 *  before the device gather. */
static void routing_is_a_bijection(util::u32_t member_count, util::u64_t seed, util::data_epoch_t epoch) {
    std::size_t const world = static_cast<std::size_t>(segments_per_member_k) * member_count;
    st::basic_vector<util::u8_t> seen;
    if (!expect_true(!st::failed(seen.resize(world)), "bijection scratch alloc failed")) return;
    for (std::size_t index = 0; index != world; ++index) seen[index] = 0u;
    bool bijection = true;
    for (std::size_t destination = 0; destination != world; ++destination) {
        std::size_t const source = util::epoch_permutation(seed, epoch, destination, world);
        if (source >= world || seen[source] != 0u) bijection = false;
        else seen[source] = 1u;
    }
    expect_true(bijection, "the routing must be a bijection over {} world segments", world);
}

/** Bags per member in the bag-cohesion tier — the latched routing capacity. */
inline constexpr util::u32_t bags_per_member_k = 3u;

/** The bag ID a PAD segment has — one past the last valid bag, the mixer's no-bag @c end
 *  sentinel. */
inline constexpr util::u32_t pad_bag_k = bags_per_member_k;

/** Bag @p bag 's live segment count — deliberately different per bag ( @c {3,1,2} ), so the tier
 *  exercises the fixed-capacity / variable-degree law rather than a uniform special case. */
[[nodiscard]] static util::u32_t bag_degree_(util::u32_t bag) noexcept {
    util::u32_t const degrees[bags_per_member_k] = {3u, 1u, 2u};
    return degrees[bag];
}

/** The segment index where bag @p bag 's contiguous run starts in a fabricated bagged batch. */
[[nodiscard]] static util::u32_t bag_first_segment_(util::u32_t bag) noexcept {
    util::u32_t first = 0;
    for (util::u32_t earlier = 0; earlier != bag; ++earlier) first += bag_degree_(earlier);
    return first;
}

/**
 *  @brief Fabricate member @p member 's bagged loaded batch: @ref bags_per_member_k bags of
 *      @ref bag_degree_ segments packed contiguously in bag order, each @ref tokens_per_segment_k
 *      columns of @ref signature_token_ , the tail PAD (zero length). Mirrors
 *      @ref fabricate_loaded_ 's staging.
 *  @param[in,out] batch The member's loaded batch, its geometry, bags, and tokens overwritten.
 *  @param[in] member This member's world coordinate, folded into every token signature.
 *  @param[in,out] stream The member's stream the staging enqueues on.
 */
[[nodiscard]] static util::expected_t fabricate_bagged_(multimodal_batch_on_gpu_t& batch, util::u32_t member,
                                                        util::gpu_stream_t& stream) noexcept {
    util::u32_t const S = segments_per_member_k, L = tokens_per_segment_k;

    util::switch_to_gpu_t gpu_scope {stream};
    if (!gpu_scope.ok())
        return util::expected_t::fail(util::status_code_t::device_k, "context entry failed for member {}", member);

    pinned1_t<util::u32_t> offsets = make_zeros<pinned1_t<util::u32_t>>({static_cast<std::size_t>(S) + 1u});
    pinned1_t<util::u32_t> lengths = make_zeros<pinned1_t<util::u32_t>>({S});
    pinned1_t<modality_t> modality = make_zeros<pinned1_t<modality_t>>({S});
    pinned1_t<util::u32_t> bags = make_zeros<pinned1_t<util::u32_t>>({S});
    pinned1_t<util::token_id_t> tokens = make_zeros<pinned1_t<util::token_id_t>>(
        {batch.tokens.extent(0) * batch.tokens.extent(1)});
    if (!offsets || !lengths || !modality || !bags || !tokens)
        return util::expected_t::fail(util::status_code_t::out_of_memory_k, "bagged fabrication scratch alloc failed");

    util::u32_t segment = 0;
    for (util::u32_t bag = 0; bag != bags_per_member_k; ++bag)
        for (util::u32_t position = 0; position != bag_degree_(bag); ++position, ++segment) {
            offsets[segment] = segment * L;
            lengths[segment] = L;
            modality[segment] = modality_t::text_k;
            bags[segment] = bag;
            for (util::u32_t j = 0; j != L; ++j) tokens[offsets[segment] + j] = signature_token_(member, segment, j);
        }
    for (util::u32_t pad = segment; pad != S; ++pad) {
        offsets[pad] = segment * L;
        // Keeps the fabrication self-consistent; the mixer's nonzero-length filter is what actually
        // excludes a PAD from every bag, whatever its tag.
        bags[pad] = pad_bag_k;
    }
    offsets[S] = segment * L;

    if (util::expected_t staged = util::copy_host_to_device(offsets.view(), batch.segment_offsets.span(), stream);
        !staged)
        return staged;
    if (util::expected_t staged = util::copy_host_to_device(lengths.view(), batch.segment_lengths.span(), stream);
        !staged)
        return staged;
    if (util::expected_t staged = util::copy_host_to_device(modality.view(), batch.segment_modality.span(), stream);
        !staged)
        return staged;
    if (util::expected_t staged = util::copy_host_to_device(bags.view(), batch.segment_bag.span(), stream); !staged)
        return staged;
    if (util::expected_t staged = util::copy_host_to_device(tokens.view(), batch.tokens.span().flatten<1>(), stream);
        !staged)
        return staged;
    if (CUresult synced = cuStreamSynchronize(stream.get()); synced != CUDA_SUCCESS)
        return util::cuda_failure(synced, "bagged fabrication sync failed for member {}", member);
    return {};
}

/** One host-mirrored mixed segment — what the geometry kernel must have written for it. */
struct mirrored_segment_t {

    /** The segment's flat pack position in the mixed grid. */
    util::u32_t offset = 0;

    /** The segment's live token count. */
    util::u32_t length = 0;

    /** The destination-local bag the segment was renumbered into. */
    util::u32_t bag = 0;

    /** The world member the segment was gathered from. */
    util::u32_t source_member = 0;

    /** The segment's index within its source member's batch. */
    util::u32_t source_segment = 0;
};

/** A member's host-mirrored bag walk — each placed segment's placement and world source,
 *  index-aligned to the mixed batch; everything past @c size() is the PAD tail. */
using mirrored_geometry_t = util::stack_array<mirrored_segment_t, segments_per_member_k>;

/**
 *  @brief Host-mirror the bag-cohesive geometry walk for destination @p member — the same
 *      trial-then-commit cursor scan the kernel runs, over the fabrication's known degrees. A bag
 *      that cannot land whole is skipped whole, exactly as the kernel drops it.
 *  @param[in] member The destination's world coordinate — its bag range's base.
 *  @param[in] world_member_count The world's member count — the bag permutation spans
 *      @c bags_per_member_k·world_member_count .
 *  @param[in] seed The routing seed the mixer was built with.
 *  @param[in] epoch The data epoch the mix ran at.
 *  @param[in] rows The mixed grid's row capacity.
 *  @param[in] columns_per_row The mixed grid's row width.
 *  @param[out] dropped The segments of the bags skipped whole.
 */
[[nodiscard]] static mirrored_geometry_t mirror_bag_geometry_(util::u32_t member, util::u32_t world_member_count,
                                                              util::u64_t seed, util::data_epoch_t epoch,
                                                              util::u32_t rows, util::u32_t columns_per_row,
                                                              util::u32_t& dropped) noexcept {
    util::u32_t const L = tokens_per_segment_k;
    util::u32_t const capacity = rows * columns_per_row;
    std::size_t const world_bags = static_cast<std::size_t>(bags_per_member_k) * world_member_count;
    mirrored_geometry_t mirror;
    util::u32_t cursor = 0;
    dropped = 0;
    for (util::u32_t bag = 0; bag != bags_per_member_k; ++bag) {
        std::size_t const destination_bag = static_cast<std::size_t>(member) * bags_per_member_k + bag;
        std::size_t const source_bag_world = util::epoch_permutation(seed, epoch, destination_bag, world_bags);
        util::u32_t const source_member = static_cast<util::u32_t>(source_bag_world / bags_per_member_k);
        util::u32_t const source_bag = static_cast<util::u32_t>(source_bag_world % bags_per_member_k);
        util::u32_t const degree = bag_degree_(source_bag);

        util::u32_t trial_cursor = cursor;
        bool fits = mirror.size() + degree <= segments_per_member_k;
        for (util::u32_t position = 0; position != degree && fits; ++position) {
            util::u32_t const room_in_row = columns_per_row - trial_cursor % columns_per_row;
            if (L > room_in_row) trial_cursor += room_in_row;
            fits = trial_cursor + L <= capacity;
            trial_cursor += L;
        }
        if (!fits) { // dropped whole, exactly as the kernel drops it
            dropped += degree;
            continue;
        }

        for (util::u32_t position = 0; position != degree; ++position) {
            util::u32_t const room_in_row = columns_per_row - cursor % columns_per_row;
            if (L > room_in_row) cursor += room_in_row;
            mirrored_segment_t placed;
            placed.offset = cursor;
            placed.length = L;
            placed.bag = bag;
            placed.source_member = source_member;
            placed.source_segment = bag_first_segment_(source_bag) + position;
            [[maybe_unused]] util::status_t const mirrored = mirror.push_back(placed);
            cursor += L;
        }
    }
    return mirror;
}

/**
 *  @brief Read member @p member 's bagged mixed batch back and assert it equals the
 *      host mirror exactly.
 *
 *  It checks every placed segment's offset, length, destination-local bag, and tokens; contiguous
 *  bag runs in ascending order (cohesion); and a zero-length, no-bag tail — so a dropped bag is
 *  provably dropped @b whole and a surviving one provably intact.
 *
 *  @param[in,out] mixed The member's mixed batch to read back.
 *  @param[in] member This member's world coordinate — its destination range's base.
 *  @param[in] world_member_count The world's member count — the bag permutation spans
 *      @c bags_per_member_k·world_member_count .
 *  @param[in] seed The routing seed the mixer was built with.
 *  @param[in] epoch The data epoch the mix ran at.
 *  @param[in,out] stream The member's stream the readback enqueues on.
 */
[[nodiscard]] static util::expected_t verify_bagged_(multimodal_batch_on_gpu_t& mixed, util::u32_t member,
                                                     util::u32_t world_member_count, util::u64_t seed,
                                                     util::data_epoch_t epoch, util::gpu_stream_t& stream) noexcept {
    util::u32_t const S = segments_per_member_k, L = tokens_per_segment_k;
    util::u32_t const rows = static_cast<util::u32_t>(mixed.tokens.extent(0));
    util::u32_t const columns = static_cast<util::u32_t>(mixed.tokens.extent(1));

    util::switch_to_gpu_t gpu_scope {stream};
    if (!gpu_scope.ok())
        return util::expected_t::fail(util::status_code_t::device_k, "context entry failed for member {}", member);

    pinned1_t<util::u32_t> offsets = make_uninitialized<pinned1_t<util::u32_t>>({static_cast<std::size_t>(S) + 1u});
    pinned1_t<util::u32_t> lengths = make_uninitialized<pinned1_t<util::u32_t>>({S});
    pinned1_t<util::u32_t> bags = make_uninitialized<pinned1_t<util::u32_t>>({S});
    pinned1_t<util::u32_t> dropped = make_uninitialized<pinned1_t<util::u32_t>>({1});
    pinned1_t<util::token_id_t> tokens = make_uninitialized<pinned1_t<util::token_id_t>>(
        {static_cast<std::size_t>(rows) * columns});
    if (!offsets || !lengths || !bags || !dropped || !tokens)
        return util::expected_t::fail(util::status_code_t::out_of_memory_k, "bagged readback scratch alloc failed");

    if (util::expected_t read = util::copy_device_to_host(mixed.segment_offsets.view(), offsets.span(), stream); !read)
        return read;
    if (util::expected_t read = util::copy_device_to_host(mixed.segment_lengths.view(), lengths.span(), stream); !read)
        return read;
    if (util::expected_t read = util::copy_device_to_host(mixed.segment_bag.view(), bags.span(), stream); !read)
        return read;
    if (util::expected_t read = util::copy_device_to_host(mixed.dropped.view(), dropped.span(), stream); !read)
        return read;
    if (util::expected_t read = util::copy_device_to_host(mixed.tokens.view().flatten<1>(), tokens.span(), stream);
        !read)
        return read;
    if (CUresult synced = cuStreamSynchronize(stream.get()); synced != CUDA_SUCCESS)
        return util::cuda_failure(synced, "bagged readback sync failed for member {}", member);

    util::u32_t mirrored_dropped = 0;
    mirrored_geometry_t const mirror = mirror_bag_geometry_(member, world_member_count, seed, epoch, rows, columns,
                                                            mirrored_dropped);
    expect_true(dropped[0] == mirrored_dropped, "member {}: counted {} dropped segments, the mirror drops {}", member,
                dropped[0], mirrored_dropped);
    util::u32_t const placed = static_cast<util::u32_t>(mirror.size());
    for (util::u32_t s = 0; s != placed; ++s) {
        mirrored_segment_t const& want = mirror[s];
        expect_true(offsets[s] == want.offset && lengths[s] == want.length && bags[s] == want.bag,
                    "member {} segment {}: placed (offset={},length={},bag={}) != mirror ({},{},{})", member, s,
                    offsets[s], lengths[s], bags[s], want.offset, want.length, want.bag);
        expect_true(s == 0 || bags[s] >= bags[s - 1u], "member {} segment {}: bag {} breaks the cohesive order", member,
                    s, bags[s]);
        for (util::u32_t j = 0; j != L; ++j) {
            util::token_id_t const got = tokens[offsets[s] + j];
            util::token_id_t const want_token = signature_token_(want.source_member, want.source_segment, j);
            expect_true(got == want_token, "member {} segment {} column {}: token {} != source ({},{}) token {}",
                        member, s, j, got, want.source_member, want.source_segment, want_token);
        }
    }
    for (util::u32_t s = placed; s != S; ++s)
        expect_true(lengths[s] == 0u && bags[s] == pad_bag_k,
                    "member {} segment {}: the tail must be a no-bag PAD (length={} bag={})", member, s, lengths[s],
                    bags[s]);
    return {};
}

/** One tier's per-member surfaces — the movable half of the world stand-up both tiers share;
 *  the domains and the mixer pin references into it, so they are constructed in the tier itself
 *  and never moved. */
struct mixer_members_t {

    /** The visible device ordinals, table-shaped for the arena visibility lists. */
    st::basic_vector<util::i32_t> devices {};

    /** One stream per member — every fabrication, mix, and readback enqueues here. */
    st::basic_vector<util::gpu_stream_t> streams {};

    /** One peer-visible arena per member — the batches and the mixer's scratch live off these. */
    st::basic_vector<util::gpu_arena_t> arenas {};

    /** Each member's loaded (source) batch. */
    st::basic_vector<multimodal_batch_on_gpu_t> loaded {};

    /** Each member's mixed (destination) batch. */
    st::basic_vector<multimodal_batch_on_gpu_t> mixed {};
};

/** Stand up @p member_count members' streams, peer-visible arenas, and loaded / mixed batches at
 *  @p shape — the bring-up both mixer tiers share; fabrication and the domains stay tier-local. */
[[nodiscard]] static util::expected<mixer_members_t> stand_up_members_(util::u32_t member_count,
                                                                       multimodal_batch_shape_t shape) noexcept {
    using result_t = util::expected<mixer_members_t>;
    mixer_members_t members;
    if (st::failed(members.devices.resize(member_count)))
        return result_t::fail(util::status_code_t::out_of_memory_k, "device table alloc failed");
    for (util::u32_t member = 0; member != member_count; ++member)
        members.devices[member] = static_cast<util::i32_t>(member);
    if (st::failed(members.streams.reserve(member_count)) || st::failed(members.arenas.reserve(member_count)) ||
        st::failed(members.loaded.reserve(member_count)) || st::failed(members.mixed.reserve(member_count)))
        return result_t::fail(util::status_code_t::out_of_memory_k, "per-member table reservation failed");
    for (util::u32_t member = 0; member != member_count; ++member) {
        auto [stream, make_stream_status] = util::gpu_stream_t::make(static_cast<int>(member));
        if (!make_stream_status) return result_t::fail(std::move(make_stream_status));
        if (st::failed(members.streams.push_back(std::move(stream))))
            return result_t::fail(util::status_code_t::out_of_memory_k, "member {} stream append failed", member);
        util::switch_to_gpu_t gpu_scope {members.streams[member]};
        auto [arena, make_arena_status] = util::gpu_arena_t::make(static_cast<int>(member), 512ull << 20,
                                                                  members.devices);
        if (!make_arena_status) return result_t::fail(std::move(make_arena_status));
        if (st::failed(members.arenas.push_back(std::move(arena))))
            return result_t::fail(util::status_code_t::out_of_memory_k, "member {} arena append failed", member);
        auto [loaded_batch, make_loaded_batch_status] = multimodal_batch_on_gpu_t::make(
            util::gpu_arena_alloc_t {members.arenas[member]}, shape);
        if (!make_loaded_batch_status) return result_t::fail(std::move(make_loaded_batch_status));
        if (st::failed(members.loaded.push_back(std::move(loaded_batch))))
            return result_t::fail(util::status_code_t::out_of_memory_k, "member {} loaded batch append failed", member);
        auto [mixed_batch, make_mixed_batch_status] = multimodal_batch_on_gpu_t::make(
            util::gpu_arena_alloc_t {members.arenas[member]}, shape);
        if (!make_mixed_batch_status) return result_t::fail(std::move(make_mixed_batch_status));
        if (st::failed(members.mixed.push_back(std::move(mixed_batch))))
            return result_t::fail(util::status_code_t::out_of_memory_k, "member {} mixed batch append failed", member);
        util::expected_t zeroed = util::zero_async(members.loaded[member].dropped.span(), members.streams[member]);
        if (zeroed) zeroed = util::zero_async(members.mixed[member].dropped.span(), members.streams[member]);
        if (!zeroed) return result_t::fail(std::move(zeroed));
    }
    return result_t {std::move(members)};
}

/**
 *  @brief The mixer tier: stand up the domains over every visible GPU — one process per
 *      host_domain, reading the same @c UFORM_HOST_DOMAIN_COUNT contract as the base suite, so a
 *      two-node run exercises the InfiniBand full-batch exchange.
 *
 *  It fabricates each member's loaded batch under its @b global member signature, runs one mix
 *  through the eager overload, and verifies every mixed segment against its world source. Skips
 *  cleanly when the box cannot host a multicast multimem_domain (the make fails loudly).
 */
static case_result_t test_routes_segments() {
    int device_count = 0;
    if (cuDeviceGetCount(&device_count) != CUDA_SUCCESS || device_count < 1)
        return case_result_t::skipped(no_gpu_device_k);
    util::u32_t const member_count = fu::min_of<util::u32_t>(static_cast<util::u32_t>(device_count), members_cap_k);
    util::u64_t const seed = 0xA5A5C0FFEEULL;
    util::data_epoch_t const epoch {1u};

    util::u32_t const host_domain_index = util::host_domain_index_from_environment();
    util::u32_t const host_domain_count = util::host_domain_count_from_environment();
    auto [topology, bootstrap_status] = util::mesh_topology_t::bootstrap(host_domain_index, host_domain_count,
                                                                         &pause_between_polls, member_count);
    if (!expect_true(bootstrap_status, "mesh bootstrap: {}", bootstrap_status.description_)) return {};
    // The routing world spans every multimem_domain's members; this domain's GPUs cover the world
    // coordinates [first, first + member_count) the signatures and the oracle both use.
    util::u32_t const world_member_count = member_count * host_domain_count;
    util::u32_t const first_gpu_index_in_colocated_domain = topology.shape.gpu_index_of_multimem_member(
        topology.multimem_domain_index(), 0u);

    multimodal_batch_shape_t shape;
    shape.max_rows_per_gpu = 2u;
    shape.max_segments_per_gpu = segments_per_member_k;
    shape.max_columns_per_row = 64u; // > segments_per_member_k·tokens_per_segment_k, so no packing row-jump

    auto [members, stand_up_status] = stand_up_members_(member_count, shape);
    if (!expect_true(stand_up_status, "member stand-up: {}", stand_up_status.description_)) return {};
    for (util::u32_t member = 0; member != member_count; ++member)
        if (util::expected_t built = fabricate_loaded_(
                members.loaded[member], first_gpu_index_in_colocated_domain + member, members.streams[member]);
            !expect_true(built, "fabricate loaded on GPU {}: {}", member, built.description_))
            return {};

    routing_is_a_bijection(world_member_count, seed, epoch);

    auto [multimem, make_multimem_status] = util::multimem_domain_t::make(
        topology, members.devices, 1024u + multimodal_batch_mixer_t::payload_scalars_needed(shape, topology.shape),
        members.streams, wait_limit_k.count());
    if (!expect_true(make_multimem_status, "multimem_domain make (needs multicast capability): {}",
                     make_multimem_status.description_))
        return {};
    auto [colocated, make_colocated_status] = util::colocated_domain_t::make(topology);
    if (!expect_true(make_colocated_status, "colocated_domain make: {}", make_colocated_status.description_)) return {};
    auto [mixer, make_mixer_status] = multimodal_batch_mixer_t::make(multimem, colocated, members.arenas, shape, seed,
                                                                     members.streams);
    if (!expect_true(make_mixer_status, "mixer make: {}", make_mixer_status.description_)) return {};

    std::span<multimodal_batch_on_gpu_t const> const loaded_view = members.loaded;
    if (util::expected_t mixed_ok = mixer(epoch, loaded_view, members.mixed, members.streams);
        !expect_true(mixed_ok, "the mix should run: {}", mixed_ok.description_))
        return {};

    for (util::u32_t member = 0; member != member_count; ++member)
        if (util::expected_t verified = verify_mixed_(members.mixed[member],
                                                      first_gpu_index_in_colocated_domain + member, world_member_count,
                                                      seed, epoch, members.streams[member]);
            !verified)
            expect_true(false, "verify mixed on GPU {}: {}", member, verified.description_);
    return {};
}

/**
 *  @brief The bag-cohesion tier: the same bring-up as @ref test_routes_segments but with a nonzero
 *      bag budget, bagged fabrication, and the exact host mirror of the bag walk as the oracle.
 *
 *  Runs twice — @p rows × @p columns_per_row sized roomy (every bag lands) and tight (the grid
 *  barely holds one member's own load, so a permutation draw that concentrates heavy bags on one
 *  destination forces whole-bag drops; the mirror predicts each drop exactly, so the tier is a
 *  strict oracle either way). Every source batch stays valid — only the receive side can overflow.
 */
static case_result_t test_routes_bags(util::u32_t rows, util::u32_t columns_per_row) {
    int device_count = 0;
    if (cuDeviceGetCount(&device_count) != CUDA_SUCCESS || device_count < 1)
        return case_result_t::skipped(no_gpu_device_k);
    util::u32_t const member_count = fu::min_of<util::u32_t>(static_cast<util::u32_t>(device_count), members_cap_k);
    util::u64_t const seed = 0xBA65C0FFEEULL;
    util::data_epoch_t const epoch {2u};

    util::u32_t const host_domain_index = util::host_domain_index_from_environment();
    util::u32_t const host_domain_count = util::host_domain_count_from_environment();
    auto [topology, bootstrap_status] = util::mesh_topology_t::bootstrap(host_domain_index, host_domain_count,
                                                                         &pause_between_polls, member_count);
    if (!expect_true(bootstrap_status, "mesh bootstrap: {}", bootstrap_status.description_)) return {};
    util::u32_t const world_member_count = member_count * host_domain_count;
    util::u32_t const first_gpu_index_in_colocated_domain = topology.shape.gpu_index_of_multimem_member(
        topology.multimem_domain_index(), 0u);

    multimodal_batch_shape_t shape;
    shape.max_rows_per_gpu = rows;
    shape.max_segments_per_gpu = segments_per_member_k;
    shape.max_columns_per_row = columns_per_row;
    shape.max_bags_per_gpu = bags_per_member_k;

    auto [members, stand_up_status] = stand_up_members_(member_count, shape);
    if (!expect_true(stand_up_status, "member stand-up: {}", stand_up_status.description_)) return {};
    for (util::u32_t member = 0; member != member_count; ++member)
        if (util::expected_t built = fabricate_bagged_(
                members.loaded[member], first_gpu_index_in_colocated_domain + member, members.streams[member]);
            !expect_true(built, "fabricate bagged on GPU {}: {}", member, built.description_))
            return {};

    auto [multimem, make_multimem_status] = util::multimem_domain_t::make(
        topology, members.devices, 1024u + multimodal_batch_mixer_t::payload_scalars_needed(shape, topology.shape),
        members.streams, wait_limit_k.count());
    if (!expect_true(make_multimem_status, "multimem_domain make (needs multicast capability): {}",
                     make_multimem_status.description_))
        return {};
    auto [colocated, make_colocated_status] = util::colocated_domain_t::make(topology);
    if (!expect_true(make_colocated_status, "colocated_domain make: {}", make_colocated_status.description_)) return {};
    auto [mixer, make_mixer_status] = multimodal_batch_mixer_t::make(multimem, colocated, members.arenas, shape, seed,
                                                                     members.streams);
    if (!expect_true(make_mixer_status, "mixer make: {}", make_mixer_status.description_)) return {};

    std::span<multimodal_batch_on_gpu_t const> const loaded_view = members.loaded;
    if (util::expected_t mixed_ok = mixer(epoch, loaded_view, members.mixed, members.streams);
        !expect_true(mixed_ok, "the bagged mix should run: {}", mixed_ok.description_))
        return {};

    for (util::u32_t member = 0; member != member_count; ++member)
        if (util::expected_t verified = verify_bagged_(members.mixed[member],
                                                       first_gpu_index_in_colocated_domain + member, world_member_count,
                                                       seed, epoch, members.streams[member]);
            !verified)
            expect_true(false, "verify bagged on GPU {}: {}", member, verified.description_);
    return {};
}

int main(int argc, char** argv) {
    // The mixer runs eagerly (no captured graph) and its barrier is a device spin. Under CUDA's
    // default lazy module loading, the first kernel launched after the barrier blocks on its module
    // load while the barrier already spins — stranding the single host thread that still owes the
    // other members' arrivals. Eager loading resolves every module up front, so the post-barrier
    // launch never touches the loader. The suite initializes CUDA, so the setting lands before it.
    ::setenv("CUDA_MODULE_LOADING", "EAGER", 1);
    start_test_run(argc, argv);
    [[maybe_unused]] CUresult const initialized = cuInit(0); // a failure leaves no device to count, so each case skips
    run_test_case("test_routes_segments", test_routes_segments);
    run_test_case("test_routes_bags(2, 64)", test_routes_bags, 2u, 64u); // every bag lands whole
    // Tight: 20 cells hold one member's own 18-token load (sources stay valid) but not
    // every receive combination — a destination drawing two heavy bags overflows and must
    // drop whole, mirror-verified.
    run_test_case("test_routes_bags(1, 20)", test_routes_bags, 1u, 20u);
    return finish_test_run();
}
