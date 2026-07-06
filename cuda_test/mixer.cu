/**
 *  @brief Conformance suite for @ref multimodal_batch_mixer_t — the cross-GPU segment shuffle. Builds the
 *      composition root ( @ref mesh_topology_t::bootstrap → per-GPU arenas / streams →
 *      @ref multimem_domain_t → @ref colocated_domain_t → the mixer), fabricates every member's loaded batch
 *      with a token signature distinct per @c (global_member,segment,column) , runs one mix, and asserts each
 *      mixed segment equals its @ref epoch_permutation source — the routing bijection, the length / modality
 *      descriptor travel, and the token content. One process per host_domain under the same
 *      @c UFORM_HOST_DOMAIN_COUNT contract as the base suite: one node exercises the NVLink island alone,
 *      two nodes add the InfiniBand full-batch exchange to same-index landings.
 *  @file cuda_test/mixer.cu
 *  @author Ash Vardanian
 *  @date July 6, 2026
 */
#include <cstddef> // `std::size_t`
#include <cstdlib> // `::setenv`, `std::getenv`, `std::atoi` — eager module loading, the host_domain contract

#include <utility>   // `std::move`
#include <algorithm> // `std::min`
#include <span>      // `std::span`

#include <cuda.h> // `cuInit`, `cuDeviceGetCount`, `cuStreamSynchronize`

#include "cuda_core/types.hpp"                  // `epoch_permutation`, `data_epoch_t`, `token_id_t`, `modality_t`
#include "cuda_core/array.hpp"                  // `array`
#include "cuda_core/gpu_allocators.hpp"         // `gpu_arena_t`, `gpu_arena_alloc_t`, `pinned_alloc`
#include "cuda_core/schedule.cuh"               // `gpu_stream_t`, `switch_to_gpu_t`, `copy_host_to_device`
#include "cuda_core/mesh.hpp"                   // `mesh_topology_t`, `host_domain_index_from_environment`
#include "cuda_core/multimem_collectives.cuh"   // `multimem_domain_t`
#include "cuda_core/infiniband_collectives.cuh" // `colocated_domain_t`
#include "cuda_core/multimodal_batch.hpp"       // `multimodal_batch_on_gpu_t`, the shape, the mixer
#include "cuda_test/harness.hpp"                // `test_suite_t`

using namespace unum::uform::core;
using namespace unum::uform::test;

/** @brief Segments per member — the routing axis; small, so the whole world permutation is easy to mirror. */
inline constexpr u32_t segments_per_member_k = 8u;
/** @brief Tokens per segment — a short run whose per-column signature the oracle reproduces exactly. */
inline constexpr u32_t tokens_per_segment_k = 3u;
/** @brief Members to exercise on one box — one NVLink island's worth, capped by the visible device count. */
inline constexpr u32_t members_cap_k = 8u;

/** @brief A rank-1 pinned host tensor of @p value_type_ — the sanctioned host staging / readback buffer. */
template <typename value_type_>
using pinned1_t = nk::tensor<value_type_, pinned_alloc<value_type_>, 1>;

/** @brief The deterministic token a member's segment writes at a column — distinct per @c (member,segment,column)
 *      so a mislaid route mismatches. Wraps into @ref token_id_t ; the oracle wraps identically. */
[[nodiscard]] static token_id_t signature_token_(u32_t member, u32_t segment, u32_t column) noexcept {
    return static_cast<token_id_t>((member * 131u + segment) * 17u + column + 1u);
}

/** @brief Fabricate member @p member 's loaded batch: @ref segments_per_member_k text segments, each
 *      @ref tokens_per_segment_k columns, packed contiguously into row 0, tokens carrying @ref signature_token_ .
 *      Staged host→device through the pinned + @ref copy_host_to_device path, synchronized before returning.
 *  @param[inout] batch The member's loaded batch, its geometry and tokens overwritten.
 *  @param[in] member This member's @b global index across every island, folded into every token signature.
 *  @param[inout] stream The member's stream the staging enqueues on. */
[[nodiscard]] static expected_t fabricate_loaded_(multimodal_batch_on_gpu_t &batch, u32_t member,
                                                  gpu_stream_t &stream) noexcept {
    u32_t const S = segments_per_member_k, L = tokens_per_segment_k;
    u32_t const rows = static_cast<u32_t>(batch.tokens.extent(0));
    u32_t const columns = static_cast<u32_t>(batch.tokens.extent(1));

    switch_to_gpu_t gpu_scope {stream};
    if (!gpu_scope.ok()) return expected_t::fail(status_t::cuda_error_k, "context entry failed for member {}", member);

    auto offsets = pinned1_t<u32_t>::try_zeros({static_cast<std::size_t>(S) + 1u}, pinned_alloc<u32_t> {});
    auto lengths = pinned1_t<u32_t>::try_zeros({S}, pinned_alloc<u32_t> {});
    auto modality = pinned1_t<modality_t>::try_zeros({S}, pinned_alloc<modality_t> {});
    auto tokens = pinned1_t<token_id_t>::try_zeros({static_cast<std::size_t>(rows) * columns},
                                                   pinned_alloc<token_id_t> {});
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
        return expected_t::fail(synced, "loaded fabrication sync failed for member {}", member);
    return {};
}

/** @brief Read member @p member 's mixed batch back and assert every segment equals its @ref epoch_permutation
 *      source: destination @c member·S+s pulls world source @c src , whose member is @c src/S and segment
 *      @c src%S ; the mixed segment must carry that source's length, its @c text_k modality, and its tokens.
 *  @param[inout] suite The conformance suite the checks report into.
 *  @param[inout] mixed The member's mixed batch to read back.
 *  @param[in] member This member's @b global index across every island — its destination range's base.
 *  @param[in] member_count The world's member count — the permutation's segment span is @c S·member_count .
 *  @param[in] seed The routing seed the mixer was built with.
 *  @param[in] epoch The data epoch the mix ran at.
 *  @param[inout] stream The member's stream the readback enqueues on. */
[[nodiscard]] static expected_t verify_mixed_(test_suite_t &suite, multimodal_batch_on_gpu_t &mixed, u32_t member,
                                              u32_t member_count, u64_t seed, data_epoch_t epoch,
                                              gpu_stream_t &stream) noexcept {
    u32_t const S = segments_per_member_k, L = tokens_per_segment_k;
    u32_t const rows = static_cast<u32_t>(mixed.tokens.extent(0));
    u32_t const columns = static_cast<u32_t>(mixed.tokens.extent(1));

    switch_to_gpu_t gpu_scope {stream};
    if (!gpu_scope.ok()) return expected_t::fail(status_t::cuda_error_k, "context entry failed for member {}", member);

    auto offsets = pinned1_t<u32_t>::try_zeros({static_cast<std::size_t>(S) + 1u}, pinned_alloc<u32_t> {});
    auto lengths = pinned1_t<u32_t>::try_zeros({S}, pinned_alloc<u32_t> {});
    auto modality = pinned1_t<modality_t>::try_zeros({S}, pinned_alloc<modality_t> {});
    auto tokens = pinned1_t<token_id_t>::try_zeros({static_cast<std::size_t>(rows) * columns},
                                                   pinned_alloc<token_id_t> {});
    if (!offsets || !lengths || !modality || !tokens)
        return expected_t::fail(status_t::bad_alloc_k, "mixed readback scratch alloc failed");

    if (expected_t read = copy_device_to_host(mixed.segment_offsets.view(), offsets.span(), stream); !read) return read;
    if (expected_t read = copy_device_to_host(mixed.segment_lengths.view(), lengths.span(), stream); !read) return read;
    if (expected_t read = copy_device_to_host(mixed.segment_modality.view(), modality.span(), stream); !read)
        return read;
    if (expected_t read = copy_device_to_host(mixed.tokens.view().flatten<1>(), tokens.span(), stream); !read)
        return read;
    if (CUresult synced = cuStreamSynchronize(stream.get()); synced != CUDA_SUCCESS)
        return expected_t::fail(synced, "mixed readback sync failed for member {}", member);

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

/** @brief Assert the routing is a bijection over the world's @c S·member_count segments — every destination maps
 *      to a distinct source — mirroring @ref epoch_permutation host-side so a broken permutation is caught even
 *      before the device gather. */
static void routing_is_a_bijection(test_suite_t &suite, u32_t member_count, u64_t seed, data_epoch_t epoch) {
    std::size_t const world = static_cast<std::size_t>(segments_per_member_k) * member_count;
    array<u8_t> seen;
    if (seen.try_resize(world) != status_t::success_k) return suite.check(false, "bijection scratch alloc failed");
    for (std::size_t index = 0; index != world; ++index) seen[index] = 0u;
    bool bijection = true;
    for (std::size_t destination = 0; destination != world; ++destination) {
        std::size_t const source = epoch_permutation(seed, epoch, destination, world);
        if (source >= world || seen[source] != 0u) bijection = false;
        else seen[source] = 1u;
    }
    suite.check(bijection, "the routing must be a bijection over {} world segments", world);
}

/** @brief The mixer tier: stand up the domains over every visible GPU — one process per host_domain,
 *      reading the same @c UFORM_HOST_DOMAIN_COUNT contract as the base suite, so a two-node run exercises
 *      the InfiniBand full-batch exchange — fabricate each member's loaded batch under its @b global member
 *      signature, run one mix through the eager overload, and verify every mixed segment against its world
 *      source. Skips cleanly when the box has no multicast-capable island (the domain make fails loudly). */
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
    // The routing world spans every island's members; this island's GPUs carry the global indices
    // [global_member_base, global_member_base + member_count) the signatures and the oracle both use.
    u32_t const world_member_count = member_count * host_domain_count;
    u32_t const global_member_base = topology.multimem_domain_index() * member_count;

    multimodal_batch_shape_t shape;
    shape.max_rows_per_gpu = 2u;
    shape.max_segments_per_gpu = segments_per_member_k;
    shape.max_columns_per_row = 64u; // > segments_per_member_k·tokens_per_segment_k, so no packing row-jump

    array<i32_t> devices;
    if (devices.try_resize(member_count) != status_t::success_k) return suite.check(false, "device table alloc failed");
    for (u32_t member = 0; member != member_count; ++member) devices[member] = static_cast<i32_t>(member);

    array<gpu_stream_t> streams;
    array<gpu_arena_t> arenas;
    array<multimodal_batch_on_gpu_t> loaded;
    array<multimodal_batch_on_gpu_t> mixed;
    if (streams.try_reserve(member_count) != status_t::success_k ||
        arenas.try_reserve(member_count) != status_t::success_k ||
        loaded.try_reserve(member_count) != status_t::success_k ||
        mixed.try_reserve(member_count) != status_t::success_k)
        return suite.check(false, "per-member table reservation failed");

    for (u32_t member = 0; member != member_count; ++member) {
        auto stream_made = gpu_stream_t::make(static_cast<int>(member));
        if (!stream_made) return suite.check(false, "stream make on GPU {}: {}", member, stream_made.description_);
        (void)streams.try_push_back(std::move(stream_made.value_));
        switch_to_gpu_t gpu_scope {streams[member]};
        auto arena_made = gpu_arena_t::make(static_cast<int>(member), 512ull << 20, devices.view());
        if (!arena_made) return suite.check(false, "arena make on GPU {}: {}", member, arena_made.description_);
        (void)arenas.try_push_back(std::move(arena_made.value_));
        auto loaded_made = multimodal_batch_on_gpu_t::make(gpu_arena_alloc_t {arenas[member]}, shape);
        if (!loaded_made)
            return suite.check(false, "loaded batch make on GPU {}: {}", member, loaded_made.description_);
        (void)loaded.try_push_back(std::move(loaded_made.value_));
        auto mixed_made = multimodal_batch_on_gpu_t::make(gpu_arena_alloc_t {arenas[member]}, shape);
        if (!mixed_made) return suite.check(false, "mixed batch make on GPU {}: {}", member, mixed_made.description_);
        (void)mixed.try_push_back(std::move(mixed_made.value_));
        if (expected_t built = fabricate_loaded_(loaded[member], global_member_base + member, streams[member]); !built)
            return suite.check(false, "fabricate loaded on GPU {}: {}", member, built.description_);
    }

    routing_is_a_bijection(suite, world_member_count, seed, epoch);

    auto multimem_made = multimem_domain_t::make(topology, devices.view(), 1024u,
                                                 std::span<gpu_stream_t> {streams.data(), streams.size()});
    if (!multimem_made)
        return suite.check(false, "multimem_domain make (needs a multicast island): {}", multimem_made.description_);
    multimem_domain_t multimem = std::move(multimem_made.value_);
    auto colocated_made = colocated_domain_t::make(topology);
    if (!colocated_made) return suite.check(false, "colocated_domain make: {}", colocated_made.description_);
    colocated_domain_t colocated = std::move(colocated_made.value_);
    auto mixer_made = multimodal_batch_mixer_t::make(multimem, colocated,
                                                     std::span<gpu_arena_t> {arenas.data(), arenas.size()}, shape, seed,
                                                     std::span<gpu_stream_t> {streams.data(), streams.size()});
    if (!mixer_made) return suite.check(false, "mixer make: {}", mixer_made.description_);
    multimodal_batch_mixer_t mixer = std::move(mixer_made.value_);

    std::span<multimodal_batch_on_gpu_t const> const loaded_view {loaded.data(), loaded.size()};
    if (expected_t mixed_ok = mixer(epoch, loaded_view,
                                    std::span<multimodal_batch_on_gpu_t> {mixed.data(), mixed.size()},
                                    std::span<gpu_stream_t> {streams.data(), streams.size()});
        !mixed_ok)
        return suite.check(false, "the mix should run: {}", mixed_ok.description_);

    for (u32_t member = 0; member != member_count; ++member)
        if (expected_t verified = verify_mixed_(suite, mixed[member], global_member_base + member, world_member_count,
                                                seed, epoch, streams[member]);
            !verified)
            suite.check(false, "verify mixed on GPU {}: {}", member, verified.description_);
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
    return suite.finish();
}
