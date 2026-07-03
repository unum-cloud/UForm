/**
 *  @brief Device-initiated collectives across "multimem_domains" over GPUDirect Async Kernel-Initiated 
 *      (GDAKI) network communications : the vertex all-gather and the force all-reduce-sum every
 *      cross-cluster loss step rides.
 * 
 *  @file cuda/primitives/multimem_collectives.cuh
 *  @author Ash Vardanian
 *  @date July 2, 2026
 *
 *  Both collectives walk the same @b recursive-doubling schedule over @c D = multimem_domain_count()
 *  domains: @c ceil(log2 D) rounds, one @ref colocated_link_t per round, partner
 *  @c this_domain^(1<<round) . Every kernel is a capture-safe graph node — the posts and waits are
 *  device-initiated ( @ref gdaki_post_write / @ref gdaki_load_acquire_u32 ), and per-step variability
 *  reaches replays only through device memory:
 *
 *      - @b all-gather — each round posts the contiguous run of domain blocks this domain already holds
 *        into the partner's symmetric buffer at the @b identical byte offset, so blocks land domain-major
 *        with no accumulate; after @c log2(D) rounds every domain holds all @c D blocks. This is how the
 *        contrastive losses receive remote vertex towers.
 *      - @b all-reduce-sum — each round posts this domain's Float32 shard into the partner's landing
 *        buffer, waits for the partner's, and folds it in with one round-to-nearest-even add per element
 *        ( @c __fadd_rn , no fused-multiply-add contraction) — the SPMD-static schedule fixes the fold
 *        order, so every domain associates identically, bit-for-bit. This is the cross-domain gradient
 *        reduce.
 *
 *  @section collectives_epochs Arrival flags are epochs, never events
 *
 *  Every round's arrival proof is a 4-byte @b inline flag write on the @b same queue pair as the payload:
 *  Reliable-Connection delivery is in-order, so the flag's arrival proves the payload landed. Flags are
 *  @b monotonic never-reset counters — a waiter spins with @c >= against the absolute target
 *  @c *flag_epoch+flag_epoch_offset+round+1 , so replays of a captured step need no flag reset (a reset
 *  would be a host write inside the replay loop). The host advances the epoch scalar once per step, past
 *  the sum of all collectives' round counts; @p flag_epoch_offset windows multiple collectives inside one
 *  captured step so their flags never alias.
 */
#pragma once

#include <cstdint> // `std::uint64_t`

#include <cuda_bf16.h> // `__nv_bfloat16` — the gathered vertex scalar

#include "types.cuh"                // `expected`, `expected_t`, `status_t`
#include "schedule.cuh"             // `gpu_stream_t`, the rank-1 view / span aliases
#include "infiniband_transport.cuh" // `colocated_link_t`, `gdaki_epoch_t`, the device post / drain inlines

namespace unum::uform::primitives {

/** @brief Rank-1 BFloat16 mutable device span, shaped @b [D·block_scalars] — the domain-major gathered
 *      vertex tower of all @c D multimem_domains. */
using bf16r1_span_t = nk::tensor_span<__nv_bfloat16, 1>;
/** @brief Rank-1 device view of the per-round link table, shaped @b [ceil(log2(D))] — one
 *      @ref colocated_link_t per recursive-doubling round, indexed by round. */
using colocated_links_view_t = nk::tensor_view<colocated_link_t, 1>;
/** @brief Rank-1 device view of the flag-epoch scalar, shaped @b [1] — the monotonic base every flag
 *      window keys on; the host advances it once per step. */
using gdaki_epoch_view_t = nk::tensor_view<gdaki_epoch_t, 1>;

/**
 *  @brief The static description of one all-gather invocation — everything about it that is not the
 *      buffer, the links, or the epoch, bundled so the launcher and its kernels take one argument
 *      instead of four lonely scalars.
 */
struct gdaki_allgather_plan_t {
    /** @brief Scalars per domain block; the gathered buffer spans @b [D·block_scalars] . */
    std::size_t block_scalars = 0;
    /** @brief Symmetric-heap byte offset of the gathered buffer — identical on every domain. */
    std::size_t landing_offset_bytes = 0;
    /** @brief This rank's multimem_domain index — the recursive-doubling coordinate. */
    u32_t this_multimem_domain = 0;
    /** @brief This collective's window inside the step's flag budget (see the epochs section). */
    u32_t flag_epoch_offset = 0;
};

#if defined(UFORM_WITH_INFINIBAND)

#if defined(__CUDACC__)

/** @brief Lane width of the single-block doorbell kernels; only lane 0 posts and spins. */
inline constexpr unsigned gdaki_doorbell_block_threads_k = 32u;
/** @brief Grid cap of the bandwidth-bound accumulate so a huge shard still launches bounded (grid-strided). */
inline constexpr unsigned gdaki_accumulate_grid_cap_k = 4096u;

/**
 *  @brief Post-and-drain step of one all-reduce round: lane 0 posts this domain's shard, then the arrival
 *      flag, to the round's partner, and drains the one signaled completion.
 *
 *  The payload write is unsignaled; only the 4-byte inline flag write is signaled — exactly one
 *  completion per round is produced and drained, so the small send ring never overflows across replays.
 */
static __global__ void gdaki_allreduce_post_kernel_(colocated_links_view_t links, u32_t round, f32_t const *shard,
                                                    std::size_t shard_scalars, gdaki_epoch_view_t flag_epoch) {
    if (threadIdx.x != 0) return;
    colocated_link_t const link = links[round];
    u32_t const flag_value = flag_epoch[0] + round + 1u;
    gdaki_post_write(link.queue, reinterpret_cast<u64_t>(shard), link.local_access_key,
                     link.remote_heap_base + link.landing_offset, link.remote_access_key,
                     static_cast<u32_t>(shard_scalars * sizeof(f32_t)), false, nullptr);
    gdaki_post_write(link.queue, 0u, link.local_access_key, link.remote_heap_base + link.remote_flag_offset,
                     link.remote_access_key, sizeof(u32_t), true, &flag_value);
    gdaki_drain_one(link.completion);
}

/** @brief Flag-wait step of one round: lane 0 acquire-spins until the inbound flag reaches the absolute
 *      target (monotonic @c >= — the flag never resets), then a system fence orders the partner's landed
 *      payload before the next kernel, propagated to the block by the closing @c __syncthreads . */
static __global__ void gdaki_flag_wait_kernel_(colocated_links_view_t links, u32_t round, u32_t flag_epoch_offset,
                                               gdaki_epoch_view_t flag_epoch) {
    if (threadIdx.x == 0) {
        colocated_link_t const link = links[round];
        u32_t const target = flag_epoch[0] + flag_epoch_offset + round + 1u;
        while (gdaki_load_acquire_u32(link.arrival_flag) < target) {
            // Spin until the partner's flag write for this round lands; its payload already did.
        }
        __threadfence_system();
    }
    __syncthreads();
}

/** @brief Accumulate step of one all-reduce round: fold the partner's landed shard into this domain's,
 *      one explicit round-to-nearest-even add per element — no fused-multiply-add contraction, so the
 *      SPMD-static fold order is bit-identical on every domain. */
static __global__ void gdaki_accumulate_f32_kernel_(colocated_links_view_t links, u32_t round, f32_t *shard,
                                                    std::size_t shard_scalars) {
    colocated_link_t const link = links[round];
    f32_t const *landing = reinterpret_cast<f32_t const *>(link.local_landing.data());
    std::size_t const stride = static_cast<std::size_t>(gridDim.x) * blockDim.x;
    for (std::size_t index = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x; index < shard_scalars;
         index += stride)
        shard[index] = __fadd_rn(shard[index], landing[index]);
}

/**
 *  @brief Post-and-drain step of one all-gather round: lane 0 posts the contiguous run of domain blocks
 *      this domain currently holds into the partner's symmetric @c gathered buffer at the @b same byte
 *      offset, then the flag, and drains the one completion.
 *
 *  At round @c r the held run spans domains @c [base,base+2^r) with @c base=this_domain&~(2^r-1) ;
 *  symmetric buffers make identical byte offsets address identical logical slots, so the partner receives
 *  exactly the half it lacks at the right domain-major position.
 */
static __global__ void gdaki_allgather_post_kernel_(colocated_links_view_t links, u32_t round,
                                                    gdaki_allgather_plan_t plan, char const *gathered_bytes,
                                                    gdaki_epoch_view_t flag_epoch) {
    if (threadIdx.x != 0) return;
    colocated_link_t const link = links[round];
    // The recursive-doubling bit math keys on the link's true round; the loop index drives only the
    // table slot and the monotonic flag window.
    std::size_t const block_bytes = plan.block_scalars * sizeof(__nv_bfloat16);
    u32_t const bit_round = link.round_index;
    u32_t const group_base = plan.this_multimem_domain & ~((1u << bit_round) - 1u);
    std::size_t const send_byte_offset = static_cast<std::size_t>(group_base) * block_bytes;
    std::size_t const send_bytes = (static_cast<std::size_t>(1u) << bit_round) * block_bytes;
    u32_t const flag_value = flag_epoch[0] + plan.flag_epoch_offset + round + 1u;

    gdaki_post_write(link.queue, reinterpret_cast<u64_t>(gathered_bytes + send_byte_offset), link.local_access_key,
                     link.remote_heap_base + plan.landing_offset_bytes + send_byte_offset, link.remote_access_key,
                     static_cast<u32_t>(send_bytes), false, nullptr);
    gdaki_post_write(link.queue, 0u, link.local_access_key, link.remote_heap_base + link.remote_flag_offset,
                     link.remote_access_key, sizeof(u32_t), true, &flag_value);
    gdaki_drain_one(link.completion);
}

/**
 *  @brief Device-initiated inter-domain all-reduce-sum of a Float32 @p shard — the cross-domain gradient
 *      reduce, enqueued on @p stream as capture-safe graph nodes.
 *
 *  Rounds run in order (round @c r 's accumulate feeds round @c r+1 's post); within a round:
 *  post-and-drain, wait, accumulate. An empty @p links table is the single-domain identity.
 */
[[nodiscard]] inline expected_t launch_gdaki_allreduce_sum(f32r1_span_t shard, colocated_links_view_t links,
                                                           gdaki_epoch_view_t flag_epoch,
                                                           gpu_stream_t const &stream) noexcept {
    if (links.empty()) return {};
    if (shard.empty()) return expected_t::fail(status_t::invalid_input_k, "all-reduce of an empty shard");
    if (flag_epoch.empty()) return expected_t::fail(status_t::invalid_input_k, "all-reduce needs the epoch scalar");

    std::size_t const shard_scalars = shard.extent(0);
    unsigned accumulate_grid = static_cast<unsigned>((shard_scalars + kernel_block_size_default_k - 1) /
                                                     kernel_block_size_default_k);
    if (accumulate_grid == 0u) accumulate_grid = 1u;
    if (accumulate_grid > gdaki_accumulate_grid_cap_k) accumulate_grid = gdaki_accumulate_grid_cap_k;

    dim3 const doorbell_grid {1u, 1u, 1u};
    dim3 const doorbell_block {gdaki_doorbell_block_threads_k, 1u, 1u};
    f32_t const *shard_read = shard.data();
    f32_t *shard_write = shard.data();
    u32_t const allreduce_flag_offset = 0u;
    for (u32_t round = 0; round != static_cast<u32_t>(links.extent(0)); ++round) {
        expected_t launched = launch_kernel(reinterpret_cast<void const *>(&gdaki_allreduce_post_kernel_),
                                            doorbell_grid, doorbell_block, stream, links, round, shard_read,
                                            shard_scalars, flag_epoch);
        if (launched)
            launched = launch_kernel(reinterpret_cast<void const *>(&gdaki_flag_wait_kernel_), doorbell_grid,
                                     doorbell_block, stream, links, round, allreduce_flag_offset, flag_epoch);
        if (launched)
            launched = launch_kernel(reinterpret_cast<void const *>(&gdaki_accumulate_f32_kernel_),
                                     dim3 {accumulate_grid, 1u, 1u}, dim3 {kernel_block_size_default_k, 1u, 1u}, stream,
                                     links, round, shard_write, shard_scalars);
        if (!launched) return launched;
    }
    return {};
}

/**
 *  @brief Device-initiated inter-domain all-gather of domain-major BFloat16 blocks inside @p gathered —
 *      how every domain receives every other domain's vertex tower, enqueued on @p stream as
 *      capture-safe graph nodes.
 *
 *  @param[in] gathered The symmetric @b [D·block_scalars] buffer; this domain's own block must already
 *      sit at its domain-major slot. After @c log2(D) rounds every block is filled.
 *  @param[in] plan The invocation's static description: block size, symmetric offset, this domain's
 *      coordinate, and the flag window.
 */
[[nodiscard]] inline expected_t launch_gdaki_allgather(bf16r1_span_t gathered, gdaki_allgather_plan_t plan,
                                                       colocated_links_view_t links, gdaki_epoch_view_t flag_epoch,
                                                       gpu_stream_t const &stream) noexcept {
    if (links.empty()) return {};
    if (gathered.empty() || plan.block_scalars == 0)
        return expected_t::fail(status_t::invalid_input_k, "all-gather of an empty buffer (block_scalars={})",
                                plan.block_scalars);
    if (flag_epoch.empty()) return expected_t::fail(status_t::invalid_input_k, "all-gather needs the epoch scalar");

    char const *gathered_bytes = reinterpret_cast<char const *>(gathered.data());
    dim3 const doorbell_grid {1u, 1u, 1u};
    dim3 const doorbell_block {gdaki_doorbell_block_threads_k, 1u, 1u};
    for (u32_t round = 0; round != static_cast<u32_t>(links.extent(0)); ++round) {
        expected_t launched = launch_kernel(reinterpret_cast<void const *>(&gdaki_allgather_post_kernel_),
                                            doorbell_grid, doorbell_block, stream, links, round, plan, gathered_bytes,
                                            flag_epoch);
        if (launched)
            launched = launch_kernel(reinterpret_cast<void const *>(&gdaki_flag_wait_kernel_), doorbell_grid,
                                     doorbell_block, stream, links, round, plan.flag_epoch_offset, flag_epoch);
        if (!launched) return launched;
    }
    return {};
}

#endif // defined(__CUDACC__)

#else // !defined(UFORM_WITH_INFINIBAND)

/** @brief Without the ibverbs build gate a multi-domain all-reduce cannot exist; the single-domain
 *      identity (an empty link table) still succeeds. */
[[nodiscard]] inline expected_t launch_gdaki_allreduce_sum(f32r1_span_t, colocated_links_view_t links,
                                                           gdaki_epoch_view_t, gpu_stream_t const &) noexcept {
    if (links.empty()) return {};
    return expected_t::fail(status_t::other_k,
                            "built without UFORM_WITH_INFINIBAND — rebuild with libibverbs + libmlx5 for " //
                            "cross-domain collectives");
}

/** @brief The all-gather sibling of the gate stub above. */
[[nodiscard]] inline expected_t launch_gdaki_allgather(bf16r1_span_t, gdaki_allgather_plan_t,
                                                       colocated_links_view_t links, gdaki_epoch_view_t,
                                                       gpu_stream_t const &) noexcept {
    if (links.empty()) return {};
    return expected_t::fail(status_t::other_k,
                            "built without UFORM_WITH_INFINIBAND — rebuild with libibverbs + libmlx5 for " //
                            "cross-domain collectives");
}

#endif // defined(UFORM_WITH_INFINIBAND)

} // namespace unum::uform::primitives
