/**
 *  @file src/utilities/multimem_switch.cu
 *  @author Ash Vardanian
 *  @date July 3, 2026
 *  @brief The switch-tier kernels: every @c multimem PTX instruction of the project, compiled
 *      device-only for the multicast-capable codes into the bundle @ref multimem_switch_module_t
 *      loads by name.
 *
 *  The kernels keep C linkage, so their names in the bundle are the ones spelled here, and no host
 *  code of this unit is ever linked: a build without a multicast-capable code embeds an empty
 *  bundle instead.
 */
#include "uform/utilities/multimem_collectives.cuh"

using namespace unum::uform::utilities;

/** One switch-reduced load: the sum of this address across every bound replica. */
[[nodiscard]] static __device__ inline f32_t multimem_load_reduce_add_(f32_t const *multimem_address) noexcept {
    f32_t value;
    asm volatile("multimem.ld_reduce.relaxed.sys.global.add.f32 %0, [%1];"
                 : "=f"(value)
                 : "l"(multimem_address)
                 : "memory");
    return value;
}

/** One multicast release-add on the arrival counter of every replica. */
static __device__ inline void multimem_red_add_release_(u32_t *multimem_address, u32_t value) noexcept {
    asm volatile("multimem.red.release.sys.global.add.u32 [%0], %1;" : : "l"(multimem_address), "r"(value) : "memory");
}

/**
 *  @brief Domain barrier: lane 0 release-adds 1 into every replica's counter, then acquire-spins on
 *      its @b own counter until all @c member_count arrivals of this round landed, and finally
 *      self-advances its @b own round scalar.
 *
 *  Monotonic — neither counter ever resets, so a captured step replays cleanly, and every round
 *  read and write stays on the member's own stream: no enqueue order can hand a barrier a stale
 *  round. Traps once the spin outlasts @c member.wait_nanoseconds .
 *
 *  __grid_size__ @c [1,1,1] , one warp; no dynamic shared memory.
 */
extern "C" __global__ void multimem_barrier_switch_kernel_(multimem_member_t member) {
    if (threadIdx.x != 0u) return;
    u32_t const target = *member.round + member.member_count;
    multimem_red_add_release_(member.multimem_flag, 1u);
    u64_t const deadline = global_nanoseconds() + member.wait_nanoseconds;
    while (multimem_load_acquire_(member.own_flag()) < target)
        if (global_nanoseconds() > deadline)
            trap_with(member.fault, gpu_fault_code_t::barrier_deadline_k, member.member_index, target, 0u);
    __threadfence_system();
    *member.round = target; // Self-advance: only this member's own kernels ever read its round.
}

/**
 *  @brief Reduce-scatter shard: @c destination[i] = Σ_members payload[shard_begin+i] , one
 *      switch-reduced load per element — deterministic with no fold order at all.
 *
 *  __grid_size__ grid-stride over @c shard_scalars ; no dynamic shared memory.
 */
extern "C" __global__ void multimem_reduce_scatter_switch_kernel_(multimem_member_t member, f32r1_span_t destination,
                                                                  std::size_t shard_begin) {
    std::size_t const shard_scalars = destination.extent(0);
    std::size_t const stride = static_cast<std::size_t>(gridDim.x) * blockDim.x;
    for (std::size_t index = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x; index < shard_scalars;
         index += stride)
        destination[index] = multimem_load_reduce_add_(member.multimem_payload + shard_begin + index);
}

/**
 *  @brief The optimizer's domain-reduced gradient: one switch-fused load per element from the
 *      consumer's window at @p shard_begin , scaled by @p world_scale into the world mean.
 *
 *  __grid_size__ grid-stride; no dynamic shared memory.
 */
extern "C" __global__ void adafactor_reduce_gradient_switch_kernel_(multimem_member_t member, f32r1_span_t destination,
                                                                    std::size_t shard_begin, f32_t world_scale) {
    std::size_t const total = destination.extent(0);
    std::size_t const stride = static_cast<std::size_t>(gridDim.x) * blockDim.x;
    for (std::size_t index = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x; index < total;
         index += stride)
        destination[index] = multimem_load_reduce_add_(member.multimem_payload + shard_begin + index) * world_scale;
}
