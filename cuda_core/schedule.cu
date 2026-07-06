/**
 *  @brief The device-side barrier-round increment declared in @ref schedule.cuh — the one kernel
 *      that lets a captured collective region self-advance its round.
 *  @file cuda_core/schedule.cu
 *  @author Ash Vardanian
 *  @date July 5, 2026
 */
#include <cuda.h> // `CUdeviceptr`, driver launch surface

#include "cuda_core/schedule.cuh" // `barrier_round_t`, `launch_kernel`, `u32r1_span_t`

using namespace unum::uform::core;

/**
 *  @brief One thread advances the round scalar and publishes it system-wide, so a captured replay
 *      re-runs the bump on every launch instead of freezing at the capture-time value.
 *
 *  __grid_size__ @c [1,1,1] , one thread; no dynamic shared memory.
 */
static __global__ void barrier_round_increment_kernel_(u32r1_span_t round, u32_t increment) {
    if (threadIdx.x == 0u) {
        round[0] += increment;
        __threadfence_system();
    }
}

expected_t barrier_round_t::advance(gpu_stream_t &stream) noexcept {
    return launch_kernel(reinterpret_cast<void const *>(&barrier_round_increment_kernel_), dim3 {1, 1, 1},
                         dim3 {1, 1, 1}, stream, device_round_, increment_);
}
