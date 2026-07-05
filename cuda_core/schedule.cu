/**
 *  @brief The device-side barrier-generation increment declared in @ref schedule.cuh — the one kernel
 *      that lets a captured collective region self-advance its generation.
 *  @file cuda_core/schedule.cu
 *  @author Ash Vardanian
 *  @date July 5, 2026
 */
#include <cuda.h> // `CUdeviceptr`, driver launch surface

#include "cuda_core/schedule.cuh" // `barrier_generation_t`, `launch_kernel`, `u32r1_span_t`

using namespace unum::uform::core;

/**
 *  @brief One thread advances the generation scalar and publishes it system-wide, so a captured replay
 *      re-runs the bump on every launch instead of freezing at the capture-time value.
 *
 *  __grid_size__ @c [1,1,1] , one thread; no dynamic shared memory.
 */
static __global__ void barrier_generation_increment_kernel_(u32r1_span_t generation, u32_t increment) {
    if (threadIdx.x == 0u) {
        generation[0] += increment;
        __threadfence_system();
    }
}

expected_t barrier_generation_t::advance(gpu_stream_t &stream) noexcept {
    return launch_kernel(reinterpret_cast<void const *>(&barrier_generation_increment_kernel_), dim3 {1, 1, 1},
                         dim3 {1, 1, 1}, stream, device_generation_, increment_);
}
