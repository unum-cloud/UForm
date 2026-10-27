/**
 *  @file src/train/numerics.hpp
 *  @author Ash Vardanian
 *  @date October 26, 2026
 *  @brief The one mapping from the capabilities a mesh agreed on to the kernel table every training
 *      component calls, which @c trainer.cu fills from the GPU class's kits.
 */
#pragma once

#include <numkong/capabilities.h> // `nk_capability_t`

#include "uform/utilities/vram_nvidia_schedule.cuh" // `gpu_numerics_t`

namespace unum::uform::train {

using utilities::gpu_numerics_t;

/** The NumKong kernels a run over GPUs of @p capabilities calls: those of the newest GPU class
 *  the mask holds. */
[[nodiscard]] gpu_numerics_t gpu_numerics_of(nk_capability_t capabilities) noexcept;

} // namespace unum::uform::train
