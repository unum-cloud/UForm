/**
 *  @file include/uform/encoder/vram_kits.hpp
 *  @author Ash Vardanian
 *  @date October 25, 2026
 *  @brief The GPU kits the encoder tower runs on: NumKong's packed products, packed attention and
 *      element-wise steps per NVIDIA and AMD generation, and a ladder per family whose floor is the
 *      vendor's generic kit.
 *
 *  Every kernel a kit names queues on the stream it is handed. Where a generation carries only some
 *  of a family's kernels, its kit borrows the rest from its vendor's earlier generations, and
 *  @c capability_k is the union of every bit the kit calls, as in @ref ram_kits.hpp . NVIDIA and
 *  AMD masks never overlap, so one ladder per family serves both vendors.
 */
#pragma once

#include <type_traits> // `std::conditional_t`

#include <numkong/attention.h> // `nk_attention_pack_*`, `nk_attention_packed_*`
#include <numkong/cast.h>      // `nk_cast_*`
#include <numkong/dots.h>      // `nk_dots_pack_*`, `nk_dots_packed_*`
#include <numkong/each.h>      // `nk_each_*`

#include "uform/encoder/entries.hpp"  // `device_kernels_t`, `vram_nvidia_kernels_k`, `vram_amd_kernels_k`
#include "uform/encoder/ram_kits.hpp" // `highest_kit`

namespace unum::uform::encoder {

#pragma region Products

/** NVIDIA's own encoder steps, the table its kernel unit defines. */
struct vram_nvidia_entries_kit_t {
    static constexpr nk_capability_t capability_k = nk_cap_cuda_k;
    [[nodiscard]] static device_kernels_t const &kernels() noexcept { return vram_nvidia_kernels_k; }
};

/** AMD's own encoder steps, the table its kernel unit defines. */
struct vram_amd_entries_kit_t {
    static constexpr nk_capability_t capability_k = nk_cap_rocm_k;
    [[nodiscard]] static device_kernels_t const &kernels() noexcept { return vram_amd_kernels_k; }
};

#if NUMKONG_TARGET_CUDA
struct cuda_products_kit_t {
    static constexpr nk_capability_t capability_k = nk_cap_cuda_k;
    static constexpr bool compiled_k = true;
    static constexpr auto pack_size_bf16 = nk_dots_pack_size_bf16_cuda;
    static constexpr auto pack_bf16 = nk_dots_pack_bf16_cuda;
    static constexpr auto multiply_bf16 = nk_dots_packed_bf16_cuda;
    static constexpr auto pack_size_e4m3 = nk_dots_pack_size_e4m3_cuda;
    static constexpr auto pack_e4m3 = nk_dots_pack_e4m3_cuda;
    static constexpr auto multiply_e4m3 = nk_dots_packed_e4m3_cuda;
};
#else
struct cuda_products_kit_t {
    static constexpr nk_capability_t capability_k = nk_cap_cuda_k;
    static constexpr bool compiled_k = false;
};
#endif // NUMKONG_TARGET_CUDA

#if NUMKONG_TARGET_AMPERE
struct ampere_products_kit_t {
    static constexpr nk_capability_t capability_k = nk_cap_ampere_k;
    static constexpr bool compiled_k = true;
    static constexpr auto pack_size_bf16 = nk_dots_pack_size_bf16_ampere;
    static constexpr auto pack_bf16 = nk_dots_pack_bf16_ampere;
    static constexpr auto multiply_bf16 = nk_dots_packed_bf16_ampere;
    static constexpr auto pack_size_e4m3 = nk_dots_pack_size_e4m3_ampere;
    static constexpr auto pack_e4m3 = nk_dots_pack_e4m3_ampere;
    static constexpr auto multiply_e4m3 = nk_dots_packed_e4m3_ampere;
};
#else
struct ampere_products_kit_t {
    static constexpr nk_capability_t capability_k = nk_cap_ampere_k;
    static constexpr bool compiled_k = false;
};
#endif // NUMKONG_TARGET_AMPERE

#if NUMKONG_TARGET_AMPERE && NUMKONG_TARGET_HOPPER

/** Hopper's BFloat16 products beside Ampere's FP8 ones. */
struct hopper_products_kit_t {
    static constexpr nk_capability_t capability_k = nk_cap_ampere_k | nk_cap_hopper_k;
    static constexpr bool compiled_k = true;
    static constexpr auto pack_size_bf16 = nk_dots_pack_size_bf16_hopper;
    static constexpr auto pack_bf16 = nk_dots_pack_bf16_hopper;
    static constexpr auto multiply_bf16 = nk_dots_packed_bf16_hopper;
    static constexpr auto pack_size_e4m3 = nk_dots_pack_size_e4m3_ampere;
    static constexpr auto pack_e4m3 = nk_dots_pack_e4m3_ampere;
    static constexpr auto multiply_e4m3 = nk_dots_packed_e4m3_ampere;
};
#else
struct hopper_products_kit_t {
    static constexpr nk_capability_t capability_k = nk_cap_ampere_k | nk_cap_hopper_k;
    static constexpr bool compiled_k = false;
};
#endif // NUMKONG_TARGET_AMPERE && NUMKONG_TARGET_HOPPER

#if NUMKONG_TARGET_BLACKWELL
struct blackwell_products_kit_t {
    static constexpr nk_capability_t capability_k = nk_cap_blackwell_k;
    static constexpr bool compiled_k = true;
    static constexpr auto pack_size_bf16 = nk_dots_pack_size_bf16_blackwell;
    static constexpr auto pack_bf16 = nk_dots_pack_bf16_blackwell;
    static constexpr auto multiply_bf16 = nk_dots_packed_bf16_blackwell;
    static constexpr auto pack_size_e4m3 = nk_dots_pack_size_e4m3_blackwell;
    static constexpr auto pack_e4m3 = nk_dots_pack_e4m3_blackwell;
    static constexpr auto multiply_e4m3 = nk_dots_packed_e4m3_blackwell;
};
#else
struct blackwell_products_kit_t {
    static constexpr nk_capability_t capability_k = nk_cap_blackwell_k;
    static constexpr bool compiled_k = false;
};
#endif // NUMKONG_TARGET_BLACKWELL

#if NUMKONG_TARGET_AMPERE && NUMKONG_TARGET_BLACKWELLRTX

/** Consumer Blackwell's FP8 products beside Ampere's BFloat16 ones. */
struct blackwellrtx_products_kit_t {
    static constexpr nk_capability_t capability_k = nk_cap_ampere_k | nk_cap_blackwellrtx_k;
    static constexpr bool compiled_k = true;
    static constexpr auto pack_size_bf16 = nk_dots_pack_size_bf16_ampere;
    static constexpr auto pack_bf16 = nk_dots_pack_bf16_ampere;
    static constexpr auto multiply_bf16 = nk_dots_packed_bf16_ampere;
    static constexpr auto pack_size_e4m3 = nk_dots_pack_size_e4m3_blackwellrtx;
    static constexpr auto pack_e4m3 = nk_dots_pack_e4m3_blackwellrtx;
    static constexpr auto multiply_e4m3 = nk_dots_packed_e4m3_blackwellrtx;
};
#else
struct blackwellrtx_products_kit_t {
    static constexpr nk_capability_t capability_k = nk_cap_ampere_k | nk_cap_blackwellrtx_k;
    static constexpr bool compiled_k = false;
};
#endif // NUMKONG_TARGET_AMPERE && NUMKONG_TARGET_BLACKWELLRTX

#if NUMKONG_TARGET_ROCM
struct rocm_products_kit_t {
    static constexpr nk_capability_t capability_k = nk_cap_rocm_k;
    static constexpr bool compiled_k = true;
    static constexpr auto pack_size_bf16 = nk_dots_pack_size_bf16_rocm;
    static constexpr auto pack_bf16 = nk_dots_pack_bf16_rocm;
    static constexpr auto multiply_bf16 = nk_dots_packed_bf16_rocm;
    static constexpr auto pack_size_e4m3 = nk_dots_pack_size_e4m3_rocm;
    static constexpr auto pack_e4m3 = nk_dots_pack_e4m3_rocm;
    static constexpr auto multiply_e4m3 = nk_dots_packed_e4m3_rocm;
};
#else
struct rocm_products_kit_t {
    static constexpr nk_capability_t capability_k = nk_cap_rocm_k;
    static constexpr bool compiled_k = false;
};
#endif // NUMKONG_TARGET_ROCM

#if NUMKONG_TARGET_CDNA3
struct cdna3_products_kit_t {
    static constexpr nk_capability_t capability_k = nk_cap_cdna3_k;
    static constexpr bool compiled_k = true;
    static constexpr auto pack_size_bf16 = nk_dots_pack_size_bf16_cdna3;
    static constexpr auto pack_bf16 = nk_dots_pack_bf16_cdna3;
    static constexpr auto multiply_bf16 = nk_dots_packed_bf16_cdna3;
    static constexpr auto pack_size_e4m3 = nk_dots_pack_size_e4m3_cdna3;
    static constexpr auto pack_e4m3 = nk_dots_pack_e4m3_cdna3;
    static constexpr auto multiply_e4m3 = nk_dots_packed_e4m3_cdna3;
};
#else
struct cdna3_products_kit_t {
    static constexpr nk_capability_t capability_k = nk_cap_cdna3_k;
    static constexpr bool compiled_k = false;
};
#endif // NUMKONG_TARGET_CDNA3

#if NUMKONG_TARGET_CDNA4
struct cdna4_products_kit_t {
    static constexpr nk_capability_t capability_k = nk_cap_cdna4_k;
    static constexpr bool compiled_k = true;
    static constexpr auto pack_size_bf16 = nk_dots_pack_size_bf16_cdna4;
    static constexpr auto pack_bf16 = nk_dots_pack_bf16_cdna4;
    static constexpr auto multiply_bf16 = nk_dots_packed_bf16_cdna4;
    static constexpr auto pack_size_e4m3 = nk_dots_pack_size_e4m3_cdna4;
    static constexpr auto pack_e4m3 = nk_dots_pack_e4m3_cdna4;
    static constexpr auto multiply_e4m3 = nk_dots_packed_e4m3_cdna4;
};
#else
struct cdna4_products_kit_t {
    static constexpr nk_capability_t capability_k = nk_cap_cdna4_k;
    static constexpr bool compiled_k = false;
};
#endif // NUMKONG_TARGET_CDNA4

#if NUMKONG_TARGET_CDNA5
struct cdna5_products_kit_t {
    static constexpr nk_capability_t capability_k = nk_cap_cdna5_k;
    static constexpr bool compiled_k = true;
    static constexpr auto pack_size_bf16 = nk_dots_pack_size_bf16_cdna5;
    static constexpr auto pack_bf16 = nk_dots_pack_bf16_cdna5;
    static constexpr auto multiply_bf16 = nk_dots_packed_bf16_cdna5;
    static constexpr auto pack_size_e4m3 = nk_dots_pack_size_e4m3_cdna5;
    static constexpr auto pack_e4m3 = nk_dots_pack_e4m3_cdna5;
    static constexpr auto multiply_e4m3 = nk_dots_packed_e4m3_cdna5;
};
#else
struct cdna5_products_kit_t {
    static constexpr nk_capability_t capability_k = nk_cap_cdna5_k;
    static constexpr bool compiled_k = false;
};
#endif // NUMKONG_TARGET_CDNA5

/** The products kit for one device mask, its vendor's generic kit the floor. */
template <nk_capability_t caps_>
struct vram_products_ladder {
    using floor_t = std::conditional_t<(caps_ & nk_cap_rocm_k) != 0, rocm_products_kit_t, cuda_products_kit_t>;
    using kit_t = typename highest_kit<caps_, floor_t, blackwell_products_kit_t, blackwellrtx_products_kit_t,
                                       hopper_products_kit_t, ampere_products_kit_t, cdna5_products_kit_t,
                                       cdna4_products_kit_t, cdna3_products_kit_t>::type;
    static constexpr nk_capability_t picked_k = kit_t::capability_k;
};

#pragma endregion Products

#pragma region Attention

#if NUMKONG_TARGET_CUDA
struct cuda_attention_kit_t {
    static constexpr nk_capability_t capability_k = nk_cap_cuda_k;
    static constexpr bool compiled_k = true;
    static constexpr auto pack_size_bf16 = nk_attention_pack_size_bf16_cuda;
    static constexpr auto pack_bf16 = nk_attention_pack_bf16_cuda;
    static constexpr auto attend_bf16 = nk_attention_packed_bf16_cuda;
    static constexpr auto pack_size_e4m3 = nk_attention_pack_size_e4m3_cuda;
    static constexpr auto pack_e4m3 = nk_attention_pack_e4m3_cuda;
    static constexpr auto attend_e4m3 = nk_attention_packed_e4m3_cuda;
};
#else
struct cuda_attention_kit_t {
    static constexpr nk_capability_t capability_k = nk_cap_cuda_k;
    static constexpr bool compiled_k = false;
};
#endif // NUMKONG_TARGET_CUDA

#if NUMKONG_TARGET_AMPERE
struct ampere_attention_kit_t {
    static constexpr nk_capability_t capability_k = nk_cap_ampere_k;
    static constexpr bool compiled_k = true;
    static constexpr auto pack_size_bf16 = nk_attention_pack_size_bf16_ampere;
    static constexpr auto pack_bf16 = nk_attention_pack_bf16_ampere;
    static constexpr auto attend_bf16 = nk_attention_packed_bf16_ampere;
    static constexpr auto pack_size_e4m3 = nk_attention_pack_size_e4m3_ampere;
    static constexpr auto pack_e4m3 = nk_attention_pack_e4m3_ampere;
    static constexpr auto attend_e4m3 = nk_attention_packed_e4m3_ampere;
};
#else
struct ampere_attention_kit_t {
    static constexpr nk_capability_t capability_k = nk_cap_ampere_k;
    static constexpr bool compiled_k = false;
};
#endif // NUMKONG_TARGET_AMPERE

#if NUMKONG_TARGET_HOPPER
struct hopper_attention_kit_t {
    static constexpr nk_capability_t capability_k = nk_cap_hopper_k;
    static constexpr bool compiled_k = true;
    static constexpr auto pack_size_bf16 = nk_attention_pack_size_bf16_hopper;
    static constexpr auto pack_bf16 = nk_attention_pack_bf16_hopper;
    static constexpr auto attend_bf16 = nk_attention_packed_bf16_hopper;
    static constexpr auto pack_size_e4m3 = nk_attention_pack_size_e4m3_hopper;
    static constexpr auto pack_e4m3 = nk_attention_pack_e4m3_hopper;
    static constexpr auto attend_e4m3 = nk_attention_packed_e4m3_hopper;
};
#else
struct hopper_attention_kit_t {
    static constexpr nk_capability_t capability_k = nk_cap_hopper_k;
    static constexpr bool compiled_k = false;
};
#endif // NUMKONG_TARGET_HOPPER

#if NUMKONG_TARGET_BLACKWELL
struct blackwell_attention_kit_t {
    static constexpr nk_capability_t capability_k = nk_cap_blackwell_k;
    static constexpr bool compiled_k = true;
    static constexpr auto pack_size_bf16 = nk_attention_pack_size_bf16_blackwell;
    static constexpr auto pack_bf16 = nk_attention_pack_bf16_blackwell;
    static constexpr auto attend_bf16 = nk_attention_packed_bf16_blackwell;
    static constexpr auto pack_size_e4m3 = nk_attention_pack_size_e4m3_blackwell;
    static constexpr auto pack_e4m3 = nk_attention_pack_e4m3_blackwell;
    static constexpr auto attend_e4m3 = nk_attention_packed_e4m3_blackwell;
};
#else
struct blackwell_attention_kit_t {
    static constexpr nk_capability_t capability_k = nk_cap_blackwell_k;
    static constexpr bool compiled_k = false;
};
#endif // NUMKONG_TARGET_BLACKWELL

#if NUMKONG_TARGET_AMPERE && NUMKONG_TARGET_BLACKWELLRTX

/** Consumer Blackwell's FP8 attention beside Ampere's BFloat16 one. */
struct blackwellrtx_attention_kit_t {
    static constexpr nk_capability_t capability_k = nk_cap_ampere_k | nk_cap_blackwellrtx_k;
    static constexpr bool compiled_k = true;
    static constexpr auto pack_size_bf16 = nk_attention_pack_size_bf16_ampere;
    static constexpr auto pack_bf16 = nk_attention_pack_bf16_ampere;
    static constexpr auto attend_bf16 = nk_attention_packed_bf16_ampere;
    static constexpr auto pack_size_e4m3 = nk_attention_pack_size_e4m3_blackwellrtx;
    static constexpr auto pack_e4m3 = nk_attention_pack_e4m3_blackwellrtx;
    static constexpr auto attend_e4m3 = nk_attention_packed_e4m3_blackwellrtx;
};
#else
struct blackwellrtx_attention_kit_t {
    static constexpr nk_capability_t capability_k = nk_cap_ampere_k | nk_cap_blackwellrtx_k;
    static constexpr bool compiled_k = false;
};
#endif // NUMKONG_TARGET_AMPERE && NUMKONG_TARGET_BLACKWELLRTX

#if NUMKONG_TARGET_ROCM
struct rocm_attention_kit_t {
    static constexpr nk_capability_t capability_k = nk_cap_rocm_k;
    static constexpr bool compiled_k = true;
    static constexpr auto pack_size_bf16 = nk_attention_pack_size_bf16_rocm;
    static constexpr auto pack_bf16 = nk_attention_pack_bf16_rocm;
    static constexpr auto attend_bf16 = nk_attention_packed_bf16_rocm;
    static constexpr auto pack_size_e4m3 = nk_attention_pack_size_e4m3_rocm;
    static constexpr auto pack_e4m3 = nk_attention_pack_e4m3_rocm;
    static constexpr auto attend_e4m3 = nk_attention_packed_e4m3_rocm;
};
#else
struct rocm_attention_kit_t {
    static constexpr nk_capability_t capability_k = nk_cap_rocm_k;
    static constexpr bool compiled_k = false;
};
#endif // NUMKONG_TARGET_ROCM

#if NUMKONG_TARGET_ROCM && NUMKONG_TARGET_CDNA3

/** CDNA3's BFloat16 attention beside the generic ROCm FP8 one. */
struct cdna3_attention_kit_t {
    static constexpr nk_capability_t capability_k = nk_cap_rocm_k | nk_cap_cdna3_k;
    static constexpr bool compiled_k = true;
    static constexpr auto pack_size_bf16 = nk_attention_pack_size_bf16_cdna3;
    static constexpr auto pack_bf16 = nk_attention_pack_bf16_cdna3;
    static constexpr auto attend_bf16 = nk_attention_packed_bf16_cdna3;
    static constexpr auto pack_size_e4m3 = nk_attention_pack_size_e4m3_rocm;
    static constexpr auto pack_e4m3 = nk_attention_pack_e4m3_rocm;
    static constexpr auto attend_e4m3 = nk_attention_packed_e4m3_rocm;
};
#else
struct cdna3_attention_kit_t {
    static constexpr nk_capability_t capability_k = nk_cap_rocm_k | nk_cap_cdna3_k;
    static constexpr bool compiled_k = false;
};
#endif // NUMKONG_TARGET_ROCM && NUMKONG_TARGET_CDNA3

#if NUMKONG_TARGET_CDNA4
struct cdna4_attention_kit_t {
    static constexpr nk_capability_t capability_k = nk_cap_cdna4_k;
    static constexpr bool compiled_k = true;
    static constexpr auto pack_size_bf16 = nk_attention_pack_size_bf16_cdna4;
    static constexpr auto pack_bf16 = nk_attention_pack_bf16_cdna4;
    static constexpr auto attend_bf16 = nk_attention_packed_bf16_cdna4;
    static constexpr auto pack_size_e4m3 = nk_attention_pack_size_e4m3_cdna4;
    static constexpr auto pack_e4m3 = nk_attention_pack_e4m3_cdna4;
    static constexpr auto attend_e4m3 = nk_attention_packed_e4m3_cdna4;
};
#else
struct cdna4_attention_kit_t {
    static constexpr nk_capability_t capability_k = nk_cap_cdna4_k;
    static constexpr bool compiled_k = false;
};
#endif // NUMKONG_TARGET_CDNA4

#if NUMKONG_TARGET_CDNA5
struct cdna5_attention_kit_t {
    static constexpr nk_capability_t capability_k = nk_cap_cdna5_k;
    static constexpr bool compiled_k = true;
    static constexpr auto pack_size_bf16 = nk_attention_pack_size_bf16_cdna5;
    static constexpr auto pack_bf16 = nk_attention_pack_bf16_cdna5;
    static constexpr auto attend_bf16 = nk_attention_packed_bf16_cdna5;
    static constexpr auto pack_size_e4m3 = nk_attention_pack_size_e4m3_cdna5;
    static constexpr auto pack_e4m3 = nk_attention_pack_e4m3_cdna5;
    static constexpr auto attend_e4m3 = nk_attention_packed_e4m3_cdna5;
};
#else
struct cdna5_attention_kit_t {
    static constexpr nk_capability_t capability_k = nk_cap_cdna5_k;
    static constexpr bool compiled_k = false;
};
#endif // NUMKONG_TARGET_CDNA5

/** The attention kit for one device mask, its vendor's generic kit the floor. */
template <nk_capability_t caps_>
struct vram_attention_ladder {
    using floor_t = std::conditional_t<(caps_ & nk_cap_rocm_k) != 0, rocm_attention_kit_t, cuda_attention_kit_t>;
    using kit_t = typename highest_kit<caps_, floor_t, blackwell_attention_kit_t, blackwellrtx_attention_kit_t,
                                       hopper_attention_kit_t, ampere_attention_kit_t, cdna5_attention_kit_t,
                                       cdna4_attention_kit_t, cdna3_attention_kit_t>::type;
    static constexpr nk_capability_t picked_k = kit_t::capability_k;
};

#pragma endregion Attention

#pragma region Elementwise

#if NUMKONG_TARGET_CUDA
struct cuda_elementwise_kit_t {
    static constexpr nk_capability_t capability_k = nk_cap_cuda_k;
    static constexpr bool compiled_k = true;
    static constexpr auto cast = nk_cast_cuda;
    static constexpr auto sum_f32 = nk_each_sum_f32_cuda;
    static constexpr auto rmscast_bf16 = nk_each_rmscast_bf16_cuda;
    static constexpr auto rmsnorm_bf16 = nk_each_rmsnorm_bf16_cuda;
    static constexpr auto swiglu_f32 = nk_each_swiglu_f32_cuda;
    static constexpr auto swiglu_bf16 = nk_each_swiglu_bf16_cuda;
};
#else
struct cuda_elementwise_kit_t {
    static constexpr nk_capability_t capability_k = nk_cap_cuda_k;
    static constexpr bool compiled_k = false;
};
#endif // NUMKONG_TARGET_CUDA

#if NUMKONG_TARGET_CUDA && NUMKONG_TARGET_AMPERE

/** Ampere's casts, norms and BFloat16 gates over the generic CUDA sums and Float32 gates. */
struct ampere_elementwise_kit_t {
    static constexpr nk_capability_t capability_k = nk_cap_cuda_k | nk_cap_ampere_k;
    static constexpr bool compiled_k = true;
    static constexpr auto cast = nk_cast_ampere;
    static constexpr auto sum_f32 = nk_each_sum_f32_cuda;
    static constexpr auto rmscast_bf16 = nk_each_rmscast_bf16_ampere;
    static constexpr auto rmsnorm_bf16 = nk_each_rmsnorm_bf16_ampere;
    static constexpr auto swiglu_f32 = nk_each_swiglu_f32_cuda;
    static constexpr auto swiglu_bf16 = nk_each_swiglu_bf16_ampere;
};
#else
struct ampere_elementwise_kit_t {
    static constexpr nk_capability_t capability_k = nk_cap_cuda_k | nk_cap_ampere_k;
    static constexpr bool compiled_k = false;
};
#endif // NUMKONG_TARGET_CUDA && NUMKONG_TARGET_AMPERE

#if NUMKONG_TARGET_CUDA && NUMKONG_TARGET_AMPERE && NUMKONG_TARGET_ADA

/** Ada's FP8 casts over Ampere's norms and gates. */
struct ada_elementwise_kit_t {
    static constexpr nk_capability_t capability_k = nk_cap_cuda_k | nk_cap_ampere_k | nk_cap_ada_k;
    static constexpr bool compiled_k = true;
    static constexpr auto cast = nk_cast_ada;
    static constexpr auto sum_f32 = nk_each_sum_f32_cuda;
    static constexpr auto rmscast_bf16 = nk_each_rmscast_bf16_ampere;
    static constexpr auto rmsnorm_bf16 = nk_each_rmsnorm_bf16_ampere;
    static constexpr auto swiglu_f32 = nk_each_swiglu_f32_cuda;
    static constexpr auto swiglu_bf16 = nk_each_swiglu_bf16_ampere;
};
#else
struct ada_elementwise_kit_t {
    static constexpr nk_capability_t capability_k = nk_cap_cuda_k | nk_cap_ampere_k | nk_cap_ada_k;
    static constexpr bool compiled_k = false;
};
#endif // NUMKONG_TARGET_CUDA && NUMKONG_TARGET_AMPERE && NUMKONG_TARGET_ADA

#if NUMKONG_TARGET_ROCM
struct rocm_elementwise_kit_t {
    static constexpr nk_capability_t capability_k = nk_cap_rocm_k;
    static constexpr bool compiled_k = true;
    static constexpr auto cast = nk_cast_rocm;
    static constexpr auto sum_f32 = nk_each_sum_f32_rocm;
    static constexpr auto rmscast_bf16 = nk_each_rmscast_bf16_rocm;
    static constexpr auto rmsnorm_bf16 = nk_each_rmsnorm_bf16_rocm;
    static constexpr auto swiglu_f32 = nk_each_swiglu_f32_rocm;
    static constexpr auto swiglu_bf16 = nk_each_swiglu_bf16_rocm;
};
#else
struct rocm_elementwise_kit_t {
    static constexpr nk_capability_t capability_k = nk_cap_rocm_k;
    static constexpr bool compiled_k = false;
};
#endif // NUMKONG_TARGET_ROCM

/** The element-wise kit for one device mask, its vendor's generic kit the floor. */
template <nk_capability_t caps_>
struct vram_elementwise_ladder {
    using floor_t = std::conditional_t<(caps_ & nk_cap_rocm_k) != 0, rocm_elementwise_kit_t, cuda_elementwise_kit_t>;
    using kit_t = typename highest_kit<caps_, floor_t, ada_elementwise_kit_t, ampere_elementwise_kit_t>::type;
    static constexpr nk_capability_t picked_k = kit_t::capability_k;
};

#pragma endregion Elementwise

} // namespace unum::uform::encoder
