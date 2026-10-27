/**
 *  @file include/uform/encoder/ram_kits.hpp
 *  @author Ash Vardanian
 *  @date October 25, 2026
 *  @brief The CPU kits the encoder tower runs on: NumKong's packed products, its packed
 *      attention and its element-wise steps, one kit per capability that changes any of them,
 *      and a ladder per family picking the highest kit a capability mask runs.
 *
 *  A kit names NumKong's per-capability kernels directly, never a @c _best dispatcher, so the
 *  class unit that instantiates a tower compiles its calls once, for its class. Where a
 *  capability carries only some of a family's kernels, its kit borrows the rest from its
 *  lineage, and @c capability_k is the union of every bit the kit calls, so a ladder admits it
 *  only on a mask holding them all. A kit this build's NumKong did not compile keeps its name
 *  with @c compiled_k false, so every ladder stays one list.
 *
 *  Every kit carries BFloat16 products, and E4M3 ones where its capability has them: precision is a
 *  runtime value of the tower, never a type, so a class whose products kit has no E4M3 kernel
 *  refuses an E4M3 load, as @ref e4m3_products tells. Where a whole ISA family lacks a kernel, its
 *  kits compose the operation from the family's own kernels, as @ref mrope rotates without
 *  @c rope_bf16 , or, with nothing to compose from, borrow the serial kit's row, the only place a
 *  @c _serial kernel is named.
 *
 *  The host entries close the file as @c inline @c static function templates over the element-wise
 *  kit: each works one row or one segment in Float32 through the kit's casts, reductions and
 *  rotations, and every class unit compiles its own copy under its own flags, so the linker never
 *  hands one class the copy another compiled.
 */
#pragma once

#include <cmath>   // `std::fmax` — the E4M3 scale floor; `std::sqrt`, `std::exp` — the composed norms and gates
#include <cstring> // `std::memcpy` — byte rows and angle rows

#include <type_traits> // `std::conditional_t`, `std::is_same_v`, `std::is_const_v`

#include <numkong/attention.h>    // `nk_attention_pack_*`, `nk_attention_packed_*`, `nk_attention_rope_bf16_*`
#include <numkong/cast.h>         // `nk_cast_*`
#include <numkong/dot.h>          // `nk_dot_f32_*` — the composed norms' sums of squares
#include <numkong/dots.h>         // `nk_dots_pack_*`, `nk_dots_packed_*`
#include <numkong/each.h>         // `nk_each_sum_f32_*`, `nk_each_scale_f32_*`, `nk_each_rmsnorm_bf16_*`, ...
#include <numkong/reduce.h>       // `nk_reduce_minmax_f32_*`
#include <numkong/trigonometry.h> // `nk_trig_sin_f32_*`, `nk_trig_cos_f32_*`

#include "uform/encoder/entries.hpp"      // `segment_layout_t`, `rope_grids_t`, `e4m3_range_k`
#include "uform/utilities/ram_device.hpp" // `pool_view_t`, the threads the integer entries split across

namespace unum::uform::encoder {

/** The first of @p kits_ whose every capability bit @p caps_ holds and whose kernels this build
 *  compiled, else @p floor_ . Reading order is the priority order. */
template <nk_capability_t caps_, typename floor_, typename... kits_>
struct highest_kit {
    using type = floor_;
};

template <nk_capability_t caps_, typename floor_, typename head_, typename... tail_>
struct highest_kit<caps_, floor_, head_, tail_...> {
    using type = std::conditional_t<head_::compiled_k && (caps_ & head_::capability_k) == head_::capability_k, head_,
                                    typename highest_kit<caps_, floor_, tail_...>::type>;
};

/** Whether a products kit carries E4M3 products; a class whose kit has none refuses E4M3 loads
 *  rather than fall to serial code. */
template <typename products_kit_type_>
concept e4m3_products = requires { products_kit_type_::multiply_e4m3; };

/** Whether an element-wise kit carries NumKong's rotation; @ref mrope composes one without it. */
template <typename elementwise_kit_type_>
concept rope_rotations = requires { elementwise_kit_type_::rope_bf16; };

#pragma region Products

/** The packed products every CPU runs. */
struct serial_products_kit_t {
    static constexpr nk_capability_t capability_k = nk_cap_serial_k;
    static constexpr bool compiled_k = true;
    static constexpr auto pack_size_bf16 = nk_dots_pack_size_bf16_serial;
    static constexpr auto pack_bf16 = nk_dots_pack_bf16_serial;
    static constexpr auto multiply_bf16 = nk_dots_packed_bf16_serial;
    static constexpr auto pack_size_e4m3 = nk_dots_pack_size_e4m3_serial;
    static constexpr auto pack_e4m3 = nk_dots_pack_e4m3_serial;
    static constexpr auto multiply_e4m3 = nk_dots_packed_e4m3_serial;
};

#if NUMKONG_TARGET_HASWELL
struct haswell_products_kit_t {
    static constexpr nk_capability_t capability_k = nk_cap_haswell_k;
    static constexpr bool compiled_k = true;
    static constexpr auto pack_size_bf16 = nk_dots_pack_size_bf16_haswell;
    static constexpr auto pack_bf16 = nk_dots_pack_bf16_haswell;
    static constexpr auto multiply_bf16 = nk_dots_packed_bf16_haswell;
    static constexpr auto pack_size_e4m3 = nk_dots_pack_size_e4m3_haswell;
    static constexpr auto pack_e4m3 = nk_dots_pack_e4m3_haswell;
    static constexpr auto multiply_e4m3 = nk_dots_packed_e4m3_haswell;
};
#else
struct haswell_products_kit_t {
    static constexpr nk_capability_t capability_k = nk_cap_haswell_k;
    static constexpr bool compiled_k = false;
};
#endif // NUMKONG_TARGET_HASWELL

#if NUMKONG_TARGET_SKYLAKE
struct skylake_products_kit_t {
    static constexpr nk_capability_t capability_k = nk_cap_skylake_k;
    static constexpr bool compiled_k = true;
    static constexpr auto pack_size_bf16 = nk_dots_pack_size_bf16_skylake;
    static constexpr auto pack_bf16 = nk_dots_pack_bf16_skylake;
    static constexpr auto multiply_bf16 = nk_dots_packed_bf16_skylake;
    static constexpr auto pack_size_e4m3 = nk_dots_pack_size_e4m3_skylake;
    static constexpr auto pack_e4m3 = nk_dots_pack_e4m3_skylake;
    static constexpr auto multiply_e4m3 = nk_dots_packed_e4m3_skylake;
};
#else
struct skylake_products_kit_t {
    static constexpr nk_capability_t capability_k = nk_cap_skylake_k;
    static constexpr bool compiled_k = false;
};
#endif // NUMKONG_TARGET_SKYLAKE

#if NUMKONG_TARGET_GENOA
struct genoa_products_kit_t {
    static constexpr nk_capability_t capability_k = nk_cap_genoa_k;
    static constexpr bool compiled_k = true;
    static constexpr auto pack_size_bf16 = nk_dots_pack_size_bf16_genoa;
    static constexpr auto pack_bf16 = nk_dots_pack_bf16_genoa;
    static constexpr auto multiply_bf16 = nk_dots_packed_bf16_genoa;
    static constexpr auto pack_size_e4m3 = nk_dots_pack_size_e4m3_genoa;
    static constexpr auto pack_e4m3 = nk_dots_pack_e4m3_genoa;
    static constexpr auto multiply_e4m3 = nk_dots_packed_e4m3_genoa;
};
#else
struct genoa_products_kit_t {
    static constexpr nk_capability_t capability_k = nk_cap_genoa_k;
    static constexpr bool compiled_k = false;
};
#endif // NUMKONG_TARGET_GENOA

#if NUMKONG_TARGET_SAPPHIREAMX
struct sapphireamx_products_kit_t {
    static constexpr nk_capability_t capability_k = nk_cap_sapphireamx_k;
    static constexpr bool compiled_k = true;
    static constexpr auto pack_size_bf16 = nk_dots_pack_size_bf16_sapphireamx;
    static constexpr auto pack_bf16 = nk_dots_pack_bf16_sapphireamx;
    static constexpr auto multiply_bf16 = nk_dots_packed_bf16_sapphireamx;
    static constexpr auto pack_size_e4m3 = nk_dots_pack_size_e4m3_sapphireamx;
    static constexpr auto pack_e4m3 = nk_dots_pack_e4m3_sapphireamx;
    static constexpr auto multiply_e4m3 = nk_dots_packed_e4m3_sapphireamx;
};
#else
struct sapphireamx_products_kit_t {
    static constexpr nk_capability_t capability_k = nk_cap_sapphireamx_k;
    static constexpr bool compiled_k = false;
};
#endif // NUMKONG_TARGET_SAPPHIREAMX

#if NUMKONG_TARGET_SAPPHIREAMX && NUMKONG_TARGET_DIAMOND

/** Diamond Rapids' FP8 products beside Sapphire Rapids' BFloat16 tiles. */
struct diamond_products_kit_t {
    static constexpr nk_capability_t capability_k = nk_cap_sapphireamx_k | nk_cap_diamond_k;
    static constexpr bool compiled_k = true;
    static constexpr auto pack_size_bf16 = nk_dots_pack_size_bf16_sapphireamx;
    static constexpr auto pack_bf16 = nk_dots_pack_bf16_sapphireamx;
    static constexpr auto multiply_bf16 = nk_dots_packed_bf16_sapphireamx;
    static constexpr auto pack_size_e4m3 = nk_dots_pack_size_e4m3_diamond;
    static constexpr auto pack_e4m3 = nk_dots_pack_e4m3_diamond;
    static constexpr auto multiply_e4m3 = nk_dots_packed_e4m3_diamond;
};
#else
struct diamond_products_kit_t {
    static constexpr nk_capability_t capability_k = nk_cap_sapphireamx_k | nk_cap_diamond_k;
    static constexpr bool compiled_k = false;
};
#endif // NUMKONG_TARGET_SAPPHIREAMX && NUMKONG_TARGET_DIAMOND

#if NUMKONG_TARGET_NEON

/** The NEON floor: BFloat16 products only, as NumKong's FP8 products start at FHM. */
struct neon_products_kit_t {
    static constexpr nk_capability_t capability_k = nk_cap_neon_k;
    static constexpr bool compiled_k = true;
    static constexpr auto pack_size_bf16 = nk_dots_pack_size_bf16_neon;
    static constexpr auto pack_bf16 = nk_dots_pack_bf16_neon;
    static constexpr auto multiply_bf16 = nk_dots_packed_bf16_neon;
};
#else
struct neon_products_kit_t {
    static constexpr nk_capability_t capability_k = nk_cap_neon_k;
    static constexpr bool compiled_k = false;
};
#endif // NUMKONG_TARGET_NEON

#if NUMKONG_TARGET_NEONBFDOT && NUMKONG_TARGET_NEONFHM

/** Armv8.6's BFloat16 dot products beside the FP16 fused multiplies its FP8 products widen
 *  through. */
struct neonbfdot_products_kit_t {
    static constexpr nk_capability_t capability_k = nk_cap_neonbfdot_k | nk_cap_neonfhm_k;
    static constexpr bool compiled_k = true;
    static constexpr auto pack_size_bf16 = nk_dots_pack_size_bf16_neonbfdot;
    static constexpr auto pack_bf16 = nk_dots_pack_bf16_neonbfdot;
    static constexpr auto multiply_bf16 = nk_dots_packed_bf16_neonbfdot;
    static constexpr auto pack_size_e4m3 = nk_dots_pack_size_e4m3_neonfhm;
    static constexpr auto pack_e4m3 = nk_dots_pack_e4m3_neonfhm;
    static constexpr auto multiply_e4m3 = nk_dots_packed_e4m3_neonfhm;
};
#else
struct neonbfdot_products_kit_t {
    static constexpr nk_capability_t capability_k = nk_cap_neonbfdot_k | nk_cap_neonfhm_k;
    static constexpr bool compiled_k = false;
};
#endif // NUMKONG_TARGET_NEONBFDOT && NUMKONG_TARGET_NEONFHM

#if NUMKONG_TARGET_NEONBFDOT && NUMKONG_TARGET_NEONFP8

/** Vera's native FP8 dot products beside the Armv8.6 BFloat16 ones. */
struct neonfp8_products_kit_t {
    static constexpr nk_capability_t capability_k = nk_cap_neonbfdot_k | nk_cap_neonfp8_k;
    static constexpr bool compiled_k = true;
    static constexpr auto pack_size_bf16 = nk_dots_pack_size_bf16_neonbfdot;
    static constexpr auto pack_bf16 = nk_dots_pack_bf16_neonbfdot;
    static constexpr auto multiply_bf16 = nk_dots_packed_bf16_neonbfdot;
    static constexpr auto pack_size_e4m3 = nk_dots_pack_size_e4m3_neonfp8;
    static constexpr auto pack_e4m3 = nk_dots_pack_e4m3_neonfp8;
    static constexpr auto multiply_e4m3 = nk_dots_packed_e4m3_neonfp8;
};
#else
struct neonfp8_products_kit_t {
    static constexpr nk_capability_t capability_k = nk_cap_neonbfdot_k | nk_cap_neonfp8_k;
    static constexpr bool compiled_k = false;
};
#endif // NUMKONG_TARGET_NEONBFDOT && NUMKONG_TARGET_NEONFP8

#if NUMKONG_TARGET_SME
struct sme_products_kit_t {
    static constexpr nk_capability_t capability_k = nk_cap_sme_k;
    static constexpr bool compiled_k = true;
    static constexpr auto pack_size_bf16 = nk_dots_pack_size_bf16_sme;
    static constexpr auto pack_bf16 = nk_dots_pack_bf16_sme;
    static constexpr auto multiply_bf16 = nk_dots_packed_bf16_sme;
    static constexpr auto pack_size_e4m3 = nk_dots_pack_size_e4m3_sme;
    static constexpr auto pack_e4m3 = nk_dots_pack_e4m3_sme;
    static constexpr auto multiply_e4m3 = nk_dots_packed_e4m3_sme;
};
#else
struct sme_products_kit_t {
    static constexpr nk_capability_t capability_k = nk_cap_sme_k;
    static constexpr bool compiled_k = false;
};
#endif // NUMKONG_TARGET_SME

#if NUMKONG_TARGET_RVV
struct rvv_products_kit_t {
    static constexpr nk_capability_t capability_k = nk_cap_rvv_k;
    static constexpr bool compiled_k = true;
    static constexpr auto pack_size_bf16 = nk_dots_pack_size_bf16_rvv;
    static constexpr auto pack_bf16 = nk_dots_pack_bf16_rvv;
    static constexpr auto multiply_bf16 = nk_dots_packed_bf16_rvv;
    static constexpr auto pack_size_e4m3 = nk_dots_pack_size_e4m3_rvv;
    static constexpr auto pack_e4m3 = nk_dots_pack_e4m3_rvv;
    static constexpr auto multiply_e4m3 = nk_dots_packed_e4m3_rvv;
};
#else
struct rvv_products_kit_t {
    static constexpr nk_capability_t capability_k = nk_cap_rvv_k;
    static constexpr bool compiled_k = false;
};
#endif // NUMKONG_TARGET_RVV

#if NUMKONG_TARGET_V128

/** WebAssembly's fixed-width SIMD: BFloat16 products only, as NumKong's FP8 products need relaxed
 *  SIMD. */
struct v128_products_kit_t {
    static constexpr nk_capability_t capability_k = nk_cap_v128_k;
    static constexpr bool compiled_k = true;
    static constexpr auto pack_size_bf16 = nk_dots_pack_size_bf16_v128;
    static constexpr auto pack_bf16 = nk_dots_pack_bf16_v128;
    static constexpr auto multiply_bf16 = nk_dots_packed_bf16_v128;
};
#else
struct v128_products_kit_t {
    static constexpr nk_capability_t capability_k = nk_cap_v128_k;
    static constexpr bool compiled_k = false;
};
#endif // NUMKONG_TARGET_V128

#if NUMKONG_TARGET_V128RELAXED
struct v128relaxed_products_kit_t {
    static constexpr nk_capability_t capability_k = nk_cap_v128relaxed_k;
    static constexpr bool compiled_k = true;
    static constexpr auto pack_size_bf16 = nk_dots_pack_size_bf16_v128relaxed;
    static constexpr auto pack_bf16 = nk_dots_pack_bf16_v128relaxed;
    static constexpr auto multiply_bf16 = nk_dots_packed_bf16_v128relaxed;
    static constexpr auto pack_size_e4m3 = nk_dots_pack_size_e4m3_v128relaxed;
    static constexpr auto pack_e4m3 = nk_dots_pack_e4m3_v128relaxed;
    static constexpr auto multiply_e4m3 = nk_dots_packed_e4m3_v128relaxed;
};
#else
struct v128relaxed_products_kit_t {
    static constexpr nk_capability_t capability_k = nk_cap_v128relaxed_k;
    static constexpr bool compiled_k = false;
};
#endif // NUMKONG_TARGET_V128RELAXED

/** The products kit for one capability ceiling. */
template <nk_capability_t caps_>
struct products_ladder {
    using kit_t =
        typename highest_kit<caps_, serial_products_kit_t, diamond_products_kit_t, sapphireamx_products_kit_t,
                             genoa_products_kit_t, skylake_products_kit_t, haswell_products_kit_t, sme_products_kit_t,
                             neonfp8_products_kit_t, neonbfdot_products_kit_t, neon_products_kit_t, rvv_products_kit_t,
                             v128relaxed_products_kit_t, v128_products_kit_t>::type;
};

#pragma endregion Products

#pragma region Attention

/** The packed attention every CPU runs. */
struct serial_attention_kit_t {
    static constexpr nk_capability_t capability_k = nk_cap_serial_k;
    static constexpr bool compiled_k = true;
    static constexpr auto pack_size_bf16 = nk_attention_pack_size_bf16_serial;
    static constexpr auto pack_bf16 = nk_attention_pack_bf16_serial;
    static constexpr auto attend_bf16 = nk_attention_packed_bf16_serial;
    static constexpr auto pack_size_e4m3 = nk_attention_pack_size_e4m3_serial;
    static constexpr auto pack_e4m3 = nk_attention_pack_e4m3_serial;
    static constexpr auto attend_e4m3 = nk_attention_packed_e4m3_serial;
};

#if NUMKONG_TARGET_HASWELL
struct haswell_attention_kit_t {
    static constexpr nk_capability_t capability_k = nk_cap_haswell_k;
    static constexpr bool compiled_k = true;
    static constexpr auto pack_size_bf16 = nk_attention_pack_size_bf16_haswell;
    static constexpr auto pack_bf16 = nk_attention_pack_bf16_haswell;
    static constexpr auto attend_bf16 = nk_attention_packed_bf16_haswell;
    static constexpr auto pack_size_e4m3 = nk_attention_pack_size_e4m3_haswell;
    static constexpr auto pack_e4m3 = nk_attention_pack_e4m3_haswell;
    static constexpr auto attend_e4m3 = nk_attention_packed_e4m3_haswell;
};
#else
struct haswell_attention_kit_t {
    static constexpr nk_capability_t capability_k = nk_cap_haswell_k;
    static constexpr bool compiled_k = false;
};
#endif // NUMKONG_TARGET_HASWELL

#if NUMKONG_TARGET_SKYLAKE
struct skylake_attention_kit_t {
    static constexpr nk_capability_t capability_k = nk_cap_skylake_k;
    static constexpr bool compiled_k = true;
    static constexpr auto pack_size_bf16 = nk_attention_pack_size_bf16_skylake;
    static constexpr auto pack_bf16 = nk_attention_pack_bf16_skylake;
    static constexpr auto attend_bf16 = nk_attention_packed_bf16_skylake;
    static constexpr auto pack_size_e4m3 = nk_attention_pack_size_e4m3_skylake;
    static constexpr auto pack_e4m3 = nk_attention_pack_e4m3_skylake;
    static constexpr auto attend_e4m3 = nk_attention_packed_e4m3_skylake;
};
#else
struct skylake_attention_kit_t {
    static constexpr nk_capability_t capability_k = nk_cap_skylake_k;
    static constexpr bool compiled_k = false;
};
#endif // NUMKONG_TARGET_SKYLAKE

#if NUMKONG_TARGET_GENOA
struct genoa_attention_kit_t {
    static constexpr nk_capability_t capability_k = nk_cap_genoa_k;
    static constexpr bool compiled_k = true;
    static constexpr auto pack_size_bf16 = nk_attention_pack_size_bf16_genoa;
    static constexpr auto pack_bf16 = nk_attention_pack_bf16_genoa;
    static constexpr auto attend_bf16 = nk_attention_packed_bf16_genoa;
    static constexpr auto pack_size_e4m3 = nk_attention_pack_size_e4m3_genoa;
    static constexpr auto pack_e4m3 = nk_attention_pack_e4m3_genoa;
    static constexpr auto attend_e4m3 = nk_attention_packed_e4m3_genoa;
};
#else
struct genoa_attention_kit_t {
    static constexpr nk_capability_t capability_k = nk_cap_genoa_k;
    static constexpr bool compiled_k = false;
};
#endif // NUMKONG_TARGET_GENOA

#if NUMKONG_TARGET_SAPPHIREAMX
struct sapphireamx_attention_kit_t {
    static constexpr nk_capability_t capability_k = nk_cap_sapphireamx_k;
    static constexpr bool compiled_k = true;
    static constexpr auto pack_size_bf16 = nk_attention_pack_size_bf16_sapphireamx;
    static constexpr auto pack_bf16 = nk_attention_pack_bf16_sapphireamx;
    static constexpr auto attend_bf16 = nk_attention_packed_bf16_sapphireamx;
    static constexpr auto pack_size_e4m3 = nk_attention_pack_size_e4m3_sapphireamx;
    static constexpr auto pack_e4m3 = nk_attention_pack_e4m3_sapphireamx;
    static constexpr auto attend_e4m3 = nk_attention_packed_e4m3_sapphireamx;
};
#else
struct sapphireamx_attention_kit_t {
    static constexpr nk_capability_t capability_k = nk_cap_sapphireamx_k;
    static constexpr bool compiled_k = false;
};
#endif // NUMKONG_TARGET_SAPPHIREAMX

#if NUMKONG_TARGET_NEON
struct neon_attention_kit_t {
    static constexpr nk_capability_t capability_k = nk_cap_neon_k;
    static constexpr bool compiled_k = true;
    static constexpr auto pack_size_bf16 = nk_attention_pack_size_bf16_neon;
    static constexpr auto pack_bf16 = nk_attention_pack_bf16_neon;
    static constexpr auto attend_bf16 = nk_attention_packed_bf16_neon;
    static constexpr auto pack_size_e4m3 = nk_attention_pack_size_e4m3_neon;
    static constexpr auto pack_e4m3 = nk_attention_pack_e4m3_neon;
    static constexpr auto attend_e4m3 = nk_attention_packed_e4m3_neon;
};
#else
struct neon_attention_kit_t {
    static constexpr nk_capability_t capability_k = nk_cap_neon_k;
    static constexpr bool compiled_k = false;
};
#endif // NUMKONG_TARGET_NEON

#if NUMKONG_TARGET_NEONBFDOT && NUMKONG_TARGET_NEONFHM

/** Armv8.6's BFloat16 attention beside the FP16 fused multiplies its FP8 attention widens
 *  through. */
struct neonbfdot_attention_kit_t {
    static constexpr nk_capability_t capability_k = nk_cap_neonbfdot_k | nk_cap_neonfhm_k;
    static constexpr bool compiled_k = true;
    static constexpr auto pack_size_bf16 = nk_attention_pack_size_bf16_neonbfdot;
    static constexpr auto pack_bf16 = nk_attention_pack_bf16_neonbfdot;
    static constexpr auto attend_bf16 = nk_attention_packed_bf16_neonbfdot;
    static constexpr auto pack_size_e4m3 = nk_attention_pack_size_e4m3_neonfhm;
    static constexpr auto pack_e4m3 = nk_attention_pack_e4m3_neonfhm;
    static constexpr auto attend_e4m3 = nk_attention_packed_e4m3_neonfhm;
};
#else
struct neonbfdot_attention_kit_t {
    static constexpr nk_capability_t capability_k = nk_cap_neonbfdot_k | nk_cap_neonfhm_k;
    static constexpr bool compiled_k = false;
};
#endif // NUMKONG_TARGET_NEONBFDOT && NUMKONG_TARGET_NEONFHM

#if NUMKONG_TARGET_SME
struct sme_attention_kit_t {
    static constexpr nk_capability_t capability_k = nk_cap_sme_k;
    static constexpr bool compiled_k = true;
    static constexpr auto pack_size_bf16 = nk_attention_pack_size_bf16_sme;
    static constexpr auto pack_bf16 = nk_attention_pack_bf16_sme;
    static constexpr auto attend_bf16 = nk_attention_packed_bf16_sme;
    static constexpr auto pack_size_e4m3 = nk_attention_pack_size_e4m3_sme;
    static constexpr auto pack_e4m3 = nk_attention_pack_e4m3_sme;
    static constexpr auto attend_e4m3 = nk_attention_packed_e4m3_sme;
};
#else
struct sme_attention_kit_t {
    static constexpr nk_capability_t capability_k = nk_cap_sme_k;
    static constexpr bool compiled_k = false;
};
#endif // NUMKONG_TARGET_SME

#if NUMKONG_TARGET_RVV
struct rvv_attention_kit_t {
    static constexpr nk_capability_t capability_k = nk_cap_rvv_k;
    static constexpr bool compiled_k = true;
    static constexpr auto pack_size_bf16 = nk_attention_pack_size_bf16_rvv;
    static constexpr auto pack_bf16 = nk_attention_pack_bf16_rvv;
    static constexpr auto attend_bf16 = nk_attention_packed_bf16_rvv;
    static constexpr auto pack_size_e4m3 = nk_attention_pack_size_e4m3_rvv;
    static constexpr auto pack_e4m3 = nk_attention_pack_e4m3_rvv;
    static constexpr auto attend_e4m3 = nk_attention_packed_e4m3_rvv;
};
#else
struct rvv_attention_kit_t {
    static constexpr nk_capability_t capability_k = nk_cap_rvv_k;
    static constexpr bool compiled_k = false;
};
#endif // NUMKONG_TARGET_RVV

#if NUMKONG_TARGET_V128RELAXED
struct v128relaxed_attention_kit_t {
    static constexpr nk_capability_t capability_k = nk_cap_v128relaxed_k;
    static constexpr bool compiled_k = true;
    static constexpr auto pack_size_bf16 = nk_attention_pack_size_bf16_v128relaxed;
    static constexpr auto pack_bf16 = nk_attention_pack_bf16_v128relaxed;
    static constexpr auto attend_bf16 = nk_attention_packed_bf16_v128relaxed;
    static constexpr auto pack_size_e4m3 = nk_attention_pack_size_e4m3_v128relaxed;
    static constexpr auto pack_e4m3 = nk_attention_pack_e4m3_v128relaxed;
    static constexpr auto attend_e4m3 = nk_attention_packed_e4m3_v128relaxed;
};
#else
struct v128relaxed_attention_kit_t {
    static constexpr nk_capability_t capability_k = nk_cap_v128relaxed_k;
    static constexpr bool compiled_k = false;
};
#endif // NUMKONG_TARGET_V128RELAXED

/** The attention kit for one capability ceiling. */
template <nk_capability_t caps_>
struct attention_ladder {
    using kit_t = typename highest_kit<caps_, serial_attention_kit_t, sapphireamx_attention_kit_t,
                                       genoa_attention_kit_t, skylake_attention_kit_t, haswell_attention_kit_t,
                                       sme_attention_kit_t, neonbfdot_attention_kit_t, neon_attention_kit_t,
                                       rvv_attention_kit_t, v128relaxed_attention_kit_t>::type;
};

#pragma endregion Attention

#pragma region Elementwise

/** The casts, element-wise arithmetic, reductions, rotations, norms and gates every CPU runs. */
struct serial_elementwise_kit_t {
    static constexpr nk_capability_t capability_k = nk_cap_serial_k;
    static constexpr bool compiled_k = true;
    static constexpr auto cast = nk_cast_serial;
    static constexpr auto sum_f32 = nk_each_sum_f32_serial;
    static constexpr auto scale_f32 = nk_each_scale_f32_serial;
    static constexpr auto blend_f32 = nk_each_blend_f32_serial;
    static constexpr auto fma_f32 = nk_each_fma_f32_serial;
    static constexpr auto reduce_f32 = nk_reduce_minmax_f32_serial;
    static constexpr auto sine_f32 = nk_trig_sin_f32_serial;
    static constexpr auto cosine_f32 = nk_trig_cos_f32_serial;
    static constexpr auto rope_bf16 = nk_attention_rope_bf16_serial;
    static constexpr auto rmscast_bf16 = nk_each_rmscast_bf16_serial;
    static constexpr auto rmsnorm_bf16 = nk_each_rmsnorm_bf16_serial;
    static constexpr auto swiglu_f32 = nk_each_swiglu_f32_serial;
    static constexpr auto swiglu_bf16 = nk_each_swiglu_bf16_serial;
};

#if NUMKONG_TARGET_HASWELL
struct haswell_elementwise_kit_t {
    static constexpr nk_capability_t capability_k = nk_cap_haswell_k;
    static constexpr bool compiled_k = true;
    static constexpr auto cast = nk_cast_haswell;
    static constexpr auto sum_f32 = nk_each_sum_f32_haswell;
    static constexpr auto scale_f32 = nk_each_scale_f32_haswell;
    static constexpr auto blend_f32 = nk_each_blend_f32_haswell;
    static constexpr auto fma_f32 = nk_each_fma_f32_haswell;
    static constexpr auto reduce_f32 = nk_reduce_minmax_f32_haswell;
    static constexpr auto sine_f32 = nk_trig_sin_f32_haswell;
    static constexpr auto cosine_f32 = nk_trig_cos_f32_haswell;
    static constexpr auto rope_bf16 = nk_attention_rope_bf16_haswell;
    static constexpr auto rmscast_bf16 = nk_each_rmscast_bf16_haswell;
    static constexpr auto rmsnorm_bf16 = nk_each_rmsnorm_bf16_haswell;
    static constexpr auto swiglu_f32 = nk_each_swiglu_f32_haswell;
    static constexpr auto swiglu_bf16 = nk_each_swiglu_bf16_haswell;
};
#else
struct haswell_elementwise_kit_t {
    static constexpr nk_capability_t capability_k = nk_cap_haswell_k;
    static constexpr bool compiled_k = false;
};
#endif // NUMKONG_TARGET_HASWELL

#if NUMKONG_TARGET_SKYLAKE
struct skylake_elementwise_kit_t {
    static constexpr nk_capability_t capability_k = nk_cap_skylake_k;
    static constexpr bool compiled_k = true;
    static constexpr auto cast = nk_cast_skylake;
    static constexpr auto sum_f32 = nk_each_sum_f32_skylake;
    static constexpr auto scale_f32 = nk_each_scale_f32_skylake;
    static constexpr auto blend_f32 = nk_each_blend_f32_skylake;
    static constexpr auto fma_f32 = nk_each_fma_f32_skylake;
    static constexpr auto reduce_f32 = nk_reduce_minmax_f32_skylake;
    static constexpr auto sine_f32 = nk_trig_sin_f32_skylake;
    static constexpr auto cosine_f32 = nk_trig_cos_f32_skylake;
    static constexpr auto rope_bf16 = nk_attention_rope_bf16_skylake;
    static constexpr auto rmscast_bf16 = nk_each_rmscast_bf16_skylake;
    static constexpr auto rmsnorm_bf16 = nk_each_rmsnorm_bf16_skylake;
    static constexpr auto swiglu_f32 = nk_each_swiglu_f32_skylake;
    static constexpr auto swiglu_bf16 = nk_each_swiglu_bf16_skylake;
};
#else
struct skylake_elementwise_kit_t {
    static constexpr nk_capability_t capability_k = nk_cap_skylake_k;
    static constexpr bool compiled_k = false;
};
#endif // NUMKONG_TARGET_SKYLAKE

#if NUMKONG_TARGET_SKYLAKE && NUMKONG_TARGET_ICELAKE

/** Icelake's casts over Skylake's arithmetic. */
struct icelake_elementwise_kit_t {
    static constexpr nk_capability_t capability_k = nk_cap_skylake_k | nk_cap_icelake_k;
    static constexpr bool compiled_k = true;
    static constexpr auto cast = nk_cast_icelake;
    static constexpr auto sum_f32 = nk_each_sum_f32_skylake;
    static constexpr auto scale_f32 = nk_each_scale_f32_skylake;
    static constexpr auto blend_f32 = nk_each_blend_f32_skylake;
    static constexpr auto fma_f32 = nk_each_fma_f32_skylake;
    static constexpr auto reduce_f32 = nk_reduce_minmax_f32_skylake;
    static constexpr auto sine_f32 = nk_trig_sin_f32_skylake;
    static constexpr auto cosine_f32 = nk_trig_cos_f32_skylake;
    static constexpr auto rope_bf16 = nk_attention_rope_bf16_skylake;
    static constexpr auto rmscast_bf16 = nk_each_rmscast_bf16_skylake;
    static constexpr auto rmsnorm_bf16 = nk_each_rmsnorm_bf16_skylake;
    static constexpr auto swiglu_f32 = nk_each_swiglu_f32_skylake;
    static constexpr auto swiglu_bf16 = nk_each_swiglu_bf16_skylake;
};
#else
struct icelake_elementwise_kit_t {
    static constexpr nk_capability_t capability_k = nk_cap_skylake_k | nk_cap_icelake_k;
    static constexpr bool compiled_k = false;
};
#endif // NUMKONG_TARGET_SKYLAKE && NUMKONG_TARGET_ICELAKE

#if NUMKONG_TARGET_SKYLAKE && NUMKONG_TARGET_ICELAKE && NUMKONG_TARGET_GENOA

/** Genoa's BFloat16 norms over Icelake's casts and Skylake's arithmetic. */
struct genoa_elementwise_kit_t {
    static constexpr nk_capability_t capability_k = nk_cap_skylake_k | nk_cap_icelake_k | nk_cap_genoa_k;
    static constexpr bool compiled_k = true;
    static constexpr auto cast = nk_cast_icelake;
    static constexpr auto sum_f32 = nk_each_sum_f32_skylake;
    static constexpr auto scale_f32 = nk_each_scale_f32_skylake;
    static constexpr auto blend_f32 = nk_each_blend_f32_skylake;
    static constexpr auto fma_f32 = nk_each_fma_f32_skylake;
    static constexpr auto reduce_f32 = nk_reduce_minmax_f32_skylake;
    static constexpr auto sine_f32 = nk_trig_sin_f32_skylake;
    static constexpr auto cosine_f32 = nk_trig_cos_f32_skylake;
    static constexpr auto rope_bf16 = nk_attention_rope_bf16_skylake;
    static constexpr auto rmscast_bf16 = nk_each_rmscast_bf16_skylake;
    static constexpr auto rmsnorm_bf16 = nk_each_rmsnorm_bf16_genoa;
    static constexpr auto swiglu_f32 = nk_each_swiglu_f32_skylake;
    static constexpr auto swiglu_bf16 = nk_each_swiglu_bf16_skylake;
};
#else
struct genoa_elementwise_kit_t {
    static constexpr nk_capability_t capability_k = nk_cap_skylake_k | nk_cap_icelake_k | nk_cap_genoa_k;
    static constexpr bool compiled_k = false;
};
#endif // NUMKONG_TARGET_SKYLAKE && NUMKONG_TARGET_ICELAKE && NUMKONG_TARGET_GENOA

#if NUMKONG_TARGET_SKYLAKE && NUMKONG_TARGET_GENOA && NUMKONG_TARGET_SAPPHIRE

/** Sapphire Rapids' FP16 casts over Genoa's norms and Skylake's arithmetic. */
struct sapphire_elementwise_kit_t {
    static constexpr nk_capability_t capability_k = nk_cap_skylake_k | nk_cap_genoa_k | nk_cap_sapphire_k;
    static constexpr bool compiled_k = true;
    static constexpr auto cast = nk_cast_sapphire;
    static constexpr auto sum_f32 = nk_each_sum_f32_skylake;
    static constexpr auto scale_f32 = nk_each_scale_f32_skylake;
    static constexpr auto blend_f32 = nk_each_blend_f32_skylake;
    static constexpr auto fma_f32 = nk_each_fma_f32_skylake;
    static constexpr auto reduce_f32 = nk_reduce_minmax_f32_skylake;
    static constexpr auto sine_f32 = nk_trig_sin_f32_skylake;
    static constexpr auto cosine_f32 = nk_trig_cos_f32_skylake;
    static constexpr auto rope_bf16 = nk_attention_rope_bf16_skylake;
    static constexpr auto rmscast_bf16 = nk_each_rmscast_bf16_skylake;
    static constexpr auto rmsnorm_bf16 = nk_each_rmsnorm_bf16_genoa;
    static constexpr auto swiglu_f32 = nk_each_swiglu_f32_skylake;
    static constexpr auto swiglu_bf16 = nk_each_swiglu_bf16_skylake;
};
#else
struct sapphire_elementwise_kit_t {
    static constexpr nk_capability_t capability_k = nk_cap_skylake_k | nk_cap_genoa_k | nk_cap_sapphire_k;
    static constexpr bool compiled_k = false;
};
#endif // NUMKONG_TARGET_SKYLAKE && NUMKONG_TARGET_GENOA && NUMKONG_TARGET_SAPPHIRE

#if NUMKONG_TARGET_NEON
struct neon_elementwise_kit_t {
    static constexpr nk_capability_t capability_k = nk_cap_neon_k;
    static constexpr bool compiled_k = true;
    static constexpr auto cast = nk_cast_neon;
    static constexpr auto sum_f32 = nk_each_sum_f32_neon;
    static constexpr auto scale_f32 = nk_each_scale_f32_neon;
    static constexpr auto blend_f32 = nk_each_blend_f32_neon;
    static constexpr auto fma_f32 = nk_each_fma_f32_neon;
    static constexpr auto reduce_f32 = nk_reduce_minmax_f32_neon;
    static constexpr auto sine_f32 = nk_trig_sin_f32_neon;
    static constexpr auto cosine_f32 = nk_trig_cos_f32_neon;
    static constexpr auto rope_bf16 = nk_attention_rope_bf16_neon;
    static constexpr auto rmscast_bf16 = nk_each_rmscast_bf16_neon;
    static constexpr auto rmsnorm_bf16 = nk_each_rmsnorm_bf16_neon;
    static constexpr auto swiglu_f32 = nk_each_swiglu_f32_neon;
    static constexpr auto swiglu_bf16 = nk_each_swiglu_bf16_neon;
};
#else
struct neon_elementwise_kit_t {
    static constexpr nk_capability_t capability_k = nk_cap_neon_k;
    static constexpr bool compiled_k = false;
};
#endif // NUMKONG_TARGET_NEON

/** Row @p row of a strided matrix starting at @p base , rows @p stride bytes apart. */
template <typename value_type_>
inline static value_type_ *row_at_(value_type_ *base, nk_size_t row, nk_size_t stride) noexcept { // vendor-boundary
    using void_t = std::conditional_t<std::is_const_v<value_type_>, void const, void>;
    using byte_t = std::conditional_t<std::is_const_v<value_type_>, uu::b8_t const, uu::b8_t>;
    return static_cast<value_type_ *>(
        static_cast<void_t *>(static_cast<byte_t *>(static_cast<void_t *>(base)) + row * stride));
}

/** RMSNorm over BFloat16 rows and its Float32-input twin, for classes NumKong ships no fused kernel
 *  for: each group widens a stack chunk at a time through @p cast_ , sums its squares through
 *  @p dot_f32_ , and takes the inverse RMS and the gain through @p scale_f32_ or @p fma_f32_ before
 *  narrowing back, with NumKong's signature and a null gain meaning one. */
template <auto cast_, auto dot_f32_, auto scale_f32_, auto fma_f32_>
struct composed_norms {

    /** Scalars one stack chunk widens at a time. */
    static constexpr nk_size_t chunk_k = 256;

    static nk_status_t rmsnorm_bf16(nk_bf16_t const *x, nk_f32_t const *gamma, nk_bf16_t *y,
                                    nk_size_t rows, // vendor-boundary
                                    nk_size_t groups, nk_size_t columns, nk_size_t x_stride, nk_size_t y_stride,
                                    nk_f32_t epsilon, nk_stream_t stream) noexcept {
        return normalize_(x, nk_bf16_k, gamma, y, rows, groups, columns, x_stride, y_stride, epsilon, stream);
    }

    static nk_status_t rmscast_bf16(nk_f32_t const *x, nk_f32_t const *gamma, nk_bf16_t *y,
                                    nk_size_t rows, // vendor-boundary
                                    nk_size_t groups, nk_size_t columns, nk_size_t x_stride, nk_size_t y_stride,
                                    nk_f32_t epsilon, nk_stream_t stream) noexcept {
        return normalize_(x, nk_f32_k, gamma, y, rows, groups, columns, x_stride, y_stride, epsilon, stream);
    }

  private:
    template <typename input_type_>
    static nk_status_t normalize_(input_type_ const *x, nk_dtype_t dtype, nk_f32_t const *gamma, nk_bf16_t *y,
                                  nk_size_t rows, nk_size_t groups, nk_size_t columns, nk_size_t x_stride,
                                  nk_size_t y_stride, nk_f32_t epsilon, nk_stream_t stream) noexcept {
        nk_f32_t chunk[chunk_k];
        nk_f32_t const zero = 0.0f;
        for (nk_size_t row = 0; row != rows; ++row) {
            input_type_ const *const x_row = row_at_(x, row, x_stride);
            nk_bf16_t *const y_row = row_at_(y, row, y_stride);
            for (nk_size_t group = 0; group != groups; ++group) {
                input_type_ const *const source = x_row + group * columns;
                nk_bf16_t *const target = y_row + group * columns;
                nk_f64_t sum_of_squares = 0;
                for (nk_size_t first = 0; first < columns; first += chunk_k) {
                    nk_size_t const count = columns - first < chunk_k ? columns - first : chunk_k;
                    nk_f64_t squares = 0;
                    if (nk_status_t const status = cast_(source + first, dtype, chunk, nk_f32_k, count, stream);
                        status != nk_success_k)
                        return status;
                    if (nk_status_t const status = dot_f32_(chunk, chunk, count, &squares, stream);
                        status != nk_success_k)
                        return status;
                    sum_of_squares += squares;
                }
                nk_f32_t const inverse_rms = 1.0f /
                                             std::sqrt(static_cast<nk_f32_t>(sum_of_squares / columns) + epsilon);
                for (nk_size_t first = 0; first < columns; first += chunk_k) {
                    nk_size_t const count = columns - first < chunk_k ? columns - first : chunk_k;
                    nk_status_t status = cast_(source + first, dtype, chunk, nk_f32_k, count, stream);
                    if (status == nk_success_k)
                        status = gamma
                                     ? fma_f32_(chunk, gamma + first, chunk, count, &inverse_rms, &zero, chunk, stream)
                                     : scale_f32_(chunk, count, &inverse_rms, &zero, chunk, stream);
                    if (status == nk_success_k)
                        status = cast_(chunk, nk_f32_k, target + first, nk_bf16_k, count, stream);
                    if (status != nk_success_k) return status;
                }
            }
        }
        return nk_success_k;
    }
};

/** SwiGLU, @c silu(gate·gate_scale)·up·output_scale , as a Float32 loop the class unit's flags
 *  vectorize, for classes NumKong has no exponential for; BFloat16 rows widen a stack chunk at a
 *  time through @p cast_ , and a null @c up leaves the plain SiLU. */
template <auto cast_>
struct composed_swiglu {

    /** Scalars one stack chunk widens at a time. */
    static constexpr nk_size_t chunk_k = 256;

    static nk_status_t swiglu_f32(nk_f32_t const *gate, nk_f32_t const *up, nk_f32_t *y,
                                  nk_size_t rows, // vendor-boundary
                                  nk_size_t columns, nk_size_t gate_stride, nk_size_t up_stride, nk_size_t y_stride,
                                  nk_f32_t gate_scale, nk_f32_t output_scale, nk_stream_t) noexcept {
        for (nk_size_t row = 0; row != rows; ++row)
            gate_row_(
                {row_at_(gate, row, gate_stride), columns},
                up ? std::span<nk_f32_t const> {row_at_(up, row, up_stride), columns} : std::span<nk_f32_t const> {},
                {row_at_(y, row, y_stride), columns}, gate_scale, output_scale);
        return nk_success_k;
    }

    static nk_status_t swiglu_bf16(nk_bf16_t const *gate, nk_bf16_t const *up, nk_bf16_t *y,
                                   nk_size_t rows, // vendor-boundary
                                   nk_size_t columns, nk_size_t gate_stride, nk_size_t up_stride, nk_size_t y_stride,
                                   nk_f32_t gate_scale, nk_f32_t output_scale, nk_stream_t stream) noexcept {
        nk_f32_t gates[chunk_k], ups[chunk_k];
        for (nk_size_t row = 0; row != rows; ++row) {
            nk_bf16_t const *const gate_row = row_at_(gate, row, gate_stride);
            nk_bf16_t const *const up_row = up ? row_at_(up, row, up_stride) : nullptr;
            nk_bf16_t *const y_row = row_at_(y, row, y_stride);
            for (nk_size_t first = 0; first < columns; first += chunk_k) {
                nk_size_t const count = columns - first < chunk_k ? columns - first : chunk_k;
                nk_status_t status = cast_(gate_row + first, nk_bf16_k, gates, nk_f32_k, count, stream);
                if (status == nk_success_k && up_row)
                    status = cast_(up_row + first, nk_bf16_k, ups, nk_f32_k, count, stream);
                if (status != nk_success_k) return status;
                gate_row_({gates, count},
                          up_row ? std::span<nk_f32_t const> {ups, count} : std::span<nk_f32_t const> {},
                          {gates, count}, gate_scale, output_scale);
                if (status = cast_(gates, nk_f32_k, y_row + first, nk_bf16_k, count, stream); status != nk_success_k)
                    return status;
            }
        }
        return nk_success_k;
    }

  private:

    /** One row of @c silu(gate·gate_scale)·up·output_scale into @p y ; an empty @p up gates by
     *  one. */
    static void gate_row_(std::span<nk_f32_t const> gate, std::span<nk_f32_t const> up, std::span<nk_f32_t> y,
                          nk_f32_t gate_scale, nk_f32_t output_scale) noexcept {
        for (nk_size_t column = 0; column != gate.size(); ++column) {
            nk_f32_t const scaled = gate[column] * gate_scale;
            nk_f32_t const gated = scaled / (1.0f + std::exp(-scaled)) * (up.empty() ? 1.0f : up[column]);
            y[column] = gated * output_scale;
        }
    }
};

#if NUMKONG_TARGET_RVV

/** RISC-V vectors' casts, arithmetic, reductions and trigonometry, which also rotate as @ref mrope
 *  composes them, with the norms and gates composed from them, since NumKong has no RISC-V kernel
 *  for those. */
struct rvv_elementwise_kit_t {
    static constexpr nk_capability_t capability_k = nk_cap_rvv_k;
    static constexpr bool compiled_k = true;
    static constexpr auto cast = nk_cast_rvv;
    static constexpr auto sum_f32 = nk_each_sum_f32_rvv;
    static constexpr auto scale_f32 = nk_each_scale_f32_rvv;
    static constexpr auto blend_f32 = nk_each_blend_f32_rvv;
    static constexpr auto fma_f32 = nk_each_fma_f32_rvv;
    static constexpr auto reduce_f32 = nk_reduce_minmax_f32_rvv;
    static constexpr auto sine_f32 = nk_trig_sin_f32_rvv;
    static constexpr auto cosine_f32 = nk_trig_cos_f32_rvv;
    using norms_t = composed_norms<nk_cast_rvv, nk_dot_f32_rvv, nk_each_scale_f32_rvv, nk_each_fma_f32_rvv>;
    using gates_t = composed_swiglu<nk_cast_rvv>;
    static constexpr auto rmscast_bf16 = norms_t::rmscast_bf16;
    static constexpr auto rmsnorm_bf16 = norms_t::rmsnorm_bf16;
    static constexpr auto swiglu_f32 = gates_t::swiglu_f32;
    static constexpr auto swiglu_bf16 = gates_t::swiglu_bf16;
};
#else
struct rvv_elementwise_kit_t {
    static constexpr nk_capability_t capability_k = nk_cap_rvv_k;
    static constexpr bool compiled_k = false;
};
#endif // NUMKONG_TARGET_RVV

#if NUMKONG_TARGET_V128

/** WebAssembly's fixed-width sums over the serial rest: NumKong runs the casts, arithmetic,
 *  reductions and trigonometry only under relaxed SIMD, so fixed-width SIMD has nothing to compose
 *  them from. */
struct v128_elementwise_kit_t {
    static constexpr nk_capability_t capability_k = nk_cap_v128_k;
    static constexpr bool compiled_k = true;
    static constexpr auto cast = serial_elementwise_kit_t::cast;
    static constexpr auto sum_f32 = nk_each_sum_f32_v128;
    static constexpr auto scale_f32 = serial_elementwise_kit_t::scale_f32;
    static constexpr auto blend_f32 = serial_elementwise_kit_t::blend_f32;
    static constexpr auto fma_f32 = serial_elementwise_kit_t::fma_f32;
    static constexpr auto reduce_f32 = serial_elementwise_kit_t::reduce_f32;
    static constexpr auto sine_f32 = serial_elementwise_kit_t::sine_f32;
    static constexpr auto cosine_f32 = serial_elementwise_kit_t::cosine_f32;
    static constexpr auto rope_bf16 = serial_elementwise_kit_t::rope_bf16;
    static constexpr auto rmscast_bf16 = serial_elementwise_kit_t::rmscast_bf16;
    static constexpr auto rmsnorm_bf16 = serial_elementwise_kit_t::rmsnorm_bf16;
    static constexpr auto swiglu_f32 = serial_elementwise_kit_t::swiglu_f32;
    static constexpr auto swiglu_bf16 = serial_elementwise_kit_t::swiglu_bf16;
};
#else
struct v128_elementwise_kit_t {
    static constexpr nk_capability_t capability_k = nk_cap_v128_k;
    static constexpr bool compiled_k = false;
};
#endif // NUMKONG_TARGET_V128

#if NUMKONG_TARGET_V128 && NUMKONG_TARGET_V128RELAXED

/** Relaxed SIMD's casts, arithmetic, reductions and trigonometry, which also rotate as @ref mrope
 *  composes them, over the fixed-width sums, with the norms and gates composed from them, since
 *  NumKong has no WebAssembly kernel for those. */
struct v128relaxed_elementwise_kit_t {
    static constexpr nk_capability_t capability_k = nk_cap_v128_k | nk_cap_v128relaxed_k;
    static constexpr bool compiled_k = true;
    static constexpr auto cast = nk_cast_v128relaxed;
    static constexpr auto sum_f32 = nk_each_sum_f32_v128;
    static constexpr auto scale_f32 = nk_each_scale_f32_v128relaxed;
    static constexpr auto blend_f32 = nk_each_blend_f32_v128relaxed;
    static constexpr auto fma_f32 = nk_each_fma_f32_v128relaxed;
    static constexpr auto reduce_f32 = nk_reduce_minmax_f32_v128relaxed;
    static constexpr auto sine_f32 = nk_trig_sin_f32_v128relaxed;
    static constexpr auto cosine_f32 = nk_trig_cos_f32_v128relaxed;
    using norms_t = composed_norms<nk_cast_v128relaxed, nk_dot_f32_v128relaxed, nk_each_scale_f32_v128relaxed,
                                   nk_each_fma_f32_v128relaxed>;
    using gates_t = composed_swiglu<nk_cast_v128relaxed>;
    static constexpr auto rmscast_bf16 = norms_t::rmscast_bf16;
    static constexpr auto rmsnorm_bf16 = norms_t::rmsnorm_bf16;
    static constexpr auto swiglu_f32 = gates_t::swiglu_f32;
    static constexpr auto swiglu_bf16 = gates_t::swiglu_bf16;
};
#else
struct v128relaxed_elementwise_kit_t {
    static constexpr nk_capability_t capability_k = nk_cap_v128_k | nk_cap_v128relaxed_k;
    static constexpr bool compiled_k = false;
};
#endif // NUMKONG_TARGET_V128 && NUMKONG_TARGET_V128RELAXED

/** The element-wise kit for one capability ceiling. */
template <nk_capability_t caps_>
struct elementwise_ladder {
    using kit_t = typename highest_kit<caps_, serial_elementwise_kit_t, sapphire_elementwise_kit_t,
                                       genoa_elementwise_kit_t, icelake_elementwise_kit_t, skylake_elementwise_kit_t,
                                       haswell_elementwise_kit_t, neon_elementwise_kit_t, rvv_elementwise_kit_t,
                                       v128relaxed_elementwise_kit_t, v128_elementwise_kit_t>::type;
};

#pragma endregion Elementwise

#pragma region Entries

/** Rows @c [t,:] of @p rows for each text segment's cells, copied from @c table[token] ; other rows
 *  stay as they are. Token IDs are trusted: packers write bytes only. */
[[nodiscard]] inline static uu::status_t gather_bytes(segment_layout_t const &layout,
                                                      nk::tensor_view<token_id_t, 1> grid_tokens, bf16r2_view_t table,
                                                      bf16r2_span_t rows, uu::pool_view_t pool) noexcept {
    std::size_t const row_bytes = rows.extent(1) * sizeof(nk_bf16_t);
    return pool.for_n(layout.lengths.extent(0), [&](std::size_t segment, std::size_t) noexcept {
        if (layout.modality[segment] != modality_t::text_k) return;
        for (u32_t cell = 0; cell != layout.lengths[segment]; ++cell) {
            std::size_t const token = grid_tokens[layout.grid_offsets[segment] + cell];
            std::memcpy(rows.row(layout.packed_offsets[segment] + cell).data(), table.row(token).data(), row_bytes);
        }
    });
}

/** The gain and bias rows of @p affine @b [2,spatial·channels] , channels innermost, that turn a
 *  widened pixel row into @c (byte/255−mean)·inverse_std in one fused multiply-add. */
inline static void fill_patch_affine(f32r1_view_t mean, f32r1_view_t inverse_std, f32r2_span_t affine) noexcept {
    std::size_t const channels = mean.extent(0), width = affine.extent(1);
    f32_t *const gain = affine.row(0).data(), *const bias = affine.row(1).data();
    for (std::size_t scalar = 0; scalar != width; ++scalar) {
        std::size_t const channel = scalar % channels;
        gain[scalar] = inverse_std[channel] / 255.0f;
        bias[scalar] = -mean[channel] * inverse_std[channel];
    }
}

/** Segment @p segment 's pixel cells @b [cells,spatial,channels] , when it is @p target , into
 *  patch rows @b [rows,spatial·channels] : each cell widens into @p wide , takes the gain and bias
 *  of @ref fill_patch_affine , and narrows into its row. */
template <typename elementwise_kit_type_>
[[nodiscard]] inline static nk_status_t stage_patches(segment_layout_t const &layout, std::size_t segment,
                                                      modality_t target, nk::tensor_view<pixel_t, 3> grid_pixels,
                                                      f32r2_view_t affine, f32r1_span_t wide,
                                                      bf16r2_span_t patches) noexcept {
    if (layout.modality[segment] != target) return nk_success_k;
    std::size_t const width = affine.extent(1);
    f32_t const one = 1.0f;
    nk_status_t status = nk_success_k;
    for (u32_t cell = 0; status == nk_success_k && cell != layout.lengths[segment]; ++cell) {
        pixel_t const *const pixels = grid_pixels.row(layout.grid_offsets[segment] + cell).data();
        status = elementwise_kit_type_::cast(pixels, nk_u8_k, wide.data(), nk_f32_k, width, nullptr);
        if (status == nk_success_k)
            status = elementwise_kit_type_::fma_f32(wide.data(), affine.row(0).data(), affine.row(1).data(), width,
                                                    &one, &one, wide.data(), nullptr);
        if (status == nk_success_k)
            status = elementwise_kit_type_::cast(wide.data(), nk_f32_k,
                                                 patches.row(layout.packed_offsets[segment] + cell).data(), nk_bf16_k,
                                                 width, nullptr);
    }
    return status;
}

/** Segment @p segment 's raw frames @b [cells,chunk_samples] , when it is audio, narrowed into its
 *  rows in one cast, as both sides keep its cells back to back. */
template <typename elementwise_kit_type_>
[[nodiscard]] inline static nk_status_t stage_frames(segment_layout_t const &layout, std::size_t segment,
                                                     nk::tensor_view<f32_t, 2> grid_samples,
                                                     bf16r2_span_t frames) noexcept {
    std::size_t const cells = layout.lengths[segment];
    if (layout.modality[segment] != modality_t::audio_k || cells == 0) return nk_success_k;
    return elementwise_kit_type_::cast(grid_samples.row(layout.grid_offsets[segment]).data(), nk_f32_k,
                                       frames.row(layout.packed_offsets[segment]).data(), nk_bf16_k,
                                       cells * grid_samples.extent(1), nullptr);
}

/** Every packed row's @c (t,h,w) into @p coords @b [rows,3] : text and audio @c (k,0,0) , image
 *  @c (0,k/cols,k%cols) , video @c (group·tube_frames·frame_stride,r,c) , PAD rows untouched. */
[[nodiscard]] inline static uu::status_t fill_coords(segment_layout_t const &layout,
                                                     nk::tensor_view<u16_t, 2> image_shapes,
                                                     nk::tensor_view<u16_t, 2> video_shapes, u32_t tube_frames,
                                                     u32_t frame_stride, i16r2_span_t coords,
                                                     uu::pool_view_t pool) noexcept {
    return pool.for_n(layout.lengths.extent(0), [&](std::size_t segment, std::size_t) noexcept {
        modality_t const modality = layout.modality[segment];
        for (u32_t cell = 0; cell != layout.lengths[segment]; ++cell) {
            u32_t temporal = 0, height = 0, width = 0;
            switch (modality) {
            case modality_t::text_k:
            case modality_t::audio_k: temporal = cell; break;
            case modality_t::image_k: {
                u32_t const columns = image_shapes[segment, 1];
                if (columns == 0u) continue;
                height = cell / columns, width = cell % columns;
                break;
            }
            case modality_t::video_k: {
                u32_t const columns = video_shapes[segment, 2];
                u32_t const groups_per_frame = static_cast<u32_t>(video_shapes[segment, 1]) * columns;
                if (groups_per_frame == 0u) continue;
                u32_t const planar = cell % groups_per_frame;
                temporal = cell / groups_per_frame * tube_frames * frame_stride;
                height = planar / columns, width = planar % columns;
                break;
            }
            }
            std::size_t const row = layout.packed_offsets[segment] + cell;
            coords[row, 0] = static_cast<i16_t>(temporal);
            coords[row, 1] = static_cast<i16_t>(height);
            coords[row, 2] = static_cast<i16_t>(width);
        }
    });
}

/** Rotate row @p row 's @p heads head vectors from column @p role_offset in place: the NeoX
 *  split-half pair @c (k,k+head_width/2) turns by its angle at the row's @c (t,h,w) , each
 *  coordinate clamped into its grid. The row's angles gather into @p wide , temporal pairs first,
 *  their cosines and sines follow them, and every head turns by both. A kit without NumKong's
 *  rotation widens each head behind them and turns its halves with two fused multiplies each. */
template <typename elementwise_kit_type_>
[[nodiscard]] inline static nk_status_t mrope(bf16r2_span_t values, std::size_t row, u32_t role_offset, u32_t heads,
                                              u32_t head_width, i16r2_view_t coords, rope_grids_t angles,
                                              f32r1_span_t wide) noexcept {
    std::size_t const pairs = head_width / 2u;
    f32_t *const angle = wide.data(), *const cosine = angle + pairs, *const sine = cosine + pairs;
    f32r2_view_t const grids[3] = {angles.temporal, angles.height, angles.width};
    std::size_t gathered = 0;
    for (std::size_t axis = 0; axis != 3; ++axis) {
        i16_t const coordinate = coords[row, axis];
        std::size_t position = coordinate > 0 ? static_cast<std::size_t>(coordinate) : 0;
        if (position >= grids[axis].extent(0)) position = grids[axis].extent(0) - 1u;
        std::memcpy(angle + gathered, grids[axis].row(position).data(), grids[axis].extent(1) * sizeof(f32_t));
        gathered += grids[axis].extent(1);
    }
    nk_status_t status = elementwise_kit_type_::cosine_f32(angle, pairs, cosine, nullptr);
    if (status == nk_success_k) status = elementwise_kit_type_::sine_f32(angle, pairs, sine, nullptr);
    nk_bf16_t *const rotated = values.row(row).data() + role_offset;
    if constexpr (rope_rotations<elementwise_kit_type_>) {
        std::size_t const stride = values.extent(1) * sizeof(nk_bf16_t);
        if (status == nk_success_k)
            status = elementwise_kit_type_::rope_bf16(rotated, cosine, sine, rotated, 1, heads, head_width, stride,
                                                      stride, nullptr);
    }
    else {
        f32_t *const low = sine + pairs, *const high = low + pairs, *const turned = high + pairs;
        f32_t const one = 1.0f, zero = 0.0f, minus_one = -1.0f;
        for (u32_t head = 0; status == nk_success_k && head != heads; ++head) {
            nk_bf16_t *const vector = rotated + static_cast<std::size_t>(head) * head_width;
            status = elementwise_kit_type_::cast(vector, nk_bf16_k, low, nk_f32_k, head_width, nullptr);
            if (status == nk_success_k) // turned = low·sine + high·cosine
                status = elementwise_kit_type_::fma_f32(low, sine, low, pairs, &one, &zero, turned, nullptr);
            if (status == nk_success_k)
                status = elementwise_kit_type_::fma_f32(high, cosine, turned, pairs, &one, &one, turned, nullptr);
            if (status == nk_success_k) // low = low·cosine − high·sine
                status = elementwise_kit_type_::fma_f32(high, sine, high, pairs, &one, &zero, high, nullptr);
            if (status == nk_success_k)
                status = elementwise_kit_type_::fma_f32(low, cosine, high, pairs, &one, &minus_one, low, nullptr);
            if (status == nk_success_k)
                status = elementwise_kit_type_::cast(low, nk_f32_k, vector, nk_bf16_k, pairs, nullptr);
            if (status == nk_success_k)
                status = elementwise_kit_type_::cast(turned, nk_f32_k, vector + pairs, nk_bf16_k, pairs, nullptr);
        }
    }
    return status;
}

/** Quantize row @p row of @p values by its own @c max_abs/240 into @p shadow , keeping the scale in
 *  @p token_scales : a BFloat16 row widens into @p wide first, a Float32 one is read in place, and
 *  the scaled row lands in @p wide before it narrows. */
template <typename elementwise_kit_type_, typename value_type_>
[[nodiscard]] inline static nk_status_t quantize_tokens_e4m3(nk::tensor_view<value_type_, 2> values, std::size_t row,
                                                             e4m3r2_span_t shadow, f32r1_span_t token_scales,
                                                             f32r1_span_t wide) noexcept {
    std::size_t const width = values.extent(1);
    f32_t const *source = wide.data();
    nk_status_t status = nk_success_k;
    if constexpr (std::is_same_v<value_type_, f32_t>) source = values.row(row).data();
    else status = elementwise_kit_type_::cast(values.row(row).data(), nk_bf16_k, wide.data(), nk_f32_k, width, nullptr);
    f32_t lowest = 0.0f, highest = 0.0f;
    nk_size_t lowest_index = 0, highest_index = 0;
    if (status == nk_success_k)
        status = elementwise_kit_type_::reduce_f32(source, width, sizeof(f32_t), &lowest, &lowest_index, &highest,
                                                   &highest_index, nullptr);
    f32_t const scale = std::fmax(std::fmax(-lowest, highest) / e4m3_range_k, e4m3_descale_floor_k);
    f32_t const inverse = 1.0f / scale, shift = 0.0f;
    token_scales[row] = scale;
    if (status == nk_success_k)
        status = elementwise_kit_type_::scale_f32(source, width, &inverse, &shift, wide.data(), nullptr);
    if (status == nk_success_k)
        status = elementwise_kit_type_::cast(wide.data(), nk_f32_k, shadow.row(row).data(), nk_e4m3_k, width, nullptr);
    return status;
}

/** Row @p row of @c values=products·weight_descale·token_scales+residual , narrowed: the
 *  residual widens into @p wide and the products blend into it there; an empty @p residual drops
 *  its term. */
template <typename elementwise_kit_type_>
[[nodiscard]] inline static nk_status_t rescale_tokens(f32r2_view_t products, std::size_t row,
                                                       f32r1_view_t token_scales, f32_t weight_descale,
                                                       bf16r2_view_t residual, bf16r2_span_t values,
                                                       f32r1_span_t wide) noexcept {
    std::size_t const width = values.extent(1);
    f32_t const weight = weight_descale * token_scales[row], one = 1.0f, zero = 0.0f;
    f32_t const *const product = products.row(row).data();
    nk_status_t status = nk_success_k;
    if (residual.empty())
        status = elementwise_kit_type_::scale_f32(product, width, &weight, &zero, wide.data(), nullptr);
    else {
        status = elementwise_kit_type_::cast(residual.row(row).data(), nk_bf16_k, wide.data(), nk_f32_k, width,
                                             nullptr);
        if (status == nk_success_k)
            status = elementwise_kit_type_::blend_f32(product, wide.data(), width, &weight, &one, wide.data(), nullptr);
    }
    if (status == nk_success_k)
        status = elementwise_kit_type_::cast(wide.data(), nk_f32_k, values.row(row).data(), nk_bf16_k, width, nullptr);
    return status;
}

#pragma endregion Entries

} // namespace unum::uform::encoder
