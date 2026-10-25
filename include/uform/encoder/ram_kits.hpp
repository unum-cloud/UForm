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
 *  Every kit carries both block precisions, as precision is a runtime value of the tower, never a
 *  type.
 */
#pragma once

#include <numkong/attention.h> // `nk_attention_pack_*`, `nk_attention_packed_*`
#include <numkong/cast.h>      // `nk_cast_*`
#include <numkong/dots.h>      // `nk_dots_pack_*`, `nk_dots_packed_*`
#include <numkong/each.h> // `nk_each_sum_f32_*`, `nk_each_rmscast_bf16_*`, `nk_each_rmsnorm_bf16_*`, `nk_each_swiglu_*`

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

/** The NEON floor: BFloat16 products in NEON, FP8 ones in serial code. */
struct neon_products_kit_t {
    static constexpr nk_capability_t capability_k = nk_cap_neon_k;
    static constexpr bool compiled_k = true;
    static constexpr auto pack_size_bf16 = nk_dots_pack_size_bf16_neon;
    static constexpr auto pack_bf16 = nk_dots_pack_bf16_neon;
    static constexpr auto multiply_bf16 = nk_dots_packed_bf16_neon;
    static constexpr auto pack_size_e4m3 = nk_dots_pack_size_e4m3_serial;
    static constexpr auto pack_e4m3 = nk_dots_pack_e4m3_serial;
    static constexpr auto multiply_e4m3 = nk_dots_packed_e4m3_serial;
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

/** WebAssembly's fixed-width SIMD: BFloat16 products in it, FP8 ones in serial code. */
struct v128_products_kit_t {
    static constexpr nk_capability_t capability_k = nk_cap_v128_k;
    static constexpr bool compiled_k = true;
    static constexpr auto pack_size_bf16 = nk_dots_pack_size_bf16_v128;
    static constexpr auto pack_bf16 = nk_dots_pack_bf16_v128;
    static constexpr auto multiply_bf16 = nk_dots_packed_bf16_v128;
    static constexpr auto pack_size_e4m3 = nk_dots_pack_size_e4m3_serial;
    static constexpr auto pack_e4m3 = nk_dots_pack_e4m3_serial;
    static constexpr auto multiply_e4m3 = nk_dots_packed_e4m3_serial;
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
    static constexpr nk_capability_t picked_k = kit_t::capability_k;
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
    static constexpr nk_capability_t picked_k = kit_t::capability_k;
};

#pragma endregion Attention

#pragma region Elementwise

/** The casts, sums, norms and gates every CPU runs. */
struct serial_elementwise_kit_t {
    static constexpr nk_capability_t capability_k = nk_cap_serial_k;
    static constexpr bool compiled_k = true;
    static constexpr auto cast = nk_cast_serial;
    static constexpr auto sum_f32 = nk_each_sum_f32_serial;
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

#if NUMKONG_TARGET_RVV

/** RISC-V vectors' casts and sums over serial norms and gates. */
struct rvv_elementwise_kit_t {
    static constexpr nk_capability_t capability_k = nk_cap_rvv_k;
    static constexpr bool compiled_k = true;
    static constexpr auto cast = nk_cast_rvv;
    static constexpr auto sum_f32 = nk_each_sum_f32_rvv;
    static constexpr auto rmscast_bf16 = nk_each_rmscast_bf16_serial;
    static constexpr auto rmsnorm_bf16 = nk_each_rmsnorm_bf16_serial;
    static constexpr auto swiglu_f32 = nk_each_swiglu_f32_serial;
    static constexpr auto swiglu_bf16 = nk_each_swiglu_bf16_serial;
};
#else
struct rvv_elementwise_kit_t {
    static constexpr nk_capability_t capability_k = nk_cap_rvv_k;
    static constexpr bool compiled_k = false;
};
#endif // NUMKONG_TARGET_RVV

#if NUMKONG_TARGET_V128

/** WebAssembly's fixed-width sums over serial casts, norms and gates. */
struct v128_elementwise_kit_t {
    static constexpr nk_capability_t capability_k = nk_cap_v128_k;
    static constexpr bool compiled_k = true;
    static constexpr auto cast = nk_cast_serial;
    static constexpr auto sum_f32 = nk_each_sum_f32_v128;
    static constexpr auto rmscast_bf16 = nk_each_rmscast_bf16_serial;
    static constexpr auto rmsnorm_bf16 = nk_each_rmsnorm_bf16_serial;
    static constexpr auto swiglu_f32 = nk_each_swiglu_f32_serial;
    static constexpr auto swiglu_bf16 = nk_each_swiglu_bf16_serial;
};
#else
struct v128_elementwise_kit_t {
    static constexpr nk_capability_t capability_k = nk_cap_v128_k;
    static constexpr bool compiled_k = false;
};
#endif // NUMKONG_TARGET_V128

#if NUMKONG_TARGET_V128 && NUMKONG_TARGET_V128RELAXED

/** Relaxed SIMD's casts over the fixed-width sums. */
struct v128relaxed_elementwise_kit_t {
    static constexpr nk_capability_t capability_k = nk_cap_v128_k | nk_cap_v128relaxed_k;
    static constexpr bool compiled_k = true;
    static constexpr auto cast = nk_cast_v128relaxed;
    static constexpr auto sum_f32 = nk_each_sum_f32_v128;
    static constexpr auto rmscast_bf16 = nk_each_rmscast_bf16_serial;
    static constexpr auto rmsnorm_bf16 = nk_each_rmsnorm_bf16_serial;
    static constexpr auto swiglu_f32 = nk_each_swiglu_f32_serial;
    static constexpr auto swiglu_bf16 = nk_each_swiglu_bf16_serial;
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
    static constexpr nk_capability_t picked_k = kit_t::capability_k;
};

#pragma endregion Elementwise

} // namespace unum::uform::encoder
