/**
 *  @file src/lib.hpp
 *  @author Ash Vardanian
 *  @date October 17, 2026
 *  @brief What every verb of the C interface shares: the outcome and progress crossings, the
 *      executor and arena handles, and the one entry every verb's call takes.
 *
 *  Each verb file includes this, so one verb reads the same in every header it implements. An
 *  executor runs one call at a time: an eager one on its caller, a deferred one on its own pool's
 *  thread 0, which forks the rest of the pool from inside its slice, so nothing waits in a table.
 *
 *  Every verb opens the same way:
 *
 *  @code{.cpp}
 *  UFORM_EXPORT void uform_text_encode(uform_text_encode_t *arguments) {
 *      if (submitted<uform_text_encode>(arguments)) return;
 *      call_outcome_t outcome(*arguments);
 *      uform_arena_t const arena = claim_arena(arguments->arena, arguments->executor, {inputs}, outcome);
 *      if (!arena) return;
 *      // ...
 *  }
 *  @endcode
 */
#pragma once

#include <cstddef> // `std::size_t`

#include <initializer_list> // `std::initializer_list` - the inputs a call checks against its arena
#include <limits>           // `std::numeric_limits` - the overflow guard of a typed allocation
#include <type_traits>      // `std::is_trivially_copyable_v`
#include <variant>          // `std::variant` - the executor's kind, the arena's residency, a deferred call

#include <forkunion/atomics.hpp>      // `fu::standard_atomic_ref` & kin - the classes' atomic references
#include <forkunion/capabilities.hpp> // `fu::x86_pause_t`, `fu::arm64_wfet_t` & waiter kin
#include <forkunion/types.hpp>        // `fu::spin_mutex_t`, `fu::wait_capped_k`
#include <numkong/capabilities.h>     // `nk_capability_t`, `nk_stream_t`
#include <numkong/memory.h>       // `nk_allocator_t`, `nk_allocator_init_heap`
#include <numkong/memory.hpp>     // `nk::stream_t` - the stream an executor owns

#include "uform/uform.h" // every argument structure

#include "uform/utilities/array.hpp"      // `per_gpu_array`
#include "uform/utilities/ram_device.hpp" // `executor_t`, `lent_pool_t`, `pool_view_t`, `topology_t`
#include "uform/utilities/types.hpp"      // `status_t`, the integer aliases, `sz::string_view_t`, `st::basic_vector`
#include "uform/encoder/tower.hpp"   // `geometry_t`, `precision_t`, `outputs_t` - what a model's towers take

namespace unum::uform {

namespace fu = ashvardanian::forkunion;
namespace nk = ashvardanian::numkong;
namespace sz = ashvardanian::stringzilla;
namespace st = ashvardanian::smashtable;

using utilities::b8_t;
using utilities::bytes_of;
using utilities::bytes_span_t;
using utilities::bytes_view_t;
using utilities::caller_exclusivity_t;
using utilities::checkpoint_to_load_t;
using utilities::divide_round_up;
using utilities::executor_t;
using utilities::expected;
using utilities::expected_t;
using utilities::f32_t;
using utilities::f32r2_span_t;
using utilities::f32r2_view_t;
using utilities::f64_t;
using utilities::i32_t;
using utilities::i64_t;
using utilities::lent_pool_t;
using utilities::max_gpus_per_host_k;
using utilities::per_gpu_array;
using utilities::pixel_t;
using utilities::pool_view_t;
using utilities::round_up_to_multiple;
using utilities::status_t;
using utilities::tape_view;
using utilities::token_id_t;
using utilities::topology_t;
using utilities::u16_t;
using utilities::u32_t;
using utilities::u64_t;
using utilities::u8_t;
using utilities::wraps_on_multiply;

/** Reports a progress handler hears at most when its call names zero steps. */
inline constexpr std::size_t default_progress_steps_k = 10;

/** One core in this many stays idle when an executor derives its thread count, so threads are 7/8
 *  of the cores. */
inline constexpr std::size_t idle_cores_denominator_k = 8;

/** A borrowed byte run of the C interface, a path or a spelling, as the engine's string view. */
[[nodiscard]] inline sz::string_view_t text_of(uform_bytes_view_t bytes) noexcept {
    return {static_cast<char const *>(static_cast<void const *>(bytes.data)), bytes.length};
}

/** The @p count entries of a C tape as the engine's typed tape, whose elements reach the last
 *  offset; null offsets make an empty tape, and null data no elements. */
template <typename element_type_>
[[nodiscard]] tape_view<element_type_> tape_of(uform_tape_view_t tape, std::size_t count) noexcept {
    if (!tape.offsets) return {};
    return {{static_cast<element_type_ *>(tape.data), tape.data ? tape.offsets[count] : 0}, {tape.offsets, count + 1}};
}

#pragma region Classes

/** The CPU classes of @c uform_cpu_class_t as the engine names them, with the same ordinals. */
enum class cpu_class_t : u32_t {
    unknown_k = uform_cpu_class_unknown_k,
    serial_k = uform_cpu_class_serial_k,
    intel_haswell_k = uform_cpu_class_intel_haswell_k,
    intel_sierra_k = uform_cpu_class_intel_sierra_k,
    intel_skylake_k = uform_cpu_class_intel_skylake_k,
    intel_icelake_k = uform_cpu_class_intel_icelake_k,
    amd_genoa_k = uform_cpu_class_amd_genoa_k,
    intel_sapphire_k = uform_cpu_class_intel_sapphire_k,
    intel_granite_k = uform_cpu_class_intel_granite_k,
    intel_diamond_k = uform_cpu_class_intel_diamond_k,
    aws_g2_k = uform_cpu_class_aws_g2_k,
    apple_m2_k = uform_cpu_class_apple_m2_k,
    apple_m4_k = uform_cpu_class_apple_m4_k,
    aws_g5_k = uform_cpu_class_aws_g5_k,
    nvidia_vera_k = uform_cpu_class_nvidia_vera_k,
    alibaba_c930_k = uform_cpu_class_alibaba_c930_k,
    wasm32_v128_k = uform_cpu_class_wasm32_v128_k,
    wasm64_extended_k = uform_cpu_class_wasm64_extended_k,
    monoversioned_k = uform_cpu_class_monoversioned_k,
};

/** The GPU classes of @c uform_gpu_class_t as the engine names them, with the same ordinals. */
enum class gpu_class_t : u32_t {
    unknown_k = uform_gpu_class_unknown_k,
    nvidia_ampere_k = uform_gpu_class_nvidia_ampere_k,
    nvidia_ada_k = uform_gpu_class_nvidia_ada_k,
    nvidia_hopper_k = uform_gpu_class_nvidia_hopper_k,
    nvidia_blackwell_k = uform_gpu_class_nvidia_blackwell_k,
    nvidia_blackwellrtx_k = uform_gpu_class_nvidia_blackwellrtx_k,
    nvidia_rubin_k = uform_gpu_class_nvidia_rubin_k,
    amd_cdna3_k = uform_gpu_class_amd_cdna3_k,
    amd_cdna4_k = uform_gpu_class_amd_cdna4_k,
    amd_cdna5_k = uform_gpu_class_amd_cdna5_k,
    apple_metal7_k = uform_gpu_class_apple_metal7_k,
    apple_metal9_k = uform_gpu_class_apple_metal9_k,
    apple_metal10_k = uform_gpu_class_apple_metal10_k,
};

/** Lowers one CPU class onto NumKong's compute mask - cumulative within a platform, so the chains
 *  pick their best row under it. */
constexpr nk_capability_t numkong_capabilities_of(cpu_class_t cpu_class) noexcept {
    switch (cpu_class) {
    case cpu_class_t::intel_haswell_k: return nk_cap_serial_k | nk_cap_haswell_k;
    case cpu_class_t::intel_sierra_k:
        return numkong_capabilities_of(cpu_class_t::intel_haswell_k) | nk_cap_alder_k | nk_cap_sierra_k;
    case cpu_class_t::intel_skylake_k: return numkong_capabilities_of(cpu_class_t::intel_haswell_k) | nk_cap_skylake_k;
    case cpu_class_t::intel_icelake_k: return numkong_capabilities_of(cpu_class_t::intel_skylake_k) | nk_cap_icelake_k;
    case cpu_class_t::amd_genoa_k: return numkong_capabilities_of(cpu_class_t::intel_icelake_k) | nk_cap_genoa_k;
    case cpu_class_t::intel_sapphire_k:
        return numkong_capabilities_of(cpu_class_t::amd_genoa_k) | nk_cap_sapphire_k | nk_cap_sapphireamx_k;
    case cpu_class_t::intel_granite_k:
        return numkong_capabilities_of(cpu_class_t::intel_sapphire_k) | nk_cap_graniteamx_k;
    case cpu_class_t::intel_diamond_k:
        return numkong_capabilities_of(cpu_class_t::intel_granite_k) | nk_cap_alder_k | nk_cap_sierra_k |
               nk_cap_diamond_k;
    case cpu_class_t::aws_g2_k: return nk_cap_serial_k | nk_cap_neon_k;
    case cpu_class_t::apple_m2_k:
    case cpu_class_t::aws_g5_k:
        return numkong_capabilities_of(cpu_class_t::aws_g2_k) | nk_cap_neonsdot_k | nk_cap_neonbfdot_k |
               nk_cap_neonfhm_k;
    case cpu_class_t::apple_m4_k:
        return numkong_capabilities_of(cpu_class_t::apple_m2_k) | nk_cap_sme_k | nk_cap_smebi32_k;
    case cpu_class_t::nvidia_vera_k: return numkong_capabilities_of(cpu_class_t::apple_m2_k) | nk_cap_neonfp8_k;
    case cpu_class_t::alibaba_c930_k: return nk_cap_serial_k | nk_cap_rvv_k | nk_cap_rvvbb_k;
    case cpu_class_t::wasm32_v128_k: return nk_cap_serial_k | nk_cap_v128_k;
    case cpu_class_t::wasm64_extended_k:
        return numkong_capabilities_of(cpu_class_t::wasm32_v128_k) | nk_cap_v128relaxed_k;
    case cpu_class_t::monoversioned_k: return nk_cpu_capabilities_compiled_();
    default: return nk_cap_serial_k;
    }
}

/** Lowers one GPU class onto NumKong's GPU capabilities. On NVIDIA, the Ampere & Ada bits
 *  accumulate, while Hopper, Blackwell & RTX Blackwell each run only their own kits, and Rubin
 *  rides the Blackwell kits until its own land. AMD and Apple generations accumulate over
 *  their vendor's floor. */
constexpr nk_capability_t numkong_capabilities_of(gpu_class_t gpu_class) noexcept {
    nk_capability_t const ampere = nk_cap_cuda_k | nk_cap_ampere_k, ada = ampere | nk_cap_ada_k;
    nk_capability_t const cdna3 = nk_cap_rocm_k | nk_cap_cdna3_k, cdna4 = cdna3 | nk_cap_cdna4_k;
    nk_capability_t const metal9 = nk_cap_metal_k | nk_cap_apple9_k;
    switch (gpu_class) {
    case gpu_class_t::nvidia_ada_k: return ada;
    case gpu_class_t::nvidia_hopper_k: return ada | nk_cap_hopper_k;
    case gpu_class_t::nvidia_blackwell_k:
    case gpu_class_t::nvidia_rubin_k: return ada | nk_cap_blackwell_k;
    case gpu_class_t::nvidia_blackwellrtx_k: return ada | nk_cap_blackwellrtx_k;
    case gpu_class_t::amd_cdna3_k: return cdna3;
    case gpu_class_t::amd_cdna4_k: return cdna4;
    case gpu_class_t::amd_cdna5_k: return cdna4 | nk_cap_cdna5_k;
    case gpu_class_t::apple_metal7_k: return nk_cap_metal_k;
    case gpu_class_t::apple_metal9_k: return metal9;
    case gpu_class_t::apple_metal10_k: return metal9 | nk_cap_apple10_k;
    case gpu_class_t::nvidia_ampere_k: return ampere;
    case gpu_class_t::unknown_k: return nk_cap_serial_k;
    }
    return nk_cap_serial_k;
}

/** The class a refused rung degrades to - the next one down its own lineage. On x86 the AVX-512
 *  P-cores and the AVX-VNNI E-cores share the Haswell floor. */
constexpr cpu_class_t parent_class_of(cpu_class_t cpu_class) noexcept {
    switch (cpu_class) {
    case cpu_class_t::intel_diamond_k: return cpu_class_t::intel_granite_k;
    case cpu_class_t::intel_granite_k: return cpu_class_t::intel_sapphire_k;
    case cpu_class_t::intel_sapphire_k: return cpu_class_t::amd_genoa_k;
    case cpu_class_t::amd_genoa_k: return cpu_class_t::intel_icelake_k;
    case cpu_class_t::intel_icelake_k: return cpu_class_t::intel_skylake_k;
    case cpu_class_t::intel_skylake_k: return cpu_class_t::intel_haswell_k;
    case cpu_class_t::intel_sierra_k: return cpu_class_t::intel_haswell_k;
    case cpu_class_t::nvidia_vera_k: return cpu_class_t::apple_m2_k;
    case cpu_class_t::aws_g5_k: return cpu_class_t::apple_m2_k;
    case cpu_class_t::apple_m4_k: return cpu_class_t::apple_m2_k;
    case cpu_class_t::apple_m2_k: return cpu_class_t::aws_g2_k;
    case cpu_class_t::wasm64_extended_k: return cpu_class_t::wasm32_v128_k;
    default: return cpu_class_t::serial_k;
    }
}

/** The class of a device as its vendor's driver reports its architecture: NVIDIA's compute
 *  capability, AMD's gfx major and minor digits; unknown for a device no class covers. */
constexpr gpu_class_t gpu_class_of(uform_gpu_vendor_t vendor, u32_t architecture_major,
                                   u32_t architecture_minor) noexcept {
    if (vendor == uform_gpu_vendor_nvidia_k) {
        if (architecture_major == 12) return gpu_class_t::nvidia_blackwellrtx_k;
        if (architecture_major == 10) return gpu_class_t::nvidia_blackwell_k;
        if (architecture_major == 9) return gpu_class_t::nvidia_hopper_k;
        if (architecture_major * 10 + architecture_minor >= 89) return gpu_class_t::nvidia_ada_k;
        if (architecture_major == 8) return gpu_class_t::nvidia_ampere_k;
    }
    if (vendor == uform_gpu_vendor_amd_k) {
        if (architecture_major == 9 && architecture_minor == 4) return gpu_class_t::amd_cdna3_k;
        if (architecture_major == 9 && architecture_minor == 5) return gpu_class_t::amd_cdna4_k;
        if (architecture_major == 12 && architecture_minor == 5) return gpu_class_t::amd_cdna5_k;
    }
    return gpu_class_t::unknown_k;
}

/** The wait instruction one CPU class pairs: x86 spins PAUSE, the WAITPKG Xeons and the monitored
 *  Arm & RISC-V classes arm TPAUSE, WFE & WRS, WebAssembly threads sleep in @c memory.atomic.wait,
 *  everything else yields - as does any class whose @c FORKUNION_TARGET_<BIT> is off in the probe
 *  list. WFE sleeps on uncapped waits and yields on capped ones, so it needs no bit to confirm; the
 *  timed WFET only a monoversioned build reaches, where a preset vouches for it. */
template <cpu_class_t cpu_class_>
struct waiter_for_cpu_class {
    using type = fu::standard_yield_t;
};
#if FORKUNION_TARGET_X86_PAUSE
template <>
struct waiter_for_cpu_class<cpu_class_t::intel_haswell_k> {
    using type = fu::x86_pause_t;
};
template <>
struct waiter_for_cpu_class<cpu_class_t::intel_skylake_k> {
    using type = fu::x86_pause_t;
};
template <>
struct waiter_for_cpu_class<cpu_class_t::intel_icelake_k> {
    using type = fu::x86_pause_t;
};
template <>
struct waiter_for_cpu_class<cpu_class_t::amd_genoa_k> {
    using type = fu::x86_pause_t;
};
template <>
struct waiter_for_cpu_class<cpu_class_t::intel_sapphire_k> {
    using type = fu::x86_pause_t;
};
template <>
struct waiter_for_cpu_class<cpu_class_t::intel_granite_k> {
    using type = fu::x86_pause_t;
};
#endif
#if FORKUNION_TARGET_X86_TPAUSE
template <>
struct waiter_for_cpu_class<cpu_class_t::intel_sierra_k> {
    using type = fu::x86_tpause_t;
};
template <>
struct waiter_for_cpu_class<cpu_class_t::intel_diamond_k> {
    using type = fu::x86_tpause_t;
};
#endif
#if FORKUNION_TARGET_ARM64_YIELD
template <>
struct waiter_for_cpu_class<cpu_class_t::aws_g2_k> {
    using type = fu::arm64_yield_t;
};
template <>
struct waiter_for_cpu_class<cpu_class_t::apple_m2_k> {
    using type = fu::arm64_yield_t;
};
template <>
struct waiter_for_cpu_class<cpu_class_t::apple_m4_k> {
    using type = fu::arm64_yield_t;
};
#endif
#if FORKUNION_TARGET_ARM64_WFE
template <>
struct waiter_for_cpu_class<cpu_class_t::aws_g5_k> {
    using type = fu::arm64_wfe_t;
};
template <>
struct waiter_for_cpu_class<cpu_class_t::nvidia_vera_k> {
    using type = fu::arm64_wfe_t;
};
#endif
#if FORKUNION_TARGET_RISCV64_WRS
template <>
struct waiter_for_cpu_class<cpu_class_t::alibaba_c930_k> {
    using type = fu::riscv64_wrs_t;
};
#endif
#if FORKUNION_TARGET_WASM_WAIT
template <>
struct waiter_for_cpu_class<cpu_class_t::wasm32_v128_k> {
    using type = fu::wasm_wait_t;
};
template <>
struct waiter_for_cpu_class<cpu_class_t::wasm64_extended_k> {
    using type = fu::wasm_wait_t;
};
#endif

/** The waiter a CPU class pairs, spelled short for the class units. */
template <cpu_class_t cpu_class_>
using waiter_of = typename waiter_for_cpu_class<cpu_class_>::type;

/** The atomic reference one CPU class pairs: single-instruction LSE read-modify-writes under RCpc
 *  acquiring loads on the Armv8.3+ classes, conditional adds on the CMPCCXADD Xeons, @c amocas on
 *  Zacas RISC-V, and on WebAssembly threads a @c memory.atomic.notify after every releasing write,
 *  so the sleeping waiter wakes on it - the floors, the older classes, and any class whose
 *  @c FORKUNION_TARGET_<BIT> is off in the probe list take the standard one. */
template <cpu_class_t cpu_class_>
struct atomic_ref_for_cpu_class {
    template <typename value_type_>
    using type = fu::standard_atomic_ref<value_type_>;
};
#if FORKUNION_TARGET_X86_CMPCCXADD
template <>
struct atomic_ref_for_cpu_class<cpu_class_t::intel_sierra_k> {
    template <typename value_type_>
    using type = fu::x86_cmpccxadd_atomic_ref<value_type_>;
};
template <>
struct atomic_ref_for_cpu_class<cpu_class_t::intel_diamond_k> {
    template <typename value_type_>
    using type = fu::x86_cmpccxadd_atomic_ref<value_type_>;
};
#endif
#if FORKUNION_TARGET_ARM64_RCPC
template <>
struct atomic_ref_for_cpu_class<cpu_class_t::apple_m2_k> {
    template <typename value_type_>
    using type = fu::arm64_rcpc_atomic_ref<value_type_>;
};
template <>
struct atomic_ref_for_cpu_class<cpu_class_t::apple_m4_k> {
    template <typename value_type_>
    using type = fu::arm64_rcpc_atomic_ref<value_type_>;
};
template <>
struct atomic_ref_for_cpu_class<cpu_class_t::aws_g5_k> {
    template <typename value_type_>
    using type = fu::arm64_rcpc_atomic_ref<value_type_>;
};
template <>
struct atomic_ref_for_cpu_class<cpu_class_t::nvidia_vera_k> {
    template <typename value_type_>
    using type = fu::arm64_rcpc_atomic_ref<value_type_>;
};
#elif FORKUNION_TARGET_ARM64_LSE
template <>
struct atomic_ref_for_cpu_class<cpu_class_t::apple_m2_k> {
    template <typename value_type_>
    using type = fu::arm64_lse_atomic_ref<value_type_>;
};
template <>
struct atomic_ref_for_cpu_class<cpu_class_t::apple_m4_k> {
    template <typename value_type_>
    using type = fu::arm64_lse_atomic_ref<value_type_>;
};
template <>
struct atomic_ref_for_cpu_class<cpu_class_t::aws_g5_k> {
    template <typename value_type_>
    using type = fu::arm64_lse_atomic_ref<value_type_>;
};
template <>
struct atomic_ref_for_cpu_class<cpu_class_t::nvidia_vera_k> {
    template <typename value_type_>
    using type = fu::arm64_lse_atomic_ref<value_type_>;
};
#endif
#if FORKUNION_TARGET_RISCV64_ZACAS
template <>
struct atomic_ref_for_cpu_class<cpu_class_t::alibaba_c930_k> {
    template <typename value_type_>
    using type = fu::riscv64_zacas_atomic_ref<value_type_>;
};
#endif
#if FORKUNION_TARGET_WASM_WAIT
template <>
struct atomic_ref_for_cpu_class<cpu_class_t::wasm32_v128_k> {
    template <typename value_type_>
    using type = fu::wasm_atomic_ref<value_type_>;
};
template <>
struct atomic_ref_for_cpu_class<cpu_class_t::wasm64_extended_k> {
    template <typename value_type_>
    using type = fu::wasm_atomic_ref<value_type_>;
};
#endif
#if UFORM_TARGET_MONOVERSIONED
template <>
struct waiter_for_cpu_class<cpu_class_t::monoversioned_k> {
    using type = fu::preferred_yield_t;
};
template <>
struct atomic_ref_for_cpu_class<cpu_class_t::monoversioned_k> {
    template <typename value_type_>
    using type = fu::preferred_atomic_ref<value_type_>;
};
#endif

/** The spin hints every core of an architecture carries - PAUSE, YIELD, Zihintpause: the runtime
 *  confirms nothing about them, and a class differing by one alone earns no rung. */
inline constexpr fu::capabilities_t baseline_hints_k = fu::capability_x86_pause_k | fu::capability_arm64_yield_k |
                                                       fu::capability_riscv64_pause_k;

/** Lowers one CPU class onto ForkUnion's mask - the bits the runtime must confirm before the
 *  class's waiter and atomic reference are trusted, read off the two types themselves. */
template <cpu_class_t cpu_class_>
constexpr fu::capabilities_t forkunion_capabilities_of() noexcept {
    constexpr fu::capabilities_t waits_k = waiter_of<cpu_class_>::capability_k;
    constexpr fu::capabilities_t atomics_k = atomic_ref_for_cpu_class<cpu_class_>::template type<u32_t>::capabilities_k;
    return static_cast<fu::capabilities_t>((waits_k | atomics_k) & ~baseline_hints_k);
}

/** ForkUnion's mask for a class named at run time, read off the template per class. */
constexpr fu::capabilities_t forkunion_capabilities_of(cpu_class_t cpu_class) noexcept {
    switch (cpu_class) {
    case cpu_class_t::intel_haswell_k: return forkunion_capabilities_of<cpu_class_t::intel_haswell_k>();
    case cpu_class_t::intel_sierra_k: return forkunion_capabilities_of<cpu_class_t::intel_sierra_k>();
    case cpu_class_t::intel_skylake_k: return forkunion_capabilities_of<cpu_class_t::intel_skylake_k>();
    case cpu_class_t::intel_icelake_k: return forkunion_capabilities_of<cpu_class_t::intel_icelake_k>();
    case cpu_class_t::amd_genoa_k: return forkunion_capabilities_of<cpu_class_t::amd_genoa_k>();
    case cpu_class_t::intel_sapphire_k: return forkunion_capabilities_of<cpu_class_t::intel_sapphire_k>();
    case cpu_class_t::intel_granite_k: return forkunion_capabilities_of<cpu_class_t::intel_granite_k>();
    case cpu_class_t::intel_diamond_k: return forkunion_capabilities_of<cpu_class_t::intel_diamond_k>();
    case cpu_class_t::aws_g2_k: return forkunion_capabilities_of<cpu_class_t::aws_g2_k>();
    case cpu_class_t::apple_m2_k: return forkunion_capabilities_of<cpu_class_t::apple_m2_k>();
    case cpu_class_t::apple_m4_k: return forkunion_capabilities_of<cpu_class_t::apple_m4_k>();
    case cpu_class_t::aws_g5_k: return forkunion_capabilities_of<cpu_class_t::aws_g5_k>();
    case cpu_class_t::nvidia_vera_k: return forkunion_capabilities_of<cpu_class_t::nvidia_vera_k>();
    case cpu_class_t::alibaba_c930_k: return forkunion_capabilities_of<cpu_class_t::alibaba_c930_k>();
    case cpu_class_t::wasm32_v128_k: return forkunion_capabilities_of<cpu_class_t::wasm32_v128_k>();
    case cpu_class_t::wasm64_extended_k: return forkunion_capabilities_of<cpu_class_t::wasm64_extended_k>();
    case cpu_class_t::monoversioned_k: return forkunion_capabilities_of<cpu_class_t::monoversioned_k>();
    default: return forkunion_capabilities_of<cpu_class_t::serial_k>();
    }
}

/** The C enum's class as the engine names it; @c uform_cpu_class_unknown_k derives nothing, so it
 *  names the serial floor. */
constexpr cpu_class_t cpu_class_of(uform_cpu_class_t cpu_class) noexcept {
    if (cpu_class == uform_cpu_class_unknown_k || cpu_class > uform_cpu_class_monoversioned_k)
        return cpu_class_t::serial_k;
    return static_cast<cpu_class_t>(cpu_class);
}

/** What one handle may use: the masks this machine runs when nothing is pinned, the pinned class's
 *  masks when it runs them. */
struct cpu_ceiling_t {
    nk_capability_t compute = nk_cap_serial_k;
    fu::capabilities_t sync = fu::capabilities_unknown_k;
};

/** Whether @p rung fits under @p ceiling: its kits and waiters need nothing the ceiling lacks. */
constexpr bool cpu_rung_fits(cpu_class_t rung, cpu_ceiling_t ceiling) noexcept {
    return (numkong_capabilities_of(rung) & ~ceiling.compute) == 0 &&
           (forkunion_capabilities_of(rung) & ~ceiling.sync) == 0;
}

/** The capabilities a device of @p device_class runs under @p pinned: its own class's when
 *  nothing is pinned, the pinned class's when the device holds it, and zero when it does not. */
constexpr nk_capability_t gpu_pin_capabilities(gpu_class_t device_class, uform_gpu_class_t pinned) noexcept {
    nk_capability_t const device = numkong_capabilities_of(device_class);
    if (pinned == uform_gpu_class_unknown_k) return device;
    if (pinned > uform_gpu_class_apple_metal10_k) return 0;
    nk_capability_t const wanted = numkong_capabilities_of(static_cast<gpu_class_t>(pinned));
    return (device & wanted) == wanted ? wanted : 0;
}

/** CPU classes newest first, the order a ladder tries them in. */
template <cpu_class_t... classes_>
struct cpu_ladder {};

/** GPU classes of one vendor newest first, the order its ladder tries them in. */
template <gpu_class_t... classes_>
struct gpu_ladder {};

/** This architecture's ladder, the serial floor last, or a monoversioned build's one class. */
#if UFORM_TARGET_MONOVERSIONED
using cpu_ladder_t = cpu_ladder<cpu_class_t::monoversioned_k>;
#elif NUMKONG_ARCH_X8664_
using cpu_ladder_t =
    cpu_ladder<cpu_class_t::intel_diamond_k, cpu_class_t::intel_granite_k, cpu_class_t::intel_sapphire_k,
               cpu_class_t::amd_genoa_k, cpu_class_t::intel_icelake_k, cpu_class_t::intel_skylake_k,
               cpu_class_t::intel_sierra_k, cpu_class_t::intel_haswell_k, cpu_class_t::serial_k>;
#elif NUMKONG_ARCH_ARM64_
using cpu_ladder_t = cpu_ladder<cpu_class_t::nvidia_vera_k, cpu_class_t::apple_m4_k, cpu_class_t::aws_g5_k,
                                cpu_class_t::apple_m2_k, cpu_class_t::aws_g2_k, cpu_class_t::serial_k>;
#elif NUMKONG_ARCH_RISCV64_
using cpu_ladder_t = cpu_ladder<cpu_class_t::alibaba_c930_k, cpu_class_t::serial_k>;
#elif NUMKONG_ARCH_WASM_
using cpu_ladder_t = cpu_ladder<cpu_class_t::wasm64_extended_k, cpu_class_t::wasm32_v128_k, cpu_class_t::serial_k>;
#else
using cpu_ladder_t = cpu_ladder<cpu_class_t::serial_k>;
#endif

/** The NVIDIA classes newest first, the vendor floor last, which runs every NVIDIA device without a
 *  class of its own on NumKong's generic CUDA kits. */
using nvidia_ladder_t =
    gpu_ladder<gpu_class_t::nvidia_blackwellrtx_k, gpu_class_t::nvidia_blackwell_k, gpu_class_t::nvidia_hopper_k,
               gpu_class_t::nvidia_ada_k, gpu_class_t::nvidia_ampere_k, gpu_class_t::unknown_k>;

/** The AMD classes newest first, the vendor floor last, on NumKong's generic ROCm kits. */
using amd_ladder_t =
    gpu_ladder<gpu_class_t::amd_cdna5_k, gpu_class_t::amd_cdna4_k, gpu_class_t::amd_cdna3_k, gpu_class_t::unknown_k>;

/** The rung a GPU class degrades to within its vendor, the vendor floor last. */
constexpr gpu_class_t parent_class_of(gpu_class_t gpu_class) noexcept {
    switch (gpu_class) {
    case gpu_class_t::nvidia_rubin_k: return gpu_class_t::nvidia_blackwell_k;
    case gpu_class_t::nvidia_blackwell_k:
    case gpu_class_t::nvidia_blackwellrtx_k:
    case gpu_class_t::nvidia_hopper_k: return gpu_class_t::nvidia_ada_k;
    case gpu_class_t::nvidia_ada_k: return gpu_class_t::nvidia_ampere_k;
    case gpu_class_t::amd_cdna5_k: return gpu_class_t::amd_cdna4_k;
    case gpu_class_t::amd_cdna4_k: return gpu_class_t::amd_cdna3_k;
    case gpu_class_t::apple_metal10_k: return gpu_class_t::apple_metal9_k;
    case gpu_class_t::apple_metal9_k: return gpu_class_t::apple_metal7_k;
    default: return gpu_class_t::unknown_k;
    }
}

/** Whether the build compiled the unit of @p cpu_class, as CMake decided and told the dispatch unit
 *  through @c UFORM_TARGET_<CLASS>; a ladder never picks a class without one. */
constexpr bool cpu_class_compiled(cpu_class_t cpu_class) noexcept {
    switch (cpu_class) {
#if UFORM_TARGET_SERIAL
    case cpu_class_t::serial_k: return true;
#endif
#if UFORM_TARGET_INTEL_HASWELL
    case cpu_class_t::intel_haswell_k: return true;
#endif
#if UFORM_TARGET_INTEL_SIERRA
    case cpu_class_t::intel_sierra_k: return true;
#endif
#if UFORM_TARGET_INTEL_SKYLAKE
    case cpu_class_t::intel_skylake_k: return true;
#endif
#if UFORM_TARGET_INTEL_ICELAKE
    case cpu_class_t::intel_icelake_k: return true;
#endif
#if UFORM_TARGET_AMD_GENOA
    case cpu_class_t::amd_genoa_k: return true;
#endif
#if UFORM_TARGET_INTEL_SAPPHIRE
    case cpu_class_t::intel_sapphire_k: return true;
#endif
#if UFORM_TARGET_INTEL_GRANITE
    case cpu_class_t::intel_granite_k: return true;
#endif
#if UFORM_TARGET_INTEL_DIAMOND
    case cpu_class_t::intel_diamond_k: return true;
#endif
#if UFORM_TARGET_AWS_G2
    case cpu_class_t::aws_g2_k: return true;
#endif
#if UFORM_TARGET_APPLE_M2
    case cpu_class_t::apple_m2_k: return true;
#endif
#if UFORM_TARGET_APPLE_M4
    case cpu_class_t::apple_m4_k: return true;
#endif
#if UFORM_TARGET_AWS_G5
    case cpu_class_t::aws_g5_k: return true;
#endif
#if UFORM_TARGET_NVIDIA_VERA
    case cpu_class_t::nvidia_vera_k: return true;
#endif
#if UFORM_TARGET_ALIBABA_C930
    case cpu_class_t::alibaba_c930_k: return true;
#endif
#if UFORM_TARGET_WASM32_V128
    case cpu_class_t::wasm32_v128_k: return true;
#endif
#if UFORM_TARGET_WASM64_EXTENDED
    case cpu_class_t::wasm64_extended_k: return true;
#endif
#if UFORM_TARGET_MONOVERSIONED
    case cpu_class_t::monoversioned_k: return true;
#endif
    default: return false;
    }
}

/** The ceiling @p pinned sets, the one caller of NumKong's and ForkUnion's CPU detection; a class
 *  this machine does not run, one past the enum, and a ceiling no compiled class fits, like one
 *  below a monoversioned build's masks, are refused as unsupported. */
inline expected<cpu_ceiling_t> pick_cpu_ceiling(uform_cpu_class_t pinned) noexcept {
    cpu_ceiling_t picked;
    if (nk_cpu_capabilities_enabled(&picked.compute) != nk_success_k) picked.compute = nk_cap_serial_k;
    picked.sync = fu::cpu_capabilities();
    if (pinned != uform_cpu_class_unknown_k) {
        if (pinned > uform_cpu_class_monoversioned_k)
            return expected<cpu_ceiling_t>::fail(status_t::unsupported_k, "no CPU class has that ordinal");
        cpu_class_t const pinned_class = cpu_class_of(pinned);
        cpu_ceiling_t const wanted {numkong_capabilities_of(pinned_class), forkunion_capabilities_of(pinned_class)};
        if ((picked.compute & wanted.compute) != wanted.compute || (picked.sync & wanted.sync) != wanted.sync)
            return expected<cpu_ceiling_t>::fail(status_t::unsupported_k, "the CPU does not run the pinned class");
        picked = wanted;
    }
    for (auto rung = uform_cpu_class_serial_k; rung <= uform_cpu_class_monoversioned_k;
         rung = static_cast<uform_cpu_class_t>(rung + 1))
        if (cpu_class_compiled(cpu_class_of(rung)) && cpu_rung_fits(cpu_class_of(rung), picked)) return picked;
    return expected<cpu_ceiling_t>::fail(status_t::unsupported_k, "the build compiled no class under that ceiling");
}

/** Whether the build compiled the unit of @p gpu_class, as CMake told the dispatch unit through
 *  @c UFORM_TARGET_<CLASS>; the vendor floor's unit comes with its vendor. */
constexpr bool gpu_class_compiled(gpu_class_t gpu_class) noexcept {
    switch (gpu_class) {
    case gpu_class_t::unknown_k: return true;
#if UFORM_TARGET_NVIDIA_AMPERE
    case gpu_class_t::nvidia_ampere_k: return true;
#endif
#if UFORM_TARGET_NVIDIA_ADA
    case gpu_class_t::nvidia_ada_k: return true;
#endif
#if UFORM_TARGET_NVIDIA_HOPPER
    case gpu_class_t::nvidia_hopper_k: return true;
#endif
#if UFORM_TARGET_NVIDIA_BLACKWELL
    case gpu_class_t::nvidia_blackwell_k: return true;
#endif
#if UFORM_TARGET_NVIDIA_BLACKWELLRTX
    case gpu_class_t::nvidia_blackwellrtx_k: return true;
#endif
#if UFORM_TARGET_AMD_CDNA3
    case gpu_class_t::amd_cdna3_k: return true;
#endif
#if UFORM_TARGET_AMD_CDNA4
    case gpu_class_t::amd_cdna4_k: return true;
#endif
#if UFORM_TARGET_AMD_CDNA5
    case gpu_class_t::amd_cdna5_k: return true;
#endif
    default: return false;
    }
}

/** Whether the build compiled the units of @p home 's vendor, as CMake told the dispatch unit
 *  through @c UFORM_ARCH_CUDA_ and @c UFORM_ARCH_ROCM_ ; every class of a compiled vendor has a
 *  unit, the ones advancing nothing past their parent answering no table. */
constexpr bool gpu_vendor_compiled([[maybe_unused]] nk_capability_t vendor) noexcept {
#if UFORM_ARCH_CUDA_
    if (vendor == nk_cap_cuda_k) return true;
#endif
#if UFORM_ARCH_ROCM_
    if (vendor == nk_cap_rocm_k) return true;
#endif
    return false;
}

/** The verbs one CPU class compiles in its class unit for the C layer, which is compiled once at
 *  the baseline: how its threads wait on a word another thread writes. */
struct host_verbs_t {

    /** Returns once @p word no longer holds @p seen, reading it through the class's atomic
     *  reference and waiting between reads through the class's waiter. */
    void (*wait)(u64_t &word, u64_t seen) noexcept;
};

/** The host verbs one class unit compiles, defined in @c encoder_specializations.hpp , which the
 *  class units alone include, so no body compiled for a class reaches a caller of another. */
template <cpu_class_t cpu_class_>
struct host_class_tables {
    static host_verbs_t const *find() noexcept;
    static void wait(u64_t &word, u64_t seen) noexcept;
};

/** The host verbs of the newest compiled class on @p ladder that fits under @p ceiling, one of
 *  which @ref pick_cpu_ceiling made sure of. */
template <cpu_class_t... classes_>
host_verbs_t const *pick_host_verbs(cpu_ladder<classes_...>, cpu_ceiling_t ceiling) noexcept {
    host_verbs_t const *picked = nullptr;
    auto rung = [&]<cpu_class_t class_>() noexcept {
        if constexpr (cpu_class_compiled(class_))
            if (!picked && cpu_rung_fits(class_, ceiling)) picked = host_class_tables<class_>::find();
    };
    (rung.template operator()<classes_>(), ...);
    return picked;
}

#pragma endregion Classes

#pragma region Outcomes

/** The C status a failure of the engine's own vocabulary crosses into, whose values it shares. */
[[nodiscard]] inline uform_status_t status_of(status_t status) noexcept { return static_cast<uform_status_t>(status); }

/** The C status a NumKong failure crosses into; whatever a device answers beyond shapes and memory
 *  is the device's. */
[[nodiscard]] inline uform_status_t status_of(nk_status_t status) noexcept {
    switch (status) {
    case nk_success_k: return uform_status_success_k;
    case nk_bad_alloc_k: return uform_status_out_of_memory_k;
    case nk_unexpected_dimensions_k: return uform_status_shape_mismatch_k;
    case nk_misaligned_k: return uform_status_unsupported_layout_k;
    case nk_missing_kernel_k: return uform_status_unsupported_k;
    default: return uform_status_device_k;
    }
}

/** Marks @p outcome as running; no waiter watches for this store, so it is relaxed. The C layer is
 *  compiled once at the baseline, so its shared words go through the standard reference, which
 *  every class's waiter reads. */
inline void propagate_pending(uform_outcome_t *outcome) noexcept {
    outcome->message = nullptr;
    fu::standard_atomic_ref<uform_status_t> {outcome->status}.store(uform_status_deferred_k, std::memory_order_relaxed);
}

/** The one final write to @p outcome: the message, then the status every waiter acquires. The call
 *  touches nothing it was passed after this, and the message is null exactly on success. */
inline bool propagate_error(uform_status_t status, uform_outcome_t *outcome,
                            char const *message) noexcept { // vendor-boundary: the C outcome's static message
    bool const succeeded = status == uform_status_success_k;
    outcome->message = succeeded ? nullptr : message;
    fu::standard_atomic_ref<uform_status_t> {outcome->status}.store(status, std::memory_order_release);
    return succeeded;
}

#pragma endregion Outcomes

#pragma region Progress

/** Whether the call a progress handler watches carries on or unwinds. */
enum class continuation_t : u8_t {
    continue_k,
    cancel_k,
};

/** A half-open range along a call's batch axis that the engine takes whole. */
struct slice_t {
    std::size_t begin = 0;
    std::size_t end = 0;
};

/**
 *  @brief One call's reports: the C handler, serialized for the executor's threads, which the
 *      contract promises it never sees concurrently, thinned to the caller's steps, and the slices
 *      a batch verb hands its engine whole.
 *
 *  Without a handler a batch is one slice and nothing is reported; with one it splits into
 *  @c progress.steps nearly equal slices, ten when zero. A verb counting its own units, a load's
 *  tensors, forwards each tick and only the ones crossing into the next step reach the handler.
 */
class progress_ticks_t {
  public:
    progress_ticks_t(uform_progress_t progress, std::size_t total) noexcept : progress_(progress), total_(total) {}
    progress_ticks_t(progress_ticks_t const &) = delete;
    progress_ticks_t &operator=(progress_ticks_t const &) = delete;

    /** Walks the slices of a batch in order; slice @c i starts at @c ⌈i · total/slices⌉ , so
     *  lengths differ by one at most. */
    class iterator {
      public:
        iterator(std::size_t index, std::size_t slices, std::size_t total) noexcept
            : index_(index), slices_(slices), total_(total) {}
        slice_t operator*() const noexcept { return {boundary_(index_), boundary_(index_ + 1)}; }
        iterator &operator++() noexcept { return ++index_, *this; }
        bool operator!=(iterator const &other) const noexcept { return index_ != other.index_; }

      private:
        [[nodiscard]] std::size_t boundary_(std::size_t index) const noexcept {
            return divide_round_up(index * total_, slices_);
        }

        std::size_t index_;
        std::size_t slices_;
        std::size_t total_;
    };

    /** Slices the batch splits into: one per step with a handler, the whole batch without,
     *  none when empty. */
    [[nodiscard]] std::size_t slices() const noexcept {
        if (!total_) return 0;
        return progress_.callback ? steps_(total_) : 1;
    }

    [[nodiscard]] iterator begin() const noexcept { return {0, slices(), total_}; }
    [[nodiscard]] iterator end() const noexcept { return {slices(), slices(), total_}; }

    /** Reports @p completed of the batch's units, reaching the handler only when it closes
     *  a new step. */
    continuation_t operator()(std::size_t completed) noexcept { return (*this)(completed, total_); }

    /** The sibling for verbs that know their own population, like a load counting its tensors. A
     *  slice end @c ⌈i · population/steps⌉ closes exactly step @c i . */
    continuation_t operator()(std::size_t completed, std::size_t population) noexcept {
        if (!progress_.callback) return continuation_t::continue_k;
        st::unique_lock<fu::spin_mutex_t> lock(mutex_);
        std::size_t const steps = steps_(population);
        std::size_t const step = completed >= population ? steps : completed * steps / population;
        if (step <= closed_) return continuation_t::continue_k;
        closed_ = step;
        return progress_.callback(completed, population, progress_.state) == uform_cancel_k
                   ? continuation_t::cancel_k
                   : continuation_t::continue_k;
    }

  private:

    /** The caller's steps, ten when zero, and never more than @p population units can close. */
    [[nodiscard]] std::size_t steps_(std::size_t population) const noexcept {
        std::size_t const asked = progress_.steps ? progress_.steps : default_progress_steps_k;
        return asked < population ? asked : (population ? population : 1);
    }

    uform_progress_t progress_;
    std::size_t total_;

    /** The last step a forwarded tick closed, guarded by @ref mutex_ like the handler itself. */
    std::size_t closed_ = 0;
    fu::spin_mutex_t mutex_;
};

#pragma endregion Progress

#pragma region Memory

/**
 *  @brief Memory that one call at a time rewinds and refills through one NumKong allocation policy:
 *      chunks that double and never move, so a steady loop allocates once and every view a call
 *      hands out stays put.
 *
 *  Chunks are returned through the policy on the stream it was given, so a device arena dies before
 *  its stream. Movable so a per-device table can hold it, never move-assigned, since that would
 *  leak the destination's chunks.
 */
class chunked_arena_t {
  public:
    chunked_arena_t(nk_allocator_t policy, nk_stream_t stream, std::size_t limit) noexcept
        : policy_(policy), stream_(stream), limit_(limit) {}
    chunked_arena_t(chunked_arena_t &&) noexcept = default;
    chunked_arena_t &operator=(chunked_arena_t &&) = delete;
    ~chunked_arena_t() noexcept {
        for (bytes_span_t const chunk : chunks_) policy_.free(chunk.data(), chunk.size(), policy_.handle, stream_);
    }

    /** The stream its chunks are allocated and freed on; null for host pages. */
    [[nodiscard]] nk_stream_t stream() const noexcept { return stream_; }

    /** Forgets every allocation, keeping the chunks, which invalidates every view the previous
     *  call handed out. */
    void rewind() noexcept { current_ = 0, used_ = 0; }

    /** Whether @p address lies in memory this arena handed out, which its next rewind reuses. */
    [[nodiscard]] bool contains(void const *address) const noexcept {
        for (bytes_span_t const chunk : chunks_)
            if (address >= chunk.data() && address < chunk.data() + chunk.size()) return true;
        return false;
    }

    /** Hands out @p bytes aligned for any kernel, growing by a chunk as large as everything held so
     *  far when full, or null when the policy or the limit refuses. */
    [[nodiscard]] void *allocate(std::size_t bytes) noexcept {
        std::size_t const aligned = round_up_to_multiple(bytes, nk_default_alignment_k);
        if (aligned < bytes) return nullptr; // saturated within an alignment of the address space
        for (; current_ != chunks_.size(); ++current_, used_ = 0)
            if (chunks_[current_].size() - used_ >= aligned) {
                b8_t *const landed = chunks_[current_].data() + used_;
                used_ += aligned;
                return landed;
            }
        std::size_t held = 0;
        for (bytes_span_t const chunk : chunks_) held += chunk.size();
        std::size_t const capacity = held > aligned ? held : aligned;
        if (limit_ && capacity > limit_ - held) return nullptr;
        if (!policy_.allocate) return nullptr;
        b8_t *const allocated = static_cast<b8_t *>(policy_.allocate(capacity, policy_.handle, stream_));
        if (!allocated) return nullptr;
        if (st::failed(chunks_.push_back(bytes_span_t {allocated, capacity})))
            return policy_.free(allocated, capacity, policy_.handle, stream_), nullptr;
        current_ = chunks_.size() - 1, used_ = aligned;
        return allocated;
    }

    /** Hands out @p count values of @p value_type_, or null when they cannot be allocated. */
    template <typename value_type_>
    [[nodiscard]] value_type_ *allocate(std::size_t count) noexcept {
        static_assert(std::is_trivially_copyable_v<value_type_>, "Arenas hold plain outputs only");
        if (count > (std::numeric_limits<std::size_t>::max)() / sizeof(value_type_)) return nullptr;
        return static_cast<value_type_ *>(allocate(count * sizeof(value_type_)));
    }

  private:
    nk_allocator_t policy_;
    nk_stream_t stream_;

    /** Bytes the chunks may span together, or zero for no ceiling. */
    std::size_t limit_;
    st::basic_vector<bytes_span_t> chunks_;

    /** The chunk being filled, and the bytes of it already handed out since the last rewind. */
    std::size_t current_ = 0;
    std::size_t used_ = 0;
};

/** NumKong's aligned heap, the policy every host arena grows through. */
[[nodiscard]] inline nk_allocator_t host_pages_policy() noexcept {
    nk_allocator_t policy {};
    [[maybe_unused]] nk_status_t const status = nk_allocator_init_heap(&policy, nk_default_alignment_k);
    return policy;
}

/** Host pages, which serve calls on the calling thread or any RAM executor and may outlive the
 *  executor that made them. */
struct host_arena_t {
    chunked_arena_t chunks {host_pages_policy(), nullptr, 0};
};

/** The memory of one GPU, grown through NumKong's device allocator on one stream of the executor
 *  that made it. */
struct device_arena_t {
    chunked_arena_t chunks;
};

/** The chunks either residency's arena grows. */
[[nodiscard]] inline chunked_arena_t &chunks_of(std::variant<host_arena_t, device_arena_t> &memory) noexcept {
    if (host_arena_t *const host = std::get_if<host_arena_t>(&memory)) return host->chunks;
    return std::get<device_arena_t>(memory).chunks;
}

#pragma endregion Memory

#pragma region Executors

/** A device stream an executor opened and closes, its vendor named by the device it
 *  was opened on. */
using owned_stream_t = nk::stream_t;

/** The host threads of a RAM executor, spawned here or lent by another library, and where its
 *  models place their bytes. */
struct ram_executor_t {
    std::variant<executor_t, lent_pool_t> pool;

    /** What the threads may run: the machine's masks, or the pinned class's. */
    cpu_ceiling_t ceiling;

    /** The NUMA nodes the executor's models place their bytes in. */
    st::basic_vector<i32_t> memory_domains;
};

/** One GPU of a VRAM executor: its stream, and the scratch its calls rewind. */
struct vram_device_t {

    /** The UForm ordinal, the device's position in the @c uform_gpu_devices listing, NVIDIA
     *  devices first, which @c uform_vram_options_t names and a model's replicas match by. */
    i32_t ordinal = 0;

    /** The vendor's own ordinal, as @c cuDeviceGet or @c hipSetDevice takes it. */
    i32_t vendor_ordinal = 0;

    /** What the device runs, whose group names the vendor every NumKong call dispatches on. */
    nk_capability_t capabilities = 0;

    /** The class the device dispatches as: its driver's report, or the pinned class it holds. */
    gpu_class_t gpu_class = gpu_class_t::unknown_k;

    /** Declared before the scratch, so the scratch returns its chunks while the
     *  stream still lives. */
    owned_stream_t stream;

    /** Activations, grown to a high-water mark under the executor's @c scratch_limit. */
    device_arena_t scratch;

    /** Pinned host pages both the host and the device address, where a call packs what
     *  the device reads. */
    chunked_arena_t transfers;
};

/** The devices of a VRAM executor and its decoders, a caller-inclusive pool whose thread 0 is the
 *  caller: it stages and feeds the devices and waits on them, while the rest decode for them. */
struct vram_executor_t {
    per_gpu_array<vram_device_t> devices;
    executor_t decoders;

    /** Host pages for what only the decoders read: the plans and every thread's decode window. */
    host_arena_t staging;
};

/** The right to dispatch on @p pool , owned or lent. */
[[nodiscard]] inline pool_view_t view_of(std::variant<executor_t, lent_pool_t> const &pool) noexcept {
    return std::visit([](auto const &alternative) noexcept { return alternative.view(); }, pool);
}

#pragma endregion Executors

#pragma region Models

/** Where an encoder's tower lives; there is no NVMe home for models. */
enum class encoder_home_t : u32_t {
    ram_k,
    vram_nvidia_k,
    vram_amd_k,
};

/** The executor each home's verbs take: a host tower's thread pool, a device tower's stream. */
template <encoder_home_t home_>
struct encoder_home_executor {
    using type = nk_stream_t;
};
template <>
struct encoder_home_executor<encoder_home_t::ram_k> {
    using type = pool_view_t;
};

/** The verbs one encoder variant compiles in its class unit, defined in @c encoder_punned.hpp . */
template <encoder_home_t home_>
struct encoder_verbs;

/** The variant a load picked, as its home's verb table, which alone casts each tower back
 *  to its type. */
using encoder_home_tables_t =
    std::variant<encoder_verbs<encoder_home_t::ram_k> const *, encoder_verbs<encoder_home_t::vram_nvidia_k> const *,
                 encoder_verbs<encoder_home_t::vram_amd_k> const *>;

/** One copy of a model's tower on one GPU, matched to an executor's device by its UForm ordinal. */
struct vram_replica_t {
    i32_t ordinal = 0;
    void *tower = nullptr;
};

#pragma endregion Models

#pragma region Deferred Calls

/** Every argument structure that names an executor, so a delegated call runs a copy its caller's
 *  scope cannot end; whatever the copy points to stays the caller's until final. */
using deferred_arguments_t = std::variant<                                                        //
    std::monostate,                                                                               //
    uform_model_load_ram_t, uform_model_load_ram_buffer_t,                                        //
    uform_model_load_vram_t, uform_model_load_vram_buffer_t, uform_model_describe_t,              //
    uform_text_encode_t, uform_text_choose_t, uform_image_encode_t, uform_image_encode_pixels_t,  //
    uform_video_encode_t, uform_audio_encode_t, uform_audio_encode_samples_t,                     //
    uform_mesh_bootstrap_t, uform_trainer_init_vram_t, uform_trainer_step_t, uform_trainer_save_t //
    >;

/** Whether an executor runs no call, or one call from its claim to its end. */
enum class call_claim_t : u32_t {
    free_k,
    held_k,
};

/** The handles one call names, besides its executor, whose frees wait for the call to end. */
inline constexpr std::size_t named_handles_max_k = 4;

/** The one call that holds an executor, from its claim to its end. */
struct holding_call_t {

    /** Taken by an exchange at the entry, released by a store just before the status. */
    call_claim_t claim = call_claim_t::free_k;

    /** The holding call's arguments; meeting them again, the entry runs the verb's body. */
    void const *arguments = nullptr;

    /** A delegated call's copy of its arguments, which outlives the caller's structure. */
    deferred_arguments_t delegated_arguments;

    /** The caller's outcome, published last. */
    uform_outcome_t *outcome = nullptr;

    /** The status the verb body recorded, which @ref end_call publishes. */
    uform_outcome_t recorded {uform_status_internal_k, "The call finished without reporting a status"};

    /** The counts of the handles the call names, raised at the entry, lowered at its end. */
    u64_t *named_calls[named_handles_max_k] {};

    /** The body pool thread 0 runs for a delegated call. */
    void (*delegated_body)(void *) = nullptr;

    /** The kernel ID of the thread running the body, or zero before thread 0 takes it up. */
    u64_t body_thread = 0;
};

/** Lets go of @p arena when the call whose published outcome is @p call holds it. */
void release_arena(uform_arena_t arena, uform_outcome_t const *call) noexcept;

/** Takes @p executor for one call; false while another call holds it. */
bool take_call_claim(uform_executor_handle_t &executor) noexcept;

/** Whether a caller-exclusive pool backs @p executor , running deferred calls on its threads. */
bool delegates(uform_executor_handle_t &executor) noexcept;

/** Starts the held call on the pool's thread 0, which runs @p body and ends the call. */
uform_status_t delegate_call(uform_executor_handle_t &executor, void (*body)(void *)) noexcept;

/** Keeps @p recorded , the body's status, for @ref end_call to publish. */
void record_status(uform_executor_t executor, uform_outcome_t const &recorded) noexcept;

/** Ends the call holding @p executor : the named counts, the claim, then the status. */
void end_call(uform_executor_handle_t &executor) noexcept;

/** Returns once no executor call names the handle whose count is @p named_calls . */
void wait_named_calls(u64_t &named_calls) noexcept;

/** Whether the calling thread runs the body of the call holding @p executor . */
bool runs_holding_call(uform_executor_handle_t &executor) noexcept;

/** Returns once no call holds @p executor and its pool finished the last one. */
void wait_while_claimed(uform_executor_handle_t &executor) noexcept;

/**
 *  @brief Every verb's entry: claims the call's executor and runs its body, answering true once the
 *      call needs nothing more from the verb.
 *
 *  A blocking call, or one on a pool that needs its caller, runs its body here; a deferred one on a
 *  caller-exclusive pool runs it on the pool's thread 0. The body is the same verb meeting its own
 *  arguments again, which answers false so the verb runs. A call with no executor answers false at
 *  once and is final when it returns.
 */
template <auto verb_, typename arguments_type_>
bool submitted(arguments_type_ *arguments) noexcept {
    // Dependent, so the handle's members resolve at instantiation
    auto *const executor = arguments->executor;
    if (!executor || arguments == executor->holding_call.arguments) return false;
    if (!take_call_claim(*executor))
        return propagate_error(uform_status_contended_k, arguments->outcome, "Another call holds the executor"), true;
    auto &call = executor->holding_call;
    call.outcome = arguments->outcome;
    call.recorded = {uform_status_internal_k, "The call finished without reporting a status"};
    std::size_t named = 0;
    auto const name = [&](auto handle) noexcept {
        // Outputs for handles the call creates, and handles this build never defines, hold no count
        if constexpr (requires { handle->named_calls; })
            if (handle) {
                fu::standard_atomic_ref<u64_t> {handle->named_calls}.fetch_add(1, std::memory_order_relaxed);
                call.named_calls[named++] = &handle->named_calls;
            }
    };
    if constexpr (requires { arguments->model; }) name(arguments->model);
    if constexpr (requires { arguments->arena; }) name(arguments->arena);
    if constexpr (requires { arguments->mesh; }) name(arguments->mesh);
    if constexpr (requires { arguments->trainer; }) name(arguments->trainer);
    propagate_pending(arguments->outcome); // before the body or the dispatch, never after
    if (arguments->completion != uform_completion_deferred_k || !delegates(*executor)) {
        call.arguments = arguments;
        verb_(arguments);
        return end_call(*executor), true;
    }
    call.arguments = &call.delegated_arguments.template emplace<arguments_type_>(*arguments);
    using executor_handle_t = std::remove_pointer_t<decltype(arguments->executor)>;
    uform_status_t const delegated = delegate_call(*executor, [](void *context) noexcept {
        auto &held = static_cast<executor_handle_t *>(context)->holding_call;
        verb_(std::get_if<arguments_type_>(&held.delegated_arguments));
    });
    if (delegated != uform_status_success_k)
        call.recorded = {delegated, "The lent pool runs its lender's dispatch"}, end_call(*executor);
    return true;
}

/**
 *  @brief A verb's outcome while it runs: every step records into a private one, and the end lets
 *      go of the arena before the status leaves.
 *
 *  Declared first in a verb, it unwinds last. Without an executor it publishes the status itself.
 *  With one it hands the status to @ref end_call, which publishes it once the executor is free, so
 *  a waiter that sees it final may reuse the arena and the executor at once. A path that forgot to
 *  report finishes @c uform_status_internal_k rather than hanging its waiters.
 */
class call_outcome_t {
  public:
    /** For the verbs that take no executor. */
    explicit call_outcome_t(uform_outcome_t *published) noexcept : published_(published) {
        propagate_pending(published_);
    }

    /** For a verb's call, whose executor, when it names one, publishes the status. */
    template <typename arguments_type_>
    explicit call_outcome_t(arguments_type_ const &arguments) noexcept
        : published_(arguments.outcome), executor_(arguments.executor) {
        if constexpr (requires { arguments.arena; }) arena_ = arguments.arena;
        if (!executor_) propagate_pending(published_);
    }

    ~call_outcome_t() noexcept {
        release_arena(arena_, published_);
        if (executor_) record_status(executor_, recorded_);
        else propagate_error(recorded_.status, published_, recorded_.message);
    }
    call_outcome_t(call_outcome_t const &) = delete;
    call_outcome_t &operator=(call_outcome_t const &) = delete;

    /** The private record every step reports into. */
    operator uform_outcome_t *() noexcept { return &recorded_; }

    /** The caller's outcome, which identifies the call holding an arena. */
    [[nodiscard]] uform_outcome_t const *published() const noexcept { return published_; }

  private:
    uform_outcome_t *published_;
    uform_executor_t executor_ = nullptr;
    uform_arena_t arena_ = nullptr;
    uform_outcome_t recorded_ {uform_status_internal_k, "The call finished without reporting a status"};
};

/**
 *  @brief Holds @p arena for the call @p outcome tracks and rewinds it, freeing the
 *      previous call's outputs.
 *  @return The arena, or null once @p outcome records why not.
 *
 *  Refuses a null arena, host pages named by a VRAM executor, device memory named by another
 *  executor than the one that made it, an arena another call holds, and an input of @p inputs
 *  still inside it.
 */
uform_arena_t claim_arena(uform_arena_t arena, uform_executor_t executor, std::initializer_list<void const *> inputs,
                          call_outcome_t &outcome) noexcept;

#pragma endregion Deferred Calls

} // namespace unum::uform

namespace uf = unum::uform;

/** The state behind @c uform_executor_t: the one call it holds and one residency's executor; the
 *  variant is the kind tag, so no second word can disagree. */
struct uform_executor_handle_t {

    /** The one call that holds the executor, from its claim to its end. */
    uf::holding_call_t holding_call;

    /** How its callers wait on what its threads write: the verbs of the CPU class they run, a
     *  static table that outlives the executor. */
    uf::host_verbs_t const *host_verbs = nullptr;

    std::variant<uf::ram_executor_t, uf::vram_executor_t> executor;
};

/** What @c uform_arena_t owns: memory that one call at a time rewinds and refills, as host pages or
 *  on one device, and the call holding it. */
struct uform_arena_handle_t {

    /** Fixed at @c uform_arena_init by the executor that made it. */
    std::variant<uf::host_arena_t, uf::device_arena_t> memory;

    /** The executor whose first stream the device memory rides, the only one whose calls it serves;
     *  null for host pages, which serve any call computing on the host. */
    uform_executor_t executor = nullptr;

    /** The published outcome of the call holding the arena, from its claim until just before its
     *  final store, read through the standard reference since a call may end on a pool thread. */
    uform_outcome_t const *holder = nullptr;

    /** Executor calls naming the arena that have not ended, which its free waits out. */
    uf::u64_t named_calls = 0;
};

/** What @c uform_model_t owns: the geometry its checkpoint carried, the precision its blocks run
 *  in, and its packed towers; the variant is the residency, so the checkpoint itself is closed once
 *  the load completes. */
struct uform_model_handle_t {
    uf::encoder::geometry_t geometry;
    uf::encoder::precision_t precision = uf::encoder::precision_t::bf16_k;

    /** Bytes every tower holds together, by the extents and dtypes of what each packs. */
    std::size_t memory_usage = 0;

    /** Each question type's calibrated inverse temperature, in host memory where a choice's softmax
     *  runs; empty without the byte decoder. */
    uf::st::basic_vector<uf::f32_t> choice_inverse_temperatures;

    /** The variant picked once at load: its home's verb table, which alone casts
     *  each tower back. */
    uf::encoder_home_tables_t tables;

    /** The one host tower, serving any RAM executor, or one replica per GPU the load named, every
     *  one of the picked class. */
    std::variant<void *, uf::per_gpu_array<uf::vram_replica_t>> towers;

    uform_model_handle_t() noexcept = default;
    uform_model_handle_t(uform_model_handle_t const &) = delete;
    uform_model_handle_t &operator=(uform_model_handle_t const &) = delete;

    /** Executor calls naming the model that have not ended, which its free waits out. */
    uf::u64_t named_calls = 0;

    /** Destroys every tower through the verbs that made it; defined beside them in
     *  @c model.cpp . */
    ~uform_model_handle_t() noexcept;
};
