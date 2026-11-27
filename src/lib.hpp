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

#include <array>            // `std::array` - the handles a call names
#include <initializer_list> // `std::initializer_list` - the inputs a call checks against its arena
#include <limits>           // `std::numeric_limits` - the overflow guard of a typed allocation
#include <type_traits>      // `std::is_trivially_copyable_v`
#include <variant>          // `std::variant` - the executor's kind, the arena's residency, a deferred call

#include <forkunion/atomics.hpp>      // `fu::standard_atomic_ref` & kin - the classes' atomic references
#include <forkunion/capabilities.hpp> // `fu::x86_pause_t`, `fu::arm64_wfet_t` & waiter kin
#include <forkunion/forkunion.hpp>    // `fu::pool_t`, `fu::topology_t`
#include <forkunion/locks.hpp>        // `fu::spin_mutex_t` - the executors' holds
#include <forkunion/topology.hpp>     // `fu::current_thread_id` - the thread running an executor's call
#include <forkunion/types.hpp>        // `fu::wait_capped_k`
#include <numkong/capabilities.h>     // `nk_capability_t`, `nk_stream_t`

#include "uform/uform.h" // every argument structure

#include "uform/utilities/array.hpp"       // `per_gpu_array`
#include "uform/utilities/ram_device.hpp"  // `status_from_`, `no_memory_domain_k`, `spawn_pool`
#include "uform/utilities/types.hpp"       // `status_t`, the integer aliases, `sz::string_view_t`, `st::basic_vector`
#include "uform/utilities/vram_device.hpp" // `gpu_drivers_t`, `gpu_executor_t`, `spawn_gpu_executor`
#include "uform/encoder/tower.hpp"         // `geometry_t`, `precision_t`, `outputs_t` - what a model's towers take

namespace unum::uform {

namespace fu = ashvardanian::forkunion;
namespace nk = ashvardanian::numkong;
namespace sz = ashvardanian::stringzilla;
namespace st = ashvardanian::smashtable;

/** Reports a progress handler hears at most when its call names zero steps. */
inline constexpr std::size_t default_progress_steps_k = 10;

/** A byte run of the C interface, a path or a spelling, as a string view valid while the caller
 *  keeps it. */
[[nodiscard]] inline sz::string_view_t text_of(uform_bytes_view_t bytes) noexcept {
    return {nk::as_values<char>(std::as_bytes(std::span<util::u8_t const> {bytes.data, bytes.length})).data(),
            bytes.length};
}

/** Whether the @p count + 1 offsets of a caller's tape or edge list never step back, since an
 *  entry whose end precedes its start would wrap its length; null offsets name no entries. */
template <typename view_type_>
[[nodiscard]] bool offsets_ascend(view_type_ const& view, std::size_t count) noexcept {
    for (std::size_t item = 0; view.offsets && item != count; ++item)
        if (view.offsets[item + 1] < view.offsets[item]) return false;
    return true;
}

/** The @p count entries of a C tape as the engine's typed tape, whose elements reach the last
 *  offset; null offsets make an empty tape, and null data no elements. */
template <typename element_type_>
[[nodiscard]] util::tape_view<element_type_> tape_of(uform_tape_view_t tape, std::size_t count) noexcept {
    if (!tape.offsets) return {};
    return {{static_cast<element_type_*>(tape.data), tape.data ? tape.offsets[count] : 0}, {tape.offsets, count + 1}};
}

#pragma region Classes

/** The CPU classes of @c uform_cpu_class_t as the engine names them, with the same values. */
enum class cpu_class_t : util::u32_t {
    unknown_k = uform_cpu_class_unknown_k,
    serial_k = uform_cpu_class_serial_k,
    monoversioned_k = uform_cpu_class_monoversioned_k,
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
    ibm_power10_k = uform_cpu_class_ibm_power10_k,
    wasm32_v128_k = uform_cpu_class_wasm32_v128_k,
    wasm64_extended_k = uform_cpu_class_wasm64_extended_k,
};

static_assert(uform_cpu_class_serial_k == 1 && uform_cpu_class_monoversioned_k == 2 &&
              uform_cpu_class_ibm_power10_k == 18);

/** The GPU classes of @c uform_gpu_class_t as the engine names them, with the same values. */
enum class gpu_class_t : util::u32_t {
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

/** Every CPU class the C enum names, in value order - the one table pins and walks check against,
 *  so a value naming no class this build has is never one. */
inline constexpr uform_cpu_class_t cpu_classes_k[] {
    uform_cpu_class_serial_k,          uform_cpu_class_monoversioned_k,  uform_cpu_class_intel_haswell_k,
    uform_cpu_class_intel_sierra_k,    uform_cpu_class_intel_skylake_k,  uform_cpu_class_intel_icelake_k,
    uform_cpu_class_amd_genoa_k,       uform_cpu_class_intel_sapphire_k, uform_cpu_class_intel_granite_k,
    uform_cpu_class_intel_diamond_k,   uform_cpu_class_aws_g2_k,         uform_cpu_class_apple_m2_k,
    uform_cpu_class_apple_m4_k,        uform_cpu_class_aws_g5_k,         uform_cpu_class_nvidia_vera_k,
    uform_cpu_class_alibaba_c930_k,    uform_cpu_class_ibm_power10_k,    uform_cpu_class_wasm32_v128_k,
    uform_cpu_class_wasm64_extended_k,
};

/** Every GPU class the C enum names, in value order. */
inline constexpr uform_gpu_class_t gpu_classes_k[] {
    uform_gpu_class_nvidia_ampere_k,    uform_gpu_class_nvidia_ada_k,          uform_gpu_class_nvidia_hopper_k,
    uform_gpu_class_nvidia_blackwell_k, uform_gpu_class_nvidia_blackwellrtx_k, uform_gpu_class_nvidia_rubin_k,
    uform_gpu_class_amd_cdna3_k,        uform_gpu_class_amd_cdna4_k,           uform_gpu_class_amd_cdna5_k,
    uform_gpu_class_apple_metal7_k,     uform_gpu_class_apple_metal9_k,        uform_gpu_class_apple_metal10_k,
};

/** Whether @p table lists @p value. */
template <typename value_type_, std::size_t count_>
constexpr bool listed_in(value_type_ const (&table)[count_], value_type_ value) noexcept {
    for (value_type_ const listed : table)
        if (listed == value) return true;
    return false;
}

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
    case cpu_class_t::ibm_power10_k: return nk_cap_serial_k | nk_cap_powervsx_k | nk_cap_powermma_k;
    case cpu_class_t::wasm32_v128_k: return nk_cap_serial_k | nk_cap_v128_k;
    case cpu_class_t::wasm64_extended_k:
        return numkong_capabilities_of(cpu_class_t::wasm32_v128_k) | nk_cap_v128relaxed_k;
    case cpu_class_t::monoversioned_k: return nk_capabilities_compiled_cpu_mask();
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

/** The wait instruction one CPU class pairs: x86 spins PAUSE, the WAITPKG Xeons and the monitored
 *  Arm & RISC-V classes arm TPAUSE, WFE & WRS, WebAssembly threads sleep in @c memory.atomic.wait,
 *  everything else yields - as does any class whose @c FORKUNION_TARGET_<BIT> is off in the probe
 *  list. WFE sleeps on uncapped waits and yields on capped ones, so it needs no bit to confirm; the
 *  timed WFET only a monoversioned build reaches, where a preset vouches for it. */
template <cpu_class_t cpu_class_>
struct waiter_for_cpu_class {
    using type = fu::standard_yield_t;
};
#if FORKUNION_TARGET_X8664_PAUSE
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
#if FORKUNION_TARGET_X8664_TPAUSE
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
#if FORKUNION_TARGET_PPC64_PAUSE
template <>
struct waiter_for_cpu_class<cpu_class_t::ibm_power10_k> {
    using type = fu::ppc64_pause_t;
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
#if FORKUNION_TARGET_X8664_CMPCCXADD
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
#if FORKUNION_TARGET_PPC64_AMO
template <>
struct atomic_ref_for_cpu_class<cpu_class_t::ibm_power10_k> {
    template <typename value_type_>
    using type = fu::ppc64_amo_atomic_ref<value_type_>;
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

/** The spin hints every core of an architecture supports - PAUSE, YIELD, Zihintpause and Power's
 *  priority nops: the runtime confirms nothing about them, and a class that differs from another by
 *  one of them alone earns no rung. */
inline constexpr fu::capabilities_t baseline_hints_k = fu::capability_x86_pause_k | fu::capability_arm64_yield_k |
                                                       fu::capability_riscv64_pause_k | fu::capability_ppc64_yield_k;

/** Lowers one CPU class onto ForkUnion's mask - the bits the runtime must confirm before the
 *  class's waiter and atomic reference are trusted, read off the two types themselves. */
template <cpu_class_t cpu_class_>
constexpr fu::capabilities_t forkunion_capabilities_of() noexcept {
    constexpr fu::capabilities_t waits_k = waiter_of<cpu_class_>::capability_k;
    constexpr fu::capabilities_t atomics_k =
        atomic_ref_for_cpu_class<cpu_class_>::template type<util::u32_t>::capabilities_k;
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
    case cpu_class_t::ibm_power10_k: return forkunion_capabilities_of<cpu_class_t::ibm_power10_k>();
    case cpu_class_t::wasm32_v128_k: return forkunion_capabilities_of<cpu_class_t::wasm32_v128_k>();
    case cpu_class_t::wasm64_extended_k: return forkunion_capabilities_of<cpu_class_t::wasm64_extended_k>();
    case cpu_class_t::monoversioned_k: return forkunion_capabilities_of<cpu_class_t::monoversioned_k>();
    default: return forkunion_capabilities_of<cpu_class_t::serial_k>();
    }
}

/** The C enum's class as the engine names it; @c uform_cpu_class_unknown_k derives nothing, so
 *  it names the serial floor. */
constexpr cpu_class_t cpu_class_of(uform_cpu_class_t cpu_class) noexcept {
    if (!listed_in(cpu_classes_k, cpu_class)) return cpu_class_t::serial_k;
    return static_cast<cpu_class_t>(cpu_class);
}

/** What one handle may use: the masks this machine runs, intersected with its ceiling class's. */
struct cpu_ceiling_t {
    nk_capability_t compute = nk_cap_serial_k;
    fu::capabilities_t sync = fu::capabilities_unknown_k;
};

/** Whether @p rung fits under @p ceiling: its kits and waiters need nothing the ceiling lacks. */
constexpr bool cpu_rung_fits(cpu_class_t rung, cpu_ceiling_t ceiling) noexcept {
    return (numkong_capabilities_of(rung) & ~ceiling.compute) == 0 &&
           (forkunion_capabilities_of(rung) & ~ceiling.sync) == 0;
}

/** The allow-mask a pool spawns under @p ceiling : every thread and memory facility, and only the
 *  instruction bits the ceiling holds. */
constexpr fu_capabilities_t forkunion_allowed_of(cpu_ceiling_t ceiling) noexcept {
    return static_cast<fu_capabilities_t>((fu_capabilities_all_k & ~fu_capability_any_instruction_k) | ceiling.sync |
                                          baseline_hints_k);
}

/** CPU classes newest first, the order a ladder tries them in. */
template <cpu_class_t... classes_>
struct cpu_ladder {};

/** GPU classes of one vendor newest first, the order its ladder tries them in. */
template <gpu_class_t... classes_>
struct gpu_ladder {};

/** Every class carrying variants, each lineage newest first and the serial floor last, or the one
 *  class of a monoversioned build; the walk skips a rung the build left out, the list keeps it. */
#if UFORM_TARGET_MONOVERSIONED
using cpu_ladder_t = cpu_ladder<cpu_class_t::monoversioned_k>;
inline constexpr cpu_class_t cpu_ladder_floor_k = cpu_class_t::monoversioned_k;
#else
using cpu_ladder_t =
    cpu_ladder<cpu_class_t::intel_diamond_k, cpu_class_t::intel_granite_k, cpu_class_t::intel_sapphire_k,
               cpu_class_t::amd_genoa_k, cpu_class_t::intel_icelake_k, cpu_class_t::intel_skylake_k,
               cpu_class_t::intel_sierra_k, cpu_class_t::intel_haswell_k, cpu_class_t::nvidia_vera_k,
               cpu_class_t::apple_m4_k, cpu_class_t::aws_g5_k, cpu_class_t::apple_m2_k, cpu_class_t::aws_g2_k,
               cpu_class_t::alibaba_c930_k, cpu_class_t::ibm_power10_k, cpu_class_t::wasm64_extended_k,
               cpu_class_t::wasm32_v128_k, cpu_class_t::serial_k>;
inline constexpr cpu_class_t cpu_ladder_floor_k = cpu_class_t::serial_k;
#endif

/** The NVIDIA classes newest first, Ampere the floor; Rubin compiles no unit, so its devices
 *  pick Blackwell. */
using nvidia_ladder_t =
    gpu_ladder<gpu_class_t::nvidia_rubin_k, gpu_class_t::nvidia_blackwellrtx_k, gpu_class_t::nvidia_blackwell_k,
               gpu_class_t::nvidia_hopper_k, gpu_class_t::nvidia_ada_k, gpu_class_t::nvidia_ampere_k>;

/** The AMD classes newest first, CDNA3 the floor. */
using amd_ladder_t = gpu_ladder<gpu_class_t::amd_cdna5_k, gpu_class_t::amd_cdna4_k, gpu_class_t::amd_cdna3_k>;

/** The Apple classes newest first, the Metal 7 family the floor. */
using apple_ladder_t =
    gpu_ladder<gpu_class_t::apple_metal10_k, gpu_class_t::apple_metal9_k, gpu_class_t::apple_metal7_k>;

/** The rung a GPU class degrades to within its vendor, @c unknown_k past the oldest. */
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
#if UFORM_TARGET_IBM_POWER10
    case cpu_class_t::ibm_power10_k: return true;
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

/** The masks this machine runs intersected with those of the class @p ceiling names, from the one
 *  caller of NumKong's and ForkUnion's CPU detection, which grants the AMX tiles the CPU offers
 *  and drops them where the kernel refuses. A value no class takes is an invalid argument, and a
 *  ceiling no compiled class fits, like one below a monoversioned build's masks, is unsupported,
 *  while one above the machine answers the machine's own masks. */
inline util::expected<cpu_ceiling_t> pick_cpu_ceiling(uform_cpu_class_t ceiling) noexcept {
    if (ceiling != uform_cpu_class_unknown_k && !listed_in(cpu_classes_k, ceiling))
        return util::expected<cpu_ceiling_t>::fail(util::status_code_t::invalid_argument_k,
                                                   "No CPU class has that value");
    cpu_ceiling_t picked;
    if (nk_capabilities_detected_cpu(&picked.compute) != nk_success_k) picked.compute = nk_cap_serial_k;
    picked.sync = fu::cpu_capabilities();
    nk_capability_t const tiles = picked.compute & (nk_cap_sapphireamx_k | nk_cap_graniteamx_k | nk_cap_diamondamx_k);
    if (tiles && nk_configure_thread_cpu(tiles) != nk_success_k) picked.compute &= ~tiles;
    if (ceiling != uform_cpu_class_unknown_k) {
        cpu_class_t const ceiling_class = cpu_class_of(ceiling);
        picked.compute &= numkong_capabilities_of(ceiling_class);
        picked.sync = static_cast<fu::capabilities_t>(picked.sync & forkunion_capabilities_of(ceiling_class));
    }
    for (uform_cpu_class_t const rung : cpu_classes_k)
        if (cpu_class_compiled(cpu_class_of(rung)) && cpu_rung_fits(cpu_class_of(rung), picked)) return picked;
    return util::expected<cpu_ceiling_t>::fail(util::status_code_t::unsupported_k,
                                               "The build compiled no class under that ceiling");
}

/** Whether the build compiled the unit of @p gpu_class, as CMake told the dispatch unit through
 *  @c UFORM_TARGET_<CLASS>. */
constexpr bool gpu_class_compiled(gpu_class_t gpu_class) noexcept {
    switch (gpu_class) {
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
#if UFORM_TARGET_APPLE_METAL7
    case gpu_class_t::apple_metal7_k: return true;
#endif
#if UFORM_TARGET_APPLE_METAL9
    case gpu_class_t::apple_metal9_k: return true;
#endif
#if UFORM_TARGET_APPLE_METAL10
    case gpu_class_t::apple_metal10_k: return true;
#endif
    default: return false;
    }
}

/** Rungs of a ladder this build compiled; a monoversioned build compiles at most one. */
template <cpu_class_t... classes_>
consteval std::size_t compiled_rungs(cpu_ladder<classes_...>) noexcept {
    return (std::size_t {cpu_class_compiled(classes_)} + ... + 0);
}
template <gpu_class_t... classes_>
consteval std::size_t compiled_rungs(gpu_ladder<classes_...>) noexcept {
    return (std::size_t {gpu_class_compiled(classes_)} + ... + 0);
}

#if UFORM_TARGET_MONOVERSIONED
static_assert(compiled_rungs(cpu_ladder_t {}) <= 1);
static_assert(compiled_rungs(nvidia_ladder_t {}) <= 1);
static_assert(compiled_rungs(amd_ladder_t {}) <= 1);
static_assert(compiled_rungs(apple_ladder_t {}) <= 1);
#endif

/** Whether the build compiled the units of @p vendor , as CMake told the dispatch unit through
 *  @c UFORM_ARCH_CUDA_ , @c UFORM_ARCH_ROCM_ and @c UFORM_ARCH_METAL_ ; every class CMake kept has
 *  a unit, the ones advancing nothing past their parent answering no table. */
constexpr bool gpu_vendor_compiled([[maybe_unused]] nk_capability_t vendor) noexcept {
#if UFORM_ARCH_CUDA_
    if (vendor == nk_cap_cuda_k) return true;
#endif
#if UFORM_ARCH_ROCM_
    if (vendor == nk_cap_rocm_k) return true;
#endif
#if UFORM_ARCH_METAL_
    if (vendor == nk_cap_metal_k) return true;
#endif
    return false;
}

/** The vendor whose devices @p gpu_class names; unknown for the zero class. */
constexpr uform_gpu_vendor_t gpu_vendor_of(gpu_class_t gpu_class) noexcept {
    switch (gpu_class) {
    case gpu_class_t::nvidia_ampere_k:
    case gpu_class_t::nvidia_ada_k:
    case gpu_class_t::nvidia_hopper_k:
    case gpu_class_t::nvidia_blackwell_k:
    case gpu_class_t::nvidia_blackwellrtx_k:
    case gpu_class_t::nvidia_rubin_k: return uform_gpu_vendor_nvidia_k;
    case gpu_class_t::amd_cdna3_k:
    case gpu_class_t::amd_cdna4_k:
    case gpu_class_t::amd_cdna5_k: return uform_gpu_vendor_amd_k;
    case gpu_class_t::apple_metal7_k:
    case gpu_class_t::apple_metal9_k:
    case gpu_class_t::apple_metal10_k: return uform_gpu_vendor_apple_k;
    case gpu_class_t::unknown_k: return uform_gpu_vendor_unknown_k;
    }
    return uform_gpu_vendor_unknown_k;
}

/** Whether @p gpu_class runs on a device of @p vendor : built, of that vendor, and its kits all
 *  among the @p compiled ones NumKong's loader runs there. */
inline bool gpu_class_serves(gpu_class_t gpu_class, uform_gpu_vendor_t vendor, nk_capability_t compiled) noexcept {
    nk_capability_t const kits = numkong_capabilities_of(gpu_class);
    return gpu_class_compiled(gpu_class) && gpu_vendor_of(gpu_class) == vendor && (compiled & kits) == kits;
}

/** Whether @p gpu_class sits at or under @p ceiling on its lineage, as every class does under no
 *  ceiling. */
constexpr bool gpu_class_under_ceiling(gpu_class_t gpu_class, uform_gpu_class_t ceiling) noexcept {
    if (ceiling == uform_gpu_class_unknown_k) return true;
    for (gpu_class_t rung = static_cast<gpu_class_t>(ceiling);; rung = parent_class_of(rung)) {
        if (rung == gpu_class) return true;
        if (rung == gpu_class_t::unknown_k) return false;
    }
}

/** A set of GPU classes, one bit per C enum value. */
using gpu_class_mask_t = util::u32_t;

/** The bit of @p gpu_class in a @ref gpu_class_mask_t . */
constexpr gpu_class_mask_t gpu_class_bit(gpu_class_t gpu_class) noexcept {
    return gpu_class_mask_t {1} << static_cast<util::u32_t>(gpu_class);
}

/** Every class of @p ladder that loads on a device of @p vendor whose loader runs the @p compiled
 *  kits. */
template <gpu_class_t... classes_>
gpu_class_mask_t gpu_classes_loading(gpu_ladder<classes_...>, uform_gpu_vendor_t vendor,
                                     nk_capability_t compiled) noexcept {
    gpu_class_mask_t loading = 0;
    auto rung = [&]<gpu_class_t class_>() noexcept {
        if (gpu_class_serves(class_, vendor, compiled)) loading |= gpu_class_bit(class_);
    };
    (rung.template operator()<classes_>(), ...);
    return loading;
}

/** The newest class of @p ladder in @p loaded at or under @p ceiling , or @c unknown_k where none
 *  is; @p loaded is one device's set, or every device's sets intersected, as one model's replicas
 *  run one class. */
template <gpu_class_t... classes_>
gpu_class_t pick_gpu_class(gpu_ladder<classes_...>, gpu_class_mask_t loaded, uform_gpu_class_t ceiling) noexcept {
    gpu_class_t picked = gpu_class_t::unknown_k;
    auto rung = [&]<gpu_class_t class_>() noexcept {
        if (picked == gpu_class_t::unknown_k && (loaded & gpu_class_bit(class_)) &&
            gpu_class_under_ceiling(class_, ceiling))
            picked = class_;
    };
    (rung.template operator()<classes_>(), ...);
    return picked;
}

/** Whether an executor runs no call, or one call from its claim to its end. */
enum class call_claim_t : util::u32_t {
    free_k,
    held_k,
};

/** The waits one CPU class compiles in its class unit for the C layer, which is compiled once at
 *  the baseline: each reads its word through the class's atomic reference and waits between reads
 *  through the class's waiter. */
struct host_verbs_t {

    /** Returns the status of @p outcome once it leaves @c uform_status_deferred_k . */
    uform_status_t (*wait)(uform_outcome_t const* outcome) noexcept;

    /** Returns once no call holds the executor whose claim is @p claim . */
    void (*wait_while_claimed)(call_claim_t const* claim) noexcept;

    /** Returns once no executor call names the handle whose count is @p named_calls . */
    void (*wait_named_calls)(util::u32_t const* named_calls) noexcept;
};

/** The host verbs one class unit compiles, defined in @c encoder_specializations.hpp , which the
 *  class units alone include, so no body compiled for a class reaches a caller of another. */
template <cpu_class_t cpu_class_>
struct host_class_tables {
    static host_verbs_t const* find() noexcept;
};

/** The host verbs of the newest compiled class on @p ladder that fits under @p ceiling, one of
 *  which @ref pick_cpu_ceiling made sure of. */
template <cpu_class_t... classes_>
host_verbs_t const* pick_host_verbs(cpu_ladder<classes_...>, cpu_ceiling_t ceiling) noexcept {
    host_verbs_t const* picked = nullptr;
    auto rung = [&]<cpu_class_t class_>() noexcept {
        if constexpr (cpu_class_compiled(class_))
            if (!picked && cpu_rung_fits(class_, ceiling)) picked = host_class_tables<class_>::find();
    };
    (rung.template operator()<classes_>(), ...);
    return picked;
}

/** The host verbs of the newest compiled class this machine admits, picked once, which every
 *  outcome, claim and named-call wait of the C layer goes through; defined outside the class units,
 *  each of which sees only its own class's tables. */
host_verbs_t const& host_verbs() noexcept;

#pragma endregion Classes

#pragma region Outcomes

/** The C status a failure of the engine's own vocabulary crosses into, whose values it shares. */
[[nodiscard]] inline uform_status_t status_of(util::status_t status) noexcept {
    return static_cast<uform_status_t>(status.code);
}

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
inline void propagate_pending(uform_outcome_t* outcome) noexcept {
    outcome->message = nullptr;
    fu::standard_atomic_ref<uform_status_t> {outcome->status}.store(uform_status_deferred_k, std::memory_order_relaxed);
}

/** The one final write to @p outcome: the message, then the status every waiter acquires. The call
 *  touches nothing it was passed after this, and the message is null exactly on success. */
inline bool propagate_error(uform_status_t status, uform_outcome_t* outcome,
                            char const* message) noexcept { // vendor-boundary: the C outcome's static message
    bool const succeeded = status == uform_status_success_k;
    outcome->message = succeeded ? nullptr : message;
    fu::standard_atomic_ref<uform_status_t> {outcome->status}.store(status, std::memory_order_release);
    return succeeded;
}

/** An engine failure crosses with its own note, else with the boundary's static @p note , else
 *  names itself. */
inline bool propagate_error(util::status_t status, uform_outcome_t* outcome, char const* note = nullptr) noexcept {
    uform_status_t const code = status_of(status);
    return propagate_error(code, outcome, status.note ? status.note : note ? note : uform_status_name(code));
}

#pragma endregion Outcomes

#pragma region Progress

/** Whether the call a progress handler watches continues or unwinds. */
enum class continuation_t : util::u8_t {
    continue_k,
    cancel_k,
};

/** A half-open range along a call's batch axis that the engine takes whole. */
struct slice_t {
    std::size_t begin = 0;
    std::size_t end = 0;
};

/**
 *  @brief One call's reports: the C handler, thinned to the caller's steps, and the slices a batch
 *      verb hands its engine whole. Only the calling thread ticks, between slices its pool joined,
 *      so the handler never runs concurrently and needs no lock.
 *
 *  Without a handler a batch is one slice and nothing is reported; with one it splits into
 *  @c progress.steps nearly equal slices, ten when zero. A verb counting its own units, a load's
 *  tensors, forwards each tick and only the ones crossing into the next step reach the handler.
 */
class progress_ticks_t {
  public:
    progress_ticks_t(uform_progress_t progress, std::size_t total) noexcept : progress_(progress), total_(total) {}
    progress_ticks_t(progress_ticks_t const&) = delete;
    progress_ticks_t& operator=(progress_ticks_t const&) = delete;

    /** Walks the slices of a batch in order; slice @c i starts at @c ⌈i · total/slices⌉ , so
     *  lengths differ by one at most. */
    class iterator {
      public:
        iterator(std::size_t index, std::size_t slices, std::size_t total) noexcept
            : index_(index), slices_(slices), total_(total) {}
        slice_t operator*() const noexcept { return {boundary_(index_), boundary_(index_ + 1)}; }
        iterator& operator++() noexcept { return ++index_, *this; }
        bool operator!=(iterator const& other) const noexcept { return index_ != other.index_; }

      private:
        [[nodiscard]] std::size_t boundary_(std::size_t index) const noexcept {
            return util::divide_round_up(index * total_, slices_);
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

    /** The last step a forwarded tick closed. */
    std::size_t closed_ = 0;
};

#pragma endregion Progress

#pragma region Memory

/** Memory one call at a time rewinds and allocates from, whose owner reserves and frees its
 *  chunks. */
using util::chunked_arena_t;

#pragma endregion Memory

#pragma region Executors

/** One vendor's bundle as every device of the process loaded it. */
using util::kernel_tables;

#pragma region Shared VRAM

/** The GPU side of the process, built at its first VRAM call: every vendor's driver, every device
 *  they report, NVIDIA first, then AMD, then Apple, a device's position here being its ordinal in
 *  the C interface, the classes whose bundles loaded on each, and every vendor's kernels, loaded
 *  once per device and shared by every executor there. A device no class runs on is listed too and
 *  refuses all work. */
struct gpu_runtime_t {
    util::gpu_drivers_t drivers;
    util::gpu_device_t devices[3 * util::gpu_devices_capacity_k];
    gpu_class_mask_t loaded[3 * util::gpu_devices_capacity_k] = {};
    kernel_tables<util::cuda_module_t> cuda_kernels;
    kernel_tables<util::hip_module_t> hip_kernels;
    kernel_tables<util::metal_module_t> metal_kernels;
    std::size_t devices_count = 0;

    gpu_runtime_t() noexcept;
};

/** The one runtime every VRAM call passes its drivers down from. */
inline gpu_runtime_t& gpu_runtime() noexcept {
    static gpu_runtime_t runtime;
    return runtime;
}

/** Every vendor's driver, loaded at the first VRAM call. */
inline util::gpu_drivers_t& gpu_drivers() noexcept { return gpu_runtime().drivers; }

/** Every GPU the drivers report, as the ABI numbers them. */
inline std::span<util::gpu_device_t const> gpu_devices() noexcept {
    gpu_runtime_t const& runtime = gpu_runtime();
    return {runtime.devices, runtime.devices_count};
}

/** The classes whose bundles load on each device @ref gpu_devices lists, asked once at listing. */
inline std::span<gpu_class_mask_t const> gpu_classes_loaded() noexcept {
    gpu_runtime_t const& runtime = gpu_runtime();
    return {runtime.loaded, runtime.devices_count};
}

#pragma endregion Shared VRAM

/** The position in @ref gpu_devices of device @p vendor_ordinal of @p vendor , or -1 when none
 *  is listed there. */
[[nodiscard]] inline util::i32_t listed_ordinal(util::gpu_vendor_t vendor, std::size_t vendor_ordinal) noexcept {
    std::span<util::gpu_device_t const> const listed = gpu_devices();
    for (std::size_t index = 0; index != listed.size(); ++index)
        if (listed[index].vendor == vendor && listed[index].vendor_ordinal == vendor_ordinal)
            return static_cast<util::i32_t>(index);
    return -1;
}

/** The position in @ref gpu_devices of the device @p stream of an @p executor_type_ queues on. */
template <typename executor_type_>
[[nodiscard]] util::i32_t listed_ordinal_of(typename executor_type_::stream_t const& stream) noexcept {
    return listed_ordinal(executor_type_::vendor_k, stream.ordinal);
}

#pragma endregion Executors

#pragma region Models

/** Where an encoder's tower lives; there is no NVMe residency for models. */
enum class vendored_residency_t : util::u32_t {
    ram_k,
    vram_nvidia_k,
    vram_amd_k,
    vram_apple_k,
};

/** The vendor in the C interface whose devices @p residency names; unknown for RAM. */
constexpr uform_gpu_vendor_t gpu_vendor_of(vendored_residency_t residency) noexcept {
    switch (residency) {
    case vendored_residency_t::vram_nvidia_k: return uform_gpu_vendor_nvidia_k;
    case vendored_residency_t::vram_amd_k: return uform_gpu_vendor_amd_k;
    case vendored_residency_t::vram_apple_k: return uform_gpu_vendor_apple_k;
    case vendored_residency_t::ram_k: return uform_gpu_vendor_unknown_k;
    }
    return uform_gpu_vendor_unknown_k;
}

/** The executor each residency's verbs take: a host tower's thread pool, a device tower's stream
 *  record of its vendor's executor. */
template <vendored_residency_t residency_>
struct vendored_residency_executor;
template <>
struct vendored_residency_executor<vendored_residency_t::ram_k> {
    using type = fu::pool_t&;
};
template <>
struct vendored_residency_executor<vendored_residency_t::vram_nvidia_k> {
    using type = util::cuda_executor_t::stream_t&;
};
template <>
struct vendored_residency_executor<vendored_residency_t::vram_amd_k> {
    using type = util::hip_executor_t::stream_t&;
};
template <>
struct vendored_residency_executor<vendored_residency_t::vram_apple_k> {
    using type = util::metal_executor_t::stream_t&;
};

/** The verbs one encoder variant compiles in its class unit, defined in @c encoder_punned.hpp . */
template <vendored_residency_t residency_>
struct encoder_verbs;

/** The variant a load picked, as its residency's verb table, which alone casts each tower back
 *  to its type. */
using encoder_residency_tables_t = std::variant<
    encoder_verbs<vendored_residency_t::ram_k> const*, encoder_verbs<vendored_residency_t::vram_nvidia_k> const*,
    encoder_verbs<vendored_residency_t::vram_amd_k> const*, encoder_verbs<vendored_residency_t::vram_apple_k> const*>;

/** One copy of a model's tower on one GPU, matched to an executor's stream by its position in
 *  @ref gpu_devices , and freed with that device bound. */
struct vram_replica_t {
    util::i32_t ordinal = 0;
    void* tower = nullptr;
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

/** The one call that holds an executor, from its claim to its end. */
struct holding_call_t {

    /** Taken by an exchange at the entry, released by a store just before the status. */
    call_claim_t claim = call_claim_t::free_k;

    /** The holding call's arguments; meeting them again, the entry runs the verb's body. */
    void const* arguments = nullptr;

    /** A delegated call's copy of its arguments, which outlives the caller's structure. */
    deferred_arguments_t delegated_arguments;

    /** The caller's outcome, published last. */
    uform_outcome_t* outcome = nullptr;

    /** The status the verb body recorded, which @ref end_call publishes. */
    uform_outcome_t recorded {uform_status_internal_k, "The call finished without reporting a status"};

    /** The counts of the handles the call names besides its executor, raised at the entry,
     *  lowered at its end. */
    std::array<util::u32_t*, 4> named_calls {};

    /** The kernel ID of the thread running the body: an eager call's caller, or a delegated one's
     *  pool thread 0; zero outside the body. */
    util::u64_t body_thread = 0;
};

/** Lets go of @p arena when the call whose published outcome is @p call holds it. */
void release_arena(uform_arena_t arena, uform_outcome_t const* call) noexcept;

/** How a call stands with an arena it names: refused by another holder, newly taken, or taken
 *  before by itself. */
enum class arena_hold_t { refused_k, taken_k, retaken_k };

/** Holds @p arena for the call whose published outcome is @p call ; a null arena holds nothing,
 *  left for the verb to refuse, and has no generation to advance. */
arena_hold_t hold_arena(uform_arena_t arena, uform_outcome_t const* call) noexcept;

/** Takes @p executor for one call, once its pool's thread 0 left the last call's slice; false while
 *  another call holds it. */
bool take_call_claim(uform_executor_handle_t& executor) noexcept;

/** Whether a caller-exclusive pool backs @p executor , running deferred calls on its threads. */
bool delegates(uform_executor_handle_t& executor) noexcept;

/** Keeps @p recorded , the body's status, for @ref end_call to publish. */
void record_status(uform_executor_t executor, uform_outcome_t const& recorded) noexcept;

/** Ends the call holding @p executor : the named counts, the claim, then the status. */
void end_call(uform_executor_handle_t& executor) noexcept;

/** Arguments naming a model, an arena, a mesh or a trainer whose count the call raises; outputs for
 *  handles a call creates, and handles this build never defines, hold no count. */
template <typename arguments_type_>
concept a_model_naming_call = requires(arguments_type_& arguments) { arguments.model->named_calls; };
template <typename arguments_type_>
concept an_arena_naming_call = requires(arguments_type_& arguments) { arguments.arena->named_calls; };
template <typename arguments_type_>
concept a_mesh_naming_call = requires(arguments_type_& arguments) { arguments.mesh->named_calls; };
template <typename arguments_type_>
concept a_trainer_naming_call = requires(arguments_type_& arguments) { arguments.trainer->named_calls; };

/** Marks the calling thread as the one running @p call 's body. */
inline void hold_body(holding_call_t& call) noexcept {
    fu::standard_atomic_ref<util::u64_t>(call.body_thread).store(fu::current_thread_id(), std::memory_order_relaxed);
}

/** Whether the calling thread runs the body of the call holding @p executor . */
bool runs_holding_call(uform_executor_handle_t& executor) noexcept;

/** Takes the claim of @p executor once no call holds it, joining the slice the last call's thread 0
 *  may still be leaving, so the caller alone joins, parks or frees its pool. */
void take_idle_claim(uform_executor_handle_t& executor) noexcept;

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
bool submitted(arguments_type_* arguments) noexcept {
    // Dependent, so the handle's members resolve at instantiation
    auto* const executor = arguments->executor;
    if (!executor) return false;
    // Any thread compares while the claim holder writes, so both sides go through the reference
    if (arguments ==
        fu::standard_atomic_ref<void const*>(executor->holding_call.arguments).load(std::memory_order_relaxed))
        return false;
    if constexpr (an_arena_naming_call<arguments_type_>)
        if (arguments->arena && fu::standard_atomic_ref<util::u32_t> {arguments->arena->exports}.load())
            return propagate_error(uform_status_contended_k, arguments->outcome,
                                   "An export of the arena's outputs is open"),
                   true;
    if (!take_call_claim(*executor))
        return propagate_error(uform_status_contended_k, arguments->outcome, "Another call holds the executor"), true;
    if constexpr (an_arena_naming_call<arguments_type_>) {
        arena_hold_t const hold = hold_arena(arguments->arena, arguments->outcome);
        // An export opened since the check above sees the hold and backs out, or this load sees it
        bool const exported = hold == arena_hold_t::taken_k &&
                              fu::standard_atomic_ref<util::u32_t> {arguments->arena->exports}.load() != 0;
        if (hold == arena_hold_t::refused_k || exported) {
            if (exported) release_arena(arguments->arena, arguments->outcome);
            fu::standard_atomic_ref<call_claim_t> {executor->holding_call.claim}.store(call_claim_t::free_k,
                                                                                       std::memory_order_release);
            return propagate_error(
                       uform_status_contended_k, arguments->outcome,
                       exported ? "An export of the arena's outputs is open" : "The arena serves another call"),
                   true;
        }
        // Relaxed, as the arena hold's release and the export handshake order it for readers
        if (hold == arena_hold_t::taken_k)
            fu::standard_atomic_ref<util::u64_t> {arguments->arena->generation}.fetch_add(1, std::memory_order_relaxed);
    }
    auto& call = executor->holding_call;
    call.outcome = arguments->outcome;
    call.recorded = {uform_status_internal_k, "The call finished without reporting a status"};
    std::size_t named = 0;
    auto const name = [&](auto* handle) noexcept {
        if (!handle) return;
        fu::standard_atomic_ref<util::u32_t> {handle->named_calls}.fetch_add(1, std::memory_order_relaxed);
        call.named_calls[named++] = &handle->named_calls;
    };
    if constexpr (a_model_naming_call<arguments_type_>) name(arguments->model);
    if constexpr (an_arena_naming_call<arguments_type_>) name(arguments->arena);
    if constexpr (a_mesh_naming_call<arguments_type_>) name(arguments->mesh);
    if constexpr (a_trainer_naming_call<arguments_type_>) name(arguments->trainer);
    propagate_pending(arguments->outcome); // before the body or the dispatch, never after
    if (arguments->completion != uform_completion_deferred_k || !delegates(*executor)) {
        fu::standard_atomic_ref<void const*>(call.arguments).store(arguments, std::memory_order_relaxed);
        hold_body(call);
        verb_(arguments);
        return end_call(*executor), true;
    }
    auto& copy = call.delegated_arguments.template emplace<arguments_type_>(*arguments);
    fu::standard_atomic_ref<void const*>(call.arguments).store(&copy, std::memory_order_relaxed);
    using executor_handle_t = std::remove_pointer_t<decltype(arguments->executor)>;
    // Thread 0 runs the whole call in its slice, forking the rest of the pool as it goes
    fu_for_threads_t const delegated_body = [](fu_lambda_context_t context, std::size_t thread, std::size_t) noexcept {
        if (thread) return;
        auto& held = *static_cast<executor_handle_t*>(context);
        hold_body(held.holding_call);
        verb_(std::get_if<arguments_type_>(&held.holding_call.delegated_arguments));
        end_call(held);
    };
    // A local token: thread 0 may end the call, and a waiter free the executor, before it lands
    fu_generation_t generation = 0;
    if (fu::failed(executor->threads.unsafe_for_threads(delegated_body, executor, generation))) {
        // No body runs to let go of the arena the entry took
        if constexpr (an_arena_naming_call<arguments_type_>) release_arena(arguments->arena, arguments->outcome);
        call.recorded = {uform_status_contended_k, "The executor's threads run another dispatch"}, end_call(*executor);
    }
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
    explicit call_outcome_t(uform_outcome_t* published) noexcept : published_(published) {
        propagate_pending(published_);
    }

    /** For a verb's call, whose executor, when it names one, publishes the status. */
    template <typename arguments_type_>
    explicit call_outcome_t(arguments_type_ const& arguments) noexcept
        : published_(arguments.outcome), executor_(arguments.executor) {
        if constexpr (an_arena_naming_call<arguments_type_>) arena_ = arguments.arena;
        if (!executor_) propagate_pending(published_);
    }

    ~call_outcome_t() noexcept {
        release_arena(arena_, published_);
        if (executor_) record_status(executor_, recorded_);
        else propagate_error(recorded_.status, published_, recorded_.message);
    }
    call_outcome_t(call_outcome_t const&) = delete;
    call_outcome_t& operator=(call_outcome_t const&) = delete;

    /** The private record every step reports into. */
    operator uform_outcome_t*() noexcept { return &recorded_; }

    /** The caller's outcome, which identifies the call holding an arena. */
    [[nodiscard]] uform_outcome_t const* published() const noexcept { return published_; }

  private:
    uform_outcome_t* published_;
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
uform_arena_t claim_arena(uform_arena_t arena, uform_executor_t executor, std::initializer_list<void const*> inputs,
                          call_outcome_t& outcome) noexcept;

/**
 *  @brief Makes room in @p arena for allocations of @p bytes , as their footprints sum: host
 *      pages, or device memory in the order of its executor's first stream, binding that stream's
 *      device before it allocates.
 *
 *  @par Statuses:
 *  - @c out_of_memory_k The host or the device refused a chunk, or the chunk table could not grow.
 *  - @c device_k The executor's first device could not be bound.
 */
util::status_t reserve_arena(uform_arena_handle_t& arena, std::size_t bytes) noexcept;

#pragma endregion Deferred Calls

} // namespace unum::uform

namespace uf = unum::uform;
namespace util = unum::uform::utilities;

/** The state behind @c uform_executor_t: the one call it holds, its host threads, and the GPUs it
 *  drives, if any; the variant is the kind tag, so no second word can disagree. */
struct uform_executor_handle_t {

    /** The one call that holds the executor, from its claim to its end. */
    uf::holding_call_t holding_call;

    /** Arenas made with the executor and not yet freed, which refuse its free; set through the
     *  standard reference, as any thread may make or free one. */
    util::u32_t arenas = 0;

    /** What the threads may run: the machine's masks under the executor's ceiling class. */
    uf::cpu_ceiling_t cpu_ceiling;

    /** Nothing for a RAM executor, or the streams a VRAM executor queues on, one per GPU it drives,
     *  each dying after its stream drained and its scratch was freed. */
    util::gpu_executor_t device;

    /** The class every device of a VRAM executor dispatches as: the best class all of them load
     *  under the executor's ceiling, whose capabilities name the vendor every NumKong call
     *  dispatches on; unknown on a RAM executor. */
    uf::gpu_class_t gpu_class = uf::gpu_class_t::unknown_k;

    /** Host pages for what only a VRAM executor's decoders read: the plans and every thread's
     *  decode window, reserved before each call allocates them. */
    uf::chunked_arena_t staging;

    /** A RAM executor's pool, or a VRAM executor's decoders, which the destructor joins before it
     *  frees the staging pages they write. */
    uf::fu::pool_t threads;

    uform_executor_handle_t() noexcept = default;
    uform_executor_handle_t(uform_executor_handle_t const&) = delete;
    uform_executor_handle_t& operator=(uform_executor_handle_t const&) = delete;

    /** Joins the decoders first, as the destructor's body runs before any member dies: the pool
     *  swapped out into a temporary dies with it, joining its threads. Then frees the staging
     *  pages they wrote. */
    ~uform_executor_handle_t() noexcept {
        threads = uf::fu::pool_t {};
        util::release_host_pages(staging);
    }
};

/** What @c uform_arena_t owns: memory that one call at a time rewinds and refills, as host pages or
 *  on one device, and the call holding it. */
struct uform_arena_handle_t {

    /** Host pages, or the memory of the first device of @ref executor , fixed at
     *  @c uform_arena_init . */
    uf::chunked_arena_t chunks;

    /** The executor the arena was made with, which counts it until its free. Device memory rides
     *  the first stream of a VRAM one and serves only its calls; host pages serve any host call. */
    uform_executor_t executor = nullptr;

    /** The published outcome of the call holding the arena, from its claim until just before its
     *  final store, read through the standard reference since a call may end on a pool thread. */
    uform_outcome_t const* holder = nullptr;

    /** Executor calls naming the arena that have not ended, which its free waits out. */
    util::u32_t named_calls = 0;

    /** Open exports of the outputs, which refuse every call naming the arena and its free; set
     *  through the standard reference, as a consumer closes one from any thread. */
    util::u32_t exports = 0;

    /** Calls that took the arena and found no export open, each advancing it once before it
     *  writes, through the standard reference; @c uform_arena_generation reads it. */
    util::u64_t generation = 0;
};

/** What @c uform_model_t owns: the geometry its checkpoint held, the precision its blocks run in,
 *  and its packed towers, so the checkpoint closes after the load. */
struct uform_model_handle_t {
    uf::encoder::geometry_t geometry;
    uf::encoder::precision_t precision = uf::encoder::precision_t::bf16_k;

    /** Bytes every tower holds together, by the extents and dtypes of what each packs. */
    std::size_t memory_usage = 0;

    /** Each question type's calibrated inverse temperature, in host memory where a choice's softmax
     *  runs; empty without the byte decoder. */
    uf::st::basic_vector<util::f32_t> choice_inverse_temperatures;

    /** The variant picked once at load: its residency's verb table, which alone casts
     *  each tower back. */
    uf::encoder_residency_tables_t tables;

    /** The one host tower, serving any RAM executor, or one replica per GPU the load named, every
     *  one of the picked class; @ref tables names the residency. */
    util::per_gpu_array<uf::vram_replica_t> towers;

    /** Executor calls naming the model that have not ended, which its free waits out. */
    util::u32_t named_calls = 0;

    uform_model_handle_t() noexcept = default;
    uform_model_handle_t(uform_model_handle_t const&) = delete;
    uform_model_handle_t& operator=(uform_model_handle_t const&) = delete;

    /** Destroys every tower through the verbs that made it; defined beside them in
     *  @c model.cpp . */
    ~uform_model_handle_t() noexcept;
};
