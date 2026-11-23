/**
 *  @file src/encoder_punned.hpp
 *  @author Ash Vardanian
 *  @date October 25, 2026
 *  @brief The verbs one encoder variant compiles in its class unit, as function pointers over its
 *      tower passed as @c void*, and the ladder walks that pick a variant at load.
 *
 *  Only @c encoder_specializations.hpp , which the class units alone include, defines the tables a
 *  class answers; the dispatch unit sees their declarations and asks only the classes the build
 *  compiled, so no body compiled for a class reaches a caller of another.
 */
#pragma once

#include <span>        // `std::span` — the scratch a call writes, owned by its caller
#include <type_traits> // `std::conditional_t`

#include "uform/encoder/vram_kits.hpp" // `vram_nvidia_entries_kit_t`, `vram_amd_entries_kit_t`, `vram_apple_entries_kit_t`

#include "lib.hpp" // `vendored_residency_t`, `cpu_class_t`, `gpu_class_t`, the ladders

namespace unum::uform {

/**
 *  @brief The verbs one encoder variant compiles in its class unit, over its tower passed as
 *      @c void* , which only these entries cast back to its type.
 *
 *  A host tower ignores the stream its @c create is handed; a device tower packs on it and
 *  returns once its weights are ready.
 */
template <vendored_residency_t residency_>
struct encoder_verbs {
    using executor_t = typename vendored_residency_executor<residency_>::type;

    /** The CPU class a host variant's kits compile for; @c unknown_k on a device. */
    cpu_class_t cpu_class;

    /** The GPU class a device variant's kits compile for; unknown on the host. */
    gpu_class_t gpu_class;

    expected<void*> (*create)(checkpoint_to_load_t const& checkpoint, encoder::geometry_t const& geometry,
                              encoder::precision_t precision, nk_stream_t stream) noexcept;
    void (*destroy)(void* tower) noexcept;
    expected<std::size_t> (*scratch_bytes)(void const* tower, std::size_t tokens, std::size_t segments,
                                           executor_t executor) noexcept;
    expected_t (*encode)(void const* tower, encoder::multimodal_batch_view_t batch, encoder::outputs_t outputs,
                         bytes_span_t scratch, executor_t executor) noexcept;
};

/** The variant one class unit compiles for one residency: @c find answers its verb table, or null
 *  where the class picks nothing its parent does not, so a ladder falls to the next rung. */
template <vendored_residency_t residency_, auto class_>
struct encoder_class_tables {
    static encoder_verbs<residency_> const* find() noexcept;
};

/** The vendor's capability group a device residency runs on. */
constexpr nk_capability_t vendor_of(vendored_residency_t residency) noexcept {
    switch (residency) {
    case vendored_residency_t::vram_amd_k: return nk_cap_rocm_k;
    case vendored_residency_t::vram_apple_k: return nk_cap_metal_k;
    default: return nk_cap_cuda_k;
    }
}

/** The entries kit of a device residency's vendor: the steps and copies every class of the vendor
 *  shares, defined only in a build compiling the vendor, so a caller first asks
 *  @ref gpu_vendor_compiled . */
template <vendored_residency_t residency_>
using entries_kit_of =
    std::conditional_t<residency_ == vendored_residency_t::vram_amd_k, encoder::vram_amd_entries_kit_t,
                       std::conditional_t<residency_ == vendored_residency_t::vram_apple_k,
                                          encoder::vram_apple_entries_kit_t, encoder::vram_nvidia_entries_kit_t>>;

/** The newest compiled class on @p ladder that fits under @p ceiling, as its verbs, one of which
 *  @ref pick_cpu_ceiling made sure of. */
template <cpu_class_t... classes_>
encoder_verbs<vendored_residency_t::ram_k> const* pick_cpu_variant(cpu_ladder<classes_...>,
                                                                   cpu_ceiling_t ceiling) noexcept {
    encoder_verbs<vendored_residency_t::ram_k> const* picked = nullptr;
    auto rung = [&]<cpu_class_t class_>() noexcept {
        if constexpr (cpu_class_compiled(class_))
            if (!picked && cpu_rung_fits(class_, ceiling))
                picked = encoder_class_tables<vendored_residency_t::ram_k, class_>::find();
    };
    (rung.template operator()<classes_>(), ...);
    return picked;
}

/** No variant in a build without the residency's vendor. */
template <vendored_residency_t residency_, gpu_class_t... classes_>
encoder_verbs<residency_> const* pick_gpu_variant(gpu_ladder<classes_...>, uform_gpu_class_t,
                                                  nk_capability_t) noexcept {
    return nullptr;
}

/** The newest class on @p ladder whose kits a device of @p device_capabilities runs, at or under
 *  @p ceiling , as its verbs. */
template <vendored_residency_t residency_, gpu_class_t... classes_>
    requires(gpu_vendor_compiled(vendor_of(residency_)))
encoder_verbs<residency_> const* pick_gpu_variant(gpu_ladder<classes_...>, uform_gpu_class_t ceiling,
                                                  nk_capability_t device_capabilities) noexcept {
    encoder_verbs<residency_> const* picked = nullptr;
    if (!(device_capabilities & vendor_of(residency_))) return picked;
    nk_capability_t const capped = ceiling == uform_gpu_class_unknown_k
                                       ? device_capabilities
                                       : numkong_capabilities_of(static_cast<gpu_class_t>(ceiling));
    nk_capability_t const usable = device_capabilities & capped;
    auto rung = [&]<gpu_class_t class_>() noexcept {
        constexpr nk_capability_t wanted = numkong_capabilities_of(class_);
        if constexpr (gpu_class_compiled(class_))
            if (!picked && (usable & wanted) == wanted) picked = encoder_class_tables<residency_, class_>::find();
    };
    (rung.template operator()<classes_>(), ...);
    return picked;
}

/** Calls @p visitor with the VRAM residency and the class ladder of the vendor
 *  @p device_capabilities names, so the C verbs never name a vendor. */
template <typename visitor_type_>
bool visit_vram_residency(nk_capability_t device_capabilities, visitor_type_&& visitor) noexcept {
    if (device_capabilities & nk_cap_rocm_k)
        return visitor.template operator()<vendored_residency_t::vram_amd_k, amd_ladder_t>();
    if (device_capabilities & nk_cap_metal_k)
        return visitor.template operator()<vendored_residency_t::vram_apple_k, apple_ladder_t>();
    return visitor.template operator()<vendored_residency_t::vram_nvidia_k, nvidia_ladder_t>();
}

} // namespace unum::uform
