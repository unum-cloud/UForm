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

#include <span>        // `std::span` — the scratch a call borrows
#include <type_traits> // `std::conditional_t`

#include "uform/encoder/vram_kits.hpp" // `vram_nvidia_entries_kit_t`, `vram_amd_entries_kit_t`

#include "lib.hpp" // `encoder_home_t`, `cpu_class_t`, `gpu_class_t`, the ladders

namespace unum::uform {

/**
 *  @brief The verbs one encoder variant compiles in its class unit, over its tower passed as
 *      @c void* , which only these entries cast back to its type.
 *
 *  A host tower ignores the stream its @c create is handed; a device tower packs on it and
 *  returns once its weights are ready.
 */
template <encoder_home_t home_>
struct encoder_verbs {
    using executor_t = typename encoder_home_executor<home_>::type;

    /** The CPU class a host variant's kits compile for; unknown on a device. */
    cpu_class_t cpu_class;

    /** The GPU class a device variant's kits compile for; unknown on the host and on a vendor
     *  floor. */
    gpu_class_t gpu_class;

    core::expected<void *> (*create)(core::checkpoint_to_load_t const &checkpoint, encoder::geometry_t const &geometry,
                                     encoder::precision_t precision, nk_stream_t stream) noexcept;
    void (*destroy)(void *tower) noexcept;
    core::expected<std::size_t> (*scratch_bytes)(void const *tower, std::size_t tokens, std::size_t segments,
                                                 executor_t executor) noexcept;
    core::expected_t (*encode)(void const *tower, core::multimodal_batch_view_t batch, encoder::outputs_t outputs,
                               std::span<b8_t> scratch, executor_t executor) noexcept;
};

/** The variant one class unit compiles for one home: @c find answers its verb table, or null where
 *  the class picks nothing its parent does not, so a ladder falls to the next rung. */
template <encoder_home_t home_, auto class_>
struct encoder_class_tables {
    static encoder_verbs<home_> const *find() noexcept;
};

/** The vendor's capability group a device home runs on. */
constexpr nk_capability_t vendor_of(encoder_home_t home) noexcept {
    return home == encoder_home_t::vram_amd_k ? nk_cap_rocm_k : nk_cap_cuda_k;
}

/** The entries kit of a device home's vendor: the steps and copies every class of the vendor
 *  shares, defined only in a build compiling the vendor, so a caller first asks
 *  @ref gpu_vendor_compiled . */
template <encoder_home_t home_>
using entries_kit_of = std::conditional_t<home_ == encoder_home_t::vram_amd_k, encoder::vram_amd_entries_kit_t,
                                          encoder::vram_nvidia_entries_kit_t>;

/** The newest compiled class on @p ladder whose kits @p available runs, whose atomic reference
 *  ForkUnion admits, and which @p pinned takes, as its verbs; the serial floor always answers. */
template <cpu_class_t... classes_>
encoder_verbs<encoder_home_t::ram_k> const *pick_cpu_variant(cpu_ladder<classes_...>, uform_cpu_class_t pinned,
                                                             nk_capability_t available) noexcept {
    encoder_verbs<encoder_home_t::ram_k> const *picked = nullptr;
    auto rung = [&]<cpu_class_t class_>() noexcept {
        if constexpr (cpu_class_compiled(class_)) {
            if (picked) return;
            if (class_ != cpu_class_t::serial_k &&
                (!cpu_pin_takes(pinned, class_) || !numkong_admits(available, class_) || !forkunion_admits(class_)))
                return;
            picked = encoder_class_tables<encoder_home_t::ram_k, class_>::find();
        }
    };
    (rung.template operator()<classes_>(), ...);
    return picked;
}

/** The newest class on @p ladder whose kits a device of @p device_capabilities runs and @p pinned
 *  takes, as its verbs; the vendor floor answers any device of the vendor nothing pins. */
template <encoder_home_t home_, gpu_class_t... classes_>
encoder_verbs<home_> const *pick_gpu_variant(gpu_ladder<classes_...>, uform_gpu_class_t pinned,
                                             nk_capability_t device_capabilities) noexcept {
    encoder_verbs<home_> const *picked = nullptr;
    if constexpr (gpu_vendor_compiled(vendor_of(home_))) {
        if (!(device_capabilities & vendor_of(home_))) return picked;
        nk_capability_t const ceiling = pinned == uform_gpu_class_unknown_k
                                            ? device_capabilities
                                            : numkong_capabilities_of(static_cast<gpu_class_t>(pinned));
        auto rung = [&]<gpu_class_t class_>() noexcept {
            if (picked) return;
            if constexpr (class_ == gpu_class_t::unknown_k) {
                if (pinned != uform_gpu_class_unknown_k) return;
            }
            else {
                nk_capability_t const wanted = numkong_capabilities_of(class_);
                if ((device_capabilities & wanted) != wanted || (ceiling & wanted) != wanted) return;
            }
            picked = encoder_class_tables<home_, class_>::find();
        };
        (rung.template operator()<classes_>(), ...);
    }
    return picked;
}

/** Calls @p visitor with the VRAM home and the class ladder of the vendor @p device_capabilities
 *  names, so the C verbs never name a vendor. */
template <typename visitor_type_>
bool visit_vram_home(nk_capability_t device_capabilities, visitor_type_ &&visitor) noexcept {
    if (device_capabilities & nk_cap_rocm_k)
        return visitor.template operator()<encoder_home_t::vram_amd_k, amd_ladder_t>();
    return visitor.template operator()<encoder_home_t::vram_nvidia_k, nvidia_ladder_t>();
}

} // namespace unum::uform
