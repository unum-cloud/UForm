/**
 *  @file src/encoder_specializations.hpp
 *  @author Ash Vardanian
 *  @date October 25, 2026
 *  @brief The one place encoder variants are enumerated: the towers a class spells its variant
 *      with, the static bodies each variant's verb table points at, and the class tables a class
 *      unit instantiates once.
 *
 *  Only class units include it. A host class picks its kits from NumKong's mask for the class, and
 *  a device class from its generation's mask. A class whose kits match its parent's answers no
 *  table, so the ladder falls to the parent.
 */
#pragma once

#include <new>         // `std::nothrow`
#include <type_traits> // `std::is_same_v`

#include "uform/encoder/ram_kits.hpp"   // `products_ladder`, `attention_ladder`, `elementwise_ladder`
#include "uform/encoder/ram_tower.hpp"  // `ram_tower`
#include "uform/encoder/vram_kits.hpp"  // `vram_*_ladder`
#include "uform/encoder/vram_tower.hpp" // `vram_tower`

#include "encoder_punned.hpp" // `encoder_verbs`, `encoder_class_tables`, `entries_kit_of`

namespace unum::uform {

#pragma region Residencies

/** The host tower one CPU class runs: each kit the pick of its own ladder under the mask of
 *  @p kits_class_ , and the atomic reference of @p atomics_class_ . */
template <cpu_class_t kits_class_, cpu_class_t atomics_class_ = kits_class_>
using ram_tower_of =
    encoder::ram_tower<typename encoder::products_ladder<numkong_capabilities_of(kits_class_)>::kit_t,
                       typename encoder::attention_ladder<numkong_capabilities_of(kits_class_)>::kit_t,
                       typename encoder::elementwise_ladder<numkong_capabilities_of(kits_class_)>::kit_t,
                       atomic_ref_for_cpu_class<atomics_class_>::template type>;

/** The device tower one GPU class runs. Its load stages through the element-wise kit of the
 *  build's CPU floor, so a monoversioned build links no other host kit. */
template <vendored_residency_t residency_, gpu_class_t gpu_class_>
using vram_tower_of =
    encoder::vram_tower<typename encoder::vram_products_ladder<numkong_capabilities_of(gpu_class_)>::kit_t,
                        typename encoder::vram_attention_ladder<numkong_capabilities_of(gpu_class_)>::kit_t,
                        typename encoder::vram_elementwise_ladder<numkong_capabilities_of(gpu_class_)>::kit_t,
                        entries_kit_of<residency_>,
                        typename encoder::elementwise_ladder<numkong_capabilities_of(cpu_ladder_floor_k)>::kit_t>;

/** Whether a class picks kits its parent does not; the serial floor always does, and so does a
 *  class whose parent the build left out, like the oldest GPU class or a monoversioned build's one
 *  class. A host class takes its own atomic reference on both sides, as each names it through an
 *  alias template of its own. */
template <vendored_residency_t residency_, auto class_>
consteval bool class_advances() noexcept {
    if constexpr (residency_ == vendored_residency_t::ram_k) {
        if constexpr (class_ == cpu_class_t::serial_k || !cpu_class_compiled(parent_class_of(class_))) return true;
        else return !std::is_same_v<ram_tower_of<class_>, ram_tower_of<parent_class_of(class_), class_>>;
    }
    else if constexpr (!gpu_class_compiled(parent_class_of(class_))) return true;
    else return !std::is_same_v<vram_tower_of<residency_, class_>, vram_tower_of<residency_, parent_class_of(class_)>>;
}

#pragma endregion Residencies

#pragma region Bodies

/** The verbs of one host tower type, every one a static member of a template over that type, so no
 *  body compiled in one class unit merges into another's. */
template <typename tower_type_>
struct ram_bodies {
    static util::expected<void*> create(util::checkpoint_to_load_t const& checkpoint,
                                        encoder::geometry_t const& geometry, encoder::precision_t precision,
                                        nk_stream_t) noexcept {
        auto [tower, status] = tower_type_::make(checkpoint, geometry, precision);
        if (!status) return util::expected<void*>::fail(std::move(status));
        tower_type_* const handle = new (std::nothrow) tower_type_(std::move(tower));
        if (!handle) return util::expected<void*>::fail(util::status_code_t::out_of_memory_k, "No room for the tower");
        return util::expected<void*> {handle};
    }

    static void destroy(void* tower) noexcept { delete static_cast<tower_type_*>(tower); }

    static util::expected<std::size_t> scratch_bytes(void const* tower, std::size_t tokens, std::size_t segments,
                                                     util::pool_view_t pool) noexcept {
        return static_cast<tower_type_ const*>(tower)->scratch_bytes(tokens, segments, pool.threads_count());
    }

    static util::expected_t encode(void const* tower, encoder::multimodal_batch_view_t batch,
                                   encoder::outputs_t outputs, util::bytes_span_t scratch,
                                   util::pool_view_t pool) noexcept {
        return static_cast<tower_type_ const*>(tower)->encode(batch, outputs, scratch, pool);
    }
};

/** The verbs of one device tower type, packing in memory its vendor's entries kit allocates and
 *  waiting through that kit for the packs before the tower is handed out. */
template <typename tower_type_>
struct vram_bodies {
    using entries_kit_t = typename tower_type_::steps_t::entries_kit_t;

    static util::expected<void*> create(util::checkpoint_to_load_t const& checkpoint,
                                        encoder::geometry_t const& geometry, encoder::precision_t precision,
                                        nk_stream_t stream) noexcept {
        nk_allocator_t const policy = entries_kit_t::policy(stream, uform_residency_vram_k);
        if (!policy.allocate)
            return util::expected<void*>::fail(util::status_code_t::device_k,
                                               "Tower memory setup failed: no device allocator");
        auto [tower, status] = tower_type_::make(checkpoint, geometry, precision, policy, stream);
        if (!status) return util::expected<void*>::fail(std::move(status));
        // The packs and the operands staged for them land before the tower is handed out
        nk_status_t const landed = entries_kit_t::synchronize(stream);
        if (landed != nk_success_k)
            return util::expected<void*>::fail(encoder::tower_status_of(landed), "load failed: {}",
                                               nk_status_name(landed));
        tower_type_* const handle = new (std::nothrow) tower_type_(std::move(tower));
        if (!handle) return util::expected<void*>::fail(util::status_code_t::out_of_memory_k, "No room for the tower");
        return util::expected<void*> {handle};
    }

    static void destroy(void* tower) noexcept { delete static_cast<tower_type_*>(tower); }

    static util::expected<std::size_t> scratch_bytes(void const* tower, std::size_t tokens, std::size_t segments,
                                                     nk_stream_t) noexcept {
        return static_cast<tower_type_ const*>(tower)->scratch_bytes(tokens, segments);
    }

    static util::expected_t encode(void const* tower, encoder::multimodal_batch_view_t batch,
                                   encoder::outputs_t outputs, util::bytes_span_t scratch,
                                   nk_stream_t stream) noexcept {
        return static_cast<tower_type_ const*>(tower)->encode(batch, outputs, scratch, stream);
    }
};

template <vendored_residency_t residency_, auto class_>
encoder_verbs<residency_> const* encoder_class_tables<residency_, class_>::find() noexcept {
    if constexpr (residency_ == vendored_residency_t::ram_k) {
        if constexpr (!class_advances<residency_, class_>()) return nullptr;
        else {
            using bodies_t = ram_bodies<ram_tower_of<class_>>;
            static constexpr encoder_verbs<residency_> verbs_k {class_,
                                                                gpu_class_t::unknown_k,
                                                                &bodies_t::create,
                                                                &bodies_t::destroy,
                                                                &bodies_t::scratch_bytes,
                                                                &bodies_t::encode};
            return &verbs_k;
        }
    }
    else {
        if constexpr (!class_advances<residency_, class_>()) return nullptr;
        else {
            using bodies_t = vram_bodies<vram_tower_of<residency_, class_>>;
            static constexpr encoder_verbs<residency_> verbs_k {cpu_class_t::unknown_k,   class_,
                                                                &bodies_t::create,        &bodies_t::destroy,
                                                                &bodies_t::scratch_bytes, &bodies_t::encode};
            return &verbs_k;
        }
    }
}

template <cpu_class_t cpu_class_>
void host_class_tables<cpu_class_>::wait(util::u64_t& word, util::u64_t seen) noexcept {
    using atomic_ref_t = typename atomic_ref_for_cpu_class<cpu_class_>::template type<util::u64_t>;
    util::u64_t const* const watched = &word;
    while (atomic_ref_t {word}.load(std::memory_order_acquire) == seen)
        waiter_of<cpu_class_> {}(watched, seen, std::size_t {0}, fu::wait_capped_k);
}

template <cpu_class_t cpu_class_>
host_verbs_t const* host_class_tables<cpu_class_>::find() noexcept {
    constexpr cpu_class_t parent_k = parent_class_of(cpu_class_);
    static_assert(class_advances<vendored_residency_t::ram_k, cpu_class_>() ||
                      !std::is_same_v<waiter_of<cpu_class_>, waiter_of<parent_k>> ||
                      !std::is_same_v<typename atomic_ref_for_cpu_class<cpu_class_>::template type<util::u64_t>,
                                      typename atomic_ref_for_cpu_class<parent_k>::template type<util::u64_t>>,
                  "A compiled class no longer advances past the one below");
    static constexpr host_verbs_t verbs_k {&host_class_tables::wait};
    return &verbs_k;
}

#pragma endregion Bodies

} // namespace unum::uform
