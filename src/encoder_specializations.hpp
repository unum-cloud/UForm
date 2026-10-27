/**
 *  @file src/encoder_specializations.hpp
 *  @author Ash Vardanian
 *  @date October 25, 2026
 *  @brief The one place encoder variants are enumerated: the towers a class spells its variant
 *      with, the static bodies each variant's verb table points at, and the class tables a class
 *      unit instantiates once.
 *
 *  Only class units include it. A host class picks its kits from NumKong's mask for the class, a
 *  device class from its generation's mask, and a vendor floor from the vendor's generic kits. A
 *  class whose kits match its parent's answers no table, so the ladder falls to the parent.
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

/** The mask a device class picks its kits under: its generation's, or its vendor's generic
 *  floor. */
template <encoder_home_t home_, gpu_class_t gpu_class_>
inline constexpr nk_capability_t device_capabilities_of = gpu_class_ == gpu_class_t::unknown_k
                                                              ? vendor_of(home_)
                                                              : numkong_capabilities_of(gpu_class_);

/** The device tower one GPU class runs. Its unit compiles for the serial CPU class, so its load
 *  stages through that class's element-wise kit. */
template <encoder_home_t home_, gpu_class_t gpu_class_>
using vram_tower_of =
    encoder::vram_tower<typename encoder::vram_products_ladder<device_capabilities_of<home_, gpu_class_>>::kit_t,
                        typename encoder::vram_attention_ladder<device_capabilities_of<home_, gpu_class_>>::kit_t,
                        typename encoder::vram_elementwise_ladder<device_capabilities_of<home_, gpu_class_>>::kit_t,
                        entries_kit_of<home_>,
                        typename encoder::elementwise_ladder<numkong_capabilities_of(cpu_class_t::serial_k)>::kit_t>;

/** Whether a host class picks kits its parent does not; the serial floor always does. Both sides
 *  take the class's own atomic reference, as each class names it through an alias template of its
 *  own. */
template <cpu_class_t cpu_class_>
constexpr bool ram_variant_advances() noexcept {
    if constexpr (cpu_class_ == cpu_class_t::serial_k) return true;
    else return !std::is_same_v<ram_tower_of<cpu_class_>, ram_tower_of<parent_class_of(cpu_class_), cpu_class_>>;
}

/** Whether a device class picks a tower its parent does not; the vendor floor always does. */
template <encoder_home_t home_, gpu_class_t gpu_class_>
constexpr bool vram_variant_advances() noexcept {
    if constexpr (gpu_class_ == gpu_class_t::unknown_k) return true;
    else return !std::is_same_v<vram_tower_of<home_, gpu_class_>, vram_tower_of<home_, parent_class_of(gpu_class_)>>;
}

#pragma endregion Residencies

#pragma region Bodies

/** The verbs of one host tower type, every one a static member of a template over that type, so no
 *  body compiled in one class unit merges into another's. */
template <typename tower_type_>
struct ram_bodies {
    static expected<void *> create(checkpoint_to_load_t const &checkpoint, encoder::geometry_t const &geometry,
                                   encoder::precision_t precision, nk_stream_t) noexcept {
        expected<tower_type_> made = tower_type_::make(checkpoint, geometry, precision);
        if (!made) return expected<void *>::fail(std::move(made));
        tower_type_ *const tower = new (std::nothrow) tower_type_(std::move(*made));
        if (!tower) return expected<void *>::fail(status_t::bad_alloc_k, "tower allocation failed");
        return expected<void *> {tower};
    }

    static void destroy(void *tower) noexcept { delete static_cast<tower_type_ *>(tower); }

    static expected<std::size_t> scratch_bytes(void const *tower, std::size_t tokens, std::size_t segments,
                                               pool_view_t pool) noexcept {
        return static_cast<tower_type_ const *>(tower)->scratch_bytes(tokens, segments, pool.threads_count());
    }

    static expected_t encode(void const *tower, encoder::multimodal_batch_view_t batch, encoder::outputs_t outputs,
                             bytes_span_t scratch, pool_view_t pool) noexcept {
        return static_cast<tower_type_ const *>(tower)->encode(batch, outputs, scratch, pool);
    }
};

/** The verbs of one device tower type, packing in memory NumKong's device allocator places for
 *  @p capabilities_ and waiting for the packs before the tower is handed out. */
template <typename tower_type_, nk_capability_t capabilities_>
struct vram_bodies {
    static expected<void *> create(checkpoint_to_load_t const &checkpoint, encoder::geometry_t const &geometry,
                                   encoder::precision_t precision, nk_stream_t stream) noexcept {
        nk_allocator_t policy {};
        if (nk_status_t const status = nk_allocator_init_device_best(&policy, capabilities_); status != nk_success_k)
            return expected<void *>::fail(encoder::tower_status_of(status), "tower memory setup failed: {}",
                                          nk_status_name(status));
        expected<tower_type_> made = tower_type_::make(checkpoint, geometry, precision, policy, stream);
        if (!made) return expected<void *>::fail(std::move(made));
        nk_status_t const landed = nk_stream_synchronize_best(capabilities_, stream); // the packs and staged operands
        if (landed != nk_success_k)
            return expected<void *>::fail(encoder::tower_status_of(landed), "load failed: {}", nk_status_name(landed));
        tower_type_ *const tower = new (std::nothrow) tower_type_(std::move(*made));
        if (!tower) return expected<void *>::fail(status_t::bad_alloc_k, "tower allocation failed");
        return expected<void *> {tower};
    }

    static void destroy(void *tower) noexcept { delete static_cast<tower_type_ *>(tower); }

    static expected<std::size_t> scratch_bytes(void const *tower, std::size_t tokens, std::size_t segments,
                                               nk_stream_t) noexcept {
        return static_cast<tower_type_ const *>(tower)->scratch_bytes(tokens, segments);
    }

    static expected_t encode(void const *tower, encoder::multimodal_batch_view_t batch, encoder::outputs_t outputs,
                             bytes_span_t scratch, nk_stream_t stream) noexcept {
        return static_cast<tower_type_ const *>(tower)->encode(batch, outputs, scratch, stream);
    }
};

template <encoder_home_t home_, auto class_>
encoder_verbs<home_> const *encoder_class_tables<home_, class_>::find() noexcept {
    if constexpr (home_ == encoder_home_t::ram_k) {
        if constexpr (!ram_variant_advances<class_>()) return nullptr;
        else {
            using bodies_t = ram_bodies<ram_tower_of<class_>>;
            static constexpr encoder_verbs<home_> verbs_k {class_,
                                                           gpu_class_t::unknown_k,
                                                           &bodies_t::create,
                                                           &bodies_t::destroy,
                                                           &bodies_t::scratch_bytes,
                                                           &bodies_t::encode};
            return &verbs_k;
        }
    }
    else {
        if constexpr (!vram_variant_advances<home_, class_>()) return nullptr;
        else {
            using bodies_t = vram_bodies<vram_tower_of<home_, class_>, device_capabilities_of<home_, class_>>;
            static constexpr encoder_verbs<home_> verbs_k {cpu_class_t::unknown_k,   class_,
                                                           &bodies_t::create,        &bodies_t::destroy,
                                                           &bodies_t::scratch_bytes, &bodies_t::encode};
            return &verbs_k;
        }
    }
}

template <cpu_class_t cpu_class_>
void host_class_tables<cpu_class_>::wait(u64_t &word, u64_t seen) noexcept {
    using atomic_ref_t = typename atomic_ref_for_cpu_class<cpu_class_>::template type<u64_t>;
    u64_t const *const watched = &word;
    while (atomic_ref_t {word}.load(std::memory_order_acquire) == seen)
        waiter_of<cpu_class_> {}(watched, seen, std::size_t {0}, fu::wait_capped_k);
}

template <cpu_class_t cpu_class_>
host_verbs_t const *host_class_tables<cpu_class_>::find() noexcept {
    static constexpr host_verbs_t verbs_k {&host_class_tables::wait};
    return &verbs_k;
}

#pragma endregion Bodies

} // namespace unum::uform
