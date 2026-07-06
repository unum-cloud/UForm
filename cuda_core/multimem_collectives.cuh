/**
 *  @brief Collectives @b inside one multimem_domain: the domain's members reduce and broadcast through
 *      @c multimem PTX over one shared multicast object — the level-1 transport of the two-level
 *      contrastive step and the distributed optimizer.
 *  @file cuda/primitives/multimem_collectives.cuh
 *  @author Ash Vardanian
 *  @date July 3, 2026
 *
 *  The domain's replicas live in one @ref gpu_symmetric_arena_t : a single contiguous VA range whose
 *  per-member slices are physically resident on each member's device, so a peer's address is
 *  @c base+member·stride — plain arithmetic, no pointer tables anywhere. Every member additionally binds
 *  its slice into one multicast object, and one multicast address reaches all replicas at once. That
 *  turns the collective the loss and optimizer steps need — the @b reduce-scatter — into a single
 *  instruction per element: a member issues @c multimem.ld_reduce.add.f32 over its shard of the
 *  multicast range, and the NVSwitch returns the @b sum across every member's replica in one load,
 *  deterministic by construction (one fused reduction, no ordering to fix).
 *
 *  Synchronization is the same monotonic-round discipline the InfiniBand collectives use: an arrival
 *  flag lives inside every replica, a member @b adds 1 through the multicast address (so every replica's
 *  counter grows), and waiters acquire-spin on their @b own replica until the round target — no resets,
 *  so captured steps replay without host writes. The round scalar is the domain's own, inside the lead
 *  replica, advanced once per step through @ref multimem_domain_t::advance .
 *
 *  When the platform lacks multicast ( @c CU_DEVICE_ATTRIBUTE_MULTICAST_SUPPORTED off — pre-Hopper or
 *  fabric-disabled boxes), @ref multimem_domain_t::make still succeeds in the @b peer-read tier: the
 *  symmetric range is device-visible everywhere, and the reduce walks the members serially — the same
 *  fixed fold order on every member, bit-identical across ranks, just without the switch reduction.
 */
#pragma once

#include <cstddef> // `std::size_t`

#include <span> // `std::span` — device-ID and stream lists

#include <cuda.h> // `cuMulticastCreate`, `cuMulticastBindMem`, `cuMemMap`, `cuMemSetAccess`

#include "cuda_core/types.hpp"          // `expected`, `expected_t`, `status_t`, the scalar aliases
#include "cuda_core/array.hpp"          // `array` — the runtime-sized per-member tables
#include "cuda_core/gpu_allocators.hpp" // `gpu_arena_t` conventions, `round_up_to_multiple`
#include "cuda_core/schedule.cuh"       // `gpu_stream_t`, `launch_kernel`, `current_context_t`, the rank-1 aliases

namespace unum::uform::core {

/**
 *  @brief The symmetric memory of one multimem_domain: one contiguous VA range whose per-member slices
 *      are physically resident on each member's device, every slice at @c base+member·stride — so a
 *      peer's address is arithmetic, never a table. Fixed-size, allocated once at @ref make ; no
 *      carving, no growth. Lives here, beside its one consumer @ref multimem_domain_t , not with the
 *      single-GPU allocators.
 *
 *  The slices' physical handles are created shareable ( @c POSIX_FILE_DESCRIPTOR ) and are exposed
 *  through @ref member_handle so a multicast object can bind them — the one capability @ref gpu_arena_t
 *  chunks deliberately lack. Every member device is granted access to the whole range (the peer-read
 *  tier and debugging read straight through it).
 */
struct gpu_symmetric_arena_t {

    gpu_symmetric_arena_t() noexcept = default;
    gpu_symmetric_arena_t(gpu_symmetric_arena_t const &) = delete;
    gpu_symmetric_arena_t &operator=(gpu_symmetric_arena_t const &) = delete;
    gpu_symmetric_arena_t(gpu_symmetric_arena_t &&other) noexcept { swap_(other); }
    gpu_symmetric_arena_t &operator=(gpu_symmetric_arena_t &&other) noexcept {
        if (this != &other) {
            destroy_();
            swap_(other);
        }
        return *this;
    }
    ~gpu_symmetric_arena_t() noexcept { destroy_(); }

    /**
     *  @brief Reserve and map the whole symmetric range — once, at setup, with a current context.
     *
     *  @param[in] devices Member device ordinals, slice order.
     *  @param[in] bytes_per_member Usable bytes of one member's slice.
     *  @param[in] alignment_granularity An extra stride-alignment requirement (e.g. the multicast
     *      granularity the caller queried); the stride also honors every member's physical granularity.
     */
    [[nodiscard]] static expected<gpu_symmetric_arena_t> make(std::span<i32_t const> devices,
                                                              std::size_t bytes_per_member,
                                                              std::size_t alignment_granularity = 1) noexcept;

    /** @brief Member count, matching the @c devices order of @ref make . */
    [[nodiscard]] std::size_t member_count() const noexcept { return devices_.size(); }
    /** @brief Member @p member 's device ordinal. */
    [[nodiscard]] i32_t member_device(std::size_t member) const noexcept { return devices_[member]; }
    /** @brief Byte distance between adjacent members' slices — the peer-address arithmetic constant. */
    [[nodiscard]] std::size_t member_stride() const noexcept { return member_stride_; }
    /** @brief The whole range's base — member @p m 's slice starts at @c base()+m·member_stride() . */
    [[nodiscard]] std::byte *base() noexcept { return reinterpret_cast<std::byte *>(base_); }
    /** @brief Member @p member 's usable slice. */
    [[nodiscard]] std::span<std::byte> member_slice(std::size_t member) noexcept {
        return {reinterpret_cast<std::byte *>(base_ + member * member_stride_), bytes_per_member_};
    }
    /** @brief Member @p member 's shareable physical handle — what a multicast object binds. */
    [[nodiscard]] CUmemGenericAllocationHandle member_handle(std::size_t member) const noexcept {
        return handles_[member];
    }

  private:
    /** @brief Slice-creation properties: device-resident, pinned, RDMA-capable, and shareable — the
     *      last one is what multicast binding requires. */
    [[nodiscard]] static CUmemAllocationProp allocation_properties_(int device) noexcept {
        CUmemAllocationProp properties {};
        properties.type = CU_MEM_ALLOCATION_TYPE_PINNED;
        properties.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
        properties.location.id = device;
        properties.allocFlags.gpuDirectRDMACapable = 1;
        properties.requestedHandleTypes = CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR;
        return properties;
    }

    void swap_(gpu_symmetric_arena_t &other) noexcept {
        std::swap(base_, other.base_);
        std::swap(bytes_per_member_, other.bytes_per_member_);
        std::swap(member_stride_, other.member_stride_);
        std::swap(devices_, other.devices_);
        std::swap(handles_, other.handles_);
    }

    void destroy_() noexcept {
        if (base_ == 0) return;
        // A published handle implies a mapped slice — `make` publishes only after the map succeeds.
        for (std::size_t member = 0; member != handles_.size(); ++member) {
            if (handles_[member] == 0) continue;
            cuMemUnmap(base_ + member * member_stride_, member_stride_);
            cuMemRelease(handles_[member]);
        }
        cuMemAddressFree(base_, member_stride_ * devices_.size());
        base_ = 0;
        devices_.reset();
        handles_.reset();
    }

    /** @brief Start of the one contiguous reservation covering every member's slice. */
    CUdeviceptr base_ = 0;
    /** @brief Usable bytes per slice, as requested at @ref make . */
    std::size_t bytes_per_member_ = 0;
    /** @brief Granularity-rounded slice stride; also each slice's physical size. */
    std::size_t member_stride_ = 0;
    /** @brief Member devices, slice order. */
    array<i32_t> devices_ {};
    /** @brief The shareable physical handles, one per member. */
    array<CUmemGenericAllocationHandle> handles_ {};
};

/** @brief Which reduction tier a @ref multimem_domain_t runs: NVSwitch-fused @c multimem loads, or the
 *      serial peer-read fold — decided once at @c make by @c CU_DEVICE_ATTRIBUTE_MULTICAST_SUPPORTED ,
 *      identical results either way (both tiers are bit-deterministic). */
enum class multimem_tier_t : u32_t {
    switch_reduced_k = 0,
    peer_read_k = 1,
};

/**
 *  @brief One member's addresses inside the domain — the scalar-only POD kernels take by value.
 *
 *  @c multimem_* pointers fan out to every replica (null on the peer-read tier); everything else is
 *  arithmetic over the one contiguous replica range: member @c m 's replica starts at
 *  @c replicas+m·replica_stride , its arrival flag at @c +flag_offset . All addresses are capture-stable
 *  for the domain's lifetime.
 */
struct multimem_member_t {
    /** @brief Multicast base of the Float32 payload — `multimem` loads and stores fan out here. */
    f32_t *multimem_payload = nullptr;
    /** @brief Multicast address of the arrival counter. */
    u32_t *multimem_flag = nullptr;
    /** @brief Base of the contiguous symmetric range covering every member's replica. */
    std::byte *replicas = nullptr;
    /** @brief The domain's round scalar (inside the lead replica) every barrier keys on. */
    u32_t const *round = nullptr;
    /** @brief Byte distance between adjacent members' replicas. */
    std::size_t replica_stride = 0;
    /** @brief Byte offset of the arrival counter inside every replica. */
    std::size_t flag_offset = 0;
    /** @brief This member's index — whose replica is "own". */
    u32_t member_index = 0;
    /** @brief The domain's member count — the barrier's arrival target and the peer fold's extent. */
    u32_t member_count = 0;

#if defined(__CUDACC__)
    /** @brief Member @p member 's payload replica base. */
    [[nodiscard]] __device__ f32_t *peer_payload(u32_t member) const noexcept {
        return reinterpret_cast<f32_t *>(replicas + member * replica_stride);
    }
    /** @brief Member @p member 's arrival counter. */
    [[nodiscard]] __device__ u32_t *peer_flag(u32_t member) const noexcept {
        return reinterpret_cast<u32_t *>(replicas + member * replica_stride + flag_offset);
    }
    /** @brief This member's own arrival counter — the address the barrier spins on. */
    [[nodiscard]] __device__ u32_t const *own_flag() const noexcept { return peer_flag(member_index); }
#endif
};

#if defined(__CUDACC__)

/** @brief One switch-reduced load: the sum of this address across every bound replica. */
[[nodiscard]] __device__ inline f32_t multimem_load_reduce_add_(f32_t const *multimem_address) noexcept {
    f32_t value;
    asm volatile("multimem.ld_reduce.relaxed.sys.global.add.f32 %0, [%1];"
                 : "=f"(value)
                 : "l"(multimem_address)
                 : "memory");
    return value;
}

/** @brief One multicast release-add on the arrival counter of every replica. */
__device__ inline void multimem_red_add_release_(u32_t *multimem_address, u32_t value) noexcept {
    asm volatile("multimem.red.release.sys.global.add.u32 [%0], %1;" : : "l"(multimem_address), "r"(value) : "memory");
}

/** @brief System-scope acquire load of this member's own arrival counter. */
[[nodiscard]] __device__ inline u32_t multimem_load_acquire_(u32_t const *own_address) noexcept {
    u32_t value;
    asm volatile("ld.acquire.sys.global.u32 %0, [%1];" : "=r"(value) : "l"(own_address) : "memory");
    return value;
}

#endif // defined(__CUDACC__)

/** @brief A consumer's disjoint window of the domain payload: Float32 scalars
 *      @b [offset,offset+scalars) inside every replica — handed out by @ref multimem_domain_t::reserve ,
 *      so consumers' reduces never alias by construction. */
struct multimem_window_t {
    std::size_t offset = 0;
    std::size_t scalars = 0;
};

/**
 *  @brief One multimem_domain's collective surface: every member's same-sized Float32 replica inside one
 *      @ref gpu_symmetric_arena_t , bound into one multicast object — the replicas @b are whatever force
 *      or gradient surface the consumers reduce over.
 *
 *  Created @b once in main, sized for every consumer's window (each exposes a
 *  @c payload_scalars_needed helper), and threaded by reference — multicast objects are a scarce
 *  fabric resource, so consumers @ref reserve disjoint windows instead of wiring their own domains.
 *
 *  @section multimem_ordering The barrier ordering contract
 *
 *  The arrival flag and the round are shared by every consumer, so all ranks must issue their
 *  @ref advance / @ref barrier sequences in the @b same order across consumers within a step — the
 *  natural property of an SPMD step program. Violating it is a hang (arrivals are monotonic, a
 *  mis-paired barrier waits forever), never a silent corruption.
 *
 *  Falls back to the peer-read tier when multicast is unsupported — same API, same determinism, the
 *  reduce just walks members serially. @ref tier() names which one is live.
 */
struct multimem_domain_t {

  private:
    multimem_domain_t() noexcept = default;

  public:
    multimem_domain_t(multimem_domain_t const &) = delete;
    multimem_domain_t &operator=(multimem_domain_t const &) = delete;
    multimem_domain_t(multimem_domain_t &&other) noexcept { swap_(other); }
    multimem_domain_t &operator=(multimem_domain_t &&other) noexcept {
        if (this != &other) {
            destroy_();
            swap_(other);
        }
        return *this;
    }
    ~multimem_domain_t() noexcept { destroy_(); }

    /**
     *  @brief Build the domain over @p devices with @p payload_scalars Float32 per replica: the
     *      symmetric replica range, the multicast object binding every slice, one multicast mapping,
     *      and the zeroed arrival counters. Runs once, at setup, outside any capture. The barrier
     *      round is the domain's own: a scalar inside the lead replica, advanced once per step
     *      through @ref advance — consumers never allocate or upload rounds themselves.
     */
    [[nodiscard]] static expected<multimem_domain_t> make(std::span<i32_t const> devices, std::size_t payload_scalars,
                                                          std::span<gpu_stream_t> streams) noexcept;

    /** @brief The reduction tier this domain runs — see @ref multimem_tier_t . */
    [[nodiscard]] multimem_tier_t tier() const noexcept {
        return multimem_handle_ != 0 ? multimem_tier_t::switch_reduced_k : multimem_tier_t::peer_read_k;
    }
    /** @brief Domain member count. */
    [[nodiscard]] std::size_t member_count() const noexcept { return replicas_.member_count(); }
    /** @brief Float32 payload scalars per replica. */
    [[nodiscard]] std::size_t payload_scalars() const noexcept { return payload_scalars_; }
    /** @brief Member @p member 's own payload replica — where its contribution is written. */
    [[nodiscard]] f32r1_span_t payload(std::size_t member) noexcept {
        return {reinterpret_cast<f32_t *>(replicas_.member_slice(member).data()), payload_scalars_};
    }

    /** @brief Reserve @p scalars of the payload for one consumer — bump-reserve at construction time,
     *      loud failure past capacity; windows are disjoint by construction. Mirrors the arena
     *      discipline: no per-window free, the domain outlives its consumers. */
    [[nodiscard]] expected<multimem_window_t> reserve(std::size_t scalars) noexcept {
        if (reserved_scalars_ + scalars > payload_scalars_)
            return expected<multimem_window_t>::fail(
                status_t::bad_alloc_k, "window reservation of {} scalars exceeds the domain's {} ({} reserved)",
                scalars, payload_scalars_, reserved_scalars_);
        multimem_window_t const window {reserved_scalars_, scalars};
        reserved_scalars_ += scalars;
        return expected<multimem_window_t> {window};
    }

    /** @brief The one enqueued host write of a step — see @ref barrier_round_t::advance . Lead
     *      stream, before any member's barrier. */
    [[nodiscard]] expected_t advance(gpu_stream_t &stream) noexcept { return round_.advance(stream); }

    /** @brief Member @p member_index 's kernel-facing addresses. */
    [[nodiscard]] multimem_member_t member(std::size_t member_index) noexcept {
        std::size_t const flag_offset = flag_offset_();
        multimem_member_t addresses;
        if (multimem_pointer_ != 0) {
            addresses.multimem_payload = reinterpret_cast<f32_t *>(multimem_pointer_);
            addresses.multimem_flag = reinterpret_cast<u32_t *>(multimem_pointer_ + flag_offset);
        }
        addresses.replicas = replicas_.base();
        addresses.round = round_pointer_();
        addresses.replica_stride = replicas_.member_stride();
        addresses.flag_offset = flag_offset;
        addresses.member_index = static_cast<u32_t>(member_index);
        addresses.member_count = static_cast<u32_t>(member_count());
        return addresses;
    }

    /** @brief Enqueue the domain barrier for @p member_index — one arrival, one acquire-spin against
     *      the domain's own round (see @ref advance ). On the peer tier the arrival walks the members
     *      through the symmetric range instead. */
    [[nodiscard]] expected_t barrier(std::size_t member_index, gpu_stream_t &stream) noexcept;

    /** @brief Enqueue member @p member_index 's reduce-scatter shard: @p destination receives the
     *      domain-wide sum of @c [shard_begin,shard_begin+destination.extent(0)) . Call after the
     *      barrier proved every member's payload complete. */
    [[nodiscard]] expected_t reduce_scatter(std::size_t member_index, f32r1_span_t destination, std::size_t shard_begin,
                                            gpu_stream_t &stream) noexcept;

  private:
    /** @brief Byte offset of the arrival counter inside every replica — pure layout arithmetic over
     *      @ref payload_scalars_ , never stored; the round sits one word above. */
    [[nodiscard]] std::size_t flag_offset_() const noexcept {
        return round_up_to_multiple(payload_scalars_ * sizeof(f32_t), cache_line_bytes_k);
    }

    /** @brief The barrier round scalar — GPU memory inside the lead replica, one word above the
     *      flag; derived, never stored, so no second copy of the layout exists. */
    [[nodiscard]] u32_t *round_pointer_() noexcept {
        std::span<std::byte> lead_slice = replicas_.member_slice(0);
        return reinterpret_cast<u32_t *>(lead_slice.data() + flag_offset_() + sizeof(u32_t));
    }

    void swap_(multimem_domain_t &other) noexcept {
        std::swap(replicas_, other.replicas_);
        std::swap(multimem_handle_, other.multimem_handle_);
        std::swap(multimem_pointer_, other.multimem_pointer_);
        std::swap(round_, other.round_);
        std::swap(payload_scalars_, other.payload_scalars_);
        std::swap(reserved_scalars_, other.reserved_scalars_);
    }

    /** @brief Reverse the multicast machinery; the symmetric arena member tears itself down after. */
    void destroy_() noexcept {
        std::size_t const bound_bytes = replicas_.member_stride();
        if (multimem_pointer_ != 0) {
            cuMemUnmap(multimem_pointer_, bound_bytes);
            cuMemAddressFree(multimem_pointer_, bound_bytes);
            multimem_pointer_ = 0;
        }
        if (multimem_handle_ != 0) {
            for (std::size_t member = 0; member != replicas_.member_count(); ++member) {
                CUdevice device_handle = 0;
                if (cuDeviceGet(&device_handle, replicas_.member_device(member)) == CUDA_SUCCESS)
                    cuMulticastUnbind(multimem_handle_, device_handle, 0, bound_bytes);
            }
            cuMemRelease(multimem_handle_);
            multimem_handle_ = 0;
        }
    }

    /** @brief The one contiguous replica range — every member's slice, peer-addressable by arithmetic. */
    gpu_symmetric_arena_t replicas_ {};
    /** @brief The multicast object every slice is bound into; 0 on the peer-read tier. */
    CUmemGenericAllocationHandle multimem_handle_ = 0;
    /** @brief The one multicast VA shared by all members; 0 on the peer-read tier. */
    CUdeviceptr multimem_pointer_ = 0;
    /** @brief The domain's monotonic step counter — host count, pinned staging, and the device
     *      round scalar it uploads into; see @ref barrier_round_t . */
    barrier_round_t round_ {};
    /** @brief Float32 payload scalars per replica. */
    std::size_t payload_scalars_ = 0;
    /** @brief The window-reservation cursor; see @ref reserve . */
    std::size_t reserved_scalars_ = 0;
};

} // namespace unum::uform::core
