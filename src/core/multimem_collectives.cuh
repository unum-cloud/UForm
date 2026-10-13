/**
 *  @brief Collectives @b inside one multimem_domain: the domain's members reduce and broadcast through
 *      @c multimem PTX over one shared multicast object — the level-1 transport of the two-level
 *      contrastive step and the distributed optimizer.
 *  @file src/core/multimem_collectives.cuh
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
 *  so captured steps replay without host writes. Each member keys the target on its @b own round scalar,
 *  one word above its arrival flag, and its barrier kernel self-advances it after the spin — every read
 *  and write of a round happens on the stream that owns it, so no enqueue order can present a barrier
 *  with a stale round, and any number of barriers compose with no pairing discipline.
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

#include "core/types.hpp"          // `expected`, `expected_t`, `status_t`, the scalar aliases
#include "core/array.hpp"          // `array` — the runtime-sized per-member tables
#include "core/gpu_allocators.hpp" // `gpu_arena_t` conventions, `round_up_to_multiple`
#include "core/schedule.cuh"       // `gpu_stream_t`, `launch_kernel`, `current_context_t`, the rank-1 aliases
#include "core/mesh.hpp"           // `mesh_topology_t` — the multimem_domain may span processes; the fabric
                                   //     handles ride the mesh rendezvous

namespace unum::uform::core {

/**
 *  @brief The symmetric memory of one multimem_domain: one contiguous VA range whose per-member slices
 *      are physically resident on each member's device, every slice at @c base+member·stride — so a
 *      peer's address is arithmetic, never a table.
 *
 *  Fixed-size, allocated once at @ref make ; no carving, no growth. Lives here, beside its one consumer
 *  @ref multimem_domain_t , not with the single-GPU allocators.
 *
 *  A multimem_domain may span @b more @b than @b one @b process — on an NVL72 the 72-GPU island is 18
 *  tray processes, and even a `{4,8,16}` HGX box runs two 4-GPU processes over one 8-GPU island. So the
 *  arena is @b per @b process: it reserves the whole @c member_count·stride VA, but each process
 *  @ref make -s only its @b local slices (physically resident on the GPUs it drives) and @ref adopt_peer -s
 *  every remote member's slice from the fabric handle its owner published. The slices' physical handles
 *  are created @c CU_MEM_HANDLE_TYPE_FABRIC — the NVLink-fabric shareable form NVL72 needs and a single
 *  passthrough box also honors — exported through @ref export_local_fabric_handle and exposed through
 *  @ref member_handle so a multicast object can bind them. Every local device is granted access to the
 *  whole range, so the peer-read tier reads every replica straight through it.
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
     *  @brief Reserve the whole @c member_count·stride symmetric range and create @b this process's local
     *      slices — once, at setup, with a current context. Remote members' slices are filled afterwards
     *      through @ref adopt_peer .
     *
     *  @param[in] local_devices The device ordinals this process drives — the slices it creates; they
     *      occupy domain members @c [member_offset,member_offset+local_devices.size()) .
     *  @param[in] member_count The whole multimem_domain's member count (may exceed @p local_devices ).
     *  @param[in] member_offset Where this process's slices sit inside the domain — the first member index.
     *  @param[in] bytes_per_member Usable bytes of one member's slice.
     *  @param[in] alignment_granularity An extra stride-alignment requirement (e.g. the multicast
     *      granularity the caller queried); the stride also honors every local device's physical granularity.
     */
    [[nodiscard]] static expected<gpu_symmetric_arena_t> make(std::span<i32_t const> local_devices, u32_t member_count,
                                                              u32_t member_offset, std::size_t bytes_per_member,
                                                              std::size_t alignment_granularity = 1) noexcept;

    /** Export domain member @p member 's slice as a fabric handle — what its owner publishes so the
     *  domain's other processes can @ref adopt_peer it. @p member must be one this process created. */
    [[nodiscard]] expected<CUmemFabricHandle> export_fabric_handle(std::size_t member) const noexcept;

    /** Import a remote member's slice from the fabric @p handle its owner exported and map it at
     *  domain member @p member 's offset — the peer half of the cross-process symmetric range. */
    [[nodiscard]] expected_t adopt_peer(std::size_t member, CUmemFabricHandle const &handle) noexcept;

    /** The whole multimem_domain's member count — the symmetric range's extent; the handle table's
     *  size @b is this, so it is derived, never stored. */
    [[nodiscard]] std::size_t member_count() const noexcept { return handles_.size(); }

    /** How many of them this process created locally. */
    [[nodiscard]] std::size_t local_member_count() const noexcept { return devices_.size(); }

    /** Local slice @p local_index 's device ordinal (this process's own devices, slice order). */
    [[nodiscard]] i32_t local_device(std::size_t local_index) const noexcept { return devices_[local_index]; }

    /** Byte distance between adjacent members' slices — the peer-address arithmetic constant. */
    [[nodiscard]] std::size_t member_stride() const noexcept { return member_stride_; }

    /** The whole range's base — member @p m 's slice starts at @c base()+m·member_stride() . */
    [[nodiscard]] std::byte *base() noexcept { return reinterpret_cast<std::byte *>(base_); }

    /** Member @p member 's usable slice. */
    [[nodiscard]] std::span<std::byte> member_slice(std::size_t member) noexcept {
        return {reinterpret_cast<std::byte *>(base_ + member * member_stride_), bytes_per_member_};
    }

    /** Member @p member 's shareable physical handle — what a multicast object binds. */
    [[nodiscard]] CUmemGenericAllocationHandle member_handle(std::size_t member) const noexcept {
        return handles_[member];
    }

  private:

    /** Slice-creation properties: device-resident, pinned, RDMA-capable, and shareable — the
     *  last one is what multicast binding requires. */
    [[nodiscard]] static CUmemAllocationProp allocation_properties_(int device) noexcept {
        CUmemAllocationProp properties {};
        properties.type = CU_MEM_ALLOCATION_TYPE_PINNED;
        properties.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
        properties.location.id = device;
        properties.allocFlags.gpuDirectRDMACapable = 1;
        properties.requestedHandleTypes = CU_MEM_HANDLE_TYPE_FABRIC;
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
        // A non-zero handle implies a mapped slice — local ones created, remote ones adopted; either way
        // unmap then release. `make` / `adopt_peer` publish a handle only after the map succeeds.
        for (std::size_t member = 0; member != handles_.size(); ++member) {
            if (handles_[member] == 0) continue;
            cuMemUnmap(base_ + member * member_stride_, member_stride_);
            cuMemRelease(handles_[member]);
        }
        cuMemAddressFree(base_, member_stride_ * handles_.size());
        base_ = 0;
        devices_.reset();
        handles_.reset();
    }

    /** Start of the one contiguous reservation covering every member's slice. */
    CUdeviceptr base_ = 0;

    /** Usable bytes per slice, as requested at @ref make . */
    std::size_t bytes_per_member_ = 0;

    /** Granularity-rounded slice stride; also each slice's physical size. */
    std::size_t member_stride_ = 0;

    /** This process's local devices, slice order — the slices it created (not the whole domain). */
    array<i32_t> devices_ {};

    /** The physical handles indexed by @b domain member: local ones created, remote ones adopted;
     *  0 for a member not yet filled. */
    array<CUmemGenericAllocationHandle> handles_ {};
};

/** Which reduction tier a @ref multimem_domain_t runs: NVSwitch-fused @c multimem loads, or the
 *  serial peer-read fold — decided once at @c make by @c CU_DEVICE_ATTRIBUTE_MULTICAST_SUPPORTED ,
 *  identical results either way (both tiers are bit-deterministic). */
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

    /** Multicast base of the Float32 payload — `multimem` loads and stores fan out here. */
    f32_t *multimem_payload = nullptr;

    /** Multicast address of the arrival counter. */
    u32_t *multimem_flag = nullptr;

    /** Base of the contiguous symmetric range covering every member's replica. */
    std::byte *replicas = nullptr;

    /** This member's own round scalar (one word above its arrival flag) — read for the barrier
     *  target and self-advanced by the barrier kernels, never touched by any other member. */
    u32_t *round = nullptr;

    /** Byte distance between adjacent members' replicas. */
    std::size_t replica_stride = 0;

    /** Byte offset of the arrival counter inside every replica. */
    std::size_t flag_offset = 0;

    /** This member's index — whose replica is "own". */
    u32_t member_index = 0;

    /** The domain's member count — the barrier's arrival target and the peer fold's extent. */
    u32_t member_count = 0;

#if defined(__CUDACC__)

    /** Member @p member 's payload replica base. */
    [[nodiscard]] __device__ f32_t *peer_payload(u32_t member) const noexcept {
        return reinterpret_cast<f32_t *>(replicas + member * replica_stride);
    }

    /** Member @p member 's arrival counter. */
    [[nodiscard]] __device__ u32_t *peer_flag(u32_t member) const noexcept {
        return reinterpret_cast<u32_t *>(replicas + member * replica_stride + flag_offset);
    }

    /** This member's own arrival counter — the address the barrier spins on. */
    [[nodiscard]] __device__ u32_t const *own_flag() const noexcept { return peer_flag(member_index); }
#endif
};

#if defined(__CUDACC__)

/** One switch-reduced load: the sum of this address across every bound replica. */
[[nodiscard]] __device__ inline f32_t multimem_load_reduce_add_(f32_t const *multimem_address) noexcept {
    f32_t value;
    asm volatile("multimem.ld_reduce.relaxed.sys.global.add.f32 %0, [%1];"
                 : "=f"(value)
                 : "l"(multimem_address)
                 : "memory");
    return value;
}

/** One multicast release-add on the arrival counter of every replica. */
__device__ inline void multimem_red_add_release_(u32_t *multimem_address, u32_t value) noexcept {
    asm volatile("multimem.red.release.sys.global.add.u32 [%0], %1;" : : "l"(multimem_address), "r"(value) : "memory");
}

/** System-scope acquire load of this member's own arrival counter. */
[[nodiscard]] __device__ inline u32_t multimem_load_acquire_(u32_t const *own_address) noexcept {
    u32_t value;
    asm volatile("ld.acquire.sys.global.u32 %0, [%1];" : "=r"(value) : "l"(own_address) : "memory");
    return value;
}

#endif // defined(__CUDACC__)

/** A consumer's disjoint window of the domain payload: Float32 scalars
 *  @b [offset,offset+scalars) inside every replica — handed out by @ref multimem_domain_t::reserve ,
 *  so consumers' reduces never alias by construction. */
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
 *  The arrival flag is shared by every consumer, so all ranks must issue their @ref barrier calls in
 *  the @b same order across consumers within a step — the natural property of an SPMD step program.
 *  Each barrier is its own round (the kernel self-advances the member's round scalar), so consumers
 *  never pair or count rounds. Violating the order is a hang (arrivals are monotonic, a mis-ordered
 *  barrier waits forever), never a silent corruption.
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
     *  @brief Build the domain with @p payload_scalars Float32 per replica: the symmetric replica range,
     *      the multicast object, one multicast mapping, and the zeroed arrival counters. Runs once, at
     *      setup, outside any capture.
     *
     *  The replica range holds this process's created slices and every peer's adopted ones, exchanged
     *  over the mesh. The domain lead creates the multicast object, and every process imports it and
     *  binds its local slices into it.
     *
     *  The multimem_domain may span processes ( @c gpus_per_multimem_domain > the local device count on a
     *  `{4,8,16}` box or an NVL72), so the world @p topology — not the length of @p local_devices — sets
     *  the member count, this process's member offset, and the rendezvous the fabric handles ride. Each
     *  member's barrier round is a scalar inside its @b own replica, zeroed here beside the arrival flag
     *  and self-advanced by its barrier kernels — consumers never allocate or upload rounds themselves.
     *
     *  @param[in] topology The bootstrapped world (see @ref mesh_topology_t ).
     *  @param[in] local_devices The device ordinals this process drives — the slices it creates and binds.
     *  @param[in] payload_scalars Float32 payload scalars per replica, sized for every consumer's window.
     *  @param[in] local_streams One stream per local device; @c local_streams[0] carries the host staging.
     */
    [[nodiscard]] static expected<multimem_domain_t> make(mesh_topology_t const &topology,
                                                          std::span<i32_t const> local_devices,
                                                          std::size_t payload_scalars,
                                                          std::span<gpu_stream_t> local_streams) noexcept;

    /** The reduction tier this domain runs — see @ref multimem_tier_t . */
    [[nodiscard]] multimem_tier_t tier() const noexcept {
        return multimem_handle_ != 0 ? multimem_tier_t::switch_reduced_k : multimem_tier_t::peer_read_k;
    }

    /** Domain member count. */
    [[nodiscard]] std::size_t member_count() const noexcept { return replicas_.member_count(); }

    /** Float32 payload scalars per replica. */
    [[nodiscard]] std::size_t payload_scalars() const noexcept { return payload_scalars_; }

    /** Member @p member 's own payload replica — where its contribution is written. */
    [[nodiscard]] f32r1_span_t payload(std::size_t member) noexcept {
        return {reinterpret_cast<f32_t *>(replicas_.member_slice(member).data()), payload_scalars_};
    }

    /** Reserve @p scalars of the payload for one consumer — bump-reserve at construction time,
     *  loud failure past capacity; windows are disjoint by construction. Mirrors the arena
     *  discipline: no per-window free, the domain outlives its consumers. */
    [[nodiscard]] expected<multimem_window_t> reserve(std::size_t scalars) noexcept {
        if (reserved_scalars_ + scalars > payload_scalars_)
            return expected<multimem_window_t>::fail(
                status_t::bad_alloc_k, "window reservation of {} scalars exceeds the domain's {} ({} reserved)",
                scalars, payload_scalars_, reserved_scalars_);
        multimem_window_t const window {reserved_scalars_, scalars};
        reserved_scalars_ += scalars;
        return expected<multimem_window_t> {window};
    }

    /** Member @p member_index 's kernel-facing addresses. */
    [[nodiscard]] multimem_member_t member(std::size_t member_index) noexcept {
        std::size_t const flag_offset = flag_offset_();
        multimem_member_t addresses;
        if (multimem_pointer_ != 0) {
            addresses.multimem_payload = reinterpret_cast<f32_t *>(multimem_pointer_);
            addresses.multimem_flag = reinterpret_cast<u32_t *>(multimem_pointer_ + flag_offset);
        }
        addresses.replicas = replicas_.base();
        addresses.round = reinterpret_cast<u32_t *>(replicas_.member_slice(member_index).data() + flag_offset +
                                                    sizeof(u32_t));
        addresses.replica_stride = replicas_.member_stride();
        addresses.flag_offset = flag_offset;
        addresses.member_index = static_cast<u32_t>(member_index);
        addresses.member_count = static_cast<u32_t>(member_count());
        return addresses;
    }

    /** Enqueue the domain barrier for @p member_index — one arrival, one acquire-spin against the
     *  member's own round scalar, which the kernel self-advances after the spin. On the peer tier the
     *  arrival walks the members through the symmetric range instead. */
    [[nodiscard]] expected_t barrier(std::size_t member_index, gpu_stream_t &stream) noexcept;

    /** Enqueue member @p member_index 's reduce-scatter shard: @p destination receives the
     *  domain-wide sum of @c [shard_begin,shard_begin+destination.extent(0)) . Call after the
     *  barrier proved every member's payload complete. */
    [[nodiscard]] expected_t reduce_scatter(std::size_t member_index, f32r1_span_t destination, std::size_t shard_begin,
                                            gpu_stream_t &stream) noexcept;

  private:

    /** Byte offset of the arrival counter inside every replica — pure layout arithmetic over
     *  @ref payload_scalars_ , never stored; each member's round scalar sits one word above its flag. */
    [[nodiscard]] std::size_t flag_offset_() const noexcept {
        return round_up_to_multiple(payload_scalars_ * sizeof(f32_t), cache_line_bytes_k);
    }

    void swap_(multimem_domain_t &other) noexcept {
        std::swap(replicas_, other.replicas_);
        std::swap(multimem_handle_, other.multimem_handle_);
        std::swap(multimem_pointer_, other.multimem_pointer_);
        std::swap(payload_scalars_, other.payload_scalars_);
        std::swap(reserved_scalars_, other.reserved_scalars_);
        std::swap(member_offset_, other.member_offset_);
    }

    /** Reverse the multicast machinery; the symmetric arena member tears itself down after. This
     *  process unbinds only the slices it bound — its @b local devices. */
    void destroy_() noexcept {
        std::size_t const bound_bytes = replicas_.member_stride();
        if (multimem_pointer_ != 0) {
            cuMemUnmap(multimem_pointer_, bound_bytes);
            cuMemAddressFree(multimem_pointer_, bound_bytes);
            multimem_pointer_ = 0;
        }
        if (multimem_handle_ != 0) {
            for (std::size_t local = 0; local != replicas_.local_member_count(); ++local) {
                CUdevice device_handle = 0;
                if (cuDeviceGet(&device_handle, replicas_.local_device(local)) == CUDA_SUCCESS)
                    cuMulticastUnbind(multimem_handle_, device_handle, 0, bound_bytes);
            }
            cuMemRelease(multimem_handle_);
            multimem_handle_ = 0;
        }
    }

    /** The one contiguous replica range — every member's slice, peer-addressable by arithmetic. */
    gpu_symmetric_arena_t replicas_ {};

    /** The multicast object every slice is bound into; 0 on the peer-read tier. */
    CUmemGenericAllocationHandle multimem_handle_ = 0;

    /** The one multicast VA shared by all members; 0 on the peer-read tier. */
    CUdeviceptr multimem_pointer_ = 0;

    /** Float32 payload scalars per replica. */
    std::size_t payload_scalars_ = 0;

    /** The window-reservation cursor; see @ref reserve . */
    std::size_t reserved_scalars_ = 0;

    /** This process's first domain member — 0 iff it is the domain lead. */
    u32_t member_offset_ = 0;
};

} // namespace unum::uform::core
