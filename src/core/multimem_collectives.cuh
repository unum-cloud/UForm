/**
 *  @file src/core/multimem_collectives.cuh
 *  @author Ash Vardanian
 *  @date July 3, 2026
 *  @brief Collectives @b inside one multimem_domain: the domain's members reduce and broadcast
 *      through @c multimem PTX over one shared multicast object — the level-1 transport of the
 *      two-level contrastive step and the distributed optimizer.
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
 *  The @c multimem PTX lives in its own unit, @c multimem_switch.cu , compiled only for the
 *  multicast-capable codes of the build and embedded as a bundle the domain loads by name when it
 *  picks the switch tier. When the platform lacks multicast (
 *  @c CU_DEVICE_ATTRIBUTE_MULTICAST_SUPPORTED off — pre-Hopper or fabric-disabled boxes), or the
 *  bundle carries no code the device runs, @ref multimem_domain_t::make still succeeds in the
 *  @b peer-read tier: the symmetric range is device-visible everywhere, and the reduce walks the
 *  members serially — the same fixed fold order on every member, bit-identical across ranks, just
 *  without the switch reduction.
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
 *  A multimem_domain may span @b more @b than @b one @b process — on an NVL72 the 72-GPU island is
 *  18 tray processes, and even a `{4,8,16}` HGX box runs two 4-GPU processes over one 8-GPU island.
 *  So the arena is @b per @b process: it reserves the whole @c member_count·stride VA, but each
 *  process @ref make -s only its @b local slices (physically resident on the GPUs it drives) and
 *  @ref adopt_peer -s every remote member's slice from the fabric handle its owner published. The
 *  slices' physical handles are created shareable only when the domain spans processes:
 *  @c CU_MEM_HANDLE_TYPE_FABRIC where every local device exports it, exported through
 *  @ref export_fabric_handle , and POSIX file descriptors on a fabric-less box, whose handoff to
 *  another process needs a Unix-socket exchange not implemented yet, so their export refuses. A
 *  one-process domain shares no handle at all. Slices are RDMA-capable where the device registers
 *  VMM memory with an adapter, and @ref member_handle exposes them so a multicast object can bind
 *  them. Every local device is granted access to the whole range, so the peer-read tier reads every
 *  replica straight through it.
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

    /** The shareable handle a domain of @p member_count members over @p local_devices creates its
     *  slices with: none within one process, else fabric handles where every local device exports
     *  them, else POSIX file descriptors. */
    [[nodiscard]] static expected<CUmemAllocationHandleType> shareable_handle_type(std::span<i32_t const> local_devices,
                                                                                   u32_t member_count) noexcept;

    /** Export domain member @p member 's slice as a fabric handle — what its owner publishes so the
     *  domain's other processes can @ref adopt_peer it. @p member must be one this process created,
     *  and slices shared as POSIX file descriptors refuse, as their cross-process handoff is not
     *  implemented. */
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
    [[nodiscard]] CUdeviceptr base() const noexcept { return base_; }

    /** Member @p member 's usable slice. */
    [[nodiscard]] std::span<std::byte> member_slice(std::size_t member) noexcept {
        return {device_pointer<std::byte>(base_ + member * member_stride_), bytes_per_member_};
    }

    /** Member @p member 's shareable physical handle — what a multicast object binds. */
    [[nodiscard]] CUmemGenericAllocationHandle member_handle(std::size_t member) const noexcept {
        return handles_[member];
    }

    /** The shareable handle the slices were created with, see @ref shareable_handle_type . */
    [[nodiscard]] CUmemAllocationHandleType handle_type() const noexcept { return handle_type_; }

  private:

    /** Slice-creation properties: device-resident, pinned, RDMA-capable wherever the device
     *  registers VMM memory with an adapter, and shareable as @ref handle_type_ . */
    [[nodiscard]] CUmemAllocationProp allocation_properties_(int device) const noexcept {
        int rdma_supported = 0;
        cuDeviceGetAttribute(&rdma_supported, CU_DEVICE_ATTRIBUTE_GPU_DIRECT_RDMA_WITH_CUDA_VMM_SUPPORTED, device);
        CUmemAllocationProp properties {};
        properties.type = CU_MEM_ALLOCATION_TYPE_PINNED;
        properties.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
        properties.location.id = device;
        properties.allocFlags.gpuDirectRDMACapable = rdma_supported != 0;
        properties.requestedHandleTypes = handle_type_;
        return properties;
    }

    void swap_(gpu_symmetric_arena_t &other) noexcept {
        std::swap(base_, other.base_);
        std::swap(bytes_per_member_, other.bytes_per_member_);
        std::swap(member_stride_, other.member_stride_);
        std::swap(handle_type_, other.handle_type_);
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

    /** The shareable handle every local slice is created with. */
    CUmemAllocationHandleType handle_type_ = CU_MEM_HANDLE_TYPE_NONE;

    /** This process's local devices, slice order — the slices it created (not the whole domain). */
    array<i32_t> devices_ {};

    /** The physical handles indexed by @b domain member: local ones created, remote ones adopted;
     *  0 for a member not yet filled. */
    array<CUmemGenericAllocationHandle> handles_ {};
};

/** Which reduction tier a @ref multimem_domain_t runs: NVSwitch-fused @c multimem loads, or the
 *  serial peer-read fold — decided once at @c make by @ref multimem_tier_choice_t , identical
 *  results either way (both tiers are bit-deterministic). */
enum class multimem_tier_t : u32_t {
    switch_reduced_k = 0,
    peer_read_k = 1,
};

/** One device's switch-tier kernels, resolved by name from the switch bundle loaded into its
 *  context. */
struct multimem_switch_kernels_t {
    CUfunction barrier = nullptr;
    CUfunction reduce_scatter = nullptr;
    CUfunction reduce_gradient = nullptr;
};

/**
 *  @brief The switch bundle loaded into one device's primary context: the module, the context it
 *      retains to unload it, and the kernels resolved from it.
 *
 *  The bundle is a fatbin of @c multimem_switch.cu for the multicast-capable codes this build
 *  lists, empty when it lists none, so a device it carries no code for refuses the load and its
 *  domain reads peers.
 */
struct multimem_switch_module_t {

    multimem_switch_module_t() noexcept = default;
    multimem_switch_module_t(multimem_switch_module_t const &) = delete;
    multimem_switch_module_t &operator=(multimem_switch_module_t const &) = delete;
    multimem_switch_module_t(multimem_switch_module_t &&other) noexcept { swap_(other); }
    multimem_switch_module_t &operator=(multimem_switch_module_t &&other) noexcept {
        if (this != &other) {
            destroy_();
            swap_(other);
        }
        return *this;
    }
    ~multimem_switch_module_t() noexcept { destroy_(); }

    /** Load the switch bundle onto @p device and resolve its kernels; fails when the bundle is
     *  empty or carries no code @p device runs. */
    [[nodiscard]] static expected<multimem_switch_module_t> load(i32_t device) noexcept;

    /** The kernels, valid in @p device 's primary context for the module's lifetime. */
    [[nodiscard]] multimem_switch_kernels_t const &kernels() const noexcept { return kernels_; }

  private:
    void swap_(multimem_switch_module_t &other) noexcept {
        std::swap(context_, other.context_);
        std::swap(module_, other.module_);
        std::swap(device_, other.device_);
        std::swap(kernels_, other.kernels_);
    }

    void destroy_() noexcept;

    /** The device's retained primary context, which the module lives in. */
    CUcontext context_ = nullptr;
    CUmodule module_ = nullptr;
    i32_t device_ = -1;
    multimem_switch_kernels_t kernels_ {};
};

/**
 *  @brief The tier a domain runs, why, and the switch modules that tier needs.
 *
 *  The switch tier needs more than one member, multicast on every local device, and the switch
 *  bundle loading on each of them; anything less falls back to the peer-read tier, and @ref reason
 *  says which check failed.
 */
struct multimem_tier_choice_t {

    multimem_tier_t tier = multimem_tier_t::peer_read_k;

    /** Why the domain runs @ref tier , naming the device and the failure when it fell back. */
    sz::string_t reason {};

    /** One loaded bundle per local device, in local order, on the switch tier; empty otherwise. */
    array<multimem_switch_module_t> modules {};

    /** Probe @p local_devices for a domain of @p member_count members; fails only when a query
     *  fails. */
    [[nodiscard]] static expected<multimem_tier_choice_t> make(u32_t member_count,
                                                               std::span<i32_t const> local_devices) noexcept;
};

/**
 *  @brief One member's addresses inside the domain — the scalar-only POD kernels take by value.
 *
 *  @c multimem_* pointers fan out to every replica (null on the peer-read tier); everything else is
 *  arithmetic over the one contiguous replica range: member @c m 's payload starts at
 *  @c replica_payloads+m·replica_stride_words , its arrival flag at @c replica_flags+m·replica_stride_words .
 *  All addresses are capture-stable for the domain's lifetime.
 */
struct multimem_member_t {

    /** Multicast base of the Float32 payload — `multimem` loads and stores fan out here. */
    f32_t *multimem_payload = nullptr;

    /** Multicast address of the arrival counter. */
    u32_t *multimem_flag = nullptr;

    /** Member 0's payload replica, the start of the contiguous symmetric range. */
    f32_t *replica_payloads = nullptr;

    /** Member 0's arrival counter. */
    u32_t *replica_flags = nullptr;

    /** This member's own round scalar (one word above its arrival flag) — read for the barrier
     *  target and self-advanced by the barrier kernels, never touched by any other member. */
    u32_t *round = nullptr;

    /** Distance between adjacent members' replicas, in 4-byte payload scalars or flag words alike. */
    std::size_t replica_stride_words = 0;

    /** This member's index — whose replica is "own". */
    u32_t member_index = 0;

    /** The domain's member count — the barrier's arrival target and the peer fold's extent. */
    u32_t member_count = 0;

    /** How long the barrier spins for the other members before it traps. */
    u64_t wait_nanoseconds = 0;

#if defined(__CUDACC__)

    /** Member @p member 's payload replica base. */
    [[nodiscard]] __device__ f32_t *peer_payload(u32_t member) const noexcept {
        return replica_payloads + member * replica_stride_words;
    }

    /** Member @p member 's arrival counter. */
    [[nodiscard]] __device__ u32_t *peer_flag(u32_t member) const noexcept {
        return replica_flags + member * replica_stride_words;
    }

    /** This member's own arrival counter — the address the barrier spins on. */
    [[nodiscard]] __device__ u32_t const *own_flag() const noexcept { return peer_flag(member_index); }
#endif
};

#if defined(__CUDACC__)

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
 *  never pair or count rounds. Violating the order traps once a mis-ordered barrier outwaits the
 *  domain's timeout, never a silent corruption.
 *
 *  Falls back to the peer-read tier when multicast is unsupported or the switch kernels do not load
 *  — same API, same determinism, the reduce just walks members serially. @ref tier() names which
 *  one is live and @ref description says why. Every local pair of members must reach each other
 *  over P2P, or @ref make refuses.
 */
struct multimem_domain_t {

    /** The empty domain a composition root places before @ref make fills it. */
    multimem_domain_t() noexcept = default;
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
     *  @param[inout] topology The bootstrapped world (see @ref mesh_topology_t ); one wiring
     *      generation is claimed from it, so a domain made twice in one job meets under fresh keys.
     *  @param[in] local_devices The device ordinals this process drives — the slices it creates and binds.
     *  @param[in] payload_scalars Float32 payload scalars per replica, sized for every consumer's window.
     *  @param[in] local_streams One stream per local device; @c local_streams[0] carries the host staging.
     *  @param[in] timeout_seconds Deadline on the rendezvous and on every later barrier's spin.
     */
    [[nodiscard]] static expected<multimem_domain_t> make(mesh_topology_t &topology,
                                                          std::span<i32_t const> local_devices,
                                                          std::size_t payload_scalars,
                                                          std::span<gpu_stream_t> local_streams,
                                                          f64_t timeout_seconds) noexcept;

    /** The reduction tier this domain runs — see @ref multimem_tier_t . */
    [[nodiscard]] multimem_tier_t tier() const noexcept { return tier_.tier; }

    /** Why the domain runs its @ref tier , naming the failed check when it fell back to reading
     *  peers. */
    [[nodiscard]] sz::string_view_t description() const noexcept { return tier_.reason.view(); }

    /** Member @p member_index 's switch kernels, in its device's primary context; switch tier
     *  only. */
    [[nodiscard]] multimem_switch_kernels_t const &switch_kernels(std::size_t member_index) const noexcept {
        return tier_.modules[member_index - member_offset_].kernels();
    }

    /** Domain member count. */
    [[nodiscard]] std::size_t member_count() const noexcept { return replicas_.member_count(); }

    /** This process's first member: its local device @c d is domain member @c member_offset()+d . */
    [[nodiscard]] std::size_t member_offset() const noexcept { return member_offset_; }

    /** How many of the members this process drives. */
    [[nodiscard]] std::size_t local_member_count() const noexcept { return replicas_.local_member_count(); }

    /** Float32 payload scalars per replica. */
    [[nodiscard]] std::size_t payload_scalars() const noexcept { return payload_scalars_; }

    /** Member @p member 's own payload replica — where its contribution is written. */
    [[nodiscard]] f32r1_span_t payload(std::size_t member) noexcept {
        return {device_pointer<f32_t>(replicas_.base() + member * replicas_.member_stride()), payload_scalars_};
    }

    /** Member @p member 's @p window as bytes, for consumers staging non-Float32 data where every process's
     *  members can read it. */
    [[nodiscard]] b8r1_span_t window_bytes(std::size_t member, multimem_window_t window) noexcept {
        return {
            device_pointer<b8_t>(replicas_.base() + member * replicas_.member_stride() + window.offset * sizeof(f32_t)),
            window.scalars * sizeof(f32_t)};
    }

    /** Payload scalars one window of @p scalars takes before the next, rounded to whole cache lines
     *  so every window starts on one; a domain's sizing counts every window but the last through
     *  it. */
    [[nodiscard]] static constexpr std::size_t window_scalars(std::size_t scalars) noexcept {
        return round_up_to_multiple(scalars, cache_line_bytes_k / sizeof(f32_t));
    }

    /** Reserve @p scalars of the payload for one consumer — bump-reserve at construction time, loud
     *  failure past capacity; windows are disjoint by construction and start on a cache line, so
     *  vectorized, 8-byte and TMA access holds in any window. Mirrors the arena discipline: no
     *  per-window free, the domain outlives its consumers. */
    [[nodiscard]] expected<multimem_window_t> reserve(std::size_t scalars) noexcept {
        if (reserved_scalars_ + scalars > payload_scalars_)
            return expected<multimem_window_t>::fail(
                status_t::bad_alloc_k, "window reservation of {} scalars exceeds the domain's {} ({} reserved)",
                scalars, payload_scalars_, reserved_scalars_);
        multimem_window_t const window {reserved_scalars_, scalars};
        reserved_scalars_ += window_scalars(scalars);
        static_assert(cache_line_bytes_k % sizeof(f32_t) == 0, "windows start on whole cache lines");
        return expected<multimem_window_t> {window};
    }

    /** Member @p member_index 's kernel-facing addresses. */
    [[nodiscard]] multimem_member_t member(std::size_t member_index) noexcept {
        std::size_t const flag_offset = flag_offset_();
        multimem_member_t addresses;
        if (multimem_pointer_ != 0) {
            addresses.multimem_payload = device_pointer<f32_t>(multimem_pointer_);
            addresses.multimem_flag = device_pointer<u32_t>(multimem_pointer_ + flag_offset);
        }
        CUdeviceptr const replicas = replicas_.base();
        addresses.replica_payloads = device_pointer<f32_t>(replicas);
        addresses.replica_flags = device_pointer<u32_t>(replicas + flag_offset);
        addresses.replica_stride_words = replicas_.member_stride() / sizeof(u32_t);
        addresses.round = addresses.replica_flags + member_index * addresses.replica_stride_words + 1u;
        addresses.member_index = static_cast<u32_t>(member_index);
        addresses.member_count = static_cast<u32_t>(member_count());
        addresses.wait_nanoseconds = wait_nanoseconds_;
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
        std::swap(tier_, other.tier_);
        std::swap(multimem_handle_, other.multimem_handle_);
        std::swap(multimem_pointer_, other.multimem_pointer_);
        std::swap(payload_scalars_, other.payload_scalars_);
        std::swap(reserved_scalars_, other.reserved_scalars_);
        std::swap(member_offset_, other.member_offset_);
        std::swap(wait_nanoseconds_, other.wait_nanoseconds_);
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

    /** The tier chosen at @ref make , with the switch modules it loaded. */
    multimem_tier_choice_t tier_ {};

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

    /** How long a barrier spins for the other members before it traps, from the timeout. */
    u64_t wait_nanoseconds_ = 0;
};

} // namespace unum::uform::core
