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
 *  Synchronization is the same monotonic-epoch discipline the InfiniBand collectives use: an arrival
 *  flag lives inside every replica, a member @b adds 1 through the multicast address (so every replica's
 *  counter grows), and waiters acquire-spin on their @b own replica until the epoch target — no resets,
 *  so captured steps replay without host writes. The epoch scalar is the domain's own, inside the lead
 *  replica, advanced once per step through @ref multimem_domain_t::advance .
 *
 *  When the platform lacks multicast ( @c CU_DEVICE_ATTRIBUTE_MULTICAST_SUPPORTED off — pre-Hopper or
 *  fabric-disabled boxes), @ref multimem_domain_t::make still succeeds in the @b peer-read tier: the
 *  symmetric range is device-visible everywhere, and the reduce walks the members serially — the same
 *  fixed fold order on every member, bit-identical across ranks, just without the switch reduction.
 */
#pragma once

#include <cstddef> // `std::size_t`
#include <span>    // `std::span` — device-ID and stream lists

#include <cuda.h> // `cuMulticastCreate`, `cuMulticastBindMem`, `cuMemMap`, `cuMemSetAccess`

#include "types.cuh"          // `expected`, `expected_t`, `status_t`, the scalar aliases
#include "array.cuh"          // `array` — the runtime-sized per-member tables
#include "schedule.cuh"       // `gpu_stream_t`, `launch_kernel`, `current_context_t`, the rank-1 aliases
#include "gpu_allocators.cuh" // `gpu_arena_t` conventions, `round_up_to_multiple`

namespace unum::uform::primitives {

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
                                                              std::size_t alignment_granularity = 1) noexcept {
        using result_t = expected<gpu_symmetric_arena_t>;
        if (devices.empty() || bytes_per_member == 0)
            return result_t::fail(status_t::invalid_input_k, "a symmetric arena needs members and bytes ({} / {})",
                                  devices.size(), bytes_per_member);
        gpu_symmetric_arena_t arena;
        arena.bytes_per_member_ = bytes_per_member;
        if (arena.devices_.try_assign({devices.data(), devices.size()}) != status_t::success_k ||
            arena.handles_.try_resize(devices.size()) != status_t::success_k)
            return result_t::fail(status_t::bad_alloc_k, "member table allocation failed ({})", devices.size());

        // The stride honors the coarsest requirement: every member's physical granularity plus the
        // caller's alignment (a multicast binding granularity, typically).
        std::size_t stride = alignment_granularity != 0 ? alignment_granularity : 1;
        for (i32_t const device : devices) {
            CUmemAllocationProp properties = allocation_properties_(device);
            std::size_t granularity = 0;
            if (CUresult result = cuMemGetAllocationGranularity(&granularity, &properties,
                                                                CU_MEM_ALLOC_GRANULARITY_RECOMMENDED);
                result != CUDA_SUCCESS)
                return result_t::fail(result, "granularity query failed (device={})", device);
            if (granularity > stride) stride = granularity;
        }
        arena.member_stride_ = round_up_to_multiple(bytes_per_member, stride);

        std::size_t const total_bytes = arena.member_stride_ * devices.size();
        if (CUresult result = cuMemAddressReserve(&arena.base_, total_bytes, 0, 0, 0); result != CUDA_SUCCESS)
            return result_t::fail(result, "symmetric VA reservation failed ({} bytes)", total_bytes);

        for (std::size_t member = 0; member != devices.size(); ++member) {
            CUmemAllocationProp properties = allocation_properties_(devices[member]);
            CUdeviceptr const slice = arena.base_ + member * arena.member_stride_;
            if (CUresult result = cuMemCreate(&arena.handles_[member], arena.member_stride_, &properties, 0);
                result != CUDA_SUCCESS)
                return result_t::fail(result, "slice allocation failed (device={} bytes={})", devices[member],
                                      arena.member_stride_);
            if (CUresult result = cuMemMap(slice, arena.member_stride_, 0, arena.handles_[member], 0);
                result != CUDA_SUCCESS)
                return result_t::fail(result, "slice mapping failed (member={})", member);
            arena.mapped_members_ = member + 1;
            for (i32_t const reader : devices) {
                CUmemAccessDesc access {};
                access.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
                access.location.id = reader;
                access.flags = CU_MEM_ACCESS_FLAGS_PROT_READWRITE;
                if (CUresult result = cuMemSetAccess(slice, arena.member_stride_, &access, 1); result != CUDA_SUCCESS)
                    return result_t::fail(result, "slice access grant failed (member={} reader={})", member, reader);
            }
        }
        return result_t {std::move(arena)};
    }

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
        std::swap(mapped_members_, other.mapped_members_);
        std::swap(devices_, other.devices_);
        std::swap(handles_, other.handles_);
    }

    void destroy_() noexcept {
        if (base_ == 0) return;
        for (std::size_t member = 0; member != mapped_members_; ++member)
            cuMemUnmap(base_ + member * member_stride_, member_stride_);
        for (std::size_t member = 0; member != handles_.size(); ++member)
            if (handles_[member] != 0) cuMemRelease(handles_[member]);
        cuMemAddressFree(base_, member_stride_ * devices_.size());
        base_ = 0;
        mapped_members_ = 0;
        devices_.reset();
        handles_.reset();
    }

    /** @brief Start of the one contiguous reservation covering every member's slice. */
    CUdeviceptr base_ = 0;
    /** @brief Usable bytes per slice, as requested at @ref make . */
    std::size_t bytes_per_member_ = 0;
    /** @brief Granularity-rounded slice stride; also each slice's physical size. */
    std::size_t member_stride_ = 0;
    /** @brief How many slices are mapped — the teardown high-water mark on failed construction. */
    std::size_t mapped_members_ = 0;
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
    /** @brief The domain's epoch scalar (inside the lead replica) every barrier keys on. */
    u32_t const *epoch = nullptr;
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

/**
 *  @brief Domain barrier: lane 0 release-adds 1 into every replica's counter, then acquire-spins on its
 *      @b own counter until all @c member_count arrivals of this epoch landed. Monotonic — the counter
 *      never resets, so a captured step replays with the host advancing only the epoch scalar.
 *
 *  __grid_size__ @c [1,1,1] , one warp; no dynamic shared memory.
 */
static __global__ void multimem_barrier_kernel_(multimem_member_t member) {
    if (threadIdx.x != 0u) return;
    u32_t const target = *member.epoch + member.member_count;
    multimem_red_add_release_(member.multimem_flag, 1u);
    while (multimem_load_acquire_(member.own_flag()) < target) {
        // Spin until every member's arrival of this epoch reached our replica.
    }
    __threadfence_system();
}

/**
 *  @brief Peer-read sibling of @ref multimem_barrier_kernel_ for the no-multicast tier: lane 0 adds into
 *      every member's counter through the symmetric range, then spins on its own.
 *
 *  __grid_size__ @c [1,1,1] , one warp; no dynamic shared memory.
 */
static __global__ void multimem_barrier_peer_kernel_(multimem_member_t member) {
    if (threadIdx.x != 0u) return;
    u32_t const target = *member.epoch + member.member_count;
    for (u32_t peer = 0; peer != member.member_count; ++peer) {
        u32_t *flag = member.peer_flag(peer);
        asm volatile("red.release.sys.global.add.u32 [%0], %1;" : : "l"(flag), "r"(1u) : "memory");
    }
    while (multimem_load_acquire_(member.own_flag()) < target) {
        // Spin until every member's arrival of this epoch reached our replica.
    }
    __threadfence_system();
}

/**
 *  @brief Reduce-scatter shard: @c destination[i] = Σ_members payload[shard_begin+i] , one switch-reduced
 *      load per element — deterministic with no fold order at all.
 *
 *  __grid_size__ grid-stride over @c shard_scalars ; no dynamic shared memory.
 */
static __global__ void multimem_reduce_scatter_kernel_(multimem_member_t member, f32r1_span_t destination,
                                                       std::size_t shard_begin) {
    std::size_t const shard_scalars = destination.extent(0);
    std::size_t const stride = static_cast<std::size_t>(gridDim.x) * blockDim.x;
    for (std::size_t index = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x; index < shard_scalars;
         index += stride)
        destination[index] = multimem_load_reduce_add_(member.multimem_payload + shard_begin + index);
}

/**
 *  @brief Peer-read sibling of @ref multimem_reduce_scatter_kernel_ : the same shard, folded serially
 *      over the members through the symmetric range — a fixed order, bit-identical on every member.
 *
 *  __grid_size__ grid-stride over @c shard_scalars ; no dynamic shared memory.
 */
static __global__ void multimem_reduce_scatter_peer_kernel_(multimem_member_t member, f32r1_span_t destination,
                                                            std::size_t shard_begin) {
    std::size_t const shard_scalars = destination.extent(0);
    std::size_t const stride = static_cast<std::size_t>(gridDim.x) * blockDim.x;
    for (std::size_t index = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x; index < shard_scalars;
         index += stride) {
        f32_t total = 0.0f;
        for (u32_t peer = 0; peer != member.member_count; ++peer)
            total = __fadd_rn(total, member.peer_payload(peer)[shard_begin + index]);
        destination[index] = total;
    }
}

#endif // defined(__CUDACC__)

/**
 *  @brief One multimem_domain's collective surface: every member's same-sized Float32 replica inside one
 *      @ref gpu_symmetric_arena_t , bound into one multicast object — the replicas @b are whatever force
 *      or gradient surface the consumer reduces over.
 *
 *  Falls back to the peer-read tier when multicast is unsupported — same API, same determinism, the
 *  reduce just walks members serially. @ref tier() names which one is live.
 */
struct multimem_domain_t {

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
     *  @brief Build the domain over @p devices with @p payload_scalars Float32 per replica: the
     *      symmetric replica range, the multicast object binding every slice, one multicast mapping,
     *      and the zeroed arrival counters. Runs once, at setup, outside any capture. The barrier epoch
     *      is the domain's own: a scalar inside the lead replica, advanced once per step through
     *      @ref advance — consumers never carve or upload epochs themselves.
     */
    [[nodiscard]] static expected<multimem_domain_t> make(std::span<i32_t const> devices, std::size_t payload_scalars,
                                                          std::span<gpu_stream_t const> streams) noexcept {
        using result_t = expected<multimem_domain_t>;
        std::size_t const member_count = devices.size();
        if (member_count == 0) return result_t::fail(status_t::invalid_input_k, "a domain needs members");
        if (streams.size() != member_count)
            return result_t::fail(status_t::invalid_input_k, "every member needs its stream ({} vs {})", member_count,
                                  streams.size());
        // Host-side staging (memsets, the epoch reset) needs a current context; the lead stream owns one.
        switch_to_gpu_t gpu_scope {streams[0]};
        if (!gpu_scope.ok())
            return result_t::fail(status_t::cuda_error_k, "context entry failed (device={})", devices[0]);

        multimem_domain_t domain;
        domain.payload_scalars_ = payload_scalars;
        // Replica layout: [payload f32×N | flag u32 | epoch u32], regions cache-line aligned. The epoch
        // scalar is live only in the lead replica, peer-read by every barrier.
        domain.flag_offset_ = (payload_scalars * sizeof(f32_t) + 127ull) & ~127ull;
        std::size_t const replica_bytes = domain.flag_offset_ + 2 * sizeof(u32_t);

        bool multimem_everywhere = true;
        for (i32_t const device : devices) {
            int supported = 0;
            if (CUresult result = cuDeviceGetAttribute(&supported, CU_DEVICE_ATTRIBUTE_MULTICAST_SUPPORTED, device);
                result != CUDA_SUCCESS)
                return result_t::fail(result, "multicast attribute query failed (device={})", device);
            multimem_everywhere = multimem_everywhere && supported != 0;
        }

        // The multicast binding granularity dictates the replica stride; the symmetric arena also
        // honors every member's physical granularity.
        std::size_t multimem_granularity = 1;
        CUmulticastObjectProp multimem_properties {};
        if (multimem_everywhere) {
            multimem_properties.numDevices = static_cast<unsigned>(member_count);
            multimem_properties.handleTypes = CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR;
            multimem_properties.size = replica_bytes;
            if (CUresult result = cuMulticastGetGranularity(&multimem_granularity, &multimem_properties,
                                                            CU_MULTICAST_GRANULARITY_RECOMMENDED);
                result != CUDA_SUCCESS)
                return result_t::fail(result, "multicast granularity query failed");
        }
        expected<gpu_symmetric_arena_t> replicas = gpu_symmetric_arena_t::make(devices, replica_bytes,
                                                                               multimem_granularity);
        if (!replicas) return result_t::fail(std::move(replicas));
        domain.replicas_ = std::move(replicas.value_);

        // The multicast object over every member's slice, and one multicast VA shared by all members.
        if (multimem_everywhere) {
            std::size_t const bound_bytes = domain.replicas_.member_stride();
            multimem_properties.size = bound_bytes;
            if (CUresult result = cuMulticastCreate(&domain.multimem_handle_, &multimem_properties);
                result != CUDA_SUCCESS)
                return result_t::fail(result, "multicast object creation failed (members={} bytes={})", member_count,
                                      bound_bytes);
            for (i32_t const device : devices) {
                CUdevice device_handle = 0;
                if (CUresult result = cuDeviceGet(&device_handle, device); result != CUDA_SUCCESS)
                    return result_t::fail(result, "device handle failed ({})", device);
                if (CUresult result = cuMulticastAddDevice(domain.multimem_handle_, device_handle);
                    result != CUDA_SUCCESS)
                    return result_t::fail(result, "multicast add-device failed (device={})", device);
            }
            for (std::size_t member = 0; member != member_count; ++member)
                if (CUresult result = cuMulticastBindMem(domain.multimem_handle_, 0,
                                                         domain.replicas_.member_handle(member), 0, bound_bytes, 0);
                    result != CUDA_SUCCESS)
                    return result_t::fail(result, "multicast bind failed (member={})", member);
            if (CUresult result = cuMemAddressReserve(&domain.multimem_pointer_, bound_bytes, 0, 0, 0);
                result != CUDA_SUCCESS)
                return result_t::fail(result, "multicast reservation failed");
            if (CUresult result = cuMemMap(domain.multimem_pointer_, bound_bytes, 0, domain.multimem_handle_, 0);
                result != CUDA_SUCCESS)
                return result_t::fail(result, "multicast mapping failed");
            for (i32_t const device : devices) {
                CUmemAccessDesc access {};
                access.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
                access.location.id = device;
                access.flags = CU_MEM_ACCESS_FLAGS_PROT_READWRITE;
                if (CUresult result = cuMemSetAccess(domain.multimem_pointer_, bound_bytes, &access, 1);
                    result != CUDA_SUCCESS)
                    return result_t::fail(result, "multicast access grant failed (device={})", device);
            }
        }

        // Zero every member's flag and the lead replica's epoch.
        for (std::size_t member = 0; member != member_count; ++member) {
            CUdeviceptr const counters = reinterpret_cast<CUdeviceptr>(domain.replicas_.member_slice(member).data()) +
                                         domain.flag_offset_;
            if (CUresult result = cuMemsetD8(counters, 0, 2 * sizeof(u32_t)); result != CUDA_SUCCESS)
                return result_t::fail(result, "arrival counter reset failed (member={})", member);
        }
        return result_t {std::move(domain)};
    }

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

    /** @brief The one host write of a step: upload the epoch the coming barriers key on and advance the
     *      host counter by the member count. Enqueue on the lead stream before any member's barrier. */
    [[nodiscard]] expected_t advance(gpu_stream_t const &stream) noexcept {
        if (CUresult result = cuMemcpyHtoDAsync(reinterpret_cast<CUdeviceptr>(epoch_pointer_()), &host_epoch_,
                                                sizeof(host_epoch_), stream.get());
            result != CUDA_SUCCESS)
            return expected_t::fail(result, "domain epoch upload failed");
        host_epoch_ += static_cast<u32_t>(member_count());
        return {};
    }

    /** @brief Member @p member_index 's kernel-facing addresses. */
    [[nodiscard]] multimem_member_t member(std::size_t member_index) noexcept {
        multimem_member_t addresses;
        if (multimem_pointer_ != 0) {
            addresses.multimem_payload = reinterpret_cast<f32_t *>(multimem_pointer_);
            addresses.multimem_flag = reinterpret_cast<u32_t *>(multimem_pointer_ + flag_offset_);
        }
        addresses.replicas = replicas_.base();
        addresses.epoch = epoch_pointer_();
        addresses.replica_stride = replicas_.member_stride();
        addresses.flag_offset = flag_offset_;
        addresses.member_index = static_cast<u32_t>(member_index);
        addresses.member_count = static_cast<u32_t>(member_count());
        return addresses;
    }

#if defined(__CUDACC__)
    /** @brief Enqueue the domain barrier for @p member_index — one arrival, one acquire-spin against
     *      the domain's own epoch (see @ref advance ). On the peer tier the arrival walks the members
     *      through the symmetric range instead. */
    [[nodiscard]] expected_t barrier(std::size_t member_index, gpu_stream_t const &stream) noexcept {
        dim3 const one {1u, 1u, 1u};
        dim3 const warp {warp_size_k, 1u, 1u};
        if (tier() == multimem_tier_t::switch_reduced_k)
            return launch_kernel(reinterpret_cast<void const *>(&multimem_barrier_kernel_), one, warp, stream,
                                 member(member_index));
        return launch_kernel(reinterpret_cast<void const *>(&multimem_barrier_peer_kernel_), one, warp, stream,
                             member(member_index));
    }

    /** @brief Enqueue member @p member_index 's reduce-scatter shard: @p destination receives the
     *      domain-wide sum of @c [shard_begin,shard_begin+destination.extent(0)) . Call after the
     *      barrier proved every member's payload complete. */
    [[nodiscard]] expected_t reduce_scatter(std::size_t member_index, f32r1_span_t destination, std::size_t shard_begin,
                                            gpu_stream_t const &stream) noexcept {
        std::size_t const shard_scalars = destination.extent(0);
        if (shard_begin + shard_scalars > payload_scalars_)
            return expected_t::fail(status_t::shape_unsupported_k,
                                    "shard [{},{}) exceeds the domain payload of {} scalars", shard_begin,
                                    shard_begin + shard_scalars, payload_scalars_);
        dim3 const block_shape {kernel_block_size_default_k, 1u, 1u};
        unsigned grid = static_cast<unsigned>((shard_scalars + kernel_block_size_default_k - 1u) /
                                              kernel_block_size_default_k);
        if (grid == 0u) grid = 1u;
        if (grid > 4096u) grid = 4096u;
        dim3 const grid_shape {grid, 1u, 1u};
        if (tier() == multimem_tier_t::switch_reduced_k)
            return launch_kernel(reinterpret_cast<void const *>(&multimem_reduce_scatter_kernel_), grid_shape,
                                 block_shape, stream, member(member_index), destination, shard_begin);
        return launch_kernel(reinterpret_cast<void const *>(&multimem_reduce_scatter_peer_kernel_), grid_shape,
                             block_shape, stream, member(member_index), destination, shard_begin);
    }
#endif // defined(__CUDACC__)

  private:
    /** @brief The barrier epoch scalar — GPU memory inside the lead replica, one word above the flag;
     *      derived, never stored, so no second copy of the layout exists. */
    [[nodiscard]] u32_t *epoch_pointer_() noexcept {
        return reinterpret_cast<u32_t *>(replicas_.member_slice(0).data() + flag_offset_ + sizeof(u32_t));
    }

    void swap_(multimem_domain_t &other) noexcept {
        std::swap(replicas_, other.replicas_);
        std::swap(multimem_handle_, other.multimem_handle_);
        std::swap(multimem_pointer_, other.multimem_pointer_);
        std::swap(host_epoch_, other.host_epoch_);
        std::swap(payload_scalars_, other.payload_scalars_);
        std::swap(flag_offset_, other.flag_offset_);
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
    /** @brief The host-side epoch counter @ref advance uploads and bumps once per step. */
    u32_t host_epoch_ = 0;
    /** @brief Float32 payload scalars per replica. */
    std::size_t payload_scalars_ = 0;
    /** @brief Byte offset of the arrival counter inside every replica; the epoch sits one word above. */
    std::size_t flag_offset_ = 0;
};

} // namespace unum::uform::primitives
