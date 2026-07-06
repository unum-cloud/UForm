/**
 *  @brief The multimem collectives' implementation: the barrier / reduce-scatter kernels and the heavy
 *      construction bodies — compiled once into the `uform` library, never per consumer.
 *  @file cuda_core/multimem_collectives.cu
 *  @author Ash Vardanian
 *  @date July 3, 2026
 */
#include "cuda_core/multimem_collectives.cuh"

namespace unum::uform::core {

/**
 *  @brief Domain barrier: lane 0 release-adds 1 into every replica's counter, then acquire-spins on its
 *      @b own counter until all @c member_count arrivals of this round landed. Monotonic — the
 *      counter never resets, so a captured step replays cleanly; the round itself is device
 *      self-advanced by @ref barrier_round_t::advance , not host-bumped.
 *
 *  __grid_size__ @c [1,1,1] , one warp; no dynamic shared memory.
 */
static __global__ void multimem_barrier_kernel_(multimem_member_t member) {
    if (threadIdx.x != 0u) return;
    u32_t const target = *member.round + member.member_count;
    multimem_red_add_release_(member.multimem_flag, 1u);
    while (multimem_load_acquire_(member.own_flag()) < target) {
        // Spin until every member's arrival of this round reached our replica.
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
    u32_t const target = *member.round + member.member_count;
    for (u32_t peer = 0; peer != member.member_count; ++peer) {
        u32_t *flag = member.peer_flag(peer);
        asm volatile("red.release.sys.global.add.u32 [%0], %1;" : : "l"(flag), "r"(1u) : "memory");
    }
    while (multimem_load_acquire_(member.own_flag()) < target) {
        // Spin until every member's arrival of this round reached our replica.
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
        for (u32_t peer = 0; peer != member.member_count; ++peer) {
            f32_t const *peer_payload = member.peer_payload(peer);
            total = __fadd_rn(total, peer_payload[shard_begin + index]);
        }
        destination[index] = total;
    }
}

expected<gpu_symmetric_arena_t> gpu_symmetric_arena_t::make(std::span<i32_t const> devices,
                                                            std::size_t bytes_per_member,
                                                            std::size_t alignment_granularity) noexcept {
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
        // Create into a local and publish only after the map succeeds, so "published handle" and
        // "mapped slice" are the same fact — teardown needs no separate high-water counter.
        CUmemGenericAllocationHandle handle = 0;
        if (CUresult result = cuMemCreate(&handle, arena.member_stride_, &properties, 0); result != CUDA_SUCCESS)
            return result_t::fail(result, "slice allocation failed (device={} bytes={})", devices[member],
                                  arena.member_stride_);
        if (CUresult result = cuMemMap(slice, arena.member_stride_, 0, handle, 0); result != CUDA_SUCCESS) {
            cuMemRelease(handle);
            return result_t::fail(result, "slice mapping failed (member={})", member);
        }
        arena.handles_[member] = handle;
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

expected<multimem_domain_t> multimem_domain_t::make(std::span<i32_t const> devices, std::size_t payload_scalars,
                                                    std::span<gpu_stream_t> streams) noexcept {
    using result_t = expected<multimem_domain_t>;
    std::size_t const member_count = devices.size();
    if (member_count == 0) return result_t::fail(status_t::invalid_input_k, "a domain needs members");
    if (streams.size() != member_count)
        return result_t::fail(status_t::invalid_input_k, "every member needs its stream ({} vs {})", member_count,
                              streams.size());
    // Host-side staging (memsets, the round reset) needs a current context; the lead stream owns one.
    switch_to_gpu_t gpu_scope {streams[0]};
    if (!gpu_scope.ok()) return result_t::fail(status_t::cuda_error_k, "context entry failed (device={})", devices[0]);

    multimem_domain_t domain;
    domain.payload_scalars_ = payload_scalars;
    // Replica layout: [payload f32×N | flag u32 | round u32], regions cache-line aligned. The
    // round scalar is live only in the lead replica, peer-read by every barrier.
    std::size_t const replica_bytes = domain.flag_offset_() + 2 * sizeof(u32_t);

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
        if (CUresult result = cuMulticastCreate(&domain.multimem_handle_, &multimem_properties); result != CUDA_SUCCESS)
            return result_t::fail(result, "multicast object creation failed (members={} bytes={})", member_count,
                                  bound_bytes);
        for (i32_t const device : devices) {
            CUdevice device_handle = 0;
            if (CUresult result = cuDeviceGet(&device_handle, device); result != CUDA_SUCCESS)
                return result_t::fail(result, "device handle failed ({})", device);
            if (CUresult result = cuMulticastAddDevice(domain.multimem_handle_, device_handle); result != CUDA_SUCCESS)
                return result_t::fail(result, "multicast add-device failed (device={})", device);
        }
        for (std::size_t member = 0; member != member_count; ++member)
            if (CUresult result = cuMulticastBindMem(domain.multimem_handle_, 0, domain.replicas_.member_handle(member),
                                                     0, bound_bytes, 0);
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

    // Zero every member's flag and the lead replica's round.
    for (std::size_t member = 0; member != member_count; ++member) {
        std::span<std::byte> member_slice = domain.replicas_.member_slice(member);
        CUdeviceptr const counters = reinterpret_cast<CUdeviceptr>(member_slice.data()) + domain.flag_offset_();
        if (CUresult result = cuMemsetD8(counters, 0, 2 * sizeof(u32_t)); result != CUDA_SUCCESS)
            return result_t::fail(result, "arrival counter reset failed (member={})", member);
    }
    // The domain's monotonic step counter, seeding the lead replica's round scalar on the lead
    // stream (ordered after the cuMemsetD8 zero above); every advance self-bumps it on the device.
    expected<barrier_round_t> round = barrier_round_t::make(u32r1_span_t {domain.round_pointer_(), 1},
                                                            static_cast<u32_t>(member_count), streams[0]);
    if (!round) return result_t::fail(std::move(round));
    domain.round_ = std::move(round.value_);
    return result_t {std::move(domain)};
}

expected_t multimem_domain_t::barrier(std::size_t member_index, gpu_stream_t &stream) noexcept {
    dim3 const one {1u, 1u, 1u};
    dim3 const warp {warp_size_k, 1u, 1u};
    if (tier() == multimem_tier_t::switch_reduced_k)
        return launch_kernel(reinterpret_cast<void const *>(&multimem_barrier_kernel_), one, warp, stream,
                             member(member_index));
    return launch_kernel(reinterpret_cast<void const *>(&multimem_barrier_peer_kernel_), one, warp, stream,
                         member(member_index));
}

expected_t multimem_domain_t::reduce_scatter(std::size_t member_index, f32r1_span_t destination,
                                             std::size_t shard_begin, gpu_stream_t &stream) noexcept {
    std::size_t const shard_scalars = destination.extent(0);
    if (shard_begin + shard_scalars > payload_scalars_)
        return expected_t::fail(status_t::shape_unsupported_k, "shard [{},{}) exceeds the domain payload of {} scalars",
                                shard_begin, shard_begin + shard_scalars, payload_scalars_);
    dim3 const block_shape {kernel_block_size_default_k, 1u, 1u};
    dim3 const grid_shape {grid_size_for(shard_scalars), 1u, 1u};
    if (tier() == multimem_tier_t::switch_reduced_k)
        return launch_kernel(reinterpret_cast<void const *>(&multimem_reduce_scatter_kernel_), grid_shape, block_shape,
                             stream, member(member_index), destination, shard_begin);
    return launch_kernel(reinterpret_cast<void const *>(&multimem_reduce_scatter_peer_kernel_), grid_shape, block_shape,
                         stream, member(member_index), destination, shard_begin);
}

} // namespace unum::uform::core
