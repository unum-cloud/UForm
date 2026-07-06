/**
 *  @brief The multimem collectives' implementation: the barrier / reduce-scatter kernels and the heavy
 *      construction bodies — compiled once into the `uform` library, never per consumer.
 *  @file cuda_core/multimem_collectives.cu
 *  @author Ash Vardanian
 *  @date July 3, 2026
 */
#include <charconv> // `std::from_chars` — fabric-handle hex parsing

#include "cuda_core/multimem_collectives.cuh"

namespace unum::uform::core {

/** @brief A fabric handle is 64 opaque bytes; the mesh rendezvous carries text, so it travels 128-hex. */
static bool format_fabric_handle_(sz::string &text, CUmemFabricHandle const &handle) noexcept {
    bool built = true;
    for (unsigned char const byte : handle.data) built = built && format_into(text, "{:02x}", static_cast<u32_t>(byte));
    return built;
}

/** @brief Parse a 128-hex fabric-handle token back into its 64 bytes; @c false on any non-hex or short token. */
static bool parse_fabric_handle_(sz::string_view token, CUmemFabricHandle &handle) noexcept {
    if (token.size() != 2u * sizeof(handle.data)) return false;
    for (std::size_t index = 0; index != sizeof(handle.data); ++index) {
        char const *pair = token.data() + 2u * index;
        unsigned value = 0;
        std::from_chars_result const result = std::from_chars(pair, pair + 2, value, 16);
        if (result.ec != std::errc {} || result.ptr != pair + 2) return false;
        handle.data[index] = static_cast<unsigned char>(value);
    }
    return true;
}

/**
 *  @brief Domain barrier: lane 0 release-adds 1 into every replica's counter, then acquire-spins on its
 *      @b own counter until all @c member_count arrivals of this round landed, and finally self-advances
 *      its @b own round scalar. Monotonic — neither counter ever resets, so a captured step replays
 *      cleanly, and every round read and write stays on the member's own stream: no enqueue order can
 *      hand a barrier a stale round.
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
    *member.round = target; // Self-advance: only this member's own kernels ever read its round.
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
    *member.round = target; // Self-advance: only this member's own kernels ever read its round.
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

/** @brief Grant every local device read-write access to the slice mapped at @p slice — shared by the
 *      local-create and peer-adopt paths, so the peer-read tier reads every replica through the range. */
static expected_t grant_local_access_(CUdeviceptr slice, std::size_t bytes, std::span<i32_t const> local_devices,
                                      u32_t member) noexcept {
    for (i32_t const reader : local_devices) {
        CUmemAccessDesc access {};
        access.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
        access.location.id = reader;
        access.flags = CU_MEM_ACCESS_FLAGS_PROT_READWRITE;
        if (CUresult result = cuMemSetAccess(slice, bytes, &access, 1); result != CUDA_SUCCESS)
            return expected_t::fail(result, "slice access grant failed (member={} reader={})", member, reader);
    }
    return {};
}

expected<gpu_symmetric_arena_t> gpu_symmetric_arena_t::make(std::span<i32_t const> local_devices, u32_t member_count,
                                                            u32_t member_offset, std::size_t bytes_per_member,
                                                            std::size_t alignment_granularity) noexcept {
    using result_t = expected<gpu_symmetric_arena_t>;
    if (local_devices.empty() || bytes_per_member == 0 || member_count == 0 ||
        member_offset + local_devices.size() > member_count)
        return result_t::fail(status_t::invalid_input_k,
                              "the local slices must lie inside the domain (local={} offset={} count={} bytes={})",
                              local_devices.size(), member_offset, member_count, bytes_per_member);
    gpu_symmetric_arena_t arena;
    arena.bytes_per_member_ = bytes_per_member;
    if (arena.devices_.try_assign({local_devices.data(), local_devices.size()}) != status_t::success_k ||
        arena.handles_.try_resize(member_count) != status_t::success_k)
        return result_t::fail(status_t::bad_alloc_k, "member table allocation failed ({} of {})", local_devices.size(),
                              member_count);

    // The stride honors the coarsest requirement: every local device's physical granularity plus the
    // caller's alignment (a multicast binding granularity, typically).
    std::size_t stride = alignment_granularity != 0 ? alignment_granularity : 1;
    for (i32_t const device : local_devices) {
        CUmemAllocationProp properties = allocation_properties_(device);
        std::size_t granularity = 0;
        if (CUresult result = cuMemGetAllocationGranularity(&granularity, &properties,
                                                            CU_MEM_ALLOC_GRANULARITY_RECOMMENDED);
            result != CUDA_SUCCESS)
            return result_t::fail(result, "granularity query failed (device={})", device);
        if (granularity > stride) stride = granularity;
    }
    arena.member_stride_ = round_up_to_multiple(bytes_per_member, stride);

    // Reserve the WHOLE domain's VA — remote members map into it through `adopt_peer`.
    std::size_t const total_bytes = arena.member_stride_ * member_count;
    if (CUresult result = cuMemAddressReserve(&arena.base_, total_bytes, 0, 0, 0); result != CUDA_SUCCESS)
        return result_t::fail(result, "symmetric VA reservation failed ({} bytes)", total_bytes);

    // Create + map this process's local slices at their domain-member offsets.
    for (std::size_t local = 0; local != local_devices.size(); ++local) {
        u32_t const member = member_offset + static_cast<u32_t>(local);
        CUmemAllocationProp properties = allocation_properties_(local_devices[local]);
        CUdeviceptr const slice = arena.base_ + static_cast<std::size_t>(member) * arena.member_stride_;
        // Create into a local and publish only after the map succeeds, so "published handle" and
        // "mapped slice" are the same fact — teardown needs no separate high-water counter.
        CUmemGenericAllocationHandle handle = 0;
        if (CUresult result = cuMemCreate(&handle, arena.member_stride_, &properties, 0); result != CUDA_SUCCESS)
            return result_t::fail(result, "slice allocation failed (device={} bytes={})", local_devices[local],
                                  arena.member_stride_);
        if (CUresult result = cuMemMap(slice, arena.member_stride_, 0, handle, 0); result != CUDA_SUCCESS) {
            cuMemRelease(handle);
            return result_t::fail(result, "slice mapping failed (member={})", member);
        }
        arena.handles_[member] = handle;
        if (expected_t granted = grant_local_access_(slice, arena.member_stride_, arena.devices_.view(), member);
            !granted)
            return result_t::fail(std::move(granted));
    }
    return result_t {std::move(arena)};
}

expected<CUmemFabricHandle> gpu_symmetric_arena_t::export_fabric_handle(std::size_t member) const noexcept {
    using result_t = expected<CUmemFabricHandle>;
    CUmemFabricHandle fabric {};
    if (CUresult result = cuMemExportToShareableHandle(&fabric, handles_[member], CU_MEM_HANDLE_TYPE_FABRIC, 0);
        result != CUDA_SUCCESS)
        return result_t::fail(result, "slice fabric-handle export failed (member={})", member);
    return result_t {fabric};
}

expected_t gpu_symmetric_arena_t::adopt_peer(std::size_t member, CUmemFabricHandle const &handle) noexcept {
    CUmemGenericAllocationHandle imported = 0;
    if (CUresult result = cuMemImportFromShareableHandle(
            &imported, const_cast<void *>(static_cast<void const *>(&handle)), CU_MEM_HANDLE_TYPE_FABRIC);
        result != CUDA_SUCCESS)
        return expected_t::fail(result, "peer slice import failed (member={})", member);
    CUdeviceptr const slice = base_ + member * member_stride_;
    if (CUresult result = cuMemMap(slice, member_stride_, 0, imported, 0); result != CUDA_SUCCESS) {
        cuMemRelease(imported);
        return expected_t::fail(result, "peer slice mapping failed (member={})", member);
    }
    handles_[member] = imported;
    return grant_local_access_(slice, member_stride_, devices_.view(), static_cast<u32_t>(member));
}

expected<multimem_domain_t> multimem_domain_t::make(mesh_topology_t const &topology,
                                                    std::span<i32_t const> local_devices, std::size_t payload_scalars,
                                                    std::span<gpu_stream_t> local_streams) noexcept {
    using result_t = expected<multimem_domain_t>;
    std::size_t const local_count = local_devices.size();
    if (local_count == 0) return result_t::fail(status_t::invalid_input_k, "a domain needs local devices");
    if (local_streams.size() != local_count)
        return result_t::fail(status_t::invalid_input_k, "every local device needs its stream ({} vs {})", local_count,
                              local_streams.size());

    // The island's geometry comes from the world, not the local span: it may cross processes. The domain's
    // host_domains are the contiguous run [domain_first_host, +host_domains_per_domain); this process's
    // slices occupy members [member_offset, member_offset+local_count).
    mesh_shape_t const &shape = topology.shape;
    u32_t const gpus_per_host = shape.gpus_per_host_domain;
    u32_t const member_count = shape.gpus_per_multimem_domain;
    u32_t const member_offset = shape.index_in_multimem_domain(topology.first_gpu_index());
    u32_t const host_domains_per_domain = member_count / gpus_per_host;
    u32_t const domain_first_host = topology.multimem_domain_index() * host_domains_per_domain;
    bool const is_lead = member_offset == 0;
    f64_t const timeout_seconds = 120.0;

    // Host-side staging (creates, binds, memsets) needs a current context; the lead local stream owns
    // one — every VMM call names its device explicitly, so one context serves all. No stream is
    // retained: the domain outlives its bring-up streams by design.
    switch_to_gpu_t gpu_scope {local_streams[0]};
    if (!gpu_scope.ok())
        return result_t::fail(status_t::cuda_error_k, "context entry failed (device={})", local_devices[0]);

    multimem_domain_t domain;
    domain.payload_scalars_ = payload_scalars;
    domain.member_offset_ = member_offset;
    // Replica layout: [payload f32×N | flag u32 | round u32], regions cache-line aligned. Each member's
    // round scalar lives in its own replica, self-advanced by its barrier kernels.
    std::size_t const replica_bytes = domain.flag_offset_() + 2 * sizeof(u32_t);

    // A 1-member domain takes the peer-read tier outright: cuMulticastCreate rejects a single-device
    // object, and a 1-member "multicast" is a self-read anyway.
    bool multimem_everywhere = member_count > 1;
    for (i32_t const device : local_devices) {
        int supported = 0;
        if (CUresult result = cuDeviceGetAttribute(&supported, CU_DEVICE_ATTRIBUTE_MULTICAST_SUPPORTED, device);
            result != CUDA_SUCCESS)
            return result_t::fail(result, "multicast attribute query failed (device={})", device);
        multimem_everywhere = multimem_everywhere && supported != 0;
    }

    // The multicast binding granularity dictates the replica stride; the symmetric arena also
    // honors every local device's physical granularity.
    std::size_t multimem_granularity = 1;
    CUmulticastObjectProp multimem_properties {};
    if (multimem_everywhere) {
        multimem_properties.numDevices = member_count;
        multimem_properties.handleTypes = CU_MEM_HANDLE_TYPE_FABRIC;
        multimem_properties.size = replica_bytes;
        if (CUresult result = cuMulticastGetGranularity(&multimem_granularity, &multimem_properties,
                                                        CU_MULTICAST_GRANULARITY_RECOMMENDED);
            result != CUDA_SUCCESS)
            return result_t::fail(result, "multicast granularity query failed");
    }

    // 1. Reserve the whole domain VA and create this process's local slices.
    expected<gpu_symmetric_arena_t> replicas = gpu_symmetric_arena_t::make(local_devices, member_count, member_offset,
                                                                           replica_bytes, multimem_granularity);
    if (!replicas) return result_t::fail(std::move(replicas));
    domain.replicas_ = std::move(replicas.value_);

    // 2. Exchange slice fabric handles across the domain's processes and adopt every peer's slices, so the
    //    whole symmetric range is mapped in every process. A single-process island skips it.
    if (host_domains_per_domain > 1) {
        sz::string slices_text;
        for (std::size_t local = 0; local != local_count; ++local) {
            expected<CUmemFabricHandle> exported = domain.replicas_.export_fabric_handle(member_offset +
                                                                                         static_cast<u32_t>(local));
            if (!exported) return result_t::fail(std::move(exported));
            if ((local != 0 && !format_into(slices_text, " ")) || !format_fabric_handle_(slices_text, exported.value_))
                return result_t::fail(status_t::bad_alloc_k, "slice handle payload assembly failed");
        }
        if (expected_t published = topology.publish_text("multimem.slice", slices_text.view()); !published)
            return result_t::fail(std::move(published));
        for (u32_t peer_host = domain_first_host; peer_host != domain_first_host + host_domains_per_domain;
             ++peer_host) {
            if (peer_host == topology.host_domain_index) continue;
            sz::string peer_text;
            if (expected_t awaited = topology.await_text(peer_host, "multimem.slice", peer_text, timeout_seconds);
                !awaited)
                return result_t::fail(std::move(awaited));
            u32_t const peer_member_base = (peer_host - domain_first_host) * gpus_per_host;
            u32_t adopted_count = 0;
            for (sz::string_view token : peer_text.view().split(sz::whitespaces_set())) {
                if (token.empty()) continue;
                if (adopted_count >= gpus_per_host)
                    return result_t::fail(status_t::io_error_k, "host_domain {} published too many slice handles",
                                          peer_host);
                CUmemFabricHandle handle {};
                if (!parse_fabric_handle_(token, handle))
                    return result_t::fail(status_t::io_error_k, "malformed slice handle from host_domain {}",
                                          peer_host);
                if (expected_t adopted = domain.replicas_.adopt_peer(peer_member_base + adopted_count, handle);
                    !adopted)
                    return result_t::fail(std::move(adopted));
                ++adopted_count;
            }
            if (adopted_count != gpus_per_host)
                return result_t::fail(status_t::io_error_k, "host_domain {} published {} slice handles, expected {}",
                                      peer_host, adopted_count, gpus_per_host);
        }
    }

    // 3. The multicast object: the domain lead creates it and publishes its fabric handle; every process
    //    imports it, adds its local devices, binds its local slices at multicast offset 0, and maps the VA.
    if (multimem_everywhere) {
        std::size_t const bound_bytes = domain.replicas_.member_stride();
        multimem_properties.size = bound_bytes;
        if (is_lead) {
            if (CUresult result = cuMulticastCreate(&domain.multimem_handle_, &multimem_properties);
                result != CUDA_SUCCESS)
                return result_t::fail(result, "multicast object creation failed (members={} bytes={})", member_count,
                                      bound_bytes);
            if (host_domains_per_domain > 1) {
                CUmemFabricHandle fabric {};
                if (CUresult result = cuMemExportToShareableHandle(&fabric, domain.multimem_handle_,
                                                                   CU_MEM_HANDLE_TYPE_FABRIC, 0);
                    result != CUDA_SUCCESS)
                    return result_t::fail(result, "multicast fabric-handle export failed");
                sz::string mc_text;
                if (!format_fabric_handle_(mc_text, fabric))
                    return result_t::fail(status_t::bad_alloc_k, "multicast handle payload assembly failed");
                if (expected_t published = topology.publish_text("multimem.mc", mc_text.view()); !published)
                    return result_t::fail(std::move(published));
            }
        }
        else {
            sz::string mc_text;
            if (expected_t awaited = topology.await_text(domain_first_host, "multimem.mc", mc_text, timeout_seconds);
                !awaited)
                return result_t::fail(std::move(awaited));
            CUmemFabricHandle fabric {};
            bool parsed = false;
            for (sz::string_view token : mc_text.view().split(sz::whitespaces_set())) {
                if (token.empty()) continue;
                parsed = parse_fabric_handle_(token, fabric);
                break;
            }
            if (!parsed)
                return result_t::fail(status_t::io_error_k, "malformed multicast handle from host_domain {}",
                                      domain_first_host);
            if (CUresult result = cuMemImportFromShareableHandle(&domain.multimem_handle_, &fabric,
                                                                 CU_MEM_HANDLE_TYPE_FABRIC);
                result != CUDA_SUCCESS)
                return result_t::fail(result, "multicast object import failed");
        }
        // Add this process's local devices and bind its local slices — no peer dependency, so no barrier.
        for (i32_t const device : local_devices) {
            CUdevice device_handle = 0;
            if (CUresult result = cuDeviceGet(&device_handle, device); result != CUDA_SUCCESS)
                return result_t::fail(result, "device handle failed ({})", device);
            if (CUresult result = cuMulticastAddDevice(domain.multimem_handle_, device_handle); result != CUDA_SUCCESS)
                return result_t::fail(result, "multicast add-device failed (device={})", device);
        }
        for (std::size_t local = 0; local != local_count; ++local) {
            u32_t const member = member_offset + static_cast<u32_t>(local);
            if (CUresult result = cuMulticastBindMem(domain.multimem_handle_, 0, domain.replicas_.member_handle(member),
                                                     0, bound_bytes, 0);
                result != CUDA_SUCCESS)
                return result_t::fail(result, "multicast bind failed (member={})", member);
        }
        if (CUresult result = cuMemAddressReserve(&domain.multimem_pointer_, bound_bytes, 0, 0, 0);
            result != CUDA_SUCCESS)
            return result_t::fail(result, "multicast reservation failed");
        if (CUresult result = cuMemMap(domain.multimem_pointer_, bound_bytes, 0, domain.multimem_handle_, 0);
            result != CUDA_SUCCESS)
            return result_t::fail(result, "multicast mapping failed");
        if (expected_t granted = grant_local_access_(domain.multimem_pointer_, bound_bytes, local_devices,
                                                     member_offset);
            !granted)
            return result_t::fail(std::move(granted));
    }

    // 4. Zero this process's local slices' arrival flag and round scalar — each process owns its
    //    physical replicas, and a zero round makes the first barrier's target exactly one round of
    //    arrivals; the barrier kernels self-advance from there.
    for (std::size_t local = 0; local != local_count; ++local) {
        u32_t const member = member_offset + static_cast<u32_t>(local);
        std::span<std::byte> member_slice = domain.replicas_.member_slice(member);
        CUdeviceptr const counters = reinterpret_cast<CUdeviceptr>(member_slice.data()) + domain.flag_offset_();
        if (CUresult result = cuMemsetD8(counters, 0, 2 * sizeof(u32_t)); result != CUDA_SUCCESS)
            return result_t::fail(result, "arrival counter reset failed (member={})", member);
    }

    // 5. Every process has finished binding before the first collective runs.
    if (expected_t synchronized = topology.barrier("multimem", timeout_seconds); !synchronized)
        return result_t::fail(std::move(synchronized));
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
