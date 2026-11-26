/**
 *  @file src/utilities/multimem_collectives.cu
 *  @author Ash Vardanian
 *  @date July 3, 2026
 *  @brief The multimem collectives' implementation: the peer-read barrier / reduce-scatter kernels,
 *      the switch bundle's loader, and the heavy construction bodies — compiled once into the
 *      @c uform_train library, never per consumer.
 */
#include <charconv> // `std::from_chars` — fabric-handle hex parsing

#include "uform/utilities/multimem_collectives.cuh"

namespace util = unum::uform::utilities;

/** The fatbin of <tt>multimem_switch.cu</tt> the build embeds, empty when it lists no
 *  multicast-capable code. */
extern "C" util::b8_t const uform_multimem_switch_bundle_start[], uform_multimem_switch_bundle_end[];

/** A fabric handle is 64 opaque bytes; the mesh rendezvous sends text, so it travels 128-hex. */
static bool format_fabric_handle_(sz::string_t& text, CUmemFabricHandle const& handle) noexcept {
    bool built = true;
    for (unsigned char const byte : handle.data)
        built = built && util::format_into(text, "{:02x}", static_cast<util::u32_t>(byte));
    return built;
}

/** Parse a 128-hex fabric-handle token back into its 64 bytes; @c false on any non-hex
 *  or short token. */
static bool parse_fabric_handle_(sz::string_view_t token, CUmemFabricHandle& handle) noexcept {
    if (token.size() != 2u * sizeof(handle.data)) return false;
    for (std::size_t index = 0; index != sizeof(handle.data); ++index) {
        char const* pair = token.data() + 2u * index;
        unsigned value = 0;
        std::from_chars_result const result = std::from_chars(pair, pair + 2, value, 16);
        if (result.ec != std::errc {} || result.ptr != pair + 2) return false;
        handle.data[index] = static_cast<unsigned char>(value);
    }
    return true;
}

/**
 *  @brief Peer-read sibling of @c multimem_barrier_switch_kernel_ for the no-multicast tier: lane 0
 *      adds into every member's counter through the symmetric range, then spins on its own, and
 *      self-advances its round.
 *
 *  Traps once the spin outlasts @c member.wait_nanoseconds .
 *
 *  __grid_size__ @c [1,1,1] , one warp; no dynamic shared memory.
 */
static __global__ void multimem_barrier_peer_kernel_(util::multimem_member_t member) {
    if (threadIdx.x != 0u) return;
    util::u32_t const target = *member.round + member.member_count;
    for (util::u32_t peer = 0; peer != member.member_count; ++peer) {
        util::u32_t* flag = member.peer_flag(peer);
        asm volatile("red.release.sys.global.add.u32 [%0], %1;" : : "l"(flag), "r"(1u) : "memory");
    }
    util::u64_t const deadline = util::global_nanoseconds() + member.wait_nanoseconds;
    while (multimem_load_acquire_(member.own_flag()) < target)
        if (util::global_nanoseconds() > deadline)
            util::trap_with(member.fault, util::gpu_fault_code_t::barrier_deadline_k, member.member_index, target, 0u);
    __threadfence_system();
    *member.round = target; // Self-advance: only this member's own kernels ever read its round.
}

/**
 *  @brief Peer-read sibling of @c multimem_reduce_scatter_switch_kernel_ : the same shard,
 *      folded serially over the members through the symmetric range — a fixed order,
 *      bit-identical on every member.
 *
 *  __grid_size__ grid-stride over @c shard_scalars ; no dynamic shared memory.
 */
static __global__ void multimem_reduce_scatter_peer_kernel_(util::multimem_member_t member,
                                                            util::f32r1_span_t destination, std::size_t shard_begin) {
    std::size_t const shard_scalars = destination.extent(0);
    std::size_t const stride = static_cast<std::size_t>(gridDim.x) * blockDim.x;
    for (std::size_t index = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x; index < shard_scalars;
         index += stride) {
        util::f32_t total = 0.0f;
        for (util::u32_t peer = 0; peer != member.member_count; ++peer) {
            util::f32_t const* peer_payload = member.peer_payload(peer);
            total = __fadd_rn(total, peer_payload[shard_begin + index]);
        }
        destination[index] = total;
    }
}

/** Grant every local device read-write access to the slice mapped at @p slice — shared by
 *  the local-create and peer-adopt paths, so the peer-read tier reads every replica
 *  through the range. */
static util::expected_t grant_local_access_(CUdeviceptr slice, std::size_t bytes,
                                            std::span<util::i32_t const> local_devices, util::u32_t member) noexcept {
    for (util::i32_t const reader : local_devices) {
        CUmemAccessDesc access {};
        access.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
        access.location.id = reader;
        access.flags = CU_MEM_ACCESS_FLAGS_PROT_READWRITE;
        if (CUresult result = cuMemSetAccess(slice, bytes, &access, 1); result != CUDA_SUCCESS)
            return util::cuda_failure(result, "slice access grant failed (member={} reader={})", member, reader);
    }
    return {};
}

/** Adopt the slice handles host_domain @p peer_host published in @p peer_text into @p replicas ,
 *  from member @p member_base on, requiring exactly @p gpus_per_host well-formed handles. */
[[nodiscard]] static util::expected_t adopt_peer_slices_(util::gpu_symmetric_arena_t& replicas,
                                                         sz::string_view_t peer_text, util::u32_t peer_host,
                                                         util::u32_t member_base, util::u32_t gpus_per_host) noexcept {
    util::u32_t adopted_count = 0;
    for (sz::string_view_t token : peer_text.split(sz::whitespaces_set())) {
        if (token.empty()) continue;
        if (adopted_count >= gpus_per_host)
            return util::expected_t::fail(util::status_code_t::io_k, "host_domain {} published too many slice handles",
                                          peer_host);
        CUmemFabricHandle handle {};
        if (!parse_fabric_handle_(token, handle))
            return util::expected_t::fail(util::status_code_t::io_k, "malformed slice handle from host_domain {}",
                                          peer_host);
        if (util::expected_t adopted = replicas.adopt_peer(member_base + adopted_count, handle); !adopted)
            return adopted;
        ++adopted_count;
    }
    if (adopted_count != gpus_per_host)
        return util::expected_t::fail(util::status_code_t::io_k,
                                      "host_domain {} published {} slice handles, expected {}", peer_host,
                                      adopted_count, gpus_per_host);
    return {};
}

/** Export the multicast object @p multimem_handle as a fabric handle and publish it under @p key
 *  for the domain's other processes to import. */
[[nodiscard]] static util::expected_t publish_multicast_handle_(CUmemGenericAllocationHandle multimem_handle,
                                                                util::mesh_topology_t const& topology,
                                                                sz::string_view_t key) noexcept {
    CUmemFabricHandle fabric {};
    if (CUresult result = cuMemExportToShareableHandle(&fabric, multimem_handle, CU_MEM_HANDLE_TYPE_FABRIC, 0);
        result != CUDA_SUCCESS)
        return util::cuda_failure(result, "multicast fabric-handle export failed");
    sz::string_t mc_text;
    if (!format_fabric_handle_(mc_text, fabric))
        return util::expected_t::fail(util::status_code_t::out_of_memory_k, "Multicast handle payload assembly failed");
    return topology.publish_text(key, mc_text.view());
}

util::expected<util::multimem_switch_module_t> util::multimem_switch_module_t::load(util::i32_t device) noexcept {
    using result_t = util::expected<util::multimem_switch_module_t>;
    if (uform_multimem_switch_bundle_end - uform_multimem_switch_bundle_start == 0)
        return result_t::fail(util::status_code_t::unsupported_k, "This build compiled no switch kernels");
    util::multimem_switch_module_t bundle;
    CUdevice device_handle = 0;
    if (CUresult result = cuDeviceGet(&device_handle, device); result != CUDA_SUCCESS)
        return util::cuda_failure<util::multimem_switch_module_t>(result, "device handle query failed (device={})",
                                                                  device);
    if (CUresult result = cuDevicePrimaryCtxRetain(&bundle.context_, device_handle); result != CUDA_SUCCESS)
        return util::cuda_failure<util::multimem_switch_module_t>(result, "primary context retain failed (device={})",
                                                                  device);
    bundle.device_ = device;
    if (CUresult result = cuCtxPushCurrent(bundle.context_); result != CUDA_SUCCESS)
        return util::cuda_failure<util::multimem_switch_module_t>(result, "context push failed (device={})", device);
    CUresult loaded = cuModuleLoadData(&bundle.module_, uform_multimem_switch_bundle_start);
    if (loaded == CUDA_SUCCESS)
        loaded = cuModuleGetFunction(&bundle.kernels_.barrier, bundle.module_, "multimem_barrier_switch_kernel_");
    if (loaded == CUDA_SUCCESS)
        loaded = cuModuleGetFunction(&bundle.kernels_.reduce_scatter, bundle.module_,
                                     "multimem_reduce_scatter_switch_kernel_");
    if (loaded == CUDA_SUCCESS)
        loaded = cuModuleGetFunction(&bundle.kernels_.reduce_gradient, bundle.module_,
                                     "adafactor_reduce_gradient_switch_kernel_");
    cuCtxPopCurrent(nullptr);
    if (loaded != CUDA_SUCCESS)
        return util::cuda_failure<util::multimem_switch_module_t>(loaded, "switch bundle load failed (device={})",
                                                                  device);
    return result_t {std::move(bundle)};
}

void util::multimem_switch_module_t::destroy_() noexcept {
    if (context_ == nullptr) return;
    if (module_ != nullptr && cuCtxPushCurrent(context_) == CUDA_SUCCESS) {
        cuModuleUnload(module_);
        cuCtxPopCurrent(nullptr);
    }
    CUdevice device_handle = 0;
    if (cuDeviceGet(&device_handle, device_) == CUDA_SUCCESS) cuDevicePrimaryCtxRelease(device_handle);
    context_ = nullptr;
    module_ = nullptr;
    kernels_ = {};
}

util::expected<util::multimem_tier_choice_t> util::multimem_tier_choice_t::make(
    util::u32_t member_count, std::span<util::i32_t const> local_devices) noexcept {
    using result_t = util::expected<util::multimem_tier_choice_t>;
    util::multimem_tier_choice_t choice;
    // A 1-member domain takes the peer-read tier outright: cuMulticastCreate rejects a
    // single-device object, and a 1-member "multicast" is a self-read anyway.
    if (member_count == 1) {
        if (!util::format_into(choice.reason, "peer-read: a one-member domain reads its own replica"))
            return result_t::fail(util::status_code_t::out_of_memory_k, "Tier reason formatting failed");
        return result_t {std::move(choice)};
    }
    for (util::i32_t const device : local_devices) {
        int supported = 0;
        if (CUresult result = cuDeviceGetAttribute(&supported, CU_DEVICE_ATTRIBUTE_MULTICAST_SUPPORTED, device);
            result != CUDA_SUCCESS)
            return util::cuda_failure<util::multimem_tier_choice_t>(
                result, "multicast attribute query failed (device={})", device);
        if (supported != 0) continue;
        if (!util::format_into(choice.reason, "peer-read: device {} has no switch multicast", device))
            return result_t::fail(util::status_code_t::out_of_memory_k, "Tier reason formatting failed");
        return result_t {std::move(choice)};
    }
    for (util::i32_t const device : local_devices) {
        auto [bundle, status] = util::multimem_switch_module_t::load(device);
        if (!status) {
            choice.modules = {};
            if (!util::format_into(choice.reason,
                                   "peer-read: device {} has switch multicast, but the switch kernels do not " //
                                   "load on it: {}",
                                   device,
                                   (status.description_.empty() ? sz::string_view_t {util::to_string(status.status_)}
                                                                : status.description_.view())))
                return result_t::fail(util::status_code_t::out_of_memory_k, "Tier reason formatting failed");
            return result_t {std::move(choice)};
        }
        if (st::failed(choice.modules.push_back(std::move(bundle))))
            return result_t::fail(util::status_code_t::out_of_memory_k, "No room for the switch module table");
    }
    choice.tier = util::multimem_tier_t::switch_reduced_k;
    if (!util::format_into(choice.reason, "switch-reduced across {} members", member_count))
        return result_t::fail(util::status_code_t::out_of_memory_k, "Tier reason formatting failed");
    return result_t {std::move(choice)};
}

util::expected<CUmemAllocationHandleType> util::gpu_symmetric_arena_t::shareable_handle_type(
    std::span<util::i32_t const> local_devices, util::u32_t member_count) noexcept {
    using result_t = util::expected<CUmemAllocationHandleType>;
    if (member_count <= local_devices.size()) return result_t {CU_MEM_HANDLE_TYPE_NONE};
    for (util::i32_t const device : local_devices) {
        int supported = 0;
        if (CUresult result = cuDeviceGetAttribute(&supported, CU_DEVICE_ATTRIBUTE_HANDLE_TYPE_FABRIC_SUPPORTED,
                                                   device);
            result != CUDA_SUCCESS)
            return util::cuda_failure<CUmemAllocationHandleType>(
                result, "fabric-handle attribute query failed (device={})", device);
        if (supported == 0) return result_t {CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR};
    }
    return result_t {CU_MEM_HANDLE_TYPE_FABRIC};
}

util::expected<util::gpu_symmetric_arena_t> util::gpu_symmetric_arena_t::make(
    std::span<util::i32_t const> local_devices, util::u32_t member_count, util::u32_t member_offset,
    std::size_t bytes_per_member, std::size_t alignment_granularity) noexcept {
    using result_t = util::expected<util::gpu_symmetric_arena_t>;
    if (local_devices.empty() || bytes_per_member == 0 || member_count == 0 ||
        member_offset + local_devices.size() > member_count)
        return result_t::fail(util::status_code_t::invalid_argument_k,
                              "the local slices must lie inside the domain (local={} offset={} count={} bytes={})",
                              local_devices.size(), member_offset, member_count, bytes_per_member);
    auto [handle_type, status] = shareable_handle_type(local_devices, member_count);
    if (!status) return result_t::fail(std::move(status));
    util::gpu_symmetric_arena_t arena;
    arena.handle_type_ = handle_type;
    arena.bytes_per_member_ = bytes_per_member;
    if (st::failed(arena.devices_.assign(local_devices)) || st::failed(arena.handles_.resize(member_count)))
        return result_t::fail(util::status_code_t::out_of_memory_k, "member table allocation failed ({} of {})",
                              local_devices.size(), member_count);

    // The stride honors the coarsest requirement: every local device's physical granularity plus
    // the caller's alignment (a multicast binding granularity, typically).
    std::size_t stride = alignment_granularity != 0 ? alignment_granularity : 1;
    for (util::i32_t const device : local_devices) {
        CUmemAllocationProp properties = arena.allocation_properties_(device);
        std::size_t granularity = 0;
        if (CUresult result = cuMemGetAllocationGranularity(&granularity, &properties,
                                                            CU_MEM_ALLOC_GRANULARITY_RECOMMENDED);
            result != CUDA_SUCCESS)
            return util::cuda_failure<util::gpu_symmetric_arena_t>(result, "granularity query failed (device={})",
                                                                   device);
        if (granularity > stride) stride = granularity;
    }
    arena.member_stride_ = util::round_up_to_multiple(bytes_per_member, stride);

    // Reserve the WHOLE domain's VA — remote members map into it through `adopt_peer`.
    std::size_t const total_bytes = arena.member_stride_ * member_count;
    if (CUresult result = cuMemAddressReserve(&arena.base_, total_bytes, 0, 0, 0); result != CUDA_SUCCESS)
        return util::cuda_failure<util::gpu_symmetric_arena_t>(result, "symmetric VA reservation failed ({} bytes)",
                                                               total_bytes);

    // Create + map this process's local slices at their domain-member offsets.
    for (std::size_t local = 0; local != local_devices.size(); ++local) {
        util::u32_t const member = member_offset + static_cast<util::u32_t>(local);
        CUmemAllocationProp properties = arena.allocation_properties_(local_devices[local]);
        CUdeviceptr const slice = arena.base_ + static_cast<std::size_t>(member) * arena.member_stride_;
        // Create into a local and publish only after the map succeeds, so "published handle" and
        // "mapped slice" are the same fact — teardown needs no separate high-water counter.
        CUmemGenericAllocationHandle handle = 0;
        if (CUresult result = cuMemCreate(&handle, arena.member_stride_, &properties, 0); result != CUDA_SUCCESS)
            return util::cuda_failure<util::gpu_symmetric_arena_t>(
                result, "slice allocation failed (device={} bytes={})", local_devices[local], arena.member_stride_);
        if (CUresult result = cuMemMap(slice, arena.member_stride_, 0, handle, 0); result != CUDA_SUCCESS) {
            cuMemRelease(handle);
            return util::cuda_failure<util::gpu_symmetric_arena_t>(result, "slice mapping failed (member={})", member);
        }
        arena.handles_[member] = handle;
        if (util::expected_t granted = grant_local_access_(slice, arena.member_stride_, arena.devices_, member);
            !granted)
            return result_t::fail(std::move(granted));
    }
    return result_t {std::move(arena)};
}

util::expected<CUmemFabricHandle> util::gpu_symmetric_arena_t::export_fabric_handle(std::size_t member) const noexcept {
    using result_t = util::expected<CUmemFabricHandle>;
    if (handle_type_ != CU_MEM_HANDLE_TYPE_FABRIC)
        return result_t::fail(util::status_code_t::unsupported_k,
                              "member {}'s slice has no fabric handle: a domain spanning processes on devices " //
                              "without fabric handles needs a POSIX file-descriptor handoff, not implemented yet",
                              member);
    CUmemFabricHandle fabric {};
    if (CUresult result = cuMemExportToShareableHandle(&fabric, handles_[member], CU_MEM_HANDLE_TYPE_FABRIC, 0);
        result != CUDA_SUCCESS)
        return util::cuda_failure<CUmemFabricHandle>(result, "slice fabric-handle export failed (member={})", member);
    return result_t {fabric};
}

util::expected_t util::gpu_symmetric_arena_t::adopt_peer(std::size_t member, CUmemFabricHandle const& handle) noexcept {
    CUmemGenericAllocationHandle imported = 0;
    CUmemFabricHandle shared = handle; // the driver takes a writable pointer it only reads through
    if (CUresult result = cuMemImportFromShareableHandle(&imported, &shared, CU_MEM_HANDLE_TYPE_FABRIC);
        result != CUDA_SUCCESS)
        return util::cuda_failure(result, "peer slice import failed (member={})", member);
    CUdeviceptr const slice = base_ + member * member_stride_;
    if (CUresult result = cuMemMap(slice, member_stride_, 0, imported, 0); result != CUDA_SUCCESS) {
        cuMemRelease(imported);
        return util::cuda_failure(result, "peer slice mapping failed (member={})", member);
    }
    handles_[member] = imported;
    return grant_local_access_(slice, member_stride_, devices_, static_cast<util::u32_t>(member));
}

util::expected<util::multimem_domain_t> util::multimem_domain_t::make(util::mesh_topology_t& topology,
                                                                      std::span<util::i32_t const> local_devices,
                                                                      std::size_t payload_scalars,
                                                                      util::cuda_executor_t& local_streams,
                                                                      util::f64_t timeout_seconds) noexcept {
    using result_t = util::expected<util::multimem_domain_t>;
    std::size_t const local_count = local_devices.size();
    if (local_count == 0)
        return result_t::fail(util::status_code_t::invalid_argument_k, "A domain needs local devices");
    if (local_streams.size() != local_count)
        return result_t::fail(util::status_code_t::invalid_argument_k, "every local device needs its stream ({} vs {})",
                              local_count, local_streams.size());

    // The island's geometry comes from the world, not the local span: it may cross processes. The
    // domain's host_domains are the contiguous run [domain_first_host, +host_domains_per_domain);
    // this process's slices occupy members [member_offset, member_offset+local_count).
    util::mesh_shape_t const& shape = topology.shape;
    util::u32_t const gpus_per_host = shape.gpus_per_host_domain;
    util::u32_t const member_count = shape.gpus_per_multimem_domain;
    util::u32_t const member_offset = shape.index_in_multimem_domain(topology.first_gpu_index());
    util::u32_t const host_domains_per_domain = member_count / gpus_per_host;
    util::u32_t const domain_first_host = topology.multimem_domain_index() * host_domains_per_domain;
    bool const is_lead = member_offset == 0;
    util::u32_t const collective_id = topology.next_collective_id();
    sz::string_t slice_key, multicast_key, barrier_key;
    if (!util::format_into(slice_key, "multimem.{}.slice", collective_id) ||
        !util::format_into(multicast_key, "multimem.{}.mc", collective_id) ||
        !util::format_into(barrier_key, "multimem.{}", collective_id))
        return result_t::fail(util::status_code_t::out_of_memory_k, "Rendezvous key formatting failed");

    // Host-side staging (creates, binds, memsets) needs a current context; the lead local stream
    // owns one — every VMM call names its device explicitly, so one context serves all. No stream
    // is retained: the domain outlives its bring-up streams by design.
    [[maybe_unused]] util::status_t const gpu_scope = (local_streams.stream(0)).bind();
    if (gpu_scope.failed())
        return result_t::fail(util::status_code_t::device_k, "context entry failed (device={})", local_devices[0]);

    util::multimem_domain_t domain;
    domain.payload_scalars_ = payload_scalars;
    domain.member_offset_ = member_offset;
    domain.wait_nanoseconds_ = static_cast<util::u64_t>(timeout_seconds * 1e9);
    domain.faults_ = topology.faults();
    // Replica layout: [payload f32×N | flag u32 | round u32], regions cache-line aligned. Each member's
    // round scalar lives in its own replica, self-advanced by its barrier kernels.
    std::size_t const replica_bytes = domain.flag_offset_() + 2 * sizeof(util::u32_t);

    if (util::expected_t reached = util::peers_reach_each_other(local_devices); !reached)
        return result_t::fail(std::move(reached));
    auto [tier, make_tier_status] = util::multimem_tier_choice_t::make(member_count, local_devices);
    if (!make_tier_status) return result_t::fail(std::move(make_tier_status));
    domain.tier_ = std::move(tier);
    bool const multimem_everywhere = domain.tier() == util::multimem_tier_t::switch_reduced_k;

    // The multicast binding granularity dictates the replica stride; the symmetric arena also
    // honors every local device's physical granularity.
    std::size_t multimem_granularity = 1;
    CUmulticastObjectProp multimem_properties {};
    if (multimem_everywhere) {
        auto [handle_type, pick_status] = util::gpu_symmetric_arena_t::shareable_handle_type(local_devices,
                                                                                             member_count);
        if (!pick_status) return result_t::fail(std::move(pick_status));
        multimem_properties.numDevices = member_count;
        multimem_properties.handleTypes = handle_type;
        multimem_properties.size = replica_bytes;
        if (CUresult result = cuMulticastGetGranularity(&multimem_granularity, &multimem_properties,
                                                        CU_MULTICAST_GRANULARITY_RECOMMENDED);
            result != CUDA_SUCCESS)
            return util::cuda_failure<util::multimem_domain_t>(result, "multicast granularity query failed");
    }

    // 1. Reserve the whole domain VA and create this process's local slices.
    auto [replicas, make_replicas_status] = util::gpu_symmetric_arena_t::make(
        local_devices, member_count, member_offset, replica_bytes, multimem_granularity);
    if (!make_replicas_status) return result_t::fail(std::move(make_replicas_status));
    domain.replicas_ = std::move(replicas);

    // 2. Exchange slice fabric handles across the domain's processes and adopt every peer's slices, so the
    //    whole symmetric range is mapped in every process. A single-process island skips it.
    if (host_domains_per_domain > 1) {
        sz::string_t slices_text;
        for (std::size_t local = 0; local != local_count; ++local) {
            auto [fabric_handle, export_status] = domain.replicas_.export_fabric_handle(
                member_offset + static_cast<util::u32_t>(local));
            if (!export_status) return result_t::fail(std::move(export_status));
            if ((local != 0 && !util::format_into(slices_text, " ")) ||
                !format_fabric_handle_(slices_text, fabric_handle))
                return result_t::fail(util::status_code_t::out_of_memory_k, "Slice handle payload assembly failed");
        }
        if (util::expected_t published = topology.publish_text(slice_key.view(), slices_text.view()); !published)
            return result_t::fail(std::move(published));
        for (util::u32_t peer_host = domain_first_host; peer_host != domain_first_host + host_domains_per_domain;
             ++peer_host) {
            if (peer_host == topology.host_domain_index) continue;
            sz::string_t peer_text;
            if (util::expected_t awaited = topology.await_text(peer_host, slice_key.view(), peer_text, timeout_seconds);
                !awaited)
                return result_t::fail(std::move(awaited));
            util::u32_t const peer_member_base = (peer_host - domain_first_host) * gpus_per_host;
            if (util::expected_t adopted = adopt_peer_slices_(domain.replicas_, peer_text.view(), peer_host,
                                                              peer_member_base, gpus_per_host);
                !adopted)
                return result_t::fail(std::move(adopted));
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
                return util::cuda_failure<util::multimem_domain_t>(
                    result, "multicast object creation failed (members={} bytes={})", member_count, bound_bytes);
            util::expected_t published = host_domains_per_domain > 1
                                             ? publish_multicast_handle_(domain.multimem_handle_, topology,
                                                                         multicast_key.view())
                                             : util::expected_t {};
            if (!published) return result_t::fail(std::move(published));
        }
        else {
            sz::string_t mc_text;
            if (util::expected_t awaited = topology.await_text(domain_first_host, multicast_key.view(), mc_text,
                                                               timeout_seconds);
                !awaited)
                return result_t::fail(std::move(awaited));
            CUmemFabricHandle fabric {};
            bool parsed = false;
            for (sz::string_view_t token : mc_text.view().split(sz::whitespaces_set())) {
                if (token.empty()) continue;
                parsed = parse_fabric_handle_(token, fabric);
                break;
            }
            if (!parsed)
                return result_t::fail(util::status_code_t::io_k, "malformed multicast handle from host_domain {}",
                                      domain_first_host);
            if (CUresult result = cuMemImportFromShareableHandle(&domain.multimem_handle_, &fabric,
                                                                 CU_MEM_HANDLE_TYPE_FABRIC);
                result != CUDA_SUCCESS)
                return util::cuda_failure<util::multimem_domain_t>(result, "multicast object import failed");
        }
        // Add this process's local devices and bind its local slices — no peer dependency,
        // so no barrier.
        for (util::i32_t const device : local_devices) {
            CUdevice device_handle = 0;
            if (CUresult result = cuDeviceGet(&device_handle, device); result != CUDA_SUCCESS)
                return util::cuda_failure<util::multimem_domain_t>(result, "device handle failed ({})", device);
            if (CUresult result = cuMulticastAddDevice(domain.multimem_handle_, device_handle); result != CUDA_SUCCESS)
                return util::cuda_failure<util::multimem_domain_t>(result, "multicast add-device failed (device={})",
                                                                   device);
        }
        for (std::size_t local = 0; local != local_count; ++local) {
            util::u32_t const member = member_offset + static_cast<util::u32_t>(local);
            if (CUresult result = cuMulticastBindMem(domain.multimem_handle_, 0, domain.replicas_.member_handle(member),
                                                     0, bound_bytes, 0);
                result != CUDA_SUCCESS)
                return util::cuda_failure<util::multimem_domain_t>(result, "multicast bind failed (member={})", member);
        }
        if (CUresult result = cuMemAddressReserve(&domain.multimem_pointer_, bound_bytes, 0, 0, 0);
            result != CUDA_SUCCESS)
            return util::cuda_failure<util::multimem_domain_t>(result, "multicast reservation failed");
        if (CUresult result = cuMemMap(domain.multimem_pointer_, bound_bytes, 0, domain.multimem_handle_, 0);
            result != CUDA_SUCCESS)
            return util::cuda_failure<util::multimem_domain_t>(result, "multicast mapping failed");
        if (util::expected_t granted = grant_local_access_(domain.multimem_pointer_, bound_bytes, local_devices,
                                                           member_offset);
            !granted)
            return result_t::fail(std::move(granted));
    }

    // 4. Zero this process's local slices' arrival flag and round scalar — each process owns its
    //    physical replicas, and a zero round makes the first barrier's target exactly one round of
    //    arrivals; the barrier kernels self-advance from there.
    for (std::size_t local = 0; local != local_count; ++local) {
        util::u32_t const member = member_offset + static_cast<util::u32_t>(local);
        CUdeviceptr const counters = domain.replicas_.base() + member * domain.replicas_.member_stride() +
                                     domain.flag_offset_();
        if (CUresult result = cuMemsetD8(counters, 0, 2 * sizeof(util::u32_t)); result != CUDA_SUCCESS)
            return util::cuda_failure<util::multimem_domain_t>(result, "arrival counter reset failed (member={})",
                                                               member);
        // The reset ran on the legacy stream, which a peer's barrier kernel never waits on.
        if (CUresult result = cuStreamSynchronize(CU_STREAM_LEGACY); result != CUDA_SUCCESS)
            return util::cuda_failure<util::multimem_domain_t>(result, "arrival counter reset drain failed (member={})",
                                                               member);
    }

    // 5. Every process has finished binding before the first collective runs.
    if (util::expected_t synchronized = topology.barrier(barrier_key.view(), timeout_seconds); !synchronized)
        return result_t::fail(std::move(synchronized));
    return result_t {std::move(domain)};
}

util::expected_t util::multimem_domain_t::barrier(std::size_t member_index, util::executor_stream_t& stream) noexcept {
    dim3 const one {1u, 1u, 1u};
    dim3 const warp {util::warp_size_k, 1u, 1u};
    switch (tier()) {
    case util::multimem_tier_t::switch_reduced_k:
        return util::launch_function(switch_kernels(member_index).barrier, one, warp, stream, member(member_index));
    case util::multimem_tier_t::peer_read_k:
        return util::launch_kernel(&multimem_barrier_peer_kernel_, one, warp, stream, member(member_index));
    }
    return util::expected_t::fail(util::status_code_t::invalid_argument_k, "unknown multimem tier {}",
                                  static_cast<util::u32_t>(tier()));
}

util::expected_t util::multimem_domain_t::reduce_scatter(std::size_t member_index, util::f32r1_span_t destination,
                                                         std::size_t shard_begin,
                                                         util::executor_stream_t& stream) noexcept {
    std::size_t const shard_scalars = destination.extent(0);
    if (shard_begin + shard_scalars > payload_scalars_)
        return util::expected_t::fail(util::status_code_t::unsupported_k,
                                      "shard [{},{}) exceeds the domain payload of {} scalars", shard_begin,
                                      shard_begin + shard_scalars, payload_scalars_);
    dim3 const block_shape {util::kernel_block_size_default_k, 1u, 1u};
    dim3 const grid_shape {util::grid_size_for(shard_scalars), 1u, 1u};
    switch (tier()) {
    case util::multimem_tier_t::switch_reduced_k:
        return util::launch_function(switch_kernels(member_index).reduce_scatter, grid_shape, block_shape, stream,
                                     member(member_index), destination, shard_begin);
    case util::multimem_tier_t::peer_read_k:
        return util::launch_kernel(&multimem_reduce_scatter_peer_kernel_, grid_shape, block_shape, stream,
                                   member(member_index), destination, shard_begin);
    }
    return util::expected_t::fail(util::status_code_t::invalid_argument_k, "unknown multimem tier {}",
                                  static_cast<util::u32_t>(tier()));
}
