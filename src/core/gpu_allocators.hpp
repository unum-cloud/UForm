/**
 *  @brief Single-GPU memory allocators of the UForm v4 primitives cell: a growing bump arena + a
 *      stream-ordered allocator, each owning memory of exactly @b one device.
 *  @file src/core/gpu_allocators.hpp
 *  @author Ash Vardanian
 *  @date July 2, 2026
 *
 *  Two allocators for the two @b single-GPU memory disciplines the stack needs, both on the CUDA Driver
 *  API (the multi-device symmetric memory of a multimem_domain lives with the multimem collectives,
 *  beside its one consumer):
 *
 *    - @ref gpu_arena_t — a virtual-memory (VMM) bump arena for @b homogeneous fixed-shape workloads, where
 *      every call allocates the same sequence of sizes — model training and inference steps. It owns the
 *      weights, activations, and step scratch: reserves a large VA range once, maps physical chunks in on
 *      demand, and hands out aligned allocations as pure pointer arithmetic — no deallocation, only cursor
 *      resets, so an allocation sequence is deterministic across captured-graph replays and legal inside
 *      capture. Growth ( @c cuMemMap ) is a host-immediate init-time operation, never a step-path one.
 *    - @ref gpu_alloc — a stream-ordered allocator ( @c cuMemAllocAsync / @c cuMemFreeAsync ) for exactly
 *      the opposite: @b heterogeneous shape-varying workloads, where every call differs — the data loaders'
 *      variable-length blobs and decode scratch, living outside captured graphs. Satisfies
 *      @c std::alloc_traits , so it plugs into NumKong's owning
 *      @c nk::tensor<value_type,gpu_alloc<value_type>,rank> directly.
 *    - @ref gpu_arena_alloc — the same @c std::alloc_traits facade over a borrowed arena, so model code
 *      instantiates owning tensors whose storage is arena allocations (deallocation is a no-op; reclamation
 *      is the arena's cursor rewind).
 *
 *  There is no in-house memory ledger: the driver already keeps the truth — @c cuMemGetInfo for a
 *  device's totals, the default pool's attributes for the stream-ordered tier's usage and high-water
 *  marks — and @ref gpu_memory_report reads it on demand. Per-owner attribution is each owner's own
 *  accessors ( @ref gpu_arena_t::used / @ref gpu_arena_t::mapped ).
 *
 *  The arena's physical chunks are created @c gpuDirectRDMACapable wherever the device supports it, so the same
 *  chunks can later be registered as InfiniBand MRs or bound to a multicast object — symmetric / multimem backing is a property
 *  added to @b this arena at creation, not a third allocator: running the identical allocation program on every
 *  rank yields identical offsets, which is what makes a peer's buffer addressable as @c remote_base+offset .
 */
#pragma once

#include <cstddef> // `std::size_t`, `std::byte`

#include <utility>     // `std::exchange`, `std::forward`
#include <type_traits> // `std::true_type`
#include <span>        // `std::span` — the `visible_to` device-ID list

#include <cuda.h>      // `CUdeviceptr`, `cuMemAddressReserve`, `cuMemMap`, `CUresult`, `cuGetErrorString`
#include <cuda_bf16.h> // `__nv_bfloat16` behind the `bf16_t` alias
#include <cuda_fp8.h>  // `__nv_fp8_e4m3` behind the `e4m3_t` alias

#include "core/types.hpp" // `expected`, `status_t`
#include "core/array.hpp" // `array` for exception-free chunk bookkeeping

namespace unum::uform::core {

using e4m3_t = __nv_fp8_e4m3;
using bf16_t = __nv_bfloat16;

/** The CUDA-scalar half of the rank-tagged borrow vocabulary in @c types.hpp . */
using bf16r1_view_t = nk::tensor_view<bf16_t, 1>;
using bf16r1_span_t = nk::tensor_span<bf16_t, 1>;
using bf16r2_view_t = nk::tensor_view<bf16_t, 2>;
using bf16r2_span_t = nk::tensor_span<bf16_t, 2>;
using bf16r3_view_t = nk::tensor_view<bf16_t, 3>;
using bf16r3_span_t = nk::tensor_span<bf16_t, 3>;
using e4m3r1_view_t = nk::tensor_view<e4m3_t, 1>;
using e4m3r1_span_t = nk::tensor_span<e4m3_t, 1>;
using e4m3r2_view_t = nk::tensor_view<e4m3_t, 2>;
using e4m3r2_span_t = nk::tensor_span<e4m3_t, 2>;
using e4m3r3_view_t = nk::tensor_view<e4m3_t, 3>;
using e4m3r3_span_t = nk::tensor_span<e4m3_t, 3>;

/** Failure carrying a CUDA Driver error: @ref expected::fail tagged @c cuda_error_k , with the driver's
 *  own error string appended to the description and @p cu_error kept in @c cu_error_ . */
template <typename value_type_ = void, typename... arg_types_>
[[nodiscard]] expected<value_type_> cuda_failure(CUresult cu_error, format_at_t format, arg_types_ &&...args) noexcept {
    expected<value_type_> result = expected<value_type_>::fail(status_t::cuda_error_k, format,
                                                               std::forward<arg_types_>(args)...);
    result.cu_error_ = static_cast<i32_t>(cu_error);
    char const *cu_error_string = nullptr;
    if (cuGetErrorString(cu_error, &cu_error_string) != CUDA_SUCCESS || cu_error_string == nullptr)
        cu_error_string = "unknown";
    (void)result.description_.append(": ");
    (void)result.description_.append(cu_error_string);
    return result;
}

/** The Driver API address of the device @p pointer — the one place a pointer becomes a @c CUdeviceptr . */
[[nodiscard]] inline CUdeviceptr device_address(void const *pointer) noexcept {
    return reinterpret_cast<CUdeviceptr>(pointer);
}

/** The typed pointer at the Driver API @p address — the inverse of @ref device_address . */
template <typename value_type_>
[[nodiscard]] inline value_type_ *device_pointer(CUdeviceptr address) noexcept {
    return reinterpret_cast<value_type_ *>(address);
}

/** Byte alignment of every arena allocation; matches the 256-byte guarantee of the CUDA allocators, so
 *  an allocation base is always vector-load aligned. */
inline constexpr std::size_t arena_alignment_k = 256u;

/** Cache-line byte count — the region-isolation alignment of symmetric-heap layouts, so flags,
 *  landings, and payloads never false-share a line. */
inline constexpr std::size_t cache_line_bytes_k = 128u;

/** Round @p bytes up to the next multiple of @p granularity (any positive step). */
[[nodiscard]] inline constexpr std::size_t round_up_to_multiple(std::size_t bytes, std::size_t granularity) noexcept {
    return (bytes + granularity - 1u) / granularity * granularity;
}

/** One device's memory picture, straight from the driver — see @ref gpu_memory_report . */
struct gpu_memory_report_t {

    /** HBM not yet allocated by anyone in any process, per @c cuMemGetInfo . */
    std::size_t free_bytes = 0;

    /** The device's total HBM. */
    std::size_t total_bytes = 0;

    /** Bytes currently allocated from the device's default stream-ordered pool. */
    std::size_t pool_used_bytes = 0;

    /** The pool's high-water mark since its last reset. */
    std::size_t pool_peak_bytes = 0;
};

/** Poll the driver for @p device 's memory picture — the diagnostics counterpart of the owners'
 *  own accessors ( @c arena.used()/mapped() ); there is no in-house ledger to keep in sync. The
 *  calling thread must hold a context on @p device (the @c cuMemGetInfo scope). */
[[nodiscard]] inline expected<gpu_memory_report_t> gpu_memory_report(i32_t device) noexcept {
    gpu_memory_report_t report;
    if (CUresult result = cuMemGetInfo(&report.free_bytes, &report.total_bytes); result != CUDA_SUCCESS)
        return cuda_failure<gpu_memory_report_t>(result, "memory info query failed (device={})", device);
    CUdevice device_handle = 0;
    CUmemoryPool pool = nullptr;
    if (cuDeviceGet(&device_handle, device) == CUDA_SUCCESS &&
        cuDeviceGetDefaultMemPool(&pool, device_handle) == CUDA_SUCCESS) {
        cuuint64_t used = 0, peak = 0;
        if (cuMemPoolGetAttribute(pool, CU_MEMPOOL_ATTR_USED_MEM_CURRENT, &used) == CUDA_SUCCESS)
            report.pool_used_bytes = static_cast<std::size_t>(used);
        if (cuMemPoolGetAttribute(pool, CU_MEMPOOL_ATTR_USED_MEM_HIGH, &peak) == CUDA_SUCCESS)
            report.pool_peak_bytes = static_cast<std::size_t>(peak);
    }
    return expected<gpu_memory_report_t> {report};
}

/**
 *  @brief Growing bump arena over CUDA virtual memory — the owner of every long-lived device buffer: weights,
 *      activations, step scratch, and (later) the symmetric / multimem-bound heap.
 *
 *  Built for @b homogeneous fixed-shape workloads — model training and inference, where the batch shape is
 *  frozen at step construction and every step allocates the identical sequence of sizes. A workload whose
 *  shapes vary call to call belongs on @ref gpu_alloc instead; a bump cursor has no free list, so shape
 *  variance here only accretes.
 *
 *  A large VA range is reserved once at @ref make ; physical memory arrives lazily in granularity-rounded
 *  chunks as @ref allocate_bytes outgrows the mapped prefix, so an over-generous reservation costs address
 *  space, not HBM. Allocation is pointer arithmetic and never fails while the reservation lasts; freeing is
 *  only @ref rewind / @ref release (cursor rewinds — no free list, ever), which keeps every allocation
 *  sequence deterministic across captured-graph replays. Growth must happen @b outside graph capture; a
 *  steady-state step performs none.
 *
 *  Not stream-ordered by design: an allocation orders against nothing (the memory already exists), and the one
 *  hazard — rewinding a cursor while enqueued kernels still read the allocations above it — is owned by the step
 *  schedule, not the allocator. Move-only; the destructor unmaps and releases everything.
 *
 *  Requires a current CUDA context on the calling thread (retain the device's primary context at startup).
 */
struct gpu_arena_t {

    /** One physical backing chunk mapped into the reserved range; kept only for teardown. */
    struct physical_chunk_t {
        CUmemGenericAllocationHandle handle = 0;
        std::size_t bytes = 0;
    };

    /** A cursor position taken with @ref snapshot and rewound to with @ref rewind — a strong type
     *  so a snapshot can't be confused with any other byte count in flight. */
    struct snapshot_t {
        std::size_t bytes = 0;
    };

    /** Start of the reserved VA range. */
    CUdeviceptr base_ = 0;

    /** VA reservation — the growth ceiling. */
    std::size_t reserved_bytes_ = 0;

    /** Physically backed prefix of the reservation. */
    std::size_t mapped_bytes_ = 0;

    /** Bump cursor within the mapped prefix. */
    std::size_t used_bytes_ = 0;

    /** Physical allocation granularity of @ref device_ . */
    std::size_t granularity_ = 0;

    /** Device ordinal the physical chunks live on. */
    int device_ = 0;

    /** Mapped chunks, in mapping order, for teardown. */
    array<physical_chunk_t> chunks_ {};

    /** Peer devices named in @c visible_to at @ref make — baked configuration every chunk
     *  (present and future) is granted to; nothing appends to it after construction. */
    array<i32_t> visible_to_ {};

  private:
    gpu_arena_t() noexcept = default;

  public:
    gpu_arena_t(gpu_arena_t const &) = delete;
    gpu_arena_t &operator=(gpu_arena_t const &) = delete;

    gpu_arena_t(gpu_arena_t &&other) noexcept
        : base_(std::exchange(other.base_, 0)), reserved_bytes_(std::exchange(other.reserved_bytes_, 0)),
          mapped_bytes_(std::exchange(other.mapped_bytes_, 0)), used_bytes_(std::exchange(other.used_bytes_, 0)),
          granularity_(std::exchange(other.granularity_, 0)), device_(std::exchange(other.device_, 0)),
          chunks_(std::move(other.chunks_)), visible_to_(std::move(other.visible_to_)) {}

    gpu_arena_t &operator=(gpu_arena_t &&other) noexcept {
        if (this == &other) return *this;
        destroy_();
        base_ = std::exchange(other.base_, 0);
        reserved_bytes_ = std::exchange(other.reserved_bytes_, 0);
        mapped_bytes_ = std::exchange(other.mapped_bytes_, 0);
        used_bytes_ = std::exchange(other.used_bytes_, 0);
        granularity_ = std::exchange(other.granularity_, 0);
        device_ = std::exchange(other.device_, 0);
        chunks_ = std::move(other.chunks_);
        visible_to_ = std::move(other.visible_to_);
        return *this;
    }

    ~gpu_arena_t() noexcept { destroy_(); }

    /**
     *  @brief Reserve @p reserve_bytes of VA on @p device ; no physical memory is committed yet.
     *
     *  The chunks later mapped in are created @c gpuDirectRDMACapable where the device supports it, so the arena's
     *  allocations can be MR-registered or multicast-bound without re-allocation.
     *
     *  Cross-device visibility is @b construction-time configuration: every peer named in
     *  @p visible_to is granted read-write access to every chunk this arena ever maps (a VMM
     *  page-table grant — one-time cost, no address motion, capture-stable peer views). There is no
     *  post-hoc grant; consumers @b validate with @ref is_visible_to and fail loudly.
     *
     *  @param[in] device Device ordinal the arena's physical chunks will live on.
     *  @param[in] reserve_bytes Growth ceiling; rounded up to the device's allocation granularity.
     *  @param[in] visible_to Peer device ordinals that must be able to read this arena over NVLink:
     *      the other members of its domain, each refused at once when it has no P2P path to
     *      @p device .
     */
    [[nodiscard]] static expected<gpu_arena_t> make(int device, std::size_t reserve_bytes,
                                                    std::span<i32_t const> visible_to = {}) noexcept {
        gpu_arena_t arena;
        arena.device_ = device;
        for (i32_t const peer : visible_to) {
            if (peer == device) continue;
            int can_access = 0;
            if (CUresult result = cuDeviceCanAccessPeer(&can_access, peer, device); result != CUDA_SUCCESS)
                return cuda_failure<gpu_arena_t>(result, "P2P capability query failed (device={} peer={})", device,
                                                 peer);
            if (can_access == 0)
                return expected<gpu_arena_t>::fail(status_t::shape_unsupported_k,
                                                   "device {} cannot read the arena of device {}: no P2P path", peer,
                                                   device);
            if (arena.visible_to_.push_back(peer) != status_t::success_k)
                return expected<gpu_arena_t>::fail(status_t::bad_alloc_k, "peer table allocation failed");
        }
        CUmemAllocationProp properties = allocation_properties_(device);
        if (CUresult result = cuMemGetAllocationGranularity(&arena.granularity_, &properties,
                                                            CU_MEM_ALLOC_GRANULARITY_MINIMUM);
            result != CUDA_SUCCESS)
            return cuda_failure<gpu_arena_t>(result, "granularity query failed (device={})", device);
        arena.reserved_bytes_ = round_up_to_multiple(reserve_bytes, arena.granularity_);
        if (CUresult result = cuMemAddressReserve(&arena.base_, arena.reserved_bytes_, 0, 0, 0); result != CUDA_SUCCESS)
            return cuda_failure<gpu_arena_t>(result, "VA reservation failed (device={} bytes={})", device,
                                             arena.reserved_bytes_);
        return expected<gpu_arena_t> {std::move(arena)};
    }

    /** Bytes allocated so far. */
    [[nodiscard]] std::size_t used() const noexcept { return used_bytes_; }

    /** The device ordinal the arena's memory lives on. */
    [[nodiscard]] int device() const noexcept { return device_; }

    /** Physically committed bytes; grows monotonically, never shrinks before destruction. */
    [[nodiscard]] std::size_t mapped() const noexcept { return mapped_bytes_; }

    /** The current cursor position, to be rewound to with @ref rewind . */
    [[nodiscard]] snapshot_t snapshot() const noexcept { return {used_bytes_}; }

    /** Whether @p device can read this arena: the home device, or a peer named in the
     *  @c visible_to set at @ref make . Consumers needing cross-device reads validate with this
     *  and fail loudly — visibility is never granted after construction. */
    [[nodiscard]] bool is_visible_to(int device) const noexcept {
        if (device == device_) return true;
        for (i32_t const peer : visible_to_.view())
            if (peer == device) return true;
        return false;
    }

    /** Allocate @p bytes at @p alignment off the bump cursor, growing the physical backing when the
     *  mapped prefix is exhausted. Returns @c nullptr when the reservation ceiling is hit or growth
     *  fails — on a bump arena a null can mean nothing else. */
    [[nodiscard]] std::byte *allocate_bytes(std::size_t bytes, std::size_t alignment = arena_alignment_k) noexcept {
        return allocate_object<std::byte>(bytes, alignment);
    }

    /** Allocate @p count elements of @c value_type_ off the bump cursor — the typed sibling of
     *  @ref allocate_bytes , mirroring `std::pmr::polymorphic_allocator::allocate_object` . */
    template <typename value_type_>
    [[nodiscard]] value_type_ *allocate_object(std::size_t count = 1,
                                               std::size_t alignment = arena_alignment_k) noexcept {
        std::size_t const bytes = count * sizeof(value_type_);
        std::size_t const aligned_cursor = round_up_to_multiple(used_bytes_, alignment);
        if (aligned_cursor + bytes > reserved_bytes_) return nullptr;
        if (aligned_cursor + bytes > mapped_bytes_)
            if (grow_to_(aligned_cursor + bytes) != CUDA_SUCCESS) return nullptr;
        used_bytes_ = aligned_cursor + bytes;
        return device_pointer<value_type_>(base_ + aligned_cursor);
    }

    /**
     *  @brief Allocate @p bytes guaranteed to lie within @b one physical chunk, so the range can be
     *      MR-registered: @c nvidia_peermem rejects ( @c EFAULT ) any @c ibv_reg_mr range spanning two
     *      @c cuMemCreate handles, and a bump allocation crossing the growth boundary does exactly that.
     *
     *  Places the range at the virgin mapping frontier — a fresh chunk start — and grows the whole
     *  extent as a single @c cuMemCreate , skipping the mapped-but-unused tail (under one granularity in
     *  the wire-at-startup pattern, since growth maps exactly the shortfall). The returned pointer is
     *  granularity-aligned. @c nullptr on ceiling or growth failure, like @ref allocate_bytes .
     */
    [[nodiscard]] std::byte *allocate_registrable_bytes(std::size_t bytes) noexcept {
        return allocate_registrable_object<std::byte>(bytes);
    }

    /** Allocate @p count elements of @c value_type_ inside one physical chunk — the typed sibling of
     *  @ref allocate_registrable_bytes . */
    template <typename value_type_>
    [[nodiscard]] value_type_ *allocate_registrable_object(std::size_t count) noexcept {
        std::size_t const bytes = count * sizeof(value_type_);
        std::size_t const chunk_start = mapped_bytes_;
        if (chunk_start + bytes > reserved_bytes_) return nullptr;
        if (grow_to_(chunk_start + bytes) != CUDA_SUCCESS) return nullptr;
        used_bytes_ = chunk_start + bytes;
        return device_pointer<value_type_>(base_ + chunk_start);
    }

    /** Rewind the cursor to a @ref snapshot ; the physical backing stays mapped. The caller owns the
     *  hazard of rewinding under allocations still referenced by enqueued work. */
    void rewind(snapshot_t snapshot) noexcept { used_bytes_ = snapshot.bytes; }

    /** Rewind the cursor to zero, reclaiming every allocation at once — the
     *  `std::pmr::monotonic_buffer_resource::release` semantic, strengthened: the physical backing stays
     *  mapped for reuse. */
    void release() noexcept { used_bytes_ = 0; }

  private:

    /** The one chunk-creation property set: device-resident, pinned, and RDMA-capable wherever the device can
     *  register VMM memory with an adapter - consumer cards refuse that flag outright, and have no adapter. */
    [[nodiscard]] static CUmemAllocationProp allocation_properties_(int device) noexcept {
        int rdma_supported = 0;
        cuDeviceGetAttribute(&rdma_supported, CU_DEVICE_ATTRIBUTE_GPU_DIRECT_RDMA_WITH_CUDA_VMM_SUPPORTED, device);
        CUmemAllocationProp properties {};
        properties.type = CU_MEM_ALLOCATION_TYPE_PINNED;
        properties.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
        properties.location.id = device;
        properties.allocFlags.gpuDirectRDMACapable = rdma_supported != 0;
        return properties;
    }

    /** Map new granularity-rounded physical chunks until @p target_bytes of the reservation is backed. */
    [[nodiscard]] CUresult grow_to_(std::size_t target_bytes) noexcept {
        CUmemAllocationProp properties = allocation_properties_(device_);
        CUmemAccessDesc access {};
        access.location = properties.location;
        access.flags = CU_MEM_ACCESS_FLAGS_PROT_READWRITE;
        while (mapped_bytes_ < target_bytes) {
            std::size_t const chunk_bytes = round_up_to_multiple(target_bytes - mapped_bytes_, granularity_);
            CUmemGenericAllocationHandle handle = 0;
            if (CUresult result = cuMemCreate(&handle, chunk_bytes, &properties, 0); result != CUDA_SUCCESS)
                return result;
            if (CUresult result = cuMemMap(base_ + mapped_bytes_, chunk_bytes, 0, handle, 0); result != CUDA_SUCCESS) {
                cuMemRelease(handle);
                return result;
            }
            if (CUresult result = cuMemSetAccess(base_ + mapped_bytes_, chunk_bytes, &access, 1);
                result != CUDA_SUCCESS) {
                cuMemUnmap(base_ + mapped_bytes_, chunk_bytes);
                cuMemRelease(handle);
                return result;
            }
            // Every chunk honors the `visible_to_` set baked at make.
            for (i32_t const peer : visible_to_.view()) {
                CUmemAccessDesc peer_access {};
                peer_access.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
                peer_access.location.id = peer;
                peer_access.flags = CU_MEM_ACCESS_FLAGS_PROT_READWRITE;
                if (CUresult result = cuMemSetAccess(base_ + mapped_bytes_, chunk_bytes, &peer_access, 1);
                    result != CUDA_SUCCESS) {
                    cuMemUnmap(base_ + mapped_bytes_, chunk_bytes);
                    cuMemRelease(handle);
                    return result;
                }
            }
            if (chunks_.push_back({handle, chunk_bytes}) != status_t::success_k) {
                cuMemUnmap(base_ + mapped_bytes_, chunk_bytes);
                cuMemRelease(handle);
                return CUDA_ERROR_OUT_OF_MEMORY; // Host-side chunk-table growth failed.
            }
            mapped_bytes_ += chunk_bytes;
        }
        return CUDA_SUCCESS;
    }

    /** Unmap the whole mapped prefix, release every physical handle, and free the VA reservation. */
    void destroy_() noexcept {
        if (base_ == 0) return;
        if (mapped_bytes_ != 0) cuMemUnmap(base_, mapped_bytes_);
        for (physical_chunk_t const &chunk : chunks_) cuMemRelease(chunk.handle);
        cuMemAddressFree(base_, reserved_bytes_);
        base_ = 0;
        chunks_.reset();
        reserved_bytes_ = mapped_bytes_ = used_bytes_ = 0;
    }
};

/**
 *  @brief Standard-allocator facade over a borrowed @ref gpu_arena_t , so owning containers size
 *      arena allocations without the bespoke allocation spelling.
 *
 *  @c nk::tensor<f32_t,gpu_arena_alloc<f32_t>,2>::uninitialized({rows,cols},gpu_arena_alloc<f32_t>{arena})
 *  allocates — and grows the arena — automatically; @ref array composes the same way.
 *
 *  Deallocation is deliberately a no-op: a bump arena reclaims through @ref gpu_arena_t::rewind /
 *  @ref gpu_arena_t::release , never per buffer — so a container backed by this allocator must not outlive
 *  its arena, nor a rewind past its allocation. The handle is trivially copyable, rebinds freely, and never
 *  owns the arena.
 */
template <typename value_type_ = std::byte>
struct gpu_arena_alloc {
    using value_type = value_type_;
    using size_type = std::size_t;
    using difference_type = std::ptrdiff_t;

    /** The handle is just an arena pointer — move it with the container so moves stay @c noexcept . */
    using propagate_on_container_move_assignment = std::true_type;

    template <typename other_type_>
    struct rebind {
        using other = gpu_arena_alloc<other_type_>;
    };

    /** The borrowed arena every allocation comes from. */
    gpu_arena_t *arena_ = nullptr;

    constexpr gpu_arena_alloc() noexcept = default;
    explicit constexpr gpu_arena_alloc(gpu_arena_t &arena) noexcept : arena_(&arena) {}
    template <typename other_type_>
    constexpr gpu_arena_alloc(gpu_arena_alloc<other_type_> const &other) noexcept : arena_(other.arena_) {}

    /** Allocate @p count elements off the arena, growing its physical backing if needed; @c nullptr
     *  on exhaustion or when unbound. */
    [[nodiscard]] value_type *allocate(std::size_t count) noexcept {
        return arena_ != nullptr ? arena_->allocate_object<value_type>(count) : nullptr;
    }

    /** A no-op: bump-arena memory is reclaimed by @ref gpu_arena_t::rewind , never per buffer. */
    void deallocate(value_type *, std::size_t) noexcept {}

    [[nodiscard]] constexpr bool operator==(gpu_arena_alloc const &other) const noexcept {
        return arena_ == other.arena_;
    }
};

/** The canonical byte-typed arena-allocator handle — what APIs receive and rebind. */
using gpu_arena_alloc_t = gpu_arena_alloc<b8_t>;

/**
 *  @brief The RDMA-registrable sibling of @ref gpu_arena_alloc : every allocation lies within one
 *      physical chunk of the arena — the precondition @c nvidia_peermem places on @c ibv_reg_mr — via
 *      @ref gpu_arena_t::allocate_registrable_bytes .
 *
 *  For the symmetric heaps the InfiniBand collectives register with their NICs; everything else stays
 *  on the ordinary arena allocator.
 *
 *  Same contract as @ref gpu_arena_alloc otherwise: borrowed arena, no-op deallocate, free rebind.
 */
template <typename value_type_ = std::byte>
struct gpu_registrable_alloc {
    using value_type = value_type_;
    using size_type = std::size_t;
    using difference_type = std::ptrdiff_t;

    /** The handle is just an arena pointer — move it with the container so moves stay @c noexcept . */
    using propagate_on_container_move_assignment = std::true_type;

    template <typename other_type_>
    struct rebind {
        using other = gpu_registrable_alloc<other_type_>;
    };

    /** The borrowed arena every allocation comes from. */
    gpu_arena_t *arena_ = nullptr;

    constexpr gpu_registrable_alloc() noexcept = default;
    explicit constexpr gpu_registrable_alloc(gpu_arena_t &arena) noexcept : arena_(&arena) {}
    template <typename other_type_>
    constexpr gpu_registrable_alloc(gpu_registrable_alloc<other_type_> const &other) noexcept : arena_(other.arena_) {}

    /** Allocate @p count elements inside one fresh physical chunk; @c nullptr on exhaustion or
     *  when unbound. */
    [[nodiscard]] value_type *allocate(std::size_t count) noexcept {
        return arena_ != nullptr ? arena_->allocate_registrable_object<value_type>(count) : nullptr;
    }

    /** A no-op: bump-arena memory is reclaimed by @ref gpu_arena_t::rewind , never per buffer. */
    void deallocate(value_type *, std::size_t) noexcept {}

    [[nodiscard]] constexpr bool operator==(gpu_registrable_alloc const &other) const noexcept {
        return arena_ == other.arena_;
    }
};

/** The canonical byte-typed registrable-allocator handle — what the collective heaps receive. */
using gpu_registrable_alloc_t = gpu_registrable_alloc<b8_t>;

/**
 *  @brief Pinned host-memory allocator — page-locked pages the DMA engines read directly, for the two
 *      host-side surfaces of a replayed step: generation staging cells that captured copy nodes re-read
 *      at every launch, and loss landings polled without a stream synchronize.
 *
 *  Stateless; needs a current context at the call, like every driver allocation here. Satisfies
 *  @c std::alloc_traits , so NumKong's owning tensor composes directly:
 *
 *      nk::tensor<f32_t, pinned_alloc<f32_t>, 1> landings;
 */
template <typename value_type_>
struct pinned_alloc {
    using value_type = value_type_;
    using size_type = std::size_t;
    using difference_type = std::ptrdiff_t;
    using propagate_on_container_move_assignment = std::true_type;

    template <typename other_type_>
    struct rebind {
        using other = pinned_alloc<other_type_>;
    };

    constexpr pinned_alloc() noexcept = default;
    template <typename other_type_>
    constexpr pinned_alloc(pinned_alloc<other_type_> const &) noexcept {}

    /** Pin @p count elements of portable host memory; @c nullptr on failure. */
    [[nodiscard]] value_type *allocate(std::size_t count) noexcept {
        void *base = nullptr;
        if (cuMemHostAlloc(&base, count * sizeof(value_type), CU_MEMHOSTALLOC_PORTABLE) != CUDA_SUCCESS) return nullptr;
        return static_cast<value_type *>(base);
    }

    void deallocate(value_type *base, std::size_t) noexcept {
        if (base != nullptr) cuMemFreeHost(base);
    }

    [[nodiscard]] constexpr bool operator==(pinned_alloc const &) const noexcept { return true; }
};

/** The canonical byte-typed pinned-allocator handle — what APIs receive and rebind. */
using pinned_alloc_t = pinned_alloc<b8_t>;

/**
 *  @brief Stream-ordered device allocator for @b heterogeneous shape-varying workloads — the exact opposite
 *      of @ref gpu_arena_t : every call may bring a different size, so buffers are individually allocated
 *      and freed.
 *
 *  The data loaders' tier — variable-length blobs and decode scratch that live @b outside captured
 *  graphs, where @c cuMemAllocAsync is legal.
 *
 *  Allocation and deallocation enqueue on the bound stream, so a loader's alloc → decode → consume → free
 *  chain needs no host fences; the memory is valid only in that stream's order (host access requires a
 *  sync). Satisfies @c std::alloc_traits — a stateful, trivially-copyable handle over the device's
 *  default memory pool — so NumKong's owning tensor composes directly:
 *  @c nk::tensor<f32_t,gpu_alloc<f32_t>,2>::uninitialized({rows,cols},alloc) . Only @c uninitialized applies:
 *  the other factories ( @c zeros / @c full ) initialize through host writes, which device memory
 *  forbids — fill with a kernel or @c cuMemsetD*Async instead.
 */
template <typename value_type_ = std::byte>
struct gpu_alloc {
    using value_type = value_type_;
    using size_type = std::size_t;
    using difference_type = std::ptrdiff_t;

    /** The handle is just a stream — move it with the container so moves stay @c noexcept . */
    using propagate_on_container_move_assignment = std::true_type;

    template <typename other_type_>
    struct rebind {
        using other = gpu_alloc<other_type_>;
    };

    /** Stream every allocation and deallocation is ordered on; the default stream when unset. */
    CUstream stream = nullptr;

    constexpr gpu_alloc() noexcept = default;
    explicit constexpr gpu_alloc(CUstream stream) noexcept : stream(stream) {}
    template <typename other_type_>
    constexpr gpu_alloc(gpu_alloc<other_type_> const &other) noexcept : stream(other.stream) {}

    /** Stream-ordered allocation from the device's default memory pool; @c nullptr on failure —
     *  matching NumKong's non-throwing allocator contract. The pool's own attributes carry the
     *  usage and high-water marks (see @ref gpu_memory_report ). */
    [[nodiscard]] value_type *allocate(std::size_t count) noexcept {
        CUdeviceptr pointer = 0;
        if (cuMemAllocAsync(&pointer, count * sizeof(value_type), stream) != CUDA_SUCCESS) return nullptr;
        return device_pointer<value_type>(pointer);
    }

    /** Stream-ordered release back to the pool; safe after every enqueued consumer on @ref stream . */
    void deallocate(value_type *pointer, std::size_t) noexcept {
        if (pointer != nullptr) cuMemFreeAsync(device_address(pointer), stream);
    }

    [[nodiscard]] constexpr bool operator==(gpu_alloc const &other) const noexcept { return stream == other.stream; }
};

/** The canonical byte-typed stream-ordered-allocator handle — what APIs receive and rebind. */
using gpu_alloc_t = gpu_alloc<b8_t>;

/** The rank-tagged owning-tensor vocabulary — @c *_tensor_arena_t owns device memory off a
 *  @ref gpu_arena_t , @c *_tensor_pinned_t owns page-locked host memory; the borrowing @c *_view_t /
 *  @c *_span_t counterparts live in @c types.hpp , the BFloat16 / E4M3 ones above. */
using b8r1_tensor_arena_t = nk::tensor<b8_t, gpu_arena_alloc<b8_t>, 1>;
using b8r1_tensor_registrable_t = nk::tensor<b8_t, gpu_registrable_alloc<b8_t>, 1>;
using i16r2_tensor_arena_t = nk::tensor<i16_t, gpu_arena_alloc<i16_t>, 2>;
using i32r1_tensor_arena_t = nk::tensor<i32_t, gpu_arena_alloc<i32_t>, 1>;
using u32r1_tensor_arena_t = nk::tensor<u32_t, gpu_arena_alloc<u32_t>, 1>;
using u32r2_tensor_arena_t = nk::tensor<u32_t, gpu_arena_alloc<u32_t>, 2>;
using f32r1_tensor_arena_t = nk::tensor<f32_t, gpu_arena_alloc<f32_t>, 1>;
using f32r2_tensor_arena_t = nk::tensor<f32_t, gpu_arena_alloc<f32_t>, 2>;
using f32r3_tensor_arena_t = nk::tensor<f32_t, gpu_arena_alloc<f32_t>, 3>;
using bf16r1_tensor_arena_t = nk::tensor<bf16_t, gpu_arena_alloc<bf16_t>, 1>;
using bf16r2_tensor_arena_t = nk::tensor<bf16_t, gpu_arena_alloc<bf16_t>, 2>;
using bf16r3_tensor_arena_t = nk::tensor<bf16_t, gpu_arena_alloc<bf16_t>, 3>;
using e4m3r1_tensor_arena_t = nk::tensor<e4m3_t, gpu_arena_alloc<e4m3_t>, 1>;
using e4m3r2_tensor_arena_t = nk::tensor<e4m3_t, gpu_arena_alloc<e4m3_t>, 2>;

using u32r1_tensor_pinned_t = nk::tensor<u32_t, pinned_alloc<u32_t>, 1>;
using u32r2_tensor_pinned_t = nk::tensor<u32_t, pinned_alloc<u32_t>, 2>;
using f32r1_tensor_pinned_t = nk::tensor<f32_t, pinned_alloc<f32_t>, 1>;

} // namespace unum::uform::core
