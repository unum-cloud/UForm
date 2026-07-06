/**
 *  @brief Execution scheduling: owning CUDA streams and capture-once / replay-forever graphs.
 *  @file cuda/primitives/schedule.cuh
 *  @author Ash Vardanian
 *  @date July 2, 2026
 *
 *  Two artifacts every step loop keys on:
 *
 *      1. @ref gpu_stream_t — a move-only owning wrapper over a real (non-null) @c CUstream , holding the
 *         device's primary context retained, so ordering is explicit and capture is legal (stream capture
 *         rejects the legacy null stream).
 *      2. @ref gpu_replay_graph_t — the capture-once / replay-forever contract: @ref make captures a
 *         body of kernel launches into a @c CUgraph , instantiates it, uploads the executable, and keeps
 *         only that; every later step is one @c cuGraphLaunch with @b zero host-side re-derivation.
 *
 *  @section schedule_contract The capture contract
 *
 *  A captured body must be a pure enqueue program: no host synchronization, no device-to-host reads, and
 *  @b no allocation. Per-step host-varying values travel through device memory the body reads at replay
 *  (learnable scalars, staged relation columns) — never through re-captured code. The
 *  contract is enforced, not trusted: after capture the graph's nodes are walked, and any
 *  allocation / free node is a @b loud failure — an allocating body escapes every arena budget and
 *  deadlocks device-local multi-GPU worlds at first launch, and there is no correct fallback.
 *
 *  Construction absorbs the whole setup pipeline — capture, instantiate, guard, @b upload — so a
 *  @ref gpu_replay_graph_t that exists is always launchable, with device staging already done. Uploading
 *  eagerly matters: lazily staged MMIO nodes are a measured first-launch deadlock on device-initiated
 *  transports.
 *
 *  @section schedule_regions The captured region is model math only
 *
 *  A captured graph holds @b model math only — entry / embedding, the block stack, the final RMSNorm, the
 *  @b local loss, and backward. The distributed work — cross-GPU loss exchange, the gradient reduce, the
 *  @ref barrier_round_t barriers, and the optimizer's world step — is @b never fused into it; it is
 *  its own region the parent scope sequences after the model region. UForm is a distributed-loss training
 *  stack, so a monolithic all-in-one graph welds the pure math to world state, and the round, the
 *  reduce, and the loss exchange can no longer be reasoned about — or replayed — independently. Forward and
 *  backward stay together (one contiguous, shape-stable computation); the only boundary is model-vs-collective.
 *
 *  @section schedule_fork Multi-GPU capture — fork / join
 *
 *  Launching one graph per GPU serializes at the driver when the bodies touch shared resources, and
 *  serialized launches deadlock the moment rank 0 parks at a cross-rank barrier its peers never reach.
 *  The multi-stream @ref make therefore records @b one graph across all ranks: the trunk stream
 *  records a fork event, every branch stream waits only on the fork (never on a sibling, so branch
 *  kernels stay mutually independent and co-resident at replay), and the trunk joins on every branch
 *  before ending the capture. One @c cuGraphLaunch then makes every rank's work resident together.
 */
#pragma once

#include <cstddef> // `std::size_t`
#include <cstring> // `std::memset` — zeroing pinned scalar cells

#include <utility> // `std::exchange`, `std::forward`
#include <array>   // `std::array` — the deterministic-algorithm candidate table
#include <span>    // `std::span` — the multi-stream capture rank list

#include <cuda.h>      // `CUstream`, `CUgraph`, `CUgraphExec`, `CUevent`
#include <cuda_bf16.h> // `__nv_bfloat16` — the training-currency scalar of the rank-1 aliases
#include <cublasLt.h>  // `cublasLtHandle_t`, `cublasLtMatmul` — the tensor-core matmul sessions
#include <cudnn.h>     // `cudnnHandle_t` — the fused-attention session (graphs live with their consumers)

#include <numkong/tensor.hpp> // `nk::tensor_view`, `nk::tensor_span` for the typed transfer helpers

#include "cuda_core/types.hpp"          // `expected`, `expected_t`, `status_t`
#include "cuda_core/array.hpp"          // `array` for the node-guard walk
#include "cuda_core/gpu_allocators.hpp" // `gpu_arena_alloc` backing the owning rank-1 tensor aliases

namespace unum::uform::core {

namespace nk = ashvardanian::numkong;

/** @brief Rank-1 token-id device / host view — the batch's token currency. */
using tokens_view_t = nk::tensor_view<token_id_t, 1>;
/** @brief Rank-1 token-id mutable device / host span — the batch's token currency. */
using tokens_span_t = nk::tensor_span<token_id_t, 1>;
/** @brief Rank-1 token-id device-only buffer owned through a @ref gpu_arena_t — the token tape. */
using tokens_tensor_arena_t = nk::tensor<token_id_t, gpu_arena_alloc<token_id_t>, 1>;

/**
 *  @brief A move-only owning wrapper over a real (non-null) @c CUstream on one device.
 *
 *  Creation retains the device's primary context and creates the stream inside it, so a caller never
 *  juggles contexts to use the stream; destruction destroys the stream and releases the retained context.
 *  Non-blocking by default: an explicit stream never implicitly synchronizes with the legacy stream.
 */
struct gpu_stream_t {

  private:
    gpu_stream_t() noexcept = default;

  public:
    gpu_stream_t(gpu_stream_t const &) = delete;
    gpu_stream_t &operator=(gpu_stream_t const &) = delete;

    gpu_stream_t(gpu_stream_t &&other) noexcept
        : stream_(std::exchange(other.stream_, nullptr)), context_(std::exchange(other.context_, nullptr)),
          device_(std::exchange(other.device_, -1)) {}
    gpu_stream_t &operator=(gpu_stream_t &&other) noexcept {
        if (this != &other) {
            destroy_();
            stream_ = std::exchange(other.stream_, nullptr);
            context_ = std::exchange(other.context_, nullptr);
            device_ = std::exchange(other.device_, -1);
        }
        return *this;
    }
    ~gpu_stream_t() noexcept { destroy_(); }

    /** @brief Retain @p device 's primary context and create a stream inside it.
     *  @param[in] device The CUDA device ordinal the stream belongs to.
     *  @param[in] flags @c CU_STREAM_NON_BLOCKING by default — no implicit legacy-stream ordering.
     *  @param[in] priority Stream priority; 0 is the default tier, negative is more urgent. */
    [[nodiscard]] static expected<gpu_stream_t> make(int device, unsigned flags = CU_STREAM_NON_BLOCKING,
                                                     int priority = 0) noexcept {
        CUdevice device_handle = 0;
        if (CUresult result = cuDeviceGet(&device_handle, device); result != CUDA_SUCCESS)
            return expected<gpu_stream_t>::fail(result, "device handle query failed (device={})", device);
        gpu_stream_t stream;
        stream.device_ = device;
        if (CUresult result = cuDevicePrimaryCtxRetain(&stream.context_, device_handle); result != CUDA_SUCCESS)
            return expected<gpu_stream_t>::fail(result, "primary context retain failed (device={})", device);
        if (CUresult result = cuCtxPushCurrent(stream.context_); result != CUDA_SUCCESS)
            return expected<gpu_stream_t>::fail(result, "context push failed (device={})", device);
        CUresult const created = cuStreamCreateWithPriority(&stream.stream_, flags, priority);
        cuCtxPopCurrent(nullptr);
        if (created != CUDA_SUCCESS)
            return expected<gpu_stream_t>::fail(created, "stream creation failed (device={} priority={})", device,
                                                priority);
        return expected<gpu_stream_t> {std::move(stream)};
    }

    /** @brief The wrapped stream handle. */
    [[nodiscard]] CUstream get() const noexcept { return stream_; }
    /** @brief Implicit handle access — the stream passes straight into launchers. */
    operator CUstream() const noexcept { return stream_; }
    /** @brief The retained primary context the stream lives in. */
    [[nodiscard]] CUcontext context() const noexcept { return context_; }
    /** @brief The device ordinal the stream belongs to. */
    [[nodiscard]] int device() const noexcept { return device_; }

  private:
    void destroy_() noexcept {
        if (stream_ != nullptr) {
            cuCtxPushCurrent(context_);
            cuStreamDestroy(stream_);
            cuCtxPopCurrent(nullptr);
            stream_ = nullptr;
        }
        if (context_ != nullptr) {
            CUdevice device_handle = 0;
            if (cuDeviceGet(&device_handle, device_) == CUDA_SUCCESS) cuDevicePrimaryCtxRelease(device_handle);
            context_ = nullptr;
        }
        device_ = -1;
    }

    CUstream stream_ = nullptr;
    CUcontext context_ = nullptr;
    int device_ = -1;
};

/** @brief Enqueue a host→device copy of a rank-1 view into a same-typed device span on @p stream — the
 *      typed spelling of @c cuMemcpyHtoDAsync , with the arguments in the order the name reads: host
 *      source, device destination. Extents and element sizes travel with the views, never as
 *      hand-multiplied @c sizeof operands. An empty @p host_source is a no-op; a source longer than the
 *      destination is a loud failure. */
template <typename value_type_>
[[nodiscard]] inline expected_t copy_host_to_device(nk::tensor_view<value_type_, 1> host_source,
                                                    nk::tensor_span<value_type_, 1> device_destination,
                                                    gpu_stream_t &stream) noexcept {
    if (host_source.empty()) return {};
    if (device_destination.extent(0) < host_source.extent(0))
        return expected_t::fail(status_t::invalid_input_k,
                                "device destination ({} scalars) is smaller than the host source ({})",
                                device_destination.extent(0), host_source.extent(0));
    if (CUresult result = cuMemcpyHtoDAsync(reinterpret_cast<CUdeviceptr>(device_destination.data()),
                                            host_source.data(), host_source.extent(0) * sizeof(value_type_),
                                            stream.get());
        result != CUDA_SUCCESS)
        return expected_t::fail(result, "host-to-device copy failed ({} scalars)", host_source.extent(0));
    return {};
}

/** @brief The mirror of @ref copy_host_to_device : enqueue a device→host copy of a rank-1 view into a
 *      same-typed host span on @p stream — the typed spelling of @c cuMemcpyDtoHAsync , device source
 *      first, host destination second. An empty @p device_source is a no-op; a source longer than the
 *      destination is a loud failure. The host span must stay alive until the stream reaches the copy. */
template <typename value_type_>
[[nodiscard]] inline expected_t copy_device_to_host(nk::tensor_view<value_type_, 1> device_source,
                                                    nk::tensor_span<value_type_, 1> host_destination,
                                                    gpu_stream_t &stream) noexcept {
    if (device_source.empty()) return {};
    if (host_destination.extent(0) < device_source.extent(0))
        return expected_t::fail(status_t::invalid_input_k,
                                "host destination ({} scalars) is smaller than the device source ({})",
                                host_destination.extent(0), device_source.extent(0));
    if (CUresult result = cuMemcpyDtoHAsync(host_destination.data(),
                                            reinterpret_cast<CUdeviceptr>(device_source.data()),
                                            device_source.extent(0) * sizeof(value_type_), stream.get());
        result != CUDA_SUCCESS)
        return expected_t::fail(result, "device-to-host copy failed ({} scalars)", device_source.extent(0));
    return {};
}

#if defined(__CUDACC__)

/**
 *  @brief Launch a compiled-in kernel through the Driver API — the checked replacement for the
 *      triple-chevron syntax, which returns no error code, fails silently on impossible configurations,
 *      and drags a hidden CUDA Runtime dependency behind every launch:
 *      https://ashvardanian.com/posts/less-wrong-cuda-hello-world/
 *
 *  The kernel's entry address resolves to a @c CUfunction through @c cudaGetFuncBySymbol — the one
 *  sanctioned interop symbol, declared by nvcc's own implicit headers and serviced by the fatbin
 *  registration stub nvcc statically embeds into every kernel-carrying binary regardless — and the launch
 *  itself is one checked @c cuLaunchKernel . No dynamic @c libcudart enters the link.
 *
 *  Arguments bind by const reference (temporaries included — the driver copies their bytes out before
 *  returning), and every argument's type must match the kernel's parameter type @b exactly: the driver
 *  copies raw bytes, so the implicit conversions a triple-chevron call would perform do not apply —
 *  convert spans to views at the call site.
 */
/** @brief Grid cap of every grid-strided launch: a bandwidth-bound kernel gains nothing past this, and
 *      an unbounded grid stalls capture-time instantiation on huge shards. */
inline constexpr unsigned kernel_grid_cap_k = 4096u;

/** @brief The one grid-size formula for grid-strided kernels over @p scalars elements: at least one
 *      CTA, at most @ref kernel_grid_cap_k . */
[[nodiscard]] inline constexpr unsigned grid_size_for(std::size_t scalars,
                                                      unsigned block = kernel_block_size_default_k) noexcept {
    std::size_t const grid = divide_round_up(scalars, block);
    if (grid == 0) return 1u;
    return grid > kernel_grid_cap_k ? kernel_grid_cap_k : static_cast<unsigned>(grid);
}

template <typename... arg_types_>
[[nodiscard]] inline expected_t launch_kernel(void const *kernel_symbol, dim3 grid_size, dim3 block_size,
                                              gpu_stream_t &stream, arg_types_ const &...args) noexcept {
    static_assert(sizeof...(arg_types_) != 0, "a kernel without arguments has nothing to compute");
    CUfunction function = nullptr;
    if (int result = static_cast<int>(cudaGetFuncBySymbol(&function, kernel_symbol)); result != 0)
        return expected_t::fail(status_t::cuda_error_k, "kernel symbol resolution failed (code={})", result);
    void *argument_addresses[] = {const_cast<void *>(static_cast<void const *>(&args))...};
    if (CUresult result = cuLaunchKernel(function, grid_size.x, grid_size.y, grid_size.z, block_size.x, block_size.y,
                                         block_size.z, 0, stream.get(), argument_addresses, nullptr);
        result != CUDA_SUCCESS)
        return expected_t::fail(result, "kernel launch failed (grid=[{},{},{}] block=[{},{},{}])", grid_size.x,
                                grid_size.y, grid_size.z, block_size.x, block_size.y, block_size.z);
    return {};
}

/**
 *  @brief All-lanes-get-the-total shuffle sum of one warp — the fixed-order tree every deterministic
 *      reduction in the cell builds on.
 *
 *  Hand-rolled rather than @c cub::WarpReduce deliberately: the XOR-shuffle butterfly already delivers
 *  the total to @b every lane with zero shared memory, while CUB's warp reduction returns the total in
 *  lane 0 only and asks for a @c TempStorage — the broadcast we need back costs exactly the shuffle we
 *  started with. CUB stays the tool for the primitives it genuinely owns (device-wide sorts, scans,
 *  @c LowerBound ).
 */
[[nodiscard]] __device__ inline f32_t warp_reduce_broadcast(f32_t value) noexcept {
    for (unsigned delta = warp_size_k / 2u; delta != 0u; delta /= 2u)
        value += __shfl_xor_sync(0xffffffffu, value, delta);
    return value;
}

/** @brief All-threads-get-the-total CTA sum: per-warp shuffle trees, lane-0 partials staged through
 *      static shared memory, one final warp tree, one broadcast cell — fixed-order, deterministic,
 *      sized for @p block_size_ threads. */
template <unsigned block_size_>
[[nodiscard]] __device__ inline f32_t block_reduce_broadcast(f32_t value) noexcept {
    static_assert(block_size_ % warp_size_k == 0, "the staging array assumes whole warps");
    constexpr unsigned warps_k = block_size_ / warp_size_k;
    __shared__ f32_t warp_partials[warps_k];
    __shared__ f32_t block_total;
    unsigned const lane_index = threadIdx.x % warp_size_k;
    unsigned const warp_index = threadIdx.x / warp_size_k;
    f32_t const warp_total = warp_reduce_broadcast(value);
    if (lane_index == 0u) warp_partials[warp_index] = warp_total;
    __syncthreads();
    if (warp_index == 0u) {
        f32_t staged = lane_index < warps_k ? warp_partials[lane_index] : 0.0f;
        staged = warp_reduce_broadcast(staged);
        if (lane_index == 0u) block_total = staged;
    }
    __syncthreads();
    return block_total;
}

#endif // defined(__CUDACC__)

/** @brief How a §10 model region runs: captured once into a @ref gpu_replay_graph_t and replayed every step
 *      (the production default), or re-enqueued eagerly each step — the bring-up / debugging path, and the
 *      other half of every capture-vs-eager parity test. A trainer carries it as a run knob; the two modes must
 *      land on bitwise-identical weights over identical seeds. */
/** @brief How the vendor sessions pin their per-shape algorithms. The default takes the library's own first
 *      deterministic heuristic pick — zero construction cost, the production behavior. The benchmarked mode
 *      times every gate-passing candidate at construction and pins the fastest, memoizing per argument
 *      combination so repeated shapes measure once; picks are measured on this machine at this moment, so
 *      tuned multi-host runs are for profiling, never for bitwise-parity conformance. */
enum class algo_pick_t : u8_t {
    heuristic_k = 0,
    benchmarked_k = 1,
};

enum capture_t : u32_t {
    replayed_k = 0,
    eager_k = 1,
};

/**
 *  @brief The capture-once / replay-forever graph: @ref make turns a body of enqueues into one
 *      uploaded @c CUgraphExec , and every later step is a single @c cuGraphLaunch .
 *
 *  Move-only, and @b always launchable when it exists: construction captures, enforces the no-allocation
 *  contract, instantiates, destroys the source graph, and uploads the executable on the capture stream —
 *  there is no half-built state to query and no deferred work hiding behind the first launch. The capture
 *  body is a template parameter returning @ref expected_t — no type-erased callable, no heap, no
 *  exceptions.
 */
struct gpu_replay_graph_t {

    gpu_replay_graph_t() noexcept = default;
    gpu_replay_graph_t(gpu_replay_graph_t const &) = delete;
    gpu_replay_graph_t &operator=(gpu_replay_graph_t const &) = delete;

    gpu_replay_graph_t(gpu_replay_graph_t &&other) noexcept : exec_(std::exchange(other.exec_, nullptr)) {}
    gpu_replay_graph_t &operator=(gpu_replay_graph_t &&other) noexcept {
        if (this != &other) {
            destroy_();
            exec_ = std::exchange(other.exec_, nullptr);
        }
        return *this;
    }
    ~gpu_replay_graph_t() noexcept { destroy_(); }

    /**
     *  @brief Capture @p body 's enqueues on the single @p stream into a replay-ready executable.
     *
     *  @param[in] stream A real (non-null) stream; the legacy null stream cannot capture. The executable
     *      is also uploaded on it before returning.
     *  @param[in] body A callable returning @ref expected_t whose only effect is enqueuing onto @p stream .
     */
    template <typename body_type_>
    [[nodiscard]] static expected<gpu_replay_graph_t> make(CUstream stream, body_type_ &&body) noexcept {
        if (stream == nullptr)
            return expected<gpu_replay_graph_t>::fail(status_t::invalid_input_k,
                                                      "capture needs a real stream, not the null stream");
        if (CUresult result = cuStreamBeginCapture(stream, CU_STREAM_CAPTURE_MODE_RELAXED); result != CUDA_SUCCESS)
            return expected<gpu_replay_graph_t>::fail(result, "stream capture begin failed");
        expected_t emitted = body();
        CUgraph graph = nullptr;
        CUresult const ended = cuStreamEndCapture(stream, &graph);
        if (!emitted) {
            if (graph != nullptr) cuGraphDestroy(graph);
            return expected<gpu_replay_graph_t>::fail(std::move(emitted));
        }
        if (ended != CUDA_SUCCESS || graph == nullptr) {
            if (graph != nullptr) cuGraphDestroy(graph);
            return expected<gpu_replay_graph_t>::fail(ended, "stream capture end failed");
        }
        return adopt_(graph, stream);
    }

    /**
     *  @brief Capture a multi-GPU step across @p streams into @b one replay-ready executable — the fork /
     *      join topology from the file header, keeping every rank's kernels co-resident at replay.
     *
     *  @param[in] streams One real stream per rank; @c streams[0] is the trunk that begins and ends the
     *      capture and carries the upload. Each stream may live on a different device; peer access must
     *      already be wired.
     *  @param[in] body A callable invoked once per rank as @c body(rank) returning @ref expected_t ,
     *      enqueuing rank-local work onto @c streams[rank] . Branch bodies must not depend on each other.
     */
    template <typename body_type_>
    [[nodiscard]] static expected<gpu_replay_graph_t> make(std::span<CUstream const> streams,
                                                           body_type_ &&body) noexcept {
        if (streams.empty())
            return expected<gpu_replay_graph_t>::fail(status_t::invalid_input_k, "capture needs at least one stream");
        for (std::size_t rank = 0; rank != streams.size(); ++rank)
            if (streams[rank] == nullptr)
                return expected<gpu_replay_graph_t>::fail(status_t::invalid_input_k,
                                                          "capture needs real streams (rank {} is null)", rank);
        if (streams.size() == 1) return make(streams.front(), [&body]() noexcept { return body(std::size_t {0}); });

        CUstream const trunk = streams.front();
        CUevent fork_event = nullptr;
        array<CUevent> join_events;
        if (join_events.try_resize(streams.size() - 1u) != status_t::success_k)
            return expected<gpu_replay_graph_t>::fail(status_t::bad_alloc_k,
                                                      "join-event table allocation failed (ranks={})", streams.size());
        expected_t outcome = make_event_for_(trunk, fork_event);
        for (std::size_t branch = 0; outcome && branch != join_events.size(); ++branch)
            outcome = make_event_for_(streams[branch + 1u], join_events[branch]);

        CUgraph graph = nullptr;
        bool capturing = false;
        if (outcome) {
            CUresult const begun = cuStreamBeginCapture(trunk, CU_STREAM_CAPTURE_MODE_RELAXED);
            capturing = begun == CUDA_SUCCESS;
            if (!capturing) outcome = expected_t::fail(begun, "stream capture begin failed (trunk)");
        }
        if (outcome)
            if (CUresult result = cuEventRecord(fork_event, trunk); result != CUDA_SUCCESS)
                outcome = expected_t::fail(result, "fork-event record failed");
        if (outcome) outcome = body(std::size_t {0});
        for (std::size_t branch = 0; outcome && branch != join_events.size(); ++branch) {
            CUstream const branch_stream = streams[branch + 1u];
            if (CUresult result = cuStreamWaitEvent(branch_stream, fork_event, 0); result != CUDA_SUCCESS) {
                outcome = expected_t::fail(result, "fork-event wait failed (rank={})", branch + 1u);
                break;
            }
            outcome = body(branch + 1u);
            if (!outcome) break;
            if (CUresult result = cuEventRecord(join_events[branch], branch_stream); result != CUDA_SUCCESS)
                outcome = expected_t::fail(result, "join-event record failed (rank={})", branch + 1u);
        }
        for (std::size_t branch = 0; outcome && branch != join_events.size(); ++branch)
            if (CUresult result = cuStreamWaitEvent(trunk, join_events[branch], 0); result != CUDA_SUCCESS)
                outcome = expected_t::fail(result, "join-event wait failed (rank={})", branch + 1u);

        if (capturing) {
            CUresult const ended = cuStreamEndCapture(trunk, &graph);
            if (outcome && (ended != CUDA_SUCCESS || graph == nullptr))
                outcome = expected_t::fail(ended, "stream capture end failed (trunk)");
        }
        (void)destroy_event_(fork_event);
        for (CUevent &event : join_events) (void)destroy_event_(event);
        if (!outcome) {
            if (graph != nullptr) cuGraphDestroy(graph);
            return expected<gpu_replay_graph_t>::fail(std::move(outcome));
        }
        return adopt_(graph, trunk);
    }

    /** @brief Replay the executable on @p stream : one @c cuGraphLaunch — staging already happened at
     *      @ref make . */
    [[nodiscard]] expected_t launch(CUstream stream) const noexcept {
        if (exec_ == nullptr)
            return expected_t::fail(status_t::invalid_input_k, "launching an empty (default or moved-from) graph");
        if (CUresult result = cuGraphLaunch(exec_, stream); result != CUDA_SUCCESS)
            return expected_t::fail(result, "graph launch failed");
        return {};
    }

  private:
    /** @brief Enforce the no-allocation contract, instantiate @p graph , and upload the executable on
     *      @p stream — after this the object is fully replay-ready. */
    [[nodiscard]] static expected<gpu_replay_graph_t> adopt_(CUgraph graph, CUstream stream) noexcept {
        expected_t guarded = guard_alloc_nodes_(graph);
        if (!guarded) {
            cuGraphDestroy(graph);
            return expected<gpu_replay_graph_t>::fail(std::move(guarded));
        }
        gpu_replay_graph_t replay;
        CUresult const instantiated = cuGraphInstantiate(&replay.exec_, graph, 0);
        cuGraphDestroy(graph);
        if (instantiated != CUDA_SUCCESS) {
            replay.exec_ = nullptr;
            return expected<gpu_replay_graph_t>::fail(instantiated, "graph instantiation failed");
        }
        if (CUresult result = cuGraphUpload(replay.exec_, stream); result != CUDA_SUCCESS)
            return expected<gpu_replay_graph_t>::fail(result, "graph upload failed");
        return expected<gpu_replay_graph_t> {std::move(replay)};
    }

    /** @brief The no-allocation contract, enforced: any allocation / free node in the captured graph is a
     *      loud failure — it escapes the arena budget and deadlocks device-local multi-GPU first launch. */
    [[nodiscard]] static expected_t guard_alloc_nodes_(CUgraph graph) noexcept {
        std::size_t node_count = 0;
        if (CUresult result = cuGraphGetNodes(graph, nullptr, &node_count); result != CUDA_SUCCESS)
            return expected_t::fail(result, "graph node query failed");
        if (node_count == 0) return {};
        array<CUgraphNode> nodes;
        if (nodes.try_resize_uninitialized(node_count) != status_t::success_k)
            return expected_t::fail(status_t::bad_alloc_k, "graph node table allocation failed (nodes={})", node_count);
        if (CUresult result = cuGraphGetNodes(graph, nodes.data(), &node_count); result != CUDA_SUCCESS)
            return expected_t::fail(result, "graph node listing failed");
        std::size_t allocation_nodes = 0;
        for (CUgraphNode node : nodes) {
            CUgraphNodeType node_type;
            if (CUresult result = cuGraphNodeGetType(node, &node_type); result != CUDA_SUCCESS)
                return expected_t::fail(result, "graph node type query failed");
            allocation_nodes += node_type == CU_GRAPH_NODE_TYPE_MEM_ALLOC || node_type == CU_GRAPH_NODE_TYPE_MEM_FREE;
        }
        if (allocation_nodes != 0)
            return expected_t::fail(                                                                          //
                status_t::invalid_input_k,                                                                    //
                "the captured body allocates: {} alloc/free nodes among {} — an allocating body escapes the " //
                "arena budget and deadlocks device-local multi-GPU first launch; stage buffers before capture",
                allocation_nodes, node_count);
        return {};
    }

    /** @brief Create a timing-free event in @p stream 's own context, so recording it there is legal. */
    [[nodiscard]] static expected_t make_event_for_(CUstream stream, CUevent &event) noexcept {
        CUcontext stream_context = nullptr;
        if (CUresult result = cuStreamGetCtx(stream, &stream_context); result != CUDA_SUCCESS)
            return expected_t::fail(result, "stream context query failed");
        if (CUresult result = cuCtxPushCurrent(stream_context); result != CUDA_SUCCESS)
            return expected_t::fail(result, "context push failed");
        CUresult const created = cuEventCreate(&event, CU_EVENT_DISABLE_TIMING);
        cuCtxPopCurrent(nullptr);
        if (created != CUDA_SUCCESS) return expected_t::fail(created, "event creation failed");
        return {};
    }

    static bool destroy_event_(CUevent &event) noexcept {
        if (event == nullptr) return true;
        bool const destroyed = cuEventDestroy(event) == CUDA_SUCCESS;
        event = nullptr;
        return destroyed;
    }

    void destroy_() noexcept {
        if (exec_ != nullptr) cuGraphExecDestroy(exec_);
        exec_ = nullptr;
    }

    CUgraphExec exec_ = nullptr;
};

/**
 *  @brief A shared cuBLASLt session: one handle per device, owned once and borrowed by every plan — the
 *      tensor-core matmul entry point for losses, projection heads, and anything else with large matrix
 *      products. Slow primitives are hated here: wide GEMMs go through this, never hand-rolled loops.
 */
struct cublaslt_algo_picks_t;

struct cublaslt_session_t {

  private:
    cublaslt_session_t() noexcept = default;

  public:
    cublaslt_session_t(cublaslt_session_t const &) = delete;
    cublaslt_session_t &operator=(cublaslt_session_t const &) = delete;
    cublaslt_session_t(cublaslt_session_t &&other) noexcept
        : handle_(std::exchange(other.handle_, nullptr)), algo_pick_(other.algo_pick_),
          picks_(std::exchange(other.picks_, nullptr)) {}
    cublaslt_session_t &operator=(cublaslt_session_t &&other) noexcept {
        if (this != &other) {
            destroy_();
            handle_ = std::exchange(other.handle_, nullptr);
            algo_pick_ = other.algo_pick_;
            picks_ = std::exchange(other.picks_, nullptr);
        }
        return *this;
    }
    ~cublaslt_session_t() noexcept { destroy_(); }

    /** @brief Create the handle — once per device, outside any capture. @p algo_pick is the run's algorithm
     *      policy and @p picks the composition-root-owned memo the benchmarked mode fills; under
     *      @ref algo_pick_t::heuristic_k the memo is never touched and may be null. */
    [[nodiscard]] static expected<cublaslt_session_t> make(algo_pick_t algo_pick,
                                                           cublaslt_algo_picks_t *picks) noexcept {
        cublaslt_session_t session;
        session.algo_pick_ = algo_pick;
        session.picks_ = picks;
        if (cublasStatus_t status = cublasLtCreate(&session.handle_); status != CUBLAS_STATUS_SUCCESS)
            return expected<cublaslt_session_t>::fail(
                status_t::cuda_error_k, "cuBLASLt handle creation failed (status={})", static_cast<int>(status));
        return expected<cublaslt_session_t> {std::move(session)};
    }

    /** @brief The wrapped handle, borrowed by plans. */
    [[nodiscard]] cublasLtHandle_t get() const noexcept { return handle_; }
    /** @brief The run's algorithm policy, applied by every plan this session builds. */
    [[nodiscard]] algo_pick_t algo_pick() const noexcept { return algo_pick_; }
    /** @brief The shared per-shape pick memo; null under the heuristic policy. */
    [[nodiscard]] cublaslt_algo_picks_t *picks() const noexcept { return picks_; }

  private:
    void destroy_() noexcept {
        if (handle_ != nullptr) cublasLtDestroy(std::exchange(handle_, nullptr));
    }

    cublasLtHandle_t handle_ = nullptr;
    algo_pick_t algo_pick_ = algo_pick_t::heuristic_k;
    cublaslt_algo_picks_t *picks_ = nullptr;
};

/**
 *  @brief The static description of one @ref cublaslt_plan_t : stored operand extents and row strides
 *      (all row-major, in elements), the transpose ops, and the effective product dimensions they imply.
 *      BFloat16 operands with Float32 accumulation and output — the one precision recipe this project
 *      trains with.
 */
struct cublaslt_plan_config_t {
    /** @brief Stored extents of operand A, before @ref transpose_a . */
    u32_t a_rows = 0, a_cols = 0;
    /** @brief Elements between consecutive rows of A (its allocated row width). */
    u32_t a_row_stride = 0;
    /** @brief Stored extents of operand B, before @ref transpose_b . */
    u32_t b_rows = 0, b_cols = 0;
    /** @brief Elements between consecutive rows of B. */
    u32_t b_row_stride = 0;
    /** @brief Extents of the output C. */
    u32_t c_rows = 0, c_cols = 0;
    /** @brief Elements between consecutive rows of C. */
    u32_t c_row_stride = 0;
    /** @brief Whether the product uses the transposed reading of the stored operand. */
    bool transpose_a = false, transpose_b = false;
    /** @brief Operand element types — BFloat16 by default; @c CUDA_R_8F_E4M3 selects the FP8 path. */
    cudaDataType_t a_type = CUDA_R_16BF, b_type = CUDA_R_16BF;
    /** @brief Output element type — Float32 accumulator surface by default, BFloat16 for activations. */
    cudaDataType_t c_type = CUDA_R_32F;
    /** @brief Optional per-tensor FP8 descale scalars (Float32, device-resident, capture-stable
     *      addresses) pinned into the descriptor at @ref cublaslt_plan_t::make — the tensor-core native
     *      dequantisation, no post-matmul descale kernel. Null means unscaled. */
    f32_t const *a_descale = nullptr;
    f32_t const *b_descale = nullptr;
};

/** @brief One matmul argument combination — every @ref cublaslt_plan_config_t field except the runtime
 *      descale addresses, which travel as presence bits: two plans differing only in where their descale
 *      scalars live pin the same algorithm. */
struct cublaslt_algo_key_t {
    u32_t a_rows = 0, a_cols = 0, a_row_stride = 0;
    u32_t b_rows = 0, b_cols = 0, b_row_stride = 0;
    u32_t c_rows = 0, c_cols = 0, c_row_stride = 0;
    u32_t a_type = 0, b_type = 0, c_type = 0;
    u8_t transpose_a = 0, transpose_b = 0;
    u8_t a_descaled = 0, b_descaled = 0;

    friend bool operator==(cublaslt_algo_key_t const &, cublaslt_algo_key_t const &) noexcept = default;

    [[nodiscard]] static cublaslt_algo_key_t from(cublaslt_plan_config_t const &config) noexcept {
        cublaslt_algo_key_t key;
        key.a_rows = config.a_rows, key.a_cols = config.a_cols, key.a_row_stride = config.a_row_stride;
        key.b_rows = config.b_rows, key.b_cols = config.b_cols, key.b_row_stride = config.b_row_stride;
        key.c_rows = config.c_rows, key.c_cols = config.c_cols, key.c_row_stride = config.c_row_stride;
        key.a_type = static_cast<u32_t>(config.a_type), key.b_type = static_cast<u32_t>(config.b_type);
        key.c_type = static_cast<u32_t>(config.c_type);
        key.transpose_a = config.transpose_a ? 1u : 0u, key.transpose_b = config.transpose_b ? 1u : 0u;
        key.a_descaled = config.a_descale != nullptr ? 1u : 0u;
        key.b_descaled = config.b_descale != nullptr ? 1u : 0u;
        return key;
    }
};

/** @brief One benchmarked pick: the argument combination it answers and the fastest deterministic algorithm
 *      measured for it, with the measurement for the construction report. */
struct cublaslt_algo_pick_t {
    cublaslt_algo_key_t key {};
    cublasLtMatmulAlgo_t algorithm {};
    std::size_t workspace_bytes = 0;
    f32_t microseconds = 0.0f;
};

/** @brief The run's per-shape pick memo — owned at the composition root beside the session tables and shared
 *      by every replica's session, so identical blocks across replicas measure once. A future cluster-wide
 *      pre-profiling pass (uniform GPUs timing disjoint shape subsets simultaneously) merges into this same
 *      container. */
struct cublaslt_algo_picks_t {

    /** @brief The recorded pick for @p key , or null when the shape is unmeasured. */
    [[nodiscard]] cublaslt_algo_pick_t const *find(cublaslt_algo_key_t const &key) const noexcept {
        for (cublaslt_algo_pick_t const &pick : std::span<cublaslt_algo_pick_t const> {picks_.view()})
            if (pick.key == key) return &pick;
        return nullptr;
    }

    /** @brief Record one measured pick. */
    [[nodiscard]] status_t try_record(cublaslt_algo_pick_t pick) noexcept { return picks_.try_push_back(pick); }

    /** @brief How many distinct argument combinations were measured. */
    [[nodiscard]] std::size_t size() const noexcept { return picks_.size(); }

    /** @brief The summed per-call winners — the construction report's one number. */
    [[nodiscard]] f32_t total_microseconds() const noexcept {
        f32_t total = 0.0f;
        for (cublaslt_algo_pick_t const &pick : std::span<cublaslt_algo_pick_t const> {picks_.view()})
            total += pick.microseconds;
        return total;
    }

  private:
    array<cublaslt_algo_pick_t> picks_ {};
};

/** @brief One measured winner among a plan's gate-passing candidates — @ref cublaslt_benchmark_candidates
 *      hands it back for @ref cublaslt_plan_t::make to pin and memoize. */
struct cublaslt_measured_t {
    std::size_t candidate_index = 0;
    std::size_t workspace_bytes = 0;
    f32_t microseconds = 0.0f;
};

/** @brief Time every gate-passing candidate of one product on transient device operands and return the
 *      fastest — the benchmarked pick's stopwatch, compiled in @c schedule.cu . Allocates the operands and
 *      a workspace for the run's duration only, so benchmarked construction needs transient headroom for
 *      the largest product it measures. */
[[nodiscard]] expected<cublaslt_measured_t> cublaslt_benchmark_candidates(
    cublaslt_session_t &session, cublasLtMatmulDesc_t description, cublasLtMatrixLayout_t a_layout,
    cublasLtMatrixLayout_t b_layout, cublasLtMatrixLayout_t c_layout, cublaslt_plan_config_t const &config,
    std::span<cublasLtMatmulHeuristicResult_t const> candidates) noexcept;

/**
 *  @brief One pre-planned tensor-core matmul: descriptors, layouts, and the pinned algorithm — the
 *      session's policy picks it (see @ref algo_pick_t ) at @ref make , so a @ref matmul call is a pure
 *      enqueue with a fixed kernel — hidden allocation cannot happen (the workspace is caller-owned) and
 *      results replay bit-identically (a pinned algorithm never re-tunes).
 */
struct cublaslt_plan_t {

    cublaslt_plan_t() noexcept = default;
    cublaslt_plan_t(cublaslt_plan_t const &) = delete;
    cublaslt_plan_t &operator=(cublaslt_plan_t const &) = delete;
    cublaslt_plan_t(cublaslt_plan_t &&other) noexcept
        : description_(std::exchange(other.description_, nullptr)), a_layout_(std::exchange(other.a_layout_, nullptr)),
          b_layout_(std::exchange(other.b_layout_, nullptr)), c_layout_(std::exchange(other.c_layout_, nullptr)),
          algorithm_(other.algorithm_), workspace_bytes_(std::exchange(other.workspace_bytes_, 0)),
          a_cols_(std::exchange(other.a_cols_, 0)), b_cols_(std::exchange(other.b_cols_, 0)),
          c_cols_(std::exchange(other.c_cols_, 0)) {}
    cublaslt_plan_t &operator=(cublaslt_plan_t &&other) noexcept {
        if (this != &other) {
            destroy_();
            description_ = std::exchange(other.description_, nullptr);
            a_layout_ = std::exchange(other.a_layout_, nullptr);
            b_layout_ = std::exchange(other.b_layout_, nullptr);
            c_layout_ = std::exchange(other.c_layout_, nullptr);
            algorithm_ = other.algorithm_;
            workspace_bytes_ = std::exchange(other.workspace_bytes_, 0);
            a_cols_ = std::exchange(other.a_cols_, 0);
            b_cols_ = std::exchange(other.b_cols_, 0);
            c_cols_ = std::exchange(other.c_cols_, 0);
        }
        return *this;
    }
    ~cublaslt_plan_t() noexcept { destroy_(); }

    /** @brief Build the descriptors and pin the best @b deterministic heuristic algorithm — once, at
     *      setup, outside any capture. The engine's chooser runs unbounded and reports what its pick
     *      needs ( @ref workspace_bytes — the caller allocates exactly that, and the arena budget is
     *      the real constraint); determinism is enforced by name, never by a size cap: the pinned
     *      algorithm must be single-split with no cross-CTA reduction, so no atomics ever touch the
     *      accumulator. The preference restricts the search to reduction-free algorithms up front and
     *      scans a deep candidate pool — for unaligned shapes (the 258-wide byte-vocabulary decoder)
     *      the deterministic pick ranks well below every split-K candidate, so an unrestricted or
     *      shallow scan would surface nothing the gate accepts. */
    [[nodiscard]] static expected<cublaslt_plan_t> make(cublaslt_session_t &session,
                                                        cublaslt_plan_config_t config) noexcept {
        cublaslt_plan_t plan;
        if (cublasStatus_t status = cublasLtMatmulDescCreate(&plan.description_, CUBLAS_COMPUTE_32F, CUDA_R_32F);
            status != CUBLAS_STATUS_SUCCESS)
            return expected<cublaslt_plan_t>::fail(
                status_t::cuda_error_k, "matmul description creation failed (status={})", static_cast<int>(status));
        cublasOperation_t const a_operation = config.transpose_a ? CUBLAS_OP_T : CUBLAS_OP_N;
        cublasOperation_t const b_operation = config.transpose_b ? CUBLAS_OP_T : CUBLAS_OP_N;
        cublasLtMatmulDescSetAttribute(plan.description_, CUBLASLT_MATMUL_DESC_TRANSA, &a_operation,
                                       sizeof(a_operation));
        cublasLtMatmulDescSetAttribute(plan.description_, CUBLASLT_MATMUL_DESC_TRANSB, &b_operation,
                                       sizeof(b_operation));
        // FP8 dequantisation is the tensor core's own: the descale scalars live at capture-stable device
        // addresses, so pinning the pointers into the descriptor is replay-safe — values may change,
        // addresses never do. Fast accumulation is the FP8-native mode.
        if (config.a_descale != nullptr)
            cublasLtMatmulDescSetAttribute(plan.description_, CUBLASLT_MATMUL_DESC_A_SCALE_POINTER, &config.a_descale,
                                           sizeof(config.a_descale));
        if (config.b_descale != nullptr)
            cublasLtMatmulDescSetAttribute(plan.description_, CUBLASLT_MATMUL_DESC_B_SCALE_POINTER, &config.b_descale,
                                           sizeof(config.b_descale));
        if (config.a_type == CUDA_R_8F_E4M3 || config.b_type == CUDA_R_8F_E4M3) {
            std::int8_t const fast_accumulation = 1;
            cublasLtMatmulDescSetAttribute(plan.description_, CUBLASLT_MATMUL_DESC_FAST_ACCUM, &fast_accumulation,
                                           sizeof(fast_accumulation));
        }

        // Row-major layouts of the stored operands; the ops above provide any transposed reading.
        auto make_layout = [](cublasLtMatrixLayout_t &layout, cudaDataType_t type, u32_t rows, u32_t cols,
                              u32_t row_stride) noexcept {
            if (cublasLtMatrixLayoutCreate(&layout, type, rows, cols, static_cast<i64_t>(row_stride)) !=
                CUBLAS_STATUS_SUCCESS)
                return false;
            cublasLtOrder_t const row_major = CUBLASLT_ORDER_ROW;
            return cublasLtMatrixLayoutSetAttribute(layout, CUBLASLT_MATRIX_LAYOUT_ORDER, &row_major,
                                                    sizeof(row_major)) == CUBLAS_STATUS_SUCCESS;
        };
        if (!make_layout(plan.a_layout_, config.a_type, config.a_rows, config.a_cols, config.a_row_stride) ||
            !make_layout(plan.b_layout_, config.b_type, config.b_rows, config.b_cols, config.b_row_stride) ||
            !make_layout(plan.c_layout_, config.c_type, config.c_rows, config.c_cols, config.c_row_stride))
            return expected<cublaslt_plan_t>::fail(status_t::cuda_error_k, "matrix layout creation failed");

        cublasLtMatmulPreference_t preference = nullptr;
        if (cublasLtMatmulPreferenceCreate(&preference) != CUBLAS_STATUS_SUCCESS)
            return expected<cublaslt_plan_t>::fail(status_t::cuda_error_k, "matmul preference creation failed");
        // No artificial ceiling: the chooser ranks freely, the pick reports its exact appetite, and
        // the arena decides whether that fits — loudly, at the caller's allocation.
        std::uint64_t const unbounded = ~std::uint64_t {0};
        cublasLtMatmulPreferenceSetAttribute(preference, CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, &unbounded,
                                             sizeof(unbounded));
        std::uint32_t const reduction_none = CUBLASLT_REDUCTION_SCHEME_NONE;
        // Reduction-free picks only — see the brief: split-K would outrank the deterministic algorithm.
        cublasLtMatmulPreferenceSetAttribute(preference, CUBLASLT_MATMUL_PREF_REDUCTION_SCHEME_MASK, &reduction_none,
                                             sizeof(reduction_none));
        std::array<cublasLtMatmulHeuristicResult_t, 64> heuristics {};
        int heuristic_count = 0;
        cublasStatus_t const searched = cublasLtMatmulAlgoGetHeuristic(
            session.get(), plan.description_, plan.a_layout_, plan.b_layout_, plan.c_layout_, plan.c_layout_,
            preference, static_cast<int>(heuristics.size()), heuristics.data(), &heuristic_count);
        cublasLtMatmulPreferenceDestroy(preference);
        if (searched != CUBLAS_STATUS_SUCCESS || heuristic_count == 0)
            return expected<cublaslt_plan_t>::fail( //
                status_t::shape_unsupported_k,
                "no tensor-core algorithm for this product (a=[{}x{}] b=[{}x{}] c=[{}x{}] status={})", config.a_rows,
                config.a_cols, config.b_rows, config.b_cols, config.c_rows, config.c_cols, static_cast<int>(searched));
        // The determinism gate: single split, no cross-CTA reduction scheme — the properties a size
        // cap only ever enforced by accident.
        std::array<cublasLtMatmulHeuristicResult_t, 64> passing {};
        std::size_t passing_count = 0;
        for (int candidate = 0; candidate != heuristic_count; ++candidate) {
            cublasLtMatmulHeuristicResult_t const &heuristic = heuristics[static_cast<std::size_t>(candidate)];
            std::int32_t splits = 0;
            std::uint32_t reduction = CUBLASLT_REDUCTION_SCHEME_NONE;
            std::size_t written = 0;
            if (cublasLtMatmulAlgoConfigGetAttribute(&heuristic.algo, CUBLASLT_ALGO_CONFIG_SPLITK_NUM, &splits,
                                                     sizeof(splits), &written) != CUBLAS_STATUS_SUCCESS)
                continue;
            if (cublasLtMatmulAlgoConfigGetAttribute(&heuristic.algo, CUBLASLT_ALGO_CONFIG_REDUCTION_SCHEME, &reduction,
                                                     sizeof(reduction), &written) != CUBLAS_STATUS_SUCCESS)
                continue;
            if (splits > 1 || reduction != CUBLASLT_REDUCTION_SCHEME_NONE) continue;
            passing[passing_count++] = heuristic;
        }
        if (passing_count == 0)
            return expected<cublaslt_plan_t>::fail(
                status_t::shape_unsupported_k,
                "no deterministic tensor-core algorithm among {} candidates (a=[{}x{}] b=[{}x{}] c=[{}x{}])",
                heuristic_count, config.a_rows, config.a_cols, config.b_rows, config.b_cols, config.c_rows,
                config.c_cols);
        plan.a_cols_ = config.a_cols;
        plan.b_cols_ = config.b_cols;
        plan.c_cols_ = config.c_cols;

        // The session's policy picks among the gate-passers: the heuristic's own first choice by default,
        // or the memoized measured winner under the benchmarked mode.
        if (session.algo_pick() == algo_pick_t::benchmarked_k && session.picks() != nullptr) {
            cublaslt_algo_key_t const key = cublaslt_algo_key_t::from(config);
            if (cublaslt_algo_pick_t const *memoized = session.picks()->find(key); memoized != nullptr) {
                plan.algorithm_ = memoized->algorithm;
                plan.workspace_bytes_ = memoized->workspace_bytes;
                return expected<cublaslt_plan_t> {std::move(plan)};
            }
            expected<cublaslt_measured_t> measured = cublaslt_benchmark_candidates(
                session, plan.description_, plan.a_layout_, plan.b_layout_, plan.c_layout_, config,
                {passing.data(), passing_count});
            if (!measured) return expected<cublaslt_plan_t>::fail(std::move(measured));
            cublasLtMatmulHeuristicResult_t const &winner = passing[measured.value_.candidate_index];
            cublaslt_algo_pick_t pick;
            pick.key = key;
            pick.algorithm = winner.algo;
            pick.workspace_bytes = winner.workspaceSize;
            pick.microseconds = measured.value_.microseconds;
            if (session.picks()->try_record(pick) != status_t::success_k)
                return expected<cublaslt_plan_t>::fail(status_t::bad_alloc_k, "algorithm pick memo append failed");
            plan.algorithm_ = winner.algo;
            plan.workspace_bytes_ = winner.workspaceSize;
            return expected<cublaslt_plan_t> {std::move(plan)};
        }
        plan.algorithm_ = passing[0].algo;
        plan.workspace_bytes_ = passing[0].workspaceSize;
        return expected<cublaslt_plan_t> {std::move(plan)};
    }

    /** @brief The workspace bytes the pinned algorithm needs; the caller sizes one shared buffer to the
     *      maximum over its plans. */
    [[nodiscard]] std::size_t workspace_bytes() const noexcept { return workspace_bytes_; }

  private:
    /** @brief The shared enqueue: the row extent of each operand rides its view (set fresh every call —
     *      no persistent layout state to revert), while the inner extents must equal the columns the
     *      plan pinned at @ref make . */
    /** @brief The row-major leading dimension (elements between consecutive rows) of an operand view —
     *      @c extent(1) for a contiguous buffer, the parent width for a column slice. */
    template <typename view_type_>
    [[nodiscard]] static u32_t leading_dimension_(view_type_ const &view) noexcept {
        using element_type = typename view_type_::value_type;
        return static_cast<u32_t>(view.stride_bytes(0) /
                                  static_cast<typename view_type_::difference_type>(sizeof(element_type)));
    }

    [[nodiscard]] expected_t run_(cublaslt_session_t const &session, void const *a, u32_t a_rows, u32_t a_cols,
                                  u32_t a_ld, void const *b, u32_t b_rows, u32_t b_cols, u32_t b_ld, void *c,
                                  u32_t c_rows, u32_t c_cols, u32_t c_ld, f32_t beta, b8r1_span_t workspace,
                                  gpu_stream_t const &stream) const noexcept {
        if (a_cols != a_cols_ || b_cols != b_cols_ || c_cols != c_cols_)
            return expected_t::fail(status_t::shape_unsupported_k,
                                    "operand columns {{{},{},{}}} disagree with the plan {{{},{},{}}}", a_cols, b_cols,
                                    c_cols, a_cols_, b_cols_, c_cols_);
        auto set_shape = [](cublasLtMatrixLayout_t layout, u32_t rows, u32_t leading) noexcept {
            std::uint64_t const row_extent = rows;
            std::int64_t const stride = leading;
            return cublasLtMatrixLayoutSetAttribute(layout, CUBLASLT_MATRIX_LAYOUT_ROWS, &row_extent,
                                                    sizeof(row_extent)) == CUBLAS_STATUS_SUCCESS &&
                   cublasLtMatrixLayoutSetAttribute(layout, CUBLASLT_MATRIX_LAYOUT_LD, &stride, sizeof(stride)) ==
                       CUBLAS_STATUS_SUCCESS;
        };
        if (!set_shape(a_layout_, a_rows, a_ld) || !set_shape(b_layout_, b_rows, b_ld) ||
            !set_shape(c_layout_, c_rows, c_ld))
            return expected_t::fail(status_t::cuda_error_k, "layout row/stride update failed");
        f32_t const alpha = 1.0f;
        if (cublasStatus_t status = cublasLtMatmul(session.get(), description_, &alpha, a, a_layout_, b, b_layout_,
                                                   &beta, c, c_layout_, c, c_layout_, &algorithm_, workspace.data(),
                                                   workspace.extent(0), stream.get());
            status != CUBLAS_STATUS_SUCCESS)
            return expected_t::fail(status_t::cuda_error_k, "tensor-core matmul failed (status={})",
                                    static_cast<int>(status));
        return {};
    }

  public:
    /** @brief BFloat16 operands. Rows ride the views; the K/N columns are checked against the plan. */
    [[nodiscard]] expected_t matmul(cublaslt_session_t const &session, bf16r2_view_t a, bf16r2_view_t b,
                                    bf16r2_span_t c, f32_t beta, b8r1_span_t workspace,
                                    gpu_stream_t const &stream) const noexcept {
        return run_(session, a.data(), static_cast<u32_t>(a.extent(0)), static_cast<u32_t>(a.extent(1)),
                    leading_dimension_(a), b.data(), static_cast<u32_t>(b.extent(0)), static_cast<u32_t>(b.extent(1)),
                    leading_dimension_(b), c.data(), static_cast<u32_t>(c.extent(0)), static_cast<u32_t>(c.extent(1)),
                    leading_dimension_(c), beta, workspace, stream);
    }
    /** @brief E4M3 operands (descales pinned at make); C is a BFloat16 activation surface. */
    [[nodiscard]] expected_t matmul(cublaslt_session_t const &session, e4m3r2_view_t a, e4m3r2_view_t b,
                                    bf16r2_span_t c, f32_t beta, b8r1_span_t workspace,
                                    gpu_stream_t const &stream) const noexcept {
        return run_(session, a.data(), static_cast<u32_t>(a.extent(0)), static_cast<u32_t>(a.extent(1)),
                    leading_dimension_(a), b.data(), static_cast<u32_t>(b.extent(0)), static_cast<u32_t>(b.extent(1)),
                    leading_dimension_(b), c.data(), static_cast<u32_t>(c.extent(0)), static_cast<u32_t>(c.extent(1)),
                    leading_dimension_(c), beta, workspace, stream);
    }
    /** @brief BFloat16 operands accumulating into a Float32 surface — the loss-logit and gradient-fold
     *      products whose plans pin @c c_type=CUDA_R_32F at @ref make . */
    [[nodiscard]] expected_t matmul(cublaslt_session_t const &session, bf16r2_view_t a, bf16r2_view_t b, f32r2_span_t c,
                                    f32_t beta, b8r1_span_t workspace, gpu_stream_t const &stream) const noexcept {
        return run_(session, a.data(), static_cast<u32_t>(a.extent(0)), static_cast<u32_t>(a.extent(1)),
                    leading_dimension_(a), b.data(), static_cast<u32_t>(b.extent(0)), static_cast<u32_t>(b.extent(1)),
                    leading_dimension_(b), c.data(), static_cast<u32_t>(c.extent(0)), static_cast<u32_t>(c.extent(1)),
                    leading_dimension_(c), beta, workspace, stream);
    }

  private:
    void destroy_() noexcept {
        if (c_layout_ != nullptr) cublasLtMatrixLayoutDestroy(std::exchange(c_layout_, nullptr));
        if (b_layout_ != nullptr) cublasLtMatrixLayoutDestroy(std::exchange(b_layout_, nullptr));
        if (a_layout_ != nullptr) cublasLtMatrixLayoutDestroy(std::exchange(a_layout_, nullptr));
        if (description_ != nullptr) cublasLtMatmulDescDestroy(std::exchange(description_, nullptr));
    }

    cublasLtMatmulDesc_t description_ = nullptr;
    cublasLtMatrixLayout_t a_layout_ = nullptr;
    cublasLtMatrixLayout_t b_layout_ = nullptr;
    cublasLtMatrixLayout_t c_layout_ = nullptr;
    cublasLtMatmulAlgo_t algorithm_ {};
    std::size_t workspace_bytes_ = 0;
    /** @brief The K/N columns pinned at @ref make — the inner extents a call's operands must match; the
     *      row extents are the only per-call freedom (they ride the caller's views). */
    u32_t a_cols_ = 0, b_cols_ = 0, c_cols_ = 0;
};

/**
 *  @brief A move-only owning wrapper over a @c CUevent — the cross-stream ordering token (fork / join
 *      seams, gather-ready signals) with the timing machinery disabled.
 */
struct gpu_event_t {
    gpu_event_t() noexcept = default;
    gpu_event_t(gpu_event_t const &) = delete;
    gpu_event_t &operator=(gpu_event_t const &) = delete;
    gpu_event_t(gpu_event_t &&other) noexcept : event_(std::exchange(other.event_, nullptr)) {}
    gpu_event_t &operator=(gpu_event_t &&other) noexcept {
        if (this != &other) {
            if (event_ != nullptr) cuEventDestroy(event_);
            event_ = std::exchange(other.event_, nullptr);
        }
        return *this;
    }
    ~gpu_event_t() noexcept {
        if (event_ != nullptr) cuEventDestroy(event_);
    }

    /** @brief Create the event — once, at setup. */
    [[nodiscard]] static expected<gpu_event_t> make() noexcept {
        gpu_event_t event;
        if (CUresult result = cuEventCreate(&event.event_, CU_EVENT_DISABLE_TIMING); result != CUDA_SUCCESS)
            return expected<gpu_event_t>::fail(result, "event creation failed");
        return expected<gpu_event_t> {std::move(event)};
    }

    /** @brief Create a timing-capable event — the benchmarked algorithm pick's stopwatch; ordering-only
     *      consumers keep @ref make 's cheaper @c CU_EVENT_DISABLE_TIMING events. */
    [[nodiscard]] static expected<gpu_event_t> make_timed() noexcept {
        gpu_event_t event;
        if (CUresult result = cuEventCreate(&event.event_, CU_EVENT_DEFAULT); result != CUDA_SUCCESS)
            return expected<gpu_event_t>::fail(result, "timed event creation failed");
        return expected<gpu_event_t> {std::move(event)};
    }

    /** @brief The wrapped handle for record / wait calls. */
    [[nodiscard]] CUevent get() const noexcept { return event_; }

  private:
    CUevent event_ = nullptr;
};

/**
 *  @brief The monotonic step counter of one collective surface — the device round scalar @b is
 *      the count, and it is the sole source of truth (no host shadow, no pinned cell). One behavior:
 *      @ref advance enqueues a one-thread device increment (@c round += increment , then a
 *      @c __threadfence_system() ), once per step on the surface's lead stream before any barrier.
 *      Because the bump lives on the device, a captured collective region self-advances the round
 *      on every @c cuGraphLaunch replay with no host intervention. The scalar is seeded @c 0u-increment
 *      at @ref make so the first advance yields round 0. Counts never reset; a replayed barrier's
 *      target only grows.
 */
struct barrier_round_t {

    barrier_round_t() noexcept = default;
    barrier_round_t(barrier_round_t const &) = delete;
    barrier_round_t &operator=(barrier_round_t const &) = delete;
    barrier_round_t(barrier_round_t &&) noexcept = default;
    barrier_round_t &operator=(barrier_round_t &&) noexcept = default;

    /** @brief Seed the device round scalar — once, at setup, with a current context.
     *  @param[in] device_round The one-scalar device destination, borrowed for this object's lifetime.
     *  @param[in] increment How far one step advances the count (a member count, a round count).
     *  @param[in] stream The lead stream the seed is enqueued on, ordered before the first @ref advance. */
    [[nodiscard]] static expected<barrier_round_t> make(u32r1_span_t device_round, u32_t increment,
                                                        gpu_stream_t &stream) noexcept {
        barrier_round_t counter;
        counter.device_round_ = device_round;
        counter.increment_ = increment;
        // Seed one increment below zero so the first advance yields round 0 (device is the sole source of truth).
        if (CUresult result = cuMemsetD32Async(reinterpret_cast<CUdeviceptr>(device_round.data()), 0u - increment, 1,
                                               stream.get());
            result != CUDA_SUCCESS)
            return expected<barrier_round_t>::fail(result, "round seed failed");
        return expected<barrier_round_t> {std::move(counter)};
    }

    /** @brief Enqueue the one-thread device increment — once per step, on the surface's lead stream,
     *      before any barrier. Capture-safe: a captured region re-runs the bump on every replay. */
    [[nodiscard]] expected_t advance(gpu_stream_t &stream) noexcept;

  private:
    /** @brief The borrowed one-scalar device destination the barriers spin against and @ref advance bumps. */
    u32r1_span_t device_round_ {};
    /** @brief One step's advance — the arrivals a barrier waits for. */
    u32_t increment_ = 0;
};

/**
 *  @brief For this scope, the calling thread switches to @p stream 's GPU — and switches back at scope
 *      end. The Driver API needs a current context for every allocation, module load, and launch, and
 *      one process drives many GPUs from one thread, so per-device work must enter that device's
 *      context for its duration; the context comes from the object that already owns it (the stream),
 *      never from a separate retain path. Check @ref ok() before relying on the scope.
 */
struct switch_to_gpu_t {
    explicit switch_to_gpu_t(gpu_stream_t &stream) noexcept : ok_(cuCtxPushCurrent(stream.context()) == CUDA_SUCCESS) {}
    switch_to_gpu_t(switch_to_gpu_t const &) = delete;
    switch_to_gpu_t &operator=(switch_to_gpu_t const &) = delete;
    ~switch_to_gpu_t() noexcept {
        if (ok_) cuCtxPopCurrent(nullptr);
    }
    /** @brief Whether the switch happened — a false scope must fail the caller loudly. */
    [[nodiscard]] bool ok() const noexcept { return ok_; }

  private:
    bool ok_ = false;
};

/**
 *  @brief A shared cuDNN session: one handle per device, owned once and borrowed by every fused-graph
 *      consumer — the fused-attention entry point. The graphs themselves live with their consumers
 *      (e.g. the transformer block owns its SDPA plans); this owner carries only the handle, bound to
 *      the caller's stream at execute time.
 */
struct cudnn_session_t {

  private:
    cudnn_session_t() noexcept = default;

  public:
    cudnn_session_t(cudnn_session_t const &) = delete;
    cudnn_session_t &operator=(cudnn_session_t const &) = delete;
    cudnn_session_t(cudnn_session_t &&other) noexcept
        : handle_(std::exchange(other.handle_, nullptr)), algo_pick_(other.algo_pick_) {}
    cudnn_session_t &operator=(cudnn_session_t &&other) noexcept {
        if (this != &other) {
            destroy_();
            handle_ = std::exchange(other.handle_, nullptr);
            algo_pick_ = other.algo_pick_;
        }
        return *this;
    }
    ~cudnn_session_t() noexcept { destroy_(); }

    /** @brief Create the handle — once per device, outside any capture. @p algo_pick carries the run's
     *      algorithm policy; the SDPA plans keep their heuristic engine under both values today, and
     *      @ref algo_pick_t::benchmarked_k will iterate the frontend's engine configs in a later wave. */
    [[nodiscard]] static expected<cudnn_session_t> make(algo_pick_t algo_pick) noexcept {
        cudnn_session_t session;
        session.algo_pick_ = algo_pick;
        if (cudnnStatus_t status = cudnnCreate(&session.handle_); status != CUDNN_STATUS_SUCCESS)
            return expected<cudnn_session_t>::fail(status_t::cuda_error_k, "cuDNN handle creation failed ({})",
                                                   cudnnGetErrorString(status));
        return expected<cudnn_session_t> {std::move(session)};
    }

    /** @brief Bind the handle to @p stream — cuDNN executes on whatever stream the handle carries, so
     *      every consumer re-binds before its enqueue. */
    [[nodiscard]] expected_t bind(gpu_stream_t &stream) noexcept {
        if (cudnnStatus_t status = cudnnSetStream(handle_, reinterpret_cast<cudaStream_t>(stream.get()));
            status != CUDNN_STATUS_SUCCESS)
            return expected_t::fail(status_t::cuda_error_k, "cuDNN stream binding failed ({})",
                                    cudnnGetErrorString(status));
        return {};
    }

    /** @brief The wrapped handle, borrowed by graph builders and executes. */
    [[nodiscard]] cudnnHandle_t get() const noexcept { return handle_; }
    /** @brief The run's algorithm policy — carried for the later engine-config benchmarking wave. */
    [[nodiscard]] algo_pick_t algo_pick() const noexcept { return algo_pick_; }

  private:
    void destroy_() noexcept {
        if (handle_ != nullptr) cudnnDestroy(std::exchange(handle_, nullptr));
    }

    cudnnHandle_t handle_ = nullptr;
    algo_pick_t algo_pick_ = algo_pick_t::heuristic_k;
};

/** @brief The unifying shape across the collective tiers — a typed handle bundling specialized symmetric memory
 *      with an enqueue-only collective op. Every buffer names its element type and is invocable on a stream
 *      (multimem or infiniband); the per-buffer typed memory accessors ( @c payload / @c own_block /
 *      @c send_block / @c recv_block ) expose the specialized memory. Each buffer @c static_assert s it, so a
 *      drifted signature fails at compile time and generic code (mixer, telemetry, tests) can name any collective
 *      uniformly. */
template <typename buffer_type_>
concept collective_buffer = requires(buffer_type_ buffer, gpu_stream_t &stream) {
    typename buffer_type_::value_type;
    { buffer(stream) };
};

} // namespace unum::uform::core
