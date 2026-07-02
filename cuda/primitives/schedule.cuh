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
 *  (learnable scalars, staged relation slabs, epoch counters) — never through re-captured code. The
 *  contract is enforced, not trusted: after capture the graph's nodes are walked, and any
 *  allocation / free node is a @b loud failure — an allocating body escapes every arena budget and
 *  deadlocks device-local multi-GPU worlds at first launch, and there is no correct fallback.
 *
 *  Construction absorbs the whole setup pipeline — capture, instantiate, guard, @b upload — so a
 *  @ref gpu_replay_graph_t that exists is always launchable, with device staging already done. Uploading
 *  eagerly matters: lazily staged MMIO nodes are a measured first-launch deadlock on device-initiated
 *  transports.
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
#include <span>    // `std::span` — the multi-stream capture rank list
#include <utility> // `std::exchange`, `std::forward`

#include <cuda.h> // `CUstream`, `CUgraph`, `CUgraphExec`, `CUevent`

#include <numkong/tensor.hpp> // `nk::tensor_view`, `nk::tensor_span` for the typed transfer helpers

#include "types.cuh"      // `expected`, `expected_t`, `status_t`
#include "array.cuh"      // `array` for the node-guard walk
#include "allocators.cuh" // `gpu_arena_alloc` backing the owning rank-1 tensor aliases

namespace unum::uform::primitives {

namespace nk = ashvardanian::numkong;

/** @brief Rank-1 UInt32 device / host view. */
using u32r1_view_t = nk::tensor_view<u32_t, 1>;
/** @brief Rank-1 Float32 device / host view. */
using f32r1_view_t = nk::tensor_view<f32_t, 1>;
/** @brief Rank-1 UInt32 mutable device / host span. */
using u32r1_span_t = nk::tensor_span<u32_t, 1>;
/** @brief Rank-1 Float32 mutable device / host span. */
using f32r1_span_t = nk::tensor_span<f32_t, 1>;
/** @brief Rank-1 UInt32 device-only buffer owned through a @ref gpu_arena_t . */
using u32r1_tensor_arena_t = nk::tensor<u32_t, gpu_arena_alloc<u32_t>, 1>;
/** @brief Rank-1 Float32 device-only buffer owned through a @ref gpu_arena_t . */
using f32r1_tensor_arena_t = nk::tensor<f32_t, gpu_arena_alloc<f32_t>, 1>;

/**
 *  @brief A move-only owning wrapper over a real (non-null) @c CUstream on one device.
 *
 *  Creation retains the device's primary context and creates the stream inside it, so a caller never
 *  juggles contexts to use the stream; destruction destroys the stream and releases the retained context.
 *  Non-blocking by default: an explicit stream never implicitly synchronizes with the legacy stream.
 */
struct gpu_stream_t {

    gpu_stream_t() noexcept = default;
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
                                                    gpu_stream_t const &stream) noexcept {
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
                                                    gpu_stream_t const &stream) noexcept {
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

} // namespace unum::uform::primitives
