/**
 *  @file include/uform/utilities/vram_nvidia_schedule.cuh
 *  @author Ash Vardanian
 *  @date July 2, 2026
 *  @brief Execution scheduling: owning CUDA streams and capture-once / replay-forever graphs.
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

#include <utility>     // `std::exchange`, `std::forward`
#include <span>        // `std::span` — the multi-stream capture rank list
#include <type_traits> // `std::type_identity_t`, `std::is_trivially_copyable_v` — kernels take arguments by bytes

#include <cuda.h>      // `CUstream`, `CUgraph`, `CUgraphExec`, `CUevent`
#include <cuda_bf16.h> // `__nv_bfloat16` — the training-currency scalar of the rank-1 aliases

#include <numkong/capabilities.h> // `nk_dots_pack_size_punned_t`, `nk_kernel_cast_punned_t` — the untyped kernel shapes
#include <numkong/tensor.hpp>     // `nk::tensor_view`, `nk::tensor_span` for the typed transfer helpers

#include "uform/utilities/vram_nvidia_device.hpp" // `gpu_arena_alloc`, `bf16_t`, `cuda_failure`
#include "uform/utilities/types.hpp" // `expected`, `expected_t`, `status_t`, `st::basic_vector` for the node-guard walk

namespace unum::uform::utilities {

namespace nk = ashvardanian::numkong;

/** Rank-1 token-id device / host view — the batch's token currency. */
using tokens_view_t = nk::tensor_view<token_id_t, 1>;

/** Rank-1 token-id mutable device / host span — the batch's token currency. */
using tokens_span_t = nk::tensor_span<token_id_t, 1>;

/** Rank-1 token-id device-only buffer owned through a @ref gpu_arena_t — the token tape. */
using tokens_tensor_arena_t = nk::tensor<token_id_t, gpu_arena_alloc<token_id_t>, 1>;

/** One named weight matrix with its gradient — what models publish and optimizers consume.
 *  Vector parameters (norm scales, biases) are @b [1,cols] . The optimizer reads the gradient and
 *  writes the weights, hence the mutability split. The publishing cell's @c fill spells @ref name —
 *  the checkpoint key every SafeTensors parser matches on — since it owns the structure the name
 *  describes; the composition root adds only its @c model. prefix. */
struct differentiable_weights_t {

    /** The owned checkpoint name — short names ride StringZilla's inline buffer. */
    sz::string_t name {};
    nk::tensor_span<bf16_t, 2> weights {};
    nk::tensor_view<bf16_t, 2> weights_grad {};

    /** The same views under a copy of @ref name , the one allocation that can fail. */
    [[nodiscard]] expected<differentiable_weights_t> copy() const noexcept {
        differentiable_weights_t result {{}, weights, weights_grad};
        if (sz::failed(result.name.assign(name.view())))
            return expected<differentiable_weights_t>::fail(status_t::bad_alloc_k, "weight name copy failed: {}",
                                                            name.view());
        return expected<differentiable_weights_t> {std::move(result)};
    }
};

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

    /**
     *  @brief Retain @p device 's primary context and create a stream inside it.
     *  @param[in] device The CUDA device ordinal the stream belongs to.
     *  @param[in] flags @c CU_STREAM_NON_BLOCKING by default — no implicit legacy-stream ordering.
     *  @param[in] priority Stream priority; 0 is the default tier, negative is more urgent.
     */
    [[nodiscard]] static expected<gpu_stream_t> make(int device, unsigned flags = CU_STREAM_NON_BLOCKING,
                                                     int priority = 0) noexcept {
        CUdevice device_handle = 0;
        if (CUresult result = cuDeviceGet(&device_handle, device); result != CUDA_SUCCESS)
            return cuda_failure<gpu_stream_t>(result, "device handle query failed (device={})", device);
        gpu_stream_t stream;
        stream.device_ = device;
        if (CUresult result = cuDevicePrimaryCtxRetain(&stream.context_, device_handle); result != CUDA_SUCCESS)
            return cuda_failure<gpu_stream_t>(result, "primary context retain failed (device={})", device);
        if (CUresult result = cuCtxPushCurrent(stream.context_); result != CUDA_SUCCESS)
            return cuda_failure<gpu_stream_t>(result, "context push failed (device={})", device);
        CUresult const created = cuStreamCreateWithPriority(&stream.stream_, flags, priority);
        cuCtxPopCurrent(nullptr);
        if (created != CUDA_SUCCESS)
            return cuda_failure<gpu_stream_t>(created, "stream creation failed (device={} priority={})", device,
                                              priority);
        return expected<gpu_stream_t> {std::move(stream)};
    }

    /** The wrapped stream handle. */
    [[nodiscard]] CUstream get() const noexcept { return stream_; }

    /** Implicit handle access — the stream passes straight into launchers. */
    operator CUstream() const noexcept { return stream_; }

    /** The same handle as NumKong's opaque stream, for its kernels and runtime calls. */
    [[nodiscard]] nk_stream_t numkong() const noexcept { return reinterpret_cast<nk_stream_t>(stream_); }

    /** The retained primary context the stream lives in. */
    [[nodiscard]] CUcontext context() const noexcept { return context_; }

    /** The device ordinal the stream belongs to. */
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

/** Enqueue a host → device copy of a rank-1 view into a same-typed device span on @p stream — the
 *  typed spelling of @c cuMemcpyHtoDAsync , with the arguments in the order the name reads: host
 *  source, device destination. Extents and element sizes travel with the views, never as
 *  hand-multiplied @c sizeof operands. The device side names the scalar type, so a host `std::span`
 *  converts. An empty @p host_source is a no-op; a source longer than the destination is a loud
 *  failure. */
template <typename value_type_>
[[nodiscard]] inline expected_t copy_host_to_device(std::type_identity_t<nk::tensor_view<value_type_, 1>> host_source,
                                                    nk::tensor_span<value_type_, 1> device_destination,
                                                    gpu_stream_t &stream) noexcept {
    if (host_source.empty()) return {};
    if (device_destination.extent(0) < host_source.extent(0))
        return expected_t::fail(status_t::invalid_argument_k,
                                "device destination ({} scalars) is smaller than the host source ({})",
                                device_destination.extent(0), host_source.extent(0));
    if (CUresult result = cuMemcpyHtoDAsync(device_address(device_destination.data()), host_source.data(),
                                            host_source.extent(0) * sizeof(value_type_), stream.get());
        result != CUDA_SUCCESS)
        return cuda_failure(result, "host-to-device copy failed ({} scalars)", host_source.extent(0));
    return {};
}

/** The mirror of @ref copy_host_to_device : enqueue a device → host copy of a rank-1 view into a
 *  same-typed host span on @p stream — the typed spelling of @c cuMemcpyDtoHAsync , device source
 *  first, host destination second, which converts from a `std::span` as the source names the type.
 *  An empty @p device_source is a no-op; a source longer than the destination is a loud failure.
 *  The host span must stay alive until the stream reaches the copy. */
template <typename value_type_>
[[nodiscard]] inline expected_t copy_device_to_host(
    nk::tensor_view<value_type_, 1> device_source,
    std::type_identity_t<nk::tensor_span<value_type_, 1>> host_destination, gpu_stream_t &stream) noexcept {
    if (device_source.empty()) return {};
    if (host_destination.extent(0) < device_source.extent(0))
        return expected_t::fail(status_t::invalid_argument_k,
                                "host destination ({} scalars) is smaller than the device source ({})",
                                host_destination.extent(0), device_source.extent(0));
    if (CUresult result = cuMemcpyDtoHAsync(host_destination.data(), device_address(device_source.data()),
                                            device_source.extent(0) * sizeof(value_type_), stream.get());
        result != CUDA_SUCCESS)
        return cuda_failure(result, "device-to-host copy failed ({} scalars)", device_source.extent(0));
    return {};
}

/** The device-only sibling of @ref copy_host_to_device : enqueue a device → device copy of a rank-1 view
 *  into a same-typed device span on @p stream , which names the type, so a span source converts. An
 *  empty @p source is a no-op; a source longer than the destination is a loud failure. */
template <typename value_type_>
[[nodiscard]] inline expected_t copy_device_to_device(std::type_identity_t<nk::tensor_view<value_type_, 1>> source,
                                                      nk::tensor_span<value_type_, 1> destination,
                                                      gpu_stream_t &stream) noexcept {
    if (source.empty()) return {};
    if (destination.extent(0) < source.extent(0))
        return expected_t::fail(status_t::invalid_argument_k,
                                "device destination ({} scalars) is smaller than the source ({})",
                                destination.extent(0), source.extent(0));
    if (CUresult result = cuMemcpyDtoDAsync(device_address(destination.data()), device_address(source.data()),
                                            source.extent(0) * sizeof(value_type_), stream.get());
        result != CUDA_SUCCESS)
        return cuda_failure(result, "device-to-device copy failed ({} scalars)", source.extent(0));
    return {};
}

/** Enqueue zeroing every scalar of the contiguous device @p destination on @p stream — the typed spelling of
 *  @c cuMemsetD8Async . A strided @p destination is a loud failure. */
template <typename value_type_, std::size_t rank_>
[[nodiscard]] inline expected_t zero_async(nk::tensor_span<value_type_, rank_> destination,
                                           gpu_stream_t &stream) noexcept {
    if (destination.empty()) return {};
    if (!destination.is_contiguous())
        return expected_t::fail(status_t::invalid_argument_k, "cannot zero a strided device span");
    if (CUresult result = cuMemsetD8Async(device_address(destination.data()), 0,
                                          destination.numel() * sizeof(value_type_), stream.get());
        result != CUDA_SUCCESS)
        return cuda_failure(result, "device zeroing failed ({} scalars)", destination.numel());
    return {};
}

/** The NumKong C ABI spelling of BFloat16 @p scalars — the one place UForm's CUDA scalar becomes NumKong's. */
[[nodiscard]] inline nk_bf16_t const *numkong_scalars(bf16_t const *scalars) noexcept {
    return reinterpret_cast<nk_bf16_t const *>(scalars);
}

/** The mutable twin of the BFloat16 @ref numkong_scalars . */
[[nodiscard]] inline nk_bf16_t *numkong_scalars(bf16_t *scalars) noexcept {
    return reinterpret_cast<nk_bf16_t *>(scalars);
}

/** The NumKong C ABI spelling of E4M3 @p scalars . */
[[nodiscard]] inline nk_e4m3_t const *numkong_scalars(e4m3_t const *scalars) noexcept {
    return reinterpret_cast<nk_e4m3_t const *>(scalars);
}

#pragma region Numerics

/** NumKong's per-capability kernel shapes the training engine calls, where they take typed
 *  scalars. */
using dots_pack_bf16_kernel_t = nk_status_t (*)(nk_bf16_t const *b, nk_size_t columns, nk_size_t depth,
                                                nk_size_t b_stride, void *b_packed, nk_size_t columns_begin,
                                                nk_size_t columns_end, nk_stream_t stream);
using dots_pack_e4m3_kernel_t = nk_status_t (*)(nk_e4m3_t const *b, nk_size_t columns, nk_size_t depth,
                                                nk_size_t b_stride, void *b_packed, nk_size_t columns_begin,
                                                nk_size_t columns_end, nk_stream_t stream);
using dots_packed_bf16_kernel_t = nk_status_t (*)(nk_bf16_t const *a, void const *b_packed, nk_f32_t *c, nk_size_t rows,
                                                  nk_size_t columns, nk_size_t depth, nk_size_t a_stride,
                                                  nk_size_t c_stride, void *workspace, nk_size_t workspace_bytes,
                                                  nk_stream_t stream);
using dots_symmetric_bf16_kernel_t = nk_status_t (*)(nk_bf16_t const *vectors, nk_size_t vector_count, nk_size_t depth,
                                                     nk_size_t stride, nk_f32_t *result, nk_size_t result_stride,
                                                     nk_size_t rows_begin, nk_size_t rows_end, void *workspace,
                                                     nk_size_t workspace_bytes, nk_stream_t stream);
using dots_packed_e4m3_kernel_t = nk_status_t (*)(nk_e4m3_t const *a, void const *b_packed, nk_f32_t *c, nk_size_t rows,
                                                  nk_size_t columns, nk_size_t depth, nk_size_t a_stride,
                                                  nk_size_t c_stride, void *workspace, nk_size_t workspace_bytes,
                                                  nk_stream_t stream);
using attention_pack_bf16_kernel_t = nk_status_t (*)(nk_size_t key_value_head_count, nk_size_t depth,
                                                     nk_u32_t const *key_offsets, nk_u32_t const *key_lengths,
                                                     nk_size_t segment_count, nk_bf16_t const *keys,
                                                     nk_size_t key_stride, nk_bf16_t const *values,
                                                     nk_size_t value_stride, void *key_value_packed,
                                                     nk_size_t tasks_begin, nk_size_t tasks_end, nk_stream_t stream);
using attention_pack_e4m3_kernel_t = nk_status_t (*)(nk_size_t key_value_head_count, nk_size_t depth,
                                                     nk_u32_t const *key_offsets, nk_u32_t const *key_lengths,
                                                     nk_size_t segment_count, nk_e4m3_t const *keys,
                                                     nk_size_t key_stride, nk_e4m3_t const *values,
                                                     nk_size_t value_stride, void *key_value_packed,
                                                     nk_size_t tasks_begin, nk_size_t tasks_end, nk_stream_t stream);
using attention_packed_bf16_kernel_t = nk_status_t (*)(
    nk_size_t head_count, nk_size_t key_value_head_count, nk_size_t depth, nk_u32_t const *query_offsets,
    nk_size_t query_token_count, nk_f32_t scale, nk_size_t keys_before, nk_size_t keys_after, nk_bf16_t const *queries,
    nk_size_t query_stride, void const *key_value_packed, nk_f32_t *output, nk_size_t output_stride,
    nk_f32_t *log_sum_exp, nk_size_t tasks_begin, nk_size_t tasks_end, void *workspace, nk_size_t workspace_bytes,
    nk_stream_t stream);
using attention_packed_e4m3_kernel_t = nk_status_t (*)(
    nk_size_t head_count, nk_size_t key_value_head_count, nk_size_t depth, nk_u32_t const *query_offsets,
    nk_size_t query_token_count, nk_f32_t scale, nk_size_t keys_before, nk_size_t keys_after, nk_e4m3_t const *queries,
    nk_size_t query_stride, void const *key_value_packed, nk_f32_t *output, nk_size_t output_stride,
    nk_f32_t *log_sum_exp, nk_size_t tasks_begin, nk_size_t tasks_end, void *workspace, nk_size_t workspace_bytes,
    nk_stream_t stream);
using attention_packed_gradients_bf16_kernel_t = nk_status_t (*)(
    nk_size_t head_count, nk_size_t key_value_head_count, nk_size_t depth, nk_u32_t const *query_offsets,
    nk_size_t query_token_count, nk_f32_t scale, nk_size_t keys_before, nk_size_t keys_after, nk_bf16_t const *queries,
    nk_size_t query_stride, void const *key_value_packed, nk_f32_t const *output, nk_f32_t const *output_gradient,
    nk_size_t output_stride, nk_f32_t const *log_sum_exp, nk_f32_t *query_gradient, nk_size_t query_gradient_stride,
    nk_f32_t *key_gradient, nk_f32_t *value_gradient, nk_size_t key_value_gradient_stride, nk_size_t tasks_begin,
    nk_size_t tasks_end, void *workspace, nk_size_t workspace_bytes, nk_stream_t stream);
using attention_rope_bf16_kernel_t = nk_status_t (*)(nk_bf16_t const *x, nk_f32_t const *cos, nk_f32_t const *sin,
                                                     nk_bf16_t *y, nk_size_t rows, nk_size_t head_count,
                                                     nk_size_t depth, nk_size_t x_stride, nk_size_t y_stride,
                                                     nk_stream_t stream);
using each_sum_f32_kernel_t = nk_status_t (*)(nk_f32_t const *a, nk_f32_t const *b, nk_size_t count, nk_f32_t *result,
                                              nk_stream_t stream);
using each_rmsnorm_bf16_kernel_t = nk_status_t (*)(nk_bf16_t const *x, nk_f32_t const *gamma, nk_bf16_t *y,
                                                   nk_size_t rows, nk_size_t groups, nk_size_t columns,
                                                   nk_size_t x_stride, nk_size_t y_stride, nk_f32_t epsilon,
                                                   nk_stream_t stream);
using each_rmscast_bf16_kernel_t = nk_status_t (*)(nk_f32_t const *x, nk_f32_t const *gamma, nk_bf16_t *y,
                                                   nk_size_t rows, nk_size_t groups, nk_size_t columns,
                                                   nk_size_t x_stride, nk_size_t y_stride, nk_f32_t epsilon,
                                                   nk_stream_t stream);
using each_swiglu_f32_kernel_t = nk_status_t (*)(nk_f32_t const *gate, nk_f32_t const *up, nk_f32_t *y, nk_size_t rows,
                                                 nk_size_t columns, nk_size_t gate_stride, nk_size_t up_stride,
                                                 nk_size_t y_stride, nk_f32_t gate_scale, nk_f32_t output_scale,
                                                 nk_stream_t stream);
using each_swiglu_bf16_kernel_t = nk_status_t (*)(nk_bf16_t const *gate, nk_bf16_t const *up, nk_bf16_t *y,
                                                  nk_size_t rows, nk_size_t columns, nk_size_t gate_stride,
                                                  nk_size_t up_stride, nk_size_t y_stride, nk_f32_t gate_scale,
                                                  nk_f32_t output_scale, nk_stream_t stream);

/** The NumKong kernels the training engine calls, each the GPU class's own; @c src/ fills it. */
struct gpu_numerics_t {
    nk_dots_pack_size_punned_t pack_size_bf16 = nullptr;
    dots_pack_bf16_kernel_t pack_bf16 = nullptr;
    dots_packed_bf16_kernel_t multiply_bf16 = nullptr;
    nk_dots_packed_workspace_size_punned_t multiply_workspace_size_bf16 = nullptr;
    dots_symmetric_bf16_kernel_t symmetric_bf16 = nullptr;
    nk_dots_pack_size_punned_t pack_size_e4m3 = nullptr;
    dots_pack_e4m3_kernel_t pack_e4m3 = nullptr;
    dots_packed_e4m3_kernel_t multiply_e4m3 = nullptr;
    nk_attention_pack_size_punned_t attention_pack_size_bf16 = nullptr;
    attention_pack_bf16_kernel_t attention_pack_bf16 = nullptr;
    attention_packed_bf16_kernel_t attend_bf16 = nullptr;
    attention_packed_gradients_bf16_kernel_t attend_gradients_bf16 = nullptr;
    nk_attention_pack_size_punned_t attention_pack_size_e4m3 = nullptr;
    attention_pack_e4m3_kernel_t attention_pack_e4m3 = nullptr;
    attention_packed_e4m3_kernel_t attend_e4m3 = nullptr;
    attention_rope_bf16_kernel_t rope_bf16 = nullptr;
    nk_kernel_cast_punned_t cast = nullptr;
    each_sum_f32_kernel_t sum_f32 = nullptr;
    each_rmsnorm_bf16_kernel_t rmsnorm_bf16 = nullptr;
    each_rmscast_bf16_kernel_t rmscast_bf16 = nullptr;
    each_swiglu_f32_kernel_t swiglu_f32 = nullptr;
    each_swiglu_bf16_kernel_t swiglu_bf16 = nullptr;
};

#pragma endregion Numerics

#if defined(__CUDACC__)

/** Grid cap of every grid-strided launch: a bandwidth-bound kernel gains nothing past this, and
 *  an unbounded grid stalls capture-time instantiation on huge shards. */
inline constexpr unsigned kernel_grid_cap_k = 4096u;

/** The one grid-size formula for grid-strided kernels over @p scalars elements: at least one
 *  CTA, at most @ref kernel_grid_cap_k . */
[[nodiscard]] inline constexpr unsigned grid_size_for(std::size_t scalars,
                                                      unsigned block = kernel_block_size_default_k) noexcept {
    std::size_t const grid = divide_round_up(scalars, block);
    if (grid == 0) return 1u;
    return grid > kernel_grid_cap_k ? kernel_grid_cap_k : static_cast<unsigned>(grid);
}

/** Launch @p function , a kernel resolved from a loaded module or by @ref launch_kernel , as one
 *  checked @c cuLaunchKernel ; each argument's type must match its parameter's exactly, as the
 *  driver copies raw bytes. */
template <typename... arg_types_>
[[nodiscard]] inline expected_t launch_function(CUfunction function, dim3 grid_size, dim3 block_size,
                                                gpu_stream_t &stream, arg_types_... args) noexcept {
    static_assert(sizeof...(arg_types_) != 0, "a kernel without arguments has nothing to compute");
    static_assert((std::is_trivially_copyable_v<arg_types_> && ...), "kernels take their arguments by bytes");
    if (function == nullptr) return expected_t::fail(status_t::invalid_argument_k, "no kernel to launch");
    void *argument_addresses[] = {&args...};
    if (CUresult result = cuLaunchKernel(function, grid_size.x, grid_size.y, grid_size.z, block_size.x, block_size.y,
                                         block_size.z, 0, stream.get(), argument_addresses, nullptr);
        result != CUDA_SUCCESS)
        return cuda_failure(result, "kernel launch failed (grid=[{},{},{}] block=[{},{},{}])", grid_size.x, grid_size.y,
                            grid_size.z, block_size.x, block_size.y, block_size.z);
    return {};
}

/**
 *  @brief Launch a compiled-in kernel through the Driver API — the checked replacement for the
 *      triple-chevron syntax, which returns no error code, fails silently on impossible configurations,
 *      and drags a hidden CUDA Runtime dependency behind every launch.
 *
 *  The case against the triple-chevron syntax: https://ashvardanian.com/posts/less-wrong-cuda-hello-world/
 *
 *  The kernel's entry address resolves to a @c CUfunction through @c cudaGetFuncBySymbol — the one
 *  sanctioned interop symbol, declared by nvcc's own implicit headers and serviced by the fatbin
 *  registration stub nvcc statically embeds into every kernel-carrying binary regardless — and the
 *  launch itself is one checked @c cuLaunchKernel . No dynamic @c libcudart enters the link.
 *
 *  Arguments bind by const reference (temporaries included — the driver copies their bytes out
 *  before returning), and every argument's type must match the kernel's parameter type @b exactly:
 *  the driver copies raw bytes, so the implicit conversions a triple-chevron call would perform do
 *  not apply — convert spans to views at the call site. @ref launch_function launches what a loaded
 *  module resolved.
 */
template <typename... arg_types_>
[[nodiscard]] inline expected_t launch_kernel(void const *kernel_symbol, dim3 grid_size, dim3 block_size,
                                              gpu_stream_t &stream, arg_types_ const &...args) noexcept {
    CUfunction function = nullptr;
    if (int result = static_cast<int>(cudaGetFuncBySymbol(&function, kernel_symbol)); result != 0)
        return expected_t::fail(status_t::device_error_k, "kernel symbol resolution failed (code={})", result);
    return launch_function(function, grid_size, block_size, stream, args...);
}

/** The typed spelling of @ref launch_kernel : takes the @c __global__ function itself, so every argument
 *  converts to its parameter's exact type before the driver copies its bytes — spans become views, and an
 *  argument no parameter accepts fails to compile. */
template <typename... parameter_types_>
[[nodiscard]] inline expected_t launch_kernel(void (*kernel)(parameter_types_...), dim3 grid_size, dim3 block_size,
                                              gpu_stream_t &stream,
                                              std::type_identity_t<parameter_types_> const &...arguments) noexcept {
    return launch_kernel(reinterpret_cast<void const *>(kernel), grid_size, block_size, stream, arguments...);
}

/**
 *  @brief The BFloat16 matrix transpose @c destination=sourceᵀ , tiled 64x64 through shared memory with 16-byte
 *      global accesses on both sides; either operand may be a strided view into a wider buffer.
 *
 *  Each thread loads two neighbouring source rows, so the pair packs into 32-bit words that make the transposed
 *  rows contiguous in shared memory. Edge tiles are masked, and operands whose pointers or row strides are not
 *  16-byte multiples fall back to element accesses.
 *
 *  Launches a @c [ceil(columns/64),ceil(rows/64),1] grid of @c [256,1,1] blocks with 8.4 KB of static shared memory.
 */
__global__ void transpose_bf16_kernel(bf16r2_view_t source, bf16r2_span_t destination);

/** Enqueue @p destination = @p source ᵀ for BFloat16 matrices: one capture-safe launch of
 *  @ref transpose_bf16_kernel on @p stream . */
[[nodiscard]] inline expected_t transpose_bf16(bf16r2_view_t source, bf16r2_span_t destination,
                                               gpu_stream_t &stream) noexcept {
    if (destination.extent(0) != source.extent(1) || destination.extent(1) != source.extent(0))
        return expected_t::fail(status_t::unsupported_k, "[{}x{}] cannot hold the transposed [{}x{}]",
                                destination.extent(0), destination.extent(1), source.extent(0), source.extent(1));
    if (source.extent(0) == 0 || source.extent(1) == 0) return {};
    dim3 const grid {static_cast<u32_t>(divide_round_up(source.extent(1), std::size_t {64})),
                     static_cast<u32_t>(divide_round_up(source.extent(0), std::size_t {64})), 1u};
    return launch_kernel(&transpose_bf16_kernel, grid, dim3 {256u, 1u, 1u}, stream, source, destination);
}

/** Lift a NumKong status into the cell's diagnostics, naming the @p step that returned it. */
[[nodiscard]] inline expected_t numkong_result(nk_status_t status, sz::string_view_t step) noexcept {
    if (status == nk_success_k) return {};
    return expected_t::fail(status_t::device_error_k, "{} failed: {}", step, nk_status_name(status));
}

/** The bytes the pack of a @b [columns,depth] BFloat16 B operand takes under @p numerics . */
[[nodiscard]] expected<std::size_t> pack_bytes_bf16(std::size_t columns, std::size_t depth,
                                                    gpu_numerics_t const &numerics) noexcept;

/** Allocate off @p arena a buffer for the NumKong pack of a @b [columns,depth] BFloat16 B operand. */
[[nodiscard]] expected<b8r1_tensor_arena_t> allocate_pack_bf16(std::size_t columns, std::size_t depth,
                                                               gpu_numerics_t const &numerics,
                                                               gpu_arena_t &arena) noexcept;

/** Enqueue NumKong's pack of @p b , read row-major as the @b [columns,depth] B operand of
 *  @ref multiply_bf16 , into @p packed , sized by @ref pack_bytes_bf16 . */
[[nodiscard]] expected_t pack_bf16(bf16r2_view_t b, b8r1_span_t packed, gpu_numerics_t const &numerics,
                                   gpu_stream_t &stream) noexcept;

/** Enqueue the Float32 product @c c=a·Bᵀ against @p packed_b , the pack of a @b [columns,depth] B with
 *  @c columns=c.extent(1) and @c depth=a.extent(1) . NumKong products replay bit-identically. */
[[nodiscard]] expected_t multiply_bf16(bf16r2_view_t a, b8r1_view_t packed_b, f32r2_span_t c,
                                       gpu_numerics_t const &numerics, gpu_stream_t &stream) noexcept;

/**
 *  @brief Pack @p weight @b [outputs,inputs] as the B operand of its forward product into @p packed , and its
 *      transpose, staged in @p transposed with rows padded to 8 for NumKong's 16-byte stride rule, as that of its
 *      data gradient into @p packed_transposed . Call once per optimizer step, after the weights change.
 *  @param[in] transposed At least @c inputs·round_up(outputs,8) scalars of scratch.
 */
[[nodiscard]] expected_t pack_weight_bf16(bf16r2_view_t weight, bf16r1_span_t transposed, b8r1_span_t packed,
                                          b8r1_span_t packed_transposed, gpu_numerics_t const &numerics,
                                          gpu_stream_t &stream) noexcept;

/** The workspace bytes @ref weight_gradient_bf16 takes for a @b [outputs,inputs] weight over
 *  @p tokens under @p numerics , where NumKong splits the token depth into slices it sums in a
 *  fixed order. */
[[nodiscard]] expected<std::size_t> weight_gradient_workspace_bytes_bf16(std::size_t outputs, std::size_t inputs,
                                                                         std::size_t tokens,
                                                                         gpu_numerics_t const &numerics) noexcept;

/**
 *  @brief One weight gradient @c dW=dYᵀ·X : transposes @p output_grad and @p input with token rows padded to 8,
 *      since NumKong reads A row-major and packs B row-major, multiplies @c dYᵀ by the packed @c Xᵀ , and narrows
 *      the Float32 product into @p weight_grad .
 *  @param[in] output_grad @b [tokens,outputs] the product's BFloat16 output gradient.
 *  @param[in] input @b [tokens,inputs] the product's forward A operand.
 *  @param[out] weight_grad @b [outputs,inputs] the contiguous BFloat16 weight gradient.
 *  @param[in] transposed_grad,transposed_input At least @c outputs·round_up(tokens,8) and
 *      @c inputs·round_up(tokens,8) scalars of scratch.
 *  @param[in] packed_input At least @ref pack_bytes_bf16 of @c [inputs,tokens] .
 *  @param[in] product At least @c outputs·inputs Float32 scalars of scratch.
 *  @param[in] workspace At least @ref weight_gradient_workspace_bytes_bf16 , or empty for the
 *      unsplit product.
 *  @param[inout] stream The stream every launch enqueues on.
 */
[[nodiscard]] expected_t weight_gradient_bf16(bf16r2_view_t output_grad, bf16r2_view_t input, bf16r2_span_t weight_grad,
                                              bf16r1_span_t transposed_grad, bf16r1_span_t transposed_input,
                                              b8r1_span_t packed_input, f32r1_span_t product, b8r1_span_t workspace,
                                              gpu_numerics_t const &numerics, gpu_stream_t &stream) noexcept;

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

/** All-threads-get-the-total CTA sum: per-warp shuffle trees, lane-0 partials staged through
 *  static shared memory, one final warp tree, one broadcast cell — fixed-order, deterministic,
 *  sized for @p block_size_ threads. */
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

/** The device-wide nanosecond clock every spin-wait measures its deadline against. */
[[nodiscard]] __device__ inline u64_t global_nanoseconds() noexcept {
    u64_t value;
    asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(value));
    return value;
}

#endif // defined(__CUDACC__)

/** How a §10 model region runs: captured once into a @ref gpu_replay_graph_t and replayed every step
 *  (the production default), or re-enqueued eagerly each step — the bring-up / debugging path, and the
 *  other half of every capture-vs-eager parity test. A trainer carries it as a run knob; the two modes must
 *  land on bitwise-identical weights over identical seeds. */
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
            return expected<gpu_replay_graph_t>::fail(status_t::invalid_argument_k,
                                                      "capture needs a real stream, not the null stream");
        if (CUresult result = cuStreamBeginCapture(stream, CU_STREAM_CAPTURE_MODE_RELAXED); result != CUDA_SUCCESS)
            return cuda_failure<gpu_replay_graph_t>(result, "stream capture begin failed");
        expected_t emitted = body();
        CUgraph graph = nullptr;
        CUresult const ended = cuStreamEndCapture(stream, &graph);
        if (!emitted) {
            if (graph != nullptr) cuGraphDestroy(graph);
            return expected<gpu_replay_graph_t>::fail(std::move(emitted));
        }
        if (ended != CUDA_SUCCESS || graph == nullptr) {
            if (graph != nullptr) cuGraphDestroy(graph);
            return cuda_failure<gpu_replay_graph_t>(ended, "stream capture end failed");
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
            return expected<gpu_replay_graph_t>::fail(status_t::invalid_argument_k,
                                                      "capture needs at least one stream");
        for (std::size_t rank = 0; rank != streams.size(); ++rank)
            if (streams[rank] == nullptr)
                return expected<gpu_replay_graph_t>::fail(status_t::invalid_argument_k,
                                                          "capture needs real streams (rank {} is null)", rank);
        if (streams.size() == 1) return make(streams.front(), [&body]() noexcept { return body(std::size_t {0}); });

        CUstream const trunk = streams.front();
        CUevent fork_event = nullptr;
        st::basic_vector<CUevent> join_events;
        if (st::failed(join_events.resize(streams.size() - 1u)))
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
            if (!capturing) outcome = cuda_failure(begun, "stream capture begin failed (trunk)");
        }
        if (outcome)
            if (CUresult result = cuEventRecord(fork_event, trunk); result != CUDA_SUCCESS)
                outcome = cuda_failure(result, "fork-event record failed");
        if (outcome) outcome = body(std::size_t {0});
        for (std::size_t branch = 0; outcome && branch != join_events.size(); ++branch) {
            CUstream const branch_stream = streams[branch + 1u];
            if (CUresult result = cuStreamWaitEvent(branch_stream, fork_event, 0); result != CUDA_SUCCESS) {
                outcome = cuda_failure(result, "fork-event wait failed (rank={})", branch + 1u);
                break;
            }
            outcome = body(branch + 1u);
            if (!outcome) break;
            if (CUresult result = cuEventRecord(join_events[branch], branch_stream); result != CUDA_SUCCESS)
                outcome = cuda_failure(result, "join-event record failed (rank={})", branch + 1u);
        }
        for (std::size_t branch = 0; outcome && branch != join_events.size(); ++branch)
            if (CUresult result = cuStreamWaitEvent(trunk, join_events[branch], 0); result != CUDA_SUCCESS)
                outcome = cuda_failure(result, "join-event wait failed (rank={})", branch + 1u);

        if (capturing) {
            CUresult const ended = cuStreamEndCapture(trunk, &graph);
            if (outcome && (ended != CUDA_SUCCESS || graph == nullptr))
                outcome = cuda_failure(ended, "stream capture end failed (trunk)");
        }
        (void)destroy_event_(fork_event);
        for (CUevent &event : join_events) (void)destroy_event_(event);
        if (!outcome) {
            if (graph != nullptr) cuGraphDestroy(graph);
            return expected<gpu_replay_graph_t>::fail(std::move(outcome));
        }
        return adopt_(graph, trunk);
    }

    /** Replay the executable on @p stream : one @c cuGraphLaunch — staging already happened at
     *  @ref make . */
    [[nodiscard]] expected_t launch(CUstream stream) const noexcept {
        if (exec_ == nullptr)
            return expected_t::fail(status_t::invalid_argument_k, "launching an empty (default or moved-from) graph");
        if (CUresult result = cuGraphLaunch(exec_, stream); result != CUDA_SUCCESS)
            return cuda_failure(result, "graph launch failed");
        return {};
    }

  private:

    /** Enforce the no-allocation contract, instantiate @p graph , and upload the executable on
     *  @p stream — after this the object is fully replay-ready. */
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
            return cuda_failure<gpu_replay_graph_t>(instantiated, "graph instantiation failed");
        }
        if (CUresult result = cuGraphUpload(replay.exec_, stream); result != CUDA_SUCCESS)
            return cuda_failure<gpu_replay_graph_t>(result, "graph upload failed");
        return expected<gpu_replay_graph_t> {std::move(replay)};
    }

    /** The no-allocation contract, enforced: any allocation / free node in the captured graph is a
     *  loud failure — it escapes the arena budget and deadlocks device-local multi-GPU first launch. */
    [[nodiscard]] static expected_t guard_alloc_nodes_(CUgraph graph) noexcept {
        std::size_t node_count = 0;
        if (CUresult result = cuGraphGetNodes(graph, nullptr, &node_count); result != CUDA_SUCCESS)
            return cuda_failure(result, "graph node query failed");
        if (node_count == 0) return {};
        st::basic_vector<CUgraphNode> nodes;
        if (st::failed(nodes.resize(st::assume_uninitialized, node_count)))
            return expected_t::fail(status_t::bad_alloc_k, "graph node table allocation failed (nodes={})", node_count);
        if (CUresult result = cuGraphGetNodes(graph, nodes.data(), &node_count); result != CUDA_SUCCESS)
            return cuda_failure(result, "graph node listing failed");
        std::size_t allocation_nodes = 0;
        for (CUgraphNode node : nodes) {
            CUgraphNodeType node_type;
            if (CUresult result = cuGraphNodeGetType(node, &node_type); result != CUDA_SUCCESS)
                return cuda_failure(result, "graph node type query failed");
            allocation_nodes += node_type == CU_GRAPH_NODE_TYPE_MEM_ALLOC || node_type == CU_GRAPH_NODE_TYPE_MEM_FREE;
        }
        if (allocation_nodes != 0)
            return expected_t::fail(                                                                          //
                status_t::invalid_argument_k,                                                                 //
                "the captured body allocates: {} alloc/free nodes among {} — an allocating body escapes the " //
                "arena budget and deadlocks device-local multi-GPU first launch; stage buffers before capture",
                allocation_nodes, node_count);
        return {};
    }

    /** Create a timing-free event in @p stream 's own context, so recording it there is legal. */
    [[nodiscard]] static expected_t make_event_for_(CUstream stream, CUevent &event) noexcept {
        CUcontext stream_context = nullptr;
        if (CUresult result = cuStreamGetCtx(stream, &stream_context); result != CUDA_SUCCESS)
            return cuda_failure(result, "stream context query failed");
        if (CUresult result = cuCtxPushCurrent(stream_context); result != CUDA_SUCCESS)
            return cuda_failure(result, "context push failed");
        CUresult const created = cuEventCreate(&event, CU_EVENT_DISABLE_TIMING);
        cuCtxPopCurrent(nullptr);
        if (created != CUDA_SUCCESS) return cuda_failure(created, "event creation failed");
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

/** A move-only owning wrapper over a @c CUevent — the cross-stream ordering token (fork / join
 *  boundaries, gather-ready signals) with the timing machinery disabled. */
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

    /** Create the event — once, at setup. */
    [[nodiscard]] static expected<gpu_event_t> make() noexcept {
        gpu_event_t event;
        if (CUresult result = cuEventCreate(&event.event_, CU_EVENT_DISABLE_TIMING); result != CUDA_SUCCESS)
            return cuda_failure<gpu_event_t>(result, "event creation failed");
        return expected<gpu_event_t> {std::move(event)};
    }

    /** The wrapped handle for record / wait calls. */
    [[nodiscard]] CUevent get() const noexcept { return event_; }

  private:
    CUevent event_ = nullptr;
};

/**
 *  @brief The monotonic step counter of one collective surface — the device round scalar @b is
 *      the count, and it is the sole source of truth (no host shadow, no pinned cell).
 *
 *  One behavior: @ref advance enqueues a one-thread device increment, @c round += increment followed by a
 *  @c __threadfence_system() , once per step on the surface's lead stream before any barrier.
 *  Because the bump lives on the device, a captured collective region self-advances the round
 *  on every @c cuGraphLaunch replay with no host intervention. The scalar is seeded @c 0u-increment
 *  at @ref make so the first advance yields round 0. Counts never reset; a replayed barrier's
 *  target only grows.
 */
struct barrier_round_t {

    barrier_round_t() noexcept = default;
    barrier_round_t(barrier_round_t const &) = delete;
    barrier_round_t &operator=(barrier_round_t const &) = delete;
    barrier_round_t(barrier_round_t &&) noexcept = default;
    barrier_round_t &operator=(barrier_round_t &&) noexcept = default;

    /**
     *  @brief Seed the device round scalar — once, at setup, with a current context.
     *  @param[in] device_round The one-scalar device destination, borrowed for this object's lifetime.
     *  @param[in] increment How far one step advances the count (a member count, a round count).
     *  @param[in] stream The lead stream the seed is enqueued on, ordered before the first @ref advance.
     */
    [[nodiscard]] static expected<barrier_round_t> make(u32r1_span_t device_round, u32_t increment,
                                                        gpu_stream_t &stream) noexcept {
        barrier_round_t counter;
        counter.device_round_ = device_round;
        counter.increment_ = increment;
        // Seed one increment below zero so the first advance yields round 0 (device is the sole source of truth).
        if (CUresult result = cuMemsetD32Async(device_address(device_round.data()), 0u - increment, 1, stream.get());
            result != CUDA_SUCCESS)
            return cuda_failure<barrier_round_t>(result, "round seed failed");
        return expected<barrier_round_t> {std::move(counter)};
    }

    /** Enqueue the one-thread device increment — once per step, on the surface's lead stream,
     *  before any barrier. Capture-safe: a captured region re-runs the bump on every replay. */
    [[nodiscard]] expected_t advance(gpu_stream_t &stream) noexcept;

  private:

    /** The borrowed one-scalar device destination the barriers spin against and @ref advance bumps. */
    u32r1_span_t device_round_ {};

    /** One step's advance — the arrivals a barrier waits for. */
    u32_t increment_ = 0;
};

/**
 *  @brief For this scope, the calling thread switches to @p stream 's GPU — and switches back at scope
 *      end.
 *
 *  The Driver API needs a current context for every allocation, module load, and launch, and
 *  one process drives many GPUs from one thread, so per-device work must enter that device's
 *  context for its duration; the context comes from the object that already owns it (the stream),
 *  never from a separate retain path. Check @ref ok() before relying on the scope.
 */
struct switch_to_gpu_t {
    explicit switch_to_gpu_t(gpu_stream_t &stream) noexcept : ok_(cuCtxPushCurrent(stream.context()) == CUDA_SUCCESS) {}
    switch_to_gpu_t(switch_to_gpu_t const &) = delete;
    switch_to_gpu_t &operator=(switch_to_gpu_t const &) = delete;
    ~switch_to_gpu_t() noexcept {
        if (ok_) cuCtxPopCurrent(nullptr);
    }

    /** Whether the switch happened — a false scope must fail the caller loudly. */
    [[nodiscard]] bool ok() const noexcept { return ok_; }

  private:
    bool ok_ = false;
};

/** The unifying shape across the collective tiers — a typed handle bundling specialized symmetric memory
 *  with an enqueue-only collective op. Every buffer names its element type and is invocable on a stream
 *  (multimem or infiniband); the per-buffer typed memory accessors ( @c payload / @c own_block /
 *  @c send_block / @c recv_block ) expose the specialized memory. Each buffer @c static_assert s it, so a
 *  drifted signature fails at compile time and generic code (mixer, telemetry, tests) can name any collective
 *  uniformly. */
template <typename buffer_type_>
concept collective_buffer = requires(buffer_type_ buffer, gpu_stream_t &stream) {
    typename buffer_type_::value_type;
    { buffer(stream) };
};

} // namespace unum::uform::utilities
