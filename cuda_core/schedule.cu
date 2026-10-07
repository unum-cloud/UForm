/**
 *  @brief The device-side kernels declared in @ref schedule.cuh : the barrier-round increment that lets a
 *      captured collective region self-advance its round, and the BFloat16 transpose.
 *  @file cuda_core/schedule.cu
 *  @author Ash Vardanian
 *  @date July 5, 2026
 */
#include <cstdint> // `std::uintptr_t`

#include <cuda.h> // `CUdeviceptr`, driver launch surface

#include "cuda_core/schedule.cuh" // `barrier_round_t`, `launch_kernel`, `u32r1_span_t`

using namespace unum::uform::core;

/**
 *  @brief One thread advances the round scalar and publishes it system-wide, so a captured replay
 *      re-runs the bump on every launch instead of freezing at the capture-time value.
 *
 *  __grid_size__ @c [1,1,1] , one thread; no dynamic shared memory.
 */
static __global__ void barrier_round_increment_kernel_(u32r1_span_t round, u32_t increment) {
    if (threadIdx.x == 0u) {
        round[0] += increment;
        __threadfence_system();
    }
}

expected_t barrier_round_t::advance(gpu_stream_t &stream) noexcept {
    return launch_kernel(reinterpret_cast<void const *>(&barrier_round_increment_kernel_), dim3 {1, 1, 1},
                         dim3 {1, 1, 1}, stream, device_round_, increment_);
}

/** @brief Loads the eight elements at @p column of @p row as four packed words; out-of-range cells read zero. */
[[nodiscard]] static __device__ inline uint4 transpose_load_(bf16_t const *row_start, std::size_t column,
                                                             std::size_t columns, bool aligned) {
    if (aligned && column + 8u <= columns) return *reinterpret_cast<uint4 const *>(row_start + column);
    u16_t const *const halves = reinterpret_cast<u16_t const *>(row_start);
    u32_t words[4] = {0u, 0u, 0u, 0u};
    for (u32_t lane = 0; lane != 8u && column + lane < columns; ++lane)
        words[lane >> 1] |= static_cast<u32_t>(halves[column + lane]) << (16u * (lane & 1u));
    return make_uint4(words[0], words[1], words[2], words[3]);
}

__global__ void __launch_bounds__(256)
    unum::uform::core::transpose_bf16_kernel(bf16r2_view_t source, bf16r2_span_t destination) {
    __shared__ u32_t tile[64][33];
    std::size_t const rows = source.extent(0), columns = source.extent(1);
    std::size_t const source_stride = source.stride_bytes(0) / sizeof(bf16_t);
    std::size_t const destination_stride = destination.stride_bytes(0) / sizeof(bf16_t);
    bool const aligned = ((reinterpret_cast<std::uintptr_t>(source.data()) |
                           reinterpret_cast<std::uintptr_t>(destination.data()) |
                           static_cast<std::uintptr_t>(source.stride_bytes(0)) |
                           static_cast<std::uintptr_t>(destination.stride_bytes(0))) &
                          15u) == 0u;
    std::size_t const row_start = static_cast<std::size_t>(blockIdx.y) * 64u;
    std::size_t const column_start = static_cast<std::size_t>(blockIdx.x) * 64u;
    u32_t const pair = threadIdx.x >> 3, quad = threadIdx.x & 7u;

    std::size_t const upper_row = row_start + 2u * pair, column = column_start + quad * 8u;
    uint4 const zero = make_uint4(0u, 0u, 0u, 0u);
    uint4 const upper = upper_row < rows ? transpose_load_(source.data() + upper_row * source_stride, column, columns,
                                                          aligned)
                                         : zero;
    uint4 const lower = upper_row + 1u < rows
                            ? transpose_load_(source.data() + (upper_row + 1u) * source_stride, column, columns,
                                              aligned)
                            : zero;
    u32_t const first[4] = {upper.x, upper.y, upper.z, upper.w};
    u32_t const second[4] = {lower.x, lower.y, lower.z, lower.w};
#pragma unroll
    for (u32_t word = 0; word != 4u; ++word) {
        tile[quad * 8u + 2u * word][pair] = __byte_perm(first[word], second[word], 0x5410);
        tile[quad * 8u + 2u * word + 1u][pair] = __byte_perm(first[word], second[word], 0x7632);
    }
    __syncthreads();

    for (u32_t chunk = threadIdx.x; chunk < 512u; chunk += 256u) {
        std::size_t const target_row = column_start + (chunk >> 3), target_column = row_start + (chunk & 7u) * 8u;
        if (target_row >= columns || target_column >= rows) continue;
        u32_t const *words = &tile[chunk >> 3][(chunk & 7u) * 4u];
        bf16_t *const target = destination.data() + target_row * destination_stride + target_column;
        if (aligned && target_column + 8u <= rows)
            *reinterpret_cast<uint4 *>(target) = make_uint4(words[0], words[1], words[2], words[3]);
        else {
            u16_t *const halves = reinterpret_cast<u16_t *>(target);
            for (u32_t lane = 0; lane != 8u && target_column + lane < rows; ++lane)
                halves[lane] = static_cast<u16_t>(words[lane >> 1] >> (16u * (lane & 1u)));
        }
    }
}

/** @brief Element bytes of one cuBLASLt operand dtype — the benchmark's transient-buffer sizing. */
[[nodiscard]] static std::size_t element_bytes_(u32_t data_type) noexcept {
    switch (static_cast<cudaDataType_t>(data_type)) {
    case CUDA_R_8F_E4M3: return 1;
    case CUDA_R_16BF: return 2;
    default: return 4;
    }
}

expected<cublaslt_measured_t> unum::uform::core::cublaslt_benchmark_candidates(
    cublaslt_session_t &session, cublasLtMatmulDesc_t description, cublasLtMatrixLayout_t a_layout,
    cublasLtMatrixLayout_t b_layout, cublasLtMatrixLayout_t c_layout, cublaslt_plan_config_t const &config,
    std::span<cublasLtMatmulHeuristicResult_t const> candidates) noexcept {
    using result_t = expected<cublaslt_measured_t>;
    if (candidates.empty()) return result_t::fail(status_t::invalid_input_k, "nothing to benchmark");

    // Transient operands at the stored extents, plus one workspace at the largest candidate appetite.
    std::size_t const a_bytes = static_cast<std::size_t>(config.a_rows) * config.a_row_stride *
                                element_bytes_(static_cast<u32_t>(config.a_type));
    std::size_t const b_bytes = static_cast<std::size_t>(config.b_rows) * config.b_row_stride *
                                element_bytes_(static_cast<u32_t>(config.b_type));
    std::size_t const c_bytes = static_cast<std::size_t>(config.c_rows) * config.c_row_stride *
                                element_bytes_(static_cast<u32_t>(config.c_type));
    std::size_t workspace_bytes = 0;
    for (cublasLtMatmulHeuristicResult_t const &candidate : candidates)
        workspace_bytes = candidate.workspaceSize > workspace_bytes ? candidate.workspaceSize : workspace_bytes;

    CUdeviceptr a = 0, b = 0, c = 0, workspace = 0;
    CUstream stream = nullptr;
    expected<gpu_event_t> begin = gpu_event_t::make_timed();
    expected<gpu_event_t> end = gpu_event_t::make_timed();
    result_t outcome = result_t::fail(status_t::cuda_error_k, "benchmark bring-up failed");
    if (!begin || !end) return outcome;
    if (cuMemAlloc(&a, a_bytes) != CUDA_SUCCESS || cuMemAlloc(&b, b_bytes) != CUDA_SUCCESS ||
        cuMemAlloc(&c, c_bytes) != CUDA_SUCCESS ||
        (workspace_bytes != 0 && cuMemAlloc(&workspace, workspace_bytes) != CUDA_SUCCESS))
        outcome = result_t::fail(status_t::bad_alloc_k, "benchmark operands do not fit ({}+{}+{}+{} transient bytes)",
                                 a_bytes, b_bytes, c_bytes, workspace_bytes);
    else if (cuStreamCreate(&stream, CU_STREAM_NON_BLOCKING) != CUDA_SUCCESS)
        outcome = result_t::fail(status_t::cuda_error_k, "benchmark stream creation failed");
    else {
        // Nonzero bytes everywhere — the timing is value-blind, the operands must merely be resident.
        cuMemsetD8(a, 0x3C, a_bytes);
        cuMemsetD8(b, 0x3C, b_bytes);
        cuMemsetD8(c, 0, c_bytes);
        f32_t const alpha = 1.0f, beta = 0.0f;
        cublaslt_measured_t fastest;
        f32_t fastest_ms = -1.0f;
        for (std::size_t candidate = 0; candidate != candidates.size(); ++candidate) {
            cublasLtMatmulAlgo_t const &algorithm = candidates[candidate].algo;
            auto enqueue = [&]() noexcept {
                return cublasLtMatmul(session.get(), description, &alpha, reinterpret_cast<void const *>(a), a_layout,
                                      reinterpret_cast<void const *>(b), b_layout, &beta, reinterpret_cast<void *>(c),
                                      c_layout, reinterpret_cast<void *>(c), c_layout, &algorithm,
                                      reinterpret_cast<void *>(workspace), workspace_bytes, stream);
            };
            bool warmed = true;
            for (int warm = 0; warm != 2 && warmed; ++warm) warmed = enqueue() == CUBLAS_STATUS_SUCCESS;
            if (!warmed) continue; // a candidate the check accepted but the launch rejects is simply skipped
            constexpr int iterations_k = 10;
            if (cuEventRecord(begin.value_.get(), stream) != CUDA_SUCCESS) continue;
            bool timed = true;
            for (int iteration = 0; iteration != iterations_k && timed; ++iteration)
                timed = enqueue() == CUBLAS_STATUS_SUCCESS;
            if (!timed || cuEventRecord(end.value_.get(), stream) != CUDA_SUCCESS ||
                cuEventSynchronize(end.value_.get()) != CUDA_SUCCESS)
                continue;
            f32_t elapsed_ms = 0.0f;
            if (cuEventElapsedTime(&elapsed_ms, begin.value_.get(), end.value_.get()) != CUDA_SUCCESS) continue;
            f32_t const per_call_ms = elapsed_ms / static_cast<f32_t>(iterations_k);
            if (fastest_ms < 0.0f || per_call_ms < fastest_ms) {
                fastest_ms = per_call_ms;
                fastest.candidate_index = candidate;
                fastest.workspace_bytes = candidates[candidate].workspaceSize;
                fastest.microseconds = per_call_ms * 1000.0f;
            }
        }
        outcome = fastest_ms < 0.0f
                      ? result_t::fail(status_t::cuda_error_k, "no benchmark candidate survived its own launches")
                      : result_t {fastest};
    }
    if (stream != nullptr) cuStreamDestroy(stream);
    if (workspace != 0) cuMemFree(workspace);
    if (c != 0) cuMemFree(c);
    if (b != 0) cuMemFree(b);
    if (a != 0) cuMemFree(a);
    return outcome;
}
