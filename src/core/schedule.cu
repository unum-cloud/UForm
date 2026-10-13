/**
 *  @brief The bodies declared in @ref schedule.cuh : the barrier-round increment that lets a captured
 *      collective region self-advance its round, the BFloat16 transpose, and the NumKong products built on it.
 *  @file src/core/schedule.cu
 *  @author Ash Vardanian
 *  @date July 5, 2026
 */
#include <cstdint> // `std::uintptr_t`

#include <cuda.h> // `CUdeviceptr`, driver launch surface

#include "core/schedule.cuh" // `barrier_round_t`, `launch_kernel`, `transpose_bf16`, the NumKong products

namespace uc = unum::uform::core;
using namespace uc;

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

/** Loads the eight elements at @p column of @p row as four packed words; out-of-range cells read zero. */
[[nodiscard]] static __device__ inline uint4 transpose_load_(bf16_t const *row_start, std::size_t column,
                                                             std::size_t columns, bool aligned) {
    if (aligned && column + 8u <= columns) return *reinterpret_cast<uint4 const *>(row_start + column);
    u16_t const *const halves = reinterpret_cast<u16_t const *>(row_start);
    u32_t words[4] = {0u, 0u, 0u, 0u};
    for (u32_t lane = 0; lane != 8u && column + lane < columns; ++lane)
        words[lane >> 1] |= static_cast<u32_t>(halves[column + lane]) << (16u * (lane & 1u));
    return make_uint4(words[0], words[1], words[2], words[3]);
}

__global__ void __launch_bounds__(256) uc::transpose_bf16_kernel(bf16r2_view_t source, bf16r2_span_t destination) {
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
    uint4 const upper = upper_row < rows
                            ? transpose_load_(source.data() + upper_row * source_stride, column, columns, aligned)
                            : zero;
    uint4 const lower = upper_row + 1u < rows ? transpose_load_(source.data() + (upper_row + 1u) * source_stride,
                                                                column, columns, aligned)
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

expected<std::size_t> uc::pack_bytes_bf16(std::size_t columns, std::size_t depth,
                                          nk_capability_t capabilities) noexcept {
    nk_size_t bytes = 0;
    if (nk_status_t const status = nk_dots_pack_size_bf16_best(columns, depth, capabilities, &bytes);
        status != nk_success_k)
        return expected<std::size_t>::fail(status_t::cuda_error_k, "pack sizing [{}x{}] failed: {}", columns, depth,
                                           nk_status_name(status));
    return expected<std::size_t> {static_cast<std::size_t>(bytes)};
}

expected<b8r1_tensor_arena_t> uc::allocate_pack_bf16(std::size_t columns, std::size_t depth,
                                                     nk_capability_t capabilities, gpu_arena_t &arena) noexcept {
    using result_t = expected<b8r1_tensor_arena_t>;
    expected<std::size_t> bytes = pack_bytes_bf16(columns, depth, capabilities);
    if (!bytes) return result_t::fail(std::move(bytes));
    b8r1_tensor_arena_t packed =
        b8r1_tensor_arena_t::uninitialized({bytes.value_}, gpu_arena_alloc<b8_t> {arena}).value;
    if (!packed)
        return result_t::fail(status_t::bad_alloc_k, "[{}x{}] pack allocation failed ({} bytes, arena_used={})",
                              columns, depth, bytes.value_, arena.used());
    return result_t {std::move(packed)};
}

expected_t uc::pack_bf16(bf16r2_view_t b, b8r1_span_t packed, nk_capability_t capabilities,
                         gpu_stream_t &stream) noexcept {
    std::size_t const columns = b.extent(0);
    return numkong_result(
        nk_dots_pack_bf16_best(reinterpret_cast<nk_bf16_t const *>(b.data()), columns, b.extent(1), b.stride_bytes(0),
                               packed.data(), 0, columns, capabilities, stream.numkong()),
        "packing");
}

expected_t uc::multiply_bf16(bf16r2_view_t a, b8r1_view_t packed_b, f32r2_span_t c, nk_capability_t capabilities,
                             gpu_stream_t &stream) noexcept {
    if (c.extent(0) != a.extent(0))
        return expected_t::fail(status_t::shape_unsupported_k, "[{}x{}]·Bᵀ cannot fill [{}x{}]", a.extent(0),
                                a.extent(1), c.extent(0), c.extent(1));
    return numkong_result(nk_dots_packed_bf16_best(reinterpret_cast<nk_bf16_t const *>(a.data()), packed_b.data(),
                                                   c.data(), a.extent(0), c.extent(1), a.extent(1), a.stride_bytes(0),
                                                   c.stride_bytes(0), capabilities, stream.numkong()),
                          "product");
}

expected_t uc::pack_weight_bf16(bf16r2_view_t weight, bf16r1_span_t transposed, b8r1_span_t packed,
                                b8r1_span_t packed_transposed, nk_capability_t capabilities,
                                gpu_stream_t &stream) noexcept {
    std::size_t const outputs = weight.extent(0), inputs = weight.extent(1);
    bf16r2_span_t const padded {transposed.data(), inputs, round_up_to_multiple(outputs, std::size_t {8})};
    bf16r2_span_t const weight_transposed = padded(nk::all, nk::range(0, outputs), nk::slice);
    expected_t staged = pack_bf16(weight, packed, capabilities, stream);
    if (staged) staged = transpose_bf16(weight, weight_transposed, stream);
    if (staged) staged = pack_bf16(weight_transposed, packed_transposed, capabilities, stream);
    return staged;
}

expected_t uc::weight_gradient_bf16(bf16r2_view_t output_grad, bf16r2_view_t input, bf16r2_span_t weight_grad,
                                    bf16r1_span_t transposed_grad, bf16r1_span_t transposed_input,
                                    b8r1_span_t packed_input, f32r1_span_t product, nk_capability_t capabilities,
                                    gpu_stream_t &stream) noexcept {
    std::size_t const tokens = output_grad.extent(0), outputs = output_grad.extent(1), inputs = input.extent(1);
    std::size_t const padded_tokens = round_up_to_multiple(tokens, std::size_t {8});
    bf16r2_span_t const output_grad_padded {transposed_grad.data(), outputs, padded_tokens};
    bf16r2_span_t const input_padded {transposed_input.data(), inputs, padded_tokens};
    bf16r2_span_t const output_grad_transposed = output_grad_padded(nk::all, nk::range(0, tokens), nk::slice);
    bf16r2_span_t const input_transposed = input_padded(nk::all, nk::range(0, tokens), nk::slice);
    f32r2_span_t const weight_grad_wide {product.data(), outputs, inputs};
    expected_t staged = transpose_bf16(output_grad, output_grad_transposed, stream);
    if (staged) staged = transpose_bf16(input, input_transposed, stream);
    if (staged) staged = pack_bf16(input_transposed, packed_input, capabilities, stream);
    if (staged) staged = multiply_bf16(output_grad_transposed, packed_input, weight_grad_wide, capabilities, stream);
    if (staged)
        staged = numkong_result(nk_cast_best(product.data(), nk_f32_k, weight_grad.data(), nk_bf16_k, outputs * inputs,
                                             capabilities, stream.numkong()),
                                "weight gradient narrowing");
    return staged;
}
