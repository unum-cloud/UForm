/**
 *  @file src/utilities/vram_nvidia_schedule.cu
 *  @author Ash Vardanian
 *  @date July 5, 2026
 *  @brief The bodies declared in @ref vram_nvidia_schedule.cuh : the barrier-round increment that
 *      lets a captured collective region self-advance its round, the BFloat16 transpose, and the
 *      NumKong products built on it.
 */
#include <cuda.h>      // `CUdeviceptr`, driver launch surface
#include <cuda/memory> // `cuda::is_aligned` — the transpose's 16-byte fast-path test

#include "uform/utilities/vram_nvidia_schedule.cuh" // `barrier_round_t`, `launch_kernel`, `transpose_bf16`, the NumKong products

namespace util = unum::uform::utilities;

/**
 *  @brief One thread advances the round scalar and publishes it system-wide, so a captured replay
 *      re-runs the bump on every launch instead of freezing at the capture-time value.
 *
 *  __grid_size__ @c [1,1,1] , one thread; no dynamic shared memory.
 */
static __global__ void barrier_round_increment_kernel_(util::u32r1_span_t round, util::u32_t increment) {
    if (threadIdx.x == 0u) {
        round[0] += increment;
        __threadfence_system();
    }
}

util::expected_t util::barrier_round_t::advance(util::executor_stream_t& stream) noexcept {
    return util::launch_kernel(&barrier_round_increment_kernel_, dim3 {1, 1, 1}, dim3 {1, 1, 1}, stream, device_round_,
                               increment_);
}

/** Whether a transpose's rows start and stride on 16-byte boundaries, so they load in vectors. */
enum class transpose_alignment_t : bool {
    unaligned_k,
    aligned_k,
};

/** Loads the eight elements at @p column of @p row as four packed words; out-of-range
 *  cells read zero. */
[[nodiscard]] static __device__ inline uint4 transpose_load_(util::bf16_t const* row_start,
                                                             std::size_t column, // device-boundary
                                                             std::size_t columns,
                                                             transpose_alignment_t alignment) noexcept {
    if (alignment == transpose_alignment_t::aligned_k && column + 8u <= columns) {
        uint4 vector;
        asm("ld.global.v4.u32 {%0, %1, %2, %3}, [%4];"
            : "=r"(vector.x), "=r"(vector.y), "=r"(vector.z), "=r"(vector.w)
            : "l"(row_start + column));
        return vector;
    }
    util::u32_t words[4] = {0u, 0u, 0u, 0u};
    for (util::u32_t lane = 0; lane != 8u && column + lane < columns; ++lane)
        words[lane >> 1] |= static_cast<util::u32_t>(__bfloat16_as_ushort(row_start[column + lane]))
                            << (16u * (lane & 1u));
    return make_uint4(words[0], words[1], words[2], words[3]);
}

__global__ void __launch_bounds__(256) util::transpose_bf16_kernel(util::bf16r2_view_t source,
                                                                   util::bf16r2_span_t destination) {
    __shared__ util::u32_t tile[64][33];
    std::size_t const rows = source.extent(0), columns = source.extent(1);
    std::size_t const source_stride = source.stride_bytes(0) / sizeof(util::bf16_t);
    std::size_t const destination_stride = destination.stride_bytes(0) / sizeof(util::bf16_t);
    bool const aligned = cuda::is_aligned(source.data(), 16u) && cuda::is_aligned(destination.data(), 16u) &&
                         ((source.stride_bytes(0) | destination.stride_bytes(0)) & 15) == 0;
    transpose_alignment_t const alignment = aligned ? transpose_alignment_t::aligned_k
                                                    : transpose_alignment_t::unaligned_k;
    std::size_t const row_start = static_cast<std::size_t>(blockIdx.y) * 64u;
    std::size_t const column_start = static_cast<std::size_t>(blockIdx.x) * 64u;
    util::u32_t const pair = threadIdx.x >> 3, quad = threadIdx.x & 7u;

    std::size_t const upper_row = row_start + 2u * pair, column = column_start + quad * 8u;
    uint4 const zero = make_uint4(0u, 0u, 0u, 0u);
    uint4 const upper = upper_row < rows
                            ? transpose_load_(source.data() + upper_row * source_stride, column, columns, alignment)
                            : zero;
    uint4 const lower = upper_row + 1u < rows ? transpose_load_(source.data() + (upper_row + 1u) * source_stride,
                                                                column, columns, alignment)
                                              : zero;
    util::u32_t const first[4] = {upper.x, upper.y, upper.z, upper.w};
    util::u32_t const second[4] = {lower.x, lower.y, lower.z, lower.w};
#pragma unroll
    for (util::u32_t word = 0; word != 4u; ++word) {
        util::u32_t const tile_row = quad * 8u + 2u * word; // the two transposed rows this word's columns become
        tile[tile_row][pair] = __byte_perm(first[word], second[word], 0x5410);
        tile[tile_row + 1u][pair] = __byte_perm(first[word], second[word], 0x7632);
    }
    __syncthreads();

    for (util::u32_t chunk = threadIdx.x; chunk < 512u; chunk += 256u) {
        std::size_t const target_row = column_start + (chunk >> 3), target_column = row_start + (chunk & 7u) * 8u;
        if (target_row >= columns || target_column >= rows) continue;
        util::u32_t const* words = &tile[chunk >> 3][(chunk & 7u) * 4u];
        util::bf16_t* const target = destination.data() + target_row * destination_stride + target_column;
        if (alignment == transpose_alignment_t::aligned_k && target_column + 8u <= rows)
            asm volatile("st.global.v4.u32 [%0], {%1, %2, %3, %4};"
                         :
                         : "l"(target), "r"(words[0]), "r"(words[1]), "r"(words[2]), "r"(words[3])
                         : "memory");
        else
            for (util::u32_t lane = 0; lane != 8u && target_column + lane < rows; ++lane)
                target[lane] = __ushort_as_bfloat16(static_cast<util::u16_t>(words[lane >> 1] >> (16u * (lane & 1u))));
    }
}

util::expected<std::size_t> util::pack_bytes_bf16(std::size_t columns, std::size_t depth,
                                                  util::gpu_numerics_t const& numerics) noexcept {
    nk_size_t bytes = 0;
    if (nk_status_t const status = numerics.pack_size_bf16(columns, depth, &bytes); status != nk_success_k)
        return util::expected<std::size_t>::fail(util::status_code_t::device_k, "pack sizing [{}x{}] failed: {}",
                                                 columns, depth, nk_status_name(status));
    return util::expected<std::size_t> {static_cast<std::size_t>(bytes)};
}

util::expected<util::b8r1_tensor_arena_t> util::allocate_pack_bf16(std::size_t columns, std::size_t depth,
                                                                   util::gpu_numerics_t const& numerics,
                                                                   util::gpu_arena_t& arena) noexcept {
    using result_t = util::expected<util::b8r1_tensor_arena_t>;
    auto [bytes, status] = util::pack_bytes_bf16(columns, depth, numerics);
    if (!status) return result_t::fail(std::move(status));
    util::b8r1_tensor_arena_t packed;
    if (!util::allocate_uninitialized(packed, {bytes}, util::gpu_arena_alloc<util::b8_t> {arena}))
        return result_t::fail(util::status_code_t::out_of_memory_k,
                              "[{}x{}] pack allocation failed ({} bytes, arena_used={})", columns, depth, bytes,
                              arena.used());
    return result_t {std::move(packed)};
}

util::expected_t util::pack_bf16(util::bf16r2_view_t b, util::b8r1_span_t packed, util::gpu_numerics_t const& numerics,
                                 util::executor_stream_t& stream) noexcept {
    std::size_t const columns = b.extent(0);
    return util::numkong_result(
        numerics.pack_bf16(nk::as_values<nk_bf16_t>(nk::as_bytes(b)).data(), columns, b.extent(1), b.stride_bytes(0),
                           packed.data(), 0, columns, util::numkong_stream(stream)),
        "packing");
}

util::expected_t util::multiply_bf16(util::bf16r2_view_t a, util::b8r1_view_t packed_b, util::f32r2_span_t c,
                                     util::gpu_numerics_t const& numerics, util::executor_stream_t& stream) noexcept {
    if (c.extent(0) != a.extent(0))
        return util::expected_t::fail(util::status_code_t::unsupported_k, "[{}x{}]·Bᵀ cannot fill [{}x{}]", a.extent(0),
                                      a.extent(1), c.extent(0), c.extent(1));
    return util::numkong_result(
        numerics.multiply_bf16(nk::as_values<nk_bf16_t>(nk::as_bytes(a)).data(), packed_b.data(), c.data(), a.extent(0),
                               c.extent(1), a.extent(1), a.stride_bytes(0), c.stride_bytes(0), NUMKONG_NULL, 0,
                               util::numkong_stream(stream)),
        "product");
}

util::expected_t util::pack_weight_bf16(util::bf16r2_view_t weight, util::bf16r1_span_t transposed,
                                        util::b8r1_span_t packed, util::b8r1_span_t packed_transposed,
                                        util::gpu_numerics_t const& numerics,
                                        util::executor_stream_t& stream) noexcept {
    std::size_t const outputs = weight.extent(0), inputs = weight.extent(1);
    util::bf16r2_span_t const padded {transposed.data(), inputs, util::round_up_to_multiple(outputs, std::size_t {8})};
    util::bf16r2_span_t const weight_transposed = padded[nk::all, nk::range(0, outputs), nk::slice];
    util::expected_t staged = util::pack_bf16(weight, packed, numerics, stream);
    if (staged) staged = util::transpose_bf16(weight, weight_transposed, stream);
    if (staged) staged = util::pack_bf16(weight_transposed, packed_transposed, numerics, stream);
    return staged;
}

util::expected<std::size_t> util::weight_gradient_workspace_bytes_bf16(std::size_t outputs, std::size_t inputs,
                                                                       std::size_t tokens,
                                                                       util::gpu_numerics_t const& numerics) noexcept {
    nk_size_t bytes = 0;
    if (nk_status_t const status = numerics.multiply_workspace_size_bf16(outputs, inputs, tokens, &bytes);
        status != nk_success_k)
        return util::expected<std::size_t>::fail(util::status_code_t::device_k,
                                                 "weight gradient workspace sizing [{}x{}x{}] failed: {}", outputs,
                                                 inputs, tokens, nk_status_name(status));
    return util::expected<std::size_t> {static_cast<std::size_t>(bytes)};
}

util::expected_t util::weight_gradient_bf16(util::bf16r2_view_t output_grad, util::bf16r2_view_t input,
                                            util::bf16r2_span_t weight_grad, util::bf16r1_span_t transposed_grad,
                                            util::bf16r1_span_t transposed_input, util::b8r1_span_t packed_input,
                                            util::f32r1_span_t product, util::b8r1_span_t workspace,
                                            util::gpu_numerics_t const& numerics,
                                            util::executor_stream_t& stream) noexcept {
    std::size_t const tokens = output_grad.extent(0), outputs = output_grad.extent(1), inputs = input.extent(1);
    std::size_t const padded_tokens = util::round_up_to_multiple(tokens, std::size_t {8});
    util::bf16r2_span_t const output_grad_padded {transposed_grad.data(), outputs, padded_tokens};
    util::bf16r2_span_t const input_padded {transposed_input.data(), inputs, padded_tokens};
    util::bf16r2_span_t const output_grad_transposed = output_grad_padded[nk::all, nk::range(0, tokens), nk::slice];
    util::bf16r2_span_t const input_transposed = input_padded[nk::all, nk::range(0, tokens), nk::slice];
    util::expected_t staged = util::transpose_bf16(output_grad, output_grad_transposed, stream);
    if (staged) staged = util::transpose_bf16(input, input_transposed, stream);
    if (staged) staged = util::pack_bf16(input_transposed, packed_input, numerics, stream);
    if (staged)
        staged = util::numkong_result(
            numerics.multiply_bf16(nk::as_values<nk_bf16_t>(nk::as_bytes(output_grad_transposed)).data(),
                                   packed_input.data(), product.data(), outputs, inputs, tokens,
                                   output_grad_transposed.stride_bytes(0), inputs * sizeof(util::f32_t),
                                   workspace.data(), workspace.extent(0), util::numkong_stream(stream)),
            "weight gradient product");
    if (staged)
        staged = util::numkong_result(numerics.cast(product.data(), nk_f32_k, weight_grad.data(), nk_bf16_k,
                                                    outputs * inputs, util::numkong_stream(stream)),
                                      "weight gradient narrowing");
    return staged;
}
