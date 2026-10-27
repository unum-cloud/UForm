/**
 *  @file src/train/introspect.cu
 *  @author Ash Vardanian
 *  @date July 7, 2026
 *  @brief Device probe kernels and host method bodies of @ref telemetry_ledger_t declared in
 *      @ref introspect.cuh — the translation unit compiled into the `uform_train` static library
 *      beside @ref optimizer.cu and @ref multimem_collectives.cu .
 */
#include <cmath> // `std::sqrt` — a sample window's L2

#include <unistd.h> // `::write`, `STDERR_FILENO` — the host diagnostic channel

#include <cuda.h> // `cuMemsetD8`, `cuMemcpyDtoH`, `cuStreamSynchronize`

#include "uform/train/introspect.cuh"

using namespace unum::uform::train;
using namespace unum::uform::utilities;

/** How many values one signal probe samples at most — one warp, 256 strides per lane. */
inline constexpr std::size_t telemetry_sample_budget_k = 32u * 256u;

/**
 *  @brief One signal probe: the warp stride-samples @p values , ballots the zero / non-finite verdicts,
 *      folds an XOR checksum and an L2 across lanes, and lane 0 updates @p row in place.
 *
 *  Traps only on a non-finite value, whatever the policy; an all-zero window is counted, and a violation
 *  only under @ref nonzero_signal_k .
 *  __grid_size__ @c [1,1,1] , one warp; no dynamic shared memory.
 */
static __global__ void telemetry_signal_probe_kernel_(f32r1_view_t values, nk::tensor_span<telemetry_row_t, 1> row,
                                                      telemetry_policy_t policy, u32_t row_index, gpu_fault_t *fault) {
    if (threadIdx.x >= 32u) return;
    std::size_t const scalars = values.extent(0);
    std::size_t const samples = scalars < telemetry_sample_budget_k ? scalars : telemetry_sample_budget_k;
    std::size_t const sample_stride = scalars / samples;
    bool any_nonzero = false, any_nonfinite = false;
    f32_t energy = 0.0f;
    u64_t checksum = 0;
    for (std::size_t index = threadIdx.x; index < samples; index += 32u) {
        f32_t const value = values[index * sample_stride];
        any_nonzero |= value != 0.0f;
        any_nonfinite |= !isfinite(value);
        energy += value * value;
        checksum ^= static_cast<u64_t>(__float_as_uint(value)) << (index & 31u);
    }
    unsigned const nonzero_mask = __ballot_sync(0xFFFFFFFFu, any_nonzero);
    unsigned const nonfinite_mask = __ballot_sync(0xFFFFFFFFu, any_nonfinite);
    for (u32_t offset = 16u; offset != 0u; offset >>= 1u) {
        energy += __shfl_down_sync(0xFFFFFFFFu, energy, offset);
        checksum ^= __shfl_down_sync(0xFFFFFFFFu, checksum, offset);
    }
    if (threadIdx.x != 0u) return;
    telemetry_row_t &state = row[0];
    state.updates += 1u;
    state.last_scalar = std::sqrt(energy);
    state.last_checksum = checksum;
    if (nonzero_mask == 0u) {
        state.zero_observations += 1u;
        if (policy == telemetry_policy_t::nonzero_signal_k) state.violations += 1u;
    }
    if (nonfinite_mask != 0u) {
        state.nonfinite_observations += 1u;
        trap_with(fault, gpu_fault_code_t::nonfinite_signal_k, row_index, static_cast<u32_t>(samples),
                  static_cast<u32_t>(scalars));
    }
}

/**
 *  @brief One round probe: a single thread reads the member's own round scalar, counts a violation when
 *      it moved backwards, and records the value.
 *
 *  __grid_size__ @c [1,1,1] , one thread live; no dynamic shared memory.
 */
static __global__ void telemetry_round_probe_kernel_(multimem_member_t member,
                                                     nk::tensor_span<telemetry_row_t, 1> row) {
    if (threadIdx.x != 0u || blockIdx.x != 0u) return;
    telemetry_row_t &state = row[0];
    f32_t const value = static_cast<f32_t>(*member.round);
    state.updates += 1u;
    if (value < state.last_scalar) state.violations += 1u;
    state.last_scalar = value;
}

expected<telemetry_ledger_t> telemetry_ledger_t::make(gpu_arena_t &arena, u32_t channels, u32_t max_probes,
                                                      gpu_stream_t &stream) noexcept {
    using result_t = expected<telemetry_ledger_t>;
    telemetry_ledger_t ledger;
    ledger.channels_ = channels;
    if (channels == 0u) return result_t {std::move(ledger)};            // an explicit disarmed make is legal
    std::size_t const rows = static_cast<std::size_t>(max_probes) + 1u; // row 0 is the inert sentinel
    ledger.rows_ = nk::tensor<telemetry_row_t, gpu_arena_alloc<telemetry_row_t>, 1>::uninitialized(
                       {rows}, gpu_arena_alloc<telemetry_row_t> {arena})
                       .value;
    if (!ledger.rows_) return result_t::fail(status_t::bad_alloc_k, "ledger row table allocation failed ({})", rows);
    switch_to_gpu_t gpu_scope {stream};
    if (!gpu_scope.ok()) return result_t::fail(status_t::cuda_error_k, "ledger context entry failed");
    if (expected_t reset = zero_async(ledger.rows_.span(), stream); !reset) return result_t::fail(std::move(reset));
    if (CUresult synced = cuStreamSynchronize(stream.get()); synced != CUDA_SUCCESS)
        return cuda_failure<telemetry_ledger_t>(synced, "ledger row table reset drain failed");
    if (st::failed(ledger.names_.reserve(rows)) || st::failed(ledger.policies_.reserve(rows)) ||
        st::failed(ledger.loss_stats_.reserve(rows)))
        return result_t::fail(status_t::bad_alloc_k, "ledger host table allocation failed ({})", rows);
    // Row 0's host entries exist so probe handles index all tables uniformly.
    if (st::failed(ledger.names_.push_back(sz::string_t {})) ||
        st::failed(ledger.policies_.push_back(telemetry_policy_t::finite_signal_k)) ||
        st::failed(ledger.loss_stats_.push_back(telemetry_loss_stats_t {})))
        return result_t::fail(status_t::bad_alloc_k, "ledger sentinel row allocation failed");
    return result_t {std::move(ledger)};
}

expected<telemetry_probe_t> telemetry_ledger_t::register_probe(sz::string_view_t name, telemetry_channel_t channel,
                                                               telemetry_policy_t policy) noexcept {
    using result_t = expected<telemetry_probe_t>;
    if (!armed(channel)) return result_t {telemetry_probe_t {}}; // the inert probe — feeding it is a no-op
    if (registered_ + 1u >= rows_.extent(0))
        return result_t::fail(status_t::bad_alloc_k, "ledger probe registration of \"{}\" exceeds the {} rows", name,
                              rows_.extent(0) - 1u);
    sz::string_t owned;
    if (sz::failed(owned.assign(name)))
        return result_t::fail(status_t::bad_alloc_k, "ledger probe name allocation failed");
    if (st::failed(names_.push_back(std::move(owned))) || st::failed(policies_.push_back(std::move(policy))) ||
        st::failed(loss_stats_.push_back(telemetry_loss_stats_t {})))
        return result_t::fail(status_t::bad_alloc_k, "ledger probe table growth failed");
    registered_ += 1u;
    return result_t {telemetry_probe_t {registered_}};
}

expected_t telemetry_ledger_t::probe_signal(telemetry_probe_t probe, f32r1_view_t values, gpu_fault_t *fault,
                                            gpu_stream_t &stream) noexcept {
    if (probe.row == 0u || values.extent(0) == 0) return {};
    nk::tensor_span<telemetry_row_t, 1> const row {rows_.data() + probe.row, 1u};
    return launch_kernel(&telemetry_signal_probe_kernel_, dim3 {1u, 1u, 1u}, dim3 {32u, 1u, 1u}, stream, values, row,
                         policies_[probe.row], probe.row, fault);
}

expected_t telemetry_ledger_t::probe_round(telemetry_probe_t probe, multimem_member_t member,
                                           gpu_stream_t &stream) noexcept {
    if (probe.row == 0u) return {};
    nk::tensor_span<telemetry_row_t, 1> const row {rows_.data() + probe.row, 1u};
    return launch_kernel(&telemetry_round_probe_kernel_, dim3 {1u, 1u, 1u}, dim3 {32u, 1u, 1u}, stream, member, row);
}

void telemetry_ledger_t::scalar(telemetry_probe_t probe, f64_t sum, f64_t count) noexcept {
    if (probe.row == 0u || count == 0.0) return;
    telemetry_loss_stats_t &stats = loss_stats_[probe.row];
    stats.weighted_sum += sum;
    stats.weight += count;
    stats.scored_steps += 1u;
    f32_t const mean = static_cast<f32_t>(sum / count);
    if (stats.scored_steps == 1u) stats.first_mean = mean;
    stats.last_mean = mean;
}

telemetry_loss_stats_t telemetry_ledger_t::losses(telemetry_probe_t probe) const noexcept {
    if (probe.row == 0u) return {};
    return loss_stats_[probe.row];
}

expected<telemetry_row_t> telemetry_ledger_t::row(telemetry_probe_t probe, gpu_stream_t &stream) noexcept {
    using result_t = expected<telemetry_row_t>;
    if (probe.row == 0u) return result_t {telemetry_row_t {}};
    switch_to_gpu_t gpu_scope {stream};
    if (!gpu_scope.ok()) return result_t::fail(status_t::cuda_error_k, "ledger row context entry failed");
    if (CUresult synced = cuStreamSynchronize(stream.get()); synced != CUDA_SUCCESS)
        return cuda_failure<telemetry_row_t>(synced, "ledger row synchronization failed");
    telemetry_row_t state;
    if (CUresult copied = cuMemcpyDtoH(&state, device_address(rows_.data() + probe.row), sizeof(telemetry_row_t));
        copied != CUDA_SUCCESS)
        return cuda_failure<telemetry_row_t>(copied, "ledger row readback failed");
    return result_t {state};
}

expected_t telemetry_ledger_t::report(gpu_stream_t &stream) noexcept {
    if (channels_ == 0u || registered_ == 0u) return {};
    switch_to_gpu_t gpu_scope {stream};
    if (!gpu_scope.ok()) return expected_t::fail(status_t::cuda_error_k, "ledger report context entry failed");
    if (CUresult synced = cuStreamSynchronize(stream.get()); synced != CUDA_SUCCESS)
        return cuda_failure(synced, "ledger report synchronization failed");
    st::basic_vector<telemetry_row_t> states;
    if (st::failed(states.resize(registered_ + 1u)))
        return expected_t::fail(status_t::bad_alloc_k, "ledger report staging failed");
    if (CUresult copied = cuMemcpyDtoH(states.data(), device_address(rows_.data()),
                                       (registered_ + 1u) * sizeof(telemetry_row_t));
        copied != CUDA_SUCCESS)
        return cuda_failure(copied, "ledger report readback failed");
    for (u32_t index = 1; index <= registered_; ++index) {
        telemetry_row_t const &state = states[index];
        telemetry_loss_stats_t const &stats = loss_stats_[index];
        sz::string_t line;
        if (!format_into(line,
                         "uform-telemetry: {} — updates={} zeros={} nonfinite={} violations={} last={} | " //
                         "scored={} mean={} first={} last_mean={}\n",
                         names_[index].view(), state.updates, state.zero_observations, state.nonfinite_observations,
                         state.violations, state.last_scalar, stats.scored_steps, stats.mean(), stats.first_mean,
                         stats.last_mean))
            return expected_t::fail(status_t::bad_alloc_k, "ledger report line formatting failed");
        ssize_t const written = ::write(STDERR_FILENO, line.view().data(), line.view().size());
        (void)written;
    }
    return {};
}
