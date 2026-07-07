/**
 *  @brief The run's one telemetry and introspection surface: @ref telemetry_ledger_t — a root-owned,
 *      always-real, runtime-disarmed ledger every consumer accepts by reference and feeds from its own
 *      boundaries, unifying the trainer's loss telemetry with the debug-grade signal, round, and timing
 *      probes that used to be ad-hoc printf archaeology.
 *  @file cuda_core/introspect.cuh
 *  @author Ash Vardanian
 *  @date July 7, 2026
 *
 *  The design contract, in order of importance:
 *
 *      - @b Always-real: consumers take @c telemetry_ledger_t& unconditionally — no nullable pointers,
 *        no compile-time twins. A disarmed ledger (the @ref telemetry_ledger_t::disarmed factory, or a
 *        channel the @ref telemetry_ledger_t::make mask left dark) makes every call an inert branch:
 *        no device memory, no kernels, no registered rows.
 *      - @b Root-owned, borrowed: @c main (or a suite's @c main ) constructs the one ledger and threads
 *        it by reference like the arenas and the domains (§8); the library never reads environment
 *        variables — arming is the launcher boundary's decision, mapped in the binaries' entry points.
 *      - @b Stateful device rows: each registered probe owns one @ref telemetry_row_t in a device table,
 *        updated in place by one-warp capture-safe kernels — so invariants the instant checks cannot
 *        express (changed-since-last-step, monotone rounds, zero-observation counting) are one probe.
 *      - @b Policy over dogma: a zero signal is @b legal (a zero gradient is a clean no-op by the
 *        optimizer's own contract) — @ref finite_signal_k counts zeros and never trips on them; only a
 *        non-finite value is unconditionally fatal, on every policy.
 *      - @b Monotone host statistics: loss summands accumulate into @ref telemetry_loss_stats_t running
 *        sums, so any window's mean is a subtraction of two snapshots — derive, don't store.
 */
#pragma once

#include <cstddef> // `std::size_t`

#include <numkong/tensor.hpp> // `nk::tensor`, `nk::tensor_span` — the device row table

#include "cuda_core/types.hpp"                // `expected`, `expected_t`, `status_t`, the scalar aliases
#include "cuda_core/array.hpp"                // `array` — the per-probe host tables
#include "cuda_core/gpu_allocators.hpp"       // `gpu_arena_t`, `gpu_arena_alloc`
#include "cuda_core/schedule.cuh"             // `gpu_stream_t`
#include "cuda_core/multimem_collectives.cuh" // `multimem_member_t` — the round probe's subject

namespace unum::uform::core {

/** @brief The ledger's runtime channels — a bitmask baked at @ref telemetry_ledger_t::make , never
 *      toggled afterwards, so armed and unarmed enqueues can never mix inside one capture. */
enum telemetry_channel_t : u32_t {
    /** @brief Host-fed loss summands — the trainer's per-head telemetry. */
    telemetry_losses_k = 1u << 0,
    /** @brief Device signal probes over Float32 spans — the gradient-boundary tripwires. */
    telemetry_signals_k = 1u << 1,
    /** @brief Device probes over a member's barrier-round scalar — monotonicity and progress. */
    telemetry_rounds_k = 1u << 2,
    /** @brief Host-fed durations — the same scalar path; adopters land incrementally. */
    telemetry_timings_k = 1u << 3,
};

/** @brief What a signal probe enforces beyond bookkeeping. Non-finite values are fatal on @b every
 *      policy; the policies only differ in what else counts as a violation. */
enum telemetry_policy_t : u32_t {
    /** @brief Count all-zero sample windows, never trip on them — zero signals are legal no-ops. */
    finite_signal_k = 0,
    /** @brief An all-zero sample window increments @ref telemetry_row_t::violations . */
    nonzero_signal_k = 1,
    /** @brief A value below the previous reading increments violations — the rounds' invariant. */
    monotone_signal_k = 2,
};

/** @brief A registered probe's handle — a strong-typed row index into the ledger's device table.
 *      The zero value is the inert sentinel every disarmed or dark-channel registration returns:
 *      feeding it is a well-formed no-op, so consumers never branch. */
struct telemetry_probe_t {
    u32_t row = 0;
};

/** @brief One probe's device-resident state, updated in place by the one-warp probe kernels — the
 *      currency of changed-since-last-step, zero-observation, and monotonicity questions. */
struct telemetry_row_t {
    /** @brief Probe executions, capture replays included. */
    u32_t updates = 0;
    /** @brief All-zero sample windows seen — legal under @ref finite_signal_k , a violation above. */
    u32_t zero_observations = 0;
    /** @brief Non-finite observations recorded before the trap fires. */
    u32_t nonfinite_observations = 0;
    /** @brief Policy violations accumulated. */
    u32_t violations = 0;
    /** @brief The last reading — a round value, or a sample window's L2. */
    f32_t last_scalar = 0.0f;
    /** @brief Keeps the row a clean 32 bytes. */
    u32_t padding_ = 0;
    /** @brief XOR-fold of the sampled bits — the changed-since-last-step currency. */
    u64_t last_checksum = 0;
};

/** @brief One loss probe's host-side running statistics — monotone accumulators, so any window's mean
 *      is a subtraction of two snapshots. A @c weight of zero means the probe never scored. */
struct telemetry_loss_stats_t {
    /** @brief Running @c Σ(summand) across every scored reading. */
    f64_t weighted_sum = 0.0;
    /** @brief Running @c Σ(count) — the sample mass behind @ref weighted_sum . */
    f64_t weight = 0.0;
    /** @brief How many readings scored (arrived with a nonzero count). */
    u64_t scored_steps = 0;
    /** @brief The first scored reading's mean — kept for the report line, not for assertions. */
    f32_t first_mean = 0.0f;
    /** @brief The last scored reading's mean. */
    f32_t last_mean = 0.0f;

    /** @brief The all-time sample-weighted mean; zero when never scored. */
    [[nodiscard]] f64_t mean() const noexcept { return weight != 0.0 ? weighted_sum / weight : 0.0; }
};

/**
 *  @brief The run's telemetry and introspection ledger — see the file header for the contract. Move-only,
 *      like the domains; constructed once at the composition root and borrowed by every consumer.
 */
struct telemetry_ledger_t {

    telemetry_ledger_t() noexcept = default;
    telemetry_ledger_t(telemetry_ledger_t const &) = delete;
    telemetry_ledger_t &operator=(telemetry_ledger_t const &) = delete;
    telemetry_ledger_t(telemetry_ledger_t &&) noexcept = default;
    telemetry_ledger_t &operator=(telemetry_ledger_t &&) noexcept = default;
    ~telemetry_ledger_t() noexcept = default;

    /** @brief The inert ledger — every method an early-out, no device memory, no rows. */
    [[nodiscard]] static telemetry_ledger_t disarmed() noexcept { return telemetry_ledger_t {}; }

    /** @brief The armed ledger: a zeroed device row table from @p arena (which must be visible to every
     *      local device — probes write rows from any member's stream) and host tables for up to
     *      @p max_probes registrations. @p stream orders the table's zeroing (a transient bring-up
     *      stream is fine — nothing is retained). @p channels is final — no toggle exists, so a capture
     *      can never bake a mix of armed and unarmed enqueues. */
    [[nodiscard]] static expected<telemetry_ledger_t> make(gpu_arena_t &arena, u32_t channels, u32_t max_probes,
                                                           gpu_stream_t &stream) noexcept;

    /** @brief Whether @p channel was armed at construction. */
    [[nodiscard]] bool armed(telemetry_channel_t channel) const noexcept { return (channels_ & channel) != 0u; }

    /** @brief Claim one named row for a probe on @p channel under @p policy — bump-registered with loud
     *      overflow, mirroring the domains' reserve discipline. A disarmed ledger or a dark channel
     *      returns the inert zero probe without consuming a row. */
    [[nodiscard]] expected<telemetry_probe_t> register_probe(sz::string_view name, telemetry_channel_t channel,
                                                             telemetry_policy_t policy) noexcept;

    /** @brief Enqueue one signal probe over @p values on @p stream : one warp stride-samples the span,
     *      updates the probe's row in place (L2, checksum, zero / non-finite counts, violations), and
     *      traps only on a non-finite value. Capture-safe — pure enqueue, fixed row address. */
    [[nodiscard]] expected_t probe_signal(telemetry_probe_t probe, f32r1_view_t values, gpu_stream_t &stream) noexcept;

    /** @brief Enqueue one round probe over @p member 's own barrier-round scalar: records the value and
     *      counts a violation when it moved backwards. Capture-safe, one thread. */
    [[nodiscard]] expected_t probe_round(telemetry_probe_t probe, multimem_member_t member,
                                         gpu_stream_t &stream) noexcept;

    /** @brief Fold one host-side loss reading into the probe's running statistics: @p sum is the
     *      weighted summand, @p count its sample mass — a zero count never scores, so an unscored head
     *      leaves no trace. */
    void scalar(telemetry_probe_t probe, f64_t sum, f64_t count) noexcept;

    /** @brief The probe's running loss statistics — a value snapshot; subtract two to window a mean. */
    [[nodiscard]] telemetry_loss_stats_t losses(telemetry_probe_t probe) const noexcept;

    /** @brief The probe's device row, synchronized through @p stream — a quiescent-point read; never
     *      call from inside a capture. */
    [[nodiscard]] expected<telemetry_row_t> row(telemetry_probe_t probe, gpu_stream_t &stream) noexcept;

    /** @brief Drain the device table through @p stream and emit one diagnostic line per registered
     *      probe on the host channel. */
    [[nodiscard]] expected_t report(gpu_stream_t &stream) noexcept;

  private:
    /** @brief The armed channel mask; zero means disarmed. */
    u32_t channels_ = 0;
    /** @brief The bump cursor — rows are 1-based, row 0 is the inert sentinel. */
    u32_t registered_ = 0;
    /** @brief The device row table, @b [max_probes+1] , zeroed at make. */
    nk::tensor<telemetry_row_t, gpu_arena_alloc<telemetry_row_t>, 1> rows_ {};
    /** @brief Owned per-probe labels, row order. */
    array<sz::string> names_ {};
    /** @brief Per-probe policies, baked into every enqueue. */
    array<telemetry_policy_t> policies_ {};
    /** @brief Per-probe host loss statistics, row order. */
    array<telemetry_loss_stats_t> loss_stats_ {};
};

} // namespace unum::uform::core
