/**
 *  @brief The Base training conformance test: builds the same @ref trainer_t the binary and the C verbs drive and
 *      asserts the invariants the loop itself never checks — deep replay conformance for the tokens and omni
 *      mixtures, plus a cross-product sweep over every media subset, row budget, and model-region mode.
 *  @file test/base.cu
 *  @author Ash Vardanian
 *  @date July 6, 2026
 *
 *  This is a test: a plain @c int @c main builds one @ref test_suite_t and threads it by reference into every
 *  case, consuming @c src/train/trainer.hpp 's typed surface instead of copying the driver. Two meshes are made
 *  once in @c main — the bootstrapped world and a one-GPU narrowing of it — and every case builds its own trainer
 *  over one of them, exactly as the C verbs compose production. Launched under Slurm via
 *  @c scripts/slurm/test_base-4-8-16.sbatch , the same binary runs the distributed tiers: @c main honors
 *  @c UFORM_HOST_DOMAIN_COUNT , so a two-node world exercises the InfiniBand reduce under every case below.
 *
 *  The case matrix:
 *
 *      - @ref learning_rate_schedule — the warmup / cosine policy's properties, pure host math, no GPU.
 *      - @ref run_base_conformance for the tokens and omni mixtures at full width — @b A every replica
 *        bitwise-matches the lead, @b B every member's barrier round advanced and they all agree, @b C an
 *        eager trainer of the same seed lands on the captured run, @b D each head's loss fell between its first
 *        and last scored steps, @b E every live head's loss read nonzero at least once, @b F the captured
 *        window moved the weights off their seed.
 *      - @ref mixture_sweep — two width tiers: every media subset (2³) crossed with one-row and two-row
 *        budgets on the one-GPU mesh (plus the uneven five-row and 96-column omni corners), then every subset
 *        again at the bootstrapped world's full width. Each cell steps eagerly, then captures and replays over
 *        the same trainer, so both model-region modes run under every mixture.
 *      - @ref graph_capture — the capture contract holds and two replays agree, on the one-GPU mesh.
 */
#include <cmath>   // `std::isfinite`
#include <cstdlib> // `std::getenv`, `std::strtod`
#include <cstring> // `std::memcmp`, `std::memcpy`, `std::strlen`

#include <algorithm> // `std::max` — the reusable staging tensor's size
#include <span>      // `std::span` — device lists

#include <cuda.h> // `cuInit`, `cuStreamSynchronize`

#include "train/trainer.hpp" // `trainer_config_t`, `trainer_t`, `mesh_t`, `synthetic_source_t`
#include "harness.hpp"       // `test_suite_t` — the cumulative pass/fail tally

using namespace unum::uform::core;
using namespace unum::uform::train;
using namespace unum::uform::test;

/**
 *  @brief The tiny shared geometry — @c depth=2 , @c hidden=256 , @c 16 steps, and a two-step warmup so even
 *      the tiny run crosses both schedule phases.
 *
 *  Tokens ride a @c 2×64 rectangle; the omni variant fills every media slice over a @c 4×64 one, so the keyed
 *  partition holds all four modalities live at every step and the composite loss measures the same mixture
 *  throughout — descent stays comparable. Sixteen steps, because descent is asserted on the @b world mean: at a
 *  16-GPU world the data spreads twice as wide as one box's, and the eight-step horizon left the token head's
 *  early-vs-late comparison genuinely marginal (a ~0.7% rise on the world mean) while every machinery invariant
 *  held.
 */
[[nodiscard]] static trainer_config_t test_geometry_(bool omni) noexcept {
    trainer_config_t config;
    config.batch = multimodal_batch_shape_t {2u, 2u, 64u};
    if (omni) {
        config.batch = multimodal_batch_shape_t {4u, 4u, 64u};
        config.batch.image = image_batch_shape_t {8u, 3u};
        config.batch.video = video_batch_shape_t {8u, 3u, 2u, 1u};
        config.batch.audio = audio_batch_shape_t {256u, 16000u};
    }
    config.depth = 2;
    config.hidden = 256;
    config.steps = 16;
    config.warmup_steps = 2;
    config.telemetry_channels = telemetry_losses_k | telemetry_signals_k | telemetry_rounds_k;
    return config;
}

/** The test tier's rendezvous deadline — short enough that a dead peer never costs the surviving
 *  domain two minutes per case, but long enough for the real two-node cold start: the FIRST exchange
 *  rides NFS attribute-cache latency plus the peers' eager-module-loading skew, which can take tens
 *  of seconds (a warm local rendezvous completes in well under one). */
inline constexpr f64_t test_rendezvous_timeout_seconds_k = 60.0;

/** One width tier: the mesh every cell of the tier builds its trainer over, and the devices it drives. */
struct tier_t {
    mesh_t &mesh;
    std::span<i32_t const> devices;
};

/**
 *  @brief Narrow the bootstrapped @p world into @p placed : this process alone drives its first @p gpus GPUs as an
 *      isolated one-host-domain mesh — the sweep's cheap tier and the capture case's world.
 *
 *  Safe because a single-host topology carries no rendezvous state; the one cross-field invariant is that
 *  @c host_domains sizes to @c shape.host_domain_count() .
 */
[[nodiscard]] static expected_t narrow_mesh_(mesh_t &placed, mesh_topology_t const &world, u32_t gpus) noexcept {
    placed.topology.shape.gpus_per_host_domain = gpus;
    placed.topology.shape.gpus_per_multimem_domain = gpus;
    placed.topology.shape.gpus_per_colocated_domain = gpus;
    placed.topology.host_domain_index = 0;
    if (placed.topology.host_domains.resize(1) != status_t::success_k)
        return expected_t::fail(status_t::bad_alloc_k, "narrowed host-domain table allocation failed");
    placed.topology.host_domains[0] = world.host_domains[world.host_domain_index];
    placed.topology.host_domains[0].host_domain_index = 0;
    placed.topology.host_domains[0].device_count = gpus;
    expected<colocated_domain_t> colocated = colocated_domain_t::make(placed.topology);
    if (!colocated) return expected_t::fail(std::move(colocated));
    placed.colocated = std::move(colocated.value_);
    return {};
}

/** The world's health, owned by @c main beside the suite and threaded by reference into every case:
 *  a device-side fault poisons the local context (every later case fails with the same unrelated
 *  "unspecified launch failure"), and a dead peer domain makes every later rendezvous wait its full
 *  deadline — either way the first detection reports once and the remaining cases return immediately,
 *  so a faulted run finishes in seconds instead of minutes. */
struct world_health_t {

    /** Cleared at the first poisoned-context or unresponsive-peer detection; later cases skip. */
    bool alive = true;

    /** Consecutive trainer bring-up failures — a healthy world never fails one, so two in a row
     *  mean the peer domain is gone. */
    u32_t consecutive_setup_failures = 0;
};

/** The reason every case after the first unhealthy detection skips with. */
inline constexpr char const *world_unhealthy_k = "an earlier case found the world unhealthy";

/**
 *  @brief Probe @p health before a case: report the first poisoned-context or unresponsive-peer detection
 *      once, and tell every later case to return immediately.
 *  @param[inout] suite The suite the one abort line reports into.
 *  @param[inout] health The world's health record, cleared at the first detection.
 */
[[nodiscard]] static bool world_still_healthy_(test_suite_t &suite, world_health_t &health) noexcept {
    if (!health.alive) return false;
    if (health.consecutive_setup_failures >= 2u) {
        health.alive = false;
        suite.check(false, "peer domain unresponsive — aborting the remaining cases");
        return false;
    }
    expected<gpu_stream_t> probe = gpu_stream_t::make(0);
    if (probe) return true;
    health.alive = false;
    suite.check(false, "device context poisoned — aborting the remaining cases: {}", probe.description_);
    return false;
}

/** Build @p placed over @p tier for @p config , tallying bring-up failures into @p health . */
[[nodiscard]] static expected_t make_trainer_(trainer_t &placed, tier_t tier, trainer_config_t const &config,
                                              world_health_t &health) noexcept {
    expected_t built = trainer_t::make(placed, tier.mesh, config, tier.devices);
    health.consecutive_setup_failures = built ? 0u : health.consecutive_setup_failures + 1u;
    return built;
}

/** Enqueue steps @b [begin,end) of the synthetic loop. */
[[nodiscard]] static expected_t run_steps_(trainer_t &trainer, u32_t begin, u32_t end) noexcept {
    synthetic_source_t source;
    step_report_t report;
    for (u32_t ordinal = begin; ordinal != end; ++ordinal)
        if (expected_t stepped = trainer.step(source, optimizer_step_t {ordinal}, report); !stepped) return stepped;
    return {};
}

/** DtoH one replica's whole registration list into a byte-comparable host array — the snapshot behind
 *  assertions A, C, and the gradient-propagation check. */
[[nodiscard]] static expected<array<bf16_t>> snapshot_weights_(array<differentiable_weights_t> const &weights,
                                                               gpu_stream_t &stream) noexcept {
    using result_t = expected<array<bf16_t>>;
    switch_to_gpu_t gpu_scope {stream};
    if (!gpu_scope.ok()) return result_t::fail(status_t::cuda_error_k, "context entry failed");
    std::span<differentiable_weights_t const> const list = weights.view();
    std::size_t total = 0, largest = 0;
    for (differentiable_weights_t const &matrix : list) {
        std::size_t const scalars = matrix.weights.numel();
        total += scalars;
        largest = std::max(largest, scalars);
    }
    array<bf16_t> host;
    if (host.resize_uninitialized(total) != status_t::success_k)
        return result_t::fail(status_t::bad_alloc_k, "snapshot allocation failed");
    nk::tensor<bf16_t, pinned_alloc<bf16_t>, 1> staging =
        nk::tensor<bf16_t, pinned_alloc<bf16_t>, 1>::uninitialized({largest}, pinned_alloc<bf16_t> {}).value;
    if (!staging) return result_t::fail(status_t::bad_alloc_k, "pinned staging allocation failed");
    std::size_t cursor = 0;
    for (differentiable_weights_t const &matrix : list) {
        std::size_t const scalars = matrix.weights.numel();
        if (nk::failed(staging.resize({scalars})))
            return result_t::fail(status_t::bad_alloc_k, "staging resize failed");
        nk::tensor_view<bf16_t, 1> const source = matrix.weights.flatten<1>();
        if (expected_t copied = copy_device_to_host(source, staging.span(), stream); !copied)
            return result_t::fail(std::move(copied));
        cuStreamSynchronize(stream.get());
        std::memcpy(host.data() + cursor, staging.data(), scalars * sizeof(bf16_t));
        cursor += scalars;
    }
    return result_t {std::move(host)};
}

/** DtoH the lead replica's shared @c d(normalized) - the score output whose two captured replays must agree. */
[[nodiscard]] static expected<array<bf16_t>> snapshot_grad_(base_replica_t &replica, gpu_stream_t &stream) noexcept {
    using result_t = expected<array<bf16_t>>;
    switch_to_gpu_t gpu_scope {stream};
    if (!gpu_scope.ok()) return result_t::fail(status_t::cuda_error_k, "context entry failed");
    bf16r2_view_t const view = replica.normalized_grad.view();
    std::size_t const scalars = static_cast<std::size_t>(view.extent(0)) * view.extent(1);
    nk::tensor<bf16_t, pinned_alloc<bf16_t>, 1> staging =
        nk::tensor<bf16_t, pinned_alloc<bf16_t>, 1>::uninitialized({scalars}, pinned_alloc<bf16_t> {}).value;
    if (!staging) return result_t::fail(status_t::bad_alloc_k, "pinned staging allocation failed");
    if (expected_t copied = copy_device_to_host(view.flatten<1>(), staging.span(), stream); !copied)
        return result_t::fail(std::move(copied));
    cuStreamSynchronize(stream.get());
    array<bf16_t> host;
    if (host.resize_uninitialized(scalars) != status_t::success_k)
        return result_t::fail(status_t::bad_alloc_k, "snapshot allocation failed");
    std::memcpy(host.data(), staging.data(), scalars * sizeof(bf16_t));
    return result_t {std::move(host)};
}

/** Bytewise-equal two snapshots, as replicas fed one all-reduced gradient must agree exactly. */
[[nodiscard]] static bool snapshots_match_(array<bf16_t> const &left, array<bf16_t> const &right) noexcept {
    return left.size() == right.size() && std::memcmp(left.data(), right.data(), left.size() * sizeof(bf16_t)) == 0;
}

/** Two snapshots agree within 1% of the first's peak, for runs whose reductions may add in any order. */
[[nodiscard]] static bool snapshots_near_(array<bf16_t> const &left, array<bf16_t> const &right) noexcept {
    if (left.size() != right.size()) return false;
    f32_t largest = 0.0f, worst = 0.0f;
    for (std::size_t index = 0; index != left.size(); ++index) {
        f32_t const expected_value = __bfloat162float(left[index]);
        largest = std::max(largest, std::fabs(expected_value));
        worst = std::max(worst, std::fabs(expected_value - __bfloat162float(right[index])));
    }
    return worst <= 1.0e-2f * largest;
}

/** The warmup / cosine policy's own properties, pure host math: a strictly monotone ramp topping out
 *  exactly at the peak, a never-rising decay bounded by @b [floor, peak] , and a tail settling onto the
 *  floor. */
static void learning_rate_schedule(test_suite_t &suite) {
    trainer_config_t config;
    config.learning_rate = 1.0e-3f;
    config.min_learning_rate = 1.0e-5f;
    config.warmup_steps = 10;
    config.steps = 100;

    bool ramp_monotone = true;
    for (u32_t ordinal = 1; ordinal < config.warmup_steps; ++ordinal)
        ramp_monotone = ramp_monotone && config.learning_rate_for(optimizer_step_t {ordinal}) >
                                             config.learning_rate_for(optimizer_step_t {ordinal - 1u});
    suite.check(ramp_monotone, "the warmup ramp is strictly monotone");
    f32_t const peak = config.learning_rate_for(optimizer_step_t {config.warmup_steps - 1u});
    suite.check(peak == config.learning_rate, "the ramp tops out exactly at the peak: got {} want {}", peak,
                config.learning_rate);

    bool decay_monotone = true, bounded = true;
    f32_t previous = peak;
    for (u32_t ordinal = config.warmup_steps; ordinal < config.steps; ++ordinal) {
        f32_t const rate = config.learning_rate_for(optimizer_step_t {ordinal});
        decay_monotone = decay_monotone && rate <= previous;
        bounded = bounded && rate >= config.min_learning_rate && rate <= config.learning_rate;
        previous = rate;
    }
    suite.check(decay_monotone, "the cosine decay never rises");
    suite.check(bounded, "every decay rate stays within the floor-to-peak band");
    f32_t const tail = config.learning_rate_for(optimizer_step_t {config.steps - 1u});
    suite.check(tail <= config.min_learning_rate * 1.1f, "the tail settles onto the floor: got {} want <= {}", tail,
                config.min_learning_rate * 1.1f);
}

/** The checkpoint naming contract over one built replica: the registration list carries the semantic
 *  per-weight names every SafeTensors parser keys by (under the driver's @c model. prefix), derived from
 *  @p config and asserted in full order — a @c fill reorder or rename fails loudly here before it silently
 *  re-keys a checkpoint. Piggybacks on an already-built trainer, so it costs no bring-up of its own. */
static void check_weight_naming_(test_suite_t &suite, char const *label, trainer_config_t const &config,
                                 std::span<differentiable_weights_t const> weights) {
    std::size_t index = 0;
    auto expect = [&](sz::string_view_t want) {
        if (index >= weights.size()) {
            suite.check(false, "[{}] weight {} missing: want {}", label, index, want);
            return;
        }
        suite.check(weights[index].name.view() == want, "[{}] weight {}: got {} want {}", label, index,
                    weights[index].name.view(), want);
        ++index;
    };
    expect("embedding");
    if (config.batch.contains(modality_t::image_k)) expect("image_projection");
    if (config.batch.contains(modality_t::video_k)) expect("video_projection");
    if (config.batch.contains(modality_t::audio_k)) expect("audio_projection");
    sz::string_t name;
    for (u32_t layer_index = 0; layer_index != config.depth; ++layer_index)
        for (sz::string_view_t part : {"qkv", "out", "gate_up", "down", "pre_norm", "post_norm"}) {
            name.clear();
            if (!format_into(name, "blocks.{}.{}", layer_index, part))
                return suite.check(false, "[{}] weight name format failed", label);
            expect(name.view());
        }
    expect("final_norm");
    expect("pool_query");
    expect("head.norm");
    expect("head.fc1");
    expect("head.fc2");
    expect("masked_tokens");
    if (config.batch.contains(modality_t::image_k)) expect("masked_image");
    if (config.batch.contains(modality_t::video_k)) expect("masked_video");
    if (config.batch.contains(modality_t::audio_k)) expect("masked_audio");
    suite.check(index == weights.size(), "[{}] registration count: got {} want {}", label, weights.size(), index);
}

/** Bring up a captured trainer for one mixture, run its loop reading telemetry every step, and fold the
 *  invariants A, B, D, E and F into @p suite ; then an eager trainer of the same seed checks C. D and E read the
 *  ledger's running loss statistics — every local member's sample mass, not a lead-only coin flip. */
static void run_base_conformance(test_suite_t &suite, world_health_t &health, char const *label,
                                 trainer_config_t const &config, tier_t tier) {
    if (!world_still_healthy_(suite, health)) return suite.skip(world_unhealthy_k);
    mesh_topology_t const &topology = tier.mesh.topology;
    array<bf16_t> captured_lead;
    {
        trainer_t trainer;
        if (expected_t built = make_trainer_(trainer, tier, config, health); !built)
            return suite.check(false, "[{}] trainer bring-up: {}", label, built.description_);
        u32_t const gpus = static_cast<u32_t>(trainer.replicas.size());
        check_weight_naming_(suite, label, config, trainer.replicas[0].weights.view());

        // Assertion F's reference: the lead replica's freshly seeded weights, untouched by the capture.
        expected<array<bf16_t>> seed_lead = snapshot_weights_(trainer.replicas[0].weights, trainer.streams[0]);
        if (!seed_lead) return suite.check(false, "[{}] seed snapshot: {}", label, seed_lead.description_);

        // The captured run, telemetry every step. A midpoint snapshot splits the run into two halves; assertion D
        // compares their sample-weighted means, so a head scored on only some steps contributes its whole sample
        // mass, not one noisy endpoint.
        telemetry_probe_t const head_probes[4] = {trainer.probes.token, trainer.probes.image, trainer.probes.video,
                                                  trainer.probes.audio};
        telemetry_loss_stats_t mid_stats[4] {};
        synthetic_source_t source;
        step_report_t report;
        for (u32_t ordinal = 0; ordinal != config.steps; ++ordinal) {
            if (expected_t ran = trainer.step(source, optimizer_step_t {ordinal}, report); !ran)
                return suite.check(false, "[{}] captured step {}: {}", label, ordinal, ran.description_);
            (void)trainer.telemetry(optimizer_step_t {ordinal});
            if (ordinal + 1u == config.steps / 2u)
                for (u32_t head = 0; head != 4u; ++head) mid_stats[head] = trainer.ledger.losses(head_probes[head]);
        }
        telemetry_loss_stats_t end_stats[4] {};
        for (u32_t head = 0; head != 4u; ++head) end_stats[head] = trainer.ledger.losses(head_probes[head]);

        // At multi-domain worlds a node's telemetry sees only its slice of the objective the optimizer
        // minimizes — a local half can rise while the world mean falls, and that is descent, not a defect.
        // Fold every host_domain's summands through the mesh exchange, so D judges the world mean and every
        // node reaches the same verdict.
        u32_t const host_domains = static_cast<u32_t>(topology.host_domains.size());
        if (host_domains > 1u) {
            sz::string_t mine;
            bool formatted = true;
            for (u32_t head = 0; head != 4u; ++head)
                formatted = formatted &&
                            format_into(mine, "{:.17e} {:.17e} {:.17e} {:.17e} {}\n", mid_stats[head].weighted_sum,
                                        mid_stats[head].weight, end_stats[head].weighted_sum, end_stats[head].weight,
                                        end_stats[head].scored_steps);
            sz::string_t key;
            formatted = formatted && static_cast<bool>(format_into(key, "conformance.{}.loss_summands", label));
            if (!formatted) return suite.check(false, "[{}] loss-summand exchange formatting failed", label);
            if (expected_t published = topology.publish_text(key.view(), mine.view()); !published)
                return suite.check(false, "[{}] loss-summand publish: {}", label, published.description_);
            telemetry_loss_stats_t world_mid[4] {}, world_end[4] {};
            for (u32_t domain_index = 0; domain_index != host_domains; ++domain_index) {
                sz::string_t text;
                if (domain_index == topology.host_domain_index) {
                    if (sz::failed(text.assign(mine.view())))
                        return suite.check(false, "[{}] loss-summand staging failed", label);
                }
                else if (expected_t arrived = topology.await_text(domain_index, key.view(), text,
                                                                  test_rendezvous_timeout_seconds_k);
                         !arrived)
                    return suite.check(false, "[{}] loss-summand exchange: {}", label, arrived.description_);
                char const *cursor = text.c_str();
                for (u32_t head = 0; head != 4u; ++head) {
                    char *next = nullptr;
                    world_mid[head].weighted_sum += std::strtod(cursor, &next); // vendor-boundary: libc parse
                    world_mid[head].weight += std::strtod(next, &next);         // vendor-boundary
                    world_end[head].weighted_sum += std::strtod(next, &next);   // vendor-boundary
                    world_end[head].weight += std::strtod(next, &next);         // vendor-boundary
                    world_end[head].scored_steps += static_cast<u64_t>(std::strtod(next, &next)); // vendor-boundary
                    cursor = next;
                }
            }
            for (u32_t head = 0; head != 4u; ++head) {
                mid_stats[head] = world_mid[head];
                end_stats[head] = world_end[head];
            }
        }
        if (expected_t drained = trainer.synchronize(); !drained)
            return suite.check(false, "[{}] drain: {}", label, drained.description_);
        expected<array<bf16_t>> lead = snapshot_weights_(trainer.replicas[0].weights, trainer.streams[0]);
        if (!lead) return suite.check(false, "[{}] lead snapshot: {}", label, lead.description_);
        captured_lead = std::move(lead.value_);

        // A — every replica byte-matches the lead. Vacuous at one GPU (the peer loop is empty), but the run stands.
        bool replicas_identical = true;
        for (u32_t member_index = 1; member_index < gpus; ++member_index) {
            expected<array<bf16_t>> peer = snapshot_weights_(trainer.replicas[member_index].weights,
                                                             trainer.streams[member_index]);
            replicas_identical = replicas_identical && peer && snapshots_match_(peer.value_, captured_lead);
        }
        suite.check(replicas_identical, "[{}] every replica matches the lead bitwise (A)", label);

        // B — every member's own barrier round advanced off zero, never moved backwards, and they all agree:
        //     the ledger's monotone round rows carry all three (each member's scalar counts its own barriers).
        expected<telemetry_row_t> lead_round = trainer.ledger.row(trainer.probes.rounds[0], trainer.streams[0]);
        bool rounds_read = static_cast<bool>(lead_round), round_uniform = true, round_monotone = true;
        for (u32_t member_index = 0; member_index < gpus; ++member_index) {
            expected<telemetry_row_t> member_round = trainer.ledger.row(trainer.probes.rounds[member_index],
                                                                        trainer.streams[member_index]);
            rounds_read = rounds_read && member_round;
            round_uniform = round_uniform && member_round && lead_round &&
                            member_round.value_.last_scalar == lead_round.value_.last_scalar;
            round_monotone = round_monotone && member_round && member_round.value_.violations == 0u;
        }
        suite.check(rounds_read && lead_round.value_.last_scalar > 0.0f,
                    "[{}] barrier round advanced off zero: got {} (B)", label,
                    rounds_read ? lead_round.value_.last_scalar : 0.0f);
        suite.check(round_uniform, "[{}] barrier round reads identically through every member (B)", label);
        suite.check(round_monotone, "[{}] no member's barrier round ever moved backwards (B)", label);

        // D — per live head, the sample-weighted mean of the run's late half fell below the early half's;
        //     the late half is a subtraction of two monotone snapshots. A head both halves scored votes;
        //     the sums of voting heads decide. Loose by design — one deterministic descent, not convergence.
        f64_t early_mean_sum = 0.0, late_mean_sum = 0.0;
        u32_t voting_heads = 0;
        for (u32_t head = 0; head != 4u; ++head) {
            f64_t const early_weight = mid_stats[head].weight;
            f64_t const late_weight = end_stats[head].weight - mid_stats[head].weight;
            if (early_weight == 0.0 || late_weight == 0.0) continue; // unscored half — E guards coverage
            early_mean_sum += mid_stats[head].weighted_sum / early_weight;
            late_mean_sum += (end_stats[head].weighted_sum - mid_stats[head].weighted_sum) / late_weight;
            ++voting_heads;
        }
        suite.check(voting_heads != 0u && std::isfinite(late_mean_sum) && late_mean_sum < early_mean_sum,
                    "[{}] mean loss fell over {} steps: late-half {} want < early-half {} ({} heads voting) (D)", label,
                    config.steps, late_mean_sum, early_mean_sum, voting_heads);

        // E — every live head scored at least once: the keyed partition fed every modality rows, and every
        //     head actually trained.
        suite.check(end_stats[0].scored_steps > 0u, "[{}] the token head trained (E)", label);
        if (config.batch.contains(modality_t::image_k))
            suite.check(end_stats[1].scored_steps > 0u, "[{}] the image head trained (E)", label);
        if (config.batch.contains(modality_t::video_k))
            suite.check(end_stats[2].scored_steps > 0u, "[{}] the video head trained (E)", label);
        if (config.batch.contains(modality_t::audio_k))
            suite.check(end_stats[3].scored_steps > 0u, "[{}] the audio head trained (E)", label);

        // F — the captured window moved the weights off their seed: gradients propagated through the whole
        //     backbone and the optimizer applied them.
        suite.check(!snapshots_match_(seed_lead.value_, captured_lead),
                    "[{}] the captured window moved the weights off their seed (F)", label);
    }

    // C — capture == eager: a fresh trainer of the same seed starts from the same weights and fresh moments, runs
    //     the same steps eagerly, and lands on the captured run's weights. The strongest single check.
    trainer_config_t eager_config = config;
    eager_config.capture = eager_k;
    trainer_t eager;
    if (expected_t built = make_trainer_(eager, tier, eager_config, health); !built)
        return suite.check(false, "[{}] eager trainer bring-up: {}", label, built.description_);
    if (expected_t ran = run_steps_(eager, 0u, config.steps); !ran)
        return suite.check(false, "[{}] eager run: {}", label, ran.description_);
    if (expected_t drained = eager.synchronize(); !drained)
        return suite.check(false, "[{}] eager drain: {}", label, drained.description_);
    expected<array<bf16_t>> eager_lead = snapshot_weights_(eager.replicas[0].weights, eager.streams[0]);
    suite.check(eager_lead && snapshots_near_(eager_lead.value_, captured_lead),
                "[{}] captured replay lands on the eager run (C)", label);
}

/** One sweep cell: bring up an eager trainer, run two steps, then capture and replay two more over the same
 *  trainer — both model-region modes under one mixture — asserting finite losses and the final-norm extents every
 *  head reconstructs against. */
static void run_sweep_cell_(test_suite_t &suite, world_health_t &health, tier_t tier, sz::string_view_t label,
                            trainer_config_t config) {
    if (!world_still_healthy_(suite, health)) return suite.skip(world_unhealthy_k);
    config.capture = eager_k;
    trainer_t trainer;
    if (expected_t built = make_trainer_(trainer, tier, config, health); !built)
        return suite.check(false, "[{}] trainer bring-up: {}", label, built.description_);

    // The eager state first, then the captured one over the same trainer — capture must also hold after the
    // optimizer has stepped (the capture-stable addresses survive weight updates).
    if (expected_t ran = run_steps_(trainer, 0u, 2u); !ran)
        return suite.check(false, "[{}] eager steps: {}", label, ran.description_);
    f32_t const eager_loss = trainer.telemetry(optimizer_step_t {1u}).composite();
    if (expected_t captured = trainer.capture(); !captured)
        return suite.check(false, "[{}] capture: {}", label, captured.description_);
    if (expected_t ran = run_steps_(trainer, 2u, 4u); !ran)
        return suite.check(false, "[{}] replayed steps: {}", label, ran.description_);
    f32_t const replayed_loss = trainer.telemetry(optimizer_step_t {3u}).composite();

    suite.check(std::isfinite(eager_loss) && std::isfinite(replayed_loss),
                "[{}] losses finite through both modes: eager {} replayed {}", label, eager_loss, replayed_loss);
    std::size_t const max_tokens = static_cast<std::size_t>(config.batch.max_rows_per_gpu) *
                                   config.batch.max_columns_per_row;
    bf16r2_view_t const normalized = trainer.replicas[0].tower.normalized();
    suite.check(normalized.extent(0) == max_tokens && normalized.extent(1) == config.hidden,
                "[{}] normalized carries [max_tokens, hidden]: got [{}, {}] want [{}, {}]", label, normalized.extent(0),
                normalized.extent(1), max_tokens, config.hidden);

    if (expected_t drained = trainer.synchronize(); !drained)
        suite.check(false, "[{}] end-of-case drain: {}", label, drained.description_);
}

/** One sweep cell's config: @p subset 's media bits over @p rows full-width rows of @p columns cells. */
[[nodiscard]] static trainer_config_t sweep_config_(u32_t subset, u32_t rows, u32_t columns) noexcept {
    trainer_config_t config = test_geometry_(false);
    config.batch = multimodal_batch_shape_t {rows, rows, columns};
    config.steps = 4;
    if (subset & 1u) config.batch.image = image_batch_shape_t {8u, 3u};
    if (subset & 2u) config.batch.video = video_batch_shape_t {8u, 3u, 2u, 1u};
    if (subset & 4u) config.batch.audio = audio_batch_shape_t {256u, 16000u};
    return config;
}

/**
 *  @brief The two-tier sweep.
 *
 *  The cheap tier walks every media subset (2³) crossed with one-row and two-row budgets on the one-GPU mesh —
 *  every modality-presence combination, the zero-live-rows corner, and the 1-member peer-read domain — plus two
 *  corners the cross-product never reaches: an omni cell with five rows over four live modalities (the uneven
 *  partition, whose extra row rotates across epochs) and an omni cell at 96 columns (the non-square patch
 *  factorization). The wide tier reruns every subset at the bootstrapped world's full width, so shape handling
 *  composes with the real collectives — and widens to the multi-node world automatically under the sbatch launcher.
 */
static void mixture_sweep(test_suite_t &suite, world_health_t &health, tier_t wide, tier_t cheap) {
    for (u32_t subset = 0; subset != 8u; ++subset)
        for (u32_t rows = 1; rows != 3u; ++rows) {
            sz::string_t label;
            if (!format_into(label, "sweep/1gpu/tokens{}{}{}/rows={}", (subset & 1u) ? ",image" : "",
                             (subset & 2u) ? ",video" : "", (subset & 4u) ? ",audio" : "", rows))
                return suite.check(false, "sweep label format failed");
            suite.run(label.view(), run_sweep_cell_, health, cheap, label.view(), sweep_config_(subset, rows, 64u));
        }
    suite.run("sweep/1gpu/omni/rows=5", run_sweep_cell_, health, cheap, "sweep/1gpu/omni/rows=5",
              sweep_config_(7u, 5u, 64u));
    suite.run("sweep/1gpu/omni/rows=2,columns=96", run_sweep_cell_, health, cheap, "sweep/1gpu/omni/rows=2,columns=96",
              sweep_config_(7u, 2u, 96u));
    trainer_config_t tuned = sweep_config_(7u, 2u, 64u);
    tuned.algo_pick = algo_pick_t::benchmarked_k;
    suite.run("sweep/1gpu/omni/rows=2,tuned", run_sweep_cell_, health, cheap, "sweep/1gpu/omni/rows=2,tuned", tuned);
    for (u32_t subset = 0; subset != 8u; ++subset) {
        sz::string_t label;
        if (!format_into(label, "sweep/wide/tokens{}{}{}/rows=2", (subset & 1u) ? ",image" : "",
                         (subset & 2u) ? ",video" : "", (subset & 4u) ? ",audio" : ""))
            return suite.check(false, "sweep label format failed");
        suite.run(label.view(), run_sweep_cell_, health, wide, label.view(), sweep_config_(subset, 2u, 64u));
    }
}

/** The capture contract holds, one forward and one backward replay graph per replica build, and two forward replays
 *  over one fixed eager preamble produce the same shared @c d(normalized) up to reduction order. */
static void graph_capture(test_suite_t &suite, world_health_t &health, tier_t tier) {
    if (!world_still_healthy_(suite, health)) return suite.skip(world_unhealthy_k);
    trainer_config_t const config = test_geometry_(false);
    trainer_t trainer;
    if (expected_t built = make_trainer_(trainer, tier, config, health); !built)
        return suite.check(false, "capture trainer bring-up: {}", built.description_);
    std::size_t const gpus = trainer.replicas.size();
    suite.check(trainer.forward_graphs.size() == gpus && trainer.backward_graphs.size() == gpus,
                "one captured forward and backward region per replica: got {} and {} want {}",
                trainer.forward_graphs.size(), trainer.backward_graphs.size(), gpus);
    if (trainer.forward_graphs.size() != gpus) return;

    switch_to_gpu_t gpu_scope {trainer.streams[0]};
    if (!gpu_scope.ok()) return suite.check(false, "capture context entry");
    base_replica_t &lead = trainer.replicas[0];
    expected_t prepared = lead.draw(config, 0u, data_epoch_t {0u}, trainer.streams[0]);
    if (prepared) prepared = lead.prepare(config, data_epoch_t {0u}, trainer.streams[0]);
    if (!prepared) return suite.check(false, "capture preamble: {}", prepared.description_);
    if (expected_t first = trainer.forward_graphs[0].launch(trainer.streams[0].get()); !first)
        return suite.check(false, "first replay: {}", first.description_);
    expected<array<bf16_t>> first_grad = snapshot_grad_(lead, trainer.streams[0]);
    if (expected_t second = trainer.forward_graphs[0].launch(trainer.streams[0].get()); !second)
        return suite.check(false, "second replay: {}", second.description_);
    expected<array<bf16_t>> second_grad = snapshot_grad_(lead, trainer.streams[0]);
    suite.check(first_grad && second_grad && snapshots_near_(first_grad.value_, second_grad.value_),
                "two captured replays agree");
}

int main(int, char **argv) {
    // Lazy module loading deadlocks against the InfiniBand doorbell kernels' flag spins: the spinning
    // wait blocks its context, the peer's next launch blocks on module loading behind it, and neither
    // side progresses. Eager loading before `cuInit` front-loads every module — the mixer suite's fix.
    ::setenv("CUDA_MODULE_LOADING", "EAGER", 1);

    test_suite_t suite = test_suite_t::make(argv[0]);

    // The schedule's properties are pure host math — asserted before any GPU work.
    suite.run("learning_rate_schedule", learning_rate_schedule);

    int device_count = 0;
    if (cuInit(0) != CUDA_SUCCESS || cuDeviceGetCount(&device_count) != CUDA_SUCCESS || device_count <= 0) {
        suite.skip("capture/replay", no_cuda_device_k);
        suite.skip("conformance/tokens", no_cuda_device_k);
        suite.skip("conformance/omni", no_cuda_device_k);
        suite.skip("sweep", no_cuda_device_k);
        return suite.finish();
    }
    array<i32_t> devices;
    for (int device = 0; device != device_count; ++device)
        if (devices.push_back(static_cast<i32_t>(device)) != status_t::success_k) {
            suite.check(false, "device table allocation failed");
            return suite.finish();
        }

    // One process per host_domain; the sbatch launcher raises UFORM_HOST_DOMAIN_COUNT for the two-node tier and
    // names the exchange, which this main maps onto the mesh options, as the library reads no environment.
    mesh_options_t options;
    options.host_domain_index = host_domain_index_from_environment();
    options.host_domain_count = host_domain_count_from_environment();
    options.timeout_seconds = test_rendezvous_timeout_seconds_k;
    if (char const *directory = std::getenv("UFORM_EXCHANGE_DIR"); directory != nullptr)
        options.directory = sz::string_view_t {directory, std::strlen(directory)};
    if (char const *job = std::getenv("UFORM_JOB_ID"); job != nullptr)
        options.job = sz::string_view_t {job, std::strlen(job)};
    mesh_t wide_mesh;
    if (expected_t joined = mesh_t::make(wide_mesh, options, devices.view()); !joined) {
        suite.check(false, "mesh bootstrap: {}", joined.description_);
        return suite.finish();
    }
    mesh_t cheap_mesh;
    if (expected_t narrowed = narrow_mesh_(cheap_mesh, wide_mesh.topology, 1u); !narrowed) {
        suite.check(false, "narrowed mesh: {}", narrowed.description_);
        return suite.finish();
    }
    tier_t const wide {wide_mesh, devices.view()};
    tier_t const cheap {cheap_mesh, devices.view().first(1)};

    world_health_t health;
    suite.run("capture/replay", graph_capture, health, cheap);
    suite.run("conformance/tokens", run_base_conformance, health, "tokens", test_geometry_(false), wide);
    suite.run("conformance/omni", run_base_conformance, health, "omni", test_geometry_(true), wide);
    mixture_sweep(suite, health, wide, cheap);
    return suite.finish();
}
