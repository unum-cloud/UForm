/**
 *  @file test/base.cu
 *  @author Ash Vardanian
 *  @date July 6, 2026
 *  @brief The Base training conformance test: builds the same @ref trainer_t the binary and the C
 *      verbs drive and asserts the invariants the loop never checks — deep replay conformance for
 *      tokens and omni mixtures, and a sweep over media subsets, row budgets, model-region modes.
 *
 *  This is a test: a plain @c int @c main runs every case through @ref run_test_case , consuming
 *  @c uform/train/trainer.hpp 's typed surface instead of copying the driver. Two meshes are made
 *  once in @c main — the bootstrapped world and a one-GPU narrowing of it — and every case builds
 *  its own trainer over one of them, exactly as the C verbs compose production. Launched under
 *  Slurm via @c scripts/slurm/test_base-4-8-16.sbatch , the same binary runs the distributed
 *  tiers: @c main honors @c UFORM_HOST_DOMAIN_COUNT , so a two-node world exercises the InfiniBand
 *  reduce under every case below.
 *
 *  The case matrix:
 *
 *      - @ref test_learning_rate_schedule — the warmup / cosine policy's properties, pure host
 *        math, no GPU.
 *      - @ref test_base_conformance covers the tokens and omni mixtures at full width.
 *      - @ref mixture_sweep covers every media subset on the one-GPU mesh, then at full width.
 *      - @ref test_graph_capture — the capture contract holds and two replays agree, on the
 *        one-GPU mesh.
 *      - @ref test_choices_replay — choices alone replay through the items source: their term
 *        falls.
 *      - @ref test_spectrum_spike : the pooled-code regularizers ride a contrastive step, and a
 *        forced gradient spike refuses the next step with the weights untouched.
 *      - @ref test_spectrum_world : on two or more GPUs every rank reaches the same coding rate
 *        bitwise, matching one GPU holding every row; a one-GPU world skips it.
 *
 *  @ref test_base_conformance checks six things at full width: @b A every replica bitwise-matches
 *  the lead, @b B every member's barrier round advanced and they all agree, @b C an eager trainer
 *  of the same seed lands on the captured run, @b D each head's loss fell between its first and
 *  last scored steps, @b E every live head's loss read nonzero at least once, @b F the captured
 *  window moved the weights off their seed.
 *
 *  @ref mixture_sweep runs two width tiers: every media subset (2³) crossed with one-row and
 *  two-row budgets on the one-GPU mesh (plus the uneven five-row and 96-column omni corners), then
 *  every subset again at the bootstrapped world's full width. Each cell steps eagerly, then
 *  captures and replays over the same trainer, so both model-region modes run under every mixture.
 */
#undef NDEBUG // ! A test's oracle must stay live in every build

#include <cmath>   // `std::isfinite`
#include <cstdlib> // `std::getenv`, `std::strtod`
#include <cstring> // `std::memcmp`, `std::memcpy`, `std::strlen`

#include <span>   // `std::span` — device lists
#include <vector> // `std::vector` — the world's pooled rows and coding rates

#include <cuda.h> // `cuInit`, `cuStreamSynchronize`

#include "uform/train/trainer.hpp" // `trainer_config_t`, `trainer_t`, `mesh_t`, `synthetic_source_t`

#include "train/numerics.hpp" // `gpu_numerics_of`

#include "harness.hpp" // `run_test_case`, `expect_true`

namespace ue = unum::uform::encoder;
using ue::audio_batch_shape_t;
using ue::image_batch_shape_t;
using ue::modality_t;
using ue::multimodal_batch_shape_t;
using ue::video_batch_shape_t;
namespace util = unum::uform::utilities;
using namespace unum::uform::train;
using namespace unum::uform::test;

/** The data a test geometry batches: text tokens alone, or every media slice as well. */
enum class test_mixture_t : bool { tokens_k, omni_k };

/**
 *  @brief The tiny shared geometry — @c depth=2 , @c hidden=256 , @c 16 steps, and a two-step
 *      warmup so even the tiny run crosses both schedule phases.
 *
 *  Tokens ride a @c 2×64 rectangle; the omni variant fills every media slice over a @c 4×64 one, so
 *  the keyed partition holds all four modalities live at every step and the composite loss measures
 *  the same mixture throughout — descent stays comparable. Sixteen steps, because descent is
 *  asserted on the @b world mean: at a 16-GPU world the data spreads twice as wide as one box's,
 *  and the eight-step horizon left the token head's early-vs-late comparison genuinely marginal (a
 *  ~0.7% rise on the world mean) while every machinery invariant held.
 */
[[nodiscard]] static trainer_config_t test_geometry_(test_mixture_t mixture, util::u64_t seed) noexcept {
    trainer_config_t config;
    config.seed = seed;
    config.batch = multimodal_batch_shape_t {2u, 2u, 64u};
    if (mixture == test_mixture_t::omni_k) {
        config.batch = multimodal_batch_shape_t {4u, 4u, 64u};
        config.batch.image = image_batch_shape_t {8u, 3u};
        config.batch.video = video_batch_shape_t {8u, 3u, 2u, 1u};
        config.batch.audio = audio_batch_shape_t {256u, 16000u};
    }
    config.depth = 2;
    config.hidden = 256;
    config.steps = 16;
    config.warmup_steps = 2;
    // Sixteen toy-width steps need the rate the descent checks were calibrated at, not a tier's.
    config.learning_rate = 5.0e-3f;
    config.report_every = 1; // the cases read telemetry after every step, so the summands hold that step alone
    config.telemetry_channels = telemetry_losses_k | telemetry_signals_k | telemetry_rounds_k;
    return config;
}

/** One width tier: the mesh every cell of the tier builds its trainer over, the executor whose
 *  streams it queues on, the devices it drives, and the kernels they call. */
struct tier_t {
    mesh_t& mesh;
    util::cuda_executor_t& streams;
    std::span<util::i32_t const> devices;
    util::gpu_numerics_t numerics;
};

/**
 *  @brief Narrow the bootstrapped @p world into @p placed : this process alone drives its first
 *      @p gpus GPUs as an isolated one-host-domain mesh — the sweep's cheap tier and the
 *      capture case's world.
 *
 *  Safe because a single-host topology has no rendezvous state; the one cross-field invariant
 *  is that @c host_domains sizes to @c shape.host_domain_count() .
 */
[[nodiscard]] static util::expected_t narrow_mesh_(mesh_t& placed, util::mesh_topology_t const& world,
                                                   util::u32_t gpus) noexcept {
    placed.topology.shape.gpus_per_host_domain = gpus;
    placed.topology.shape.gpus_per_multimem_domain = gpus;
    placed.topology.shape.gpus_per_colocated_domain = gpus;
    placed.topology.host_domain_index = 0;
    if (st::failed(placed.topology.host_domains.resize(1)))
        return util::expected_t::fail(util::status_code_t::out_of_memory_k,
                                      "narrowed host-domain table allocation failed");
    placed.topology.host_domains[0] = world.host_domains[world.host_domain_index];
    placed.topology.host_domains[0].host_domain_index = 0;
    placed.topology.host_domains[0].device_count = gpus;
    auto [colocated, status] = util::colocated_domain_t::make(placed.topology);
    if (!status) return util::expected_t::fail(std::move(status));
    placed.colocated = std::move(colocated);
    return {};
}

/** The world's health, owned by @c main and threaded by reference into every case:
 *  a device-side fault poisons the local context (every later case fails with the same unrelated
 *  "unspecified launch failure"), and a dead peer domain makes every later rendezvous wait its full
 *  deadline — either way the first detection reports once and the remaining cases return
 *  immediately, so a faulted run finishes in seconds instead of minutes. */
struct world_health_t {

    /** Cleared at the first poisoned-context or unresponsive-peer detection; later cases skip. */
    bool alive = true;

    /** Consecutive trainer bring-up failures — a healthy world never fails one, so two in a row
     *  mean the peer domain is gone. */
    util::u32_t consecutive_setup_failures = 0;
};

/** The reason every case after the first unhealthy detection skips with. */
inline constexpr char const* world_unhealthy_k = "needs a healthy world";

/**
 *  @brief Probe @p health before a case: report the first poisoned-context or unresponsive-peer
 *      detection once, and tell every later case to return immediately.
 *  @param[in,out] health The world's health record, cleared at the first detection.
 */
[[nodiscard]] static bool world_still_healthy_(world_health_t& health) noexcept {
    if (!health.alive) return false;
    if (health.consecutive_setup_failures >= 2u) {
        health.alive = false;
        expect_true(false, "peer domain unresponsive — aborting the remaining cases");
        return false;
    }
    int const lead[] = {0};
    auto [probe, status] = spawn_test_executor(lead);
    if (status) return true;
    health.alive = false;
    expect_true(false, "device context poisoned — aborting the remaining cases: {}", status.description_);
    return false;
}

/** Build @p placed over @p tier for @p config , tallying bring-up failures into @p health . */
[[nodiscard]] static util::expected_t make_trainer_(trainer_t& placed, tier_t tier, trainer_config_t const& config,
                                                    world_health_t& health) noexcept {
    util::expected_t built = trainer_t::make(placed, tier.mesh, tier.streams, config, tier.devices, tier.numerics);
    health.consecutive_setup_failures = built ? 0u : health.consecutive_setup_failures + 1u;
    return built;
}

/** Enqueue steps @b [begin,end) of the synthetic loop. */
[[nodiscard]] static util::expected_t run_steps_(trainer_t& trainer, util::u32_t begin, util::u32_t end) noexcept {
    synthetic_source_t source;
    step_report_t report;
    for (util::u32_t ordinal = begin; ordinal != end; ++ordinal)
        if (util::expected_t stepped = trainer.step(batch_source_of(source), util::optimizer_step_t {ordinal}, report);
            !stepped)
            return stepped;
    return {};
}

/** DtoH one replica's whole registration list into a byte-comparable host array — the snapshot
 *  behind assertions A, C, and the gradient-propagation check. */
[[nodiscard]] static util::expected<st::basic_vector<util::bf16_t>> snapshot_weights_(
    st::basic_vector<util::differentiable_weights_t> const& weights, util::executor_stream_t& stream) noexcept {
    using result_t = util::expected<st::basic_vector<util::bf16_t>>;
    [[maybe_unused]] util::status_t const gpu_scope = (stream).bind();
    if (gpu_scope.failed()) return result_t::fail(util::status_code_t::device_k, "context entry failed");
    std::span<util::differentiable_weights_t const> const list = weights;
    std::size_t total = 0, largest = 0;
    for (util::differentiable_weights_t const& matrix : list) {
        std::size_t const scalars = matrix.weights.numel();
        total += scalars;
        largest = fu::max_of(largest, scalars);
    }
    st::basic_vector<util::bf16_t> host;
    if (st::failed(host.resize(st::assume_uninitialized, total)))
        return result_t::fail(util::status_code_t::out_of_memory_k, "snapshot allocation failed");
    nk::tensor<util::bf16_t, util::pinned_alloc<util::bf16_t>, 1> staging =
        make_uninitialized<nk::tensor<util::bf16_t, util::pinned_alloc<util::bf16_t>, 1>>({largest});
    if (!staging) return result_t::fail(util::status_code_t::out_of_memory_k, "pinned staging allocation failed");
    std::size_t cursor = 0;
    for (util::differentiable_weights_t const& matrix : list) {
        std::size_t const scalars = matrix.weights.numel();
        if (nk::failed(staging.resize({scalars})))
            return result_t::fail(util::status_code_t::out_of_memory_k, "staging resize failed");
        nk::tensor_view<util::bf16_t, 1> const source = matrix.weights.flatten<1>();
        if (util::expected_t copied = util::copy_device_to_host(source, staging.span(), stream); !copied)
            return result_t::fail(std::move(copied));
        cuStreamSynchronize(util::native_stream(stream));
        std::memcpy(host.data() + cursor, staging.data(), scalars * sizeof(util::bf16_t));
        cursor += scalars;
    }
    return result_t {std::move(host)};
}

/** DtoH the lead replica's shared @c d(normalized) - the score output whose two captured
 *  replays must agree. */
[[nodiscard]] static util::expected<st::basic_vector<util::bf16_t>> snapshot_grad_(
    base_replica_t& replica, util::executor_stream_t& stream) noexcept {
    using result_t = util::expected<st::basic_vector<util::bf16_t>>;
    [[maybe_unused]] util::status_t const gpu_scope = (stream).bind();
    if (gpu_scope.failed()) return result_t::fail(util::status_code_t::device_k, "context entry failed");
    util::bf16r2_view_t const view = replica.normalized_grad.view();
    std::size_t const scalars = static_cast<std::size_t>(view.extent(0)) * view.extent(1);
    nk::tensor<util::bf16_t, util::pinned_alloc<util::bf16_t>, 1> staging =
        make_uninitialized<nk::tensor<util::bf16_t, util::pinned_alloc<util::bf16_t>, 1>>({scalars});
    if (!staging) return result_t::fail(util::status_code_t::out_of_memory_k, "pinned staging allocation failed");
    if (util::expected_t copied = util::copy_device_to_host(view.flatten<1>(), staging.span(), stream); !copied)
        return result_t::fail(std::move(copied));
    cuStreamSynchronize(util::native_stream(stream));
    st::basic_vector<util::bf16_t> host;
    if (st::failed(host.resize(st::assume_uninitialized, scalars)))
        return result_t::fail(util::status_code_t::out_of_memory_k, "snapshot allocation failed");
    std::memcpy(host.data(), staging.data(), scalars * sizeof(util::bf16_t));
    return result_t {std::move(host)};
}

/** Bytewise-equal two snapshots, as replicas fed one all-reduced gradient must agree exactly. */
[[nodiscard]] static bool snapshots_match_(st::basic_vector<util::bf16_t> const& left,
                                           st::basic_vector<util::bf16_t> const& right) noexcept {
    return left.size() == right.size() &&
           std::memcmp(left.data(), right.data(), left.size() * sizeof(util::bf16_t)) == 0;
}

/** Two snapshots agree within 1% of the first's peak, for runs whose reductions may add
 *  in any order. */
[[nodiscard]] static bool snapshots_near_(st::basic_vector<util::bf16_t> const& left,
                                          st::basic_vector<util::bf16_t> const& right) noexcept {
    if (left.size() != right.size()) return false;
    util::f32_t largest = 0.0f, worst = 0.0f;
    for (std::size_t index = 0; index != left.size(); ++index) {
        util::f32_t const expected_value = __bfloat162float(left[index]);
        largest = fu::max_of(largest, std::fabs(expected_value));
        worst = fu::max_of(worst, std::fabs(expected_value - __bfloat162float(right[index])));
    }
    return worst <= 1.0e-2f * largest;
}

/** The warmup / cosine policy's own properties, pure host math: a strictly monotone ramp topping
 *  out exactly at the peak, a never-rising decay bounded by @b [floor, peak] , and a tail settling
 *  onto the floor. */
static void test_learning_rate_schedule() {
    trainer_config_t config;
    config.seed = test_run.settings.seed;
    config.learning_rate = 1.0e-3f;
    config.min_learning_rate = 1.0e-5f;
    config.warmup_steps = 10;
    config.steps = 100;

    bool ramp_monotone = true;
    for (util::u32_t ordinal = 1; ordinal < config.warmup_steps; ++ordinal)
        ramp_monotone = ramp_monotone && config.learning_rate_for(util::optimizer_step_t {ordinal}) >
                                             config.learning_rate_for(util::optimizer_step_t {ordinal - 1u});
    expect_true(ramp_monotone, "the warmup ramp is strictly monotone");
    util::f32_t const peak = config.learning_rate_for(util::optimizer_step_t {config.warmup_steps - 1u});
    expect_true(peak == config.learning_rate, "the ramp tops out exactly at the peak: got {} want {}", peak,
                config.learning_rate);

    bool decay_monotone = true, bounded = true;
    util::f32_t previous = peak;
    for (util::u32_t ordinal = config.warmup_steps; ordinal < config.steps; ++ordinal) {
        util::f32_t const rate = config.learning_rate_for(util::optimizer_step_t {ordinal});
        decay_monotone = decay_monotone && rate <= previous;
        bounded = bounded && rate >= config.min_learning_rate && rate <= config.learning_rate;
        previous = rate;
    }
    expect_true(decay_monotone, "the cosine decay never rises");
    expect_true(bounded, "every decay rate stays within the floor-to-peak band");
    util::f32_t const tail = config.learning_rate_for(util::optimizer_step_t {config.steps - 1u});
    expect_true(tail <= config.min_learning_rate * 1.1f, "the tail settles onto the floor: got {} want <= {}", tail,
                config.min_learning_rate * 1.1f);
}

/** The checkpoint naming contract over one built replica: the registration list holds the semantic
 *  per-weight names every SafeTensors parser keys by (under the driver's @c model. prefix), derived
 *  from @p config and asserted in full order — a @c fill reorder or rename fails loudly here before
 *  it silently re-keys a checkpoint. Piggybacks on an already-built trainer, so it costs no
 *  bring-up of its own. */
static void check_weight_naming_(char const* label, trainer_config_t const& config,
                                 std::span<util::differentiable_weights_t const> weights) {
    std::size_t index = 0;
    auto expect = [&](sz::string_view_t want) {
        if (index >= weights.size()) {
            expect_true(false, "[{}] weight {} missing: want {}", label, index, want);
            return;
        }
        expect_true(weights[index].name.view() == want, "[{}] weight {}: got {} want {}", label, index,
                    weights[index].name.view(), want);
        ++index;
    };
    expect("embedding");
    if (config.batch.contains(modality_t::image_k)) expect("image_projection");
    if (config.batch.contains(modality_t::video_k)) expect("video_projection");
    if (config.batch.contains(modality_t::audio_k)) expect("audio_projection");
    sz::string_t name;
    for (util::u32_t layer_index = 0; layer_index != config.depth; ++layer_index)
        for (sz::string_view_t part : {"qkv", "out", "gate_up", "down", "pre_norm", "post_norm"}) {
            name.clear();
            if (!expect_true(util::format_into(name, "blocks.{}.{}", layer_index, part),
                             "[{}] weight name format failed", label))
                return;
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
    expect_true(index == weights.size(), "[{}] registration count: got {} want {}", label, weights.size(), index);
}

/** Bring up a captured trainer for one mixture, run its loop reading telemetry every step, and fold
 *  the invariants A, B, D, E and F into the run; then an eager trainer of the same seed checks C.
 *  D and E read the ledger's running loss statistics — every local member's sample mass, not a
 *  lead-only coin flip. */
static case_result_t test_base_conformance(world_health_t& health, char const* label, trainer_config_t const& config,
                                           tier_t tier) {
    if (!world_still_healthy_(health)) return case_result_t::skipped(world_unhealthy_k);
    util::mesh_topology_t const& topology = tier.mesh.topology;
    st::basic_vector<util::bf16_t> captured_lead;
    {
        trainer_t trainer;
        if (util::expected_t built = make_trainer_(trainer, tier, config, health);
            !expect_true(built, "[{}] trainer bring-up: {}", label, built.description_))
            return {};
        util::u32_t const gpus = static_cast<util::u32_t>(trainer.replicas.size());
        check_weight_naming_(label, config, trainer.replicas[0].weights);

        // Assertion F's reference: the lead replica's freshly seeded weights, untouched
        // by the capture.
        auto [seed_lead, snapshot_seed_lead_status] = snapshot_weights_(trainer.replicas[0].weights,
                                                                        trainer.streams->stream(0));
        if (!expect_true(snapshot_seed_lead_status, "[{}] seed snapshot: {}", label,
                         snapshot_seed_lead_status.description_))
            return {};

        // The captured run, telemetry every step. A midpoint snapshot splits the run into two
        // halves; assertion D compares their sample-weighted means, so a head scored on only some
        // steps contributes its whole sample mass, not one noisy endpoint.
        telemetry_probe_t const head_probes[4] = {trainer.probes.token, trainer.probes.image, trainer.probes.video,
                                                  trainer.probes.audio};
        telemetry_loss_stats_t mid_stats[4] {};
        synthetic_source_t source;
        step_report_t report;
        for (util::u32_t ordinal = 0; ordinal != config.steps; ++ordinal) {
            if (util::expected_t ran = trainer.step(batch_source_of(source), util::optimizer_step_t {ordinal}, report);
                !expect_true(ran, "[{}] captured step {}: {}", label, ordinal, ran.description_))
                return {};
            [[maybe_unused]] telemetry_t const drained = trainer.telemetry(util::optimizer_step_t {ordinal});
            if (ordinal + 1u == config.steps / 2u)
                for (util::u32_t head = 0; head != 4u; ++head)
                    mid_stats[head] = trainer.ledger.losses(head_probes[head]);
        }
        telemetry_loss_stats_t end_stats[4] {};
        for (util::u32_t head = 0; head != 4u; ++head) end_stats[head] = trainer.ledger.losses(head_probes[head]);

        // At multi-domain worlds a node's telemetry sees only its slice of the objective the
        // optimizer minimizes — a local half can rise while the world mean falls, and that is
        // descent, not a defect. Fold every host_domain's summands through the mesh exchange, so D
        // judges the world mean and every node reaches the same verdict.
        util::u32_t const host_domains = static_cast<util::u32_t>(topology.host_domains.size());
        if (host_domains > 1u) {
            sz::string_t mine;
            bool formatted = true;
            for (util::u32_t head = 0; head != 4u; ++head)
                formatted = formatted && util::format_into(mine, "{:.17e} {:.17e} {:.17e} {:.17e} {}\n",
                                                           mid_stats[head].weighted_sum, mid_stats[head].weight,
                                                           end_stats[head].weighted_sum, end_stats[head].weight,
                                                           end_stats[head].scored_steps);
            sz::string_t key;
            formatted = formatted && static_cast<bool>(util::format_into(key, "conformance.{}.loss_summands", label));
            if (!expect_true(formatted, "[{}] loss-summand exchange formatting failed", label)) return {};
            if (util::expected_t published = topology.publish_text(key.view(), mine.view());
                !expect_true(published, "[{}] loss-summand publish: {}", label, published.description_))
                return {};
            telemetry_loss_stats_t world_mid[4] {}, world_end[4] {};
            for (util::u32_t domain_index = 0; domain_index != host_domains; ++domain_index) {
                sz::string_t text;
                if (domain_index == topology.host_domain_index) {
                    if (!expect_true(!sz::failed(text.assign(mine.view())), "[{}] loss-summand staging failed", label))
                        return {};
                }
                else if (util::expected_t arrived = topology.await_text(domain_index, key.view(), text,
                                                                        wait_limit_k.count());
                         !expect_true(arrived, "[{}] loss-summand exchange: {}", label, arrived.description_))
                    return {};
                char const* cursor = text.c_str();
                for (util::u32_t head = 0; head != 4u; ++head) {
                    char* next = nullptr;
                    world_mid[head].weighted_sum += std::strtod(cursor, &next); // vendor-boundary: libc parse
                    world_mid[head].weight += std::strtod(next, &next);         // vendor-boundary
                    world_end[head].weighted_sum += std::strtod(next, &next);   // vendor-boundary
                    world_end[head].weight += std::strtod(next, &next);         // vendor-boundary
                    world_end[head].scored_steps += static_cast<util::u64_t>(
                        std::strtod(next, &next)); // vendor-boundary
                    cursor = next;
                }
            }
            for (util::u32_t head = 0; head != 4u; ++head) {
                mid_stats[head] = world_mid[head];
                end_stats[head] = world_end[head];
            }
        }
        if (util::expected_t drained = trainer.synchronize();
            !expect_true(drained, "[{}] drain: {}", label, drained.description_))
            return {};
        auto [lead, snapshot_lead_status] = snapshot_weights_(trainer.replicas[0].weights, trainer.streams->stream(0));
        if (!expect_true(snapshot_lead_status, "[{}] lead snapshot: {}", label, snapshot_lead_status.description_))
            return {};
        captured_lead = std::move(lead);

        // A — every replica byte-matches the lead. Vacuous at one GPU (the peer loop is empty), but
        // the run stands.
        bool replicas_identical = true;
        for (util::u32_t member_index = 1; member_index < gpus; ++member_index) {
            auto [peer, snapshot_peer_status] = snapshot_weights_(trainer.replicas[member_index].weights,
                                                                  trainer.streams->stream(member_index));
            replicas_identical = replicas_identical && snapshot_peer_status && snapshots_match_(peer, captured_lead);
        }
        expect_true(replicas_identical, "[{}] every replica matches the lead bitwise (A)", label);

        // B — every member's own barrier round advanced off zero, never moved backwards, and they
        // all agree: the ledger's monotone round rows hold all three (each member's scalar counts
        // its own barriers).
        auto [lead_round, read_lead_round_status] = trainer.ledger.row(trainer.probes.rounds[0],
                                                                       trainer.streams->stream(0));
        bool rounds_read = static_cast<bool>(read_lead_round_status), round_uniform = true, round_monotone = true;
        for (util::u32_t member_index = 0; member_index < gpus; ++member_index) {
            auto [member_round, read_member_round_status] = trainer.ledger.row(trainer.probes.rounds[member_index],
                                                                               trainer.streams->stream(member_index));
            rounds_read = rounds_read && read_member_round_status;
            round_uniform = round_uniform && read_member_round_status && read_lead_round_status &&
                            member_round.last_scalar == lead_round.last_scalar;
            round_monotone = round_monotone && read_member_round_status && member_round.violations == 0u;
        }
        expect_true(rounds_read && lead_round.last_scalar > 0.0f, "[{}] barrier round advanced off zero: got {} (B)",
                    label, rounds_read ? lead_round.last_scalar : 0.0f);
        expect_true(round_uniform, "[{}] barrier round reads identically through every member (B)", label);
        expect_true(round_monotone, "[{}] no member's barrier round ever moved backwards (B)", label);

        // D — per live head, the sample-weighted mean of the run's late half fell below
        // the early half's;
        //     the late half is a subtraction of two monotone snapshots. A head both halves scored votes;
        //     the sums of voting heads decide. Loose by design — one deterministic descent, not convergence.
        util::f64_t early_mean_sum = 0.0, late_mean_sum = 0.0;
        util::u32_t voting_heads = 0;
        for (util::u32_t head = 0; head != 4u; ++head) {
            util::f64_t const early_weight = mid_stats[head].weight;
            util::f64_t const late_weight = end_stats[head].weight - mid_stats[head].weight;
            if (early_weight == 0.0 || late_weight == 0.0) continue; // unscored half — E guards coverage
            early_mean_sum += mid_stats[head].weighted_sum / early_weight;
            late_mean_sum += (end_stats[head].weighted_sum - mid_stats[head].weighted_sum) / late_weight;
            ++voting_heads;
        }
        expect_true(voting_heads != 0u && std::isfinite(late_mean_sum) && late_mean_sum < early_mean_sum,
                    "[{}] mean loss fell over {} steps: late-half {} want < early-half {} ({} heads voting) (D)", label,
                    config.steps, late_mean_sum, early_mean_sum, voting_heads);

        // E — every live head scored at least once: the keyed partition fed every modality
        // rows, and every
        //     head actually trained.
        expect_true(end_stats[0].scored_steps > 0u, "[{}] the token head trained (E)", label);
        if (config.batch.contains(modality_t::image_k))
            expect_true(end_stats[1].scored_steps > 0u, "[{}] the image head trained (E)", label);
        if (config.batch.contains(modality_t::video_k))
            expect_true(end_stats[2].scored_steps > 0u, "[{}] the video head trained (E)", label);
        if (config.batch.contains(modality_t::audio_k))
            expect_true(end_stats[3].scored_steps > 0u, "[{}] the audio head trained (E)", label);

        // F — the captured window moved the weights off their seed: gradients propagated
        // through the whole
        //     backbone and the optimizer applied them.
        expect_true(!snapshots_match_(seed_lead, captured_lead),
                    "[{}] the captured window moved the weights off their seed (F)", label);
    }

    // C — capture == eager: a fresh trainer of the same seed starts from the same weights and
    // fresh moments, runs
    //     the same steps eagerly, and lands on the captured run's weights. The strongest single check.
    trainer_config_t eager_config = config;
    eager_config.capture = util::eager_k;
    trainer_t eager;
    if (util::expected_t built = make_trainer_(eager, tier, eager_config, health);
        !expect_true(built, "[{}] eager trainer bring-up: {}", label, built.description_))
        return {};
    if (util::expected_t ran = run_steps_(eager, 0u, config.steps);
        !expect_true(ran, "[{}] eager run: {}", label, ran.description_))
        return {};
    if (util::expected_t drained = eager.synchronize();
        !expect_true(drained, "[{}] eager drain: {}", label, drained.description_))
        return {};
    auto [eager_lead, status] = snapshot_weights_(eager.replicas[0].weights, eager.streams->stream(0));
    expect_true(status && snapshots_near_(eager_lead, captured_lead), "[{}] captured replay lands on the eager run (C)",
                label);
    return {};
}

/** One sweep cell: bring up an eager trainer, run two steps, then capture and replay two more over
 *  the same trainer — both model-region modes under one mixture — asserting finite losses and the
 *  final-norm extents every head reconstructs against. */
static case_result_t test_sweep_cell(world_health_t& health, tier_t tier, sz::string_view_t label,
                                     trainer_config_t config) {
    if (!world_still_healthy_(health)) return case_result_t::skipped(world_unhealthy_k);
    config.capture = util::eager_k;
    trainer_t trainer;
    if (util::expected_t built = make_trainer_(trainer, tier, config, health);
        !expect_true(built, "[{}] trainer bring-up: {}", label, built.description_))
        return {};

    // The eager state first, then the captured one over the same trainer — capture must also hold
    // after the optimizer has stepped (the capture-stable addresses survive weight updates).
    if (util::expected_t ran = run_steps_(trainer, 0u, 2u);
        !expect_true(ran, "[{}] eager steps: {}", label, ran.description_))
        return {};
    util::f32_t const eager_loss = trainer.telemetry(util::optimizer_step_t {1u}).composite();
    if (util::expected_t captured = trainer.capture();
        !expect_true(captured, "[{}] capture: {}", label, captured.description_))
        return {};
    if (util::expected_t ran = run_steps_(trainer, 2u, 4u);
        !expect_true(ran, "[{}] replayed steps: {}", label, ran.description_))
        return {};
    util::f32_t const replayed_loss = trainer.telemetry(util::optimizer_step_t {3u}).composite();

    expect_true(std::isfinite(eager_loss) && std::isfinite(replayed_loss),
                "[{}] losses finite through both modes: eager {} replayed {}", label, eager_loss, replayed_loss);
    std::size_t const max_tokens = static_cast<std::size_t>(config.batch.max_rows_per_gpu) *
                                   config.batch.max_columns_per_row;
    util::bf16r2_view_t const normalized = trainer.replicas[0].tower.normalized();
    expect_true(normalized.extent(0) == max_tokens && normalized.extent(1) == config.hidden,
                "[{}] normalized spans [max_tokens, hidden]: got [{}, {}] want [{}, {}]", label, normalized.extent(0),
                normalized.extent(1), max_tokens, config.hidden);

    if (util::expected_t drained = trainer.synchronize(); !drained)
        expect_true(false, "[{}] end-of-case drain: {}", label, drained.description_);
    return {};
}

/** One sweep cell's config: @p subset 's media bits over @p rows full-width rows of
 *  @p columns cells. */
[[nodiscard]] static trainer_config_t sweep_config_(util::u32_t subset, util::u32_t rows, util::u32_t columns,
                                                    util::u64_t seed) noexcept {
    trainer_config_t config = test_geometry_(test_mixture_t::tokens_k, seed);
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
 *  The cheap tier walks every media subset (2³) crossed with one-row and two-row budgets on the
 *  one-GPU mesh — every modality-presence combination, the zero-live-rows corner, and the 1-member
 *  peer-read domain — plus two corners the cross-product never reaches: an omni cell with five rows
 *  over four live modalities (the uneven partition, whose extra row rotates across epochs) and an
 *  omni cell at 96 columns (the non-square patch factorization). The wide tier reruns every subset
 *  at the bootstrapped world's full width, so shape handling composes with the real collectives —
 *  and widens to the multi-node world automatically under the sbatch launcher.
 */
static void mixture_sweep(world_health_t& health, tier_t wide, tier_t cheap) {
    for (util::u32_t subset = 0; subset != 8u; ++subset)
        for (util::u32_t rows = 1; rows != 3u; ++rows) {
            sz::string_t label;
            if (!expect_true(util::format_into(label, "test_sweep_cell(cheap, {}, {}, 64)", subset, rows),
                             "sweep label format failed"))
                return;
            run_test_case(label.c_str(), test_sweep_cell, health, cheap, label.view(),
                          sweep_config_(subset, rows, 64u, test_run.settings.seed));
        }
    run_test_case("test_sweep_cell(cheap, 7, 5, 64)", test_sweep_cell, health, cheap,
                  "test_sweep_cell(cheap, 7, 5, 64)", sweep_config_(7u, 5u, 64u, test_run.settings.seed));
    run_test_case("test_sweep_cell(cheap, 7, 2, 96)", test_sweep_cell, health, cheap,
                  "test_sweep_cell(cheap, 7, 2, 96)", sweep_config_(7u, 2u, 96u, test_run.settings.seed));
    for (util::u32_t subset = 0; subset != 8u; ++subset) {
        sz::string_t label;
        if (!expect_true(util::format_into(label, "test_sweep_cell(wide, {}, 2, 64)", subset),
                         "sweep label format failed"))
            return;
        run_test_case(label.c_str(), test_sweep_cell, health, wide, label.view(),
                      sweep_config_(subset, 2u, 64u, test_run.settings.seed));
    }
}

/** @ref batch_capacity keeps a segment per row at least and one per cell at most, and otherwise
 *  gives the rows times the density, rounded up. */
static void test_segment_budget() {
    multimodal_batch_shape_t const shape {0u, 0u, 64u};
    expect_true(batch_capacity(shape, 64u, 3.05).max_segments_per_gpu == 196u,
                "64 rows at 3.05 segments per row hold 196 segments");
    expect_true(batch_capacity(shape, 64u, 0.5).max_segments_per_gpu == 64u, "a row holds one segment at least");
    expect_true(batch_capacity(shape, 2u, 1000.0).max_segments_per_gpu == 128u, "a row holds one per cell at most");
    expect_true(batch_capacity(shape, 7u, 1.0).max_rows_per_gpu == 7u, "the rows stay as given");
}

/** The synthetic corpus packs every token row with the segment budget's share of it, so the
 *  envelope probe measures the real batch's density: at three segments per row, each row's
 *  segments cover it exactly once and in order, and the step still trains finitely. */
static case_result_t test_packed_rows(world_health_t& health, tier_t tier) {
    if (!world_still_healthy_(health)) return case_result_t::skipped(world_unhealthy_k);
    trainer_config_t config = test_geometry_(test_mixture_t::tokens_k, test_run.settings.seed);
    config.batch = batch_capacity(config.batch, 2u, 3.0);
    config.capture = util::eager_k;
    trainer_t trainer;
    if (util::expected_t built = make_trainer_(trainer, tier, config, health);
        !expect_true(built, "packed trainer bring-up: {}", built.description_))
        return {};
    [[maybe_unused]] util::status_t const gpu_scope = (trainer.streams->stream(0)).bind();
    if (!expect_true(gpu_scope.succeeded(), "packed context entry")) return {};
    base_replica_t& lead = trainer.replicas[0];
    if (util::expected_t drawn = lead.draw(config, 0u, util::data_epoch_t {0u}, trainer.streams->stream(0));
        !expect_true(drawn, "packed draw: {}", drawn.description_))
        return {};
    std::size_t const rows = config.batch.max_rows_per_gpu, columns = config.batch.max_columns_per_row;
    std::size_t const segments = config.batch.max_segments_per_gpu;
    util::u32r1_tensor_pinned_t offsets = make_uninitialized<util::u32r1_tensor_pinned_t>({segments + 1});
    util::u32r1_tensor_pinned_t lengths = make_uninitialized<util::u32r1_tensor_pinned_t>({segments});
    if (!expect_true(offsets && lengths &&
                         util::copy_device_to_host(lead.batch.segment_offsets.view(), offsets.span(),
                                                   trainer.streams->stream(0)) &&
                         util::copy_device_to_host(lead.batch.segment_lengths.view(), lengths.span(),
                                                   trainer.streams->stream(0)) &&
                         cuStreamSynchronize(util::native_stream(trainer.streams->stream(0))) == CUDA_SUCCESS,
                     "packed readback"))
        return {};
    bool covered = segments == 3u * rows && offsets[segments] == rows * columns;
    for (std::size_t row = 0; covered && row != rows; ++row) {
        std::size_t cursor = row * columns;
        for (std::size_t index = 3u * row; covered && index != 3u * row + 3u; ++index) {
            covered = offsets[index] == cursor && lengths[index] != 0u;
            cursor += lengths[index];
        }
        covered = covered && cursor == (row + 1u) * columns;
    }
    expect_true(covered, "{} rows of {} columns cut into three segments each, in order", rows, columns);
    if (util::expected_t ran = run_steps_(trainer, 0u, 2u); !expect_true(ran, "packed steps: {}", ran.description_))
        return {};
    util::f32_t const loss = trainer.telemetry(util::optimizer_step_t {1u}).composite();
    expect_true(std::isfinite(loss), "packed rows train finitely: loss {}", loss);
    if (util::expected_t drained = trainer.synchronize(); !drained)
        expect_true(false, "packed end-of-case drain: {}", drained.description_);
    return {};
}

/** The capture contract holds, one forward and one backward replay graph per replica build, and
 *  two forward replays over one fixed eager preamble produce the same shared @c d(normalized) up
 *  to reduction order. */
static case_result_t test_graph_capture(world_health_t& health, tier_t tier) {
    if (!world_still_healthy_(health)) return case_result_t::skipped(world_unhealthy_k);
    trainer_config_t const config = test_geometry_(test_mixture_t::tokens_k, test_run.settings.seed);
    trainer_t trainer;
    if (util::expected_t built = make_trainer_(trainer, tier, config, health);
        !expect_true(built, "capture trainer bring-up: {}", built.description_))
        return {};
    std::size_t const gpus = trainer.replicas.size();
    expect_true(trainer.forward_graphs.size() == gpus && trainer.backward_graphs.size() == gpus,
                "one captured forward and backward region per replica: got {} and {} want {}",
                trainer.forward_graphs.size(), trainer.backward_graphs.size(), gpus);
    if (trainer.forward_graphs.size() != gpus) return {};

    [[maybe_unused]] util::status_t const gpu_scope = (trainer.streams->stream(0)).bind();
    if (!expect_true(gpu_scope.succeeded(), "capture context entry")) return {};
    base_replica_t& lead = trainer.replicas[0];
    util::expected_t prepared = lead.draw(config, 0u, util::data_epoch_t {0u}, trainer.streams->stream(0));
    if (prepared) prepared = lead.prepare(config, util::data_epoch_t {0u}, trainer.streams->stream(0));
    if (!expect_true(prepared, "capture preamble: {}", prepared.description_)) return {};
    if (util::expected_t first = trainer.forward_graphs[0].launch(util::native_stream(trainer.streams->stream(0)));
        !expect_true(first, "first replay: {}", first.description_))
        return {};
    auto [first_grad, snapshot_first_grad_status] = snapshot_grad_(lead, trainer.streams->stream(0));
    if (util::expected_t second = trainer.forward_graphs[0].launch(util::native_stream(trainer.streams->stream(0)));
        !expect_true(second, "second replay: {}", second.description_))
        return {};
    auto [second_grad, snapshot_second_grad_status] = snapshot_grad_(lead, trainer.streams->stream(0));
    expect_true(snapshot_first_grad_status && snapshot_second_grad_status && snapshots_near_(first_grad, second_grad),
                "two captured replays agree");
    return {};
}

/** Download one replica's @b [groups] Float32 temperatures, β then bias, into @p host . */
[[nodiscard]] static bool read_temperatures_(base_replica_t& replica, util::executor_stream_t& stream,
                                             st::basic_vector<util::f32_t>& host) noexcept {
    std::size_t const groups = replica.token_inverse_temperatures.numel();
    if (st::failed(host.resize(st::assume_uninitialized, 2u * groups))) return false;
    util::f32r1_tensor_pinned_t staging = make_uninitialized<util::f32r1_tensor_pinned_t>({2u * groups});
    if (!staging) return false;
    util::f32r1_span_t const halves = staging.span();
    if (!util::copy_device_to_host(replica.token_inverse_temperatures.view(), halves[nk::range(0, groups), nk::slice],
                                   stream) ||
        !util::copy_device_to_host(replica.token_biases.view(), halves[nk::range(groups, 2u * groups), nk::slice],
                                   stream))
        return false;
    cuStreamSynchronize(util::native_stream(stream));
    std::memcpy(host.data(), staging.data(), 2u * groups * sizeof(util::f32_t));
    return true;
}

/** The late-interaction temperatures and their Adafactor moments survive a checkpoint: a
 *  trainer resumed from another's file takes the identical next temperature step, which zeroed
 *  moments would not. */
static case_result_t test_late_interaction_resume(world_health_t& health, tier_t tier) {
    if (!world_still_healthy_(health)) return case_result_t::skipped(world_unhealthy_k);
    trainer_config_t config = test_geometry_(test_mixture_t::tokens_k, test_run.settings.seed);
    config.objective = objective_t::contrastive_k;
    config.late_interaction.weight = 1.0f;
    config.late_interaction.width = 32;
    config.capture = util::eager_k;
    scratch_directory_t const scratch;
    sz::string_t path;
    if (!expect_true(util::format_into(path, "{}/late_interaction.safetensors", scratch.path.c_str()),
                     "checkpoint path formatting failed"))
        return {};
    st::basic_vector<util::f32_t> started, continued, resumed;
    {
        trainer_t trainer;
        if (util::expected_t built = make_trainer_(trainer, tier, config, health);
            !expect_true(built, "late-interaction trainer bring-up: {}", built.description_))
            return {};
        [[maybe_unused]] util::status_t const gpu_scope = (trainer.streams->stream(0)).bind();
        util::expected_t stepped = read_temperatures_(trainer.replicas[0], trainer.streams->stream(0), started)
                                       ? run_steps_(trainer, 0, 3)
                                       : util::expected_t::fail(util::status_code_t::device_k,
                                                                "temperature download failed");
        if (stepped) stepped = trainer.synchronize();
        if (stepped) stepped = trainer.save(path.view());
        if (stepped) stepped = run_steps_(trainer, 3, 4);
        if (stepped) stepped = trainer.synchronize();
        if (!expect_true(stepped && read_temperatures_(trainer.replicas[0], trainer.streams->stream(0), continued),
                         "late-interaction steps: {}", stepped.description_))
            return {};
    }
    trainer_t trainer;
    if (util::expected_t built = make_trainer_(trainer, tier, config, health);
        !expect_true(built, "resumed trainer bring-up: {}", built.description_))
        return {};
    [[maybe_unused]] util::status_t const gpu_scope = (trainer.streams->stream(0)).bind();
    util::expected_t stepped = trainer.resume(path.view());
    if (stepped) stepped = run_steps_(trainer, 3, 4);
    if (stepped) stepped = trainer.synchronize();
    if (!expect_true(stepped && read_temperatures_(trainer.replicas[0], trainer.streams->stream(0), resumed),
                     "resumed step: {}", stepped.description_))
        return {};
    expect_true(std::memcmp(started.data(), continued.data(), started.size() * sizeof(util::f32_t)) != 0,
                "four late-interaction steps move the temperatures");
    expect_true(std::memcmp(continued.data(), resumed.data(), continued.size() * sizeof(util::f32_t)) == 0,
                "the resumed trainer takes the identical temperature step: β {} vs {}, bias {} vs {}", continued[0],
                resumed[0], continued[continued.size() / 2], resumed[resumed.size() / 2]);
    return {};
}

/** Calibrated choices train end to end through the items source the C verb drives: four text
 *  states, each offering two or three legal bytes against a one-hot or soft target, replay eight
 *  captured steps with the masked term weighted out — the choices term reads finite and positive
 *  first, falls by the last step, and the token term stays zero throughout. */
static case_result_t test_choices_replay(world_health_t& health, tier_t tier) {
    if (!world_still_healthy_(health)) return case_result_t::skipped(world_unhealthy_k);
    trainer_config_t config = test_geometry_(test_mixture_t::tokens_k, test_run.settings.seed);
    // Rows as wide as the model: the envelope's one-row probe reads choices off its tokens.
    config.batch = multimodal_batch_shape_t {2u, 4u, 256u};
    config.batch.max_choices_per_segment = 4u;
    config.steps = 8;
    config.text_weight = 0.0f;
    config.choices.weight = 1.0f;
    trainer_t trainer;
    if (util::expected_t built = make_trainer_(trainer, tier, config, health);
        !expect_true(built, "choices trainer bring-up: {}", built.description_))
        return {};
    items_source_t source;
    if (util::expected_t made = items_source_t::make(source, trainer);
        !expect_true(made, "choices items source: {}", made.description_))
        return {};

    char const text[] = "2+2="           //
                        "is ice cold? "  //
                        "pick a vowel: " //
                        "is fire cold? ";
    std::size_t const offsets[] = {0, 4, 17, 31, sizeof(text) - 1};
    modality_t const modalities[] = {modality_t::text_k, modality_t::text_k, modality_t::text_k, modality_t::text_k};
    std::size_t const choice_offsets[] = {0, 3, 5, 8, 10};
    util::u8_t const choice_bytes[] = {'3', '4', '5', 'Y', 'N', 'a', 'e', 'x', 'Y', 'N'};
    util::f32_t const targets[] = {0.0f, 1.0f, 0.0f, 1.0f, 0.0f, 0.5f, 0.5f, 0.0f, 0.0f, 1.0f};
    source.items.modalities = modalities;
    source.items.contents = {std::as_bytes(std::span {text}), offsets};
    source.items.choices = {{choice_bytes, choice_offsets}, {targets, choice_offsets}};

    util::f32_t first = 0.0f, last = 0.0f, token = 0.0f;
    step_report_t report;
    for (util::u32_t ordinal = 0; ordinal != config.steps; ++ordinal) {
        if (util::expected_t stepped = trainer.step(batch_source_of(source), util::optimizer_step_t {ordinal}, report);
            !expect_true(stepped, "choices step {}: {}", ordinal, stepped.description_))
            return {};
        telemetry_t const telemetry = trainer.telemetry(util::optimizer_step_t {ordinal});
        if (ordinal == 0u) first = telemetry.choices;
        last = telemetry.choices;
        token = fu::max_of(token, std::fabs(telemetry.token_ce));
    }
    expect_true(std::isfinite(first) && first > 0.0f, "the first choices term is finite and positive: {}", first);
    // Seed 42 settles from 1.17 onto the soft target's entropy floor, ln2/4 ≈ 0.17: a wide margin.
    expect_true(std::isfinite(last) && last < 0.5f * first, "the choices term halves: first {} last {}", first, last);
    expect_true(token == 0.0f, "the weighted-out token term reads zero: peak {}", token);
    if (util::expected_t drained = trainer.synchronize(); !drained)
        expect_true(false, "choices end-of-case drain: {}", drained.description_);
    return {};
}

/** The regularizers ride a contrastive step, and a forced gradient spike fails the next step before
 *  the optimizer moves a weight: a threshold far below one makes every second gradient a spike. */
static case_result_t test_spectrum_spike(world_health_t& health, tier_t tier) {
    if (!world_still_healthy_(health)) return case_result_t::skipped(world_unhealthy_k);
    trainer_config_t config = test_geometry_(test_mixture_t::tokens_k, test_run.settings.seed);
    config.objective = objective_t::contrastive_k;
    config.tcr_lambda = 0.1f;
    config.lipschitz_lambda = 1.0e-3f;
    config.grad_spike_threshold = 1.0e-6f;
    trainer_t trainer;
    if (util::expected_t built = make_trainer_(trainer, tier, config, health);
        !expect_true(built, "spectrum trainer bring-up: {}", built.description_))
        return {};
    if (util::expected_t first = run_steps_(trainer, 0u, 1u);
        !expect_true(first, "the first step, with nothing to spike against: {}", first.description_))
        return {};
    [[maybe_unused]] util::status_t const gpu_scope = (trainer.streams->stream(0)).bind();
    util::f32_t terms[2] = {};
    if (util::expected_t copied = util::copy_device_to_host(trainer.spectrum.terms(0), std::span<util::f32_t> {terms},
                                                            trainer.streams->stream(0));
        !expect_true(copied, "terms download: {}", copied.description_))
        return {};
    cuStreamSynchronize(util::native_stream(trainer.streams->stream(0)));
    expect_true(std::isfinite(terms[0]) && terms[0] > 0.0f && std::isfinite(terms[1]) && terms[1] > 0.0f,
                "the coding rate {} and the top eigenvalue {} read finite and positive", terms[0], terms[1]);
    auto [weights_before, snapshot_weights_before_status] = snapshot_weights_(trainer.replicas[0].weights,
                                                                              trainer.streams->stream(0));
    util::expected_t const spiked = run_steps_(trainer, 1u, 2u);
    auto [weights_after, snapshot_weights_after_status] = snapshot_weights_(trainer.replicas[0].weights,
                                                                            trainer.streams->stream(0));
    expect_true(!spiked && spiked.status_ == util::status_code_t::internal_k,
                "a threshold of {} refuses the second step", config.grad_spike_threshold);
    expect_true(snapshot_weights_before_status && snapshot_weights_after_status &&
                    snapshots_match_(weights_before, weights_after),
                "the refused step leaves the weights untouched");
    if (util::expected_t drained = trainer.synchronize(); !drained)
        expect_true(false, "spectrum end-of-case drain: {}", drained.description_);
    return {};
}

/** Every rank of a world of two or more GPUs reaches the same coding rate bit for bit, and it
 *  matches one GPU holding every row of the world. */
static case_result_t test_spectrum_world(world_health_t& health, tier_t wide, tier_t cheap) {
    if (!world_still_healthy_(health)) return case_result_t::skipped(world_unhealthy_k);
    util::mesh_topology_t& topology = wide.mesh.topology;
    std::size_t const gpus = wide.devices.size(), world_gpus = topology.shape.gpus_per_colocated_domain;
    if (world_gpus < 2u) return case_result_t::skipped("needs two GPUs in the world");
    std::size_t const rows = 24, hidden = 64, world_rows = rows * world_gpus;
    // A shared offset gives the rows the anisotropy pooled codes have.
    std::vector<util::f64_t> world_values(world_rows * hidden);
    std::vector<nk_bf16_t> world_pooled(world_rows * hidden);
    for (std::size_t index = 0; index != world_values.size(); ++index)
        world_values[index] =
            0.3 + util::f64_t(util::splitmix64_hash_t {}(test_run.settings.seed, 13u, index) >> 40) * 0x1.0p-23 - 1.0;
    if (!expect_true(narrow_to_bf16(world_values, world_pooled), "host rounding failed")) return {};

    // The coding rate over `devices`, each holding its slice of the world rows from `first_row`,
    // read back from every rank.
    auto coding_rates = [&](mesh_t& mesh, std::span<util::i32_t const> devices, util::gpu_numerics_t const& numerics,
                            std::size_t rows_per_gpu, std::size_t first_row,
                            std::vector<util::f32_t>& rates) noexcept -> expected_t {
        pooled_spectrum_config_t config;
        config.rows_per_gpu = static_cast<util::u32_t>(rows_per_gpu);
        config.hidden = static_cast<util::u32_t>(hidden);
        config.rows_per_world = static_cast<util::u32_t>(world_rows);
        config.coding_rate_weight = 1.0f;
        auto [streams, spawn_status] = spawn_test_executor(devices);
        if (!spawn_status) return util::expected_t::fail(std::move(spawn_status));
        util::per_gpu_array<util::gpu_arena_t> arenas;
        util::per_gpu_array<util::bf16r2_tensor_arena_t> pooled, pooled_grad;
        for (std::size_t rank = 0; rank != devices.size(); ++rank) {
            [[maybe_unused]] util::status_t const gpu_scope = (streams.stream(rank)).bind();
            auto [arena, make_arena_status] = util::gpu_arena_t::make(devices[rank], 1ull << 28, devices);
            if (!make_arena_status) return util::expected_t::fail(std::move(make_arena_status));
            if (arenas.push_back(std::move(arena)) != util::status_code_t::success_k)
                return util::expected_t::fail(util::status_code_t::out_of_memory_k, "arena table full");
            util::gpu_arena_alloc<util::bf16_t> bf16_alloc {arenas[rank]};
            if (pooled.push_back(make_uninitialized<util::bf16r2_tensor_arena_t>({rows_per_gpu, hidden}, bf16_alloc)) !=
                    util::status_code_t::success_k ||
                pooled_grad.push_back(make_uninitialized<util::bf16r2_tensor_arena_t>(
                    {rows_per_gpu, hidden}, bf16_alloc)) != util::status_code_t::success_k ||
                !pooled[rank] || !pooled_grad[rank])
                return util::expected_t::fail(util::status_code_t::out_of_memory_k, "pooled rows allocation failed");
            std::size_t const begin = (first_row + rank * rows_per_gpu) * hidden;
            cuMemcpyHtoD(util::device_address(pooled[rank].data()), world_pooled.data() + begin,
                         rows_per_gpu * hidden * sizeof(nk_bf16_t));
            if (util::expected_t zeroed = util::zero_async(pooled_grad[rank].span(), streams.stream(rank)); !zeroed)
                return zeroed;
        }
        auto [domain, make_domain_status] = util::multimem_domain_t::make(
            mesh.topology, devices, distributed_pooled_spectrum_t::payload_scalars_needed(config), streams,
            wait_limit_k.count());
        if (!make_domain_status) return util::expected_t::fail(std::move(make_domain_status));
        auto [spectrum, make_spectrum_status] = distributed_pooled_spectrum_t::make(
            domain, mesh.colocated, arenas.span(), streams, {}, 0, config, numerics, wait_limit_k.count());
        if (!make_spectrum_status) return util::expected_t::fail(std::move(make_spectrum_status));
        for (std::size_t rank = 0; rank != devices.size(); ++rank)
            if (util::expected_t shaped = spectrum(rank, pooled[rank].view(), pooled_grad[rank].span(),
                                                   streams.stream(rank));
                !shaped)
                return shaped;
        rates.assign(devices.size(), 0.0f);
        for (std::size_t rank = 0; rank != devices.size(); ++rank) {
            [[maybe_unused]] util::status_t const gpu_scope = (streams.stream(rank)).bind();
            util::f32_t terms[2] = {};
            if (util::expected_t copied = util::copy_device_to_host(
                    spectrum.terms(rank), std::span<util::f32_t> {terms}, streams.stream(rank));
                !copied)
                return copied;
            if (CUresult result = cuStreamSynchronize(util::native_stream(streams.stream(rank)));
                result != CUDA_SUCCESS)
                return util::cuda_failure(result, "rank {} drain failed", rank);
            rates[rank] = terms[0];
        }
        return {};
    };

    std::vector<util::f32_t> world_rates, single_rate;
    std::size_t const first_row = topology.first_gpu_index() * rows;
    if (util::expected_t ran = coding_rates(wide.mesh, wide.devices, wide.numerics, rows, first_row, world_rates);
        !expect_true(ran, "world coding rate: {}", ran.description_))
        return {};
    if (util::expected_t ran = coding_rates(cheap.mesh, cheap.devices, cheap.numerics, world_rows, 0, single_rate);
        !expect_true(ran, "single-GPU coding rate: {}", ran.description_))
        return {};
    for (std::size_t rank = 1; rank != gpus; ++rank)
        expect_true(std::memcmp(&world_rates[rank], &world_rates[0], sizeof(util::f32_t)) == 0,
                    "rank {} reads the coding rate {} where rank 0 reads {}", rank, world_rates[rank], world_rates[0]);
    // The world sums Float32 Grams in another order than one GPU's NumKong product does.
    expect_true(std::fabs(world_rates[0] - single_rate[0]) <= 1.0e-4f * single_rate[0],
                "the world's coding rate {} against one GPU holding every row {}", world_rates[0], single_rate[0]);
    return {};
}

int main(int argc, char** argv) {
    // Lazy module loading deadlocks against the InfiniBand doorbell kernels' flag spins: the
    // spinning wait blocks its context, the peer's next launch blocks on module loading behind it,
    // and neither side progresses. Eager loading before `cuInit` front-loads every module — the
    // mixer suite's fix.
    ::setenv("CUDA_MODULE_LOADING", "EAGER", 1);

    start_test_run(argc, argv);

    // The schedule and the segment budget are pure host math — asserted before any GPU work.
    run_test_case("test_learning_rate_schedule", test_learning_rate_schedule);
    run_test_case("test_segment_budget", test_segment_budget);

    int device_count = 0;
    if (cuInit(0) != CUDA_SUCCESS || cuDeviceGetCount(&device_count) != CUDA_SUCCESS || device_count <= 0) {
        for (char const* label :
             {"test_packed_rows", "test_graph_capture", "test_late_interaction_resume", "test_choices_replay",
              "test_spectrum_spike", "test_spectrum_world", "test_base_conformance(\"tokens\")",
              "test_base_conformance(\"omni\")", "mixture_sweep"})
            run_test_case(label, [] { return case_result_t::skipped(no_gpu_device_k); });
        return finish_test_run();
    }
    st::basic_vector<util::i32_t> devices;
    for (int device = 0; device != device_count; ++device)
        if (!expect_true(!st::failed(devices.push_back(static_cast<util::i32_t>(device))),
                         "device table allocation failed"))
            return finish_test_run();

    // One process per host_domain; the sbatch launcher raises UFORM_HOST_DOMAIN_COUNT for the
    // two-node tier and names the exchange, which this main maps onto the mesh options, as the
    // library reads no environment.
    mesh_options_t options;
    options.host_domain_index = util::host_domain_index_from_environment();
    options.host_domain_count = util::host_domain_count_from_environment();
    options.timeout_seconds = wait_limit_k.count();
    if (char const* directory = std::getenv("UFORM_EXCHANGE_DIR"); directory != nullptr)
        options.directory = sz::string_view_t {directory, std::strlen(directory)};
    if (char const* job = std::getenv("UFORM_JOB_ID"); job != nullptr)
        options.job = sz::string_view_t {job, std::strlen(job)};
    if (!expect_true(nk_capabilities_enabled_cuda(0, &options.capabilities) == nk_success_k,
                     "NumKong cannot query device 0"))
        return finish_test_run();
    mesh_t wide_mesh;
    if (util::expected_t joined = mesh_t::make(wide_mesh, options, devices);
        !expect_true(joined, "mesh bootstrap: {}", joined.description_))
        return finish_test_run();
    mesh_t cheap_mesh;
    if (util::expected_t narrowed = narrow_mesh_(cheap_mesh, wide_mesh.topology, 1u);
        !expect_true(narrowed, "narrowed mesh: {}", narrowed.description_))
        return finish_test_run();
    util::gpu_numerics_t const numerics = gpu_numerics_of(options.capabilities);
    auto [wide_streams, wide_spawn_status] = spawn_test_executor(devices);
    auto [cheap_streams, cheap_spawn_status] = spawn_test_executor({devices.data(), 1});
    if (!expect_true(wide_spawn_status && cheap_spawn_status, "tier executors: {}{}", wide_spawn_status.description_,
                     cheap_spawn_status.description_))
        return finish_test_run();
    tier_t const wide {wide_mesh, wide_streams, devices, numerics};
    tier_t const cheap {cheap_mesh, cheap_streams, std::span<util::i32_t const> {devices}.first(1), numerics};

    world_health_t health;
    run_test_case("test_packed_rows", test_packed_rows, health, cheap);
    run_test_case("test_graph_capture", test_graph_capture, health, cheap);
    run_test_case("test_late_interaction_resume", test_late_interaction_resume, health, cheap);
    run_test_case("test_choices_replay", test_choices_replay, health, cheap);
    run_test_case("test_spectrum_spike", test_spectrum_spike, health, cheap);
    run_test_case("test_spectrum_world", test_spectrum_world, health, wide, cheap);
    run_test_case(R"(test_base_conformance("tokens"))", test_base_conformance, health, "tokens",
                  test_geometry_(test_mixture_t::tokens_k, test_run.settings.seed), wide);
    run_test_case(R"(test_base_conformance("omni"))", test_base_conformance, health, "omni",
                  test_geometry_(test_mixture_t::omni_k, test_run.settings.seed), wide);
    mixture_sweep(health, wide, cheap);
    return finish_test_run();
}
