/**
 *  @brief The Base training conformance test: builds the same @ref base_trainer_t the binary drives and asserts
 *      the invariants the loop itself never checks — deep replay conformance for the tokens and omni mixtures,
 *      plus a cross-product sweep over every media subset, row budget, and model-region mode.
 *  @file cuda_test/base.cu
 *  @author Ash Vardanian
 *  @date July 6, 2026
 *
 *  This is a test: a plain @c int @c main builds one @ref test_suite_t and threads it by reference into every
 *  case, consuming @c cuda_train/base.cuh 's typed surface instead of copying the driver. The domain, colocated
 *  surface, and optimizer are locals in each case's frame, threaded by reference, exactly as
 *  @ref base_trainer_t::run owns them. Launched under Slurm via @c cuda_test/base-4-8-16.sbatch , the same
 *  binary runs the distributed tiers: @c main honors @c UFORM_HOST_DOMAIN_COUNT , so a two-node world exercises
 *  the InfiniBand reduce under every case below.
 *
 *  The case matrix:
 *
 *      - @ref learning_rate_schedule — the warmup / cosine policy's properties, pure host math, no GPU.
 *      - @ref run_base_conformance for the tokens and omni mixtures at full width — @b A every replica
 *        bitwise-matches the lead, @b B every member's barrier round advanced and they all agree, @b C a
 *        reseeded eager rerun lands bitwise on the captured run, @b D each head's loss fell between its first
 *        and last scored steps, @b E every live head's loss read nonzero at least once, @b F the captured
 *        window moved the weights off their seed.
 *      - @ref mixture_sweep — two width tiers: every media subset (2³) crossed with one-row and two-row
 *        budgets on an isolated one-GPU world (plus the uneven five-row and 96-column omni corners), then
 *        every subset again at the bootstrapped world's full width. Each cell steps eagerly, then captures
 *        and replays over the same trainer, so both model-region modes run under every mixture.
 *      - @ref graph_capture — the capture contract holds and two replays are bitwise-identical; runs on the
 *        one-GPU world and needs no domain, so it stays green even where multicast is not provisioned.
 */
#include <cmath>   // `std::isfinite`
#include <cstdlib> // `std::getenv`, `std::atoi` — the distributed tier's host_domain count
#include <cstring> // `std::memcmp`, `std::memcpy`

#include <algorithm> // `std::max` — the reusable staging tensor's size
#include <span>      // `std::span` — the replay-graph table decays to a view for `step`

#include <cuda.h> // `cuInit`, `cuStreamSynchronize`

#include "cuda_train/base.cuh"   // `base_config_t`, `base_replica_t`, `base_trainer_t`
#include "cuda_test/harness.hpp" // `test_suite_t` — the cumulative pass/fail tally

using namespace unum::uform::core;
using namespace unum::uform::train;
using namespace unum::uform::test;

/** @brief The tiny shared geometry — @c depth=2 , @c hidden=256 , @c 8 steps, and a two-step warmup so even
 *      the tiny run crosses both schedule phases. Tokens ride a @c 2×64 rectangle; the omni variant fills
 *      every media slice over a @c 4×64 one, so the keyed partition holds all four modalities live at every
 *      step and the composite loss measures the same mixture throughout — descent stays comparable. */
[[nodiscard]] static base_config_t test_geometry_(bool omni) noexcept {
    base_config_t config;
    config.batch = multimodal_batch_shape_t {2u, 2u, 64u};
    if (omni) {
        config.batch = multimodal_batch_shape_t {4u, 4u, 64u};
        config.batch.image = image_batch_shape_t {8u, 3u};
        config.batch.video = video_batch_shape_t {8u, 3u, 2u, 1u};
        config.batch.audio = audio_batch_shape_t {256u, 16000u};
    }
    config.depth = 2;
    config.hidden = 256;
    config.steps = 8;
    config.warmup_steps = 2;
    return config;
}

/** @brief A narrowed copy of the bootstrapped world: this process alone drives @p gpus GPUs as an isolated
 *      one-host-domain mesh — the sweep's cheap tier and the capture case's world. Safe because a single-host
 *      topology carries no rendezvous state; the one cross-field invariant is that @c host_domains sizes to
 *      @c shape.host_domain_count() . */
[[nodiscard]] static expected<mesh_topology_t> narrowed_topology_(mesh_topology_t const &world, u32_t gpus) noexcept {
    using result_t = expected<mesh_topology_t>;
    mesh_topology_t narrowed;
    narrowed.shape.gpus_per_host_domain = gpus;
    narrowed.shape.gpus_per_multimem_domain = gpus;
    narrowed.shape.gpus_per_colocated_domain = gpus;
    narrowed.host_domain_index = 0;
    if (narrowed.host_domains.try_resize(1) != status_t::success_k)
        return result_t::fail(status_t::bad_alloc_k, "narrowed host-domain table allocation failed");
    narrowed.host_domains[0] = world.host_domains[world.host_domain_index];
    narrowed.host_domains[0].host_domain_index = 0;
    narrowed.host_domains[0].device_count = gpus;
    return result_t {std::move(narrowed)};
}

/** @brief The one multimem domain over @p windows disjoint optimizer windows of one replica's payload — one
 *      window suffices for a single run, two let the captured and the assertion-C eager optimizers each reserve
 *      their own fresh moments from the same domain. */
[[nodiscard]] static expected<multimem_domain_t> make_domain_(mesh_topology_t const &topology, base_trainer_t &trainer,
                                                              u32_t windows) noexcept {
    array<differentiable_weights_t> const &lead = trainer.replicas[0].weights;
    std::span<differentiable_weights_t const> const list {lead.view().data(), lead.size()};
    std::size_t const window = distributed_adafactor_t::payload_scalars_needed(list);
    return multimem_domain_t::make(topology, trainer.devices.view(), static_cast<std::size_t>(windows) * window,
                                   trainer.streams);
}

/** @brief The distributed Adafactor over the transient flat replica-major registration table — the same
 *      construction @ref base_trainer_t::run performs. A second call on the same domain reserves the next
 *      disjoint window (fresh moments). */
[[nodiscard]] static expected<distributed_adafactor_t> make_optimizer_(base_trainer_t &trainer,
                                                                       multimem_domain_t &domain,
                                                                       colocated_domain_t &colocated) noexcept {
    using result_t = expected<distributed_adafactor_t>;
    array<differentiable_weights_t> registrations;
    for (std::size_t member_index = 0; member_index != trainer.replicas.size(); ++member_index) {
        array<differentiable_weights_t> const &owned = trainer.replicas[member_index].weights;
        std::span<differentiable_weights_t const> const list {owned.view().data(), owned.size()};
        for (differentiable_weights_t const &entry : list)
            if (registrations.try_push_back(entry) != status_t::success_k)
                return result_t::fail(status_t::bad_alloc_k, "registration table allocation failed");
    }
    adafactor_config_t optimizer_config; // the per-step learning rate travels through `step`, not the config
    return distributed_adafactor_t::make(domain, colocated, trainer.arenas,
                                         {registrations.view().data(), registrations.size()},
                                         trainer.replicas[0].weights.size(), optimizer_config, trainer.streams);
}

/** @brief Enqueue steps @b [begin,end) of the two-region loop — an empty @p graphs span runs the model region
 *      eagerly, a filled one replays the captured graphs. */
[[nodiscard]] static expected_t run_steps_(base_trainer_t &trainer, distributed_adafactor_t &optimizer,
                                           std::span<gpu_replay_graph_t> graphs, u32_t begin, u32_t end) noexcept {
    for (u32_t ordinal = begin; ordinal != end; ++ordinal)
        if (expected_t stepped = trainer.step(optimizer_step_t {ordinal}, optimizer, graphs); !stepped) return stepped;
    return {};
}

/** @brief Drain every replica's stream — the barrier before any host-side comparison. */
static void synchronize_(base_trainer_t &trainer) noexcept {
    for (std::size_t member_index = 0; member_index != trainer.streams.size(); ++member_index) {
        switch_to_gpu_t gpu_scope {trainer.streams[member_index]};
        cuStreamSynchronize(trainer.streams[member_index].get());
    }
}

/** @brief DtoH one replica's whole registration list into a byte-comparable host array — the snapshot behind
 *      assertions A, C, and the gradient-propagation check. */
[[nodiscard]] static expected<array<bf16_t>> snapshot_weights_(array<differentiable_weights_t> const &weights,
                                                               gpu_stream_t &stream) noexcept {
    using result_t = expected<array<bf16_t>>;
    switch_to_gpu_t gpu_scope {stream};
    if (!gpu_scope.ok()) return result_t::fail(status_t::cuda_error_k, "context entry failed");
    std::span<differentiable_weights_t const> const list {weights.view().data(), weights.size()};
    std::size_t total = 0, largest = 0;
    for (differentiable_weights_t const &matrix : list) {
        std::size_t const scalars = matrix.weights.numel();
        total += scalars;
        largest = std::max(largest, scalars);
    }
    array<bf16_t> host;
    if (host.try_resize_uninitialized(total) != status_t::success_k)
        return result_t::fail(status_t::bad_alloc_k, "snapshot allocation failed");
    nk::tensor<bf16_t, pinned_alloc<bf16_t>, 1> staging = nk::tensor<bf16_t, pinned_alloc<bf16_t>, 1>::try_empty(
        {largest}, pinned_alloc<bf16_t> {});
    if (!staging) return result_t::fail(status_t::bad_alloc_k, "pinned staging allocation failed");
    std::size_t cursor = 0;
    for (differentiable_weights_t const &matrix : list) {
        std::size_t const scalars = matrix.weights.numel();
        if (!staging.try_resize({scalars})) return result_t::fail(status_t::bad_alloc_k, "staging resize failed");
        nk::tensor_view<bf16_t, 1> const source = matrix.weights.flatten<1>();
        if (expected_t copied = copy_device_to_host(source, staging.span(), stream); !copied)
            return result_t::fail(std::move(copied));
        cuStreamSynchronize(stream.get());
        std::memcpy(host.data() + cursor, staging.data(), scalars * sizeof(bf16_t));
        cursor += scalars;
    }
    return result_t {std::move(host)};
}

/** @brief DtoH the lead replica's shared @c d(normalized) — the deterministic score output whose two captured
 *      replays must byte-match. */
[[nodiscard]] static expected<array<bf16_t>> snapshot_grad_(base_replica_t &replica, gpu_stream_t &stream) noexcept {
    using result_t = expected<array<bf16_t>>;
    switch_to_gpu_t gpu_scope {stream};
    if (!gpu_scope.ok()) return result_t::fail(status_t::cuda_error_k, "context entry failed");
    bf16r2_view_t const view = replica.normalized_grad.view();
    std::size_t const scalars = static_cast<std::size_t>(view.extent(0)) * view.extent(1);
    nk::tensor<bf16_t, pinned_alloc<bf16_t>, 1> staging = nk::tensor<bf16_t, pinned_alloc<bf16_t>, 1>::try_empty(
        {scalars}, pinned_alloc<bf16_t> {});
    if (!staging) return result_t::fail(status_t::bad_alloc_k, "pinned staging allocation failed");
    if (expected_t copied = copy_device_to_host(view.flatten<1>(), staging.span(), stream); !copied)
        return result_t::fail(std::move(copied));
    cuStreamSynchronize(stream.get());
    array<bf16_t> host;
    if (host.try_resize_uninitialized(scalars) != status_t::success_k)
        return result_t::fail(status_t::bad_alloc_k, "snapshot allocation failed");
    std::memcpy(host.data(), staging.data(), scalars * sizeof(bf16_t));
    return result_t {std::move(host)};
}

/** @brief Bytewise-equal two snapshots — the comparison behind assertions A and C. */
[[nodiscard]] static bool snapshots_match_(array<bf16_t> const &left, array<bf16_t> const &right) noexcept {
    return left.size() == right.size() && std::memcmp(left.data(), right.data(), left.size() * sizeof(bf16_t)) == 0;
}

/** @brief DtoH member @p member_index 's own barrier round through its accessor, on its own stream and
 *      context — zero when absent or when the readback slips (assertion B treats zero as "never
 *      advanced"). */
[[nodiscard]] static concurrency_round_t read_round_(multimem_domain_t &domain, u32_t member_index,
                                                     gpu_stream_t &stream) noexcept {
    u32_t const *const round = domain.member(member_index).round;
    if (round == nullptr) return concurrency_round_t {0u};
    switch_to_gpu_t gpu_scope {stream};
    if (!gpu_scope.ok()) return concurrency_round_t {0u};
    u32r1_tensor_pinned_t staging = u32r1_tensor_pinned_t::try_empty({1}, pinned_alloc<u32_t> {});
    nk::tensor_view<u32_t, 1> const source {round, 1};
    if (!staging || !copy_device_to_host(source, staging.span(), stream)) return concurrency_round_t {0u};
    cuStreamSynchronize(stream.get());
    return concurrency_round_t {staging[0]};
}

/** @brief The warmup / cosine policy's own properties, pure host math: a strictly monotone ramp topping out
 *      exactly at the peak, a never-rising decay bounded by @b [floor, peak] , and a tail settling onto the
 *      floor. */
static void learning_rate_schedule(test_suite_t &suite) {
    base_config_t config;
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

/** @brief The checkpoint naming contract over one built replica: the registration list carries the semantic
 *      per-weight names every SafeTensors parser keys by (under the driver's @c model. prefix), derived from
 *      @p config and asserted in full order — a @c fill reorder or rename fails loudly here before it silently
 *      re-keys a checkpoint. Piggybacks on an already-built trainer, so it costs no bring-up of its own. */
static void check_weight_naming_(test_suite_t &suite, char const *label, base_config_t const &config,
                                 std::span<differentiable_weights_t const> weights) {
    std::size_t index = 0;
    auto expect = [&](sz::string_view want) {
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
    sz::string name;
    for (u32_t layer_index = 0; layer_index != config.depth; ++layer_index)
        for (sz::string_view part : {"qkv", "out", "gate_up", "down", "pre_norm", "post_norm"}) {
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

/** @brief Bring up the trainer for one mixture, run the captured two-region loop reading telemetry every step,
 *      and fold the five invariants (A replica identity, B round progression, C capture == eager, D loss
 *      descent, E every live head trained) into @p suite . */
static void run_base_conformance(test_suite_t &suite, char const *label, base_config_t const &config,
                                 mesh_topology_t const &topology) {
    base_trainer_t trainer;
    if (expected_t built = base_trainer_t::make(trainer, config, topology); !built)
        return suite.check(false, "[{}] trainer bring-up: {}", label, built.description_);
    u32_t const gpus = static_cast<u32_t>(trainer.replicas.size());
    suite.check(gpus != 0u, "[{}] at least one visible GPU: got {}", label, gpus);
    if (gpus == 0u) return;
    check_weight_naming_(suite, label, config, trainer.replicas[0].weights.view());

    // Two disjoint windows so the captured optimizer and the assertion-C eager optimizer get fresh moments.
    expected<multimem_domain_t> domain = make_domain_(topology, trainer, 2u);
    if (!domain) return suite.check(false, "[{}] domain setup: {}", label, domain.description_);
    expected<colocated_domain_t> colocated = colocated_domain_t::make(topology);
    if (!colocated) return suite.check(false, "[{}] colocated setup: {}", label, colocated.description_);
    expected<distributed_adafactor_t> optimizer = make_optimizer_(trainer, domain.value_, colocated.value_);
    if (!optimizer) return suite.check(false, "[{}] optimizer setup: {}", label, optimizer.description_);
    per_gpu_array<gpu_replay_graph_t> graphs;
    if (expected_t captured = trainer.capture(graphs); !captured)
        return suite.check(false, "[{}] model-region capture: {}", label, captured.description_);
    // Assertion F's reference: the lead replica's freshly seeded weights, untouched by the capture.
    expected<array<bf16_t>> seed_lead = snapshot_weights_(trainer.replicas[0].weights, trainer.streams[0]);
    if (!seed_lead) return suite.check(false, "[{}] seed snapshot: {}", label, seed_lead.description_);

    // The captured run, telemetry every step. Assertion D compares each head against itself — its first
    // and last scored (nonzero) readouts — so a step whose keyed draw masked nothing for one head (a
    // routine event on short token rows) never skews the comparison with a missing term. Assertion E's
    // per-modality coverage accumulates from the same readouts.
    f32_t first_scored_sum = 0.0f, last_scored_sum = 0.0f;
    f32_t token_first = 0.0f, token_last = 0.0f, image_first = 0.0f, image_last = 0.0f;
    f32_t video_first = 0.0f, video_last = 0.0f, audio_first = 0.0f, audio_last = 0.0f;
    for (u32_t ordinal = 0; ordinal != config.steps; ++ordinal) {
        if (expected_t ran = trainer.step(optimizer_step_t {ordinal}, optimizer.value_, graphs); !ran)
            return suite.check(false, "[{}] captured step {}: {}", label, ordinal, ran.description_);
        base_trainer_t::telemetry_t const telemetry = trainer.telemetry(domain.value_, optimizer_step_t {ordinal});
        if (telemetry.token_ce != 0.0f) {
            if (token_first == 0.0f) token_first = telemetry.token_ce;
            token_last = telemetry.token_ce;
        }
        if (telemetry.image_mse != 0.0f) {
            if (image_first == 0.0f) image_first = telemetry.image_mse;
            image_last = telemetry.image_mse;
        }
        if (telemetry.video_mse != 0.0f) {
            if (video_first == 0.0f) video_first = telemetry.video_mse;
            video_last = telemetry.video_mse;
        }
        if (telemetry.audio_mse != 0.0f) {
            if (audio_first == 0.0f) audio_first = telemetry.audio_mse;
            audio_last = telemetry.audio_mse;
        }
    }
    bool const token_trained = token_first != 0.0f, image_trained = image_first != 0.0f;
    bool const video_trained = video_first != 0.0f, audio_trained = audio_first != 0.0f;
    first_scored_sum = token_first + image_first + video_first + audio_first;
    last_scored_sum = token_last + image_last + video_last + audio_last;
    synchronize_(trainer);
    expected<array<bf16_t>> captured_lead = snapshot_weights_(trainer.replicas[0].weights, trainer.streams[0]);
    if (!captured_lead) return suite.check(false, "[{}] lead snapshot: {}", label, captured_lead.description_);

    // A — every replica byte-matches the lead. Vacuous at one GPU (the peer loop is empty), but the run stands.
    bool replicas_identical = true;
    for (u32_t member_index = 1; member_index < gpus; ++member_index) {
        expected<array<bf16_t>> peer = snapshot_weights_(trainer.replicas[member_index].weights,
                                                         trainer.streams[member_index]);
        replicas_identical = replicas_identical && peer && snapshots_match_(peer.value_, captured_lead.value_);
    }
    suite.check(replicas_identical, "[{}] every replica matches the lead bitwise (A)", label);

    // B — every member's own barrier round advanced off zero and they all agree: with self-advancing
    //     rounds the uniformity is a real invariant (each member's scalar counts its own passed barriers).
    concurrency_round_t const lead_round = read_round_(domain.value_, 0u, trainer.streams[0]);
    bool round_uniform = true;
    for (u32_t member_index = 0; member_index < gpus; ++member_index)
        round_uniform = round_uniform &&
                        read_round_(domain.value_, member_index, trainer.streams[member_index]).ordinal ==
                            lead_round.ordinal;
    suite.check(lead_round.ordinal > 0u, "[{}] barrier round advanced off zero: got {} (B)", label, lead_round.ordinal);
    suite.check(round_uniform, "[{}] barrier round reads identically through every member (B)", label);

    // D — the summed per-head loss fell between each head's first and last scored steps. Loose by
    //     design — one deterministic descent, not convergence; each head is measured against itself.
    suite.check(std::isfinite(last_scored_sum) && last_scored_sum < first_scored_sum,
                "[{}] mean loss fell over {} steps: got {} want < {} (D)", label, config.steps, last_scored_sum,
                first_scored_sum);

    // E — every live head's loss read nonzero at least once: the keyed partition fed every modality rows, and
    //     every head actually trained.
    suite.check(token_trained, "[{}] the token head trained (E)", label);
    if (config.batch.contains(modality_t::image_k))
        suite.check(image_trained, "[{}] the image head trained (E)", label);
    if (config.batch.contains(modality_t::video_k))
        suite.check(video_trained, "[{}] the video head trained (E)", label);
    if (config.batch.contains(modality_t::audio_k))
        suite.check(audio_trained, "[{}] the audio head trained (E)", label);

    // F — the captured window moved the weights off their seed: gradients propagated through the whole
    //     backbone and the optimizer applied them.
    suite.check(!snapshots_match_(seed_lead.value_, captured_lead.value_),
                "[{}] the captured window moved the weights off their seed (F)", label);

    // C — capture == eager: reseed the replicas identically, take fresh moments on the second window, rerun the
    //     same steps eagerly, and land bitwise on the captured run's weights. The strongest single check.
    if (expected_t reseeded = trainer.reseed(); !reseeded)
        return suite.check(false, "[{}] reseed: {}", label, reseeded.description_);
    expected<distributed_adafactor_t> eager_optimizer = make_optimizer_(trainer, domain.value_, colocated.value_);
    if (!eager_optimizer)
        return suite.check(false, "[{}] eager optimizer setup: {}", label, eager_optimizer.description_);
    if (expected_t ran = run_steps_(trainer, eager_optimizer.value_, {}, 0u, config.steps); !ran)
        return suite.check(false, "[{}] eager run: {}", label, ran.description_);
    synchronize_(trainer);
    expected<array<bf16_t>> eager_lead = snapshot_weights_(trainer.replicas[0].weights, trainer.streams[0]);
    suite.check(eager_lead && snapshots_match_(eager_lead.value_, captured_lead.value_),
                "[{}] captured replay lands bitwise on the eager run (C)", label);
}

/** @brief One sweep cell: bring up the trainer, run two eager steps, then capture and replay two more over the
 *      same trainer — both model-region modes under one mixture — asserting finite losses and the final-norm
 *      extents every head reconstructs against. */
static void run_sweep_cell_(test_suite_t &suite, mesh_topology_t const &topology, sz::string_view label,
                            base_config_t const &config) {
    base_trainer_t trainer;
    if (expected_t built = base_trainer_t::make(trainer, config, topology); !built)
        return suite.check(false, "[{}] trainer bring-up: {}", label, built.description_);
    if (trainer.replicas.size() == 0) return suite.check(false, "[{}] no visible GPU", label);
    expected<multimem_domain_t> domain = make_domain_(topology, trainer, 1u);
    if (!domain) return suite.check(false, "[{}] domain setup: {}", label, domain.description_);
    expected<colocated_domain_t> colocated = colocated_domain_t::make(topology);
    if (!colocated) return suite.check(false, "[{}] colocated setup: {}", label, colocated.description_);
    expected<distributed_adafactor_t> optimizer = make_optimizer_(trainer, domain.value_, colocated.value_);
    if (!optimizer) return suite.check(false, "[{}] optimizer setup: {}", label, optimizer.description_);

    // The eager state first, then the captured one over the same trainer — capture must also hold after the
    // optimizer has stepped (the capture-stable addresses survive weight updates).
    if (expected_t ran = run_steps_(trainer, optimizer.value_, {}, 0u, 2u); !ran)
        return suite.check(false, "[{}] eager steps: {}", label, ran.description_);
    f32_t const eager_loss = trainer.telemetry(domain.value_, optimizer_step_t {1u}).composite();
    per_gpu_array<gpu_replay_graph_t> graphs;
    if (expected_t captured = trainer.capture(graphs); !captured)
        return suite.check(false, "[{}] capture: {}", label, captured.description_);
    if (expected_t ran = run_steps_(trainer, optimizer.value_, graphs, 2u, 4u); !ran)
        return suite.check(false, "[{}] replayed steps: {}", label, ran.description_);
    f32_t const replayed_loss = trainer.telemetry(domain.value_, optimizer_step_t {3u}).composite();

    suite.check(std::isfinite(eager_loss) && std::isfinite(replayed_loss),
                "[{}] losses finite through both modes: eager {} replayed {}", label, eager_loss, replayed_loss);
    std::size_t const max_tokens = static_cast<std::size_t>(config.batch.max_rows_per_gpu) *
                                   config.batch.max_columns_per_row;
    bf16r2_view_t const normalized = trainer.replicas[0].tower.normalized();
    suite.check(normalized.extent(0) == max_tokens && normalized.extent(1) == config.hidden,
                "[{}] normalized carries [max_tokens, hidden]: got [{}, {}] want [{}, {}]", label, normalized.extent(0),
                normalized.extent(1), max_tokens, config.hidden);
}

/** @brief One sweep cell's config: @p subset 's media bits over @p rows full-width rows of @p columns cells. */
[[nodiscard]] static base_config_t sweep_config_(u32_t subset, u32_t rows, u32_t columns) noexcept {
    base_config_t config = test_geometry_(false);
    config.batch = multimodal_batch_shape_t {rows, rows, columns};
    config.steps = 4;
    if (subset & 1u) config.batch.image = image_batch_shape_t {8u, 3u};
    if (subset & 2u) config.batch.video = video_batch_shape_t {8u, 3u, 2u, 1u};
    if (subset & 4u) config.batch.audio = audio_batch_shape_t {256u, 16000u};
    return config;
}

/** @brief The two-tier sweep. The cheap tier walks every media subset (2³) crossed with one-row and two-row
 *      budgets on an isolated one-GPU world — every modality-presence combination, the zero-live-rows corner,
 *      and the 1-member peer-read domain — plus two corners the cross-product never reaches: an omni cell with
 *      five rows over four live modalities (the uneven partition, whose extra row rotates across epochs) and an
 *      omni cell at 96 columns (the non-square patch factorization). The wide tier reruns every subset at the
 *      bootstrapped world's full width, so shape handling composes with the real collectives — and widens to
 *      the multi-node world automatically under the sbatch launcher. */
static void mixture_sweep(test_suite_t &suite, mesh_topology_t const &world, mesh_topology_t const &narrow) {
    for (u32_t subset = 0; subset != 8u; ++subset)
        for (u32_t rows = 1; rows != 3u; ++rows) {
            sz::string label;
            if (!format_into(label, "1gpu:tokens{}{}{} rows={}", (subset & 1u) ? "+image" : "",
                             (subset & 2u) ? "+video" : "", (subset & 4u) ? "+audio" : "", rows))
                return suite.check(false, "sweep label format failed");
            run_sweep_cell_(suite, narrow, label.view(), sweep_config_(subset, rows, 64u));
        }
    run_sweep_cell_(suite, narrow, "1gpu:omni rows=5", sweep_config_(7u, 5u, 64u));
    run_sweep_cell_(suite, narrow, "1gpu:omni rows=2 columns=96", sweep_config_(7u, 2u, 96u));
    base_config_t tuned = sweep_config_(7u, 2u, 64u);
    tuned.algo_pick = algo_pick_t::benchmarked_k;
    run_sweep_cell_(suite, narrow, "1gpu:omni rows=2 tuned", tuned);
    for (u32_t subset = 0; subset != 8u; ++subset) {
        sz::string label;
        if (!format_into(label, "wide:tokens{}{}{} rows=2", (subset & 1u) ? "+image" : "",
                         (subset & 2u) ? "+video" : "", (subset & 4u) ? "+audio" : ""))
            return suite.check(false, "sweep label format failed");
        run_sweep_cell_(suite, world, label.view(), sweep_config_(subset, 2u, 64u));
    }
}

/** @brief The capture contract holds — one replay graph per replica builds — and two replays over one fixed
 *      eager preamble produce a bitwise-identical shared @c d(normalized) . Needs no domain or optimizer: the
 *      model region is pure local math, so this stays green even where multicast is not provisioned. */
static void graph_capture(test_suite_t &suite, mesh_topology_t const &topology) {
    base_config_t const config = test_geometry_(false);
    base_trainer_t trainer;
    if (expected_t built = base_trainer_t::make(trainer, config, topology); !built)
        return suite.check(false, "capture trainer bring-up: {}", built.description_);
    u32_t const gpus = static_cast<u32_t>(trainer.replicas.size());
    if (gpus == 0u) return suite.check(false, "capture: no visible GPU");
    per_gpu_array<gpu_replay_graph_t> graphs;
    if (expected_t captured = trainer.capture(graphs); !captured)
        return suite.check(false, "model-region capture: {}", captured.description_);
    suite.check(graphs.size() == gpus, "one captured model region per replica: got {} want {}", graphs.size(), gpus);

    switch_to_gpu_t gpu_scope {trainer.streams[0]};
    if (!gpu_scope.ok()) return suite.check(false, "capture context entry");
    if (expected_t prepared = trainer.replicas[0].prepare(config, 0u, data_epoch_t {0u}, trainer.streams[0]); !prepared)
        return suite.check(false, "capture preamble: {}", prepared.description_);
    if (expected_t first = graphs[0].launch(trainer.streams[0].get()); !first)
        return suite.check(false, "first replay: {}", first.description_);
    expected<array<bf16_t>> first_grad = snapshot_grad_(trainer.replicas[0], trainer.streams[0]);
    if (expected_t second = graphs[0].launch(trainer.streams[0].get()); !second)
        return suite.check(false, "second replay: {}", second.description_);
    expected<array<bf16_t>> second_grad = snapshot_grad_(trainer.replicas[0], trainer.streams[0]);
    suite.check(first_grad && second_grad && snapshots_match_(first_grad.value_, second_grad.value_),
                "two captured replays are bitwise identical");
}

int main() {
    test_suite_t suite {"base"};

    // The schedule's properties are pure host math — asserted before any GPU work.
    learning_rate_schedule(suite);

    if (cuInit(0) != CUDA_SUCCESS) {
        suite.check(false, "cuInit failed");
        return suite.finish();
    }

    // One process per host_domain; the sbatch launcher raises UFORM_HOST_DOMAIN_COUNT for the two-node tier,
    // and single-host worlds need no exchange directory.
    u32_t const host_domain_index = host_domain_index_from_environment();
    char const *const domain_count_text = std::getenv("UFORM_HOST_DOMAIN_COUNT");
    u32_t const host_domain_count = domain_count_text != nullptr ? static_cast<u32_t>(std::atoi(domain_count_text))
                                                                 : 1u;
    expected<mesh_topology_t> topology = mesh_topology_t::bootstrap(host_domain_index, host_domain_count);
    if (!topology) {
        suite.check(false, "mesh bootstrap: {}", topology.description_);
        return suite.finish();
    }

    expected<mesh_topology_t> narrow = narrowed_topology_(topology.value_, 1u);
    if (!narrow) {
        suite.check(false, "narrowed topology: {}", narrow.description_);
        return suite.finish();
    }

    run_base_conformance(suite, "tokens", test_geometry_(false), topology.value_);
    run_base_conformance(suite, "omni", test_geometry_(true), topology.value_);
    mixture_sweep(suite, topology.value_, narrow.value_);
    graph_capture(suite, narrow.value_);
    return suite.finish();
}
