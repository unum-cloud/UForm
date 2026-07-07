/**
 *  @brief The Base training conformance test: builds the same @ref base_trainer_t the binary drives and asserts
 *      the invariants the loop itself never checks — deep replay conformance for the tokens and omni mixtures,
 *      plus a cross-product sweep over every media subset, row budget, and model-region mode.
 *  @file cuda_test/base.cu
 *  @author Ash Vardanian
 *  @date July 6, 2026
 *
 *  This is a test: a plain @c int @c main builds one @ref test_suite_t and threads it by reference into every
 *  case, consuming @c cuda_train/base.cuh 's typed surface instead of copying the driver. The collective
 *  surfaces — one multimem domain and one cross-domain all-reduce per width tier — are created once in @c main and
 *  threaded by reference into every case, exactly as @ref base_trainer_t::run composes production; each case
 *  keeps only its own trainer and optimizer. Launched under Slurm via @c cuda_test/base-4-8-16.sbatch , the
 *  same binary runs the distributed tiers: @c main honors @c UFORM_HOST_DOMAIN_COUNT , so a two-node world
 *  exercises the InfiniBand reduce under every case below.
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

/** @brief One optimizer window's payload budget at the tiny test geometry, with generous margin — the omni
 *      mixture's registration list needs ~1.7M scalars. A tier's shared domain multiplies this by its
 *      window count; @c reserve fails loudly if a geometry change ever outgrows it. */
inline constexpr std::size_t tier_window_scalars_k = 4u << 20;

/** @brief The test tier's rendezvous deadline — short enough that a dead peer never costs the surviving
 *      domain two minutes per case, but long enough for the real two-node cold start: the FIRST exchange
 *      rides NFS attribute-cache latency plus the peers' eager-module-loading skew, which can take tens
 *      of seconds (a warm local rendezvous completes in well under one). */
inline constexpr f64_t test_rendezvous_timeout_seconds_k = 60.0;

/** @brief One width tier's shared surfaces: the topology plus the collective state every cell of the tier
 *      reuses instead of re-creating — threaded by reference from @c main , like production. */
struct tier_t {
    /** @brief The tier's world description. */
    mesh_topology_t const &topology;
    /** @brief The tier's one multicast domain — every case reserves its own window from it. */
    multimem_domain_t &domain;
    /** @brief The tier's one cross-domain all-reduce — every case's optimizer borrows it. */
    infiniband_all_reduce<f32_t> &world_reduce;
};

/** @brief One collective surface shared by every consumer of a tier — multicast objects are a scarce
 *      fabric resource, so the suite creates one per tier in @c main and threads it by reference, exactly
 *      as @ref base_trainer_t::run composes production. Bring-up streams are transient locals (the domain
 *      retains no stream), and the self-advancing barrier rounds make reuse across many optimizer
 *      instances safe by construction: every barrier derives its target from its own member's scalar. */
[[nodiscard]] static expected<multimem_domain_t> make_tier_domain_(mesh_topology_t const &topology,
                                                                   u32_t windows) noexcept {
    using result_t = expected<multimem_domain_t>;
    u32_t const gpus = topology.shape.gpus_per_host_domain;
    array<i32_t> devices;
    per_gpu_array<gpu_stream_t> streams;
    for (u32_t device = 0; device != gpus; ++device) {
        if (devices.try_push_back(static_cast<i32_t>(device)) != status_t::success_k)
            return result_t::fail(status_t::bad_alloc_k, "tier device table allocation failed");
        expected<gpu_stream_t> stream = gpu_stream_t::make(static_cast<i32_t>(device));
        if (!stream) return result_t::fail(std::move(stream));
        if (streams.try_push_back(std::move(stream.value_)) != status_t::success_k)
            return result_t::fail(status_t::bad_alloc_k, "tier stream table full");
    }
    return multimem_domain_t::make(topology, devices.view(), static_cast<std::size_t>(windows) * tier_window_scalars_k,
                                   streams);
}

/** @brief The suite-lifetime arena hosting the tiers' world-reduce heaps — on device 0, visible to every
 *      local device, because the ranks peer-read the gateway's world sum over NVLink. */
[[nodiscard]] static expected<gpu_arena_t> make_reduce_arena_(u32_t gpus) noexcept {
    using result_t = expected<gpu_arena_t>;
    array<i32_t> devices;
    for (u32_t device = 0; device != gpus; ++device)
        if (devices.try_push_back(static_cast<i32_t>(device)) != status_t::success_k)
            return result_t::fail(status_t::bad_alloc_k, "reduce-arena device table allocation failed");
    expected<gpu_stream_t> lead_stream = gpu_stream_t::make(0);
    if (!lead_stream) return result_t::fail(std::move(lead_stream));
    switch_to_gpu_t gpu_scope {lead_stream.value_};
    if (!gpu_scope.ok()) return result_t::fail(status_t::cuda_error_k, "context entry failed");
    return gpu_arena_t::make(0, 256ull << 20, devices.view());
}

/** @brief One tier's world all-reduce buffer, wired once and borrowed by every case — sized like a domain
 *      window ( @ref tier_window_scalars_k , the elementwise reduce leaves a smaller case's tail unread).
 *      A single-domain tier gets the zero-round identity, wired without touching InfiniBand. */
[[nodiscard]] static expected<infiniband_all_reduce<f32_t>> make_tier_reduce_(colocated_domain_t &colocated,
                                                                              gpu_arena_t &arena) noexcept {
    using result_t = expected<infiniband_all_reduce<f32_t>>;
    expected<gpu_stream_t> wiring_stream = gpu_stream_t::make(0);
    if (!wiring_stream) return result_t::fail(std::move(wiring_stream));
    switch_to_gpu_t gpu_scope {wiring_stream.value_};
    if (!gpu_scope.ok()) return result_t::fail(status_t::cuda_error_k, "context entry failed");
    result_t wired = infiniband_all_reduce<f32_t>::make(colocated, 0, arena, tier_window_scalars_k,
                                                        wiring_stream.value_, test_rendezvous_timeout_seconds_k);
    // The round-scalar seed is enqueued on the transient wiring stream — drain it before the stream dies.
    if (wired)
        if (CUresult synced = cuStreamSynchronize(wiring_stream.value_.get()); synced != CUDA_SUCCESS)
            return result_t::fail(synced, "tier reduce wiring drain failed");
    return wired;
}

/** @brief The distributed Adafactor over the transient flat replica-major registration table — the same
 *      construction @ref base_trainer_t::run performs. A second call on the same domain reserves the next
 *      disjoint window (fresh moments), borrowing the tier's one world all-reduce. */
[[nodiscard]] static expected<distributed_adafactor_t> make_optimizer_(
    base_trainer_t &trainer, multimem_domain_t &domain, infiniband_all_reduce<f32_t> &world_reduce) noexcept {
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
    return distributed_adafactor_t::make(domain, world_reduce, trainer.arenas,
                                         {registrations.view().data(), registrations.size()},
                                         trainer.replicas[0].weights.size(), optimizer_config, trainer.streams);
}

/** @brief The world's health, owned by @c main beside the suite and threaded by reference into every case:
 *      a device-side fault poisons the local context (every later case fails with the same unrelated
 *      "unspecified launch failure"), and a dead peer domain makes every later rendezvous wait its full
 *      deadline — either way the first detection reports once and the remaining cases return immediately,
 *      so a faulted run finishes in seconds instead of minutes. */
struct world_health_t {
    /** @brief Cleared at the first poisoned-context or unresponsive-peer detection; later cases skip. */
    bool alive = true;
    /** @brief Consecutive optimizer-setup failures — a healthy world never fails setup, so two in a row
     *      mean the peer domain is gone. */
    u32_t consecutive_setup_failures = 0;
};

/** @brief Probe @p health before a case: report the first poisoned-context or unresponsive-peer detection
 *      once, and tell every later case to return immediately.
 *  @param[inout] suite The suite the one abort line reports into.
 *  @param[inout] health The world's health record, cleared at the first detection. */
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

/** @brief Drain every member's stream before a case's trainer — and the wired collectives borrowing its
 *      arenas — leaves scope: a doorbell or completion-drain kernel still in flight would otherwise poke
 *      NIC structures the teardown frees, and mlx5's pools recycle those pages into the next case's
 *      queues (a roaming device fault the page registry alone cannot prevent). */
[[nodiscard]] static expected_t drain_all_streams_(base_trainer_t &trainer) noexcept {
    for (std::size_t member_index = 0; member_index != trainer.streams.size(); ++member_index) {
        switch_to_gpu_t gpu_scope {trainer.streams[member_index]};
        if (!gpu_scope.ok())
            return expected_t::fail(status_t::cuda_error_k, "context entry failed (member={})", member_index);
        if (CUresult synced = cuStreamSynchronize(trainer.streams[member_index].get()); synced != CUDA_SUCCESS)
            return expected_t::fail(synced, "end-of-case stream drain failed (member={})", member_index);
    }
    return {};
}

/** @brief Bring up the trainer for one mixture, run the captured two-region loop reading telemetry every step,
 *      and fold the five invariants (A replica identity, B round progression, C capture == eager, D loss
 *      descent, E every live head trained) into @p suite . */
static void run_base_conformance(test_suite_t &suite, world_health_t &health, char const *label,
                                 base_config_t const &config, mesh_topology_t const &topology,
                                 multimem_domain_t &domain, infiniband_all_reduce<f32_t> &world_reduce) {
    if (!world_still_healthy_(suite, health)) return;
    base_trainer_t trainer;
    if (expected_t built = base_trainer_t::make(trainer, config, topology); !built)
        return suite.check(false, "[{}] trainer bring-up: {}", label, built.description_);
    u32_t const gpus = static_cast<u32_t>(trainer.replicas.size());
    suite.check(gpus != 0u, "[{}] at least one visible GPU: got {}", label, gpus);
    if (gpus == 0u) return;
    check_weight_naming_(suite, label, config, trainer.replicas[0].weights.view());

    // The captured optimizer and the assertion-C eager optimizer each reserve a fresh disjoint window
    // from the tier's shared domain and borrow the tier's one world all-reduce.
    expected<distributed_adafactor_t> optimizer = make_optimizer_(trainer, domain, world_reduce);
    if (!optimizer) {
        ++health.consecutive_setup_failures;
        return suite.check(false, "[{}] optimizer setup: {}", label, optimizer.description_);
    }
    health.consecutive_setup_failures = 0;
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
        base_trainer_t::telemetry_t const telemetry = trainer.telemetry(domain, optimizer_step_t {ordinal});
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
    concurrency_round_t const lead_round = read_round_(domain, 0u, trainer.streams[0]);
    bool round_uniform = true;
    for (u32_t member_index = 0; member_index < gpus; ++member_index)
        round_uniform = round_uniform &&
                        read_round_(domain, member_index, trainer.streams[member_index]).ordinal == lead_round.ordinal;
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
    expected<distributed_adafactor_t> eager_optimizer = make_optimizer_(trainer, domain, world_reduce);
    if (!eager_optimizer)
        return suite.check(false, "[{}] eager optimizer setup: {}", label, eager_optimizer.description_);
    if (expected_t ran = run_steps_(trainer, eager_optimizer.value_, {}, 0u, config.steps); !ran)
        return suite.check(false, "[{}] eager run: {}", label, ran.description_);
    synchronize_(trainer);
    expected<array<bf16_t>> eager_lead = snapshot_weights_(trainer.replicas[0].weights, trainer.streams[0]);
    suite.check(eager_lead && snapshots_match_(eager_lead.value_, captured_lead.value_),
                "[{}] captured replay lands bitwise on the eager run (C)", label);

    if (expected_t drained = drain_all_streams_(trainer); !drained)
        suite.check(false, "[{}] end-of-case drain: {}", label, drained.description_);
}

/** @brief One sweep cell: bring up the trainer, run two eager steps, then capture and replay two more over the
 *      same trainer — both model-region modes under one mixture — asserting finite losses and the final-norm
 *      extents every head reconstructs against. */
static void run_sweep_cell_(test_suite_t &suite, world_health_t &health, mesh_topology_t const &topology,
                            multimem_domain_t &domain, infiniband_all_reduce<f32_t> &world_reduce,
                            sz::string_view label, base_config_t const &config) {
    if (!world_still_healthy_(suite, health)) return;
    base_trainer_t trainer;
    if (expected_t built = base_trainer_t::make(trainer, config, topology); !built)
        return suite.check(false, "[{}] trainer bring-up: {}", label, built.description_);
    if (trainer.replicas.size() == 0) return suite.check(false, "[{}] no visible GPU", label);
    expected<distributed_adafactor_t> optimizer = make_optimizer_(trainer, domain, world_reduce);
    if (!optimizer) {
        ++health.consecutive_setup_failures;
        return suite.check(false, "[{}] optimizer setup: {}", label, optimizer.description_);
    }
    health.consecutive_setup_failures = 0;

    // The eager state first, then the captured one over the same trainer — capture must also hold after the
    // optimizer has stepped (the capture-stable addresses survive weight updates).
    if (expected_t ran = run_steps_(trainer, optimizer.value_, {}, 0u, 2u); !ran)
        return suite.check(false, "[{}] eager steps: {}", label, ran.description_);
    f32_t const eager_loss = trainer.telemetry(domain, optimizer_step_t {1u}).composite();
    per_gpu_array<gpu_replay_graph_t> graphs;
    if (expected_t captured = trainer.capture(graphs); !captured)
        return suite.check(false, "[{}] capture: {}", label, captured.description_);
    if (expected_t ran = run_steps_(trainer, optimizer.value_, graphs, 2u, 4u); !ran)
        return suite.check(false, "[{}] replayed steps: {}", label, ran.description_);
    f32_t const replayed_loss = trainer.telemetry(domain, optimizer_step_t {3u}).composite();

    suite.check(std::isfinite(eager_loss) && std::isfinite(replayed_loss),
                "[{}] losses finite through both modes: eager {} replayed {}", label, eager_loss, replayed_loss);
    std::size_t const max_tokens = static_cast<std::size_t>(config.batch.max_rows_per_gpu) *
                                   config.batch.max_columns_per_row;
    bf16r2_view_t const normalized = trainer.replicas[0].tower.normalized();
    suite.check(normalized.extent(0) == max_tokens && normalized.extent(1) == config.hidden,
                "[{}] normalized carries [max_tokens, hidden]: got [{}, {}] want [{}, {}]", label, normalized.extent(0),
                normalized.extent(1), max_tokens, config.hidden);

    if (expected_t drained = drain_all_streams_(trainer); !drained)
        suite.check(false, "[{}] end-of-case drain: {}", label, drained.description_);
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
static void mixture_sweep(test_suite_t &suite, world_health_t &health, tier_t wide, tier_t cheap) {
    for (u32_t subset = 0; subset != 8u; ++subset)
        for (u32_t rows = 1; rows != 3u; ++rows) {
            sz::string label;
            if (!format_into(label, "1gpu:tokens{}{}{} rows={}", (subset & 1u) ? "+image" : "",
                             (subset & 2u) ? "+video" : "", (subset & 4u) ? "+audio" : "", rows))
                return suite.check(false, "sweep label format failed");
            run_sweep_cell_(suite, health, cheap.topology, cheap.domain, cheap.world_reduce, label.view(),
                            sweep_config_(subset, rows, 64u));
        }
    run_sweep_cell_(suite, health, cheap.topology, cheap.domain, cheap.world_reduce, "1gpu:omni rows=5",
                    sweep_config_(7u, 5u, 64u));
    run_sweep_cell_(suite, health, cheap.topology, cheap.domain, cheap.world_reduce, "1gpu:omni rows=2 columns=96",
                    sweep_config_(7u, 2u, 96u));
    base_config_t tuned = sweep_config_(7u, 2u, 64u);
    tuned.algo_pick = algo_pick_t::benchmarked_k;
    run_sweep_cell_(suite, health, cheap.topology, cheap.domain, cheap.world_reduce, "1gpu:omni rows=2 tuned", tuned);
    for (u32_t subset = 0; subset != 8u; ++subset) {
        sz::string label;
        if (!format_into(label, "wide:tokens{}{}{} rows=2", (subset & 1u) ? "+image" : "",
                         (subset & 2u) ? "+video" : "", (subset & 4u) ? "+audio" : ""))
            return suite.check(false, "sweep label format failed");
        run_sweep_cell_(suite, health, wide.topology, wide.domain, wide.world_reduce, label.view(),
                        sweep_config_(subset, 2u, 64u));
    }
}

/** @brief The capture contract holds — one replay graph per replica builds — and two replays over one fixed
 *      eager preamble produce a bitwise-identical shared @c d(normalized) . Needs no domain or optimizer: the
 *      model region is pure local math, so this stays green even where multicast is not provisioned. */
static void graph_capture(test_suite_t &suite, world_health_t &health, mesh_topology_t const &topology) {
    if (!world_still_healthy_(suite, health)) return;
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
    // Lazy module loading deadlocks against the InfiniBand doorbell kernels' flag spins: the spinning
    // wait blocks its context, the peer's next launch blocks on module loading behind it, and neither
    // side progresses. Eager loading before `cuInit` front-loads every module — the mixer suite's fix.
    ::setenv("CUDA_MODULE_LOADING", "EAGER", 1);

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
    u32_t const host_domain_count = host_domain_count_from_environment();
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

    // The collective surfaces are created once per width tier and threaded by reference — multicast
    // objects are a scarce fabric resource, and the self-advancing barrier rounds make one domain safe
    // for every consumer in turn. Window budget: 2 per conformance case + 8 wide cells, 19 cheap cells.
    expected<multimem_domain_t> wide_domain = make_tier_domain_(topology.value_, 16u);
    if (!wide_domain) {
        suite.check(false, "wide-tier domain: {}", wide_domain.description_);
        return suite.finish();
    }
    expected<colocated_domain_t> wide_colocated = colocated_domain_t::make(topology.value_);
    if (!wide_colocated) {
        suite.check(false, "wide-tier colocated surface: {}", wide_colocated.description_);
        return suite.finish();
    }
    expected<multimem_domain_t> cheap_domain = make_tier_domain_(narrow.value_, 24u);
    if (!cheap_domain) {
        suite.check(false, "cheap-tier domain: {}", cheap_domain.description_);
        return suite.finish();
    }
    expected<colocated_domain_t> cheap_colocated = colocated_domain_t::make(narrow.value_);
    if (!cheap_colocated) {
        suite.check(false, "cheap-tier colocated surface: {}", cheap_colocated.description_);
        return suite.finish();
    }

    // Each tier's one cross-domain all-reduce buffer, wired here — at the composition root — and borrowed
    // by every case's optimizer, exactly as each case reserves its window from the tier's shared domain.
    // Its heap lives in a suite-lifetime arena visible to every device (the ranks peer-read the world sum),
    // sized like the domain payload; the cheap tier's single-domain buffer is the zero-round identity.
    expected<gpu_arena_t> reduce_arena = make_reduce_arena_(topology.value_.shape.gpus_per_host_domain);
    if (!reduce_arena) {
        suite.check(false, "tier reduce arena: {}", reduce_arena.description_);
        return suite.finish();
    }
    expected<infiniband_all_reduce<f32_t>> wide_reduce = make_tier_reduce_(wide_colocated.value_, reduce_arena.value_);
    if (!wide_reduce) {
        suite.check(false, "wide-tier world all-reduce: {}", wide_reduce.description_);
        return suite.finish();
    }
    expected<infiniband_all_reduce<f32_t>> cheap_reduce = make_tier_reduce_(cheap_colocated.value_,
                                                                            reduce_arena.value_);
    if (!cheap_reduce) {
        suite.check(false, "cheap-tier world all-reduce: {}", cheap_reduce.description_);
        return suite.finish();
    }

    tier_t const wide {topology.value_, wide_domain.value_, wide_reduce.value_};
    tier_t const cheap {narrow.value_, cheap_domain.value_, cheap_reduce.value_};
    world_health_t health;

    run_base_conformance(suite, health, "tokens", test_geometry_(false), wide.topology, wide.domain, wide.world_reduce);
    run_base_conformance(suite, health, "omni", test_geometry_(true), wide.topology, wide.domain, wide.world_reduce);
    mixture_sweep(suite, health, wide, cheap);
    graph_capture(suite, health, narrow.value_);
    return suite.finish();
}
