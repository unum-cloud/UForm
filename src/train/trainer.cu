/**
 *  @brief The composition declared in @c src/train/trainer.hpp : the mesh bootstrap, the trainer's wire-ordered
 *      construction and its step, and the two batch sources.
 *  @file src/train/trainer.cu
 *  @author Ash Vardanian
 *  @date October 18, 2026
 *
 *  @section trainer_impl_items How caller items reach the mixer
 *
 *  An items step plans every item on the host through the same decoders the encode verbs use, groups items by bag,
 *  and deals the bags round-robin over the devices, each bag whole on one device and each item one row-aligned
 *  segment. The pinned host batches upload to the loaded batches, and the mixer routes them into the replicas'
 *  batches. Under a contrastive objective the mixer's routes are read back, every item's mesh-wide vertex is
 *  resolved, and each device's relations are rebuilt over the vertices it holds, their ids rebased onto the
 *  vertices the referenced items landed on.
 */
#include <algorithm> // `std::max`, `std::min`, `std::sort`
#include <charconv>  // `std::from_chars` — the rows peers measured
#include <limits>    // `std::numeric_limits`

#include <cuda.h> // `cuInit`, `cuStreamSynchronize`

#include "media/audio.hpp"    // `decode_audio`
#include "media/image.hpp"    // `plan_image`, `decode_image`, `image_channels_k`
#include "media/planning.hpp" // `video_tube_grid`
#include "media/video.hpp"    // `demux_video`, `decode_video_frames`, `video_channels_k`
#include "train/trainer.hpp"

using namespace unum::uform::core;
using namespace unum::uform::train;
namespace um = unum::uform::media;

/** A vertex, item or segment index that names nothing. */
static constexpr u32_t absent_k = std::numeric_limits<u32_t>::max();

/** One head's @c {λ·Σ, count} summand pair — the currency the ledger's loss rows accumulate. */
struct summand_pair_t {
    f64_t sum = 0.0;
    f64_t count = 0.0;
};

/** Read one head's @b [2] @c {λ·Σ, count} summands back to host. Best-effort: a slipped readback
 *  reports zeros, never fails the run. Caller holds the stream's context. */
[[nodiscard]] static summand_pair_t read_summands_(f32r1_tensor_arena_t &summands, gpu_stream_t &stream) noexcept {
    f32r1_tensor_pinned_t staging = f32r1_tensor_pinned_t::uninitialized({2}, pinned_alloc<f32_t> {}).value;
    if (!staging) return {};
    if (!copy_device_to_host(summands.view(), staging.span(), stream)) return {};
    cuStreamSynchronize(stream.get());
    return summand_pair_t {static_cast<f64_t>(staging[0]), static_cast<f64_t>(staging[1])};
}

#pragma region Mesh

expected_t mesh_t::make(mesh_t &placed, mesh_options_t const &options, std::span<i32_t const> devices) noexcept {
    if (devices.empty()) return expected_t::fail(status_t::invalid_input_k, "a mesh needs at least one device");
    if (options.host_domain_count == 0 || options.host_domain_index >= options.host_domain_count)
        return expected_t::fail(status_t::invalid_input_k, "host domain {} of {} is out of range",
                                options.host_domain_index, options.host_domain_count);
    if (options.host_domain_count > 1 && (options.directory.empty() || options.job.empty()))
        return expected_t::fail(status_t::invalid_input_k, "a run with peers needs an exchange directory and a job");
    if (CUresult result = cuInit(0); result != CUDA_SUCCESS) return cuda_failure(result, "driver start failed");
    sz::string_t directory, job;
    if (sz::failed(directory.assign(options.directory)) || sz::failed(job.assign(options.job)))
        return expected_t::fail(status_t::bad_alloc_k, "exchange path staging failed");
    u32_t const multimem = options.gpus_per_multimem_domain ? options.gpus_per_multimem_domain
                                                            : static_cast<u32_t>(devices.size());
    expected<mesh_topology_t> topology = mesh_topology_t::bootstrap(
        options.host_domain_index, options.host_domain_count, multimem, directory.c_str(), job.c_str(),
        options.timeout_seconds);
    if (!topology) return expected_t::fail(std::move(topology));

    bool own_members = devices.size() == topology.value_.shape.gpus_per_host_domain;
    for (std::size_t local = 0; local != devices.size() && own_members; ++local) // ordinal d must name member d
        own_members = devices[local] == static_cast<i32_t>(local);
    if (!own_members)
        return expected_t::fail(status_t::shape_unsupported_k,
                                "the executor must name this process's {} NVLink-domain members, ordinals 0..{} in " //
                                "member order",
                                topology.value_.shape.gpus_per_host_domain,
                                topology.value_.shape.gpus_per_host_domain - 1u);
    placed.topology = std::move(topology.value_);
    expected<colocated_domain_t> colocated = colocated_domain_t::make(placed.topology);
    if (!colocated) return expected_t::fail(std::move(colocated));
    placed.colocated = std::move(colocated.value_);
    placed.timeout_seconds = options.timeout_seconds;
    return {};
}

#pragma endregion Mesh

#pragma region Envelope

/** The two envelopes the measuring pass fits through, in rows per GPU: small enough that any model
 *  that trains at all can be measured, and far enough apart that the driver's allocation
 *  granularity can't flatten the slope. */
static constexpr u32_t smaller_probe_rows_k = 1u, larger_probe_rows_k = 4u;

/** One device's footprint after a probe step: the bytes the driver handed out since the run began,
 *  and the arena's share of them. */
struct footprint_t {
    std::size_t device_bytes = 0, arena_bytes = 0;
};

/** Bind @p placed to @p mesh , @p config and @p devices , make one stream per device, and record
 *  what each device has free into @p free_bytes — the part of @ref trainer_t::make before the
 *  envelope is known. */
[[nodiscard]] static expected_t prepare_(trainer_t &placed, mesh_t &mesh, trainer_config_t const &config,
                                         std::span<i32_t const> devices, std::span<std::size_t> free_bytes) noexcept {
    mesh_topology_t const &topology = mesh.topology;
    u32_t const gpus = static_cast<u32_t>(devices.size());
    if (gpus == 0 || gpus != topology.shape.gpus_per_host_domain)
        return expected_t::fail(status_t::invalid_input_k, "the trainer drives the mesh's {} devices, not {}",
                                topology.shape.gpus_per_host_domain, gpus);
    placed.config = config;
    placed.mesh = &mesh;
    if (placed.devices.assign(devices) != status_t::success_k)
        return expected_t::fail(status_t::bad_alloc_k, "device table allocation failed");
    for (u32_t member = 0; member != gpus; ++member) {
        expected<gpu_stream_t> stream = gpu_stream_t::make(devices[member]);
        if (!stream) return expected_t::fail(std::move(stream));
        if (placed.streams.push_back(std::move(stream.value_)) != status_t::success_k)
            return expected_t::fail(status_t::bad_alloc_k, "stream table full");
        switch_to_gpu_t gpu_scope {placed.streams[member]};
        if (!gpu_scope.ok()) return expected_t::fail(status_t::cuda_error_k, "context entry failed");
        expected<gpu_memory_report_t> memory = gpu_memory_report(devices[member]);
        if (!memory) return expected_t::fail(std::move(memory));
        free_bytes[member] = memory->free_bytes;
    }
    placed.envelope.free_bytes = *std::min_element(free_bytes.begin(), free_bytes.end());
    return {};
}

[[nodiscard]] static expected_t assemble_(trainer_t &placed, mesh_t &mesh, std::size_t reserve_bytes) noexcept;

/** Build a trainer over @p config at @p rows rows and segments per GPU, its arenas reserving all
 *  the free memory, take one synthetic step, and read every device's footprint against
 *  @p free_bytes , what it had free before the first probe. */
[[nodiscard]] static expected_t probe_footprint_(mesh_t &mesh, trainer_config_t config, std::span<i32_t const> devices,
                                                 u32_t rows, std::span<std::size_t const> free_bytes,
                                                 std::span<footprint_t> footprints) noexcept {
    config.batch.max_rows_per_gpu = rows;
    config.batch.max_segments_per_gpu = rows;
    trainer_t probe;
    std::size_t probe_free_bytes[max_gpus_per_host_k] = {};
    expected_t built = prepare_(probe, mesh, config, devices, {probe_free_bytes, devices.size()});
    if (built) built = assemble_(probe, mesh, probe.envelope.free_bytes);
    if (!built) return built;
    synthetic_source_t source;
    step_report_t report;
    if (expected_t stepped = probe.step(source, optimizer_step_t {0}, report); !stepped) return stepped;
    if (expected_t drained = probe.synchronize(); !drained) return drained;
    for (std::size_t member = 0; member != devices.size(); ++member) {
        switch_to_gpu_t gpu_scope {probe.streams[member]};
        if (!gpu_scope.ok()) return expected_t::fail(status_t::cuda_error_k, "context entry failed");
        expected<gpu_memory_report_t> memory = gpu_memory_report(devices[member]);
        if (!memory) return expected_t::fail(std::move(memory));
        footprints[member].device_bytes = free_bytes[member] - std::min(memory->free_bytes, free_bytes[member]);
        footprints[member].arena_bytes = probe.arenas[member].used(); // exact, where mapped() moves whole chunks
    }
    return {};
}

/** Agree on the fewest @p rows any process measured: the lead collects every peer's count and
 *  publishes the minimum, which every peer adopts. */
[[nodiscard]] static expected<u32_t> agree_on_rows_(mesh_t const &mesh, u32_t rows) noexcept {
    using result_t = expected<u32_t>;
    mesh_topology_t const &topology = mesh.topology;
    if (topology.host_domain_count() <= 1) return result_t {rows};
    sz::string_t text;
    auto parsed = [&](u32_t &value) noexcept {
        return std::from_chars(text.data(), text.data() + text.size(), value).ec == std::errc {};
    };
    if (!topology.is_lead()) {
        u32_t agreed = 0;
        if (!format_into(text, "{}", rows)) return result_t::fail(status_t::bad_alloc_k, "rows formatting failed");
        expected_t exchanged = topology.publish_text("envelope", text.view());
        if (exchanged) exchanged = topology.await_text(0u, "envelope.agreed", text, mesh.timeout_seconds);
        if (!exchanged) return result_t::fail(std::move(exchanged));
        if (!parsed(agreed))
            return result_t::fail(status_t::invalid_input_k, "the lead agreed on {} rows", text.view());
        return result_t {agreed};
    }
    u32_t agreed = rows;
    for (u32_t peer = 1; peer != topology.host_domain_count(); ++peer) {
        u32_t peer_rows = 0;
        if (expected_t awaited = topology.await_text(peer, "envelope", text, mesh.timeout_seconds); !awaited)
            return result_t::fail(std::move(awaited));
        if (!parsed(peer_rows))
            return result_t::fail(status_t::invalid_input_k, "host domain {} measured {} rows", peer, text.view());
        agreed = std::min(agreed, peer_rows);
    }
    text.clear();
    if (!format_into(text, "{}", agreed)) return result_t::fail(status_t::bad_alloc_k, "rows formatting failed");
    if (expected_t published = topology.publish_text("envelope.agreed", text.view()); !published)
        return result_t::fail(std::move(published));
    return result_t {agreed};
}

/**
 *  @brief The rows per GPU at @p config 's columns — the configured ones, or the most every device
 *      of every process holds — and the arena reservation they need.
 *
 *  Each device's footprint is fitted as @c fixed+rows·row through the two probes, and the margin
 *  is the footprint outside the arena at the larger probe, the decoders' workspaces, graphs and
 *  the symmetric heap, held free once more as the part a larger envelope may grow nonlinearly.
 *  The arena's own high-water mark is fitted the same way, and the reservation is that line one
 *  row past the envelope: the batch sources allocate their loaded batches from the arena after
 *  the trainer, and one row of the trainer's own growth holds them.
 *  @retval status_t::bad_alloc_k The configured rows do not fit, with the measured numbers.
 */
[[nodiscard]] static expected<envelope_t> measure_envelope_(mesh_t &mesh, trainer_config_t const &config,
                                                            std::span<i32_t const> devices,
                                                            std::span<std::size_t const> free_bytes) noexcept {
    using result_t = expected<envelope_t>;
    std::size_t const gpus = devices.size();
    footprint_t smaller[max_gpus_per_host_k] = {}, larger[max_gpus_per_host_k] = {};
    expected_t probed = probe_footprint_(mesh, config, devices, smaller_probe_rows_k, free_bytes, {smaller, gpus});
    if (probed) probed = probe_footprint_(mesh, config, devices, larger_probe_rows_k, free_bytes, {larger, gpus});
    if (!probed) return result_t::fail(std::move(probed));

    u32_t const probe_span = larger_probe_rows_k - smaller_probe_rows_k;
    u32_t const configured_rows = config.batch.max_rows_per_gpu;
    u64_t const representable_rows = std::numeric_limits<u32_t>::max() / config.batch.max_columns_per_row;
    envelope_t envelope; // the device holding the fewest rows decides
    for (std::size_t member = 0; member != gpus; ++member) {
        footprint_t const &low = smaller[member], &high = larger[member];
        if (high.device_bytes <= low.device_bytes || high.arena_bytes < low.arena_bytes)
            return result_t::fail(status_t::other_k, "device {} took {} bytes at {} rows and {} at {}", devices[member],
                                  low.device_bytes, smaller_probe_rows_k, high.device_bytes, larger_probe_rows_k);
        envelope_t measured;
        measured.free_bytes = free_bytes[member];
        measured.row_bytes = (high.device_bytes - low.device_bytes) / probe_span;
        measured.fixed_bytes = low.device_bytes - std::min(low.device_bytes, measured.row_bytes * smaller_probe_rows_k);
        measured.margin_bytes = high.device_bytes - std::min(high.device_bytes, high.arena_bytes);
        std::size_t const kept_bytes = measured.fixed_bytes + measured.margin_bytes;
        u64_t const fitting_rows = measured.free_bytes > kept_bytes
                                       ? (measured.free_bytes - kept_bytes) / measured.row_bytes
                                       : 0u;
        measured.rows = configured_rows ? configured_rows
                                        : static_cast<u32_t>(std::min(fitting_rows, representable_rows));
        if (measured.rows == 0 || measured.rows > fitting_rows)
            return result_t::fail(
                status_t::bad_alloc_k,
                "device {} holds {} rows, not {}: {} bytes fixed, {} per row and a {}-byte margin, " //
                "{} free",
                devices[member], fitting_rows, measured.rows, measured.fixed_bytes, measured.row_bytes,
                measured.margin_bytes, measured.free_bytes);
        if (member == 0 || measured.rows < envelope.rows) envelope = measured;
    }
    if (!configured_rows) {
        expected<u32_t> agreed = agree_on_rows_(mesh, envelope.rows);
        if (!agreed) return result_t::fail(std::move(agreed));
        envelope.rows = *agreed;
    }
    for (std::size_t member = 0; member != gpus; ++member) {
        std::size_t const arena_row_bytes = (larger[member].arena_bytes - smaller[member].arena_bytes) / probe_span;
        std::size_t const arena_fixed_bytes = smaller[member].arena_bytes - arena_row_bytes * smaller_probe_rows_k;
        envelope.reserve_bytes = std::max(envelope.reserve_bytes,
                                          arena_fixed_bytes + arena_row_bytes * (std::size_t {envelope.rows} + 1u));
    }
    return result_t {envelope};
}

#pragma endregion Envelope

#pragma region Trainer

/** Everything @ref trainer_t::make builds after the streams and the envelope, over @p placed 's
 *  sized config, each arena reserving @p reserve_bytes . */
[[nodiscard]] static expected_t assemble_(trainer_t &placed, mesh_t &mesh, std::size_t reserve_bytes) noexcept {
    trainer_config_t const &config = placed.config;
    mesh_topology_t const &topology = mesh.topology;
    std::span<i32_t const> const devices = placed.devices.view();
    u32_t const gpus = static_cast<u32_t>(devices.size());
    for (u32_t member = 0; member != gpus; ++member) {
        switch_to_gpu_t gpu_scope {placed.streams[member]};
        if (!gpu_scope.ok()) return expected_t::fail(status_t::cuda_error_k, "context entry failed");
        expected<gpu_arena_t> arena = gpu_arena_t::make(devices[member], reserve_bytes, devices);
        if (!arena) return expected_t::fail(std::move(arena));
        if (placed.arenas.push_back(std::move(arena.value_)) != status_t::success_k)
            return expected_t::fail(status_t::bad_alloc_k, "arena table full");
    }

    { // rows stay inert zero handles when the channel is dark or the head absent, so feeding never branches
        switch_to_gpu_t gpu_scope {placed.streams[0]};
        if (!gpu_scope.ok()) return expected_t::fail(status_t::cuda_error_k, "context entry failed");
        expected<telemetry_ledger_t> ledger = telemetry_ledger_t::make(placed.arenas[0], config.telemetry_channels,
                                                                       256u, placed.streams[0]);
        if (!ledger) return expected_t::fail(std::move(ledger));
        placed.ledger = std::move(ledger.value_);
    }
    auto loss_probe = [&](char const *name, bool live, telemetry_probe_t &into) noexcept -> expected_t {
        if (!live) return {};
        expected<telemetry_probe_t> registered = placed.ledger.register_probe(name, telemetry_losses_k,
                                                                              finite_signal_k);
        if (!registered) return expected_t::fail(std::move(registered));
        into = registered.value_;
        return {};
    };
    bool const reconstructs = config.reconstructs();
    if (expected_t probed = loss_probe("base/token_ce", reconstructs, placed.probes.token); !probed) return probed;
    if (expected_t probed = loss_probe("base/image_mse", reconstructs && config.batch.contains(modality_t::image_k),
                                       placed.probes.image);
        !probed)
        return probed;
    if (expected_t probed = loss_probe("base/video_mse", reconstructs && config.batch.contains(modality_t::video_k),
                                       placed.probes.video);
        !probed)
        return probed;
    if (expected_t probed = loss_probe("base/audio_mse", reconstructs && config.batch.contains(modality_t::audio_k),
                                       placed.probes.audio);
        !probed)
        return probed;
    if (expected_t probed = loss_probe("base/contrastive", config.contrasts(), placed.probes.contrastive); !probed)
        return probed;
    for (u32_t member = 0; member != gpus; ++member) {
        sz::string_t name;
        if (!format_into(name, "base/round/{}", member))
            return expected_t::fail(status_t::bad_alloc_k, "round probe name formatting failed");
        expected<telemetry_probe_t> registered = placed.ledger.register_probe(name.view(), telemetry_rounds_k,
                                                                              monotone_signal_k);
        if (!registered) return expected_t::fail(std::move(registered));
        if (placed.probes.rounds.push_back(registered.value_) != status_t::success_k)
            return expected_t::fail(status_t::bad_alloc_k, "round probe table full");
    }

    std::size_t const segments = config.batch.max_segments_per_gpu;
    std::size_t const vertices_per_mesh = segments * topology.shape.gpus_per_colocated_domain;
    for (u32_t member = 0; member != gpus; ++member) {
        switch_to_gpu_t gpu_scope {placed.streams[member]};
        if (!gpu_scope.ok()) return expected_t::fail(status_t::cuda_error_k, "context entry failed");
        expected<base_replica_t> replica = base_replica_t::make(placed.arenas[member], config, vertices_per_mesh);
        if (!replica) return expected_t::fail(std::move(replica));
        if (placed.replicas.push_back(std::move(replica.value_)) != status_t::success_k)
            return expected_t::fail(status_t::bad_alloc_k, "replica table full");
    }
    if (expected_t seeded = placed.reseed(); !seeded) return seeded;

    array<differentiable_weights_t> registrations; // a transient the optimizer copies into its per-replica state
    for (u32_t member = 0; member != gpus; ++member)
        for (differentiable_weights_t const &entry : placed.replicas[member].weights.view()) {
            expected<differentiable_weights_t> copied = entry.copy();
            if (!copied || registrations.push_back(std::move(copied.value_)) != status_t::success_k)
                return expected_t::fail(status_t::bad_alloc_k, "registration table allocation failed");
        }
    std::span<differentiable_weights_t const> const lead_weights = placed.replicas[0].weights.view();
    std::size_t const gradient_scalars = distributed_adafactor_t::payload_scalars_needed(lead_weights);
    contrastive_segments_config_t contrastive_config;
    contrastive_config.from_vertices_per_gpu = static_cast<u32_t>(segments);
    contrastive_config.hidden = config.hidden;
    contrastive_config.to_vertices_per_gpu = static_cast<u32_t>(segments);
    contrastive_config.gpu_subgraphs_per_step = topology.shape.gpus_per_colocated_domain;
    std::size_t const force_scalars = config.contrasts() ? distributed_contrastive_segments_t::payload_scalars_needed(
                                                               contrastive_config, topology.shape)
                                                         : 0u;
    std::size_t const mixer_scalars = multimodal_batch_mixer_t::payload_scalars_needed(config.batch, topology.shape);
    expected<multimem_domain_t> domain = multimem_domain_t::make(
        topology, placed.devices.view(), gradient_scalars + force_scalars + mixer_scalars, placed.streams);
    if (!domain) return expected_t::fail(std::move(domain));
    placed.domain = std::move(domain.value_);
    {
        switch_to_gpu_t gpu_scope {placed.streams[0]};
        if (!gpu_scope.ok()) return expected_t::fail(status_t::cuda_error_k, "context entry failed");
        expected<infiniband_all_reduce<f32_t>> world_reduce = infiniband_all_reduce<f32_t>::make(
            mesh.colocated, 0, placed.arenas[0], gradient_scalars, placed.streams[0]);
        if (!world_reduce) return expected_t::fail(std::move(world_reduce));
        placed.world_reduce = std::move(world_reduce.value_);
    }
    expected<distributed_adafactor_t> optimizer = distributed_adafactor_t::make(
        placed.domain, placed.world_reduce, placed.arenas, registrations.view(), lead_weights.size(),
        adafactor_config_t {}, placed.streams, placed.ledger);
    if (!optimizer) return expected_t::fail(std::move(optimizer));
    placed.optimizer = std::move(optimizer.value_);

    if (config.contrasts()) {
        per_gpu_array<bf16r2_view_t> towers;
        per_gpu_array<contrastive_relations_view_t> relations;
        per_gpu_array<bf16r2_span_t> forces;
        for (u32_t member = 0; member != gpus; ++member) {
            base_replica_t &replica = placed.replicas[member];
            if (towers.push_back(replica.vertices.view()) != status_t::success_k ||
                relations.push_back(replica.relations.view()) != status_t::success_k ||
                forces.push_back(replica.vertices_grad.span()) != status_t::success_k)
                return expected_t::fail(status_t::bad_alloc_k, "contrastive view table full");
        }
        u32_t const full_width[1] = {config.hidden};
        u32r1_view_t const prefixes = config.contrastive.matryoshka_count
                                          ? u32r1_view_t {config.contrastive.matryoshka_dimensions,
                                                          config.contrastive.matryoshka_count}
                                          : u32r1_view_t {full_width, 1};
        expected<distributed_contrastive_segments_t> contrastive = distributed_contrastive_segments_t::make(
            placed.domain, mesh.colocated, placed.arenas, placed.streams, towers.view(), relations.view(),
            forces.view(), contrastive_config, prefixes);
        if (!contrastive) return expected_t::fail(std::move(contrastive));
        placed.contrastive = std::move(contrastive.value_);
    }
    if (config.capture == replayed_k) return placed.capture();
    return {};
}

expected_t trainer_t::make(trainer_t &placed, mesh_t &mesh, trainer_config_t const &requested,
                           std::span<i32_t const> devices) noexcept {
    std::size_t free_bytes[max_gpus_per_host_k] = {};
    if (expected_t prepared = prepare_(placed, mesh, requested, devices, {free_bytes, devices.size()}); !prepared)
        return prepared;
    expected<envelope_t> measured = measure_envelope_(mesh, requested, devices, {free_bytes, devices.size()});
    if (!measured) return expected_t::fail(std::move(measured));
    placed.envelope = *measured;
    if (!requested.batch.max_rows_per_gpu) {
        placed.config.batch.max_rows_per_gpu = placed.envelope.rows;
        placed.config.batch.max_segments_per_gpu = placed.envelope.rows;
    }

    expected_t assembled = assemble_(placed, mesh, placed.envelope.reserve_bytes);
    if (assembled || assembled.status_ != status_t::bad_alloc_k) return assembled;
    std::size_t committed_bytes = 0;
    for (gpu_arena_t const &arena : placed.arenas.view()) committed_bytes = std::max(committed_bytes, arena.mapped());
    return expected_t::fail(status_t::bad_alloc_k,
                            "{}: {} rows of {} columns per GPU outgrew the device, its arena committed {} of the {} " //
                            "bytes free",
                            assembled.description_.view(), placed.config.batch.max_rows_per_gpu,
                            placed.config.batch.max_columns_per_row, committed_bytes, placed.envelope.free_bytes);
}

/** Upload the saved Float32 vector @p name into @p destination when @p checkpoint holds it, on
 *  @p stream . */
[[nodiscard]] static expected_t restore_vector_(checkpoint_to_load_t const &checkpoint, sz::string_view_t name,
                                                f32r1_span_t destination, gpu_stream_t &stream) noexcept {
    f32r2_view_t const saved = checkpoint.float32(name);
    if (saved.empty()) return {};
    if (saved.extent(0) != 1u || saved.extent(1) != destination.extent(0))
        return expected_t::fail(status_t::shape_unsupported_k, "{} holds [{},{}], the run [1,{}]", name,
                                saved.extent(0), saved.extent(1), destination.extent(0));
    return copy_host_to_device(saved.flatten<1>(), destination, stream);
}

/** Download the device vector @p values into @p checkpoint as the @b [1,n] Float32 entry
 *  @p name . */
[[nodiscard]] static expected_t insert_vector_(checkpoint_to_save_t &checkpoint, sz::string_view_t name,
                                               f32r1_view_t values, gpu_stream_t &stream) noexcept {
    array<f32_t> staging;
    if (staging.resize_uninitialized(values.extent(0)) != status_t::success_k)
        return expected_t::fail(status_t::bad_alloc_k, "{} staging failed", name);
    if (expected_t copied = copy_device_to_host(values, staging.span(), stream); !copied) return copied;
    if (CUresult result = cuStreamSynchronize(stream.get()); result != CUDA_SUCCESS)
        return cuda_failure(result, "{} download failed", name);
    return checkpoint.insert(name, f32r2_view_t {staging.data(), 1, staging.size()});
}

expected_t trainer_t::resume(sz::string_view_t path) noexcept {
    switch_to_gpu_t lead_scope {streams[0]};
    if (!lead_scope.ok()) return expected_t::fail(status_t::cuda_error_k, "context entry failed");
    expected<checkpoint_to_load_t> loaded = checkpoint_to_load_t::make(path);
    if (!loaded) return expected_t::fail(std::move(loaded));
    for (std::size_t member = 0; member != replicas.size(); ++member) {
        switch_to_gpu_t gpu_scope {streams[member]};
        if (expected_t warmed = load_weights(loaded.value_, replicas[member].weights, streams[member]); !warmed)
            return warmed;
        if (!config.contrasts()) continue;
        base_replica_t &replica = replicas[member];
        expected_t restored = restore_vector_(*loaded, "contrastive.inverse_temperatures",
                                              replica.inverse_temperatures.span(), streams[member]);
        if (restored) restored = restore_vector_(*loaded, "contrastive.biases", replica.biases.span(), streams[member]);
        if (!restored) return restored;
    }
    // After the weights, since every state the file misses restarts from them.
    if (expected_t restored = optimizer.load(*loaded, streams); !restored) return restored;
    if (expected_t drained = synchronize(); !drained) return drained; // the uploads read the payload `loaded` maps
    completed_steps = static_cast<u32_t>(loaded.value_.step());
    return {};
}

expected_t trainer_t::capture() noexcept {
    for (std::size_t member = 0; member != replicas.size(); ++member) {
        switch_to_gpu_t gpu_scope {streams[member]};
        if (!gpu_scope.ok()) return expected_t::fail(status_t::cuda_error_k, "context entry failed");
        expected<gpu_replay_graph_t> forward = gpu_replay_graph_t::make(
            streams[member].get(), [&]() noexcept { return replicas[member].forward_region(streams[member]); });
        if (!forward) return expected_t::fail(std::move(forward));
        expected<gpu_replay_graph_t> backward = gpu_replay_graph_t::make(
            streams[member].get(), [&]() noexcept { return replicas[member].backward_region(streams[member]); });
        if (!backward) return expected_t::fail(std::move(backward));
        if (forward_graphs.push_back(std::move(forward.value_)) != status_t::success_k ||
            backward_graphs.push_back(std::move(backward.value_)) != status_t::success_k)
            return expected_t::fail(status_t::bad_alloc_k, "model-graph table full");
    }
    return {};
}

expected_t trainer_t::step(batch_source_t &source, optimizer_step_t optimizer_step, step_report_t &report) noexcept {
    data_epoch_t const data_epoch {optimizer_step.ordinal};
    report.optimizer_step = optimizer_step;
    report.learning_rate = config.learning_rate_for(optimizer_step);
    report.truncated = 0;
    if (expected_t filled = source.fill(*this, data_epoch, report); !filled) return filled;
    std::size_t const gpus = replicas.size();
    for (std::size_t member = 0; member != gpus; ++member) {
        switch_to_gpu_t gpu_scope {streams[member]};
        if (!gpu_scope.ok()) return expected_t::fail(status_t::cuda_error_k, "context entry failed");
        expected_t stepped = replicas[member].prepare(config, data_epoch, streams[member]);
        if (stepped)
            stepped = forward_graphs.size() ? forward_graphs[member].launch(streams[member].get())
                                            : replicas[member].forward_region(streams[member]);
        if (!stepped) return stepped;
    }
    if (config.contrasts())
        for (std::size_t member = 0; member != gpus; ++member)
            if (expected_t contrasted = contrastive(member, replicas[member].recipe(config), streams[member]);
                !contrasted)
                return contrasted;
    for (std::size_t member = 0; member != gpus; ++member) {
        switch_to_gpu_t gpu_scope {streams[member]};
        if (!gpu_scope.ok()) return expected_t::fail(status_t::cuda_error_k, "context entry failed");
        expected_t stepped = backward_graphs.size() ? backward_graphs[member].launch(streams[member].get())
                                                    : replicas[member].backward_region(streams[member]);
        if (stepped) stepped = optimizer.step(member, report.learning_rate, streams[member]);
        if (!stepped) return stepped;
    }
    completed_steps = optimizer_step.ordinal + 1u;
    return {};
}

expected_t trainer_t::reseed() noexcept {
    u32_t const groups = std::max(config.contrastive.temperature_groups, 1u);
    array<f32_t> inverse_temperatures, biases;
    if (config.contrasts()) {
        if (inverse_temperatures.resize(groups) != status_t::success_k || biases.resize(groups) != status_t::success_k)
            return expected_t::fail(status_t::bad_alloc_k, "temperature staging failed");
        for (u32_t group = 0; group != groups; ++group)
            inverse_temperatures[group] = config.contrastive.inverse_temperature,
            biases[group] = config.contrastive.bias;
    }
    for (std::size_t member = 0; member != replicas.size(); ++member) {
        switch_to_gpu_t gpu_scope {streams[member]};
        if (!gpu_scope.ok()) return expected_t::fail(status_t::cuda_error_k, "context entry failed");
        base_replica_t &replica = replicas[member];
        if (expected_t seeded = seed_weights(replica.weights, streams[member]); !seeded) return seeded;
        if (!config.contrasts()) continue;
        expected_t staged = copy_host_to_device(f32r1_view_t {inverse_temperatures.data(), groups},
                                                replica.inverse_temperatures.span(), streams[member]);
        if (staged)
            staged = copy_host_to_device(f32r1_view_t {biases.data(), groups}, replica.biases.span(), streams[member]);
        if (staged) staged = zero_async(replica.temperature_groups.span(), streams[member]);
        if (!staged) return staged;
        cuStreamSynchronize(streams[member].get());
    }
    return {};
}

telemetry_t trainer_t::telemetry(optimizer_step_t optimizer_step) noexcept {
    telemetry_t telemetry;
    telemetry.optimizer_step = optimizer_step;
    std::size_t const gpus = replicas.size();
    if (gpus == 0) return telemetry;
    // Accumulate every member's summands before the fold — at wide worlds a lead-only reading is a coin flip,
    // not a statistic. The all-member pairs are also what the ledger's loss rows accumulate.
    summand_pair_t token {}, image {}, video {}, audio {};
    auto accumulate = [](summand_pair_t &into, f32r1_tensor_arena_t &summands, gpu_stream_t &stream) noexcept {
        if (summands.empty()) return;
        summand_pair_t const read = read_summands_(summands, stream);
        into.sum += read.sum;
        into.count += read.count;
    };
    for (std::size_t member = 0; member != gpus; ++member) {
        switch_to_gpu_t gpu_scope {streams[member]};
        if (!gpu_scope.ok()) continue;
        cuStreamSynchronize(streams[member].get());
        base_replica_t &replica = replicas[member];
        accumulate(token, replica.tokens_summands, streams[member]);
        accumulate(image, replica.image_summands, streams[member]);
        accumulate(video, replica.video_summands, streams[member]);
        accumulate(audio, replica.audio_summands, streams[member]);
        (void)ledger.probe_round(probes.rounds[member], domain.member(domain.member_offset() + member),
                                 streams[member]);
    }
    telemetry.token_ce = token.count != 0.0 ? static_cast<f32_t>(token.sum / token.count) : 0.0f;
    telemetry.image_mse = image.count != 0.0 ? static_cast<f32_t>(image.sum / image.count) : 0.0f;
    telemetry.video_mse = video.count != 0.0 ? static_cast<f32_t>(video.sum / video.count) : 0.0f;
    telemetry.audio_mse = audio.count != 0.0 ? static_cast<f32_t>(audio.sum / audio.count) : 0.0f;
    ledger.scalar(probes.token, token.sum, token.count);
    ledger.scalar(probes.image, image.sum, image.count);
    ledger.scalar(probes.video, video.sum, video.count);
    ledger.scalar(probes.audio, audio.sum, audio.count);

    switch_to_gpu_t gpu_scope {streams[0]};
    if (!gpu_scope.ok()) return telemetry;
    if (config.contrasts()) {
        f32r1_tensor_pinned_t staging = f32r1_tensor_pinned_t::uninitialized({gpus}, pinned_alloc<f32_t> {}).value;
        if (staging && copy_device_to_host(contrastive.loss_outs(), staging.span(), streams[0])) {
            cuStreamSynchronize(streams[0].get());
            f64_t sum = 0.0;
            for (std::size_t member = 0; member != gpus; ++member) sum += staging[member];
            telemetry.contrastive = static_cast<f32_t>(sum);
            ledger.scalar(probes.contrastive, sum, 1.0);
        }
    }
    if (u32_t const *const round = domain.member(domain.member_offset()).round; round != nullptr) {
        u32r1_tensor_pinned_t staging = u32r1_tensor_pinned_t::uninitialized({1}, pinned_alloc<u32_t> {}).value;
        nk::tensor_view<u32_t, 1> const source {round, 1};
        if (staging && copy_device_to_host(source, staging.span(), streams[0])) {
            cuStreamSynchronize(streams[0].get());
            telemetry.concurrency_round = concurrency_round_t {staging[0]};
        }
    }
    return telemetry;
}

expected_t trainer_t::save(sz::string_view_t path) noexcept {
    if (!mesh->topology.is_lead()) return {};
    switch_to_gpu_t gpu_scope {streams[0]};
    if (!gpu_scope.ok()) return expected_t::fail(status_t::cuda_error_k, "context entry failed");
    checkpoint_to_save_t checkpoint; // the Float32 state goes first, so every entry lands aligned
    expected_t saved = optimizer.save(checkpoint, streams[0]);
    if (saved && config.contrasts())
        saved = insert_vector_(checkpoint, "contrastive.inverse_temperatures", replicas[0].inverse_temperatures.view(),
                               streams[0]);
    if (saved && config.contrasts())
        saved = insert_vector_(checkpoint, "contrastive.biases", replicas[0].biases.view(), streams[0]);
    if (saved) saved = save_weights(checkpoint, config, replicas[0].weights, completed_steps, streams[0]);
    if (saved) saved = checkpoint.save(path);
    return saved;
}

expected_t trainer_t::synchronize() noexcept {
    for (std::size_t member = 0; member != streams.size(); ++member) {
        switch_to_gpu_t gpu_scope {streams[member]};
        if (!gpu_scope.ok())
            return expected_t::fail(status_t::cuda_error_k, "context entry failed (member={})", member);
        if (CUresult result = cuStreamSynchronize(streams[member].get()); result != CUDA_SUCCESS)
            return cuda_failure(result, "stream drain failed (member={})", member);
    }
    return {};
}

void telemetry_t::report() const noexcept {
    sz::string_t line;
    if (format_into(line, "  step {:>4}  masked CE {:.6f}  round {}", optimizer_step.ordinal + 1u, token_ce,
                    concurrency_round.ordinal))
        report_line(line.view());
    auto term = [&](char const *name, f32_t value) noexcept {
        line.clear();
        if (value != 0.0f && format_into(line, "            {} {:.6f}", name, value)) report_line(line.view());
    };
    term("image MSE", image_mse);
    term("video MSE", video_mse);
    term("audio MSE", audio_mse);
    term("contrastive", contrastive);
}

#pragma endregion Trainer

#pragma region Sources

expected_t synthetic_source_t::fill(trainer_t &trainer, data_epoch_t data_epoch, step_report_t &) noexcept {
    trainer_config_t const &config = trainer.config;
    u32_t const first = trainer.mesh->topology.first_gpu_index();
    u32_t const segments = config.batch.max_segments_per_gpu;
    for (std::size_t member = 0; member != trainer.replicas.size(); ++member) {
        switch_to_gpu_t gpu_scope {trainer.streams[member]};
        if (!gpu_scope.ok()) return expected_t::fail(status_t::cuda_error_k, "context entry failed");
        base_replica_t &replica = trainer.replicas[member];
        u32_t const gpu_index = first + static_cast<u32_t>(member);
        if (expected_t drawn = replica.draw(config, gpu_index, data_epoch, trainer.streams[member]); !drawn)
            return drawn;
        if (!config.contrasts()) continue;
        if (expected_t related = replica.relations.upload_self_edges(
                segments, {}, {}, static_cast<std::size_t>(gpu_index) * segments,
                config.contrastive.ambient_repulsion_weight, trainer.streams[member]);
            !related)
            return related;
        trainer.contrastive.rebind_relations(member, replica.relations.view());
    }
    return {};
}

/** The bytes of item @p item . */
[[nodiscard]] static std::span<b8_t const> item_bytes_(training_items_t const &items, std::size_t item) noexcept {
    return {items.contents + items.offsets[item], items.offsets[item + 1] - items.offsets[item]};
}

/** The video slice of @p shape with its frame stride read as at least one. */
[[nodiscard]] static video_batch_shape_t video_shape_(multimodal_batch_shape_t const &shape) noexcept {
    video_batch_shape_t video = shape.video;
    video.frame_stride = std::max(video.frame_stride, 1u);
    return video;
}

expected_t items_source_t::make(items_source_t &placed, trainer_t &trainer) noexcept {
    trainer_config_t const &config = trainer.config;
    multimodal_batch_shape_t const &shape = config.batch;
    std::size_t const gpus = trainer.replicas.size(), segments = shape.max_segments_per_gpu;
    if (config.contrasts() && trainer.mesh->topology.host_domain_count() > 1)
        return expected_t::fail(status_t::shape_unsupported_k,
                                "relations rebase within one process; a mesh of several processes contrasts caller " //
                                "items once the mixer reports its routes across hosts");
    if (shape.contains(modality_t::image_k) && shape.image.channels != um::image_channels_k)
        return expected_t::fail(status_t::shape_unsupported_k, "decoded images are RGBA, not {} channels",
                                shape.image.channels);
    if (shape.contains(modality_t::video_k) && shape.video.channels != um::video_channels_k)
        return expected_t::fail(status_t::shape_unsupported_k, "decoded video frames are RGB, not {} channels",
                                shape.video.channels);
    for (std::size_t member = 0; member != gpus; ++member) {
        switch_to_gpu_t gpu_scope {trainer.streams[member]};
        if (!gpu_scope.ok()) return expected_t::fail(status_t::cuda_error_k, "context entry failed");
        expected<multimodal_batch_on_gpu_t> loaded = multimodal_batch_on_gpu_t::make(
            gpu_arena_alloc_t {trainer.arenas[member]}, shape);
        if (!loaded) return expected_t::fail(std::move(loaded));
        expected<multimodal_batch_on_cpu_t> staged = multimodal_batch_on_cpu_t::make(pinned_alloc_t {}, shape);
        if (!staged) return expected_t::fail(std::move(staged));
        if (placed.loaded_.push_back(std::move(loaded.value_)) != status_t::success_k ||
            placed.staged_.push_back(std::move(staged.value_)) != status_t::success_k)
            return expected_t::fail(status_t::bad_alloc_k, "batch table full");
    }
    expected<multimodal_batch_mixer_t> mixer = multimodal_batch_mixer_t::make(
        trainer.domain, trainer.mesh->colocated, trainer.arenas, shape, config.seed, trainer.streams);
    if (!mixer) return expected_t::fail(std::move(mixer));
    placed.mixer_ = std::move(mixer.value_);

    std::size_t const total_segments = gpus * segments;
    std::size_t const edges = config.contrastive.max_edges_per_gpu ? config.contrastive.max_edges_per_gpu : segments;
    bool allocated = placed.tokens_.resize(total_segments) == status_t::success_k &&
                     placed.placements_.resize(total_segments) == status_t::success_k &&
                     placed.destinations_.resize(total_segments) == status_t::success_k &&
                     placed.order_.resize(total_segments) == status_t::success_k &&
                     placed.packed_items_.resize(total_segments) == status_t::success_k;
    if (shape.contains(modality_t::audio_k))
        allocated = allocated && placed.window_.resize(std::size_t {shape.max_columns_per_row} *
                                                       shape.audio.chunk_samples) == status_t::success_k;
    if (config.contrasts()) {
        allocated = allocated && placed.offsets_.resize(3 * (segments + 1)) == status_t::success_k &&
                    placed.ids_.resize(3 * edges) == status_t::success_k &&
                    placed.weights_.resize(2 * edges) == status_t::success_k &&
                    placed.lengths_.resize(total_segments) == status_t::success_k &&
                    placed.groups_.resize(segments) == status_t::success_k;
        placed.sources_ = u32r1_tensor_pinned_t::uninitialized({total_segments}, pinned_alloc<u32_t> {}).value;
        placed.pooled_ = nk::tensor<bf16_t, pinned_alloc<bf16_t>, 1>::uninitialized({total_segments * config.hidden},
                                                                                    pinned_alloc<bf16_t> {})
                             .value;
        allocated = allocated && placed.sources_ && placed.pooled_;
    }
    if (!allocated) return expected_t::fail(status_t::bad_alloc_k, "item routing scratch allocation failed");
    return {};
}

expected_t items_source_t::fill(trainer_t &trainer, data_epoch_t data_epoch, step_report_t &report) noexcept {
    if (expected_t packed = pack_(trainer, report); !packed) return packed;
    std::span<multimodal_batch_on_gpu_t const> const loaded = loaded_.view();
    for (std::size_t member = 0; member != trainer.replicas.size(); ++member)
        if (expected_t mixed = mixer_(member, data_epoch, loaded, trainer.replicas[member].batch,
                                      trainer.streams[member]);
            !mixed)
            return mixed;
    if (!trainer.config.contrasts()) return {};
    return relate_(trainer);
}

expected_t items_source_t::pack_(trainer_t &trainer, step_report_t &report) noexcept {
    multimodal_batch_shape_t const &shape = trainer.config.batch;
    std::size_t const gpus = trainer.replicas.size(), count = items.modalities.size();
    u32_t const segments = shape.max_segments_per_gpu, columns = shape.max_columns_per_row;
    u32_t const capacity = shape.max_rows_per_gpu * columns, bags_per_gpu = shape.max_bags_per_gpu;
    if (count > gpus * segments)
        return expected_t::fail(status_t::invalid_input_k, "{} items exceed the {} segments of the devices", count,
                                gpus * segments);
    video_batch_shape_t const video = video_shape_(shape);

    if (expected_t drained = trainer.synchronize(); !drained) return drained; // prior uploads read the pinned twins

    for (std::size_t item = 0; item != count; ++item) {
        modality_t const modality = items.modalities[item];
        std::span<b8_t const> const bytes = item_bytes_(items, item);
        if (!shape.contains(modality))
            return expected_t::fail(status_t::shape_unsupported_k, "item {} has a modality the run left out", item);
        std::size_t kept = 0, spanned = 0;
        if (modality == modality_t::text_k) spanned = bytes.size(), kept = std::min<std::size_t>(spanned, columns);
        else if (modality == modality_t::image_k) {
            expected<um::image_plan_t> planned = um::plan_image(bytes, shape.image.patch_side, columns, 1u);
            if (!planned) return expected_t::fail(std::move(planned));
            kept = std::size_t {planned->grid.rows} * planned->grid.columns;
            spanned = divide_round_up(planned->native.width, shape.image.patch_side) *
                      divide_round_up(planned->native.height, shape.image.patch_side);
        }
        else if (modality == modality_t::video_k) {
            expected<um::demuxed_t> clip = um::demux_video(bytes);
            if (!clip) return expected_t::fail(std::move(clip));
            um::video_grid_t const grid = um::video_tube_grid(clip->frames, clip->extent, video, columns);
            std::size_t const sampled = divide_round_up(clip->frames ? clip->frames : 1u, video.frame_stride);
            kept = std::size_t {grid.temporal_groups} * grid.patch_rows * grid.patch_columns;
            spanned = divide_round_up(sampled, video.tube_frames) *
                      divide_round_up(clip->extent.width, video.patch_side) *
                      divide_round_up(clip->extent.height, video.patch_side);
        }
        else {
            f32r1_span_t const window = window_.span();
            if (expected_t decoded = um::decode_audio(bytes, shape.audio.sample_rate, window); !decoded) return decoded;
            std::size_t heard = window_.size(); // the decoder pads a short clip with exact silence
            while (heard && window_[heard - 1] == 0.0f) --heard;
            std::size_t const frames = divide_round_up(heard, shape.audio.chunk_samples);
            kept = frames ? frames : 1u;
            spanned = frames == columns ? frames + 1u : kept;
        }
        tokens_[item] = static_cast<u32_t>(kept);
        report.truncated += spanned > kept;
        order_[item] = static_cast<u32_t>(item);
        placements_[item] = absent_k;
    }
    if (items.bags)
        std::sort(order_.begin(), order_.begin() + count, [&](u32_t left, u32_t right) noexcept {
            return items.bags[left] != items.bags[right] ? items.bags[left] < items.bags[right] : left < right;
        });

    u32_t cursors[max_gpus_per_host_k] = {}, used_segments[max_gpus_per_host_k] = {},
          used_bags[max_gpus_per_host_k] = {};
    for (std::size_t segment = 0; segment != packed_items_.size(); ++segment) packed_items_[segment] = absent_k;
    std::size_t group = 0;
    for (std::size_t begin = 0, end = 0; begin != count; begin = end, ++group) {
        end = begin + 1;
        while (items.bags && end != count && items.bags[order_[end]] == items.bags[order_[begin]]) ++end;
        u32_t live = 0;
        for (std::size_t position = begin; position != end; ++position) live += tokens_[order_[position]] != 0;
        if (live == 0) continue;
        std::size_t device = gpus;
        for (std::size_t attempt = 0; attempt != gpus && device == gpus; ++attempt) {
            std::size_t const candidate = (group + attempt) % gpus;
            u32_t cursor = cursors[candidate];
            for (std::size_t position = begin; position != end; ++position) {
                u32_t const kept = tokens_[order_[position]];
                if (!kept) continue;
                u32_t const room = columns - cursor % columns;
                cursor += (kept > room ? room : 0u) + kept;
            }
            bool const fits = cursor <= capacity && used_segments[candidate] + live <= segments &&
                              (!bags_per_gpu || used_bags[candidate] < bags_per_gpu);
            if (fits) device = candidate;
        }
        if (device == gpus)
            return expected_t::fail(status_t::invalid_input_k, "the items exceed what the devices' envelopes hold");

        multimodal_batch_on_cpu_t &staged = staged_[device];
        for (std::size_t position = begin; position != end; ++position) {
            u32_t const item = order_[position], kept = tokens_[item];
            if (!kept) continue;
            u32_t const room = columns - cursors[device] % columns;
            if (kept > room) cursors[device] += room;
            u32_t const segment = used_segments[device]++, offset = cursors[device];
            cursors[device] += kept;
            modality_t const modality = items.modalities[item];
            std::span<b8_t const> const bytes = item_bytes_(items, item);
            staged.segment_offsets[segment] = offset;
            staged.segment_lengths[segment] = kept;
            staged.segment_modality[segment] = modality;
            staged.segment_bag[segment] = used_bags[device];
            placements_[item] = static_cast<u32_t>(device * segments + segment);
            packed_items_[device * segments + segment] = item;
            if (modality == modality_t::text_k) {
                token_id_t *const tokens = staged.tokens.data() + offset;
                for (u32_t token = 0; token != kept; ++token) tokens[token] = static_cast<token_id_t>(bytes[token]);
            }
            else if (modality == modality_t::image_k) {
                expected<um::image_plan_t> planned = um::plan_image(bytes, shape.image.patch_side, columns, 1u);
                if (!planned) return expected_t::fail(std::move(planned));
                std::size_t const side = shape.image.patch_side;
                std::size_t const cell = patch_scalars(shape.image.patch_side, shape.image.channels);
                image_shape_span_t const shapes = staged.image_shapes.span();
                shapes[segment, 0] = planned->grid.rows;
                shapes[segment, 1] = planned->grid.columns;
                um::pixel_patches_span_t const cells {
                    staged.images.data() + std::size_t {offset} * cell,
                    {planned->grid.rows, planned->grid.columns, side, side, std::size_t {um::image_channels_k}}};
                if (expected_t decoded = um::decode_image(bytes, *planned, cells); !decoded) return decoded;
            }
            else if (modality == modality_t::video_k) {
                expected<um::demuxed_t> clip = um::demux_video(bytes);
                if (!clip) return expected_t::fail(std::move(clip));
                um::video_grid_t const grid = um::video_tube_grid(clip->frames, clip->extent, video, columns);
                std::size_t const side = video.patch_side;
                std::size_t const cell = tube_scalars(video.tube_frames, video.patch_side, video.channels);
                video_shape_span_t const shapes = staged.video_shapes.span();
                shapes[segment, 0] = grid.temporal_groups;
                shapes[segment, 1] = grid.patch_rows;
                shapes[segment, 2] = grid.patch_columns;
                um::pixel_tubes_span_t const tubes {
                    staged.videos.data() + std::size_t {offset} * cell,
                    {grid.temporal_groups, std::size_t {grid.patch_rows} * grid.patch_columns, video.tube_frames, side,
                     side, std::size_t {um::video_channels_k}}};
                if (expected_t decoded = um::decode_video_frames(*clip, grid, video, tubes); !decoded) return decoded;
            }
            else {
                f32r1_span_t const frames {
                    staged.audio_samples.data() + std::size_t {offset} * shape.audio.chunk_samples,
                    std::size_t {kept} * shape.audio.chunk_samples};
                if (expected_t decoded = um::decode_audio(bytes, shape.audio.sample_rate, frames); !decoded)
                    return decoded;
            }
        }
        ++used_bags[device];
    }

    for (std::size_t device = 0; device != gpus; ++device) {
        multimodal_batch_on_cpu_t &staged = staged_[device];
        multimodal_batch_on_gpu_t &loaded = loaded_[device];
        for (u32_t segment = used_segments[device]; segment != segments; ++segment) {
            staged.segment_offsets[segment] = cursors[device];
            staged.segment_lengths[segment] = 0u;
            staged.segment_modality[segment] = modality_t::text_k;
            staged.segment_bag[segment] = bags_per_gpu;
        }
        staged.segment_offsets[segments] = cursors[device];
        gpu_stream_t &stream = trainer.streams[device];
        switch_to_gpu_t gpu_scope {stream};
        if (!gpu_scope.ok()) return expected_t::fail(status_t::cuda_error_k, "context entry failed");
        expected_t copied = copy_host_to_device(staged.segment_offsets.view(), loaded.segment_offsets.span(), stream);
        if (copied) copied = copy_host_to_device(staged.segment_lengths.view(), loaded.segment_lengths.span(), stream);
        if (copied)
            copied = copy_host_to_device(staged.segment_modality.view(), loaded.segment_modality.span(), stream);
        if (copied) copied = copy_host_to_device(staged.segment_bag.view(), loaded.segment_bag.span(), stream);
        if (copied)
            copied = copy_host_to_device(staged.tokens.view().flatten<1>(), loaded.tokens.span().flatten<1>(), stream);
        if (copied)
            copied = copy_host_to_device(staged.image_shapes.view().flatten<1>(),
                                         loaded.image_shapes.span().flatten<1>(), stream);
        if (copied)
            copied = copy_host_to_device(staged.video_shapes.view().flatten<1>(),
                                         loaded.video_shapes.span().flatten<1>(), stream);
        if (copied)
            copied = copy_host_to_device(staged.images.view().flatten<1>(), loaded.images.span().flatten<1>(), stream);
        if (copied)
            copied = copy_host_to_device(staged.videos.view().flatten<1>(), loaded.videos.span().flatten<1>(), stream);
        if (copied)
            copied = copy_host_to_device(staged.audio_samples.view().flatten<1>(),
                                         loaded.audio_samples.span().flatten<1>(), stream);
        if (!copied) return copied;
    }
    return {};
}

/** Sort one row's edges ascending by id, carrying the weights along, and drop repeated ids — the order the
 *  relation columns' binary search reads. Rows are short, so an insertion sort without scratch. */
[[nodiscard]] static u32_t sort_row_(u32_t *ids, f32_t *weights, u32_t count) noexcept {
    for (u32_t index = 1; index < count; ++index)
        for (u32_t back = index; back != 0 && ids[back - 1] > ids[back]; --back) {
            std::swap(ids[back - 1], ids[back]);
            if (weights) std::swap(weights[back - 1], weights[back]);
        }
    u32_t kept = 0;
    for (u32_t index = 0; index != count; ++index) {
        if (kept && ids[kept - 1] == ids[index]) continue;
        ids[kept] = ids[index];
        if (weights) weights[kept] = weights[index];
        ++kept;
    }
    return kept;
}

expected_t items_source_t::relate_(trainer_t &trainer) noexcept {
    trainer_config_t const &config = trainer.config;
    std::size_t const gpus = trainer.replicas.size(), count = items.modalities.size();
    u32_t const segments = config.batch.max_segments_per_gpu;
    std::size_t const total_segments = gpus * segments, edges = ids_.size() / 3;
    u32_t const groups = std::max(config.contrastive.temperature_groups, 1u);

    for (std::size_t member = 0; member != gpus; ++member) {
        switch_to_gpu_t gpu_scope {trainer.streams[member]};
        if (!gpu_scope.ok()) return expected_t::fail(status_t::cuda_error_k, "context entry failed");
        u32r1_span_t const landing {sources_.data() + member * segments, segments};
        if (expected_t copied = copy_device_to_host(mixer_.mixed_sources(member), landing, trainer.streams[member]);
            !copied)
            return copied;
    }
    if (expected_t drained = trainer.synchronize(); !drained) return drained;
    for (std::size_t item = 0; item != count; ++item) destinations_[item] = absent_k;
    for (std::size_t vertex = 0; vertex != total_segments; ++vertex) {
        u32_t const source = sources_[vertex];
        u32_t const item = source < total_segments ? packed_items_[source] : absent_k;
        sources_[vertex] = item;
        lengths_[vertex] = item != absent_k ? tokens_[item] : 0u;
        if (item != absent_k) destinations_[item] = static_cast<u32_t>(vertex);
    }

    item_edges_t const *const sets[3] = {&items.attract, &items.repel, &items.ignore};
    for (std::size_t member = 0; member != gpus; ++member) {
        u32_t edge_counts[3] = {};
        for (std::size_t set = 0; set != 3; ++set) {
            item_edges_t const &edges_of = *sets[set];
            u32_t *const offsets = offsets_.data() + set * (segments + 1);
            u32_t *const ids = ids_.data() + set * edges;
            f32_t *const weights = set != 2 ? weights_.data() + set * edges : nullptr;
            u32_t written = 0;
            offsets[0] = 0;
            for (u32_t row = 0; row != segments; ++row) {
                u32_t const item = sources_[member * segments + row];
                u32_t const row_begin = written;
                if (item != absent_k && edges_of.offsets)
                    for (std::size_t edge = edges_of.offsets[item]; edge != edges_of.offsets[item + 1]; ++edge) {
                        u32_t const target = edges_of.ids[edge];
                        if (target >= count)
                            return expected_t::fail(status_t::invalid_input_k, "item {} names item {} of {}", item,
                                                    target, count);
                        if (destinations_[target] == absent_k) continue;
                        if (written == edges)
                            return expected_t::fail(status_t::invalid_input_k,
                                                    "a device's edges exceed the {} it holds per set", edges);
                        ids[written] = destinations_[target];
                        if (weights) weights[written] = edges_of.weights ? edges_of.weights[edge] : 1.0f;
                        ++written;
                    }
                written = row_begin +
                          sort_row_(ids + row_begin, weights ? weights + row_begin : nullptr, written - row_begin);
                offsets[row + 1] = written;
            }
            edge_counts[set] = written;
        }
        for (u32_t row = 0; row != segments; ++row) {
            u32_t const item = sources_[member * segments + row];
            u32_t const group = item != absent_k && items.temperature_groups ? items.temperature_groups[item] : 0u;
            if (group >= groups)
                return expected_t::fail(status_t::invalid_input_k, "item {} selects group {} of {}", item, group,
                                        groups);
            groups_[row] = temperature_group_t {group};
        }

        base_replica_t &replica = trainer.replicas[member];
        gpu_stream_t &stream = trainer.streams[member];
        switch_to_gpu_t gpu_scope {stream};
        if (!gpu_scope.ok()) return expected_t::fail(status_t::cuda_error_k, "context entry failed");
        std::size_t const rows = segments + 1u;
        bool const ignores = items.ignore.offsets != nullptr;
        expected_t staged = copy_host_to_device(u32r1_view_t {lengths_.data(), total_segments},
                                                replica.to_vertex_lengths.span(), stream);
        if (staged)
            staged = copy_host_to_device(nk::tensor_view<temperature_group_t, 1> {groups_.data(), segments},
                                         replica.temperature_groups.span(), stream);
        if (staged)
            staged = replica.relations.upload(
                u32r1_view_t {offsets_.data(), rows}, u32r1_view_t {ids_.data(), edge_counts[0]},
                f32r1_view_t {weights_.data(), edge_counts[0]}, u32r1_view_t {offsets_.data() + rows, rows},
                u32r1_view_t {ids_.data() + edges, edge_counts[1]},
                f32r1_view_t {weights_.data() + edges, edge_counts[1]},
                ignores ? u32r1_view_t {offsets_.data() + 2 * rows, rows} : u32r1_view_t {},
                ignores ? u32r1_view_t {ids_.data() + 2 * edges, edge_counts[2]} : u32r1_view_t {},
                replica.batch.segment_lengths.view(), replica.to_vertex_lengths.view(), total_segments,
                member * segments, config.contrastive.ambient_repulsion_weight, stream);
        if (staged) // the host columns are reused for the next device, so their copies land first
            if (CUresult result = cuStreamSynchronize(stream.get()); result != CUDA_SUCCESS)
                staged = cuda_failure(result, "relation staging drain failed (member={})", member);
        if (!staged) return staged;
        trainer.contrastive.rebind_relations(member, replica.relations.view());
    }
    return {};
}

expected<bf16r2_view_t> items_source_t::pooled(trainer_t &trainer) noexcept {
    using result_t = expected<bf16r2_view_t>;
    std::size_t const segments = trainer.config.batch.max_segments_per_gpu, hidden = trainer.config.hidden;
    std::size_t const gpus = trainer.replicas.size();
    if (!pooled_) return result_t::fail(status_t::shape_unsupported_k, "only a contrastive objective pools items");
    for (std::size_t member = 0; member != gpus; ++member) {
        switch_to_gpu_t gpu_scope {trainer.streams[member]};
        if (!gpu_scope.ok()) return result_t::fail(status_t::cuda_error_k, "context entry failed");
        bf16r1_span_t const landing {pooled_.data() + member * segments * hidden, segments * hidden};
        if (expected_t copied = copy_device_to_host(trainer.replicas[member].vertices.view().flatten<1>(), landing,
                                                    trainer.streams[member]);
            !copied)
            return result_t::fail(std::move(copied));
    }
    if (expected_t drained = trainer.synchronize(); !drained) return result_t::fail(std::move(drained));
    return result_t {bf16r2_view_t {pooled_.data(), gpus * segments, hidden}};
}

#pragma endregion Sources
