/**
 *  @file src/train/trainer.cu
 *  @author Ash Vardanian
 *  @date October 18, 2026
 *  @brief The composition declared in @c uform/train/trainer.hpp : the mesh bootstrap, the
 *      trainer's wire-ordered construction and its step, and the two batch sources.
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
#include <cmath> // `std::sqrt` — the rows a curved footprint fits

#include <unistd.h> // `getpid`

#include <algorithm> // `std::max`, `std::min`, `std::sort`
#include <charconv>  // `std::from_chars` — the rows peers measured
#include <limits>    // `std::numeric_limits`

#include <cuda.h> // `cuInit`, `cuStreamSynchronize`
#include <nvml.h> // `nvmlDeviceGetComputeRunningProcesses_v3` — the bytes this process holds

#include "uform/media/audio.hpp"    // `decode_audio`
#include "uform/media/image.hpp"    // `plan_image`, `decode_image`, `image_channels_k`
#include "uform/media/planning.hpp" // `video_tube_grid`
#include "uform/media/video.hpp"    // `demux_video`, `decode_video_frames`, `video_channels_k`
#include "uform/train/trainer.hpp"

using namespace unum::uform::core;
using namespace unum::uform::train;
namespace um = unum::uform::media;

/** A vertex, item or segment index that names nothing. */
static constexpr u32_t absent_k = std::numeric_limits<u32_t>::max();

/** Where an item of @p kept cells and @p modality starts from @p cursor : media on their own half
 *  or whole row, text at the cursor unless it would cross a row. */
[[nodiscard]] static u32_t placed_cursor_(u32_t cursor, u32_t kept, modality_t modality, u32_t columns) noexcept {
    if (u32_t const cells = media_cells(modality, columns); cells != 0u) return (cursor + cells - 1u) / cells * cells;
    u32_t const room = columns - cursor % columns;
    return kept > room ? cursor + room : cursor;
}

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
        options.timeout_seconds, options.capabilities);
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

/** The three envelopes the measuring pass fits through, in rows per GPU: small enough that any
 *  model that trains at all can be measured, and far enough apart that the driver's allocation
 *  granularity can't flatten the slope, while the middle one exposes the growth by rows². */
static constexpr u32_t smaller_probe_rows_k = 1u, middle_probe_rows_k = 4u, larger_probe_rows_k = 16u;

/** A footprint @c fixed+rows·row+rows²·pair in bytes. */
struct footprint_fit_t {
    f64_t fixed = 0.0, row = 0.0, pair = 0.0;

    [[nodiscard]] f64_t at(f64_t rows) const noexcept { return fixed + rows * row + rows * rows * pair; }

    /** The most rows whose footprint stays within @p budget bytes, zero when none does. */
    [[nodiscard]] u64_t rows_within(f64_t budget) const noexcept {
        if (at(1.0) > budget) return 0u;
        f64_t const rows = pair > 0.0 ? (std::sqrt(row * row + 4.0 * pair * (budget - fixed)) - row) / (2.0 * pair)
                                      : (budget - fixed) / row;
        u64_t fitting = static_cast<u64_t>(rows);
        while (fitting != 0u && at(static_cast<f64_t>(fitting)) > budget) --fitting; // the rounding's last row
        return fitting;
    }
};

/** One device's footprint after a probe step: the bytes the driver gave this process since the
 *  run began, and the arena's share of them. */
struct footprint_t {
    std::size_t device_bytes = 0, arena_bytes = 0;
};

/** One device before the run allocated: what it had free, and what this process held. */
struct device_budget_t {
    std::size_t free_bytes = 0, held_bytes = 0;
};

/** The bytes the driver attributes to this process on @p device , as a plain or an MPS client, so a
 *  probe's growth reads as its own whatever else grows on the device meanwhile. */
[[nodiscard]] static expected<std::size_t> held_bytes_(i32_t device) noexcept {
    using result_t = expected<std::size_t>;
    CUdevice handle = 0;
    char bus_id[NVML_DEVICE_PCI_BUS_ID_BUFFER_SIZE] = {};
    if (CUresult result = cuDeviceGet(&handle, device); result != CUDA_SUCCESS)
        return cuda_failure<std::size_t>(result, "device handle query failed (device={})", device);
    if (CUresult result = cuDeviceGetPCIBusId(bus_id, sizeof(bus_id), handle); result != CUDA_SUCCESS)
        return cuda_failure<std::size_t>(result, "PCI bus query failed (device={})", device);
    if (nvmlReturn_t status = nvmlInit_v2(); status != NVML_SUCCESS)
        return result_t::fail(status_t::cuda_error_k, "NVML initialization failed: {}", nvmlErrorString(status));
    nvmlDevice_t nvml_device = nullptr;
    nvmlReturn_t status = nvmlDeviceGetHandleByPciBusId_v2(bus_id, &nvml_device);
    unsigned int const process = static_cast<unsigned int>(getpid());
    std::size_t held = 0, entries = 0;
    st::basic_vector<nvmlProcessInfo_t> processes;
    // An MPS client appears in both lists with the same bytes, so the first list naming this
    // process counts.
    for (auto const query : {&nvmlDeviceGetComputeRunningProcesses_v3, &nvmlDeviceGetMPSComputeRunningProcesses_v3}) {
        if (entries != 0) break;
        unsigned int count = static_cast<unsigned int>(processes.size());
        if (status == NVML_SUCCESS) status = query(nvml_device, &count, processes.data());
        // The list can grow between the size query and the read, so the read retries until it fits.
        while (status == NVML_ERROR_INSUFFICIENT_SIZE)
            status = st::failed(processes.resize(st::assume_uninitialized, count))
                         ? NVML_ERROR_MEMORY
                         : query(nvml_device, &count, processes.data());
        for (std::size_t index = 0; status == NVML_SUCCESS && index != count; ++index)
            if (processes[index].pid == process && processes[index].usedGpuMemory != NVML_VALUE_NOT_AVAILABLE)
                held += processes[index].usedGpuMemory, ++entries;
    }
    nvmlShutdown();
    if (status != NVML_SUCCESS)
        return result_t::fail(status_t::cuda_error_k, "NVML process query failed (device={}): {}", device,
                              nvmlErrorString(status));
    // A PID namespace or a driver withholding per-process bytes would read as zero growth and an
    // unbounded fit.
    if (entries == 0)
        return result_t::fail(status_t::cuda_error_k, "NVML lists no bytes for process {} on device {}", process,
                              device);
    return result_t {held};
}

/** Bind @p placed to @p mesh , @p config , @p devices and @p capabilities , make one stream per
 *  device, and record what each device has free and this process holds into @p budgets — the part
 *  of @ref trainer_t::make before the envelope is known. */
[[nodiscard]] static expected_t prepare_(trainer_t &placed, mesh_t &mesh, trainer_config_t const &config,
                                         std::span<i32_t const> devices, nk_capability_t capabilities,
                                         std::span<device_budget_t> budgets) noexcept {
    mesh_topology_t const &topology = mesh.topology;
    std::size_t const gpus = devices.size();
    if (gpus == 0 || gpus != topology.shape.gpus_per_host_domain)
        return expected_t::fail(status_t::invalid_input_k, "the trainer drives the mesh's {} devices, not {}",
                                topology.shape.gpus_per_host_domain, gpus);
    placed.config = config;
    placed.mesh = &mesh;
    placed.capabilities = capabilities;
    if (st::failed(placed.devices.assign(devices)))
        return expected_t::fail(status_t::bad_alloc_k, "device table allocation failed");
    for (std::size_t member = 0; member != gpus; ++member) {
        expected<gpu_stream_t> stream = gpu_stream_t::make(devices[member]);
        if (!stream) return expected_t::fail(std::move(stream));
        if (placed.streams.push_back(std::move(stream.value_)) != status_t::success_k)
            return expected_t::fail(status_t::bad_alloc_k, "stream table full");
        switch_to_gpu_t gpu_scope {placed.streams[member]};
        if (!gpu_scope.ok()) return expected_t::fail(status_t::cuda_error_k, "context entry failed");
        expected<gpu_memory_report_t> memory = gpu_memory_report(devices[member]);
        if (!memory) return expected_t::fail(std::move(memory));
        expected<std::size_t> held = held_bytes_(devices[member]);
        if (!held) return expected_t::fail(std::move(held));
        budgets[member] = {memory->free_bytes, *held};
    }
    placed.envelope.free_bytes = std::min_element(budgets.begin(), budgets.end(), [](auto const &a, auto const &b) {
                                     return a.free_bytes < b.free_bytes;
                                 })->free_bytes;
    return {};
}

[[nodiscard]] static expected_t assemble_(trainer_t &placed, mesh_t &mesh, std::size_t reserve_bytes) noexcept;

/** How many segments a row of @p config holds: its own ratio when it sets both budgets, as the C
 *  interface does, else the density its data source charged. */
[[nodiscard]] static f64_t segments_per_row_(trainer_config_t const &config) noexcept {
    multimodal_batch_shape_t const &batch = config.batch;
    if (batch.max_rows_per_gpu && batch.max_segments_per_gpu)
        return static_cast<f64_t>(batch.max_segments_per_gpu) / static_cast<f64_t>(batch.max_rows_per_gpu);
    return config.segments_per_row;
}

/** Build a trainer over @p config at @p rows rows and segments per GPU, its arenas reserving all
 *  the free memory, take one synthetic step, and read every device's footprint against
 *  @p budgets , what it had free and held before the first probe. */
[[nodiscard]] static expected_t probe_footprint_(mesh_t &mesh, trainer_config_t config, std::span<i32_t const> devices,
                                                 nk_capability_t capabilities, u32_t rows,
                                                 std::span<device_budget_t const> budgets,
                                                 std::span<footprint_t> footprints) noexcept {
    config.batch = batch_capacity(config.batch, rows, segments_per_row_(config));
    trainer_t probe;
    device_budget_t probe_budgets[max_gpus_per_host_k] = {};
    expected_t built = prepare_(probe, mesh, config, devices, capabilities, {probe_budgets, devices.size()});
    if (built) built = assemble_(probe, mesh, probe.envelope.free_bytes);
    if (!built) return built;
    synthetic_source_t source;
    step_report_t report;
    if (expected_t stepped = probe.step(batch_source_of(source), optimizer_step_t {0}, report); !stepped)
        return stepped;
    if (expected_t drained = probe.synchronize(); !drained) return drained;
    // A data source fills every GPU's batches for the next steps and mixes them from the arenas
    // after the trainer, so the probe allocates the same, and the fit carries them per row; a
    // held-out source allocates its own.
    std::size_t const sources = config.eval_every && !config.eval_mixture.empty() ? 2 : 1;
    st::basic_vector<multimodal_batch_on_gpu_t> filled;
    st::basic_vector<multimodal_batch_mixer_t> mixers;
    for (std::size_t source_index = 0; source_index != sources; ++source_index) {
        for (std::size_t member = 0; member != devices.size(); ++member) {
            switch_to_gpu_t gpu_scope {probe.streams[member]};
            if (!gpu_scope.ok()) return expected_t::fail(status_t::cuda_error_k, "context entry failed");
            std::size_t const batches = source_index == 0 ? config.batches_per_fill : 1; // held-out refills per eval
            for (std::size_t batch_index = 0; batch_index != batches; ++batch_index) {
                expected<multimodal_batch_on_gpu_t> batch = multimodal_batch_on_gpu_t::make(
                    gpu_arena_alloc_t {probe.arenas[member]}, config.batch);
                if (!batch) return expected_t::fail(std::move(batch));
                if (st::failed(filled.push_back(std::move(batch.value_))))
                    return expected_t::fail(status_t::bad_alloc_k, "batch table allocation failed");
            }
        }
        expected<multimodal_batch_mixer_t> mixer = multimodal_batch_mixer_t::make(
            probe.domain, mesh.colocated, probe.arenas, config.batch, config.seed, probe.streams, mesh.timeout_seconds);
        if (!mixer) return expected_t::fail(std::move(mixer));
        if (st::failed(mixers.push_back(std::move(mixer.value_))))
            return expected_t::fail(status_t::bad_alloc_k, "mixer table allocation failed");
    }
    for (std::size_t member = 0; member != devices.size(); ++member) {
        switch_to_gpu_t gpu_scope {probe.streams[member]};
        if (!gpu_scope.ok()) return expected_t::fail(status_t::cuda_error_k, "context entry failed");
        expected<std::size_t> held = held_bytes_(devices[member]);
        if (!held) return expected_t::fail(std::move(held));
        footprints[member].device_bytes = *held - std::min(*held, budgets[member].held_bytes);
        footprints[member].arena_bytes = probe.arenas[member].used(); // exact, where mapped() moves whole chunks
    }
    // A host destroys its queue pairs only once every host drained its last acknowledgement.
    sz::string_t tag;
    if (!format_into(tag, "probe-{}", mesh.topology.next_collective_id()))
        return expected_t::fail(status_t::bad_alloc_k, "barrier tag assembly failed");
    return mesh.topology.barrier(tag.view(), mesh.timeout_seconds);
}

/** Agree on the fewest @p rows any process measured: the lead collects every peer's count and
 *  publishes the minimum, which every peer adopts. */
[[nodiscard]] static expected<u32_t> agree_on_rows_(mesh_t &mesh, u32_t rows) noexcept {
    using result_t = expected<u32_t>;
    mesh_topology_t &topology = mesh.topology;
    if (topology.host_domain_count() <= 1) return result_t {rows};
    sz::string_t text, measured_key, agreed_key;
    u32_t const collective_id = topology.next_collective_id();
    if (!format_into(measured_key, "envelope.{}", collective_id) ||
        !format_into(agreed_key, "envelope.{}.agreed", collective_id))
        return result_t::fail(status_t::bad_alloc_k, "envelope key formatting failed");
    auto parsed = [&](u32_t &value) noexcept {
        return std::from_chars(text.data(), text.data() + text.size(), value).ec == std::errc {};
    };
    if (!topology.is_lead()) {
        u32_t agreed = 0;
        if (!format_into(text, "{}", rows)) return result_t::fail(status_t::bad_alloc_k, "rows formatting failed");
        expected_t exchanged = topology.publish_text(measured_key.view(), text.view());
        if (exchanged) exchanged = topology.await_text(0u, agreed_key.view(), text, mesh.timeout_seconds);
        if (!exchanged) return result_t::fail(std::move(exchanged));
        if (!parsed(agreed))
            return result_t::fail(status_t::invalid_input_k, "the lead agreed on {} rows", text.view());
        return result_t {agreed};
    }
    u32_t agreed = rows;
    for (std::size_t peer = 1; peer != topology.host_domain_count(); ++peer) {
        u32_t peer_rows = 0;
        if (expected_t awaited = topology.await_text(static_cast<u32_t>(peer), measured_key.view(), text,
                                                     mesh.timeout_seconds);
            !awaited)
            return result_t::fail(std::move(awaited));
        if (!parsed(peer_rows))
            return result_t::fail(status_t::invalid_input_k, "host domain {} measured {} rows", peer, text.view());
        agreed = std::min(agreed, peer_rows);
    }
    text.clear();
    if (!format_into(text, "{}", agreed)) return result_t::fail(status_t::bad_alloc_k, "rows formatting failed");
    if (expected_t published = topology.publish_text(agreed_key.view(), text.view()); !published)
        return result_t::fail(std::move(published));
    return result_t {agreed};
}

/**
 *  @brief The rows per GPU at @p config 's columns — the configured ones, or the most every device
 *      of every process holds — and the arena reservation they need.
 *
 *  The arena's high-water mark is exact, so it is fitted as @c fixed+rows·row+rows²·pair through
 *  the three probes, each of which also allocates a data source's loaded batch and mixer. Each
 *  device's footprint takes the arena's curvature and is fitted through the smaller and the larger
 *  probe, and the margin is the footprint outside the arena at the larger probe — the decoders'
 *  workspaces, graphs, the symmetric heap — held free once more. The reservation is the arena's
 *  curve one row past the envelope.
 *  @retval status_t::bad_alloc_k The configured rows do not fit, with the measured numbers.
 */
[[nodiscard]] static expected<envelope_t> measure_envelope_(mesh_t &mesh, trainer_config_t const &config,
                                                            std::span<i32_t const> devices,
                                                            nk_capability_t capabilities,
                                                            std::span<device_budget_t const> budgets) noexcept {
    using result_t = expected<envelope_t>;
    if (config.late_interacts() && config.batch.max_rows_per_gpu == 0)
        return result_t::fail(status_t::invalid_input_k,
                              "late interaction costs rows² times the query span times the columns, so its rows " //
                              "are set, never fitted to memory: pass --rows, or the batch's max_rows_per_device");
    std::size_t const gpus = devices.size();
    footprint_t smaller[max_gpus_per_host_k] = {}, middle[max_gpus_per_host_k] = {}, larger[max_gpus_per_host_k] = {};
    expected_t probed = probe_footprint_(mesh, config, devices, capabilities, smaller_probe_rows_k, budgets,
                                         {smaller, gpus});
    if (probed)
        probed = probe_footprint_(mesh, config, devices, capabilities, middle_probe_rows_k, budgets, {middle, gpus});
    if (probed)
        probed = probe_footprint_(mesh, config, devices, capabilities, larger_probe_rows_k, budgets, {larger, gpus});
    if (!probed) return result_t::fail(std::move(probed));

    f64_t const low_rows = smaller_probe_rows_k, middle_rows = middle_probe_rows_k, high_rows = larger_probe_rows_k;
    auto arena_fit = [&](std::size_t member) noexcept {
        f64_t const low = static_cast<f64_t>(smaller[member].arena_bytes);
        f64_t const lower_slope = (static_cast<f64_t>(middle[member].arena_bytes) - low) / (middle_rows - low_rows);
        f64_t const upper_slope = (static_cast<f64_t>(larger[member].arena_bytes) -
                                   static_cast<f64_t>(middle[member].arena_bytes)) /
                                  (high_rows - middle_rows);
        footprint_fit_t fit;
        fit.pair = std::max((upper_slope - lower_slope) / (high_rows - low_rows), 0.0);
        fit.row = lower_slope - fit.pair * (low_rows + middle_rows);
        fit.fixed = low - fit.row * low_rows - fit.pair * low_rows * low_rows;
        return fit;
    };
    u32_t const configured_rows = config.batch.max_rows_per_gpu;
    u64_t const representable_rows = std::numeric_limits<u32_t>::max() / config.batch.max_columns_per_row;
    envelope_t envelope; // the device holding the fewest rows decides
    for (std::size_t member = 0; member != gpus; ++member) {
        footprint_t const &low = smaller[member], &high = larger[member];
        if (high.device_bytes <= low.device_bytes || high.arena_bytes < low.arena_bytes)
            return result_t::fail(status_t::other_k, "device {} took {} bytes at {} rows and {} at {}", devices[member],
                                  low.device_bytes, smaller_probe_rows_k, high.device_bytes, larger_probe_rows_k);
        footprint_fit_t device = arena_fit(member);
        device.row = (static_cast<f64_t>(high.device_bytes) - static_cast<f64_t>(low.device_bytes) -
                      device.pair * (high_rows * high_rows - low_rows * low_rows)) /
                     (high_rows - low_rows);
        device.fixed = static_cast<f64_t>(low.device_bytes) - device.row * low_rows - device.pair * low_rows * low_rows;
        envelope_t measured;
        measured.free_bytes = budgets[member].free_bytes;
        measured.row_bytes = static_cast<std::size_t>(std::max(device.row, 0.0));
        measured.pair_bytes = static_cast<std::size_t>(device.pair);
        measured.fixed_bytes = static_cast<std::size_t>(std::max(device.fixed, 0.0));
        measured.margin_bytes = high.device_bytes - std::min(high.device_bytes, high.arena_bytes);
        device.fixed += static_cast<f64_t>(config.reservoir_fixed_bytes);
        device.row += static_cast<f64_t>(config.reservoir_row_bytes);
        u64_t const fitting_rows = device.rows_within(static_cast<f64_t>(measured.free_bytes) -
                                                      static_cast<f64_t>(measured.margin_bytes));
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
    for (std::size_t member = 0; member != gpus; ++member)
        envelope.reserve_bytes = std::max(envelope.reserve_bytes,
                                          static_cast<std::size_t>(arena_fit(member).at(f64_t(envelope.rows) + 1.0)));
    return result_t {envelope};
}

#pragma endregion Envelope

#pragma region Trainer

/** Everything @ref trainer_t::make builds after the streams and the envelope, over @p placed 's
 *  sized config, each arena reserving @p reserve_bytes . */
[[nodiscard]] static expected_t assemble_(trainer_t &placed, mesh_t &mesh, std::size_t reserve_bytes) noexcept {
    trainer_config_t const &config = placed.config;
    mesh_topology_t const &topology = mesh.topology;
    std::span<i32_t const> const devices = placed.devices;
    std::size_t const gpus = devices.size();
    for (std::size_t member = 0; member != gpus; ++member) {
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
    if (expected_t probed = loss_probe("base/late_interaction", config.late_interacts(),
                                       placed.probes.late_interaction);
        !probed)
        return probed;
    if (expected_t probed = loss_probe("base/choices", config.chooses(), placed.probes.choices); !probed) return probed;
    for (std::size_t member = 0; member != gpus; ++member) {
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
    for (std::size_t member = 0; member != gpus; ++member) {
        switch_to_gpu_t gpu_scope {placed.streams[member]};
        if (!gpu_scope.ok()) return expected_t::fail(status_t::cuda_error_k, "context entry failed");
        expected<base_replica_t> replica = base_replica_t::make(placed.arenas[member], config, vertices_per_mesh,
                                                                placed.capabilities);
        if (!replica) return expected_t::fail(std::move(replica));
        if (placed.replicas.push_back(std::move(replica.value_)) != status_t::success_k)
            return expected_t::fail(status_t::bad_alloc_k, "replica table full");
    }
    if (expected_t seeded = placed.reseed(); !seeded) return seeded;

    st::basic_vector<differentiable_weights_t>
        registrations; // a transient the optimizer copies into its per-replica state
    for (std::size_t member = 0; member != gpus; ++member)
        for (differentiable_weights_t const &entry : placed.replicas[member].weights) {
            expected<differentiable_weights_t> copied = entry.copy();
            if (!copied || st::failed(registrations.push_back(std::move(copied.value_))))
                return expected_t::fail(status_t::bad_alloc_k, "registration table allocation failed");
        }
    std::span<differentiable_weights_t const> const lead_weights = placed.replicas[0].weights;
    std::size_t const gradient_scalars = distributed_adafactor_t::payload_scalars_needed(lead_weights);
    contrastive_segments_config_t contrastive_config;
    contrastive_config.from_vertices_per_gpu = static_cast<u32_t>(segments);
    contrastive_config.hidden = config.hidden;
    contrastive_config.to_vertices_per_gpu = static_cast<u32_t>(segments);
    contrastive_config.gpu_subgraphs_per_step = topology.shape.gpus_per_colocated_domain;
    contrastive_config.temperature_groups = std::max(config.contrastive.temperature_groups, 1u);
    contrastive_config.pair_loss = config.contrastive.pair_loss;
    std::size_t const force_scalars = config.contrasts() ? distributed_contrastive_segments_t::payload_scalars_needed(
                                                               contrastive_config, topology.shape)
                                                         : 0u;
    std::size_t const mixer_scalars = multimodal_batch_mixer_t::payload_scalars_needed(config.batch, topology.shape);
    late_interaction_config_t const &late_interaction = config.late_interaction;
    std::size_t const max_tokens = std::size_t {config.batch.max_rows_per_gpu} * config.batch.max_columns_per_row;
    contrastive_tokens_config_t tokens_config;
    tokens_config.segments_per_gpu = static_cast<u32_t>(segments);
    tokens_config.head_tokens_per_gpu = static_cast<u32_t>(max_tokens);
    std::size_t const query_tokens = std::min(max_tokens, segments * late_interaction.max_query_span);
    tokens_config.query_tokens_per_gpu = static_cast<u32_t>(round_up_to_multiple(
        late_interaction.max_query_tokens_per_gpu ? std::size_t {late_interaction.max_query_tokens_per_gpu}
                                                  : query_tokens,
        std::size_t {8}));
    tokens_config.query_span_tokens = late_interaction.max_query_span;
    tokens_config.documents_per_gpu = late_interaction.max_documents_per_gpu ? late_interaction.max_documents_per_gpu
                                                                             : static_cast<u32_t>(segments);
    tokens_config.document_tokens_per_gpu = static_cast<u32_t>(
        late_interaction.max_document_tokens_per_gpu ? late_interaction.max_document_tokens_per_gpu : max_tokens);
    tokens_config.band_tokens = static_cast<u32_t>(
        round_up_to_multiple(std::size_t {config.batch.max_columns_per_row}, std::size_t {warp_size_k}));
    tokens_config.width = late_interaction.width;
    tokens_config.gpu_subgraphs_per_step = topology.shape.gpus_per_colocated_domain;
    tokens_config.temperature_groups = contrastive_config.temperature_groups;
    tokens_config.pair_loss = late_interaction.pair_loss;
    tokens_config.weight = late_interaction.weight;
    std::size_t const token_scalars = config.late_interacts()
                                          ? distributed_contrastive_tokens_t::payload_scalars_needed(tokens_config)
                                          : 0u;
    expected<multimem_domain_t> domain = multimem_domain_t::make(
        mesh.topology, placed.devices,
        multimem_domain_t::window_scalars(gradient_scalars) + multimem_domain_t::window_scalars(force_scalars) +
            multimem_domain_t::window_scalars(token_scalars) + mixer_scalars,
        placed.streams, mesh.timeout_seconds);
    if (!domain) return expected_t::fail(std::move(domain));
    placed.domain = std::move(domain.value_);
    {
        switch_to_gpu_t gpu_scope {placed.streams[0]};
        if (!gpu_scope.ok()) return expected_t::fail(status_t::cuda_error_k, "context entry failed");
        expected<infiniband_all_reduce<f32_t>> world_reduce = infiniband_all_reduce<f32_t>::make(
            mesh.colocated, 0, placed.arenas[0], gradient_scalars, placed.streams[0], mesh.timeout_seconds);
        if (!world_reduce) return expected_t::fail(std::move(world_reduce));
        placed.world_reduce = std::move(world_reduce.value_);
    }
    expected<distributed_adafactor_t> optimizer = distributed_adafactor_t::make(
        placed.domain, placed.world_reduce, placed.arenas, registrations, lead_weights.size(),
        adafactor_config_t {.beta1 = config.adafactor_beta1}, placed.streams, placed.ledger);
    if (!optimizer) return expected_t::fail(std::move(optimizer));
    placed.optimizer = std::move(optimizer.value_);

    if (config.contrasts()) {
        per_gpu_array<bf16r2_view_t> towers;
        per_gpu_array<contrastive_relations_view_t> relations;
        per_gpu_array<bf16r2_span_t> forces;
        for (std::size_t member = 0; member != gpus; ++member) {
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
            forces.view(), contrastive_config, prefixes, placed.capabilities, mesh.timeout_seconds);
        if (!contrastive) return expected_t::fail(std::move(contrastive));
        placed.contrastive = std::move(contrastive.value_);
        for (std::size_t member = 0; member != gpus; ++member) {
            switch_to_gpu_t gpu_scope {placed.streams[member]};
            if (!gpu_scope.ok()) return expected_t::fail(status_t::cuda_error_k, "context entry failed");
            expected<adafactor_temperatures_t> temperatures = adafactor_temperatures_t::make(
                placed.arenas[member], contrastive_config.temperature_groups, adafactor_config_t {}, "contrastive",
                placed.streams[member]);
            if (!temperatures) return expected_t::fail(std::move(temperatures));
            if (placed.temperatures.push_back(std::move(temperatures.value_)) != status_t::success_k)
                return expected_t::fail(status_t::bad_alloc_k, "temperature optimizer table full");
        }
    }
    if (config.late_interacts()) {
        expected<distributed_contrastive_tokens_t> tokens = distributed_contrastive_tokens_t::make(
            placed.domain, mesh.colocated, placed.arenas, placed.streams, tokens_config, placed.capabilities,
            mesh.timeout_seconds);
        if (!tokens) return expected_t::fail(std::move(tokens));
        placed.late_interaction = std::move(tokens.value_);
        for (std::size_t member = 0; member != gpus; ++member) {
            switch_to_gpu_t gpu_scope {placed.streams[member]};
            if (!gpu_scope.ok()) return expected_t::fail(status_t::cuda_error_k, "context entry failed");
            expected<adafactor_temperatures_t> temperatures = adafactor_temperatures_t::make(
                placed.arenas[member], tokens_config.temperature_groups, adafactor_config_t {}, "late_interaction",
                placed.streams[member]);
            if (!temperatures) return expected_t::fail(std::move(temperatures));
            if (placed.token_temperatures.push_back(std::move(temperatures.value_)) != status_t::success_k)
                return expected_t::fail(status_t::bad_alloc_k, "token temperature optimizer table full");
        }
    }
    if (config.capture == replayed_k) return placed.capture();
    return {};
}

expected_t trainer_t::make(trainer_t &placed, mesh_t &mesh, trainer_config_t const &requested,
                           std::span<i32_t const> devices, nk_capability_t capabilities) noexcept {
    device_budget_t budgets[max_gpus_per_host_k] = {};
    if (expected_t prepared = prepare_(placed, mesh, requested, devices, capabilities, {budgets, devices.size()});
        !prepared)
        return prepared;
    expected<envelope_t> measured = measure_envelope_(mesh, requested, devices, capabilities,
                                                      {budgets, devices.size()});
    if (!measured) return expected_t::fail(std::move(measured));
    placed.envelope = *measured;
    if (!requested.batch.max_rows_per_gpu || !requested.batch.max_segments_per_gpu)
        placed.config.batch = batch_capacity(placed.config.batch, placed.envelope.rows, segments_per_row_(requested));

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
    st::basic_vector<f32_t> staging;
    if (st::failed(staging.resize(st::assume_uninitialized, values.extent(0))))
        return expected_t::fail(status_t::bad_alloc_k, "{} staging failed", name);
    if (expected_t copied = copy_device_to_host(values, std::span<f32_t> {staging}, stream); !copied) return copied;
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
        if (restored) restored = temperatures[member].load(*loaded, streams[member]);
        if (restored && config.late_interacts())
            restored = restore_vector_(*loaded, "late_interaction.inverse_temperatures",
                                       replica.token_inverse_temperatures.span(), streams[member]);
        if (restored && config.late_interacts())
            restored = restore_vector_(*loaded, "late_interaction.biases", replica.token_biases.span(),
                                       streams[member]);
        if (restored && config.late_interacts()) restored = token_temperatures[member].load(*loaded, streams[member]);
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
            streams[member].get(), [&]() noexcept { return replicas[member].forward_region(config, streams[member]); });
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

expected_t trainer_t::step(batch_source_t source, optimizer_step_t optimizer_step, step_report_t &report) noexcept {
    data_epoch_t const data_epoch {optimizer_step.ordinal};
    report.optimizer_step = optimizer_step;
    report.learning_rate = config.learning_rate_for(optimizer_step);
    report.truncated = 0;
    if (expected_t filled = source.fill(source.state, *this, data_epoch, report); !filled) return filled;
    std::size_t const gpus = replicas.size();
    for (std::size_t member = 0; member != gpus; ++member) {
        switch_to_gpu_t gpu_scope {streams[member]};
        if (!gpu_scope.ok()) return expected_t::fail(status_t::cuda_error_k, "context entry failed");
        expected_t stepped = replicas[member].prepare(config, data_epoch, streams[member]);
        if (stepped)
            stepped = forward_graphs.size() ? forward_graphs[member].launch(streams[member].get())
                                            : replicas[member].forward_region(config, streams[member]);
        if (!stepped) return stepped;
    }
    if (config.contrasts())
        for (std::size_t member = 0; member != gpus; ++member)
            if (expected_t contrasted = contrastive(member, replicas[member].recipe(config), streams[member]);
                !contrasted)
                return contrasted;
    if (config.late_interacts()) {
        per_gpu_array<gpu_tokens_inputs_t> inputs;
        for (std::size_t member = 0; member != gpus; ++member) {
            base_replica_t &replica = replicas[member];
            gpu_tokens_inputs_t const member_inputs {replica.tower.token_vectors(), replica.tower.token_offsets(),
                                                     replica.batch.view(),          replica.relations.view(),
                                                     replica.token_recipe(config),  replica.token_vectors_grad.span()};
            if (inputs.push_back(member_inputs) != status_t::success_k)
                return expected_t::fail(status_t::bad_alloc_k, "late-interaction input table full");
        }
        if (expected_t contrasted = late_interaction(inputs.view(), streams.span()); !contrasted) return contrasted;
    }
    for (std::size_t member = 0; member != gpus; ++member) {
        switch_to_gpu_t gpu_scope {streams[member]};
        if (!gpu_scope.ok()) return expected_t::fail(status_t::cuda_error_k, "context entry failed");
        expected_t stepped = backward_graphs.size() ? backward_graphs[member].launch(streams[member].get())
                                                    : replicas[member].backward_region(streams[member]);
        if (stepped) stepped = optimizer.step(member, report.learning_rate, streams[member]);
        if (stepped && config.contrasts())
            stepped = temperatures[member].step(replicas[member].inverse_temperatures.span(),
                                                replicas[member].biases.span(), contrastive.temperature_grads(member),
                                                report.learning_rate, streams[member]);
        if (stepped && config.late_interacts())
            stepped = token_temperatures[member].step(
                replicas[member].token_inverse_temperatures.span(), replicas[member].token_biases.span(),
                late_interaction.temperature_grads(member), report.learning_rate, streams[member]);
        if (!stepped) return stepped;
    }
    completed_steps = optimizer_step.ordinal + 1u;
    return {};
}

expected<retrieval_t> trainer_t::evaluate(batch_source_t source, data_epoch_t data_epoch) noexcept {
    using result_t = expected<retrieval_t>;
    if (!config.contrasts()) return result_t::fail(status_t::invalid_input_k, "only a contrastive run pools vertices");
    step_report_t report;
    if (expected_t filled = source.fill(source.state, *this, data_epoch, report); !filled)
        return result_t::fail(std::move(filled));
    std::size_t const gpus = replicas.size(), columns = std::size_t(retrieval_column_t::count_k);
    for (std::size_t member = 0; member != gpus; ++member) {
        switch_to_gpu_t gpu_scope {streams[member]};
        if (!gpu_scope.ok()) return result_t::fail(status_t::cuda_error_k, "context entry failed");
        base_replica_t &replica = replicas[member];
        expected_t encoded = replica.tower.pack_weights(streams[member]);
        if (encoded) encoded = replica.tower.forward(replica.batch, replica.vertices.span(), streams[member]);
        if (!encoded) return result_t::fail(std::move(encoded));
    }
    for (std::size_t member = 0; member != gpus; ++member)
        if (expected_t scored = contrastive.evaluate(member, replicas[member].recipe(config), streams[member]); !scored)
            return result_t::fail(std::move(scored));
    std::size_t const rows = config.batch.max_segments_per_gpu;
    nk::tensor<f32_t, pinned_alloc<f32_t>, 3> tables =
        nk::tensor<f32_t, pinned_alloc<f32_t>, 3>::uninitialized({gpus, rows, columns}, pinned_alloc<f32_t> {}).value;
    if (!tables) return result_t::fail(status_t::bad_alloc_k, "retrieval staging allocation failed");
    for (std::size_t member = 0; member != gpus; ++member) {
        switch_to_gpu_t gpu_scope {streams[member]};
        if (!gpu_scope.ok()) return result_t::fail(status_t::cuda_error_k, "context entry failed");
        if (expected_t copied = copy_device_to_host(contrastive.retrieval(member).flatten<1>(),
                                                    tables[member, nk::slice].flatten<1>(), streams[member]);
            !copied)
            return result_t::fail(std::move(copied));
    }
    if (expected_t drained = synchronize(); !drained) return result_t::fail(std::move(drained));
    f64_t sums[std::size_t(retrieval_column_t::count_k)] = {};
    for (std::size_t member = 0; member != gpus; ++member)
        for (std::size_t row = 0; row != rows; ++row)
            for (std::size_t column = 0; column != columns; ++column) sums[column] += tables[member, row, column];
    auto const sum = [&](retrieval_column_t which) noexcept { return sums[std::size_t(which)]; };
    auto const ratio = [](f64_t numerator, f64_t denominator) noexcept {
        return denominator != 0.0 ? numerator / denominator : 0.0;
    };
    retrieval_t retrieval;
    retrieval.completed_steps = completed_steps;
    retrieval.ranked = sum(retrieval_column_t::ranked_k);
    retrieval.recall_at_1 = ratio(sum(retrieval_column_t::hit_at_1_k), retrieval.ranked);
    retrieval.recall_at_5 = ratio(sum(retrieval_column_t::hit_at_5_k), retrieval.ranked);
    retrieval.chance = ratio(sum(retrieval_column_t::chance_k), retrieval.ranked);
    retrieval.cosine_gap = ratio(sum(retrieval_column_t::attract_cosine_k), sum(retrieval_column_t::attracts_k)) -
                           ratio(sum(retrieval_column_t::ambient_cosine_k), sum(retrieval_column_t::ambients_k));
    retrieval.hards = sum(retrieval_column_t::hards_k);
    retrieval.hard_above = ratio(sum(retrieval_column_t::hard_above_k), retrieval.hards);
    // The participation ratio of the centred unit vertices' Gram matrix: its squared trace over its
    // squared Frobenius norm.
    f64_t const trace = sum(retrieval_column_t::centred_norm_k);
    retrieval.effective_rank = ratio(trace * trace, sum(retrieval_column_t::centred_products_k));
    return result_t {retrieval};
}

expected_t trainer_t::reseed() noexcept {
    std::size_t const groups = std::max<std::size_t>(config.contrastive.temperature_groups, 1);
    st::basic_vector<f32_t> inverse_temperatures, biases, token_inverse_temperatures, token_biases;
    if (config.contrasts()) {
        if (st::failed(inverse_temperatures.resize(groups)) || st::failed(biases.resize(groups)) ||
            st::failed(token_inverse_temperatures.resize(groups)) || st::failed(token_biases.resize(groups)))
            return expected_t::fail(status_t::bad_alloc_k, "temperature staging failed");
        for (std::size_t group = 0; group != groups; ++group)
            inverse_temperatures[group] = config.contrastive.inverse_temperature,
            biases[group] = config.contrastive.bias,
            token_inverse_temperatures[group] = config.late_interaction.inverse_temperature,
            token_biases[group] = config.late_interaction.bias;
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
        if (staged && config.late_interacts())
            staged = copy_host_to_device(f32r1_view_t {token_inverse_temperatures.data(), groups},
                                         replica.token_inverse_temperatures.span(), streams[member]);
        if (staged && config.late_interacts())
            staged = copy_host_to_device(f32r1_view_t {token_biases.data(), groups}, replica.token_biases.span(),
                                         streams[member]);
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
    summand_pair_t token {}, image {}, video {}, audio {}, choices {};
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
        u32_t dropped = 0; // pageable, so the copy has landed once the call returns
        if (copy_device_to_host(replica.batch.dropped.view(), std::span<u32_t> {&dropped, 1}, streams[member]) &&
            zero_async(replica.batch.dropped.span(), streams[member]))
            telemetry.dropped += dropped;
        accumulate(token, replica.tokens_summands, streams[member]);
        accumulate(image, replica.image_summands, streams[member]);
        accumulate(video, replica.video_summands, streams[member]);
        accumulate(audio, replica.audio_summands, streams[member]);
        accumulate(choices, replica.choices_summands, streams[member]);
        (void)ledger.probe_round(probes.rounds[member], domain.member(domain.member_offset() + member),
                                 streams[member]);
    }
    telemetry.token_ce = token.count != 0.0 ? static_cast<f32_t>(token.sum / token.count) : 0.0f;
    telemetry.image_mse = image.count != 0.0 ? static_cast<f32_t>(image.sum / image.count) : 0.0f;
    telemetry.video_mse = video.count != 0.0 ? static_cast<f32_t>(video.sum / video.count) : 0.0f;
    telemetry.audio_mse = audio.count != 0.0 ? static_cast<f32_t>(audio.sum / audio.count) : 0.0f;
    telemetry.choices = choices.count != 0.0 ? static_cast<f32_t>(choices.sum / choices.count) : 0.0f;
    ledger.scalar(probes.token, token.sum, token.count);
    ledger.scalar(probes.image, image.sum, image.count);
    ledger.scalar(probes.video, video.sum, video.count);
    ledger.scalar(probes.audio, audio.sum, audio.count);
    ledger.scalar(probes.choices, choices.sum, choices.count);

    switch_to_gpu_t gpu_scope {streams[0]};
    if (!gpu_scope.ok()) return telemetry;
    if (config.contrasts()) {
        f32r1_tensor_pinned_t staging = f32r1_tensor_pinned_t::uninitialized({1}, pinned_alloc<f32_t> {}).value;
        if (staging && copy_device_to_host(contrastive.world_loss(0), staging.span(), streams[0])) {
            cuStreamSynchronize(streams[0].get());
            telemetry.contrastive = staging[0];
            ledger.scalar(probes.contrastive, staging[0], 1.0);
        }
    }
    if (config.late_interacts()) {
        f32r1_tensor_pinned_t staging = f32r1_tensor_pinned_t::uninitialized({1}, pinned_alloc<f32_t> {}).value;
        if (staging && copy_device_to_host(late_interaction.world_loss(0), staging.span(), streams[0])) {
            cuStreamSynchronize(streams[0].get());
            telemetry.late_interaction = staging[0];
            ledger.scalar(probes.late_interaction, staging[0], 1.0);
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
    if (saved && config.contrasts()) saved = temperatures[0].save(checkpoint, streams[0]);
    if (saved && config.late_interacts())
        saved = insert_vector_(checkpoint, "late_interaction.inverse_temperatures",
                               replicas[0].token_inverse_temperatures.view(), streams[0]);
    if (saved && config.late_interacts())
        saved = insert_vector_(checkpoint, "late_interaction.biases", replicas[0].token_biases.view(), streams[0]);
    if (saved && config.late_interacts()) saved = token_temperatures[0].save(checkpoint, streams[0]);
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
    term("late interaction", late_interaction);
    term("choices", choices);
    line.clear();
    if (dropped != 0 && format_into(line, "            dropped {} segments", dropped)) report_line(line.view());
}

void retrieval_t::report() const noexcept {
    sz::string_t line;
    if (format_into(line,
                    "  eval after {:>4}  recall@1 {:.4f}  recall@5 {:.4f}  chance {:.4f}  over {:.0f} rows  " //
                    "cosine gap {:.4f}  hard above {:.4f} of {:.0f}  effective rank {:.2f}",
                    completed_steps, recall_at_1, recall_at_5, chance, ranked, cosine_gap, hard_above, hards,
                    effective_rank))
        report_line(line.view());
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

/** How many choices item @p item offers, zero when the step names none. */
[[nodiscard]] static std::size_t item_choice_count_(training_items_t const &items, std::size_t item) noexcept {
    if (!items.choices.offsets) return 0;
    return items.choices.offsets[item + 1] - items.choices.offsets[item];
}

/** Write item @p item 's choices and targets into row @p segment of @p staged , padding the rest;
 *  a byte offered twice is refused, since one logit holds one target. */
[[nodiscard]] static expected_t stage_choices_(training_items_t const &items, std::size_t item,
                                               multimodal_batch_on_cpu_t &staged, std::size_t segment) noexcept {
    std::size_t const width = staged.segment_choices.extent(1), offered = item_choice_count_(items, item);
    std::size_t const first = offered != 0 ? items.choices.offsets[item] : 0u;
    for (std::size_t choice = 0; choice != width; ++choice) {
        bool const live = choice < offered;
        token_id_t const byte = live ? token_id_t {items.choices.bytes[first + choice]} : byte_pad_k;
        for (std::size_t earlier = 0; live && earlier != choice; ++earlier)
            if (staged.segment_choices[segment, earlier] == byte)
                return expected_t::fail(status_t::invalid_input_k, "item {} offers byte {} twice", item, byte);
        staged.segment_choices[segment, choice] = byte;
        staged.segment_choice_targets[segment, choice] = live ? items.choices.targets[first + choice] : 0.0f;
    }
    return {};
}

/** The video slice of @p shape with its frame stride read as at least one. */
[[nodiscard]] static video_batch_shape_t video_shape_(multimodal_batch_shape_t const &shape) noexcept {
    video_batch_shape_t video = shape.video;
    video.frame_stride = std::max(video.frame_stride, 1u);
    return video;
}

expected_t items_source_t::make(items_source_t &placed, trainer_t &trainer) noexcept {
    switch_to_gpu_t lead_scope {trainer.streams[0]}; // pinned host pages allocate inside a context
    if (!lead_scope.ok()) return expected_t::fail(status_t::cuda_error_k, "context entry failed");
    trainer_config_t const &config = trainer.config;
    multimodal_batch_shape_t const &shape = config.batch;
    std::size_t const gpus = trainer.replicas.size(), segments = shape.max_segments_per_gpu;
    if (config.contrasts() && trainer.mesh->topology.host_domain_count() > 1)
        return expected_t::fail(status_t::shape_unsupported_k,
                                "every process passes its own items, and the edges of an item the mixer routes to " //
                                "another host stay with the process that passed it");
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
        trainer.domain, trainer.mesh->colocated, trainer.arenas, shape, config.seed, trainer.streams,
        trainer.mesh->timeout_seconds);
    if (!mixer) return expected_t::fail(std::move(mixer));
    placed.mixer_ = std::move(mixer.value_);

    std::size_t const total_segments = gpus * segments;
    std::size_t const edges = config.contrastive.max_edges_per_gpu
                                  ? config.contrastive.max_edges_per_gpu
                                  : relation_edges_capacity(static_cast<u32_t>(segments), shape.max_bag_segments);
    bool allocated = !st::failed(placed.tokens_.resize(total_segments)) &&
                     !st::failed(placed.destinations_.resize(total_segments)) &&
                     !st::failed(placed.order_.resize(total_segments));
    placed.packed_items_ = u32r2_tensor_pinned_t::uninitialized({gpus, segments}, pinned_alloc<u32_t> {}).value;
    allocated = allocated && placed.packed_items_;
    if (shape.contains(modality_t::audio_k))
        allocated = allocated && !st::failed(placed.window_.resize(
                                     std::size_t {media_cells(modality_t::audio_k, shape.max_columns_per_row)} *
                                     shape.audio.chunk_samples));
    if (config.contrasts()) {
        expected<contrastive_edges_pinned_t> edge_sets = contrastive_edges_pinned_t::make(static_cast<u32_t>(segments),
                                                                                          static_cast<u32_t>(edges));
        if (!edge_sets) return expected_t::fail(std::move(edge_sets));
        placed.edges_ = std::move(edge_sets.value_);
        placed.sources_ = u32r2_tensor_pinned_t::uninitialized({gpus, segments}, pinned_alloc<u32_t> {}).value;
        placed.pooled_ = nk::tensor<bf16_t, pinned_alloc<bf16_t>, 3>::uninitialized({gpus, segments, config.hidden},
                                                                                    pinned_alloc<bf16_t> {})
                             .value;
        allocated = allocated && placed.sources_ && placed.pooled_ &&
                    !st::failed(placed.lengths_.resize(total_segments)) && !st::failed(placed.groups_.resize(segments));
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
    std::size_t const choices_width = staged_[0].segment_choices.extent(1);

    if (expected_t drained = trainer.synchronize(); !drained) return drained; // prior uploads read the pinned twins

    for (std::size_t item = 0; item != count; ++item) {
        modality_t const modality = items.modalities[item];
        std::span<b8_t const> const bytes = item_bytes_(items, item);
        if (!shape.contains(modality))
            return expected_t::fail(status_t::shape_unsupported_k, "item {} has a modality the run left out", item);
        std::size_t const offered = item_choice_count_(items, item);
        if (offered != 0 && modality != modality_t::text_k)
            return expected_t::fail(status_t::shape_unsupported_k, "item {} makes a choice, which only text makes",
                                    item);
        if (offered > choices_width)
            return expected_t::fail(status_t::invalid_input_k,
                                    "item {} offers {} choices, more than the {} a segment holds", item, offered,
                                    choices_width);
        std::size_t kept = 0, spanned = 0;
        // A choice appends the mask byte, and a long state loses its head, never its end.
        if (modality == modality_t::text_k)
            spanned = bytes.size() + (offered != 0), kept = std::min<std::size_t>(spanned, columns);
        else if (modality == modality_t::image_k) {
            expected<um::image_plan_t> planned = um::plan_image(bytes, shape.image.patch_side, columns);
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
            f32r1_span_t const window = std::span<f32_t> {window_};
            if (expected_t decoded = um::decode_audio(bytes, shape.audio.sample_rate, window); !decoded) return decoded;
            std::size_t heard = window_.size(); // the decoder pads a short clip with exact silence
            while (heard && window_[heard - 1] == 0.0f) --heard;
            std::size_t const frames = divide_round_up(heard, shape.audio.chunk_samples);
            kept = frames ? frames : 1u;
            spanned = frames == media_cells(modality_t::audio_k, columns) ? frames + 1u : kept;
        }
        tokens_[item] = static_cast<u32_t>(kept);
        report.truncated += spanned > kept;
        order_[item] = static_cast<u32_t>(item);
    }
    if (items.bags)
        std::sort(order_.begin(), order_.begin() + count, [&](u32_t left, u32_t right) noexcept {
            return items.bags[left] != items.bags[right] ? items.bags[left] < items.bags[right] : left < right;
        });

    u32_t cursors[max_gpus_per_host_k] = {}, used_segments[max_gpus_per_host_k] = {},
          used_bags[max_gpus_per_host_k] = {};
    for (std::size_t index = 0; index != packed_items_.numel(); ++index) packed_items_[index] = absent_k;
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
                cursor = placed_cursor_(cursor, kept, items.modalities[order_[position]], columns) + kept;
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
            cursors[device] = placed_cursor_(cursors[device], kept, items.modalities[item], columns);
            u32_t const segment = used_segments[device]++, offset = cursors[device];
            cursors[device] += kept;
            modality_t const modality = items.modalities[item];
            std::span<b8_t const> const bytes = item_bytes_(items, item);
            staged.segment_offsets[segment] = offset;
            staged.segment_lengths[segment] = kept;
            staged.segment_modality[segment] = modality;
            staged.segment_bag[segment] = used_bags[device];
            staged.segment_role[segment] = segment_role_t::anchor_k;
            staged.segment_prefix_length[segment] = 0u;
            staged.segment_key[segment] = 0u;
            staged.segment_side[segment] = items.sides ? items.sides[item] : late_interaction_side_t::none_k;
            std::size_t const offered = item_choice_count_(items, item);
            if (expected_t staged_choices = stage_choices_(items, item, staged, segment); !staged_choices)
                return staged_choices;
            packed_items_[device, segment] = item;
            std::size_t const cells = staged.tokens.numel();
            auto const run = nk::range(offset, offset + kept);
            if (modality == modality_t::text_k) {
                tokens_span_t const cell_tokens = staged.tokens.flatten<1>();
                tokens_span_t const tokens = cell_tokens[run, nk::slice];
                std::size_t const state = kept - (offered != 0), first = offered != 0 ? bytes.size() - state : 0u;
                for (std::size_t token = 0; token != state; ++token)
                    tokens[token] = static_cast<token_id_t>(bytes[first + token]);
                if (offered != 0) tokens[state] = byte_mask_k;
            }
            else if (modality == modality_t::image_k) {
                expected<um::image_plan_t> planned = um::plan_image(bytes, shape.image.patch_side, columns);
                if (!planned) return expected_t::fail(std::move(planned));
                std::size_t const side = shape.image.patch_side;
                staged.image_shapes[segment, 0] = planned->grid.rows;
                staged.image_shapes[segment, 1] = planned->grid.columns;
                nk::tensor_span<pixel_t, 2> const image_cells = as_rows(staged.images, cells);
                um::pixel_patches_span_t const patches = image_cells[run, nk::slice].reshape<5>(
                    {planned->grid.rows, planned->grid.columns, side, side, std::size_t {um::image_channels_k}});
                if (expected_t decoded = um::decode_image(bytes, *planned, patches); !decoded) return decoded;
            }
            else if (modality == modality_t::video_k) {
                expected<um::demuxed_t> clip = um::demux_video(bytes);
                if (!clip) return expected_t::fail(std::move(clip));
                um::video_grid_t const grid = um::video_tube_grid(clip->frames, clip->extent, video, columns);
                std::size_t const side = video.patch_side;
                staged.video_shapes[segment, 0] = grid.temporal_groups;
                staged.video_shapes[segment, 1] = grid.patch_rows;
                staged.video_shapes[segment, 2] = grid.patch_columns;
                nk::tensor_span<pixel_t, 2> const video_cells = as_rows(staged.videos, cells);
                um::pixel_tubes_span_t const tubes = video_cells[run, nk::slice].reshape<6>(
                    {grid.temporal_groups, std::size_t {grid.patch_rows} * grid.patch_columns, video.tube_frames, side,
                     side, std::size_t {um::video_channels_k}});
                if (expected_t decoded = um::decode_video_frames(*clip, grid, video, tubes); !decoded) return decoded;
            }
            else {
                f32r2_span_t const audio_cells = as_rows(staged.audio_samples, cells);
                f32r1_span_t const frames = audio_cells[run, nk::slice].flatten<1>();
                if (expected_t decoded = um::decode_audio(bytes, shape.audio.sample_rate, frames); !decoded)
                    return decoded;
            }
        }
        ++used_bags[device];
    }

    for (std::size_t device = 0; device != gpus; ++device) {
        multimodal_batch_on_cpu_t &staged = staged_[device];
        multimodal_batch_on_gpu_t &loaded = loaded_[device];
        for (std::size_t segment = used_segments[device]; segment != segments; ++segment) {
            staged.segment_offsets[segment] = cursors[device];
            staged.segment_lengths[segment] = 0u;
            staged.segment_modality[segment] = modality_t::text_k;
            staged.segment_bag[segment] = bags_per_gpu;
            staged.segment_role[segment] = segment_role_t::anchor_k;
            staged.segment_prefix_length[segment] = 0u;
            staged.segment_key[segment] = 0u;
            staged.segment_side[segment] = late_interaction_side_t::none_k;
            for (std::size_t choice = 0; choice != choices_width; ++choice) {
                staged.segment_choices[segment, choice] = byte_pad_k;
                staged.segment_choice_targets[segment, choice] = 0.0f;
            }
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
        if (copied) copied = copy_host_to_device(staged.segment_role.view(), loaded.segment_role.span(), stream);
        if (copied)
            copied = copy_host_to_device(staged.segment_prefix_length.view(), loaded.segment_prefix_length.span(),
                                         stream);
        if (copied) copied = copy_host_to_device(staged.segment_key.view(), loaded.segment_key.span(), stream);
        if (copied) copied = copy_host_to_device(staged.segment_side.view(), loaded.segment_side.span(), stream);
        if (copied && choices_width != 0)
            copied = copy_host_to_device(staged.segment_choices.view().flatten<1>(),
                                         loaded.segment_choices.span().flatten<1>(), stream);
        if (copied && choices_width != 0)
            copied = copy_host_to_device(staged.segment_choice_targets.view().flatten<1>(),
                                         loaded.segment_choice_targets.span().flatten<1>(), stream);
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
        if (copied) copied = zero_async(loaded.dropped.span(), stream);
        if (!copied) return copied;
    }
    return {};
}

/** Sort one row's edges ascending by id, carrying the weights along, and drop repeated ids — the order the
 *  relation columns' binary search reads. Rows are short, so an insertion sort without scratch. */
[[nodiscard]] static u32_t sort_row_(u32r1_span_t ids, f32r1_span_t weights) noexcept {
    std::size_t const count = ids.extent(0);
    for (std::size_t index = 1; index < count; ++index)
        for (std::size_t back = index; back != 0 && ids[back - 1] > ids[back]; --back) {
            std::swap(ids[back - 1], ids[back]);
            if (!weights.empty()) std::swap(weights[back - 1], weights[back]);
        }
    u32_t kept = 0;
    for (std::size_t index = 0; index != count; ++index) {
        if (kept && ids[kept - 1] == ids[index]) continue;
        ids[kept] = ids[index];
        if (!weights.empty()) weights[kept] = weights[index];
        ++kept;
    }
    return kept;
}

expected_t items_source_t::relate_(trainer_t &trainer) noexcept {
    trainer_config_t const &config = trainer.config;
    std::size_t const gpus = trainer.replicas.size(), count = items.modalities.size();
    u32_t const segments = config.batch.max_segments_per_gpu;
    std::size_t const total_segments = gpus * segments, edges = edges_.capacity();
    u32_t const groups = std::max(config.contrastive.temperature_groups, 1u);

    for (std::size_t member = 0; member != gpus; ++member) {
        switch_to_gpu_t gpu_scope {trainer.streams[member]};
        if (!gpu_scope.ok()) return expected_t::fail(status_t::cuda_error_k, "context entry failed");
        u32r2_view_t const world_sources = mixer_.world_sources(member);
        if (expected_t copied = copy_device_to_host(world_sources[member, nk::slice], sources_[member, nk::slice],
                                                    trainer.streams[member]);
            !copied)
            return copied;
    }
    if (expected_t drained = trainer.synchronize(); !drained) return drained;
    for (std::size_t item = 0; item != count; ++item) destinations_[item] = absent_k;
    u32r1_span_t const vertex_items = sources_.flatten<1>();
    for (std::size_t vertex = 0; vertex != total_segments; ++vertex) {
        u32_t const source = vertex_items[vertex];
        u32_t const item = source < total_segments ? packed_items_[source] : absent_k;
        vertex_items[vertex] = item;
        lengths_[vertex] = item != absent_k ? tokens_[item] : 0u;
        if (item != absent_k) destinations_[item] = static_cast<u32_t>(vertex);
    }

    item_edges_t const *const sets[3] = {&items.attract, &items.repel, &items.ignore};
    for (std::size_t member = 0; member != gpus; ++member) {
        for (std::size_t set_index = 0; set_index != 3; ++set_index) {
            contrastive_edge_set_t const set = static_cast<contrastive_edge_set_t>(set_index);
            item_edges_t const &edges_of = *sets[set_index];
            u32r1_span_t const offsets = edges_.offsets(set), ids = edges_.ids(set);
            f32r1_span_t const weights = edges_.weights(set);
            u32_t written = 0;
            offsets[0] = 0;
            for (std::size_t row = 0; row != segments; ++row) {
                u32_t const item = sources_[member, row];
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
                        if (!weights.empty()) weights[written] = edges_of.weights ? edges_of.weights[edge] : 1.0f;
                        ++written;
                    }
                auto const row_edges = nk::range(row_begin, written);
                if (written != row_begin)
                    written = row_begin + sort_row_(ids[row_edges, nk::slice],
                                                    weights.empty() ? f32r1_span_t {} : weights[row_edges, nk::slice]);
                offsets[row + 1] = written;
            }
        }
        for (std::size_t row = 0; row != segments; ++row) {
            u32_t const item = sources_[member, row];
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
        bool const ignores = items.ignore.offsets != nullptr;
        expected_t staged = copy_host_to_device(std::span<u32_t const> {lengths_}, replica.to_vertex_lengths.span(),
                                                stream);
        if (staged)
            staged = copy_host_to_device(std::span<temperature_group_t const> {groups_},
                                         replica.temperature_groups.span(), stream);
        if (staged)
            staged = replica.relations.upload(edges_.attract(), edges_.repel(),
                                              ignores ? edges_.ignore() : contrastive_edges_view_t {},
                                              replica.batch.segment_lengths.view(), replica.to_vertex_lengths.view(),
                                              u64r1_view_t {}, u64r1_view_t {}, total_segments, member * segments,
                                              config.contrastive.ambient_repulsion_weight, stream);
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
        if (expected_t copied = copy_device_to_host(trainer.replicas[member].vertices.view().flatten<1>(),
                                                    pooled_[member, nk::slice].flatten<1>(), trainer.streams[member]);
            !copied)
            return result_t::fail(std::move(copied));
    }
    if (expected_t drained = trainer.synchronize(); !drained) return result_t::fail(std::move(drained));
    bf16r2_view_t const pooled = pooled_.reshape<2>({gpus * segments, hidden});
    return result_t {pooled};
}

#pragma endregion Sources
