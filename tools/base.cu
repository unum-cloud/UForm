/**
 *  @file tools/base.cu
 *  @author Ash Vardanian
 *  @date July 6, 2026
 *  @brief The Base training binary — a thin @c main over the training library in
 *      @c src/train/trainer.hpp .
 *
 *  It parses the command line into a @ref trainer_config_t , maps the launcher's environment onto
 *  the mesh options, bootstraps the mesh, makes the trainer and its batch source, and steps until
 *  @c config.steps with telemetry and checkpoints on cadence.
 *
 *  The batch source is the mixture: @c --mixture @c tokens (the default) trains the byte token head
 *  alone on the synthetic draw — span-masked byte pretraining, the "Uniform Byte Models" root the
 *  Reranker and Embedder warm-start from — and @c --mixture @c omni adds the raw-media
 *  masked-autoencoder heads (image / video patches, 1-D audio frames) over one heterogeneous batch.
 *  A @c --mixture path to a mixture JSON in @c scripts/data trains on its datasets instead,
 *  lighting the media slices its entries name. @c --eager drops the model-region capture.
 */
#include <cstdlib> // `::setenv`, `std::getenv` — the launcher-contract boundary
#include <cstring> // `std::strlen`
#include <cstdio>  // `std::setvbuf`, `stdout`

#include <cuda.h>       // `cuInit`, `cuDeviceGetCount`
#include <fmt/format.h> // `fmt::println`

#include "train/datasets.hpp" // `mixture_t`, `dataset_source_t`, and the trainer

/** Prints a StringZilla view as the text it spans. */
template <>
struct fmt::formatter<ashvardanian::stringzilla::string_view_t> : fmt::formatter<std::string_view> {
    auto format(std::string_view text, fmt::format_context &context) const {
        return fmt::formatter<std::string_view>::format(text, context);
    }
};

/** Prints an owning StringZilla string as the text it holds. */
template <>
struct fmt::formatter<ashvardanian::stringzilla::string_t> : fmt::formatter<std::string_view> {
    auto format(std::string_view text, fmt::format_context &context) const {
        return fmt::formatter<std::string_view>::format(text, context);
    }
};

namespace ul = unum::uform::loaders;
using namespace unum::uform::train;

/** Tag @p message with the binary name, write it to standard error, and hand back main's failure
 *  code. */
[[nodiscard]] static int fail_out_(sz::string_view_t message) noexcept {
    fmt::println(stderr, "uform_base: {}", message);
    return 1;
}

/** Step @p trainer from its completed steps to @c config.steps over @p source , with telemetry and
 *  checkpoints on cadence and a checkpoint at the end. */
[[nodiscard]] static int step_(trainer_t &trainer, trainer_config_t const &config, batch_source_t &source) noexcept {
    auto checkpoint = [&]() noexcept {
        expected_t saved = trainer.save(config.checkpoint);
        if (saved)
            fmt::println("uform_base: checkpoint written to {} at step {}", config.checkpoint, trainer.completed_steps);
        return saved;
    };
    step_report_t report;
    for (u32_t ordinal = trainer.completed_steps; ordinal < config.steps; ++ordinal) {
        if (expected_t stepped = trainer.step(source, optimizer_step_t {ordinal}, report); !stepped) {
            fmt::println(stderr, "uform_base: step {} failed: {}", ordinal, stepped.description_);
            return 1;
        }
        u32_t const completed = ordinal + 1u;
        if (config.report_every && completed % config.report_every == 0u)
            trainer.telemetry(optimizer_step_t {ordinal}).report();
        if (!config.checkpoint.empty() && config.checkpoint_every && completed % config.checkpoint_every == 0u &&
            completed != config.steps)
            if (expected_t saved = checkpoint(); !saved) return fail_out_(saved.description_.view());
    }
    if (!config.checkpoint.empty())
        if (expected_t saved = checkpoint(); !saved) return fail_out_(saved.description_.view());
    if (expected_t drained = trainer.synchronize(); !drained) return fail_out_(drained.description_.view());
    {
        switch_to_gpu_t gpu_scope {trainer.streams[0]};
        if (expected_t reported = trainer.ledger.report(trainer.streams[0]); !reported)
            fmt::println(stderr, "uform_base: {}", reported.description_);
    }
    fmt::println("uform_base: done");
    return 0;
}

/** Bootstrap the mesh over every visible device, make the trainer and the source @p mixture names,
 *  and step. */
[[nodiscard]] static int train_(trainer_config_t config, mesh_options_t const &options, mixture_t mixture) noexcept {
    if (cuInit(0) != CUDA_SUCCESS) return fail_out_("cuInit failed");
    int device_count = 0;
    if (cuDeviceGetCount(&device_count) != CUDA_SUCCESS || device_count <= 0) return fail_out_("no CUDA devices");
    array<i32_t> devices;
    for (int device = 0; device != device_count; ++device)
        if (devices.push_back(static_cast<i32_t>(device)) != status_t::success_k)
            return fail_out_("device table allocation failed");

    mesh_t mesh;
    if (expected_t joined = mesh_t::make(mesh, options, devices.view()); !joined)
        return fail_out_(joined.description_.view());
    fmt::println("uform_base: tokens{}{}{} mixture, {} GPU(s), host_domain {} of {}",
                 config.batch.contains(modality_t::image_k) ? "+image" : "",
                 config.batch.contains(modality_t::video_k) ? "+video" : "",
                 config.batch.contains(modality_t::audio_k) ? "+audio" : "", device_count, options.host_domain_index,
                 options.host_domain_count);

    // The loaders open before the trainer, so the envelope leaves room for their reservoirs.
    expected<ul::gpu_direct_storage_session_t> storage = ul::gpu_direct_storage_session_t::make();
    if (!storage) return fail_out_(storage.description_.view());
    dataset_source_t datasets;
    if (!config.mixture.empty())
        if (expected_t opened = dataset_source_t::open(datasets, config, *storage, std::move(mixture)); !opened)
            return fail_out_(opened.description_.view());

    trainer_t trainer;
    if (expected_t built = trainer_t::make(trainer, mesh, config, devices.view()); !built)
        return fail_out_(built.description_.view());
    envelope_t const &envelope = trainer.envelope;
    f64_t const mebibyte = 1024.0 * 1024.0;
    fmt::println(
        "uform_base: {} rows of {} columns per GPU: {:.1f} MiB free, {:.1f} MiB fixed + {:.1f} MiB per row, " //
        "{:.1f} MiB margin, {:.1f} MiB + {:.1f} MiB per row for the reservoir, the arena reserves {:.1f} MiB",
        envelope.rows, config.batch.max_columns_per_row, envelope.free_bytes / mebibyte,
        envelope.fixed_bytes / mebibyte, envelope.row_bytes / mebibyte, envelope.margin_bytes / mebibyte,
        config.reservoir_fixed_bytes / mebibyte, config.reservoir_row_bytes / mebibyte,
        envelope.reserve_bytes / mebibyte);
    fmt::println("uform_base: multimem tier {}", trainer.domain.description());
    if (!config.resume.empty()) {
        if (expected_t resumed = trainer.resume(config.resume); !resumed) return fail_out_(resumed.description_.view());
        fmt::println("uform_base: resumed from {} at step {}", config.resume, trainer.completed_steps);
    }
    int stepped = 0;
    if (config.mixture.empty()) {
        synthetic_source_t synthetic;
        stepped = step_(trainer, config, synthetic);
    }
    else if (expected_t made = dataset_source_t::make(datasets, trainer); !made)
        return fail_out_(made.description_.view());
    else stepped = step_(trainer, config, datasets);
    // No host tears its queue pairs down while a peer still awaits its last acknowledgement.
    if (stepped == 0)
        if (expected_t finished = mesh.topology.barrier("finish", mesh.timeout_seconds); !finished)
            return fail_out_(finished.description_.view());
    return stepped;
}

int main(int argc, char **argv) {
    // Lazy module loading deadlocks against the InfiniBand doorbell kernels' flag spins (see the mixer
    // and base conformance suites); eager loading before `cuInit` front-loads every module.
    ::setenv("CUDA_MODULE_LOADING", "EAGER", 1);
    // The trainer's telemetry writes the descriptor directly, so every line here flushes at once.
    std::setvbuf(stdout, nullptr, _IOLBF, 0);
    expected<trainer_config_t> parsed = trainer_config_t::parse(argc, argv);
    if (!parsed) return fail_out_(parsed.description_.view());
    trainer_config_t config = std::move(parsed.value_);
    mixture_t mixture;
    if (!config.mixture.empty()) {
        expected<mixture_t> read = mixture_t::read(config.mixture);
        if (!read) return fail_out_(read.description_.view());
        mixture = std::move(read.value_);
        config.batch = mixture.shape(config.batch);
        fmt::println("uform_base: mixture {} of {} datasets from {}", mixture.name, mixture.datasets.size(),
                     config.mixture);
    }
    // The library never reads the environment, so the launcher contract maps onto the config and the mesh here:
    // UFORM_DEBUG_ASSERTS arms the gradient-boundary signal probes, the rest place this process in the run.
    if (std::getenv("UFORM_DEBUG_ASSERTS") != nullptr) config.telemetry_channels |= telemetry_signals_k;
    mesh_options_t options;
    options.host_domain_index = host_domain_index_from_environment();
    options.host_domain_count = host_domain_count_from_environment();
    if (char const *multimem = std::getenv("UFORM_GPUS_PER_MULTIMEM_DOMAIN"); multimem != nullptr)
        options.gpus_per_multimem_domain = static_cast<u32_t>(std::atoi(multimem));
    if (char const *directory = std::getenv("UFORM_EXCHANGE_DIR"); directory != nullptr)
        options.directory = sz::string_view_t {directory, std::strlen(directory)};
    if (char const *job = std::getenv("UFORM_JOB_ID"); job != nullptr)
        options.job = sz::string_view_t {job, std::strlen(job)};
    return train_(config, options, std::move(mixture));
}
