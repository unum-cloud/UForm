/**
 *  @file tools/base.cu
 *  @author Ash Vardanian
 *  @date July 6, 2026
 *  @brief The Base training binary — a thin @c main over the training library in
 *      @c uform/train/trainer.hpp .
 *
 *  It parses the command line into a @ref trainer_config_t , maps the launcher's environment onto
 *  the mesh options, bootstraps the mesh, makes the trainer and its batch source, and steps until
 *  @c config.steps with telemetry and checkpoints on cadence.
 *
 *  The batch source is the mixture: @c --mixture @c tokens (the default) trains the byte token head
 *  alone on the synthetic draw — span-masked byte pretraining, the "Uniform Byte Models" root the
 *  Reranker and Embedder warm-start from — and @c --mixture @c omni adds the raw-media
 *  masked-autoencoder heads (image / video patches, 1-D audio frames) over one heterogeneous batch.
 *  A @c --mixture path to a mixture JSON in @c scripts/mixtures trains on its datasets instead,
 *  lighting the media slices its entries name. @c --eager drops the model-region capture.
 *
 *  @c --choices-weight trains calibrated choices through the byte decoder over the mixture's
 *  @c choices entries, and @c --brier-weight sets the Brier term's weight beside their soft
 *  cross-entropy, 0.5 unless given.
 *
 *  Every flag reads its default from @c UFORM_<FLAG> , so @c UFORM_TIER=tiny stands for @c --tier
 *  @c tiny . Each step prints, at most once a second, the steps done of the run. On one host the
 *  first SIGINT or SIGTERM stops the run before its next step: it writes the checkpoint at the
 *  steps completed, prints the phase it stopped in, and exits with 130. Across hosts a signal ends
 *  the process at once, since one host leaving the step loop alone would leave its peers waiting
 *  in their next collective forever.
 */
#include <cmath>   // `std::isfinite`
#include <csignal> // `std::signal`, `SIGINT`, `SIGTERM`
#include <cstdlib> // `::setenv`, `std::getenv` — the launcher-contract boundary
#include <cstring> // `std::strlen`
#include <cstdio>  // `std::setvbuf`, `stdout`

#include <array>       // `std::array` - the flag table
#include <limits>      // `std::numeric_limits` - the bounds of the 32-bit counts
#include <string_view> // `std::string_view`

#include <cuda.h>       // `cuInit`, `cuDeviceGetCount`
#include <fmt/format.h> // `fmt::println`

#include "uform/train/datasets.hpp" // `mixture_t`, `dataset_source_t`, and the trainer

#include "train/numerics.hpp" // `gpu_numerics_of`

#include "arguments.hpp" // `flag_t`, `parse_or_exit`, `stop_requested`, `verb_phase_t`

/** Prints a StringZilla view as the text it spans. */
template <>
struct fmt::formatter<ashvardanian::stringzilla::string_view_t> : fmt::formatter<std::string_view> {
    auto format(std::string_view text, fmt::format_context& context) const {
        return fmt::formatter<std::string_view>::format(text, context);
    }
};

namespace ue = unum::uform::encoder;
namespace ul = unum::uform::loaders;
namespace tools = unum::uform::tools;
using namespace unum::uform::train;

/** Tag @p message with the binary name, write it to standard error, and hand back
 *  main's failure code. */
[[nodiscard]] static int fail_out_(sz::string_view_t message) noexcept {
    fmt::println(stderr, "uform_base: {}", message);
    return 1;
}

/** Step @p trainer from its completed steps to @c config.steps over @p source , with
 *  telemetry, checkpoints and held-out evaluations over @p held_out on cadence, and a
 *  checkpoint at the end, or at the step a signal stopped the run before. */
[[nodiscard]] static int step_(trainer_t& trainer, trainer_config_t const& config, batch_source_t source,
                               batch_source_t held_out) noexcept {
    auto checkpoint = [&]() noexcept {
        expected_t saved = trainer.save(config.checkpoint);
        if (saved)
            fmt::println("uform_base: checkpoint written to {} at step {}", config.checkpoint, trainer.completed_steps);
        return saved;
    };
    step_report_t report;
    tools::verb_phase_t phase {.name = "train", .unit = "steps"};
    for (u32_t ordinal = trainer.completed_steps; ordinal < config.steps; ++ordinal) {
        if (tools::stop_requested) break;
        if (expected_t stepped = trainer.step(source, optimizer_step_t {ordinal}, report); !stepped) {
            fmt::println(stderr, "uform_base: step {} failed: {}", ordinal, stepped.description_);
            return 1;
        }
        u32_t const completed = ordinal + 1u;
        // The loop polls the flag itself, so the answer the handler gives the library goes unread.
        [[maybe_unused]] uform_continuation_t const reported = tools::verb_progress(completed, config.steps, &phase);
        if (config.report_every && completed % config.report_every == 0u)
            trainer.telemetry(optimizer_step_t {ordinal}).report();
        if (config.eval_every && completed % config.eval_every == 0u) {
            // Held-out draws key past the training steps, so a synthetic source never repeats
            // a step's batch.
            auto [evaluation, status] = trainer.evaluate(held_out, data_epoch_t {config.steps + ordinal});
            if (!status) return fail_out_(status.description_.view());
            evaluation.report();
        }
        if (!config.checkpoint.empty() && config.checkpoint_every && completed % config.checkpoint_every == 0u &&
            completed != config.steps)
            if (expected_t saved = checkpoint(); !saved) return fail_out_(saved.description_.view());
    }
    if (tools::stop_requested) tools::verb_stopped(phase);
    if (!config.checkpoint.empty())
        if (expected_t saved = checkpoint(); !saved) return fail_out_(saved.description_.view());
    if (expected_t drained = trainer.synchronize(); !drained) return fail_out_(drained.description_.view());
    {
        switch_to_gpu_t gpu_scope {trainer.streams[0]};
        if (expected_t reported = trainer.ledger.report(trainer.streams[0]); !reported)
            fmt::println(stderr, "uform_base: {}", reported.description_);
    }
    if (tools::stop_requested) return tools::exit_stopped_k;
    fmt::println("uform_base: done");
    return 0;
}

/** Bootstrap the mesh over every visible device, make the trainer and the source @p mixture
 *  names, and step. */
[[nodiscard]] static int train_(trainer_config_t config, mesh_options_t options, mixture_t mixture) noexcept {
    if (cuInit(0) != CUDA_SUCCESS) return fail_out_("cuInit failed");
    int device_count = 0;
    if (cuDeviceGetCount(&device_count) != CUDA_SUCCESS || device_count <= 0) return fail_out_("no CUDA devices");
    st::basic_vector<i32_t> devices;
    for (int device = 0; device != device_count; ++device)
        if (st::failed(devices.push_back(static_cast<i32_t>(device))))
            return fail_out_("device table allocation failed");
    if (nk_capabilities_enabled_cuda(0, &options.capabilities) != nk_success_k)
        return fail_out_("NumKong cannot query device 0");

    mesh_t mesh;
    if (expected_t joined = mesh_t::make(mesh, options, devices); !joined) return fail_out_(joined.description_.view());
    fmt::println("uform_base: tokens{}{}{} mixture, {} GPU(s), host_domain {} of {}",
                 config.batch.contains(modality_t::image_k) ? "+image" : "",
                 config.batch.contains(modality_t::video_k) ? "+video" : "",
                 config.batch.contains(modality_t::audio_k) ? "+audio" : "", device_count, options.host_domain_index,
                 options.host_domain_count);

    if (config.eval_every && config.eval_mixture.empty() && !config.mixture.empty())
        return fail_out_("--eval-every over a mixture needs --eval-mixture, whose shards training never draws");
    mixture_t held_out_mixture;
    if (config.eval_every && !config.eval_mixture.empty()) {
        auto [eval_mixture, read_status] = mixture_t::read(config.eval_mixture);
        if (!read_status) return fail_out_(read_status.description_.view());
        held_out_mixture = std::move(eval_mixture);
    }

    // Counted before the spawn, since a caller-inclusive pool pins this thread to one core.
    auto [topology, make_topology_status] = topology_t::make();
    if (!make_topology_status) return fail_out_(make_topology_status.description_.view());
    std::size_t const allowed_cores = topology.logical_cores_count();
    fu::pool_t decoders; // empty, so the calling thread decodes, unless a mixture pairs media
    if (mixture.pairs() || held_out_mixture.pairs())
        if (expected_t spawned = spawn_pool(decoders, topology, allowed_cores, fu::caller_inclusive_k,
                                            fu_capabilities_all_k);
            !spawned)
            return fail_out_(spawned.description_.view());
    std::size_t const spawned_threads = decoders.threads_count();

    // The loaders open before the trainer, so the envelope leaves room for their reservoirs.
    auto [storage, make_status] = ul::gpu_direct_storage_session_t::make(
        allowed_cores > spawned_threads ? allowed_cores - spawned_threads : 1);
    if (!make_status) return fail_out_(make_status.description_.view());
    dataset_source_t datasets, held_out;
    if (!config.mixture.empty())
        if (expected_t opened = dataset_source_t::open(datasets, config, storage, std::move(mixture), decoders,
                                                       config.batches_per_fill);
            !opened)
            return fail_out_(opened.description_.view());
    if (config.eval_every && !config.eval_mixture.empty()) {
        // Each source charges its own reservoir, so the envelope makes room for both.
        std::size_t const training_fixed = config.reservoir_fixed_bytes, training_row = config.reservoir_row_bytes;
        // Every evaluation refills, so each scores fresh entries rather than a fill 16 evals old.
        if (expected_t opened = dataset_source_t::open(held_out, config, storage, std::move(held_out_mixture), decoders,
                                                       1);
            !opened)
            return fail_out_(opened.description_.view());
        config.reservoir_fixed_bytes += training_fixed, config.reservoir_row_bytes += training_row;
    }

    trainer_t trainer;
    if (expected_t built = trainer_t::make(trainer, mesh, config, devices, gpu_numerics_of(options.capabilities));
        !built)
        return fail_out_(built.description_.view());
    envelope_t const& envelope = trainer.envelope;
    f64_t const mebibyte = 1024.0 * 1024.0;
    fmt::println(
        "uform_base: {} rows of {} columns and {} segments per GPU: {:.1f} MiB free, {:.1f} MiB fixed + "      //
        "{:.1f} MiB per row, {:.1f} MiB margin, {:.1f} MiB + {:.1f} MiB per row for the reservoir, the arena " //
        "reserves {:.1f} MiB",
        envelope.rows, config.batch.max_columns_per_row, trainer.config.batch.max_segments_per_gpu,
        envelope.free_bytes / mebibyte, envelope.fixed_bytes / mebibyte, envelope.row_bytes / mebibyte,
        envelope.margin_bytes / mebibyte, config.reservoir_fixed_bytes / mebibyte,
        config.reservoir_row_bytes / mebibyte, envelope.reserve_bytes / mebibyte);
    fmt::println("uform_base: multimem tier {}", trainer.domain.description());
    if (!config.resume.empty()) {
        if (expected_t resumed = trainer.resume(config.resume); !resumed) return fail_out_(resumed.description_.view());
        fmt::println("uform_base: resumed from {} at step {}", config.resume, trainer.completed_steps);
    }
    int stepped = 0;
    if (!config.eval_mixture.empty() && config.eval_every)
        if (expected_t made = dataset_source_t::make(held_out, trainer, 1); !made)
            return fail_out_(made.description_.view());
    synthetic_source_t synthetic;
    batch_source_t const evaluated = config.eval_mixture.empty() ? batch_source_of(synthetic)
                                                                 : batch_source_of(held_out);
    if (config.mixture.empty()) stepped = step_(trainer, config, batch_source_of(synthetic), evaluated);
    else if (expected_t made = dataset_source_t::make(datasets, trainer, config.batches_per_fill); !made)
        return fail_out_(made.description_.view());
    else stepped = step_(trainer, config, batch_source_of(datasets), evaluated);
    // No host tears its queue pairs down while a peer still awaits its last acknowledgement; a
    // stopped run is a single host's, so it has no peer to wait for.
    if (stepped == 0)
        if (expected_t finished = mesh.topology.barrier("finish", mesh.timeout_seconds); !finished)
            return fail_out_(finished.description_.view());
    return stepped;
}

/** Exits with status 1 after printing that @p flag 's @p value does not parse as @p expected ,
 *  the line every flag table prints. */
template <typename value_type_>
[[noreturn]] static void exit_unparsed_(std::string_view flag, value_type_ const& value,
                                        std::string_view expected) noexcept {
    fmt::println(stderr, "{}=\"{}\" does not parse, expected {}", flag, value, expected);
    std::exit(1);
}

/** The finite non-negative real @p value , or the exit naming @p flag ; a @p positive one
 *  also refuses zero. */
static f32_t weight_or_exit_(std::string_view flag, double value, bool positive = false) noexcept {
    if (!std::isfinite(value) || (positive && value == 0))
        exit_unparsed_(flag, value, positive ? "a positive real" : "a non-negative real");
    return static_cast<f32_t>(value);
}

/** The names @c --objective accepts, in the order of @c objectives_k. */
static constexpr std::array<std::string_view, 3> objective_names_k {"masked", "contrastive", "joint"};

/** The objective each name of @c objective_names_k selects. */
static constexpr std::array<uform_objective_t, 3> objectives_k {uform_objective_masked_k, uform_objective_contrastive_k,
                                                                uform_objective_joint_k};

/** The names @c --pair-loss accepts, in the order of @c pair_loss_forms_k. */
static constexpr std::array<std::string_view, 4> pair_loss_names_k {"derived", "balanced", "ranked", "sigmoid"};

/** The pair loss each name of @c pair_loss_names_k selects. */
static constexpr std::array<uform_pair_loss_form_t, 4> pair_loss_forms_k {
    uform_pair_loss_form_derived_k, uform_pair_loss_form_balanced_k, uform_pair_loss_form_ranked_k,
    uform_pair_loss_form_sigmoid_k};

/** Parse @p argv and its @c UFORM_ variables into a run description, or exit: with status 0 after
 *  the usage on @c --help , with status 1 after one line on a value that does not parse. */
[[nodiscard]] static trainer_config_t parse_config_(int argc, char const* const* argv) noexcept {
    using tools::choice_t, tools::count_t, tools::flag_t, tools::switch_t;
    trainer_config_t config;
    std::size_t constexpr u32_max_k = (std::numeric_limits<u32_t>::max)();
    std::string_view tier_name, mixture, eval_mixture, checkpoint, resume, telemetry;
    count_t steps {config.steps, 0, u32_max_k}, rows {0, 0, u32_max_k},
        warmup_steps {config.warmup_steps, 0, u32_max_k}, nvdec_engines {config.nvdec_engines, 0, u32_max_k},
        batches_per_fill {config.batches_per_fill}, checkpoint_every {config.checkpoint_every, 0, u32_max_k},
        report_every {config.report_every, 0, u32_max_k}, eval_every {config.eval_every, 0, u32_max_k};
    // A negative rate is no value a flag parses to, so it stands for the tier's own.
    double learning_rate = -1, late_interaction_weight = config.late_interaction.weight,
           choices_weight = config.choices.weight, brier_weight = config.choices.brier_weight,
           tcr_lambda = config.tcr_lambda, tcr_epsilon_squared = config.tcr_epsilon_squared,
           lipschitz_lambda = config.lipschitz_lambda, grad_spike_threshold = config.grad_spike_threshold;
    std::uint32_t seed = static_cast<std::uint32_t>(config.seed);
    choice_t objective_choice {objective_names_k};
    choice_t pair_loss_choice {pair_loss_names_k};
    switch_t eager = switch_t::absent_k;
    std::array<flag_t, 26> const flags {{
        {"--tier", "name", "tiny, small, medium, large or xl: the width, depth, rows and learning rate", &tier_name},
        {"--mixture", "path", "a mixture JSON, or the synthetic tokens or omni draw", &mixture},
        {"--objective", "name", "masked, contrastive or joint: which losses a step relaxes", &objective_choice},
        {"--steps", "count", "the run's length in optimizer steps, counted across resumes", &steps},
        {"--rows", "count", "rows per GPU, zero to measure them from free memory", &rows},
        {"--learning-rate", "real", "the peak learning rate, the tier's unless given", &learning_rate},
        {"--warmup-steps", "count", "linear warmup steps, zero for a tenth of the run up to 400", &warmup_steps},
        {"--seed", "seed", "the run's base seed, which every draw derives from", &seed},
        {"--nvdec-engines", "count", "the NVDEC engines a GPU has, which a video mixture needs", &nvdec_engines},
        {"--batches-per-fill", "count", "steps' batches one data fill packs per GPU", &batches_per_fill},
        {"--pair-loss", "name", "the contrastive pair loss, each starting at its own logit scale", &pair_loss_choice},
        {"--late-interaction-weight", "real", "the late-interaction loss's weight, zero for none",
         &late_interaction_weight},
        {"--choices-weight", "real", "the calibrated choices' weight, zero for none", &choices_weight},
        {"--brier-weight", "real", "the Brier term's weight beside the choices' cross-entropy", &brier_weight},
        {"--tcr-lambda", "real", "the Total Coding Rate's weight, zero for none", &tcr_lambda},
        {"--tcr-epsilon-squared", "real", "the Total Coding Rate's positive distortion target", &tcr_epsilon_squared},
        {"--lipschitz-lambda", "real", "the Lipschitz penalty's weight, zero for none", &lipschitz_lambda},
        {"--grad-spike-threshold", "real", "the gradient-norm spike that fails a step, zero for none",
         &grad_spike_threshold},
        {"--checkpoint", "path", "where the checkpoint is written, none unless given", &checkpoint},
        {"--checkpoint-every", "count", "steps between checkpoints, zero for the last alone", &checkpoint_every},
        {"--report-every", "count", "steps between telemetry lines, zero for none", &report_every},
        {"--eval-every", "count", "steps between held-out evaluations, zero for none", &eval_every},
        {"--eval-mixture", "path", "the mixture JSON the held-out evaluations draw from", &eval_mixture},
        {"--resume", "path", "a checkpoint to continue from, at its saved step", &resume},
        {"--telemetry", "names", "signals or timings, comma-separated: the extra ledger channels", &telemetry},
        {"--eager", "", "run every step eagerly, without capturing the model region", &eager},
    }};
    tools::parse_or_exit(argc, argv, "uform_base --tier <name> [flags]", flags);

    trainer_tier_t const* tier = nullptr;
    for (trainer_tier_t const& known : trainer_tiers_k)
        if (tier_name == known.name) tier = &known;
    if (tier_name.empty())
        fmt::println(stderr, "uform_base: pass --tier tiny, small, medium, large or xl"), std::exit(1);
    if (!tier) exit_unparsed_("--tier", tier_name, "tiny, small, medium, large or xl");
    config.tier = *tier;
    config.hidden = tier->hidden, config.depth = tier->depth;
    config.learning_rate = learning_rate < 0 ? tier->learning_rate
                                             : weight_or_exit_("--learning-rate", learning_rate, true);
    // Every tier trains the nested prefixes 64…1024 that fit its width, then the full width.
    for (u32_t width = 64u; width <= (tier->hidden < 1024u ? tier->hidden : 1024u) && width != tier->hidden;
         width *= 2u)
        config.contrastive.matryoshka_dimensions[config.contrastive.matryoshka_count++] = width;
    config.contrastive.matryoshka_dimensions[config.contrastive.matryoshka_count++] = tier->hidden;

    if (mixture.ends_with(".json")) config.mixture = mixture;
    else if (mixture == "omni") {
        // The omni preset: 8-pixel RGB patches, 2-frame RGB tubes, 320-sample audio frames.
        config.batch.image = ue::image_batch_shape_t {8u, 3u};
        config.batch.video = ue::video_batch_shape_t {8u, 3u, 2u, 1u};
        config.batch.audio = ue::audio_batch_shape_t {320u, 16000u};
    }
    else if (!mixture.empty() && mixture != "tokens")
        exit_unparsed_("--mixture", mixture, "a path ending in .json, tokens or omni");
    if (!eval_mixture.empty() && !eval_mixture.ends_with(".json"))
        exit_unparsed_("--eval-mixture", eval_mixture, "a path ending in .json");
    config.eval_mixture = eval_mixture;
    config.checkpoint = checkpoint;
    config.resume = resume;
    for (std::string_view names = telemetry; !names.empty();) {
        std::size_t const comma = names.find(',');
        std::string_view const name = names.substr(0, comma);
        if (name == "signals") config.telemetry_channels |= telemetry_signals_k;
        else if (name == "timings") config.telemetry_channels |= telemetry_timings_k;
        else exit_unparsed_("--telemetry", telemetry, "signals or timings, comma-separated");
        names = comma == std::string_view::npos ? std::string_view {} : names.substr(comma + 1);
    }

    uform_objective_t const objective = objectives_k[objective_choice.selected];
    uform_pair_loss_form_t const pair_loss = pair_loss_forms_k[pair_loss_choice.selected];
    switch (objective) {
    case uform_objective_masked_k: config.objective = objective_t::masked_k; break;
    case uform_objective_contrastive_k: config.objective = objective_t::contrastive_k; break;
    case uform_objective_joint_k: config.objective = objective_t::joint_k; break;
    }
    if (pair_loss != uform_pair_loss_form_derived_k) {
        config.contrastive.pair_loss = pair_loss == uform_pair_loss_form_balanced_k
                                           ? contrastive_pair_loss_form_t::balanced_k
                                       : pair_loss == uform_pair_loss_form_ranked_k
                                           ? contrastive_pair_loss_form_t::ranked_k
                                           : contrastive_pair_loss_form_t::sigmoid_k;
        // Each form starts at its own logit scale: the sigmoid sum at SigLIP's β=10, b=−10.
        bool const sigmoid = pair_loss == uform_pair_loss_form_sigmoid_k;
        config.contrastive.inverse_temperature = sigmoid ? 10.0f : 1.0f;
        config.contrastive.bias = sigmoid ? -10.0f : 0.0f;
    }
    config.steps = static_cast<u32_t>(steps.value);
    config.batch.max_rows_per_gpu = static_cast<u32_t>(rows.value);
    config.warmup_steps = static_cast<u32_t>(warmup_steps.value);
    config.seed = seed;
    config.nvdec_engines = static_cast<u32_t>(nvdec_engines.value);
    config.batches_per_fill = batches_per_fill.value;
    config.checkpoint_every = static_cast<u32_t>(checkpoint_every.value);
    config.report_every = static_cast<u32_t>(report_every.value);
    config.eval_every = static_cast<u32_t>(eval_every.value);
    config.late_interaction.weight = weight_or_exit_("--late-interaction-weight", late_interaction_weight);
    config.choices.weight = weight_or_exit_("--choices-weight", choices_weight);
    config.choices.brier_weight = weight_or_exit_("--brier-weight", brier_weight);
    config.tcr_lambda = weight_or_exit_("--tcr-lambda", tcr_lambda);
    config.tcr_epsilon_squared = weight_or_exit_("--tcr-epsilon-squared", tcr_epsilon_squared, true);
    config.lipschitz_lambda = weight_or_exit_("--lipschitz-lambda", lipschitz_lambda);
    config.grad_spike_threshold = weight_or_exit_("--grad-spike-threshold", grad_spike_threshold);
    if (eager == switch_t::present_k) config.capture = eager_k;
    return config;
}

int main(int argc, char** argv) {
    // Lazy module loading deadlocks against the InfiniBand doorbell kernels' flag spins (see the
    // mixer and base conformance suites); eager loading before `cuInit` front-loads every module.
    ::setenv("CUDA_MODULE_LOADING", "EAGER", 1);
    // The trainer's telemetry writes the descriptor directly, so every line here flushes at once.
    std::setvbuf(stdout, nullptr, _IOLBF, 0);
    trainer_config_t config = parse_config_(argc, argv);
    mixture_t mixture;
    if (!config.mixture.empty()) {
        auto [training_mixture, status] = mixture_t::read(config.mixture);
        if (!status) return fail_out_(status.description_.view());
        mixture = std::move(training_mixture);
        bool queried = false;
        for (mixture_entry_t const& entry : mixture.datasets)
            for (late_interaction_side_t const side : {entry.sides.anchor, entry.sides.positive, entry.sides.negative})
                queried |= side == late_interaction_side_t::query_k ||
                           side == late_interaction_side_t::query_and_document_k;
        if (config.late_interacts() && !queried) {
            fmt::println(stderr,
                         "uform_base: late interaction needs query rows, but no entry of {} names a `query` side; " //
                         "add \"query\": \"anchor\" to an entry or pass --late-interaction-weight 0",
                         config.mixture);
            return 1;
        }
        config.batch = mixture.shape(config.batch);
        fmt::println("uform_base: mixture {} of {} datasets from {}", mixture.name, mixture.datasets.size(),
                     config.mixture);
    }
    if (config.chooses() && !config.batch.max_choices_per_segment)
        return fail_out_("--choices-weight needs a mixture with a \"choices\" entry; pass one or --choices-weight 0");
    // Media runs take rows of two images or recordings; text-only runs take long documents.
    config.batch.max_columns_per_row = config.batch.media_live() ? config.tier.media_columns : config.tier.text_columns;
    fmt::println("uform_base: tier {}: hidden {}, depth {}, {} columns", config.tier.name, config.hidden, config.depth,
                 config.batch.max_columns_per_row);
    // The library never reads the environment, so the launcher contract places this process here.
    mesh_options_t options;
    options.host_domain_index = host_domain_index_from_environment();
    options.host_domain_count = host_domain_count_from_environment();
    if (char const* multimem = std::getenv("UFORM_GPUS_PER_MULTIMEM_DOMAIN"); multimem != nullptr)
        options.gpus_per_multimem_domain = static_cast<u32_t>(std::atoi(multimem));
    if (char const* directory = std::getenv("UFORM_EXCHANGE_DIR"); directory != nullptr)
        options.directory = sz::string_view_t {directory, std::strlen(directory)};
    if (char const* job = std::getenv("UFORM_JOB_ID"); job != nullptr)
        options.job = sz::string_view_t {job, std::strlen(job)};
    // Hosts step in lockstep, so only a lone host may leave the loop at a step of its own choosing.
    if (options.host_domain_count == 1) {
        std::signal(SIGINT, tools::request_stop);
        std::signal(SIGTERM, tools::request_stop);
    }
    return train_(config, options, std::move(mixture));
}
