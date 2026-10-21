/**
 *  @brief The implementation of the Base pre-training library declared in @c src/train/base.cuh : the
 *      synthetic-mixture kernels, the weight seeding and checkpoint helpers, and every out-of-line body of
 *      @ref trainer_config_t and @ref base_replica_t .
 *  @file src/train/base.cu
 *  @author Ash Vardanian
 *  @date July 6, 2026
 *
 *  No @c main lives here — the binary's thin entry is @c tools/base.cu , and the conformance test is
 *  @c test/base.cu .
 *
 *  @section base_impl_mixture The synthetic mixture, drawn like the real mixer composes
 *
 *  Each step partitions the row budget across the live modalities with a keyed draw — any counts, no
 *  per-modality caps — and assigns the rows in the @ref modality_t order (tokens, image, video, audio), each row
 *  modality-uniform with one segment, so projections tile contiguous row runs. The partition, the contents, and
 *  the corruption are all pure functions of @c (seed,data_epoch,gpu_index_in_colocated_domain) — the mesh's
 *  global world coordinate, so every world GPU trains on distinct data: every run and the capture-vs-eager
 *  parity re-run draw bitwise-identically, with no files.
 *
 *  @section base_impl_schedule The learning-rate schedule
 *
 *  The rate is the trainer's policy, computed on host per step (legal because the optimizer region is eager):
 *  a linear ramp from zero to the peak over @c warmup_steps , then a cosine decay to @c min_learning_rate over
 *  the run's remainder — the validated draft envelope's shape.
 */
#include <cmath>   // `std::cos` — the cosine decay, `std::sqrt` — the initialization's fan-in scale, `std::isfinite`
#include <cstring> // `std::strlen`

#include <unistd.h> // `::write`, `STDOUT_FILENO` — the host diagnostic channel

#include <algorithm> // `std::max`, `std::min`
#include <charconv>  // `std::from_chars` — the command line's numbers
#include <limits>    // `std::numeric_limits` — the i16 coordinate ceiling
#include <numbers>   // `std::numbers::pi_v` — the decay's half-period

#include <cuda.h> // `cuMemcpyHtoDAsync`, `cuMemcpyDtoHAsync`, `cuStreamSynchronize`, `cuMemsetD8`

#include "train/base.cuh"

using namespace unum::uform::core;
using namespace unum::uform::train;
namespace ut = unum::uform::train;

/**
 *  @brief Fill the token grid with a keyed synthetic byte corpus — every row, media rows included (their bytes
 *      are simply never read); the draw is a pure function of @c (seed,data_epoch,gpu_index,row,column) with
 *      the @b world GPU coordinate.
 *
 *  @b grid_size @c [rows,1,1] , one CTA per row; no dynamic shared memory.
 */
static __global__ void synthetic_tokens_kernel_(nk::tensor_span<token_id_t, 2> tokens, u64_t seed,
                                                data_epoch_t data_epoch, u32_t gpu_index_in_colocated_domain) {
    u32_t const columns = static_cast<u32_t>(tokens.extent(1));
    u32_t const row = blockIdx.x;
    for (u32_t column = threadIdx.x; column < columns; column += blockDim.x) {
        u64_t const draw = splitmix64_hash_t {}(seed, data_epoch.ordinal, gpu_index_in_colocated_domain, row, column);
        tokens[row, column] = static_cast<token_id_t>(draw % 256u);
    }
}

/**
 *  @brief Write every row's segment descriptor from the host-drawn partition: rows below @p image_begin are
 *      tokens, then the image / video / audio runs in @ref modality_t order (an empty run leaves consecutive
 *      begins equal).
 *
 *  One full-width segment per row; image / video rows also carry the near-square
 *  @c (patch_rows,patch_cols) aspect the coordinate adapter reads.
 *
 *  @b grid_size @c [rows,1,1] , one CTA per row; only thread 0 writes; no dynamic shared memory.
 */
static __global__ void synthetic_segments_kernel_(u32r1_span_t segment_offsets, u32r1_span_t segment_lengths,
                                                  modality_span_t segment_modality, image_shape_span_t image_shapes,
                                                  video_shape_span_t video_shapes, u32_t columns, u32_t image_begin,
                                                  u32_t video_begin, u32_t audio_begin, u16_t patch_rows,
                                                  u16_t patch_cols) {
    u32_t const rows = static_cast<u32_t>(segment_lengths.extent(0));
    u32_t const row = blockIdx.x;
    if (threadIdx.x != 0u) return;
    modality_t modality = modality_t::text_k;
    if (row >= audio_begin) modality = modality_t::audio_k;
    else if (row >= video_begin) modality = modality_t::video_k;
    else if (row >= image_begin) modality = modality_t::image_k;
    segment_offsets[row] = row * columns;
    segment_lengths[row] = columns;
    segment_modality[row] = modality;
    if (modality == modality_t::image_k) {
        image_shapes[row, 0] = patch_rows;
        image_shapes[row, 1] = patch_cols;
    }
    if (modality == modality_t::video_k) {
        video_shapes[row, 0] = 1u;
        video_shapes[row, 1] = patch_rows;
        video_shapes[row, 2] = patch_cols;
    }
    if (row == 0u) segment_offsets[rows] = rows * columns;
}

/**
 *  @brief Fill one media pixel tensor (flattened @b [rows, patch_scalars] ) with keyed byte noise — the raw
 *      image / video content the tower re-patchifies, identical across runs.
 *
 *  @b grid_size capped grid-stride over @c rows·patch_scalars scalars; no dynamic shared memory.
 */
static __global__ void synthetic_pixels_kernel_(nk::tensor_span<pixel_t, 2> media, u64_t seed, data_epoch_t data_epoch,
                                                u32_t gpu_index_in_colocated_domain, u32_t salt) {
    u32_t const width = static_cast<u32_t>(media.extent(1));
    std::size_t const total = static_cast<std::size_t>(media.extent(0)) * width;
    std::size_t const stride = static_cast<std::size_t>(gridDim.x) * blockDim.x;
    for (std::size_t index = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x; index < total;
         index += stride) {
        u32_t const row = static_cast<u32_t>(index / width);
        u32_t const column = static_cast<u32_t>(index % width);
        u64_t const draw = splitmix64_hash_t {}(seed, data_epoch.ordinal + salt, gpu_index_in_colocated_domain, row,
                                                column);
        media[row, column] = static_cast<pixel_t>(draw & 0xFFu);
    }
}

/**
 *  @brief Fill the audio waveform tensor (flattened @b [frames, chunk_samples] ) with keyed noise in
 *      @b [-1,1] — the raw samples the tower's learnable frontend projects.
 *
 *  @b grid_size capped grid-stride over @c frames·chunk_samples scalars; no dynamic shared memory.
 */
static __global__ void synthetic_audio_kernel_(f32r2_span_t frames, u64_t seed, data_epoch_t data_epoch,
                                               u32_t gpu_index_in_colocated_domain, u32_t salt) {
    u32_t const width = static_cast<u32_t>(frames.extent(1));
    std::size_t const total = static_cast<std::size_t>(frames.extent(0)) * width;
    std::size_t const stride = static_cast<std::size_t>(gridDim.x) * blockDim.x;
    for (std::size_t index = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x; index < total;
         index += stride) {
        u32_t const row = static_cast<u32_t>(index / width);
        u32_t const column = static_cast<u32_t>(index % width);
        u64_t const draw = splitmix64_hash_t {}(seed, data_epoch.ordinal + salt, gpu_index_in_colocated_domain, row,
                                                column);
        f32_t const uniform = static_cast<f32_t>(draw >> 40) * 0x1.0p-24f;
        frames[row, column] = uniform * 2.0f - 1.0f;
    }
}

void ut::report_line(sz::string_view_t line) noexcept {
    ::write(STDOUT_FILENO, line.data(), line.size());
    ::write(STDOUT_FILENO, "\n", 1);
}

/** One media row's @c (patch_rows,patch_cols) grid factorization — the aspect descriptor the
 *  coordinate adapter reads. */
struct grid_factor_t {
    u16_t patch_rows = 0, patch_cols = 0;
};

/** The near-square factorization of one media row's @p columns patches. Picks the largest divisor no
 *  greater than the square root as the column count, so the grid stays as square as the budget allows. */
[[nodiscard]] static grid_factor_t square_factor_(u32_t columns) noexcept {
    u32_t side = 1u;
    while ((side + 1u) * (side + 1u) <= columns) ++side;
    while (side > 1u && columns % side != 0u) --side;
    return grid_factor_t {static_cast<u16_t>(columns / side), static_cast<u16_t>(side)};
}

/** Deterministic uniform weight fill matched to ~N(0,std²) variance, identical on every replica —
 *  replica identity is an initialization invariant the conformance tests assert. */
[[nodiscard]] static f32_t hash_fill_(std::size_t index, u32_t salt, f32_t std) noexcept {
    u64_t const mixed = splitmix64_hash_t {}(salt, 0u, 1u, static_cast<u32_t>(index), 0u);
    f32_t const uniform = static_cast<f32_t>(mixed >> 40) * 0x1.0p-24f;
    return (uniform - 0.5f) * 3.4641016f * std;
}

/** Whether @p matrix projects back into the residual stream: each block's attention and MLP
 *  output. */
[[nodiscard]] static bool writes_residual_(differentiable_weights_t const &matrix) noexcept {
    return matrix.name.view().ends_with(".out") || matrix.name.view().ends_with(".down");
}

expected_t ut::seed_weights(st::basic_vector<differentiable_weights_t> const &weights, gpu_stream_t &stream) noexcept {
    std::span<differentiable_weights_t const> const list {weights};
    std::size_t residual_writers = 0;
    for (differentiable_weights_t const &matrix : list) residual_writers += writes_residual_(matrix) ? 1u : 0u;
    for (std::size_t weight_index = 0; weight_index != list.size(); ++weight_index) {
        differentiable_weights_t const &matrix = list[weight_index];
        std::size_t const rows = matrix.weights.extent(0), columns = matrix.weights.extent(1);
        // Matrices are [outputs,inputs]; the residual writers split one unit of variance across the
        // stack.
        f32_t const deviation =
            1.0f / std::sqrt(static_cast<f32_t>(writes_residual_(matrix) ? columns * residual_writers : columns));
        nk::tensor<bf16_t, pinned_alloc<bf16_t>, 1> staging =
            nk::tensor<bf16_t, pinned_alloc<bf16_t>, 1>::uninitialized({rows * columns}, pinned_alloc<bf16_t> {}).value;
        if (!staging) return expected_t::fail(status_t::bad_alloc_k, "weight staging failed (matrix={})", weight_index);
        bf16r1_span_t const host = staging.span();
        for (std::size_t index = 0; index != host.extent(0); ++index)
            host[index] = __float2bfloat16( //
                rows == 1 ? 1.0f : hash_fill_(index, static_cast<u32_t>(weight_index) + 11u, deviation));
        if (expected_t copied = copy_host_to_device(staging.view(), matrix.weights.flatten<1>(), stream); !copied)
            return copied;
        // The staging tensor dies this iteration — the enqueued copy must land first.
        cuStreamSynchronize(stream.get());
    }
    return {};
}

f32_t trainer_config_t::learning_rate_for(optimizer_step_t optimizer_step) const noexcept {
    u32_t const warmup = warmup_steps ? warmup_steps : std::min(400u, steps / 10u);
    if (optimizer_step.ordinal < warmup)
        return learning_rate * static_cast<f32_t>(optimizer_step.ordinal + 1u) / static_cast<f32_t>(warmup);
    u32_t const decay_steps = steps > warmup ? steps - warmup : 1u;
    f32_t const progress = static_cast<f32_t>(optimizer_step.ordinal - warmup) / static_cast<f32_t>(decay_steps);
    f32_t const cosine = 0.5f * (1.0f + std::cos(std::numbers::pi_v<f32_t> * std::min(progress, 1.0f)));
    return min_learning_rate + (learning_rate - min_learning_rate) * cosine;
}

/** The masking corruptor's keyed recipe — the run's knobs, its seed derived once from the run seed. */
[[nodiscard]] static masking_recipe_t masking_recipe_(trainer_config_t const &config) noexcept {
    masking_recipe_t recipe = config.masking;
    recipe.seed = config.seed * 2654435761u + 17u;
    return recipe;
}

/** One step's keyed row partition: the live modalities in @ref modality_t order, each with its row
 *  count and the first row of its contiguous run. */
struct row_partition_t {

    /** Rows per modality, indexed by @ref modality_t — zero for absent modalities. */
    u32_t rows[4] = {};

    /** The first row of each modality's run — prefix sums of @ref rows in modality order. */
    u32_t begins[4] = {};
};

/** Partition the row budget across the live modalities — any counts, no per-modality caps, exactly the
 *  freedom the real mixer has. Every live modality gets the floor share; the keyed draw rotates which ones
 *  carry the remainder, so every mixture recurs across passes (and a one-row omni batch deterministically
 *  leaves most heads with zero live rows — the dynamic-M edge). */
[[nodiscard]] static row_partition_t partition_rows_(trainer_config_t const &config,
                                                     u32_t gpu_index_in_colocated_domain,
                                                     data_epoch_t data_epoch) noexcept {
    multimodal_batch_shape_t const &shape = config.batch;
    bool const live[4] = {true, shape.contains(modality_t::image_k), shape.contains(modality_t::video_k),
                          shape.contains(modality_t::audio_k)};
    u32_t live_count = 0;
    for (bool flag : live) live_count += flag ? 1u : 0u;
    u32_t const total = shape.max_rows_per_gpu;
    u32_t const floor_rows = total / live_count, extra = total % live_count;
    u64_t const draw = splitmix64_hash_t {}(config.seed, data_epoch.ordinal, gpu_index_in_colocated_domain, 5u, 0u);
    u32_t const rotation = static_cast<u32_t>(draw % live_count);

    row_partition_t partition;
    u32_t position = 0;
    for (std::size_t modality = 0; modality != 4; ++modality) {
        if (!live[modality]) continue;
        u32_t const bonus = ((position + live_count - rotation) % live_count) < extra ? 1u : 0u;
        partition.rows[modality] = floor_rows + bonus;
        ++position;
    }
    u32_t cursor = 0;
    for (std::size_t modality = 0; modality != 4; ++modality) {
        partition.begins[modality] = cursor;
        cursor += partition.rows[modality];
    }
    return partition;
}

/** Fold the run description into the tower's make-config: the block geometry from the batch envelope,
 *  one media entry per live batch slice — the batch shape stays the single source of modality presence. The
 *  rotary tables cover the largest coordinate a row reaches: its column count, or the absolute source frame
 *  the last tube of a one-patch clip starts at. */
[[nodiscard]] static encoder_tower_config_t tower_config_(trainer_config_t const &config) noexcept {
    multimodal_batch_shape_t const &batch = config.batch;
    u32_t const max_tokens = batch.max_rows_per_gpu * batch.max_columns_per_row;
    encoder_tower_config_t tower;
    tower.vocabulary = byte_vocabulary_k;
    tower.depth = config.depth;
    tower.block.hidden = config.hidden;
    tower.block.ffn_expansion = 2;
    tower.block.max_tokens = max_tokens;
    tower.block.max_segments = batch.max_segments_per_gpu;
    tower.block.max_segment_tokens = batch.max_columns_per_row;
    u64_t positions = batch.max_columns_per_row;
    if (batch.contains(modality_t::image_k)) {
        tower.image_patch_height = batch.image.patch_side;
        tower.image_patch_width = batch.image.patch_side;
        tower.image_channels = batch.image.channels;
    }
    if (batch.contains(modality_t::video_k)) {
        u32_t const frame_stride = std::max(batch.video.frame_stride, 1u);
        tower.video_tube_frames = batch.video.tube_frames;
        tower.video_frame_stride = frame_stride;
        tower.video_patch_height = batch.video.patch_side;
        tower.video_patch_width = batch.video.patch_side;
        tower.video_channels = batch.video.channels;
        u64_t const last_tube =
            static_cast<u64_t>(batch.max_columns_per_row - 1u) * batch.video.tube_frames * frame_stride + 1u;
        positions = std::max(positions, last_tube);
    }
    if (batch.contains(modality_t::audio_k)) tower.audio_chunk_samples = batch.audio.chunk_samples;
    tower.rope_max_positions = static_cast<u32_t>(std::min<u64_t>(positions, std::numeric_limits<u32_t>::max()));
    tower.token_width = config.late_interacts() ? config.late_interaction.width : 0u;
    return tower;
}

expected_t ut::load_weights(checkpoint_to_load_t const &checkpoint,
                            st::basic_vector<differentiable_weights_t> const &weights, gpu_stream_t &stream) noexcept {
    std::span<differentiable_weights_t const> const list {weights};
    sz::string_t name_text;
    for (differentiable_weights_t const &matrix : list) {
        name_text.clear();
        if (!format_into(name_text, "model.{}", matrix.name))
            return expected_t::fail(status_t::bad_alloc_k, "matrix name format failed ({})", matrix.name);
        nk::tensor_view<nk_bf16_t, 2> const saved = checkpoint[name_text.view()];
        if (saved.empty())
            return expected_t::fail(status_t::invalid_input_k, "the checkpoint misses {}", name_text.view());
        if (saved.extent(0) != matrix.weights.extent(0) || saved.extent(1) != matrix.weights.extent(1))
            return expected_t::fail(status_t::shape_unsupported_k, "{} is [{},{}] in the checkpoint, [{},{}] live",
                                    name_text.view(), saved.extent(0), saved.extent(1), matrix.weights.extent(0),
                                    matrix.weights.extent(1));
        if (CUresult result = cuMemcpyHtoDAsync(device_address(matrix.weights.data()), saved.data(),
                                                saved.numel() * sizeof(nk_bf16_t), stream.get());
            result != CUDA_SUCCESS)
            return cuda_failure(result, "{} upload failed", name_text.view());
    }
    return {};
}

expected_t ut::save_weights(checkpoint_to_save_t &checkpoint, trainer_config_t const &config,
                            st::basic_vector<differentiable_weights_t> const &weights, u32_t steps,
                            gpu_stream_t &stream) noexcept {
    switch_to_gpu_t gpu_scope {stream};
    if (!gpu_scope.ok()) return expected_t::fail(status_t::cuda_error_k, "context entry failed");
    std::span<differentiable_weights_t const> const list {weights};
    u64_t largest_scalars = 0;
    for (differentiable_weights_t const &matrix : list)
        if (matrix.weights.numel() > largest_scalars) largest_scalars = matrix.weights.numel();
    st::basic_vector<nk_bf16_t> staging;
    if (st::failed(staging.resize(st::assume_uninitialized, largest_scalars)))
        return expected_t::fail(status_t::bad_alloc_k, "checkpoint staging failed ({} scalars)", largest_scalars);
    sz::string_t name_text;
    for (differentiable_weights_t const &matrix : list) {
        name_text.clear();
        if (!format_into(name_text, "model.{}", matrix.name))
            return expected_t::fail(status_t::bad_alloc_k, "matrix name format failed ({})", matrix.name);
        std::size_t const rows = matrix.weights.extent(0), cols = matrix.weights.extent(1);
        if (CUresult result = cuMemcpyDtoHAsync(staging.data(), device_address(matrix.weights.data()),
                                                rows * cols * sizeof(nk_bf16_t), stream.get());
            result != CUDA_SUCCESS)
            return cuda_failure(result, "{} download failed", name_text.view());
        if (CUresult result = cuStreamSynchronize(stream.get()); result != CUDA_SUCCESS)
            return cuda_failure(result, "{} download drain failed", name_text.view());
        if (expected_t inserted = checkpoint.insert(name_text.view(), {staging.data(), rows, cols}); !inserted)
            return inserted;
    }

    encoder_tower_config_t const tower = tower_config_(config);
    sz::string_t text; // every value is a string, and lists join their channels with commas
    auto record = [&](sz::string_view_t key, auto value) noexcept {
        text.clear();
        return format_into(text, "{}", value) && static_cast<bool>(checkpoint.metadata(key, text.view()));
    };
    auto record_list = [&](sz::string_view_t key, f32_t const *values, std::size_t count) noexcept {
        text.clear();
        bool formatted = true;
        for (std::size_t index = 0; index != count; ++index)
            formatted = formatted && (index == 0 || format_into(text, ",")) && format_into(text, "{}", values[index]);
        return formatted && static_cast<bool>(checkpoint.metadata(key, text.view()));
    };
    bool const recorded =
        record("step", steps) && record("depth", tower.depth) && record("hidden", tower.block.hidden) &&
        record("ffn_expansion", tower.block.ffn_expansion) && record("head_width", transformer_head_dim_k) &&
        record("rms_epsilon", tower.block.rms_eps) && record("rope_temporal", tower.rope.temporal) &&
        record("rope_height", tower.rope.height) && record("rope_width", tower.rope.width) &&
        record("rope_base", tower.rope_base) && record("context", tower.rope_max_positions) &&
        record("image_patch_side", tower.image_patch_height) && record("image_channels", tower.image_channels) &&
        record_list("image_pixel_mean", tower.image_pixel_mean, tower.image_channels) &&
        record_list("image_pixel_std", tower.image_pixel_std, tower.image_channels) &&
        record("video_patch_side", tower.video_patch_height) && record("video_tube_frames", tower.video_tube_frames) &&
        record("video_frame_stride", tower.video_frame_stride) && record("video_channels", tower.video_channels) &&
        record_list("video_pixel_mean", tower.video_pixel_mean, tower.video_channels) &&
        record_list("video_pixel_std", tower.video_pixel_std, tower.video_channels) &&
        record("audio_chunk_samples", tower.audio_chunk_samples) &&
        record("audio_sample_rate", config.batch.audio.sample_rate);
    if (!recorded) return expected_t::fail(status_t::bad_alloc_k, "checkpoint metadata staging failed");
    return {};
}

expected<trainer_config_t> trainer_config_t::parse(int argc, char **argv) noexcept {
    using result_t = expected<trainer_config_t>;
    trainer_config_t config;
    for (int arg = 1; arg < argc; ++arg) {
        sz::string_view_t const flag {argv[arg], std::strlen(argv[arg])};
        if (flag == "--eager") {
            config.capture = eager_k;
            continue;
        }
        if (arg + 1 == argc) return result_t::fail(status_t::invalid_input_k, "{} takes a value", flag);
        ++arg;
        sz::string_view_t const value {argv[arg], std::strlen(argv[arg])};
        auto number = [&](auto &into) noexcept {
            std::from_chars_result const read = std::from_chars(value.data(), value.data() + value.size(), into);
            return read.ec == std::errc {} && read.ptr == value.data() + value.size();
        };
        bool taken = true;
        if (flag == "--steps") taken = number(config.steps);
        else if (flag == "--hidden") taken = number(config.hidden);
        else if (flag == "--depth") taken = number(config.depth);
        else if (flag == "--rows") {
            // One segment per row, so the segment budget tracks the row budget.
            taken = number(config.batch.max_rows_per_gpu);
            config.batch.max_segments_per_gpu = config.batch.max_rows_per_gpu;
        }
        else if (flag == "--columns") taken = number(config.batch.max_columns_per_row);
        else if (flag == "--learning-rate") taken = number(config.learning_rate) && config.learning_rate > 0.0f;
        else if (flag == "--warmup-steps") taken = number(config.warmup_steps);
        else if (flag == "--seed") taken = number(config.seed);
        else if (flag == "--nvdec-engines") taken = number(config.nvdec_engines);
        else if (flag == "--batches-per-fill") taken = number(config.batches_per_fill) && config.batches_per_fill > 0;
        else if (flag == "--pair-loss") {
            if (value == "balanced") config.contrastive.pair_loss = contrastive_pair_loss_form_t::balanced_k;
            else if (value == "ranked") config.contrastive.pair_loss = contrastive_pair_loss_form_t::ranked_k;
            else if (value == "sigmoid") config.contrastive.pair_loss = contrastive_pair_loss_form_t::sigmoid_k;
            else taken = false;
            // Each form starts at its own logit scale: the sigmoid sum at SigLIP's β=10, b=−10.
            bool const sigmoid = config.contrastive.pair_loss == contrastive_pair_loss_form_t::sigmoid_k;
            config.contrastive.inverse_temperature = sigmoid ? 10.0f : 1.0f;
            config.contrastive.bias = sigmoid ? -10.0f : 0.0f;
        }
        else if (flag == "--late-interaction-weight")
            taken = number(config.late_interaction.weight) && std::isfinite(config.late_interaction.weight) &&
                    config.late_interaction.weight >= 0.0f;
        else if (flag == "--objective") {
            if (value == "masked") config.objective = objective_t::masked_k;
            else if (value == "contrastive") config.objective = objective_t::contrastive_k;
            else if (value == "joint") config.objective = objective_t::joint_k;
            else taken = false;
        }
        else if (flag == "--mixture") {
            if (value.ends_with(".json")) config.mixture = value;
            else if (value == "omni") {
                // The omni preset: 8-pixel RGB image patches, 2-frame RGB video tubes, 256-sample
                // audio frames.
                config.batch.image = image_batch_shape_t {8u, 3u};
                config.batch.video = video_batch_shape_t {8u, 3u, 2u, 1u};
                config.batch.audio = audio_batch_shape_t {256u, 16000u};
            }
            else taken = value == "tokens";
        }
        else if (flag == "--checkpoint") config.checkpoint = value;
        else if (flag == "--checkpoint-every") taken = number(config.checkpoint_every);
        else if (flag == "--report-every") taken = number(config.report_every);
        else if (flag == "--eval-every") taken = number(config.eval_every);
        else if (flag == "--eval-mixture") taken = value.ends_with(".json"), config.eval_mixture = value;
        else if (flag == "--resume") config.resume = value;
        else if (flag == "--telemetry") {
            if (value == "signals") config.telemetry_channels |= telemetry_signals_k;
            else if (value == "timings") config.telemetry_channels |= telemetry_timings_k;
            else taken = false;
        }
        else return result_t::fail(status_t::invalid_input_k, "unknown flag {}", flag);
        if (!taken) return result_t::fail(status_t::invalid_input_k, "{} does not take {}", flag, value);
    }
    return result_t {std::move(config)};
}

expected<base_replica_t> base_replica_t::make(gpu_arena_t &arena, trainer_config_t const &config,
                                              std::size_t vertices_per_mesh) noexcept {
    using result_t = expected<base_replica_t>;
    multimodal_batch_shape_t const &shape = config.batch;
    u32_t const max_tokens = shape.max_rows_per_gpu * shape.max_columns_per_row;
    encoder_tower_config_t const tower_config = tower_config_(config);
    if (tower_config.rope_max_positions > static_cast<u32_t>(std::numeric_limits<i16_t>::max()))
        return result_t::fail(
            status_t::shape_unsupported_k,
            "coordinates reach {} positions, past the i16 ceiling of {} (columns={} tube_frames={} " //
            "frame_stride={})",
            tower_config.rope_max_positions, std::numeric_limits<i16_t>::max(), shape.max_columns_per_row,
            shape.video.tube_frames, shape.video.frame_stride);

    base_replica_t replica;
    expected<encoder_tower_bf16_t> tower = encoder_tower_bf16_t::make(arena, tower_config, shape);
    if (!tower) return result_t::fail(std::move(tower));
    replica.tower = std::move(tower.value_);
    expected<multimodal_batch_on_gpu_t> batch = multimodal_batch_on_gpu_t::make(gpu_arena_alloc_t {arena}, shape);
    if (!batch) return result_t::fail(std::move(batch));
    replica.batch = std::move(batch.value_);
    gpu_arena_alloc<token_id_t> token_alloc {arena};
    gpu_arena_alloc<u32_t> u32_alloc {arena};
    gpu_arena_alloc<f32_t> f32_alloc {arena};
    gpu_arena_alloc<bf16_t> bf16_alloc {arena};

    bool const reconstructs = config.reconstructs();
    bool const image_live = reconstructs && shape.contains(modality_t::image_k);
    bool const video_live = reconstructs && shape.contains(modality_t::video_k);
    bool const audio_live = reconstructs && shape.contains(modality_t::audio_k);
    if (reconstructs) {
        masked_tokens_config_t tokens_config;
        tokens_config.hidden = config.hidden;
        tokens_config.max_tokens = max_tokens;
        tokens_config.weight = config.text_weight;
        expected<masked_tokens_t> tokens = masked_tokens_t::make(arena, tokens_config);
        if (!tokens) return result_t::fail(std::move(tokens));
        replica.tokens = std::move(tokens.value_);
        replica.clean_tokens = tokens_tensor_arena_t::uninitialized({max_tokens}, token_alloc).value;
        replica.token_loss_weights = u32r1_tensor_arena_t::uninitialized({max_tokens}, u32_alloc).value;
        replica.tokens_summands = f32r1_tensor_arena_t::uninitialized({2}, f32_alloc).value;
        replica.normalized_grad = bf16r2_tensor_arena_t::uninitialized({max_tokens, config.hidden}, bf16_alloc).value;
        if (!replica.clean_tokens || !replica.token_loss_weights || !replica.tokens_summands ||
            !replica.normalized_grad)
            return result_t::fail(status_t::bad_alloc_k, "masked scratch allocation failed");
    }

    // Each live media head with its clean target and summands — the same order on every replica.
    if (image_live) {
        masked_patches_config_t head_config;
        head_config.hidden = config.hidden;
        head_config.patch_pixels = patch_scalars(shape.image.patch_side, shape.image.channels);
        head_config.max_patches = max_tokens;
        head_config.mask_ratio = config.image_mask_ratio;
        head_config.weight = config.image_weight;
        head_config.modality = modality_t::image_k;
        expected<masked_patches_t> head = masked_patches_t::make(arena, head_config);
        if (!head) return result_t::fail(std::move(head));
        replica.image = std::move(head.value_);
        replica.clean_image_patches =
            bf16r2_tensor_arena_t::uninitialized({max_tokens, head_config.patch_pixels}, bf16_alloc).value;
        replica.image_summands = f32r1_tensor_arena_t::uninitialized({2}, f32_alloc).value;
        if (!replica.clean_image_patches || !replica.image_summands)
            return result_t::fail(status_t::bad_alloc_k, "image scratch allocation failed");
    }
    if (video_live) {
        masked_patches_config_t head_config;
        head_config.hidden = config.hidden;
        head_config.patch_pixels = tube_scalars(shape.video.tube_frames, shape.video.patch_side, //
                                                shape.video.channels);
        head_config.max_patches = max_tokens;
        head_config.mask_ratio = config.video_mask_ratio;
        head_config.weight = config.video_weight;
        head_config.modality = modality_t::video_k;
        expected<masked_patches_t> head = masked_patches_t::make(arena, head_config);
        if (!head) return result_t::fail(std::move(head));
        replica.video = std::move(head.value_);
        replica.clean_video_patches =
            bf16r2_tensor_arena_t::uninitialized({max_tokens, head_config.patch_pixels}, bf16_alloc).value;
        replica.video_summands = f32r1_tensor_arena_t::uninitialized({2}, f32_alloc).value;
        if (!replica.clean_video_patches || !replica.video_summands)
            return result_t::fail(status_t::bad_alloc_k, "video scratch allocation failed");
    }
    if (audio_live) {
        masked_frames_config_t head_config;
        head_config.hidden = config.hidden;
        head_config.chunk_samples = shape.audio.chunk_samples;
        head_config.max_frames = max_tokens;
        head_config.mask_ratio = config.audio_mask_ratio;
        head_config.weight = config.audio_weight;
        expected<masked_frames_t> head = masked_frames_t::make(arena, head_config);
        if (!head) return result_t::fail(std::move(head));
        replica.audio = std::move(head.value_);
        replica.clean_audio =
            f32r2_tensor_arena_t::uninitialized({max_tokens, shape.audio.chunk_samples}, f32_alloc).value;
        replica.audio_summands = f32r1_tensor_arena_t::uninitialized({2}, f32_alloc).value;
        if (!replica.clean_audio || !replica.audio_summands)
            return result_t::fail(status_t::bad_alloc_k, "audio scratch allocation failed");
    }

    if (config.contrasts()) {
        u32_t const segments = shape.max_segments_per_gpu;
        u32_t const groups = std::max(config.contrastive.temperature_groups, 1u);
        u32_t const edges = config.contrastive.max_edges_per_gpu
                                ? config.contrastive.max_edges_per_gpu
                                : relation_edges_capacity(segments, shape.max_bag_segments);
        replica.vertices = bf16r2_tensor_arena_t::uninitialized({segments, config.hidden}, bf16_alloc).value;
        replica.vertices_grad = bf16r2_tensor_arena_t::uninitialized({segments, config.hidden}, bf16_alloc).value;
        replica.inverse_temperatures = f32r1_tensor_arena_t::uninitialized({groups}, f32_alloc).value;
        replica.biases = f32r1_tensor_arena_t::uninitialized({groups}, f32_alloc).value;
        replica.temperature_groups =
            nk::tensor<temperature_group_t, gpu_arena_alloc<temperature_group_t>, 1>::uninitialized(
                {segments}, gpu_arena_alloc<temperature_group_t> {arena})
                .value;
        replica.to_vertex_lengths = u32r1_tensor_arena_t::uninitialized({vertices_per_mesh}, u32_alloc).value;
        if (!replica.vertices || !replica.vertices_grad || !replica.inverse_temperatures || !replica.biases ||
            !replica.temperature_groups || !replica.to_vertex_lengths)
            return result_t::fail(status_t::bad_alloc_k, "contrastive scratch allocation failed");
        expected<contrastive_relations_t> relations = contrastive_relations_t::make(arena, segments, edges);
        if (!relations) return result_t::fail(std::move(relations));
        replica.relations = std::move(relations.value_);
    }
    if (config.late_interacts()) {
        if (!config.contrasts())
            return result_t::fail(status_t::invalid_input_k, "late interaction reads the contrastive relations");
        u32_t const groups = std::max(config.contrastive.temperature_groups, 1u);
        replica.token_vectors_grad =
            bf16r2_tensor_arena_t::uninitialized({max_tokens, config.late_interaction.width}, bf16_alloc).value;
        replica.token_inverse_temperatures = f32r1_tensor_arena_t::uninitialized({groups}, f32_alloc).value;
        replica.token_biases = f32r1_tensor_arena_t::uninitialized({groups}, f32_alloc).value;
        if (!replica.token_vectors_grad || !replica.token_inverse_temperatures || !replica.token_biases)
            return result_t::fail(status_t::bad_alloc_k, "late-interaction scratch allocation failed");
    }

    // The registration list — tower first, then each live head, identical order on every replica. The tower
    // spells its own weight names; the heads take their instance names here, where the modalities are told apart.
    if (expected_t listed = replica.tower.fill(replica.weights); !listed) return result_t::fail(std::move(listed));
    if (reconstructs)
        if (expected_t listed = replica.tokens.fill("masked_tokens", replica.weights); !listed)
            return result_t::fail(std::move(listed));
    if (image_live)
        if (expected_t listed = replica.image.fill("masked_image", replica.weights); !listed)
            return result_t::fail(std::move(listed));
    if (video_live)
        if (expected_t listed = replica.video.fill("masked_video", replica.weights); !listed)
            return result_t::fail(std::move(listed));
    if (audio_live)
        if (expected_t listed = replica.audio.fill("masked_audio", replica.weights); !listed)
            return result_t::fail(std::move(listed));
    for (f32r1_tensor_arena_t const *pair : replica.summand_pairs_())
        if (!pair->empty())
            if (CUresult result = cuMemsetD8(device_address(pair->data()), 0, pair->numel() * sizeof(f32_t));
                result != CUDA_SUCCESS)
                return cuda_failure<base_replica_t>(result, "loss summands reset failed");
    return result_t {std::move(replica)};
}

expected_t base_replica_t::draw(trainer_config_t const &config, u32_t gpu_index_in_colocated_domain,
                                data_epoch_t data_epoch, gpu_stream_t &stream) noexcept {
    multimodal_batch_shape_t const &shape = config.batch;
    row_partition_t const partition = partition_rows_(config, gpu_index_in_colocated_domain, data_epoch);
    grid_factor_t const factor = square_factor_(shape.max_columns_per_row);
    dim3 const row_grid {shape.max_rows_per_gpu, 1u, 1u};
    dim3 const block_shape {kernel_block_size_default_k, 1u, 1u};
    std::size_t const max_tokens = static_cast<std::size_t>(shape.max_rows_per_gpu) * shape.max_columns_per_row;

    // The keyed byte grid, then every row's segment descriptor from the drawn partition.
    if (expected_t drawn = launch_kernel(&synthetic_tokens_kernel_, row_grid, block_shape, stream, batch.tokens.span(),
                                         config.seed, data_epoch, gpu_index_in_colocated_domain);
        !drawn)
        return drawn;
    if (expected_t laid = launch_kernel(&synthetic_segments_kernel_, row_grid, block_shape, stream,
                                        batch.segment_offsets.span(), batch.segment_lengths.span(),
                                        batch.segment_modality.span(), batch.image_shapes.span(),
                                        batch.video_shapes.span(), shape.max_columns_per_row, partition.begins[1],
                                        partition.begins[2], partition.begins[3], factor.patch_rows, factor.patch_cols);
        !laid)
        return laid;
    // Every synthetic segment queries and is queried with no prefix, so late interaction
    // retrieves each segment's own tokens, as the self-edge relations pair them.
    if (CUresult result = cuMemsetD8Async(device_address(batch.segment_side.data()),
                                          static_cast<u8_t>(late_interaction_side_t::query_and_document_k),
                                          batch.segment_side.numel(), stream.get());
        result != CUDA_SUCCESS)
        return cuda_failure(result, "synthetic side fill failed");
    if (expected_t cleared = zero_async(batch.segment_prefix_length.span(), stream); !cleared) return cleared;

    // Each live modality's raw content — filled across the whole tensor, keyed identically every run.
    if (shape.contains(modality_t::image_k)) {
        u32_t const patch_pixels = patch_scalars(shape.image.patch_side, shape.image.channels);
        nk::tensor_span<pixel_t, 2> const raw = batch.images.span().reshape<2>({max_tokens, patch_pixels});
        dim3 const grid {grid_size_for(raw.extent(0) * raw.extent(1)), 1u, 1u};
        if (expected_t filled = launch_kernel(&synthetic_pixels_kernel_, grid, block_shape, stream, raw, config.seed,
                                              data_epoch, gpu_index_in_colocated_domain, 101u);
            !filled)
            return filled;
    }
    if (shape.contains(modality_t::video_k)) {
        u32_t const patch_pixels = tube_scalars(shape.video.tube_frames, shape.video.patch_side, //
                                                shape.video.channels);
        nk::tensor_span<pixel_t, 2> const raw = batch.videos.span().reshape<2>({max_tokens, patch_pixels});
        dim3 const grid {grid_size_for(raw.extent(0) * raw.extent(1)), 1u, 1u};
        if (expected_t filled = launch_kernel(&synthetic_pixels_kernel_, grid, block_shape, stream, raw, config.seed,
                                              data_epoch, gpu_index_in_colocated_domain, 202u);
            !filled)
            return filled;
    }
    if (shape.contains(modality_t::audio_k)) {
        f32r2_span_t const raw = batch.audio_samples.span().reshape<2>({max_tokens, shape.audio.chunk_samples});
        dim3 const grid {grid_size_for(raw.extent(0) * raw.extent(1)), 1u, 1u};
        if (expected_t filled = launch_kernel(&synthetic_audio_kernel_, grid, block_shape, stream, raw, config.seed,
                                              data_epoch, gpu_index_in_colocated_domain, 303u);
            !filled)
            return filled;
    }
    return {};
}

expected_t base_replica_t::prepare(trainer_config_t const &config, data_epoch_t data_epoch,
                                   gpu_stream_t &stream) noexcept {
    if (tokens_summands.empty()) return {}; // masked heads exist only where make allocated the token head's summands
    if (config.report_every != 0 && data_epoch.ordinal % config.report_every == 0) // the step after a report
        for (f32r1_tensor_arena_t *pair : summand_pairs_())
            if (!pair->empty())
                if (expected_t zeroed = zero_async(pair->span(), stream); !zeroed) return zeroed;
    multimodal_batch_shape_t const &shape = config.batch;
    bool const image_live = image_summands.data() != nullptr;
    bool const video_live = video_summands.data() != nullptr;
    bool const audio_live = audio_summands.data() != nullptr;
    masking_recipe_t const recipe = masking_recipe_(config);
    tokens_span_t const tokens_flat = batch.tokens.span().flatten<1>();
    std::size_t const max_tokens = tokens_flat.extent(0);

    // The token target and its distortion — clean snapshot first, then the keyed span corruption.
    if (expected_t snapshot = copy_device_to_device<token_id_t>(tokens_flat, clean_tokens.span(), stream); !snapshot)
        return snapshot;
    if (expected_t corrupted = tokens.corrupt(clean_tokens.view(), tokens_flat, token_loss_weights.span(),
                                              batch.segment_offsets.view(), batch.segment_lengths.view(),
                                              batch.segment_modality.view(), recipe, data_epoch, stream);
        !corrupted)
        return corrupted;
    if (!image_live && !video_live && !audio_live) return {};

    // The media clean targets — staged from the still-pristine batch, before any raw row is zeroed.
    if (image_live || video_live) {
        bf16r2_span_t const image_target = image_live ? clean_image_patches.span() : bf16r2_span_t {};
        bf16r2_span_t const video_target = video_live ? clean_video_patches.span() : bf16r2_span_t {};
        if (expected_t staged = tower.patchify_media_targets(batch, image_target, video_target, stream); !staged)
            return staged;
    }
    if (audio_live)
        if (expected_t snapshot = copy_device_to_device<f32_t>(batch.audio_samples.span().flatten<1>(),
                                                               clean_audio.span().flatten<1>(), stream);
            !snapshot)
            return snapshot;

    // The media masks and the in-place raw corruption — sampled eagerly so the masks persist into the
    // captured forward region.
    u32r1_view_t const segment_offsets = batch.segment_offsets.view();
    u32r1_view_t const segment_lengths = batch.segment_lengths.view();
    modality_view_t const segment_modality = batch.segment_modality.view();
    if (image_live) {
        if (expected_t sampled = image.sample_mask(recipe, data_epoch, segment_offsets, segment_lengths,
                                                   segment_modality, stream);
            !sampled)
            return sampled;
        u32_t const patch_pixels = patch_scalars(shape.image.patch_side, shape.image.channels);
        nk::tensor_span<pixel_t, 2> const raw = batch.images.span().reshape<2>({max_tokens, patch_pixels});
        if (expected_t corrupted = image.corrupt_raw(raw, stream); !corrupted) return corrupted;
    }
    if (video_live) {
        if (expected_t sampled = video.sample_mask(recipe, data_epoch, segment_offsets, segment_lengths,
                                                   segment_modality, stream);
            !sampled)
            return sampled;
        u32_t const patch_pixels = tube_scalars(shape.video.tube_frames, shape.video.patch_side, //
                                                shape.video.channels);
        nk::tensor_span<pixel_t, 2> const raw = batch.videos.span().reshape<2>({max_tokens, patch_pixels});
        if (expected_t corrupted = video.corrupt_raw(raw, stream); !corrupted) return corrupted;
    }
    if (audio_live) {
        if (expected_t sampled = audio.sample_mask(recipe, data_epoch, segment_offsets, segment_lengths,
                                                   segment_modality, stream);
            !sampled)
            return sampled;
        f32r2_span_t const raw = batch.audio_samples.span().reshape<2>({max_tokens, shape.audio.chunk_samples});
        if (expected_t corrupted = audio.corrupt_raw(raw, stream); !corrupted) return corrupted;
    }
    return {};
}

expected_t base_replica_t::forward_region(gpu_stream_t &stream) noexcept {
    // A head is live iff its summand pair was allocated at make.
    bool const reconstructs = tokens_summands.data() != nullptr;
    bool const image_live = image_summands.data() != nullptr;
    bool const video_live = video_summands.data() != nullptr;
    bool const audio_live = audio_summands.data() != nullptr;

    if (expected_t packed = tower.pack_weights(stream); !packed) return packed;
    if (reconstructs)
        if (expected_t packed = tokens.pack_weights(stream); !packed) return packed;
    if (image_live)
        if (expected_t packed = image.pack_weights(stream); !packed) return packed;
    if (video_live)
        if (expected_t packed = video.pack_weights(stream); !packed) return packed;
    if (audio_live)
        if (expected_t packed = audio.pack_weights(stream); !packed) return packed;
    if (expected_t encoded = tower.forward(batch, vertices.span(), stream); !encoded) return encoded;
    if (!reconstructs) return {};

    // The shared d(normalized): the token head overwrites first, so the reset matters only when media heads add.
    if (image_live || video_live || audio_live)
        if (expected_t reset = zero_async(normalized_grad.span(), stream); !reset) return reset;
    if (expected_t scored = tokens(tower.normalized(), clean_tokens.view(), token_loss_weights.view(),
                                   tokens_summands.span(), normalized_grad.span(), stream);
        !scored)
        return scored;
    if (image_live)
        if (expected_t scored = image(tower.normalized(), clean_image_patches.view(), image_summands.span(),
                                      normalized_grad.span(), stream);
            !scored)
            return scored;
    if (video_live)
        if (expected_t scored = video(tower.normalized(), clean_video_patches.view(), video_summands.span(),
                                      normalized_grad.span(), stream);
            !scored)
            return scored;
    if (audio_live)
        if (expected_t scored = audio(tower.normalized(), clean_audio.view(), audio_summands.span(),
                                      normalized_grad.span(), stream);
            !scored)
            return scored;
    return {};
}

expected_t base_replica_t::backward_region(gpu_stream_t &stream) noexcept {
    return tower.backward(vertices_grad.view(), normalized_grad.view(), batch, stream, token_vectors_grad.view());
}

contrastive_recipe_t base_replica_t::recipe(trainer_config_t const &config) const noexcept {
    contrastive_recipe_t recipe;
    recipe.inverse_temperature = config.contrastive.inverse_temperature;
    recipe.bias = config.contrastive.bias;
    recipe.ambient_repulsion_weight = config.contrastive.ambient_repulsion_weight;
    recipe.inverse_temperature_device = inverse_temperatures.view();
    recipe.bias_device = biases.view();
    recipe.from_vertex_temperature_group = temperature_groups.view();
    return recipe;
}

contrastive_recipe_t base_replica_t::token_recipe(trainer_config_t const &config) const noexcept {
    contrastive_recipe_t recipe;
    recipe.inverse_temperature = config.late_interaction.inverse_temperature;
    recipe.bias = config.late_interaction.bias;
    recipe.ambient_repulsion_weight = config.contrastive.ambient_repulsion_weight;
    recipe.inverse_temperature_device = token_inverse_temperatures.view();
    recipe.bias_device = token_biases.view();
    recipe.from_vertex_temperature_group = temperature_groups.view();
    return recipe;
}
