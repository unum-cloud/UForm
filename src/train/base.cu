/**
 *  @file src/train/base.cu
 *  @author Ash Vardanian
 *  @date July 6, 2026
 *  @brief The implementation of the Base pre-training library declared in @c uform/train/base.cuh :
 *      the synthetic-mixture kernels, the weight seeding and checkpoint helpers, and every
 *      out-of-line body of @ref trainer_config_t and @ref base_replica_t .
 *
 *  No @c main lives here — the binary's thin entry is @c tools/base.cu , and the conformance test
 *  is @c test/base.cu .
 *
 *  @section base_impl_mixture The synthetic mixture, drawn like the real mixer composes
 *
 *  Each step partitions the row budget across the live modalities with a keyed draw — any counts,
 *  no per-modality caps — and assigns the rows in the @ref modality_t order (tokens, image, video,
 *  audio), each row modality-uniform with one segment, so projections tile contiguous row runs. The
 *  partition, the contents, and the corruption are all pure functions of
 *  @c (seed,data_epoch,gpu_index_in_colocated_domain) — the mesh's global world coordinate, so
 *  every world GPU trains on distinct data: every run and the capture-vs-eager parity re-run draw
 *  bitwise-identically, with no files.
 *
 *  @section base_impl_schedule The learning-rate schedule
 *
 *  The rate is the trainer's policy, computed on host per step (legal because the optimizer region
 *  is eager): a linear ramp from zero to the peak over @c warmup_steps , then a cosine decay to
 *  @c min_learning_rate over the run's remainder — the validated draft envelope's shape.
 */
#include <cmath> // `std::cos`, `std::sqrt`, `std::ceil`

#include <unistd.h> // `::write`, `STDOUT_FILENO` — the host diagnostic channel

#include <algorithm> // `std::clamp`
#include <limits>    // `std::numeric_limits` — the i16 coordinate ceiling
#include <numbers>   // `std::numbers::pi_v` — the decay's half-period

#include <cuda.h> // `cuMemcpyHtoDAsync`, `cuMemcpyDtoHAsync`, `cuStreamSynchronize`, `cuMemsetD8`

#include "uform/train/base.cuh"

namespace ue = unum::uform::encoder;
using ue::audio_batch_shape_t;
using ue::byte_vocabulary_k;
using ue::image_batch_shape_t;
using ue::image_shape_span_t;
using ue::late_interaction_side_t;
using ue::modality_span_t;
using ue::modality_t;
using ue::modality_view_t;
using ue::multimodal_batch_on_gpu_t;
using ue::multimodal_batch_shape_t;
using ue::pad_choices;
using ue::patch_scalars;
using ue::tube_scalars;
using ue::video_batch_shape_t;
using ue::video_shape_span_t;
namespace util = unum::uform::utilities;
using namespace unum::uform::train;
namespace ut = unum::uform::train;

/**
 *  @brief Fill the token grid with a keyed synthetic byte corpus — every row, media rows included
 *      (their bytes are simply never read); the draw is a pure function of
 *      @c (seed,data_epoch,gpu_index,row,column) with the @b world GPU coordinate.
 *
 *  @b grid_size @c [rows,1,1] , one CTA per row; no dynamic shared memory.
 */
static __global__ void synthetic_tokens_kernel_(nk::tensor_span<util::token_id_t, 2> tokens, util::u64_t seed,
                                                util::data_epoch_t data_epoch,
                                                util::u32_t gpu_index_in_colocated_domain) {
    util::u32_t const columns = static_cast<util::u32_t>(tokens.extent(1));
    util::u32_t const row = blockIdx.x;
    for (util::u32_t column = threadIdx.x; column < columns; column += blockDim.x) {
        util::u64_t const draw = util::splitmix64_hash_t {}(seed, data_epoch.ordinal, gpu_index_in_colocated_domain,
                                                            row, column);
        tokens[row, column] = static_cast<util::token_id_t>(draw % 256u);
    }
}

/**
 *  @brief Write every row's segment descriptor from the host-drawn partition: rows below
 *      @p image_begin are tokens, then the image / video / audio runs in @ref modality_t order (an
 *      empty run leaves consecutive begins equal).
 *
 *  A token row splits into equal slices, as many as the segment budget gives each row, so the
 *  envelope probe runs at the real batch's density. A media row is one full-width segment with
 *  the near-square @c (patch_rows,patch_cols) aspect the coordinate adapter reads, then PAD.
 *
 *  @b grid_size @c [rows,1,1] , one CTA per row; only thread 0 writes; no dynamic shared memory.
 */
static __global__ void synthetic_segments_kernel_(util::u32r1_span_t segment_offsets,
                                                  util::u32r1_span_t segment_lengths, modality_span_t segment_modality,
                                                  image_shape_span_t image_shapes, video_shape_span_t video_shapes,
                                                  util::u32_t columns, util::u32_t image_begin, util::u32_t video_begin,
                                                  util::u32_t audio_begin, util::u16_t patch_rows,
                                                  util::u16_t patch_cols) {
    util::u32_t const segments = static_cast<util::u32_t>(segment_lengths.extent(0)), rows = gridDim.x;
    util::u32_t const row = blockIdx.x, per_row = segments / rows, first = row * per_row;
    if (threadIdx.x != 0u) return;
    modality_t modality = modality_t::text_k;
    if (row >= audio_begin) modality = modality_t::audio_k;
    else if (row >= video_begin) modality = modality_t::video_k;
    else if (row >= image_begin) modality = modality_t::image_k;
    util::u32_t const live = modality == modality_t::text_k ? per_row : 1u, slice = columns / live;
    for (util::u32_t index = 0; index != per_row; ++index) {
        // The last live slice takes the row's remainder; PAD repeats the row's end with no length.
        util::u32_t const begin = index < live ? index * slice : columns;
        util::u32_t const end = index + 1u == live ? columns : fu::min_of(begin + slice, columns);
        segment_offsets[first + index] = row * columns + begin;
        segment_lengths[first + index] = index < live ? end - begin : 0u;
        segment_modality[first + index] = index < live ? modality : modality_t::text_k;
    }
    if (modality == modality_t::image_k) {
        image_shapes[first, 0] = patch_rows;
        image_shapes[first, 1] = patch_cols;
    }
    if (modality == modality_t::video_k) {
        video_shapes[first, 0] = 1u;
        video_shapes[first, 1] = patch_rows;
        video_shapes[first, 2] = patch_cols;
    }
    if (row != 0u) return;
    // The segments past the rows' shares PAD at the grid's end.
    for (util::u32_t index = rows * per_row; index != segments; ++index) {
        segment_offsets[index] = rows * columns;
        segment_lengths[index] = 0u;
        segment_modality[index] = modality_t::text_k;
    }
    segment_offsets[segments] = rows * columns;
}

/**
 *  @brief Fill one media pixel tensor (flattened @b [rows, patch_scalars] ) with keyed byte noise —
 *      the raw image / video content the tower re-patchifies, identical across runs.
 *
 *  @b grid_size capped grid-stride over @c rows · patch_scalars scalars; no dynamic shared memory.
 */
static __global__ void synthetic_pixels_kernel_(nk::tensor_span<util::pixel_t, 2> media, util::u64_t seed,
                                                util::data_epoch_t data_epoch,
                                                util::u32_t gpu_index_in_colocated_domain, util::u32_t salt) {
    util::u32_t const width = static_cast<util::u32_t>(media.extent(1));
    std::size_t const total = static_cast<std::size_t>(media.extent(0)) * width;
    std::size_t const stride = static_cast<std::size_t>(gridDim.x) * blockDim.x;
    for (std::size_t index = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x; index < total;
         index += stride) {
        util::u32_t const row = static_cast<util::u32_t>(index / width);
        util::u32_t const column = static_cast<util::u32_t>(index % width);
        util::u64_t const draw = util::splitmix64_hash_t {}(seed, data_epoch.ordinal + salt,
                                                            gpu_index_in_colocated_domain, row, column);
        media[row, column] = static_cast<util::pixel_t>(draw & 0xFFu);
    }
}

/**
 *  @brief Fill the audio waveform tensor (flattened @b [frames, chunk_samples] ) with keyed noise
 *      in @b [-1,1] — the raw samples the tower's learnable frontend projects.
 *
 *  @b grid_size capped grid-stride over @c frames · chunk_samples scalars; no
 *      dynamic shared memory.
 */
static __global__ void synthetic_audio_kernel_(util::f32r2_span_t frames, util::u64_t seed,
                                               util::data_epoch_t data_epoch, util::u32_t gpu_index_in_colocated_domain,
                                               util::u32_t salt) {
    util::u32_t const width = static_cast<util::u32_t>(frames.extent(1));
    std::size_t const total = static_cast<std::size_t>(frames.extent(0)) * width;
    std::size_t const stride = static_cast<std::size_t>(gridDim.x) * blockDim.x;
    for (std::size_t index = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x; index < total;
         index += stride) {
        util::u32_t const row = static_cast<util::u32_t>(index / width);
        util::u32_t const column = static_cast<util::u32_t>(index % width);
        util::u64_t const draw = util::splitmix64_hash_t {}(seed, data_epoch.ordinal + salt,
                                                            gpu_index_in_colocated_domain, row, column);
        util::f32_t const uniform = static_cast<util::f32_t>(draw >> 40) * 0x1.0p-24f;
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
    util::u16_t patch_rows = 0, patch_cols = 0;
};

/** The near-square factorization of one media row's @p columns patches. Picks the largest
 *  divisor no greater than the square root as the column count, so the grid stays as square as
 *  the budget allows. */
[[nodiscard]] static grid_factor_t square_factor_(util::u32_t columns) noexcept {
    util::u32_t side = 1u;
    while ((side + 1u) * (side + 1u) <= columns) ++side;
    while (side > 1u && columns % side != 0u) --side;
    return grid_factor_t {static_cast<util::u16_t>(columns / side), static_cast<util::u16_t>(side)};
}

/** Deterministic uniform weight fill matched to ~N(0,std²) variance, identical on every replica —
 *  replica identity is an initialization invariant the conformance tests assert. */
[[nodiscard]] static util::f32_t hash_fill_(std::size_t index, util::u32_t salt, util::f32_t std) noexcept {
    util::u64_t const mixed = util::splitmix64_hash_t {}(salt, 0u, 1u, static_cast<util::u32_t>(index), 0u);
    util::f32_t const uniform = static_cast<util::f32_t>(mixed >> 40) * 0x1.0p-24f;
    return (uniform - 0.5f) * 3.4641016f * std;
}

/** Whether @p matrix projects back into the residual stream: each block's attention
 *  and MLP output. */
[[nodiscard]] static bool writes_residual_(util::differentiable_weights_t const& matrix) noexcept {
    return matrix.name.view().ends_with(".out") || matrix.name.view().ends_with(".down");
}

multimodal_batch_shape_t ut::batch_capacity(multimodal_batch_shape_t shape, std::size_t rows,
                                            util::f64_t segments_per_row) noexcept {
    util::f64_t const per_row = std::clamp(segments_per_row, 1.0, static_cast<util::f64_t>(shape.max_columns_per_row));
    util::u32_t const segments = static_cast<util::u32_t>(std::ceil(static_cast<util::f64_t>(rows) * per_row));
    if (shape.max_segments_per_gpu) // the bags keep their share of the segments, as each needs one
        shape.max_bags_per_gpu = static_cast<util::u32_t>(util::divide_round_up(
            std::size_t {shape.max_bags_per_gpu} * segments, std::size_t {shape.max_segments_per_gpu}));
    shape.max_rows_per_gpu = static_cast<util::u32_t>(rows);
    shape.max_segments_per_gpu = segments;
    return shape;
}

/** How many matrices of @p list project back into the residual stream, splitting one
 *  unit of variance. */
[[nodiscard]] static std::size_t residual_writers_(std::span<util::differentiable_weights_t const> list) noexcept {
    std::size_t residual_writers = 0;
    for (util::differentiable_weights_t const& matrix : list) residual_writers += writes_residual_(matrix) ? 1u : 0u;
    return residual_writers;
}

/** Seed matrix @p weight_index of @p list to its deterministic init, the salt keyed by its
 *  position, so a matrix re-seeded alone matches the one a whole-list seed writes. */
[[nodiscard]] static util::expected_t seed_matrix_(std::span<util::differentiable_weights_t const> list,
                                                   std::size_t weight_index, std::size_t residual_writers,
                                                   util::executor_stream_t& stream) noexcept {
    util::differentiable_weights_t const& matrix = list[weight_index];
    std::size_t const rows = matrix.weights.extent(0), columns = matrix.weights.extent(1);
    // Matrices are [outputs,inputs]; the residual writers split one unit of variance
    // across the stack.
    std::size_t const fan_in = writes_residual_(matrix) ? columns * residual_writers : columns;
    util::f32_t const deviation = 1.0f / std::sqrt(static_cast<util::f32_t>(fan_in));
    nk::tensor<util::bf16_t, util::pinned_alloc<util::bf16_t>, 1> staging;
    if (!util::allocate_uninitialized(staging, {rows * columns}, util::pinned_alloc<util::bf16_t> {}))
        return util::expected_t::fail(util::status_code_t::out_of_memory_k, "weight staging failed (matrix={})",
                                      weight_index);
    util::bf16r1_span_t const host = staging.span();
    for (std::size_t index = 0; index != host.extent(0); ++index)
        host[index] = __float2bfloat16( //
            rows == 1 ? 1.0f : hash_fill_(index, static_cast<util::u32_t>(weight_index) + 11u, deviation));
    if (util::expected_t copied = util::copy_host_to_device(staging.view(), matrix.weights.flatten<1>(), stream);
        !copied)
        return copied;
    // The staging tensor dies with this call — the enqueued copy must land first.
    cuStreamSynchronize(util::native_stream(stream));
    return {};
}

util::expected_t ut::seed_weights(st::basic_vector<util::differentiable_weights_t> const& weights,
                                  util::executor_stream_t& stream) noexcept {
    std::span<util::differentiable_weights_t const> const list {weights};
    std::size_t const residual_writers = residual_writers_(list);
    for (std::size_t weight_index = 0; weight_index != list.size(); ++weight_index)
        if (util::expected_t seeded = seed_matrix_(list, weight_index, residual_writers, stream); !seeded)
            return seeded;
    return {};
}

util::f32_t trainer_config_t::learning_rate_for(util::optimizer_step_t optimizer_step) const noexcept {
    util::u32_t const warmup = warmup_steps ? warmup_steps : fu::min_of(400u, steps / 10u);
    if (optimizer_step.ordinal < warmup)
        return learning_rate * static_cast<util::f32_t>(optimizer_step.ordinal + 1u) / static_cast<util::f32_t>(warmup);
    util::u32_t const decay_steps = steps > warmup ? steps - warmup : 1u;
    util::f32_t const progress = static_cast<util::f32_t>(optimizer_step.ordinal - warmup) /
                                 static_cast<util::f32_t>(decay_steps);
    util::f32_t const cosine = 0.5f * (1.0f + std::cos(std::numbers::pi_v<util::f32_t> * fu::min_of(progress, 1.0f)));
    return min_learning_rate + (learning_rate - min_learning_rate) * cosine;
}

/** The masking corruptor's keyed recipe — the run's knobs, its seed derived once from
 *  the run seed. */
[[nodiscard]] static masking_recipe_t masking_recipe_(trainer_config_t const& config) noexcept {
    masking_recipe_t recipe = config.masking;
    recipe.seed = config.seed * 2654435761u + 17u;
    return recipe;
}

/** One step's keyed row partition: the live modalities in @ref modality_t order, each with its row
 *  count and the first row of its contiguous run. */
struct row_partition_t {

    /** Rows per modality, indexed by @ref modality_t — zero for absent modalities. */
    util::u32_t rows[4] = {};

    /** The first row of each modality's run — prefix sums of @ref rows in modality order. */
    util::u32_t begins[4] = {};
};

/** Partition the row budget across the live modalities — any counts, no per-modality caps, exactly
 *  the freedom the real mixer has. Every live modality gets the floor share; the keyed draw rotates
 *  which ones take the remainder, so every mixture recurs across passes (and a one-row omni batch
 *  deterministically leaves most heads with zero live rows — the dynamic-M edge). */
[[nodiscard]] static row_partition_t partition_rows_(trainer_config_t const& config,
                                                     util::u32_t gpu_index_in_colocated_domain,
                                                     util::data_epoch_t data_epoch) noexcept {
    multimodal_batch_shape_t const& shape = config.batch;
    bool const live[4] = {true, shape.contains(modality_t::image_k), shape.contains(modality_t::video_k),
                          shape.contains(modality_t::audio_k)};
    util::u32_t live_count = 0;
    for (bool flag : live) live_count += flag ? 1u : 0u;
    util::u32_t const total = shape.max_rows_per_gpu;
    util::u32_t const floor_rows = total / live_count, extra = total % live_count;
    util::u64_t const draw = util::splitmix64_hash_t {}(config.seed, data_epoch.ordinal, gpu_index_in_colocated_domain,
                                                        5u, 0u);
    util::u32_t const rotation = static_cast<util::u32_t>(draw % live_count);

    row_partition_t partition;
    util::u32_t position = 0;
    for (std::size_t modality = 0; modality != 4; ++modality) {
        if (!live[modality]) continue;
        util::u32_t const bonus = ((position + live_count - rotation) % live_count) < extra ? 1u : 0u;
        partition.rows[modality] = floor_rows + bonus;
        ++position;
    }
    util::u32_t cursor = 0;
    for (std::size_t modality = 0; modality != 4; ++modality) {
        partition.begins[modality] = cursor;
        cursor += partition.rows[modality];
    }
    return partition;
}

/** Fold the run description into the tower's make-config: the block geometry from the batch
 *  envelope, one media entry per live batch slice — the batch shape stays the single source of
 *  modality presence. The rotary tables cover the largest coordinate a row reaches: its column
 *  count, or the absolute source frame the last tube of a one-patch clip starts at. */
[[nodiscard]] static encoder_tower_config_t tower_config_(trainer_config_t const& config) noexcept {
    multimodal_batch_shape_t const& batch = config.batch;
    util::u32_t const max_tokens = batch.max_rows_per_gpu * batch.max_columns_per_row;
    encoder_tower_config_t tower;
    tower.vocabulary = byte_vocabulary_k;
    tower.depth = config.depth;
    tower.block.hidden = config.hidden;
    tower.block.max_tokens = max_tokens;
    tower.block.max_segments = batch.max_segments_per_gpu;
    tower.block.max_segment_tokens = batch.max_columns_per_row;
    util::u64_t positions = batch.max_columns_per_row;
    if (batch.contains(modality_t::image_k)) {
        tower.image_patch_height = batch.image.patch_side;
        tower.image_patch_width = batch.image.patch_side;
        tower.image_channels = batch.image.channels;
    }
    if (batch.contains(modality_t::video_k)) {
        util::u32_t const frame_stride = fu::max_of(batch.video.frame_stride, 1u);
        tower.video_tube_frames = batch.video.tube_frames;
        tower.video_frame_stride = frame_stride;
        tower.video_patch_height = batch.video.patch_side;
        tower.video_patch_width = batch.video.patch_side;
        tower.video_channels = batch.video.channels;
        util::u64_t const last_tube =
            static_cast<util::u64_t>(batch.max_columns_per_row - 1u) * batch.video.tube_frames * frame_stride + 1u;
        positions = fu::max_of(positions, last_tube);
    }
    if (batch.contains(modality_t::audio_k)) tower.audio_chunk_samples = batch.audio.chunk_samples;
    tower.rope_max_positions = static_cast<util::u32_t>(
        fu::min_of<util::u64_t>(positions, (std::numeric_limits<util::u32_t>::max)()));
    tower.token_width = config.late_interacts() ? config.late_interaction.width : 0u;
    return tower;
}

util::expected_t ut::load_weights(util::checkpoint_to_load_t const& checkpoint,
                                  st::basic_vector<util::differentiable_weights_t> const& weights,
                                  util::executor_stream_t& stream) noexcept {
    std::span<util::differentiable_weights_t const> const list {weights};
    std::size_t const residual_writers = residual_writers_(list);
    sz::string_t name_text;
    for (std::size_t weight_index = 0; weight_index != list.size(); ++weight_index) {
        util::differentiable_weights_t const& matrix = list[weight_index];
        name_text.clear();
        if (!util::format_into(name_text, "model.{}", matrix.name))
            return util::expected_t::fail(util::status_code_t::out_of_memory_k, "matrix name format failed ({})",
                                          matrix.name);
        nk::tensor_view<nk_bf16_t, 2> const saved = checkpoint[name_text.view()];
        // A base trained without the decoder or the token head warm-starts a run adding either.
        bool const optional = matrix.name.view() == "masked_tokens" || matrix.name.view().starts_with("token_head.");
        if (saved.empty() && optional) {
            if (util::expected_t seeded = seed_matrix_(list, weight_index, residual_writers, stream); !seeded)
                return seeded;
            continue;
        }
        if (saved.empty())
            return util::expected_t::fail(util::status_code_t::invalid_argument_k, "the checkpoint misses {}",
                                          name_text.view());
        if (saved.extent(0) != matrix.weights.extent(0) || saved.extent(1) != matrix.weights.extent(1))
            return util::expected_t::fail(
                util::status_code_t::unsupported_k, "{} is [{},{}] in the checkpoint, [{},{}] live", name_text.view(),
                saved.extent(0), saved.extent(1), matrix.weights.extent(0), matrix.weights.extent(1));
        if (CUresult result = cuMemcpyHtoDAsync(util::device_address(matrix.weights.data()), saved.data(),
                                                saved.numel() * sizeof(nk_bf16_t), util::native_stream(stream));
            result != CUDA_SUCCESS)
            return util::cuda_failure(result, "{} upload failed", name_text.view());
    }
    return {};
}

util::expected_t ut::save_weights(util::checkpoint_to_save_t& checkpoint, trainer_config_t const& config,
                                  st::basic_vector<util::differentiable_weights_t> const& weights, util::u32_t steps,
                                  util::executor_stream_t& stream) noexcept {
    [[maybe_unused]] util::status_t const gpu_scope = (stream).bind();
    if (gpu_scope.failed()) return util::expected_t::fail(util::status_code_t::device_k, "Context entry failed");
    std::span<util::differentiable_weights_t const> const list {weights};
    util::u64_t largest_scalars = 0;
    for (util::differentiable_weights_t const& matrix : list)
        if (matrix.weights.numel() > largest_scalars) largest_scalars = matrix.weights.numel();
    st::basic_vector<nk_bf16_t> staging;
    if (st::failed(staging.resize(st::assume_uninitialized, largest_scalars)))
        return util::expected_t::fail(util::status_code_t::out_of_memory_k, "checkpoint staging failed ({} scalars)",
                                      largest_scalars);
    sz::string_t name_text;
    for (util::differentiable_weights_t const& matrix : list) {
        name_text.clear();
        if (!util::format_into(name_text, "model.{}", matrix.name))
            return util::expected_t::fail(util::status_code_t::out_of_memory_k, "matrix name format failed ({})",
                                          matrix.name);
        std::size_t const rows = matrix.weights.extent(0), cols = matrix.weights.extent(1);
        if (CUresult result = cuMemcpyDtoHAsync(staging.data(), util::device_address(matrix.weights.data()),
                                                rows * cols * sizeof(nk_bf16_t), util::native_stream(stream));
            result != CUDA_SUCCESS)
            return util::cuda_failure(result, "{} download failed", name_text.view());
        if (CUresult result = cuStreamSynchronize(util::native_stream(stream)); result != CUDA_SUCCESS)
            return util::cuda_failure(result, "{} download drain failed", name_text.view());
        if (util::expected_t inserted = checkpoint.insert(name_text.view(), {staging.data(), rows, cols}); !inserted)
            return inserted;
    }

    encoder_tower_config_t const tower = tower_config_(config);
    sz::string_t text; // every value is a string, and lists join their channels with commas
    auto record = [&](sz::string_view_t key, auto value) noexcept {
        text.clear();
        return util::format_into(text, "{}", value) && static_cast<bool>(checkpoint.metadata(key, text.view()));
    };
    auto record_list = [&](sz::string_view_t key, std::span<util::f32_t const> values) noexcept {
        text.clear();
        bool formatted = true;
        for (std::size_t index = 0; index != values.size(); ++index)
            formatted = formatted && (index == 0 || util::format_into(text, ",")) &&
                        util::format_into(text, "{}", values[index]);
        return formatted && static_cast<bool>(checkpoint.metadata(key, text.view()));
    };
    bool const recorded =
        record("step", steps) && record("depth", tower.depth) && record("hidden", tower.block.hidden) &&
        record("ffn_expansion", tower.block.ffn_expansion) && record("head_width", transformer_head_dim_k) &&
        record("rms_epsilon", tower.block.rms_eps) && record("rope_temporal", tower.rope.temporal) &&
        record("rope_height", tower.rope.height) && record("rope_width", tower.rope.width) &&
        record("rope_base", tower.rope_base) && record("context", tower.rope_max_positions) &&
        record("columns", config.batch.max_columns_per_row) && record("image_patch_side", tower.image_patch_height) &&
        record("image_channels", tower.image_channels) &&
        record_list("image_pixel_mean", {tower.image_pixel_mean, tower.image_channels}) &&
        record_list("image_pixel_std", {tower.image_pixel_std, tower.image_channels}) &&
        record("video_patch_side", tower.video_patch_height) && record("video_tube_frames", tower.video_tube_frames) &&
        record("video_frame_stride", tower.video_frame_stride) && record("video_channels", tower.video_channels) &&
        record_list("video_pixel_mean", {tower.video_pixel_mean, tower.video_channels}) &&
        record_list("video_pixel_std", {tower.video_pixel_std, tower.video_channels}) &&
        record("audio_chunk_samples", tower.audio_chunk_samples) &&
        record("audio_sample_rate", config.batch.audio.sample_rate);
    if (!recorded)
        return util::expected_t::fail(util::status_code_t::out_of_memory_k, "Checkpoint metadata staging failed");
    return {};
}

util::expected<base_replica_t> base_replica_t::make(util::gpu_arena_t& arena, trainer_config_t const& config,
                                                    std::size_t vertices_per_mesh,
                                                    util::gpu_numerics_t const& numerics) noexcept {
    using result_t = util::expected<base_replica_t>;
    multimodal_batch_shape_t const& shape = config.batch;
    util::u32_t const max_tokens = shape.max_rows_per_gpu * shape.max_columns_per_row;
    encoder_tower_config_t const tower_config = tower_config_(config);
    if (tower_config.rope_max_positions > static_cast<util::u32_t>((std::numeric_limits<util::i16_t>::max)()))
        return result_t::fail(
            util::status_code_t::unsupported_k,
            "coordinates reach {} positions, past the i16 ceiling of {} (columns={} tube_frames={} " //
            "frame_stride={})",
            tower_config.rope_max_positions, (std::numeric_limits<util::i16_t>::max)(), shape.max_columns_per_row,
            shape.video.tube_frames, shape.video.frame_stride);

    base_replica_t replica;
    auto [tower, make_tower_status] = encoder_tower_bf16_t::make(arena, tower_config, shape, numerics);
    if (!make_tower_status) return result_t::fail(std::move(make_tower_status));
    replica.tower = std::move(tower);
    auto [batch, make_batch_status] = multimodal_batch_on_gpu_t::make(util::gpu_arena_alloc_t {arena}, shape);
    if (!make_batch_status) return result_t::fail(std::move(make_batch_status));
    replica.batch = std::move(batch);
    util::gpu_arena_alloc<util::token_id_t> token_alloc {arena};
    util::gpu_arena_alloc<util::u32_t> u32_alloc {arena};
    util::gpu_arena_alloc<util::f32_t> f32_alloc {arena};
    util::gpu_arena_alloc<util::bf16_t> bf16_alloc {arena};

    bool const reconstructs = config.reconstructs(), chooses = config.chooses();
    bool const image_live = reconstructs && shape.contains(modality_t::image_k);
    bool const video_live = reconstructs && shape.contains(modality_t::video_k);
    bool const audio_live = reconstructs && shape.contains(modality_t::audio_k);
    if (chooses && shape.max_choices_per_segment == 0)
        return result_t::fail(util::status_code_t::invalid_argument_k,
                              "Choices need the batch's max_choices_per_segment");
    if (reconstructs || chooses) {
        masked_tokens_config_t tokens_config;
        tokens_config.hidden = config.hidden;
        tokens_config.max_tokens = max_tokens;
        tokens_config.weight = config.text_weight;
        tokens_config.max_segments = chooses ? shape.max_segments_per_gpu : 0u;
        auto [tokens, make_tokens_status] = masked_tokens_t::make(arena, tokens_config, numerics);
        if (!make_tokens_status) return result_t::fail(std::move(make_tokens_status));
        replica.tokens = std::move(tokens);
        if (!util::allocate_uninitialized(replica.normalized_grad, {max_tokens, config.hidden}, bf16_alloc))
            return result_t::fail(util::status_code_t::out_of_memory_k, "No room for the decoder scratch");
    }
    if (reconstructs) {
        bool const allocated = util::allocate_uninitialized(replica.clean_tokens, {max_tokens}, token_alloc) &&
                               util::allocate_uninitialized(replica.token_loss_weights, {max_tokens}, u32_alloc) &&
                               util::allocate_uninitialized(replica.tokens_summands, {2}, f32_alloc);
        if (!allocated) return result_t::fail(util::status_code_t::out_of_memory_k, "No room for the masked scratch");
    }
    if (chooses) {
        if (!util::allocate_uninitialized(replica.choices_summands, {2}, f32_alloc))
            return result_t::fail(util::status_code_t::out_of_memory_k, "No room for the choices scratch");
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
        auto [head, make_head_status] = masked_patches_t::make(arena, head_config, numerics);
        if (!make_head_status) return result_t::fail(std::move(make_head_status));
        replica.image = std::move(head);
        bool const allocated = util::allocate_uninitialized(replica.clean_image_patches,
                                                            {max_tokens, head_config.patch_pixels}, bf16_alloc) &&
                               util::allocate_uninitialized(replica.image_summands, {2}, f32_alloc);
        if (!allocated) return result_t::fail(util::status_code_t::out_of_memory_k, "No room for the image scratch");
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
        auto [head, make_head_status] = masked_patches_t::make(arena, head_config, numerics);
        if (!make_head_status) return result_t::fail(std::move(make_head_status));
        replica.video = std::move(head);
        bool const allocated = util::allocate_uninitialized(replica.clean_video_patches,
                                                            {max_tokens, head_config.patch_pixels}, bf16_alloc) &&
                               util::allocate_uninitialized(replica.video_summands, {2}, f32_alloc);
        if (!allocated) return result_t::fail(util::status_code_t::out_of_memory_k, "No room for the video scratch");
    }
    if (audio_live) {
        masked_frames_config_t head_config;
        head_config.hidden = config.hidden;
        head_config.chunk_samples = shape.audio.chunk_samples;
        head_config.max_frames = max_tokens;
        head_config.mask_ratio = config.audio_mask_ratio;
        head_config.weight = config.audio_weight;
        auto [head, make_head_status] = masked_frames_t::make(arena, head_config, numerics);
        if (!make_head_status) return result_t::fail(std::move(make_head_status));
        replica.audio = std::move(head);
        bool const allocated = util::allocate_uninitialized(replica.clean_audio,
                                                            {max_tokens, shape.audio.chunk_samples}, f32_alloc) &&
                               util::allocate_uninitialized(replica.audio_summands, {2}, f32_alloc);
        if (!allocated) return result_t::fail(util::status_code_t::out_of_memory_k, "No room for the audio scratch");
    }

    if (config.contrasts()) {
        util::u32_t const segments = shape.max_segments_per_gpu;
        util::u32_t const groups = fu::max_of(config.contrastive.temperature_groups, 1u);
        util::u32_t const edges = config.contrastive.max_edges_per_gpu
                                      ? config.contrastive.max_edges_per_gpu
                                      : relation_edges_capacity(segments, shape.max_bag_segments);
        bool const allocated = util::allocate_uninitialized(replica.vertices, {segments, config.hidden}, bf16_alloc) &&
                               util::allocate_uninitialized(replica.vertices_grad, {segments, config.hidden},
                                                            bf16_alloc) &&
                               util::allocate_uninitialized(replica.inverse_temperatures, {groups}, f32_alloc) &&
                               util::allocate_uninitialized(replica.biases, {groups}, f32_alloc) &&
                               util::allocate_uninitialized(replica.temperature_groups, {segments},
                                                            util::gpu_arena_alloc<temperature_group_t> {arena}) &&
                               util::allocate_uninitialized(replica.to_vertex_lengths, {vertices_per_mesh}, u32_alloc);
        if (!allocated)
            return result_t::fail(util::status_code_t::out_of_memory_k, "No room for the contrastive scratch");
        auto [relations, make_relations_status] = contrastive_relations_t::make(arena, segments, edges);
        if (!make_relations_status) return result_t::fail(std::move(make_relations_status));
        replica.relations = std::move(relations);
    }
    if (config.late_interacts()) {
        if (!config.contrasts())
            return result_t::fail(util::status_code_t::invalid_argument_k,
                                  "Late interaction reads the contrastive relations");
        util::u32_t const groups = fu::max_of(config.contrastive.temperature_groups, 1u);
        bool const allocated = util::allocate_uninitialized(replica.token_vectors_grad,
                                                            {max_tokens, config.late_interaction.width}, bf16_alloc) &&
                               util::allocate_uninitialized(replica.token_inverse_temperatures, {groups}, f32_alloc) &&
                               util::allocate_uninitialized(replica.token_biases, {groups}, f32_alloc);
        if (!allocated)
            return result_t::fail(util::status_code_t::out_of_memory_k, "No room for the late-interaction scratch");
    }

    // The registration list — tower first, then each live head, identical order on every replica.
    // The tower spells its own weight names; the heads take their instance names here, where the
    // modalities are told apart.
    if (util::expected_t listed = replica.tower.fill(replica.weights); !listed)
        return result_t::fail(std::move(listed));
    if (reconstructs || chooses)
        if (util::expected_t listed = replica.tokens.fill("masked_tokens", replica.weights); !listed)
            return result_t::fail(std::move(listed));
    if (image_live)
        if (util::expected_t listed = replica.image.fill("masked_image", replica.weights); !listed)
            return result_t::fail(std::move(listed));
    if (video_live)
        if (util::expected_t listed = replica.video.fill("masked_video", replica.weights); !listed)
            return result_t::fail(std::move(listed));
    if (audio_live)
        if (util::expected_t listed = replica.audio.fill("masked_audio", replica.weights); !listed)
            return result_t::fail(std::move(listed));
    for (util::f32r1_tensor_arena_t const* pair : replica.summand_pairs_())
        if (!pair->empty())
            if (CUresult result = cuMemsetD8(util::device_address(pair->data()), 0,
                                             pair->numel() * sizeof(util::f32_t));
                result != CUDA_SUCCESS)
                return util::cuda_failure<base_replica_t>(result, "loss summands reset failed");
    if (CUresult result = cuMemsetD8(util::device_address(replica.batch.dropped.data()), 0,
                                     replica.batch.dropped.numel() * sizeof(util::u32_t));
        result != CUDA_SUCCESS)
        return util::cuda_failure<base_replica_t>(result, "drop count reset failed");
    // The resets ran on the legacy stream, which the replica's non-blocking streams never wait on.
    if (CUresult result = cuStreamSynchronize(CU_STREAM_LEGACY); result != CUDA_SUCCESS)
        return util::cuda_failure<base_replica_t>(result, "loss summands reset drain failed");
    return result_t {std::move(replica)};
}

util::expected_t base_replica_t::draw(trainer_config_t const& config, util::u32_t gpu_index_in_colocated_domain,
                                      util::data_epoch_t data_epoch, util::executor_stream_t& stream) noexcept {
    multimodal_batch_shape_t const& shape = config.batch;
    row_partition_t const partition = partition_rows_(config, gpu_index_in_colocated_domain, data_epoch);
    grid_factor_t const factor = square_factor_(shape.max_columns_per_row);
    dim3 const row_grid {shape.max_rows_per_gpu, 1u, 1u};
    dim3 const block_shape {util::kernel_block_size_default_k, 1u, 1u};
    std::size_t const max_tokens = static_cast<std::size_t>(shape.max_rows_per_gpu) * shape.max_columns_per_row;

    // The keyed byte grid, then every row's segment descriptor from the drawn partition.
    if (util::expected_t drawn = util::launch_kernel(&synthetic_tokens_kernel_, row_grid, block_shape, stream,
                                                     batch.tokens.span(), config.seed, data_epoch,
                                                     gpu_index_in_colocated_domain);
        !drawn)
        return drawn;
    if (util::expected_t laid = util::launch_kernel(
            &synthetic_segments_kernel_, row_grid, block_shape, stream, batch.segment_offsets.span(),
            batch.segment_lengths.span(), batch.segment_modality.span(), batch.image_shapes.span(),
            batch.video_shapes.span(), shape.max_columns_per_row, partition.begins[1], partition.begins[2],
            partition.begins[3], factor.patch_rows, factor.patch_cols);
        !laid)
        return laid;
    // Every synthetic segment queries and is queried with no prefix, so late interaction
    // retrieves each segment's own tokens, as the self-edge relations pair them.
    if (CUresult result = cuMemsetD8Async(util::device_address(batch.segment_side.data()),
                                          static_cast<util::u8_t>(late_interaction_side_t::query_and_document_k),
                                          batch.segment_side.numel(), util::native_stream(stream));
        result != CUDA_SUCCESS)
        return util::cuda_failure(result, "synthetic side fill failed");
    if (util::expected_t cleared = util::zero_async(batch.segment_prefix_length.span(), stream); !cleared)
        return cleared;
    // No synthetic segment makes a choice.
    if (util::expected_t padded = pad_choices(batch, stream); !padded) return padded;

    // Each live modality's raw content — filled across the whole tensor, keyed
    // identically every run.
    if (shape.contains(modality_t::image_k)) {
        util::u32_t const patch_pixels = patch_scalars(shape.image.patch_side, shape.image.channels);
        nk::tensor_span<util::pixel_t, 2> const raw = batch.images.span().reshape<2>({max_tokens, patch_pixels});
        dim3 const grid {util::grid_size_for(raw.extent(0) * raw.extent(1)), 1u, 1u};
        if (util::expected_t filled = util::launch_kernel(&synthetic_pixels_kernel_, grid, block_shape, stream, raw,
                                                          config.seed, data_epoch, gpu_index_in_colocated_domain, 101u);
            !filled)
            return filled;
    }
    if (shape.contains(modality_t::video_k)) {
        util::u32_t const patch_pixels = tube_scalars(shape.video.tube_frames, shape.video.patch_side, //
                                                      shape.video.channels);
        nk::tensor_span<util::pixel_t, 2> const raw = batch.videos.span().reshape<2>({max_tokens, patch_pixels});
        dim3 const grid {util::grid_size_for(raw.extent(0) * raw.extent(1)), 1u, 1u};
        if (util::expected_t filled = util::launch_kernel(&synthetic_pixels_kernel_, grid, block_shape, stream, raw,
                                                          config.seed, data_epoch, gpu_index_in_colocated_domain, 202u);
            !filled)
            return filled;
    }
    if (shape.contains(modality_t::audio_k)) {
        util::f32r2_span_t const raw = batch.audio_samples.span().reshape<2>({max_tokens, shape.audio.chunk_samples});
        dim3 const grid {util::grid_size_for(raw.extent(0) * raw.extent(1)), 1u, 1u};
        if (util::expected_t filled = util::launch_kernel(&synthetic_audio_kernel_, grid, block_shape, stream, raw,
                                                          config.seed, data_epoch, gpu_index_in_colocated_domain, 303u);
            !filled)
            return filled;
    }
    return {};
}

util::expected_t base_replica_t::prepare(trainer_config_t const& config, util::data_epoch_t data_epoch,
                                         util::executor_stream_t& stream) noexcept {
    if (config.report_every != 0 && data_epoch.ordinal % config.report_every == 0) // the step after a report
        for (util::f32r1_tensor_arena_t* pair : summand_pairs_()) {
            if (pair->empty()) continue;
            if (util::expected_t zeroed = util::zero_async(pair->span(), stream); !zeroed) return zeroed;
        }
    if (tokens_summands.empty()) return {}; // masked heads exist only where make allocated the token head's summands
    multimodal_batch_shape_t const& shape = config.batch;
    bool const image_live = image_summands.data() != nullptr;
    bool const video_live = video_summands.data() != nullptr;
    bool const audio_live = audio_summands.data() != nullptr;
    masking_recipe_t const recipe = masking_recipe_(config);
    util::tokens_span_t const tokens_flat = batch.tokens.span().flatten<1>();
    std::size_t const max_tokens = tokens_flat.extent(0);

    // The token target and its distortion — clean snapshot first, then the keyed span corruption.
    if (util::expected_t snapshot = util::copy_device_to_device<util::token_id_t>(tokens_flat, clean_tokens.span(),
                                                                                  stream);
        !snapshot)
        return snapshot;
    if (util::expected_t corrupted = tokens.corrupt(clean_tokens.view(), tokens_flat, token_loss_weights.span(),
                                                    batch.segment_offsets.view(), batch.segment_lengths.view(),
                                                    batch.segment_modality.view(), batch.segment_choices.view(), recipe,
                                                    data_epoch, stream);
        !corrupted)
        return corrupted;
    if (!image_live && !video_live && !audio_live) return {};

    // The media clean targets — staged from the still-pristine batch, before any raw row is zeroed.
    if (image_live || video_live) {
        util::bf16r2_span_t const image_target = image_live ? clean_image_patches.span() : util::bf16r2_span_t {};
        util::bf16r2_span_t const video_target = video_live ? clean_video_patches.span() : util::bf16r2_span_t {};
        if (util::expected_t staged = tower.patchify_media_targets(batch, image_target, video_target, stream); !staged)
            return staged;
    }
    if (audio_live)
        if (util::expected_t snapshot = util::copy_device_to_device<util::f32_t>(
                batch.audio_samples.span().flatten<1>(), clean_audio.span().flatten<1>(), stream);
            !snapshot)
            return snapshot;

    // The media masks and the in-place raw corruption — sampled eagerly so the masks persist into
    // the captured forward region.
    util::u32r1_view_t const segment_offsets = batch.segment_offsets.view();
    util::u32r1_view_t const segment_lengths = batch.segment_lengths.view();
    modality_view_t const segment_modality = batch.segment_modality.view();
    if (image_live) {
        if (util::expected_t sampled = image.sample_mask(recipe, data_epoch, segment_offsets, segment_lengths,
                                                         segment_modality, stream);
            !sampled)
            return sampled;
        util::u32_t const patch_pixels = patch_scalars(shape.image.patch_side, shape.image.channels);
        nk::tensor_span<util::pixel_t, 2> const raw = batch.images.span().reshape<2>({max_tokens, patch_pixels});
        if (util::expected_t corrupted = image.corrupt_raw(raw, stream); !corrupted) return corrupted;
    }
    if (video_live) {
        if (util::expected_t sampled = video.sample_mask(recipe, data_epoch, segment_offsets, segment_lengths,
                                                         segment_modality, stream);
            !sampled)
            return sampled;
        util::u32_t const patch_pixels = tube_scalars(shape.video.tube_frames, shape.video.patch_side, //
                                                      shape.video.channels);
        nk::tensor_span<util::pixel_t, 2> const raw = batch.videos.span().reshape<2>({max_tokens, patch_pixels});
        if (util::expected_t corrupted = video.corrupt_raw(raw, stream); !corrupted) return corrupted;
    }
    if (audio_live) {
        if (util::expected_t sampled = audio.sample_mask(recipe, data_epoch, segment_offsets, segment_lengths,
                                                         segment_modality, stream);
            !sampled)
            return sampled;
        util::f32r2_span_t const raw = batch.audio_samples.span().reshape<2>({max_tokens, shape.audio.chunk_samples});
        if (util::expected_t corrupted = audio.corrupt_raw(raw, stream); !corrupted) return corrupted;
    }
    return {};
}

util::expected_t base_replica_t::forward_region(trainer_config_t const& config,
                                                util::executor_stream_t& stream) noexcept {
    // A head is live iff its summand pair was allocated at make.
    bool const reconstructs = tokens_summands.data() != nullptr;
    bool const chooses = choices_summands.data() != nullptr;
    bool const image_live = image_summands.data() != nullptr;
    bool const video_live = video_summands.data() != nullptr;
    bool const audio_live = audio_summands.data() != nullptr;

    if (util::expected_t packed = tower.pack_weights(stream); !packed) return packed;
    if (reconstructs || chooses)
        if (util::expected_t packed = tokens.pack_weights(stream); !packed) return packed;
    if (image_live)
        if (util::expected_t packed = image.pack_weights(stream); !packed) return packed;
    if (video_live)
        if (util::expected_t packed = video.pack_weights(stream); !packed) return packed;
    if (audio_live)
        if (util::expected_t packed = audio.pack_weights(stream); !packed) return packed;
    if (util::expected_t encoded = tower.forward(batch, vertices.span(), stream); !encoded) return encoded;
    if (!reconstructs && !chooses) return {};

    // The shared d(normalized) and the decoder's gradient: the token head overwrites both first, so
    // the reset matters only when it does not run and the others add.
    if (!reconstructs || image_live || video_live || audio_live)
        if (util::expected_t reset = util::zero_async(normalized_grad.span(), stream); !reset) return reset;
    if (!reconstructs)
        if (util::expected_t reset = util::zero_async(tokens.decoder_gradients(), stream); !reset) return reset;
    if (reconstructs)
        if (util::expected_t scored = tokens(tower.normalized(), clean_tokens.view(), token_loss_weights.view(),
                                             tokens_summands.span(), normalized_grad.span(), stream);
            !scored)
            return scored;
    if (chooses)
        if (util::expected_t scored = tokens.choose(tower.normalized(), batch.view(), config.choices,
                                                    choices_summands.span(), normalized_grad.span(), stream);
            !scored)
            return scored;
    if (image_live)
        if (util::expected_t scored = image(tower.normalized(), clean_image_patches.view(), image_summands.span(),
                                            normalized_grad.span(), stream);
            !scored)
            return scored;
    if (video_live)
        if (util::expected_t scored = video(tower.normalized(), clean_video_patches.view(), video_summands.span(),
                                            normalized_grad.span(), stream);
            !scored)
            return scored;
    if (audio_live)
        if (util::expected_t scored = audio(tower.normalized(), clean_audio.view(), audio_summands.span(),
                                            normalized_grad.span(), stream);
            !scored)
            return scored;
    return {};
}

util::expected_t base_replica_t::backward_region(util::executor_stream_t& stream) noexcept {
    return tower.backward(vertices_grad.view(), normalized_grad.view(), batch, stream, token_vectors_grad.view());
}

contrastive_recipe_t base_replica_t::recipe(trainer_config_t const& config) const noexcept {
    contrastive_recipe_t recipe;
    recipe.inverse_temperature = config.contrastive.inverse_temperature;
    recipe.bias = config.contrastive.bias;
    recipe.ambient_repulsion_weight = config.contrastive.ambient_repulsion_weight;
    recipe.inverse_temperature_device = inverse_temperatures.view();
    recipe.bias_device = biases.view();
    recipe.from_vertex_temperature_group = temperature_groups.view();
    return recipe;
}

contrastive_recipe_t base_replica_t::token_recipe(trainer_config_t const& config) const noexcept {
    contrastive_recipe_t recipe;
    recipe.inverse_temperature = config.late_interaction.inverse_temperature;
    recipe.bias = config.late_interaction.bias;
    recipe.ambient_repulsion_weight = config.contrastive.ambient_repulsion_weight;
    recipe.inverse_temperature_device = token_inverse_temperatures.view();
    recipe.bias_device = token_biases.view();
    recipe.from_vertex_temperature_group = temperature_groups.view();
    return recipe;
}
