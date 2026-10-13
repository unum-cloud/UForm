/**
 *  @brief The implementation of the Base pre-training library declared in @c src/train/base.cuh : the
 *      synthetic-mixture kernels, the file-local host helpers, and every out-of-line body of
 *      @ref base_config_t / @ref base_replica_t / @ref base_trainer_t .
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
#include <cmath>   // `std::cos` — the cosine decay
#include <cstdlib> // `std::atoi`, `std::getenv`
#include <cstring> // `std::strlen`

#include <unistd.h> // `::write`, `STDOUT_FILENO`, `STDERR_FILENO` — the host diagnostic channel

#include <numbers> // `std::numbers::pi_v` — the decay's half-period

#include <cuda.h> // `cuInit`, `cuMemcpyDtoDAsync`, `cuMemsetD8Async`, `cuStreamSynchronize`

#include "train/base.cuh"

#include "core/checkpoint.hpp" // `checkpoint_to_save_t`, `checkpoint_to_load_t` — snapshot and warm start

using namespace unum::uform::core;
using namespace unum::uform::train;

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

/** Write one line to STDOUT through POSIX @c ::write , newline-terminated. */
static void report_(sz::string_view_t line) noexcept {
    ::write(STDOUT_FILENO, line.data(), line.size());
    ::write(STDOUT_FILENO, "\n", 1);
}

/** Tag @p message with the binary name, write it to STDERR, and hand back main's failure code — so
 *  every fallible phase collapses to one line: @c return @c fail_out_(x.description_.view()) . */
[[nodiscard]] static int fail_out_(sz::string_view_t message) noexcept {
    sz::string_t line;
    if (format_into(line, "uform_base: {}", message)) {
        ::write(STDERR_FILENO, line.data(), line.size());
        ::write(STDERR_FILENO, "\n", 1);
    }
    return 1;
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

/** Seed every matrix of one replica's registration list: full matrices from the deterministic hash
 *  fill, vector scales (rows==1) exactly one. Identical on every replica; the parity test reruns it so
 *  both runs start from equals. */
[[nodiscard]] static expected_t seed_weights_(array<differentiable_weights_t> const &weights,
                                              gpu_stream_t &stream) noexcept {
    std::span<differentiable_weights_t const> const list {weights.view().data(), weights.size()};
    for (std::size_t weight_index = 0; weight_index != list.size(); ++weight_index) {
        differentiable_weights_t const &matrix = list[weight_index];
        std::size_t const rows = matrix.weights.extent(0), columns = matrix.weights.extent(1);
        nk::tensor<bf16_t, pinned_alloc<bf16_t>, 1> staging =
            nk::tensor<bf16_t, pinned_alloc<bf16_t>, 1>::uninitialized({rows * columns}, pinned_alloc<bf16_t> {}).value;
        if (!staging) return expected_t::fail(status_t::bad_alloc_k, "weight staging failed (matrix={})", weight_index);
        bf16r1_span_t const host = staging.span();
        for (std::size_t index = 0; index != host.extent(0); ++index)
            host[index] = __float2bfloat16( //
                rows == 1 ? 1.0f : hash_fill_(index, static_cast<u32_t>(weight_index) + 11u, 0.2f));
        if (expected_t copied = copy_host_to_device(staging.view(), matrix.weights.flatten<1>(), stream); !copied)
            return copied;
        // The staging tensor dies this iteration — the enqueued copy must land first.
        cuStreamSynchronize(stream.get());
    }
    return {};
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

f32_t base_config_t::learning_rate_for(optimizer_step_t optimizer_step) const noexcept {
    if (optimizer_step.ordinal < warmup_steps)
        return learning_rate * static_cast<f32_t>(optimizer_step.ordinal + 1u) / static_cast<f32_t>(warmup_steps);
    u32_t const decay_steps = steps > warmup_steps ? steps - warmup_steps : 1u;
    f32_t const progress = static_cast<f32_t>(optimizer_step.ordinal - warmup_steps) / static_cast<f32_t>(decay_steps);
    f32_t const cosine = 0.5f * (1.0f + std::cos(std::numbers::pi_v<f32_t> * std::min(progress, 1.0f)));
    return min_learning_rate + (learning_rate - min_learning_rate) * cosine;
}

/** The masking corruptor's keyed recipe — its seed derived once from the run seed. */
[[nodiscard]] static masking_recipe_t masking_recipe_(base_config_t const &config) noexcept {
    masking_recipe_t recipe;
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
[[nodiscard]] static row_partition_t partition_rows_(base_config_t const &config, u32_t gpu_index_in_colocated_domain,
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
    for (u32_t modality = 0; modality != 4u; ++modality) {
        if (!live[modality]) continue;
        u32_t const bonus = ((position + live_count - rotation) % live_count) < extra ? 1u : 0u;
        partition.rows[modality] = floor_rows + bonus;
        ++position;
    }
    u32_t cursor = 0;
    for (u32_t modality = 0; modality != 4u; ++modality) {
        partition.begins[modality] = cursor;
        cursor += partition.rows[modality];
    }
    return partition;
}

/** Fold the run description into the tower's make-config: the block geometry from the batch envelope,
 *  one media entry per live batch slice — the batch shape stays the single source of modality presence. */
[[nodiscard]] static encoder_tower_config_t tower_config_(base_config_t const &config) noexcept {
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
    tower.rope_max_positions = batch.max_columns_per_row;
    if (batch.contains(modality_t::image_k)) {
        tower.image_patch_height = batch.image.patch_side;
        tower.image_patch_width = batch.image.patch_side;
        tower.image_channels = batch.image.channels;
    }
    if (batch.contains(modality_t::video_k)) {
        tower.video_tube_frames = batch.video.tube_frames;
        tower.video_patch_height = batch.video.patch_side;
        tower.video_patch_width = batch.video.patch_side;
        tower.video_channels = batch.video.channels;
    }
    if (batch.contains(modality_t::audio_k)) tower.audio_chunk_samples = batch.audio.chunk_samples;
    return tower;
}

/** Warm-start one replica from @p checkpoint : every weight matrix uploads its by-name tensor —
 *  a missing name or a shape mismatch fails loudly (warm starts are all-or-nothing; partial loads
 *  would silently train a half-seeded model). Async on @p stream off the pinned payload; the caller
 *  drains before the checkpoint's backing is released. */
[[nodiscard]] static expected_t load_checkpoint_(checkpoint_to_load_t const &checkpoint,
                                                 array<differentiable_weights_t> const &weights,
                                                 gpu_stream_t &stream) noexcept {
    std::span<differentiable_weights_t const> const list {weights.view().data(), weights.size()};
    sz::string_t name_text;
    for (differentiable_weights_t const &matrix : list) {
        name_text.clear();
        if (!format_into(name_text, "model.{}", matrix.name))
            return expected_t::fail(status_t::bad_alloc_k, "matrix name format failed ({})", matrix.name);
        nk::tensor_view<bf16_t, 2> const saved = checkpoint[name_text.view()];
        if (saved.empty())
            return expected_t::fail(status_t::invalid_input_k, "the checkpoint misses {}", name_text.view());
        if (saved.extent(0) != matrix.weights.extent(0) || saved.extent(1) != matrix.weights.extent(1))
            return expected_t::fail(status_t::shape_unsupported_k, "{} is [{},{}] in the checkpoint, [{},{}] live",
                                    name_text.view(), saved.extent(0), saved.extent(1), matrix.weights.extent(0),
                                    matrix.weights.extent(1));
        if (expected_t uploaded = copy_host_to_device(saved.flatten<1>(), matrix.weights.flatten<1>(), stream);
            !uploaded)
            return uploaded;
    }
    return {};
}

/** Snapshot one replica's registration list to @p path with the run's step count, each matrix keyed
 *  by the name its cell's @c fill spelled, under this composition root's @c model. prefix. */
[[nodiscard]] static expected_t save_checkpoint_(sz::string_view_t path, array<differentiable_weights_t> const &weights,
                                                 u32_t steps, gpu_stream_t &stream) noexcept {
    switch_to_gpu_t gpu_scope {stream};
    if (!gpu_scope.ok()) return expected_t::fail(status_t::cuda_error_k, "context entry failed");
    std::span<differentiable_weights_t const> const list {weights.view().data(), weights.size()};
    u64_t total_scalars = 0;
    for (differentiable_weights_t const &matrix : list) total_scalars += matrix.weights.numel();
    expected<checkpoint_to_save_t> made = checkpoint_to_save_t::make(pinned_alloc_t {});
    if (!made) return expected_t::fail(std::move(made));
    checkpoint_to_save_t checkpoint = std::move(made.value_);
    if (expected_t reserved = checkpoint.try_reserve(total_scalars, list.size()); !reserved) return reserved;
    sz::string_t name_text;
    for (differentiable_weights_t const &matrix : list) {
        name_text.clear();
        if (!format_into(name_text, "model.{}", matrix.name))
            return expected_t::fail(status_t::bad_alloc_k, "matrix name format failed ({})", matrix.name);
        if (expected_t inserted = checkpoint.try_insert(name_text.view(), matrix.weights, stream); !inserted)
            return inserted;
    }
    return checkpoint.save(path, steps);
}

base_config_t base_config_t::parse(int argc, char **argv) noexcept {
    base_config_t config;
    u32_t const count = static_cast<u32_t>(argc);
    for (u32_t arg = 1; arg < count; ++arg) {
        sz::string_view_t const flag {argv[arg], std::strlen(argv[arg])};
        if (flag == "--eager") {
            config.capture = eager_k;
            continue;
        }
        if (flag == "--tune") {
            config.algo_pick = algo_pick_t::benchmarked_k;
            continue;
        }
        if (arg + 1 >= count) break;
        ++arg;
        sz::string_view_t const value {argv[arg], std::strlen(argv[arg])};
        if (flag == "--steps") config.steps = std::atoi(argv[arg]);
        else if (flag == "--hidden") config.hidden = std::atoi(argv[arg]);
        else if (flag == "--depth") config.depth = std::atoi(argv[arg]);
        else if (flag == "--rows") {
            // One segment per row in the synthetic corpus, so the segment budget tracks the row budget.
            config.batch.max_rows_per_gpu = std::atoi(argv[arg]);
            config.batch.max_segments_per_gpu = config.batch.max_rows_per_gpu;
        }
        else if (flag == "--columns") config.batch.max_columns_per_row = std::atoi(argv[arg]);
        else if (flag == "--mixture" && value == "omni") {
            // The omni preset: 8-pixel RGB image patches, 2-frame RGB video tubes, 256-sample audio frames.
            config.batch.image = image_batch_shape_t {8u, 3u};
            config.batch.video = video_batch_shape_t {8u, 3u, 2u, 1u};
            config.batch.audio = audio_batch_shape_t {256u, 16000u};
        }
        else if (flag == "--checkpoint") config.checkpoint = value;
        else if (flag == "--resume") config.resume = value;
        else if (flag == "--telemetry") {
            if (value == "signals") config.telemetry_channels |= telemetry_signals_k;
            else if (value == "timings") config.telemetry_channels |= telemetry_timings_k;
        }
    }
    return config;
}

expected<base_replica_t> base_replica_t::make(gpu_arena_t &arena, cudnn_session_t &dnn,
                                              base_config_t const &config) noexcept {
    using result_t = expected<base_replica_t>;
    multimodal_batch_shape_t const &shape = config.batch;
    u32_t const max_tokens = shape.max_rows_per_gpu * shape.max_columns_per_row;

    base_replica_t replica;
    expected<encoder_tower_bf16_t> tower = encoder_tower_bf16_t::make(arena, dnn, tower_config_(config), shape);
    if (!tower) return result_t::fail(std::move(tower));
    replica.tower = std::move(tower.value_);
    expected<multimodal_batch_on_gpu_t> batch = multimodal_batch_on_gpu_t::make(gpu_arena_alloc_t {arena}, shape);
    if (!batch) return result_t::fail(std::move(batch));
    replica.batch = std::move(batch.value_);

    // The token head and its scratch — always live; the shared d(normalized) is every head's accumulator.
    masked_tokens_config_t tokens_config;
    tokens_config.hidden = config.hidden;
    tokens_config.max_tokens = max_tokens;
    expected<masked_tokens_t> tokens = masked_tokens_t::make(arena, tokens_config);
    if (!tokens) return result_t::fail(std::move(tokens));
    replica.tokens = std::move(tokens.value_);
    gpu_arena_alloc<token_id_t> token_alloc {arena};
    gpu_arena_alloc<u32_t> u32_alloc {arena};
    gpu_arena_alloc<f32_t> f32_alloc {arena};
    gpu_arena_alloc<bf16_t> bf16_alloc {arena};
    replica.clean_tokens = tokens_tensor_arena_t::uninitialized({max_tokens}, token_alloc).value;
    replica.token_loss_weights = u32r1_tensor_arena_t::uninitialized({max_tokens}, u32_alloc).value;
    replica.tokens_summands = f32r1_tensor_arena_t::uninitialized({2}, f32_alloc).value;
    replica.normalized_grad = bf16r2_tensor_arena_t::uninitialized({max_tokens, config.hidden}, bf16_alloc).value;
    if (!replica.clean_tokens || !replica.token_loss_weights || !replica.tokens_summands || !replica.normalized_grad)
        return result_t::fail(status_t::bad_alloc_k, "masked scratch allocation failed");

    // Each live media head with its clean target and summands — the same order on every replica.
    if (shape.contains(modality_t::image_k)) {
        masked_patches_config_t head_config;
        head_config.hidden = config.hidden;
        head_config.patch_pixels = patch_scalars(shape.image.patch_side, shape.image.channels);
        head_config.max_patches = max_tokens;
        head_config.mask_ratio = 0.75f;
        expected<masked_patches_t> head = masked_patches_t::make(arena, head_config);
        if (!head) return result_t::fail(std::move(head));
        replica.image = std::move(head.value_);
        replica.clean_image_patches =
            bf16r2_tensor_arena_t::uninitialized({max_tokens, head_config.patch_pixels}, bf16_alloc).value;
        replica.image_summands = f32r1_tensor_arena_t::uninitialized({2}, f32_alloc).value;
        if (!replica.clean_image_patches || !replica.image_summands)
            return result_t::fail(status_t::bad_alloc_k, "image scratch allocation failed");
    }
    if (shape.contains(modality_t::video_k)) {
        masked_patches_config_t head_config;
        head_config.hidden = config.hidden;
        head_config.patch_pixels = tube_scalars(shape.video.tube_frames, shape.video.patch_side, //
                                                shape.video.channels);
        head_config.max_patches = max_tokens;
        head_config.mask_ratio = 0.90f;
        expected<masked_patches_t> head = masked_patches_t::make(arena, head_config);
        if (!head) return result_t::fail(std::move(head));
        replica.video = std::move(head.value_);
        replica.clean_video_patches =
            bf16r2_tensor_arena_t::uninitialized({max_tokens, head_config.patch_pixels}, bf16_alloc).value;
        replica.video_summands = f32r1_tensor_arena_t::uninitialized({2}, f32_alloc).value;
        if (!replica.clean_video_patches || !replica.video_summands)
            return result_t::fail(status_t::bad_alloc_k, "video scratch allocation failed");
    }
    if (shape.contains(modality_t::audio_k)) {
        masked_frames_config_t head_config;
        head_config.hidden = config.hidden;
        head_config.chunk_samples = shape.audio.chunk_samples;
        head_config.max_frames = max_tokens;
        head_config.mask_ratio = 0.75f;
        expected<masked_frames_t> head = masked_frames_t::make(arena, head_config);
        if (!head) return result_t::fail(std::move(head));
        replica.audio = std::move(head.value_);
        replica.clean_audio =
            f32r2_tensor_arena_t::uninitialized({max_tokens, shape.audio.chunk_samples}, f32_alloc).value;
        replica.audio_summands = f32r1_tensor_arena_t::uninitialized({2}, f32_alloc).value;
        if (!replica.clean_audio || !replica.audio_summands)
            return result_t::fail(status_t::bad_alloc_k, "audio scratch allocation failed");
    }

    // The registration list — tower first, then each live head, identical order on every replica. The tower
    // spells its own weight names; the heads take their instance names here, where the modalities are told apart.
    if (expected_t listed = replica.tower.fill(replica.weights); !listed) return result_t::fail(std::move(listed));
    if (expected_t listed = replica.tokens.fill("masked_tokens", replica.weights); !listed)
        return result_t::fail(std::move(listed));
    if (shape.contains(modality_t::image_k))
        if (expected_t listed = replica.image.fill("masked_image", replica.weights); !listed)
            return result_t::fail(std::move(listed));
    if (shape.contains(modality_t::video_k))
        if (expected_t listed = replica.video.fill("masked_video", replica.weights); !listed)
            return result_t::fail(std::move(listed));
    if (shape.contains(modality_t::audio_k))
        if (expected_t listed = replica.audio.fill("masked_audio", replica.weights); !listed)
            return result_t::fail(std::move(listed));
    return result_t {std::move(replica)};
}

// TODO: The REAL-DATA JOIN — the biggest remaining gap. This synthetic draw is the boundary where the
// validated pieces meet: the Parquet loaders decode green end to end (tokens/image/video/audio fill a
// multimodal_batch_on_gpu_t, keyed by the same (seed, gpu_index_in_colocated_domain, step) contract), and
// the cross-node multimodal_batch_mixer_t is validated at 16 GPUs — but nothing wires loader fills into
// the trainer's loaded batches and runs the mixer between fill and model_region. The join replaces this
// function's synthetic kernels with loader fills under a config switch, keeps the synthetic path for the
// conformance suites, and threads the loaders' GDS session + CPU decode pool from the composition root.
expected_t base_replica_t::prepare(base_config_t const &config, u32_t gpu_index_in_colocated_domain,
                                   data_epoch_t data_epoch, gpu_stream_t &stream) noexcept {
    multimodal_batch_shape_t const &shape = config.batch;
    bool const image_live = shape.contains(modality_t::image_k);
    bool const video_live = shape.contains(modality_t::video_k);
    bool const audio_live = shape.contains(modality_t::audio_k);
    masking_recipe_t const recipe = masking_recipe_(config);
    row_partition_t const partition = partition_rows_(config, gpu_index_in_colocated_domain, data_epoch);
    grid_factor_t const factor = square_factor_(shape.max_columns_per_row);
    dim3 const row_grid {shape.max_rows_per_gpu, 1u, 1u};
    dim3 const block_shape {kernel_block_size_default_k, 1u, 1u};
    tokens_span_t const tokens_flat = batch.tokens.span().flatten<1>();
    std::size_t const max_tokens = tokens_flat.extent(0);

    // 1. The keyed byte grid, then every row's segment descriptor from the drawn partition.
    if (expected_t drawn = launch_kernel(reinterpret_cast<void const *>(&synthetic_tokens_kernel_), row_grid,
                                         block_shape, stream, batch.tokens.span(), config.seed, data_epoch,
                                         gpu_index_in_colocated_domain);
        !drawn)
        return drawn;
    if (expected_t laid = launch_kernel(reinterpret_cast<void const *>(&synthetic_segments_kernel_), row_grid,
                                        block_shape, stream, batch.segment_offsets.span(), batch.segment_lengths.span(),
                                        batch.segment_modality.span(), batch.image_shapes.span(),
                                        batch.video_shapes.span(), shape.max_columns_per_row, partition.begins[1],
                                        partition.begins[2], partition.begins[3], factor.patch_rows, factor.patch_cols);
        !laid)
        return laid;

    // 2. Each live modality's raw content — filled across the whole tensor, keyed identically every run.
    if (image_live) {
        u32_t const patch_pixels = patch_scalars(shape.image.patch_side, shape.image.channels);
        nk::tensor_span<pixel_t, 2> const raw = batch.images.span().reshape<2>({max_tokens, patch_pixels});
        dim3 const grid {grid_size_for(raw.extent(0) * raw.extent(1)), 1u, 1u};
        if (expected_t filled = launch_kernel(reinterpret_cast<void const *>(&synthetic_pixels_kernel_), grid,
                                              block_shape, stream, raw, config.seed, data_epoch,
                                              gpu_index_in_colocated_domain, 101u);
            !filled)
            return filled;
    }
    if (video_live) {
        u32_t const patch_pixels = tube_scalars(shape.video.tube_frames, shape.video.patch_side, //
                                                shape.video.channels);
        nk::tensor_span<pixel_t, 2> const raw = batch.videos.span().reshape<2>({max_tokens, patch_pixels});
        dim3 const grid {grid_size_for(raw.extent(0) * raw.extent(1)), 1u, 1u};
        if (expected_t filled = launch_kernel(reinterpret_cast<void const *>(&synthetic_pixels_kernel_), grid,
                                              block_shape, stream, raw, config.seed, data_epoch,
                                              gpu_index_in_colocated_domain, 202u);
            !filled)
            return filled;
    }
    if (audio_live) {
        f32r2_span_t const raw = batch.audio_samples.span().reshape<2>({max_tokens, shape.audio.chunk_samples});
        dim3 const grid {grid_size_for(raw.extent(0) * raw.extent(1)), 1u, 1u};
        if (expected_t filled = launch_kernel(reinterpret_cast<void const *>(&synthetic_audio_kernel_), grid,
                                              block_shape, stream, raw, config.seed, data_epoch,
                                              gpu_index_in_colocated_domain, 303u);
            !filled)
            return filled;
    }

    // 3. The token target and its distortion — clean snapshot first, then the keyed span corruption.
    if (CUresult result = cuMemcpyDtoDAsync(reinterpret_cast<CUdeviceptr>(clean_tokens.data()),
                                            reinterpret_cast<CUdeviceptr>(tokens_flat.data()),
                                            max_tokens * sizeof(token_id_t), stream.get());
        result != CUDA_SUCCESS)
        return expected_t::fail(result, "clean-token snapshot failed (gpu_index={})", gpu_index_in_colocated_domain);
    if (expected_t corrupted = tokens.corrupt(clean_tokens.view(), tokens_flat, token_loss_weights.span(),
                                              batch.segment_offsets.view(), batch.segment_lengths.view(),
                                              batch.segment_modality.view(), recipe, data_epoch, stream);
        !corrupted)
        return corrupted;
    if (!image_live && !video_live && !audio_live) return {};

    // 4. The media clean targets — staged from the still-pristine batch, before any raw row is zeroed.
    if (image_live || video_live) {
        bf16r2_span_t const image_target = image_live ? clean_image_patches.span() : bf16r2_span_t {};
        bf16r2_span_t const video_target = video_live ? clean_video_patches.span() : bf16r2_span_t {};
        if (expected_t staged = tower.patchify_media_targets(batch, image_target, video_target, stream); !staged)
            return staged;
    }
    if (audio_live)
        if (CUresult result = cuMemcpyDtoDAsync(
                reinterpret_cast<CUdeviceptr>(clean_audio.data()),
                reinterpret_cast<CUdeviceptr>(batch.audio_samples.data()),
                static_cast<std::size_t>(max_tokens) * shape.audio.chunk_samples * sizeof(f32_t), stream.get());
            result != CUDA_SUCCESS)
            return expected_t::fail(result, "clean-audio snapshot failed (gpu_index={})",
                                    gpu_index_in_colocated_domain);

    // 5. The media masks and the in-place raw corruption — sampled eagerly so the masks persist into the
    //    captured model region.
    if (image_live) {
        if (expected_t sampled = image.sample_mask(recipe, data_epoch, stream); !sampled) return sampled;
        u32_t const patch_pixels = patch_scalars(shape.image.patch_side, shape.image.channels);
        nk::tensor_span<pixel_t, 2> const raw = batch.images.span().reshape<2>({max_tokens, patch_pixels});
        if (expected_t corrupted = image.corrupt_raw(raw, stream); !corrupted) return corrupted;
    }
    if (video_live) {
        if (expected_t sampled = video.sample_mask(recipe, data_epoch, stream); !sampled) return sampled;
        u32_t const patch_pixels = tube_scalars(shape.video.tube_frames, shape.video.patch_side, //
                                                shape.video.channels);
        nk::tensor_span<pixel_t, 2> const raw = batch.videos.span().reshape<2>({max_tokens, patch_pixels});
        if (expected_t corrupted = video.corrupt_raw(raw, stream); !corrupted) return corrupted;
    }
    if (audio_live) {
        if (expected_t sampled = audio.sample_mask(recipe, data_epoch, stream); !sampled) return sampled;
        f32r2_span_t const raw = batch.audio_samples.span().reshape<2>({max_tokens, shape.audio.chunk_samples});
        if (expected_t corrupted = audio.corrupt_raw(raw, stream); !corrupted) return corrupted;
    }
    return {};
}

expected_t base_replica_t::model_region(gpu_stream_t &stream) noexcept {
    // A head is live iff its summand pair was allocated at make.
    bool const image_live = image_summands.data() != nullptr;
    bool const video_live = video_summands.data() != nullptr;
    bool const audio_live = audio_summands.data() != nullptr;
    bool const composite = image_live || video_live || audio_live;

    if (expected_t packed = tower.pack_weights(stream); !packed) return packed;
    if (expected_t packed = tokens.pack_weights(stream); !packed) return packed;
    if (image_live)
        if (expected_t packed = image.pack_weights(stream); !packed) return packed;
    if (video_live)
        if (expected_t packed = video.pack_weights(stream); !packed) return packed;
    if (audio_live)
        if (expected_t packed = audio.pack_weights(stream); !packed) return packed;
    if (expected_t encoded = tower.forward(batch, bf16r2_span_t {}, stream); !encoded) return encoded;

    // The shared d(normalized): the token head overwrites first, so the reset matters only when media heads add.
    if (composite) {
        bf16r2_view_t const view = normalized_grad.view();
        std::size_t const bytes = static_cast<std::size_t>(view.extent(0)) * view.extent(1) * sizeof(bf16_t);
        if (CUresult result = cuMemsetD8Async(reinterpret_cast<CUdeviceptr>(normalized_grad.data()), 0, bytes,
                                              stream.get());
            result != CUDA_SUCCESS)
            return expected_t::fail(result, "shared d(normalized) reset failed");
    }
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

    return tower.backward(bf16r2_view_t {}, normalized_grad.view(), batch, stream);
}

expected_t base_trainer_t::make(base_trainer_t &trainer, base_config_t const &config, mesh_topology_t const &topology,
                                telemetry_ledger_t &ledger) noexcept {
    trainer.config = config;
    trainer.ledger = &ledger;
    u32_t const gpus = topology.shape.gpus_per_host_domain;
    trainer.first_gpu_index_in_colocated_domain = topology.shape.gpu_index_of_host_member(topology.host_domain_index,
                                                                                          0u);
    if (trainer.devices.try_resize(gpus) != status_t::success_k)
        return expected_t::fail(status_t::bad_alloc_k, "device table allocation failed");
    for (u32_t member_index = 0; member_index != gpus; ++member_index)
        trainer.devices[member_index] = static_cast<i32_t>(member_index);

    // Per GPU: the stream, the arena, the cuDNN session (placed address-stable, borrowed by pointer for the
    // replica's whole lifetime), then the replica itself — which publishes its own registration list.
    for (u32_t member_index = 0; member_index != gpus; ++member_index) {
        expected<gpu_stream_t> stream = gpu_stream_t::make(trainer.devices[member_index]);
        if (!stream) return expected_t::fail(std::move(stream));
        if (trainer.streams.try_push_back(std::move(stream.value_)) != status_t::success_k)
            return expected_t::fail(status_t::bad_alloc_k, "stream table full");
        switch_to_gpu_t gpu_scope {trainer.streams[member_index]};
        if (!gpu_scope.ok()) return expected_t::fail(status_t::cuda_error_k, "context entry failed");
        expected<gpu_arena_t> arena = gpu_arena_t::make(trainer.devices[member_index], 8ull << 30,
                                                        trainer.devices.view());
        if (!arena) return expected_t::fail(std::move(arena));
        if (trainer.arenas.try_push_back(std::move(arena.value_)) != status_t::success_k)
            return expected_t::fail(status_t::bad_alloc_k, "arena table full");
        expected<cudnn_session_t> dnn_session = cudnn_session_t::make(config.algo_pick);
        if (!dnn_session) return expected_t::fail(std::move(dnn_session));
        if (trainer.dnn.try_push_back(std::move(dnn_session.value_)) != status_t::success_k)
            return expected_t::fail(status_t::bad_alloc_k, "dnn session table full");
        expected<base_replica_t> replica = base_replica_t::make(trainer.arenas[member_index], trainer.dnn[member_index],
                                                                config);
        if (!replica) return expected_t::fail(std::move(replica));
        if (trainer.replicas.try_push_back(std::move(replica.value_)) != status_t::success_k)
            return expected_t::fail(status_t::bad_alloc_k, "replica table full");
    }

    // The trainer's ledger rows: one loss probe per live head, one round probe per member — inert zero
    // handles whenever the corresponding channel is dark, so the feeding sites never branch.
    auto loss_probe = [&](char const *name, bool live, telemetry_probe_t &into) -> expected_t {
        if (!live) return {};
        expected<telemetry_probe_t> registered = ledger.register_probe(name, telemetry_losses_k, finite_signal_k);
        if (!registered) return expected_t::fail(std::move(registered));
        into = registered.value_;
        return {};
    };
    if (expected_t probed = loss_probe("base/token_ce", true, trainer.probes.token); !probed) return probed;
    if (expected_t probed = loss_probe("base/image_mse", config.batch.contains(modality_t::image_k),
                                       trainer.probes.image);
        !probed)
        return probed;
    if (expected_t probed = loss_probe("base/video_mse", config.batch.contains(modality_t::video_k),
                                       trainer.probes.video);
        !probed)
        return probed;
    if (expected_t probed = loss_probe("base/audio_mse", config.batch.contains(modality_t::audio_k),
                                       trainer.probes.audio);
        !probed)
        return probed;
    for (u32_t member_index = 0; member_index != gpus; ++member_index) {
        sz::string_t name;
        if (!format_into(name, "base/round/{}", member_index))
            return expected_t::fail(status_t::bad_alloc_k, "round probe name formatting failed");
        expected<telemetry_probe_t> registered = ledger.register_probe(name.view(), telemetry_rounds_k,
                                                                       monotone_signal_k);
        if (!registered) return expected_t::fail(std::move(registered));
        if (trainer.probes.rounds.try_push_back(registered.value_) != status_t::success_k)
            return expected_t::fail(status_t::bad_alloc_k, "round probe table full");
    }
    return trainer.reseed();
}

expected_t base_trainer_t::capture(per_gpu_array<gpu_replay_graph_t> &graphs) noexcept {
    u32_t const gpus = static_cast<u32_t>(replicas.size());
    for (u32_t member_index = 0; member_index != gpus; ++member_index) {
        switch_to_gpu_t gpu_scope {streams[member_index]};
        if (!gpu_scope.ok()) return expected_t::fail(status_t::cuda_error_k, "context entry failed");
        u32_t const mi = member_index; // bound by value — each captured body re-reads its own replica
        expected<gpu_replay_graph_t> graph = gpu_replay_graph_t::make(
            streams[mi].get(), [&]() noexcept { return replicas[mi].model_region(streams[mi]); });
        if (!graph) return expected_t::fail(std::move(graph));
        if (graphs.try_push_back(std::move(graph.value_)) != status_t::success_k)
            return expected_t::fail(status_t::bad_alloc_k, "model-graph table full");
    }
    return {};
}

expected_t base_trainer_t::step(optimizer_step_t optimizer_step, distributed_adafactor_t &optimizer,
                                std::span<gpu_replay_graph_t> graphs) noexcept {
    // Synthetic mode: the pass over data coincides with the optimizer iteration.
    data_epoch_t const data_epoch {optimizer_step.ordinal};
    f32_t const learning_rate = config.learning_rate_for(optimizer_step);
    u32_t const gpus = static_cast<u32_t>(replicas.size());
    for (u32_t member_index = 0; member_index != gpus; ++member_index) {
        switch_to_gpu_t gpu_scope {streams[member_index]};
        if (!gpu_scope.ok()) return expected_t::fail(status_t::cuda_error_k, "context entry failed");
        expected_t stepped = replicas[member_index].prepare(config, first_gpu_index_in_colocated_domain + member_index,
                                                            data_epoch, streams[member_index]);
        if (stepped)
            stepped = graphs.empty() ? replicas[member_index].model_region(streams[member_index])
                                     : graphs[member_index].launch(streams[member_index].get());
        if (stepped) stepped = optimizer.step(member_index, learning_rate, streams[member_index]);
        if (!stepped) return stepped;
    }
    return {};
}

expected_t base_trainer_t::reseed() noexcept {
    u32_t const gpus = static_cast<u32_t>(replicas.size());
    for (u32_t member_index = 0; member_index != gpus; ++member_index) {
        switch_to_gpu_t gpu_scope {streams[member_index]};
        if (!gpu_scope.ok()) return expected_t::fail(status_t::cuda_error_k, "context entry failed");
        if (expected_t seeded = seed_weights_(replicas[member_index].weights, streams[member_index]); !seeded)
            return seeded;
    }
    return {};
}

base_trainer_t::telemetry_t base_trainer_t::telemetry(multimem_domain_t &domain,
                                                      optimizer_step_t optimizer_step) noexcept {
    telemetry_t telemetry;
    telemetry.optimizer_step = optimizer_step;
    u32_t const gpus = static_cast<u32_t>(replicas.size());
    if (gpus == 0) return telemetry;
    // Accumulate every member's summands before the fold — at wide worlds the data spreads across the
    // whole colocated_domain, and a lead-only reading is ~2 rows of one head on ~half the steps: a coin
    // flip, not a statistic. The all-member pairs are also what the ledger's loss rows accumulate.
    summand_pair_t token {}, image {}, video {}, audio {};
    bool const image_live = config.batch.contains(modality_t::image_k);
    bool const video_live = config.batch.contains(modality_t::video_k);
    bool const audio_live = config.batch.contains(modality_t::audio_k);
    auto accumulate = [](summand_pair_t &into, summand_pair_t read) {
        into.sum += read.sum;
        into.count += read.count;
    };
    for (u32_t member_index = 0; member_index != gpus; ++member_index) {
        switch_to_gpu_t gpu_scope {streams[member_index]};
        if (!gpu_scope.ok()) continue;
        cuStreamSynchronize(streams[member_index].get());
        base_replica_t &replica = replicas[member_index];
        accumulate(token, read_summands_(replica.tokens_summands, streams[member_index]));
        if (image_live) accumulate(image, read_summands_(replica.image_summands, streams[member_index]));
        if (video_live) accumulate(video, read_summands_(replica.video_summands, streams[member_index]));
        if (audio_live) accumulate(audio, read_summands_(replica.audio_summands, streams[member_index]));
        if (ledger != nullptr)
            (void)ledger->probe_round(probes.rounds[member_index], domain.member(member_index), streams[member_index]);
    }
    telemetry.token_ce = token.count != 0.0 ? static_cast<f32_t>(token.sum / token.count) : 0.0f;
    telemetry.image_mse = image.count != 0.0 ? static_cast<f32_t>(image.sum / image.count) : 0.0f;
    telemetry.video_mse = video.count != 0.0 ? static_cast<f32_t>(video.sum / video.count) : 0.0f;
    telemetry.audio_mse = audio.count != 0.0 ? static_cast<f32_t>(audio.sum / audio.count) : 0.0f;
    if (ledger != nullptr) {
        ledger->scalar(probes.token, token.sum, token.count);
        ledger->scalar(probes.image, image.sum, image.count);
        ledger->scalar(probes.video, video.sum, video.count);
        ledger->scalar(probes.audio, audio.sum, audio.count);
    }
    switch_to_gpu_t gpu_scope {streams[0]};
    if (u32_t const *const round = domain.member(0).round; gpu_scope.ok() && round != nullptr) {
        u32r1_tensor_pinned_t staging = u32r1_tensor_pinned_t::uninitialized({1}, pinned_alloc<u32_t> {}).value;
        nk::tensor_view<u32_t, 1> const source {round, 1};
        if (staging && copy_device_to_host(source, staging.span(), streams[0])) {
            cuStreamSynchronize(streams[0].get());
            telemetry.concurrency_round = concurrency_round_t {staging[0]};
        }
    }
    return telemetry;
}

void base_trainer_t::telemetry_t::report() const noexcept {
    sz::string_t line;
    if (format_into(line, "  step {:>4}  masked CE {:.6f}  round {}", optimizer_step.ordinal + 1u, token_ce,
                    concurrency_round.ordinal))
        report_(line.view());
    if (image_mse != 0.0f) {
        line.clear();
        if (format_into(line, "            image MSE {:.6f}", image_mse)) report_(line.view());
    }
    if (video_mse != 0.0f) {
        line.clear();
        if (format_into(line, "            video MSE {:.6f}", video_mse)) report_(line.view());
    }
    if (audio_mse != 0.0f) {
        line.clear();
        if (format_into(line, "            audio MSE {:.6f}", audio_mse)) report_(line.view());
    }
}

int base_trainer_t::run(base_config_t const &config) noexcept {
    if (cuInit(0) != CUDA_SUCCESS) return fail_out_("cuInit failed");

    // 1. The mesh: one process per host_domain; single-host worlds need no exchange directory.
    u32_t const host_domain_index = host_domain_index_from_environment();
    u32_t const host_domain_count = host_domain_count_from_environment();
    expected<mesh_topology_t> topology = mesh_topology_t::bootstrap(host_domain_index, host_domain_count);
    if (!topology) return fail_out_(topology.description_.view());
    sz::string_t banner;
    if (format_into(banner, "uform_base: tokens{}{}{} mixture, {} GPU(s), host_domain {} of {}",
                    config.batch.contains(modality_t::image_k) ? "+image" : "",
                    config.batch.contains(modality_t::video_k) ? "+video" : "",
                    config.batch.contains(modality_t::audio_k) ? "+audio" : "",
                    topology.value_.shape.gpus_per_host_domain, host_domain_index, host_domain_count))
        report_(banner.view());

    // 2. The run's one telemetry ledger — constructed here, at the composition root, over a small
    //    all-visible arena (probes write their rows from any member's stream), and threaded by reference
    //    like the domains; a zero channel mask yields the inert disarmed instance. Then the trainer —
    //    built in place in this frame (its towers borrow the placed session tables, so it never moves).
    array<i32_t> local_devices;
    for (u32_t device = 0; device != topology.value_.shape.gpus_per_host_domain; ++device)
        if (local_devices.try_push_back(static_cast<i32_t>(device)) != status_t::success_k)
            return fail_out_("ledger device table allocation failed");
    expected<gpu_arena_t> ledger_arena = gpu_arena_t::make(0, 4u << 20, local_devices.view());
    if (!ledger_arena) return fail_out_(ledger_arena.description_.view());
    expected<telemetry_ledger_t> ledger_made = [&]() -> expected<telemetry_ledger_t> {
        // A transient bring-up stream orders the row table's zeroing; nothing of it is retained.
        expected<gpu_stream_t> bring_up = gpu_stream_t::make(0);
        if (!bring_up) return expected<telemetry_ledger_t>::fail(std::move(bring_up));
        return telemetry_ledger_t::make(ledger_arena.value_, config.telemetry_channels, 256u, bring_up.value_);
    }();
    if (!ledger_made) return fail_out_(ledger_made.description_.view());
    telemetry_ledger_t &ledger = ledger_made.value_;
    base_trainer_t trainer;
    if (expected_t built = make(trainer, config, topology.value_, ledger); !built)
        return fail_out_(built.description_.view());
    u32_t const gpus = static_cast<u32_t>(trainer.replicas.size());

    // 3. The one multimem / colocated surface and the optimizer — locals threaded by reference. The flat
    //    replica-major registration table is a construction-time transient: the optimizer copies the views into
    //    its own per-replica state. A contrastive driver would add its own payload to the domain sizing here.
    array<differentiable_weights_t> registrations;
    for (u32_t member_index = 0; member_index != gpus; ++member_index) {
        array<differentiable_weights_t> const &owned = trainer.replicas[member_index].weights;
        std::span<differentiable_weights_t const> const list {owned.view().data(), owned.size()};
        for (differentiable_weights_t const &entry : list)
            if (registrations.try_push_back(entry) != status_t::success_k)
                return fail_out_("registration table allocation failed");
    }
    std::size_t const weights_per_replica = trainer.replicas[0].weights.size();
    std::span<differentiable_weights_t const> const lead_list {trainer.replicas[0].weights.view().data(),
                                                               weights_per_replica};
    expected<multimem_domain_t> domain_made = multimem_domain_t::make(
        topology.value_, trainer.devices.view(), distributed_adafactor_t::payload_scalars_needed(lead_list),
        trainer.streams);
    if (!domain_made) return fail_out_(domain_made.description_.view());
    multimem_domain_t domain = std::move(domain_made.value_);
    expected<colocated_domain_t> colocated_made = colocated_domain_t::make(topology.value_);
    if (!colocated_made) return fail_out_(colocated_made.description_.view());
    colocated_domain_t colocated = std::move(colocated_made.value_);
    // The one cross-domain all-reduce buffer of the run — wired here, at the composition root, and
    // borrowed by the optimizer (which constructs no transport of its own).
    infiniband_all_reduce<f32_t> world_reduce;
    {
        switch_to_gpu_t gpu_scope {trainer.streams[0]};
        if (!gpu_scope.ok()) return fail_out_("context entry failed wiring the world all-reduce");
        expected<infiniband_all_reduce<f32_t>> world_reduce_made = infiniband_all_reduce<f32_t>::make(
            colocated, 0, trainer.arenas[0], distributed_adafactor_t::payload_scalars_needed(lead_list),
            trainer.streams[0]);
        if (!world_reduce_made) return fail_out_(world_reduce_made.description_.view());
        world_reduce = std::move(world_reduce_made.value_);
    }
    adafactor_config_t optimizer_config; // the per-step learning rate travels through `step`, not the config
    expected<distributed_adafactor_t> optimizer_made = distributed_adafactor_t::make(
        domain, world_reduce, trainer.arenas, {registrations.view().data(), registrations.size()}, weights_per_replica,
        optimizer_config, trainer.streams, ledger);
    if (!optimizer_made) return fail_out_(optimizer_made.description_.view());
    distributed_adafactor_t optimizer = std::move(optimizer_made.value_);

    // 4. Warm start under --resume: every replica uploads the same file's tensors (replicas are
    //    bitwise-identical by construction) and the run continues from the file's saved step, so the
    //    learning-rate schedule and the keyed data draws pick up exactly where the save left off. The
    //    load precedes the capture — captured graphs read the weights by address, never by value.
    u32_t resumed_step = 0;
    if (!config.resume.empty()) {
        switch_to_gpu_t lead_scope {trainer.streams[0]};
        if (!lead_scope.ok()) return fail_out_("context entry failed for the warm start");
        expected<checkpoint_to_load_t> loaded = checkpoint_to_load_t::make(config.resume, pinned_alloc_t {});
        if (!loaded) return fail_out_(loaded.description_.view());
        for (u32_t member_index = 0; member_index != gpus; ++member_index) {
            switch_to_gpu_t gpu_scope {trainer.streams[member_index]};
            if (expected_t warmed = load_checkpoint_(loaded.value_, trainer.replicas[member_index].weights,
                                                     trainer.streams[member_index]);
                !warmed)
                return fail_out_(warmed.description_.view());
        }
        // The uploads ride the pinned payload — drain before `loaded` releases it.
        for (u32_t member_index = 0; member_index != gpus; ++member_index) {
            switch_to_gpu_t gpu_scope {trainer.streams[member_index]};
            if (cuStreamSynchronize(trainer.streams[member_index].get()) != CUDA_SUCCESS)
                return fail_out_("warm-start drain failed");
        }
        resumed_step = static_cast<u32_t>(loaded.value_.step());
        sz::string_t line;
        if (format_into(line, "uform_base: resumed {} tensors from {} at step {}", loaded.value_.size(), config.resume,
                        resumed_step))
            report_(line.view());
    }

    // 5. Capture each replica's model region (the collective region stays eager); `--eager` skips it and
    //    the loop below re-enqueues the region inline.
    per_gpu_array<gpu_replay_graph_t> graphs;
    if (config.capture == replayed_k)
        if (expected_t captured = trainer.capture(graphs); !captured) return fail_out_(captured.description_.view());

    // 6. The steady loop — one whole two-region step per ordinal, telemetry on cadence.
    for (u32_t ordinal = resumed_step; ordinal != resumed_step + config.steps; ++ordinal) {
        if (expected_t stepped = trainer.step(optimizer_step_t {ordinal}, optimizer, graphs); !stepped) {
            sz::string_t line;
            if (!format_into(line, "step {} failed: {}", ordinal, stepped.description_.view())) return 1;
            return fail_out_(line.view());
        }
        if ((ordinal + 1u) % 10u == 0u) trainer.telemetry(domain, optimizer_step_t {ordinal}).report();
    }

    // 7. Checkpoint the lead replica (all replicas are bitwise-identical by construction) at the run's
    //    final step — a later --resume continues the schedule from here — then drain.
    if (!config.checkpoint.empty()) {
        if (expected_t saved = save_checkpoint_(config.checkpoint, trainer.replicas[0].weights,
                                                resumed_step + config.steps, trainer.streams[0]);
            !saved)
            return fail_out_(saved.description_.view());
        sz::string_t line;
        if (format_into(line, "uform_base: checkpoint written to {}", config.checkpoint)) report_(line.view());
    }
    for (u32_t member_index = 0; member_index != gpus; ++member_index) {
        switch_to_gpu_t gpu_scope {trainer.streams[member_index]};
        cuStreamSynchronize(trainer.streams[member_index].get());
    }
    {
        switch_to_gpu_t gpu_scope {trainer.streams[0]};
        if (expected_t reported = ledger.report(trainer.streams[0]); !reported) report_(reported.description_.view());
    }
    report_("uform_base: done");
    return 0;
}
