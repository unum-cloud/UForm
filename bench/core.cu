/**
 *  @file bench/core.cu
 *  @author Ash Vardanian
 *  @date October 19, 2026
 *  @brief GPU benchmarks of the core cell's training kernels at depth 6 and hidden 384.
 *
 *  Each row times device windows of back-to-back calls over keyed synthetic inputs:
 *
 *  - "tower/forward_backward": a text-only tower forward to the pooled vertices and its backward.
 *  - "contrastive/loss": the rank-local contrastive loss and its forces over self-edges.
 */
#include <cuda.h> // `cuInit`, `cuMemcpyHtoD`

#include "core/array.hpp"                // `array`
#include "core/contrastive_segments.cuh" // `contrastive_segments_t`, `contrastive_relations_t`
#include "core/encoder_tower.hpp"        // `encoder_tower_bf16_t`
#include "harness.hpp"                   // `loop_t`, `time_windows`, `run`

using namespace unum::uform::core;
using namespace unum::uform::bench;

/** Transformer blocks of the benchmarked tower. */
inline constexpr u32_t depth_k = 6;

/** Hidden width of the tower and of the contrastive vertices. */
inline constexpr u32_t hidden_k = 384;

/** Rows of the benchmarked batch, each holding @ref segments_per_row_k text segments. */
inline constexpr u32_t rows_k = 8, columns_k = 256, segments_per_row_k = 2;

/** Segments of the batch, which are also the vertices of the contrastive row. */
inline constexpr u32_t segments_k = rows_k * segments_per_row_k;

/** Uploads @p count BFloat16 values in @b [-0.05,0.05) to @p device , keyed by @p salt . */
[[nodiscard]] static expected_t upload_keyed_(bf16_t *device, std::size_t count, u64_t seed, u64_t salt) noexcept {
    array<bf16_t> staging;
    if (staging.resize(count) != status_t::success_k)
        return expected_t::fail(status_t::bad_alloc_k, "staging {} scalars failed", count);
    for (std::size_t index = 0; index != count; ++index) {
        f32_t const uniform = static_cast<f32_t>(splitmix64_hash_t {}(seed, salt, index) >> 40) * 0x1.0p-24f;
        staging[index] = __float2bfloat16((uniform - 0.5f) * 0.1f);
    }
    if (cuMemcpyHtoD(device_address(device), staging.data(), count * sizeof(bf16_t)) != CUDA_SUCCESS)
        return expected_t::fail(status_t::cuda_error_k, "uploading {} scalars failed", count);
    return {};
}

/** One text-only tower forward to the pooled vertices and its backward from a keyed force. */
static void tower_forward_backward(loop_t &loop, u64_t seed) {
    expected<gpu_stream_t> stream = gpu_stream_t::make(0);
    if (!stream) return loop.fail(stream.description_.view());
    switch_to_gpu_t gpu_scope {*stream};
    if (!gpu_scope.ok()) return loop.fail("context entry failed");
    expected<gpu_arena_t> arena = gpu_arena_t::make(0, 1ull << 30);
    if (!arena) return loop.fail(arena.description_.view());
    expected<cudnn_session_t> cudnn = cudnn_session_t::make(algo_pick_t::heuristic_k);
    if (!cudnn) return loop.fail(cudnn.description_.view());

    encoder_tower_config_t config;
    config.vocabulary = 258;
    config.depth = depth_k;
    config.block.hidden = hidden_k;
    config.block.max_tokens = rows_k * columns_k;
    config.block.max_segments = segments_k;
    config.block.max_segment_tokens = columns_k;
    config.rope_max_positions = columns_k;
    multimodal_batch_shape_t const shape {rows_k, segments_k, columns_k};
    expected<encoder_tower_bf16_t> tower = encoder_tower_bf16_t::make(*arena, *cudnn, config, shape);
    if (!tower) return loop.fail(tower.description_.view());
    array<differentiable_weights_t> weights;
    if (expected_t filled = tower->fill(weights); !filled) return loop.fail(filled.description_.view());
    for (std::size_t matrix = 0; matrix != weights.size(); ++matrix)
        if (expected_t seeded = upload_keyed_(weights[matrix].weights.data(),
                                              weights[matrix].weights.extent(0) * weights[matrix].weights.extent(1),
                                              seed, matrix);
            !seeded)
            return loop.fail(seeded.description_.view());

    expected<multimodal_batch_on_gpu_t> batch = multimodal_batch_on_gpu_t::make(gpu_arena_alloc<b8_t> {*arena}, shape);
    if (!batch) return loop.fail(batch.description_.view());
    u32_t const segment_tokens = columns_k / segments_per_row_k;
    u32_t offsets[segments_k + 1], lengths[segments_k];
    modality_t modalities[segments_k];
    for (u32_t segment = 0; segment != segments_k; ++segment) {
        offsets[segment] = segment * segment_tokens;
        lengths[segment] = segment_tokens;
        modalities[segment] = modality_t::text_k;
    }
    offsets[segments_k] = segments_k * segment_tokens;
    array<token_id_t> tokens;
    if (tokens.resize(rows_k * columns_k) != status_t::success_k) return loop.fail("token staging failed");
    for (std::size_t token = 0; token != tokens.size(); ++token)
        tokens[token] = static_cast<token_id_t>(splitmix64_hash_t {}(seed, token) % 256u);
    if (cuMemcpyHtoD(device_address(batch->segment_offsets.data()), offsets, sizeof(offsets)) != CUDA_SUCCESS ||
        cuMemcpyHtoD(device_address(batch->segment_lengths.data()), lengths, sizeof(lengths)) != CUDA_SUCCESS ||
        cuMemcpyHtoD(device_address(batch->segment_modality.data()), modalities, sizeof(modalities)) != CUDA_SUCCESS ||
        cuMemcpyHtoD(device_address(batch->tokens.data()), tokens.data(), tokens.size() * sizeof(token_id_t)) !=
            CUDA_SUCCESS)
        return loop.fail("batch upload failed");

    bf16_t *vertices = arena->allocate_object<bf16_t>(segments_k * hidden_k);
    bf16_t *vertices_grad = arena->allocate_object<bf16_t>(segments_k * hidden_k);
    if (vertices == nullptr || vertices_grad == nullptr) return loop.fail("vertex allocation failed");
    if (expected_t seeded = upload_keyed_(vertices_grad, segments_k * hidden_k, seed, weights.size()); !seeded)
        return loop.fail(seeded.description_.view());
    if (expected_t packed = tower->pack_weights(*stream); !packed) return loop.fail(packed.description_.view());

    time_windows(loop, stream->get(), [&]() noexcept -> expected_t {
        expected_t ran = tower->forward(*batch, bf16r2_span_t {vertices, segments_k, hidden_k}, *stream);
        if (ran) ran = tower->backward(bf16r2_view_t {vertices_grad, segments_k, hidden_k}, {}, *batch, *stream);
        return ran;
    });
}

/** One rank-local contrastive loss and its forces over self-edge relations. */
static void contrastive_loss(loop_t &loop, u64_t seed) {
    expected<gpu_stream_t> stream = gpu_stream_t::make(0);
    if (!stream) return loop.fail(stream.description_.view());
    switch_to_gpu_t gpu_scope {*stream};
    if (!gpu_scope.ok()) return loop.fail("context entry failed");
    expected<gpu_arena_t> arena = gpu_arena_t::make(0, 1ull << 30);
    if (!arena) return loop.fail(arena.description_.view());

    contrastive_segments_config_t config;
    config.from_vertices_per_gpu = segments_k;
    config.hidden = hidden_k;
    config.to_vertices_per_gpu = segments_k;
    config.gpu_subgraphs_per_step = 1;
    u32_t const prefix_dims[1] = {hidden_k};
    expected<contrastive_segments_t> loss = contrastive_segments_t::make(*arena, config, {prefix_dims, 1});
    if (!loss) return loop.fail(loss.description_.view());
    expected<contrastive_relations_t> relations = contrastive_relations_t::make(*arena, segments_k, 16);
    if (!relations) return loop.fail(relations.description_.view());
    if (expected_t staged = relations->upload_self_edges(segments_k, {}, {}, 0, 1.0f, *stream); !staged)
        return loop.fail(staged.description_.view());

    std::size_t const scalars = std::size_t(segments_k) * hidden_k;
    bf16_t *tower = arena->allocate_object<bf16_t>(scalars);
    bf16_t *grad = arena->allocate_object<bf16_t>(scalars);
    f32_t *column_grad = arena->allocate_object<f32_t>(scalars);
    f32_t *loss_out = arena->allocate_object<f32_t>(1);
    if (tower == nullptr || grad == nullptr || column_grad == nullptr || loss_out == nullptr)
        return loop.fail("vertex allocation failed");
    if (expected_t seeded = upload_keyed_(tower, scalars, seed, 0); !seeded)
        return loop.fail(seeded.description_.view());
    gpu_subgraph_t subgraph;
    subgraph.to_vertices = bf16r2_view_t {tower, segments_k, hidden_k};
    subgraph.to_vertices_grad = f32r2_span_t {column_grad, segments_k, hidden_k};

    time_windows(loop, stream->get(), [&]() noexcept -> expected_t {
        return (*loss)(bf16r2_view_t {tower, segments_k, hidden_k}, {&subgraph, 1}, relations->view(),
                       contrastive_recipe_t {}, segments_k, f32r1_span_t {loss_out, 1},
                       bf16r2_span_t {grad, segments_k, hidden_k}, *stream);
    });
}

int main() {
    settings_t const settings = read_settings();
    print(settings);
    int device_count = 0;
    if (cuInit(0) != CUDA_SUCCESS || cuDeviceGetCount(&device_count) != CUDA_SUCCESS || device_count == 0)
        return fmt::println(stderr, "no CUDA device detected"), 1;
    bool const tower_failed = run(settings, "tower/forward_backward", tower_forward_backward, settings.seed);
    bool const loss_failed = run(settings, "contrastive/loss", contrastive_loss, settings.seed);
    return tower_failed || loss_failed ? 1 : 0;
}
