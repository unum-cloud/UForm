/**
 *  @file bench/core.cu
 *  @author Ash Vardanian
 *  @date October 19, 2026
 *  @brief GPU benchmarks of the core cell's training kernels at depth 6 and hidden 384.
 *
 *  Each row times device windows of back-to-back calls over keyed synthetic inputs:
 *
 *  - "core.tower.forward_backward": a text-only tower forward to the pooled vertices, and its
 *    backward.
 *  - "core.contrastive.loss": the rank-local contrastive loss and its forces over self-edges.
 *  - "core.contrastive.tokens": the late-interaction loss and its forces, 256 queries of 64 tokens
 *    against 256 documents of 1024 tokens at width 128, each query attracting one document
 *    under ranked BCE.
 */
#include <chrono> // `std::chrono::duration` — the CUDA-event windows

#include <cuda.h> // `cuInit`, `cuMemcpyHtoD`, `cuEventRecord`, `cuEventElapsedTime`

#include "uform/train/contrastive_segments.cuh" // `contrastive_segments_t`, `contrastive_relations_t`
#include "uform/train/contrastive_tokens.cuh"   // `contrastive_tokens_t`
#include "uform/train/encoder_tower.hpp"        // `encoder_tower_bf16_t`
#include "uform/utilities/types.hpp"            // `st::basic_vector`

#include "train/numerics.hpp" // `gpu_numerics_of`

#include "harness.hpp" // `loop_t`, `run`

namespace ue = unum::uform::encoder;
using ue::late_interaction_side_t;
using ue::modality_t;
using ue::multimodal_batch_on_gpu_t;
using ue::multimodal_batch_shape_t;
using ue::multimodal_batch_view_t;
namespace util = unum::uform::utilities;
using namespace unum::uform::train;
using namespace unum::uform::bench;
namespace ut = unum::uform::train;

/** Transformer blocks of the benchmarked tower. */
inline constexpr util::u32_t depth_k = 6;

/** Hidden width of the tower and of the contrastive vertices. */
inline constexpr util::u32_t hidden_k = 384;

/** Rows of the benchmarked batch, each holding @ref segments_per_row_k text segments. */
inline constexpr util::u32_t rows_k = 8, columns_k = 256, segments_per_row_k = 2;

/** Segments of the batch, which are also the vertices of the contrastive row. */
inline constexpr util::u32_t segments_k = rows_k * segments_per_row_k;

/** Uploads BFloat16 values in @b [-0.05,0.05) over @p device , keyed by @p salt . */
[[nodiscard]] static util::expected_t upload_keyed_(std::span<util::bf16_t> device, util::u64_t seed,
                                                    util::u64_t salt) noexcept {
    std::size_t const count = device.size();
    st::basic_vector<util::bf16_t> staging;
    if (st::failed(staging.resize(count)))
        return util::expected_t::fail(util::status_code_t::out_of_memory_k, "staging {} scalars failed", count);
    for (std::size_t index = 0; index != count; ++index) {
        util::f32_t const uniform = static_cast<util::f32_t>(util::splitmix64_hash_t {}(seed, salt, index) >> 40) *
                                    0x1.0p-24f;
        staging[index] = __float2bfloat16((uniform - 0.5f) * 0.1f);
    }
    if (cuMemcpyHtoD(util::device_address(device.data()), staging.data(), count * sizeof(util::bf16_t)) != CUDA_SUCCESS)
        return util::expected_t::fail(util::status_code_t::device_k, "uploading {} scalars failed", count);
    return {};
}

/**
 *  @brief Times @p launch on @p stream until @p loop ends, in windows of back-to-back launches
 *      between two CUDA events, doubling a window while it lasts under a millisecond.
 *
 *  @p launch enqueues one call and answers its @c expected_t ; the first refusal fails the loop.
 */
template <typename launch_type_>
static void time_windows(loop_t& loop, CUstream stream, launch_type_&& launch) noexcept {
    CUevent start = nullptr, stop = nullptr;
    if (cuEventCreate(&start, CU_EVENT_DEFAULT) != CUDA_SUCCESS ||
        cuEventCreate(&stop, CU_EVENT_DEFAULT) != CUDA_SUCCESS)
        return loop.fail("CUDA event creation failed");
    std::size_t window = 1;
    for ([[maybe_unused]] std::size_t call : loop) {
        if (cuEventRecord(start, stream) != CUDA_SUCCESS) {
            loop.fail("CUDA event record failed");
            break;
        }
        for (std::size_t index = 0; index != window; ++index)
            if (util::expected_t launched = launch(); !launched) {
                loop.fail(launched.description_.view());
                break;
            }
        if (loop.failed()) break;
        util::f32_t milliseconds = 0;
        if (cuEventRecord(stop, stream) != CUDA_SUCCESS || cuEventSynchronize(stop) != CUDA_SUCCESS ||
            cuEventElapsedTime(&milliseconds, start, stop) != CUDA_SUCCESS) {
            loop.fail("CUDA event timing failed");
            break;
        }
        std::chrono::duration<util::f32_t, std::milli> const elapsed {milliseconds};
        loop.add_window(elapsed, window);
        if (elapsed < std::chrono::milliseconds(1)) window *= 2;
    }
    [[maybe_unused]] CUresult const start_destroyed = cuEventDestroy(start);
    [[maybe_unused]] CUresult const stop_destroyed = cuEventDestroy(stop);
}

/** One text-only tower forward to the pooled vertices and its backward from a keyed force. */
static void tower_forward_backward(loop_t& loop, util::u64_t seed) {
    auto [stream, make_stream_status] = util::gpu_stream_t::make(0);
    if (!make_stream_status) return loop.fail(make_stream_status.description_.view());
    util::switch_to_gpu_t gpu_scope {stream};
    if (!gpu_scope.ok()) return loop.fail("context entry failed");
    auto [arena, make_arena_status] = util::gpu_arena_t::make(0, 1ull << 30);
    if (!make_arena_status) return loop.fail(make_arena_status.description_.view());

    encoder_tower_config_t config;
    config.vocabulary = 258;
    config.depth = depth_k;
    config.block.hidden = hidden_k;
    config.block.max_tokens = rows_k * columns_k;
    config.block.max_segments = segments_k;
    config.block.max_segment_tokens = columns_k;
    config.rope_max_positions = columns_k;
    multimodal_batch_shape_t const shape {rows_k, segments_k, columns_k};
    nk_capability_t capabilities = 0;
    if (nk_capabilities_enabled_cuda(0, &capabilities) != nk_success_k) return loop.fail("capability query failed");
    auto [tower, make_tower_status] = encoder_tower_bf16_t::make(arena, config, shape,
                                                                 ut::gpu_numerics_of(capabilities));
    if (!make_tower_status) return loop.fail(make_tower_status.description_.view());
    st::basic_vector<util::differentiable_weights_t> weights;
    if (util::expected_t filled = tower.fill(weights); !filled) return loop.fail(filled.description_.view());
    for (std::size_t matrix = 0; matrix != weights.size(); ++matrix)
        if (util::expected_t seeded = upload_keyed_(
                {weights[matrix].weights.data(), weights[matrix].weights.extent(0) * weights[matrix].weights.extent(1)},
                seed, matrix);
            !seeded)
            return loop.fail(seeded.description_.view());

    auto [batch, make_batch_status] = multimodal_batch_on_gpu_t::make(util::gpu_arena_alloc<util::b8_t> {arena}, shape);
    if (!make_batch_status) return loop.fail(make_batch_status.description_.view());
    util::u32_t const segment_tokens = columns_k / segments_per_row_k;
    util::u32_t offsets[segments_k + 1], lengths[segments_k];
    modality_t modalities[segments_k];
    for (util::u32_t segment = 0; segment != segments_k; ++segment) {
        offsets[segment] = segment * segment_tokens;
        lengths[segment] = segment_tokens;
        modalities[segment] = modality_t::text_k;
    }
    offsets[segments_k] = segments_k * segment_tokens;
    st::basic_vector<util::token_id_t> tokens;
    if (st::failed(tokens.resize(rows_k * columns_k))) return loop.fail("token staging failed");
    for (std::size_t token = 0; token != tokens.size(); ++token)
        tokens[token] = static_cast<util::token_id_t>(util::splitmix64_hash_t {}(seed, token) % 256u);
    if (cuMemcpyHtoD(util::device_address(batch.segment_offsets.data()), offsets, sizeof(offsets)) != CUDA_SUCCESS ||
        cuMemcpyHtoD(util::device_address(batch.segment_lengths.data()), lengths, sizeof(lengths)) != CUDA_SUCCESS ||
        cuMemcpyHtoD(util::device_address(batch.segment_modality.data()), modalities, sizeof(modalities)) !=
            CUDA_SUCCESS ||
        cuMemcpyHtoD(util::device_address(batch.tokens.data()), tokens.data(),
                     tokens.size() * sizeof(util::token_id_t)) != CUDA_SUCCESS)
        return loop.fail("batch upload failed");

    util::bf16_t* vertices = arena.allocate_object<util::bf16_t>(segments_k * hidden_k);
    util::bf16_t* vertices_grad = arena.allocate_object<util::bf16_t>(segments_k * hidden_k);
    if (vertices == nullptr || vertices_grad == nullptr) return loop.fail("vertex allocation failed");
    if (util::expected_t seeded = upload_keyed_({vertices_grad, segments_k * hidden_k}, seed, weights.size()); !seeded)
        return loop.fail(seeded.description_.view());
    if (util::expected_t packed = tower.pack_weights(stream); !packed) return loop.fail(packed.description_.view());

    time_windows(loop, stream.get(), [&]() noexcept -> expected_t {
        util::expected_t ran = tower.forward(batch, util::bf16r2_span_t {vertices, segments_k, hidden_k}, stream);
        if (ran) ran = tower.backward(util::bf16r2_view_t {vertices_grad, segments_k, hidden_k}, {}, batch, stream);
        return ran;
    });
}

/** One rank-local contrastive loss and its forces over self-edge relations. */
static void contrastive_loss(loop_t& loop, util::u64_t seed) {
    auto [stream, make_stream_status] = util::gpu_stream_t::make(0);
    if (!make_stream_status) return loop.fail(make_stream_status.description_.view());
    util::switch_to_gpu_t gpu_scope {stream};
    if (!gpu_scope.ok()) return loop.fail("context entry failed");
    auto [arena, make_arena_status] = util::gpu_arena_t::make(0, 1ull << 30);
    if (!make_arena_status) return loop.fail(make_arena_status.description_.view());

    contrastive_segments_config_t config;
    config.from_vertices_per_gpu = segments_k;
    config.hidden = hidden_k;
    config.to_vertices_per_gpu = segments_k;
    config.gpu_subgraphs_per_step = 1;
    util::u32_t const prefix_dims[1] = {hidden_k};
    nk_capability_t capabilities = 0;
    if (nk_capabilities_enabled_cuda(0, &capabilities) != nk_success_k) return loop.fail("capability query failed");
    auto [loss, make_loss_status] = contrastive_segments_t::make(arena, config, {prefix_dims, 1},
                                                                 ut::gpu_numerics_of(capabilities));
    if (!make_loss_status) return loop.fail(make_loss_status.description_.view());
    auto [relations, make_relations_status] = contrastive_relations_t::make(arena, segments_k, 16);
    if (!make_relations_status) return loop.fail(make_relations_status.description_.view());
    if (util::expected_t staged = relations.upload_self_edges(segments_k, {}, {}, 0, 1.0f, stream); !staged)
        return loop.fail(staged.description_.view());

    std::size_t const scalars = std::size_t(segments_k) * hidden_k;
    util::bf16_t* tower = arena.allocate_object<util::bf16_t>(scalars);
    util::bf16_t* grad = arena.allocate_object<util::bf16_t>(scalars);
    util::f32_t* column_grad = arena.allocate_object<util::f32_t>(scalars);
    util::f32_t* loss_out = arena.allocate_object<util::f32_t>(1);
    if (tower == nullptr || grad == nullptr || column_grad == nullptr || loss_out == nullptr)
        return loop.fail("vertex allocation failed");
    if (util::expected_t seeded = upload_keyed_({tower, scalars}, seed, 0); !seeded)
        return loop.fail(seeded.description_.view());
    gpu_subgraph_t subgraph;
    subgraph.to_vertices = util::bf16r2_view_t {tower, segments_k, hidden_k};
    subgraph.to_vertices_grad = util::f32r2_span_t {column_grad, segments_k, hidden_k};

    time_windows(loop, stream.get(), [&]() noexcept -> expected_t {
        return loss(util::bf16r2_view_t {tower, segments_k, hidden_k}, {&subgraph, 1}, relations.view(),
                    contrastive_recipe_t {}, util::f32r1_span_t {loss_out, 1},
                    util::bf16r2_span_t {grad, segments_k, hidden_k}, stream);
    });
}

/** One single-GPU late-interaction loss and its forces at the reranker shape of the plan. */
static void contrastive_tokens(loop_t& loop, util::u64_t seed) {
    constexpr util::u32_t queries_k = 256, query_tokens_k = 64, documents_k = 256, document_tokens_k = 1024,
                          width_k = 128;
    constexpr util::u32_t segments = queries_k + documents_k;
    constexpr util::u32_t head_rows = queries_k * query_tokens_k + documents_k * document_tokens_k;
    auto [stream, make_stream_status] = util::gpu_stream_t::make(0);
    if (!make_stream_status) return loop.fail(make_stream_status.description_.view());
    util::switch_to_gpu_t gpu_scope {stream};
    if (!gpu_scope.ok()) return loop.fail("context entry failed");
    auto [arena, make_arena_status] = util::gpu_arena_t::make(0, 4ull << 30);
    if (!make_arena_status) return loop.fail(make_arena_status.description_.view());

    contrastive_tokens_config_t config;
    config.segments_per_gpu = segments;
    config.head_tokens_per_gpu = head_rows;
    config.query_tokens_per_gpu = queries_k * query_tokens_k;
    config.documents_per_gpu = documents_k;
    config.document_tokens_per_gpu = documents_k * document_tokens_k;
    config.band_tokens = document_tokens_k;
    config.width = width_k;
    nk_capability_t capabilities = 0;
    if (nk_capabilities_enabled_cuda(0, &capabilities) != nk_success_k) return loop.fail("capability query failed");
    auto [loss, make_loss_status] = contrastive_tokens_t::make(arena, config, ut::gpu_numerics_of(capabilities));
    if (!make_loss_status) return loop.fail(make_loss_status.description_.view());

    // Query i attracts document i; every other document repels it at the ambient weight.
    st::basic_vector<util::u32_t> offsets, attract_offsets, attract_ids, repel_offsets;
    st::basic_vector<util::f32_t> attract_scores;
    if (st::failed(offsets.resize(segments + 1u)) || st::failed(attract_offsets.resize(segments + 1u)) ||
        st::failed(repel_offsets.resize(segments + 1u)) || st::failed(attract_ids.resize(queries_k)) ||
        st::failed(attract_scores.resize(queries_k)))
        return loop.fail("host staging failed");
    st::basic_vector<late_interaction_side_t> sides;
    if (st::failed(sides.resize(segments))) return loop.fail("host staging failed");
    for (util::u32_t segment = 0; segment != segments; ++segment) {
        bool const query = segment < queries_k;
        sides[segment] = query ? late_interaction_side_t::query_k : late_interaction_side_t::document_k;
        offsets[segment + 1u] = offsets[segment] + (query ? query_tokens_k : document_tokens_k);
        if (query) attract_ids[segment] = queries_k + segment, attract_scores[segment] = 1.0f;
        attract_offsets[segment + 1u] = fu::min_of(segment + 1u, queries_k);
        repel_offsets[segment + 1u] = 0;
    }
    std::size_t const scalars = std::size_t(head_rows) * width_k;
    util::bf16_t* vectors = arena.allocate_object<util::bf16_t>(scalars);
    util::bf16_t* grad = arena.allocate_object<util::bf16_t>(scalars);
    util::u32_t* offsets_device = arena.allocate_object<util::u32_t>(segments + 1u);
    late_interaction_side_t* sides_device = arena.allocate_object<late_interaction_side_t>(segments);
    util::f32_t* loss_out = arena.allocate_object<util::f32_t>(1);
    if (vectors == nullptr || grad == nullptr || offsets_device == nullptr || sides_device == nullptr ||
        loss_out == nullptr)
        return loop.fail("token allocation failed");
    if (util::expected_t seeded = upload_keyed_({vectors, scalars}, seed, 1); !seeded)
        return loop.fail(seeded.description_.view());
    cuMemcpyHtoD(util::device_address(offsets_device), offsets.data(), (segments + 1u) * sizeof(util::u32_t));
    cuMemcpyHtoD(util::device_address(sides_device), sides.data(), segments * sizeof(late_interaction_side_t));
    auto [relations, make_relations_status] = contrastive_relations_t::make(arena, segments, queries_k);
    if (!make_relations_status) return loop.fail(make_relations_status.description_.view());
    if (util::expected_t staged = relations.upload(
            {std::span<util::u32_t const> {attract_offsets}, std::span<util::u32_t const> {attract_ids},
             std::span<util::f32_t const> {attract_scores}},
            {std::span<util::u32_t const> {repel_offsets}, {}, {}}, {}, {}, {}, {}, {}, segments, 0, 1.0f, stream);
        !staged)
        return loop.fail(staged.description_.view());
    multimodal_batch_view_t batch;
    batch.segment_side = nk::tensor_view<late_interaction_side_t, 1> {sides_device, segments};
    contrastive_recipe_t recipe;
    recipe.inverse_temperature = 10.0f;

    time_windows(loop, stream.get(), [&]() noexcept -> expected_t {
        return loss(util::bf16r2_view_t {vectors, head_rows, width_k},
                    util::u32r1_view_t {offsets_device, segments + 1u}, batch, relations.view(), recipe,
                    util::f32r1_span_t {loss_out, 1}, util::bf16r2_span_t {grad, head_rows, width_k}, stream);
    });
}

int main(int argc, char** argv) {
    bench_settings_t const settings = read_bench_settings(argc, argv);
    print(probe_machine());
    print(settings);
    int device_count = 0;
    if (cuInit(0) != CUDA_SUCCESS || cuDeviceGetCount(&device_count) != CUDA_SUCCESS || device_count == 0)
        return fmt::println(stderr, "no GPU device"), 1;
    bool const tower_failed = run(settings, "core.tower.forward_backward", tower_forward_backward, settings.seed);
    bool const loss_failed = run(settings, "core.contrastive.loss", contrastive_loss, settings.seed);
    bool const tokens_failed = run(settings, "core.contrastive.tokens", contrastive_tokens, settings.seed);
    return tower_failed || loss_failed || tokens_failed ? 1 : 0;
}
