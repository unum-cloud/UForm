/**
 *  @file test/contrastive.cu
 *  @author Ash Vardanian
 *  @date July 3, 2026
 *  @brief Conformance test of the contrastive force layout: the loss object against a Float64 host
 *      oracle, finite-difference force checks, capture-replay, relation-graph negatives, and the
 *      multi-GPU distributed step against the single-GPU global reference.
 *
 *  Tiers scale with the visible hardware: the single-GPU oracle matrix runs on one GPU; the
 *  distributed-step tier joins when at least two GPUs are visible.
 */
#include <cmath>   // `std::fabs`, `std::sqrt`, `std::exp`, `std::log1p`
#include <cstdlib> // `std::getenv`, `std::atoi`, `std::atof`

#include <span>    // `std::span` — the oracle's borrowed towers
#include <utility> // `std::move`
#include <vector>  // `std::vector`

#include <cuda.h> // `cuInit`, `cuCtxPushCurrent`, `cuMemcpyHtoD`, `cuMemsetD8` — the raw staging boundary

#include "core/infiniband_collectives.cuh" // `infiniband_coordinates_t` — the corrupt-payload negative test
#include "core/contrastive_segments.cuh"
#include "core/encoder_tower.hpp" // `encoder_tower_bf16_t` — the pool + head the FP64 parity case gates
#include "harness.hpp"            // `test_suite_t`

using namespace unum::uform::core;
using namespace unum::uform::test;

/** Whether the driver shows at least @p wanted devices. */
[[nodiscard]] static bool cuda_devices_visible_(int wanted) noexcept {
    int device_count = 0;
    return cuDeviceGetCount(&device_count) == CUDA_SUCCESS && device_count >= wanted;
}

/** Deterministic pseudo-random filler in roughly [-0.5, 0.5). */
static f32_t hash_fill(std::size_t index, std::size_t salt) {
    std::size_t const mixed = (index * 2654435761ull + salt * 40503ull + 12345ull) % 1000ull;
    return static_cast<f32_t>(mixed) / 1000.0f - 0.5f;
}

/** The host-side pair resolver — mirrors `contrastive_pair_relation` decision for decision. */
struct host_relations_t {
    std::vector<u32_t> attract_offsets, attract_ids, repel_offsets, repel_ids, ignore_offsets, ignore_ids;
    std::vector<f32_t> attract_scores, repel_scores;
    std::vector<u32_t> vertex_effective, neighbor_effective;
    std::size_t self_edge_base = 0;
    f32_t ambient = 1.0f;

    struct resolved_edge_t {
        f64_t sign, score;
    };
    [[nodiscard]] resolved_edge_t resolve(u32_t vertex, u32_t neighbor_global) const {
        if (!vertex_effective.empty() && vertex_effective[vertex] == 0) return {0.0, 0.0};
        if (!neighbor_effective.empty() && neighbor_effective[neighbor_global] == 0) return {0.0, 0.0};
        auto in_run = [&](std::vector<u32_t> const &offsets, std::vector<u32_t> const &ids,
                          std::size_t *found_position) -> bool {
            if (offsets.empty()) return false;
            for (u32_t position = offsets[vertex]; position != offsets[vertex + 1]; ++position)
                if (ids[position] == neighbor_global) {
                    if (found_position) *found_position = position;
                    return true;
                }
            return false;
        };
        if (in_run(ignore_offsets, ignore_ids, nullptr)) return {0.0, 0.0};
        if (attract_ids.empty())
            return neighbor_global == self_edge_base + vertex ? resolved_edge_t {1.0, 1.0}
                                                              : resolved_edge_t {-1.0, f64_t(ambient)};
        std::size_t position = 0;
        if (in_run(attract_offsets, attract_ids, &position)) return {1.0, f64_t(attract_scores[position])};
        if (in_run(repel_offsets, repel_ids, &position)) return {-1.0, f64_t(repel_scores[position])};
        return {-1.0, f64_t(ambient)};
    }
};

/** The Float64 oracle: the exact per-header math of the segments loss over the @b [vertices,hidden]
 *  and @b [neighbors,hidden] towers, held on the BFloat16 grid. */
static f64_t host_loss(f64r2_view_t vertices, f64r2_view_t neighbors, std::span<u32_t const> prefixes,
                       host_relations_t const &relations, f64_t beta, f64_t bias, std::size_t vertices_per_mesh) {
    std::size_t const num_vertices = vertices.extent(0), num_neighbors = neighbors.extent(0);
    f64_t loss = 0.0;
    for (std::size_t vertex = 0; vertex != num_vertices; ++vertex) {
        // Signed weight totals over the whole neighbor axis.
        f64_t pos_weight = 0.0, neg_weight = 0.0;
        for (std::size_t neighbor = 0; neighbor != num_neighbors; ++neighbor) {
            auto [target, weight] = relations.resolve(u32_t(vertex), u32_t(neighbor));
            if (target > 0) pos_weight += weight;
            else if (target < 0) neg_weight += weight;
        }
        f64_t const pos_scale = 1.0 / std::max(pos_weight, f64_t(1.0e-12));
        f64_t const neg_scale = 1.0 / std::max(neg_weight, f64_t(1.0e-12));
        f64_t vertex_loss = 0.0;
        for (std::size_t prefix_index = 0; prefix_index != prefixes.size(); ++prefix_index) {
            auto const prefix = nk::range(0, prefixes[prefix_index]);
            f64r1_view_t const ours = vertices[vertex, prefix, nk::slice];
            f64_t const vertex_norm = std::sqrt(dot(ours, ours));
            for (std::size_t neighbor = 0; neighbor != num_neighbors; ++neighbor) {
                auto [target, weight] = relations.resolve(u32_t(vertex), u32_t(neighbor));
                if (weight == 0.0) continue;
                f64r1_view_t const theirs = neighbors[neighbor, prefix, nk::slice];
                f64_t const raw = dot(ours, theirs), neighbor_norm = std::sqrt(dot(theirs, theirs));
                f64_t const denominator = vertex_norm * neighbor_norm;
                f64_t const cosine = denominator > 1.0e-2 ? raw / denominator : 0.0;
                f64_t const logit = beta * cosine + bias;
                f64_t const negated = -target * logit;
                f64_t const softplus = std::max(negated, 0.0) + std::log1p(std::exp(-std::fabs(negated)));
                vertex_loss += weight * softplus * (target > 0 ? pos_scale : neg_scale);
            }
        }
        loss += vertex_loss;
    }
    return loss / f64_t(vertices_per_mesh);
}

/** One single-GPU oracle case: run the loss, compare to the host oracle, finite-difference the force on a few
 *  sampled channels, and check captured replays land where the eager run did. */
static void run_oracle_case(test_suite_t &suite, std::size_t num_vertices, std::size_t hidden,
                            std::vector<u32_t> const &prefixes, bool explicit_edges, bool with_pad,
                            f32_t ambient_weight) {
    if (!cuda_devices_visible_(1)) return suite.skip(no_cuda_device_k);
    auto stream_made = gpu_stream_t::make(0);
    if (!stream_made) return suite.check(false, "stream: {}", stream_made.description_);
    gpu_stream_t stream = std::move(stream_made.value_);
    cuCtxPushCurrent(stream.context());
    auto arena_made = gpu_arena_t::make(0, 1ull << 30);
    if (!arena_made) return suite.check(false, "arena: {}", arena_made.description_);
    gpu_arena_t arena = std::move(arena_made.value_);

    contrastive_segments_config_t config;
    config.from_vertices_per_gpu = u32_t(num_vertices);
    config.hidden = u32_t(hidden);
    config.to_vertices_per_gpu = u32_t(num_vertices);
    config.gpu_subgraphs_per_step = 1;
    auto loss_made = contrastive_segments_t::make(arena, config, std::span {prefixes});
    if (!loss_made) return suite.check(false, "loss make: {}", loss_made.description_);
    contrastive_segments_t loss = std::move(loss_made.value_);

    // The tower on the BFloat16 grid in Float64 for the oracle, its bit patterns for the device.
    std::size_t const scalars = num_vertices * hidden;
    std::vector<f64_t> host_tower(scalars);
    std::vector<nk_bf16_t> staging(scalars);
    for (std::size_t index = 0; index != scalars; ++index) host_tower[index] = hash_fill(index, num_vertices + hidden);
    if (!narrow_to_bf16(host_tower, staging)) return suite.check(false, "tower rounding failed");
    bf16_t *tower = arena.allocate_object<bf16_t>(scalars);
    bf16_t *grad = arena.allocate_object<bf16_t>(scalars);
    f32_t *grad_f32 = arena.allocate_object<f32_t>(scalars);
    f32_t *loss_out = arena.allocate_object<f32_t>(1);
    u32_t *effective = arena.allocate_object<u32_t>(num_vertices);
    cuMemcpyHtoD(device_address(tower), staging.data(), scalars * sizeof(nk_bf16_t));

    // Relations: self-edge default, or explicit attract/repel with an ignore override; PAD via the
    // effective-length columns.
    host_relations_t host_relations;
    host_relations.ambient = ambient_weight;
    auto relations_made = contrastive_relations_t::make(arena, u32_t(num_vertices), 16);
    if (!relations_made) return suite.check(false, "relations make: {}", relations_made.description_);
    contrastive_relations_t relations = std::move(relations_made.value_);
    std::vector<u32_t> host_effective(num_vertices, 1u);
    if (with_pad) host_effective[num_vertices - 1] = 0u;
    cuMemcpyHtoD(device_address(effective), host_effective.data(), num_vertices * sizeof(u32_t));
    u32r1_view_t const effective_view {effective, with_pad ? num_vertices : 0};
    if (with_pad) {
        host_relations.vertex_effective = host_effective;
        host_relations.neighbor_effective = host_effective;
    }
    expected_t staged {};
    if (explicit_edges) {
        // Vertex 0 attracts {1 (w=2)}, ignores {0}; vertex 1 repels {0 (w=5)}; the rest are ambient.
        host_relations.attract_offsets = {0, 1};
        host_relations.attract_ids = {1};
        host_relations.attract_scores = {2.0f};
        host_relations.repel_offsets = {0, 0, 1};
        host_relations.repel_ids = {0};
        host_relations.repel_scores = {5.0f};
        host_relations.ignore_offsets = {0, 1};
        host_relations.ignore_ids = {0};
        for (std::size_t vertex = 1; vertex != num_vertices; ++vertex) {
            host_relations.attract_offsets.push_back(u32_t(host_relations.attract_ids.size()));
            if (vertex >= 2) host_relations.repel_offsets.push_back(u32_t(host_relations.repel_ids.size()));
            host_relations.ignore_offsets.push_back(u32_t(host_relations.ignore_ids.size()));
        }
        staged = relations.upload(std::span {host_relations.attract_offsets}, std::span {host_relations.attract_ids},
                                  std::span {host_relations.attract_scores}, std::span {host_relations.repel_offsets},
                                  std::span {host_relations.repel_ids}, std::span {host_relations.repel_scores},
                                  std::span {host_relations.ignore_offsets}, std::span {host_relations.ignore_ids},
                                  effective_view, effective_view, num_vertices, 0, ambient_weight, stream);
    }
    else
        staged = relations.upload_self_edges(u32_t(num_vertices), effective_view, effective_view, 0, ambient_weight,
                                             stream);
    if (!staged) return suite.check(false, "relations upload: {}", staged.description_);

    gpu_subgraph_t subgraph;
    subgraph.to_vertices = bf16r2_view_t {tower, num_vertices, hidden};
    subgraph.global_offset = 0;
    subgraph.to_vertices_grad = f32r2_span_t {grad_f32, num_vertices, hidden};
    contrastive_recipe_t recipe;
    recipe.ambient_repulsion_weight = ambient_weight;

    auto run_once = [&]() noexcept -> expected_t {
        return loss(bf16r2_view_t {tower, num_vertices, hidden}, {&subgraph, 1}, relations.view(), recipe,
                    f32r1_span_t {loss_out, 1}, bf16r2_span_t {grad, num_vertices, hidden}, stream);
    };
    if (expected_t ran = run_once(); !ran) return suite.check(false, "loss enqueue: {}", ran.description_);
    cuStreamSynchronize(stream.get());
    f32_t device_loss = 0;
    cuMemcpyDtoH(&device_loss, device_address(loss_out), sizeof(f32_t));

    std::size_t const live_vertices = with_pad ? num_vertices - 1u : num_vertices; // the mean skips PAD
    auto const oracle_of = [&](std::vector<f64_t> const &towers) noexcept {
        f64r2_view_t const rows {towers.data(), num_vertices, hidden};
        return host_loss(rows, rows, prefixes, host_relations, 1.0, 0.0, live_vertices);
    };
    f64_t const oracle = oracle_of(host_tower);
    f64_t const relative = std::fabs(f64_t(device_loss) - oracle) / std::max(std::fabs(oracle), 1.0e-9);
    suite.check(relative < 5.0e-3, "loss {:.6f} matches the Float64 oracle {:.6f}: relative {:.2e}", device_loss,
                oracle, relative);

    // Finite-difference force check on a few sampled channels: the analytic BF16 gradient must sit
    // within BF16 resolution of (L(x+h) - L(x-h)) / 2h.
    std::vector<f64_t> host_grad(scalars);
    cuMemcpyDtoH(staging.data(), device_address(grad), scalars * sizeof(nk_bf16_t));
    if (!widen_from_bf16(staging, host_grad)) return suite.check(false, "gradient widening failed");
    std::vector<f32_t> host_column(scalars);
    cuMemcpyDtoH(host_column.data(), device_address(grad_f32), scalars * sizeof(f32_t));
    for (std::size_t sample = 0; sample != 6; ++sample) {
        std::size_t const index = (sample * 977ull + 13ull) % scalars;
        f64_t const step_size = 1.0e-2;
        std::vector<f64_t> plus = host_tower, minus = host_tower;
        plus[index] += step_size, minus[index] -= step_size;
        if (!narrow_to_bf16(plus, staging) || !narrow_to_bf16(minus, staging))
            return suite.check(false, "perturbation rounding failed");
        f64_t const numeric = (oracle_of(plus) - oracle_of(minus)) / (plus[index] - minus[index]);
        // The full derivative sums the row-side (bf16 output) and column-side (f32 subgraph) forces.
        f64_t const analytic = host_grad[index] + host_column[index];
        f64_t const difference = std::fabs(numeric - analytic);
        if (difference > 5.0e-2 * std::max(std::fabs(numeric), 0.05)) {
            suite.check(false, "force[{}] matches central finite differences: numeric {:.5f} analytic {:.5f}", index,
                        numeric, analytic);
            break;
        }
    }

    // Capture once, replay twice: the replays land where the eager run did, up to reduction order.
    std::vector<f64_t> const eager_grad = host_grad;
    auto graph_made = gpu_replay_graph_t::make(stream.get(), [&]() noexcept { return run_once(); });
    if (!graph_made) return suite.check(false, "capture: {}", graph_made.description_);
    for (int replay = 0; replay != 2; ++replay)
        if (expected_t launched = graph_made.value_.launch(stream.get()); !launched)
            return suite.check(false, "replay launch: {}", launched.description_);
    cuStreamSynchronize(stream.get());
    cuMemcpyDtoH(staging.data(), device_address(grad), scalars * sizeof(nk_bf16_t));
    if (!widen_from_bf16(staging, host_grad)) return suite.check(false, "gradient widening failed");
    f64_t largest = 0.0, worst = 0.0;
    for (std::size_t index = 0; index != scalars; ++index) {
        largest = std::max(largest, std::fabs(eager_grad[index]));
        worst = std::max(worst, std::fabs(eager_grad[index] - host_grad[index]));
    }
    suite.check(worst <= 1.0e-2 * largest, "captured replays match eager: worst {} over largest {}", worst, largest);
    cuCtxPopCurrent(nullptr);
}

/** Negative tests: every invalid staging must fail loudly, never fall back. */
static void run_negative_cases(test_suite_t &suite) {
    if (!cuda_devices_visible_(1)) return suite.skip(no_cuda_device_k);
    auto stream = gpu_stream_t::make(0);
    cuCtxPushCurrent(stream.value_.context());
    auto arena = gpu_arena_t::make(0, 64ull << 20);
    auto relations = contrastive_relations_t::make(arena.value_, 4, 4);
    u32_t offsets[5] = {0, 2, 2, 2, 2};
    u32_t unsorted_ids[2] = {3, 1};
    f32_t scores[2] = {1.0f, 1.0f};
    u32_t empty_offsets[5] = {0, 0, 0, 0, 0};
    auto unsorted = relations.value_.upload({offsets, 5}, {unsorted_ids, 2}, {scores, 2}, {empty_offsets, 5}, {}, {},
                                            {}, {}, {}, {}, 8, 0, 1.0f, stream.value_);
    suite.check(!unsorted, "unsorted run is rejected loudly");
    u32_t overflowing_ids[6] = {0, 1, 2, 3, 4, 5};
    u32_t overflow_offsets[5] = {0, 6, 6, 6, 6};
    f32_t overflow_scores[6] = {1, 1, 1, 1, 1, 1};
    auto overflow = relations.value_.upload({overflow_offsets, 5}, {overflowing_ids, 6}, {overflow_scores, 6},
                                            {empty_offsets, 5}, {}, {}, {}, {}, {}, {}, 8, 0, 1.0f, stream.value_);
    suite.check(!overflow, "edge-capacity overflow is rejected loudly");
    infiniband_coordinates_t coordinates;
    suite.check(!infiniband_coordinates_t::from_text("lid banana\n", coordinates), "corrupt coordinates are rejected");
    cuCtxPopCurrent(nullptr);
}

/** One loss call's three outputs read back: the loss summand and the first group's log-β and bias
 *  gradients. */
struct loss_readout_t {
    f64_t loss = 0.0, log_inverse_temperature_grad = 0.0, bias_grad = 0.0;
};

/** The sharded oracle: one global batch split across a simulated two-host, eight-GPU world on one
 *  device. Every simulated rank scores its rows against all sixteen subgraphs, and the per-rank
 *  loss and temperature summands must add up to one global call and to the Float64 oracle, PAD
 *  rows excluded from the mean. */
static void run_sharded_oracle_case(test_suite_t &suite) {
    if (!cuda_devices_visible_(1)) return suite.skip(no_cuda_device_k);
    constexpr u32_t ranks_k = 16, vertices_k = 8, hidden_k = 64, world_k = ranks_k * vertices_k;
    constexpr f32_t beta_k = 2.5f, bias_k = -1.5f;
    u32_t const prefixes[1] = {hidden_k};
    auto stream_made = gpu_stream_t::make(0);
    if (!stream_made) return suite.check(false, "stream: {}", stream_made.description_);
    gpu_stream_t stream = std::move(stream_made.value_);
    cuCtxPushCurrent(stream.context());
    auto arena_made = gpu_arena_t::make(0, 1ull << 30);
    if (!arena_made) return suite.check(false, "arena: {}", arena_made.description_);
    gpu_arena_t arena = std::move(arena_made.value_);

    // Every third rank ends in a PAD row, which neither scores nor counts toward the mean.
    std::vector<f64_t> host_tower(world_k * hidden_k);
    std::vector<nk_bf16_t> staging(world_k * hidden_k);
    u32_t host_lengths[world_k];
    for (std::size_t index = 0; index != host_tower.size(); ++index) host_tower[index] = hash_fill(index, 31);
    if (!narrow_to_bf16(host_tower, staging)) return suite.check(false, "tower rounding failed");
    u32_t live_vertices = 0;
    for (u32_t vertex = 0; vertex != world_k; ++vertex) {
        host_lengths[vertex] = (vertex / vertices_k) % 3u == 0u && vertex % vertices_k == vertices_k - 1u ? 0u : 4u;
        live_vertices += host_lengths[vertex] != 0u;
    }
    bf16r2_span_t const tower {arena.allocate_object<bf16_t>(world_k * hidden_k), world_k, hidden_k};
    bf16r2_span_t const row_grad {arena.allocate_object<bf16_t>(world_k * hidden_k), world_k, hidden_k};
    f32r2_span_t const column_grad {arena.allocate_object<f32_t>(world_k * hidden_k), world_k, hidden_k};
    u32r1_span_t const world_lengths {arena.allocate_object<u32_t>(world_k), world_k};
    f32r1_span_t const temperatures {arena.allocate_object<f32_t>(2), 2};
    f32r2_span_t const outputs {arena.allocate_object<f32_t>(3 * (ranks_k + 1)), ranks_k + 1, 3};
    f32_t const host_temperatures[2] = {beta_k, bias_k};
    cuMemcpyHtoD(device_address(tower.data()), staging.data(), staging.size() * sizeof(nk_bf16_t));
    cuMemcpyHtoD(device_address(world_lengths.data()), host_lengths, sizeof(host_lengths));
    cuMemcpyHtoD(device_address(temperatures.data()), host_temperatures, sizeof(host_temperatures));
    contrastive_recipe_t recipe;
    recipe.inverse_temperature_device = temperatures[nk::range(0, 1), nk::slice];
    recipe.bias_device = temperatures[nk::range(1, 2), nk::slice];

    // One loss call over the rows from `first_vertex`, its three outputs in row `call`.
    auto run_call = [&](contrastive_segments_t &loss, contrastive_relations_t &relations, u32_t first_vertex,
                        u32_t rows, nk::tensor_view<gpu_subgraph_t, 1> subgraphs, u32_t call) noexcept -> expected_t {
        auto const owned = nk::range(first_vertex, first_vertex + rows);
        expected_t staged = relations.upload_self_edges(rows, world_lengths[owned, nk::slice], world_lengths,
                                                        first_vertex, 1.0f, stream);
        if (staged)
            staged = loss(tower[owned, nk::slice], subgraphs, relations.view(), recipe,
                          outputs[call, nk::range(0, 1), nk::slice], row_grad[owned, nk::slice], stream,
                          outputs[call, nk::range(1, 2), nk::slice], outputs[call, nk::range(2, 3), nk::slice]);
        return staged;
    };

    contrastive_segments_config_t sharded_config;
    sharded_config.from_vertices_per_gpu = vertices_k;
    sharded_config.hidden = hidden_k;
    sharded_config.to_vertices_per_gpu = vertices_k;
    sharded_config.gpu_subgraphs_per_step = ranks_k;
    contrastive_segments_config_t global_config = sharded_config;
    global_config.from_vertices_per_gpu = world_k;
    global_config.to_vertices_per_gpu = world_k;
    global_config.gpu_subgraphs_per_step = 1;
    auto sharded_loss = contrastive_segments_t::make(arena, sharded_config, {prefixes, 1});
    auto global_loss = contrastive_segments_t::make(arena, global_config, {prefixes, 1});
    auto sharded_relations = contrastive_relations_t::make(arena, vertices_k, 0);
    auto global_relations = contrastive_relations_t::make(arena, world_k, 0);
    if (!sharded_loss || !global_loss || !sharded_relations || !global_relations)
        return suite.check(false, "sharded bring-up: {}{}", sharded_loss.description_, global_loss.description_);

    bf16r3_span_t const tower_planes = tower.reshape<3>({ranks_k, vertices_k, hidden_k});
    f32r3_span_t const column_grad_planes = column_grad.reshape<3>({ranks_k, vertices_k, hidden_k});
    gpu_subgraph_t subgraphs[ranks_k];
    for (u32_t rank = 0; rank != ranks_k; ++rank) {
        subgraphs[rank].to_vertices = tower_planes[rank, nk::slice];
        subgraphs[rank].global_offset = std::size_t {rank} * vertices_k;
        subgraphs[rank].to_vertices_grad = column_grad_planes[rank, nk::slice];
    }
    gpu_subgraph_t const whole[1] = {{tower, 0, column_grad}};
    expected_t ran {};
    for (u32_t rank = 0; rank != ranks_k && ran; ++rank)
        ran = run_call(sharded_loss.value_, sharded_relations.value_, rank * vertices_k, vertices_k,
                       {subgraphs, ranks_k}, rank);
    if (ran) ran = run_call(global_loss.value_, global_relations.value_, 0, world_k, {whole, 1}, ranks_k);
    if (!ran) return suite.check(false, "sharded enqueue: {}", ran.description_);
    cuStreamSynchronize(stream.get());
    f32_t host_outputs[ranks_k + 1][3];
    cuMemcpyDtoH(host_outputs, device_address(outputs.data()), sizeof(host_outputs));
    loss_readout_t sharded, global;
    for (u32_t rank = 0; rank != ranks_k; ++rank) {
        sharded.loss += host_outputs[rank][0];
        sharded.log_inverse_temperature_grad += host_outputs[rank][1];
        sharded.bias_grad += host_outputs[rank][2];
    }
    global.loss = host_outputs[ranks_k][0];
    global.log_inverse_temperature_grad = host_outputs[ranks_k][1];
    global.bias_grad = host_outputs[ranks_k][2];

    // The Float64 oracle and its central differences in log-β and in the bias.
    host_relations_t host_relations;
    host_relations.vertex_effective.assign(host_lengths, host_lengths + world_k);
    host_relations.neighbor_effective.assign(host_lengths, host_lengths + world_k);
    f64r2_view_t const host_rows {host_tower.data(), world_k, hidden_k};
    auto oracle_at = [&](f64_t beta, f64_t bias) noexcept {
        return host_loss(host_rows, host_rows, prefixes, host_relations, beta, bias, live_vertices);
    };
    f64_t const step_size = 1.0e-4;
    loss_readout_t oracle;
    oracle.loss = oracle_at(beta_k, bias_k);
    oracle.log_inverse_temperature_grad = (oracle_at(beta_k * std::exp(step_size), bias_k) -
                                           oracle_at(beta_k * std::exp(-step_size), bias_k)) /
                                          (2.0 * step_size);
    oracle.bias_grad = (oracle_at(beta_k, bias_k + step_size) - oracle_at(beta_k, bias_k - step_size)) /
                       (2.0 * step_size);

    auto relative = [](f64_t got, f64_t want) noexcept {
        return std::fabs(got - want) / std::max(std::fabs(want), 1.0e-9);
    };
    // The split moves the Float32 sums no further than rounding, and both land near Float64.
    f64_t const tolerance = 1.0e-6, float64_tolerance = 1.0e-5;
    suite.check(relative(sharded.loss, global.loss) < tolerance, "sharded loss {:.9f} vs global {:.9f}", sharded.loss,
                global.loss);
    suite.check(relative(sharded.log_inverse_temperature_grad, global.log_inverse_temperature_grad) < tolerance,
                "sharded log-β gradient {:.9f} vs global {:.9f}", sharded.log_inverse_temperature_grad,
                global.log_inverse_temperature_grad);
    suite.check(relative(sharded.bias_grad, global.bias_grad) < tolerance,
                "sharded bias gradient {:.9f} vs global {:.9f}", sharded.bias_grad, global.bias_grad);
    suite.check(relative(sharded.loss, oracle.loss) < float64_tolerance,
                "sharded loss {:.9f} vs Float64 {:.9f}: {:.2e}", sharded.loss, oracle.loss,
                relative(sharded.loss, oracle.loss));
    suite.check(relative(sharded.log_inverse_temperature_grad, oracle.log_inverse_temperature_grad) < float64_tolerance,
                "sharded log-β gradient {:.9f} vs Float64 {:.9f}: {:.2e}", sharded.log_inverse_temperature_grad,
                oracle.log_inverse_temperature_grad,
                relative(sharded.log_inverse_temperature_grad, oracle.log_inverse_temperature_grad));
    suite.check(relative(sharded.bias_grad, oracle.bias_grad) < float64_tolerance,
                "sharded bias gradient {:.9f} vs Float64 {:.9f}: {:.2e}", sharded.bias_grad, oracle.bias_grad,
                relative(sharded.bias_grad, oracle.bias_grad));
    cuCtxPopCurrent(nullptr);
}

/** The distributed tier: 2 GPUs vs the single-GPU global reference on identical data. */
static void run_distributed_case(test_suite_t &suite) {
    if (!cuda_devices_visible_(2)) return suite.skip("needs two CUDA devices");
    constexpr std::size_t gpus_k = 2;
    constexpr u32_t vertices_k = 8, hidden_k = 64;
    constexpr std::size_t scalars_k = std::size_t(vertices_k) * hidden_k;
    i32_t devices[gpus_k] = {0, 1};
    u32_t prefix_dims[1] = {hidden_k};

    // Global data on the BFloat16 grid: 16 vertices; rank r owns rows [8r, 8r+8).
    std::vector<f64_t> global_tower(gpus_k * scalars_k);
    std::vector<nk_bf16_t> staging(global_tower.size());
    for (std::size_t index = 0; index != global_tower.size(); ++index) global_tower[index] = hash_fill(index, 7);
    if (!narrow_to_bf16(global_tower, staging)) return suite.check(false, "tower rounding failed");
    nk::tensor_span<nk_bf16_t, 2> const staging_rows {staging.data(), gpus_k, scalars_k};

    per_gpu_array<gpu_stream_t> streams;
    per_gpu_array<gpu_arena_t> arenas;
    contrastive_relations_t owners[gpus_k];
    contrastive_relations_view_t relations[gpus_k];
    bf16r2_view_t towers[gpus_k];
    bf16r2_span_t grads[gpus_k];
    contrastive_segments_config_t config;
    config.from_vertices_per_gpu = vertices_k;
    config.hidden = hidden_k;
    config.to_vertices_per_gpu = vertices_k;
    config.gpu_subgraphs_per_step = gpus_k;
    // The composition-root order: topology → arenas (visibility declared) → streams → the ONE
    // multimem domain → the ONE colocated domain → the consumer, borrowing all of it.
    for (std::size_t rank = 0; rank != gpus_k; ++rank) {
        auto stream_made = gpu_stream_t::make(devices[rank]);
        if (!stream_made || streams.push_back(std::move(stream_made.value_)) != status_t::success_k)
            return suite.check(false, "distributed stream make");
        cuCtxPushCurrent(streams[rank].context());
        auto arena_made = gpu_arena_t::make(devices[rank], 1ull << 30, {devices, gpus_k});
        if (!arena_made || arenas.push_back(std::move(arena_made.value_)) != status_t::success_k)
            return suite.check(false, "distributed arena make");
        bf16_t *tower = arenas[rank].allocate_object<bf16_t>(scalars_k);
        bf16_t *grad = arenas[rank].allocate_object<bf16_t>(scalars_k);
        cuMemcpyHtoD(device_address(tower), staging_rows[rank, nk::slice].data(), scalars_k * sizeof(nk_bf16_t));
        towers[rank] = bf16r2_view_t {tower, vertices_k, hidden_k};
        grads[rank] = bf16r2_span_t {grad, vertices_k, hidden_k};
        owners[rank] = std::move(contrastive_relations_t::make(arenas[rank], vertices_k, 0).value_);
        auto staged = owners[rank].upload_self_edges(vertices_k, {}, {}, rank * std::size_t(vertices_k), 1.0f,
                                                     streams[rank]);
        if (!staged) return suite.check(false, "distributed relations: {}", staged.description_);
        relations[rank] = owners[rank].view();
        cuCtxPopCurrent(nullptr);
    }
    // A one-process world: the whole box is one host_domain and one multimem_domain (D=1), so the
    // colocated domain is the trivial identity and no exchange happens.
    auto topology = mesh_topology_t::bootstrap(0, 1, 0, "/tmp/uform_exchange_local", "distributed_case");
    if (!topology) return suite.check(false, "single-domain bootstrap: {}", topology.description_);
    auto domain = multimem_domain_t::make(
        topology.value_, {devices, gpus_k},
        distributed_contrastive_segments_t::payload_scalars_needed(config, topology.value_.shape), streams, 120.0);
    if (!domain) return suite.check(false, "multimem domain make: {}", domain.description_);
    auto colocated = colocated_domain_t::make(topology.value_);
    if (!colocated) return suite.check(false, "colocated domain make: {}", colocated.description_);
    auto distributed = distributed_contrastive_segments_t::make(domain.value_, colocated.value_, arenas, streams,
                                                                {towers, gpus_k}, {relations, gpus_k}, {grads, gpus_k},
                                                                config, {prefix_dims, 1});
    if (!distributed) return suite.check(false, "distributed make: {}", distributed.description_);
    auto ran = distributed.value_(contrastive_recipe_t {}, streams);
    if (!ran) return suite.check(false, "distributed run: {}", ran.description_);
    for (std::size_t rank = 0; rank != gpus_k; ++rank) {
        cuCtxPushCurrent(streams[rank].context());
        cuStreamSynchronize(streams[rank].get());
        cuCtxPopCurrent(nullptr);
    }
    f64_t distributed_loss = 0.0;
    std::vector<f64_t> distributed_grad(gpus_k * scalars_k);
    cuCtxPushCurrent(streams[0].context());
    f32_t summands[gpus_k] = {};
    cuMemcpyDtoH(summands, device_address(distributed.value_.loss_outs().data()), sizeof(summands));
    for (std::size_t rank = 0; rank != gpus_k; ++rank) {
        distributed_loss += summands[rank];
        cuMemcpyDtoH(staging_rows[rank, nk::slice].data(), device_address(grads[rank].data()),
                     scalars_k * sizeof(nk_bf16_t));
    }
    cuCtxPopCurrent(nullptr);
    if (!widen_from_bf16(staging, distributed_grad)) return suite.check(false, "gradient widening failed");

    // The Float64 oracle over the SAME global grid — the sum of rank summands must match it.
    host_relations_t host_relations;
    u32_t const prefixes[1] = {hidden_k};
    auto const oracle_of = [&](std::vector<f64_t> const &towers) noexcept {
        f64r2_view_t const rows {towers.data(), gpus_k * vertices_k, hidden_k};
        return host_loss(rows, rows, prefixes, host_relations, 1.0, 0.0, gpus_k * vertices_k);
    };
    f64_t const oracle = oracle_of(global_tower);
    f64_t const relative = std::fabs(distributed_loss - oracle) / std::max(std::fabs(oracle), 1.0e-9);
    sz::string_view_t const tier = domain.value_.description();
    suite.check(relative < 5.0e-3,
                "distributed loss sum {:.6f} matches the global Float64 oracle {:.6f}: relative {:.2e} over {}",
                distributed_loss, oracle, relative, tier);

    // Finite-difference one channel of rank 1's tower against the combined distributed force.
    std::size_t const probe_index = scalars_k + 321; // Inside rank 1's rows.
    f64_t const step_size = 1.0e-2;
    std::vector<f64_t> plus = global_tower, minus = global_tower;
    plus[probe_index] += step_size, minus[probe_index] -= step_size;
    if (!narrow_to_bf16(plus, staging) || !narrow_to_bf16(minus, staging))
        return suite.check(false, "perturbation rounding failed");
    f64_t const numeric = (oracle_of(plus) - oracle_of(minus)) / (plus[probe_index] - minus[probe_index]);
    f64_t const analytic = distributed_grad[probe_index];
    suite.check(std::fabs(numeric - analytic) < 5.0e-2 * std::max(std::fabs(numeric), 0.05),
                "distributed force matches the global finite difference: numeric {:.5f} distributed {:.5f}", numeric,
                analytic);
}

/** The cross-domain tier: one process per multimem_domain, GDAKI gather + all-reduce over the
 *  local HCAs, verified against the same global Float64 oracle each process computes independently.
 *  Activated by UFORM_HOST_DOMAIN_INDEX in the environment. */
static void run_cross_domain_case(test_suite_t &suite) {
#if !defined(UFORM_WITH_INFINIBAND)
    suite.skip("built without UFORM_WITH_INFINIBAND");
#else
    if (!cuda_devices_visible_(1)) return suite.skip(no_cuda_device_k);
    int device_count = 0;
    cuDeviceGetCount(&device_count);
    u32_t const gpus = u32_t(device_count);
    u32_t const domain = u32_t(std::atoi(std::getenv("UFORM_HOST_DOMAIN_INDEX")));
    auto topology_made = mesh_topology_t::bootstrap(domain, 2, gpus);
    if (!topology_made) return suite.check(false, "bootstrap: {}", topology_made.description_);
    mesh_topology_t topology = std::move(topology_made.value_);

    constexpr u32_t vertices_k = 8, hidden_k = 64;
    std::size_t const scalars_k = std::size_t(vertices_k) * hidden_k;
    u32_t const world_gpus = 2 * gpus;
    u32_t prefix_dims[1] = {hidden_k};
    std::vector<f64_t> global_tower(world_gpus * scalars_k);
    std::vector<nk_bf16_t> staging(global_tower.size());
    for (std::size_t index = 0; index != global_tower.size(); ++index) global_tower[index] = hash_fill(index, 7);
    if (!narrow_to_bf16(global_tower, staging)) return suite.check(false, "tower rounding failed");
    nk::tensor_span<nk_bf16_t, 2> const staging_rows {staging.data(), world_gpus, scalars_k};

    per_gpu_array<gpu_stream_t> streams;
    per_gpu_array<gpu_arena_t> arenas;
    std::vector<contrastive_relations_t> owners(gpus);
    std::vector<contrastive_relations_view_t> relations(gpus);
    std::vector<bf16r2_view_t> towers(gpus);
    std::vector<bf16r2_span_t> grads(gpus);
    contrastive_segments_config_t config;
    config.from_vertices_per_gpu = vertices_k;
    config.hidden = hidden_k;
    config.to_vertices_per_gpu = vertices_k;
    config.gpu_subgraphs_per_step = world_gpus;
    std::vector<i32_t> devices(gpus);
    for (u32_t rank = 0; rank != gpus; ++rank) devices[rank] = i32_t(rank);
    for (u32_t rank = 0; rank != gpus; ++rank) {
        auto stream_made = gpu_stream_t::make(i32_t(rank));
        if (!stream_made || streams.push_back(std::move(stream_made.value_)) != status_t::success_k)
            return suite.check(false, "cross-domain stream make on GPU {}", rank);
        cuCtxPushCurrent(streams[rank].context());
        auto arena_made = gpu_arena_t::make(i32_t(rank), 1ull << 30, {devices.data(), gpus});
        if (!arena_made || arenas.push_back(std::move(arena_made.value_)) != status_t::success_k)
            return suite.check(false, "cross-domain arena make on GPU {}", rank);
        bf16_t *tower = arenas[rank].allocate_object<bf16_t>(scalars_k);
        bf16_t *grad = arenas[rank].allocate_object<bf16_t>(scalars_k);
        std::size_t const global_rank = std::size_t(domain) * gpus + rank;
        cuMemcpyHtoD(device_address(tower), staging_rows[global_rank, nk::slice].data(), scalars_k * sizeof(nk_bf16_t));
        towers[rank] = bf16r2_view_t {tower, vertices_k, hidden_k};
        grads[rank] = bf16r2_span_t {grad, vertices_k, hidden_k};
        owners[rank] = std::move(contrastive_relations_t::make(arenas[rank], vertices_k, 0).value_);
        auto staged = owners[rank].upload_self_edges(vertices_k, {}, {}, global_rank * std::size_t(vertices_k), 1.0f,
                                                     streams[rank]);
        if (!staged) return suite.check(false, "cross-domain relations: {}", staged.description_);
        relations[rank] = owners[rank].view();
        cuCtxPopCurrent(nullptr);
    }
    // The composition roots, in main-order: the ONE multimem domain, the ONE colocated domain.
    auto multimem = multimem_domain_t::make(
        topology, {devices.data(), gpus},
        distributed_contrastive_segments_t::payload_scalars_needed(config, topology.shape), streams, 120.0);
    if (!multimem) return suite.check(false, "multimem domain make: {}", multimem.description_);
    auto colocated = colocated_domain_t::make(topology);
    if (!colocated) return suite.check(false, "colocated domain make: {}", colocated.description_);
    auto distributed = distributed_contrastive_segments_t::make(multimem.value_, colocated.value_, arenas, streams,
                                                                {towers.data(), gpus}, {relations.data(), gpus},
                                                                {grads.data(), gpus}, config, {prefix_dims, 1});
    if (!distributed) return suite.check(false, "cross-domain make: {}", distributed.description_);
    sz::string_view_t const tier = multimem.value_.description();
    u32_t const world_domains = distributed.value_.world_domains();
    auto ran = distributed.value_(contrastive_recipe_t {}, streams);
    if (!ran)
        return suite.check(false, "cross-domain run over {} domains on {}: {}", world_domains, tier, ran.description_);
    for (u32_t rank = 0; rank != gpus; ++rank) {
        cuCtxPushCurrent(streams[rank].context());
        CUresult const synchronized = cuStreamSynchronize(streams[rank].get());
        cuCtxPopCurrent(nullptr);
        if (synchronized != CUDA_SUCCESS)
            return suite.check(false, "cross-domain sync on GPU {}: result {}", rank, static_cast<int>(synchronized));
    }

    // Combine the two domains' loss sums over the mesh exchange; each process checks the global oracle.
    f64_t domain_loss = 0.0;
    {
        cuCtxPushCurrent(streams[0].context());
        std::vector<f32_t> summands(gpus);
        cuMemcpyDtoH(summands.data(), device_address(distributed.value_.loss_outs().data()), gpus * sizeof(f32_t));
        cuCtxPopCurrent(nullptr);
        for (f32_t const summand : summands) domain_loss += summand;
    }
    sz::string_t loss_text;
    if (!format_into(loss_text, "{}", domain_loss)) return suite.check(false, "loss text formatting");
    if (expected_t published = topology.publish_text("loss", loss_text.view()); !published)
        return suite.check(false, "loss publish: {}", published.description_);
    sz::string_t peer_text;
    if (expected_t awaited = topology.await_text(1u - domain, "loss", peer_text, 60.0); !awaited)
        return suite.check(false, "peer loss await: {}", awaited.description_);
    f64_t const global_loss = domain_loss + std::atof(peer_text.c_str());

    host_relations_t host_relations;
    auto const oracle_of = [&](std::vector<f64_t> const &towers) noexcept {
        f64r2_view_t const rows {towers.data(), world_gpus * vertices_k, hidden_k};
        return host_loss(rows, rows, prefix_dims, host_relations, 1.0, 0.0, world_gpus * vertices_k);
    };
    f64_t const oracle = oracle_of(global_tower);
    f64_t const relative = std::fabs(global_loss - oracle) / std::max(std::fabs(oracle), 1.0e-9);
    suite.check(relative < 5.0e-3,
                "cross-domain loss sum {:.6f} matches the global Float64 oracle {:.6f}: relative {:.2e} over {}",
                global_loss, oracle, relative, tier);

    // Finite-difference one locally-owned channel against the fully-reduced force.
    std::size_t const probe_index = (std::size_t(domain) * gpus + 1) * scalars_k + 321;
    std::vector<nk_bf16_t> local_staging(scalars_k);
    std::vector<f64_t> local_grad(scalars_k);
    cuCtxPushCurrent(streams[1 % gpus].context());
    cuMemcpyDtoH(local_staging.data(), device_address(grads[1 % gpus].data()), scalars_k * sizeof(nk_bf16_t));
    cuCtxPopCurrent(nullptr);
    if (!widen_from_bf16(local_staging, local_grad)) return suite.check(false, "gradient widening failed");
    f64_t const step_size = 1.0e-2;
    std::vector<f64_t> plus = global_tower, minus = global_tower;
    plus[probe_index] += step_size, minus[probe_index] -= step_size;
    if (!narrow_to_bf16(plus, staging) || !narrow_to_bf16(minus, staging))
        return suite.check(false, "perturbation rounding failed");
    f64_t const numeric = (oracle_of(plus) - oracle_of(minus)) / (plus[probe_index] - minus[probe_index]);
    f64_t const analytic = local_grad[probe_index % scalars_k];
    suite.check(std::fabs(numeric - analytic) < 5.0e-2 * std::max(std::fabs(numeric), 0.05),
                "cross-domain force matches the global finite difference: numeric {:.5f} distributed {:.5f}", numeric,
                analytic);
    expected_t closed = topology.barrier("done", 60.0);
    suite.check(static_cast<bool>(closed), "the closing barrier should complete: {}", closed.description_);
#endif
}

/** The FP64 attention-pool + head parity gate: seed a tiny text tower with keyed weights, run the
 *  contrastive forward over a fabricated ragged batch (one PAD segment included — the historical pool
 *  failure mode was PAD masking, never arithmetic), and compare the emitted vertices against a
 *  double-precision host replica of exactly the new math — the softmax pool over the tower's own
 *  final-norm states, then RMSNorm → fc1 → SiLU → fc2. The backward runs the same chain in reverse, and
 *  the pool's own state and query gradients must match the host softmax backward to the BFloat16 floor. */
static void run_pool_parity_case(test_suite_t &suite) {
    if (!cuda_devices_visible_(1)) return suite.skip(no_cuda_device_k);
    constexpr u32_t rows_k = 2, segments_k = 4, columns_k = 8, hidden_k = 256, tokens_k = rows_k * columns_k;
    constexpr u32_t heads_k = hidden_k / transformer_head_dim_k;
    auto stream = gpu_stream_t::make(0);
    if (!stream) return suite.check(false, "pool-parity stream make: {}", stream.description_);
    cuCtxPushCurrent(stream.value_.context());
    auto arena = gpu_arena_t::make(0, 1ull << 30, {});
    if (!arena) {
        suite.check(false, "pool-parity bring-up: {}", arena.description_);
        cuCtxPopCurrent(nullptr);
        return;
    }

    encoder_tower_config_t config;
    config.vocabulary = 258;
    config.depth = 1;
    config.block.hidden = hidden_k;
    config.block.max_tokens = tokens_k;
    config.block.max_segments = segments_k;
    config.block.max_segment_tokens = columns_k;
    config.rope_max_positions = columns_k;
    multimodal_batch_shape_t shape {rows_k, segments_k, columns_k};
    auto tower = encoder_tower_bf16_t::make(arena.value_, config, shape);
    if (!tower) {
        suite.check(false, "pool-parity tower make: {}", tower.description_);
        cuCtxPopCurrent(nullptr);
        return;
    }

    // Keyed weight seeding through the registration list, each matrix kept on the BFloat16 grid in
    // Float64 for the host replica.
    array<differentiable_weights_t> weights;
    if (!tower.value_.fill(weights)) {
        suite.check(false, "pool-parity registration");
        cuCtxPopCurrent(nullptr);
        return;
    }
    std::vector<std::vector<f64_t>> host_weights(weights.size());
    std::vector<nk_bf16_t> staging;
    for (std::size_t matrix = 0; matrix != weights.size(); ++matrix) {
        std::size_t const scalars = weights[matrix].weights.numel();
        host_weights[matrix].resize(scalars);
        staging.resize(scalars);
        for (std::size_t index = 0; index != scalars; ++index)
            host_weights[matrix][index] = hash_fill(index, matrix + 101);
        if (!narrow_to_bf16(host_weights[matrix], staging)) {
            suite.check(false, "pool-parity weight rounding");
            cuCtxPopCurrent(nullptr);
            return;
        }
        cuMemcpyHtoD(device_address(weights[matrix].weights.data()), staging.data(), scalars * sizeof(nk_bf16_t));
    }

    // A fabricated ragged batch: three live text segments of lengths {8, 5, 2}, a gap token closing the
    // second row, and one PAD — the batch packer's contract (offsets repeat on PAD, lengths carry the live
    // counts).
    auto batch = multimodal_batch_on_gpu_t::make(gpu_arena_alloc<b8_t> {arena.value_}, shape);
    if (!batch) {
        suite.check(false, "pool-parity batch make: {}", batch.description_);
        cuCtxPopCurrent(nullptr);
        return;
    }
    u32_t const host_offsets[segments_k + 1] = {0, 8, 13, 15, 15};
    u32_t const host_lengths[segments_k] = {8, 5, 2, 0};
    modality_t const host_modality[segments_k] = {modality_t::text_k, modality_t::text_k, modality_t::text_k,
                                                  modality_t::text_k};
    std::vector<token_id_t> host_tokens(tokens_k);
    for (u32_t token = 0; token != tokens_k; ++token) host_tokens[token] = static_cast<token_id_t>(token % 251u);
    cuMemcpyHtoD(device_address(batch.value_.segment_offsets.data()), host_offsets, sizeof(host_offsets));
    cuMemcpyHtoD(device_address(batch.value_.segment_lengths.data()), host_lengths, sizeof(host_lengths));
    cuMemcpyHtoD(device_address(batch.value_.segment_modality.data()), host_modality, sizeof(host_modality));
    cuMemcpyHtoD(device_address(batch.value_.tokens.data()), host_tokens.data(), tokens_k * sizeof(token_id_t));

    // Forward with vertices, then read back the tower's own final-norm states — the oracle's input, so
    // the parity isolates exactly the new pool + head math.
    bf16r2_tensor_arena_t vertices_tensor =
        bf16r2_tensor_arena_t::uninitialized({segments_k, hidden_k}, gpu_arena_alloc<bf16_t> {arena.value_}).value;
    bf16r2_tensor_arena_t vertices_grad_tensor =
        bf16r2_tensor_arena_t::uninitialized({segments_k, hidden_k}, gpu_arena_alloc<bf16_t> {arena.value_}).value;
    if (!vertices_tensor || !vertices_grad_tensor) {
        suite.check(false, "pool-parity vertices alloc");
        cuCtxPopCurrent(nullptr);
        return;
    }
    cuMemsetD8(device_address(vertices_tensor.data()), 0, std::size_t(segments_k) * hidden_k * sizeof(bf16_t));
    cuMemsetD8(device_address(vertices_grad_tensor.data()), 0, std::size_t(segments_k) * hidden_k * sizeof(bf16_t));
    auto ran = tower.value_.pack_weights(stream.value_);
    if (ran) ran = tower.value_.forward(batch.value_, vertices_tensor.span(), stream.value_);
    suite.check(static_cast<bool>(ran), "pool-parity forward: {}", ran.description_);
    if (!ran) {
        cuCtxPopCurrent(nullptr);
        return;
    }
    cuStreamSynchronize(stream.value_.get());

    // The tower's states and vertices, widened once.
    std::vector<f64_t> host_states(std::size_t(tokens_k) * hidden_k),
        device_vertices(std::size_t(segments_k) * hidden_k);
    staging.resize(host_states.size());
    cuMemcpyDtoH(staging.data(), device_address(tower.value_.normalized().data()), staging.size() * sizeof(nk_bf16_t));
    bool widened = widen_from_bf16(staging, host_states);
    staging.resize(device_vertices.size());
    cuMemcpyDtoH(staging.data(), device_address(vertices_tensor.data()), staging.size() * sizeof(nk_bf16_t));
    widened = widened && widen_from_bf16(staging, device_vertices);

    // Host replica in double precision: per-(segment,head) softmax pool, then RMSNorm → fc1 → SiLU → fc2.
    // Weight order in the registration list: ... final_norm, pool_query, head.norm, head.fc1, head.fc2.
    std::size_t const query_index = weights.size() - 4, head_norm_index = weights.size() - 3;
    std::size_t const fc1_index = weights.size() - 2, fc2_index = weights.size() - 1;
    f64r2_view_t const host_query_rows {host_weights[query_index].data(), heads_k, transformer_head_dim_k};
    f64r1_view_t const host_head_norm {host_weights[head_norm_index].data(), hidden_k};
    f64r2_view_t const host_fc1_rows {host_weights[fc1_index].data(), hidden_k, hidden_k};
    f64r2_view_t const host_fc2_rows {host_weights[fc2_index].data(), hidden_k, hidden_k};
    f64r3_view_t const host_state_planes {host_states.data(), {tokens_k, heads_k, transformer_head_dim_k}};
    double const scale = 1.0 / std::sqrt(double(transformer_head_dim_k));
    std::vector<double> host_pooled(std::size_t(segments_k) * hidden_k, 0.0);
    f64r3_span_t const host_pooled_planes {host_pooled.data(), {segments_k, heads_k, transformer_head_dim_k}};
    // The pool's backward rides the same softmax with the keyed force as d(pooled) and K = V = the states:
    // dS = P∘(dO·Vᵀ − dO·O), d(states) = P·dO + scale·dS·q, d(query) = scale·Σ dS·K.
    std::vector<f64_t> host_force(std::size_t(segments_k) * hidden_k);
    std::vector<nk_bf16_t> host_force_staging(host_force.size());
    for (std::size_t index = 0; index != host_force.size(); ++index) host_force[index] = hash_fill(index, 777) * 0.1;
    widened = widened && narrow_to_bf16(host_force, host_force_staging);
    if (!widened) {
        suite.check(false, "pool-parity staging conversions");
        cuCtxPopCurrent(nullptr);
        return;
    }
    f64r3_view_t const host_force_planes {host_force.data(), {segments_k, heads_k, transformer_head_dim_k}};
    std::vector<double> host_states_grad(std::size_t(tokens_k) * hidden_k, 0.0), host_query_grad(hidden_k, 0.0);
    f64r3_span_t const host_states_grad_planes {host_states_grad.data(), {tokens_k, heads_k, transformer_head_dim_k}};
    f64r2_span_t const host_query_grad_rows {host_query_grad.data(), heads_k, transformer_head_dim_k};
    for (u32_t segment = 0; segment != segments_k; ++segment) {
        u32_t const length = host_lengths[segment];
        if (length == 0u) continue; // a PAD vertex is don't-care — the loss PADs it out, so no oracle here
        u32_t const offset = host_offsets[segment];
        for (u32_t head = 0; head != heads_k; ++head) {
            f64r1_view_t const query_head = host_query_rows[head, nk::slice];
            f64r1_view_t const force_head = host_force_planes[segment, head, nk::slice];
            std::vector<double> probabilities(length);
            double score_max = -1e300, denominator = 0.0;
            for (u32_t token = 0; token != length; ++token) {
                probabilities[token] = dot(query_head, host_state_planes[offset + token, head, nk::slice]) * scale;
                score_max = std::max(score_max, probabilities[token]);
            }
            for (u32_t token = 0; token != length; ++token)
                denominator += probabilities[token] = std::exp(probabilities[token] - score_max);
            for (u32_t token = 0; token != length; ++token) probabilities[token] /= denominator;
            for (u32_t channel = 0; channel != transformer_head_dim_k; ++channel) {
                double value = 0.0;
                for (u32_t token = 0; token != length; ++token)
                    value += probabilities[token] * host_state_planes[offset + token, head, channel];
                host_pooled_planes[segment, head, channel] = value;
            }
            double const output_dot = dot(force_head, host_pooled_planes[segment, head, nk::slice]);
            for (u32_t token = 0; token != length; ++token) {
                f64r1_view_t const state = host_state_planes[offset + token, head, nk::slice];
                double const score_grad = probabilities[token] * (dot(force_head, state) - output_dot);
                for (u32_t channel = 0; channel != transformer_head_dim_k; ++channel) {
                    host_states_grad_planes[offset + token, head, channel] +=
                        probabilities[token] * force_head[channel] + scale * score_grad * query_head[channel];
                    host_query_grad_rows[head, channel] += scale * score_grad * state[channel];
                }
            }
        }
    }
    f64r2_view_t const host_pooled_rows {host_pooled.data(), segments_k, hidden_k};
    f64r2_view_t const device_vertices_rows {device_vertices.data(), segments_k, hidden_k};
    std::vector<double> host_vertices(std::size_t(segments_k) * hidden_k, 0.0);
    f64r2_span_t const host_vertices_rows {host_vertices.data(), segments_k, hidden_k};
    for (u32_t segment = 0; segment != segments_k; ++segment) {
        if (host_lengths[segment] == 0u) continue; // PAD vertices stay outside the parity
        f64r1_view_t const pooled_row = host_pooled_rows[segment, nk::slice];
        double const inverse_rms = 1.0 / std::sqrt(dot(pooled_row, pooled_row) / hidden_k + 1e-6);
        std::vector<double> normalized(hidden_k), activated(hidden_k);
        for (u32_t channel = 0; channel != hidden_k; ++channel)
            normalized[channel] = pooled_row[channel] * inverse_rms * host_head_norm[channel];
        for (u32_t out = 0; out != hidden_k; ++out) {
            double const projected = dot(host_fc1_rows.row(out), normalized);
            activated[out] = projected / (1.0 + std::exp(-projected));
        }
        for (u32_t out = 0; out != hidden_k; ++out)
            host_vertices_rows[segment, out] = dot(host_fc2_rows.row(out), activated);
    }
    double worst = 0.0;
    for (u32_t segment = 0; segment != segments_k; ++segment) {
        if (host_lengths[segment] == 0u) continue; // PAD vertices stay outside the parity
        for (u32_t channel = 0; channel != hidden_k; ++channel)
            worst = std::max(worst,
                             std::abs(host_vertices_rows[segment, channel] - device_vertices_rows[segment, channel]) /
                                 (std::abs(host_vertices_rows[segment, channel]) + 1.0));
    }
    suite.check(worst < 3e-2,
                "pool + head forward matches the FP64 oracle to the BFloat16 floor: worst relative {:.3e}", worst);

    // Backward smoke over the same chain: the keyed vertices force, then every pool / head gradient finite
    // and the query gradient nonzero (the pool actually differentiated).
    cuMemcpyHtoD(device_address(vertices_grad_tensor.data()), host_force_staging.data(),
                 host_force_staging.size() * sizeof(nk_bf16_t));
    auto propagated = tower.value_.backward(vertices_grad_tensor.view(), bf16r2_view_t {}, batch.value_, stream.value_);
    suite.check(static_cast<bool>(propagated), "pool-parity backward: {}", propagated.description_);
    if (propagated) {
        cuStreamSynchronize(stream.value_.get());
        bool all_finite = true;
        double query_grad_energy = 0.0;
        std::vector<f64_t> gradient;
        for (std::size_t matrix = query_index; matrix != weights.size(); ++matrix) {
            std::size_t const scalars = weights[matrix].weights_grad.numel();
            staging.resize(scalars), gradient.resize(scalars);
            cuMemcpyDtoH(staging.data(), device_address(weights[matrix].weights_grad.data()),
                         scalars * sizeof(nk_bf16_t));
            all_finite = all_finite && widen_from_bf16(staging, gradient);
            for (f64_t const value : gradient) all_finite = all_finite && std::isfinite(value);
            if (matrix == query_index) query_grad_energy = dot(gradient, gradient);
        }
        suite.check(all_finite, "every pool / head gradient is finite");
        suite.check(query_grad_energy > 0.0, "the pool query gradient is nonzero");
    }

    // The tower's d(normalized) is overwritten by the block stack's backward before it returns, so a second
    // pool replays forward + backward over the tower's states and query, into the spent vertices buffer.
    auto pool = tower_pool_t::make(arena.value_, config.block);
    f32r2_tensor_arena_t states_grad =
        f32r2_tensor_arena_t::uninitialized({tokens_k, hidden_k}, gpu_arena_alloc<f32_t> {arena.value_}).value;
    f32r2_tensor_arena_t value_grad =
        f32r2_tensor_arena_t::uninitialized({tokens_k, hidden_k}, gpu_arena_alloc<f32_t> {arena.value_}).value;
    bf16r1_tensor_arena_t query_grad =
        bf16r1_tensor_arena_t::uninitialized({hidden_k}, gpu_arena_alloc<bf16_t> {arena.value_}).value;
    if (!pool || !states_grad || !value_grad || !query_grad) {
        suite.check(false, "pool-parity pool make: {}", pool.description_);
        cuCtxPopCurrent(nullptr);
        return;
    }
    bf16r1_view_t const query {weights[query_index].weights.data(), hidden_k};
    auto pooled = pool.value_.forward(query, tower.value_.normalized(), batch.value_.segment_offsets.view(),
                                      batch.value_.segment_lengths.view(), vertices_tensor.span(), stream.value_);
    if (pooled)
        pooled = pool.value_.backward(query, vertices_grad_tensor.view(), states_grad.span(), value_grad.span(),
                                      query_grad.span(), stream.value_);
    suite.check(static_cast<bool>(pooled), "pool-parity pool backward: {}", pooled.description_);
    if (pooled) {
        cuStreamSynchronize(stream.value_.get());
        std::vector<f32_t> device_states_grad(host_states_grad.size());
        cuMemcpyDtoH(device_states_grad.data(), device_address(states_grad.data()),
                     device_states_grad.size() * sizeof(f32_t));
        std::vector<f64_t> device_query_grad(hidden_k);
        staging.resize(hidden_k);
        cuMemcpyDtoH(staging.data(), device_address(query_grad.data()), hidden_k * sizeof(nk_bf16_t));
        bool const query_widened = widen_from_bf16(staging, device_query_grad);
        std::vector<bool> live_rows(tokens_k, false);
        for (u32_t segment = 0; segment != segments_k; ++segment)
            for (u32_t token = 0; token != host_lengths[segment]; ++token)
                live_rows[host_offsets[segment] + token] = true;
        f32r2_view_t const device_states_grad_rows {device_states_grad.data(), tokens_k, hidden_k};
        f64r2_view_t const host_states_grad_rows {host_states_grad.data(), tokens_k, hidden_k};
        double states_peak = 0.0, states_worst = 0.0, query_peak = 0.0, query_worst = 0.0;
        bool gaps_zero = true;
        for (u32_t token = 0; token != tokens_k; ++token)
            for (u32_t channel = 0; channel != hidden_k; ++channel) {
                f64_t const device_value = device_states_grad_rows[token, channel];
                if (!live_rows[token]) {
                    gaps_zero = gaps_zero && device_value == 0.0;
                    continue;
                }
                states_peak = std::max(states_peak, std::abs(host_states_grad_rows[token, channel]));
                states_worst = std::max(states_worst, std::abs(host_states_grad_rows[token, channel] - device_value));
            }
        for (u32_t channel = 0; channel != hidden_k; ++channel) {
            query_peak = std::max(query_peak, std::abs(host_query_grad[channel]));
            query_worst = std::max(query_worst, std::abs(host_query_grad[channel] - device_query_grad[channel]));
        }
        suite.check(query_widened, "the query gradient widens");
        states_worst /= states_peak;
        query_worst /= query_peak;
        // NumKong rounds P and dS to BFloat16 for its tensor-core products, and the query gradient lands in
        // BFloat16: a few 2⁻⁹ unit roundoffs over F32 sums.
        suite.check(states_worst < 1e-2,
                    "pool d(states) matches the FP64 oracle on live rows to the BFloat16 floor: worst relative {:.3e}",
                    states_worst);
        suite.check(gaps_zero, "pool d(states) is exactly zero on gap rows");
        suite.check(query_worst < 1e-2,
                    "pool query gradient matches the FP64 oracle to the BFloat16 floor: worst relative {:.3e}",
                    query_worst);
    }

    // The quarter-precision twin: quantize the same tower and require every live vertex to stay on the
    // BFloat16 tower's direction — the blocks carry the E4M3 error, the tail is a BFloat16 snapshot.
    auto twin = tower.value_.quantize_weights(arena.value_, stream.value_);
    suite.check(static_cast<bool>(twin), "twin quantization: {}", twin.description_);
    if (twin) {
        bf16r2_tensor_arena_t twin_vertices =
            bf16r2_tensor_arena_t::uninitialized({segments_k, hidden_k}, gpu_arena_alloc<bf16_t> {arena.value_}).value;
        if (!twin_vertices) {
            suite.check(false, "twin vertices alloc");
            cuCtxPopCurrent(nullptr);
            return;
        }
        auto twin_ran = twin.value_.forward(batch.value_, twin_vertices.span(), stream.value_);
        suite.check(static_cast<bool>(twin_ran), "twin forward: {}", twin_ran.description_);
        if (twin_ran) {
            cuStreamSynchronize(stream.value_.get());
            std::vector<f64_t> device_twin_vertices(std::size_t(segments_k) * hidden_k);
            staging.resize(device_twin_vertices.size());
            cuMemcpyDtoH(staging.data(), device_address(twin_vertices.data()), staging.size() * sizeof(nk_bf16_t));
            suite.check(widen_from_bf16(staging, device_twin_vertices), "the twin vertices widen");
            f64r2_view_t const device_twin_vertices_rows {device_twin_vertices.data(), segments_k, hidden_k};
            double worst_cosine = 1.0;
            for (u32_t segment = 0; segment != segments_k; ++segment) {
                if (host_lengths[segment] == 0u) continue; // PAD vertices stay outside the parity
                f64r1_view_t const reference = device_vertices_rows[segment, nk::slice];
                f64r1_view_t const quantized = device_twin_vertices_rows[segment, nk::slice];
                worst_cosine = std::min(worst_cosine,
                                        dot(reference, quantized) /
                                            (std::sqrt(dot(reference, reference) * dot(quantized, quantized)) + 1e-30));
            }
            // Keyed-random weights are the quantizer's worst case (trained weights are smoother), and
            // this observed 0.979 at depth 1 already beats the R&D's FP8 embedding acceptance gates.
            suite.check(worst_cosine > 0.95,
                        "twin vertices stay on the BFloat16 tower's direction: worst cosine {:.4f}", worst_cosine);
        }
    }
    cuCtxPopCurrent(nullptr);
}

int main(int, char **argv) {
    test_suite_t suite = test_suite_t::make(argv[0]);
    [[maybe_unused]] CUresult const initialized = cuInit(0); // a failure leaves no device to count, so each case skips
    if (std::getenv("UFORM_HOST_DOMAIN_INDEX") != nullptr) {
        suite.run("cross_domain", run_cross_domain_case);
        return suite.finish();
    }

    suite.run("pool_parity", run_pool_parity_case);
    suite.run("oracle/self-edges", run_oracle_case, 8, 64, {64}, false, false, 1.0f);
    suite.run("oracle/matryoshka", run_oracle_case, 8, 64, {32, 64}, false, false, 1.0f);
    suite.run("oracle/explicit-edges", run_oracle_case, 8, 64, {64}, true, false, 0.5f);
    suite.run("oracle/pad-rows", run_oracle_case, 8, 64, {64}, false, true, 1.0f);
    suite.run("oracle/no-ambient", run_oracle_case, 8, 64, {64}, false, false, 0.0f);
    suite.run("oracle/odd-rows-wide", run_oracle_case, 33, 256, {32, 256}, false, false, 1.0f);
    suite.run("negatives", run_negative_cases);
    suite.run("sharded_oracle", run_sharded_oracle_case);
    suite.run("distributed", run_distributed_case);
    return suite.finish();
}
