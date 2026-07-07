/**
 *  @brief Conformance test of the contrastive force layout: the loss object against a Float64 host
 *      oracle, finite-difference force checks, bitwise capture-replay, relation-graph negatives, and the
 *      multi-GPU distributed step against the single-GPU global reference.
 *  @file cuda_test/contrastive.cu
 *  @author Ash Vardanian
 *  @date July 3, 2026
 *
 *  Plain `int main`, labelled pass/fail lines, non-zero exit on any failure. Tiers scale with the
 *  visible hardware: the single-GPU oracle matrix always runs; the distributed-step tier joins when at
 *  least two GPUs are visible.
 */
#include <cstring>
#include <cstdio>
#include <cmath>

#include <utility>
#include <vector>

#include <cuda.h> // `cuInit`, `cuCtxPushCurrent`, `cuMemcpyHtoD`, `cuMemsetD8` — the raw staging boundary

#include "cuda_core/infiniband_collectives.cuh" // `infiniband_coordinates_t` — the corrupt-payload negative test
#include "cuda_core/contrastive_segments.cuh"
#include "cuda_core/encoder_tower.hpp" // `encoder_tower_bf16_t` — the pool + head the FP64 parity case gates

using namespace unum::uform::core;

static int failures = 0;

static void check(bool condition, char const *label) {
    std::printf("  [%s] %s\n", condition ? "pass" : "FAIL", label);
    if (!condition) ++failures;
}

/** @brief Deterministic pseudo-random filler in roughly [-0.5, 0.5). */
static f32_t hash_fill(std::size_t index, std::size_t salt) {
    std::size_t const mixed = (index * 2654435761ull + salt * 40503ull + 12345ull) % 1000ull;
    return static_cast<f32_t>(mixed) / 1000.0f - 0.5f;
}

/** @brief The host-side pair resolver — mirrors `contrastive_pair_relation` decision for decision. */
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

/** @brief The Float64 oracle: the exact per-header math of the segments loss over host BFloat16 towers. */
static f64_t host_loss(std::vector<bf16_t> const &vertices, std::vector<bf16_t> const &neighbors,
                       std::size_t num_vertices, std::size_t num_neighbors, std::size_t hidden,
                       std::vector<u32_t> const &prefixes, host_relations_t const &relations, f64_t beta, f64_t bias,
                       std::size_t vertices_per_mesh) {
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
            std::size_t const width = prefixes[prefix_index];
            f64_t vertex_norm = 0.0;
            for (std::size_t channel = 0; channel != width; ++channel) {
                f64_t const value = f64_t(__bfloat162float(vertices[vertex * hidden + channel]));
                vertex_norm += value * value;
            }
            vertex_norm = std::sqrt(vertex_norm);
            for (std::size_t neighbor = 0; neighbor != num_neighbors; ++neighbor) {
                auto [target, weight] = relations.resolve(u32_t(vertex), u32_t(neighbor));
                if (weight == 0.0) continue;
                f64_t raw = 0.0, neighbor_norm = 0.0;
                for (std::size_t channel = 0; channel != width; ++channel) {
                    f64_t const ours = f64_t(__bfloat162float(vertices[vertex * hidden + channel]));
                    f64_t const theirs = f64_t(__bfloat162float(neighbors[neighbor * hidden + channel]));
                    raw += ours * theirs;
                    neighbor_norm += theirs * theirs;
                }
                neighbor_norm = std::sqrt(neighbor_norm);
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

/** @brief One single-GPU oracle case: run the loss, compare to the host oracle, finite-difference the
 *      force on a few sampled channels, and prove eager == replayed bitwise. */
static void run_oracle_case(char const *label, std::size_t num_vertices, std::size_t hidden,
                            std::vector<u32_t> const &prefixes, bool explicit_edges, bool with_pad,
                            f32_t ambient_weight) {
    std::printf("case %s (V=%zu H=%zu K=%zu edges=%d pad=%d ambient=%g)\n", label, num_vertices, hidden,
                prefixes.size(), int(explicit_edges), int(with_pad), ambient_weight);
    auto stream_made = gpu_stream_t::make(0);
    if (!stream_made) return check(false, "stream");
    gpu_stream_t stream = std::move(stream_made.value_);
    cuCtxPushCurrent(stream.context());
    auto arena_made = gpu_arena_t::make(0, 1ull << 30);
    if (!arena_made) return check(false, "arena");
    gpu_arena_t arena = std::move(arena_made.value_);
    auto blas_made = cublaslt_session_t::make(algo_pick_t::heuristic_k, nullptr);
    if (!blas_made) return check(false, "blas session");
    cublaslt_session_t blas = std::move(blas_made.value_);

    contrastive_segments_config_t config;
    config.from_vertices_per_gpu = u32_t(num_vertices);
    config.hidden = u32_t(hidden);
    config.to_vertices_per_gpu = u32_t(num_vertices);
    config.gpu_subgraphs_per_step = 1;
    auto loss_made = contrastive_segments_t::make(arena, blas, config, {prefixes.data(), prefixes.size()});
    if (!loss_made) {
        std::printf("    %s\n", loss_made.description_.c_str());
        return check(false, "loss make");
    }
    contrastive_segments_t loss = std::move(loss_made.value_);

    // Towers, gradients, loss scalar.
    std::size_t const scalars = num_vertices * hidden;
    std::vector<bf16_t> host_tower(scalars);
    for (std::size_t index = 0; index != scalars; ++index)
        host_tower[index] = __float2bfloat16(hash_fill(index, num_vertices + hidden));
    bf16_t *tower = arena.allocate_object<bf16_t>(scalars);
    bf16_t *grad = arena.allocate_object<bf16_t>(scalars);
    f32_t *grad_f32 = arena.allocate_object<f32_t>(scalars);
    f32_t *loss_out = arena.allocate_object<f32_t>(1);
    u32_t *effective = arena.allocate_object<u32_t>(num_vertices);
    cuMemcpyHtoD(reinterpret_cast<CUdeviceptr>(tower), host_tower.data(), scalars * sizeof(bf16_t));

    // Relations: self-edge default, or explicit attract/repel with an ignore override; PAD via the
    // effective-length columns.
    host_relations_t host_relations;
    host_relations.ambient = ambient_weight;
    auto relations_made = contrastive_relations_t::make(arena, u32_t(num_vertices), 16);
    if (!relations_made) return check(false, "relations make");
    contrastive_relations_t relations = std::move(relations_made.value_);
    std::vector<u32_t> host_effective(num_vertices, 1u);
    if (with_pad) host_effective[num_vertices - 1] = 0u;
    cuMemcpyHtoD(reinterpret_cast<CUdeviceptr>(effective), host_effective.data(), num_vertices * sizeof(u32_t));
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
        staged = relations.upload({host_relations.attract_offsets.data(), host_relations.attract_offsets.size()},
                                  {host_relations.attract_ids.data(), host_relations.attract_ids.size()},
                                  {host_relations.attract_scores.data(), host_relations.attract_scores.size()},
                                  {host_relations.repel_offsets.data(), host_relations.repel_offsets.size()},
                                  {host_relations.repel_ids.data(), host_relations.repel_ids.size()},
                                  {host_relations.repel_scores.data(), host_relations.repel_scores.size()},
                                  {host_relations.ignore_offsets.data(), host_relations.ignore_offsets.size()},
                                  {host_relations.ignore_ids.data(), host_relations.ignore_ids.size()}, effective_view,
                                  effective_view, num_vertices, 0, ambient_weight, stream);
    }
    else
        staged = relations.upload_self_edges(u32_t(num_vertices), effective_view, effective_view, 0, ambient_weight,
                                             stream);
    if (!staged) {
        std::printf("    %s\n", staged.description_.c_str());
        return check(false, "relations upload");
    }

    gpu_subgraph_t subgraph;
    subgraph.to_vertices = bf16r2_view_t {tower, num_vertices, hidden};
    subgraph.global_offset = 0;
    subgraph.to_vertices_grad = f32r2_span_t {grad_f32, num_vertices, hidden};
    contrastive_recipe_t recipe;
    recipe.ambient_repulsion_weight = ambient_weight;

    auto run_once = [&]() noexcept -> expected_t {
        return loss(bf16r2_view_t {tower, num_vertices, hidden}, {&subgraph, 1}, relations.view(), recipe, num_vertices,
                    f32r1_span_t {loss_out, 1}, bf16r2_span_t {grad, num_vertices, hidden}, stream);
    };
    if (expected_t ran = run_once(); !ran) {
        std::printf("    %s\n", ran.description_.c_str());
        return check(false, "loss enqueue");
    }
    cuStreamSynchronize(stream.get());
    f32_t device_loss = 0;
    cuMemcpyDtoH(&device_loss, reinterpret_cast<CUdeviceptr>(loss_out), sizeof(f32_t));

    std::vector<u32_t> host_prefixes(prefixes.begin(), prefixes.end());
    f64_t const oracle = host_loss(host_tower, host_tower, num_vertices, num_vertices, hidden, host_prefixes,
                                   host_relations, 1.0, 0.0, num_vertices);
    f64_t const relative = std::fabs(f64_t(device_loss) - oracle) / std::max(std::fabs(oracle), 1.0e-9);
    std::printf("    loss device=%.6f oracle=%.6f relative=%.2e\n", device_loss, oracle, relative);
    check(relative < 5.0e-3, "loss matches the Float64 oracle");

    // Finite-difference force check on a few sampled channels: the analytic BF16 gradient must sit
    // within BF16 resolution of (L(x+h) - L(x-h)) / 2h.
    std::vector<bf16_t> host_grad(scalars);
    cuMemcpyDtoH(host_grad.data(), reinterpret_cast<CUdeviceptr>(grad), scalars * sizeof(bf16_t));
    std::vector<f32_t> host_column(scalars);
    cuMemcpyDtoH(host_column.data(), reinterpret_cast<CUdeviceptr>(grad_f32), scalars * sizeof(f32_t));
    bool forces_ok = true;
    for (std::size_t sample = 0; sample != 6 && forces_ok; ++sample) {
        std::size_t const index = (sample * 977ull + 13ull) % scalars;
        f64_t const step_size = 1.0e-2;
        std::vector<bf16_t> plus = host_tower, minus = host_tower;
        plus[index] = __float2bfloat16(__bfloat162float(host_tower[index]) + f32_t(step_size));
        minus[index] = __float2bfloat16(__bfloat162float(host_tower[index]) - f32_t(step_size));
        f64_t const h = f64_t(__bfloat162float(plus[index])) - f64_t(__bfloat162float(minus[index]));
        f64_t const numeric = (host_loss(plus, plus, num_vertices, num_vertices, hidden, host_prefixes, host_relations,
                                         1.0, 0.0, num_vertices) -
                               host_loss(minus, minus, num_vertices, num_vertices, hidden, host_prefixes,
                                         host_relations, 1.0, 0.0, num_vertices)) /
                              h;
        // The full derivative sums the row-side (bf16 output) and column-side (f32 subgraph) forces.
        f64_t const analytic = f64_t(__bfloat162float(host_grad[index])) + f64_t(host_column[index]);
        f64_t const difference = std::fabs(numeric - analytic);
        if (difference > 5.0e-2 * std::max(std::fabs(numeric), 0.05)) {
            std::printf("    force[%zu]: numeric=%.5f analytic=%.5f\n", index, numeric, analytic);
            forces_ok = false;
        }
    }
    check(forces_ok, "forces match central finite differences");

    // Bitwise determinism: the same call twice produces identical bytes.
    std::vector<bf16_t> first_grad = host_grad;
    if (expected_t ran = run_once(); !ran) return check(false, "second enqueue");
    cuStreamSynchronize(stream.get());
    cuMemcpyDtoH(host_grad.data(), reinterpret_cast<CUdeviceptr>(grad), scalars * sizeof(bf16_t));
    check(std::memcmp(first_grad.data(), host_grad.data(), scalars * sizeof(bf16_t)) == 0,
          "two eager runs are bitwise identical");

    // Capture once, replay twice — still bitwise identical to eager.
    auto graph_made = gpu_replay_graph_t::make(stream.get(), [&]() noexcept { return run_once(); });
    if (!graph_made) {
        std::printf("    %s\n", graph_made.description_.c_str());
        return check(false, "capture");
    }
    for (int replay = 0; replay != 2; ++replay)
        if (expected_t launched = graph_made.value_.launch(stream.get()); !launched)
            return check(false, "replay launch");
    cuStreamSynchronize(stream.get());
    cuMemcpyDtoH(host_grad.data(), reinterpret_cast<CUdeviceptr>(grad), scalars * sizeof(bf16_t));
    check(std::memcmp(first_grad.data(), host_grad.data(), scalars * sizeof(bf16_t)) == 0,
          "captured replays are bitwise identical to eager");
    cuCtxPopCurrent(nullptr);
}

/** @brief Negative tests: every invalid staging must fail loudly, never fall back. */
static void run_negative_cases() {
    std::printf("case negatives\n");
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
    check(!unsorted, "unsorted run is rejected loudly");
    u32_t overflowing_ids[6] = {0, 1, 2, 3, 4, 5};
    u32_t overflow_offsets[5] = {0, 6, 6, 6, 6};
    f32_t overflow_scores[6] = {1, 1, 1, 1, 1, 1};
    auto overflow = relations.value_.upload({overflow_offsets, 5}, {overflowing_ids, 6}, {overflow_scores, 6},
                                            {empty_offsets, 5}, {}, {}, {}, {}, {}, {}, 8, 0, 1.0f, stream.value_);
    check(!overflow, "edge-capacity overflow is rejected loudly");
    infiniband_coordinates_t coordinates;
    check(!infiniband_coordinates_t::from_text("lid banana\n", coordinates), "corrupt coordinates are rejected");
    cuCtxPopCurrent(nullptr);
}

/** @brief The distributed tier: 2 GPUs vs the single-GPU global reference on identical data. */
static void run_distributed_case() {
    int device_count = 0;
    cuDeviceGetCount(&device_count);
    if (device_count < 2) {
        std::printf("case distributed: SKIP (only %d GPU visible)\n", device_count);
        return;
    }
    std::printf("case distributed (2 GPUs vs single-GPU global reference)\n");
    constexpr std::size_t gpus_k = 2;
    constexpr u32_t vertices_k = 8, hidden_k = 64;
    constexpr std::size_t scalars_k = std::size_t(vertices_k) * hidden_k;
    i32_t devices[gpus_k] = {0, 1};
    u32_t prefix_dims[1] = {hidden_k};

    // Global data: 16 vertices; rank r owns rows [8r, 8r+8).
    std::vector<bf16_t> global_tower(2 * scalars_k);
    for (std::size_t index = 0; index != global_tower.size(); ++index)
        global_tower[index] = __float2bfloat16(hash_fill(index, 7));

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
    per_gpu_array<cublaslt_session_t> sessions;
    for (std::size_t rank = 0; rank != gpus_k; ++rank) {
        auto stream_made = gpu_stream_t::make(devices[rank]);
        if (!stream_made || streams.try_push_back(std::move(stream_made.value_)) != status_t::success_k) {
            check(false, "distributed stream make");
            return;
        }
        cuCtxPushCurrent(streams[rank].context());
        auto arena_made = gpu_arena_t::make(devices[rank], 1ull << 30, {devices, gpus_k});
        if (!arena_made || arenas.try_push_back(std::move(arena_made.value_)) != status_t::success_k) {
            check(false, "distributed arena make");
            return;
        }
        auto session_made = cublaslt_session_t::make(algo_pick_t::heuristic_k, nullptr);
        if (!session_made || sessions.try_push_back(std::move(session_made.value_)) != status_t::success_k) {
            check(false, "distributed session make");
            return;
        }
        bf16_t *tower = arenas[rank].allocate_object<bf16_t>(scalars_k);
        bf16_t *grad = arenas[rank].allocate_object<bf16_t>(scalars_k);
        cuMemcpyHtoD(reinterpret_cast<CUdeviceptr>(tower), global_tower.data() + rank * scalars_k,
                     scalars_k * sizeof(bf16_t));
        towers[rank] = bf16r2_view_t {tower, vertices_k, hidden_k};
        grads[rank] = bf16r2_span_t {grad, vertices_k, hidden_k};
        owners[rank] = std::move(contrastive_relations_t::make(arenas[rank], vertices_k, 0).value_);
        auto staged = owners[rank].upload_self_edges(vertices_k, {}, {}, rank * std::size_t(vertices_k), 1.0f,
                                                     streams[rank]);
        if (!staged) {
            check(false, "distributed relations");
            return;
        }
        relations[rank] = owners[rank].view();
        cuCtxPopCurrent(nullptr);
    }
    // A one-process world: the whole box is one host_domain and one multimem_domain (D=1), so the
    // colocated domain is the trivial identity and no exchange happens.
    auto topology = mesh_topology_t::bootstrap(0, 1, 0, "/tmp/uform_exchange_local", "distributed_case");
    if (!topology) {
        std::printf("    %s\n", topology.description_.c_str());
        check(false, "single-domain bootstrap");
        return;
    }
    auto domain = multimem_domain_t::make(topology.value_, {devices, gpus_k},
                                          distributed_contrastive_segments_t::payload_scalars_needed(config), streams);
    if (!domain) {
        std::printf("    %s\n", domain.description_.c_str());
        check(false, "multimem domain make");
        return;
    }
    auto colocated = colocated_domain_t::make(topology.value_);
    if (!colocated) {
        std::printf("    %s\n", colocated.description_.c_str());
        check(false, "colocated domain make");
        return;
    }
    auto distributed = distributed_contrastive_segments_t::make(domain.value_, colocated.value_, arenas, streams,
                                                                sessions, {towers, gpus_k}, {relations, gpus_k},
                                                                {grads, gpus_k}, config, {prefix_dims, 1});
    if (!distributed) {
        std::printf("    %s\n", distributed.description_.c_str());
        check(false, "distributed make");
        return;
    }
    auto ran = distributed.value_(contrastive_recipe_t {}, streams);
    if (!ran) {
        std::printf("    %s\n", ran.description_.c_str());
        check(false, "distributed run");
        return;
    }
    for (std::size_t rank = 0; rank != gpus_k; ++rank) {
        cuCtxPushCurrent(streams[rank].context());
        cuStreamSynchronize(streams[rank].get());
        cuCtxPopCurrent(nullptr);
    }
    f64_t distributed_loss = 0.0;
    std::vector<bf16_t> distributed_grad(2 * scalars_k);
    cuCtxPushCurrent(streams[0].context());
    f32_t summands[gpus_k] = {};
    cuMemcpyDtoH(summands, reinterpret_cast<CUdeviceptr>(distributed.value_.loss_outs().data()), sizeof(summands));
    for (std::size_t rank = 0; rank != gpus_k; ++rank) {
        distributed_loss += summands[rank];
        cuMemcpyDtoH(distributed_grad.data() + rank * scalars_k, reinterpret_cast<CUdeviceptr>(grads[rank].data()),
                     scalars_k * sizeof(bf16_t));
    }
    cuCtxPopCurrent(nullptr);

    // The Float64 oracle over the SAME global grid — the sum of rank summands must match it.
    host_relations_t host_relations;
    std::vector<u32_t> prefixes = {hidden_k};
    f64_t const oracle = host_loss(global_tower, global_tower, 2 * vertices_k, 2 * vertices_k, hidden_k, prefixes,
                                   host_relations, 1.0, 0.0, 2 * vertices_k);
    f64_t const relative = std::fabs(distributed_loss - oracle) / std::max(std::fabs(oracle), 1.0e-9);
    std::printf("    loss distributed=%.6f oracle=%.6f relative=%.2e (tier=%s)\n", distributed_loss, oracle, relative,
                domain.value_.tier() == multimem_tier_t::switch_reduced_k ? "switch_reduced" : "peer_read");
    check(relative < 5.0e-3, "distributed loss sum matches the global Float64 oracle");

    // Finite-difference one channel of rank 1's tower against the combined distributed force.
    std::size_t const probe_index = scalars_k + 321; // Inside rank 1's rows.
    f64_t const step_size = 1.0e-2;
    std::vector<bf16_t> plus = global_tower, minus = global_tower;
    plus[probe_index] = __float2bfloat16(__bfloat162float(global_tower[probe_index]) + f32_t(step_size));
    minus[probe_index] = __float2bfloat16(__bfloat162float(global_tower[probe_index]) - f32_t(step_size));
    f64_t const h = f64_t(__bfloat162float(plus[probe_index])) - f64_t(__bfloat162float(minus[probe_index]));
    f64_t const numeric = (host_loss(plus, plus, 2 * vertices_k, 2 * vertices_k, hidden_k, prefixes, host_relations,
                                     1.0, 0.0, 2 * vertices_k) -
                           host_loss(minus, minus, 2 * vertices_k, 2 * vertices_k, hidden_k, prefixes, host_relations,
                                     1.0, 0.0, 2 * vertices_k)) /
                          h;
    f64_t const analytic = f64_t(__bfloat162float(distributed_grad[probe_index]));
    std::printf("    force probe: numeric=%.5f distributed=%.5f\n", numeric, analytic);
    check(std::fabs(numeric - analytic) < 5.0e-2 * std::max(std::fabs(numeric), 0.05),
          "distributed force matches the global finite difference");
}

/** @brief The cross-domain tier: one process per multimem_domain, GDAKI gather + all-reduce over the
 *      local HCAs, verified against the same global Float64 oracle each process computes independently.
 *      Activated by UFORM_HOST_DOMAIN_INDEX in the environment. */
static int run_cross_domain_case() {
#if !defined(UFORM_WITH_INFINIBAND)
    std::printf("cross-domain: SKIP (built without UFORM_WITH_INFINIBAND)\n");
    return 0;
#else
    int device_count = 0;
    cuDeviceGetCount(&device_count);
    u32_t const gpus = u32_t(device_count);
    u32_t const domain = u32_t(std::atoi(std::getenv("UFORM_HOST_DOMAIN_INDEX")));
    std::printf("case cross-domain (domain %u, %u local GPUs, 2 domains over GDAKI)\n", domain, gpus);
    auto topology_made = mesh_topology_t::bootstrap(domain, 2, gpus);
    if (!topology_made) {
        std::printf("  bootstrap: %s\n", topology_made.description_.c_str());
        return 1;
    }
    mesh_topology_t topology = std::move(topology_made.value_);

    constexpr u32_t vertices_k = 8, hidden_k = 64;
    std::size_t const scalars_k = std::size_t(vertices_k) * hidden_k;
    u32_t const world_gpus = 2 * gpus;
    u32_t prefix_dims[1] = {hidden_k};
    std::vector<bf16_t> global_tower(world_gpus * scalars_k);
    for (std::size_t index = 0; index != global_tower.size(); ++index)
        global_tower[index] = __float2bfloat16(hash_fill(index, 7));

    per_gpu_array<gpu_stream_t> streams;
    per_gpu_array<gpu_arena_t> arenas;
    per_gpu_array<cublaslt_session_t> sessions;
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
        if (!stream_made || streams.try_push_back(std::move(stream_made.value_)) != status_t::success_k) return 1;
        cuCtxPushCurrent(streams[rank].context());
        auto arena_made = gpu_arena_t::make(i32_t(rank), 1ull << 30, {devices.data(), gpus});
        if (!arena_made || arenas.try_push_back(std::move(arena_made.value_)) != status_t::success_k) return 1;
        auto session_made = cublaslt_session_t::make(algo_pick_t::heuristic_k, nullptr);
        if (!session_made || sessions.try_push_back(std::move(session_made.value_)) != status_t::success_k) return 1;
        bf16_t *tower = arenas[rank].allocate_object<bf16_t>(scalars_k);
        bf16_t *grad = arenas[rank].allocate_object<bf16_t>(scalars_k);
        std::size_t const global_rank = std::size_t(domain) * gpus + rank;
        cuMemcpyHtoD(reinterpret_cast<CUdeviceptr>(tower), global_tower.data() + global_rank * scalars_k,
                     scalars_k * sizeof(bf16_t));
        towers[rank] = bf16r2_view_t {tower, vertices_k, hidden_k};
        grads[rank] = bf16r2_span_t {grad, vertices_k, hidden_k};
        owners[rank] = std::move(contrastive_relations_t::make(arenas[rank], vertices_k, 0).value_);
        auto staged = owners[rank].upload_self_edges(vertices_k, {}, {}, global_rank * std::size_t(vertices_k), 1.0f,
                                                     streams[rank]);
        if (!staged) {
            check(false, "cross-domain relations");
            return 1;
        }
        relations[rank] = owners[rank].view();
        cuCtxPopCurrent(nullptr);
    }
    // The composition roots, in main-order: the ONE multimem domain, the ONE colocated domain.
    auto multimem = multimem_domain_t::make(
        topology, {devices.data(), gpus}, distributed_contrastive_segments_t::payload_scalars_needed(config), streams);
    if (!multimem) {
        std::printf("  multimem: %s\n", multimem.description_.c_str());
        check(false, "multimem domain make");
        return 1;
    }
    auto colocated = colocated_domain_t::make(topology);
    if (!colocated) {
        std::printf("  colocated: %s\n", colocated.description_.c_str());
        check(false, "colocated domain make");
        return 1;
    }
    auto distributed = distributed_contrastive_segments_t::make(
        multimem.value_, colocated.value_, arenas, streams, sessions, {towers.data(), gpus}, {relations.data(), gpus},
        {grads.data(), gpus}, config, {prefix_dims, 1});
    if (!distributed) {
        std::printf("  make: %s\n", distributed.description_.c_str());
        check(false, "cross-domain make");
        return 1;
    }
    std::printf("  fabric wired (domains=%u, tier=%s)\n", distributed.value_.world_domains(),
                multimem.value_.tier() == multimem_tier_t::switch_reduced_k ? "switch_reduced" : "peer_read");
    auto ran = distributed.value_(contrastive_recipe_t {}, streams);
    if (!ran) {
        std::printf("  run: %s\n", ran.description_.c_str());
        check(false, "cross-domain run");
        return 1;
    }
    for (u32_t rank = 0; rank != gpus; ++rank) {
        cuCtxPushCurrent(streams[rank].context());
        CUresult const synchronized = cuStreamSynchronize(streams[rank].get());
        cuCtxPopCurrent(nullptr);
        if (synchronized != CUDA_SUCCESS) {
            std::printf("  sync failed (rank=%u result=%d)\n", rank, synchronized);
            check(false, "cross-domain sync");
            return 1;
        }
    }

    // Combine the two domains' loss sums over the mesh exchange; each process checks the global oracle.
    f64_t domain_loss = 0.0;
    {
        cuCtxPushCurrent(streams[0].context());
        std::vector<f32_t> summands(gpus);
        cuMemcpyDtoH(summands.data(), reinterpret_cast<CUdeviceptr>(distributed.value_.loss_outs().data()),
                     gpus * sizeof(f32_t));
        cuCtxPopCurrent(nullptr);
        for (f32_t const summand : summands) domain_loss += summand;
    }
    sz::string loss_text;
    if (!format_into(loss_text, "{}", domain_loss)) return 1;
    if (!topology.publish_text("loss", loss_text.view())) return 1;
    sz::string peer_text;
    if (!topology.await_text(1u - domain, "loss", peer_text, 60.0)) return 1;
    f64_t const global_loss = domain_loss + std::atof(peer_text.c_str());

    host_relations_t host_relations;
    std::vector<u32_t> prefixes = {hidden_k};
    f64_t const oracle = host_loss(global_tower, global_tower, world_gpus * vertices_k, world_gpus * vertices_k,
                                   hidden_k, prefixes, host_relations, 1.0, 0.0, world_gpus * vertices_k);
    f64_t const relative = std::fabs(global_loss - oracle) / std::max(std::fabs(oracle), 1.0e-9);
    std::printf("  loss global=%.6f oracle=%.6f relative=%.2e\n", global_loss, oracle, relative);
    check(relative < 5.0e-3, "cross-domain loss sum matches the global Float64 oracle");

    // Finite-difference one locally-owned channel against the fully-reduced force.
    std::size_t const probe_index = (std::size_t(domain) * gpus + 1) * scalars_k + 321;
    std::vector<bf16_t> local_grad(scalars_k);
    cuCtxPushCurrent(streams[1 % gpus].context());
    cuMemcpyDtoH(local_grad.data(), reinterpret_cast<CUdeviceptr>(grads[1 % gpus].data()), scalars_k * sizeof(bf16_t));
    cuCtxPopCurrent(nullptr);
    f64_t const step_size = 1.0e-2;
    std::vector<bf16_t> plus = global_tower, minus = global_tower;
    plus[probe_index] = __float2bfloat16(__bfloat162float(global_tower[probe_index]) + f32_t(step_size));
    minus[probe_index] = __float2bfloat16(__bfloat162float(global_tower[probe_index]) - f32_t(step_size));
    f64_t const h = f64_t(__bfloat162float(plus[probe_index])) - f64_t(__bfloat162float(minus[probe_index]));
    f64_t const numeric = (host_loss(plus, plus, world_gpus * vertices_k, world_gpus * vertices_k, hidden_k, prefixes,
                                     host_relations, 1.0, 0.0, world_gpus * vertices_k) -
                           host_loss(minus, minus, world_gpus * vertices_k, world_gpus * vertices_k, hidden_k, prefixes,
                                     host_relations, 1.0, 0.0, world_gpus * vertices_k)) /
                          h;
    f64_t const analytic = f64_t(__bfloat162float(local_grad[probe_index % scalars_k]));
    std::printf("  force probe: numeric=%.5f distributed=%.5f\n", numeric, analytic);
    check(std::fabs(numeric - analytic) < 5.0e-2 * std::max(std::fabs(numeric), 0.05),
          "cross-domain force matches the global finite difference");
    if (!topology.barrier("done", 60.0)) return 1;
    std::printf("%s: %d failure(s)\n", failures == 0 ? "PASS" : "FAIL", failures);
    return failures == 0 ? 0 : 1;
#endif
}

/** @brief The FP64 attention-pool + head parity gate: seed a tiny text tower with keyed weights, run the
 *      contrastive forward over a fabricated ragged batch (one PAD segment included — the historical pool
 *      failure mode was PAD masking, never arithmetic), and compare the emitted vertices against a
 *      double-precision host replica of exactly the new math — the softmax pool over the tower's own
 *      final-norm states, then RMSNorm → fc1 → SiLU → fc2. The backward runs the same chain in reverse:
 *      every pool / head weight gradient must match the host fold to the BFloat16 floor. */
static void run_pool_parity_case() {
    std::printf("case pool-parity (FP64 host oracle over the pool + head)\n");
    constexpr u32_t rows_k = 2, segments_k = 4, columns_k = 8, hidden_k = 256, tokens_k = rows_k * columns_k;
    constexpr u32_t heads_k = hidden_k / transformer_head_dim_k;
    auto stream = gpu_stream_t::make(0);
    if (!stream) {
        check(false, "pool-parity stream make");
        return;
    }
    cuCtxPushCurrent(stream.value_.context());
    auto arena = gpu_arena_t::make(0, 1ull << 30, {});
    auto blas = cublaslt_session_t::make(algo_pick_t::heuristic_k, nullptr);
    auto cudnn = cudnn_session_t::make(algo_pick_t::heuristic_k);
    if (!arena || !blas || !cudnn) {
        check(false, "pool-parity bring-up");
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
    auto tower = encoder_tower_bf16_t::make(arena.value_, blas.value_, cudnn.value_, config, shape);
    if (!tower) {
        std::printf("    %s\n", tower.description_.c_str());
        check(false, "pool-parity tower make");
        cuCtxPopCurrent(nullptr);
        return;
    }

    // Keyed weight seeding through the registration list — the same values the host replica reads back.
    array<differentiable_weights_t> weights;
    if (!tower.value_.fill(weights)) {
        check(false, "pool-parity registration");
        cuCtxPopCurrent(nullptr);
        return;
    }
    std::vector<bf16_t> staging;
    for (std::size_t matrix = 0; matrix != weights.size(); ++matrix) {
        std::size_t const scalars = weights[matrix].weights.extent(0) * weights[matrix].weights.extent(1);
        staging.resize(scalars);
        for (std::size_t index = 0; index != scalars; ++index)
            staging[index] = __float2bfloat16(hash_fill(index, matrix + 101));
        cuMemcpyHtoD(reinterpret_cast<CUdeviceptr>(weights[matrix].weights.data()), staging.data(),
                     scalars * sizeof(bf16_t));
    }

    // A fabricated ragged batch: three live text segments of lengths {8, 5, 3} and one PAD — the batch
    // packer's contract (offsets repeat on PAD, lengths carry the live counts).
    auto batch = multimodal_batch_on_gpu_t::make(gpu_arena_alloc<b8_t> {arena.value_}, shape);
    if (!batch) {
        check(false, "pool-parity batch make");
        cuCtxPopCurrent(nullptr);
        return;
    }
    u32_t const offsets_host[segments_k + 1] = {0, 8, 13, 16, 16};
    u32_t const lengths_host[segments_k] = {8, 5, 3, 0};
    modality_t const modality_host[segments_k] = {modality_t::text_k, modality_t::text_k, modality_t::text_k,
                                                  modality_t::text_k};
    std::vector<token_id_t> tokens_host(tokens_k);
    for (u32_t token = 0; token != tokens_k; ++token) tokens_host[token] = static_cast<token_id_t>(token % 251u);
    cuMemcpyHtoD(reinterpret_cast<CUdeviceptr>(batch.value_.segment_offsets.data()), offsets_host,
                 sizeof(offsets_host));
    cuMemcpyHtoD(reinterpret_cast<CUdeviceptr>(batch.value_.segment_lengths.data()), lengths_host,
                 sizeof(lengths_host));
    cuMemcpyHtoD(reinterpret_cast<CUdeviceptr>(batch.value_.segment_modality.data()), modality_host,
                 sizeof(modality_host));
    cuMemcpyHtoD(reinterpret_cast<CUdeviceptr>(batch.value_.tokens.data()), tokens_host.data(),
                 tokens_k * sizeof(token_id_t));

    // Forward with vertices, then read back the tower's own final-norm states — the oracle's input, so
    // the parity isolates exactly the new pool + head math.
    std::vector<bf16_t> vertices_device(std::size_t(segments_k) * hidden_k);
    bf16r2_tensor_arena_t vertices_tensor = bf16r2_tensor_arena_t::try_empty({segments_k, hidden_k},
                                                                             gpu_arena_alloc<bf16_t> {arena.value_});
    bf16r2_tensor_arena_t vertices_grad_tensor = bf16r2_tensor_arena_t::try_empty(
        {segments_k, hidden_k}, gpu_arena_alloc<bf16_t> {arena.value_});
    if (!vertices_tensor || !vertices_grad_tensor) {
        check(false, "pool-parity vertices alloc");
        cuCtxPopCurrent(nullptr);
        return;
    }
    cuMemsetD8(reinterpret_cast<CUdeviceptr>(vertices_tensor.data()), 0,
               std::size_t(segments_k) * hidden_k * sizeof(bf16_t));
    cuMemsetD8(reinterpret_cast<CUdeviceptr>(vertices_grad_tensor.data()), 0,
               std::size_t(segments_k) * hidden_k * sizeof(bf16_t));
    auto ran = tower.value_.forward(batch.value_, vertices_tensor.span(), stream.value_);
    if (!ran) std::printf("    %s\n", ran.description_.c_str());
    check(static_cast<bool>(ran), "pool-parity forward");
    if (!ran) {
        cuCtxPopCurrent(nullptr);
        return;
    }
    cuStreamSynchronize(stream.value_.get());

    std::vector<bf16_t> states_host(std::size_t(tokens_k) * hidden_k);
    cuMemcpyDtoH(states_host.data(), reinterpret_cast<CUdeviceptr>(tower.value_.normalized().data()),
                 states_host.size() * sizeof(bf16_t));
    cuMemcpyDtoH(vertices_device.data(), reinterpret_cast<CUdeviceptr>(vertices_tensor.data()),
                 vertices_device.size() * sizeof(bf16_t));

    // Host replica in double precision: per-(segment,head) softmax pool, then RMSNorm → fc1 → SiLU → fc2.
    // Weight order in the registration list: ... final_norm, pool_query, head.norm, head.fc1, head.fc2.
    std::size_t const query_index = weights.size() - 4, head_norm_index = weights.size() - 3;
    std::size_t const fc1_index = weights.size() - 2, fc2_index = weights.size() - 1;
    auto weight_at = [&](std::size_t matrix, std::size_t index) {
        return double(__bfloat162float(__float2bfloat16(hash_fill(index, matrix + 101))));
    };
    double const scale = 1.0 / std::sqrt(double(transformer_head_dim_k));
    std::vector<double> pooled_host(std::size_t(segments_k) * hidden_k, 0.0);
    for (u32_t segment = 0; segment != segments_k; ++segment) {
        u32_t const length = lengths_host[segment];
        if (length == 0u) continue; // a PAD vertex is don't-care — the loss PADs it out, so no oracle here
        u32_t const offset = offsets_host[segment];
        u32_t const pool_length = length;
        for (u32_t head = 0; head != heads_k; ++head) {
            double score_max = -1e300;
            std::vector<double> scores(pool_length);
            for (u32_t token = 0; token != pool_length; ++token) {
                double dot = 0.0;
                for (u32_t channel = 0; channel != transformer_head_dim_k; ++channel) {
                    u32_t const base = head * transformer_head_dim_k + channel;
                    dot += weight_at(query_index, base) *
                           double(__bfloat162float(states_host[std::size_t(offset + token) * hidden_k + base]));
                }
                scores[token] = dot * scale;
                score_max = std::max(score_max, scores[token]);
            }
            double denominator = 0.0;
            for (u32_t token = 0; token != pool_length; ++token) denominator += std::exp(scores[token] - score_max);
            for (u32_t channel = 0; channel != transformer_head_dim_k; ++channel) {
                u32_t const base = head * transformer_head_dim_k + channel;
                double value = 0.0;
                for (u32_t token = 0; token != pool_length; ++token)
                    value += std::exp(scores[token] - score_max) / denominator *
                             double(__bfloat162float(states_host[std::size_t(offset + token) * hidden_k + base]));
                pooled_host[std::size_t(segment) * hidden_k + base] = value;
            }
        }
    }
    std::vector<double> vertices_host(std::size_t(segments_k) * hidden_k, 0.0);
    for (u32_t segment = 0; segment != segments_k; ++segment) {
        if (lengths_host[segment] == 0u) continue; // PAD vertices stay outside the parity
        double sum_squares = 0.0;
        for (u32_t channel = 0; channel != hidden_k; ++channel) {
            double const value = pooled_host[std::size_t(segment) * hidden_k + channel];
            sum_squares += value * value;
        }
        double const inverse_rms = 1.0 / std::sqrt(sum_squares / hidden_k + 1e-6);
        std::vector<double> normalized(hidden_k), activated(hidden_k);
        for (u32_t channel = 0; channel != hidden_k; ++channel)
            normalized[channel] = pooled_host[std::size_t(segment) * hidden_k + channel] * inverse_rms *
                                  weight_at(head_norm_index, channel);
        for (u32_t out = 0; out != hidden_k; ++out) {
            double fc1 = 0.0;
            for (u32_t in = 0; in != hidden_k; ++in)
                fc1 += weight_at(fc1_index, std::size_t(out) * hidden_k + in) * normalized[in];
            activated[out] = fc1 / (1.0 + std::exp(-fc1));
        }
        for (u32_t out = 0; out != hidden_k; ++out) {
            double fc2 = 0.0;
            for (u32_t in = 0; in != hidden_k; ++in)
                fc2 += weight_at(fc2_index, std::size_t(out) * hidden_k + in) * activated[in];
            vertices_host[std::size_t(segment) * hidden_k + out] = fc2;
        }
    }
    double worst = 0.0;
    for (u32_t segment = 0; segment != segments_k; ++segment) {
        if (lengths_host[segment] == 0u) continue; // PAD vertices stay outside the parity
        for (u32_t channel = 0; channel != hidden_k; ++channel) {
            std::size_t const index = std::size_t(segment) * hidden_k + channel;
            worst = std::max(worst, std::abs(vertices_host[index] - double(__bfloat162float(vertices_device[index]))) /
                                        (std::abs(vertices_host[index]) + 1.0));
        }
    }
    std::printf("    forward parity worst relative error %.3e\n", worst);
    check(worst < 3e-2, "pool + head forward matches the FP64 oracle to the BFloat16 floor");

    // Backward smoke over the same chain: a keyed vertices force, then every pool / head gradient finite
    // and the query gradient nonzero (the pool actually differentiated).
    std::vector<bf16_t> force_host(std::size_t(segments_k) * hidden_k);
    for (std::size_t index = 0; index != force_host.size(); ++index)
        force_host[index] = __float2bfloat16(hash_fill(index, 777) * 0.1f);
    cuMemcpyHtoD(reinterpret_cast<CUdeviceptr>(vertices_grad_tensor.data()), force_host.data(),
                 force_host.size() * sizeof(bf16_t));
    auto propagated = tower.value_.backward(vertices_grad_tensor.view(), bf16r2_view_t {}, batch.value_, stream.value_);
    if (!propagated) std::printf("    %s\n", propagated.description_.c_str());
    check(static_cast<bool>(propagated), "pool-parity backward");
    if (propagated) {
        cuStreamSynchronize(stream.value_.get());
        bool all_finite = true;
        double query_grad_energy = 0.0;
        for (std::size_t matrix = query_index; matrix != weights.size(); ++matrix) {
            std::size_t const scalars = weights[matrix].weights_grad.extent(0) * weights[matrix].weights_grad.extent(1);
            staging.resize(scalars);
            cuMemcpyDtoH(staging.data(), reinterpret_cast<CUdeviceptr>(weights[matrix].weights_grad.data()),
                         scalars * sizeof(bf16_t));
            for (std::size_t index = 0; index != scalars; ++index) {
                f32_t const value = __bfloat162float(staging[index]);
                all_finite = all_finite && std::isfinite(value);
                if (matrix == query_index) query_grad_energy += std::abs(double(value));
            }
        }
        check(all_finite, "every pool / head gradient is finite");
        check(query_grad_energy > 0.0, "the pool query gradient is nonzero");
    }
    cuCtxPopCurrent(nullptr);
}

int main() {
    cuInit(0);
    int device_count = 0;
    cuDeviceGetCount(&device_count);
    std::printf("contrastive conformance: %d visible GPU(s)\n", device_count);
    if (device_count == 0) {
        std::printf("no GPUs — cannot run\n");
        return 1;
    }
    if (std::getenv("UFORM_HOST_DOMAIN_INDEX") != nullptr) return run_cross_domain_case();

    run_pool_parity_case();
    run_oracle_case("self-edges", 8, 64, {64}, false, false, 1.0f);
    run_oracle_case("matryoshka", 8, 64, {32, 64}, false, false, 1.0f);
    run_oracle_case("explicit-edges", 8, 64, {64}, true, false, 0.5f);
    run_oracle_case("pad-rows", 8, 64, {64}, false, true, 1.0f);
    run_oracle_case("no-ambient", 8, 64, {64}, false, false, 0.0f);
    run_oracle_case("odd-rows-wide", 33, 256, {32, 256}, false, false, 1.0f);
    run_negative_cases();
    run_distributed_case();

    std::printf("%s: %d failure(s)\n", failures == 0 ? "PASS" : "FAIL", failures);
    return failures == 0 ? 0 : 1;
}
