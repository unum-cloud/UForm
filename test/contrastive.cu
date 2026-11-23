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
#undef NDEBUG // ! A test's oracle must stay live in every build

#include <cmath>   // `std::fabs`, `std::sqrt`, `std::exp`, `std::log1p`
#include <cstdlib> // `std::getenv`, `std::atoi`, `std::atof`
#include <cstring> // `std::memcpy`, `std::memset` — the tokens cases' compacted head rows, the cosines per row

#include <array>   // `std::array` — the pool replica's per-segment scratch
#include <span>    // `std::span` — the oracle's borrowed towers
#include <utility> // `std::move`
#include <vector>  // `std::vector`

#include <cuda.h> // `cuInit`, `cuCtxPushCurrent`, `cuMemcpyHtoD`, `cuMemsetD8` — the raw staging boundary

#include "uform/train/contrastive_segments.cuh"
#include "uform/train/contrastive_tokens.cuh" // `contrastive_tokens_t`, `distributed_contrastive_tokens_t`
#include "uform/train/encoder_tower.hpp"      // `encoder_tower_bf16_t` — the pool + head the FP64 parity case gates
#include "uform/utilities/infiniband_collectives.cuh" // `infiniband_coordinates_t` — the corrupt-payload negative test
#include "uform/train/trainer.hpp"                    // `trainer_t`, `items_source_t` — the sides case's step

#include "train/numerics.hpp" // `gpu_numerics_of`

#include "harness.hpp" // `run_test_case`, `expect_true`

using namespace unum::uform::train;
namespace util = unum::uform::utilities;
using namespace unum::uform::test;
namespace ut = unum::uform::train;

/** Whether the driver shows at least @p wanted devices. */
[[nodiscard]] static bool cuda_devices_visible_(int wanted) noexcept {
    int device_count = 0;
    return cuDeviceGetCount(&device_count) == CUDA_SUCCESS && device_count >= wanted;
}

/** The host-only edge owner the oracle cases relate into. */
using host_edges_t = contrastive_edges<nk::aligned_allocator<util::b8_t>>;

/** Deterministic pseudo-random filler in roughly [-0.5, 0.5). */
static util::f32_t hash_fill(std::size_t index, std::size_t salt) {
    std::size_t const mixed = (index * 2654435761ull + salt * 40503ull + 12345ull) % 1000ull;
    return static_cast<util::f32_t>(mixed) / 1000.0f - 0.5f;
}

/** The host-side pair resolver — mirrors `contrastive_pair_relation` decision for decision. */
struct host_relations_t {
    std::vector<util::u32_t> attract_offsets, attract_ids, repel_offsets, repel_ids, ignore_offsets, ignore_ids;
    std::vector<util::f32_t> attract_scores, repel_scores;
    std::vector<util::u32_t> vertex_effective, neighbor_effective;
    std::vector<util::u64_t> vertex_keys, neighbor_keys;
    std::size_t self_edge_base = 0;
    util::f32_t ambient = 1.0f;

    struct resolved_edge_t {
        util::f64_t target, weight;
    };
    [[nodiscard]] resolved_edge_t resolve(util::u32_t vertex, util::u32_t neighbor_global) const {
        if (!vertex_effective.empty() && vertex_effective[vertex] == 0) return {0.0, 0.0};
        if (!neighbor_effective.empty() && neighbor_effective[neighbor_global] == 0) return {0.0, 0.0};
        auto in_run = [&](std::vector<util::u32_t> const& offsets, std::vector<util::u32_t> const& ids,
                          std::size_t* found_position) -> bool {
            if (offsets.empty()) return false;
            for (util::u32_t position = offsets[vertex]; position != offsets[vertex + 1]; ++position)
                if (ids[position] == neighbor_global) {
                    if (found_position) *found_position = position;
                    return true;
                }
            return false;
        };
        if (in_run(ignore_offsets, ignore_ids, nullptr)) return {0.0, 0.0};
        bool const is_self = neighbor_global == self_edge_base + vertex;
        bool const keyed = !vertex_keys.empty();
        if (attract_ids.empty() && !keyed)
            return is_self ? resolved_edge_t {1.0, 1.0} : resolved_edge_t {-1.0, util::f64_t(ambient)};
        if (is_self) return {0.0, 0.0};
        if (keyed && vertex_keys[vertex] != 0 && vertex_keys[vertex] == neighbor_keys[neighbor_global])
            return {1.0, 1.0};
        std::size_t position = 0;
        if (in_run(attract_offsets, attract_ids, &position)) return {1.0, util::f64_t(attract_scores[position])};
        if (in_run(repel_offsets, repel_ids, &position)) return {-1.0, util::f64_t(repel_scores[position])};
        return {-1.0, util::f64_t(ambient)};
    }

    /** Copy the three sets of @p edges , every ID shifted by @p base . */
    void adopt(host_edges_t const& edges, util::u32_t base = 0) {
        auto copy = [&](contrastive_edges_view_t set, std::vector<util::u32_t>& offsets, std::vector<util::u32_t>& ids,
                        std::vector<util::f32_t>* scores) {
            util::u32_t const shift = static_cast<util::u32_t>(ids.size());
            if (offsets.empty()) offsets.push_back(0);
            for (std::size_t row = 1; row != set.offsets.extent(0); ++row) offsets.push_back(shift + set.offsets[row]);
            for (std::size_t edge = 0; edge != set.ids.extent(0); ++edge) {
                ids.push_back(base + set.ids[edge]);
                if (scores) scores->push_back(set.weights[edge]);
            }
        };
        copy(edges.attract(), attract_offsets, attract_ids, &attract_scores);
        copy(edges.repel(), repel_offsets, repel_ids, &repel_scores);
        copy(edges.ignore(), ignore_offsets, ignore_ids, nullptr);
    }
};

/** One GPU's bags as a loader writes them: the bag, role, length, repel weight and key
 *  of every segment. */
struct host_bags_t {
    std::vector<util::u32_t> bags, lengths;
    std::vector<segment_role_t> roles;
    std::vector<util::f32_t> weights;
    std::vector<util::u64_t> keys;

    /** Append one segment. */
    void add(util::u32_t bag, segment_role_t role, util::u32_t length, util::f32_t weight = 1.0f, util::u64_t key = 0) {
        bags.push_back(bag), roles.push_back(role), lengths.push_back(length), weights.push_back(weight);
        keys.push_back(key);
    }

    /** Relate @c [first,first+count) of these segments as one GPU's rows over all of them
     *  as the world. */
    [[nodiscard]] util::expected_t relate(host_edges_t& edges, util::u32_t first, util::u32_t count) const {
        auto const rows = nk::range(first, first + count);
        util::u32r1_view_t const bag_column = std::span {bags}, length_column = std::span {lengths};
        nk::tensor_view<segment_role_t, 1> const role_column = std::span {roles};
        util::f32r1_view_t const weight_column = std::span {weights};
        return relate_roles(bag_column[rows, nk::slice], role_column[rows, nk::slice], length_column[rows, nk::slice],
                            weight_column[rows, nk::slice], first, edges);
    }
};

/** Ten segments in four bags with one PAD among them: a triplet {0,1,2} whose negative weighs 2, a
 *  multi-positive {3,4,5}, a PAD {6}, and a pair {7,8} with a light negative 9; segments 1, 4 and 8
 *  share one key across three bags, which attract through the resolver and name no edge. */
[[nodiscard]] static host_bags_t ragged_bags_() {
    using role_t = segment_role_t;
    host_bags_t bags;
    bags.add(0, role_t::anchor_k, 4), bags.add(0, role_t::positive_k, 3, 1.0f, 11);
    bags.add(0, role_t::negative_k, 2, 2.0f);
    bags.add(1, role_t::anchor_k, 5), bags.add(1, role_t::positive_k, 1, 1.0f, 11), bags.add(1, role_t::positive_k, 2);
    bags.add(2, role_t::anchor_k, 0);
    bags.add(3, role_t::anchor_k, 6), bags.add(3, role_t::positive_k, 2, 1.0f, 11);
    bags.add(3, role_t::negative_k, 3, 0.5f);
    return bags;
}

/** The Float64 oracle: the exact per-header math of the segments loss over the @b [vertices,hidden]
 *  and @b [neighbors,hidden] towers, held on the BFloat16 grid, in the balanced or the sigmoid
 *  @p form ; the sigmoid one weighs every repel on the attracts' scale. */
static util::f64_t host_loss(f64r2_view_t vertices, f64r2_view_t neighbors, std::span<util::u32_t const> prefixes,
                             host_relations_t const& relations, util::f64_t beta, util::f64_t bias,
                             std::size_t vertices_per_mesh, contrastive_pair_loss_form_t form) {
    bool const sum_negatives = form == contrastive_pair_loss_form_t::sigmoid_k;
    std::size_t const num_vertices = vertices.extent(0), num_neighbors = neighbors.extent(0);
    util::f64_t loss = 0.0;
    for (std::size_t vertex = 0; vertex != num_vertices; ++vertex) {
        // Signed weight totals over the whole neighbor axis.
        util::f64_t pos_weight = 0.0, neg_weight = 0.0;
        for (std::size_t neighbor = 0; neighbor != num_neighbors; ++neighbor) {
            host_relations_t::resolved_edge_t const resolved = relations.resolve(util::u32_t(vertex),
                                                                                 util::u32_t(neighbor));
            if (resolved.target > 0) pos_weight += resolved.weight;
            else if (resolved.target < 0) neg_weight += resolved.weight;
        }
        util::f64_t const pos_scale = 1.0 /
                                      fu::max_of(pos_weight, sum_negatives ? util::f64_t(1.0) : util::f64_t(1.0e-12));
        util::f64_t const neg_scale = sum_negatives ? pos_scale : 1.0 / fu::max_of(neg_weight, util::f64_t(1.0e-12));
        util::f64_t vertex_loss = 0.0;
        for (std::size_t prefix_index = 0; prefix_index != prefixes.size(); ++prefix_index) {
            auto const prefix = nk::range(0, prefixes[prefix_index]);
            f64r1_view_t const ours = vertices[vertex, prefix, nk::slice];
            util::f64_t const vertex_norm = std::sqrt(dot(ours, ours));
            for (std::size_t neighbor = 0; neighbor != num_neighbors; ++neighbor) {
                host_relations_t::resolved_edge_t const resolved = relations.resolve(util::u32_t(vertex),
                                                                                     util::u32_t(neighbor));
                if (resolved.weight == 0.0) continue;
                f64r1_view_t const theirs = neighbors[neighbor, prefix, nk::slice];
                util::f64_t const raw = dot(ours, theirs), neighbor_norm = std::sqrt(dot(theirs, theirs));
                util::f64_t const denominator = vertex_norm * neighbor_norm;
                util::f64_t const cosine = denominator > 1.0e-2 ? raw / denominator : 0.0;
                util::f64_t const logit = beta * cosine + bias;
                util::f64_t const negated = -resolved.target * logit;
                util::f64_t const softplus = fu::max_of(negated, 0.0) + std::log1p(std::exp(-std::fabs(negated)));
                vertex_loss += resolved.weight * softplus * (resolved.target > 0 ? pos_scale : neg_scale);
            }
        }
        loss += vertex_loss;
    }
    return loss / util::f64_t(vertices_per_mesh);
}

/** The Float64 ranked oracle: per anchor row and prefix, every positive's scaled cosine against
 *  every negative's, @c Σ_p Σ_n w_p · w_n · softplus(β · (c_n − c_p))/(Σw_p · Σw_n) , where the
 *  bias cancels; @p targets , @p weights and @p scaled hold one entry per neighbor. */
static util::f64_t host_ranked_loss(f64r2_view_t vertices, f64r2_view_t neighbors,
                                    std::span<util::u32_t const> prefixes, host_relations_t const& relations,
                                    util::f64_t beta, std::size_t vertices_per_mesh, std::span<util::f64_t> targets,
                                    std::span<util::f64_t> weights, std::span<util::f64_t> scaled) {
    std::size_t const num_vertices = vertices.extent(0), num_neighbors = neighbors.extent(0);
    util::f64_t loss = 0.0;
    for (std::size_t vertex = 0; vertex != num_vertices; ++vertex) {
        util::f64_t positive_weight = 0.0, negative_weight = 0.0;
        for (std::size_t neighbor = 0; neighbor != num_neighbors; ++neighbor) {
            host_relations_t::resolved_edge_t const resolved = relations.resolve(util::u32_t(vertex),
                                                                                 util::u32_t(neighbor));
            targets[neighbor] = resolved.target, weights[neighbor] = resolved.weight;
            if (resolved.target > 0) positive_weight += resolved.weight;
            else if (resolved.target < 0) negative_weight += resolved.weight;
        }
        if (positive_weight == 0.0 || negative_weight == 0.0) continue;
        for (std::size_t prefix_index = 0; prefix_index != prefixes.size(); ++prefix_index) {
            auto const prefix = nk::range(0, prefixes[prefix_index]);
            f64r1_view_t const ours = vertices[vertex, prefix, nk::slice];
            util::f64_t const vertex_norm = std::sqrt(dot(ours, ours));
            for (std::size_t neighbor = 0; neighbor != num_neighbors; ++neighbor) {
                f64r1_view_t const theirs = neighbors[neighbor, prefix, nk::slice];
                util::f64_t const denominator = vertex_norm * std::sqrt(dot(theirs, theirs));
                scaled[neighbor] = denominator > 1.0e-2 ? beta * dot(ours, theirs) / denominator : 0.0;
            }
            for (std::size_t positive = 0; positive != num_neighbors; ++positive) {
                if (!(targets[positive] > 0) || weights[positive] == 0.0) continue;
                for (std::size_t negative = 0; negative != num_neighbors; ++negative) {
                    if (!(targets[negative] < 0) || weights[negative] == 0.0) continue;
                    util::f64_t const margin = scaled[negative] - scaled[positive];
                    util::f64_t const softplus = fu::max_of(margin, 0.0) + std::log1p(std::exp(-std::fabs(margin)));
                    loss += weights[positive] * weights[negative] * softplus / (positive_weight * negative_weight);
                }
            }
        }
    }
    return loss / util::f64_t(vertices_per_mesh);
}

/** The Float64 oracle of @p form , the ranked one scratching in @p targets , @p weights and
 *  @p scaled . */
static util::f64_t host_pair_loss(contrastive_pair_loss_form_t form, f64r2_view_t vertices, f64r2_view_t neighbors,
                                  std::span<util::u32_t const> prefixes, host_relations_t const& relations,
                                  util::f64_t beta, util::f64_t bias, std::size_t vertices_per_mesh,
                                  std::span<util::f64_t> targets, std::span<util::f64_t> weights,
                                  std::span<util::f64_t> scaled) {
    if (form == contrastive_pair_loss_form_t::ranked_k)
        return host_ranked_loss(vertices, neighbors, prefixes, relations, beta, vertices_per_mesh, targets, weights,
                                scaled);
    return host_loss(vertices, neighbors, prefixes, relations, beta, bias, vertices_per_mesh, form);
}

/** Which relation graph an oracle case stages. */
enum class oracle_graph_t : util::u32_t {
    self_edges_k = 0,
    explicit_edges_k = 1,
    roles_k = 2,
};

/** One single-GPU oracle case: run the loss, compare to the host oracle, finite-difference the
 *  force on a few sampled channels, and check captured replays land where the eager run did. */
static case_result_t test_oracle(std::size_t num_vertices, std::size_t hidden, std::vector<util::u32_t> const& prefixes,
                                 oracle_graph_t graph, bool has_pad, util::f32_t ambient_weight,
                                 contrastive_pair_loss_form_t form, util::f32_t inverse_temperature, util::f32_t bias) {
    if (!cuda_devices_visible_(1)) return case_result_t::skipped(no_gpu_device_k);
    auto [stream, make_stream_status] = util::gpu_stream_t::make(0);
    if (!expect_true(make_stream_status, "stream: {}", make_stream_status.description_)) return {};
    cuCtxPushCurrent(stream.context());
    auto [arena, make_arena_status] = util::gpu_arena_t::make(0, 1ull << 30);
    if (!expect_true(make_arena_status, "arena: {}", make_arena_status.description_)) return {};

    contrastive_segments_config_t config;
    config.from_vertices_per_gpu = util::u32_t(num_vertices);
    config.hidden = util::u32_t(hidden);
    config.to_vertices_per_gpu = util::u32_t(num_vertices);
    config.gpu_subgraphs_per_step = 1;
    config.pair_loss = form;
    nk_capability_t capabilities = 0;
    if (!expect_true(nk_capabilities_enabled_cuda(0, &capabilities) == nk_success_k, "capability query failed"))
        return {};
    util::gpu_numerics_t const numerics = ut::gpu_numerics_of(capabilities);
    auto [loss, make_loss_status] = contrastive_segments_t::make(arena, config, std::span {prefixes}, numerics);
    if (!expect_true(make_loss_status, "loss make: {}", make_loss_status.description_)) return {};

    // The tower on the BFloat16 grid in Float64 for the oracle, its bit patterns for the device;
    // the squared index keeps rows wider than the filler's period from repeating each other.
    std::size_t const scalars = num_vertices * hidden;
    st::basic_vector<util::f64_t> host_tower, host_grad, targets, weights, scaled;
    std::vector<nk_bf16_t> staging(scalars);
    if (!expect_true(!st::failed(host_tower.resize(scalars)) && !st::failed(host_grad.resize(scalars)) &&
                         !st::failed(targets.resize(num_vertices)) && !st::failed(weights.resize(num_vertices)) &&
                         !st::failed(scaled.resize(num_vertices)),
                     "host staging failed"))
        return {};
    for (std::size_t index = 0; index != scalars; ++index)
        host_tower[index] = hash_fill(index * index, num_vertices + hidden);
    if (!expect_true(narrow_to_bf16(host_tower, staging), "tower rounding failed")) return {};
    util::bf16_t* tower = arena.allocate_object<util::bf16_t>(scalars);
    util::bf16_t* grad = arena.allocate_object<util::bf16_t>(scalars);
    util::f32_t* grad_f32 = arena.allocate_object<util::f32_t>(scalars);
    util::f32_t* loss_out = arena.allocate_object<util::f32_t>(1);
    util::u32_t* effective = arena.allocate_object<util::u32_t>(num_vertices);
    cuMemcpyHtoD(util::device_address(tower), staging.data(), scalars * sizeof(nk_bf16_t));

    // Relations: self-edge default, or explicit attract/repel with an ignore override; PAD via the
    // effective-length columns.
    host_relations_t host_relations;
    host_relations.ambient = ambient_weight;
    auto [relations, make_relations_status] = contrastive_relations_t::make(arena, util::u32_t(num_vertices), 32);
    if (!expect_true(make_relations_status, "relations make: {}", make_relations_status.description_)) return {};
    host_bags_t const bags = ragged_bags_();
    if (!expect_true(graph != oracle_graph_t::roles_k || bags.lengths.size() == num_vertices,
                     "the role graph spans {} vertices, not {}", bags.lengths.size(), num_vertices))
        return {};
    std::vector<util::u32_t> host_effective(num_vertices, 1u);
    if (has_pad) host_effective[num_vertices - 1] = 0u;
    if (graph == oracle_graph_t::roles_k) host_effective = bags.lengths;
    cuMemcpyHtoD(util::device_address(effective), host_effective.data(), num_vertices * sizeof(util::u32_t));
    bool const padded = has_pad || graph == oracle_graph_t::roles_k;
    util::u32r1_view_t const effective_view {effective, padded ? num_vertices : 0};
    if (padded) {
        host_relations.vertex_effective = host_effective;
        host_relations.neighbor_effective = host_effective;
    }
    util::expected_t staged {};
    if (graph == oracle_graph_t::explicit_edges_k) {
        // Vertex 0 attracts {1 (w=2)} and ignores {2}; vertex 1 repels {0 (w=5)}; no vertex names
        // itself, and the rest are ambient.
        host_relations.attract_offsets = {0, 1};
        host_relations.attract_ids = {1};
        host_relations.attract_scores = {2.0f};
        host_relations.repel_offsets = {0, 0, 1};
        host_relations.repel_ids = {0};
        host_relations.repel_scores = {5.0f};
        host_relations.ignore_offsets = {0, 1};
        host_relations.ignore_ids = {2};
        for (std::size_t vertex = 1; vertex != num_vertices; ++vertex) {
            host_relations.attract_offsets.push_back(util::u32_t(host_relations.attract_ids.size()));
            if (vertex >= 2) host_relations.repel_offsets.push_back(util::u32_t(host_relations.repel_ids.size()));
            host_relations.ignore_offsets.push_back(util::u32_t(host_relations.ignore_ids.size()));
        }
        staged = relations.upload(
            {std::span {host_relations.attract_offsets}, std::span {host_relations.attract_ids},
             std::span {host_relations.attract_scores}},
            {std::span {host_relations.repel_offsets}, std::span {host_relations.repel_ids},
             std::span {host_relations.repel_scores}},
            {std::span {host_relations.ignore_offsets}, std::span {host_relations.ignore_ids}, {}}, effective_view,
            effective_view, {}, {}, num_vertices, 0, ambient_weight, stream);
    }
    else if (graph == oracle_graph_t::roles_k) {
        auto [edges, status] = host_edges_t::make(util::u32_t(num_vertices), 32);
        if (!expect_true(status, "edges make: {}", status.description_)) return {};
        staged = bags.relate(edges, 0, util::u32_t(num_vertices));
        if (!expect_true(staged, "relate: {}", staged.description_)) return {};
        host_relations.adopt(edges);
        host_relations.vertex_keys = bags.keys;
        host_relations.neighbor_keys = bags.keys;
        util::u64r1_span_t const keys {arena.allocate_object<util::u64_t>(num_vertices), num_vertices};
        cuMemcpyHtoD(util::device_address(keys.data()), bags.keys.data(), num_vertices * sizeof(util::u64_t));
        staged = relations.upload(edges.attract(), edges.repel(), edges.ignore(), effective_view, effective_view, keys,
                                  keys, num_vertices, 0, ambient_weight, stream);
    }
    else
        staged = relations.upload_self_edges(util::u32_t(num_vertices), effective_view, effective_view, 0,
                                             ambient_weight, stream);
    if (!expect_true(staged, "relations upload: {}", staged.description_)) return {};

    gpu_subgraph_t subgraph;
    subgraph.to_vertices = util::bf16r2_view_t {tower, num_vertices, hidden};
    subgraph.global_offset = 0;
    subgraph.to_vertices_grad = util::f32r2_span_t {grad_f32, num_vertices, hidden};
    contrastive_recipe_t recipe;
    recipe.inverse_temperature = inverse_temperature;
    recipe.bias = bias;
    recipe.ambient_repulsion_weight = ambient_weight;

    auto run_once = [&]() noexcept -> expected_t {
        return loss(util::bf16r2_view_t {tower, num_vertices, hidden}, {&subgraph, 1}, relations.view(), recipe,
                    util::f32r1_span_t {loss_out, 1}, util::bf16r2_span_t {grad, num_vertices, hidden}, stream);
    };
    if (util::expected_t ran = run_once(); !expect_true(ran, "loss enqueue: {}", ran.description_)) return {};
    cuStreamSynchronize(stream.get());
    util::f32_t device_loss = 0;
    cuMemcpyDtoH(&device_loss, util::device_address(loss_out), sizeof(util::f32_t));

    std::size_t live_vertices = 0; // the mean skips PAD
    for (util::u32_t const length : host_effective) live_vertices += !padded || length != 0u;
    auto const oracle_of = [&](st::basic_vector<util::f64_t> const& towers) noexcept {
        f64r2_view_t const rows {towers.data(), num_vertices, hidden};
        return host_pair_loss(form, rows, rows, prefixes, host_relations, inverse_temperature, bias, live_vertices,
                              targets, weights, scaled);
    };
    util::f64_t const oracle = oracle_of(host_tower);
    util::f64_t const relative = std::fabs(util::f64_t(device_loss) - oracle) / fu::max_of(std::fabs(oracle), 1.0e-9);
    expect_true(relative < 5.0e-3, "loss {:.6f} matches the Float64 oracle {:.6f}: relative {:.2e}", device_loss,
                oracle, relative);
    // Each cell's softplus stays under 2 at unit β and zero bias, so repel-only rows stay bounded.
    expect_true(device_loss < 2.0f * 1.5f * util::f32_t(num_vertices), "loss {:.6f} stays bounded by its cells",
                device_loss);

    // Finite-difference force check on a few sampled channels, as one vector against its own norm,
    // so a small true force cannot hide under an absolute floor: (L(x+h) - L(x-h)) / 2h within 1%.
    cuMemcpyDtoH(staging.data(), util::device_address(grad), scalars * sizeof(nk_bf16_t));
    if (!expect_true(widen_from_bf16(staging, host_grad), "gradient widening failed")) return {};
    std::vector<util::f32_t> host_column(scalars);
    cuMemcpyDtoH(host_column.data(), util::device_address(grad_f32), scalars * sizeof(util::f32_t));
    util::f64_t error_squares = 0.0, numeric_squares = 0.0;
    for (std::size_t sample = 0; sample != 6; ++sample) {
        std::size_t const index = (sample * 977ull + 13ull) % scalars;
        util::f64_t const step_size = 1.0e-2, original = host_tower[index];
        host_tower[index] = original + step_size;
        if (!expect_true(narrow_to_bf16(host_tower, staging), "perturbation rounding failed")) return {};
        util::f64_t const plus = host_tower[index], loss_plus = oracle_of(host_tower);
        host_tower[index] = original - step_size;
        if (!expect_true(narrow_to_bf16(host_tower, staging), "perturbation rounding failed")) return {};
        util::f64_t const minus = host_tower[index], loss_minus = oracle_of(host_tower);
        host_tower[index] = original;
        util::f64_t const numeric = (loss_plus - loss_minus) / (plus - minus);
        // The full derivative sums the row-side (bf16 output) and column-side
        // (f32 subgraph) forces.
        util::f64_t const analytic = host_grad[index] + host_column[index];
        error_squares += (numeric - analytic) * (numeric - analytic), numeric_squares += numeric * numeric;
    }
    util::f64_t const force_error = std::sqrt(error_squares), force_norm = std::sqrt(numeric_squares);
    expect_true(force_error <= 1.0e-2 * force_norm + 1.0e-12,
                "the sampled force matches central finite differences: error {:.3e} over norm {:.3e}", force_error,
                force_norm);

    // The retrieval table against the Float64 rows, over the oracle's tower and then over one whose
    // vertices all lie within a few degrees of one direction, where uncentred cosine sums cancel.
    auto const check_retrieval = [&](st::basic_vector<util::f64_t> const& values, char const* tower_name) noexcept {
        if (util::expected_t evaluated = loss.evaluate(util::bf16r2_view_t {tower, num_vertices, hidden},
                                                       {&subgraph, 1}, relations.view(), recipe, stream);
            !expect_true(evaluated, "evaluate over the {} tower: {}", tower_name, evaluated.description_))
            return;
        cuStreamSynchronize(stream.get());
        std::size_t const columns = std::size_t(retrieval_column_t::count_k);
        std::vector<util::f32_t> table(num_vertices * columns);
        cuMemcpyDtoH(table.data(), util::device_address(loss.retrieval().data()), table.size() * sizeof(util::f32_t));
        util::f32r2_view_t const table_rows {table.data(), num_vertices, columns};
        f64r2_view_t const rows {values.data(), num_vertices, hidden};
        st::basic_vector<util::f64_t> units, centre, cosines;
        if (!expect_true(!st::failed(units.resize(num_vertices * hidden)) && !st::failed(centre.resize(hidden)) &&
                             !st::failed(cosines.resize(num_vertices)),
                         "retrieval staging over the {} tower failed", tower_name))
            return;
        f64r2_span_t const unit_rows {units.data(), num_vertices, hidden};
        util::f64_t live = 0.0;
        for (std::size_t row = 0; row != num_vertices; ++row) {
            util::f64_t const norm = std::sqrt(dot(rows[row, nk::slice], rows[row, nk::slice]));
            if (host_effective[row] == 0u || norm == 0.0) continue;
            live += 1.0;
            for (std::size_t channel = 0; channel != hidden; ++channel)
                unit_rows[row, channel] = rows[row, channel] / norm, centre[channel] += unit_rows[row, channel];
        }
        for (util::f64_t& channel : centre) channel /= live;
        for (std::size_t row = 0; row != num_vertices; ++row)
            for (std::size_t channel = 0; channel != hidden && host_effective[row] != 0u; ++channel)
                unit_rows[row, channel] -= centre[channel];
        util::f64_t worst = 0.0, trace = 0.0, frobenius = 0.0, device_trace = 0.0, device_frobenius = 0.0;
        for (std::size_t row = 0; row != num_vertices; ++row) {
            util::f64_t expected[std::size_t(retrieval_column_t::count_k)] = {};
            auto const cell = [&](retrieval_column_t which) noexcept -> f64_t& { return expected[std::size_t(which)]; };
            std::memset(cosines.data(), 0, num_vertices * sizeof(util::f64_t));
            util::f64_t best = -1e300, positives = 0.0, negatives = 0.0, above = 0.0;
            if (host_effective[row] != 0u)
                cell(retrieval_column_t::centred_norm_k) = dot(unit_rows[row, nk::slice], unit_rows[row, nk::slice]);
            for (std::size_t other = 0; other != num_vertices && host_effective[row] != 0u; ++other) {
                if (host_effective[other] == 0u) continue;
                f64r1_view_t const ours = rows[row, nk::slice], theirs = rows[other, nk::slice];
                util::f64_t const denominator = std::sqrt(dot(ours, ours) * dot(theirs, theirs));
                cosines[other] = denominator > 1.0e-2 ? dot(ours, theirs) / denominator : 0.0;
                util::f64_t const centred = dot(unit_rows[row, nk::slice], unit_rows[other, nk::slice]);
                cell(retrieval_column_t::centred_products_k) += centred * centred;
                host_relations_t::resolved_edge_t const resolved = host_relations.resolve(util::u32_t(row),
                                                                                          util::u32_t(other));
                if (resolved.target > 0 && resolved.weight != 0.0)
                    best = fu::max_of(best, cosines[other]), positives += 1.0,
                    cell(retrieval_column_t::attract_cosine_k) += cosines[other];
            }
            for (std::size_t other = 0; other != num_vertices && host_effective[row] != 0u; ++other) {
                host_relations_t::resolved_edge_t const resolved = host_relations.resolve(util::u32_t(row),
                                                                                          util::u32_t(other));
                if (!(resolved.target < 0) || resolved.weight == 0.0) continue;
                bool const outranks = cosines[other] > best;
                negatives += 1.0, above += outranks;
                bool hard = false;
                for (util::u32_t edge = host_relations.repel_offsets[row];
                     edge != host_relations.repel_offsets[row + 1]; ++edge)
                    hard = hard || host_relations.repel_ids[edge] == other;
                if (hard) cell(retrieval_column_t::hards_k) += 1.0, cell(retrieval_column_t::hard_above_k) += outranks;
                else
                    cell(retrieval_column_t::ambients_k) += 1.0,
                        cell(retrieval_column_t::ambient_cosine_k) += cosines[other];
            }
            bool const ranked = positives != 0.0 && negatives != 0.0;
            cell(retrieval_column_t::ranked_k) = ranked;
            cell(retrieval_column_t::hit_at_1_k) = ranked && above == 0.0;
            cell(retrieval_column_t::hit_at_5_k) = ranked && above < 5.0;
            cell(retrieval_column_t::chance_k) = ranked ? positives / (positives + negatives) : 0.0;
            cell(retrieval_column_t::attracts_k) = positives;
            for (std::size_t column = 0; column != columns; ++column)
                worst = fu::max_of(worst, std::fabs(util::f64_t(table_rows[row, column]) - expected[column]));
            trace += cell(retrieval_column_t::centred_norm_k);
            frobenius += cell(retrieval_column_t::centred_products_k);
            device_trace += table_rows[row, std::size_t(retrieval_column_t::centred_norm_k)];
            device_frobenius += table_rows[row, std::size_t(retrieval_column_t::centred_products_k)];
        }
        util::f64_t const rank = trace * trace / frobenius,
                          device_rank = device_trace * device_trace / device_frobenius;
        expect_true(worst < 1.0e-2, "the retrieval table over the {} tower matches the Float64 rows: worst {:.2e}",
                    tower_name, worst);
        expect_true(std::fabs(device_rank - rank) <= 1.0e-2 * rank,
                    "the effective rank over the {} tower matches Float64: {:.4f} against {:.4f}", tower_name,
                    device_rank, rank);
    };
    if (graph == oracle_graph_t::roles_k) {
        check_retrieval(host_tower, "oracle");
        st::basic_vector<util::f64_t> parallel;
        if (!expect_true(!st::failed(parallel.resize(scalars)), "parallel tower staging failed")) return {};
        for (std::size_t index = 0; index != scalars; ++index)
            parallel[index] = 1.0 + 0.05 * hash_fill(index, num_vertices + hidden + 1);
        if (!expect_true(narrow_to_bf16(parallel, staging) && widen_from_bf16(staging, parallel),
                         "parallel tower rounding failed"))
            return {};
        cuMemcpyHtoD(util::device_address(tower), staging.data(), scalars * sizeof(nk_bf16_t));
        check_retrieval(parallel, "nearly parallel");
        if (!expect_true(narrow_to_bf16(host_tower, staging), "tower rounding failed")) return {};
        cuMemcpyHtoD(util::device_address(tower), staging.data(), scalars * sizeof(nk_bf16_t));
    }

    // Capture once, replay twice: the replays land where the eager run did, up to reduction order.
    st::basic_vector<util::f64_t> eager_grad;
    if (!expect_true(!st::failed(eager_grad.assign(host_grad)), "eager gradient staging failed")) return {};
    auto [replay_graph, make_replay_graph_status] = util::gpu_replay_graph_t::make(
        stream.get(), [&]() noexcept { return run_once(); });
    if (!expect_true(make_replay_graph_status, "capture: {}", make_replay_graph_status.description_)) return {};
    for (int replay = 0; replay != 2; ++replay)
        if (util::expected_t launched = replay_graph.launch(stream.get());
            !expect_true(launched, "replay launch: {}", launched.description_))
            return {};
    cuStreamSynchronize(stream.get());
    cuMemcpyDtoH(staging.data(), util::device_address(grad), scalars * sizeof(nk_bf16_t));
    if (!expect_true(widen_from_bf16(staging, host_grad), "gradient widening failed")) return {};
    util::f64_t largest = 0.0, worst = 0.0;
    for (std::size_t index = 0; index != scalars; ++index) {
        largest = fu::max_of(largest, std::fabs(eager_grad[index]));
        worst = fu::max_of(worst, std::fabs(eager_grad[index] - host_grad[index]));
    }
    expect_true(worst <= 1.0e-2 * largest, "captured replays match eager: worst {} over largest {}", worst, largest);
    cuCtxPopCurrent(nullptr);
    return {};
}

/** Negative tests: every invalid staging must fail loudly, never fall back. */
static case_result_t test_negatives() {
    if (!cuda_devices_visible_(1)) return case_result_t::skipped(no_gpu_device_k);
    auto [stream, make_stream_status] = util::gpu_stream_t::make(0);
    if (!expect_true(make_stream_status, "stream: {}", make_stream_status.description_)) return {};
    cuCtxPushCurrent(stream.context());
    auto [arena, make_arena_status] = util::gpu_arena_t::make(0, 64ull << 20);
    if (!expect_true(make_arena_status, "arena: {}", make_arena_status.description_)) return {};
    auto [relations, make_relations_status] = contrastive_relations_t::make(arena, 4, 4);
    if (!expect_true(make_relations_status, "relations: {}", make_relations_status.description_)) return {};
    util::u32_t offsets[5] = {0, 2, 2, 2, 2};
    util::u32_t unsorted_ids[2] = {3, 1};
    util::f32_t scores[2] = {1.0f, 1.0f};
    util::u32_t empty_offsets[5] = {0, 0, 0, 0, 0};
    util::expected_t unsorted = relations.upload({{offsets, 5}, {unsorted_ids, 2}, {scores, 2}},
                                                 {{empty_offsets, 5}, {}, {}}, {}, {}, {}, {}, {}, 8, 0, 1.0f, stream);
    expect_true(!unsorted, "unsorted run is rejected loudly");
    util::u32_t overflowing_ids[6] = {0, 1, 2, 3, 4, 5};
    util::u32_t overflow_offsets[5] = {0, 6, 6, 6, 6};
    util::f32_t overflow_scores[6] = {1, 1, 1, 1, 1, 1};
    util::expected_t overflow = relations.upload({{overflow_offsets, 5}, {overflowing_ids, 6}, {overflow_scores, 6}},
                                                 {{empty_offsets, 5}, {}, {}}, {}, {}, {}, {}, {}, 8, 0, 1.0f, stream);
    expect_true(!overflow, "edge-capacity overflow is rejected loudly");
    util::infiniband_coordinates_t coordinates;
    expect_true(!util::infiniband_coordinates_t::from_text("lid banana\n", coordinates),
                "corrupt coordinates are rejected");
    cuCtxPopCurrent(nullptr);
    return {};
}

/** One loss call's three outputs read back: the loss summand and the first group's log-β
 *  and bias gradients. */
struct loss_readout_t {
    util::f64_t loss = 0.0, log_inverse_temperature_grad = 0.0, bias_grad = 0.0;
};

/** The sharded oracle: one global batch split across a simulated two-host, eight-GPU world on one
 *  device. Every simulated rank scores its rows against all sixteen subgraphs, and the per-rank
 *  loss and temperature summands must add up to one global call and to the Float64 oracle, PAD rows
 *  excluded from the mean. Under @c roles_k each rank holds a triplet, a multi-positive bag and a
 *  pair, and the triplets' anchors share keys with those four ranks away. */
static case_result_t test_sharded_oracle(oracle_graph_t graph, contrastive_pair_loss_form_t form) {
    if (!cuda_devices_visible_(1)) return case_result_t::skipped(no_gpu_device_k);
    constexpr util::u32_t ranks_k = 16, vertices_k = 8, hidden_k = 64, world_k = ranks_k * vertices_k;
    constexpr util::f32_t beta_k = 2.5f, bias_k = -1.5f;
    util::u32_t const prefixes[1] = {hidden_k};
    auto [stream, make_stream_status] = util::gpu_stream_t::make(0);
    if (!expect_true(make_stream_status, "stream: {}", make_stream_status.description_)) return {};
    cuCtxPushCurrent(stream.context());
    auto [arena, make_arena_status] = util::gpu_arena_t::make(0, 1ull << 30);
    if (!expect_true(make_arena_status, "arena: {}", make_arena_status.description_)) return {};

    // Every third rank ends in a PAD row, which neither scores nor counts toward the mean.
    st::basic_vector<util::f64_t> host_tower, targets, weights, scaled;
    std::vector<nk_bf16_t> staging(world_k * hidden_k);
    if (!expect_true(!st::failed(host_tower.resize(world_k * hidden_k)) && !st::failed(targets.resize(world_k)) &&
                         !st::failed(weights.resize(world_k)) && !st::failed(scaled.resize(world_k)),
                     "host staging failed"))
        return {};
    util::u32_t host_lengths[world_k];
    for (std::size_t index = 0; index != host_tower.size(); ++index) host_tower[index] = hash_fill(index, 31);
    if (!expect_true(narrow_to_bf16(host_tower, staging), "tower rounding failed")) return {};
    util::u32_t live_vertices = 0;
    host_bags_t world_bags;
    for (util::u32_t vertex = 0; vertex != world_k; ++vertex) {
        host_lengths[vertex] = (vertex / vertices_k) % 3u == 0u && vertex % vertices_k == vertices_k - 1u ? 0u : 4u;
        live_vertices += host_lengths[vertex] != 0u;
        using role_t = segment_role_t;
        util::u32_t const rank = vertex / vertices_k, local = vertex % vertices_k;
        util::u32_t const local_bags[vertices_k] = {0, 0, 0, 1, 1, 1, 2, 2};
        role_t const local_roles[vertices_k] = {role_t::anchor_k, role_t::positive_k, role_t::negative_k,
                                                role_t::anchor_k, role_t::positive_k, role_t::positive_k,
                                                role_t::anchor_k, role_t::positive_k};
        world_bags.add(rank * 3u + local_bags[local], local_roles[local], host_lengths[vertex],
                       local == 2u ? 1.5f : 1.0f, local == 0u ? rank % 4u + 1u : 0u);
    }
    util::bf16r2_span_t const tower {arena.allocate_object<util::bf16_t>(world_k * hidden_k), world_k, hidden_k};
    util::bf16r2_span_t const row_grad {arena.allocate_object<util::bf16_t>(world_k * hidden_k), world_k, hidden_k};
    util::f32r2_span_t const column_grad {arena.allocate_object<util::f32_t>(world_k * hidden_k), world_k, hidden_k};
    util::u32r1_span_t const world_lengths {arena.allocate_object<util::u32_t>(world_k), world_k};
    util::u64r1_span_t const world_keys {arena.allocate_object<util::u64_t>(world_k), world_k};
    util::f32r1_span_t const temperatures {arena.allocate_object<util::f32_t>(2), 2};
    util::f32r2_span_t const outputs {arena.allocate_object<util::f32_t>(3 * (ranks_k + 1)), ranks_k + 1, 3};
    util::f32_t const host_temperatures[2] = {beta_k, bias_k};
    cuMemcpyHtoD(util::device_address(tower.data()), staging.data(), staging.size() * sizeof(nk_bf16_t));
    cuMemcpyHtoD(util::device_address(world_lengths.data()), host_lengths, sizeof(host_lengths));
    cuMemcpyHtoD(util::device_address(world_keys.data()), world_bags.keys.data(), world_k * sizeof(util::u64_t));
    cuMemcpyHtoD(util::device_address(temperatures.data()), host_temperatures, sizeof(host_temperatures));
    contrastive_recipe_t recipe;
    recipe.inverse_temperature_device = temperatures[nk::range(0, 1), nk::slice];
    recipe.bias_device = temperatures[nk::range(1, 2), nk::slice];

    // One loss call over the rows from `first_vertex`, its three outputs in row `call`.
    auto run_call = [&](contrastive_segments_t& loss, contrastive_relations_t& relations, host_edges_t& edges,
                        util::u32_t first_vertex, util::u32_t rows, nk::tensor_view<gpu_subgraph_t, 1> subgraphs,
                        util::u32_t call) noexcept -> expected_t {
        auto const owned = nk::range(first_vertex, first_vertex + rows);
        util::expected_t staged {};
        if (graph == oracle_graph_t::roles_k) {
            staged = world_bags.relate(edges, first_vertex, rows);
            if (staged)
                staged = relations.upload(edges.attract(), edges.repel(), edges.ignore(),
                                          world_lengths[owned, nk::slice], world_lengths, world_keys[owned, nk::slice],
                                          world_keys, world_k, first_vertex, 1.0f, stream);
            // The host edges are refilled by the next call, so their copies land first.
            if (staged && cuStreamSynchronize(stream.get()) != CUDA_SUCCESS)
                staged = util::expected_t::fail(util::status_code_t::device_k, "relation staging drain failed");
        }
        else
            staged = relations.upload_self_edges(rows, world_lengths[owned, nk::slice], world_lengths, first_vertex,
                                                 1.0f, stream);
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
    sharded_config.pair_loss = form;
    contrastive_segments_config_t global_config = sharded_config;
    global_config.from_vertices_per_gpu = world_k;
    global_config.to_vertices_per_gpu = world_k;
    global_config.gpu_subgraphs_per_step = 1;
    nk_capability_t capabilities = 0;
    if (!expect_true(nk_capabilities_enabled_cuda(0, &capabilities) == nk_success_k, "capability query failed"))
        return {};
    util::gpu_numerics_t const numerics = ut::gpu_numerics_of(capabilities);
    auto [sharded_loss, make_sharded_loss_status] = contrastive_segments_t::make(arena, sharded_config, {prefixes, 1},
                                                                                 numerics);
    auto [global_loss, make_global_loss_status] = contrastive_segments_t::make(arena, global_config, {prefixes, 1},
                                                                               numerics);
    constexpr util::u32_t sharded_edges_k = 64, global_edges_k = ranks_k * sharded_edges_k;
    auto [sharded_relations, make_sharded_relations_status] = contrastive_relations_t::make(arena, vertices_k,
                                                                                            sharded_edges_k);
    auto [global_relations, make_global_relations_status] = contrastive_relations_t::make(arena, world_k,
                                                                                          global_edges_k);
    auto [sharded_edges, make_sharded_edges_status] = host_edges_t::make(vertices_k, sharded_edges_k);
    auto [global_edges, make_global_edges_status] = host_edges_t::make(world_k, global_edges_k);
    if (!expect_true(make_sharded_loss_status && make_global_loss_status && make_sharded_relations_status &&
                         make_global_relations_status && make_sharded_edges_status && make_global_edges_status,
                     "sharded bring-up: {}{}", make_sharded_loss_status.description_,
                     make_global_loss_status.description_))
        return {};

    util::bf16r3_span_t const tower_planes = tower.reshape<3>({ranks_k, vertices_k, hidden_k});
    util::f32r3_span_t const column_grad_planes = column_grad.reshape<3>({ranks_k, vertices_k, hidden_k});
    gpu_subgraph_t subgraphs[ranks_k];
    for (util::u32_t rank = 0; rank != ranks_k; ++rank) {
        subgraphs[rank].to_vertices = tower_planes[rank, nk::slice];
        subgraphs[rank].global_offset = std::size_t {rank} * vertices_k;
        subgraphs[rank].to_vertices_grad = column_grad_planes[rank, nk::slice];
    }
    gpu_subgraph_t const whole[1] = {{tower, 0, column_grad}};
    util::expected_t ran {};
    for (util::u32_t rank = 0; rank != ranks_k && ran; ++rank)
        ran = run_call(sharded_loss, sharded_relations, sharded_edges, rank * vertices_k, vertices_k,
                       {subgraphs, ranks_k}, rank);
    if (ran) ran = run_call(global_loss, global_relations, global_edges, 0, world_k, {whole, 1}, ranks_k);
    if (!expect_true(ran, "sharded enqueue: {}", ran.description_)) return {};
    cuStreamSynchronize(stream.get());
    util::f32_t host_outputs[ranks_k + 1][3];
    cuMemcpyDtoH(host_outputs, util::device_address(outputs.data()), sizeof(host_outputs));
    loss_readout_t sharded, global;
    for (util::u32_t rank = 0; rank != ranks_k; ++rank) {
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
    if (graph == oracle_graph_t::roles_k) {
        host_relations.adopt(global_edges);
        host_relations.vertex_keys = world_bags.keys;
        host_relations.neighbor_keys = world_bags.keys;
    }
    f64r2_view_t const host_rows {host_tower.data(), world_k, hidden_k};
    auto oracle_at = [&](util::f64_t beta, util::f64_t bias) noexcept {
        return host_pair_loss(form, host_rows, host_rows, prefixes, host_relations, beta, bias, live_vertices, targets,
                              weights, scaled);
    };
    util::f64_t const step_size = 1.0e-4;
    loss_readout_t oracle;
    oracle.loss = oracle_at(beta_k, bias_k);
    oracle.log_inverse_temperature_grad = (oracle_at(beta_k * std::exp(step_size), bias_k) -
                                           oracle_at(beta_k * std::exp(-step_size), bias_k)) /
                                          (2.0 * step_size);
    oracle.bias_grad = (oracle_at(beta_k, bias_k + step_size) - oracle_at(beta_k, bias_k - step_size)) /
                       (2.0 * step_size);

    auto relative = [](util::f64_t got, util::f64_t want) noexcept {
        return std::fabs(got - want) / fu::max_of(std::fabs(want), 1.0e-9);
    };
    // The split moves the Float32 sums no further than rounding, and both land near Float64.
    util::f64_t const tolerance = 1.0e-6, float64_tolerance = 1.0e-5;
    expect_true(relative(sharded.loss, global.loss) < tolerance, "sharded loss {:.9f} vs global {:.9f}", sharded.loss,
                global.loss);
    expect_true(relative(sharded.log_inverse_temperature_grad, global.log_inverse_temperature_grad) < tolerance,
                "sharded log-β gradient {:.9f} vs global {:.9f}", sharded.log_inverse_temperature_grad,
                global.log_inverse_temperature_grad);
    expect_true(relative(sharded.bias_grad, global.bias_grad) < tolerance,
                "sharded bias gradient {:.9f} vs global {:.9f}", sharded.bias_grad, global.bias_grad);
    expect_true(relative(sharded.loss, oracle.loss) < float64_tolerance,
                "sharded loss {:.9f} vs Float64 {:.9f}: {:.2e}", sharded.loss, oracle.loss,
                relative(sharded.loss, oracle.loss));
    expect_true(relative(sharded.log_inverse_temperature_grad, oracle.log_inverse_temperature_grad) < float64_tolerance,
                "sharded log-β gradient {:.9f} vs Float64 {:.9f}: {:.2e}", sharded.log_inverse_temperature_grad,
                oracle.log_inverse_temperature_grad,
                relative(sharded.log_inverse_temperature_grad, oracle.log_inverse_temperature_grad));
    expect_true(relative(sharded.bias_grad, oracle.bias_grad) < float64_tolerance,
                "sharded bias gradient {:.9f} vs Float64 {:.9f}: {:.2e}", sharded.bias_grad, oracle.bias_grad,
                relative(sharded.bias_grad, oracle.bias_grad));
    cuCtxPopCurrent(nullptr);
    return {};
}

/** The distributed tier: every visible GPU vs the single-GPU global reference on identical data. */
static case_result_t test_distributed() {
    int device_count = 0;
    if (cuDeviceGetCount(&device_count) != CUDA_SUCCESS || device_count < 2)
        return case_result_t::skipped("needs two CUDA devices");
    // The topology spans the whole box, so the domain takes every visible GPU and shares no handle.
    std::size_t const gpus = fu::min_of<std::size_t>(static_cast<std::size_t>(device_count), util::max_gpus_per_host_k);
    constexpr util::u32_t vertices_k = 8, hidden_k = 64;
    constexpr std::size_t scalars_k = std::size_t(vertices_k) * hidden_k;
    util::i32_t devices[util::max_gpus_per_host_k] = {};
    for (std::size_t rank = 0; rank != gpus; ++rank) devices[rank] = static_cast<util::i32_t>(rank);
    util::u32_t prefix_dims[1] = {hidden_k};

    // Global data on the BFloat16 grid: 8 vertices per rank; rank r owns rows [8r, 8r+8).
    st::basic_vector<util::f64_t> global_tower, distributed_grad;
    if (!expect_true(!st::failed(global_tower.resize(gpus * scalars_k)) &&
                         !st::failed(distributed_grad.resize(gpus * scalars_k)),
                     "host staging failed"))
        return {};
    std::vector<nk_bf16_t> staging(global_tower.size());
    for (std::size_t index = 0; index != global_tower.size(); ++index) global_tower[index] = hash_fill(index, 7);
    if (!expect_true(narrow_to_bf16(global_tower, staging), "tower rounding failed")) return {};
    nk::tensor_span<nk_bf16_t, 2> const staging_rows {staging.data(), gpus, scalars_k};

    util::per_gpu_array<util::gpu_stream_t> streams;
    util::per_gpu_array<util::gpu_arena_t> arenas;
    contrastive_relations_t owners[util::max_gpus_per_host_k];
    contrastive_relations_view_t relations[util::max_gpus_per_host_k];
    util::bf16r2_view_t towers[util::max_gpus_per_host_k];
    util::bf16r2_span_t grads[util::max_gpus_per_host_k];
    contrastive_segments_config_t config;
    config.from_vertices_per_gpu = vertices_k;
    config.hidden = hidden_k;
    config.to_vertices_per_gpu = vertices_k;
    config.gpu_subgraphs_per_step = gpus;
    // The composition-root order: topology → arenas (visibility declared) → streams → the ONE
    // multimem domain → the ONE colocated domain → the consumer, borrowing all of it.
    for (std::size_t rank = 0; rank != gpus; ++rank) {
        auto [stream, make_stream_status] = util::gpu_stream_t::make(devices[rank]);
        if (!expect_true(make_stream_status && streams.push_back(std::move(stream)) == util::status_code_t::success_k,
                         "distributed stream make"))
            return {};
        cuCtxPushCurrent(streams[rank].context());
        auto [arena, make_arena_status] = util::gpu_arena_t::make(devices[rank], 1ull << 30, {devices, gpus});
        if (!expect_true(make_arena_status && arenas.push_back(std::move(arena)) == util::status_code_t::success_k,
                         "distributed arena make"))
            return {};
        util::bf16_t* tower = arenas[rank].allocate_object<util::bf16_t>(scalars_k);
        util::bf16_t* grad = arenas[rank].allocate_object<util::bf16_t>(scalars_k);
        cuMemcpyHtoD(util::device_address(tower), staging_rows[rank, nk::slice].data(), scalars_k * sizeof(nk_bf16_t));
        towers[rank] = util::bf16r2_view_t {tower, vertices_k, hidden_k};
        grads[rank] = util::bf16r2_span_t {grad, vertices_k, hidden_k};
        auto [owner, status] = contrastive_relations_t::make(arenas[rank], vertices_k, 0);
        if (!expect_true(status, "distributed relations make: {}", status.description_)) return {};
        owners[rank] = std::move(owner);
        util::expected_t staged = owners[rank].upload_self_edges(vertices_k, {}, {}, rank * std::size_t(vertices_k),
                                                                 1.0f, streams[rank]);
        if (!expect_true(staged, "distributed relations: {}", staged.description_)) return {};
        relations[rank] = owners[rank].view();
        cuCtxPopCurrent(nullptr);
    }
    // A one-process world: the whole box is one host_domain and one multimem_domain (D=1), so the
    // colocated domain is the trivial identity and no exchange happens.
    auto [topology, bootstrap_status] = util::mesh_topology_t::bootstrap(
        0, 1, &pause_between_polls, 0, "/tmp/uform_exchange_local", "distributed_case");
    if (!expect_true(bootstrap_status, "single-domain bootstrap: {}", bootstrap_status.description_)) return {};
    auto [domain, make_domain_status] = util::multimem_domain_t::make(
        topology, {devices, gpus}, distributed_contrastive_segments_t::payload_scalars_needed(config, topology.shape),
        streams, wait_limit_k.count());
    if (!expect_true(make_domain_status, "multimem domain make: {}", make_domain_status.description_)) return {};
    auto [colocated, make_colocated_status] = util::colocated_domain_t::make(topology);
    if (!expect_true(make_colocated_status, "colocated domain make: {}", make_colocated_status.description_)) return {};
    nk_capability_t capabilities = 0;
    if (!expect_true(nk_capabilities_enabled_cuda(0, &capabilities) == nk_success_k, "capability query failed"))
        return {};
    util::gpu_numerics_t const numerics = ut::gpu_numerics_of(capabilities);
    auto [distributed, make_distributed_status] = distributed_contrastive_segments_t::make(
        domain, colocated, arenas, streams, {towers, gpus}, {relations, gpus}, {grads, gpus}, config, {prefix_dims, 1},
        numerics);
    if (!expect_true(make_distributed_status, "distributed make: {}", make_distributed_status.description_)) return {};
    util::expected_t ran = distributed(contrastive_recipe_t {}, streams);
    if (!expect_true(ran, "distributed run: {}", ran.description_)) return {};
    for (std::size_t rank = 0; rank != gpus; ++rank) {
        cuCtxPushCurrent(streams[rank].context());
        cuStreamSynchronize(streams[rank].get());
        cuCtxPopCurrent(nullptr);
    }
    util::f64_t distributed_loss = 0.0;
    cuCtxPushCurrent(streams[0].context());
    util::f32_t summands[util::max_gpus_per_host_k] = {};
    cuMemcpyDtoH(summands, util::device_address(distributed.loss_outs().data()), gpus * sizeof(util::f32_t));
    for (std::size_t rank = 0; rank != gpus; ++rank) {
        distributed_loss += summands[rank];
        cuMemcpyDtoH(staging_rows[rank, nk::slice].data(), util::device_address(grads[rank].data()),
                     scalars_k * sizeof(nk_bf16_t));
    }
    cuCtxPopCurrent(nullptr);
    if (!expect_true(widen_from_bf16(staging, distributed_grad), "gradient widening failed")) return {};

    // The Float64 oracle over the SAME global grid — the sum of rank summands must match it.
    host_relations_t host_relations;
    util::u32_t const prefixes[1] = {hidden_k};
    auto const oracle_of = [&](st::basic_vector<util::f64_t> const& towers) noexcept {
        f64r2_view_t const rows {towers.data(), gpus * vertices_k, hidden_k};
        return host_loss(rows, rows, prefixes, host_relations, 1.0, 0.0, gpus * vertices_k,
                         contrastive_pair_loss_form_t::balanced_k);
    };
    util::f64_t const oracle = oracle_of(global_tower);
    util::f64_t const relative = std::fabs(distributed_loss - oracle) / fu::max_of(std::fabs(oracle), 1.0e-9);
    sz::string_view_t const tier = domain.description();
    expect_true(relative < 5.0e-3,
                "distributed loss sum {:.6f} matches the global Float64 oracle {:.6f}: relative {:.2e} over {}",
                distributed_loss, oracle, relative, tier);

    // Finite-difference one channel of rank 1's tower against the combined distributed force.
    std::size_t const probe_index = scalars_k + 321; // Inside rank 1's rows.
    util::f64_t const step_size = 1.0e-2, original = global_tower[probe_index];
    global_tower[probe_index] = original + step_size;
    if (!expect_true(narrow_to_bf16(global_tower, staging), "perturbation rounding failed")) return {};
    util::f64_t const plus = global_tower[probe_index], loss_plus = oracle_of(global_tower);
    global_tower[probe_index] = original - step_size;
    if (!expect_true(narrow_to_bf16(global_tower, staging), "perturbation rounding failed")) return {};
    util::f64_t const minus = global_tower[probe_index], loss_minus = oracle_of(global_tower);
    util::f64_t const numeric = (loss_plus - loss_minus) / (plus - minus);
    util::f64_t const analytic = distributed_grad[probe_index];
    expect_true(std::fabs(numeric - analytic) < 5.0e-2 * fu::max_of(std::fabs(numeric), 0.05),
                "distributed force matches the global finite difference: numeric {:.5f} distributed {:.5f}", numeric,
                analytic);
    return {};
}

/** The cross-domain tier: one process per multimem_domain, GDAKI gather + all-reduce over the local
 *  HCAs, verified against the same global Float64 oracle each process computes independently.
 *  Activated by UFORM_HOST_DOMAIN_INDEX in the environment. */
static case_result_t test_cross_domain() {
#if !UFORM_WITH_INFINIBAND
    return case_result_t::skipped("built without UFORM_WITH_INFINIBAND");
#else
    if (!cuda_devices_visible_(1)) return case_result_t::skipped(no_gpu_device_k);
    int device_count = 0;
    cuDeviceGetCount(&device_count);
    util::u32_t const gpus = util::u32_t(device_count);
    util::u32_t const domain = util::u32_t(std::atoi(std::getenv("UFORM_HOST_DOMAIN_INDEX")));
    auto [topology, bootstrap_status] = util::mesh_topology_t::bootstrap(domain, 2, &pause_between_polls, gpus);
    if (!expect_true(bootstrap_status, "bootstrap: {}", bootstrap_status.description_)) return {};

    constexpr util::u32_t vertices_k = 8, hidden_k = 64;
    std::size_t const scalars_k = std::size_t(vertices_k) * hidden_k;
    util::u32_t const world_gpus = 2 * gpus;
    util::u32_t prefix_dims[1] = {hidden_k};
    st::basic_vector<util::f64_t> global_tower, local_grad;
    if (!expect_true(
            !st::failed(global_tower.resize(world_gpus * scalars_k)) && !st::failed(local_grad.resize(scalars_k)),
            "host staging failed"))
        return {};
    std::vector<nk_bf16_t> staging(global_tower.size());
    for (std::size_t index = 0; index != global_tower.size(); ++index) global_tower[index] = hash_fill(index, 7);
    if (!expect_true(narrow_to_bf16(global_tower, staging), "tower rounding failed")) return {};
    nk::tensor_span<nk_bf16_t, 2> const staging_rows {staging.data(), world_gpus, scalars_k};

    util::per_gpu_array<util::gpu_stream_t> streams;
    util::per_gpu_array<util::gpu_arena_t> arenas;
    std::vector<contrastive_relations_t> owners(gpus);
    std::vector<contrastive_relations_view_t> relations(gpus);
    std::vector<util::bf16r2_view_t> towers(gpus);
    std::vector<util::bf16r2_span_t> grads(gpus);
    contrastive_segments_config_t config;
    config.from_vertices_per_gpu = vertices_k;
    config.hidden = hidden_k;
    config.to_vertices_per_gpu = vertices_k;
    config.gpu_subgraphs_per_step = world_gpus;
    std::vector<util::i32_t> devices(gpus);
    for (util::u32_t rank = 0; rank != gpus; ++rank) devices[rank] = util::i32_t(rank);
    for (util::u32_t rank = 0; rank != gpus; ++rank) {
        auto [stream, make_stream_status] = util::gpu_stream_t::make(util::i32_t(rank));
        if (!expect_true(make_stream_status && streams.push_back(std::move(stream)) == util::status_code_t::success_k,
                         "cross-domain stream make on GPU {}", rank))
            return {};
        cuCtxPushCurrent(streams[rank].context());
        auto [arena, make_arena_status] = util::gpu_arena_t::make(util::i32_t(rank), 1ull << 30,
                                                                  {devices.data(), gpus});
        if (!expect_true(make_arena_status && arenas.push_back(std::move(arena)) == util::status_code_t::success_k,
                         "cross-domain arena make on GPU {}", rank))
            return {};
        util::bf16_t* tower = arenas[rank].allocate_object<util::bf16_t>(scalars_k);
        util::bf16_t* grad = arenas[rank].allocate_object<util::bf16_t>(scalars_k);
        std::size_t const global_rank = std::size_t(domain) * gpus + rank;
        cuMemcpyHtoD(util::device_address(tower), staging_rows[global_rank, nk::slice].data(),
                     scalars_k * sizeof(nk_bf16_t));
        towers[rank] = util::bf16r2_view_t {tower, vertices_k, hidden_k};
        grads[rank] = util::bf16r2_span_t {grad, vertices_k, hidden_k};
        auto [owner, status] = contrastive_relations_t::make(arenas[rank], vertices_k, 0);
        if (!expect_true(status, "cross-domain relations make: {}", status.description_)) return {};
        owners[rank] = std::move(owner);
        util::expected_t staged = owners[rank].upload_self_edges(
            vertices_k, {}, {}, global_rank * std::size_t(vertices_k), 1.0f, streams[rank]);
        if (!expect_true(staged, "cross-domain relations: {}", staged.description_)) return {};
        relations[rank] = owners[rank].view();
        cuCtxPopCurrent(nullptr);
    }
    // The composition roots, in main-order: the ONE multimem domain, the ONE colocated domain.
    auto [multimem, make_multimem_status] = util::multimem_domain_t::make(
        topology, {devices.data(), gpus},
        distributed_contrastive_segments_t::payload_scalars_needed(config, topology.shape), streams,
        wait_limit_k.count());
    if (!expect_true(make_multimem_status, "multimem domain make: {}", make_multimem_status.description_)) return {};
    auto [colocated, make_colocated_status] = util::colocated_domain_t::make(topology);
    if (!expect_true(make_colocated_status, "colocated domain make: {}", make_colocated_status.description_)) return {};
    nk_capability_t capabilities = 0;
    if (!expect_true(nk_capabilities_enabled_cuda(0, &capabilities) == nk_success_k, "capability query failed"))
        return {};
    util::gpu_numerics_t const numerics = ut::gpu_numerics_of(capabilities);
    auto [distributed, make_distributed_status] = distributed_contrastive_segments_t::make(
        multimem, colocated, arenas, streams, {towers.data(), gpus}, {relations.data(), gpus}, {grads.data(), gpus},
        config, {prefix_dims, 1}, numerics);
    if (!expect_true(make_distributed_status, "cross-domain make: {}", make_distributed_status.description_)) return {};
    sz::string_view_t const tier = multimem.description();
    util::u32_t const world_domains = distributed.world_domains();
    util::expected_t ran = distributed(contrastive_recipe_t {}, streams);
    if (!expect_true(ran, "cross-domain run over {} domains on {}: {}", world_domains, tier, ran.description_))
        return {};
    for (util::u32_t rank = 0; rank != gpus; ++rank) {
        cuCtxPushCurrent(streams[rank].context());
        CUresult const synchronized = cuStreamSynchronize(streams[rank].get());
        cuCtxPopCurrent(nullptr);
        if (!expect_true(synchronized == CUDA_SUCCESS, "cross-domain sync on GPU {}: result {}", rank,
                         static_cast<int>(synchronized)))
            return {};
    }

    // Combine the two domains' loss sums over the mesh exchange; each process checks
    // the global oracle.
    util::f64_t domain_loss = 0.0;
    {
        cuCtxPushCurrent(streams[0].context());
        std::vector<util::f32_t> summands(gpus);
        cuMemcpyDtoH(summands.data(), util::device_address(distributed.loss_outs().data()), gpus * sizeof(util::f32_t));
        cuCtxPopCurrent(nullptr);
        for (util::f32_t const summand : summands) domain_loss += summand;
    }
    sz::string_t loss_text;
    if (!expect_true(util::format_into(loss_text, "{}", domain_loss), "loss text formatting")) return {};
    if (util::expected_t published = topology.publish_text("loss", loss_text.view());
        !expect_true(published, "loss publish: {}", published.description_))
        return {};
    sz::string_t peer_text;
    if (util::expected_t awaited = topology.await_text(1u - domain, "loss", peer_text, wait_limit_k.count());
        !expect_true(awaited, "peer loss await: {}", awaited.description_))
        return {};
    util::f64_t const global_loss = domain_loss + std::atof(peer_text.c_str());

    host_relations_t host_relations;
    auto const oracle_of = [&](st::basic_vector<util::f64_t> const& towers) noexcept {
        f64r2_view_t const rows {towers.data(), world_gpus * vertices_k, hidden_k};
        return host_loss(rows, rows, prefix_dims, host_relations, 1.0, 0.0, world_gpus * vertices_k,
                         contrastive_pair_loss_form_t::balanced_k);
    };
    util::f64_t const oracle = oracle_of(global_tower);
    util::f64_t const relative = std::fabs(global_loss - oracle) / fu::max_of(std::fabs(oracle), 1.0e-9);
    expect_true(relative < 5.0e-3,
                "cross-domain loss sum {:.6f} matches the global Float64 oracle {:.6f}: relative {:.2e} over {}",
                global_loss, oracle, relative, tier);

    // Finite-difference one locally-owned channel against the fully-reduced force.
    std::size_t const probe_index = (std::size_t(domain) * gpus + 1) * scalars_k + 321;
    std::vector<nk_bf16_t> local_staging(scalars_k);
    cuCtxPushCurrent(streams[1 % gpus].context());
    cuMemcpyDtoH(local_staging.data(), util::device_address(grads[1 % gpus].data()), scalars_k * sizeof(nk_bf16_t));
    cuCtxPopCurrent(nullptr);
    if (!expect_true(widen_from_bf16(local_staging, local_grad), "gradient widening failed")) return {};
    util::f64_t const step_size = 1.0e-2, original = global_tower[probe_index];
    global_tower[probe_index] = original + step_size;
    if (!expect_true(narrow_to_bf16(global_tower, staging), "perturbation rounding failed")) return {};
    util::f64_t const plus = global_tower[probe_index], loss_plus = oracle_of(global_tower);
    global_tower[probe_index] = original - step_size;
    if (!expect_true(narrow_to_bf16(global_tower, staging), "perturbation rounding failed")) return {};
    util::f64_t const minus = global_tower[probe_index], loss_minus = oracle_of(global_tower);
    util::f64_t const numeric = (loss_plus - loss_minus) / (plus - minus);
    util::f64_t const analytic = local_grad[probe_index % scalars_k];
    expect_true(std::fabs(numeric - analytic) < 5.0e-2 * fu::max_of(std::fabs(numeric), 0.05),
                "cross-domain force matches the global finite difference: numeric {:.5f} distributed {:.5f}", numeric,
                analytic);
    util::expected_t closed = topology.barrier("done", wait_limit_k.count());
    expect_true(static_cast<bool>(closed), "the closing barrier should complete: {}", closed.description_);
#endif
    return {};
}

/** The FP64 attention-pool + head parity gate: seed a tiny text tower with keyed weights, run the
 *  contrastive forward over a fabricated ragged batch (one PAD segment included — the historical
 *  pool failure mode was PAD masking, never arithmetic), and compare the emitted vertices against
 *  a double-precision host replica of exactly the new math — the softmax pool over the tower's own
 *  final-norm states, then RMSNorm → fc1 → SiLU → fc2. The backward runs the same chain in
 *  reverse, and the pool's own state and query gradients must match the host softmax backward to
 *  the BFloat16 floor. */
static case_result_t test_pool_parity() {
    if (!cuda_devices_visible_(1)) return case_result_t::skipped(no_gpu_device_k);
    constexpr util::u32_t rows_k = 2, segments_k = 4, columns_k = 8, hidden_k = 256, tokens_k = rows_k * columns_k;
    constexpr util::u32_t heads_k = hidden_k / transformer_head_dim_k;
    auto [stream, make_stream_status] = util::gpu_stream_t::make(0);
    if (!expect_true(make_stream_status, "pool-parity stream make: {}", make_stream_status.description_)) return {};
    nk_capability_t capabilities = 0;
    if (!expect_true(nk_capabilities_enabled_cuda(0, &capabilities) == nk_success_k, "capability query failed"))
        return {};
    util::gpu_numerics_t const numerics = ut::gpu_numerics_of(capabilities);
    cuCtxPushCurrent(stream.context());
    auto [arena, make_arena_status] = util::gpu_arena_t::make(0, 1ull << 30, {});
    if (!make_arena_status) {
        expect_true(false, "pool-parity bring-up: {}", make_arena_status.description_);
        cuCtxPopCurrent(nullptr);
        return {};
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
    auto [tower, make_tower_status] = encoder_tower_bf16_t::make(arena, config, shape, numerics);
    if (!make_tower_status) {
        expect_true(false, "pool-parity tower make: {}", make_tower_status.description_);
        cuCtxPopCurrent(nullptr);
        return {};
    }

    // Keyed weight seeding through the registration list, each matrix kept on the BFloat16 grid in
    // Float64 for the host replica.
    st::basic_vector<util::differentiable_weights_t> weights;
    if (!tower.fill(weights)) {
        expect_true(false, "pool-parity registration");
        cuCtxPopCurrent(nullptr);
        return {};
    }
    st::basic_vector<st::basic_vector<util::f64_t>> host_weights;
    std::vector<nk_bf16_t> staging;
    if (st::failed(host_weights.resize(weights.size()))) {
        expect_true(false, "pool-parity weight staging");
        cuCtxPopCurrent(nullptr);
        return {};
    }
    for (std::size_t matrix = 0; matrix != weights.size(); ++matrix) {
        std::size_t const scalars = weights[matrix].weights.numel();
        if (st::failed(host_weights[matrix].resize(scalars))) {
            expect_true(false, "pool-parity weight staging");
            cuCtxPopCurrent(nullptr);
            return {};
        }
        staging.resize(scalars);
        for (std::size_t index = 0; index != scalars; ++index)
            host_weights[matrix][index] = hash_fill(index, matrix + 101);
        if (!narrow_to_bf16(host_weights[matrix], staging)) {
            expect_true(false, "pool-parity weight rounding");
            cuCtxPopCurrent(nullptr);
            return {};
        }
        cuMemcpyHtoD(util::device_address(weights[matrix].weights.data()), staging.data(), scalars * sizeof(nk_bf16_t));
    }

    // A fabricated ragged batch: three live text segments of lengths {8, 5, 2}, a gap token closing
    // the second row, and one PAD — the batch packer's contract (offsets repeat on PAD, lengths
    // carry the live counts).
    auto [batch, make_batch_status] = multimodal_batch_on_gpu_t::make(util::gpu_arena_alloc<util::b8_t> {arena}, shape);
    if (!make_batch_status) {
        expect_true(false, "pool-parity batch make: {}", make_batch_status.description_);
        cuCtxPopCurrent(nullptr);
        return {};
    }
    util::u32_t const host_offsets[segments_k + 1] = {0, 8, 13, 15, 15};
    util::u32_t const host_lengths[segments_k] = {8, 5, 2, 0};
    modality_t const host_modality[segments_k] = {modality_t::text_k, modality_t::text_k, modality_t::text_k,
                                                  modality_t::text_k};
    std::vector<util::token_id_t> host_tokens(tokens_k);
    for (util::u32_t token = 0; token != tokens_k; ++token)
        host_tokens[token] = static_cast<util::token_id_t>(token % 251u);
    cuMemcpyHtoD(util::device_address(batch.segment_offsets.data()), host_offsets, sizeof(host_offsets));
    cuMemcpyHtoD(util::device_address(batch.segment_lengths.data()), host_lengths, sizeof(host_lengths));
    cuMemcpyHtoD(util::device_address(batch.segment_modality.data()), host_modality, sizeof(host_modality));
    cuMemcpyHtoD(util::device_address(batch.tokens.data()), host_tokens.data(), tokens_k * sizeof(util::token_id_t));

    // Forward with vertices, then read back the tower's own final-norm states — the oracle's input,
    // so the parity isolates exactly the new pool + head math.
    util::bf16r2_tensor_arena_t vertices_tensor = make_uninitialized<util::bf16r2_tensor_arena_t>(
        {segments_k, hidden_k}, util::gpu_arena_alloc<util::bf16_t> {arena});
    util::bf16r2_tensor_arena_t vertices_grad_tensor = make_uninitialized<util::bf16r2_tensor_arena_t>(
        {segments_k, hidden_k}, util::gpu_arena_alloc<util::bf16_t> {arena});
    if (!vertices_tensor || !vertices_grad_tensor) {
        expect_true(false, "pool-parity vertices alloc");
        cuCtxPopCurrent(nullptr);
        return {};
    }
    cuMemsetD8(util::device_address(vertices_tensor.data()), 0,
               std::size_t(segments_k) * hidden_k * sizeof(util::bf16_t));
    cuMemsetD8(util::device_address(vertices_grad_tensor.data()), 0,
               std::size_t(segments_k) * hidden_k * sizeof(util::bf16_t));
    util::expected_t ran = tower.pack_weights(stream);
    if (ran) ran = tower.forward(batch, vertices_tensor.span(), stream);
    expect_true(static_cast<bool>(ran), "pool-parity forward: {}", ran.description_);
    if (!ran) {
        cuCtxPopCurrent(nullptr);
        return {};
    }
    cuStreamSynchronize(stream.get());

    // The tower's states and vertices, widened once, beside the host replica's own rows.
    st::basic_vector<util::f64_t> host_states, device_vertices, host_pooled, host_force, host_states_grad,
        host_query_grad, host_vertices;
    if (st::failed(host_states.resize(std::size_t(tokens_k) * hidden_k)) ||
        st::failed(device_vertices.resize(std::size_t(segments_k) * hidden_k)) ||
        st::failed(host_pooled.resize(std::size_t(segments_k) * hidden_k)) ||
        st::failed(host_force.resize(std::size_t(segments_k) * hidden_k)) ||
        st::failed(host_states_grad.resize(std::size_t(tokens_k) * hidden_k)) ||
        st::failed(host_query_grad.resize(hidden_k)) ||
        st::failed(host_vertices.resize(std::size_t(segments_k) * hidden_k))) {
        expect_true(false, "pool-parity host staging");
        cuCtxPopCurrent(nullptr);
        return {};
    }
    staging.resize(host_states.size());
    cuMemcpyDtoH(staging.data(), util::device_address(tower.normalized().data()), staging.size() * sizeof(nk_bf16_t));
    bool widened = widen_from_bf16(staging, host_states);
    staging.resize(device_vertices.size());
    cuMemcpyDtoH(staging.data(), util::device_address(vertices_tensor.data()), staging.size() * sizeof(nk_bf16_t));
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
    f64r3_span_t const host_pooled_planes {host_pooled.data(), {segments_k, heads_k, transformer_head_dim_k}};
    // The pool's backward rides the same softmax with the keyed force as d(pooled) and K = V = the states:
    // dS = P∘(dO·Vᵀ − dO·O), d(states) = P·dO + scale·dS·q, d(query) = scale·Σ dS·K.
    std::vector<nk_bf16_t> host_force_staging(host_force.size());
    for (std::size_t index = 0; index != host_force.size(); ++index) host_force[index] = hash_fill(index, 777) * 0.1;
    widened = widened && narrow_to_bf16(host_force, host_force_staging);
    if (!widened) {
        expect_true(false, "pool-parity staging conversions");
        cuCtxPopCurrent(nullptr);
        return {};
    }
    f64r3_view_t const host_force_planes {host_force.data(), {segments_k, heads_k, transformer_head_dim_k}};
    f64r3_span_t const host_states_grad_planes {host_states_grad.data(), {tokens_k, heads_k, transformer_head_dim_k}};
    f64r2_span_t const host_query_grad_rows {host_query_grad.data(), heads_k, transformer_head_dim_k};
    for (util::u32_t segment = 0; segment != segments_k; ++segment) {
        util::u32_t const length = host_lengths[segment];
        if (length == 0u) continue; // a PAD vertex is don't-care — the loss PADs it out, so no oracle here
        util::u32_t const offset = host_offsets[segment];
        for (util::u32_t head = 0; head != heads_k; ++head) {
            f64r1_view_t const query_head = host_query_rows[head, nk::slice];
            f64r1_view_t const force_head = host_force_planes[segment, head, nk::slice];
            std::array<util::f64_t, columns_k> probabilities;
            double score_max = -1e300, denominator = 0.0;
            for (util::u32_t token = 0; token != length; ++token) {
                probabilities[token] = dot(query_head, host_state_planes[offset + token, head, nk::slice]) * scale;
                score_max = fu::max_of(score_max, probabilities[token]);
            }
            for (util::u32_t token = 0; token != length; ++token)
                denominator += probabilities[token] = std::exp(probabilities[token] - score_max);
            for (util::u32_t token = 0; token != length; ++token) probabilities[token] /= denominator;
            for (util::u32_t channel = 0; channel != transformer_head_dim_k; ++channel) {
                double value = 0.0;
                for (util::u32_t token = 0; token != length; ++token)
                    value += probabilities[token] * host_state_planes[offset + token, head, channel];
                host_pooled_planes[segment, head, channel] = value;
            }
            double const output_dot = dot(force_head, host_pooled_planes[segment, head, nk::slice]);
            for (util::u32_t token = 0; token != length; ++token) {
                f64r1_view_t const state = host_state_planes[offset + token, head, nk::slice];
                double const score_grad = probabilities[token] * (dot(force_head, state) - output_dot);
                for (util::u32_t channel = 0; channel != transformer_head_dim_k; ++channel) {
                    host_states_grad_planes[offset + token, head, channel] +=
                        probabilities[token] * force_head[channel] + scale * score_grad * query_head[channel];
                    host_query_grad_rows[head, channel] += scale * score_grad * state[channel];
                }
            }
        }
    }
    f64r2_view_t const host_pooled_rows {host_pooled.data(), segments_k, hidden_k};
    f64r2_view_t const device_vertices_rows {device_vertices.data(), segments_k, hidden_k};
    f64r2_span_t const host_vertices_rows {host_vertices.data(), segments_k, hidden_k};
    for (util::u32_t segment = 0; segment != segments_k; ++segment) {
        if (host_lengths[segment] == 0u) continue; // PAD vertices stay outside the parity
        f64r1_view_t const pooled_row = host_pooled_rows[segment, nk::slice];
        double const inverse_rms = 1.0 / std::sqrt(dot(pooled_row, pooled_row) / hidden_k + 1e-6);
        std::array<util::f64_t, hidden_k> normalized, activated;
        for (util::u32_t channel = 0; channel != hidden_k; ++channel)
            normalized[channel] = pooled_row[channel] * inverse_rms * host_head_norm[channel];
        for (util::u32_t out = 0; out != hidden_k; ++out) {
            double const projected = dot(host_fc1_rows.row(out), normalized);
            activated[out] = projected / (1.0 + std::exp(-projected));
        }
        for (util::u32_t out = 0; out != hidden_k; ++out)
            host_vertices_rows[segment, out] = dot(host_fc2_rows.row(out), activated);
    }
    double worst = 0.0;
    for (util::u32_t segment = 0; segment != segments_k; ++segment) {
        if (host_lengths[segment] == 0u) continue; // PAD vertices stay outside the parity
        for (util::u32_t channel = 0; channel != hidden_k; ++channel)
            worst = fu::max_of(worst,
                               std::abs(host_vertices_rows[segment, channel] - device_vertices_rows[segment, channel]) /
                                   (std::abs(host_vertices_rows[segment, channel]) + 1.0));
    }
    expect_true(worst < 3e-2,
                "pool + head forward matches the FP64 oracle to the BFloat16 floor: worst relative {:.3e}", worst);

    // Backward smoke over the same chain: the keyed vertices force, then every pool / head gradient
    // finite and the query gradient nonzero (the pool actually differentiated).
    cuMemcpyHtoD(util::device_address(vertices_grad_tensor.data()), host_force_staging.data(),
                 host_force_staging.size() * sizeof(nk_bf16_t));
    util::expected_t propagated = tower.backward(vertices_grad_tensor.view(), util::bf16r2_view_t {}, batch, stream);
    expect_true(static_cast<bool>(propagated), "pool-parity backward: {}", propagated.description_);
    if (propagated) {
        cuStreamSynchronize(stream.get());
        bool all_finite = true;
        double query_grad_energy = 0.0;
        st::basic_vector<util::f64_t> gradient;
        for (std::size_t matrix = query_index; matrix != weights.size(); ++matrix) {
            std::size_t const scalars = weights[matrix].weights_grad.numel();
            staging.resize(scalars);
            if (!expect_true(!st::failed(gradient.resize(scalars)), "pool-parity gradient staging")) break;
            cuMemcpyDtoH(staging.data(), util::device_address(weights[matrix].weights_grad.data()),
                         scalars * sizeof(nk_bf16_t));
            all_finite = all_finite && widen_from_bf16(staging, gradient);
            for (util::f64_t const value : gradient) all_finite = all_finite && std::isfinite(value);
            if (matrix == query_index) query_grad_energy = dot(gradient, gradient);
        }
        expect_true(all_finite, "every pool / head gradient is finite");
        expect_true(query_grad_energy > 0.0, "the pool query gradient is nonzero");
    }

    // The tower's d(normalized) is overwritten by the block stack's backward before it returns,
    // so a second pool replays forward + backward over the tower's states and query, into the
    // spent vertices buffer.
    auto [pool, make_pool_status] = tower_pool_t::make(arena, config.block, numerics);
    util::f32r2_tensor_arena_t states_grad = make_uninitialized<util::f32r2_tensor_arena_t>(
        {tokens_k, hidden_k}, util::gpu_arena_alloc<util::f32_t> {arena});
    util::f32r2_tensor_arena_t value_grad = make_uninitialized<util::f32r2_tensor_arena_t>(
        {tokens_k, hidden_k}, util::gpu_arena_alloc<util::f32_t> {arena});
    util::bf16r1_tensor_arena_t query_grad = make_uninitialized<util::bf16r1_tensor_arena_t>(
        {hidden_k}, util::gpu_arena_alloc<util::bf16_t> {arena});
    if (!make_pool_status || !states_grad || !value_grad || !query_grad) {
        expect_true(false, "pool-parity pool make: {}", make_pool_status.description_);
        cuCtxPopCurrent(nullptr);
        return {};
    }
    util::bf16r1_view_t const query {weights[query_index].weights.data(), hidden_k};
    util::expected_t pooled = pool.forward(query, tower.normalized(), batch.segment_offsets.view(),
                                           batch.segment_lengths.view(), vertices_tensor.span(), stream);
    if (pooled)
        pooled = pool.backward(query, vertices_grad_tensor.view(), states_grad.span(), value_grad.span(),
                               query_grad.span(), stream);
    expect_true(static_cast<bool>(pooled), "pool-parity pool backward: {}", pooled.description_);
    if (pooled) {
        cuStreamSynchronize(stream.get());
        std::vector<util::f32_t> device_states_grad(host_states_grad.size());
        cuMemcpyDtoH(device_states_grad.data(), util::device_address(states_grad.data()),
                     device_states_grad.size() * sizeof(util::f32_t));
        st::basic_vector<util::f64_t> device_query_grad;
        if (!expect_true(!st::failed(device_query_grad.resize(hidden_k)), "pool-parity query gradient staging")) {
            cuCtxPopCurrent(nullptr);
            return {};
        }
        staging.resize(hidden_k);
        cuMemcpyDtoH(staging.data(), util::device_address(query_grad.data()), hidden_k * sizeof(nk_bf16_t));
        bool const query_widened = widen_from_bf16(staging, device_query_grad);
        std::vector<bool> live_rows(tokens_k, false);
        for (util::u32_t segment = 0; segment != segments_k; ++segment)
            for (util::u32_t token = 0; token != host_lengths[segment]; ++token)
                live_rows[host_offsets[segment] + token] = true;
        util::f32r2_view_t const device_states_grad_rows {device_states_grad.data(), tokens_k, hidden_k};
        f64r2_view_t const host_states_grad_rows {host_states_grad.data(), tokens_k, hidden_k};
        double states_peak = 0.0, states_worst = 0.0, query_peak = 0.0, query_worst = 0.0;
        bool gaps_zero = true;
        for (util::u32_t token = 0; token != tokens_k; ++token)
            for (util::u32_t channel = 0; channel != hidden_k; ++channel) {
                util::f64_t const device_value = device_states_grad_rows[token, channel];
                if (!live_rows[token]) {
                    gaps_zero = gaps_zero && device_value == 0.0;
                    continue;
                }
                states_peak = fu::max_of(states_peak, std::abs(host_states_grad_rows[token, channel]));
                states_worst = fu::max_of(states_worst, std::abs(host_states_grad_rows[token, channel] - device_value));
            }
        for (util::u32_t channel = 0; channel != hidden_k; ++channel) {
            query_peak = fu::max_of(query_peak, std::abs(host_query_grad[channel]));
            query_worst = fu::max_of(query_worst, std::abs(host_query_grad[channel] - device_query_grad[channel]));
        }
        expect_true(query_widened, "the query gradient widens");
        states_worst /= states_peak;
        query_worst /= query_peak;
        // NumKong rounds P and dS to BFloat16 for its tensor-core products, and the query gradient
        // lands in BFloat16: a few 2⁻⁹ unit roundoffs over F32 sums.
        expect_true(states_worst < 1e-2,
                    "pool d(states) matches the FP64 oracle on live rows to the BFloat16 floor: worst relative {:.3e}",
                    states_worst);
        expect_true(gaps_zero, "pool d(states) is exactly zero on gap rows");
        expect_true(query_worst < 1e-2,
                    "pool query gradient matches the FP64 oracle to the BFloat16 floor: worst relative {:.3e}",
                    query_worst);
    }

    // The quarter-precision twin: quantize the same tower and require every live vertex to
    // stay on the BFloat16 tower's direction — the blocks carry the E4M3 error, the tail is
    // a BFloat16 snapshot.
    auto [twin, quantize_status] = tower.quantize_weights(arena, stream);
    expect_true(static_cast<bool>(quantize_status), "twin quantization: {}", quantize_status.description_);
    if (quantize_status) {
        util::bf16r2_tensor_arena_t twin_vertices = make_uninitialized<util::bf16r2_tensor_arena_t>(
            {segments_k, hidden_k}, util::gpu_arena_alloc<util::bf16_t> {arena});
        if (!twin_vertices) {
            expect_true(false, "twin vertices alloc");
            cuCtxPopCurrent(nullptr);
            return {};
        }
        util::expected_t twin_ran = twin.forward(batch, twin_vertices.span(), stream);
        expect_true(static_cast<bool>(twin_ran), "twin forward: {}", twin_ran.description_);
        if (twin_ran) {
            cuStreamSynchronize(stream.get());
            st::basic_vector<util::f64_t> device_twin_vertices;
            if (!expect_true(!st::failed(device_twin_vertices.resize(std::size_t(segments_k) * hidden_k)),
                             "twin vertices staging")) {
                cuCtxPopCurrent(nullptr);
                return {};
            }
            staging.resize(device_twin_vertices.size());
            cuMemcpyDtoH(staging.data(), util::device_address(twin_vertices.data()),
                         staging.size() * sizeof(nk_bf16_t));
            expect_true(widen_from_bf16(staging, device_twin_vertices), "the twin vertices widen");
            f64r2_view_t const device_twin_vertices_rows {device_twin_vertices.data(), segments_k, hidden_k};
            double worst_cosine = 1.0;
            for (util::u32_t segment = 0; segment != segments_k; ++segment) {
                if (host_lengths[segment] == 0u) continue; // PAD vertices stay outside the parity
                f64r1_view_t const reference = device_vertices_rows[segment, nk::slice];
                f64r1_view_t const quantized = device_twin_vertices_rows[segment, nk::slice];
                worst_cosine = fu::min_of(
                    worst_cosine, dot(reference, quantized) /
                                      (std::sqrt(dot(reference, reference) * dot(quantized, quantized)) + 1e-30));
            }
            // Keyed-random weights are the quantizer's worst case (trained weights are
            // smoother), and this observed 0.979 at depth 1 already beats the R&D's FP8
            // embedding acceptance gates.
            expect_true(worst_cosine > 0.95,
                        "twin vertices stay on the BFloat16 tower's direction: worst cosine {:.4f}", worst_cosine);
        }
    }
    cuCtxPopCurrent(nullptr);
    return {};
}

/** One GPU's late-interaction layout for the tokens cases: a triplet bag, a multi-positive bag, a
 *  PAD segment and a segment that both queries and is queried, over ragged spans and documents laid
 *  across three bands. */
struct host_tokens_layout_t {
    static constexpr std::size_t segments_k = 8, width_k = 32, head_rows_k = 96, query_rows_k = 16;
    static constexpr std::size_t documents_k = 8, document_rows_k = 128, band_k = 32;
    static constexpr util::u32_t spans_k[segments_k] = {3, 20, 9, 2, 17, 12, 0, 6};
    static constexpr late_interaction_side_t sides_k[segments_k] = {
        late_interaction_side_t::query_k,    late_interaction_side_t::document_k,
        late_interaction_side_t::document_k, late_interaction_side_t::query_k,
        late_interaction_side_t::document_k, late_interaction_side_t::document_k,
        late_interaction_side_t::none_k,     late_interaction_side_t::query_and_document_k};

    /** The head rows one copy fills, every sided span back to back. */
    [[nodiscard]] static constexpr std::size_t live_rows() noexcept {
        std::size_t rows = 0;
        for (util::u32_t const span : spans_k) rows += span;
        return rows;
    }

    [[nodiscard]] static bool queries(late_interaction_side_t side) noexcept {
        return side == late_interaction_side_t::query_k || side == late_interaction_side_t::query_and_document_k;
    }
    [[nodiscard]] static bool documents(late_interaction_side_t side) noexcept {
        return side == late_interaction_side_t::document_k || side == late_interaction_side_t::query_and_document_k;
    }
};

/** The Float64 ranked MaxSim oracle over @p gpus copies of @ref host_tokens_layout_t , whose
 *  head rows sit back to back in @p vectors : @c S(q,d)=mean_t max_s⟨q_t,d_s⟩ , and per query
 *  row the weighted softplus of @c −β(s⁺−s⁻) over its positive and negative pairs, averaged over
 *  the live queries. */
static util::f64_t host_tokens_loss(std::span<util::f64_t const> vectors, std::size_t gpus,
                                    host_relations_t const& relations, util::f64_t beta) {
    using layout_t = host_tokens_layout_t;
    std::size_t const segments = gpus * layout_t::segments_k, width = layout_t::width_k;
    std::vector<std::size_t> first_rows(segments + 1u, 0);
    for (std::size_t segment = 0; segment != segments; ++segment)
        first_rows[segment + 1u] = first_rows[segment] + layout_t::spans_k[segment % layout_t::segments_k];
    auto row_of = [&](std::size_t segment, std::size_t token) {
        return (segment / layout_t::segments_k) * layout_t::head_rows_k + first_rows[segment] -
               first_rows[segment - segment % layout_t::segments_k] + token;
    };
    auto score = [&](std::size_t query, std::size_t document) {
        util::f64_t sum = 0;
        for (std::size_t token = 0; token != layout_t::spans_k[query % layout_t::segments_k]; ++token) {
            util::f64_t best = -1.0e300;
            for (std::size_t other = 0; other != layout_t::spans_k[document % layout_t::segments_k]; ++other) {
                util::f64_t const product = dot(f64r1_view_t {vectors.data() + row_of(query, token) * width, width},
                                                f64r1_view_t {vectors.data() + row_of(document, other) * width, width});
                best = fu::max_of(best, product);
            }
            sum += best;
        }
        return sum / util::f64_t(layout_t::spans_k[query % layout_t::segments_k]);
    };
    util::f64_t loss = 0;
    std::size_t live_queries = 0;
    struct scored_t {
        util::f64_t score, weight;
    };
    std::vector<scored_t> positives, negatives;
    for (std::size_t query = 0; query != segments; ++query) {
        late_interaction_side_t const side = layout_t::sides_k[query % layout_t::segments_k];
        if (!layout_t::queries(side) || layout_t::spans_k[query % layout_t::segments_k] == 0) continue;
        ++live_queries;
        positives.clear(), negatives.clear();
        for (std::size_t document = 0; document != segments; ++document) {
            late_interaction_side_t const other = layout_t::sides_k[document % layout_t::segments_k];
            if (!layout_t::documents(other) || layout_t::spans_k[document % layout_t::segments_k] == 0) continue;
            host_relations_t::resolved_edge_t const resolved = relations.resolve(util::u32_t(query),
                                                                                 util::u32_t(document));
            if (resolved.weight == 0) continue;
            (resolved.target > 0 ? positives : negatives).push_back({beta * score(query, document), resolved.weight});
        }
        util::f64_t positive_weight = 0, negative_weight = 0, row = 0;
        for (scored_t const& positive : positives) positive_weight += positive.weight;
        for (scored_t const& negative : negatives) negative_weight += negative.weight;
        for (scored_t const& positive : positives)
            for (scored_t const& negative : negatives) {
                util::f64_t const margin = negative.score - positive.score;
                row += positive.weight * negative.weight *
                       (fu::max_of(margin, 0.0) + std::log1p(std::exp(-std::fabs(margin))));
            }
        if (positive_weight > 0 && negative_weight > 0) loss += row / (positive_weight * negative_weight);
    }
    return loss / util::f64_t(fu::max_of<std::size_t>(live_queries, 1));
}

/** The per-GPU relation rows of @ref host_tokens_layout_t on GPU @p gpu of @p gpus : the
 *  triplet's attract and hard negative, the multi-positive bag's two positives, and the two-sided
 *  segment attracting the next GPU's first positive while ignoring itself. Appended to @p world
 *  as global rows. */
static void append_tokens_relations(host_relations_t& world, std::size_t gpu, std::size_t gpus) {
    util::u32_t const base = util::u32_t(gpu * host_tokens_layout_t::segments_k);
    util::u32_t const next = util::u32_t(((gpu + 1u) % gpus) * host_tokens_layout_t::segments_k);
    if (world.attract_offsets.empty()) world.attract_offsets = world.repel_offsets = world.ignore_offsets = {0};
    for (util::u32_t segment = 0; segment != host_tokens_layout_t::segments_k; ++segment) {
        if (segment == 0) world.attract_ids.push_back(base + 1), world.attract_scores.push_back(1.0f);
        if (segment == 0) world.repel_ids.push_back(base + 2), world.repel_scores.push_back(2.0f);
        if (segment == 3)
            world.attract_ids.insert(world.attract_ids.end(), {base + 4, base + 5}),
                world.attract_scores.insert(world.attract_scores.end(), {1.0f, 0.5f});
        if (segment == 7) {
            util::u32_t const low = fu::min_of(base + 1, next + 1), high = fu::max_of(base + 1, next + 1);
            world.attract_ids.push_back(low), world.attract_scores.push_back(1.0f);
            if (high != low) world.attract_ids.push_back(high), world.attract_scores.push_back(1.0f);
            world.ignore_ids.push_back(base + 7);
        }
        world.attract_offsets.push_back(util::u32_t(world.attract_ids.size()));
        world.repel_offsets.push_back(util::u32_t(world.repel_ids.size()));
        world.ignore_offsets.push_back(util::u32_t(world.ignore_ids.size()));
    }
}

/** One GPU's device inputs of the tokens cases: the head rows, their offsets and sides, the
 *  relation rows and the outputs, all off its arena. */
struct device_tokens_inputs_t {
    util::bf16r2_view_t token_vectors {};
    util::u32r1_view_t token_offsets {};
    multimodal_batch_view_t batch {};
    contrastive_relations_t relations {};
    util::bf16r2_span_t token_vectors_grad {};
    util::f32r1_span_t loss_out {};
};

/** Stage @p copies back-to-back copies of the layout, host copies @c [first,first+copies) of
 *  @p staging , with their live head rows compacted the way the token head lays them out, and the
 *  matching relation rows of @p host into @p arena . */
[[nodiscard]] static util::expected<device_tokens_inputs_t> stage_tokens_inputs(
    util::gpu_arena_t& arena, std::vector<nk_bf16_t> const& staging, std::size_t first, std::size_t copies,
    host_relations_t const& host, std::size_t world_segments, util::gpu_stream_t& stream) {
    using layout_t = host_tokens_layout_t;
    using result_t = util::expected<device_tokens_inputs_t>;
    std::size_t const segments = copies * layout_t::segments_k, rows = copies * layout_t::head_rows_k;
    std::size_t const width = layout_t::width_k, live_rows = layout_t::live_rows();
    std::vector<util::u32_t> offsets(segments + 1u, 0);
    std::vector<late_interaction_side_t> sides(segments);
    for (std::size_t segment = 0; segment != segments; ++segment) {
        sides[segment] = layout_t::sides_k[segment % layout_t::segments_k];
        offsets[segment + 1u] = offsets[segment] + layout_t::spans_k[segment % layout_t::segments_k];
    }
    std::vector<nk_bf16_t> compacted(rows * width, nk_bf16_t {});
    for (std::size_t copy = 0; copy != copies; ++copy)
        std::memcpy(compacted.data() + copy * live_rows * width,
                    staging.data() + (first + copy) * layout_t::head_rows_k * width,
                    live_rows * width * sizeof(nk_bf16_t));
    device_tokens_inputs_t inputs;
    util::bf16_t* vectors = arena.allocate_object<util::bf16_t>(rows * width);
    util::bf16_t* grad = arena.allocate_object<util::bf16_t>(rows * width);
    util::u32_t* offsets_device = arena.allocate_object<util::u32_t>(segments + 1u);
    late_interaction_side_t* sides_device = arena.allocate_object<late_interaction_side_t>(segments);
    util::f32_t* loss = arena.allocate_object<util::f32_t>(1);
    cuMemcpyHtoD(util::device_address(vectors), compacted.data(), compacted.size() * sizeof(nk_bf16_t));
    cuMemcpyHtoD(util::device_address(offsets_device), offsets.data(), offsets.size() * sizeof(util::u32_t));
    cuMemcpyHtoD(util::device_address(sides_device), sides.data(), sides.size() * sizeof(late_interaction_side_t));
    inputs.token_vectors = util::bf16r2_view_t {vectors, rows, width};
    inputs.token_vectors_grad = util::bf16r2_span_t {grad, rows, width};
    inputs.token_offsets = util::u32r1_view_t {offsets_device, segments + 1u};
    inputs.batch.segment_side = nk::tensor_view<late_interaction_side_t, 1> {sides_device, segments};
    inputs.loss_out = util::f32r1_span_t {loss, 1};

    // This range's relation rows, sliced from the world rows.
    std::size_t const row_begin = first * layout_t::segments_k;
    auto rebased = [&](std::vector<util::u32_t> const& world_offsets) {
        std::vector<util::u32_t> local;
        for (std::size_t row = row_begin; row <= row_begin + segments; ++row)
            local.push_back(world_offsets[row] - world_offsets[row_begin]);
        return local;
    };
    auto sliced = [&](std::vector<util::u32_t> const& world_offsets, auto const& world_values) {
        return std::vector(world_values.begin() + world_offsets[row_begin],
                           world_values.begin() + world_offsets[row_begin + segments]);
    };
    std::vector<util::u32_t> attract_offsets = rebased(host.attract_offsets),
                             repel_offsets = rebased(host.repel_offsets);
    std::vector<util::u32_t> ignore_offsets = rebased(host.ignore_offsets);
    std::vector<util::u32_t> attract_ids = sliced(host.attract_offsets, host.attract_ids);
    std::vector<util::u32_t> repel_ids = sliced(host.repel_offsets, host.repel_ids);
    std::vector<util::u32_t> ignore_ids = sliced(host.ignore_offsets, host.ignore_ids);
    std::vector<util::f32_t> attract_scores = sliced(host.attract_offsets, host.attract_scores);
    std::vector<util::f32_t> repel_scores = sliced(host.repel_offsets, host.repel_scores);
    auto [relations, status] = contrastive_relations_t::make(arena, util::u32_t(segments), 64);
    if (!status) return result_t::fail(std::move(status));
    inputs.relations = std::move(relations);
    util::expected_t uploaded = inputs.relations.upload(
        {std::span {attract_offsets}, std::span {attract_ids}, std::span {attract_scores}},
        {std::span {repel_offsets}, std::span {repel_ids}, std::span {repel_scores}},
        {std::span {ignore_offsets}, std::span {ignore_ids}, {}}, {}, {}, {}, {}, world_segments, row_begin,
        host.ambient, stream);
    if (!uploaded) return result_t::fail(std::move(uploaded));
    return result_t {std::move(inputs)};
}

/** The tokens loss geometry for @p copies back-to-back layouts on one GPU and
 *  @p subgraphs query tiles. */
[[nodiscard]] static contrastive_tokens_config_t tokens_config_(std::size_t copies, std::size_t subgraphs) noexcept {
    using layout_t = host_tokens_layout_t;
    contrastive_tokens_config_t config;
    config.segments_per_gpu = util::u32_t(copies * layout_t::segments_k);
    config.head_tokens_per_gpu = util::u32_t(copies * layout_t::head_rows_k);
    config.query_tokens_per_gpu = util::u32_t(copies * layout_t::query_rows_k);
    config.documents_per_gpu = util::u32_t(copies * layout_t::documents_k);
    config.document_tokens_per_gpu = util::u32_t(copies * layout_t::document_rows_k);
    config.band_tokens = util::u32_t(layout_t::band_k);
    config.width = util::u32_t(layout_t::width_k);
    config.gpu_subgraphs_per_step = util::u32_t(subgraphs);
    config.pair_loss = contrastive_pair_loss_form_t::ranked_k;
    return config;
}

/** The single-GPU tokens loss against the Float64 ranked MaxSim oracle, with central finite
 *  differences on query and document token entries. */
static case_result_t test_tokens_oracle() {
    using layout_t = host_tokens_layout_t;
    if (!cuda_devices_visible_(1)) return case_result_t::skipped(no_gpu_device_k);
    auto [stream, make_stream_status] = util::gpu_stream_t::make(0);
    if (!expect_true(make_stream_status, "stream: {}", make_stream_status.description_)) return {};
    cuCtxPushCurrent(stream.context());
    auto [arena, make_arena_status] = util::gpu_arena_t::make(0, 1ull << 30);
    if (!expect_true(make_arena_status, "arena: {}", make_arena_status.description_)) return {};

    std::size_t const scalars = layout_t::head_rows_k * layout_t::width_k;
    st::basic_vector<util::f64_t> vectors, gradient;
    if (!expect_true(!st::failed(vectors.resize(scalars)) && !st::failed(gradient.resize(scalars)),
                     "host staging failed"))
        return {};
    std::vector<nk_bf16_t> staging(scalars);
    for (std::size_t index = 0; index != scalars; ++index) vectors[index] = hash_fill(index, 29);
    if (!expect_true(narrow_to_bf16(vectors, staging) && widen_from_bf16(staging, vectors), "vector rounding failed"))
        return {};
    host_relations_t host;
    append_tokens_relations(host, 0, 1);
    auto [inputs, stage_status] = stage_tokens_inputs(arena, staging, 0, 1, host, layout_t::segments_k, stream);
    if (!expect_true(stage_status, "inputs: {}", stage_status.description_)) return {};
    nk_capability_t capabilities = 0;
    if (!expect_true(nk_capabilities_enabled_cuda(0, &capabilities) == nk_success_k, "capability query failed"))
        return {};
    util::gpu_numerics_t const numerics = ut::gpu_numerics_of(capabilities);
    auto [loss, make_loss_status] = contrastive_tokens_t::make(arena, tokens_config_(1, 1), numerics);
    if (!expect_true(make_loss_status, "loss make: {}", make_loss_status.description_)) return {};
    util::f64_t const beta = 4.0;
    contrastive_recipe_t recipe;
    recipe.inverse_temperature = util::f32_t(beta);
    util::expected_t ran = loss(inputs.token_vectors, inputs.token_offsets, inputs.batch, inputs.relations.view(),
                                recipe, inputs.loss_out, inputs.token_vectors_grad, stream);
    if (!expect_true(ran, "loss run: {}", ran.description_)) return {};
    cuStreamSynchronize(stream.get());
    util::f32_t device_loss = 0;
    cuMemcpyDtoH(&device_loss, util::device_address(inputs.loss_out.data()), sizeof(util::f32_t));
    util::f64_t const oracle = host_tokens_loss(vectors, 1, host, beta);
    util::f64_t const relative = std::fabs(util::f64_t(device_loss) - oracle) / fu::max_of(std::fabs(oracle), 1.0e-9);
    expect_true(relative < 5.0e-3, "tokens loss {:.6f} matches the Float64 oracle {:.6f}: relative {:.2e}", device_loss,
                oracle, relative);

    cuMemcpyDtoH(staging.data(), util::device_address(inputs.token_vectors_grad.data()), scalars * sizeof(nk_bf16_t));
    if (!expect_true(widen_from_bf16(staging, gradient), "gradient widening failed")) return {};
    // Query tokens of the triplet and the multi-positive bag, tokens of three documents, and
    // the two-sided segment.
    std::size_t const rows[] = {0, 2, 32, 33, 4, 12, 30, 53, 65};
    for (std::size_t const row : rows)
        for (std::size_t const channel : {std::size_t {3}, std::size_t {17}}) {
            std::size_t const index = row * layout_t::width_k + channel;
            util::f64_t const step = 1.0e-4, original = vectors[index];
            vectors[index] = original + step;
            util::f64_t const plus = host_tokens_loss(vectors, 1, host, beta);
            vectors[index] = original - step;
            util::f64_t const minus = host_tokens_loss(vectors, 1, host, beta);
            vectors[index] = original;
            util::f64_t const numeric = (plus - minus) / (2.0 * step);
            expect_true(std::fabs(numeric - gradient[index]) <= 2.0e-2 * fu::max_of(std::fabs(numeric), 0.02),
                        "token [{},{}] force: numeric {:.5f} analytic {:.5f}", row, channel, numeric, gradient[index]);
        }
    cuCtxPopCurrent(nullptr);
    return {};
}

/** The distributed tokens loss over every visible GPU, up to the host's limit, against the
 *  single-GPU loss over the same world laid out back to back. */
static case_result_t test_tokens_distributed() {
    using layout_t = host_tokens_layout_t;
    int device_count = 0;
    if (cuDeviceGetCount(&device_count) != CUDA_SUCCESS || device_count < 2)
        return case_result_t::skipped("needs two CUDA devices");
    std::size_t const gpus = fu::min_of<std::size_t>(static_cast<std::size_t>(device_count), util::max_gpus_per_host_k);
    std::size_t const rank_scalars = layout_t::head_rows_k * layout_t::width_k;
    util::i32_t devices[util::max_gpus_per_host_k] = {};
    for (std::size_t rank = 0; rank != gpus; ++rank) devices[rank] = static_cast<util::i32_t>(rank);
    st::basic_vector<util::f64_t> vectors, reference_gradient, widened;
    if (!expect_true(!st::failed(vectors.resize(gpus * rank_scalars)) &&
                         !st::failed(reference_gradient.resize(gpus * rank_scalars)) &&
                         !st::failed(widened.resize(rank_scalars)),
                     "host staging failed"))
        return {};
    std::vector<nk_bf16_t> staging(vectors.size());
    for (std::size_t index = 0; index != vectors.size(); ++index) vectors[index] = hash_fill(index, 31);
    if (!expect_true(narrow_to_bf16(vectors, staging), "vector rounding failed")) return {};
    nk_capability_t capabilities = 0;
    if (!expect_true(nk_capabilities_enabled_cuda(0, &capabilities) == nk_success_k, "capability query failed"))
        return {};
    util::gpu_numerics_t const numerics = ut::gpu_numerics_of(capabilities);
    host_relations_t host;
    for (std::size_t rank = 0; rank != gpus; ++rank) append_tokens_relations(host, rank, gpus);
    std::size_t const world_segments = gpus * layout_t::segments_k;
    contrastive_recipe_t recipe;
    recipe.inverse_temperature = 4.0f;

    // The single-GPU reference over the whole world on device 0.
    util::f32_t reference_loss = 0;
    {
        auto [stream, make_stream_status] = util::gpu_stream_t::make(0);
        if (!expect_true(make_stream_status, "stream: {}", make_stream_status.description_)) return {};
        cuCtxPushCurrent(stream.context());
        auto [arena, make_arena_status] = util::gpu_arena_t::make(0, 1ull << 30);
        if (!expect_true(make_arena_status, "arena: {}", make_arena_status.description_)) return {};
        auto [inputs, stage_status] = stage_tokens_inputs(arena, staging, 0, gpus, host, world_segments, stream);
        auto [loss, make_loss_status] = stage_status
                                            ? contrastive_tokens_t::make(arena, tokens_config_(gpus, 1), numerics)
                                            : util::expected<contrastive_tokens_t>::fail(std::move(stage_status));
        util::expected_t ran = make_loss_status ? loss(inputs.token_vectors, inputs.token_offsets, inputs.batch,
                                                       inputs.relations.view(), recipe, inputs.loss_out,
                                                       inputs.token_vectors_grad, stream)
                                                : util::expected_t::fail(std::move(make_loss_status));
        if (!expect_true(ran, "reference run: {}", ran.description_)) return {};
        cuStreamSynchronize(stream.get());
        cuMemcpyDtoH(&reference_loss, util::device_address(inputs.loss_out.data()), sizeof(util::f32_t));
        std::vector<nk_bf16_t> narrowed(vectors.size());
        cuMemcpyDtoH(narrowed.data(), util::device_address(inputs.token_vectors_grad.data()),
                     narrowed.size() * sizeof(nk_bf16_t));
        if (!expect_true(widen_from_bf16(narrowed, reference_gradient), "gradient widening failed")) return {};
        cuCtxPopCurrent(nullptr);
    }

    util::per_gpu_array<util::gpu_stream_t> streams;
    util::per_gpu_array<util::gpu_arena_t> arenas;
    device_tokens_inputs_t inputs[util::max_gpus_per_host_k];
    for (std::size_t rank = 0; rank != gpus; ++rank) {
        auto [rank_stream, make_rank_stream_status] = util::gpu_stream_t::make(devices[rank]);
        if (!expect_true(
                make_rank_stream_status && streams.push_back(std::move(rank_stream)) == util::status_code_t::success_k,
                "distributed stream make"))
            return {};
        cuCtxPushCurrent(streams[rank].context());
        auto [rank_arena, make_rank_arena_status] = util::gpu_arena_t::make(devices[rank], 1ull << 30, {devices, gpus});
        if (!expect_true(
                make_rank_arena_status && arenas.push_back(std::move(rank_arena)) == util::status_code_t::success_k,
                "distributed arena make"))
            return {};
        auto [rank_inputs, status] = stage_tokens_inputs(arenas[rank], staging, rank, 1, host, world_segments,
                                                         streams[rank]);
        if (!expect_true(status, "rank {} inputs: {}", rank, status.description_)) return {};
        inputs[rank] = std::move(rank_inputs);
        cuCtxPopCurrent(nullptr);
    }
    contrastive_tokens_config_t const config = tokens_config_(1, gpus);
    auto [topology, bootstrap_status] = util::mesh_topology_t::bootstrap(
        0, 1, &pause_between_polls, 0, "/tmp/uform_exchange_local", "tokens_distributed_case");
    if (!expect_true(bootstrap_status, "single-domain bootstrap: {}", bootstrap_status.description_)) return {};
    auto [domain, make_domain_status] = util::multimem_domain_t::make(
        topology, {devices, gpus}, distributed_contrastive_tokens_t::payload_scalars_needed(config), streams,
        wait_limit_k.count());
    if (!expect_true(make_domain_status, "multimem domain make: {}", make_domain_status.description_)) return {};
    auto [colocated, make_colocated_status] = util::colocated_domain_t::make(topology);
    if (!expect_true(make_colocated_status, "colocated domain make: {}", make_colocated_status.description_)) return {};
    auto [distributed, make_distributed_status] = distributed_contrastive_tokens_t::make(domain, colocated, arenas,
                                                                                         streams, config, numerics);
    if (!expect_true(make_distributed_status, "distributed make: {}", make_distributed_status.description_)) return {};
    gpu_tokens_inputs_t step_inputs[util::max_gpus_per_host_k];
    for (std::size_t rank = 0; rank != gpus; ++rank)
        step_inputs[rank] = {inputs[rank].token_vectors,
                             inputs[rank].token_offsets,
                             inputs[rank].batch,
                             inputs[rank].relations.view(),
                             recipe,
                             inputs[rank].token_vectors_grad};
    if (util::expected_t ran = distributed({step_inputs, gpus}, streams.span());
        !expect_true(ran, "distributed run: {}", ran.description_))
        return {};
    std::size_t const live_scalars = layout_t::live_rows() * layout_t::width_k;
    f64r2_view_t const reference_rows {reference_gradient.data(), gpus, live_scalars};
    util::f64_t worst = 0, largest = 0;
    util::f32_t world_loss = 0;
    std::vector<nk_bf16_t> narrowed(rank_scalars);
    for (std::size_t rank = 0; rank != gpus; ++rank) {
        cuCtxPushCurrent(streams[rank].context());
        cuStreamSynchronize(streams[rank].get());
        if (rank == 0)
            cuMemcpyDtoH(&world_loss, util::device_address(distributed.world_loss(0).data()), sizeof(util::f32_t));
        cuMemcpyDtoH(narrowed.data(), util::device_address(inputs[rank].token_vectors_grad.data()),
                     rank_scalars * sizeof(nk_bf16_t));
        cuCtxPopCurrent(nullptr);
        if (!expect_true(widen_from_bf16(narrowed, widened), "gradient widening failed")) return {};
        for (std::size_t index = 0; index != live_scalars; ++index) {
            util::f64_t const expected = reference_rows[rank, index];
            largest = fu::max_of(largest, std::fabs(expected));
            worst = fu::max_of(worst, std::fabs(widened[index] - expected));
        }
    }
    util::f64_t const relative = std::fabs(util::f64_t(world_loss) - reference_loss) /
                                 fu::max_of(std::fabs(util::f64_t(reference_loss)), 1.0e-9);
    expect_true(relative < 1.0e-4, "{}-GPU tokens loss {:.6f} matches the single-GPU {:.6f}: relative {:.2e}", gpus,
                world_loss, reference_loss, relative);
    expect_true(worst <= 1.0e-2 * largest, "{}-GPU token forces match the single-GPU: worst {:.3g} of {:.3g}", gpus,
                worst, largest);
    return {};
}

/** The tokens loss across two multimem domains, one process each, against the single-GPU loss over
 *  the same world on this process's first GPU: the world loss and this process's token forces.
 *  Runs when @c UFORM_HOST_DOMAIN_INDEX names the process's domain. */
static case_result_t test_tokens_cross_domain() {
#if !UFORM_WITH_INFINIBAND
    return case_result_t::skipped("built without UFORM_WITH_INFINIBAND");
#else
    using layout_t = host_tokens_layout_t;
    if (!cuda_devices_visible_(1)) return case_result_t::skipped(no_gpu_device_k);
    int device_count = 0;
    cuDeviceGetCount(&device_count);
    std::size_t const gpus = fu::min_of<std::size_t>(static_cast<std::size_t>(device_count), util::max_gpus_per_host_k);
    util::u32_t const home_domain = util::u32_t(std::atoi(std::getenv("UFORM_HOST_DOMAIN_INDEX")));
    // A job of its own, since the segments case before it rendezvoused under the launcher's job
    // in this process.
    char const* const launcher_job = std::getenv("UFORM_JOB_ID");
    sz::string_t job;
    if (!expect_true(util::format_into(job, "{}-tokens", launcher_job != nullptr ? launcher_job : "cross_domain"),
                     "job name formatting"))
        return {};
    auto [topology, bootstrap_status] = util::mesh_topology_t::bootstrap(home_domain, 2, &pause_between_polls,
                                                                         util::u32_t(gpus), {}, job.view());
    if (!expect_true(bootstrap_status, "bootstrap: {}", bootstrap_status.description_)) return {};
    std::size_t const world_gpus = 2u * gpus, rank_scalars = layout_t::head_rows_k * layout_t::width_k;
    std::size_t const live_scalars = layout_t::live_rows() * layout_t::width_k;
    st::basic_vector<util::f64_t> vectors, reference_gradient, widened;
    if (!expect_true(!st::failed(vectors.resize(world_gpus * rank_scalars)) &&
                         !st::failed(reference_gradient.resize(world_gpus * live_scalars)) &&
                         !st::failed(widened.resize(rank_scalars)),
                     "host staging failed"))
        return {};
    std::vector<nk_bf16_t> staging(vectors.size());
    for (std::size_t index = 0; index != vectors.size(); ++index) vectors[index] = hash_fill(index, 37);
    if (!expect_true(narrow_to_bf16(vectors, staging), "vector rounding failed")) return {};
    nk_capability_t capabilities = 0;
    if (!expect_true(nk_capabilities_enabled_cuda(0, &capabilities) == nk_success_k, "capability query failed"))
        return {};
    util::gpu_numerics_t const numerics = ut::gpu_numerics_of(capabilities);
    host_relations_t host;
    for (std::size_t world_gpu = 0; world_gpu != world_gpus; ++world_gpu)
        append_tokens_relations(host, world_gpu, world_gpus);
    std::size_t const world_segments = world_gpus * layout_t::segments_k;
    contrastive_recipe_t recipe;
    recipe.inverse_temperature = 4.0f;
    util::i32_t devices[util::max_gpus_per_host_k] = {};
    for (std::size_t rank = 0; rank != gpus; ++rank) devices[rank] = static_cast<util::i32_t>(rank);

    // The single-GPU reference over the whole world on this process's first device.
    util::f32_t reference_loss = 0;
    {
        auto [stream, make_stream_status] = util::gpu_stream_t::make(0);
        if (!expect_true(make_stream_status, "stream: {}", make_stream_status.description_)) return {};
        cuCtxPushCurrent(stream.context());
        auto [arena, make_arena_status] = util::gpu_arena_t::make(0, 1ull << 30);
        if (!expect_true(make_arena_status, "arena: {}", make_arena_status.description_)) return {};
        auto [inputs, stage_status] = stage_tokens_inputs(arena, staging, 0, world_gpus, host, world_segments, stream);
        auto [loss, make_loss_status] = stage_status
                                            ? contrastive_tokens_t::make(arena, tokens_config_(world_gpus, 1), numerics)
                                            : util::expected<contrastive_tokens_t>::fail(std::move(stage_status));
        util::expected_t ran = make_loss_status ? loss(inputs.token_vectors, inputs.token_offsets, inputs.batch,
                                                       inputs.relations.view(), recipe, inputs.loss_out,
                                                       inputs.token_vectors_grad, stream)
                                                : util::expected_t::fail(std::move(make_loss_status));
        if (!expect_true(ran, "reference run: {}", ran.description_)) return {};
        cuStreamSynchronize(stream.get());
        cuMemcpyDtoH(&reference_loss, util::device_address(inputs.loss_out.data()), sizeof(util::f32_t));
        std::vector<nk_bf16_t> narrowed(reference_gradient.size());
        cuMemcpyDtoH(narrowed.data(), util::device_address(inputs.token_vectors_grad.data()),
                     narrowed.size() * sizeof(nk_bf16_t));
        if (!expect_true(widen_from_bf16(narrowed, reference_gradient), "gradient widening failed")) return {};
        cuCtxPopCurrent(nullptr);
    }

    util::per_gpu_array<util::gpu_stream_t> streams;
    util::per_gpu_array<util::gpu_arena_t> arenas;
    device_tokens_inputs_t inputs[util::max_gpus_per_host_k];
    for (std::size_t rank = 0; rank != gpus; ++rank) {
        auto [rank_stream, make_rank_stream_status] = util::gpu_stream_t::make(devices[rank]);
        if (!expect_true(
                make_rank_stream_status && streams.push_back(std::move(rank_stream)) == util::status_code_t::success_k,
                "cross-domain stream make on GPU {}", rank))
            return {};
        cuCtxPushCurrent(streams[rank].context());
        auto [rank_arena, make_rank_arena_status] = util::gpu_arena_t::make(devices[rank], 1ull << 30, {devices, gpus});
        if (!expect_true(
                make_rank_arena_status && arenas.push_back(std::move(rank_arena)) == util::status_code_t::success_k,
                "cross-domain arena make on GPU {}", rank))
            return {};
        auto [rank_inputs, status] = stage_tokens_inputs(arenas[rank], staging, home_domain * gpus + rank, 1, host,
                                                         world_segments, streams[rank]);
        if (!expect_true(status, "rank {} inputs: {}", rank, status.description_)) return {};
        inputs[rank] = std::move(rank_inputs);
        cuCtxPopCurrent(nullptr);
    }
    contrastive_tokens_config_t const config = tokens_config_(1, world_gpus);
    auto [multimem, make_multimem_status] = util::multimem_domain_t::make(
        topology, {devices, gpus}, distributed_contrastive_tokens_t::payload_scalars_needed(config), streams,
        wait_limit_k.count());
    if (!expect_true(make_multimem_status, "multimem domain make: {}", make_multimem_status.description_)) return {};
    auto [colocated, make_colocated_status] = util::colocated_domain_t::make(topology);
    if (!expect_true(make_colocated_status, "colocated domain make: {}", make_colocated_status.description_)) return {};
    auto [distributed, make_distributed_status] = distributed_contrastive_tokens_t::make(multimem, colocated, arenas,
                                                                                         streams, config, numerics);
    if (!expect_true(make_distributed_status, "cross-domain make: {}", make_distributed_status.description_)) return {};
    gpu_tokens_inputs_t step_inputs[util::max_gpus_per_host_k];
    for (std::size_t rank = 0; rank != gpus; ++rank)
        step_inputs[rank] = {inputs[rank].token_vectors,
                             inputs[rank].token_offsets,
                             inputs[rank].batch,
                             inputs[rank].relations.view(),
                             recipe,
                             inputs[rank].token_vectors_grad};
    // Two steps over the same inputs enqueued back to back, with no host wait between them, so a
    // gather of the second landing on blocks the first still reads would show as a mismatch.
    f64r2_view_t const reference_rows {reference_gradient.data(), world_gpus, live_scalars};
    std::vector<nk_bf16_t> narrowed(rank_scalars);
    for (util::u32_t step = 0; step != 2u; ++step) {
        if (util::expected_t ran = distributed({step_inputs, gpus}, streams.span());
            !expect_true(ran, "cross-domain step {}: {}", step, ran.description_))
            return {};
        if (step == 0u) continue;
        util::f64_t worst = 0, largest = 0;
        util::f32_t world_loss = 0;
        for (std::size_t rank = 0; rank != gpus; ++rank) {
            cuCtxPushCurrent(streams[rank].context());
            CUresult const synchronized = cuStreamSynchronize(streams[rank].get());
            if (rank == 0)
                cuMemcpyDtoH(&world_loss, util::device_address(distributed.world_loss(0).data()), sizeof(util::f32_t));
            cuMemcpyDtoH(narrowed.data(), util::device_address(inputs[rank].token_vectors_grad.data()),
                         rank_scalars * sizeof(nk_bf16_t));
            cuCtxPopCurrent(nullptr);
            if (!expect_true(synchronized == CUDA_SUCCESS, "cross-domain sync on GPU {}: result {}", rank,
                             static_cast<int>(synchronized)))
                return {};
            if (!expect_true(widen_from_bf16(narrowed, widened), "gradient widening failed")) return {};
            std::size_t const world_gpu = home_domain * gpus + rank;
            for (std::size_t index = 0; index != live_scalars; ++index) {
                util::f64_t const expected = reference_rows[world_gpu, index];
                largest = fu::max_of(largest, std::fabs(expected));
                worst = fu::max_of(worst, std::fabs(widened[index] - expected));
            }
        }
        util::f64_t const relative = std::fabs(util::f64_t(world_loss) - reference_loss) /
                                     fu::max_of(std::fabs(util::f64_t(reference_loss)), 1.0e-9);
        expect_true(relative < 1.0e-4,
                    "step {}: 2-domain tokens loss {:.6f} matches the single-GPU {:.6f}: relative {:.2e}", step,
                    world_loss, reference_loss, relative);
        expect_true(worst <= 1.0e-2 * largest,
                    "step {}: 2-domain token forces match the single-GPU: worst {:.3g} of {:.3g}", step, worst,
                    largest);
    }
    util::expected_t closed = topology.barrier("tokens_done", wait_limit_k.count());
    expect_true(static_cast<bool>(closed), "the closing barrier should complete: {}", closed.description_);
#endif
    return {};
}

/** Late interaction through the items source the C interface steps: two text pairs whose anchors
 *  query and whose positives are documents, so the token head scores each query against the
 *  documents and the step reports a finite, nonzero late-interaction term. The source copies every
 *  item's side into the batch, which it once left unset. */
static case_result_t test_items_sides() {
    int device_count = 0;
    if (cuDeviceGetCount(&device_count) != CUDA_SUCCESS || device_count < 1)
        return case_result_t::skipped(no_gpu_device_k);
    st::basic_vector<util::i32_t> devices;
    for (int device = 0; device != device_count; ++device)
        if (!expect_true(!st::failed(devices.push_back(static_cast<util::i32_t>(device))), "device table")) return {};
    ut::mesh_options_t options;
    if (!expect_true(nk_capabilities_enabled_cuda(0, &options.capabilities) == nk_success_k, "capability query failed"))
        return {};
    ut::mesh_t mesh;
    if (util::expected_t joined = ut::mesh_t::make(mesh, options, devices);
        !expect_true(joined, "mesh: {}", joined.description_))
        return {};

    ut::trainer_config_t config;
    config.batch = multimodal_batch_shape_t {4u, 8u, 64u};
    config.batch.max_bags_per_gpu = 4u;
    config.batch.max_bag_segments = 2u;
    config.depth = 1;
    config.hidden = 256;
    config.steps = 1;
    config.seed = test_run.settings.seed;
    config.capture = util::eager_k;
    config.objective = ut::objective_t::contrastive_k;
    config.late_interaction.weight = 1.0f;
    config.late_interaction.width = 32;
    ut::trainer_t trainer;
    if (util::expected_t built = ut::trainer_t::make(trainer, mesh, config, devices,
                                                     ut::gpu_numerics_of(options.capabilities));
        !expect_true(built, "trainer: {}", built.description_))
        return {};
    ut::items_source_t source;
    if (util::expected_t made = ut::items_source_t::make(source, trainer);
        !expect_true(made, "items source: {}", made.description_))
        return {};

    char const text[] = "which cat sleeps on the mat?"               //
                        "the grey cat sleeps on the mat by the door" //
                        "how far is the moon?"                       //
                        "the moon is three hundred thousand kilometres away";
    std::size_t const offsets[] = {0, 28, 70, 90, sizeof(text) - 1};
    modality_t const modalities[] = {modality_t::text_k, modality_t::text_k, modality_t::text_k, modality_t::text_k};
    util::u32_t const bags[] = {0, 0, 1, 1};
    std::size_t const attract_offsets[] = {0, 1, 2, 3, 4};
    util::u32_t const attract_ids[] = {1, 0, 3, 2};
    late_interaction_side_t const sides[] = {late_interaction_side_t::query_k, late_interaction_side_t::document_k,
                                             late_interaction_side_t::query_k, late_interaction_side_t::document_k};
    source.items.modalities = modalities;
    source.items.contents = {std::as_bytes(std::span {text}), offsets};
    source.items.bags = bags;
    source.items.attract = {{attract_ids, attract_offsets}, {}};
    source.items.sides = sides;
    ut::step_report_t report;
    if (util::expected_t stepped = trainer.step(ut::batch_source_of(source), util::optimizer_step_t {0}, report);
        !expect_true(stepped, "step: {}", stepped.description_))
        return {};
    ut::telemetry_t const telemetry = trainer.telemetry(util::optimizer_step_t {0});
    expect_true(std::isfinite(telemetry.late_interaction) && telemetry.late_interaction != 0.0f,
                "the items' query sides reach late interaction: its term is {}", telemetry.late_interaction);
    return {};
}

int main(int argc, char** argv) {
    // Lazy loading would strand the one host thread still owing other members' barrier arrivals.
    ::setenv("CUDA_MODULE_LOADING", "EAGER", 1);
    start_test_run(argc, argv);
    [[maybe_unused]] CUresult const initialized = cuInit(0); // a failure leaves no device to count, so each case skips
    if (std::getenv("UFORM_HOST_DOMAIN_INDEX") != nullptr) {
        run_test_case("test_cross_domain", test_cross_domain);
        run_test_case("test_tokens_cross_domain", test_tokens_cross_domain);
        return finish_test_run();
    }

    run_test_case("test_pool_parity", test_pool_parity);
    using graph_t = oracle_graph_t;
    constexpr contrastive_pair_loss_form_t balanced_k = contrastive_pair_loss_form_t::balanced_k;
    constexpr contrastive_pair_loss_form_t ranked_k = contrastive_pair_loss_form_t::ranked_k;
    constexpr contrastive_pair_loss_form_t sigmoid_k = contrastive_pair_loss_form_t::sigmoid_k;
    run_test_case("test_oracle(8, 64, {64}, graph_t::self_edges_k, false, 1.0f, balanced_k, 1.0f, 0.0f)", test_oracle,
                  8, 64, std::vector<util::u32_t> {64}, graph_t::self_edges_k, false, 1.0f, balanced_k, 1.0f, 0.0f);
    run_test_case("test_oracle(8, 64, {32, 64}, graph_t::self_edges_k, false, 1.0f, balanced_k, 1.0f, 0.0f)",
                  test_oracle, 8, 64, std::vector<util::u32_t> {32, 64}, graph_t::self_edges_k, false, 1.0f, balanced_k,
                  1.0f, 0.0f);
    run_test_case("test_oracle(8, 64, {64}, graph_t::explicit_edges_k, false, 0.5f, balanced_k, 1.0f, 0.0f)",
                  test_oracle, 8, 64, std::vector<util::u32_t> {64}, graph_t::explicit_edges_k, false, 0.5f, balanced_k,
                  1.0f, 0.0f);
    run_test_case("test_oracle(10, 64, {32, 64}, graph_t::roles_k, false, 1.0f, balanced_k, 1.0f, 0.0f)", test_oracle,
                  10, 64, std::vector<util::u32_t> {32, 64}, graph_t::roles_k, false, 1.0f, balanced_k, 1.0f, 0.0f);
    run_test_case("test_oracle(10, 64, {32, 64}, graph_t::roles_k, false, 1.0f, ranked_k, 1.0f, 0.0f)", test_oracle, 10,
                  64, std::vector<util::u32_t> {32, 64}, graph_t::roles_k, false, 1.0f, ranked_k, 1.0f, 0.0f);
    run_test_case("test_oracle(10, 64, {32, 64}, graph_t::roles_k, false, 1.0f, sigmoid_k, 1.0f, 0.0f)", test_oracle,
                  10, 64, std::vector<util::u32_t> {32, 64}, graph_t::roles_k, false, 1.0f, sigmoid_k, 1.0f, 0.0f);
    run_test_case("test_oracle(8, 64, {64}, graph_t::self_edges_k, false, 1.0f, ranked_k, 1.0f, 0.0f)", test_oracle, 8,
                  64, std::vector<util::u32_t> {64}, graph_t::self_edges_k, false, 1.0f, ranked_k, 1.0f, 0.0f);
    run_test_case("test_oracle(8, 64, {64}, graph_t::self_edges_k, true, 1.0f, balanced_k, 1.0f, 0.0f)", test_oracle, 8,
                  64, std::vector<util::u32_t> {64}, graph_t::self_edges_k, true, 1.0f, balanced_k, 1.0f, 0.0f);
    run_test_case("test_oracle(8, 64, {64}, graph_t::self_edges_k, false, 0.0f, balanced_k, 1.0f, 0.0f)", test_oracle,
                  8, 64, std::vector<util::u32_t> {64}, graph_t::self_edges_k, false, 0.0f, balanced_k, 1.0f, 0.0f);
    run_test_case("test_oracle(33, 256, {32, 256}, graph_t::self_edges_k, false, 1.0f, balanced_k, 1.0f, 0.0f)",
                  test_oracle, 33, 256, std::vector<util::u32_t> {32, 256}, graph_t::self_edges_k, false, 1.0f,
                  balanced_k, 1.0f, 0.0f);
    run_test_case("test_oracle(64, 1024, {1024}, graph_t::self_edges_k, false, 1.0f, sigmoid_k, 10.0f, -10.0f)",
                  test_oracle, 64, 1024, std::vector<util::u32_t> {1024}, graph_t::self_edges_k, false, 1.0f, sigmoid_k,
                  10.0f, -10.0f);
    run_test_case("test_oracle(64, 1024, {1024}, graph_t::self_edges_k, false, 1.0f, ranked_k, 10.0f, -10.0f)",
                  test_oracle, 64, 1024, std::vector<util::u32_t> {1024}, graph_t::self_edges_k, false, 1.0f, ranked_k,
                  10.0f, -10.0f);
    run_test_case("test_negatives", test_negatives);
    run_test_case("test_sharded_oracle(graph_t::self_edges_k, balanced_k)", test_sharded_oracle, graph_t::self_edges_k,
                  balanced_k);
    run_test_case("test_sharded_oracle(graph_t::roles_k, balanced_k)", test_sharded_oracle, graph_t::roles_k,
                  balanced_k);
    run_test_case("test_sharded_oracle(graph_t::roles_k, ranked_k)", test_sharded_oracle, graph_t::roles_k, ranked_k);
    run_test_case("test_sharded_oracle(graph_t::roles_k, sigmoid_k)", test_sharded_oracle, graph_t::roles_k, sigmoid_k);
    run_test_case("test_distributed", test_distributed);
    run_test_case("test_tokens_oracle", test_tokens_oracle);
    run_test_case("test_tokens_distributed", test_tokens_distributed);
    run_test_case("test_items_sides", test_items_sides);
    return finish_test_run();
}
