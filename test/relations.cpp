/**
 *  @file test/relations.cpp
 *  @author Ash Vardanian
 *  @date October 19, 2026
 *  @brief Conformance test of the world relations: every process of a simulated mesh builds its
 *      GPUs' bag relations over the global vertex axis, and every slice must equal the
 *      single-process construction of the whole world. Host-only, no GPU.
 *
 *  One case per mesh, @c {1,1,1} , @c {2,8,16} and @c {4,8,16} , each over keyed bags with PAD
 *  rows: a process numbers its vertices from its first GPU, so a partner's ID lands on the GPU
 *  that holds it, inside the world axis the loss reads, and the attract graph stays symmetric.
 *  Two more cases pin the role rules of one bag and the edge capacity those rules derive.
 */
#include "uform/train/contrastive_relations.cuh" // `relate_roles`, `contrastive_edges`
#include "uform/utilities/mesh.hpp"              // `mesh_shape_t`, `mesh_topology_t`, `make_mesh_shape`

#include "harness.hpp"

using namespace unum::uform::train;
using namespace unum::uform::utilities;
using namespace unum::uform::test;

/** Segments per GPU, and the most GPUs a simulated world spans. */
static constexpr u32_t segments_k = 12u, max_world_gpus_k = 16u;

/** The cap of attract edges one GPU's bags name: a segment in a bag of three attracts two. */
static constexpr u32_t max_edges_k = 2u * segments_k;

/** The host-only edge owner the cases build into. */
using host_edges_t = contrastive_edges<nk::aligned_allocator<b8_t>>;

/** One GPU's attract rows by global vertex ID. */
struct gpu_relations_t {
    u32_t offsets[segments_k + 1] = {};
    u32_t ids[max_edges_k] = {};
    u32_t edges = 0;
};

/** The world's keyed mixed columns: runs of one to three segments per bag, some segments PAD,
 *  and an unclaimed PAD tail tagged with no bag. */
struct world_columns_t {
    u32_t bags[max_world_gpus_k][segments_k] = {};
    u32_t lengths[max_world_gpus_k][segments_k] = {};
};

[[nodiscard]] static u64_t mix_(u64_t seed, u64_t value) noexcept {
    u64_t mixed = seed ^ (value * 0x9e3779b97f4a7c15ull);
    mixed = (mixed ^ (mixed >> 30)) * 0xbf58476d1ce4e5b9ull;
    mixed = (mixed ^ (mixed >> 27)) * 0x94d049bb133111ebull;
    return mixed ^ (mixed >> 31);
}

static void fill_world_(world_columns_t &world, u32_t world_gpus, u64_t seed) noexcept {
    for (u32_t gpu = 0; gpu != world_gpus; ++gpu) {
        u32_t segment = 0, bag = 0;
        u32_t const tail = segments_k - static_cast<u32_t>(mix_(seed, gpu) % 3u);
        while (segment != tail) {
            u32_t const run = 1u + static_cast<u32_t>(mix_(seed, gpu * 1000u + segment) % 3u);
            for (u32_t member = 0; member != run && segment != tail; ++member, ++segment) {
                world.bags[gpu][segment] = bag;
                world.lengths[gpu][segment] = mix_(seed, gpu * 7919u + segment) % 5u == 0u ? 0u : 3u;
            }
            ++bag;
        }
        for (; segment != segments_k; ++segment) {
            world.bags[gpu][segment] = ~0u;
            world.lengths[gpu][segment] = 0u;
        }
    }
}

/** Build the relations of every GPU @p topology drives into @p relations , indexed by world GPU:
 *  every bag's first segment anchors it and the rest are its positives. */
[[nodiscard]] static bool relate_process_(mesh_topology_t const &topology, world_columns_t const &world,
                                          gpu_relations_t *relations) noexcept {
    expected<host_edges_t> made = host_edges_t::make(segments_k, max_edges_k);
    if (!made) return false;
    host_edges_t &edges = made.value_;
    for (u32_t member = 0; member != topology.shape.gpus_per_host_domain; ++member) {
        u32_t const gpu = topology.first_gpu_index() + member;
        segment_role_t roles[segments_k];
        for (u32_t segment = 0; segment != segments_k; ++segment)
            roles[segment] = segment != 0 && world.bags[gpu][segment - 1] == world.bags[gpu][segment]
                                 ? segment_role_t::positive_k
                                 : segment_role_t::anchor_k;
        expected_t related = relate_roles(u32r1_view_t {world.bags[gpu], segments_k}, {roles, segments_k},
                                          u32r1_view_t {world.lengths[gpu], segments_k}, {}, gpu * segments_k, edges);
        if (!related || edges.repel().ids.extent(0) != 0 || edges.ignore().ids.extent(0) != 0) return false;
        contrastive_edges_view_t const attract = edges.attract();
        gpu_relations_t &into = relations[gpu];
        into.edges = static_cast<u32_t>(attract.ids.extent(0));
        for (u32_t row = 0; row != segments_k + 1; ++row) into.offsets[row] = attract.offsets[row];
        for (u32_t edge = 0; edge != into.edges; ++edge) into.ids[edge] = attract.ids[edge];
    }
    return true;
}

/** Every process of the @p host , @p multimem , @p colocated mesh against the whole world. */
static void slices_match_world(test_suite_t &suite, u32_t host, u32_t multimem, u32_t colocated) {
    expected<mesh_shape_t> shape = make_mesh_shape(host, multimem, colocated);
    if (!shape) return suite.check(false, "mesh shape: {}", shape.description_);
    world_columns_t world;
    fill_world_(world, colocated, suite.settings.seed);

    mesh_topology_t whole;
    whole.shape = mesh_shape_t {colocated, colocated, colocated};
    gpu_relations_t global[max_world_gpus_k], sliced[max_world_gpus_k];
    suite.check(relate_process_(whole, world, global), "the single-process world should relate");
    for (u32_t process = 0; process != shape->host_domain_count(); ++process) {
        mesh_topology_t topology;
        topology.shape = *shape;
        topology.host_domain_index = process;
        suite.check(relate_process_(topology, world, sliced), "process {} should relate", process);
    }

    u32_t const world_vertices = colocated * segments_k;
    for (u32_t gpu = 0; gpu != colocated; ++gpu) {
        gpu_relations_t const &expected_rows = global[gpu], &rows = sliced[gpu];
        bool same = rows.edges == expected_rows.edges;
        for (u32_t row = 0; row != segments_k + 1 && same; ++row)
            same = rows.offsets[row] == expected_rows.offsets[row];
        for (u32_t edge = 0; edge != rows.edges && same; ++edge) same = rows.ids[edge] == expected_rows.ids[edge];
        suite.check(same, "GPU {}'s slice should equal the world construction", gpu);
        for (u32_t row = 0; row != segments_k; ++row)
            for (u32_t edge = rows.offsets[row]; edge != rows.offsets[row + 1]; ++edge) {
                u32_t const partner = rows.ids[edge];
                suite.check(partner < world_vertices && partner / segments_k == gpu,
                            "GPU {} row {} names vertex {}, off its own GPU's run of the {}-vertex world", gpu, row,
                            partner, world_vertices);
                bool mirrored = false;
                gpu_relations_t const &back = sliced[partner / segments_k];
                u32_t const partner_row = partner % segments_k;
                for (u32_t back_edge = back.offsets[partner_row]; back_edge != back.offsets[partner_row + 1];
                     ++back_edge)
                    mirrored = mirrored || back.ids[back_edge] == gpu * segments_k + row;
                suite.check(mirrored, "vertex {} attracts {} but not back", gpu * segments_k + row, partner);
                suite.check(world.lengths[gpu][row] != 0u && world.lengths[partner / segments_k][partner_row] != 0u,
                            "PAD vertex {} or {} carries an edge", gpu * segments_k + row, partner);
            }
    }
}

/** The run of @p row in @p set as one comparable list of global IDs and weights. */
[[nodiscard]] static bool row_is_(contrastive_edges_view_t set, u32_t row, std::initializer_list<u32_t> ids,
                                  std::initializer_list<f32_t> weights) noexcept {
    u32_t const begin = set.offsets[row], end = set.offsets[row + 1];
    if (end - begin != ids.size()) return false;
    u32_t const *const expected_ids = ids.begin();
    f32_t const *const expected_weights = weights.begin();
    for (u32_t index = 0; index != ids.size(); ++index) {
        if (set.ids[begin + index] != expected_ids[index]) return false;
        if (weights.size() != 0 && set.weights[begin + index] != expected_weights[index]) return false;
    }
    return true;
}

/** One bag of an anchor, two positives and two weighted negatives, then a PAD and a singleton, from
 *  vertex 100: the anchor and positives attract each other, the anchor and each negative repel each
 *  other at the negative's weight, every other pair inside the bag is ignored, and no vertex ever
 *  names itself. */
static void bag_roles(test_suite_t &suite) {
    constexpr u32_t segments = 7, base = 100;
    using role_t = segment_role_t;
    u32_t const bags[segments] = {0, 0, 0, 0, 0, 1, 2};
    u32_t const lengths[segments] = {4, 3, 2, 5, 1, 0, 6};
    role_t const roles[segments] = {role_t::anchor_k,   role_t::positive_k, role_t::positive_k, role_t::negative_k,
                                    role_t::negative_k, role_t::anchor_k,   role_t::anchor_k};
    f32_t const weights[segments] = {1.0f, 1.0f, 1.0f, 2.0f, 0.5f, 1.0f, 1.0f};
    expected<host_edges_t> made = host_edges_t::make(segments, 16);
    if (!made) return suite.check(false, "edges: {}", made.description_);
    host_edges_t &edges = made.value_;
    expected_t related = relate_roles(u32r1_view_t {bags, segments}, {roles, segments},
                                      u32r1_view_t {lengths, segments}, f32r1_view_t {weights, segments}, base, edges);
    if (!related) return suite.check(false, "relate: {}", related.description_);
    contrastive_edges_view_t const attract = edges.attract(), repel = edges.repel(), ignore = edges.ignore();
    suite.check(row_is_(attract, 0, {101, 102}, {1, 1}) && row_is_(repel, 0, {103, 104}, {2.0f, 0.5f}) &&
                    row_is_(ignore, 0, {}, {}),
                "the anchor attracts its positives and repels its negatives at their weights");
    suite.check(
        row_is_(attract, 1, {100, 102}, {1, 1}) && row_is_(repel, 1, {}, {}) && row_is_(ignore, 1, {103, 104}, {}),
        "a positive attracts the anchor and the other positive and ignores the negatives");
    suite.check(
        row_is_(attract, 3, {}, {}) && row_is_(repel, 3, {100}, {2.0f}) && row_is_(ignore, 3, {101, 102, 104}, {}),
        "a negative repels the anchor at its own weight and ignores the rest of its bag");
    bool const inert = row_is_(attract, 5, {}, {}) && row_is_(repel, 5, {}, {}) && row_is_(ignore, 5, {}, {}) &&
                       row_is_(attract, 6, {}, {}) && row_is_(repel, 6, {}, {}) && row_is_(ignore, 6, {}, {});
    suite.check(inert, "a PAD vertex and a singleton bag name no edge");
}

/** The mixer's two bands: two pairs' media lead, their captions trail, then PAD, from vertex 100.
 *  Each caption still attracts its own pair's media across the bands, in ascending vertex order,
 *  and bags that descend twice are refused. */
static void bag_bands(test_suite_t &suite) {
    constexpr u32_t segments = 5, base = 100;
    using role_t = segment_role_t;
    u32_t const bags[segments] = {0, 1, 0, 1, 2};
    u32_t const lengths[segments] = {2048, 2048, 12, 9, 0};
    role_t const roles[segments] = {role_t::positive_k, role_t::positive_k, role_t::anchor_k, role_t::anchor_k,
                                    role_t::anchor_k};
    expected<host_edges_t> made = host_edges_t::make(segments, 8);
    if (!made) return suite.check(false, "edges: {}", made.description_);
    host_edges_t &edges = made.value_;
    expected_t related = relate_roles(u32r1_view_t {bags, segments}, {roles, segments},
                                      u32r1_view_t {lengths, segments}, {}, base, edges);
    if (!related) return suite.check(false, "relate: {}", related.description_);
    contrastive_edges_view_t const attract = edges.attract();
    suite.check(row_is_(attract, 0, {102}, {1}) && row_is_(attract, 1, {103}, {1}) && row_is_(attract, 2, {100}, {1}) &&
                    row_is_(attract, 3, {101}, {1}) && row_is_(attract, 4, {}, {}),
                "a bag split across the bands relates whole, and PAD names nothing");
    u32_t const twice[segments] = {0, 1, 0, 1, 0};
    suite.check(!relate_roles(u32r1_view_t {twice, segments}, {roles, segments}, u32r1_view_t {lengths, segments}, {},
                              base, edges),
                "bags descending twice are refused");
}

/** The derived capacity is exact: four bags of an anchor and two positives fill the attract set of
 *  @ref relation_edges_capacity to the last edge, and one edge less is refused loudly. */
static void bag_capacity(test_suite_t &suite) {
    constexpr u32_t segments = 12, bag_segments = 3;
    using role_t = segment_role_t;
    u32_t bags[segments], lengths[segments];
    role_t roles[segments];
    for (u32_t segment = 0; segment != segments; ++segment) {
        bags[segment] = segment / bag_segments, lengths[segment] = 2;
        roles[segment] = segment % bag_segments == 0 ? role_t::anchor_k : role_t::positive_k;
    }
    u32_t const capacity = relation_edges_capacity(segments, bag_segments);
    expected<host_edges_t> exact = host_edges_t::make(segments, capacity);
    expected<host_edges_t> short_by_one = host_edges_t::make(segments, capacity - 1u);
    if (!exact || !short_by_one) return suite.check(false, "edges: {}", exact.description_);
    expected_t related = relate_roles(u32r1_view_t {bags, segments}, {roles, segments},
                                      u32r1_view_t {lengths, segments}, {}, 0, exact.value_);
    suite.check(related && exact.value_.attract().ids.extent(0) == capacity,
                "bags of three fill the derived {} attract edges exactly", capacity);
    suite.check(!relate_roles(u32r1_view_t {bags, segments}, {roles, segments}, u32r1_view_t {lengths, segments}, {}, 0,
                              short_by_one.value_),
                "one edge short of the derived capacity is refused");
}

int main(int, char **argv) {
    test_suite_t suite = test_suite_t::make(argv[0]);
    suite.run("world/1-1-1", slices_match_world, 1u, 1u, 1u);
    suite.run("world/2-8-16", slices_match_world, 2u, 8u, 16u);
    suite.run("world/4-8-16", slices_match_world, 4u, 8u, 16u);
    suite.run("roles/bag", bag_roles);
    suite.run("roles/bands", bag_bands);
    suite.run("roles/capacity", bag_capacity);
    return suite.finish();
}
