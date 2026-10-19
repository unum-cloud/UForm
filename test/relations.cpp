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
 */
#include "core/mesh.hpp"                  // `mesh_shape_t`, `mesh_topology_t`, `make_mesh_shape`
#include "core/contrastive_relations.cuh" // `relate_bags`
#include "harness.hpp"

using namespace unum::uform::core;
using namespace unum::uform::test;

/** Segments per GPU, and the most GPUs a simulated world spans. */
static constexpr u32_t segments_k = 12u, max_world_gpus_k = 16u;

/** The cap of attract edges one GPU's bags name: a segment in a bag of three attracts two. */
static constexpr u32_t max_edges_k = 2u * segments_k;

/** One GPU's attract rows by global vertex ID. */
struct gpu_relations_t {
    u32_t offsets[segments_k + 1] = {};
    u32_t ids[max_edges_k] = {};
    f32_t weights[max_edges_k] = {};
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

/** Build the relations of every GPU @p topology drives into @p relations , indexed by world GPU. */
[[nodiscard]] static bool relate_process_(mesh_topology_t const &topology, world_columns_t const &world,
                                          gpu_relations_t *relations) noexcept {
    for (u32_t member = 0; member != topology.shape.gpus_per_host_domain; ++member) {
        u32_t const gpu = topology.first_gpu_index() + member;
        gpu_relations_t &into = relations[gpu];
        expected<u32_t> written = relate_bags(
            u32r1_view_t {world.bags[gpu], segments_k}, u32r1_view_t {world.lengths[gpu], segments_k}, gpu * segments_k,
            u32r1_span_t {into.offsets, segments_k + 1}, u32r1_span_t {into.ids, max_edges_k},
            f32r1_span_t {into.weights, max_edges_k});
        if (!written) return false;
        into.edges = *written;
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

int main(int, char **argv) {
    test_suite_t suite = test_suite_t::make(argv[0]);
    suite.run("world/1-1-1", slices_match_world, 1u, 1u, 1u);
    suite.run("world/2-8-16", slices_match_world, 2u, 8u, 16u);
    suite.run("world/4-8-16", slices_match_world, 4u, 8u, 16u);
    return suite.finish();
}
