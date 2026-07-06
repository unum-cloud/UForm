/**
 *  @brief Conformance test of the mesh bootstrap — the one job `mesh.hpp` exists for: minting each
 *      host_domain's facts and reconciling them over a shared filesystem into one agreed world.
 *  @file cuda_test/mesh.cpp
 *  @author Ash Vardanian
 *  @date July 5, 2026
 *
 *  Three cases, each proving an operation that must @b succeed — the integer accessors are one-line
 *  divisions and carry no test:
 *
 *      - @ref exchange_round_trips — the durable primitive every wiring phase reuses: `publish_text`
 *        writes atomically and `await_text` reads the bytes back verbatim. Hand-built topology, no GPU.
 *      - @ref single_box_world — the common one-box case: `bootstrap(0,1)` assembles the whole topology
 *        from the local probe and never opens the exchange directory.
 *      - @ref world_reconciles — the substance: a lead reads a peer's facts, validates agreement, and
 *        assembles the world; a follower adopts the lead's manifest. Both are driven against a @b matching
 *        simulated peer planted with this box's real GPU identity, so agreement genuinely passes. Needs a
 *        multicast + P2P capable GPU (bootstrap front-loads `discover_mesh_shape`); a box without one SKIPs.
 *
 *  The genuine multi-node happy path is @ref run_cluster_exchange , driven by the sibling
 *  `mesh-4-8-16.sbatch`: every rank bootstraps for real and must assemble a @b byte-identical world table.
 */
#include <cstdio>  // `std::fopen`, `std::fwrite`, `std::fclose` — planting a simulated peer's exchange files
#include <cstdlib> // `std::getenv`, `std::atoi`

#include <utility>      // `std::move`
#include <system_error> // `std::error_code` — the out-param every `std::filesystem` call reports through
#include <filesystem>   // `std::filesystem` — the temporary exchange directory

#include <unistd.h> // `::getpid`

#include <cuda.h> // `cuInit`, `cuDeviceGet`, `cuDeviceGetName`

#include "cuda_core/mesh.hpp"
#include "cuda_test/harness.hpp"

using namespace unum::uform::core;
using namespace unum::uform::test;
namespace fs = std::filesystem;

/** @brief One host_domain's facts in the exact line-oriented wire format `build_facts_` emits — what a
 *      peer publishes under its `mesh` key for the lead to read. */
static sz::string facts_payload(sz::string_view job, u32_t index, u32_t count, u32_t multimem, u32_t devices,
                                sz::string_view hostname, sz::string_view device_name) {
    sz::string text;
    (void)format_into(text,
                      "schema {}\njob {}\nhost_domain_index {}\nhost_domain_count {}\n" //
                      "multimem {}\nhostname {}\ndevices {}\ndevice_name {}\nuuids",
                      mesh_schema_k, job, index, count, multimem, hostname, devices, device_name);
    // Synthetic per-GPU UUIDs, distinct across host_domains and never colliding with a real GPU id, so the
    // lead's world-uniqueness check sees this faux peer's devices as its own disjoint set.
    for (u32_t gpu = 0; gpu != devices; ++gpu)
        (void)format_into(text, " fa11ba55fa11ba55fa11ba55{:08x}", (index << 16) | gpu);
    return text;
}

/** @brief A merged manifest in the wire format `build_manifest_` emits — what the lead publishes for every
 *      follower to adopt; every row carries @p device_name so a follower's self-row agrees with its probe. */
static sz::string manifest_payload(sz::string_view job, u32_t multimem, u32_t rows, u32_t devices,
                                   sz::string_view device_name) {
    sz::string text;
    (void)format_into(text, "schema {}\njob {}\nmultimem {}\n", mesh_schema_k, job, multimem);
    for (u32_t row = 0; row != rows; ++row)
        (void)format_into(text, "host_domain {} hostname faux-{} devices {} device_name {}\n", row, row, devices,
                          device_name);
    return text;
}

/** @brief Write @p content into @p path — one simulated peer's exchange file, planted before the bootstrap
 *      under test reads it. Uses stdio, never a `std::*stream`. */
static void plant(fs::path const &path, sz::string const &content) {
    std::FILE *file = std::fopen(path.c_str(), "wb");
    if (file == nullptr) return;
    (void)std::fwrite(content.data(), 1, content.size(), file);
    std::fclose(file);
}

/** @brief The device-0 model name with spaces replaced by underscores — exactly what
 *      `probe_local_host_domain_` records, so a planted peer's `device_name` matches this box's probe and
 *      the uniformity check passes. */
static bool probe_local_device_name(sz::string &name) {
    CUdevice device = 0;
    if (cuDeviceGet(&device, 0) != CUDA_SUCCESS) return false;
    char raw[64] = {};
    if (cuDeviceGetName(raw, sizeof(raw) - 1, device) != CUDA_SUCCESS) return false;
    for (char *character = raw; *character != '\0'; ++character)
        if (*character == ' ') *character = '_';
    name = raw;
    return true;
}

/** @brief The durable exchange round-trips a payload byte-for-byte: `publish_text` writes it atomically
 *      (temp + rename, no leftover), `await_text` reads it whole, and a second publish replaces the first —
 *      the primitive every wiring phase carries its coordinates over. Hand-built topology, no GPU. */
static void exchange_round_trips(test_suite_t &suite, fs::path const &base) {
    fs::path const directory = base / "exchange";
    std::error_code error;
    fs::create_directories(directory, error);

    mesh_topology_t topology;
    if (topology.host_domains.try_resize(2) != status_t::success_k) return suite.check(false, "topology table alloc");
    topology.host_domain_index = 0;
    (void)format_into(topology.exchange_directory, "{}", directory.c_str());

    sz::string_view const payload = "qpn 42\npsn 7\nrkey 3735928559\n";
    suite.check(static_cast<bool>(topology.publish_text("wiring", payload)), "publish_text should succeed");
    suite.check(!fs::exists(directory / "host_domain0.wiring.tmp"), "publish should leave no temporary file");
    sz::string received;
    expected_t awaited = topology.await_text(0, "wiring", received, 5.0);
    suite.check(static_cast<bool>(awaited), "await_text should find the payload: {}", awaited.description_);
    suite.check(received.view() == payload, "payload should round-trip verbatim: got {}", received);

    sz::string_view const replacement = "qpn 99\n";
    suite.check(static_cast<bool>(topology.publish_text("wiring", replacement)), "republish_text should succeed");
    sz::string again;
    (void)topology.await_text(0, "wiring", again, 5.0);
    suite.check(again.view() == replacement, "republish should atomically replace: got {}", again);
}

/** @brief The single-host_domain world: `bootstrap(0,1)` assembles the whole topology from the local probe
 *      and never opens the exchange directory — the common one-box case. */
static void single_box_world(test_suite_t &suite, fs::path const &base, u32_t local_devices) {
    sz::string const base_str {base.c_str()};
    auto result = mesh_topology_t::bootstrap(0, 1, local_devices, base_str.c_str(), "single_box", 10.0);
    if (!result) return suite.check(false, "single-box bootstrap should succeed: {}", result.description_);
    mesh_topology_t topology = std::move(*result);
    suite.check(topology.host_domain_count() == 1, "world should hold one host_domain: got {}",
                topology.host_domain_count());
    suite.check(topology.is_lead(), "the sole host_domain should lead");
    suite.check(topology.self().device_count == local_devices, "self facts should carry the probed device count: {}",
                topology.self().device_count);
    suite.check(topology.exchange_directory.empty(), "a single-box world should not touch the filesystem");
    suite.check(!fs::exists(base / "single_box"), "a single-box world should create no exchange files");
}

/** @brief The substance: a lead reads a peer's published facts, validates agreement, and assembles the
 *      world; a follower adopts the lead's manifest. Both halves run against a matching simulated peer
 *      carrying this box's real GPU identity, and each must arrive at the correct world table. */
static void world_reconciles(test_suite_t &suite, fs::path const &base, u32_t local_devices) {
    sz::string device_name;
    if (!probe_local_device_name(device_name)) return suite.check(false, "probing the local GPU name");
    sz::string const base_str {base.c_str()};
    std::error_code error;

    // The lead assembles the world by reading a peer's facts, then publishing the merged manifest.
    {
        char const *job = "reconcile_lead";
        fs::path const directory = base / job;
        fs::create_directories(directory, error);
        plant(directory / "host_domain1.mesh",
              facts_payload(job, 1, 2, local_devices, local_devices, "faux-peer", device_name));
        plant(directory / "host_domain1.mesh.arrive", "1\n"); // the peer's arrival at the closing barrier
        auto result = mesh_topology_t::bootstrap(0, 2, local_devices, base_str.c_str(), job, 10.0);
        if (!result) suite.check(false, "lead should assemble the world: {}", result.description_);
        else {
            mesh_topology_t lead = std::move(*result);
            mesh_host_domain_t const &peer = lead.host_domains[1];
            sz::string_view const peer_host {peer.hostname};
            suite.check(lead.host_domain_count() == 2, "lead world size: got {} want 2", lead.host_domain_count());
            suite.check(lead.host_domains[0].device_count == local_devices, "lead self row should reflect the probe");
            suite.check(peer_host == sz::string_view {"faux-peer"}, "peer hostname: got {} want faux-peer", peer_host);
            suite.check(sz::string_view {peer.device_name} == device_name.view(), "peer device_name: got {} want {}",
                        sz::string_view {peer.device_name}, device_name);
        }
    }

    // A follower rebuilds the same world by adopting the lead's manifest, validating its own row.
    {
        char const *job = "reconcile_follower";
        fs::path const directory = base / job;
        fs::create_directories(directory, error);
        plant(directory / "host_domain0.manifest", manifest_payload(job, local_devices, 2, local_devices, device_name));
        plant(directory / "host_domain0.mesh.release", "1\n"); // the lead's release at the closing barrier
        auto result = mesh_topology_t::bootstrap(1, 2, local_devices, base_str.c_str(), job, 10.0);
        if (!result) suite.check(false, "follower should adopt the world: {}", result.description_);
        else {
            mesh_topology_t follower = std::move(*result);
            mesh_host_domain_t const &self = follower.self();
            suite.check(follower.host_domain_count() == 2, "follower world size: got {} want 2",
                        follower.host_domain_count());
            suite.check(self.device_count == local_devices && sz::string_view {self.device_name} == device_name.view(),
                        "follower self row should agree with its probe: got {}", sz::string_view {self.device_name});
        }
    }
}

/** @brief The cluster tier, activated by `$UFORM_HOST_DOMAIN_INDEX` and driven by `mesh-4-8-16.sbatch`:
 *      every rank bootstraps the world for real, then proves the assembled world table is byte-identical
 *      across ranks (round-trip fidelity + phase ordering) and the closing barrier completes. The world
 *      size travels as `$UFORM_HOST_DOMAIN_COUNT` (Slurm's `$SLURM_NTASKS` as a fallback). */
static void run_cluster_exchange(test_suite_t &suite) {
    u32_t const index = host_domain_index_from_environment();
    u32_t count = 0;
    for (char const *variable : {"UFORM_HOST_DOMAIN_COUNT", "SLURM_NTASKS"})
        if (char const *value = std::getenv(variable)) {
            count = static_cast<u32_t>(std::atoi(value));
            break;
        }
    if (count == 0) return suite.check(false, "world size should be set ($UFORM_HOST_DOMAIN_COUNT / $SLURM_NTASKS)");

    auto result = mesh_topology_t::bootstrap(index, count);
    if (!result) return suite.check(false, "the cluster should bootstrap: {}", result.description_);
    mesh_topology_t topology = std::move(*result);
    suite.check(topology.host_domain_count() == count, "world size should match the launch: got {}",
                topology.host_domain_count());

    sz::string table;
    for (u32_t peer = 0; peer != count; ++peer) {
        mesh_host_domain_t const &entry = topology.host_domains[peer];
        (void)format_into(table, "{} {} {} {}\n", entry.host_domain_index, sz::string_view {entry.hostname},
                          entry.device_count, sz::string_view {entry.device_name});
    }
    (void)topology.publish_text("table", table.view());
    bool identical = true;
    for (u32_t peer = 0; peer != count && identical; ++peer) {
        if (peer == index) continue;
        sz::string other;
        if (expected_t awaited = topology.await_text(peer, "table", other, 60.0); !awaited) identical = false;
        else identical = other.view() == table.view();
    }
    suite.check(identical, "every rank should assemble a byte-identical world table");
    suite.check(static_cast<bool>(topology.barrier("done", 60.0)), "the closing barrier should complete");
}

int main() {
    test_suite_t suite {"mesh conformance"};
    bool const cluster = std::getenv("UFORM_HOST_DOMAIN_INDEX") != nullptr;

    sz::string base_name;
    (void)format_into(base_name, "uform_mesh_test_{}", ::getpid());
    fs::path const base = fs::temp_directory_path() / base_name.c_str();
    std::error_code error;
    fs::remove_all(base, error);
    fs::create_directories(base, error);

    // Host-only: the exchange primitive needs no driver.
    exchange_round_trips(suite, base);

    if (cuInit(0) != CUDA_SUCCESS) {
        fs::remove_all(base, error);
        return suite.finish();
    }
    if (cluster) {
        fs::remove_all(base, error);
        run_cluster_exchange(suite);
        return suite.finish();
    }

    // Bootstrap front-loads `discover_mesh_shape`, so the assembly tiers need a multicast + P2P GPU.
    auto discovered = discover_mesh_shape();
    if (discovered) {
        u32_t const local_devices = (*discovered).gpus_per_host_domain;
        single_box_world(suite, base, local_devices);
        world_reconciles(suite, base, local_devices);
    }

    fs::remove_all(base, error);
    return suite.finish();
}
