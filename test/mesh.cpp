/**
 *  @file test/mesh.cpp
 *  @author Ash Vardanian
 *  @date July 5, 2026
 *  @brief Conformance test of the mesh bootstrap — the one job `mesh.hpp` exists for: minting each
 *      host_domain's facts and reconciling them over a shared filesystem into one agreed world.
 *
 *  Five cases, the first four proving an operation that must @b succeed — the integer accessors
 *  are one-line divisions and carry no test:
 *
 *      - @ref exchange_round_trips — the durable primitive every wiring phase reuses: `publish_text`
 *        writes atomically and `await_text` reads the bytes back verbatim. Hand-built topology, no GPU.
 *      - @ref rendezvous_generations — two forked processes wire one phase three times over one
 *        directory, and every round reads its own generation's payload. No GPU.
 *      - @ref single_box_world — the common one-box case: `bootstrap(0,1)` assembles the whole topology
 *        from the local probe and never opens the exchange directory.
 *      - @ref world_reconciles — the substance: a lead reads a peer's facts, validates agreement, and
 *        assembles the world; a follower adopts the lead's manifest. Both are driven against a @b matching
 *        simulated peer planted with this box's real GPU identity, so agreement genuinely passes,
 *        and a peer of other capabilities is refused. Needs a visible GPU (bootstrap front-loads
 *        `discover_mesh_shape`); a box without one SKIPs.
 *      - @ref barrier_traps_past_deadline — a domain barrier no other arrival reaches traps after
 *        the domain's timeout rather than hanging, and leaves the sentence naming why in the mesh's
 *        fault table. Needs a visible GPU and runs last, since the trap poisons the context.
 *
 *  The genuine multi-node happy path is @ref run_cluster_exchange , driven by
 *  `scripts/slurm/test_mesh-4-8-16.sbatch`: every rank bootstraps for real and must assemble a @b byte-identical
 *  world table.
 */
#include <cstdio>  // `std::fopen`, `std::fwrite`, `std::fclose` — planting a simulated peer's exchange files
#include <cstdlib> // `std::getenv`, `std::atoi`

#include <unistd.h>   // `::getpid`, `::fork`, `::_exit`
#include <sys/wait.h> // `::waitpid`, `WIFEXITED`, `WEXITSTATUS`

#include <chrono> // `std::chrono::milliseconds`
#include <thread> // `std::this_thread::sleep_for` — the lead's late publish

#include <utility>      // `std::move`
#include <system_error> // `std::error_code` — the out-param every `std::filesystem` call reports through
#include <filesystem>   // `std::filesystem` — the temporary exchange directory

#include <cuda.h> // `cuInit`, `cuDeviceGet`, `cuDeviceGetName`

#include "uform/core/mesh.hpp"
#include "uform/core/multimem_collectives.cuh"

#include "harness.hpp"

using namespace unum::uform::core;
using namespace unum::uform::test;
namespace fs = std::filesystem;

/** One host_domain's facts in the exact line-oriented wire format `build_facts_` emits — what a
 *  peer publishes under its `mesh` key for the lead to read. */
static sz::string_t facts_payload(sz::string_view_t job, u32_t index, u32_t count, u32_t multimem, u32_t devices,
                                  sz::string_view_t hostname, sz::string_view_t device_name, u64_t capabilities) {
    sz::string_t text;
    (void)format_into(text,
                      "schema {}\njob {}\nhost_domain_index {}\nhost_domain_count {}\n" //
                      "multimem {}\ncapabilities {:x}\nhostname {}\ndevices {}\ndevice_name {}\nuuids",
                      mesh_schema_k, job, index, count, multimem, capabilities, hostname, devices, device_name);
    // Synthetic per-GPU UUIDs, distinct across host_domains and never colliding with a real GPU id, so the
    // lead's world-uniqueness check sees this faux peer's devices as its own disjoint set.
    for (u32_t gpu = 0; gpu != devices; ++gpu)
        (void)format_into(text, " fa11ba55fa11ba55fa11ba55{:08x}", (index << 16) | gpu);
    return text;
}

/** A merged manifest in the wire format `build_manifest_` emits — what the lead publishes for every
 *  follower to adopt; every row carries @p device_name so a follower's self-row agrees with its probe. */
static sz::string_t manifest_payload(sz::string_view_t job, u32_t multimem, u32_t rows, u32_t devices,
                                     sz::string_view_t device_name) {
    sz::string_t text;
    (void)format_into(text, "schema {}\njob {}\nmultimem {}\n", mesh_schema_k, job, multimem);
    for (u32_t row = 0; row != rows; ++row)
        (void)format_into(text, "host_domain {} hostname faux-{} devices {} device_name {}\n", row, row, devices,
                          device_name);
    return text;
}

/** Write @p content into @p path — one simulated peer's exchange file, planted before the bootstrap
 *  under test reads it. Uses stdio, never a `std::*stream`. */
static void plant(fs::path const &path, sz::string_view_t content) {
    std::FILE *file = std::fopen(path.c_str(), "wb");
    if (file == nullptr) return;
    (void)std::fwrite(content.data(), 1, content.size(), file);
    std::fclose(file);
}

/** The device-0 model name with spaces replaced by underscores — exactly what
 *  `probe_local_host_domain_` records, so a planted peer's `device_name` matches this box's probe and
 *  the uniformity check passes. */
static bool probe_local_device_name(sz::string_t &name) {
    CUdevice device = 0;
    if (cuDeviceGet(&device, 0) != CUDA_SUCCESS) return false;
    char raw[64] = {};
    if (cuDeviceGetName(raw, sizeof(raw) - 1, device) != CUDA_SUCCESS) return false;
    for (char *character = raw; *character != '\0'; ++character)
        if (*character == ' ') *character = '_';
    return sz::succeeded(name.assign(sz::string_view_t {raw}));
}

/** The durable exchange round-trips a payload byte-for-byte: `publish_text` writes it atomically
 *  (temp + rename, no leftover), `await_text` reads it whole, and a second publish replaces the first —
 *  the primitive every wiring phase carries its coordinates over. Hand-built topology, no GPU. */
static void exchange_round_trips(test_suite_t &suite, fs::path const &base) {
    fs::path const directory = base / "exchange";
    std::error_code error;
    fs::create_directories(directory, error);

    mesh_topology_t topology;
    if (st::failed(topology.host_domains.resize(2))) return suite.check(false, "topology table alloc");
    topology.host_domain_index = 0;
    topology.poll_pause = &pause_between_polls;
    (void)format_into(topology.exchange_directory, "{}", directory.c_str());

    sz::string_view_t const payload = "qpn 42\npsn 7\nrkey 3735928559\n";
    suite.check(static_cast<bool>(topology.publish_text("wiring", payload)), "publish_text should succeed");
    suite.check(!fs::exists(directory / "host_domain0.wiring.tmp"), "publish should leave no temporary file");
    sz::string_t received;
    expected_t awaited = topology.await_text(0, "wiring", received, 5.0);
    suite.check(static_cast<bool>(awaited), "await_text should find the payload: {}", awaited.description_);
    suite.check(received.view() == payload, "payload should round-trip verbatim: got {}", received);

    sz::string_view_t const replacement = "qpn 99\n";
    suite.check(static_cast<bool>(topology.publish_text("wiring", replacement)), "republish_text should succeed");
    sz::string_t again;
    (void)topology.await_text(0, "wiring", again, 5.0);
    suite.check(again.view() == replacement, "republish should atomically replace: got {}", again);
}

/** A hand-built two-host_domain topology over @p directory , as host_domain @p index : what
 *  `bootstrap` assembles, without the GPU probe the rendezvous itself never needs. */
static bool two_host_topology(mesh_topology_t &topology, fs::path const &directory, u32_t index) {
    if (st::failed(topology.host_domains.resize(2))) return false;
    topology.shape = mesh_shape_t {1u, 1u, 2u};
    topology.host_domain_index = index;
    topology.poll_pause = &pause_between_polls;
    return format_into(topology.exchange_directory, "{}", directory.c_str());
}

/** Wire @p rounds phases the way every transport does: claim a generation, publish a payload
 *  naming the round under its key, read the peer's payload back, and cross the generation's
 *  barrier. The lead publishes late in every round past the first, so a follower reading an
 *  earlier generation's file would see the wrong round. */
static bool wire_rounds(mesh_topology_t &topology, u32_t rounds) {
    u32_t const peer = 1u - topology.host_domain_index;
    for (u32_t round = 0; round != rounds; ++round) {
        u32_t const collective_id = topology.next_collective_id();
        sz::string_t key, barrier_key, payload, expected_payload, received;
        if (!format_into(key, "multimem.{}.slice", collective_id) ||
            !format_into(barrier_key, "multimem.{}", collective_id) ||
            !format_into(payload, "round {} host {}", round, topology.host_domain_index) ||
            !format_into(expected_payload, "round {} host {}", round, peer))
            return false;
        if (round != 0 && topology.is_lead()) std::this_thread::sleep_for(std::chrono::milliseconds {300});
        if (!topology.publish_text(key.view(), payload.view())) return false;
        if (!topology.await_text(peer, key.view(), received, 30.0) || received.view() != expected_payload.view())
            return false;
        if (!topology.barrier(barrier_key.view(), 30.0)) return false;
    }
    return true;
}

/** The probe-trainer pattern across two processes sharing one exchange directory: both wire one
 *  phase three times, as the two measuring trainers and the real one each make a multimem domain,
 *  and every round must read its own generation's payload, never the one before. Host-only: a
 *  forked peer, no GPU. */
static void rendezvous_generations(test_suite_t &suite, fs::path const &base) {
    fs::path const directory = base / "generations";
    std::error_code error;
    fs::create_directories(directory, error);
    constexpr u32_t rounds_k = 3;
    pid_t const child = ::fork();
    if (child < 0) return suite.check(false, "fork failed");
    if (child == 0) {
        mesh_topology_t follower;
        ::_exit(two_host_topology(follower, directory, 1u) && wire_rounds(follower, rounds_k) ? 0 : 1);
    }
    mesh_topology_t lead;
    bool const wired = two_host_topology(lead, directory, 0u) && wire_rounds(lead, rounds_k);
    int status = 0;
    ::waitpid(child, &status, 0);
    suite.check(wired, "the lead should read every round's own payload");
    suite.check(WIFEXITED(status) && WEXITSTATUS(status) == 0, "the follower should read every round's own payload");
    suite.check(lead.next_collective_id() == rounds_k, "every round should claim one generation");
}

/** The single-host_domain world: `bootstrap(0,1)` assembles the whole topology from the local probe
 *  and never opens the exchange directory — the common one-box case. */
static void single_box_world(test_suite_t &suite, fs::path const &base, u32_t local_devices) {
    if (local_devices == 0) return suite.skip(no_cuda_device_k);
    auto result = mesh_topology_t::bootstrap(0, 1, &pause_between_polls, local_devices, base.c_str(), "single_box",
                                             10.0);
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

/** The substance: a lead reads a peer's published facts, validates agreement, and assembles the
 *  world; a follower adopts the lead's manifest. Both halves run against a matching simulated peer
 *  carrying this box's real GPU identity, and each must arrive at the correct world table. */
static void world_reconciles(test_suite_t &suite, fs::path const &base, u32_t local_devices) {
    if (local_devices == 0) return suite.skip(no_cuda_device_k);
    sz::string_t device_name;
    if (!probe_local_device_name(device_name)) return suite.check(false, "probing the local GPU name");
    std::error_code error;

    // The lead assembles the world by reading a peer's facts, then publishing the merged manifest.
    {
        char const *job = "reconcile_lead";
        fs::path const directory = base / job;
        fs::create_directories(directory, error);
        plant(directory / "host_domain1.mesh",
              facts_payload(job, 1, 2, local_devices, local_devices, "faux-peer", device_name, 0x3));
        plant(directory / "host_domain1.mesh.arrive", "1\n"); // the peer's arrival at the closing barrier
        auto result = mesh_topology_t::bootstrap(0, 2, &pause_between_polls, local_devices, base.c_str(), job, 10.0,
                                                 0x3);
        if (!result) suite.check(false, "lead should assemble the world: {}", result.description_);
        else {
            mesh_topology_t lead = std::move(*result);
            mesh_host_domain_t const &peer = lead.host_domains[1];
            sz::string_view_t const peer_host {peer.hostname};
            suite.check(lead.host_domain_count() == 2, "lead world size: got {} want 2", lead.host_domain_count());
            suite.check(lead.host_domains[0].device_count == local_devices, "lead self row should reflect the probe");
            suite.check(peer_host == sz::string_view_t {"faux-peer"}, "peer hostname: got {} want faux-peer",
                        peer_host);
            suite.check(sz::string_view_t {peer.device_name} == device_name.view(), "peer device_name: got {} want {}",
                        sz::string_view_t {peer.device_name}, device_name);
        }
    }

    // The lead refuses a peer whose GPUs dispatch with other capabilities.
    {
        char const *job = "reconcile_capabilities";
        fs::path const directory = base / job;
        fs::create_directories(directory, error);
        plant(directory / "host_domain1.mesh",
              facts_payload(job, 1, 2, local_devices, local_devices, "faux-peer", device_name, 0x1));
        auto result = mesh_topology_t::bootstrap(0, 2, &pause_between_polls, local_devices, base.c_str(), job, 10.0,
                                                 0x3);
        suite.check(!result && result.status_ == status_t::shape_unsupported_k,
                    "lead should refuse a peer of other capabilities: {}", result.description_);
    }

    // A follower rebuilds the same world by adopting the lead's manifest, validating its own row.
    {
        char const *job = "reconcile_follower";
        fs::path const directory = base / job;
        fs::create_directories(directory, error);
        plant(directory / "host_domain0.manifest", manifest_payload(job, local_devices, 2, local_devices, device_name));
        plant(directory / "host_domain0.mesh.release", "1\n"); // the lead's release at the closing barrier
        auto result = mesh_topology_t::bootstrap(1, 2, &pause_between_polls, local_devices, base.c_str(), job, 10.0);
        if (!result) suite.check(false, "follower should adopt the world: {}", result.description_);
        else {
            mesh_topology_t follower = std::move(*result);
            mesh_host_domain_t const &self = follower.self();
            suite.check(follower.host_domain_count() == 2, "follower world size: got {} want 2",
                        follower.host_domain_count());
            suite.check(
                self.device_count == local_devices && sz::string_view_t {self.device_name} == device_name.view(),
                "follower self row should agree with its probe: got {}", sz::string_view_t {self.device_name});
        }
    }
}

/** A barrier whose other arrivals never come traps once it outwaits the domain's timeout instead of
 *  hanging, and member 0's fault record names the barrier target it gave up on. Member 0's round
 *  scalar is set one round ahead, so even a one-GPU domain waits for an arrival no member will
 *  make. The trap poisons the context, so this case runs last. */
static void barrier_traps_past_deadline(test_suite_t &suite, fs::path const &base, u32_t local_devices) {
    if (local_devices == 0) return suite.skip(no_cuda_device_k);
    auto bootstrapped = mesh_topology_t::bootstrap(0, 1, &pause_between_polls, local_devices, base.c_str(),
                                                   "barrier_deadline", 10.0);
    if (!bootstrapped) return suite.check(false, "single-box bootstrap: {}", bootstrapped.description_);
    mesh_topology_t topology = std::move(bootstrapped.value_);
    per_gpu_array<i32_t> devices;
    per_gpu_array<gpu_stream_t> streams;
    for (u32_t device = 0; device != local_devices; ++device) {
        expected<gpu_stream_t> stream = gpu_stream_t::make(static_cast<i32_t>(device));
        if (!stream) return suite.check(false, "stream on GPU {}: {}", device, stream.description_);
        if (devices.push_back(static_cast<i32_t>(device)) != status_t::success_k ||
            streams.push_back(std::move(stream.value_)) != status_t::success_k)
            return suite.check(false, "member tables full");
    }
    f64_t const timeout_seconds = 1.0;
    auto made = multimem_domain_t::make(topology, devices.view(), 1u, streams.span(), timeout_seconds);
    if (!made) return suite.check(false, "multimem domain make: {}", made.description_);
    multimem_domain_t domain = std::move(made.value_);

    switch_to_gpu_t gpu_scope {streams[0]};
    if (!gpu_scope.ok()) return suite.check(false, "context entry failed");
    if (cuMemsetD32(device_address(domain.member(0).round), 1u, 1) != CUDA_SUCCESS)
        return suite.check(false, "round scalar advance failed");
    auto const started = std::chrono::steady_clock::now();
    expected_t const launched = domain.barrier(0, streams[0]);
    CUresult const drained = cuStreamSynchronize(streams[0]);
    std::chrono::duration<f64_t> const waited = std::chrono::steady_clock::now() - started;
    suite.check(static_cast<bool>(launched), "barrier launch: {}", launched.description_);
    suite.check(drained != CUDA_SUCCESS, "the starved barrier should trap, not pass");
    suite.check(waited.count() < 10.0 * timeout_seconds, "the trap should land near the {}s timeout, took {}s",
                timeout_seconds, waited.count());

    sz::string_t expected_sentence, sentence;
    (void)format_into(expected_sentence, "member 0 waited past its deadline at barrier target {}", local_devices + 1u);
    std::span<gpu_fault_t> const faults = topology.faults();
    (void)format_fault_into(sentence, faults[0]);
    suite.check(sentence.view() == expected_sentence.view(), "the trap should record why: got \"{}\"", sentence);
}

/** The cluster tier, activated by `$UFORM_HOST_DOMAIN_INDEX` and driven by `mesh-4-8-16.sbatch`:
 *  every rank bootstraps the world for real, then proves the assembled world table is byte-identical
 *  across ranks (round-trip fidelity + phase ordering) and the closing barrier completes. The world
 *  size travels as `$UFORM_HOST_DOMAIN_COUNT` (Slurm's `$SLURM_NTASKS` as a fallback). */
static void run_cluster_exchange(test_suite_t &suite, u32_t local_devices) {
    if (local_devices == 0) return suite.skip(no_cuda_device_k);
    u32_t const index = host_domain_index_from_environment();
    u32_t count = 0;
    for (char const *variable : {"UFORM_HOST_DOMAIN_COUNT", "SLURM_NTASKS"})
        if (char const *value = std::getenv(variable)) {
            count = static_cast<u32_t>(std::atoi(value));
            break;
        }
    if (count == 0) return suite.check(false, "world size should be set ($UFORM_HOST_DOMAIN_COUNT / $SLURM_NTASKS)");

    auto result = mesh_topology_t::bootstrap(index, count, &pause_between_polls);
    if (!result) return suite.check(false, "the cluster should bootstrap: {}", result.description_);
    mesh_topology_t topology = std::move(*result);
    suite.check(topology.host_domain_count() == count, "world size should match the launch: got {}",
                topology.host_domain_count());

    sz::string_t table;
    for (u32_t peer = 0; peer != count; ++peer) {
        mesh_host_domain_t const &entry = topology.host_domains[peer];
        (void)format_into(table, "{} {} {} {}\n", entry.host_domain_index, sz::string_view_t {entry.hostname},
                          entry.device_count, sz::string_view_t {entry.device_name});
    }
    (void)topology.publish_text("table", table.view());
    bool identical = true;
    for (u32_t peer = 0; peer != count && identical; ++peer) {
        if (peer == index) continue;
        sz::string_t other;
        if (expected_t awaited = topology.await_text(peer, "table", other, 60.0); !awaited) identical = false;
        else identical = other.view() == table.view();
    }
    suite.check(identical, "every rank should assemble a byte-identical world table");
    suite.check(static_cast<bool>(topology.barrier("done", 60.0)), "the closing barrier should complete");
}

int main(int, char **argv) {
    test_suite_t suite = test_suite_t::make(argv[0]);
    bool const cluster = std::getenv("UFORM_HOST_DOMAIN_INDEX") != nullptr;

    sz::string_t base_name;
    (void)format_into(base_name, "uform_mesh_test_{}", ::getpid());
    fs::path const base = fs::temp_directory_path() / base_name.c_str();
    std::error_code error;
    fs::remove_all(base, error);
    fs::create_directories(base, error);

    // Host-only: the exchange primitive needs no driver.
    suite.run("exchange_round_trips", exchange_round_trips, base);
    suite.run("rendezvous_generations", rendezvous_generations, base);

    // Bootstrap front-loads `discover_mesh_shape`, so the assembly tiers need a visible GPU.
    u32_t local_devices = 0;
    if (cuInit(0) == CUDA_SUCCESS)
        if (auto discovered = discover_mesh_shape(); discovered) local_devices = discovered->gpus_per_host_domain;
    if (cluster) suite.run("cluster_exchange", run_cluster_exchange, local_devices);
    else {
        suite.run("single_box_world", single_box_world, base, local_devices);
        suite.run("world_reconciles", world_reconciles, base, local_devices);
        suite.run("barrier_traps_past_deadline", barrier_traps_past_deadline, base, local_devices);
    }

    fs::remove_all(base, error);
    return suite.finish();
}
