/**
 *  @file test/mesh.cpp
 *  @author Ash Vardanian
 *  @date July 5, 2026
 *  @brief Conformance test of the mesh bootstrap — the one job @c mesh.hpp exists for: minting each
 *      host_domain's facts and reconciling them over a shared filesystem into one agreed world.
 *
 *  Five cases, the first four proving an operation that must @b succeed — the integer accessors
 *  are one-line divisions and carry no test:
 *
 *      - @ref test_exchange_round_trips — the durable primitive every wiring phase reuses:
 *        @c publish_text writes atomically and @c await_text reads the bytes back verbatim.
 *        Hand-built topology, no GPU.
 *      - @ref test_rendezvous_generations — two forked processes wire one phase three times over
 *        one directory, and every round reads its own generation's payload. No GPU.
 *      - @ref test_single_box_world — the common one-box case: @c bootstrap(0,1) assembles the
 *        whole topology from the local probe and never opens the exchange directory.
 *      - @ref test_world_reconciles — the substance: a lead reads a peer's facts, validates
 *        agreement and assembles the world; a follower adopts the lead's manifest. Needs a GPU.
 *      - @ref test_barrier_traps_past_deadline — a barrier no other arrival reaches traps after
 *        the domain's timeout rather than hanging. Needs a GPU, runs last.
 *
 *  @ref test_world_reconciles drives both sides against a @b matching simulated peer planted with
 *  this box's real GPU identity, so agreement genuinely passes, and a peer of another GPU class is
 *  refused; bootstrap front-loads @c discover_mesh_shape, so a box without a visible GPU SKIPs.
 *  @ref test_barrier_traps_past_deadline leaves the sentence naming why in the mesh's fault table,
 *  and runs last, since the trap poisons the context.
 *
 *  The genuine multi-node happy path is @ref test_cluster_exchange , driven by
 *  @c scripts/slurm/test_mesh-4-8-16.sbatch: every rank bootstraps for real and must assemble a
 *  @b byte-identical world table.
 */
#undef NDEBUG // ! A test's oracle must stay live in every build

#include <cstdio>  // `std::fopen`, `std::fwrite`, `std::fclose` — planting a simulated peer's exchange files
#include <cstdlib> // `std::getenv`, `std::atoi`

#include <unistd.h>   // `::fork`, `::_exit`
#include <sys/wait.h> // `::waitpid`, `WIFEXITED`, `WEXITSTATUS`

#include <chrono> // `std::chrono::milliseconds`
#include <thread> // `std::this_thread::sleep_for` — the lead's late publish

#include <utility>      // `std::move`
#include <system_error> // `std::error_code` — the out-param every `std::filesystem` call reports through
#include <filesystem>   // `std::filesystem` — the temporary exchange directory

#include <cuda.h> // `cuInit`, `cuDeviceGet`, `cuDeviceGetName`

#include "uform/utilities/mesh.hpp"
#include "uform/utilities/multimem_collectives.cuh"

#include "harness.hpp" // `run_test_case`, `expect_true`, `scratch_directory_t`

namespace util = unum::uform::utilities;
using namespace unum::uform::test;
namespace fs = std::filesystem;

/** One host_domain's facts in the exact line-oriented wire format `build_facts_` emits — what a
 *  peer publishes under its `mesh` key for the lead to read. */
static sz::string_t facts_payload(sz::string_view_t job, util::u32_t index, util::u32_t count, util::u32_t multimem,
                                  util::u32_t devices, sz::string_view_t hostname, sz::string_view_t device_name,
                                  util::u32_t gpu_class) {
    sz::string_t text;
    expect_true(
        util::format_into(text,
                          "schema {}\njob {}\nhost_domain_index {}\nhost_domain_count {}\n" //
                          "multimem {}\ngpu_class {}\nhostname {}\ndevices {}\ndevice_name {}\nuuids",
                          util::mesh_schema_k, job, index, count, multimem, gpu_class, hostname, devices, device_name),
        "the text should format");
    // Synthetic per-GPU UUIDs, distinct across host_domains and never colliding with a real GPU id,
    // so the lead's world-uniqueness check sees this faux peer's devices as its own disjoint set.
    for (util::u32_t gpu = 0; gpu != devices; ++gpu)
        expect_true(util::format_into(text, " fa11ba55fa11ba55fa11ba55{:08x}", (index << 16) | gpu),
                    "the text should format");
    return text;
}

/** A merged manifest in the wire format `build_manifest_` emits — what the lead publishes for
 *  every follower to adopt; every row carries @p device_name so a follower's self-row agrees
 *  with its probe. */
static sz::string_t manifest_payload(sz::string_view_t job, util::u32_t multimem, util::u32_t rows, util::u32_t devices,
                                     sz::string_view_t device_name) {
    sz::string_t text;
    expect_true(util::format_into(text, "schema {}\njob {}\nmultimem {}\n", util::mesh_schema_k, job, multimem),
                "the text should format");
    for (util::u32_t row = 0; row != rows; ++row)
        expect_true(util::format_into(text, "host_domain {} hostname faux-{} devices {} device_name {}\n", row, row,
                                      devices, device_name),
                    "the text should format");
    return text;
}

/** Write @p content into @p path — one simulated peer's exchange file, planted before the bootstrap
 *  under test reads it. Uses stdio, never a `std::*stream`. */
static void plant(fs::path const& path, sz::string_view_t content) {
    std::FILE* file = std::fopen(path.c_str(), "wb");
    if (file == nullptr) return;
    expect_true(std::fwrite(content.data(), 1, content.size(), file) == content.size(), "planting {} should write it",
                path.c_str());
    std::fclose(file);
}

/** The device-0 model name with spaces replaced by underscores — exactly what
 *  `probe_local_host_domain_` records, so a planted peer's `device_name` matches this box's probe
 *  and the uniformity check passes. */
static bool probe_local_device_name(sz::string_t& name) {
    CUdevice device = 0;
    if (cuDeviceGet(&device, 0) != CUDA_SUCCESS) return false;
    char raw[64] = {};
    if (cuDeviceGetName(raw, sizeof(raw) - 1, device) != CUDA_SUCCESS) return false;
    for (char* character = raw; *character != '\0'; ++character)
        if (*character == ' ') *character = '_';
    return sz::succeeded(name.assign(sz::string_view_t {raw}));
}

/** The durable exchange round-trips a payload byte-for-byte: `publish_text` writes it
 *  atomically (temp + rename, no leftover), `await_text` reads it whole, and a second publish
 *  replaces the first — the primitive every wiring phase carries its coordinates over.
 *  Hand-built topology, no GPU. */
static void test_exchange_round_trips() {
    scratch_directory_t const scratch;
    fs::path const& base = scratch.path;
    fs::path const directory = base / "exchange";
    std::error_code error;
    fs::create_directories(directory, error);

    util::mesh_topology_t topology;
    if (!expect_true(!st::failed(topology.host_domains.resize(2)), "topology table alloc")) return;
    topology.host_domain_index = 0;
    topology.poll_pause = &pause_between_polls;
    expect_true(util::format_into(topology.exchange_directory, "{}", directory.c_str()), "the text should format");

    sz::string_view_t const payload = "qpn 42\npsn 7\nrkey 3735928559\n";
    expect_true(static_cast<bool>(topology.publish_text("wiring", payload)), "publish_text should succeed");
    expect_true(!fs::exists(directory / "host_domain0.wiring.tmp"), "publish should leave no temporary file");
    sz::string_t received;
    util::expected_t awaited = topology.await_text(0, "wiring", received, wait_limit_k.count());
    expect_true(static_cast<bool>(awaited), "await_text should find the payload: {}", awaited.description_);
    expect_true(received.view() == payload, "payload should round-trip verbatim: got {}", received);

    sz::string_view_t const replacement = "qpn 99\n";
    expect_true(static_cast<bool>(topology.publish_text("wiring", replacement)), "republish_text should succeed");
    sz::string_t again;
    [[maybe_unused]] util::expected_t const reawaited = topology.await_text(0, "wiring", again, wait_limit_k.count());
    expect_true(again.view() == replacement, "republish should atomically replace: got {}", again);
}

/** A hand-built two-host_domain topology over @p directory , as host_domain @p index : what
 *  `bootstrap` assembles, without the GPU probe the rendezvous itself never needs. */
static bool two_host_topology(util::mesh_topology_t& topology, fs::path const& directory, util::u32_t index) {
    if (st::failed(topology.host_domains.resize(2))) return false;
    topology.shape = util::mesh_shape_t {1u, 1u, 2u};
    topology.host_domain_index = index;
    topology.poll_pause = &pause_between_polls;
    return util::format_into(topology.exchange_directory, "{}", directory.c_str());
}

/** Wire @p rounds phases the way every transport does: claim a generation, publish a payload
 *  naming the round under its key, read the peer's payload back, and cross the generation's
 *  barrier. The lead publishes late in every round past the first, so a follower reading an
 *  earlier generation's file would see the wrong round. */
static bool wire_rounds(util::mesh_topology_t& topology, util::u32_t rounds) {
    util::u32_t const peer = 1u - topology.host_domain_index;
    for (util::u32_t round = 0; round != rounds; ++round) {
        util::u32_t const collective_id = topology.next_collective_id();
        sz::string_t key, barrier_key, payload, expected_payload, received;
        if (!util::format_into(key, "multimem.{}.slice", collective_id) ||
            !util::format_into(barrier_key, "multimem.{}", collective_id) ||
            !util::format_into(payload, "round {} host {}", round, topology.host_domain_index) ||
            !util::format_into(expected_payload, "round {} host {}", round, peer))
            return false;
        if (round != 0 && topology.is_lead()) std::this_thread::sleep_for(std::chrono::milliseconds {300});
        if (!topology.publish_text(key.view(), payload.view())) return false;
        if (!topology.await_text(peer, key.view(), received, wait_limit_k.count()) ||
            received.view() != expected_payload.view())
            return false;
        if (!topology.barrier(barrier_key.view(), wait_limit_k.count())) return false;
    }
    return true;
}

/** The probe-trainer pattern across two processes sharing one exchange directory: both wire one
 *  phase three times, as the two measuring trainers and the real one each make a multimem domain,
 *  and every round must read its own generation's payload, never the one before. Host-only: a
 *  forked peer, no GPU. */
static void test_rendezvous_generations() {
    scratch_directory_t const scratch;
    fs::path const& base = scratch.path;
    fs::path const directory = base / "generations";
    std::error_code error;
    fs::create_directories(directory, error);
    constexpr util::u32_t rounds_k = 3;
    pid_t const child = ::fork();
    if (!expect_true(child >= 0, "fork failed")) return;
    if (child == 0) {
        util::mesh_topology_t follower;
        ::_exit(two_host_topology(follower, directory, 1u) && wire_rounds(follower, rounds_k) ? 0 : 1);
    }
    util::mesh_topology_t lead;
    bool const wired = two_host_topology(lead, directory, 0u) && wire_rounds(lead, rounds_k);
    int status = 0;
    ::waitpid(child, &status, 0);
    expect_true(wired, "the lead should read every round's own payload");
    expect_true(WIFEXITED(status) && WEXITSTATUS(status) == 0, "the follower should read every round's own payload");
    expect_true(lead.next_collective_id() == rounds_k, "every round should claim one generation");
}

/** The single-host_domain world: `bootstrap(0,1)` assembles the whole topology from the local probe
 *  and never opens the exchange directory — the common one-box case. */
static case_result_t test_single_box_world(util::u32_t local_devices) {
    if (local_devices == 0) return case_result_t::skipped(no_gpu_device_k);
    scratch_directory_t const scratch;
    fs::path const& base = scratch.path;
    auto [topology, status] = util::mesh_topology_t::bootstrap(0, 1, &pause_between_polls, local_devices, base.c_str(),
                                                               "single_box", wait_limit_k.count());
    if (!expect_true(status, "single-box bootstrap should succeed: {}", status.description_)) return {};
    expect_true(topology.host_domain_count() == 1, "world should hold one host_domain: got {}",
                topology.host_domain_count());
    expect_true(topology.is_lead(), "the sole host_domain should lead");
    expect_true(topology.self().device_count == local_devices, "self facts should carry the probed device count: {}",
                topology.self().device_count);
    expect_true(topology.exchange_directory.empty(), "a single-box world should not touch the filesystem");
    expect_true(!fs::exists(base / "single_box"), "a single-box world should create no exchange files");
    return {};
}

/** The substance: a lead reads a peer's published facts, validates agreement, and assembles the
 *  world; a follower adopts the lead's manifest. Both halves run against a matching simulated peer
 *  carrying this box's real GPU identity, and each must arrive at the correct world table. */
static case_result_t test_world_reconciles(util::u32_t local_devices) {
    if (local_devices == 0) return case_result_t::skipped(no_gpu_device_k);
    scratch_directory_t const scratch;
    fs::path const& base = scratch.path;
    sz::string_t device_name;
    if (!expect_true(probe_local_device_name(device_name), "probing the local GPU name")) return {};
    std::error_code error;

    // The lead assembles the world by reading a peer's facts, then publishing the merged manifest.
    {
        char const* job = "reconcile_lead";
        fs::path const directory = base / job;
        fs::create_directories(directory, error);
        plant(directory / "host_domain1.mesh",
              facts_payload(job, 1, 2, local_devices, local_devices, "faux-peer", device_name, 0x3));
        plant(directory / "host_domain1.mesh.arrive", "1\n"); // the peer's arrival at the closing barrier
        auto [lead, bootstrap_status] = util::mesh_topology_t::bootstrap(0, 2, &pause_between_polls, local_devices,
                                                                         base.c_str(), job, wait_limit_k.count(), 0x3);
        if (!bootstrap_status) expect_true(false, "lead should assemble the world: {}", bootstrap_status.description_);
        else {
            util::mesh_host_domain_t const& peer = lead.host_domains[1];
            sz::string_view_t const peer_host {peer.hostname};
            expect_true(lead.host_domain_count() == 2, "lead world size: got {} want 2", lead.host_domain_count());
            expect_true(lead.host_domains[0].device_count == local_devices, "lead self row should reflect the probe");
            expect_true(peer_host == sz::string_view_t {"faux-peer"}, "peer hostname: got {} want faux-peer",
                        peer_host);
            expect_true(sz::string_view_t {peer.device_name} == device_name.view(), "peer device_name: got {} want {}",
                        sz::string_view_t {peer.device_name}, device_name);
        }
    }

    // The lead refuses a peer whose GPUs train as another class.
    {
        char const* job = "reconcile_gpu_class";
        fs::path const directory = base / job;
        fs::create_directories(directory, error);
        plant(directory / "host_domain1.mesh",
              facts_payload(job, 1, 2, local_devices, local_devices, "faux-peer", device_name, 0x1));
        auto [topology, bootstrap_status] = util::mesh_topology_t::bootstrap(
            0, 2, &pause_between_polls, local_devices, base.c_str(), job, wait_limit_k.count(), 0x3);
        expect_true(!bootstrap_status && bootstrap_status.status_ == util::status_code_t::unsupported_k,
                    "lead should refuse a peer of another GPU class: {}", bootstrap_status.description_);
    }

    // A follower rebuilds the same world by adopting the lead's manifest, validating its own row.
    {
        char const* job = "reconcile_follower";
        fs::path const directory = base / job;
        fs::create_directories(directory, error);
        plant(directory / "host_domain0.manifest", manifest_payload(job, local_devices, 2, local_devices, device_name));
        plant(directory / "host_domain0.mesh.release", "1\n"); // the lead's release at the closing barrier
        auto [follower, bootstrap_status] = util::mesh_topology_t::bootstrap(1, 2, &pause_between_polls, local_devices,
                                                                             base.c_str(), job, wait_limit_k.count());
        if (!bootstrap_status) expect_true(false, "follower should adopt the world: {}", bootstrap_status.description_);
        else {
            util::mesh_host_domain_t const& self = follower.self();
            expect_true(follower.host_domain_count() == 2, "follower world size: got {} want 2",
                        follower.host_domain_count());
            expect_true(
                self.device_count == local_devices && sz::string_view_t {self.device_name} == device_name.view(),
                "follower self row should agree with its probe: got {}", sz::string_view_t {self.device_name});
        }
    }
    return {};
}

/** A barrier whose other arrivals never come traps once it outwaits the domain's timeout instead of
 *  hanging, and member 0's fault record names the barrier target it gave up on. Member 0's round
 *  scalar is set one round ahead, so even a one-GPU domain waits for an arrival no member will
 *  make. The trap poisons the context, so this case runs last. */
static case_result_t test_barrier_traps_past_deadline(util::u32_t local_devices) {
    if (local_devices == 0) return case_result_t::skipped(no_gpu_device_k);
    scratch_directory_t const scratch;
    fs::path const& base = scratch.path;
    auto [topology, bootstrap_status] = util::mesh_topology_t::bootstrap(
        0, 1, &pause_between_polls, local_devices, base.c_str(), "barrier_deadline", wait_limit_k.count());
    if (!expect_true(bootstrap_status, "single-box bootstrap: {}", bootstrap_status.description_)) return {};
    util::per_gpu_array<util::i32_t> devices;
    util::per_gpu_array<util::gpu_stream_t> streams;
    for (util::u32_t device = 0; device != local_devices; ++device) {
        auto [stream, status] = util::gpu_stream_t::make(static_cast<util::i32_t>(device));
        if (!expect_true(status, "stream on GPU {}: {}", device, status.description_)) return {};
        if (!expect_true(devices.push_back(static_cast<util::i32_t>(device)) == util::status_code_t::success_k &&
                             streams.push_back(std::move(stream)) == util::status_code_t::success_k,
                         "member tables full"))
            return {};
    }
    util::f64_t const timeout_seconds = 1.0;
    auto [domain, make_status] = util::multimem_domain_t::make(topology, devices.view(), 1u, streams.span(),
                                                               timeout_seconds);
    if (!expect_true(make_status, "multimem domain make: {}", make_status.description_)) return {};

    util::switch_to_gpu_t gpu_scope {streams[0]};
    if (!expect_true(gpu_scope.ok(), "context entry failed")) return {};
    if (!expect_true(cuMemsetD32(util::device_address(domain.member(0).round), 1u, 1) == CUDA_SUCCESS,
                     "round scalar advance failed"))
        return {};
    auto const started = std::chrono::steady_clock::now();
    util::expected_t const launched = domain.barrier(0, streams[0]);
    CUresult const drained = cuStreamSynchronize(streams[0]);
    std::chrono::duration<util::f64_t> const waited = std::chrono::steady_clock::now() - started;
    expect_true(static_cast<bool>(launched), "barrier launch: {}", launched.description_);
    expect_true(drained != CUDA_SUCCESS, "the starved barrier should trap, not pass");
    expect_true(waited.count() < 10.0 * timeout_seconds, "the trap should land near the {}s timeout, took {}s",
                timeout_seconds, waited.count());

    sz::string_t expected_sentence, sentence;
    expect_true(util::format_into(expected_sentence, "member 0 waited past its deadline at barrier target {}",
                                  local_devices + 1u),
                "the text should format");
    std::span<util::gpu_fault_t> const faults = topology.faults();
    expect_true(util::format_fault_into(sentence, faults[0]), "the fault sentence should format");
    expect_true(sentence.view() == expected_sentence.view(), "the trap should record why: got \"{}\"", sentence);
    return {};
}

/** The cluster tier, activated by `$UFORM_HOST_DOMAIN_INDEX` and driven by `mesh-4-8-16.sbatch`:
 *  every rank bootstraps the world for real, then proves the assembled world table is
 *  byte-identical across ranks (round-trip fidelity + phase ordering) and the closing barrier
 *  completes. The world size travels as `$UFORM_HOST_DOMAIN_COUNT` (Slurm's `$SLURM_NTASKS`
 *  as a fallback). */
static case_result_t test_cluster_exchange(util::u32_t local_devices) {
    if (local_devices == 0) return case_result_t::skipped(no_gpu_device_k);
    util::u32_t const index = util::host_domain_index_from_environment();
    util::u32_t count = 0;
    for (char const* variable : {"UFORM_HOST_DOMAIN_COUNT", "SLURM_NTASKS"})
        if (char const* value = std::getenv(variable)) {
            count = static_cast<util::u32_t>(std::atoi(value));
            break;
        }
    if (!expect_true(count != 0, "world size should be set ($UFORM_HOST_DOMAIN_COUNT / $SLURM_NTASKS)")) return {};

    auto [topology, status] = util::mesh_topology_t::bootstrap(index, count, &pause_between_polls);
    if (!expect_true(status, "the cluster should bootstrap: {}", status.description_)) return {};
    expect_true(topology.host_domain_count() == count, "world size should match the launch: got {}",
                topology.host_domain_count());

    sz::string_t table;
    for (util::u32_t peer = 0; peer != count; ++peer) {
        util::mesh_host_domain_t const& entry = topology.host_domains[peer];
        expect_true(
            util::format_into(table, "{} {} {} {}\n", entry.host_domain_index, sz::string_view_t {entry.hostname},
                              entry.device_count, sz::string_view_t {entry.device_name}),
            "the text should format");
    }
    expect_true(static_cast<bool>(topology.publish_text("table", table.view())), "the table should publish");
    bool identical = true;
    for (util::u32_t peer = 0; peer != count && identical; ++peer) {
        if (peer == index) continue;
        sz::string_t other;
        if (util::expected_t awaited = topology.await_text(peer, "table", other, wait_limit_k.count()); !awaited)
            identical = false;
        else identical = other.view() == table.view();
    }
    expect_true(identical, "every rank should assemble a byte-identical world table");
    expect_true(static_cast<bool>(topology.barrier("done", wait_limit_k.count())),
                "the closing barrier should complete");
    return {};
}

int main(int argc, char** argv) {
    start_test_run(argc, argv);
    bool const cluster = std::getenv("UFORM_HOST_DOMAIN_INDEX") != nullptr;

    // Host-only: the exchange primitive needs no driver.
    run_test_case("test_exchange_round_trips", test_exchange_round_trips);
    run_test_case("test_rendezvous_generations", test_rendezvous_generations);

    // Bootstrap front-loads `discover_mesh_shape`, so the assembly tiers need a visible GPU.
    util::u32_t local_devices = 0;
    if (cuInit(0) == CUDA_SUCCESS)
        if (auto discovered = util::discover_mesh_shape(); discovered) local_devices = discovered->gpus_per_host_domain;
    if (cluster) run_test_case("test_cluster_exchange", test_cluster_exchange, local_devices);
    else {
        run_test_case("test_single_box_world", test_single_box_world, local_devices);
        run_test_case("test_world_reconciles", test_world_reconciles, local_devices);
        run_test_case("test_barrier_traps_past_deadline", test_barrier_traps_past_deadline, local_devices);
    }
    return finish_test_run();
}
