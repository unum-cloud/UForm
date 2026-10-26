/**
 *  @file include/uform/core/mesh.hpp
 *  @author Ash Vardanian
 *  @date July 2, 2026
 *  @brief The multi-level GPU mesh: nested domain sizes, this rank's coordinate, and the world
 *      bootstrap.
 *
 *  Four artifacts every distributed primitive keys on:
 *
 *      1. @ref mesh_shape_t — the three nested domain sizes, the single source of truth for cluster
 *         geometry; every grouping question is integer arithmetic off them.
 *      2. @ref mesh_location_t — this rank's coordinate plus the reachability handle mesh-wide kernels
 *         read, passed by value through kernel-argument lists.
 *      3. @ref mesh_topology_t — the bootstrapped world: the shape, this host_domain's identity, every
 *         peer host_domain's published facts, and the once-per-job exchange primitives later wiring
 *         phases reuse.
 *      4. @ref make_mesh_shape / @ref discover_mesh_shape / @ref mesh_topology_t::bootstrap — the
 *         validated construction paths.
 *
 *  @section mesh_purpose Purpose — why a bootstrap exchange exists at all
 *
 *  The launcher (SSH fan-out, Slurm, Kubernetes) hands every process what is known @b before it starts:
 *  identity, count, binary, environment. What it cannot hand out is minted at runtime, by each
 *  host_domain, after boot — and there is no transport to send it over, because the transport is exactly
 *  what it wires:
 *
 *      1. InfiniBand queue-pair coordinates — QP numbers, PSNs, LIDs / GIDs — assigned by the NIC driver
 *         at queue-pair creation.
 *      2. Memory-region @c rkeys — produced by each host_domain's MR registration of its own arena.
 *      3. GPU heap base addresses — produced by each host_domain's VMM reservation.
 *      4. Multicast-object fabric handles — exported by one process per multimem_domain, imported by its
 *         siblings before the domain-wide bind.
 *
 *  Secondary roles, in descending importance: @b phase @b ordering (arenas allocated → MRs registered → QPs
 *  connected → training starts; @ref mesh_topology_t::barrier keeps any host_domain from running ahead),
 *  @b agreement @b validation (GPU model, count and capabilities, world size, multimem extent,
 *  schema and job identity), and @b boot @b verification (a bounded await turns a wedged
 *  host_domain into a named failure, never a hang). The exchange runs once per job, before step 0;
 *  the moment the transport is wired, all synchronization moves onto NVLink / InfiniBand and the
 *  directory is never touched again.
 *
 *  @section mesh_worlds Three concrete worlds
 *
 *      - Two HGX H100 boxes over InfiniBand: one process per box drives all 8 GPUs, and the multicast
 *        reach ends at the box — shape @c {8,8,16} : 2 host_domains, each its own multimem_domain.
 *      - Two GB200 NVL72 racks: 18 compute trays per rack, one process per tray driving its 4 Blackwell
 *        GPUs, NVLink multicast spanning the whole rack — shape @c {4,72,144} : 36 host_domains, 18 per
 *        multimem_domain, 2 multimem_domains.
 *      - Two Vera Rubin NVL144 racks: the same tray topology, one process per tray driving its 8 Rubin
 *        GPU dies, rack-wide multicast — shape @c {8,144,288} : 36 host_domains, 2 multimem_domains.
 *
 *  The middle number is why @c gpus_per_multimem_domain is @b explicit configuration: a process on an
 *  NVL72 tray sees 4 devices, and the 72-GPU multicast reach is a fabric property no local probe can
 *  observe — it is asserted at launch, cross-checked during the exchange, and proven when the multicast
 *  object binds. Per-multimem_domain wiring (the fabric handle, the domain's heap and QP digest) is
 *  published once per domain by its lead, so cross-rack setup reads O(domains) files, not O(processes).
 *
 *  @section mesh_stance Design stance
 *
 *      - The colocation is @b homogeneous for now: every host_domain carries the same count of identical
 *        GPUs, so three integers describe the whole world; heterogeneity is rejected at bootstrap, never
 *        papered over.
 *      - No failure recovery: the project is built around @b over-provisioning instead — a hot-spare
 *        host_domain takes a dead peer's index and the world re-bootstraps at the same shape from the
 *        last checkpoint. Membership is fixed for a job's lifetime; detection is bounded waits plus
 *        supervisor exit codes.
 *      - Launcher-agnostic through a deliberately tiny contract: identity and configuration come in
 *        through environment variables ( @c $UFORM_HOST_DOMAIN_INDEX or a scheduler's equivalent,
 *        @c $UFORM_GPUS_PER_MULTIMEM_DOMAIN , @c $UFORM_EXCHANGE_DIR , @c $UFORM_JOB_ID ), the exit code
 *        goes out. Raw SSH fan-out, Slurm ( @c $SLURM_NODEID ), and Kubernetes indexed Jobs all satisfy
 *        it without touching this code.
 *      - Why not NCCL, NVSHMEM, or MPI + PMI, which solve the same bootstrap: each drags a runtime
 *        dependency and its own failure modes for a once-per-job exchange of a few hundred bytes. NCCL's
 *        store needs a TCP server and open ports; NVSHMEM itself bootstraps over MPI / PMI; and none
 *        offers failure recovery anyway — transparent member replacement is incompatible with captured
 *        graphs and symmetric heaps, whose addresses are baked at capture time. A shared filesystem is
 *        the one channel a GPU cluster already has (the datasets and checkpoints live on it), and it
 *        stays inspectable with @c ls and @c cat at 3 a.m.
 */
#pragma once

#include <cstdint> // `std::uint32_t`
#include <cstring> // `std::memcpy`, `std::strcmp`
#include <cstdio>  // `std::fopen`, `std::fwrite`, `std::fread`, `std::rename`
#include <cstdlib> // `std::getenv`
#include <cerrno>  // `errno`, `EEXIST`, `ENOMEM` — the directory creation's failures

#include <unistd.h>   // `::gethostname`, `::fsync`
#include <sys/stat.h> // `::mkdir` — the exchange directory, created level by level

#include <charconv>  // `std::from_chars`
#include <chrono>    // `std::chrono::steady_clock` — the exchange timeout clock
#include <span>      // `std::span` — the probed GPU-uuid list handed to `build_facts_`
#include <algorithm> // `std::sort` — the world-GPU-uniqueness check

#include <cuda.h> // `CUdevice`, `cuDeviceGetAttribute`, `cuDeviceCanAccessPeer`

#include <stringzilla/stringzilla.hpp> // `sz::string_t`, `sz::string_view_t` — exception-free text handling

#include "uform/core/array.hpp"          // `per_gpu_array` for the device UUIDs
#include "uform/core/gpu_allocators.hpp" // `cuda_failure`
#include "uform/core/types.hpp"          // `expected`, `expected_t`, `status_t`, `format_into`, `st::basic_vector`

namespace unum::uform::core {

namespace sz = ashvardanian::stringzilla;

/**
 *  @brief The three-domain mesh shape — the host-side single source of truth for cluster geometry.
 *
 *  Names the three @b nested GPU domains a run spans, smallest to largest, and stores @b only their three
 *  sizes — every grouping question is integer arithmetic off them, never stored:
 *
 *      - @c host_domain — the GPUs @b one process drives (one OS image; a whole HGX box, or one NVL72
 *        compute tray).
 *      - @c multimem_domain — the GPUs reachable by @c multimem.* over NVLink / NVSwitch (HGX = 8,
 *        NVL72 = 72, NVL144 = 144), addressed by a single multimem VA. One or more host_domains compose
 *        a multimem_domain.
 *      - @c colocated_domain — the whole InfiniBand world (~10^5), reached by device-initiated one-sided
 *        RDMA. One or more multimem_domains compose the colocated_domain; it shares @b one global batch.
 *
 *  Invariant: @c host≤multimem≤colocated , @c multimem%host==0 , @c colocated%multimem==0 — enforced by
 *  @ref make_mesh_shape . The mesh is uniform @b by design: every host_domain carries the same count of
 *  identical GPUs, so three integers describe the whole world — heterogeneous worlds are rejected at
 *  bootstrap, never papered over. The concrete worlds in the file header: two HGX H100 boxes are
 *  @c {8,8,16} , two GB200 NVL72 racks are @c {4,72,144} , two Vera Rubin NVL144 racks are @c {8,144,288} .
 *  Immutable after rendezvous — the topology is identity, not state. A rank is one GPU, named by a single
 *  @c u32_t @c gpu_index in @c [0,gpus_per_colocated_domain) ; there are no rank/index aliases.
 */
struct mesh_shape_t {

    /** GPUs one process drives — a whole HGX box, or one NVL72 / NVL144 compute tray. */
    u32_t gpus_per_host_domain = 1u;

    /** GPUs in one NVLink domain reachable by @c multimem.* (8 on HGX, 72 / 144 on NVL racks). */
    u32_t gpus_per_multimem_domain = 1u;

    /** Total data-parallel GPUs sharing @b one global batch — the whole InfiniBand world (~10^5). */
    u32_t gpus_per_colocated_domain = 1u;

    /** Which host_domain a global @p gpu_index belongs to. */
    [[nodiscard]] u32_t host_domain_of(u32_t gpu_index) const noexcept { return gpu_index / gpus_per_host_domain; }

    /** Which multimem_domain a global @p gpu_index belongs to. */
    [[nodiscard]] u32_t multimem_domain_of(u32_t gpu_index) const noexcept {
        return gpu_index / gpus_per_multimem_domain;
    }

    /** A global @p gpu_index 's position @b within its multimem_domain — the intra-domain collective rank. */
    [[nodiscard]] u32_t index_in_multimem_domain(u32_t gpu_index) const noexcept {
        return gpu_index % gpus_per_multimem_domain;
    }

    /** The colocated_domain (world) coordinate of host_domain @p host_domain_index 's member
     *  @p member_index — the inverse of @ref host_domain_of , so a process turns its local member
     *  indexes into world coordinates. */
    [[nodiscard]] u32_t gpu_index_of_host_member(u32_t host_domain_index, u32_t member_index) const noexcept {
        return host_domain_index * gpus_per_host_domain + member_index;
    }

    /** The colocated_domain (world) coordinate of multimem_domain @p multimem_domain_index 's
     *  member @p member_index — the inverse of @ref multimem_domain_of + @ref index_in_multimem_domain . */
    [[nodiscard]] u32_t gpu_index_of_multimem_member(u32_t multimem_domain_index, u32_t member_index) const noexcept {
        return multimem_domain_index * gpus_per_multimem_domain + member_index;
    }

    /** Number of host_domains (processes) in the colocated_domain. */
    [[nodiscard]] u32_t host_domain_count() const noexcept { return gpus_per_colocated_domain / gpus_per_host_domain; }

    /** Number of multimem_domains in the colocated_domain — the inter-domain collective width. */
    [[nodiscard]] u32_t multimem_domain_count() const noexcept {
        return gpus_per_colocated_domain / gpus_per_multimem_domain;
    }
};

/**
 *  @brief This rank's mesh coordinate plus the reachability handle mesh-wide kernels read — passed
 *      @b by value into kernels: device-readable only, no stream, no host pointers, no ownership.
 *
 *  Everything else a kernel needs derives from @c topology + @c gpu_index (the intra-domain collective
 *  rank is @c topology.index_in_multimem_domain(gpu_index) , the global reduce divisor is
 *  @c topology.gpus_per_colocated_domain ). The cross-host_domain reachability table — one link per
 *  remote multimem_domain, reached over InfiniBand — joins this struct with the RDMA transport primitive;
 *  until then a single multimem_domain is the whole reachable world.
 */
struct mesh_location_t {

    /** The three nested-domain sizes (by value) — domains derived from @c gpu_index . */
    mesh_shape_t topology {};

    /** This rank's global coordinate in @c [0,gpus_per_colocated_domain) — the only coordinate stored. */
    u32_t gpu_index = 0u;

    /** Multicast VA of this rank's multimem_domain; a @c multimem.* op against @c +offset fans
     *  across the whole domain, so there is no per-peer pointer list for intra-domain peers. Null
     *  until the symmetric heap binds it; null at @c gpus_per_multimem_domain==1 forever. */
    void *multimem_domain_base = nullptr;
};

/**
 *  @brief Validate the three explicit domain sizes against the nesting invariants and assemble the
 *      shape — the construction path for hand-configured worlds.
 *
 *  @param[in] gpus_per_host_domain GPUs this process drives; @c >=1 .
 *  @param[in] gpus_per_multimem_domain One NVLink multicast domain; a multiple of the host domain.
 *  @param[in] gpus_per_colocated_domain The whole data-parallel world; a multiple of the multimem domain.
 */
[[nodiscard]] inline expected<mesh_shape_t> make_mesh_shape(u32_t gpus_per_host_domain, u32_t gpus_per_multimem_domain,
                                                            u32_t gpus_per_colocated_domain) noexcept {
    if (gpus_per_host_domain == 0 || gpus_per_host_domain > gpus_per_multimem_domain ||
        gpus_per_multimem_domain > gpus_per_colocated_domain)
        return expected<mesh_shape_t>::fail(
            status_t::invalid_input_k, "need 1 <= host <= multimem <= colocated (host={} multimem={} colocated={})",
            gpus_per_host_domain, gpus_per_multimem_domain, gpus_per_colocated_domain);
    if (gpus_per_multimem_domain % gpus_per_host_domain != 0 ||
        gpus_per_colocated_domain % gpus_per_multimem_domain != 0)
        return expected<mesh_shape_t>::fail(status_t::invalid_input_k,
                                            "domains must nest evenly (host={} multimem={} colocated={})",
                                            gpus_per_host_domain, gpus_per_multimem_domain, gpus_per_colocated_domain);
    return mesh_shape_t {gpus_per_host_domain, gpus_per_multimem_domain, gpus_per_colocated_domain};
}

/**
 *  @brief Discover the single-host_domain shape: all three domains span the visible devices.
 *
 *  Neither multicast nor P2P is required here: a @ref multimem_domain_t without multicast runs its
 *  peer-read tier, and only a domain asks @ref peers_reach_each_other of its own members.
 *
 *  Covers the common one-box world (an HGX box driven by one process); the multi-host_domain world builds
 *  on it through @ref mesh_topology_t::bootstrap . The probe sees @b only this process's devices — a wider
 *  fabric extent (NVL72 / NVL144) is explicit configuration, cross-checked during the exchange and proven
 *  when the multicast object binds. Requires @c cuInit to have run; needs no CUDA context.
 */
[[nodiscard]] inline expected<mesh_shape_t> discover_mesh_shape() noexcept {
    int device_count = 0;
    if (CUresult result = cuDeviceGetCount(&device_count); result != CUDA_SUCCESS)
        return cuda_failure<mesh_shape_t>(result, "device count query failed");
    if (device_count <= 0) return expected<mesh_shape_t>::fail(status_t::invalid_input_k, "no CUDA devices visible");
    u32_t const gpus = static_cast<u32_t>(device_count);
    return mesh_shape_t {gpus, gpus, gpus};
}

/** Whether every ordered pair of @p devices has a P2P path, failing with @c shape_unsupported_k on
 *  the first pair without one — what the members of one domain need, since they read each other's
 *  memory. */
[[nodiscard]] inline expected_t peers_reach_each_other(std::span<i32_t const> devices) noexcept {
    for (i32_t const device_index : devices) {
        CUdevice device = 0;
        if (CUresult result = cuDeviceGet(&device, device_index); result != CUDA_SUCCESS)
            return cuda_failure(result, "device handle query failed (device={})", device_index);
        for (i32_t const peer_index : devices) {
            if (peer_index == device_index) continue;
            CUdevice peer = 0;
            if (CUresult result = cuDeviceGet(&peer, peer_index); result != CUDA_SUCCESS)
                return cuda_failure(result, "device handle query failed (device={})", peer_index);
            int can_access = 0;
            if (CUresult result = cuDeviceCanAccessPeer(&can_access, device, peer); result != CUDA_SUCCESS)
                return cuda_failure(result, "P2P capability query failed (device={} peer={})", device_index,
                                    peer_index);
            if (can_access == 0)
                return expected_t::fail(status_t::shape_unsupported_k,
                                        "devices {} and {} share a domain but no P2P path", device_index, peer_index);
        }
    }
    return {};
}

/** Longest hostname (including the terminator) a @ref mesh_host_domain_t records. */
inline constexpr std::size_t hostname_capacity_k = 64u;

/** Longest GPU model name (including the terminator) a @ref mesh_host_domain_t records. */
inline constexpr std::size_t device_name_capacity_k = 64u;

/** Version tag heading every exchange payload, so a foreign or future-format file in the exchange
 *  directory fails loudly on its first line instead of misparsing. */
inline constexpr char mesh_schema_k[] = "uform.mesh.2";

/** One host_domain's published facts — the payload of the bootstrap exchange, one entry per
 *  host_domain in @ref mesh_topology_t::host_domains . POD, so an @c st::basic_vector holds the
 *  table. */
struct mesh_host_domain_t {

    /** The host_domain's index in @c [0,host_domain_count) — identity from the launcher, never derived. */
    u32_t host_domain_index = 0;

    /** GPUs the host_domain probed locally; the bootstrap validates this is uniform across the world. */
    u32_t device_count = 0;

    /** The host_domain's hostname — for diagnostics and for the transport wiring to name its peers. */
    char hostname[hostname_capacity_k] = {};

    /** The GPU model, spaces replaced with underscores; the bootstrap validates the world runs
     *  identical GPUs, not just identical counts. */
    char device_name[device_name_capacity_k] = {};
};

/** Why a kernel trapped — the meaning of a @ref gpu_fault_t 's three operands. */
enum class gpu_fault_code_t : u32_t {
    none_k = 0,

    /** A multimem barrier outwaited its deadline: member index, barrier target. */
    barrier_deadline_k,

    /** An InfiniBand round outwaited its arrival flag: round, flag target. */
    flag_deadline_k,

    /** An InfiniBand completion queue outwaited its completion: queue number, consumer index. */
    completion_deadline_k,

    /** An InfiniBand send completed with an error: syndrome, vendor syndrome, queue number. */
    completion_error_k,

    /** A telemetry probe sampled a non-finite value: probe row, scalars sampled, scalars probed. */
    nonfinite_signal_k,
};

/** Why a kernel trapped, written to pinned memory the host reads after the context dies. */
struct gpu_fault_t {
    gpu_fault_code_t code;
    u32_t first;
    u32_t second;
    u32_t third;
};

/** Append the sentence naming @p fault to @p text ; @c false only on allocation failure. */
[[nodiscard]] inline bool format_fault_into(sz::string_t &text, gpu_fault_t const &fault) noexcept {
    switch (fault.code) {
    case gpu_fault_code_t::none_k: return format_into(text, "no fault");
    case gpu_fault_code_t::barrier_deadline_k:
        return format_into(text, "member {} waited past its deadline at barrier target {}", fault.first, fault.second);
    case gpu_fault_code_t::flag_deadline_k:
        return format_into(text, "round {} waited past its deadline at flag target {}", fault.first, fault.second);
    case gpu_fault_code_t::completion_deadline_k:
        return format_into(text, "completion queue {} waited past its deadline at consumer index {}", fault.first,
                           fault.second);
    case gpu_fault_code_t::completion_error_k:
        return format_into(text, "send completion error syndrome=0x{:x} vendor=0x{:x} cqn={}", fault.first,
                           fault.second, fault.third);
    case gpu_fault_code_t::nonfinite_signal_k:
        return format_into(text, "telemetry probe {} observed a non-finite signal ({} of {} scalars sampled)",
                           fault.first, fault.second, fault.third);
    }
    return format_into(text, "unknown fault {}", static_cast<u32_t>(fault.code));
}

#if defined(__CUDACC__)

/** Record why this kernel dies in @p fault , when it has one, then trap. */
__device__ inline void trap_with(gpu_fault_t *fault, gpu_fault_code_t code, u32_t first, u32_t second,
                                 u32_t third) noexcept {
    if (fault != nullptr) {
        fault->first = first;
        fault->second = second;
        fault->third = third;
        fault->code = code;
        __threadfence_system();
    }
    __trap();
}

#endif // defined(__CUDACC__)

/** The wait between two reads of the exchange directory, picked by whoever bootstraps the mesh. */
using mesh_pause_t = void (*)() noexcept;

/**
 *  @brief The bootstrapped world: the validated @ref mesh_shape_t , this host_domain's identity, every
 *      peer host_domain's published facts, and the once-per-job exchange primitives later wiring phases
 *      reuse.
 *
 *  @ref bootstrap composes the two discovery levels: the local hardware probe (
 *  @ref discover_mesh_shape — the device count — plus the GPU model name) and a cross-host_domain
 *  fact exchange over a shared filesystem, the only medium a fresh multi-host_domain job can rely
 *  on before any transport exists. Every payload is a tiny line-oriented text file, atomically
 *  published (write-temp + @c fsync + @c rename , so a peer never reads a half-written file). The
 *  exchange is @b lead-aggregated and costs @b O(host_domain_count) file operations, never
 *  O(host_domain_count²): every host_domain publishes one facts file; the lead reads them all,
 *  validates agreement (identical GPU model, count and capabilities, world size, multimem extent
 *  and job — a mismatch fails loudly naming both parties), and publishes @b one merged manifest;
 *  every follower reads only that manifest. The same @ref publish_text / @ref await_text / @ref
 *  barrier primitives then carry every later wiring phase — InfiniBand queue-pair coordinates,
 *  memory-region rkeys, symmetric-heap base addresses, multicast fabric handles — so the transport
 *  adds payloads, not machinery.
 *
 *  The exchange directory is @b explicit configuration ( @c bootstrap argument or @c $UFORM_EXCHANGE_DIR )
 *  — there is no default path, because a guessed directory that exists locally but is not actually shared
 *  would hang the world. Every byte of the @b path is therefore @b predetermined before launch; the only
 *  runtime-minted parts of the exchange are the @b payload @b contents — the probed facts here, the
 *  queue-pair coordinates, rkeys, and heap bases in the later wiring phases. The job id is equally
 *  explicit ( @c $UFORM_JOB_ID , minted @b once by the launcher: a scheduler job id or a timestamp) and
 *  namespaces everything, so many jobs and many node families can share one filesystem — a payload names
 *  its own schema, job, and index, and anything that disagrees with the directory it sits in is rejected.
 *  On a dual GB200 NVL72 world (36 tray processes, shape @c {4,72,144} ) the layout stays deliberately
 *  @c cat -inspectable:
 *
 *  @code{.sh}
 *  $ cat /shared/exchange/20260702-193412-8841/host_domain7.mesh
 *  schema uform.mesh.2
 *  job 20260702-193412-8841
 *  host_domain_index 7
 *  host_domain_count 36
 *  multimem 72
 *  capabilities 17000000000000
 *  hostname nvl72-rack0-tray7
 *  devices 4
 *  device_name NVIDIA_GB200
 *  uuids 5c1e0a...c9 5c1e0a...ca 5c1e0a...cb 5c1e0a...cc
 *
 *  $ cat /shared/exchange/20260702-193412-8841/host_domain0.manifest
 *  schema uform.mesh.2
 *  job 20260702-193412-8841
 *  multimem 72
 *  host_domain 0 hostname nvl72-rack0-tray0 devices 4 device_name NVIDIA_GB200
 *  host_domain 1 hostname nvl72-rack0-tray1 devices 4 device_name NVIDIA_GB200
 *  ...
 *  @endcode
 *
 *  A single-host_domain world ( @c host_domain_count()==1 ) never touches the filesystem. The whole
 *  exchange happens once per job, on the host, outside any CUDA-graph capture. Move-only; the shape is
 *  immutable after @ref bootstrap returns — the topology is identity, not state. It also owns the
 *  @ref faults table its local GPUs' kernels record into before they trap.
 */
struct mesh_topology_t {

    mesh_topology_t() noexcept = default;
    mesh_topology_t(mesh_topology_t const &) = delete;
    mesh_topology_t &operator=(mesh_topology_t const &) = delete;
    mesh_topology_t(mesh_topology_t &&other) noexcept { swap_(other); }
    mesh_topology_t &operator=(mesh_topology_t &&other) noexcept {
        if (this != &other) {
            destroy_();
            swap_(other);
        }
        return *this;
    }
    ~mesh_topology_t() noexcept { destroy_(); }

    /** The validated three-domain shape — see the concrete worlds in the file header. */
    mesh_shape_t shape {};

    /** This host_domain's index in @c [0,host_domain_count) — identity from the launcher. */
    u32_t host_domain_index = 0;

    /** Every host_domain's published facts, indexed by host_domain index. */
    st::basic_vector<mesh_host_domain_t> host_domains {};

    /** The resolved @c <directory>/<job_id> exchange root; empty in a single-host_domain world. */
    sz::string_t exchange_directory {};

    /** The wait between two reads of @ref exchange_directory while a peer's payload is missing. */
    mesh_pause_t poll_pause = nullptr;

    /** Number of host_domains in the world — the table's extent, agreeing with
     *  @c shape.host_domain_count() by construction: @ref bootstrap sizes both from the same argument
     *  and never resizes either. */
    [[nodiscard]] u32_t host_domain_count() const noexcept { return static_cast<u32_t>(host_domains.size()); }

    /** This host_domain's own published facts. */
    [[nodiscard]] mesh_host_domain_t const &self() const noexcept { return host_domains[host_domain_index]; }

    /** Whether this host_domain owns @c gpu_index 0 — the one that aggregates the exchange and
     *  writes checkpoints and metrics. */
    [[nodiscard]] bool is_lead() const noexcept { return host_domain_index == 0; }

    /** The global @c gpu_index of this host_domain's local CUDA device ordinal 0 —
     *  @c host_domain_index·gpus_per_host_domain . This is the local↔global mapping every wiring
     *  phase relies on: local ordinal @c d is global @c gpu_index @c first_gpu_index()+d . */
    [[nodiscard]] u32_t first_gpu_index() const noexcept { return host_domain_index * shape.gpus_per_host_domain; }

    /** Which multimem_domain this host_domain belongs to — on an NVL72 tray, which rack. */
    [[nodiscard]] u32_t multimem_domain_index() const noexcept { return shape.multimem_domain_of(first_gpu_index()); }

    /** Whether this host_domain leads its multimem_domain — the process that creates the multicast
     *  object, exports its fabric handle, and publishes the per-domain wiring digest. */
    [[nodiscard]] bool is_multimem_domain_lead() const noexcept {
        return shape.index_in_multimem_domain(first_gpu_index()) == 0;
    }

    /** One pinned fault record per local GPU, in local order, readable after a trap killed its
     *  context; empty for a topology @ref bootstrap did not make. */
    [[nodiscard]] std::span<gpu_fault_t> faults() const noexcept {
        return {faults_, faults_ != nullptr ? shape.gpus_per_host_domain : 0u};
    }

    /**
     *  @brief Probe the local hardware, exchange facts with every peer host_domain, validate agreement,
     *      and assemble the world — the one construction path for @ref mesh_topology_t .
     *
     *  @param[in] host_domain_index This host_domain's identity in @c [0,host_domain_count) , from the
     *      launcher (see @ref host_domain_index_from_environment ).
     *  @param[in] host_domain_count Number of host_domains in the job — explicit, never guessed: a
     *      host_domain cannot know how many peers will eventually arrive.
     *  @param[in] poll_pause The wait between two reads of the exchange directory, kept as
     *      @ref poll_pause for every later wiring phase.
     *  @param[in] gpus_per_multimem_domain The NVLink multicast extent — a fabric property no local probe
     *      can observe, so it is explicit: @c 0 resolves @c $UFORM_GPUS_PER_MULTIMEM_DOMAIN , then
     *      defaults to the host_domain itself (every pre-NVL72 box, where multicast ends at the box).
     *  @param[in] directory The shared directory every host_domain sees; empty resolves
     *      @c $UFORM_EXCHANGE_DIR . A multi-host_domain bootstrap without one fails loudly — there is no
     *      guessable default.
     *  @param[in] job_id The launch's unique namespace; empty resolves @c $UFORM_JOB_ID , and a
     *      multi-host_domain bootstrap without one fails loudly — an anonymous shared namespace is how
     *      node families collide.
     *  @param[in] timeout_seconds Upper bound on every await; a missing peer fails loudly, never hangs.
     *  @param[in] capabilities The NumKong capabilities this host_domain's GPUs dispatch with after
     *      any class pin, which every peer must share, so no two replicas run different kernels.
     */
    [[nodiscard]] static expected<mesh_topology_t> bootstrap(
        u32_t host_domain_index, u32_t host_domain_count, mesh_pause_t poll_pause, u32_t gpus_per_multimem_domain = 0,
        sz::string_view_t directory = {}, sz::string_view_t job_id = {}, f64_t timeout_seconds = 120.0,
        u64_t capabilities = 0) noexcept {
        if (host_domain_count == 0 || host_domain_index >= host_domain_count)
            return expected<mesh_topology_t>::fail( //
                status_t::invalid_input_k,
                "host_domain_index must be in [0,host_domain_count) (host_domain_index={} host_domain_count={})",
                host_domain_index, host_domain_count);
        expected<mesh_shape_t> local = discover_mesh_shape();
        if (!local) return expected<mesh_topology_t>::fail(std::move(local));
        u32_t const local_devices = local.value_.gpus_per_host_domain;

        // The multimem extent is explicit configuration (see the parameter docs); the nesting invariants
        // are enforced by the same make_mesh_shape every hand-configured world goes through.
        if (gpus_per_multimem_domain == 0)
            if (char const *value = std::getenv("UFORM_GPUS_PER_MULTIMEM_DOMAIN"); value != nullptr) // vendor-boundary
                if (!parse_u32_(sz::string_view_t {value}, gpus_per_multimem_domain))
                    return expected<mesh_topology_t>::fail(
                        status_t::invalid_input_k, "$UFORM_GPUS_PER_MULTIMEM_DOMAIN is not a number: \"{}\"", value);
        if (gpus_per_multimem_domain == 0) gpus_per_multimem_domain = local_devices;
        expected<mesh_shape_t> shape = make_mesh_shape(local_devices, gpus_per_multimem_domain,
                                                       local_devices * host_domain_count);
        if (!shape) return expected<mesh_topology_t>::fail(std::move(shape));

        expected<local_probe_t> probed = probe_local_host_domain_(host_domain_index, local_devices);
        if (!probed) return expected<mesh_topology_t>::fail(std::move(probed));

        mesh_topology_t topology;
        topology.shape = shape.value_;
        topology.host_domain_index = host_domain_index;
        topology.poll_pause = poll_pause;
        if (st::failed(topology.host_domains.resize(host_domain_count)))
            return expected<mesh_topology_t>::fail(
                status_t::bad_alloc_k, "host_domain table allocation failed (host_domains={})", host_domain_count);
        topology.host_domains[host_domain_index] = probed.value_.facts;
        if (expected_t pinned = topology.pin_faults_(); !pinned)
            return expected<mesh_topology_t>::fail(std::move(pinned));
        if (host_domain_count == 1) return expected<mesh_topology_t> {std::move(topology)};

        // The exchange root and the job id are explicit configuration; guessing either is how worlds hang
        // (an unshared directory) or how node families collide (an anonymous namespace).
        sz::string_view_t base = directory, job = job_id;
        if (base.empty())
            if (char const *value = std::getenv("UFORM_EXCHANGE_DIR"); value != nullptr) // vendor-boundary
                base = sz::string_view_t {value};
        if (base.empty())
            return expected<mesh_topology_t>::fail( //
                status_t::invalid_input_k,
                "a multi-host_domain bootstrap needs a shared exchange directory: pass one or set " //
                "$UFORM_EXCHANGE_DIR (host_domain_count={})",
                host_domain_count);
        if (job.empty())
            if (char const *value = std::getenv("UFORM_JOB_ID"); value != nullptr) // vendor-boundary
                job = sz::string_view_t {value};
        if (job.empty())
            return expected<mesh_topology_t>::fail( //
                status_t::invalid_input_k,
                "a multi-host_domain bootstrap needs a unique job id: pass one or set $UFORM_JOB_ID " //
                "(host_domain_count={})",
                host_domain_count);
        if (base.contains('\0') || job.contains('\0'))
            return expected<mesh_topology_t>::fail(status_t::invalid_input_k,
                                                   "the exchange directory and the job id hold no NUL byte");
        if (!format_into(topology.exchange_directory, "{}/{}", base, job))
            return expected<mesh_topology_t>::fail(status_t::bad_alloc_k, "exchange path assembly failed");
        if (int const directory_error = make_directories_(topology.exchange_directory.view()); directory_error != 0)
            return expected<mesh_topology_t>::fail(status_t::io_error_k,
                                                   "creating the exchange directory failed (path={} error={})",
                                                   topology.exchange_directory, directory_error);

        sz::string_t facts_text;
        if (!build_facts_(facts_text, probed.value_.facts, topology.shape, job, capabilities,
                          probed.value_.uuids.view()))
            return expected<mesh_topology_t>::fail(status_t::bad_alloc_k, "facts assembly failed (host_domain={})",
                                                   host_domain_index);
        if (expected_t published = topology.publish_text("mesh", facts_text.view()); !published)
            return expected<mesh_topology_t>::fail(std::move(published));

        expected_t exchanged = host_domain_index == 0
                                   ? topology.collect_world_as_lead_(job, probed.value_, capabilities, timeout_seconds)
                                   : topology.adopt_manifest_(job, probed.value_.facts, timeout_seconds);
        if (!exchanged) return expected<mesh_topology_t>::fail(std::move(exchanged));

        if (expected_t synchronized = topology.barrier("mesh", timeout_seconds); !synchronized)
            return expected<mesh_topology_t>::fail(std::move(synchronized));
        return expected<mesh_topology_t> {std::move(topology)};
    }

    /** Claim the next wiring generation: every process makes its transports in one order, so
     *  the @c N th claim names the same phase everywhere, and a phase that runs twice in one
     *  job, like the multimem domain of a probe trainer and of the real one, never reads the
     *  earlier phase's files. */
    [[nodiscard]] u32_t next_collective_id() noexcept { return next_collective_id_++; }

    /** Durably publish this host_domain's @p text under the exchange @p key : written to a
     *  temporary file, @c fsync -ed, then atomically @c rename -d into place, so a peer never reads a
     *  half-written payload. The primitive every wiring phase reuses to add payloads, not machinery. */
    [[nodiscard]] expected_t publish_text(sz::string_view_t key, sz::string_view_t text) const noexcept {
        if (host_domain_count() <= 1) return {};
        expected<sz::string_t> final_path = host_domain_file_path_(host_domain_index, key);
        if (!final_path) return expected_t::fail(std::move(final_path));
        sz::string_t temporary_path;
        if (!format_into(temporary_path, "{}.tmp", final_path.value_))
            return expected_t::fail(status_t::bad_alloc_k, "exchange path assembly failed (key={})", key);
        std::FILE *file = std::fopen(temporary_path.c_str(), "w");
        if (file == nullptr)
            return expected_t::fail(status_t::io_error_k, "opening \"{}\" for writing failed", temporary_path);
        bool const written = std::fwrite(text.data(), 1u, text.size(), file) == text.size() && std::fflush(file) == 0 &&
                             ::fsync(fileno(file)) == 0;
        std::fclose(file);
        if (!written)
            return expected_t::fail(status_t::io_error_k, "writing {} bytes to \"{}\" failed", text.size(),
                                    temporary_path);
        if (std::rename(temporary_path.c_str(), final_path.value_.c_str()) != 0)
            return expected_t::fail(status_t::io_error_k, "renaming \"{}\" into place failed (error={})",
                                    final_path.value_, errno);
        return {};
    }

    /** Poll, through @ref poll_pause , until @p peer_host_domain 's payload under @p key exists,
     *  then read it whole into @p out ; fails naming the full path once @p timeout_seconds pass. */
    [[nodiscard]] expected_t await_text(u32_t peer_host_domain, sz::string_view_t key, sz::string_t &out,
                                        f64_t timeout_seconds) const noexcept {
        expected<sz::string_t> path = host_domain_file_path_(peer_host_domain, key);
        if (!path) return expected_t::fail(std::move(path));
        auto const deadline = std::chrono::steady_clock::now() + std::chrono::duration<f64_t>(timeout_seconds);
        while (true) {
            if (std::FILE *file = std::fopen(path.value_.c_str(), "r"); file != nullptr) {
                char buffer[512];
                out.clear();
                for (std::size_t read_count = 0; (read_count = std::fread(buffer, 1, sizeof(buffer), file)) != 0;)
                    if (sz::failed(out.append(sz::string_view_t {buffer, read_count}))) {
                        std::fclose(file);
                        return expected_t::fail(status_t::bad_alloc_k, "reading \"{}\" exhausted the host heap",
                                                path.value_);
                    }
                std::fclose(file);
                return {};
            }
            if (std::chrono::steady_clock::now() >= deadline)
                return expected_t::fail(status_t::io_error_k, "no payload \"{}\" appeared within {}s", path.value_,
                                        timeout_seconds);
            poll_pause();
        }
    }

    /** A filesystem barrier across every host_domain, named by @p tag , in O(host_domain_count)
     *  file operations: a follower drops its @c <tag>.arrive marker and waits for the lead's
     *  @c <tag>.release ; the lead collects every arrival, then publishes the release. Distinct tags
     *  give independent barriers within one job ( @c "mesh" , @c "connected" , @c "heap" ). A
     *  single-host_domain world returns immediately. */
    [[nodiscard]] expected_t barrier(sz::string_view_t tag, f64_t timeout_seconds) const noexcept {
        if (host_domain_count() <= 1) return {};
        sz::string_t arrive_key, release_key;
        if (!format_into(arrive_key, "{}.arrive", tag) || !format_into(release_key, "{}.release", tag))
            return expected_t::fail(status_t::bad_alloc_k, "barrier key assembly failed (tag={})", tag);
        sz::string_t content;
        if (host_domain_index == 0) {
            for (u32_t peer_index = 1; peer_index != host_domain_count(); ++peer_index)
                if (expected_t arrived = await_text(peer_index, arrive_key.view(), content, timeout_seconds); !arrived)
                    return arrived;
            return publish_text(release_key.view(), "1\n");
        }
        if (expected_t announced = publish_text(arrive_key.view(), "1\n"); !announced) return announced;
        return await_text(0u, release_key.view(), content, timeout_seconds);
    }

  private:

    /** One parsed facts payload: the decimal fields plus views @b borrowing from the payload text. */
    struct facts_t {
        sz::string_view_t schema {};
        sz::string_view_t job {};
        sz::string_view_t hostname {};
        sz::string_view_t device_name {};

        /** The space-separated 32-hex GPU UUIDs (one per device), borrowing from the payload text —
         *  the physical GPU identities the lead proves disjoint across host_domains. */
        sz::string_view_t uuids {};
        u32_t host_domain_index = 0;
        u32_t host_domain_count = 0;
        u32_t multimem = 0;
        u32_t devices = 0;

        /** The NumKong capabilities the process's GPUs dispatch with, after its executor's pin. */
        u64_t capabilities = 0;
    };

    /** One probed physical GPU identity paired with the host_domain that claims it — the flat table
     *  the lead sorts once to prove no GPU is owned twice (an unpartitioned @c CUDA_VISIBLE_DEVICES ). */
    struct claimed_gpu_t {
        CUuuid uuid {};
        u32_t owner = 0;
    };

    /** The local hardware probe: this host_domain's published facts plus the CUDA UUIDs of its
     *  visible GPUs — kept @b probe-local (the world table stores identity, not per-GPU uuids), fed to
     *  @ref build_facts_ on publish and to @ref collect_world_as_lead_ for the uniqueness proof. */
    struct local_probe_t {
        mesh_host_domain_t facts {};
        per_gpu_array<CUuuid> uuids {};
    };

    /** Parse a whole decimal token; @c false on anything but a clean number. */
    [[nodiscard]] static bool parse_u32_(sz::string_view_t token, u32_t &value) noexcept {
        std::from_chars_result const result = std::from_chars(token.data(), token.data() + token.size(), value);
        return result.ec == std::errc {} && result.ptr == token.data() + token.size();
    }

    /** Parse a whole hexadecimal token; @c false on anything but a clean number. */
    [[nodiscard]] static bool parse_hex_u64_(sz::string_view_t token, u64_t &value) noexcept {
        std::from_chars_result const result = std::from_chars(token.data(), token.data() + token.size(), value, 16);
        return result.ec == std::errc {} && result.ptr == token.data() + token.size();
    }

    /** Bounded copy of a token into a fixed char field, always NUL-terminated. */
    static void copy_token_(sz::string_view_t token, std::span<char> destination) noexcept {
        std::size_t const count = token.size() < destination.size() - 1u ? token.size() : destination.size() - 1u;
        std::memcpy(destination.data(), token.data(), count);
        destination[count] = '\0';
    }

    /** Append one GPU UUID as 32 lowercase hex characters — the wire form of a @c CUuuid . */
    [[nodiscard]] static bool format_uuid_(sz::string_t &text, CUuuid const &uuid) noexcept {
        bool built = true;
        for (char const byte : uuid.bytes) built = built && format_into(text, "{:02x}", static_cast<u8_t>(byte));
        return built;
    }

    /** Parse a 32-hex GPU UUID token back into a @c CUuuid ; @c false on any non-hex or short token. */
    [[nodiscard]] static bool parse_uuid_(sz::string_view_t token, CUuuid &uuid) noexcept {
        if (token.size() != 2u * sizeof(uuid.bytes)) return false;
        for (std::size_t byte_index = 0; byte_index != sizeof(uuid.bytes); ++byte_index) {
            char const *pair = token.data() + 2u * byte_index;
            unsigned value = 0;
            std::from_chars_result const result = std::from_chars(pair, pair + 2, value, 16);
            if (result.ec != std::errc {} || result.ptr != pair + 2) return false;
            uuid.bytes[byte_index] = static_cast<char>(value);
        }
        return true;
    }

    /** Probe this host_domain's own facts: hostname plus the GPU model, verified uniform across
     *  the local devices too — the shape describes identical GPUs, and a box with mixed cards would
     *  break every "three integers describe the world" derivation downstream. */
    [[nodiscard]] static expected<local_probe_t> probe_local_host_domain_(u32_t host_domain_index,
                                                                          u32_t device_count) noexcept {
        local_probe_t probe;
        mesh_host_domain_t &self = probe.facts;
        self.host_domain_index = host_domain_index;
        self.device_count = device_count;
        if (::gethostname(self.hostname, sizeof(self.hostname) - 1u) != 0) self.hostname[0] = '\0';
        for (u32_t device_ordinal = 0; device_ordinal != device_count; ++device_ordinal) {
            CUdevice device = 0;
            if (CUresult result = cuDeviceGet(&device, static_cast<int>(device_ordinal)); result != CUDA_SUCCESS)
                return cuda_failure<local_probe_t>(result, "device handle query failed (device={})", device_ordinal);
            char name[device_name_capacity_k] = {};
            if (CUresult result = cuDeviceGetName(name, sizeof(name) - 1u, device); result != CUDA_SUCCESS)
                return cuda_failure<local_probe_t>(result, "device name query failed (device={})", device_ordinal);
            for (char *character = name; *character != '\0'; ++character)
                if (*character == ' ') *character = '_';
            if (device_ordinal == 0) std::memcpy(self.device_name, name, sizeof(name));
            else if (std::strcmp(self.device_name, name) != 0)
                return expected<local_probe_t>::fail(status_t::shape_unsupported_k,
                                                     "mixed local GPUs: device 0 is {}, device {} is {}",
                                                     self.device_name, device_ordinal, name);
            CUuuid uuid {};
            // The globally-unique physical id — the token the lead proves no two host_domains share.
            if (CUresult result = cuDeviceGetUuid(&uuid, device); result != CUDA_SUCCESS)
                return cuda_failure<local_probe_t>(result, "device uuid query failed (device={})", device_ordinal);
            if (probe.uuids.push_back(uuid) != status_t::success_k)
                return expected<local_probe_t>::fail(status_t::shape_unsupported_k,
                                                     "a host_domain drives {} GPUs, over the {} cap", device_count,
                                                     max_gpus_per_host_k);
        }
        return probe;
    }

    /** Assemble the line-oriented facts payload — the exact inverse of @ref parse_facts_ .
     *  @c false only on allocation failure. */
    [[nodiscard]] static bool build_facts_(sz::string_t &text, mesh_host_domain_t const &self,
                                           mesh_shape_t const &shape, sz::string_view_t job, u64_t capabilities,
                                           std::span<CUuuid const> uuids) noexcept {
        bool built = format_into(text,
                                 "schema {}\n"            //
                                 "job {}\n"               //
                                 "host_domain_index {}\n" //
                                 "host_domain_count {}\n" //
                                 "multimem {}\n"          //
                                 "capabilities {:x}\n"    //
                                 "hostname {}\n"          //
                                 "devices {}\n"           //
                                 "device_name {}\n"       //
                                 "uuids",
                                 mesh_schema_k, job, self.host_domain_index, shape.host_domain_count(),
                                 shape.gpus_per_multimem_domain, capabilities, self.hostname, self.device_count,
                                 self.device_name);
        for (CUuuid const &uuid : uuids) built = built && format_into(text, " ") && format_uuid_(text, uuid);
        return built;
    }

    /** Parse a line-oriented @c <key> @c <value> facts payload with the partition API — SIMD
     *  scans, zero allocations, zero exceptions. @c false when any field is missing or malformed;
     *  field-level agreement stays with the caller, which owns the diagnostics. */
    [[nodiscard]] static bool parse_facts_(sz::string_view_t payload, facts_t &facts) noexcept {
        sz::byteset_t const whitespaces = sz::whitespaces_set();
        bool index_parsed = false, count_parsed = false, multimem_parsed = false, devices_parsed = false,
             capabilities_parsed = false;
        for (sz::string_view_t line : payload.split(sz::newlines_set())) {
            auto [key, match, value] = line.partition(' ');
            value = value.strip(whitespaces);
            if (key == "schema") facts.schema = value;
            else if (key == "job") facts.job = value;
            else if (key == "hostname") facts.hostname = value;
            else if (key == "device_name") facts.device_name = value;
            else if (key == "uuids") facts.uuids = value;
            else if (key == "host_domain_index") index_parsed = parse_u32_(value, facts.host_domain_index);
            else if (key == "host_domain_count") count_parsed = parse_u32_(value, facts.host_domain_count);
            else if (key == "multimem") multimem_parsed = parse_u32_(value, facts.multimem);
            else if (key == "capabilities") capabilities_parsed = parse_hex_u64_(value, facts.capabilities);
            else if (key == "devices") devices_parsed = parse_u32_(value, facts.devices);
        }
        return index_parsed && count_parsed && multimem_parsed && capabilities_parsed && devices_parsed &&
               !facts.schema.empty() && !facts.job.empty() && !facts.hostname.empty() && !facts.device_name.empty() &&
               !facts.uuids.empty();
    }

    /** Assemble the manifest payload from a complete table — the exact inverse of
     *  @ref parse_manifest_ . @c false only on allocation failure. */
    [[nodiscard]] static bool build_manifest_(sz::string_t &text, st::basic_vector<mesh_host_domain_t> const &table,
                                              mesh_shape_t const &shape, sz::string_view_t job) noexcept {
        bool built = format_into(text, "schema {}\njob {}\nmultimem {}\n", mesh_schema_k, job,
                                 shape.gpus_per_multimem_domain);
        for (mesh_host_domain_t const &entry : table)
            built = built &&
                    format_into(text, "host_domain {} hostname {} devices {} device_name {}\n", entry.host_domain_index,
                                entry.hostname, entry.device_count, entry.device_name);
        return built;
    }

    /** Parse and validate a whole manifest into @p table : the schema, job, and multimem header
     *  lines must match, every row @c host_domain @c <i> @c hostname @c <h> @c devices @c <d>
     *  @c device_name @c <m> must be well-formed and in range, and exactly @c table.size() rows must
     *  be present. Emits no diagnostics — the caller owns the message, with the whole payload in it. */
    [[nodiscard]] static bool parse_manifest_(sz::string_view_t payload, sz::string_view_t job,
                                              u32_t gpus_per_multimem_domain,
                                              st::basic_vector<mesh_host_domain_t> &table) noexcept {
        auto take_token = [](sz::string_view_t &row) noexcept {
            // Pop the leading space-separated token off the row with the partition API.
            auto [token, match, rest] = row.partition(' ');
            row = rest;
            return token;
        };
        auto parse_row = [&take_token](sz::string_view_t row, mesh_host_domain_t &entry) noexcept {
            if (take_token(row) != "host_domain" || !parse_u32_(take_token(row), entry.host_domain_index)) return false;
            if (take_token(row) != "hostname") return false;
            copy_token_(take_token(row), entry.hostname);
            if (take_token(row) != "devices" || !parse_u32_(take_token(row), entry.device_count)) return false;
            if (take_token(row) != "device_name") return false;
            copy_token_(take_token(row), entry.device_name);
            return entry.hostname[0] != '\0' && entry.device_name[0] != '\0';
        };

        sz::byteset_t const whitespaces = sz::whitespaces_set();
        std::size_t parsed_rows = 0;
        u32_t multimem = 0;
        bool schema_matched = false, job_matched = false, multimem_matched = false;
        for (sz::string_view_t line : payload.split(sz::newlines_set())) {
            auto [key, match, value] = line.partition(' ');
            if (key == "schema") schema_matched = value.strip(whitespaces) == mesh_schema_k;
            else if (key == "job") job_matched = value.strip(whitespaces) == job;
            else if (key == "multimem")
                multimem_matched = parse_u32_(value.strip(whitespaces), multimem) &&
                                   multimem == gpus_per_multimem_domain;
            else if (key == "host_domain") {
                mesh_host_domain_t entry;
                if (!parse_row(line, entry) || entry.host_domain_index >= table.size()) return false;
                table[entry.host_domain_index] = entry;
                ++parsed_rows;
            }
        }
        return schema_matched && job_matched && multimem_matched && parsed_rows == table.size();
    }

    /** The lead's half of the exchange: read every peer's facts, validate agreement against this
     *  host_domain's own @p probed facts, @p self_capabilities and the world @ref shape , fill
     *  @ref host_domains , and publish the merged manifest. */
    [[nodiscard]] expected_t collect_world_as_lead_(sz::string_view_t job, local_probe_t const &probed,
                                                    u64_t self_capabilities, f64_t timeout_seconds) noexcept {
        mesh_host_domain_t const &self = probed.facts;
        st::basic_vector<claimed_gpu_t> claimed;
        // The world's physical GPU identities, this host_domain's first; sorted once at the end to prove
        // none is claimed twice — O(world log world), never a pairwise scan.
        if (st::failed(claimed.reserve(shape.gpus_per_colocated_domain)))
            return expected_t::fail(status_t::bad_alloc_k, "GPU-uniqueness table allocation failed ({} GPUs)",
                                    shape.gpus_per_colocated_domain);
        for (CUuuid const &uuid : probed.uuids.view())
            if (st::failed(claimed.push_back({uuid, 0u})))
                return expected_t::fail(status_t::bad_alloc_k, "GPU-uniqueness table push failed");

        for (u32_t peer_index = 1; peer_index != host_domain_count(); ++peer_index) {
            sz::string_t peer_facts;
            if (expected_t awaited = await_text(peer_index, "mesh", peer_facts, timeout_seconds); !awaited)
                return awaited;
            facts_t facts;
            if (!parse_facts_(peer_facts.view(), facts))
                return expected_t::fail(status_t::io_error_k, "malformed facts from host_domain {}: \"{}\"", peer_index,
                                        peer_facts);
            if (facts.schema != sz::string_view_t {mesh_schema_k})
                return expected_t::fail(status_t::io_error_k, "host_domain {} published schema \"{}\", expected {}",
                                        peer_index, facts.schema, mesh_schema_k);
            if (facts.job != job)
                return expected_t::fail(status_t::io_error_k, "host_domain {} belongs to job \"{}\", expected \"{}\"",
                                        peer_index, facts.job, job);
            if (facts.host_domain_index != peer_index)
                return expected_t::fail(status_t::io_error_k, "host_domain {} published a mismatched index {}",
                                        peer_index, facts.host_domain_index);
            if (facts.host_domain_count != host_domain_count())
                return expected_t::fail( //
                    status_t::shape_unsupported_k,
                    "world-size disagreement: host_domain {} was launched with {} host_domains, host_domain 0 with {}",
                    peer_index, facts.host_domain_count, host_domain_count());
            if (facts.multimem != shape.gpus_per_multimem_domain)
                return expected_t::fail( //
                    status_t::shape_unsupported_k,
                    "multimem-extent disagreement: host_domain {} was launched with {}, host_domain 0 with {}",
                    peer_index, facts.multimem, shape.gpus_per_multimem_domain);
            if (facts.capabilities != self_capabilities)
                return expected_t::fail( //
                    status_t::shape_unsupported_k,
                    "capability disagreement: host_domain {} dispatches with {:#x}, host_domain 0 with {:#x}",
                    peer_index, facts.capabilities, self_capabilities);
            mesh_host_domain_t peer;
            peer.host_domain_index = peer_index;
            peer.device_count = facts.devices;
            copy_token_(facts.hostname, peer.hostname);
            copy_token_(facts.device_name, peer.device_name);
            if (peer.device_count != self.device_count || std::strcmp(peer.device_name, self.device_name) != 0)
                return expected_t::fail( //
                    status_t::shape_unsupported_k,
                    "the mesh must be uniform: host_domain {} ({}) has {}×{}, host_domain 0 ({}) has {}×{}", peer_index,
                    peer.hostname, peer.device_count, peer.device_name, self.hostname, self.device_count,
                    self.device_name);
            std::size_t peer_uuids = 0;
            // The peer's physical GPU identities join the world table; one uuid per claimed device.
            for (sz::string_view_t token : facts.uuids.split(sz::whitespaces_set())) {
                if (token.empty()) continue;
                CUuuid uuid {};
                if (!parse_uuid_(token, uuid))
                    return expected_t::fail(status_t::io_error_k,
                                            "host_domain {} published a malformed GPU uuid \"{}\"", peer_index, token);
                if (st::failed(claimed.push_back({uuid, peer_index})))
                    return expected_t::fail(status_t::bad_alloc_k, "GPU-uniqueness table push failed");
                ++peer_uuids;
            }
            if (peer_uuids != facts.devices)
                return expected_t::fail(status_t::io_error_k, "host_domain {} listed {} GPU uuids for {} devices",
                                        peer_index, peer_uuids, facts.devices);
            host_domains[peer_index] = peer;
        }

        auto const by_uuid = [](claimed_gpu_t const &left, claimed_gpu_t const &right) noexcept {
            return std::memcmp(left.uuid.bytes, right.uuid.bytes, sizeof(left.uuid.bytes)) < 0;
        };
        // No physical GPU is owned by two host_domains — the guard against an unpartitioned
        // CUDA_VISIBLE_DEVICES (co-located processes seeing the same devices). One sort, one adjacent scan.
        std::sort(claimed.data(), claimed.data() + claimed.size(), by_uuid);
        for (std::size_t index = 1; index < claimed.size(); ++index)
            if (std::memcmp(claimed[index].uuid.bytes, claimed[index - 1u].uuid.bytes,
                            sizeof(claimed[index].uuid.bytes)) == 0) {
                sz::string_t uuid_hex;
                (void)format_uuid_(uuid_hex, claimed[index].uuid);
                return expected_t::fail( //
                    status_t::shape_unsupported_k,
                    "GPU {} is claimed by both host_domain {} and host_domain {} — the launcher did not partition " "CU" "DA" "_V" "IS" "IB" "LE" "_D" "EV" "IC" "ES" " a" "cr" "os" "s " "co" "-l" "oc" "at" "ed" " p" "ro" "ce" "ss" "e" "s",
                    uuid_hex, claimed[index - 1u].owner, claimed[index].owner);
            }

        sz::string_t manifest;
        if (!build_manifest_(manifest, host_domains, shape, job))
            return expected_t::fail(status_t::bad_alloc_k, "manifest assembly failed (rows={})", host_domain_count());
        return publish_text("manifest", manifest.view());
    }

    /** A follower's half of the exchange: read the lead's manifest, adopt the world table, and
     *  verify the row describing this host_domain agrees with the local @p probed facts. */
    [[nodiscard]] expected_t adopt_manifest_(sz::string_view_t job, mesh_host_domain_t const &probed,
                                             f64_t timeout_seconds) noexcept {
        sz::string_t manifest;
        if (expected_t awaited = await_text(0u, "manifest", manifest, timeout_seconds); !awaited) return awaited;
        if (!parse_manifest_(manifest.view(), job, shape.gpus_per_multimem_domain, host_domains))
            return expected_t::fail(status_t::io_error_k, "malformed or foreign manifest (host_domain={}): \"{}\"",
                                    host_domain_index, manifest);
        if (self().device_count != probed.device_count || std::strcmp(self().device_name, probed.device_name) != 0)
            return expected_t::fail( //
                status_t::shape_unsupported_k,
                "the manifest disagrees with the local probe (host_domain={}): manifest says {}×{}, probed {}×{}",
                host_domain_index, self().device_count, self().device_name, probed.device_count, probed.device_name);
        return {};
    }

    /** The @c <exchange_directory>/host_domain<N>.<key> path of one host_domain's payload. */
    [[nodiscard]] expected<sz::string_t> host_domain_file_path_(u32_t host_domain,
                                                                sz::string_view_t key) const noexcept {
        sz::string_t path;
        if (!format_into(path, "{}/host_domain{}.{}", exchange_directory, host_domain, key))
            return expected<sz::string_t>::fail(status_t::bad_alloc_k, "exchange path assembly failed (key={})", key);
        return expected<sz::string_t> {std::move(path)};
    }

    /** Create @p path and every missing parent, as @c mkdir @c -p does: zero once all exist, else
     *  the @c errno of the level that failed. */
    [[nodiscard]] static int make_directories_(sz::string_view_t path) noexcept {
        sz::string_t level;
        for (std::size_t end = 1; end <= path.size(); ++end) {
            if (end != path.size() && path[end] != '/') continue;
            if (sz::failed(level.assign(path.substr(0, end)))) return ENOMEM;
            if (::mkdir(level.c_str(), 0777) != 0 && errno != EEXIST) return errno;
        }
        return 0;
    }

    /** Pin the zeroed @ref faults table inside device 0's primary context, retained for as long as
     *  the table lives; unified addressing maps it at the same address on every device. */
    [[nodiscard]] expected_t pin_faults_() noexcept {
        CUdevice device = 0;
        if (CUresult result = cuDeviceGet(&device, 0); result != CUDA_SUCCESS)
            return cuda_failure(result, "device handle query failed (device=0)");
        if (CUresult result = cuDevicePrimaryCtxRetain(&fault_context_, device); result != CUDA_SUCCESS)
            return cuda_failure(result, "primary context retain failed (device=0)");
        if (CUresult result = cuCtxPushCurrent(fault_context_); result != CUDA_SUCCESS)
            return cuda_failure(result, "context push failed (device=0)");
        std::size_t const bytes = shape.gpus_per_host_domain * sizeof(gpu_fault_t);
        void *table = nullptr;
        CUresult const pinned = cuMemHostAlloc(&table, bytes, CU_MEMHOSTALLOC_PORTABLE | CU_MEMHOSTALLOC_DEVICEMAP);
        cuCtxPopCurrent(nullptr);
        if (pinned != CUDA_SUCCESS) return cuda_failure(pinned, "fault table allocation failed ({} bytes)", bytes);
        std::memset(table, 0, bytes);
        faults_ = static_cast<gpu_fault_t *>(table);
        return {};
    }

    void swap_(mesh_topology_t &other) noexcept {
        std::swap(shape, other.shape);
        std::swap(host_domain_index, other.host_domain_index);
        std::swap(host_domains, other.host_domains);
        std::swap(exchange_directory, other.exchange_directory);
        std::swap(poll_pause, other.poll_pause);
        std::swap(next_collective_id_, other.next_collective_id_);
        std::swap(fault_context_, other.fault_context_);
        std::swap(faults_, other.faults_);
    }

    /** Free the fault table, then let go of the context it lives in; a free after a trap killed
     *  that context may fail, and nothing is left to do about it. */
    void destroy_() noexcept {
        if (faults_ != nullptr) {
            cuCtxPushCurrent(fault_context_);
            cuMemFreeHost(faults_);
            cuCtxPopCurrent(nullptr);
            faults_ = nullptr;
        }
        if (fault_context_ != nullptr) {
            CUdevice device = 0;
            if (cuDeviceGet(&device, 0) == CUDA_SUCCESS) cuDevicePrimaryCtxRelease(device);
            fault_context_ = nullptr;
        }
    }

    /** The generation the next wiring phase keys its payloads and barriers with. */
    u32_t next_collective_id_ = 0;

    /** Device 0's primary context, retained while @ref faults_ lives in it. */
    CUcontext fault_context_ = nullptr;

    /** The pinned @ref faults table; null for a topology @ref bootstrap did not make. */
    gpu_fault_t *faults_ = nullptr;
};

/** This host_domain's index in the world, read from whichever launcher spawned the process —
 *  @c $UFORM_HOST_DOMAIN_INDEX , then Slurm's @c $SLURM_NODEID , then Open MPI's
 *  @c $OMPI_COMM_WORLD_RANK ; 0 if none is set. Reading the launcher's environment costs nothing and
 *  links nothing — no MPI library enters the build. Identity only: it selects @b which host_domain,
 *  never a count — the counts are explicit @ref mesh_topology_t::bootstrap arguments, so a
 *  misconfigured launcher cannot change the math. */
[[nodiscard]] inline u32_t host_domain_index_from_environment() noexcept {
    // `std::getenv` speaks C strings; the names never escape this loop.
    char const *const variables[] = {"UFORM_HOST_DOMAIN_INDEX", "SLURM_NODEID", // vendor-boundary
                                     "OMPI_COMM_WORLD_RANK"};                   // vendor-boundary
    for (char const *variable : variables)                                      // vendor-boundary
        if (char const *value = std::getenv(variable); value != nullptr)        // vendor-boundary
            return static_cast<u32_t>(std::atoi(value));
    return 0u;
}

/** The world's host_domain count, read at the @b launcher boundary — @c $UFORM_HOST_DOMAIN_COUNT ,
 *  then Slurm's @c $SLURM_JOB_NUM_NODES , then Open MPI's @c $OMPI_COMM_WORLD_SIZE ; 1 if none is
 *  set. For the mains alone, which pass the count on as @ref mesh_topology_t::bootstrap 's explicit
 *  argument — bootstrap itself never reads it, so a misconfigured launcher cannot change the math
 *  behind an already-constructed world. The whole contract is tabled in @c scripts/slurm/README.md . */
[[nodiscard]] inline u32_t host_domain_count_from_environment() noexcept {
    // `std::getenv` speaks C strings; the names never escape this loop.
    char const *const variables[] = {"UFORM_HOST_DOMAIN_COUNT", "SLURM_JOB_NUM_NODES", // vendor-boundary
                                     "OMPI_COMM_WORLD_SIZE"};
    for (char const *variable : variables)                                 // vendor-boundary
        if (char const *value = std::getenv(variable); value != nullptr) { // vendor-boundary
            u32_t const count = static_cast<u32_t>(std::atoi(value));
            return count != 0u ? count : 1u;
        }
    return 1u;
}

} // namespace unum::uform::core
