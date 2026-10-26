/**
 *  @file include/uform/core/infiniband_collectives.cuh
 *  @author Ash Vardanian
 *  @date July 2, 2026
 *  @brief Device-initiated InfiniBand collectives across multimem_domains — the vertex all-gather and
 *      the force / gradient all-reduce-sum, each an argument-free method of one member's wired
 *      @ref infiniband_reach_t .
 *
 *  The doorbell path is the one NVIDIA documents as GDAKI, GPUDirect Async Kernel-Initiated.
 *
 *  Both collectives walk the same @b recursive-doubling schedule over @c D = multimem_domain_count()
 *  domains: @c ceil(log2 D) rounds, one @ref infiniband_link_t per round, partner
 *  @c this_domain^(1<<round) . Every kernel is a capture-safe graph node — the posts and waits are
 *  device-initiated ( @ref infiniband_post_write / @ref infiniband_load_acquire_u32 ), and per-step variability
 *  reaches replays only through device memory:
 *
 *      - @b all-gather — each round posts the contiguous run of domain blocks this domain already holds
 *        into the partner's symmetric buffer at the @b identical byte offset, so blocks land domain-major
 *        with no accumulate; after @c log2(D) rounds every domain holds all @c D blocks. This is how the
 *        contrastive losses receive remote vertex towers.
 *      - @b all-reduce-sum — each round posts this domain's Float32 shard into the partner's landing
 *        buffer, waits for the partner's, and folds it in with one round-to-nearest-even add per element
 *        ( @c __fadd_rn , no fused-multiply-add contraction) — the SPMD-static schedule fixes the fold
 *        order, so every domain associates identically, bit-for-bit. This is the cross-domain gradient
 *        reduce.
 *
 *  @section collectives_rounds Arrival flags are rounds, never events
 *
 *  Every round's arrival proof is a 4-byte @b inline flag write on the @b same queue pair as the payload:
 *  Reliable-Connection delivery is in-order, so the flag's arrival proves the payload landed. Flags are
 *  @b monotonic never-reset counters — a waiter spins with @c >= against the absolute target
 *  @c *round+round+1 , so replays of a captured step need no flag reset (a reset would be a host
 *  write inside the replay loop). Each @ref infiniband_reach_t owns its round scalar and
 *  @c advance() s it once per step by its round count — distinct members' flags can never alias, because
 *  they live in distinct heaps.
 *
 *  @section infiniband_layers The transport underneath, smallest dependency surface first
 *
 *      1. Wire PODs — @ref infiniband_coordinates_t (what two queue pairs exchange to connect),
 *         @ref infiniband_queue_handles_t / @ref infiniband_completion_handles_t (what a kernel needs
 *         to post and drain), and @ref infiniband_link_t (how one GPU reaches one remote
 *         multimem_domain). These are ibverbs-free and always compiled.
 *      2. @ref infiniband_endpoint_t — one (GPU, NIC) Reliable-Connection queue pair with its memory
 *         region registered over GPU VRAM: dmabuf-first ( @c cuMemGetHandleForAddressRange +
 *         @c ibv_reg_dmabuf_mr ), @c nvidia_peermem @c ibv_reg_mr fallback. Host-posted writes and
 *         completion polling stay as the bring-up / smoke-test path.
 *      3. @ref infiniband_queue_mapping_t / @ref infiniband_completion_mapping_t — the device-initiated
 *         layer NVIDIA documents as GDAKI: @c mlx5dv_init_obj exposes the raw send queue, doorbell
 *         record, and BlueFlame register, and the mapping registers them into the GPU's address space
 *         so @ref infiniband_post_write builds one MLX5 work-queue entry and rings the doorbell
 *         entirely from device code — capture-safe graph nodes, no @c ibv_post_send .
 *      4. @ref infiniband_reach_t — one member's wired reach into the other domains, owning the
 *         registered symmetric heap and running the two collectives above as methods.
 *
 *  @section infiniband_platform Platform contract
 *
 *  LID-routed same-subnet InfiniBand (GID carried but unused), pkey 0, GID index 0 — the single-fabric
 *  cluster this project targets. The BlueFlame register lives in NIC BAR / MMIO space: mapping it for
 *  the GPU needs @c CU_MEMHOSTREGISTER_IOMEMORY , and a platform whose IOMMU / ACS configuration
 *  refuses that registration cannot run the device-initiated path —
 *  @ref infiniband_queue_mapping_t::make names that failure explicitly. CQE compression must be off
 *  (validated at mapping time). Everything below layer 1 builds only under @c UFORM_WITH_INFINIBAND ;
 *  without it the wire PODs remain and every factory fails loudly naming the gate.
 */
#pragma once

#include <cstdint> // `std::uint8_t`, `std::uint16_t`, `std::uint64_t`
#include <cstring> // `std::memcpy`, `std::strcmp`
#include <ctime>   // `nanosleep` — the completion-poll backoff

#include <utility>  // `std::exchange`
#include <span>     // `std::span` — the DMA ring ranges inside the wire PODs
#include <charconv> // `std::from_chars` — coordinate payload parsing
#include <chrono>   // `std::chrono::steady_clock` — the completion-poll deadline

#include <cuda.h>      // `cuMemGetHandleForAddressRange`, `cuMemHostRegister`, `cuMemHostGetDevicePointer`
#include <cuda/memory> // `cuda::align_down`, `cuda::align_up` — the pages covering a host range

#if defined(UFORM_WITH_INFINIBAND)
#include <unistd.h> // `::close`, `sysconf` — dmabuf descriptors and page-aligned registrations

#include <infiniband/mlx5dv.h> // `mlx5dv_init_obj`, `mlx5_wqe_*`, `mlx5_cqe64` — the direct-verbs layer
#include <infiniband/verbs.h>  // `ibv_context`, `ibv_pd`, `ibv_cq`, `ibv_qp` — the RC bring-up
#endif

#include "uform/core/gpu_allocators.hpp" // `gpu_arena_t` backing the device producer / consumer counters
#include "uform/core/mesh.hpp"           // `mesh_topology_t` — the coordinate exchange rides the mesh rendezvous
#include "uform/core/schedule.cuh"       // `b8r1_span_t` — the registered GPU range travels as one typed span
#include "uform/core/types.hpp"          // `expected`, `expected_t`, `status_t`, `format_into`

namespace unum::uform::core {

/** Monotonic never-reset arrival counter the GDAKI collectives spin on; lives in device memory. */
using infiniband_round_t = u32_t;

/** The send-doorbell entry in the mlx5 two-entry doorbell-record array (entry 0 is the receive side). */
inline constexpr std::size_t mlx5_send_doorbell_index_k = 1;

/** The set-consumer-index entry in the mlx5 completion-queue doorbell array (entry 1 arms interrupts). */
inline constexpr std::size_t mlx5_completion_consumer_index_k = 0;

/** Owner-bit mask of an mlx5 completion entry's @c op_own byte; hardware flips it per ring wrap. */
inline constexpr u8_t mlx5_completion_owner_mask_k = 1u;

/** Descriptor-segment count (16-byte units) of an RDMA-write work-queue entry: control + remote
 *  address + data = 3. */
inline constexpr u32_t infiniband_write_descriptor_segments_k = 3u;

/** The mlx5 inline-data marker OR'd into a work-queue entry's inline byte-count header — spelled as
 *  a project constant so the wire PODs stay free of @c <infiniband/mlx5dv.h> . */
inline constexpr u32_t infiniband_inline_data_flag_k = 0x80000000u;

/** What one queue pair publishes so a peer can connect and address its registered GPU buffer — the
 *  wire POD of the RDMA rendezvous, exchanged as text over the mesh's filesystem rendezvous. */
struct infiniband_coordinates_t {

    /** The port's local identifier — the same-subnet InfiniBand route. */
    u32_t local_identifier = 0;

    /** The queue pair number the peer's RTR state targets. */
    u32_t queue_pair_number = 0;

    /** The initial packet sequence number; both ends exchange theirs, reliability needs agreement only. */
    u32_t packet_sequence_number = 0;

    /** The memory region's remote key authorizing the peer's writes and reads. */
    u32_t remote_key = 0;

    /** The registered GPU buffer's base virtual address; a peer targets @c buffer_address+offset . */
    u64_t buffer_address = 0;

    /** The port's global identifier — carried for a future routed fabric, unused on LID routing. */
    u8_t global_identifier[16] = {};

    /** Serialize into the line-oriented exchange payload; @c false only on allocation failure. */
    [[nodiscard]] bool to_text(sz::string_t &text) const noexcept {
        bool built = format_into(text, "lid {}\nqpn {}\npsn {}\nrkey {}\naddr {}\ngid ", local_identifier,
                                 queue_pair_number, packet_sequence_number, remote_key, buffer_address);
        for (u8_t byte : global_identifier) built = built && format_into(text, "{:02x}", byte);
        return built && sz::succeeded(text.push_back('\n'));
    }

    /** Parse the payload @ref to_text produced; @c false when any field is missing or malformed. */
    [[nodiscard]] static bool from_text(sz::string_view_t payload, infiniband_coordinates_t &coordinates) noexcept {
        auto parse_number = [](sz::string_view_t token, auto &value) noexcept {
            std::from_chars_result const result = std::from_chars(token.data(), token.data() + token.size(), value);
            return result.ec == std::errc {} && result.ptr == token.data() + token.size();
        };
        sz::byteset_t const whitespaces = sz::whitespaces_set();
        bool lid_parsed = false, qpn_parsed = false, psn_parsed = false, rkey_parsed = false, addr_parsed = false,
             gid_parsed = false;
        for (sz::string_view_t line : payload.split(sz::newlines_set())) {
            auto [key, match, value] = line.partition(' ');
            value = value.strip(whitespaces);
            if (key == "lid") lid_parsed = parse_number(value, coordinates.local_identifier);
            else if (key == "qpn") qpn_parsed = parse_number(value, coordinates.queue_pair_number);
            else if (key == "psn") psn_parsed = parse_number(value, coordinates.packet_sequence_number);
            else if (key == "rkey") rkey_parsed = parse_number(value, coordinates.remote_key);
            else if (key == "addr") addr_parsed = parse_number(value, coordinates.buffer_address);
            else if (key == "gid") {
                gid_parsed = value.size() == 2 * sizeof(coordinates.global_identifier);
                for (std::size_t byte_index = 0; gid_parsed && byte_index != sizeof(coordinates.global_identifier);
                     ++byte_index) {
                    char const *pair = value.data() + 2 * byte_index;
                    std::from_chars_result const result = std::from_chars(
                        pair, pair + 2, coordinates.global_identifier[byte_index], 16);
                    gid_parsed = result.ec == std::errc {} && result.ptr == pair + 2;
                }
            }
        }
        return lid_parsed && qpn_parsed && psn_parsed && rkey_parsed && addr_parsed && gid_parsed;
    }
};

/** Publish this rank's @p local coordinates under @p key and await + parse the @p peer_host_domain 's
 *  payload under the same key — one QP handshake over the mesh's filesystem rendezvous. */
[[nodiscard]] inline expected_t exchange_infiniband_coordinates(mesh_topology_t const &topology, sz::string_view_t key,
                                                                infiniband_coordinates_t const &local,
                                                                u32_t peer_host_domain, infiniband_coordinates_t &peer,
                                                                f64_t timeout_seconds) noexcept {
    sz::string_t local_text;
    if (!local.to_text(local_text))
        return expected_t::fail(status_t::bad_alloc_k, "coordinate payload assembly failed (key={})", key);
    if (expected_t published = topology.publish_text(key, local_text.view()); !published) return published;
    sz::string_t peer_text;
    if (expected_t awaited = topology.await_text(peer_host_domain, key, peer_text, timeout_seconds); !awaited)
        return awaited;
    if (!infiniband_coordinates_t::from_text(peer_text.view(), peer))
        return expected_t::fail(status_t::io_error_k, "malformed coordinates from host_domain {} (key={}): \"{}\"",
                                peer_host_domain, key, peer_text);
    return {};
}

/** The pointer-and-scalar handles a kernel needs to post one work-queue entry and ring the
 *  doorbell — a trivially-copyable POD passed by value; the owning @ref infiniband_queue_mapping_t fills
 *  it. All pointers are in the GPU's address space. */
struct infiniband_queue_handles_t {

    /** The send work-queue entry storage (host DMA memory the NIC fetches from), spanning
     *  @c send_queue_entry_count·send_queue_stride bytes. */
    std::span<std::byte> send_queue {};

    /** The big-endian doorbell-record array; the send entry holds the producer counter the NIC polls. */
    u32_t *doorbell_record = nullptr;

    /** The BlueFlame / User-Access-Region doorbell register — NIC BAR / MMIO space. */
    void *blueflame_register = nullptr;

    /** Device counter of posted entries (the producer index); atomically claimed per post. */
    u32_t *producer_index = nullptr;

    /** The queue pair number, baked into every control segment. */
    u32_t queue_pair_number = 0;

    /** Send-queue capacity in basic blocks; the producer index wraps modulo this. */
    u32_t send_queue_entry_count = 0;

    /** Bytes per basic block (64 on ConnectX-7); the entry stride in the buffer. */
    u32_t send_queue_stride = 0;
};

/**
 *  @brief The handles a kernel needs to reap one signaled send completion — the drain counterpart of
 *      @ref infiniband_queue_handles_t .
 *
 *  The three hardware unknowns ( @c entry_size , @c initial_owner , @c compressed ) are captured at
 *  mapping time; the valid 64-byte entry lives in the @b high bytes of a wider cell.
 */
struct infiniband_completion_handles_t {

    /** The completion-entry ring (host DMA memory the NIC writes into), spanning
     *  @c entry_count·entry_size bytes. */
    std::span<std::byte> ring {};

    /** The big-endian completion doorbell record; set-consumer-index at entry 0. */
    u32_t *doorbell_record = nullptr;

    /** Device counter of reaped completions (the consumer index), in GPU VRAM. */
    u32_t *consumer_index = nullptr;

    /** Ring capacity in entries — a power of two; the consumer index masks modulo it. */
    u32_t entry_count = 0;

    /** Bytes per ring cell (64 or 128); the valid 64-byte entry sits in the high bytes. */
    u32_t entry_size = 0;

    /** @c log2(entry_count) ; the owner-bit polarity at index @c i is @c (i>>entry_log_count)&1 . */
    u32_t entry_log_count = 0;

    /** The completion queue number, carried for diagnostics. */
    u32_t completion_queue_number = 0;

    /** Owner-bit polarity of a valid entry at consumer index zero. */
    u32_t initial_owner = 0;

    /** 1 when entry compression is active — the mapping requires 0 (uncompressed). */
    u32_t compressed = 0;
};

/**
 *  @brief How one GPU reaches one @b remote multimem_domain's symmetric heap: the device-initiated send
 *      path, the peer's landing coordinates, and the local arrival flag — one link per remote domain per
 *      recursive-doubling round, never one per remote GPU.
 *
 *  Symmetric heaps lay identical offsets on every rank, so any tensor's remote target is
 *  @c remote_heap_base+offset with the @b local offset — no per-tensor lookup. Built once at wiring time,
 *  outside capture, from an @ref infiniband_endpoint_t + the GDAKI mappings + exchanged coordinates.
 */
struct infiniband_link_t {

    /** Which remote multimem_domain this link reaches. */
    u32_t remote_multimem_domain_index = 0;

    /** Recursive-doubling round this link serves: the partner is @c this_domain^(1<<round_index) . */
    u32_t round_index = 0;

    /** Peer domain's symmetric-heap base virtual address; targets are @c base+offset . */
    u64_t remote_heap_base = 0;

    /** Remote key authorizing writes into the peer heap. */
    u32_t remote_access_key = 0;

    /** Local key authorizing the adapter to source from this rank's registered heap. */
    u32_t local_access_key = 0;

    /** The device-initiated send path of this link's queue pair. */
    infiniband_queue_handles_t queue {};

    /** The completion parser paired with @ref queue ; one drain per signaled post. */
    infiniband_completion_handles_t completion {};

    /** Symmetric-heap byte offset of the even-step landing buffer, the partner's and this rank's alike. */
    std::size_t landing_offset = 0;

    /** Byte stride from a round's even-step landing to its odd-step twin — the parity doubling
     *  that keeps a fast partner's next-step write off a buffer the slower side still folds; zero
     *  when the payload doubles as the landing (the all-gather). */
    u32_t landing_parity_stride = 0;

    /** Symmetric-heap byte offset of the partner's even-step arrival flag; the odd twin sits one
     *  @c u32_t further. */
    std::size_t remote_flag_offset = 0;

    /** This rank's monotonic even-step arrival flag (device VRAM) the waiting kernel
     *  acquire-spins on; the odd-step twin is the next @c u32_t . */
    infiniband_round_t *arrival_flag = nullptr;

    /** How long the flag wait and the completion drain spin before they trap. */
    u64_t wait_nanoseconds = 0;
};

#if defined(UFORM_WITH_INFINIBAND)

/** One InfiniBand access-flag set shared by the memory region and the queue pair. */
inline constexpr int infiniband_access_flags_k = IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_WRITE |
                                                 IBV_ACCESS_REMOTE_READ;

/** Completion-queue and work-request depth; the host path posts a handful of writes per exchange. */
inline constexpr int infiniband_queue_depth_k = 16;

/** Whether a host-posted write also carries immediate data (consuming a posted receive on the peer). */
enum class infiniband_signal_t : u32_t {
    plain_signaled_k = 0,
    with_immediate_k = 1,
};

/** Which kind of host memory a page registration covers: ordinary DMA memory, or NIC BAR space such as
 *  the BlueFlame register, which the driver registers as I/O memory. */
enum class host_memory_t : bool { dma_k, mmio_k };

/** One registered host page and its reference count — mlx5 packs many doorbell records (and
 *  neighboring rings) per OS page, so several mappings routinely share one page. A page someone
 *  outside the registry registered first is @b borrowed ( @c owned false) and never unregistered. */
struct host_page_registration_t {

    /** The page base; null marks a free table entry reused by the next @c acquire . */
    void *page = nullptr;

    /** Live acquisitions covering this page; the last @c release unregisters an owned page. */
    u32_t references = 0;

    /** Whether this registry performed the registration — only owned pages are unregistered. */
    bool owned = false;
};

/**
 *  @brief The NIC's ledger of host pages registered for GPU access — owned by @ref infiniband_nic_t ,
 *      borrowed by every queue / completion mapping on that NIC.
 *
 *  mlx5 allocates work queues, completion rings, and doorbell records from page-shared pools, so two
 *  mappings routinely land on one OS page. An unshared @c cuMemHostRegister per mapping then either fails
 *  ( @c CUDA_ERROR_HOST_MEMORY_ALREADY_REGISTERED , leaving a range's tail pages unmapped for the GPU) or,
 *  worse, the first mapping's teardown unregisters a page the second mapping's doorbell kernel still
 *  dereferences — a roaming device fault (Xid 43). Registering page by page with reference counts makes
 *  both impossible: a shared page is counted, a partially-covered range still gets its tail pages, and a
 *  page lives until its @b last user releases it.
 */
struct host_page_registry_t {

    /**
     *  @brief Register every page covering @c [host_pointer,host_pointer+span) for GPU access — counting
     *      pages other mappings already hold — and resolve @p device_pointer for @p host_pointer .
     *  @param[in] host_pointer The NIC structure's host address; any alignment.
     *  @param[in] span The structure's byte extent.
     *  @param[in] memory NIC BAR space (the BlueFlame register) versus ordinary DMA memory.
     *  @param[out] device_pointer The GPU-dereferenceable address for @p host_pointer .
     */
    [[nodiscard]] expected_t acquire(void *host_pointer, std::size_t span, host_memory_t memory,
                                     CUdeviceptr &device_address) noexcept {
        device_address = 0;
        std::size_t const page_bytes = static_cast<std::size_t>(sysconf(_SC_PAGESIZE));
        b8_t *const first = cuda::align_down(static_cast<b8_t *>(host_pointer), page_bytes);
        b8_t *const last = cuda::align_up(static_cast<b8_t *>(host_pointer) + span, page_bytes);
        unsigned const flags = CU_MEMHOSTREGISTER_DEVICEMAP |
                               (memory == host_memory_t::mmio_k ? CU_MEMHOSTREGISTER_IOMEMORY : 0u);
        for (b8_t *page = first; page != last; page += page_bytes) {
            if (host_page_registration_t *held = find_(page); held != nullptr) {
                ++held->references;
                continue;
            }
            host_page_registration_t entry {page, 1u, false};
            CUresult const registered = cuMemHostRegister(entry.page, page_bytes, flags);
            if (registered == CUDA_SUCCESS) entry.owned = true;
            else if (registered != CUDA_ERROR_HOST_MEMORY_ALREADY_REGISTERED) {
                release(first, static_cast<std::size_t>(page - first)); // exactly the pages acquired so far
                return cuda_failure(registered, "host page registration failed ({} memory, page={})",
                                    memory == host_memory_t::mmio_k ? "NIC MMIO" : "DMA", entry.page);
            }
            if (status_t inserted = insert_(entry); inserted != status_t::success_k) {
                if (entry.owned) cuMemHostUnregister(entry.page);
                release(first, static_cast<std::size_t>(page - first));
                return expected_t::fail(inserted, "page-registration table allocation failed");
            }
        }
        if (CUresult result = cuMemHostGetDevicePointer(&device_address, host_pointer, 0); result != CUDA_SUCCESS) {
            release(host_pointer, span);
            return cuda_failure(result, "device-pointer resolution failed");
        }
        return {};
    }

    /** Drop one reference from every page covering @c [host_pointer,host_pointer+span) — the exact
     *  range of a prior @ref acquire ; the last reference unregisters an owned page. A zero @p span
     *  covers no page and is the no-op an unwound partial acquire relies on. */
    void release(void *host_pointer, std::size_t span) noexcept {
        if (host_pointer == nullptr) return;
        std::size_t const page_bytes = static_cast<std::size_t>(sysconf(_SC_PAGESIZE));
        b8_t *const first = cuda::align_down(static_cast<b8_t *>(host_pointer), page_bytes);
        b8_t *const last = cuda::align_up(static_cast<b8_t *>(host_pointer) + span, page_bytes);
        for (b8_t *page = first; page != last; page += page_bytes) {
            host_page_registration_t *held = find_(page);
            if (held == nullptr || held->references == 0) continue; // release without acquire — nothing to drop
            if (--held->references == 0) {
                if (held->owned) cuMemHostUnregister(held->page);
                *held = host_page_registration_t {}; // the entry becomes free for reuse
            }
        }
    }

  private:

    /** The live entry for @p page , or null. */
    [[nodiscard]] host_page_registration_t *find_(void *page) noexcept {
        for (host_page_registration_t &entry : pages_)
            if (entry.page == page) return &entry;
        return nullptr;
    }

    /** Record @p entry in a free table entry, growing the table when none is free. */
    [[nodiscard]] status_t insert_(host_page_registration_t entry) noexcept {
        for (host_page_registration_t &free_entry : pages_)
            if (free_entry.page == nullptr) {
                free_entry = entry;
                return status_t::success_k;
            }
        return st::failed(pages_.push_back(std::move(entry))) ? status_t::bad_alloc_k : status_t::success_k;
    }

    /** The page table — a handful of entries, walked linearly. */
    st::basic_vector<host_page_registration_t> pages_ {};
};

/** One acquired page range a mapping must release — the exact @c (host,span) it passed to
 *  @ref host_page_registry_t::acquire ; a null host is the empty range. */
struct host_page_range_t {

    /** The NIC structure's host address, as acquired. */
    void *host = nullptr;

    /** The structure's byte extent, as acquired. */
    std::size_t span = 0;
};

/**
 *  @brief One opened mlx5 adapter serving one GPU: the device context, the protection domain, the page
 *      registry, and the port identity (LID, GID, MTU) every endpoint on this GPU borrows — opened
 *      @b once, in main, never per queue pair.
 *
 *  Move-only; teardown deallocates the domain and closes the device.
 */
struct infiniband_nic_t {

    infiniband_nic_t() noexcept = default;
    infiniband_nic_t(infiniband_nic_t const &) = delete;
    infiniband_nic_t &operator=(infiniband_nic_t const &) = delete;
    infiniband_nic_t(infiniband_nic_t &&other) noexcept
        : device_context_(std::exchange(other.device_context_, nullptr)),
          protection_domain_(std::exchange(other.protection_domain_, nullptr)),
          primary_context_(std::exchange(other.primary_context_, nullptr)), paired_gpu_(other.paired_gpu_),
          page_registry_(std::move(other.page_registry_)), local_identifier_(other.local_identifier_),
          path_mtu_(other.path_mtu_), port_(other.port_), global_identifier_index_(other.global_identifier_index_) {
        std::memcpy(global_identifier_, other.global_identifier_, sizeof(global_identifier_));
    }
    infiniband_nic_t &operator=(infiniband_nic_t &&other) noexcept {
        if (this != &other) {
            destroy_();
            device_context_ = std::exchange(other.device_context_, nullptr);
            protection_domain_ = std::exchange(other.protection_domain_, nullptr);
            primary_context_ = std::exchange(other.primary_context_, nullptr);
            paired_gpu_ = other.paired_gpu_;
            page_registry_ = std::move(other.page_registry_);
            local_identifier_ = other.local_identifier_;
            path_mtu_ = other.path_mtu_;
            port_ = other.port_;
            global_identifier_index_ = other.global_identifier_index_;
            std::memcpy(global_identifier_, other.global_identifier_, sizeof(global_identifier_));
        }
        return *this;
    }
    ~infiniband_nic_t() noexcept { destroy_(); }

    /**
     *  @brief Open the adapter serving this host_domain's @b local GPU ordinal @p local_gpu_index :
     *      the one sharing the deepest PCIe switch with that GPU.
     *
     *  A card's functions tie on depth, so the first whose port is active and can route wins: one
     *  with a LID on InfiniBand, one with a RoCEv2 GID carrying an IPv4 address on Ethernet.
     *  Allocates the protection domain and caches the port identity.
     *
     *  @param[in] infiniband_port The port on the adapter; single-port ConnectX parts use 1.
     */
    [[nodiscard]] static expected<infiniband_nic_t> make(u32_t local_gpu_index, u8_t infiniband_port = 1) noexcept;

    /** The opened device context endpoints create queues against — borrowed, still owned here. */
    [[nodiscard]] ibv_context *context() const noexcept { return device_context_; }

    /** The shared protection domain endpoints register regions and queue pairs in. */
    [[nodiscard]] ibv_pd *protection_domain() const noexcept { return protection_domain_; }

    /** The port's local identifier — the same-subnet InfiniBand route. */
    [[nodiscard]] u32_t local_identifier() const noexcept { return local_identifier_; }

    /** The port's active MTU, baked into every queue pair's RTR transition. */
    [[nodiscard]] unsigned path_mtu() const noexcept { return path_mtu_; }

    /** The port number every queue pair binds. */
    [[nodiscard]] u8_t port() const noexcept { return port_; }

    /** The 16-byte port GID every queue pair's global route header names. */
    [[nodiscard]] u8_t const *global_identifier() const noexcept { return global_identifier_; }

    /** The GID table entry @ref global_identifier came from, the source of every route header. */
    [[nodiscard]] u8_t global_identifier_index() const noexcept { return global_identifier_index_; }

    /** The NIC's page-registration ledger every mapping on this NIC acquires through — mlx5's
     *  page-shared pools make the reference counting mandatory (see @ref host_page_registry_t ). */
    [[nodiscard]] host_page_registry_t &page_registry() noexcept { return page_registry_; }

  private:
    void destroy_() noexcept {
        if (protection_domain_ != nullptr) ibv_dealloc_pd(std::exchange(protection_domain_, nullptr));
        if (device_context_ != nullptr) ibv_close_device(std::exchange(device_context_, nullptr));
        if (primary_context_ != nullptr) {
            primary_context_ = nullptr;
            cuDevicePrimaryCtxRelease(paired_gpu_);
        }
    }

    /** The opened mlx5 device. */
    ibv_context *device_context_ = nullptr;

    /** The one protection domain every endpoint on this GPU shares. */
    ibv_pd *protection_domain_ = nullptr;

    /** The paired GPU's primary context, retained for the NIC's lifetime — the page registrations
     *  below are scoped to it, and a transient wiring stream releasing the last reference would reset
     *  the context and destroy them under the doorbell kernels. */
    CUcontext primary_context_ = nullptr;

    /** The paired GPU the retained context (and every registration) belongs to. */
    CUdevice paired_gpu_ = 0;

    /** The page-registration ledger — outlives every mapping on this NIC, exactly like the
     *  protection domain above. */
    host_page_registry_t page_registry_ {};

    /** Cached port identity — queried once at @ref make . */
    u32_t local_identifier_ = 0;

    /** The port's active MTU. */
    unsigned path_mtu_ = 0;

    /** The bound port number. */
    u8_t port_ = 1;

    /** The GID table entry the port routes from. */
    u8_t global_identifier_index_ = 0;

    /** The port's global identifier. */
    u8_t global_identifier_[16] = {};
};

/**
 *  @brief One Reliable-Connection endpoint talking to one remote GPU: completion queue, RC queue pair,
 *      and a memory region over GPU VRAM — all inside a borrowed @ref infiniband_nic_t 's device
 *      context and protection domain.
 *
 *  The region registers dmabuf-first ( @c cuMemGetHandleForAddressRange exports the range,
 *  @c ibv_reg_dmabuf_mr registers the descriptor with the GPU virtual address as the I/O address so peer
 *  @c buffer_address+offset math lands on the right byte); on any failure it falls back to a plain
 *  @c ibv_reg_mr serviced by the @c nvidia_peermem kernel module. Host-posted writes and completion polls
 *  remain the bring-up / smoke-test path; production traffic runs through the GDAKI mappings layered on
 *  this endpoint's queue pair. Move-only; teardown reverses construction: region, queue pair, completion
 *  queue, dmabuf descriptor — the adapter stays with its NIC owner.
 */
struct infiniband_endpoint_t {

    infiniband_endpoint_t() noexcept = default;
    infiniband_endpoint_t(infiniband_endpoint_t const &) = delete;
    infiniband_endpoint_t &operator=(infiniband_endpoint_t const &) = delete;

    infiniband_endpoint_t(infiniband_endpoint_t &&other) noexcept
        : completion_queue_(std::exchange(other.completion_queue_, nullptr)),
          queue_pair_(std::exchange(other.queue_pair_, nullptr)),
          memory_region_(std::exchange(other.memory_region_, nullptr)),
          gpu_buffer_(std::exchange(other.gpu_buffer_, b8r1_span_t {})),
          dmabuf_descriptor_(std::exchange(other.dmabuf_descriptor_, -1)), infiniband_port_(other.infiniband_port_),
          global_identifier_index_(other.global_identifier_index_), path_mtu_(other.path_mtu_),
          local_coordinates_(other.local_coordinates_), peer_coordinates_(other.peer_coordinates_) {}
    infiniband_endpoint_t &operator=(infiniband_endpoint_t &&other) noexcept {
        if (this != &other) {
            destroy_();
            completion_queue_ = std::exchange(other.completion_queue_, nullptr);
            queue_pair_ = std::exchange(other.queue_pair_, nullptr);
            memory_region_ = std::exchange(other.memory_region_, nullptr);
            gpu_buffer_ = std::exchange(other.gpu_buffer_, b8r1_span_t {});
            dmabuf_descriptor_ = std::exchange(other.dmabuf_descriptor_, -1);
            infiniband_port_ = other.infiniband_port_;
            global_identifier_index_ = other.global_identifier_index_;
            path_mtu_ = other.path_mtu_;
            local_coordinates_ = other.local_coordinates_;
            peer_coordinates_ = other.peer_coordinates_;
        }
        return *this;
    }
    ~infiniband_endpoint_t() noexcept { destroy_(); }

    /**
     *  @brief Build the completion queue + RC queue pair inside @p nic 's protection domain and
     *      register @p gpu_buffer as a GPUDirect memory region — everything a peer handshake needs.
     *
     *  @param[in] nic The GPU's opened adapter — borrowed for this endpoint's whole lifetime.
     *  @param[in] gpu_buffer The GPU VRAM range peers will read and write, shaped @b [buffer_bytes] —
     *      typically a window over a @ref gpu_arena_t (its chunks are created GPUDirect-RDMA-capable).
     */
    [[nodiscard]] static expected<infiniband_endpoint_t> make(infiniband_nic_t &nic, b8r1_span_t gpu_buffer) noexcept;

    /** What this endpoint publishes for the peer's @ref connect . */
    [[nodiscard]] infiniband_coordinates_t const &local_coordinates() const noexcept { return local_coordinates_; }

    /** The queue pair the GDAKI mapping layers onto — borrowed, still owned here. */
    [[nodiscard]] ibv_qp *queue_pair() const noexcept { return queue_pair_; }

    /** The completion queue the GDAKI mapping layers onto — borrowed, still owned here. */
    [[nodiscard]] ibv_cq *completion_queue() const noexcept { return completion_queue_; }

    /** The registered memory region's local key — the source authorization every post carries. */
    [[nodiscard]] u32_t local_key() const noexcept { return memory_region_->lkey; }

    /**
     *  @brief Drive the queue pair to Ready-To-Send against @p peer : INIT (port + pkey + access) → RTR
     *      (peer queue-pair number and sequence number over an LID-routed address vector) → RTS.
     *
     *  Infinite receiver-not-ready retries: a write racing ahead of the peer's posted receive is retried,
     *  never dropped — the host-side handshake needs no strict ordering.
     */
    [[nodiscard]] expected_t connect(infiniband_coordinates_t const &peer) noexcept;

    /** Host-post one signaled RDMA write of @p length bytes from @c gpu_buffer+local_offset into
     *  the peer's @c buffer_address+remote_offset — the bring-up / smoke-test path; production posts
     *  ride @ref infiniband_post_write . */
    [[nodiscard]] expected_t post_write(std::size_t local_offset, std::size_t remote_offset, std::size_t length,
                                        infiniband_signal_t signal = infiniband_signal_t::plain_signaled_k) noexcept {
        if (length == 0 || local_offset + length > gpu_buffer_.extent(0))
            return expected_t::fail(status_t::invalid_input_k,
                                    "write of {} bytes at local offset {} exceeds the registered {} bytes", length,
                                    local_offset, gpu_buffer_.extent(0));
        ibv_sge scatter_gather {};
        scatter_gather.addr = device_address(gpu_buffer_.data()) + local_offset;
        scatter_gather.length = static_cast<u32_t>(length);
        scatter_gather.lkey = memory_region_->lkey;

        ibv_send_wr work_request {};
        work_request.sg_list = &scatter_gather;
        work_request.num_sge = 1;
        work_request.opcode = signal == infiniband_signal_t::with_immediate_k ? IBV_WR_RDMA_WRITE_WITH_IMM
                                                                              : IBV_WR_RDMA_WRITE;
        work_request.send_flags = IBV_SEND_SIGNALED;
        work_request.wr.rdma.remote_addr = peer_coordinates_.buffer_address + remote_offset;
        work_request.wr.rdma.rkey = peer_coordinates_.remote_key;
        ibv_send_wr *rejected = nullptr;
        if (ibv_post_send(queue_pair_, &work_request, &rejected) != 0)
            return expected_t::fail(status_t::io_error_k, "posting the RDMA write failed ({} bytes)", length);
        return {};
    }

    /** Post a zero-scatter receive so a peer's write-with-immediate completion can be consumed. */
    [[nodiscard]] expected_t post_receive() noexcept {
        ibv_recv_wr work_request {};
        ibv_recv_wr *rejected = nullptr;
        if (ibv_post_recv(queue_pair_, &work_request, &rejected) != 0)
            return expected_t::fail(status_t::io_error_k, "posting the receive failed");
        return {};
    }

    /** Spin-poll (50 µs backoff) until one completion is reaped or @p timeout_seconds elapses. */
    [[nodiscard]] expected_t poll_completion(f64_t timeout_seconds) noexcept {
        auto const deadline = std::chrono::steady_clock::now() + std::chrono::duration<f64_t>(timeout_seconds);
        while (true) {
            ibv_wc completion {};
            int const reaped = ibv_poll_cq(completion_queue_, 1, &completion);
            if (reaped < 0) return expected_t::fail(status_t::io_error_k, "polling the completion queue failed");
            if (reaped > 0) {
                if (completion.status != IBV_WC_SUCCESS)
                    return expected_t::fail(status_t::io_error_k, "completion failed (status={} vendor={})",
                                            static_cast<int>(completion.status), completion.vendor_err);
                return {};
            }
            if (std::chrono::steady_clock::now() >= deadline)
                return expected_t::fail(status_t::io_error_k, "no completion within {}s", timeout_seconds);
            timespec const backoff {0, 50'000};
            nanosleep(&backoff, nullptr);
        }
    }

  private:

    /** Register the GPU buffer as a GPUDirect region in @p protection_domain : dmabuf-first,
     *  @c nvidia_peermem fallback. */
    [[nodiscard]] expected_t register_gpu_memory_(ibv_pd *protection_domain) noexcept;

    void destroy_() noexcept {
        if (memory_region_ != nullptr) ibv_dereg_mr(std::exchange(memory_region_, nullptr));
        if (queue_pair_ != nullptr) ibv_destroy_qp(std::exchange(queue_pair_, nullptr));
        if (completion_queue_ != nullptr) ibv_destroy_cq(std::exchange(completion_queue_, nullptr));
        if (dmabuf_descriptor_ >= 0) ::close(std::exchange(dmabuf_descriptor_, -1));
        gpu_buffer_ = b8r1_span_t {};
    }

    /** This endpoint's completion queue — its own, so drains never cross lanes. */
    ibv_cq *completion_queue_ = nullptr;

    /** The Reliable-Connection queue pair — the single-poster concurrency granularity. */
    ibv_qp *queue_pair_ = nullptr;

    /** The GPUDirect memory region over @ref gpu_buffer_ . */
    ibv_mr *memory_region_ = nullptr;

    /** The registered GPU VRAM range peers read and write. */
    b8r1_span_t gpu_buffer_ {};

    /** The exported dmabuf descriptor when that registration path won; -1 on the fallback. */
    int dmabuf_descriptor_ = -1;

    /** The bound port, from the owning NIC. */
    u8_t infiniband_port_ = 1;

    /** The GID table entry route headers leave from, from the owning NIC. */
    u8_t global_identifier_index_ = 0;

    /** The port's active MTU, from the owning NIC. */
    unsigned path_mtu_ = 0;

    /** What this endpoint publishes for the peer's @ref connect . */
    infiniband_coordinates_t local_coordinates_ {};

    /** The connected peer's coordinates, kept for host-posted writes. */
    infiniband_coordinates_t peer_coordinates_ {};
};

/**
 *  @brief The GPU-reachable send path of one queue pair: the work-queue buffer, the doorbell record, and
 *      the BlueFlame register mapped into GPU address space, plus a device producer-index counter.
 *
 *  Layers onto a built, connected @ref infiniband_endpoint_t 's queue pair (borrowed, never owned) — build it
 *  once after Ready-To-Send, @b outside any capture. The BlueFlame registration
 *  ( @c CU_MEMHOSTREGISTER_IOMEMORY over NIC BAR space) is @b the platform gate for device-initiated
 *  GDAKI; its failure is reported by name.
 */
struct infiniband_queue_mapping_t {

    infiniband_queue_mapping_t() noexcept = default;
    infiniband_queue_mapping_t(infiniband_queue_mapping_t const &) = delete;
    infiniband_queue_mapping_t &operator=(infiniband_queue_mapping_t const &) = delete;
    infiniband_queue_mapping_t(infiniband_queue_mapping_t &&other) noexcept
        : handles_(std::exchange(other.handles_, infiniband_queue_handles_t {})),
          registry_(std::exchange(other.registry_, nullptr)),
          send_queue_range_(std::exchange(other.send_queue_range_, host_page_range_t {})),
          doorbell_range_(std::exchange(other.doorbell_range_, host_page_range_t {})),
          blueflame_range_(std::exchange(other.blueflame_range_, host_page_range_t {})) {}
    infiniband_queue_mapping_t &operator=(infiniband_queue_mapping_t &&other) noexcept {
        if (this != &other) {
            destroy_();
            handles_ = std::exchange(other.handles_, infiniband_queue_handles_t {});
            registry_ = std::exchange(other.registry_, nullptr);
            send_queue_range_ = std::exchange(other.send_queue_range_, host_page_range_t {});
            doorbell_range_ = std::exchange(other.doorbell_range_, host_page_range_t {});
            blueflame_range_ = std::exchange(other.blueflame_range_, host_page_range_t {});
        }
        return *this;
    }
    ~infiniband_queue_mapping_t() noexcept { destroy_(); }

    /** Expose @p queue_pair 's raw mlx5 send queue / doorbell / BlueFlame register and map all
     *  three for the GPU through @p registry (pages are shared across queue pairs, so acquisitions
     *  are reference-counted); the producer counter is allocated from @p arena (capture-stable). */
    [[nodiscard]] static expected<infiniband_queue_mapping_t> make(ibv_qp *queue_pair, gpu_arena_t &arena,
                                                                   host_page_registry_t &registry) noexcept;

    /** A by-value snapshot for kernels and @ref infiniband_post_write . */
    [[nodiscard]] infiniband_queue_handles_t device_handles() const noexcept { return handles_; }

  private:
    void destroy_() noexcept {
        if (registry_ != nullptr) {
            for (host_page_range_t *range : {&blueflame_range_, &doorbell_range_, &send_queue_range_}) {
                registry_->release(range->host, range->span);
                *range = host_page_range_t {};
            }
            registry_ = nullptr;
        }
        handles_ = infiniband_queue_handles_t {};
    }

    infiniband_queue_handles_t handles_ {};

    /** The owning NIC's page ledger — borrowed; outlives the mapping like the protection domain. */
    host_page_registry_t *registry_ = nullptr;

    /** The send queue's acquired range, released through @ref registry_ at teardown. */
    host_page_range_t send_queue_range_ {};

    /** The doorbell record's acquired range — routinely a page shared with other queue pairs. */
    host_page_range_t doorbell_range_ {};

    /** The BlueFlame register's acquired range — NIC BAR space. */
    host_page_range_t blueflame_range_ {};
};

/** The GPU-reachable completion path of one queue pair: the completion ring and its doorbell
 *  record mapped for the GPU (both ordinary host DMA memory), plus a device consumer-index counter
 *  and the hardware geometry captured at build time. */
struct infiniband_completion_mapping_t {

    infiniband_completion_mapping_t() noexcept = default;
    infiniband_completion_mapping_t(infiniband_completion_mapping_t const &) = delete;
    infiniband_completion_mapping_t &operator=(infiniband_completion_mapping_t const &) = delete;
    infiniband_completion_mapping_t(infiniband_completion_mapping_t &&other) noexcept
        : handles_(std::exchange(other.handles_, infiniband_completion_handles_t {})),
          registry_(std::exchange(other.registry_, nullptr)),
          ring_range_(std::exchange(other.ring_range_, host_page_range_t {})),
          doorbell_range_(std::exchange(other.doorbell_range_, host_page_range_t {})) {}
    infiniband_completion_mapping_t &operator=(infiniband_completion_mapping_t &&other) noexcept {
        if (this != &other) {
            destroy_();
            handles_ = std::exchange(other.handles_, infiniband_completion_handles_t {});
            registry_ = std::exchange(other.registry_, nullptr);
            ring_range_ = std::exchange(other.ring_range_, host_page_range_t {});
            doorbell_range_ = std::exchange(other.doorbell_range_, host_page_range_t {});
        }
        return *this;
    }
    ~infiniband_completion_mapping_t() noexcept { destroy_(); }

    /** Expose @p completion_queue 's raw mlx5 ring + doorbell and map both for the GPU through
     *  @p registry (pages are shared across completion queues, so acquisitions are
     *  reference-counted); the consumer counter is allocated from @p arena . The ring must be a power
     *  of two and uncompressed. */
    [[nodiscard]] static expected<infiniband_completion_mapping_t> make(ibv_cq *completion_queue, gpu_arena_t &arena,
                                                                        host_page_registry_t &registry) noexcept;

    /** A by-value snapshot for kernels and @ref infiniband_drain_one . */
    [[nodiscard]] infiniband_completion_handles_t device_handles() const noexcept { return handles_; }

  private:
    void destroy_() noexcept {
        if (registry_ != nullptr) {
            for (host_page_range_t *range : {&doorbell_range_, &ring_range_}) {
                registry_->release(range->host, range->span);
                *range = host_page_range_t {};
            }
            registry_ = nullptr;
        }
        handles_ = infiniband_completion_handles_t {};
    }

    infiniband_completion_handles_t handles_ {};

    /** The owning NIC's page ledger — borrowed; outlives the mapping like the protection domain. */
    host_page_registry_t *registry_ = nullptr;

    /** The completion ring's acquired range, released through @ref registry_ at teardown. */
    host_page_range_t ring_range_ {};

    /** The completion doorbell's acquired range — routinely a page shared with other queues. */
    host_page_range_t doorbell_range_ {};
};

#if defined(__CUDACC__)

/** Big-endian-ize a 32-bit work-queue field (the NIC reads every scalar big-endian). */
[[nodiscard]] __device__ inline u32_t infiniband_byte_swap_32(u32_t value) noexcept {
    return __byte_perm(value, 0u, 0x0123u);
}

/** Big-endian-ize a 64-bit field: swap each half, then swap the halves. */
[[nodiscard]] __device__ inline u64_t infiniband_byte_swap_64(u64_t value) noexcept {
    u32_t const low_swapped = infiniband_byte_swap_32(static_cast<u32_t>(value));
    u32_t const high_swapped = infiniband_byte_swap_32(static_cast<u32_t>(value >> 32));
    return (static_cast<u64_t>(low_swapped) << 32) | static_cast<u64_t>(high_swapped);
}

/** System-scope acquire load of a completion entry's @c op_own byte — observing the flipped owner
 *  bit orders the rest of the entry (written by the NIC before it) as visible. */
[[nodiscard]] __device__ inline u8_t infiniband_load_acquire_u8(u8_t const *pointer) noexcept {
    unsigned value;
    asm volatile("ld.acquire.sys.global.u8 %0, [%1];" : "=r"(value) : "l"(pointer));
    return static_cast<u8_t>(value);
}

/** System-scope acquire load of an arrival flag — pairs with the partner's flag write so its prior
 *  payload write (ordered before the flag on the same queue pair) is observed first. */
[[nodiscard]] __device__ inline u32_t infiniband_load_acquire_u32(u32_t const *pointer) noexcept {
    u32_t value;
    asm volatile("ld.acquire.sys.global.u32 %0, [%1];" : "=r"(value) : "l"(pointer));
    return value;
}

/** Whether a device-posted write raises one reapable completion for @ref infiniband_drain_one . */
enum class infiniband_signaling_t : bool { unsignaled_k, signaled_k };

/**
 *  @brief Post @b one RDMA-write work-queue entry and ring the doorbell entirely from device code — the
 *      GDAKI primitive.
 *
 *  Claims the next producer cell, writes the control + remote-address segments followed by @b either a
 *  pointer local-data segment ( @p inline_four_bytes null) @b or a 4-byte inline segment (the arrival-flag
 *  write), fences, advances the send doorbell record, fences, and stores the control segment's first 8
 *  bytes to the BlueFlame register. One 48-byte entry fills exactly one 64-byte basic block, so each post
 *  advances the producer index by one. A @c signaled_k post raises one reapable completion for
 *  @ref infiniband_drain_one ; delivery on a Reliable Connection is in-order, so an unsignaled payload write
 *  followed by a signaled inline flag write on the @b same queue pair makes the flag's arrival the proof
 *  the payload landed. Call from @b one thread.
 */
__device__ inline void infiniband_post_write(infiniband_queue_handles_t handles, void const *local_source,
                                             u32_t local_access_key, u64_t remote_address, u32_t remote_access_key,
                                             u32_t byte_count, infiniband_signaling_t signaling,
                                             void const *inline_four_bytes) noexcept {
    u32_t const producer_index = atomicAdd(handles.producer_index, 1u);
    u32_t const cell_offset = (producer_index % handles.send_queue_entry_count) * handles.send_queue_stride;
    std::byte *entry_bytes = handles.send_queue.data() + cell_offset;

    auto *control = reinterpret_cast<mlx5_wqe_ctrl_seg *>(entry_bytes);
    auto *remote = reinterpret_cast<mlx5_wqe_raddr_seg *>(entry_bytes + sizeof(mlx5_wqe_ctrl_seg));
    std::byte *after_remote = entry_bytes + sizeof(mlx5_wqe_ctrl_seg) + sizeof(mlx5_wqe_raddr_seg);

    u32_t const opmod_index_opcode = infiniband_byte_swap_32(((producer_index & 0xffffu) << 8) |
                                                             static_cast<u32_t>(MLX5_OPCODE_RDMA_WRITE));
    u32_t const qpn_descriptor_size = infiniband_byte_swap_32((handles.queue_pair_number << 8) |
                                                              infiniband_write_descriptor_segments_k);
    // Control segment: opcode + this entry's index in the low 16 bits, then (QPN << 8 | descriptor count).
    control->opmod_idx_opcode = opmod_index_opcode;
    control->qpn_ds = qpn_descriptor_size;
    control->signature = 0;
    control->dci_stream_channel_id = 0;
    control->fm_ce_se = signaling == infiniband_signaling_t::signaled_k ? static_cast<u8_t>(MLX5_WQE_CTRL_CQ_UPDATE)
                                                                        : u8_t {0};
    control->imm = 0;

    // Remote-address segment: where on the peer's GPU the bytes land, under its remote key.
    remote->raddr = infiniband_byte_swap_64(remote_address);
    remote->rkey = infiniband_byte_swap_32(remote_access_key);
    remote->reserved = 0;

    if (inline_four_bytes != nullptr) {
        // Inline data segment: a big-endian header (inline flag | byte count), then the 4 bytes verbatim.
        auto *inline_header = reinterpret_cast<u32_t *>(after_remote);
        *inline_header = infiniband_byte_swap_32(infiniband_inline_data_flag_k | 4u);
        std::byte const *source = static_cast<std::byte const *>(inline_four_bytes);
        for (u32_t byte_index = 0; byte_index != 4u; ++byte_index) after_remote[4u + byte_index] = source[byte_index];
    }
    else {
        // Local-data segment: the source range inside this endpoint's registered buffer, under our key.
        auto *data = reinterpret_cast<mlx5_wqe_data_seg *>(after_remote);
        data->byte_count = infiniband_byte_swap_32(byte_count);
        data->lkey = infiniband_byte_swap_32(local_access_key);
        data->addr = infiniband_byte_swap_64(reinterpret_cast<u64_t>(local_source));
    }

    // The entry must be visible in system memory before the NIC observes the advanced doorbell record.
    __threadfence_system();
    handles.doorbell_record[mlx5_send_doorbell_index_k] = infiniband_byte_swap_32((producer_index + 1u) & 0xffffu);
    // The doorbell record must be visible before the BlueFlame ring that tells the NIC to fetch.
    __threadfence_system();
    u64_t const doorbell = static_cast<u64_t>(opmod_index_opcode) | (static_cast<u64_t>(qpn_descriptor_size) << 32);
    // Ring BlueFlame: the control segment's first 8 bytes, one 64-bit MMIO store, halves already big-endian.
    *static_cast<volatile u64_t *>(handles.blueflame_register) = doorbell;
    __threadfence_system();
}

/**
 *  @brief Reap exactly @b one signaled send completion entirely from device code — the GDAKI drain.
 *
 *  Acquire-spins on the consumer cell's owner bit until the NIC flips it to this pass's polarity,
 *  traps loudly on a requester error or once the spin outlasts @p wait_nanoseconds , then bumps the
 *  consumer index and publishes it to the set-consumer-index doorbell so the small ring never fills
 *  across replays. Call from @b one thread; bound every signaled post with one drain.
 */
__device__ inline void infiniband_drain_one(infiniband_completion_handles_t completion,
                                            u64_t wait_nanoseconds) noexcept {
    u32_t const consumer_index = *completion.consumer_index;
    u32_t const cell = consumer_index & (completion.entry_count - 1u);
    std::byte *cell_bytes = completion.ring.data() + static_cast<std::size_t>(cell) * completion.entry_size;
    auto *entry = reinterpret_cast<mlx5_cqe64 *>(cell_bytes + (completion.entry_size - sizeof(mlx5_cqe64)));
    u32_t const expected_owner = (completion.initial_owner ^ ((consumer_index >> completion.entry_log_count) & 1u)) &
                                 1u;

    u64_t const deadline = global_nanoseconds() + wait_nanoseconds;
    while (true) {
        if (global_nanoseconds() > deadline) {
            std::printf("uform: completion queue %u waited past its deadline at consumer index %u\n",
                        completion.completion_queue_number, consumer_index);
            __trap();
        }
        u8_t const op_own = infiniband_load_acquire_u8(&entry->op_own);
        if ((static_cast<u32_t>(op_own) & mlx5_completion_owner_mask_k) != expected_owner) continue; // not yet ours
        u32_t const opcode = static_cast<u32_t>(op_own) >> 4;
        if (opcode == static_cast<u32_t>(MLX5_CQE_INVALID)) continue; // owner flipped before the NIC wrote
        if (opcode == static_cast<u32_t>(MLX5_CQE_REQ_ERR)) {
            auto const *error = reinterpret_cast<mlx5_err_cqe const *>(entry);
            // `printf` is the ONLY vendor-supported device-side formatter (`std::format` is host-only,
            // allocating, throwing) — and this dying kernel's one channel to name its syndrome
            // before the trap.
            std::printf("infiniband_drain_one: send completion error syndrome=0x%x vendor=0x%x cqn=%u\n",
                        static_cast<u32_t>(error->syndrome), static_cast<u32_t>(error->vendor_err_synd),
                        completion.completion_queue_number);
            __trap();
        }
        break;
    }

    *completion.consumer_index = consumer_index + 1u;
    __threadfence_system();
    // Publish the consumer index (24-bit field, big-endian) so the completion ring never fills.
    completion.doorbell_record[mlx5_completion_consumer_index_k] = infiniband_byte_swap_32((consumer_index + 1u) &
                                                                                           0xffffffu);
    __threadfence_system();
}

#endif // defined(__CUDACC__)
#endif // defined(UFORM_WITH_INFINIBAND)

/** Rank-1 device view of the per-round link table, shaped @b [ceil(log2(D))] — one
 *  @ref infiniband_link_t per recursive-doubling round, indexed by round. */
using infiniband_links_view_t = nk::tensor_view<infiniband_link_t, 1>;

/** Rank-1 device view of the flag-round scalar, shaped @b [1] — the monotonic base every
 *  flag target keys on; the owning member advances it once per step. */
using infiniband_round_view_t = nk::tensor_view<infiniband_round_t, 1>;

/** The static description of one all-gather invocation — the two scalars its kernels need
 *  beyond the links and the round. Internal to @ref infiniband_reach_t 's dispatch: the member
 *  bakes both at wiring time and builds this by value at enqueue; consumers never construct one. */
struct infiniband_allgather_plan_t {

    /** Bytes per domain block; the gathered payload spans @b [D·block_bytes] at heap offset 0. */
    std::size_t block_bytes = 0;

    /** This rank's multimem_domain index — the recursive-doubling coordinate. */
    u32_t this_multimem_domain = 0;
};

/** The static description of one all-to-all invocation — this domain's per-peer send blocks live at
 *  @b [D·block_bytes] ( block @c p is what this domain sends domain @c p ; block @c self stays local ), and
 *  each round's landing receives that round's partner's block. Internal to @ref infiniband_reach_t 's
 *  dispatch; consumers never construct one. */
struct infiniband_alltoall_plan_t {

    /** Bytes this domain sends to (and receives from) any one peer — the per-pair block size. */
    std::size_t block_bytes = 0;
};

#if defined(UFORM_WITH_INFINIBAND)

/** Rank-1 device-only table of fabric links owned through a @ref gpu_arena_t . */
using infiniband_links_tensor_t = nk::tensor<infiniband_link_t, gpu_arena_alloc<infiniband_link_t>, 1>;

/**
 *  @brief The shared RDMA body every typed collective handle holds — one member's wired reach into the
 *      other multimem_domains, the level-2 counterpart of @ref multimem_domain_t 's per-member surface.
 *
 *  It owns the registered symmetric heap, one RC endpoint with its device-initiated queue / completion
 *  mappings per recursive-doubling round, the exchanged coordinates, the device-resident link table, and
 *  the collective's round scalar.
 *
 *  @b Internal : consumers hold @ref infiniband_all_reduce , @ref infiniband_all_gather , or
 *  @ref infiniband_all_to_all , whose @c make constructs this body through @ref colocated_domain_t —
 *  the handle's type @b is the role, so a heap layout can never contradict the collective it runs, and
 *  the adapters underneath are always the shared ones. Payload, links, round, and the domain coordinate
 *  are all baked at wiring time; the per-collective methods here are the handles' implementation.
 */
struct infiniband_reach_t {

  private:
    friend struct colocated_domain_t;

    /** The typed handles set each layout's payload and landing bytes at their @c make . */
    template <typename value_type_>
    friend struct infiniband_all_reduce;

    /** @copydoc infiniband_all_reduce */
    template <typename value_type_>
    friend struct infiniband_all_gather;

    /** @copydoc infiniband_all_reduce */
    template <typename value_type_>
    friend struct infiniband_all_to_all;

    /**
     *  @brief Byte offsets of one member's registered symmetric heap — the identical formula on every
     *      domain: the symmetric-heap contract the remote writes rely on.
     *
     *  Layout: @b [payload | parity landing pairs | parity flag pairs] , every region 128-byte aligned; the
     *  typed handles set the payload and landing bytes, and @ref colocated_domain_t derives the rounds.
     */
    struct heap_layout_t {

        /** The shard / world buffer the collective operates on — the heap's prefix. */
        std::size_t payload_bytes = 0;

        /** Per-round landing size; 0 means the payload doubles as the landing. */
        std::size_t landing_bytes = 0;

        /** Recursive-doubling rounds — the link, landing, and flag count; the wiring derives it
         *  from the world's domain count, so the typed handles never spell it. */
        u32_t rounds = 0;

        [[nodiscard]] std::size_t landings_offset() const noexcept {
            return round_up_to_multiple(payload_bytes, cache_line_bytes_k);
        }

        /** Landings and flags are @b parity-doubled (even / odd step): without the second copy a
         *  fast partner's next-step write can land on a buffer the slower side is still folding —
         *  the flag only proves arrival, never consumption. */
        [[nodiscard]] std::size_t flags_offset() const noexcept {
            return round_up_to_multiple(landings_offset() + 2u * rounds * landing_bytes, cache_line_bytes_k);
        }

        [[nodiscard]] std::size_t total_bytes() const noexcept {
            return round_up_to_multiple(flags_offset() + 2u * rounds * sizeof(u32_t), cache_line_bytes_k);
        }
    };

    /** The shared wiring body behind @ref colocated_domain_t : allocate the heap, wire one
     *  endpoint per round on the shared @p nic , exchange coordinates, prove both ends armed, and
     *  stage the links. The @p stream seeds the member's round scalar (ordered after its zero). */
    [[nodiscard]] static expected<infiniband_reach_t> make_(mesh_topology_t const &topology, u32_t member_index,
                                                            infiniband_nic_t &nic, gpu_arena_t &arena,
                                                            gpu_stream_t &stream, heap_layout_t layout,
                                                            sz::string_view_t key_prefix,
                                                            f64_t timeout_seconds) noexcept;

  public:

    /** The registered payload region — the shard the all-reduce sums in place, or the
     *  domain-major world buffer the all-gather fills. */
    [[nodiscard]] b8r1_span_t payload() noexcept { return {heap_.data(), layout_.payload_bytes}; }

    /** An all-gather member's own domain-major cell inside @ref payload — where this domain's
     *  block must be staged before @ref allgather ; the coordinate stays baked, never re-derived
     *  by consumers. */
    [[nodiscard]] b8r1_span_t own_block() noexcept {
        std::size_t const block_bytes = layout_.payload_bytes >> layout_.rounds;
        return {heap_.data() + static_cast<std::size_t>(multimem_domain_index_) * block_bytes, block_bytes};
    }

    /** The one enqueued host write of a step: stage the round into the pinned cell and
     *  enqueue its upload — on @p stream , before this step's collective. A no-op on the zero-round
     *  single-domain identity. */
    [[nodiscard]] expected_t advance(gpu_stream_t &stream) noexcept {
        if (layout_.rounds == 0) return {};
        return round_.advance(stream);
    }

    /** Whether this member is the zero-round single-domain identity — every collective a no-op. */
    [[nodiscard]] bool identity() const noexcept { return layout_.rounds == 0; }

    /** How many multimem_domains this member's recursive doubling spans — @c 2^rounds , exact
     *  because the wiring rejects non-power-of-two domain counts. */
    [[nodiscard]] u32_t multimem_domain_count() const noexcept { return 1u << layout_.rounds; }

    /**
     *  @brief Device-initiated inter-domain all-reduce-sum of this member's Float32 payload, in place —
     *      the cross-domain gradient / force reduce, enqueued on @p stream as capture-safe graph nodes.
     *
     *  Rounds run in order (round @c r 's accumulate feeds round @c r+1 's post); within a round:
     *  post-and-drain, wait, accumulate. Call @ref advance first, once per step. Zero rounds — the
     *  single-domain identity — enqueues nothing.
     */
    [[nodiscard]] expected_t allreduce_sum(gpu_stream_t &stream) noexcept;

    /**
     *  @brief Device-initiated inter-domain all-gather of the domain-major blocks inside this member's
     *      payload — how every domain receives every other domain's vertex tower, enqueued on @p stream
     *      as capture-safe graph nodes.
     *
     *  This domain's own block must already sit at its domain-major cell; after @c log2(D) rounds every
     *  block is filled. Call @ref advance first, once per step.
     */
    [[nodiscard]] expected_t allgather(gpu_stream_t &stream) noexcept;

    /** This domain's outgoing block for @p destination_domain inside the send region — where the caller
     *  stages what @ref alltoall ships to that domain. @c destination_domain equal to this domain's own index
     *  is the local block, kept in place and read directly. Meaningful only on an all-to-all-wired member. */
    [[nodiscard]] b8r1_span_t send_block(u32_t destination_domain) noexcept {
        std::size_t const block_bytes = layout_.landing_bytes;
        return {heap_.data() + static_cast<std::size_t>(destination_domain) * block_bytes, block_bytes};
    }

    /** The block @ref alltoall received from round @p round 's partner ( @c multimem_domain_index ^ 2^round )
     *  — that round's landing at @p step_parity (landings are even/odd doubled, so the caller names
     *  which step's landing it consumes). Meaningful only on an all-to-all-wired member. */
    [[nodiscard]] b8r1_span_t recv_block(u32_t round, u32_t step_parity) noexcept {
        std::size_t const block_bytes = layout_.landing_bytes;
        std::size_t const landing = layout_.landings_offset() +
                                    (static_cast<std::size_t>(round) * 2u + (step_parity & 1u)) * block_bytes;
        return {heap_.data() + landing, block_bytes};
    }

    /**
     *  @brief Device-initiated inter-domain all-to-all: ship each peer this domain's send block for it and land
     *      the peer's block for this domain, enqueued on @p stream as capture-safe graph nodes.
     *
     *  Stage every @ref send_block first; after the rounds each @ref recv_block holds its partner's block
     *  and the local block ( @ref send_block for this domain ) stays in place. Call @ref advance first,
     *  once per step. Two domains today — one pairwise exchange; beyond two needs the multi-hop schedule.
     */
    [[nodiscard]] expected_t alltoall(gpu_stream_t &stream) noexcept;

  private:

    /** The heap's byte-offset formula — identical on every domain, baked at wiring. */
    heap_layout_t layout_ {};

    /** This rank's multimem_domain index — the recursive-doubling coordinate, baked at wiring. */
    u32_t multimem_domain_index_ = 0;

    /** The member's monotonic step counter — host count, pinned staging, and the device
     *  round scalar it uploads into; see @ref barrier_round_t . */
    barrier_round_t round_ {};

    /** The registered symmetric heap: payload, per-round landings, per-round flags. Registrable
     *  allocation — one physical chunk — since @c nvidia_peermem rejects handle-spanning ranges. */
    b8r1_tensor_registrable_t heap_ {};

    /** One connected RC endpoint per round; the two mapping tables below layer onto them. */
    st::basic_vector<infiniband_endpoint_t> endpoints_ {};

    /** Per-round device-initiated send paths, owning the GPU registrations of each queue pair's
     *  work queue, doorbell record, and BlueFlame register. */
    st::basic_vector<infiniband_queue_mapping_t> queues_ {};

    /** Per-round completion parsers, owning the GPU registrations of each ring + doorbell. */
    st::basic_vector<infiniband_completion_mapping_t> completions_ {};

    /** The device link table the doorbell kernels index by round. */
    infiniband_links_tensor_t links_ {};

    /** The device round scalar; see @ref advance . */
    u32r1_tensor_arena_t round_scalar_ {};
};

/**
 *  @brief This process's membership of the mesh's colocated_domain: the opened NICs of every local GPU
 *      and the coordinate-exchange context.
 *
 *  It is created @b once in main, right after the mesh bootstrap, and threaded by reference to every
 *  cross-domain consumer, exactly like the per-rank arenas and the one @ref multimem_domain_t .
 *
 *  Wiring a handle is a consumer's own lane — its queue pairs (the single-poster concurrency
 *  granularity, so consumers' collectives can overlap), its heap, its round; this domain only lends
 *  the shared adapters and the exchange. The typed handles ( @ref infiniband_all_reduce ,
 *  @ref infiniband_all_gather , @ref infiniband_all_to_all ) are the whole wiring surface — the
 *  handle's type is the role, and rendezvous keys never appear in any API.
 */
struct colocated_domain_t {

    /** Open each local GPU's adapter — @c mlx5_<d> for local ordinal @c d , the box's GPU↔HCA
     *  pairing — once for the process: one per local GPU, however many processes share the
     *  multimem_domain. */
    [[nodiscard]] static expected<colocated_domain_t> make(mesh_topology_t &topology) noexcept;

    /** The bootstrapped world this domain was wired against. */
    [[nodiscard]] mesh_topology_t const &topology() const noexcept { return *topology_; }

    /** How many multimem_domains the world spans — 1 means no member ever posts. */
    [[nodiscard]] u32_t multimem_domain_count() const noexcept { return topology_->shape.multimem_domain_count(); }

  private:

    /** The typed collective handles are the only wiring consumers — the handle's type is the role. */
    template <typename value_type_>
    friend struct infiniband_all_reduce;

    /** @copydoc infiniband_all_reduce */
    template <typename value_type_>
    friend struct infiniband_all_gather;

    /** @copydoc infiniband_all_reduce */
    template <typename value_type_>
    friend struct infiniband_all_to_all;

    /**
     *  @brief Wire one typed handle's shared RDMA body: mint the rendezvous namespace, derive the
     *      recursive-doubling rounds from the world's domain span, and build @p layout 's heap,
     *      endpoints, and links over the member's shared adapter.
     *
     *  The namespace is the topology's next wiring generation, rank-consistent because every rank
     *  constructs its transports in the same SPMD order — keys never appear in any API.
     *  Single-domain worlds return the zero-round identity without touching InfiniBand. Runs once,
     *  at setup, outside any capture; the calling thread must hold the member's context.
     *
     *  @param[in] member_index This GPU's index inside its multimem_domain; the partner domain's same-index
     *      GPU pairs with it, wherever its process sits. A member another process drives still mints its id
     *      but comes back unwired: no heap, the rounds kept for @c multimem_domain_count() , never enqueued.
     *  @param[inout] arena The member's arena; the heap, the link table, and the mapping counters
     *      allocate from it (its chunks are GPUDirect-RDMA-capable by construction).
     *  @param[inout] stream The member's lead stream; the round scalar is seeded on it at setup.
     *  @param[in] layout The handle's heap shape — payload and landing bytes; the rounds are derived here.
     *  @param[in] timeout_seconds Deadline on the coordinate and armed-peer rendezvous, and on
     *      every device-side flag wait and completion drain of the wired rounds.
     */
    [[nodiscard]] expected<infiniband_reach_t> reach_(u32_t member_index, gpu_arena_t &arena, gpu_stream_t &stream,
                                                      infiniband_reach_t::heap_layout_t layout,
                                                      f64_t timeout_seconds) noexcept;

    /** The bootstrapped world — borrowed; main owns it for the process lifetime, and every wired
     *  handle claims its rendezvous generation from it. */
    mesh_topology_t *topology_ = nullptr;

    /** One opened adapter per local GPU, local order: entry @c d serves domain member @c member_offset+d . */
    per_gpu_array<infiniband_nic_t> nics_ {};
};

/** Cross-domain Float32 all-reduce-sum in place — the typed op-handle over @ref colocated_domain_t . Wires
 *  one endpoint per recursive-doubling round at @ref make (no lazy-init), auto-keys its rendezvous, and exposes
 *  the shard as a typed @c nk::tensor_span the caller stages before each step. @c operator() folds @c advance
 *  and the reduce into one enqueue. */
template <typename value_type_>
struct infiniband_all_reduce {
    using value_type = value_type_;

    /** Wire the reduce's shared body: heap @b [shard | parity landings | parity flags] , one
     *  endpoint per recursive-doubling round — a single-domain world gets the zero-round identity. */
    [[nodiscard]] static expected<infiniband_all_reduce> make(colocated_domain_t &domain, u32_t member_index,
                                                              gpu_arena_t &arena, std::size_t scalars,
                                                              gpu_stream_t &stream, f64_t timeout_seconds) noexcept {
        using result_t = expected<infiniband_all_reduce>;
        infiniband_reach_t::heap_layout_t layout;
        layout.payload_bytes = scalars * sizeof(value_type_);
        layout.landing_bytes = layout.payload_bytes;
        auto wired = domain.reach_(member_index, arena, stream, layout, timeout_seconds);
        if (!wired) return result_t::fail(std::move(wired));
        infiniband_all_reduce handle;
        handle.reach_ = std::move(wired.value_);
        return result_t {std::move(handle)};
    }

    /** The typed shard summed in place — the caller stages its contribution here before each step. */
    [[nodiscard]] nk::tensor_span<value_type_, 1> payload() noexcept {
        return typed_span<value_type_>(reach_.payload());
    }

    /** Advance the round and enqueue the in-place cross-domain sum — one op per step, capture-safe. */
    [[nodiscard]] expected_t operator()(gpu_stream_t &stream) noexcept {
        if (expected_t advanced = reach_.advance(stream); !advanced) return advanced;
        return reach_.allreduce_sum(stream);
    }

    /** Whether this buffer is the zero-round single-domain identity — the collective a no-op. */
    [[nodiscard]] bool identity() const noexcept { return reach_.identity(); }

    /** How many multimem_domains the reduce spans — the world-mean divisor's domain factor. */
    [[nodiscard]] u32_t multimem_domain_count() const noexcept { return reach_.multimem_domain_count(); }

  private:
    infiniband_reach_t reach_ {};
};

/** Cross-domain all-gather of domain-major blocks — the typed op-handle. This domain's block sits at
 *  @ref own_block before @c operator() ; after, every domain's block fills @ref payload . */
template <typename value_type_>
struct infiniband_all_gather {
    using value_type = value_type_;

    /** Wire the gather's shared body: the domain-major payload @b [domains·block] doubles as the
     *  landing (blocks gather in place) — a single-domain world gets the zero-round identity. */
    [[nodiscard]] static expected<infiniband_all_gather> make(colocated_domain_t &domain, u32_t member_index,
                                                              gpu_arena_t &arena, std::size_t block_scalars,
                                                              gpu_stream_t &stream, f64_t timeout_seconds) noexcept {
        using result_t = expected<infiniband_all_gather>;
        infiniband_reach_t::heap_layout_t layout;
        layout.payload_bytes = static_cast<std::size_t>(domain.multimem_domain_count()) * block_scalars *
                               sizeof(value_type_);
        layout.landing_bytes = 0;
        auto wired = domain.reach_(member_index, arena, stream, layout, timeout_seconds);
        if (!wired) return result_t::fail(std::move(wired));
        infiniband_all_gather handle;
        handle.reach_ = std::move(wired.value_);
        return result_t {std::move(handle)};
    }

    /** The whole domain-major world buffer — @b [domains·block] , filled after @c operator() . */
    [[nodiscard]] nk::tensor_span<value_type_, 1> payload() noexcept {
        return typed_span<value_type_>(reach_.payload());
    }

    /** This domain's own block — staged here before the gather. */
    [[nodiscard]] nk::tensor_span<value_type_, 1> own_block() noexcept {
        return typed_span<value_type_>(reach_.own_block());
    }

    /** Advance the round and enqueue the domain-major all-gather — one op per step, capture-safe. */
    [[nodiscard]] expected_t operator()(gpu_stream_t &stream) noexcept {
        if (expected_t advanced = reach_.advance(stream); !advanced) return advanced;
        return reach_.allgather(stream);
    }

  private:
    infiniband_reach_t reach_ {};
};

/** Cross-domain all-to-all of per-peer blocks — the typed op-handle. The caller stages each
 *  @ref send_block before @c operator() ; after, each @ref recv_block holds its round-partner's block, and this
 *  domain's own block stays in place. Two domains today (a pairwise exchange); beyond two fails loud. */
template <typename value_type_>
struct infiniband_all_to_all {
    using value_type = value_type_;

    /** Wire the exchange's shared body: per-peer send blocks @b [domains·block] in the payload,
     *  one parity landing pair per round — a single-domain world gets the zero-round identity.
     *  Recursive doubling reaches every peer only at two domains; beyond that a direct all-to-all
     *  needs the D-1 links (or a multi-hop schedule) the round wiring does not lay down. */
    [[nodiscard]] static expected<infiniband_all_to_all> make(colocated_domain_t &domain, u32_t member_index,
                                                              gpu_arena_t &arena, std::size_t block_scalars,
                                                              gpu_stream_t &stream, f64_t timeout_seconds) noexcept {
        using result_t = expected<infiniband_all_to_all>;
        u32_t const domains = domain.multimem_domain_count();
        if (domains > 2u)
            return result_t::fail(status_t::shape_unsupported_k,
                                  "all-to-all is wired for two domains; {} needs the multi-hop schedule", domains);
        infiniband_reach_t::heap_layout_t layout;
        layout.payload_bytes = static_cast<std::size_t>(domains) * block_scalars * sizeof(value_type_);
        layout.landing_bytes = block_scalars * sizeof(value_type_);
        auto wired = domain.reach_(member_index, arena, stream, layout, timeout_seconds);
        if (!wired) return result_t::fail(std::move(wired));
        infiniband_all_to_all handle;
        handle.reach_ = std::move(wired.value_);
        return result_t {std::move(handle)};
    }

    /** The outgoing block for @p destination_domain — staged before the exchange. */
    [[nodiscard]] nk::tensor_span<value_type_, 1> send_block(u32_t destination_domain) noexcept {
        return typed_span<value_type_>(reach_.send_block(destination_domain));
    }

    /** The parity of this handle's @b latest exchange — even/odd landings alternate per step, and
     *  a consumer reading @b several handles wired in lockstep (the mixer's per-member buffers) names
     *  the shared mix parity from its @b own handle, whose count already ticked. Host-side, so this
     *  serves the eager consumer only — a captured replay's device-side advances are invisible here. */
    [[nodiscard]] u32_t latest_parity() const noexcept {
        return (exchanges_ + 1u) & 1u; // exchange e (1-based) used parity (e-1)&1
    }

    /** The block received from round @p round 's partner at @p step_parity — valid after the
     *  exchange of that parity; see @ref latest_parity for the eager-only host contract. */
    [[nodiscard]] nk::tensor_span<value_type_, 1> recv_block(u32_t round, u32_t step_parity) noexcept {
        return typed_span<value_type_>(reach_.recv_block(round, step_parity));
    }

    /** Advance the round and enqueue the all-to-all exchange — one op per step, capture-safe. */
    [[nodiscard]] expected_t operator()(gpu_stream_t &stream) noexcept {
        if (expected_t advanced = reach_.advance(stream); !advanced) return advanced;
        expected_t exchanged = reach_.alltoall(stream);
        if (exchanged) ++exchanges_;
        return exchanged;
    }

  private:
    infiniband_reach_t reach_ {};

    /** Host-side exchange count mirroring the device round — only for @ref recv_block 's parity
     *  pick, legal because the mixer consumes eagerly (see the accessor's contract). */
    u32_t exchanges_ = 0;
};

#else // !defined(UFORM_WITH_INFINIBAND)

/** The build-gate stub: only the zero-round single-domain identity exists — every collective is
 *  a no-op, and wiring a real member fails loudly naming the gate. */
struct infiniband_reach_t {
    [[nodiscard]] b8r1_span_t payload() noexcept { return {}; }
    [[nodiscard]] b8r1_span_t own_block() noexcept { return {}; }
    [[nodiscard]] expected_t advance(gpu_stream_t &) noexcept { return {}; }
    [[nodiscard]] expected_t allreduce_sum(gpu_stream_t &) noexcept { return {}; }
    [[nodiscard]] expected_t allgather(gpu_stream_t &) noexcept { return {}; }
    [[nodiscard]] expected_t alltoall(gpu_stream_t &) noexcept { return {}; }
    [[nodiscard]] b8r1_span_t send_block(u32_t) noexcept { return {}; }
    [[nodiscard]] b8r1_span_t recv_block(u32_t, u32_t) noexcept { return {}; }
    [[nodiscard]] bool identity() const noexcept { return true; }
    [[nodiscard]] u32_t multimem_domain_count() const noexcept { return 1; }
};

/** The build-gate stub of the colocated domain: single-domain topologies work (nothing to
 *  wire); multi-domain worlds fail loudly naming the gate. */
struct colocated_domain_t {
    [[nodiscard]] static expected<colocated_domain_t> make(mesh_topology_t &topology) noexcept {
        if (topology.shape.multimem_domain_count() != 1)
            return expected<colocated_domain_t>::fail(
                status_t::other_k,
                "built without UFORM_WITH_INFINIBAND — rebuild with libibverbs + libmlx5 for " //
                "cross-domain collectives");
        colocated_domain_t domain;
        domain.topology_ = &topology;
        return expected<colocated_domain_t> {std::move(domain)};
    }
    [[nodiscard]] mesh_topology_t const &topology() const noexcept { return *topology_; }
    [[nodiscard]] u32_t multimem_domain_count() const noexcept { return 1u; }

  private:
    mesh_topology_t *topology_ = nullptr;
};

/** Build-gate stubs of the typed collective handles — the zero-round identity: @c make succeeds with a
 *  no-op handle, the payload accessors return empty spans, and @c operator() enqueues nothing. */
template <typename value_type_>
struct infiniband_all_reduce {
    using value_type = value_type_;
    [[nodiscard]] static expected<infiniband_all_reduce> make(colocated_domain_t &, u32_t, gpu_arena_t &, std::size_t,
                                                              gpu_stream_t &, f64_t) noexcept {
        return expected<infiniband_all_reduce> {infiniband_all_reduce {}};
    }
    [[nodiscard]] nk::tensor_span<value_type_, 1> payload() noexcept { return {}; }
    [[nodiscard]] expected_t operator()(gpu_stream_t &) noexcept { return {}; }
    [[nodiscard]] bool identity() const noexcept { return true; }
    [[nodiscard]] u32_t multimem_domain_count() const noexcept { return 1; }
};
template <typename value_type_>
struct infiniband_all_gather {
    using value_type = value_type_;
    [[nodiscard]] static expected<infiniband_all_gather> make(colocated_domain_t &, u32_t, gpu_arena_t &, std::size_t,
                                                              gpu_stream_t &, f64_t) noexcept {
        return expected<infiniband_all_gather> {infiniband_all_gather {}};
    }
    [[nodiscard]] nk::tensor_span<value_type_, 1> payload() noexcept { return {}; }
    [[nodiscard]] nk::tensor_span<value_type_, 1> own_block() noexcept { return {}; }
    [[nodiscard]] expected_t operator()(gpu_stream_t &) noexcept { return {}; }
};
template <typename value_type_>
struct infiniband_all_to_all {
    using value_type = value_type_;
    [[nodiscard]] static expected<infiniband_all_to_all> make(colocated_domain_t &, u32_t, gpu_arena_t &, std::size_t,
                                                              gpu_stream_t &, f64_t) noexcept {
        return expected<infiniband_all_to_all> {infiniband_all_to_all {}};
    }
    [[nodiscard]] nk::tensor_span<value_type_, 1> send_block(u32_t) noexcept { return {}; }
    [[nodiscard]] u32_t latest_parity() const noexcept { return 0; }
    [[nodiscard]] nk::tensor_span<value_type_, 1> recv_block(u32_t, u32_t) noexcept { return {}; }
    [[nodiscard]] expected_t operator()(gpu_stream_t &) noexcept { return {}; }
};

#endif // defined(UFORM_WITH_INFINIBAND)

/** The three InfiniBand buffers share the unified enqueue-only shape — a drifted signature fails here. */
static_assert(collective_buffer<infiniband_all_reduce<f32_t>>, "infiniband_all_reduce must model collective_buffer");
static_assert(collective_buffer<infiniband_all_gather<bf16_t>>, "infiniband_all_gather must model collective_buffer");
static_assert(collective_buffer<infiniband_all_to_all<b8_t>>, "infiniband_all_to_all must model collective_buffer");

} // namespace unum::uform::core
