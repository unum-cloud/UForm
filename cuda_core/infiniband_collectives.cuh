/**
 *  @brief Device-initiated InfiniBand collectives across multimem_domains — the vertex all-gather and
 *      the force / gradient all-reduce-sum, each an argument-free method of one member's wired
 *      @ref infiniband_member_t (the doorbell path NVIDIA documents as GDAKI, GPUDirect Async
 *      Kernel-Initiated).
 *
 *  @file cuda/primitives/infiniband_collectives.cuh
 *  @author Ash Vardanian
 *  @date July 2, 2026
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
 *  @section collectives_generations Arrival flags are generations, never events
 *
 *  Every round's arrival proof is a 4-byte @b inline flag write on the @b same queue pair as the payload:
 *  Reliable-Connection delivery is in-order, so the flag's arrival proves the payload landed. Flags are
 *  @b monotonic never-reset counters — a waiter spins with @c >= against the absolute target
 *  @c *generation+round+1 , so replays of a captured step need no flag reset (a reset would be a host
 *  write inside the replay loop). Each @ref infiniband_member_t owns its generation scalar and
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
 *      4. @ref infiniband_member_t — one member's wired reach into the other domains, owning the
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

#include <utility>  // `std::exchange`
#include <span>     // `std::span` — the DMA ring ranges inside the wire PODs
#include <charconv> // `std::from_chars` — coordinate payload parsing
#include <chrono>   // `std::chrono::steady_clock` — the completion-poll deadline
#include <thread>   // `std::this_thread::sleep_for` — the completion-poll backoff

#include <cuda.h> // `cuMemGetHandleForAddressRange`, `cuMemHostRegister`, `cuMemHostGetDevicePointer`

#if defined(UFORM_WITH_INFINIBAND)
#include <unistd.h> // `::close`, `sysconf` — dmabuf descriptors and page-aligned registrations

#include <infiniband/mlx5dv.h> // `mlx5dv_init_obj`, `mlx5_wqe_*`, `mlx5_cqe64` — the direct-verbs layer
#include <infiniband/verbs.h>  // `ibv_context`, `ibv_pd`, `ibv_cq`, `ibv_qp` — the RC bring-up
#endif

#include "cuda_core/types.cuh"          // `expected`, `expected_t`, `status_t`, `format_into`
#include "cuda_core/gpu_allocators.cuh" // `gpu_arena_t` carving the device producer / consumer counters
#include "cuda_core/schedule.cuh"       // `b8r1_span_t` — the registered GPU range travels as one typed span
#include "cuda_core/mesh.cuh"           // `mesh_topology_t` — the coordinate exchange rides the mesh rendezvous

namespace unum::uform::core {

/** @brief Monotonic never-reset arrival counter the GDAKI collectives spin on; lives in device memory. */
using infiniband_generation_t = u32_t;

/** @brief The send-doorbell entry in the mlx5 two-entry doorbell-record array (entry 0 is the receive side). */
inline constexpr std::size_t mlx5_send_doorbell_index_k = 1;
/** @brief The set-consumer-index entry in the mlx5 completion-queue doorbell array (entry 1 arms interrupts). */
inline constexpr std::size_t mlx5_completion_consumer_index_k = 0;
/** @brief Owner-bit mask of an mlx5 completion entry's @c op_own byte; hardware flips it per ring wrap. */
inline constexpr u8_t mlx5_completion_owner_mask_k = 1u;
/** @brief Descriptor-segment count (16-byte units) of an RDMA-write work-queue entry: control + remote
 *      address + data = 3. */
inline constexpr u32_t infiniband_write_descriptor_segments_k = 3u;
/** @brief The mlx5 inline-data marker OR'd into a work-queue entry's inline byte-count header — spelled as
 *      a project constant so the wire PODs stay free of @c <infiniband/mlx5dv.h> . */
inline constexpr u32_t infiniband_inline_data_flag_k = 0x80000000u;

/**
 *  @brief What one queue pair publishes so a peer can connect and address its registered GPU buffer — the
 *      wire POD of the RDMA rendezvous, exchanged as text over the mesh's filesystem rendezvous.
 */
struct infiniband_coordinates_t {
    /** @brief The port's local identifier — the same-subnet InfiniBand route. */
    u32_t local_identifier = 0;
    /** @brief The queue pair number the peer's RTR state targets. */
    u32_t queue_pair_number = 0;
    /** @brief The initial packet sequence number; both ends exchange theirs, reliability needs agreement only. */
    u32_t packet_sequence_number = 0;
    /** @brief The memory region's remote key authorizing the peer's writes and reads. */
    u32_t remote_key = 0;
    /** @brief The registered GPU buffer's base virtual address; a peer targets @c buffer_address+offset . */
    u64_t buffer_address = 0;
    /** @brief The port's global identifier — carried for a future routed fabric, unused on LID routing. */
    u8_t global_identifier[16] = {};

    /** @brief Serialize into the line-oriented exchange payload; @c false only on allocation failure. */
    [[nodiscard]] bool to_text(sz::string &text) const noexcept {
        bool built = format_into(text, "lid {}\nqpn {}\npsn {}\nrkey {}\naddr {}\ngid ", local_identifier,
                                 queue_pair_number, packet_sequence_number, remote_key, buffer_address);
        for (u8_t byte : global_identifier) built = built && format_into(text, "{:02x}", byte);
        return built && text.try_push_back('\n');
    }

    /** @brief Parse the payload @ref to_text produced; @c false when any field is missing or malformed. */
    [[nodiscard]] static bool from_text(sz::string_view payload, infiniband_coordinates_t &coordinates) noexcept {
        auto parse_number = [](sz::string_view token, auto &value) noexcept {
            std::from_chars_result const result = std::from_chars(token.data(), token.data() + token.size(), value);
            return result.ec == std::errc {} && result.ptr == token.data() + token.size();
        };
        sz::byteset const whitespaces = sz::whitespaces_set();
        bool lid_parsed = false, qpn_parsed = false, psn_parsed = false, rkey_parsed = false, addr_parsed = false,
             gid_parsed = false;
        for (sz::string_view line : payload.split(sz::newlines_set())) {
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

/** @brief Publish this rank's @p local coordinates under @p key and await + parse the @p peer_host_domain 's
 *      payload under the same key — one QP handshake over the mesh's filesystem rendezvous. */
[[nodiscard]] inline expected_t exchange_infiniband_coordinates(mesh_topology_t const &topology, sz::string_view key,
                                                                infiniband_coordinates_t const &local,
                                                                u32_t peer_host_domain, infiniband_coordinates_t &peer,
                                                                f64_t timeout_seconds) noexcept {
    sz::string local_text;
    if (!local.to_text(local_text))
        return expected_t::fail(status_t::bad_alloc_k, "coordinate payload assembly failed (key={})", key);
    if (expected_t published = topology.publish_text(key, local_text.view()); !published) return published;
    sz::string peer_text;
    if (expected_t awaited = topology.await_text(peer_host_domain, key, peer_text, timeout_seconds); !awaited)
        return awaited;
    if (!infiniband_coordinates_t::from_text(peer_text.view(), peer))
        return expected_t::fail(status_t::io_error_k, "malformed coordinates from host_domain {} (key={}): \"{}\"",
                                peer_host_domain, key, peer_text);
    return {};
}

/**
 *  @brief The pointer-and-scalar handles a kernel needs to post one work-queue entry and ring the
 *      doorbell — a trivially-copyable POD passed by value; the owning @ref infiniband_queue_mapping_t fills
 *      it. All pointers are in the GPU's address space.
 */
struct infiniband_queue_handles_t {
    /** @brief The send work-queue entry storage (host DMA memory the NIC fetches from), spanning
     *      @c send_queue_entry_count·send_queue_stride bytes. */
    std::span<std::byte> send_queue {};
    /** @brief The big-endian doorbell-record array; the send entry holds the producer counter the NIC polls. */
    u32_t *doorbell_record = nullptr;
    /** @brief The BlueFlame / User-Access-Region doorbell register — NIC BAR / MMIO space. */
    void *blueflame_register = nullptr;
    /** @brief Device counter of posted entries (the producer index); atomically claimed per post. */
    u32_t *producer_index = nullptr;
    /** @brief The queue pair number, baked into every control segment. */
    u32_t queue_pair_number = 0;
    /** @brief Send-queue capacity in basic blocks; the producer index wraps modulo this. */
    u32_t send_queue_entry_count = 0;
    /** @brief Bytes per basic block (64 on ConnectX-7); the entry stride in the buffer. */
    u32_t send_queue_stride = 0;
};

/**
 *  @brief The handles a kernel needs to reap one signaled send completion — the drain counterpart of
 *      @ref infiniband_queue_handles_t . The three hardware unknowns ( @c entry_size , @c initial_owner ,
 *      @c compressed ) are captured at mapping time; the valid 64-byte entry lives in the @b high bytes
 *      of a wider cell.
 */
struct infiniband_completion_handles_t {
    /** @brief The completion-entry ring (host DMA memory the NIC writes into), spanning
     *      @c entry_count·entry_size bytes. */
    std::span<std::byte> ring {};
    /** @brief The big-endian completion doorbell record; set-consumer-index at entry 0. */
    u32_t *doorbell_record = nullptr;
    /** @brief Device counter of reaped completions (the consumer index), in GPU VRAM. */
    u32_t *consumer_index = nullptr;
    /** @brief Ring capacity in entries — a power of two; the consumer index masks modulo it. */
    u32_t entry_count = 0;
    /** @brief Bytes per ring cell (64 or 128); the valid 64-byte entry sits in the high bytes. */
    u32_t entry_size = 0;
    /** @brief @c log2(entry_count) ; the owner-bit polarity at index @c i is @c (i>>entry_log_count)&1 . */
    u32_t entry_log_count = 0;
    /** @brief The completion queue number, carried for diagnostics. */
    u32_t completion_queue_number = 0;
    /** @brief Owner-bit polarity of a valid entry at consumer index zero. */
    u32_t initial_owner = 0;
    /** @brief 1 when entry compression is active — the mapping requires 0 (uncompressed). */
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
    /** @brief Which remote multimem_domain this link reaches. */
    u32_t remote_multimem_domain_index = 0;
    /** @brief Recursive-doubling round this link serves: the partner is @c this_domain^(1<<round_index) . */
    u32_t round_index = 0;
    /** @brief Peer domain's symmetric-heap base virtual address; targets are @c base+offset . */
    u64_t remote_heap_base = 0;
    /** @brief Remote key authorizing writes into the peer heap. */
    u32_t remote_access_key = 0;
    /** @brief Local key authorizing the adapter to source from this rank's registered heap. */
    u32_t local_access_key = 0;
    /** @brief The device-initiated send path of this link's queue pair. */
    infiniband_queue_handles_t queue {};
    /** @brief The completion parser paired with @ref queue ; one drain per signaled post. */
    infiniband_completion_handles_t completion {};
    /** @brief This rank's landing buffer where the partner's remote write arrives, shaped
     *      @b [landing_bytes] — the consuming collective reinterprets the scalar type it landed. */
    b8r1_span_t local_landing {};
    /** @brief Symmetric-heap byte offset of the partner's landing buffer. */
    std::size_t landing_offset = 0;
    /** @brief Symmetric-heap byte offset of the partner's arrival flag. */
    std::size_t remote_flag_offset = 0;
    /** @brief This rank's monotonic arrival flag (device VRAM) the waiting kernel acquire-spins on. */
    infiniband_generation_t *arrival_flag = nullptr;
};

#if defined(UFORM_WITH_INFINIBAND)

/** @brief One InfiniBand access-flag set shared by the memory region and the queue pair. */
inline constexpr int infiniband_access_flags_k = IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_WRITE |
                                                 IBV_ACCESS_REMOTE_READ;
/** @brief Completion-queue and work-request depth; the host path posts a handful of writes per exchange. */
inline constexpr int infiniband_queue_depth_k = 16;

/** @brief Whether a host-posted write also carries immediate data (consuming a posted receive on the peer). */
enum class infiniband_signal_t : u32_t {
    plain_signaled_k = 0,
    with_immediate_k = 1,
};

/**
 *  @brief One opened mlx5 adapter serving one GPU: the device context, the protection domain, and the
 *      port identity (LID, GID, MTU) every endpoint on this GPU borrows — opened @b once, in main,
 *      never per queue pair. Move-only; teardown deallocates the domain and closes the device.
 */
struct infiniband_nic_t {

    infiniband_nic_t() noexcept = default;
    infiniband_nic_t(infiniband_nic_t const &) = delete;
    infiniband_nic_t &operator=(infiniband_nic_t const &) = delete;
    infiniband_nic_t(infiniband_nic_t &&other) noexcept
        : device_context_(std::exchange(other.device_context_, nullptr)),
          protection_domain_(std::exchange(other.protection_domain_, nullptr)),
          local_identifier_(other.local_identifier_), path_mtu_(other.path_mtu_), port_(other.port_) {
        std::memcpy(global_identifier_, other.global_identifier_, sizeof(global_identifier_));
    }
    infiniband_nic_t &operator=(infiniband_nic_t &&other) noexcept {
        if (this != &other) {
            destroy_();
            device_context_ = std::exchange(other.device_context_, nullptr);
            protection_domain_ = std::exchange(other.protection_domain_, nullptr);
            local_identifier_ = other.local_identifier_;
            path_mtu_ = other.path_mtu_;
            port_ = other.port_;
            std::memcpy(global_identifier_, other.global_identifier_, sizeof(global_identifier_));
        }
        return *this;
    }
    ~infiniband_nic_t() noexcept { destroy_(); }

    /** @brief Open the adapter serving this host_domain's @b local GPU ordinal @p local_gpu_index (name
     *      @c mlx5_<local_gpu_index> ) — HCAs are numbered @b per @b box, so the GPU↔HCA pairing is local,
     *      never the global @c gpu_index (which would ask for @c mlx5_8 on the second box). The
     *      @c mlx5_<ordinal> pairing assumes index-identity affinity; resolving the NIC from the GPU's PCI
     *      locality is the follow-up. Allocates the protection domain and caches the port identity.
     *  @param[in] infiniband_port The port on the adapter; single-port ConnectX parts use 1. */
    [[nodiscard]] static expected<infiniband_nic_t> make(u32_t local_gpu_index, u8_t infiniband_port = 1) noexcept;

    /** @brief The opened device context endpoints create queues against — borrowed, still owned here. */
    [[nodiscard]] ibv_context *context() const noexcept { return device_context_; }
    /** @brief The shared protection domain endpoints register regions and queue pairs in. */
    [[nodiscard]] ibv_pd *protection_domain() const noexcept { return protection_domain_; }
    /** @brief The port's local identifier — the same-subnet InfiniBand route. */
    [[nodiscard]] u32_t local_identifier() const noexcept { return local_identifier_; }
    /** @brief The port's active MTU, baked into every queue pair's RTR transition. */
    [[nodiscard]] unsigned path_mtu() const noexcept { return path_mtu_; }
    /** @brief The port number every queue pair binds. */
    [[nodiscard]] u8_t port() const noexcept { return port_; }
    /** @brief The 16-byte port GID — carried in coordinates for a future routed fabric. */
    [[nodiscard]] u8_t const *global_identifier() const noexcept { return global_identifier_; }

  private:
    void destroy_() noexcept {
        if (protection_domain_ != nullptr) ibv_dealloc_pd(std::exchange(protection_domain_, nullptr));
        if (device_context_ != nullptr) ibv_close_device(std::exchange(device_context_, nullptr));
    }

    /** @brief The opened mlx5 device. */
    ibv_context *device_context_ = nullptr;
    /** @brief The one protection domain every endpoint on this GPU shares. */
    ibv_pd *protection_domain_ = nullptr;
    /** @brief Cached port identity — queried once at @ref make . */
    u32_t local_identifier_ = 0;
    /** @brief The port's active MTU. */
    unsigned path_mtu_ = 0;
    /** @brief The bound port number. */
    u8_t port_ = 1;
    /** @brief The port's global identifier. */
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
          path_mtu_(other.path_mtu_), local_coordinates_(other.local_coordinates_),
          peer_coordinates_(other.peer_coordinates_) {}
    infiniband_endpoint_t &operator=(infiniband_endpoint_t &&other) noexcept {
        if (this != &other) {
            destroy_();
            completion_queue_ = std::exchange(other.completion_queue_, nullptr);
            queue_pair_ = std::exchange(other.queue_pair_, nullptr);
            memory_region_ = std::exchange(other.memory_region_, nullptr);
            gpu_buffer_ = std::exchange(other.gpu_buffer_, b8r1_span_t {});
            dmabuf_descriptor_ = std::exchange(other.dmabuf_descriptor_, -1);
            infiniband_port_ = other.infiniband_port_;
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

    /** @brief What this endpoint publishes for the peer's @ref connect . */
    [[nodiscard]] infiniband_coordinates_t const &local_coordinates() const noexcept { return local_coordinates_; }
    /** @brief The queue pair the GDAKI mapping layers onto — borrowed, still owned here. */
    [[nodiscard]] ibv_qp *queue_pair() const noexcept { return queue_pair_; }
    /** @brief The completion queue the GDAKI mapping layers onto — borrowed, still owned here. */
    [[nodiscard]] ibv_cq *completion_queue() const noexcept { return completion_queue_; }
    /** @brief The registered memory region's local key — the source authorization every post carries. */
    [[nodiscard]] u32_t local_key() const noexcept { return memory_region_->lkey; }

    /**
     *  @brief Drive the queue pair to Ready-To-Send against @p peer : INIT (port + pkey + access) → RTR
     *      (peer queue-pair number and sequence number over an LID-routed address vector) → RTS.
     *
     *  Infinite receiver-not-ready retries: a write racing ahead of the peer's posted receive is retried,
     *  never dropped — the host-side handshake needs no strict ordering.
     */
    [[nodiscard]] expected_t connect(infiniband_coordinates_t const &peer) noexcept;

    /** @brief Host-post one signaled RDMA write of @p length bytes from @c gpu_buffer+local_offset into
     *      the peer's @c buffer_address+remote_offset — the bring-up / smoke-test path; production posts
     *      ride @ref infiniband_post_write . */
    [[nodiscard]] expected_t post_write(std::size_t local_offset, std::size_t remote_offset, std::size_t length,
                                        infiniband_signal_t signal = infiniband_signal_t::plain_signaled_k) noexcept {
        if (length == 0 || local_offset + length > gpu_buffer_.extent(0))
            return expected_t::fail(status_t::invalid_input_k,
                                    "write of {} bytes at local offset {} exceeds the registered {} bytes", length,
                                    local_offset, gpu_buffer_.extent(0));
        ibv_sge scatter_gather {};
        scatter_gather.addr = reinterpret_cast<u64_t>(gpu_buffer_.data()) + local_offset;
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

    /** @brief Post a zero-scatter receive so a peer's write-with-immediate completion can be consumed. */
    [[nodiscard]] expected_t post_receive() noexcept {
        ibv_recv_wr work_request {};
        ibv_recv_wr *rejected = nullptr;
        if (ibv_post_recv(queue_pair_, &work_request, &rejected) != 0)
            return expected_t::fail(status_t::io_error_k, "posting the receive failed");
        return {};
    }

    /** @brief Spin-poll (50 µs backoff) until one completion is reaped or @p timeout_seconds elapses. */
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
            std::this_thread::sleep_for(std::chrono::microseconds {50});
        }
    }

  private:
    /** @brief Register the GPU buffer as a GPUDirect region in @p protection_domain : dmabuf-first,
     *      @c nvidia_peermem fallback. */
    [[nodiscard]] expected_t register_gpu_memory_(ibv_pd *protection_domain) noexcept;

    void destroy_() noexcept {
        if (memory_region_ != nullptr) ibv_dereg_mr(std::exchange(memory_region_, nullptr));
        if (queue_pair_ != nullptr) ibv_destroy_qp(std::exchange(queue_pair_, nullptr));
        if (completion_queue_ != nullptr) ibv_destroy_cq(std::exchange(completion_queue_, nullptr));
        if (dmabuf_descriptor_ >= 0) ::close(std::exchange(dmabuf_descriptor_, -1));
        gpu_buffer_ = b8r1_span_t {};
    }

    /** @brief This endpoint's completion queue — its own, so drains never cross lanes. */
    ibv_cq *completion_queue_ = nullptr;
    /** @brief The Reliable-Connection queue pair — the single-poster concurrency granularity. */
    ibv_qp *queue_pair_ = nullptr;
    /** @brief The GPUDirect memory region over @ref gpu_buffer_ . */
    ibv_mr *memory_region_ = nullptr;
    /** @brief The registered GPU VRAM range peers read and write. */
    b8r1_span_t gpu_buffer_ {};
    /** @brief The exported dmabuf descriptor when that registration path won; -1 on the fallback. */
    int dmabuf_descriptor_ = -1;
    /** @brief The bound port, from the owning NIC. */
    u8_t infiniband_port_ = 1;
    /** @brief The port's active MTU, from the owning NIC. */
    unsigned path_mtu_ = 0;
    /** @brief What this endpoint publishes for the peer's @ref connect . */
    infiniband_coordinates_t local_coordinates_ {};
    /** @brief The connected peer's coordinates, kept for host-posted writes. */
    infiniband_coordinates_t peer_coordinates_ {};
};

/** @brief Register the page(s) covering @p host_pointer for GPU access and resolve its device pointer;
 *      @p registration_base receives the page-floor base to later unregister (null when the page was
 *      already registered by someone else and must stay theirs). */
[[nodiscard]] inline expected_t infiniband_map_for_device(void *host_pointer, std::size_t span, bool io_memory,
                                                          void *&registration_base, void *&device_pointer) noexcept {
    registration_base = nullptr;
    device_pointer = nullptr;
    long const page_size = sysconf(_SC_PAGESIZE);
    std::uintptr_t const page_mask = ~(static_cast<std::uintptr_t>(page_size) - 1u);
    std::uintptr_t const value = reinterpret_cast<std::uintptr_t>(host_pointer);
    void *base = reinterpret_cast<void *>(value & page_mask);
    std::size_t const bytes = ((value + span + static_cast<std::uintptr_t>(page_size) - 1u) & page_mask) -
                              (value & page_mask);

    unsigned const flags = CU_MEMHOSTREGISTER_DEVICEMAP | (io_memory ? CU_MEMHOSTREGISTER_IOMEMORY : 0u);
    CUresult const registered = cuMemHostRegister(base, bytes, flags);
    if (registered == CUDA_SUCCESS) registration_base = base;
    else if (registered != CUDA_ERROR_HOST_MEMORY_ALREADY_REGISTERED)
        return expected_t::fail(registered, "host registration failed ({} memory, {} bytes)",
                                io_memory ? "NIC MMIO" : "DMA", bytes);
    CUdeviceptr resolved = 0;
    if (CUresult result = cuMemHostGetDevicePointer(&resolved, host_pointer, 0); result != CUDA_SUCCESS) {
        if (registration_base != nullptr) cuMemHostUnregister(std::exchange(registration_base, nullptr));
        return expected_t::fail(result, "device-pointer resolution failed");
    }
    device_pointer = reinterpret_cast<void *>(resolved);
    return {};
}

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
          send_queue_registration_(std::exchange(other.send_queue_registration_, nullptr)),
          doorbell_registration_(std::exchange(other.doorbell_registration_, nullptr)),
          blueflame_registration_(std::exchange(other.blueflame_registration_, nullptr)) {}
    infiniband_queue_mapping_t &operator=(infiniband_queue_mapping_t &&other) noexcept {
        if (this != &other) {
            destroy_();
            handles_ = std::exchange(other.handles_, infiniband_queue_handles_t {});
            send_queue_registration_ = std::exchange(other.send_queue_registration_, nullptr);
            doorbell_registration_ = std::exchange(other.doorbell_registration_, nullptr);
            blueflame_registration_ = std::exchange(other.blueflame_registration_, nullptr);
        }
        return *this;
    }
    ~infiniband_queue_mapping_t() noexcept { destroy_(); }

    /** @brief Expose @p queue_pair 's raw mlx5 send queue / doorbell / BlueFlame register and map all
     *      three for the GPU; the producer counter is allocated from @p arena (capture-stable address). */
    [[nodiscard]] static expected<infiniband_queue_mapping_t> make(ibv_qp *queue_pair, gpu_arena_t &arena) noexcept;

    /** @brief A by-value snapshot for kernels and @ref infiniband_post_write . */
    [[nodiscard]] infiniband_queue_handles_t device_handles() const noexcept { return handles_; }

  private:
    void destroy_() noexcept {
        for (void **base : {&blueflame_registration_, &doorbell_registration_, &send_queue_registration_})
            if (*base != nullptr) cuMemHostUnregister(std::exchange(*base, nullptr));
        handles_ = infiniband_queue_handles_t {};
    }

    infiniband_queue_handles_t handles_ {};
    void *send_queue_registration_ = nullptr;
    void *doorbell_registration_ = nullptr;
    void *blueflame_registration_ = nullptr;
};

/**
 *  @brief The GPU-reachable completion path of one queue pair: the completion ring and its doorbell
 *      record mapped for the GPU (both ordinary host DMA memory), plus a device consumer-index counter
 *      and the hardware geometry captured at build time.
 */
struct infiniband_completion_mapping_t {

    infiniband_completion_mapping_t() noexcept = default;
    infiniband_completion_mapping_t(infiniband_completion_mapping_t const &) = delete;
    infiniband_completion_mapping_t &operator=(infiniband_completion_mapping_t const &) = delete;
    infiniband_completion_mapping_t(infiniband_completion_mapping_t &&other) noexcept
        : handles_(std::exchange(other.handles_, infiniband_completion_handles_t {})),
          ring_registration_(std::exchange(other.ring_registration_, nullptr)),
          doorbell_registration_(std::exchange(other.doorbell_registration_, nullptr)) {}
    infiniband_completion_mapping_t &operator=(infiniband_completion_mapping_t &&other) noexcept {
        if (this != &other) {
            destroy_();
            handles_ = std::exchange(other.handles_, infiniband_completion_handles_t {});
            ring_registration_ = std::exchange(other.ring_registration_, nullptr);
            doorbell_registration_ = std::exchange(other.doorbell_registration_, nullptr);
        }
        return *this;
    }
    ~infiniband_completion_mapping_t() noexcept { destroy_(); }

    /** @brief Expose @p completion_queue 's raw mlx5 ring + doorbell and map both for the GPU; the
     *      consumer counter is allocated from @p arena . The ring must be a power of two and uncompressed. */
    [[nodiscard]] static expected<infiniband_completion_mapping_t> make(ibv_cq *completion_queue,
                                                                        gpu_arena_t &arena) noexcept;

    /** @brief A by-value snapshot for kernels and @ref infiniband_drain_one . */
    [[nodiscard]] infiniband_completion_handles_t device_handles() const noexcept { return handles_; }

  private:
    void destroy_() noexcept {
        for (void **base : {&doorbell_registration_, &ring_registration_})
            if (*base != nullptr) cuMemHostUnregister(std::exchange(*base, nullptr));
        handles_ = infiniband_completion_handles_t {};
    }

    infiniband_completion_handles_t handles_ {};
    void *ring_registration_ = nullptr;
    void *doorbell_registration_ = nullptr;
};

#if defined(__CUDACC__)

/** @brief Big-endian-ize a 32-bit work-queue field (the NIC reads every scalar big-endian). */
[[nodiscard]] __device__ inline u32_t infiniband_byte_swap_32(u32_t value) noexcept {
    return __byte_perm(value, 0u, 0x0123u);
}

/** @brief Big-endian-ize a 64-bit field: swap each half, then swap the halves. */
[[nodiscard]] __device__ inline u64_t infiniband_byte_swap_64(u64_t value) noexcept {
    u32_t const low_swapped = infiniband_byte_swap_32(static_cast<u32_t>(value));
    u32_t const high_swapped = infiniband_byte_swap_32(static_cast<u32_t>(value >> 32));
    return (static_cast<u64_t>(low_swapped) << 32) | static_cast<u64_t>(high_swapped);
}

/** @brief System-scope acquire load of a completion entry's @c op_own byte — observing the flipped owner
 *      bit orders the rest of the entry (written by the NIC before it) as visible. */
[[nodiscard]] __device__ inline u8_t infiniband_load_acquire_u8(u8_t const *pointer) noexcept {
    unsigned value;
    asm volatile("ld.acquire.sys.global.u8 %0, [%1];" : "=r"(value) : "l"(pointer));
    return static_cast<u8_t>(value);
}

/** @brief System-scope acquire load of an arrival flag — pairs with the partner's flag write so its prior
 *      payload write (ordered before the flag on the same queue pair) is observed first. */
[[nodiscard]] __device__ inline u32_t infiniband_load_acquire_u32(u32_t const *pointer) noexcept {
    u32_t value;
    asm volatile("ld.acquire.sys.global.u32 %0, [%1];" : "=r"(value) : "l"(pointer));
    return value;
}

/**
 *  @brief Post @b one RDMA-write work-queue entry and ring the doorbell entirely from device code — the
 *      GDAKI primitive.
 *
 *  Claims the next producer cell, writes the control + remote-address segments followed by @b either a
 *  pointer local-data segment ( @p inline_four_bytes null) @b or a 4-byte inline segment (the arrival-flag
 *  write), fences, advances the send doorbell record, fences, and stores the control segment's first 8
 *  bytes to the BlueFlame register. One 48-byte entry fills exactly one 64-byte basic block, so each post
 *  advances the producer index by one. A @p signaled post raises one reapable completion for
 *  @ref infiniband_drain_one ; delivery on a Reliable Connection is in-order, so an unsignaled payload write
 *  followed by a signaled inline flag write on the @b same queue pair makes the flag's arrival the proof
 *  the payload landed. Call from @b one thread.
 */
__device__ inline void infiniband_post_write(infiniband_queue_handles_t handles, u64_t local_gpu_address,
                                             u32_t local_access_key, u64_t remote_address, u32_t remote_access_key,
                                             u32_t byte_count, bool signaled, void const *inline_four_bytes) noexcept {
    u32_t const producer_index = atomicAdd(handles.producer_index, 1u);
    u32_t const cell_offset = (producer_index % handles.send_queue_entry_count) * handles.send_queue_stride;
    u8_t *entry_bytes = reinterpret_cast<u8_t *>(handles.send_queue.data()) + cell_offset;

    auto *control = reinterpret_cast<mlx5_wqe_ctrl_seg *>(entry_bytes);
    auto *remote = reinterpret_cast<mlx5_wqe_raddr_seg *>(entry_bytes + sizeof(mlx5_wqe_ctrl_seg));
    u8_t *after_remote = entry_bytes + sizeof(mlx5_wqe_ctrl_seg) + sizeof(mlx5_wqe_raddr_seg);

    // Control segment: opcode + this entry's index in the low 16 bits, then (QPN << 8 | descriptor count).
    u32_t const opmod_index_opcode = infiniband_byte_swap_32(((producer_index & 0xffffu) << 8) |
                                                             static_cast<u32_t>(MLX5_OPCODE_RDMA_WRITE));
    u32_t const qpn_descriptor_size = infiniband_byte_swap_32((handles.queue_pair_number << 8) |
                                                              infiniband_write_descriptor_segments_k);
    control->opmod_idx_opcode = opmod_index_opcode;
    control->qpn_ds = qpn_descriptor_size;
    control->signature = 0;
    control->dci_stream_channel_id = 0;
    control->fm_ce_se = signaled ? static_cast<u8_t>(MLX5_WQE_CTRL_CQ_UPDATE) : u8_t {0};
    control->imm = 0;

    // Remote-address segment: where on the peer's GPU the bytes land, under its remote key.
    remote->raddr = infiniband_byte_swap_64(remote_address);
    remote->rkey = infiniband_byte_swap_32(remote_access_key);
    remote->reserved = 0;

    if (inline_four_bytes != nullptr) {
        // Inline data segment: a big-endian header (inline flag | byte count), then the 4 bytes verbatim.
        auto *inline_header = reinterpret_cast<u32_t *>(after_remote);
        *inline_header = infiniband_byte_swap_32(infiniband_inline_data_flag_k | 4u);
        u8_t const *source = static_cast<u8_t const *>(inline_four_bytes);
        for (u32_t byte_index = 0; byte_index != 4u; ++byte_index) after_remote[4u + byte_index] = source[byte_index];
    }
    else {
        // Local-data segment: the source range inside this endpoint's registered buffer, under our key.
        auto *data = reinterpret_cast<mlx5_wqe_data_seg *>(after_remote);
        data->byte_count = infiniband_byte_swap_32(byte_count);
        data->lkey = infiniband_byte_swap_32(local_access_key);
        data->addr = infiniband_byte_swap_64(local_gpu_address);
    }

    // The entry must be visible in system memory before the NIC observes the advanced doorbell record.
    __threadfence_system();
    handles.doorbell_record[mlx5_send_doorbell_index_k] = infiniband_byte_swap_32((producer_index + 1u) & 0xffffu);
    // The doorbell record must be visible before the BlueFlame ring that tells the NIC to fetch.
    __threadfence_system();
    // Ring BlueFlame: the control segment's first 8 bytes, one 64-bit MMIO store, halves already big-endian.
    u64_t const doorbell = static_cast<u64_t>(opmod_index_opcode) | (static_cast<u64_t>(qpn_descriptor_size) << 32);
    *static_cast<volatile u64_t *>(handles.blueflame_register) = doorbell;
    __threadfence_system();
}

/**
 *  @brief Reap exactly @b one signaled send completion entirely from device code — the GDAKI drain.
 *
 *  Acquire-spins on the consumer cell's owner bit until the NIC flips it to this pass's polarity, traps
 *  loudly on a requester error, then bumps the consumer index and publishes it to the set-consumer-index
 *  doorbell so the small ring never fills across replays. Call from @b one thread; bound every signaled
 *  post with one drain.
 */
__device__ inline void infiniband_drain_one(infiniband_completion_handles_t completion) noexcept {
    u32_t const consumer_index = *completion.consumer_index;
    u32_t const cell = consumer_index & (completion.entry_count - 1u);
    u8_t *cell_bytes = reinterpret_cast<u8_t *>(completion.ring.data()) +
                       static_cast<std::size_t>(cell) * completion.entry_size;
    auto *entry = reinterpret_cast<mlx5_cqe64 *>(cell_bytes + (completion.entry_size - sizeof(mlx5_cqe64)));
    u32_t const expected_owner = (completion.initial_owner ^ ((consumer_index >> completion.entry_log_count) & 1u)) &
                                 1u;

    while (true) {
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

/** @brief Rank-1 device view of the per-round link table, shaped @b [ceil(log2(D))] — one
 *      @ref infiniband_link_t per recursive-doubling round, indexed by round. */
using infiniband_links_view_t = nk::tensor_view<infiniband_link_t, 1>;
/** @brief Rank-1 device view of the flag-generation scalar, shaped @b [1] — the monotonic base every
 *      flag target keys on; the owning member advances it once per step. */
using infiniband_generation_view_t = nk::tensor_view<infiniband_generation_t, 1>;

/**
 *  @brief The static description of one all-gather invocation — the two scalars its kernels need
 *      beyond the links and the generation. Internal to @ref infiniband_member_t 's dispatch: the member
 *      bakes both at wiring time and builds this by value at enqueue; consumers never construct one.
 */
struct infiniband_allgather_plan_t {
    /** @brief Bytes per domain block; the gathered payload spans @b [D·block_bytes] at heap offset 0. */
    std::size_t block_bytes = 0;
    /** @brief This rank's multimem_domain index — the recursive-doubling coordinate. */
    u32_t this_multimem_domain = 0;
};

#if defined(UFORM_WITH_INFINIBAND)

/** @brief Rank-1 device-only table of fabric links owned through a @ref gpu_arena_t . */
using infiniband_links_tensor_t = nk::tensor<infiniband_link_t, gpu_arena_alloc<infiniband_link_t>, 1>;

/**
 *  @brief One member's wired reach into the other multimem_domains: the registered symmetric heap,
 *      one RC
 *      endpoint with its device-initiated queue / completion mappings per recursive-doubling round,
 *      the exchanged coordinates, the device-resident link table, and the collective's generation
 *      scalar — the level-2 counterpart of @ref multimem_domain_t 's per-member surface.
 *
 *  Constructed only through @ref colocated_domain_t 's role factories — @c wire_allreduce (a Float32
 *  shard summed across domains, per-round landings) or @c wire_allgather (domain-major blocks gathered
 *  in place, the payload doubling as the landing) — so a member's heap layout can never contradict the
 *  collective it runs, and the adapters underneath are always the shared ones. The collectives
 *  themselves are the argument-free methods @ref allreduce_sum and @ref allgather : payload, links,
 *  generation, and the domain coordinate are all baked at wiring time.
 */
struct infiniband_member_t {

  private:
    friend struct colocated_domain_t;

    /**
     *  @brief Byte offsets of one member's registered symmetric heap — the identical formula on every
     *      domain: the
     *      symmetric-heap contract the remote writes rely on. Layout: @b [payload | rounds landings |
     *      rounds flags] , every region 128-byte aligned; derived only by the role factories.
     */
    struct heap_layout_t {
        /** @brief The shard / world buffer the collective operates on — the heap's prefix. */
        std::size_t payload_bytes = 0;
        /** @brief Per-round landing size; 0 means the payload doubles as the landing. */
        std::size_t landing_bytes = 0;
        /** @brief Recursive-doubling rounds — the link, landing, and flag count. */
        u32_t rounds = 0;

        [[nodiscard]] std::size_t landings_offset() const noexcept {
            return round_up_to_multiple(payload_bytes, cache_line_bytes_k);
        }
        [[nodiscard]] std::size_t flags_offset() const noexcept {
            return round_up_to_multiple(landings_offset() + rounds * landing_bytes, cache_line_bytes_k);
        }
        [[nodiscard]] std::size_t total_bytes() const noexcept {
            return round_up_to_multiple(flags_offset() + rounds * sizeof(u32_t), cache_line_bytes_k);
        }
    };

    /** @brief The shared wiring body behind @ref colocated_domain_t 's role factories: allocate the heap,
     *      wire one endpoint per round on the shared @p nic , exchange coordinates, stage the links. The
     *      @p stream seeds the member's generation scalar (ordered after its zero). */
    [[nodiscard]] static expected<infiniband_member_t> make_(mesh_topology_t const &topology, u32_t member_index,
                                                             infiniband_nic_t &nic, gpu_arena_t &arena,
                                                             gpu_stream_t &stream, heap_layout_t layout,
                                                             sz::string_view key_prefix,
                                                             f64_t timeout_seconds) noexcept;

  public:
    /** @brief The registered payload region — the shard the all-reduce sums in place, or the
     *      domain-major world buffer the all-gather fills. */
    [[nodiscard]] b8r1_span_t payload() noexcept { return {heap_.data(), layout_.payload_bytes}; }
    /** @brief An all-gather member's own domain-major cell inside @ref payload — where this domain's
     *      block must be staged before @ref allgather ; the coordinate stays baked, never re-derived
     *      by consumers. */
    [[nodiscard]] b8r1_span_t own_block() noexcept {
        std::size_t const block_bytes = layout_.payload_bytes >> layout_.rounds;
        return {heap_.data() + static_cast<std::size_t>(multimem_domain_index_) * block_bytes, block_bytes};
    }
    /** @brief The one enqueued host write of a step: stage the generation into the pinned cell and
     *      enqueue its upload — on @p stream , before this step's collective. A no-op on the zero-round
     *      single-domain identity. */
    [[nodiscard]] expected_t advance(gpu_stream_t &stream) noexcept {
        if (layout_.rounds == 0) return {};
        return generation_.advance(stream);
    }

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
     *      as capture-safe graph nodes. This domain's own block must already sit at its domain-major
     *      cell; after @c log2(D) rounds every block is filled. Call @ref advance first, once per step.
     */
    [[nodiscard]] expected_t allgather(gpu_stream_t &stream) noexcept;

  private:
    /** @brief The heap's byte-offset formula — identical on every domain, baked by the role factory. */
    heap_layout_t layout_ {};
    /** @brief This rank's multimem_domain index — the recursive-doubling coordinate, baked at wiring. */
    u32_t multimem_domain_index_ = 0;
    /** @brief The member's monotonic step counter — host count, pinned staging, and the device
     *      generation scalar it uploads into; see @ref barrier_generation_t . */
    barrier_generation_t generation_ {};
    /** @brief The registered symmetric heap: payload, per-round landings, per-round flags. */
    b8r1_tensor_arena_t heap_ {};
    /** @brief One connected RC endpoint per round; the two mapping tables below layer onto them. */
    array<infiniband_endpoint_t> endpoints_ {};
    /** @brief Per-round device-initiated send paths, owning the GPU registrations of each queue pair's
     *      work queue, doorbell record, and BlueFlame register. */
    array<infiniband_queue_mapping_t> queues_ {};
    /** @brief Per-round completion parsers, owning the GPU registrations of each ring + doorbell. */
    array<infiniband_completion_mapping_t> completions_ {};
    /** @brief The device link table the doorbell kernels index by round. */
    infiniband_links_tensor_t links_ {};
    /** @brief The device generation scalar; see @ref advance . */
    u32r1_tensor_arena_t generation_scalar_ {};
};

/**
 *  @brief This process's membership of the mesh's colocated_domain: the opened NICs of every local GPU
 *      and the coordinate-exchange context — created @b once in main, right after the mesh bootstrap,
 *      and threaded by reference to every cross-domain consumer, exactly like the per-rank arenas and
 *      the one @ref multimem_domain_t .
 *
 *  Wiring a member is a consumer's own lane — its queue pairs (the single-poster concurrency
 *  granularity, so consumers' collectives can overlap), its heap, its generation; this domain only lends
 *  the shared adapters and the exchange. Batch-shuffle payload movement is added later as a third role
 *  ( @c wire_alltoall with per-pair sizes) without touching the NIC or member layers.
 */
struct colocated_domain_t {

    /** @brief Open each local GPU's adapter — @c mlx5_<m> for local ordinal @c m , the box's GPU↔HCA
     *      pairing — once for the process. The @c member loop spans @c gpus_per_multimem_domain under the
     *      current one-process-per-multimem_domain model; a multi-process island narrows it to the local
     *      GPU count alongside the cross-process multicast bind. */
    [[nodiscard]] static expected<colocated_domain_t> make(mesh_topology_t const &topology) noexcept;

    /**
     *  @brief Wire one member for the cross-domain Float32 all-reduce-sum of a @p shard_bytes payload:
     *      heap @b [shard | rounds landings | rounds flags] . One member per (consumer, role) — own
     *      queue pairs, own generation. Runs once, at setup, outside any capture; the calling thread must
     *      hold the member's context.
     *
     *  @param[in] member_index This GPU's index inside its multimem_domain; the partner's same-index
     *      GPU pairs with it.
     *  @param[in] arena The member's arena; the heap, the link table, and the mapping counters allocate
     *      from it (its chunks are GPUDirect-RDMA-capable by construction).
     *  @param[in] stream The member's lead stream; the generation scalar is seeded on it at setup.
     *  @param[in] key_prefix Exchange-key namespace; keys are @c "<prefix>.<round>" , so distinct
     *      consumers never collide.
     */
    [[nodiscard]] expected<infiniband_member_t> wire_allreduce(u32_t member_index, gpu_arena_t &arena,
                                                               gpu_stream_t &stream, std::size_t shard_bytes,
                                                               sz::string_view key_prefix,
                                                               f64_t timeout_seconds = 120.0) noexcept;

    /** @brief Wire one member for the cross-domain all-gather of domain-major @p block_bytes blocks:
     *      the payload spans @b [domains·block_bytes] and doubles as the landing — blocks land in
     *      place. Same contract as @ref wire_allreduce otherwise. */
    [[nodiscard]] expected<infiniband_member_t> wire_allgather(u32_t member_index, gpu_arena_t &arena,
                                                               gpu_stream_t &stream, std::size_t block_bytes,
                                                               sz::string_view key_prefix,
                                                               f64_t timeout_seconds = 120.0) noexcept;

    /** @brief The bootstrapped world this domain was wired against. */
    [[nodiscard]] mesh_topology_t const &topology() const noexcept { return *topology_; }
    /** @brief How many multimem_domains the world spans — 1 means no member ever posts. */
    [[nodiscard]] u32_t multimem_domain_count() const noexcept { return topology_->shape.multimem_domain_count(); }

  private:
    /** @brief The bootstrapped world — borrowed; main owns it for the process lifetime. */
    mesh_topology_t const *topology_ = nullptr;
    /** @brief One opened adapter per local GPU, member order. */
    per_gpu_array<infiniband_nic_t> nics_ {};
};

#else // !defined(UFORM_WITH_INFINIBAND)

/** @brief The build-gate stub: only the zero-round single-domain identity exists — every collective is
 *      a no-op, and wiring a real member fails loudly naming the gate. */
struct infiniband_member_t {
    [[nodiscard]] b8r1_span_t payload() noexcept { return {}; }
    [[nodiscard]] b8r1_span_t own_block() noexcept { return {}; }
    [[nodiscard]] expected_t advance(gpu_stream_t &) noexcept { return {}; }
    [[nodiscard]] expected_t allreduce_sum(gpu_stream_t &) noexcept { return {}; }
    [[nodiscard]] expected_t allgather(gpu_stream_t &) noexcept { return {}; }
};

/** @brief The build-gate stub of the colocated domain: single-domain topologies work (nothing to
 *      wire); multi-domain worlds fail loudly naming the gate. */
struct colocated_domain_t {
    [[nodiscard]] static expected<colocated_domain_t> make(mesh_topology_t const &topology) noexcept {
        if (topology.shape.multimem_domain_count() != 1)
            return expected<colocated_domain_t>::fail(
                status_t::other_k,
                "built without UFORM_WITH_INFINIBAND — rebuild with libibverbs + libmlx5 for " //
                "cross-domain collectives");
        colocated_domain_t domain;
        domain.topology_ = &topology;
        return expected<colocated_domain_t> {std::move(domain)};
    }
    [[nodiscard]] expected<infiniband_member_t> wire_allreduce(u32_t, gpu_arena_t &, gpu_stream_t &, std::size_t,
                                                               sz::string_view, f64_t = 120.0) noexcept {
        return expected<infiniband_member_t> {infiniband_member_t {}};
    }
    [[nodiscard]] expected<infiniband_member_t> wire_allgather(u32_t, gpu_arena_t &, gpu_stream_t &, std::size_t,
                                                               sz::string_view, f64_t = 120.0) noexcept {
        return expected<infiniband_member_t> {infiniband_member_t {}};
    }
    [[nodiscard]] mesh_topology_t const &topology() const noexcept { return *topology_; }
    [[nodiscard]] u32_t multimem_domain_count() const noexcept { return 1u; }

  private:
    mesh_topology_t const *topology_ = nullptr;
};

#endif // defined(UFORM_WITH_INFINIBAND)

} // namespace unum::uform::core
