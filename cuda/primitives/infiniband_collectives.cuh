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
 *  @section collectives_epochs Arrival flags are epochs, never events
 *
 *  Every round's arrival proof is a 4-byte @b inline flag write on the @b same queue pair as the payload:
 *  Reliable-Connection delivery is in-order, so the flag's arrival proves the payload landed. Flags are
 *  @b monotonic never-reset counters — a waiter spins with @c >= against the absolute target
 *  @c *epoch+round+1 , so replays of a captured step need no flag reset (a reset would be a host write
 *  inside the replay loop). Each @ref infiniband_member_t owns its epoch scalar and @c advance() s it
 *  once per step by its round count — distinct members' flags can never alias, because they live in
 *  distinct heaps.
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

#include <charconv> // `std::from_chars` — coordinate payload parsing
#include <chrono>   // `std::chrono::steady_clock` — the completion-poll deadline
#include <span>     // `std::span` — the DMA ring ranges inside the wire PODs
#include <thread>   // `std::this_thread::sleep_for` — the completion-poll backoff
#include <utility>  // `std::exchange`

#include <cuda.h> // `cuMemGetHandleForAddressRange`, `cuMemHostRegister`, `cuMemHostGetDevicePointer`

#if defined(UFORM_WITH_INFINIBAND)
#include <unistd.h> // `::close`, `sysconf` — dmabuf descriptors and page-aligned registrations

#include <infiniband/mlx5dv.h> // `mlx5dv_init_obj`, `mlx5_wqe_*`, `mlx5_cqe64` — the direct-verbs layer
#include <infiniband/verbs.h>  // `ibv_context`, `ibv_pd`, `ibv_cq`, `ibv_qp` — the RC bring-up
#endif

#include "types.cuh"          // `expected`, `expected_t`, `status_t`, `format_into`
#include "gpu_allocators.cuh" // `gpu_arena_t` carving the device producer / consumer counters
#include "schedule.cuh"       // `b8r1_span_t` — the registered GPU range travels as one typed span
#include "mesh.cuh"           // `mesh_topology_t` — the coordinate exchange rides the mesh rendezvous

namespace unum::uform::primitives {

/** @brief Monotonic never-reset arrival counter the GDAKI collectives spin on; lives in device memory. */
using infiniband_epoch_t = u32_t;

/** @brief The send-doorbell slot in the mlx5 two-entry doorbell-record array (entry 0 is the receive side). */
inline constexpr std::size_t mlx5_send_doorbell_index_k = 1;
/** @brief The set-consumer-index slot in the mlx5 completion-queue doorbell array (entry 1 arms interrupts). */
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
    /** @brief The big-endian doorbell-record array; the send slot holds the producer counter the NIC polls. */
    u32_t *doorbell_record = nullptr;
    /** @brief The BlueFlame / User-Access-Region doorbell register — NIC BAR / MMIO space. */
    void *blueflame_register = nullptr;
    /** @brief Device counter of posted entries (the producer index); atomically claimed per post. */
    u32_t *producer_index = nullptr;
    /** @brief The queue pair number, baked into every control segment. */
    u32_t queue_pair_number = 0;
    /** @brief Send-queue capacity in basic blocks; the producer index wraps modulo this. */
    u32_t send_queue_entry_count = 0;
    /** @brief Bytes per basic block (64 on ConnectX-7); the slot stride in the buffer. */
    u32_t send_queue_stride = 0;
};

/**
 *  @brief The handles a kernel needs to reap one signaled send completion — the drain counterpart of
 *      @ref infiniband_queue_handles_t . The three hardware unknowns ( @c entry_size , @c initial_owner ,
 *      @c compressed ) are captured at mapping time; the valid 64-byte entry lives in the @b high bytes
 *      of a wider slot.
 */
struct infiniband_completion_handles_t {
    /** @brief The completion-entry ring (host DMA memory the NIC writes into), spanning
     *      @c entry_count·entry_size bytes. */
    std::span<std::byte> ring {};
    /** @brief The big-endian completion doorbell record; set-consumer-index at slot 0. */
    u32_t *doorbell_record = nullptr;
    /** @brief Device counter of reaped completions (the consumer index), in GPU VRAM. */
    u32_t *consumer_index = nullptr;
    /** @brief Ring capacity in entries — a power of two; the consumer index masks modulo it. */
    u32_t entry_count = 0;
    /** @brief Bytes per ring slot (64 or 128); the valid 64-byte entry sits in the high bytes. */
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
    infiniband_epoch_t *arrival_flag = nullptr;
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
 *  @brief One (GPU, NIC) Reliable-Connection endpoint talking to one remote GPU: device context,
 *      protection domain, completion queue, RC queue pair, and a memory region over GPU VRAM.
 *
 *  The region registers dmabuf-first ( @c cuMemGetHandleForAddressRange exports the range,
 *  @c ibv_reg_dmabuf_mr registers the descriptor with the GPU virtual address as the I/O address so peer
 *  @c buffer_address+offset math lands on the right byte); on any failure it falls back to a plain
 *  @c ibv_reg_mr serviced by the @c nvidia_peermem kernel module. Host-posted writes and completion polls
 *  remain the bring-up / smoke-test path; production traffic runs through the GDAKI mappings layered on
 *  this endpoint's queue pair. Move-only; teardown reverses construction: region, queue pair, completion
 *  queue, protection domain, dmabuf descriptor, device.
 */
struct infiniband_endpoint_t {

    infiniband_endpoint_t() noexcept = default;
    infiniband_endpoint_t(infiniband_endpoint_t const &) = delete;
    infiniband_endpoint_t &operator=(infiniband_endpoint_t const &) = delete;

    infiniband_endpoint_t(infiniband_endpoint_t &&other) noexcept
        : device_context_(std::exchange(other.device_context_, nullptr)),
          protection_domain_(std::exchange(other.protection_domain_, nullptr)),
          completion_queue_(std::exchange(other.completion_queue_, nullptr)),
          queue_pair_(std::exchange(other.queue_pair_, nullptr)),
          memory_region_(std::exchange(other.memory_region_, nullptr)),
          gpu_buffer_(std::exchange(other.gpu_buffer_, b8r1_span_t {})),
          dmabuf_descriptor_(std::exchange(other.dmabuf_descriptor_, -1)), infiniband_port_(other.infiniband_port_),
          path_mtu_(other.path_mtu_), local_coordinates_(other.local_coordinates_),
          peer_coordinates_(other.peer_coordinates_) {}
    infiniband_endpoint_t &operator=(infiniband_endpoint_t &&other) noexcept {
        if (this != &other) {
            destroy_();
            device_context_ = std::exchange(other.device_context_, nullptr);
            protection_domain_ = std::exchange(other.protection_domain_, nullptr);
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
     *  @brief Open @p nic_name , build the protection domain / completion queue / RC queue pair, and
     *      register @p gpu_buffer as a GPUDirect memory region — everything a peer handshake needs.
     *
     *  @param[in] nic_name The mlx5 device name (e.g. @c "mlx5_0" ) from the ibverbs device list.
     *  @param[in] infiniband_port The port on that adapter; single-port ConnectX parts use 1.
     *  @param[in] gpu_buffer The GPU VRAM range peers will read and write, shaped @b [buffer_bytes] —
     *      typically a window over a @ref gpu_arena_t (its chunks are created GPUDirect-RDMA-capable).
     */
    [[nodiscard]] static expected<infiniband_endpoint_t> make(char const *nic_name, u8_t infiniband_port,
                                                              b8r1_span_t gpu_buffer) noexcept {
        if (nic_name == nullptr || gpu_buffer.empty())
            return expected<infiniband_endpoint_t>::fail(status_t::invalid_input_k,
                                                         "an endpoint needs a NIC name and a non-empty GPU buffer");
        infiniband_endpoint_t endpoint;
        endpoint.infiniband_port_ = infiniband_port;
        endpoint.gpu_buffer_ = gpu_buffer;

        if (expected_t opened = endpoint.open_device_(nic_name); !opened)
            return expected<infiniband_endpoint_t>::fail(std::move(opened));

        ibv_port_attr port_attributes {};
        if (ibv_query_port(endpoint.device_context_, infiniband_port, &port_attributes) != 0)
            return expected<infiniband_endpoint_t>::fail(status_t::io_error_k, "port {} query failed on \"{}\"",
                                                         infiniband_port, nic_name);
        endpoint.path_mtu_ = static_cast<unsigned>(port_attributes.active_mtu);
        ibv_gid port_gid {};
        if (ibv_query_gid(endpoint.device_context_, infiniband_port, 0, &port_gid) != 0)
            return expected<infiniband_endpoint_t>::fail(status_t::io_error_k, "GID query failed on \"{}\" port {}",
                                                         nic_name, infiniband_port);

        endpoint.protection_domain_ = ibv_alloc_pd(endpoint.device_context_);
        if (endpoint.protection_domain_ == nullptr)
            return expected<infiniband_endpoint_t>::fail(status_t::io_error_k,
                                                         "protection-domain allocation failed on {}", nic_name);
        endpoint.completion_queue_ = ibv_create_cq(endpoint.device_context_, infiniband_queue_depth_k, nullptr, nullptr,
                                                   0);
        if (endpoint.completion_queue_ == nullptr)
            return expected<infiniband_endpoint_t>::fail(status_t::io_error_k, "completion-queue creation failed on {}",
                                                         nic_name);

        ibv_qp_init_attr queue_pair_config {};
        queue_pair_config.send_cq = endpoint.completion_queue_;
        queue_pair_config.recv_cq = endpoint.completion_queue_;
        queue_pair_config.qp_type = IBV_QPT_RC;
        queue_pair_config.sq_sig_all = 0;
        queue_pair_config.cap.max_send_wr = infiniband_queue_depth_k;
        queue_pair_config.cap.max_recv_wr = infiniband_queue_depth_k;
        queue_pair_config.cap.max_send_sge = 1;
        queue_pair_config.cap.max_recv_sge = 1;
        endpoint.queue_pair_ = ibv_create_qp(endpoint.protection_domain_, &queue_pair_config);
        if (endpoint.queue_pair_ == nullptr)
            return expected<infiniband_endpoint_t>::fail(
                status_t::io_error_k, "Reliable-Connection queue-pair creation failed on {}", nic_name);

        if (expected_t registered = endpoint.register_gpu_memory_(); !registered)
            return expected<infiniband_endpoint_t>::fail(std::move(registered));

        infiniband_coordinates_t &coordinates = endpoint.local_coordinates_;
        coordinates.local_identifier = port_attributes.lid;
        coordinates.queue_pair_number = endpoint.queue_pair_->qp_num;
        coordinates.packet_sequence_number = 0;
        coordinates.remote_key = endpoint.memory_region_->rkey;
        coordinates.buffer_address = reinterpret_cast<u64_t>(gpu_buffer.data());
        std::memcpy(coordinates.global_identifier, port_gid.raw, sizeof(coordinates.global_identifier));
        return expected<infiniband_endpoint_t> {std::move(endpoint)};
    }

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
    [[nodiscard]] expected_t connect(infiniband_coordinates_t const &peer) noexcept {
        peer_coordinates_ = peer;

        ibv_qp_attr to_initialized {};
        to_initialized.qp_state = IBV_QPS_INIT;
        to_initialized.pkey_index = 0;
        to_initialized.port_num = infiniband_port_;
        to_initialized.qp_access_flags = infiniband_access_flags_k;
        int const initialized_mask = IBV_QP_STATE | IBV_QP_PKEY_INDEX | IBV_QP_PORT | IBV_QP_ACCESS_FLAGS;
        if (ibv_modify_qp(queue_pair_, &to_initialized, initialized_mask) != 0)
            return expected_t::fail(status_t::io_error_k, "queue-pair transition to INIT failed");

        ibv_qp_attr to_ready_to_receive {};
        to_ready_to_receive.qp_state = IBV_QPS_RTR;
        to_ready_to_receive.path_mtu = static_cast<ibv_mtu>(path_mtu_);
        to_ready_to_receive.dest_qp_num = peer.queue_pair_number;
        to_ready_to_receive.rq_psn = peer.packet_sequence_number;
        to_ready_to_receive.max_dest_rd_atomic = 1;
        to_ready_to_receive.min_rnr_timer = 12;
        to_ready_to_receive.ah_attr.is_global = 0;
        to_ready_to_receive.ah_attr.dlid = static_cast<u16_t>(peer.local_identifier);
        to_ready_to_receive.ah_attr.sl = 0;
        to_ready_to_receive.ah_attr.src_path_bits = 0;
        to_ready_to_receive.ah_attr.port_num = infiniband_port_;
        int const ready_to_receive_mask = IBV_QP_STATE | IBV_QP_AV | IBV_QP_PATH_MTU | IBV_QP_DEST_QPN | IBV_QP_RQ_PSN |
                                          IBV_QP_MAX_DEST_RD_ATOMIC | IBV_QP_MIN_RNR_TIMER;
        if (ibv_modify_qp(queue_pair_, &to_ready_to_receive, ready_to_receive_mask) != 0)
            return expected_t::fail(status_t::io_error_k, "queue-pair transition to RTR failed (peer lid={} qpn={})",
                                    peer.local_identifier, peer.queue_pair_number);

        ibv_qp_attr to_ready_to_send {};
        to_ready_to_send.qp_state = IBV_QPS_RTS;
        to_ready_to_send.sq_psn = local_coordinates_.packet_sequence_number;
        to_ready_to_send.timeout = 14;
        to_ready_to_send.retry_cnt = 7;
        to_ready_to_send.rnr_retry = 7;
        to_ready_to_send.max_rd_atomic = 1;
        int const ready_to_send_mask = IBV_QP_STATE | IBV_QP_SQ_PSN | IBV_QP_TIMEOUT | IBV_QP_RETRY_CNT |
                                       IBV_QP_RNR_RETRY | IBV_QP_MAX_QP_RD_ATOMIC;
        if (ibv_modify_qp(queue_pair_, &to_ready_to_send, ready_to_send_mask) != 0)
            return expected_t::fail(status_t::io_error_k, "queue-pair transition to RTS failed");
        return {};
    }

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
    /** @brief Open the mlx5 device named @p nic_name from the ibverbs device list. */
    [[nodiscard]] expected_t open_device_(char const *nic_name) noexcept {
        int device_count = 0;
        ibv_device **device_list = ibv_get_device_list(&device_count);
        if (device_list == nullptr)
            return expected_t::fail(status_t::io_error_k, "no RDMA devices visible (is the RDMA stack loaded?)");
        for (int device_index = 0; device_index != device_count; ++device_index) {
            char const *name = ibv_get_device_name(device_list[device_index]);
            if (name != nullptr && std::strcmp(name, nic_name) == 0) {
                device_context_ = ibv_open_device(device_list[device_index]);
                break;
            }
        }
        ibv_free_device_list(device_list);
        if (device_context_ == nullptr)
            return expected_t::fail(status_t::io_error_k, "RDMA device \"{}\" not found or failed to open", nic_name);
        return {};
    }

    /** @brief Register the GPU buffer as a GPUDirect region: dmabuf-first, @c nvidia_peermem fallback. */
    [[nodiscard]] expected_t register_gpu_memory_() noexcept {
        int dmabuf_descriptor = -1;
        CUresult const exported = cuMemGetHandleForAddressRange(
            &dmabuf_descriptor, reinterpret_cast<CUdeviceptr>(gpu_buffer_.data()), gpu_buffer_.extent(0),
            CU_MEM_RANGE_HANDLE_TYPE_DMA_BUF_FD, 0);
        if (exported == CUDA_SUCCESS && dmabuf_descriptor >= 0) {
            memory_region_ = ibv_reg_dmabuf_mr(protection_domain_, 0, gpu_buffer_.extent(0),
                                               reinterpret_cast<u64_t>(gpu_buffer_.data()), dmabuf_descriptor,
                                               infiniband_access_flags_k);
            if (memory_region_ != nullptr) {
                dmabuf_descriptor_ = dmabuf_descriptor;
                return {};
            }
            ::close(dmabuf_descriptor);
        }
        memory_region_ = ibv_reg_mr(protection_domain_, gpu_buffer_.data(), gpu_buffer_.extent(0),
                                    infiniband_access_flags_k);
        if (memory_region_ == nullptr)
            return expected_t::fail(status_t::io_error_k,
                                    "GPU memory-region registration failed on both the dmabuf and the " //
                                    "nvidia_peermem paths ({} bytes)",
                                    gpu_buffer_.extent(0));
        return {};
    }

    void destroy_() noexcept {
        if (memory_region_ != nullptr) ibv_dereg_mr(std::exchange(memory_region_, nullptr));
        if (queue_pair_ != nullptr) ibv_destroy_qp(std::exchange(queue_pair_, nullptr));
        if (completion_queue_ != nullptr) ibv_destroy_cq(std::exchange(completion_queue_, nullptr));
        if (protection_domain_ != nullptr) ibv_dealloc_pd(std::exchange(protection_domain_, nullptr));
        if (dmabuf_descriptor_ >= 0) ::close(std::exchange(dmabuf_descriptor_, -1));
        if (device_context_ != nullptr) ibv_close_device(std::exchange(device_context_, nullptr));
        gpu_buffer_ = b8r1_span_t {};
    }

    ibv_context *device_context_ = nullptr;
    ibv_pd *protection_domain_ = nullptr;
    ibv_cq *completion_queue_ = nullptr;
    ibv_qp *queue_pair_ = nullptr;
    ibv_mr *memory_region_ = nullptr;
    b8r1_span_t gpu_buffer_ {};
    int dmabuf_descriptor_ = -1;
    u8_t infiniband_port_ = 1;
    unsigned path_mtu_ = 0;
    infiniband_coordinates_t local_coordinates_ {};
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
     *      three for the GPU; the producer counter is carved from @p arena (capture-stable address). */
    [[nodiscard]] static expected<infiniband_queue_mapping_t> make(ibv_qp *queue_pair, gpu_arena_t &arena) noexcept {
        if (queue_pair == nullptr)
            return expected<infiniband_queue_mapping_t>::fail(status_t::invalid_input_k, "null queue pair");
        mlx5dv_qp raw_queue {};
        mlx5dv_obj object {};
        object.qp.in = queue_pair;
        object.qp.out = &raw_queue;
        if (mlx5dv_init_obj(&object, MLX5DV_OBJ_QP) != 0)
            return expected<infiniband_queue_mapping_t>::fail(status_t::io_error_k,
                                                              "the mlx5 provider refused to expose the queue pair");
        if (raw_queue.sq.buf == nullptr || raw_queue.dbrec == nullptr || raw_queue.bf.reg == nullptr ||
            raw_queue.sq.wqe_cnt == 0)
            return expected<infiniband_queue_mapping_t>::fail(
                status_t::io_error_k, "the mlx5 provider exposed a null send queue / doorbell / BlueFlame register");

        infiniband_queue_mapping_t mapping;
        mapping.handles_.queue_pair_number = queue_pair->qp_num;
        mapping.handles_.send_queue_entry_count = raw_queue.sq.wqe_cnt;
        mapping.handles_.send_queue_stride = raw_queue.sq.stride;

        std::size_t const send_queue_bytes = static_cast<std::size_t>(raw_queue.sq.wqe_cnt) *
                                             static_cast<std::size_t>(raw_queue.sq.stride);
        void *send_queue_device = nullptr;
        if (expected_t mapped = infiniband_map_for_device(raw_queue.sq.buf, send_queue_bytes, false,
                                                          mapping.send_queue_registration_, send_queue_device);
            !mapped)
            return expected<infiniband_queue_mapping_t>::fail(std::move(mapped));
        mapping.handles_.send_queue = {static_cast<std::byte *>(send_queue_device), send_queue_bytes};
        void *doorbell_device = nullptr;
        if (expected_t mapped = infiniband_map_for_device(raw_queue.dbrec, sizeof(u32_t) * 2, false,
                                                          mapping.doorbell_registration_, doorbell_device);
            !mapped)
            return expected<infiniband_queue_mapping_t>::fail(std::move(mapped));
        mapping.handles_.doorbell_record = static_cast<u32_t *>(doorbell_device);

        std::size_t const blueflame_bytes = raw_queue.bf.size != 0 ? raw_queue.bf.size : sizeof(u64_t);
        if (expected_t mapped = infiniband_map_for_device(raw_queue.bf.reg, blueflame_bytes, true,
                                                          mapping.blueflame_registration_,
                                                          mapping.handles_.blueflame_register);
            !mapped)
            return expected<infiniband_queue_mapping_t>::fail( //
                status_t::io_error_k,
                "the BlueFlame MMIO page would not map into GPU address space (CU_MEMHOSTREGISTER_IOMEMORY) — " //
                "device-initiated GDAKI is blocked on this platform: {}",
                mapped.description_);

        mapping.handles_.producer_index = arena.allocate_object<u32_t>(1);
        if (mapping.handles_.producer_index == nullptr)
            return expected<infiniband_queue_mapping_t>::fail(
                status_t::bad_alloc_k, "producer-index counter carve failed (arena_used={})", arena.used());
        if (CUresult result = cuMemsetD8(reinterpret_cast<CUdeviceptr>(mapping.handles_.producer_index), 0,
                                         sizeof(u32_t));
            result != CUDA_SUCCESS)
            return expected<infiniband_queue_mapping_t>::fail(result, "producer-index counter reset failed");
        return expected<infiniband_queue_mapping_t> {std::move(mapping)};
    }

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
     *      consumer counter is carved from @p arena . The ring must be a power of two and uncompressed. */
    [[nodiscard]] static expected<infiniband_completion_mapping_t> make(ibv_cq *completion_queue,
                                                                        gpu_arena_t &arena) noexcept {
        if (completion_queue == nullptr)
            return expected<infiniband_completion_mapping_t>::fail(status_t::invalid_input_k, "null completion queue");
        mlx5dv_cq raw_completion {};
        mlx5dv_obj object {};
        object.cq.in = completion_queue;
        object.cq.out = &raw_completion;
        if (mlx5dv_init_obj(&object, MLX5DV_OBJ_CQ) != 0)
            return expected<infiniband_completion_mapping_t>::fail(
                status_t::io_error_k, "the mlx5 provider refused to expose the completion queue");
        if (raw_completion.buf == nullptr || raw_completion.dbrec == nullptr || raw_completion.cqe_cnt == 0 ||
            raw_completion.cqe_size == 0)
            return expected<infiniband_completion_mapping_t>::fail(
                status_t::io_error_k, "the mlx5 provider exposed a null completion ring / doorbell");
        if ((raw_completion.cqe_cnt & (raw_completion.cqe_cnt - 1u)) != 0)
            return expected<infiniband_completion_mapping_t>::fail(
                status_t::shape_unsupported_k, "the completion ring must be a power of two (got {})",
                raw_completion.cqe_cnt);

        infiniband_completion_mapping_t mapping;
        mapping.handles_.entry_count = raw_completion.cqe_cnt;
        mapping.handles_.entry_size = raw_completion.cqe_size;
        u32_t log_count = 0;
        while ((raw_completion.cqe_cnt >> log_count) > 1u) ++log_count;
        mapping.handles_.entry_log_count = log_count;
        mapping.handles_.completion_queue_number = raw_completion.cqn;
        // Owner-bit polarity and compression, confirmed at the two-node bring-up gate: a fresh ring is
        // pre-initialized with the invalid opcode, so the drain separates "not yet written" by opcode;
        // owner polarity at consumer index zero is 0, and the ring is uncompressed 64-byte entries.
        mapping.handles_.initial_owner = 0u;
        mapping.handles_.compressed = 0u;

        std::size_t const ring_bytes = static_cast<std::size_t>(raw_completion.cqe_cnt) *
                                       static_cast<std::size_t>(raw_completion.cqe_size);
        void *ring_device = nullptr;
        if (expected_t mapped = infiniband_map_for_device(raw_completion.buf, ring_bytes, false,
                                                          mapping.ring_registration_, ring_device);
            !mapped)
            return expected<infiniband_completion_mapping_t>::fail(std::move(mapped));
        mapping.handles_.ring = {static_cast<std::byte *>(ring_device), ring_bytes};
        void *doorbell_device = nullptr;
        if (expected_t mapped = infiniband_map_for_device(raw_completion.dbrec, sizeof(u32_t) * 2, false,
                                                          mapping.doorbell_registration_, doorbell_device);
            !mapped)
            return expected<infiniband_completion_mapping_t>::fail(std::move(mapped));
        mapping.handles_.doorbell_record = static_cast<u32_t *>(doorbell_device);

        mapping.handles_.consumer_index = arena.allocate_object<u32_t>(1);
        if (mapping.handles_.consumer_index == nullptr)
            return expected<infiniband_completion_mapping_t>::fail(
                status_t::bad_alloc_k, "consumer-index counter carve failed (arena_used={})", arena.used());
        if (CUresult result = cuMemsetD8(reinterpret_cast<CUdeviceptr>(mapping.handles_.consumer_index), 0,
                                         sizeof(u32_t));
            result != CUDA_SUCCESS)
            return expected<infiniband_completion_mapping_t>::fail(result, "consumer-index counter reset failed");
        return expected<infiniband_completion_mapping_t> {std::move(mapping)};
    }

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
 *  Claims the next producer slot, writes the control + remote-address segments followed by @b either a
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
    u32_t const slot_offset = (producer_index % handles.send_queue_entry_count) * handles.send_queue_stride;
    u8_t *entry_bytes = reinterpret_cast<u8_t *>(handles.send_queue.data()) + slot_offset;

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
 *  Acquire-spins on the consumer slot's owner bit until the NIC flips it to this pass's polarity, traps
 *  loudly on a requester error, then bumps the consumer index and publishes it to the set-consumer-index
 *  doorbell so the small ring never fills across replays. Call from @b one thread; bound every signaled
 *  post with one drain.
 */
__device__ inline void infiniband_drain_one(infiniband_completion_handles_t completion) noexcept {
    u32_t const consumer_index = *completion.consumer_index;
    u32_t const slot = consumer_index & (completion.entry_count - 1u);
    u8_t *slot_bytes = reinterpret_cast<u8_t *>(completion.ring.data()) +
                       static_cast<std::size_t>(slot) * completion.entry_size;
    auto *entry = reinterpret_cast<mlx5_cqe64 *>(slot_bytes + (completion.entry_size - sizeof(mlx5_cqe64)));
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
/** @brief Rank-1 device view of the flag-epoch scalar, shaped @b [1] — the monotonic base every flag
 *      target keys on; the owning member advances it once per step. */
using infiniband_epoch_view_t = nk::tensor_view<infiniband_epoch_t, 1>;

/**
 *  @brief The static description of one all-gather invocation — the two scalars its kernels need
 *      beyond the links and the epoch. Internal to @ref infiniband_member_t 's dispatch: the member
 *      bakes both at wiring time and builds this by value at enqueue; consumers never construct one.
 */
struct infiniband_allgather_plan_t {
    /** @brief Bytes per domain block; the gathered payload spans @b [D·block_bytes] at heap offset 0. */
    std::size_t block_bytes = 0;
    /** @brief This rank's multimem_domain index — the recursive-doubling coordinate. */
    u32_t this_multimem_domain = 0;
};

#if defined(UFORM_WITH_INFINIBAND)

#if defined(__CUDACC__)

/** @brief Lane width of the single-block doorbell kernels; only lane 0 posts and spins. */
inline constexpr unsigned infiniband_doorbell_block_threads_k = 32u;
/** @brief Grid cap of the bandwidth-bound accumulate so a huge shard still launches bounded (grid-strided). */
inline constexpr unsigned infiniband_accumulate_grid_cap_k = 4096u;

/**
 *  @brief Post-and-drain step of one all-reduce round: lane 0 posts this domain's shard, then the arrival
 *      flag, to the round's partner, and drains the one signaled completion.
 *
 *  The payload write is unsignaled; only the 4-byte inline flag write is signaled — exactly one
 *  completion per round is produced and drained, so the small send ring never overflows across replays.
 */
static __global__ void infiniband_allreduce_post_kernel_(infiniband_links_view_t links, u32_t round, f32_t const *shard,
                                                         std::size_t shard_scalars,
                                                         infiniband_epoch_view_t flag_epoch) {
    if (threadIdx.x != 0) return;
    infiniband_link_t const link = links[round];
    u32_t const flag_value = flag_epoch[0] + round + 1u;
    infiniband_post_write(link.queue, reinterpret_cast<u64_t>(shard), link.local_access_key,
                          link.remote_heap_base + link.landing_offset, link.remote_access_key,
                          static_cast<u32_t>(shard_scalars * sizeof(f32_t)), false, nullptr);
    infiniband_post_write(link.queue, 0u, link.local_access_key, link.remote_heap_base + link.remote_flag_offset,
                          link.remote_access_key, sizeof(u32_t), true, &flag_value);
    infiniband_drain_one(link.completion);
}

/** @brief Flag-wait step of one round: lane 0 acquire-spins until the inbound flag reaches the absolute
 *      target (monotonic @c >= — the flag never resets), then a system fence orders the partner's landed
 *      payload before the next kernel, propagated to the block by the closing @c __syncthreads . */
static __global__ void infiniband_flag_wait_kernel_(infiniband_links_view_t links, u32_t round,
                                                    infiniband_epoch_view_t flag_epoch) {
    if (threadIdx.x == 0) {
        infiniband_link_t const link = links[round];
        u32_t const target = flag_epoch[0] + round + 1u;
        while (infiniband_load_acquire_u32(link.arrival_flag) < target) {
            // Spin until the partner's flag write for this round lands; its payload already did.
        }
        __threadfence_system();
    }
    __syncthreads();
}

/** @brief Accumulate step of one all-reduce round: fold the partner's landed shard into this domain's,
 *      one explicit round-to-nearest-even add per element — no fused-multiply-add contraction, so the
 *      SPMD-static fold order is bit-identical on every domain. */
static __global__ void infiniband_accumulate_f32_kernel_(infiniband_links_view_t links, u32_t round, f32_t *shard,
                                                         std::size_t shard_scalars) {
    infiniband_link_t const link = links[round];
    f32_t const *landing = reinterpret_cast<f32_t const *>(link.local_landing.data());
    std::size_t const stride = static_cast<std::size_t>(gridDim.x) * blockDim.x;
    for (std::size_t index = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x; index < shard_scalars;
         index += stride)
        shard[index] = __fadd_rn(shard[index], landing[index]);
}

/**
 *  @brief Post-and-drain step of one all-gather round: lane 0 posts the contiguous run of domain blocks
 *      this domain currently holds into the partner's symmetric @c gathered buffer at the @b same byte
 *      offset, then the flag, and drains the one completion.
 *
 *  At round @c r the held run spans domains @c [base,base+2^r) with @c base=this_domain&~(2^r-1) ;
 *  symmetric buffers make identical byte offsets address identical logical slots, so the partner receives
 *  exactly the half it lacks at the right domain-major position.
 */
static __global__ void infiniband_allgather_post_kernel_(infiniband_links_view_t links, u32_t round,
                                                         infiniband_allgather_plan_t plan, char const *gathered_bytes,
                                                         infiniband_epoch_view_t flag_epoch) {
    if (threadIdx.x != 0) return;
    infiniband_link_t const link = links[round];
    // The recursive-doubling bit math keys on the link's true round; the loop index drives only the
    // table slot and the monotonic flag window. The gathered payload sits at heap offset 0 on every
    // domain — the symmetric contract — so identical byte offsets address identical logical slots.
    u32_t const bit_round = link.round_index;
    u32_t const group_base = plan.this_multimem_domain & ~((1u << bit_round) - 1u);
    std::size_t const send_byte_offset = static_cast<std::size_t>(group_base) * plan.block_bytes;
    std::size_t const send_bytes = (static_cast<std::size_t>(1u) << bit_round) * plan.block_bytes;
    u32_t const flag_value = flag_epoch[0] + round + 1u;

    infiniband_post_write(link.queue, reinterpret_cast<u64_t>(gathered_bytes + send_byte_offset), link.local_access_key,
                          link.remote_heap_base + send_byte_offset, link.remote_access_key,
                          static_cast<u32_t>(send_bytes), false, nullptr);
    infiniband_post_write(link.queue, 0u, link.local_access_key, link.remote_heap_base + link.remote_flag_offset,
                          link.remote_access_key, sizeof(u32_t), true, &flag_value);
    infiniband_drain_one(link.completion);
}

#endif // defined(__CUDACC__)

/** @brief Rank-1 device-only table of fabric links owned through a @ref gpu_arena_t . */
using infiniband_links_tensor_t = nk::tensor<infiniband_link_t, gpu_arena_alloc<infiniband_link_t>, 1>;

/**
 *  @brief One member's wired reach into the other multimem_domains: the registered symmetric heap,
 *      one RC
 *      endpoint with its device-initiated queue / completion mappings per recursive-doubling round,
 *      the exchanged coordinates, the device-resident link table, and the collective's epoch scalar —
 *      the level-2 counterpart of @ref multimem_domain_t 's per-member surface.
 *
 *  Constructed only through the role factories — @ref make_allreduce (a Float32 shard summed across
 *  domains, per-round landings) or @ref make_allgather (domain-major blocks gathered in place, the
 *  payload doubling as the landing) — so a member's heap layout can never contradict the collective it
 *  runs. The collectives themselves are the argument-free methods @ref allreduce_sum and
 *  @ref allgather : payload, links, epoch, and the domain coordinate are all baked at wiring time.
 */
struct infiniband_member_t {

    /**
     *  @brief Wire this member for the cross-domain Float32 all-reduce-sum of a @p shard_bytes payload:
     *      heap @b [shard | rounds landings | rounds flags] . Runs once, at setup, outside any capture;
     *      the calling thread must hold the member's context.
     *
     *  @param[in] topology The bootstrapped world; coordinates travel over its exchange.
     *  @param[in] member_index This GPU's index inside its multimem_domain; NIC @c mlx5_<global_gpu>
     *      and the partner's same-index GPU pair with it.
     *  @param[in] arena The member's arena; the heap, the link table, and the mapping counters carve
     *      from it (its chunks are GPUDirect-RDMA-capable by construction).
     *  @param[in] key_prefix Exchange-key namespace; keys are @c "<prefix>.<round>" , so distinct
     *      collectives never collide.
     */
    [[nodiscard]] static expected<infiniband_member_t> make_allreduce(mesh_topology_t const &topology,
                                                                      u32_t member_index, gpu_arena_t &arena,
                                                                      std::size_t shard_bytes,
                                                                      sz::string_view key_prefix,
                                                                      f64_t timeout_seconds = 120.0) noexcept {
        heap_layout_t layout;
        layout.payload_bytes = shard_bytes;
        layout.landing_bytes = shard_bytes;
        layout.rounds = rounds_for_(topology);
        return make_(topology, member_index, arena, layout, key_prefix, timeout_seconds);
    }

    /** @brief Wire this member for the cross-domain all-gather of domain-major @p block_bytes blocks:
     *      the payload spans @b [domains·block_bytes] and doubles as the landing — blocks land in
     *      place. Same contract as @ref make_allreduce otherwise. */
    [[nodiscard]] static expected<infiniband_member_t> make_allgather(mesh_topology_t const &topology,
                                                                      u32_t member_index, gpu_arena_t &arena,
                                                                      std::size_t block_bytes,
                                                                      sz::string_view key_prefix,
                                                                      f64_t timeout_seconds = 120.0) noexcept {
        heap_layout_t layout;
        layout.payload_bytes = static_cast<std::size_t>(topology.shape.multimem_domain_count()) * block_bytes;
        layout.landing_bytes = 0;
        layout.rounds = rounds_for_(topology);
        return make_(topology, member_index, arena, layout, key_prefix, timeout_seconds);
    }

  private:
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

        [[nodiscard]] std::size_t landings_offset() const noexcept { return align_(payload_bytes); }
        [[nodiscard]] std::size_t flags_offset() const noexcept {
            return align_(landings_offset() + rounds * landing_bytes);
        }
        [[nodiscard]] std::size_t total_bytes() const noexcept {
            return align_(flags_offset() + rounds * sizeof(u32_t));
        }

      private:
        [[nodiscard]] static std::size_t align_(std::size_t bytes) noexcept { return (bytes + 127ull) & ~127ull; }
    };

    /** @brief The recursive-doubling round count of the topology's domain grid. */
    [[nodiscard]] static u32_t rounds_for_(mesh_topology_t const &topology) noexcept {
        u32_t const domains = topology.shape.multimem_domain_count();
        u32_t rounds = 0;
        while ((1u << rounds) < domains) ++rounds;
        return rounds;
    }

    /** @brief The shared wiring body behind both role factories — see @ref make_allreduce for the
     *      contract. */
    [[nodiscard]] static expected<infiniband_member_t> make_(mesh_topology_t const &topology, u32_t member_index,
                                                             gpu_arena_t &arena, heap_layout_t layout,
                                                             sz::string_view key_prefix,
                                                             f64_t timeout_seconds) noexcept {
        using result_t = expected<infiniband_member_t>;
        mesh_shape_t const &shape = topology.shape;
        u32_t const gpus = shape.gpus_per_multimem_domain;
        u32_t const domains = shape.multimem_domain_count();
        u32_t const domain_index = topology.multimem_domain_index();
        if (layout.rounds == 0 || layout.payload_bytes == 0)
            return result_t::fail(status_t::invalid_input_k, "a member needs rounds and a payload (rounds={} bytes={})",
                                  layout.rounds, layout.payload_bytes);

        infiniband_member_t member;
        member.layout_ = layout;
        member.multimem_domain_index_ = domain_index;
        std::size_t const heap_bytes = layout.total_bytes();
        gpu_arena_alloc<b8_t> byte_alloc {arena};
        gpu_arena_alloc<infiniband_link_t> link_alloc {arena};
        gpu_arena_alloc<u32_t> u32_alloc {arena};
        member.heap_ = b8r1_tensor_arena_t::try_empty({heap_bytes}, byte_alloc);
        member.links_ = infiniband_links_tensor_t::try_empty({layout.rounds}, link_alloc);
        member.epoch_ = u32r1_tensor_arena_t::try_empty({1}, u32_alloc);
        if (!member.heap_ || !member.links_ || !member.epoch_)
            return result_t::fail(status_t::bad_alloc_k, "member heap / table carve failed ({} bytes, arena_used={})",
                                  heap_bytes, arena.used());
        if (cuMemsetD8(reinterpret_cast<CUdeviceptr>(member.heap_.data()), 0, heap_bytes) != CUDA_SUCCESS ||
            cuMemsetD8(reinterpret_cast<CUdeviceptr>(member.epoch_.data()), 0, sizeof(u32_t)) != CUDA_SUCCESS)
            return result_t::fail(status_t::cuda_error_k, "member heap reset failed");
        if (member.endpoints_.try_resize(layout.rounds) != status_t::success_k ||
            member.queues_.try_resize(layout.rounds) != status_t::success_k ||
            member.completions_.try_resize(layout.rounds) != status_t::success_k)
            return result_t::fail(status_t::bad_alloc_k, "member tables failed ({} rounds)", layout.rounds);
        array<infiniband_link_t> host_links {};
        if (host_links.try_resize(layout.rounds) != status_t::success_k)
            return result_t::fail(status_t::bad_alloc_k, "host link staging failed");

        sz::string nic_name;
        if (!format_into(nic_name, "mlx5_{}", topology.first_gpu_index() + member_index))
            return result_t::fail(status_t::bad_alloc_k, "nic name formatting failed");
        b8r1_span_t const heap_span {member.heap_.data(), heap_bytes};
        for (u32_t round = 0; round != layout.rounds; ++round) {
            u32_t const partner_domain = domain_index ^ (1u << round);
            if (partner_domain >= domains)
                return result_t::fail(status_t::shape_unsupported_k,
                                      "non-power-of-two domain counts are unsupported ({} domains)", domains);
            expected<infiniband_endpoint_t> endpoint = infiniband_endpoint_t::make(nic_name.c_str(), 1u, heap_span);
            if (!endpoint) return result_t::fail(std::move(endpoint));
            member.endpoints_[round] = std::move(endpoint.value_);
            sz::string key;
            if (!format_into(key, "{}.{}", key_prefix, round))
                return result_t::fail(status_t::bad_alloc_k, "exchange key formatting failed");
            u32_t const partner_host = shape.host_domain_of(partner_domain * gpus + member_index);
            infiniband_coordinates_t peer;
            if (expected_t exchanged = exchange_infiniband_coordinates(topology, key.view(),
                                                                       member.endpoints_[round].local_coordinates(),
                                                                       partner_host, peer, timeout_seconds);
                !exchanged)
                return result_t::fail(std::move(exchanged));
            if (expected_t connected = member.endpoints_[round].connect(peer); !connected)
                return result_t::fail(std::move(connected));
            expected<infiniband_queue_mapping_t> queue = infiniband_queue_mapping_t::make(
                member.endpoints_[round].queue_pair(), arena);
            if (!queue) return result_t::fail(std::move(queue));
            member.queues_[round] = std::move(queue.value_);
            expected<infiniband_completion_mapping_t> completion = infiniband_completion_mapping_t::make(
                member.endpoints_[round].completion_queue(), arena);
            if (!completion) return result_t::fail(std::move(completion));
            member.completions_[round] = std::move(completion.value_);

            infiniband_link_t link;
            link.remote_multimem_domain_index = partner_domain;
            link.round_index = round;
            link.remote_heap_base = peer.buffer_address;
            link.remote_access_key = peer.remote_key;
            link.local_access_key = member.endpoints_[round].local_key();
            link.queue = member.queues_[round].device_handles();
            link.completion = member.completions_[round].device_handles();
            if (layout.landing_bytes != 0) {
                // Each round its own landing, so a fast partner never overwrites a slower one's
                // still-unfolded payload.
                std::size_t const landing = layout.landings_offset() + round * layout.landing_bytes;
                link.local_landing = b8r1_span_t {member.heap_.data() + landing, layout.landing_bytes};
                link.landing_offset = landing;
            }
            else {
                // The payload doubles as the landing — the all-gather's domain-major buffer.
                link.local_landing = b8r1_span_t {member.heap_.data(), layout.payload_bytes};
                link.landing_offset = 0;
            }
            link.remote_flag_offset = layout.flags_offset() + round * sizeof(u32_t);
            link.arrival_flag = reinterpret_cast<infiniband_epoch_t *>(member.heap_.data() + link.remote_flag_offset);
            host_links[round] = link;
        }
        if (CUresult result = cuMemcpyHtoD(reinterpret_cast<CUdeviceptr>(member.links_.data()),
                                           host_links.view().data(), layout.rounds * sizeof(infiniband_link_t));
            result != CUDA_SUCCESS)
            return result_t::fail(result, "link table upload failed");
        return result_t {std::move(member)};
    }

  public:
    /** @brief The registered payload region — the shard the all-reduce sums in place, or the
     *      domain-major world buffer the all-gather fills. */
    [[nodiscard]] b8r1_span_t payload() noexcept { return {heap_.data(), layout_.payload_bytes}; }
    /** @brief An all-gather member's own domain-major slot inside @ref payload — where this domain's
     *      block must be staged before @ref allgather ; the coordinate stays baked, never re-derived
     *      by consumers. */
    [[nodiscard]] b8r1_span_t own_block() noexcept {
        std::size_t const block_bytes = layout_.payload_bytes >> layout_.rounds;
        return {heap_.data() + static_cast<std::size_t>(multimem_domain_index_) * block_bytes, block_bytes};
    }
    /** @brief The one host write of a step: upload the epoch and advance the host counter by the round
     *      count — enqueue on @p stream before this step's collective. */
    [[nodiscard]] expected_t advance(gpu_stream_t const &stream) noexcept {
        if (CUresult result = cuMemcpyHtoDAsync(reinterpret_cast<CUdeviceptr>(epoch_.data()), &host_epoch_,
                                                sizeof(host_epoch_), stream.get());
            result != CUDA_SUCCESS)
            return expected_t::fail(result, "member epoch upload failed");
        host_epoch_ += layout_.rounds;
        return {};
    }

#if defined(__CUDACC__)
    /**
     *  @brief Device-initiated inter-domain all-reduce-sum of this member's Float32 payload, in place —
     *      the cross-domain gradient / force reduce, enqueued on @p stream as capture-safe graph nodes.
     *
     *  Rounds run in order (round @c r 's accumulate feeds round @c r+1 's post); within a round:
     *  post-and-drain, wait, accumulate. Call @ref advance first, once per step.
     */
    [[nodiscard]] expected_t allreduce_sum(gpu_stream_t const &stream) noexcept {
        if (layout_.landing_bytes == 0)
            return expected_t::fail(status_t::invalid_input_k,
                                    "this member is wired for the all-gather, not the all-reduce");
        std::size_t const shard_scalars = layout_.payload_bytes / sizeof(f32_t);
        f32_t *const shard = reinterpret_cast<f32_t *>(heap_.data());
        unsigned accumulate_grid = static_cast<unsigned>((shard_scalars + kernel_block_size_default_k - 1) /
                                                         kernel_block_size_default_k);
        if (accumulate_grid == 0u) accumulate_grid = 1u;
        if (accumulate_grid > infiniband_accumulate_grid_cap_k) accumulate_grid = infiniband_accumulate_grid_cap_k;
        dim3 const doorbell_grid {1u, 1u, 1u};
        dim3 const doorbell_block {infiniband_doorbell_block_threads_k, 1u, 1u};
        infiniband_links_view_t const links {links_.data(), layout_.rounds};
        infiniband_epoch_view_t const epoch {epoch_.data(), 1};
        for (u32_t round = 0; round != layout_.rounds; ++round) {
            expected_t launched = launch_kernel(reinterpret_cast<void const *>(&infiniband_allreduce_post_kernel_),
                                                doorbell_grid, doorbell_block, stream, links, round,
                                                static_cast<f32_t const *>(shard), shard_scalars, epoch);
            if (launched)
                launched = launch_kernel(reinterpret_cast<void const *>(&infiniband_flag_wait_kernel_), doorbell_grid,
                                         doorbell_block, stream, links, round, epoch);
            if (launched)
                launched = launch_kernel(reinterpret_cast<void const *>(&infiniband_accumulate_f32_kernel_),
                                         dim3 {accumulate_grid, 1u, 1u}, dim3 {kernel_block_size_default_k, 1u, 1u},
                                         stream, links, round, shard, shard_scalars);
            if (!launched) return launched;
        }
        return {};
    }

    /**
     *  @brief Device-initiated inter-domain all-gather of the domain-major blocks inside this member's
     *      payload — how every domain receives every other domain's vertex tower, enqueued on @p stream
     *      as capture-safe graph nodes. This domain's own block must already sit at its domain-major
     *      slot; after @c log2(D) rounds every block is filled. Call @ref advance first, once per step.
     */
    [[nodiscard]] expected_t allgather(gpu_stream_t const &stream) noexcept {
        if (layout_.landing_bytes != 0)
            return expected_t::fail(status_t::invalid_input_k,
                                    "this member is wired for the all-reduce, not the all-gather");
        infiniband_allgather_plan_t plan;
        plan.block_bytes = layout_.payload_bytes >> layout_.rounds;
        plan.this_multimem_domain = multimem_domain_index_;
        char const *payload_bytes = reinterpret_cast<char const *>(heap_.data());
        dim3 const doorbell_grid {1u, 1u, 1u};
        dim3 const doorbell_block {infiniband_doorbell_block_threads_k, 1u, 1u};
        infiniband_links_view_t const links {links_.data(), layout_.rounds};
        infiniband_epoch_view_t const epoch {epoch_.data(), 1};
        for (u32_t round = 0; round != layout_.rounds; ++round) {
            expected_t launched = launch_kernel(reinterpret_cast<void const *>(&infiniband_allgather_post_kernel_),
                                                doorbell_grid, doorbell_block, stream, links, round, plan,
                                                payload_bytes, epoch);
            if (launched)
                launched = launch_kernel(reinterpret_cast<void const *>(&infiniband_flag_wait_kernel_), doorbell_grid,
                                         doorbell_block, stream, links, round, epoch);
            if (!launched) return launched;
        }
        return {};
    }
#endif // defined(__CUDACC__)

  private:
    /** @brief The heap's byte-offset formula — identical on every domain, baked by the role factory. */
    heap_layout_t layout_ {};
    /** @brief This rank's multimem_domain index — the recursive-doubling coordinate, baked at wiring. */
    u32_t multimem_domain_index_ = 0;
    /** @brief The host-side epoch counter @ref advance uploads and bumps once per step. */
    u32_t host_epoch_ = 0;
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
    /** @brief The device epoch scalar; see @ref advance . */
    u32r1_tensor_arena_t epoch_ {};
};

#endif // defined(UFORM_WITH_INFINIBAND)

} // namespace unum::uform::primitives
