/**
 *  @brief InfiniBand transport for the mesh: RC endpoints over GPU memory, and the GDAKI device-initiated
 *      doorbell so captured kernels post RDMA writes with no host in the control path.
 * 
 *  @file cuda/primitives/infiniband_transport.cuh
 *  @author Ash Vardanian
 *  @date July 2, 2026
 *
 *  Three layers, smallest dependency surface first:
 *
 *      1. Wire PODs — @ref rdma_coordinates_t (what two queue pairs exchange to connect),
 *         @ref gdaki_queue_handles_t / @ref gdaki_completion_handles_t (what a kernel needs to post and
 *         drain), and @ref colocated_link_t (how one GPU reaches one remote multimem_domain). These are
 *         ibverbs-free and always compiled.
 *      2. @ref rdma_endpoint_t — one (GPU, NIC) Reliable-Connection queue pair with its memory region
 *         registered over GPU VRAM: dmabuf-first ( @c cuMemGetHandleForAddressRange +
 *         @c ibv_reg_dmabuf_mr ), @c nvidia_peermem @c ibv_reg_mr fallback. Host-posted writes and
 *         completion polling stay as the bring-up / smoke-test path.
 *      3. @ref gdaki_queue_mapping_t / @ref gdaki_completion_mapping_t — the GDAKI (GPUDirect Async
 *         Kernel-Initiated) layer: @c mlx5dv_init_obj exposes the raw send queue, doorbell record, and
 *         BlueFlame register, and the mapping registers them into the GPU's address space so
 *         @ref gdaki_post_write builds one MLX5 work-queue entry and rings the doorbell entirely from
 *         device code — capture-safe graph nodes, no @c ibv_post_send .
 *
 *  Coordinates travel over the mesh exchange ( @ref mesh_topology_t::publish_text / @c await_text ) as
 *  line-oriented text — the same once-per-job filesystem rendezvous that bootstrapped the world; the
 *  transport adds payloads, not machinery.
 *
 *  @section rdma_platform Platform contract
 *
 *  LID-routed same-subnet InfiniBand (GID carried but unused), pkey 0, GID index 0 — the single-fabric
 *  cluster this project targets. The BlueFlame register lives in NIC BAR / MMIO space: mapping it for the
 *  GPU needs @c CU_MEMHOSTREGISTER_IOMEMORY , and a platform whose IOMMU / ACS configuration refuses that
 *  registration cannot run device-initiated GDAKI — @ref gdaki_queue_mapping_t::make names that failure
 *  explicitly. CQE compression must be off (validated at mapping time). Everything here builds only under
 *  @c UFORM_WITH_INFINIBAND ; without it the wire PODs remain and every factory fails loudly naming the gate.
 */
#pragma once

#include <cstdint> // `std::uint8_t`, `std::uint16_t`, `std::uint64_t`
#include <cstring> // `std::memcpy`, `std::strcmp`

#include <charconv> // `std::from_chars` — coordinate payload parsing
#include <chrono>   // `std::chrono::steady_clock` — the completion-poll deadline
#include <thread>   // `std::this_thread::sleep_for` — the completion-poll backoff
#include <utility>  // `std::exchange`

#include <cuda.h> // `cuMemGetHandleForAddressRange`, `cuMemHostRegister`, `cuMemHostGetDevicePointer`

#if defined(UFORM_WITH_INFINIBAND)
#include <unistd.h> // `::close`, `sysconf` — dmabuf descriptors and page-aligned registrations

#include <infiniband/mlx5dv.h> // `mlx5dv_init_obj`, `mlx5_wqe_*`, `mlx5_cqe64` — the direct-verbs layer
#include <infiniband/verbs.h>  // `ibv_context`, `ibv_pd`, `ibv_cq`, `ibv_qp` — the RC bring-up
#endif

#include "types.cuh"      // `expected`, `expected_t`, `status_t`, `format_into`
#include "allocators.cuh" // `gpu_arena_t` carving the device producer / consumer counters
#include "schedule.cuh"   // `b8r1_span_t` — the registered GPU range travels as one typed span
#include "mesh.cuh"       // `mesh_topology_t` — the coordinate exchange rides the mesh rendezvous

namespace unum::uform::primitives {

/** @brief Monotonic never-reset arrival counter the GDAKI collectives spin on; lives in device memory. */
using gdaki_epoch_t = u32_t;

/** @brief The send-doorbell slot in the mlx5 two-entry doorbell-record array (entry 0 is the receive side). */
inline constexpr std::size_t mlx5_send_doorbell_index_k = 1;
/** @brief The set-consumer-index slot in the mlx5 completion-queue doorbell array (entry 1 arms interrupts). */
inline constexpr std::size_t mlx5_completion_consumer_index_k = 0;
/** @brief Owner-bit mask of an mlx5 completion entry's @c op_own byte; hardware flips it per ring wrap. */
inline constexpr u8_t mlx5_completion_owner_mask_k = 1u;
/** @brief Descriptor-segment count (16-byte units) of an RDMA-write work-queue entry: control + remote
 *      address + data = 3. */
inline constexpr u32_t gdaki_write_descriptor_segments_k = 3u;
/** @brief The mlx5 inline-data marker OR'd into a work-queue entry's inline byte-count header — spelled as
 *      a project constant so the wire PODs stay free of @c <infiniband/mlx5dv.h> . */
inline constexpr u32_t gdaki_inline_data_flag_k = 0x80000000u;

/**
 *  @brief What one queue pair publishes so a peer can connect and address its registered GPU buffer — the
 *      wire POD of the RDMA rendezvous, exchanged as text over the mesh's filesystem rendezvous.
 */
struct rdma_coordinates_t {
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
    [[nodiscard]] static bool from_text(sz::string_view payload, rdma_coordinates_t &coordinates) noexcept {
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
[[nodiscard]] inline expected_t exchange_rdma_coordinates(mesh_topology_t const &topology, sz::string_view key,
                                                          rdma_coordinates_t const &local, u32_t peer_host_domain,
                                                          rdma_coordinates_t &peer, f64_t timeout_seconds) noexcept {
    sz::string local_text;
    if (!local.to_text(local_text))
        return expected_t::fail(status_t::bad_alloc_k, "coordinate payload assembly failed (key={})", key);
    if (expected_t published = topology.publish_text(key, local_text.view()); !published) return published;
    sz::string peer_text;
    if (expected_t awaited = topology.await_text(peer_host_domain, key, peer_text, timeout_seconds); !awaited)
        return awaited;
    if (!rdma_coordinates_t::from_text(peer_text.view(), peer))
        return expected_t::fail(status_t::io_error_k, "malformed coordinates from host_domain {} (key={}): \"{}\"",
                                peer_host_domain, key, peer_text);
    return {};
}

/**
 *  @brief The pointer-and-scalar handles a kernel needs to post one work-queue entry and ring the
 *      doorbell — a trivially-copyable POD passed by value; the owning @ref gdaki_queue_mapping_t fills
 *      it. All pointers are in the GPU's address space.
 */
struct gdaki_queue_handles_t {
    /** @brief The send work-queue entry storage (host DMA memory the NIC fetches from). */
    void *send_queue_buffer = nullptr;
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
 *      @ref gdaki_queue_handles_t . The three hardware unknowns ( @c entry_size , @c initial_owner ,
 *      @c compressed ) are captured at mapping time; the valid 64-byte entry lives in the @b high bytes
 *      of a wider slot.
 */
struct gdaki_completion_handles_t {
    /** @brief The completion-entry ring buffer (host DMA memory the NIC writes into). */
    void *ring_buffer = nullptr;
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
 *  outside capture, from an @ref rdma_endpoint_t + the GDAKI mappings + exchanged coordinates.
 */
struct colocated_link_t {
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
    gdaki_queue_handles_t queue {};
    /** @brief The completion parser paired with @ref queue ; one drain per signaled post. */
    gdaki_completion_handles_t completion {};
    /** @brief This rank's landing buffer where the partner's remote write arrives, shaped
     *      @b [landing_bytes] — the consuming collective reinterprets the scalar type it landed. */
    b8r1_span_t local_landing {};
    /** @brief Symmetric-heap byte offset of the partner's landing buffer. */
    std::size_t landing_offset = 0;
    /** @brief Symmetric-heap byte offset of the partner's arrival flag. */
    std::size_t remote_flag_offset = 0;
    /** @brief This rank's monotonic arrival flag (device VRAM) the waiting kernel acquire-spins on. */
    gdaki_epoch_t *arrival_flag = nullptr;
};

#if defined(UFORM_WITH_INFINIBAND)

/** @brief One InfiniBand access-flag set shared by the memory region and the queue pair. */
inline constexpr int rdma_access_flags_k = IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_WRITE | IBV_ACCESS_REMOTE_READ;
/** @brief Completion-queue and work-request depth; the host path posts a handful of writes per exchange. */
inline constexpr int rdma_queue_depth_k = 16;

/** @brief Whether a host-posted write also carries immediate data (consuming a posted receive on the peer). */
enum class rdma_signal_t : u32_t {
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
struct rdma_endpoint_t {

    rdma_endpoint_t() noexcept = default;
    rdma_endpoint_t(rdma_endpoint_t const &) = delete;
    rdma_endpoint_t &operator=(rdma_endpoint_t const &) = delete;

    rdma_endpoint_t(rdma_endpoint_t &&other) noexcept
        : device_context_(std::exchange(other.device_context_, nullptr)),
          protection_domain_(std::exchange(other.protection_domain_, nullptr)),
          completion_queue_(std::exchange(other.completion_queue_, nullptr)),
          queue_pair_(std::exchange(other.queue_pair_, nullptr)),
          memory_region_(std::exchange(other.memory_region_, nullptr)),
          gpu_buffer_(std::exchange(other.gpu_buffer_, b8r1_span_t {})),
          dmabuf_descriptor_(std::exchange(other.dmabuf_descriptor_, -1)), infiniband_port_(other.infiniband_port_),
          path_mtu_(other.path_mtu_), local_coordinates_(other.local_coordinates_),
          peer_coordinates_(other.peer_coordinates_) {}
    rdma_endpoint_t &operator=(rdma_endpoint_t &&other) noexcept {
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
    ~rdma_endpoint_t() noexcept { destroy_(); }

    /**
     *  @brief Open @p nic_name , build the protection domain / completion queue / RC queue pair, and
     *      register @p gpu_buffer as a GPUDirect memory region — everything a peer handshake needs.
     *
     *  @param[in] nic_name The mlx5 device name (e.g. @c "mlx5_0" ) from the ibverbs device list.
     *  @param[in] infiniband_port The port on that adapter; single-port ConnectX parts use 1.
     *  @param[in] gpu_buffer The GPU VRAM range peers will read and write, shaped @b [buffer_bytes] —
     *      typically a window over a @ref gpu_arena_t (its chunks are created GPUDirect-RDMA-capable).
     */
    [[nodiscard]] static expected<rdma_endpoint_t> make(char const *nic_name, u8_t infiniband_port,
                                                        b8r1_span_t gpu_buffer) noexcept {
        if (nic_name == nullptr || gpu_buffer.empty())
            return expected<rdma_endpoint_t>::fail(status_t::invalid_input_k,
                                                   "an endpoint needs a NIC name and a non-empty GPU buffer");
        rdma_endpoint_t endpoint;
        endpoint.infiniband_port_ = infiniband_port;
        endpoint.gpu_buffer_ = gpu_buffer;

        if (expected_t opened = endpoint.open_device_(nic_name); !opened)
            return expected<rdma_endpoint_t>::fail(std::move(opened));

        ibv_port_attr port_attributes {};
        if (ibv_query_port(endpoint.device_context_, infiniband_port, &port_attributes) != 0)
            return expected<rdma_endpoint_t>::fail(status_t::io_error_k, "port {} query failed on \"{}\"",
                                                   infiniband_port, nic_name);
        endpoint.path_mtu_ = static_cast<unsigned>(port_attributes.active_mtu);
        ibv_gid port_gid {};
        if (ibv_query_gid(endpoint.device_context_, infiniband_port, 0, &port_gid) != 0)
            return expected<rdma_endpoint_t>::fail(status_t::io_error_k, "GID query failed on \"{}\" port {}", nic_name,
                                                   infiniband_port);

        endpoint.protection_domain_ = ibv_alloc_pd(endpoint.device_context_);
        if (endpoint.protection_domain_ == nullptr)
            return expected<rdma_endpoint_t>::fail(status_t::io_error_k, "protection-domain allocation failed on {}",
                                                   nic_name);
        endpoint.completion_queue_ = ibv_create_cq(endpoint.device_context_, rdma_queue_depth_k, nullptr, nullptr, 0);
        if (endpoint.completion_queue_ == nullptr)
            return expected<rdma_endpoint_t>::fail(status_t::io_error_k, "completion-queue creation failed on {}",
                                                   nic_name);

        ibv_qp_init_attr queue_pair_config {};
        queue_pair_config.send_cq = endpoint.completion_queue_;
        queue_pair_config.recv_cq = endpoint.completion_queue_;
        queue_pair_config.qp_type = IBV_QPT_RC;
        queue_pair_config.sq_sig_all = 0;
        queue_pair_config.cap.max_send_wr = rdma_queue_depth_k;
        queue_pair_config.cap.max_recv_wr = rdma_queue_depth_k;
        queue_pair_config.cap.max_send_sge = 1;
        queue_pair_config.cap.max_recv_sge = 1;
        endpoint.queue_pair_ = ibv_create_qp(endpoint.protection_domain_, &queue_pair_config);
        if (endpoint.queue_pair_ == nullptr)
            return expected<rdma_endpoint_t>::fail(status_t::io_error_k,
                                                   "Reliable-Connection queue-pair creation failed on {}", nic_name);

        if (expected_t registered = endpoint.register_gpu_memory_(); !registered)
            return expected<rdma_endpoint_t>::fail(std::move(registered));

        rdma_coordinates_t &coordinates = endpoint.local_coordinates_;
        coordinates.local_identifier = port_attributes.lid;
        coordinates.queue_pair_number = endpoint.queue_pair_->qp_num;
        coordinates.packet_sequence_number = 0;
        coordinates.remote_key = endpoint.memory_region_->rkey;
        coordinates.buffer_address = reinterpret_cast<u64_t>(gpu_buffer.data());
        std::memcpy(coordinates.global_identifier, port_gid.raw, sizeof(coordinates.global_identifier));
        return expected<rdma_endpoint_t> {std::move(endpoint)};
    }

    /** @brief What this endpoint publishes for the peer's @ref connect . */
    [[nodiscard]] rdma_coordinates_t const &local_coordinates() const noexcept { return local_coordinates_; }
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
    [[nodiscard]] expected_t connect(rdma_coordinates_t const &peer) noexcept {
        peer_coordinates_ = peer;

        ibv_qp_attr to_initialized {};
        to_initialized.qp_state = IBV_QPS_INIT;
        to_initialized.pkey_index = 0;
        to_initialized.port_num = infiniband_port_;
        to_initialized.qp_access_flags = rdma_access_flags_k;
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
     *      ride @ref gdaki_post_write . */
    [[nodiscard]] expected_t post_write(std::size_t local_offset, std::size_t remote_offset, std::size_t length,
                                        rdma_signal_t signal = rdma_signal_t::plain_signaled_k) noexcept {
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
        work_request.opcode = signal == rdma_signal_t::with_immediate_k ? IBV_WR_RDMA_WRITE_WITH_IMM
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
                                               rdma_access_flags_k);
            if (memory_region_ != nullptr) {
                dmabuf_descriptor_ = dmabuf_descriptor;
                return {};
            }
            ::close(dmabuf_descriptor);
        }
        memory_region_ = ibv_reg_mr(protection_domain_, gpu_buffer_.data(), gpu_buffer_.extent(0), rdma_access_flags_k);
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
    rdma_coordinates_t local_coordinates_ {};
    rdma_coordinates_t peer_coordinates_ {};
};

/** @brief Register the page(s) covering @p host_pointer for GPU access and resolve its device pointer;
 *      @p registration_base receives the page-floor base to later unregister (null when the page was
 *      already registered by someone else and must stay theirs). */
[[nodiscard]] inline expected_t gdaki_map_for_device(void *host_pointer, std::size_t span, bool io_memory,
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
 *  Layers onto a built, connected @ref rdma_endpoint_t 's queue pair (borrowed, never owned) — build it
 *  once after Ready-To-Send, @b outside any capture. The BlueFlame registration
 *  ( @c CU_MEMHOSTREGISTER_IOMEMORY over NIC BAR space) is @b the platform gate for device-initiated
 *  GDAKI; its failure is reported by name.
 */
struct gdaki_queue_mapping_t {

    gdaki_queue_mapping_t() noexcept = default;
    gdaki_queue_mapping_t(gdaki_queue_mapping_t const &) = delete;
    gdaki_queue_mapping_t &operator=(gdaki_queue_mapping_t const &) = delete;
    gdaki_queue_mapping_t(gdaki_queue_mapping_t &&other) noexcept
        : handles_(std::exchange(other.handles_, gdaki_queue_handles_t {})),
          send_queue_registration_(std::exchange(other.send_queue_registration_, nullptr)),
          doorbell_registration_(std::exchange(other.doorbell_registration_, nullptr)),
          blueflame_registration_(std::exchange(other.blueflame_registration_, nullptr)) {}
    gdaki_queue_mapping_t &operator=(gdaki_queue_mapping_t &&other) noexcept {
        if (this != &other) {
            destroy_();
            handles_ = std::exchange(other.handles_, gdaki_queue_handles_t {});
            send_queue_registration_ = std::exchange(other.send_queue_registration_, nullptr);
            doorbell_registration_ = std::exchange(other.doorbell_registration_, nullptr);
            blueflame_registration_ = std::exchange(other.blueflame_registration_, nullptr);
        }
        return *this;
    }
    ~gdaki_queue_mapping_t() noexcept { destroy_(); }

    /** @brief Expose @p queue_pair 's raw mlx5 send queue / doorbell / BlueFlame register and map all
     *      three for the GPU; the producer counter is carved from @p arena (capture-stable address). */
    [[nodiscard]] static expected<gdaki_queue_mapping_t> make(ibv_qp *queue_pair, gpu_arena_t &arena) noexcept {
        if (queue_pair == nullptr)
            return expected<gdaki_queue_mapping_t>::fail(status_t::invalid_input_k, "null queue pair");
        mlx5dv_qp raw_queue {};
        mlx5dv_obj object {};
        object.qp.in = queue_pair;
        object.qp.out = &raw_queue;
        if (mlx5dv_init_obj(&object, MLX5DV_OBJ_QP) != 0)
            return expected<gdaki_queue_mapping_t>::fail(status_t::io_error_k,
                                                         "the mlx5 provider refused to expose the queue pair");
        if (raw_queue.sq.buf == nullptr || raw_queue.dbrec == nullptr || raw_queue.bf.reg == nullptr ||
            raw_queue.sq.wqe_cnt == 0)
            return expected<gdaki_queue_mapping_t>::fail(
                status_t::io_error_k, "the mlx5 provider exposed a null send queue / doorbell / BlueFlame register");

        gdaki_queue_mapping_t mapping;
        mapping.handles_.queue_pair_number = queue_pair->qp_num;
        mapping.handles_.send_queue_entry_count = raw_queue.sq.wqe_cnt;
        mapping.handles_.send_queue_stride = raw_queue.sq.stride;

        std::size_t const send_queue_bytes = static_cast<std::size_t>(raw_queue.sq.wqe_cnt) *
                                             static_cast<std::size_t>(raw_queue.sq.stride);
        if (expected_t mapped = gdaki_map_for_device(raw_queue.sq.buf, send_queue_bytes, false,
                                                     mapping.send_queue_registration_,
                                                     mapping.handles_.send_queue_buffer);
            !mapped)
            return expected<gdaki_queue_mapping_t>::fail(std::move(mapped));
        void *doorbell_device = nullptr;
        if (expected_t mapped = gdaki_map_for_device(raw_queue.dbrec, sizeof(u32_t) * 2, false,
                                                     mapping.doorbell_registration_, doorbell_device);
            !mapped)
            return expected<gdaki_queue_mapping_t>::fail(std::move(mapped));
        mapping.handles_.doorbell_record = static_cast<u32_t *>(doorbell_device);

        std::size_t const blueflame_bytes = raw_queue.bf.size != 0 ? raw_queue.bf.size : sizeof(u64_t);
        if (expected_t mapped = gdaki_map_for_device(raw_queue.bf.reg, blueflame_bytes, true,
                                                     mapping.blueflame_registration_,
                                                     mapping.handles_.blueflame_register);
            !mapped)
            return expected<gdaki_queue_mapping_t>::fail( //
                status_t::io_error_k,
                "the BlueFlame MMIO page would not map into GPU address space (CU_MEMHOSTREGISTER_IOMEMORY) — " //
                "device-initiated GDAKI is blocked on this platform: {}",
                mapped.description_);

        mapping.handles_.producer_index = arena.allocate_object<u32_t>(1);
        if (mapping.handles_.producer_index == nullptr)
            return expected<gdaki_queue_mapping_t>::fail(
                status_t::bad_alloc_k, "producer-index counter carve failed (arena_used={})", arena.used());
        if (CUresult result = cuMemsetD8(reinterpret_cast<CUdeviceptr>(mapping.handles_.producer_index), 0,
                                         sizeof(u32_t));
            result != CUDA_SUCCESS)
            return expected<gdaki_queue_mapping_t>::fail(result, "producer-index counter reset failed");
        return expected<gdaki_queue_mapping_t> {std::move(mapping)};
    }

    /** @brief A by-value snapshot for kernels and @ref gdaki_post_write . */
    [[nodiscard]] gdaki_queue_handles_t device_handles() const noexcept { return handles_; }

  private:
    void destroy_() noexcept {
        for (void **base : {&blueflame_registration_, &doorbell_registration_, &send_queue_registration_})
            if (*base != nullptr) cuMemHostUnregister(std::exchange(*base, nullptr));
        handles_ = gdaki_queue_handles_t {};
    }

    gdaki_queue_handles_t handles_ {};
    void *send_queue_registration_ = nullptr;
    void *doorbell_registration_ = nullptr;
    void *blueflame_registration_ = nullptr;
};

/**
 *  @brief The GPU-reachable completion path of one queue pair: the completion ring and its doorbell
 *      record mapped for the GPU (both ordinary host DMA memory), plus a device consumer-index counter
 *      and the hardware geometry captured at build time.
 */
struct gdaki_completion_mapping_t {

    gdaki_completion_mapping_t() noexcept = default;
    gdaki_completion_mapping_t(gdaki_completion_mapping_t const &) = delete;
    gdaki_completion_mapping_t &operator=(gdaki_completion_mapping_t const &) = delete;
    gdaki_completion_mapping_t(gdaki_completion_mapping_t &&other) noexcept
        : handles_(std::exchange(other.handles_, gdaki_completion_handles_t {})),
          ring_registration_(std::exchange(other.ring_registration_, nullptr)),
          doorbell_registration_(std::exchange(other.doorbell_registration_, nullptr)) {}
    gdaki_completion_mapping_t &operator=(gdaki_completion_mapping_t &&other) noexcept {
        if (this != &other) {
            destroy_();
            handles_ = std::exchange(other.handles_, gdaki_completion_handles_t {});
            ring_registration_ = std::exchange(other.ring_registration_, nullptr);
            doorbell_registration_ = std::exchange(other.doorbell_registration_, nullptr);
        }
        return *this;
    }
    ~gdaki_completion_mapping_t() noexcept { destroy_(); }

    /** @brief Expose @p completion_queue 's raw mlx5 ring + doorbell and map both for the GPU; the
     *      consumer counter is carved from @p arena . The ring must be a power of two and uncompressed. */
    [[nodiscard]] static expected<gdaki_completion_mapping_t> make(ibv_cq *completion_queue,
                                                                   gpu_arena_t &arena) noexcept {
        if (completion_queue == nullptr)
            return expected<gdaki_completion_mapping_t>::fail(status_t::invalid_input_k, "null completion queue");
        mlx5dv_cq raw_completion {};
        mlx5dv_obj object {};
        object.cq.in = completion_queue;
        object.cq.out = &raw_completion;
        if (mlx5dv_init_obj(&object, MLX5DV_OBJ_CQ) != 0)
            return expected<gdaki_completion_mapping_t>::fail(
                status_t::io_error_k, "the mlx5 provider refused to expose the completion queue");
        if (raw_completion.buf == nullptr || raw_completion.dbrec == nullptr || raw_completion.cqe_cnt == 0 ||
            raw_completion.cqe_size == 0)
            return expected<gdaki_completion_mapping_t>::fail(
                status_t::io_error_k, "the mlx5 provider exposed a null completion ring / doorbell");
        if ((raw_completion.cqe_cnt & (raw_completion.cqe_cnt - 1u)) != 0)
            return expected<gdaki_completion_mapping_t>::fail(status_t::shape_unsupported_k,
                                                              "the completion ring must be a power of two (got {})",
                                                              raw_completion.cqe_cnt);

        gdaki_completion_mapping_t mapping;
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
        if (expected_t mapped = gdaki_map_for_device(raw_completion.buf, ring_bytes, false, mapping.ring_registration_,
                                                     mapping.handles_.ring_buffer);
            !mapped)
            return expected<gdaki_completion_mapping_t>::fail(std::move(mapped));
        void *doorbell_device = nullptr;
        if (expected_t mapped = gdaki_map_for_device(raw_completion.dbrec, sizeof(u32_t) * 2, false,
                                                     mapping.doorbell_registration_, doorbell_device);
            !mapped)
            return expected<gdaki_completion_mapping_t>::fail(std::move(mapped));
        mapping.handles_.doorbell_record = static_cast<u32_t *>(doorbell_device);

        mapping.handles_.consumer_index = arena.allocate_object<u32_t>(1);
        if (mapping.handles_.consumer_index == nullptr)
            return expected<gdaki_completion_mapping_t>::fail(
                status_t::bad_alloc_k, "consumer-index counter carve failed (arena_used={})", arena.used());
        if (CUresult result = cuMemsetD8(reinterpret_cast<CUdeviceptr>(mapping.handles_.consumer_index), 0,
                                         sizeof(u32_t));
            result != CUDA_SUCCESS)
            return expected<gdaki_completion_mapping_t>::fail(result, "consumer-index counter reset failed");
        return expected<gdaki_completion_mapping_t> {std::move(mapping)};
    }

    /** @brief A by-value snapshot for kernels and @ref gdaki_drain_one . */
    [[nodiscard]] gdaki_completion_handles_t device_handles() const noexcept { return handles_; }

  private:
    void destroy_() noexcept {
        for (void **base : {&doorbell_registration_, &ring_registration_})
            if (*base != nullptr) cuMemHostUnregister(std::exchange(*base, nullptr));
        handles_ = gdaki_completion_handles_t {};
    }

    gdaki_completion_handles_t handles_ {};
    void *ring_registration_ = nullptr;
    void *doorbell_registration_ = nullptr;
};

#if defined(__CUDACC__)

/** @brief Big-endian-ize a 32-bit work-queue field (the NIC reads every scalar big-endian). */
[[nodiscard]] __device__ inline u32_t gdaki_byte_swap_32(u32_t value) noexcept {
    return __byte_perm(value, 0u, 0x0123u);
}

/** @brief Big-endian-ize a 64-bit field: swap each half, then swap the halves. */
[[nodiscard]] __device__ inline u64_t gdaki_byte_swap_64(u64_t value) noexcept {
    u32_t const low_swapped = gdaki_byte_swap_32(static_cast<u32_t>(value));
    u32_t const high_swapped = gdaki_byte_swap_32(static_cast<u32_t>(value >> 32));
    return (static_cast<u64_t>(low_swapped) << 32) | static_cast<u64_t>(high_swapped);
}

/** @brief System-scope acquire load of a completion entry's @c op_own byte — observing the flipped owner
 *      bit orders the rest of the entry (written by the NIC before it) as visible. */
[[nodiscard]] __device__ inline u8_t gdaki_load_acquire_u8(u8_t const *pointer) noexcept {
    unsigned value;
    asm volatile("ld.acquire.sys.global.u8 %0, [%1];" : "=r"(value) : "l"(pointer));
    return static_cast<u8_t>(value);
}

/** @brief System-scope acquire load of an arrival flag — pairs with the partner's flag write so its prior
 *      payload write (ordered before the flag on the same queue pair) is observed first. */
[[nodiscard]] __device__ inline u32_t gdaki_load_acquire_u32(u32_t const *pointer) noexcept {
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
 *  @ref gdaki_drain_one ; delivery on a Reliable Connection is in-order, so an unsignaled payload write
 *  followed by a signaled inline flag write on the @b same queue pair makes the flag's arrival the proof
 *  the payload landed. Call from @b one thread.
 */
__device__ inline void gdaki_post_write(gdaki_queue_handles_t handles, u64_t local_gpu_address, u32_t local_access_key,
                                        u64_t remote_address, u32_t remote_access_key, u32_t byte_count, bool signaled,
                                        void const *inline_four_bytes) noexcept {
    u32_t const producer_index = atomicAdd(handles.producer_index, 1u);
    u32_t const slot_offset = (producer_index % handles.send_queue_entry_count) * handles.send_queue_stride;
    u8_t *entry_bytes = static_cast<u8_t *>(handles.send_queue_buffer) + slot_offset;

    auto *control = reinterpret_cast<mlx5_wqe_ctrl_seg *>(entry_bytes);
    auto *remote = reinterpret_cast<mlx5_wqe_raddr_seg *>(entry_bytes + sizeof(mlx5_wqe_ctrl_seg));
    u8_t *after_remote = entry_bytes + sizeof(mlx5_wqe_ctrl_seg) + sizeof(mlx5_wqe_raddr_seg);

    // Control segment: opcode + this entry's index in the low 16 bits, then (QPN << 8 | descriptor count).
    u32_t const opmod_index_opcode = gdaki_byte_swap_32(((producer_index & 0xffffu) << 8) |
                                                        static_cast<u32_t>(MLX5_OPCODE_RDMA_WRITE));
    u32_t const qpn_descriptor_size = gdaki_byte_swap_32((handles.queue_pair_number << 8) |
                                                         gdaki_write_descriptor_segments_k);
    control->opmod_idx_opcode = opmod_index_opcode;
    control->qpn_ds = qpn_descriptor_size;
    control->signature = 0;
    control->dci_stream_channel_id = 0;
    control->fm_ce_se = signaled ? static_cast<u8_t>(MLX5_WQE_CTRL_CQ_UPDATE) : u8_t {0};
    control->imm = 0;

    // Remote-address segment: where on the peer's GPU the bytes land, under its remote key.
    remote->raddr = gdaki_byte_swap_64(remote_address);
    remote->rkey = gdaki_byte_swap_32(remote_access_key);
    remote->reserved = 0;

    if (inline_four_bytes != nullptr) {
        // Inline data segment: a big-endian header (inline flag | byte count), then the 4 bytes verbatim.
        auto *inline_header = reinterpret_cast<u32_t *>(after_remote);
        *inline_header = gdaki_byte_swap_32(gdaki_inline_data_flag_k | 4u);
        u8_t const *source = static_cast<u8_t const *>(inline_four_bytes);
        for (u32_t byte_index = 0; byte_index != 4u; ++byte_index) after_remote[4u + byte_index] = source[byte_index];
    }
    else {
        // Local-data segment: the source range inside this endpoint's registered buffer, under our key.
        auto *data = reinterpret_cast<mlx5_wqe_data_seg *>(after_remote);
        data->byte_count = gdaki_byte_swap_32(byte_count);
        data->lkey = gdaki_byte_swap_32(local_access_key);
        data->addr = gdaki_byte_swap_64(local_gpu_address);
    }

    // The entry must be visible in system memory before the NIC observes the advanced doorbell record.
    __threadfence_system();
    handles.doorbell_record[mlx5_send_doorbell_index_k] = gdaki_byte_swap_32((producer_index + 1u) & 0xffffu);
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
__device__ inline void gdaki_drain_one(gdaki_completion_handles_t completion) noexcept {
    u32_t const consumer_index = *completion.consumer_index;
    u32_t const slot = consumer_index & (completion.entry_count - 1u);
    u8_t *slot_bytes = static_cast<u8_t *>(completion.ring_buffer) +
                       static_cast<std::size_t>(slot) * completion.entry_size;
    auto *entry = reinterpret_cast<mlx5_cqe64 *>(slot_bytes + (completion.entry_size - sizeof(mlx5_cqe64)));
    u32_t const expected_owner = (completion.initial_owner ^ ((consumer_index >> completion.entry_log_count) & 1u)) &
                                 1u;

    while (true) {
        u8_t const op_own = gdaki_load_acquire_u8(&entry->op_own);
        if ((static_cast<u32_t>(op_own) & mlx5_completion_owner_mask_k) != expected_owner) continue; // not yet ours
        u32_t const opcode = static_cast<u32_t>(op_own) >> 4;
        if (opcode == static_cast<u32_t>(MLX5_CQE_INVALID)) continue; // owner flipped before the NIC wrote
        if (opcode == static_cast<u32_t>(MLX5_CQE_REQ_ERR)) {
            auto const *error = reinterpret_cast<mlx5_err_cqe const *>(entry);
            std::printf("gdaki_drain_one: send completion error syndrome=0x%x vendor=0x%x cqn=%u\n",
                        static_cast<u32_t>(error->syndrome), static_cast<u32_t>(error->vendor_err_synd),
                        completion.completion_queue_number);
            __trap();
        }
        break;
    }

    *completion.consumer_index = consumer_index + 1u;
    __threadfence_system();
    // Publish the consumer index (24-bit field, big-endian) so the completion ring never fills.
    completion.doorbell_record[mlx5_completion_consumer_index_k] = gdaki_byte_swap_32((consumer_index + 1u) &
                                                                                      0xffffffu);
    __threadfence_system();
}

#endif // defined(__CUDACC__)
#endif // defined(UFORM_WITH_INFINIBAND)

} // namespace unum::uform::primitives
