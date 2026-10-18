/**
 *  @brief Device kernels and heavy setup/collective bodies of the InfiniBand tier declared in @ref
 *      infiniband_collectives.cuh — the translation unit compiled into the `uform_core` static library
 *      beside @ref multimem_collectives.cu and @ref optimizer.cu .
 *  @file src/core/infiniband_collectives.cu
 *  @author Ash Vardanian
 *  @date July 5, 2026
 */
#include <cstring> // `std::memcpy`, `std::strcmp`

#include <cuda.h> // `cuMemGetHandleForAddressRange`, `cuMemsetD8`, `cuMemcpyHtoD`

#if defined(UFORM_WITH_INFINIBAND)
#include <unistd.h> // `::close` — dmabuf descriptors

#include <infiniband/mlx5dv.h> // `mlx5dv_init_obj`, `mlx5_wqe_*`, `mlx5_cqe64` — the direct-verbs layer
#include <infiniband/verbs.h>  // `ibv_context`, `ibv_pd`, `ibv_cq`, `ibv_qp` — the RC bring-up
#endif

#include "core/infiniband_collectives.cuh" // the declarations these bodies define

using namespace unum::uform::core;

#if defined(UFORM_WITH_INFINIBAND)

/** Lane width of the single-block doorbell kernels; only lane 0 posts and spins. */
inline constexpr unsigned infiniband_doorbell_block_threads_k = 32u;

/**
 *  @brief Post-and-drain step of one all-reduce round: lane 0 posts this domain's shard, then the arrival
 *      flag, to the round's partner, and drains the one signaled completion.
 *
 *  The payload write is unsignaled; only the 4-byte inline flag write is signaled — exactly one
 *  completion per round is produced and drained, so the small send ring never overflows across replays.
 */
static __global__ void infiniband_allreduce_post_kernel_(infiniband_links_view_t links, u32_t round, f32_t const *shard,
                                                         std::size_t shard_scalars,
                                                         infiniband_round_view_t flag_round) {
    if (threadIdx.x != 0) return;
    infiniband_link_t const link = links[round];
    u32_t const step_parity = (flag_round[0] / static_cast<u32_t>(links.extent(0))) & 1u;
    u32_t const flag_value = flag_round[0] + round + 1u;
    infiniband_post_write(link.queue, shard, link.local_access_key,
                          link.remote_heap_base + link.landing_offset + step_parity * link.landing_parity_stride,
                          link.remote_access_key, static_cast<u32_t>(shard_scalars * sizeof(f32_t)),
                          infiniband_signaling_t::unsignaled_k, nullptr);
    infiniband_post_write(link.queue, nullptr, link.local_access_key,
                          link.remote_heap_base + link.remote_flag_offset + step_parity * sizeof(u32_t),
                          link.remote_access_key, sizeof(u32_t), infiniband_signaling_t::signaled_k, &flag_value);
    infiniband_drain_one(link.completion);
}

/** Flag-wait step of one round: lane 0 acquire-spins until the inbound flag reaches the absolute
 *  target (monotonic @c >= — the flag never resets), then a system fence orders the partner's landed
 *  payload before the next kernel, propagated to the block by the closing @c __syncthreads . */
static __global__ void infiniband_flag_wait_kernel_(infiniband_links_view_t links, u32_t round,
                                                    infiniband_round_view_t flag_round) {
    if (threadIdx.x == 0) {
        infiniband_link_t const link = links[round];
        u32_t const step_parity = (flag_round[0] / static_cast<u32_t>(links.extent(0))) & 1u;
        u32_t const target = flag_round[0] + round + 1u;
        while (infiniband_load_acquire_u32(link.arrival_flag + step_parity) < target) {
            // Spin until the partner's flag write for this round lands; its payload already did.
        }
        __threadfence_system();
    }
    __syncthreads();
}

/** Accumulate step of one all-reduce round: fold the partner's landed shard into this domain's,
 *  one explicit round-to-nearest-even add per element — no fused-multiply-add contraction, so the
 *  SPMD-static fold order is bit-identical on every domain. The heap is symmetric with the shard at its
 *  offset 0, so the landing sits at its own offset past the shard. */
static __global__ void infiniband_accumulate_f32_kernel_(infiniband_links_view_t links, u32_t round, f32_t *shard,
                                                         std::size_t shard_scalars,
                                                         infiniband_round_view_t flag_round) {
    infiniband_link_t const link = links[round];
    u32_t const step_parity = (flag_round[0] / static_cast<u32_t>(links.extent(0))) & 1u;
    f32_t const *landing = shard + (link.landing_offset + step_parity * link.landing_parity_stride) / sizeof(f32_t);
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
 *  symmetric buffers make identical byte offsets address identical logical cells, so the partner receives
 *  exactly the half it lacks at the right domain-major position.
 */
static __global__ void infiniband_allgather_post_kernel_(infiniband_links_view_t links, u32_t round,
                                                         infiniband_allgather_plan_t plan, b8_t const *gathered_bytes,
                                                         infiniband_round_view_t flag_round) {
    if (threadIdx.x != 0) return;
    infiniband_link_t const link = links[round];
    // The recursive-doubling bit math keys on the link's true round; the loop index drives only the
    // table entry and the monotonic flag window. The gathered payload sits at heap offset 0 on every
    // domain — the symmetric contract — so identical byte offsets address identical logical cells.
    u32_t const bit_round = link.round_index;
    u32_t const group_base = plan.this_multimem_domain & ~((1u << bit_round) - 1u);
    std::size_t const send_byte_offset = static_cast<std::size_t>(group_base) * plan.block_bytes;
    std::size_t const send_bytes = (static_cast<std::size_t>(1u) << bit_round) * plan.block_bytes;
    u32_t const flag_value = flag_round[0] + round + 1u;

    u32_t const step_parity = (flag_round[0] / static_cast<u32_t>(links.extent(0))) & 1u;
    infiniband_post_write(link.queue, gathered_bytes + send_byte_offset, link.local_access_key,
                          link.remote_heap_base + send_byte_offset, link.remote_access_key,
                          static_cast<u32_t>(send_bytes), infiniband_signaling_t::unsignaled_k, nullptr);
    infiniband_post_write(link.queue, nullptr, link.local_access_key,
                          link.remote_heap_base + link.remote_flag_offset + step_parity * sizeof(u32_t),
                          link.remote_access_key, sizeof(u32_t), infiniband_signaling_t::signaled_k, &flag_value);
    infiniband_drain_one(link.completion);
}

/**
 *  @brief Post one all-to-all round: ship this domain's send block destined for the round's partner to that
 *      partner's landing, then a completion flag.
 *
 *  The partner is baked in @c links[round] — its @c remote_multimem_domain_index selects which send block to
 *  ship — and the symmetric heap puts each side's round landing at the same @c landing_offset , so the write
 *  needs no partner-specific arithmetic.
 *
 *  __grid_size__ @c [1,1,1] , one warp; no dynamic shared memory.
 */
static __global__ void infiniband_alltoall_post_kernel_(infiniband_links_view_t links, u32_t round,
                                                        infiniband_alltoall_plan_t plan, b8_t const *send_bytes,
                                                        infiniband_round_view_t flag_round) {
    if (threadIdx.x != 0) return;
    infiniband_link_t const link = links[round];
    std::size_t const send_byte_offset = static_cast<std::size_t>(link.remote_multimem_domain_index) * plan.block_bytes;
    u32_t const step_parity = (flag_round[0] / static_cast<u32_t>(links.extent(0))) & 1u;
    u32_t const flag_value = flag_round[0] + round + 1u;

    infiniband_post_write(link.queue, send_bytes + send_byte_offset, link.local_access_key,
                          link.remote_heap_base + link.landing_offset + step_parity * link.landing_parity_stride,
                          link.remote_access_key, static_cast<u32_t>(plan.block_bytes),
                          infiniband_signaling_t::unsignaled_k, nullptr);
    infiniband_post_write(link.queue, nullptr, link.local_access_key,
                          link.remote_heap_base + link.remote_flag_offset + step_parity * sizeof(u32_t),
                          link.remote_access_key, sizeof(u32_t), infiniband_signaling_t::signaled_k, &flag_value);
    infiniband_drain_one(link.completion);
}

expected<infiniband_nic_t> infiniband_nic_t::make(u32_t local_gpu_index, u8_t infiniband_port) noexcept {
    infiniband_nic_t nic;
    nic.port_ = infiniband_port;
    // The NIC's page registrations (`cuMemHostRegister`) are scoped to its GPU's primary context, so
    // the NIC holds one reference for its lifetime — otherwise a transient wiring stream releasing the
    // last reference resets the context and silently destroys every registration the doorbell kernels
    // still dereference.
    CUdevice paired_gpu = 0;
    if (CUresult resolved = cuDeviceGet(&paired_gpu, static_cast<int>(local_gpu_index)); resolved != CUDA_SUCCESS)
        return cuda_failure<infiniband_nic_t>(resolved, "paired GPU {} resolution failed", local_gpu_index);
    if (CUresult retained = cuDevicePrimaryCtxRetain(&nic.primary_context_, paired_gpu); retained != CUDA_SUCCESS)
        return cuda_failure<infiniband_nic_t>(retained, "primary-context retain failed (gpu={})", local_gpu_index);
    nic.paired_gpu_ = paired_gpu;
    sz::string_t nic_name;
    if (!format_into(nic_name, "mlx5_{}", local_gpu_index))
        return expected<infiniband_nic_t>::fail(status_t::bad_alloc_k, "nic name formatting failed");

    int device_count = 0;
    ibv_device **device_list = ibv_get_device_list(&device_count);
    if (device_list == nullptr)
        return expected<infiniband_nic_t>::fail(status_t::io_error_k,
                                                "no RDMA devices visible (is the RDMA stack loaded?)");
    for (int device_index = 0; device_index != device_count; ++device_index) {
        char const *name = ibv_get_device_name(device_list[device_index]);
        if (name != nullptr && std::strcmp(name, nic_name.c_str()) == 0) {
            nic.device_context_ = ibv_open_device(device_list[device_index]);
            break;
        }
    }
    ibv_free_device_list(device_list);
    if (nic.device_context_ == nullptr)
        return expected<infiniband_nic_t>::fail(status_t::io_error_k, "RDMA device \"{}\" not found or failed to open",
                                                nic_name);

    ibv_port_attr port_attributes {};
    if (ibv_query_port(nic.device_context_, infiniband_port, &port_attributes) != 0)
        return expected<infiniband_nic_t>::fail(status_t::io_error_k, "port {} query failed on \"{}\"", infiniband_port,
                                                nic_name);
    nic.local_identifier_ = port_attributes.lid;
    nic.path_mtu_ = static_cast<unsigned>(port_attributes.active_mtu);
    ibv_gid port_gid {};
    if (ibv_query_gid(nic.device_context_, infiniband_port, 0, &port_gid) != 0)
        return expected<infiniband_nic_t>::fail(status_t::io_error_k, "GID query failed on \"{}\" port {}", nic_name,
                                                infiniband_port);
    std::memcpy(nic.global_identifier_, port_gid.raw, sizeof(nic.global_identifier_));

    nic.protection_domain_ = ibv_alloc_pd(nic.device_context_);
    if (nic.protection_domain_ == nullptr)
        return expected<infiniband_nic_t>::fail(status_t::io_error_k, "protection-domain allocation failed on {}",
                                                nic_name);
    return expected<infiniband_nic_t> {std::move(nic)};
}

expected<infiniband_endpoint_t> infiniband_endpoint_t::make(infiniband_nic_t &nic, b8r1_span_t gpu_buffer) noexcept {
    if (nic.context() == nullptr || gpu_buffer.empty())
        return expected<infiniband_endpoint_t>::fail(status_t::invalid_input_k,
                                                     "an endpoint needs an opened NIC and a non-empty buffer");
    infiniband_endpoint_t endpoint;
    endpoint.infiniband_port_ = nic.port();
    endpoint.path_mtu_ = nic.path_mtu();
    endpoint.gpu_buffer_ = gpu_buffer;

    endpoint.completion_queue_ = ibv_create_cq(nic.context(), infiniband_queue_depth_k, nullptr, nullptr, 0);
    if (endpoint.completion_queue_ == nullptr)
        return expected<infiniband_endpoint_t>::fail(status_t::io_error_k, "completion-queue creation failed");

    ibv_qp_init_attr queue_pair_config {};
    queue_pair_config.send_cq = endpoint.completion_queue_;
    queue_pair_config.recv_cq = endpoint.completion_queue_;
    queue_pair_config.qp_type = IBV_QPT_RC;
    queue_pair_config.sq_sig_all = 0;
    queue_pair_config.cap.max_send_wr = infiniband_queue_depth_k;
    queue_pair_config.cap.max_recv_wr = infiniband_queue_depth_k;
    queue_pair_config.cap.max_send_sge = 1;
    queue_pair_config.cap.max_recv_sge = 1;
    endpoint.queue_pair_ = ibv_create_qp(nic.protection_domain(), &queue_pair_config);
    if (endpoint.queue_pair_ == nullptr)
        return expected<infiniband_endpoint_t>::fail(status_t::io_error_k,
                                                     "Reliable-Connection queue-pair creation failed");

    if (expected_t registered = endpoint.register_gpu_memory_(nic.protection_domain()); !registered)
        return expected<infiniband_endpoint_t>::fail(std::move(registered));

    infiniband_coordinates_t &coordinates = endpoint.local_coordinates_;
    coordinates.local_identifier = nic.local_identifier();
    coordinates.queue_pair_number = endpoint.queue_pair_->qp_num;
    coordinates.packet_sequence_number = 0;
    coordinates.remote_key = endpoint.memory_region_->rkey;
    coordinates.buffer_address = device_address(gpu_buffer.data());
    std::memcpy(coordinates.global_identifier, nic.global_identifier(), sizeof(coordinates.global_identifier));
    return expected<infiniband_endpoint_t> {std::move(endpoint)};
}

expected_t infiniband_endpoint_t::connect(infiniband_coordinates_t const &peer) noexcept {
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
    int const ready_to_send_mask = IBV_QP_STATE | IBV_QP_SQ_PSN | IBV_QP_TIMEOUT | IBV_QP_RETRY_CNT | IBV_QP_RNR_RETRY |
                                   IBV_QP_MAX_QP_RD_ATOMIC;
    if (ibv_modify_qp(queue_pair_, &to_ready_to_send, ready_to_send_mask) != 0)
        return expected_t::fail(status_t::io_error_k, "queue-pair transition to RTS failed");
    return {};
}

expected_t infiniband_endpoint_t::register_gpu_memory_(ibv_pd *protection_domain) noexcept {
    int dmabuf_descriptor = -1;
    CUdeviceptr const buffer_address = device_address(gpu_buffer_.data());
    CUresult const exported = cuMemGetHandleForAddressRange(&dmabuf_descriptor, buffer_address, gpu_buffer_.extent(0),
                                                            CU_MEM_RANGE_HANDLE_TYPE_DMA_BUF_FD, 0);
    if (exported == CUDA_SUCCESS && dmabuf_descriptor >= 0) {
        memory_region_ = ibv_reg_dmabuf_mr(protection_domain, 0, gpu_buffer_.extent(0), buffer_address,
                                           dmabuf_descriptor, infiniband_access_flags_k);
        if (memory_region_ != nullptr) {
            dmabuf_descriptor_ = dmabuf_descriptor;
            return {};
        }
        ::close(dmabuf_descriptor);
    }
    errno = 0;
    memory_region_ = ibv_reg_mr(protection_domain, gpu_buffer_.data(), gpu_buffer_.extent(0),
                                infiniband_access_flags_k);
    if (memory_region_ == nullptr)
        return expected_t::fail(status_t::io_error_k,
                                "GPU memory-region registration failed: dmabuf export {} (fd={}), " //
                                "ibv_reg_mr errno={} ({} bytes at 0x{:x})",
                                exported == CUDA_SUCCESS ? "ok" : "unsupported", dmabuf_descriptor, errno,
                                gpu_buffer_.extent(0), buffer_address);
    return {};
}

expected<infiniband_queue_mapping_t> infiniband_queue_mapping_t::make(ibv_qp *queue_pair, gpu_arena_t &arena,
                                                                      host_page_registry_t &registry) noexcept {
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

    mapping.registry_ = &registry;
    std::size_t const send_queue_bytes = static_cast<std::size_t>(raw_queue.sq.wqe_cnt) *
                                         static_cast<std::size_t>(raw_queue.sq.stride);
    CUdeviceptr send_queue_device = 0;
    if (expected_t mapped = registry.acquire(raw_queue.sq.buf, send_queue_bytes, host_memory_t::dma_k,
                                             send_queue_device);
        !mapped)
        return expected<infiniband_queue_mapping_t>::fail(std::move(mapped));
    mapping.send_queue_range_ = {raw_queue.sq.buf, send_queue_bytes};
    mapping.handles_.send_queue = {device_pointer<std::byte>(send_queue_device), send_queue_bytes};

    CUdeviceptr doorbell_device = 0;
    if (expected_t mapped = registry.acquire(raw_queue.dbrec, sizeof(u32_t) * 2, host_memory_t::dma_k, doorbell_device);
        !mapped)
        return expected<infiniband_queue_mapping_t>::fail(std::move(mapped));
    mapping.doorbell_range_ = {raw_queue.dbrec, sizeof(u32_t) * 2};
    mapping.handles_.doorbell_record = device_pointer<u32_t>(doorbell_device);

    std::size_t const blueflame_bytes = raw_queue.bf.size != 0 ? raw_queue.bf.size : sizeof(u64_t);
    CUdeviceptr blueflame_device = 0;
    if (expected_t mapped = registry.acquire(raw_queue.bf.reg, blueflame_bytes, host_memory_t::mmio_k,
                                             blueflame_device);
        !mapped)
        return expected<infiniband_queue_mapping_t>::fail( //
            status_t::io_error_k,
            "the BlueFlame MMIO page would not map into GPU address space (CU_MEMHOSTREGISTER_IOMEMORY) — " //
            "device-initiated GDAKI is blocked on this platform: {}",
            mapped.description_);
    mapping.blueflame_range_ = {raw_queue.bf.reg, blueflame_bytes};
    mapping.handles_.blueflame_register = device_pointer<void>(blueflame_device);

    mapping.handles_.producer_index = arena.allocate_object<u32_t>(1);
    if (mapping.handles_.producer_index == nullptr)
        return expected<infiniband_queue_mapping_t>::fail(
            status_t::bad_alloc_k, "producer-index counter allocation failed (arena_used={})", arena.used());
    if (CUresult result = cuMemsetD8(device_address(mapping.handles_.producer_index), 0, sizeof(u32_t));
        result != CUDA_SUCCESS)
        return cuda_failure<infiniband_queue_mapping_t>(result, "producer-index counter reset failed");
    return expected<infiniband_queue_mapping_t> {std::move(mapping)};
}

expected<infiniband_completion_mapping_t> infiniband_completion_mapping_t::make(
    ibv_cq *completion_queue, gpu_arena_t &arena, host_page_registry_t &registry) noexcept {
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
        return expected<infiniband_completion_mapping_t>::fail(status_t::shape_unsupported_k,
                                                               "the completion ring must be a power of two (got {})",
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

    mapping.registry_ = &registry;
    std::size_t const ring_bytes = static_cast<std::size_t>(raw_completion.cqe_cnt) *
                                   static_cast<std::size_t>(raw_completion.cqe_size);
    CUdeviceptr ring_device = 0;
    if (expected_t mapped = registry.acquire(raw_completion.buf, ring_bytes, host_memory_t::dma_k, ring_device);
        !mapped)
        return expected<infiniband_completion_mapping_t>::fail(std::move(mapped));
    mapping.ring_range_ = {raw_completion.buf, ring_bytes};
    mapping.handles_.ring = {device_pointer<std::byte>(ring_device), ring_bytes};

    CUdeviceptr doorbell_device = 0;
    if (expected_t mapped = registry.acquire(raw_completion.dbrec, sizeof(u32_t) * 2, host_memory_t::dma_k,
                                             doorbell_device);
        !mapped)
        return expected<infiniband_completion_mapping_t>::fail(std::move(mapped));
    mapping.doorbell_range_ = {raw_completion.dbrec, sizeof(u32_t) * 2};
    mapping.handles_.doorbell_record = device_pointer<u32_t>(doorbell_device);

    mapping.handles_.consumer_index = arena.allocate_object<u32_t>(1);
    if (mapping.handles_.consumer_index == nullptr)
        return expected<infiniband_completion_mapping_t>::fail(
            status_t::bad_alloc_k, "consumer-index counter allocation failed (arena_used={})", arena.used());
    if (CUresult result = cuMemsetD8(device_address(mapping.handles_.consumer_index), 0, sizeof(u32_t));
        result != CUDA_SUCCESS)
        return cuda_failure<infiniband_completion_mapping_t>(result, "consumer-index counter reset failed");
    return expected<infiniband_completion_mapping_t> {std::move(mapping)};
}

expected<infiniband_reach_t> infiniband_reach_t::make_(mesh_topology_t const &topology, u32_t member_index,
                                                       infiniband_nic_t &nic, gpu_arena_t &arena, gpu_stream_t &stream,
                                                       heap_layout_t layout, sz::string_view_t key_prefix,
                                                       f64_t timeout_seconds) noexcept {
    using result_t = expected<infiniband_reach_t>;
    mesh_shape_t const &shape = topology.shape;
    u32_t const gpus = shape.gpus_per_multimem_domain;
    u32_t const domains = shape.multimem_domain_count();
    u32_t const domain_index = topology.multimem_domain_index();
    if (layout.rounds == 0 || layout.payload_bytes == 0)
        return result_t::fail(status_t::invalid_input_k, "a member needs rounds and a payload (rounds={} bytes={})",
                              layout.rounds, layout.payload_bytes);

    infiniband_reach_t member;
    member.layout_ = layout;
    member.multimem_domain_index_ = domain_index;
    std::size_t const heap_bytes = layout.total_bytes();
    gpu_registrable_alloc<b8_t> heap_alloc {arena};
    gpu_arena_alloc<infiniband_link_t> link_alloc {arena};
    gpu_arena_alloc<u32_t> u32_alloc {arena};
    member.heap_ = b8r1_tensor_registrable_t::uninitialized({heap_bytes}, heap_alloc).value;
    member.links_ = infiniband_links_tensor_t::uninitialized({layout.rounds}, link_alloc).value;
    member.round_scalar_ = u32r1_tensor_arena_t::uninitialized({1}, u32_alloc).value;
    if (!member.heap_ || !member.links_ || !member.round_scalar_)
        return result_t::fail(status_t::bad_alloc_k, "member heap / table allocation failed ({} bytes, arena_used={})",
                              heap_bytes, arena.used());
    if (cuMemsetD8(device_address(member.heap_.data()), 0, heap_bytes) != CUDA_SUCCESS ||
        cuMemsetD8(device_address(member.round_scalar_.data()), 0, sizeof(u32_t)) != CUDA_SUCCESS)
        return result_t::fail(status_t::cuda_error_k, "member heap reset failed");
    expected<barrier_round_t> counter = barrier_round_t::make(u32r1_span_t {member.round_scalar_.data(), 1},
                                                              layout.rounds, stream);
    if (!counter) return result_t::fail(std::move(counter));
    member.round_ = std::move(counter.value_);
    if (member.endpoints_.resize(layout.rounds) != status_t::success_k ||
        member.queues_.resize(layout.rounds) != status_t::success_k ||
        member.completions_.resize(layout.rounds) != status_t::success_k)
        return result_t::fail(status_t::bad_alloc_k, "member tables failed ({} rounds)", layout.rounds);
    array<infiniband_link_t> host_links {};
    if (host_links.resize(layout.rounds) != status_t::success_k)
        return result_t::fail(status_t::bad_alloc_k, "host link staging failed");

    b8r1_span_t const heap_span {member.heap_.data(), heap_bytes};
    for (u32_t round = 0; round != layout.rounds; ++round) {
        u32_t const partner_domain = domain_index ^ (1u << round);
        if (partner_domain >= domains)
            return result_t::fail(status_t::shape_unsupported_k,
                                  "non-power-of-two domain counts are unsupported ({} domains)", domains);
        expected<infiniband_endpoint_t> endpoint = infiniband_endpoint_t::make(nic, heap_span);
        if (!endpoint) return result_t::fail(std::move(endpoint));
        member.endpoints_[round] = std::move(endpoint.value_);
        sz::string_t key;
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
        // The connect above arms only THIS side; the first post must find the partner's queue pair in
        // Ready-to-Receive too, or the responder NIC silently drops until the ~0.5s retry budget dies
        // (transport-retry-exceeded on the wire — loopback's lockstep pacing never exposes it). One more
        // tiny rendezvous proves both ends armed before either member returns wired.
        sz::string_t wired_key;
        if (!format_into(wired_key, "{}.{}.wired", key_prefix, round))
            return result_t::fail(status_t::bad_alloc_k, "wired key formatting failed");
        if (expected_t published = topology.publish_text(wired_key.view(), "1\n"); !published)
            return result_t::fail(std::move(published));
        sz::string_t wired_ack;
        if (expected_t armed = topology.await_text(partner_host, wired_key.view(), wired_ack, timeout_seconds); !armed)
            return result_t::fail(std::move(armed));
        expected<infiniband_queue_mapping_t> queue = infiniband_queue_mapping_t::make(
            member.endpoints_[round].queue_pair(), arena, nic.page_registry());
        if (!queue) return result_t::fail(std::move(queue));
        member.queues_[round] = std::move(queue.value_);
        expected<infiniband_completion_mapping_t> completion = infiniband_completion_mapping_t::make(
            member.endpoints_[round].completion_queue(), arena, nic.page_registry());
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
            // Each round its own even/odd landing pair, so a fast partner's next-step write never lands
            // on a buffer the slower side is still folding — the flag proves arrival, not consumption.
            std::size_t const landing = layout.landings_offset() + round * 2u * layout.landing_bytes;
            link.landing_offset = landing;
            link.landing_parity_stride = static_cast<u32_t>(layout.landing_bytes);
        }
        else {
            // The payload doubles as the landing — the all-gather's domain-major buffer, parity-free.
            link.landing_offset = 0;
            link.landing_parity_stride = 0;
        }
        link.remote_flag_offset = layout.flags_offset() + round * 2u * sizeof(u32_t);
        link.arrival_flag = typed_span<infiniband_round_t>(member.heap_.span()).data() +
                            link.remote_flag_offset / sizeof(infiniband_round_t);
        host_links[round] = link;
    }
    if (CUresult result = cuMemcpyHtoD(device_address(member.links_.data()), host_links.view().data(),
                                       layout.rounds * sizeof(infiniband_link_t));
        result != CUDA_SUCCESS)
        return cuda_failure<infiniband_reach_t>(result, "link table upload failed");
    return result_t {std::move(member)};
}

expected_t infiniband_reach_t::allreduce_sum(gpu_stream_t &stream) noexcept {
    if (layout_.rounds == 0) return {};
    std::size_t const shard_scalars = layout_.payload_bytes / sizeof(f32_t);
    f32_t *const shard = typed_span<f32_t>(heap_.span()).data();
    unsigned const accumulate_grid = grid_size_for(shard_scalars);
    dim3 const doorbell_grid {1u, 1u, 1u};
    dim3 const doorbell_block {infiniband_doorbell_block_threads_k, 1u, 1u};
    infiniband_links_view_t const links {links_.data(), layout_.rounds};
    infiniband_round_view_t const round_view {round_scalar_.data(), 1};
    for (u32_t round = 0; round != layout_.rounds; ++round) {
        expected_t launched = launch_kernel(&infiniband_allreduce_post_kernel_, doorbell_grid, doorbell_block, stream,
                                            links, round, shard, shard_scalars, round_view);
        if (launched)
            launched = launch_kernel(&infiniband_flag_wait_kernel_, doorbell_grid, doorbell_block, stream, links, round,
                                     round_view);
        if (launched)
            launched = launch_kernel(&infiniband_accumulate_f32_kernel_, dim3 {accumulate_grid, 1u, 1u},
                                     dim3 {kernel_block_size_default_k, 1u, 1u}, stream, links, round, shard,
                                     shard_scalars, round_view);
        if (!launched) return launched;
    }
    return {};
}

expected_t infiniband_reach_t::allgather(gpu_stream_t &stream) noexcept {
    if (layout_.rounds == 0) return {};
    infiniband_allgather_plan_t plan;
    plan.block_bytes = layout_.payload_bytes >> layout_.rounds;
    plan.this_multimem_domain = multimem_domain_index_;
    b8_t const *payload_bytes = heap_.data();
    dim3 const doorbell_grid {1u, 1u, 1u};
    dim3 const doorbell_block {infiniband_doorbell_block_threads_k, 1u, 1u};
    infiniband_links_view_t const links {links_.data(), layout_.rounds};
    infiniband_round_view_t const round_view {round_scalar_.data(), 1};
    for (u32_t round = 0; round != layout_.rounds; ++round) {
        expected_t launched = launch_kernel(&infiniband_allgather_post_kernel_, doorbell_grid, doorbell_block, stream,
                                            links, round, plan, payload_bytes, round_view);
        if (launched)
            launched = launch_kernel(&infiniband_flag_wait_kernel_, doorbell_grid, doorbell_block, stream, links, round,
                                     round_view);
        if (!launched) return launched;
    }
    return {};
}

expected_t infiniband_reach_t::alltoall(gpu_stream_t &stream) noexcept {
    if (layout_.rounds == 0) return {};
    infiniband_alltoall_plan_t plan;
    plan.block_bytes = layout_.landing_bytes;
    b8_t const *send_bytes = heap_.data();
    dim3 const doorbell_grid {1u, 1u, 1u};
    dim3 const doorbell_block {infiniband_doorbell_block_threads_k, 1u, 1u};
    infiniband_links_view_t const links {links_.data(), layout_.rounds};
    infiniband_round_view_t const round_view {round_scalar_.data(), 1};
    for (u32_t round = 0; round != layout_.rounds; ++round) {
        expected_t launched = launch_kernel(&infiniband_alltoall_post_kernel_, doorbell_grid, doorbell_block, stream,
                                            links, round, plan, send_bytes, round_view);
        if (launched)
            launched = launch_kernel(&infiniband_flag_wait_kernel_, doorbell_grid, doorbell_block, stream, links, round,
                                     round_view);
        if (!launched) return launched;
    }
    return {};
}

expected<colocated_domain_t> colocated_domain_t::make(mesh_topology_t const &topology) noexcept {
    using result_t = expected<colocated_domain_t>;
    colocated_domain_t domain;
    domain.topology_ = &topology;
    // A single-domain world never posts — no adapters to open, no InfiniBand hardware required.
    if (topology.shape.multimem_domain_count() == 1) return result_t {std::move(domain)};
    u32_t const gpus = topology.shape.gpus_per_host_domain;
    if (domain.nics_.resize(gpus) != status_t::success_k)
        return result_t::fail(status_t::bad_alloc_k, "NIC table allocation failed ({} GPUs)", gpus);
    for (u32_t local = 0; local != gpus; ++local) {
        expected<infiniband_nic_t> nic = infiniband_nic_t::make(local); // local ordinal → mlx5_<local>
        if (!nic) return result_t::fail(std::move(nic));
        domain.nics_[local] = std::move(nic.value_);
    }
    return result_t {std::move(domain)};
}

expected<infiniband_reach_t> colocated_domain_t::reach_(u32_t member_index, gpu_arena_t &arena, gpu_stream_t &stream,
                                                        infiniband_reach_t::heap_layout_t layout,
                                                        f64_t timeout_seconds) noexcept {
    // The single-domain identity: no partners exist, every collective is a no-op — but the rendezvous id
    // still advances, so every rank's key sequence stays aligned whatever the world's shape.
    u32_t const collective_id = next_collective_id_++;
    if (multimem_domain_count() == 1) return expected<infiniband_reach_t> {infiniband_reach_t {}};
    sz::string_t key;
    if (!format_into(key, "collective.{}", collective_id))
        return expected<infiniband_reach_t>::fail(status_t::bad_alloc_k, "rendezvous key formatting failed");
    layout.rounds = ceil_log2(topology_->shape.multimem_domain_count());
    u32_t const member_offset = topology_->shape.index_in_multimem_domain(topology_->first_gpu_index());
    if (member_index < member_offset || member_index - member_offset >= nics_.size()) {
        infiniband_reach_t unwired;
        unwired.layout_ = layout;
        return expected<infiniband_reach_t> {std::move(unwired)};
    }
    return infiniband_reach_t::make_(*topology_, member_index, nics_[member_index - member_offset], arena, stream,
                                     layout, key.view(), timeout_seconds);
}

#endif // defined(UFORM_WITH_INFINIBAND)
