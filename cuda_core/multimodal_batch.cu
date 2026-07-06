/**
 *  @brief The batch's implementation: the single segment-driven coordinate adapter ( @ref fill_coords )
 *      and the batch mixer's routing bijection and two exchange kernels — compiled once into the `uform`
 *      library. Everything stays on the GPU: peer addresses travel as kernel arguments, and every kernel
 *      re-derives its routes in-register from the keyed bijection, so the mixer owns no memory at all; the
 *      coordinate adapter is equally a pure enqueue.
 *  @file cuda_core/multimodal_batch.cu
 *  @author Ash Vardanian
 *  @date July 3, 2026
 */
#include "cuda_core/multimodal_batch.hpp"

namespace uc = unum::uform::core;
using namespace uc;

/**
 *  @brief Write one segment's shared coordinate @c (t,h,w) : CTA @c s owns segment @c s , reads its
 *      @c (offset,length,modality) , and threads stripe @c [0,length) . The @c (t,h,w) each cell gets is a
 *      pure function of its in-segment index @c k and, for the 2-D / 3-D modalities, the segment's shape
 *      descriptor — text and audio are both 1-D @c (k,0,0) (a raw audio frame walks the same position axis as
 *      a token), image the 2-D @c (0,row,col) , video the 3-D absolute-frame @c (tg·tube_frames,r,c) . PAD
 *      segments ( @c length==0 ) and a zero-width descriptor write nothing, so uncovered positions stay
 *      whatever the caller pre-zeroed.
 *
 *  __grid_size__ @c [segments,1,1] , threads stripe the segment's length; no dynamic shared memory.
 */
static __global__ void fill_coords_kernel_(u32r1_view_t segment_offsets, u32r1_view_t segment_lengths,
                                           modality_view_t segment_modality, image_shape_view_t image_shapes,
                                           video_shape_view_t video_shapes, u32_t tube_frames, i16r2_span_t coords) {
    u32_t const segment = blockIdx.x;
    u32_t const length = segment_lengths[segment];
    if (length == 0u) return;
    u32_t const offset = segment_offsets[segment];
    modality_t const modality = segment_modality[segment];
    for (u32_t k = threadIdx.x; k < length; k += blockDim.x) {
        i16_t coord_t = 0, coord_h = 0, coord_w = 0;
        switch (modality) {
        // Text and audio are both 1-D sequences — a frame walks the same position axis as a token.
        case modality_t::text_k:
        case modality_t::audio_k: coord_t = static_cast<i16_t>(k); break;
        case modality_t::image_k: {
            u32_t const patch_cols = image_shapes[segment, 1];
            if (patch_cols == 0u) continue;
            coord_h = static_cast<i16_t>(k / patch_cols);
            coord_w = static_cast<i16_t>(k % patch_cols);
            break;
        }
        case modality_t::video_k: {
            u32_t const patch_cols = video_shapes[segment, 2];
            u32_t const groups_per_frame = static_cast<u32_t>(video_shapes[segment, 1]) * patch_cols;
            if (groups_per_frame == 0u) continue;
            u32_t const time_group = k / groups_per_frame, planar = k % groups_per_frame;
            coord_t = static_cast<i16_t>(time_group * tube_frames);
            coord_h = static_cast<i16_t>(planar / patch_cols);
            coord_w = static_cast<i16_t>(planar % patch_cols);
            break;
        }
        }
        coords[offset + k, 0] = coord_t;
        coords[offset + k, 1] = coord_h;
        coords[offset + k, 2] = coord_w;
    }
}

expected_t uc::fill_coords(u32r1_span_t segment_offsets, u32r1_span_t segment_lengths, modality_span_t segment_modality,
                           image_shape_view_t image_shapes, video_shape_view_t video_shapes, u32_t tube_frames,
                           i16r2_span_t coords, gpu_stream_t &stream) noexcept {
    if (coords.extent(1) != 3)
        return expected_t::fail(status_t::shape_unsupported_k, "coordinate table must be [{},3]", coords.extent(0));
    std::size_t const segments = segment_lengths.extent(0);
    if (segment_offsets.extent(0) < segments || segment_modality.extent(0) != segments)
        return expected_t::fail(status_t::invalid_input_k,
                                "offsets ({}), lengths ({}), and modality ({}) must span the segment count",
                                segment_offsets.extent(0), segments, segment_modality.extent(0));
    if ((image_shapes.extent(0) != 0 && image_shapes.extent(1) != 2) ||
        (video_shapes.extent(0) != 0 && video_shapes.extent(1) != 3))
        return expected_t::fail(status_t::shape_unsupported_k, "shape descriptors must be image [{},2], video [{},3]",
                                image_shapes.extent(0), video_shapes.extent(0));
    if (segments == 0) return {};
    dim3 const block_shape {kernel_block_size_default_k, 1u, 1u};
    dim3 const grid {static_cast<unsigned>(segments), 1u, 1u};
    return launch_kernel(reinterpret_cast<void const *>(&fill_coords_kernel_), grid, block_shape, stream,
                         u32r1_view_t {segment_offsets.data(), segment_offsets.extent(0)},
                         u32r1_view_t {segment_lengths.data(), segment_lengths.extent(0)},
                         modality_view_t {segment_modality.data(), segment_modality.extent(0)}, image_shapes,
                         video_shapes, tube_frames, coords);
}

/** @brief One member's batch addresses — the gather kernels' read view of a peer. Carries every
 *      per-segment descriptor and every modality's content, so a routed segment's media survives the
 *      cross-GPU shuffle, not just its tokens. A modality absent from the batch leaves its pointer null. */
struct batch_source_t_ {
    token_id_t const *tokens = nullptr;
    u32_t const *segment_offsets = nullptr;
    u32_t const *segment_lengths = nullptr;
    modality_t const *segment_modality = nullptr;
    u16_t const *image_shapes = nullptr;
    u16_t const *video_shapes = nullptr;
    pixel_t const *images = nullptr;
    pixel_t const *videos = nullptr;
    f32_t const *audio_samples = nullptr;
};

/** @brief This member's mixed-batch write addresses — the gather kernel's scatter target. Mirrors the
 *      routed descriptors and content from @ref batch_source_t_ ; a modality absent here leaves its pointer
 *      null and its copy is skipped. */
struct batch_dest_t_ {
    token_id_t *tokens = nullptr;
    modality_t *segment_modality = nullptr;
    u16_t *image_shapes = nullptr;
    u16_t *video_shapes = nullptr;
    pixel_t *images = nullptr;
    pixel_t *videos = nullptr;
    f32_t *audio_samples = nullptr;
};

/** @brief Every member's addresses, passed by value as a kernel argument — @ref max_gpus_per_host_k
 *      bounds the table, so no device-resident copy and no staging exist. */
struct batch_sources_t_ {
    batch_source_t_ entries[max_gpus_per_host_k] {};
    u32_t count = 0;
};

/**
 *  @brief Write the mixed batch's geometry: thread 0 walks the destination segments serially — the
 *      repack cursor is a strict scan — deriving each segment's world source through the shared
 *      @ref epoch_permutation (read gather-wise: a destination looks up its source, equivalent to
 *      scattering by the inverse permutation, and identical on every GPU so routing needs no
 *      exchange), reading its length from the peer's table, and jumping the cursor past
 *      any row boundary the segment would cross. A segment that would overflow the destination
 *      rectangle is dropped loudly (zero length) — the packer's budgets make that a bug, not a case.
 *
 *  __grid_size__ @c [1,1,1] , one thread live; no dynamic shared memory.
 */
static __global__ void batch_geometry_kernel_(batch_sources_t_ sources, u32r1_span_t mixed_offsets,
                                              u32r1_span_t mixed_lengths, u32_t member_index, data_epoch_t data_epoch,
                                              u64_t seed, u32_t segments_per_gpu, u32_t rows_per_gpu,
                                              u32_t columns_per_row) {
    if (threadIdx.x != 0u || blockIdx.x != 0u) return;
    std::size_t const world_segments = static_cast<std::size_t>(segments_per_gpu) * sources.count;
    u32_t const capacity = rows_per_gpu * columns_per_row;
    u32_t cursor = 0;
    for (u32_t segment = 0; segment != segments_per_gpu; ++segment) {
        std::size_t const destination = static_cast<std::size_t>(member_index) * segments_per_gpu + segment;
        std::size_t const source = epoch_permutation(seed, data_epoch, destination, world_segments);
        batch_source_t_ const &peer = sources.entries[source / segments_per_gpu];
        u32_t columns = peer.segment_lengths[source % segments_per_gpu];
        // A segment never crosses its row: jump the cursor to the next row when it would.
        u32_t const room_in_row = columns_per_row - cursor % columns_per_row;
        if (columns > room_in_row) cursor += room_in_row;
        if (cursor + columns > capacity) {
            printf("uform: mixed batch overflows at segment %u (member %u) — segment dropped\n", segment, member_index);
            columns = 0;
        }
        mixed_offsets[segment] = cursor;
        mixed_lengths[segment] = columns;
        cursor += columns;
    }
    mixed_offsets[segments_per_gpu] = cursor;
}

/**
 *  @brief Gather one routed segment: CTA @c s re-derives route @c s 's world source, copies the peer's
 *      per-segment descriptors ( @ref batch_source_t_::segment_modality and the image / video @c *_shapes rows —
 *      thread 0, PAD included, so the mixed batch is fully typed) and then, dispatching on the segment's
 *      modality, copies its content ( @c tokens , or the flat @c [begin,begin+columns) patch/tube/frame run,
 *      scaled by the modality's per-cell scalar count) from the peer's loaded batch into this member's mixed
 *      batch, threads striding the run. The destination geometry is what @ref batch_geometry_kernel_ wrote.
 *
 *  __grid_size__ @c [segments_per_gpu,1,1] , threads stripe the content run; no dynamic shared memory.
 */
static __global__ void batch_gather_kernel_(batch_sources_t_ sources, batch_dest_t_ dest, u32r1_view_t mixed_offsets,
                                            u32r1_view_t mixed_lengths, u32_t member_index, data_epoch_t data_epoch,
                                            u64_t seed, u32_t segments_per_gpu, u32_t image_patch_scalars,
                                            u32_t video_tube_scalars, u32_t audio_frame_scalars) {
    u32_t const segment = blockIdx.x;
    std::size_t const world_segments = static_cast<std::size_t>(segments_per_gpu) * sources.count;
    std::size_t const destination = static_cast<std::size_t>(member_index) * segments_per_gpu + segment;
    std::size_t const source = epoch_permutation(seed, data_epoch, destination, world_segments);
    batch_source_t_ const &peer = sources.entries[source / segments_per_gpu];
    u32_t const source_segment = static_cast<u32_t>(source % segments_per_gpu);

    // Descriptors travel for every routed segment, PAD included, so the mixed batch stays fully typed.
    if (threadIdx.x == 0u) {
        if (dest.segment_modality != nullptr && peer.segment_modality != nullptr)
            dest.segment_modality[segment] = peer.segment_modality[source_segment];
        if (dest.image_shapes != nullptr && peer.image_shapes != nullptr)
            for (u32_t i = 0; i != 2u; ++i)
                dest.image_shapes[segment * 2u + i] = peer.image_shapes[source_segment * 2u + i];
        if (dest.video_shapes != nullptr && peer.video_shapes != nullptr)
            for (u32_t i = 0; i != 3u; ++i)
                dest.video_shapes[segment * 3u + i] = peer.video_shapes[source_segment * 3u + i];
    }

    u32_t const columns = mixed_lengths[segment];
    if (columns == 0u) return;
    std::size_t const source_begin = peer.segment_offsets[source_segment];
    std::size_t const destination_begin = mixed_offsets[segment];
    modality_t const modality = peer.segment_modality != nullptr ? peer.segment_modality[source_segment]
                                                                 : modality_t::text_k;
    switch (modality) {
    case modality_t::text_k:
        for (u32_t column = threadIdx.x; column < columns; column += blockDim.x)
            dest.tokens[destination_begin + column] = peer.tokens[source_begin + column];
        break;
    case modality_t::image_k:
        if (dest.images != nullptr && peer.images != nullptr) {
            std::size_t const run = static_cast<std::size_t>(columns) * image_patch_scalars;
            for (std::size_t i = threadIdx.x; i < run; i += blockDim.x)
                dest.images[destination_begin * image_patch_scalars + i] =
                    peer.images[source_begin * image_patch_scalars + i];
        }
        break;
    case modality_t::video_k:
        if (dest.videos != nullptr && peer.videos != nullptr) {
            std::size_t const run = static_cast<std::size_t>(columns) * video_tube_scalars;
            for (std::size_t i = threadIdx.x; i < run; i += blockDim.x)
                dest.videos[destination_begin * video_tube_scalars + i] =
                    peer.videos[source_begin * video_tube_scalars + i];
        }
        break;
    case modality_t::audio_k:
        if (dest.audio_samples != nullptr && peer.audio_samples != nullptr) {
            std::size_t const run = static_cast<std::size_t>(columns) * audio_frame_scalars;
            for (std::size_t i = threadIdx.x; i < run; i += blockDim.x)
                dest.audio_samples[destination_begin * audio_frame_scalars + i] =
                    peer.audio_samples[source_begin * audio_frame_scalars + i];
        }
        break;
    }
}

/** @brief Build the read view of a member's own loaded batch — its eight content and descriptor pointers, a
 *      direct NVLink peer read for a source GPU on this island. A modality absent from the batch leaves a null
 *      pointer, exactly as the gather kernels expect. */
static batch_source_t_ source_from_loaded_(multimodal_batch_on_gpu_t const &loaded) noexcept {
    batch_source_t_ source;
    source.tokens = loaded.tokens.data();
    source.segment_offsets = loaded.segment_offsets.data();
    source.segment_lengths = loaded.segment_lengths.data();
    source.segment_modality = loaded.segment_modality.data();
    source.image_shapes = loaded.image_shapes.data();
    source.video_shapes = loaded.video_shapes.data();
    source.images = loaded.images.data();
    source.videos = loaded.videos.data();
    source.audio_samples = loaded.audio_samples.data();
    return source;
}

/** @brief Build the read view of a same-index landing — the block a cross-island all-to-all deposited, holding
 *      the other island's same-index GPU's loaded batch. Every pointer is the landing base plus this tensor's
 *      shared wire offset, reinterpreted to the field's type; a dead modality's content pointer stays null (its
 *      offset addresses nothing). The descriptor tensors always exist, so their offsets are always valid. */
static batch_source_t_ source_from_landing_(b8r1_span_t landing, batch_wire_layout_t const &layout,
                                            multimodal_batch_shape_t const &shape) noexcept {
    b8_t const *base = landing.data();
    batch_source_t_ source;
    source.segment_offsets = reinterpret_cast<u32_t const *>(base + layout.offset(wire_segment_offsets_k));
    source.segment_lengths = reinterpret_cast<u32_t const *>(base + layout.offset(wire_segment_lengths_k));
    source.segment_modality = reinterpret_cast<modality_t const *>(base + layout.offset(wire_segment_modality_k));
    source.tokens = reinterpret_cast<token_id_t const *>(base + layout.offset(wire_tokens_k));
    source.image_shapes = reinterpret_cast<u16_t const *>(base + layout.offset(wire_image_shapes_k));
    source.video_shapes = reinterpret_cast<u16_t const *>(base + layout.offset(wire_video_shapes_k));
    source.images = shape.contains(modality_t::image_k)
                        ? reinterpret_cast<pixel_t const *>(base + layout.offset(wire_images_k))
                        : nullptr;
    source.videos = shape.contains(modality_t::video_k)
                        ? reinterpret_cast<pixel_t const *>(base + layout.offset(wire_videos_k))
                        : nullptr;
    source.audio_samples = shape.contains(modality_t::audio_k)
                               ? reinterpret_cast<f32_t const *>(base + layout.offset(wire_audio_samples_k))
                               : nullptr;
    return source;
}

/** @brief One tensor's region in the wire block — its identity and its device source pointer, a named pair so
 *      the pack loop is a plain table walk rather than nine hand-spelled copies. */
struct wire_copy_t_ {
    batch_wire_tensor_t tensor;
    void const *source;
};

/** @brief Serialize a member's loaded batch into its outgoing wire block: one device→device copy per live
 *      tensor into that tensor's shared wire offset, enqueued on @p stream ahead of the exchange. A dead
 *      modality's zero-byte tensor (and any null source) is skipped; the landing's source view nulls the same
 *      modality, so those bytes are never read. */
static expected_t pack_batch_into_wire_(multimodal_batch_on_gpu_t const &loaded, b8r1_span_t destination,
                                        batch_wire_layout_t const &layout, gpu_stream_t &stream) noexcept {
    wire_copy_t_ const copies[wire_tensor_count_k] = {
        {wire_segment_offsets_k, loaded.segment_offsets.data()},
        {wire_segment_lengths_k, loaded.segment_lengths.data()},
        {wire_segment_modality_k, loaded.segment_modality.data()},
        {wire_tokens_k, loaded.tokens.data()},
        {wire_image_shapes_k, loaded.image_shapes.data()},
        {wire_video_shapes_k, loaded.video_shapes.data()},
        {wire_images_k, loaded.images.data()},
        {wire_videos_k, loaded.videos.data()},
        {wire_audio_samples_k, loaded.audio_samples.data()},
    };
    for (wire_copy_t_ const &copy : copies) {
        std::size_t const bytes = layout.byte_count(copy.tensor);
        if (bytes == 0u || copy.source == nullptr) continue;
        // vendor-boundary: device→device pack of one batch tensor into its wire region — the tensors' `data()`
        // pointers are raw CUdeviceptr sources, not views, so the driver copy is spelled by hand here.
        if (CUresult packed = cuMemcpyDtoDAsync(
                reinterpret_cast<CUdeviceptr>(destination.data() + layout.offset(copy.tensor)),
                reinterpret_cast<CUdeviceptr>(copy.source), bytes, stream.get());
            packed != CUDA_SUCCESS)
            return expected_t::fail(packed, "packing batch tensor {} into the wire block failed",
                                    static_cast<u32_t>(copy.tensor));
    }
    return {};
}

expected<multimodal_batch_mixer_t> multimodal_batch_mixer_t::make(
    multimem_domain_t &multimem, colocated_domain_t &colocated, std::span<gpu_arena_t> arenas,
    multimodal_batch_shape_t shape, u64_t seed, std::span<gpu_stream_t> streams, f64_t timeout_seconds) noexcept {
    using result_t = expected<multimodal_batch_mixer_t>;
    std::size_t const members = multimem.member_count();
    if (members == 0 || arenas.size() != members)
        return result_t::fail(status_t::invalid_input_k,
                              "every multimem_domain member needs its arena ({} members, {} arenas)", members,
                              arenas.size());
    if (members > max_gpus_per_host_k)
        return result_t::fail(status_t::shape_unsupported_k, "{} members exceed max_gpus_per_host_k ({})", members,
                              max_gpus_per_host_k);
    if (streams.size() != members)
        return result_t::fail(status_t::invalid_input_k, "every member needs its stream ({} members, {} streams)",
                              members, streams.size());
    u32_t const domains = colocated.multimem_domain_count();
    if (members * domains > max_gpus_per_host_k)
        return result_t::fail(status_t::shape_unsupported_k,
                              "the {}-member × {}-island world exceeds the {}-GPU source table", members, domains,
                              max_gpus_per_host_k);
    // Level-1 precondition: every member's arena readable by every other member — visibility is the
    // arenas' construction-time configuration, validated here, never granted here.
    for (std::size_t member = 0; member != members; ++member)
        for (std::size_t peer = 0; peer != members; ++peer)
            if (!arenas[member].is_visible_to(static_cast<int>(peer)))
                return result_t::fail(status_t::invalid_input_k,
                                      "member {}'s arena is not visible to device {} — " //
                                      "construct it with `visible_to`",
                                      member, peer);
    multimodal_batch_mixer_t mixer;
    mixer.multimem_ = &multimem;
    mixer.seed_ = seed;
    mixer.shape_ = shape;
    mixer.domains_ = domains;
    mixer.home_island_ = colocated.topology().multimem_domain_index();
    mixer.wire_layout_ = batch_wire_layout_t::for_shape(shape);
    // A single-island world never posts — the all-to-all buffers stay empty and the exchange path is skipped.
    // A multi-island world wires one buffer per member, same-index-paired with the other island's GPU; each
    // buffer's block is the whole serialized batch, so a landing holds the remote same-index GPU's batch.
    if (mixer.domains_ > 1u)
        for (std::size_t member = 0; member != members; ++member) {
            switch_to_gpu_t gpu_scope {streams[member]};
            if (!gpu_scope.ok())
                return result_t::fail(status_t::cuda_error_k, "context entry failed wiring member {}", member);
            auto wired = infiniband_all_to_all<b8_t>::make(colocated, static_cast<u32_t>(member), arenas[member],
                                                           mixer.wire_layout_.total_bytes(), streams[member],
                                                           timeout_seconds);
            if (!wired) return result_t::fail(std::move(wired));
            mixer.alltoall_buffers_[member] = std::move(wired.value_);
        }
    return result_t {std::move(mixer)};
}

expected_t multimodal_batch_mixer_t::operator()(std::size_t member_index, data_epoch_t data_epoch,
                                                std::span<multimodal_batch_on_gpu_t const> loaded,
                                                multimodal_batch_on_gpu_t &mixed, gpu_stream_t &stream) noexcept {
    std::size_t const members = multimem_->member_count();
    if (member_index >= members)
        return expected_t::fail(status_t::invalid_input_k, "member {} of {}", member_index, members);
    if (loaded.size() != members)
        return expected_t::fail(status_t::invalid_input_k, "every member needs its loaded batch ({} of {})",
                                loaded.size(), members);
    switch_to_gpu_t gpu_scope {stream};
    if (!gpu_scope.ok()) return expected_t::fail(status_t::cuda_error_k, "context entry failed ({})", member_index);

    // A multi-island world ships this member's whole loaded batch to the other island's same-index GPU before
    // the barrier: pack every live tensor into the outgoing wire block, then enqueue the exchange. After it,
    // this island's member-m landing holds the other island's member-m batch, so a segment whose source GPU
    // sits on the other island is read from that landing rather than peer-read over NVLink.
    if (domains_ > 1u) {
        u32_t const other_island = home_island_ ^ 1u;
        b8r1_span_t const outgoing = alltoall_buffers_[member_index].send_block(other_island);
        if (expected_t packed = pack_batch_into_wire_(loaded[member_index], outgoing, wire_layout_, stream); !packed)
            return packed;
        if (expected_t exchanged = alltoall_buffers_[member_index](stream); !exchanged) return exchanged;
    }

    // The peer table travels as a kernel argument — no device copy, no staging, nothing owned. On one island
    // the entries are the members' loaded batches; across islands the source space doubles to
    // `members·domains_` global GPUs, entry g being loaded[g%members] when g's island is home and the
    // landing of the same-index buffer otherwise.
    auto local_source = [&](std::size_t member, batch_source_t_ &source) noexcept -> expected_t {
        source = source_from_loaded_(loaded[member]);
        if (source.tokens == nullptr || source.segment_offsets == nullptr || source.segment_lengths == nullptr ||
            source.segment_modality == nullptr)
            return expected_t::fail(status_t::invalid_input_k, "member {}'s loaded batch is unallocated", member);
        if (source.tokens == mixed.tokens.data())
            return expected_t::fail(status_t::invalid_input_k,
                                    "loaded and mixed batches must be distinct storage (member={})", member);
        return {};
    };
    batch_sources_t_ sources;
    sources.count = static_cast<u32_t>(members) * domains_;
    for (u32_t global = 0; global != sources.count; ++global) {
        u32_t const local_member = global % static_cast<u32_t>(members);
        u32_t const island = global / static_cast<u32_t>(members);
        if (island == home_island_) {
            if (expected_t built = local_source(local_member, sources.entries[global]); !built) return built;
        }
        else {
            b8r1_span_t const landing = alltoall_buffers_[local_member].recv_block(0u);
            sources.entries[global] = source_from_landing_(landing, wire_layout_, shape_);
        }
    }

    // The step's first barrier orders "everyone loaded" before any peer read; the round is the member's
    // own, self-advanced inside the barrier kernel.
    if (expected_t arrived = multimem_->barrier(member_index, stream); !arrived) return arrived;

    // This member's mixed batch is the scatter target — a modality absent from it leaves a null pointer.
    batch_dest_t_ dest;
    dest.tokens = mixed.tokens.data();
    dest.segment_modality = mixed.segment_modality.data();
    dest.image_shapes = mixed.image_shapes.data();
    dest.video_shapes = mixed.video_shapes.data();
    dest.images = mixed.images.data();
    dest.videos = mixed.videos.data();
    dest.audio_samples = mixed.audio_samples.data();

    // The per-cell scalar fan-in for each modality's flat content run — a patch/tube's element count, or, for
    // 1-D audio, the raw samples in one frame.
    u32_t const image_patch_scalars = patch_scalars(shape_.image.patch_side, shape_.image.channels);
    u32_t const video_tube_scalars = tube_scalars(shape_.video.tube_frames, shape_.video.patch_side,
                                                  shape_.video.channels);
    u32_t const audio_frame_scalars = shape_.audio.chunk_samples;

    u32r1_span_t const mixed_offsets {mixed.segment_offsets.data(), mixed.segment_offsets.extent(0)};
    u32r1_span_t const mixed_lengths {mixed.segment_lengths.data(), mixed.segment_lengths.extent(0)};
    dim3 const one_block {1u, 1u, 1u};
    dim3 const block_shape {kernel_block_size_default_k, 1u, 1u};
    dim3 const segment_grid {shape_.max_segments_per_gpu, 1u, 1u};
    // The routing destination is this member's GLOBAL index — home-island offset plus the local member — so the
    // two islands' same-index GPUs claim disjoint destination ranges in the members·domains world (in a single
    // island home_island_ is 0, so this is the local index unchanged).
    u32_t const global_member = home_island_ * static_cast<u32_t>(members) + static_cast<u32_t>(member_index);
    expected_t staged = launch_kernel(reinterpret_cast<void const *>(&batch_geometry_kernel_), one_block, block_shape,
                                      stream, sources, mixed_offsets, mixed_lengths, global_member, data_epoch, seed_,
                                      shape_.max_segments_per_gpu, shape_.max_rows_per_gpu, shape_.max_columns_per_row);
    if (staged)
        staged = launch_kernel(reinterpret_cast<void const *>(&batch_gather_kernel_), segment_grid, block_shape, stream,
                               sources, dest, u32r1_view_t {mixed_offsets.data(), mixed_offsets.extent(0)},
                               u32r1_view_t {mixed_lengths.data(), mixed_lengths.extent(0)}, global_member, data_epoch,
                               seed_, shape_.max_segments_per_gpu, image_patch_scalars, video_tube_scalars,
                               audio_frame_scalars);
    return staged;
}

expected_t multimodal_batch_mixer_t::operator()(data_epoch_t data_epoch,
                                                std::span<multimodal_batch_on_gpu_t const> loaded,
                                                std::span<multimodal_batch_on_gpu_t> mixed,
                                                std::span<gpu_stream_t> streams) noexcept {
    std::size_t const members = multimem_->member_count();
    if (streams.size() != members || mixed.size() != members)
        return expected_t::fail(status_t::invalid_input_k, "every member needs its stream and mixed batch ({} members)",
                                members);
    for (std::size_t member_index = 0; member_index != members; ++member_index)
        if (expected_t staged = (*this)(member_index, data_epoch, loaded, mixed[member_index], streams[member_index]);
            !staged)
            return staged;
    return {};
}
