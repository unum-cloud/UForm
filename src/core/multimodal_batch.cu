/**
 *  @brief The batch's implementation: the single segment-driven coordinate adapter ( @ref fill_coords )
 *      and the batch mixer's routing bijection and two exchange kernels — compiled once into the
 *      `uform_core` library.
 *  @file src/core/multimodal_batch.cu
 *  @author Ash Vardanian
 *  @date July 3, 2026
 *
 *  Everything stays on the GPU: peer addresses travel as kernel arguments, and every kernel re-derives its
 *  routes in-register from the keyed bijection, so the mixer owns no memory at all; the coordinate adapter
 *  is equally a pure enqueue.
 */
#include "core/multimodal_batch.cuh"

namespace uc = unum::uform::core;
using namespace uc;

/**
 *  @brief Write one segment's shared coordinate @c (t,h,w) : CTA @c s owns segment @c s , reads its
 *      @c (offset,length,modality) , and threads stripe @c [0,length) .
 *
 *  The @c (t,h,w) each cell gets is a pure function of its in-segment index @c k and, for the 2-D / 3-D
 *  modalities, the segment's shape descriptor — text and audio are both 1-D @c (k,0,0) (a raw audio frame
 *  walks the same position axis as a token), image the 2-D @c (0,row,col) , video the 3-D absolute-frame
 *  @c (tg·tube_frames·frame_stride,r,c) . PAD segments ( @c length==0 ) and a zero-width descriptor write nothing,
 *  so uncovered positions stay whatever the caller pre-zeroed.
 *
 *  __grid_size__ @c [segments,1,1] , threads stripe the segment's length; no dynamic shared memory.
 */
static __global__ void fill_coords_kernel_(u32r1_view_t segment_offsets, u32r1_view_t segment_lengths,
                                           modality_view_t segment_modality, image_shape_view_t image_shapes,
                                           video_shape_view_t video_shapes, u32_t tube_frames, u32_t frame_stride,
                                           i16r2_span_t coords) {
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
            coord_t = static_cast<i16_t>(time_group * tube_frames * frame_stride);
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
                           u32_t frame_stride, i16r2_span_t coords, gpu_stream_t &stream) noexcept {
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
    return launch_kernel(&fill_coords_kernel_, grid, block_shape, stream,
                         u32r1_view_t {segment_offsets.data(), segment_offsets.extent(0)},
                         u32r1_view_t {segment_lengths.data(), segment_lengths.extent(0)},
                         modality_view_t {segment_modality.data(), segment_modality.extent(0)}, image_shapes,
                         video_shapes, tube_frames, frame_stride == 0u ? 1u : frame_stride, coords);
}

/** One member's batch addresses — the gather kernels' read view of a peer. Carries every
 *  per-segment descriptor and every modality's content, so a routed segment's media survives the
 *  cross-GPU shuffle, not just its tokens. A modality absent from the batch leaves its pointer null. */
struct batch_source_t_ {
    token_id_t const *tokens = nullptr;
    u32_t const *segment_offsets = nullptr;
    u32_t const *segment_lengths = nullptr;
    modality_t const *segment_modality = nullptr;
    u32_t const *segment_bag = nullptr;
    u16_t const *image_shapes = nullptr;
    u16_t const *video_shapes = nullptr;
    pixel_t const *images = nullptr;
    pixel_t const *videos = nullptr;
    f32_t const *audio_samples = nullptr;
};

/** This member's mixed-batch write addresses — the gather kernel's scatter target. Mirrors the
 *  routed descriptors and content from @ref batch_source_t_ ; a modality absent here leaves its pointer
 *  null and its copy is skipped. */
struct batch_dest_t_ {
    token_id_t *tokens = nullptr;
    modality_t *segment_modality = nullptr;
    u16_t *image_shapes = nullptr;
    u16_t *video_shapes = nullptr;
    pixel_t *images = nullptr;
    pixel_t *videos = nullptr;
    f32_t *audio_samples = nullptr;
};

/** The sentinel the mixer's routing scratch carries for a mixed PAD segment that has no world
 *  source — a dropped bag's segments and the unclaimed tail both read as this. */
static constexpr u32_t mixed_source_absent_k = ~u32_t {0};

/** The cursor a run of @p columns starts at, jumped to the next row when it would otherwise cross
 *  a row boundary — a segment never spans two rows. */
[[nodiscard]] static __device__ u32_t row_aligned_cursor_(u32_t cursor, u32_t columns, u32_t columns_per_row) noexcept {
    u32_t const room_in_row = columns_per_row - cursor % columns_per_row;
    return columns > room_in_row ? cursor + room_in_row : cursor;
}

/** Every member's addresses, passed by value as a kernel argument — @ref max_gpus_per_host_k
 *  bounds the table, so no device-resident copy and no staging exist. */
struct batch_sources_t_ {
    batch_source_t_ entries[max_gpus_per_host_k] {};
    u32_t count = 0;

    [[nodiscard]] __device__ batch_source_t_ const &operator[](u32_t gpu) const noexcept { return entries[gpu]; }
};

/**
 *  @brief Write the mixed batch's geometry: thread 0 walks the destinations serially, as the repack
 *      cursor is a strict scan.
 *
 *  It derives each destination's world source through the shared @ref epoch_permutation
 *  (read gather-wise: a destination looks up its source, equivalent to scattering by the inverse
 *  permutation, and identical on every GPU so routing needs no exchange), records every mixed
 *  segment's world source into @p mixed_sources for the gather pass, and jumps the cursor past any
 *  row boundary a segment would cross. The routing granularity is the segment when
 *  @p max_bags_per_gpu is zero and the @b bag otherwise — the world axis becomes @c max_bags·world ,
 *  a source bag's live segments (the contiguous run tagged with its index) land together in order,
 *  and @p mixed_bags renumbers them to the destination-local bag. A bag that would overflow the
 *  destination rectangle or the segment table is dropped loudly and @b whole — never split — so a
 *  surviving bag is always intact; the packer's budgets make any drop a bug, not a case.
 *
 *  __grid_size__ @c [1,1,1] , one thread live; no dynamic shared memory.
 */
template <typename sources_type_>
static __global__ void batch_geometry_kernel_(sources_type_ sources, u32r1_span_t mixed_offsets,
                                              u32r1_span_t mixed_lengths, u32r1_span_t mixed_sources,
                                              u32r1_span_t mixed_bags, u32_t member_index, data_epoch_t data_epoch,
                                              u64_t seed, u32_t segments_per_gpu, u32_t rows_per_gpu,
                                              u32_t columns_per_row, u32_t max_bags_per_gpu) {
    if (threadIdx.x != 0u || blockIdx.x != 0u) return;
    u32_t const capacity = rows_per_gpu * columns_per_row;
    u32_t cursor = 0;

    if (max_bags_per_gpu == 0u) {
        // Per-segment routing — the masked pipeline's degenerate case: one segment per bag, the world
        // axis is the segment axis, and every destination claims exactly one source.
        std::size_t const world_segments = static_cast<std::size_t>(segments_per_gpu) * sources.count;
        for (u32_t segment = 0; segment != segments_per_gpu; ++segment) {
            std::size_t const destination = static_cast<std::size_t>(member_index) * segments_per_gpu + segment;
            std::size_t const source = epoch_permutation(seed, data_epoch, destination, world_segments);
            batch_source_t_ const &peer = sources[static_cast<u32_t>(source / segments_per_gpu)];
            u32_t columns = peer.segment_lengths[source % segments_per_gpu];
            cursor = row_aligned_cursor_(cursor, columns, columns_per_row);
            if (cursor + columns > capacity) {
                printf("uform: mixed batch overflows at segment %u (member %u) — segment dropped\n", segment,
                       member_index);
                columns = 0;
            }
            mixed_offsets[segment] = cursor;
            mixed_lengths[segment] = columns;
            mixed_sources[segment] = static_cast<u32_t>(source);
            cursor += columns;
        }
        mixed_offsets[segments_per_gpu] = cursor;
        return;
    }

    // Bag-cohesive routing: the destination walks its bag budget, each drawing one world bag; the source
    // bag's live segments are the contiguous run tagged with its index in the peer's bag column, so the
    // walk finds the run's start once and consumes it to its boundary — a tag that breaks contiguity
    // simply ends the run instead of being gathered from afar.
    std::size_t const world_bags = static_cast<std::size_t>(max_bags_per_gpu) * sources.count;
    u32_t destination_segment = 0;
    for (u32_t bag = 0; bag != max_bags_per_gpu; ++bag) {
        std::size_t const destination_bag = static_cast<std::size_t>(member_index) * max_bags_per_gpu + bag;
        std::size_t const source_bag_world = epoch_permutation(seed, data_epoch, destination_bag, world_bags);
        batch_source_t_ const &peer = sources[static_cast<u32_t>(source_bag_world / max_bags_per_gpu)];
        u32_t const source_bag = static_cast<u32_t>(source_bag_world % max_bags_per_gpu);
        u32_t const source_gpu_base = static_cast<u32_t>(source_bag_world / max_bags_per_gpu) * segments_per_gpu;

        u32_t run_begin = 0;
        while (run_begin != segments_per_gpu &&
               (peer.segment_bag[run_begin] != source_bag || peer.segment_lengths[run_begin] == 0u))
            ++run_begin;
        u32_t run_end = run_begin;
        while (run_end != segments_per_gpu && peer.segment_bag[run_end] == source_bag &&
               peer.segment_lengths[run_end] != 0u)
            ++run_end;

        // Trial placement first, then commit — a bag that cannot land whole lands not at all.
        u32_t trial_cursor = cursor;
        bool fits = destination_segment + (run_end - run_begin) <= segments_per_gpu;
        for (u32_t source_segment = run_begin; source_segment != run_end && fits; ++source_segment) {
            u32_t const columns = peer.segment_lengths[source_segment];
            trial_cursor = row_aligned_cursor_(trial_cursor, columns, columns_per_row);
            fits = trial_cursor + columns <= capacity;
            trial_cursor += columns;
        }
        if (!fits) {
            printf("uform: bag %u cannot land whole on member %u (rows or segment budget) — whole bag dropped\n", bag,
                   member_index);
            continue;
        }
        for (u32_t source_segment = run_begin; source_segment != run_end; ++source_segment, ++destination_segment) {
            u32_t const columns = peer.segment_lengths[source_segment];
            cursor = row_aligned_cursor_(cursor, columns, columns_per_row);
            mixed_offsets[destination_segment] = cursor;
            mixed_lengths[destination_segment] = columns;
            mixed_sources[destination_segment] = source_gpu_base + source_segment;
            mixed_bags[destination_segment] = bag;
            cursor += columns;
        }
    }
    // The unclaimed tail is PAD: zero length, no source, and the one-past-the-last-bag `end` sentinel.
    for (u32_t segment = destination_segment; segment != segments_per_gpu; ++segment) {
        mixed_offsets[segment] = cursor;
        mixed_lengths[segment] = 0u;
        mixed_sources[segment] = mixed_source_absent_k;
        mixed_bags[segment] = max_bags_per_gpu;
    }
    mixed_offsets[segments_per_gpu] = cursor;
}

/**
 *  @brief Gather one routed segment: CTA @c s reads route @c s 's world source from the geometry pass's
 *      @p mixed_sources table (with variable-degree bags a destination's source is not derivable per
 *      segment).
 *
 *  It copies the peer's per-segment descriptors ( @ref batch_source_t_::segment_modality and
 *  the image / video @c *_shapes rows — thread 0, so the mixed batch is fully typed) and then,
 *  dispatching on the segment's modality, copies its content ( @c tokens , or the flat
 *  @c [begin,begin+columns) patch/tube/frame run, scaled by the modality's per-cell scalar count)
 *  from the peer's loaded batch into this member's mixed batch, threads striding the run. A sourceless
 *  PAD (a dropped bag or the unclaimed tail) is typed @c text_k and carries nothing. The destination
 *  geometry is what @ref batch_geometry_kernel_ wrote.
 *
 *  __grid_size__ @c [segments_per_gpu,1,1] , threads stripe the content run; no dynamic shared memory.
 */
template <typename sources_type_>
static __global__ void batch_gather_kernel_(sources_type_ sources, batch_dest_t_ dest, u32r1_view_t mixed_offsets,
                                            u32r1_view_t mixed_lengths, u32r1_view_t mixed_sources,
                                            u32_t segments_per_gpu, u32_t image_patch_scalars, u32_t video_tube_scalars,
                                            u32_t audio_frame_scalars) {
    u32_t const segment = blockIdx.x;
    u32_t const source = mixed_sources[segment];
    if (source == mixed_source_absent_k) {
        if (threadIdx.x == 0u && dest.segment_modality != nullptr) dest.segment_modality[segment] = modality_t::text_k;
        return;
    }
    batch_source_t_ const &peer = sources[source / segments_per_gpu];
    u32_t const source_segment = source % segments_per_gpu;

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

/** The ten batch tensors, in their fixed block-serialization order — the index into a
 *  @ref batch_block_layout_t_ . A fixed family, so it is an enum, not a run of constants; every GPU
 *  shares one shape, so the order (and thus every landing's byte layout) is identical across
 *  multimem_domains. */
enum batch_block_tensor_t_ : u32_t {
    block_segment_offsets_k = 0,
    block_segment_lengths_k = 1,
    block_segment_modality_k = 2,
    block_segment_bag_k = 3,
    block_tokens_k = 4,
    block_image_shapes_k = 5,
    block_video_shapes_k = 6,
    block_images_k = 7,
    block_videos_k = 8,
    block_audio_samples_k = 9,
};

/** How many tensors one exchanged block serializes — the extent of @ref batch_block_tensor_t_ . */
static constexpr u32_t block_tensor_count_k = block_audio_samples_k + 1u;

/**
 *  @brief The serialization map of one @ref multimodal_batch_on_gpu_t inside an exchanged block: a
 *      cache-line-aligned byte offset and a byte count per tensor, packed back to back in
 *      @ref batch_block_tensor_t_ order.
 *
 *  A dead modality (its content tensor unallocated) still gets an offset but a zero byte count, so it
 *  contributes nothing and its offset is never dereferenced. @ref total_bytes is the whole block — the
 *  @c block_scalars an @c infiniband_all_gather<b8_t> is wired with.
 */
struct batch_block_layout_t_ {

    /** Each tensor's cache-line-aligned byte offset from the block base. */
    std::size_t offsets[block_tensor_count_k] {};

    /** Each tensor's serialized byte count; zero for a dead modality's content tensor. */
    std::size_t byte_counts[block_tensor_count_k] {};

    /** The whole block's cache-line-rounded size. */
    std::size_t block_bytes = 0;

    /** This tensor's byte offset from the block base. */
    [[nodiscard]] constexpr std::size_t offset(batch_block_tensor_t_ tensor) const noexcept { return offsets[tensor]; }

    /** This tensor's serialized byte count — zero when its modality is out of the composition. */
    [[nodiscard]] constexpr std::size_t byte_count(batch_block_tensor_t_ tensor) const noexcept {
        return byte_counts[tensor];
    }

    /** The whole block's size — every landing and send block spans exactly this. */
    [[nodiscard]] constexpr std::size_t total_bytes() const noexcept { return block_bytes; }
};

/** Compute the layout @p shape implies — pure shape math, so every GPU on every multimem_domain
 *  derives the identical map with no exchanged descriptor, and the mixer re-derives it on demand
 *  instead of storing it. */
[[nodiscard]] static constexpr batch_block_layout_t_ batch_block_layout_(
    multimodal_batch_shape_t const &shape) noexcept {
    batch_block_layout_t_ layout;
    std::size_t const segments = shape.max_segments_per_gpu;
    std::size_t const cells = static_cast<std::size_t>(shape.max_rows_per_gpu) * shape.max_columns_per_row;
    layout.byte_counts[block_segment_offsets_k] = (segments + 1u) * sizeof(u32_t);
    layout.byte_counts[block_segment_lengths_k] = segments * sizeof(u32_t);
    layout.byte_counts[block_segment_modality_k] = segments * sizeof(modality_t);
    // The bag column rides the block even when bags are off (the layout is fixed shape math) — those
    // bytes are dead on the masked path and small: one u32 per segment.
    layout.byte_counts[block_segment_bag_k] = segments * sizeof(u32_t);
    layout.byte_counts[block_tokens_k] = cells * sizeof(token_id_t);
    layout.byte_counts[block_image_shapes_k] = segments * 2u * sizeof(u16_t);
    layout.byte_counts[block_video_shapes_k] = segments * 3u * sizeof(u16_t);
    layout.byte_counts[block_images_k] = shape.contains(modality_t::image_k)
                                             ? cells * patch_scalars(shape.image.patch_side, shape.image.channels) *
                                                   sizeof(pixel_t)
                                             : 0u;
    layout.byte_counts[block_videos_k] =
        shape.contains(modality_t::video_k)
            ? cells * tube_scalars(shape.video.tube_frames, shape.video.patch_side, shape.video.channels) *
                  sizeof(pixel_t)
            : 0u;
    layout.byte_counts[block_audio_samples_k] = shape.contains(modality_t::audio_k)
                                                    ? cells * shape.audio.chunk_samples * sizeof(f32_t)
                                                    : 0u;
    std::size_t running = 0;
    for (u32_t tensor = 0; tensor != block_tensor_count_k; ++tensor) {
        layout.offsets[tensor] = round_up_to_multiple(running, cache_line_bytes_k);
        running = layout.offsets[tensor] + layout.byte_counts[tensor];
    }
    layout.block_bytes = round_up_to_multiple(running, cache_line_bytes_k);
    return layout;
}

/** Build the read view of a member's own loaded batch — its eight content and descriptor pointers, a
 *  direct NVLink peer read for a source GPU on this multimem_domain. A modality absent from the batch
 *  leaves a null pointer, exactly as the gather kernels expect. */
static batch_source_t_ source_from_loaded_(multimodal_batch_on_gpu_t const &loaded) noexcept {
    batch_source_t_ source;
    source.tokens = loaded.tokens.data();
    source.segment_offsets = loaded.segment_offsets.data();
    source.segment_lengths = loaded.segment_lengths.data();
    source.segment_modality = loaded.segment_modality.data();
    source.segment_bag = loaded.segment_bag.data();
    source.image_shapes = loaded.image_shapes.data();
    source.video_shapes = loaded.video_shapes.data();
    source.images = loaded.images.data();
    source.videos = loaded.videos.data();
    source.audio_samples = loaded.audio_samples.data();
    return source;
}

/** Every world GPU's packed batch in the multimem_domain's symmetric windows: world GPU @c domain·members+member
 *  is block @c domain of that member's window. Resolved arithmetically in-kernel, so no table bounds the world;
 *  a dead modality's zero-byte region reads as null, exactly as the gather kernels expect. */
struct batch_windows_t_ {
    b8_t const *first_window = nullptr;
    std::size_t member_stride = 0;
    batch_block_layout_t_ layout {};
    u32_t members = 0;
    u32_t count = 0;

    template <typename value_type_>
    [[nodiscard]] __device__ value_type_ const *region_(b8_t const *block,
                                                        batch_block_tensor_t_ tensor) const noexcept {
        if (layout.byte_counts[tensor] == 0u) return nullptr;
        return static_cast<value_type_ const *>(static_cast<void const *>(block + layout.offsets[tensor]));
    }

    [[nodiscard]] __device__ batch_source_t_ operator[](u32_t gpu) const noexcept {
        b8_t const *block = first_window + (gpu % members) * member_stride + (gpu / members) * layout.block_bytes;
        batch_source_t_ source;
        source.segment_offsets = region_<u32_t>(block, block_segment_offsets_k);
        source.segment_lengths = region_<u32_t>(block, block_segment_lengths_k);
        source.segment_modality = region_<modality_t>(block, block_segment_modality_k);
        source.segment_bag = region_<u32_t>(block, block_segment_bag_k);
        source.tokens = region_<token_id_t>(block, block_tokens_k);
        source.image_shapes = region_<u16_t>(block, block_image_shapes_k);
        source.video_shapes = region_<u16_t>(block, block_video_shapes_k);
        source.images = region_<pixel_t>(block, block_images_k);
        source.videos = region_<pixel_t>(block, block_videos_k);
        source.audio_samples = region_<f32_t>(block, block_audio_samples_k);
        return source;
    }
};

/** One tensor's region in the exchanged block — its identity and its device source pointer, a named
 *  pair so the pack loop is a plain table walk rather than ten hand-spelled copies. */
struct block_copy_t_ {
    batch_block_tensor_t_ tensor;
    void const *source;
};

/** Serialize a member's loaded batch into its outgoing block: one device → device copy per live
 *  tensor into that tensor's shared block offset, enqueued on @p stream ahead of the exchange. A dead
 *  modality's zero-byte tensor (and any null source) is skipped; the landing's source view nulls the same
 *  modality, so those bytes are never read. */
static expected_t pack_batch_into_block_(multimodal_batch_on_gpu_t const &loaded, b8r1_span_t destination,
                                         batch_block_layout_t_ const &layout, gpu_stream_t &stream) noexcept {
    block_copy_t_ const copies[block_tensor_count_k] = {
        {block_segment_offsets_k, loaded.segment_offsets.data()},
        {block_segment_lengths_k, loaded.segment_lengths.data()},
        {block_segment_modality_k, loaded.segment_modality.data()},
        {block_segment_bag_k, loaded.segment_bag.data()},
        {block_tokens_k, loaded.tokens.data()},
        {block_image_shapes_k, loaded.image_shapes.data()},
        {block_video_shapes_k, loaded.video_shapes.data()},
        {block_images_k, loaded.images.data()},
        {block_videos_k, loaded.videos.data()},
        {block_audio_samples_k, loaded.audio_samples.data()},
    };
    for (block_copy_t_ const &copy : copies) {
        std::size_t const bytes = layout.byte_count(copy.tensor);
        if (bytes == 0u || copy.source == nullptr) continue;
        // vendor-boundary: device → device pack of one batch tensor into its block region — the tensors' `data()`
        // pointers are raw CUdeviceptr sources, not views, so the driver copy is spelled by hand here.
        CUdeviceptr const target = device_address(destination.data() + layout.offset(copy.tensor));
        CUdeviceptr const source = device_address(copy.source);
        if (CUresult packed = cuMemcpyDtoDAsync(target, source, bytes, stream.get()); packed != CUDA_SUCCESS)
            return cuda_failure(packed,
                                "packing batch tensor {} into the exchanged block failed " //
                                "(destination=0x{:x} source=0x{:x} bytes={})",
                                static_cast<u32_t>(copy.tensor), target, source, bytes);
    }
    return {};
}

/** Enqueue one member's mix over @p sources : the geometry pass, then the gather into @p mixed . */
template <typename sources_type_>
static expected_t mix_(sources_type_ const &sources, multimodal_batch_on_gpu_t &mixed, u32r1_span_t mixed_sources,
                       u32_t gpu_index_in_colocated_domain, data_epoch_t data_epoch, u64_t seed,
                       multimodal_batch_shape_t const &shape, gpu_stream_t &stream) noexcept {
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
    u32_t const image_patch_scalars = patch_scalars(shape.image.patch_side, shape.image.channels);
    u32_t const video_tube_scalars = tube_scalars(shape.video.tube_frames, shape.video.patch_side,
                                                  shape.video.channels);
    u32_t const audio_frame_scalars = shape.audio.chunk_samples;

    u32r1_span_t const mixed_offsets {mixed.segment_offsets.data(), mixed.segment_offsets.extent(0)};
    u32r1_span_t const mixed_lengths {mixed.segment_lengths.data(), mixed.segment_lengths.extent(0)};
    u32r1_span_t const mixed_bags {mixed.segment_bag.data(), mixed.segment_bag.extent(0)};
    dim3 const one_block {1u, 1u, 1u};
    dim3 const block_shape {kernel_block_size_default_k, 1u, 1u};
    dim3 const segment_grid {shape.max_segments_per_gpu, 1u, 1u};
    expected_t staged = launch_kernel(&batch_geometry_kernel_<sources_type_>, one_block, block_shape, stream, sources,
                                      mixed_offsets, mixed_lengths, mixed_sources, mixed_bags,
                                      gpu_index_in_colocated_domain, data_epoch, seed, shape.max_segments_per_gpu,
                                      shape.max_rows_per_gpu, shape.max_columns_per_row, shape.max_bags_per_gpu);
    if (staged)
        staged = launch_kernel(&batch_gather_kernel_<sources_type_>, segment_grid, block_shape, stream, sources, dest,
                               u32r1_view_t {mixed_offsets.data(), mixed_offsets.extent(0)},
                               u32r1_view_t {mixed_lengths.data(), mixed_lengths.extent(0)},
                               u32r1_view_t {mixed_sources.data(), mixed_sources.extent(0)}, shape.max_segments_per_gpu,
                               image_patch_scalars, video_tube_scalars, audio_frame_scalars);
    return staged;
}

expected<multimodal_batch_mixer_t> multimodal_batch_mixer_t::make(
    multimem_domain_t &multimem, colocated_domain_t &colocated, std::span<gpu_arena_t> arenas,
    multimodal_batch_shape_t shape, u64_t seed, std::span<gpu_stream_t> streams, f64_t timeout_seconds) noexcept {
    using result_t = expected<multimodal_batch_mixer_t>;
    std::size_t const members = multimem.local_member_count();
    if (members == 0 || arenas.size() != members)
        return result_t::fail(status_t::invalid_input_k,
                              "every local multimem_domain member needs its arena ({} members, {} arenas)", members,
                              arenas.size());
    if (members > max_gpus_per_host_k)
        return result_t::fail(status_t::shape_unsupported_k, "{} members exceed max_gpus_per_host_k ({})", members,
                              max_gpus_per_host_k);
    if (streams.size() != members)
        return result_t::fail(status_t::invalid_input_k, "every member needs its stream ({} members, {} streams)",
                              members, streams.size());
    u32_t const domains = colocated.multimem_domain_count();
    // Level-1 precondition: every local member's arena readable by every other — visibility is the
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
    mixer.multimem_domain_index_ = colocated.topology().multimem_domain_index();
    mixer.first_gpu_index_in_colocated_domain_ = colocated.topology().shape.gpu_index_of_multimem_member(
        mixer.multimem_domain_index_, 0u);
    // Each member's routing scratch, on its own arena — the geometry pass writes it, the gather pass
    // reads it back within the same mix.
    if (mixer.mixed_sources_.resize(members) != status_t::success_k)
        return result_t::fail(status_t::bad_alloc_k, "routing scratch table allocation failed ({} members)", members);
    for (std::size_t member = 0; member != members; ++member) {
        switch_to_gpu_t gpu_scope {streams[member]};
        if (!gpu_scope.ok())
            return result_t::fail(status_t::cuda_error_k, "context entry failed for member {}'s routing scratch",
                                  member);
        mixer.mixed_sources_[member] = u32r1_tensor_arena_t::uninitialized({shape.max_segments_per_gpu},
                                                                           gpu_arena_alloc<u32_t> {arenas[member]})
                                           .value;
        if (!mixer.mixed_sources_[member])
            return result_t::fail(status_t::bad_alloc_k, "member {}'s routing scratch allocation failed ({} segments)",
                                  member, shape.max_segments_per_gpu);
    }
    std::size_t const block_bytes = batch_block_layout_(shape).total_bytes();
    if (payload_scalars_needed(shape, colocated.topology().shape) == 0) return result_t {std::move(mixer)};
    expected<multimem_window_t> window = multimem.reserve(domains * block_bytes / sizeof(f32_t));
    if (!window) return result_t::fail(std::move(window));
    mixer.window_ = window.value_;
    if (domains == 1u) return result_t {std::move(mixer)};
    if (mixer.gather_buffers_.resize(members) != status_t::success_k)
        return result_t::fail(status_t::bad_alloc_k, "all-gather buffer table allocation failed ({} members)", members);
    for (std::size_t member = 0; member != members; ++member) {
        switch_to_gpu_t gpu_scope {streams[member]};
        if (!gpu_scope.ok())
            return result_t::fail(status_t::cuda_error_k, "context entry failed wiring member {}", member);
        u32_t const member_index = static_cast<u32_t>(multimem.member_offset() + member);
        auto wired = infiniband_all_gather<b8_t>::make(colocated, member_index, arenas[member], block_bytes,
                                                       streams[member], timeout_seconds);
        if (!wired) return result_t::fail(std::move(wired));
        mixer.gather_buffers_[member] = std::move(wired.value_);
    }
    return result_t {std::move(mixer)};
}

std::size_t multimodal_batch_mixer_t::payload_scalars_needed(multimodal_batch_shape_t const &shape,
                                                             mesh_shape_t const &mesh) noexcept {
    if (mesh.gpus_per_host_domain == mesh.gpus_per_colocated_domain) return 0;
    return mesh.multimem_domain_count() * batch_block_layout_(shape).total_bytes() / sizeof(f32_t);
}

expected_t multimodal_batch_mixer_t::operator()(std::size_t local, data_epoch_t data_epoch,
                                                std::span<multimodal_batch_on_gpu_t const> loaded,
                                                multimodal_batch_on_gpu_t &mixed, gpu_stream_t &stream) noexcept {
    std::size_t const members = mixed_sources_.size();
    if (local >= members) return expected_t::fail(status_t::invalid_input_k, "member {} of {}", local, members);
    if (loaded.size() != members)
        return expected_t::fail(status_t::invalid_input_k, "every member needs its loaded batch ({} of {})",
                                loaded.size(), members);
    switch_to_gpu_t gpu_scope {stream};
    if (!gpu_scope.ok()) return expected_t::fail(status_t::cuda_error_k, "context entry failed ({})", local);
    std::size_t const member_index = multimem_->member_offset() + local;

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
    batch_sources_t_ sources; // a kernel argument, so the peer table needs no device copy and owns nothing
    sources.count = static_cast<u32_t>(members);
    for (u32_t member = 0; member != sources.count; ++member)
        if (expected_t built = local_source(member, sources.entries[member]); !built) return built;

    batch_block_layout_t_ const block_layout = batch_block_layout_(shape_); // pure shape math, never stored
    if (window_.scalars != 0) {
        b8r1_span_t const window = multimem_->window_bytes(member_index, window_);
        expected_t packed {};
        if (domains_ == 1u) packed = pack_batch_into_block_(loaded[local], window, block_layout, stream);
        else {
            infiniband_all_gather<b8_t> &gather = gather_buffers_[local];
            packed = pack_batch_into_block_(loaded[local], gather.own_block(), block_layout, stream);
            if (packed) packed = gather(stream);
            if (packed) packed = copy_device_to_device(b8r1_view_t {gather.payload()}, window, stream);
        }
        if (!packed) return packed;
    }

    // The step's first barrier orders "everyone loaded" before any peer read; the round is the member's
    // own, self-advanced inside the barrier kernel.
    if (expected_t arrived = multimem_->barrier(member_index, stream); !arrived) return arrived;

    // The routing destination is this member's world coordinate — the multimem_domains claim disjoint
    // destination ranges in the members·domains world (a single domain's base is 0).
    u32_t const gpu_index_in_colocated_domain = first_gpu_index_in_colocated_domain_ + static_cast<u32_t>(member_index);
    u32r1_span_t const mixed_sources = mixed_sources_[local].span();
    if (window_.scalars != 0) {
        batch_windows_t_ windows;
        windows.first_window = multimem_->window_bytes(0, window_).data();
        windows.member_stride = multimem_->member(0).replica_stride_words * sizeof(u32_t);
        windows.layout = block_layout;
        windows.members = static_cast<u32_t>(multimem_->member_count());
        windows.count = windows.members * domains_;
        return mix_(windows, mixed, mixed_sources, gpu_index_in_colocated_domain, data_epoch, seed_, shape_, stream);
    }
    return mix_(sources, mixed, mixed_sources, gpu_index_in_colocated_domain, data_epoch, seed_, shape_, stream);
}

expected_t multimodal_batch_mixer_t::operator()(data_epoch_t data_epoch,
                                                std::span<multimodal_batch_on_gpu_t const> loaded,
                                                std::span<multimodal_batch_on_gpu_t> mixed,
                                                std::span<gpu_stream_t> streams) noexcept {
    std::size_t const members = mixed_sources_.size();
    if (streams.size() != members || mixed.size() != members)
        return expected_t::fail(status_t::invalid_input_k, "every member needs its stream and mixed batch ({} members)",
                                members);
    for (std::size_t local = 0; local != members; ++local)
        if (expected_t staged = (*this)(local, data_epoch, loaded, mixed[local], streams[local]); !staged)
            return staged;
    return {};
}
