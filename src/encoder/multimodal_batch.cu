/**
 *  @file src/encoder/multimodal_batch.cu
 *  @author Ash Vardanian
 *  @date July 3, 2026
 *  @brief The batch's implementation: the single segment-driven coordinate adapter (
 *      @ref fill_coords ) and the batch mixer's routing bijection and two exchange kernels —
 *      compiled once into the @c uform_train library.
 *
 *  Everything stays on the GPU: peer addresses travel as kernel arguments, and every kernel
 *  re-derives its routes in-register from the keyed bijection, so the mixer owns no memory at all;
 *  the coordinate adapter is equally a pure enqueue.
 */
#include "uform/encoder/multimodal_batch.cuh"

namespace ue = unum::uform::encoder;
using namespace ue;
using namespace unum::uform::utilities;

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

expected_t ue::fill_coords(u32r1_view_t segment_offsets, u32r1_view_t segment_lengths, modality_view_t segment_modality,
                           image_shape_view_t image_shapes, video_shape_view_t video_shapes, u32_t tube_frames,
                           u32_t frame_stride, i16r2_span_t coords, gpu_stream_t &stream) noexcept {
    if (coords.extent(1) != 3)
        return expected_t::fail(status_t::unsupported_k, "coordinate table must be [{},3]", coords.extent(0));
    std::size_t const segments = segment_lengths.extent(0);
    if (segment_offsets.extent(0) < segments || segment_modality.extent(0) != segments)
        return expected_t::fail(status_t::invalid_argument_k,
                                "offsets ({}), lengths ({}), and modality ({}) must span the segment count",
                                segment_offsets.extent(0), segments, segment_modality.extent(0));
    if ((image_shapes.extent(0) != 0 && image_shapes.extent(1) != 2) ||
        (video_shapes.extent(0) != 0 && video_shapes.extent(1) != 3))
        return expected_t::fail(status_t::unsupported_k, "shape descriptors must be image [{},2], video [{},3]",
                                image_shapes.extent(0), video_shapes.extent(0));
    if (segments == 0) return {};
    dim3 const block_shape {kernel_block_size_default_k, 1u, 1u};
    dim3 const grid {static_cast<unsigned>(segments), 1u, 1u};
    return launch_kernel(&fill_coords_kernel_, grid, block_shape, stream, segment_offsets, segment_lengths,
                         segment_modality, image_shapes, video_shapes, tube_frames,
                         frame_stride == 0u ? 1u : frame_stride, coords);
}

expected_t ue::pad_choices(multimodal_batch_on_gpu_t &batch, gpu_stream_t &stream) noexcept {
    if (batch.segment_choices.empty()) return {};
    if (CUresult result = cuMemsetD16Async(device_address(batch.segment_choices.data()), byte_pad_k,
                                           batch.segment_choices.numel(), stream.get());
        result != CUDA_SUCCESS)
        return cuda_failure(result, "choice padding failed");
    return zero_async(batch.segment_choice_targets.span().flatten<1>(), stream);
}

/** One member's batch, the gather kernels' read view of a peer. Carries every per-segment
 *  descriptor and every modality's content as @b [cells,scalars] rows, so a routed segment's media
 *  survives the cross-GPU shuffle, not just its tokens. Positions index the flat @c tokens run. A
 *  modality absent from the batch leaves its view empty. */
struct batch_source_t_ {
    tokens_view_t tokens {};
    u32r1_view_t segment_offsets {}, segment_lengths {}, segment_bag {};
    modality_view_t segment_modality {};
    nk::tensor_view<segment_role_t, 1> segment_role {};
    u32r1_view_t segment_prefix_length {};
    u64r1_view_t segment_key {};
    nk::tensor_view<late_interaction_side_t, 1> segment_side {};
    nk::tensor_view<u16_t, 2> image_shapes {}, video_shapes {};
    nk::tensor_view<pixel_t, 2> images {}, videos {};
    f32r2_view_t audio_samples {};
    nk::tensor_view<token_id_t, 2> segment_choices {};
    f32r2_view_t segment_choice_targets {};
};

/** This member's mixed batch, the gather kernel's scatter target. Mirrors the routed descriptors
 *  and content from @ref batch_source_t_ ; a modality absent here leaves its span empty and its
 *  copy is skipped. */
struct batch_dest_t_ {
    tokens_span_t tokens {};
    modality_span_t segment_modality {};
    nk::tensor_span<segment_role_t, 1> segment_role {};
    u32r1_span_t segment_prefix_length {};
    u64r1_span_t segment_key {};
    nk::tensor_span<late_interaction_side_t, 1> segment_side {};
    nk::tensor_span<u16_t, 2> image_shapes {}, video_shapes {};
    nk::tensor_span<pixel_t, 2> images {}, videos {};
    f32r2_span_t audio_samples {};
    nk::tensor_span<token_id_t, 2> segment_choices {};
    f32r2_span_t segment_choice_targets {};
};

/** The sentinel the mixer's routing scratch carries for a mixed PAD segment that has no world
 *  source — a dropped bag's segments and the unclaimed tail both read as this. */
static constexpr u32_t mixed_source_absent_k = ~u32_t {0};

/** The cursor a run of @p columns of @p modality starts at: a media item at its next half or whole
 *  row, as @ref media_cells places it, and text jumped to the next row when it would otherwise
 *  cross a row boundary, so a segment never spans two rows. */
[[nodiscard]] static __device__ u32_t row_aligned_cursor_(u32_t cursor, u32_t columns, modality_t modality,
                                                          u32_t columns_per_row) noexcept {
    u32_t const cells = media_cells(modality, columns_per_row);
    if (cells != 0u) return (cursor + cells - 1u) / cells * cells;
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

/** The first segment of @p peer in @c [from,end) whose membership in the live run of @p bag equals
 *  @p inside , or @p end when none: a warp tests 32 segments per step and every lane returns the
 *  same lowest match. */
[[nodiscard]] static __device__ u32_t find_run_edge_(batch_source_t_ const &peer, u32_t bag, u32_t from, u32_t end,
                                                     bool inside) noexcept {
    u32_t const lane = threadIdx.x % warp_size_k;
    for (u32_t base = from; base < end; base += warp_size_k) {
        u32_t const segment = base + lane;
        bool const live = segment < end && peer.segment_bag[segment] == bag && peer.segment_lengths[segment] != 0u;
        u32_t const matches = __ballot_sync(0xFFFFFFFFu, segment < end && live == inside);
        if (matches != 0u) return base + cuda_warp_t::first(matches);
    }
    return end;
}

/**
 *  @brief Write the mixed batch's geometry: one warp walks the destinations serially, as the repack
 *      cursor is a strict scan, and searches each source bag's run 32 segments at a time.
 *
 *  It derives each destination's world source through the shared @ref epoch_permutation (read
 *  gather-wise: a destination looks up its source, equivalent to scattering by the inverse
 *  permutation, and identical on every GPU so routing needs no exchange), records every mixed
 *  segment's world source into @p mixed_sources for the gather pass, and jumps the cursor past any
 *  row boundary a segment would cross. The routing granularity is the segment when
 *  @p max_bags_per_gpu is zero and the @b bag otherwise, over a world axis of @c max_bags · world
 *  bags: a source bag's live segments (the contiguous run tagged with its index) land together,
 *  its anchors in a trailing band after every other role's segments, and @p mixed_bags renumbers
 *  them to the destination-local bag. A bag that would overflow the destination rectangle or the
 *  segment table is dropped @b whole — never split — so a surviving bag is always intact; the
 *  packer's budgets make any drop a bug, not a case. Answers how many live segments found no room,
 *  every lane alike.
 *
 *  Empty @p mixed_offsets and @p mixed_bags route a destination for its lengths and sources
 *  alone, the way @ref batch_world_geometry_kernel_ derives every other GPU's mix, which lands
 *  nowhere and reports no drop.
 */
template <typename sources_type_>
static __device__ u32_t route_destination_(sources_type_ const &sources, u32r1_span_t mixed_offsets,
                                           u32r1_span_t mixed_lengths, u32r1_span_t mixed_sources,
                                           u32r1_span_t mixed_bags, u32_t member_index, data_epoch_t data_epoch,
                                           u64_t seed, u32_t segments_per_gpu, u32_t rows_per_gpu,
                                           u32_t columns_per_row, u32_t max_bags_per_gpu) noexcept {
    bool const lands = !mixed_offsets.empty();
    bool const leader = threadIdx.x % warp_size_k == 0u; // every lane walks the same cursor, one stores it
    u32_t const capacity = rows_per_gpu * columns_per_row;
    u32_t cursor = 0, dropped = 0;

    if (max_bags_per_gpu == 0u) {
        // Per-segment routing — the masked pipeline's degenerate case: one segment per bag, the
        // world axis is the segment axis, and every destination claims exactly one source.
        std::size_t const world_segments = static_cast<std::size_t>(segments_per_gpu) * sources.count;
        for (u32_t segment = 0; segment != segments_per_gpu; ++segment) {
            std::size_t const destination = static_cast<std::size_t>(member_index) * segments_per_gpu + segment;
            std::size_t const source = epoch_permutation(seed, data_epoch, destination, world_segments);
            batch_source_t_ const &peer = sources[static_cast<u32_t>(source / segments_per_gpu)];
            u32_t columns = peer.segment_lengths[source % segments_per_gpu];
            cursor = row_aligned_cursor_(cursor, columns, peer.segment_modality[source % segments_per_gpu],
                                         columns_per_row);
            if (!lands && cursor + columns > capacity) columns = 0; // the world pass drops silently
            if (cursor + columns > capacity) ++dropped, columns = 0;
            if (leader) {
                if (lands) mixed_offsets[segment] = cursor;
                mixed_lengths[segment] = columns;
                mixed_sources[segment] = static_cast<u32_t>(source);
            }
            cursor += columns;
        }
        if (lands && leader) mixed_offsets[segments_per_gpu] = cursor;
        return dropped;
    }

    // Bag-cohesive routing: the destination walks its bag budget, each drawing one world bag;
    // the source bag's live segments are the contiguous run tagged with its index in the peer's bag
    // column, so the walk finds the run's start once and consumes it to its boundary — a tag that
    // breaks contiguity simply ends the run instead of being gathered from afar. The anchors land
    // in a trailing band after every other role, the loader's own layout, so a caption never
    // strands a row beside full-row media. The walk runs twice with the same decisions: the first
    // sizes the leading band, the second writes both, each band ascending.
    std::size_t const world_bags = static_cast<std::size_t>(max_bags_per_gpu) * sources.count;
    u32_t lead_cursor = 0, trail_cursor = 0, lead_segments = 0, trail_segments = 0, trail_begin = 0;
    for (u32_t pass = 0; pass != 2u; ++pass) {
        bool const writes = pass == 1u;
        trail_begin = writes ? static_cast<u32_t>(divide_round_up(lead_cursor, columns_per_row)) * columns_per_row : 0u;
        u32_t const lead_total = lead_segments;
        lead_cursor = trail_cursor = lead_segments = trail_segments = 0;
        for (u32_t bag = 0; bag != max_bags_per_gpu; ++bag) {
            std::size_t const destination_bag = static_cast<std::size_t>(member_index) * max_bags_per_gpu + bag;
            std::size_t const source_bag_world = epoch_permutation(seed, data_epoch, destination_bag, world_bags);
            batch_source_t_ const &peer = sources[static_cast<u32_t>(source_bag_world / max_bags_per_gpu)];
            u32_t const source_bag = static_cast<u32_t>(source_bag_world % max_bags_per_gpu);
            u32_t const source_gpu_base = static_cast<u32_t>(source_bag_world / max_bags_per_gpu) * segments_per_gpu;

            u32_t const run_begin = find_run_edge_(peer, source_bag, 0u, segments_per_gpu, true);
            u32_t const run_end = find_run_edge_(peer, source_bag, run_begin, segments_per_gpu, false);
            auto const trails = [&](u32_t source_segment) {
                return !peer.segment_role.empty() && peer.segment_role[source_segment] == segment_role_t::anchor_k;
            };

            // Trial placement first, then commit — a bag that cannot land whole lands not at all.
            u32_t lead_trial = lead_cursor, trail_trial = trail_cursor;
            bool fits = lead_segments + trail_segments + (run_end - run_begin) <= segments_per_gpu;
            for (u32_t source_segment = run_begin; source_segment != run_end && fits; ++source_segment) {
                u32_t const columns = peer.segment_lengths[source_segment];
                u32_t &trial = trails(source_segment) ? trail_trial : lead_trial;
                trial = row_aligned_cursor_(trial, columns, peer.segment_modality[source_segment], columns_per_row) +
                        columns;
                fits = divide_round_up(lead_trial, columns_per_row) + divide_round_up(trail_trial, columns_per_row) <=
                       rows_per_gpu;
            }
            if (!fits) {
                if (writes && lands) dropped += run_end - run_begin;
                continue;
            }
            for (u32_t source_segment = run_begin; source_segment != run_end; ++source_segment) {
                bool const trailing = trails(source_segment);
                u32_t const columns = peer.segment_lengths[source_segment];
                u32_t &cursor = trailing ? trail_cursor : lead_cursor;
                cursor = row_aligned_cursor_(cursor, columns, peer.segment_modality[source_segment], columns_per_row);
                u32_t const destination_segment = trailing ? lead_total + trail_segments : lead_segments;
                if (writes && leader) {
                    if (lands)
                        mixed_offsets[destination_segment] = (trailing ? trail_begin : 0u) + cursor,
                        mixed_bags[destination_segment] = bag;
                    mixed_lengths[destination_segment] = columns;
                    mixed_sources[destination_segment] = source_gpu_base + source_segment;
                }
                cursor += columns;
                ++(trailing ? trail_segments : lead_segments);
            }
        }
    }
    if (!leader) return dropped;
    // The unclaimed tail is PAD: zero length, no source, and the
    // one-past-the-last-bag `end` sentinel.
    u32_t const end = trail_segments != 0u ? trail_begin + trail_cursor : lead_cursor;
    for (u32_t segment = lead_segments + trail_segments; segment != segments_per_gpu; ++segment) {
        if (lands) mixed_offsets[segment] = end, mixed_bags[segment] = max_bags_per_gpu;
        mixed_lengths[segment] = 0u;
        mixed_sources[segment] = mixed_source_absent_k;
    }
    if (lands) mixed_offsets[segments_per_gpu] = end;
    return dropped;
}

/**
 *  @brief Write the mixed batch's geometry through @ref route_destination_ for world GPU
 *      @p member_index , adding the member's loaded drops and the routing's own to
 *      @p mixed_dropped .
 *
 *  __grid_size__ @c [1,1,1] of one warp; no dynamic shared memory.
 */
template <typename sources_type_>
static __global__ void batch_geometry_kernel_(sources_type_ sources, u32r1_span_t mixed_offsets,
                                              u32r1_span_t mixed_lengths, u32r1_span_t mixed_sources,
                                              u32r1_span_t mixed_bags, u32r1_view_t loaded_dropped,
                                              u32r1_span_t mixed_dropped, u32_t member_index, data_epoch_t data_epoch,
                                              u64_t seed, u32_t segments_per_gpu, u32_t rows_per_gpu,
                                              u32_t columns_per_row, u32_t max_bags_per_gpu) {
    if (blockIdx.x != 0u) return;
    u32_t const dropped = route_destination_(sources, mixed_offsets, mixed_lengths, mixed_sources, mixed_bags,
                                             member_index, data_epoch, seed, segments_per_gpu, rows_per_gpu,
                                             columns_per_row, max_bags_per_gpu);
    if (threadIdx.x == 0u) mixed_dropped[0] += loaded_dropped[0] + dropped;
}

/**
 *  @brief Derive every world GPU's mixed lengths and world sources, both
 *      @b [world_gpus,segments_per_gpu] , since every GPU routes from the same loaded batches
 *      through the same keyed bijection.
 *
 *  __grid_size__ @c [world_gpus,1,1] of one warp each; CTA @c g routes world GPU @c g ; no
 *  dynamic shared memory.
 */
template <typename sources_type_>
static __global__ void batch_world_geometry_kernel_(sources_type_ sources, u32r2_span_t world_lengths,
                                                    u32r2_span_t world_sources, data_epoch_t data_epoch, u64_t seed,
                                                    u32_t segments_per_gpu, u32_t rows_per_gpu, u32_t columns_per_row,
                                                    u32_t max_bags_per_gpu) {
    u32_t const gpu = blockIdx.x;
    if (gpu >= sources.count) return;
    route_destination_(sources, u32r1_span_t {}, world_lengths[gpu, nk::slice], world_sources[gpu, nk::slice],
                       u32r1_span_t {}, gpu, data_epoch, seed, segments_per_gpu, rows_per_gpu, columns_per_row,
                       max_bags_per_gpu);
}

/**
 *  @brief Gather one routed segment: CTA @c s reads route @c s 's world source from the geometry
 *      pass's @p mixed_sources table (with variable-degree bags a destination's source is not
 *      derivable per segment).
 *
 *  It copies the peer's per-segment descriptors ( @ref batch_source_t_::segment_modality , the
 *  role, prefix, key and side, the choice and target rows, and the image / video @c *_shapes rows —
 *  thread 0, so the mixed batch is fully typed) and then, dispatching on the segment's modality,
 *  copies its content ( @c tokens , or the @c [begin,begin+columns) rows of patch, tube or frame
 *  cells) from the peer's loaded batch into this member's mixed batch, threads striding the run. A
 *  sourceless PAD (a dropped bag or the unclaimed tail) is a @c text_k anchor outside late
 *  interaction that makes no choice and carries nothing, and every PAD is keyless. The destination
 *  geometry is what @ref batch_geometry_kernel_ wrote.
 *
 *  __grid_size__ @c [segments_per_gpu,1,1] , threads stripe the content run; no
 *  dynamic shared memory.
 */
template <typename sources_type_>
static __global__ void batch_gather_kernel_(sources_type_ sources, batch_dest_t_ dest, u32r1_view_t mixed_offsets,
                                            u32r1_view_t mixed_lengths, u32r1_view_t mixed_sources,
                                            u32_t segments_per_gpu) {
    u32_t const segment = blockIdx.x;
    u32_t const source = mixed_sources[segment];
    if (source == mixed_source_absent_k) {
        if (threadIdx.x != 0u) return;
        if (!dest.segment_modality.empty()) dest.segment_modality[segment] = modality_t::text_k;
        if (!dest.segment_role.empty()) dest.segment_role[segment] = segment_role_t::anchor_k;
        if (!dest.segment_prefix_length.empty()) dest.segment_prefix_length[segment] = 0u;
        if (!dest.segment_key.empty()) dest.segment_key[segment] = 0u;
        if (!dest.segment_side.empty()) dest.segment_side[segment] = late_interaction_side_t::none_k;
        for (std::size_t choice = 0; choice != dest.segment_choices.extent(1); ++choice) {
            dest.segment_choices[segment, choice] = byte_pad_k;
            dest.segment_choice_targets[segment, choice] = 0.0f;
        }
        return;
    }
    batch_source_t_ const &peer = sources[source / segments_per_gpu];
    u32_t const source_segment = source % segments_per_gpu;

    // Descriptors travel for every routed segment, PAD included, so the mixed batch
    // stays fully typed.
    if (threadIdx.x == 0u) {
        if (!dest.segment_modality.empty() && !peer.segment_modality.empty())
            dest.segment_modality[segment] = peer.segment_modality[source_segment];
        if (!dest.segment_role.empty() && !peer.segment_role.empty())
            dest.segment_role[segment] = peer.segment_role[source_segment];
        if (!dest.segment_prefix_length.empty() && !peer.segment_prefix_length.empty())
            dest.segment_prefix_length[segment] = peer.segment_prefix_length[source_segment];
        if (!dest.segment_key.empty() && !peer.segment_key.empty())
            dest.segment_key[segment] = mixed_lengths[segment] != 0u ? peer.segment_key[source_segment] : 0u;
        if (!dest.segment_side.empty() && !peer.segment_side.empty())
            dest.segment_side[segment] = peer.segment_side[source_segment];
        if (!dest.image_shapes.empty() && !peer.image_shapes.empty())
            for (u32_t i = 0; i != 2u; ++i) dest.image_shapes[segment, i] = peer.image_shapes[source_segment, i];
        if (!dest.video_shapes.empty() && !peer.video_shapes.empty())
            for (u32_t i = 0; i != 3u; ++i) dest.video_shapes[segment, i] = peer.video_shapes[source_segment, i];
        for (std::size_t choice = 0; choice != dest.segment_choices.extent(1); ++choice) {
            dest.segment_choices[segment, choice] = peer.segment_choices[source_segment, choice];
            dest.segment_choice_targets[segment, choice] = peer.segment_choice_targets[source_segment, choice];
        }
    }

    u32_t const columns = mixed_lengths[segment];
    if (columns == 0u) return;
    std::size_t const source_begin = peer.segment_offsets[source_segment];
    std::size_t const destination_begin = mixed_offsets[segment];
    auto const copy_cells = [&](auto const &from, auto const &to) noexcept {
        if (from.empty() || to.empty()) return;
        auto const source_run = from[nk::range(source_begin, source_begin + columns), nk::slice].template flatten<1>();
        auto const destination_run =
            to[nk::range(destination_begin, destination_begin + columns), nk::slice].template flatten<1>();
        for (std::size_t i = threadIdx.x; i < source_run.extent(0); i += blockDim.x) destination_run[i] = source_run[i];
    };
    modality_t const modality = !peer.segment_modality.empty() ? peer.segment_modality[source_segment]
                                                               : modality_t::text_k;
    switch (modality) {
    case modality_t::text_k:
        for (u32_t column = threadIdx.x; column < columns; column += blockDim.x)
            dest.tokens[destination_begin + column] = peer.tokens[source_begin + column];
        break;
    case modality_t::image_k: copy_cells(peer.images, dest.images); break;
    case modality_t::video_k: copy_cells(peer.videos, dest.videos); break;
    case modality_t::audio_k: copy_cells(peer.audio_samples, dest.audio_samples); break;
    }
}

/**
 *  @brief The key of every world GPU's mixed segment, read from the segment's source batch once the
 *      world pass completed @p world_sources ; a PAD or sourceless segment is keyless.
 *
 *  __grid_size__ @c [world_gpus,1,1] , threads stripe the segments; no dynamic shared memory.
 */
template <typename sources_type_>
static __global__ void batch_world_keys_kernel_(sources_type_ sources, u32r2_view_t world_lengths,
                                                u32r2_view_t world_sources, u64r2_span_t world_keys,
                                                u32_t segments_per_gpu) {
    u32_t const gpu = blockIdx.x;
    if (gpu >= sources.count) return;
    for (u32_t segment = threadIdx.x; segment < segments_per_gpu; segment += blockDim.x) {
        u32_t const source = world_sources[gpu, segment];
        u64_t key = 0u;
        if (source != mixed_source_absent_k && world_lengths[gpu, segment] != 0u) {
            batch_source_t_ const &peer = sources[source / segments_per_gpu];
            if (!peer.segment_key.empty()) key = peer.segment_key[source % segments_per_gpu];
        }
        world_keys[gpu, segment] = key;
    }
}

/** The sixteen batch tensors, in their fixed block-serialization order — the index into a
 *  @ref batch_block_layout_t_ . A fixed family, so it is an enum, not a run of constants;
 *  every GPU shares one shape, so the order (and thus every landing's byte layout) is
 *  identical across multimem_domains. */
enum batch_block_tensor_t_ : u32_t {
    block_segment_offsets_k = 0,
    block_segment_lengths_k = 1,
    block_segment_modality_k = 2,
    block_segment_bag_k = 3,
    block_segment_role_k = 4,
    block_segment_prefix_length_k = 5,
    block_segment_key_k = 6,
    block_segment_side_k = 7,
    block_tokens_k = 8,
    block_image_shapes_k = 9,
    block_video_shapes_k = 10,
    block_images_k = 11,
    block_videos_k = 12,
    block_audio_samples_k = 13,
    block_segment_choices_k = 14,
    block_segment_choice_targets_k = 15,
};

/** How many tensors one exchanged block serializes — the extent of @ref batch_block_tensor_t_ . */
static constexpr u32_t block_tensor_count_k = block_segment_choice_targets_k + 1u;

/**
 *  @brief The serialization map of one @ref multimodal_batch_on_gpu_t inside an exchanged block: a
 *      cache-line-aligned byte offset and a byte count per tensor, packed back to back in
 *      @ref batch_block_tensor_t_ order.
 *
 *  A dead modality (its content tensor unallocated) still gets an offset but a zero byte count, so
 *  it contributes nothing and its offset is never dereferenced. @ref total_bytes is the whole block
 *  — the @c block_scalars an @c infiniband_all_gather<b8_t> is wired with.
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
    // The bag column rides the block even when bags are off (the layout is fixed shape math) —
    // those bytes are dead on the masked path and small: one u32 per segment.
    layout.byte_counts[block_segment_bag_k] = segments * sizeof(u32_t);
    layout.byte_counts[block_segment_role_k] = segments * sizeof(segment_role_t);
    layout.byte_counts[block_segment_prefix_length_k] = segments * sizeof(u32_t);
    layout.byte_counts[block_segment_key_k] = segments * sizeof(u64_t);
    layout.byte_counts[block_segment_side_k] = segments * sizeof(late_interaction_side_t);
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
    std::size_t const choices = segments * shape.max_choices_per_segment;
    layout.byte_counts[block_segment_choices_k] = choices * sizeof(token_id_t);
    layout.byte_counts[block_segment_choice_targets_k] = choices * sizeof(f32_t);
    std::size_t running = 0;
    for (u32_t tensor = 0; tensor != block_tensor_count_k; ++tensor) {
        layout.offsets[tensor] = round_up_to_multiple(running, cache_line_bytes_k);
        running = layout.offsets[tensor] + layout.byte_counts[tensor];
    }
    layout.block_bytes = round_up_to_multiple(running, cache_line_bytes_k);
    return layout;
}

/** Build the read view of a member's own loaded batch, a direct NVLink peer read for a source GPU
 *  on this multimem_domain. A modality absent from the batch leaves an empty view, exactly as the
 *  gather kernels expect. */
static batch_source_t_ source_from_loaded_(multimodal_batch_on_gpu_t const &loaded) noexcept {
    std::size_t const cells = loaded.tokens.numel();
    batch_source_t_ source;
    source.tokens = loaded.tokens.flatten<1>();
    source.segment_offsets = loaded.segment_offsets.view();
    source.segment_lengths = loaded.segment_lengths.view();
    source.segment_modality = loaded.segment_modality.view();
    source.segment_bag = loaded.segment_bag.view();
    source.segment_role = loaded.segment_role.view();
    source.segment_prefix_length = loaded.segment_prefix_length.view();
    source.segment_key = loaded.segment_key.view();
    source.segment_side = loaded.segment_side.view();
    source.image_shapes = loaded.image_shapes.view();
    source.video_shapes = loaded.video_shapes.view();
    source.images = as_rows(loaded.images, cells);
    source.videos = as_rows(loaded.videos, cells);
    source.audio_samples = as_rows(loaded.audio_samples, cells);
    source.segment_choices = loaded.segment_choices.view();
    source.segment_choice_targets = loaded.segment_choice_targets.view();
    return source;
}

/** Every world GPU's packed batch in the multimem_domain's symmetric windows: world GPU @c domain ·
 *  members+member is block @c domain of that member's window. Resolved arithmetically in-kernel, so
 *  no table bounds the world; a dead modality's zero-byte region reads as null, exactly as the
 *  gather kernels expect. */
struct batch_windows_t_ {
    b8_t const *first_window = nullptr; // device-boundary
    std::size_t member_stride = 0;
    batch_block_layout_t_ layout {};
    u32_t members = 0;
    u32_t count = 0;

    template <typename value_type_>
    [[nodiscard]] __device__ nk::tensor_view<value_type_, 1> region_(b8_t const *block,
                                                                     batch_block_tensor_t_ tensor) const noexcept {
        std::size_t const bytes = layout.byte_counts[tensor];
        if (bytes == 0u) return {};
        return {static_cast<value_type_ const *>(static_cast<void const *>(block + layout.offsets[tensor])),
                bytes / sizeof(value_type_)};
    }

    [[nodiscard]] __device__ batch_source_t_ operator[](u32_t gpu) const noexcept {
        b8_t const *block = first_window + (gpu % members) * member_stride + (gpu / members) * layout.block_bytes;
        batch_source_t_ source;
        source.segment_offsets = region_<u32_t>(block, block_segment_offsets_k);
        source.segment_lengths = region_<u32_t>(block, block_segment_lengths_k);
        source.segment_modality = region_<modality_t>(block, block_segment_modality_k);
        source.segment_bag = region_<u32_t>(block, block_segment_bag_k);
        source.segment_role = region_<segment_role_t>(block, block_segment_role_k);
        source.segment_prefix_length = region_<u32_t>(block, block_segment_prefix_length_k);
        source.segment_key = region_<u64_t>(block, block_segment_key_k);
        source.segment_side = region_<late_interaction_side_t>(block, block_segment_side_k);
        source.tokens = region_<token_id_t>(block, block_tokens_k);
        std::size_t const segments = source.segment_lengths.extent(0), cells = source.tokens.extent(0);
        source.image_shapes = as_rows(region_<u16_t>(block, block_image_shapes_k), segments);
        source.video_shapes = as_rows(region_<u16_t>(block, block_video_shapes_k), segments);
        source.images = as_rows(region_<pixel_t>(block, block_images_k), cells);
        source.videos = as_rows(region_<pixel_t>(block, block_videos_k), cells);
        source.audio_samples = as_rows(region_<f32_t>(block, block_audio_samples_k), cells);
        source.segment_choices = as_rows(region_<token_id_t>(block, block_segment_choices_k), segments);
        source.segment_choice_targets = as_rows(region_<f32_t>(block, block_segment_choice_targets_k), segments);
        return source;
    }
};

/** One tensor's region in the exchanged block — its identity and its device source pointer, a named
 *  pair so the pack loop is a plain table walk rather than fourteen hand-spelled copies. */
struct block_copy_t_ {
    batch_block_tensor_t_ tensor;
    void const *source;
};

/** Serialize a member's loaded batch into its outgoing block: one device → device copy per live
 *  tensor into that tensor's shared block offset, enqueued on @p stream ahead of the exchange. A
 *  dead modality's zero-byte tensor (and any null source) is skipped; the landing's source view
 *  nulls the same modality, so those bytes are never read. */
static expected_t pack_batch_into_block_(multimodal_batch_on_gpu_t const &loaded, b8r1_span_t destination,
                                         batch_block_layout_t_ const &layout, gpu_stream_t &stream) noexcept {
    block_copy_t_ const copies[block_tensor_count_k] = {
        {block_segment_offsets_k, loaded.segment_offsets.data()},
        {block_segment_lengths_k, loaded.segment_lengths.data()},
        {block_segment_modality_k, loaded.segment_modality.data()},
        {block_segment_bag_k, loaded.segment_bag.data()},
        {block_segment_role_k, loaded.segment_role.data()},
        {block_segment_prefix_length_k, loaded.segment_prefix_length.data()},
        {block_segment_key_k, loaded.segment_key.data()},
        {block_segment_side_k, loaded.segment_side.data()},
        {block_tokens_k, loaded.tokens.data()},
        {block_image_shapes_k, loaded.image_shapes.data()},
        {block_video_shapes_k, loaded.video_shapes.data()},
        {block_images_k, loaded.images.data()},
        {block_videos_k, loaded.videos.data()},
        {block_audio_samples_k, loaded.audio_samples.data()},
        {block_segment_choices_k, loaded.segment_choices.data()},
        {block_segment_choice_targets_k, loaded.segment_choice_targets.data()},
    };
    for (block_copy_t_ const &copy : copies) {
        std::size_t const bytes = layout.byte_count(copy.tensor);
        if (bytes == 0u || copy.source == nullptr) continue;
        // vendor-boundary: device → device pack of one batch tensor into its block region — the
        // tensors' `data()` pointers are raw CUdeviceptr sources, not views, so the driver copy is
        // spelled by hand here.
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

/** Enqueue one member's mix over @p sources : the geometry pass, the gather into @p mixed , and the
 *  world pass into @p world_lengths and @p world_sources , whose own slice the gather reads. The
 *  geometry pass adds @p loaded_dropped , the member's own loaded drops, to the mixed batch's. */
template <typename sources_type_>
static expected_t mix_(sources_type_ const &sources, u32r1_view_t loaded_dropped, multimodal_batch_on_gpu_t &mixed,
                       u32r2_span_t world_lengths, u32r2_span_t world_sources, u64r2_span_t world_keys,
                       u32_t gpu_index_in_colocated_domain, data_epoch_t data_epoch, u64_t seed,
                       multimodal_batch_shape_t const &shape, gpu_stream_t &stream) noexcept {
    u32r1_span_t const mixed_sources = world_sources[gpu_index_in_colocated_domain, nk::slice];
    // This member's mixed batch is the scatter target; an absent modality leaves an empty span.
    std::size_t const cells = mixed.tokens.numel();
    batch_dest_t_ dest;
    dest.tokens = mixed.tokens.flatten<1>();
    dest.segment_modality = mixed.segment_modality.span();
    dest.segment_role = mixed.segment_role.span();
    dest.segment_prefix_length = mixed.segment_prefix_length.span();
    dest.segment_key = mixed.segment_key.span();
    dest.segment_side = mixed.segment_side.span();
    dest.image_shapes = mixed.image_shapes.span();
    dest.video_shapes = mixed.video_shapes.span();
    dest.images = as_rows(mixed.images, cells);
    dest.videos = as_rows(mixed.videos, cells);
    dest.audio_samples = as_rows(mixed.audio_samples, cells);
    dest.segment_choices = mixed.segment_choices.span();
    dest.segment_choice_targets = mixed.segment_choice_targets.span();

    u32r1_span_t const mixed_offsets = mixed.segment_offsets.span(), mixed_lengths = mixed.segment_lengths.span();
    u32r1_span_t const mixed_bags = mixed.segment_bag.span();
    dim3 const one_block {1u, 1u, 1u}, one_warp {warp_size_k, 1u, 1u};
    dim3 const block_shape {kernel_block_size_default_k, 1u, 1u};
    dim3 const segment_grid {shape.max_segments_per_gpu, 1u, 1u};
    expected_t staged = launch_kernel(
        &batch_geometry_kernel_<sources_type_>, one_block, one_warp, stream, sources, mixed_offsets, mixed_lengths,
        mixed_sources, mixed_bags, loaded_dropped, mixed.dropped.span(), gpu_index_in_colocated_domain, data_epoch,
        seed, shape.max_segments_per_gpu, shape.max_rows_per_gpu, shape.max_columns_per_row, shape.max_bags_per_gpu);
    if (staged)
        staged = launch_kernel(&batch_gather_kernel_<sources_type_>, segment_grid, block_shape, stream, sources, dest,
                               u32r1_view_t {mixed_offsets}, u32r1_view_t {mixed_lengths}, u32r1_view_t {mixed_sources},
                               shape.max_segments_per_gpu);
    if (staged)
        staged = launch_kernel(&batch_world_geometry_kernel_<sources_type_>, dim3 {sources.count, 1u, 1u}, one_warp,
                               stream, sources, world_lengths, world_sources, data_epoch, seed,
                               shape.max_segments_per_gpu, shape.max_rows_per_gpu, shape.max_columns_per_row,
                               shape.max_bags_per_gpu);
    if (staged)
        staged = launch_kernel(&batch_world_keys_kernel_<sources_type_>, dim3 {sources.count, 1u, 1u}, block_shape,
                               stream, sources, u32r2_view_t {world_lengths}, u32r2_view_t {world_sources}, world_keys,
                               shape.max_segments_per_gpu);
    return staged;
}

expected<multimodal_batch_mixer_t> multimodal_batch_mixer_t::make(
    multimem_domain_t &multimem, colocated_domain_t &colocated, std::span<gpu_arena_t> arenas,
    multimodal_batch_shape_t shape, u64_t seed, std::span<gpu_stream_t> streams, f64_t timeout_seconds) noexcept {
    using result_t = expected<multimodal_batch_mixer_t>;
    std::size_t const members = multimem.local_member_count();
    if (members == 0 || arenas.size() != members)
        return result_t::fail(status_t::invalid_argument_k,
                              "every local multimem_domain member needs its arena ({} members, {} arenas)", members,
                              arenas.size());
    if (members > max_gpus_per_host_k)
        return result_t::fail(status_t::unsupported_k, "{} members exceed max_gpus_per_host_k ({})", members,
                              max_gpus_per_host_k);
    if (streams.size() != members)
        return result_t::fail(status_t::invalid_argument_k, "every member needs its stream ({} members, {} streams)",
                              members, streams.size());
    u32_t const domains = colocated.multimem_domain_count();
    // Level-1 precondition: every local member's arena readable by every other — visibility is the
    // arenas' construction-time configuration, validated here, never granted here.
    for (std::size_t member = 0; member != members; ++member)
        for (std::size_t peer = 0; peer != members; ++peer)
            if (!arenas[member].is_visible_to(static_cast<int>(peer)))
                return result_t::fail(status_t::invalid_argument_k,
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
    // Each member's world routing tables, on its own arena: the geometry pass writes its own slice
    // of the sources, the gather pass reads it back, and the world pass fills the rest.
    std::size_t const world_gpus = colocated.topology().shape.gpus_per_colocated_domain;
    std::size_t const segments = shape.max_segments_per_gpu;
    if (mixer.world_lengths_.resize(members) != status_t::success_k ||
        mixer.world_sources_.resize(members) != status_t::success_k ||
        mixer.world_keys_.resize(members) != status_t::success_k)
        return result_t::fail(status_t::bad_alloc_k, "routing table allocation failed ({} members)", members);
    for (std::size_t member = 0; member != members; ++member) {
        switch_to_gpu_t gpu_scope {streams[member]};
        if (!gpu_scope.ok())
            return result_t::fail(status_t::device_error_k, "context entry failed for member {}'s routing tables",
                                  member);
        gpu_arena_alloc<u32_t> const routing_alloc {arenas[member]};
        mixer.world_lengths_[member] = u32r2_tensor_arena_t::uninitialized({world_gpus, segments}, routing_alloc).value;
        mixer.world_sources_[member] = u32r2_tensor_arena_t::uninitialized({world_gpus, segments}, routing_alloc).value;
        mixer.world_keys_[member] =
            u64r2_tensor_arena_t::uninitialized({world_gpus, segments}, gpu_arena_alloc<u64_t> {arenas[member]}).value;
        if (!mixer.world_lengths_[member] || !mixer.world_sources_[member] || !mixer.world_keys_[member])
            return result_t::fail(status_t::bad_alloc_k,
                                  "member {}'s routing tables allocation failed ({} GPUs of {} segments)", member,
                                  world_gpus, segments);
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
            return result_t::fail(status_t::device_error_k, "context entry failed wiring member {}", member);
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
    std::size_t const members = world_sources_.size();
    if (local >= members) return expected_t::fail(status_t::invalid_argument_k, "member {} of {}", local, members);
    if (loaded.size() != members)
        return expected_t::fail(status_t::invalid_argument_k, "every member needs its loaded batch ({} of {})",
                                loaded.size(), members);
    switch_to_gpu_t gpu_scope {stream};
    if (!gpu_scope.ok()) return expected_t::fail(status_t::device_error_k, "context entry failed ({})", local);
    std::size_t const member_index = multimem_->member_offset() + local;

    auto local_source = [&](std::size_t member, batch_source_t_ &source) noexcept -> expected_t {
        source = source_from_loaded_(loaded[member]);
        if (source.tokens.empty() || source.segment_offsets.empty() || source.segment_lengths.empty() ||
            source.segment_modality.empty())
            return expected_t::fail(status_t::invalid_argument_k, "member {}'s loaded batch is unallocated", member);
        if (source.tokens.data() == mixed.tokens.data())
            return expected_t::fail(status_t::invalid_argument_k,
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
            if (packed) packed = copy_device_to_device(gather.payload(), window, stream);
        }
        if (!packed) return packed;
    }

    // The step's first barrier orders "everyone loaded" before any peer read; the round is the
    // member's own, self-advanced inside the barrier kernel.
    if (expected_t arrived = multimem_->barrier(member_index, stream); !arrived) return arrived;

    // The routing destination is this member's world coordinate — the multimem_domains claim
    // disjoint destination ranges in the members·domains world (a single domain's base is 0).
    u32_t const gpu_index_in_colocated_domain = first_gpu_index_in_colocated_domain_ + static_cast<u32_t>(member_index);
    u32r2_span_t const world_lengths = world_lengths_[local].span(), world_sources = world_sources_[local].span();
    u64r2_span_t const world_keys = world_keys_[local].span();
    if (window_.scalars != 0) {
        batch_windows_t_ windows;
        windows.first_window = multimem_->window_bytes(0, window_).data();
        windows.member_stride = multimem_->member(0).replica_stride_words * sizeof(u32_t);
        windows.layout = block_layout;
        windows.members = static_cast<u32_t>(multimem_->member_count());
        windows.count = windows.members * domains_;
        return mix_(windows, loaded[local].dropped.view(), mixed, world_lengths, world_sources, world_keys,
                    gpu_index_in_colocated_domain, data_epoch, seed_, shape_, stream);
    }
    return mix_(sources, loaded[local].dropped.view(), mixed, world_lengths, world_sources, world_keys,
                gpu_index_in_colocated_domain, data_epoch, seed_, shape_, stream);
}

expected_t multimodal_batch_mixer_t::operator()(data_epoch_t data_epoch,
                                                std::span<multimodal_batch_on_gpu_t const> loaded,
                                                std::span<multimodal_batch_on_gpu_t> mixed,
                                                std::span<gpu_stream_t> streams) noexcept {
    std::size_t const members = world_sources_.size();
    if (streams.size() != members || mixed.size() != members)
        return expected_t::fail(status_t::invalid_argument_k,
                                "every member needs its stream and mixed batch ({} members)", members);
    for (std::size_t local = 0; local != members; ++local)
        if (expected_t staged = (*this)(local, data_epoch, loaded, mixed[local], streams[local]); !staged)
            return staged;
    return {};
}
