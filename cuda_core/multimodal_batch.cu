/**
 *  @brief The batch's implementation: the media content+coordinate adapters ( @ref patchify_images ,
 *      @ref patchify_videos , @ref token_coords ) and the batch mixer's routing bijection and two
 *      exchange kernels — compiled once into the `uform` library. Everything stays on the GPU: peer
 *      addresses travel as kernel arguments, and every kernel re-derives its routes in-register from
 *      the keyed bijection, so the mixer owns no memory at all; the adapters are equally pure enqueues.
 *  @file cuda_core/multimodal_batch.cu
 *  @author Ash Vardanian
 *  @date July 3, 2026
 */
#include "cuda_core/multimodal_batch.cuh"

namespace uc = unum::uform::core;
using namespace uc;

/** @brief The number of RGB channels every image / video pixel carries. */
inline constexpr u32_t pixel_channels_k = 3u;
/** @brief One over the maximum byte value — the pixel-to-unit-interval scale. */
inline constexpr f32_t inverse_byte_max_k = 1.0f / 255.0f;

/**
 *  @brief Raster and normalize the uniform image run into patch vectors and their coordinates: each tile
 *      is one @c patch_size² patch of one image, read from that image's own @c (patch_rows·patch_size,
 *      patch_cols·patch_size) resolution, laid out @c (channel,row,column) and normalized
 *      @c (byte/255-mean)/std , with its coordinate @c (0,patch_row,patch_col) written once. Each image's
 *      @c (patch_rows,patch_cols) comes from @p image_shapes . One CTA per (image, patch) tile.
 *
 *  __grid_size__ @c [min(images·num_patches,cap),1,1] , persistent stride; no dynamic shared memory.
 */
static __global__ void image_patchify_kernel_(pixel_run_view_t images, image_shape_view_t image_shapes,
                                              patch_span_t patches, i16r2_span_t coords, f32r1_view_t mean,
                                              f32r1_view_t inverse_std, u32_t patch_size, u32_t num_patches) {
    u32_t const patch_pixels = pixel_channels_k * patch_size * patch_size;
    u32_t const spatial = patch_size * patch_size;
    std::size_t const image_run = images.extent(1);
    std::size_t const total_tiles = static_cast<std::size_t>(images.extent(0)) * num_patches;
    pixel_t const *pixels = images.data();
    for (std::size_t tile = blockIdx.x; tile < total_tiles; tile += gridDim.x) {
        u32_t const image = static_cast<u32_t>(tile / num_patches);
        u32_t const patch = static_cast<u32_t>(tile % num_patches);
        u32_t const patch_cols = image_shapes[image, 1];
        u32_t const patch_row = patch / patch_cols, patch_col = patch % patch_cols;
        u32_t const width = patch_cols * patch_size; // the image's own pixel width
        if (threadIdx.x == 0u) {
            coords[tile, 0] = 0;
            coords[tile, 1] = static_cast<i16_t>(patch_row);
            coords[tile, 2] = static_cast<i16_t>(patch_col);
        }
        std::size_t const image_base = static_cast<std::size_t>(image) * image_run;
        for (u32_t index = threadIdx.x; index < patch_pixels; index += blockDim.x) {
            u32_t const channel = index / spatial, local = index % spatial;
            u32_t const row = patch_row * patch_size + local / patch_size;
            u32_t const column = patch_col * patch_size + local % patch_size;
            std::size_t const source = image_base +
                                       (static_cast<std::size_t>(row) * width + column) * pixel_channels_k + channel;
            f32_t const value = (static_cast<f32_t>(pixels[source]) * inverse_byte_max_k - mean[channel]) *
                                inverse_std[channel];
            patches[tile, index] = __float2bfloat16(value);
        }
    }
}

/**
 *  @brief Raster and normalize one video group into tube vectors and their coordinates: each tile is
 *      one @c tube_time×tube_height×tube_width tube of one video, laid out
 *      @c (local_time,channel,row,column) , and its coordinate @c (time_group·tube_time,row_group,
 *      column_group) written once — the temporal axis carries the tube's absolute start frame. One CTA
 *      per (video, tube) tile.
 *
 *  __grid_size__ @c [min(videos·num_tubes,cap),1,1] , persistent stride; no dynamic shared memory.
 */
static __global__ void video_tubify_kernel_(pixel_video_view_t videos, patch_span_t tubes, i16r2_span_t coords,
                                            f32r1_view_t mean, f32r1_view_t inverse_std, u32_t tube_time,
                                            u32_t tube_height, u32_t tube_width, u32_t tubes_per_row,
                                            u32_t tubes_per_column, u32_t num_tubes, u32_t frames, u32_t height,
                                            u32_t width) {
    u32_t const spatial = tube_height * tube_width;
    u32_t const per_time = pixel_channels_k * spatial;
    u32_t const tube_pixels = tube_time * per_time;
    u32_t const groups_per_frame = tubes_per_row * tubes_per_column;
    std::size_t const total_tiles = static_cast<std::size_t>(videos.extent(0)) * num_tubes;
    pixel_t const *pixels = videos.data();
    for (std::size_t tile = blockIdx.x; tile < total_tiles; tile += gridDim.x) {
        u32_t const video = static_cast<u32_t>(tile / num_tubes);
        u32_t const tube = static_cast<u32_t>(tile % num_tubes);
        u32_t const time_group = tube / groups_per_frame, planar = tube % groups_per_frame;
        u32_t const row_group = planar / tubes_per_row, column_group = planar % tubes_per_row;
        if (threadIdx.x == 0u) {
            coords[tile, 0] = static_cast<i16_t>(time_group * tube_time);
            coords[tile, 1] = static_cast<i16_t>(row_group);
            coords[tile, 2] = static_cast<i16_t>(column_group);
        }
        for (u32_t index = threadIdx.x; index < tube_pixels; index += blockDim.x) {
            u32_t const local_time = index / per_time, rem = index % per_time;
            u32_t const channel = rem / spatial, local = rem % spatial;
            u32_t const frame = time_group * tube_time + local_time;
            u32_t const row = row_group * tube_height + local / tube_width;
            u32_t const column = column_group * tube_width + local % tube_width;
            std::size_t const source = (((static_cast<std::size_t>(video) * frames + frame) * height + row) * width +
                                        column) *
                                           pixel_channels_k +
                                       channel;
            f32_t const value = (static_cast<f32_t>(pixels[source]) * inverse_byte_max_k - mean[channel]) *
                                inverse_std[channel];
            tubes[tile, index] = __float2bfloat16(value);
        }
    }
}

expected_t uc::patchify_images(pixel_run_view_t images, image_shape_view_t image_shapes, patch_span_t patches,
                               i16r2_span_t coords, f32r1_view_t pixel_mean, f32r1_view_t pixel_inverse_std,
                               u32_t patch_size, gpu_stream_t &stream) noexcept {
    std::size_t const images_count = images.extent(0);
    if (patch_size == 0 || images_count == 0)
        return expected_t::fail(status_t::shape_unsupported_k, "patchify needs images and a patch_size (got {},{})",
                                images_count, patch_size);
    u32_t const patch_pixels = pixel_channels_k * patch_size * patch_size;
    std::size_t const total_tiles = patches.extent(0);
    if (patches.extent(1) != patch_pixels || total_tiles % images_count != 0)
        return expected_t::fail(status_t::shape_unsupported_k, "patch table must be [{}·num_patches,{}]", images_count,
                                patch_pixels);
    u32_t const num_patches = static_cast<u32_t>(total_tiles / images_count);
    if (image_shapes.extent(0) != images_count || image_shapes.extent(1) != 2)
        return expected_t::fail(status_t::shape_unsupported_k, "image_shapes must be [{},2]", images_count);
    if (coords.extent(0) != total_tiles || coords.extent(1) != 3)
        return expected_t::fail(status_t::shape_unsupported_k, "coordinate table must be [{},3]", total_tiles);
    dim3 const block_shape {kernel_block_size_default_k, 1u, 1u};
    dim3 const grid {grid_size_for(total_tiles, 1u), 1u, 1u};
    return launch_kernel(reinterpret_cast<void const *>(&image_patchify_kernel_), grid, block_shape, stream, images,
                         image_shapes, patches, coords, pixel_mean, pixel_inverse_std, patch_size, num_patches);
}

expected_t uc::patchify_videos(pixel_video_view_t videos, patch_span_t tubes, i16r2_span_t coords,
                               f32r1_view_t pixel_mean, f32r1_view_t pixel_inverse_std, u32_t tube_time,
                               u32_t tube_height, u32_t tube_width, gpu_stream_t &stream) noexcept {
    u32_t const frames = static_cast<u32_t>(videos.extent(1)), height = static_cast<u32_t>(videos.extent(2));
    u32_t const width = static_cast<u32_t>(videos.extent(3)), channels = static_cast<u32_t>(videos.extent(4));
    if (tube_time == 0 || tube_height == 0 || tube_width == 0 || frames % tube_time != 0 || height % tube_height != 0 ||
        width % tube_width != 0 || channels != pixel_channels_k)
        return expected_t::fail(status_t::shape_unsupported_k, "videos [{},{},{},{},{}] disagree with tube [{},{},{}]",
                                videos.extent(0), frames, height, width, channels, tube_time, tube_height, tube_width);
    u32_t const tubes_per_row = width / tube_width, tubes_per_column = height / tube_height;
    u32_t const num_tubes = (frames / tube_time) * tubes_per_row * tubes_per_column;
    std::size_t const total_tiles = static_cast<std::size_t>(videos.extent(0)) * num_tubes;
    u32_t const tube_pixels = tube_time * pixel_channels_k * tube_height * tube_width;
    if (tubes.extent(0) != total_tiles || tubes.extent(1) != tube_pixels)
        return expected_t::fail(status_t::shape_unsupported_k, "tube table must be [{},{}]", total_tiles, tube_pixels);
    if (coords.extent(0) != total_tiles || coords.extent(1) != 3)
        return expected_t::fail(status_t::shape_unsupported_k, "coordinate table must be [{},3]", total_tiles);
    dim3 const block_shape {kernel_block_size_default_k, 1u, 1u};
    dim3 const grid {grid_size_for(total_tiles, 1u), 1u, 1u};
    return launch_kernel(reinterpret_cast<void const *>(&video_tubify_kernel_), grid, block_shape, stream, videos,
                         tubes, coords, pixel_mean, pixel_inverse_std, tube_time, tube_height, tube_width,
                         tubes_per_row, tubes_per_column, num_tubes, frames, height, width);
}

/**
 *  @brief Write one segment's token coordinates @c (position,0,0) : CTA @c s owns segment @c s , reads
 *      its @c (offset,length) , and threads stripe @c [0,length) writing the running position — PAD
 *      segments ( @c length==0 ) write nothing, so uncovered rows stay whatever the caller pre-zeroed.
 *
 *  __grid_size__ @c [segments,1,1] , threads stripe the segment's length; no dynamic shared memory.
 */
static __global__ void token_coords_kernel_(u32r1_view_t segment_offsets, u32r1_view_t segment_lengths,
                                            i16r2_span_t coords) {
    u32_t const segment = blockIdx.x;
    u32_t const length = segment_lengths[segment];
    if (length == 0u) return;
    u32_t const offset = segment_offsets[segment];
    for (u32_t index = threadIdx.x; index < length; index += blockDim.x) {
        coords[offset + index, 0] = static_cast<i16_t>(index);
        coords[offset + index, 1] = 0;
        coords[offset + index, 2] = 0;
    }
}

expected_t uc::token_coords(u32r1_view_t segment_offsets, u32r1_view_t segment_lengths, i16r2_span_t coords,
                            gpu_stream_t &stream) noexcept {
    if (coords.extent(1) != 3)
        return expected_t::fail(status_t::shape_unsupported_k, "coordinate table must be [{},3]", coords.extent(0));
    if (segment_offsets.extent(0) != segment_lengths.extent(0))
        return expected_t::fail(status_t::invalid_input_k,
                                "segment offsets ({}) and lengths ({}) must share their count",
                                segment_offsets.extent(0), segment_lengths.extent(0));
    dim3 const block_shape {kernel_block_size_default_k, 1u, 1u};
    dim3 const grid {static_cast<unsigned>(segment_lengths.extent(0)), 1u, 1u};
    return launch_kernel(reinterpret_cast<void const *>(&token_coords_kernel_), grid, block_shape, stream,
                         segment_offsets, segment_lengths, coords);
}

/** @brief One member's batch addresses — the gather kernels' view of a peer. */
struct batch_source_t_ {
    token_id_t const *tokens = nullptr;
    u32_t const *segment_offsets = nullptr;
    u32_t const *segment_lengths = nullptr;
};

/** @brief Every member's addresses, passed by value as a kernel argument — @ref max_gpus_per_host_k
 *      bounds the table, so no device-resident copy and no staging exist. */
struct batch_sources_t_ {
    batch_source_t_ entries[max_gpus_per_host_k] {};
    u32_t count = 0;
};

/**
 *  @brief Write the mixed batch's geometry: thread 0 walks the destination slots serially — the
 *      repack cursor is a strict scan — deriving each slot's world source through the shared
 *      @ref epoch_permutation (read gather-wise: a destination looks up its source, equivalent to
 *      scattering by the inverse permutation, and identical on every GPU so routing needs no
 *      exchange), reading its length from the peer's table, and jumping the cursor past
 *      any row boundary the segment would cross. A segment that would overflow the destination
 *      rectangle is dropped loudly (zero length) — the packer's budgets make that a bug, not a case.
 *
 *  __grid_size__ @c [1,1,1] , one thread live; no dynamic shared memory.
 */
static __global__ void batch_geometry_kernel_(batch_sources_t_ sources, u32r1_span_t mixed_offsets,
                                              u32r1_span_t mixed_lengths, u32_t member_index, u32_t epoch, u64_t seed,
                                              u32_t segments_per_gpu, u32_t rows_per_gpu, u32_t columns_per_row) {
    if (threadIdx.x != 0u || blockIdx.x != 0u) return;
    std::size_t const world_segments = static_cast<std::size_t>(segments_per_gpu) * sources.count;
    u32_t const capacity = rows_per_gpu * columns_per_row;
    u32_t cursor = 0;
    for (u32_t slot = 0; slot != segments_per_gpu; ++slot) {
        std::size_t const destination = static_cast<std::size_t>(member_index) * segments_per_gpu + slot;
        std::size_t const source = epoch_permutation(seed, epoch_t {epoch}, destination, world_segments);
        batch_source_t_ const &peer = sources.entries[source / segments_per_gpu];
        u32_t columns = peer.segment_lengths[source % segments_per_gpu];
        // A segment never crosses its row: jump the cursor to the next row when it would.
        u32_t const room_in_row = columns_per_row - cursor % columns_per_row;
        if (columns > room_in_row) cursor += room_in_row;
        if (cursor + columns > capacity) {
            printf("uform: mixed batch overflows at slot %u (member %u) — segment dropped\n", slot, member_index);
            columns = 0;
        }
        mixed_offsets[slot] = cursor;
        mixed_lengths[slot] = columns;
        cursor += columns;
    }
    mixed_offsets[segments_per_gpu] = cursor;
}

/**
 *  @brief Gather one routed segment: CTA @c s re-derives route @c s 's world source and copies its
 *      token span from the peer member's loaded batch into this member's mixed batch, threads
 *      striding the columns. The destination geometry is what @ref batch_geometry_kernel_ wrote.
 *
 *  __grid_size__ @c [segments_per_gpu,1,1] , threads stripe columns; no dynamic shared memory.
 */
static __global__ void batch_gather_kernel_(batch_sources_t_ sources, u32r1_view_t mixed_offsets,
                                            u32r1_view_t mixed_lengths, tokens_span_t mixed_tokens, u32_t member_index,
                                            u32_t epoch, u64_t seed, u32_t segments_per_gpu) {
    u32_t const slot = blockIdx.x;
    u32_t const columns = mixed_lengths[slot];
    if (columns == 0u) return;
    std::size_t const world_segments = static_cast<std::size_t>(segments_per_gpu) * sources.count;
    std::size_t const destination = static_cast<std::size_t>(member_index) * segments_per_gpu + slot;
    std::size_t const source = epoch_permutation(seed, epoch_t {epoch}, destination, world_segments);
    batch_source_t_ const &peer = sources.entries[source / segments_per_gpu];
    u32_t const source_begin = peer.segment_offsets[source % segments_per_gpu];
    u32_t const destination_begin = mixed_offsets[slot];
    for (u32_t column = threadIdx.x; column < columns; column += blockDim.x)
        mixed_tokens[destination_begin + column] = peer.tokens[source_begin + column];
}

expected<multimodal_batch_mixer_t> multimodal_batch_mixer_t::make(multimem_domain_t &multimem,
                                                                  colocated_domain_t &colocated,
                                                                  per_gpu_array<gpu_arena_t> &arenas,
                                                                  multimodal_batch_shape_t shape, u64_t seed) noexcept {
    using result_t = expected<multimodal_batch_mixer_t>;
    std::size_t const members = multimem.member_count();
    if (members == 0 || arenas.size() != members)
        return result_t::fail(status_t::invalid_input_k,
                              "every multimem_domain member needs its arena ({} members, {} arenas)", members,
                              arenas.size());
    if (members > max_gpus_per_host_k)
        return result_t::fail(status_t::shape_unsupported_k, "{} members exceed max_gpus_per_host_k ({})", members,
                              max_gpus_per_host_k);
    if (colocated.multimem_domain_count() != 1u)
        return result_t::fail(status_t::shape_unsupported_k,
                              "cross-host_domain mixing lands with the two-node tier ({} multimem_domains)",
                              colocated.multimem_domain_count());
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
    return result_t {std::move(mixer)};
}

expected_t multimodal_batch_mixer_t::operator()(std::size_t member_index, epoch_t epoch,
                                                per_gpu_array<multimodal_batch_on_gpu_t> const &loaded,
                                                multimodal_batch_on_gpu_t &mixed, gpu_stream_t &stream) noexcept {
    std::size_t const members = multimem_->member_count();
    if (member_index >= members)
        return expected_t::fail(status_t::invalid_input_k, "member {} of {}", member_index, members);
    if (loaded.size() != members)
        return expected_t::fail(status_t::invalid_input_k, "every member needs its loaded batch ({} of {})",
                                loaded.size(), members);
    switch_to_gpu_t gpu_scope {stream};
    if (!gpu_scope.ok()) return expected_t::fail(status_t::cuda_error_k, "context entry failed ({})", member_index);

    // The peer table travels as a kernel argument — no device copy, no staging, nothing owned.
    batch_sources_t_ sources;
    sources.count = static_cast<u32_t>(members);
    for (std::size_t member = 0; member != members; ++member) {
        batch_source_t_ source;
        source.tokens = loaded[member].tokens.data();
        source.segment_offsets = loaded[member].segment_offsets.data();
        source.segment_lengths = loaded[member].segment_lengths.data();
        if (source.tokens == nullptr || source.segment_offsets == nullptr || source.segment_lengths == nullptr)
            return expected_t::fail(status_t::invalid_input_k, "member {}'s loaded batch is unallocated", member);
        if (source.tokens == mixed.tokens.data())
            return expected_t::fail(status_t::invalid_input_k,
                                    "loaded and mixed batches must be distinct storage (member={})", member);
        sources.entries[member] = source;
    }

    // The shared phase rides the lead member's slice — see the ordering contract: this consumer's
    // advance/barrier bracket is the step's first, ordering "everyone loaded" before any peer read.
    if (member_index == 0)
        if (expected_t advanced = multimem_->advance(stream); !advanced) return advanced;
    if (expected_t arrived = multimem_->barrier(member_index, stream); !arrived) return arrived;

    u32r1_span_t const mixed_offsets {mixed.segment_offsets.data(), mixed.segment_offsets.extent(0)};
    u32r1_span_t const mixed_lengths {mixed.segment_lengths.data(), mixed.segment_lengths.extent(0)};
    tokens_span_t const mixed_tokens {mixed.tokens.data(),
                                      static_cast<std::size_t>(shape_.max_rows_per_gpu) * shape_.max_columns_per_row};
    dim3 const one_block {1u, 1u, 1u};
    dim3 const block_shape {kernel_block_size_default_k, 1u, 1u};
    dim3 const segment_grid {shape_.max_segments_per_gpu, 1u, 1u};
    expected_t staged = launch_kernel(reinterpret_cast<void const *>(&batch_geometry_kernel_), one_block, block_shape,
                                      stream, sources, mixed_offsets, mixed_lengths, static_cast<u32_t>(member_index),
                                      epoch.index, seed_, shape_.max_segments_per_gpu, shape_.max_rows_per_gpu,
                                      shape_.max_columns_per_row);
    if (staged)
        staged = launch_kernel(reinterpret_cast<void const *>(&batch_gather_kernel_), segment_grid, block_shape, stream,
                               sources, u32r1_view_t {mixed_offsets.data(), mixed_offsets.extent(0)},
                               u32r1_view_t {mixed_lengths.data(), mixed_lengths.extent(0)}, mixed_tokens,
                               static_cast<u32_t>(member_index), epoch.index, seed_, shape_.max_segments_per_gpu);
    return staged;
}

expected_t multimodal_batch_mixer_t::operator()(epoch_t epoch, per_gpu_array<multimodal_batch_on_gpu_t> const &loaded,
                                                per_gpu_array<multimodal_batch_on_gpu_t> &mixed,
                                                per_gpu_array<gpu_stream_t> &streams) noexcept {
    std::size_t const members = multimem_->member_count();
    if (streams.size() != members || mixed.size() != members)
        return expected_t::fail(status_t::invalid_input_k, "every member needs its stream and mixed batch ({} members)",
                                members);
    for (std::size_t member_index = 0; member_index != members; ++member_index)
        if (expected_t staged = (*this)(member_index, epoch, loaded, mixed[member_index], streams[member_index]);
            !staged)
            return staged;
    return {};
}
