/**
 *  @file src/loaders/paired_loader.cu
 *  @author Ash Vardanian
 *  @date October 18, 2026
 *  @brief The Parquet paired loader's bodies and its pairing kernel — see the header for the
 *      contract.
 *
 *  A fill reads both columns of the picked shards into one reservoir, lands their offsets on the
 *  host, and draws every pair there. Each batch then decodes its media into a scratch segment table
 *  over the leading rows, packs its captions into another over the trailing rows, and one kernel
 *  interleaves the two tables into the batch's, tagging each pair's two segments with its bag.
 */
#include <algorithm> // `std::upper_bound`, `std::find`, `std::min`

#include <cuda.h> // `cuStreamSynchronize`

#include "loaders/paired_loader.cuh"

namespace uc = unum::uform::core;
namespace ul = unum::uform::loaders;
using namespace uc;
using namespace ul;

/** Purpose salts folded through @ref splitmix64_hash_t so the file draw and the row draw stay
 *  decorrelated under one @c (seed,step,rank) — the same salts the single-modality loaders use. */
inline constexpr u64_t file_draw_salt_k = 0x9E3779B97F4A7C15ull;
inline constexpr u64_t row_draw_salt_k = 0xD1B54A32D192ED03ull;

/** A segment shape table, @b [segments,2] for images and @b [segments,3] for videos. */
using segment_shapes_span_t = nk::tensor_span<u16_t, 2>;

/**
 *  @brief Interleave the media and caption tables into the batch's segment table: pair @c i 's
 *      media at segment @c 2i , its caption at @c 2i+1 shifted by @p text_begin , both in bag
 *      @c i ; the segments past the pairs PAD at @p end with the bag @p absent_bag .
 *
 *  @b grid_size grid-stride over the batch's segments; no dynamic shared memory.
 */
static __global__ void pair_segments_kernel_(u32r1_view_t media_offsets, u32r1_view_t media_lengths,
                                             modality_view_t media_modality, nk::tensor_view<u16_t, 2> media_shapes,
                                             u32r1_view_t text_offsets, u32r1_view_t text_lengths, u32_t text_begin,
                                             u32r1_span_t segment_offsets, u32r1_span_t segment_lengths,
                                             modality_span_t segment_modality, u32r1_span_t segment_bag,
                                             segment_shapes_span_t shapes, u32_t end, u32_t absent_bag) {
    u32_t const segments = static_cast<u32_t>(segment_lengths.extent(0));
    u32_t const pairs = static_cast<u32_t>(media_lengths.extent(0));
    u32_t const stride = gridDim.x * blockDim.x;
    for (u32_t segment = blockIdx.x * blockDim.x + threadIdx.x; segment < segments; segment += stride) {
        u32_t const pair = segment / 2u;
        if (pair >= pairs) {
            segment_offsets[segment] = end;
            segment_lengths[segment] = 0u;
            segment_modality[segment] = modality_t::text_k;
            segment_bag[segment] = absent_bag;
        }
        else if (segment % 2u == 0u) {
            segment_offsets[segment] = media_offsets[pair];
            segment_lengths[segment] = media_lengths[pair];
            segment_modality[segment] = media_modality[pair];
            segment_bag[segment] = pair;
            for (std::size_t axis = 0; axis != shapes.extent(1); ++axis)
                shapes[segment, axis] = media_shapes[pair, axis];
        }
        else {
            segment_offsets[segment] = text_begin + text_offsets[pair];
            segment_lengths[segment] = text_lengths[pair];
            segment_modality[segment] = modality_t::text_k;
            segment_bag[segment] = pair;
        }
        if (segment == 0u) segment_offsets[segments] = end;
    }
}

/** Land the device offsets @p offsets on the host into @p landed . */
[[nodiscard]] static expected_t land_offsets_(u64r1_view_t offsets, array<u64_t> &landed,
                                              gpu_stream_t &stream) noexcept {
    if (landed.resize_uninitialized(offsets.extent(0)) != status_t::success_k)
        return expected_t::fail(status_t::bad_alloc_k, "offset landing failed ({} entries)", offsets.extent(0));
    return copy_device_to_host(offsets, landed.span(), stream);
}

expected<paired_loader_parquet_t> paired_loader_parquet_t::make(
    gpu_direct_storage_session_t &session, sz::string_view_t path_or_glob, u64_t seed, cpu_decode_pool_t &pool,
    paired_loader_parquet_config_t config, modality_t modality, multimodal_batch_shape_t const &shape) noexcept {
    using result_t = expected<paired_loader_parquet_t>;
    if (modality == modality_t::text_k || !shape.contains(modality))
        return result_t::fail(status_t::invalid_input_k, "the run carries no slice for the media of {}", path_or_glob);
    auto paths = expand_glob(path_or_glob);
    if (!paths) return result_t::fail(std::move(paths));

    paired_loader_parquet_t loader;
    loader.storage_ = &session;
    loader.pool_ = &pool;
    loader.absolute_paths_ = std::move(paths.value_);
    loader.seed_ = seed;
    loader.modality_ = modality;
    loader.shape_ = shape;
    if (sz::failed(loader.anchor_.assign(config.anchor)))
        return result_t::fail(status_t::bad_alloc_k, "column name staging failed");

    std::size_t const shards = loader.absolute_paths_.size();
    if (loader.row_prefix_.resize(shards + 1) != status_t::success_k)
        return result_t::fail(status_t::bad_alloc_k, "shard prefix allocation failed ({} shards)", shards);
    loader.row_prefix_[0] = 0;
    for (std::size_t index = 0; index != shards; ++index) {
        auto metadata = parquet_file_t::table_metadata(session, loader.absolute_paths_[index]);
        if (!metadata) return result_t::fail(std::move(metadata));
        if (metadata->rows == 0)
            return result_t::fail(status_t::invalid_input_k, "shard {} is empty", loader.absolute_paths_[index]);
        loader.row_prefix_[index + 1] = loader.row_prefix_[index] + metadata->rows;
    }

    // A media struct, the `{bytes,path}` pair the Hugging Face writers emit, reads its `bytes`
    // leaf.
    sz::string_t leaf;
    if (!format_into(leaf, "{}.bytes", config.media))
        return result_t::fail(status_t::bad_alloc_k, "column name staging failed");
    bool const nested = static_cast<bool>(
        parquet_file_t::column_metadata(session, loader.absolute_paths_[0], leaf.view()));
    if (sz::failed(loader.media_.assign(nested ? leaf.view() : config.media)))
        return result_t::fail(status_t::bad_alloc_k, "column name staging failed");
    return result_t {std::move(loader)};
}

template <typename decode_type_>
expected_t paired_loader_parquet_t::fill_(gpu_stream_t &gpu_stream, u32_t gpu_rank, u32_t training_step,
                                          std::span<multimodal_batch_on_gpu_t> batches,
                                          decode_type_ &&decode) noexcept {
    if (batches.empty()) return {};
    std::size_t const unroll = batches.size();
    std::size_t const rows = batches[0].tokens.extent(0), columns = batches[0].tokens.extent(1);
    std::size_t const segments = batches[0].segment_lengths.extent(0);
    std::size_t const pairs = std::min(rows, segments) / 2u, total_pairs = unroll * pairs;
    if (pairs == 0)
        return expected_t::fail(status_t::shape_unsupported_k, "{} rows and {} segments hold no pair", rows, segments);

    std::size_t const shard_count = absolute_paths_.size();
    std::size_t const floor_files = std::min<std::size_t>(2, shard_count);
    std::size_t const total_rows = row_prefix_[shard_count];
    constexpr splitmix64_hash_t hash {};
    array<sz::string_view_t> picked_paths;
    array<std::size_t> picked_shards;
    if (picked_paths.reserve(shard_count) != status_t::success_k ||
        picked_shards.reserve(shard_count) != status_t::success_k)
        return expected_t::fail(status_t::bad_alloc_k, "file pick allocation failed");
    std::size_t pooled_estimate = 0, attempt = 0;
    while ((pooled_estimate < total_pairs || picked_paths.size() < floor_files) && picked_paths.size() < shard_count) {
        std::size_t const position = hash(seed_, training_step, gpu_rank, file_draw_salt_k, attempt) % total_rows;
        ++attempt;
        std::size_t const file = static_cast<std::size_t>(
            std::upper_bound(row_prefix_.begin(), row_prefix_.end(), position) - row_prefix_.begin() - 1);
        if (std::find(picked_shards.begin(), picked_shards.end(), file) != picked_shards.end()) continue;
        if (picked_shards.push_back(file) != status_t::success_k ||
            picked_paths.push_back(absolute_paths_[file].view()) != status_t::success_k)
            return expected_t::fail(status_t::bad_alloc_k, "file pick append failed");
        pooled_estimate += row_prefix_[file + 1] - row_prefix_[file];
    }

    sz::string_view_t const names[2] = {anchor_.view(), media_.view()};
    auto reservoir_read = parquet_file_t::make(*storage_, picked_paths.view(), {names, 2}, gpu_stream);
    if (!reservoir_read) return expected_t::fail(std::move(reservoir_read));
    parquet_file_t const reservoir = std::move(reservoir_read.value_);
    std::size_t const pooled_rows = reservoir.rows();
    column_view_t const captions = reservoir.column(0), media = reservoir.column(1);
    std::span<u64_t const> const caption_groups = reservoir.groups(0);

    switch_to_gpu_t gpu_scope {gpu_stream};
    if (!gpu_scope.ok()) return expected_t::fail(status_t::cuda_error_k, "context entry failed for the loader stream");
    array<u64_t> caption_offsets, groups, media_offsets;
    expected_t landed = land_offsets_(captions.offsets, caption_offsets, gpu_stream);
    if (landed) landed = land_offsets_(media.offsets, media_offsets, gpu_stream);
    if (landed && !caption_groups.empty()) landed = land_offsets_(caption_groups, groups, gpu_stream);
    if (!landed) return landed;
    if (CUresult result = cuStreamSynchronize(gpu_stream.get()); result != CUDA_SUCCESS)
        return cuda_failure(result, "reservoir offsets readback failed");

    // Every pair on the host: a keyed bijection over the pool picks the rows, and a key per draw
    // picks the caption of a row with several. A row missing either half is passed over.
    u64_t const row_key = hash(seed_, gpu_rank, row_draw_salt_k);
    array<u64_t> drawn_rows, drawn_captions;
    if (drawn_rows.resize_uninitialized(total_pairs) != status_t::success_k ||
        drawn_captions.resize_uninitialized(total_pairs) != status_t::success_k)
        return expected_t::fail(status_t::bad_alloc_k, "pair draw allocation failed ({} pairs)", total_pairs);
    std::size_t found = 0;
    for (std::size_t draw = 0; draw != pooled_rows && found != total_pairs; ++draw) {
        u64_t const row = epoch_permutation(row_key, data_epoch_t {training_step}, draw, pooled_rows);
        u64_t const first = groups.size() ? groups[row] : row;
        u64_t const count = groups.size() ? groups[row + 1] - groups[row] : 1u;
        if (count == 0 || media_offsets[row + 1] == media_offsets[row]) continue;
        u64_t const caption = first + hash(row_key, training_step, draw) % count;
        if (caption_offsets[caption + 1] == caption_offsets[caption]) continue;
        drawn_rows[found] = row;
        drawn_captions[found] = caption;
        ++found;
    }
    if (found != total_pairs)
        return expected_t::fail(status_t::shape_unsupported_k,
                                "{} pooled rows across {} files hold {} complete pairs, fewer than the {} needed",
                                pooled_rows, picked_paths.size(), found, total_pairs);
    array<u64_t, gpu_alloc<u64_t>> caption_elements {gpu_alloc<u64_t> {gpu_stream.get()}};
    if (caption_elements.resize_uninitialized(total_pairs) != status_t::success_k)
        return expected_t::fail(status_t::bad_alloc_k, "caption index allocation failed ({} pairs)", total_pairs);
    if (expected_t uploaded = copy_host_to_device(u64r1_view_t {drawn_captions.data(), total_pairs},
                                                  u64r1_span_t {caption_elements.data(), total_pairs}, gpu_stream);
        !uploaded)
        return uploaded;

    column_view_t const media_tape {media.payload, media_offsets.view()};
    u32_t const shape_width = modality_ == modality_t::image_k ? 2u : modality_ == modality_t::video_k ? 3u : 0u;
    for (std::size_t batch = 0; batch != unroll; ++batch) {
        multimodal_batch_on_gpu_t &loaded = batches[batch];
        if (loaded.tokens.extent(0) != rows || loaded.segment_lengths.extent(0) != segments)
            return expected_t::fail(status_t::shape_unsupported_k, "batch {} differs from the first", batch);
        std::size_t const first_pair = batch * pairs;

        // The captions' rows, walked the way the packing kernel will: cut at the row width, never
        // crossing a row.
        std::size_t cursor = 0;
        for (std::size_t pair = 0; pair != pairs; ++pair) {
            u64_t const caption = drawn_captions[first_pair + pair];
            std::size_t const kept = std::min<std::size_t>(caption_offsets[caption + 1] - caption_offsets[caption],
                                                           columns);
            if (kept > columns - cursor % columns) cursor = round_up_to_multiple(cursor, columns);
            cursor += kept;
        }
        std::size_t const text_rows = divide_round_up(cursor, columns), media_rows = rows - text_rows;

        // One scratch table per half: offsets, lengths, modalities and, for images and videos,
        // shapes.
        array<u32_t, gpu_alloc<u32_t>> tables {gpu_alloc<u32_t> {gpu_stream.get()}};
        array<modality_t, gpu_alloc<modality_t>> modalities {gpu_alloc<modality_t> {gpu_stream.get()}};
        array<u16_t, gpu_alloc<u16_t>> media_shapes {gpu_alloc<u16_t> {gpu_stream.get()}};
        if (tables.resize_uninitialized(4 * pairs + 2) != status_t::success_k ||
            modalities.resize_uninitialized(2 * pairs) != status_t::success_k ||
            media_shapes.resize_uninitialized(std::max<std::size_t>(pairs * shape_width, 1u)) != status_t::success_k)
            return expected_t::fail(status_t::bad_alloc_k, "pair table allocation failed ({} pairs)", pairs);
        u32r1_span_t const media_offsets_table {tables.data(), pairs + 1};
        u32r1_span_t const media_lengths_table {tables.data() + pairs + 1, pairs};
        u32r1_span_t const text_offsets_table {tables.data() + 2 * pairs + 1, pairs + 1};
        u32r1_span_t const text_lengths_table {tables.data() + 3 * pairs + 2, pairs};
        modality_span_t const media_modality {modalities.data(), pairs};
        modality_span_t const text_modality {modalities.data() + pairs, pairs};

        std::span<u64_t const> const drawn {drawn_rows.data() + first_pair, pairs};
        segment_shapes_span_t const media_shape_rows {media_shapes.data(), {pairs, std::size_t {shape_width}}};
        if (expected_t decoded = decode(loaded, media_rows, media_tape, drawn, media_shape_rows, media_offsets_table,
                                        media_lengths_table, media_modality);
            !decoded)
            return decoded;
        tokens_matrix_span_t const caption_rows {loaded.tokens.data() + media_rows * columns, {text_rows, columns}};
        if (expected_t packed = pack_tokens(captions, u64r1_view_t {caption_elements.data() + first_pair, pairs},
                                            caption_rows, text_offsets_table, text_lengths_table, text_modality,
                                            gpu_stream);
            !packed)
            return packed;

        segment_shapes_span_t const shapes = modality_ == modality_t::image_k   ? loaded.image_shapes.span()
                                             : modality_ == modality_t::video_k ? loaded.video_shapes.span()
                                                                                : segment_shapes_span_t {};
        u32_t const text_begin = static_cast<u32_t>(media_rows * columns);
        if (expected_t paired = launch_kernel(
                &pair_segments_kernel_, dim3 {grid_size_for(segments), 1u, 1u},
                dim3 {kernel_block_size_default_k, 1u, 1u}, gpu_stream,
                u32r1_view_t {media_offsets_table.data(), pairs + 1}, u32r1_view_t {media_lengths_table.data(), pairs},
                modality_view_t {media_modality.data(), pairs}, nk::tensor_view<u16_t, 2> {media_shape_rows},
                u32r1_view_t {text_offsets_table.data(), pairs + 1}, u32r1_view_t {text_lengths_table.data(), pairs},
                text_begin, loaded.segment_offsets.span(), loaded.segment_lengths.span(),
                loaded.segment_modality.span(), loaded.segment_bag.span(), shapes,
                static_cast<u32_t>(text_begin + cursor), static_cast<u32_t>(pairs));
            !paired)
            return paired;
    }
    // The reservoir and the scratch tables free with this scope, after the stream retires every
    // read of them.
    if (CUresult result = cuStreamSynchronize(gpu_stream.get()); result != CUDA_SUCCESS)
        return cuda_failure(result, "paired fill drain failed");
    return {};
}

expected_t paired_loader_parquet_t::fill(gpu_stream_t &gpu_stream, nvjpeg_session_t &decoder, u32_t gpu_rank,
                                         u32_t training_step, std::span<multimodal_batch_on_gpu_t> batches) noexcept {
    if (modality_ != modality_t::image_k)
        return expected_t::fail(status_t::invalid_input_k, "the {} column holds no images", media_.view());
    return fill_(gpu_stream, gpu_rank, training_step, batches,
                 [&](multimodal_batch_on_gpu_t &loaded, std::size_t media_rows, column_view_t tape,
                     std::span<u64_t const> drawn, segment_shapes_span_t shapes, u32r1_span_t offsets,
                     u32r1_span_t lengths, modality_span_t modalities) noexcept {
                     pixel_patches_span_t const pixels {loaded.images.data(),
                                                        {media_rows, loaded.images.extent(1), loaded.images.extent(2),
                                                         loaded.images.extent(3), loaded.images.extent(4)}};
                     return decode_images(decoder, *pool_, tape, drawn, {&shape_.image, 1}, pixels, shapes, offsets,
                                          lengths, modalities, gpu_stream);
                 });
}

expected_t paired_loader_parquet_t::fill(gpu_stream_t &gpu_stream, nvdec_session_t &decoder, u32_t gpu_rank,
                                         u32_t training_step, std::span<multimodal_batch_on_gpu_t> batches) noexcept {
    if (modality_ != modality_t::video_k)
        return expected_t::fail(status_t::invalid_input_k, "the {} column holds no videos", media_.view());
    video_batch_shape_t video = shape_.video;
    video.frame_stride = std::max(video.frame_stride, 1u);
    return fill_(gpu_stream, gpu_rank, training_step, batches,
                 [&](multimodal_batch_on_gpu_t &loaded, std::size_t media_rows, column_view_t tape,
                     std::span<u64_t const> drawn, segment_shapes_span_t shapes, u32r1_span_t offsets,
                     u32r1_span_t lengths, modality_span_t modalities) noexcept {
                     pixel_tubes_span_t const tubes {
                         loaded.videos.data(),
                         {media_rows, loaded.videos.extent(1), loaded.videos.extent(2), loaded.videos.extent(3),
                          loaded.videos.extent(4), loaded.videos.extent(5)}};
                     return decode_videos(decoder, *pool_, tape, drawn, {&video, 1}, tubes, shapes, offsets, lengths,
                                          modalities, gpu_stream);
                 });
}

expected_t paired_loader_parquet_t::fill(gpu_stream_t &gpu_stream, u32_t gpu_rank, u32_t training_step,
                                         std::span<multimodal_batch_on_gpu_t> batches) noexcept {
    if (modality_ != modality_t::audio_k)
        return expected_t::fail(status_t::invalid_input_k, "the {} column holds no recordings", media_.view());
    return fill_(
        gpu_stream, gpu_rank, training_step, batches,
        [&](multimodal_batch_on_gpu_t &loaded, std::size_t media_rows, column_view_t tape, std::span<u64_t const> drawn,
            segment_shapes_span_t, u32r1_span_t offsets, u32r1_span_t lengths, modality_span_t modalities) noexcept {
            audio_samples_span_t const samples {
                loaded.audio_samples.data(),
                {media_rows, loaded.audio_samples.extent(1), loaded.audio_samples.extent(2)}};
            return decode_audios(*pool_, tape, drawn, {&shape_.audio, 1}, samples, offsets, lengths, modalities,
                                 gpu_stream);
        });
}
