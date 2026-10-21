/**
 *  @file src/loaders/paired_loader.cu
 *  @author Ash Vardanian
 *  @date October 18, 2026
 *  @brief The Parquet paired loader's bodies and its pairing kernel — see the header for the
 *      contract.
 *
 *  A fill reads the anchor, positive and negative columns of the picked shards into one reservoir,
 *  lands their offsets on the host, and draws every pair there. Each batch then decodes or packs
 *  its positives over the leading rows and its negatives over the next, packs its anchors over the
 *  trailing rows, and one kernel interleaves the role tables into the batch's, tagging each pair's
 *  segments with its bag.
 */
#include <algorithm> // `std::upper_bound`, `std::find`, `std::min`
#include <span>      // `std::as_bytes`, `std::as_writable_bytes`

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

/** The salts of a list positive's element draw and of a negative's; the caption's is unsalted. */
inline constexpr u64_t positive_draw_salt_k = 0x94D049BB133111EBull;
inline constexpr u64_t negative_draw_salt_k = 0xBF58476D1CE4E5B9ull;

/** The role tables' rows: a pair's positive, its anchor and its negative, in segment order. */
inline constexpr u32_t positive_role_k = 0u, anchor_role_k = 1u, negative_role_k = 2u;

/** A segment shape table, @b [segments,2] for images and @b [segments,3] for videos. */
using segment_shapes_span_t = nk::tensor_span<u16_t, 2>;

/** The batch's per-segment tags the pairing kernel writes beside the geometry. */
struct pair_tags_span_t_ {
    modality_span_t modality {};
    u32r1_span_t bag {};
    nk::tensor_span<segment_role_t, 1> role {};
    u32r1_span_t prefix_length {};
    u64r1_span_t key {};
    nk::tensor_span<late_interaction_side_t, 1> side {};
};

/**
 *  @brief Interleave the role tables into the batch's segment table, tagging each segment's bag and
 *      role.
 *
 *  Pair @c i takes its positive, its anchor shifted by @p anchor_begin , and with three roles its
 *  negative shifted by @p negative_begin . All three share bag @c i , so the mixer routes a triplet
 *  whole and the relations read its negative as a repel edge of its anchor. A text segment's prefix
 *  length is the bytes its role's prefix kept, and the segments past the pairs PAD at @p end with
 *  bag @p absent_bag . Keys and sides stay zero and @c none_k until mixtures name them.
 *
 *  @b grid_size grid-stride over the batch's segments; no dynamic shared memory.
 */
static __global__ void pair_segments_kernel_(u32r2_view_t role_offsets, u32r2_view_t role_lengths,
                                             nk::tensor_view<modality_t, 2> role_modalities,
                                             nk::tensor_view<u16_t, 3> media_shapes, u32_t anchor_begin,
                                             u32_t negative_begin, u32_t anchor_prefix_bytes,
                                             u32_t positive_prefix_bytes, u32r1_span_t segment_offsets,
                                             u32r1_span_t segment_lengths, pair_tags_span_t_ tags,
                                             segment_shapes_span_t shapes, u32_t end, u32_t absent_bag) {
    u32_t const segments = static_cast<u32_t>(segment_lengths.extent(0));
    u32_t const roles = static_cast<u32_t>(role_lengths.extent(0)), pairs = static_cast<u32_t>(role_lengths.extent(1));
    u32_t const stride = gridDim.x * blockDim.x;
    for (u32_t segment = blockIdx.x * blockDim.x + threadIdx.x; segment < segments; segment += stride) {
        u32_t const pair = segment / roles, role = segment % roles;
        tags.key[segment] = 0u;
        tags.side[segment] = late_interaction_side_t::none_k;
        if (pair >= pairs) {
            segment_offsets[segment] = end;
            segment_lengths[segment] = 0u;
            tags.modality[segment] = modality_t::text_k;
            tags.bag[segment] = absent_bag;
            tags.role[segment] = segment_role_t::anchor_k;
            tags.prefix_length[segment] = 0u;
        }
        else {
            u32_t const begin = role == anchor_role_k ? anchor_begin : role == negative_role_k ? negative_begin : 0u;
            u32_t const length = role_lengths[role, pair];
            modality_t const modality = role_modalities[role, pair];
            u32_t const prefix_bytes = role == anchor_role_k ? anchor_prefix_bytes : positive_prefix_bytes;
            segment_offsets[segment] = begin + role_offsets[role, pair];
            segment_lengths[segment] = length;
            tags.modality[segment] = modality;
            tags.bag[segment] = pair;
            tags.role[segment] = role == anchor_role_k     ? segment_role_t::anchor_k
                                 : role == negative_role_k ? segment_role_t::negative_k
                                                           : segment_role_t::positive_k;
            tags.prefix_length[segment] = modality == modality_t::text_k ? std::min(prefix_bytes, length) : 0u;
            if (role != anchor_role_k)
                for (std::size_t axis = 0; axis != shapes.extent(1); ++axis)
                    shapes[segment, axis] = media_shapes[role == negative_role_k ? 1u : 0u, pair, axis];
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

/** Land the device values of a @c list<int32> column, @p payload , on the host into @p landed . */
[[nodiscard]] static expected_t land_rows_(b8r1_view_t payload, array<i32_t> &landed, gpu_stream_t &stream) noexcept {
    std::size_t const values = payload.extent(0) / sizeof(i32_t);
    if (landed.resize_uninitialized(values) != status_t::success_k)
        return expected_t::fail(status_t::bad_alloc_k, "row landing failed ({} rows)", values);
    return copy_device_to_host<b8_t>(payload, std::as_writable_bytes(landed.span()), stream);
}

/** A row's run of elements in a column: the first one and their count. */
struct elements_t {
    u64_t first = 0, count = 0;
};

/** Row @p row 's elements of a list column with @p groups , or the row itself of a flat one. */
[[nodiscard]] static elements_t elements_of_(array<u64_t> const &groups, u64_t row) noexcept {
    if (groups.empty()) return {row, 1u};
    return {groups[row], groups[row + 1] - groups[row]};
}

expected<paired_loader_parquet_t> paired_loader_parquet_t::make(
    gpu_direct_storage_session_t &session, sz::string_view_t path_or_glob, u64_t seed, cpu_decode_pool_t &pool,
    paired_loader_parquet_config_t config, modality_t modality, multimodal_batch_shape_t const &shape) noexcept {
    using result_t = expected<paired_loader_parquet_t>;
    if (modality != modality_t::text_k && !shape.contains(modality))
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
    if (sz::failed(loader.anchor_.assign(config.anchor)) || sz::failed(loader.negative_.assign(config.negative)) ||
        sz::failed(loader.anchor_prefix_.assign(config.anchor_prefix)) ||
        sz::failed(loader.positive_prefix_.assign(config.positive_prefix)))
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
        loader.reservoir_.charge(*metadata);
    }
    if (modality == modality_t::text_k) {
        if (sz::failed(loader.positive_.assign(config.positive)))
            return result_t::fail(status_t::bad_alloc_k, "column name staging failed");
        return result_t {std::move(loader)};
    }

    // A media struct, the `{bytes,path}` pair the Hugging Face writers emit, reads its `bytes`
    // leaf.
    sz::string_t leaf;
    if (!format_into(leaf, "{}.bytes", config.positive))
        return result_t::fail(status_t::bad_alloc_k, "column name staging failed");
    bool const nested = static_cast<bool>(
        parquet_file_t::column_metadata(session, loader.absolute_paths_[0], leaf.view()));
    if (sz::failed(loader.positive_.assign(nested ? leaf.view() : config.positive)))
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
    bool const negatives = !negative_.empty(), media = modality_ != modality_t::text_k;
    std::size_t const roles = negatives ? 3u : 2u;
    std::size_t const pairs = std::min(rows, segments) / roles, total_pairs = unroll * pairs;
    if (pairs == 0)
        return expected_t::fail(status_t::shape_unsupported_k, "{} rows and {} segments hold no pair", rows, segments);

    // Each pick's first row in the reservoir, which concatenates the picked shards in pick order.
    std::size_t const shard_count = absolute_paths_.size();
    std::size_t const floor_files = std::min<std::size_t>(2, shard_count);
    std::size_t const total_rows = row_prefix_[shard_count];
    constexpr splitmix64_hash_t hash {};
    array<sz::string_view_t> picked_paths;
    array<std::size_t> picked_shards, picked_bases;
    if (picked_paths.reserve(shard_count) != status_t::success_k ||
        picked_shards.reserve(shard_count) != status_t::success_k ||
        picked_bases.reserve(shard_count) != status_t::success_k)
        return expected_t::fail(status_t::bad_alloc_k, "file pick allocation failed");
    std::size_t pooled_estimate = 0, attempt = 0;
    while ((pooled_estimate < total_pairs || picked_paths.size() < floor_files) && picked_paths.size() < shard_count) {
        std::size_t const position = hash(seed_, training_step, gpu_rank, file_draw_salt_k, attempt) % total_rows;
        ++attempt;
        std::size_t const file = static_cast<std::size_t>(
            std::upper_bound(row_prefix_.begin(), row_prefix_.end(), position) - row_prefix_.begin() - 1);
        if (std::find(picked_shards.begin(), picked_shards.end(), file) != picked_shards.end()) continue;
        if (picked_shards.push_back(file) != status_t::success_k ||
            picked_bases.push_back(pooled_estimate) != status_t::success_k ||
            picked_paths.push_back(absolute_paths_[file].view()) != status_t::success_k)
            return expected_t::fail(status_t::bad_alloc_k, "file pick append failed");
        pooled_estimate += row_prefix_[file + 1] - row_prefix_[file];
    }

    sz::string_view_t const names[3] = {anchor_.view(), positive_.view(), negative_.view()};
    auto reservoir_read = parquet_file_t::make(*storage_, picked_paths.view(), {names, roles}, gpu_stream);
    if (!reservoir_read) return expected_t::fail(std::move(reservoir_read));
    parquet_file_t const reservoir = std::move(reservoir_read.value_);
    std::size_t const pooled_rows = reservoir.rows();
    column_view_t const anchors = reservoir.column(0), positives = reservoir.column(1);
    column_view_t const listed_negatives = negatives ? reservoir.column(2) : column_view_t {};

    switch_to_gpu_t gpu_scope {gpu_stream};
    if (!gpu_scope.ok()) return expected_t::fail(status_t::cuda_error_k, "context entry failed for the loader stream");
    // Text negatives are strings grouped per row; media negatives are rows grouped by list offsets.
    array<u64_t> anchor_offsets, anchor_groups, positive_offsets, positive_groups, negative_offsets, negative_groups;
    array<i32_t> negative_rows;
    expected_t landed = land_offsets_(anchors.offsets, anchor_offsets, gpu_stream);
    if (landed) landed = land_offsets_(positives.offsets, positive_offsets, gpu_stream);
    if (landed && !reservoir.groups(0).empty()) landed = land_offsets_(reservoir.groups(0), anchor_groups, gpu_stream);
    if (landed && !reservoir.groups(1).empty())
        landed = land_offsets_(reservoir.groups(1), positive_groups, gpu_stream);
    if (landed && negatives && media) {
        landed = land_offsets_(listed_negatives.offsets, negative_groups, gpu_stream);
        if (landed) landed = land_rows_(listed_negatives.payload, negative_rows, gpu_stream);
    }
    if (landed && negatives && !media) {
        landed = land_offsets_(listed_negatives.offsets, negative_offsets, gpu_stream);
        if (landed && !reservoir.groups(2).empty())
            landed = land_offsets_(reservoir.groups(2), negative_groups, gpu_stream);
    }
    if (!landed) return landed;
    if (CUresult result = cuStreamSynchronize(gpu_stream.get()); result != CUDA_SUCCESS)
        return cuda_failure(result, "reservoir offsets readback failed");

    // Every pair on the host: a keyed bijection over the pool picks the rows, and a key per draw
    // picks the element of a list. A row missing any role is passed over.
    auto const is_empty = [](array<u64_t> const &offsets, u64_t element) noexcept {
        return offsets[element + 1] == offsets[element];
    };
    u64_t const row_key = hash(seed_, gpu_rank, row_draw_salt_k);
    array<u64_t> drawn;
    if (drawn.resize_uninitialized(roles * total_pairs) != status_t::success_k)
        return expected_t::fail(status_t::bad_alloc_k, "pair draw allocation failed ({} pairs)", total_pairs);
    u64r2_span_t const drawn_roles {drawn.data(), roles, total_pairs};
    std::size_t found = 0;
    for (std::size_t draw = 0; draw != pooled_rows && found != total_pairs; ++draw) {
        u64_t const row = epoch_permutation(row_key, data_epoch_t {training_step}, draw, pooled_rows);
        auto const [anchor_first, anchor_count] = elements_of_(anchor_groups, row);
        auto const [positive_first, positive_count] = elements_of_(positive_groups, row);
        if (anchor_count == 0 || positive_count == 0) continue;
        u64_t const anchor = anchor_first + hash(row_key, training_step, draw) % anchor_count;
        u64_t const positive = positive_first +
                               hash(row_key, training_step, draw, positive_draw_salt_k) % positive_count;
        if (is_empty(anchor_offsets, anchor) || is_empty(positive_offsets, positive)) continue;
        if (negatives) {
            auto const [negative_first, negative_count] = elements_of_(negative_groups, row);
            if (negative_count == 0) continue;
            u64_t negative = negative_first + hash(row_key, training_step, draw, negative_draw_salt_k) % negative_count;
            if (media) {
                // A media negative is a shard-local row of the positive column, rebased here.
                std::size_t const pick = static_cast<std::size_t>(
                    std::upper_bound(picked_bases.begin(), picked_bases.end(), row) - picked_bases.begin() - 1);
                std::size_t const shard = picked_shards[pick];
                std::size_t const shard_rows = row_prefix_[shard + 1] - row_prefix_[shard];
                i32_t const local = negative_rows[negative];
                if (local < 0 || static_cast<std::size_t>(local) >= shard_rows)
                    return expected_t::fail(status_t::invalid_input_k, "{} names negative row {} of a {}-row shard",
                                            picked_paths[pick], local, shard_rows);
                negative = picked_bases[pick] + static_cast<u64_t>(local);
            }
            if (is_empty(media ? positive_offsets : negative_offsets, negative)) continue;
            drawn_roles[negative_role_k, found] = negative;
        }
        drawn_roles[positive_role_k, found] = positive;
        drawn_roles[anchor_role_k, found] = anchor;
        ++found;
    }
    if (found != total_pairs)
        return expected_t::fail(status_t::shape_unsupported_k,
                                "{} pooled rows across {} files hold {} complete pairs, fewer than the {} needed",
                                pooled_rows, picked_paths.size(), found, total_pairs);
    array<u64_t, gpu_alloc<u64_t>> drawn_on_gpu {gpu_alloc<u64_t> {gpu_stream.get()}};
    array<b8_t, gpu_alloc<b8_t>> prefixes {gpu_alloc<b8_t> {gpu_stream.get()}};
    std::size_t const anchor_prefix_bytes = anchor_prefix_.size(), positive_prefix_bytes = positive_prefix_.size();
    if (drawn_on_gpu.resize_uninitialized(drawn.size()) != status_t::success_k ||
        prefixes.resize_uninitialized(std::max<std::size_t>(anchor_prefix_bytes + positive_prefix_bytes, 1u)) !=
            status_t::success_k)
        return expected_t::fail(status_t::bad_alloc_k, "pair index allocation failed ({} pairs)", total_pairs);
    b8r1_span_t const anchor_prefix {prefixes.data(), anchor_prefix_bytes};
    b8r1_span_t const positive_prefix {prefixes.data() + anchor_prefix_bytes, positive_prefix_bytes};
    expected_t uploaded = copy_host_to_device<u64_t>(drawn.view(), drawn_on_gpu.span(), gpu_stream);
    if (uploaded)
        uploaded = copy_host_to_device<b8_t>(std::as_bytes(std::span {anchor_prefix_.data(), anchor_prefix_bytes}),
                                             anchor_prefix, gpu_stream);
    if (uploaded)
        uploaded = copy_host_to_device<b8_t>(std::as_bytes(std::span {positive_prefix_.data(), positive_prefix_bytes}),
                                             positive_prefix, gpu_stream);
    if (!uploaded) return uploaded;
    // The same draws per batch, `[roles, unroll, pairs]`, on the host and on the device.
    nk::tensor_view<u64_t, 3> const drawn_batches {drawn.data(), {roles, unroll, pairs}};
    nk::tensor_view<u64_t, 3> const drawn_on_gpu_batches {drawn_on_gpu.data(), {roles, unroll, pairs}};

    // A media role decodes through the host-landed offsets; a text role packs on the device.
    column_view_t const positive_tape = media ? column_view_t {positives.payload, positive_offsets.view()} : positives;
    column_view_t const negative_tape = media ? positive_tape : listed_negatives;
    u32_t const shape_width = modality_ == modality_t::image_k ? 2u : modality_ == modality_t::video_k ? 3u : 0u;
    for (std::size_t batch = 0; batch != unroll; ++batch) {
        multimodal_batch_on_gpu_t &loaded = batches[batch];
        if (loaded.tokens.extent(0) != rows || loaded.segment_lengths.extent(0) != segments)
            return expected_t::fail(status_t::shape_unsupported_k, "batch {} differs from the first", batch);

        // A role's text rows, walked the way the packing kernel will: cut at the row width, never
        // crossing a row.
        auto const walk = [&](u32_t role, array<u64_t> const &offsets, std::size_t prefix_bytes) noexcept {
            std::size_t cursor = 0;
            for (std::size_t pair = 0; pair != pairs; ++pair) {
                u64_t const element = drawn_batches[role, batch, pair];
                std::size_t const kept = std::min<std::size_t>(prefix_bytes + offsets[element + 1] - offsets[element],
                                                               columns);
                if (kept > columns - cursor % columns) cursor = round_up_to_multiple(cursor, columns);
                cursor += kept;
            }
            return cursor;
        };
        std::size_t const anchor_cells = walk(anchor_role_k, anchor_offsets, anchor_prefix_bytes);
        std::size_t const anchor_rows = divide_round_up(anchor_cells, columns);
        std::size_t positive_rows = rows - anchor_rows, negative_rows_count = 0;
        if (media && negatives) negative_rows_count = positive_rows - positive_rows / 2, positive_rows /= 2;
        if (!media) {
            positive_rows = divide_round_up(walk(positive_role_k, positive_offsets, positive_prefix_bytes), columns);
            if (negatives)
                negative_rows_count = divide_round_up(walk(negative_role_k, negative_offsets, positive_prefix_bytes),
                                                      columns);
            if (positive_rows + negative_rows_count + anchor_rows > rows)
                return expected_t::fail(status_t::shape_unsupported_k, "{} pairs of texts need {} of the {} rows",
                                        pairs, positive_rows + negative_rows_count + anchor_rows, rows);
        }

        // One scratch table row per role: offsets, lengths, modalities and, for images and videos,
        // the positive and negative media shapes.
        array<u32_t, gpu_alloc<u32_t>> offsets {gpu_alloc<u32_t> {gpu_stream.get()}};
        array<u32_t, gpu_alloc<u32_t>> lengths {gpu_alloc<u32_t> {gpu_stream.get()}};
        array<modality_t, gpu_alloc<modality_t>> modalities {gpu_alloc<modality_t> {gpu_stream.get()}};
        array<u16_t, gpu_alloc<u16_t>> media_shapes {gpu_alloc<u16_t> {gpu_stream.get()}};
        if (offsets.resize_uninitialized(roles * (pairs + 1)) != status_t::success_k ||
            lengths.resize_uninitialized(roles * pairs) != status_t::success_k ||
            modalities.resize_uninitialized(roles * pairs) != status_t::success_k ||
            media_shapes.resize_uninitialized(std::max<std::size_t>(2 * pairs * shape_width, 1u)) !=
                status_t::success_k)
            return expected_t::fail(status_t::bad_alloc_k, "pair table allocation failed ({} pairs)", pairs);
        u32r2_span_t const role_offsets {offsets.data(), roles, pairs + 1}, role_lengths {lengths.data(), roles, pairs};
        nk::tensor_span<modality_t, 2> const role_modalities {modalities.data(), roles, pairs};
        nk::tensor_span<u16_t, 3> const media_shape_tables {media_shapes.data(),
                                                            {std::size_t {2}, pairs, std::size_t {shape_width}}};

        // Positives over the leading rows, negatives over the next, anchors over the trailing ones.
        auto const fill_role = [&](u32_t role, std::size_t row_begin, std::size_t row_end, column_view_t tape,
                                   std::size_t shapes_row) noexcept -> expected_t {
            if (!media)
                return pack_tokens(tape, drawn_on_gpu_batches[role, batch, nk::slice], positive_prefix,
                                   loaded.tokens[nk::range(row_begin, row_end), nk::slice],
                                   role_offsets[role, nk::slice], role_lengths[role, nk::slice],
                                   role_modalities[role, nk::slice], gpu_stream);
            return decode(loaded, row_begin, row_end, tape, drawn_batches[role, batch, nk::slice],
                          media_shape_tables[shapes_row, nk::slice], role_offsets[role, nk::slice],
                          role_lengths[role, nk::slice], role_modalities[role, nk::slice]);
        };
        expected_t filled = fill_role(positive_role_k, 0, positive_rows, positive_tape, 0);
        if (filled && negatives)
            filled = fill_role(negative_role_k, positive_rows, positive_rows + negative_rows_count, negative_tape, 1);
        if (filled)
            filled = pack_tokens(anchors, drawn_on_gpu_batches[anchor_role_k, batch, nk::slice], anchor_prefix,
                                 loaded.tokens[nk::range(rows - anchor_rows, rows), nk::slice],
                                 role_offsets[anchor_role_k, nk::slice], role_lengths[anchor_role_k, nk::slice],
                                 role_modalities[anchor_role_k, nk::slice], gpu_stream);
        if (!filled) return filled;

        segment_shapes_span_t const shapes = modality_ == modality_t::image_k   ? loaded.image_shapes.span()
                                             : modality_ == modality_t::video_k ? loaded.video_shapes.span()
                                                                                : segment_shapes_span_t {};
        u32_t const anchor_begin = static_cast<u32_t>((rows - anchor_rows) * columns);
        u32_t const negative_begin = static_cast<u32_t>(positive_rows * columns);
        pair_tags_span_t_ tags;
        tags.modality = loaded.segment_modality.span();
        tags.bag = loaded.segment_bag.span();
        tags.role = loaded.segment_role.span();
        tags.prefix_length = loaded.segment_prefix_length.span();
        tags.key = loaded.segment_key.span();
        tags.side = loaded.segment_side.span();
        if (expected_t paired = launch_kernel(
                &pair_segments_kernel_, dim3 {grid_size_for(segments), 1u, 1u},
                dim3 {kernel_block_size_default_k, 1u, 1u}, gpu_stream, role_offsets, role_lengths, role_modalities,
                media_shape_tables, anchor_begin, negative_begin, static_cast<u32_t>(anchor_prefix_bytes),
                static_cast<u32_t>(positive_prefix_bytes), loaded.segment_offsets.span(), loaded.segment_lengths.span(),
                tags, shapes, static_cast<u32_t>(anchor_begin + anchor_cells), static_cast<u32_t>(pairs));
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
        return expected_t::fail(status_t::invalid_input_k, "the {} column holds no images", positive_.view());
    return fill_(gpu_stream, gpu_rank, training_step, batches,
                 [&](multimodal_batch_on_gpu_t &loaded, std::size_t row_begin, std::size_t row_end, column_view_t tape,
                     u64r1_view_t drawn, segment_shapes_span_t shapes, u32r1_span_t offsets, u32r1_span_t lengths,
                     modality_span_t modalities) noexcept {
                     pixel_patches_span_t const pixels = loaded.images[nk::range(row_begin, row_end), nk::slice];
                     return decode_images(decoder, *pool_, tape, drawn, {&shape_.image, 1}, pixels, shapes, offsets,
                                          lengths, modalities, gpu_stream);
                 });
}

expected_t paired_loader_parquet_t::fill(gpu_stream_t &gpu_stream, nvdec_session_t &decoder, u32_t gpu_rank,
                                         u32_t training_step, std::span<multimodal_batch_on_gpu_t> batches) noexcept {
    if (modality_ != modality_t::video_k)
        return expected_t::fail(status_t::invalid_input_k, "the {} column holds no videos", positive_.view());
    video_batch_shape_t video = shape_.video;
    video.frame_stride = std::max(video.frame_stride, 1u);
    return fill_(gpu_stream, gpu_rank, training_step, batches,
                 [&](multimodal_batch_on_gpu_t &loaded, std::size_t row_begin, std::size_t row_end, column_view_t tape,
                     u64r1_view_t drawn, segment_shapes_span_t shapes, u32r1_span_t offsets, u32r1_span_t lengths,
                     modality_span_t modalities) noexcept {
                     pixel_tubes_span_t const tubes = loaded.videos[nk::range(row_begin, row_end), nk::slice];
                     return decode_videos(decoder, *pool_, tape, drawn, {&video, 1}, tubes, shapes, offsets, lengths,
                                          modalities, gpu_stream);
                 });
}

expected_t paired_loader_parquet_t::fill(gpu_stream_t &gpu_stream, u32_t gpu_rank, u32_t training_step,
                                         std::span<multimodal_batch_on_gpu_t> batches) noexcept {
    if (modality_ != modality_t::audio_k && modality_ != modality_t::text_k)
        return expected_t::fail(status_t::invalid_input_k, "the {} column holds no recordings or texts",
                                positive_.view());
    // A text pair packs its positives in the fill itself and never calls the decode.
    return fill_(
        gpu_stream, gpu_rank, training_step, batches,
        [&](multimodal_batch_on_gpu_t &loaded, std::size_t row_begin, std::size_t row_end, column_view_t tape,
            u64r1_view_t drawn, segment_shapes_span_t, u32r1_span_t offsets, u32r1_span_t lengths,
            modality_span_t modalities) noexcept {
            audio_samples_span_t const samples = loaded.audio_samples[nk::range(row_begin, row_end), nk::slice];
            return decode_audios(*pool_, tape, drawn, {&shape_.audio, 1}, samples, offsets, lengths, modalities,
                                 gpu_stream);
        });
}
