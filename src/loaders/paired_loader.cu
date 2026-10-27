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
#include <cmath> // `std::ceil`, the pairs a fill's rows hold

#include <algorithm> // `std::upper_bound`, `std::lower_bound`, `std::min`
#include <span>      // `std::as_bytes`, `std::as_writable_bytes`

#include <cuda.h> // `cuStreamSynchronize`

#include "uform/loaders/paired_loader.cuh"

namespace ue = unum::uform::encoder;
namespace ul = unum::uform::loaders;
using namespace ue;
using namespace unum::uform::utilities;
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

/** The sidecar column naming each row's media, the xxh64 of its original bytes. */
inline constexpr char sample_key_column_k[] = "key";

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
 *  whole and the relations read its negative as a repel edge of its anchor. The anchor and the
 *  positive carry the pair's sample key from @p pair_keys , the negative none. A text segment's
 *  prefix length is the bytes its role's prefix kept, and the segments past the first @p pairs PAD
 *  at @p end with bag @p pairs . Sides stay @c none_k until mixtures name them.
 *
 *  @b grid_size grid-stride over the batch's segments; no dynamic shared memory.
 */
static __global__ void pair_segments_kernel_(
    u32r2_view_t role_offsets, u32r2_view_t role_lengths, nk::tensor_view<modality_t, 2> role_modalities,
    nk::tensor_view<u16_t, 3> media_shapes, u32_t anchor_begin, u32_t negative_begin, u32_t anchor_prefix_bytes,
    u32_t positive_prefix_bytes, u64r1_view_t pair_keys, role_sides_t sides, u32r1_span_t segment_offsets,
    u32r1_span_t segment_lengths, pair_tags_span_t_ tags, segment_shapes_span_t shapes, u32_t end, u32_t pairs) {
    u32_t const segments = static_cast<u32_t>(segment_lengths.extent(0));
    u32_t const roles = static_cast<u32_t>(role_lengths.extent(0));
    u32_t const stride = gridDim.x * blockDim.x;
    for (u32_t segment = blockIdx.x * blockDim.x + threadIdx.x; segment < segments; segment += stride) {
        u32_t const pair = segment / roles, role = segment % roles;
        tags.key[segment] = 0u;
        tags.side[segment] = late_interaction_side_t::none_k;
        if (pair >= pairs) {
            segment_offsets[segment] = end;
            segment_lengths[segment] = 0u;
            tags.modality[segment] = modality_t::text_k;
            tags.bag[segment] = pairs;
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
            if (role != negative_role_k) tags.key[segment] = pair_keys[pair];
            tags.side[segment] = role == anchor_role_k     ? sides.anchor
                                 : role == negative_role_k ? sides.negative
                                                           : sides.positive;
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
[[nodiscard]] static expected_t land_offsets_(u64r1_view_t offsets, st::basic_vector<u64_t> &landed,
                                              gpu_stream_t &stream) noexcept {
    if (st::failed(landed.resize(st::assume_uninitialized, offsets.extent(0))))
        return expected_t::fail(status_t::bad_alloc_k, "offset landing failed ({} entries)", offsets.extent(0));
    return copy_device_to_host(offsets, std::span<u64_t> {landed}, stream);
}

/** Land the device bytes @p payload on the host into @p landed . */
[[nodiscard]] static expected_t land_bytes_(b8r1_view_t payload, st::basic_vector<char> &landed,
                                            gpu_stream_t &stream) noexcept {
    if (st::failed(landed.resize(st::assume_uninitialized, payload.extent(0))))
        return expected_t::fail(status_t::bad_alloc_k, "byte landing failed ({} bytes)", payload.extent(0));
    return copy_device_to_host<b8_t>(payload, std::as_writable_bytes(std::span<char> {landed}), stream);
}

/** Land the device values of a @c list<int32> column, @p payload , on the host into @p landed . */
[[nodiscard]] static expected_t land_rows_(b8r1_view_t payload, st::basic_vector<i32_t> &landed,
                                           gpu_stream_t &stream) noexcept {
    std::size_t const values = payload.extent(0) / sizeof(i32_t);
    if (st::failed(landed.resize(st::assume_uninitialized, values)))
        return expected_t::fail(status_t::bad_alloc_k, "row landing failed ({} rows)", values);
    return copy_device_to_host<b8_t>(payload, std::as_writable_bytes(std::span<i32_t> {landed}), stream);
}

/** A row's run of elements in a column: the first one and their count. */
struct elements_t {
    u64_t first = 0, count = 0;
};

/** Row @p row 's elements of a list column with @p groups , or the row itself of a flat one. */
[[nodiscard]] static elements_t elements_of_(st::basic_vector<u64_t> const &groups, u64_t row) noexcept {
    if (groups.empty()) return {row, 1u};
    return {groups[row], groups[row + 1] - groups[row]};
}

expected<paired_loader_parquet_t> paired_loader_parquet_t::make(gpu_direct_storage_session_t &session,
                                                                sz::string_view_t path_or_glob, u64_t seed,
                                                                pool_view_t pool, paired_loader_parquet_config_t config,
                                                                modality_t modality,
                                                                multimodal_batch_shape_t const &shape) noexcept {
    using result_t = expected<paired_loader_parquet_t>;
    if (modality != modality_t::text_k && !shape.contains(modality))
        return result_t::fail(status_t::invalid_argument_k, "the run carries no slice for the media of {}",
                              path_or_glob);
    auto paths = expand_glob(path_or_glob);
    if (!paths) return result_t::fail(std::move(paths));

    paired_loader_parquet_t loader;
    loader.storage_ = &session;
    loader.pool_ = pool;
    loader.seed_ = seed;
    loader.modality_ = modality;
    loader.shape_ = shape;
    loader.sides_ = config.sides;
    if (sz::failed(loader.anchor_.assign(config.anchor)) || sz::failed(loader.negative_.assign(config.negative)) ||
        sz::failed(loader.anchor_prefix_.assign(config.anchor_prefix)) ||
        sz::failed(loader.positive_prefix_.assign(config.positive_prefix)))
        return result_t::fail(status_t::bad_alloc_k, "column name staging failed");

    // A media positive lives in the sibling tar, so only the text roles must fill a row.
    sz::string_view_t const roles[3] = {config.anchor, config.negative,
                                        modality == modality_t::text_k ? config.positive : sz::string_view_t {}};
    if (st::failed(loader.shard_starts_.push_back(0)))
        return result_t::fail(status_t::bad_alloc_k, "shard table allocation failed");
    std::size_t groups = 0;
    for (sz::string_t const &path : paths.value_) {
        auto metadata = parquet_file_t::table_metadata(session, path, std::span<sz::string_view_t const> {roles});
        if (!metadata) return result_t::fail(std::move(metadata));
        if (metadata->rows == 0) return result_t::fail(status_t::invalid_argument_k, "shard {} is empty", path);
        std::size_t const rows_before = loader.rows();
        if (expected_t appended = loader.groups_.append_shard(path, *metadata); !appended)
            return result_t::fail(std::move(appended));
        groups += metadata->group_rows.size();
        loader.reservoir_.charge(*metadata);
        if (loader.rows() == rows_before) continue; // the draw leaves the shard out, and so do its tables
        std::size_t first_row = 0;
        for (std::size_t const group_rows : metadata->group_rows) {
            if (st::failed(loader.group_starts_.push_back(std::size_t {first_row})))
                return result_t::fail(status_t::bad_alloc_k, "row-group table append failed");
            first_row += group_rows;
        }
        if (st::failed(loader.group_starts_.push_back(std::size_t {first_row})) ||
            st::failed(loader.shard_starts_.push_back(loader.group_starts_.size())) ||
            st::failed(loader.shard_keys_.push_back(static_cast<u64_t>(path.hash()))))
            return result_t::fail(status_t::bad_alloc_k, "shard table append failed");
    }
    if (loader.rows() == 0)
        return result_t::fail(status_t::invalid_argument_k,
                              "{} holds no row with every role column: anchor {}, negative {}", path_or_glob,
                              config.anchor, config.negative.empty() ? sz::string_view_t {"none"} : config.negative);
    if (expected_t reserved = loader.groups_.reserve(groups); !reserved) return result_t::fail(std::move(reserved));
    if (parquet_file_t::column_metadata(session, paths.value_[0], sample_key_column_k) &&
        sz::failed(loader.key_.assign(sample_key_column_k)))
        return result_t::fail(status_t::bad_alloc_k, "column name staging failed");
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
    bool const nested = static_cast<bool>(parquet_file_t::column_metadata(session, paths.value_[0], leaf.view()));
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
    // Draw as many pairs as the segments and this loader's rows hold, never more than the reservoir
    // was charged for; each batch admits the keyed prefix whose rows fit.
    std::size_t const row_pairs = static_cast<std::size_t>(
        std::ceil(static_cast<f64_t>(rows) * pairs_per_row(columns)));
    std::size_t const pairs = std::min(segments / roles, row_pairs), total_pairs = unroll * pairs;
    if (pairs == 0) return expected_t::fail(status_t::unsupported_k, "{} segments hold no pair", segments);

    if (expected_t picked = groups_.draw(seed_, training_step, gpu_rank, file_draw_salt_k, total_pairs, 2); !picked)
        return picked;
    constexpr splitmix64_hash_t hash {};
    sz::strings_view_t const picked_paths = groups_.picked_paths();
    std::span<u32_t const> const picked_shards = groups_.picked_shards();
    tape_view<u32_t const> const picked_groups = groups_.picked_groups();
    // A drawable shard's group starts, its row count at the back.
    auto const starts_of = [&](std::size_t shard) noexcept {
        return std::span<std::size_t const> {group_starts_}.subspan(shard_starts_[shard],
                                                                    shard_starts_[shard + 1] - shard_starts_[shard]);
    };

    // Each read group's first reservoir row: the reservoir holds the picks in order and each pick's
    // groups ascending, so the reads index the picked tape's elements.
    st::basic_vector<std::size_t> read_bases;
    if (st::failed(read_bases.resize(st::assume_uninitialized, picked_groups.elements.size() + 1)))
        return expected_t::fail(status_t::bad_alloc_k, "read table allocation failed");
    std::size_t reservoir_rows = 0;
    for (std::size_t pick = 0; pick != picked_groups.size(); ++pick) {
        std::span<std::size_t const> const starts = starts_of(picked_shards[pick]);
        for (std::size_t read = picked_groups.offsets[pick]; read != picked_groups.offsets[pick + 1]; ++read) {
            u32_t const group = picked_groups.elements[read];
            read_bases[read] = reservoir_rows;
            reservoir_rows += starts[group + 1] - starts[group];
        }
    }
    read_bases[picked_groups.elements.size()] = reservoir_rows;

    sz::string_view_t names[4] = {anchor_.view(), positive_.view(), negative_.view(), {}};
    names[roles] = key_.view();
    std::size_t const read_columns = key_.empty() ? roles : roles + 1u;
    auto reservoir_read = parquet_file_t::make(
        *storage_, picked_paths, std::span<sz::string_view_t const> {names, read_columns}, picked_groups, gpu_stream);
    if (!reservoir_read) return expected_t::fail(std::move(reservoir_read));
    parquet_file_t const reservoir = std::move(reservoir_read.value_);
    std::size_t const pooled_rows = reservoir.rows();
    if (pooled_rows != reservoir_rows)
        return expected_t::fail(status_t::invalid_argument_k, "the read decoded {} rows of {} footer rows", pooled_rows,
                                reservoir_rows);
    column_view_t const anchors = reservoir.column(0), positives = reservoir.column(1);
    column_view_t const listed_negatives = negatives ? reservoir.column(2) : column_view_t {};

    switch_to_gpu_t gpu_scope {gpu_stream};
    if (!gpu_scope.ok())
        return expected_t::fail(status_t::device_error_k, "context entry failed for the loader stream");
    // Text negatives are strings grouped per row; media negatives are rows grouped by list offsets.
    st::basic_vector<u64_t> anchor_offsets, anchor_groups, positive_offsets, positive_groups, negative_offsets,
        negative_groups;
    st::basic_vector<i32_t> negative_rows;
    st::basic_vector<u64_t> key_offsets;
    st::basic_vector<char> key_bytes;
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
    if (landed && !key_.empty()) {
        column_view_t const sample_keys = reservoir.column(roles);
        landed = land_offsets_(sample_keys.offsets, key_offsets, gpu_stream);
        if (landed) landed = land_bytes_(sample_keys.payload, key_bytes, gpu_stream);
    }
    if (!landed) return landed;
    if (CUresult result = cuStreamSynchronize(gpu_stream.get()); result != CUDA_SUCCESS)
        return cuda_failure(result, "reservoir offsets readback failed");

    // Every pair on the host: a keyed bijection over the pool picks the rows, and a key per draw
    // picks the element of a list. A row missing any role is passed over.
    auto const is_empty = [](st::basic_vector<u64_t> const &offsets, u64_t element) noexcept {
        return offsets[element + 1] == offsets[element];
    };
    u64_t const row_key = hash(seed_, gpu_rank, row_draw_salt_k);
    st::basic_vector<u64_t> drawn, keys;
    if (st::failed(drawn.resize(st::assume_uninitialized, roles * total_pairs)) ||
        st::failed(keys.resize(st::assume_uninitialized, total_pairs)))
        return expected_t::fail(status_t::bad_alloc_k, "pair draw allocation failed ({} pairs)", total_pairs);
    u64r2_span_t const drawn_roles {drawn.data(), roles, total_pairs};
    // A pool smaller than the fill's pairs repeats in fresh keyed passes, and a pass that adds no
    // pair ends the draw.
    std::size_t found = 0, found_before_pass = 0;
    for (std::size_t draw = 0; pooled_rows != 0 && found != total_pairs; ++draw) {
        std::size_t const pass = draw / pooled_rows, position = draw % pooled_rows;
        if (position == 0 && pass != 0) {
            if (found == found_before_pass) break;
            found_before_pass = found;
        }
        u64_t const pass_key = pass == 0 ? row_key : hash(row_key, pass);
        u64_t const row = epoch_permutation(pass_key, data_epoch_t {training_step}, position, pooled_rows);
        auto const [anchor_first, anchor_count] = elements_of_(anchor_groups, row);
        auto const [positive_first, positive_count] = elements_of_(positive_groups, row);
        if (anchor_count == 0 || positive_count == 0) continue;
        u64_t const anchor = anchor_first + hash(row_key, training_step, draw) % anchor_count;
        u64_t const positive = positive_first +
                               hash(row_key, training_step, draw, positive_draw_salt_k) % positive_count;
        if (is_empty(anchor_offsets, anchor) || is_empty(positive_offsets, positive)) continue;
        std::size_t const read = static_cast<std::size_t>(std::upper_bound(read_bases.begin(), read_bases.end(), row) -
                                                          read_bases.begin() - 1);
        std::size_t const pick = static_cast<std::size_t>(
            std::upper_bound(picked_groups.offsets.begin(), picked_groups.offsets.end(), read) -
            picked_groups.offsets.begin() - 1);
        std::size_t const shard = picked_shards[pick];
        std::span<std::size_t const> const starts = starts_of(shard);
        u64_t const local_row = starts[picked_groups.elements[read]] + (row - read_bases[read]);
        if (negatives) {
            auto const [negative_first, negative_count] = elements_of_(negative_groups, row);
            if (negative_count == 0) continue;
            u64_t negative = negative_first + hash(row_key, training_step, draw, negative_draw_salt_k) % negative_count;
            if (media) {
                // A media negative is a shard-local row of the positive column, rebased here
                // when its group was drawn.
                std::size_t const shard_rows = starts.back();
                i32_t const local = negative_rows[negative];
                if (local < 0 || static_cast<std::size_t>(local) >= shard_rows)
                    return expected_t::fail(status_t::invalid_argument_k, "{} names negative row {} of a {}-row shard",
                                            picked_paths[pick], local, shard_rows);
                u32_t const group = static_cast<u32_t>(
                    std::upper_bound(starts.begin(), starts.end(), static_cast<std::size_t>(local)) - starts.begin() -
                    1);
                std::span<u32_t const> const read_groups = picked_groups[pick];
                auto const resident = std::lower_bound(read_groups.begin(), read_groups.end(), group);
                if (resident == read_groups.end() || *resident != group) continue;
                std::size_t const negative_read = picked_groups.offsets[pick] +
                                                  static_cast<std::size_t>(resident - read_groups.begin());
                negative = read_bases[negative_read] + (static_cast<u64_t>(local) - starts[group]);
            }
            if (is_empty(media ? positive_offsets : negative_offsets, negative)) continue;
            drawn_roles[negative_role_k, found] = negative;
        }
        drawn_roles[positive_role_k, found] = positive;
        drawn_roles[anchor_role_k, found] = anchor;
        // A sidecar's key names the media across rows; without one, a row is one sample in every
        // entry that reads its shard, such as a caption in another language.
        sz::string_view_t const sample_key = key_.empty() ? sz::string_view_t {}
                                                          : sz::string_view_t {key_bytes.data() + key_offsets[row],
                                                                               key_offsets[row + 1] - key_offsets[row]};
        keys[found] = (key_.empty() ? hash(shard_keys_[shard], local_row) : static_cast<u64_t>(sample_key.hash())) | 1u;
        ++found;
    }
    if (found != total_pairs)
        return expected_t::fail(status_t::unsupported_k,
                                "{} pooled rows across {} files hold {} complete pairs, fewer than the {} needed",
                                pooled_rows, picked_paths.size(), found, total_pairs);
    st::basic_vector<u64_t, gpu_alloc<u64_t>> drawn_on_gpu {gpu_alloc<u64_t> {gpu_stream.get()}},
        keys_on_gpu {gpu_alloc<u64_t> {gpu_stream.get()}};
    st::basic_vector<b8_t, gpu_alloc<b8_t>> prefixes {gpu_alloc<b8_t> {gpu_stream.get()}};
    std::size_t const anchor_prefix_bytes = anchor_prefix_.size(), positive_prefix_bytes = positive_prefix_.size();
    if (st::failed(drawn_on_gpu.resize(st::assume_uninitialized, drawn.size())) ||
        st::failed(keys_on_gpu.resize(st::assume_uninitialized, keys.size())) ||
        st::failed(prefixes.resize(st::assume_uninitialized,
                                   std::max<std::size_t>(anchor_prefix_bytes + positive_prefix_bytes, 1u))))
        return expected_t::fail(status_t::bad_alloc_k, "pair index allocation failed ({} pairs)", total_pairs);
    b8r1_span_t const anchor_prefix {prefixes.data(), anchor_prefix_bytes};
    b8r1_span_t const positive_prefix {prefixes.data() + anchor_prefix_bytes, positive_prefix_bytes};
    expected_t uploaded = copy_host_to_device<u64_t>(std::span<u64_t const> {drawn}, std::span<u64_t> {drawn_on_gpu},
                                                     gpu_stream);
    if (uploaded)
        uploaded = copy_host_to_device<u64_t>(std::span<u64_t const> {keys}, std::span<u64_t> {keys_on_gpu},
                                              gpu_stream);
    if (uploaded)
        uploaded = copy_host_to_device<b8_t>(std::as_bytes(std::span {anchor_prefix_.data(), anchor_prefix_bytes}),
                                             anchor_prefix, gpu_stream);
    if (uploaded)
        uploaded = copy_host_to_device<b8_t>(std::as_bytes(std::span {positive_prefix_.data(), positive_prefix_bytes}),
                                             positive_prefix, gpu_stream);
    if (!uploaded) return uploaded;
    // The draws per batch, `[roles, unroll, pairs]`, on the host.
    nk::tensor_view<u64_t, 3> const drawn_batches {drawn.data(), {roles, unroll, pairs}};

    // A media role decodes through the host-landed offsets; a text role packs on the device.
    column_view_t const positive_tape =
        media ? column_view_t {positives.payload, std::span<u64_t const> {positive_offsets}} : positives;
    column_view_t const negative_tape = media ? positive_tape : listed_negatives;
    u32_t const shape_width = modality_ == modality_t::image_k ? 2u : modality_ == modality_t::video_k ? 3u : 0u;
    // One scratch table row per role: offsets, lengths, modalities and, for images and videos, the
    // positive and negative media shapes; every batch reuses them in stream order.
    st::basic_vector<u32_t, gpu_alloc<u32_t>> offsets {gpu_alloc<u32_t> {gpu_stream.get()}};
    st::basic_vector<u32_t, gpu_alloc<u32_t>> lengths {gpu_alloc<u32_t> {gpu_stream.get()}};
    st::basic_vector<modality_t, gpu_alloc<modality_t>> modalities {gpu_alloc<modality_t> {gpu_stream.get()}};
    st::basic_vector<u16_t, gpu_alloc<u16_t>> media_shapes {gpu_alloc<u16_t> {gpu_stream.get()}};
    if (st::failed(offsets.resize(st::assume_uninitialized, roles * (pairs + 1))) ||
        st::failed(lengths.resize(st::assume_uninitialized, roles * pairs)) ||
        st::failed(modalities.resize(st::assume_uninitialized, roles * pairs)) ||
        st::failed(media_shapes.resize(st::assume_uninitialized, std::max<std::size_t>(2 * pairs * shape_width, 1u))))
        return expected_t::fail(status_t::bad_alloc_k, "pair table allocation failed ({} pairs)", pairs);
    u32r2_span_t const role_offsets {offsets.data(), roles, pairs + 1}, role_lengths {lengths.data(), roles, pairs};
    nk::tensor_span<modality_t, 2> const role_modalities {modalities.data(), roles, pairs};
    nk::tensor_span<u16_t, 3> const media_shape_tables {media_shapes.data(),
                                                        {std::size_t {2}, pairs, std::size_t {shape_width}}};
    for (std::size_t batch = 0; batch != unroll; ++batch) {
        multimodal_batch_on_gpu_t &loaded = batches[batch];
        if (loaded.tokens.extent(0) != rows || loaded.segment_lengths.extent(0) != segments)
            return expected_t::fail(status_t::unsupported_k, "batch {} differs from the first", batch);

        // One text past a role's cursor, packed the way the kernel packs it: cut at the row width,
        // never crossing a row.
        auto const advance = [&](std::size_t cursor, u32_t role, std::size_t pair,
                                 st::basic_vector<u64_t> const &offsets, std::size_t prefix_bytes) noexcept {
            u64_t const element = drawn_batches[role, batch, pair];
            std::size_t const kept = std::min<std::size_t>(prefix_bytes + offsets[element + 1] - offsets[element],
                                                           columns);
            if (kept > columns - cursor % columns) cursor = round_up_to_multiple(cursor, columns);
            return cursor + kept;
        };
        // Admit whole pairs in keyed order while their rows fit: captions pack back to back, and
        // media positives and negatives fill rows of their own, `media_cells` cells each.
        std::size_t admitted = 0, anchor_cells = 0, positive_cells = 0, negative_cells = 0;
        std::size_t const media_per_row = media ? columns / media_cells(modality_, static_cast<u32_t>(columns)) : 0;
        for (; admitted != pairs; ++admitted) {
            std::size_t const anchor = advance(anchor_cells, anchor_role_k, admitted, anchor_offsets,
                                               anchor_prefix_bytes);
            std::size_t positive = 0, negative = 0, needed = divide_round_up(anchor, columns);
            if (media) needed += divide_round_up(admitted + 1, media_per_row) * (negatives ? 2u : 1u);
            else {
                positive = advance(positive_cells, positive_role_k, admitted, positive_offsets, positive_prefix_bytes);
                if (negatives)
                    negative = advance(negative_cells, negative_role_k, admitted, negative_offsets,
                                       positive_prefix_bytes);
                needed += divide_round_up(positive, columns) + divide_round_up(negative, columns);
            }
            if (needed > rows) break;
            anchor_cells = anchor, positive_cells = positive, negative_cells = negative;
        }
        if (admitted == 0)
            return expected_t::fail(status_t::unsupported_k, "batch {} of {} rows fits no whole pair", batch, rows);
        std::size_t const anchor_rows = divide_round_up(anchor_cells, columns);
        std::size_t positive_rows = rows - anchor_rows, negative_rows_count = 0;
        if (media && negatives) negative_rows_count = positive_rows - positive_rows / 2, positive_rows /= 2;
        if (!media) {
            positive_rows = divide_round_up(positive_cells, columns);
            negative_rows_count = divide_round_up(negative_cells, columns);
        }
        // The admitted prefix of a role's per-pair tables, each a contiguous row of its table.
        auto const drawn_of = [&](u64_t const *base, u32_t role) noexcept {
            return u64r1_view_t {base + (role * unroll + batch) * pairs, admitted};
        };
        auto const offsets_of = [&](u32_t role) noexcept {
            return u32r1_span_t {offsets.data() + role * (pairs + 1), admitted + 1};
        };
        auto const lengths_of = [&](u32_t role) noexcept {
            return u32r1_span_t {lengths.data() + role * pairs, admitted};
        };
        auto const modalities_of = [&](u32_t role) noexcept {
            return nk::tensor_span<modality_t, 1> {modalities.data() + role * pairs, admitted};
        };

        // Positives over the leading rows, negatives over the next, anchors over the trailing ones.
        auto const fill_role = [&](u32_t role, std::size_t row_begin, std::size_t row_end, column_view_t tape,
                                   std::size_t shapes_row) noexcept -> expected_t {
            if (!media)
                return pack_tokens(tape, drawn_of(drawn_on_gpu.data(), role), positive_prefix,
                                   loaded.tokens[nk::range(row_begin, row_end), nk::slice], offsets_of(role),
                                   lengths_of(role), modalities_of(role), loaded.dropped.span(), gpu_stream);
            nk::tensor_span<u16_t, 2> const shapes_of_role = media_shape_tables[shapes_row, nk::slice];
            return decode(loaded, row_begin, row_end, tape, drawn_of(drawn.data(), role),
                          shapes_of_role[nk::range(std::size_t {0}, admitted), nk::slice], offsets_of(role),
                          lengths_of(role), modalities_of(role));
        };
        expected_t filled = fill_role(positive_role_k, 0, positive_rows, positive_tape, 0);
        if (filled && negatives)
            filled = fill_role(negative_role_k, positive_rows, positive_rows + negative_rows_count, negative_tape, 1);
        if (filled)
            filled = pack_tokens(anchors, drawn_of(drawn_on_gpu.data(), anchor_role_k), anchor_prefix,
                                 loaded.tokens[nk::range(rows - anchor_rows, rows), nk::slice],
                                 offsets_of(anchor_role_k), lengths_of(anchor_role_k), modalities_of(anchor_role_k),
                                 loaded.dropped.span(), gpu_stream);
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
                static_cast<u32_t>(positive_prefix_bytes), u64r1_view_t {keys_on_gpu.data() + batch * pairs, admitted},
                sides_, loaded.segment_offsets.span(), loaded.segment_lengths.span(), tags, shapes,
                static_cast<u32_t>(anchor_begin + anchor_cells), static_cast<u32_t>(admitted));
            !paired)
            return paired;
    }
    // The reservoir and scratch tables free stream-ordered, after every read this stream queued.
    return {};
}

expected_t paired_loader_parquet_t::fill(gpu_stream_t &gpu_stream, nvjpeg_session_t &decoder, u32_t gpu_rank,
                                         u32_t training_step, std::span<multimodal_batch_on_gpu_t> batches) noexcept {
    if (modality_ != modality_t::image_k)
        return expected_t::fail(status_t::invalid_argument_k, "the {} column holds no images", positive_.view());
    return fill_(gpu_stream, gpu_rank, training_step, batches,
                 [&](multimodal_batch_on_gpu_t &loaded, std::size_t row_begin, std::size_t row_end, column_view_t tape,
                     u64r1_view_t drawn, segment_shapes_span_t shapes, u32r1_span_t offsets, u32r1_span_t lengths,
                     modality_span_t modalities) noexcept {
                     pixel_patches_span_t const pixels = loaded.images[nk::range(row_begin, row_end), nk::slice];
                     return decode_images(decoder, pool_, tape, drawn, {&shape_.image, 1}, pixels, shapes, offsets,
                                          lengths, modalities, loaded.dropped.span(), gpu_stream);
                 });
}

expected_t paired_loader_parquet_t::fill(gpu_stream_t &gpu_stream, nvdec_session_t &decoder, u32_t gpu_rank,
                                         u32_t training_step, std::span<multimodal_batch_on_gpu_t> batches) noexcept {
    if (modality_ != modality_t::video_k)
        return expected_t::fail(status_t::invalid_argument_k, "the {} column holds no videos", positive_.view());
    video_batch_shape_t video = shape_.video;
    video.frame_stride = std::max(video.frame_stride, 1u);
    return fill_(gpu_stream, gpu_rank, training_step, batches,
                 [&](multimodal_batch_on_gpu_t &loaded, std::size_t row_begin, std::size_t row_end, column_view_t tape,
                     u64r1_view_t drawn, segment_shapes_span_t shapes, u32r1_span_t offsets, u32r1_span_t lengths,
                     modality_span_t modalities) noexcept {
                     pixel_tubes_span_t const tubes = loaded.videos[nk::range(row_begin, row_end), nk::slice];
                     return decode_videos(decoder, pool_, tape, drawn, {&video, 1}, tubes, shapes, offsets, lengths,
                                          modalities, gpu_stream);
                 });
}

expected_t paired_loader_parquet_t::fill(gpu_stream_t &gpu_stream, u32_t gpu_rank, u32_t training_step,
                                         std::span<multimodal_batch_on_gpu_t> batches) noexcept {
    if (modality_ != modality_t::audio_k && modality_ != modality_t::text_k)
        return expected_t::fail(status_t::invalid_argument_k, "the {} column holds no recordings or texts",
                                positive_.view());
    // A text pair packs its positives in the fill itself and never calls the decode.
    return fill_(
        gpu_stream, gpu_rank, training_step, batches,
        [&](multimodal_batch_on_gpu_t &loaded, std::size_t row_begin, std::size_t row_end, column_view_t tape,
            u64r1_view_t drawn, segment_shapes_span_t, u32r1_span_t offsets, u32r1_span_t lengths,
            modality_span_t modalities) noexcept {
            audio_samples_span_t const samples = loaded.audio_samples[nk::range(row_begin, row_end), nk::slice];
            return decode_audios(pool_, tape, drawn, {&shape_.audio, 1}, samples, offsets, lengths, modalities,
                                 loaded.dropped.span(), gpu_stream);
        });
}
