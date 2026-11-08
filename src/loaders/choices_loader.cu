/**
 *  @file src/loaders/choices_loader.cu
 *  @author Ash Vardanian
 *  @date October 25, 2026
 *  @brief The Parquet choices loader's bodies and its packing kernel — see the header
 *      for the contract.
 *
 *  A fill reads the state, choices and targets columns of the drawn row groups into one reservoir,
 *  lands the offsets, the choices and the targets on the host, draws every row there, and lays each
 *  batch's segment table and choice rows out on the host too. One kernel then copies each state's
 *  kept tail from the reservoir and appends the mask byte.
 */
#include <cstring> // `std::memcpy` — a target's little-endian Float32 bytes

#include <algorithm> // `std::min`
#include <span>      // `std::as_writable_bytes`

#include <cuda.h> // `cuStreamSynchronize`

#include "uform/loaders/choices_loader.cuh"

using namespace unum::uform::encoder;
using namespace unum::uform::utilities;
using namespace unum::uform::loaders;

/** Purpose salts folded through @ref splitmix64_hash_t so the file draw and the row draw stay
 *  decorrelated under one @c (seed,step,rank) — the same salts the other loaders use. */
inline constexpr u64_t file_draw_salt_k = 0x9E3779B97F4A7C15ull;
inline constexpr u64_t row_draw_salt_k = 0xD1B54A32D192ED03ull;

/**
 *  @brief Copy each choice segment's kept state tail from the reservoir into the token grid and
 *      append the mask byte, the position the decoder answers at.
 *
 *  __grid_size__ @c [segments,1,1] , threads stripe the segment; a PAD segment exits at once.
 */
static __global__ void choices_states_kernel_(b8r1_view_t states, u64r1_view_t sources, u32r1_view_t segment_offsets,
                                              u32r1_view_t segment_lengths, tokens_span_t tokens) {
    u32_t const segment = blockIdx.x, length = segment_lengths[segment];
    if (length == 0u) return;
    std::size_t const begin = segment_offsets[segment], source = sources[segment];
    for (u32_t column = threadIdx.x; column + 1u < length; column += blockDim.x)
        tokens[begin + column] = static_cast<token_id_t>(static_cast<u8_t>(states[source + column]));
    if (threadIdx.x == 0u) tokens[begin + length - 1u] = byte_mask_k;
}

/** Land the device offsets @p offsets on the host into @p landed . */
[[nodiscard]] static expected_t land_offsets_(u64r1_view_t offsets, st::basic_vector<u64_t> &landed,
                                              gpu_stream_t &stream) noexcept {
    if (st::failed(landed.resize(st::assume_uninitialized, offsets.extent(0))))
        return expected_t::fail(status_t::bad_alloc_k, "offset landing failed ({} entries)", offsets.extent(0));
    return copy_device_to_host(offsets, std::span<u64_t> {landed}, stream);
}

/** Land the device bytes @p payload on the host into @p landed . */
[[nodiscard]] static expected_t land_bytes_(b8r1_view_t payload, st::basic_vector<u8_t> &landed,
                                            gpu_stream_t &stream) noexcept {
    if (st::failed(landed.resize(st::assume_uninitialized, payload.extent(0))))
        return expected_t::fail(status_t::bad_alloc_k, "byte landing failed ({} bytes)", payload.extent(0));
    return copy_device_to_host<b8_t>(payload, std::as_writable_bytes(std::span<u8_t> {landed}), stream);
}

expected<choices_loader_parquet_t> choices_loader_parquet_t::make(gpu_direct_storage_session_t &session,
                                                                  sz::string_view_t path_or_glob, u64_t seed,
                                                                  choices_loader_parquet_config_t config) noexcept {
    using result_t = expected<choices_loader_parquet_t>;
    expected<st::basic_vector<sz::string_t>> paths = expand_glob(path_or_glob);
    if (!paths) return result_t::fail(std::move(paths));

    choices_loader_parquet_t loader;
    loader.storage_ = &session;
    loader.seed_ = seed;
    if (sz::failed(loader.state_.assign(config.state)) || sz::failed(loader.choices_.assign(config.choices)) ||
        sz::failed(loader.targets_.assign(config.targets)))
        return result_t::fail(status_t::bad_alloc_k, "column name staging failed");
    sz::string_view_t const columns[3] = {config.state, config.choices, config.targets};
    std::size_t groups = 0;
    for (sz::string_t const &path : paths.value_) {
        expected<parquet_length_metadata_t> metadata = parquet_file_t::table_metadata(
            session, path, std::span<sz::string_view_t const> {columns});
        if (!metadata) return result_t::fail(std::move(metadata));
        if (metadata->rows == 0) return result_t::fail(status_t::invalid_argument_k, "shard {} is empty", path);
        if (expected_t appended = loader.groups_.append_shard(path, *metadata); !appended)
            return result_t::fail(std::move(appended));
        groups += metadata->group_rows.size();
        loader.reservoir_.charge(*metadata);
    }
    if (loader.rows() == 0)
        return result_t::fail(status_t::invalid_argument_k, "{} holds no row with a state, choices and targets",
                              path_or_glob);
    if (expected_t reserved = loader.groups_.reserve(groups); !reserved) return result_t::fail(std::move(reserved));
    return result_t {std::move(loader)};
}

expected_t choices_loader_parquet_t::fill(gpu_stream_t &gpu_stream, u32_t gpu_rank, u32_t training_step,
                                          std::span<multimodal_batch_on_gpu_t> batches) noexcept {
    if (batches.empty()) return {};
    std::size_t const unroll = batches.size();
    std::size_t const rows = batches[0].tokens.extent(0), columns = batches[0].tokens.extent(1);
    std::size_t const segments = batches[0].segment_lengths.extent(0);
    std::size_t const width = batches[0].segment_choices.extent(1), total_segments = unroll * segments;
    if (width == 0) return expected_t::fail(status_t::unsupported_k, "the batches carry no choice columns");

    if (expected_t picked = groups_.draw(seed_, training_step, gpu_rank, file_draw_salt_k, total_segments, 2); !picked)
        return picked;
    constexpr splitmix64_hash_t hash {};
    sz::string_view_t const names[3] = {state_.view(), choices_.view(), targets_.view()};
    expected<parquet_file_t> reservoir_read = parquet_file_t::make(*storage_, groups_.picked_paths(),
                                                                   std::span<sz::string_view_t const> {names},
                                                                   groups_.picked_groups(), gpu_stream);
    if (!reservoir_read) return expected_t::fail(std::move(reservoir_read));
    parquet_file_t const reservoir = std::move(reservoir_read.value_);
    std::size_t const pooled_rows = reservoir.rows();
    column_view_t const states = reservoir.column(0);

    switch_to_gpu_t gpu_scope {gpu_stream};
    if (!gpu_scope.ok())
        return expected_t::fail(status_t::device_error_k, "context entry failed for the loader stream");
    st::basic_vector<u64_t> state_offsets, choice_offsets, target_offsets;
    st::basic_vector<u8_t> choice_bytes, target_bytes;
    expected_t landed = land_offsets_(states.offsets, state_offsets, gpu_stream);
    if (landed) landed = land_offsets_(reservoir.column(1).offsets, choice_offsets, gpu_stream);
    if (landed) landed = land_bytes_(reservoir.column(1).payload, choice_bytes, gpu_stream);
    if (landed) landed = land_offsets_(reservoir.column(2).offsets, target_offsets, gpu_stream);
    if (landed) landed = land_bytes_(reservoir.column(2).payload, target_bytes, gpu_stream);
    if (!landed) return landed;
    if (CUresult result = cuStreamSynchronize(gpu_stream.get()); result != CUDA_SUCCESS)
        return cuda_failure(result, "reservoir readback failed");

    // A row offers distinct bytes that fit a segment, each with one Float32 target.
    auto const well_formed = [&](u64_t row) noexcept {
        u64_t const offered = choice_offsets[row + 1] - choice_offsets[row];
        if (offered == 0 || offered > width || target_offsets[row + 1] - target_offsets[row] != offered * sizeof(f32_t))
            return false;
        u8_t const *const bytes = choice_bytes.data() + choice_offsets[row];
        for (u64_t choice = 1; choice < offered; ++choice)
            for (u64_t earlier = 0; earlier != choice; ++earlier)
                if (bytes[earlier] == bytes[choice]) return false;
        return true;
    };
    // A pool smaller than the fill's segments repeats in fresh keyed passes, and a pass that adds
    // no row ends the draw.
    u64_t const row_key = hash(seed_, gpu_rank, row_draw_salt_k);
    st::basic_vector<u64_t> drawn;
    if (st::failed(drawn.resize(st::assume_uninitialized, total_segments)))
        return expected_t::fail(status_t::bad_alloc_k, "row draw allocation failed ({} rows)", total_segments);
    std::size_t found = 0, found_before_pass = 0;
    for (std::size_t draw = 0; pooled_rows != 0 && found != total_segments; ++draw) {
        std::size_t const pass = draw / pooled_rows, position = draw % pooled_rows;
        if (position == 0 && pass != 0) {
            if (found == found_before_pass) break;
            found_before_pass = found;
        }
        u64_t const pass_key = pass == 0 ? row_key : hash(row_key, pass);
        u64_t const row = epoch_permutation(pass_key, data_epoch_t {training_step}, position, pooled_rows);
        if (well_formed(row)) drawn[found++] = row;
    }
    if (found != total_segments)
        return expected_t::fail(status_t::unsupported_k,
                                "{} pooled rows across {} files hold no choice of at most {} distinct bytes with " //
                                "a target each",
                                pooled_rows, groups_.picked_paths().size(), width);

    // Each batch's tables are laid out on the host, one batch at a time, and reused once uploaded.
    st::basic_vector<u32_t> offsets, lengths;
    st::basic_vector<modality_t> modalities;
    st::basic_vector<u64_t> sources;
    st::basic_vector<token_id_t> choices;
    st::basic_vector<f32_t> targets;
    st::basic_vector<u64_t, gpu_alloc<u64_t>> sources_on_gpu {gpu_alloc<u64_t> {gpu_stream.get()}};
    if (st::failed(offsets.resize(segments + 1)) || st::failed(lengths.resize(segments)) ||
        st::failed(modalities.resize(segments)) || st::failed(sources.resize(segments)) ||
        st::failed(choices.resize(segments * width)) || st::failed(targets.resize(segments * width)) ||
        st::failed(sources_on_gpu.resize(st::assume_uninitialized, segments)))
        return expected_t::fail(status_t::bad_alloc_k, "choice table allocation failed ({} segments)", segments);
    std::size_t const capacity = rows * columns;
    u64r2_view_t const drawn_rows {drawn.data(), unroll, segments};
    nk::tensor_span<token_id_t, 2> const choice_rows {choices.data(), segments, width};
    f32r2_span_t const target_rows {targets.data(), segments, width};
    for (std::size_t batch = 0; batch != unroll; ++batch) {
        multimodal_batch_on_gpu_t &loaded = batches[batch];
        if (loaded.tokens.extent(0) != rows || loaded.segment_lengths.extent(0) != segments ||
            loaded.segment_choices.extent(1) != width)
            return expected_t::fail(status_t::unsupported_k, "batch {} differs from the first", batch);

        // A state keeps its tail and the mask byte at the row width, never crossing a row; the
        // segments past the rows' room PAD.
        std::size_t cursor = 0;
        bool room = true;
        for (std::size_t segment = 0; segment != segments; ++segment) {
            u64_t const row = drawn_rows[batch, segment];
            std::size_t kept = room ? std::min<std::size_t>(state_offsets[row + 1] - state_offsets[row] + 1u, columns)
                                    : 0u;
            if (kept > columns - cursor % columns) cursor = round_up_to_multiple(cursor, columns);
            if (cursor + kept > capacity) room = false, kept = 0;
            offsets[segment] = static_cast<u32_t>(cursor);
            lengths[segment] = static_cast<u32_t>(kept);
            modalities[segment] = modality_t::text_k;
            sources[segment] = state_offsets[row + 1] - (kept != 0 ? kept - 1u : 0u);
            u64_t const offered = kept != 0 ? choice_offsets[row + 1] - choice_offsets[row] : 0u;
            for (std::size_t choice = 0; choice != width; ++choice) {
                bool const live = choice < offered;
                choice_rows[segment, choice] = live ? token_id_t {choice_bytes[choice_offsets[row] + choice]}
                                                    : byte_pad_k;
                f32_t target = 0.0f;
                if (live)
                    std::memcpy(&target, target_bytes.data() + target_offsets[row] + choice * sizeof(f32_t),
                                sizeof(f32_t));
                target_rows[segment, choice] = target;
            }
            cursor += kept;
        }
        offsets[segments] = static_cast<u32_t>(cursor);

        expected_t staged = copy_host_to_device(std::span<u32_t const> {offsets}, loaded.segment_offsets.span(),
                                                gpu_stream);
        if (staged)
            staged = copy_host_to_device(std::span<u32_t const> {lengths}, loaded.segment_lengths.span(), gpu_stream);
        if (staged)
            staged = copy_host_to_device(std::span<modality_t const> {modalities}, loaded.segment_modality.span(),
                                         gpu_stream);
        if (staged)
            staged = copy_host_to_device<u64_t>(std::span<u64_t const> {sources}, std::span<u64_t> {sources_on_gpu},
                                                gpu_stream);
        if (staged)
            staged = copy_host_to_device(std::span<token_id_t const> {choices},
                                         loaded.segment_choices.span().flatten<1>(), gpu_stream);
        if (staged)
            staged = copy_host_to_device(std::span<f32_t const> {targets},
                                         loaded.segment_choice_targets.span().flatten<1>(), gpu_stream);
        // A choice names no bag, role, key, prefix or side, so none keeps stale bytes.
        if (staged) staged = zero_async(loaded.segment_bag.span(), gpu_stream);
        if (staged) staged = zero_async(loaded.segment_side.span(), gpu_stream);
        if (staged) staged = zero_async(loaded.segment_role.span(), gpu_stream);
        if (staged) staged = zero_async(loaded.segment_key.span(), gpu_stream);
        if (staged) staged = zero_async(loaded.segment_prefix_length.span(), gpu_stream);
        if (staged)
            staged = launch_kernel(&choices_states_kernel_, dim3 {static_cast<u32_t>(segments), 1u, 1u},
                                   dim3 {kernel_block_size_default_k, 1u, 1u}, gpu_stream, states.payload,
                                   u64r1_view_t {sources_on_gpu.data(), segments}, loaded.segment_offsets.view(),
                                   loaded.segment_lengths.view(), loaded.tokens.span().flatten<1>());
        // The next batch rewrites the host tables and the sources, so this one's copies land first.
        if (staged)
            if (CUresult result = cuStreamSynchronize(gpu_stream.get()); result != CUDA_SUCCESS)
                staged = cuda_failure(result, "choice fill drain failed (batch={})", batch);
        if (!staged) return staged;
    }
    return {};
}
