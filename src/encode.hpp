/**
 *  @file src/encode.hpp
 *  @author Ash Vardanian
 *  @date October 17, 2026
 *  @brief What every encode verb shares once its packer is chosen: the plans, the sites a call computes on, and the
 *      body that sizes the outputs, packs each slice and runs the model's tower over it.
 *
 *  Included by @c encode.cpp, whose verbs take decoded inputs, and by @c encode_media.cpp, whose verbs decode files,
 *  so a build without the codecs links the first alone.
 *
 *  A RAM call packs its batch and borrows the tower's scratch from its own arena, which keeps the high-water mark. A
 *  VRAM call plans in the executor's host pages and packs what the devices read into its first device's pinned
 *  pages; each device copies its run of the slice into its own scratch, writes its share of the outputs there, and
 *  copies it into the arena when the arena lives on another device.
 */
#pragma once

#include <cmath>   // `std::exp` - a choice's softmax
#include <cstring> // `std::memcpy`, `std::memset`

#include <initializer_list> // `std::initializer_list` - the inputs a call checks against its arena
#include <limits>           // `std::numeric_limits` - the peak a choice's softmax starts below
#include <span>             // `std::span` - an item's bytes as the decoders take them

#include "uform/media/image.hpp" // `image_plan_t`, `image_extent_t`, `image_channels_k`

#include "encoder_punned.hpp" // `encoder_verbs` - what a model's towers answer to

namespace unum::uform {

namespace ue = unum::uform::encoder;
namespace um = unum::uform::media;

using core::f32_t;
using core::modality_t;
using core::pixel_t;
using core::token_id_t;
using core::u16_t;
using core::u32_t;

#pragma region Plans

/** A failure the steps of a call report before it is final, with its static message. */
struct failure_t {
    uform_status_t status = uform_status_success_k;
    char const *message = nullptr;
};

/** The C status a media decoder's failure crosses into: whatever the bytes do not decode to is malformed. */
inline uform_status_t media_status_of(core::status_t status) noexcept {
    switch (status) {
    case core::status_t::bad_alloc_k: return uform_status_out_of_memory_k;
    case core::status_t::shape_unsupported_k: return uform_status_unsupported_k;
    default: return uform_status_malformed_k;
    }
}

/** What the plan of a call decides about one item before any slice packs it. */
struct item_plan_t {

    /** Tokens the item keeps after truncation; an empty item keeps none and pools to a zero row. */
    u32_t tokens = 0;

    /** Tokens the item spans before truncation. */
    std::size_t spanned = 0;

    /** The factorization its coordinates read: an image's @c (rows,columns) , a clip's @c (groups,rows,columns) . */
    u16_t shape[3] = {};

    /** An encoded image's format, decoded extent and DCT scale. */
    um::image_plan_t image {};
};

/** Bytes one value of an output @p scalar spans, zero for the scalars outputs never land as. */
inline std::size_t output_bytes(uform_scalar_t scalar) noexcept {
    switch (scalar) {
    case uform_scalar_f32_k: return 4;
    case uform_scalar_f16_k:
    case uform_scalar_bf16_k: return 2;
    default: return 0;
    }
}

inline nk_dtype_t output_dtype(uform_scalar_t scalar) noexcept {
    switch (scalar) {
    case uform_scalar_f16_k: return nk_f16_k;
    case uform_scalar_bf16_k: return nk_bf16_k;
    default: return nk_f32_k;
    }
}

/** The refusals of every image verb: a model without the image entry, or one whose pixels are not RGBA. */
inline failure_t validate_images(ue::geometry_t const &geometry) noexcept {
    if (!geometry.image_patch_side) return {uform_status_unsupported_k, "The model was not trained on images"};
    if (geometry.image_channels != um::image_channels_k)
        return {uform_status_unsupported_k, "The model's images are not RGBA"};
    return {};
}

/** The patch count an image of @p native extent spans before truncation. */
inline std::size_t native_patches(um::image_extent_t native, u32_t patch_side) noexcept {
    return core::divide_round_up(native.width, patch_side) * core::divide_round_up(native.height, patch_side);
}

/** The refusal of every audio verb: a model without the audio entry. */
inline failure_t validate_audio(ue::geometry_t const &geometry) noexcept {
    if (!geometry.audio_chunk_samples || !geometry.audio_sample_rate)
        return {uform_status_unsupported_k, "The model was not trained on audio"};
    return {};
}

#pragma endregion Plans

#pragma region Sites

/** The C status a tower's failure crosses into: a batch it calls invalid is a packing defect of the encode verbs. */
inline failure_t tower_failure(core::status_t status) noexcept {
    switch (status) {
    case core::status_t::invalid_input_k: return {uform_status_internal_k, "The tower refused a packed batch"};
    case core::status_t::bad_alloc_k: return {uform_status_out_of_memory_k, "The tower ran out of memory"};
    case core::status_t::shape_unsupported_k:
        return {uform_status_unsupported_k, "No kernel of the model's class takes the batch's shapes"};
    default: return {status_of(status), "The tower or its device failed to encode the batch"};
    }
}

/** @p view over @p data in place of its own storage, its shape kept; an absent view stays absent. */
template <typename view_type_>
view_type_ rebased(view_type_ view, void const *data) noexcept {
    if (!view.byte_data()) return view;
    return view_type_ {static_cast<char const *>(data), view.shape()};
}

/** @p rows over @p staging in place of their own storage, their shape kept; absent rows stay
 *  absent. */
inline ue::rows_span_t rebased(ue::rows_span_t rows, bytes_span_t staging) noexcept {
    if (rows.empty()) return rows;
    return {nk::tensor_span<b8_t, 2> {staging.data(), rows.bytes.extent(0), rows.bytes.extent(1)}, rows.dtype};
}

/** A RAM call: the host tower on a pool, packing into the call's own arena, which also lends the scratch. */
struct ram_site_t {
    fu::flat_pool_t &pool;

    /** What only the host reads, and what the tower reads, both the call's arena on the host. */
    chunked_arena_t &staging;
    chunked_arena_t &transfers;

    /** The variant the model's load picked, and its one host tower. */
    encoder_verbs<encoder_home_t::ram_k> const &verbs;
    void const *tower = nullptr;
    std::span<b8_t> scratch {};

    failure_t reserve(std::size_t tokens, std::size_t segments, std::size_t, ue::outputs_t) noexcept {
        core::expected<std::size_t> bytes = verbs.scratch_bytes(tower, tokens, segments, pool);
        if (!bytes) return tower_failure(bytes.status_);
        b8_t *const data = static_cast<b8_t *>(staging.allocate(*bytes));
        if (!data && *bytes) return {uform_status_out_of_memory_k, "Failed to allocate the activations"};
        scratch = {data, *bytes};
        return {};
    }

    failure_t encode(core::multimodal_batch_view_t const &batch, bytes_view_t, std::span<std::size_t const>,
                     ue::outputs_t outputs) noexcept {
        core::expected_t encoded = verbs.encode(tower, batch, outputs, scratch, pool);
        return encoded ? failure_t {} : tower_failure(encoded.status_);
    }

    failure_t land(bytes_span_t target, bytes_view_t source) noexcept {
        std::memcpy(target.data(), source.data(), source.size());
        return {};
    }
    failure_t finish() noexcept { return {}; }
};

/** One device of a VRAM call: the executor's device, the model's replica on it, and its regions of the scratch. */
struct vram_shard_t {
    vram_device_t *device = nullptr;
    vram_replica_t const *replica = nullptr;
    std::span<b8_t> scratch {};

    /** Room for a whole slice's cells, of which the shard copies in only its own run, at the run's offset. */
    bytes_span_t cells {};

    /** @b [segments,2] and @b [segments,3] the media shapes of the shard's run. */
    nk::tensor_span<u16_t, 2> image_shapes {}, video_shapes {};

    /** Where a device other than the arena's writes its outputs before copying them over; empty on
     *  the arena's. */
    bytes_span_t embeddings {}, states {}, token_vectors {};

    /** @b [segments,masked_vocabulary] where every shard writes its logits before copying them into
     *  host pages. */
    core::f32r2_span_t masked_logits {};
};

/** A VRAM call: plans on the decoder pool into host pages, packs into pinned pages, and splits each
 *  slice across the devices holding a replica, by token count, through the verbs of the variant
 *  @p home_ 's load picked, moving each shard's run in and its outputs out through the vendor's
 *  copies. */
template <encoder_home_t home_>
struct vram_site {
    using entries_kit_t = entries_kit_of<home_>;

    fu::flat_pool_t &pool;
    chunked_arena_t &staging;
    chunked_arena_t &transfers;

    /** The stream of the device the call's arena lives on, whose shard writes the arena directly. */
    nk_stream_t arena_stream = nullptr;
    encoder_verbs<home_> const &verbs;
    per_gpu_array<vram_shard_t> shards {};
    std::size_t cell_bytes = 0;

    failure_t reserve(std::size_t tokens, std::size_t segments, std::size_t bytes_per_cell,
                      ue::outputs_t surfaces) noexcept {
        cell_bytes = bytes_per_cell;
        for (vram_shard_t &shard : shards) {
            core::expected<std::size_t> bytes = verbs.scratch_bytes(shard.replica->tower, tokens, segments,
                                                                    shard.device->stream.get());
            if (!bytes) return tower_failure(bytes.status_);
            chunked_arena_t &memory = shard.device->scratch.chunks;
            memory.rewind();
            failure_t failure {};
            auto const allocate = [&](std::size_t size) noexcept {
                b8_t *const data = static_cast<b8_t *>(memory.allocate(size));
                if (!data && size) failure = {uform_status_out_of_memory_k, "The executor's scratch limit is too low"};
                return bytes_span_t {data, data ? size : 0};
            };
            auto const shapes = [&](std::size_t axes) noexcept {
                bytes_span_t const shape_bytes = allocate(segments * axes * sizeof(u16_t));
                return nk::tensor_span<u16_t, 2> {static_cast<u16_t *>(static_cast<void *>(shape_bytes.data())),
                                                  segments, axes};
            };
            bool const elsewhere = shard.device->stream.get() != arena_stream;
            auto const staging_of = [&](ue::rows_span_t surface, std::size_t rows) noexcept {
                return elsewhere && !surface.empty() ? allocate(rows * surface.bytes.extent(1)) : bytes_span_t {};
            };
            shard.scratch = allocate(*bytes);
            shard.cells = allocate(tokens * cell_bytes);
            shard.image_shapes = shapes(2);
            shard.video_shapes = shapes(3);
            shard.embeddings = staging_of(surfaces.embeddings, segments);
            shard.states = staging_of(surfaces.states, tokens);
            shard.token_vectors = staging_of(surfaces.token_vectors, tokens);
            if (!surfaces.masked_logits.empty()) {
                std::size_t const vocabulary = surfaces.masked_logits.extent(1);
                bytes_span_t const logits = allocate(segments * vocabulary * sizeof(f32_t));
                shard.masked_logits = {static_cast<f32_t *>(static_cast<void *>(logits.data())), segments, vocabulary};
            }
            if (failure.message) return failure;
        }
        return {};
    }

    /** Runs the items of @p batch , whose cells @p cells hold, on every shard, each taking a contiguous run of about
     *  equal tokens, their outputs landing at the run's rows. */
    failure_t encode(core::multimodal_batch_view_t const &batch, bytes_view_t cells,
                     std::span<std::size_t const> packed_offsets, ue::outputs_t outputs) noexcept {
        std::size_t const segments = packed_offsets.size() - 1, origin = packed_offsets[0];
        std::size_t const tokens = packed_offsets[segments] - origin, count = shards.size();
        failure_t const moved {uform_status_device_k, "A device failed to copy a batch or its outputs"};
        std::size_t first = 0;
        for (std::size_t index = 0; index != count && first != segments; ++index) {
            std::size_t last = first;
            std::size_t const target = core::divide_round_up((index + 1) * tokens, count);
            while (last != segments && (packed_offsets[last + 1] - origin <= target || index + 1 == count)) ++last;
            if (last == first) continue;
            vram_shard_t const &shard = shards[index];
            nk_stream_t const stream = shard.device->stream.get();
            auto const copied = [&](bytes_span_t into, bytes_view_t from) noexcept {
                return from.empty() || entries_kit_t::copy(into, from, stream) == nk_success_k;
            };
            std::size_t const begin = packed_offsets[first] - origin, end = packed_offsets[last] - origin;
            std::size_t const items = last - first;
            auto const shard_items = nk::range(first, last);
            nk::tensor_span<u16_t, 2> const image_shapes = shard.image_shapes[nk::range(0, items), nk::slice];
            nk::tensor_span<u16_t, 2> const video_shapes = shard.video_shapes[nk::range(0, items), nk::slice];
            std::size_t const run_first = begin * cell_bytes, run_bytes = (end - begin) * cell_bytes;
            if (!copied(shard.cells.subspan(run_first, run_bytes), cells.subspan(run_first, run_bytes)) ||
                !copied(core::bytes_of(image_shapes), core::bytes_of(batch.image_shapes[shard_items, nk::slice])) ||
                !copied(core::bytes_of(video_shapes), core::bytes_of(batch.video_shapes[shard_items, nk::slice])))
                return moved;

            core::multimodal_batch_view_t shard_batch = batch;
            shard_batch.segment_offsets = batch.segment_offsets[nk::range(first, last + 1), nk::slice];
            shard_batch.segment_lengths = batch.segment_lengths[shard_items, nk::slice];
            shard_batch.segment_modality = batch.segment_modality[shard_items, nk::slice];
            shard_batch.image_shapes = image_shapes;
            shard_batch.video_shapes = video_shapes;
            shard_batch.tokens = rebased(batch.tokens, shard.cells.data());
            shard_batch.images = rebased(batch.images, shard.cells.data());
            shard_batch.videos = rebased(batch.videos, shard.cells.data());
            shard_batch.audio_samples = rebased(batch.audio_samples, shard.cells.data());

            ue::outputs_t const landing {outputs.embeddings.slice(first, items),
                                         outputs.states.slice(begin, end - begin),
                                         outputs.token_vectors.slice(begin, end - begin),
                                         {}};
            bool const elsewhere = stream != arena_stream;
            ue::outputs_t shard_outputs = landing;
            if (elsewhere)
                shard_outputs = {rebased(landing.embeddings, shard.embeddings),
                                 rebased(landing.states, shard.states),
                                 rebased(landing.token_vectors, shard.token_vectors),
                                 {}};
            if (!outputs.masked_logits
                     .empty()) // they land in host pages, so every shard computes them in its own memory
                shard_outputs.masked_logits = shard.masked_logits[nk::range(0, items), nk::slice];
            core::expected_t encoded = verbs.encode(shard.replica->tower, shard_batch, shard_outputs, shard.scratch,
                                                    stream);
            if (!encoded) return tower_failure(encoded.status_);
            if (elsewhere &&
                (!copied(core::bytes_of(landing.embeddings.bytes), core::bytes_of(shard_outputs.embeddings.bytes)) ||
                 !copied(core::bytes_of(landing.states.bytes), core::bytes_of(shard_outputs.states.bytes)) ||
                 !copied(core::bytes_of(landing.token_vectors.bytes),
                         core::bytes_of(shard_outputs.token_vectors.bytes))))
                return moved;
            if (!outputs.masked_logits.empty() &&
                !copied(core::bytes_of(outputs.masked_logits[nk::range(first, last), nk::slice]),
                        core::bytes_of(shard_outputs.masked_logits)))
                return moved;
            first = last;
        }
        return {};
    }

    /** Copies host bytes into the arena on the first shard's stream, which @ref finish waits for. */
    failure_t land(bytes_span_t target, bytes_view_t source) noexcept {
        vram_shard_t const &first = shards[0];
        if (entries_kit_t::copy(target, source, first.device->stream.get()) != nk_success_k)
            return {uform_status_device_k, "A device failed to copy into the arena"};
        return {};
    }

    failure_t finish() noexcept {
        failure_t failure {};
        for (vram_shard_t const &shard : shards)
            if (nk_status_t const status = nk_stream_synchronize_best(shard.device->capabilities,
                                                                      shard.device->stream.get());
                status != nk_success_k && !failure.message)
                failure = {status_of(status), "A device failed the work queued on it"};
        return failure;
    }
};

#pragma endregion Sites

#pragma region Calls

/** The surfaces a call's arguments name, and the scalar they land as. */
struct requested_t {
    uform_rows_view_t *embeddings = nullptr;
    uform_rows_tape_view_t *states = nullptr;
    uform_rows_tape_view_t *token_vectors = nullptr;
    uform_scalar_t scalar = uform_scalar_f32_k;
};

/** The surfaces @p arguments name; a choice names none of them, as it lands probabilities. */
template <typename arguments_type_>
requested_t requested_of(arguments_type_ const &arguments) noexcept {
    if constexpr (requires { arguments.probabilities; }) return {};
    else return {arguments.embeddings, arguments.states, arguments.token_vectors, arguments.outputs_scalar};
}

/** Zeroes every output and report, so a call refused or failing before it lands leaves them null. */
template <typename arguments_type_>
void clear_outputs(arguments_type_ &arguments) noexcept {
    requested_t const requested = requested_of(arguments);
    if (requested.embeddings) *requested.embeddings = {};
    if (requested.states) *requested.states = {};
    if (requested.token_vectors) *requested.token_vectors = {};
    if constexpr (requires { arguments.probabilities; })
        if (arguments.probabilities) *arguments.probabilities = {};
    if (arguments.stats) *arguments.stats = {};
}

/** The refusals @p tape shares with every tape a call reads: storage, offsets and @p element . */
inline failure_t validate_tape(uform_tape_view_t const &tape, std::size_t count, uform_scalar_t element) noexcept {
    if (!tape.offsets || (!tape.data && count && tape.offsets[count] != tape.offsets[0]))
        return {uform_status_invalid_argument_k, "The tape must name its storage and offsets"};
    if (tape.scalar != element) return {uform_status_unsupported_layout_k, "The tape holds another scalar"};
    for (std::size_t item = 0; item != count; ++item)
        if (tape.offsets[item + 1] < tape.offsets[item])
            return {uform_status_malformed_k, "The tape's offsets must not decrease"};
    return {};
}

/** The refusals of a choice beyond its states': the decoder, the choices and the groups. */
inline failure_t validate_choices(uform_text_choose_t const &arguments) noexcept {
    ue::geometry_t const &geometry = arguments.model->geometry;
    if (!geometry.masked_vocabulary) return {uform_status_unsupported_k, "The model carries no byte decoder"};
    if (!arguments.probabilities) return {uform_status_invalid_argument_k, "The probabilities must not be null"};
    if (failure_t const refused = validate_tape(arguments.choices, arguments.count, uform_scalar_u8_k); refused.message)
        return refused;
    if (arguments.temperature_groups)
        for (std::size_t item = 0; item != arguments.count; ++item)
            if (arguments.temperature_groups[item] >= geometry.choice_temperature_groups)
                return {uform_status_invalid_argument_k, "A temperature group exceeds the model's"};
    return {};
}

/** The refusals every encode verb shares before it claims its arena. */
template <typename arguments_type_>
failure_t validate_call(arguments_type_ const &arguments, uform_tape_view_t const &tape,
                        uform_scalar_t element) noexcept {
    if (!arguments.model) return {uform_status_invalid_argument_k, "The model must not be null"};
    requested_t const requested = requested_of(arguments);
    if constexpr (requires { arguments.probabilities; }) {
        if (failure_t const refused = validate_choices(arguments); refused.message) return refused;
    }
    else if (!requested.embeddings && !requested.states && !requested.token_vectors)
        return {uform_status_invalid_argument_k, "Name the embeddings, the states or the token vectors"};
    if (requested.token_vectors && !arguments.model->geometry.token_width)
        return {uform_status_unsupported_k, "The model carries no token head"};
    if (!output_bytes(requested.scalar)) return {uform_status_unsupported_k, "Outputs land as f32, f16 or bf16 only"};
    return validate_tape(tape, arguments.count, element);
}

/** Turns the decoder's @p logits at each item's mask into probabilities over its choices alone, at
 *  its group's inverse temperature, on the host, and lands them in @p memory through @p site . */
template <typename site_type_>
failure_t land_choices(uform_text_choose_t &arguments, core::f32r2_view_t logits, site_type_ &site,
                       chunked_arena_t &memory) noexcept {
    uform_model_handle_t const &model = *arguments.model;
    std::size_t const count = arguments.count;
    core::tape_view<u8_t const> const choices = tape_of<u8_t const>(arguments.choices, count);
    std::size_t const origin = choices.offsets[0], total = choices.offsets[count] - origin;
    std::size_t *const offsets_memory = site.transfers.template allocate<std::size_t>(count + 1);
    f32_t *const probabilities_memory = site.transfers.template allocate<f32_t>(total ? total : 1);
    std::size_t *const landed_offsets_memory = memory.template allocate<std::size_t>(count + 1);
    f32_t *const landed_memory = memory.template allocate<f32_t>(total ? total : 1);
    if (!offsets_memory || !probabilities_memory || !landed_offsets_memory || !landed_memory)
        return {uform_status_out_of_memory_k, "Failed to allocate the probabilities"};
    std::span<std::size_t> const offsets {offsets_memory, count + 1};
    std::span<f32_t> const probabilities {probabilities_memory, total};
    std::span<std::size_t> const landed_offsets {landed_offsets_memory, count + 1};
    std::span<f32_t> const landed {landed_memory, total};
    for (std::size_t item = 0; item != count; ++item) {
        std::span<u8_t const> const offered = choices[item];
        std::span<f32_t> const chances = probabilities.subspan(choices.offsets[item] - origin, offered.size());
        f32_t const *const row = logits.row(item).data();
        f32_t const beta =
            model.choice_inverse_temperatures[arguments.temperature_groups ? arguments.temperature_groups[item] : 0];
        f32_t peak = -std::numeric_limits<f32_t>::infinity(), sum = 0.0f;
        for (u8_t const byte : offered) peak = std::max(peak, row[byte]);
        for (std::size_t choice = 0; choice != offered.size(); ++choice)
            sum += chances[choice] = std::exp(beta * (row[offered[choice]] - peak));
        for (f32_t &chance : chances) chance /= sum;
        offsets[item] = choices.offsets[item] - origin;
    }
    offsets[count] = total;
    failure_t failure = site.land(std::as_writable_bytes(landed_offsets), std::as_bytes(offsets));
    if (!failure.message) failure = site.land(std::as_writable_bytes(landed), std::as_bytes(probabilities));
    if (!failure.message) failure = site.finish();
    if (!failure.message) *arguments.probabilities = {landed.data(), landed_offsets.data(), uform_scalar_f32_k};
    return failure;
}

/** Runs @p step for every item of @c [begin,end) on @p pool , answering the first failure any reported. */
template <typename step_type_>
failure_t for_items(fu::flat_pool_t &pool, std::size_t begin, std::size_t end, step_type_ step) noexcept {
    failure_t first {};
    fu::spin_mutex_t guard;
    pool.for_n(end - begin, [&](std::size_t task, auto at) noexcept {
        failure_t const failure = step(begin + task, static_cast<std::size_t>(at.thread));
        if (!failure.message) return;
        st::unique_lock<fu::spin_mutex_t> lock(guard);
        if (!first.message) first = failure;
    });
    return first;
}

/**
 *  @brief The body every encode verb shares once its arena is claimed: plan, size the outputs, pack and encode each
 *      slice on @p site , land the views and the stats.
 *  @param[in] memory The call's arena, where the outputs land.
 *
 *  Each slice finishes before the next one packs, so the pinned pages a device still reads are never overwritten,
 *  and a progress report means its items are encoded.
 */
template <typename arguments_type_, typename packer_type_, typename site_type_>
void encode_items(arguments_type_ &arguments, packer_type_ &packer, site_type_ &site, chunked_arena_t &memory,
                  uform_outcome_t *outcome) noexcept {
    ue::geometry_t const &geometry = arguments.model->geometry;
    chunked_arena_t &staging = site.staging, &transfers = site.transfers;
    std::size_t const count = arguments.count, threads = site.pool.threads_count();
    auto const fail = [&](failure_t failure) noexcept { propagate_error(failure.status, outcome, failure.message); };

    if (failure_t const prepared = packer.prepare(staging, threads, geometry); prepared.message) return fail(prepared);
    item_plan_t *const plans = staging.template allocate<item_plan_t>(count ? count : 1);
    if (!plans) return fail({uform_status_out_of_memory_k, "Failed to allocate the plans"});
    for (std::size_t item = 0; item != count; ++item) plans[item] = {};
    failure_t const planned = for_items(site.pool, 0, count, [&](std::size_t item, std::size_t thread) noexcept {
        return packer.plan(item, plans[item], geometry, thread);
    });
    if (planned.message) return fail(planned);

    std::size_t tokens = 0, truncated = 0;
    for (std::size_t item = 0; item != count; ++item)
        tokens += plans[item].tokens, truncated += plans[item].spanned > plans[item].tokens;

    requested_t const requested = requested_of(arguments);
    std::size_t const row_bytes = geometry.hidden * output_bytes(requested.scalar);
    std::size_t const token_row_bytes = geometry.token_width * output_bytes(requested.scalar);
    bool const ragged = requested.states || requested.token_vectors;
    std::size_t *const offsets_memory = staging.template allocate<std::size_t>(count + 1);
    std::size_t *const landed_offsets_memory = ragged ? memory.template allocate<std::size_t>(count + 1) : nullptr;
    auto const output = [&](void *requested_view, chunked_arena_t &into, std::size_t bytes) noexcept {
        return requested_view ? static_cast<b8_t *>(into.allocate(bytes ? bytes : 1)) : nullptr;
    };
    b8_t *const embeddings = output(requested.embeddings, memory, count * row_bytes);
    b8_t *const states = output(requested.states, memory, tokens * row_bytes);
    b8_t *const token_vectors = output(requested.token_vectors, memory, tokens * token_row_bytes);
    f32_t *masked_logits = nullptr; // in host pages, where the choices' softmax reads them
    if constexpr (requires { arguments.probabilities; })
        masked_logits = transfers.template allocate<f32_t>(count * geometry.masked_vocabulary + 1);
    if (!offsets_memory || (ragged && !landed_offsets_memory) || (requested.embeddings && !embeddings) ||
        (requested.states && !states) || (requested.token_vectors && !token_vectors) ||
        (requires { arguments.probabilities; } && !masked_logits))
        return fail({uform_status_out_of_memory_k, "Failed to allocate the outputs"});
    std::span<std::size_t> const offsets {offsets_memory, count + 1};
    std::span<std::size_t> const landed_offsets {landed_offsets_memory, ragged ? count + 1 : 0};
    nk_dtype_t const dtype = output_dtype(requested.scalar);
    ue::outputs_t const surfaces {{nk::tensor_span<b8_t, 2> {embeddings, count, row_bytes}, dtype},
                                  {nk::tensor_span<b8_t, 2> {states, tokens, row_bytes}, dtype},
                                  {nk::tensor_span<b8_t, 2> {token_vectors, tokens, token_row_bytes}, dtype},
                                  {masked_logits, count, geometry.masked_vocabulary}};
    offsets[0] = 0;
    for (std::size_t item = 0; item != count; ++item) offsets[item + 1] = offsets[item] + plans[item].tokens;

    progress_ticks_t ticks {arguments.progress, count};
    std::size_t widest_items = 0, widest_tokens = 0;
    for (slice_t const slice : ticks) {
        std::size_t const items = slice.end - slice.begin, cells = offsets[slice.end] - offsets[slice.begin];
        widest_items = items > widest_items ? items : widest_items;
        widest_tokens = cells > widest_tokens ? cells : widest_tokens;
    }
    std::size_t const cell_bytes = packer.cell_bytes(geometry);
    u32_t *const segment_offsets = transfers.template allocate<u32_t>(widest_items + 1);
    u32_t *const segment_lengths = transfers.template allocate<u32_t>(widest_items + 1);
    modality_t *const segment_modality = transfers.template allocate<modality_t>(widest_items + 1);
    u16_t *const image_shapes = transfers.template allocate<u16_t>(widest_items * 2 + 2);
    u16_t *const video_shapes = transfers.template allocate<u16_t>(widest_items * 3 + 3);
    b8_t *const cells_memory = static_cast<b8_t *>(transfers.allocate(widest_tokens * cell_bytes + 1));
    b8_t *const zero_row_memory = static_cast<b8_t *>(transfers.allocate(row_bytes));
    if (!segment_offsets || !segment_lengths || !segment_modality || !image_shapes || !video_shapes || !cells_memory ||
        !zero_row_memory)
        return fail({uform_status_out_of_memory_k, "Failed to allocate the batch"});
    bytes_span_t const cells {cells_memory, widest_tokens * cell_bytes};
    bytes_span_t const zero_row {zero_row_memory, row_bytes};
    nk::tensor_span<u16_t, 2> const image_shape_rows {image_shapes, widest_items + 1, 2};
    nk::tensor_span<u16_t, 2> const video_shape_rows {video_shapes, widest_items + 1, 3};
    std::memset(zero_row.data(), 0, zero_row.size());
    if (widest_tokens)
        if (failure_t const reserved = site.reserve(widest_tokens, widest_items, cell_bytes, surfaces);
            reserved.message)
            return fail(reserved);

    for (slice_t const slice : ticks) {
        std::size_t const items = slice.end - slice.begin, first_row = offsets[slice.begin];
        std::size_t const slice_tokens = offsets[slice.end] - first_row;
        for (std::size_t item = 0; item != items; ++item) {
            item_plan_t const &plan = plans[slice.begin + item];
            segment_offsets[item] = static_cast<u32_t>(offsets[slice.begin + item] - first_row);
            segment_lengths[item] = plan.tokens;
            segment_modality[item] = packer_type_::modality_k;
            for (std::size_t axis = 0; axis != 2; ++axis) image_shape_rows[item, axis] = plan.shape[axis];
            for (std::size_t axis = 0; axis != 3; ++axis) video_shape_rows[item, axis] = plan.shape[axis];
        }
        segment_offsets[items] = static_cast<u32_t>(slice_tokens);
        failure_t const staged = for_items(
            site.pool, slice.begin, slice.end, [&](std::size_t item, std::size_t thread) noexcept {
                return packer.stage(
                    item, plans[item],
                    cells.subspan((offsets[item] - first_row) * cell_bytes, plans[item].tokens * cell_bytes), geometry,
                    thread);
            });
        if (staged.message) return fail(staged);

        if (slice_tokens) {
            std::size_t const side = geometry.image_patch_side, video_side = geometry.video_patch_side;
            core::multimodal_batch_view_t batch;
            batch.segment_offsets = {segment_offsets, items + 1};
            batch.segment_lengths = {segment_lengths, items};
            batch.segment_modality = {segment_modality, items};
            batch.image_shapes = {image_shapes, items, 2};
            batch.video_shapes = {video_shapes, items, 3};
            void *const content = cells.data();
            switch (packer_type_::modality_k) {
            case modality_t::text_k: batch.tokens = {static_cast<token_id_t *>(content), 1, slice_tokens}; break;
            case modality_t::image_k:
                batch.images = {static_cast<pixel_t *>(content),
                                {1, slice_tokens, side, side, geometry.image_channels}};
                break;
            case modality_t::video_k:
                batch.videos = {
                    static_cast<pixel_t *>(content),
                    {1, slice_tokens, geometry.video_tube_frames, video_side, video_side, geometry.video_channels}};
                break;
            case modality_t::audio_k:
                batch.audio_samples = {static_cast<f32_t *>(content), {1, slice_tokens, geometry.audio_chunk_samples}};
                break;
            }
            ue::outputs_t outputs {surfaces.embeddings.slice(slice.begin, items),
                                   surfaces.states.slice(first_row, slice_tokens),
                                   surfaces.token_vectors.slice(first_row, slice_tokens),
                                   {}};
            if (masked_logits)
                outputs.masked_logits = surfaces.masked_logits[nk::range(slice.begin, slice.end), nk::slice];
            if (failure_t const encoded = site.encode(batch, cells, offsets.subspan(slice.begin, items + 1), outputs);
                encoded.message)
                return fail(encoded);
        }
        else if (embeddings) // every item of the slice is empty, which the tower refuses, and each pools to a zero row
            for (std::size_t item = slice.begin; item != slice.end; ++item)
                if (failure_t const landed = site.land(core::bytes_of(surfaces.embeddings.bytes.row(item)), zero_row);
                    landed.message)
                    return fail(landed);
        if (failure_t const finished = site.finish(); finished.message) return fail(finished);
        if (ticks(slice.end) == continuation_t::cancel_k)
            return fail({uform_status_cancelled_k, "The caller's handler stopped the call"});
    }
    if (ragged)
        if (failure_t const landed = site.land(std::as_writable_bytes(landed_offsets), std::as_bytes(offsets));
            landed.message)
            return fail(landed);
    if (failure_t const finished = site.finish(); finished.message) return fail(finished);
    if constexpr (requires { arguments.probabilities; })
        if (failure_t const chosen = land_choices(arguments, surfaces.masked_logits, site, memory); chosen.message)
            return fail(chosen);

    uform_scalar_t const scalar = requested.scalar;
    if (requested.embeddings) *requested.embeddings = {embeddings, row_bytes, geometry.hidden, scalar};
    if (requested.states) *requested.states = {{states, row_bytes, geometry.hidden, scalar}, landed_offsets.data()};
    if (requested.token_vectors)
        *requested.token_vectors = {{token_vectors, token_row_bytes, geometry.token_width, scalar},
                                    landed_offsets.data()};
    if (arguments.stats) *arguments.stats = {truncated, tokens};
    propagate_error(uform_status_success_k, outcome, nullptr);
}

/** A VRAM call through @p verbs : one shard per device of @p devices holding one of
 *  @p replicas . */
template <encoder_home_t home_, typename arguments_type_, typename packer_type_>
void encode_on_devices(arguments_type_ &arguments, packer_type_ &packer, encoder_verbs<home_> const &verbs,
                       per_gpu_array<vram_replica_t> const &replicas, vram_executor_t &devices, chunked_arena_t &memory,
                       call_outcome_t &outcome) noexcept {
    auto const fail = [&](failure_t failure) noexcept { propagate_error(failure.status, outcome, failure.message); };
    vram_device_t &lead = devices.devices[0];
    devices.staging.chunks.rewind();
    lead.transfers.rewind();
    vram_site<home_> site {devices.decoders, devices.staging.chunks, lead.transfers, memory.stream(), verbs};
    for (vram_device_t &device : devices.devices)
        for (vram_replica_t const &replica : replicas)
            if (replica.ordinal == device.ordinal && (device.capabilities & vendor_of(home_)))
                if (site.shards.push_back({&device, &replica}) != core::status_t::success_k)
                    return fail({uform_status_internal_k, "More devices than one process drives"});
    if (!site.shards.size()) return fail({uform_status_device_k, "The model holds no replica on the executor"});
    encode_items(arguments, packer, site, memory, outcome);
    site.finish(); // a failed slice leaves work queued, which drains before the next call reuses what it reads
}

/** Every encode verb after its entry: refuse, claim the arena, pick where the call computes, and encode. */
template <typename arguments_type_, typename packer_type_>
void encode_call(arguments_type_ &arguments, packer_type_ &packer, uform_tape_view_t const &tape,
                 uform_scalar_t element, std::initializer_list<void const *> inputs, call_outcome_t &outcome) noexcept {
    clear_outputs(arguments);
    auto const fail = [&](failure_t failure) noexcept { propagate_error(failure.status, outcome, failure.message); };
    if (failure_t const refused = validate_call(arguments, tape, element); refused.message) return fail(refused);
    if (failure_t const refused = packer.validate(arguments.model->geometry); refused.message) return fail(refused);
    uform_arena_t const arena = claim_arena(arguments.arena, arguments.executor, inputs, outcome);
    if (!arena) return;
    chunked_arena_t &memory = chunks_of(arena->memory);
    uform_executor_t const executor = arguments.executor;
    uform_model_handle_t const &model = *arguments.model;

    std::visit(
        [&]<encoder_home_t home_>(encoder_verbs<home_> const *verbs) noexcept {
            if constexpr (home_ == encoder_home_t::ram_k) {
                void *const tower = std::get<void *>(model.towers);
                if (executor && !std::holds_alternative<ram_executor_t>(executor->executor))
                    return fail({uform_status_device_k, "A RAM model computes on a RAM executor or none"});
                if (executor) {
                    ram_site_t site {std::get<ram_executor_t>(executor->executor).pool, memory, memory, *verbs, tower};
                    return encode_items(arguments, packer, site, memory, outcome);
                }
                fu::flat_pool_t caller;
                if (fu::failed(caller.spawn(1)))
                    return fail({uform_status_out_of_memory_k, "Failed to set up the caller"});
                ram_site_t site {caller, memory, memory, *verbs, tower};
                return encode_items(arguments, packer, site, memory, outcome);
            }
            else if constexpr (!gpu_vendor_compiled(vendor_of(home_)))
                return fail({uform_status_unsupported_k, "This build drives no GPU of the model's vendor"});
            else {
                vram_executor_t *const devices = executor ? std::get_if<vram_executor_t>(&executor->executor) : nullptr;
                if (!devices) return fail({uform_status_device_k, "A VRAM model computes on a VRAM executor"});
                encode_on_devices(arguments, packer, *verbs, std::get<per_gpu_array<vram_replica_t>>(model.towers),
                                  *devices, memory, outcome);
            }
        },
        model.tables);
}

#pragma endregion Calls

} // namespace unum::uform
