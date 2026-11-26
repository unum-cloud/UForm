/**
 *  @file src/encode.hpp
 *  @author Ash Vardanian
 *  @date October 17, 2026
 *  @brief What every encode verb shares once its packer is chosen: the plans, the sites a call
 *      computes on, and the body that sizes the outputs, packs each slice and runs the model's
 *      tower over it.
 *
 *  Included by @c encode.cpp, whose verbs take decoded inputs, and by @c encode_media.cpp, whose
 *  verbs decode files, so a build without the codecs links the first alone.
 *
 *  A RAM call packs its batch and reserves the tower's scratch from its own arena, which keeps the
 *  high-water mark. A VRAM call plans in the executor's host pages and packs what the devices read
 *  into its first device's pinned pages; each device copies its run of the slice into its own
 *  scratch, writes its share of the outputs there, and copies it into the arena when the arena
 *  lives on another device.
 */
#pragma once

#include <cmath>   // `std::exp` - a choice's softmax
#include <cstring> // `std::memcpy`, `std::memset`

#include <initializer_list> // `std::initializer_list` - the inputs a call checks against its arena
#include <limits>           // `std::numeric_limits` - the peak a choice's softmax starts below
#include <span>             // `std::span` - an item's bytes as the decoders take them

#include <forkunion/locks.hpp> // `fu::spin_mutex_t` - the guard over a batch's results

#include "uform/media/image.hpp" // `image_plan_t`, `image_extent_t`, `image_channels_k`

#include "encoder_punned.hpp" // `encoder_verbs` - what a model's towers answer to

namespace unum::uform {

namespace ue = unum::uform::encoder;
namespace um = unum::uform::media;

using encoder::modality_t;

#pragma region Plans

/** A failure the steps of a call report before it is final, with its static message. */
struct failure_t {
    uform_status_t status = uform_status_success_k;
    char const* message = nullptr;
};

/** The C failure a media decoder's @p status crosses into, with the decoder's note or else the
 *  static @p note : whatever the bytes do not decode to is malformed. */
inline failure_t media_failure(util::status_t status, char const* note) noexcept {
    char const* const noted = status.note ? status.note : note;
    switch (status.code) {
    case util::status_code_t::out_of_memory_k: return {uform_status_out_of_memory_k, noted};
    case util::status_code_t::unsupported_k: return {uform_status_unsupported_k, noted};
    default: return {uform_status_malformed_k, noted};
    }
}

/** What the plan of a call decides about one item before any slice packs it. */
struct item_plan_t {

    /** Tokens the item keeps after truncation; an empty item keeps none and pools to a zero row. */
    util::u32_t tokens = 0;

    /** Tokens the item spans before truncation. */
    std::size_t spanned = 0;

    /** The factorization its coordinates read: an image's @c (rows,columns) , a clip's
     *  @c (groups,rows,columns) . */
    util::u16_t shape[3] = {};

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

/** The refusals of every image verb: a model without the image entry, or one whose pixels
 *  are not RGBA. */
inline failure_t validate_images(ue::geometry_t const& geometry) noexcept {
    if (!geometry.image_patch_side) return {uform_status_unsupported_k, "The model was not trained on images"};
    if (geometry.image_channels != um::image_channels_k)
        return {uform_status_unsupported_k, "The model's images are not RGBA"};
    return {};
}

/** The patch count an image of @p native extent spans before truncation. */
inline std::size_t native_patches(um::image_extent_t native, util::u32_t patch_side) noexcept {
    return util::divide_round_up(std::size_t {native.width}, patch_side) *
           util::divide_round_up(std::size_t {native.height}, patch_side);
}

/** The refusal of every audio verb: a model without the audio entry. */
inline failure_t validate_audio(ue::geometry_t const& geometry) noexcept {
    if (!geometry.audio_chunk_samples || !geometry.audio_sample_rate)
        return {uform_status_unsupported_k, "The model was not trained on audio"};
    return {};
}

#pragma endregion Plans

#pragma region Sites

/** The C status a tower's failure crosses into: a batch it calls invalid is a packing defect of
 *  the encode verbs. */
inline failure_t tower_failure(util::status_t status) noexcept {
    switch (status.code) {
    case util::status_code_t::invalid_argument_k:
        return {uform_status_internal_k, status.note ? status.note : "The tower refused a packed batch"};
    case util::status_code_t::out_of_memory_k:
        return {uform_status_out_of_memory_k, status.note ? status.note : "The tower ran out of memory"};
    case util::status_code_t::unsupported_k:
        return {uform_status_unsupported_k,
                status.note ? status.note : "No kernel of the model's class takes the batch's shapes"};
    default:
        return {status_of(status), status.note ? status.note : "The tower or its device failed to encode the batch"};
    }
}

/** @p view over @p data in place of its own storage, its shape kept; an absent
 *  view stays absent. */
template <typename view_type_>
view_type_ rebased(view_type_ view, void const* data) noexcept {
    if (!view.byte_data()) return view;
    return view_type_ {static_cast<char const*>(data), view.shape()};
}

/** @p rows over @p staging in place of their own storage, their shape kept; absent
 *  rows stay absent. */
inline ue::rows_span_t rebased(ue::rows_span_t rows, util::bytes_span_t staging) noexcept {
    if (rows.empty()) return rows;
    return {nk::tensor_span<util::b8_t, 2> {staging.data(), rows.bytes.extent(0), rows.bytes.extent(1)}, rows.dtype};
}

/** A failure of the engine's own vocabulary, with its note or else the static @p note . */
inline failure_t failure_of(util::status_t status, char const* note) noexcept {
    return {status_of(status), status.note ? status.note : note};
}

/** A RAM call: the host tower on a pool, packing into the call's own arena, which also holds the
 *  staging, the transfers and the scratch, each reserved before it is allocated. */
struct ram_site_t {
    fu::pool_t& pool;
    uform_arena_handle_t& arena;

    /** What only the host reads, and what the tower reads, both the call's arena on the host. */
    chunked_arena_t& staging;
    chunked_arena_t& transfers;

    /** The variant the model's load picked, and its one host tower. */
    encoder_verbs<vendored_residency_t::ram_k> const& verbs;
    void const* tower = nullptr;
    util::bytes_span_t scratch {};

    /** Makes room for @p bytes of host allocations in the call's arena. */
    failure_t reserve_staging(std::size_t bytes) noexcept {
        util::status_t const reserved = reserve_arena(arena, bytes);
        return reserved.failed() ? failure_of(reserved, "Failed to allocate the staging") : failure_t {};
    }

    /** Reserves the tower's scratch for @p tokens live tokens in @p segments items, with
     *  @p transfer_bytes of transfers beside it, and allocates the scratch. */
    failure_t reserve(std::size_t tokens, std::size_t segments, std::size_t, ue::outputs_t, std::size_t,
                      std::size_t transfer_bytes) noexcept {
        std::size_t bytes = 0;
        if (tokens) {
            auto [needed, status] = verbs.scratch_bytes(tower, tokens, segments, pool);
            if (!status) return tower_failure(status.status_);
            bytes = needed;
        }
        if (failure_t const reserved = reserve_staging(chunked_arena_t::footprint(bytes) + transfer_bytes);
            reserved.message)
            return reserved;
        util::b8_t* const data = static_cast<util::b8_t*>(staging.allocate(bytes));
        if (!data && bytes) return {uform_status_out_of_memory_k, "Failed to allocate the activations"};
        scratch = {data, bytes};
        return {};
    }

    failure_t encode(ue::multimodal_batch_view_t const& batch, util::bytes_view_t, std::span<std::size_t const>,
                     ue::outputs_t outputs) noexcept {
        util::expected_t encoded = verbs.encode(tower, batch, outputs, scratch, pool);
        return encoded ? failure_t {} : tower_failure(encoded.status_);
    }

    failure_t land(util::bytes_span_t target, util::bytes_view_t source) noexcept {
        std::memcpy(target.data(), source.data(), source.size());
        return {};
    }
    failure_t finish() noexcept { return {}; }
};

/** One stream of a VRAM call: the executor's stream record, the model's replica on its device, and
 *  its regions of the stream's activations. */
template <typename stream_type_>
struct vram_shard {
    stream_type_* stream = nullptr;
    vram_replica_t const* replica = nullptr;
    util::bytes_span_t scratch {};

    /** Room for a whole slice's cells, of which the shard copies in only its own run, at
     *  the run's offset. */
    util::bytes_span_t cells {};

    /** @b [segments,2] and @b [segments,3] the media shapes of the shard's run. */
    nk::tensor_span<util::u16_t, 2> image_shapes {}, video_shapes {};

    /** Where a device other than the arena's writes its outputs before copying them over; empty
     *  on the arena's. */
    util::bytes_span_t embeddings {}, states {}, token_vectors {};

    /** @b [segments,masked_vocabulary] where every shard writes its logits before copying them
     *  into host pages. */
    util::f32r2_span_t masked_logits {};
};

/** A VRAM call: plans on the decoder pool into the executor's host pages, packs into the lead
 *  stream's pinned pages, and splits each slice across the streams whose devices hold a replica, by
 *  token count, through the verbs of the variant @p residency_ 's load picked, moving each shard's
 *  run in and its outputs out through the vendor's copies. */
template <vendored_residency_t residency_>
struct vram_site {
    using entries_kit_t = entries_kit_of<residency_>;
    using stream_t = typename executor_of<residency_>::stream_t;
    using shard_t = vram_shard<stream_t>;
    static constexpr failure_t unbound_k {uform_status_device_k, "A device failed to become current"};

    fu::pool_t& pool;
    chunked_arena_t& staging;
    chunked_arena_t& transfers;

    /** The executor's first stream, on whose device the call's arena and pinned pages live, and
     *  whose shard writes the arena directly. */
    stream_t& lead;
    encoder_verbs<residency_> const& verbs;
    util::per_gpu_array<shard_t> shards {};
    std::size_t cell_bytes = 0;

    /** Makes room for @p bytes of host allocations in the executor's staging pages. */
    failure_t reserve_staging(std::size_t bytes) noexcept {
        util::status_t const reserved = util::reserve_host_pages(staging, bytes);
        return reserved.failed() ? failure_of(reserved, "Failed to allocate the staging") : failure_t {};
    }

    /** Reserves every shard's activations for @p tokens live tokens in @p segments items, and the
     *  lead stream's @p transfer_bytes of pinned pages, then allocates each shard's regions. */
    failure_t reserve(std::size_t tokens, std::size_t segments, std::size_t bytes_per_cell, ue::outputs_t surfaces,
                      std::size_t masked_vocabulary, std::size_t transfer_bytes) noexcept {
        cell_bytes = bytes_per_cell;
        std::size_t lead_reserved = 0;
        for (shard_t& shard : shards) {
            stream_t& stream = *shard.stream;
            if (stream.bind().failed()) return unbound_k;
            std::size_t bytes = 0;
            if (tokens) {
                auto [needed, status] = verbs.scratch_bytes(shard.replica->tower, tokens, segments, stream);
                if (!status) return tower_failure(status.status_);
                bytes = needed;
            }
            bool const elsewhere = &stream != &lead;
            auto const staged_bytes = [&](ue::rows_span_t surface, std::size_t rows) noexcept {
                return elsewhere && !surface.empty() ? rows * surface.bytes.extent(1) : 0;
            };
            std::size_t const footprint = chunked_arena_t::footprint(bytes) +
                                          chunked_arena_t::footprint(tokens * cell_bytes) +
                                          chunked_arena_t::footprint<util::u16_t>(segments * 2) +
                                          chunked_arena_t::footprint<util::u16_t>(segments * 3) +
                                          chunked_arena_t::footprint(staged_bytes(surfaces.embeddings, segments)) +
                                          chunked_arena_t::footprint(staged_bytes(surfaces.states, tokens)) +
                                          chunked_arena_t::footprint(staged_bytes(surfaces.token_vectors, tokens)) +
                                          chunked_arena_t::footprint<util::f32_t>(segments * masked_vocabulary);
            std::size_t const pinned = elsewhere ? 0 : transfer_bytes;
            if (util::status_t const reserved = stream.memory.reserve(stream, footprint, pinned); reserved.failed())
                return failure_of(reserved, "The executor's scratch limit is too low");
            lead_reserved += !elsewhere;
            chunked_arena_t& memory = stream.memory.activations;
            failure_t failure {};
            auto const allocate = [&](std::size_t size) noexcept {
                util::b8_t* const data = static_cast<util::b8_t*>(memory.allocate(size));
                if (!data && size) failure = {uform_status_out_of_memory_k, "The executor's scratch limit is too low"};
                return util::bytes_span_t {data, data ? size : 0};
            };
            auto const shapes = [&](std::size_t axes) noexcept {
                util::bytes_span_t const shape_bytes = allocate(segments * axes * sizeof(util::u16_t));
                return nk::as_writable_values<util::u16_t>(shape_bytes).reshape<2>({segments, axes});
            };
            shard.scratch = allocate(bytes);
            shard.cells = allocate(tokens * cell_bytes);
            shard.image_shapes = shapes(2);
            shard.video_shapes = shapes(3);
            shard.embeddings = allocate(staged_bytes(surfaces.embeddings, segments));
            shard.states = allocate(staged_bytes(surfaces.states, tokens));
            shard.token_vectors = allocate(staged_bytes(surfaces.token_vectors, tokens));
            if (masked_vocabulary) {
                util::bytes_span_t const logits = allocate(segments * masked_vocabulary * sizeof(util::f32_t));
                shard.masked_logits = nk::as_writable_values<util::f32_t>(logits).reshape<2>(
                    {segments, masked_vocabulary});
            }
            if (failure.message) return failure;
        }
        // A lead holding no replica still holds the pinned pages
        if (!lead_reserved) {
            if (lead.bind().failed()) return unbound_k;
            if (util::status_t const reserved = lead.memory.reserve(lead, 0, transfer_bytes); reserved.failed())
                return failure_of(reserved, "Failed to allocate the pinned pages");
        }
        return {};
    }

    /** Runs the items of @p batch , whose cells @p cells hold, on every shard, each taking a
     *  contiguous run of about equal tokens, their outputs landing at the run's rows. */
    failure_t encode(ue::multimodal_batch_view_t const& batch, util::bytes_view_t cells,
                     std::span<std::size_t const> packed_offsets, ue::outputs_t outputs) noexcept {
        std::size_t const segments = packed_offsets.size() - 1, origin = packed_offsets[0];
        std::size_t const tokens = packed_offsets[segments] - origin, count = shards.size();
        failure_t const moved {uform_status_device_k, "A device failed to copy a batch or its outputs"};
        std::size_t first = 0;
        for (std::size_t index = 0; index != count && first != segments; ++index) {
            std::size_t last = first;
            std::size_t const target = util::divide_round_up((index + 1) * tokens, count);
            while (last != segments && (packed_offsets[last + 1] - origin <= target || index + 1 == count)) ++last;
            if (last == first) continue;
            shard_t const& shard = shards[index];
            if (shard.stream->bind().failed()) return unbound_k;
            auto& stream = *shard.stream;
            auto const copied = [&](util::bytes_span_t into, util::bytes_view_t from) noexcept {
                return from.empty() || entries_kit_t::copy(into, from, stream) == nk_success_k;
            };
            std::size_t const begin = packed_offsets[first] - origin, end = packed_offsets[last] - origin;
            std::size_t const items = last - first;
            auto const shard_items = nk::range(first, last);
            nk::tensor_span<util::u16_t, 2> const image_shapes = shard.image_shapes[nk::range(0, items), nk::slice];
            nk::tensor_span<util::u16_t, 2> const video_shapes = shard.video_shapes[nk::range(0, items), nk::slice];
            std::size_t const run_first = begin * cell_bytes, run_bytes = (end - begin) * cell_bytes;
            if (!copied(shard.cells.subspan(run_first, run_bytes), cells.subspan(run_first, run_bytes)) ||
                !copied(nk::as_writable_bytes(image_shapes).values().value,
                        nk::as_bytes(batch.image_shapes[shard_items, nk::slice]).values().value) ||
                !copied(nk::as_writable_bytes(video_shapes).values().value,
                        nk::as_bytes(batch.video_shapes[shard_items, nk::slice]).values().value))
                return moved;

            ue::multimodal_batch_view_t shard_batch = batch;
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
            bool const elsewhere = shard.stream != &lead;
            ue::outputs_t shard_outputs = landing;
            if (elsewhere)
                shard_outputs = {rebased(landing.embeddings, shard.embeddings),
                                 rebased(landing.states, shard.states),
                                 rebased(landing.token_vectors, shard.token_vectors),
                                 {}};
            if (!outputs.masked_logits
                     .empty()) // they land in host pages, so every shard computes them in its own memory
                shard_outputs.masked_logits = shard.masked_logits[nk::range(0, items), nk::slice];
            util::expected_t encoded = verbs.encode(shard.replica->tower, shard_batch, shard_outputs, shard.scratch,
                                                    stream);
            if (!encoded) return tower_failure(encoded.status_);
            if (elsewhere &&
                (!copied(landing.embeddings.bytes.values().value, shard_outputs.embeddings.bytes.values().value) ||
                 !copied(landing.states.bytes.values().value, shard_outputs.states.bytes.values().value) ||
                 !copied(landing.token_vectors.bytes.values().value, shard_outputs.token_vectors.bytes.values().value)))
                return moved;
            if (!outputs.masked_logits.empty() &&
                !copied(nk::as_writable_bytes(outputs.masked_logits[nk::range(first, last), nk::slice]).values().value,
                        nk::as_bytes(shard_outputs.masked_logits).values().value))
                return moved;
            first = last;
        }
        return {};
    }

    /** Copies host bytes into the arena on the lead stream, which @ref finish waits for. */
    failure_t land(util::bytes_span_t target, util::bytes_view_t source) noexcept {
        if (lead.bind().failed()) return unbound_k;
        if (entries_kit_t::copy(target, source, lead) != nk_success_k)
            return {uform_status_device_k, "A device failed to copy into the arena"};
        return {};
    }

    /** Waits for every shard's stream and the lead's, then binds the lead's device, where later
     *  outputs grow. */
    failure_t finish() noexcept {
        failure_t failure {};
        for (shard_t const& shard : shards)
            if (util::status_t const drained = shard.stream->wait(); drained.failed() && !failure.message)
                failure = failure_of(drained, "A device failed the work queued on it");
        if (util::status_t const drained = lead.wait(); drained.failed() && !failure.message)
            failure = failure_of(drained, "A device failed the work queued on it");
        if (lead.bind().failed() && !failure.message) failure = unbound_k;
        return failure;
    }
};

#pragma endregion Sites

#pragma region Calls

/** A choice's arguments, which land probabilities over its choices rather than any surface. */
template <typename arguments_type_>
concept a_choice_call = requires(arguments_type_& arguments) { arguments.probabilities; };

/** The surfaces a call's arguments name, and the scalar they land as. */
struct requested_t {
    uform_rows_view_t* embeddings = nullptr;
    uform_rows_tape_view_t* states = nullptr;
    uform_rows_tape_view_t* token_vectors = nullptr;
    uform_scalar_t scalar = uform_scalar_f32_k;
};

/** The surfaces @p arguments name; a choice names none of them, as it lands probabilities. */
template <typename arguments_type_>
requested_t requested_of(arguments_type_ const& arguments) noexcept {
    if constexpr (a_choice_call<arguments_type_>) return {};
    else return {arguments.embeddings, arguments.states, arguments.token_vectors, arguments.outputs_scalar};
}

/** Zeroes every output and report, so a call refused or failing before it lands
 *  leaves them null. */
template <typename arguments_type_>
void clear_outputs(arguments_type_& arguments) noexcept {
    requested_t const requested = requested_of(arguments);
    if (requested.embeddings) *requested.embeddings = {};
    if (requested.states) *requested.states = {};
    if (requested.token_vectors) *requested.token_vectors = {};
    if constexpr (a_choice_call<arguments_type_>)
        if (arguments.probabilities) *arguments.probabilities = {};
    if (arguments.stats) *arguments.stats = {};
}

/** The refusals @p tape shares with every tape a call reads: storage, ascending offsets and
 *  @p element . */
inline failure_t validate_tape(uform_tape_view_t const& tape, std::size_t count, uform_scalar_t element) noexcept {
    if (!tape.offsets || (!tape.data && count && tape.offsets[count] != tape.offsets[0]))
        return {uform_status_invalid_argument_k, "The tape must name its storage and offsets"};
    if (!offsets_ascend(tape, count)) return {uform_status_invalid_argument_k, "A tape's offsets descend"};
    if (tape.scalar != element) return {uform_status_unsupported_layout_k, "The tape holds another scalar"};
    return {};
}

/** The refusals of a choice beyond its states': the decoder, the choices and the groups. */
inline failure_t validate_choices(uform_text_choose_t const& arguments) noexcept {
    ue::geometry_t const& geometry = arguments.model->geometry;
    if (!geometry.masked_vocabulary) return {uform_status_unsupported_k, "The model has no byte decoder"};
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
failure_t validate_call(arguments_type_ const& arguments, uform_tape_view_t const& tape,
                        uform_scalar_t element) noexcept {
    if (!arguments.model) return {uform_status_uninitialized_k, "The model must not be null"};
    requested_t const requested = requested_of(arguments);
    if constexpr (a_choice_call<arguments_type_>) {
        if (failure_t const refused = validate_choices(arguments); refused.message) return refused;
    }
    else if (!requested.embeddings && !requested.states && !requested.token_vectors)
        return {uform_status_invalid_argument_k, "Name the embeddings, the states or the token vectors"};
    if (requested.token_vectors && !arguments.model->geometry.token_width)
        return {uform_status_unsupported_k, "The model has no token head"};
    if (!output_bytes(requested.scalar)) return {uform_status_unsupported_k, "Outputs land as f32, f16 or bf16 only"};
    return validate_tape(tape, arguments.count, element);
}

/** The values every choice of @p arguments offers together, which its probabilities take. */
inline std::size_t choice_values(uform_text_choose_t const& arguments) noexcept {
    util::tape_view<util::u8_t const> const choices = tape_of<util::u8_t const>(arguments.choices, arguments.count);
    return choices.offsets[arguments.count] - choices.offsets[0];
}

/** The footprint the probabilities of @p count items over @p values choices take, on either side:
 *  their offsets, then the values. */
inline std::size_t choice_footprint(std::size_t count, std::size_t values) noexcept {
    return chunked_arena_t::footprint<std::size_t>(count + 1) +
           chunked_arena_t::footprint<util::f32_t>(values ? values : 1);
}

/** Where a choice lands its probabilities: their offsets and values in host pages, where the
 *  softmax writes them, and in the call's arena, where they land; each pair is allocated with the
 *  reservation it was counted in. */
struct choice_buffers_t {
    std::span<std::size_t> offsets {}, landed_offsets {};
    std::span<util::f32_t> probabilities {}, landed {};
};

/** Allocates a pair of buffers for @p count items over @p values choices from @p arena , which the
 *  call reserved for them, or empty spans when it is full. */
inline void allocate_choice_pair(chunked_arena_t& arena, std::size_t count, std::size_t values,
                                 std::span<std::size_t>& offsets, std::span<util::f32_t>& probabilities) noexcept {
    std::size_t* const offsets_memory = arena.template allocate<std::size_t>(count + 1);
    util::f32_t* const values_memory = arena.template allocate<util::f32_t>(values ? values : 1);
    if (offsets_memory && values_memory) offsets = {offsets_memory, count + 1}, probabilities = {values_memory, values};
}

/** Turns the decoder's @p logits at each item's mask into probabilities over its choices alone, at
 *  its group's inverse temperature, on the host, and lands them in @p memory through @p site . */
template <typename site_type_>
failure_t land_choices(uform_text_choose_t& arguments, util::f32r2_view_t logits, site_type_& site,
                       choice_buffers_t const& buffers) noexcept {
    uform_model_handle_t const& model = *arguments.model;
    std::size_t const count = arguments.count;
    util::tape_view<util::u8_t const> const choices = tape_of<util::u8_t const>(arguments.choices, count);
    std::size_t const origin = choices.offsets[0], total = choices.offsets[count] - origin;
    std::span<std::size_t> const offsets = buffers.offsets;
    std::span<util::f32_t> const probabilities = buffers.probabilities.first(total);
    std::span<std::size_t> const landed_offsets = buffers.landed_offsets;
    std::span<util::f32_t> const landed = buffers.landed.first(total);
    for (std::size_t item = 0; item != count; ++item) {
        std::span<util::u8_t const> const offered = choices[item];
        std::span<util::f32_t> const chances = probabilities.subspan(choices.offsets[item] - origin, offered.size());
        util::f32_t const* const row = logits.row(item).data();
        util::f32_t const beta =
            model.choice_inverse_temperatures[arguments.temperature_groups ? arguments.temperature_groups[item] : 0];
        util::f32_t peak = -std::numeric_limits<util::f32_t>::infinity(), sum = 0.0f;
        for (util::u8_t const byte : offered) peak = fu::max_of(peak, row[byte]);
        for (std::size_t choice = 0; choice != offered.size(); ++choice)
            sum += chances[choice] = std::exp(beta * (row[offered[choice]] - peak));
        for (util::f32_t& chance : chances) chance /= sum;
        offsets[item] = choices.offsets[item] - origin;
    }
    offsets[count] = total;
    failure_t failure = site.land(std::as_writable_bytes(landed_offsets), std::as_bytes(offsets));
    if (!failure.message) failure = site.land(std::as_writable_bytes(landed), std::as_bytes(probabilities));
    if (!failure.message) failure = site.finish();
    if (!failure.message) *arguments.probabilities = {landed.data(), landed_offsets.data(), uform_scalar_f32_k};
    return failure;
}

/** Runs @p step for every item of @c [begin,end) on @p pool , answering the first
 *  failure any reported. */
template <typename step_type_>
failure_t for_items(fu::pool_t& pool, std::size_t begin, std::size_t end, step_type_ step) noexcept {
    failure_t first {};
    fu::spin_mutex_t guard;
    util::status_t const dispatched = util::status_from_(
        pool.for_n(end - begin, [&](std::size_t task, std::size_t thread) noexcept {
            failure_t const failure = step(begin + task, thread);
            if (!failure.message) return;
            st::unique_lock<fu::spin_mutex_t> lock(guard);
            if (!first.message) first = failure;
        }));
    if (dispatched != util::status_code_t::success_k)
        return {status_of(dispatched), "The executor's pool refused the batch"};
    return first;
}

/** Lands @p zero_row as the embedding of every item of @p slice , all of them empty, which the
 *  tower refuses and each pools to a zero row. */
template <typename site_type_>
failure_t land_zero_rows(site_type_& site, ue::rows_span_t embeddings, slice_t slice,
                         util::bytes_span_t zero_row) noexcept {
    for (std::size_t item = slice.begin; item != slice.end; ++item)
        if (failure_t const landed = site.land(embeddings.bytes.row(item).values().value, zero_row); landed.message)
            return landed;
    return {};
}

/**
 *  @brief The body every encode verb shares once its arena is claimed: plan, size the outputs, pack
 *      and encode each slice on @p site , land the views and the stats.
 *  @param[in] arena The call's arena, where the outputs land.
 *
 *  Each slice finishes before the next one packs, so the pinned pages a device still reads are
 *  never overwritten, and a progress report means its items are encoded.
 */
template <typename arguments_type_, typename packer_type_, typename site_type_>
void encode_items(arguments_type_& arguments, packer_type_& packer, site_type_& site, uform_arena_handle_t& arena,
                  uform_outcome_t* outcome) noexcept {
    chunked_arena_t& memory = arena.chunks;
    ue::geometry_t const& geometry = arguments.model->geometry;
    chunked_arena_t &staging = site.staging, &transfers = site.transfers;
    std::size_t const count = arguments.count, threads = site.pool.threads_count();
    auto const fail = [&](failure_t failure) noexcept { propagate_error(failure.status, outcome, failure.message); };

    if (failure_t const prepared = packer.prepare(site, threads, geometry); prepared.message) return fail(prepared);
    if (failure_t const reserved = site.reserve_staging(chunked_arena_t::footprint<item_plan_t>(count ? count : 1) +
                                                        chunked_arena_t::footprint<std::size_t>(count + 1));
        reserved.message)
        return fail(reserved);
    item_plan_t* const plans = staging.template allocate<item_plan_t>(count ? count : 1);
    std::size_t* const offsets_memory = staging.template allocate<std::size_t>(count + 1);
    if (!plans || !offsets_memory) return fail({uform_status_out_of_memory_k, "Failed to allocate the plans"});
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
    std::size_t const cell_bytes = packer.cell_bytes(geometry);
    if (util::wraps_on_multiply(tokens, fu::max_of(fu::max_of(row_bytes, token_row_bytes), cell_bytes)) ||
        util::wraps_on_multiply(count, geometry.masked_vocabulary))
        return fail({uform_status_out_of_memory_k, "The batch's outputs overflow the address space"});
    bool const ragged = requested.states || requested.token_vectors;
    std::size_t values = 0; // the probabilities a choice lands, on the host and in the arena
    if constexpr (a_choice_call<arguments_type_>) values = choice_values(arguments);
    // An output named lands at least one byte, as the outputs below take no less
    auto const output_footprint = [](std::size_t bytes) noexcept {
        return chunked_arena_t::footprint(bytes ? bytes : 1);
    };
    std::size_t const output_bytes = (ragged ? chunked_arena_t::footprint<std::size_t>(count + 1) : 0) +
                                     (requested.embeddings ? output_footprint(count * row_bytes) : 0) +
                                     (requested.states ? output_footprint(tokens * row_bytes) : 0) +
                                     (requested.token_vectors ? output_footprint(tokens * token_row_bytes) : 0) +
                                     (a_choice_call<arguments_type_> ? choice_footprint(count, values) : 0);
    if (util::status_t const reserved = reserve_arena(arena, output_bytes); reserved.failed())
        return fail(failure_of(reserved, "Failed to allocate the outputs"));
    std::size_t* const landed_offsets_memory = ragged ? memory.template allocate<std::size_t>(count + 1) : nullptr;
    auto const output = [&](void* requested_view, chunked_arena_t& into, std::size_t bytes) noexcept {
        return requested_view ? static_cast<util::b8_t*>(into.allocate(bytes ? bytes : 1)) : nullptr;
    };
    util::b8_t* const embeddings = output(requested.embeddings, memory, count * row_bytes);
    util::b8_t* const states = output(requested.states, memory, tokens * row_bytes);
    util::b8_t* const token_vectors = output(requested.token_vectors, memory, tokens * token_row_bytes);
    if (!offsets_memory || (ragged && !landed_offsets_memory) || (requested.embeddings && !embeddings) ||
        (requested.states && !states) || (requested.token_vectors && !token_vectors))
        return fail({uform_status_out_of_memory_k, "Failed to allocate the outputs"});
    choice_buffers_t choice; // where a choice lands, allocated with the outputs and the transfers
    if constexpr (a_choice_call<arguments_type_>) {
        allocate_choice_pair(memory, count, values, choice.landed_offsets, choice.landed);
        if (choice.landed_offsets.empty())
            return fail({uform_status_out_of_memory_k, "Failed to allocate the probabilities"});
    }
    std::span<std::size_t> const offsets {offsets_memory, count + 1};
    std::span<std::size_t> const landed_offsets {landed_offsets_memory, ragged ? count + 1 : 0};
    nk_dtype_t const dtype = output_dtype(requested.scalar);
    ue::outputs_t surfaces {{nk::tensor_span<util::b8_t, 2> {embeddings, count, row_bytes}, dtype},
                            {nk::tensor_span<util::b8_t, 2> {states, tokens, row_bytes}, dtype},
                            {nk::tensor_span<util::b8_t, 2> {token_vectors, tokens, token_row_bytes}, dtype},
                            {}};
    offsets[0] = 0;
    for (std::size_t item = 0; item != count; ++item) offsets[item + 1] = offsets[item] + plans[item].tokens;

    progress_ticks_t ticks {arguments.progress, count};
    std::size_t widest_items = 0, widest_tokens = 0;
    for (slice_t const slice : ticks) {
        std::size_t const items = slice.end - slice.begin, cells = offsets[slice.end] - offsets[slice.begin];
        widest_items = items > widest_items ? items : widest_items;
        widest_tokens = cells > widest_tokens ? cells : widest_tokens;
    }
    // The pinned pages every slice packs into, the logits a choice reads, and its probabilities
    std::size_t const masked_vocabulary = a_choice_call<arguments_type_> ? geometry.masked_vocabulary : 0;
    std::size_t const transfer_bytes =
        (masked_vocabulary ? chunked_arena_t::footprint<util::f32_t>(count * masked_vocabulary + 1) : 0) +
        chunked_arena_t::footprint<util::u32_t>(widest_items + 1) * 2 +
        chunked_arena_t::footprint<modality_t>(widest_items + 1) +
        chunked_arena_t::footprint<util::u16_t>(widest_items * 2 + 2) +
        chunked_arena_t::footprint<util::u16_t>(widest_items * 3 + 3) +
        chunked_arena_t::footprint(widest_tokens * cell_bytes + 1) + chunked_arena_t::footprint(row_bytes) +
        (a_choice_call<arguments_type_> ? choice_footprint(count, values) : 0);
    if (failure_t const reserved = site.reserve(widest_tokens, widest_items, cell_bytes, surfaces, masked_vocabulary,
                                                transfer_bytes);
        reserved.message)
        return fail(reserved);
    util::f32_t* masked_logits = nullptr; // in host pages, where the choices' softmax reads them
    if (masked_vocabulary) {
        masked_logits = transfers.template allocate<util::f32_t>(count * masked_vocabulary + 1);
        if (!masked_logits) return fail({uform_status_out_of_memory_k, "Failed to allocate the logits"});
        surfaces.masked_logits = {masked_logits, count, masked_vocabulary};
    }
    util::u32_t* const segment_offsets = transfers.template allocate<util::u32_t>(widest_items + 1);
    util::u32_t* const segment_lengths = transfers.template allocate<util::u32_t>(widest_items + 1);
    modality_t* const segment_modality = transfers.template allocate<modality_t>(widest_items + 1);
    util::u16_t* const image_shapes = transfers.template allocate<util::u16_t>(widest_items * 2 + 2);
    util::u16_t* const video_shapes = transfers.template allocate<util::u16_t>(widest_items * 3 + 3);
    util::b8_t* const cells_memory = static_cast<util::b8_t*>(transfers.allocate(widest_tokens * cell_bytes + 1));
    util::b8_t* const zero_row_memory = static_cast<util::b8_t*>(transfers.allocate(row_bytes));
    if (!segment_offsets || !segment_lengths || !segment_modality || !image_shapes || !video_shapes || !cells_memory ||
        !zero_row_memory)
        return fail({uform_status_out_of_memory_k, "Failed to allocate the batch"});
    if constexpr (a_choice_call<arguments_type_>) {
        allocate_choice_pair(transfers, count, values, choice.offsets, choice.probabilities);
        if (choice.offsets.empty()) return fail({uform_status_out_of_memory_k, "Failed to allocate the probabilities"});
    }
    util::bytes_span_t const cells {cells_memory, widest_tokens * cell_bytes};
    util::bytes_span_t const zero_row {zero_row_memory, row_bytes};
    nk::tensor_span<util::u16_t, 2> const image_shape_rows {image_shapes, widest_items + 1, 2};
    nk::tensor_span<util::u16_t, 2> const video_shape_rows {video_shapes, widest_items + 1, 3};
    std::memset(zero_row.data(), 0, zero_row.size());

    for (slice_t const slice : ticks) {
        std::size_t const items = slice.end - slice.begin, first_row = offsets[slice.begin];
        std::size_t const slice_tokens = offsets[slice.end] - first_row;
        for (std::size_t item = 0; item != items; ++item) {
            item_plan_t const& plan = plans[slice.begin + item];
            segment_offsets[item] = static_cast<util::u32_t>(offsets[slice.begin + item] - first_row);
            segment_lengths[item] = plan.tokens;
            segment_modality[item] = packer_type_::modality_k;
            for (std::size_t axis = 0; axis != 2; ++axis) image_shape_rows[item, axis] = plan.shape[axis];
            for (std::size_t axis = 0; axis != 3; ++axis) video_shape_rows[item, axis] = plan.shape[axis];
        }
        segment_offsets[items] = static_cast<util::u32_t>(slice_tokens);
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
            ue::multimodal_batch_view_t batch;
            batch.segment_offsets = {segment_offsets, items + 1};
            batch.segment_lengths = {segment_lengths, items};
            batch.segment_modality = {segment_modality, items};
            batch.image_shapes = {image_shapes, items, 2};
            batch.video_shapes = {video_shapes, items, 3};
            void* const content = cells.data();
            switch (packer_type_::modality_k) {
            case modality_t::text_k: batch.tokens = {static_cast<util::token_id_t*>(content), 1, slice_tokens}; break;
            case modality_t::image_k:
                batch.images = {static_cast<util::pixel_t*>(content),
                                {1, slice_tokens, side, side, geometry.image_channels}};
                break;
            case modality_t::video_k:
                batch.videos = {
                    static_cast<util::pixel_t*>(content),
                    {1, slice_tokens, geometry.video_tube_frames, video_side, video_side, geometry.video_channels}};
                break;
            case modality_t::audio_k:
                batch.audio_samples = {static_cast<util::f32_t*>(content),
                                       {1, slice_tokens, geometry.audio_chunk_samples}};
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
        else if (embeddings)
            if (failure_t const landed = land_zero_rows(site, surfaces.embeddings, slice, zero_row); landed.message)
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
    if constexpr (a_choice_call<arguments_type_>)
        if (failure_t const chosen = land_choices(arguments, surfaces.masked_logits, site, choice); chosen.message)
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

/** A VRAM call through @p verbs : one shard per stream of @p gpus whose device holds one of
 *  @p replicas , planned on @p decoders into @p staging . */
template <vendored_residency_t residency_, typename arguments_type_, typename packer_type_>
void encode_on_devices(arguments_type_& arguments, packer_type_& packer, encoder_verbs<residency_> const& verbs,
                       util::per_gpu_array<vram_replica_t> const& replicas, executor_of<residency_>& gpus,
                       chunked_arena_t& staging, fu::pool_t& decoders, uform_arena_handle_t& arena,
                       call_outcome_t& outcome) noexcept {
    auto const fail = [&](failure_t failure) noexcept { propagate_error(failure.status, outcome, failure.message); };
    // The executor serializes its streams, so a second call on it answers contended
    auto [claim, claim_status] = gpus.claim();
    if (claim_status.failed()) return fail({uform_status_contended_k, "Another call holds the executor"});
    auto& lead = gpus.stream(0);
    staging.rewind();
    // The arena and the pinned pages live on the lead device, where the outputs grow first
    if (lead.bind().failed()) return fail(vram_site<residency_>::unbound_k);
    vram_site<residency_> site {decoders, staging, lead.memory.transfers, lead, verbs};
    for (std::size_t index = 0; index != gpus.size(); ++index) {
        auto& stream = gpus.stream(index);
        util::i32_t const ordinal = listed_ordinal_of<executor_of<residency_>>(stream);
        for (vram_replica_t const& replica : replicas)
            if (replica.ordinal == ordinal &&
                site.shards.push_back({&stream, &replica}) != util::status_code_t::success_k)
                return fail({uform_status_internal_k, "More devices than one process drives"});
    }
    if (!site.shards.size()) return fail({uform_status_device_k, "The model holds no replica on the executor"});
    encode_items(arguments, packer, site, arena, outcome);
    site.finish(); // a failed slice leaves work queued, which drains before the next call reuses what it reads
}

/** Every encode verb after its entry: refuse, claim the arena, pick where the call
 *  computes, and encode. */
template <typename arguments_type_, typename packer_type_>
void encode_call(arguments_type_& arguments, packer_type_& packer, uform_tape_view_t const& tape,
                 uform_scalar_t element, std::initializer_list<void const*> inputs, call_outcome_t& outcome) noexcept {
    clear_outputs(arguments);
    auto const fail = [&](failure_t failure) noexcept { propagate_error(failure.status, outcome, failure.message); };
    if (failure_t const refused = validate_call(arguments, tape, element); refused.message) return fail(refused);
    if (failure_t const refused = packer.validate(arguments.model->geometry); refused.message) return fail(refused);
    uform_arena_t const arena = claim_arena(arguments.arena, arguments.executor, inputs, outcome);
    if (!arena) return;
    uform_executor_t const executor = arguments.executor;
    uform_model_handle_t const& model = *arguments.model;

    std::visit(
        [&]<vendored_residency_t residency_>(encoder_verbs<residency_> const* verbs) noexcept {
            if constexpr (residency_ == vendored_residency_t::ram_k) {
                if (executor && !std::holds_alternative<std::monostate>(executor->device))
                    return fail({uform_status_device_k, "A RAM model computes on a RAM executor or none"});
                fu::pool_t caller; // empty, so it runs every dispatch on the calling thread
                ram_site_t site {executor ? executor->threads : caller,
                                 *arena,
                                 arena->chunks,
                                 arena->chunks,
                                 *verbs,
                                 model.towers[0].tower};
                return encode_items(arguments, packer, site, *arena, outcome);
            }
            else if constexpr (!gpu_vendor_compiled(vendor_of(residency_)))
                return fail({uform_status_unsupported_k, "This build drives no GPU of the model's vendor"});
            else {
                auto* const gpus = executor ? std::get_if<executor_of<residency_>>(&executor->device) : nullptr;
                if (!gpus)
                    return fail({uform_status_device_k, "A VRAM model computes on a VRAM executor of its vendor"});
                encode_on_devices(arguments, packer, *verbs, model.towers, *gpus, executor->staging, executor->threads,
                                  *arena, outcome);
            }
        },
        model.tables);
}

#pragma endregion Calls

} // namespace unum::uform
