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

#include <cstring> // `std::memcpy`, `std::memset`

#include <initializer_list> // `std::initializer_list` - the inputs a call checks against its arena
#include <span>             // `std::span` - an item's bytes as the decoders take them

#include "lib.hpp"
#include "media/image.hpp" // `image_plan_t`, `image_extent_t`, `image_channels_k`

namespace unum::uform {

namespace ui = unum::uform::inference;
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

/** Entry @p item of @p tape , whose elements are @p element_bytes wide, as bytes. */
inline std::span<b8_t const> entry_of(uform_tape_view_t const &tape, std::size_t item,
                                      std::size_t element_bytes) noexcept {
    std::size_t const begin = tape.offsets[item], end = tape.offsets[item + 1];
    return {static_cast<b8_t const *>(tape.data) + begin * element_bytes, (end - begin) * element_bytes};
}

/** The refusals of every image verb: a model without the image entry, or one whose pixels are not RGBA. */
inline failure_t validate_images(ui::geometry_t const &geometry) noexcept {
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
inline failure_t validate_audio(ui::geometry_t const &geometry) noexcept {
    if (!geometry.audio_chunk_samples || !geometry.audio_sample_rate)
        return {uform_status_unsupported_k, "The model was not trained on audio"};
    return {};
}

#pragma endregion Plans

#pragma region Sites

/** The C status a tower's failure crosses into: a batch it calls invalid is a packing defect of the encode verbs. */
inline failure_t tower_failure(core::status_t status) noexcept {
    if (status == core::status_t::invalid_input_k) return {uform_status_internal_k, "The tower refused a packed batch"};
    return {status_of(status), "The tower failed to encode the batch"};
}

/** @p view over @p data in place of its own storage, its shape kept; an absent view stays absent. */
template <typename view_type_>
view_type_ rebased(view_type_ view, void const *data) noexcept {
    if (!view.byte_data()) return view;
    return view_type_ {static_cast<char const *>(data), view.shape()};
}

/** A RAM call: the host tower on a pool, packing into the call's own arena, which also lends the scratch. */
struct ram_site_t {
    fu::flat_pool_t &pool;

    /** What only the host reads, and what the tower reads, both the call's arena on the host. */
    chunked_arena_t &staging;
    chunked_arena_t &transfers;
    ram_model_t const &model;
    std::span<b8_t> scratch {};

    failure_t reserve(std::size_t tokens, std::size_t segments, std::size_t, ui::outputs_t, std::size_t) noexcept {
        core::expected<std::size_t> bytes = model.tower.scratch_bytes(tokens, segments, model.capabilities);
        if (!bytes) return tower_failure(bytes.status_);
        b8_t *const data = static_cast<b8_t *>(staging.allocate(*bytes));
        if (!data && *bytes) return {uform_status_out_of_memory_k, "Failed to allocate the activations"};
        scratch = {data, *bytes};
        return {};
    }

    failure_t encode(core::multimodal_batch_view_t const &batch, b8_t const *, std::span<std::size_t const>,
                     ui::outputs_t outputs, std::size_t) noexcept {
        core::expected_t encoded = model.tower.encode(batch, outputs, {model.capabilities, nullptr, scratch}, pool);
        return encoded ? failure_t {} : tower_failure(encoded.status_);
    }

    failure_t land(void *target, void const *source, std::size_t bytes) noexcept {
        std::memcpy(target, source, bytes);
        return {};
    }
    failure_t finish() noexcept { return {}; }
};

/** One device of a VRAM call: the executor's device, the model's replica on it, and its regions of the scratch. */
struct vram_shard_t {
    vram_device_t *device = nullptr;
    vram_replica_t const *replica = nullptr;

    /** The vendor's copies, which move the shard's run of each slice in and its outputs out. */
    ui::device_kernels_t const *kernels = nullptr;
    std::span<b8_t> scratch {};

    /** Room for a whole slice's cells, of which the shard copies in only its own run, at the run's offset. */
    b8_t *cells = nullptr;

    /** The media shapes of the shard's run. */
    u16_t *image_shapes = nullptr;
    u16_t *video_shapes = nullptr;

    /** Where a device other than the arena's writes its outputs before copying them over; null on the arena's. */
    b8_t *embeddings = nullptr;
    b8_t *states = nullptr;
};

/** A VRAM call: plans on the decoder pool into host pages, packs into pinned pages, and splits each slice across the
 *  devices holding a replica, by token count. */
struct vram_site_t {
    fu::flat_pool_t &pool;
    chunked_arena_t &staging;
    chunked_arena_t &transfers;

    /** The stream of the device the call's arena lives on, whose shard writes the arena directly. */
    nk_stream_t arena_stream = nullptr;
    per_gpu_array<vram_shard_t> shards {};
    std::size_t cell_bytes = 0;

    failure_t reserve(std::size_t tokens, std::size_t segments, std::size_t bytes_per_cell, ui::outputs_t surfaces,
                      std::size_t row_bytes) noexcept {
        cell_bytes = bytes_per_cell;
        for (vram_shard_t &shard : shards) {
            core::expected<std::size_t> bytes = shard.replica->tower.scratch_bytes(tokens, segments,
                                                                                   shard.device->capabilities);
            if (!bytes) return tower_failure(bytes.status_);
            device_arena_t &memory = shard.device->scratch;
            memory.rewind();
            failure_t failure {};
            auto const allocate = [&](std::size_t size) noexcept {
                b8_t *const data = static_cast<b8_t *>(memory.allocate(size));
                if (!data && size) failure = {uform_status_out_of_memory_k, "The executor's scratch limit is too low"};
                return data;
            };
            shard.scratch = {allocate(*bytes), *bytes};
            shard.cells = allocate(tokens * cell_bytes);
            shard.image_shapes = static_cast<u16_t *>(static_cast<void *>(allocate(segments * 2 * sizeof(u16_t))));
            shard.video_shapes = static_cast<u16_t *>(static_cast<void *>(allocate(segments * 3 * sizeof(u16_t))));
            bool const elsewhere = shard.device->stream.get() != arena_stream;
            shard.embeddings = elsewhere && surfaces.embeddings ? allocate(segments * row_bytes) : nullptr;
            shard.states = elsewhere && surfaces.states ? allocate(tokens * row_bytes) : nullptr;
            if (failure.message) return failure;
        }
        return {};
    }

    /** Runs the items of @p batch , whose cells @p cells hold, on every shard, each taking a contiguous run of about
     *  equal tokens, their outputs landing at the run's rows. */
    failure_t encode(core::multimodal_batch_view_t const &batch, b8_t const *cells,
                     std::span<std::size_t const> packed_offsets, ui::outputs_t outputs,
                     std::size_t row_bytes) noexcept {
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
            auto const copied = [&](void *into, void const *from, std::size_t bytes) noexcept {
                return !bytes || shard.kernels->copy(into, from, bytes, stream) == nk_success_k;
            };
            std::size_t const begin = packed_offsets[first] - origin, end = packed_offsets[last] - origin;
            std::size_t const items = last - first;
            auto const shard_items = nk::range(first, last);
            nk::tensor_view<u16_t, 2> const image_shapes = batch.image_shapes[shard_items, nk::slice];
            nk::tensor_view<u16_t, 2> const video_shapes = batch.video_shapes[shard_items, nk::slice];
            if (!copied(shard.cells + begin * cell_bytes, cells + begin * cell_bytes, (end - begin) * cell_bytes) ||
                !copied(shard.image_shapes, image_shapes.data(), image_shapes.numel() * sizeof(u16_t)) ||
                !copied(shard.video_shapes, video_shapes.data(), video_shapes.numel() * sizeof(u16_t)))
                return moved;

            core::multimodal_batch_view_t shard_batch = batch;
            shard_batch.segment_offsets = batch.segment_offsets[nk::range(first, last + 1), nk::slice];
            shard_batch.segment_lengths = batch.segment_lengths[shard_items, nk::slice];
            shard_batch.segment_modality = batch.segment_modality[shard_items, nk::slice];
            shard_batch.image_shapes = {shard.image_shapes, items, 2};
            shard_batch.video_shapes = {shard.video_shapes, items, 3};
            shard_batch.tokens = rebased(batch.tokens, shard.cells);
            shard_batch.images = rebased(batch.images, shard.cells);
            shard_batch.videos = rebased(batch.videos, shard.cells);
            shard_batch.audio_samples = rebased(batch.audio_samples, shard.cells);

            b8_t *const embeddings = outputs.embeddings ? static_cast<b8_t *>(outputs.embeddings) + first * row_bytes
                                                        : nullptr;
            b8_t *const states = outputs.states ? static_cast<b8_t *>(outputs.states) + begin * row_bytes : nullptr;
            bool const elsewhere = stream != arena_stream;
            ui::outputs_t const shard_outputs {elsewhere && embeddings ? shard.embeddings : embeddings,
                                               elsewhere && states ? shard.states : states, outputs.dtype};
            core::expected_t encoded = shard.replica->tower.encode(shard_batch, shard_outputs,
                                                                   {shard.device->capabilities, stream, shard.scratch});
            if (!encoded) return tower_failure(encoded.status_);
            if (elsewhere && embeddings && !copied(embeddings, shard.embeddings, items * row_bytes)) return moved;
            if (elsewhere && states && !copied(states, shard.states, (end - begin) * row_bytes)) return moved;
            first = last;
        }
        return {};
    }

    /** Copies host bytes into the arena on the first shard's stream, which @ref finish waits for. */
    failure_t land(void *target, void const *source, std::size_t bytes) noexcept {
        vram_shard_t const &first = shards[0];
        if (first.kernels->copy(target, source, bytes, first.device->stream.get()) != nk_success_k)
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

/** Zeroes every output and report, so a call refused or failing before it lands leaves them null. */
template <typename arguments_type_>
void clear_outputs(arguments_type_ &arguments) noexcept {
    if (arguments.embeddings) *arguments.embeddings = {};
    if (arguments.states) *arguments.states = {};
    if (arguments.stats) *arguments.stats = {};
}

/** The refusals every encode verb shares before it claims its arena. */
template <typename arguments_type_>
failure_t validate_call(arguments_type_ const &arguments, uform_tape_view_t const &tape,
                        uform_scalar_t element) noexcept {
    if (!arguments.model) return {uform_status_invalid_argument_k, "The model must not be null"};
    if (!arguments.embeddings && !arguments.states)
        return {uform_status_invalid_argument_k, "Name the embeddings, the states, or both"};
    if (!output_bytes(arguments.outputs_scalar))
        return {uform_status_unsupported_k, "Outputs land as f32, f16 or bf16 only"};
    if (!tape.offsets || (!tape.data && arguments.count && tape.offsets[arguments.count] != tape.offsets[0]))
        return {uform_status_invalid_argument_k, "The tape must name its storage and offsets"};
    if (tape.scalar != element) return {uform_status_unsupported_layout_k, "The tape holds another scalar"};
    for (std::size_t item = 0; item != arguments.count; ++item)
        if (tape.offsets[item + 1] < tape.offsets[item])
            return {uform_status_malformed_k, "The tape's offsets must not decrease"};
    return {};
}

/** Runs @p step for every item of @c [begin,end) on @p pool , answering the first failure any reported. */
template <typename step_type_>
failure_t for_items(fu::flat_pool_t &pool, std::size_t begin, std::size_t end, step_type_ step) noexcept {
    failure_t first {};
    fu::spin_mutex_t guard;
    pool.for_n(end - begin, [&](std::size_t task, auto at) noexcept {
        failure_t const failure = step(begin + task, static_cast<std::size_t>(at.thread));
        if (!failure.message) return;
        std::lock_guard<fu::spin_mutex_t> lock(guard);
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
    ui::geometry_t const &geometry = arguments.model->geometry;
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

    std::size_t const row_bytes = std::size_t {geometry.hidden} * output_bytes(arguments.outputs_scalar);
    std::size_t *const offsets = staging.template allocate<std::size_t>(count + 1);
    std::size_t *const landed_offsets = arguments.states ? memory.template allocate<std::size_t>(count + 1) : nullptr;
    b8_t *const embeddings = arguments.embeddings ? static_cast<b8_t *>(memory.allocate(count * row_bytes)) : nullptr;
    b8_t *const states = arguments.states ? static_cast<b8_t *>(memory.allocate(tokens * row_bytes)) : nullptr;
    if (!offsets || (arguments.states && !landed_offsets) || (arguments.embeddings && !embeddings && count) ||
        (arguments.states && !states && tokens))
        return fail({uform_status_out_of_memory_k, "Failed to allocate the outputs"});
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
    b8_t *const cells = static_cast<b8_t *>(transfers.allocate(widest_tokens * cell_bytes + 1));
    b8_t *const zero_row = static_cast<b8_t *>(transfers.allocate(row_bytes));
    if (!segment_offsets || !segment_lengths || !segment_modality || !image_shapes || !video_shapes || !cells ||
        !zero_row)
        return fail({uform_status_out_of_memory_k, "Failed to allocate the batch"});
    nk::tensor_span<u16_t, 2> const image_shape_rows {image_shapes, widest_items + 1, 2};
    nk::tensor_span<u16_t, 2> const video_shape_rows {video_shapes, widest_items + 1, 3};
    std::memset(zero_row, 0, row_bytes);
    ui::outputs_t const surfaces {embeddings, states, output_dtype(arguments.outputs_scalar)};
    if (widest_tokens)
        if (failure_t const reserved = site.reserve(widest_tokens, widest_items, cell_bytes, surfaces, row_bytes);
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
                return packer.stage(item, plans[item], cells + (offsets[item] - first_row) * cell_bytes, geometry,
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
            void *const content = cells;
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
            ui::outputs_t outputs = surfaces;
            if (embeddings) outputs.embeddings = embeddings + slice.begin * row_bytes;
            if (states) outputs.states = states + first_row * row_bytes;
            if (failure_t const encoded = site.encode(batch, cells, {offsets + slice.begin, items + 1}, outputs,
                                                      row_bytes);
                encoded.message)
                return fail(encoded);
        }
        else if (embeddings) // every item of the slice is empty, which the tower refuses, and each pools to a zero row
            for (std::size_t item = 0; item != items; ++item)
                if (failure_t const landed = site.land(embeddings + (slice.begin + item) * row_bytes, zero_row,
                                                       row_bytes);
                    landed.message)
                    return fail(landed);
        if (failure_t const finished = site.finish(); finished.message) return fail(finished);
        if (ticks(slice.end) == continuation_t::cancel_k)
            return fail({uform_status_cancelled_k, "The caller's handler stopped the call"});
    }
    if (landed_offsets)
        if (failure_t const landed = site.land(landed_offsets, offsets, (count + 1) * sizeof(std::size_t));
            landed.message)
            return fail(landed);
    if (failure_t const finished = site.finish(); finished.message) return fail(finished);

    uform_scalar_t const scalar = arguments.outputs_scalar;
    if (arguments.embeddings) *arguments.embeddings = {embeddings, row_bytes, geometry.hidden, scalar};
    if (arguments.states) *arguments.states = {{states, row_bytes, geometry.hidden, scalar}, landed_offsets};
    if (arguments.stats) *arguments.stats = {truncated, tokens};
    propagate_error(uform_status_success_k, outcome, nullptr);
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
    chunked_arena_t &memory = std::visit([](chunked_arena_t &held) noexcept -> chunked_arena_t & { return held; },
                                         arena->memory);
    uform_executor_t const executor = arguments.executor;

    if (ram_model_t const *const ram = std::get_if<ram_model_t>(&arguments.model->residency)) {
        if (executor && !std::holds_alternative<ram_executor_t>(executor->executor))
            return fail({uform_status_device_k, "A RAM model computes on a RAM executor or none"});
        if (executor) {
            ram_site_t site {std::get<ram_executor_t>(executor->executor).pool, memory, memory, *ram};
            return encode_items(arguments, packer, site, memory, outcome);
        }
        fu::flat_pool_t caller;
        if (fu::failed(caller.spawn(1))) return fail({uform_status_out_of_memory_k, "Failed to set up the caller"});
        ram_site_t site {caller, memory, memory, *ram};
        return encode_items(arguments, packer, site, memory, outcome);
    }

    vram_model_t const &replicated = std::get<vram_model_t>(arguments.model->residency);
    vram_executor_t *const devices = executor ? std::get_if<vram_executor_t>(&executor->executor) : nullptr;
    if (!devices) return fail({uform_status_device_k, "A VRAM model computes on a VRAM executor"});
    vram_device_t &lead = devices->devices[0];
    devices->staging.rewind();
    lead.transfers.rewind();
    vram_site_t site {devices->decoders, devices->staging, lead.transfers, memory.stream()};
    for (vram_device_t &device : devices->devices)
        for (vram_replica_t const &replica : replicated.replicas)
            if (replica.ordinal == device.ordinal && replica.capabilities == device.capabilities) {
                ui::device_kernels_t const *const kernels = ui::device_kernels_of(device.capabilities);
                if (!kernels) return fail({uform_status_unsupported_k, "This build holds no kernels for a device"});
                if (site.shards.push_back({&device, &replica, kernels}) != core::status_t::success_k)
                    return fail({uform_status_internal_k, "More devices than one process drives"});
            }
    if (!site.shards.size()) return fail({uform_status_device_k, "The model holds no replica on the executor"});
    encode_items(arguments, packer, site, memory, outcome);
    site.finish(); // a failed slice leaves work queued, which drains before the next call reuses what it reads
}

#pragma endregion Calls

} // namespace unum::uform
