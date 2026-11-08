/**
 *  @file src/encode.cpp
 *  @author Ash Vardanian
 *  @date October 17, 2026
 *  @brief The verbs of @c encode.h over decoded inputs: pack each input form into the
 *      batch contract and run the model's tower over it. The verbs decoding files live
 *      in @c encode_media.cpp.
 *
 *  Every verb plans its items first, so the outputs are sized before anything is decoded, then
 *  packs the slices its progress handler asks for into one batch row of segments laid back to back,
 *  so short items cost no padding:
 *
 *      Verb                  One token                       Planned from
 *      text                  one byte                        the tape alone
 *      image pixels          one patch of the resized crop   the height and width
 *      image                 one patch of the resized crop   the file's header
 *      video                 one tube of patches             a demux of the container
 *      audio samples         one frame of chunk samples      the count, resampled to the model's rate
 *      audio                 one frame of chunk samples      a decode into the context, its silent tail dropped
 *
 *  Media decode on the executor's threads: its pool on the host, its decoder pool beside GPUs, and
 *  one thread of the caller's without an executor.
 */
#include "uform/media/audio.hpp"    // `resample_mono`
#include "uform/media/planning.hpp" // `image_patch_grid`

#include "encode.hpp"

using namespace unum::uform;

#pragma region Packers

/** UTF-8 bytes, each a token, cut at the context. */
struct text_packer_t {
    static constexpr modality_t modality_k = modality_t::text_k;
    tape_view<b8_t const> tape;

    std::size_t cell_bytes(ue::geometry_t const &) const noexcept { return sizeof(token_id_t); }
    failure_t validate(ue::geometry_t const &) const noexcept { return {}; }
    failure_t prepare(chunked_arena_t &, std::size_t, ue::geometry_t const &) noexcept { return {}; }

    failure_t plan(std::size_t item, item_plan_t &plan, ue::geometry_t const &geometry, std::size_t) const noexcept {
        plan.spanned = tape[item].size();
        plan.tokens = static_cast<u32_t>(plan.spanned < geometry.columns ? plan.spanned : geometry.columns);
        return {};
    }

    failure_t stage(std::size_t item, item_plan_t const &plan, bytes_span_t cells, ue::geometry_t const &,
                    std::size_t) const noexcept {
        bytes_view_t const bytes = tape[item];
        token_id_t *const tokens = static_cast<token_id_t *>(static_cast<void *>(cells.data()));
        for (u32_t token = 0; token != plan.tokens; ++token) tokens[token] = static_cast<token_id_t>(bytes[token]);
        return {};
    }
};

/** A state's UTF-8 bytes and the mask byte a choice is read at, its head cut so the mask
 *  survives the context. */
struct choice_packer_t {
    static constexpr modality_t modality_k = modality_t::text_k;
    tape_view<b8_t const> tape;

    std::size_t cell_bytes(ue::geometry_t const &) const noexcept { return sizeof(token_id_t); }
    failure_t validate(ue::geometry_t const &) const noexcept { return {}; }
    failure_t prepare(chunked_arena_t &, std::size_t, ue::geometry_t const &) noexcept { return {}; }

    failure_t plan(std::size_t item, item_plan_t &plan, ue::geometry_t const &geometry, std::size_t) const noexcept {
        plan.spanned = tape[item].size() + 1;
        plan.tokens = static_cast<u32_t>(plan.spanned < geometry.columns ? plan.spanned : geometry.columns);
        return {};
    }

    failure_t stage(std::size_t item, item_plan_t const &plan, bytes_span_t cells, ue::geometry_t const &,
                    std::size_t) const noexcept {
        bytes_view_t const bytes = tape[item];
        token_id_t *const tokens = static_cast<token_id_t *>(static_cast<void *>(cells.data()));
        std::size_t const kept = plan.tokens - 1, skipped = bytes.size() - kept;
        for (std::size_t token = 0; token != kept; ++token)
            tokens[token] = static_cast<token_id_t>(bytes[skipped + token]);
        tokens[kept] = ue::byte_mask_k;
        return {};
    }
};

/** Interleaved RGB rows, cropped to their patch grid's aspect and resized on the host. */
struct pixels_packer_t {
    static constexpr modality_t modality_k = modality_t::image_k;
    tape_view<b8_t const> tape;
    std::span<u32_t const> heights;
    std::span<u32_t const> widths;

    /** @b [threads,pixels·4] one RGBA copy of the largest item per thread, which
     *  the resize reads. */
    nk::tensor_span<pixel_t, 2> expanded {};

    std::size_t cell_bytes(ue::geometry_t const &geometry) const noexcept {
        return std::size_t {geometry.image_patch_side} * geometry.image_patch_side * geometry.image_channels;
    }

    failure_t validate(ue::geometry_t const &geometry) const noexcept {
        if (failure_t const refused = validate_images(geometry); refused.message) return refused;
        std::size_t const count = tape.size();
        if (count && (heights.empty() || widths.empty()))
            return {uform_status_invalid_argument_k, "Every item needs its height and width"};
        for (std::size_t item = 0; item != count; ++item)
            if (tape[item].size() != std::size_t {heights[item]} * widths[item] * 3)
                return {uform_status_shape_mismatch_k, "An item's bytes disagree with its height and width"};
        return {};
    }

    failure_t prepare(chunked_arena_t &staging, std::size_t threads, ue::geometry_t const &) noexcept {
        std::size_t widest = 0;
        for (std::size_t item = 0; item != tape.size(); ++item) {
            std::size_t const bytes = std::size_t {heights[item]} * widths[item] * um::image_channels_k;
            widest = bytes > widest ? bytes : widest;
        }
        if (!widest) return {};
        pixel_t *const rows = staging.allocate<pixel_t>(widest * threads);
        if (!rows) return {uform_status_out_of_memory_k, "Failed to allocate the pixel staging"};
        expanded = {rows, threads, widest};
        return {};
    }

    failure_t plan(std::size_t item, item_plan_t &plan, ue::geometry_t const &geometry, std::size_t) const noexcept {
        um::image_extent_t const native {widths[item], heights[item]};
        if (!native.width || !native.height) return {};
        um::patch_grid_t const grid = um::image_patch_grid(ue::media_cells(ue::modality_t::image_k, geometry.columns),
                                                           native);
        plan.tokens = static_cast<u32_t>(grid.rows) * grid.columns;
        plan.spanned = native_patches(native, geometry.image_patch_side);
        plan.shape[0] = grid.rows, plan.shape[1] = grid.columns;
        return {};
    }

    failure_t stage(std::size_t item, item_plan_t const &plan, bytes_span_t cells, ue::geometry_t const &geometry,
                    std::size_t thread) const noexcept {
        if (!plan.tokens) return {};
        std::size_t const height = heights[item], width = widths[item], pixels = height * width;
        nk::tensor_view<b8_t, 2> const rgb {tape[item].data(), pixels, 3};
        pixel_t *const rgba = expanded.row(thread).data();
        nk::tensor_span<pixel_t, 2> const rgba_pixels {rgba, pixels, 4};
        for (std::size_t pixel = 0; pixel != pixels; ++pixel) {
            for (std::size_t channel = 0; channel != 3; ++channel)
                rgba_pixels[pixel, channel] = static_cast<pixel_t>(rgb[pixel, channel]);
            rgba_pixels[pixel, 3] = 255;
        }
        std::size_t const side = geometry.image_patch_side;
        expected_t const resized = um::resize_image(
            {rgba, {height, width, um::image_channels_k}},
            {static_cast<pixel_t *>(static_cast<void *>(cells.data())), {plan.shape[0], plan.shape[1], side, side, 4}});
        if (!resized) return {media_status_of(resized.status_), "An image could not be resized"};
        return {};
    }
};

/** Mono Float32 samples, resampled to the model's rate and cut into frames. */
struct samples_packer_t {
    static constexpr modality_t modality_k = modality_t::audio_k;
    tape_view<f32_t const> tape;
    u32_t sample_rate;

    std::size_t cell_bytes(ue::geometry_t const &geometry) const noexcept {
        return std::size_t {geometry.audio_chunk_samples} * sizeof(f32_t);
    }

    failure_t validate(ue::geometry_t const &geometry) const noexcept {
        if (failure_t const refused = validate_audio(geometry); refused.message) return refused;
        if (!sample_rate) return {uform_status_invalid_argument_k, "The sample rate must not be zero"};
        return {};
    }

    failure_t prepare(chunked_arena_t &, std::size_t, ue::geometry_t const &) noexcept { return {}; }

    /** Samples item @p item spans at the model's rate of @p geometry . */
    std::size_t resampled(std::size_t item, ue::geometry_t const &geometry) const noexcept {
        std::size_t const samples = tape[item].size();
        if (sample_rate == geometry.audio_sample_rate) return samples;
        return divide_round_up(samples * geometry.audio_sample_rate, sample_rate);
    }

    failure_t plan(std::size_t item, item_plan_t &plan, ue::geometry_t const &geometry, std::size_t) const noexcept {
        plan.spanned = divide_round_up(resampled(item, geometry), geometry.audio_chunk_samples);
        u32_t const cells = ue::media_cells(ue::modality_t::audio_k, geometry.columns);
        plan.tokens = static_cast<u32_t>(plan.spanned < cells ? plan.spanned : cells);
        return {};
    }

    failure_t stage(std::size_t item, item_plan_t const &plan, bytes_span_t cells, ue::geometry_t const &geometry,
                    std::size_t) const noexcept {
        f32_t *const frames = static_cast<f32_t *>(static_cast<void *>(cells.data()));
        std::size_t const capacity = std::size_t {plan.tokens} * geometry.audio_chunk_samples;
        std::size_t const total = resampled(item, geometry), kept = total < capacity ? total : capacity;
        std::span<f32_t const> const samples = tape[item];
        if (sample_rate == geometry.audio_sample_rate) std::memcpy(frames, samples.data(), kept * sizeof(f32_t));
        else if (kept) {
            expected_t const resampled_into = um::resample_mono(samples, sample_rate, geometry.audio_sample_rate,
                                                                {frames, kept});
            if (!resampled_into) return {media_status_of(resampled_into.status_), "A recording could not be resampled"};
        }
        std::memset(frames + kept, 0, (capacity - kept) * sizeof(f32_t));
        return {};
    }
};

#pragma endregion Packers

extern "C" {

UFORM_EXPORT void uform_text_encode(uform_text_encode_t *arguments) {
    if (submitted<uform_text_encode>(arguments)) return;
    call_outcome_t outcome(arguments->outcome, arguments->arena);
    uform_tape_view_t const tape = arguments->texts;
    text_packer_t packer {tape_of<b8_t const>(tape, arguments->count)};
    encode_call(*arguments, packer, tape, uform_scalar_u8_k, {tape.data, tape.offsets}, outcome);
}

UFORM_EXPORT void uform_text_choose(uform_text_choose_t *arguments) {
    if (submitted<uform_text_choose>(arguments)) return;
    call_outcome_t outcome(arguments->outcome, arguments->arena);
    uform_tape_view_t const tape = arguments->texts;
    choice_packer_t packer {tape_of<b8_t const>(tape, arguments->count)};
    encode_call(
        *arguments, packer, tape, uform_scalar_u8_k,
        {tape.data, tape.offsets, arguments->choices.data, arguments->choices.offsets, arguments->temperature_groups},
        outcome);
}

UFORM_EXPORT void uform_image_encode_pixels(uform_image_encode_pixels_t *arguments) {
    if (submitted<uform_image_encode_pixels>(arguments)) return;
    call_outcome_t outcome(arguments->outcome, arguments->arena);
    uform_tape_view_t const tape = arguments->pixels;
    std::size_t const count = arguments->count;
    pixels_packer_t packer {tape_of<b8_t const>(tape, count),
                            {arguments->heights, arguments->heights ? count : 0},
                            {arguments->widths, arguments->widths ? count : 0}};
    encode_call(*arguments, packer, tape, uform_scalar_u8_k,
                {tape.data, tape.offsets, arguments->heights, arguments->widths}, outcome);
}

UFORM_EXPORT void uform_audio_encode_samples(uform_audio_encode_samples_t *arguments) {
    if (submitted<uform_audio_encode_samples>(arguments)) return;
    call_outcome_t outcome(arguments->outcome, arguments->arena);
    uform_tape_view_t const tape = arguments->samples;
    samples_packer_t packer {tape_of<f32_t const>(tape, arguments->count), arguments->sample_rate};
    encode_call(*arguments, packer, tape, uform_scalar_f32_k, {tape.data, tape.offsets}, outcome);
}

} // extern "C"
