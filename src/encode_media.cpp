/**
 *  @file src/encode_media.cpp
 *  @author Ash Vardanian
 *  @date October 17, 2026
 *  @brief The verbs of @c encode.h over encoded files: images, clips and recordings, planned from their headers or a
 *      demux and decoded on the host by the codecs of @c src/media.
 *
 *  A build without the codecs links @c encode_media_absent.cpp in its place.
 */
#include "encode.hpp"
#include "media/audio.hpp" // `decode_audio`
#include "media/video.hpp" // `demux_video`, `decode_video_frames`

using namespace unum::uform;

#pragma region Packers

/** JPEG, PNG and WebP files, planned from their headers and decoded on the host. */
struct images_packer_t {
    static constexpr modality_t modality_k = modality_t::image_k;
    uform_tape_view_t tape;

    std::size_t cell_bytes(ue::geometry_t const &geometry) const noexcept {
        return std::size_t {geometry.image_patch_side} * geometry.image_patch_side * geometry.image_channels;
    }
    failure_t validate(ue::geometry_t const &geometry) const noexcept { return validate_images(geometry); }
    failure_t prepare(chunked_arena_t &, std::size_t, ue::geometry_t const &) noexcept { return {}; }

    failure_t plan(std::size_t item, item_plan_t &plan, ue::geometry_t const &geometry, std::size_t) const noexcept {
        core::expected<um::image_plan_t> planned = um::plan_image(entry_of(tape, item, 1), geometry.image_patch_side,
                                                                  geometry.columns);
        if (!planned) return {media_status_of(planned.status_), "An image file could not be probed"};
        plan.image = *planned;
        plan.tokens = static_cast<u32_t>(planned->grid.rows) * planned->grid.columns;
        plan.spanned = native_patches(planned->native, geometry.image_patch_side);
        plan.shape[0] = planned->grid.rows, plan.shape[1] = planned->grid.columns;
        return {};
    }

    failure_t stage(std::size_t item, item_plan_t const &plan, b8_t *cells, ue::geometry_t const &geometry,
                    std::size_t) const noexcept {
        std::size_t const side = geometry.image_patch_side;
        core::expected_t const decoded = um::decode_image(
            entry_of(tape, item, 1), plan.image,
            {static_cast<pixel_t *>(static_cast<void *>(cells)), {plan.shape[0], plan.shape[1], side, side, 4}});
        if (!decoded) return {media_status_of(decoded.status_), "An image file could not be decoded"};
        return {};
    }
};

/** The video modality's slice of @p geometry , as the planner and the decoder take it. */
static core::video_batch_shape_t video_shape_of(ue::geometry_t const &geometry) noexcept {
    return {geometry.video_patch_side, geometry.video_channels, geometry.video_tube_frames,
            geometry.video_frame_stride ? geometry.video_frame_stride : 1};
}

/** MP4 or WebM clips, demuxed to plan their tubes and decoded in software. */
struct videos_packer_t {
    static constexpr modality_t modality_k = modality_t::video_k;
    uform_tape_view_t tape;

    std::size_t cell_bytes(ue::geometry_t const &geometry) const noexcept {
        return std::size_t {geometry.video_scalars()};
    }

    failure_t validate(ue::geometry_t const &geometry) const noexcept {
        if (!geometry.video_patch_side || !geometry.video_tube_frames)
            return {uform_status_unsupported_k, "The model was not trained on video"};
        if (geometry.video_channels != um::video_channels_k)
            return {uform_status_unsupported_k, "The model's video frames are not RGB"};
        return {};
    }

    failure_t prepare(chunked_arena_t &, std::size_t, ue::geometry_t const &) noexcept { return {}; }

    failure_t plan(std::size_t item, item_plan_t &plan, ue::geometry_t const &geometry, std::size_t) const noexcept {
        core::expected<um::demuxed_t> clip = um::demux_video(entry_of(tape, item, 1));
        if (!clip) return {media_status_of(clip.status_), "A clip could not be demuxed"};
        core::video_batch_shape_t const shape = video_shape_of(geometry);
        um::video_grid_t const grid = um::video_tube_grid(clip->frames, clip->extent, shape, geometry.columns);
        std::size_t const sampled = core::divide_round_up(clip->frames ? clip->frames : 1, shape.frame_stride);
        plan.tokens = static_cast<u32_t>(grid.temporal_groups) * grid.patch_rows * grid.patch_columns;
        plan.spanned = core::divide_round_up(sampled, shape.tube_frames) *
                       native_patches(clip->extent, geometry.video_patch_side);
        plan.shape[0] = grid.temporal_groups, plan.shape[1] = grid.patch_rows, plan.shape[2] = grid.patch_columns;
        return {};
    }

    failure_t stage(std::size_t item, item_plan_t const &plan, b8_t *cells, ue::geometry_t const &geometry,
                    std::size_t) const noexcept {
        core::expected<um::demuxed_t> clip = um::demux_video(entry_of(tape, item, 1));
        if (!clip) return {media_status_of(clip.status_), "A clip could not be demuxed"};
        core::video_batch_shape_t const shape = video_shape_of(geometry);
        std::size_t const side = geometry.video_patch_side;
        um::pixel_tubes_span_t const tubes {static_cast<pixel_t *>(static_cast<void *>(cells)),
                                            {plan.shape[0], std::size_t {plan.shape[1]} * plan.shape[2],
                                             shape.tube_frames, side, side, um::video_channels_k}};
        core::expected_t const decoded = um::decode_video_frames(*clip, {plan.shape[0], plan.shape[1], plan.shape[2]},
                                                                 shape, tubes);
        if (!decoded) return {media_status_of(decoded.status_), "A clip could not be decoded"};
        return {};
    }
};

/** WAV, FLAC, MP3 or Opus files, decoded into the whole context once to find their length, then into their frames. */
struct recordings_packer_t {
    static constexpr modality_t modality_k = modality_t::audio_k;
    uform_tape_view_t tape;

    /** One context of samples per thread, which a plan decodes into. */
    f32_t *window = nullptr;
    std::size_t window_samples = 0;

    std::size_t cell_bytes(ue::geometry_t const &geometry) const noexcept {
        return std::size_t {geometry.audio_chunk_samples} * sizeof(f32_t);
    }
    failure_t validate(ue::geometry_t const &geometry) const noexcept { return validate_audio(geometry); }

    failure_t prepare(chunked_arena_t &staging, std::size_t threads, ue::geometry_t const &geometry) noexcept {
        window_samples = std::size_t {core::media_cells(core::modality_t::audio_k, geometry.columns)} *
                         geometry.audio_chunk_samples;
        window = staging.allocate<f32_t>(window_samples * threads);
        if (!window) return {uform_status_out_of_memory_k, "Failed to allocate the recording staging"};
        return {};
    }

    /** The decoder pads a short clip with exact silence, so its frames end with the last nonzero sample; a clip
     *  reaching the window's last frame counts as cut there. */
    failure_t plan(std::size_t item, item_plan_t &plan, ue::geometry_t const &geometry,
                   std::size_t thread) const noexcept {
        f32_t *const samples = window + thread * window_samples;
        core::expected_t const decoded = um::decode_audio(entry_of(tape, item, 1), geometry.audio_sample_rate,
                                                          {samples, window_samples});
        if (!decoded) return {media_status_of(decoded.status_), "A recording could not be decoded"};
        std::size_t heard = window_samples;
        while (heard && samples[heard - 1] == 0.0f) --heard;
        std::size_t const frames = core::divide_round_up(heard, geometry.audio_chunk_samples);
        plan.tokens = static_cast<u32_t>(frames ? frames : 1);
        plan.spanned = frames == core::media_cells(core::modality_t::audio_k, geometry.columns) ? frames + 1
                                                                                                : plan.tokens;
        return {};
    }

    failure_t stage(std::size_t item, item_plan_t const &plan, b8_t *cells, ue::geometry_t const &geometry,
                    std::size_t) const noexcept {
        core::expected_t const decoded = um::decode_audio(entry_of(tape, item, 1), geometry.audio_sample_rate,
                                                          {static_cast<f32_t *>(static_cast<void *>(cells)),
                                                           std::size_t {plan.tokens} * geometry.audio_chunk_samples});
        if (!decoded) return {media_status_of(decoded.status_), "A recording could not be decoded"};
        return {};
    }
};

#pragma endregion Packers

extern "C" {

UFORM_EXPORT void uform_image_encode(uform_image_encode_t *arguments) {
    if (submitted<uform_image_encode>(arguments)) return;
    call_outcome_t outcome(arguments->outcome, arguments->arena);
    uform_tape_view_t const tape = arguments->images;
    images_packer_t packer {tape};
    encode_call(*arguments, packer, tape, uform_scalar_u8_k, {tape.data, tape.offsets}, outcome);
}

UFORM_EXPORT void uform_video_encode(uform_video_encode_t *arguments) {
    if (submitted<uform_video_encode>(arguments)) return;
    call_outcome_t outcome(arguments->outcome, arguments->arena);
    uform_tape_view_t const tape = arguments->videos;
    videos_packer_t packer {tape};
    encode_call(*arguments, packer, tape, uform_scalar_u8_k, {tape.data, tape.offsets}, outcome);
}

UFORM_EXPORT void uform_audio_encode(uform_audio_encode_t *arguments) {
    if (submitted<uform_audio_encode>(arguments)) return;
    call_outcome_t outcome(arguments->outcome, arguments->arena);
    uform_tape_view_t const tape = arguments->recordings;
    recordings_packer_t packer {tape};
    encode_call(*arguments, packer, tape, uform_scalar_u8_k, {tape.data, tape.offsets}, outcome);
}

} // extern "C"
