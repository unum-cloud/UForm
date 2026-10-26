/**
 *  @file src/encode_media_absent.cpp
 *  @author Ash Vardanian
 *  @date October 18, 2026
 *  @brief What a build without the codecs links in place of @c encode_media.cpp and libswresample: the verbs over
 *      encoded files and the resampler, each reporting @c uform_status_unsupported_k .
 *
 *  WebAssembly builds no codec, so the browser decodes with @c createImageBitmap or WebAudio, at the model's rate,
 *  and calls @c uform_image_encode_pixels or @c uform_audio_encode_samples .
 */
#include "uform/media/audio.hpp" // `resample_mono`

#include "encode.hpp"

using namespace unum::uform;

/** Finishes a verb over encoded files as unsupported, its outputs null. */
template <typename arguments_type_>
static void refuse_files(arguments_type_ &arguments, call_outcome_t &outcome) noexcept {
    clear_outputs(arguments);
    propagate_error(uform_status_unsupported_k, outcome, "This build decodes no files; pass decoded pixels or samples");
}

um::expected_t um::resample_mono(std::span<f32_t const>, u32_t, u32_t, f32r1_span_t) noexcept {
    return expected_t::fail(status_t::shape_unsupported_k, "this build has no resampler");
}

extern "C" {

UFORM_EXPORT void uform_image_encode(uform_image_encode_t *arguments) {
    if (submitted<uform_image_encode>(arguments)) return;
    call_outcome_t outcome(arguments->outcome, arguments->arena);
    refuse_files(*arguments, outcome);
}

UFORM_EXPORT void uform_video_encode(uform_video_encode_t *arguments) {
    if (submitted<uform_video_encode>(arguments)) return;
    call_outcome_t outcome(arguments->outcome, arguments->arena);
    refuse_files(*arguments, outcome);
}

UFORM_EXPORT void uform_audio_encode(uform_audio_encode_t *arguments) {
    if (submitted<uform_audio_encode>(arguments)) return;
    call_outcome_t outcome(arguments->outcome, arguments->arena);
    refuse_files(*arguments, outcome);
}

} // extern "C"
