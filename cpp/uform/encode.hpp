/**
 *  @file cpp/uform/encode.hpp
 *  @author Ash Vardanian
 *  @date November 18, 2026
 *  @brief What the forward passes of @c uform/encode.h take and answer: the surfaces a verb keeps,
 *      the views it writes, and the knobs and reports of each input form.
 *
 *  The verbs themselves are members of @c model in @c uform/model.hpp, since a receiver of a
 *  model is what runs them; this header holds only what they share.
 */
#pragma once

#include <cstddef> // `std::size_t`
#include <cstdint> // `std::uint32_t`

#include <span> // `std::span`

#include "uform/encode.h"  // `uform_*_encode_stats_t`, `uform_tape_view_t`
#include "uform/types.hpp" // `outputs`, `progress_t`, `tape_view`, `written_rows_t`, `written_rows_tape_t`

namespace unum::uform {

/** The surfaces an encode verb keeps from its one forward, named after the C fields they fill;
 *  @c operator| joins them. */
enum class encode_outputs_t : std::uint32_t {

    /** One pooled embedding per item, past the attention pool and the projection head. */
    embeddings_k = 1,

    /** One final-norm state per token left after truncation. */
    states_k = 2,

    /** One unit vector of the token head per token left after truncation; a model without the head
     *  fails the call with @c status_code_t::unsupported_k. */
    token_vectors_k = 4,
};

constexpr encode_outputs_t operator|(encode_outputs_t left, encode_outputs_t right) noexcept {
    return static_cast<encode_outputs_t>(static_cast<std::uint32_t>(left) | static_cast<std::uint32_t>(right));
}

/** Whether every surface @p surface names is kept by @p outputs. */
constexpr bool contains(encode_outputs_t outputs, encode_outputs_t surface) noexcept {
    return (static_cast<std::uint32_t>(outputs) & static_cast<std::uint32_t>(surface)) ==
           static_cast<std::uint32_t>(surface);
}

/** What an encode verb wrote into the arena it was given: the surfaces its @c encode_outputs_t
 *  named, each empty when left out, and none normalized. */
struct encoded_t {

    /** One pooled embedding of the model's dimensions per item. */
    written_rows_t embeddings {};

    /** One final-norm state per token left after truncation, item by item. */
    written_rows_tape_t states {};

    /** One unit vector of the model's token width per token left after truncation, item by item. */
    written_rows_tape_t token_vectors {};
};

/** Points @p arguments at the surfaces @p kept names inside @p written, sized for @p count items,
 *  and leaves the rest @c NULL, which the library skips. */
template <typename arguments_type_>
void kept_surfaces_(arguments_type_& arguments, encode_outputs_t kept, encoded_t& written, std::size_t count) noexcept {
    if (contains(kept, encode_outputs_t::embeddings_k)) {
        arguments.embeddings = &written.embeddings.view;
        written.embeddings.written = count;
    }
    if (contains(kept, encode_outputs_t::states_k)) {
        arguments.states = &written.states.view;
        written.states.written = count;
    }
    if (contains(kept, encode_outputs_t::token_vectors_k)) {
        arguments.token_vectors = &written.token_vectors.view;
        written.token_vectors.written = count;
    }
}

/** The knobs of an encode verb reporting @p stats_type_: the scalar every output lands as, which a
 *  bare @c uform_scalar_t converts into, a handler, and a report. */
template <typename stats_type_>
struct encode_config {

    /** The scalar type every output lands as. */
    uform_scalar_t outputs_scalar = uform_scalar_unknown_k;

    /** Hears the items encoded, and may cancel the call. */
    progress_t progress {};

    /** The report the call overwrites; @c nullptr skips it. */
    stats_type_* stats = nullptr;

    encode_config(uform_scalar_t scalar, progress_t handler = {}, stats_type_* report = nullptr) noexcept
        : outputs_scalar(scalar), progress(handler), stats(report) {}
};

/** The knobs of @c text_encode. */
using text_encode_config_t = encode_config<uform_text_encode_stats_t>;

/** The knobs of @c image_encode and @c image_encode_pixels. */
using image_encode_config_t = encode_config<uform_image_encode_stats_t>;

/** The knobs of @c video_encode. */
using video_encode_config_t = encode_config<uform_video_encode_stats_t>;

/** The knobs of @c audio_encode and @c audio_encode_samples. */
using audio_encode_config_t = encode_config<uform_audio_encode_stats_t>;

/** The knobs of @c text_choose. */
struct text_choose_config_t {

    /** The question type of each item, below the model's @c choice_temperature_groups; empty puts
     *  every item in group zero. */
    std::span<std::uint32_t const> temperature_groups {};

    /** Hears the items scored, and may cancel the call. */
    progress_t progress {};

    /** The report the call overwrites; @c nullptr skips it. */
    uform_text_encode_stats_t* stats = nullptr;
};

/** Where @c text_choose writes one probability per choice: the C tape and its count, from which
 *  the answer assembles a @c tape_view over the arena. */
template <>
struct outputs<tape_view<float>> {
    uform_tape_view_t written {};
    std::size_t count = 0;

    tape_view<float> answer() const noexcept {
        if (!written.offsets) return {};
        std::span<std::size_t const> const offsets {written.offsets, count + 1};
        return {{static_cast<float const*>(written.data), offsets.back()}, offsets};
    }
};

} // namespace unum::uform
