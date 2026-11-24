/**
 *  @file cpp/uform/model.hpp
 *  @author Ash Vardanian
 *  @date November 18, 2026
 *  @brief The model lifecycle of @c uform/model.h and the verbs its receivers run: describing the
 *      model, and every forward pass of @c uform/encode.h.
 *
 *  A model answers its verbs through a receiver, whose type picks how they complete: the model
 *  itself blocks on the calling thread and @c on blocks on an executor, both answering a
 *  @c status_t or an @c expected, while @c deferred submits to an executor and answers a
 *  @c deferred call its one owner polls, waits on or awaits. Each verb answers the statuses
 *  that its C function lists.
 *
 *  @code{.cpp}
 *  namespace uf = unum::uform;
 *  auto [executor, executor_status] = uf::executor_t::for_ram();
 *  auto [arena, arena_status] = uf::arena_t::make({.executor = executor.handle()});
 *  auto [model, model_status] = uf::model_t::load_ram(
 *      {.path = uf::bytes_view("model.safetensors"), .quantization = uform_scalar_bf16_k});
 *  auto [texts, texts_status] = model.on(executor).text_encode(captions, uf::encode_outputs_t::embeddings_k,
 *                                                              uform_scalar_f32_k, arena);
 *  @endcode
 */
#pragma once

#include <cstddef> // `std::size_t`
#include <cstdint> // `std::uint8_t`, `std::uint32_t`

#include <span> // `std::span`

#include "uform/model.h"    // `uform_model_*` verbs and their argument structs
#include "uform/encode.hpp" // `encoded_t`, `encode_outputs_t`, `kept_surfaces_`, `encode_config`
#include "uform/types.hpp"  // `deferrable`, `answered_`, `opened_`, `owner`, `executor_t`, `arena_t`

namespace unum::uform {

/**
 *  @brief A model and its verbs, completed as @p receiver_type_ says: the owning handle blocks on
 *      the calling thread, @c on blocks on an executor, and both answer a @c status_t or an
 *      @c expected, while @c deferred submits to an executor and answers a @c deferred call.
 *
 *  Each verb is written once, filling its C struct with the result as its output. Every input,
 *  handler, report and arena a deferred call names stays the caller's until it is final.
 */
template <typename receiver_type_>
class model {
    static constexpr uform_completion_t completion_k = receiver_type_::completion;
    template <typename value_type_ = void>
    using deferrable = uform::deferrable<value_type_, completion_k>;
    receiver_type_ receiver_;

    template <typename>
    friend class model;
    explicit model(receiver_type_ receiver) noexcept : receiver_(std::move(receiver)) {}

  public:
    model() noexcept
        requires owns_handle<receiver_type_>
    = default;

    uform_model_t handle() const noexcept { return receiver_.handle(); }

    /** The output field a lifecycle verb writes the new handle into. */
    uform_model_t* handle_field() noexcept
        requires owns_handle<receiver_type_>
    {
        return receiver_.handle_field();
    }

    /** The same handle, whose verbs block on @p executor. */
    model<caller_owned<uform_model_t, uform_completion_blocking_k>> on(executor_t const& executor) const noexcept {
        return model<caller_owned<uform_model_t, uform_completion_blocking_k>>({receiver_.handle(), executor.handle()});
    }

    /** The same handle, whose verbs submit to @p executor, each answering a @c deferred call. */
    model<caller_owned<uform_model_t, uform_completion_deferred_k>> deferred(
        executor_t const& executor) const noexcept {
        return model<caller_owned<uform_model_t, uform_completion_deferred_k>>({receiver_.handle(), executor.handle()});
    }

    /** Reports the model's geometry, placement, classes and heads in one call. */
    deferrable<uform_model_description_t> describe() const noexcept {
        return answered_<deferrable<uform_model_description_t>>(
            [&](uform_outcome_t* outcome, outputs<uform_model_description_t>* written) noexcept {
                uform_model_describe_t arguments {receiver_.handle(), receiver_.executor(), outcome, completion_k,
                                                  &written->value};
                uform_model_describe(&arguments);
            });
    }

    /** Encodes UTF-8 @p texts through one forward, a token per byte, keeping the surfaces @p kept
     *  names in @p arena. */
    deferrable<encoded_t> text_encode(tape_view<char> texts, encode_outputs_t kept, text_encode_config_t const& config,
                                      arena_t& arena) const noexcept {
        if (status_t const refusal = texts.check(); refusal.failed()) return answered_<deferrable<encoded_t>>(refusal);
        return answered_<deferrable<encoded_t>>([&](uform_outcome_t* outcome, outputs<encoded_t>* written) noexcept {
            uform_text_encode_t arguments {
                .model = receiver_.handle(),
                .executor = receiver_.executor(),
                .outcome = outcome,
                .completion = completion_k,
                .count = texts.count(),
                .texts = texts.view(),
                .arena = arena.handle(),
                .outputs_scalar = config.outputs_scalar,
                .progress = config.progress.raw(),
                .stats = config.stats,
            };
            kept_surfaces_(arguments, kept, written->value, texts.count());
            uform_text_encode(&arguments);
        });
    }

    /** Scores each item's legal @p choices, one byte each, after its UTF-8 state in @p texts,
     *  answering one probability per choice in @p arena, a softmax over the item's choices. */
    deferrable<tape_view<float>> text_choose(tape_view<char> texts, tape_view<char> choices,
                                             text_choose_config_t const& config, arena_t& arena) const noexcept {
        if (status_t const refusal = texts.check(); refusal.failed())
            return answered_<deferrable<tape_view<float>>>(refusal);
        if (status_t const refusal = choices.check(); refusal.failed())
            return answered_<deferrable<tape_view<float>>>(refusal);
        // The call has one count, so only this side sees the texts, choices and groups disagree.
        if (choices.count() != texts.count() ||
            (!config.temperature_groups.empty() && config.temperature_groups.size() != texts.count()))
            return answered_<deferrable<tape_view<float>>>(
                {status_code_t::shape_mismatch_k, "Every item needs its choices and, if any, one group"});
        return answered_<deferrable<tape_view<float>>>(
            [&](uform_outcome_t* outcome, outputs<tape_view<float>>* written) noexcept {
                written->count = texts.count();
                uform_text_choose_t arguments {
                    .model = receiver_.handle(),
                    .executor = receiver_.executor(),
                    .outcome = outcome,
                    .completion = completion_k,
                    .count = texts.count(),
                    .texts = texts.view(),
                    .choices = choices.view(),
                    .temperature_groups = config.temperature_groups.data(),
                    .probabilities = &written->written,
                    .arena = arena.handle(),
                    .progress = config.progress.raw(),
                    .stats = config.stats,
                };
                uform_text_choose(&arguments);
            });
    }

    /** Decodes and encodes JPEG, PNG or WebP @p images; an undecodable file fails the call with
     *  @c status_code_t::malformed_k. */
    deferrable<encoded_t> image_encode(tape_view<std::uint8_t> images, encode_outputs_t kept,
                                       image_encode_config_t const& config, arena_t& arena) const noexcept {
        if (status_t const refusal = images.check(); refusal.failed()) return answered_<deferrable<encoded_t>>(refusal);
        return answered_<deferrable<encoded_t>>([&](uform_outcome_t* outcome, outputs<encoded_t>* written) noexcept {
            uform_image_encode_t arguments {
                .model = receiver_.handle(),
                .executor = receiver_.executor(),
                .outcome = outcome,
                .completion = completion_k,
                .count = images.count(),
                .images = images.view(),
                .arena = arena.handle(),
                .outputs_scalar = config.outputs_scalar,
                .progress = config.progress.raw(),
                .stats = config.stats,
            };
            kept_surfaces_(arguments, kept, written->value, images.count());
            uform_image_encode(&arguments);
        });
    }

    /** Encodes decoded images of interleaved 8-bit RGB, item @c i holding @p heights [i] rows of
     *  @p widths [i] pixels. */
    deferrable<encoded_t> image_encode_pixels(tape_view<std::uint8_t> pixels, std::span<std::uint32_t const> heights,
                                              std::span<std::uint32_t const> widths, encode_outputs_t kept,
                                              image_encode_config_t const& config, arena_t& arena) const noexcept {
        if (status_t const refusal = pixels.check(); refusal.failed()) return answered_<deferrable<encoded_t>>(refusal);
        // The call has one count, so only this side sees the pixels and their shapes disagree.
        if (heights.size() != pixels.count() || widths.size() != pixels.count())
            return answered_<deferrable<encoded_t>>(
                {status_code_t::shape_mismatch_k, "Every item needs one height and one width"});
        return answered_<deferrable<encoded_t>>([&](uform_outcome_t* outcome, outputs<encoded_t>* written) noexcept {
            uform_image_encode_pixels_t arguments {
                .model = receiver_.handle(),
                .executor = receiver_.executor(),
                .outcome = outcome,
                .completion = completion_k,
                .count = pixels.count(),
                .pixels = pixels.view(),
                .heights = heights.data(),
                .widths = widths.data(),
                .arena = arena.handle(),
                .outputs_scalar = config.outputs_scalar,
                .progress = config.progress.raw(),
                .stats = config.stats,
            };
            kept_surfaces_(arguments, kept, written->value, pixels.count());
            uform_image_encode_pixels(&arguments);
        });
    }

    /** Decodes and encodes MP4 or WebM @p videos. */
    deferrable<encoded_t> video_encode(tape_view<std::uint8_t> videos, encode_outputs_t kept,
                                       video_encode_config_t const& config, arena_t& arena) const noexcept {
        if (status_t const refusal = videos.check(); refusal.failed()) return answered_<deferrable<encoded_t>>(refusal);
        return answered_<deferrable<encoded_t>>([&](uform_outcome_t* outcome, outputs<encoded_t>* written) noexcept {
            uform_video_encode_t arguments {
                .model = receiver_.handle(),
                .executor = receiver_.executor(),
                .outcome = outcome,
                .completion = completion_k,
                .count = videos.count(),
                .videos = videos.view(),
                .arena = arena.handle(),
                .outputs_scalar = config.outputs_scalar,
                .progress = config.progress.raw(),
                .stats = config.stats,
            };
            kept_surfaces_(arguments, kept, written->value, videos.count());
            uform_video_encode(&arguments);
        });
    }

    /** Decodes and encodes WAV, FLAC, MP3 or Opus @p recordings, resampled to mono. */
    deferrable<encoded_t> audio_encode(tape_view<std::uint8_t> recordings, encode_outputs_t kept,
                                       audio_encode_config_t const& config, arena_t& arena) const noexcept {
        if (status_t const refusal = recordings.check(); refusal.failed())
            return answered_<deferrable<encoded_t>>(refusal);
        return answered_<deferrable<encoded_t>>([&](uform_outcome_t* outcome, outputs<encoded_t>* written) noexcept {
            uform_audio_encode_t arguments {
                .model = receiver_.handle(),
                .executor = receiver_.executor(),
                .outcome = outcome,
                .completion = completion_k,
                .count = recordings.count(),
                .recordings = recordings.view(),
                .arena = arena.handle(),
                .outputs_scalar = config.outputs_scalar,
                .progress = config.progress.raw(),
                .stats = config.stats,
            };
            kept_surfaces_(arguments, kept, written->value, recordings.count());
            uform_audio_encode(&arguments);
        });
    }

    /** Encodes decoded mono @p samples sharing one @p sample_rate, resampled to the model's rate
     *  where it differs. */
    deferrable<encoded_t> audio_encode_samples(tape_view<float> samples, std::uint32_t sample_rate,
                                               encode_outputs_t kept, audio_encode_config_t const& config,
                                               arena_t& arena) const noexcept {
        if (status_t const refusal = samples.check(); refusal.failed())
            return answered_<deferrable<encoded_t>>(refusal);
        return answered_<deferrable<encoded_t>>([&](uform_outcome_t* outcome, outputs<encoded_t>* written) noexcept {
            uform_audio_encode_samples_t arguments {
                .model = receiver_.handle(),
                .executor = receiver_.executor(),
                .outcome = outcome,
                .completion = completion_k,
                .count = samples.count(),
                .samples = samples.view(),
                .sample_rate = sample_rate,
                .arena = arena.handle(),
                .outputs_scalar = config.outputs_scalar,
                .progress = config.progress.raw(),
                .stats = config.stats,
            };
            kept_surfaces_(arguments, kept, written->value, samples.count());
            uform_audio_encode_samples(&arguments);
        });
    }

    /** Copies a safetensors checkpoint file into host memory and packs it for the CPU, blocking. */
    static expected<model> load_ram(uform_model_load_ram_t arguments) noexcept
        requires owns_handle<receiver_type_>
    {
        arguments.completion = uform_completion_blocking_k;
        return opened_<model>(uform_model_load_ram, arguments, &uform_model_load_ram_t::model);
    }

    /** Copies a checkpoint held in memory into host memory and packs it for the CPU, blocking. */
    static expected<model> load_ram_buffer(uform_model_load_ram_buffer_t arguments) noexcept
        requires owns_handle<receiver_type_>
    {
        arguments.completion = uform_completion_blocking_k;
        return opened_<model>(uform_model_load_ram_buffer, arguments, &uform_model_load_ram_buffer_t::model);
    }

    /** Copies a checkpoint file onto the GPUs @p arguments names and packs it there, blocking. */
    static expected<model> load_vram(uform_model_load_vram_t arguments) noexcept
        requires owns_handle<receiver_type_>
    {
        arguments.completion = uform_completion_blocking_k;
        return opened_<model>(uform_model_load_vram, arguments, &uform_model_load_vram_t::model);
    }

    /** Copies a checkpoint held in memory onto the GPUs @p arguments names, blocking. */
    static expected<model> load_vram_buffer(uform_model_load_vram_buffer_t arguments) noexcept
        requires owns_handle<receiver_type_>
    {
        arguments.completion = uform_completion_blocking_k;
        return opened_<model>(uform_model_load_vram_buffer, arguments, &uform_model_load_vram_buffer_t::model);
    }
};

/** Owns one loaded encoder, shared by every executor of its residency; destruction waits for its
 *  deferred calls. */
using model_t = model<owning<uform_model_t, uform_model_free>>;

/** A model the caller owns and its executor, completing its verbs as @p completion_ says. */
template <uform_completion_t completion_>
using model_on = model<caller_owned<uform_model_t, completion_>>;

} // namespace unum::uform
