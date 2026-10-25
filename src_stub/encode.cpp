/**
 *  @file src_stub/encode.cpp
 *  @author Ash Vardanian
 *  @date October 17, 2026
 *  @brief A stand-in for the @c uform_*_encode C ABI: deterministic rows of the right shape, so a binding
 *      checks offsets, truncation stats, scalars and arena lifetimes without a tower.
 *
 *  Every verb decodes nothing and runs no model. An item's pooled embedding is a run of xorshift draws seeded
 *  by the FNV-1a hash of its bytes, and each of its states is seeded by that hash continued over the token's
 *  position, so equal inputs land equal outputs under any completion. What a token is depends on the verb:
 *
 *      Verb                  One token
 *      text                  one byte of UTF-8
 *      image, video, audio   16 bytes of the encoded file
 *      image pixels          one 8 x 8 patch of pixels
 *      audio samples         256 samples
 *
 *  An item spanning more than 512 tokens keeps its first 512 and counts as truncated.
 */
#include <algorithm>        // `std::min`
#include <array>            // `std::array` - a token's position as bytes
#include <bit>              // `std::bit_cast`
#include <initializer_list> // `std::initializer_list`

#include "lib.hpp"

using namespace unum::uform;

/** One item as a verb reads it: the bytes seeding its rows, and the tokens they make before truncation. */
struct item_t {
    std::byte const *data;
    std::size_t length;
    std::size_t tokens;
};

/** The bytes of entry @p item of @p tape, whose elements are @p element_bytes wide. */
static item_t entry_of(uform_tape_view_t tape, std::size_t item, std::size_t element_bytes) {
    std::size_t const begin = tape.offsets[item], end = tape.offsets[item + 1];
    return {static_cast<std::byte const *>(tape.data) + begin * element_bytes, (end - begin) * element_bytes,
            end - begin};
}

/** Zeroes every output and report, so a call refused before it lands leaves them null. */
template <typename arguments_>
static void clear_outputs(arguments_ &arguments) {
    if (arguments.embeddings) *arguments.embeddings = {};
    if (arguments.states) *arguments.states = {};
    if (arguments.token_vectors) *arguments.token_vectors = {};
    if (arguments.stats) *arguments.stats = {};
}

/** The refusals every encode verb shares: no model, no output named, token vectors the stand-in's
 *  model has no head for, an output scalar it does not write, and a tape without storage, of
 *  another element, or with decreasing offsets. */
template <typename arguments_>
static bool reject_call(arguments_ const &arguments, uform_tape_view_t tape, uform_scalar_t element,
                        uform_outcome_t *outcome) {
    if (reject_if(!arguments.model, uform_status_invalid_argument_k, "The model must not be null", outcome))
        return true;
    if (reject_if(!arguments.embeddings && !arguments.states && !arguments.token_vectors,
                  uform_status_invalid_argument_k, "Name the embeddings, the states or the token vectors", outcome))
        return true;
    if (reject_if(arguments.token_vectors, uform_status_unsupported_k, "The model carries no token head", outcome))
        return true;
    if (reject_if(!output_scalar_bytes(arguments.outputs_scalar), uform_status_unsupported_k,
                  "Outputs land as f32, f16 or bf16 only", outcome))
        return true;
    if (reject_if(!tape.offsets || (!tape.data && arguments.count), uform_status_invalid_argument_k,
                  "The tape must name storage", outcome))
        return true;
    if (reject_if(tape.scalar != element, uform_status_unsupported_layout_k, "The tape holds another scalar", outcome))
        return true;
    for (std::size_t item = 0; item != arguments.count; ++item)
        if (reject_if(tape.offsets[item + 1] < tape.offsets[item], uform_status_malformed_k,
                      "The tape's offsets must not decrease", outcome))
            return true;
    return false;
}

/**
 *  @brief Lands the outputs of @p arguments in its arena: one embedding per item, and one state per token
 *      kept, item by item, reporting after each slice of items.
 *  @param[in] inputs Every pointer the call reads, refused when it lies inside the arena.
 *  @param[in] item_at Reads item @c i as an @c item_t.
 */
template <typename arguments_, typename item_at_>
static void encode(arguments_ &arguments, std::initializer_list<void const *> inputs, item_at_ item_at,
                   uform_outcome_t *outcome) {
    uform_arena_handle_t *const arena = claim_arena(arguments.arena, arguments.outcome, inputs, outcome);
    if (!arena) return;
    std::size_t const count = arguments.count;
    std::size_t tokens = 0, truncated = 0;
    for (std::size_t item = 0; item != count; ++item) {
        std::size_t const spanned = item_at(item).tokens;
        truncated += spanned > context_k;
        tokens += std::min(spanned, context_k);
    }

    uform_scalar_t const scalar = arguments.outputs_scalar;
    std::size_t const row_bytes = dimensions_k * output_scalar_bytes(scalar);
    std::size_t const offsets_bytes = arguments.states ? (count + 1) * sizeof(std::size_t) : 0;
    std::size_t const embeddings_bytes = arguments.embeddings ? count * row_bytes : 0;
    std::size_t const states_bytes = arguments.states ? tokens * row_bytes : 0;
    void *const base = rewind(*arena, offsets_bytes + embeddings_bytes + states_bytes, outcome);
    if (!base) return;
    std::size_t *const offsets = arguments.states ? static_cast<std::size_t *>(base) : nullptr;
    std::byte *const embeddings = static_cast<std::byte *>(base) + offsets_bytes; // rows after offsets stay aligned
    std::byte *const states = embeddings + embeddings_bytes;
    if (offsets) offsets[0] = 0;

    progress_ticks_t const ticks {arguments.progress, count};
    std::size_t item = 0, row = 0;
    for (std::size_t slice = 0; slice != ticks.slices(); ++slice) {
        for (; item != ticks.end_of(slice); ++item) {
            item_t const entry = item_at(item);
            std::uint64_t const seed = fnv1a(entry.data, entry.length);
            std::size_t const kept = std::min(entry.tokens, context_k);
            if (arguments.embeddings) write_row(embeddings + item * row_bytes, scalar, seed);
            if (offsets) {
                for (std::size_t token = 0; token != kept; ++token) {
                    std::array<std::byte, sizeof(std::uint64_t)> const position =
                        std::bit_cast<std::array<std::byte, sizeof(std::uint64_t)>>(std::uint64_t {token});
                    std::uint64_t const token_seed = fnv1a(position.data(), position.size(), seed);
                    write_row(states + (row + token) * row_bytes, scalar, token_seed);
                }
                offsets[item + 1] = row + kept;
            }
            row += kept;
        }
        if (reject_if(!ticks.carry_on(item), uform_status_cancelled_k, "The caller's handler stopped the call",
                      outcome))
            return;
    }

    if (arguments.embeddings) *arguments.embeddings = {embeddings, row_bytes, dimensions_k, scalar};
    if (arguments.states) *arguments.states = {{states, row_bytes, dimensions_k, scalar}, offsets};
    if (arguments.stats) *arguments.stats = {truncated, tokens};
}

/** Encodes a verb whose items are encoded files, a token per 16 of their bytes. */
template <typename arguments_>
static void encode_files(arguments_ &arguments, uform_tape_view_t files, uform_outcome_t *outcome) {
    clear_outputs(arguments);
    if (reject_call(arguments, files, uform_scalar_u8_k, outcome)) return;
    encode(
        arguments, {files.data, files.offsets},
        [&](std::size_t item) {
            item_t entry = entry_of(files, item, 1);
            entry.tokens = divide_round_up(entry.length, file_token_bytes_k);
            return entry;
        },
        outcome);
}

extern "C" {

void uform_text_encode(uform_text_encode_t *arguments) {
    if (through_executor<uform_text_encode>(arguments)) return;
    deferred_outcome_t outcome(arguments->outcome, arguments->arena);
    clear_outputs(*arguments);
    uform_tape_view_t const texts = arguments->texts;
    if (reject_call(*arguments, texts, uform_scalar_u8_k, outcome)) return;
    encode(
        *arguments, {texts.data, texts.offsets}, [&](std::size_t item) { return entry_of(texts, item, 1); }, outcome);
}

void uform_text_choose(uform_text_choose_t *arguments) {
    if (through_executor<uform_text_choose>(arguments)) return;
    deferred_outcome_t outcome(arguments->outcome, arguments->arena);
    if (arguments->probabilities) *arguments->probabilities = {};
    if (arguments->stats) *arguments->stats = {};
    if (reject_if(!arguments->model, uform_status_invalid_argument_k, "The model must not be null", outcome)) return;
    reject_if(true, uform_status_unsupported_k, "The model carries no byte decoder", outcome);
}

void uform_image_encode(uform_image_encode_t *arguments) {
    if (through_executor<uform_image_encode>(arguments)) return;
    deferred_outcome_t outcome(arguments->outcome, arguments->arena);
    encode_files(*arguments, arguments->images, outcome);
}

void uform_image_encode_pixels(uform_image_encode_pixels_t *arguments) {
    if (through_executor<uform_image_encode_pixels>(arguments)) return;
    deferred_outcome_t outcome(arguments->outcome, arguments->arena);
    clear_outputs(*arguments);
    uform_tape_view_t const pixels = arguments->pixels;
    if (reject_call(*arguments, pixels, uform_scalar_u8_k, outcome)) return;
    std::size_t const count = arguments->count;
    uint32_t const *const heights = arguments->heights;
    uint32_t const *const widths = arguments->widths;
    if (reject_if(count && (!heights || !widths), uform_status_invalid_argument_k,
                  "Every item needs its height and width", outcome))
        return;
    for (std::size_t item = 0; item != count; ++item)
        if (reject_if(entry_of(pixels, item, 1).length != std::size_t {heights[item]} * widths[item] * 3,
                      uform_status_shape_mismatch_k, "An item's bytes disagree with its height and width", outcome))
            return;
    encode(
        *arguments, {pixels.data, pixels.offsets, heights, widths},
        [&](std::size_t item) {
            item_t entry = entry_of(pixels, item, 1);
            entry.tokens = divide_round_up(heights[item], patch_side_k) * divide_round_up(widths[item], patch_side_k);
            return entry;
        },
        outcome);
}

void uform_video_encode(uform_video_encode_t *arguments) {
    if (through_executor<uform_video_encode>(arguments)) return;
    deferred_outcome_t outcome(arguments->outcome, arguments->arena);
    encode_files(*arguments, arguments->videos, outcome);
}

void uform_audio_encode(uform_audio_encode_t *arguments) {
    if (through_executor<uform_audio_encode>(arguments)) return;
    deferred_outcome_t outcome(arguments->outcome, arguments->arena);
    encode_files(*arguments, arguments->recordings, outcome);
}

void uform_audio_encode_samples(uform_audio_encode_samples_t *arguments) {
    if (through_executor<uform_audio_encode_samples>(arguments)) return;
    deferred_outcome_t outcome(arguments->outcome, arguments->arena);
    clear_outputs(*arguments);
    uform_tape_view_t const samples = arguments->samples;
    if (reject_call(*arguments, samples, uform_scalar_f32_k, outcome)) return;
    if (reject_if(!arguments->sample_rate, uform_status_invalid_argument_k, "The sample rate must not be zero",
                  outcome))
        return;
    encode(
        *arguments, {samples.data, samples.offsets},
        [&](std::size_t item) {
            item_t entry = entry_of(samples, item, sizeof(float));
            entry.tokens = divide_round_up(entry.tokens, chunk_samples_k);
            return entry;
        },
        outcome);
}

} // extern "C"
