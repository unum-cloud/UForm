/**
 *  @file test/gradients.cu
 *  @author Ash Vardanian
 *  @date October 19, 2026
 *  @brief Backpropagation and optimizer conformance: every analytic gradient of the masked replica
 *      against central finite differences, the masking and initialization invariants convergence
 *      rests on, and Adafactor against a Float64 host reference.
 *
 *  The cases:
 *
 *      - @ref finite_differences over the token-only and the omni mixtures, covering the block
 *        backward, the embedding scatter, the media projections and the loss heads.
 *      - @ref test_token_head_finite_differences : the late-interaction token head and its join
 *        into the tower, under a linear loss over the token vectors, with prefixes and side-less
 *        segments left out.
 *      - @ref test_masked_text_rows : every text row with bytes has at least one masked
 *        position.
 *      - @ref test_masked_media_rows : each media head scores only the cells of its own modality.
 *      - @ref test_chance_cross_entropy : a freshly seeded model starts at the uniform byte
 *        prediction.
 *      - @ref test_adafactor_reference : two Adafactor steps on a matrix and a vector land where
 *        the paper's update, replayed in Float64 on the host, puts them, up to one BFloat16
 *        spacing.
 *      - @ref test_adafactor_small_steps : steps far below a BFloat16 spacing still move the
 *        weights by their mean.
 *      - @ref test_block_host_replay : one block's forward and attention backward over ragged,
 *        padded segments with M-RoPE coordinates on every axis against a Float64 host replay, the
 *        one check exact enough for the attention and rotary paths.
 *      - @ref test_block_recompute_replays : a block's forward replayed from its input stash after
 *        another block overwrote their shared scratch reproduces its backward bitwise.
 *      - @ref test_choices_cross_entropy : the calibrated choices' loss, their force on the answer
 *        rows and the decoder's gradient against a Float64 oracle, then a central difference.
 *      - @ref test_spectrum_replay : the coding rate, the Lipschitz penalty and their force on the
 *        pooled rows against a Float64 replay, whose own force a central difference checks.
 *
 *  @ref finite_differences perturbs each registered matrix along its own gradient, by steps rounded
 *  onto the BFloat16 grid, and compares the loss change of two forwards with the gradient's dot
 *  product with the rounded step. That covers the block backward, the embedding scatter, the media
 *  projections, the token cross-entropy head and the patch and frame MSE heads.
 *
 *  The forward and the backward both round their activations to BFloat16, so a forward's loss
 *  has a few thousandths of a nat of rounding noise and a finite difference agrees with the
 *  analytic gradient to a few percent, not to Float32 precision; a wrong Jacobian term misses by
 *  tens of percent, and the block replay checks the attention path exactly.
 */
#undef NDEBUG // ! A test's oracle must stay live in every build

#include <cmath>   // `std::fabs`, `std::sqrt`, `std::log`, `std::pow`, `std::frexp`, `std::ldexp`
#include <cstdlib> // `::setenv`
#include <cstdio>  // `std::remove`
#include <cstring> // `std::memcpy`, `std::memcmp`

#include <algorithm> // `std::fill`, `std::copy`
#include <limits>    // `std::numeric_limits`
#include <span>      // `std::span`

#include <cuda.h> // `cuInit`, `cuMemcpyHtoD`, `cuMemcpyDtoH`, `cuStreamSynchronize`

#include <numkong/reduce.hpp> // `nk::reduce_moments`

#include "uform/utilities/mesh.hpp"          // `mesh_topology_t`
#include "uform/train/optimizer.hpp"         // `adafactor_t`
#include "uform/train/pooled_spectrum.cuh"   // `distributed_pooled_spectrum_t`
#include "uform/train/transformer_block.cuh" // `transformer_block_bf16_t`, `transformer_scratch_t`
#include "uform/train/base.cuh"              // `base_replica_t`, `trainer_config_t`, `seed_weights`

#include "train/numerics.hpp" // `gpu_numerics_of`

#include "harness.hpp" // `run_test_case`, `expect_true`, `dot`, `narrow_to_bf16`, `widen_from_bf16`

namespace ue = unum::uform::encoder;
using ue::audio_batch_shape_t;
using ue::byte_pad_k;
using ue::byte_vocabulary_k;
using ue::image_batch_shape_t;
using ue::late_interaction_side_t;
using ue::modality_t;
using ue::multimodal_batch_on_gpu_t;
using ue::multimodal_batch_shape_t;
using ue::video_batch_shape_t;
namespace util = unum::uform::utilities;
using namespace unum::uform::train;
using namespace unum::uform::test;

/** Download the BFloat16 scalars of @p device through @p staging into @p widened , widened once by
 *  NumKong. */
[[nodiscard]] static bool download_widened_(std::span<util::bf16_t const> device, st::basic_vector<nk_bf16_t>& staging,
                                            st::basic_vector<util::f64_t>& widened) noexcept {
    std::size_t const count = device.size();
    if (st::failed(staging.resize(st::assume_uninitialized, count)) ||
        st::failed(widened.resize(st::assume_uninitialized, count)))
        return false;
    cuMemcpyDtoH(staging.data(), util::device_address(device.data()), count * sizeof(nk_bf16_t));
    return widen_from_bf16(staging, widened);
}

/** Narrow @p values onto the BFloat16 grid in place and upload them through @p staging to
 *  @p device , landed before the non-blocking stream reads them. */
[[nodiscard]] static bool upload_narrowed_(std::span<util::f64_t> values, st::basic_vector<nk_bf16_t>& staging,
                                           util::bf16_t* device) noexcept {
    if (st::failed(staging.resize(st::assume_uninitialized, values.size())) || !narrow_to_bf16(values, staging))
        return false;
    cuMemcpyHtoD(util::device_address(device), staging.data(), staging.size() * sizeof(nk_bf16_t));
    cuCtxSynchronize(); // a pageable copy may return before its DMA lands
    return true;
}

/** The GPU every case uses, owned by @c main : one stream whose context the cases enter, and the
 *  kernels they call. Each case allocates its own arena, so a case's allocations die with it. */
struct device_t {
    util::cuda_executor_t& executor;
    util::executor_stream_t& stream;
    util::gpu_numerics_t numerics {};
};

/** Bytes one case's arena reserves: the omni replica at the test geometry needs about a
 *  quarter of it. */
inline constexpr std::size_t case_arena_bytes_k = 1ull << 30;

/** The relative error a finite difference may show against the analytic gradient: the
 *  BFloat16 activations' loss noise over a quarter-nat step, with room left to catch a
 *  15% Jacobian error. */
inline constexpr util::f64_t finite_difference_tolerance_k = 0x1.0p-4;

/** A two-block, two-head tower over four rows of 64 columns, scoring the masked objective;
 *  @p is_omni lights every media slice, so the keyed partition holds one row of each modality. */
[[nodiscard]] static trainer_config_t test_geometry_(bool is_omni) noexcept {
    trainer_config_t config;
    config.batch = multimodal_batch_shape_t {4u, 4u, 64u};
    if (is_omni) {
        config.batch.image = image_batch_shape_t {8u, 3u};
        config.batch.video = video_batch_shape_t {8u, 3u, 2u, 1u};
        config.batch.audio = audio_batch_shape_t {256u, 16000u};
    }
    config.depth = 2;
    config.hidden = 256;
    config.objective = objective_t::masked_k;
    return config;
}

/** Build and seed a replica for @p config over @p arena , then draw and corrupt the batch of epoch
 *  @p epoch . */
[[nodiscard]] static util::expected<base_replica_t> make_replica_(util::gpu_arena_t& arena, device_t& device,
                                                                  trainer_config_t const& config,
                                                                  util::data_epoch_t epoch) noexcept {
    auto [replica, status] = base_replica_t::make(arena, config, 0, device.numerics);
    util::expected_t staged = status ? seed_weights(replica.weights, device.stream)
                                     : util::expected_t::fail(std::move(status));
    if (staged) staged = replica.draw(config, 0, epoch, device.stream);
    if (staged) staged = replica.prepare(config, epoch, device.stream);
    if (!staged) return util::expected<base_replica_t>::fail(std::move(staged));
    return util::expected<base_replica_t> {std::move(replica)};
}

/** Draw and corrupt the batch of epoch @p epoch into an existing @p replica . */
[[nodiscard]] static util::expected_t redraw_(base_replica_t& replica, trainer_config_t const& config,
                                              util::data_epoch_t epoch, util::executor_stream_t& stream) noexcept {
    util::expected_t staged = replica.draw(config, 0, epoch, stream);
    if (staged) staged = replica.prepare(config, epoch, stream);
    return staged;
}

/** One @c {sum,count} summand pair read back, zero when the head is absent. */
struct summand_pair_t {
    util::f64_t sum = 0, count = 0;
};

[[nodiscard]] static summand_pair_t read_pair_(util::f32r1_tensor_arena_t const& summands) noexcept {
    if (summands.empty()) return {};
    util::f32_t host[2] = {0.0f, 0.0f};
    cuMemcpyDtoH(host, util::device_address(summands.data()), sizeof(host));
    return summand_pair_t {host[0], host[1]};
}

/** Run the forward region of @p config from zeroed summands and return the composite loss the
 *  backward differentiates: every live head's weighted sum over its own count. */
[[nodiscard]] static util::expected<util::f64_t> forward_loss_(base_replica_t& replica, trainer_config_t const& config,
                                                               util::executor_stream_t& stream) noexcept {
    util::f32r1_tensor_arena_t* const pairs[] = {&replica.tokens_summands, &replica.image_summands,
                                                 &replica.video_summands, &replica.audio_summands};
    for (util::f32r1_tensor_arena_t* pair : pairs)
        if (!pair->empty())
            if (util::expected_t zeroed = util::zero_async(pair->span(), stream); !zeroed)
                return util::expected<util::f64_t>::fail(std::move(zeroed));
    if (util::expected_t forwarded = replica.forward_region(config, stream); !forwarded)
        return util::expected<util::f64_t>::fail(std::move(forwarded));
    if (CUresult result = cuStreamSynchronize(util::native_stream(stream)); result != CUDA_SUCCESS)
        return util::cuda_failure<util::f64_t>(result, "forward drain failed");
    util::f64_t loss = 0;
    for (util::f32r1_tensor_arena_t* pair : pairs) {
        summand_pair_t const read = read_pair_(*pair);
        if (read.count != 0) loss += read.sum / read.count;
    }
    return util::expected<util::f64_t> {loss};
}

/** Perturb every registered matrix along its own gradient and compare the central difference of the
 *  composite loss with the gradient's dot product with the BFloat16-rounded step. */
static void finite_differences(device_t& device, bool is_omni) {
    [[maybe_unused]] util::status_t const scope = (device.stream).bind();
    // One block keeps every perturbed matrix's loss change far above the BFloat16 noise of a deeper
    // stack; the block's own backward is checked exactly by block_host_replay.
    trainer_config_t config = test_geometry_(is_omni);
    config.depth = 1;
    config.seed = test_run.settings.seed;
    auto [arena, make_arena_status] = util::gpu_arena_t::make(0, case_arena_bytes_k);
    if (!expect_true(make_arena_status, "arena: {}", make_arena_status.description_)) return;
    auto [replica, make_replica_status] = make_replica_(arena, device, config, util::data_epoch_t {0});
    if (!expect_true(make_replica_status, "replica: {}", make_replica_status.description_)) return;
    auto [base_loss, forward_base_loss_status] = forward_loss_(replica, config, device.stream);
    if (!expect_true(forward_base_loss_status, "forward: {}", forward_base_loss_status.description_)) return;
    if (util::expected_t backward = replica.backward_region(device.stream);
        !expect_true(backward, "backward: {}", backward.description_))
        return;
    cuStreamSynchronize(util::native_stream(device.stream));

    st::basic_vector<util::f64_t> weights, gradient, plus, minus;
    st::basic_vector<nk_bf16_t> staging;
    for (util::differentiable_weights_t const& matrix : replica.weights) {
        std::size_t const scalars = matrix.weights.numel();
        if (!expect_true(download_widened_({matrix.weights.data(), scalars}, staging, weights) &&
                             download_widened_({matrix.weights_grad.data(), scalars}, staging, gradient) &&
                             !st::failed(plus.resize(st::assume_uninitialized, scalars)) &&
                             !st::failed(minus.resize(st::assume_uninitialized, scalars)) &&
                             !st::failed(staging.resize(st::assume_uninitialized, scalars)),
                         "host staging for {} failed", matrix.name))
            return;
        util::f64_t largest = 0;
        for (util::f64_t const value : gradient) largest = fu::max_of(largest, std::fabs(value));
        // The head and pool weights outside the masked objective take no gradient and move no loss.
        if (largest == 0) continue;
        util::f64_t const weight_rms = std::sqrt(dot(weights, weights) / static_cast<util::f64_t>(scalars));
        util::f64_t predicted = 0;
        // Double the step until the BF16-rounded move predicts an eighth of a nat, never past
        // the matrix's RMS.
        util::f64_t const first_step = largest / (8.0 * dot(gradient, gradient));
        for (util::f64_t step = first_step; step == first_step || (predicted < 0.125 && step <= weight_rms);
             step *= 2) {
            for (std::size_t index = 0; index != scalars; ++index)
                plus[index] = weights[index] + step * gradient[index] / largest,
                minus[index] = weights[index] - step * gradient[index] / largest;
            if (!expect_true(narrow_to_bf16(plus, staging) && narrow_to_bf16(minus, staging),
                             "host rounding for {} failed", matrix.name))
                return;
            predicted = dot(gradient, plus) - dot(gradient, minus);
        }
        bool uploaded = upload_narrowed_(plus, staging, matrix.weights.data());
        auto [loss_plus, forward_loss_plus_status] = forward_loss_(replica, config, device.stream);
        uploaded = uploaded && upload_narrowed_(minus, staging, matrix.weights.data());
        auto [loss_minus, forward_loss_minus_status] = forward_loss_(replica, config, device.stream);
        uploaded = uploaded && upload_narrowed_(weights, staging, matrix.weights.data());
        if (!expect_true(uploaded && forward_loss_plus_status && forward_loss_minus_status,
                         "perturbed forward of {} failed", matrix.name))
            return;
        util::f64_t const measured = loss_plus - loss_minus;
        util::f64_t const gap = std::fabs(measured - predicted), error = gap / fu::max_of(std::fabs(predicted), 1.0e-6);
        expect_true(error <= finite_difference_tolerance_k,
                    "{}: loss moved {:.6g}, the gradient predicts {:.6g}, off by {:.3g} relative (loss {:.4f})",
                    matrix.name, measured, predicted, error, base_loss);
    }
}

static void test_finite_differences_tokens(device_t& device) { finite_differences(device, false); }

static void test_finite_differences_omni(device_t& device) { finite_differences(device, true); }

/** Host copies of the batch's segment table. */
struct host_segments_t {
    st::basic_vector<util::u32_t> offsets {}, lengths {};
    st::basic_vector<modality_t> modality {};
};

[[nodiscard]] static host_segments_t read_segments_(multimodal_batch_on_gpu_t const& batch) noexcept {
    host_segments_t segments;
    std::size_t const count = batch.segment_lengths.extent(0);
    if (st::failed(segments.offsets.resize(count)) || st::failed(segments.lengths.resize(count)) ||
        st::failed(segments.modality.resize(count)))
        return {};
    cuMemcpyDtoH(segments.offsets.data(), util::device_address(batch.segment_offsets.data()),
                 count * sizeof(util::u32_t));
    cuMemcpyDtoH(segments.lengths.data(), util::device_address(batch.segment_lengths.data()),
                 count * sizeof(util::u32_t));
    cuMemcpyDtoH(segments.modality.data(), util::device_address(batch.segment_modality.data()),
                 count * sizeof(modality_t));
    return segments;
}

/** The linear loss @c ⟨R,V⟩ over the token head's unit vectors @c V after one tower forward,
 *  normalized in Float64 from the Float32 projections downloaded into @p rows , since the BFloat16
 *  vectors' rounding noise would swamp a central difference of the projection. */
[[nodiscard]] static util::expected<util::f64_t> token_head_loss_(base_replica_t& replica,
                                                                  st::basic_vector<util::f64_t> const& direction,
                                                                  st::basic_vector<util::f32_t>& rows,
                                                                  util::executor_stream_t& stream) noexcept {
    util::expected_t forwarded = replica.tower.pack_weights(stream);
    if (forwarded) forwarded = replica.tower.forward(replica.batch, {}, stream);
    if (!forwarded) return util::expected<util::f64_t>::fail(std::move(forwarded));
    cuStreamSynchronize(util::native_stream(stream));
    util::f32r2_view_t const projections = replica.tower.token_projections();
    std::size_t const tokens = projections.extent(0), width = projections.extent(1);
    if (st::failed(rows.resize(st::assume_uninitialized, tokens * width)))
        return util::expected<util::f64_t>::fail(util::status_code_t::out_of_memory_k,
                                                 "token projection staging failed");
    cuMemcpyDtoH(rows.data(), util::device_address(projections.data()), tokens * width * sizeof(util::f32_t));
    util::f32r2_view_t const projection_rows {rows.data(), tokens, width};
    f64r2_view_t const direction_rows {direction.data(), tokens, width};
    util::f64_t const epsilon = replica.tower.config().block.rms_eps;
    util::f64_t loss = 0;
    for (std::size_t token = 0; token != tokens; ++token) {
        util::f64_t squares = 0, along = 0;
        for (std::size_t channel = 0; channel != width; ++channel) {
            util::f64_t const value = projection_rows[token, channel];
            squares += value * value, along += value * direction_rows[token, channel];
        }
        loss += along / std::sqrt(fu::max_of(squares, epsilon));
    }
    return util::expected<util::f64_t> {loss};
}

/** The token head and its Float32 join into the tower: a fixed linear loss over the token vectors,
 *  whose gradient is its direction, against central differences of the projection and the final
 *  norm, and the norm scale against a Float64 replay, with each segment's prefix and the side-less
 *  segments left out of the compact rows. */
static void test_token_head_finite_differences(device_t& device) {
    [[maybe_unused]] util::status_t const scope = (device.stream).bind();
    trainer_config_t config = test_geometry_(false);
    config.depth = 1;
    config.seed = test_run.settings.seed;
    config.objective = objective_t::contrastive_k;
    config.late_interaction.weight = 1.0f;
    config.late_interaction.width = 32;
    auto [arena, make_arena_status] = util::gpu_arena_t::make(0, case_arena_bytes_k);
    if (!expect_true(make_arena_status, "arena: {}", make_arena_status.description_)) return;
    auto [replica, make_replica_status] = base_replica_t::make(arena, config, config.batch.max_segments_per_gpu,
                                                               device.numerics);
    util::expected_t staged = make_replica_status ? seed_weights(replica.weights, device.stream)
                                                  : util::expected_t::fail(std::move(make_replica_status));
    if (staged) staged = replica.draw(config, 0, util::data_epoch_t {0}, device.stream);
    if (!expect_true(staged, "replica: {}", staged.description_)) return;
    cuStreamSynchronize(util::native_stream(device.stream));

    // Sides and prefixes over the four drawn segments: a query, a document, a side-less segment and
    // one of both.
    host_segments_t const segments = read_segments_(replica.batch);
    std::size_t const count = segments.lengths.size();
    late_interaction_side_t const sides[] = {late_interaction_side_t::query_k, late_interaction_side_t::document_k,
                                             late_interaction_side_t::none_k,
                                             late_interaction_side_t::query_and_document_k};
    util::u32_t const prefixes[] = {3, 0, 2, 5};
    util::u32_t expected_offsets[5] = {};
    for (std::size_t segment = 0; segment != count; ++segment) {
        util::u32_t const length = segments.lengths[segment];
        bool const sided = sides[segment] != late_interaction_side_t::none_k;
        expected_offsets[segment + 1] = expected_offsets[segment] +
                                        (sided && length > prefixes[segment] ? length - prefixes[segment] : 0u);
    }
    cuMemcpyHtoD(util::device_address(replica.batch.segment_side.data()), sides, sizeof(sides));
    cuMemcpyHtoD(util::device_address(replica.batch.segment_prefix_length.data()), prefixes, sizeof(prefixes));

    util::bf16r2_view_t const token_vectors = replica.tower.token_vectors();
    st::basic_vector<util::f64_t> direction;
    st::basic_vector<nk_bf16_t> staging;
    st::basic_vector<util::f32_t> projection_staging;
    if (!expect_true(!st::failed(direction.resize(st::assume_uninitialized, token_vectors.numel())),
                     "direction staging failed"))
        return;
    for (std::size_t index = 0; index != direction.size(); ++index)
        direction[index] = util::f64_t((index * 2654435761ull + 40503ull) % 1000ull) / 1000.0 - 0.5;
    if (!expect_true(upload_narrowed_(direction, staging, replica.token_vectors_grad.data()),
                     "direction upload failed"))
        return;
    auto [base_loss, evaluate_base_loss_status] = token_head_loss_(replica, direction, projection_staging,
                                                                   device.stream);
    if (!expect_true(evaluate_base_loss_status, "forward: {}", evaluate_base_loss_status.description_)) return;
    util::u32_t offsets[5] = {};
    cuMemcpyDtoH(offsets, util::device_address(replica.tower.token_offsets().data()), sizeof(offsets));
    bool const laid_out = std::equal(offsets, offsets + count + 1, expected_offsets);
    expect_true(laid_out, "token head spans skip prefixes and side-less segments: total {} expected {}", offsets[count],
                expected_offsets[count]);
    if (util::expected_t backward = replica.tower.backward({}, {}, replica.batch, device.stream,
                                                           replica.token_vectors_grad.view());
        !expect_true(backward, "backward: {}", backward.description_))
        return;
    cuStreamSynchronize(util::native_stream(device.stream));

    // The norm scale's gradient against a Float64 replay of the head over the same states: through
    // the L2 normalization it moves the loss only at second order along most directions, too little
    // for a difference.
    st::basic_vector<util::f64_t> states, scale, scale_grad, projection;
    util::differentiable_weights_t const *norm = nullptr, *projected = nullptr;
    for (util::differentiable_weights_t const& matrix : replica.weights)
        if (matrix.name.view() == "token_head.norm") norm = &matrix;
        else if (matrix.name.view() == "token_head.projection") projected = &matrix;
    util::bf16r2_view_t const normalized = replica.tower.normalized();
    auto const [normalized_values, normalized_status] = normalized.values();
    std::size_t const hidden = config.hidden, width = config.late_interaction.width;
    if (!expect_true(norm != nullptr && projected != nullptr && nk::succeeded(normalized_status) &&
                         download_widened_(normalized_values, staging, states) &&
                         download_widened_({norm->weights.data(), hidden}, staging, scale) &&
                         download_widened_({norm->weights_grad.data(), hidden}, staging, scale_grad) &&
                         download_widened_({projected->weights.data(), width * hidden}, staging, projection),
                     "token head replay staging failed"))
        return;
    util::f64_t const epsilon = replica.tower.config().block.rms_eps;
    f64r2_view_t const projection_rows {projection.data(), width, hidden};
    f64r2_view_t const direction_rows {direction.data(), direction.size() / width, width};
    st::basic_vector<util::f64_t> replayed, unit, normed;
    if (!expect_true(!st::failed(replayed.resize(hidden)) && !st::failed(unit.resize(width)) &&
                         !st::failed(normed.resize(hidden)),
                     "token head replay staging failed"))
        return;
    for (std::size_t segment = 0; segment != count; ++segment)
        for (util::u32_t token = 0; token != expected_offsets[segment + 1] - expected_offsets[segment]; ++token) {
            std::size_t const row = expected_offsets[segment] + token;
            util::f64_t const* const state = states.data() +
                                             (segments.offsets[segment] + prefixes[segment] + token) * hidden;
            util::f64_t squares = 0;
            for (std::size_t channel = 0; channel != hidden; ++channel) squares += state[channel] * state[channel];
            util::f64_t const inverse_rms = 1.0 / std::sqrt(squares / util::f64_t(hidden) + epsilon);
            for (std::size_t channel = 0; channel != hidden; ++channel)
                normed[channel] = state[channel] * inverse_rms * scale[channel];
            util::f64_t norm_squared = 0, along = 0;
            for (std::size_t output = 0; output != width; ++output) {
                unit[output] = 0;
                for (std::size_t channel = 0; channel != hidden; ++channel)
                    unit[output] += projection_rows[output, channel] * normed[channel];
                norm_squared += unit[output] * unit[output];
            }
            util::f64_t const inverse_norm = 1.0 / std::sqrt(fu::max_of(norm_squared, epsilon));
            for (std::size_t output = 0; output != width; ++output)
                unit[output] *= inverse_norm, along += unit[output] * direction_rows[row, output];
            for (std::size_t channel = 0; channel != hidden; ++channel) {
                util::f64_t normed_grad = 0;
                for (std::size_t output = 0; output != width; ++output)
                    normed_grad += projection_rows[output, channel] *
                                   (direction_rows[row, output] - along * unit[output]) * inverse_norm;
                replayed[channel] += normed_grad * state[channel] * inverse_rms;
            }
        }
    util::f64_t gap = 0, magnitude = 0;
    for (std::size_t channel = 0; channel != hidden; ++channel)
        gap += (scale_grad[channel] - replayed[channel]) * (scale_grad[channel] - replayed[channel]),
            magnitude += replayed[channel] * replayed[channel];
    expect_true(std::sqrt(gap) <= 2.0e-2 * std::sqrt(magnitude),
                "token_head.norm gradient matches the Float64 replay: relative {:.3g}", std::sqrt(gap / magnitude));

    st::basic_vector<util::f64_t> weights, gradient, plus, minus;
    for (util::differentiable_weights_t const& matrix : replica.weights) {
        sz::string_view_t const name = matrix.name.view();
        if (name != "token_head.projection" && name != "final_norm") continue;
        std::size_t const scalars = matrix.weights.numel();
        if (!expect_true(download_widened_({matrix.weights.data(), scalars}, staging, weights) &&
                             download_widened_({matrix.weights_grad.data(), scalars}, staging, gradient) &&
                             !st::failed(plus.resize(st::assume_uninitialized, scalars)) &&
                             !st::failed(minus.resize(st::assume_uninitialized, scalars)),
                         "host staging for {} failed", name))
            return;
        util::f64_t largest = 0;
        for (util::f64_t const value : gradient) largest = fu::max_of(largest, std::fabs(value));
        if (largest == 0) {
            expect_true(false, "{} takes no gradient through the token head", name);
            continue;
        }
        util::f64_t const weight_rms = std::sqrt(dot(weights, weights) / static_cast<util::f64_t>(scalars));
        util::f64_t const step = fu::min_of(largest / (8.0 * dot(gradient, gradient)), weight_rms / 4.0);
        for (std::size_t index = 0; index != scalars; ++index)
            plus[index] = weights[index] + step * gradient[index] / largest,
            minus[index] = weights[index] - step * gradient[index] / largest;
        bool uploaded = upload_narrowed_(plus, staging, matrix.weights.data());
        auto [loss_plus, evaluate_loss_plus_status] = token_head_loss_(replica, direction, projection_staging,
                                                                       device.stream);
        uploaded = uploaded && upload_narrowed_(minus, staging, matrix.weights.data());
        auto [loss_minus, evaluate_loss_minus_status] = token_head_loss_(replica, direction, projection_staging,
                                                                         device.stream);
        uploaded = uploaded && upload_narrowed_(weights, staging, matrix.weights.data());
        if (!expect_true(uploaded && evaluate_loss_plus_status && evaluate_loss_minus_status,
                         "perturbed forward of {} failed", name))
            return;
        util::f64_t const predicted = dot(gradient, plus) - dot(gradient, minus);
        util::f64_t const measured = loss_plus - loss_minus;
        util::f64_t const error = std::fabs(measured - predicted) / fu::max_of(std::fabs(predicted), 1.0e-6);
        expect_true(error <= finite_difference_tolerance_k,
                    "{}: loss moved {:.6g}, the gradient predicts {:.6g}, off by {:.3g} relative (loss {:.4f})", name,
                    measured, predicted, error, base_loss);
    }
}

/** Every text row with bytes has at least one masked position, over sixteen 64-byte rows
 *  and eight epochs. */
static void test_masked_text_rows(device_t& device) {
    [[maybe_unused]] util::status_t const scope = (device.stream).bind();
    trainer_config_t config = test_geometry_(false);
    config.batch = multimodal_batch_shape_t {16u, 16u, 64u};
    config.seed = test_run.settings.seed;
    auto [arena, make_arena_status] = util::gpu_arena_t::make(0, case_arena_bytes_k);
    if (!expect_true(make_arena_status, "arena: {}", make_arena_status.description_)) return;
    auto [replica, make_replica_status] = make_replica_(arena, device, config, util::data_epoch_t {0});
    if (!expect_true(make_replica_status, "replica: {}", make_replica_status.description_)) return;
    st::basic_vector<util::u32_t> weights;
    for (util::u32_t epoch = 0; epoch != 8u; ++epoch) {
        if (util::expected_t drawn = epoch == 0 ? util::expected_t {}
                                                : redraw_(replica, config, util::data_epoch_t {epoch}, device.stream);
            !expect_true(drawn, "epoch {}: {}", epoch, drawn.description_))
            return;
        cuStreamSynchronize(util::native_stream(device.stream));
        std::size_t const tokens = replica.token_loss_weights.extent(0);
        if (!expect_true(!st::failed(weights.resize(tokens)), "weights staging failed")) return;
        cuMemcpyDtoH(weights.data(), util::device_address(replica.token_loss_weights.data()),
                     tokens * sizeof(util::u32_t));
        host_segments_t const segments = read_segments_(replica.batch);
        util::u32_t unmasked = 0;
        for (std::size_t segment = 0; segment != segments.lengths.size(); ++segment) {
            if (segments.modality[segment] != modality_t::text_k || segments.lengths[segment] == 0) continue;
            util::u32_t masked = 0;
            for (util::u32_t column = 0; column != segments.lengths[segment]; ++column)
                masked += weights[segments.offsets[segment] + column];
            unmasked += masked == 0 ? 1u : 0u;
        }
        expect_true(unmasked == 0, "epoch {}: {} of {} text rows have no masked byte", epoch, unmasked,
                    segments.lengths.size());
    }
}

/** Each media head counts only masked cells of its own modality, and some of them,
 *  across four epochs. */
static void test_masked_media_rows(device_t& device) {
    [[maybe_unused]] util::status_t const scope = (device.stream).bind();
    trainer_config_t config = test_geometry_(true);
    config.seed = test_run.settings.seed;
    auto [arena, make_arena_status] = util::gpu_arena_t::make(0, case_arena_bytes_k);
    if (!expect_true(make_arena_status, "arena: {}", make_arena_status.description_)) return;
    auto [replica, make_replica_status] = make_replica_(arena, device, config, util::data_epoch_t {0});
    if (!expect_true(make_replica_status, "replica: {}", make_replica_status.description_)) return;
    for (util::u32_t epoch = 0; epoch != 4u; ++epoch) {
        util::expected_t drawn = epoch == 0 ? util::expected_t {}
                                            : redraw_(replica, config, util::data_epoch_t {epoch}, device.stream);
        auto [loss, forward_status] = drawn ? forward_loss_(replica, config, device.stream)
                                            : util::expected<util::f64_t>::fail(std::move(drawn));
        if (!expect_true(forward_status, "epoch {}: {}", epoch, forward_status.description_)) return;
        host_segments_t const segments = read_segments_(replica.batch);
        util::f64_t cells[4] = {0, 0, 0, 0};
        for (std::size_t segment = 0; segment != segments.lengths.size(); ++segment)
            cells[static_cast<util::u32_t>(segments.modality[segment])] += segments.lengths[segment];
        struct head_t {
            char const* name;
            modality_t modality;
            util::f32r1_tensor_arena_t const& summands;
        };
        head_t const heads[] = {{"image", modality_t::image_k, replica.image_summands},
                                {"video", modality_t::video_k, replica.video_summands},
                                {"audio", modality_t::audio_k, replica.audio_summands}};
        for (head_t const& head : heads) {
            util::f64_t const scored = read_pair_(head.summands).count;
            util::f64_t const live = cells[static_cast<util::u32_t>(head.modality)];
            expect_true(scored > 0 && scored <= live, "epoch {}: the {} head scored {} cells of {} {} cells", epoch,
                        head.name, scored, live, head.name);
        }
    }
}

/** A freshly seeded six-block tower predicts uniform random bytes at about the uniform
 *  cross-entropy: logits of unit variance add half a nat over @c ln(258) . */
static void test_chance_cross_entropy(device_t& device) {
    [[maybe_unused]] util::status_t const scope = (device.stream).bind();
    trainer_config_t config = test_geometry_(false);
    config.batch = multimodal_batch_shape_t {8u, 8u, 64u};
    config.depth = 6;
    config.hidden = 384;
    config.seed = test_run.settings.seed;
    auto [arena, make_arena_status] = util::gpu_arena_t::make(0, case_arena_bytes_k);
    if (!expect_true(make_arena_status, "arena: {}", make_arena_status.description_)) return;
    auto [replica, make_replica_status] = make_replica_(arena, device, config, util::data_epoch_t {0});
    if (!expect_true(make_replica_status, "replica: {}", make_replica_status.description_)) return;
    auto [loss, forward_status] = forward_loss_(replica, config, device.stream);
    if (!expect_true(forward_status, "forward: {}", forward_status.description_)) return;
    util::f64_t const uniform = std::log(static_cast<util::f64_t>(byte_vocabulary_k));
    expect_true(loss > uniform - 0.25 && loss < uniform + 1.0,
                "the seeded tower starts at {:.4f} nats per masked byte, the uniform prediction scores {:.4f}", loss,
                uniform);
}

/** One matrix and its gradient on the device beside their widened host copies, for
 *  the Adafactor cases. */
struct optimized_matrix_t {
    util::bf16r2_tensor_arena_t weights {}, gradient {};
    st::basic_vector<util::f64_t> host_weights {}, host_gradient {};
};

[[nodiscard]] static util::expected_t allocate_matrix_(optimized_matrix_t& matrix, util::gpu_arena_t& arena,
                                                       std::size_t rows, std::size_t columns) noexcept {
    util::gpu_arena_alloc<util::bf16_t> bf16_alloc {arena};
    matrix.weights = make_uninitialized<util::bf16r2_tensor_arena_t>({rows, columns}, bf16_alloc);
    matrix.gradient = make_uninitialized<util::bf16r2_tensor_arena_t>({rows, columns}, bf16_alloc);
    if (!matrix.weights || !matrix.gradient || st::failed(matrix.host_weights.resize(rows * columns)) ||
        st::failed(matrix.host_gradient.resize(rows * columns)))
        return util::expected_t::fail(util::status_code_t::out_of_memory_k, "matrix [{},{}] allocation failed", rows,
                                      columns);
    return {};
}

/** A uniform draw in @b [-1,1) keyed on the suite seed, the salt and the index. */
[[nodiscard]] static util::f32_t keyed_uniform_(util::u64_t seed, util::u32_t salt, std::size_t index) noexcept {
    util::u64_t const mixed = util::splitmix64_hash_t {}(seed, salt, index);
    return static_cast<util::f32_t>(mixed >> 40) * 0x1.0p-23f - 1.0f;
}

/** The distance between two BFloat16 neighbours around @p value . */
[[nodiscard]] static util::f64_t bf16_spacing_(util::f64_t value) noexcept {
    int exponent = 0;
    std::frexp(value, &exponent);
    return std::ldexp(1.0, exponent - 8);
}

/** Two Adafactor steps on a @b [24,40] matrix and a @b [1,40] vector against the paper's update
 *  replayed in Float64 on the host: factored second moments decaying at @c 1−t^−0.8 , the RMS clip,
 *  the first moment of the clipped update under @p beta1 , and the step. Each step restarts the
 *  host replay from the device's rounded weights, so the comparison allows one BFloat16 spacing. A
 *  kept first moment must also survive a save and a load into a fresh optimizer unchanged. */
static void test_adafactor_reference(device_t& device, util::f32_t beta1) {
    [[maybe_unused]] util::status_t const scope = (device.stream).bind();
    auto [arena, make_arena_status] = util::gpu_arena_t::make(0, 64ull << 20);
    if (!expect_true(make_arena_status, "arena: {}", make_arena_status.description_)) return;
    std::size_t const shapes[2][2] = {{24, 40}, {1, 40}};
    optimized_matrix_t matrices[2];
    st::basic_vector<util::differentiable_weights_t> table;
    st::basic_vector<nk_bf16_t> staging;
    for (util::u32_t matrix = 0; matrix != 2u; ++matrix) {
        if (util::expected_t allocated = allocate_matrix_(matrices[matrix], arena, shapes[matrix][0],
                                                          shapes[matrix][1]);
            !expect_true(allocated, "{}", allocated.description_))
            return;
        for (std::size_t index = 0; index != matrices[matrix].host_weights.size(); ++index)
            matrices[matrix].host_weights[index] = keyed_uniform_(test_run.settings.seed, matrix, index);
        if (!expect_true(upload_narrowed_(matrices[matrix].host_weights, staging, matrices[matrix].weights.data()),
                         "weights upload failed"))
            return;
        util::differentiable_weights_t entry {{}, matrices[matrix].weights.span(), matrices[matrix].gradient.view()};
        if (!expect_true(!sz::failed(entry.name.assign(matrix == 0 ? "matrix" : "vector")) &&
                             !st::failed(table.push_back(std::move(entry))),
                         "weight table growth failed"))
            return;
    }
    adafactor_config_t const config {.beta1 = beta1};
    auto [optimizer, make_optimizer_status] = adafactor_t::make(arena, table, config, device.numerics, device.stream);
    if (!expect_true(make_optimizer_status, "optimizer: {}", make_optimizer_status.description_)) return;

    util::f32_t const learning_rate = 1.0e-2f;
    st::basic_vector<util::f64_t> row_moments[2], column_moments[2], masters[2], first_moments[2], device_weights;
    for (util::u32_t matrix = 0; matrix != 2u; ++matrix) {
        if (!expect_true(!st::failed(row_moments[matrix].resize(shapes[matrix][0])) &&
                             !st::failed(column_moments[matrix].resize(shapes[matrix][1])) &&
                             !st::failed(masters[matrix].resize(matrices[matrix].host_weights.size())) &&
                             !st::failed(first_moments[matrix].resize(matrices[matrix].host_weights.size())),
                         "host moments allocation failed"))
            return;
        for (std::size_t index = 0; index != masters[matrix].size(); ++index)
            masters[matrix][index] = matrices[matrix].host_weights[index];
    }
    for (util::u32_t step = 1; step != 3u; ++step) {
        for (util::u32_t matrix = 0; matrix != 2u; ++matrix) {
            optimized_matrix_t& state = matrices[matrix];
            for (std::size_t index = 0; index != state.host_gradient.size(); ++index)
                state.host_gradient[index] = keyed_uniform_(test_run.settings.seed, 16u * step + matrix + 2u, index);
            if (!expect_true(upload_narrowed_(state.host_gradient, staging, state.gradient.data()),
                             "gradient upload failed"))
                return;
        }
        if (util::expected_t stepped = optimizer.step(learning_rate, device.stream);
            !expect_true(stepped, "step {}: {}", step, stepped.description_))
            return;
        cuStreamSynchronize(util::native_stream(device.stream));
        util::f64_t const decay = fu::min_of<util::f64_t>(
            config.beta2, 1.0 - std::pow(static_cast<util::f64_t>(step), -config.decay_exponent));
        for (util::u32_t matrix = 0; matrix != 2u; ++matrix) {
            optimized_matrix_t& state = matrices[matrix];
            std::size_t const rows = shapes[matrix][0], columns = shapes[matrix][1];
            f64r2_view_t const gradient {state.host_gradient.data(), rows, columns};
            for (std::size_t row = 0; row != rows; ++row)
                row_moments[matrix][row] = decay * row_moments[matrix][row] +
                                           (1.0 - decay) *
                                               (dot(gradient.row(row), gradient.row(row)) / columns + config.epsilon1);
            for (std::size_t column = 0; column != columns; ++column) {
                util::f64_t energy = 0;
                for (std::size_t row = 0; row != rows; ++row) energy += gradient[row, column] * gradient[row, column];
                column_moments[matrix][column] = decay * column_moments[matrix][column] +
                                                 (1.0 - decay) * (energy / rows + config.epsilon1);
            }
            util::f64_t row_mean = 0, update_energy = 0;
            for (std::size_t row = 0; row != rows; ++row) row_mean += row_moments[matrix][row] / rows;
            auto update_at = [&](std::size_t row, std::size_t column) noexcept {
                return gradient[row, column] /
                       std::sqrt(row_moments[matrix][row] / row_mean * column_moments[matrix][column]);
            };
            for (std::size_t row = 0; row != rows; ++row)
                for (std::size_t column = 0; column != columns; ++column)
                    update_energy += update_at(row, column) * update_at(row, column);
            util::f64_t const clip = 1.0 / fu::max_of(1.0, std::sqrt(update_energy / (rows * columns)) /
                                                               config.clip_threshold);
            if (!expect_true(download_widened_({state.weights.data(), rows * columns}, staging, device_weights),
                             "weights readback failed"))
                return;
            f64r2_view_t const got {device_weights.data(), rows, columns};
            f64r2_view_t const started {state.host_weights.data(), rows, columns};
            f64r2_span_t const master {masters[matrix].data(), rows, columns};
            f64r2_span_t const first_moment {first_moments[matrix].data(), rows, columns};
            util::u32_t off_grid = 0;
            util::f64_t worst = 0;
            for (std::size_t row = 0; row != rows; ++row)
                for (std::size_t column = 0; column != columns; ++column) {
                    util::f64_t const start = rows == 1 ? master[row, column] : started[row, column];
                    util::f64_t const clipped = clip * update_at(row, column);
                    first_moment[row, column] = beta1 * first_moment[row, column] + (1.0 - beta1) * clipped;
                    util::f64_t const corrected = first_moment[row, column] /
                                                  (1.0 - std::pow(util::f64_t(beta1), util::f64_t(step)));
                    util::f64_t const wanted = start - learning_rate * (beta1 != 0.0f ? corrected : clipped);
                    util::f64_t const distance = std::fabs(got[row, column] - wanted) / bf16_spacing_(wanted);
                    worst = fu::max_of(worst, distance);
                    off_grid += distance > 1.0 ? 1u : 0u;
                    master[row, column] = wanted;
                }
            expect_true(off_grid == 0,
                        "step {}: {} of {} {} scalars sit beyond one BFloat16 spacing of the host replay, " //
                        "the worst {:.2f} spacings",
                        step, off_grid, rows * columns, rows == 1 ? "vector" : "matrix", worst);
            std::memcpy(state.host_weights.data(), device_weights.data(), rows * columns * sizeof(util::f64_t));
        }
    }
    if (beta1 == 0.0f) return;

    // The first moment round-trips: saved, loaded into a fresh optimizer, and saved again bitwise.
    scratch_directory_t const scratch;
    sz::string_t path;
    if (!expect_true(util::format_into(path, "{}/first-moment.safetensors", scratch.path.c_str()),
                     "checkpoint path formatting failed"))
        return;
    util::checkpoint_to_save_t saved;
    if (util::expected_t written = optimizer.save(saved, device.stream);
        !expect_true(written, "save: {}", written.description_))
        return;
    if (util::expected_t written = saved.save(path.view());
        !expect_true(written, "save to {}: {}", path.view(), written.description_))
        return;
    auto [checkpoint, make_checkpoint_status] = util::checkpoint_to_load_t::make(path.view());
    if (!expect_true(make_checkpoint_status, "load: {}", make_checkpoint_status.description_)) return;
    auto [optimizer_resumed, make_optimizer_resumed_status] = adafactor_t::make(arena, table, config, device.numerics,
                                                                                device.stream);
    if (!expect_true(make_optimizer_resumed_status, "resumed optimizer: {}",
                     make_optimizer_resumed_status.description_))
        return;
    if (util::expected_t restored = optimizer_resumed.load(checkpoint, device.stream);
        !expect_true(restored, "restore: {}", restored.description_))
        return;
    util::checkpoint_to_save_t resaved;
    if (util::expected_t written = optimizer_resumed.save(resaved, device.stream);
        !expect_true(written, "resave: {}", written.description_))
        return;
    std::remove(path.c_str());
    if (util::expected_t written = resaved.save(path.view());
        !expect_true(written, "resave to {}: {}", path.view(), written.description_))
        return;
    auto [checkpoint_reloaded, make_checkpoint_reloaded_status] = util::checkpoint_to_load_t::make(path.view());
    if (!expect_true(make_checkpoint_reloaded_status, "reload: {}", make_checkpoint_reloaded_status.description_))
        return;
    for (char const* name : {"optimizer.matrix.first_moment", "optimizer.vector.first_moment"}) {
        nk::tensor_view<util::f32_t, 2> const before = checkpoint.float32(name),
                                              after = checkpoint_reloaded.float32(name);
        bool const same = !before.empty() && before.extent(1) == after.extent(1) &&
                          std::memcmp(before.data(), after.data(), before.extent(1) * sizeof(util::f32_t)) == 0;
        expect_true(same, "{} survives a save and a load", name);
    }
}

/** A @b [64,256] matrix of ones under a unit gradient takes one step of @c 2^−10 , a quarter of
 *  the BFloat16 spacing below one: the weights must move by that step on average instead of
 *  rounding back to one. */
static void test_adafactor_small_steps(device_t& device) {
    [[maybe_unused]] util::status_t const scope = (device.stream).bind();
    auto [arena, make_arena_status] = util::gpu_arena_t::make(0, 64ull << 20);
    if (!expect_true(make_arena_status, "arena: {}", make_arena_status.description_)) return;
    std::size_t const rows = 64, columns = 256;
    optimized_matrix_t matrix;
    st::basic_vector<nk_bf16_t> staging;
    if (util::expected_t allocated = allocate_matrix_(matrix, arena, rows, columns);
        !expect_true(allocated, "{}", allocated.description_))
        return;
    for (std::size_t index = 0; index != rows * columns; ++index)
        matrix.host_weights[index] = 1.0, matrix.host_gradient[index] = 1.0;
    if (!expect_true(upload_narrowed_(matrix.host_weights, staging, matrix.weights.data()) &&
                         upload_narrowed_(matrix.host_gradient, staging, matrix.gradient.data()),
                     "upload failed"))
        return;
    st::basic_vector<util::differentiable_weights_t> table;
    util::differentiable_weights_t entry {{}, matrix.weights.span(), matrix.gradient.view()};
    if (!expect_true(!sz::failed(entry.name.assign("ones")) && !st::failed(table.push_back(std::move(entry))),
                     "weight table growth failed"))
        return;
    auto [optimizer, make_optimizer_status] = adafactor_t::make(arena, table, adafactor_config_t {}, device.numerics,
                                                                device.stream);
    if (!expect_true(make_optimizer_status, "optimizer: {}", make_optimizer_status.description_)) return;
    util::f64_t const learning_rate = 0x1.0p-10;
    if (util::expected_t stepped = optimizer.step(static_cast<util::f32_t>(learning_rate), device.stream);
        !expect_true(stepped, "step: {}", stepped.description_))
        return;
    cuStreamSynchronize(util::native_stream(device.stream));
    if (!expect_true(download_widened_({matrix.weights.data(), rows * columns}, staging, matrix.host_weights),
                     "weights readback failed"))
        return;
    auto const [moments, reduce_status] = nk::reduce_moments<nk::f64_t>(matrix.host_weights);
    if (!expect_true(nk::succeeded(reduce_status), "weights moments: {}", nk::status_name(reduce_status))) return;
    util::f64_t const mean_step = 1.0 - moments.sum / static_cast<util::f64_t>(rows * columns);
    // Each scalar moves by 2^-8 with probability 1/4, so the mean of 16384 has a
    // 1.3e-5 standard error.
    expect_true(std::fabs(mean_step - learning_rate) < 0.1 * learning_rate,
                "the mean step is {:.4g}, the learning rate {:.4g}", mean_step, learning_rate);
}

/** The Float64 oracle of one choice: the soft-target cross-entropy plus @p brier_weight times the
 *  Brier term of the softmax of @p logits against @p targets , its logit gradient written into
 *  @p gradient . */
[[nodiscard]] static util::f64_t choice_loss_(std::span<util::f64_t const> logits, std::span<util::f64_t const> targets,
                                              util::f64_t brier_weight, std::span<util::f64_t> gradient) noexcept {
    util::f64_t largest = logits[0], total = 0, cross_entropy = 0, brier = 0, correlation = 0;
    for (util::f64_t const logit : logits) largest = fu::max_of(largest, logit);
    for (util::f64_t const logit : logits) total += std::exp(logit - largest);
    for (std::size_t choice = 0; choice != logits.size(); ++choice) {
        util::f64_t const probability = std::exp(logits[choice] - largest) / total,
                          miss = probability - targets[choice];
        cross_entropy -= targets[choice] * std::log(probability);
        brier += miss * miss, correlation += miss * probability;
    }
    for (std::size_t choice = 0; choice != logits.size(); ++choice) {
        util::f64_t const probability = std::exp(logits[choice] - largest) / total,
                          miss = probability - targets[choice];
        gradient[choice] = miss + brier_weight * 2.0 * probability * (miss - correlation);
    }
    return cross_entropy + brier_weight * brier;
}

/** The choices term over a seeded decoder against its Float64 oracle: a segment of targets
 *  @c {1,0} , one of @c {0.5,0.5} , one making no choice and a PAD. The loss counts the two
 *  choices, the force lands on @c d(normalized) at their answer rows alone, added onto what was
 *  there, the decoder's gradient is added onto too, and a central difference of the loss along
 *  that gradient agrees with it. */
static void test_choices_cross_entropy(device_t& device) {
    [[maybe_unused]] util::status_t const scope = (device.stream).bind();
    util::u32_t const rows = 4, columns = 64, hidden = 256, width = 4, tokens = rows * columns;
    std::size_t const decoder_scalars = std::size_t {byte_vocabulary_k} * hidden;
    auto [arena, make_arena_status] = util::gpu_arena_t::make(0, case_arena_bytes_k);
    if (!expect_true(make_arena_status, "arena: {}", make_arena_status.description_)) return;
    masked_tokens_config_t head_config;
    head_config.hidden = hidden, head_config.max_tokens = tokens, head_config.max_segments = rows;
    auto [head, make_head_status] = masked_tokens_t::make(arena, head_config, device.numerics);
    if (!expect_true(make_head_status, "head: {}", make_head_status.description_)) return;
    multimodal_batch_shape_t shape {rows, rows, columns};
    shape.max_choices_per_segment = width;
    auto [batch, make_batch_status] = multimodal_batch_on_gpu_t::make(util::gpu_arena_alloc_t {arena}, shape);
    if (!expect_true(make_batch_status, "batch: {}", make_batch_status.description_)) return;
    util::gpu_arena_alloc<util::bf16_t> bf16_alloc {arena};
    util::bf16r2_tensor_arena_t normalized = make_uninitialized<util::bf16r2_tensor_arena_t>({tokens, hidden},
                                                                                             bf16_alloc);
    util::bf16r2_tensor_arena_t normalized_grad = make_uninitialized<util::bf16r2_tensor_arena_t>({tokens, hidden},
                                                                                                  bf16_alloc);
    util::f32r1_tensor_arena_t summands = make_uninitialized<util::f32r1_tensor_arena_t>(
        {2}, util::gpu_arena_alloc<util::f32_t> {arena});
    if (!expect_true(normalized && normalized_grad && summands, "choice scratch allocation failed")) return;

    // Segment s spans row s, the PAD at the end none; an answer sits at its segment's last token.
    util::u32_t const offsets[rows + 1] = {0, 64, 128, 192, 192}, lengths[rows] = {64, 40, 64, 0};
    util::u32_t const answers[2] = {63, 103};
    util::token_id_t const pad = byte_pad_k;
    util::token_id_t const choices[rows * width] = {'A', 'B', pad, pad, 's', 'r', pad, pad,
                                                    pad, pad, pad, pad, pad, pad, pad, pad};
    util::f32_t const targets[rows * width] = {1.0f, 0.0f, 0, 0, 0.5f, 0.5f, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
    cuMemcpyHtoD(util::device_address(batch.segment_offsets.data()), offsets, sizeof(offsets));
    cuMemcpyHtoD(util::device_address(batch.segment_lengths.data()), lengths, sizeof(lengths));
    cuMemcpyHtoD(util::device_address(batch.segment_choices.data()), choices, sizeof(choices));
    cuMemcpyHtoD(util::device_address(batch.segment_choice_targets.data()), targets, sizeof(targets));

    st::basic_vector<util::differentiable_weights_t> table;
    if (util::expected_t listed = head.fill("masked_tokens", table); !expect_true(listed, "{}", listed.description_))
        return;
    if (util::expected_t seeded = seed_weights(table, device.stream); !expect_true(seeded, "{}", seeded.description_))
        return;
    st::basic_vector<util::f64_t> states, prior_grad, prior_weight_grad, decoder;
    st::basic_vector<nk_bf16_t> staging;
    if (!expect_true(!st::failed(states.resize(std::size_t {tokens} * hidden)) &&
                         !st::failed(prior_grad.resize(std::size_t {tokens} * hidden)) &&
                         !st::failed(prior_weight_grad.resize(decoder_scalars)),
                     "host staging failed"))
        return;
    for (std::size_t index = 0; index != states.size(); ++index)
        states[index] = keyed_uniform_(test_run.settings.seed, 72u, index) * 1.7320508f,
        prior_grad[index] = keyed_uniform_(test_run.settings.seed, 73u, index) * 0.01f;
    for (std::size_t index = 0; index != decoder_scalars; ++index)
        prior_weight_grad[index] = keyed_uniform_(test_run.settings.seed, 74u, index) * 0.01f;
    if (!expect_true(upload_narrowed_(states, staging, normalized.data()) &&
                         upload_narrowed_(prior_grad, staging, normalized_grad.data()) &&
                         upload_narrowed_(prior_weight_grad, staging, head.decoder_gradients().data()) &&
                         download_widened_({table[0].weights.data(), decoder_scalars}, staging, decoder),
                     "state staging failed"))
        return;

    choices_config_t const config {2.0f, 0.5f};
    auto const choose = [&]() noexcept -> expected<summand_pair_t> {
        util::expected_t chosen = util::zero_async(summands.span(), device.stream);
        if (chosen) chosen = head.pack_weights(device.stream);
        if (chosen)
            chosen = head.choose(normalized.view(), batch.view(), config, summands.span(), normalized_grad.span(),
                                 device.stream);
        if (!chosen) return util::expected<summand_pair_t>::fail(std::move(chosen));
        cuStreamSynchronize(util::native_stream(device.stream));
        return util::expected<summand_pair_t> {read_pair_(summands)};
    };
    auto [summand_pair, choose_summand_pair_status] = choose();
    if (!expect_true(choose_summand_pair_status, "choose: {}", choose_summand_pair_status.description_)) return;

    // The oracle over the two choices, each force scaled by λ over the live choice count.
    std::size_t const offered = 2;
    util::f64_t const scale = config.weight / 2.0;
    st::basic_vector<util::f64_t> force, weight_force;
    if (!expect_true(!st::failed(force.resize(states.size())) && !st::failed(weight_force.resize(decoder_scalars)),
                     "oracle staging failed"))
        return;
    nk::tensor_view<util::token_id_t, 2> const choice_rows {choices, rows, width};
    util::f32r2_view_t const target_rows {targets, rows, width};
    f64r2_view_t const decoder_rows {decoder.data(), byte_vocabulary_k, hidden};
    f64r2_view_t const state_rows {states.data(), tokens, hidden};
    f64r2_span_t const force_rows {force.data(), tokens, hidden};
    f64r2_span_t const weight_force_rows {weight_force.data(), byte_vocabulary_k, hidden};
    util::f64_t oracle = 0;
    for (std::size_t live = 0; live != 2; ++live) {
        util::f64_t logits[offered] = {}, wanted[offered] = {}, gradient[offered] = {};
        std::size_t const row = answers[live];
        for (std::size_t choice = 0; choice != offered; ++choice) {
            std::size_t const byte = choice_rows[live, choice];
            for (std::size_t channel = 0; channel != hidden; ++channel)
                logits[choice] += decoder_rows[byte, channel] * state_rows[row, channel];
            wanted[choice] = target_rows[live, choice];
        }
        oracle += choice_loss_(logits, wanted, config.brier_weight, gradient);
        for (std::size_t choice = 0; choice != offered; ++choice) {
            std::size_t const byte = choice_rows[live, choice];
            for (std::size_t channel = 0; channel != hidden; ++channel) {
                force_rows[row, channel] += scale * gradient[choice] * decoder_rows[byte, channel];
                weight_force_rows[byte, channel] += scale * gradient[choice] * state_rows[row, channel];
            }
        }
    }
    util::f64_t const oracle_loss = config.weight * oracle / 2.0, measured_loss = summand_pair.sum / summand_pair.count;
    expect_true(summand_pair.count == 2.0, "two segments make a choice, {} counted", summand_pair.count);
    expect_true(std::fabs(measured_loss - oracle_loss) <= 1.0e-3 * fu::max_of(1.0, oracle_loss),
                "the choices loss is {:.6f}, the Float64 oracle {:.6f}", measured_loss, oracle_loss);

    st::basic_vector<util::f64_t> landed_grad, landed_weight_grad;
    if (!expect_true(
            download_widened_({normalized_grad.data(), states.size()}, staging, landed_grad) &&
                download_widened_({head.decoder_gradients().data(), decoder_scalars}, staging, landed_weight_grad),
            "gradient readback failed"))
        return;
    util::f64_t untouched = 0, gap = 0, magnitude = 0;
    for (std::size_t index = 0; index != states.size(); ++index) {
        std::size_t const row = index / hidden;
        util::f64_t const added = landed_grad[index] - prior_grad[index];
        if (row != answers[0] && row != answers[1]) untouched = fu::max_of(untouched, std::fabs(added));
        else gap += (added - force[index]) * (added - force[index]), magnitude += force[index] * force[index];
    }
    expect_true(untouched == 0.0, "rows without an answer keep their force, off by {:.3g}", untouched);
    expect_true(std::sqrt(gap) <= 2.0e-2 * std::sqrt(magnitude),
                "the answer rows' added force matches the oracle: relative {:.3g}", std::sqrt(gap / magnitude));
    gap = 0, magnitude = 0;
    for (std::size_t index = 0; index != decoder_scalars; ++index) {
        util::f64_t const added = landed_weight_grad[index] - prior_weight_grad[index];
        gap += (added - weight_force[index]) * (added - weight_force[index]);
        magnitude += weight_force[index] * weight_force[index];
    }
    expect_true(std::sqrt(gap) <= 2.0e-2 * std::sqrt(magnitude),
                "the decoder's added gradient matches the oracle: relative {:.3g}", std::sqrt(gap / magnitude));

    // A central difference along the oracle's decoder gradient, the step doubled until the
    // BFloat16-rounded move predicts an eighth of the loss.
    st::basic_vector<util::f64_t> plus, minus;
    if (!expect_true(!st::failed(plus.resize(decoder_scalars)) && !st::failed(minus.resize(decoder_scalars)),
                     "perturbation staging failed"))
        return;
    if (!expect_true(!st::failed(staging.resize(st::assume_uninitialized, decoder_scalars)),
                     "perturbation staging failed"))
        return;
    util::f64_t largest = 0, predicted = 0;
    for (util::f64_t const value : weight_force) largest = fu::max_of(largest, std::fabs(value));
    util::f64_t const weight_rms = std::sqrt(dot(decoder, decoder) / static_cast<util::f64_t>(decoder_scalars));
    util::f64_t const first_step = largest / (8.0 * dot(weight_force, weight_force));
    for (util::f64_t step = first_step; step == first_step || (predicted < 0.125 * oracle_loss && step <= weight_rms);
         step *= 2) {
        for (std::size_t index = 0; index != decoder_scalars; ++index)
            plus[index] = decoder[index] + step * weight_force[index] / largest,
            minus[index] = decoder[index] - step * weight_force[index] / largest;
        if (!expect_true(narrow_to_bf16(plus, staging) && narrow_to_bf16(minus, staging), "host rounding failed"))
            return;
        predicted = dot(weight_force, plus) - dot(weight_force, minus);
    }
    bool uploaded = upload_narrowed_(plus, staging, table[0].weights.data());
    auto [loss_plus, choose_loss_plus_status] = choose();
    uploaded = uploaded && upload_narrowed_(minus, staging, table[0].weights.data());
    auto [loss_minus, choose_loss_minus_status] = choose();
    if (!expect_true(uploaded && choose_loss_plus_status && choose_loss_minus_status, "perturbed choose failed"))
        return;
    util::f64_t const measured = loss_plus.sum / loss_plus.count - loss_minus.sum / loss_minus.count;
    util::f64_t const error = std::fabs(measured - predicted) / fu::max_of(std::fabs(predicted), 1.0e-6);
    expect_true(error <= finite_difference_tolerance_k,
                "the decoder: loss moved {:.6g}, the gradient predicts {:.6g}, off by {:.3g} relative", measured,
                predicted, error);
}

/** Unit RMS normalization of the run @p input into @p output , the forward's own epsilon. */
static void normalize_rms_(auto const& input, auto const& output, util::f64_t epsilon) noexcept {
    std::size_t const count = input.size();
    util::f64_t const inverse = 1.0 / std::sqrt(dot(input, input) / static_cast<util::f64_t>(count) + epsilon);
    for (std::size_t index = 0; index != count; ++index) output[index] = input[index] * inverse;
}

/** The base the block replay's rotary angle grid is built over. */
inline constexpr util::f64_t rope_base_k = 10000.0;

/** The angle of rotary pair @p pair at a token's @c (t,h,w) @p coordinates , as the block's grid
 *  bakes it: the temporal axis takes the first pairs at the whole head's frequency spacing, height
 *  and width the rest at their own. */
[[nodiscard]] static util::f64_t rope_angle_(std::size_t pair, nk::tensor_view<util::i16_t, 1> coordinates) noexcept {
    rope_shape_t const shape {};
    std::size_t const temporal = shape.temporal / 2, height = shape.height / 2;
    if (pair < temporal)
        return coordinates[0] * std::pow(rope_base_k, -2.0 * pair / (shape.temporal + shape.height + shape.width));
    if (pair < temporal + height)
        return coordinates[1] * std::pow(rope_base_k, -2.0 * (pair - temporal) / shape.height);
    return coordinates[2] * std::pow(rope_base_k, -2.0 * (pair - temporal - height) / shape.width);
}

/** Rotate every NeoX split-half pair @c (k,k+64) of the head vector @p values by @p sign times its
 *  angle at @p coordinates ; a sign of @c -1 is the inverse the backward applies. */
static void rotate_(auto&& values, nk::tensor_view<util::i16_t, 1> coordinates, util::f64_t sign) noexcept {
    std::size_t const pairs = transformer_head_dim_k / 2;
    for (std::size_t pair = 0; pair != pairs; ++pair) {
        util::f64_t const angle = sign * rope_angle_(pair, coordinates);
        util::f64_t const real = values[pair], imaginary = values[pair + pairs];
        values[pair] = real * std::cos(angle) - imaginary * std::sin(angle);
        values[pair + pairs] = real * std::sin(angle) + imaginary * std::cos(angle);
    }
}

/** Seed @p block 's matrices for unit-variance outputs from draws salted @p salt onward and its
 *  norm scales to one, uploaded through @p staging , leaving the six narrowed host copies in
 *  @p weights . */
[[nodiscard]] static bool seed_block_(transformer_block_bf16_t& block, util::u64_t seed, util::u32_t salt,
                                      st::basic_vector<nk_bf16_t>& staging,
                                      st::basic_vector<util::f64_t>* weights) noexcept {
    std::size_t const hidden = block.config().hidden, ffn = block.config().ffn_expansion * hidden;
    struct seeded_t {
        util::bf16r1_span_t weights;
        std::size_t fan_in;
    };
    seeded_t const seeded[] = {{block.qkv_weight(), hidden},     {block.out_weight(), hidden},
                               {block.gate_up_weight(), hidden}, {block.down_weight(), ffn},
                               {block.pre_norm_scale(), 0},      {block.post_norm_scale(), 0}};
    for (util::u32_t matrix = 0; matrix != 6u; ++matrix) {
        if (st::failed(weights[matrix].resize(seeded[matrix].weights.extent(0)))) return false;
        util::f64_t const deviation = seeded[matrix].fan_in ? 1.0 / std::sqrt(util::f64_t(seeded[matrix].fan_in)) : 0.0;
        for (std::size_t index = 0; index != weights[matrix].size(); ++index)
            weights[matrix][index] = deviation ? keyed_uniform_(seed, salt + matrix, index) * 1.7320508 * deviation
                                               : 1.0;
        if (!upload_narrowed_(weights[matrix], staging, seeded[matrix].weights.data())) return false;
    }
    return true;
}

/** One block over four segments in 64-token windows, every other one 40 tokens long, with two heads
 *  and M-RoPE coordinates on all three axes: its forward and its attention backward against a
 *  Float64 host replay, the output of every token, and the dQ, dK and dV every segment's softmax
 *  owes back through the inverse rotation and the QK-norm, zero past each segment's length. */
static void test_block_host_replay(device_t& device) {
    [[maybe_unused]] util::status_t const scope = (device.stream).bind();
    std::size_t const hidden = 256, sequence = 64, segments = 4, tokens = sequence * segments;
    std::size_t const heads = hidden / transformer_head_dim_k, ffn = 2 * hidden;
    auto [arena, make_arena_status] = util::gpu_arena_t::make(0, case_arena_bytes_k);
    if (!expect_true(make_arena_status, "arena: {}", make_arena_status.description_)) return;
    transformer_block_config_t config;
    config.hidden = hidden;
    config.ffn_expansion = 2;
    config.max_tokens = tokens;
    config.max_segments = segments;
    config.max_segment_tokens = sequence;
    auto [scratch, make_scratch_status] = transformer_scratch_t::make(arena, config, {}, 0, device.numerics);
    if (!expect_true(make_scratch_status, "scratch: {}", make_scratch_status.description_)) return;
    auto [block, make_block_status] = transformer_block_bf16_t::make(arena, scratch, config, device.numerics);
    if (!expect_true(make_block_status, "block: {}", make_block_status.description_)) return;
    auto [angles, make_angles_status] = rope_angles_t::make(arena, sequence, rope_shape_t {},
                                                            static_cast<util::f32_t>(rope_base_k));
    if (!expect_true(make_angles_status, "angles: {}", make_angles_status.description_)) return;

    // Weights of unit-variance outputs, unit norm scales, and unit-variance inputs
    // and output gradients.
    st::basic_vector<util::f64_t> weights[6];
    st::basic_vector<nk_bf16_t> staging;
    if (!expect_true(seed_block_(block, test_run.settings.seed, 32u, staging, weights), "weights upload failed"))
        return;
    st::basic_vector<util::f32_t> input, output_grad, output;
    st::basic_vector<util::f64_t> widened_input;
    st::basic_vector<util::u32_t> offsets, lengths;
    st::basic_vector<util::i16_t> coordinates;
    if (!expect_true(!st::failed(input.resize(tokens * hidden)) && !st::failed(widened_input.resize(tokens * hidden)) &&
                         !st::failed(output_grad.resize(tokens * hidden)) &&
                         !st::failed(output.resize(tokens * hidden)) && !st::failed(offsets.resize(segments + 1)) &&
                         !st::failed(lengths.resize(segments)) && !st::failed(coordinates.resize(tokens * 3)),
                     "host staging failed"))
        return;
    for (std::size_t index = 0; index != input.size(); ++index)
        input[index] = keyed_uniform_(test_run.settings.seed, 40u, index) * 1.7320508f,
        output_grad[index] = keyed_uniform_(test_run.settings.seed, 41u, index);
    for (std::size_t segment = 0; segment != segments; ++segment)
        offsets[segment] = static_cast<util::u32_t>(segment * sequence),
        lengths[segment] = static_cast<util::u32_t>(segment % 2 ? sequence * 5 / 8 : sequence);
    offsets[segments] = static_cast<util::u32_t>(tokens);
    // Distinct coordinates on every axis, so each rotary band of every head turns.
    nk::tensor_span<util::i16_t, 2> const token_coordinates {coordinates.data(), tokens, 3};
    for (std::size_t token = 0; token != tokens; ++token) {
        std::size_t const position = token % sequence;
        token_coordinates[token, 0] = static_cast<util::i16_t>(position % 7);
        token_coordinates[token, 1] = static_cast<util::i16_t>(position / 7 % 5);
        token_coordinates[token, 2] = static_cast<util::i16_t>(position % 3);
    }
    if (!expect_true(nk::cast<nk::f32_t, nk::f64_t>(input, widened_input) == nk::status_t::success_k,
                     "input widening failed"))
        return;
    util::f32_t* const device_input = arena.allocate_object<util::f32_t>(tokens * hidden);
    util::f32_t* const device_output = arena.allocate_object<util::f32_t>(tokens * hidden);
    util::f32_t* const device_output_grad = arena.allocate_object<util::f32_t>(tokens * hidden);
    util::f32_t* const device_input_grad = arena.allocate_object<util::f32_t>(tokens * hidden);
    util::u32_t* const device_offsets = arena.allocate_object<util::u32_t>(segments + 1);
    util::u32_t* const device_lengths = arena.allocate_object<util::u32_t>(segments);
    util::i16_t* const device_coordinates = arena.allocate_object<util::i16_t>(tokens * 3);
    cuMemcpyHtoD(util::device_address(device_input), input.data(), input.size() * sizeof(util::f32_t));
    cuMemcpyHtoD(util::device_address(device_output_grad), output_grad.data(),
                 output_grad.size() * sizeof(util::f32_t));
    cuMemcpyHtoD(util::device_address(device_offsets), offsets.data(), offsets.size() * sizeof(util::u32_t));
    cuMemcpyHtoD(util::device_address(device_lengths), lengths.data(), lengths.size() * sizeof(util::u32_t));
    cuMemcpyHtoD(util::device_address(device_coordinates), coordinates.data(),
                 coordinates.size() * sizeof(util::i16_t));
    cuCtxSynchronize(); // the pageable copies land before the non-blocking stream reads them
    util::u32r1_view_t const segment_offsets {device_offsets, segments + 1}, segment_lengths {device_lengths, segments};
    util::i16r2_view_t const coordinate_view {device_coordinates, tokens, 3};
    util::expected_t staged = block.pack_weights(device.stream);
    if (staged)
        staged = block.forward(util::f32r2_view_t {device_input, tokens, hidden},
                               util::f32r2_span_t {device_output, tokens, hidden}, segment_offsets, segment_lengths,
                               angles.view(), coordinate_view, device.stream);
    if (staged)
        staged = block.backward(util::f32r2_view_t {device_output_grad, tokens, hidden},
                                util::f32r2_span_t {device_input_grad, tokens, hidden}, segment_offsets,
                                segment_lengths, angles.view(), coordinate_view, device.stream);
    if (!expect_true(staged, "block step: {}", staged.description_)) return;
    cuStreamSynchronize(util::native_stream(device.stream));
    cuMemcpyDtoH(output.data(), util::device_address(device_output), output.size() * sizeof(util::f32_t));
    st::basic_vector<util::f64_t> narrowed_grad, packed_grad;
    if (!expect_true(download_widened_({scratch.narrowed_grad.data(), tokens * hidden}, staging, narrowed_grad) &&
                         download_widened_({scratch.packed_qkv.data(), tokens * 3 * hidden}, staging, packed_grad),
                     "gradients readback failed"))
        return;
    f64r2_view_t const qkv_weights {weights[0].data(), 3 * hidden, hidden};
    f64r2_view_t const out_weights {weights[1].data(), hidden, hidden};
    f64r2_view_t const gate_up_weights {weights[2].data(), 2 * ffn, hidden};
    f64r2_view_t const down_weights {weights[3].data(), hidden, ffn};
    f64r2_view_t const inputs {widened_input.data(), tokens, hidden}, narrowed {narrowed_grad.data(), tokens, hidden};
    util::f32r2_view_t const outputs {output.data(), tokens, hidden};
    f64r4_view_t const packed {packed_grad.data(), {tokens, 3, heads, transformer_head_dim_k}};

    // The attention's output gradient, from the narrowed out-projection gradient the device
    // multiplied by W_out.
    st::basic_vector<util::f64_t> attention_output_grad;
    if (!expect_true(!st::failed(attention_output_grad.resize(tokens * hidden)), "staging failed")) return;
    f64r2_span_t const output_grad_rows {attention_output_grad.data(), tokens, hidden};
    f64r3_view_t const output_grads {attention_output_grad.data(), {tokens, heads, transformer_head_dim_k}};
    for (std::size_t token = 0; token != tokens; ++token)
        for (std::size_t channel = 0; channel != hidden; ++channel) {
            util::f64_t sum = 0;
            for (std::size_t inner = 0; inner != hidden; ++inner)
                sum += narrowed[token, inner] * out_weights[inner, channel];
            output_grad_rows[token, channel] = sum;
        }

    // The host forward: pre-norm, QKV, the QK-norm, attention per segment and head, then
    // the MLP half.
    st::basic_vector<util::f64_t> products, normalized, attended, normalized_row, branch, mlp_input, gate_up, activated;
    if (!expect_true(!st::failed(products.resize(tokens * 3 * hidden)) &&
                         !st::failed(normalized.resize(tokens * 3 * hidden)) &&
                         !st::failed(attended.resize(tokens * hidden)) && !st::failed(normalized_row.resize(hidden)) &&
                         !st::failed(branch.resize(hidden)) && !st::failed(mlp_input.resize(hidden)) &&
                         !st::failed(gate_up.resize(2 * ffn)) && !st::failed(activated.resize(ffn)),
                     "host replay staging failed"))
        return;
    f64r2_span_t const product_rows {products.data(), tokens, 3 * hidden};
    f64r2_span_t const normalized_rows {normalized.data(), tokens, 3 * hidden};
    f64r4_span_t const pre_norm {products.data(), {tokens, 3, heads, transformer_head_dim_k}};
    f64r4_span_t const qkv {normalized.data(), {tokens, 3, heads, transformer_head_dim_k}};
    for (std::size_t token = 0; token != tokens; ++token) {
        normalize_rms_(inputs.row(token), std::span<util::f64_t> {normalized_row}, config.rms_eps);
        for (std::size_t output_index = 0; output_index != 3 * hidden; ++output_index)
            product_rows[token, output_index] = normalized_rows[token, output_index] = dot(
                qkv_weights.row(output_index), normalized_row);
        for (std::size_t role = 0; role != 2; ++role)
            for (std::size_t head = 0; head != heads; ++head) {
                normalize_rms_(pre_norm[token, role, head, nk::slice], qkv[token, role, head, nk::slice],
                               config.rms_eps);
                rotate_(qkv[token, role, head, nk::slice], token_coordinates[token, nk::slice], 1.0);
            }
    }
    util::f64_t const scale = 1.0 / std::sqrt(static_cast<util::f64_t>(transformer_head_dim_k));
    st::basic_vector<util::f64_t> probability_cells, scores_grad_cells;
    if (!expect_true(!st::failed(probability_cells.resize(sequence * sequence)) &&
                         !st::failed(scores_grad_cells.resize(sequence * sequence)),
                     "host replay staging failed"))
        return;
    f64r2_span_t const probabilities {probability_cells.data(), sequence, sequence};
    f64r2_span_t const scores_grad {scores_grad_cells.data(), sequence, sequence};
    f64r3_span_t const attended_heads {attended.data(), {tokens, heads, transformer_head_dim_k}};
    util::f64_t worst_grad[3] = {0, 0, 0}, largest_grad[3] = {0, 0, 0};
    for (std::size_t segment = 0; segment != segments; ++segment)
        for (std::size_t head = 0; head != heads; ++head) {
            std::size_t const first = segment * sequence, length = lengths[segment];
            for (std::size_t query = 0; query != length; ++query) {
                util::f64_t largest = -1.0e300, total = 0;
                for (std::size_t key = 0; key != length; ++key) {
                    probabilities[query, key] =
                        dot(qkv[first + query, 0, head, nk::slice], qkv[first + key, 1, head, nk::slice]) * scale;
                    largest = fu::max_of(largest, probabilities[query, key]);
                }
                for (std::size_t key = 0; key != length; ++key)
                    total += probabilities[query, key] = std::exp(probabilities[query, key] - largest);
                for (std::size_t key = 0; key != length; ++key) probabilities[query, key] /= total;
                for (std::size_t channel = 0; channel != transformer_head_dim_k; ++channel) {
                    util::f64_t value = 0;
                    for (std::size_t key = 0; key != length; ++key)
                        value += probabilities[query, key] * qkv[first + key, 2, head, channel];
                    attended_heads[first + query, head, channel] = value;
                }
            }
            // The backward from the device's own attention-output gradient: dV, dS, then dQ and dK
            // through the QK-norm.
            for (std::size_t query = 0; query != length; ++query) {
                auto const output_grad = output_grads[first + query, head, nk::slice];
                util::f64_t const output_dot = dot(output_grad, attended_heads[first + query, head, nk::slice]);
                for (std::size_t key = 0; key != length; ++key) {
                    util::f64_t const probability_grad = dot(output_grad, qkv[first + key, 2, head, nk::slice]);
                    scores_grad[query, key] = probabilities[query, key] * (probability_grad - output_dot) * scale;
                }
            }
            for (std::size_t role = 0; role != 3; ++role)
                for (std::size_t position = 0; position != sequence; ++position) {
                    util::f64_t normalized_grad[transformer_head_dim_k];
                    for (std::size_t channel = 0; channel != transformer_head_dim_k; ++channel) {
                        util::f64_t sum = 0;
                        for (std::size_t other = 0; other != length; ++other)
                            sum += role == 0 ? scores_grad[position, other] * qkv[first + other, 1, head, channel]
                                   : role == 1
                                       ? scores_grad[other, position] * qkv[first + other, 0, head, channel]
                                       : probabilities[other, position] * output_grads[first + other, head, channel];
                        normalized_grad[channel] = position < length ? sum : 0.0;
                    }
                    if (role != 2) rotate_(normalized_grad, token_coordinates[first + position, nk::slice], -1.0);
                    auto const pre = pre_norm[first + position, role, head, nk::slice];
                    util::f64_t const inverse =
                        role == 2 ? 1.0 : 1.0 / std::sqrt(dot(pre, pre) / transformer_head_dim_k + config.rms_eps);
                    util::f64_t const projection = dot(normalized_grad, pre) * inverse / transformer_head_dim_k;
                    for (std::size_t channel = 0; channel != transformer_head_dim_k; ++channel) {
                        util::f64_t const wanted =
                            role == 2 ? normalized_grad[channel]
                                      : (normalized_grad[channel] - pre[channel] * inverse * projection) * inverse;
                        util::f64_t const got = packed[first + position, role, head, channel];
                        worst_grad[role] = fu::max_of(worst_grad[role], std::fabs(got - wanted));
                        largest_grad[role] = fu::max_of(largest_grad[role], std::fabs(wanted));
                    }
                }
        }
    f64r2_view_t const attended_rows {attended.data(), tokens, hidden};
    util::f64_t worst_output = 0, largest_output = 0;
    for (std::size_t token = 0; token != tokens; ++token) {
        for (std::size_t channel = 0; channel != hidden; ++channel)
            branch[channel] = inputs[token, channel] + dot(out_weights.row(channel), attended_rows.row(token));
        normalize_rms_(branch, std::span<util::f64_t> {mlp_input}, config.rms_eps);
        for (std::size_t output_index = 0; output_index != 2 * ffn; ++output_index)
            gate_up[output_index] = dot(gate_up_weights.row(output_index), mlp_input);
        for (std::size_t feature = 0; feature != ffn; ++feature)
            activated[feature] = gate_up[feature] / (1.0 + std::exp(-gate_up[feature])) * gate_up[ffn + feature];
        for (std::size_t channel = 0; channel != hidden; ++channel) {
            util::f64_t const wanted = branch[channel] + dot(down_weights.row(channel), activated);
            worst_output = fu::max_of(worst_output, std::fabs(wanted - outputs[token, channel]));
            largest_output = fu::max_of(largest_output, std::fabs(wanted));
        }
    }
    // BFloat16 activations hold about three decimal digits, so two percent of the largest value
    // bounds every cell.
    expect_true(worst_output <= 0.02 * largest_output, "forward: off by {:.4g} of {:.4g}", worst_output,
                largest_output);
    char const* const roles[] = {"dQ", "dK", "dV"};
    for (util::u32_t role = 0; role != 3u; ++role)
        expect_true(worst_grad[role] <= 0.02 * largest_grad[role], "{}: off by {:.4g} of {:.4g} over {} segments",
                    roles[role], worst_grad[role], largest_grad[role], segments);
}

/** A block's forward replayed from its own @c input() after another block's forward
 *  overwrote their shared scratch reproduces its backward bitwise, as the tower's
 *  recomputing backward relies on. */
static void test_block_recompute_replays(device_t& device) {
    [[maybe_unused]] util::status_t const scope = (device.stream).bind();
    std::size_t const hidden = 256, sequence = 64, segments = 2, tokens = sequence * segments;
    auto [arena, make_arena_status] = util::gpu_arena_t::make(0, case_arena_bytes_k);
    if (!expect_true(make_arena_status, "arena: {}", make_arena_status.description_)) return;
    transformer_block_config_t config;
    config.hidden = hidden;
    config.ffn_expansion = 2;
    config.max_tokens = tokens;
    config.max_segments = segments;
    config.max_segment_tokens = sequence;
    auto [scratch, make_scratch_status] = transformer_scratch_t::make(arena, config, {}, 0, device.numerics);
    if (!expect_true(make_scratch_status, "scratch: {}", make_scratch_status.description_)) return;
    auto [block, make_block_status] = transformer_block_bf16_t::make(arena, scratch, config, device.numerics);
    if (!expect_true(make_block_status, "block: {}", make_block_status.description_)) return;
    auto [other, make_other_status] = transformer_block_bf16_t::make(arena, scratch, config, device.numerics);
    if (!expect_true(make_other_status, "other block: {}", make_other_status.description_)) return;
    auto [angles, make_angles_status] = rope_angles_t::make(arena, sequence, rope_shape_t {},
                                                            static_cast<util::f32_t>(rope_base_k));
    if (!expect_true(make_angles_status, "angles: {}", make_angles_status.description_)) return;
    st::basic_vector<util::f64_t> weights[6];
    st::basic_vector<nk_bf16_t> staging;
    if (!expect_true(seed_block_(block, test_run.settings.seed, 48u, staging, weights) &&
                         seed_block_(other, test_run.settings.seed, 56u, staging, weights),
                     "weights upload failed"))
        return;
    st::basic_vector<util::f32_t> rows;
    if (!expect_true(!st::failed(rows.resize(3 * tokens * hidden)), "host staging failed")) return;
    for (std::size_t index = 0; index != rows.size(); ++index)
        rows[index] = keyed_uniform_(test_run.settings.seed, 64u, index) * 1.7320508f;
    util::u32_t const offsets[] = {0, sequence, tokens}, lengths[] = {sequence, sequence * 5 / 8};
    util::i16_t coordinates[tokens * 3] = {};
    for (std::size_t token = 0; token != tokens; ++token)
        coordinates[token * 3] = static_cast<util::i16_t>(token % sequence);
    util::f32_t* const device_rows = arena.allocate_object<util::f32_t>(3 * tokens * hidden);
    util::f32_t* const device_output = arena.allocate_object<util::f32_t>(tokens * hidden);
    util::f32_t* const device_input_grad = arena.allocate_object<util::f32_t>(tokens * hidden);
    util::u32_t* const device_offsets = arena.allocate_object<util::u32_t>(segments + 1);
    util::u32_t* const device_lengths = arena.allocate_object<util::u32_t>(segments);
    util::i16_t* const device_coordinates = arena.allocate_object<util::i16_t>(tokens * 3);
    cuMemcpyHtoD(util::device_address(device_rows), rows.data(), rows.size() * sizeof(util::f32_t));
    cuMemcpyHtoD(util::device_address(device_offsets), offsets, sizeof(offsets));
    cuMemcpyHtoD(util::device_address(device_lengths), lengths, sizeof(lengths));
    cuMemcpyHtoD(util::device_address(device_coordinates), coordinates, sizeof(coordinates));
    cuCtxSynchronize();
    util::u32r1_view_t const segment_offsets {device_offsets, segments + 1}, segment_lengths {device_lengths, segments};
    util::i16r2_view_t const coordinate_view {device_coordinates, tokens, 3};
    util::f32r2_view_t const input {device_rows, tokens, hidden},
        other_input {device_rows + tokens * hidden, tokens, hidden};
    util::f32r2_view_t const output_grad {device_rows + 2 * tokens * hidden, tokens, hidden};
    util::f32r2_span_t const output {device_output, tokens, hidden}, input_grad {device_input_grad, tokens, hidden};

    // The first pass is the golden one; the second lets the other block overwrite the shared
    // stashes and replays from the input stash, as `encoder_tower.cu`'s backward does.
    util::bf16r1_view_t const weight_grads[] = {block.qkv_weight_grad(), block.out_weight_grad(),
                                                block.gate_up_weight_grad(), block.down_weight_grad()};
    st::basic_vector<util::f32_t> input_grads[2];
    st::basic_vector<util::f64_t> grads[2][4];
    util::expected_t staged = block.pack_weights(device.stream);
    if (staged) staged = other.pack_weights(device.stream);
    for (util::u32_t pass = 0; staged && pass != 2u; ++pass) {
        staged = block.forward(input, output, segment_offsets, segment_lengths, angles.view(), coordinate_view,
                               device.stream);
        if (staged && pass == 1)
            staged = other.forward(other_input, output, segment_offsets, segment_lengths, angles.view(),
                                   coordinate_view, device.stream);
        if (staged && pass == 1)
            staged = block.forward(block.input(), output, segment_offsets, segment_lengths, angles.view(),
                                   coordinate_view, device.stream);
        if (staged)
            staged = block.backward(output_grad, input_grad, segment_offsets, segment_lengths, angles.view(),
                                    coordinate_view, device.stream);
        if (!staged) break;
        cuStreamSynchronize(util::native_stream(device.stream));
        if (!expect_true(!st::failed(input_grads[pass].resize(tokens * hidden)), "host staging failed")) return;
        cuMemcpyDtoH(input_grads[pass].data(), util::device_address(device_input_grad),
                     tokens * hidden * sizeof(util::f32_t));
        for (util::u32_t matrix = 0; matrix != 4u; ++matrix)
            if (!expect_true(download_widened_({weight_grads[matrix].data(), weight_grads[matrix].extent(0)}, staging,
                                               grads[pass][matrix]),
                             "gradients readback failed"))
                return;
    }
    if (!expect_true(staged, "block step: {}", staged.description_)) return;
    expect_true(std::memcmp(input_grads[0].data(), input_grads[1].data(), tokens * hidden * sizeof(util::f32_t)) == 0,
                "the input gradient survives the replay bitwise");
    char const* const names[] = {"qkv", "out", "gate_up", "down"};
    for (util::u32_t matrix = 0; matrix != 4u; ++matrix)
        expect_true(std::memcmp(grads[0][matrix].data(), grads[1][matrix].data(),
                                grads[0][matrix].size() * sizeof(util::f64_t)) == 0,
                    "the {} weight gradient survives the replay bitwise", names[matrix]);
}

/** Normalize every row of @p pooled into @p units , an all-zero row staying zero, and return the
 *  inverse norms in @p inverse_norms . */
static void normalize_rows_(f64r2_view_t pooled, f64r2_span_t units, std::span<util::f64_t> inverse_norms) noexcept {
    std::size_t const rows = pooled.extent(0), hidden = pooled.extent(1);
    for (std::size_t row = 0; row != rows; ++row) {
        std::span<util::f64_t const> const values {&pooled[row, 0], hidden};
        util::f64_t const energy = dot(values, values);
        inverse_norms[row] = energy > 0.0 ? 1.0 / std::sqrt(energy) : 0.0;
        for (std::size_t channel = 0; channel != hidden; ++channel)
            units[row, channel] = values[channel] * inverse_norms[row];
    }
}

/** The coding rate @c ½ · logdet(I+α · ZᵀZ) of the unit rows @p units in Float64, leaving the
 *  Cholesky factor of @c I+α · ZᵀZ in the lower triangle of @p lower . */
[[nodiscard]] static util::f64_t coding_rate_replay_(f64r2_view_t units, util::f64_t alpha,
                                                     f64r2_span_t lower) noexcept {
    std::size_t const rows = units.extent(0), hidden = units.extent(1);
    for (std::size_t row = 0; row != hidden; ++row)
        for (std::size_t column = 0; column <= row; ++column) {
            util::f64_t gram = 0.0;
            for (std::size_t unit = 0; unit != rows; ++unit) gram += units[unit, row] * units[unit, column];
            lower[row, column] = (row == column ? 1.0 : 0.0) + alpha * gram;
        }
    util::f64_t rate = 0.0;
    for (std::size_t pivot = 0; pivot != hidden; ++pivot) {
        for (std::size_t inner = 0; inner != pivot; ++inner)
            lower[pivot, pivot] -= lower[pivot, inner] * lower[pivot, inner];
        lower[pivot, pivot] = std::sqrt(lower[pivot, pivot]);
        for (std::size_t row = pivot + 1; row != hidden; ++row) {
            for (std::size_t inner = 0; inner != pivot; ++inner)
                lower[row, pivot] -= lower[row, inner] * lower[pivot, inner];
            lower[row, pivot] /= lower[pivot, pivot];
        }
        rate += std::log(lower[pivot, pivot]);
    }
    return rate;
}

/** The pooled-code loss @c −λ_tcr · R+λ_lip · ‖Z · v‖² of the raw rows @p pooled at the fixed
 *  direction @p direction , in Float64. */
[[nodiscard]] static util::f64_t spectrum_loss_replay_(f64r2_view_t pooled, pooled_spectrum_config_t config,
                                                       std::span<util::f64_t const> direction) noexcept {
    std::size_t const rows = pooled.extent(0), hidden = pooled.extent(1);
    st::basic_vector<util::f64_t> units, inverse_norms, lower;
    if (st::failed(units.resize(rows * hidden)) || st::failed(inverse_norms.resize(rows)) ||
        st::failed(lower.resize(hidden * hidden)))
        return std::numeric_limits<util::f64_t>::quiet_NaN();
    f64r2_span_t const unit_rows {units.data(), rows, hidden};
    normalize_rows_(pooled, unit_rows, inverse_norms);
    util::f64_t const alpha = util::f64_t(hidden) / (util::f64_t(config.rows_per_world) * config.epsilon_squared);
    util::f64_t const rate = coding_rate_replay_(unit_rows, alpha, f64r2_span_t {lower.data(), hidden, hidden});
    util::f64_t lipschitz = 0.0;
    for (std::size_t row = 0; row != rows; ++row) {
        util::f64_t const projection = dot(std::span<util::f64_t const> {&unit_rows[row, 0], hidden}, direction);
        lipschitz += projection * projection;
    }
    return -util::f64_t(config.coding_rate_weight) * rate + util::f64_t(config.lipschitz_weight) * lipschitz;
}

/** The pooled-code regularizers on one GPU against a Float64 replay of the same formulas: the
 *  coding rate, the top eigenvalue after the configured power steps, and the force on every raw
 *  row, with an all-zero row taking none. The replay's own force is checked by central differences
 *  of its loss, and a second call reproduces the terms bitwise and doubles the force. */
static void test_spectrum_replay(device_t& device) {
    [[maybe_unused]] util::status_t const scope = (device.stream).bind();
    // Three tiles of channels run the panel, the trailing update and the mirror; 37 rows pad to 40.
    std::size_t const rows = 37, hidden = 96, empty_row = 5;
    pooled_spectrum_config_t config;
    config.rows_per_gpu = static_cast<util::u32_t>(rows);
    config.hidden = static_cast<util::u32_t>(hidden);
    config.rows_per_world = static_cast<util::u32_t>(rows);
    config.coding_rate_weight = 1.0f;
    config.epsilon_squared = 0.5f;
    config.lipschitz_weight = 0.5f;

    // A shared offset gives the rows a pooled code's anisotropy, so the top eigenvalue stands out.
    st::basic_vector<util::f64_t> pooled_host;
    st::basic_vector<nk_bf16_t> staging;
    if (!expect_true(!st::failed(pooled_host.resize(rows * hidden)) && !st::failed(staging.resize(rows * hidden)),
                     "host staging failed"))
        return;
    f64r2_span_t const pooled_rows {pooled_host.data(), rows, hidden};
    for (std::size_t row = 0; row != rows; ++row)
        for (std::size_t channel = 0; channel != hidden; ++channel)
            pooled_rows[row, channel] = row == empty_row
                                            ? 0.0
                                            : 0.3 + keyed_uniform_(test_run.settings.seed, 11u, row * hidden + channel);
    if (!expect_true(narrow_to_bf16(pooled_host, staging), "host rounding failed")) return;

    // The Float64 replay: the power iteration off the exact unit rows, the coding rate off the
    // BFloat16 ones the device's Gram reads.
    st::basic_vector<util::f64_t> units, rounded, inverse_norms, lower, direction, image, projections, solve,
        expected_force;
    st::basic_vector<nk_bf16_t> rounded_staging;
    if (!expect_true(!st::failed(units.resize(rows * hidden)) && !st::failed(rounded.resize(rows * hidden)) &&
                         !st::failed(inverse_norms.resize(rows)) && !st::failed(lower.resize(hidden * hidden)) &&
                         !st::failed(direction.resize(hidden)) && !st::failed(image.resize(hidden)) &&
                         !st::failed(projections.resize(rows)) && !st::failed(solve.resize(hidden)) &&
                         !st::failed(expected_force.resize(rows * hidden)) &&
                         !st::failed(rounded_staging.resize(rows * hidden)),
                     "replay staging failed"))
        return;
    f64r2_span_t const unit_rows {units.data(), rows, hidden};
    f64r2_span_t const factor {lower.data(), hidden, hidden};
    normalize_rows_(pooled_rows, unit_rows, inverse_norms);
    std::copy(units.begin(), units.end(), rounded.begin());
    if (!expect_true(narrow_to_bf16(rounded, rounded_staging), "unit rounding failed")) return;
    util::f64_t const alpha = util::f64_t(hidden) / (util::f64_t(rows) * config.epsilon_squared);
    util::f64_t const expected_rate = coding_rate_replay_(f64r2_view_t {rounded.data(), rows, hidden}, alpha, factor);
    std::fill(direction.begin(), direction.end(), 1.0 / std::sqrt(util::f64_t(hidden)));
    for (util::u32_t iteration = 0;; ++iteration) {
        for (std::size_t row = 0; row != rows; ++row)
            projections[row] = dot(std::span<util::f64_t const> {&unit_rows[row, 0], hidden}, direction);
        if (iteration == config.power_iterations) break;
        std::fill(image.begin(), image.end(), 0.0);
        for (std::size_t row = 0; row != rows; ++row)
            for (std::size_t channel = 0; channel != hidden; ++channel)
                image[channel] += unit_rows[row, channel] * projections[row];
        util::f64_t const norm = std::sqrt(dot(image, image));
        for (std::size_t channel = 0; channel != hidden; ++channel) direction[channel] = image[channel] / norm;
    }
    util::f64_t const expected_top = dot(projections, projections);

    // Every row's force with the Gram factored off `gram_units`, returning the largest magnitude.
    auto replay_forces = [&](f64r2_view_t gram_units, f64r2_span_t forces) noexcept {
        (void)coding_rate_replay_(gram_units, alpha, factor);
        util::f64_t largest = 0.0;
        for (std::size_t row = 0; row != rows; ++row) {
            std::span<util::f64_t const> const unit {&unit_rows[row, 0], hidden};
            std::copy(unit.begin(), unit.end(), solve.begin());
            for (std::size_t pivot = 0; pivot != hidden; ++pivot) {
                for (std::size_t inner = 0; inner != pivot; ++inner)
                    solve[pivot] -= factor[pivot, inner] * solve[inner];
                solve[pivot] /= factor[pivot, pivot];
            }
            for (std::size_t pivot = hidden; pivot-- != 0;) {
                for (std::size_t inner = pivot + 1; inner != hidden; ++inner)
                    solve[pivot] -= factor[inner, pivot] * solve[inner];
                solve[pivot] /= factor[pivot, pivot];
            }
            for (std::size_t channel = 0; channel != hidden; ++channel)
                solve[channel] = -util::f64_t(config.coding_rate_weight) * alpha * solve[channel] +
                                 2.0 * util::f64_t(config.lipschitz_weight) * projections[row] * direction[channel];
            util::f64_t const radial = dot(unit, solve);
            for (std::size_t channel = 0; channel != hidden; ++channel) {
                forces[row, channel] = (solve[channel] - unit[channel] * radial) * inverse_norms[row];
                largest = fu::max_of(largest, std::fabs(forces[row, channel]));
            }
        }
        return largest;
    };

    // The exact force against central differences of the replayed loss at its settled direction.
    f64r2_span_t const forces {expected_force.data(), rows, hidden};
    util::f64_t const largest_exact = replay_forces(unit_rows, forces);
    for (std::size_t probe = 0; probe != 4; ++probe) {
        std::size_t const row = (probe * 11 + 1) % rows, channel = (probe * 29 + 3) % hidden;
        util::f64_t const original = pooled_rows[row, channel], step = 1.0e-6;
        pooled_rows[row, channel] = original + step;
        util::f64_t const plus = spectrum_loss_replay_(pooled_rows, config, direction);
        pooled_rows[row, channel] = original - step;
        util::f64_t const minus = spectrum_loss_replay_(pooled_rows, config, direction);
        pooled_rows[row, channel] = original;
        util::f64_t const measured = (plus - minus) / (2.0 * step);
        expect_true(std::fabs(measured - forces[row, channel]) <= 1.0e-6 * largest_exact,
                    "replay force [{},{}] {:.9g} against its central difference {:.9g}", row, channel,
                    forces[row, channel], measured);
    }
    // The device's Gram reads the BFloat16 unit rows, moving its force by under 1% of the peak.
    util::f64_t const largest_force = replay_forces(f64r2_view_t {rounded.data(), rows, hidden}, forces);

    // The one-GPU world: one host domain and one multimem domain, every collective the identity.
    auto [arena, make_arena_status] = util::gpu_arena_t::make(0, case_arena_bytes_k);
    if (!expect_true(make_arena_status, "arena: {}", make_arena_status.description_)) return;
    util::gpu_arena_alloc<util::bf16_t> bf16_alloc {arena};
    util::bf16r2_tensor_arena_t pooled = make_uninitialized<util::bf16r2_tensor_arena_t>({rows, hidden}, bf16_alloc);
    util::bf16r2_tensor_arena_t pooled_grad = make_uninitialized<util::bf16r2_tensor_arena_t>({rows, hidden},
                                                                                              bf16_alloc);
    if (!expect_true(pooled && pooled_grad, "pooled rows allocation failed")) return;
    cuMemcpyHtoD(util::device_address(pooled.data()), staging.data(), staging.size() * sizeof(nk_bf16_t));
    if (util::expected_t zeroed = util::zero_async(pooled_grad.span(), device.stream);
        !expect_true(zeroed, "gradient zeroing: {}", zeroed.description_))
        return;
    auto [topology, bootstrap_status] = util::mesh_topology_t::bootstrap(
        0, 1, &pause_between_polls, 0, "/tmp/uform_exchange_local", "spectrum_replay");
    if (!expect_true(bootstrap_status, "single-domain bootstrap: {}", bootstrap_status.description_)) return;
    util::i32_t const devices[1] = {0};
    util::cuda_executor_t& streams = device.executor;
    auto [domain, make_domain_status] = util::multimem_domain_t::make(
        topology, devices, distributed_pooled_spectrum_t::payload_scalars_needed(config), streams,
        wait_limit_k.count());
    if (!expect_true(make_domain_status, "multimem domain: {}", make_domain_status.description_)) return;
    auto [colocated, make_colocated_status] = util::colocated_domain_t::make(topology);
    if (!expect_true(make_colocated_status, "colocated domain: {}", make_colocated_status.description_)) return;
    auto [spectrum, make_spectrum_status] = distributed_pooled_spectrum_t::make(
        domain, colocated, std::span<util::gpu_arena_t> {&arena, 1}, streams, {}, 0, config, device.numerics);
    if (!expect_true(make_spectrum_status, "spectrum: {}", make_spectrum_status.description_)) return;

    util::f32_t terms[2][2] = {};
    st::basic_vector<util::f64_t> gradient;
    for (std::size_t call = 0; call != 2; ++call) {
        if (util::expected_t shaped = spectrum(0, pooled.view(), pooled_grad.span(), device.stream);
            !expect_true(shaped, "call {}: {}", call, shaped.description_))
            return;
        if (util::expected_t copied = util::copy_device_to_host(spectrum.terms(0), std::span<util::f32_t> {terms[call]},
                                                                device.stream);
            !expect_true(copied, "terms download: {}", copied.description_))
            return;
        cuStreamSynchronize(util::native_stream(device.stream));
        if (!expect_true(download_widened_({pooled_grad.data(), rows * hidden}, staging, gradient),
                         "gradient download failed"))
            return;
        f64r2_view_t const gradient_rows {gradient.data(), rows, hidden};
        util::f64_t const scale = util::f64_t(call + 1);
        util::f64_t worst = 0.0, empty = 0.0;
        for (std::size_t row = 0; row != rows; ++row)
            for (std::size_t channel = 0; channel != hidden; ++channel) {
                worst = fu::max_of(worst, std::fabs(gradient_rows[row, channel] - scale * forces[row, channel]));
                if (row == empty_row) empty = fu::max_of(empty, std::fabs(gradient_rows[row, channel]));
            }
        // One BFloat16 rounding per call over Float32 solves of a condition number near 200.
        expect_true(worst <= 0x1.0p-7 * scale * largest_force,
                    "call {}: the force misses the replay by {:.3g} of its peak {:.3g}", call, worst,
                    scale * largest_force);
        expect_true(empty == 0.0, "call {}: the all-zero row takes no force, got {:.3g}", call, empty);
    }
    expect_true(std::fabs(terms[0][0] - expected_rate) <= 1.0e-5 * expected_rate,
                "the coding rate {:.7g} against the replay's {:.7g}", terms[0][0], expected_rate);
    expect_true(std::fabs(terms[0][1] - expected_top) <= 1.0e-5 * expected_top,
                "the top eigenvalue {:.7g} against the replay's {:.7g}", terms[0][1], expected_top);
    expect_true(std::memcmp(terms[0], terms[1], sizeof(terms[0])) == 0,
                "a second call reproduces the terms bitwise: rate {} vs {}, top {} vs {}", terms[0][0], terms[1][0],
                terms[0][1], terms[1][1]);
}

int main(int argc, char** argv) {
    ::setenv("CUDA_MODULE_LOADING", "EAGER", 1);
    start_test_run(argc, argv);
    int device_count = 0;
    char const* const labels[] = {"test_finite_differences_tokens",
                                  "test_finite_differences_omni",
                                  "test_token_head_finite_differences",
                                  "test_masked_text_rows",
                                  "test_masked_media_rows",
                                  "test_chance_cross_entropy",
                                  "test_adafactor_reference(device, 0.0f)",
                                  "test_adafactor_reference(device, 0.9f)",
                                  "test_adafactor_small_steps",
                                  "test_block_host_replay",
                                  "test_block_recompute_replays",
                                  "test_choices_cross_entropy",
                                  "test_spectrum_replay"};
    if (cuInit(0) != CUDA_SUCCESS || cuDeviceGetCount(&device_count) != CUDA_SUCCESS || device_count <= 0) {
        for (char const* label : labels) run_test_case(label, [] { return case_result_t::skipped(no_gpu_device_k); });
        return finish_test_run();
    }
    int const stream_devices[] = {static_cast<int>(0)};
    auto [stream_executor, status] = spawn_test_executor(stream_devices);
    if (!expect_true(status, "stream: {}", status.description_)) return finish_test_run();
    auto& stream = stream_executor.stream(0);
    [[maybe_unused]] util::status_t const scope = (stream).bind();
    nk_capability_t capabilities = 0;
    if (!expect_true(nk_capabilities_enabled_cuda(0, &capabilities) == nk_success_k, "NumKong cannot query device 0"))
        return finish_test_run();
    device_t device {stream_executor, stream, gpu_numerics_of(capabilities)};
    run_test_case("test_finite_differences_tokens", test_finite_differences_tokens, device);
    run_test_case("test_finite_differences_omni", test_finite_differences_omni, device);
    run_test_case("test_token_head_finite_differences", test_token_head_finite_differences, device);
    run_test_case("test_masked_text_rows", test_masked_text_rows, device);
    run_test_case("test_masked_media_rows", test_masked_media_rows, device);
    run_test_case("test_chance_cross_entropy", test_chance_cross_entropy, device);
    run_test_case("test_adafactor_reference(device, 0.0f)", test_adafactor_reference, device, 0.0f);
    run_test_case("test_adafactor_reference(device, 0.9f)", test_adafactor_reference, device, 0.9f);
    run_test_case("test_adafactor_small_steps", test_adafactor_small_steps, device);
    run_test_case("test_block_host_replay", test_block_host_replay, device);
    run_test_case("test_block_recompute_replays", test_block_recompute_replays, device);
    run_test_case("test_choices_cross_entropy", test_choices_cross_entropy, device);
    run_test_case("test_spectrum_replay", test_spectrum_replay, device);
    return finish_test_run();
}
