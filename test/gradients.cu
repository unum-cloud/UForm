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
 *      - @ref finite_differences over the token-only and the omni mixtures: each registered matrix
 *        is perturbed along its own gradient, by steps rounded onto the BFloat16 grid, and the loss
 *        change of two forwards is compared with the gradient's dot product with the rounded step.
 *        That covers the block backward, the embedding scatter, the media projections, the token
 *        cross-entropy head and the patch and frame MSE heads.
 *      - @ref token_head_finite_differences : the late-interaction token head and its join into the
 *        tower, under a linear loss over the token vectors, with prefixes and side-less segments
 *        left out.
 *      - @ref masked_text_rows : every text row with bytes carries at least one masked position.
 *      - @ref masked_media_rows : each media head scores only the cells of its own modality.
 *      - @ref chance_cross_entropy : a freshly seeded model starts at the uniform byte prediction.
 *      - @ref adafactor_reference : two Adafactor steps on a matrix and a vector land where the
 *        paper's update, replayed in Float64 on the host, puts them, up to one BFloat16 spacing.
 *      - @ref adafactor_small_steps : steps far below a BFloat16 spacing still move the weights by
 *        their mean.
 *      - @ref block_host_replay : one block's forward and attention backward over ragged, padded
 *        segments with M-RoPE coordinates on every axis against a Float64 host replay, the one
 *        check exact enough for the attention and rotary paths.
 *
 *  The forward and the backward both round their activations to BFloat16, so a forward's loss
 *  carries a few thousandths of a nat of rounding noise and a finite difference agrees with the
 *  analytic gradient to a few percent, not to Float32 precision; a wrong Jacobian term misses by
 *  tens of percent, and the block replay checks the attention path exactly.
 */
#include <cmath>   // `std::fabs`, `std::sqrt`, `std::log`, `std::pow`, `std::frexp`, `std::ldexp`
#include <cstdlib> // `::setenv`
#include <cstdio>  // `std::remove`
#include <cstring> // `std::memcpy`, `std::memcmp`

#include <unistd.h> // `::getpid`

#include <algorithm> // `std::max`, `std::min`
#include <span>      // `std::span`

#include <cuda.h> // `cuInit`, `cuMemcpyHtoD`, `cuMemcpyDtoH`, `cuStreamSynchronize`

#include <numkong/reduce.hpp> // `nk::reduce_moments`

#include "train/base.cuh"             // `base_replica_t`, `trainer_config_t`, `seed_weights`
#include "core/optimizer.hpp"         // `adafactor_t`
#include "core/transformer_block.cuh" // `transformer_block_bf16_t`, `transformer_scratch_t`
#include "harness.hpp"                // `test_suite_t`, `dot`, `narrow_to_bf16`, `widen_from_bf16`

using namespace unum::uform::core;
using namespace unum::uform::train;
using namespace unum::uform::test;

/** Download @p count BFloat16 scalars at @p device into @p widened , widened once by NumKong. */
[[nodiscard]] static bool download_widened_(bf16_t const *device, std::size_t count,
                                            st::basic_vector<f64_t> &widened) noexcept {
    st::basic_vector<nk_bf16_t> staging;
    if (st::failed(staging.resize(st::assume_uninitialized, count)) ||
        st::failed(widened.resize(st::assume_uninitialized, count)))
        return false;
    cuMemcpyDtoH(staging.data(), device_address(device), count * sizeof(nk_bf16_t));
    return widen_from_bf16(staging, widened);
}

/** Narrow @p values onto the BFloat16 grid in place and upload them to @p device , landed before
 *  the non-blocking stream reads them. */
[[nodiscard]] static bool upload_narrowed_(std::span<f64_t> values, bf16_t *device) noexcept {
    st::basic_vector<nk_bf16_t> staging;
    if (st::failed(staging.resize(st::assume_uninitialized, values.size())) || !narrow_to_bf16(values, staging))
        return false;
    cuMemcpyHtoD(device_address(device), staging.data(), staging.size() * sizeof(nk_bf16_t));
    cuCtxSynchronize(); // a pageable copy may return before its DMA lands
    return true;
}

/** The GPU every case borrows, owned by @c main : one stream whose context the cases enter. Each
 *  case allocates its own arena, so a case's allocations die with it. */
struct device_t {
    gpu_stream_t stream;
};

/** Bytes one case's arena reserves: the omni replica at the test geometry needs about a quarter of
 *  it. */
inline constexpr std::size_t case_arena_bytes_k = 1ull << 30;

/** The relative error a finite difference may show against the analytic gradient: the BFloat16
 *  activations' loss noise over a quarter-nat step, with room left to catch a 15% Jacobian
 *  error. */
inline constexpr f64_t finite_difference_tolerance_k = 0x1.0p-4;

/** A two-block, two-head tower over four rows of 64 columns, scoring the masked objective; @p omni
 *  lights every media slice, so the keyed partition holds one row of each modality. */
[[nodiscard]] static trainer_config_t test_geometry_(bool omni) noexcept {
    trainer_config_t config;
    config.batch = multimodal_batch_shape_t {4u, 4u, 64u};
    if (omni) {
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
[[nodiscard]] static expected<base_replica_t> make_replica_(gpu_arena_t &arena, device_t &device,
                                                            trainer_config_t const &config,
                                                            data_epoch_t epoch) noexcept {
    expected<base_replica_t> replica = base_replica_t::make(arena, config, 0);
    expected_t staged = replica ? seed_weights(replica->weights, device.stream) : expected_t::fail(std::move(replica));
    if (staged) staged = replica->draw(config, 0, epoch, device.stream);
    if (staged) staged = replica->prepare(config, epoch, device.stream);
    if (!staged) return expected<base_replica_t>::fail(std::move(staged));
    return replica;
}

/** Draw and corrupt the batch of epoch @p epoch into an existing @p replica . */
[[nodiscard]] static expected_t redraw_(base_replica_t &replica, trainer_config_t const &config, data_epoch_t epoch,
                                        gpu_stream_t &stream) noexcept {
    expected_t staged = replica.draw(config, 0, epoch, stream);
    if (staged) staged = replica.prepare(config, epoch, stream);
    return staged;
}

/** One @c {sum,count} summand pair read back, zero when the head is absent. */
struct summand_pair_t {
    f64_t sum = 0, count = 0;
};

[[nodiscard]] static summand_pair_t read_pair_(f32r1_tensor_arena_t const &summands) noexcept {
    if (summands.empty()) return {};
    f32_t host[2] = {0.0f, 0.0f};
    cuMemcpyDtoH(host, device_address(summands.data()), sizeof(host));
    return summand_pair_t {host[0], host[1]};
}

/** Run the forward region from zeroed summands and return the composite loss the backward
 *  differentiates: every live head's weighted sum over its own count. */
[[nodiscard]] static expected<f64_t> forward_loss_(base_replica_t &replica, gpu_stream_t &stream) noexcept {
    f32r1_tensor_arena_t *const pairs[] = {&replica.tokens_summands, &replica.image_summands, &replica.video_summands,
                                           &replica.audio_summands};
    for (f32r1_tensor_arena_t *pair : pairs)
        if (!pair->empty())
            if (expected_t zeroed = zero_async(pair->span(), stream); !zeroed)
                return expected<f64_t>::fail(std::move(zeroed));
    if (expected_t forwarded = replica.forward_region(stream); !forwarded)
        return expected<f64_t>::fail(std::move(forwarded));
    if (CUresult result = cuStreamSynchronize(stream.get()); result != CUDA_SUCCESS)
        return cuda_failure<f64_t>(result, "forward drain failed");
    f64_t loss = 0;
    for (f32r1_tensor_arena_t *pair : pairs) {
        summand_pair_t const read = read_pair_(*pair);
        if (read.count != 0) loss += read.sum / read.count;
    }
    return expected<f64_t> {loss};
}

/** Perturb every registered matrix along its own gradient and compare the central difference of the
 *  composite loss with the gradient's dot product with the BFloat16-rounded step. */
static void finite_differences(test_suite_t &suite, device_t &device, bool omni) {
    switch_to_gpu_t scope {device.stream};
    // One block keeps every perturbed matrix's loss change far above the BFloat16 noise of a deeper
    // stack; the block's own backward is checked exactly by block_host_replay.
    trainer_config_t config = test_geometry_(omni);
    config.depth = 1;
    config.seed = suite.settings.seed;
    expected<gpu_arena_t> arena = gpu_arena_t::make(0, case_arena_bytes_k);
    if (!arena) return suite.check(false, "arena: {}", arena.description_);
    expected<base_replica_t> made = make_replica_(*arena, device, config, data_epoch_t {0});
    if (!made) return suite.check(false, "replica: {}", made.description_);
    base_replica_t &replica = *made;
    expected<f64_t> const base_loss = forward_loss_(replica, device.stream);
    if (!base_loss) return suite.check(false, "forward: {}", base_loss.description_);
    if (expected_t backward = replica.backward_region(device.stream); !backward)
        return suite.check(false, "backward: {}", backward.description_);
    cuStreamSynchronize(device.stream.get());

    st::basic_vector<f64_t> weights, gradient, plus, minus;
    st::basic_vector<nk_bf16_t> staging;
    for (differentiable_weights_t const &matrix : replica.weights) {
        std::size_t const scalars = matrix.weights.numel();
        if (!download_widened_(matrix.weights.data(), scalars, weights) ||
            !download_widened_(matrix.weights_grad.data(), scalars, gradient) ||
            st::failed(plus.resize(st::assume_uninitialized, scalars)) ||
            st::failed(minus.resize(st::assume_uninitialized, scalars)) ||
            st::failed(staging.resize(st::assume_uninitialized, scalars)))
            return suite.check(false, "host staging for {} failed", matrix.name);
        f64_t largest = 0;
        for (f64_t const value : gradient) largest = std::max(largest, std::fabs(value));
        // The head and pool weights outside the masked objective take no gradient and move no loss.
        if (largest == 0) continue;
        f64_t const weight_rms = std::sqrt(dot(weights, weights) / static_cast<f64_t>(scalars));
        f64_t predicted = 0;
        // Double the step until the BF16-rounded move predicts an eighth of a nat, never past the
        // matrix's RMS.
        f64_t const first_step = largest / (8.0 * dot(gradient, gradient));
        for (f64_t step = first_step; step == first_step || (predicted < 0.125 && step <= weight_rms); step *= 2) {
            for (std::size_t index = 0; index != scalars; ++index)
                plus[index] = weights[index] + step * gradient[index] / largest,
                minus[index] = weights[index] - step * gradient[index] / largest;
            if (!narrow_to_bf16(plus, staging) || !narrow_to_bf16(minus, staging))
                return suite.check(false, "host rounding for {} failed", matrix.name);
            predicted = dot(gradient, plus) - dot(gradient, minus);
        }
        bool uploaded = upload_narrowed_(plus, matrix.weights.data());
        expected<f64_t> const loss_plus = forward_loss_(replica, device.stream);
        uploaded = uploaded && upload_narrowed_(minus, matrix.weights.data());
        expected<f64_t> const loss_minus = forward_loss_(replica, device.stream);
        uploaded = uploaded && upload_narrowed_(weights, matrix.weights.data());
        if (!uploaded || !loss_plus || !loss_minus)
            return suite.check(false, "perturbed forward of {} failed", matrix.name);
        f64_t const measured = *loss_plus - *loss_minus;
        f64_t const gap = std::fabs(measured - predicted), error = gap / std::max(std::fabs(predicted), 1.0e-6);
        suite.check(error <= finite_difference_tolerance_k,
                    "{}: loss moved {:.6g}, the gradient predicts {:.6g}, off by {:.3g} relative (loss {:.4f})",
                    matrix.name, measured, predicted, error, *base_loss);
    }
}

static void finite_differences_tokens(test_suite_t &suite, device_t &device) {
    finite_differences(suite, device, false);
}

static void finite_differences_omni(test_suite_t &suite, device_t &device) { finite_differences(suite, device, true); }

/** Host copies of the batch's segment table. */
struct host_segments_t {
    st::basic_vector<u32_t> offsets {}, lengths {};
    st::basic_vector<modality_t> modality {};
};

[[nodiscard]] static host_segments_t read_segments_(multimodal_batch_on_gpu_t const &batch) noexcept {
    host_segments_t segments;
    std::size_t const count = batch.segment_lengths.extent(0);
    if (st::failed(segments.offsets.resize(count)) || st::failed(segments.lengths.resize(count)) ||
        st::failed(segments.modality.resize(count)))
        return {};
    cuMemcpyDtoH(segments.offsets.data(), device_address(batch.segment_offsets.data()), count * sizeof(u32_t));
    cuMemcpyDtoH(segments.lengths.data(), device_address(batch.segment_lengths.data()), count * sizeof(u32_t));
    cuMemcpyDtoH(segments.modality.data(), device_address(batch.segment_modality.data()), count * sizeof(modality_t));
    return segments;
}

/** The linear loss @c ⟨R,V⟩ over the token head's unit vectors @c V after one tower forward,
 *  normalized in Float64 from the Float32 projections, since the BFloat16 vectors' rounding noise
 *  would swamp a central difference of the projection. */
[[nodiscard]] static expected<f64_t> token_head_loss_(base_replica_t &replica, st::basic_vector<f64_t> const &direction,
                                                      gpu_stream_t &stream) noexcept {
    expected_t forwarded = replica.tower.pack_weights(stream);
    if (forwarded) forwarded = replica.tower.forward(replica.batch, {}, stream);
    if (!forwarded) return expected<f64_t>::fail(std::move(forwarded));
    cuStreamSynchronize(stream.get());
    f32r2_view_t const projections = replica.tower.token_projections();
    std::size_t const tokens = projections.extent(0), width = projections.extent(1);
    st::basic_vector<f32_t> rows;
    if (st::failed(rows.resize(st::assume_uninitialized, tokens * width)))
        return expected<f64_t>::fail(status_t::bad_alloc_k, "token projection staging failed");
    cuMemcpyDtoH(rows.data(), device_address(projections.data()), tokens * width * sizeof(f32_t));
    f32r2_view_t const projection_rows {rows.data(), tokens, width};
    f64r2_view_t const direction_rows {direction.data(), tokens, width};
    f64_t const epsilon = replica.tower.config().block.rms_eps;
    f64_t loss = 0;
    for (std::size_t token = 0; token != tokens; ++token) {
        f64_t squares = 0, along = 0;
        for (std::size_t channel = 0; channel != width; ++channel) {
            f64_t const value = projection_rows[token, channel];
            squares += value * value, along += value * direction_rows[token, channel];
        }
        loss += along / std::sqrt(std::max(squares, epsilon));
    }
    return expected<f64_t> {loss};
}

/** The token head and its Float32 join into the tower: a fixed linear loss over the token vectors,
 *  whose gradient is its direction, against central differences of the projection and the final
 *  norm, and the norm scale against a Float64 replay, with each segment's prefix and the side-less
 *  segments left out of the compact rows. */
static void token_head_finite_differences(test_suite_t &suite, device_t &device) {
    switch_to_gpu_t scope {device.stream};
    trainer_config_t config = test_geometry_(false);
    config.depth = 1;
    config.seed = suite.settings.seed;
    config.objective = objective_t::contrastive_k;
    config.late_interaction.weight = 1.0f;
    config.late_interaction.width = 32;
    expected<gpu_arena_t> arena = gpu_arena_t::make(0, case_arena_bytes_k);
    if (!arena) return suite.check(false, "arena: {}", arena.description_);
    expected<base_replica_t> made = base_replica_t::make(*arena, config, config.batch.max_segments_per_gpu);
    expected_t staged = made ? seed_weights(made->weights, device.stream) : expected_t::fail(std::move(made));
    if (staged) staged = made->draw(config, 0, data_epoch_t {0}, device.stream);
    if (!staged) return suite.check(false, "replica: {}", staged.description_);
    base_replica_t &replica = *made;
    cuStreamSynchronize(device.stream.get());

    // Sides and prefixes over the four drawn segments: a query, a document, a side-less segment and
    // one of both.
    host_segments_t const segments = read_segments_(replica.batch);
    std::size_t const count = segments.lengths.size();
    late_interaction_side_t const sides[] = {late_interaction_side_t::query_k, late_interaction_side_t::document_k,
                                             late_interaction_side_t::none_k,
                                             late_interaction_side_t::query_and_document_k};
    u32_t const prefixes[] = {3, 0, 2, 5};
    u32_t expected_offsets[5] = {};
    for (std::size_t segment = 0; segment != count; ++segment) {
        u32_t const length = segments.lengths[segment];
        bool const sided = sides[segment] != late_interaction_side_t::none_k;
        expected_offsets[segment + 1] = expected_offsets[segment] +
                                        (sided && length > prefixes[segment] ? length - prefixes[segment] : 0u);
    }
    cuMemcpyHtoD(device_address(replica.batch.segment_side.data()), sides, sizeof(sides));
    cuMemcpyHtoD(device_address(replica.batch.segment_prefix_length.data()), prefixes, sizeof(prefixes));

    bf16r2_view_t const token_vectors = replica.tower.token_vectors();
    st::basic_vector<f64_t> direction;
    if (st::failed(direction.resize(st::assume_uninitialized, token_vectors.numel())))
        return suite.check(false, "direction staging failed");
    for (std::size_t index = 0; index != direction.size(); ++index)
        direction[index] = f64_t((index * 2654435761ull + 40503ull) % 1000ull) / 1000.0 - 0.5;
    if (!upload_narrowed_(direction, replica.token_vectors_grad.data()))
        return suite.check(false, "direction upload failed");
    expected<f64_t> const base_loss = token_head_loss_(replica, direction, device.stream);
    if (!base_loss) return suite.check(false, "forward: {}", base_loss.description_);
    u32_t offsets[5] = {};
    cuMemcpyDtoH(offsets, device_address(replica.tower.token_offsets().data()), sizeof(offsets));
    bool const laid_out = std::equal(offsets, offsets + count + 1, expected_offsets);
    suite.check(laid_out, "token head spans skip prefixes and side-less segments: total {} expected {}", offsets[count],
                expected_offsets[count]);
    if (expected_t backward = replica.tower.backward({}, {}, replica.batch, device.stream,
                                                     replica.token_vectors_grad.view());
        !backward)
        return suite.check(false, "backward: {}", backward.description_);
    cuStreamSynchronize(device.stream.get());

    // The norm scale's gradient against a Float64 replay of the head over the same states: through
    // the L2 normalization it moves the loss only at second order along most directions, too little
    // for a difference.
    st::basic_vector<f64_t> states, scale, scale_grad, projection;
    differentiable_weights_t const *norm = nullptr, *projected = nullptr;
    for (differentiable_weights_t const &matrix : replica.weights)
        if (matrix.name.view() == "token_head.norm") norm = &matrix;
        else if (matrix.name.view() == "token_head.projection") projected = &matrix;
    bf16r2_view_t const normalized = replica.tower.normalized();
    std::size_t const hidden = config.hidden, width = config.late_interaction.width;
    if (norm == nullptr || projected == nullptr || !download_widened_(normalized.data(), normalized.numel(), states) ||
        !download_widened_(norm->weights.data(), hidden, scale) ||
        !download_widened_(norm->weights_grad.data(), hidden, scale_grad) ||
        !download_widened_(projected->weights.data(), width * hidden, projection))
        return suite.check(false, "token head replay staging failed");
    f64_t const epsilon = replica.tower.config().block.rms_eps;
    f64r2_view_t const projection_rows {projection.data(), width, hidden};
    f64r2_view_t const direction_rows {direction.data(), direction.size() / width, width};
    st::basic_vector<f64_t> replayed, unit, normed;
    if (st::failed(replayed.resize(hidden)) || st::failed(unit.resize(width)) || st::failed(normed.resize(hidden)))
        return suite.check(false, "token head replay staging failed");
    for (std::size_t segment = 0; segment != count; ++segment)
        for (u32_t token = 0; token != expected_offsets[segment + 1] - expected_offsets[segment]; ++token) {
            std::size_t const row = expected_offsets[segment] + token;
            f64_t const *const state = states.data() + (segments.offsets[segment] + prefixes[segment] + token) * hidden;
            f64_t squares = 0;
            for (std::size_t channel = 0; channel != hidden; ++channel) squares += state[channel] * state[channel];
            f64_t const inverse_rms = 1.0 / std::sqrt(squares / f64_t(hidden) + epsilon);
            for (std::size_t channel = 0; channel != hidden; ++channel)
                normed[channel] = state[channel] * inverse_rms * scale[channel];
            f64_t norm_squared = 0, along = 0;
            for (std::size_t output = 0; output != width; ++output) {
                unit[output] = 0;
                for (std::size_t channel = 0; channel != hidden; ++channel)
                    unit[output] += projection_rows[output, channel] * normed[channel];
                norm_squared += unit[output] * unit[output];
            }
            f64_t const inverse_norm = 1.0 / std::sqrt(std::max(norm_squared, epsilon));
            for (std::size_t output = 0; output != width; ++output)
                unit[output] *= inverse_norm, along += unit[output] * direction_rows[row, output];
            for (std::size_t channel = 0; channel != hidden; ++channel) {
                f64_t normed_grad = 0;
                for (std::size_t output = 0; output != width; ++output)
                    normed_grad += projection_rows[output, channel] *
                                   (direction_rows[row, output] - along * unit[output]) * inverse_norm;
                replayed[channel] += normed_grad * state[channel] * inverse_rms;
            }
        }
    f64_t gap = 0, magnitude = 0;
    for (std::size_t channel = 0; channel != hidden; ++channel)
        gap += (scale_grad[channel] - replayed[channel]) * (scale_grad[channel] - replayed[channel]),
            magnitude += replayed[channel] * replayed[channel];
    suite.check(std::sqrt(gap) <= 2.0e-2 * std::sqrt(magnitude),
                "token_head.norm gradient matches the Float64 replay: relative {:.3g}", std::sqrt(gap / magnitude));

    st::basic_vector<f64_t> weights, gradient, plus, minus;
    for (differentiable_weights_t const &matrix : replica.weights) {
        sz::string_view_t const name = matrix.name.view();
        if (name != "token_head.projection" && name != "final_norm") continue;
        std::size_t const scalars = matrix.weights.numel();
        if (!download_widened_(matrix.weights.data(), scalars, weights) ||
            !download_widened_(matrix.weights_grad.data(), scalars, gradient) ||
            st::failed(plus.resize(st::assume_uninitialized, scalars)) ||
            st::failed(minus.resize(st::assume_uninitialized, scalars)))
            return suite.check(false, "host staging for {} failed", name);
        f64_t largest = 0;
        for (f64_t const value : gradient) largest = std::max(largest, std::fabs(value));
        if (largest == 0) {
            suite.check(false, "{} takes no gradient through the token head", name);
            continue;
        }
        f64_t const weight_rms = std::sqrt(dot(weights, weights) / static_cast<f64_t>(scalars));
        f64_t const step = std::min(largest / (8.0 * dot(gradient, gradient)), weight_rms / 4.0);
        for (std::size_t index = 0; index != scalars; ++index)
            plus[index] = weights[index] + step * gradient[index] / largest,
            minus[index] = weights[index] - step * gradient[index] / largest;
        bool uploaded = upload_narrowed_(plus, matrix.weights.data());
        expected<f64_t> const loss_plus = token_head_loss_(replica, direction, device.stream);
        uploaded = uploaded && upload_narrowed_(minus, matrix.weights.data());
        expected<f64_t> const loss_minus = token_head_loss_(replica, direction, device.stream);
        uploaded = uploaded && upload_narrowed_(weights, matrix.weights.data());
        if (!uploaded || !loss_plus || !loss_minus) return suite.check(false, "perturbed forward of {} failed", name);
        f64_t const predicted = dot(gradient, plus) - dot(gradient, minus);
        f64_t const measured = *loss_plus - *loss_minus;
        f64_t const error = std::fabs(measured - predicted) / std::max(std::fabs(predicted), 1.0e-6);
        suite.check(error <= finite_difference_tolerance_k,
                    "{}: loss moved {:.6g}, the gradient predicts {:.6g}, off by {:.3g} relative (loss {:.4f})", name,
                    measured, predicted, error, *base_loss);
    }
}

/** Every text row with bytes carries at least one masked position, over sixteen 64-byte rows and
 *  eight epochs. */
static void masked_text_rows(test_suite_t &suite, device_t &device) {
    switch_to_gpu_t scope {device.stream};
    trainer_config_t config = test_geometry_(false);
    config.batch = multimodal_batch_shape_t {16u, 16u, 64u};
    config.seed = suite.settings.seed;
    expected<gpu_arena_t> arena = gpu_arena_t::make(0, case_arena_bytes_k);
    if (!arena) return suite.check(false, "arena: {}", arena.description_);
    expected<base_replica_t> replica = make_replica_(*arena, device, config, data_epoch_t {0});
    if (!replica) return suite.check(false, "replica: {}", replica.description_);
    st::basic_vector<u32_t> weights;
    for (u32_t epoch = 0; epoch != 8u; ++epoch) {
        if (expected_t drawn = epoch == 0 ? expected_t {}
                                          : redraw_(*replica, config, data_epoch_t {epoch}, device.stream);
            !drawn)
            return suite.check(false, "epoch {}: {}", epoch, drawn.description_);
        cuStreamSynchronize(device.stream.get());
        std::size_t const tokens = replica->token_loss_weights.extent(0);
        if (st::failed(weights.resize(tokens))) return suite.check(false, "weights staging failed");
        cuMemcpyDtoH(weights.data(), device_address(replica->token_loss_weights.data()), tokens * sizeof(u32_t));
        host_segments_t const segments = read_segments_(replica->batch);
        u32_t unmasked = 0;
        for (std::size_t segment = 0; segment != segments.lengths.size(); ++segment) {
            if (segments.modality[segment] != modality_t::text_k || segments.lengths[segment] == 0) continue;
            u32_t masked = 0;
            for (u32_t column = 0; column != segments.lengths[segment]; ++column)
                masked += weights[segments.offsets[segment] + column];
            unmasked += masked == 0 ? 1u : 0u;
        }
        suite.check(unmasked == 0, "epoch {}: {} of {} text rows carry no masked byte", epoch, unmasked,
                    segments.lengths.size());
    }
}

/** Each media head counts only masked cells of its own modality, and some of them, across four
 *  epochs. */
static void masked_media_rows(test_suite_t &suite, device_t &device) {
    switch_to_gpu_t scope {device.stream};
    trainer_config_t config = test_geometry_(true);
    config.seed = suite.settings.seed;
    expected<gpu_arena_t> arena = gpu_arena_t::make(0, case_arena_bytes_k);
    if (!arena) return suite.check(false, "arena: {}", arena.description_);
    expected<base_replica_t> replica = make_replica_(*arena, device, config, data_epoch_t {0});
    if (!replica) return suite.check(false, "replica: {}", replica.description_);
    for (u32_t epoch = 0; epoch != 4u; ++epoch) {
        expected_t drawn = epoch == 0 ? expected_t {} : redraw_(*replica, config, data_epoch_t {epoch}, device.stream);
        expected<f64_t> const loss = drawn ? forward_loss_(*replica, device.stream)
                                           : expected<f64_t>::fail(std::move(drawn));
        if (!loss) return suite.check(false, "epoch {}: {}", epoch, loss.description_);
        host_segments_t const segments = read_segments_(replica->batch);
        f64_t cells[4] = {0, 0, 0, 0};
        for (std::size_t segment = 0; segment != segments.lengths.size(); ++segment)
            cells[static_cast<u32_t>(segments.modality[segment])] += segments.lengths[segment];
        struct head_t {
            char const *name;
            modality_t modality;
            f32r1_tensor_arena_t const &summands;
        };
        head_t const heads[] = {{"image", modality_t::image_k, replica->image_summands},
                                {"video", modality_t::video_k, replica->video_summands},
                                {"audio", modality_t::audio_k, replica->audio_summands}};
        for (head_t const &head : heads) {
            f64_t const scored = read_pair_(head.summands).count;
            f64_t const live = cells[static_cast<u32_t>(head.modality)];
            suite.check(scored > 0 && scored <= live, "epoch {}: the {} head scored {} cells of {} {} cells", epoch,
                        head.name, scored, live, head.name);
        }
    }
}

/** A freshly seeded six-block tower predicts uniform random bytes at about the uniform
 *  cross-entropy: logits of unit variance add half a nat over @c ln(258) . */
static void chance_cross_entropy(test_suite_t &suite, device_t &device) {
    switch_to_gpu_t scope {device.stream};
    trainer_config_t config = test_geometry_(false);
    config.batch = multimodal_batch_shape_t {8u, 8u, 64u};
    config.depth = 6;
    config.hidden = 384;
    config.seed = suite.settings.seed;
    expected<gpu_arena_t> arena = gpu_arena_t::make(0, case_arena_bytes_k);
    if (!arena) return suite.check(false, "arena: {}", arena.description_);
    expected<base_replica_t> replica = make_replica_(*arena, device, config, data_epoch_t {0});
    if (!replica) return suite.check(false, "replica: {}", replica.description_);
    expected<f64_t> const loss = forward_loss_(*replica, device.stream);
    if (!loss) return suite.check(false, "forward: {}", loss.description_);
    f64_t const uniform = std::log(static_cast<f64_t>(byte_vocabulary_k));
    suite.check(*loss > uniform - 0.25 && *loss < uniform + 1.0,
                "the seeded tower starts at {:.4f} nats per masked byte, the uniform prediction scores {:.4f}", *loss,
                uniform);
}

/** One matrix and its gradient on the device beside their widened host copies, for the Adafactor
 *  cases. */
struct optimized_matrix_t {
    bf16r2_tensor_arena_t weights {}, gradient {};
    st::basic_vector<f64_t> host_weights {}, host_gradient {};
};

[[nodiscard]] static expected_t allocate_matrix_(optimized_matrix_t &matrix, gpu_arena_t &arena, std::size_t rows,
                                                 std::size_t columns) noexcept {
    gpu_arena_alloc<bf16_t> bf16_alloc {arena};
    matrix.weights = bf16r2_tensor_arena_t::uninitialized({rows, columns}, bf16_alloc).value;
    matrix.gradient = bf16r2_tensor_arena_t::uninitialized({rows, columns}, bf16_alloc).value;
    if (!matrix.weights || !matrix.gradient || st::failed(matrix.host_weights.resize(rows * columns)) ||
        st::failed(matrix.host_gradient.resize(rows * columns)))
        return expected_t::fail(status_t::bad_alloc_k, "matrix [{},{}] allocation failed", rows, columns);
    return {};
}

/** A uniform draw in @b [-1,1) keyed on the suite seed, the salt and the index. */
[[nodiscard]] static f32_t keyed_uniform_(u64_t seed, u32_t salt, std::size_t index) noexcept {
    u64_t const mixed = splitmix64_hash_t {}(seed, salt, index);
    return static_cast<f32_t>(mixed >> 40) * 0x1.0p-23f - 1.0f;
}

/** The distance between two BFloat16 neighbours around @p value . */
[[nodiscard]] static f64_t bf16_spacing_(f64_t value) noexcept {
    int exponent = 0;
    std::frexp(value, &exponent);
    return std::ldexp(1.0, exponent - 8);
}

/** Two Adafactor steps on a @b [24,40] matrix and a @b [1,40] vector against the paper's update
 *  replayed in Float64 on the host: factored second moments decaying at @c 1−t^−0.8 , the RMS clip,
 *  the first moment of the clipped update under @p beta1 , and the step. Each step restarts the
 *  host replay from the device's rounded weights, so the comparison allows one BFloat16 spacing. A
 *  kept first moment must also survive a save and a load into a fresh optimizer unchanged. */
static void adafactor_reference(test_suite_t &suite, device_t &device, f32_t beta1) {
    switch_to_gpu_t scope {device.stream};
    expected<gpu_arena_t> arena = gpu_arena_t::make(0, 64ull << 20);
    if (!arena) return suite.check(false, "arena: {}", arena.description_);
    std::size_t const shapes[2][2] = {{24, 40}, {1, 40}};
    optimized_matrix_t matrices[2];
    st::basic_vector<differentiable_weights_t> table;
    for (u32_t matrix = 0; matrix != 2u; ++matrix) {
        if (expected_t allocated = allocate_matrix_(matrices[matrix], *arena, shapes[matrix][0], shapes[matrix][1]);
            !allocated)
            return suite.check(false, "{}", allocated.description_);
        for (std::size_t index = 0; index != matrices[matrix].host_weights.size(); ++index)
            matrices[matrix].host_weights[index] = keyed_uniform_(suite.settings.seed, matrix, index);
        if (!upload_narrowed_(matrices[matrix].host_weights, matrices[matrix].weights.data()))
            return suite.check(false, "weights upload failed");
        differentiable_weights_t entry {{}, matrices[matrix].weights.span(), matrices[matrix].gradient.view()};
        if (sz::failed(entry.name.assign(matrix == 0 ? "matrix" : "vector")) ||
            st::failed(table.push_back(std::move(entry))))
            return suite.check(false, "weight table growth failed");
    }
    adafactor_config_t const config {.beta1 = beta1};
    expected<adafactor_t> optimizer = adafactor_t::make(*arena, table, config, device.stream);
    if (!optimizer) return suite.check(false, "optimizer: {}", optimizer.description_);

    f32_t const learning_rate = 1.0e-2f;
    st::basic_vector<f64_t> row_moments[2], column_moments[2], masters[2], first_moments[2];
    for (u32_t matrix = 0; matrix != 2u; ++matrix) {
        if (st::failed(row_moments[matrix].resize(shapes[matrix][0])) ||
            st::failed(column_moments[matrix].resize(shapes[matrix][1])) ||
            st::failed(masters[matrix].resize(matrices[matrix].host_weights.size())) ||
            st::failed(first_moments[matrix].resize(matrices[matrix].host_weights.size())))
            return suite.check(false, "host moments allocation failed");
        for (std::size_t index = 0; index != masters[matrix].size(); ++index)
            masters[matrix][index] = matrices[matrix].host_weights[index];
    }
    for (u32_t step = 1; step != 3u; ++step) {
        for (u32_t matrix = 0; matrix != 2u; ++matrix) {
            optimized_matrix_t &state = matrices[matrix];
            for (std::size_t index = 0; index != state.host_gradient.size(); ++index)
                state.host_gradient[index] = keyed_uniform_(suite.settings.seed, 16u * step + matrix + 2u, index);
            if (!upload_narrowed_(state.host_gradient, state.gradient.data()))
                return suite.check(false, "gradient upload failed");
        }
        if (expected_t stepped = optimizer->step(learning_rate, device.stream); !stepped)
            return suite.check(false, "step {}: {}", step, stepped.description_);
        cuStreamSynchronize(device.stream.get());
        f64_t const decay = std::min<f64_t>(config.beta2,
                                            1.0 - std::pow(static_cast<f64_t>(step), -config.decay_exponent));
        for (u32_t matrix = 0; matrix != 2u; ++matrix) {
            optimized_matrix_t &state = matrices[matrix];
            std::size_t const rows = shapes[matrix][0], columns = shapes[matrix][1];
            f64r2_view_t const gradient {state.host_gradient.data(), rows, columns};
            for (std::size_t row = 0; row != rows; ++row)
                row_moments[matrix][row] = decay * row_moments[matrix][row] +
                                           (1.0 - decay) *
                                               (dot(gradient.row(row), gradient.row(row)) / columns + config.epsilon1);
            for (std::size_t column = 0; column != columns; ++column) {
                f64_t energy = 0;
                for (std::size_t row = 0; row != rows; ++row) energy += gradient[row, column] * gradient[row, column];
                column_moments[matrix][column] = decay * column_moments[matrix][column] +
                                                 (1.0 - decay) * (energy / rows + config.epsilon1);
            }
            f64_t row_mean = 0, update_energy = 0;
            for (std::size_t row = 0; row != rows; ++row) row_mean += row_moments[matrix][row] / rows;
            auto update_at = [&](std::size_t row, std::size_t column) noexcept {
                return gradient[row, column] /
                       std::sqrt(row_moments[matrix][row] / row_mean * column_moments[matrix][column]);
            };
            for (std::size_t row = 0; row != rows; ++row)
                for (std::size_t column = 0; column != columns; ++column)
                    update_energy += update_at(row, column) * update_at(row, column);
            f64_t const clip = 1.0 / std::max(1.0, std::sqrt(update_energy / (rows * columns)) / config.clip_threshold);
            st::basic_vector<f64_t> device_weights;
            if (!download_widened_(state.weights.data(), rows * columns, device_weights))
                return suite.check(false, "weights readback failed");
            f64r2_view_t const got {device_weights.data(), rows, columns};
            f64r2_view_t const started {state.host_weights.data(), rows, columns};
            f64r2_span_t const master {masters[matrix].data(), rows, columns};
            f64r2_span_t const first_moment {first_moments[matrix].data(), rows, columns};
            u32_t off_grid = 0;
            f64_t worst = 0;
            for (std::size_t row = 0; row != rows; ++row)
                for (std::size_t column = 0; column != columns; ++column) {
                    f64_t const start = rows == 1 ? master[row, column] : started[row, column];
                    f64_t const clipped = clip * update_at(row, column);
                    first_moment[row, column] = beta1 * first_moment[row, column] + (1.0 - beta1) * clipped;
                    f64_t const wanted = start - learning_rate * (beta1 != 0.0f ? first_moment[row, column] : clipped);
                    f64_t const distance = std::fabs(got[row, column] - wanted) / bf16_spacing_(wanted);
                    worst = std::max(worst, distance);
                    off_grid += distance > 1.0 ? 1u : 0u;
                    master[row, column] = wanted;
                }
            suite.check(off_grid == 0,
                        "step {}: {} of {} {} scalars sit beyond one BFloat16 spacing of the host replay, " //
                        "the worst {:.2f} spacings",
                        step, off_grid, rows * columns, rows == 1 ? "vector" : "matrix", worst);
            std::memcpy(state.host_weights.data(), device_weights.data(), rows * columns * sizeof(f64_t));
        }
    }
    if (beta1 == 0.0f) return;

    // The first moment round-trips: saved, loaded into a fresh optimizer, and saved again bitwise.
    sz::string_t path;
    if (!format_into(path, "/tmp/uform-test-first-moment-{}.safetensors", ::getpid()))
        return suite.check(false, "checkpoint path formatting failed");
    checkpoint_to_save_t saved;
    if (expected_t written = optimizer->save(saved, device.stream); !written)
        return suite.check(false, "save: {}", written.description_);
    if (expected_t written = saved.save(path.view()); !written)
        return suite.check(false, "save to {}: {}", path.view(), written.description_);
    expected<checkpoint_to_load_t> loaded = checkpoint_to_load_t::make(path.view());
    if (!loaded) return suite.check(false, "load: {}", loaded.description_);
    expected<adafactor_t> resumed = adafactor_t::make(*arena, table, config, device.stream);
    if (!resumed) return suite.check(false, "resumed optimizer: {}", resumed.description_);
    if (expected_t restored = resumed->load(*loaded, device.stream); !restored)
        return suite.check(false, "restore: {}", restored.description_);
    checkpoint_to_save_t resaved;
    if (expected_t written = resumed->save(resaved, device.stream); !written)
        return suite.check(false, "resave: {}", written.description_);
    std::remove(path.c_str());
    if (expected_t written = resaved.save(path.view()); !written)
        return suite.check(false, "resave to {}: {}", path.view(), written.description_);
    expected<checkpoint_to_load_t> reloaded = checkpoint_to_load_t::make(path.view());
    std::remove(path.c_str());
    if (!reloaded) return suite.check(false, "reload: {}", reloaded.description_);
    for (char const *name : {"optimizer.matrix.first_moment", "optimizer.vector.first_moment"}) {
        nk::tensor_view<f32_t, 2> const before = loaded->float32(name), after = reloaded->float32(name);
        bool const same = !before.empty() && before.extent(1) == after.extent(1) &&
                          std::memcmp(before.data(), after.data(), before.extent(1) * sizeof(f32_t)) == 0;
        suite.check(same, "{} survives a save and a load", name);
    }
}

/** A @b [64,256] matrix of ones under a unit gradient takes one step of @c 2^−10 , a quarter of
 *  the BFloat16 spacing below one: the weights must move by that step on average instead of
 *  rounding back to one. */
static void adafactor_small_steps(test_suite_t &suite, device_t &device) {
    switch_to_gpu_t scope {device.stream};
    expected<gpu_arena_t> arena = gpu_arena_t::make(0, 64ull << 20);
    if (!arena) return suite.check(false, "arena: {}", arena.description_);
    std::size_t const rows = 64, columns = 256;
    optimized_matrix_t matrix;
    if (expected_t allocated = allocate_matrix_(matrix, *arena, rows, columns); !allocated)
        return suite.check(false, "{}", allocated.description_);
    for (std::size_t index = 0; index != rows * columns; ++index)
        matrix.host_weights[index] = 1.0, matrix.host_gradient[index] = 1.0;
    if (!upload_narrowed_(matrix.host_weights, matrix.weights.data()) ||
        !upload_narrowed_(matrix.host_gradient, matrix.gradient.data()))
        return suite.check(false, "upload failed");
    st::basic_vector<differentiable_weights_t> table;
    differentiable_weights_t entry {{}, matrix.weights.span(), matrix.gradient.view()};
    if (sz::failed(entry.name.assign("ones")) || st::failed(table.push_back(std::move(entry))))
        return suite.check(false, "weight table growth failed");
    expected<adafactor_t> optimizer = adafactor_t::make(*arena, table, adafactor_config_t {}, device.stream);
    if (!optimizer) return suite.check(false, "optimizer: {}", optimizer.description_);
    f64_t const learning_rate = 0x1.0p-10;
    if (expected_t stepped = optimizer->step(static_cast<f32_t>(learning_rate), device.stream); !stepped)
        return suite.check(false, "step: {}", stepped.description_);
    cuStreamSynchronize(device.stream.get());
    if (!download_widened_(matrix.weights.data(), rows * columns, matrix.host_weights))
        return suite.check(false, "weights readback failed");
    f64_t const total = nk::reduce_moments<nk::f64_t>(matrix.host_weights).value.sum;
    f64_t const mean_step = 1.0 - total / static_cast<f64_t>(rows * columns);
    // Each scalar moves by 2^-8 with probability 1/4, so the mean of 16384 carries a 1.3e-5
    // standard error.
    suite.check(std::fabs(mean_step - learning_rate) < 0.1 * learning_rate,
                "the mean step is {:.4g}, the learning rate {:.4g}", mean_step, learning_rate);
}

/** Unit RMS normalization of the run @p input into @p output , the forward's own epsilon. */
static void normalize_rms_(auto const &input, auto const &output, f64_t epsilon) noexcept {
    std::size_t const count = input.size();
    f64_t const inverse = 1.0 / std::sqrt(dot(input, input) / static_cast<f64_t>(count) + epsilon);
    for (std::size_t index = 0; index != count; ++index) output[index] = input[index] * inverse;
}

/** The base the block replay's rotary angle grid is built over. */
inline constexpr f64_t rope_base_k = 10000.0;

/** The angle of rotary pair @p pair at a token's @c (t,h,w) @p coordinates , as the block's grid
 *  bakes it: the temporal axis takes the first pairs at the whole head's frequency spacing, height
 *  and width the rest at their own. */
[[nodiscard]] static f64_t rope_angle_(std::size_t pair, nk::tensor_view<i16_t, 1> coordinates) noexcept {
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
static void rotate_(auto &&values, nk::tensor_view<i16_t, 1> coordinates, f64_t sign) noexcept {
    std::size_t const pairs = transformer_head_dim_k / 2;
    for (std::size_t pair = 0; pair != pairs; ++pair) {
        f64_t const angle = sign * rope_angle_(pair, coordinates);
        f64_t const real = values[pair], imaginary = values[pair + pairs];
        values[pair] = real * std::cos(angle) - imaginary * std::sin(angle);
        values[pair + pairs] = real * std::sin(angle) + imaginary * std::cos(angle);
    }
}

/** One block over four segments in 64-token windows, every other one 40 tokens long, with two heads
 *  and M-RoPE coordinates on all three axes: its forward and its attention backward against a
 *  Float64 host replay, the output of every token, and the dQ, dK and dV every segment's softmax
 *  owes back through the inverse rotation and the QK-norm, zero past each segment's length. */
static void block_host_replay(test_suite_t &suite, device_t &device) {
    switch_to_gpu_t scope {device.stream};
    std::size_t const hidden = 256, sequence = 64, segments = 4, tokens = sequence * segments;
    std::size_t const heads = hidden / transformer_head_dim_k, ffn = 2 * hidden;
    expected<gpu_arena_t> arena = gpu_arena_t::make(0, case_arena_bytes_k);
    if (!arena) return suite.check(false, "arena: {}", arena.description_);
    transformer_block_config_t config;
    config.hidden = hidden;
    config.ffn_expansion = 2;
    config.max_tokens = tokens;
    config.max_segments = segments;
    config.max_segment_tokens = sequence;
    expected<transformer_scratch_t> scratch = transformer_scratch_t::make(*arena, config, {}, 0);
    if (!scratch) return suite.check(false, "scratch: {}", scratch.description_);
    expected<transformer_block_bf16_t> block = transformer_block_bf16_t::make(*arena, *scratch, config);
    if (!block) return suite.check(false, "block: {}", block.description_);
    expected<rope_angles_t> angles = rope_angles_t::make(*arena, sequence, rope_shape_t {},
                                                         static_cast<f32_t>(rope_base_k));
    if (!angles) return suite.check(false, "angles: {}", angles.description_);

    // Weights of unit-variance outputs, unit norm scales, and unit-variance inputs and output
    // gradients.
    struct seeded_t {
        bf16r1_span_t weights;
        std::size_t fan_in;
    };
    seeded_t const seeded[] = {{block->qkv_weight(), hidden},     {block->out_weight(), hidden},
                               {block->gate_up_weight(), hidden}, {block->down_weight(), ffn},
                               {block->pre_norm_scale(), 0},      {block->post_norm_scale(), 0}};
    st::basic_vector<f64_t> weights[6];
    for (u32_t matrix = 0; matrix != 6u; ++matrix) {
        if (st::failed(weights[matrix].resize(seeded[matrix].weights.extent(0))))
            return suite.check(false, "staging failed");
        f64_t const deviation = seeded[matrix].fan_in ? 1.0 / std::sqrt(f64_t(seeded[matrix].fan_in)) : 0.0;
        for (std::size_t index = 0; index != weights[matrix].size(); ++index)
            weights[matrix][index] =
                deviation ? keyed_uniform_(suite.settings.seed, 32u + matrix, index) * 1.7320508 * deviation : 1.0;
        if (!upload_narrowed_(weights[matrix], seeded[matrix].weights.data()))
            return suite.check(false, "weights upload failed");
    }
    st::basic_vector<f32_t> input, output_grad, output;
    st::basic_vector<f64_t> widened_input;
    st::basic_vector<u32_t> offsets, lengths;
    st::basic_vector<i16_t> coordinates;
    if (st::failed(input.resize(tokens * hidden)) || st::failed(widened_input.resize(tokens * hidden)) ||
        st::failed(output_grad.resize(tokens * hidden)) || st::failed(output.resize(tokens * hidden)) ||
        st::failed(offsets.resize(segments + 1)) || st::failed(lengths.resize(segments)) ||
        st::failed(coordinates.resize(tokens * 3)))
        return suite.check(false, "host staging failed");
    for (std::size_t index = 0; index != input.size(); ++index)
        input[index] = keyed_uniform_(suite.settings.seed, 40u, index) * 1.7320508f,
        output_grad[index] = keyed_uniform_(suite.settings.seed, 41u, index);
    for (std::size_t segment = 0; segment != segments; ++segment)
        offsets[segment] = static_cast<u32_t>(segment * sequence),
        lengths[segment] = static_cast<u32_t>(segment % 2 ? sequence * 5 / 8 : sequence);
    offsets[segments] = static_cast<u32_t>(tokens);
    // Distinct coordinates on every axis, so each rotary band of every head turns.
    nk::tensor_span<i16_t, 2> const token_coordinates {coordinates.data(), tokens, 3};
    for (std::size_t token = 0; token != tokens; ++token) {
        std::size_t const position = token % sequence;
        token_coordinates[token, 0] = static_cast<i16_t>(position % 7);
        token_coordinates[token, 1] = static_cast<i16_t>(position / 7 % 5);
        token_coordinates[token, 2] = static_cast<i16_t>(position % 3);
    }
    if (nk::cast<nk::f32_t, nk::f64_t>(input, widened_input) != nk::status_t::success_k)
        return suite.check(false, "input widening failed");
    f32_t *const device_input = arena->allocate_object<f32_t>(tokens * hidden);
    f32_t *const device_output = arena->allocate_object<f32_t>(tokens * hidden);
    f32_t *const device_output_grad = arena->allocate_object<f32_t>(tokens * hidden);
    f32_t *const device_input_grad = arena->allocate_object<f32_t>(tokens * hidden);
    u32_t *const device_offsets = arena->allocate_object<u32_t>(segments + 1);
    u32_t *const device_lengths = arena->allocate_object<u32_t>(segments);
    i16_t *const device_coordinates = arena->allocate_object<i16_t>(tokens * 3);
    cuMemcpyHtoD(device_address(device_input), input.data(), input.size() * sizeof(f32_t));
    cuMemcpyHtoD(device_address(device_output_grad), output_grad.data(), output_grad.size() * sizeof(f32_t));
    cuMemcpyHtoD(device_address(device_offsets), offsets.data(), offsets.size() * sizeof(u32_t));
    cuMemcpyHtoD(device_address(device_lengths), lengths.data(), lengths.size() * sizeof(u32_t));
    cuMemcpyHtoD(device_address(device_coordinates), coordinates.data(), coordinates.size() * sizeof(i16_t));
    cuCtxSynchronize(); // the pageable copies land before the non-blocking stream reads them
    u32r1_view_t const segment_offsets {device_offsets, segments + 1}, segment_lengths {device_lengths, segments};
    i16r2_view_t const coordinate_view {device_coordinates, tokens, 3};
    expected_t staged = block->pack_weights(device.stream);
    if (staged)
        staged = block->forward(f32r2_view_t {device_input, tokens, hidden},
                                f32r2_span_t {device_output, tokens, hidden}, segment_offsets, segment_lengths,
                                angles->view(), coordinate_view, device.stream);
    if (staged)
        staged = block->backward(f32r2_view_t {device_output_grad, tokens, hidden},
                                 f32r2_span_t {device_input_grad, tokens, hidden}, segment_offsets, segment_lengths,
                                 angles->view(), coordinate_view, device.stream);
    if (!staged) return suite.check(false, "block step: {}", staged.description_);
    cuStreamSynchronize(device.stream.get());
    cuMemcpyDtoH(output.data(), device_address(device_output), output.size() * sizeof(f32_t));
    st::basic_vector<f64_t> narrowed_grad, packed_grad;
    if (!download_widened_(scratch->narrowed_grad.data(), tokens * hidden, narrowed_grad) ||
        !download_widened_(scratch->packed_qkv.data(), tokens * 3 * hidden, packed_grad))
        return suite.check(false, "gradients readback failed");
    f64r2_view_t const qkv_weights {weights[0].data(), 3 * hidden, hidden};
    f64r2_view_t const out_weights {weights[1].data(), hidden, hidden};
    f64r2_view_t const gate_up_weights {weights[2].data(), 2 * ffn, hidden};
    f64r2_view_t const down_weights {weights[3].data(), hidden, ffn};
    f64r2_view_t const inputs {widened_input.data(), tokens, hidden}, narrowed {narrowed_grad.data(), tokens, hidden};
    f32r2_view_t const outputs {output.data(), tokens, hidden};
    f64r4_view_t const packed {packed_grad.data(), {tokens, 3, heads, transformer_head_dim_k}};

    // The attention's output gradient, from the narrowed out-projection gradient the device
    // multiplied by W_out.
    st::basic_vector<f64_t> attention_output_grad;
    if (st::failed(attention_output_grad.resize(tokens * hidden))) return suite.check(false, "staging failed");
    f64r2_span_t const output_grad_rows {attention_output_grad.data(), tokens, hidden};
    f64r3_view_t const output_grads {attention_output_grad.data(), {tokens, heads, transformer_head_dim_k}};
    for (std::size_t token = 0; token != tokens; ++token)
        for (std::size_t channel = 0; channel != hidden; ++channel) {
            f64_t sum = 0;
            for (std::size_t inner = 0; inner != hidden; ++inner)
                sum += narrowed[token, inner] * out_weights[inner, channel];
            output_grad_rows[token, channel] = sum;
        }

    // The host forward: pre-norm, QKV, the QK-norm, attention per segment and head, then the MLP
    // half.
    st::basic_vector<f64_t> products, normalized, attended, normalized_row, branch, mlp_input, gate_up, activated;
    if (st::failed(products.resize(tokens * 3 * hidden)) || st::failed(normalized.resize(tokens * 3 * hidden)) ||
        st::failed(attended.resize(tokens * hidden)) || st::failed(normalized_row.resize(hidden)) ||
        st::failed(branch.resize(hidden)) || st::failed(mlp_input.resize(hidden)) ||
        st::failed(gate_up.resize(2 * ffn)) || st::failed(activated.resize(ffn)))
        return suite.check(false, "host replay staging failed");
    f64r2_span_t const product_rows {products.data(), tokens, 3 * hidden};
    f64r2_span_t const normalized_rows {normalized.data(), tokens, 3 * hidden};
    f64r4_span_t const pre_norm {products.data(), {tokens, 3, heads, transformer_head_dim_k}};
    f64r4_span_t const qkv {normalized.data(), {tokens, 3, heads, transformer_head_dim_k}};
    for (std::size_t token = 0; token != tokens; ++token) {
        normalize_rms_(inputs.row(token), std::span<f64_t> {normalized_row}, config.rms_eps);
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
    f64_t const scale = 1.0 / std::sqrt(static_cast<f64_t>(transformer_head_dim_k));
    st::basic_vector<f64_t> probability_cells, scores_grad_cells;
    if (st::failed(probability_cells.resize(sequence * sequence)) ||
        st::failed(scores_grad_cells.resize(sequence * sequence)))
        return suite.check(false, "host replay staging failed");
    f64r2_span_t const probabilities {probability_cells.data(), sequence, sequence};
    f64r2_span_t const scores_grad {scores_grad_cells.data(), sequence, sequence};
    f64r3_span_t const attended_heads {attended.data(), {tokens, heads, transformer_head_dim_k}};
    f64_t worst_grad[3] = {0, 0, 0}, largest_grad[3] = {0, 0, 0};
    for (std::size_t segment = 0; segment != segments; ++segment)
        for (std::size_t head = 0; head != heads; ++head) {
            std::size_t const first = segment * sequence, length = lengths[segment];
            for (std::size_t query = 0; query != length; ++query) {
                f64_t largest = -1.0e300, total = 0;
                for (std::size_t key = 0; key != length; ++key) {
                    probabilities[query, key] =
                        dot(qkv[first + query, 0, head, nk::slice], qkv[first + key, 1, head, nk::slice]) * scale;
                    largest = std::max(largest, probabilities[query, key]);
                }
                for (std::size_t key = 0; key != length; ++key)
                    total += probabilities[query, key] = std::exp(probabilities[query, key] - largest);
                for (std::size_t key = 0; key != length; ++key) probabilities[query, key] /= total;
                for (std::size_t channel = 0; channel != transformer_head_dim_k; ++channel) {
                    f64_t value = 0;
                    for (std::size_t key = 0; key != length; ++key)
                        value += probabilities[query, key] * qkv[first + key, 2, head, channel];
                    attended_heads[first + query, head, channel] = value;
                }
            }
            // The backward from the device's own attention-output gradient: dV, dS, then dQ and dK
            // through the QK-norm.
            for (std::size_t query = 0; query != length; ++query) {
                auto const output_grad = output_grads[first + query, head, nk::slice];
                f64_t const output_dot = dot(output_grad, attended_heads[first + query, head, nk::slice]);
                for (std::size_t key = 0; key != length; ++key) {
                    f64_t const probability_grad = dot(output_grad, qkv[first + key, 2, head, nk::slice]);
                    scores_grad[query, key] = probabilities[query, key] * (probability_grad - output_dot) * scale;
                }
            }
            for (std::size_t role = 0; role != 3; ++role)
                for (std::size_t position = 0; position != sequence; ++position) {
                    f64_t normalized_grad[transformer_head_dim_k];
                    for (std::size_t channel = 0; channel != transformer_head_dim_k; ++channel) {
                        f64_t sum = 0;
                        for (std::size_t other = 0; other != length; ++other)
                            sum += role == 0 ? scores_grad[position, other] * qkv[first + other, 1, head, channel]
                                   : role == 1
                                       ? scores_grad[other, position] * qkv[first + other, 0, head, channel]
                                       : probabilities[other, position] * output_grads[first + other, head, channel];
                        normalized_grad[channel] = position < length ? sum : 0.0;
                    }
                    if (role != 2) rotate_(normalized_grad, token_coordinates[first + position, nk::slice], -1.0);
                    auto const pre = pre_norm[first + position, role, head, nk::slice];
                    f64_t const inverse =
                        role == 2 ? 1.0 : 1.0 / std::sqrt(dot(pre, pre) / transformer_head_dim_k + config.rms_eps);
                    f64_t const projection = dot(normalized_grad, pre) * inverse / transformer_head_dim_k;
                    for (std::size_t channel = 0; channel != transformer_head_dim_k; ++channel) {
                        f64_t const wanted = role == 2
                                                 ? normalized_grad[channel]
                                                 : (normalized_grad[channel] - pre[channel] * inverse * projection) *
                                                       inverse;
                        f64_t const got = packed[first + position, role, head, channel];
                        worst_grad[role] = std::max(worst_grad[role], std::fabs(got - wanted));
                        largest_grad[role] = std::max(largest_grad[role], std::fabs(wanted));
                    }
                }
        }
    f64r2_view_t const attended_rows {attended.data(), tokens, hidden};
    f64_t worst_output = 0, largest_output = 0;
    for (std::size_t token = 0; token != tokens; ++token) {
        for (std::size_t channel = 0; channel != hidden; ++channel)
            branch[channel] = inputs[token, channel] + dot(out_weights.row(channel), attended_rows.row(token));
        normalize_rms_(branch, std::span<f64_t> {mlp_input}, config.rms_eps);
        for (std::size_t output_index = 0; output_index != 2 * ffn; ++output_index)
            gate_up[output_index] = dot(gate_up_weights.row(output_index), mlp_input);
        for (std::size_t feature = 0; feature != ffn; ++feature)
            activated[feature] = gate_up[feature] / (1.0 + std::exp(-gate_up[feature])) * gate_up[ffn + feature];
        for (std::size_t channel = 0; channel != hidden; ++channel) {
            f64_t const wanted = branch[channel] + dot(down_weights.row(channel), activated);
            worst_output = std::max(worst_output, std::fabs(wanted - outputs[token, channel]));
            largest_output = std::max(largest_output, std::fabs(wanted));
        }
    }
    // BFloat16 activations hold about three decimal digits, so two percent of the largest value
    // bounds every cell.
    suite.check(worst_output <= 0.02 * largest_output, "forward: off by {:.4g} of {:.4g}", worst_output,
                largest_output);
    char const *const roles[] = {"dQ", "dK", "dV"};
    for (u32_t role = 0; role != 3u; ++role)
        suite.check(worst_grad[role] <= 0.02 * largest_grad[role], "{}: off by {:.4g} of {:.4g} over {} segments",
                    roles[role], worst_grad[role], largest_grad[role], segments);
}

int main(int, char **argv) {
    ::setenv("CUDA_MODULE_LOADING", "EAGER", 1);
    test_suite_t suite = test_suite_t::make(argv[0]);
    int device_count = 0;
    char const *const cases[] = {"gradients/tokens",    "gradients/omni",         "gradients/token_head",
                                 "masking/text_rows",   "masking/media_rows",     "init/chance_entropy",
                                 "adafactor/reference", "adafactor/first_moment", "adafactor/small_steps",
                                 "block/host_replay"};
    if (cuInit(0) != CUDA_SUCCESS || cuDeviceGetCount(&device_count) != CUDA_SUCCESS || device_count <= 0) {
        for (char const *name : cases) suite.skip(name, no_cuda_device_k);
        return suite.finish();
    }
    expected<gpu_stream_t> stream = gpu_stream_t::make(0);
    if (!stream) {
        suite.check(false, "stream: {}", stream.description_);
        return suite.finish();
    }
    switch_to_gpu_t scope {*stream};
    device_t device {std::move(*stream)};
    suite.run("gradients/tokens", finite_differences_tokens, device);
    suite.run("gradients/omni", finite_differences_omni, device);
    suite.run("gradients/token_head", token_head_finite_differences, device);
    suite.run("masking/text_rows", masked_text_rows, device);
    suite.run("masking/media_rows", masked_media_rows, device);
    suite.run("init/chance_entropy", chance_cross_entropy, device);
    suite.run("adafactor/reference", adafactor_reference, device, 0.0f);
    suite.run("adafactor/first_moment", adafactor_reference, device, 0.9f);
    suite.run("adafactor/small_steps", adafactor_small_steps, device);
    suite.run("block/host_replay", block_host_replay, device);
    return suite.finish();
}
