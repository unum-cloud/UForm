/**
 *  @file test/vram.c
 *  @author Ash Vardanian
 *  @date October 18, 2026
 *  @brief The C interface on one NVIDIA GPU against the same checkpoint on the host: text, pixels and samples,
 *      blocking, deferred and split into slices, each landing within the BFloat16 floor of the RAM outputs.
 *
 *  The outputs of a VRAM call stay on the device, so only this test copies them back, with the CUDA driver API.
 *  The checkpoint comes from @c argv[1], as written by @c test/tiny_checkpoint.cpp.
 */
#include <math.h>    // `fabsf`
#include <stdbool.h> // `bool`
#include <stdint.h>  // `uint8_t`, `uint32_t`
#include <stdio.h>   // `printf`
#include <stdlib.h>  // `exit`, `malloc`, `free`
#include <string.h>  // `memset`, `strlen`

#include <cuda.h> // `cuMemcpyDtoH`, the primary context the executor's streams ride

#include "uform/uform.h"

/** The largest relative difference a GPU output may show against the host's, from BFloat16 rounding alone. */
static float const tolerance = 0.05f;

static void expect(bool must_be_true, char const *message) {
    if (must_be_true) return;
    printf("Assert: %s\n", message);
    exit(-1);
}

static void expect_success(uform_outcome_t const *outcome, char const *what) {
    if (outcome->status == uform_status_success_k) return;
    printf("Assert: %s: %s %s\n", what, uform_status_name(outcome->status), outcome->message ? outcome->message : "");
    exit(-1);
}

/** The verbs this test drives, all over decoded inputs. */
typedef enum verb_t { verb_text_k, verb_pixels_k, verb_samples_k } verb_t;

static char const *const verb_names[] = {"text", "pixels", "samples"};

enum { items_count = 4, image_bytes = 17 * 9 * 3 + 40 * 24 * 3 + 64 * 64 * 3 + 5 * 3 * 3 };
enum { samples_count = 1000 + 3000 + 256 + 7 };

/** Every verb's four items, laid as the tapes take them. */
typedef struct inputs_t {
    char texts[700];
    size_t text_offsets[items_count + 1];
    uint8_t pixels[image_bytes];
    size_t pixel_offsets[items_count + 1];
    uint32_t heights[items_count], widths[items_count];
    float samples[samples_count];
    size_t sample_offsets[items_count + 1];
} inputs_t;

static void fill_inputs(inputs_t *inputs) {
    char const *const texts[items_count] = {"hello", "", NULL, "the quick brown fox jumps over the lazy dog"};
    size_t cursor = 0;
    for (size_t item = 0; item != items_count; ++item) {
        inputs->text_offsets[item] = cursor;
        if (texts[item])
            memcpy(inputs->texts + cursor, texts[item], strlen(texts[item])), cursor += strlen(texts[item]);
        else memset(inputs->texts + cursor, 'a', 600), cursor += 600;
    }
    inputs->text_offsets[items_count] = cursor;

    uint32_t const heights[items_count] = {9, 24, 64, 3}, widths[items_count] = {17, 40, 64, 5};
    cursor = 0;
    for (size_t item = 0; item != items_count; ++item) {
        inputs->heights[item] = heights[item], inputs->widths[item] = widths[item];
        inputs->pixel_offsets[item] = cursor;
        cursor += (size_t)heights[item] * widths[item] * 3;
    }
    inputs->pixel_offsets[items_count] = cursor;
    for (size_t byte = 0; byte != image_bytes; ++byte) inputs->pixels[byte] = (uint8_t)(byte * 31 + byte / 97);

    size_t const lengths[items_count] = {1000, 3000, 256, 7};
    cursor = 0;
    for (size_t item = 0; item != items_count; ++item) inputs->sample_offsets[item] = cursor, cursor += lengths[item];
    inputs->sample_offsets[items_count] = cursor;
    for (size_t sample = 0; sample != samples_count; ++sample)
        inputs->samples[sample] = (float)((sample * 37) % 101) / 101.0f - 0.5f;
}

/** One call's outputs as it lands them, in host or device memory. */
typedef struct landed_t {
    uform_outcome_t outcome;
    uform_rows_view_t embeddings;
    uform_rows_tape_view_t states;
    size_t tokens;
} landed_t;

/** Runs @p verb over @p inputs into @p landed, waiting for a deferred call before returning. */
static void encode(verb_t verb, inputs_t const *inputs, uform_model_t model, uform_executor_t executor,
                   uform_arena_t arena, uform_completion_t completion, uform_progress_t progress, landed_t *landed) {
    memset(landed, 0, sizeof(*landed));
    uform_text_encode_stats_t text_stats = {0};
    uform_image_encode_stats_t image_stats = {0};
    uform_audio_encode_stats_t audio_stats = {0};
    switch (verb) {
    case verb_text_k:
        uform_text_encode(&(uform_text_encode_t) {.model = model,
                                                  .executor = executor,
                                                  .outcome = &landed->outcome,
                                                  .completion = completion,
                                                  .count = items_count,
                                                  .texts = {inputs->texts, inputs->text_offsets, uform_scalar_u8_k},
                                                  .embeddings = &landed->embeddings,
                                                  .states = &landed->states,
                                                  .arena = arena,
                                                  .outputs_scalar = uform_scalar_f32_k,
                                                  .progress = progress,
                                                  .stats = &text_stats});
        break;
    case verb_pixels_k:
        uform_image_encode_pixels(
            &(uform_image_encode_pixels_t) {.model = model,
                                            .executor = executor,
                                            .outcome = &landed->outcome,
                                            .completion = completion,
                                            .count = items_count,
                                            .pixels = {inputs->pixels, inputs->pixel_offsets, uform_scalar_u8_k},
                                            .heights = inputs->heights,
                                            .widths = inputs->widths,
                                            .embeddings = &landed->embeddings,
                                            .states = &landed->states,
                                            .arena = arena,
                                            .outputs_scalar = uform_scalar_f32_k,
                                            .progress = progress,
                                            .stats = &image_stats});
        break;
    case verb_samples_k:
        uform_audio_encode_samples(
            &(uform_audio_encode_samples_t) {.model = model,
                                             .executor = executor,
                                             .outcome = &landed->outcome,
                                             .completion = completion,
                                             .count = items_count,
                                             .samples = {inputs->samples, inputs->sample_offsets, uform_scalar_f32_k},
                                             .sample_rate = 16000,
                                             .embeddings = &landed->embeddings,
                                             .states = &landed->states,
                                             .arena = arena,
                                             .outputs_scalar = uform_scalar_f32_k,
                                             .progress = progress,
                                             .stats = &audio_stats});
        break;
    }
    if (completion == uform_completion_deferred_k) uform_outcome_wait(&landed->outcome);
    landed->tokens = text_stats.tokens + image_stats.tokens + audio_stats.tokens;
}

/** One call's outputs copied into host memory. */
typedef struct fetched_t {
    float *embeddings;
    float *states;
    size_t offsets[items_count + 1];
    size_t tokens;
} fetched_t;

static void copy_out(void *target, void const *source, size_t bytes, bool on_device) {
    if (!bytes) return;
    if (!on_device) return (void)memcpy(target, source, bytes);
    CUresult const copied = cuMemcpyDtoH(target, (CUdeviceptr)source, bytes);
    expect(copied == CUDA_SUCCESS, "Copying device outputs back failed");
}

static fetched_t fetch(landed_t const *landed, size_t dimensions, bool on_device) {
    fetched_t fetched = {0};
    fetched.tokens = landed->tokens;
    fetched.embeddings = malloc(items_count * dimensions * sizeof(float));
    fetched.states = malloc((landed->tokens ? landed->tokens : 1) * dimensions * sizeof(float));
    expect(fetched.embeddings && fetched.states, "Host copies of the outputs");
    expect(landed->embeddings.stride == dimensions * sizeof(float), "Rows are packed f32");
    copy_out(fetched.embeddings, landed->embeddings.data, items_count * dimensions * sizeof(float), on_device);
    copy_out(fetched.states, landed->states.rows.data, landed->tokens * dimensions * sizeof(float), on_device);
    copy_out(fetched.offsets, landed->states.offsets, sizeof(fetched.offsets), on_device);
    return fetched;
}

/** The largest over rows of the row's largest absolute difference relative to the reference row's largest value. */
static float relative_error(float const *rows, float const *reference, size_t count, size_t dimensions) {
    float worst = 0.0f;
    for (size_t row = 0; row != count; ++row) {
        float difference = 0.0f, scale = 0.0f;
        for (size_t column = 0; column != dimensions; ++column) {
            float const expected = reference[row * dimensions + column];
            float const delta = fabsf(rows[row * dimensions + column] - expected);
            difference = delta > difference ? delta : difference;
            scale = fabsf(expected) > scale ? fabsf(expected) : scale;
        }
        float const relative = scale > 0.0f ? difference / scale : difference;
        worst = relative > worst ? relative : worst;
    }
    return worst;
}

static uform_continuation_t carry_on(size_t completed, size_t total, void *state) {
    (void)completed, (void)total;
    ++*(size_t *)state;
    return uform_continue_k;
}

int main(int argc, char *argv[]) {
    expect(argc > 1, "Pass the tiny checkpoint's path");
    uform_bytes_view_t const path = {(uint8_t const *)argv[1], strlen(argv[1])};
    int32_t const device = 0;
    uform_outcome_t outcome = {0};

    uform_executor_t host = NULL, gpu = NULL;
    uform_executor_for_ram(&(uform_executor_for_ram_t) {.executor = &host, .outcome = &outcome, .threads = 2});
    expect_success(&outcome, "RAM executor");
    uform_executor_for_vram(&(
        uform_executor_for_vram_t) {.executor = &gpu, .outcome = &outcome, .vram = {&device, 1}, .deferred_limit = 4});
    expect_success(&outcome, "VRAM executor");

    uform_model_t host_model = NULL, gpu_model = NULL;
    uform_model_load_ram(&(uform_model_load_ram_t) {.model = &host_model,
                                                    .executor = host,
                                                    .outcome = &outcome,
                                                    .path = path,
                                                    .quantization = uform_scalar_bf16_k});
    expect_success(&outcome, "RAM load");
    uform_model_load_vram(&(uform_model_load_vram_t) {.model = &gpu_model,
                                                      .executor = gpu,
                                                      .outcome = &outcome,
                                                      .path = path,
                                                      .vram = {&device, 1},
                                                      .quantization = uform_scalar_bf16_k});
    expect_success(&outcome, "VRAM load");
    uform_model_description_t description = {0};
    uform_model_describe(&(uform_model_describe_t) {.model = gpu_model,
                                                    .executor = gpu,
                                                    .outcome = &outcome,
                                                    .description = &description});
    expect_success(&outcome, "Describe");
    size_t const dimensions = description.dimensions;
    printf("VRAM model: %zu dimensions, %zu context, %s\n", dimensions, description.context, description.capabilities);

    uform_arena_t host_arena = NULL, gpu_arena = NULL;
    uform_arena_init(&(uform_arena_init_t) {.arena = &host_arena, .executor = host, .outcome = &outcome});
    expect_success(&outcome, "Host arena");
    uform_arena_init(&(uform_arena_init_t) {.arena = &gpu_arena, .executor = gpu, .outcome = &outcome});
    expect_success(&outcome, "Device arena");

    CUcontext context = NULL;
    CUdevice handle = 0;
    expect(cuInit(0) == CUDA_SUCCESS && cuDeviceGet(&handle, device) == CUDA_SUCCESS, "CUDA driver");
    expect(cuDevicePrimaryCtxRetain(&context, handle) == CUDA_SUCCESS, "Primary context");
    expect(cuCtxSetCurrent(context) == CUDA_SUCCESS, "Current context");

    static inputs_t inputs;
    fill_inputs(&inputs);
    float worst = 0.0f;
    for (verb_t verb = verb_text_k; verb <= verb_samples_k; ++verb) {
        landed_t landed;
        uform_progress_t const quiet = {0};
        encode(verb, &inputs, host_model, host, host_arena, uform_completion_blocking_k, quiet, &landed);
        expect_success(&landed.outcome, "RAM encode");
        fetched_t const reference = fetch(&landed, dimensions, false);

        char const *const runs[] = {"blocking", "deferred", "two slices"};
        for (size_t run = 0; run != 3; ++run) {
            size_t reports = 0;
            uform_progress_t const progress = {run == 2 ? carry_on : NULL, &reports, 2};
            uform_completion_t const completion = run == 1 ? uform_completion_deferred_k : uform_completion_blocking_k;
            encode(verb, &inputs, gpu_model, gpu, gpu_arena, completion, progress, &landed);
            expect_success(&landed.outcome, "VRAM encode");
            expect(landed.tokens == reference.tokens, "Both residencies keep the same tokens");
            fetched_t const fetched = fetch(&landed, dimensions, true);
            expect(memcmp(fetched.offsets, reference.offsets, sizeof(fetched.offsets)) == 0, "Offsets agree");
            expect(run != 2 || reports == 2, "Two slices report twice");
            float const embeddings = relative_error(fetched.embeddings, reference.embeddings, items_count, dimensions);
            float const states = relative_error(fetched.states, reference.states, reference.tokens, dimensions);
            printf("%-8s %-10s tokens %4zu: max relative error embeddings %.5f, states %.5f\n", verb_names[verb],
                   runs[run], fetched.tokens, embeddings, states);
            worst = embeddings > worst ? embeddings : worst;
            worst = states > worst ? states : worst;
            free(fetched.embeddings), free(fetched.states);
        }
        free(reference.embeddings), free(reference.states);
    }
    expect(worst <= tolerance, "VRAM outputs stay within the BFloat16 floor of RAM's");

    uform_arena_free(gpu_arena);
    uform_arena_free(host_arena);
    uform_model_free(gpu_model);
    uform_model_free(host_model);
    uform_executor_free(gpu);
    uform_executor_free(host);
    cuDevicePrimaryCtxRelease(handle);
    printf("Test: VRAM against RAM - PASSED, worst relative error %.5f\n", worst);
    return 0;
}
