/**
 *  @file test/vram.c
 *  @author Ash Vardanian
 *  @date October 18, 2026
 *  @brief The C interface on one NVIDIA GPU against the same checkpoint on the host: text, pixels
 *      and samples, blocking, deferred and split into slices, each landing within the BFloat16
 *      floor of the RAM outputs.
 *
 *  The token head's vectors and the byte decoder's choices over the same texts land within it too.
 *  A one-process mesh over every visible GPU must report the class they dispatch as. The outputs of
 *  a VRAM call stay on the device, so only this test copies them back, with the CUDA driver API.
 *  The checkpoint comes from @c argv[1], as written by @c test/tiny_checkpoint.cpp.
 */
#undef NDEBUG // ! A test's oracle must stay live in every build

#define _GNU_SOURCE // `sched_getaffinity` and `CPU_COUNT`

#include <stdio.h>  // `printf`
#include <stdlib.h> // `exit`, `malloc`, `free`
#include <string.h> // `memset`, `strlen`
#include <time.h>   // `time`, `clock` - the seed a random run draws

#include <inttypes.h> // `PRIu64`
#include <math.h>     // `fabsf`
#include <stdbool.h>  // `bool`
#include <stdint.h>   // `uint8_t`, `uint32_t`

#include <cuda.h> // `cuMemcpyDtoH`, the primary context the executor's streams ride

#include "uform/uform.h"

#if UFORM_OS_LINUX_
#include <sched.h> // `sched_getaffinity`, `CPU_COUNT`
#endif

/*  region Shared C Checks  */
static size_t failed_checks = 0;
static size_t failed_checks_before_case = 0;
static size_t passed_cases = 0;
static size_t failed_cases = 0;
static size_t skipped_cases = 0;
static char const* current_case = NULL;
static char const* program_name = "";
static char const* test_filter = NULL;
static uint64_t test_seed = 42;
static double test_scale = 1.0;
static bool refuse_skips = false;

/** Fills the settings above from the file's own spelled-out variables and flags. */
static void read_test_settings(int argc, char** argv);

/** SplitMix64's finalizer, a bijection that spreads every input bit over all 64 output bits. */
static inline uint64_t split_mix_bits(uint64_t value) {
    value = (value ^ (value >> 30)) * 0xBF58476D1CE4E5B9ull;
    value = (value ^ (value >> 27)) * 0x94D049BB133111EBull;
    return value ^ (value >> 31);
}

/** Counts a failed check at @p file : @p line without stopping, so one run lists every mismatch. */
static inline bool expect_at(bool is_met, char const* file, int line, char const* what) {
    if (is_met) return true;
    fflush(stdout);
    fprintf(stderr, "  check failed at %s:%d\n    %s\n", file, line, what);
    ++failed_checks;
    return false;
}

/** The text of a setting: its @p flag when given, else its environment @p variable , else NULL. */
static inline char const* setting_text(int argc, char** argv, char const* variable, char const* flag) {
    size_t const flag_length = strlen(flag);
    for (int index = 1; index < argc; ++index) {
        if (strcmp(argv[index], flag) == 0)
            return index + 1 < argc && strncmp(argv[index + 1], "--", 2) != 0 ? argv[index + 1] : "1";
        if (strncmp(argv[index], flag, flag_length) == 0 && argv[index][flag_length] == '=')
            return argv[index] + flag_length + 1;
    }
    char const* text = getenv(variable);
    return text && *text ? text : NULL;
}

/** Exits with one line naming a setting that does not parse, so a bad value never runs anything. */
static inline void exit_unparsed(char const* variable, char const* text, char const* expected) {
    fprintf(stderr, "%s=\"%s\" does not parse, expected %s\n", variable, text, expected);
    exit(1);
}

/** The run's seed: 42 when unset, drawn afresh for @c random , else any unsigned 64-bit value. */
static inline uint64_t read_seed_setting(int argc, char** argv, char const* variable, char const* flag) {
    char const* text = setting_text(argc, argv, variable, flag);
    char* end = NULL;
    if (!text) return 42;
    if (strcmp(text, "random") == 0) return split_mix_bits((uint64_t)time(NULL) ^ (uint64_t)clock());
    uint64_t const seed = (uint64_t)strtoull(text, &end, 10);
    if (*end != '\0' || *text == '-') exit_unparsed(variable, text, "an unsigned integer or random");
    return seed;
}

/** A positive real, @p fallback when unset. */
static inline double read_real_setting(int argc, char** argv, char const* variable, char const* flag, double fallback) {
    char const* text = setting_text(argc, argv, variable, flag);
    char* end = NULL;
    if (!text) return fallback;
    double const value = strtod(text, &end);
    if (*end != '\0' || !(value > 0)) exit_unparsed(variable, text, "a positive number like 0.1 or 10");
    return value;
}

/** A switch, @p fallback when unset: 0, false, off and no turn it off, anything else on. */
static inline bool read_flag_setting(int argc, char** argv, char const* variable, char const* flag, bool fallback) {
    char const* text = setting_text(argc, argv, variable, flag);
    if (!text) return fallback;
    return strcmp(text, "0") != 0 && strcmp(text, "false") != 0 && strcmp(text, "off") != 0 && strcmp(text, "no") != 0;
}

/** Reads the settings and prints the settings line, before the first case. */
static inline void start_test_run(int argc, char** argv) {
    program_name = argc != 0 ? argv[0] : "";
    read_test_settings(argc, argv);
    printf("- Seed: %" PRIu64 ", scale: %g, filter: %s\n", test_seed, test_scale, test_filter ? test_filter : "none");
}

/** Starts the case @p label unless the filter leaves it out, matched by substring. */
static inline bool start_test_case(char const* label) {
    if (test_filter && !strstr(label, test_filter)) return false;
    current_case = label;
    failed_checks_before_case = failed_checks;
    return true;
}

/** Ends the case @ref start_test_case began: @p skip_reason when it skipped, NULL when it ran. */
static inline void finish_test_case(char const* skip_reason) {
    size_t const failed = failed_checks - failed_checks_before_case;
    if (skip_reason) {
        ++skipped_cases;
        printf("- %s ... skipped: %s\n", current_case, skip_reason);
    }
    else if (failed == 0) {
        ++passed_cases;
        printf("- %s ... ok\n", current_case);
    }
    else {
        ++failed_cases;
        fflush(stdout);
        fprintf(stderr, "- %s ... FAILED: %zu failed check(s)\n", current_case, failed);
        fprintf(stderr, "  rerun: %s --seed %" PRIu64 " --filter '%s'\n", program_name, test_seed, current_case);
    }
    current_case = NULL;
}

/** The exit code: 1 when a case or a check outside any case failed, or nothing ran; 3 when
 *  skips are refused and one happened; else 0. */
static inline int finish_test_run(void) {
    printf("%zu passed, %zu failed, %zu skipped, seed %" PRIu64 "\n", passed_cases, failed_cases, skipped_cases,
           test_seed);
    if (failed_cases != 0 || failed_checks != 0 || passed_cases + skipped_cases == 0) return 1;
    if (skipped_cases != 0 && refuse_skips) return 3;
    return 0;
}
/*  endregion Shared C Checks  */

static void read_test_settings(int argc, char** argv) {
    test_seed = read_seed_setting(argc, argv, "UFORM_SEED", "--seed");
    test_filter = setting_text(argc, argv, "UFORM_FILTER", "--filter");
    test_scale = read_real_setting(argc, argv, "UFORM_SCALE", "--scale", 1.0);
    refuse_skips = read_flag_setting(argc, argv, "UFORM_NO_SKIPS", "--no-skips", false);
}

#define expect_(condition) expect_at((condition), __FILE__, __LINE__, #condition)

/** Counts a failed check at @p file : @p line unless @p outcome succeeded, naming why not. */
static bool expect_success_at(uform_outcome_t const* outcome, char const* file, int line) {
    return expect_at(outcome->status == uform_status_success_k, file, line,
                     outcome->message ? outcome->message : uform_status_name(outcome->status));
}

/** The largest relative difference a GPU output may show against the host's, from
 *  BFloat16 rounding alone. */
static float const tolerance = 0.05f;

/** The cores the calling thread may run on, or zero where the OS keeps no per-thread mask. */
static int caller_cores(void) {
#if UFORM_OS_LINUX_
    cpu_set_t cores;
    return sched_getaffinity(0, sizeof(cores), &cores) == 0 ? CPU_COUNT(&cores) : 0;
#else
    return 0;
#endif
}

/** The verbs this test drives, all over decoded inputs. */
typedef enum verb_t { verb_text_k, verb_pixels_k, verb_samples_k } verb_t;

static char const* const verb_names[] = {"text", "pixels", "samples"};

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

static void fill_inputs(inputs_t* inputs) {
    char const* const texts[items_count] = {"hello", "", NULL, "the quick brown fox jumps over the lazy dog"};
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

/** Runs @p verb over @p inputs into @p landed, final on return even when deferred. */
static void encode(verb_t verb, inputs_t const* inputs, uform_model_t model, uform_executor_t executor,
                   uform_arena_t arena, uform_completion_t completion, uform_progress_t progress, landed_t* landed) {
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
    expect_(landed->outcome.status != uform_status_deferred_k);
    landed->tokens = text_stats.tokens + image_stats.tokens + audio_stats.tokens;
}

/** One call's outputs copied into host memory. */
typedef struct fetched_t {
    float* embeddings;
    float* states;
    size_t offsets[items_count + 1];
    size_t tokens;
} fetched_t;

static void copy_out(void* target, void const* source, size_t bytes, bool is_on_device) {
    if (!bytes) return;
    if (!is_on_device) {
        memcpy(target, source, bytes);
        return;
    }
    CUresult const copied = cuMemcpyDtoH(target, (CUdeviceptr)source, bytes);
    expect_(copied == CUDA_SUCCESS);
}

static fetched_t fetch(landed_t const* landed, size_t dimensions, bool is_on_device) {
    fetched_t fetched = {0};
    fetched.tokens = landed->tokens;
    fetched.embeddings = malloc(items_count * dimensions * sizeof(float));
    fetched.states = malloc((landed->tokens ? landed->tokens : 1) * dimensions * sizeof(float));
    expect_(fetched.embeddings && fetched.states);
    expect_(landed->embeddings.stride == dimensions * sizeof(float));
    copy_out(fetched.embeddings, landed->embeddings.data, items_count * dimensions * sizeof(float), is_on_device);
    copy_out(fetched.states, landed->states.rows.data, landed->tokens * dimensions * sizeof(float), is_on_device);
    copy_out(fetched.offsets, landed->states.offsets, sizeof(fetched.offsets), is_on_device);
    return fetched;
}

/** The largest over rows of the row's largest absolute difference relative to the reference
 *  row's largest value. */
static float relative_error(float const* rows, float const* reference, size_t count, size_t dimensions) {
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

enum { choices_count = 2 + 3 + 4 + 2 };
static uint8_t const choice_bytes[choices_count] = {'y', 'n', 'A', 'B', 'C', '0', '1', '2', '3', 's', 't'};
static size_t const choice_offsets[items_count + 1] = {0, 2, 5, 9, 11};

/** The token head's vectors and the byte decoder's choices over the texts, copied to the host. */
typedef struct heads_t {
    float* token_vectors;
    size_t token_offsets[items_count + 1];
    float probabilities[choices_count];
    size_t probability_offsets[items_count + 1];
} heads_t;

static heads_t read_heads(inputs_t const* inputs, uform_model_t model, uform_executor_t executor, uform_arena_t arena,
                          size_t token_width, bool is_on_device) {
    heads_t heads = {0};
    uform_outcome_t outcome = {0};
    uform_tape_view_t const texts = {inputs->texts, inputs->text_offsets, uform_scalar_u8_k};
    uform_rows_tape_view_t token_vectors = {0};
    uform_text_encode(&(uform_text_encode_t) {.model = model,
                                              .executor = executor,
                                              .outcome = &outcome,
                                              .count = items_count,
                                              .texts = texts,
                                              .token_vectors = &token_vectors,
                                              .arena = arena,
                                              .outputs_scalar = uform_scalar_f32_k});
    expect_success_at(&outcome, __FILE__, __LINE__);
    expect_(token_vectors.rows.stride == token_width * sizeof(float));
    copy_out(heads.token_offsets, token_vectors.offsets, sizeof(heads.token_offsets), is_on_device);
    size_t const tokens = heads.token_offsets[items_count];
    heads.token_vectors = malloc((tokens ? tokens : 1) * token_width * sizeof(float));
    expect_(heads.token_vectors != NULL);
    copy_out(heads.token_vectors, token_vectors.rows.data, tokens * token_width * sizeof(float), is_on_device);

    uform_tape_view_t probabilities = {0};
    uform_text_choose(&(uform_text_choose_t) {.model = model,
                                              .executor = executor,
                                              .outcome = &outcome,
                                              .count = items_count,
                                              .texts = texts,
                                              .choices = {choice_bytes, choice_offsets, uform_scalar_u8_k},
                                              .probabilities = &probabilities,
                                              .arena = arena});
    expect_success_at(&outcome, __FILE__, __LINE__);
    copy_out(heads.probability_offsets, probabilities.offsets, sizeof(heads.probability_offsets), is_on_device);
    copy_out(heads.probabilities, probabilities.data, sizeof(heads.probabilities), is_on_device);
    return heads;
}

static uform_continuation_t carry_on(size_t completed, size_t total, void* state) {
    (void)completed, (void)total;
    ++*(size_t*)state;
    return uform_continue_k;
}

int main(int argc, char* argv[]) {
    printf("UForm %s\n- Compiled for: %s\n- This machine: %s\n", uform_version_name(), uform_capabilities_compiled(),
           uform_capabilities_available());
    start_test_run(argc, argv);
    if (!expect_(argc > 1)) return finish_test_run();
    uform_bytes_view_t const path = {(uint8_t const*)argv[1], strlen(argv[1])};
    int32_t const device = 0;
    uform_outcome_t outcome = {0};

    uform_executor_t host = NULL, gpu = NULL;
    uform_executor_for_ram(&(uform_executor_for_ram_t) {.executor = &host, .outcome = &outcome, .threads = 2});
    expect_success_at(&outcome, __FILE__, __LINE__);
    int const cores = caller_cores();
    uform_executor_for_vram(&(uform_executor_for_vram_t) {.executor = &gpu, .outcome = &outcome, .vram = {&device, 1}});
    expect_success_at(&outcome, __FILE__, __LINE__);
    bool const creator_keeps_its_cores = caller_cores() == cores;

    uform_model_t host_model = NULL, gpu_model = NULL;
    uform_model_load_ram(&(uform_model_load_ram_t) {.model = &host_model,
                                                    .executor = host,
                                                    .outcome = &outcome,
                                                    .path = path,
                                                    .quantization = uform_scalar_bf16_k});
    expect_success_at(&outcome, __FILE__, __LINE__);
    uform_model_load_vram(&(uform_model_load_vram_t) {.model = &gpu_model,
                                                      .executor = gpu,
                                                      .outcome = &outcome,
                                                      .path = path,
                                                      .vram = {&device, 1},
                                                      .quantization = uform_scalar_bf16_k});
    expect_success_at(&outcome, __FILE__, __LINE__);
    uform_model_description_t description = {0};
    uform_model_describe(&(uform_model_describe_t) {.model = gpu_model,
                                                    .executor = gpu,
                                                    .outcome = &outcome,
                                                    .description = &description});
    expect_success_at(&outcome, __FILE__, __LINE__);
    size_t const dimensions = description.dimensions;
    printf("- VRAM model: %zu dimensions, %zu context, %s\n", dimensions, description.context,
           uform_gpu_class_name(description.gpu_class));

    uform_arena_t host_arena = NULL, gpu_arena = NULL;
    uform_arena_init(&(uform_arena_init_t) {.arena = &host_arena, .executor = host, .outcome = &outcome});
    expect_success_at(&outcome, __FILE__, __LINE__);
    uform_arena_init(&(uform_arena_init_t) {.arena = &gpu_arena, .executor = gpu, .outcome = &outcome});
    expect_success_at(&outcome, __FILE__, __LINE__);

    CUcontext context = NULL;
    CUdevice handle = 0;
    expect_(cuInit(0) == CUDA_SUCCESS && cuDeviceGet(&handle, device) == CUDA_SUCCESS);
    expect_(cuDevicePrimaryCtxRetain(&context, handle) == CUDA_SUCCESS);
    expect_(cuCtxSetCurrent(context) == CUDA_SUCCESS);
    // Every case below needs the executors, models and context; a failed setup ends the run.
    if (failed_checks != 0) return finish_test_run();

    if (start_test_case("test_executor_leaves_its_creator_cores"))
        expect_(creator_keeps_its_cores), finish_test_case(NULL);

    if (start_test_case("test_mesh_dispatches_as_a_known_class")) {
        // A mesh spans every visible GPU, so it gets an executor of its own over all of them.
        int visible = 0;
        int32_t members[64];
        expect_(cuDeviceGetCount(&visible) == CUDA_SUCCESS && visible <= 64);
        for (int member = 0; member != visible; ++member) members[member] = member;
        uform_executor_t box = NULL;
        uform_executor_for_vram(
            &(uform_executor_for_vram_t) {.executor = &box, .outcome = &outcome, .vram = {members, (size_t)visible}});
        expect_success_at(&outcome, __FILE__, __LINE__);
        uform_mesh_t mesh = NULL;
        uform_mesh_bootstrap(&(uform_mesh_bootstrap_t) {.mesh = &mesh, .executor = box, .outcome = &outcome});
        // A build without the trainer has no mesh and refuses one as unsupported.
        bool const trains = outcome.status != uform_status_unsupported_k;
        if (trains) {
            expect_success_at(&outcome, __FILE__, __LINE__);
            uform_mesh_description_t layout = {0};
            uform_mesh_describe(&(uform_mesh_describe_t) {.mesh = mesh, .outcome = &outcome, .description = &layout});
            expect_success_at(&outcome, __FILE__, __LINE__);
            expect_(layout.gpu_class != uform_gpu_class_unknown_k);
            printf("- Mesh: %zu GPUs dispatching as %s\n", layout.gpus_per_host_domain,
                   uform_gpu_class_name(layout.gpu_class));
            uform_mesh_free(mesh);
        }
        uform_executor_free(box);
        finish_test_case(trains ? NULL : "built without UFORM_BUILD_TRAIN");
    }

    // A load names its device by the UForm ordinal, which counts AMD devices after NVIDIA's.
    if (start_test_case("test_model_loads_on_every_listed_ordinal")) {
        uform_gpu_device_t listed[64];
        size_t listed_count = 0;
        uform_gpu_devices(&(uform_gpu_devices_t) {.outcome = &outcome,
                                                  .devices = listed,
                                                  .devices_capacity = 64,
                                                  .devices_count = &listed_count});
        expect_success_at(&outcome, __FILE__, __LINE__);
        for (size_t position = 0; position != listed_count && position != 64; ++position) {
            int32_t const ordinal = (int32_t)position;
            uform_executor_t single = NULL;
            uform_model_t replica = NULL;
            uform_executor_for_vram(
                &(uform_executor_for_vram_t) {.executor = &single, .outcome = &outcome, .vram = {&ordinal, 1}});
            expect_success_at(&outcome, __FILE__, __LINE__);
            uform_model_load_vram(&(uform_model_load_vram_t) {.model = &replica,
                                                              .executor = single,
                                                              .outcome = &outcome,
                                                              .path = path,
                                                              .vram = {&ordinal, 1},
                                                              .quantization = uform_scalar_bf16_k});
            expect_success_at(&outcome, __FILE__, __LINE__);
            uform_model_free(replica);
            uform_executor_free(single);
        }
        finish_test_case(NULL);
    }

    if (start_test_case("test_vram_matches_ram_within_the_bfloat16_floor")) {
        static inputs_t inputs;
        fill_inputs(&inputs);
        float worst = 0.0f;
        for (verb_t verb = verb_text_k; verb <= verb_samples_k; ++verb) {
            landed_t landed;
            uform_progress_t const quiet = {0};
            encode(verb, &inputs, host_model, host, host_arena, uform_completion_blocking_k, quiet, &landed);
            expect_success_at(&landed.outcome, __FILE__, __LINE__);
            fetched_t const reference = fetch(&landed, dimensions, false);

            char const* const runs[] = {"blocking", "deferred", "two slices"};
            for (size_t run = 0; run != 3; ++run) {
                size_t reports = 0;
                uform_progress_t const progress = {run == 2 ? carry_on : NULL, &reports, 2};
                uform_completion_t const completion = run == 1 ? uform_completion_deferred_k
                                                               : uform_completion_blocking_k;
                encode(verb, &inputs, gpu_model, gpu, gpu_arena, completion, progress, &landed);
                expect_success_at(&landed.outcome, __FILE__, __LINE__);
                expect_(landed.tokens == reference.tokens);
                fetched_t const fetched = fetch(&landed, dimensions, true);
                expect_(memcmp(fetched.offsets, reference.offsets, sizeof(fetched.offsets)) == 0);
                expect_(run != 2 || reports == 2);
                float const embeddings = relative_error(fetched.embeddings, reference.embeddings, items_count,
                                                        dimensions);
                float const states = relative_error(fetched.states, reference.states, reference.tokens, dimensions);
                printf("%-8s %-10s tokens %4zu: max relative error embeddings %.5f, states %.5f\n", verb_names[verb],
                       runs[run], fetched.tokens, embeddings, states);
                worst = embeddings > worst ? embeddings : worst;
                worst = states > worst ? states : worst;
                free(fetched.embeddings), free(fetched.states);
            }
            free(reference.embeddings), free(reference.states);
        }

        size_t const token_width = description.token_width;
        heads_t const reference = read_heads(&inputs, host_model, host, host_arena, token_width, false);
        heads_t const fetched = read_heads(&inputs, gpu_model, gpu, gpu_arena, token_width, true);
        expect_(memcmp(fetched.token_offsets, reference.token_offsets, sizeof(fetched.token_offsets)) == 0);
        expect_(memcmp(fetched.probability_offsets, choice_offsets, sizeof(choice_offsets)) == 0 &&
                memcmp(reference.probability_offsets, choice_offsets, sizeof(choice_offsets)) == 0);
        float const token_vectors = relative_error(fetched.token_vectors, reference.token_vectors,
                                                   reference.token_offsets[items_count], token_width);
        float probabilities = 0.0f;
        for (size_t item = 0; item != items_count; ++item) {
            float fetched_sum = 0.0f, reference_sum = 0.0f;
            for (size_t choice = choice_offsets[item]; choice != choice_offsets[item + 1]; ++choice) {
                float const delta = fabsf(fetched.probabilities[choice] - reference.probabilities[choice]);
                probabilities = delta > probabilities ? delta : probabilities;
                fetched_sum += fetched.probabilities[choice], reference_sum += reference.probabilities[choice];
            }
            expect_(fabsf(fetched_sum - 1.0f) < 1e-3f && fabsf(reference_sum - 1.0f) < 1e-3f);
        }
        printf("heads    blocking   tokens %4zu: max relative error token vectors %.5f, probabilities %.5f\n",
               reference.token_offsets[items_count], token_vectors, probabilities);
        worst = token_vectors > worst ? token_vectors : worst;
        worst = probabilities > worst ? probabilities : worst;
        free(fetched.token_vectors), free(reference.token_vectors);
        expect_(worst <= tolerance);
        finish_test_case(NULL);
    }

    uform_arena_free(gpu_arena);
    uform_arena_free(host_arena);
    uform_model_free(gpu_model);
    uform_model_free(host_model);
    uform_executor_free(gpu);
    uform_executor_free(host);
    cuDevicePrimaryCtxRelease(handle);
    return finish_test_run();
}

#undef expect_
