/**
 *  @file c/test.c
 *  @author Ash Vardanian
 *  @date October 17, 2026
 *  @brief Unit tests for the pure-C ABI of UForm.
 *
 *  Exercises the interface every binding consumes: executors, model loading and description,
 *  every encode verb's output shapes and stats, the arenas every output lands in, the contract
 *  of @c uform_outcome_t, deferred calls and their progress, and every refusal a caller can
 *  provoke. A blocking call's outcome is final once it returns; a deferred one is polled,
 *  drained or waited on.
 *
 *  The same file runs against the engine and against the synchronous stand-in of @c src_stub/, so
 *  it asserts contracts rather than numbers: widths come from @c uform_model_describe, and only
 *  text, where every byte is a token, pins exact token counts. The checkpoint comes from @c argv[1]
 *  or @c UFORM_TEST_CHECKPOINT, as written by @c test/tiny_checkpoint.cpp; without either, a small
 *  file is written to the temporary directory, which only the stand-in loads.
 */
#undef NDEBUG // ! A test's oracle must stay live in every build
#ifndef UFORM_TEST_STUB
#define UFORM_TEST_STUB 0
#endif

#if !UFORM_TEST_STUB   // the stand-in runs every call on its caller, built as C99 without threads to race
#include <stdatomic.h> // `atomic_int` - the gate a deferred call waits at
#endif
#include <stdio.h>  // `printf`, `snprintf`, `fopen`, `fwrite`, `remove`
#include <stdlib.h> // `exit`, `getenv`, `strtod`, `strtoull`
#include <string.h> // `memcmp`, `memset`, `strcmp`, `strlen`
#include <time.h>   // `time` - the deadline a spin gives up at

#include <inttypes.h> // `PRIu64`
#include <math.h>     // `sqrtf`
#include <stdbool.h>  // `bool`
#include <stdint.h>   // `uint8_t`, `uint32_t`

#include "uform/uform.h"

#if UFORM_OS_WINDOWS_
#include <io.h> // `_write`
#else
#include <unistd.h> // `write`, `STDERR_FILENO`
#endif

/*  region Shared C Checks  */
static size_t failed_checks = 0;
static size_t failed_checks_before_case = 0;
static size_t passed_cases = 0;
static size_t failed_cases = 0;
static size_t skipped_cases = 0;
static char const* current_case = NULL;
static char const* program_name = "";
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

/** Writes @p text to the standard error through a raw @c write, which a signal handler may call. */
static void write_text(char const* text) {
#if UFORM_OS_WINDOWS_
    (void)!_write(2, text, (unsigned)strlen(text));
#else
    (void)!write(STDERR_FILENO, text, strlen(text));
#endif
}

/** Counts a failed check at @p file : @p line without stopping, so one run lists every mismatch. */
static inline bool expect_at(bool is_met, char const* file, int line, char const* what) {
    if (is_met) return true;
    fflush(stdout);
    char message[512];
    snprintf(message, sizeof message, "  check failed at %s:%d\n    %s\n", file, line, what);
    write_text(message);
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
    char message[512];
    snprintf(message, sizeof message, "%s=\"%s\" does not parse, expected %s\n", variable, text, expected);
    write_text(message);
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
    printf("- Seed: %" PRIu64 ", scale: %g\n", test_seed, test_scale);
}

/** Starts the case @p label, which every run runs. */
static inline void start_test_case(char const* label) {
    current_case = label;
    failed_checks_before_case = failed_checks;
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
        char message[512];
        snprintf(message, sizeof message, "- %s ... FAILED: %zu failed check(s)\n", current_case, failed);
        write_text(message);
        snprintf(message, sizeof message, "  rerun: %s --seed %" PRIu64 "\n", program_name, test_seed);
        write_text(message);
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
    test_scale = read_real_setting(argc, argv, "UFORM_SCALE", "--scale", 1.0);
    refuse_skips = read_flag_setting(argc, argv, "UFORM_NO_SKIPS", "--no-skips", false);
}

#define expect_(condition) expect_at((condition), __FILE__, __LINE__, #condition)

/** The checkpoint every test loads. */
static char const* checkpoint_path = NULL;

static uform_bytes_view_t view_of(char const* text) {
    return (uform_bytes_view_t) {(uint8_t const*)text, strlen(text)};
}

/** Bytes one value of an output @p scalar spans. */
static size_t scalar_bytes(uform_scalar_t scalar) { return scalar == uform_scalar_f32_k ? 4 : 2; }

static uform_executor_t create_executor(void) {
    uform_outcome_t outcome = {0};
    uform_executor_t executor = NULL;
    uform_executor_for_ram(&(uform_executor_for_ram_t) {.executor = &executor, .outcome = &outcome, .threads = 2});
    expect_at(!outcome.message, __FILE__, __LINE__, outcome.message);
    expect_(executor != NULL);
    return executor;
}

static uform_arena_t create_arena(uform_executor_t executor) {
    uform_outcome_t outcome = {0};
    uform_arena_t arena = NULL;
    uform_arena_init(&(uform_arena_init_t) {.arena = &arena, .executor = executor, .outcome = &outcome});
    expect_at(!outcome.message, __FILE__, __LINE__, outcome.message);
    expect_(arena != NULL);
    return arena;
}

/** Loads the checkpoint in BFloat16 through @p executor and reports its geometry
 *  into @p description. */
static uform_model_t load_model(uform_executor_t executor, uform_model_description_t* description) {
    uform_outcome_t outcome = {0};
    uform_model_t model = NULL;
    uform_model_load_ram(&(uform_model_load_ram_t) {.model = &model,
                                                    .executor = executor,
                                                    .outcome = &outcome,
                                                    .path = view_of(checkpoint_path),
                                                    .quantization = uform_scalar_bf16_k});
    expect_at(!outcome.message, __FILE__, __LINE__, outcome.message);
    expect_(model != NULL);
    uform_model_describe(&(uform_model_describe_t) {.model = model,
                                                    .executor = executor,
                                                    .outcome = &outcome,
                                                    .description = description});
    expect_at(!outcome.message, __FILE__, __LINE__, outcome.message);
    expect_(description->dimensions > 0 && description->context > 0);
    return model;
}

enum { texts_capacity = 16, text_bytes_capacity = 8192 };

/** A tape of UTF-8 texts the tests encode, each item a run of one letter. */
typedef struct texts_t {
    uint8_t bytes[text_bytes_capacity];
    size_t offsets[texts_capacity + 1];
    size_t count;
} texts_t;

/** Fills @p texts with @p count items of the given @p lengths. */
static void fill_texts(texts_t* texts, size_t count, size_t const* lengths) {
    expect_(count <= texts_capacity);
    texts->count = count;
    texts->offsets[0] = 0;
    for (size_t item = 0; item != count; ++item) {
        size_t const begin = texts->offsets[item];
        expect_(begin + lengths[item] <= text_bytes_capacity);
        memset(texts->bytes + begin, 'a' + (int)(item % 26), lengths[item]);
        texts->offsets[item + 1] = begin + lengths[item];
    }
}

static uform_tape_view_t tape_of(texts_t const* texts) {
    return (uform_tape_view_t) {texts->bytes, texts->offsets, uform_scalar_u8_k};
}

/** Checks a run of output rows: present, of the model's width, written packed as @p scalar. */
static void check_rows(uform_rows_view_t const* rows, size_t dimensions, uform_scalar_t scalar) {
    expect_(rows->data != NULL);
    expect_(rows->columns == dimensions);
    expect_(rows->scalar == scalar);
    expect_(rows->stride == dimensions * scalar_bytes(scalar));
}

/** Checks per-token states against the call's stats: offsets from zero, never decreasing, no item
 *  past the context, and the last offset the tokens counted. */
static void check_states(uform_rows_tape_view_t const* states, size_t count, uform_model_description_t const* model,
                         uform_scalar_t scalar, size_t tokens) {
    check_rows(&states->rows, model->dimensions, scalar);
    expect_(states->offsets != NULL);
    expect_(states->offsets[0] == 0);
    for (size_t item = 0; item != count; ++item) {
        expect_(states->offsets[item + 1] >= states->offsets[item]);
        expect_(states->offsets[item + 1] - states->offsets[item] <= model->context);
    }
    expect_(states->offsets[count] == tokens);
}

/** The library-wide probes, which take no handle and cannot fail, and every name
 *  table read back. */
static void test_library_probes(void) {
    expect_(uform_capabilities_compiled() != NULL);
    expect_(uform_capabilities_available() != NULL);
    int major = -1, minor = -1, patch = -1;
    uform_version(&major, &minor, &patch);
    expect_(major == UFORM_VERSION_MAJOR && minor == UFORM_VERSION_MINOR && patch == UFORM_VERSION_PATCH);
    char core[64];
    snprintf(core, sizeof(core), "%d.%d.%d", major, minor, patch);
    char const* const named = uform_version_name();
    expect_(strncmp(named, core, strlen(core)) == 0 && (named[strlen(core)] == '\0' || named[strlen(core)] == '-'));

    uform_status_t const statuses[] = {
        uform_status_success_k,
        uform_status_deferred_k,
        uform_status_out_of_memory_k,
        uform_status_shape_mismatch_k,
        uform_status_unsupported_layout_k,
        uform_status_unknown_kind_k,
        uform_status_io_k,
        uform_status_device_k,
        uform_status_malformed_k,
        uform_status_internal_k,
        uform_status_uninitialized_k,
        uform_status_config_mismatch_k,
        uform_status_contended_k,
        uform_status_cancelled_k,
        uform_status_unsupported_k,
        uform_status_invalid_argument_k,
        uform_status_overflow_k,
    };
    for (size_t position = 0; position != sizeof(statuses) / sizeof(statuses[0]); ++position) {
        uform_status_t read = uform_status_internal_k;
        expect_(uform_status_from_name(view_of(uform_status_name(statuses[position])), &read) ==
                    uform_status_success_k &&
                read == statuses[position]);
    }
    expect_(strcmp(uform_status_name(uform_status_deferred_k), "deferred") == 0);
    expect_(strcmp(uform_status_name(uform_status_out_of_memory_k), "out-of-memory") == 0);
    expect_(strcmp(uform_status_name(uform_status_uninitialized_k), "uninitialized") == 0);
    expect_(strcmp(uform_status_name((uform_status_t)6), "unrecognized") == 0);
    uform_status_t untouched_status = uform_status_io_k;
    expect_(uform_status_from_name(view_of("no-such"), &untouched_status) == uform_status_invalid_argument_k &&
            untouched_status == uform_status_io_k);

    uform_scalar_t const scalars[] = {
        uform_scalar_f64_k,  uform_scalar_f32_k,  uform_scalar_bf16_k, uform_scalar_f16_k, uform_scalar_e5m2_k,
        uform_scalar_e4m3_k, uform_scalar_e3m2_k, uform_scalar_e2m3_k, uform_scalar_i8_k,  uform_scalar_i4x2_k,
        uform_scalar_i32_k,  uform_scalar_u8_k,   uform_scalar_u4x2_k, uform_scalar_u1x8_k};
    for (size_t position = 0; position != sizeof(scalars) / sizeof(scalars[0]); ++position) {
        uform_scalar_t read = uform_scalar_unknown_k;
        expect_(uform_scalar_from_name(view_of(uform_scalar_name(scalars[position])), &read) ==
                    uform_status_success_k &&
                read == scalars[position]);
    }
    expect_(strcmp(uform_scalar_name(uform_scalar_f32_k), "f32") == 0);
    uform_scalar_t untouched_scalar = uform_scalar_f32_k;
    expect_(uform_scalar_from_name(view_of("f33"), &untouched_scalar) == uform_status_invalid_argument_k &&
            untouched_scalar == uform_scalar_f32_k);

    uform_mood_t mood = uform_mood_parked_k;
    expect_(uform_mood_from_name(view_of("spinning"), &mood) == uform_status_success_k &&
            mood == uform_mood_spinning_k);
    expect_(uform_mood_from_name(view_of("sleepy"), &mood) == uform_status_invalid_argument_k &&
            mood == uform_mood_spinning_k);
    expect_(strcmp(uform_mood_name(uform_mood_parked_k), "parked") == 0);
    uform_modality_t const modalities[] = {uform_modality_text_k, uform_modality_image_k, uform_modality_video_k,
                                           uform_modality_audio_k};
    for (size_t position = 0; position != 4; ++position) {
        uform_modality_t read = uform_modality_audio_k;
        expect_(uform_modality_from_name(view_of(uform_modality_name(modalities[position])), &read) ==
                    uform_status_success_k &&
                read == modalities[position]);
    }
    expect_(strcmp(uform_multimem_tier_name(uform_multimem_tier_peer_read_k), "peer-read") == 0);
    uform_multimem_tier_t tier = uform_multimem_tier_switch_reduced_k;
    expect_(uform_multimem_tier_from_name(view_of("peer-read"), &tier) == uform_status_success_k &&
            tier == uform_multimem_tier_peer_read_k);
    expect_(uform_multimem_tier_from_name(view_of("ring"), &tier) == uform_status_invalid_argument_k &&
            tier == uform_multimem_tier_peer_read_k);
    for (int value = 0; value != 4; ++value) {
        uform_pair_loss_form_t form = uform_pair_loss_form_sigmoid_k;
        uform_late_interaction_side_t side = uform_late_interaction_side_query_and_document_k;
        expect_(uform_pair_loss_form_from_name(view_of(uform_pair_loss_form_name((uform_pair_loss_form_t)value)),
                                               &form) == uform_status_success_k &&
                form == (uform_pair_loss_form_t)value);
        expect_(uform_late_interaction_side_from_name(
                    view_of(uform_late_interaction_side_name((uform_late_interaction_side_t)value)), &side) ==
                    uform_status_success_k &&
                side == (uform_late_interaction_side_t)value);
    }
    expect_(strcmp(uform_late_interaction_side_name(uform_late_interaction_side_query_and_document_k),
                   "query-and-document") == 0);
    uform_pair_loss_form_t untouched_form = uform_pair_loss_form_ranked_k;
    expect_(uform_pair_loss_form_from_name(view_of("steep"), &untouched_form) == uform_status_invalid_argument_k &&
            untouched_form == uform_pair_loss_form_ranked_k);
    expect_(strcmp(uform_gpu_vendor_name(uform_gpu_vendor_unknown_k), "unknown") == 0 &&
            strcmp(uform_gpu_vendor_name(uform_gpu_vendor_nvidia_k), "nvidia") == 0 &&
            strcmp(uform_gpu_vendor_name(uform_gpu_vendor_amd_k), "amd") == 0 &&
            strcmp(uform_gpu_vendor_name(uform_gpu_vendor_apple_k), "apple") == 0);
    expect_(strcmp(uform_gpu_vendor_name((uform_gpu_vendor_t)7), "unrecognized") == 0);
    expect_(strcmp(uform_host_access_name(uform_host_access_refused_k), "refused") == 0);
    expect_(strcmp(uform_host_access_name(uform_host_access_readable_k), "readable") == 0);
    expect_(strcmp(uform_host_access_name((uform_host_access_t)2), "unrecognized") == 0);
    expect_(strcmp(uform_residency_name(uform_residency_ram_k), "ram") == 0 &&
            strcmp(uform_residency_name(uform_residency_vram_k), "vram") == 0);
    expect_(strcmp(uform_residency_name(uform_residency_unknown_k), "unknown") == 0);
    expect_(strcmp(uform_residency_name((uform_residency_t)3), "unrecognized") == 0);
    // Every enum declaring an unknown value spells it "unknown" and reads that spelling back.
    uform_scalar_t unknown_scalar = uform_scalar_f32_k;
    uform_cpu_class_t unknown_cpu_class = uform_cpu_class_serial_k;
    uform_gpu_class_t unknown_gpu_class = uform_gpu_class_nvidia_ampere_k;
    expect_(uform_scalar_from_name(view_of("unknown"), &unknown_scalar) == uform_status_success_k &&
            unknown_scalar == uform_scalar_unknown_k);
    expect_(uform_cpu_class_from_name(view_of("unknown"), &unknown_cpu_class) == uform_status_success_k &&
            unknown_cpu_class == uform_cpu_class_unknown_k);
    expect_(uform_gpu_class_from_name(view_of("unknown"), &unknown_gpu_class) == uform_status_success_k &&
            unknown_gpu_class == uform_gpu_class_unknown_k);

    // Classes fit a byte, so walking every byte meets each class and every gap between blocks;
    // the CPU walk starts at "unknown", which a pin spells too.
    for (int value = 0; value != 256; ++value) {
        uform_cpu_class_t const code = (uform_cpu_class_t)value;
        char const* const spelled = uform_cpu_class_name(code);
        uform_cpu_class_t read = uform_cpu_class_serial_k;
        uform_status_t const status = uform_cpu_class_from_name(view_of(spelled), &read);
        expect_(strcmp(spelled, "unrecognized") == 0 ? status == uform_status_invalid_argument_k
                                                     : status == uform_status_success_k && read == code);
    }
    for (int value = 1; value != 256; ++value) {
        uform_gpu_class_t const code = (uform_gpu_class_t)value;
        char const* const spelled = uform_gpu_class_name(code);
        uform_gpu_class_t read = uform_gpu_class_unknown_k;
        uform_status_t const status = uform_gpu_class_from_name(view_of(spelled), &read);
        expect_(strcmp(spelled, "unrecognized") == 0 ? status == uform_status_invalid_argument_k
                                                     : status == uform_status_success_k && read == code);
    }
    expect_(strcmp(uform_cpu_class_name(uform_cpu_class_ibm_power10_k), "ibm-power10") == 0);
    expect_(strcmp(uform_cpu_class_name(uform_cpu_class_intel_haswell_k), "intel-haswell") == 0);
    uform_cpu_class_t untouched_cpu_class = uform_cpu_class_intel_haswell_k;
    expect_(uform_cpu_class_from_name(view_of("intel-haswel"), &untouched_cpu_class) ==
                uform_status_invalid_argument_k &&
            untouched_cpu_class == uform_cpu_class_intel_haswell_k);
    uform_gpu_class_t untouched_gpu_class = uform_gpu_class_nvidia_ada_k;
    expect_(uform_gpu_class_from_name(view_of("nvidia-volta"), &untouched_gpu_class) ==
                uform_status_invalid_argument_k &&
            untouched_gpu_class == uform_gpu_class_nvidia_ada_k);

    uform_outcome_t outcome = {0};
    uform_gpu_device_t devices[8];
    size_t devices_count = 0;
    uform_gpu_devices(&(uform_gpu_devices_t) {.outcome = &outcome,
                                              .devices = devices,
                                              .devices_capacity = 8,
                                              .devices_count = &devices_count});
    expect_at(!outcome.message, __FILE__, __LINE__, outcome.message);
    for (size_t position = 0; position != devices_count && position != 8; ++position) {
        printf("- GPU #%zu: %s %s, %u units, %llu MiB\n", position, devices[position].name,
               uform_gpu_class_name(devices[position].gpu_class), devices[position].compute_units,
               (unsigned long long)(devices[position].memory_bytes >> 20));
        expect_(devices[position].vendor != uform_gpu_vendor_unknown_k);
    }
}

/** Every executor kind: RAM always, VRAM where the build and the machine have one, a pool the
 *  caller owns nowhere yet; a drain of an idle executor succeeds, and every free takes @c NULL. */
static void test_executors(void) {
    uform_executor_t const executor = create_executor();
    uform_outcome_t outcome = {uform_status_io_k, "a stale message the call must overwrite"};
    uform_executor_drain(&(uform_executor_drain_t) {.executor = executor, .outcome = &outcome});
    expect_(outcome.status == uform_status_success_k && outcome.message == NULL);

    // One named memory domain spawns the threads near it; node 0 is the one every machine has.
    int32_t const first_domain[] = {0};
    uform_executor_t placed = NULL;
    uform_executor_for_ram(&(uform_executor_for_ram_t) {.executor = &placed,
                                                        .outcome = &outcome,
                                                        .ram = {first_domain, 1, uform_cpu_class_unknown_k},
                                                        .threads = 2});
    expect_(outcome.status == uform_status_success_k && placed != NULL);
    expect_(uform_executor_free(placed) == uform_status_success_k);

    uform_executor_t gpu = NULL;
    uform_executor_for_vram(&(uform_executor_for_vram_t) {.executor = &gpu, .outcome = &outcome});
    expect_(outcome.status == uform_status_success_k || outcome.status == uform_status_device_k);
    expect_((outcome.status == uform_status_success_k) == (gpu != NULL));
    expect_(uform_executor_free(gpu) == uform_status_success_k);

    // Streams pair with the devices named beside them, one each, which every build checks before
    // looking for a device; a null stream is refused once the device answers.
    void* const caller_streams[] = {&outcome, &outcome};
    gpu = NULL;
    uform_executor_for_vram(
        &(uform_executor_for_vram_t) {.executor = &gpu, .outcome = &outcome, .streams = caller_streams});
    expect_(outcome.status == uform_status_invalid_argument_k && gpu == NULL);
    int32_t const device_twice[] = {0, 0};
    uform_executor_for_vram(&(uform_executor_for_vram_t) {.executor = &gpu,
                                                          .outcome = &outcome,
                                                          .vram = {.devices = device_twice, .devices_count = 2},
                                                          .streams = caller_streams});
    expect_(outcome.status == uform_status_invalid_argument_k && gpu == NULL);
    uform_gpu_device_t first_listed[1] = {0};
    size_t listed_count = 0;
    uform_gpu_devices(&(uform_gpu_devices_t) {.outcome = &outcome,
                                              .devices = first_listed,
                                              .devices_capacity = 1,
                                              .devices_count = &listed_count});
    void* const null_streams[] = {NULL};
    uform_executor_for_vram(&(uform_executor_for_vram_t) {.executor = &gpu,
                                                          .outcome = &outcome,
                                                          .vram = {.devices = device_twice, .devices_count = 1},
                                                          .streams = null_streams});
    uform_status_t const null_refusal = !listed_count ? uform_status_device_k
                                        : first_listed[0].gpu_class == uform_gpu_class_unknown_k
                                            ? uform_status_unsupported_k
                                            : uform_status_invalid_argument_k;
    expect_(outcome.status == null_refusal && gpu == NULL);
    uform_executor_for_vram(&(uform_executor_for_vram_t) {.executor = &gpu,
                                                          .outcome = &outcome,
                                                          .vram = {.devices = NULL, .devices_count = 1},
                                                          .streams = caller_streams});
    expect_(outcome.status == uform_status_invalid_argument_k);

    expect_(uform_executor_free(NULL) == uform_status_success_k);
    uform_arena_free(NULL);
    uform_model_free(NULL);
    uform_mesh_free(NULL);
    uform_trainer_free(NULL);
    expect_(uform_executor_free(executor) == uform_status_success_k);
}

/** A checkpoint loads from a path and from a buffer, in either quantization, and its description
 *  echoes the quantization asked for; another quantization and a missing file are refused,
 *  leaving no model. */
static void test_model_load(void) {
    uform_executor_t const executor = create_executor();
    uform_model_description_t description = {0};
    uform_model_t const model = load_model(executor, &description);
    expect_(description.quantization == uform_scalar_bf16_k);
    expect_(description.cpu_class != uform_cpu_class_unknown_k);
    expect_(description.gpu_class == uform_gpu_class_unknown_k);

    uform_outcome_t pin_outcome = {0};
    // A monoversioned build compiles its one class alone, so no serial floor is there to cap at.
    bool const monoversioned = description.cpu_class == uform_cpu_class_monoversioned_k;
    uform_cpu_class_t const capped_class = monoversioned ? uform_cpu_class_monoversioned_k : uform_cpu_class_serial_k;
    uform_model_t capped = NULL;
    uform_model_load_ram(&(uform_model_load_ram_t) {.model = &capped,
                                                    .executor = executor,
                                                    .outcome = &pin_outcome,
                                                    .path = view_of(checkpoint_path),
                                                    .ram = {.cpu_class_ceiling = capped_class},
                                                    .quantization = uform_scalar_bf16_k});
    expect_at(!pin_outcome.message, __FILE__, __LINE__, pin_outcome.message);
    uform_model_description_t capped_description = {0};
    uform_model_describe(&(uform_model_describe_t) {.model = capped,
                                                    .executor = executor,
                                                    .outcome = &pin_outcome,
                                                    .description = &capped_description});
    expect_(capped_description.cpu_class == capped_class);
    uform_model_free(capped);

    // A ceiling of another family shares only the serial floor with this CPU, so it runs as serial.
    uform_model_t foreign = NULL;
    uform_cpu_class_t const foreign_class = description.cpu_class == uform_cpu_class_wasm64_extended_k
                                                ? uform_cpu_class_intel_diamond_k
                                                : uform_cpu_class_wasm64_extended_k;
    uform_model_load_ram(&(uform_model_load_ram_t) {.model = &foreign,
                                                    .executor = executor,
                                                    .outcome = &pin_outcome,
                                                    .path = view_of(checkpoint_path),
                                                    .ram = {.cpu_class_ceiling = foreign_class},
                                                    .quantization = uform_scalar_bf16_k});
    if (monoversioned) {
        expect_(pin_outcome.status == uform_status_unsupported_k);
        expect_(foreign == NULL);
    }
    else {
        expect_at(!pin_outcome.message, __FILE__, __LINE__, pin_outcome.message);
        uform_model_description_t foreign_description = {0};
        uform_model_describe(&(uform_model_describe_t) {.model = foreign,
                                                        .executor = executor,
                                                        .outcome = &pin_outcome,
                                                        .description = &foreign_description});
        expect_(foreign_description.cpu_class == uform_cpu_class_serial_k);
        uform_model_free(foreign);
    }

    FILE* const file = fopen(checkpoint_path, "rb");
    expect_(file != NULL);
    fseek(file, 0, SEEK_END);
    long const length = ftell(file);
    fseek(file, 0, SEEK_SET);
    uint8_t* const buffer = (uint8_t*)malloc((size_t)length);
    expect_(buffer && fread(buffer, 1, (size_t)length, file) == (size_t)length);
    fclose(file);

    uform_outcome_t outcome = {0};
    uform_model_t buffered = NULL;
    uform_scalar_t quantization = uform_scalar_e4m3_k;
    uform_model_load_ram_buffer(&(uform_model_load_ram_buffer_t) {.model = &buffered,
                                                                  .executor = executor,
                                                                  .outcome = &outcome,
                                                                  .buffer = {buffer, (size_t)length},
                                                                  .quantization = quantization});
    // A class without E4M3 products refuses E4M3, so the buffer path is then checked in BFloat16.
    if (outcome.status == uform_status_unsupported_k) {
        expect_(buffered == NULL);
        quantization = uform_scalar_bf16_k;
        uform_model_load_ram_buffer(&(uform_model_load_ram_buffer_t) {.model = &buffered,
                                                                      .executor = executor,
                                                                      .outcome = &outcome,
                                                                      .buffer = {buffer, (size_t)length},
                                                                      .quantization = quantization});
    }
    expect_at(!outcome.message, __FILE__, __LINE__, outcome.message);
    free(buffer);
    uform_model_description_t buffered_description = {0};
    uform_model_describe(&(uform_model_describe_t) {.model = buffered,
                                                    .executor = executor,
                                                    .outcome = &outcome,
                                                    .description = &buffered_description});
    expect_at(!outcome.message, __FILE__, __LINE__, outcome.message);
    expect_(buffered_description.quantization == quantization);
    expect_(buffered_description.dimensions == description.dimensions);

    uform_model_t refused = NULL;
    uform_model_load_ram(&(uform_model_load_ram_t) {.model = &refused,
                                                    .executor = executor,
                                                    .outcome = &outcome,
                                                    .path = view_of(checkpoint_path),
                                                    .quantization = uform_scalar_f32_k});
    expect_(outcome.status == uform_status_unsupported_k);
    expect_(refused == NULL);
    uform_model_load_ram(&(uform_model_load_ram_t) {.model = &refused,
                                                    .executor = executor,
                                                    .outcome = &outcome,
                                                    .path = view_of("/no/such/uform/checkpoint.safetensors"),
                                                    .quantization = uform_scalar_bf16_k});
    expect_(outcome.status == uform_status_io_k);
    expect_(refused == NULL);
    char const nul_path[] = "tiny.safetensors\0.missing";
    uform_model_load_ram(&(uform_model_load_ram_t) {.model = &refused,
                                                    .executor = executor,
                                                    .outcome = &outcome,
                                                    .path = {(uint8_t const*)nul_path, sizeof(nul_path) - 1},
                                                    .quantization = uform_scalar_bf16_k});
    expect_(outcome.status == uform_status_invalid_argument_k);
    expect_(refused == NULL);

    uform_model_free(buffered);
    uform_model_free(model);
    expect_(uform_executor_free(executor) == uform_status_success_k);
}

/** Text in every output scalar: one embedding per item, one state per byte up to the context, and
 *  an item past the context cut there and counted; a scalar no output takes is refused with every
 *  output left null. */
static void test_text_shapes(void) {
    uform_executor_t const executor = create_executor();
    uform_arena_t const arena = create_arena(executor);
    uform_model_description_t model_description = {0};
    uform_model_t const model = load_model(executor, &model_description);
    size_t const context = model_description.context;
    expect_(context < text_bytes_capacity - 16);
    size_t const lengths[] = {1, 5, context + 7};
    texts_t texts;
    fill_texts(&texts, 3, lengths);

    uform_scalar_t const scalars[] = {uform_scalar_f32_k, uform_scalar_bf16_k, uform_scalar_f16_k};
    for (size_t position = 0; position != 3; ++position) {
        uform_outcome_t outcome = {0};
        uform_rows_view_t embeddings = {0};
        uform_rows_tape_view_t states = {0};
        uform_text_encode_stats_t stats = {0};
        uform_text_encode(&(uform_text_encode_t) {.model = model,
                                                  .executor = executor,
                                                  .outcome = &outcome,
                                                  .count = texts.count,
                                                  .texts = tape_of(&texts),
                                                  .embeddings = &embeddings,
                                                  .states = &states,
                                                  .arena = arena,
                                                  .outputs_scalar = scalars[position],
                                                  .stats = &stats});
        expect_at(!outcome.message, __FILE__, __LINE__, outcome.message);
        check_rows(&embeddings, model_description.dimensions, scalars[position]);
        check_states(&states, texts.count, &model_description, scalars[position], stats.tokens);
        expect_(stats.truncated == 1);
        expect_(stats.tokens == 1 + 5 + context);
        expect_(states.offsets[1] == 1 && states.offsets[2] == 6);
    }

    uform_outcome_t outcome = {0};
    uform_rows_view_t embeddings = {0};
    uform_text_encode(&(uform_text_encode_t) {.model = model,
                                              .executor = executor,
                                              .outcome = &outcome,
                                              .count = texts.count,
                                              .texts = tape_of(&texts),
                                              .embeddings = &embeddings,
                                              .arena = arena,
                                              .outputs_scalar = uform_scalar_u8_k});
    expect_(outcome.status == uform_status_unsupported_k);
    expect_(embeddings.data == NULL);

    uform_model_free(model);
    uform_arena_free(arena);
    expect_(uform_executor_free(executor) == uform_status_success_k);
}

/** Token vectors and choices: with the heads, unit-length vectors on the states' offsets and
 *  probabilities summing to one over each item's choices, a group past the model's refused; without
 *  them, @c unsupported_k. */
static void test_heads(void) {
    uform_executor_t const executor = create_executor();
    uform_arena_t const arena = create_arena(executor);
    uform_model_description_t description = {0};
    uform_model_t const model = load_model(executor, &description);
    size_t const lengths[] = {3, 9, description.context + 4};
    texts_t texts;
    fill_texts(&texts, 3, lengths);

    uform_outcome_t outcome = {0};
    uform_rows_tape_view_t states = {0}, token_vectors = {0};
    uform_text_encode(&(uform_text_encode_t) {.model = model,
                                              .executor = executor,
                                              .outcome = &outcome,
                                              .count = texts.count,
                                              .texts = tape_of(&texts),
                                              .states = &states,
                                              .token_vectors = &token_vectors,
                                              .arena = arena,
                                              .outputs_scalar = uform_scalar_f32_k});
    if (!description.token_width) expect_(outcome.status == uform_status_unsupported_k);
    else {
        expect_at(!outcome.message, __FILE__, __LINE__, outcome.message);
        expect_(token_vectors.rows.columns == description.token_width);
        for (size_t item = 0; item <= texts.count; ++item) expect_(token_vectors.offsets[item] == states.offsets[item]);
        for (size_t row = 0; row != token_vectors.offsets[texts.count]; ++row) {
            float const* const vector = (float const*)((char const*)token_vectors.rows.data +
                                                       row * token_vectors.rows.stride);
            float squares = 0;
            for (size_t column = 0; column != description.token_width; ++column)
                squares += vector[column] * vector[column];
            expect_(squares > 0.98f && squares < 1.02f);
        }
    }

    uint8_t const choice_bytes[] = {'y', 'n', 'A', 'B', 'C', 's'};
    size_t const choice_offsets[] = {0, 2, 5, 6};
    uint32_t const groups[] = {0, 0, 0};
    uform_tape_view_t probabilities = {0};
    uform_text_choose_t choose = {.model = model,
                                  .executor = executor,
                                  .outcome = &outcome,
                                  .count = texts.count,
                                  .texts = tape_of(&texts),
                                  .choices = {choice_bytes, choice_offsets, uform_scalar_u8_k},
                                  .temperature_groups = groups,
                                  .probabilities = &probabilities,
                                  .arena = arena};
    uform_text_choose(&choose);
    if (!description.choice_temperature_groups) {
        expect_(outcome.status == uform_status_unsupported_k);
        expect_(probabilities.data == NULL);
    }
    else {
        expect_at(!outcome.message, __FILE__, __LINE__, outcome.message);
        float const* const values = (float const*)probabilities.data;
        for (size_t item = 0; item != texts.count; ++item) {
            expect_(probabilities.offsets[item + 1] - probabilities.offsets[item] ==
                    choice_offsets[item + 1] - choice_offsets[item]);
            float sum = 0;
            for (size_t choice = probabilities.offsets[item]; choice != probabilities.offsets[item + 1]; ++choice)
                expect_(values[choice] >= 0 && values[choice] <= 1), sum += values[choice];
            expect_(sum > 0.999f && sum < 1.001f);
        }
        uint32_t const past[] = {0, (uint32_t)description.choice_temperature_groups, 0};
        choose.temperature_groups = past;
        uform_text_choose(&choose);
        expect_(outcome.status == uform_status_invalid_argument_k);
    }

    uform_model_free(model);
    uform_arena_free(arena);
    expect_(uform_executor_free(executor) == uform_status_success_k);
}

/** CRC-32 of @p length bytes, as PNG chunks store it. */
static uint32_t crc32_of(uint8_t const* bytes, size_t length, uint32_t crc) {
    crc = ~crc;
    for (size_t position = 0; position != length; ++position) {
        crc ^= bytes[position];
        for (int bit = 0; bit != 8; ++bit) crc = (crc >> 1) ^ (0xEDB88320u & (0u - (crc & 1u)));
    }
    return ~crc;
}

static void put_big_endian(uint8_t* at, uint32_t value) {
    at[0] = (uint8_t)(value >> 24), at[1] = (uint8_t)(value >> 16), at[2] = (uint8_t)(value >> 8);
    at[3] = (uint8_t)value;
}

/** Appends one PNG chunk of @p type around @p length bytes of @p data at @p at, answering
 *  the bytes written. */
static size_t put_png_chunk(uint8_t* at, char const* type, uint8_t const* data, size_t length) {
    put_big_endian(at, (uint32_t)length);
    memcpy(at + 4, type, 4);
    if (length) memcpy(at + 8, data, length);
    put_big_endian(at + 8 + length, crc32_of(at + 4, length + 4, 0));
    return length + 12;
}

enum { png_side = 8, png_raw_bytes = png_side * (1 + png_side * 3), png_capacity = 320 };

/** Writes a valid 8 x 8 RGB PNG of a gradient into @p png, its pixels in one stored deflate block,
 *  answering its length; the buffer must hold @c png_capacity bytes. */
static size_t write_png(uint8_t* png) {
    static uint8_t const signature[8] = {0x89, 'P', 'N', 'G', '\r', '\n', 0x1A, '\n'};
    memcpy(png, signature, 8);
    uint8_t header[13] = {0};
    put_big_endian(header, png_side), put_big_endian(header + 4, png_side);
    header[8] = 8, header[9] = 2;
    size_t length = 8 + put_png_chunk(png + 8, "IHDR", header, 13);

    uint8_t stream[2 + 5 + png_raw_bytes + 4];
    stream[0] = 0x78, stream[1] = 0x01, stream[2] = 0x01;
    stream[3] = png_raw_bytes & 0xFF, stream[4] = png_raw_bytes >> 8;
    stream[5] = (uint8_t)~stream[3], stream[6] = (uint8_t)~stream[4];
    uint8_t* const raw = stream + 7;
    for (size_t row = 0; row != png_side; ++row) {
        raw[row * (1 + png_side * 3)] = 0;
        for (size_t byte = 0; byte != png_side * 3; ++byte)
            raw[row * (1 + png_side * 3) + 1 + byte] = (uint8_t)(row * 32 + byte * 8);
    }
    uint32_t adler_low = 1, adler_high = 0;
    for (size_t byte = 0; byte != png_raw_bytes; ++byte)
        adler_low = (adler_low + raw[byte]) % 65521, adler_high = (adler_high + adler_low) % 65521;
    put_big_endian(stream + 7 + png_raw_bytes, (adler_high << 16) | adler_low);
    length += put_png_chunk(png + length, "IDAT", stream, sizeof(stream));
    return length + put_png_chunk(png + length, "IEND", NULL, 0);
}

static void put_little_endian(uint8_t* at, uint32_t value, size_t bytes) {
    for (size_t byte = 0; byte != bytes; ++byte) at[byte] = (uint8_t)(value >> (byte * 8));
}

enum { wav_samples = 1600, wav_bytes = 44 + wav_samples * 2 };

/** Writes a valid 16 kHz mono 16-bit WAV of a sawtooth into @p wav, answering its length. */
static size_t write_wav(uint8_t* wav) {
    memcpy(wav, "RIFF", 4), put_little_endian(wav + 4, wav_bytes - 8, 4), memcpy(wav + 8, "WAVEfmt ", 8);
    put_little_endian(wav + 16, 16, 4), put_little_endian(wav + 20, 1, 2), put_little_endian(wav + 22, 1, 2);
    put_little_endian(wav + 24, 16000, 4), put_little_endian(wav + 28, 32000, 4);
    put_little_endian(wav + 32, 2, 2), put_little_endian(wav + 34, 16, 2);
    memcpy(wav + 36, "data", 4), put_little_endian(wav + 40, wav_samples * 2, 4);
    for (size_t sample = 0; sample != wav_samples; ++sample)
        put_little_endian(wav + 44 + sample * 2, (uint32_t)(int16_t)((sample % 64) * 512 - 16384), 2);
    return wav_bytes;
}

/** The outputs of one media call, checked alike whatever the verb. */
typedef struct media_outputs_t {
    uform_outcome_t outcome;
    uform_rows_view_t embeddings;
    uform_rows_tape_view_t states;
    size_t truncated;
    size_t tokens;
} media_outputs_t;

static void check_media(media_outputs_t const* outputs, size_t count, uform_model_description_t const* model) {
    expect_at(!outputs->outcome.message, __FILE__, __LINE__, outputs->outcome.message);
    check_rows(&outputs->embeddings, model->dimensions, uform_scalar_f32_k);
    check_states(&outputs->states, count, model, uform_scalar_f32_k, outputs->tokens);
    expect_(outputs->truncated <= count);
    for (size_t item = 0; item != count; ++item)
        expect_(outputs->states.offsets[item + 1] > outputs->states.offsets[item]);
}

/** Every media verb lands one embedding per item and states whose offsets agree with its stats. A
 *  file the engine cannot decode fails whole as malformed, leaving its outputs null; the stand-in
 *  decodes nothing, so it encodes any bytes. */
static void test_media_shapes(void) {
    uform_executor_t const executor = create_executor();
    uform_arena_t const arena = create_arena(executor);
    uform_model_description_t model = {0};
    uform_model_t const handle = load_model(executor, &model);

    uint8_t files[2 * png_capacity];
    size_t const png_length = write_png(files);
    memcpy(files + png_length, files, png_length);
    size_t const file_offsets[] = {0, png_length, 2 * png_length};
    uform_image_encode_stats_t image_stats = {0};
    media_outputs_t outputs = {0};
    uform_image_encode(&(uform_image_encode_t) {.model = handle,
                                                .executor = executor,
                                                .outcome = &outputs.outcome,
                                                .count = 2,
                                                .images = {files, file_offsets, uform_scalar_u8_k},
                                                .embeddings = &outputs.embeddings,
                                                .states = &outputs.states,
                                                .arena = arena,
                                                .outputs_scalar = uform_scalar_f32_k,
                                                .stats = &image_stats});
    outputs.truncated = image_stats.truncated, outputs.tokens = image_stats.tokens;
    // A build without host decoders refuses every encoded file, leaving the decoded paths to check.
    bool const decodes_files = outputs.outcome.status != uform_status_unsupported_k;
    if (decodes_files) {
        check_media(&outputs, 2, &model);
        expect_(outputs.states.offsets[1] * 2 == outputs.states.offsets[2]);
    }
    else expect_(outputs.embeddings.data == NULL && outputs.states.offsets == NULL);

    /* A JPEG cut after its frame header plans from the header, then fails whole as malformed, and
     *  the process lives. */
    uint8_t const cut_jpeg[] = {0xFF, 0xD8, 0xFF, 0xC0, 0x00, 0x11, 0x08, 0x00, 0x10, 0x00, 0x10,
                                0x03, 0x01, 0x22, 0x00, 0x02, 0x11, 0x01, 0x03, 0x11, 0x01};
    size_t const cut_offsets[] = {0, sizeof(cut_jpeg)};
    memset(&outputs, 0, sizeof(outputs)), memset(&image_stats, 0, sizeof(image_stats));
    uform_image_encode(&(uform_image_encode_t) {.model = handle,
                                                .executor = executor,
                                                .outcome = &outputs.outcome,
                                                .count = 1,
                                                .images = {cut_jpeg, cut_offsets, uform_scalar_u8_k},
                                                .embeddings = &outputs.embeddings,
                                                .states = &outputs.states,
                                                .arena = arena,
                                                .outputs_scalar = uform_scalar_f32_k,
                                                .stats = &image_stats});
    outputs.truncated = image_stats.truncated, outputs.tokens = image_stats.tokens;
    if (!decodes_files || outputs.outcome.status == uform_status_malformed_k)
        expect_(outputs.embeddings.data == NULL && outputs.states.offsets == NULL);
    else check_media(&outputs, 1, &model);

#if !UFORM_TEST_STUB // the stand-in decodes nothing, so it encodes any bytes
    /* Bytes of no image format fail as malformed, the format sniffer's note in place of the
     *  status's own name. */
    uint8_t const unrecognized[] = {'n', 'o', 't', ' ', 'a', 'n', ' ', 'i', 'm', 'a', 'g', 'e'};
    size_t const unrecognized_offsets[] = {0, sizeof(unrecognized)};
    memset(&outputs, 0, sizeof(outputs)), memset(&image_stats, 0, sizeof(image_stats));
    uform_image_encode(&(uform_image_encode_t) {.model = handle,
                                                .executor = executor,
                                                .outcome = &outputs.outcome,
                                                .count = 1,
                                                .images = {unrecognized, unrecognized_offsets, uform_scalar_u8_k},
                                                .embeddings = &outputs.embeddings,
                                                .states = &outputs.states,
                                                .arena = arena,
                                                .outputs_scalar = uform_scalar_f32_k,
                                                .stats = &image_stats});
    if (decodes_files) {
        expect_(outputs.outcome.status == uform_status_malformed_k);
        expect_(outputs.outcome.message && strcmp(outputs.outcome.message, "Unrecognized image format") == 0);
    }
#endif

    enum { pixel_bytes = 16 * 8 * 3 + 5 * 3 * 3 };
    uint8_t pixels[pixel_bytes];
    for (size_t byte = 0; byte != pixel_bytes; ++byte) pixels[byte] = (uint8_t)(byte * 7);
    size_t const pixel_offsets[] = {0, 16 * 8 * 3, pixel_bytes};
    uint32_t const heights[] = {8, 3}, widths[] = {16, 5};
    memset(&outputs, 0, sizeof(outputs)), memset(&image_stats, 0, sizeof(image_stats));
    uform_image_encode_pixels_t encode_pixels = {.model = handle,
                                                 .executor = executor,
                                                 .outcome = &outputs.outcome,
                                                 .count = 2,
                                                 .pixels = {pixels, pixel_offsets, uform_scalar_u8_k},
                                                 .heights = heights,
                                                 .widths = widths,
                                                 .embeddings = &outputs.embeddings,
                                                 .states = &outputs.states,
                                                 .arena = arena,
                                                 .outputs_scalar = uform_scalar_f32_k,
                                                 .stats = &image_stats};
    uform_image_encode_pixels(&encode_pixels);
    outputs.truncated = image_stats.truncated, outputs.tokens = image_stats.tokens;
    check_media(&outputs, 2, &model);
    uint32_t const wrong_heights[] = {8, 4};
    encode_pixels.heights = wrong_heights;
    uform_image_encode_pixels(&encode_pixels);
    expect_(outputs.outcome.status == uform_status_shape_mismatch_k);
    expect_(outputs.embeddings.data == NULL && outputs.states.offsets == NULL);

    uint8_t clip[300];
    for (size_t byte = 0; byte != sizeof(clip); ++byte) clip[byte] = (uint8_t)(byte * 13);
    size_t const clip_offsets[] = {0, sizeof(clip)};
    uform_video_encode_stats_t video_stats = {0};
    memset(&outputs, 0, sizeof(outputs));
    uform_video_encode(&(uform_video_encode_t) {.model = handle,
                                                .executor = executor,
                                                .outcome = &outputs.outcome,
                                                .count = 1,
                                                .videos = {clip, clip_offsets, uform_scalar_u8_k},
                                                .embeddings = &outputs.embeddings,
                                                .states = &outputs.states,
                                                .arena = arena,
                                                .outputs_scalar = uform_scalar_f32_k,
                                                .stats = &video_stats});
    outputs.truncated = video_stats.truncated, outputs.tokens = video_stats.tokens;
    if (!decodes_files || outputs.outcome.status == uform_status_malformed_k)
        expect_(outputs.embeddings.data == NULL && outputs.states.offsets == NULL);
    else check_media(&outputs, 1, &model);

    uint8_t recording[wav_bytes];
    size_t const recording_offsets[] = {0, write_wav(recording)};
    uform_audio_encode_stats_t audio_stats = {0};
    memset(&outputs, 0, sizeof(outputs));
    uform_audio_encode(&(uform_audio_encode_t) {.model = handle,
                                                .executor = executor,
                                                .outcome = &outputs.outcome,
                                                .count = 1,
                                                .recordings = {recording, recording_offsets, uform_scalar_u8_k},
                                                .embeddings = &outputs.embeddings,
                                                .states = &outputs.states,
                                                .arena = arena,
                                                .outputs_scalar = uform_scalar_f32_k,
                                                .stats = &audio_stats});
    outputs.truncated = audio_stats.truncated, outputs.tokens = audio_stats.tokens;
    if (decodes_files) check_media(&outputs, 1, &model);
    else expect_(outputs.embeddings.data == NULL && outputs.states.offsets == NULL);

    enum { samples_count = 1000 + 300 };
    float samples[samples_count];
    for (size_t sample = 0; sample != samples_count; ++sample) samples[sample] = (float)(sample % 50) / 50.0f - 0.5f;
    size_t const sample_offsets[] = {0, 1000, samples_count};
    memset(&outputs, 0, sizeof(outputs)), memset(&audio_stats, 0, sizeof(audio_stats));
    uform_audio_encode_samples(
        &(uform_audio_encode_samples_t) {.model = handle,
                                         .executor = executor,
                                         .outcome = &outputs.outcome,
                                         .count = 2,
                                         .samples = {samples, sample_offsets, uform_scalar_f32_k},
                                         .sample_rate = 16000,
                                         .embeddings = &outputs.embeddings,
                                         .states = &outputs.states,
                                         .arena = arena,
                                         .outputs_scalar = uform_scalar_f32_k,
                                         .stats = &audio_stats});
    outputs.truncated = audio_stats.truncated, outputs.tokens = audio_stats.tokens;
    check_media(&outputs, 2, &model);
    expect_(outputs.states.offsets[1] >= outputs.states.offsets[2] - outputs.states.offsets[1]);

    uform_model_free(handle);
    uform_arena_free(arena);
    expect_(uform_executor_free(executor) == uform_status_success_k);
}

/** One text call's outputs, the arena they land in, and its outcome, which a deferred call
 *  holds until final. */
typedef struct text_outputs_t {
    uform_outcome_t outcome;
    uform_rows_view_t embeddings;
    uform_rows_tape_view_t states;
    uform_arena_t arena;
} text_outputs_t;

/** Submits a text encode of @p texts into @p outputs; its descriptor dies when this returns. */
static void encode_texts(uform_model_t model, uform_executor_t executor, uform_completion_t completion,
                         texts_t const* texts, text_outputs_t* outputs, uform_progress_t progress) {
    uform_text_encode(&(uform_text_encode_t) {.model = model,
                                              .executor = executor,
                                              .outcome = &outputs->outcome,
                                              .completion = completion,
                                              .count = texts->count,
                                              .texts = tape_of(texts),
                                              .embeddings = &outputs->embeddings,
                                              .states = &outputs->states,
                                              .arena = outputs->arena,
                                              .outputs_scalar = uform_scalar_f32_k,
                                              .progress = progress});
}

/** Whether two encodes of the same texts landed the same bytes. */
static bool same_outputs(text_outputs_t const* first, text_outputs_t const* second, size_t count) {
    size_t const rows = first->states.offsets[count];
    return rows == second->states.offsets[count] &&
           memcmp(first->embeddings.data, second->embeddings.data, count * first->embeddings.stride) == 0 &&
           memcmp(first->states.rows.data, second->states.rows.data, rows * first->states.rows.stride) == 0 &&
           memcmp(first->states.offsets, second->states.offsets, (count + 1) * sizeof(size_t)) == 0;
}

static void drain(uform_executor_t executor) {
    uform_outcome_t outcome = {0};
    uform_executor_drain(&(uform_executor_drain_t) {.executor = executor, .outcome = &outcome});
    expect_at(!outcome.message, __FILE__, __LINE__, outcome.message);
}

/** An executor refuses to be freed while an arena made with it lives, a RAM one included, and
 *  frees once that arena is gone. */
static void test_executor_free_with_live_arena(void) {
    uform_executor_t const executor = create_executor();
    uform_arena_t const arena = create_arena(executor);
    expect_(uform_executor_free(executor) == uform_status_contended_k);
    expect_(uform_arena_free(arena) == uform_status_success_k);
    expect_(uform_executor_free(executor) == uform_status_success_k);
}

/** A deferred call finishes on the executor's own thread while its caller only polls, lands what a
 *  blocking one does though its descriptor died at submission, and refuses every other call on its
 *  executor meanwhile; without an executor a call is final when it returns. */
static void test_blocking_and_deferred(void) {
    uform_executor_t const executor = create_executor();
    uform_model_description_t description = {0};
    uform_model_t const model = load_model(executor, &description);
    size_t const lengths[] = {3, 40, 17, 1};
    texts_t texts;
    fill_texts(&texts, 4, lengths);
    text_outputs_t blocking = {.arena = create_arena(NULL)};
    text_outputs_t deferred = {.arena = create_arena(executor)};
    uform_progress_t const silent = {0};

    encode_texts(model, executor, uform_completion_blocking_k, &texts, &blocking, silent);
    expect_at(!blocking.outcome.message, __FILE__, __LINE__, blocking.outcome.message);

    encode_texts(model, executor, uform_completion_deferred_k, &texts, &deferred, silent);
    expect_(uform_outcome_wait(&deferred.outcome) == uform_status_success_k);
    expect_(same_outputs(&blocking, &deferred, texts.count));
    encode_texts(model, executor, uform_completion_blocking_k, &texts, &blocking, silent);
    expect_(!blocking.outcome.message);

    // An arena made by an executor serves a call without one, since both compute on the host.
    encode_texts(model, NULL, uform_completion_deferred_k, &texts, &deferred, silent);
    expect_(deferred.outcome.status == uform_status_success_k);
    expect_(same_outputs(&blocking, &deferred, texts.count));

    // A drain waits for the call holding the executor, then parks its threads until the next call.
    encode_texts(model, executor, uform_completion_deferred_k, &texts, &deferred, silent);
    drain(executor);
    expect_(deferred.outcome.status == uform_status_success_k);
    encode_texts(model, executor, uform_completion_blocking_k, &texts, &blocking, silent);
    expect_(!blocking.outcome.message);

    // Freeing the executor waits for the call holding it, over host pages no executor made.
    expect_(uform_arena_free(deferred.arena) == uform_status_success_k);
    deferred.arena = create_arena(NULL);
    encode_texts(model, executor, uform_completion_deferred_k, &texts, &deferred, silent);
    expect_(uform_executor_free(executor) == uform_status_success_k);
    expect_(deferred.outcome.status == uform_status_success_k);
    expect_(same_outputs(&blocking, &deferred, texts.count));

    uform_arena_free(deferred.arena);
    uform_arena_free(blocking.arena);
    uform_model_free(model);
}

#if !UFORM_TEST_STUB

/** How long a check waits for what it expects to happen. A passing test never waits it out, so it
 *  bounds hangs rather than speed, and leaves room for sanitizers and emulated CPUs. */
static time_t const wait_limit_seconds = 120;

/** Holds the call reporting to it at its first progress report until the test opens it, so a
 *  deferred call stays in flight while the test probes the executor and the arena it holds. */
typedef struct gate_t {

    /** The progress report the call stopped at, written before @ref reached. */
    size_t completed, total;

    /** Set by the held call once it reaches the gate. */
    atomic_int reached;

    /** Set by the test to let the call go on. */
    atomic_int opened;
} gate_t;

static uform_continuation_t wait_at_gate(size_t completed, size_t total, void* state) {
    gate_t* const gate = (gate_t*)state;
    gate->completed = completed, gate->total = total;
    atomic_store(&gate->reached, 1);
    time_t const deadline = time(NULL) + wait_limit_seconds;
    while (!atomic_load(&gate->opened) && time(NULL) < deadline) {}
    return uform_continue_k;
}

/** While a deferred call runs on the executor's own thread, its executor and its arena refuse
 *  every other call; once the gate opens, polling alone sees the call land what a blocking one
 *  does, and both serve again. */
static void test_held_calls(void) {
    uform_executor_t const executor = create_executor();
    uform_model_description_t description = {0};
    uform_model_t const model = load_model(executor, &description);
    size_t const lengths[] = {3, 40, 17, 1};
    texts_t texts;
    fill_texts(&texts, 4, lengths);
    text_outputs_t blocking = {.arena = create_arena(executor)};
    text_outputs_t deferred = {.arena = create_arena(executor)};
    uform_progress_t const silent = {0};
    encode_texts(model, executor, uform_completion_blocking_k, &texts, &blocking, silent);
    expect_at(!blocking.outcome.message, __FILE__, __LINE__, blocking.outcome.message);

    gate_t gate = {0, 0, 0, 0};
    encode_texts(model, executor, uform_completion_deferred_k, &texts, &deferred,
                 (uform_progress_t) {wait_at_gate, &gate, 1});
    time_t const deadline = time(NULL) + wait_limit_seconds;
    while (!atomic_load(&gate.reached) && time(NULL) < deadline) {}
    expect_(atomic_load(&gate.reached));
    expect_(gate.completed <= gate.total && gate.total == texts.count);
    expect_(uform_outcome_status(&deferred.outcome) == uform_status_deferred_k);

    text_outputs_t contender = {.arena = blocking.arena};
    encode_texts(model, executor, uform_completion_blocking_k, &texts, &contender, silent);
    expect_(contender.outcome.status == uform_status_contended_k);
    encode_texts(model, executor, uform_completion_deferred_k, &texts, &contender, silent);
    expect_(contender.outcome.status == uform_status_contended_k);
    text_outputs_t same_arena = {.arena = deferred.arena};
    encode_texts(model, NULL, uform_completion_blocking_k, &texts, &same_arena, silent);
    expect_(same_arena.outcome.status == uform_status_contended_k);
    expect_(same_arena.embeddings.data == NULL);

    atomic_store(&gate.opened, 1);
    time_t const landing_deadline = time(NULL) + wait_limit_seconds;
    while (uform_outcome_status(&deferred.outcome) == uform_status_deferred_k && time(NULL) < landing_deadline) {}
    expect_(deferred.outcome.status == uform_status_success_k);
    expect_(same_outputs(&blocking, &deferred, texts.count));
    encode_texts(model, NULL, uform_completion_blocking_k, &texts, &same_arena, silent);
    expect_(!same_arena.outcome.message);

    uform_arena_free(deferred.arena);
    uform_arena_free(blocking.arena);
    uform_model_free(model);
    expect_(uform_executor_free(executor) == uform_status_success_k);
}

#endif

/** An arena keeps its high-water mark, so a repeated call lands where the last did; inputs inside
 *  it are refused; a running call holds it, refusing another call meanwhile, until final; and it
 *  describes the host pages a RAM executor gave it. */
static void test_arenas(void) {
    uform_executor_t const executor = create_executor();
    uform_model_description_t description = {0};
    uform_model_t const model = load_model(executor, &description);
    size_t const lengths[] = {30, 12, 64};
    texts_t texts;
    fill_texts(&texts, 3, lengths);
    text_outputs_t outputs = {.arena = create_arena(executor)};
    uform_progress_t const silent = {0};

    encode_texts(model, executor, uform_completion_blocking_k, &texts, &outputs, silent);
    expect_at(!outputs.outcome.message, __FILE__, __LINE__, outputs.outcome.message);
    void const* const first_landing = outputs.embeddings.data;
    encode_texts(model, executor, uform_completion_blocking_k, &texts, &outputs, silent);
    expect_at(!outputs.outcome.message, __FILE__, __LINE__, outputs.outcome.message);
    expect_(outputs.embeddings.data == first_landing);

    // The previous call's offsets live in the arena the next call rewinds.
    uform_outcome_t outcome = {0};
    uform_rows_view_t embeddings = {0};
    uform_text_encode(&(uform_text_encode_t) {.model = model,
                                              .executor = executor,
                                              .outcome = &outcome,
                                              .count = texts.count,
                                              .texts = {texts.bytes, outputs.states.offsets, uform_scalar_u8_k},
                                              .embeddings = &embeddings,
                                              .arena = outputs.arena,
                                              .outputs_scalar = uform_scalar_f32_k});
    expect_(outcome.status == uform_status_invalid_argument_k);
    expect_(embeddings.data == NULL);

    uform_arena_description_t placement = {uform_residency_unknown_k, 0, uform_gpu_vendor_apple_k, 0,
                                           uform_host_access_refused_k};
    uform_arena_describe(
        &(uform_arena_describe_t) {.arena = outputs.arena, .outcome = &outcome, .description = &placement});
    expect_(outcome.status == uform_status_success_k && placement.residency == uform_residency_ram_k &&
            placement.device == -1 && placement.vendor == uform_gpu_vendor_unknown_k &&
            placement.vendor_ordinal == -1 && placement.host_access == uform_host_access_readable_k);
    uform_arena_describe(&(uform_arena_describe_t) {.arena = NULL, .outcome = &outcome, .description = &placement});
    expect_(outcome.status == uform_status_uninitialized_k);
    uform_arena_describe(&(uform_arena_describe_t) {.arena = outputs.arena, .outcome = &outcome, .description = NULL});
    expect_(outcome.status == uform_status_invalid_argument_k);

    // An open export refuses every call naming the arena and its free, until the last one closes.
    expect_(uform_arena_export_open(NULL) == uform_status_invalid_argument_k);
    expect_(uform_arena_export_open(outputs.arena) == uform_status_success_k);
    expect_(uform_arena_export_open(outputs.arena) == uform_status_success_k);
    encode_texts(model, executor, uform_completion_blocking_k, &texts, &outputs, silent);
    expect_(outputs.outcome.status == uform_status_contended_k);
    expect_(uform_arena_free(outputs.arena) == uform_status_contended_k);
    uform_arena_describe(
        &(uform_arena_describe_t) {.arena = outputs.arena, .outcome = &outcome, .description = &placement});
    expect_(outcome.status == uform_status_success_k);
    uform_arena_export_close(outputs.arena);
    expect_(uform_arena_free(outputs.arena) == uform_status_contended_k);
    uform_arena_export_close(outputs.arena);
    uform_arena_export_close(outputs.arena); // a close past the last open leaves the count at zero
    uform_arena_export_close(NULL);
    encode_texts(model, executor, uform_completion_blocking_k, &texts, &outputs, silent);
    expect_at(!outputs.outcome.message, __FILE__, __LINE__, outputs.outcome.message);

    expect_(uform_arena_free(outputs.arena) == uform_status_success_k);
    uform_model_free(model);
    expect_(uform_executor_free(executor) == uform_status_success_k);
}

/** What a progress handler heard over one call, and the report at which it stops the call;
 *  zero never stops. */
typedef struct progress_reports_t {
    size_t calls;
    size_t completed;
    size_t total;
    bool ascending;
    size_t cancel_at;
} progress_reports_t;

static uform_continuation_t record_progress(size_t completed, size_t total, void* state) {
    progress_reports_t* const reports = (progress_reports_t*)state;
    reports->ascending = reports->ascending && completed > reports->completed && completed <= total;
    reports->calls++;
    reports->completed = completed;
    reports->total = total;
    return reports->calls == reports->cancel_at ? uform_cancel_k : uform_continue_k;
}

/** A handler asking for four reports hears four, in order, the last covering the batch; zero steps
 *  derive ten; a handler stopping the call at its first report cancels it with every output left
 *  null, blocking or deferred. */
static void test_progress(void) {
    uform_executor_t const executor = create_executor();
    uform_model_description_t description = {0};
    uform_model_t const model = load_model(executor, &description);
    size_t lengths[texts_capacity];
    for (size_t item = 0; item != texts_capacity; ++item) lengths[item] = 3 + item;
    texts_t texts;
    fill_texts(&texts, 13, lengths);
    text_outputs_t outputs = {.arena = create_arena(executor)};

    progress_reports_t reports = {0, 0, 0, true, 0};
    encode_texts(model, executor, uform_completion_blocking_k, &texts, &outputs,
                 (uform_progress_t) {record_progress, &reports, 4});
    expect_at(!outputs.outcome.message, __FILE__, __LINE__, outputs.outcome.message);
    expect_(reports.ascending && reports.calls == 4);
    expect_(reports.completed == texts.count && reports.total == texts.count);

    reports = (progress_reports_t) {0, 0, 0, true, 0};
    encode_texts(model, executor, uform_completion_blocking_k, &texts, &outputs,
                 (uform_progress_t) {record_progress, &reports, 0});
    expect_(reports.ascending && reports.calls == 10);

    reports = (progress_reports_t) {0, 0, 0, true, 1};
    encode_texts(model, executor, uform_completion_blocking_k, &texts, &outputs,
                 (uform_progress_t) {record_progress, &reports, 4});
    expect_(outputs.outcome.status == uform_status_cancelled_k);
    expect_(reports.calls == 1);
    expect_(outputs.embeddings.data == NULL && outputs.states.offsets == NULL);

    reports = (progress_reports_t) {0, 0, 0, true, 1};
    encode_texts(model, executor, uform_completion_deferred_k, &texts, &outputs,
                 (uform_progress_t) {record_progress, &reports, 4});
    expect_(uform_outcome_wait(&outputs.outcome) == uform_status_cancelled_k);
    expect_(outputs.embeddings.data == NULL);

    uform_arena_free(outputs.arena);
    uform_model_free(model);
    expect_(uform_executor_free(executor) == uform_status_success_k);
}

#if !UFORM_TEST_STUB

/** Loads @p header as a safetensors file over @p payload_bytes zeros, answering the outcome. */
static uform_outcome_t load_forged(uform_executor_t executor, char const* header, size_t payload_bytes) {
    size_t const header_bytes = strlen(header), header_length = (header_bytes + 7) / 8 * 8;
    size_t const length = 8 + header_length + payload_bytes;
    uint8_t* const buffer = (uint8_t*)calloc(length, 1);
    expect_(buffer != NULL);
    for (size_t byte = 0; byte != 8; ++byte) buffer[byte] = (uint8_t)((uint64_t)header_length >> (byte * 8));
    memset(buffer + 8, ' ', header_length);
    memcpy(buffer + 8, header, header_bytes);
    uform_outcome_t outcome = {0};
    uform_model_t model = NULL;
    uform_model_load_ram_buffer(&(uform_model_load_ram_buffer_t) {.model = &model,
                                                                  .executor = executor,
                                                                  .outcome = &outcome,
                                                                  .buffer = {buffer, length},
                                                                  .quantization = uform_scalar_bf16_k});
    uform_model_free(model);
    free(buffer);
    return outcome;
}
#endif

/** Every refusal a caller can provoke: a null arena, a null model, no output named, a tape without
 *  storage, and GPU-only verbs given a RAM executor; against the engine also forged checkpoints and
 *  pixel counts that wrap. */
static void test_refusals(void) {
    uform_executor_t const executor = create_executor();
    uform_arena_t const arena = create_arena(executor);
    uform_model_description_t description = {0};
    uform_model_t const model = load_model(executor, &description);
    size_t const lengths[] = {4, 4};
    texts_t texts;
    fill_texts(&texts, 2, lengths);

    uform_outcome_t outcome = {0};
    uform_rows_view_t embeddings = {0};
    uform_rows_tape_view_t states = {0};
    uform_text_encode_t encode = {.model = model,
                                  .executor = executor,
                                  .outcome = &outcome,
                                  .count = texts.count,
                                  .texts = tape_of(&texts),
                                  .embeddings = &embeddings,
                                  .states = &states,
                                  .arena = NULL,
                                  .outputs_scalar = uform_scalar_f32_k};
    uform_text_encode(&encode);
    expect_(outcome.status == uform_status_invalid_argument_k);
    expect_(embeddings.data == NULL && states.offsets == NULL);

    encode.arena = arena, encode.embeddings = NULL, encode.states = NULL;
    uform_text_encode(&encode);
    expect_(outcome.status == uform_status_invalid_argument_k);

    encode.model = NULL;
    uform_text_encode(&encode);
    expect_(outcome.status == uform_status_uninitialized_k);

    encode.model = model, encode.embeddings = &embeddings, encode.texts.offsets = NULL;
    uform_text_encode(&encode);
    expect_(outcome.status == uform_status_invalid_argument_k);

    size_t const descending[] = {0, 8, 4};
    encode.texts.offsets = descending;
    uform_text_encode(&encode);
    expect_(outcome.status == uform_status_invalid_argument_k);
    encode.texts = tape_of(&texts);

#if !UFORM_TEST_STUB // the stand-in loads any readable bytes and multiplies pixels unchecked
    uint8_t const wrapped_pixels[26] = {0};
    size_t const wrapped_offsets[] = {0, sizeof(wrapped_pixels)};
    uint32_t const wrapped_heights[] = {2154230017u}, wrapped_widths[] = {2854344542u};
    uform_image_encode_pixels(
        &(uform_image_encode_pixels_t) {.model = model,
                                        .executor = executor,
                                        .outcome = &outcome,
                                        .count = 1,
                                        .pixels = {wrapped_pixels, wrapped_offsets, uform_scalar_u8_k},
                                        .heights = wrapped_heights,
                                        .widths = wrapped_widths,
                                        .embeddings = &embeddings,
                                        .arena = arena,
                                        .outputs_scalar = uform_scalar_f32_k});
    expect_(outcome.status == uform_status_shape_mismatch_k);

    expect_(load_forged(executor,
                        "{\"model.embedding\":{\"dtype\":\"BF16\",\"shape\":[1152921504606846976,8]," //
                        "\"data_offsets\":[0,0]}}",
                        0)
                .status == uform_status_malformed_k);
    char const geometry[] =
        "{\"__metadata__\":{\"depth\":\"1\",\"hidden\":\"8\",\"ffn_expansion\":\"1\",\"head_width\":\"8\"," //
        "\"rms_epsilon\":\"0.00001\",\"rope_temporal\":\"4\",\"rope_height\":\"2\",\"rope_width\":\"2\","   //
        "\"rope_base\":\"10000\",\"context\":\"4\",\"image_patch_side\":\"0\",\"image_channels\":\"1\","    //
        "\"image_pixel_mean\":\"0.5\",\"image_pixel_std\":\"0.5\",\"video_patch_side\":\"1\","              //
        "\"video_tube_frames\":\"%s\",\"video_frame_stride\":\"1\",\"video_channels\":\"2\","               //
        "\"video_pixel_mean\":\"0.5,0.5\",\"video_pixel_std\":\"0.5,0.5\",\"audio_chunk_samples\":\"0\","   //
        "\"audio_sample_rate\":\"16000\"},"                                                                 //
        "\"model.embedding\":{\"dtype\":\"BF16\",\"shape\":[%zu,8],\"data_offsets\":[0,%zu]},"              //
        "\"model.blocks.0.gate_up\":{\"dtype\":\"BF16\",\"shape\":[16,8],\"data_offsets\":[%zu,%zu]},"      //
        "\"model.video_projection\":{\"dtype\":\"BF16\",\"shape\":[8,2],\"data_offsets\":[%zu,%zu]}}";
    char header[2048];
    size_t const short_vocabulary = 200, full_vocabulary = 258;
    snprintf(header, sizeof(header), geometry, "1", short_vocabulary, short_vocabulary * 16, short_vocabulary * 16,
             short_vocabulary * 16 + 256, short_vocabulary * 16 + 256, short_vocabulary * 16 + 288);
    expect_(load_forged(executor, header, short_vocabulary * 16 + 288).status == uform_status_malformed_k);
    // Two-channel tubes of 2^31+1 frames take 2^32+2 scalars, which 32 bits wrap to 2.
    snprintf(header, sizeof(header), geometry, "2147483649", full_vocabulary, full_vocabulary * 16,
             full_vocabulary * 16, full_vocabulary * 16 + 256, full_vocabulary * 16 + 256, full_vocabulary * 16 + 288);
    expect_(load_forged(executor, header, full_vocabulary * 16 + 288).status == uform_status_malformed_k);
#endif

    uform_mesh_t mesh = NULL;
    uform_mesh_bootstrap(&(uform_mesh_bootstrap_t) {.mesh = &mesh, .executor = executor, .outcome = &outcome});
    expect_(outcome.status == uform_status_device_k);
    expect_(mesh == NULL);
    uform_trainer_t trainer = NULL;
    uform_trainer_init_vram(&(uform_trainer_init_vram_t) {.trainer = &trainer,
                                                          .executor = executor,
                                                          .outcome = &outcome,
                                                          .depth = 2,
                                                          .hidden = 256,
                                                          .schedule = {.steps = 1}});
    expect_(outcome.status == uform_status_device_k);
    expect_(trainer == NULL);
    uform_trainer_step(&(uform_trainer_step_t) {.trainer = NULL, .executor = executor, .outcome = &outcome});
    expect_(outcome.status == uform_status_uninitialized_k);
    uform_trainer_save(&(uform_trainer_save_t) {.trainer = NULL, .executor = executor, .outcome = &outcome});
    expect_(outcome.status == uform_status_uninitialized_k);
    uform_mesh_describe(&(uform_mesh_describe_t) {.mesh = NULL, .outcome = &outcome, .description = NULL});
    expect_(outcome.status == uform_status_uninitialized_k);
    uform_model_describe(&(uform_model_describe_t) {.model = NULL, .executor = executor, .outcome = &outcome});
    expect_(outcome.status == uform_status_uninitialized_k);

    uform_model_t gpu_model = NULL;
    uform_model_load_vram(&(uform_model_load_vram_t) {.model = &gpu_model,
                                                      .executor = executor,
                                                      .outcome = &outcome,
                                                      .path = view_of(checkpoint_path),
                                                      .quantization = uform_scalar_bf16_k});
    expect_(outcome.status == uform_status_device_k);
    expect_(gpu_model == NULL);

    uform_model_free(model);
    uform_arena_free(arena);
    expect_(uform_executor_free(executor) == uform_status_success_k);
}

/** Writes a small stand-in checkpoint to the temporary directory, answering its path. */
static char const* write_scratch_checkpoint(void) {
    static char path[1024];
    char const* directory = getenv("TMPDIR");
    if (!directory || !*directory) directory = getenv("TEMP");
    if (!directory || !*directory) directory = "/tmp";
    snprintf(path, sizeof(path), "%s/uform-test-checkpoint.safetensors", directory);
    FILE* const file = fopen(path, "wb");
    expect_(file != NULL);
    char const contents[] = "{\"__metadata__\":{\"stand-in\":\"any readable bytes load\"}}";
    uint64_t const header_length = sizeof(contents) - 1;
    uint8_t length_bytes[8];
    for (size_t byte = 0; byte != 8; ++byte) length_bytes[byte] = (uint8_t)(header_length >> (byte * 8));
    expect_(fwrite(length_bytes, 1, 8, file) == 8 && fwrite(contents, 1, header_length, file) == header_length);
    fclose(file);
    return path;
}

/** The cosine of row @p i of @p texts and the one row of @p images, which come unnormalized. */
static float quickstart_cosine(uform_rows_view_t texts, size_t i, uform_rows_view_t images) {
    float const* text = (float const*)((char const*)texts.data + i * texts.stride);
    float const* image = (float const*)images.data;
    float dot = 0, texts_norm = 0, images_norm = 0;
    for (size_t k = 0; k != texts.columns; ++k)
        dot += text[k] * image[k], texts_norm += text[k] * text[k], images_norm += image[k] * image[k];
    return dot / sqrtf(texts_norm * images_norm);
}

/** The Quickstart of @c c/README.md , line for line, but loading the suite's checkpoint in place
 *  of @c model.safetensors , taking the photo from @ref write_png in place of reading @c cat.png ,
 *  and expecting each status where the README chains them. */
static void test_quickstart(void) {
    uform_outcome_t outcome = {0};
    uform_executor_t executor = NULL;
    uform_model_t model = NULL;
    uform_arena_t texts_arena = NULL, images_arena = NULL;
    char const* path = checkpoint_path;
    uform_executor_for_ram(&(uform_executor_for_ram_t) {.executor = &executor, .outcome = &outcome});
    if (!outcome.status)
        uform_model_load_ram(&(uform_model_load_ram_t) {.model = &model,
                                                        .executor = executor,
                                                        .outcome = &outcome,
                                                        .path = {(uint8_t const*)path, strlen(path)},
                                                        .quantization = uform_scalar_bf16_k});
    if (!outcome.status)
        uform_arena_init(&(uform_arena_init_t) {.arena = &texts_arena, .executor = executor, .outcome = &outcome});
    if (!outcome.status)
        uform_arena_init(&(uform_arena_init_t) {.arena = &images_arena, .executor = executor, .outcome = &outcome});

    char const* captions[] = {"a cat on a mat", "a sunrise over the bay"};
    char const caption_bytes[] = "a cat on a mata sunrise over the bay";
    size_t const caption_offsets[] = {0, 14, 36};
    static unsigned char photo[png_capacity];
    size_t const photo_offsets[] = {0, write_png(photo)};

    uform_rows_view_t texts = {0}, images = {0};
    if (!outcome.status)
        uform_text_encode(&(uform_text_encode_t) {.model = model,
                                                  .executor = executor,
                                                  .outcome = &outcome,
                                                  .count = 2,
                                                  .texts = {caption_bytes, caption_offsets, uform_scalar_u8_k},
                                                  .embeddings = &texts,
                                                  .arena = texts_arena,
                                                  .outputs_scalar = uform_scalar_f32_k});
    if (!outcome.status)
        uform_image_encode(&(uform_image_encode_t) {.model = model,
                                                    .executor = executor,
                                                    .outcome = &outcome,
                                                    .count = 1,
                                                    .images = {photo, photo_offsets, uform_scalar_u8_k},
                                                    .embeddings = &images,
                                                    .arena = images_arena,
                                                    .outputs_scalar = uform_scalar_f32_k});

    // A build without host decoders refuses the photo, which the README's build decodes.
    bool const decodes_files = outcome.status != uform_status_unsupported_k;
    expect_at(!outcome.status || !decodes_files, __FILE__, __LINE__,
              outcome.message ? outcome.message : uform_status_name(outcome.status));
    size_t const best = !decodes_files ? 0 : quickstart_cosine(texts, 1, images) > quickstart_cosine(texts, 0, images);
    float const cosine = decodes_files ? quickstart_cosine(texts, best, images) : 0;
    expect_(strlen(captions[best]) != 0 && cosine == cosine);

    uform_arena_free(images_arena), uform_arena_free(texts_arena);
    uform_model_free(model);
    expect_(uform_executor_free(executor) == uform_status_success_k);
}

/** Prints the version line, then the capabilities as "- Compiled for:" and "- This machine:". */
static void print_machine(FILE* stream) {
    fprintf(stream, "UForm %s\n- Compiled for: %s\n- This machine: %s\n", uform_version_name(),
            uform_capabilities_compiled(), uform_capabilities_available());
}

int main(int argc, char* argv[]) {
    print_machine(stdout);
    // ctest hides passing output, so the job summary is where a green run shows what it covered
    char const* const summary_path = getenv("GITHUB_STEP_SUMMARY");
    FILE* const summary = summary_path ? fopen(summary_path, "a") : NULL;
    if (summary) print_machine(summary), fclose(summary);
    start_test_run(argc, argv);

    char const* const environment_checkpoint = getenv("UFORM_TEST_CHECKPOINT");
    char const* scratch = NULL;
    if (argc > 1) checkpoint_path = argv[1];
    else if (environment_checkpoint && *environment_checkpoint) checkpoint_path = environment_checkpoint;
    else checkpoint_path = scratch = write_scratch_checkpoint();
    printf("- Checkpoint: %s\n", checkpoint_path);

    start_test_case("test_library_probes"), test_library_probes(), finish_test_case(NULL);
    start_test_case("test_executors"), test_executors(), finish_test_case(NULL);
    start_test_case("test_model_load"), test_model_load(), finish_test_case(NULL);
    start_test_case("test_text_shapes"), test_text_shapes(), finish_test_case(NULL);
    start_test_case("test_heads"), test_heads(), finish_test_case(NULL);
    start_test_case("test_media_shapes"), test_media_shapes(), finish_test_case(NULL);
    start_test_case("test_executor_free_with_live_arena"), test_executor_free_with_live_arena(), finish_test_case(NULL);
    start_test_case("test_blocking_and_deferred"), test_blocking_and_deferred(), finish_test_case(NULL);
#if !UFORM_TEST_STUB
    start_test_case("test_held_calls"), test_held_calls(), finish_test_case(NULL);
#endif
    start_test_case("test_arenas"), test_arenas(), finish_test_case(NULL);
    start_test_case("test_progress"), test_progress(), finish_test_case(NULL);
    start_test_case("test_refusals"), test_refusals(), finish_test_case(NULL);
    start_test_case("test_quickstart"), test_quickstart(), finish_test_case(NULL);

    if (scratch) remove(scratch);
    return finish_test_run();
}

#undef expect_
