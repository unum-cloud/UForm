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
#ifndef UFORM_TEST_STUB
#define UFORM_TEST_STUB 0
#endif

#if !UFORM_TEST_STUB   // the stand-in runs every call on its caller, built as C99 without threads to race
#include <stdatomic.h> // `atomic_int` - the gate a deferred call waits at
#endif
#include <stdbool.h> // `bool`
#include <stdint.h>  // `uint8_t`, `uint32_t`
#include <stdio.h>   // `printf`, `fopen`, `fwrite`, `remove`
#include <stdlib.h>  // `exit`, `getenv`
#include <string.h>  // `memcmp`, `memset`, `strcmp`, `strlen`, `strstr`

#include <forkunion/forkunion.h> // `fu_pool_new`, `fu_topology_new` - the pool a lender spawns

#include "uform/uform.h"

static void expect(bool must_be_true, char const* message) {
    if (must_be_true) return;
    printf("Assert: %s\n", message ? message : "C unit test failed");
    exit(-1);
}

/** A substring of the tests to run, from @c UFORM_FILTER; @c NULL runs every test. */
static char const* test_filter = NULL;

/** Whether the test @p name passes @c UFORM_FILTER. */
static bool selected(char const* name) { return !test_filter || strstr(name, test_filter) != NULL; }

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
    expect(!outcome.message, outcome.message);
    expect(executor != NULL, "A RAM executor must be created");
    return executor;
}

static uform_arena_t create_arena(uform_executor_t executor) {
    uform_outcome_t outcome = {0};
    uform_arena_t arena = NULL;
    uform_arena_init(&(uform_arena_init_t) {.arena = &arena, .executor = executor, .outcome = &outcome});
    expect(!outcome.message, outcome.message);
    expect(arena != NULL, "An arena must be created");
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
    expect(!outcome.message, outcome.message);
    expect(model != NULL, "A model must be loaded");
    uform_model_describe(&(uform_model_describe_t) {
        .model = model, .executor = executor, .outcome = &outcome, .description = description});
    expect(!outcome.message, outcome.message);
    expect(description->dimensions > 0 && description->context > 0, "A model reports its geometry");
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
    expect(count <= texts_capacity, "Too many texts");
    texts->count = count;
    texts->offsets[0] = 0;
    for (size_t item = 0; item != count; ++item) {
        size_t const begin = texts->offsets[item];
        expect(begin + lengths[item] <= text_bytes_capacity, "Too many text bytes");
        memset(texts->bytes + begin, 'a' + (int)(item % 26), lengths[item]);
        texts->offsets[item + 1] = begin + lengths[item];
    }
}

static uform_tape_view_t tape_of(texts_t const* texts) {
    return (uform_tape_view_t) {texts->bytes, texts->offsets, uform_scalar_u8_k};
}

/** Checks a run of output rows: present, of the model's width, written packed as @p scalar. */
static void check_rows(uform_rows_view_t const* rows, size_t dimensions, uform_scalar_t scalar) {
    expect(rows->data != NULL, "Rows must land in the arena");
    expect(rows->columns == dimensions, "Rows span the model's dimensions");
    expect(rows->scalar == scalar, "Rows land as the scalar asked for");
    expect(rows->stride == dimensions * scalar_bytes(scalar), "Rows are written packed");
}

/** Checks per-token states against the call's stats: offsets from zero, never decreasing, no item
 *  past the context, and the last offset the tokens counted. */
static void check_states(uform_rows_tape_view_t const* states, size_t count, uform_model_description_t const* model,
                         uform_scalar_t scalar, size_t tokens) {
    check_rows(&states->rows, model->dimensions, scalar);
    expect(states->offsets != NULL, "States carry their offsets");
    expect(states->offsets[0] == 0, "The offsets start at zero");
    for (size_t item = 0; item != count; ++item) {
        expect(states->offsets[item + 1] >= states->offsets[item], "The offsets never decrease");
        expect(states->offsets[item + 1] - states->offsets[item] <= model->context, "No item passes the context");
    }
    expect(states->offsets[count] == tokens, "The last offset is the tokens counted");
}

/** The library-wide probes, which take no handle and cannot fail, and every name
 *  table read back. */
static void test_library_probes(void) {
    printf("Test: Library Probes\n");
    expect(uform_capabilities_compiled() != NULL, "The compiled capabilities must be reported");
    expect(uform_capabilities_available() != NULL, "The available capabilities must be reported");

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
        uform_status_config_mismatch_k,
        uform_status_contended_k,
        uform_status_cancelled_k,
        uform_status_unsupported_k,
        uform_status_invalid_argument_k,
    };
    for (size_t position = 0; position != sizeof(statuses) / sizeof(statuses[0]); ++position) {
        uform_status_t read = uform_status_internal_k;
        expect(
            uform_status_from_name(view_of(uform_status_name(statuses[position])), &read) == uform_status_success_k &&
                read == statuses[position],
            "Every status reads back from its name");
    }
    expect(strcmp(uform_status_name(uform_status_deferred_k), "deferred") == 0, "The non-final status spells deferred");
    expect(strcmp(uform_status_name(uform_status_out_of_memory_k), "out-of-memory") == 0, "Names are dash-separated");
    expect(strcmp(uform_status_name((uform_status_t)6), "unrecognized") == 0, "An unused value spells unrecognized");
    uform_status_t untouched_status = uform_status_io_k;
    expect(uform_status_from_name(view_of("no-such"), &untouched_status) == uform_status_invalid_argument_k &&
               untouched_status == uform_status_io_k,
           "An unrecognized status name fails and writes nothing");

    uform_scalar_t const scalars[] = {
        uform_scalar_f64_k,  uform_scalar_f32_k,  uform_scalar_bf16_k, uform_scalar_f16_k, uform_scalar_e5m2_k,
        uform_scalar_e4m3_k, uform_scalar_e3m2_k, uform_scalar_e2m3_k, uform_scalar_i8_k,  uform_scalar_i4x2_k,
        uform_scalar_i32_k,  uform_scalar_u8_k,   uform_scalar_u4x2_k, uform_scalar_u1x8_k};
    for (size_t position = 0; position != sizeof(scalars) / sizeof(scalars[0]); ++position) {
        uform_scalar_t read = uform_scalar_unknown_k;
        expect(uform_scalar_from_name(view_of(uform_scalar_name(scalars[position])), &read) == uform_status_success_k &&
                   read == scalars[position],
               "Every scalar reads back from its name");
    }
    expect(strcmp(uform_scalar_name(uform_scalar_f32_k), "f32") == 0, "A scalar spells its name");
    uform_scalar_t untouched_scalar = uform_scalar_f32_k;
    expect(uform_scalar_from_name(view_of("f33"), &untouched_scalar) == uform_status_invalid_argument_k &&
               untouched_scalar == uform_scalar_f32_k,
           "An unrecognized scalar fails and writes nothing");

    uform_mood_t mood = uform_mood_parked_k;
    expect(uform_mood_from_name(view_of("spinning"), &mood) == uform_status_success_k && mood == uform_mood_spinning_k,
           "A mood reads back from its name");
    expect(uform_mood_from_name(view_of("sleepy"), &mood) == uform_status_invalid_argument_k &&
               mood == uform_mood_spinning_k,
           "An unrecognized mood fails rather than reading as parked");
    expect(strcmp(uform_mood_name(uform_mood_parked_k), "parked") == 0, "A mood spells its name");
    uform_modality_t const modalities[] = {uform_modality_text_k, uform_modality_image_k, uform_modality_video_k,
                                           uform_modality_audio_k};
    for (size_t position = 0; position != 4; ++position) {
        uform_modality_t read = uform_modality_audio_k;
        expect(uform_modality_from_name(view_of(uform_modality_name(modalities[position])), &read) ==
                       uform_status_success_k &&
                   read == modalities[position],
               "Every modality reads back from its name");
    }
    expect(strcmp(uform_multimem_tier_name(uform_multimem_tier_peer_read_k), "peer-read") == 0,
           "A multimem tier spells its name");
    uform_multimem_tier_t tier = uform_multimem_tier_switch_reduced_k;
    expect(uform_multimem_tier_from_name(view_of("peer-read"), &tier) == uform_status_success_k &&
               tier == uform_multimem_tier_peer_read_k,
           "A multimem tier reads back from its name");
    expect(uform_multimem_tier_from_name(view_of("ring"), &tier) == uform_status_invalid_argument_k &&
               tier == uform_multimem_tier_peer_read_k,
           "An unrecognized multimem tier fails rather than reading as switch-reduced");

    for (uform_cpu_class_t code = uform_cpu_class_serial_k; code <= uform_cpu_class_wasm64_extended_k; ++code) {
        uform_cpu_class_t read = uform_cpu_class_unknown_k;
        expect(uform_cpu_class_from_name(view_of(uform_cpu_class_name(code)), &read) == uform_status_success_k &&
                   read == code,
               "Every CPU class reads back from its name");
    }
    for (uform_gpu_class_t code = uform_gpu_class_nvidia_ampere_k; code <= uform_gpu_class_apple_metal10_k; ++code) {
        uform_gpu_class_t read = uform_gpu_class_unknown_k;
        expect(uform_gpu_class_from_name(view_of(uform_gpu_class_name(code)), &read) == uform_status_success_k &&
                   read == code,
               "Every GPU class reads back from its name");
    }
    expect(strcmp(uform_cpu_class_name(uform_cpu_class_intel_haswell_k), "intel-haswell") == 0,
           "A CPU class spells its name as USearch does");
    uform_gpu_class_t untouched_gpu_class = uform_gpu_class_nvidia_ada_k;
    expect(
        uform_gpu_class_from_name(view_of("nvidia-volta"), &untouched_gpu_class) == uform_status_invalid_argument_k &&
            untouched_gpu_class == uform_gpu_class_nvidia_ada_k,
        "An unrecognized GPU class fails and writes nothing");

    uform_outcome_t outcome = {0};
    uform_gpu_device_t devices[8];
    size_t devices_count = 0;
    uform_gpu_devices(&(uform_gpu_devices_t) {.outcome = &outcome,
                                              .devices = devices,
                                              .devices_capacity = 8,
                                              .devices_count = &devices_count});
    expect(!outcome.message, outcome.message);
    for (size_t position = 0; position != devices_count && position != 8; ++position) {
        printf("- GPU #%zu: %s %s, %u units, %llu MiB\n", position, devices[position].name,
               uform_gpu_class_name(devices[position].gpu_class), devices[position].compute_units,
               (unsigned long long)(devices[position].memory_bytes >> 20));
        expect(devices[position].vendor != uform_gpu_vendor_unknown_k, "A listed GPU names its vendor");
    }
    printf("Test: Library Probes - PASSED\n");
}

/** Every executor kind: RAM always, VRAM where the build and the machine have one, a borrowed pool
 *  nowhere yet; a drain of an idle executor succeeds, and every free takes @c NULL. */
static void test_executors(void) {
    printf("Test: Executors\n");
    uform_executor_t const executor = create_executor();
    uform_outcome_t outcome = {uform_status_io_k, "a stale message the call must overwrite"};
    uform_executor_drain(&(uform_executor_drain_t) {.executor = executor, .outcome = &outcome});
    expect(outcome.status == uform_status_success_k && outcome.message == NULL, "An idle drain succeeds clean");

    uform_executor_t gpu = NULL;
    uform_executor_for_vram(&(uform_executor_for_vram_t) {.executor = &gpu, .outcome = &outcome});
    expect(outcome.status == uform_status_success_k || outcome.status == uform_status_device_k,
           "A VRAM executor exists, or the build or machine lacks a GPU");
    expect((outcome.status == uform_status_success_k) == (gpu != NULL), "Only a created executor is written");
    uform_executor_free(gpu);

    uform_executor_free(NULL);
    uform_arena_free(NULL);
    uform_model_free(NULL);
    uform_mesh_free(NULL);
    uform_trainer_free(NULL);
    uform_executor_free(executor);
    printf("Test: Executors - PASSED\n");
}

/** A checkpoint loads from a path and from a buffer, in either quantization, and its description
 *  echoes the quantization asked for; another quantization and a missing file are refused,
 *  leaving no model. */
static void test_model_load(void) {
    printf("Test: Model Load\n");
    uform_executor_t const executor = create_executor();
    uform_model_description_t description = {0};
    uform_model_t const model = load_model(executor, &description);
    expect(description.quantization == uform_scalar_bf16_k, "A model runs in the quantization asked for");
    expect(description.cpu_class != uform_cpu_class_unknown_k, "A RAM model names the class it was packed for");
    expect(description.gpu_class == uform_gpu_class_unknown_k, "A RAM model names no GPU class");

    uform_outcome_t pin_outcome = {0};
    uform_model_t serial = NULL;
    uform_model_load_ram(&(uform_model_load_ram_t) {.model = &serial,
                                                    .executor = executor,
                                                    .outcome = &pin_outcome,
                                                    .path = view_of(checkpoint_path),
                                                    .ram = {.cpu_class = uform_cpu_class_serial_k},
                                                    .quantization = uform_scalar_bf16_k});
    expect(!pin_outcome.message, pin_outcome.message);
    uform_model_description_t serial_description = {0};
    uform_model_describe(&(uform_model_describe_t) {
        .model = serial, .executor = executor, .outcome = &pin_outcome, .description = &serial_description});
    expect(serial_description.cpu_class == uform_cpu_class_serial_k, "A pinned load packs for the pinned class");
    uform_model_free(serial);
    uform_model_t foreign = NULL;
    uform_cpu_class_t const foreign_class = description.cpu_class == uform_cpu_class_wasm64_extended_k
                                                ? uform_cpu_class_intel_diamond_k
                                                : uform_cpu_class_wasm64_extended_k;
    uform_model_load_ram(&(uform_model_load_ram_t) {.model = &foreign,
                                                    .executor = executor,
                                                    .outcome = &pin_outcome,
                                                    .path = view_of(checkpoint_path),
                                                    .ram = {.cpu_class = foreign_class},
                                                    .quantization = uform_scalar_bf16_k});
    expect(pin_outcome.status == uform_status_unsupported_k, "A class this CPU does not run is refused");
    expect(foreign == NULL, "A refused pin leaves no model");

    FILE* const file = fopen(checkpoint_path, "rb");
    expect(file != NULL, "The checkpoint must be readable");
    fseek(file, 0, SEEK_END);
    long const length = ftell(file);
    fseek(file, 0, SEEK_SET);
    uint8_t* const buffer = (uint8_t*)malloc((size_t)length);
    expect(buffer && fread(buffer, 1, (size_t)length, file) == (size_t)length, "The checkpoint must be read");
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
        expect(buffered == NULL, "A refused load leaves no model");
        quantization = uform_scalar_bf16_k;
        uform_model_load_ram_buffer(&(uform_model_load_ram_buffer_t) {.model = &buffered,
                                                                      .executor = executor,
                                                                      .outcome = &outcome,
                                                                      .buffer = {buffer, (size_t)length},
                                                                      .quantization = quantization});
    }
    expect(!outcome.message, outcome.message);
    free(buffer);
    uform_model_description_t buffered_description = {0};
    uform_model_describe(&(uform_model_describe_t) {
        .model = buffered, .executor = executor, .outcome = &outcome, .description = &buffered_description});
    expect(!outcome.message, outcome.message);
    expect(buffered_description.quantization == quantization, "A buffer loads in the quantization asked for");
    expect(buffered_description.dimensions == description.dimensions, "A buffer loads the geometry a path does");

    uform_model_t refused = NULL;
    uform_model_load_ram(&(uform_model_load_ram_t) {.model = &refused,
                                                    .executor = executor,
                                                    .outcome = &outcome,
                                                    .path = view_of(checkpoint_path),
                                                    .quantization = uform_scalar_f32_k});
    expect(outcome.status == uform_status_unsupported_k, "Blocks run in bf16 or e4m3 only");
    expect(refused == NULL, "A refused load leaves no model");
    uform_model_load_ram(&(uform_model_load_ram_t) {.model = &refused,
                                                    .executor = executor,
                                                    .outcome = &outcome,
                                                    .path = view_of("/no/such/uform/checkpoint.safetensors"),
                                                    .quantization = uform_scalar_bf16_k});
    expect(outcome.status == uform_status_io_k, "A missing checkpoint is an I/O failure");
    expect(refused == NULL, "A failed load leaves no model");
    char const nul_path[] = "tiny.safetensors\0.missing";
    uform_model_load_ram(&(uform_model_load_ram_t) {.model = &refused,
                                                    .executor = executor,
                                                    .outcome = &outcome,
                                                    .path = {(uint8_t const *)nul_path, sizeof(nul_path) - 1},
                                                    .quantization = uform_scalar_bf16_k});
    expect(outcome.status == uform_status_invalid_argument_k, "A checkpoint path holding a NUL is refused");
    expect(refused == NULL, "A refused path leaves no model");

    uform_model_free(buffered);
    uform_model_free(model);
    uform_executor_free(executor);
    printf("Test: Model Load - PASSED\n");
}

/** Text in every output scalar: one embedding per item, one state per byte up to the context, and
 *  an item past the context cut there and counted; a scalar no output takes is refused with every
 *  output left null. */
static void test_text_shapes(void) {
    printf("Test: Text Shapes\n");
    uform_executor_t const executor = create_executor();
    uform_arena_t const arena = create_arena(executor);
    uform_model_description_t model_description = {0};
    uform_model_t const model = load_model(executor, &model_description);
    size_t const context = model_description.context;
    expect(context < text_bytes_capacity - 16, "The test's long item must pass the context");
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
        expect(!outcome.message, outcome.message);
        check_rows(&embeddings, model_description.dimensions, scalars[position]);
        check_states(&states, texts.count, &model_description, scalars[position], stats.tokens);
        expect(stats.truncated == 1, "Only the item past the context is truncated");
        expect(stats.tokens == 1 + 5 + context, "Every byte is a token, up to the context");
        expect(states.offsets[1] == 1 && states.offsets[2] == 6, "Each item spans its bytes");
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
    expect(outcome.status == uform_status_unsupported_k, "Outputs land as floats");
    expect(embeddings.data == NULL, "A refused call leaves its outputs NULL");

    uform_model_free(model);
    uform_arena_free(arena);
    uform_executor_free(executor);
    printf("Test: Text Shapes - PASSED\n");
}

/** Token vectors and choices: with the heads, unit-length vectors on the states' offsets and
 *  probabilities summing to one over each item's choices, a group past the model's refused; without
 *  them, @c unsupported_k. */
static void test_heads(void) {
    printf("Test: Heads\n");
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
    if (!description.token_width) expect(outcome.status == uform_status_unsupported_k, "No token head, no vectors");
    else {
        expect(!outcome.message, outcome.message);
        expect(token_vectors.rows.columns == description.token_width, "Token vectors are token_width wide");
        for (size_t item = 0; item <= texts.count; ++item)
            expect(token_vectors.offsets[item] == states.offsets[item], "Token vectors share the states' offsets");
        for (size_t row = 0; row != token_vectors.offsets[texts.count]; ++row) {
            float const* const vector =
                (float const*)((char const*)token_vectors.rows.data + row * token_vectors.rows.stride);
            float squares = 0;
            for (size_t column = 0; column != description.token_width; ++column)
                squares += vector[column] * vector[column];
            expect(squares > 0.98f && squares < 1.02f, "Every token vector has unit length");
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
        expect(outcome.status == uform_status_unsupported_k, "No byte decoder, no choices");
        expect(probabilities.data == NULL, "A refused choice leaves its probabilities NULL");
    }
    else {
        expect(!outcome.message, outcome.message);
        float const* const values = (float const*)probabilities.data;
        for (size_t item = 0; item != texts.count; ++item) {
            expect(probabilities.offsets[item + 1] - probabilities.offsets[item] ==
                       choice_offsets[item + 1] - choice_offsets[item],
                   "One probability per choice");
            float sum = 0;
            for (size_t choice = probabilities.offsets[item]; choice != probabilities.offsets[item + 1]; ++choice)
                expect(values[choice] >= 0 && values[choice] <= 1, "Probabilities lie in [0,1]"),
                    sum += values[choice];
            expect(sum > 0.999f && sum < 1.001f, "Each item's probabilities sum to one");
        }
        uint32_t const past[] = {0, (uint32_t)description.choice_temperature_groups, 0};
        choose.temperature_groups = past;
        uform_text_choose(&choose);
        expect(outcome.status == uform_status_invalid_argument_k, "A group past the model's is refused");
    }

    uform_model_free(model);
    uform_arena_free(arena);
    uform_executor_free(executor);
    printf("Test: Heads - PASSED\n");
}

/** CRC-32 of @p length bytes, as PNG chunks carry it. */
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
    expect(!outputs->outcome.message, outputs->outcome.message);
    check_rows(&outputs->embeddings, model->dimensions, uform_scalar_f32_k);
    check_states(&outputs->states, count, model, uform_scalar_f32_k, outputs->tokens);
    expect(outputs->truncated <= count, "No more items are truncated than encoded");
    for (size_t item = 0; item != count; ++item)
        expect(outputs->states.offsets[item + 1] > outputs->states.offsets[item], "Every media item spans a token");
}

/** Every media verb lands one embedding per item and states whose offsets agree with its stats. A
 *  file the engine cannot decode fails whole as malformed, leaving its outputs null; the stand-in
 *  decodes nothing, so it encodes any bytes. */
static void test_media_shapes(void) {
    printf("Test: Media Shapes\n");
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
        expect(outputs.states.offsets[1] * 2 == outputs.states.offsets[2], "Equal images span equal tokens");
    }
    else expect(outputs.embeddings.data == NULL && outputs.states.offsets == NULL, "A refused call leaves outputs NULL");

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
        expect(outputs.embeddings.data == NULL && outputs.states.offsets == NULL, "A failed call leaves outputs NULL");
    else check_media(&outputs, 1, &model);

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
    expect(outputs.outcome.status == uform_status_shape_mismatch_k, "Pixels must fill their height and width");
    expect(outputs.embeddings.data == NULL && outputs.states.offsets == NULL, "A refused call leaves outputs NULL");

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
        expect(outputs.embeddings.data == NULL && outputs.states.offsets == NULL, "A failed call leaves outputs NULL");
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
    else expect(outputs.embeddings.data == NULL && outputs.states.offsets == NULL, "A refused call leaves outputs NULL");

    enum { samples_count = 1000 + 300 };
    float samples[samples_count];
    for (size_t sample = 0; sample != samples_count; ++sample) samples[sample] = (float)(sample % 50) / 50.0f - 0.5f;
    size_t const sample_offsets[] = {0, 1000, samples_count};
    memset(&outputs, 0, sizeof(outputs)), memset(&audio_stats, 0, sizeof(audio_stats));
    uform_audio_encode_samples(&(uform_audio_encode_samples_t) {.model = handle,
                                                                .executor = executor,
                                                                .outcome = &outputs.outcome,
                                                                .count = 2,
                                                                .samples = {samples, sample_offsets,
                                                                            uform_scalar_f32_k},
                                                                .sample_rate = 16000,
                                                                .embeddings = &outputs.embeddings,
                                                                .states = &outputs.states,
                                                                .arena = arena,
                                                                .outputs_scalar = uform_scalar_f32_k,
                                                                .stats = &audio_stats});
    outputs.truncated = audio_stats.truncated, outputs.tokens = audio_stats.tokens;
    check_media(&outputs, 2, &model);
    expect(outputs.states.offsets[1] >= outputs.states.offsets[2] - outputs.states.offsets[1],
           "A longer recording spans no fewer frames");

    uform_model_free(handle);
    uform_arena_free(arena);
    uform_executor_free(executor);
    printf("Test: Media Shapes - PASSED\n");
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
    expect(!outcome.message, outcome.message);
}

/** A deferred call finishes on the executor's own thread while its caller only polls, lands what a
 *  blocking one does though its descriptor died at submission, and refuses every other call on its
 *  executor meanwhile; without an executor a call is final when it returns. */
static void test_blocking_and_deferred(void) {
    printf("Test: Blocking and Deferred\n");
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
    expect(!blocking.outcome.message, blocking.outcome.message);

    encode_texts(model, executor, uform_completion_deferred_k, &texts, &deferred, silent);
    expect(uform_outcome_wait(&deferred.outcome) == uform_status_success_k, "A waited call finishes");
    expect(same_outputs(&blocking, &deferred, texts.count), "A waited call lands what a blocking one does");
    encode_texts(model, executor, uform_completion_blocking_k, &texts, &blocking, silent);
    expect(!blocking.outcome.message, "A final call lets go of its executor at once");

    // An arena made by an executor serves a call without one, since both compute on the host.
    encode_texts(model, NULL, uform_completion_deferred_k, &texts, &deferred, silent);
    expect(deferred.outcome.status == uform_status_success_k, "A call without an executor is final when it returns");
    expect(same_outputs(&blocking, &deferred, texts.count), "Every receiver agrees");

    // A drain waits for the call holding the executor, then parks its threads until the next call.
    encode_texts(model, executor, uform_completion_deferred_k, &texts, &deferred, silent);
    drain(executor);
    expect(deferred.outcome.status == uform_status_success_k, "A drain waits for the call holding the executor");
    encode_texts(model, executor, uform_completion_blocking_k, &texts, &blocking, silent);
    expect(!blocking.outcome.message, "A parked executor takes the next call");

    // Freeing the executor waits for the call holding it.
    encode_texts(model, executor, uform_completion_deferred_k, &texts, &deferred, silent);
    uform_executor_free(executor);
    expect(deferred.outcome.status == uform_status_success_k, "A free waits for the call holding the executor");
    expect(same_outputs(&blocking, &deferred, texts.count), "A call outliving its submission lands its outputs");

    uform_arena_free(deferred.arena);
    uform_arena_free(blocking.arena);
    uform_model_free(model);
    printf("Test: Blocking and Deferred - PASSED\n");
}

#if !UFORM_TEST_STUB

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

static uform_continuation_t wait_at_gate(size_t completed, size_t total, void *state) {
    gate_t *const gate = (gate_t *)state;
    gate->completed = completed, gate->total = total;
    atomic_store(&gate->reached, 1);
    while (!atomic_load(&gate->opened)) {}
    return uform_continue_k;
}

/** While a deferred call runs on the executor's own thread, its executor and its arena refuse
 *  every other call; once the gate opens, polling alone sees the call land what a blocking one
 *  does, and both serve again. */
static void test_held_calls(void) {
    printf("Test: Held Calls\n");
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
    expect(!blocking.outcome.message, blocking.outcome.message);

    gate_t gate = {0, 0, 0, 0};
    encode_texts(model, executor, uform_completion_deferred_k, &texts, &deferred,
                 (uform_progress_t) {wait_at_gate, &gate, 1});
    while (!atomic_load(&gate.reached)) {}
    expect(gate.completed <= gate.total && gate.total == texts.count, "A report counts the batch's items");
    expect(uform_outcome_status(&deferred.outcome) == uform_status_deferred_k, "A running call is deferred");

    text_outputs_t contender = {.arena = blocking.arena};
    encode_texts(model, executor, uform_completion_blocking_k, &texts, &contender, silent);
    expect(contender.outcome.status == uform_status_contended_k, "A held executor refuses a blocking call");
    encode_texts(model, executor, uform_completion_deferred_k, &texts, &contender, silent);
    expect(contender.outcome.status == uform_status_contended_k, "A held executor refuses a deferred call");
    text_outputs_t borrower = {.arena = deferred.arena};
    encode_texts(model, NULL, uform_completion_blocking_k, &texts, &borrower, silent);
    expect(borrower.outcome.status == uform_status_contended_k, "A held arena refuses a call without an executor");
    expect(borrower.embeddings.data == NULL, "A refused call leaves its outputs NULL");

    atomic_store(&gate.opened, 1);
    while (uform_outcome_status(&deferred.outcome) == uform_status_deferred_k) {}
    expect(deferred.outcome.status == uform_status_success_k, "Polling alone sees a deferred call finish");
    expect(same_outputs(&blocking, &deferred, texts.count), "A deferred call lands what a blocking one does");
    encode_texts(model, NULL, uform_completion_blocking_k, &texts, &borrower, silent);
    expect(!borrower.outcome.message, "A final call lets go of its arena");

    uform_arena_free(deferred.arena);
    uform_arena_free(blocking.arena);
    uform_model_free(model);
    uform_executor_free(executor);
    printf("Test: Held Calls - PASSED\n");
}

#endif

/** An executor over a pool its lender spawned encodes what an owned one does, and a null
 *  pool is refused. */
static void test_executor_on_pool(void) {
    printf("Test: Executor On Pool\n");
    fu_topology_t topology = NULL;
    fu_pool_t pool = NULL;
    expect(fu_topology_new(&topology) == fu_success_k, "A topology must be harvested");
    expect(fu_pool_new("uform_test", fu_capabilities_all_k, &pool) == fu_success_k, "A pool must be created");
    size_t const threads = (fu_comptime_capabilities() & fu_capability_os_threads_k) ? 2 : 1;
    expect(fu_pool_spawn(topology, pool, threads, fu_caller_inclusive_k) == fu_success_k, "A pool must spawn");

    uform_outcome_t outcome = {0};
    uform_executor_t lent = NULL;
    uform_executor_on_pool(&(uform_executor_on_pool_t) {.executor = &lent, .outcome = &outcome, .pool = pool});
    expect(!outcome.message, outcome.message);
    expect(lent != NULL, "An executor over a lent pool must be created");

    uform_executor_t const owned = create_executor();
    uform_model_description_t description = {0};
    uform_model_t const model = load_model(owned, &description);
    size_t const lengths[] = {3, 40, 17, 1};
    texts_t texts;
    fill_texts(&texts, 4, lengths);
    text_outputs_t on_owned = {.arena = create_arena(owned)};
    text_outputs_t on_lent = {.arena = create_arena(lent)};
    uform_progress_t const silent = {0};

    encode_texts(model, owned, uform_completion_blocking_k, &texts, &on_owned, silent);
    expect(!on_owned.outcome.message, on_owned.outcome.message);
    encode_texts(model, lent, uform_completion_blocking_k, &texts, &on_lent, silent);
    expect(!on_lent.outcome.message, on_lent.outcome.message);
    expect(same_outputs(&on_owned, &on_lent, texts.count), "A lent pool lands what an owned one does");

    uform_arena_free(on_lent.arena);
    uform_arena_free(on_owned.arena);
    uform_model_free(model);
    uform_executor_free(owned);
    uform_executor_free(lent);
    fu_pool_delete(pool);
    fu_topology_delete(topology);

    uform_outcome_t refused = {0};
    uform_executor_t none = NULL;
    uform_executor_on_pool(&(uform_executor_on_pool_t) {.executor = &none, .outcome = &refused, .pool = NULL});
    expect(refused.status != uform_status_success_k, "A null pool is refused");
    expect(none == NULL, "A refused pool leaves no executor");
    printf("Test: Executor On Pool - PASSED\n");
}

/** An arena keeps its high-water mark, so a repeated call lands where the last did; inputs inside
 *  it are refused; a running call holds it, refusing another call meanwhile, until final; and it
 *  describes the host pages a RAM executor gave it. */
static void test_arenas(void) {
    printf("Test: Arenas\n");
    uform_executor_t const executor = create_executor();
    uform_model_description_t description = {0};
    uform_model_t const model = load_model(executor, &description);
    size_t const lengths[] = {30, 12, 64};
    texts_t texts;
    fill_texts(&texts, 3, lengths);
    text_outputs_t outputs = {.arena = create_arena(executor)};
    uform_progress_t const silent = {0};

    encode_texts(model, executor, uform_completion_blocking_k, &texts, &outputs, silent);
    expect(!outputs.outcome.message, outputs.outcome.message);
    void const* const first_landing = outputs.embeddings.data;
    encode_texts(model, executor, uform_completion_blocking_k, &texts, &outputs, silent);
    expect(!outputs.outcome.message, outputs.outcome.message);
    expect(outputs.embeddings.data == first_landing, "A steady loop reuses the arena it grew once");

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
    expect(outcome.status == uform_status_invalid_argument_k, "Inputs inside the rewound arena are refused");
    expect(embeddings.data == NULL, "A refused call leaves its outputs NULL");

    uform_arena_description_t placement = {uform_residency_unknown_k, 0};
    uform_arena_describe(
        &(uform_arena_describe_t) {.arena = outputs.arena, .outcome = &outcome, .description = &placement});
    expect(outcome.status == uform_status_success_k && placement.residency == uform_residency_ram_k &&
               placement.device == -1,
           "A RAM executor's arena lives in host pages");
    uform_arena_describe(&(uform_arena_describe_t) {.arena = NULL, .outcome = &outcome, .description = &placement});
    expect(outcome.status == uform_status_invalid_argument_k, "A NULL arena is refused");

    uform_arena_free(outputs.arena);
    uform_model_free(model);
    uform_executor_free(executor);
    printf("Test: Arenas - PASSED\n");
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
    printf("Test: Progress\n");
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
    expect(!outputs.outcome.message, outputs.outcome.message);
    expect(reports.ascending && reports.calls == 4, "Every step is reported once, in order");
    expect(reports.completed == texts.count && reports.total == texts.count, "The last report covers the batch");

    reports = (progress_reports_t) {0, 0, 0, true, 0};
    encode_texts(model, executor, uform_completion_blocking_k, &texts, &outputs,
                 (uform_progress_t) {record_progress, &reports, 0});
    expect(reports.ascending && reports.calls == 10, "Zero steps derive ten");

    reports = (progress_reports_t) {0, 0, 0, true, 1};
    encode_texts(model, executor, uform_completion_blocking_k, &texts, &outputs,
                 (uform_progress_t) {record_progress, &reports, 4});
    expect(outputs.outcome.status == uform_status_cancelled_k, "A handler's cancel stops the call");
    expect(reports.calls == 1, "A cancelled call reports nothing more");
    expect(outputs.embeddings.data == NULL && outputs.states.offsets == NULL, "A cancelled call leaves outputs NULL");

    reports = (progress_reports_t) {0, 0, 0, true, 1};
    encode_texts(model, executor, uform_completion_deferred_k, &texts, &outputs,
                 (uform_progress_t) {record_progress, &reports, 4});
    expect(uform_outcome_wait(&outputs.outcome) == uform_status_cancelled_k, "A deferred call cancels too");
    expect(outputs.embeddings.data == NULL, "A cancelled deferred call leaves outputs NULL");

    uform_arena_free(outputs.arena);
    uform_model_free(model);
    uform_executor_free(executor);
    printf("Test: Progress - PASSED\n");
}

/** Every refusal a caller can provoke: a null arena, a null model, no output named, a tape without
 *  storage or with decreasing offsets, and GPU-only verbs given a RAM executor. */
static void test_refusals(void) {
    printf("Test: Refusals\n");
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
    expect(outcome.status == uform_status_invalid_argument_k, "A null arena is refused");
    expect(embeddings.data == NULL && states.offsets == NULL, "A refused call leaves outputs NULL");

    encode.arena = arena, encode.embeddings = NULL, encode.states = NULL;
    uform_text_encode(&encode);
    expect(outcome.status == uform_status_invalid_argument_k, "A call naming no output is refused");

    encode.embeddings = &embeddings, encode.model = NULL;
    uform_text_encode(&encode);
    expect(outcome.status == uform_status_invalid_argument_k, "A null model is refused");

    encode.model = model, encode.texts.offsets = NULL;
    uform_text_encode(&encode);
    expect(outcome.status == uform_status_invalid_argument_k, "A tape without offsets is refused");

    size_t const decreasing[] = {0, 6, 4};
    encode.texts.offsets = decreasing;
    uform_text_encode(&encode);
    expect(outcome.status == uform_status_malformed_k, "Decreasing offsets are malformed");
    expect(embeddings.data == NULL, "A refused call leaves outputs NULL");

    uform_mesh_t mesh = NULL;
    uform_mesh_bootstrap(&(uform_mesh_bootstrap_t) {.mesh = &mesh, .executor = executor, .outcome = &outcome});
    expect(outcome.status == uform_status_device_k, "A mesh needs a VRAM executor");
    expect(mesh == NULL, "A refused bootstrap leaves no mesh");
    uform_trainer_t trainer = NULL;
    uform_trainer_init_vram(&(uform_trainer_init_vram_t) {.trainer = &trainer,
                                                          .executor = executor,
                                                          .outcome = &outcome,
                                                          .depth = 2,
                                                          .hidden = 256,
                                                          .schedule = {.steps = 1}});
    expect(outcome.status == uform_status_device_k, "A trainer needs a VRAM executor");
    expect(trainer == NULL, "A refused trainer is never created");

    uform_model_t gpu_model = NULL;
    uform_model_load_vram(&(uform_model_load_vram_t) {.model = &gpu_model,
                                                      .executor = executor,
                                                      .outcome = &outcome,
                                                      .path = view_of(checkpoint_path),
                                                      .quantization = uform_scalar_bf16_k});
    expect(outcome.status == uform_status_device_k, "A VRAM load needs a VRAM executor");
    expect(gpu_model == NULL, "A refused load leaves no model");

    uform_model_free(model);
    uform_arena_free(arena);
    uform_executor_free(executor);
    printf("Test: Refusals - PASSED\n");
}

/** Writes a small stand-in checkpoint to the temporary directory, answering its path. */
static char const* write_scratch_checkpoint(void) {
    static char path[1024];
    char const* directory = getenv("TMPDIR");
    if (!directory || !*directory) directory = getenv("TEMP");
    if (!directory || !*directory) directory = "/tmp";
    snprintf(path, sizeof(path), "%s/uform-test-checkpoint.safetensors", directory);
    FILE* const file = fopen(path, "wb");
    expect(file != NULL, "The scratch checkpoint must be writable");
    char const contents[] = "{\"__metadata__\":{\"stand-in\":\"any readable bytes load\"}}";
    uint64_t const header_length = sizeof(contents) - 1;
    uint8_t length_bytes[8];
    for (size_t byte = 0; byte != 8; ++byte) length_bytes[byte] = (uint8_t)(header_length >> (byte * 8));
    expect(fwrite(length_bytes, 1, 8, file) == 8 && fwrite(contents, 1, header_length, file) == header_length,
           "The scratch checkpoint must be written");
    fclose(file);
    return path;
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
    char const* const filter = getenv("UFORM_FILTER");
    if (filter && *filter) test_filter = filter;
    printf("- Filter: %s\n", test_filter ? test_filter : "none");

    char const* const environment_checkpoint = getenv("UFORM_TEST_CHECKPOINT");
    char const* scratch = NULL;
    if (argc > 1) checkpoint_path = argv[1];
    else if (environment_checkpoint && *environment_checkpoint) checkpoint_path = environment_checkpoint;
    else checkpoint_path = scratch = write_scratch_checkpoint();
    printf("- Checkpoint: %s\n", checkpoint_path);

    if (selected("library_probes")) test_library_probes();
    if (selected("executors")) test_executors();
    if (selected("model_load")) test_model_load();
    if (selected("text_shapes")) test_text_shapes();
    if (selected("heads")) test_heads();
    if (selected("media_shapes")) test_media_shapes();
    if (selected("blocking_and_deferred")) test_blocking_and_deferred();
#if !UFORM_TEST_STUB
    if (selected("held_calls")) test_held_calls();
#endif
    if (selected("executor_on_pool")) test_executor_on_pool();
    if (selected("arenas")) test_arenas();
    if (selected("progress")) test_progress();
    if (selected("refusals")) test_refusals();

    if (scratch) remove(scratch);
    printf("All tests passed!\n");
    return 0;
}
