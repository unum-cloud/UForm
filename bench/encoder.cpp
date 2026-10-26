/**
 *  @file bench/encoder.cpp
 *  @author Ash Vardanian
 *  @date October 25, 2026
 *  @brief Encoder throughput and latency through the C interface alone, at batch 1 and 64.
 *
 *  Loads one checkpoint onto the host or the first GPU, under a pinned class or the machine's own,
 *  and times blocking calls over synthetic lowercase texts that fill the model's context:
 *
 *  - "encode/batch_1" and "encode/batch_64": @c uform_text_encode to pooled embeddings, rated in
 *    tokens per second.
 *  - "choose/batch_1" and "choose/batch_64": @c uform_text_choose over four choices per item,
 *    rated in items per second; a checkpoint without the byte decoder skips them.
 *
 *  Every row reports its median call in milliseconds beside the rate.
 *
 *  Usage: `uform_bench_encoder --checkpoint <path> [--residency ram|vram] [--cpu-class <name>]
 *  [--gpu-class <name>] [--precision bf16|e4m3]`.
 */
#include <cstdint> // `std::uint8_t`, `std::int32_t`
#include <cstring> // `std::strcmp`, `std::memcpy`

#include <chrono>      // `std::chrono::steady_clock` — each call's latency
#include <random>      // `std::mt19937_64`, `std::uniform_int_distribution` — the synthetic texts
#include <string_view> // `std::string_view`
#include <vector>      // `std::vector` — the tapes, filled before the timed loop

#include <fmt/format.h> // `fmt::println`

#include "uform/encode.h" // `uform_text_encode`, `uform_text_choose`
#include "uform/model.h"  // `uform_model_load_ram`, `uform_model_load_vram`, `uform_model_describe`
#include "uform/types.h"  // `uform_executor_for_ram`, `uform_arena_init`, the class and scalar names

#include "harness.hpp" // `loop_t`, `run`, `read_settings`

using namespace unum::uform::bench;

using steady_clock_t = std::chrono::steady_clock;

/** The choices every item of a choice row scores, one byte each. */
static constexpr std::string_view choice_letters_k = "ABCD";

/** The handles every row calls through, and what its synthetic inputs derive from. */
struct session_t {

    /** The executor every call runs on. */
    uform_executor_t executor = nullptr;

    /** The loaded checkpoint. */
    uform_model_t model = nullptr;

    /** The arena every call's outputs land in. */
    uform_arena_t arena = nullptr;

    /** The checkpoint's geometry, whose context sizes the texts. */
    uform_model_description_t description {};

    /** Seed the synthetic texts derive from. */
    std::uint64_t seed = 0;
};

/** Items end to end with one more offset than items, as a @c uform_tape_view_t lays them out. */
struct tape_t {

    /** Every item's bytes, end to end. */
    std::vector<std::uint8_t> bytes;

    /** Where each item starts, and the end of the last. */
    std::vector<std::size_t> offsets;

    /** The tape as the C interface borrows it. */
    [[nodiscard]] uform_tape_view_t view() const noexcept { return {bytes.data(), offsets.data(), uform_scalar_u8_k}; }
};

/** @p count items of @p length lowercase letters each, drawn from @p seed . */
static tape_t random_letters(std::uint64_t seed, std::size_t count, std::size_t length) {
    tape_t tape {std::vector<std::uint8_t>(count * length), std::vector<std::size_t>(count + 1)};
    std::mt19937_64 generator {seed};
    std::uniform_int_distribution<int> letters {'a', 'z'};
    for (std::uint8_t &byte : tape.bytes) byte = static_cast<std::uint8_t>(letters(generator));
    for (std::size_t item = 0; item <= count; ++item) tape.offsets[item] = item * length;
    return tape;
}

/** @p count copies of @p item , end to end. */
static tape_t repeated_item(std::string_view item, std::size_t count) {
    tape_t tape {std::vector<std::uint8_t>(count * item.size()), std::vector<std::size_t>(count + 1)};
    for (std::size_t copy = 0; copy != count; ++copy)
        std::memcpy(tape.bytes.data() + copy * item.size(), item.data(), item.size());
    for (std::size_t copy = 0; copy <= count; ++copy) tape.offsets[copy] = copy * item.size();
    return tape;
}

/** The outcome's message, or its status's name where it carries none. */
static std::string_view message_of(uform_outcome_t const &outcome) noexcept {
    return outcome.message ? outcome.message : uform_status_name(outcome.status);
}

/** The bytes of @p text as the C interface borrows a name or a path. */
static uform_bytes_view_t bytes_of(std::string_view text) noexcept {
    return {reinterpret_cast<std::uint8_t const *>(text.data()), text.size()};
}

/** One @c uform_text_encode of @p batch texts filling the context to pooled embeddings. */
static void encode_texts(loop_t &loop, session_t const &session, std::size_t batch) {
    tape_t const texts = random_letters(session.seed, batch, session.description.context);
    uform_outcome_t outcome {};
    uform_rows_view_t embeddings {};
    uform_text_encode_stats_t stats {};
    uform_text_encode_t arguments {.model = session.model,
                                   .executor = session.executor,
                                   .outcome = &outcome,
                                   .count = batch,
                                   .texts = texts.view(),
                                   .embeddings = &embeddings,
                                   .arena = session.arena,
                                   .outputs_scalar = uform_scalar_f32_k,
                                   .stats = &stats};
    uform_text_encode(&arguments);
    if (outcome.status != uform_status_success_k) return loop.fail(message_of(outcome));
    loop.count_items(stats.tokens, "tokens");
    for ([[maybe_unused]] std::size_t call : loop) {
        steady_clock_t::time_point const start = steady_clock_t::now();
        uform_text_encode(&arguments);
        if (outcome.status != uform_status_success_k) return loop.fail(message_of(outcome));
        loop.add_window(steady_clock_t::now() - start, 1);
    }
}

/** One @c uform_text_choose of @p batch states, each ending a token short of the context so the
 *  mask byte fits, over @ref choice_letters_k . */
static void choose_items(loop_t &loop, session_t const &session, std::size_t batch) {
    tape_t const texts = random_letters(session.seed, batch, session.description.context - 1);
    tape_t const choices = repeated_item(choice_letters_k, batch);
    uform_outcome_t outcome {};
    uform_tape_view_t probabilities {};
    uform_text_choose_t arguments {.model = session.model,
                                   .executor = session.executor,
                                   .outcome = &outcome,
                                   .count = batch,
                                   .texts = texts.view(),
                                   .choices = choices.view(),
                                   .probabilities = &probabilities,
                                   .arena = session.arena};
    loop.count_items(batch, "items");
    for ([[maybe_unused]] std::size_t call : loop) {
        steady_clock_t::time_point const start = steady_clock_t::now();
        uform_text_choose(&arguments);
        if (outcome.status != uform_status_success_k) return loop.fail(message_of(outcome));
        loop.add_window(steady_clock_t::now() - start, 1);
    }
}

int main(int argc, char **argv) {
    char const *checkpoint = nullptr, *residency = "ram", *cpu_class_text = "", *gpu_class_text = "",
               *precision_text = "bf16";
    for (int index = 1; index + 1 < argc; index += 2) {
        char const *flag = argv[index], *value = argv[index + 1];
        if (!std::strcmp(flag, "--checkpoint")) checkpoint = value;
        else if (!std::strcmp(flag, "--residency")) residency = value;
        else if (!std::strcmp(flag, "--cpu-class")) cpu_class_text = value;
        else if (!std::strcmp(flag, "--gpu-class")) gpu_class_text = value;
        else if (!std::strcmp(flag, "--precision")) precision_text = value;
        else return fmt::println(stderr, "uform_bench_encoder: unknown flag {}", flag), 1;
    }
    if (argc % 2 == 0 || !checkpoint)
        return fmt::println(stderr, "Usage: uform_bench_encoder --checkpoint <path> [--residency ram|vram] " //
                                    "[--cpu-class <name>] [--gpu-class <name>] [--precision bf16|e4m3]"),
               1;
    std::string_view const home = residency;
    if (home != "ram" && home != "vram")
        return fmt::println(stderr, "uform_bench_encoder: --residency takes ram or vram, not {}", home), 1;
    uform_cpu_class_t const cpu_class = uform_cpu_class_from_name(bytes_of(cpu_class_text));
    uform_gpu_class_t const gpu_class = uform_gpu_class_from_name(bytes_of(gpu_class_text));
    if (*cpu_class_text && cpu_class == uform_cpu_class_unknown_k)
        return fmt::println(stderr, "uform_bench_encoder: {} names no CPU class", cpu_class_text), 1;
    if (*gpu_class_text && gpu_class == uform_gpu_class_unknown_k)
        return fmt::println(stderr, "uform_bench_encoder: {} names no GPU class", gpu_class_text), 1;
    if ((*cpu_class_text && home != "ram") || (*gpu_class_text && home != "vram"))
        return fmt::println(stderr, "uform_bench_encoder: --cpu-class pins ram, --gpu-class pins vram"), 1;
    uform_scalar_t const precision = uform_scalar_from_name(bytes_of(precision_text));
    if (precision != uform_scalar_bf16_k && precision != uform_scalar_e4m3_k)
        return fmt::println(stderr, "uform_bench_encoder: --precision takes bf16 or e4m3, not {}", precision_text), 1;

    settings_t const settings = read_settings();
    print(settings);
    session_t session;
    session.seed = settings.seed;
    std::int32_t const device = 0;
    uform_outcome_t outcome {};
    if (home == "vram") {
        uform_executor_for_vram_t arguments {.executor = &session.executor,
                                             .outcome = &outcome,
                                             .vram = {&device, 1, gpu_class}};
        uform_executor_for_vram(&arguments);
    }
    else {
        uform_executor_for_ram_t arguments {.executor = &session.executor,
                                            .outcome = &outcome,
                                            .ram = {nullptr, 0, cpu_class}};
        uform_executor_for_ram(&arguments);
    }
    if (outcome.status != uform_status_success_k)
        return fmt::println(stderr, "uform_bench_encoder: {}", message_of(outcome)), 1;
    uform_arena_init_t arena_arguments {.arena = &session.arena, .executor = session.executor, .outcome = &outcome};
    uform_arena_init(&arena_arguments);
    if (outcome.status != uform_status_success_k)
        return fmt::println(stderr, "uform_bench_encoder: {}", message_of(outcome)), 1;
    if (home == "vram") {
        uform_model_load_vram_t arguments {.model = &session.model,
                                           .executor = session.executor,
                                           .outcome = &outcome,
                                           .path = bytes_of(checkpoint),
                                           .vram = {&device, 1, gpu_class},
                                           .quantization = precision};
        uform_model_load_vram(&arguments);
    }
    else {
        uform_model_load_ram_t arguments {.model = &session.model,
                                          .executor = session.executor,
                                          .outcome = &outcome,
                                          .path = bytes_of(checkpoint),
                                          .ram = {nullptr, 0, cpu_class},
                                          .quantization = precision};
        uform_model_load_ram(&arguments);
    }
    if (outcome.status != uform_status_success_k)
        return fmt::println(stderr, "uform_bench_encoder: {}", message_of(outcome)), 1;
    uform_model_describe_t describe_arguments {.model = session.model,
                                               .executor = session.executor,
                                               .outcome = &outcome,
                                               .description = &session.description};
    uform_model_describe(&describe_arguments);
    if (outcome.status != uform_status_success_k)
        return fmt::println(stderr, "uform_bench_encoder: {}", message_of(outcome)), 1;
    char const *const class_name = home == "vram" ? uform_gpu_class_name(session.description.gpu_class)
                                                  : uform_cpu_class_name(session.description.cpu_class);
    fmt::println("- Model: {} in {} as {}, {} dimensions, context {}", class_name, home,
                 uform_scalar_name(session.description.quantization), session.description.dimensions,
                 session.description.context);

    bool failed = false;
    failed |= run(settings, "encode/batch_1", encode_texts, session, 1);
    failed |= run(settings, "encode/batch_64", encode_texts, session, 64);
    if (session.description.choice_temperature_groups) {
        failed |= run(settings, "choose/batch_1", choose_items, session, 1);
        failed |= run(settings, "choose/batch_64", choose_items, session, 64);
    }
    else fmt::println("choose: skipped, the checkpoint carries no byte decoder");
    uform_model_free(session.model);
    uform_arena_free(session.arena);
    uform_executor_free(session.executor);
    return failed ? 1 : 0;
}
