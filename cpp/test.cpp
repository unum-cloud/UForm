/**
 *  @file cpp/test.cpp
 *  @author Ash Vardanian
 *  @date November 18, 2026
 *  @brief The test of the C++ SDK: every handle it owns, every encode verb's refusals, blocking,
 *      deferred and awaited calls, and the training refusals, against the engine and the stub.
 *
 *  The engine's run takes the seeded checkpoint as @c argv[1]; without one, the model loads from a
 *  buffer of arbitrary bytes, which only the stub accepts. Every check reads the model's geometry
 *  from @c describe rather than assuming one, so both libraries pass the same assertions.
 */
#undef NDEBUG // ! A test's oracle must stay live in every build

#include <cmath>   // `std::abs`
#include <cstddef> // `std::size_t`
#include <cstdint> // `std::uint8_t`, `std::uint32_t`
#include <cstdlib> // `std::abort`, `EXIT_SUCCESS`, `EXIT_FAILURE`

#include <array>           // `std::array` - the test loop's ring
#include <coroutine>       // `std::suspend_never`
#include <source_location> // `std::source_location` - where a check failed
#include <span>            // `std::span`
#include <string_view>     // `std::string_view`
#include <utility>         // `std::move`

#include <fmt/format.h> // `fmt::println`

#include "uform/uform.hpp"

namespace uf = unum::uform;

/** Checks the run failed, counted rather than compiled out, so Release builds keep every oracle. */
static std::size_t failed_checks = 0;

/** Counts a failed check when @p is_satisfied is false, naming the line that checked it. */
static void expect_true(bool is_satisfied, std::source_location where = std::source_location::current()) noexcept {
    if (is_satisfied) return;
    ++failed_checks;
    fmt::println(stderr, "Failed the check at {}:{}", where.file_name(), where.line());
}

/** The scheduler every awaited call posts its polls to: a fixed ring, allocating nothing. */
struct test_loop_t {
    struct posted_t {
        void (*poll)(void*) noexcept;
        void* poll_state;
    };
    std::array<posted_t, 64> ring {};
    std::size_t head = 0, tail = 0;

    void post(void (*poll)(void*) noexcept, void* state) noexcept {
        expect_true(tail - head < ring.size());
        ring[tail++ % ring.size()] = {poll, state};
    }
    void run() noexcept {
        while (head != tail) {
            posted_t const next = ring[head++ % ring.size()];
            next.poll(next.poll_state);
        }
    }
};

/** A coroutine started eagerly and finished by the loop; its frame is the test's own. */
struct test_task_t {
    struct promise_type {
        test_task_t get_return_object() noexcept { return {}; }
        std::suspend_never initial_suspend() noexcept { return {}; }
        std::suspend_never final_suspend() noexcept { return {}; }
        void return_void() noexcept {}
        void unhandled_exception() noexcept { std::abort(); }
    };
};

/** Counts the reports it hears and lets the call go on, or stops it at the first. */
struct counting_handler_t {
    std::size_t reports = 0;
    uform_continuation_t answer = uform_continue_k;

    uform_continuation_t operator()(std::size_t, std::size_t) noexcept {
        ++reports;
        return answer;
    }
};

/** Everything a case needs: the executor, the model and where it came from, and its geometry. */
struct fixture_t {
    uf::executor_t executor;
    uf::model_t model;
    uform_model_description_t description {};
};

static std::string_view const captions[] = {"a cat on a mat", "a sunrise over the bay", "a dog in the fog"};

static void test_statuses() {
    auto [value, status] = uf::expected<std::size_t> {42};
    expect_true(!status.failed() && value == 42);

    auto [nothing, missing_status] = uf::expected<std::size_t> {{}, {uf::status_code_t::io_k, "The file is missing"}};
    expect_true(missing_status == uf::status_code_t::io_k && nothing == 0);
}

static void test_executors_and_arenas(fixture_t& fixture) {
    expect_true(!fixture.executor.drain(uform_mood_spinning_k).failed());
    expect_true(!fixture.executor.drain().failed());

    auto [arena, arena_status] = uf::arena_t::make({.executor = fixture.executor.handle()});
    expect_true(!arena_status.failed() && arena.handle());
    auto [description, describe_status] = arena.describe();
    expect_true(!describe_status.failed() && description.residency == uform_residency_ram_k &&
                description.device == -1 && description.vendor == uform_gpu_vendor_unknown_k &&
                description.vendor_ordinal == -1 && description.host_access == uform_host_access_readable_k);

    // A moved arena leaves an empty owner behind, which frees nothing.
    uf::arena_t moved = std::move(arena);
    expect_true(moved.handle() && !arena.handle());

    std::array<uform_gpu_device_t, 4> devices {};
    auto [count, devices_status] = uf::gpu_devices(devices);
    expect_true(!devices_status.failed());
    fmt::println("- GPUs: {}", count);
}

static void test_model_lifecycle(fixture_t& fixture) {
    auto [missing, missing_status] = uf::model_t::load_ram(
        {.path = uf::bytes_view("/nonexistent/uform.safetensors"), .quantization = uform_scalar_bf16_k});
    expect_true(missing_status.failed() && !missing.handle());

    auto [unquantized,
          unquantized_status] = uf::model_t::load_ram({.path = uf::bytes_view("/nonexistent/uform.safetensors")});
    expect_true(unquantized_status.failed() && !unquantized.handle());

    auto [description, describe_status] = fixture.model.on(fixture.executor).describe();
    expect_true(!describe_status.failed() && description.dimensions != 0);
    expect_true(description.cpu_class != uform_cpu_class_unknown_k);

    uf::deferred<uform_model_description_t> call = fixture.model.deferred(fixture.executor).describe();
    auto [deferred_description, deferred_status] = call.wait();
    expect_true(!deferred_status.failed() && deferred_description.dimensions == description.dimensions);

    uf::model_t closed;
    auto [closed_description, closed_status] = closed.on(fixture.executor).describe();
    expect_true(closed_status == uf::status_code_t::uninitialized_k && closed_description.dimensions == 0);
}

static void test_text_encode(fixture_t& fixture) {
    std::size_t gathered_offsets[4];
    char gathered_elements[64];
    auto [texts, gather_status] = uf::gathered<char>(captions, gathered_offsets, gathered_elements);
    expect_true(!gather_status.failed());
    expect_true(texts.count() == 3 && texts.entry(1).size() == captions[1].size());
    char cramped_elements[8];
    expect_true(uf::gathered<char>(captions, gathered_offsets, cramped_elements).status ==
                uf::status_code_t::invalid_argument_k);
    expect_true(uf::gathered<char>(captions, std::span(gathered_offsets).first(3), gathered_elements).status ==
                uf::status_code_t::invalid_argument_k);

    auto [arena, arena_status] = uf::arena_t::make({.executor = fixture.executor.handle()});
    expect_true(!arena_status.failed());
    uform_text_encode_stats_t stats {};
    counting_handler_t handler;
    auto [encoded, encode_status] = fixture.model.on(fixture.executor)
                                        .text_encode(
                                            texts, uf::encode_outputs_t::embeddings_k | uf::encode_outputs_t::states_k,
                                            {uform_scalar_f32_k, {handler, 3}, &stats}, arena);
    expect_true(!encode_status.failed());
    expect_true(handler.reports != 0);

    std::size_t const dimensions = fixture.description.dimensions;
    expect_true(encoded.embeddings.written == 3 && encoded.embeddings.view.columns == dimensions);
    for (std::size_t item = 0; item != 3; ++item) expect_true(encoded.embeddings.row<float>(item).size() == dimensions);
    expect_true(encoded.embeddings.row<double>(0).empty());

    std::span<std::size_t const> const offsets = encoded.states.offsets();
    expect_true(encoded.states.written == 3 && offsets.size() == 4 && offsets.back() == stats.tokens);
    expect_true(encoded.states.item(2).written == offsets[3] - offsets[2]);
    expect_true(encoded.token_vectors.written == 0 && encoded.token_vectors.offsets().empty());

    counting_handler_t stopper {.answer = uform_cancel_k};
    auto [cancelled, cancelled_status] = fixture.model.on(fixture.executor)
                                             .text_encode(texts, uf::encode_outputs_t::embeddings_k,
                                                          {uform_scalar_f32_k, {stopper, 1}}, arena);
    expect_true(cancelled_status == uf::status_code_t::cancelled_k && cancelled.embeddings.view.data == nullptr);
}

static void test_refusals(fixture_t& fixture) {
    auto [arena, arena_status] = uf::arena_t::make({.executor = fixture.executor.handle()});
    expect_true(!arena_status.failed());
    std::string_view const joined = "first documentthe second";
    std::size_t const joined_offsets[] = {0, 14, 24};

    std::size_t const overrun_offsets[] = {0, 14, 99};
    auto [overrun, overrun_status] = fixture.model.on(fixture.executor)
                                         .text_encode({joined, overrun_offsets}, uf::encode_outputs_t::embeddings_k,
                                                      uform_scalar_f32_k, arena);
    expect_true(overrun_status == uf::status_code_t::shape_mismatch_k && overrun.embeddings.written == 0);

    auto [empty, empty_status] = fixture.model.on(fixture.executor)
                                     .text_encode({joined, {}}, uf::encode_outputs_t::embeddings_k, uform_scalar_f32_k,
                                                  arena);
    expect_true(empty_status == uf::status_code_t::invalid_argument_k);

    // A refusal before the call crosses is final from the start, deferred or not.
    uf::deferred<uf::encoded_t> refused =
        fixture.model.deferred(fixture.executor)
            .text_encode({joined, overrun_offsets}, uf::encode_outputs_t::embeddings_k, uform_scalar_f32_k, arena);
    expect_true(refused.status() == uf::status_code_t::shape_mismatch_k);

    std::uint8_t const pixels[12] {};
    std::size_t const pixel_offsets[] = {0, 12};
    std::uint32_t const heights[] = {2}, widths[] = {2, 2};
    auto [misshapen, misshapen_status] = fixture.model.on(fixture.executor)
                                             .image_encode_pixels({pixels, pixel_offsets}, heights, widths,
                                                                  uf::encode_outputs_t::embeddings_k,
                                                                  uform_scalar_f32_k, arena);
    expect_true(misshapen_status == uf::status_code_t::shape_mismatch_k);

    std::string_view const choices = "yn";
    std::size_t const choice_offsets[] = {0, 2};
    auto [unpaired, unpaired_status] =
        fixture.model.on(fixture.executor).text_choose({joined, joined_offsets}, {choices, choice_offsets}, {}, arena);
    expect_true(unpaired_status == uf::status_code_t::shape_mismatch_k);
}

static void test_text_choose(fixture_t& fixture) {
    auto [arena, arena_status] = uf::arena_t::make({.executor = fixture.executor.handle()});
    expect_true(!arena_status.failed());
    std::string_view const states = "keep this candidate?expand which neighbour?";
    std::size_t const state_offsets[] = {0, 20, 43};
    std::string_view const choices = "ynABCD";
    std::size_t const choice_offsets[] = {0, 2, 6};
    auto [probabilities, choose_status] =
        fixture.model.on(fixture.executor).text_choose({states, state_offsets}, {choices, choice_offsets}, {}, arena);
    if (fixture.description.choice_temperature_groups == 0) {
        expect_true(choose_status == uf::status_code_t::unsupported_k);
        return;
    }
    expect_true(!choose_status.failed() && probabilities.count() == 2);
    float total = 0;
    for (float const probability : probabilities.entry(1)) total += probability;
    expect_true(std::abs(total - 1.0f) < 1e-3f);
}

static void test_deferred(fixture_t& fixture) {
    auto [arena, arena_status] = uf::arena_t::make({.executor = fixture.executor.handle()});
    expect_true(!arena_status.failed());
    std::size_t gathered_offsets[4];
    char gathered_elements[64];
    auto [texts, gather_status] = uf::gathered<char>(captions, gathered_offsets, gathered_elements);
    expect_true(!gather_status.failed());

    uf::deferred<uf::encoded_t> call = fixture.model.deferred(fixture.executor)
                                           .text_encode(texts, uf::encode_outputs_t::embeddings_k, uform_scalar_f32_k,
                                                        arena);
    uf::status_code_t const polled = call.status();
    expect_true(polled == uf::status_code_t::deferred_k || polled == uf::status_code_t::success_k);
    auto [encoded, encode_status] = call.wait();
    expect_true(!encode_status.failed() && encoded.embeddings.written == 3);

    // Waiting again answers the same final call.
    auto [again, again_status] = call.wait();
    expect_true(!again_status.failed() && again.embeddings.view.data == encoded.embeddings.view.data);
}

/** Awaits a deferred encoding on @p loop, as a caller's own task would. */
static test_task_t await_encoding(fixture_t& fixture, uf::tape_view<char> texts, uf::arena_t& arena, test_loop_t& loop,
                                  std::size_t& awaited_rows) {
    uf::deferred<uf::encoded_t> call = fixture.model.deferred(fixture.executor)
                                           .text_encode(texts, uf::encode_outputs_t::embeddings_k, uform_scalar_f32_k,
                                                        arena);
    auto [encoded, status] = co_await call.wait(loop);
    expect_true(!status.failed());
    awaited_rows = encoded.embeddings.written;
}

/** Encodes on two executors at once on @p loop, as the README's fan-out does, writing the rows
 *  both answered into @p awaited_rows. */
static test_task_t await_fan_out(fixture_t& fixture, uf::executor_t const& other_executor, uf::tape_view<char> texts,
                                 test_loop_t& loop, std::size_t& awaited_rows) {
    auto [first_arena, first_arena_status] = uf::arena_t::make({.executor = fixture.executor.handle()});
    auto [second_arena, second_arena_status] = uf::arena_t::make({.executor = other_executor.handle()});
    expect_true(!first_arena_status.failed() && !second_arena_status.failed());
    uf::deferred<uf::encoded_t> first = fixture.model.deferred(fixture.executor)
                                            .text_encode(texts, uf::encode_outputs_t::embeddings_k, uform_scalar_f32_k,
                                                         first_arena);
    uf::deferred<uf::encoded_t> second = fixture.model.deferred(other_executor)
                                             .text_encode(texts, uf::encode_outputs_t::embeddings_k, uform_scalar_f32_k,
                                                          second_arena);
    auto [first_encoded, first_status] = co_await first.wait(loop); // both calls were already running
    auto [second_encoded, second_status] = co_await second.wait(loop);
    expect_true(!first_status.failed() && !second_status.failed());
    awaited_rows = first_encoded.embeddings.written + second_encoded.embeddings.written;
}

/** Awaits @p call on @p loop, writing what it answered into @p awaited. */
static test_task_t await_status(uf::deferred<void>& call, test_loop_t& loop, uf::status_t& awaited) {
    awaited = co_await call.wait(loop);
}

/** Finishes the outcome @p state points at, as a library thread would. */
static void finish_by_hand(void* state) noexcept {
    static_cast<uform_outcome_t*>(state)->status = uform_status_success_k;
}

static void test_awaiter(fixture_t& fixture) {
    test_loop_t loop;
    auto [arena, arena_status] = uf::arena_t::make({.executor = fixture.executor.handle()});
    expect_true(!arena_status.failed());
    std::size_t gathered_offsets[4];
    char gathered_elements[64];
    auto [texts, gather_status] = uf::gathered<char>(captions, gathered_offsets, gathered_elements);
    expect_true(!gather_status.failed());

    std::size_t awaited_rows = 0;
    await_encoding(fixture, texts, arena, loop, awaited_rows);
    expect_true(loop.tail - loop.head <= loop.ring.size());
    loop.run();
    expect_true(awaited_rows == 3);

    auto [other_executor, other_executor_status] = uf::executor_t::for_ram({.threads = 1});
    expect_true(!other_executor_status.failed());
    std::size_t fanned_rows = 0;
    await_fan_out(fixture, other_executor, texts, loop, fanned_rows);
    loop.run();
    expect_true(fanned_rows == 6);

    // A call no library thread finishes yet suspends its awaiter, which re-polls until the hand
    // finishing it runs between two polls.
    uform_outcome_t* pending = nullptr;
    uf::deferred<void> call {[&](uform_outcome_t* outcome, uf::outputs<void>*) noexcept {
        outcome->status = uform_status_deferred_k;
        pending = outcome;
    }};
    uf::status_t awaited {uf::status_code_t::internal_k};
    await_status(call, loop, awaited);
    expect_true(loop.tail - loop.head == 1);
    loop.post(&finish_by_hand, pending);
    loop.run();
    expect_true(!awaited.failed() && loop.head == loop.tail);
}

static void test_training_refusals(fixture_t& fixture) {
    auto [mesh, mesh_status] = uf::mesh_t::bootstrap({.executor = fixture.executor.handle()});
    expect_true(mesh_status == uf::status_code_t::device_k && !mesh.handle());
    auto [trainer, trainer_status] = uf::trainer_t::init_vram(
        {.executor = fixture.executor.handle(), .depth = 2, .hidden = 256, .schedule = {.steps = 1}});
    expect_true(trainer_status == uf::status_code_t::device_k && !trainer.handle());

    uform_modality_t const modalities[] = {uform_modality_text_k};
    std::uint8_t const contents[] = {'h', 'i'};
    std::size_t const offsets[] = {0, 2};
    auto [embeddings, step_status] = trainer.on(fixture.executor).step(modalities, {contents, offsets});
    expect_true(step_status == uf::status_code_t::uninitialized_k && embeddings.written == 0);
    auto [unpaired, unpaired_status] = trainer.on(fixture.executor).step({}, {contents, offsets});
    expect_true(unpaired_status == uf::status_code_t::shape_mismatch_k);
    expect_true(trainer.on(fixture.executor).save(uf::bytes_view("checkpoint.safetensors")) ==
                uf::status_code_t::uninitialized_k);
    expect_true(mesh.describe().status == uf::status_code_t::uninitialized_k);
}

int main(int argc, char* argv[]) {
    fmt::println("UForm {}", uform_version_name());
    fmt::println("- Compiled for: {}", uform_capabilities_compiled());
    fmt::println("- This machine: {}", uform_capabilities_available());

    test_statuses();

    fixture_t fixture;
    auto [executor, executor_status] = uf::executor_t::for_ram({.threads = 2});
    expect_true(!executor_status.failed());
    fixture.executor = std::move(executor);

    // The stub loads any bytes, while the engine reads only the seeded checkpoint it is handed.
    std::uint8_t const arbitrary[] = {'u', 'f', 'o', 'r', 'm'};
    auto [model, model_status] = argc > 1 ? uf::model_t::load_ram({.executor = fixture.executor.handle(),
                                                                   .path = uf::bytes_view(argv[1]),
                                                                   .quantization = uform_scalar_bf16_k})
                                          : uf::model_t::load_ram_buffer({.executor = fixture.executor.handle(),
                                                                          .buffer = uf::bytes_view(arbitrary),
                                                                          .quantization = uform_scalar_bf16_k});
    expect_true(!model_status.failed());
    fixture.model = std::move(model);
    auto [description, describe_status] = fixture.model.describe(); // the calling thread
    expect_true(!describe_status.failed());
    fixture.description = description;
    fmt::println("- Model: {} dimensions, {} tokens of context", description.dimensions, description.context);

    test_executors_and_arenas(fixture);
    test_model_lifecycle(fixture);
    test_text_encode(fixture);
    test_refusals(fixture);
    test_text_choose(fixture);
    test_deferred(fixture);
    test_awaiter(fixture);
    test_training_refusals(fixture);
    fmt::println("{} failed checks", failed_checks);
    return failed_checks == 0 ? EXIT_SUCCESS : EXIT_FAILURE;
}
