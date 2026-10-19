/**
 *  @file bench/harness.hpp
 *  @author Ash Vardanian
 *  @date October 19, 2026
 *  @brief Shared timing for the GPU benchmarks: settings, the timed loop, CUDA-event windows, rows.
 *
 *  A benchmark is a plain function over a @ref loop_t that sets up untimed, then times its launches
 *  through @ref time_windows , which brackets doubling windows of back-to-back launches between two
 *  CUDA events, so a row reports device time rather than the host's enqueue rate. @ref run prints
 *  one row per benchmark the filter selects.
 *
 *  @verbatim
 *  Variable          Default  Meaning
 *  UFORM_FILTER      none     Benchmarks to run, by name, as an ECMAScript regex or a substring
 *  UFORM_SEED        42       Seed the synthetic inputs derive from, an unsigned integer or random
 *  UFORM_WARMUP      1s       Untimed run per benchmark, like 200ms or 1s
 *  UFORM_TIME_LIMIT  10s      Timed run per benchmark, like 200ms or 10s
 *  @endverbatim
 */
#pragma once

#include <cstdlib> // `std::getenv`, `std::exit`

#include <charconv>    // `std::from_chars`
#include <chrono>      // `std::chrono::steady_clock`, `std::chrono::milliseconds`
#include <random>      // `std::random_device`
#include <regex>       // `std::regex`, `std::regex_search`
#include <string_view> // `std::string_view`
#include <type_traits> // `std::type_identity_t`

#include <cuda.h>       // `cuEventCreate`, `cuEventRecord`, `cuEventElapsedTime`
#include <fmt/format.h> // `fmt::println`

#include "uform/types.h" // `uform_version_name`, `uform_capabilities_compiled`, `uform_capabilities_available`

#include "core/types.hpp" // `expected_t`, `sz`, the scalar aliases

/** Prints a StringZilla view as the text it spans. */
template <>
struct fmt::formatter<ashvardanian::stringzilla::string_view_t> : fmt::formatter<std::string_view> {
    auto format(std::string_view text, fmt::format_context &context) const {
        return fmt::formatter<std::string_view>::format(text, context);
    }
};

/** Prints an owning StringZilla string as the text it holds. */
template <>
struct fmt::formatter<ashvardanian::stringzilla::string_t> : fmt::formatter<std::string_view> {
    auto format(std::string_view text, fmt::format_context &context) const {
        return fmt::formatter<std::string_view>::format(text, context);
    }
};

namespace unum::uform::bench {

using namespace unum::uform::core;

/** The text of the environment variable @p name , empty when it is unset. */
[[nodiscard]] inline std::string_view environment_text(char const *name) noexcept {
    char const *const text = std::getenv(name);
    return text == nullptr ? std::string_view {} : std::string_view {text};
}

/** Exits with status 1, naming the variable @p name whose @p text does not parse as @p wanted . */
[[noreturn]] inline void refuse_setting(char const *name, std::string_view text, char const *wanted) noexcept {
    fmt::println(stderr, "{}=\"{}\" does not parse, expected {}", name, text, wanted);
    std::exit(1);
}

/** The duration the variable @p name spells as an unsigned integer and @c ms or @c s , or
 *  @p fallback when it is unset; exits with status 1 when it does not parse. */
[[nodiscard]] inline std::chrono::milliseconds environment_duration(char const *name,
                                                                    std::chrono::milliseconds fallback) noexcept {
    std::string_view const text = environment_text(name);
    if (text.empty()) return fallback;
    u64_t count = 0;
    auto const [unit_begin, parse_status] = std::from_chars(text.data(), text.data() + text.size(), count);
    std::string_view const unit = text.substr(static_cast<std::size_t>(unit_begin - text.data()));
    if (parse_status == std::errc {} && unit == "ms") return std::chrono::milliseconds(count);
    if (parse_status == std::errc {} && unit == "s") return std::chrono::seconds(count);
    refuse_setting(name, text, "a duration like 200ms or 10s");
}

/** A regex over benchmark names, or a substring when it does not compile; empty selects all. */
struct filter_t {

    /** The filter as given. */
    std::string_view text {};

    /** The filter compiled, matching nothing when @ref text is not a regex. */
    std::regex pattern {};

    /** Compiles @p text , keeping the empty pattern when it is not a regex. */
    [[nodiscard]] static filter_t make(std::string_view text) noexcept {
        filter_t filter {text, {}};
        if (text.empty()) return filter;
        try {
            filter.pattern.assign(text.begin(), text.end(),
                                  std::regex_constants::ECMAScript | std::regex_constants::nosubs);
        }
        catch (std::regex_error const &) {
        }
        return filter;
    }

    /** Whether the benchmark named @p name runs. */
    [[nodiscard]] bool selects(std::string_view name) const noexcept {
        return text.empty() || std::regex_search(name.begin(), name.end(), pattern) ||
               name.find(text) != std::string_view::npos;
    }
};

/** Every setting the benchmarks read, defaults as initializers, filled by @ref read_settings . */
struct settings_t {

    /** Benchmarks to run. */
    filter_t filter {};

    /** Seed the synthetic inputs derive from. */
    u64_t seed = 42;

    /** Untimed run per benchmark. */
    std::chrono::milliseconds warmup {1000};

    /** Timed run per benchmark. */
    std::chrono::milliseconds time_limit {10000};
};

/** Reads every @ref settings_t variable, exiting with status 1 on the first that does not parse. */
[[nodiscard]] inline settings_t read_settings() noexcept {
    settings_t settings;
    settings.filter = filter_t::make(environment_text("UFORM_FILTER"));
    std::string_view const seed_text = environment_text("UFORM_SEED");
    if (seed_text == "random") settings.seed = std::random_device {}();
    else if (!seed_text.empty()) {
        char const *const seed_end = seed_text.data() + seed_text.size();
        auto const [parsed_end, parse_status] = std::from_chars(seed_text.data(), seed_end, settings.seed);
        if (parse_status != std::errc {} || parsed_end != seed_end)
            refuse_setting("UFORM_SEED", seed_text, "an unsigned integer or random");
    }
    settings.warmup = environment_duration("UFORM_WARMUP", settings.warmup);
    settings.time_limit = environment_duration("UFORM_TIME_LIMIT", settings.time_limit);
    return settings;
}

/** Prints the version, the capabilities and the settings, one per line. */
inline void print(settings_t const &settings) noexcept {
    fmt::println("UForm {}", uform_version_name());
    fmt::println("- Compiled for: {}", uform_capabilities_compiled());
    fmt::println("- This machine: {}", uform_capabilities_available());
    fmt::println("- Seed: {}", settings.seed);
    std::string_view const filter = settings.filter.text;
    fmt::println("- Filter: {}", filter.empty() ? std::string_view {"none"} : filter);
    fmt::println("- Warmup: {}ms, time limit: {}ms", settings.warmup.count(), settings.time_limit.count());
}

/**
 *  @brief One benchmark's timed loop, iterated as `for (std::size_t call : loop)`.
 *
 *  Setup above the loop stays untimed, and the clock starts at @c begin . The loop runs untimed for
 *  the warm-up, then counts calls until the time limit, reading the clock again only once the calls
 *  so far have grown by a 64th, so a short call doesn't time the clock itself. Windows timed on the
 *  device through @ref add_window replace the loop's own calls and wall time.
 */
class loop_t {
    using steady_clock_t = std::chrono::steady_clock;
    using seconds_t = std::chrono::duration<f64_t>;

    /** Where the loop is: untimed, timed, finished by the time limit, or failed. */
    enum class phase_t : u8_t { warming_up_k, timed_k, done_k, failed_k };

    /** The calls so far, divided by this, run between two reads of the clock. */
    static constexpr std::size_t clock_stride_k = 64;

    std::chrono::milliseconds warmup_, time_limit_;
    steady_clock_t::time_point start_ {};
    steady_clock_t::duration elapsed_ {};
    seconds_t window_elapsed_ {};
    std::size_t calls_ = 0, next_check_ = 0, window_calls_ = 0;
    phase_t phase_ = phase_t::warming_up_k;

    bool keep_running() noexcept {
        if (phase_ == phase_t::done_k || phase_ == phase_t::failed_k) return false;
        if (calls_ < next_check_) return true;
        elapsed_ = steady_clock_t::now() - start_;
        if (phase_ == phase_t::warming_up_k && elapsed_ >= warmup_) {
            phase_ = phase_t::timed_k;
            start_ = steady_clock_t::now();
            elapsed_ = {}, window_elapsed_ = {};
            calls_ = 0, window_calls_ = 0;
        }
        else if (phase_ == phase_t::timed_k && elapsed_ >= time_limit_) {
            phase_ = phase_t::done_k;
            return false;
        }
        next_check_ = calls_ + calls_ / clock_stride_k + 1;
        return true;
    }

  public:
    struct end_t {};
    struct iterator_t {
        loop_t *loop;
        bool operator!=(end_t) noexcept { return loop->keep_running(); }
        std::size_t operator*() const noexcept { return loop->calls_; }
        void operator++() noexcept { ++loop->calls_; }
    };

    loop_t(std::chrono::milliseconds warmup, std::chrono::milliseconds time_limit) noexcept
        : warmup_(warmup), time_limit_(time_limit) {}

    iterator_t begin() noexcept { return start_ = steady_clock_t::now(), iterator_t {this}; }
    end_t end() const noexcept { return {}; }

    /** Stops the loop, printing @p description to standard error, and reports the row failed. */
    void fail(sz::string_view_t description) noexcept {
        fmt::println(stderr, "  benchmark failed: {}", description);
        phase_ = phase_t::failed_k;
    }

    /** Whether the benchmark failed, which fails the binary. */
    [[nodiscard]] bool failed() const noexcept { return phase_ == phase_t::failed_k; }

    /** Adds @p calls the device ran back to back in @p elapsed . */
    void add_window(seconds_t elapsed, std::size_t calls) noexcept {
        window_elapsed_ += elapsed;
        window_calls_ += calls;
    }

    /** Prints the row of the finished benchmark under @p name . */
    void report(std::string_view name) const noexcept {
        if (phase_ == phase_t::failed_k) return fmt::println("{:<32} FAILED", name);
        f64_t const seconds = window_calls_ ? window_elapsed_.count() : seconds_t(elapsed_).count();
        std::size_t const calls = window_calls_ ? window_calls_ : calls_;
        if (calls == 0) return fmt::println("{:<32} no timed calls", name);
        fmt::println("{:<32} {:>10.3f} ms/call {:>10.1f} calls/s {:>8} calls", name, seconds * 1e3 / calls,
                     calls / seconds, calls);
    }
};

/**
 *  @brief Times @p launch on @p stream until @p loop ends, in windows of back-to-back launches
 *      between two CUDA events, doubling a window while it lasts under a millisecond.
 *
 *  @p launch enqueues one call and answers its @c expected_t ; the first refusal fails the loop.
 */
template <typename launch_type_>
void time_windows(loop_t &loop, CUstream stream, launch_type_ &&launch) noexcept {
    CUevent start = nullptr, stop = nullptr;
    if (cuEventCreate(&start, CU_EVENT_DEFAULT) != CUDA_SUCCESS ||
        cuEventCreate(&stop, CU_EVENT_DEFAULT) != CUDA_SUCCESS)
        return loop.fail("CUDA event creation failed");
    std::size_t window = 1;
    for ([[maybe_unused]] std::size_t call : loop) {
        if (cuEventRecord(start, stream) != CUDA_SUCCESS) {
            loop.fail("CUDA event record failed");
            break;
        }
        for (std::size_t index = 0; index != window; ++index)
            if (expected_t launched = launch(); !launched) {
                loop.fail(launched.description_.view());
                break;
            }
        if (loop.failed()) break;
        f32_t milliseconds = 0;
        if (cuEventRecord(stop, stream) != CUDA_SUCCESS || cuEventSynchronize(stop) != CUDA_SUCCESS ||
            cuEventElapsedTime(&milliseconds, start, stop) != CUDA_SUCCESS) {
            loop.fail("CUDA event timing failed");
            break;
        }
        std::chrono::duration<f32_t, std::milli> const elapsed {milliseconds};
        loop.add_window(elapsed, window);
        if (elapsed < std::chrono::milliseconds(1)) window *= 2;
    }
    [[maybe_unused]] CUresult const start_destroyed = cuEventDestroy(start);
    [[maybe_unused]] CUresult const stop_destroyed = cuEventDestroy(stop);
}

/**
 *  @brief Runs @p bench_function on a fresh @ref loop_t and @p arguments when @p settings select
 *      @p name , then prints its row, answering whether it failed.
 *
 *  Takes a function pointer rather than any callable, so a benchmark whose signature drifts fails
 *  to compile here.
 */
template <typename... parameters_types_>
[[nodiscard]] bool run(settings_t const &settings, std::string_view name,
                       void (*bench_function)(loop_t &, parameters_types_...),
                       std::type_identity_t<parameters_types_>... arguments) noexcept {
    if (!settings.filter.selects(name)) return false;
    loop_t loop {settings.warmup, settings.time_limit};
    bench_function(loop, arguments...);
    loop.report(name);
    return loop.failed();
}

} // namespace unum::uform::bench
