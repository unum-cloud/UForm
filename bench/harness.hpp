/**
 *  @file bench/harness.hpp
 *  @author Ash Vardanian
 *  @date October 19, 2026
 *  @brief Shared timing for the benchmarks: settings, the timed loop, its windows, rows.
 *
 *  A benchmark is a plain function over a @ref loop_t that sets up untimed, then times its calls
 *  in windows it adds through @ref loop_t::add_window , so a row reports the time the benchmark
 *  measured rather than the host's enqueue rate, and the median of its windows beside the rate.
 *  @ref run prints one row per benchmark the filter selects.
 *
 *  Every setting is a @c UFORM_ variable, and a flag of the same name overrides it:
 *
 *  @verbatim
 *  Variable          Flag          Default  Meaning
 *  UFORM_FILTER      --filter      none     Benchmarks to run, by label, as an ECMAScript regex or a substring
 *  UFORM_SEED        --seed        42       Seed the synthetic inputs derive from, an unsigned integer or random
 *  UFORM_WARMUP      --warmup      1s       Untimed run per benchmark, like 200ms or 1s
 *  UFORM_TIME_LIMIT  --time-limit  10s      Timed run per benchmark, like 200ms or 10s
 *  @endverbatim
 */
#pragma once

#include <cctype>  // `std::tolower`
#include <cmath>   // `std::isfinite`
#include <cstdint> // `std::uint64_t`, `std::uint8_t`
#include <cstdio>  // `std::fprintf`, `std::snprintf`
#include <cstdlib> // `std::getenv`, `std::exit`, `std::strtod`
#include <cstring> // `std::strcmp`, `std::strncmp`

#include <algorithm>    // `std::nth_element` — the median window
#include <array>        // `std::array` — the windows the median reads
#include <charconv>     // `std::from_chars`
#include <chrono>       // `std::chrono::steady_clock`, `std::chrono::milliseconds`
#include <optional>     // `std::optional` — the filter's regex
#include <random>       // `std::random_device`
#include <regex>        // `std::regex`, `std::regex_search`
#include <string>       // `std::string` — the filter's pattern
#include <string_view>  // `std::string_view`
#include <system_error> // `std::errc`
#include <type_traits>  // `std::type_identity_t`

#include <fmt/format.h> // `fmt::println`

#include "uform/types.h" // `uform_version_name`, `uform_capabilities_compiled`, `uform_capabilities_available`

namespace unum::uform::bench {

#pragma region Shared Settings

/** Exits with one line naming a setting that does not parse, so a bad value never runs anything. */
[[noreturn]] inline void exit_unparsed(char const* variable, char const* text, char const* expected) noexcept {
    std::fprintf(stderr, "%s=\"%s\" does not parse, expected %s\n", variable, text, expected);
    std::exit(1);
}

/** The text of a setting: its @p flag when given, else its environment @p variable , else null. */
inline char const* setting_text(int argc, char** argv, char const* variable, std::string_view flag) noexcept {
    for (int index = 1; index < argc; ++index) {
        std::string_view const argument {argv[index]};
        if (argument == flag)
            return index + 1 < argc && std::strncmp(argv[index + 1], "--", 2) != 0 ? argv[index + 1] : "1";
        if (argument.size() > flag.size() && argument.starts_with(flag) && argument[flag.size()] == '=')
            return argv[index] + flag.size() + 1;
    }
    char const* text = std::getenv(variable);
    return text && *text ? text : nullptr;
}

/** A switch, @p fallback when unset: 0, false, off and no turn it off, anything else on. */
inline bool read_flag_setting(int argc, char** argv, char const* variable, std::string_view flag,
                              bool fallback = false) noexcept {
    char const* text = setting_text(argc, argv, variable, flag);
    if (!text) return fallback;
    for (char const* off : {"0", "false", "off", "no"})
        if (std::strcmp(text, off) == 0) return false;
    return true;
}

/** A whole number, @p fallback when unset. */
inline std::size_t read_count_setting(int argc, char** argv, char const* variable, std::string_view flag,
                                      std::size_t fallback) noexcept {
    char const* text = setting_text(argc, argv, variable, flag);
    if (!text) return fallback;
    std::string_view const spelled {text};
    std::size_t count = 0;
    auto const [end, error] = std::from_chars(spelled.data(), spelled.data() + spelled.size(), count);
    if (error != std::errc {} || end != spelled.data() + spelled.size())
        exit_unparsed(variable, text, "a whole number");
    return count;
}

/** A positive real, @p fallback when unset. */
inline double read_real_setting(int argc, char** argv, char const* variable, std::string_view flag,
                                double fallback) noexcept {
    char const* text = setting_text(argc, argv, variable, flag);
    if (!text) return fallback;
    char* end = nullptr;
    double const value = std::strtod(text, &end);
    if (*end != '\0' || !(value > 0) || !std::isfinite(value))
        exit_unparsed(variable, text, "a positive number like 0.1 or 10");
    return value;
}

/** A positive duration such as 200ms, 10s or 2m, @p fallback when unset. */
inline std::chrono::milliseconds read_duration_setting(int argc, char** argv, char const* variable,
                                                       std::string_view flag,
                                                       std::chrono::milliseconds fallback) noexcept {
    char const* text = setting_text(argc, argv, variable, flag);
    if (!text) return fallback;
    std::string_view const spelled {text};
    std::uint32_t count = 0;
    auto const [end, error] = std::from_chars(spelled.data(), spelled.data() + spelled.size(), count);
    std::string_view const unit = spelled.substr(static_cast<std::size_t>(end - spelled.data()));
    if (error == std::errc {} && count != 0) {
        if (unit == "ms") return std::chrono::milliseconds(count);
        if (unit == "s") return std::chrono::seconds(count);
        if (unit == "m") return std::chrono::minutes(count);
    }
    exit_unparsed(variable, text, "a duration like 200ms, 10s or 2m");
}

/** The run's seed: 42 when unset, drawn afresh for @c random , else any unsigned 64-bit value. */
inline std::uint64_t read_seed_setting(int argc, char** argv, char const* variable, std::string_view flag) noexcept {
    char const* text = setting_text(argc, argv, variable, flag);
    if (!text) return 42;
    std::string_view const spelled {text};
    if (spelled == "random") {
        std::random_device device;
        return (static_cast<std::uint64_t>(device()) << 32) ^ device();
    }
    std::uint64_t seed = 0;
    auto const [end, error] = std::from_chars(spelled.data(), spelled.data() + spelled.size(), seed);
    if (error != std::errc {} || end != spelled.data() + spelled.size())
        exit_unparsed(variable, text, "an unsigned integer or random");
    return seed;
}

/** Which labels a run selects: a regex search when the pattern compiles, else a substring. */
struct filter_t {
    std::string pattern;
    std::optional<std::regex> regex;

    bool selects(std::string_view label) const noexcept {
        if (pattern.empty()) return true;
        if (regex) return std::regex_search(label.begin(), label.end(), *regex);
        return label.find(pattern) != std::string_view::npos;
    }
};

/** The filter a setting spells, compiled once; empty, selecting everything, when unset. */
inline filter_t read_filter_setting(int argc, char** argv, char const* variable, std::string_view flag) noexcept {
    filter_t filter;
    char const* text = setting_text(argc, argv, variable, flag);
    if (!text) return filter;
    filter.pattern = text;
#if defined(__cpp_exceptions) && __cpp_exceptions
    try {
        filter.regex.emplace(filter.pattern);
    }
    catch (std::regex_error const&) {
    }
#else
    filter.regex.emplace(filter.pattern);
#endif
    return filter;
}

/** What every suite and bench reads before its own settings; @c program starts the rerun lines. */
struct common_settings_t {
    std::uint64_t seed = 42;
    filter_t filter;
    char const* program = "";
};

/** SplitMix64's finalizer, a bijection that spreads every input bit over all 64 output bits. */
constexpr std::uint64_t split_mix_bits(std::uint64_t value) noexcept {
    value = (value ^ (value >> 30)) * 0xBF58476D1CE4E5B9ull;
    value = (value ^ (value >> 27)) * 0x94D049BB133111EBull;
    return value ^ (value >> 31);
}

/** The @p index -th stream of the family @p name under @p seed , identical in every language. */
constexpr std::uint64_t stream_key(std::uint64_t seed, std::string_view name, std::uint64_t index = 0) noexcept {
    std::uint64_t hashed = 0xCBF29CE484222325ull;
    for (char const character : name) hashed = (hashed ^ static_cast<std::uint8_t>(character)) * 0x100000001B3ull;
    return split_mix_bits(split_mix_bits(seed ^ hashed) + index);
}

/**
 *  @brief SplitMix64 over one 64-bit state, seeded by a @ref stream_key , bit-identical to the
 *      JavaScript generator NumKong ships.
 *
 *  Distributions are not specified to draw the same sequence on every standard library, so a
 *  failure would stop reproducing from its seed when the toolchain changed; @ref below is the
 *  portable bounded draw.
 */
struct split_mix_generator_t {
    using result_type = std::uint64_t;
    std::uint64_t state = 0;

    explicit split_mix_generator_t(std::uint64_t key) noexcept : state(key) {}
    static constexpr result_type(min)() noexcept { return 0; }
    static constexpr result_type(max)() noexcept { return ~result_type {0}; }
    result_type operator()() noexcept { return split_mix_bits(state += 0x9E3779B97F4A7C15ull); }

    /** A uniform draw below @p bound , zero for an empty bound, without the bias of a modulo. */
    std::uint64_t below(std::uint64_t bound) noexcept {
        if (bound == 0) return 0;
        std::uint64_t const rejection_floor = (0u - bound) % bound;
        std::uint64_t draw = (*this)();
        while (draw < rejection_floor) draw = (*this)();
        return draw % bound;
    }
};

#pragma endregion Shared Settings

/** What every benchmark reads: the common settings, then how long each one warms up and runs. */
struct bench_settings_t : common_settings_t {
    std::chrono::milliseconds warmup {1000};
    std::chrono::milliseconds time_limit {10000};
};

/** Reads every @ref bench_settings_t field, exiting with status 1 on the first unparsable one. */
[[nodiscard]] inline bench_settings_t read_bench_settings(int argc, char** argv) noexcept {
    bench_settings_t settings;
    settings.seed = read_seed_setting(argc, argv, "UFORM_SEED", "--seed");
    settings.filter = read_filter_setting(argc, argv, "UFORM_FILTER", "--filter");
    settings.warmup = read_duration_setting(argc, argv, "UFORM_WARMUP", "--warmup", settings.warmup);
    settings.time_limit = read_duration_setting(argc, argv, "UFORM_TIME_LIMIT", "--time-limit", settings.time_limit);
    settings.program = argc != 0 ? argv[0] : "";
    return settings;
}

/** The library's version and the capabilities it was compiled for and finds on this machine. */
struct machine_t {
    char const* version = "";
    char const* compiled = "";
    char const* available = "";
};

/** Reads @ref machine_t through the C interface. */
[[nodiscard]] inline machine_t probe_machine() noexcept {
    return {uform_version_name(), uform_capabilities_compiled(), uform_capabilities_available()};
}

/** Prints the version line, then the capabilities as "- Compiled for:" and "- This machine:". */
inline void print(machine_t const& machine) noexcept {
    fmt::println("UForm {}", machine.version);
    fmt::println("- Compiled for: {}", machine.compiled);
    fmt::println("- This machine: {}", machine.available);
}

/** Prints the settings, one <tt>- Name: value</tt> line each. */
inline void print(bench_settings_t const& settings) noexcept {
    fmt::println("- Seed: {}", settings.seed);
    std::string_view const filter = settings.filter.pattern;
    fmt::println("- Filter: {}", filter.empty() ? std::string_view {"none"} : filter);
    fmt::println("- Warmup: {}ms, time limit: {}ms", settings.warmup.count(), settings.time_limit.count());
}

/**
 *  @brief One benchmark's timed loop, iterated as `for (std::size_t call : loop)`.
 *
 *  Setup above the loop stays untimed, and the clock starts at @c begin . The loop runs untimed for
 *  the warm-up, then counts calls until the time limit, reading the clock again only once the calls
 *  so far have grown by a 64th, so a short call doesn't time the clock itself. Windows timed
 *  through @ref add_window replace the loop's own calls and wall time, and the median of their
 *  per-call times is the row's latency.
 */
class loop_t {
    using steady_clock_t = std::chrono::steady_clock;
    using seconds_t = std::chrono::duration<double>;

    /** Where the loop is: untimed, timed, finished by the time limit, or failed. */
    enum class phase_t : std::uint8_t { warming_up_k, timed_k, done_k, failed_k };

    /** The calls so far, divided by this, run between two reads of the clock. */
    static constexpr std::size_t clock_stride_k = 64;

    /** Windows the median reads; once they fill, every other one is dropped and the stride between
     *  kept windows doubles, so the kept ones still span the whole timed run. */
    static constexpr std::size_t median_windows_k = 1024;

    std::chrono::milliseconds warmup_, time_limit_;
    steady_clock_t::time_point start_ {};
    steady_clock_t::duration elapsed_ {};
    seconds_t window_elapsed_ {};
    std::size_t calls_ = 0, next_check_ = 0, window_calls_ = 0;
    std::array<double, median_windows_k> window_latencies_ {};
    std::size_t windows_ = 0, windows_kept_ = 0, windows_stride_ = 1;
    std::size_t items_per_call_ = 1;
    std::string_view items_unit_ = "calls";
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
            windows_ = 0, windows_kept_ = 0, windows_stride_ = 1;
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
        loop_t* loop;
        bool operator!=(end_t) noexcept { return loop->keep_running(); }
        std::size_t operator*() const noexcept { return loop->calls_; }
        void operator++() noexcept { ++loop->calls_; }
    };

    loop_t(std::chrono::milliseconds warmup, std::chrono::milliseconds time_limit) noexcept
        : warmup_(warmup), time_limit_(time_limit) {}

    iterator_t begin() noexcept { return start_ = steady_clock_t::now(), iterator_t {this}; }
    end_t end() const noexcept { return {}; }

    /** Stops the loop, printing @p description to standard error, and reports the row failed. */
    void fail(std::string_view description) noexcept {
        fmt::println(stderr, "  benchmark failed: {}", description);
        phase_ = phase_t::failed_k;
    }

    /** Whether the benchmark failed, which fails the binary. */
    [[nodiscard]] bool failed() const noexcept { return phase_ == phase_t::failed_k; }

    /** Rates the row in @p items of @p unit per call, like tokens, instead of in calls. */
    void count_items(std::size_t items, std::string_view unit) noexcept {
        items_per_call_ = items;
        items_unit_ = unit;
    }

    /** Adds @p calls that ran back to back in @p elapsed . */
    void add_window(seconds_t elapsed, std::size_t calls) noexcept {
        window_elapsed_ += elapsed;
        window_calls_ += calls;
        if (windows_++ % windows_stride_ != 0) return;
        if (windows_kept_ == median_windows_k) {
            for (std::size_t kept = 0; kept != median_windows_k / 2; ++kept)
                window_latencies_[kept] = window_latencies_[kept * 2];
            windows_kept_ = median_windows_k / 2;
            windows_stride_ *= 2;
        }
        window_latencies_[windows_kept_++] = elapsed.count() / static_cast<double>(calls);
    }

    /** Prints the row of the finished benchmark under @p name . */
    void report(std::string_view name) const noexcept {
        if (phase_ == phase_t::failed_k) return fmt::println("{:<32} FAILED", name);
        double const seconds = window_calls_ ? window_elapsed_.count() : seconds_t(elapsed_).count();
        std::size_t const calls = window_calls_ ? window_calls_ : calls_;
        if (calls == 0) return fmt::println("{:<32} no timed calls", name);
        std::array<double, median_windows_k> latencies = window_latencies_;
        double median = seconds / static_cast<double>(calls);
        if (windows_kept_) {
            std::nth_element(latencies.begin(), latencies.begin() + windows_kept_ / 2,
                             latencies.begin() + windows_kept_);
            median = latencies[windows_kept_ / 2];
        }
        double const items = static_cast<double>(calls) * static_cast<double>(items_per_call_);
        fmt::println("{:<32} {:>10.3f} ms median {:>14.1f} {}/s {:>8} calls", name, median * 1e3, items / seconds,
                     items_unit_, calls);
    }
};

/**
 *  @brief Runs @p bench_function on a fresh @ref loop_t and @p arguments when @p settings select
 *      @p name , then prints its row, answering whether it failed.
 *
 *  Takes a function pointer rather than any callable, so a benchmark whose signature drifts fails
 *  to compile here.
 */
template <typename... parameters_types_>
[[nodiscard]] bool run(bench_settings_t const& settings, std::string_view name,
                       void (*bench_function)(loop_t&, parameters_types_...),
                       std::type_identity_t<parameters_types_>... arguments) noexcept {
    if (!settings.filter.selects(name)) return false;
    loop_t loop {settings.warmup, settings.time_limit};
    bench_function(loop, arguments...);
    loop.report(name);
    return loop.failed();
}

} // namespace unum::uform::bench
