/**
 *  @file test/harness.hpp
 *  @author Ash Vardanian
 *  @date July 5, 2026
 *  @brief Shared checks for the C++ and CUDA suites: the run, its settings, checks, skips, crashes.
 *
 *  A suite's @c main calls @ref start_test_run , which prints @ref probe_machine and reads the
 *  settings, then @ref run_test_case runs each case, printing one verdict line and, under a
 *  failure, a rerun line on stderr. The run in progress, @ref test_run , is the harness's one
 *  mutable global: checks fire inside helpers and on worker threads, so they count into it rather
 *  than into a tally threaded through every call. A check's message mirrors @ref expected::fail , a
 *  format string plus its operands with the call site captured through @ref format_at_t , so a
 *  failing check names its file, line and values, and a passing one prints nothing at all. A case
 *  that cannot run answers @ref case_result_t::skipped .
 *
 *  Every setting is a @c UFORM_ variable, and a flag of the same name overrides it:
 *
 *  @verbatim
 *  Variable               Flag               Default  Meaning
 *  UFORM_SEED             --seed             42       Seed every draw derives from, an unsigned integer or random
 *  UFORM_SCALE            --scale            1        Scales sweeps and iteration counts: 0.1 for a smoke run, 10
 *                                                     for a thorough one
 *  UFORM_FILTER           --filter           none     Cases to run, by label, as an ECMAScript regex
 *  UFORM_TRAP_ON_FAILURE  --trap-on-failure  off      Stops in a debugger at the first failed check
 *  UFORM_NO_SKIPS         --no-skips         off      Exits with 3 when any case skipped
 *  UFORM_DEVICES          --devices          derived  Comma-separated UForm device ordinals the GPU cases run on;
 *                                                     unset takes the first listed device of every vendor
 *  @endverbatim
 */
#pragma once

#include <cctype>  // `std::tolower`
#include <cmath>   // `std::fabs`, `std::fmax`, `std::llround`
#include <cstdint> // `std::uint64_t`
#include <cstdio>  // `std::FILE`, `std::setvbuf`, `std::fflush`, `stdout`
#include <cstdlib> // `std::getenv`, `std::exit`, `std::strtod`, `std::abort`
#include <cstring> // `std::strcmp`, `std::strlen`

#include <algorithm>       // `std::max`
#include <atomic>          // `std::atomic` — the run's failed checks
#include <charconv>        // `std::from_chars`
#include <chrono>          // `std::chrono::steady_clock`, `std::chrono::milliseconds` — the wait limit, the poll pause
#include <filesystem>      // `std::filesystem::remove_all` — the scratch directories
#include <format>          // `std::vformat_to`, `std::make_format_args`
#include <functional>      // `std::invoke`
#include <random>          // `std::random_device`
#include <regex>           // `std::regex`, `std::regex_search`
#include <source_location> // `std::source_location`
#include <span>            // `std::span` — the oracles' host staging
#include <string_view>     // `std::string_view`
#include <system_error>    // `std::errc`, `std::error_code`
#include <thread>          // `std::this_thread::sleep_for`, `std::this_thread::yield`
#include <type_traits>     // `std::invoke_result_t`, `std::is_void_v`

#include <fmt/format.h> // `fmt::println`

#include <numkong/cast.hpp> // `nk::cast` — the oracles' BFloat16 rounding and widening
#include <numkong/dot.hpp>  // `nk::dot` — the oracles' products

#include "uform/types.h" // `uform_version_name`, `uform_capabilities_available`, `uform_gpu_devices`

#include "uform/utilities/array.hpp" // `per_gpu_array` — the devices
#include "uform/utilities/types.hpp" // `format_at_t`, `format_into_t`, `sz`, the scalar aliases

#if UFORM_OS_WINDOWS_
/*  Winsock 2 first keeps Winsock 1.1 out, unless a user's earlier `<windows.h>` brought it in. */
#if !defined(_WINSOCKAPI_)
#include <winsock2.h>
#endif
#include <intrin.h>  // `__debugbreak`
#include <io.h>      // `_write`
#include <windows.h> // `SetUnhandledExceptionFilter`
#elif !UFORM_OS_WASI_
#include <csignal>  // `std::signal`, `std::raise`, `SIGTRAP`
#include <unistd.h> // `write`, `mkdtemp`
#endif              // UFORM_OS_WINDOWS_
#if (defined(__GLIBC__) || UFORM_OS_APPLE_) && __has_include(<execinfo.h>)
#include <execinfo.h> // `backtrace`, `backtrace_symbols_fd`
#endif

/** Prints a StringZilla view as the text it spans. */
template <>
struct fmt::formatter<ashvardanian::stringzilla::string_view_t> : fmt::formatter<std::string_view> {
    auto format(std::string_view text, fmt::format_context& context) const {
        return fmt::formatter<std::string_view>::format(text, context);
    }
};

/** Prints an owning StringZilla string as the text it holds. */
template <>
struct fmt::formatter<ashvardanian::stringzilla::string_t> : fmt::formatter<std::string_view> {
    auto format(std::string_view text, fmt::format_context& context) const {
        return fmt::formatter<std::string_view>::format(text, context);
    }
};

namespace unum::uform::test {

/** Stops in an attached debugger, or dumps core without one; crash handlers leave SIGTRAP alone. */
inline void trap_into_debugger() noexcept {
#if UFORM_OS_WINDOWS_
    __debugbreak();
#elif UFORM_OS_WASI_
    __builtin_trap();
#else
    std::raise(SIGTRAP);
#endif
}

/** Arms the crash handlers, which name the case a crash ends. */
inline void install_test_signal_handlers() noexcept;

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

/** Writes one line to @p stream, formatted from @p format and @p args, then its newline. */
template <typename... args_types_>
void print_line(std::FILE* stream, fmt::format_string<args_types_...> format, args_types_&&... args) {
    fmt::println(stream, format, std::forward<args_types_>(args)...);
}

#pragma region Shared Settings

/** Exits with one line naming a setting that does not parse, so a bad value never runs anything. */
[[noreturn]] inline void exit_unparsed(char const* variable, char const* text, char const* expected) noexcept {
    print_line(stderr, "{}=\"{}\" does not parse, expected {}", variable, text, expected);
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
    auto const [end, parse_status] = std::from_chars(spelled.data(), spelled.data() + spelled.size(), count);
    if (parse_status != std::errc {} || end != spelled.data() + spelled.size())
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
    auto const [end, parse_status] = std::from_chars(spelled.data(), spelled.data() + spelled.size(), count);
    std::string_view const unit = spelled.substr(static_cast<std::size_t>(end - spelled.data()));
    if (parse_status == std::errc {} && count != 0) {
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
    auto const [end, parse_status] = std::from_chars(spelled.data(), spelled.data() + spelled.size(), seed);
    if (parse_status != std::errc {} || end != spelled.data() + spelled.size())
        exit_unparsed(variable, text, "an unsigned integer or random");
    return seed;
}

/** Which labels a run selects: those its regex matches somewhere, every label while unset. */
struct regex_filter_t {
    std::string_view text;
    std::regex regex {""};

    bool selects(std::string_view label) const noexcept { return std::regex_search(label.begin(), label.end(), regex); }
};

/** The filter a setting spells, compiled once; one that does not compile exits as a bad setting. */
inline regex_filter_t read_regex_filter_setting(int argc, char** argv, char const* variable,
                                                std::string_view flag) noexcept {
    regex_filter_t filter;
    char const* text = setting_text(argc, argv, variable, flag);
    if (!text) return filter;
    filter.text = text;
#if defined(__cpp_exceptions) && __cpp_exceptions
    try {
        filter.regex.assign(filter.text.begin(), filter.text.end());
    }
    catch (std::regex_error const&) {
        exit_unparsed(variable, text, "a regular expression");
    }
#else
    filter.regex.assign(filter.text.begin(), filter.text.end());
#endif
    return filter;
}

/** What every suite and bench reads before its own settings; @c program starts the rerun lines. */
struct common_settings_t {
    std::uint64_t seed = 42;
    regex_filter_t regex_filter;
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
 *  @brief SplitMix64 over one 64-bit counter, seeded by a @ref stream_key , bit-identical to the
 *      JavaScript generator NumKong ships.
 *
 *  Distributions are not specified to draw the same sequence on every standard library, so a
 *  failure would stop reproducing from its seed when the toolchain changed; @ref below is the
 *  portable bounded draw.
 */
struct split_mix_generator_t {
    using result_type = std::uint64_t;
    std::uint64_t counter = 0;

    explicit split_mix_generator_t(std::uint64_t key) noexcept : counter(key) {}
    static constexpr result_type(min)() noexcept { return 0; }
    static constexpr result_type(max)() noexcept { return ~result_type {0}; }
    result_type operator()() noexcept { return split_mix_bits(counter += 0x9E3779B97F4A7C15ull); }

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

/** Every GPU @c uform_gpu_devices lists, each at its UForm device ordinal. */
[[nodiscard]] inline util::per_gpu_array<uform_gpu_device_t> listed_gpu_devices() noexcept {
    util::per_gpu_array<uform_gpu_device_t> listed;
    std::size_t count = 0;
    uform_outcome_t outcome {};
    [[maybe_unused]] util::status_t const sized = listed.resize(listed.capacity());
    uform_gpu_devices_t listing {&outcome, listed.begin(), listed.size(), &count};
    uform_gpu_devices(&listing);
    [[maybe_unused]] util::status_t const trimmed = listed.resize(fu::min_of<std::size_t>(count, listed.capacity()));
    return listed;
}

/** The ordinals @p text lists, comma-separated, or the first listed device of every vendor when it
 *  is null; exits with status 1 on text that does not parse or names a device past the listing. */
[[nodiscard]] inline util::per_gpu_array<util::i32_t> parse_devices(char const* text) noexcept {
    util::per_gpu_array<uform_gpu_device_t> const listed = listed_gpu_devices();
    std::size_t const listed_count = listed.size();
    util::per_gpu_array<util::i32_t> devices;
    if (!text) {
        for (std::size_t ordinal = 0; ordinal != listed_count; ++ordinal) {
            if (ordinal != 0 && listed[ordinal - 1].vendor == listed[ordinal].vendor) continue;
            [[maybe_unused]] util::status_t const pushed = devices.push_back(static_cast<util::i32_t>(ordinal));
        }
        return devices;
    }
    std::string_view const spelled {text};
    for (std::size_t begin = 0; begin <= spelled.size();) {
        std::size_t const end = fu::min_of(spelled.find(',', begin), spelled.size());
        util::u32_t ordinal = 0;
        auto const [parsed_end, parse_status] = std::from_chars(spelled.data() + begin, spelled.data() + end, ordinal);
        if (parse_status != std::errc {} || parsed_end != spelled.data() + end)
            exit_unparsed("UFORM_DEVICES", text, "unsigned integers like 0,1");
        if (ordinal >= listed_count ||
            devices.push_back(static_cast<util::i32_t>(ordinal)) != util::status_code_t::success_k) {
            fmt::println(stderr, "UFORM_DEVICES names device {}, past the {} listed", ordinal, listed_count);
            std::exit(1);
        }
        begin = end + 1;
    }
    return devices;
}

/** What a suite reads: the common settings, the run's own, and the GPUs its device cases run on. */
struct test_settings_t : common_settings_t {
    double scale = 1.0;
    bool trap_on_failure = false;
    bool refuse_skips = false;

    /** UForm device ordinals the GPU cases run on, positions in @c uform_gpu_devices . */
    util::per_gpu_array<util::i32_t> devices {};
};

/** Reads every @ref test_settings_t field, exiting with status 1 on the first unparsable one. */
inline test_settings_t read_test_settings(int argc, char** argv) noexcept;

#pragma region Shared Run

/** What a case reports beyond its checks: that it ran, or why it skipped. */
struct case_result_t {
    char const* skip_reason = nullptr;

    [[nodiscard]] static constexpr case_result_t skipped(char const* reason) noexcept { return {reason}; }
};

/**
 *  @brief The run in progress: the harness's one sanctioned mutable global.
 *
 *  Checks fire on worker threads and inside helpers as well as in a case's own body, so the run
 *  counts failed checks in one place and @ref run_test_case diffs that count around each case,
 *  rather than threading a tally through every call. After a crash, printing @c test_run from a
 *  core dump names the case in progress and the last failed check, and the signal handlers print
 *  the case too.
 */
struct test_run_t {
    test_settings_t settings {};
    char const* current_case = nullptr;
    std::atomic<std::size_t> failed_checks {0};
    std::atomic<char const*> last_failed_file {nullptr};
    std::atomic<std::uint_least32_t> last_failed_line {0};
    std::size_t passed_cases = 0;
    std::size_t failed_cases = 0;
    std::size_t skipped_cases = 0;
};

inline test_run_t test_run;

/** How long any wait in a test may take; a passing case never waits it out, so it bounds hangs. */
inline constexpr std::chrono::seconds wait_limit_k {120};

/** Yields until @p predicate holds; false when @ref wait_limit_k passed first. */
template <typename predicate_type_>
bool spin_until(predicate_type_&& predicate) noexcept {
    auto const deadline = std::chrono::steady_clock::now() + wait_limit_k;
    while (!predicate()) {
        if (std::chrono::steady_clock::now() > deadline) return false;
        std::this_thread::yield();
    }
    return true;
}

/** @p baseline iterations under the @c SCALE setting, never fewer than one. */
inline std::size_t scaled_iterations(std::size_t baseline) noexcept {
    double const scaled = static_cast<double>(baseline) * test_run.settings.scale;
    return scaled < 1 ? 1 : static_cast<std::size_t>(std::llround(scaled));
}

/** The step that visits a @p coverage fraction of a sweep, at least one. */
inline std::size_t sweep_stride(double coverage) noexcept {
    double const stride = 1.0 / coverage;
    return stride < 1 ? 1 : static_cast<std::size_t>(std::llround(stride));
}

/** The @p index -th stream of the case in progress, so a filtered run draws what a full one did. */
inline std::uint64_t case_stream_key(std::uint64_t index = 0) noexcept {
    return stream_key(test_run.settings.seed, test_run.current_case ? test_run.current_case : "", index);
}

/** Line-buffers stdout, prints the machine, reads the settings, arms the crash handlers and
 *  prints the settings line, before the first case. */
inline void start_test_run(int argc, char** argv) noexcept {
    std::setvbuf(stdout, nullptr, _IOLBF, BUFSIZ); // before any output, or the C library keeps its buffering
    print(probe_machine());
    test_run.settings = read_test_settings(argc, argv);
    install_test_signal_handlers();
    print_line(stdout, "- Seed: {}, scale: {}", test_run.settings.seed, test_run.settings.scale);
    print_line(
        stdout, "- Filter: {}",
        test_run.settings.regex_filter.text.empty() ? std::string_view("none") : test_run.settings.regex_filter.text);
}

/** Counts one failed check at @p where , and traps when the run asked to stop at the first. */
inline void record_failed_check(std::source_location where) noexcept {
    std::fflush(stdout); // so a log that merges both streams keeps the failure under its case
    print_line(stderr, "  check failed at {}:{}", where.file_name(), where.line());
    test_run.last_failed_file.store(where.file_name(), std::memory_order_relaxed);
    test_run.last_failed_line.store(where.line(), std::memory_order_relaxed);
    test_run.failed_checks.fetch_add(1, std::memory_order_relaxed);
    if (test_run.settings.trap_on_failure) trap_into_debugger();
}

/** Holds when @p is_satisfied does; otherwise counts a failed check at the caller's line. */
inline bool expect_true(bool is_satisfied, std::source_location where = std::source_location::current()) noexcept {
    if (!is_satisfied) record_failed_check(where);
    return is_satisfied;
}

/** A case label spelled as a regex that matches only itself, every metacharacter escaped. */
struct regex_escaped_t {
    std::string_view label;
};

} // namespace unum::uform::test

/** Writes a @ref regex_escaped_t straight to the output, a backslash before each metacharacter. */
template <>
struct fmt::formatter<unum::uform::test::regex_escaped_t> {
    constexpr auto parse(fmt::format_parse_context& context) { return context.begin(); }
    template <typename context_type_>
    auto format(auto const& escaped, context_type_& context) const {
        auto output = context.out();
        for (char const character : escaped.label) {
            if (std::string_view {"\\^$.|?*+()[]{}"}.find(character) != std::string_view::npos) *output++ = '\\';
            *output++ = character;
        }
        return output;
    }
};

namespace unum::uform::test {

/** Prints one case's verdict, and the line that reruns it alone when it failed. */
inline void report_test_case(char const* label, std::size_t failed_checks, char const* skip_reason,
                             double seconds) noexcept {
    if (skip_reason) {
        print_line(stdout, "- {} ... skipped: {}", label, skip_reason);
        return;
    }
    if (failed_checks == 0) {
        print_line(stdout, "- {} ... ok ({:.2f} s)", label, seconds);
        return;
    }
    std::fflush(stdout);
    print_line(stderr, "- {} ... FAILED: {} failed check(s)", label, failed_checks);
    print_line(stderr, "  rerun: {} --seed {} --filter '^{}$'", test_run.settings.program, test_run.settings.seed,
               regex_escaped_t {label});
}

/** Runs one case unless the filter leaves it out, and records whether it passed or skipped. */
template <typename case_function_type_, typename... argument_types_>
void run_test_case(char const* label, case_function_type_&& case_function, argument_types_&&... arguments) noexcept {
    if (!test_run.settings.regex_filter.selects(label)) return;
    test_run.current_case = label;
    std::size_t const failed_before = test_run.failed_checks.load();
    auto const started = std::chrono::steady_clock::now();
    case_result_t result {};
    using returned_type = std::invoke_result_t<case_function_type_, argument_types_...>;
    if constexpr (std::is_void_v<returned_type>)
        std::invoke(case_function, std::forward<argument_types_>(arguments)...);
    else result = std::invoke(case_function, std::forward<argument_types_>(arguments)...);
    std::chrono::duration<double> const elapsed = std::chrono::steady_clock::now() - started;
    std::size_t const failed_checks = test_run.failed_checks.load() - failed_before;
    if (result.skip_reason) ++test_run.skipped_cases;
    else if (failed_checks != 0) ++test_run.failed_cases;
    else ++test_run.passed_cases;
    report_test_case(label, failed_checks, result.skip_reason, elapsed.count());
    test_run.current_case = nullptr;
}

/** The exit code: 1 when a case or a check outside any case failed, or nothing ran; 3 when
 *  skips are refused and one happened; else 0. */
inline int finish_test_run() noexcept {
    print_line(stdout, "{} passed, {} failed, {} skipped, seed {}", test_run.passed_cases, test_run.failed_cases,
               test_run.skipped_cases, test_run.settings.seed);
    bool const failed = test_run.failed_cases != 0 || test_run.failed_checks.load() != 0;
    if (failed || test_run.passed_cases + test_run.skipped_cases == 0) return 1;
    if (test_run.skipped_cases != 0 && test_run.settings.refuse_skips) return 3;
    return 0;
}

#pragma endregion Shared Run

inline test_settings_t read_test_settings(int argc, char** argv) noexcept {
    test_settings_t settings;
    settings.seed = read_seed_setting(argc, argv, "UFORM_SEED", "--seed");
    settings.regex_filter = read_regex_filter_setting(argc, argv, "UFORM_FILTER", "--filter");
    settings.scale = read_real_setting(argc, argv, "UFORM_SCALE", "--scale", 1.0);
    settings.trap_on_failure = read_flag_setting(argc, argv, "UFORM_TRAP_ON_FAILURE", "--trap-on-failure");
    settings.refuse_skips = read_flag_setting(argc, argv, "UFORM_NO_SKIPS", "--no-skips");
    settings.devices = parse_devices(setting_text(argc, argv, "UFORM_DEVICES", "--devices"));
    settings.program = argc != 0 ? argv[0] : "";
    return settings;
}

/** The reason every case needing a GPU skips with on a machine that shows none. */
inline constexpr char const* no_gpu_device_k = "no GPU device";

/** Pauses 50 ms between two polls of a mesh exchange directory, as the trainer does. */
inline void pause_between_polls() noexcept { std::this_thread::sleep_for(std::chrono::milliseconds {50}); }

/** Prints @p context formatted with @p args , four spaces in, under a failed check's location. */
template <typename... args_types_>
void print_context_(util::format_at_t const& context, args_types_&&... args) noexcept {
    sz::string_t text;
    std::vformat_to(util::format_into_t {&text}, context.format, std::make_format_args(args...));
    std::fprintf(stderr, "    %.*s\n", static_cast<int>(text.size()), text.data());
}

/** Holds when @p is_met does, read as a condition, so an @ref expected reads as itself; otherwise
 *  counts a failed check at the caller's line and prints the formatted @p context and @p args. */
template <typename condition_type_, typename... args_types_>
bool expect_true(condition_type_ const& is_met, util::format_at_t context, args_types_&&... args) noexcept {
    if (is_met) return true;
    record_failed_check(context.location);
    print_context_(context, args...);
    return false;
}

/** Holds when @p actual equals @p wanted ; otherwise counts a failed check at the caller's line and
 *  prints @p context and both values under it. */
template <typename actual_type_, typename wanted_type_, typename... args_types_>
bool expect_equal(actual_type_ const& actual, wanted_type_ const& wanted, util::format_at_t context,
                  args_types_&&... args) noexcept {
    if (actual == wanted) return true;
    record_failed_check(context.location);
    print_context_(context, args...);
    print_context_("expected {}, got {}", wanted, actual);
    return false;
}

/** Holds when @p actual and @p wanted differ by at most @p tolerance times the larger of their
 *  magnitudes and 1; otherwise counts a failed check and prints @p context with both values. */
template <typename... args_types_>
bool expect_near(util::f64_t actual, util::f64_t wanted, util::f64_t tolerance, util::format_at_t context,
                 args_types_&&... args) noexcept {
    if (std::fabs(actual - wanted) <= tolerance * std::fmax(std::fmax(std::fabs(actual), std::fabs(wanted)), 1.0))
        return true;
    record_failed_check(context.location);
    print_context_(context, args...);
    print_context_("expected {:.6g} within {:.4g}, got {:.6g}", wanted, tolerance, actual);
    return false;
}

#if UFORM_OS_POSIX_

/** A directory of its own under @c TMPDIR , named after the case in progress, removed with all it
 *  holds when it goes out of scope; @ref path stays empty when it could not be made. */
struct scratch_directory_t {
    std::filesystem::path path;

    scratch_directory_t() noexcept {
        char const* const base = std::getenv("TMPDIR");
        std::string pattern = fmt::format("{}/uform-{}-XXXXXX", base && *base ? base : "/tmp",
                                          test_run.current_case ? test_run.current_case : "run");
        if (::mkdtemp(pattern.data())) path = pattern;
    }
    scratch_directory_t(scratch_directory_t const&) = delete;
    scratch_directory_t& operator=(scratch_directory_t const&) = delete;
    ~scratch_directory_t() noexcept {
        std::error_code ignored;
        if (!path.empty()) std::filesystem::remove_all(path, ignored);
    }
};

#endif // UFORM_OS_POSIX_

#if !UFORM_OS_WASI_

/** Writes @p text to stderr in one @c write , which a crash handler may call, unlike @c fprintf. */
inline void write_to_stderr_(char const* text) noexcept {
    std::size_t const length = std::strlen(text);
#if UFORM_OS_WINDOWS_
    [[maybe_unused]] int const written = _write(2, text, static_cast<unsigned>(length));
#else
    [[maybe_unused]] ssize_t const written = ::write(2, text, length);
#endif
}

/** Writes the line naming the case a crash of kind @p name ends, from @ref test_run . */
inline void write_crash_notice_(char const* name) noexcept {
    write_to_stderr_("\n*** ");
    write_to_stderr_(name);
    write_to_stderr_(" in case '");
    write_to_stderr_(test_run.current_case ? test_run.current_case : "none");
    write_to_stderr_("' - backtrace follows ***\n");
}

#endif // !UFORM_OS_WASI_

#if UFORM_OS_WINDOWS_

/** Names the case an unhandled structured exception ends, then lets Windows report it as usual. */
inline LONG WINAPI report_unhandled_exception_(EXCEPTION_POINTERS*) noexcept {
    write_crash_notice_("EXCEPTION");
    return EXCEPTION_CONTINUE_SEARCH;
}

#elif !UFORM_OS_WASI_

/** Names the case @p signal_number ends and prints a backtrace, then re-raises it, so the shell
 *  still sees the signal and a core dump still lands. */
inline void report_fatal_signal_(int signal_number) noexcept {
    char const* name = "SIGNAL";
    switch (signal_number) {
    case SIGSEGV: name = "SIGSEGV"; break;
    case SIGABRT: name = "SIGABRT"; break;
    case SIGILL: name = "SIGILL"; break;
    case SIGFPE: name = "SIGFPE"; break;
#if defined(SIGBUS)
    case SIGBUS: name = "SIGBUS"; break;
#endif
    }
    write_crash_notice_(name);
#if (defined(__GLIBC__) || UFORM_OS_APPLE_) && __has_include(<execinfo.h>)
    void* frames[64];
    backtrace_symbols_fd(frames, backtrace(frames, 64), 2);
#endif
    std::signal(signal_number, SIG_DFL);
    std::raise(signal_number);
}

#endif // UFORM_OS_WINDOWS_

inline void install_test_signal_handlers() noexcept {
#if UFORM_OS_WINDOWS_
    SetUnhandledExceptionFilter(report_unhandled_exception_);
#elif !UFORM_OS_WASI_
    for (int const signal_number : {SIGSEGV, SIGABRT, SIGILL, SIGFPE}) std::signal(signal_number, report_fatal_signal_);
#if defined(SIGBUS)
    std::signal(SIGBUS, report_fatal_signal_);
#endif
#endif // UFORM_OS_WINDOWS_
}

/** The Float64 views the host oracles read, beside the core's Float32 ones. */
using f64r1_view_t = nk::tensor_view<util::f64_t, 1>;
using f64r2_view_t = nk::tensor_view<util::f64_t, 2>;
using f64r2_span_t = nk::tensor_span<util::f64_t, 2>;
using f64r3_view_t = nk::tensor_view<util::f64_t, 3>;
using f64r3_span_t = nk::tensor_span<util::f64_t, 3>;
using f64r4_view_t = nk::tensor_view<util::f64_t, 4>;
using f64r4_span_t = nk::tensor_span<util::f64_t, 4>;

/** Factory function to allocate zeroed tensors, aborting on exhaustion. */
template <typename tensor_type_>
[[nodiscard]] tensor_type_ make_zeros(std::initializer_list<std::size_t> extents,
                                      typename tensor_type_::allocator_type allocator = {}) noexcept {
    auto [tensor, status] = tensor_type_::zeros(extents, allocator);
    if (nk::failed(status)) std::abort();
    return std::move(tensor);
}

/** Factory function to allocate tensors a copy or a kernel fills, aborting on exhaustion. */
template <typename tensor_type_>
[[nodiscard]] tensor_type_ make_uninitialized(std::initializer_list<std::size_t> extents,
                                              typename tensor_type_::allocator_type allocator = {}) noexcept {
    auto [tensor, status] = tensor_type_::uninitialized(extents, allocator);
    if (nk::failed(status)) std::abort();
    return std::move(tensor);
}

/** ⟨a,b⟩ of two equally long contiguous Float64 runs, compensated by NumKong; runs of unequal
 *  lengths fail the case at the caller's line. */
[[nodiscard]] inline util::f64_t dot(auto const& a, auto const& b,
                                     std::source_location where = std::source_location::current()) noexcept {
    auto const [product, status] = nk::dot<nk::f64_t>(a, b);
    expect_true(nk::succeeded(status), where);
    return product;
}

/** Narrow @p values onto the BFloat16 grid in place, leaving their bit patterns in @p staging for
 *  a device upload. */
[[nodiscard]] inline bool narrow_to_bf16(std::span<util::f64_t> values, std::span<nk_bf16_t> staging) noexcept {
    return nk::cast<nk::f64_t, nk::bf16_t>(values, staging) == nk::status_t::success_k &&
           nk::cast<nk::bf16_t, nk::f64_t>(std::span<nk_bf16_t const> {staging}, values) == nk::status_t::success_k;
}

/** Widen the BFloat16 bit patterns of a device download in @p staging into @p values . */
[[nodiscard]] inline bool widen_from_bf16(std::span<nk_bf16_t const> staging, std::span<util::f64_t> values) noexcept {
    return nk::cast<nk::bf16_t, nk::f64_t>(staging, values) == nk::status_t::success_k;
}

} // namespace unum::uform::test
