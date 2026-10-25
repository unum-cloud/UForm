/**
 *  @file test/harness.hpp
 *  @author Ash Vardanian
 *  @date July 5, 2026
 *  @brief Shared checks for the C++ and CUDA suites: failure counts, skips, settings and totals.
 *
 *  A suite's @c main makes one @ref test_suite_t and passes it by reference into every case
 *  through @ref test_suite_t::run , which prints one result line per case and a rerun line under
 *  each failure; there is no file-scope counter and no singleton. @ref test_suite_t::check mirrors
 *  @ref expected::fail , a format string plus its operands with the call site captured through
 *  @ref format_at_t , so a failing check names its file, line and values, and a passing one is
 *  silent.
 *
 *  Every suite reads its settings once, in @ref test_suite_t::make :
 *
 *  @verbatim
 *  Variable           Default  Meaning
 *  UFORM_SEED         42       Seed the suite's draws derive from, an unsigned integer or random
 *  UFORM_FILTER       none     Cases to run, by name, as an ECMAScript regex or else as a substring
 *  UFORM_TEST_FILTER  none     The same filter, read when UFORM_FILTER is unset
 *  UFORM_DEVICES      derived  Comma-separated UForm device ordinals the GPU cases run on; unset takes
 *                              the first listed device of every vendor
 *  @endverbatim
 */
#pragma once

#include <cmath>   // `std::fabs`, `std::fmax`
#include <cstdio>  // `std::setvbuf`, `stdout`
#include <cstdlib> // `std::getenv`, `std::exit`

#include <algorithm>   // `std::min`
#include <charconv>    // `std::from_chars`
#include <format>      // `std::vformat_to`, `std::make_format_args`
#include <random>      // `std::random_device`
#include <regex>       // `std::regex`, `std::regex_search`
#include <span>        // `std::span` — the oracles' host staging
#include <string_view> // `std::string_view`
#include <type_traits> // `std::type_identity_t`

#include <fmt/format.h> // `fmt::println`
#include <fmt/ranges.h> // `fmt::join` — the devices

#include <numkong/cast.hpp> // `nk::cast` — the oracles' BFloat16 rounding and widening
#include <numkong/dot.hpp>  // `nk::dot` — the oracles' products

#include "uform/types.h" // `uform_version_name`, `uform_capabilities_available`, `uform_gpu_devices`

#include "uform/core/types.hpp" // `format_at_t`, `format_into_t`, `sz`, the scalar aliases
#include "uform/core/array.hpp" // `per_gpu_array` — the devices

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

namespace unum::uform::test {

using namespace unum::uform::core;

/** The reason every case needing a GPU skips with on a machine that shows none. */
inline constexpr char const *no_cuda_device_k = "no CUDA device detected";

/** The file name of a `__FILE__`-style path, the trailing component after the last `/`. */
[[nodiscard]] constexpr std::string_view path_basename(std::string_view path) noexcept {
    std::size_t const slash = path.rfind('/');
    return slash == std::string_view::npos ? path : path.substr(slash + 1);
}

/** The text of the environment variable @p name , empty when it is unset. */
[[nodiscard]] inline std::string_view environment_text(char const *name) noexcept {
    char const *const text = std::getenv(name);
    return text == nullptr ? std::string_view {} : std::string_view {text};
}

/** A regex over case names, or a plain substring when it does not compile; empty selects all. */
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

    /** Whether the case named @p name runs. */
    [[nodiscard]] bool selects(std::string_view name) const noexcept {
        return text.empty() || std::regex_search(name.begin(), name.end(), pattern) ||
               name.find(text) != std::string_view::npos;
    }
};

/** Every setting a suite reads, its default as the initializer, filled by @ref read_settings . */
struct settings_t {

    /** Cases to run. */
    filter_t filter {};

    /** Seed every draw of the suite descends from. */
    u64_t seed = 42;

    /** UForm device ordinals the GPU cases run on, positions in @c uform_gpu_devices . */
    per_gpu_array<i32_t> devices {};
};

/** Every GPU @c uform_gpu_devices lists, each at its UForm device ordinal. */
[[nodiscard]] inline per_gpu_array<uform_gpu_device_t> listed_gpu_devices() noexcept {
    per_gpu_array<uform_gpu_device_t> listed;
    std::size_t count = 0;
    uform_outcome_t outcome {};
    [[maybe_unused]] status_t const sized = listed.resize(listed.capacity());
    uform_gpu_devices_t listing {&outcome, listed.begin(), listed.size(), &count};
    uform_gpu_devices(&listing);
    [[maybe_unused]] status_t const trimmed = listed.resize(std::min<std::size_t>(count, listed.capacity()));
    return listed;
}

/** The ordinals @c UFORM_DEVICES lists, comma-separated, or the first listed device of every vendor
 *  when it is unset; exits with status 1 on text that does not parse or names a device past the
 *  listing. */
[[nodiscard]] inline per_gpu_array<i32_t> read_devices() noexcept {
    per_gpu_array<uform_gpu_device_t> const listed = listed_gpu_devices();
    std::size_t const listed_count = listed.size();
    per_gpu_array<i32_t> devices;
    std::string_view const text = environment_text("UFORM_DEVICES");
    if (text.empty()) {
        for (std::size_t ordinal = 0; ordinal != listed_count; ++ordinal) {
            if (ordinal != 0 && listed[ordinal - 1].vendor == listed[ordinal].vendor) continue;
            [[maybe_unused]] status_t const pushed = devices.push_back(static_cast<i32_t>(ordinal));
        }
        return devices;
    }
    for (std::size_t begin = 0; begin <= text.size();) {
        std::size_t const end = std::min(text.find(',', begin), text.size());
        u32_t ordinal = 0;
        auto const [parsed_end, parse_status] = std::from_chars(text.data() + begin, text.data() + end, ordinal);
        if (parse_status != std::errc {} || parsed_end != text.data() + end) {
            fmt::println(stderr, "UFORM_DEVICES=\"{}\" does not parse, expected unsigned integers like 0,1", text);
            std::exit(1);
        }
        if (ordinal >= listed_count || devices.push_back(static_cast<i32_t>(ordinal)) != status_t::success_k) {
            fmt::println(stderr, "UFORM_DEVICES names device {}, past the {} listed", ordinal, listed_count);
            std::exit(1);
        }
        begin = end + 1;
    }
    return devices;
}

/** Reads every @ref settings_t variable, exiting with status 1 on a malformed @c UFORM_SEED or
 *  @c UFORM_DEVICES . */
[[nodiscard]] inline settings_t read_settings() noexcept {
    settings_t settings;
    std::string_view filter_text = environment_text("UFORM_FILTER");
    if (filter_text.empty()) filter_text = environment_text("UFORM_TEST_FILTER");
    settings.filter = filter_t::make(filter_text);
    std::string_view const seed_text = environment_text("UFORM_SEED");
    if (seed_text == "random") settings.seed = std::random_device {}();
    else if (!seed_text.empty()) {
        char const *const seed_end = seed_text.data() + seed_text.size();
        auto const [parsed_end, parse_status] = std::from_chars(seed_text.data(), seed_end, settings.seed);
        if (parse_status != std::errc {} || parsed_end != seed_end) {
            fmt::println(stderr, "UFORM_SEED=\"{}\" does not parse, expected an unsigned integer or random", seed_text);
            std::exit(1);
        }
    }
    settings.devices = read_devices();
    return settings;
}

/**
 *  @brief The counts one suite's @c main owns and every case rolls into, made by @ref make and
 *      threaded by reference into every case.
 *
 *  A check outside any case fails the suite too, through @ref finish .
 */
struct test_suite_t {

    /** What @ref read_settings parsed. */
    settings_t settings {};

    /** The binary's @c argv[0] , which turns a rerun line into a command. */
    std::string_view program {};

    /** Cases that ran without a failure. */
    u32_t passed = 0;

    /** Cases that did not run, or stopped short. */
    u32_t skipped = 0;

    /** Cases with at least one failed check. */
    u32_t failed = 0;

    /** Failed checks across the whole suite, inside cases and out. */
    u32_t failures = 0;

    /** Why the running case stopped short, or null while it runs. */
    char const *case_skip_reason = nullptr;

    /** Reads the settings, then prints the version, capabilities and settings, one a line. */
    [[nodiscard]] static test_suite_t make(char const *program) noexcept {
        std::setvbuf(stdout, nullptr, _IOLBF, 0); // a redirected log keeps each case line beside its failures
        test_suite_t suite;
        suite.settings = read_settings();
        suite.program = program;
        fmt::println("UForm {}", uform_version_name());
        fmt::println("- Compiled for: {}", uform_capabilities_compiled());
        fmt::println("- This machine: {}", uform_capabilities_available());
        fmt::println("- Seed: {}", suite.settings.seed);
        std::string_view const filter = suite.settings.filter.text;
        fmt::println("- Filter: {}", filter.empty() ? std::string_view {"none"} : filter);
        if (suite.settings.devices.size() == 0) fmt::println("- Devices: none");
        else fmt::println("- Devices: {}", fmt::join(suite.settings.devices, ","));
        return suite;
    }

    /**
     *  @brief Runs @p case_function on @p arguments when the filter selects @p name , then prints
     *      its result line, and a rerun line under a failure.
     *
     *  Takes a function pointer rather than any callable, so a case whose signature drifts fails to
     *  compile here instead of hiding behind a lambda at every call site.
     */
    template <typename... parameters_types_>
    void run(std::string_view name, void (*case_function)(test_suite_t &, parameters_types_...),
             std::type_identity_t<parameters_types_>... arguments) noexcept {
        if (!settings.filter.selects(name)) return;
        u32_t const failures_before = failures;
        case_skip_reason = nullptr;
        case_function(*this, arguments...);
        u32_t const case_failures = failures - failures_before;
        if (case_failures != 0) {
            ++failed;
            fmt::println("FAILED  {:<32} : {} failures", name, case_failures);
            fmt::println(stderr, "  rerun: UFORM_SEED={} UFORM_FILTER='^{}$' {}", settings.seed, name, program);
        }
        else if (case_skip_reason != nullptr) skip(name, case_skip_reason);
        else {
            ++passed;
            fmt::println("passed  {:<32} : 0 failures", name);
        }
    }

    /** Stops the running case short for @p reason , so it reports skipped unless it failed. */
    void skip(char const *reason) noexcept { case_skip_reason = reason; }

    /** Reports the case @p name skipped for @p reason without running it, if selected. */
    void skip(std::string_view name, char const *reason) noexcept {
        if (!settings.filter.selects(name)) return;
        ++skipped;
        fmt::println("skipped  {:<32} : {}", name, reason);
    }

    /**
     *  @brief Counts one failure when @p condition is false, formatting @p message and @p args the
     *      way @ref expected::fail does and printing them under the call site. A pass is silent.
     *
     *  The @ref format_at_t parameter captures the caller's @c std::source_location behind the
     *  variadic pack, so a check reads `check(got == want, "peer: got {} want {}", got, want)`.
     */
    template <typename... arg_types_>
    void check(bool condition, format_at_t message, arg_types_ &&...args) noexcept {
        if (condition) return;
        sz::string_t text;
        std::vformat_to(format_into_t {&text}, message.format, std::make_format_args(args...));
        report_failure_(message, text);
    }

    /** Counts one failure when @p actual differs from @p wanted , printing both after @p label . */
    template <typename actual_type_, typename wanted_type_, typename... arg_types_>
    void check_equal(actual_type_ const &actual, wanted_type_ const &wanted, format_at_t label,
                     arg_types_ &&...args) noexcept {
        if (actual == wanted) return;
        sz::string_t text;
        std::vformat_to(format_into_t {&text}, label.format, std::make_format_args(args...));
        [[maybe_unused]] bool const formatted = format_into(text, ": expected {}, got {}", wanted, actual);
        report_failure_(label, text);
    }

    /** Counts one failure when @p actual and @p wanted differ by more than @p tolerance times the
     *  larger of their magnitudes and 1, printing both and the difference after @p label . */
    template <typename... arg_types_>
    void check_near(f64_t actual, f64_t wanted, f64_t tolerance, format_at_t label, arg_types_ &&...args) noexcept {
        f64_t const difference = std::fabs(actual - wanted);
        if (difference <= tolerance * std::fmax(std::fmax(std::fabs(actual), std::fabs(wanted)), 1.0)) return;
        sz::string_t text;
        std::vformat_to(format_into_t {&text}, label.format, std::make_format_args(args...));
        [[maybe_unused]] bool const formatted = format_into(text, ": expected {:.6g}, got {:.6g}, off by {:.4g}",
                                                            wanted, actual, difference);
        report_failure_(label, text);
    }

    /** Prints the totals and answers the exit code: 1 on any failed check, or on a filter that
     *  selected no case, since a mistyped filter would otherwise read like a green suite. */
    [[nodiscard]] int finish() const noexcept {
        fmt::println("total {} passed, {} skipped, {} failed, seed {}", passed, skipped, failed, settings.seed);
        if (failures != 0) return 1;
        if (!settings.filter.text.empty() && passed + skipped == 0) {
            fmt::println(stderr, "UFORM_FILTER='{}' matched no case, so nothing ran", settings.filter.text);
            return 1;
        }
        return 0;
    }

  private:

    /** Counts the failure and prints the formatted @p text under the call site of @p site . */
    void report_failure_(format_at_t const &site, sz::string_t const &text) noexcept {
        ++failures;
        fmt::println(stderr, "  check failed at {}:{}: {}", path_basename(site.location.file_name()),
                     site.location.line(), text);
    }
};

/** The Float64 borrows the host oracles read, beside the core's Float32 ones. */
using f64r1_view_t = nk::tensor_view<f64_t, 1>;
using f64r2_view_t = nk::tensor_view<f64_t, 2>;
using f64r2_span_t = nk::tensor_span<f64_t, 2>;
using f64r3_view_t = nk::tensor_view<f64_t, 3>;
using f64r3_span_t = nk::tensor_span<f64_t, 3>;
using f64r4_view_t = nk::tensor_view<f64_t, 4>;
using f64r4_span_t = nk::tensor_span<f64_t, 4>;

/** ⟨a,b⟩ of two equally long contiguous Float64 runs, compensated by NumKong. */
[[nodiscard]] inline f64_t dot(auto const &a, auto const &b) noexcept { return nk::dot<nk::f64_t>(a, b).value; }

/** Narrow @p values onto the BFloat16 grid in place, leaving their bit patterns in @p staging for a
 *  device upload. */
[[nodiscard]] inline bool narrow_to_bf16(std::span<f64_t> values, std::span<nk_bf16_t> staging) noexcept {
    return nk::cast<nk::f64_t, nk::bf16_t>(values, staging) == nk::status_t::success_k &&
           nk::cast<nk::bf16_t, nk::f64_t>(std::span<nk_bf16_t const> {staging}, values) == nk::status_t::success_k;
}

/** Widen the BFloat16 bit patterns of a device download in @p staging into @p values . */
[[nodiscard]] inline bool widen_from_bf16(std::span<nk_bf16_t const> staging, std::span<f64_t> values) noexcept {
    return nk::cast<nk::bf16_t, nk::f64_t>(staging, values) == nk::status_t::success_k;
}

} // namespace unum::uform::test
