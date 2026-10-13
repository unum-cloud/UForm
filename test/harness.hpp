/**
 *  @brief The one shared testing primitive every `test/` binary threads through its cases — the
 *      cumulative pass/fail tally, reusing the library's own diagnostic channel.
 *  @file test/harness.hpp
 *  @author Ash Vardanian
 *  @date July 5, 2026
 *
 *  A test is a plain `int main` that constructs one @ref test_suite_t and passes it @b by reference into
 *  every case; there is no file-scope counter and no singleton. The single assertion primitive @ref
 *  test_suite_t::check mirrors @ref expected::fail exactly — a format string plus its operands, with the
 *  call site's @c std::source_location captured through @ref format_at_t — so a failing check names its
 *  file, line, and the @b actual-versus-expected values, and a passing check is @b silent. A whole green
 *  run prints one summary line; the only per-check output is a failure that explains itself.
 */
#pragma once

#include <cstdlib> // `std::getenv` — the `UFORM_TEST_FILTER` launcher contract

#include <unistd.h> // `::write`, `STDERR_FILENO`

#include <format> // `std::vformat_to`, `std::make_format_args`
#include <regex>  // `std::regex`, `std::regex_search` — the case-name filter

#include "core/types.hpp" // `expected`, `format_into`, `format_at_t`, `format_into_t`, `sz`, the scalar aliases

namespace unum::uform::test {

using namespace unum::uform::core;

/** The file name of a `__FILE__`-style path — the trailing component after the last `/`. */
[[nodiscard]] inline char const *path_basename(char const *path) noexcept {
    char const *base = path;
    for (char const *character = path; *character != '\0'; ++character)
        if (*character == '/') base = character + 1;
    return base;
}

/** Write one already-assembled line to standard error — the host-diagnostic channel (§3), never
 *  `printf`. */
inline void emit_line(sz::string_view_t line) noexcept {
    ssize_t const written = ::write(STDERR_FILENO, line.data(), line.size());
    (void)written;
}

/**
 *  @brief The launcher's case filter: `UFORM_TEST_FILTER` holds a regular expression matched against each
 *      case's formatted name — the NumKong / StringZilla convention.
 *
 *  A targeted question about one case answers in minutes instead of re-running the whole matrix.
 *  An unset variable accepts every case; a malformed pattern warns once and also accepts every case,
 *  because a broken filter must never silently skip the suite.
 */
[[nodiscard]] inline bool test_filter_accepts(sz::string_view_t case_name) noexcept {
    static char const *const pattern = std::getenv("UFORM_TEST_FILTER"); // vendor-boundary: launcher contract
    if (pattern == nullptr) return true;
    struct parsed_filter_t {
        std::regex expression {};
        bool valid = false;
    };
    static parsed_filter_t const filter = []() noexcept {
        parsed_filter_t parsed;
        try {
            parsed.expression = std::regex {pattern, std::regex_constants::ECMAScript | std::regex_constants::nosubs};
            parsed.valid = true;
        }
        catch (std::regex_error const &) {
            sz::string_t line;
            (void)format_into(line, "UFORM_TEST_FILTER is not a valid regular expression: {}\n", pattern);
            emit_line(line.view());
        }
        return parsed;
    }();
    if (!filter.valid) return true; // a broken filter must never silently skip the suite
    return std::regex_search(case_name.data(), case_name.data() + case_name.size(), filter.expression);
}

/**
 *  @brief The cumulative outcome of a test binary: a running pass/fail tally, constructed in `main` and
 *      threaded by reference into every case.
 *
 *  Silent on success — a passing check only advances the count;
 *  a failing one prints @c file:line and the formatted reason, then the process exits non-zero.
 */
struct test_suite_t {

    /** The binary's name, printed in the summary line. */
    sz::string_view_t name {};

    /** Checks that passed. */
    u32_t passed = 0;

    /** Checks that failed; the process exits non-zero when it is not zero. */
    u32_t failed = 0;

    /**
     *  @brief Assert @p condition ; on failure, format @p message + @p args the way @ref expected::fail
     *      does and print it under the call site's @c file:line . A pass is silent — only the tally moves.
     *
     *  The @ref format_at_t parameter captures the caller's @c std::source_location behind the variadic
     *  pack, so a check reads `check(got == want, "peer device_name: got {} want {}", got, want)` and a
     *  failure explains itself with the real operands, no separate label.
     */
    template <typename... arg_types_>
    void check(bool condition, format_at_t message, arg_types_ &&...args) noexcept {
        if (condition) {
            ++passed;
            return;
        }
        ++failed;
        sz::string_t line;
        (void)format_into(line, "  FAIL {}:{}  ", path_basename(message.location.file_name()),
                          static_cast<u32_t>(message.location.line()));
        std::vformat_to(format_into_t {&line}, message.format, std::make_format_args(args...));
        (void)line.try_push_back('\n');
        emit_line(line.view());
    }

    /** The one summary line and the process exit code. */
    [[nodiscard]] int finish() noexcept {
        sz::string_t line;
        (void)format_into(line, "{}: {} ({}/{} checks passed)\n", failed == 0 ? "PASS" : "FAIL", name, passed,
                          passed + failed);
        emit_line(line.view());
        return failed == 0 ? 0 : 1;
    }
};

} // namespace unum::uform::test
