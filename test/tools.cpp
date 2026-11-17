/**
 *  @file test/tools.cpp
 *  @author Ash Vardanian
 *  @date November 16, 2026
 *  @brief Drives one UForm binary, which the arguments after <tt>--</tt> run, through the flag and
 *      signal conventions every binary shares.
 *
 *  Each run checks that @c --help exits with 0, that a value that does not parse prints one line
 *  naming the flag and exits with 1, and that the flag's @c UFORM_ variable is read the same way.
 *  With @c UFORM_TEST_CHECKPOINT naming a checkpoint, @c uform_eval also calibrates on generated
 *  choices, takes a SIGINT once it prints progress, and must print @c stopped @c in, exit with 130
 *  and write no output.
 */
#undef NDEBUG // ! A test's oracle must stay live in every build

#include <cstdio>  // `std::fopen`, `std::remove`
#include <cstdlib> // `std::getenv`, `::setenv`, `::unsetenv`
#include <cstring> // `std::strrchr`, `std::strncpy`

#include <poll.h>     // `poll`
#include <signal.h>   // `kill`, `SIGINT`
#include <sys/wait.h> // `waitpid`, `WIFEXITED`, `WEXITSTATUS`
#include <unistd.h>   // `fork`, `execv`, `pipe`, `dup2`

#include <initializer_list> // `std::initializer_list`
#include <string_view>      // `std::string_view`
#include <utility>          // `std::forward`

#include <fmt/format.h>

/** What one run of the binary printed to both streams, and how it exited. */
struct run_t {
    int exit_code = -1;
    char output[1 << 16] {};
    std::size_t length = 0;

    [[nodiscard]] std::string_view text() const noexcept { return {output, length}; }
    [[nodiscard]] bool contains(std::string_view part) const noexcept {
        return text().find(part) != std::string_view::npos;
    }
};

/** The words before the binary's own arguments: an emulator and its options in a cross build. */
static char** command_words = nullptr;
static int command_words_count = 0;

/** How long a run may stay silent, the one wait limit every UForm suite shares. */
static constexpr int wait_limit_milliseconds_k = 120 * 1000;

/** Runs the binary with @p arguments, both streams into @p run ; once @p interrupt_after appears in
 *  them, sends one SIGINT. */
static void run_(run_t& run, std::initializer_list<char const*> arguments, std::string_view interrupt_after = {}) {
    // `execv` takes writable words, so each argument is copied into storage the child may own
    static char words[64][512];
    char* argv[64] {};
    int argc = 0;
    auto const add = [&](char const* word) {
        argv[argc] = std::strncpy(words[argc], word, sizeof(words[0]) - 1), ++argc;
    };
    for (int word = 0; word != command_words_count; ++word) add(command_words[word]);
    for (char const* argument : arguments) add(argument);
    int channel[2];
    if (::pipe(channel) != 0) return;
    pid_t const child = ::fork();
    if (child == 0) {
        ::dup2(channel[1], STDOUT_FILENO), ::dup2(channel[1], STDERR_FILENO);
        ::close(channel[0]), ::close(channel[1]);
        ::execv(argv[0], argv);
        ::_exit(127);
    }
    ::close(channel[1]);
    run.length = 0;
    bool interrupted = interrupt_after.empty();
    pollfd waiting {channel[0], POLLIN, 0};
    while (::poll(&waiting, 1, wait_limit_milliseconds_k) == 1) {
        ssize_t const read = ::read(channel[0], run.output + run.length, sizeof(run.output) - 1 - run.length);
        if (read <= 0) break;
        run.length += static_cast<std::size_t>(read);
        if (!interrupted && run.contains(interrupt_after)) ::kill(child, SIGINT), interrupted = true;
    }
    ::close(channel[0]);
    int status = 0;
    ::waitpid(child, &status, 0);
    run.exit_code = WIFEXITED(status) ? WEXITSTATUS(status) : -1;
}

/** A flag of the binary, a value it refuses, the grammar its error line names, and its variable. */
struct case_t {
    std::string_view binary;
    char const* positional;
    char const* flag;
    char const* refused;
    std::string_view grammar;
    char const* variable;
};

static constexpr case_t cases_k[] = {
    {"uform_eval", "model.safetensors", "--batch", "0", "a positive count", "UFORM_BATCH"},
    {"uform_grow", "in.safetensors", "--deepen=1", "1", "no value", "UFORM_DEEPEN"},
    {"uform_base", nullptr, "--steps", "many", "an unsigned integer", "UFORM_STEPS"},
};

static std::size_t failures_count = 0;

template <typename... args_types_>
static void expect_(bool is_met, fmt::format_string<args_types_...> format, args_types_&&... args) {
    if (is_met) return;
    ++failures_count;
    fmt::println(stderr, "FAIL: {}", fmt::format(format, std::forward<args_types_>(args)...));
}

int main(int argc, char** argv) {
    for (int index = 1; index < argc; ++index)
        if (std::string_view {argv[index]} == "--") {
            command_words = argv + index + 1, command_words_count = argc - index - 1;
            break;
        }
    if (command_words_count == 0) {
        fmt::println(stderr, "Usage: {} -- <command that runs a UForm binary>", argv[0]);
        return 1;
    }
    char const* const slash = std::strrchr(command_words[command_words_count - 1], '/');
    std::string_view const binary = slash ? slash + 1 : command_words[command_words_count - 1];
    case_t const* known = nullptr;
    for (case_t const& candidate : cases_k)
        if (binary == candidate.binary) known = &candidate;
    if (!known) return fmt::println(stderr, "no cases for {}", binary), 1;
    static run_t run;

    run_(run, {"--help"});
    expect_(run.exit_code == 0 && run.contains("Usage: ") && run.contains("[env: UFORM_"), "--help exited {} with {:?}",
            run.exit_code, run.text());

    bool const toggle = std::string_view {known->flag}.ends_with("=1");
    if (known->positional && toggle) run_(run, {known->positional, known->flag});
    else if (known->positional) run_(run, {known->positional, known->flag, known->refused});
    else run_(run, {known->flag, known->refused});
    std::string_view const name = std::string_view {known->flag}.substr(0, std::string_view {known->flag}.find('='));
    char line[256];
    std::size_t const line_length = fmt::format_to_n(line, sizeof(line), "{}=\"{}\" does not parse, expected {}\n",
                                                     name, known->refused, known->grammar)
                                        .size;
    expect_(run.exit_code == 1 && run.text() == std::string_view {line, line_length}, "{} {} exited {} with {:?}",
            known->flag, known->refused, run.exit_code, run.text());

    // A switch's variable takes 1, 0, true or false, so any other word fails like a bad value.
    ::setenv(known->variable, toggle ? "maybe" : known->refused, 1);
    run_(run, {"--help"});
    ::unsetenv(known->variable);
    expect_(run.exit_code == 1 && run.text().starts_with(known->variable) && run.contains(" does not parse, expected "),
            "{} exited {} with {:?}", known->variable, run.exit_code, run.text());

    char const* const checkpoint = std::getenv("UFORM_TEST_CHECKPOINT");
    if (binary == "uform_eval" && checkpoint && *checkpoint) {
        char const* const choices_path = "uform_test_tools.choices.jsonl";
        char const* const output_path = "uform_test_tools.calibrated.safetensors";
        std::FILE* choices = std::fopen(choices_path, "wb");
        for (std::size_t item = 0; choices && item != 20000; ++item)
            fmt::println(choices,
                         R"({{"state": "caption {} of a long held-out list", "choices": "abcd", )" //
                         R"("targets": [0.7, 0.1, 0.1, 0.1], "group": {}}})",
                         item, item % 3);
        if (choices) std::fclose(choices);
        std::remove(output_path);
        run_(run, {checkpoint, "--calibrate", choices_path, "--output", output_path, "--batch", "16"}, "calibrate: ");
        std::FILE* written = std::fopen(output_path, "rb");
        expect_(run.exit_code == 130 && run.contains("stopped in calibrate\n") && !written,
                "SIGINT during calibration exited {} with {:?}", run.exit_code, run.text());
        if (written) std::fclose(written), std::remove(output_path);
        std::remove(choices_path);
    }
    fmt::println("{}: {} failures", binary, failures_count);
    return failures_count ? 1 : 0;
}
