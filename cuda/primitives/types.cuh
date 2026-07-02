/**
 *  @brief Foundational vocabulary of the UForm v4 primitives cell: statuses, error channel, launch constants.
 *  @file cuda/primitives/types.cuh
 *  @author Ash Vardanian
 *  @date July 2, 2026
 *
 *  The primitives cell holds the self-contained CUDA building blocks the rest of the v4 stack assembles —
 *  losses, collectives, packers — each a header-only @c .cuh with its own kernels and @c launch_* wrappers,
 *  parameterized by NumKong spans and a caller-owned stream + workspace, with no vendor-library sessions, no
 *  hidden state, and no allocation on any launch path.
 */
#pragma once

#include <cstddef> // `std::ptrdiff_t`
#include <cstdint> // `std::int32_t`, `std::uint32_t`

#include <format>          // `std::vformat_to`, `std::make_format_args`, `std::formatter`
#include <source_location> // `std::source_location`
#include <utility>         // `std::move`, `std::forward`

#include <cuda.h> // `CUresult`, `cuGetErrorString`

#include <stringzilla/stringzilla.hpp> // `sz::string` — the exception-free diagnostics carrier

/** @brief `std::format` support for StringZilla slices, so diagnostics interpolate them directly —
 *      no conversion shims at call sites. */
template <>
struct std::formatter<ashvardanian::stringzilla::string_view> : std::formatter<std::string_view> {
    auto format(ashvardanian::stringzilla::string_view text, std::format_context &context) const {
        return std::formatter<std::string_view>::format(std::string_view {text.data(), text.size()}, context);
    }
};

/** @brief `std::format` support for owning StringZilla strings — same bridge, by reference. */
template <>
struct std::formatter<ashvardanian::stringzilla::string> : std::formatter<std::string_view> {
    auto format(ashvardanian::stringzilla::string const &text, std::format_context &context) const {
        return std::formatter<std::string_view>::format(std::string_view {text.data(), text.size()}, context);
    }
};

namespace unum::uform::primitives {

namespace sz = ashvardanian::stringzilla;

/** @brief Thread count of one CUDA warp — the granularity of every shuffle-based reduction tree. */
inline constexpr unsigned warp_size_k = 32u;

/** @brief Default thread count per CTA for hand-rolled kernels. Fits the CUB @c BlockReduce<float,256>
 *      temp-storage budget (1 KB SMEM) with full occupancy on sm_90 and sm_100. */
inline constexpr unsigned kernel_block_size_default_k = 256u;

/** @brief Outcome of any fallible operation. @c success_k==0 ; all error codes are negative + stable. */
enum class status_t : std::int32_t {
    /** @brief The operation completed; outputs are valid. */
    success_k = 0,

    /** @brief A caller-provided buffer was too small or a device allocation failed. */
    bad_alloc_k = -10,
    /** @brief A caller argument was null, empty, or otherwise invalid. */
    invalid_input_k = -11,
    /** @brief The requested geometry / dtype / extents are unsupported by the dispatched kernel. */
    shape_unsupported_k = -12,
    /** @brief A CUDA Runtime call failed. */
    cuda_error_k = -13,
    /** @brief A host I/O / file operation failed (read / write / rename / timeout on a peer's file). */
    io_error_k = -19,

    /** @brief An unclassified failure; prefer a specific code above when one fits. */
    other_k = -1,
};

/** @brief Static-string name for a @ref status_t value (the single name table; no allocation). */
[[nodiscard]] inline char const *to_string(status_t status) noexcept {
    switch (status) {
    case status_t::success_k: return "success";
    case status_t::bad_alloc_k: return "bad_alloc";
    case status_t::invalid_input_k: return "invalid_input";
    case status_t::shape_unsupported_k: return "shape_unsupported";
    case status_t::cuda_error_k: return "cuda_error";
    case status_t::io_error_k: return "io_error";
    case status_t::other_k: return "other";
    }
    return "unknown";
}

/** @brief An output iterator @c std::vformat_to writes through, appending straight into an exception-free
 *      StringZilla string. An allocation failure latches into the shared @c failed flag (when bound) and
 *      truncates the text — for diagnostics that is acceptable as-is; for payloads, @ref format_into
 *      surfaces the flag. */
struct format_into_t {
    using difference_type = std::ptrdiff_t;

    sz::string *text = nullptr;
    bool *failed = nullptr;

    format_into_t &operator*() noexcept { return *this; }
    format_into_t &operator++() noexcept { return *this; }
    format_into_t operator++(int) noexcept { return *this; }
    format_into_t &operator=(char character) noexcept {
        if (!text->try_push_back(character) && failed != nullptr) *failed = true;
        return *this;
    }
};

/** @brief `std::format` straight into an exception-free StringZilla string, appending — the one spelling
 *      for every payload and path assembly. @c false only on allocation failure (the text is then
 *      truncated); the format string is compile-time checked. */
template <typename... arg_types_>
[[nodiscard]] inline bool format_into(sz::string &text, std::format_string<arg_types_...> format_string,
                                      arg_types_ &&...args) noexcept {
    bool failed = false;
    std::vformat_to(format_into_t {&text, &failed}, format_string.get(), std::make_format_args(args...));
    return !failed;
}

/** @brief A format string bundled with the @c std::source_location of the call site that spelled it. The
 *      implicit conversion from a string literal lets @ref expected::fail capture the failure site behind
 *      its variadic pack, where a defaulted trailing argument cannot go — and without a macro. */
struct format_at_t {
    char const *format = nullptr;
    std::source_location location {};

    format_at_t(char const *format, std::source_location location = std::source_location::current()) noexcept
        : format(format), location(location) {}
};

/**
 *  @brief Minimal host-side value-or-error carrier — the return type of every fallible host function in the
 *      primitives cell.
 *
 *  Move-only: the explicit @c noexcept on the defaulted move members is a compile-time check that every
 *  instantiating @c value_type_ moves without throwing, so a result propagates up the call chain as nothing
 *  but @c noexcept moves. The description is an exception-free StringZilla string, materialized with
 *  @c std::format at the failure site — the only place the operands (shapes, counts, labels) are still in
 *  scope — and @ref format_at_t records that site's @c std::source_location .
 */
template <typename value_type_>
struct expected {
    value_type_ value_ {};
    status_t status_ = status_t::success_k;
    CUresult cu_error_ = CUDA_SUCCESS;
    sz::string description_ {};
    std::source_location location_ {};

    expected() noexcept = delete;
    expected(expected const &) noexcept = delete;
    expected &operator=(expected const &) noexcept = delete;

    expected(expected &&) noexcept = default;
    expected &operator=(expected &&) noexcept = default;

    /** @brief Success: adopts @p value . */
    expected(value_type_ value) noexcept : value_(std::move(value)) {}

    /** @brief @c true iff the operation succeeded and @ref value_ is readable. */
    [[nodiscard]] explicit operator bool() const noexcept { return status_ == status_t::success_k; }

    /** @brief Failure tagged with a @ref status_t : materializes @p format + @p args into the description
     *      and records the failure site @ref format_at_t captured. */
    template <typename... arg_types_>
    [[nodiscard]] static expected fail(status_t status, format_at_t format, arg_types_ &&...args) noexcept {
        expected result {value_type_ {}};
        result.status_ = status;
        result.location_ = format.location;
        std::vformat_to(format_into_t {&result.description_}, format.format, std::make_format_args(args...));
        return result;
    }

    /** @brief Failure carrying a CUDA Driver error: same as the @ref status_t overload, tagged
     *      @c cuda_error_k , with the driver's own error string appended to the description. */
    template <typename... arg_types_>
    [[nodiscard]] static expected fail(CUresult cu_error, format_at_t format, arg_types_ &&...args) noexcept {
        expected result = fail(status_t::cuda_error_k, format, std::forward<arg_types_>(args)...);
        result.cu_error_ = cu_error;
        char const *cu_error_string = nullptr;
        if (cuGetErrorString(cu_error, &cu_error_string) != CUDA_SUCCESS || cu_error_string == nullptr)
            cu_error_string = "unknown";
        (void)result.description_.try_append(": ");
        (void)result.description_.try_append(cu_error_string);
        return result;
    }

    /** @brief Failure adopted from another @ref expected 's error channel: the status, driver error,
     *      description, and failure site travel @b unchanged — the composition path for helpers, so
     *      forwarding never reformats and never loses the original location. */
    template <typename other_type_>
    [[nodiscard]] static expected fail(expected<other_type_> &&other) noexcept {
        expected result {value_type_ {}};
        result.status_ = other.status_;
        result.cu_error_ = other.cu_error_;
        result.description_ = std::move(other.description_);
        result.location_ = other.location_;
        return result;
    }
};

/**
 *  @brief The value-less @ref expected — the return type of fallible actions that produce no result
 *      (publish, await, barrier), carrying the same status / driver-error / description / location
 *      channel. Default-constructed means success: @c return @c {} .
 */
template <>
struct expected<void> {
    status_t status_ = status_t::success_k;
    CUresult cu_error_ = CUDA_SUCCESS;
    sz::string description_ {};
    std::source_location location_ {};

    expected() noexcept = default;
    expected(expected const &) noexcept = delete;
    expected &operator=(expected const &) noexcept = delete;

    expected(expected &&) noexcept = default;
    expected &operator=(expected &&) noexcept = default;

    /** @brief @c true iff the action succeeded. */
    [[nodiscard]] explicit operator bool() const noexcept { return status_ == status_t::success_k; }

    /** @brief Failure tagged with a @ref status_t : materializes @p format + @p args into the description
     *      and records the failure site @ref format_at_t captured. */
    template <typename... arg_types_>
    [[nodiscard]] static expected fail(status_t status, format_at_t format, arg_types_ &&...args) noexcept {
        expected result;
        result.status_ = status;
        result.location_ = format.location;
        std::vformat_to(format_into_t {&result.description_}, format.format, std::make_format_args(args...));
        return result;
    }

    /** @brief Failure carrying a CUDA Driver error: same as the @ref status_t overload, tagged
     *      @c cuda_error_k , with the driver's own error string appended to the description. */
    template <typename... arg_types_>
    [[nodiscard]] static expected fail(CUresult cu_error, format_at_t format, arg_types_ &&...args) noexcept {
        expected result = fail(status_t::cuda_error_k, format, std::forward<arg_types_>(args)...);
        result.cu_error_ = cu_error;
        char const *cu_error_string = nullptr;
        if (cuGetErrorString(cu_error, &cu_error_string) != CUDA_SUCCESS || cu_error_string == nullptr)
            cu_error_string = "unknown";
        (void)result.description_.try_append(": ");
        (void)result.description_.try_append(cu_error_string);
        return result;
    }

    /** @brief Failure adopted from another @ref expected 's error channel, unchanged. */
    template <typename other_type_>
    [[nodiscard]] static expected fail(expected<other_type_> &&other) noexcept {
        expected result;
        result.status_ = other.status_;
        result.cu_error_ = other.cu_error_;
        result.description_ = std::move(other.description_);
        result.location_ = other.location_;
        return result;
    }
};

/** @brief The everyday spelling of the value-less @ref expected . */
using expected_t = expected<void>;

} // namespace unum::uform::primitives
