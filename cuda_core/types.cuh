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

#include <cuda.h>      // `CUresult`, `cuGetErrorString`
#include <cuda_bf16.h> // `__nv_bfloat16` behind the `bf16_t` alias
#include <cuda_fp8.h>  // `__nv_fp8_e4m3` behind the `e4m3_t` alias

#include <numkong/tensor.hpp> // `nk::tensor_span`, `nk::tensor_view` — `differentiable_weights_t`

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

namespace unum::uform::core {

namespace sz = ashvardanian::stringzilla;
namespace nk = ashvardanian::numkong;

using b8_t = std::byte;

using u8_t = std::uint8_t;
using u16_t = std::uint16_t;
using u32_t = std::uint32_t;
using u64_t = std::uint64_t;

using i8_t = std::int8_t;
using i16_t = std::int16_t;
using i32_t = std::int32_t;
using i64_t = std::int64_t;

/** @brief One vocabulary entry's index — the byte-level vocabulary (256 bytes + the special tokens)
 *      fits 16 bits, and every token tape, staging slot, and embedding lookup trades in this width. */
using token_id_t = u16_t;

/** @brief One raw image / video channel sample, as decoders emit it — 8-bit, interleaved channels;
 *      normalization is the entry projection's job, inside the differentiated step. */
using pixel_t = u8_t;

using e4m3_t = __nv_fp8_e4m3;
using bf16_t = __nv_bfloat16;
using f32_t = float;
using f64_t = double;

/** @brief Thread count of one CUDA warp — the granularity of every shuffle-based reduction tree. */
inline constexpr unsigned warp_size_k = 32u;

/** @brief Default thread count per CTA for hand-rolled kernels. Fits the CUB @c BlockReduce<f32_t,256>
 *      temp-storage budget (1 KB SMEM) with full occupancy on sm_90 and sm_100. */
inline constexpr unsigned kernel_block_size_default_k = 256u;

/** @brief The typical upper bound on the number of GPUs attached to the same host process
 *      within the same "multimem" domain.
 *  @sa Defines the upper bound for @ref mesh_shape_t::gpus_per_host_domain. */
inline constexpr unsigned max_gpus_per_host_k = 32u;

/** @brief One pass over the dataset — a strong type, so the keyed draws (shuffling, mixing, masking)
 *      can never confuse it with any other counter in an argument list. */
struct epoch_t {
    u32_t index = 0;
};

/** @brief @c ceil(numerator/denominator) over non-negative integers — grid sizing, chunk counts. */
[[nodiscard]] inline constexpr std::size_t divide_round_up(std::size_t numerator, std::size_t denominator) noexcept {
    return (numerator + denominator - 1u) / denominator;
}

/** @brief @c ceil(log2(value)) for @c value>=1 — the recursive-doubling round count over @c value
 *      domains. */
[[nodiscard]] inline constexpr u32_t ceil_log2(u32_t value) noexcept {
    u32_t rounds = 0;
    while ((1u << rounds) < value) ++rounds;
    return rounds;
}

/** @brief Outcome of any fallible operation. @c success_k==0 ; all error codes are negative + stable. */
enum class status_t : i32_t {
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

/** @brief One weight matrix with its gradient — what models publish and optimizers consume. Vector
 *      parameters (norm scales, biases) are @b [1,cols] . The optimizer reads the gradient and writes
 *      the weights, hence the mutability split. */
struct differentiable_weights_t {
    nk::tensor_span<bf16_t, 2> weights {};
    nk::tensor_view<bf16_t, 2> weights_grad {};
};

} // namespace unum::uform::core
