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

#include <utility>         // `std::move`, `std::forward`
#include <format>          // `std::vformat_to`, `std::make_format_args`, `std::formatter`
#include <source_location> // `std::source_location`

#include <cuda.h>      // `CUresult`, `cuGetErrorString`
#include <cuda_bf16.h> // `__nv_bfloat16` behind the `bf16_t` alias
#include <cuda_fp8.h>  // `__nv_fp8_e4m3` behind the `e4m3_t` alias

#include <numkong/tensor.hpp>          // `nk::tensor_span`, `nk::tensor_view` — `differentiable_weights_t`
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

using e4m3_t = __nv_fp8_e4m3;
using bf16_t = __nv_bfloat16;
using f32_t = float;
using f64_t = double;

/**
 *  The rank-tagged borrow vocabulary over the scalar dtypes — @c <dtype><rank>_view_t reads,
 *  @c <dtype><rank>_span_t writes, each a borrow into memory owned elsewhere. The owning
 *  @c *_tensor_arena_t / @c *_tensor_pinned_t counterparts live in @c gpu_allocators.cuh , beside the
 *  allocator they bind. Concept-named views (a token rectangle, a pixel image) stay with their cell.
 */
using b8r1_view_t = nk::tensor_view<b8_t, 1>;
using b8r1_span_t = nk::tensor_span<b8_t, 1>;
using i16r2_view_t = nk::tensor_view<i16_t, 2>;
using i16r2_span_t = nk::tensor_span<i16_t, 2>;
using i32r1_view_t = nk::tensor_view<i32_t, 1>;
using i32r1_span_t = nk::tensor_span<i32_t, 1>;
using u32r1_view_t = nk::tensor_view<u32_t, 1>;
using u32r1_span_t = nk::tensor_span<u32_t, 1>;
using u64r1_view_t = nk::tensor_view<u64_t, 1>;
using u64r1_span_t = nk::tensor_span<u64_t, 1>;
using f32r1_view_t = nk::tensor_view<f32_t, 1>;
using f32r1_span_t = nk::tensor_span<f32_t, 1>;
using f32r2_view_t = nk::tensor_view<f32_t, 2>;
using f32r2_span_t = nk::tensor_span<f32_t, 2>;
using f32r3_view_t = nk::tensor_view<f32_t, 3>;
using f32r3_span_t = nk::tensor_span<f32_t, 3>;
using bf16r1_view_t = nk::tensor_view<bf16_t, 1>;
using bf16r1_span_t = nk::tensor_span<bf16_t, 1>;
using bf16r2_view_t = nk::tensor_view<bf16_t, 2>;
using bf16r2_span_t = nk::tensor_span<bf16_t, 2>;
using e4m3r1_view_t = nk::tensor_view<e4m3_t, 1>;
using e4m3r1_span_t = nk::tensor_span<e4m3_t, 1>;
using e4m3r2_view_t = nk::tensor_view<e4m3_t, 2>;
using e4m3r2_span_t = nk::tensor_span<e4m3_t, 2>;
using e4m3r3_view_t = nk::tensor_view<e4m3_t, 3>;
using e4m3r3_span_t = nk::tensor_span<e4m3_t, 3>;

/** @brief One vocabulary entry's index — the byte-level vocabulary (256 bytes + the special tokens)
 *      fits 16 bits, and every token tape, staging cell, and embedding lookup trades in this width. */
using token_id_t = u16_t;

/** @brief One raw image / video channel sample, as decoders emit it — 8-bit, interleaved channels;
 *      normalization is the entry projection's job, inside the differentiated step. */
using pixel_t = u8_t;

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

/**
 *  @brief The project's one keyed hasher — a splitmix64 core, used three ways and nothing else, so no
 *      free-standing @c mix / @c combine / @c derive helpers proliferate. Pure and device-callable;
 *      stateless by design (no engine object to step, unlike @c std::minstd or a @c curandState ), so a
 *      thread hashes @c (key, its own index) independently and any result is recomputable from its inputs
 *      on any host or GPU and across restarts. Every keyed draw — dataset shuffling, batch mixing, span
 *      masking, dropout — builds on this; @ref epoch_permutation is the one primitive it cannot be (a
 *      hash collides, a permutation does not).
 *
 *  - @c hash(word) avalanches one word (the finalizer).
 *  - @c hash(key, value) folds @p value into @p key (one mixing step).
 *  - @c hash(a, b, c, ...) folds every argument left to right — the seeder over @c (seed, epoch, rank,
 *    index, salt, ...) , and the indexed draw @c hash(key, index) is just the two-argument case.
 */
struct splitmix64_hash_t {
    /** @brief Avalanche one 64-bit word — the splitmix64 finalizer. */
    [[nodiscard]] constexpr u64_t operator()(u64_t word) const noexcept {
        word = (word ^ (word >> 30)) * 0xBF58476D1CE4E5B9ull;
        word = (word ^ (word >> 27)) * 0x94D049BB133111EBull;
        return word ^ (word >> 31);
    }
    /** @brief Fold @p value into @p key — one order-dependent mixing step (golden-ratio increment). */
    [[nodiscard]] constexpr u64_t operator()(u64_t key, u64_t value) const noexcept {
        return (*this)(key + 0x9E3779B97F4A7C15ull + value);
    }
    /** @brief Fold three or more values left to right by recursing on the two-argument step. */
    template <typename... rest_>
        requires(sizeof...(rest_) >= 1)
    [[nodiscard]] constexpr u64_t operator()(u64_t key, u64_t value, rest_... rest) const noexcept {
        return (*this)((*this)(key, value), static_cast<u64_t>(rest)...);
    }
};

/** @brief The keyed bijection over @c [0,count) : a 4-round Feistel — its round the one
 *      @ref splitmix64_hash_t — over the smallest even-bit domain covering @p count , cycle-walked back
 *      into range. Identical on every host and GPU for the same @c (seed,epoch) , so no shuffle ever
 *      needs an exchange, and @b without replacement (a bijection) — the guarantee the hasher's draws,
 *      which collide, cannot give. */
[[nodiscard]] constexpr std::size_t epoch_permutation(u64_t seed, epoch_t epoch, std::size_t index,
                                                      std::size_t count) noexcept {
    splitmix64_hash_t const hash {};
    u64_t const key = hash(seed, epoch.index);
    u32_t const half_bits = (ceil_log2(static_cast<u32_t>(count)) + 1u) / 2u;
    u32_t const half_mask = (1u << half_bits) - 1u;
    std::size_t value = index;
    do {
        u32_t left = static_cast<u32_t>(value >> half_bits) & half_mask;
        u32_t right = static_cast<u32_t>(value) & half_mask;
        for (u32_t round = 0; round != 4; ++round) {
            u32_t const mixed = static_cast<u32_t>(hash(key, round, right)) & half_mask;
            u32_t const swapped = right;
            right = left ^ mixed;
            left = swapped;
        }
        value = (static_cast<std::size_t>(left) << half_bits) | right;
    } while (value >= count);
    return value;
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
    union {
        value_type_ value_; // Live only on the success path (status_ == success_k); managed by hand below.
    };
    status_t status_ = status_t::success_k;
    CUresult cu_error_ = CUDA_SUCCESS;
    sz::string description_ {};
    std::source_location location_ {};

    expected() noexcept = delete;
    expected(expected const &) noexcept = delete;
    expected &operator=(expected const &) noexcept = delete;

    /** @brief Success: adopts @p value , making the union member live. */
    expected(value_type_ value) noexcept : value_(std::move(value)) {}

    /** @brief Move: the error channel travels unconditionally; the value is adopted only when @p other is
     *      on the success path, since the union member is live only then. The @c noexcept is the same
     *      compile-time proof that @c value_type_ moves without throwing. */
    expected(expected &&other) noexcept
        : status_(other.status_), cu_error_(other.cu_error_), description_(std::move(other.description_)),
          location_(other.location_) {
        if (status_ == status_t::success_k) ::new (static_cast<void *>(&value_)) value_type_(std::move(other.value_));
    }
    expected &operator=(expected &&other) noexcept {
        if (this != &other) {
            if (status_ == status_t::success_k) value_.~value_type_();
            status_ = other.status_;
            cu_error_ = other.cu_error_;
            description_ = std::move(other.description_);
            location_ = other.location_;
            if (status_ == status_t::success_k)
                ::new (static_cast<void *>(&value_)) value_type_(std::move(other.value_));
        }
        return *this;
    }
    ~expected() noexcept {
        if (status_ == status_t::success_k) value_.~value_type_();
    }

    /** @brief @c true iff the operation succeeded and @ref value_ is readable. */
    [[nodiscard]] explicit operator bool() const noexcept { return status_ == status_t::success_k; }

    /** @brief Reach the held value on the success path — the @c std::optional / @c std::expected idiom,
     *      so a consumer writes @c result->field or @c *result instead of @c result.value_ . Reading it
     *      on a failed @ref expected is a logic error, gated by @ref operator bool . */
    [[nodiscard]] value_type_ *operator->() noexcept { return &value_; }
    [[nodiscard]] value_type_ const *operator->() const noexcept { return &value_; }
    [[nodiscard]] value_type_ &operator*() noexcept { return value_; }
    [[nodiscard]] value_type_ const &operator*() const noexcept { return value_; }

    /** @brief Failure tagged with a @ref status_t : materializes @p format + @p args into the description
     *      and records the failure site @ref format_at_t captured. */
    template <typename... arg_types_>
    [[nodiscard]] static expected fail(status_t status, format_at_t format, arg_types_ &&...args) noexcept {
        expected result {failure_t {}, status};
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
        expected result {failure_t {}, other.status_};
        result.cu_error_ = other.cu_error_;
        result.description_ = std::move(other.description_);
        result.location_ = other.location_;
        return result;
    }

  private:
    struct failure_t {};
    /** @brief Failure ctor: sets a non-success @p status and leaves the union @ref value_ unconstructed,
     *      so the @c status_==success_k ⇔ value_ live invariant the destructor relies on holds at once. */
    expected(failure_t, status_t status) noexcept : status_(status) {}
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
