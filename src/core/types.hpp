/**
 *  @brief Foundational vocabulary of the UForm v4 primitives cell: statuses, error channel, launch constants.
 *  @file src/core/types.hpp
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

#include <utility>         // `std::move`
#include <new>             // `std::launder` — typed spans over byte regions
#include <type_traits>     // `std::is_implicit_lifetime_v`
#include <format>          // `std::vformat_to`, `std::make_format_args`, `std::formatter`
#include <source_location> // `std::source_location`

#include <numkong/tensor.hpp>          // `nk::tensor_span`, `nk::tensor_view` — the rank-tagged borrows
#include <stringzilla/stringzilla.hpp> // `sz::string_t` — the exception-free diagnostics carrier

/** `std::format` support for StringZilla slices, so diagnostics interpolate them directly —
 *  no conversion shims at call sites. */
template <>
struct std::formatter<ashvardanian::stringzilla::string_view_t> : std::formatter<std::string_view> {
    auto format(ashvardanian::stringzilla::string_view_t text, std::format_context &context) const {
        return std::formatter<std::string_view>::format(std::string_view {text.data(), text.size()}, context);
    }
};

/** `std::format` support for owning StringZilla strings — same bridge, by reference. */
template <>
struct std::formatter<ashvardanian::stringzilla::string_t> : std::formatter<std::string_view> {
    auto format(ashvardanian::stringzilla::string_t const &text, std::format_context &context) const {
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

using f32_t = float;
using f64_t = double;

/** The rank-tagged borrow vocabulary over the scalar dtypes — @c <dtype><rank>_view_t reads,
 *      @c <dtype><rank>_span_t writes, each a borrow into memory owned elsewhere. The owning
 *      @c *_tensor_arena_t / @c *_tensor_pinned_t counterparts live in @c gpu_allocators.hpp , beside the
 *      allocator they bind, as do the CUDA-scalar @c bf16_t / @c e4m3_t borrows. Concept-named views (a token
 *      rectangle, a pixel image) stay with their cell. */
using b8r1_view_t = nk::tensor_view<b8_t, 1>;
using b8r1_span_t = nk::tensor_span<b8_t, 1>;
using i16r2_view_t = nk::tensor_view<i16_t, 2>;
using i16r2_span_t = nk::tensor_span<i16_t, 2>;
using i32r1_view_t = nk::tensor_view<i32_t, 1>;
using i32r1_span_t = nk::tensor_span<i32_t, 1>;
using u32r1_view_t = nk::tensor_view<u32_t, 1>;
using u32r1_span_t = nk::tensor_span<u32_t, 1>;
using u32r2_view_t = nk::tensor_view<u32_t, 2>;
using u32r2_span_t = nk::tensor_span<u32_t, 2>;
using u64r1_view_t = nk::tensor_view<u64_t, 1>;
using u64r1_span_t = nk::tensor_span<u64_t, 1>;
using u64r2_view_t = nk::tensor_view<u64_t, 2>;
using u64r2_span_t = nk::tensor_span<u64_t, 2>;
using f32r1_view_t = nk::tensor_view<f32_t, 1>;
using f32r1_span_t = nk::tensor_span<f32_t, 1>;
using f32r2_view_t = nk::tensor_view<f32_t, 2>;
using f32r2_span_t = nk::tensor_span<f32_t, 2>;
using f32r3_view_t = nk::tensor_view<f32_t, 3>;
using f32r3_span_t = nk::tensor_span<f32_t, 3>;

/** The whole @p value_type_ scalars of @p bytes , a trailing partial scalar dropped — the one place a byte region
 *  takes on a scalar type. The region comes from an allocation, which creates implicit-lifetime scalars on first
 *  access, so laundering the address is all it takes, and it touches no memory, so device regions qualify. */
template <typename value_type_>
[[nodiscard]] inline nk::tensor_span<value_type_, 1> typed_span(b8r1_span_t bytes) noexcept {
    static_assert(std::is_implicit_lifetime_v<value_type_>);
    std::size_t const count = bytes.extent(0) / sizeof(value_type_);
    return {std::launder(static_cast<value_type_ *>(static_cast<void *>(bytes.data()))), count};
}

/** The read-only twin of @ref typed_span . */
template <typename value_type_>
[[nodiscard]] inline nk::tensor_view<value_type_, 1> typed_view(b8r1_view_t bytes) noexcept {
    static_assert(std::is_implicit_lifetime_v<value_type_>);
    std::size_t const count = bytes.extent(0) / sizeof(value_type_);
    return {std::launder(static_cast<value_type_ const *>(static_cast<void const *>(bytes.data()))), count};
}

/** One vocabulary entry's index — the byte-level vocabulary (256 bytes + the special tokens)
 *  fits 16 bits, and every token tape, staging cell, and embedding lookup trades in this width. */
using token_id_t = u16_t;

/** One raw image / video channel sample, as decoders emit it — 8-bit, interleaved channels;
 *  normalization is the entry projection's job, inside the differentiated step. */
using pixel_t = u8_t;

/** Thread count of one CUDA warp — the granularity of every shuffle-based reduction tree. */
inline constexpr unsigned warp_size_k = 32u;

/** Default thread count per CTA for hand-rolled kernels. Fits the CUB @c BlockReduce<f32_t,256>
 *  temp-storage budget (1 KB SMEM) with full occupancy on sm_90 and sm_100. */
inline constexpr unsigned kernel_block_size_default_k = 256u;

/**
 *  @brief The typical upper bound on the number of GPUs attached to the same host process
 *      within the same "multimem" domain.
 *  @sa Defines the upper bound for @ref mesh_shape_t::gpus_per_host_domain.
 */
inline constexpr unsigned max_gpus_per_host_k = 32u;

/** One pass over the data — the sole key of every deterministic draw (shuffling, mixing, masking,
 *  synthetic contents). Distinct from @ref optimizer_step_t : a resumed run at the same data epoch
 *  reproduces the same draws regardless of how many optimizer steps preceded it. */
struct data_epoch_t {
    u32_t ordinal = 0;
};

/** One optimizer iteration — the learning-rate schedule's clock and the report / checkpoint
 *  cadence. Never a draw key. Synthetic corpora key the pass on the iteration; that crossing is
 *  spelled @c data_epoch_t {optimizer_step.ordinal} exactly where it happens. */
struct optimizer_step_t {
    u32_t ordinal = 0;
};

/** One observed round of a device-side collective barrier — @ref barrier_round_t 's self-advanced
 *  scalar, read back for telemetry and conformance. Never a caller input: the device bumps it, the
 *  host only observes. */
struct concurrency_round_t {
    u32_t ordinal = 0;
};

/** @c ceil(numerator/denominator) over non-negative integers — grid sizing, chunk counts. */
[[nodiscard]] inline constexpr std::size_t divide_round_up(std::size_t numerator, std::size_t denominator) noexcept {
    return (numerator + denominator - 1u) / denominator;
}

/** @c ceil(log2(value)) for @c value>=1 — the recursive-doubling round count over @c value
 *  domains. */
[[nodiscard]] inline constexpr u32_t ceil_log2(u32_t value) noexcept {
    u32_t rounds = 0;
    while ((1u << rounds) < value) ++rounds;
    return rounds;
}

/**
 *  @brief The project's one keyed hasher — a splitmix64 core, used three ways and nothing else, so no
 *      free-standing @c mix / @c combine / @c derive helpers proliferate.
 *
 *  Pure and device-callable; stateless by design (no engine object to step, unlike @c std::minstd or a
 *  @c curandState ), so a thread hashes its own @c (key,index) pair independently and any result is
 *  recomputable from its inputs on any host or GPU and across restarts. Every keyed draw — dataset
 *  shuffling, batch mixing, span masking, dropout — builds on this; @ref epoch_permutation is the one
 *  primitive it cannot be (a hash collides, a permutation does not).
 *
 *  - @c hash(word) avalanches one word (the finalizer).
 *  - @c hash(key,value) folds @p value into @p key (one mixing step).
 *  - @c hash(a,b,c,...) folds every argument left to right — the seeder over
 *    @c (seed,data_epoch,member,index,salt,...) , and the indexed draw @c hash(key,index) is just the
 *    two-argument case.
 */
struct splitmix64_hash_t {

    /** Avalanche one 64-bit word — the splitmix64 finalizer. */
    [[nodiscard]] constexpr u64_t operator()(u64_t word) const noexcept {
        word = (word ^ (word >> 30)) * 0xBF58476D1CE4E5B9ull;
        word = (word ^ (word >> 27)) * 0x94D049BB133111EBull;
        return word ^ (word >> 31);
    }

    /** Fold @p value into @p key — one order-dependent mixing step (golden-ratio increment). */
    [[nodiscard]] constexpr u64_t operator()(u64_t key, u64_t value) const noexcept {
        return (*this)(key + 0x9E3779B97F4A7C15ull + value);
    }

    /** Fold three or more values left to right by recursing on the two-argument step. */
    template <typename... rest_>
        requires(sizeof...(rest_) >= 1)
    [[nodiscard]] constexpr u64_t operator()(u64_t key, u64_t value, rest_... rest) const noexcept {
        return (*this)((*this)(key, value), static_cast<u64_t>(rest)...);
    }
};

/** The keyed bijection over @c [0,count) : a 4-round Feistel — its round the one
 *  @ref splitmix64_hash_t — over the smallest even-bit domain covering @p count , cycle-walked back
 *  into range. Identical on every host and GPU for the same @c (seed,data_epoch) , so no shuffle ever
 *  needs an exchange, and @b without replacement (a bijection) — the guarantee the hasher's draws,
 *  which collide, cannot give. */
[[nodiscard]] constexpr std::size_t epoch_permutation(u64_t seed, data_epoch_t data_epoch, std::size_t index,
                                                      std::size_t count) noexcept {
    splitmix64_hash_t const hash {};
    u64_t const key = hash(seed, data_epoch.ordinal);
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

/** Outcome of any fallible operation. @c success_k==0 ; all error codes are negative + stable. */
enum class status_t : i32_t {

    /** The operation completed; outputs are valid. */
    success_k = 0,

    /** A caller-provided buffer was too small or a device allocation failed. */
    bad_alloc_k = -10,

    /** A caller argument was null, empty, or otherwise invalid. */
    invalid_input_k = -11,

    /** The requested geometry / dtype / extents are unsupported by the dispatched kernel. */
    shape_unsupported_k = -12,

    /** A CUDA Runtime call failed. */
    cuda_error_k = -13,

    /** A host I/O / file operation failed (read / write / rename / timeout on a peer's file). */
    io_error_k = -19,

    /** An unclassified failure; prefer a specific code above when one fits. */
    other_k = -1,
};

/** Static-string name for a @ref status_t value (the single name table; no allocation). */
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

/** An output iterator @c std::vformat_to writes through, appending straight into an exception-free
 *  StringZilla string. An allocation failure latches into the shared @c failed flag (when bound) and
 *  truncates the text — for diagnostics that is acceptable as-is; for payloads, @ref format_into
 *  surfaces the flag. */
struct format_into_t {
    using difference_type = std::ptrdiff_t;

    sz::string_t *text = nullptr;
    bool *failed = nullptr;

    format_into_t &operator*() noexcept { return *this; }
    format_into_t &operator++() noexcept { return *this; }
    format_into_t operator++(int) noexcept { return *this; }
    format_into_t &operator=(char character) noexcept {
        if (sz::failed(text->push_back(character)) && failed != nullptr) *failed = true;
        return *this;
    }
};

/** `std::format` straight into an exception-free StringZilla string, appending — the one spelling
 *  for every payload and path assembly. @c false only on allocation failure (the text is then
 *  truncated); the format string is compile-time checked. */
template <typename... arg_types_>
[[nodiscard]] inline bool format_into(sz::string_t &text, std::format_string<arg_types_...> format_string,
                                      arg_types_ &&...args) noexcept {
    bool failed = false;
    std::vformat_to(format_into_t {&text, &failed}, format_string.get(), std::make_format_args(args...));
    return !failed;
}

/** A format string bundled with the @c std::source_location of the call site that spelled it. The
 *  implicit conversion from a string literal lets @ref expected::fail capture the failure site behind
 *  its variadic pack, where a defaulted trailing argument cannot go — and without a macro. */
struct format_at_t {
    char const *format = nullptr;
    std::source_location location {};

    format_at_t(char const *format, // vendor-boundary: the literal `std::vformat_to` consumes
                std::source_location location = std::source_location::current()) noexcept
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
    i32_t cu_error_ = 0; // The CUDA Driver `CUresult` code, zero (`CUDA_SUCCESS`) unless set by `cuda_failure`.
    sz::string_t description_ {};
    std::source_location location_ {};

    expected() noexcept = delete;
    expected(expected const &) noexcept = delete;
    expected &operator=(expected const &) noexcept = delete;

    /** Success: adopts @p value , making the union member live. */
    expected(value_type_ value) noexcept : value_(std::move(value)) {}

    /** Move: the error channel travels unconditionally; the value is adopted only when @p other is
     *  on the success path, since the union member is live only then. The @c noexcept is the same
     *  compile-time proof that @c value_type_ moves without throwing. */
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

    /** @c true iff the operation succeeded and @ref value_ is readable. */
    [[nodiscard]] explicit operator bool() const noexcept { return status_ == status_t::success_k; }

    /** Reach the held value on the success path, so a consumer writes @c result->field or @c *result
     *  instead of @c result.value_ . Reading it on a failed @ref expected is a logic error, gated by
     *  @ref operator bool . */
    [[nodiscard]] value_type_ *operator->() noexcept { return &value_; }
    [[nodiscard]] value_type_ const *operator->() const noexcept { return &value_; }
    [[nodiscard]] value_type_ &operator*() noexcept { return value_; }
    [[nodiscard]] value_type_ const &operator*() const noexcept { return value_; }

    /** Failure tagged with a @ref status_t : materializes @p format + @p args into the description
     *  and records the failure site @ref format_at_t captured. */
    template <typename... arg_types_>
    [[nodiscard]] static expected fail(status_t status, format_at_t format, arg_types_ &&...args) noexcept {
        expected result {failure_t {}, status};
        result.location_ = format.location;
        std::vformat_to(format_into_t {&result.description_}, format.format, std::make_format_args(args...));
        return result;
    }

    /** Failure adopted from another @ref expected 's error channel: the status, driver error,
     *  description, and failure site travel @b unchanged — the composition path for helpers, so
     *  forwarding never reformats and never loses the original location. */
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

    /** Failure ctor: sets a non-success @p status and leaves the union @ref value_ unconstructed,
     *  so the @c status_==success_k ⇔ value_ live invariant the destructor relies on holds at once. */
    expected(failure_t, status_t status) noexcept : status_(status) {}
};

/** The value-less @ref expected — the return type of fallible actions that produce no result
 *  (publish, await, barrier), carrying the same status / driver-error / description / location
 *  channel. Default-constructed means success: @c return @c {} . */
template <>
struct expected<void> {
    status_t status_ = status_t::success_k;
    i32_t cu_error_ = 0; // The CUDA Driver `CUresult` code, zero (`CUDA_SUCCESS`) unless set by `cuda_failure`.
    sz::string_t description_ {};
    std::source_location location_ {};

    expected() noexcept = default;
    expected(expected const &) noexcept = delete;
    expected &operator=(expected const &) noexcept = delete;

    expected(expected &&) noexcept = default;
    expected &operator=(expected &&) noexcept = default;

    /** @c true iff the action succeeded. */
    [[nodiscard]] explicit operator bool() const noexcept { return status_ == status_t::success_k; }

    /** Failure tagged with a @ref status_t : materializes @p format + @p args into the description
     *  and records the failure site @ref format_at_t captured. */
    template <typename... arg_types_>
    [[nodiscard]] static expected fail(status_t status, format_at_t format, arg_types_ &&...args) noexcept {
        expected result;
        result.status_ = status;
        result.location_ = format.location;
        std::vformat_to(format_into_t {&result.description_}, format.format, std::make_format_args(args...));
        return result;
    }

    /** Failure adopted from another @ref expected 's error channel, unchanged. */
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

/** The everyday spelling of the value-less @ref expected . */
using expected_t = expected<void>;

} // namespace unum::uform::core

/** `std::format` support for statuses, printing the name @c to_string gives. */
template <>
struct std::formatter<unum::uform::core::status_t> : std::formatter<std::string_view> {
    auto format(auto status, std::format_context &context) const {
        return std::formatter<std::string_view>::format(to_string(status), context);
    }
};
