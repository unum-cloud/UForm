/**
 *  @file include/uform/utilities/types.hpp
 *  @author Ash Vardanian
 *  @date July 2, 2026
 *  @brief Foundational vocabulary of the UForm v4 primitives cell: statuses, error
 *      channel, launch constants.
 *
 *  The primitives cell holds the self-contained CUDA building blocks the rest of the v4 stack
 *  assembles — losses, collectives, packers — each a header-only @c .cuh with its own kernels and
 *  @c launch_* wrappers, parameterized by NumKong spans and a caller-owned stream + workspace, with
 *  no vendor-library sessions, no hidden state, and no allocation on any launch path.
 */
#pragma once

#include <cassert> // `assert`
#include <cstddef> // `std::ptrdiff_t`
#include <cstdint> // `std::int32_t`, `std::uint32_t`

#include <concepts>        // `std::unsigned_integral`
#include <limits>          // `std::numeric_limits`
#include <utility>         // `std::move`, `std::tuple_size`, `std::tuple_element`, `std::initializer_list`
#include <span>            // `std::span` — byte runs and tape offsets
#include <new>             // placement `new` — the value an `expected` holds
#include <type_traits>     // `std::type_identity_t`, `std::is_trivially_copyable_v`
#include <format>          // `std::vformat_to`, `std::make_format_args`, `std::formatter`
#include <source_location> // `std::source_location`

#include <forkunion/types.hpp>         // `ashvardanian::forkunion`, which `fu` names
#include <numkong/tensor.hpp>          // `nk::tensor_span`, `nk::tensor_view` — the rank-tagged views and spans
#include <stringzilla/string.hpp>      // `sz::string_t` — the exception-free diagnostics carrier
#include <smashtable/basic_vector.hpp> // `st::basic_vector` — the exception-free growable array

#include "uform/types.h" // `uform_status_t`, whose values `status_code_t` takes

/** @c std::format support for StringZilla slices, so diagnostics interpolate them directly —
 *  no conversion shims at call sites. */
template <>
struct std::formatter<ashvardanian::stringzilla::string_view_t> : std::formatter<std::string_view> {
    auto format(ashvardanian::stringzilla::string_view_t text, std::format_context& context) const noexcept {
        return std::formatter<std::string_view>::format(std::string_view {text}, context);
    }
};

/** @c std::format support for owning StringZilla strings — same bridge, by reference. */
template <>
struct std::formatter<ashvardanian::stringzilla::string_t> : std::formatter<std::string_view> {
    auto format(ashvardanian::stringzilla::string_t const& text, std::format_context& context) const noexcept {
        return std::formatter<std::string_view>::format(std::string_view {text}, context);
    }
};

namespace unum::uform::utilities {

namespace fu = ashvardanian::forkunion;
namespace sz = ashvardanian::stringzilla;
namespace nk = ashvardanian::numkong;
namespace st = ashvardanian::smashtable;

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

/** The rank-tagged view vocabulary over the scalar dtypes — the view type of a dtype and rank
 *  reads, the span type writes, each pointing into memory owned elsewhere and valid while it lives.
 *  The owning @c *_tensor_arena_t / @c *_tensor_pinned_t counterparts live beside the allocator
 *  they bind in @c gpu_allocators.hpp , as do the CUDA-scalar @c bf16_t / @c e4m3_t views.
 *  Concept-named views (a token rectangle, a pixel image) stay with their cell. */
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
using f32r4_view_t = nk::tensor_view<f32_t, 4>;
using f32r4_span_t = nk::tensor_span<f32_t, 4>;

/** Entries end to end and @c count+1 offsets into them, as the C ABI's tapes hold them;
 *  @c elements runs from where every offset counts to the last offset, and an empty tape
 *  holds no entries. */
template <typename element_type_>
struct tape_view {
    std::span<element_type_> elements {};
    std::span<std::size_t const> offsets {};

    [[nodiscard]] std::size_t size() const noexcept { return offsets.empty() ? 0 : offsets.size() - 1; }
    [[nodiscard]] bool empty() const noexcept { return offsets.empty(); }
    [[nodiscard]] std::span<element_type_> operator[](std::size_t index) const noexcept {
        assert(offsets[index + 1] >= offsets[index] && "A tape's offsets must not decrease");
        return elements.subspan(offsets[index], offsets[index + 1] - offsets[index]);
    }
};

/** A read-only run of raw bytes. */
using bytes_view_t = std::span<b8_t const>;

/** A writable run of raw bytes. */
using bytes_span_t = std::span<b8_t>;

/** One raw byte, as the Shared VRAM region spells it in every engine. */
using byte_t = std::byte;

/** One vocabulary entry's index — the byte-level vocabulary (256 bytes + the special tokens) fits
 *  16 bits, and every token tape, staging cell, and embedding lookup trades in this width. */
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
 *  @sa Defines the upper bound for @ref util::mesh_shape_t::gpus_per_host_domain.
 */
inline constexpr unsigned max_gpus_per_host_k = 32u;

/** Devices one vendor's listing holds at most, so the process-wide listing needs no heap. */
inline constexpr std::size_t gpu_devices_capacity_k = 64;

/** One vendor's kernel bundle as every device loaded it, which @c vram_device.hpp defines and the
 *  GPU runtime owns; a stream points at its vendor's. */
template <typename module_type_>
struct kernel_tables;

/** The GPU vendors one binary drives at once, each through a driver opened at first use, with the
 *  values of @c uform_gpu_vendor_t , so crossing the C ABI is a cast. */
enum class gpu_vendor_t : u32_t {

    /** No vendor, as a device no driver reports holds. */
    unknown_k = uform_gpu_vendor_unknown_k,

    /** NVIDIA, through the CUDA driver. */
    nvidia_k = uform_gpu_vendor_nvidia_k,

    /** AMD, through the HIP runtime. */
    amd_k = uform_gpu_vendor_amd_k,

    /** Apple, through Metal. */
    apple_k = uform_gpu_vendor_apple_k,
};

/** One GPU as its vendor's driver reports it - the vendor, that vendor's own ordinal and its raw
 *  architecture - with what a listing reports, so no caller asks a vendor SDK. */
struct gpu_device_t {
    gpu_vendor_t vendor = gpu_vendor_t::unknown_k;
    u32_t vendor_ordinal = 0;

    /** The CUDA compute capability, the AMD @c gfx version or the newest Apple GPU family, major
     *  then minor. */
    u32_t architecture_major = 0;
    u32_t architecture_minor = 0;

    /** Lanes of one warp, wavefront or simdgroup, as the driver reports it; zero where only a built
     *  pipeline reports it. */
    u32_t warp_lanes = 0;

    /** Streaming multiprocessors, compute units or GPU cores. */
    u32_t compute_units = 0;

    /** Device memory, or the working set Metal recommends for unified memory. */
    u64_t memory_bytes = 0;

    /** The vendor's own name for the device, NUL-terminated and truncated to fit. */
    char name[64] = {};
};

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

/** @p value divided by @p divisor and rounded up, without the overflow <tt>value + divisor - 1</tt>
 *  risks near the limit. */
template <std::unsigned_integral integer_type_>
[[nodiscard]] constexpr integer_type_ divide_round_up(integer_type_ value,
                                                      std::type_identity_t<integer_type_> divisor) noexcept {
    return static_cast<integer_type_>(value / divisor + (value % divisor != 0));
}

/** The least multiple of @p multiple at or above @p value, saturating at the greatest
 *  multiple @p integer_type_ holds. */
template <std::unsigned_integral integer_type_>
[[nodiscard]] constexpr integer_type_ round_up_to_multiple(integer_type_ value,
                                                           std::type_identity_t<integer_type_> multiple) noexcept {
    integer_type_ const below = static_cast<integer_type_>(value - value % multiple);
    if (below == value || below > (std::numeric_limits<integer_type_>::max)() - multiple) return below;
    return static_cast<integer_type_>(below + multiple);
}

/** Whether @p a times @p b exceeds what @p integer_type_ holds. */
template <std::unsigned_integral integer_type_>
[[nodiscard]] constexpr bool wraps_on_multiply(integer_type_ a, std::type_identity_t<integer_type_> b) noexcept {
    return b != 0 && a > (std::numeric_limits<integer_type_>::max)() / b;
}

/** @c ceil(log2(value)) for @c value>=1 — the recursive-doubling round count over
 *  @c value domains. */
[[nodiscard]] inline constexpr u32_t ceil_log2(u32_t value) noexcept {
    u32_t rounds = 0;
    while ((1u << rounds) < value) ++rounds;
    return rounds;
}

/**
 *  @brief The project's one keyed hasher — a splitmix64 core, used three ways and nothing else, so
 *      no free-standing @c mix / @c combine / @c derive helpers proliferate.
 *
 *  Pure and device-callable; stateless by design (no engine object to step, unlike
 *  @c std::minstd or a @c curandState ), so a thread hashes its own @c (key,index) pair
 *  independently and any result is recomputable from its inputs on any host or GPU and across
 *  restarts. Every keyed draw — dataset shuffling, batch mixing, span masking, dropout — builds
 *  on this; @ref epoch_permutation is the one primitive it cannot be (a hash collides, a
 *  permutation does not).
 *
 *  - @c hash(word) avalanches one word (the finalizer).
 *  - @c hash(key,value) folds @p value into @p key (one mixing step).
 *  - @c hash(a,b,c,...) folds every argument left to right — the seeder over
 *    @c (seed,data_epoch,member,index,salt,...) , and the indexed draw @c hash(key,index) is just
 *    the two-argument case.
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
 *  @ref util::splitmix64_hash_t — over the smallest even-bit domain covering @p count ,
 *  cycle-walked back into range. Identical on every host and GPU for the same
 *  @c (seed,data_epoch) , so no shuffle ever needs an exchange, and @b without replacement (a
 *  bijection) — the guarantee the hasher's draws, which collide, cannot give. */
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

/**
 *  @brief Code of any fallible operation. Every enumerator takes its spelling and value from
 *      @c uform_status_t , so crossing the C ABI is a cast.
 *
 *  Any verb that allocates, binds a device or touches a file may also return a resource status.
 *  A verb's own block lists only what its contract adds.
 */
enum class status_code_t : i32_t {

    /** The operation completed; outputs are valid. */
    success_k = uform_status_success_k,

    /** Queued work still runs; its outcome is not final yet. */
    deferred_k = uform_status_deferred_k,

    /** A resource status: a caller-provided buffer was too small or a device allocation failed. */
    out_of_memory_k = uform_status_out_of_memory_k,

    /** A caller argument was null, empty, or otherwise invalid. */
    invalid_argument_k = uform_status_invalid_argument_k,

    /** The requested geometry / dtype / extents are unsupported by the dispatched kernel. */
    unsupported_k = uform_status_unsupported_k,

    /** A resource status: a device runtime call failed. */
    device_k = uform_status_device_k,

    /** A resource status: a host file read, write or rename failed, or a peer's file timed out. */
    io_k = uform_status_io_k,

    /** A shared resource, such as a thread pool, was busy with another caller's work. */
    contended_k = uform_status_contended_k,

    /** An unclassified failure; prefer a specific code above when one fits. */
    internal_k = uform_status_internal_k,

    /** The object was used before it was opened or after it was emptied. */
    uninitialized_k = uform_status_uninitialized_k,

    /** Valid arguments that disagree with each other or with the state they name. */
    config_mismatch_k = uform_status_config_mismatch_k,
};

/** Why a call failed: a code recovery branches on, and an optional static note naming the broken
 *  invariant, step or resource, for logs and regression analysis. */
struct [[nodiscard]] status_t {
    status_code_t code = status_code_t::success_k;
    char const* note = nullptr;

    constexpr status_t() noexcept = default;
    constexpr status_t(status_code_t reason, char const* detail = nullptr) noexcept : code(reason), note(detail) {}
    constexpr bool succeeded() const noexcept { return code == status_code_t::success_k; }
    constexpr bool failed() const noexcept { return code != status_code_t::success_k; }
    friend constexpr bool operator==(status_t const& status, status_code_t code) noexcept {
        return status.code == code;
    }
};

static_assert(sizeof(status_t) == 2 * sizeof(void*) && std::is_trivially_copyable_v<status_t>,
              "a status is a code and a note pointer, returned in two registers");

/** Static-string name for a @ref status_code_t value (the single name table; no allocation). */
[[nodiscard]] inline char const* to_string(status_code_t code) noexcept {
    switch (code) {
    case status_code_t::success_k: return "success";
    case status_code_t::deferred_k: return "deferred";
    case status_code_t::out_of_memory_k: return "out_of_memory";
    case status_code_t::invalid_argument_k: return "invalid_argument";
    case status_code_t::unsupported_k: return "unsupported";
    case status_code_t::device_k: return "device";
    case status_code_t::io_k: return "io";
    case status_code_t::contended_k: return "contended";
    case status_code_t::internal_k: return "internal";
    case status_code_t::uninitialized_k: return "uninitialized";
    case status_code_t::config_mismatch_k: return "config_mismatch";
    }
    return "unknown";
}

/** The note of @p status , else its code's name. */
[[nodiscard]] inline char const* to_string(status_t status) noexcept {
    return status.note ? status.note : to_string(status.code);
}

/** An output iterator @c std::vformat_to writes through, appending straight into an exception-free
 *  StringZilla string. An allocation failure latches into the shared @c failed flag (when bound)
 *  and truncates the text — for diagnostics that is acceptable as-is; for payloads,
 *  @ref format_into surfaces the flag. */
struct format_into_t {
    using difference_type = std::ptrdiff_t;

    sz::string_t* text = nullptr;
    bool* failed = nullptr;

    format_into_t& operator*() noexcept { return *this; }
    format_into_t& operator++() noexcept { return *this; }
    format_into_t operator++(int) noexcept { return *this; }
    format_into_t& operator=(char character) noexcept {
        if (sz::failed(text->push_back(character)) && failed != nullptr) *failed = true;
        return *this;
    }
};

/** @c std::format straight into an exception-free StringZilla string, appending — the one spelling
 *  for every payload and path assembly. @c false only on allocation failure (the text is then
 *  truncated); the format string is compile-time checked. */
template <typename... arg_types_>
[[nodiscard]] inline bool format_into(sz::string_t& text, std::format_string<arg_types_...> format_string,
                                      arg_types_&&... args) noexcept {
    bool failed = false;
    std::vformat_to(format_into_t {&text, &failed}, format_string.get(), std::make_format_args(args...));
    return !failed;
}

/** Allocate @p tensor as @c tensor_type_::uninitialized builds it from @p extents and
 *  @p allocator , reporting whether NumKong succeeded; on failure @p tensor is left empty. */
template <typename tensor_type_>
[[nodiscard]] bool allocate_uninitialized(tensor_type_& tensor, std::initializer_list<std::size_t> extents,
                                          typename tensor_type_::allocator_type allocator) noexcept {
    auto [allocated, allocate_status] = tensor_type_::uninitialized(extents, allocator);
    tensor = std::move(allocated);
    return nk::succeeded(allocate_status);
}

/** A format string bundled with the @c std::source_location of the call site that spelled
 *  it. The implicit conversion from a string literal lets @ref expected::fail capture the
 *  failure site behind its variadic pack, where a defaulted trailing argument cannot go —
 *  and without a macro. */
struct format_at_t {
    char const* format = nullptr;
    std::source_location location {};

    format_at_t(char const* format, // vendor-boundary: the literal `std::vformat_to` consumes
                std::source_location location = std::source_location::current()) noexcept
        : format(format), location(location) {}
};

template <typename value_type_>
struct expected;

/** The value-less @ref expected — the return type of fallible actions that produce no result
 *  (publish, await, barrier), carrying the same status / driver-error / description / location
 *  channel. Default-constructed means success: @c return @c {} . */
template <>
struct expected<void> {
    status_t status_ {};
    i32_t cu_error_ = 0; // The CUDA Driver `CUresult` code, zero (`CUDA_SUCCESS`) unless set by `cuda_failure`.
    sz::string_t description_ {};
    std::source_location location_ {};

    expected() noexcept = default;
    expected(expected const&) noexcept = delete;
    expected& operator=(expected const&) noexcept = delete;

    expected(expected&&) noexcept = default;
    expected& operator=(expected&&) noexcept = default;

    /** @c true iff the action succeeded. */
    [[nodiscard]] explicit operator bool() const noexcept { return status_.succeeded(); }

    /** The status's own reads, so a status bound from <tt>auto [value, status]</tt> reads as a
     *  @ref status_t does in the Shared VRAM text. */
    [[nodiscard]] bool succeeded() const noexcept { return status_.succeeded(); }
    [[nodiscard]] bool failed() const noexcept { return status_.failed(); }

    /** The code and note alone, which a verb answering a @ref status_t passes on. */
    operator status_t() const noexcept { return status_; }

    /** Failure tagged with a @ref status_t : materializes @p format + @p args into the description
     *  and records the failure site @ref format_at_t captured. */
    template <typename... arg_types_>
    [[nodiscard]] static expected fail(status_t status, format_at_t format, arg_types_&&... args) noexcept {
        expected result;
        result.status_ = status;
        result.location_ = format.location;
        std::vformat_to(format_into_t {&result.description_}, format.format, std::make_format_args(args...));
        return result;
    }

    /** Failure tagged with @p code and its static @p note , with nothing to format. */
    [[nodiscard]] static expected fail(status_code_t code, char const* note,
                                       std::source_location location = std::source_location::current()) noexcept {
        expected failure;
        failure.status_ = {code, note};
        failure.location_ = location;
        return failure;
    }

    /** Failure adopted from another @ref expected 's error channel, unchanged. */
    template <typename other_type_>
    [[nodiscard]] static expected fail(expected<other_type_>&& other) noexcept {
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

/**
 *  @brief Minimal host-side value-or-error carrier — the return type of every fallible host
 *      function in the primitives cell.
 *
 *  Move-only: the explicit @c noexcept on the defaulted move members is a compile-time check that
 *  every instantiating @c value_type_ moves without throwing, so a result propagates up the call
 *  chain as nothing but @c noexcept moves. The description is an exception-free StringZilla string,
 *  materialized with @c std::format at the failure site — the only place the operands (shapes,
 *  counts, labels) are still in scope — and @ref format_at_t records that site's
 *  @c std::source_location .
 */
template <typename value_type_>
struct expected {
    union {
        value_type_ value_; // Live only on the success path (status_ == success_k); managed by hand below.
    };
    status_t status_ {};
    i32_t cu_error_ = 0; // The CUDA Driver `CUresult` code, zero (`CUDA_SUCCESS`) unless set by `cuda_failure`.
    sz::string_t description_ {};
    std::source_location location_ {};

    expected() noexcept = delete;
    expected(expected const&) noexcept = delete;
    expected& operator=(expected const&) noexcept = delete;

    /** Success: adopts @p value , making the union member live. */
    expected(value_type_ value) noexcept : value_(std::move(value)) {}

    /** The value and status the Shared VRAM text returns as <tt>return {std::move(value), ok}</tt>
     *  or <tt>return {{}, status}</tt> : @p value lives only when @p status succeeds, and the
     *  driver error, description and location keep their defaults. */
    expected(value_type_&& value, status_t status) noexcept : status_(status) {
        if (status_.succeeded()) ::new (static_cast<void*>(&value_)) value_type_(std::move(value)); // vendor-boundary
    }

    /** Move: the error channel travels unconditionally; the value is adopted only when @p other is
     *  on the success path, since the union member is live only then. The @c noexcept is the same
     *  compile-time proof that @c value_type_ moves without throwing. */
    expected(expected&& other) noexcept
        : status_(other.status_), cu_error_(other.cu_error_), description_(std::move(other.description_)),
          location_(other.location_) {
        if (status_ == status_code_t::success_k)
            ::new (static_cast<void*>(&value_)) value_type_(std::move(other.value_)); // vendor-boundary
    }
    expected& operator=(expected&& other) noexcept {
        if (this != &other) {
            if (status_ == status_code_t::success_k) value_.~value_type_();
            status_ = other.status_;
            cu_error_ = other.cu_error_;
            description_ = std::move(other.description_);
            location_ = other.location_;
            if (status_ == status_code_t::success_k)
                ::new (static_cast<void*>(&value_)) value_type_(std::move(other.value_)); // vendor-boundary
        }
        return *this;
    }
    ~expected() noexcept {
        if (status_ == status_code_t::success_k) value_.~value_type_();
    }

    /** @c true iff the operation succeeded and the value is readable. */
    [[nodiscard]] explicit operator bool() const noexcept { return status_.succeeded(); }

    /** Reach the held value on the success path, so a consumer writes @c result->field or
     *  @c *result instead of @c result.value_ . Reading it on a failed @ref expected is a logic
     *  error, gated by its @c bool conversion . */
    [[nodiscard]] value_type_* operator->() noexcept { return &value_; }
    [[nodiscard]] value_type_ const* operator->() const noexcept { return &value_; }
    [[nodiscard]] value_type_& operator*() noexcept { return value_; }
    [[nodiscard]] value_type_ const& operator*() const noexcept { return value_; }

    /** The tuple protocol behind <tt>auto [value, status] = ...</tt>: the value by value, built
     *  empty on failure so no name binds storage that was never constructed, then the status
     *  record, which reads as @c true on success. */
    template <std::size_t index_>
        requires std::is_nothrow_default_constructible_v<value_type_>
    [[nodiscard]] auto get() && noexcept {
        if constexpr (index_ == 0) {
            if (status_ != status_code_t::success_k) return value_type_();
            return value_type_(std::move(value_));
        }
        else {
            expected_t status;
            status.status_ = status_;
            status.cu_error_ = cu_error_;
            status.description_ = std::move(description_);
            status.location_ = location_;
            return status;
        }
    }

    /** Failure tagged with a @ref status_t : materializes @p format + @p args into the description
     *  and records the failure site @ref format_at_t captured. */
    template <typename... arg_types_>
    [[nodiscard]] static expected fail(status_t status, format_at_t format, arg_types_&&... args) noexcept {
        expected result {failure_t {}, status};
        result.location_ = format.location;
        std::vformat_to(format_into_t {&result.description_}, format.format, std::make_format_args(args...));
        return result;
    }

    /** Failure tagged with @p code and its static @p note , with nothing to format. */
    [[nodiscard]] static expected fail(status_code_t code, char const* note,
                                       std::source_location location = std::source_location::current()) noexcept {
        expected failure {failure_t {}, {code, note}};
        failure.location_ = location;
        return failure;
    }

    /** Failure adopted from another @ref expected 's error channel: the status, driver error,
     *  description, and failure site travel @b unchanged — the composition path for helpers, so
     *  forwarding never reformats and never loses the original location. */
    template <typename other_type_>
    [[nodiscard]] static expected fail(expected<other_type_>&& other) noexcept {
        expected result {failure_t {}, other.status_};
        result.cu_error_ = other.cu_error_;
        result.description_ = std::move(other.description_);
        result.location_ = other.location_;
        return result;
    }

  private:
    /** The tag of the failure constructor, which no brace @c {} spells, so the value-and-status
     *  constructor alone takes <tt>{{}, status}</tt> . */
    struct failure_t {
        explicit failure_t() noexcept = default;
    };

    /** Failure ctor: sets a non-success @p status and leaves the union @c value_ unconstructed, so
     *  the @c status_==success_k ⇔ value_ live invariant the destructor relies on holds at once; a
     *  success status handed here becomes @c internal_k , as no value was built. */
    expected(failure_t, status_t status) noexcept
        : status_(status.succeeded() ? status_t {status_code_t::internal_k, "An expected value is missing"} : status) {}
};

} // namespace unum::uform::utilities

namespace unum::uform {
namespace util = utilities;
} // namespace unum::uform

/** What <tt>auto [value, status] = ...</tt> looks up: the value lives in a @c union , so the tuple
 *  protocol is the only route. */
template <typename value_type_>
struct std::tuple_size<unum::uform::utilities::expected<value_type_>> : std::integral_constant<std::size_t, 2> {};

template <typename value_type_>
struct std::tuple_element<0, unum::uform::utilities::expected<value_type_>> {
    using type = value_type_;
};

template <typename value_type_>
struct std::tuple_element<1, unum::uform::utilities::expected<value_type_>> {
    using type = typename unum::uform::utilities::expected_t;
};

/** @c std::format support for statuses, printing the note or the code name @c to_string gives. */
template <>
struct std::formatter<unum::uform::utilities::status_t> : std::formatter<std::string_view> {
    auto format(auto status, std::format_context& context) const noexcept {
        return std::formatter<std::string_view>::format(to_string(status), context);
    }
};
