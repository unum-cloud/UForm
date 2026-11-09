/**
 *  @file src_stub/lib.hpp
 *  @author Ash Vardanian
 *  @date October 17, 2026
 *  @brief Machinery every stand-in verb shares: outcomes, the calls holding executors, arenas,
 *      scalars, progress, and the deterministic numbers the encoders write.
 *
 *  Nothing here is part of the C ABI. The stand-in answers every verb of the @c c/uform headers
 *  with the C++ standard library alone, so a binding suite links and runs in seconds against the
 *  interface instead of waiting on a build of the engine. Its executors start no threads, so they
 *  behave as the engine's caller-inclusive ones: every call, deferred too, runs on its caller and
 *  is final when it returns, and a call finding another one holding its executor is refused.
 */
#ifndef UNUM_UFORM_STUB_LIB_HPP
#define UNUM_UFORM_STUB_LIB_HPP

#include <cstddef> // `std::byte`, `std::size_t`
#include <cstdint> // `std::uint64_t`
#include <cstring> // `std::memcpy`

#include <bit>              // `std::bit_cast`
#include <initializer_list> // `std::initializer_list`
#include <memory>           // `std::unique_ptr`
#include <new>              // `std::nothrow`
#include <string_view>      // `std::string_view`
#include <type_traits>      // `std::is_pointer_v`, `std::remove_pointer_t`
#include <utility>          // `std::move`
#include <variant>          // `std::variant`
#include <vector>           // `std::vector`

#include <forkunion.hpp> // `fu::standard_atomic_ref`, `fu::standard_yield_t` - the serial class's verbs

#include "uform/uform.h" // every argument structure a call copies

/** What @c uform_executor_t owns: nothing, since every call runs on its caller. */
struct uform_executor_handle_t {};

/** What @c uform_arena_t owns: one host buffer, rewound by every call and grown to
 *  a high-water mark. */
struct uform_arena_handle_t {
    std::unique_ptr<std::byte[]> memory;
    std::size_t capacity = 0;

    /** The outcome of the call holding the arena, null while free; read and written under
     *  the running calls' lock. */
    uform_outcome_t const *holder = nullptr;
};

/** What @c uform_model_t owns: the geometry a checkpoint would carry, and the bytes it
 *  was read from. */
struct uform_model_handle_t {
    uform_scalar_t quantization = uform_scalar_bf16_k;
    std::size_t memory_usage = 0;
};

/** A mesh no stand-in can bootstrap, declared so its free stays callable. */
struct uform_mesh_handle_t {};

/** A trainer no stand-in can create, declared so its free stays callable. */
struct uform_trainer_handle_t {};

namespace unum::uform {

namespace fu = ashvardanian::forkunion;

/** The serial class's verbs, as the engine maps @c cpu_class_t::serial_k, the stub's only class. */
template <typename value_type_>
using serial_atomic_ref = fu::standard_atomic_ref<value_type_>;
using serial_waiter_t = fu::standard_yield_t;

/** Scalars per embedding row every stand-in model reports. */
inline constexpr std::size_t dimensions_k = 64;

/** Tokens one item spans before truncation, in every modality. */
inline constexpr std::size_t context_k = 512;

/** Bytes of an encoded file one token covers. */
inline constexpr std::size_t file_token_bytes_k = 16;

/** Pixels along the side of the square patch one token covers. */
inline constexpr std::size_t patch_side_k = 8;

/** Samples of a decoded recording one token covers. */
inline constexpr std::size_t chunk_samples_k = 256;

/** The run read as text: a path or a spelling. */
inline std::string_view as_text(uform_bytes_view_t bytes) {
    return {static_cast<char const *>(static_cast<void const *>(bytes.data)), bytes.length};
}

/** Whether @p name spells @p spelling exactly, with no terminator assumed on either side. */
inline bool named(uform_bytes_view_t name, std::string_view spelling) { return as_text(name) == spelling; }

/** Writes to @p output the one of @p values that @p spell spells as @p name, or fails with
 *  @c uform_status_invalid_argument_k and writes nothing. */
template <typename value_type_>
uform_status_t read_name(uform_bytes_view_t name, std::initializer_list<value_type_> values,
                         char const *(*spell)(value_type_), value_type_ *output) {
    if (!output) return uform_status_invalid_argument_k;
    for (value_type_ const value : values)
        if (named(name, spell(value))) return *output = value, uform_status_success_k;
    return uform_status_invalid_argument_k;
}

/** Marks @p outcome deferred, every verb's first act, so a waiter never reads a stale status. */
inline void propagate_deferred(uform_outcome_t *outcome) {
    if (outcome)
        serial_atomic_ref<uform_status_t> {outcome->status}.store(uform_status_deferred_k, std::memory_order_relaxed);
}

/** The one final write to @p outcome: the message, then the status with release ordering, which
 *  every waiter acquires. The message is null exactly on success. */
inline void propagate_error(uform_status_t status, uform_outcome_t *outcome, char const *message) {
    if (!outcome) return;
    outcome->message = status == uform_status_success_k ? nullptr : message;
    serial_atomic_ref<uform_status_t> {outcome->status}.store(status, std::memory_order_release);
}

/** Lets go of @p arena if the call whose outcome is @p call holds it. */
void release_arena(uform_arena_t arena, uform_outcome_t const *call);

/**
 *  @brief A verb's outcome while it runs: deferred from construction, and published once at
 *      destruction with whatever its guards recorded, success when none did.
 *
 *  Every verb declares it first, so it unwinds last: the arena it was given is let go before the
 *  final status lets a waiter free or reuse what the call named.
 */
class deferred_outcome_t {
    uform_outcome_t *published_;
    uform_arena_t arena_;
    uform_outcome_t recorded_ {uform_status_success_k, nullptr};

  public:
    explicit deferred_outcome_t(uform_outcome_t *published, uform_arena_t arena = nullptr) noexcept
        : published_(published), arena_(arena) {
        propagate_deferred(published_);
    }
    ~deferred_outcome_t() noexcept {
        if (arena_) release_arena(arena_, published_);
        propagate_error(recorded_.status, published_, recorded_.message);
    }
    deferred_outcome_t(deferred_outcome_t const &) = delete;
    deferred_outcome_t &operator=(deferred_outcome_t const &) = delete;

    /** The private record every guard writes into, so a verb passes this wherever
     *  an outcome goes. */
    operator uform_outcome_t *() noexcept { return &recorded_; }
};

/** Records @p status when @p wrong, answering whether it did, so a caller reads
 *  `if (reject_if(...)) return;`. */
inline bool reject_if(bool wrong, uform_status_t status, char const *message, uform_outcome_t *outcome) {
    if (!wrong) return false;
    propagate_error(status, outcome, message);
    return true;
}

/** The copied arguments of every verb that names an executor and a completion. */
using call_arguments_t = std::variant<                                                            //
    uform_model_load_ram_t, uform_model_load_ram_buffer_t, uform_model_load_vram_t,               //
    uform_model_load_vram_buffer_t, uform_model_describe_t,                                       //
    uform_text_encode_t, uform_image_encode_t, uform_image_encode_pixels_t, uform_video_encode_t, //
    uform_audio_encode_t, uform_audio_encode_samples_t, uform_text_choose_t,                      //
    uform_mesh_bootstrap_t, uform_trainer_init_vram_t, uform_trainer_step_t, uform_trainer_save_t>;

/** A call running on its caller's thread, holding its executor. */
struct call_t {

    /** The executor the call holds while it runs. */
    uform_executor_handle_t const *held_executor;

    /** A copy of the arguments with a null executor, so running them goes straight to
     *  the verb's body. */
    call_arguments_t arguments;
    void (*verb)(void *);

    /** The call's outcome, its executor, and every model, arena, mesh and trainer it names: the
     *  addresses a wait or a free looks for before it returns. */
    std::vector<void const *> handles;
};

/** Runs @p call on the calling thread, listed while it runs so a free of a handle it names waits,
 *  or finishes @p outcome contended while another call holds its executor. */
void run_holding(call_t call, uform_outcome_t *outcome);

/** Returns once no running call names @p handle. */
void wait_for_calls_using(void const *handle);

/** Holds @p arena for the call whose outcome is @p call, answering whether that call holds it now.
 *  A null arena holds nothing, left for the verb to refuse. */
bool hold_arena(uform_arena_t arena, uform_outcome_t const *call);

/** Collects @c call_t::handles from a verb's @p arguments, skipping the outputs, which name
 *  no handle yet. */
template <typename arguments_>
std::vector<void const *> handles_used_by(arguments_ const &arguments) {
    std::vector<void const *> handles {arguments.outcome, arguments.executor};
    auto const add = [&](auto handle) {
        if constexpr (!std::is_pointer_v<std::remove_pointer_t<decltype(handle)>>) handles.push_back(handle);
    };
    if constexpr (requires { arguments.model; }) add(arguments.model);
    if constexpr (requires { arguments.arena; }) add(arguments.arena);
    if constexpr (requires { arguments.mesh; }) add(arguments.mesh);
    if constexpr (requires { arguments.trainer; }) add(arguments.trainer);
    return handles;
}

/**
 *  @brief Every verb's first act. A call without an executor answers false and runs the verb's
 *      body in place.
 *
 *  A call with one is copied without it and answers true once it ran on the calling thread, holding
 *  its executor, whatever its completion, as a caller-inclusive executor of the engine runs it.
 */
template <auto verb_, typename arguments_>
bool through_executor(arguments_ *arguments) {
    uform_executor_handle_t *const executor = arguments->executor;
    if (!executor) return false;
    arguments_ copy = *arguments;
    copy.executor = nullptr;
    call_t call {executor, copy, [](void *erased) { verb_(static_cast<arguments_ *>(erased)); },
                 handles_used_by(*arguments)};
    run_holding(std::move(call), arguments->outcome);
    return true;
}

/** Claims @p arena for the call whose outcome is @p call, refusing a null arena, one another call
 *  holds, and any of @p inputs inside the memory the call is about to rewind. */
inline uform_arena_handle_t *claim_arena(uform_arena_t arena, uform_outcome_t const *call,
                                         std::initializer_list<void const *> inputs, uform_outcome_t *outcome) {
    if (reject_if(!arena, uform_status_invalid_argument_k, "The arena must not be null", outcome)) return nullptr;
    if (reject_if(!hold_arena(arena, call), uform_status_contended_k, "The arena serves another call", outcome))
        return nullptr;
    std::byte const *const begin = arena->memory.get();
    for (void const *input : inputs) {
        std::byte const *const at = static_cast<std::byte const *>(input);
        if (reject_if(at && begin && at >= begin && at < begin + arena->capacity, uform_status_invalid_argument_k,
                      "An input lies inside the arena the call rewinds", outcome))
            return nullptr;
    }
    return arena;
}

/** Rewinds a claimed @p arena and hands out its first @p bytes, growing it when they outsize its
 *  high-water mark; null once @p outcome records why not. */
inline void *rewind(uform_arena_handle_t &arena, std::size_t bytes, uform_outcome_t *outcome) {
    if (bytes <= arena.capacity && arena.memory) return arena.memory.get();
    std::unique_ptr<std::byte[]> grown(new (std::nothrow) std::byte[bytes ? bytes : 1]);
    if (reject_if(!grown, uform_status_out_of_memory_k, "Failed to grow the arena", outcome)) return nullptr;
    arena.memory = std::move(grown);
    arena.capacity = bytes;
    return arena.memory.get();
}

/** The quotient of @p value by @p divisor, rounded up. */
inline std::size_t divide_round_up(std::size_t value, std::size_t divisor) noexcept {
    return value / divisor + (value % divisor != 0);
}

/** The stored width of one output value, zero for a scalar the stand-in does not write. */
inline std::size_t output_scalar_bytes(uform_scalar_t scalar) {
    switch (scalar) {
    case uform_scalar_f32_k: return 4;
    case uform_scalar_bf16_k: return 2;
    case uform_scalar_f16_k: return 2;
    default: return 0;
    }
}

/** Narrows to the high half of an IEEE single, rounding half to even and keeping a NaN a NaN. */
inline std::uint16_t to_bf16(float value) {
    std::uint32_t const bits = std::bit_cast<std::uint32_t>(value);
    if ((bits & 0x7F800000u) == 0x7F800000u && (bits & 0x007FFFFFu)) return (bits >> 16) | 0x0040u;
    return static_cast<std::uint16_t>((bits + 0x7FFFu + ((bits >> 16) & 1u)) >> 16);
}

/** Narrows to IEEE half precision by truncation, flushing what falls below the normal
 *  range to zero. */
inline std::uint16_t to_f16(float value) {
    std::uint32_t const bits = std::bit_cast<std::uint32_t>(value);
    std::uint32_t const sign = (bits >> 16) & 0x8000u;
    int const exponent = static_cast<int>((bits >> 23) & 0xFFu) - 127 + 15;
    std::uint32_t const mantissa = (bits >> 13) & 0x3FFu;
    if (exponent <= 0) return static_cast<std::uint16_t>(sign);
    if (exponent >= 31) return static_cast<std::uint16_t>(sign | 0x7C00u);
    return static_cast<std::uint16_t>(sign | (static_cast<std::uint32_t>(exponent) << 10) | mantissa);
}

/** Writes @p value as the @p position -th element of a row of @p scalar, which
 *  @c output_scalar_bytes admits. */
inline void write_value(std::byte *row, uform_scalar_t scalar, std::size_t position, float value) {
    std::uint16_t const half = scalar == uform_scalar_bf16_k ? to_bf16(value) : to_f16(value);
    if (scalar == uform_scalar_f32_k) std::memcpy(row + position * sizeof(value), &value, sizeof(value));
    else std::memcpy(row + position * sizeof(half), &half, sizeof(half));
}

/** Continues a 64-bit FNV-1a hash over @p bytes from @p hash. */
inline std::uint64_t fnv1a(std::byte const *bytes, std::size_t length,
                           std::uint64_t hash = 0xCBF29CE484222325ull) noexcept {
    for (std::size_t position = 0; position != length; ++position)
        hash = (hash ^ static_cast<std::uint64_t>(bytes[position])) * 0x100000001B3ull;
    return hash;
}

/** Fills one row of @c dimensions_k with xorshift draws in [-1, 1) seeded by @p seed. */
inline void write_row(std::byte *row, uform_scalar_t scalar, std::uint64_t seed) {
    std::uint64_t state = seed ? seed : 0x9E3779B97F4A7C15ull;
    for (std::size_t position = 0; position != dimensions_k; ++position) {
        state ^= state << 13, state ^= state >> 7, state ^= state << 17;
        float const unit = static_cast<float>(state >> 40) * (1.0f / 16777216.0f);
        write_value(row, scalar, position, unit * 2.0f - 1.0f);
    }
}

/** How a batch verb splits its items for its handler: @c steps nearly equal slices, reported after
 *  each, or one silent slice without a handler. */
struct progress_ticks_t {
    uform_progress_t progress;
    std::size_t count;

    /** Slices the call runs in, never more than its items, and one for an empty call. */
    std::size_t slices() const noexcept {
        if (!progress.callback) return 1;
        std::size_t const steps = progress.steps ? progress.steps : 10;
        return count == 0 ? 1 : (steps < count ? steps : count);
    }

    /** One past the last item of @p slice. */
    std::size_t end_of(std::size_t slice) const noexcept { return count * (slice + 1) / slices(); }

    /** Reports @p completed items, answering whether the handler lets the call carry on. */
    bool carry_on(std::size_t completed) const {
        return !progress.callback || progress.callback(completed, count, progress.state) == uform_continue_k;
    }
};

} // namespace unum::uform

#endif // UNUM_UFORM_STUB_LIB_HPP
