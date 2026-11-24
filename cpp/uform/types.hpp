/**
 *  @file cpp/uform/types.hpp
 *  @author Ash Vardanian
 *  @date November 18, 2026
 *  @brief The pieces every C++ verb shares: statuses and expected values, blocking and deferred
 *      calls and the awaiter that polls them, owned handles, element types, rows and tapes,
 *      handlers, executors and arenas, adding no ABI of their own and throwing nothing.
 *
 *  Every verb answers through a receiver whose type picks how it completes: a blocking receiver
 *  answers a bare @c status_t or an @c expected value unpacked as a pair, and a deferred one
 *  answers a @c deferred call its one owner polls, waits on or awaits. Handlers are @c noexcept
 *  callables the caller keeps for one call, so no fault ever unwinds through the library.
 */
#pragma once

#include <cstddef> // `std::size_t`, `std::byte`
#include <cstdint> // `std::uint8_t`, `std::uint16_t`, `std::int8_t`, `std::int32_t`
#include <cstring> // `std::memcpy`

#include <concepts>    // `std::invocable`
#include <coroutine>   // `std::coroutine_handle`
#include <ranges>      // `std::ranges::sized_range`, `std::ranges::size`
#include <span>        // `std::span`
#include <string_view> // `std::string_view`
#include <type_traits> // `std::conditional_t`, `std::is_void_v`, `std::is_nothrow_invocable_r_v`
#include <utility>     // `std::exchange`, `std::move`, `std::swap`

#include "uform/types.h" // `uform_outcome_t`, `uform_outcome_status`, `uform_outcome_wait`

namespace unum::uform {

/** What went wrong, with one enumerator per status the library reports. */
enum class status_code_t : int {
    success_k = uform_status_success_k,
    deferred_k = uform_status_deferred_k,
    out_of_memory_k = uform_status_out_of_memory_k,
    shape_mismatch_k = uform_status_shape_mismatch_k,
    unsupported_layout_k = uform_status_unsupported_layout_k,
    unknown_kind_k = uform_status_unknown_kind_k,
    io_k = uform_status_io_k,
    device_k = uform_status_device_k,
    malformed_k = uform_status_malformed_k,
    internal_k = uform_status_internal_k,
    uninitialized_k = uform_status_uninitialized_k,
    config_mismatch_k = uform_status_config_mismatch_k,
    contended_k = uform_status_contended_k,
    cancelled_k = uform_status_cancelled_k,
    unsupported_k = uform_status_unsupported_k,
    invalid_argument_k = uform_status_invalid_argument_k,
};

#pragma region Shared Core

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

/** A call's value beside its status, bound natively as <tt>auto [value, status]</tt>. */
template <typename value_type_>
struct [[nodiscard]] expected {
    value_type_ value {};
    status_t status {};
};

/** Where a call writes what @p value_type_ is made of. */
template <typename value_type_>
struct outputs {

    /** The answer, which the call writes in place. */
    value_type_ value {};

    value_type_ answer() const noexcept { return value; }
};

/** A call that outputs nothing but its status. */
template <>
struct outputs<void> {};

/** Runs @p poll with @p state later on its own loop, storing the two words and allocating
 *  nothing. */
template <typename scheduler_type_>
concept scheduler = requires(scheduler_type_& loop, void (*poll)(void*) noexcept, void* state) {
    loop.post(poll, state);
};

template <typename value_type_, scheduler scheduler_type_>
class awaiter;

/**
 *  @brief A submitted call: its outcome and its outputs, owned until the call is final, while the
 *      receiver, the inputs, the arena and the handlers it was handed stay the caller's until then.
 *
 *  It never moves, as the library writes into it, and its one owner polls @c status, blocks in
 *  @c wait or awaits @c wait on its own loop; destruction waits, so nothing the call writes into
 *  goes away under it.
 */
template <typename value_type_ = void>
class [[nodiscard]] deferred {
    uform_outcome_t outcome_ {};
    outputs<value_type_> outputs_ {};

  public:
    /** What @c wait answers: what the blocking call would have. */
    using answer_type = std::conditional_t<std::is_void_v<value_type_>, status_t, expected<value_type_>>;

    /** An empty slot, final from the start, which @c resubmit fills. */
    deferred() noexcept = default;

    /** Runs @p submit with this call's outcome and outputs, where the library reports. */
    template <typename submit_type_>
        requires std::invocable<submit_type_ const&, uform_outcome_t*, outputs<value_type_>*>
    explicit deferred(submit_type_ const& submit) noexcept {
        submit(&outcome_, &outputs_);
    }

    /** Waits only while the slot's previous call still runs, then runs @p submit in its place. */
    template <typename submit_type_>
        requires std::invocable<submit_type_ const&, uform_outcome_t*, outputs<value_type_>*>
    void resubmit(submit_type_ const& submit) noexcept {
        uform_outcome_wait(&outcome_);
        outcome_ = {};
        outputs_ = {};
        submit(&outcome_, &outputs_);
    }

    /** A call refused before it crossed, final from the start. */
    explicit deferred(status_t status) noexcept : outcome_ {static_cast<uform_status_t>(status.code), status.note} {}

    deferred(deferred const&) = delete;
    deferred& operator=(deferred const&) = delete;
    ~deferred() noexcept { uform_outcome_wait(&outcome_); }

    /** @c status_code_t::deferred_k until the call is final, then its final status, polled without
     *  blocking. */
    status_code_t status() noexcept { return static_cast<status_code_t>(uform_outcome_status(&outcome_)); }

    /** Blocks until the call is final and answers what a blocking call would have. */
    answer_type wait() noexcept {
        uform_outcome_wait(&outcome_);
        status_t const status {static_cast<status_code_t>(outcome_.status), outcome_.message};
        if constexpr (std::is_void_v<value_type_>) return status;
        else if (status.failed()) return {{}, status};
        else return {outputs_.answer()};
    }

    /** Awaits the call on @p loop, which polls its status until it is final. */
    template <scheduler scheduler_type_>
    awaiter<value_type_, scheduler_type_> wait(scheduler_type_& loop) noexcept {
        return {*this, loop};
    }
};

/** A @c deferred being awaited, living in the awaiting coroutine's frame: each poll reads the
 *  status word, and one not yet final posts itself again. */
template <typename value_type_, scheduler scheduler_type_>
class awaiter {
    deferred<value_type_>& call_;
    scheduler_type_& loop_;
    std::coroutine_handle<> caller_ {};

    static void poll_(void* state) noexcept {
        awaiter& awaited = *static_cast<awaiter*>(state);
        if (awaited.await_ready()) return awaited.caller_.resume();
        awaited.loop_.post(&poll_, state);
    }

  public:
    awaiter(deferred<value_type_>& call, scheduler_type_& loop) noexcept : call_(call), loop_(loop) {}
    bool await_ready() noexcept { return call_.status() != status_code_t::deferred_k; }
    void await_suspend(std::coroutine_handle<> caller) noexcept {
        caller_ = caller;
        loop_.post(&poll_, this);
    }
    typename deferred<value_type_>::answer_type await_resume() noexcept { return call_.wait(); }
};

/** Whether @p answer_type_ is a call the caller holds rather than a final answer. */
template <typename answer_type_>
inline constexpr bool is_deferred = false;
template <typename value_type_>
inline constexpr bool is_deferred<deferred<value_type_>> = true;

/** What a final @p answer_type_ carries beside its status, @c void for a bare @c status_t. */
template <typename answer_type_>
struct answer_value {
    using type = void;
};
template <typename value_type_>
struct answer_value<expected<value_type_>> {
    using type = value_type_;
};
template <typename answer_type_>
using answer_value_t = typename answer_value<answer_type_>::type;

/** Answers @p answer_type_ from a verb's @p body: a deferred call hands C its own outcome and
 *  outputs; a blocking call writes its outputs in place, then takes the C outcome's two words once
 *  it is final. */
template <typename answer_type_, typename body_type_>
answer_type_ answered_(body_type_ const& body) noexcept {
    if constexpr (is_deferred<answer_type_>) return answer_type_(body);
    else {
        using value_type = answer_value_t<answer_type_>;
        uform_outcome_t reported {};
        outputs<value_type> written {};
        body(&reported, &written);
        status_t const status {static_cast<status_code_t>(reported.status), reported.message};
        if constexpr (std::is_void_v<value_type>) return status;
        else if (status.failed()) return {{}, status};
        else return {written.answer()};
    }
}

/** Answers a call refused before it crossed, final from the start. */
template <typename answer_type_>
answer_type_ answered_(status_t refused) noexcept {
    if constexpr (is_deferred<answer_type_> || std::is_same_v<answer_type_, status_t>) return answer_type_(refused);
    else return {{}, refused};
}

/** Owns one handle of @p handle_type_, which @p free_ releases, moving it and never copying it. */
template <typename handle_type_, void (*free_)(handle_type_)>
class owner {
    handle_type_ handle_ = nullptr;

  public:
    owner() noexcept = default;
    owner(owner&& other) noexcept : handle_(std::exchange(other.handle_, nullptr)) {}
    owner& operator=(owner&& other) noexcept {
        std::swap(handle_, other.handle_);
        return *this;
    }
    owner(owner const&) = delete;
    owner& operator=(owner const&) = delete;
    ~owner() noexcept { free_(handle_); }

    handle_type_ handle() const noexcept { return handle_; }

    /** The output field a lifecycle verb writes the new handle into. */
    handle_type_* handle_field() noexcept { return &handle_; }
};

/** Creates an @p owner_type_ through @p verb, which writes the new handle into @p field of
 *  @p arguments. */
template <typename owner_type_, typename arguments_type_, typename handle_type_>
expected<owner_type_> opened_(void (*verb)(arguments_type_*), arguments_type_& arguments,
                              handle_type_* arguments_type_::* field) noexcept {
    expected<owner_type_> opened;
    arguments.*field = opened.value.handle_field();
    opened.status = answered_<status_t>([&](uform_outcome_t* outcome, outputs<void>*) noexcept {
        arguments.outcome = outcome;
        verb(&arguments);
    });
    return opened;
}

#pragma endregion Shared Core

/** A blocking call's answer, a bare status for void verbs, or a deferred call the caller holds. */
template <typename value_type_ = void, uform_completion_t completion_ = uform_completion_blocking_k>
using deferrable = std::conditional_t<completion_ == uform_completion_deferred_k, deferred<value_type_>,
                                      std::conditional_t<std::is_void_v<value_type_>, status_t, expected<value_type_>>>;

/** A handle owned, moved and never copied, whose verbs block on the calling thread. */
template <typename handle_type_, void (*free_)(handle_type_)>
struct owning {
    static constexpr uform_completion_t completion = uform_completion_blocking_k;
    owner<handle_type_, free_> owner_;
    handle_type_ handle() const noexcept { return owner_.handle(); }
    uform_executor_t executor() const noexcept { return nullptr; }
    handle_type_* handle_field() noexcept { return owner_.handle_field(); }
};

/** A handle the caller owns and the executor its verbs run on, completing as @p completion_
 *  says. */
template <typename handle_type_, uform_completion_t completion_>
struct caller_owned {
    static constexpr uform_completion_t completion = completion_;
    handle_type_ handle_ = nullptr;
    uform_executor_t executor_ = nullptr;
    handle_type_ handle() const noexcept { return handle_; }
    uform_executor_t executor() const noexcept { return executor_; }
};

/** Only an owning receiver opens, loads and hands out cursors. */
template <typename receiver_type_>
concept owns_handle = requires(receiver_type_& receiver) { receiver.handle_field(); };

#pragma region Scalars Rows and Tapes

/** Bit containers for the formats C++ has no arithmetic type for, so a call can still name them. */
struct bf16_t {
    std::uint16_t encoding;
};
struct f16_t {
    std::uint16_t encoding;
};
struct e5m2_t {
    std::uint8_t encoding;
};
struct e4m3_t {
    std::uint8_t encoding;
};
struct e3m2_t {
    std::uint8_t encoding;
};
struct e2m3_t {
    std::uint8_t encoding;
};
struct i4x2_t {
    std::uint8_t encoding;
};
struct u4x2_t {
    std::uint8_t encoding;
};
struct u1x8_t {
    std::uint8_t encoding;
};

/** The scalar kind a C++ element type names, or @c uform_scalar_unknown_k for none. */
template <typename element_type_>
inline constexpr uform_scalar_t scalar_of = uform_scalar_unknown_k;
template <>
inline constexpr uform_scalar_t scalar_of<double> = uform_scalar_f64_k;
template <>
inline constexpr uform_scalar_t scalar_of<float> = uform_scalar_f32_k;
template <>
inline constexpr uform_scalar_t scalar_of<bf16_t> = uform_scalar_bf16_k;
template <>
inline constexpr uform_scalar_t scalar_of<f16_t> = uform_scalar_f16_k;
template <>
inline constexpr uform_scalar_t scalar_of<e5m2_t> = uform_scalar_e5m2_k;
template <>
inline constexpr uform_scalar_t scalar_of<e4m3_t> = uform_scalar_e4m3_k;
template <>
inline constexpr uform_scalar_t scalar_of<e3m2_t> = uform_scalar_e3m2_k;
template <>
inline constexpr uform_scalar_t scalar_of<e2m3_t> = uform_scalar_e2m3_k;
template <>
inline constexpr uform_scalar_t scalar_of<std::int8_t> = uform_scalar_i8_k;
template <>
inline constexpr uform_scalar_t scalar_of<i4x2_t> = uform_scalar_i4x2_k;
template <>
inline constexpr uform_scalar_t scalar_of<std::int32_t> = uform_scalar_i32_k;
template <>
inline constexpr uform_scalar_t scalar_of<std::uint8_t> = uform_scalar_u8_k;
template <>
inline constexpr uform_scalar_t scalar_of<char> = uform_scalar_u8_k;
template <>
inline constexpr uform_scalar_t scalar_of<u4x2_t> = uform_scalar_u4x2_k;
template <>
inline constexpr uform_scalar_t scalar_of<u1x8_t> = uform_scalar_u1x8_k;

/** An element type a call can state as the scalar of the tape or rows it passes or reads. */
template <typename element_type_>
concept element = scalar_of<element_type_> != uform_scalar_unknown_k;

/** Values one element packs: eight for bit sets, two for nibbles, one otherwise. */
template <element element_type_>
inline constexpr std::size_t values_per_element = 1;
template <>
inline constexpr std::size_t values_per_element<i4x2_t> = 2;
template <>
inline constexpr std::size_t values_per_element<u4x2_t> = 2;
template <>
inline constexpr std::size_t values_per_element<u1x8_t> = 8;

/** Views @p text as the unterminated run of bytes every path, buffer and name arrives in. */
inline uform_bytes_view_t bytes_view(std::string_view text) noexcept {
    return {.data = reinterpret_cast<std::uint8_t const*>(text.data()), .length = text.size()};
}

/** Views @p bytes as the caller's run every checkpoint buffer arrives in. */
inline uform_bytes_view_t bytes_view(std::span<std::uint8_t const> bytes) noexcept {
    return {.data = bytes.data(), .length = bytes.size()};
}

/**
 *  @brief A ragged batch of @p element_type_ laid out as Apache Arrow lays one out: elements end to
 *      end, and one more offset than entries, entry @c i spanning two consecutive offsets.
 *
 *  The caller owns both spans: as an input, it keeps them alive until the call is final; as an
 *  output, it reads the call's arena until the arena's next call.
 */
template <element element_type_>
struct tape_view {
    std::span<element_type_ const> elements {};
    std::span<std::size_t const> offsets {};

    /** Entries along the batch axis, one fewer than the offsets. */
    std::size_t count() const noexcept { return offsets.empty() ? 0 : offsets.size() - 1; }

    /** The elements of the entry at @p position, below @c count. */
    std::span<element_type_ const> entry(std::size_t position) const noexcept {
        return elements.subspan(offsets[position], offsets[position + 1] - offsets[position]);
    }

    /** Refuses what only the binding sees: no offsets, or a last one past the elements. */
    status_t check() const noexcept {
        if (offsets.empty()) return {status_code_t::invalid_argument_k, "A tape holds one more offset than entries"};
        if (offsets.back() > elements.size())
            return {status_code_t::shape_mismatch_k, "The tape's last offset runs past its elements"};
        return {};
    }

    uform_tape_view_t view() const noexcept { return {elements.data(), offsets.data(), scalar_of<element_type_>}; }
};

/**
 *  @brief Gathers separate items of @p element_type_ into one tape laid out in the caller's
 *      @p offsets and @p elements, for callers whose items are not laid end to end already.
 *
 *  Gathering copies every element once, which a caller already holding a tape skips by passing its
 *  @c tape_view directly. The answer views both spans, so they outlive every call it is passed to.
 */
template <element element_type_, std::ranges::sized_range items_type_>
expected<tape_view<element_type_>> gathered(items_type_ const& items, std::span<std::size_t> offsets,
                                            std::span<element_type_> elements) noexcept {
    std::size_t const count = std::ranges::size(items);
    if (offsets.size() <= count)
        return {{}, status_t {status_code_t::invalid_argument_k, "A tape holds one more offset than entries"}};
    offsets[0] = 0;
    std::size_t position = 0;
    for (std::span<element_type_ const> const item : items) {
        std::size_t const begin = offsets[position];
        if (item.size() > elements.size() - begin)
            return {{}, status_t {status_code_t::invalid_argument_k, "The items outgrow the element buffer"}};
        if (!item.empty()) std::memcpy(elements.data() + begin, item.data(), item.size_bytes());
        offsets[++position] = begin + item.size();
    }
    return {tape_view<element_type_> {elements.first(offsets[count]), offsets.first(count + 1)}};
}

/** Rows a call wrote packed into its arena, one per item, readable until the arena's next call;
 *  host code may read them only from a RAM arena. */
struct written_rows_t {

    /** The rows as the call wrote them. */
    uform_rows_view_t view {};

    /** Rows written, one per item along the call's batch axis; zero for a surface the call did not
     *  keep. */
    std::size_t written = 0;

    /** The row at @p position, below @c written, or empty when the rows hold another scalar. */
    template <element element_type_>
    std::span<element_type_ const> row(std::size_t position) const noexcept {
        if (view.scalar != scalar_of<element_type_>) return {};
        std::byte const* const first = static_cast<std::byte const*>(view.data) + position * view.stride;
        return {reinterpret_cast<element_type_ const*>(first), view.columns / values_per_element<element_type_>};
    }
};

/** A ragged run of rows a call wrote into its arena, item @c i spanning rows @c offsets()[i] up to
 *  @c offsets()[i+1]; readable until the arena's next call. */
struct written_rows_tape_t {

    /** The rows and offsets as the call wrote them. */
    uform_rows_tape_view_t view {};

    /** Items written along the call's batch axis; zero for a surface the call did not keep. */
    std::size_t written = 0;

    /** The @c written + 1 offsets into the rows, in rows. */
    std::span<std::size_t const> offsets() const noexcept {
        if (!view.offsets) return {};
        return {view.offsets, written + 1};
    }

    /** The rows of the item at @p position, below @c written. */
    written_rows_t item(std::size_t position) const noexcept {
        std::size_t const begin = view.offsets[position], end = view.offsets[position + 1];
        uform_rows_view_t rows = view.rows;
        rows.data = static_cast<std::byte const*>(view.rows.data) + begin * view.rows.stride;
        return {rows, end - begin};
    }
};

#pragma endregion Scalars Rows and Tapes

#pragma region Handlers Executors and Arenas

/** A progress handler: hears units completed of a total and answers whether the call goes on. */
template <typename handler_type_>
concept progress_handler =
    std::is_nothrow_invocable_r_v<uform_continuation_t, handler_type_&, std::size_t, std::size_t>;

/** A progress handler the caller keeps for one call, passed as its own state; empty hears
 *  nothing. */
class progress_t {
    uform_progress_t raw_ {};

    template <typename handler_type_>
    static uform_continuation_t trampoline_(std::size_t completed, std::size_t total, void* state) noexcept {
        return (*static_cast<handler_type_*>(state))(completed, total);
    }

  public:
    progress_t() noexcept = default;

    /** Hears @p handler after each of @p steps slices of a batch verb; zero derives. */
    template <progress_handler handler_type_>
    progress_t(handler_type_& handler, std::size_t steps = 0) noexcept
        : raw_ {&trampoline_<handler_type_>, &handler, steps} {}

    uform_progress_t raw() const noexcept { return raw_; }
};

/** Owns one executor: the host threads or device streams every call runs on, apart from any model;
 *  an empty one runs every call on the caller's own thread. It outlives every arena it made. */
class executor_t {
    owner<uform_executor_t, uform_executor_free> owner_;

  public:
    uform_executor_t handle() const noexcept { return owner_.handle(); }
    uform_executor_t* handle_field() noexcept { return owner_.handle_field(); }

    /** Creates an executor of host threads for RAM-resident models, near the one memory domain
     *  @p arguments names, or interleaved across the machine for none or several. */
    static expected<executor_t> for_ram(uform_executor_for_ram_t arguments = {}) noexcept {
        return opened_<executor_t>(uform_executor_for_ram, arguments, &uform_executor_for_ram_t::executor);
    }

    /** Opens one submission stream on every device @p arguments names; caller streams answer
     *  @c status_code_t::unsupported_k . */
    static expected<executor_t> for_vram(uform_executor_for_vram_t arguments = {}) noexcept {
        return opened_<executor_t>(uform_executor_for_vram, arguments, &uform_executor_for_vram_t::executor);
    }

    /** Waits until no call holds the executor, then sets its threads' idle @p mood. */
    status_t drain(uform_mood_t mood = uform_mood_parked_k) noexcept {
        return answered_<status_t>([&](uform_outcome_t* outcome, outputs<void>*) noexcept {
            uform_executor_drain_t arguments {.executor = owner_.handle(), .outcome = outcome, .mood = mood};
            uform_executor_drain(&arguments);
        });
    }
};

/** Owns one arena; every call writing outputs into it rewinds it first, ending the previous
 *  call's views. It must not outlive the executor that made it, so declare that executor first;
 *  one made with no executor lives until it is destroyed. */
class arena_t {
    owner<uform_arena_t, uform_arena_free> owner_;

  public:
    uform_arena_t handle() const noexcept { return owner_.handle(); }
    uform_arena_t* handle_field() noexcept { return owner_.handle_field(); }

    /** Creates an empty arena where the executor @p arguments names computes. */
    static expected<arena_t> make(uform_arena_init_t arguments = {}) noexcept {
        return opened_<arena_t>(uform_arena_init, arguments, &uform_arena_init_t::arena);
    }

    /** Reports where the arena's outputs live, which decides whether host code may read them. */
    expected<uform_arena_description_t> describe() const noexcept {
        return answered_<expected<uform_arena_description_t>>(
            [&](uform_outcome_t* outcome, outputs<uform_arena_description_t>* written) noexcept {
                uform_arena_describe_t arguments {.arena = owner_.handle(),
                                                  .outcome = outcome,
                                                  .description = &written->value};
                uform_arena_describe(&arguments);
            });
    }
};

/** Fills @p devices with the GPUs this process can drive, in ordinal order, answering how many
 *  there are, which may exceed what @p devices holds. */
inline expected<std::size_t> gpu_devices(std::span<uform_gpu_device_t> devices) noexcept {
    return answered_<expected<std::size_t>>([&](uform_outcome_t* outcome, outputs<std::size_t>* written) noexcept {
        uform_gpu_devices_t arguments {.outcome = outcome,
                                       .devices = devices.data(),
                                       .devices_capacity = devices.size(),
                                       .devices_count = &written->value};
        uform_gpu_devices(&arguments);
    });
}

/** The linked library's version, a release-candidate suffix included. */
inline std::string_view version_name() noexcept { return uform_version_name(); }

#pragma endregion Handlers Executors and Arenas

} // namespace unum::uform
