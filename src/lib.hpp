/**
 *  @file src/lib.hpp
 *  @author Ash Vardanian
 *  @date October 17, 2026
 *  @brief What every verb of the C interface shares: the outcome and progress crossings, the executor and arena
 *      handles, and the deferred calls every verb enters through.
 *
 *  Each verb file includes this, so one verb reads the same in every header it implements. The registry of deferred
 *  calls and the functions touching the handles' members are defined once, in @c types.cpp, so a wait or a free
 *  finds the calls of every verb naming its handle.
 *
 *  Every verb opens the same way:
 *
 *  @code{.cpp}
 *  UFORM_EXPORT void uform_text_encode(uform_text_encode_t *arguments) {
 *      if (submitted<uform_text_encode>(arguments)) return;
 *      call_outcome_t outcome(arguments->outcome, arguments->arena);
 *      uform_arena_t const arena = claim_arena(arguments->arena, arguments->executor, {inputs}, outcome);
 *      if (!arena) return;
 *      // ...
 *  }
 *  @endcode
 */
#pragma once

#include <cstddef> // `std::size_t`

#include <atomic>           // `std::atomic`, `std::atomic_ref` - the arena's holder and the outcome's status word
#include <initializer_list> // `std::initializer_list` - the inputs a call checks against its arena
#include <limits>           // `std::numeric_limits` - the overflow guard of a typed allocation
#include <mutex>            // `std::lock_guard` over the ticks' lock
#include <type_traits>      // `std::is_trivially_copyable_v`
#include <variant>          // `std::variant` - the executor's kind, the arena's residency, a deferred call

#include <forkunion.hpp>          // vendor-boundary: `fu::flat_pool_t`, `fu::spin_mutex_t`, `fu::machine_topology_t`
#include <numkong/capabilities.h> // `nk_capability_t`, `nk_stream_t`
#include <numkong/memory.h>       // `nk_allocator_t`, `nk_allocator_init_heap`
#include <numkong/memory.hpp>     // `nk::stream_t` - the stream an executor owns

#include "core/array.hpp"      // `array`, `per_gpu_array`
#include "core/types.hpp"      // `status_t`, the integer aliases, `sz::string_view_t`
#include "inference/tower.hpp" // `tower_t`, `geometry_t`, `precision_t` - what a model owns
#include "uform/uform.h"       // every argument structure

namespace unum::uform {

namespace fu = ashvardanian::forkunion;
namespace nk = ashvardanian::numkong;
namespace sz = ashvardanian::stringzilla;

using core::array;
using core::b8_t;
using core::i32_t;
using core::per_gpu_array;
using core::u64_t;
using core::u8_t;

/** Reports a progress handler hears at most when its call names zero steps. */
inline constexpr std::size_t default_progress_steps_k = 10;

/** Calls a deferred table holds when its executor's constructor names no limit. */
inline constexpr std::size_t deferred_limit_default_k = 64;

/** One core in this many stays idle when an executor derives its thread count, so threads are 7/8 of the cores. */
inline constexpr std::size_t idle_cores_denominator_k = 8;

/** The ticket no call outlives, so a blocking call runs everything its executor holds. */
inline constexpr u64_t every_ticket_k = std::numeric_limits<u64_t>::max();

/** A borrowed byte run of the C interface, a path or a spelling, as the engine's string view. */
[[nodiscard]] inline sz::string_view_t text_of(uform_bytes_view_t bytes) noexcept {
    return {static_cast<char const *>(static_cast<void const *>(bytes.data)), bytes.length};
}

#pragma region Outcomes

/** The C status a failure of the engine's own vocabulary crosses into. */
[[nodiscard]] inline uform_status_t status_of(core::status_t status) noexcept {
    switch (status) {
    case core::status_t::success_k: return uform_status_success_k;
    case core::status_t::bad_alloc_k: return uform_status_out_of_memory_k;
    case core::status_t::invalid_input_k: return uform_status_invalid_argument_k;
    case core::status_t::shape_unsupported_k: return uform_status_unsupported_k;
    case core::status_t::cuda_error_k: return uform_status_device_k;
    case core::status_t::io_error_k: return uform_status_io_k;
    case core::status_t::other_k: return uform_status_internal_k;
    }
    return uform_status_internal_k;
}

/** The C status a NumKong failure crosses into; whatever a device answers beyond shapes and memory is the device's. */
[[nodiscard]] inline uform_status_t status_of(nk_status_t status) noexcept {
    switch (status) {
    case nk_success_k: return uform_status_success_k;
    case nk_bad_alloc_k: return uform_status_out_of_memory_k;
    case nk_unexpected_dimensions_k: return uform_status_shape_mismatch_k;
    case nk_misaligned_k: return uform_status_unsupported_layout_k;
    case nk_missing_kernel_k: return uform_status_unsupported_k;
    default: return uform_status_device_k;
    }
}

/** Marks @p outcome as running; no waiter watches for this store, so it is relaxed. */
inline void propagate_pending(uform_outcome_t *outcome) noexcept {
    outcome->message = nullptr;
    std::atomic_ref<uform_status_t> {outcome->status}.store(uform_status_deferred_k, std::memory_order_relaxed);
}

/** The one final write to @p outcome: the message, then the status every waiter acquires, and a wake. The call
 *  touches nothing it was passed after this, and the message is null exactly on success. */
inline bool propagate_error(uform_status_t status, uform_outcome_t *outcome,
                            char const *message) noexcept { // vendor-boundary: the C outcome's static message
    bool const succeeded = status == uform_status_success_k;
    outcome->message = succeeded ? nullptr : message;
    std::atomic_ref<uform_status_t> final_status {outcome->status};
    final_status.store(status, std::memory_order_release);
    final_status.notify_all();
    return succeeded;
}

#pragma endregion Outcomes

#pragma region Progress

/** Whether the call a progress handler watches carries on or unwinds. */
enum class continuation_t : u8_t {
    continue_k,
    cancel_k,
};

/** A half-open range along a call's batch axis that the engine takes whole. */
struct slice_t {
    std::size_t begin = 0;
    std::size_t end = 0;
};

/**
 *  @brief One call's reports: the C handler, serialized for the executor's threads, which the contract promises it
 *      never sees concurrently, thinned to the caller's steps, and the slices a batch verb hands its engine whole.
 *
 *  Without a handler a batch is one slice and nothing is reported; with one it splits into @c progress.steps nearly
 *  equal slices, ten when zero. A verb counting its own units, a load's tensors, forwards each tick and only the
 *  ones crossing into the next step reach the handler.
 */
class progress_ticks_t {
  public:
    progress_ticks_t(uform_progress_t progress, std::size_t total) noexcept : progress_(progress), total_(total) {}
    progress_ticks_t(progress_ticks_t const &) = delete;
    progress_ticks_t &operator=(progress_ticks_t const &) = delete;

    /** Walks the slices of a batch in order; slice @c i starts at @c ⌈i·total/slices⌉ , so lengths differ by one
     *  at most. */
    class iterator {
      public:
        iterator(std::size_t index, std::size_t slices, std::size_t total) noexcept
            : index_(index), slices_(slices), total_(total) {}
        slice_t operator*() const noexcept { return {boundary_(index_), boundary_(index_ + 1)}; }
        iterator &operator++() noexcept { return ++index_, *this; }
        bool operator!=(iterator const &other) const noexcept { return index_ != other.index_; }

      private:
        [[nodiscard]] std::size_t boundary_(std::size_t index) const noexcept {
            return core::divide_round_up(index * total_, slices_);
        }

        std::size_t index_;
        std::size_t slices_;
        std::size_t total_;
    };

    /** Slices the batch splits into: one per step with a handler, the whole batch without, none when empty. */
    [[nodiscard]] std::size_t slices() const noexcept {
        if (!total_) return 0;
        return progress_.callback ? steps_(total_) : 1;
    }

    [[nodiscard]] iterator begin() const noexcept { return {0, slices(), total_}; }
    [[nodiscard]] iterator end() const noexcept { return {slices(), slices(), total_}; }

    /** Reports @p completed of the batch's units, reaching the handler only when it closes a new step. */
    continuation_t operator()(std::size_t completed) noexcept { return (*this)(completed, total_); }

    /** The sibling for verbs that know their own population, like a load counting its tensors. A slice end
     *  @c ⌈i·population/steps⌉ closes exactly step @c i . */
    continuation_t operator()(std::size_t completed, std::size_t population) noexcept {
        if (!progress_.callback) return continuation_t::continue_k;
        std::lock_guard<fu::spin_mutex_t> lock(mutex_);
        std::size_t const steps = steps_(population);
        std::size_t const step = completed >= population ? steps : completed * steps / population;
        if (step <= closed_) return continuation_t::continue_k;
        closed_ = step;
        return progress_.callback(completed, population, progress_.state) == uform_cancel_k
                   ? continuation_t::cancel_k
                   : continuation_t::continue_k;
    }

  private:

    /** The caller's steps, ten when zero, and never more than @p population units can close. */
    [[nodiscard]] std::size_t steps_(std::size_t population) const noexcept {
        std::size_t const asked = progress_.steps ? progress_.steps : default_progress_steps_k;
        return asked < population ? asked : (population ? population : 1);
    }

    uform_progress_t progress_;
    std::size_t total_;

    /** The last step a forwarded tick closed, guarded by @ref mutex_ like the handler itself. */
    std::size_t closed_ = 0;
    fu::spin_mutex_t mutex_;
};

#pragma endregion Progress

#pragma region Memory

/** A run of bytes one arena owns. */
struct arena_chunk_t {
    b8_t *data = nullptr;
    std::size_t capacity = 0;
};

/**
 *  @brief Memory that one call at a time rewinds and refills through one NumKong allocation policy: chunks that
 *      double and never move, so a steady loop allocates once and every view a call hands out stays put.
 *
 *  Chunks are returned through the policy on the stream it was given, so a device arena dies before its stream.
 *  Movable so a per-device table can hold it, never move-assigned, since that would leak the destination's chunks.
 */
class chunked_arena_t {
  public:
    chunked_arena_t(nk_allocator_t policy, nk_stream_t stream, std::size_t limit) noexcept
        : policy_(policy), stream_(stream), limit_(limit) {}
    chunked_arena_t(chunked_arena_t &&) noexcept = default;
    chunked_arena_t &operator=(chunked_arena_t &&) = delete;
    ~chunked_arena_t() noexcept {
        for (arena_chunk_t const &chunk : chunks_) policy_.free(chunk.data, chunk.capacity, policy_.handle, stream_);
    }

    /** The stream its chunks are allocated and freed on; null for host pages. */
    [[nodiscard]] nk_stream_t stream() const noexcept { return stream_; }

    /** Forgets every allocation, keeping the chunks, which invalidates every view the previous call handed out. */
    void rewind() noexcept { current_ = 0, used_ = 0; }

    /** Whether @p address lies in memory this arena handed out, which its next rewind reuses. */
    [[nodiscard]] bool contains(void const *address) const noexcept {
        for (arena_chunk_t const &chunk : chunks_)
            if (address >= chunk.data && address < chunk.data + chunk.capacity) return true;
        return false;
    }

    /** Hands out @p bytes aligned for any kernel, growing by a chunk as large as everything held so far when full,
     *  or null when the policy or the limit refuses. */
    [[nodiscard]] void *allocate(std::size_t bytes) noexcept {
        std::size_t const aligned = core::divide_round_up(bytes, nk_default_alignment_k) * nk_default_alignment_k;
        for (; current_ != chunks_.size(); ++current_, used_ = 0)
            if (chunks_[current_].capacity - used_ >= aligned) {
                b8_t *const landed = chunks_[current_].data + used_;
                used_ += aligned;
                return landed;
            }
        std::size_t held = 0;
        for (arena_chunk_t const &chunk : chunks_) held += chunk.capacity;
        std::size_t const capacity = held > aligned ? held : aligned;
        if (limit_ && held + capacity > limit_) return nullptr;
        if (!policy_.allocate) return nullptr;
        arena_chunk_t const chunk {static_cast<b8_t *>(policy_.allocate(capacity, policy_.handle, stream_)), capacity};
        if (!chunk.data) return nullptr;
        if (chunks_.push_back(chunk) != core::status_t::success_k)
            return policy_.free(chunk.data, chunk.capacity, policy_.handle, stream_), nullptr;
        current_ = chunks_.size() - 1, used_ = aligned;
        return chunk.data;
    }

    /** Hands out @p count values of @p value_type_, or null when they cannot be allocated. */
    template <typename value_type_>
    [[nodiscard]] value_type_ *allocate(std::size_t count) noexcept {
        static_assert(std::is_trivially_copyable_v<value_type_>, "Arenas hold plain outputs only");
        if (count > std::numeric_limits<std::size_t>::max() / sizeof(value_type_)) return nullptr;
        return static_cast<value_type_ *>(allocate(count * sizeof(value_type_)));
    }

  private:
    nk_allocator_t policy_;
    nk_stream_t stream_;

    /** Bytes the chunks may span together, or zero for no ceiling. */
    std::size_t limit_;
    array<arena_chunk_t> chunks_;

    /** The chunk being filled, and the bytes of it already handed out since the last rewind. */
    std::size_t current_ = 0;
    std::size_t used_ = 0;
};

/** Host pages, which serve calls on the calling thread or any RAM executor and may outlive the executor that made
 *  them. */
struct host_arena_t : chunked_arena_t {
    host_arena_t() noexcept : chunked_arena_t(heap_policy(), nullptr, 0) {}

  private:
    [[nodiscard]] static nk_allocator_t heap_policy() noexcept {
        nk_allocator_t policy {};
        [[maybe_unused]] nk_status_t const status = nk_allocator_init_heap(&policy, nk_default_alignment_k);
        return policy;
    }
};

/** The memory of one GPU, grown through NumKong's device allocator on one stream of the executor that made it. */
struct device_arena_t : chunked_arena_t {
    using chunked_arena_t::chunked_arena_t;
};

#pragma endregion Memory

#pragma region Executors

/** A device stream an executor opened and closes, its vendor named by the device it was opened on. */
using owned_stream_t = nk::stream_t;

/** The host threads of a RAM executor, spawned and owned, and where its models place their bytes. */
struct ram_executor_t {
    fu::flat_pool_t pool;

    /** What the threads run, as @c nk_cpu_capabilities_enabled reports. */
    nk_capability_t capabilities = 0;

    /** The NUMA nodes the executor's models place their bytes in. */
    array<i32_t> memory_domains;
};

/** One GPU of a VRAM executor: its stream, and the scratch its calls rewind. */
struct vram_device_t {

    /** The vendor's own ordinal, as @c cudaSetDevice or @c hipSetDevice takes it. */
    i32_t ordinal = 0;

    /** What the device runs, whose group names the vendor every NumKong call dispatches on. */
    nk_capability_t capabilities = 0;

    /** Declared before the scratch, so the scratch returns its chunks while the stream still lives. */
    owned_stream_t stream;

    /** Activations, grown to a high-water mark under the executor's @c scratch_limit. */
    device_arena_t scratch;

    /** Pinned host pages both the host and the device address, where a call packs what the device reads. */
    chunked_arena_t transfers;
};

/** The devices of a VRAM executor, and the host threads that decode what the devices cannot. */
struct vram_executor_t {
    per_gpu_array<vram_device_t> devices;
    fu::flat_pool_t decoders;

    /** Host pages for what only the decoders read: the plans and every thread's decode window. */
    host_arena_t staging;
};

#pragma endregion Executors

#pragma region Models

/** A model's one tower in host memory, packed for the CPU it was loaded on, serving any RAM executor. */
struct ram_model_t {
    nk_capability_t capabilities = 0;
    inference::tower_t tower;
};

/** One copy of a model's tower on one GPU, matched to an executor's device by ordinal and capabilities. */
struct vram_replica_t {
    i32_t ordinal = 0;
    nk_capability_t capabilities = 0;
    inference::tower_t tower;
};

/** A model replicated on every GPU its load named. */
struct vram_model_t {
    per_gpu_array<vram_replica_t> replicas;
};

#pragma endregion Models

#pragma region Deferred Calls

/** Every argument structure that names an executor, so a deferred submission keeps its call by value; whatever the
 *  copy points to stays the caller's until final. */
using deferred_arguments_t = std::variant<                                                        //
    uform_model_load_ram_t, uform_model_load_ram_buffer_t,                                        //
    uform_model_load_vram_t, uform_model_load_vram_buffer_t, uform_model_describe_t,              //
    uform_text_encode_t, uform_image_encode_t, uform_image_encode_pixels_t,                       //
    uform_video_encode_t, uform_audio_encode_t, uform_audio_encode_samples_t,                     //
    uform_mesh_bootstrap_t, uform_trainer_init_vram_t, uform_trainer_step_t, uform_trainer_save_t //
    >;

/** Where one entry of a deferred table stands. */
enum class deferred_stage_t : u8_t {

    /** Nothing in it; a submission may take it. */
    free_k,

    /** A submitted call waits in it for its wait, a drain, a blocking call or a free. */
    held_k,

    /** Some thread took the call and runs it. */
    running_k,
};

/** One entry of an executor's deferred table: the copied call, the verb that runs it, and its place in line. */
struct deferred_call_t {
    deferred_arguments_t arguments;
    void (*verb)(void *) = nullptr;

    /** Submission order, counted from one across every executor. */
    u64_t ticket = 0;
    deferred_stage_t stage = deferred_stage_t::free_k;
};

/** Every listed executor and the lock every deferred table is read under, so a wait or a free finds the calls
 *  naming its handle without asking each executor. */
struct deferred_registry_t {
    fu::spin_mutex_t guard;

    /** Bumped under @ref guard whenever a taken call becomes final, and waited on outside it. */
    std::atomic<u64_t> finished {0};

    /** The last ticket handed out. */
    u64_t tickets = 0;

    /** The newest listed executor. */
    uform_executor_handle_t *executors = nullptr;
};

/** The one registry every verb's calls list in. */
extern deferred_registry_t deferred_registry;

/** The held call whose arguments this thread passes back into their verb, which then runs them in place rather than
 *  holding them again. */
extern thread_local void const *resuming_call;

/** Lets go of @p arena when the call whose published outcome is @p call holds it. */
void release_arena(uform_arena_t arena, uform_outcome_t const *call) noexcept;

/** Runs every call @p executor holds with a ticket up to @p last on this thread, oldest first. */
void run_held(uform_executor_t executor, u64_t last) noexcept;

/** Returns once no deferred call reads or writes @p handle, running the held ones on this thread together with every
 *  call held before them on the same executor - what every wait and every free does first. */
void finish_calls_using(void const *handle) noexcept;

/** Copies @p arguments into a free entry of @p executor 's table, holding @p arena for the call, or finishes
 *  @p outcome contended when the arena serves another call or the table is full. */
void hold_deferred(uform_executor_t executor, deferred_arguments_t const &arguments, void (*verb)(void *),
                   uform_arena_t arena, uform_outcome_t *outcome) noexcept;

/**
 *  @brief Every verb's entry. A deferred call on an executor is copied into the executor's table, holding its arena,
 *      and answers true; a blocking one first runs the calls the executor holds and answers false.
 *
 *  A call the runner of a held call passes back answers false at once, so the verb runs it in place, and so does a
 *  call with no executor, which is final when it returns whatever its completion says.
 */
template <auto verb_, typename arguments_type_>
bool submitted(arguments_type_ *arguments) noexcept {
    if (resuming_call == arguments) return resuming_call = nullptr, false;
    uform_executor_t const executor = arguments->executor;
    if (!executor) return false;
    if (arguments->completion != uform_completion_deferred_k) return run_held(executor, every_ticket_k), false;
    uform_arena_t arena = nullptr;
    if constexpr (requires { arguments->arena; }) arena = arguments->arena;
    hold_deferred(
        executor, deferred_arguments_t {std::in_place_type<arguments_type_>, *arguments},
        [](void *erased) noexcept { verb_(static_cast<arguments_type_ *>(erased)); }, arena, arguments->outcome);
    return true;
}

/**
 *  @brief A verb's outcome while it runs: the caller's reads deferred from the start, every step records into a
 *      private one, and the end lets go of the arena before the one final store.
 *
 *  Declared first in a verb, it unwinds last, so a waiter that sees the final status may reuse the arena at once. A
 *  path that forgot to report finishes @c uform_status_internal_k rather than hanging its waiters.
 */
class call_outcome_t {
  public:
    explicit call_outcome_t(uform_outcome_t *published, uform_arena_t arena = nullptr) noexcept
        : published_(published), arena_(arena) {
        propagate_pending(published_);
    }
    ~call_outcome_t() noexcept {
        release_arena(arena_, published_);
        propagate_error(recorded_.status, published_, recorded_.message);
    }
    call_outcome_t(call_outcome_t const &) = delete;
    call_outcome_t &operator=(call_outcome_t const &) = delete;

    /** The private record every step reports into. */
    operator uform_outcome_t *() noexcept { return &recorded_; }

    /** The caller's outcome, which identifies the call holding an arena. */
    [[nodiscard]] uform_outcome_t const *published() const noexcept { return published_; }

  private:
    uform_outcome_t *published_;
    uform_arena_t arena_;
    uform_outcome_t recorded_ {uform_status_internal_k, "The call finished without reporting a status"};
};

/**
 *  @brief Holds @p arena for the call @p outcome tracks and rewinds it, freeing the previous call's outputs.
 *  @return The arena, or null once @p outcome records why not.
 *
 *  Refuses a null arena, host pages named by a VRAM executor, device memory named by another executor than the one
 *  that made it, an arena another call holds, and an input of @p inputs still inside it.
 */
uform_arena_t claim_arena(uform_arena_t arena, uform_executor_t executor, std::initializer_list<void const *> inputs,
                          call_outcome_t &outcome) noexcept;

#pragma endregion Deferred Calls

} // namespace unum::uform

namespace uf = unum::uform;

/** The state behind @c uform_executor_t: its deferred table and one residency's executor; the variant is the kind
 *  tag, so no second word can disagree. */
struct uform_executor_handle_t {

    /** One entry per call it tracks at once, sized at creation, so a submission never allocates. */
    uf::array<uf::deferred_call_t> deferred;

    /** The next listed executor, so a wait or a free finds the calls naming its handle. */
    uform_executor_handle_t *next_listed = nullptr;

    std::variant<uf::ram_executor_t, uf::vram_executor_t> executor;
};

/** What @c uform_arena_t owns: memory that one call at a time rewinds and refills, as host pages or on one device,
 *  and the call holding it. */
struct uform_arena_handle_t {

    /** Fixed at @c uform_arena_init by the executor that made it. */
    std::variant<uf::host_arena_t, uf::device_arena_t> memory;

    /** The executor whose first stream the device memory rides, the only one whose calls it serves; null for host
     *  pages, which serve any call computing on the host. */
    uform_executor_t executor = nullptr;

    /** The published outcome of the call holding the arena, from its claim or deferred submission until just before
     *  its final store; atomic, since a held call finishes on whichever thread runs it. */
    std::atomic<uform_outcome_t const *> holder {nullptr};
};

/** What @c uform_model_t owns: the geometry its checkpoint carried, the precision its blocks run in, and its packed
 *  towers; the variant is the residency, so the checkpoint itself is closed once the load completes. */
struct uform_model_handle_t {
    uf::inference::geometry_t geometry;
    uf::inference::precision_t precision = uf::inference::precision_t::bf16_k;

    /** Bytes every tower holds together, by the extents and dtypes of what each packs. */
    std::size_t memory_usage = 0;

    std::variant<uf::ram_model_t, uf::vram_model_t> residency;
};
