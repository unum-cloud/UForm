/**
 *  @file src_stub/types.cpp
 *  @author Ash Vardanian
 *  @date October 17, 2026
 *  @brief The vocabulary verbs that belong to no modality: outcomes, the running calls, executors,
 *      arenas, names, version and capabilities.
 *
 *  A stand-in dispatches to no kernels, so it reports @c serial wherever a binding asks for
 *  capabilities. Only the RAM executor constructor succeeds: VRAM needs a GPU backend this library
 *  never carries, and a borrowed pool is not wired yet, as in the engine.
 */
#include <algorithm> // `std::find`, `std::any_of`
#include <mutex>     // `std::mutex`, `std::lock_guard`, `std::unique_lock`
#include <variant>   // `std::visit`
#include <vector>    // `std::vector`

#include "lib.hpp"

namespace uf = unum::uform;
using namespace unum::uform;

/** Every call running on some caller's thread, so a wait or a free finds the calls naming its
 *  handle without asking each executor. */
struct calls_t {
    std::mutex guard;

    /** Bumped under @ref guard whenever a running call becomes final, and waited on outside it. */
    std::uint64_t finished = 0;

    /** Calls running on some caller's thread right now, each owned by the frame that runs it. */
    std::vector<call_t const *> running;
};

static calls_t calls;

void uf::release_arena(uform_arena_t arena, uform_outcome_t const *call) {
    if (!arena) return;
    std::lock_guard<std::mutex> lock(calls.guard);
    if (arena->holder == call) arena->holder = nullptr;
}

bool uf::hold_arena(uform_arena_t arena, uform_outcome_t const *call) {
    if (!arena) return true;
    std::lock_guard<std::mutex> lock(calls.guard);
    if (arena->holder && arena->holder != call) return false;
    arena->holder = call;
    return true;
}

/** Runs @p call, which the caller already listed as running, then delists it and
 *  wakes every waiter. */
static void run_listed(call_t &call) {
    std::visit([&](auto &arguments) { call.verb(&arguments); }, call.arguments);
    std::lock_guard<std::mutex> lock(calls.guard);
    calls.running.erase(std::find(calls.running.begin(), calls.running.end(), &call));
    serial_atomic_ref<std::uint64_t> {calls.finished}.fetch_add(1, std::memory_order_release);
}

void uf::run_holding(call_t call, uform_outcome_t *outcome) {
    {
        std::lock_guard<std::mutex> lock(calls.guard);
        if (std::any_of(calls.running.begin(), calls.running.end(),
                        [&](call_t const *running) { return running->held_executor == call.held_executor; }))
            return propagate_error(uform_status_contended_k, outcome, "Another call holds the executor"), void();
        calls.running.push_back(&call);
    }
    run_listed(call);
}

void uf::wait_for_calls_using(void const *handle) {
    if (!handle) return;
    auto const names_handle = [=](call_t const &call) {
        return std::find(call.handles.begin(), call.handles.end(), handle) != call.handles.end();
    };
    std::unique_lock<std::mutex> lock(calls.guard);
    for (;;) {
        if (std::any_of(calls.running.begin(), calls.running.end(),
                        [&](call_t const *running) { return names_handle(*running); })) {
            serial_atomic_ref<std::uint64_t> const finished {calls.finished};
            std::uint64_t const seen = finished.load(std::memory_order_relaxed);
            lock.unlock(); // the running call delists itself under this lock
            while (finished.load(std::memory_order_acquire) == seen)
                serial_waiter_t {}(&calls.finished, seen, std::size_t {0}, fu::wait_capped_k);
            lock.lock();
        }
        else return;
    }
}

extern "C" {

void uform_version(int *major, int *minor, int *patch) {
    if (major) *major = UFORM_VERSION_MAJOR;
    if (minor) *minor = UFORM_VERSION_MINOR;
    if (patch) *patch = UFORM_VERSION_PATCH;
}

char const *uform_version_name(void) { return "4.0.0-rc.1"; }
char const *uform_capabilities_compiled(void) { return "serial"; }
char const *uform_capabilities_available(void) { return "serial"; }

uform_status_t uform_outcome_wait(uform_outcome_t *outcome) {
    if (!outcome) return uform_status_invalid_argument_k;
    wait_for_calls_using(outcome);
    serial_atomic_ref<uform_status_t> status {outcome->status};
    uform_status_t current = status.load(std::memory_order_acquire);
    for (; current == uform_status_deferred_k; current = status.load(std::memory_order_acquire))
        serial_waiter_t {}(&outcome->status, current, std::size_t {0}, fu::wait_capped_k);
    return current;
}

uform_status_t uform_outcome_status(uform_outcome_t *outcome) {
    if (!outcome) return uform_status_invalid_argument_k;
    return serial_atomic_ref<uform_status_t> {outcome->status}.load(std::memory_order_acquire);
}

void uform_executor_for_ram(uform_executor_for_ram_t *arguments) {
    deferred_outcome_t outcome(arguments->outcome);
    if (reject_if(!arguments->executor, uform_status_invalid_argument_k, "The executor output must not be null",
                  outcome))
        return;
    *arguments->executor = nullptr;
    uform_executor_handle_t *const executor = new (std::nothrow) uform_executor_handle_t {};
    if (reject_if(!executor, uform_status_out_of_memory_k, "The executor could not be allocated", outcome)) return;
    *arguments->executor = executor;
}

void uform_executor_on_pool(uform_executor_on_pool_t *arguments) {
    deferred_outcome_t outcome(arguments->outcome);
    if (reject_if(!arguments->executor, uform_status_invalid_argument_k, "The executor output must not be null",
                  outcome))
        return;
    *arguments->executor = nullptr;
    if (reject_if(!arguments->pool, uform_status_invalid_argument_k, "The lent pool must not be null", outcome)) return;
    uform_executor_handle_t *const executor = new (std::nothrow) uform_executor_handle_t {};
    if (reject_if(!executor, uform_status_out_of_memory_k, "The executor could not be allocated", outcome)) return;
    *arguments->executor = executor;
}

void uform_executor_for_vram(uform_executor_for_vram_t *arguments) {
    deferred_outcome_t outcome(arguments->outcome);
    if (arguments->executor) *arguments->executor = nullptr;
    propagate_error(uform_status_device_k, outcome, "This build carries no GPU backends");
}

void uform_executor_drain(uform_executor_drain_t *arguments) {
    deferred_outcome_t outcome(arguments->outcome);
    if (arguments->executor) wait_for_calls_using(arguments->executor);
}

void uform_executor_free(uform_executor_t executor) {
    wait_for_calls_using(executor);
    delete executor;
}

void uform_arena_init(uform_arena_init_t *arguments) {
    deferred_outcome_t outcome(arguments->outcome);
    if (reject_if(!arguments->arena, uform_status_invalid_argument_k, "The arena output must not be null", outcome))
        return;
    *arguments->arena = nullptr;
    uform_arena_handle_t *const arena = new (std::nothrow) uform_arena_handle_t {};
    if (reject_if(!arena, uform_status_out_of_memory_k, "The arena could not be allocated", outcome)) return;
    *arguments->arena = arena;
}

void uform_arena_free(uform_arena_t arena) {
    wait_for_calls_using(arena);
    delete arena;
}

void uform_arena_describe(uform_arena_describe_t *arguments) {
    deferred_outcome_t outcome(arguments->outcome);
    if (reject_if(!arguments->arena || !arguments->description, uform_status_invalid_argument_k,
                  "The arena and its description must not be null", outcome))
        return;
    *arguments->description = {uform_residency_ram_k, -1};
}

char const *uform_status_name(uform_status_t status) {
    switch (status) {
    case uform_status_success_k: return "success";
    case uform_status_deferred_k: return "deferred";
    case uform_status_out_of_memory_k: return "out-of-memory";
    case uform_status_shape_mismatch_k: return "shape-mismatch";
    case uform_status_unsupported_layout_k: return "unsupported-layout";
    case uform_status_unknown_kind_k: return "unknown-kind";
    case uform_status_io_k: return "io";
    case uform_status_device_k: return "device";
    case uform_status_malformed_k: return "malformed";
    case uform_status_internal_k: return "internal";
    case uform_status_config_mismatch_k: return "config-mismatch";
    case uform_status_contended_k: return "contended";
    case uform_status_cancelled_k: return "cancelled";
    case uform_status_unsupported_k: return "unsupported";
    case uform_status_invalid_argument_k: return "invalid-argument";
    default: return "unrecognized";
    }
}

uform_status_t uform_status_from_name(uform_bytes_view_t name, uform_status_t *status) {
    return read_name<uform_status_t>(
        name,
        {uform_status_success_k, uform_status_deferred_k, uform_status_out_of_memory_k, uform_status_shape_mismatch_k,
         uform_status_unsupported_layout_k, uform_status_unknown_kind_k, uform_status_io_k, uform_status_device_k,
         uform_status_malformed_k, uform_status_internal_k, uform_status_config_mismatch_k, uform_status_contended_k,
         uform_status_cancelled_k, uform_status_unsupported_k, uform_status_invalid_argument_k},
        uform_status_name, status);
}

char const *uform_scalar_name(uform_scalar_t scalar) {
    switch (scalar) {
    case uform_scalar_unknown_k: return "unknown";
    case uform_scalar_f64_k: return "f64";
    case uform_scalar_f32_k: return "f32";
    case uform_scalar_bf16_k: return "bf16";
    case uform_scalar_f16_k: return "f16";
    case uform_scalar_e5m2_k: return "e5m2";
    case uform_scalar_e4m3_k: return "e4m3";
    case uform_scalar_e3m2_k: return "e3m2";
    case uform_scalar_e2m3_k: return "e2m3";
    case uform_scalar_i8_k: return "i8";
    case uform_scalar_i4x2_k: return "i4x2";
    case uform_scalar_i32_k: return "i32";
    case uform_scalar_u8_k: return "u8";
    case uform_scalar_u4x2_k: return "u4x2";
    case uform_scalar_u1x8_k: return "u1x8";
    default: return "unrecognized";
    }
}

uform_status_t uform_scalar_from_name(uform_bytes_view_t name, uform_scalar_t *scalar) {
    return read_name<uform_scalar_t>(
        name,
        {uform_scalar_f64_k, uform_scalar_f32_k, uform_scalar_bf16_k, uform_scalar_f16_k, uform_scalar_e5m2_k,
         uform_scalar_e4m3_k, uform_scalar_e3m2_k, uform_scalar_e2m3_k, uform_scalar_i8_k, uform_scalar_i4x2_k,
         uform_scalar_i32_k, uform_scalar_u8_k, uform_scalar_u4x2_k, uform_scalar_u1x8_k},
        uform_scalar_name, scalar);
}

char const *uform_mood_name(uform_mood_t mood) {
    switch (mood) {
    case uform_mood_parked_k: return "parked";
    case uform_mood_spinning_k: return "spinning";
    default: return "unrecognized";
    }
}

uform_status_t uform_mood_from_name(uform_bytes_view_t name, uform_mood_t *mood) {
    return read_name<uform_mood_t>(name, {uform_mood_parked_k, uform_mood_spinning_k}, uform_mood_name, mood);
}

char const *uform_cpu_class_name(uform_cpu_class_t cpu_class) {
    switch (cpu_class) {
    case uform_cpu_class_unknown_k: return "unknown";
    case uform_cpu_class_serial_k: return "serial";
    case uform_cpu_class_intel_haswell_k: return "intel-haswell";
    case uform_cpu_class_intel_sierra_k: return "intel-sierra";
    case uform_cpu_class_intel_skylake_k: return "intel-skylake";
    case uform_cpu_class_intel_icelake_k: return "intel-icelake";
    case uform_cpu_class_amd_genoa_k: return "amd-genoa";
    case uform_cpu_class_intel_sapphire_k: return "intel-sapphire";
    case uform_cpu_class_intel_granite_k: return "intel-granite";
    case uform_cpu_class_intel_diamond_k: return "intel-diamond";
    case uform_cpu_class_aws_g2_k: return "aws-g2";
    case uform_cpu_class_apple_m2_k: return "apple-m2";
    case uform_cpu_class_apple_m4_k: return "apple-m4";
    case uform_cpu_class_aws_g5_k: return "aws-g5";
    case uform_cpu_class_nvidia_vera_k: return "nvidia-vera";
    case uform_cpu_class_alibaba_c930_k: return "alibaba-c930";
    case uform_cpu_class_wasm32_v128_k: return "wasm32-v128";
    case uform_cpu_class_wasm64_extended_k: return "wasm64-extended";
    case uform_cpu_class_monoversioned_k: return "monoversioned";
    default: return "unrecognized";
    }
}

uform_status_t uform_cpu_class_from_name(uform_bytes_view_t name, uform_cpu_class_t *cpu_class) {
    return read_name<uform_cpu_class_t>(
        name,
        {uform_cpu_class_serial_k, uform_cpu_class_intel_haswell_k, uform_cpu_class_intel_sierra_k,
         uform_cpu_class_intel_skylake_k, uform_cpu_class_intel_icelake_k, uform_cpu_class_amd_genoa_k,
         uform_cpu_class_intel_sapphire_k, uform_cpu_class_intel_granite_k, uform_cpu_class_intel_diamond_k,
         uform_cpu_class_aws_g2_k, uform_cpu_class_apple_m2_k, uform_cpu_class_apple_m4_k, uform_cpu_class_aws_g5_k,
         uform_cpu_class_nvidia_vera_k, uform_cpu_class_alibaba_c930_k, uform_cpu_class_wasm32_v128_k,
         uform_cpu_class_wasm64_extended_k, uform_cpu_class_monoversioned_k},
        uform_cpu_class_name, cpu_class);
}

char const *uform_gpu_class_name(uform_gpu_class_t gpu_class) {
    switch (gpu_class) {
    case uform_gpu_class_unknown_k: return "unknown";
    case uform_gpu_class_nvidia_ampere_k: return "nvidia-ampere";
    case uform_gpu_class_nvidia_ada_k: return "nvidia-ada";
    case uform_gpu_class_nvidia_hopper_k: return "nvidia-hopper";
    case uform_gpu_class_nvidia_blackwell_k: return "nvidia-blackwell";
    case uform_gpu_class_nvidia_blackwellrtx_k: return "nvidia-blackwellrtx";
    case uform_gpu_class_nvidia_rubin_k: return "nvidia-rubin";
    case uform_gpu_class_amd_cdna3_k: return "amd-cdna3";
    case uform_gpu_class_amd_cdna4_k: return "amd-cdna4";
    case uform_gpu_class_amd_cdna5_k: return "amd-cdna5";
    case uform_gpu_class_apple_metal7_k: return "apple-metal7";
    case uform_gpu_class_apple_metal9_k: return "apple-metal9";
    case uform_gpu_class_apple_metal10_k: return "apple-metal10";
    default: return "unrecognized";
    }
}

uform_status_t uform_gpu_class_from_name(uform_bytes_view_t name, uform_gpu_class_t *gpu_class) {
    return read_name<uform_gpu_class_t>(
        name,
        {uform_gpu_class_nvidia_ampere_k, uform_gpu_class_nvidia_ada_k, uform_gpu_class_nvidia_hopper_k,
         uform_gpu_class_nvidia_blackwell_k, uform_gpu_class_nvidia_blackwellrtx_k, uform_gpu_class_nvidia_rubin_k,
         uform_gpu_class_amd_cdna3_k, uform_gpu_class_amd_cdna4_k, uform_gpu_class_amd_cdna5_k,
         uform_gpu_class_apple_metal7_k, uform_gpu_class_apple_metal9_k, uform_gpu_class_apple_metal10_k},
        uform_gpu_class_name, gpu_class);
}

void uform_gpu_devices(uform_gpu_devices_t *arguments) {
    deferred_outcome_t outcome(arguments->outcome);
    if (reject_if(!arguments->devices_count, uform_status_invalid_argument_k, "The count output must not be null",
                  outcome))
        return;
    *arguments->devices_count = 0;
}

} // extern "C"
