/**
 *  @brief Foundational vocabulary of the UForm v4 primitives cell: statuses, error channel, launch constants,
 *      mesh topology.
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

#include <cstdint> // `std::int32_t`, `std::uint32_t`

#include <format>          // `std::vformat`, `std::make_format_args`
#include <source_location> // `std::source_location`
#include <string>          // `std::string`
#include <utility>         // `std::move`, `std::forward`

#include <cuda.h> // `CUresult`, `cuGetErrorString`

namespace unum::uform::primitives {

/** @brief Thread count of one CUDA warp — the granularity of every shuffle-based reduction tree. */
inline constexpr unsigned warp_size_k = 32u;

/** @brief Default thread count per CTA for hand-rolled kernels. Fits the CUB @c BlockReduce<float,256>
 *      temp-storage budget (1 KB SMEM) with full occupancy on sm_90 and sm_100. */
inline constexpr unsigned kernel_block_size_default_k = 256u;

/** @brief Outcome of any fallible operation. @c success_k == 0; all error codes are negative + stable. */
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
    case status_t::other_k: return "other";
    }
    return "unknown";
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
 *  but @c noexcept moves. The description is materialized with @c std::format at the failure site — the
 *  only place the operands (shapes, counts, labels) are still in scope — and @ref format_at_t records that
 *  site's @c std::source_location .
 */
template <typename value_type_>
struct expected {
    value_type_ value_ {};
    status_t status_ = status_t::success_k;
    CUresult cu_error_ = CUDA_SUCCESS;
    std::string description_ {};
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
        try {
            result.description_ = std::vformat(format.format, std::make_format_args(args...));
        }
        catch (...) {
            // A malformed runtime argument; surface the bare format string so the diagnostic still lands.
            result.description_ = format.format;
        }
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
        result.description_ += ": ";
        result.description_ += cu_error_string;
        return result;
    }
};

/**
 *  @brief The three-domain mesh shape — the host-side single source of truth for cluster geometry.
 *
 *  Names the three NESTED GPU domains a run spans, smallest to largest, and stores ONLY their three sizes —
 *  every grouping question is integer arithmetic off them, never stored:
 *
 *    - @c host_domain — the GPUs ONE process drives (one box / one OS image; ≤ 8). On NVL72 a process drives
 *      2–4 GPUs, so the host_domain is SMALLER than the multimem_domain.
 *    - @c multimem_domain — the GPUs reachable by @c multimem.* over NVLink / NVSwitch (NVL8 = 8, NVL72 = 72),
 *      addressed by a single multimem VA. One or more host_domains compose a multimem_domain.
 *    - @c colocated_domain — the whole InfiniBand world (~10^5), reached by device-initiated one-sided RDMA.
 *      One or more multimem_domains compose the colocated_domain; it shares ONE global batch.
 *
 *  Invariant: @c host ≤ multimem ≤ colocated , @c multimem % host == 0 , @c colocated % multimem == 0 .
 *  Single NVL8 box: all three are 8. Two-box RDMA: host = multimem = 8, colocated = 16. NVL72 + RDMA:
 *  host = 8, multimem = 72, colocated = 144. Immutable after rendezvous — the topology is identity, not
 *  state. A rank is one GPU, named by a single @c std::uint32_t @c gpu_index in
 *  @c [0, gpus_per_colocated_domain) ; there are no rank/index aliases.
 */
struct mesh_topology_t {
    /** @brief GPUs one process drives — one host / NVLink box (≤ 8). */
    std::uint32_t gpus_per_host_domain = 1u;
    /** @brief GPUs in one NVLink domain reachable by @c multimem.* (≤ 72 today). */
    std::uint32_t gpus_per_multimem_domain = 1u;
    /** @brief Total data-parallel GPUs sharing ONE global batch — the whole InfiniBand world (~10^5). */
    std::uint32_t gpus_per_colocated_domain = 1u;

    /** @brief Which host_domain a global @p gpu_index belongs to. */
    [[nodiscard]] std::uint32_t host_domain_of(std::uint32_t gpu_index) const noexcept {
        return gpu_index / gpus_per_host_domain;
    }
    /** @brief Which multimem_domain a global @p gpu_index belongs to. */
    [[nodiscard]] std::uint32_t multimem_domain_of(std::uint32_t gpu_index) const noexcept {
        return gpu_index / gpus_per_multimem_domain;
    }
    /** @brief A global @p gpu_index 's position WITHIN its multimem_domain — the intra-domain collective rank. */
    [[nodiscard]] std::uint32_t index_in_multimem_domain(std::uint32_t gpu_index) const noexcept {
        return gpu_index % gpus_per_multimem_domain;
    }
    /** @brief Number of host_domains (processes) in the colocated_domain. */
    [[nodiscard]] std::uint32_t host_domain_count() const noexcept {
        return gpus_per_colocated_domain / gpus_per_host_domain;
    }
    /** @brief Number of multimem_domains in the colocated_domain — the inter-domain collective width. */
    [[nodiscard]] std::uint32_t multimem_domain_count() const noexcept {
        return gpus_per_colocated_domain / gpus_per_multimem_domain;
    }
};

} // namespace unum::uform::primitives
