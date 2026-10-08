/**
 *  @brief Preflight probe for distributed runs — the first `srun` step of every multi-node sbatch, so an
 *      environment gap fails in one named line on the offending node before the real job spends a minute
 *      discovering it mid-suite. Three classes of gap it names:
 *
 *        1. Runtime closure — the binary links the full `uform_train` stack, so merely starting on a node
 *           proves every shared library (cuDNN, NumKong, the verbs providers) resolves there.
 *        2. GPUDirect RDMA — for every local GPU, an arena's registrable allocation must MR-register with
 *           the same-index NIC: the exact single-chunk contract the InfiniBand collectives' symmetric heaps
 *           rely on ( @c nvidia_peermem rejects handle-spanning ranges with @c EFAULT ). The probe first
 *           bumps the arena so the mapped prefix has a used tail — replaying the growth shape that once
 *           straddled a chunk boundary — before allocating the heap-like range.
 *        3. Rendezvous filesystem — when the mesh contract names an exchange directory, it must be writable
 *           from this node.
 *
 *  @file cuda_test/env.cpp
 *  @author Ash Vardanian
 *  @date July 6, 2026
 */
#include <cstdlib> // `std::getenv`
#include <cstdio>  // `std::fopen`, `std::remove`
#include <cerrno>  // `errno` — the registration failure detail

#include <cuda.h> // `cuInit`, `cuDeviceGetCount`, `cuDevicePrimaryCtxRetain`

#if defined(UFORM_WITH_INFINIBAND)
#include <infiniband/verbs.h> // `ibv_get_device_list`, `ibv_alloc_pd`, `ibv_reg_mr`
#endif

#include "cuda_core/gpu_allocators.hpp"         // `gpu_arena_t`, `gpu_registrable_alloc`
#include "cuda_core/mesh.hpp"                   // `host_domain_count_from_environment`
#include "cuda_core/infiniband_collectives.cuh" // `infiniband_access_flags_k` — the flags the heaps register with
#include "cuda_test/harness.hpp"                // `test_suite_t`

using namespace unum::uform::core;
using namespace unum::uform::test;

/** @brief Reservation ceiling of the probe arena — roomy enough for the bump tail plus the heap-like range. */
inline constexpr std::size_t probe_reserve_bytes_k = 64ull * 1024 * 1024;
/** @brief The heap-like registrable range — sized like the optimizer's world-gradient heap. */
inline constexpr std::size_t probe_heap_bytes_k = 16ull * 1024 * 1024;
/** @brief The ordinary bump allocation preceding the heap, so the mapped prefix carries a used tail. */
inline constexpr std::size_t probe_tail_bytes_k = 3ull * 1024 * 1024;

#if defined(UFORM_WITH_INFINIBAND)
/** @brief Register one GPU's registrable arena range with its same-index NIC and report the outcome —
 *      the per-device body of the GPUDirect check.
 *  @param[inout] suite The probe suite the checks report into.
 *  @param[in] device The GPU ordinal, paired with NIC @c mlx5_<device> (clamped to the NIC count). */
static void probe_gpu_registration_(test_suite_t &suite, int device) noexcept {
    CUcontext context = nullptr;
    CUdevice handle = 0;
    if (cuDeviceGet(&handle, device) != CUDA_SUCCESS || cuDevicePrimaryCtxRetain(&context, handle) != CUDA_SUCCESS ||
        cuCtxSetCurrent(context) != CUDA_SUCCESS) {
        suite.check(false, "context entry failed (device={})", device);
        return;
    }

    expected<gpu_arena_t> arena = gpu_arena_t::make(device, probe_reserve_bytes_k);
    suite.check(bool(arena), "probe arena reservation failed (device={})", device);
    if (!arena) return;
    std::byte *const tail = arena.value_.allocate_bytes(probe_tail_bytes_k);
    std::byte *const heap = arena.value_.allocate_registrable_bytes(probe_heap_bytes_k);
    suite.check(tail != nullptr && heap != nullptr, "probe arena allocation failed (device={})", device);
    if (heap == nullptr) return;

    int nic_count = 0;
    ibv_device **const nics = ibv_get_device_list(&nic_count);
    suite.check(nic_count > 0, "no verbs devices visible (device={})", device);
    if (nic_count == 0) return;
    int const nic_index = device < nic_count ? device : nic_count - 1;
    ibv_context *const nic = ibv_open_device(nics[nic_index]);
    suite.check(nic != nullptr, "NIC open failed ({})", ibv_get_device_name(nics[nic_index]));
    if (nic == nullptr) return;
    ibv_pd *const protection_domain = ibv_alloc_pd(nic);
    suite.check(protection_domain != nullptr, "protection-domain allocation failed (device={})", device);
    if (protection_domain != nullptr) {
        errno = 0;
        ibv_mr *const region = ibv_reg_mr(protection_domain, heap, probe_heap_bytes_k, infiniband_access_flags_k);
        suite.check(region != nullptr,
                    "GPUDirect registration failed: GPU {} + {} errno={} — check nvidia_peermem and the " //
                    "single-chunk heap contract",
                    device, ibv_get_device_name(nics[nic_index]), errno);
        if (region != nullptr) ibv_dereg_mr(region);
        ibv_dealloc_pd(protection_domain);
    }
    ibv_close_device(nic);
    ibv_free_device_list(nics);
}
#endif

/** @brief When the mesh contract names an exchange directory, prove it is writable from this node by
 *      creating and removing a probe file.
 *  @param[inout] suite The probe suite the check reports into. */
static void probe_exchange_directory_(test_suite_t &suite) noexcept {
    char const *const exchange_directory = std::getenv("UFORM_EXCHANGE_DIR");
    if (host_domain_count_from_environment() == 1u) return; // Single-node runs never rendezvous.
    suite.check(exchange_directory != nullptr, "UFORM_EXCHANGE_DIR unset in a multi-node world");
    if (exchange_directory == nullptr) return;
    sz::string probe_path;
    if (!format_into(probe_path, "{}/env-probe", exchange_directory)) {
        suite.check(false, "probe path formatting failed");
        return;
    }
    std::FILE *const probe_file = std::fopen(probe_path.c_str(), "w");
    suite.check(probe_file != nullptr, "exchange directory not writable: {}", exchange_directory);
    if (probe_file != nullptr) {
        std::fclose(probe_file);
        std::remove(probe_path.c_str());
    }
}

int main() {
    test_suite_t suite {"env"};

    CUresult const initialized = cuInit(0);
    suite.check(initialized == CUDA_SUCCESS, "driver init failed ({})", static_cast<int>(initialized));
    int device_count = 0;
    if (initialized == CUDA_SUCCESS) cuDeviceGetCount(&device_count);
    suite.check(device_count > 0, "no CUDA devices visible");

#if defined(UFORM_WITH_INFINIBAND)
    for (int device = 0; device < device_count; ++device) probe_gpu_registration_(suite, device);
#endif
    probe_exchange_directory_(suite);
    return suite.finish();
}
