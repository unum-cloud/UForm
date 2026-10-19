/**
 *  @file test/env.cpp
 *  @author Ash Vardanian
 *  @date July 6, 2026
 *  @brief The preflight every multi-node sbatch runs first, naming each environment gap in one line
 *      on the node that has it, before the real job finds it mid-suite.
 *
 *  1. Runtime closure: the binary links the whole `uform_train` stack, so starting at all proves
 *     every shared library resolves on the node.
 *  2. GPUDirect RDMA: an arena allocation of every local GPU must register with the NIC of the same
 *     index, in the single chunk the InfiniBand heaps rely on, since @c nvidia_peermem refuses a
 *     range spanning handles with @c EFAULT . The arena is grown first, so the range sits past a
 *     used tail.
 *  3. Rendezvous filesystem: the exchange directory the mesh names must be writable from the node.
 */
#include <cstdlib> // `std::getenv`
#include <cstdio>  // `std::fopen`, `std::remove`
#include <cerrno>  // `errno` — the registration failure detail

#include <cuda.h> // `cuInit`, `cuDeviceGetCount`, `cuDevicePrimaryCtxRetain`

#if defined(UFORM_WITH_INFINIBAND)
#include <infiniband/verbs.h> // `ibv_get_device_list`, `ibv_alloc_pd`, `ibv_reg_mr`
#endif

#include "core/gpu_allocators.hpp"         // `gpu_arena_t`, `gpu_registrable_alloc`
#include "core/mesh.hpp"                   // `host_domain_count_from_environment`
#include "core/infiniband_collectives.cuh" // `infiniband_access_flags_k` — the flags the heaps register with
#include "harness.hpp"                     // `test_suite_t`

using namespace unum::uform::core;
using namespace unum::uform::test;

/** Reservation ceiling of the probe arena — roomy enough for the bump tail plus the heap-like range. */
inline constexpr std::size_t probe_reserve_bytes_k = 64ull * 1024 * 1024;

/** The heap-like registrable range — sized like the optimizer's world-gradient heap. */
inline constexpr std::size_t probe_heap_bytes_k = 16ull * 1024 * 1024;

/** The ordinary bump allocation preceding the heap, so the mapped prefix carries a used tail. */
inline constexpr std::size_t probe_tail_bytes_k = 3ull * 1024 * 1024;

#if defined(UFORM_WITH_INFINIBAND)

/**
 *  @brief Register one GPU's registrable arena range with its same-index NIC and report the outcome —
 *      the per-device body of the GPUDirect check.
 *  @param[inout] suite The probe suite the checks report into.
 *  @param[in] device The GPU ordinal, paired with NIC @c mlx5_<device> (clamped to the NIC count).
 */
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

/** Registers every one of the @p device_count local GPUs with its same-index NIC. */
static void probe_gpudirect_(test_suite_t &suite, int device_count) noexcept {
    if (device_count == 0) return suite.skip(no_cuda_device_k);
    for (int device = 0; device < device_count; ++device) probe_gpu_registration_(suite, device);
}
#else

/** Skips the GPUDirect check, which a build without InfiniBand cannot make. */
static void probe_gpudirect_(test_suite_t &suite, int) noexcept { suite.skip("built without UFORM_WITH_INFINIBAND"); }
#endif

/**
 *  @brief When the mesh contract names an exchange directory, prove it is writable from this node by
 *      creating and removing a probe file.
 *  @param[inout] suite The probe suite the check reports into.
 */
static void probe_exchange_directory_(test_suite_t &suite) noexcept {
    char const *const exchange_directory = std::getenv("UFORM_EXCHANGE_DIR");
    if (host_domain_count_from_environment() == 1u) return suite.skip("a single-node run needs no rendezvous");
    suite.check(exchange_directory != nullptr, "UFORM_EXCHANGE_DIR unset in a multi-node world");
    if (exchange_directory == nullptr) return;
    sz::string_t probe_path;
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

int main(int, char **argv) {
    test_suite_t suite = test_suite_t::make(argv[0]);
    suite.run("exchange_directory", probe_exchange_directory_);

    int device_count = 0;
    if (cuInit(0) != CUDA_SUCCESS || cuDeviceGetCount(&device_count) != CUDA_SUCCESS) device_count = 0;
    suite.run("gpudirect_registration", probe_gpudirect_, device_count);
    return suite.finish();
}
