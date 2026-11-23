/**
 *  @file test/env.cpp
 *  @author Ash Vardanian
 *  @date July 6, 2026
 *  @brief The preflight every multi-node sbatch runs first, naming each environment gap in one line
 *      on the node that has it, before the real job finds it mid-suite.
 *
 *  1. Runtime closure: the binary links the whole `uform_train` stack, so starting at all proves
 *     every shared library resolves on the node.
 *  2. GPUDirect RDMA: an arena allocation of every local GPU must register through the endpoint
 *     the collectives build, on the NIC the transport picks for that GPU, in the single chunk the
 *     heaps rely on, since @c nvidia_peermem refuses a range spanning handles with @c EFAULT . The
 *     arena is grown first, so the range sits past a used tail.
 *  3. Rendezvous filesystem: the exchange directory the mesh names must be writable from the node.
 */
#undef NDEBUG // ! A test's oracle must stay live in every build

#include <cstdlib> // `std::getenv`
#include <cstdio>  // `std::fopen`, `std::remove`

#include <cuda.h> // `cuInit`, `cuDeviceGetCount`, `cuDevicePrimaryCtxRetain`

#include "uform/utilities/vram_nvidia_device.hpp"     // `gpu_arena_t`, `gpu_registrable_alloc`
#include "uform/utilities/infiniband_collectives.cuh" // `infiniband_nic_t`, `infiniband_endpoint_t` — the probed transport
#include "uform/utilities/mesh.hpp"                   // `host_domain_count_from_environment`

#include "harness.hpp" // `run_test_case`, `expect_true`

namespace util = unum::uform::utilities;
using namespace unum::uform::test;

/** Reservation ceiling of the probe arena — roomy enough for the bump tail plus
 *  the heap-like range. */
inline constexpr std::size_t probe_reserve_bytes_k = 64ull * 1024 * 1024;

/** The heap-like registrable range — sized like the optimizer's world-gradient heap, and ending
 *  mid-page like the collectives' small heaps, which a dma-buf export rounds out to whole pages. */
inline constexpr std::size_t probe_heap_bytes_k = 16ull * 1024 * 1024 + 384;

/** The ordinary bump allocation preceding the heap, so the mapped prefix carries a used tail. */
inline constexpr std::size_t probe_tail_bytes_k = 3ull * 1024 * 1024;

#if UFORM_WITH_INFINIBAND

/**
 *  @brief Register one GPU's registrable arena range through the endpoint the collectives build,
 *      on the NIC the transport picks for that GPU, and report the outcome.
 *  @param[in] device The GPU ordinal.
 */
static void probe_gpu_registration_(int device) noexcept {
    CUcontext context = nullptr;
    CUdevice handle = 0;
    bool const entered = cuDeviceGet(&handle, device) == CUDA_SUCCESS &&
                         cuDevicePrimaryCtxRetain(&context, handle) == CUDA_SUCCESS &&
                         cuCtxSetCurrent(context) == CUDA_SUCCESS;
    if (!expect_true(entered, "context entry failed (device={})", device)) return;

    auto [arena, make_arena_status] = util::gpu_arena_t::make(device, probe_reserve_bytes_k);
    if (!expect_true(make_arena_status, "probe arena reservation failed (device={})", device)) return;
    std::byte* const tail = arena.allocate_bytes(probe_tail_bytes_k);
    std::byte* const heap = arena.allocate_registrable_bytes(probe_heap_bytes_k);
    expect_true(tail != nullptr && heap != nullptr, "probe arena allocation failed (device={})", device);
    if (heap == nullptr) return;

    auto [nic, make_nic_status] = util::infiniband_nic_t::make(static_cast<util::u32_t>(device));
    if (!expect_true(make_nic_status, "no NIC serves GPU {}: {}", device, make_nic_status.description_)) return;
    auto [endpoint, make_endpoint_status] = util::infiniband_endpoint_t::make(
        nic, util::b8r1_span_t {heap, probe_heap_bytes_k});
    expect_true(make_endpoint_status, "GPUDirect registration failed on GPU {}: {}", device,
                make_endpoint_status.description_);
}

/** Registers every one of the @p device_count local GPUs with its same-index NIC. */
static case_result_t test_gpudirect_registration(int device_count) noexcept {
    if (device_count == 0) return case_result_t::skipped(no_gpu_device_k);
    for (int device = 0; device < device_count; ++device) probe_gpu_registration_(device);
    return {};
}
#else

/** Skips the GPUDirect check, which a build without InfiniBand cannot make. */
static case_result_t test_gpudirect_registration(int) noexcept {
    return case_result_t::skipped("built without UFORM_WITH_INFINIBAND");
}
#endif

/**
 *  @brief When the mesh contract names an exchange directory, prove it is writable from this node
 *      by creating and removing a probe file.
 */
static case_result_t test_exchange_directory() noexcept {
    char const* const exchange_directory = std::getenv("UFORM_EXCHANGE_DIR");
    if (util::host_domain_count_from_environment() == 1u)
        return case_result_t::skipped("needs UFORM_HOST_DOMAIN_COUNT above 1");
    if (!expect_true(exchange_directory != nullptr, "UFORM_EXCHANGE_DIR unset in a multi-node world")) return {};
    sz::string_t probe_path;
    if (!expect_true(util::format_into(probe_path, "{}/env-probe", exchange_directory), "probe path formatting failed"))
        return {};
    std::FILE* const probe_file = std::fopen(probe_path.c_str(), "w");
    expect_true(probe_file != nullptr, "exchange directory not writable: {}", exchange_directory);
    if (probe_file != nullptr) {
        std::fclose(probe_file);
        std::remove(probe_path.c_str());
    }
    return {};
}

int main(int argc, char** argv) {
    start_test_run(argc, argv);
    run_test_case("test_exchange_directory", test_exchange_directory);

    int device_count = 0;
    if (cuInit(0) != CUDA_SUCCESS || cuDeviceGetCount(&device_count) != CUDA_SUCCESS) device_count = 0;
    run_test_case("test_gpudirect_registration", test_gpudirect_registration, device_count);
    return finish_test_run();
}
