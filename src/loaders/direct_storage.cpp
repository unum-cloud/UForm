/**
 *  @brief The storage session's one firewalled body: configure cuDF's KvikIO layer.
 *  @file src/loaders/direct_storage.cpp
 *
 *  See the header for the contract. cuDF lives only in this translation unit and the sibling
 *  @c dataset_shards.cpp .
 */

#include <cstring> // `std::strcmp`
#include <cstdlib> // `std::getenv` — the KvikIO compat-mode probe (read for diagnostics only)

#include <cudf/io/config_utils.hpp> // vendor-boundary: `cudf::io::kvikio_integration::set_up_kvikio`
#include <rmm/cuda_device.hpp>      // vendor-boundary: `rmm::get_num_cuda_devices`, `rmm::cuda_set_device_raii`
#include <rmm/mr/cuda_async_memory_resource.hpp> // vendor-boundary: `rmm::mr::cuda_async_memory_resource`
#include <rmm/mr/per_device_resource.hpp>        // vendor-boundary: `rmm::mr::set_per_device_resource`

#include "uform/utilities/ram_device.hpp" // `topology_t` — one KvikIO thread per core
#include "uform/loaders/direct_storage.hpp"

namespace ul = unum::uform::loaders;
using namespace ul;

expected<gpu_direct_storage_session_t> gpu_direct_storage_session_t::make() noexcept {
    using result_t = expected<gpu_direct_storage_session_t>;
    expected<topology_t> topology = topology_t::make();
    if (!topology) return result_t::fail(std::move(topology));
    std::size_t const cores = topology->logical_cores_count();
    if (cores == 0) return result_t::fail(status_t::io_k, "no logical core to run KvikIO's threads on");
    try {
        cudf::io::kvikio_integration::set_up_kvikio(static_cast<u32_t>(cores));
        // Pools outlive every buffer they serve, so only the first session installs them.
        [[maybe_unused]] static bool const pooled = [] { // vendor-boundary: RMM's resource map is process-global
            for (int device = 0; device != rmm::get_num_cuda_devices(); ++device) {
                rmm::cuda_set_device_raii const current {rmm::cuda_device_id {device}};
                rmm::mr::set_per_device_resource(rmm::cuda_device_id {device}, rmm::mr::cuda_async_memory_resource {});
            }
            return true;
        }();
    }
    catch (std::exception const &error) {
        return result_t::fail(status_t::io_k, "KvikIO setup failed at the firewall: {}", error.what());
    }
    catch (...) {
        return result_t::fail(status_t::io_k, "KvikIO setup raised a non-standard exception");
    }
    gpu_direct_storage_session_t session;
    // GDS engages only when compat mode is explicitly off; unset or "ON" means POSIX. Read for the
    // start-up log — the actual per-read choice stays inside cuDF.
    char const *compat = std::getenv("KVIKIO_COMPAT_MODE");
    session.read_path_ = (compat != nullptr && std::strcmp(compat, "OFF") == 0) ? storage_read_path_t::gpu_direct_k
                                                                                : storage_read_path_t::compatibility_k;
    return result_t {std::move(session)};
}
