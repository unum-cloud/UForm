/**
 *  @brief The storage session's one firewalled body: configure cuDF's KvikIO layer. See the header for
 *      the contract. cuDF lives only in this translation unit and the sibling @c dataset_shards.cpp .
 *  @file src/loaders/direct_storage.cpp
 */

#include <cstring> // `std::strcmp`
#include <cstdlib> // `std::getenv` — the KvikIO compat-mode probe (read for diagnostics only)

#include <cudf/io/config_utils.hpp> // vendor-boundary: `cudf::io::kvikio_integration::set_up_kvikio`

#include "loaders/direct_storage.hpp"

namespace ul = unum::uform::loaders;
using namespace ul;

expected<gpu_direct_storage_session_t> gpu_direct_storage_session_t::make() noexcept {
    using result_t = expected<gpu_direct_storage_session_t>;
    try {
        cudf::io::kvikio_integration::set_up_kvikio();
    }
    catch (std::exception const &error) {
        return result_t::fail(status_t::io_error_k, "KvikIO setup failed at the firewall: {}", error.what());
    }
    catch (...) {
        return result_t::fail(status_t::io_error_k, "KvikIO setup raised a non-standard exception");
    }
    gpu_direct_storage_session_t session;
    // GDS engages only when compat mode is explicitly off; unset or "ON" means POSIX. Read for the
    // start-up log — the actual per-read choice stays inside cuDF.
    char const *compat = std::getenv("KVIKIO_COMPAT_MODE");
    session.read_path_ = (compat != nullptr && std::strcmp(compat, "OFF") == 0) ? storage_read_path_t::gpu_direct_k
                                                                                : storage_read_path_t::compatibility_k;
    return result_t {std::move(session)};
}
