/**
 *  @file include/uform/loaders/direct_storage.hpp
 *  @author Ash Vardanian
 *  @date July 3, 2026
 *  @brief The storage configuration handle: one process-wide setup of cuDF's KvikIO layer, made at
 *      the composition root and threaded by reference into every reader.
 *
 *  KvikIO @b is cuDF's GPU Direct Storage path — it reads Parquet column chunks straight into device memory
 *  over cuFile where the filesystem supports it, and transparently falls back to threaded POSIX where it
 *  does not — so this cell owns no cuFile handle, no staging buffers, and no read code of its own; it exists
 *  to make that setup happen exactly once and to be the token a reader requires as proof it ran.
 *
 *  KvikIO chooses its read path from the environment ( @c KVIKIO_COMPAT_MODE — unset or @c ON forces
 *  POSIX, @c OFF enables GDS) and the target filesystem, per read. The session records which path the
 *  environment selected for diagnostics ( @ref read_path ); it never branches on it — that decision
 *  lives inside cuDF, one level below this boundary.
 */
#pragma once

#include "uform/utilities/types.hpp" // `expected`, `status_t`, the scalar aliases

namespace unum::uform::loaders {

using namespace unum::uform::utilities;

/** Which path KvikIO's environment selects for reads — informational, for start-up logs. */
enum class storage_read_path_t : u8_t {

    /** Threaded POSIX reads with a device copy — the portable fallback ( @c KVIKIO_COMPAT_MODE
     *  unset or @c ON , or a filesystem without GDS, such as this box's virtiofs mount). */
    compatibility_k = 0,

    /** cuFile / GPU Direct Storage reads straight to device ( @c KVIKIO_COMPAT_MODE=OFF on a
     *  GDS-capable filesystem). */
    gpu_direct_k = 1,
};

[[nodiscard]] inline char const *to_string(storage_read_path_t path) noexcept {
    return path == storage_read_path_t::gpu_direct_k ? "gpu_direct" : "compatibility";
}

/**
 *  @brief Move-only RAII token proving cuDF's KvikIO storage layer was configured — construct one at
 *      the composition root, thread it by reference into every reader.
 *
 *  Owns nothing: KvikIO's state is process-global, so construction is a one-shot @c set_up_kvikio call
 *  behind the exception firewall of this cell's @c .cpp . That call also gives KvikIO a thread per
 *  core and every device a stream-ordered RMM pool, so cuDF's frees never synchronize the device.
 */
struct gpu_direct_storage_session_t {

  private:
    gpu_direct_storage_session_t() noexcept = default;

  public:
    gpu_direct_storage_session_t(gpu_direct_storage_session_t const &) = delete;
    gpu_direct_storage_session_t &operator=(gpu_direct_storage_session_t const &) = delete;
    gpu_direct_storage_session_t(gpu_direct_storage_session_t &&) noexcept = default;
    gpu_direct_storage_session_t &operator=(gpu_direct_storage_session_t &&) noexcept = default;

    /** Configure KvikIO once — idempotent, process-wide. */
    [[nodiscard]] static expected<gpu_direct_storage_session_t> make() noexcept;

    /** The read path KvikIO's environment selected — diagnostics only. */
    [[nodiscard]] storage_read_path_t read_path() const noexcept { return read_path_; }

  private:
    storage_read_path_t read_path_ = storage_read_path_t::compatibility_k;
};

} // namespace unum::uform::loaders
