/**
 *  @file include/uform/utilities/blocking_files.hpp
 *  @author Ash Vardanian
 *  @date November 9, 2026
 *  @brief Native files read and written at absolute offsets, whole-file mappings, files replaced
 *      durably, all blocking the calling thread, and the shared libraries GPU drivers load from.
 *
 *  Every verb takes its path as a view and terminates it on the stack in the encoding the platform
 *  calls take, UTF-16 on Windows, so no verb allocates on the heap. After a crash
 *  @ref util::replace_file_durably leaves either the old file or the new one whole, which is what
 *  checkpoints and the mesh exchange files rely on.
 */
#pragma once

#include <cerrno>  // `errno`, `EINTR`, `ENOMEM`, `EINVAL`, `ENOSPC`, `EDQUOT`, `ENOTSUP`
#include <cstddef> // `std::size_t`
#include <cstdio>  // `std::rename`, `std::remove`
#include <cstdlib> // `std::malloc`, `std::free` - the copy WASI maps a file into
#include <cstring> // `std::memcpy`

#include <span>    // `std::span` - the buffer a path is terminated in
#include <utility> // `std::exchange`, `std::swap`, `std::move`

#include "uform/utilities/types.hpp" // `expected`, `expected_t`, `status_t`, `bytes_view_t`, `sz::string_view_t`

#if UFORM_OS_WINDOWS_
/*  Winsock 2 first keeps Winsock 1.1 out, unless a user's earlier `<windows.h>` brought it in. */
#if !defined(_WINSOCKAPI_)
#include <winsock2.h>
#endif
#include <windows.h> // `CreateFileW`, `ReadFile`, `WriteFile`, `FlushFileBuffers`, `MapViewOfFile`, `MoveFileExW`
#else
#include <fcntl.h>    // `open`, `fcntl`, `O_DIRECTORY`, `F_FULLFSYNC`
#include <sys/stat.h> // `fstat`
#include <unistd.h>   // `close`, `pread`, `pwrite`, `fsync`, `fdatasync`
#endif                // UFORM_OS_WINDOWS_
#if !UFORM_OS_WINDOWS_ && !UFORM_OS_WASI_
#include <dlfcn.h>    // `dlopen`, `dlsym` - each GPU driver opened at first use
#include <sys/mman.h> // `mmap`, `munmap`
#endif

namespace unum::uform::utilities {

#pragma region Native Files

/** What a caller may do to a file, and whether opening it creates or empties it. */
enum class file_access_t : u8_t {

    /** Reads alone, of a file that must already exist. */
    read_only_k,

    /** Reads and writes, of a file created or emptied by the open. */
    truncate_read_write_k,
};

/** Characters a path may span, its terminator included, which bounds the buffer each verb
 *  terminates it in. */
inline constexpr std::size_t path_capacity_k = 4096;

/** The largest run one positioned call moves, kept well inside every platform's own limit. */
inline constexpr std::size_t native_transfer_bytes_k = std::size_t {1} << 30;

#if UFORM_OS_WINDOWS_

/** The handle one open file is addressed by. */
using native_file_t = HANDLE;

/** The handle value that marks a file that never opened. */
inline native_file_t const invalid_native_file_k = INVALID_HANDLE_VALUE;

/** The character the wide path calls take. */
using path_character_t = wchar_t;

/** The status a failure the platform reports as @p reason surfaces as. */
[[nodiscard]] inline status_t native_file_status(DWORD reason) noexcept {
    switch (reason) {
    case ERROR_NOT_ENOUGH_MEMORY:
    case ERROR_OUTOFMEMORY: return status_code_t::out_of_memory_k;
    case ERROR_INVALID_PARAMETER:
    case ERROR_INVALID_HANDLE:
    case ERROR_INVALID_NAME: return status_code_t::invalid_argument_k;
    default: return status_code_t::io_k;
    }
}

/** Converts @p path into @p terminated as UTF-16, refusing a path holding a NUL, which the wide
 *  calls would cut short there, one that is not UTF-8, or one the buffer will not hold. */
[[nodiscard]] inline expected_t terminate_path_(sz::string_view_t path,
                                                std::span<path_character_t, path_capacity_k> terminated) noexcept {
    if (path.size() >= terminated.size() || path.contains('\0'))
        return expected_t::fail(status_code_t::invalid_argument_k, "the path holds a NUL or outgrows {} bytes: {}",
                                path_capacity_k, path);
    int const length = path.empty() ? 0
                                    : ::MultiByteToWideChar(CP_UTF8, MB_ERR_INVALID_CHARS, path.data(),
                                                            static_cast<int>(path.size()), terminated.data(),
                                                            static_cast<int>(terminated.size() - 1));
    if (length == 0 && !path.empty())
        return expected_t::fail(status_code_t::invalid_argument_k, "the path is not UTF-8: {}", path);
    terminated[static_cast<std::size_t>(length)] = L'\0';
    return {};
}

/** Opens @p path for positioned calls, or reports why the platform refused. */
[[nodiscard]] inline expected<native_file_t> open_native_file(sz::string_view_t path, file_access_t access) noexcept {
    path_character_t terminated[path_capacity_k];
    if (expected_t converted = terminate_path_(path, terminated); !converted)
        return expected<native_file_t>::fail(std::move(converted));
    DWORD const rights = access == file_access_t::read_only_k ? GENERIC_READ : GENERIC_READ | GENERIC_WRITE;
    // A symbolic link takes a privilege or Developer Mode to plant, so nothing mirrors `O_NOFOLLOW`
    DWORD const creation = access == file_access_t::read_only_k ? OPEN_EXISTING : CREATE_ALWAYS;
    native_file_t const file = ::CreateFileW(terminated, rights, FILE_SHARE_READ | FILE_SHARE_WRITE | FILE_SHARE_DELETE,
                                             nullptr, creation, FILE_ATTRIBUTE_NORMAL, nullptr);
    if (file == invalid_native_file_k) {
        DWORD const reason = ::GetLastError();
        return expected<native_file_t>::fail(native_file_status(reason), "opening {} failed (error={})", path, reason);
    }
    return file;
}

/** Closes @p file, which must have opened. */
[[nodiscard]] inline expected_t close_native_file(native_file_t file) noexcept {
    if (::CloseHandle(file)) return {};
    DWORD const reason = ::GetLastError();
    return expected_t::fail(native_file_status(reason), "closing a file failed (error={})", reason);
}

/** Reads at most @p buffer bytes from @p offset, answering how many arrived, which the end of the
 *  file cuts short. */
[[nodiscard]] inline expected<std::size_t> read_native_file(native_file_t file, u64_t offset,
                                                            bytes_span_t buffer) noexcept {
    std::size_t moved = 0;
    while (moved != buffer.size()) {
        bytes_span_t const asked = buffer.subspan(moved, fu::min_of(buffer.size() - moved, native_transfer_bytes_k));
        OVERLAPPED position {};
        position.Offset = static_cast<DWORD>((offset + moved) & 0xFFFFFFFFull);
        position.OffsetHigh = static_cast<DWORD>((offset + moved) >> 32);
        DWORD arrived = 0;
        BOOL const read = ::ReadFile(file, asked.data(), static_cast<DWORD>(asked.size()), &arrived, &position);
        DWORD const reason = read ? ERROR_SUCCESS : ::GetLastError();
        if (reason == ERROR_HANDLE_EOF) break;
        if (!read)
            return expected<std::size_t>::fail(native_file_status(reason), "reading at {} failed (error={})",
                                               offset + moved, reason);
        if (arrived == 0) break;
        moved += arrived;
    }
    return moved;
}

/** Writes at most @p buffer bytes at @p offset, answering how many landed, which a full volume cuts
 *  short. */
[[nodiscard]] inline expected<std::size_t> write_native_file(native_file_t file, u64_t offset,
                                                             bytes_view_t buffer) noexcept {
    std::size_t moved = 0;
    while (moved != buffer.size()) {
        bytes_view_t const asked = buffer.subspan(moved, fu::min_of(buffer.size() - moved, native_transfer_bytes_k));
        OVERLAPPED position {};
        position.Offset = static_cast<DWORD>((offset + moved) & 0xFFFFFFFFull);
        position.OffsetHigh = static_cast<DWORD>((offset + moved) >> 32);
        DWORD landed = 0;
        BOOL const written = ::WriteFile(file, asked.data(), static_cast<DWORD>(asked.size()), &landed, &position);
        DWORD const reason = written ? ERROR_SUCCESS : ::GetLastError();
        if (reason == ERROR_DISK_FULL || reason == ERROR_HANDLE_DISK_FULL) break;
        if (!written)
            return expected<std::size_t>::fail(native_file_status(reason), "writing at {} failed (error={})",
                                               offset + moved, reason);
        if (landed == 0) break;
        moved += landed;
    }
    return moved;
}

/** The bytes @p file holds. */
[[nodiscard]] inline expected<u64_t> native_file_capacity(native_file_t file) noexcept {
    LARGE_INTEGER size {};
    if (::GetFileSizeEx(file, &size)) return static_cast<u64_t>(size.QuadPart);
    DWORD const reason = ::GetLastError();
    return expected<u64_t>::fail(native_file_status(reason), "measuring a file failed (error={})", reason);
}

#else

/** The descriptor one open file is addressed by. */
using native_file_t = int;

/** The descriptor value that marks a file that never opened. */
inline constexpr native_file_t invalid_native_file_k = -1;

/** The character the path calls take. */
using path_character_t = char;

/** The status a failure the platform reports as @p reason surfaces as. */
[[nodiscard]] inline status_t native_file_status(int reason) noexcept {
    switch (reason) {
    case ENOMEM: return status_code_t::out_of_memory_k;
    case EINVAL:
    case EBADF:
    case ENAMETOOLONG: return status_code_t::invalid_argument_k;
    default: return status_code_t::io_k;
    }
}

/** Copies @p path into @p terminated, refusing a path holding a NUL, which the path calls would cut
 *  short there, or one the buffer will not hold. */
[[nodiscard]] inline expected_t terminate_path_(sz::string_view_t path,
                                                std::span<path_character_t, path_capacity_k> terminated) noexcept {
    if (path.size() >= terminated.size() || path.contains('\0'))
        return expected_t::fail(status_code_t::invalid_argument_k, "the path holds a NUL or outgrows {} bytes: {}",
                                path_capacity_k, path);
    std::memcpy(terminated.data(), path.data(), path.size());
    terminated[path.size()] = '\0';
    return {};
}

/** Opens @p path for positioned calls, or reports why the platform refused. */
[[nodiscard]] inline expected<native_file_t> open_native_file(sz::string_view_t path, file_access_t access) noexcept {
    path_character_t terminated[path_capacity_k];
    if (expected_t converted = terminate_path_(path, terminated); !converted)
        return expected<native_file_t>::fail(std::move(converted));
    // Only temporaries are truncated, and a link planted in their place must empty no file
    int const flags = access == file_access_t::read_only_k ? O_RDONLY | O_CLOEXEC
                                                           : O_RDWR | O_CREAT | O_TRUNC | O_NOFOLLOW | O_CLOEXEC;
    native_file_t const file = ::open(terminated, flags, 0644);
    if (file == invalid_native_file_k) {
        int const reason = errno;
        return expected<native_file_t>::fail(native_file_status(reason), "opening {} failed (error={})", path, reason);
    }
    return file;
}

/** Closes @p file, which must have opened. */
[[nodiscard]] inline expected_t close_native_file(native_file_t file) noexcept {
    if (::close(file) == 0) return {};
    int const reason = errno;
    return expected_t::fail(native_file_status(reason), "closing a file failed (error={})", reason);
}

/** Reads at most @p buffer bytes from @p offset, answering how many arrived, which the end of the
 *  file cuts short. */
[[nodiscard]] inline expected<std::size_t> read_native_file(native_file_t file, u64_t offset,
                                                            bytes_span_t buffer) noexcept {
    std::size_t moved = 0;
    while (moved != buffer.size()) {
        bytes_span_t const asked = buffer.subspan(moved, fu::min_of(buffer.size() - moved, native_transfer_bytes_k));
        ssize_t const arrived = ::pread(file, asked.data(), asked.size(), static_cast<off_t>(offset + moved));
        if (arrived < 0 && errno == EINTR) continue;
        if (arrived < 0) {
            int const reason = errno;
            return expected<std::size_t>::fail(native_file_status(reason), "reading at {} failed (error={})",
                                               offset + moved, reason);
        }
        if (arrived == 0) break;
        moved += static_cast<std::size_t>(arrived);
    }
    return moved;
}

/** Writes at most @p buffer bytes at @p offset, answering how many landed, which a full volume cuts
 *  short. */
[[nodiscard]] inline expected<std::size_t> write_native_file(native_file_t file, u64_t offset,
                                                             bytes_view_t buffer) noexcept {
    std::size_t moved = 0;
    while (moved != buffer.size()) {
        bytes_view_t const asked = buffer.subspan(moved, fu::min_of(buffer.size() - moved, native_transfer_bytes_k));
        ssize_t const landed = ::pwrite(file, asked.data(), asked.size(), static_cast<off_t>(offset + moved));
        if (landed < 0 && errno == EINTR) continue;
        if (landed < 0 && (errno == ENOSPC || errno == EDQUOT)) break;
        if (landed < 0) {
            int const reason = errno;
            return expected<std::size_t>::fail(native_file_status(reason), "writing at {} failed (error={})",
                                               offset + moved, reason);
        }
        if (landed == 0) break;
        moved += static_cast<std::size_t>(landed);
    }
    return moved;
}

/** The bytes @p file holds. */
[[nodiscard]] inline expected<u64_t> native_file_capacity(native_file_t file) noexcept {
    struct ::stat description {};
    if (::fstat(file, &description) == 0) return static_cast<u64_t>(description.st_size);
    int const reason = errno;
    return expected<u64_t>::fail(native_file_status(reason), "measuring a file failed (error={})", reason);
}

#endif // UFORM_OS_WINDOWS_

#if UFORM_OS_WINDOWS_

/** Makes every landed write of @p file durable. */
[[nodiscard]] inline expected_t flush_native_file(native_file_t file) noexcept {
    if (::FlushFileBuffers(file)) return {};
    DWORD const reason = ::GetLastError();
    return expected_t::fail(native_file_status(reason), "flushing a file failed (error={})", reason);
}

#elif UFORM_OS_APPLE_

/** Makes every landed write of @p file durable past the drive's cache, or as far as @c fsync
 *  reaches where the filesystem has no such barrier. */
[[nodiscard]] inline expected_t flush_native_file(native_file_t file) noexcept {
    if (::fcntl(file, F_FULLFSYNC) == 0) return {};
    if (errno == ENOTSUP && ::fsync(file) == 0) return {};
    int const reason = errno;
    return expected_t::fail(native_file_status(reason), "flushing a file failed (error={})", reason);
}

#else

/** Makes every landed write of @p file durable, without waiting on metadata a read does not
 *  need. */
[[nodiscard]] inline expected_t flush_native_file(native_file_t file) noexcept {
    if (::fdatasync(file) == 0) return {};
    int const reason = errno;
    return expected_t::fail(native_file_status(reason), "flushing a file failed (error={})", reason);
}

#endif

#pragma endregion Native Files

#pragma region Mappings

/** A read-only mapping of a whole file, unmapped on destruction, for parsers that read large files
 *  in place. WASI has no mappings, so there the bytes are read into a heap copy. */
class mapped_file_t {
  public:
    mapped_file_t() noexcept = default;
    mapped_file_t(mapped_file_t&& other) noexcept : bytes_(std::exchange(other.bytes_, {})) {}
    mapped_file_t& operator=(mapped_file_t&& other) noexcept {
        std::swap(bytes_, other.bytes_);
        return *this;
    }
    mapped_file_t(mapped_file_t const&) = delete;
    mapped_file_t& operator=(mapped_file_t const&) = delete;
    ~mapped_file_t() noexcept;

    /** Maps @p path whole, where an empty file maps to an empty view. */
    [[nodiscard]] static expected<mapped_file_t> map(sz::string_view_t path) noexcept;

    /** The mapped bytes. */
    [[nodiscard]] bytes_view_t bytes() const noexcept { return bytes_; }

  private:

    /** The mapped bytes, empty for an empty file and after a move. */
    bytes_span_t bytes_ {};
};

#if UFORM_OS_WINDOWS_

inline mapped_file_t::~mapped_file_t() noexcept {
    if (!bytes_.empty()) { [[maybe_unused]] BOOL const unmapped = ::UnmapViewOfFile(bytes_.data()); }
}

inline expected<mapped_file_t> mapped_file_t::map(sz::string_view_t path) noexcept {
    using mapped_t = expected<mapped_file_t>;
    auto [file, open_status] = open_native_file(path, file_access_t::read_only_k);
    if (!open_status) return mapped_t::fail(std::move(open_status));
    auto [capacity, measure_status] = native_file_capacity(file);
    mapped_file_t mapped;
    // A view keeps its section and its file alive, so both handles close on return.
    HANDLE const section = measure_status && capacity != 0
                               ? ::CreateFileMappingW(file, nullptr, PAGE_READONLY, 0, 0, nullptr)
                               : nullptr;
    void* const begin = section != nullptr ? ::MapViewOfFile(section, FILE_MAP_READ, 0, 0, 0) : nullptr;
    DWORD const reason = begin == nullptr ? ::GetLastError() : ERROR_SUCCESS;
    if (section != nullptr) { [[maybe_unused]] BOOL const closed_section = ::CloseHandle(section); }
    [[maybe_unused]] expected_t closed = close_native_file(file);
    if (!measure_status) return mapped_t::fail(std::move(measure_status));
    if (capacity == 0) return mapped;
    if (begin == nullptr)
        return mapped_t::fail(native_file_status(reason), "mapping {} failed (error={})", path, reason);
    mapped.bytes_ = {static_cast<b8_t*>(begin), static_cast<std::size_t>(capacity)};
    return mapped;
}

#elif UFORM_OS_WASI_

inline mapped_file_t::~mapped_file_t() noexcept {
    if (!bytes_.empty()) std::free(bytes_.data());
}

inline expected<mapped_file_t> mapped_file_t::map(sz::string_view_t path) noexcept {
    using mapped_t = expected<mapped_file_t>;
    auto [file, open_status] = open_native_file(path, file_access_t::read_only_k);
    if (!open_status) return mapped_t::fail(std::move(open_status));
    auto [capacity, measure_status] = native_file_capacity(file);
    mapped_file_t mapped;
    if (measure_status && capacity != 0) {
        void* const copy = std::malloc(static_cast<std::size_t>(capacity));
        if (copy != nullptr) mapped.bytes_ = {static_cast<b8_t*>(copy), static_cast<std::size_t>(capacity)};
    }
    auto [length, read_status] = mapped.bytes_.empty() ? expected<std::size_t> {0}
                                                       : read_native_file(file, 0, mapped.bytes_);
    [[maybe_unused]] expected_t closed = close_native_file(file);
    if (!measure_status) return mapped_t::fail(std::move(measure_status));
    if (capacity != 0 && mapped.bytes_.empty())
        return mapped_t::fail(status_code_t::out_of_memory_k, "{} needs {} bytes", path, capacity);
    if (!read_status) return mapped_t::fail(std::move(read_status));
    if (length != mapped.bytes_.size())
        return mapped_t::fail({status_code_t::io_k, "File shrank during read"}, "{} shrank to {} bytes while read",
                              path, length);
    return mapped;
}

#else

inline mapped_file_t::~mapped_file_t() noexcept {
    if (!bytes_.empty()) { [[maybe_unused]] int const unmapped = ::munmap(bytes_.data(), bytes_.size()); }
}

inline expected<mapped_file_t> mapped_file_t::map(sz::string_view_t path) noexcept {
    using mapped_t = expected<mapped_file_t>;
    auto [file, open_status] = open_native_file(path, file_access_t::read_only_k);
    if (!open_status) return mapped_t::fail(std::move(open_status));
    auto [capacity, measure_status] = native_file_capacity(file);
    mapped_file_t mapped;
    // A mapping outlives the descriptor it came from, which closes on return.
    void* const begin = measure_status && capacity != 0
                            ? ::mmap(nullptr, static_cast<std::size_t>(capacity), PROT_READ, MAP_PRIVATE, file, 0)
                            : MAP_FAILED;
    int const reason = begin == MAP_FAILED ? errno : 0;
    [[maybe_unused]] expected_t closed = close_native_file(file);
    if (!measure_status) return mapped_t::fail(std::move(measure_status));
    if (capacity == 0) return mapped;
    if (begin == MAP_FAILED)
        return mapped_t::fail(native_file_status(reason), "mapping {} failed (error={})", path, reason);
    mapped.bytes_ = {static_cast<b8_t*>(begin), static_cast<std::size_t>(capacity)};
    return mapped;
}

#endif

#pragma endregion Mappings

#pragma region Durable Files

#if UFORM_OS_WINDOWS_

/** Moves @p source over @p target in one step, replacing a target that exists, and returns once the
 *  move is on the medium. */
[[nodiscard]] inline expected_t rename_file(sz::string_view_t source, sz::string_view_t target) noexcept {
    path_character_t terminated_source[path_capacity_k], terminated_target[path_capacity_k];
    if (expected_t converted = terminate_path_(source, terminated_source); !converted) return converted;
    if (expected_t converted = terminate_path_(target, terminated_target); !converted) return converted;
    if (::MoveFileExW(terminated_source, terminated_target, MOVEFILE_REPLACE_EXISTING | MOVEFILE_WRITE_THROUGH))
        return {};
    DWORD const reason = ::GetLastError();
    return expected_t::fail(native_file_status(reason), "renaming {} over {} failed (error={})", source, target,
                            reason);
}

/** Removes the file @p path. */
[[nodiscard]] inline expected_t remove_file(sz::string_view_t path) noexcept {
    path_character_t terminated[path_capacity_k];
    if (expected_t converted = terminate_path_(path, terminated); !converted) return converted;
    if (::DeleteFileW(terminated)) return {};
    DWORD const reason = ::GetLastError();
    return expected_t::fail(native_file_status(reason), "removing {} failed (error={})", path, reason);
}

#else

/** Moves @p source over @p target in one step, replacing a target that exists; the move is durable
 *  once the directory is synced. */
[[nodiscard]] inline expected_t rename_file(sz::string_view_t source, sz::string_view_t target) noexcept {
    path_character_t terminated_source[path_capacity_k], terminated_target[path_capacity_k];
    if (expected_t converted = terminate_path_(source, terminated_source); !converted) return converted;
    if (expected_t converted = terminate_path_(target, terminated_target); !converted) return converted;
    if (std::rename(terminated_source, terminated_target) == 0) return {};
    int const reason = errno;
    return expected_t::fail(native_file_status(reason), "renaming {} over {} failed (error={})", source, target,
                            reason);
}

/** Removes the file @p path. */
[[nodiscard]] inline expected_t remove_file(sz::string_view_t path) noexcept {
    path_character_t terminated[path_capacity_k];
    if (expected_t converted = terminate_path_(path, terminated); !converted) return converted;
    if (std::remove(terminated) == 0) return {};
    int const reason = errno;
    return expected_t::fail(native_file_status(reason), "removing {} failed (error={})", path, reason);
}

#endif // UFORM_OS_WINDOWS_

#if UFORM_OS_WINDOWS_

/** Makes the entries of the directory @p path durable, flushing it through a handle that only
 *  backup semantics open. */
[[nodiscard]] inline expected_t sync_directory(sz::string_view_t path) noexcept {
    path_character_t terminated[path_capacity_k];
    if (expected_t converted = terminate_path_(path, terminated); !converted) return converted;
    HANDLE const directory = ::CreateFileW(terminated, GENERIC_READ | GENERIC_WRITE,
                                           FILE_SHARE_READ | FILE_SHARE_WRITE | FILE_SHARE_DELETE, nullptr,
                                           OPEN_EXISTING, FILE_FLAG_BACKUP_SEMANTICS, nullptr);
    DWORD const reason = directory == INVALID_HANDLE_VALUE ? ::GetLastError()
                         : ::FlushFileBuffers(directory)   ? ERROR_SUCCESS
                                                           : ::GetLastError();
    if (directory != INVALID_HANDLE_VALUE) { [[maybe_unused]] BOOL const closed = ::CloseHandle(directory); }
    if (reason == ERROR_SUCCESS) return {};
    return expected_t::fail(native_file_status(reason), "syncing the directory {} failed (error={})", path, reason);
}

#elif UFORM_OS_APPLE_

/** Makes the entries of the directory @p path durable past the drive's cache, or as far as @c fsync
 *  reaches where the filesystem has no such barrier. */
[[nodiscard]] inline expected_t sync_directory(sz::string_view_t path) noexcept {
    path_character_t terminated[path_capacity_k];
    if (expected_t converted = terminate_path_(path, terminated); !converted) return converted;
    int const directory = ::open(terminated, O_RDONLY | O_DIRECTORY | O_CLOEXEC);
    int reason = directory < 0 ? errno : 0;
    if (directory >= 0 && ::fcntl(directory, F_FULLFSYNC) != 0)
        reason = errno == ENOTSUP && ::fsync(directory) == 0 ? 0 : errno;
    if (directory >= 0) { [[maybe_unused]] int const closed = ::close(directory); }
    if (reason == 0) return {};
    return expected_t::fail(native_file_status(reason), "syncing the directory {} failed (error={})", path, reason);
}

#else

/** Makes the entries of the directory @p path durable, which the filesystem promises only for a
 *  directory that is itself @c fsync -ed. */
[[nodiscard]] inline expected_t sync_directory(sz::string_view_t path) noexcept {
    path_character_t terminated[path_capacity_k];
    if (expected_t converted = terminate_path_(path, terminated); !converted) return converted;
    int const directory = ::open(terminated, O_RDONLY | O_DIRECTORY | O_CLOEXEC);
    int const reason = directory < 0 ? errno : ::fsync(directory) == 0 ? 0 : errno;
    if (directory >= 0) { [[maybe_unused]] int const closed = ::close(directory); }
    if (reason == 0) return {};
    return expected_t::fail(native_file_status(reason), "syncing the directory {} failed (error={})", path, reason);
}

#endif

/** The directory holding @p path with its trailing separator, the root for a file at the root, or
 *  @c . for a bare name. */
[[nodiscard]] inline sz::string_view_t parent_directory_of(sz::string_view_t path) noexcept {
    std::size_t length = path.size();
    while (length != 0 && path[length - 1] != '/' && !(UFORM_OS_WINDOWS_ && path[length - 1] == '\\')) --length;
    return length != 0 ? sz::string_view_t {path.data(), length} : sz::string_view_t {"."};
}

/** Writes runs one after another into a native file from its start, each whole or failing, as
 *  @ref replace_file_durably hands it to its writer. */
class file_appender_t {
  public:
    explicit file_appender_t(native_file_t file) noexcept : file_(file) {}
    file_appender_t(file_appender_t const&) = delete;
    file_appender_t& operator=(file_appender_t const&) = delete;

    /** Writes @p bytes after everything appended before, failing on a full volume too. */
    [[nodiscard]] expected_t append(bytes_view_t bytes) noexcept {
        auto [length, status] = write_native_file(file_, offset_, bytes);
        if (!status) return expected_t::fail(std::move(status));
        offset_ += length;
        if (length != bytes.size())
            return expected_t::fail(status_code_t::io_k, "the volume filled up after {} bytes", offset_);
        return {};
    }

  private:

    /** The file the runs land in, which the appender never closes. */
    native_file_t file_;

    /** Where the next run lands. */
    u64_t offset_ = 0;
};

/**
 *  @brief Replaces @p path with what @p writer appends, so that after a crash either the old file
 *      or the new one is there whole.
 *
 *  The bytes go to a sibling @c .tmp, which is flushed, closed, renamed over @p path, and made
 *  durable by syncing the directory; a failure before the rename removes the sibling.
 *
 *  @param[in] path The file to replace, which need not exist yet.
 *  @param[in] writer Called once with a @c file_appender_t&, answering an @c expected_t.
 */
template <typename writer_type_>
[[nodiscard]] expected_t replace_file_durably(sz::string_view_t path, writer_type_&& writer) noexcept {
    constexpr char suffix[] = ".tmp";
    constexpr std::size_t suffix_length = sizeof(suffix) - 1;
    char temporary_text[path_capacity_k];
    if (path.size() + suffix_length >= path_capacity_k)
        return expected_t::fail(status_code_t::invalid_argument_k, "the path outgrows {} bytes: {}", path_capacity_k,
                                path);
    std::memcpy(temporary_text, path.data(), path.size());
    std::memcpy(temporary_text + path.size(), suffix, suffix_length);
    sz::string_view_t const temporary {temporary_text, path.size() + suffix_length};

    auto [file, status] = open_native_file(temporary, file_access_t::truncate_read_write_k);
    if (!status) return expected_t::fail(std::move(status));
    file_appender_t appender {file};
    expected_t written = writer(appender);
    if (written) written = flush_native_file(file);
    expected_t closed = close_native_file(file);
    if (written) written = std::move(closed);
    if (written) written = rename_file(temporary, path);
    if (!written) {
        [[maybe_unused]] expected_t removed = remove_file(temporary);
        return written;
    }
    return sync_directory(parent_directory_of(path));
}

#pragma endregion Durable Files

#pragma region Shared Libraries

/** A shared library opened by name and kept open for the process's life, as the entry points
 *  resolved from it are never unloaded; empty where the system has no such library or, as on
 *  WASI, no loader. */
struct shared_library_t {
    void* handle = nullptr;

    /** Opens the library named @p name , or answers the empty one. */
    static shared_library_t open([[maybe_unused]] char const* name) noexcept { // vendor-boundary: a C string
#if UFORM_OS_WINDOWS_
        return {reinterpret_cast<void*>(LoadLibraryA(name))}; // vendor-boundary: the loader answers an `HMODULE`
#elif UFORM_OS_WASI_
        return {};
#else
        return {dlopen(name, RTLD_NOW | RTLD_LOCAL)};
#endif
    }

    /** Points @p entry at the symbol exported as @p name , answering whether there is one. */
    template <typename function_type_>
    bool resolve([[maybe_unused]] char const* name, function_type_& entry) const noexcept { // vendor-boundary
        if (!handle) return false;
#if UFORM_OS_WINDOWS_
        entry = reinterpret_cast<function_type_>( // vendor-boundary: the loader answers a `FARPROC`
            GetProcAddress(static_cast<HMODULE>(handle), name));
#elif !UFORM_OS_WASI_
        entry = reinterpret_cast<function_type_>(dlsym(handle, name)); // vendor-boundary: `dlsym` answers a `void*`
#endif
        return entry != nullptr;
    }

    explicit operator bool() const noexcept { return handle != nullptr; }
};

#pragma endregion Shared Libraries

} // namespace unum::uform::utilities
