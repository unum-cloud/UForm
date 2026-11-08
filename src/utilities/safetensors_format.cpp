/**
 *  @file src/utilities/safetensors_format.cpp
 *  @author Ash Vardanian
 *  @date July 3, 2026
 *  @brief The checkpoint cell's implementation: yyjson header building / parsing, plain POSIX
 *      writes and a read-only mapping — see the header for the contract.
 */
#include <cstdio>  // `std::rename` — the written file replaces the previous checkpoint at once
#include <cstdlib> // `std::malloc`, `std::free`
#include <cstring> // `std::memcpy` — the length prefix, the header and the tensor bytes

#include <unistd.h>   // `pread`, `pwrite`, `close`
#include <fcntl.h>    // `open`
#include <sys/stat.h> // `fstat` — the file extent

#include <charconv> // `std::from_chars` — the step metadata parse
#include <utility>  // `std::exchange`

#include <yyjson.h> // the safetensors JSON header — exceptions-free C

#include "uform/utilities/safetensors_format.hpp"

#if !UFORM_OS_WASI_
#include <sys/mman.h> // `mmap`, `munmap`
#endif

namespace uu = unum::uform::utilities;
using namespace uu;

/** Bytes one scalar of @p dtype occupies. */
[[nodiscard]] static std::size_t scalar_bytes_(checkpoint_dtype_t dtype) noexcept {
    return dtype == checkpoint_dtype_t::f32_k ? sizeof(f32_t) : sizeof(nk_bf16_t);
}

/** The safetensors spelling of @p dtype . */
[[nodiscard]] static char const *dtype_name_(checkpoint_dtype_t dtype) noexcept {
    return dtype == checkpoint_dtype_t::f32_k ? "F32" : "BF16";
}

/** A move-only owning wrapper over a POSIX file descriptor. */
struct checkpoint_file_t {
    checkpoint_file_t() noexcept = default;
    checkpoint_file_t(checkpoint_file_t const &) = delete;
    checkpoint_file_t &operator=(checkpoint_file_t const &) = delete;
    checkpoint_file_t(checkpoint_file_t &&other) noexcept : descriptor_(std::exchange(other.descriptor_, -1)) {}
    checkpoint_file_t &operator=(checkpoint_file_t &&other) noexcept {
        if (this != &other) {
            if (descriptor_ >= 0) ::close(descriptor_);
            descriptor_ = std::exchange(other.descriptor_, -1);
        }
        return *this;
    }
    ~checkpoint_file_t() noexcept {
        if (descriptor_ >= 0) ::close(descriptor_);
    }

    /** Open @p path with the POSIX @p flags , creating it @c 0644 when they ask to; a path
     *  holding a NUL, which @c open would cut short there, is refused. */
    [[nodiscard]] static expected<checkpoint_file_t> make(sz::string_view_t path, int flags) noexcept {
        if (path.contains('\0'))
            return expected<checkpoint_file_t>::fail(status_t::invalid_argument_k, "the path holds a NUL byte");
        sz::string_t path_text; // `open` needs a NUL-terminated `c_str`; short paths stay inline
        if (sz::failed(path_text.assign(path)))
            return expected<checkpoint_file_t>::fail(status_t::bad_alloc_k, "path staging failed");
        checkpoint_file_t file;
        file.descriptor_ = ::open(path_text.c_str(), flags, 0644);
        if (file.descriptor_ < 0) return expected<checkpoint_file_t>::fail(status_t::io_k, "open failed: {}", path);
        return expected<checkpoint_file_t> {std::move(file)};
    }

    [[nodiscard]] int get() const noexcept { return descriptor_; }

  private:
    int descriptor_ = -1;
};

/** Write @p bytes at @p offset , whole or fail — looped, because one @c pwrite moves at
 *  most ~2 GB on Linux and may stop short before that. */
[[nodiscard]] static expected_t write_at_(checkpoint_file_t const &file, std::size_t offset,
                                          bytes_view_t bytes) noexcept {
    for (bytes_view_t remaining = bytes; !remaining.empty();) {
        ssize_t const moved = ::pwrite(file.get(), remaining.data(), remaining.size(), static_cast<off_t>(offset));
        if (moved <= 0)
            return expected_t::fail(status_t::io_k, "pwrite failed at {} ({} bytes left of {})", offset,
                                    remaining.size(), bytes.size());
        offset += static_cast<std::size_t>(moved);
        remaining = remaining.subspan(static_cast<std::size_t>(moved));
    }
    return {};
}

#if UFORM_OS_WASI_

/** Read @p bytes from @p offset , whole or fail, looped as @ref write_at_ is. */
[[nodiscard]] static expected_t read_at_(checkpoint_file_t const &file, std::size_t offset,
                                         bytes_span_t bytes) noexcept {
    for (bytes_span_t remaining = bytes; !remaining.empty();) {
        ssize_t const moved = ::pread(file.get(), remaining.data(), remaining.size(), static_cast<off_t>(offset));
        if (moved <= 0)
            return expected_t::fail(status_t::io_k, "pread failed at {} ({} bytes left of {})", offset,
                                    remaining.size(), bytes.size());
        offset += static_cast<std::size_t>(moved);
        remaining = remaining.subspan(static_cast<std::size_t>(moved));
    }
    return {};
}
#endif

/** Release @p bytes : unmapped, or freed on WASI, which reads them in for want of @c mmap . */
static void release_(bytes_span_t bytes) noexcept {
#if UFORM_OS_WASI_
    std::free(bytes.data());
#else
    ::munmap(bytes.data(), bytes.size());
#endif
}

/**
 *  @brief Serialize the safetensors header for @p entries inside @p pool : the metadata first, then
 *      a <tt>name: {dtype, shape, data_offsets}</tt> record per tensor, offsets as byte prefix sums in
 *      insertion order. The returned view points into the pool's backing.
 *
 *  @par Statuses:
 *  - @c status_t::bad_alloc_k The pool ran dry — the caller grows it and retries.
 */
[[nodiscard]] static expected<sz::string_view_t> write_header_(std::span<checkpoint_entry_t const> entries,
                                                               std::span<checkpoint_metadata_t const> pairs,
                                                               yyjson_alc *pool) noexcept {
    using serialized_t = expected<sz::string_view_t>;
    yyjson_mut_doc *document = yyjson_mut_doc_new(pool);
    if (document == nullptr) return serialized_t::fail(status_t::bad_alloc_k, "header pool ran dry");
    yyjson_mut_val *root = yyjson_mut_obj(document);
    yyjson_mut_doc_set_root(document, root);
    yyjson_mut_val *metadata = yyjson_mut_obj(document);
    bool built = root != nullptr && metadata != nullptr;
    for (std::size_t index = 0; built && index != pairs.size(); ++index) {
        checkpoint_metadata_t const &pair = pairs[index];
        yyjson_mut_val *key = yyjson_mut_strn(document, pair.key.data(), pair.key.size());
        yyjson_mut_val *value = yyjson_mut_strn(document, pair.value.data(), pair.value.size());
        built = key != nullptr && value != nullptr && yyjson_mut_obj_add(metadata, key, value);
    }
    built = built && yyjson_mut_obj_add_val(document, root, "__metadata__", metadata);
    std::size_t data_cursor = 0;
    for (std::size_t index = 0; built && index != entries.size(); ++index) {
        checkpoint_entry_t const &entry = entries[index];
        std::size_t const bytes = entry.rows * entry.cols * scalar_bytes_(entry.dtype);
        yyjson_mut_val *key = yyjson_mut_strn(document, entry.name.data(), entry.name.size());
        yyjson_mut_val *tensor = yyjson_mut_obj(document);
        yyjson_mut_val *shape = yyjson_mut_arr(document);
        yyjson_mut_val *offsets = yyjson_mut_arr(document);
        built = key != nullptr && tensor != nullptr && shape != nullptr && offsets != nullptr &&
                yyjson_mut_obj_add_str(document, tensor, "dtype", dtype_name_(entry.dtype)) &&
                yyjson_mut_arr_add_uint(document, shape, entry.rows) &&
                yyjson_mut_arr_add_uint(document, shape, entry.cols) &&
                yyjson_mut_obj_add_val(document, tensor, "shape", shape) &&
                yyjson_mut_arr_add_uint(document, offsets, data_cursor) &&
                yyjson_mut_arr_add_uint(document, offsets, data_cursor + bytes) &&
                yyjson_mut_obj_add_val(document, tensor, "data_offsets", offsets) &&
                yyjson_mut_obj_add(root, key, tensor);
        data_cursor += bytes;
    }
    std::size_t header_bytes = 0;
    char *header = built ? yyjson_mut_write_opts(document, 0, pool, &header_bytes, nullptr) : nullptr;
    if (header == nullptr) return serialized_t::fail(status_t::bad_alloc_k, "header pool ran dry");
    return serialized_t {sz::string_view_t {header, header_bytes}};
}

/** Copy @p text into an owned string. */
[[nodiscard]] static expected_t own_text_(sz::string_t &owned, sz::string_view_t text) noexcept {
    if (sz::failed(owned.assign(text))) return expected_t::fail(status_t::bad_alloc_k, "text staging failed: {}", text);
    return {};
}

/** The mirror of @ref write_header_ : parse the @p header text inside @p pool , index every BF16
 *  and F32 record into @p entries — shapes tolerated as rank-1 @c [cols] or rank-2 @c [rows,cols] ,
 *  offsets validated against @p payload_bytes — and every string metadata pair into @p pairs .
 *  Records of other dtypes are skipped, and a lookup treats them exactly like absent names. */
[[nodiscard]] static expected_t read_header_(std::span<char> header, yyjson_alc *pool, u64_t payload_bytes,
                                             st::basic_vector<checkpoint_entry_t> &entries,
                                             st::basic_vector<checkpoint_metadata_t> &pairs) noexcept {
    yyjson_doc *document = // default flags skip the trailing pad spaces; the pool owns the document
        yyjson_read_opts(header.data(), header.size(), 0, pool, nullptr);
    yyjson_val *root = document != nullptr ? yyjson_doc_get_root(document) : nullptr;
    yyjson_obj_iter iterator;
    if (root == nullptr || !yyjson_obj_iter_init(root, &iterator))
        return expected_t::fail(status_t::invalid_argument_k, "not a safetensors header");

    yyjson_val *key = nullptr;
    while ((key = yyjson_obj_iter_next(&iterator)) != nullptr) {
        yyjson_val *value = yyjson_obj_iter_get_val(key);
        sz::string_view_t const key_text {yyjson_get_str(key), yyjson_get_len(key)};
        if (key_text == "__metadata__") {
            yyjson_obj_iter metadata_iterator;
            if (!yyjson_obj_iter_init(value, &metadata_iterator))
                return expected_t::fail(status_t::invalid_argument_k, "__metadata__ is not an object");
            yyjson_val *metadata_key = nullptr;
            while ((metadata_key = yyjson_obj_iter_next(&metadata_iterator)) != nullptr) {
                yyjson_val *metadata_value = yyjson_obj_iter_get_val(metadata_key);
                char const *value_chars = yyjson_get_str(metadata_value); // null for non-strings, skipped
                if (value_chars == nullptr) continue;
                checkpoint_metadata_t pair;
                expected_t owned = own_text_(pair.key, {yyjson_get_str(metadata_key), yyjson_get_len(metadata_key)});
                if (owned) owned = own_text_(pair.value, {value_chars, yyjson_get_len(metadata_value)});
                if (!owned) return owned;
                if (st::failed(pairs.push_back(std::move(pair))))
                    return expected_t::fail(status_t::bad_alloc_k, "metadata table growth failed");
            }
            continue;
        }
        yyjson_val *dtype = yyjson_obj_get(value, "dtype");
        char const *dtype_chars = yyjson_get_str(dtype);
        if (dtype_chars == nullptr)
            return expected_t::fail(status_t::invalid_argument_k, "{} misses its dtype", key_text);
        sz::string_view_t const dtype_text {dtype_chars, yyjson_get_len(dtype)};
        checkpoint_dtype_t scalar_type = checkpoint_dtype_t::bf16_k;
        if (dtype_text == "F32") scalar_type = checkpoint_dtype_t::f32_k;
        else if (dtype_text != "BF16") continue;
        std::size_t const scalar_size = scalar_bytes_(scalar_type);

        yyjson_val *shape = yyjson_obj_get(value, "shape");
        std::size_t const rank = yyjson_arr_size(shape);
        if (rank == 0 || rank > 2)
            return expected_t::fail(status_t::unsupported_k, "{} is rank-{} — this cell stores rank-1/2", key_text,
                                    rank);
        u64_t const cols = yyjson_get_uint(yyjson_arr_get(shape, rank - 1));
        u64_t const rows = rank == 2 ? yyjson_get_uint(yyjson_arr_get(shape, 0)) : 1;

        yyjson_val *offsets = yyjson_obj_get(value, "data_offsets");
        u64_t const begin = yyjson_get_uint(yyjson_arr_get(offsets, 0));
        u64_t const end = yyjson_get_uint(yyjson_arr_get(offsets, 1));
        bool const in_bounds = begin <= end && end <= payload_bytes && begin % scalar_size == 0;
        if (!in_bounds || end - begin != rows * cols * scalar_size)
            return expected_t::fail(status_t::invalid_argument_k,
                                    "{} holds bytes [{}, {}), its shape [{},{}] disagrees", key_text, begin, end, rows,
                                    cols);

        checkpoint_entry_t entry;
        if (expected_t owned = own_text_(entry.name, key_text); !owned) return owned;
        entry.dtype = scalar_type;
        entry.rows = static_cast<std::size_t>(rows); // In bounds of a mapped payload, so 32-bit wasm fits it too.
        entry.cols = static_cast<std::size_t>(cols);
        entry.byte_offset = static_cast<std::size_t>(begin);
        if (st::failed(entries.push_back(std::move(entry))))
            return expected_t::fail(status_t::bad_alloc_k, "entry table growth failed");
    }
    return {};
}

expected_t checkpoint_to_save_t::reserve(u64_t total_bytes, u64_t tensor_count) noexcept {
    if (st::failed(payload_.reserve(total_bytes)))
        return expected_t::fail(status_t::bad_alloc_k, "payload reservation failed ({} bytes)", total_bytes);
    if (st::failed(entries_.reserve(tensor_count)))
        return expected_t::fail(status_t::bad_alloc_k, "entry table reservation failed ({} tensors)", tensor_count);
    return {};
}

expected_t checkpoint_to_save_t::insert(sz::string_view_t name, nk::tensor_view<nk_bf16_t, 2> data) noexcept {
    if (data.stride_bytes(1) != sizeof(nk_bf16_t) ||
        data.stride_bytes(0) != static_cast<std::ptrdiff_t>(data.extent(1) * sizeof(nk_bf16_t)))
        return expected_t::fail(status_t::unsupported_k, "{} must be contiguous", name);
    return insert_(name, checkpoint_dtype_t::bf16_k, bytes_of(data), data.extent(0), data.extent(1));
}

expected_t checkpoint_to_save_t::insert(sz::string_view_t name, nk::tensor_view<f32_t, 2> data) noexcept {
    if (data.stride_bytes(1) != sizeof(f32_t) ||
        data.stride_bytes(0) != static_cast<std::ptrdiff_t>(data.extent(1) * sizeof(f32_t)))
        return expected_t::fail(status_t::unsupported_k, "{} must be contiguous", name);
    return insert_(name, checkpoint_dtype_t::f32_k, bytes_of(data), data.extent(0), data.extent(1));
}

expected_t checkpoint_to_save_t::insert_(sz::string_view_t name, checkpoint_dtype_t dtype, bytes_view_t data,
                                         std::size_t rows, std::size_t cols) noexcept {
    if (name.empty() || rows * cols == 0)
        return expected_t::fail(status_t::invalid_argument_k, "an entry needs a name and data");
    std::size_t const scalar_size = scalar_bytes_(dtype);
    if (payload_.size() % scalar_size != 0)
        return expected_t::fail(status_t::invalid_argument_k, "{} would start at byte {}, unaligned for {}", name,
                                payload_.size(), dtype_name_(dtype));
    u64_t const key = static_cast<u64_t>(name.hash());
    if (auto const found = index_.find(key); found != index_.end()) {
        if (entries_[found.value()].name.view() == name)
            return expected_t::fail(status_t::invalid_argument_k, "duplicate entry: {}", name);
        return expected_t::fail(status_t::invalid_argument_k, "{} collides with {} in its name hash", name,
                                entries_[found.value()].name.view());
    }

    checkpoint_entry_t entry;
    if (expected_t owned = own_text_(entry.name, name); !owned) return owned;
    entry.dtype = dtype;
    entry.rows = rows;
    entry.cols = cols;
    entry.byte_offset = payload_.size();
    std::size_t const bytes = rows * cols * scalar_size;
    if (st::failed(payload_.resize(st::assume_uninitialized, entry.byte_offset + bytes)))
        return expected_t::fail(status_t::bad_alloc_k, "payload growth failed ({} bytes)", entry.byte_offset + bytes);
    std::memcpy(payload_.data() + entry.byte_offset, data.data(), bytes);
    if (st::failed(entries_.push_back(std::move(entry)))) {
        (void)payload_.resize(payload_.size() - bytes); // a shrink never fails
        return expected_t::fail(status_t::bad_alloc_k, "entry table growth failed");
    }
    if (st::failed(index_.upsert(st::mapping<u64_t, u32_t> {key, static_cast<u32_t>(entries_.size() - 1)}))) {
        (void)entries_.resize(entries_.size() - 1); // a shrink never fails
        (void)payload_.resize(payload_.size() - bytes);
        return expected_t::fail(status_t::bad_alloc_k, "entry index growth failed");
    }
    return {};
}

expected_t checkpoint_to_save_t::metadata(sz::string_view_t key, sz::string_view_t value) noexcept {
    for (checkpoint_metadata_t &existing : metadata_)
        if (existing.key.view() == key) return own_text_(existing.value, value);
    checkpoint_metadata_t pair;
    expected_t owned = own_text_(pair.key, key);
    if (owned) owned = own_text_(pair.value, value);
    if (owned && st::failed(metadata_.push_back(std::move(pair))))
        owned = expected_t::fail(status_t::bad_alloc_k, "metadata table growth failed");
    return owned;
}

expected_t checkpoint_to_save_t::save(sz::string_view_t path) noexcept {
    if (entries_.empty()) return expected_t::fail(status_t::invalid_argument_k, "a checkpoint needs tensors");

    // The header builds in the hoisted scratch through a pool allocator spanning its whole backing
    // — grown lazily, doubling when the pool runs dry, so the buffer settles after the first save
    // and a `clear`ed container re-inflates it without allocating.
    sz::string_view_t header {};
    for (;;) {
        std::size_t const pool_bytes = scratch_.capacity() != 0 ? scratch_.capacity() : 4096;
        if (st::failed(scratch_.resize(st::assume_uninitialized, pool_bytes)))
            return expected_t::fail(status_t::bad_alloc_k, "header scratch growth failed ({} bytes)", pool_bytes);
        yyjson_alc pool;
        if (!yyjson_alc_pool_init(&pool, scratch_.data(), scratch_.size()))
            return expected_t::fail(status_t::bad_alloc_k, "header pool setup failed ({} bytes)", scratch_.size());
        expected<sz::string_view_t> serialized = write_header_(entries_, metadata_, &pool);
        if (serialized) {
            header = *serialized;
            break;
        }
        if (st::failed(scratch_.reserve(scratch_.size() * 2)))
            return expected_t::fail(std::move(serialized)); // a dry pool with no room left to grow
    }

    // The JSON is space-padded to the next 8-byte boundary and the recorded length counts the
    // padding, so the payload starts 8-aligned — byte-compatible with Hugging Face's own writer.
    u64_t const header_length = divide_round_up(header.size(), 8) * 8;
    char const padding[8] = {' ', ' ', ' ', ' ', ' ', ' ', ' ', ' '};

    sz::string_t final_path, partial_path;
    if (sz::failed(final_path.assign(path)) || !format_into(partial_path, "{}.partial", path))
        return expected_t::fail(status_t::bad_alloc_k, "path staging failed");
    {
        expected<checkpoint_file_t> file = checkpoint_file_t::make(partial_path.view(), O_WRONLY | O_CREAT | O_TRUNC);
        if (!file) return expected_t::fail(std::move(file));
        std::string_view const header_text = header;
        expected_t written = write_at_(*file, 0, std::as_bytes(std::span {&header_length, 1}));
        if (written) written = write_at_(*file, sizeof(header_length), std::as_bytes(std::span {header_text}));
        if (written && header_length != header.size())
            written = write_at_(*file, sizeof(header_length) + header.size(),
                                std::as_bytes(std::span {padding}).first(header_length - header.size()));

        // The payload is already contiguous in insertion order — one write moves all of it.
        std::size_t const payload_base = sizeof(header_length) + header_length;
        if (written) written = write_at_(*file, payload_base, payload_);
        if (written && ::fsync(file->get()) != 0)
            written = expected_t::fail(status_t::io_k, "fsync failed: {}", partial_path.view());
        if (!written) return written;
    }
    if (std::rename(partial_path.c_str(), final_path.c_str()) != 0)
        return expected_t::fail(status_t::io_k, "renaming {} over {} failed", partial_path.view(), path);
    return {};
}

void checkpoint_to_save_t::clear() noexcept {
    entries_.clear();
    index_.clear();
    metadata_.clear();
    payload_.clear();
    scratch_.clear();
}

checkpoint_mapping_t::checkpoint_mapping_t(checkpoint_mapping_t &&other) noexcept
    : bytes_(std::exchange(other.bytes_, {})) {}

checkpoint_mapping_t &checkpoint_mapping_t::operator=(checkpoint_mapping_t &&other) noexcept {
    if (this != &other) {
        if (!bytes_.empty()) release_(bytes_);
        bytes_ = std::exchange(other.bytes_, {});
    }
    return *this;
}

checkpoint_mapping_t::~checkpoint_mapping_t() noexcept {
    if (!bytes_.empty()) release_(bytes_);
}

expected<checkpoint_mapping_t> checkpoint_mapping_t::make(sz::string_view_t path) noexcept {
    using mapped_t = expected<checkpoint_mapping_t>;
    expected<checkpoint_file_t> file = checkpoint_file_t::make(path, O_RDONLY);
    if (!file) return mapped_t::fail(std::move(file));
    struct ::stat file_stat {};
    if (::fstat(file->get(), &file_stat) != 0) return mapped_t::fail(status_t::io_k, "fstat failed: {}", path);
    std::size_t const file_bytes = static_cast<std::size_t>(file_stat.st_size);
    if (file_bytes == 0) return mapped_t::fail(status_t::invalid_argument_k, "{} is empty", path);
#if UFORM_OS_WASI_
    void *const mapped = std::malloc(file_bytes);
    if (!mapped) return mapped_t::fail(status_t::bad_alloc_k, "{} needs {} bytes", path, file_bytes);
    if (expected_t read = read_at_(*file, 0, {static_cast<b8_t *>(mapped), file_bytes}); !read) {
        std::free(mapped);
        return mapped_t::fail(std::move(read));
    }
#else
    void *const mapped = ::mmap(nullptr, file_bytes, PROT_READ, MAP_PRIVATE, file->get(), 0);
    if (mapped == MAP_FAILED) return mapped_t::fail(status_t::io_k, "mmap failed: {}", path);
#endif
    checkpoint_mapping_t mapping;
    mapping.bytes_ = {static_cast<b8_t *>(mapped), file_bytes};
    return mapped_t {std::move(mapping)};
}

expected<checkpoint_to_load_t> checkpoint_to_load_t::make(sz::string_view_t path) noexcept {
    using loaded_t = expected<checkpoint_to_load_t>;
    expected<checkpoint_mapping_t> mapping = checkpoint_mapping_t::make(path);
    if (!mapping) return loaded_t::fail(std::move(mapping));
    loaded_t loaded = view(mapping->bytes());
    if (loaded) loaded->mapping_ = std::move(*mapping);
    return loaded;
}

expected<checkpoint_to_load_t> checkpoint_to_load_t::view(bytes_view_t buffer) noexcept {
    using loaded_t = expected<checkpoint_to_load_t>;

    // The length prefix is untrusted bytes until proven otherwise, and the image's own extent is
    // the only honest bound on it: a header the image can't hold is corruption.
    u64_t header_length = 0;
    if (buffer.size() < sizeof(header_length))
        return loaded_t::fail(status_t::invalid_argument_k, "a {}-byte image is shorter than a length prefix",
                              buffer.size());
    std::memcpy(&header_length, buffer.data(), sizeof(header_length));
    if (header_length == 0 || header_length > buffer.size() - sizeof(header_length))
        return loaded_t::fail(status_t::invalid_argument_k, "the image declares a {}-byte header, it holds {}",
                              header_length, buffer.size());
    u64_t const payload_bytes = buffer.size() - sizeof(header_length) - header_length;

    // The parse scratch holds a copy of the header up front and the yyjson pool behind it — one
    // allocation, released once the tables are indexed.
    st::basic_vector<char> scratch;
    std::size_t const pool_bytes = yyjson_read_max_memory_usage(header_length, 0);
    if (st::failed(scratch.resize(st::assume_uninitialized, header_length + pool_bytes)))
        return loaded_t::fail(status_t::bad_alloc_k, "header staging failed ({} bytes)", header_length + pool_bytes);
    std::memcpy(scratch.data(), buffer.data() + sizeof(header_length), header_length);
    yyjson_alc pool;
    if (!yyjson_alc_pool_init(&pool, scratch.data() + header_length, pool_bytes))
        return loaded_t::fail(status_t::bad_alloc_k, "parse pool setup failed ({} bytes)", pool_bytes);
    checkpoint_to_load_t loaded;
    std::span<char> const header {scratch.data(), static_cast<std::size_t>(header_length)};
    if (expected_t parsed = read_header_(header, &pool, payload_bytes, loaded.entries_, loaded.metadata_); !parsed)
        return loaded_t::fail(std::move(parsed));
    for (std::size_t position = 0; position != loaded.entries_.size(); ++position) {
        sz::string_view_t const name = loaded.entries_[position].name.view();
        u64_t const key = static_cast<u64_t>(name.hash());
        if (auto const found = loaded.index_.find(key); found != loaded.index_.end())
            return loaded_t::fail(status_t::invalid_argument_k, "{} and {} share a name hash",
                                  loaded.entries_[found.value()].name.view(), name);
        if (st::failed(loaded.index_.upsert(st::mapping<u64_t, u32_t> {key, static_cast<u32_t>(position)})))
            return loaded_t::fail(status_t::bad_alloc_k, "entry index growth failed");
    }
    loaded.payload_ = buffer.subspan(sizeof(header_length) + header_length);
    return loaded_t {std::move(loaded)};
}

checkpoint_entry_t const *checkpoint_to_load_t::find_(sz::string_view_t name, checkpoint_dtype_t dtype) const noexcept {
    auto const found = index_.find(static_cast<u64_t>(name.hash()));
    if (found == index_.end()) return nullptr;
    checkpoint_entry_t const &entry = entries_[found.value()];
    return entry.name.view() == name && entry.dtype == dtype ? &entry : nullptr;
}

nk::tensor_view<nk_bf16_t, 2> checkpoint_to_load_t::operator[](sz::string_view_t name) const noexcept {
    checkpoint_entry_t const *entry = find_(name, checkpoint_dtype_t::bf16_k);
    if (entry == nullptr) return {};
    return nk::tensor_view<nk_bf16_t, 2> {
        static_cast<nk_bf16_t const *>(static_cast<void const *>(payload_.data() + entry->byte_offset)), entry->rows,
        entry->cols};
}

nk::tensor_view<f32_t, 2> checkpoint_to_load_t::float32(sz::string_view_t name) const noexcept {
    checkpoint_entry_t const *entry = find_(name, checkpoint_dtype_t::f32_k);
    if (entry == nullptr) return {};
    return nk::tensor_view<f32_t, 2> {
        static_cast<f32_t const *>(static_cast<void const *>(payload_.data() + entry->byte_offset)), entry->rows,
        entry->cols};
}

sz::string_view_t checkpoint_to_load_t::metadata(sz::string_view_t key) const noexcept {
    for (checkpoint_metadata_t const &pair : metadata_)
        if (pair.key.view() == key) return pair.value.view();
    return {};
}

u64_t checkpoint_to_load_t::step() const noexcept {
    sz::string_view_t const text = metadata("step");
    u64_t step = 0;
    (void)std::from_chars(text.data(), text.data() + text.size(), step);
    return step;
}
