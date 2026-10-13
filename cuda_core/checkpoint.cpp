/**
 *  @brief The checkpoint cell's implementation: yyjson header building / parsing and plain POSIX
 *      file I/O — see the header for the contract.
 *  @file cuda_core/checkpoint.cpp
 *  @author Ash Vardanian
 *  @date July 3, 2026
 */
#include <charconv> // `std::from_chars` — the step metadata parse

#include <unistd.h>   // `pwrite`, `pread`, `close`
#include <fcntl.h>    // `open`
#include <sys/stat.h> // `fstat` — the payload extent

#include <yyjson.h> // the safetensors JSON header — exceptions-free C

#include "cuda_core/checkpoint.hpp"

namespace uc = unum::uform::core;
using namespace uc;

/** @brief A move-only owning wrapper over a POSIX file descriptor. */
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

    /** @brief Open @p path for reading, or create-truncate it for writing. */
    [[nodiscard]] static expected<checkpoint_file_t> make(sz::string_view_t path, bool writable) noexcept {
        sz::string_t path_text; // `open` needs a NUL-terminated `c_str`; short paths stay inline
        if (sz::failed(path_text.try_assign({path.data(), path.size()})))
            return expected<checkpoint_file_t>::fail(status_t::bad_alloc_k, "path staging failed");
        checkpoint_file_t file;
        file.descriptor_ = writable ? ::open(path_text.c_str(), O_WRONLY | O_CREAT | O_TRUNC, 0644)
                                    : ::open(path_text.c_str(), O_RDONLY);
        if (file.descriptor_ < 0)
            return expected<checkpoint_file_t>::fail(status_t::io_error_k, "open failed: {}", path);
        return expected<checkpoint_file_t> {std::move(file)};
    }

    [[nodiscard]] int get() const noexcept { return descriptor_; }

  private:
    int descriptor_ = -1;
};

/** @brief Write @p count bytes at @p offset , whole or fail — looped, because one `pwrite` moves at
 *      most ~2 GB on Linux and may stop short before that. */
[[nodiscard]] static expected_t write_at_(checkpoint_file_t const &file, std::size_t offset, void const *bytes,
                                          std::size_t count) noexcept {
    char const *cursor = static_cast<char const *>(bytes);
    std::size_t remaining = count;
    while (remaining != 0) {
        ssize_t const moved = ::pwrite(file.get(), cursor, remaining, static_cast<off_t>(offset));
        if (moved <= 0)
            return expected_t::fail(status_t::io_error_k, "pwrite failed at {} ({} bytes left of {})", offset,
                                    remaining, count);
        cursor += moved;
        offset += static_cast<std::size_t>(moved);
        remaining -= static_cast<std::size_t>(moved);
    }
    return {};
}

/** @brief Read @p count bytes at @p offset , whole or fail — looped like @ref write_at_ ; a premature
 *      end-of-file is a failure, the callers size their reads from the file's own metadata. */
[[nodiscard]] static expected_t read_at_(checkpoint_file_t const &file, std::size_t offset, void *bytes,
                                         std::size_t count) noexcept {
    char *cursor = static_cast<char *>(bytes);
    std::size_t remaining = count;
    while (remaining != 0) {
        ssize_t const moved = ::pread(file.get(), cursor, remaining, static_cast<off_t>(offset));
        if (moved <= 0)
            return expected_t::fail(status_t::io_error_k, "pread failed at {} ({} bytes left of {})", offset, remaining,
                                    count);
        cursor += moved;
        offset += static_cast<std::size_t>(moved);
        remaining -= static_cast<std::size_t>(moved);
    }
    return {};
}

/** @brief Serialize the safetensors header for @p entries inside @p pool : the step metadata first,
 *      then a `name: {dtype, shape, data_offsets}` record per tensor, offsets as byte prefix sums in
 *      insertion order. The returned view points into the pool's backing.
 *  @retval status_t::bad_alloc_k The pool ran dry — the caller grows it and retries. */
[[nodiscard]] static expected<sz::string_view_t> write_header_(std::span<checkpoint_entry_t const> entries, u64_t step,
                                                               yyjson_alc *pool) noexcept {
    using serialized_t = expected<sz::string_view_t>;
    yyjson_mut_doc *document = yyjson_mut_doc_new(pool);
    if (document == nullptr) return serialized_t::fail(status_t::bad_alloc_k, "header pool ran dry");
    yyjson_mut_val *root = yyjson_mut_obj(document);
    yyjson_mut_doc_set_root(document, root);
    yyjson_mut_val *metadata = yyjson_mut_obj(document);
    sz::string_t step_text; // referenced by the DOM until serialization below; fits the inline buffer
    bool built = root != nullptr && metadata != nullptr && format_into(step_text, "{}", step) &&
                 yyjson_mut_obj_add_strn(document, metadata, "step", step_text.data(), step_text.size()) &&
                 yyjson_mut_obj_add_val(document, root, "__metadata__", metadata);
    std::size_t data_cursor = 0;
    for (std::size_t index = 0; built && index != entries.size(); ++index) {
        checkpoint_entry_t const &entry = entries[index];
        std::size_t const bytes = entry.rows * entry.cols * sizeof(bf16_t);
        yyjson_mut_val *key = yyjson_mut_strn(document, entry.name.data(), entry.name.size());
        yyjson_mut_val *tensor = yyjson_mut_obj(document);
        yyjson_mut_val *shape = yyjson_mut_arr(document);
        yyjson_mut_val *offsets = yyjson_mut_arr(document);
        built = key != nullptr && tensor != nullptr && shape != nullptr && offsets != nullptr &&
                yyjson_mut_obj_add_str(document, tensor, "dtype", "BF16") &&
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

/** @brief The mirror of @ref write_header_ : parse the @p header text inside @p pool , index every
 *      BF16 record into @p entries — shapes tolerated as rank-1 @c [cols] or rank-2 @c [rows,cols] ,
 *      offsets validated against @p payload_bytes — and pick @p step out of the metadata. Records of
 *      other dtypes are skipped: this cell trains in BFloat16 only, and a lookup treats them exactly
 *      like absent names. */
[[nodiscard]] static expected_t read_header_(std::span<char> header, yyjson_alc *pool, u64_t payload_bytes,
                                             array<checkpoint_entry_t> &entries, u64_t &step) noexcept {
    yyjson_doc *document = // default flags skip the trailing pad spaces; the pool owns the document
        yyjson_read_opts(header.data(), header.size(), 0, pool, nullptr);
    yyjson_val *root = document != nullptr ? yyjson_doc_get_root(document) : nullptr;
    yyjson_obj_iter iterator;
    if (root == nullptr || !yyjson_obj_iter_init(root, &iterator))
        return expected_t::fail(status_t::invalid_input_k, "not a safetensors header");

    yyjson_val *key = nullptr;
    while ((key = yyjson_obj_iter_next(&iterator)) != nullptr) {
        yyjson_val *value = yyjson_obj_iter_get_val(key);
        sz::string_view_t const key_text {yyjson_get_str(key), yyjson_get_len(key)};
        if (key_text == "__metadata__") {
            yyjson_val *step_value = yyjson_obj_get(value, "step");
            char const *step_chars = yyjson_get_str(step_value); // null for absent or non-string
            if (step_chars != nullptr) (void)std::from_chars(step_chars, step_chars + yyjson_get_len(step_value), step);
            continue;
        }
        yyjson_val *dtype = yyjson_obj_get(value, "dtype");
        char const *dtype_chars = yyjson_get_str(dtype);
        if (dtype_chars == nullptr) return expected_t::fail(status_t::invalid_input_k, "{} misses its dtype", key_text);
        if (sz::string_view_t {dtype_chars, yyjson_get_len(dtype)} != "BF16") continue;

        yyjson_val *shape = yyjson_obj_get(value, "shape");
        std::size_t const rank = yyjson_arr_size(shape);
        if (rank == 0 || rank > 2)
            return expected_t::fail(status_t::shape_unsupported_k, "{} is rank-{} — this cell stores rank-1/2",
                                    key_text, rank);
        u64_t const cols = yyjson_get_uint(yyjson_arr_get(shape, rank - 1));
        u64_t const rows = rank == 2 ? yyjson_get_uint(yyjson_arr_get(shape, 0)) : 1;

        yyjson_val *offsets = yyjson_obj_get(value, "data_offsets");
        u64_t const begin = yyjson_get_uint(yyjson_arr_get(offsets, 0));
        u64_t const end = yyjson_get_uint(yyjson_arr_get(offsets, 1));
        bool const in_bounds = begin <= end && end <= payload_bytes && begin % sizeof(bf16_t) == 0;
        if (!in_bounds || end - begin != rows * cols * sizeof(bf16_t))
            return expected_t::fail(status_t::invalid_input_k, "{} holds bytes [{}, {}), its shape [{},{}] disagrees",
                                    key_text, begin, end, rows, cols);

        checkpoint_entry_t entry;
        if (sz::failed(entry.name.try_assign({key_text.data(), key_text.size()})))
            return expected_t::fail(status_t::bad_alloc_k, "name staging failed: {}", key_text);
        entry.rows = rows;
        entry.cols = cols;
        entry.scalar_offset = begin / sizeof(bf16_t);
        if (entries.try_push_back(std::move(entry)) != status_t::success_k)
            return expected_t::fail(status_t::bad_alloc_k, "entry table growth failed");
    }
    return {};
}

expected<checkpoint_to_save_t> checkpoint_to_save_t::make(pinned_alloc_t allocator) noexcept {
    checkpoint_to_save_t container;
    container.payload_ = array<bf16_t, pinned_alloc<bf16_t>> {pinned_alloc<bf16_t> {allocator}};
    return expected<checkpoint_to_save_t> {std::move(container)};
}

expected_t checkpoint_to_save_t::try_reserve(u64_t total_scalars, u64_t tensor_count) noexcept {
    if (payload_.try_reserve(total_scalars) != status_t::success_k)
        return expected_t::fail(status_t::bad_alloc_k, "pinned payload reservation failed ({} scalars)", total_scalars);
    if (entries_.try_reserve(tensor_count) != status_t::success_k)
        return expected_t::fail(status_t::bad_alloc_k, "entry table reservation failed ({} tensors)", tensor_count);
    return {};
}

expected_t checkpoint_to_save_t::try_insert(sz::string_view_t name, nk::tensor_view<bf16_t, 2> data,
                                            gpu_stream_t &stream) noexcept {
    if (name.empty() || data.empty())
        return expected_t::fail(status_t::invalid_input_k, "an entry needs a name and data");
    for (checkpoint_entry_t const &existing : entries_)
        if (existing.name.view() == name)
            return expected_t::fail(status_t::invalid_input_k, "duplicate entry: {}", name);
    switch_to_gpu_t gpu_scope {stream};
    if (!gpu_scope.ok()) return expected_t::fail(status_t::cuda_error_k, "context entry failed");

    checkpoint_entry_t entry;
    if (sz::failed(entry.name.try_assign({name.data(), name.size()})))
        return expected_t::fail(status_t::bad_alloc_k, "name staging failed: {}", name);
    entry.rows = data.extent(0);
    entry.cols = data.extent(1);
    entry.scalar_offset = payload_.size();

    // Growing the pinned payload may relocate it — safe, because the per-insert synchronize below
    // guarantees no copy is ever in flight between `try_insert` calls.
    std::size_t const scalars = data.numel();
    if (payload_.try_resize_uninitialized(entry.scalar_offset + scalars) != status_t::success_k)
        return expected_t::fail(status_t::bad_alloc_k, "pinned payload growth failed ({} scalars)",
                                entry.scalar_offset + scalars);

    // Stream order places this copy after every training kernel already enqueued; the bounded
    // synchronize drains it, so the weights are snapshotted — and free to mutate — on return.
    nk::tensor_view<bf16_t, 1> const device_source = data.flatten<1>();
    nk::tensor_span<bf16_t, 1> const payload_span {payload_.data(), payload_.size()};
    nk::tensor_span<bf16_t, 1> const host_destination = payload_span(
        nk::range(entry.scalar_offset, entry.scalar_offset + scalars), nk::slice);
    expected_t copied = copy_device_to_host(device_source, host_destination, stream);
    if (copied)
        if (CUresult result = cuStreamSynchronize(stream.get()); result != CUDA_SUCCESS)
            copied = expected_t::fail(result, "snapshot synchronize failed: {}", name);
    if (copied && entries_.try_push_back(std::move(entry)) != status_t::success_k)
        copied = expected_t::fail(status_t::bad_alloc_k, "entry table growth failed");
    if (!copied) (void)payload_.try_resize(entry.scalar_offset); // a shrink never fails
    return copied;
}

expected_t checkpoint_to_save_t::save(sz::string_view_t path, u64_t step) noexcept {
    if (entries_.empty()) return expected_t::fail(status_t::invalid_input_k, "a checkpoint needs tensors");

    // The header builds in the hoisted scratch through a pool allocator spanning its whole backing —
    // grown lazily, doubling when the pool runs dry, so the buffer settles after the first save and a
    // `clear`ed container re-inflates it without allocating.
    sz::string_view_t header {};
    for (;;) {
        std::size_t const pool_bytes = scratch_.capacity() != 0 ? scratch_.capacity() : 4096;
        if (scratch_.try_resize_uninitialized(pool_bytes) != status_t::success_k)
            return expected_t::fail(status_t::bad_alloc_k, "header scratch growth failed ({} bytes)", pool_bytes);
        yyjson_alc pool;
        if (!yyjson_alc_pool_init(&pool, scratch_.data(), scratch_.size()))
            return expected_t::fail(status_t::bad_alloc_k, "header pool setup failed ({} bytes)", scratch_.size());
        expected<sz::string_view_t> serialized = write_header_(entries_.view(), step, &pool);
        if (serialized) {
            header = serialized.value_;
            break;
        }
        if (scratch_.try_reserve(scratch_.size() * 2) != status_t::success_k)
            return expected_t::fail(std::move(serialized)); // a dry pool with no room left to grow
    }

    // The JSON is space-padded to the next 8-byte boundary and the recorded length counts the padding,
    // so the payload starts 8-aligned — byte-compatible with Hugging Face's own writer.
    u64_t const header_length = round_up_to_multiple(header.size(), 8);
    char const padding[8] = {' ', ' ', ' ', ' ', ' ', ' ', ' ', ' '};

    expected<checkpoint_file_t> file = checkpoint_file_t::make(path, true);
    if (!file) return expected_t::fail(std::move(file));
    expected_t written = write_at_(file.value_, 0, &header_length, sizeof(header_length));
    if (written) written = write_at_(file.value_, sizeof(header_length), header.data(), header.size());
    if (written && header_length != header.size())
        written = write_at_(file.value_, sizeof(header_length) + header.size(), padding, header_length - header.size());
    if (!written) return written;

    // The payload is already contiguous in insertion order — one write moves all of it.
    std::size_t const payload_base = sizeof(header_length) + header_length;
    return write_at_(file.value_, payload_base, payload_.data(), payload_.size() * sizeof(bf16_t));
}

void checkpoint_to_save_t::clear() noexcept {
    entries_.clear();
    payload_.clear();
    scratch_.clear();
}

expected<checkpoint_to_load_t> checkpoint_to_load_t::make(sz::string_view_t path, pinned_alloc_t allocator) noexcept {
    using loaded_t = expected<checkpoint_to_load_t>;
    expected<checkpoint_file_t> file = checkpoint_file_t::make(path, false);
    if (!file) return loaded_t::fail(std::move(file));

    // The length prefix is attacker-controlled bytes until proven otherwise, and the file's own extent
    // is the only honest bound on it: `fstat` first, and a header the file can't hold is corruption.
    struct ::stat file_stat {};
    if (::fstat(file.value_.get(), &file_stat) != 0)
        return loaded_t::fail(status_t::io_error_k, "fstat failed: {}", path);
    u64_t const file_bytes = static_cast<u64_t>(file_stat.st_size);
    u64_t header_length = 0;
    if (file_bytes < sizeof(header_length))
        return loaded_t::fail(status_t::invalid_input_k, "{} is shorter than a length prefix", path);
    if (expected_t read = read_at_(file.value_, 0, &header_length, sizeof(header_length)); !read)
        return loaded_t::fail(std::move(read));
    if (header_length == 0 || header_length > file_bytes - sizeof(header_length))
        return loaded_t::fail(status_t::invalid_input_k, "{} declares a {}-byte header, the file holds {}", path,
                              header_length, file_bytes);
    u64_t const payload_bytes = file_bytes - sizeof(header_length) - header_length;
    if (payload_bytes % sizeof(bf16_t) != 0)
        return loaded_t::fail(status_t::invalid_input_k, "{} carries a torn payload ({} bytes)", path, payload_bytes);

    // The hoisted scratch holds the header bytes up front and the yyjson parse pool behind them — one
    // allocation for the whole parse, owned by the container.
    checkpoint_to_load_t loaded;
    loaded.payload_ = array<bf16_t, pinned_alloc<bf16_t>> {pinned_alloc<bf16_t> {allocator}};
    std::size_t const pool_bytes = yyjson_read_max_memory_usage(header_length, 0);
    if (loaded.scratch_.try_resize_uninitialized(header_length + pool_bytes) != status_t::success_k)
        return loaded_t::fail(status_t::bad_alloc_k, "header staging failed ({} bytes)", header_length + pool_bytes);
    if (expected_t read = read_at_(file.value_, sizeof(header_length), loaded.scratch_.data(), header_length); !read)
        return loaded_t::fail(std::move(read));
    yyjson_alc pool;
    if (!yyjson_alc_pool_init(&pool, loaded.scratch_.data() + header_length, pool_bytes))
        return loaded_t::fail(status_t::bad_alloc_k, "parse pool setup failed ({} bytes)", pool_bytes);
    std::span<char> const header {loaded.scratch_.data(), header_length};
    if (expected_t parsed = read_header_(header, &pool, payload_bytes, loaded.entries_, loaded.step_); !parsed)
        return loaded_t::fail(std::move(parsed));

    // The whole payload in one read, straight into pinned pages — the caller's uploads DMA from here.
    if (loaded.payload_.try_resize_uninitialized(payload_bytes / sizeof(bf16_t)) != status_t::success_k)
        return loaded_t::fail(status_t::bad_alloc_k, "pinned payload staging failed ({} bytes)", payload_bytes);
    if (expected_t read = read_at_(file.value_, sizeof(header_length) + header_length, loaded.payload_.data(),
                                   payload_bytes);
        !read)
        return loaded_t::fail(std::move(read));
    return loaded_t {std::move(loaded)};
}

nk::tensor_view<bf16_t, 2> checkpoint_to_load_t::operator[](sz::string_view_t layer_name) const noexcept {
    for (checkpoint_entry_t const &entry : entries_)
        if (entry.name.view() == layer_name)
            return nk::tensor_view<bf16_t, 2> {payload_.data() + entry.scalar_offset, entry.rows, entry.cols};
    return {};
}
