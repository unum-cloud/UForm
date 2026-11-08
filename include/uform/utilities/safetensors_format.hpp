/**
 *  @file include/uform/utilities/safetensors_format.hpp
 *  @author Ash Vardanian
 *  @date July 3, 2026
 *  @brief Safetensors checkpoints: one file, a yyjson-built JSON header of names / dtypes / shapes
 *      / offsets, then the raw little-endian tensor bytes — readable by the whole Python ecosystem,
 *      written and read here through NumKong views with no exceptions and no vendor containers.
 *
 *  @section checkpoint_containers Two containers, two directions
 *
 *  The checkpoint types are the containers — no caller-side table of views to materialize. Both are
 *  host-only: a device-resident model downloads into
 *  @ref unum::uform::utilities::checkpoint_to_save_t and uploads out of
 *  @ref unum::uform::utilities::checkpoint_to_load_t with its own copies.
 *  @ref unum::uform::utilities::checkpoint_to_save_t snapshots each inserted tensor into its
 *  payload immediately, so training may keep mutating the source; @c checkpoint_to_save_t::save is
 *  then pure host I/O. @ref unum::uform::utilities::checkpoint_to_load_t maps a whole file once and
 *  hands out zero-copy by-name views into the mapping — the caller copies what it wants and decides
 *  what a missing name means, which is the whole warm-start policy: a masked-pretraining checkpoint
 *  fills the shared encoder, the objective heads come back empty, and the caller re-seeds them.
 *
 *  Each cell's @c fill spells its own weights' names — the tower its structure ( @c embedding ,
 *  @c blocks.{i}.qkv , … ), each objective head its instance name — and the composition root adds
 *  only its @c model. prefix at both ends. Shapes are the true @b [rows,cols] extents of the rank-2
 *  views. Weights are BFloat16; the training state a resume continues from — the @c optimizer.
 *  moments and masters and the @c contrastive. temperatures — is Float32 beside them, and inference
 *  reads only @c model. names.
 *
 *  @section checkpoint_metadata Metadata is the geometry
 *
 *  Every value of the header's metadata object is a string, as the format demands. Training writes
 *  the step and the whole geometry a reader needs to rebuild the model — @c depth , @c hidden ,
 *  @c ffn_expansion , @c head_width , @c rms_epsilon , the rotary split and base, the @c context ,
 *  and each modality's patch and pixel statistics, lists comma-separated — so inference derives
 *  everything from the file alone.
 */
#pragma once

#include <cstddef> // `std::size_t`

#include <span> // `std::span` — a caller-owned checkpoint image

#include <numkong/types.h>                 // `nk_bf16_t`
#include <smashtable/basic_hash_table.hpp> // `hash_map` — every entry's position by its name's hash

#include "uform/utilities/types.hpp" // `expected`, `expected_t`, `status_t`, `sz::string_t`, `st::basic_vector`

namespace unum::uform::utilities {

/** The scalar types a checkpoint stores: BFloat16 weights, and the Float32 training
 *  state beside them. */
enum class checkpoint_dtype_t : u32_t {
    bf16_k = 0,
    f32_k = 1,
};

/** One indexed checkpoint entry: the owned name, its scalar type, the logical
 *  @b [rows,cols] extents (vectors are @b [1,cols] ), and where the tensor's bytes start
 *  inside the owner's payload. */
struct checkpoint_entry_t {

    /** The owned name text — short names ride StringZilla's inline buffer. */
    sz::string_t name {};

    /** The scalar type the payload holds this tensor in. */
    checkpoint_dtype_t dtype = checkpoint_dtype_t::bf16_k;

    /** Leading extent; @c 1 for vector parameters. */
    std::size_t rows = 0;

    /** Trailing extent. */
    std::size_t cols = 0;

    /** Byte offset inside the owner's payload, a multiple of the scalar size. */
    std::size_t byte_offset = 0;
};

/** One pair of the header's metadata object, both strings as the format stores them. */
struct checkpoint_metadata_t {
    sz::string_t key {};
    sz::string_t value {};
};

/**
 *  @brief The outbound checkpoint container: @ref insert snapshots one host tensor into the
 *      payload, so the source is free to change the moment it returns, and @ref metadata records
 *      the geometry beside it.
 *
 *  @ref save is pure host I/O. All buffers grow lazily and keep their backing across cadences;
 *  @ref reserve hoists the growth entirely. Lead-rank-only by the caller's discipline.
 */
struct checkpoint_to_save_t {

    checkpoint_to_save_t() noexcept = default;
    checkpoint_to_save_t(checkpoint_to_save_t const &) = delete;
    checkpoint_to_save_t &operator=(checkpoint_to_save_t const &) = delete;
    checkpoint_to_save_t(checkpoint_to_save_t &&) noexcept = default;
    checkpoint_to_save_t &operator=(checkpoint_to_save_t &&) noexcept = default;
    ~checkpoint_to_save_t() noexcept = default;

    /** Optional hoist: pre-size the payload to @p total_bytes and the entry table to
     *  @p tensor_count , so no @ref insert ever reallocates. */
    [[nodiscard]] expected_t reserve(u64_t total_bytes, u64_t tensor_count) noexcept;

    /** Append one entry, copying @p data into the payload @b now . @p name is copied into owned
     *  text; duplicates are rejected loudly. */
    [[nodiscard]] expected_t insert(sz::string_view_t name, nk::tensor_view<nk_bf16_t, 2> data) noexcept;

    /** Append one Float32 entry like the BFloat16 one. The payload packs entries without gaps,
     *  so a Float32 entry following an odd count of BFloat16 scalars is refused: insert the
     *  Float32 state first. */
    [[nodiscard]] expected_t insert(sz::string_view_t name, nk::tensor_view<f32_t, 2> data) noexcept;

    /** Record the metadata value of @p key , replacing an earlier one. */
    [[nodiscard]] expected_t metadata(sz::string_view_t key, sz::string_view_t value) noexcept;

    /** Write the accumulated entries and metadata to @p path in the safetensors layout — the JSON
     *  header space-padded to an 8-byte boundary like Hugging Face's own writer, then the whole
     *  contiguous payload in one shot. The file is written beside @p path and renamed over it, so a
     *  run killed mid-write keeps its previous checkpoint. */
    [[nodiscard]] expected_t save(sz::string_view_t path) noexcept;

    /** Drop the entries and metadata, keep every buffer's backing — the next cadence re-inserts and
     *  re-saves allocation-free. */
    void clear() noexcept;

    /** How many tensors the container currently holds. */
    [[nodiscard]] std::size_t size() const noexcept { return entries_.size(); }

  private:

    /** The shared body of both @ref insert overloads. */
    [[nodiscard]] expected_t insert_(sz::string_view_t name, checkpoint_dtype_t dtype, bytes_view_t data,
                                     std::size_t rows, std::size_t cols) noexcept;

    /** Every inserted tensor's bytes back-to-back in insertion order — @c save streams them to disk
     *  in a single write. */
    st::basic_vector<b8_t> payload_ {};

    /** The name / shape / offset record per inserted tensor. */
    st::basic_vector<checkpoint_entry_t> entries_ {};

    /** Each entry's position in @c entries_ by its name's hash, for the duplicate check. */
    st::hash_map<u64_t, u32_t> index_ {};

    /** The metadata pairs, in insertion order. */
    st::basic_vector<checkpoint_metadata_t> metadata_ {};

    /** The yyjson pool @c save serializes the header in — grown lazily, settled after
     *  the first save. */
    st::basic_vector<char> scratch_ {};
};

/** A read-only file mapping, unmapped on destruction; empty for a checkpoint viewed
 *  in caller memory. */
struct checkpoint_mapping_t {
    checkpoint_mapping_t() noexcept = default;
    checkpoint_mapping_t(checkpoint_mapping_t const &) = delete;
    checkpoint_mapping_t &operator=(checkpoint_mapping_t const &) = delete;
    checkpoint_mapping_t(checkpoint_mapping_t &&other) noexcept;
    checkpoint_mapping_t &operator=(checkpoint_mapping_t &&other) noexcept;
    ~checkpoint_mapping_t() noexcept;

    /** Map @p path read-only, whole. */
    [[nodiscard]] static expected<checkpoint_mapping_t> make(sz::string_view_t path) noexcept;

    /** The mapped bytes. */
    [[nodiscard]] bytes_view_t bytes() const noexcept { return bytes_; }

  private:
    bytes_span_t bytes_ {};
};

/**
 *  @brief The inbound checkpoint container: @ref make maps a safetensors file once, or
 *      @ref view adopts one already in memory, and @ref operator[] hands out zero-copy
 *      read-only views by name.
 *
 *  The header is parsed and validated up front: its declared length is bounded by the image, and
 *  every entry's @c data_offsets by its shape and the payload. Views stay valid while the container
 *  lives; strictness is the caller's policy, not this container's.
 */
struct checkpoint_to_load_t {

    checkpoint_to_load_t() noexcept = default;
    checkpoint_to_load_t(checkpoint_to_load_t const &) = delete;
    checkpoint_to_load_t &operator=(checkpoint_to_load_t const &) = delete;
    checkpoint_to_load_t(checkpoint_to_load_t &&) noexcept = default;
    checkpoint_to_load_t &operator=(checkpoint_to_load_t &&) noexcept = default;
    ~checkpoint_to_load_t() noexcept = default;

    /** Map @p path read-only and index it; the mapping lives as long as the container. */
    [[nodiscard]] static expected<checkpoint_to_load_t> make(sz::string_view_t path) noexcept;

    /** Index a checkpoint image the caller owns, which must outlive the container. */
    [[nodiscard]] static expected<checkpoint_to_load_t> view(bytes_view_t buffer) noexcept;

    /** By-name lookup: a read-only rank-2 view into the image, or an @b empty view when
     *  the file misses @p name (or holds it in a dtype other than BFloat16) — check
     *  @c .empty() and decide. */
    [[nodiscard]] nk::tensor_view<nk_bf16_t, 2> operator[](sz::string_view_t name) const noexcept;

    /** The Float32 twin of @ref operator[] , empty when the file misses @p name or holds it
     *  in another dtype. */
    [[nodiscard]] nk::tensor_view<f32_t, 2> float32(sz::string_view_t name) const noexcept;

    /** The metadata value of @p key , or an empty view when the file carries none. */
    [[nodiscard]] sz::string_view_t metadata(sz::string_view_t key) const noexcept;

    /** Every metadata pair, in file order, for a tool copying the checkpoint whole. */
    [[nodiscard]] std::span<checkpoint_metadata_t const> metadata() const noexcept { return metadata_; }

    /** Every indexed tensor's record, in file order, for a tool copying the checkpoint whole. */
    [[nodiscard]] std::span<checkpoint_entry_t const> entries() const noexcept { return entries_; }

    /** The @c step metadata the file was saved at; @c 0 when absent. */
    [[nodiscard]] u64_t step() const noexcept;

    /** How many tensors the container indexed. */
    [[nodiscard]] std::size_t size() const noexcept { return entries_.size(); }

  private:

    /** The file mapping when @ref make opened one; empty for @ref view . */
    checkpoint_mapping_t mapping_ {};

    /** The payload bytes, inside the mapping or the caller's image. */
    bytes_view_t payload_ {};

    /** The name / dtype / shape / offset record per indexed tensor. */
    st::basic_vector<checkpoint_entry_t> entries_ {};

    /** Each entry's position in @c entries_ by its name's hash, for @c find_ . */
    st::hash_map<u64_t, u32_t> index_ {};

    /** The entry named @p name in @p dtype , or null. */
    [[nodiscard]] checkpoint_entry_t const *find_(sz::string_view_t name, checkpoint_dtype_t dtype) const noexcept;

    /** The metadata pairs. */
    st::basic_vector<checkpoint_metadata_t> metadata_ {};
};

} // namespace unum::uform::utilities
