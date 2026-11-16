/**
 *  @file src/encoder/vram_kernels_apple.metal
 *  @author Ash Vardanian
 *  @date November 16, 2026
 *  @brief The Apple kernels of the tower steps NumKong has no kernel for, in MSL, which
 *      @c vram_entries_apple.cpp embeds after NumKong's MSL and compiles on each device once.
 *
 *  The Metal twin of @c vram_kernels_amd.hip , kernel for kernel, each running the body of
 *  @ref tower_kernels.cuh of the same name, restated here as MSL includes no host header. Every
 *  kernel takes one argument record at buffer zero, laid out as the host unit's record of the same
 *  name: the device address of every view first, in the order the views are named, then a
 *  @ref view_t of each view's extents and byte strides. The addresses lie in shared buffers the
 *  device's context records, which every launch makes resident, as NumKong's kernels take theirs.
 *  BFloat16 and E4M3 are read and written through NumKong's @c nk::bf16_t and @c nk::e4m3_t ,
 *  which round as its host casts do, and every division and rotation is the precise one, as the
 *  library compiles with fast math on. Two kernels beyond the HIP bundle copy and zero bytes, as a
 *  HIP stream queues those through its runtime while a Metal queue needs a kernel for each.
 */

namespace unum {
namespace uform {
namespace encoder {

/** The axes a view spans at most, as many as the patches' @b [rows,spatial,channels] . */
constant constexpr uint view_axes_k = 3;

/** One view as a kernel reads it: its extents, the first zero for an empty view, and its signed
 *  strides in bytes; axes past its rank span one element. */
struct view_t {
    ulong extents[view_axes_k];
    long strides[view_axes_k];
};

/** Where every segment's live cells sit in the batch grid, and where they land in the packed rows,
 *  over the four buffers a layout binds first. */
struct layout_t {
    view_t grid_offsets, lengths, packed_offsets, modality;
};

/** The modalities as @c modality_t numbers them. */
enum modality_t : uchar {
    text_k = 0,
    image_k = 1,
    video_k = 2,
    audio_k = 3,
};

/** E4M3's largest normal magnitude, the one max_abs maps onto, as @c e4m3_range_k . */
constant constexpr float e4m3_range_k = 240.0f;

/** The descale floor keeping a zero row representable, as @c e4m3_descale_floor_k . */
constant constexpr float e4m3_descale_floor_k = 1.0e-12f;

/** Threads of every threadgroup the quantize reduction halves, as @c entries_block_threads_k . */
constant constexpr uint block_threads_k = 256;

/** The byte offset of element @p row of @p view . */
inline long offset_of(constant view_t& view, ulong row) { return long(row) * view.strides[0]; }

/** The byte offset of element @p row , @p column of @p view . */
inline long offset_of(constant view_t& view, ulong row, ulong column) {
    return long(row) * view.strides[0] + long(column) * view.strides[1];
}

/** The byte offset of element @p row , @p column , @p channel of @p view . */
inline long offset_of(constant view_t& view, ulong row, ulong column, ulong channel) {
    return long(row) * view.strides[0] + long(column) * view.strides[1] + long(channel) * view.strides[2];
}

inline uint read_u32(device uchar const* base, long offset) { return *(device uint const*)(base + offset); }
inline ushort read_u16(device uchar const* base, long offset) { return *(device ushort const*)(base + offset); }
inline short read_i16(device uchar const* base, long offset) { return *(device short const*)(base + offset); }
inline float read_f32(device uchar const* base, long offset) { return *(device float const*)(base + offset); }
inline float read_bf16(device uchar const* base, long offset) { return nk::bf16_t::load(base + offset, 0); }

inline void write_u32(device uchar* base, long offset, uint value) { *(device uint*)(base + offset) = value; }
inline void write_u16(device uchar* base, long offset, ushort value) { *(device ushort*)(base + offset) = value; }
inline void write_i16(device uchar* base, long offset, short value) { *(device short*)(base + offset) = value; }
inline void write_f32(device uchar* base, long offset, float value) { *(device float*)(base + offset) = value; }
inline void write_bf16(device uchar* base, long offset, float value) { nk::bf16_t::store(base + offset, 0, value); }
inline void write_e4m3(device uchar* base, long offset, float value) { nk::e4m3_t::store(base + offset, 0, value); }

/** Whether @p view is the empty view a step passes for a term it drops. */
inline bool empty_view(constant view_t& view) { return view.extents[0] == 0; }

/** The angle of within-head pair @p pair for row @p token , from the temporal, height and width
 *  grids @b [positions,pairs] , temporal pairs first, as @c rope_angle . */
inline float rope_angle(uint pair, device uchar const* coords, constant view_t& coords_view, ulong token,
                        device uchar const* temporal, constant view_t& temporal_view, device uchar const* height,
                        constant view_t& height_view, device uchar const* width, constant view_t& width_view) {
    uint const temporal_pairs = uint(temporal_view.extents[1]);
    uint const height_pairs = uint(height_view.extents[1]);
    if (pair < temporal_pairs) {
        ulong const position = ulong(read_i16(coords, offset_of(coords_view, token, 0)));
        return read_f32(temporal, offset_of(temporal_view, position, pair));
    }
    if (pair < temporal_pairs + height_pairs) {
        ulong const position = ulong(read_i16(coords, offset_of(coords_view, token, 1)));
        return read_f32(height, offset_of(height_view, position, pair - temporal_pairs));
    }
    ulong const position = ulong(read_i16(coords, offset_of(coords_view, token, 2)));
    return read_f32(width, offset_of(width_view, position, pair - temporal_pairs - height_pairs));
}

/** This thread's share of row @p token 's largest magnitude over its striped channels, as
 *  @c token_largest , reading each value through @p read . */
template <typename reader_type_>
inline float token_largest(device uchar const* values, constant view_t& view, ulong token, uint lane, uint lanes,
                           reader_type_ read) {
    uint const width = uint(view.extents[1]);
    float largest = 0.0f;
    for (uint channel = lane; channel < width; channel += lanes)
        largest = fmax(largest, fabs(read(values, offset_of(view, token, channel))));
    return largest;
}

/** Quantize row @p token by its own @c max(largest/240,floor) into @p shadow , channels striped,
 *  and return that scale, as @c quantize_token . */
template <typename reader_type_>
inline float quantize_token(device uchar const* values, constant view_t& values_view, device uchar* shadow,
                            constant view_t& shadow_view, ulong token, float largest, uint lane, uint lanes,
                            reader_type_ read) {
    uint const width = uint(values_view.extents[1]);
    float const scale = fmax(precise::divide(largest, e4m3_range_k), e4m3_descale_floor_k);
    float const inverse = precise::divide(1.0f, scale);
    for (uint channel = lane; channel < width; channel += lanes)
        write_e4m3(shadow, offset_of(shadow_view, token, channel),
                   read(values, offset_of(values_view, token, channel)) * inverse);
    return scale;
}

/** Reads a BFloat16 value widened, the quantize bodies' reader of BFloat16 rows. */
struct bf16_reader_t {
    float operator()(device uchar const* base, long offset) const { return read_bf16(base, offset); }
};

/** Reads a Float32 value, the quantize bodies' reader of Float32 rows. */
struct f32_reader_t {
    float operator()(device uchar const* base, long offset) const { return read_f32(base, offset); }
};

/** The body of both quantize kernels: one threadgroup of @ref block_threads_k per token, reducing
 *  the row's largest magnitude in threadgroup memory before quantizing it. */
template <typename reader_type_>
inline void quantize_tokens(device uchar const* values, constant view_t& values_view, device uchar* shadow,
                            constant view_t& shadow_view, device uchar* token_scales,
                            constant view_t& token_scales_view, threadgroup float* partials, uint token, uint lane,
                            uint lanes, reader_type_ read) {
    partials[lane] = token_largest(values, values_view, token, lane, lanes, read);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint partner = block_threads_k / 2; partner != 0; partner /= 2) {
        if (lane < partner) partials[lane] = fmax(partials[lane], partials[lane + partner]);
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    float const scale = quantize_token(values, values_view, shadow, shadow_view, token, partials[0], lane, lanes, read);
    if (lane == 0u) write_f32(token_scales, offset_of(token_scales_view, token), scale);
}

} // namespace encoder
} // namespace uform
} // namespace unum

using namespace unum::uform::encoder;

/** The record of @c uform_fill_layout_kernel . */
struct fill_layout_arguments_t {
    ulong addresses[3];
    view_t lengths, packed_offsets, pool_offsets;
};

/** The record of @c uform_fill_coords_kernel . */
struct fill_coords_arguments_t {
    ulong addresses[7];
    layout_t layout;
    view_t image_shapes, video_shapes, coords;
    uint tube_frames, frame_stride, positions;
};

/** The record of @c uform_gather_bytes_kernel . */
struct gather_bytes_arguments_t {
    ulong addresses[7];
    layout_t layout;
    view_t tokens, table, rows;
};

/** The record of @c uform_stage_patches_kernel . */
struct stage_patches_arguments_t {
    ulong addresses[8];
    layout_t layout;
    view_t pixels, mean, inverse_std, patches;
    uint target;
};

/** The record of @c uform_stage_frames_kernel . */
struct stage_frames_arguments_t {
    ulong addresses[6];
    layout_t layout;
    view_t samples, frames;
};

/** The record of @c uform_mrope_kernel . */
struct mrope_arguments_t {
    ulong addresses[5];
    view_t values, temporal, height, width, coords;
    uint heads, head_width, role_offset;
};

/** The record of both quantize kernels. */
struct quantize_arguments_t {
    ulong addresses[3];
    view_t values, shadow, token_scales;
};

/** The record of @c uform_rescale_tokens_kernel . */
struct rescale_arguments_t {
    ulong addresses[4];
    view_t products, token_scales, residual, values;
    float weight_descale;
};

/** The record of the copy and zero kernels. */
struct bytes_arguments_t {
    ulong addresses[2];
    ulong bytes;
};

/** __grid_size__ @c [1,1,1] , one thread: the packed layout's prefix sums, serially, as
 *  @c fill_packed_layout . */
kernel void uform_fill_layout_kernel(constant fill_layout_arguments_t& arguments [[buffer(0)]],
                                     uint lane [[thread_position_in_grid]]) {
    device uchar const* const lengths = reinterpret_cast<device uchar const*>(arguments.addresses[0]);
    device uchar* const packed_offsets = reinterpret_cast<device uchar*>(arguments.addresses[1]);
    device uchar* const pool_offsets = reinterpret_cast<device uchar*>(arguments.addresses[2]);
    if (lane != 0u) return;
    ulong const segments = arguments.lengths.extents[0];
    uint offset = 0;
    for (ulong segment = 0; segment != segments; ++segment) {
        write_u32(packed_offsets, offset_of(arguments.packed_offsets, segment), offset);
        write_u32(pool_offsets, offset_of(arguments.pool_offsets, segment), uint(segment));
        offset += read_u32(lengths, offset_of(arguments.lengths, segment));
    }
    write_u32(packed_offsets, offset_of(arguments.packed_offsets, segments), offset);
    write_u32(pool_offsets, offset_of(arguments.pool_offsets, segments), uint(segments));
}

/** __grid_size__ @c [segments,1,1] , threads stripe each segment's cells: text and audio count
 *  time alone, an image its rows and columns, and a video its frame groups scaled by
 *  @c tube_frames and @c frame_stride beside them, each clamped below @c positions , as
 *  @c fill_coords_segment . */
kernel void uform_fill_coords_kernel(constant fill_coords_arguments_t& arguments [[buffer(0)]],
                                     uint segment [[threadgroup_position_in_grid]],
                                     uint lane [[thread_position_in_threadgroup]],
                                     uint lanes [[threads_per_threadgroup]]) {
    device uchar const* const lengths = reinterpret_cast<device uchar const*>(arguments.addresses[1]);
    device uchar const* const packed_offsets = reinterpret_cast<device uchar const*>(arguments.addresses[2]);
    device uchar const* const modality = reinterpret_cast<device uchar const*>(arguments.addresses[3]);
    device uchar const* const image_shapes = reinterpret_cast<device uchar const*>(arguments.addresses[4]);
    device uchar const* const video_shapes = reinterpret_cast<device uchar const*>(arguments.addresses[5]);
    device uchar* const coords = reinterpret_cast<device uchar*>(arguments.addresses[6]);
    constant layout_t& layout = arguments.layout;
    uchar const kind = modality[offset_of(layout.modality, segment)];
    ulong const target_first = read_u32(packed_offsets, offset_of(layout.packed_offsets, segment));
    uint const length = read_u32(lengths, offset_of(layout.lengths, segment));
    uint const last = arguments.positions - 1u;
    bool const spatial = kind == image_k || kind == video_k;
    bool const sequential = kind == text_k || kind == audio_k;

    // An image is one frame of its columns, and a video's frames hold its rows of columns
    uint columns = 0, per_frame = 0;
    if (kind == image_k) columns = read_u16(image_shapes, offset_of(arguments.image_shapes, segment, 1));
    if (kind == video_k) {
        columns = read_u16(video_shapes, offset_of(arguments.video_shapes, segment, 2));
        per_frame = uint(read_u16(video_shapes, offset_of(arguments.video_shapes, segment, 1))) * columns;
        if (per_frame == 0u) return;
    }
    if (kind == image_k && columns == 0u) return;
    for (uint cell = lane; cell < length; cell += lanes) {
        uint temporal = sequential ? cell : 0u, height = 0, width = 0;
        uint const planar = per_frame ? cell % per_frame : cell;
        if (per_frame) temporal = cell / per_frame * arguments.tube_frames * arguments.frame_stride;
        if (spatial) height = planar / columns, width = planar % columns;
        ulong const row = target_first + cell;
        write_i16(coords, offset_of(arguments.coords, row, 0), short(temporal < last ? temporal : last));
        write_i16(coords, offset_of(arguments.coords, row, 1), short(height < last ? height : last));
        write_i16(coords, offset_of(arguments.coords, row, 2), short(width < last ? width : last));
    }
}

/** __grid_size__ @c [segments,1,1] , threads stripe the hidden channels: one text segment's rows
 *  @c target[target_first+k,:]=table[tokens[source_first+k],:] , as @c gather_bytes_segment . */
kernel void uform_gather_bytes_kernel(constant gather_bytes_arguments_t& arguments [[buffer(0)]],
                                      uint segment [[threadgroup_position_in_grid]],
                                      uint lane [[thread_position_in_threadgroup]],
                                      uint lanes [[threads_per_threadgroup]]) {
    device uchar const* const grid_offsets = reinterpret_cast<device uchar const*>(arguments.addresses[0]);
    device uchar const* const lengths = reinterpret_cast<device uchar const*>(arguments.addresses[1]);
    device uchar const* const packed_offsets = reinterpret_cast<device uchar const*>(arguments.addresses[2]);
    device uchar const* const modality = reinterpret_cast<device uchar const*>(arguments.addresses[3]);
    device uchar const* const tokens = reinterpret_cast<device uchar const*>(arguments.addresses[4]);
    device uchar const* const table = reinterpret_cast<device uchar const*>(arguments.addresses[5]);
    device uchar* const rows = reinterpret_cast<device uchar*>(arguments.addresses[6]);
    constant layout_t& layout = arguments.layout;
    if (modality[offset_of(layout.modality, segment)] != text_k) return;
    ulong const source_first = read_u32(grid_offsets, offset_of(layout.grid_offsets, segment));
    ulong const target_first = read_u32(packed_offsets, offset_of(layout.packed_offsets, segment));
    uint const length = read_u32(lengths, offset_of(layout.lengths, segment));
    uint const hidden = uint(arguments.rows.extents[1]);
    for (uint cell = 0; cell != length; ++cell) {
        ulong const token = read_u16(tokens, offset_of(arguments.tokens, source_first + cell));
        for (uint channel = lane; channel < hidden; channel += lanes)
            write_u16(rows, offset_of(arguments.rows, target_first + cell, channel),
                      read_u16(table, offset_of(arguments.table, token, channel)));
    }
}

/** __grid_size__ @c [segments,1,1] , threads stripe each segment's positions: one pixel segment's
 *  cells @b [length,spatial,channels] , each byte over 255 less its channel's mean, times its
 *  inverse deviation, as @c stage_patches_segment . */
kernel void uform_stage_patches_kernel(constant stage_patches_arguments_t& arguments [[buffer(0)]],
                                       uint segment [[threadgroup_position_in_grid]],
                                       uint lane [[thread_position_in_threadgroup]],
                                       uint lanes [[threads_per_threadgroup]]) {
    device uchar const* const grid_offsets = reinterpret_cast<device uchar const*>(arguments.addresses[0]);
    device uchar const* const lengths = reinterpret_cast<device uchar const*>(arguments.addresses[1]);
    device uchar const* const packed_offsets = reinterpret_cast<device uchar const*>(arguments.addresses[2]);
    device uchar const* const modality = reinterpret_cast<device uchar const*>(arguments.addresses[3]);
    device uchar const* const pixels = reinterpret_cast<device uchar const*>(arguments.addresses[4]);
    device uchar const* const mean = reinterpret_cast<device uchar const*>(arguments.addresses[5]);
    device uchar const* const inverse_std = reinterpret_cast<device uchar const*>(arguments.addresses[6]);
    device uchar* const patches = reinterpret_cast<device uchar*>(arguments.addresses[7]);
    constant layout_t& layout = arguments.layout;
    if (modality[offset_of(layout.modality, segment)] != arguments.target) return;
    ulong const source_first = read_u32(grid_offsets, offset_of(layout.grid_offsets, segment));
    ulong const target_first = read_u32(packed_offsets, offset_of(layout.packed_offsets, segment));
    uint const length = read_u32(lengths, offset_of(layout.lengths, segment));
    uint const spatial = uint(arguments.pixels.extents[1]);
    uint const channels = uint(arguments.pixels.extents[2]);
    ulong const cells = ulong(length) * spatial;
    float const inverse_byte_max = 1.0f / 255.0f;
    for (ulong cell = lane; cell < cells; cell += lanes) {
        uint const position = uint(cell % spatial);
        ulong const row = cell / spatial;
        for (uint channel = 0; channel != channels; ++channel) {
            float const byte = float(pixels[offset_of(arguments.pixels, source_first + row, position, channel)]);
            float const value = (byte * inverse_byte_max - read_f32(mean, offset_of(arguments.mean, channel))) *
                                read_f32(inverse_std, offset_of(arguments.inverse_std, channel));
            write_bf16(patches, offset_of(arguments.patches, target_first + row, position, channel), value);
        }
    }
}

/** __grid_size__ @c [segments,1,1] , threads stripe each segment's samples: one audio segment's raw
 *  frames @b [length,samples] narrowed, as @c stage_frames_segment . */
kernel void uform_stage_frames_kernel(constant stage_frames_arguments_t& arguments [[buffer(0)]],
                                      uint segment [[threadgroup_position_in_grid]],
                                      uint lane [[thread_position_in_threadgroup]],
                                      uint lanes [[threads_per_threadgroup]]) {
    device uchar const* const grid_offsets = reinterpret_cast<device uchar const*>(arguments.addresses[0]);
    device uchar const* const lengths = reinterpret_cast<device uchar const*>(arguments.addresses[1]);
    device uchar const* const packed_offsets = reinterpret_cast<device uchar const*>(arguments.addresses[2]);
    device uchar const* const modality = reinterpret_cast<device uchar const*>(arguments.addresses[3]);
    device uchar const* const samples = reinterpret_cast<device uchar const*>(arguments.addresses[4]);
    device uchar* const frames = reinterpret_cast<device uchar*>(arguments.addresses[5]);
    constant layout_t& layout = arguments.layout;
    if (modality[offset_of(layout.modality, segment)] != audio_k) return;
    ulong const source_first = read_u32(grid_offsets, offset_of(layout.grid_offsets, segment));
    ulong const target_first = read_u32(packed_offsets, offset_of(layout.packed_offsets, segment));
    uint const length = read_u32(lengths, offset_of(layout.lengths, segment));
    uint const width = uint(arguments.samples.extents[1]);
    ulong const cells = ulong(length) * width;
    for (ulong cell = lane; cell < cells; cell += lanes) {
        uint const sample = uint(cell % width);
        ulong const row = cell / width;
        write_bf16(frames, offset_of(arguments.frames, target_first + row, sample),
                   read_f32(samples, offset_of(arguments.samples, source_first + row, sample)));
    }
}

/** __grid_size__ grid-stride over channel pairs: each pair's sine and cosine turn the NeoX
 *  split-half pair @c (k,k+head_width/2) of every head from @c role_offset , as @c rope_rotate with
 *  a sign of one. */
kernel void uform_mrope_kernel(constant mrope_arguments_t& arguments [[buffer(0)]],
                               uint first [[thread_position_in_grid]], uint stride [[threads_per_grid]]) {
    device uchar* const values = reinterpret_cast<device uchar*>(arguments.addresses[0]);
    device uchar const* const temporal = reinterpret_cast<device uchar const*>(arguments.addresses[1]);
    device uchar const* const height = reinterpret_cast<device uchar const*>(arguments.addresses[2]);
    device uchar const* const width = reinterpret_cast<device uchar const*>(arguments.addresses[3]);
    device uchar const* const coords = reinterpret_cast<device uchar const*>(arguments.addresses[4]);
    uint const pairs_per_head = arguments.head_width >> 1u;
    ulong const total_pairs = ulong(pairs_per_head) * arguments.values.extents[0];
    for (ulong pair_index = first; pair_index < total_pairs; pair_index += stride) {
        ulong const token = pair_index / pairs_per_head;
        uint const pair = uint(pair_index % pairs_per_head);
        float const angle = rope_angle(pair, coords, arguments.coords, token, temporal, arguments.temporal, height,
                                       arguments.height, width, arguments.width);
        float const sine = precise::sin(angle), cosine = precise::cos(angle);
        for (uint head = 0; head != arguments.heads; ++head) {
            uint const channel = arguments.role_offset + head * arguments.head_width + pair;
            long const real_offset = offset_of(arguments.values, token, channel);
            long const imaginary_offset = offset_of(arguments.values, token, channel + pairs_per_head);
            float const real = read_bf16(values, real_offset);
            float const imaginary = read_bf16(values, imaginary_offset);
            write_bf16(values, real_offset, real * cosine - imaginary * sine);
            write_bf16(values, imaginary_offset, real * sine + imaginary * cosine);
        }
    }
}

/** __grid_size__ @c [tokens,1,1] with @ref block_threads_k threads, one threadgroup per token. */
kernel void uform_quantize_bf16_tokens_kernel(constant quantize_arguments_t& arguments [[buffer(0)]],
                                              uint token [[threadgroup_position_in_grid]],
                                              uint lane [[thread_position_in_threadgroup]],
                                              uint lanes [[threads_per_threadgroup]]) {
    device uchar const* const values = reinterpret_cast<device uchar const*>(arguments.addresses[0]);
    device uchar* const shadow = reinterpret_cast<device uchar*>(arguments.addresses[1]);
    device uchar* const token_scales = reinterpret_cast<device uchar*>(arguments.addresses[2]);
    threadgroup float partials[block_threads_k];
    quantize_tokens(values, arguments.values, shadow, arguments.shadow, token_scales, arguments.token_scales, partials,
                    token, lane, lanes, bf16_reader_t {});
}

/** __grid_size__ @c [tokens,1,1] with @ref block_threads_k threads, one threadgroup per token. */
kernel void uform_quantize_f32_tokens_kernel(constant quantize_arguments_t& arguments [[buffer(0)]],
                                             uint token [[threadgroup_position_in_grid]],
                                             uint lane [[thread_position_in_threadgroup]],
                                             uint lanes [[threads_per_threadgroup]]) {
    device uchar const* const values = reinterpret_cast<device uchar const*>(arguments.addresses[0]);
    device uchar* const shadow = reinterpret_cast<device uchar*>(arguments.addresses[1]);
    device uchar* const token_scales = reinterpret_cast<device uchar*>(arguments.addresses[2]);
    threadgroup float partials[block_threads_k];
    quantize_tokens(values, arguments.values, shadow, arguments.shadow, token_scales, arguments.token_scales, partials,
                    token, lane, lanes, f32_reader_t {});
}

/** __grid_size__ grid-stride over cells: each product times the descale and its token's scale,
 *  plus the residual, an empty @c token_scales or @c residual dropping its term, as
 *  @c rescale_tokens_cells . */
kernel void uform_rescale_tokens_kernel(constant rescale_arguments_t& arguments [[buffer(0)]],
                                        uint first [[thread_position_in_grid]], uint stride [[threads_per_grid]]) {
    device uchar const* const products = reinterpret_cast<device uchar const*>(arguments.addresses[0]);
    device uchar const* const token_scales = reinterpret_cast<device uchar const*>(arguments.addresses[1]);
    device uchar const* const residual = reinterpret_cast<device uchar const*>(arguments.addresses[2]);
    device uchar* const values = reinterpret_cast<device uchar*>(arguments.addresses[3]);
    uint const width = uint(arguments.values.extents[1]);
    ulong const total = arguments.values.extents[0] * width;
    bool const scaled = !empty_view(arguments.token_scales), summed = !empty_view(arguments.residual);
    for (ulong cell = first; cell < total; cell += stride) {
        ulong const token = cell / width;
        uint const channel = uint(cell % width);
        float value = read_f32(products, offset_of(arguments.products, token, channel)) * arguments.weight_descale;
        if (scaled) value *= read_f32(token_scales, offset_of(arguments.token_scales, token));
        if (summed) value += read_bf16(residual, offset_of(arguments.residual, token, channel));
        write_bf16(values, offset_of(arguments.values, token, channel), value);
    }
}

/** __grid_size__ grid-stride over bytes: @c target[k]=source[k] . */
kernel void uform_copy_bytes_kernel(constant bytes_arguments_t& arguments [[buffer(0)]],
                                    uint first [[thread_position_in_grid]], uint stride [[threads_per_grid]]) {
    device uchar* const target = reinterpret_cast<device uchar*>(arguments.addresses[0]);
    device uchar const* const source = reinterpret_cast<device uchar const*>(arguments.addresses[1]);
    for (ulong index = first; index < arguments.bytes; index += stride) target[index] = source[index];
}

/** __grid_size__ grid-stride over bytes: @c target[k]=0 . */
kernel void uform_zero_bytes_kernel(constant bytes_arguments_t& arguments [[buffer(0)]],
                                    uint first [[thread_position_in_grid]], uint stride [[threads_per_grid]]) {
    device uchar* const target = reinterpret_cast<device uchar*>(arguments.addresses[0]);
    for (ulong index = first; index < arguments.bytes; index += stride) target[index] = 0;
}
