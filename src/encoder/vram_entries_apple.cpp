/**
 *  @file src/encoder/vram_entries_apple.cpp
 *  @author Ash Vardanian
 *  @date November 16, 2026
 *  @brief The Apple entries kit @ref vram_apple_entries_kit_t , the Metal products, attention and
 *      element-wise kits, and the launches of every kernel they dispatch.
 *
 *  The twin of @c vram_entries_amd.cpp . Metal compiles on the device, so the kernels travel as
 *  source: NumKong's @c types.metal , its dots, attention, each and cast kernels, then UForm's own,
 *  which each device builds into one library at its first launch, under Metal 3.1. The Apple9 and
 *  Apple10 tiles and passes come from two more libraries, the Apple10 one under Metal 4.0.
 *
 *  Every kernel takes one argument record, its operands' GPU addresses first. A device view holds
 *  a GPU address, as a CUDA one holds a device pointer; a view of pinned pages holds their host
 *  address, which a launch turns into the GPU address of the same bytes. Both resolve through the
 *  buffers the device's context records, and a launch makes each buffer it touches resident. A
 *  copy or fill between device buffers runs as a kernel; one touching host memory waits for the
 *  stream's work and runs on the host, as memory is unified and no kernel is then needed.
 */
#include <cstddef> // `std::size_t`
#include <cstring> // `std::memcpy`, `std::memset`, `std::strncpy`

#include <forkunion/locks.hpp> // `fu::spin_mutex_t` - a context's lock

#include "uform/encoder/vram_kits.hpp" // `vram_apple_entries_kit_t`, the Metal kits, `entries_block_threads_k`

#include "lib.hpp" // `gpu_drivers`

namespace fu = ashvardanian::forkunion;
namespace ue = unum::uform::encoder;
namespace util = unum::uform::utilities;
using namespace ue;
using util::b8_t;
using util::bytes_span_t;
using util::bytes_view_t;
using util::divide_round_up;
using util::i64_t;
using util::metal_buffer_t;
using util::metal_context_t;
using util::metal_launch_t;
using util::metal_send;
using util::metal_size_t;

#pragma region Sources

#if defined(__clang__)
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wc23-extensions"
#endif

/** The MSL of NumKong's generic kernels and of UForm's own, compiled once per device. */
static char const metal_source_k[] = {
#embed "numkong/types.metal" suffix(, )
#embed "numkong/dots/metal.metal" suffix(, )
#embed "numkong/attention/metal.metal" suffix(, )
#embed "numkong/each/metal.metal" suffix(, )
#embed "numkong/cast/metal.metal" suffix(, )
#embed "vram_kernels_apple.metal" suffix(, 0)
};

#if NUMKONG_TARGET_APPLE9

/** The MSL of NumKong's Apple9 tiles and passes, after the generic ones they share helpers with. */
static char const apple9_source_k[] = {
#embed "numkong/types.metal" suffix(, )
#embed "numkong/dots/metal.metal" suffix(, )
#embed "numkong/attention/metal.metal" suffix(, )
#embed "numkong/dots/apple9.metal" suffix(, )
#embed "numkong/attention/apple9.metal" suffix(, 0)
};
#endif

#if NUMKONG_TARGET_APPLE10

/** The MSL of NumKong's Apple10 tiles and passes over Metal 4 tensors. */
static char const apple10_source_k[] = {
#embed "numkong/types.metal" suffix(, )
#embed "numkong/dots/metal.metal" suffix(, )
#embed "numkong/attention/metal.metal" suffix(, )
#embed "numkong/dots/apple10.metal" suffix(, )
#embed "numkong/attention/apple10.metal" suffix(, 0)
};
#endif

#if defined(__clang__)
#pragma clang diagnostic pop
#endif

/** @c MTLLanguageVersion3_1 , the first with @c bfloat , and @c MTLLanguageVersion4_0 , the first
 *  with tensor operations, which only the Apple10 library takes. */
static constexpr std::size_t language_3_1_k = (3u << 16) | 1u, language_4_0_k = (4u << 16) | 0u;

/** UForm's own kernels, in the order of @ref kernel_names_k . */
enum class kernel_t : std::size_t {
    fill_layout_k,
    fill_coords_k,
    gather_bytes_k,
    stage_patches_k,
    stage_frames_k,
    mrope_k,
    quantize_bf16_tokens_k,
    quantize_f32_tokens_k,
    rescale_tokens_k,
    copy_bytes_k,
    zero_bytes_k,
    count_k,
};

static char const* const kernel_names_k[] = {
    "uform_fill_layout_kernel",          "uform_fill_coords_kernel",         "uform_gather_bytes_kernel",
    "uform_stage_patches_kernel",        "uform_stage_frames_kernel",        "uform_mrope_kernel",
    "uform_quantize_bf16_tokens_kernel", "uform_quantize_f32_tokens_kernel", "uform_rescale_tokens_kernel",
    "uform_copy_bytes_kernel",           "uform_zero_bytes_kernel",
};

static_assert(sizeof(kernel_names_k) / sizeof(kernel_names_k[0]) == std::size_t(kernel_t::count_k),
              "Every kernel has a name");

/** Most threadgroups NumKong's kernels take along one axis of their grids. */
static constexpr std::size_t grid_axis_max_k = 65535;

/** Bytes of the header a NumKong pack of B opens with, before its padded rows. */
static constexpr std::size_t packed_header_bytes_k = 64;

/** log₂ e, which the attention kernels fold into their scale to run on @c exp2 . */
static constexpr f32_t log2_e_k = 1.4426950408889634f;

#pragma endregion Sources

#pragma region Records

/** The axes a view spans at most, as @c view_axes_k in the kernels. */
static constexpr std::size_t view_axes_k = 3;

/** One view as a kernel reads it, laid out as the kernels' @c view_t : its extents, the first zero
 *  for an empty view, and its signed strides in bytes; axes past its rank span one element. */
struct metal_view_t {
    u64_t extents[view_axes_k] = {};
    i64_t strides[view_axes_k] = {};
};

/** The four views of a @ref segment_layout_t , as the kernels' @c layout_t . */
struct metal_layout_t {
    metal_view_t grid_offsets, lengths, packed_offsets, modality;
};

/** The records of UForm's kernels, laid out as the kernels' records of the same names: the GPU
 *  addresses of the views, in the order they are named, then the views themselves. */
struct fill_layout_arguments_t {
    u64_t addresses[3] = {};
    metal_view_t lengths, packed_offsets, pool_offsets;
};

struct fill_coords_arguments_t {
    u64_t addresses[7] = {};
    metal_layout_t layout;
    metal_view_t image_shapes, video_shapes, coords;
    u32_t tube_frames = 0, frame_stride = 0, positions = 0;
};

struct gather_bytes_arguments_t {
    u64_t addresses[7] = {};
    metal_layout_t layout;
    metal_view_t tokens, table, rows;
};

struct stage_patches_arguments_t {
    u64_t addresses[8] = {};
    metal_layout_t layout;
    metal_view_t pixels, mean, inverse_std, patches;
    u32_t target = 0;
};

struct stage_frames_arguments_t {
    u64_t addresses[6] = {};
    metal_layout_t layout;
    metal_view_t samples, frames;
};

struct mrope_arguments_t {
    u64_t addresses[5] = {};
    metal_view_t values, temporal, height, width, coords;
    u32_t heads = 0, head_width = 0, role_offset = 0;
};

struct quantize_arguments_t {
    u64_t addresses[3] = {};
    metal_view_t values, shadow, token_scales;
};

struct rescale_arguments_t {
    u64_t addresses[4] = {};
    metal_view_t products, token_scales, residual, values;
    f32_t weight_descale = 0;
};

struct bytes_arguments_t {
    u64_t addresses[2] = {};
    u64_t bytes = 0;
};

static_assert(sizeof(metal_view_t) == 48 && sizeof(metal_layout_t) == 192, "Views travel as the kernels read them");
static_assert(sizeof(fill_layout_arguments_t) == 168 && sizeof(fill_coords_arguments_t) == 408 &&
                  sizeof(gather_bytes_arguments_t) == 392 && sizeof(stage_patches_arguments_t) == 456 &&
                  sizeof(stage_frames_arguments_t) == 336 && sizeof(mrope_arguments_t) == 296 &&
                  sizeof(quantize_arguments_t) == 168 && sizeof(rescale_arguments_t) == 232 &&
                  sizeof(bytes_arguments_t) == 24,
              "Records travel as the kernels read them");

/** The records of NumKong's kernels, named and laid out field for field as its MSL declares them:
 *  the GPU addresses first, an absent operand's 0, then the extents and strides. */
struct nk_cross_pack_arguments_metal_t {
    u64_t b = 0, b_packed = 0, scales = 0, tensor_scale = 0;
    u64_t b_stride = 0, depth_bytes = 0, row_bytes = 0;
    nk_capability_t capability = 0;
    u32_t column_count = 0, depth = 0, depth_padded_values = 0, columns_begin = 0, columns_end = 0;
    u64_t scales_stride = 0;
    u32_t tensor_present = 0;
};

struct nk_cross_arguments_metal_t {
    u64_t a = 0, b = 0, c = 0, a_scales = 0, b_scales = 0, a_tensor_scale = 0, b_tensor_scale = 0;
    u64_t a_stride = 0, b_stride = 0, c_stride = 0;
    u32_t rows_begin = 0, rows_end = 0, column_count = 0, depth = 0;
    u32_t upper_triangle = 0;
    u64_t a_scales_stride = 0, b_scales_stride = 0;
    u32_t a_tensor_present = 0, b_tensor_present = 0;
};

struct nk_attention_pack_arguments_metal_t {
    u64_t keys = 0, values = 0, key_offsets = 0, key_lengths = 0, packed = 0, key_scales = 0, value_scales = 0,
          key_tensor_scale = 0, value_tensor_scale = 0;
    u64_t key_value_head_count = 0, depth = 0, head_bytes = 0, segments = 0, key_stride = 0, value_stride = 0,
          tasks_begin = 0, tasks_end = 0;
    u64_t key_bytes = 0, value_bytes = 0, packed_bytes = 0, lengths_bytes = 0, capability = 0;
    u64_t code_bytes = 0, key_scales_stride = 0, value_scales_stride = 0, key_scales_bytes = 0, value_scales_bytes = 0;
    u32_t key_tensor_present = 0, value_tensor_present = 0;
};

struct nk_attention_arguments_metal_t {
    u64_t queries = 0, packed = 0, output = 0, query_offsets = 0, log_sum_exp = 0, split = 0, query_scales = 0,
          query_tensor_scale = 0;
    u64_t heads = 0, kv_heads = 0, depth = 0, query_tokens = 0, query_stride = 0, output_stride = 0, keys_before = 0,
          keys_after = 0;
    u64_t tasks_begin = 0, tasks_end = 0, packed_bytes = 0, offsets_bytes = 0, log_sum_exp_bytes = 0, capability = 0;
    f32_t scale2 = 0;
    u32_t query_tensor_present = 0;
    u64_t query_scales_stride = 0;
};

struct nk_attention_backward_arguments_metal_t {
    u64_t output_gradient = 0, query_gradient = 0, key_gradient = 0, value_gradient = 0;
    nk_attention_arguments_metal_t attention;
    u64_t query_gradient_stride = 0, key_value_gradient_stride = 0, key_value_gradient_bytes = 0;
    f32_t scale = 0;
};

struct nk_attention_rope_arguments_metal_t {
    u64_t x = 0, cosine = 0, sine = 0, y = 0;
    u64_t rows = 0, heads = 0, depth = 0, x_stride = 0, y_stride = 0;
};

struct nk_each_sum_arguments_metal_t {
    u64_t a = 0, b = 0, result = 0;
    u64_t count = 0;
};

struct nk_each_rmsnorm_arguments_metal_t {
    u64_t x = 0, gamma = 0, y = 0;
    u64_t x_stride = 0, y_stride = 0;
    u32_t columns = 0;
    f32_t epsilon = 0;
};

struct nk_each_swiglu_arguments_metal_t {
    u64_t gate = 0, up = 0, y = 0;
    u64_t gate_stride = 0, up_stride = 0, y_stride = 0;
    u32_t columns = 0;
    f32_t gate_scale = 0, output_scale = 0;
};

struct nk_cast_plan_metal_t {
    u64_t from = 0, to = 0;
    u64_t count = 0, units = 0;
    u32_t from_dtype = 0, to_dtype = 0, hub = 0, unit_values = 0;
};

static_assert(sizeof(nk_cross_pack_arguments_metal_t) == 104 && sizeof(nk_cross_arguments_metal_t) == 128 &&
                  sizeof(nk_attention_pack_arguments_metal_t) == 224 && sizeof(nk_attention_arguments_metal_t) == 192 &&
                  sizeof(nk_attention_backward_arguments_metal_t) == 256 &&
                  sizeof(nk_attention_rope_arguments_metal_t) == 72 && sizeof(nk_each_sum_arguments_metal_t) == 32 &&
                  sizeof(nk_each_rmsnorm_arguments_metal_t) == 48 && sizeof(nk_each_swiglu_arguments_metal_t) == 64 &&
                  sizeof(nk_cast_plan_metal_t) == 48,
              "NumKong's records travel as its kernels read them");
static_assert(offsetof(nk_cross_pack_arguments_metal_t, column_count) == 64 &&
                  offsetof(nk_cross_pack_arguments_metal_t, scales_stride) == 88 &&
                  offsetof(nk_cross_arguments_metal_t, a_scales_stride) == 104 &&
                  offsetof(nk_attention_pack_arguments_metal_t, key_value_head_count) == 72 &&
                  offsetof(nk_attention_arguments_metal_t, scale2) == 176 &&
                  offsetof(nk_attention_arguments_metal_t, query_scales_stride) == 184 &&
                  offsetof(nk_attention_backward_arguments_metal_t, query_gradient_stride) == 224 &&
                  offsetof(nk_each_rmsnorm_arguments_metal_t, columns) == 40 &&
                  offsetof(nk_each_swiglu_arguments_metal_t, columns) == 48 &&
                  offsetof(nk_cast_plan_metal_t, count) == 16 && offsetof(nk_cast_plan_metal_t, hub) == 40,
              "NumKong's records place each field where its kernels read it");

/** @p view as its kernel's record holds it. */
template <typename view_type_>
[[nodiscard]] static metal_view_t view_of_(view_type_ const& view) noexcept {
    metal_view_t described;
    if (view.empty()) return described;
    for (std::size_t axis = 0; axis != view_axes_k; ++axis) {
        bool const spanned = axis < view.rank();
        described.extents[axis] = spanned ? static_cast<u64_t>(view.extent(axis)) : 1u;
        described.strides[axis] = spanned ? static_cast<i64_t>(view.stride_bytes(axis)) : 0;
    }
    return described;
}

/** The first byte of @p view , null for an empty one, which its kernel reads nothing of. */
template <typename view_type_>
[[nodiscard]] static void const* first_of_(view_type_ const& view) noexcept {
    return view.empty() ? nullptr : static_cast<void const*>(view.byte_data());
}

/** @p layout as its kernel's record holds it. */
[[nodiscard]] static metal_layout_t layout_of_(segment_layout_t const& layout) noexcept {
    return {view_of_(layout.grid_offsets), view_of_(layout.lengths), view_of_(layout.packed_offsets),
            view_of_(layout.modality)};
}

#pragma endregion Records

#pragma region Launches

/** The context of the Apple GPU @p stream submits to, the first listed for a null stream, or null
 *  for a stream on no listed device. */
[[nodiscard]] static metal_context_t* context_of_(nk_stream_t stream) noexcept {
    util::metal_contexts_t& contexts = uf::gpu_drivers().metal_contexts;
    if (!contexts.count) return nullptr;
    if (!stream) return &contexts.contexts[0];
    void* const device = metal_send(*contexts.contexts[0].api, static_cast<void*>(stream), "device");
    for (std::size_t index = 0; index != contexts.count; ++index)
        if (contexts.contexts[index].device == device) return &contexts.contexts[index];
    return nullptr;
}

/** The queue @p stream names on @p context 's device, its default one for a null stream. */
[[nodiscard]] static void* queue_of_(metal_context_t const& context, nk_stream_t stream) noexcept {
    return stream ? static_cast<void*>(stream) : context.queue;
}

/** Waits for the work committed to @p stream on @p context 's device. */
[[nodiscard]] static nk_status_t drain_(metal_context_t const& context, nk_stream_t stream) noexcept {
    return util::metal_synchronize(*context.api, queue_of_(context, stream)) ? nk_success_k : nk_device_code_mismatch_k;
}

/**
 *  @brief Finds the buffer of @p context holding @p pointer , either a GPU address or a host
 *      address of a shared buffer, and makes it resident for @p launch when open.
 *
 *  Writes the GPU address of the same byte to @p address and the bytes the buffer holds from there
 *  to @p bytes , and answers whether the pointer is the host address, through @p host , or the
 *  host address the GPU address sits at.
 */
[[nodiscard]] static bool resolve_(metal_context_t& context, metal_launch_t const* launch, void const* pointer,
                                   u64_t& address, std::size_t& bytes, b8_t** host = nullptr) noexcept {
    u64_t const wanted = reinterpret_cast<u64_t>(pointer);
    st::unique_lock<fu::spin_mutex_t> guard {context.lock};
    for (metal_buffer_t const& buffer : context.buffers.records()) {
        if (!buffer) continue;
        u64_t const host_first = reinterpret_cast<u64_t>(buffer.host);
        u64_t const device_first = metal_send<u64_t>(*buffer.api, buffer.handle, "gpuAddress");
        u64_t first = 0;
        if (wanted - host_first < buffer.bytes) first = host_first;
        else if (wanted - device_first < buffer.bytes) first = device_first;
        else continue;
        std::size_t const offset = static_cast<std::size_t>(wanted - first);
        address = device_first + offset, bytes = buffer.bytes - offset;
        if (host) *host = buffer.host + offset;
        // The command buffer retains the buffer before the lock lets a free release it
        if (launch && *launch)
            metal_send<void>(*launch->api, launch->encoder, "useResource:usage:", buffer.handle,
                             util::metal_usage_read_write_k);
        return true;
    }
    return false;
}

/** The host address of @p pointer : a GPU address any listed device's buffers hold becomes the
 *  host address of the same byte, and any other pointer is host memory already. */
template <typename byte_type_>
[[nodiscard]] static byte_type_* host_of_(byte_type_* pointer) noexcept {
    util::metal_contexts_t& contexts = uf::gpu_drivers().metal_contexts;
    for (std::size_t index = 0; index != contexts.count; ++index) {
        u64_t address = 0;
        std::size_t bytes = 0;
        b8_t* host = nullptr;
        if (resolve_(contexts.contexts[index], nullptr, pointer, address, bytes, &host))
            return reinterpret_cast<byte_type_*>(host);
    }
    return pointer;
}

/** Whether @p pointer lies in device memory of @p context , as a GPU address of one of its
 *  buffers, rather than in host memory or pinned pages. */
[[nodiscard]] static bool on_device_(metal_context_t& context, void const* pointer) noexcept {
    u64_t address = 0;
    std::size_t bytes = 0;
    return resolve_(context, nullptr, pointer, address, bytes) && address == reinterpret_cast<u64_t>(pointer);
}

/** One call's command buffer on @p stream , and the context of its device. */
struct metal_call_t {
    metal_context_t* context = nullptr;
    metal_launch_t launch;

    /** Opens the call on @p stream 's queue for kernel @p name of @p source under @p language . */
    [[nodiscard]] nk_status_t open(nk_stream_t stream, char const* source, char const* name,
                                   std::size_t language) noexcept {
        context = context_of_(stream);
        if (!context) return nk_missing_gpu_k;
        auto [state, status] = context->pipeline(source, name, language);
        if (!status) return nk_device_code_mismatch_k;
        launch = metal_launch_t {*context->api, queue_of_(*context, stream), state.pipeline};
        return launch ? nk_success_k : nk_device_code_mismatch_k;
    }

    /** The GPU address of @p pointer , resident for the call, into @p address ; 0 for null. */
    [[nodiscard]] bool address(void const* pointer, u64_t& address) noexcept {
        std::size_t bytes = 0;
        return bytes_after(pointer, address, bytes);
    }

    /** The GPU address of @p pointer , resident for the call, and its buffer's bytes past it. */
    [[nodiscard]] bool bytes_after(void const* pointer, u64_t& address, std::size_t& bytes) noexcept {
        address = 0, bytes = 0;
        return !pointer || resolve_(*context, &launch, pointer, address, bytes);
    }

    /** Dispatches @p groups of @p threads with a buffer barrier after them, without committing, so
     *  a next kernel reads what they wrote. */
    void then(metal_size_t groups, metal_size_t threads) noexcept {
        // `MTLBarrierScopeBuffers`
        constexpr std::size_t buffers_scope_k = 1;
        metal_send<void, metal_size_t, metal_size_t>(*launch.api, launch.encoder,
                                                     "dispatchThreadgroups:threadsPerThreadgroup:", groups, threads);
        metal_send<void>(*launch.api, launch.encoder, "memoryBarrierWithScope:", buffers_scope_k);
    }

    /** Switches the encoder to kernel @p name of @p source under @p language . */
    [[nodiscard]] nk_status_t select(char const* source, char const* name,
                                     std::size_t language) noexcept { // vendor-boundary: NumKong's C signature
        auto [state, status] = context->pipeline(source, name, language);
        if (!status) return nk_device_code_mismatch_k;
        metal_send<void>(*launch.api, launch.encoder, "setComputePipelineState:", state.pipeline);
        return nk_success_k;
    }

    /** Binds @p arguments as the record every kernel reads at buffer zero. */
    template <typename arguments_type_>
    void bind(arguments_type_ const& arguments) noexcept {
        launch.bind_bytes(&arguments, sizeof(arguments), 0);
    }

    /** Dispatches @p groups of @p threads and commits the call. */
    [[nodiscard]] nk_status_t submit(metal_size_t groups, metal_size_t threads) noexcept {
        return launch.submit(groups, threads) == util::status_code_t::success_k ? nk_success_k
                                                                                : nk_device_code_mismatch_k;
    }
};

/**
 *  @brief Launches UForm's @p kernel over @p groups threadgroups of @p threads on @p stream , the
 *      GPU addresses of @p operands opening @p arguments .
 *
 *  A null operand, an empty view, takes the address 0, which its kernel never reads. An operand
 *  outside every buffer refuses the launch with @c nk_device_memory_mismatch_k , and an empty grid
 *  queues no work at all.
 */
template <typename arguments_type_, std::size_t count_>
[[nodiscard]] static nk_status_t launch_(kernel_t kernel, std::size_t groups, std::size_t threads, nk_stream_t stream,
                                         void const* const (&operands)[count_], arguments_type_& arguments) noexcept {
    static_assert(sizeof(arguments.addresses) / sizeof(u64_t) == count_, "Every operand has an address");
    if (groups == 0) return nk_success_k;
    metal_call_t call;
    if (nk_status_t const opened = call.open(stream, metal_source_k, kernel_names_k[std::size_t(kernel)],
                                             language_3_1_k);
        opened != nk_success_k)
        return opened;
    for (std::size_t index = 0; index != count_; ++index)
        if (!call.address(operands[index], arguments.addresses[index])) return nk_device_memory_mismatch_k;
    call.bind(arguments);
    return call.submit({groups}, {threads});
}

/** Threadgroups for a grid-stride walk over @p total items. */
[[nodiscard]] static std::size_t stride_blocks_(std::size_t total) noexcept {
    std::size_t const blocks = divide_round_up(total, entries_block_threads_k);
    return blocks == 0 ? 1 : blocks < entries_grid_blocks_max_k ? blocks : entries_grid_blocks_max_k;
}

/** The smaller of @p first and @p second . */
[[nodiscard]] static constexpr std::size_t lesser_(std::size_t first, std::size_t second) noexcept {
    return first < second ? first : second;
}

#pragma endregion Launches

#pragma region Products

/** Values one packed row spans: @p depth rounded up to @p width values, plus one more step when
 *  that lands on a power-of-two byte stride, as NumKong's Metal packs pad their rows. */
[[nodiscard]] static constexpr std::size_t padded_values_(std::size_t depth, std::size_t width,
                                                          std::size_t value_bytes) noexcept {
    std::size_t values = divide_round_up(depth, width) * width;
    std::size_t const stride = values * value_bytes;
    if (stride && (stride & (stride - 1)) == 0) values += width;
    return values;
}

/**
 *  @brief Encodes C = A × Bᵀ for the rows from @p rows_begin up to @p rows_end through @p kernel of
 *      @p source , one threadgroup of @p threads per 64 × 64 tile.
 *
 *  B sits @p b_offset bytes past @p b , behind a pack's header, or at it for a symmetric call. A
 *  and B without bytes to read take C's address, as the record needs one.
 */
[[nodiscard]] static nk_status_t cross_(char const* source,
                                        std::size_t language, // vendor-boundary: NumKong's C signature
                                        std::size_t threads,  // vendor-boundary: NumKong's C signature
                                        char const* kernel, void const* a, void const* b,
                                        std::size_t b_offset, // vendor-boundary: NumKong's C signature
                                        f32_t* c, std::size_t rows_begin, std::size_t rows_end,
                                        std::size_t column_count, std::size_t depth, std::size_t row_bytes,
                                        std::size_t a_stride, std::size_t b_stride, std::size_t c_stride,
                                        u32_t upper_triangle, nk_stream_t stream) noexcept {
    constexpr std::size_t tile_k = 64;
    if ((row_bytes && ((reinterpret_cast<std::size_t>(a) | a_stride) & 15)) ||
        ((reinterpret_cast<std::size_t>(c) | c_stride) & (sizeof(f32_t) - 1)))
        return nk_misaligned_k;
    if (rows_end <= rows_begin || column_count == 0) return nk_success_k;
    if (rows_end > 0xFFFFFFFFu || column_count > 0xFFFFFFFFu || depth > 0xFFFFFFFFu) return nk_unexpected_dimensions_k;
    metal_call_t call;
    if (nk_status_t const opened = call.open(stream, source, kernel, language); opened != nk_success_k) return opened;
    nk_cross_arguments_metal_t arguments;
    if (!call.address(c, arguments.c)) return nk_device_memory_mismatch_k;
    arguments.a = arguments.b = arguments.c;
    if (row_bytes && !call.address(a, arguments.a)) return nk_device_memory_mismatch_k;
    if ((row_bytes || b_offset) && !call.address(b, arguments.b)) return nk_device_memory_mismatch_k;
    if (row_bytes || b_offset) arguments.b += b_offset;
    arguments.a_stride = a_stride, arguments.b_stride = b_stride, arguments.c_stride = c_stride / sizeof(f32_t);
    arguments.rows_begin = static_cast<u32_t>(rows_begin), arguments.rows_end = static_cast<u32_t>(rows_end);
    arguments.column_count = static_cast<u32_t>(column_count), arguments.depth = static_cast<u32_t>(depth);
    arguments.upper_triangle = upper_triangle;
    call.bind(arguments);
    return call.submit({divide_round_up(column_count, tile_k), divide_round_up(rows_end - rows_begin, tile_k), 1},
                       {threads});
}

/** Encodes the pack of B's columns @c [columns_begin,columns_end) into @p packed , recording
 *  @p capability , its rows padded to @p width values of @p value_bytes , one simdgroup per
 *  column; a window at column zero writes the header even when empty. */
[[nodiscard]] static nk_status_t cross_pack_(char const* kernel, void const* b,
                                             std::size_t columns, // vendor-boundary: NumKong's C signature
                                             std::size_t depth,   // vendor-boundary: NumKong's C signature
                                             std::size_t b_stride, void* packed,
                                             std::size_t columns_begin, // vendor-boundary: NumKong's C signature
                                             std::size_t columns_end, std::size_t width, std::size_t value_bytes,
                                             nk_capability_t capability, nk_stream_t stream) noexcept {
    constexpr std::size_t simdgroups_k = 8;
    if (reinterpret_cast<std::size_t>(packed) & 15) return nk_misaligned_k;
    if (columns > 0xFFFFFFFFu || depth > 0xFFFFFFFFu) return nk_unexpected_dimensions_k;
    columns_end = lesser_(columns_end, columns);
    if (columns_end <= columns_begin && columns_begin != 0) return nk_success_k;
    std::size_t const values = padded_values_(depth, width, value_bytes);
    std::size_t const window = columns_end > columns_begin ? columns_end - columns_begin : 0;
    metal_call_t call;
    if (nk_status_t const opened = call.open(stream, metal_source_k, kernel, language_3_1_k); opened != nk_success_k)
        return opened;
    nk_cross_pack_arguments_metal_t arguments;
    if (!call.address(packed, arguments.b_packed)) return nk_device_memory_mismatch_k;
    // An empty window reads no B, so B takes the pack's address
    arguments.b = arguments.b_packed;
    if (depth && window && !call.address(b, arguments.b)) return nk_device_memory_mismatch_k;
    arguments.b_stride = b_stride, arguments.depth_bytes = depth * value_bytes,
    arguments.row_bytes = values * value_bytes;
    arguments.capability = capability;
    arguments.column_count = static_cast<u32_t>(columns), arguments.depth = static_cast<u32_t>(depth);
    arguments.depth_padded_values = static_cast<u32_t>(values);
    arguments.columns_begin = static_cast<u32_t>(columns_begin),
    arguments.columns_end = static_cast<u32_t>(columns_end);
    call.bind(arguments);
    return call.submit({window ? divide_round_up(window, simdgroups_k) : 1}, {simdgroups_k * 32});
}

/** The products of one Apple capability over NumKong's tiles: @p source_ 's library, its threads
 *  per tile, and the padding its packs take for BFloat16 and E4M3 rows. */
template <typename kit_type_>
struct metal_products_launches;

#if NUMKONG_TARGET_METAL
template <>
struct metal_products_launches<metal_products_kit_t> {
    static constexpr char const* source_k = metal_source_k;
    static constexpr std::size_t language_k = language_3_1_k, threads_k = 256, bf16_width_k = 16, e4m3_width_k = 16;
    static constexpr char const* bf16_kernel_k = "nk_dots_bf16_metal_kernel_";
    static constexpr char const* e4m3_kernel_k = "nk_dots_e4m3_metal_kernel_";
};
#endif
#if NUMKONG_TARGET_APPLE9
template <>
struct metal_products_launches<apple9_products_kit_t> {
    static constexpr char const* source_k = apple9_source_k;
    static constexpr std::size_t language_k = language_3_1_k, threads_k = 128, bf16_width_k = 16, e4m3_width_k = 16;
    static constexpr char const* bf16_kernel_k = "nk_dots_bf16_apple9_kernel_";
    static constexpr char const* e4m3_kernel_k = "nk_dots_e4m3_apple9_kernel_";
};
#endif
#if NUMKONG_TARGET_APPLE10
template <>
struct metal_products_launches<apple10_products_kit_t> {
    static constexpr char const* source_k = apple10_source_k;
    static constexpr std::size_t language_k = language_4_0_k, threads_k = 128, bf16_width_k = 32, e4m3_width_k = 64;
    static constexpr char const* bf16_kernel_k = "nk_dots_bf16_apple10_kernel_";
    static constexpr char const* e4m3_kernel_k = "nk_dots_e4m3_apple10_kernel_";
};
#endif

/** C = A × Bᵀ over a pack of @p kit_type_ , whose size its own helper checks first. */
template <typename kit_type_, typename value_type_>
[[nodiscard]] static nk_status_t multiply_(value_type_ const* a, void const* packed, f32_t* c,
                                           std::size_t rows, // vendor-boundary: NumKong's C signature
                                           std::size_t columns, std::size_t depth, std::size_t a_stride,
                                           std::size_t c_stride, nk_stream_t stream) noexcept {
    using launches_t = metal_products_launches<kit_type_>;
    constexpr bool bf16_k = sizeof(value_type_) == sizeof(nk_bf16_t);
    nk_size_t bytes = 0;
    nk_status_t const sized = bf16_k ? kit_type_::pack_size_bf16(columns, depth, &bytes)
                                     : kit_type_::pack_size_e4m3(columns, depth, &bytes);
    if (sized != nk_success_k) return sized;
    if (reinterpret_cast<std::size_t>(packed) & 15) return nk_misaligned_k;
    std::size_t const row_bytes = padded_values_(depth, bf16_k ? launches_t::bf16_width_k : launches_t::e4m3_width_k,
                                                 sizeof(value_type_)) *
                                  sizeof(value_type_);
    return cross_(launches_t::source_k, launches_t::language_k, launches_t::threads_k,
                  bf16_k ? launches_t::bf16_kernel_k : launches_t::e4m3_kernel_k, a, packed, packed_header_bytes_k, c,
                  0, rows, columns, depth, depth * sizeof(value_type_), a_stride, row_bytes, c_stride, 0, stream);
}

/** The pack of @p kit_type_ , through the generic pack kernel of its value type. */
template <typename kit_type_, typename value_type_>
[[nodiscard]] static nk_status_t pack_(value_type_ const* b,
                                       std::size_t columns, // vendor-boundary: NumKong's C signature
                                       std::size_t depth,   // vendor-boundary: NumKong's C signature
                                       std::size_t b_stride, void* packed,
                                       std::size_t columns_begin, // vendor-boundary: NumKong's C signature
                                       std::size_t columns_end, nk_stream_t stream) noexcept {
    using launches_t = metal_products_launches<kit_type_>;
    constexpr bool bf16_k = sizeof(value_type_) == sizeof(nk_bf16_t);
    return cross_pack_(bf16_k ? "nk_dots_pack_bf16_metal_kernel_" : "nk_dots_pack_e4m3_metal_kernel_", b, columns,
                       depth, b_stride, packed, columns_begin, columns_end,
                       bf16_k ? launches_t::bf16_width_k : launches_t::e4m3_width_k, sizeof(value_type_),
                       kit_type_::capability_k, stream);
}

/** The upper triangle of the Gram matrix of @p vectors , rows @c [rows_begin,rows_end) . */
template <typename kit_type_>
[[nodiscard]] static nk_status_t symmetric_(nk_bf16_t const* vectors,
                                            std::size_t count, // vendor-boundary: NumKong's C signature
                                            std::size_t depth, // vendor-boundary: NumKong's C signature
                                            std::size_t stride, f32_t* result,
                                            std::size_t result_stride, // vendor-boundary: NumKong's C signature
                                            std::size_t rows_begin, std::size_t rows_end, nk_stream_t stream) noexcept {
    using launches_t = metal_products_launches<kit_type_>;
    return cross_(launches_t::source_k, launches_t::language_k, launches_t::threads_k, launches_t::bf16_kernel_k,
                  vectors, vectors, 0, result, rows_begin, lesser_(rows_end, count), count, depth,
                  depth * sizeof(nk_bf16_t), stride, stride, result_stride, 1, stream);
}

/** Defines the products kit @p kit_ through the launches above. */
#define uform_define_metal_products_(kit_)                                                                          \
    nk_status_t ue::kit_::pack_bf16(                                                                                \
        nk_bf16_t const* b, nk_size_t columns, nk_size_t depth, /* vendor-boundary: NumKong's C signature */        \
        nk_size_t b_stride,                                     /* vendor-boundary: NumKong's C signature */        \
        void* b_packed, nk_size_t columns_begin, nk_size_t columns_end, nk_stream_t stream) {                       \
        return pack_<kit_>(b, columns, depth, b_stride, b_packed, columns_begin, columns_end, stream);              \
    }                                                                                                               \
    nk_status_t ue::kit_::multiply_bf16(nk_bf16_t const* a, void const* b_packed, nk_f32_t* c,                      \
                                        nk_size_t rows, /* vendor-boundary: NumKong's C signature */                \
                                        nk_size_t columns, nk_size_t depth, nk_size_t a_stride, nk_size_t c_stride, \
                                        void*, nk_size_t, nk_stream_t stream) {                                     \
        return multiply_<kit_>(a, b_packed, c, rows, columns, depth, a_stride, c_stride, stream);                   \
    }                                                                                                               \
    nk_status_t ue::kit_::symmetric_bf16(                                                                           \
        nk_bf16_t const* vectors, nk_size_t vector_count, nk_size_t depth, nk_size_t stride, nk_f32_t* result,      \
        nk_size_t result_stride, nk_size_t rows_begin, nk_size_t rows_end, void*, nk_size_t, nk_stream_t stream) {  \
        return symmetric_<kit_>(vectors, vector_count, depth, stride, result, result_stride, rows_begin, rows_end,  \
                                stream);                                                                            \
    }                                                                                                               \
    nk_status_t ue::kit_::pack_e4m3(                                                                                \
        nk_e4m3_t const* b, nk_size_t columns, nk_size_t depth, /* vendor-boundary: NumKong's C signature */        \
        nk_size_t b_stride,                                     /* vendor-boundary: NumKong's C signature */        \
        void* b_packed, nk_size_t columns_begin, nk_size_t columns_end, nk_stream_t stream) {                       \
        return pack_<kit_>(b, columns, depth, b_stride, b_packed, columns_begin, columns_end, stream);              \
    }                                                                                                               \
    nk_status_t ue::kit_::multiply_e4m3(nk_e4m3_t const* a, void const* b_packed, nk_f32_t* c,                      \
                                        nk_size_t rows, /* vendor-boundary: NumKong's C signature */                \
                                        nk_size_t columns, nk_size_t depth, nk_size_t a_stride, nk_size_t c_stride, \
                                        void*, nk_size_t, nk_stream_t stream) {                                     \
        return multiply_<kit_>(a, b_packed, c, rows, columns, depth, a_stride, c_stride, stream);                   \
    }

#if NUMKONG_TARGET_METAL
uform_define_metal_products_(metal_products_kit_t)
#endif
#if NUMKONG_TARGET_APPLE9
    uform_define_metal_products_(apple9_products_kit_t)
#endif
#if NUMKONG_TARGET_APPLE10
        uform_define_metal_products_(apple10_products_kit_t)
#endif

#pragma endregion Products

#pragma region Attention

    /** The attention of one Apple capability over NumKong's passes: its library and the kernels of
     *  its forward and backward passes, the backward's tile and threads. */
    template <typename kit_type_>
    struct metal_attention_launches;

#if NUMKONG_TARGET_METAL
template <>
struct metal_attention_launches<metal_attention_kit_t> {
    static constexpr char const* source_k = metal_source_k;
    static constexpr std::size_t language_k = language_3_1_k, backward_tile_k = 8, backward_threads_k = 256;
    static constexpr char const* bf16_kernel_k = "nk_attention_packed_bf16_metal_kernel_";
    static constexpr char const* e4m3_kernel_k = "nk_attention_packed_e4m3_metal_kernel_";
    static constexpr char const* keys_kernel_k = "nk_attention_backward_keys_bf16_metal_kernel_";
    static constexpr char const* queries_kernel_k = "nk_attention_backward_queries_bf16_metal_kernel_";
};
#endif
#if NUMKONG_TARGET_APPLE9
template <>
struct metal_attention_launches<apple9_attention_kit_t> {
    static constexpr char const* source_k = apple9_source_k;
    static constexpr std::size_t language_k = language_3_1_k, backward_tile_k = 32, backward_threads_k = 128;
    static constexpr char const* bf16_kernel_k = "nk_attention_packed_bf16_apple9_kernel_";
    static constexpr char const* e4m3_kernel_k = "nk_attention_packed_e4m3_apple9_kernel_";
    static constexpr char const* keys_kernel_k = "nk_attention_backward_keys_bf16_apple9_kernel_";
    static constexpr char const* queries_kernel_k = "nk_attention_backward_queries_bf16_apple9_kernel_";
};
#endif
#if NUMKONG_TARGET_APPLE10
template <>
struct metal_attention_launches<apple10_attention_kit_t> {
    static constexpr char const* source_k = apple10_source_k;
    static constexpr std::size_t language_k = language_4_0_k, backward_tile_k = 32, backward_threads_k = 128;
    static constexpr char const* bf16_kernel_k = "nk_attention_packed_bf16_apple10_kernel_";
    static constexpr char const* e4m3_kernel_k = "nk_attention_packed_e4m3_apple10_kernel_";
    static constexpr char const* keys_kernel_k = "nk_attention_backward_keys_bf16_apple10_kernel_";
    static constexpr char const* queries_kernel_k = "nk_attention_backward_queries_bf16_apple10_kernel_";
};
#endif

/** Encodes the pack's directory, recording @p capability , for a window from task 0, then the
 *  pack of tasks @c [tasks_begin,tasks_end) of @p kv_heads heads of @p depth values of
 *  @p element_bytes , a threadgroup per task. */
[[nodiscard]] static nk_status_t attention_pack_(nk_capability_t capability, std::size_t element_bytes,
                                                 std::size_t kv_heads, std::size_t depth, u32_t const* key_offsets,
                                                 u32_t const* key_lengths, std::size_t segments, void const* keys,
                                                 std::size_t key_stride, void const* values,
                                                 std::size_t value_stride, // vendor-boundary: NumKong's C signature
                                                 void* packed, std::size_t tasks_begin, std::size_t tasks_end,
                                                 nk_stream_t stream) noexcept {
    if ((reinterpret_cast<std::size_t>(packed) & 15) ||
        ((reinterpret_cast<std::size_t>(keys) | key_stride | reinterpret_cast<std::size_t>(values) | value_stride) &
         (element_bytes - 1)) ||
        ((reinterpret_cast<std::size_t>(key_offsets) | reinterpret_cast<std::size_t>(key_lengths)) & 3))
        return nk_misaligned_k;
    if (kv_heads > 0xFFFFFFFFu || depth > 0xFFFFFFFFu || segments > 0xFFFFFFFFu) return nk_unexpected_dimensions_k;
    std::size_t const head_bytes = depth * element_bytes, row_bytes = kv_heads * head_bytes;
    std::size_t const end = lesser_(tasks_end, segments * kv_heads);
    bool const payload = tasks_begin < end && row_bytes != 0, lengths = key_lengths && segments;
    if (tasks_begin != 0 && !payload) return nk_success_k;
    metal_call_t call;
    char const* const first_kernel = tasks_begin == 0 ? "nk_attention_directory_metal_kernel_"
                                                      : "nk_attention_pack_metal_kernel_";
    if (nk_status_t const opened = call.open(stream, metal_source_k, first_kernel, language_3_1_k);
        opened != nk_success_k)
        return opened;
    nk_attention_pack_arguments_metal_t arguments;
    std::size_t packed_bytes = 0, lengths_bytes = 0, key_bytes = 0, value_bytes = 0;
    if (!call.bytes_after(packed, arguments.packed, packed_bytes) ||
        !call.address(key_offsets, arguments.key_offsets) ||
        (lengths && !call.bytes_after(key_lengths, arguments.key_lengths, lengths_bytes)))
        return nk_device_memory_mismatch_k;
    // Without a payload only the directory runs, so the keys and values take the pack's address
    arguments.keys = arguments.values = arguments.packed;
    if (payload && (!call.bytes_after(keys, arguments.keys, key_bytes) ||
                    !call.bytes_after(values, arguments.values, value_bytes)))
        return nk_device_memory_mismatch_k;
    arguments.key_value_head_count = kv_heads, arguments.depth = depth, arguments.head_bytes = head_bytes;
    arguments.segments = segments, arguments.key_stride = key_stride, arguments.value_stride = value_stride;
    arguments.tasks_begin = tasks_begin, arguments.tasks_end = end;
    arguments.key_bytes = key_bytes, arguments.value_bytes = value_bytes;
    arguments.packed_bytes = packed_bytes, arguments.lengths_bytes = lengths_bytes;
    arguments.capability = capability, arguments.code_bytes = head_bytes;
    call.bind(arguments);
    if (tasks_begin == 0) {
        if (!payload) return call.submit({1}, {32});
        call.then({1}, {32});
        if (nk_status_t const selected = call.select(metal_source_k, "nk_attention_pack_metal_kernel_", language_3_1_k);
            selected != nk_success_k)
            return selected;
    }
    return call.submit({lesser_(end - tasks_begin, grid_axis_max_k)}, {256});
}

/** Bytes of the decode split's 16 partitions per row, for calls of at most four query rows that
 *  see 512 keys or more, else 0. */
[[nodiscard]] static constexpr std::size_t split_bytes_(std::size_t heads, std::size_t depth, std::size_t query_tokens,
                                                        std::size_t keys_before) noexcept {
    bool const split = query_tokens <= 4 && depth != 0 && depth <= 256 && keys_before >= 512;
    return split ? query_tokens * heads * 16 * (depth + 2) * sizeof(f32_t) : 0;
}

/** Encodes the attention of @p kit_type_ over @p element_bytes values: its tile kernel over the
 *  window, or for calls of up to four query rows against long keys, given a @p workspace , the
 *  generic split passes and their merge. */
template <typename kit_type_>
[[nodiscard]] static nk_status_t attend_(std::size_t element_bytes, char const* kernel, char const* split_kernel,
                                         char const* merge_kernel, std::size_t heads, std::size_t kv_heads,
                                         std::size_t depth, u32_t const* query_offsets,
                                         std::size_t query_tokens, // vendor-boundary: NumKong's C signature
                                         f32_t scale, std::size_t keys_before, std::size_t keys_after,
                                         void const* queries, std::size_t query_stride, void const* packed,
                                         f32_t* output, std::size_t output_stride, f32_t* log_sum_exp,
                                         std::size_t tasks_begin, std::size_t tasks_end, void* workspace,
                                         std::size_t workspace_bytes, nk_stream_t stream) noexcept {
    using launches_t = metal_attention_launches<kit_type_>;
    if ((reinterpret_cast<std::size_t>(packed) & 15) ||
        ((reinterpret_cast<std::size_t>(queries) | query_stride) & (element_bytes - 1)) ||
        ((reinterpret_cast<std::size_t>(output) | output_stride | reinterpret_cast<std::size_t>(log_sum_exp) |
          reinterpret_cast<std::size_t>(query_offsets)) &
         3))
        return nk_misaligned_k;
    if (!kv_heads || heads % kv_heads || heads > 0xFFFFFFFFu || depth > 0xFFFFFFFFu) return nk_unexpected_dimensions_k;
    std::size_t const tasks = query_tokens * heads;
    std::size_t const split_bytes = split_bytes_(heads, depth, query_tokens, keys_before);
    if (workspace_bytes != 0 && workspace_bytes < split_bytes) return nk_unexpected_dimensions_k;
    tasks_end = lesser_(tasks_end, tasks);
    if (tasks_begin >= tasks_end || !depth) return nk_success_k;
    metal_call_t call;
    if (nk_status_t const opened = call.open(stream, launches_t::source_k, kernel, launches_t::language_k);
        opened != nk_success_k)
        return opened;
    nk_attention_arguments_metal_t arguments;
    std::size_t packed_bytes = 0, offsets_bytes = 0;
    if (!call.address(queries, arguments.queries) || !call.bytes_after(packed, arguments.packed, packed_bytes) ||
        !call.address(output, arguments.output) ||
        !call.bytes_after(query_offsets, arguments.query_offsets, offsets_bytes) ||
        !call.address(log_sum_exp, arguments.log_sum_exp))
        return nk_device_memory_mismatch_k;
    arguments.heads = heads, arguments.kv_heads = kv_heads, arguments.depth = depth;
    arguments.query_tokens = query_tokens, arguments.query_stride = query_stride;
    arguments.output_stride = output_stride, arguments.keys_before = keys_before, arguments.keys_after = keys_after;
    arguments.tasks_begin = tasks_begin, arguments.tasks_end = tasks_end;
    arguments.packed_bytes = packed_bytes, arguments.offsets_bytes = offsets_bytes;
    arguments.log_sum_exp_bytes = log_sum_exp ? tasks * sizeof(f32_t) : 0;
    arguments.capability = kit_type_::capability_k;
    arguments.scale2 = scale * log2_e_k;

    // Threadgroups walk segment × head pairs along x and the window's rows along y
    std::size_t const rows = divide_round_up(tasks_end, heads) - tasks_begin / heads;
    std::size_t const segments = offsets_bytes / sizeof(u32_t) - 1;
    std::size_t const pairs = segments > ~std::size_t {0} / heads ? ~std::size_t {0} : segments * heads;
    std::size_t const key_value_bytes = depth * kv_heads * element_bytes * 2;
    if (workspace_bytes != 0 && split_bytes != 0 && packed_bytes / key_value_bytes >= 512) {
        if (!call.address(workspace, arguments.split)) return nk_device_memory_mismatch_k;
        if (nk_status_t const selected = call.select(launches_t::source_k, split_kernel, launches_t::language_k);
            selected != nk_success_k)
            return selected;
        call.bind(arguments);
        std::size_t const groups = lesser_(pairs, grid_axis_max_k / rows);
        call.then({groups, rows, 16}, {128});
        if (nk_status_t const selected = call.select(launches_t::source_k, merge_kernel, launches_t::language_k);
            selected != nk_success_k)
            return selected;
        return call.submit({groups, rows, 1}, {32});
    }
    call.bind(arguments);
    std::size_t const row_groups = lesser_(32, divide_round_up(rows, 32));
    return call.submit({lesser_(pairs, grid_axis_max_k / row_groups), row_groups, 1}, {128});
}

/** Encodes the gradients of @p kit_type_ 's attention: the key and value gradients, then the query
 *  ones, each threadgroup taking a tile of keys or of folded query rows. */
template <typename kit_type_>
[[nodiscard]] static nk_status_t attend_gradients_(
    std::size_t heads, std::size_t kv_heads, std::size_t depth, u32_t const* query_offsets,
    std::size_t query_tokens, // vendor-boundary: NumKong's C signature
    f32_t scale, std::size_t keys_before, std::size_t keys_after, nk_bf16_t const* queries,
    std::size_t query_stride, // vendor-boundary: NumKong's C signature
    void const* packed, f32_t const* output, f32_t const* output_gradient,
    std::size_t output_stride, // vendor-boundary: NumKong's C signature
    f32_t const* log_sum_exp, f32_t* query_gradient,
    std::size_t query_gradient_stride, // vendor-boundary: NumKong's C signature
    f32_t* key_gradient,               // vendor-boundary: NumKong's C signature
    f32_t* value_gradient, std::size_t key_value_gradient_stride, std::size_t tasks_begin, std::size_t tasks_end,
    nk_stream_t stream) noexcept {
    using launches_t = metal_attention_launches<kit_type_>;
    constexpr std::size_t element_bytes_k = sizeof(nk_bf16_t);
    if ((reinterpret_cast<std::size_t>(packed) & 15) ||
        ((reinterpret_cast<std::size_t>(queries) | query_stride) & (element_bytes_k - 1)) ||
        ((reinterpret_cast<std::size_t>(output) | reinterpret_cast<std::size_t>(output_gradient) | output_stride |
          reinterpret_cast<std::size_t>(log_sum_exp) | reinterpret_cast<std::size_t>(query_gradient) |
          query_gradient_stride | reinterpret_cast<std::size_t>(key_gradient) |
          reinterpret_cast<std::size_t>(value_gradient) | key_value_gradient_stride |
          reinterpret_cast<std::size_t>(query_offsets)) &
         3))
        return nk_misaligned_k;
    if (!kv_heads || heads % kv_heads || heads > 0xFFFFFFFFu || depth > 0xFFFFFFFFu) return nk_unexpected_dimensions_k;
    if (tasks_begin >= tasks_end || !depth) return nk_success_k;
    std::size_t const rows = query_tokens * heads, gradient_row_bytes = kv_heads * depth * sizeof(f32_t);
    metal_call_t call;
    if (nk_status_t const opened = call.open(stream, launches_t::source_k, launches_t::keys_kernel_k,
                                             launches_t::language_k);
        opened != nk_success_k)
        return opened;
    nk_attention_backward_arguments_metal_t arguments;
    nk_attention_arguments_metal_t& attention = arguments.attention;
    std::size_t packed_bytes = 0, offsets_bytes = 0, key_gradient_bytes = 0, value_gradient_bytes = 0;
    if (!call.address(queries, attention.queries) || !call.bytes_after(packed, attention.packed, packed_bytes) ||
        !call.address(output, attention.output) ||
        !call.bytes_after(query_offsets, attention.query_offsets, offsets_bytes) ||
        !call.address(log_sum_exp, attention.log_sum_exp) ||
        !call.address(output_gradient, arguments.output_gradient) ||
        !call.address(query_gradient, arguments.query_gradient) ||
        !call.bytes_after(key_gradient, arguments.key_gradient, key_gradient_bytes) ||
        !call.bytes_after(value_gradient, arguments.value_gradient, value_gradient_bytes))
        return nk_device_memory_mismatch_k;
    std::size_t const segments = offsets_bytes / sizeof(u32_t) - 1;
    tasks_end = lesser_(tasks_end, segments * kv_heads);
    if (tasks_begin >= tasks_end) return nk_success_k;
    attention.heads = heads, attention.kv_heads = kv_heads, attention.depth = depth;
    attention.query_tokens = query_tokens, attention.query_stride = query_stride;
    attention.output_stride = output_stride, attention.keys_before = keys_before, attention.keys_after = keys_after;
    attention.tasks_begin = tasks_begin, attention.tasks_end = tasks_end;
    attention.packed_bytes = packed_bytes, attention.offsets_bytes = offsets_bytes;
    attention.log_sum_exp_bytes = rows * sizeof(f32_t), attention.capability = kit_type_::capability_k;
    attention.scale2 = scale * log2_e_k;
    arguments.query_gradient_stride = query_gradient_stride;
    arguments.key_value_gradient_stride = key_value_gradient_stride;
    arguments.key_value_gradient_bytes = lesser_(key_gradient_bytes, value_gradient_bytes);
    arguments.scale = scale;
    call.bind(arguments);

    // Threadgroups walk the window's tasks along x and tiles of keys or folded query rows along y
    std::size_t const key_value_bytes = gradient_row_bytes / sizeof(f32_t) * 2 * element_bytes_k;
    std::size_t const tile = launches_t::backward_tile_k, window = tasks_end - tasks_begin;
    std::size_t const key_groups = lesser_(64, divide_round_up(packed_bytes / key_value_bytes, tile));
    std::size_t const query_groups = lesser_(64, divide_round_up(rows / kv_heads, tile));
    std::size_t const keys_groups = key_groups ? key_groups : 1, queries_groups = query_groups ? query_groups : 1;
    call.then({lesser_(window, grid_axis_max_k / keys_groups), keys_groups, 1}, {launches_t::backward_threads_k});
    if (nk_status_t const selected = call.select(launches_t::source_k, launches_t::queries_kernel_k,
                                                 launches_t::language_k);
        selected != nk_success_k)
        return selected;
    return call.submit({lesser_(window, grid_axis_max_k / queries_groups), queries_groups, 1},
                       {launches_t::backward_threads_k});
}

/** Defines the attention kit @p kit_ through the launches above. */
#define uform_define_metal_attention_(kit_)                                                                            \
    nk_status_t ue::kit_::pack_bf16(nk_size_t key_value_head_count, nk_size_t depth, nk_u32_t const* key_offsets,      \
                                    nk_u32_t const* key_lengths, nk_size_t segment_count, nk_bf16_t const* keys,       \
                                    nk_size_t key_stride,    /* vendor-boundary: NumKong's C signature */              \
                                    nk_bf16_t const* values, /* vendor-boundary: NumKong's C signature */              \
                                    nk_size_t value_stride,  /* vendor-boundary: NumKong's C signature */              \
                                    void* key_value_packed, nk_size_t tasks_begin, nk_size_t tasks_end,                \
                                    nk_stream_t stream) {                                                              \
        return attention_pack_(capability_k, sizeof(nk_bf16_t), key_value_head_count, depth, key_offsets, key_lengths, \
                               segment_count, keys, key_stride, values, value_stride, key_value_packed, tasks_begin,   \
                               tasks_end, stream);                                                                     \
    }                                                                                                                  \
    nk_status_t ue::kit_::attend_bf16(nk_size_t head_count, nk_size_t key_value_head_count, nk_size_t depth,           \
                                      nk_u32_t const* query_offsets, nk_size_t query_token_count, nk_f32_t scale,      \
                                      nk_size_t keys_before, nk_size_t keys_after, nk_bf16_t const* queries,           \
                                      nk_size_t query_stride, void const* key_value_packed, nk_f32_t* output,          \
                                      nk_size_t output_stride, nk_f32_t* log_sum_exp,                                  \
                                      nk_size_t tasks_begin, /* vendor-boundary: NumKong's C signature */              \
                                      nk_size_t tasks_end,   /* vendor-boundary: NumKong's C signature */              \
                                      void* workspace,       /* vendor-boundary: NumKong's C signature */              \
                                      nk_size_t workspace_bytes, nk_stream_t stream) {                                 \
        return attend_<kit_>(sizeof(nk_bf16_t), metal_attention_launches<kit_>::bf16_kernel_k,                         \
                             "nk_attention_split_bf16_metal_kernel_", "nk_attention_merge_bf16_metal_kernel_",         \
                             head_count, key_value_head_count, depth, query_offsets, query_token_count, scale,         \
                             keys_before, keys_after, queries, query_stride, key_value_packed, output, output_stride,  \
                             log_sum_exp, tasks_begin, tasks_end, workspace, workspace_bytes, stream);                 \
    }                                                                                                                  \
    nk_status_t ue::kit_::attend_gradients_bf16(                                                                       \
        nk_size_t head_count, nk_size_t key_value_head_count, nk_size_t depth, nk_u32_t const* query_offsets,          \
        nk_size_t query_token_count, nk_f32_t scale, nk_size_t keys_before, nk_size_t keys_after,                      \
        nk_bf16_t const* queries, nk_size_t query_stride, void const* key_value_packed, nk_f32_t const* output,        \
        nk_f32_t const* output_gradient, nk_size_t output_stride, nk_f32_t const* log_sum_exp,                         \
        nk_f32_t* query_gradient, nk_size_t query_gradient_stride, nk_f32_t* key_gradient, nk_f32_t* value_gradient,   \
        nk_size_t key_value_gradient_stride, nk_size_t tasks_begin, nk_size_t tasks_end, void*, nk_size_t,             \
        nk_stream_t stream) {                                                                                          \
        return attend_gradients_<kit_>(head_count, key_value_head_count, depth, query_offsets, query_token_count,      \
                                       scale, keys_before, keys_after, queries, query_stride, key_value_packed,        \
                                       output, output_gradient, output_stride, log_sum_exp, query_gradient,            \
                                       query_gradient_stride, key_gradient, value_gradient, key_value_gradient_stride, \
                                       tasks_begin, tasks_end, stream);                                                \
    }                                                                                                                  \
    nk_status_t ue::kit_::pack_e4m3(nk_size_t key_value_head_count, nk_size_t depth, nk_u32_t const* key_offsets,      \
                                    nk_u32_t const* key_lengths, nk_size_t segment_count, nk_e4m3_t const* keys,       \
                                    nk_size_t key_stride,    /* vendor-boundary: NumKong's C signature */              \
                                    nk_e4m3_t const* values, /* vendor-boundary: NumKong's C signature */              \
                                    nk_size_t value_stride,  /* vendor-boundary: NumKong's C signature */              \
                                    void* key_value_packed, nk_size_t tasks_begin, nk_size_t tasks_end,                \
                                    nk_stream_t stream) {                                                              \
        return attention_pack_(capability_k, sizeof(nk_e4m3_t), key_value_head_count, depth, key_offsets, key_lengths, \
                               segment_count, keys, key_stride, values, value_stride, key_value_packed, tasks_begin,   \
                               tasks_end, stream);                                                                     \
    }                                                                                                                  \
    nk_status_t ue::kit_::attend_e4m3(nk_size_t head_count, nk_size_t key_value_head_count, nk_size_t depth,           \
                                      nk_u32_t const* query_offsets, nk_size_t query_token_count, nk_f32_t scale,      \
                                      nk_size_t keys_before, nk_size_t keys_after, nk_e4m3_t const* queries,           \
                                      nk_size_t query_stride, void const* key_value_packed, nk_f32_t* output,          \
                                      nk_size_t output_stride, nk_f32_t* log_sum_exp,                                  \
                                      nk_size_t tasks_begin, /* vendor-boundary: NumKong's C signature */              \
                                      nk_size_t tasks_end,   /* vendor-boundary: NumKong's C signature */              \
                                      void* workspace,       /* vendor-boundary: NumKong's C signature */              \
                                      nk_size_t workspace_bytes, nk_stream_t stream) {                                 \
        return attend_<kit_>(sizeof(nk_e4m3_t), metal_attention_launches<kit_>::e4m3_kernel_k,                         \
                             "nk_attention_split_e4m3_metal_kernel_", "nk_attention_merge_e4m3_metal_kernel_",         \
                             head_count, key_value_head_count, depth, query_offsets, query_token_count, scale,         \
                             keys_before, keys_after, queries, query_stride, key_value_packed, output, output_stride,  \
                             log_sum_exp, tasks_begin, tasks_end, workspace, workspace_bytes, stream);                 \
    }

#if NUMKONG_TARGET_METAL
uform_define_metal_attention_(metal_attention_kit_t)
#endif
#if NUMKONG_TARGET_APPLE9
    uform_define_metal_attention_(apple9_attention_kit_t)
#endif
#if NUMKONG_TARGET_APPLE10
        uform_define_metal_attention_(apple10_attention_kit_t)
#endif

#pragma endregion Attention

#pragma region Elementwise

#if NUMKONG_TARGET_METAL

    /** Threads of every element-wise threadgroup, and the most an RMSNorm one takes. */
    static constexpr std::size_t elementwise_threads_k = 256;

/** Encodes SwiGLU of @p value_bytes values, one thread per 16-byte chunk of a row, through the
 *  gated kernel, or the ungated one when @p up is null. */
[[nodiscard]] static nk_status_t swiglu_(char const* gated_kernel, char const* ungated_kernel,
                                         std::size_t value_bytes, // vendor-boundary: NumKong's C signature
                                         void const* gate, void const* up, void* y,
                                         std::size_t rows, // vendor-boundary: NumKong's C signature
                                         std::size_t columns, std::size_t gate_stride, std::size_t up_stride,
                                         std::size_t y_stride, f32_t gate_scale, f32_t output_scale,
                                         nk_stream_t stream) noexcept {
    if ((reinterpret_cast<std::size_t>(gate) | gate_stride | reinterpret_cast<std::size_t>(up) | (up ? up_stride : 0) |
         reinterpret_cast<std::size_t>(y) | y_stride) &
        (value_bytes - 1))
        return nk_misaligned_k;
    if (rows == 0 || columns == 0) return nk_success_k;
    if (rows > 0xFFFFFFFFu || columns > 0xFFFFFFFFu) return nk_unexpected_dimensions_k;
    metal_call_t call;
    if (nk_status_t const opened = call.open(stream, metal_source_k, up ? gated_kernel : ungated_kernel,
                                             language_3_1_k);
        opened != nk_success_k)
        return opened;
    nk_each_swiglu_arguments_metal_t arguments;
    if (!call.address(gate, arguments.gate) || !call.address(up, arguments.up) || !call.address(y, arguments.y))
        return nk_device_memory_mismatch_k;
    arguments.gate_stride = gate_stride, arguments.up_stride = up_stride, arguments.y_stride = y_stride;
    arguments.columns = static_cast<u32_t>(columns), arguments.gate_scale = gate_scale;
    arguments.output_scale = output_scale;
    call.bind(arguments);
    std::size_t const chunks = divide_round_up(columns, 16 / value_bytes);
    return call.submit({divide_round_up(chunks, elementwise_threads_k), rows, 1}, {elementwise_threads_k});
}

/** Encodes RMSNorm of @p input_bytes values into @p output_bytes ones, one threadgroup per vector,
 *  sized to the 16-byte chunks of its output, through the scaled kernel, or the unscaled one when
 *  @p gamma is null. */
[[nodiscard]] static nk_status_t rmsnorm_(char const* scaled_kernel, char const* unscaled_kernel,
                                          std::size_t input_bytes, std::size_t output_bytes, void const* x,
                                          f32_t const* gamma, void* y,
                                          std::size_t rows,   // vendor-boundary: NumKong's C signature
                                          std::size_t groups, // vendor-boundary: NumKong's C signature
                                          std::size_t columns, std::size_t x_stride, std::size_t y_stride,
                                          f32_t epsilon, nk_stream_t stream) noexcept {
    if (((reinterpret_cast<std::size_t>(x) | x_stride) & (input_bytes - 1)) ||
        ((reinterpret_cast<std::size_t>(y) | y_stride) & (output_bytes - 1)) ||
        (reinterpret_cast<std::size_t>(gamma) & 3))
        return nk_misaligned_k;
    if (rows == 0 || groups == 0 || columns == 0) return nk_success_k;
    if (rows > 0xFFFFFFFFu || groups > 0xFFFFFFFFu / elementwise_threads_k || columns > 0xFFFFFFFFu)
        return nk_unexpected_dimensions_k;
    metal_call_t call;
    if (nk_status_t const opened = call.open(stream, metal_source_k, gamma ? scaled_kernel : unscaled_kernel,
                                             language_3_1_k);
        opened != nk_success_k)
        return opened;
    nk_each_rmsnorm_arguments_metal_t arguments;
    if (!call.address(x, arguments.x) || !call.address(gamma, arguments.gamma) || !call.address(y, arguments.y))
        return nk_device_memory_mismatch_k;
    arguments.x_stride = x_stride, arguments.y_stride = y_stride;
    arguments.columns = static_cast<u32_t>(columns), arguments.epsilon = epsilon;
    call.bind(arguments);
    std::size_t const chunks = divide_round_up(columns, 16 / output_bytes);
    std::size_t const threads = chunks < elementwise_threads_k ? divide_round_up(chunks, 32) * 32
                                                               : elementwise_threads_k;
    return call.submit({groups, rows, 1}, {threads});
}

/** Plans into @p plan the bulk cast of @p count values from @p from_dtype to @p to_dtype : the hub
 *  NumKong's cast kernel converts through, and for one dtype on both ends the widest words both
 *  ends' @p addresses align; false when the cast writes nothing. */
[[nodiscard]] static bool cast_plan_(nk_dtype_t from_dtype, nk_dtype_t to_dtype, std::size_t count,
                                     std::size_t addresses, nk_cast_plan_metal_t& plan) noexcept {
    // `nk_cast_hub_t`: bytes, F32 to BF16, U64, I64, a quieted F32 pair and an F64 pair
    constexpr u32_t bytes_hub_k = 0, bf16_hub_k = 1, u64_hub_k = 2, i64_hub_k = 3, f32c_hub_k = 4, f64c_hub_k = 5;
    constexpr std::size_t bits_per_byte_k = 8;
    std::size_t const from_bits = nk_dtype_bits(from_dtype), to_bits = nk_dtype_bits(to_dtype);
    nk_dtype_family_t const from_family = nk_dtype_family(from_dtype), to_family = nk_dtype_family(to_dtype);
    auto const integer = [](nk_dtype_family_t family) noexcept {
        return family == nk_dtype_family_int_k || family == nk_dtype_family_uint_k;
    };
    // Values an F32 holds exactly: every type but F64 and integers past 16 bits
    auto const f32_exact = [&](nk_dtype_t dtype, nk_dtype_family_t family, std::size_t bits) noexcept {
        return dtype != nk_f64_k && dtype != nk_f64c_k && (integer(family) ? bits <= 16 : true);
    };
    std::size_t const from_per_byte = from_bits && from_bits < bits_per_byte_k ? bits_per_byte_k / from_bits : 1;
    std::size_t const to_per_byte = to_bits && to_bits < bits_per_byte_k ? bits_per_byte_k / to_bits : 1;
    plan.count = count, plan.from_dtype = static_cast<u32_t>(from_dtype), plan.to_dtype = static_cast<u32_t>(to_dtype);
    plan.unit_values = static_cast<u32_t>(from_per_byte > to_per_byte ? from_per_byte : to_per_byte);
    if (from_dtype == to_dtype) {
        plan.count = count * from_bits / bits_per_byte_k;
        plan.hub = bytes_hub_k, plan.unit_values = 16;
        while ((addresses | plan.count) & (plan.unit_values - 1)) plan.unit_values /= 2;
    }
    else if (from_dtype == nk_f32_k && to_dtype == nk_bf16_k) plan.hub = bf16_hub_k;
    else if (from_dtype == nk_f32c_k && to_dtype == nk_bf16c_k) plan.hub = bf16_hub_k, plan.count *= 2;
    else if (!from_bits || !to_bits) return false;
    else if (from_family == nk_dtype_family_uint_k && to_family == nk_dtype_family_uint_k) plan.hub = u64_hub_k;
    else if (integer(from_family) && integer(to_family)) {
        // The serial I64 hub stores no U4 and no U1
        if (to_dtype == nk_u4_k || to_dtype == nk_u1_k) return false;
        plan.hub = i64_hub_k;
    }
    else if (f32_exact(from_dtype, from_family, from_bits) && f32_exact(to_dtype, to_family, to_bits))
        plan.hub = f32c_hub_k;
    else plan.hub = f64c_hub_k;
    plan.units = divide_round_up(static_cast<std::size_t>(plan.count), std::size_t {plan.unit_values});
    return plan.units != 0;
}

#endif // NUMKONG_TARGET_METAL

#if NUMKONG_TARGET_METAL

nk_status_t ue::metal_elementwise_kit_t::cast( // vendor-boundary: NumKong's C signature, which declares no noexcept
    void const* from, nk_dtype_t from_dtype, void* to, nk_dtype_t to_dtype, nk_size_t count, nk_stream_t stream) {
    // UForm's surfaces hold no block-scaled type, whose casts take NumKong's two-pass kernels
    if (nk_dtype_is_block_scaled(from_dtype) || nk_dtype_is_block_scaled(to_dtype)) return nk_device_code_mismatch_k;
    auto const alignment = [](nk_dtype_t dtype) noexcept {
        std::size_t const bits = nk_dtype_bits(dtype);
        std::size_t const component = nk_dtype_family(dtype) == nk_dtype_family_complex_float_k ? bits / 2 : bits;
        return component >= 8 ? component / 8 : 1;
    };
    if ((reinterpret_cast<std::size_t>(from) & (alignment(from_dtype) - 1)) ||
        (reinterpret_cast<std::size_t>(to) & (alignment(to_dtype) - 1)))
        return nk_misaligned_k;
    nk_cast_plan_metal_t arguments;
    if (!cast_plan_(from_dtype, to_dtype, count,
                    reinterpret_cast<std::size_t>(from) | reinterpret_cast<std::size_t>(to), arguments))
        return nk_success_k;
    metal_call_t call;
    if (nk_status_t const opened = call.open(stream, metal_source_k, "nk_cast_metal_kernel_", language_3_1_k);
        opened != nk_success_k)
        return opened;
    if (!call.address(from, arguments.from) || !call.address(to, arguments.to)) return nk_device_memory_mismatch_k;
    call.bind(arguments);
    std::size_t const groups = lesser_(
        divide_round_up(static_cast<std::size_t>(arguments.units), elementwise_threads_k), grid_axis_max_k);
    return call.submit({groups}, {elementwise_threads_k});
}

nk_status_t ue::metal_elementwise_kit_t::sum_f32( // vendor-boundary: NumKong's C signature, which declares no noexcept
    nk_f32_t const* a, nk_f32_t const* b,
    nk_size_t n,      // vendor-boundary: NumKong's C signature
    nk_f32_t* result, // vendor-boundary: NumKong's C signature
    nk_stream_t stream) {
    if ((reinterpret_cast<std::size_t>(a) | reinterpret_cast<std::size_t>(b) | reinterpret_cast<std::size_t>(result)) &
        (sizeof(f32_t) - 1))
        return nk_misaligned_k;
    if (n == 0) return nk_success_k;
    std::size_t const chunks = divide_round_up(n, 16 / sizeof(f32_t));
    if (chunks > 0xFFFFFFFFu) return nk_unexpected_dimensions_k;
    metal_call_t call;
    if (nk_status_t const opened = call.open(stream, metal_source_k, "nk_each_sum_f32_metal_kernel_", language_3_1_k);
        opened != nk_success_k)
        return opened;
    nk_each_sum_arguments_metal_t arguments;
    if (!call.address(a, arguments.a) || !call.address(b, arguments.b) || !call.address(result, arguments.result))
        return nk_device_memory_mismatch_k;
    arguments.count = n;
    call.bind(arguments);
    return call.submit({divide_round_up(chunks, elementwise_threads_k)}, {elementwise_threads_k});
}

nk_status_t
ue::metal_elementwise_kit_t::rmscast_bf16( // vendor-boundary: NumKong's C signature, which declares no noexcept
    nk_f32_t const* x, nk_f32_t const* gamma, nk_bf16_t* y, nk_size_t rows, // vendor-boundary: NumKong's C signature
    nk_size_t groups,                                                       // vendor-boundary: NumKong's C signature
    nk_size_t columns,                                                      // vendor-boundary: NumKong's C signature
    nk_size_t x_stride, nk_size_t y_stride, nk_f32_t epsilon, nk_stream_t stream) {
    return rmsnorm_("nk_each_rmscast_bf16_scaled_metal_kernel_", "nk_each_rmscast_bf16_unscaled_metal_kernel_",
                    sizeof(f32_t), sizeof(nk_bf16_t), x, gamma, y, rows, groups, columns, x_stride, y_stride, epsilon,
                    stream);
}

nk_status_t
ue::metal_elementwise_kit_t::rmsnorm_bf16( // vendor-boundary: NumKong's C signature, which declares no noexcept
    nk_bf16_t const* x, nk_f32_t const* gamma, nk_bf16_t* y, nk_size_t rows, // vendor-boundary: NumKong's C signature
    nk_size_t groups,                                                        // vendor-boundary: NumKong's C signature
    nk_size_t columns,                                                       // vendor-boundary: NumKong's C signature
    nk_size_t x_stride, nk_size_t y_stride, nk_f32_t epsilon, nk_stream_t stream) {
    return rmsnorm_("nk_each_rmsnorm_bf16_scaled_metal_kernel_", "nk_each_rmsnorm_bf16_unscaled_metal_kernel_",
                    sizeof(nk_bf16_t), sizeof(nk_bf16_t), x, gamma, y, rows, groups, columns, x_stride, y_stride,
                    epsilon, stream);
}

nk_status_t
ue::metal_elementwise_kit_t::swiglu_f32( // vendor-boundary: NumKong's C signature, which declares no noexcept
    nk_f32_t const* gate, nk_f32_t const* up, nk_f32_t* y, nk_size_t rows, // vendor-boundary: NumKong's C signature
    nk_size_t columns,                                                     // vendor-boundary: NumKong's C signature
    nk_size_t gate_stride,                                                 // vendor-boundary: NumKong's C signature
    nk_size_t up_stride, nk_size_t y_stride, nk_f32_t gate_scale, nk_f32_t output_scale, nk_stream_t stream) {
    return swiglu_("nk_each_swiglu_f32_gated_metal_kernel_", "nk_each_swiglu_f32_ungated_metal_kernel_", sizeof(f32_t),
                   gate, up, y, rows, columns, gate_stride, up_stride, y_stride, gate_scale, output_scale, stream);
}

nk_status_t
ue::metal_elementwise_kit_t::swiglu_bf16( // vendor-boundary: NumKong's C signature, which declares no noexcept
    nk_bf16_t const* gate, nk_bf16_t const* up, nk_bf16_t* y, nk_size_t rows, // vendor-boundary: NumKong's C signature
    nk_size_t columns,                                                        // vendor-boundary: NumKong's C signature
    nk_size_t gate_stride,                                                    // vendor-boundary: NumKong's C signature
    nk_size_t up_stride, nk_size_t y_stride, nk_f32_t gate_scale, nk_f32_t output_scale, nk_stream_t stream) {
    return swiglu_("nk_each_swiglu_bf16_gated_metal_kernel_", "nk_each_swiglu_bf16_ungated_metal_kernel_",
                   sizeof(nk_bf16_t), gate, up, y, rows, columns, gate_stride, up_stride, y_stride, gate_scale,
                   output_scale, stream);
}

nk_status_t
ue::metal_elementwise_kit_t::rope_bf16( // vendor-boundary: NumKong's C signature, which declares no noexcept
    nk_bf16_t const* x, nk_f32_t const* cos, nk_f32_t const* sin, nk_bf16_t* y,
    nk_size_t rows,       // vendor-boundary: NumKong's C signature
    nk_size_t head_count, // vendor-boundary: NumKong's C signature
    nk_size_t depth, nk_size_t x_stride, nk_size_t y_stride, nk_stream_t stream) {
    if (depth % 2) return nk_unexpected_dimensions_k;
    if (((reinterpret_cast<std::size_t>(x) | x_stride | reinterpret_cast<std::size_t>(y) | y_stride) &
         (sizeof(nk_bf16_t) - 1)) ||
        ((reinterpret_cast<std::size_t>(cos) | reinterpret_cast<std::size_t>(sin)) & 3))
        return nk_misaligned_k;
    if (!rows || !head_count || !depth) return nk_success_k;
    if (depth > 0xFFFFFFFFu) return nk_unexpected_dimensions_k;
    metal_call_t call;
    if (nk_status_t const opened = call.open(stream, metal_source_k, "nk_attention_rope_bf16_metal_kernel_",
                                             language_3_1_k);
        opened != nk_success_k)
        return opened;
    nk_attention_rope_arguments_metal_t arguments;
    if (!call.address(x, arguments.x) || !call.address(cos, arguments.cosine) || !call.address(sin, arguments.sine) ||
        !call.address(y, arguments.y))
        return nk_device_memory_mismatch_k;
    arguments.rows = rows, arguments.heads = head_count, arguments.depth = depth;
    arguments.x_stride = x_stride, arguments.y_stride = y_stride;
    call.bind(arguments);
    // A simdgroup per head of every row, eight per threadgroup
    constexpr std::size_t groups_limit_k = std::size_t {1} << 20;
    return call.submit({lesser_(divide_round_up(rows * head_count, 8), groups_limit_k)}, {elementwise_threads_k});
}

#endif // NUMKONG_TARGET_METAL

#pragma endregion Elementwise

#pragma region Entries

nk_allocator_t ue::vram_apple_entries_kit_t::policy(nk_stream_t stream, uform_residency_t residency) noexcept {
    nk_allocator_t policy {};
    policy.handle = context_of_(stream);
    if (!policy.handle) return {};
    static constexpr auto allocate = [](nk_size_t bytes, void* handle, uform_residency_t residency) noexcept -> void* {
        metal_context_t& context = *static_cast<metal_context_t*>(handle);
        metal_buffer_t buffer = util::metal_allocate(context, bytes);
        if (!buffer) return nullptr;
        void* const address = residency == uform_residency_ram_k ? static_cast<void*>(buffer.host)
                                                                 : reinterpret_cast<void*>(metal_send<u64_t>(
                                                                       *context.api, buffer.handle, "gpuAddress"));
        st::unique_lock<fu::spin_mutex_t> guard {context.lock};
        auto const index = context.buffers.take();
        if (index == context.buffers.absent_k) return nullptr;
        context.buffers[index] = std::move(buffer);
        return address;
    };
    if (residency == uform_residency_ram_k)
        policy.allocate = [](nk_size_t bytes, void* handle, nk_stream_t) noexcept -> void* {
            return allocate(bytes, handle, uform_residency_ram_k);
        };
    else
        policy.allocate = [](nk_size_t bytes, void* handle, nk_stream_t) noexcept -> void* {
            return allocate(bytes, handle, uform_residency_vram_k);
        };
    // The buffer a pinned page starts at its host address, and device memory at its GPU address
    policy.free = [](void* pointer, nk_size_t, void* handle, nk_stream_t) noexcept {
        metal_context_t& context = *static_cast<metal_context_t*>(handle);
        metal_buffer_t dying;
        st::unique_lock<fu::spin_mutex_t> guard {context.lock};
        for (metal_buffer_t& buffer : context.buffers.records()) {
            if (!buffer) continue;
            bool const named = buffer.host == pointer || metal_send<u64_t>(*buffer.api, buffer.handle, "gpuAddress") ==
                                                             reinterpret_cast<u64_t>(pointer);
            if (!named) continue;
            context.buffers.vacate(context.buffers.index_of(buffer));
            dying = std::move(buffer);
            return;
        }
    };
    return policy;
}

nk_status_t ue::vram_apple_entries_kit_t::select(std::size_t) noexcept { return nk_success_k; }

nk_status_t ue::vram_apple_entries_kit_t::synchronize(nk_stream_t stream) noexcept {
    metal_context_t const* const context = context_of_(stream);
    return context ? drain_(*context, stream) : nk_missing_gpu_k;
}

nk_status_t ue::vram_apple_entries_kit_t::describe(std::size_t vendor_ordinal,
                                                   device_description_t& description) noexcept {
    metal_context_t const* const context = uf::gpu_drivers().metal_contexts.context(static_cast<int>(vendor_ordinal));
    if (!context) return nk_missing_gpu_k;
    util::metal_api_t const& api = *context->api;
    void* const pool = api.autorelease_pool_push();
    char const* const name = metal_send<char const*>(api, metal_send(api, context->device, "name"), "UTF8String");
    if (name) std::strncpy(description.name, name, sizeof(description.name) - 1);
    api.autorelease_pool_pop(pool);
    description.compute_units = context->multiprocessor_count;
    description.memory_bytes = metal_send<u64_t>(api, context->device, "recommendedMaxWorkingSetSize");
    description.architecture_major = context->family;
    description.architecture_minor = 0;
    return nk_success_k;
}

nk_capability_t ue::vram_apple_entries_kit_t::capabilities_detected(std::size_t,
                                                                    device_description_t const& description) noexcept {
    u32_t const family = description.architecture_major;
    return (family >= 7 ? nk_cap_metal_k : 0) | (family >= 9 ? nk_cap_apple9_k : 0) |
           (family >= 10 ? nk_cap_apple10_k : 0);
}

nk_capability_t ue::vram_apple_entries_kit_t::capabilities_compiled(std::size_t vendor_ordinal,
                                                                    device_description_t const& description) noexcept {
    return capabilities_detected(vendor_ordinal, description);
}

nk_status_t ue::vram_apple_entries_kit_t::copy(bytes_span_t target, bytes_view_t source, nk_stream_t stream) noexcept {
    if (source.empty()) return nk_success_k;
    metal_context_t* const context = context_of_(stream);
    if (!context) return nk_missing_gpu_k;
    // Between device buffers a kernel copies in stream order; a host side waits for the stream
    if (on_device_(*context, target.data()) && on_device_(*context, source.data())) {
        bytes_arguments_t arguments;
        arguments.bytes = source.size();
        return launch_(kernel_t::copy_bytes_k, stride_blocks_(source.size()), entries_block_threads_k, stream,
                       {target.data(), source.data()}, arguments);
    }
    if (nk_status_t const drained = drain_(*context, stream); drained != nk_success_k) return drained;
    std::memcpy(host_of_(target.data()), host_of_(source.data()), source.size());
    return nk_success_k;
}

nk_status_t ue::vram_apple_entries_kit_t::zero(bytes_span_t target, nk_stream_t stream) noexcept {
    if (target.empty()) return nk_success_k;
    metal_context_t* const context = context_of_(stream);
    if (!context) return nk_missing_gpu_k;
    if (on_device_(*context, target.data())) {
        bytes_arguments_t arguments;
        arguments.bytes = target.size();
        return launch_(kernel_t::zero_bytes_k, stride_blocks_(target.size()), entries_block_threads_k, stream,
                       {target.data(), nullptr}, arguments);
    }
    if (nk_status_t const drained = drain_(*context, stream); drained != nk_success_k) return drained;
    std::memset(host_of_(target.data()), 0, target.size());
    return nk_success_k;
}

nk_status_t ue::vram_apple_entries_kit_t::fill_layout(u32r1_view_t lengths, u32r1_span_t packed_offsets,
                                                      u32r1_span_t pool_offsets, nk_stream_t stream) noexcept {
    fill_layout_arguments_t arguments;
    arguments.lengths = view_of_(lengths), arguments.packed_offsets = view_of_(packed_offsets);
    arguments.pool_offsets = view_of_(pool_offsets);
    return launch_(kernel_t::fill_layout_k, 1, 1, stream,
                   {first_of_(lengths), first_of_(packed_offsets), first_of_(pool_offsets)}, arguments);
}

nk_status_t ue::vram_apple_entries_kit_t::fill_coords(segment_layout_t layout, nk::tensor_view<u16_t, 2> image_shapes,
                                                      nk::tensor_view<u16_t, 2> video_shapes, u32_t tube_frames,
                                                      u32_t frame_stride, u32_t positions, i16r2_span_t coords,
                                                      nk_stream_t stream) noexcept {
    fill_coords_arguments_t arguments;
    arguments.layout = layout_of_(layout);
    arguments.image_shapes = view_of_(image_shapes), arguments.video_shapes = view_of_(video_shapes);
    arguments.coords = view_of_(coords);
    arguments.tube_frames = tube_frames, arguments.frame_stride = frame_stride, arguments.positions = positions;
    return launch_(kernel_t::fill_coords_k, layout.lengths.extent(0), entries_block_threads_k, stream,
                   {first_of_(layout.grid_offsets), first_of_(layout.lengths), first_of_(layout.packed_offsets),
                    first_of_(layout.modality), first_of_(image_shapes), first_of_(video_shapes), first_of_(coords)},
                   arguments);
}

nk_status_t ue::vram_apple_entries_kit_t::gather_bytes(segment_layout_t layout, nk::tensor_view<token_id_t, 1> tokens,
                                                       bf16r2_view_t table, bf16r2_span_t rows,
                                                       nk_stream_t stream) noexcept {
    gather_bytes_arguments_t arguments;
    arguments.layout = layout_of_(layout);
    arguments.tokens = view_of_(tokens), arguments.table = view_of_(table), arguments.rows = view_of_(rows);
    return launch_(kernel_t::gather_bytes_k, layout.lengths.extent(0), entries_block_threads_k, stream,
                   {first_of_(layout.grid_offsets), first_of_(layout.lengths), first_of_(layout.packed_offsets),
                    first_of_(layout.modality), first_of_(tokens), first_of_(table), first_of_(rows)},
                   arguments);
}

nk_status_t ue::vram_apple_entries_kit_t::stage_patches(segment_layout_t layout, modality_t target,
                                                        nk::tensor_view<pixel_t, 3> pixels, f32r1_view_t mean,
                                                        f32r1_view_t inverse_std, bf16r2_span_t patches,
                                                        nk_stream_t stream) noexcept {
    nk::tensor_span<nk_bf16_t, 3> const cells {patches.data(), {patches.extent(0), pixels.extent(1), pixels.extent(2)}};
    stage_patches_arguments_t arguments;
    arguments.layout = layout_of_(layout);
    arguments.pixels = view_of_(pixels), arguments.mean = view_of_(mean), arguments.inverse_std = view_of_(inverse_std);
    arguments.patches = view_of_(cells), arguments.target = static_cast<u32_t>(target);
    return launch_(
        kernel_t::stage_patches_k, layout.lengths.extent(0), entries_block_threads_k, stream,
        {first_of_(layout.grid_offsets), first_of_(layout.lengths), first_of_(layout.packed_offsets),
         first_of_(layout.modality), first_of_(pixels), first_of_(mean), first_of_(inverse_std), first_of_(cells)},
        arguments);
}

nk_status_t ue::vram_apple_entries_kit_t::stage_frames(segment_layout_t layout, nk::tensor_view<f32_t, 2> samples,
                                                       bf16r2_span_t frames, nk_stream_t stream) noexcept {
    stage_frames_arguments_t arguments;
    arguments.layout = layout_of_(layout);
    arguments.samples = view_of_(samples), arguments.frames = view_of_(frames);
    return launch_(kernel_t::stage_frames_k, layout.lengths.extent(0), entries_block_threads_k, stream,
                   {first_of_(layout.grid_offsets), first_of_(layout.lengths), first_of_(layout.packed_offsets),
                    first_of_(layout.modality), first_of_(samples), first_of_(frames)},
                   arguments);
}

nk_status_t ue::vram_apple_entries_kit_t::mrope(bf16r2_span_t values, u32_t role_offset, u32_t heads, u32_t head_width,
                                                i16r2_view_t coords, rope_grids_t angles, nk_stream_t stream) noexcept {
    mrope_arguments_t arguments;
    arguments.values = view_of_(values), arguments.temporal = view_of_(angles.temporal);
    arguments.height = view_of_(angles.height), arguments.width = view_of_(angles.width);
    arguments.coords = view_of_(coords);
    arguments.heads = heads, arguments.head_width = head_width, arguments.role_offset = role_offset;
    return launch_(kernel_t::mrope_k, stride_blocks_(values.extent(0) * (head_width / 2u)), entries_block_threads_k,
                   stream,
                   {first_of_(values), first_of_(angles.temporal), first_of_(angles.height), first_of_(angles.width),
                    first_of_(coords)},
                   arguments);
}

nk_status_t ue::vram_apple_entries_kit_t::quantize_tokens_e4m3(bf16r2_view_t values, e4m3r2_span_t shadow,
                                                               f32r1_span_t token_scales, nk_stream_t stream) noexcept {
    quantize_arguments_t arguments;
    arguments.values = view_of_(values), arguments.shadow = view_of_(shadow);
    arguments.token_scales = view_of_(token_scales);
    return launch_(kernel_t::quantize_bf16_tokens_k, values.extent(0), entries_block_threads_k, stream,
                   {first_of_(values), first_of_(shadow), first_of_(token_scales)}, arguments);
}

nk_status_t ue::vram_apple_entries_kit_t::quantize_tokens_e4m3(f32r2_view_t values, e4m3r2_span_t shadow,
                                                               f32r1_span_t token_scales, nk_stream_t stream) noexcept {
    quantize_arguments_t arguments;
    arguments.values = view_of_(values), arguments.shadow = view_of_(shadow);
    arguments.token_scales = view_of_(token_scales);
    return launch_(kernel_t::quantize_f32_tokens_k, values.extent(0), entries_block_threads_k, stream,
                   {first_of_(values), first_of_(shadow), first_of_(token_scales)}, arguments);
}

nk_status_t ue::vram_apple_entries_kit_t::rescale_tokens(f32r2_view_t products, f32r1_view_t token_scales,
                                                         f32_t weight_descale, bf16r2_view_t residual,
                                                         bf16r2_span_t values, nk_stream_t stream) noexcept {
    rescale_arguments_t arguments;
    arguments.products = view_of_(products), arguments.token_scales = view_of_(token_scales);
    arguments.residual = view_of_(residual), arguments.values = view_of_(values);
    arguments.weight_descale = weight_descale;
    return launch_(kernel_t::rescale_tokens_k, stride_blocks_(values.extent(0) * values.extent(1)),
                   entries_block_threads_k, stream,
                   {first_of_(products), first_of_(token_scales), first_of_(residual), first_of_(values)}, arguments);
}

#pragma endregion Entries
