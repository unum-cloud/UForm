/**
 *  @file src/media/planning.cpp
 *  @author Ash Vardanian
 *  @date July 4, 2026
 *  @brief Image and video planning — see the header for the contract.
 */
#include <cmath> // `std::cbrt`, `std::sqrt`

#include <algorithm> // `std::clamp`

#include "uform/media/planning.hpp"

namespace um = unum::uform::media;
namespace uu = unum::uform::utilities;
using namespace um;
using uu::divide_round_up;

patch_grid_t um::image_patch_grid(u32_t columns, image_extent_t native) noexcept {
    f64_t const aspect = static_cast<f64_t>(fu::max_of<u32_t>(native.width, 1u)) /
                         static_cast<f64_t>(fu::max_of<u32_t>(native.height, 1u));
    u32_t const rows = std::clamp<u32_t>(static_cast<u32_t>(std::sqrt(static_cast<f64_t>(columns) / aspect)), 1u,
                                         fu::max_of<u32_t>(columns, 1u));
    return patch_grid_t {static_cast<u16_t>(rows), static_cast<u16_t>(fu::max_of<u32_t>(columns / rows, 1u))};
}

crop_t um::image_center_crop(image_extent_t source, image_extent_t target) noexcept {
    crop_t crop {};
    if (static_cast<u64_t>(source.width) * target.height > static_cast<u64_t>(target.width) * source.height) {
        crop.height = source.height;
        crop.width = static_cast<u32_t>(static_cast<u64_t>(source.height) * target.width / target.height);
    }
    else {
        crop.width = source.width;
        crop.height = static_cast<u32_t>(static_cast<u64_t>(source.width) * target.height / target.width);
    }
    crop.x = (source.width - crop.width) / 2u;
    crop.y = (source.height - crop.height) / 2u;
    return crop;
}

u32_t um::image_decode_scale(image_extent_t native, image_extent_t target) noexcept {
    u32_t level = 0;
    while (level < 3 && (native.width >> (level + 1)) >= target.width &&
           (native.height >> (level + 1)) >= target.height)
        ++level;
    return level;
}

image_extent_t um::image_decoded_extent(image_extent_t native, u32_t scale) noexcept {
    u32_t const factor = 1u << scale;
    return image_extent_t {static_cast<u32_t>(divide_round_up(native.width, factor)),
                           static_cast<u32_t>(divide_round_up(native.height, factor))};
}

video_grid_t um::video_tube_grid(u32_t frames, image_extent_t native, video_batch_shape_t const &shape,
                                 u32_t columns) noexcept {
    u32_t const stride = fu::max_of<u32_t>(shape.frame_stride, 1u);
    u32_t const depth = fu::max_of<u32_t>(shape.tube_frames, 1u);
    u32_t const side = fu::max_of<u32_t>(shape.patch_side, 1u);
    u32_t const sampled_frames = static_cast<u32_t>(divide_round_up(fu::max_of<u32_t>(frames, 1u), stride));
    u32_t temporal_groups = static_cast<u32_t>(divide_round_up(sampled_frames, depth));
    u32_t patch_rows = static_cast<u32_t>(divide_round_up(fu::max_of<u32_t>(native.height, 1u), side));
    u32_t patch_columns = static_cast<u32_t>(divide_round_up(fu::max_of<u32_t>(native.width, 1u), side));
    u32_t const budget = fu::max_of<u32_t>(columns, 1u);
    f64_t const product = static_cast<f64_t>(temporal_groups) * patch_rows * patch_columns;
    if (product < budget) {
        // A clip fills its row: the frames it has stay, and the frame grows to take the rest.
        f64_t const scale = std::sqrt(static_cast<f64_t>(budget) / product);
        patch_rows = fu::max_of<u32_t>(1u, static_cast<u32_t>(patch_rows * scale));
        patch_columns = fu::max_of<u32_t>(1u, static_cast<u32_t>(patch_columns * scale));
    }
    if (static_cast<u64_t>(temporal_groups) * patch_rows * patch_columns > budget) {
        f64_t const scale = std::cbrt(static_cast<f64_t>(budget) /
                                      (static_cast<f64_t>(temporal_groups) * patch_rows * patch_columns));
        temporal_groups = fu::max_of<u32_t>(1u, static_cast<u32_t>(temporal_groups * scale));
        patch_rows = fu::max_of<u32_t>(1u, static_cast<u32_t>(patch_rows * scale));
        patch_columns = fu::max_of<u32_t>(1u, static_cast<u32_t>(patch_columns * scale));
        // Floored scaling can leave the product a hair over budget; trim the largest axis until it fits one row.
        while (static_cast<u64_t>(temporal_groups) * patch_rows * patch_columns > budget) {
            if (patch_columns >= patch_rows && patch_columns >= temporal_groups && patch_columns > 1u) --patch_columns;
            else if (patch_rows >= temporal_groups && patch_rows > 1u) --patch_rows;
            else if (temporal_groups > 1u) --temporal_groups;
            else break;
        }
    }
    return video_grid_t {static_cast<u16_t>(fu::min_of<u32_t>(temporal_groups, 0xFFFFu)),
                         static_cast<u16_t>(fu::min_of<u32_t>(patch_rows, 0xFFFFu)),
                         static_cast<u16_t>(fu::min_of<u32_t>(patch_columns, 0xFFFFu))};
}
