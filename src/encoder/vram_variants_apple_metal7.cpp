/**
 *  @file src/encoder/vram_variants_apple_metal7.cpp
 *  @author Ash Vardanian
 *  @date November 16, 2026
 *  @brief The apple_metal7 unit: its encoder variant runs the generic Metal kits, which every
 *      Apple GPU of family 7 and later loads.
 */

#include "encoder_specializations.hpp"

template struct uf::encoder_class_tables<uf::vendored_residency_t::vram_apple_k, uf::gpu_class_t::apple_metal7_k>;
