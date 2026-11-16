/**
 *  @file src/encoder/vram_variants_apple_metal9.cpp
 *  @author Ash Vardanian
 *  @date November 16, 2026
 *  @brief The apple_metal9 unit: its encoder variant runs the Apple9 kits of the M3 and M4 GPUs,
 *      or answers no table where NumKong compiled none.
 */

#include "encoder_specializations.hpp"

template struct uf::encoder_class_tables<uf::vendored_residency_t::vram_apple_k, uf::gpu_class_t::apple_metal9_k>;
