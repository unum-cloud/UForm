/**
 *  @file src/encoder/vram_variants_apple_metal10.cpp
 *  @author Ash Vardanian
 *  @date November 16, 2026
 *  @brief The apple_metal10 unit: its encoder variant runs the Apple10 kits of the M5 GPUs, or
 *      answers no table where NumKong compiled none.
 */

#include "encoder_specializations.hpp"

template struct uf::encoder_class_tables<uf::vendored_residency_t::vram_apple_k, uf::gpu_class_t::apple_metal10_k>;
