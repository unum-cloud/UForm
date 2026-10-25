/**
 *  @file src/encoder/vram_variants_amd_cdna3.cpp
 *  @author Ash Vardanian
 *  @date October 25, 2026
 *  @brief The amd_cdna3 unit: its encoder variant runs the CDNA3 kits, or answers no table where
 *      NumKong compiled none.
 */

#include "encoder_specializations.hpp"

template struct uf::encoder_class_tables<uf::encoder_home_t::vram_amd_k, uf::gpu_class_t::amd_cdna3_k>;
