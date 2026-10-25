/**
 *  @file src/encoder/vram_variants_nvidia_blackwellrtx.cpp
 *  @author Ash Vardanian
 *  @date October 25, 2026
 *  @brief The nvidia_blackwellrtx unit: its encoder variant runs consumer Blackwell's FP8 kits, or
 *      answers no table where they match its parent's.
 */

#include "encoder_specializations.hpp"

template struct uf::encoder_class_tables<uf::encoder_home_t::vram_nvidia_k, uf::gpu_class_t::nvidia_blackwellrtx_k>;
