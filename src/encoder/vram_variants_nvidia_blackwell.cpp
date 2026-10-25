/**
 *  @file src/encoder/vram_variants_nvidia_blackwell.cpp
 *  @author Ash Vardanian
 *  @date October 25, 2026
 *  @brief The nvidia_blackwell unit: its encoder variant runs the Blackwell kits, or answers no
 *      table where they match its parent's.
 */

#include "encoder_specializations.hpp"

template struct uf::encoder_class_tables<uf::encoder_home_t::vram_nvidia_k, uf::gpu_class_t::nvidia_blackwell_k>;
