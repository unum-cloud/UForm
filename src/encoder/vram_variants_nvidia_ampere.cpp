/**
 *  @file src/encoder/vram_variants_nvidia_ampere.cpp
 *  @author Ash Vardanian
 *  @date October 25, 2026
 *  @brief The nvidia_ampere unit: its encoder variant runs the Ampere kits, or answers no table
 *      where NumKong compiled none.
 */

#include "encoder_specializations.hpp"

template struct uf::encoder_class_tables<uf::encoder_home_t::vram_nvidia_k, uf::gpu_class_t::nvidia_ampere_k>;
