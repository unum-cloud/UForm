/**
 *  @file src/encoder/vram_variants_nvidia_hopper.cpp
 *  @author Ash Vardanian
 *  @date October 25, 2026
 *  @brief The nvidia_hopper unit: its encoder variant runs the Hopper kits, or answers no table
 *      where they match its parent's.
 */

#include "encoder_specializations.hpp"

template struct uf::encoder_class_tables<uf::encoder_home_t::vram_nvidia_k, uf::gpu_class_t::nvidia_hopper_k>;
