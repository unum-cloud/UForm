/**
 *  @file src/encoder/vram_variants_amd.cpp
 *  @author Ash Vardanian
 *  @date October 25, 2026
 *  @brief The AMD floor unit: its encoder variant runs NumKong's generic ROCm kits on every AMD
 *      device without a class of its own.
 */

#include "encoder_specializations.hpp"

template struct uf::encoder_class_tables<uf::encoder_home_t::vram_amd_k, uf::gpu_class_t::unknown_k>;
