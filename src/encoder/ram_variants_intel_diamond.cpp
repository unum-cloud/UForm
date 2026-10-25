/**
 *  @file src/encoder/ram_variants_intel_diamond.cpp
 *  @author Ash Vardanian
 *  @date October 25, 2026
 *  @brief The intel_diamond unit: its encoder variant runs the AVX10.2 FP8 kits, or answers no
 *      table where they match its parent's.
 */

#include "encoder_specializations.hpp"

template struct uf::encoder_class_tables<uf::encoder_home_t::ram_k, uf::cpu_class_t::intel_diamond_k>;
