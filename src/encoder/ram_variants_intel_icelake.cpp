/**
 *  @file src/encoder/ram_variants_intel_icelake.cpp
 *  @author Ash Vardanian
 *  @date October 25, 2026
 *  @brief The intel_icelake unit: its encoder variant runs the Icelake casts over AVX-512 kits, or
 *      answers no table where they match its parent's.
 */

#include "encoder_specializations.hpp"

template struct uf::encoder_class_tables<uf::encoder_home_t::ram_k, uf::cpu_class_t::intel_icelake_k>;
