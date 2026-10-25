/**
 *  @file src/encoder/ram_variants_intel_granite.cpp
 *  @author Ash Vardanian
 *  @date October 25, 2026
 *  @brief The intel_granite unit: its encoder variant runs the AMX-FP16 kits, or answers no table
 *      where they match its parent's.
 */

#include "encoder_specializations.hpp"

template struct uf::encoder_class_tables<uf::encoder_home_t::ram_k, uf::cpu_class_t::intel_granite_k>;
