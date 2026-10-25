/**
 *  @file src/encoder/ram_variants_intel_sapphire.cpp
 *  @author Ash Vardanian
 *  @date October 25, 2026
 *  @brief The intel_sapphire unit: its encoder variant runs the AMX tile kits, or answers no table
 *      where they match its parent's.
 */

#include "encoder_specializations.hpp"

template struct uf::encoder_class_tables<uf::encoder_home_t::ram_k, uf::cpu_class_t::intel_sapphire_k>;
