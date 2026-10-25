/**
 *  @file src/encoder/ram_variants_amd_genoa.cpp
 *  @author Ash Vardanian
 *  @date October 25, 2026
 *  @brief The amd_genoa unit: its encoder variant runs the BFloat16 dot-product kits, or answers no
 *      table where they match its parent's.
 */

#include "encoder_specializations.hpp"

template struct uf::encoder_class_tables<uf::encoder_home_t::ram_k, uf::cpu_class_t::amd_genoa_k>;
