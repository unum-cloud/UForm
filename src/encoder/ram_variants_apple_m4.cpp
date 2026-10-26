/**
 *  @file src/encoder/ram_variants_apple_m4.cpp
 *  @author Ash Vardanian
 *  @date October 25, 2026
 *  @brief The apple_m4 unit: its encoder variant runs the SME tile kits, or answers no table where
 *      they match its parent's.
 */

#include "encoder_specializations.hpp"

template struct uf::encoder_class_tables<uf::encoder_home_t::ram_k, uf::cpu_class_t::apple_m4_k>;
template struct uf::host_class_tables<uf::cpu_class_t::apple_m4_k>;
