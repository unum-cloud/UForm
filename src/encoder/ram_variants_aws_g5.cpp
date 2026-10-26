/**
 *  @file src/encoder/ram_variants_aws_g5.cpp
 *  @author Ash Vardanian
 *  @date October 25, 2026
 *  @brief The aws_g5 unit: its encoder variant runs the Armv9.2 NEON kits, or answers no table
 *      where they match its parent's.
 */

#include "encoder_specializations.hpp"

template struct uf::encoder_class_tables<uf::encoder_home_t::ram_k, uf::cpu_class_t::aws_g5_k>;
template struct uf::host_class_tables<uf::cpu_class_t::aws_g5_k>;
