/**
 *  @file src/encoder/ram_variants_aws_g2.cpp
 *  @author Ash Vardanian
 *  @date October 25, 2026
 *  @brief The aws_g2 unit: its encoder variant runs the bare NEON kits, or answers no table where
 *      they match its parent's.
 */

#include "encoder_specializations.hpp"

template struct uf::encoder_class_tables<uf::encoder_home_t::ram_k, uf::cpu_class_t::aws_g2_k>;
template struct uf::host_class_tables<uf::cpu_class_t::aws_g2_k>;
