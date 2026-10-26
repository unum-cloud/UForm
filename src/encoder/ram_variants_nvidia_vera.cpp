/**
 *  @file src/encoder/ram_variants_nvidia_vera.cpp
 *  @author Ash Vardanian
 *  @date October 25, 2026
 *  @brief The nvidia_vera unit: its encoder variant runs the NEON FP8 kits, or answers no table
 *      where they match its parent's.
 */

#include "encoder_specializations.hpp"

template struct uf::encoder_class_tables<uf::encoder_home_t::ram_k, uf::cpu_class_t::nvidia_vera_k>;
template struct uf::host_class_tables<uf::cpu_class_t::nvidia_vera_k>;
