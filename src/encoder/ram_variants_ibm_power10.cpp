/**
 *  @file src/encoder/ram_variants_ibm_power10.cpp
 *  @author Ash Vardanian
 *  @date November 14, 2026
 *  @brief The ibm_power10 unit: its encoder variant runs the VSX and MMA kits, or answers no table
 *      where they match its parent's.
 */

#include "encoder_specializations.hpp"

template struct uf::encoder_class_tables<uf::encoder_home_t::ram_k, uf::cpu_class_t::ibm_power10_k>;
template struct uf::host_class_tables<uf::cpu_class_t::ibm_power10_k>;
