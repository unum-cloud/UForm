/**
 *  @file src/encoder/ram_variants_alibaba_c930.cpp
 *  @author Ash Vardanian
 *  @date October 25, 2026
 *  @brief The alibaba_c930 unit: its encoder variant runs the RISC-V vector kits, or answers no
 *      table where they match its parent's.
 */

#include "encoder_specializations.hpp"

template struct uf::encoder_class_tables<uf::encoder_home_t::ram_k, uf::cpu_class_t::alibaba_c930_k>;
