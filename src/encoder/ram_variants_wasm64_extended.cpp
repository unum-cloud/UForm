/**
 *  @file src/encoder/ram_variants_wasm64_extended.cpp
 *  @author Ash Vardanian
 *  @date October 25, 2026
 *  @brief The wasm64_extended unit: its encoder variant runs the relaxed SIMD kits, or answers no
 *      table where they match its parent's.
 */

#include "encoder_specializations.hpp"

template struct uf::encoder_class_tables<uf::encoder_home_t::ram_k, uf::cpu_class_t::wasm64_extended_k>;
