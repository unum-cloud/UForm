/**
 *  @file src/encoder/ram_variants_wasm32_v128.cpp
 *  @author Ash Vardanian
 *  @date October 25, 2026
 *  @brief The wasm32_v128 unit: its encoder variant runs the fixed-width SIMD kits, or answers no
 *      table where they match its parent's.
 */

#include "encoder_specializations.hpp"

template struct uf::encoder_class_tables<uf::encoder_home_t::ram_k, uf::cpu_class_t::wasm32_v128_k>;
