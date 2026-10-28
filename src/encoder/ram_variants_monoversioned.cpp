/**
 *  @file src/encoder/ram_variants_monoversioned.cpp
 *  @author Ash Vardanian
 *  @date October 28, 2026
 *  @brief The monoversioned unit: its encoder variant runs the kits of every capability the
 *      libraries compiled, the one class of a monoversioned build.
 */

#include "encoder_specializations.hpp"

template struct uf::encoder_class_tables<uf::encoder_home_t::ram_k, uf::cpu_class_t::monoversioned_k>;
template struct uf::host_class_tables<uf::cpu_class_t::monoversioned_k>;
