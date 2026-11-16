/**
 *  @file src/encoder/ram_variants_aws_g5.cpp
 *  @author Ash Vardanian
 *  @date November 16, 2026
 *  @brief The aws_g5 unit: its encoder variant runs its parent's NEON kits under its own waiter and
 *      atomic reference, or answers no table where they match its parent's.
 */

#include "encoder_specializations.hpp"

template struct uf::encoder_class_tables<uf::vendored_residency_t::ram_k, uf::cpu_class_t::aws_g5_k>;
template struct uf::host_class_tables<uf::cpu_class_t::aws_g5_k>;
