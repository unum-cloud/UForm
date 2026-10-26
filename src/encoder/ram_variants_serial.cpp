/**
 *  @file src/encoder/ram_variants_serial.cpp
 *  @author Ash Vardanian
 *  @date October 25, 2026
 *  @brief The serial unit: its encoder variant runs the portable floor, NumKong's serial kits, or
 *      answers no table where they match its parent's.
 */

#include "encoder_specializations.hpp"

template struct uf::encoder_class_tables<uf::encoder_home_t::ram_k, uf::cpu_class_t::serial_k>;
template struct uf::host_class_tables<uf::cpu_class_t::serial_k>;
