/**
 *  @file src/encoder/vram_variants_nvidia.cpp
 *  @author Ash Vardanian
 *  @date October 25, 2026
 *  @brief The NVIDIA floor unit: its encoder variant runs NumKong's generic CUDA kits on every
 *      NVIDIA device without a class of its own.
 */

#include "encoder_specializations.hpp"

template struct uf::encoder_class_tables<uf::encoder_home_t::vram_nvidia_k, uf::gpu_class_t::unknown_k>;
