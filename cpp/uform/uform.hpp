/**
 *  @file cpp/uform/uform.hpp
 *  @author Ash Vardanian
 *  @date November 18, 2026
 *  @brief The umbrella of the UForm C++ SDK: the header-only layer over the C interface in
 *      @c uform/uform.h, behind one include.
 *
 *  Every header here includes only the C headers under @c c/uform and the standard library, answers
 *  <tt>auto [value, status]</tt> pairs, and throws nothing. A caller only encoding may include
 *  @c uform/model.hpp alone, and a caller only training @c uform/train.hpp.
 */
#pragma once

#include "uform/types.hpp" // `status_t`, `expected`, `deferred`, `awaiter`, `executor_t`, `arena_t`, `gathered`

#include "uform/encode.hpp" // `encoded_t`, `encode_outputs_t`, `text_encode_config_t`, `text_choose_config_t`
#include "uform/model.hpp"  // `model_t`, `model_on`
#include "uform/train.hpp"  // `mesh_t`, `trainer_t`, `trainer_on`, `trainer_step_config_t`
