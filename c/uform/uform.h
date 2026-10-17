/**
 *  @file c/uform/uform.h
 *  @author Ash Vardanian
 *  @date October 1, 2026
 *  @brief The umbrella of the UForm C interface: the model lifecycle, the forward passes, the training steps,
 *      and the vocabulary they share, behind one include.
 *
 *  A caller only encoding may include @c encode.h alone, and a caller only training @c train.h; this one
 *  exists so that a binding generator, or a caller doing both, names the whole interface once.
 */
#ifndef UNUM_UFORM_H
#define UNUM_UFORM_H

#include "uform/types.h"

#include "uform/model.h"

#include "uform/encode.h"
#include "uform/train.h"

#endif // UNUM_UFORM_H
