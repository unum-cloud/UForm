/**
 *  @file golang/shim.c
 *  @author Ash Vardanian
 *  @date November 18, 2026
 *  @brief The address of the Go-side trampoline, taken where C can take it.
 */
#include "shim.h"

uform_progress_callback_t uform_go_continues(void) { return &uformContinues; }
