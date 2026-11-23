/**
 *  @file golang/shim.h
 *  @author Ash Vardanian
 *  @date November 18, 2026
 *  @brief Declarations the Go binding needs and the C header cannot give it: the Go-side
 *      trampoline a progress handler installs as a C callback.
 *
 *  The definition lives in @c shim.c because a Go file carrying an @c //export directive may hold
 *  only declarations in its preamble.
 */
#ifndef UNUM_UFORM_GO_SHIM_H
#define UNUM_UFORM_GO_SHIM_H

#include "uform/uform.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 *  @brief The callback Go defines and C calls, the one symbol here that crosses in that direction,
 *      which is why it alone wears Go's spelling.
 *
 *  cgo emits a C symbol named exactly as the Go function, and Go spells an unexported function in
 *  camel case, so the name is not this file's to choose.
 */
uform_continuation_t uformContinues(size_t completed, size_t total, void* state);

/** Hands back the trampoline above as a function pointer. cgo renders a C function as a Go wrapper
 *  rather than an address, so the pointer has to be taken on this side. */
uform_progress_callback_t uform_go_continues(void);

#ifdef __cplusplus
}
#endif

#endif // UNUM_UFORM_GO_SHIM_H
