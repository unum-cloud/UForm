/**
 *  @file swift/shim.c
 *  @author Ash Vardanian
 *  @date October 17, 2026
 *  @brief The one translation unit of the @c UFormC target, which exists so SwiftPM has a compiled half to hang
 *      the public headers off.
 *
 *  A target declaring @c publicHeadersPath is what puts @c c/ on the header search path of everything importing
 *  the module, and the headers reach each other through that path.
 *
 *  Nothing here implements the ABI - the engine is built by CMake into @c libuform_static and linked, since
 *  @c src/ pulls the NumKong, ForkUnion and StringZilla stack and carries CUDA and HIP siblings SwiftPM cannot
 *  drive. Including the headers does buy one check: the C surface has to keep parsing as C, not merely as C++.
 */
#include "uform/types.h"

#include "uform/model.h"

#include "uform/encode.h"
#include "uform/train.h"
