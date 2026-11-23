// Package uform provides Go bindings for UForm, the multimodal encoder and trainer.
//
// The package links the precompiled library, so building it compiles no tower. [Status], [Scalar],
// [Rows], [Tape], [Executor] and [Arena] are shared by every verb; [Model] encodes text, images,
// video and audio, and [Trainer] trains one on GPUs, joined across hosts by a [Mesh].
//
// File: golang/uform.go
// Author: Ash Vardanian
// Date: November 18, 2026
package uform
