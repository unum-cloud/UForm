// The forward passes: text, images, video and audio in, embeddings, states and token vectors out.
//
// File: golang/encode.go
// Author: Ash Vardanian
// Date: November 18, 2026

package uform

import "unsafe"

/*
// Every forward reports through a progress handler, so each claims only that it retains no pointer.
#cgo noescape uform_text_encode
#cgo noescape uform_text_choose
#cgo noescape uform_image_encode
#cgo noescape uform_image_encode_pixels
#cgo noescape uform_video_encode
#cgo noescape uform_audio_encode
#cgo noescape uform_audio_encode_samples

#include "uform/encode.h"
*/
import "C"

// region: Outputs

// EncodeOutputs names the surfaces an encode verb keeps from its one forward, after the C fields
// they fill; | joins them, and a surface left out costs nothing past the final norm. Binding-only.
type EncodeOutputs uint32

const (
	// EncodeOutputsEmbeddings keeps one pooled embedding per item, past the attention pool and the
	// projection head.
	EncodeOutputsEmbeddings EncodeOutputs = 1 << iota
	// EncodeOutputsStates keeps one final-norm state per token left after truncation.
	EncodeOutputsStates
	// EncodeOutputsTokenVectors keeps one unit vector of the token head per token left after
	// truncation; a model without the head fails the call with StatusUnsupported.
	EncodeOutputsTokenVectors
)

// Encoded is what an encode verb wrote into the arena it was given: the surfaces its EncodeOutputs
// named, each with a zero Count when left out, and none normalized. Binding-only.
type Encoded struct {
	// Embeddings holds one pooled embedding of the model's dimensions per item.
	Embeddings Rows
	// States holds one final-norm state per token left after truncation, item by item.
	States RowsTape
	// TokenVectors holds one unit vector of the model's token width per token left after
	// truncation, item by item.
	TokenVectors RowsTape
}

// kept points the surfaces outputs names into encoded, sized for count items, and nil for the
// rest.
func kept(outputs EncodeOutputs, encoded *Encoded, count uint) (*C.uform_rows_view_t, *C.uform_rows_tape_view_t, *C.uform_rows_tape_view_t) {
	var embeddings *C.uform_rows_view_t
	var states, tokenVectors *C.uform_rows_tape_view_t
	if outputs&EncodeOutputsEmbeddings != 0 {
		embeddings, encoded.Embeddings.count = &encoded.Embeddings.raw, count
	}
	if outputs&EncodeOutputsStates != 0 {
		states, encoded.States.count = &encoded.States.raw, count
	}
	if outputs&EncodeOutputsTokenVectors != 0 {
		tokenVectors, encoded.TokenVectors.count = &encoded.TokenVectors.raw, count
	}
	return embeddings, states, tokenVectors
}

// endregion: Outputs

// region: Configs

// TextEncodeStats is what a text call reports beside its outputs, overwritten by the call.
//
// Mirrors uform_text_encode_stats_t.
type TextEncodeStats struct {
	// Truncated counts the items cut at the model's context.
	Truncated uint
	// Tokens counts the tokens the call encoded, after truncation.
	Tokens uint
}

// ImageEncodeStats is what an image call reports beside its outputs, overwritten by the call.
//
// Mirrors uform_image_encode_stats_t.
type ImageEncodeStats struct {
	// Truncated counts the items whose patch grid was cut at the model's context.
	Truncated uint
	// Tokens counts the patches the call encoded, after truncation.
	Tokens uint
}

// VideoEncodeStats is what a video call reports beside its outputs, overwritten by the call.
//
// Mirrors uform_video_encode_stats_t.
type VideoEncodeStats struct {
	// Truncated counts the items cut at the model's context in tubes.
	Truncated uint
	// Tokens counts the tubes the call encoded, after truncation.
	Tokens uint
}

// AudioEncodeStats is what an audio call reports beside its outputs, overwritten by the call.
//
// Mirrors uform_audio_encode_stats_t.
type AudioEncodeStats struct {
	// Truncated counts the items cut at the model's context in frames.
	Truncated uint
	// Tokens counts the frames the call encoded, after truncation.
	Tokens uint
}

// Every report the library writes in place matches its C twin byte for byte.
var (
	_ = [1]byte{}[unsafe.Sizeof(TextEncodeStats{})-C.sizeof_uform_text_encode_stats_t]
	_ = [1]byte{}[unsafe.Sizeof(ImageEncodeStats{})-C.sizeof_uform_image_encode_stats_t]
	_ = [1]byte{}[unsafe.Sizeof(VideoEncodeStats{})-C.sizeof_uform_video_encode_stats_t]
	_ = [1]byte{}[unsafe.Sizeof(AudioEncodeStats{})-C.sizeof_uform_audio_encode_stats_t]
)

// TextEncodeConfig holds the knobs of TextEncode; nil derives every field.
//
// Mirrors the knobs of uform_text_encode_t.
type TextEncodeConfig struct {
	// Progress hears the items encoded and may stop the call.
	Progress ProgressCallback
	// ProgressSteps is the reports the call makes at most; zero derives ten.
	ProgressSteps uint
	// Stats is the report the call overwrites, kept by the caller until it is final; nil skips it.
	Stats *TextEncodeStats
}

// ImageEncodeConfig holds the knobs of ImageEncode and ImageEncodePixels; nil derives every field.
//
// Mirrors the knobs of uform_image_encode_t and uform_image_encode_pixels_t.
type ImageEncodeConfig struct {
	// Progress hears the items encoded and may stop the call.
	Progress ProgressCallback
	// ProgressSteps is the reports the call makes at most; zero derives ten.
	ProgressSteps uint
	// Stats is the report the call overwrites, kept by the caller until it is final; nil skips it.
	Stats *ImageEncodeStats
}

// VideoEncodeConfig holds the knobs of VideoEncode; nil derives every field.
//
// Mirrors the knobs of uform_video_encode_t.
type VideoEncodeConfig struct {
	// Progress hears the items encoded and may stop the call.
	Progress ProgressCallback
	// ProgressSteps is the reports the call makes at most; zero derives ten.
	ProgressSteps uint
	// Stats is the report the call overwrites, kept by the caller until it is final; nil skips it.
	Stats *VideoEncodeStats
}

// AudioEncodeConfig holds the knobs of AudioEncode and AudioEncodeSamples; nil derives every field.
//
// Mirrors the knobs of uform_audio_encode_t and uform_audio_encode_samples_t.
type AudioEncodeConfig struct {
	// Progress hears the items encoded and may stop the call.
	Progress ProgressCallback
	// ProgressSteps is the reports the call makes at most; zero derives ten.
	ProgressSteps uint
	// Stats is the report the call overwrites, kept by the caller until it is final; nil skips it.
	Stats *AudioEncodeStats
}

// TextChooseConfig holds the knobs of TextChoose; nil derives every field.
//
// Mirrors the knobs of uform_text_choose_t.
type TextChooseConfig struct {
	// TemperatureGroups holds the question type of each item, below the model's
	// ChoiceTemperatureGroups; empty puts every item in group zero.
	TemperatureGroups []uint32
	// Progress hears the items scored and may stop the call.
	Progress ProgressCallback
	// ProgressSteps is the reports the call makes at most; zero derives ten.
	ProgressSteps uint
	// Stats is the report the call overwrites, kept by the caller until it is final; nil skips it.
	Stats *TextEncodeStats
}

// endregion: Configs

// region: Verbs

// TextEncode encodes UTF-8 texts through one forward, a token per byte, since models are
// tokenizer-free, keeping the surfaces outputs names as outputsScalar in arena.
//
// Calls uform_text_encode.
func (on ModelOn) TextEncode(texts Tape, outputs EncodeOutputs, outputsScalar Scalar, arena *Arena,
	config *TextEncodeConfig) (Encoded, error) {
	var call callWithResult[Encoded]
	return block(&call, on.textEncode(&call, C.uform_completion_blocking_k, texts, outputs, outputsScalar, arena, config))
}

// TextEncode submits ModelOn.TextEncode.
func (on ModelDeferredOn) TextEncode(texts Tape, outputs EncodeOutputs, outputsScalar Scalar, arena *Arena,
	config *TextEncodeConfig) (*Deferred[Encoded], error) {
	deferred := new(Deferred[Encoded])
	return submitted(deferred, on.textEncode(&deferred.call, C.uform_completion_deferred_k, texts, outputs, outputsScalar, arena, config))
}

// TextChoose scores each item's legal choices, one byte each, after its UTF-8 state in texts,
// through one forward, answering one ScalarF32 probability per choice in arena, a softmax over the
// item's choices alone. A model without the byte decoder fails it with StatusUnsupported.
//
// Calls uform_text_choose.
func (on ModelOn) TextChoose(texts, choices Tape, arena *Arena, config *TextChooseConfig) (TapeView, error) {
	var call callWithResult[TapeView]
	return block(&call, on.textChoose(&call, C.uform_completion_blocking_k, texts, choices, arena, config))
}

// TextChoose submits ModelOn.TextChoose.
func (on ModelDeferredOn) TextChoose(texts, choices Tape, arena *Arena, config *TextChooseConfig) (*Deferred[TapeView], error) {
	deferred := new(Deferred[TapeView])
	return submitted(deferred, on.textChoose(&deferred.call, C.uform_completion_deferred_k, texts, choices, arena, config))
}

// ImageEncode decodes and encodes JPEG, PNG or WebP files; an undecodable file fails the call with
// StatusMalformed.
//
// Calls uform_image_encode.
func (on ModelOn) ImageEncode(images Tape, outputs EncodeOutputs, outputsScalar Scalar, arena *Arena,
	config *ImageEncodeConfig) (Encoded, error) {
	var call callWithResult[Encoded]
	return block(&call, on.imageEncode(&call, C.uform_completion_blocking_k, images, outputs, outputsScalar, arena, config))
}

// ImageEncode submits ModelOn.ImageEncode.
func (on ModelDeferredOn) ImageEncode(images Tape, outputs EncodeOutputs, outputsScalar Scalar, arena *Arena,
	config *ImageEncodeConfig) (*Deferred[Encoded], error) {
	deferred := new(Deferred[Encoded])
	return submitted(deferred, on.imageEncode(&deferred.call, C.uform_completion_deferred_k, images, outputs, outputsScalar, arena, config))
}

// ImageEncodePixels encodes decoded images of interleaved 8-bit RGB, item i holding heights[i] rows
// of widths[i] pixels; an item of another length fails with StatusShapeMismatch.
//
// Calls uform_image_encode_pixels.
func (on ModelOn) ImageEncodePixels(pixels Tape, heights, widths []uint32, outputs EncodeOutputs,
	outputsScalar Scalar, arena *Arena, config *ImageEncodeConfig) (Encoded, error) {
	var call callWithResult[Encoded]
	return block(&call, on.imageEncodePixels(&call, C.uform_completion_blocking_k, pixels, heights, widths, outputs, outputsScalar, arena, config))
}

// ImageEncodePixels submits ModelOn.ImageEncodePixels.
func (on ModelDeferredOn) ImageEncodePixels(pixels Tape, heights, widths []uint32, outputs EncodeOutputs,
	outputsScalar Scalar, arena *Arena, config *ImageEncodeConfig) (*Deferred[Encoded], error) {
	deferred := new(Deferred[Encoded])
	return submitted(deferred, on.imageEncodePixels(&deferred.call, C.uform_completion_deferred_k, pixels, heights, widths, outputs, outputsScalar, arena, config))
}

// VideoEncode decodes and encodes MP4 or WebM clips.
//
// Calls uform_video_encode.
func (on ModelOn) VideoEncode(videos Tape, outputs EncodeOutputs, outputsScalar Scalar, arena *Arena,
	config *VideoEncodeConfig) (Encoded, error) {
	var call callWithResult[Encoded]
	return block(&call, on.videoEncode(&call, C.uform_completion_blocking_k, videos, outputs, outputsScalar, arena, config))
}

// VideoEncode submits ModelOn.VideoEncode.
func (on ModelDeferredOn) VideoEncode(videos Tape, outputs EncodeOutputs, outputsScalar Scalar, arena *Arena,
	config *VideoEncodeConfig) (*Deferred[Encoded], error) {
	deferred := new(Deferred[Encoded])
	return submitted(deferred, on.videoEncode(&deferred.call, C.uform_completion_deferred_k, videos, outputs, outputsScalar, arena, config))
}

// AudioEncode decodes and encodes WAV, FLAC, MP3 or Opus recordings, resampled to mono.
//
// Calls uform_audio_encode.
func (on ModelOn) AudioEncode(recordings Tape, outputs EncodeOutputs, outputsScalar Scalar, arena *Arena,
	config *AudioEncodeConfig) (Encoded, error) {
	var call callWithResult[Encoded]
	return block(&call, on.audioEncode(&call, C.uform_completion_blocking_k, recordings, outputs, outputsScalar, arena, config))
}

// AudioEncode submits ModelOn.AudioEncode.
func (on ModelDeferredOn) AudioEncode(recordings Tape, outputs EncodeOutputs, outputsScalar Scalar, arena *Arena,
	config *AudioEncodeConfig) (*Deferred[Encoded], error) {
	deferred := new(Deferred[Encoded])
	return submitted(deferred, on.audioEncode(&deferred.call, C.uform_completion_deferred_k, recordings, outputs, outputsScalar, arena, config))
}

// AudioEncodeSamples encodes decoded mono ScalarF32 recordings sharing one sampleRate, resampled
// to the model's rate where it differs.
//
// Calls uform_audio_encode_samples.
func (on ModelOn) AudioEncodeSamples(samples Tape, sampleRate uint32, outputs EncodeOutputs, outputsScalar Scalar,
	arena *Arena, config *AudioEncodeConfig) (Encoded, error) {
	var call callWithResult[Encoded]
	return block(&call, on.audioEncodeSamples(&call, C.uform_completion_blocking_k, samples, sampleRate, outputs, outputsScalar, arena, config))
}

// AudioEncodeSamples submits ModelOn.AudioEncodeSamples.
func (on ModelDeferredOn) AudioEncodeSamples(samples Tape, sampleRate uint32, outputs EncodeOutputs,
	outputsScalar Scalar, arena *Arena, config *AudioEncodeConfig) (*Deferred[Encoded], error) {
	deferred := new(Deferred[Encoded])
	return submitted(deferred, on.audioEncodeSamples(&deferred.call, C.uform_completion_deferred_k, samples, sampleRate, outputs, outputsScalar, arena, config))
}

// endregion: Verbs

// region: Calls

func (receiver modelReceiver) textEncode(call *callWithResult[Encoded], completion C.uform_completion_t, texts Tape,
	outputs EncodeOutputs, outputsScalar Scalar, arena *Arena, config *TextEncodeConfig) error {
	options := configOf(config)
	model, executor, outputsArena, err := receiver.lowerReadableInto(arena)
	if err != nil {
		return err
	}
	arguments := open[C.uform_text_encode_t](call)
	embeddings, states, tokenVectors := kept(outputs, call.result, texts.Entries())
	*arguments = C.uform_text_encode_t{
		model:          model,
		executor:       executor,
		outcome:        call.outcome(),
		completion:     completion,
		count:          C.size_t(texts.Entries()),
		texts:          texts.lower(&call.call),
		embeddings:     embeddings,
		states:         states,
		token_vectors:  tokenVectors,
		arena:          outputsArena,
		outputs_scalar: C.uform_scalar_t(outputsScalar),
		progress:       call.progress(options.Progress, options.ProgressSteps),
		stats:          (*C.uform_text_encode_stats_t)(call.optional(unsafe.Pointer(options.Stats))),
	}
	C.uform_text_encode(arguments)
	return nil
}

func (receiver modelReceiver) textChoose(call *callWithResult[TapeView], completion C.uform_completion_t, texts, choices Tape,
	arena *Arena, config *TextChooseConfig) error {
	options := configOf(config)
	model, executor, outputsArena, err := receiver.lowerReadableInto(arena)
	if err != nil {
		return err
	}
	// The call has one count, so only this side sees the texts, the choices and the groups disagree.
	if choices.Entries() != texts.Entries() ||
		(len(options.TemperatureGroups) != 0 && uint(len(options.TemperatureGroups)) != texts.Entries()) {
		return fail(StatusShapeMismatch, "every item needs its choices and, if any, one group")
	}
	arguments := open[C.uform_text_choose_t](call)
	call.result.count = texts.Entries()
	*arguments = C.uform_text_choose_t{
		model:              model,
		executor:           executor,
		outcome:            call.outcome(),
		completion:         completion,
		count:              C.size_t(texts.Entries()),
		texts:              texts.lower(&call.call),
		choices:            choices.lower(&call.call),
		temperature_groups: call.uint32s(options.TemperatureGroups),
		probabilities:      &call.result.raw,
		arena:              outputsArena,
		progress:           call.progress(options.Progress, options.ProgressSteps),
		stats:              (*C.uform_text_encode_stats_t)(call.optional(unsafe.Pointer(options.Stats))),
	}
	C.uform_text_choose(arguments)
	return nil
}

func (receiver modelReceiver) imageEncode(call *callWithResult[Encoded], completion C.uform_completion_t, images Tape,
	outputs EncodeOutputs, outputsScalar Scalar, arena *Arena, config *ImageEncodeConfig) error {
	options := configOf(config)
	model, executor, outputsArena, err := receiver.lowerReadableInto(arena)
	if err != nil {
		return err
	}
	arguments := open[C.uform_image_encode_t](call)
	embeddings, states, tokenVectors := kept(outputs, call.result, images.Entries())
	*arguments = C.uform_image_encode_t{
		model:          model,
		executor:       executor,
		outcome:        call.outcome(),
		completion:     completion,
		count:          C.size_t(images.Entries()),
		images:         images.lower(&call.call),
		embeddings:     embeddings,
		states:         states,
		token_vectors:  tokenVectors,
		arena:          outputsArena,
		outputs_scalar: C.uform_scalar_t(outputsScalar),
		progress:       call.progress(options.Progress, options.ProgressSteps),
		stats:          (*C.uform_image_encode_stats_t)(call.optional(unsafe.Pointer(options.Stats))),
	}
	C.uform_image_encode(arguments)
	return nil
}

func (receiver modelReceiver) imageEncodePixels(call *callWithResult[Encoded], completion C.uform_completion_t, pixels Tape,
	heights, widths []uint32, outputs EncodeOutputs, outputsScalar Scalar, arena *Arena, config *ImageEncodeConfig) error {
	options := configOf(config)
	model, executor, outputsArena, err := receiver.lowerReadableInto(arena)
	if err != nil {
		return err
	}
	// The call has one count, so only this side sees the pixels and their shapes disagree.
	if uint(len(heights)) != pixels.Entries() || uint(len(widths)) != pixels.Entries() {
		return fail(StatusShapeMismatch, "every item needs one height and one width")
	}
	arguments := open[C.uform_image_encode_pixels_t](call)
	embeddings, states, tokenVectors := kept(outputs, call.result, pixels.Entries())
	*arguments = C.uform_image_encode_pixels_t{
		model:          model,
		executor:       executor,
		outcome:        call.outcome(),
		completion:     completion,
		count:          C.size_t(pixels.Entries()),
		pixels:         pixels.lower(&call.call),
		heights:        call.uint32s(heights),
		widths:         call.uint32s(widths),
		embeddings:     embeddings,
		states:         states,
		token_vectors:  tokenVectors,
		arena:          outputsArena,
		outputs_scalar: C.uform_scalar_t(outputsScalar),
		progress:       call.progress(options.Progress, options.ProgressSteps),
		stats:          (*C.uform_image_encode_stats_t)(call.optional(unsafe.Pointer(options.Stats))),
	}
	C.uform_image_encode_pixels(arguments)
	return nil
}

func (receiver modelReceiver) videoEncode(call *callWithResult[Encoded], completion C.uform_completion_t, videos Tape,
	outputs EncodeOutputs, outputsScalar Scalar, arena *Arena, config *VideoEncodeConfig) error {
	options := configOf(config)
	model, executor, outputsArena, err := receiver.lowerReadableInto(arena)
	if err != nil {
		return err
	}
	arguments := open[C.uform_video_encode_t](call)
	embeddings, states, tokenVectors := kept(outputs, call.result, videos.Entries())
	*arguments = C.uform_video_encode_t{
		model:          model,
		executor:       executor,
		outcome:        call.outcome(),
		completion:     completion,
		count:          C.size_t(videos.Entries()),
		videos:         videos.lower(&call.call),
		embeddings:     embeddings,
		states:         states,
		token_vectors:  tokenVectors,
		arena:          outputsArena,
		outputs_scalar: C.uform_scalar_t(outputsScalar),
		progress:       call.progress(options.Progress, options.ProgressSteps),
		stats:          (*C.uform_video_encode_stats_t)(call.optional(unsafe.Pointer(options.Stats))),
	}
	C.uform_video_encode(arguments)
	return nil
}

func (receiver modelReceiver) audioEncode(call *callWithResult[Encoded], completion C.uform_completion_t, recordings Tape,
	outputs EncodeOutputs, outputsScalar Scalar, arena *Arena, config *AudioEncodeConfig) error {
	options := configOf(config)
	model, executor, outputsArena, err := receiver.lowerReadableInto(arena)
	if err != nil {
		return err
	}
	arguments := open[C.uform_audio_encode_t](call)
	embeddings, states, tokenVectors := kept(outputs, call.result, recordings.Entries())
	*arguments = C.uform_audio_encode_t{
		model:          model,
		executor:       executor,
		outcome:        call.outcome(),
		completion:     completion,
		count:          C.size_t(recordings.Entries()),
		recordings:     recordings.lower(&call.call),
		embeddings:     embeddings,
		states:         states,
		token_vectors:  tokenVectors,
		arena:          outputsArena,
		outputs_scalar: C.uform_scalar_t(outputsScalar),
		progress:       call.progress(options.Progress, options.ProgressSteps),
		stats:          (*C.uform_audio_encode_stats_t)(call.optional(unsafe.Pointer(options.Stats))),
	}
	C.uform_audio_encode(arguments)
	return nil
}

func (receiver modelReceiver) audioEncodeSamples(call *callWithResult[Encoded], completion C.uform_completion_t, samples Tape,
	sampleRate uint32, outputs EncodeOutputs, outputsScalar Scalar, arena *Arena, config *AudioEncodeConfig) error {
	options := configOf(config)
	model, executor, outputsArena, err := receiver.lowerReadableInto(arena)
	if err != nil {
		return err
	}
	arguments := open[C.uform_audio_encode_samples_t](call)
	embeddings, states, tokenVectors := kept(outputs, call.result, samples.Entries())
	*arguments = C.uform_audio_encode_samples_t{
		model:          model,
		executor:       executor,
		outcome:        call.outcome(),
		completion:     completion,
		count:          C.size_t(samples.Entries()),
		samples:        samples.lower(&call.call),
		sample_rate:    C.uint32_t(sampleRate),
		embeddings:     embeddings,
		states:         states,
		token_vectors:  tokenVectors,
		arena:          outputsArena,
		outputs_scalar: C.uform_scalar_t(outputsScalar),
		progress:       call.progress(options.Progress, options.ProgressSteps),
		stats:          (*C.uform_audio_encode_stats_t)(call.optional(unsafe.Pointer(options.Stats))),
	}
	C.uform_audio_encode_samples(arguments)
	return nil
}

// endregion: Calls
