// The model lifecycle: loading an encoder into RAM or VRAM, and its receivers.
//
// File: golang/model.go
// Author: Ash Vardanian
// Date: November 18, 2026

package uform

import "unsafe"

/*
// The loads report through a progress handler, so they claim only that they retain no pointer.
#cgo noescape uform_model_load_ram
#cgo noescape uform_model_load_ram_buffer
#cgo noescape uform_model_load_vram
#cgo noescape uform_model_load_vram_buffer
#cgo noescape uform_model_describe
#cgo nocallback uform_model_describe

#include "uform/model.h"
*/
import "C"

// region: Configs

// ModelDescription is what a loaded model reports about itself.
//
// Mirrors uform_model_description_t field for field, so the library writes it in place.
type ModelDescription struct {
	// Dimensions is the scalars per embedding row; Matryoshka prefixes of it are embeddings too.
	Dimensions uint
	// Context is the tokens one item may span before it is truncated: bytes of text, patches of an
	// image, tubes of a clip, frames of a recording.
	Context uint
	// Quantization is the scalar the transformer blocks run in.
	Quantization Scalar
	// MemoryUsage is the bytes the model holds; activations are the executor's.
	MemoryUsage uint
	// CPUClass is the class a RAM model's tower was packed for, CPUClassUnknown for a VRAM model.
	CPUClass CPUClass
	// GPUClass is the class every replica of a VRAM model runs as, GPUClassUnknown for a RAM model.
	GPUClass GPUClass
	// TokenWidth is the scalars per token vector, zero when the checkpoint carries no token head.
	TokenWidth uint
	// ChoiceTemperatureGroups counts the question types the choice temperatures were calibrated
	// over: zero without the byte decoder, one when uncalibrated.
	ChoiceTemperatureGroups uint
}

// ModelLoadRamConfig is how ModelLoadRam and ModelLoadRamBuffer copy a checkpoint into host
// memory; nil derives every field.
//
// Mirrors uform_model_load_ram_t and uform_model_load_ram_buffer_t beside their path or buffer.
type ModelLoadRamConfig struct {
	// Executor is the one the load runs on; nil loads on the calling goroutine.
	Executor *Executor
	// MemoryDomains are the NUMA nodes to place the weights in; empty derives every node.
	MemoryDomains []int32
	// CPUClassCeiling is the newest class to pack and dispatch as; zero sets no ceiling.
	CPUClassCeiling CPUClass
	// Quantization is the scalar the transformer blocks run in: ScalarBF16 as the checkpoint
	// stores it, or ScalarE4M3 quantized at load; zero keeps ScalarBF16.
	Quantization Scalar
	// Progress hears the tensors placed and may stop the load, which then creates no model.
	Progress ProgressCallback
	// ProgressSteps is the reports the load makes at most; zero derives ten.
	ProgressSteps uint
}

// ModelLoadVramConfig is how ModelLoadVram and ModelLoadVramBuffer copy a checkpoint onto GPUs;
// nil derives every field.
//
// Mirrors uform_model_load_vram_t and uform_model_load_vram_buffer_t beside their path or buffer.
type ModelLoadVramConfig struct {
	// Executor is the one the upload runs on, one stream per device; nil uploads on the calling
	// goroutine.
	Executor *Executor
	// Devices are the ordinals GPUDevices lists, each holding a replica; empty derives one free
	// device.
	Devices []int32
	// GPUClassCeiling is the newest class every device dispatches as; zero sets no ceiling.
	GPUClassCeiling GPUClass
	// Quantization is the scalar the transformer blocks run in, as for ModelLoadRamConfig.
	Quantization Scalar
	// Progress hears the tensors placed and may stop the load, which then creates no model.
	Progress ProgressCallback
	// ProgressSteps is the reports the load makes at most; zero derives ten.
	ProgressSteps uint
}

// Every struct the library writes in place matches its C twin byte for byte.
var _ = [1]byte{}[unsafe.Sizeof(ModelDescription{})-C.sizeof_uform_model_description_t]

// blocks is the scalar a load runs its blocks in: the one asked for, or the checkpoint's own.
func blocks(quantization Scalar) C.uform_scalar_t {
	if quantization == ScalarUnknown {
		return C.uform_scalar_bf16_k
	}
	return C.uform_scalar_t(quantization)
}

// endregion: Configs

// region: Model

// Model is a loaded encoder: its weights, their packed copies, and on a GPU the session its
// attention runs on, shared by every executor of its residency. Its embedded ModelOn runs every
// verb on the calling goroutine.
//
// Mirrors uform_model_t; Close calls uform_model_free.
type Model struct {
	ModelOn
	handle C.uform_model_t
}

// modelOf wraps the handle a load wrote.
func modelOf(handle C.uform_model_t, err error) (*Model, error) {
	if err != nil {
		return nil, err
	}
	model := &Model{handle: handle}
	model.ModelOn = ModelOn{modelReceiver{receiver[C.uform_model_t]{handle: &model.handle}}}
	return model, nil
}

// ModelLoadRam copies a safetensors checkpoint file into host memory and packs it for the CPU.
// The checkpoint carries the model's geometry.
//
// Calls uform_model_load_ram, blocking.
func ModelLoadRam(path string, config *ModelLoadRamConfig) (*Model, error) {
	options := configOf(config)
	executor, err := options.Executor.lower()
	if err != nil {
		return nil, err
	}
	var lent lent[C.uform_model_t]
	arguments := open[C.uform_model_load_ram_t](&lent)
	*arguments = C.uform_model_load_ram_t{
		model:        lent.result,
		executor:     executor,
		outcome:      lent.outcome(),
		path:         lent.text(path),
		ram:          lent.ram(options.MemoryDomains, options.CPUClassCeiling),
		quantization: blocks(options.Quantization),
		progress:     lent.progress(options.Progress, options.ProgressSteps),
	}
	C.uform_model_load_ram(arguments)
	return modelOf(block(&lent, nil))
}

// ModelLoadRamBuffer copies a checkpoint held in memory into host memory and packs it for the CPU;
// nothing of buffer is kept.
//
// Calls uform_model_load_ram_buffer, blocking.
func ModelLoadRamBuffer(buffer []byte, config *ModelLoadRamConfig) (*Model, error) {
	options := configOf(config)
	executor, err := options.Executor.lower()
	if err != nil {
		return nil, err
	}
	var lent lent[C.uform_model_t]
	arguments := open[C.uform_model_load_ram_buffer_t](&lent)
	*arguments = C.uform_model_load_ram_buffer_t{
		model:        lent.result,
		executor:     executor,
		outcome:      lent.outcome(),
		buffer:       lent.bytes(buffer),
		ram:          lent.ram(options.MemoryDomains, options.CPUClassCeiling),
		quantization: blocks(options.Quantization),
		progress:     lent.progress(options.Progress, options.ProgressSteps),
	}
	C.uform_model_load_ram_buffer(arguments)
	return modelOf(block(&lent, nil))
}

// ModelLoadVram copies a checkpoint file onto the named GPUs and packs it there, replicated on
// each, since a model is small next to the activations it serves.
//
// Calls uform_model_load_vram, blocking.
func ModelLoadVram(path string, config *ModelLoadVramConfig) (*Model, error) {
	options := configOf(config)
	executor, err := options.Executor.lower()
	if err != nil {
		return nil, err
	}
	var lent lent[C.uform_model_t]
	arguments := open[C.uform_model_load_vram_t](&lent)
	*arguments = C.uform_model_load_vram_t{
		model:        lent.result,
		executor:     executor,
		outcome:      lent.outcome(),
		path:         lent.text(path),
		vram:         lent.vram(options.Devices, options.GPUClassCeiling),
		quantization: blocks(options.Quantization),
		progress:     lent.progress(options.Progress, options.ProgressSteps),
	}
	C.uform_model_load_vram(arguments)
	return modelOf(block(&lent, nil))
}

// ModelLoadVramBuffer copies a checkpoint held in memory onto the named GPUs and packs it there.
//
// Calls uform_model_load_vram_buffer, blocking.
func ModelLoadVramBuffer(buffer []byte, config *ModelLoadVramConfig) (*Model, error) {
	options := configOf(config)
	executor, err := options.Executor.lower()
	if err != nil {
		return nil, err
	}
	var lent lent[C.uform_model_t]
	arguments := open[C.uform_model_load_vram_buffer_t](&lent)
	*arguments = C.uform_model_load_vram_buffer_t{
		model:        lent.result,
		executor:     executor,
		outcome:      lent.outcome(),
		buffer:       lent.bytes(buffer),
		vram:         lent.vram(options.Devices, options.GPUClassCeiling),
		quantization: blocks(options.Quantization),
		progress:     lent.progress(options.Progress, options.ProgressSteps),
	}
	C.uform_model_load_vram_buffer(arguments)
	return modelOf(block(&lent, nil))
}

// Close frees the model, waiting for every deferred call naming it. It is idempotent, and a closed
// model refuses every later verb with StatusUninitialized.
func (model *Model) Close() error {
	if model == nil || model.handle == nil {
		return nil
	}
	C.uform_model_free(model.handle)
	model.handle = nil
	return nil
}

// endregion: Model

// region: Receivers

// modelReceiver is a model and the executor its verbs run on, nil for the calling goroutine.
type modelReceiver struct {
	receiver[C.uform_model_t]
}

// ModelOn is the receiver whose verbs block until their call is final: on the calling goroutine
// when embedded in a Model, and on an executor when Model.On makes it. Binding-only.
type ModelOn struct {
	modelReceiver
}

// ModelDeferredOn is the receiver whose verbs submit their calls to an executor and answer a
// Deferred, offering every verb of ModelOn with the same arguments. Binding-only.
//
// The library reads a deferred call's inputs until the call is final, so the tapes and slices a
// verb is handed must not change until then; its arena is borrowed, as a blocking verb borrows it.
type ModelDeferredOn struct {
	modelReceiver
}

// On is the receiver whose verbs block on executor, failing StatusContended while another call
// holds it. Binding-only.
func (on ModelOn) On(executor *Executor) ModelOn {
	on.executor = executor
	return on
}

// Deferred is the receiver whose verbs submit their calls to executor, each answering a Deferred
// from the same arguments. Binding-only.
func (on ModelOn) Deferred(executor *Executor) ModelDeferredOn {
	on.executor = executor
	return ModelDeferredOn{on.modelReceiver}
}

// Describe reports the model's geometry, placement, classes and heads in one call.
//
// Calls uform_model_describe.
func (on ModelOn) Describe() (ModelDescription, error) {
	var lent lent[ModelDescription]
	return block(&lent, on.describe(&lent, C.uform_completion_blocking_k))
}

// Describe submits ModelOn.Describe.
func (on ModelDeferredOn) Describe() (*Deferred[ModelDescription], error) {
	deferred := new(Deferred[ModelDescription])
	return submitted(deferred, on.describe(&deferred.lent, C.uform_completion_deferred_k))
}

func (receiver modelReceiver) describe(lent *lent[ModelDescription], completion C.uform_completion_t) error {
	model, executor, err := receiver.lower()
	if err != nil {
		return err
	}
	arguments := open[C.uform_model_describe_t](lent)
	*arguments = C.uform_model_describe_t{
		model:       model,
		executor:    executor,
		outcome:     lent.outcome(),
		completion:  completion,
		description: (*C.uform_model_description_t)(unsafe.Pointer(lent.result)),
	}
	C.uform_model_describe(arguments)
	return nil
}

// endregion: Receivers
