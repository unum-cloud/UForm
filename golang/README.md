# UForm for Go

The Go package binds the stable C ABI in `c/uform/` through cgo and links the precompiled static library, so no tower is compiled in your build.
It needs Go 1.24 or newer with cgo enabled and a C toolchain, and answers every call as `(T, error)`.
Every verb is a batch, and every output whose length the call decides lands in an arena you reuse.

- `Model` encodes text, images, video and audio into pooled embeddings, per-token states and token vectors, and scores calibrated choices over text.
- `Trainer` trains one on GPUs, joined across hosts by a `Mesh`.
- `Executor` owns the threads or streams every call runs on, apart from any model.

## Installation

```sh
go get github.com/unum-cloud/uform/golang@main
```

The package includes the headers of its own module's `c/` folder and links `libuform_static`, which you build from a checkout and name to cgo:

```sh
cmake --preset library && cmake --build --preset library --target uform_static
export CGO_LDFLAGS="-L$PWD/build_library/c -lturbojpeg -lpng -lwebp -lavformat -lavcodec -lavutil -lswresample -lswscale"
```

The archive carries UForm and its bundled libraries, while the system image, video and audio codecs its host decoders call stay shared libraries, which cgo links by name as Rust's `build.rs` does; the stand-in needs none of them.

The module carries no version suffix, so pin a branch or a commit, `@main` or `@<commit>`, which `go.sum` locks as a pseudo-version.
The suites run against the stand-in library, which answers every verb over the C++ standard library alone:

```sh
cmake --preset stub_library -D UFORM_WITH_CODECS=OFF && cmake --build --preset stub_library --target uform_static_stub
CGO_LDFLAGS="-L$PWD/build_stub_library/stub" go test -race ./golang/...
```

## Quickstart

Encode two captions and a photo into one space, then pick the caption closest to the photo:

```go
package main

import (
	"fmt"
	"log"
	"math"
	"os"

	uform "github.com/unum-cloud/uform/golang"
)

func main() {
	executor, err := uform.ExecutorForRam(nil, uform.CPUClassUnknown, 0) // the threads every receiver names
	if err != nil {
		log.Fatal(err)
	}
	defer executor.Close()
	textsArena, err := uform.NewArena(executor) // a call rewinds its arena, so two outputs need two
	if err != nil {
		log.Fatal(err)
	}
	defer textsArena.Close()
	imagesArena, err := uform.NewArena(executor)
	if err != nil {
		log.Fatal(err)
	}
	defer imagesArena.Close()
	model, err := uform.ModelLoadRam("model.safetensors", nil) // BFloat16 blocks, as the checkpoint stores them
	if err != nil {
		log.Fatal(err)
	}
	defer model.Close()

	captions := []string{"a cat on a mat", "a sunrise over the bay"}
	photo, err := os.ReadFile("cat.png")
	if err != nil {
		log.Fatal(err)
	}
	texts, err := model.On(executor).TextEncode(uform.TapeOf(captions), uform.EncodeOutputsEmbeddings, uform.ScalarF32, textsArena, nil)
	if err != nil {
		log.Fatal(err)
	}
	images, err := model.On(executor).ImageEncode(uform.TapeOf([][]byte{photo}), uform.EncodeOutputsEmbeddings, uform.ScalarF32, imagesArena, nil)
	if err != nil {
		log.Fatal(err)
	}

	image, err := uform.RowValues[float32](images.Embeddings, 0) // zero-copy from the arena; embeddings come unnormalized
	if err != nil {
		log.Fatal(err)
	}
	cosine := func(text []float32) float64 {
		var dot, textNorm, imageNorm float64
		for dimension := range text {
			dot += float64(text[dimension]) * float64(image[dimension])
			textNorm += float64(text[dimension]) * float64(text[dimension])
			imageNorm += float64(image[dimension]) * float64(image[dimension])
		}
		return dot / math.Sqrt(textNorm*imageNorm)
	}
	best, bestCosine := 0, math.Inf(-1)
	for row := range captions {
		text, err := uform.RowValues[float32](texts.Embeddings, uint(row))
		if err != nil {
			log.Fatal(err)
		}
		if similarity := cosine(text); similarity > bestCosine {
			best, bestCosine = row, similarity
		}
	}
	fmt.Printf("'%s' describes the photo best\n", captions[best])
}
```

`golang/example_test.go` holds the same program, so `go test` compiles it.

## Receivers & Deferred Calls

The receiver decides where a verb runs and how it completes, while the verb and its arguments stay the same:

- `model.TextEncode(..)` runs on the calling goroutine, through the `ModelOn` a `Model` embeds.
- `model.On(executor).TextEncode(..)` blocks on the executor, failing `StatusContended` while another call holds it.
- `model.Deferred(executor).TextEncode(..)` submits the call and answers a `*Deferred[Encoded]`.

`ModelOn` and `ModelDeferredOn` offer every model verb with the same arguments, as `TrainerOn` and `TrainerDeferredOn` do for a trainer.
A blocking verb answers `(T, error)` directly, so nothing about asynchrony reaches a caller who never asks for it.

```go
call, err := model.Deferred(executor).TextEncode(texts, uform.EncodeOutputsEmbeddings, uform.ScalarF32, arena, nil)
if err != nil {
	return err // refused before anything was submitted
}
if call.Status() == uform.StatusDeferred {
	// other work
}
encoded, err := call.Wait()
```

A deferred verb answers `(*Deferred[T], error)`, so a refusal the binding catches arrives at once, as a blocking one would, and nothing is submitted.
`Status` answers `StatusDeferred` until the call is final, then its final status, and `Wait` blocks the goroutine in cgo until then, answering the same result every time.
A goroutine calling `Wait` overlaps the wait, and a `select` reaches it through a channel that goroutine closes; the package starts no goroutine of its own.
`Close` waits the same way and drops the result, and a `Deferred` dropped without either is released by the collector once final, never blocking it.

The library reads a deferred call's inputs after the verb returns, so the tapes, slices, handlers and reports a deferred verb names must not change until the call is final; the binding pins every one of them.
An executor without threads of its own may run a deferred call at submission, final when the verb returns.

## Executors

An executor owns host threads or device streams, the activation scratch, and the pinned buffers staging passes through, grown to a high-water mark.
One executor serves any number of models of its residency.

```go
threads, err := uform.ExecutorForRam(nil, uform.CPUClassUnknown, 8)                  // eight threads, where zero takes 7/8 of the cores
nearby, err := uform.ExecutorForRam([]int32{1}, uform.CPUClassUnknown, 0)            // threads spawned near NUMA node 1
serial, err := uform.ExecutorForRam(nil, uform.CPUClassSerial, 0)                    // dispatches as the portable floor
streams, err := uform.ExecutorForVram([]int32{0, 1}, uform.GPUClassUnknown, nil, uform.CPUClassUnknown, nil, 0)
var local *uform.Executor                                                            // owns nothing, runs on the calling goroutine

err = threads.Drain(uform.MoodSpinning) // waits until no call holds it, then keeps spinning
```

The class arguments are ceilings: calls dispatch as the best class under them the hardware runs, and the zero class sets none.
Naming one memory domain spawns the threads near it, while none or several interleave them across the machine.
`ExecutorForVram` opens one stream per device, or adopts the caller's streams, one per named device, and `Close` destroys them; naming no devices drives every listed device of the first vendor with a known class.
`uform.GPUDevices(devices)` fills a slice of `GPUDevice` and answers how many GPUs the process can drive, and a device's position there is the ordinal `ExecutorForVram` and `ModelLoadVram` name it by.
An executor runs one call at a time, so two goroutines that genuinely overlap take one apiece.
`Close` waits until no call holds the executor, and a closed one refuses every later verb with `StatusUninitialized`.

## Arenas

Every output whose length only the call knows, the embeddings, the states and their offsets, lands in an `Arena` you pass, and the verb answers views over it without copying.
`uform.NewArena(executor)` places it where that executor computes, decided once, and `uform.NewArena(nil)` makes host pages for the calling goroutine.
Each call rewinds its arena, so the previous call's views and every slice they handed out end there, and a steady loop allocates once.
To keep a result past the next call, pass that call another arena; an arena serves one call at a time, so each deferred call needs its own.
An arena from a VRAM executor holds device memory, which `Describe` reports as `ResidencyVram`; where its `HostAccess` is `HostAccessRefused` no Go code may read it, so a verb given that arena fails with `StatusDevice` before it runs, while managed memory and Apple's shared memory read like host pages.
The Go binding has no pointer accessor or device export that could hand device outputs on unread, so CUDA and HIP device arenas are not usable from Go.
An arena never outlives the executor that made it, so `defer executor.Close()` comes before `defer arena.Close()`, and an arena from `uform.NewArena(nil)` lives until it is closed.

```go
arena, err := uform.NewArena(executor)
for _, batch := range batches {
	encoded, err := model.On(executor).TextEncode(batch, uform.EncodeOutputsEmbeddings, uform.ScalarF32, arena, nil)
	if err != nil {
		return err
	}
	consume(encoded.Embeddings) // rewinds next iteration
}
```

`Close` waits for every deferred call naming the arena, and a call naming an arena that serves another call finishes with `StatusContended`.

## Encoding

One verb per input form, each spelled as in C, and each taking its inputs, the `EncodeOutputs` to keep, the scalar every output lands as, the arena, and last a config, nil for none:

- `TextEncode(texts, ..)` takes UTF-8 texts, a byte per token, since models are tokenizer-free.
- `ImageEncode(files, ..)` takes JPEG, PNG or WebP files.
- `ImageEncodePixels(pixels, heights, widths, ..)` takes interleaved 8-bit RGB, `heights[i]` rows of `widths[i]` pixels per item.
- `VideoEncode(files, ..)` takes MP4 or WebM clips.
- `AudioEncode(files, ..)` takes WAV, FLAC, MP3 or Opus recordings.
- `AudioEncodeSamples(samples, sampleRate, ..)` takes mono `float32` samples.

A `Tape` passes the elements and `count + 1` offsets to the library without a copy: `BytesTape(data, offsets)` for texts and files, and `SamplesTape(data, offsets)` for recordings.
`TapeOf(items)` gathers separate strings or byte slices into a fresh tape with one copy, and `SamplesOf(recordings)` does so for samples:

```go
tape := uform.BytesTape([]byte("first documentthe second"), []uint{0, 14, 24})
encoded, err := model.TextEncode(tape, uform.EncodeOutputsStates, uform.ScalarBF16, arena, nil)
offsets := encoded.States.Offsets() // item `i` spans rows `offsets[i]` up to `offsets[i + 1]`
second, err := encoded.States.Item(1) // the rows of one item, as `Rows`
```

`EncodeOutputsEmbeddings`, `EncodeOutputsStates` and `EncodeOutputsTokenVectors` join with `|`, and a surface left out answers a zero `Count` and costs nothing past the final norm.
Token vectors are the token head's unit rows, one per state, which late-interaction rerankers score; a model without the head answers `StatusUnsupported`.
The config reaches the knobs:

```go
stats := new(uform.TextEncodeStats)
config := &uform.TextEncodeConfig{
	Stats:    stats,
	Progress: func(completed, total uint) bool { return completed*2 < total },
}
encoded, err := model.TextEncode(texts, uform.EncodeOutputsEmbeddings, uform.ScalarF32, arena, config)
```

`Rows` reports its `Count`, `Columns`, `Stride` and `Scalar`, and `uform.RowValues[float32](rows, i)` views one row's values, refused with `StatusInvalidArgument` when the rows hold another scalar type.
`RowsTape` adds the `Offsets` and an `Item` per input.
A handler answering false stops the call, which fails with `StatusCancelled` and publishes nothing.

## Choices

`TextChoose(texts, choices, ..)` scores each item's legal choices, one byte each, through one forward, and answers a `TapeView` of one `float32` probability per choice that sums to one per item:

```go
config := &uform.TextChooseConfig{TemperatureGroups: []uint32{0, 1}}
probabilities, err := model.TextChoose(uform.TapeOf([]string{"keep this candidate?", "expand which neighbour?"}),
	uform.TapeOf([]string{"yn", "ABCD"}), arena, config)
expand, err := uform.TapeEntry[float32](probabilities, 1) // the probabilities of 'A' through 'D'
```

Each group is a question type below the model's `ChoiceTemperatureGroups`, and a model without the byte decoder answers `StatusUnsupported`.

## Models

`ModelLoadRam(path, config)` and `ModelLoadVram(path, config)` copy a safetensors checkpoint into host memory or onto GPUs, and `ModelLoadRamBuffer` and `ModelLoadVramBuffer` take the checkpoint's bytes instead.
A nil config loads on the calling goroutine with BFloat16 blocks, and the config reaches the executor, the placement and its class ceiling, `ScalarE4M3` quantization and a progress handler:

```go
model, err := uform.ModelLoadVram("model.safetensors", &uform.ModelLoadVramConfig{
	Executor:     executor,
	Devices:      []int32{0},
	Quantization: uform.ScalarE4M3,
})
description, err := model.Describe() // geometry, placement, classes and heads
```

A description reports the `CPUClass` or `GPUClass` the model dispatched as, and the heads the checkpoint carries: a nonzero `TokenWidth` for token vectors, and a nonzero `ChoiceTemperatureGroups` for choices.
Loads always block, and `Close` waits for every deferred call naming the model.

## Training

Training is GPU-only.
The configuration arrives as structs mirroring the C ones field by field, so a field left at zero is derived by the library:

```go
executor, err := uform.ExecutorForVram([]int32{0, 1}, uform.GPUClassUnknown, nil, uform.CPUClassUnknown, nil, 0)
mesh, err := uform.MeshBootstrap(executor, &uform.MeshOptions{
	HostDomainIndex: rank,
	HostDomainCount: nodes,
	Directory:       "/shared/exchange",
	Job:             "run-17",
})
trainer, err := uform.TrainerInitVram(executor, &uform.TrainerInitVramConfig{
	Mesh:      mesh,
	Depth:     12,
	Hidden:    512,
	Objective: uform.ObjectiveJoint,
	Batch: uform.BatchShape{MaxRowsPerDevice: 64, MaxSegmentsPerDevice: 256, MaxColumnsPerRow: 1024,
		MaxBagsPerDevice: 128, Image: uform.ImageShape{PatchSide: 16}},
	Schedule: uform.Schedule{Steps: 100_000},
	Seed:     42,
})

stats := new(uform.TrainerStepStats)
relations := uform.Relations{Attract: uform.Edges{Offsets: []uint{0, 1, 2, 3, 4}, IDs: []uint32{1, 0, 3, 2}}}
config := &uform.TrainerStepConfig{Bags: []uint32{0, 0, 1, 1}, Relations: relations, Stats: stats}
_, err = trainer.On(executor).Step(modalities, contents, config)
err = trainer.On(executor).Save("checkpoint.safetensors", nil)
```

`BatchShape` nests an `ImageShape`, a `VideoShape` and an `AudioShape`, while `Masking`, `Contrastive`, `LateInteraction`, `Choices` and `Schedule` configure the objectives, and `PairLossForm` picks each contrastive term's pair loss.
`Relations` holds three `Edges` sets as compressed sparse rows, whose offsets and weights the binding checks against the step's items before the call.
A step's `Sides` give each item a `LateInteractionSide`, and its `Choices` and `ChoiceTargets` give each choice item its legal bytes and their target probabilities as tapes, the shapes `TextChoose` takes and answers.
`Step` answers the pooled embeddings the loss scored when its config names an `Arena`, and `Rows` of zero `Count` otherwise.
A trainer refers to its mesh, so the mesh outlives every trainer made over it, and `mesh.Describe()` answers a `MeshDescription` whose `MultimemTier` and `GPUClass` describe the run.

## Errors

Every fallible call answers an `error` whose `*Error` carries a `Status` and the library's static `Message`, and a bare `Status` is an error too, so `errors.Is` matches either.
A failure the binding catches before the call crosses the boundary reports the status the library would have used.

```go
encoded, err := model.On(executor).TextEncode(texts, uform.EncodeOutputsEmbeddings, uform.ScalarF32, arena, nil)
switch {
case errors.Is(err, uform.StatusContended):
	fmt.Println("the arena serves another call")
case err != nil:
	fmt.Println(err)
default:
	fmt.Println(encoded.Embeddings.Count(), "items")
}
```

`Status` carries the C enumerators' values, and `StatusDeferred` reaches only `Deferred.Status`.
`Status`, `Scalar`, `Mood`, `CPUClass`, `GPUClass`, `GPUVendor`, `Modality`, `Objective`, `Negatives`, `MultimemTier`, `PairLossForm` and `LateInteractionSide` print the names the C tables own through `String`, such as `"shape-mismatch"` or `"intel-haswell"`.
A panic inside a progress handler crashes the process, since it cannot unwind through the library.

## Design

- __A thin layer over the C ABI.__
  One file per header, `types.go`, `model.go`, `encode.go` and `train.go`; every name is its C name without `uform_` and `_t` in PascalCase, a verb's argument struct becomes its `*Config`, and every wrapper's documentation names the C symbol it calls or says it is binding-only.
- __The receiver chooses, the verb stays.__
  The model runs a verb on the calling goroutine, `On(executor)` blocks on an executor, and `Deferred(executor)` submits it, all with the same arguments.
- __Deferred calls use the language's own idioms.__
  `Status` polls through `uform_outcome_status` and `Wait` blocks through `uform_outcome_wait`, and refusals answer before anything is submitted.
- __No life of its own.__
  The binding starts no goroutines and keeps no queues, locks or counts; releasing a `Deferred` rides `Close`, which waits for the call first, or the collector's cleanup, which frees it only once final.
- __Arenas belong to the caller and are reused, never shared.__
  Every output whose length scales with the call lands in an arena the caller passes, and its views end at the arena's next call.
- __One pinned frame per call.__
  A call's outcome, argument struct, callback handles and result share one allocation holding no unpinned Go pointer, since cgo pins only heap objects.
- __Suites run against the stub library__, and `TestCallsAllocateOnlyTheirFrames` pins the budgets below.

| Path | Go | Rust |
|---|---|---|
| Blocking call | 1 frame | 0 |
| Refusal before the call | 1 `Error` | 1 error |
| Deferred submit and wait | 4: frame, `Deferred`, 2 in `AddCleanup` | 0 |
| Each callback | 1 handle | 0 |
