// Shared vocabulary of the UForm binding and the calls every verb crosses through.
//
// File: golang/types.go
// Author: Ash Vardanian
// Date: November 18, 2026

package uform

import (
	"fmt"
	"runtime"
	"runtime/cgo"
	"unsafe"
)

/*
#cgo CFLAGS: -I${SRCDIR}/../c
#cgo LDFLAGS: -luform_static
#cgo linux LDFLAGS: -lstdc++ -lm
#cgo darwin LDFLAGS: -lc++
#cgo windows LDFLAGS: -lstdc++

// A function is noescape when C keeps none of its Go pointers past the call, and nocallback when
// its argument structure has no callback field: an executor runs only the callbacks a call names.
#cgo noescape uform_outcome_status
#cgo nocallback uform_outcome_status
#cgo noescape uform_outcome_wait
#cgo nocallback uform_outcome_wait
#cgo noescape uform_executor_for_ram
#cgo nocallback uform_executor_for_ram
#cgo noescape uform_executor_for_vram
#cgo nocallback uform_executor_for_vram
#cgo noescape uform_executor_drain
#cgo nocallback uform_executor_drain
#cgo noescape uform_arena_init
#cgo nocallback uform_arena_init
#cgo noescape uform_arena_describe
#cgo nocallback uform_arena_describe
#cgo noescape uform_gpu_devices
#cgo nocallback uform_gpu_devices
#cgo noescape uform_status_from_name
#cgo nocallback uform_status_from_name
#cgo noescape uform_scalar_from_name
#cgo nocallback uform_scalar_from_name
#cgo noescape uform_mood_from_name
#cgo nocallback uform_mood_from_name
#cgo noescape uform_cpu_class_from_name
#cgo nocallback uform_cpu_class_from_name
#cgo noescape uform_gpu_class_from_name
#cgo nocallback uform_gpu_class_from_name
#cgo noescape uform_gpu_vendor_from_name
#cgo nocallback uform_gpu_vendor_from_name

#include "shim.h"
*/
import "C"

// region: Errors

// Status names what went wrong, numbered as the C header and USearch number it, so one failure
// carries one identity across both libraries. Validation performed here before a call crosses
// reports the status the library would have reported for the same fault.
//
// A Status is an error, so errors.Is(err, uform.StatusUnsupported) matches any failure carrying it.
type Status int32

// The statuses a call ends in. Only Deferred.Status reports StatusDeferred, the one status
// that is not final.
const (
	// StatusSuccess means the call completed and carries no message.
	StatusSuccess Status = C.uform_status_success_k
	// StatusDeferred means the call was submitted deferred and is not final yet.
	StatusDeferred Status = C.uform_status_deferred_k
	// StatusOutOfMemory means device, pinned or host memory could not be allocated.
	StatusOutOfMemory Status = C.uform_status_out_of_memory_k
	// StatusShapeMismatch means item counts, a pixel shape or a batch envelope disagree.
	StatusShapeMismatch Status = C.uform_status_shape_mismatch_k
	// StatusUnsupportedLayout means a stride or packing the model cannot read.
	StatusUnsupportedLayout Status = C.uform_status_unsupported_layout_k
	// StatusUnknownKind means a scalar, modality or name this build does not know.
	StatusUnknownKind Status = C.uform_status_unknown_kind_k
	// StatusIO means a checkpoint file could not be opened, read or written.
	StatusIO Status = C.uform_status_io_k
	// StatusDevice means the driver refused or the device faulted, or the build carries no GPU.
	StatusDevice Status = C.uform_status_device_k
	// StatusMalformed means supplied bytes could not be decoded: a checkpoint, an image, a clip or a
	// recording.
	StatusMalformed Status = C.uform_status_malformed_k
	// StatusInternal means the library reported neither success nor a failure.
	StatusInternal Status = C.uform_status_internal_k
	// StatusOverflow means a count would pass the largest value its type holds.
	StatusOverflow Status = C.uform_status_overflow_k
	// StatusUninitialized means the handle the call operates on is closed or was never opened.
	StatusUninitialized Status = C.uform_status_uninitialized_k
	// StatusConfigMismatch means the request contradicts a frozen configuration.
	StatusConfigMismatch Status = C.uform_status_config_mismatch_k
	// StatusContended means the call was refused rather than waited for: another call holds its
	// executor or arena.
	StatusContended Status = C.uform_status_contended_k
	// StatusCancelled means a progress handler asked the call to stop; it unwound and published
	// nothing.
	StatusCancelled Status = C.uform_status_cancelled_k
	// StatusUnsupported means the verb is declared but this residency or model does not implement it,
	// like a modality the model was not trained on.
	StatusUnsupported Status = C.uform_status_unsupported_k
	// StatusInvalidArgument means an argument breaks the verb's contract, like a required value left
	// nil.
	StatusInvalidArgument Status = C.uform_status_invalid_argument_k
)

// String renders the status as the library spells it.
func (status Status) String() string {
	return C.GoString(C.uform_status_name(C.uform_status_t(status)))
}

// StatusFromName reads a status back from the spelling [Status.String] renders.
func StatusFromName(name string) (Status, error) {
	var status C.uform_status_t
	err := named(C.uform_status_from_name(view(name), &status))
	return Status(status), err
}

// Error implements the error interface, so a bare Status is an errors.Is target.
func (status Status) Error() string { return status.String() }

// Error is one failed call: a status every caller can branch on, and the detail the status
// cannot carry. Reach the status with errors.Is or errors.As.
type Error struct {
	Status  Status
	Message string
}

// Error implements the error interface.
func (failure *Error) Error() string {
	return fmt.Sprintf("uform: %s: %s", failure.Status, failure.Message)
}

// Unwrap exposes the status, which is what errors.Is compares.
func (failure *Error) Unwrap() error { return failure.Status }

// fail builds the failure this binding reports for a fault it caught before the boundary.
func fail(status Status, message string) *Error {
	return &Error{Status: status, Message: message}
}

// check turns a final outcome into a Go error. Messages are static library storage and need no
// deallocation, so the copy is the only cost.
func check(outcome C.uform_outcome_t) error {
	if outcome.status == C.uform_status_success_k {
		return nil
	}
	return &Error{Status: Status(outcome.status), Message: C.GoString(outcome.message)}
}

// named turns the status a spelling lookup answers into a Go error.
func named(status C.uform_status_t) error {
	if status == C.uform_status_success_k {
		return nil
	}
	return fail(Status(status), "the spelling names no value of its enum")
}

// endregion: Errors

// region: Scalars

// Scalar names a computed or exchanged scalar type by width and encoding, numbered as USearch
// numbers it, so a run of embeddings is a run USearch indexes without conversion. Sub-byte kinds
// pack several dimensions per byte, as their xN suffixes spell.
type Scalar int32

// The scalar types rows and tapes travel in, ordered by shrinking dynamic range. The numbering
// leaves gaps for kinds that do not exist yet.
const (
	// ScalarUnknown names no scalar: an unset field, which the library refuses.
	ScalarUnknown Scalar = C.uform_scalar_unknown_k
	// ScalarF64 is 64-bit double precision.
	ScalarF64 Scalar = C.uform_scalar_f64_k
	// ScalarF32 is 32-bit single precision.
	ScalarF32 Scalar = C.uform_scalar_f32_k
	// ScalarBF16 is the brain floating-point format.
	ScalarBF16 Scalar = C.uform_scalar_bf16_k
	// ScalarF16 is IEEE half precision.
	ScalarF16 Scalar = C.uform_scalar_f16_k
	// ScalarE5M2 is 8-bit floating point with five exponent bits.
	ScalarE5M2 Scalar = C.uform_scalar_e5m2_k
	// ScalarE4M3 is 8-bit floating point with four exponent bits.
	ScalarE4M3 Scalar = C.uform_scalar_e4m3_k
	// ScalarE3M2 is 6-bit floating point with three exponent bits.
	ScalarE3M2 Scalar = C.uform_scalar_e3m2_k
	// ScalarE2M3 is 6-bit floating point with two exponent bits.
	ScalarE2M3 Scalar = C.uform_scalar_e2m3_k
	// ScalarI8 is an 8-bit signed integer.
	ScalarI8 Scalar = C.uform_scalar_i8_k
	// ScalarI4x2 packs two 4-bit signed integers per byte.
	ScalarI4x2 Scalar = C.uform_scalar_i4x2_k
	// ScalarI32 is a 32-bit signed integer.
	ScalarI32 Scalar = C.uform_scalar_i32_k
	// ScalarU8 is an 8-bit unsigned integer, the element of UTF-8 text and of encoded files.
	ScalarU8 Scalar = C.uform_scalar_u8_k
	// ScalarU4x2 packs two 4-bit unsigned integers per byte.
	ScalarU4x2 Scalar = C.uform_scalar_u4x2_k
	// ScalarU1x8 packs eight single-bit dimensions per byte, the sign of each embedding dimension.
	ScalarU1x8 Scalar = C.uform_scalar_u1x8_k
)

// String renders the scalar kind as the library spells it.
func (scalar Scalar) String() string {
	return C.GoString(C.uform_scalar_name(C.uform_scalar_t(scalar)))
}

// ScalarFromName reads a scalar back from the spelling [Scalar.String] renders.
func ScalarFromName(name string) (Scalar, error) {
	var scalar C.uform_scalar_t
	err := named(C.uform_scalar_from_name(view(name), &scalar))
	return Scalar(scalar), err
}

// dimensionsPerValue is how many logical dimensions one stored value carries: one for the
// byte-and-wider kinds, more for the packed ones.
func (scalar Scalar) dimensionsPerValue() uint {
	switch scalar {
	case ScalarI4x2, ScalarU4x2:
		return 2
	case ScalarU1x8:
		return 8
	default:
		return 1
	}
}

// Storage types for the encodings Go has no builtin for. Each is a distinct defined type so
// RowValues can tell them apart, and each holds the raw bits rather than a converted value.
type (
	// BFloat16 holds the sixteen bits of a brain floating-point value.
	BFloat16 uint16
	// Float16 holds the sixteen bits of an IEEE half-precision value.
	Float16 uint16
	// E5M2Scalar holds the eight bits of a five-exponent minifloat.
	E5M2Scalar uint8
	// E4M3Scalar holds the eight bits of a four-exponent minifloat.
	E4M3Scalar uint8
	// E3M2Scalar holds the six occupied bits of a three-exponent minifloat.
	E3M2Scalar uint8
	// E2M3Scalar holds the six occupied bits of a two-exponent minifloat.
	E2M3Scalar uint8
	// I4x2Scalar holds two 4-bit signed integers.
	I4x2Scalar int8
	// U4x2Scalar holds two 4-bit unsigned integers.
	U4x2Scalar uint8
	// U1x8Scalar holds eight single-bit dimensions.
	U1x8Scalar uint8
)

// Element constrains the element types rows and tapes may hold, ordered by shrinking dynamic range.
type Element interface {
	float64 | float32 | BFloat16 | Float16 |
		E5M2Scalar | E4M3Scalar | E3M2Scalar | E2M3Scalar |
		int8 | I4x2Scalar | int32 | uint8 | U4x2Scalar | U1x8Scalar
}

// scalarOf names the scalar kind of a constrained element type.
func scalarOf[Value Element]() Scalar {
	var probe Value
	switch any(probe).(type) {
	case float64:
		return ScalarF64
	case float32:
		return ScalarF32
	case BFloat16:
		return ScalarBF16
	case Float16:
		return ScalarF16
	case E5M2Scalar:
		return ScalarE5M2
	case E4M3Scalar:
		return ScalarE4M3
	case E3M2Scalar:
		return ScalarE3M2
	case E2M3Scalar:
		return ScalarE2M3
	case int8:
		return ScalarI8
	case I4x2Scalar:
		return ScalarI4x2
	case int32:
		return ScalarI32
	case uint8:
		return ScalarU8
	case U4x2Scalar:
		return ScalarU4x2
	case U1x8Scalar:
		return ScalarU1x8
	default:
		return ScalarUnknown
	}
}

// endregion: Scalars

// region: Rows

// Rows is a run of rows a call wrote packed into its arena, one per item, readable until the arena
// serves another call. A call into an arena the host may not read is refused before it runs.
//
// Mirrors uform_rows_view_t, beside the count the call was given.
type Rows struct {
	raw   C.uform_rows_view_t
	count uint
}

// Count reports how many rows the call wrote, zero for a surface it did not keep.
func (rows Rows) Count() uint { return rows.count }

// Columns reports the values per row, the model's dimensions or token width.
func (rows Rows) Columns() uint { return uint(rows.raw.columns) }

// Stride reports the bytes between the starts of consecutive rows.
func (rows Rows) Stride() uint { return uint(rows.raw.stride) }

// Scalar reports the scalar type the rows were written as.
func (rows Rows) Scalar() Scalar { return Scalar(rows.raw.scalar) }

// RowValues views one row's values in the arena, refused with StatusInvalidArgument when Value
// is not the scalar type they were written as or row is not below Count.
func RowValues[Value Element](rows Rows, row uint) ([]Value, error) {
	if scalar := scalarOf[Value](); scalar != rows.Scalar() {
		return nil, fail(StatusInvalidArgument, "rows written as "+rows.Scalar().String()+" cannot be read as "+scalar.String())
	}
	if row >= rows.count {
		return nil, fail(StatusInvalidArgument, "the row is past the rows the call wrote")
	}
	first := unsafe.Add(rows.raw.data, row*rows.Stride())
	return unsafe.Slice((*Value)(first), rows.Columns()/rows.Scalar().dimensionsPerValue()), nil
}

// RowsTape is a ragged run of rows a call wrote into its arena, item i spanning the rows between
// two consecutive [RowsTape.Offsets], readable until the arena serves another call.
//
// Mirrors uform_rows_tape_view_t, beside the count the call was given.
type RowsTape struct {
	raw   C.uform_rows_tape_view_t
	count uint
}

// Count reports how many items the call wrote rows for, zero for a surface it did not keep.
func (tape RowsTape) Count() uint { return tape.count }

// Offsets views the Count + 1 offsets into the rows, in rows.
func (tape RowsTape) Offsets() []uint {
	if tape.raw.offsets == nil {
		return nil
	}
	return unsafe.Slice((*uint)(unsafe.Pointer(tape.raw.offsets)), tape.count+1)
}

// Item views the rows of one item, refused with [StatusInvalidArgument] when item is not below
// [RowsTape.Count].
func (tape RowsTape) Item(item uint) (Rows, error) {
	if item >= tape.count {
		return Rows{}, fail(StatusInvalidArgument, "the item is past the items the call wrote")
	}
	offsets := tape.Offsets()
	rows := tape.raw.rows
	rows.data = unsafe.Add(rows.data, offsets[item]*uint(rows.stride))
	return Rows{raw: rows, count: offsets[item+1] - offsets[item]}, nil
}

// endregion: Rows

// region: Tapes

// Tape views a ragged batch as Apache Arrow lays one out: elements end to end, and one more
// offset than entries around them, entry i spanning offsets[i] up to offsets[i+1]. An element is
// a byte of UTF-8 text or of an encoded file, or a sample of a recording.
//
// A Tape views and never owns, and the binding pins both slices until each call it crosses is
// final, so neither may change until then.
type Tape struct {
	pointer unsafe.Pointer
	offsets []uint
	scalar  Scalar
}

// TapeOf gathers separate texts or files end to end into a fresh Tape, one entry apiece, copying
// each once.
func TapeOf[Item ~string | ~[]byte](items []Item) Tape {
	offsets := make([]uint, len(items)+1)
	for position, item := range items {
		offsets[position+1] = offsets[position] + uint(len(item))
	}
	data := make([]byte, 0, offsets[len(items)])
	for _, item := range items {
		data = append(data, item...)
	}
	return tapeOf(data, offsets)
}

// BytesTape views bytes end to end around len(offsets) - 1 texts or files, without a copy.
//
// Panics when the offsets overrun the bytes, which names a caller's arithmetic mistake rather
// than a condition the library could report.
func BytesTape(data []byte, offsets []uint) Tape { return tapeOf(data, offsets) }

// SamplesOf gathers separate mono recordings end to end into a fresh Tape, one entry apiece.
func SamplesOf(recordings [][]float32) Tape {
	offsets := make([]uint, len(recordings)+1)
	for position, recording := range recordings {
		offsets[position+1] = offsets[position] + uint(len(recording))
	}
	data := make([]float32, 0, offsets[len(recordings)])
	for _, recording := range recordings {
		data = append(data, recording...)
	}
	return tapeOf(data, offsets)
}

// SamplesTape views mono samples end to end around len(offsets) - 1 recordings, without a copy.
//
// Panics when the offsets overrun the samples.
func SamplesTape(data []float32, offsets []uint) Tape { return tapeOf(data, offsets) }

// tapeOf views data around offsets.
func tapeOf[Value Element](data []Value, offsets []uint) Tape {
	if len(offsets) == 0 {
		panic("uform: a tape needs one more offset than entries")
	}
	if last := offsets[len(offsets)-1]; last > uint(len(data)) {
		panic(fmt.Sprintf("uform: offsets ending at %d overrun %d elements", last, len(data)))
	}
	return Tape{pointer: unsafe.Pointer(unsafe.SliceData(data)), offsets: offsets, scalar: scalarOf[Value]()}
}

// Entries reports how many entries the tape spans, zero for the zero Tape.
func (tape Tape) Entries() uint {
	if len(tape.offsets) == 0 {
		return 0
	}
	return uint(len(tape.offsets)) - 1
}

// Scalar reports the scalar type the elements hold.
func (tape Tape) Scalar() Scalar { return tape.scalar }

// lower pins the tape as the batch view a verb reads; the zero Tape passes no offsets, which a
// verb's optional tapes read as none.
func (tape Tape) lower(call *call) C.uform_tape_view_t {
	if tape.pointer != nil {
		call.pinner.Pin(tape.pointer)
	}
	return C.uform_tape_view_t{
		data:    tape.pointer,
		offsets: (*C.size_t)(call.pinSlice(unsafe.Pointer(unsafe.SliceData(tape.offsets)), len(tape.offsets))),
		scalar:  C.uform_scalar_t(tape.scalar),
	}
}

// TapeView is a ragged batch a call wrote into its arena, readable until the arena serves another
// call: one probability per choice, item by item, for TextChoose.
//
// Mirrors uform_tape_view_t, beside the count the call was given.
type TapeView struct {
	raw   C.uform_tape_view_t
	count uint
}

// Entries reports how many entries the call wrote.
func (tape TapeView) Entries() uint { return tape.count }

// Scalar reports the scalar type the elements were written as.
func (tape TapeView) Scalar() Scalar { return Scalar(tape.raw.scalar) }

// TapeEntry views one entry's elements in the arena, refused with StatusInvalidArgument when
// Value is not the scalar type they were written as or entry is not below Entries.
func TapeEntry[Value Element](tape TapeView, entry uint) ([]Value, error) {
	if scalar := scalarOf[Value](); scalar != tape.Scalar() {
		return nil, fail(StatusInvalidArgument, "elements written as "+tape.Scalar().String()+" cannot be read as "+scalar.String())
	}
	if entry >= tape.count {
		return nil, fail(StatusInvalidArgument, "the entry is past the entries the call wrote")
	}
	offsets := unsafe.Slice((*uint)(unsafe.Pointer(tape.raw.offsets)), tape.count+1)
	begin, end := offsets[entry], offsets[entry+1]
	if begin == end {
		return nil, nil
	}
	var element Value
	first := unsafe.Add(tape.raw.data, begin*uint(unsafe.Sizeof(element)))
	return unsafe.Slice((*Value)(first), end-begin), nil
}

// endregion: Tapes

// region: Callbacks

// ProgressCallback is called with the units a long call finished so far and the units it finishes
// unless stopped; false stops it, and the call fails with [StatusCancelled]. It never runs on two
// threads at once, and it must not panic.
type ProgressCallback func(completed, total uint) bool

//export uformContinues
func uformContinues(completed, total C.size_t, state unsafe.Pointer) C.uform_continuation_t {
	if (*cgo.Handle)(state).Value().(ProgressCallback)(uint(completed), uint(total)) {
		return C.uform_continue_k
	}
	return C.uform_cancel_k
}

// progress lowers an optional handler and its report count as the struct a reporting verb takes.
func (call *call) progress(progress ProgressCallback, steps uint) C.uform_progress_t {
	if progress == nil {
		return C.uform_progress_t{}
	}
	return C.uform_progress_t{callback: C.uform_go_continues(), state: call.callback(1, progress), steps: C.size_t(steps)}
}

// callback passes a handler to its C trampoline as an opaque handle in the call's slot rather than
// a Go pointer, since the closure holds Go pointers the cgo rules forbid handing to C.
func (call *call) callback(slot int, handler any) unsafe.Pointer {
	call.slots.handles[slot] = cgo.NewHandle(handler)
	return unsafe.Pointer(&call.slots.handles[slot])
}

// endregion: Callbacks

// region: Executors

// Executor owns the host threads or device streams a call runs on, the activation scratch, and the
// pinned buffers staging passes through, apart from any model.
//
// One executor serves any number of models of its residency, while a model owns only its weights,
// which is what lets one pool serve several models and keeps the cost of spawning visible at the
// line that spawns it. Its kind is fixed by the constructor that produced it.
//
// A nil *Executor is the null executor: it owns nothing, and every call handed it runs on the
// calling goroutine. One call holds an executor at a time; a second call finding it held finishes
// StatusContended, and nothing queues.
type Executor struct {
	handle C.uform_executor_t
}

// Residency is where an arena's outputs live, as the executor that made it decided.
type Residency int32

const (
	// ResidencyUnknown names no residency.
	ResidencyUnknown Residency = C.uform_residency_unknown_k
	// ResidencyRam is host pages, which Go code reads directly.
	ResidencyRam Residency = C.uform_residency_ram_k
	// ResidencyVram is one GPU's memory, which Go code reads only where HostAccess lets it.
	ResidencyVram Residency = C.uform_residency_vram_k
)

// HostAccess is whether Go code may read an arena's outputs.
type HostAccess int32

const (
	// HostAccessRefused is device memory the host must not read.
	HostAccessRefused HostAccess = C.uform_host_access_refused_k
	// HostAccessReadable is host, pinned and managed memory, and Apple's shared memory.
	HostAccessReadable HostAccess = C.uform_host_access_readable_k
)

// String renders the access as the library spells it, "refused" or "readable".
func (access HostAccess) String() string {
	return C.GoString(C.uform_host_access_name(C.uform_host_access_t(access)))
}

// Mood is what an executor's threads do between a drain and its next call.
type Mood int32

const (
	// MoodParked sleeps in the kernel until the next call, freeing the cores.
	MoodParked Mood = C.uform_mood_parked_k
	// MoodSpinning keeps polling, trading the cores for the lowest wake-up latency.
	MoodSpinning Mood = C.uform_mood_spinning_k
)

// String renders the mood as the library spells it.
func (mood Mood) String() string {
	return C.GoString(C.uform_mood_name(C.uform_mood_t(mood)))
}

// MoodFromName reads a mood back from the spelling [Mood.String] renders.
func MoodFromName(name string) (Mood, error) {
	var mood C.uform_mood_t
	err := named(C.uform_mood_from_name(view(name), &mood))
	return Mood(mood), err
}

// CPUClass is a CPU family UForm ships tuned variants for, numbered as USearch numbers it; a RAM
// placement or executor may cap dispatch at one, and the zero value sets no ceiling.
type CPUClass int32

const (
	// CPUClassUnknown sets no ceiling, so dispatch may pick any class the machine runs.
	CPUClassUnknown CPUClass = C.uform_cpu_class_unknown_k
	// CPUClassSerial is the portable floor, with no SIMD assumptions.
	CPUClassSerial CPUClass = C.uform_cpu_class_serial_k
	// CPUClassMonoversioned is the one class of a monoversioned build.
	CPUClassMonoversioned CPUClass = C.uform_cpu_class_monoversioned_k
	// CPUClassIntelHaswell is Haswell-class x86: AVX2 and FMA.
	CPUClassIntelHaswell CPUClass = C.uform_cpu_class_intel_haswell_k
	// CPUClassIntelSierra is Sierra Forest-class x86: AVX-VNNI without AVX-512.
	CPUClassIntelSierra CPUClass = C.uform_cpu_class_intel_sierra_k
	// CPUClassIntelSkylake is Skylake-class x86: the AVX-512 foundation.
	CPUClassIntelSkylake CPUClass = C.uform_cpu_class_intel_skylake_k
	// CPUClassIntelIcelake is Icelake-class x86: AVX-512 VNNI and VPOPCNT.
	CPUClassIntelIcelake CPUClass = C.uform_cpu_class_intel_icelake_k
	// CPUClassAMDGenoa is Genoa-class x86: native bf16 dot products over AVX-512.
	CPUClassAMDGenoa CPUClass = C.uform_cpu_class_amd_genoa_k
	// CPUClassIntelSapphire is Sapphire Rapids-class x86: AVX-512 FP16 and the AMX tiles.
	CPUClassIntelSapphire CPUClass = C.uform_cpu_class_intel_sapphire_k
	// CPUClassIntelGranite is Granite Rapids-class x86: the AMX-FP16 tiles.
	CPUClassIntelGranite CPUClass = C.uform_cpu_class_intel_granite_k
	// CPUClassIntelDiamond is Diamond Rapids-class x86: AVX10.2's FP8 conversions.
	CPUClassIntelDiamond CPUClass = C.uform_cpu_class_intel_diamond_k
	// CPUClassAWSG2 is Graviton2-class Arm: the bare NEON floor.
	CPUClassAWSG2 CPUClass = C.uform_cpu_class_aws_g2_k
	// CPUClassAppleM2 is Armv8.6 NEON with dot products and bf16.
	CPUClassAppleM2 CPUClass = C.uform_cpu_class_apple_m2_k
	// CPUClassAppleM4 is Apple M4-class Arm: the SME outer-product tiles.
	CPUClassAppleM4 CPUClass = C.uform_cpu_class_apple_m4_k
	// CPUClassAWSG5 is Graviton5-class Arm: Armv9.2.
	CPUClassAWSG5 CPUClass = C.uform_cpu_class_aws_g5_k
	// CPUClassNVIDIAVera is Vera-class Arm: FP8 dot products.
	CPUClassNVIDIAVera CPUClass = C.uform_cpu_class_nvidia_vera_k
	// CPUClassAlibabaC930 is Alibaba C930-class RISC-V: RVA23 vectors.
	CPUClassAlibabaC930 CPUClass = C.uform_cpu_class_alibaba_c930_k
	// CPUClassIBMPower10 is IBM POWER10-class Power: VSX vectors and MMA accumulators.
	CPUClassIBMPower10 CPUClass = C.uform_cpu_class_ibm_power10_k
	// CPUClassWasm32V128 is WebAssembly with fixed-width simd128.
	CPUClassWasm32V128 CPUClass = C.uform_cpu_class_wasm32_v128_k
	// CPUClassWasm64Extended is WebAssembly with relaxed SIMD, threads and 64-bit addressing.
	CPUClassWasm64Extended CPUClass = C.uform_cpu_class_wasm64_extended_k
)

// String renders the class as the library spells it.
func (class CPUClass) String() string {
	return C.GoString(C.uform_cpu_class_name(C.uform_cpu_class_t(class)))
}

// CPUClassFromName reads a CPU class back from the spelling [CPUClass.String] renders.
func CPUClassFromName(name string) (CPUClass, error) {
	var class C.uform_cpu_class_t
	err := named(C.uform_cpu_class_from_name(view(name), &class))
	return CPUClass(class), err
}

// GPUClass is a GPU family UForm ships tuned variants for, numbered as USearch numbers it; a VRAM
// placement or executor may cap dispatch at one, and the zero value sets no ceiling.
type GPUClass int32

const (
	// GPUClassUnknown sets no ceiling, so dispatch may pick any class every named device loads.
	GPUClassUnknown GPUClass = C.uform_gpu_class_unknown_k
	// GPUClassNVIDIAAmpere is Ampere-class NVIDIA, and every newer GPU whose driver compiles its PTX.
	GPUClassNVIDIAAmpere GPUClass = C.uform_gpu_class_nvidia_ampere_k
	// GPUClassNVIDIAAda is Ada-class NVIDIA: the fp8 conversions.
	GPUClassNVIDIAAda GPUClass = C.uform_gpu_class_nvidia_ada_k
	// GPUClassNVIDIAHopper is Hopper-class NVIDIA.
	GPUClassNVIDIAHopper GPUClass = C.uform_gpu_class_nvidia_hopper_k
	// GPUClassNVIDIABlackwell is Blackwell-class NVIDIA.
	GPUClassNVIDIABlackwell GPUClass = C.uform_gpu_class_nvidia_blackwell_k
	// GPUClassNVIDIABlackwellRTX is Consumer Blackwell NVIDIA.
	GPUClassNVIDIABlackwellRTX GPUClass = C.uform_gpu_class_nvidia_blackwellrtx_k
	// GPUClassNVIDIARubin is Rubin-class NVIDIA.
	GPUClassNVIDIARubin GPUClass = C.uform_gpu_class_nvidia_rubin_k
	// GPUClassAMDCDNA3 is CDNA3 AMD: gfx942.
	GPUClassAMDCDNA3 GPUClass = C.uform_gpu_class_amd_cdna3_k
	// GPUClassAMDCDNA4 is CDNA4 AMD: gfx950.
	GPUClassAMDCDNA4 GPUClass = C.uform_gpu_class_amd_cdna4_k
	// GPUClassAMDCDNA5 is CDNA5 AMD: gfx1250.
	GPUClassAMDCDNA5 GPUClass = C.uform_gpu_class_amd_cdna5_k
	// GPUClassAppleMetal7 is Apple7-family GPUs, and every Apple GPU without a class of its own.
	GPUClassAppleMetal7 GPUClass = C.uform_gpu_class_apple_metal7_k
	// GPUClassAppleMetal9 is Apple9-family GPUs.
	GPUClassAppleMetal9 GPUClass = C.uform_gpu_class_apple_metal9_k
	// GPUClassAppleMetal10 is Apple10-family GPUs.
	GPUClassAppleMetal10 GPUClass = C.uform_gpu_class_apple_metal10_k
)

// String renders the class as the library spells it.
func (class GPUClass) String() string {
	return C.GoString(C.uform_gpu_class_name(C.uform_gpu_class_t(class)))
}

// GPUClassFromName reads a GPU class back from the spelling [GPUClass.String] renders.
func GPUClassFromName(name string) (GPUClass, error) {
	var class C.uform_gpu_class_t
	err := named(C.uform_gpu_class_from_name(view(name), &class))
	return GPUClass(class), err
}

// GPUVendor is the runtime that drives a GPU.
type GPUVendor int32

const (
	// GPUVendorUnknown names no vendor.
	GPUVendorUnknown GPUVendor = C.uform_gpu_vendor_unknown_k
	// GPUVendorNVIDIA drives the device through the CUDA driver.
	GPUVendorNVIDIA GPUVendor = C.uform_gpu_vendor_nvidia_k
	// GPUVendorAMD drives the device through the HIP runtime.
	GPUVendorAMD GPUVendor = C.uform_gpu_vendor_amd_k
	// GPUVendorApple drives the device through Metal.
	GPUVendorApple GPUVendor = C.uform_gpu_vendor_apple_k
)

// String renders the vendor as the library spells it.
func (vendor GPUVendor) String() string {
	return C.GoString(C.uform_gpu_vendor_name(C.uform_gpu_vendor_t(vendor)))
}

// GPUVendorFromName reads a GPU vendor back from the spelling [GPUVendor.String] renders.
func GPUVendorFromName(name string) (GPUVendor, error) {
	var vendor C.uform_gpu_vendor_t
	err := named(C.uform_gpu_vendor_from_name(view(name), &vendor))
	return GPUVendor(vendor), err
}

// GPUDevice is one GPU as UForm numbers it: its position in what GPUDevices fills is the ordinal
// every VRAM call names.
//
// Mirrors uform_gpu_device_t field for field, so the library writes it in place.
type GPUDevice struct {
	// Vendor is the runtime that drives the device.
	Vendor GPUVendor
	// VendorOrdinal is the device's ordinal in its vendor's runtime.
	VendorOrdinal uint32
	// GPUClass is the newest class whose code runs on it, or GPUClassUnknown where none does.
	GPUClass GPUClass
	// ComputeUnits counts streaming multiprocessors or compute units.
	ComputeUnits uint32
	// MemoryBytes is the device memory.
	MemoryBytes uint64
	name        [64]byte
}

// Name reports the vendor's own name for the device.
func (device GPUDevice) Name() string {
	length := 0
	for length < len(device.name) && device.name[length] != 0 {
		length++
	}
	return string(device.name[:length])
}

// Every struct the library writes in place matches its C twin byte for byte.
var (
	_ = [1]byte{}[unsafe.Sizeof(GPUDevice{})-C.sizeof_uform_gpu_device_t]
	_ = [1]byte{}[unsafe.Sizeof(ArenaDescription{})-C.sizeof_uform_arena_description_t]
)

// GPUDevices fills devices with the GPUs this process can drive, NVIDIA first, then AMD, then
// Apple, and reports how many there are, which may exceed len(devices).
func GPUDevices(devices []GPUDevice) (uint, error) {
	var call callWithResult[uint]
	arguments := open[C.uform_gpu_devices_t](&call)
	*arguments = C.uform_gpu_devices_t{
		outcome:          call.outcome(),
		devices:          (*C.uform_gpu_device_t)(call.pinSlice(unsafe.Pointer(unsafe.SliceData(devices)), len(devices))),
		devices_capacity: C.size_t(len(devices)),
		devices_count:    (*C.size_t)(unsafe.Pointer(call.result)),
	}
	C.uform_gpu_devices(arguments)
	return block(&call, nil)
}

// ExecutorForRam spawns host threads for RAM-resident models near the one memory domain
// memoryDomains names, or interleaved across the machine for none or several; the list is the one
// the models were loaded with, and an empty list derives it from the machine.
//
// Zero threads takes seven eighths of the cores those domains carry. A cpuClassCeiling intersects
// with what the machine runs, so a class the machine lacks narrows rather than refuses.
func ExecutorForRam(memoryDomains []int32, cpuClassCeiling CPUClass, threads uint) (*Executor, error) {
	var call callWithResult[C.uform_executor_t]
	arguments := open[C.uform_executor_for_ram_t](&call)
	*arguments = C.uform_executor_for_ram_t{
		executor: call.result,
		outcome:  call.outcome(),
		ram:      call.ram(memoryDomains, cpuClassCeiling),
		threads:  C.size_t(threads),
	}
	C.uform_executor_for_ram(arguments)
	return executorOf(block(&call, nil))
}

// ExecutorForVram opens one submission stream on every named device, all driven by the one host
// thread that calls the verbs, beside host threads on memoryDomains for the decoding the devices
// leave to the host.
//
// Streams is empty, or holds one stream per device in devices order: a CUDA or HIP stream becomes
// the executor's, which synchronizes and destroys it; a Metal queue is retained, the executor
// releasing its own reference while the caller keeps and releases theirs. Zero scratchLimit lets
// scratch grow as batches ask.
func ExecutorForVram(
	devices []int32, gpuClassCeiling GPUClass, memoryDomains []int32, cpuClassCeiling CPUClass,
	streams []unsafe.Pointer, scratchLimit uint,
) (*Executor, error) {
	// The library reads one stream per device from any non-nil pointer, so only this side sees a
	// short list.
	if len(streams) != 0 && len(streams) != len(devices) {
		return nil, fail(StatusShapeMismatch, "pass no streams or one per device")
	}
	var call callWithResult[C.uform_executor_t]
	arguments := open[C.uform_executor_for_vram_t](&call)
	*arguments = C.uform_executor_for_vram_t{
		executor:      call.result,
		outcome:       call.outcome(),
		vram:          call.vram(devices, gpuClassCeiling),
		ram:           call.ram(memoryDomains, cpuClassCeiling),
		streams:       (*unsafe.Pointer)(call.pinSlice(unsafe.Pointer(unsafe.SliceData(streams)), len(streams))),
		scratch_limit: C.size_t(scratchLimit),
	}
	C.uform_executor_for_vram(arguments)
	return executorOf(block(&call, nil))
}

// executorOf wraps the handle an executor factory wrote.
func executorOf(handle C.uform_executor_t, err error) (*Executor, error) {
	if err != nil {
		return nil, err
	}
	return &Executor{handle: handle}, nil
}

// Drain waits until no call holds this executor, then adopts mood until the next call; it only
// waits and never runs a call itself. A failure is the executor's own, such as a lost device,
// never one of a drained call.
func (executor *Executor) Drain(mood Mood) error {
	handle, err := executor.lower()
	if err != nil {
		return err
	}
	var call callWithResult[struct{}]
	arguments := open[C.uform_executor_drain_t](&call)
	*arguments = C.uform_executor_drain_t{executor: handle, outcome: call.outcome(), mood: C.uform_mood_t(mood)}
	C.uform_executor_drain(arguments)
	_, err = block(&call, nil)
	return err
}

// endregion: Executors

// region: Arenas

// Arena is memory the library writes every output whose length scales with a call into, pooled
// embeddings, per-token states and their offsets alike, reused so a steady loop allocates once.
//
// Every call given an arena rewinds and refills it, so the views of its previous call and every
// slice they handed out end there, as they do once it is closed; to keep one call's views past
// the next, give that call another arena. An arena serves one call at a time, and a second
// overlapping it fails with StatusContended. Deferred verbs hold it as blocking ones do, since
// Close waits for every deferred call naming it. A verb given an arena whose Describe refuses host
// access fails with StatusDevice before it runs, since no Go code may read its views.
type Arena struct {
	handle C.uform_arena_t
}

// NewArena makes an empty arena where executor computes, host pages or the memory of its GPU,
// decided once; a nil executor makes host pages for calls on the calling goroutine. Nothing is
// allocated until a call writes into it. An arena must not outlive the executor that made it, so
// defer [Executor.Close] before [Arena.Close]; one made with a nil executor lives until it is
// closed. A GPU arena serves only calls on its executor.
func NewArena(executor *Executor) (*Arena, error) {
	handle, err := executor.lower()
	if err != nil {
		return nil, err
	}
	var call callWithResult[C.uform_arena_t]
	arguments := open[C.uform_arena_init_t](&call)
	*arguments = C.uform_arena_init_t{arena: call.result, executor: handle, outcome: call.outcome()}
	C.uform_arena_init(arguments)
	arena, err := block(&call, nil)
	if err != nil {
		return nil, err
	}
	return &Arena{handle: arena}, nil
}

// ArenaDescription is where an arena's outputs live, which decides whether Go code may read them.
//
// Mirrors uform_arena_description_t field for field, so the library writes it in place.
type ArenaDescription struct {
	// Residency is ResidencyRam for host pages and ResidencyVram for one device's memory.
	Residency Residency
	// Device is the ordinal of the GPU holding the outputs, as GPUDevices lists it, and -1 for host
	// pages.
	Device int32
	// Vendor is the vendor whose memory it is, as GPUDevices reports it.
	Vendor GPUVendor
	// VendorOrdinal is the vendor's own device number, which Arrow and DLPack name, and -1 for host
	// pages.
	VendorOrdinal int32
	// HostAccess is whether Go code may read the outputs.
	HostAccess HostAccess
}

// The library writes the description in place.
var _ = [1]byte{}[unsafe.Sizeof(ArenaDescription{})-C.sizeof_uform_arena_description_t]

// host refuses memory Go code may not read with StatusDevice.
func (description ArenaDescription) host() error {
	if description.HostAccess != HostAccessReadable {
		return fail(StatusDevice, "the outputs are memory the host may not read")
	}
	return nil
}

// Describe reports where the arena's outputs live, as the executor that made it decided.
func (arena *Arena) Describe() (ArenaDescription, error) {
	handle, err := arena.lower()
	if err != nil {
		return ArenaDescription{}, err
	}
	var call callWithResult[ArenaDescription]
	arguments := open[C.uform_arena_describe_t](&call)
	*arguments = C.uform_arena_describe_t{
		arena:       handle,
		outcome:     call.outcome(),
		description: (*C.uform_arena_description_t)(unsafe.Pointer(call.result)),
	}
	C.uform_arena_describe(arguments)
	return block(&call, nil)
}

// endregion: Arenas

// region: Calls

// slots is what the library reads and writes for every verb: the outcome, and the handles its
// callbacks reach Go through.
type slots struct {
	outcome C.uform_outcome_t
	handles [2]cgo.Handle
}

// frame is one call's slots, argument structure and result in a single pinned allocation, whose
// only Go pointers are those the arguments hold, every one of them pinned until the call settles.
type frame[Arguments, Result any] struct {
	slots
	arguments Arguments
	result    Result
}

// call is one verb crossing into the library: the pins that keep every Go pointer its argument
// structure carries in place until the call is final, and the slots it reports through. cgo lets C
// keep a Go pointer only while it is pinned, and pinning is also what keeps every pointee on the
// heap, which the noescape promises rely on.
type call struct {
	pinner runtime.Pinner
	slots  *slots
	// arguments is the frame's argument structure, cleared once the call is final, so a settled
	// frame holds no Go pointer and its outcome may still cross to C unpinned.
	arguments []byte
}

// callWithResult is a call and the slot its result lands in, which the library writes in place.
type callWithResult[Result any] struct {
	call
	result *Result
}

// open pins a fresh frame and hands back its argument structure, which a verb fills and passes to
// C; verbs refuse before opening, so a refusal pins and allocates nothing. The arguments live in
// the frame because cgo moves to the heap whatever a C function that may call back into Go reaches.
func open[Arguments, Result any](opened *callWithResult[Result]) *Arguments {
	frame := new(frame[Arguments, Result])
	opened.pinner.Pin(frame)
	opened.slots, opened.result = &frame.slots, &frame.result
	opened.arguments = unsafe.Slice((*byte)(unsafe.Pointer(&frame.arguments)), unsafe.Sizeof(frame.arguments))
	return &frame.arguments
}

// outcome is where the library reports the call's status.
func (call *call) outcome() *C.uform_outcome_t { return &call.slots.outcome }

// status reads the outcome word without waiting.
func (call *call) status() Status {
	return Status(C.uform_outcome_status(call.outcome()))
}

// settle waits until the call is final, releases everything it pinned, and reports its failure, the
// same every time it is asked.
func (call *call) settle() error {
	if call.status() == StatusDeferred {
		C.uform_outcome_wait(call.outcome())
	}
	for slot, handle := range call.slots.handles {
		if handle != 0 {
			handle.Delete()
			call.slots.handles[slot] = 0
		}
	}
	clear(call.arguments)
	call.pinner.Unpin()
	return check(call.slots.outcome)
}

// block settles the call a verb opened and hands back its result, or the verb's refusal.
func block[Result any](call *callWithResult[Result], refusal error) (Result, error) {
	var none Result
	if refusal != nil {
		return none, refusal
	}
	if err := call.settle(); err != nil {
		return none, err
	}
	return *call.result, nil
}

// configOf reads a nil config as the zero one, by value, so generic verbs allocate nothing for it.
func configOf[Config any](config *Config) Config {
	var options Config
	if config != nil {
		options = *config
	}
	return options
}

// pin keeps a Go object in place until the call is final and hands back its address. It takes no
// type parameter, since a generic call inlined into a generic verb reaches its callee through a
// dictionary, and escape analysis then moves the verb's call to the heap.
func (call *call) pin(pointer unsafe.Pointer) unsafe.Pointer {
	call.pinner.Pin(pointer)
	return pointer
}

// pinSlice pins the backing array of a slice of length elements at data until the call is final,
// and answers nil for an empty slice.
func (call *call) pinSlice(data unsafe.Pointer, length int) unsafe.Pointer {
	if length == 0 {
		return nil
	}
	return call.pin(data)
}

// view passes a string to a call that only reads it before returning.
func view(text string) C.uform_bytes_view_t {
	view := C.uform_bytes_view_t{data: (*C.uint8_t)(unsafe.Pointer(unsafe.StringData(text)))}
	// cgo names the length's typedef apart from size_t, though the two are as wide.
	*(*C.size_t)(unsafe.Pointer(&view.length)) = C.size_t(len(text))
	return view
}

// Deferred is a submitted call, which holds its outcome and inputs until it is final, owned by the
// one goroutine that polls or waits on it. Status polls without blocking, Wait blocks until the
// call is final and hands back its result, and Close waits the same way and drops it. A Deferred
// dropped without Wait or Close is released by the collector once the call is final, never
// blocking on it: until then every later collection checks it again.
type Deferred[Result any] struct {
	call    callWithResult[Result]
	cleanup runtime.Cleanup
}

// submitted hands the call a verb opened to its caller, who waits on it or closes it, or drops it
// to the collector; a verb's refusal comes back instead, with nothing submitted.
func submitted[Result any](deferred *Deferred[Result], refusal error) (*Deferred[Result], error) {
	if refusal != nil {
		return nil, refusal
	}
	deferred.cleanup = runtime.AddCleanup(deferred, release, deferred.call.call)
	return deferred, nil
}

// release settles a dropped call once it is final, and otherwise arms itself again for the next
// collection, since the collector's goroutine must not block.
func release(call call) {
	if call.status() == StatusDeferred {
		runtime.AddCleanup(&call, release, call)
		return
	}
	_ = call.settle()
}

// Status reports StatusDeferred until the call is final, then its final status, without blocking.
func (deferred *Deferred[Result]) Status() Status { return deferred.call.status() }

// Wait blocks until the call is final, releases its pins, then hands back its result, the same
// every time it is asked.
func (deferred *Deferred[Result]) Wait() (Result, error) {
	deferred.cleanup.Stop()
	return block(&deferred.call, nil)
}

// Close blocks until the call is final, releases its pins, and reports the call's failure. It is
// idempotent and safe after Wait.
func (deferred *Deferred[Result]) Close() error {
	deferred.cleanup.Stop()
	return deferred.call.settle()
}

// endregion: Calls

// region: Receivers

// receiver is a handle's field, so a Close is seen, and the executor its verbs run on.
type receiver[Handle comparable] struct {
	handle   *Handle
	executor *Executor
}

// lower reports the handles a verb hands across, refusing a closed handle before C dereferences it.
func (receiver receiver[Handle]) lower() (Handle, C.uform_executor_t, error) {
	var none Handle
	if receiver.handle == nil || *receiver.handle == none {
		return none, nil, fail(StatusUninitialized, "the handle is closed or was never initialized")
	}
	executor, err := receiver.executor.lower()
	return *receiver.handle, executor, err
}

// lowerInto is lower for a verb writing into arena, refusing a closed one too.
func (receiver receiver[Handle]) lowerInto(arena *Arena) (Handle, C.uform_executor_t, C.uform_arena_t, error) {
	handle, executor, err := receiver.lower()
	if err != nil {
		return handle, executor, nil, err
	}
	outputs, err := arena.lower()
	return handle, executor, outputs, err
}

// lower reports the handle a verb hands across: nil for the null executor, and a refusal for a
// closed one rather than a silent fall back to the calling goroutine.
func (executor *Executor) lower() (C.uform_executor_t, error) {
	if executor == nil {
		return nil, nil
	}
	if executor.handle == nil {
		return nil, fail(StatusUninitialized, "the executor is closed or was never spawned")
	}
	return executor.handle, nil
}

// Close waits out the call holding the executor, then releases its threads, rings or streams; every
// arena it made must be closed first. It is idempotent; a closed executor must not be used again.
func (executor *Executor) Close() error {
	if executor == nil || executor.handle == nil {
		return nil
	}
	C.uform_executor_free(executor.handle)
	executor.handle = nil
	return nil
}

// lower reports the handle a verb hands across: nil for a nil arena, which the call refuses, and a
// refusal for a closed one.
func (arena *Arena) lower() (C.uform_arena_t, error) {
	if arena == nil {
		return nil, nil
	}
	if arena.handle == nil {
		return nil, fail(StatusUninitialized, "the arena is closed or was never made")
	}
	return arena.handle, nil
}

// Close releases the memory, waiting for every deferred call naming the arena, and ends every view
// over it; it comes before the Close of the executor that made the arena. While an Arrow export of
// its outputs is unreleased it fails with [StatusContended] and frees nothing, so a later Close
// retries. It is idempotent, and a closed arena must not be used again.
func (arena *Arena) Close() error {
	if arena == nil || arena.handle == nil {
		return nil
	}
	if status := C.uform_arena_free(arena.handle); status != C.uform_status_success_k {
		return fail(Status(status), "an export of the arena's outputs is still open")
	}
	arena.handle = nil
	return nil
}

// endregion: Receivers

// lowerReadableInto is lowerInto for a verb whose results wrap the arena's outputs as Go slices, so
// it refuses an arena whose memory the host may not read before the call rather than after it.
func (receiver receiver[Handle]) lowerReadableInto(arena *Arena) (Handle, C.uform_executor_t, C.uform_arena_t, error) {
	handle, executor, outputs, err := receiver.lowerInto(arena)
	if err != nil || arena == nil {
		return handle, executor, outputs, err
	}
	description, err := arena.Describe()
	if err == nil {
		err = description.host()
	}
	return handle, executor, outputs, err
}

// region: Introspection

// Version reports the version of the linked library.
func Version() string { return C.GoString(C.uform_version_name()) }

// CapabilitiesCompiled lists the hardware capabilities the linked binary was compiled with,
// comma separated.
func CapabilitiesCompiled() string { return C.GoString(C.uform_capabilities_compiled()) }

// CapabilitiesAvailable lists the hardware capabilities the current machine offers at
// runtime, comma separated.
func CapabilitiesAvailable() string { return C.GoString(C.uform_capabilities_available()) }

// endregion: Introspection

// region: Helpers

// optional pins a caller's report until the call is final, and passes a nil one as nil.
func (call *call) optional(report unsafe.Pointer) unsafe.Pointer {
	if report == nil {
		return nil
	}
	return call.pin(report)
}

// ints pins a slice of IDs until the call is final.
func (call *call) ints(values []int32) (*C.int32_t, C.size_t) {
	return (*C.int32_t)(call.pinSlice(unsafe.Pointer(unsafe.SliceData(values)), len(values))), C.size_t(len(values))
}

// ram pins the memory domains of a RAM placement or executor, beside the class ceiling.
func (call *call) ram(memoryDomains []int32, cpuClassCeiling CPUClass) C.uform_ram_options_t {
	domains, count := call.ints(memoryDomains)
	return C.uform_ram_options_t{
		memory_domains:       domains,
		memory_domains_count: count,
		cpu_class_ceiling:    C.uform_cpu_class_t(cpuClassCeiling),
	}
}

// vram pins the devices of a VRAM placement or executor, beside the class ceiling.
func (call *call) vram(devices []int32, gpuClassCeiling GPUClass) C.uform_vram_options_t {
	ordinals, count := call.ints(devices)
	return C.uform_vram_options_t{
		devices:           ordinals,
		devices_count:     count,
		gpu_class_ceiling: C.uform_gpu_class_t(gpuClassCeiling),
	}
}

// bytes pins a byte slice as the view the C interface reads, with no terminator and no copy.
func (call *call) bytes(bytes []byte) C.uform_bytes_view_t {
	return C.uform_bytes_view_t{
		data:   (*C.uint8_t)(call.pinSlice(unsafe.Pointer(unsafe.SliceData(bytes)), len(bytes))),
		length: C.size_t(len(bytes)),
	}
}

// text views a Go string as the pointer-and-length pair every path argument takes. The string is
// never terminated, by contract, so nothing is copied and nothing is freed.
func (call *call) text(text string) C.uform_bytes_view_t {
	if len(text) == 0 {
		return C.uform_bytes_view_t{}
	}
	return C.uform_bytes_view_t{
		data:   (*C.uint8_t)(call.pin(unsafe.Pointer(unsafe.StringData(text)))),
		length: C.size_t(len(text)),
	}
}

// uint32s pins a slice of per-item counts or tags as the C array a verb reads.
func (call *call) uint32s(values []uint32) *C.uint32_t {
	return (*C.uint32_t)(call.pinSlice(unsafe.Pointer(unsafe.SliceData(values)), len(values)))
}

// endregion: Helpers
