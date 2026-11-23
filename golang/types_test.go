// Tests of the shared vocabulary and of the calls every verb crosses through.
//
// File: golang/types_test.go
// Author: Ash Vardanian
// Date: November 18, 2026

package uform_test

import (
	"errors"
	"runtime"
	"testing"

	uform "github.com/unum-cloud/uform/golang"
)

// newFixture makes an executor, an arena on it, and a model loaded from arbitrary bytes, which
// only the stub accepts, closing all three when the test ends.
func newFixture(t *testing.T) (*uform.Executor, *uform.Arena, *uform.Model) {
	t.Helper()
	executor, err := uform.ExecutorForRam(nil, uform.CPUClassUnknown, 2)
	if err != nil {
		t.Fatal(err)
	}
	arena, err := uform.NewArena(executor)
	if err != nil {
		t.Fatal(err)
	}
	model, err := uform.ModelLoadRamBuffer([]byte("uform"), &uform.ModelLoadRamConfig{Executor: executor})
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() {
		model.Close()
		arena.Close()
		executor.Close()
	})
	return executor, arena, model
}

func TestStatusesAreErrors(t *testing.T) {
	var err error = &uform.Error{Status: uform.StatusShapeMismatch, Message: "the counts disagree"}
	if !errors.Is(err, uform.StatusShapeMismatch) || errors.Is(err, uform.StatusIO) {
		t.Fatalf("errors.Is misreads %v", err)
	}
	var failure *uform.Error
	if !errors.As(err, &failure) || failure.Status != uform.StatusShapeMismatch {
		t.Fatalf("errors.As misreads %v", err)
	}
	if uform.StatusShapeMismatch.String() != "shape-mismatch" || uform.ScalarBF16.String() != "bf16" {
		t.Fatalf("names are the library's: %s, %s", uform.StatusShapeMismatch, uform.ScalarBF16)
	}
}

func TestNamesRoundTrip(t *testing.T) {
	status, err := uform.StatusFromName(uform.StatusShapeMismatch.String())
	if err != nil || status != uform.StatusShapeMismatch {
		t.Fatalf("read back %v, %v", status, err)
	}
	scalar, err := uform.ScalarFromName("bf16")
	if err != nil || scalar != uform.ScalarBF16 {
		t.Fatalf("read back %v, %v", scalar, err)
	}
	modality, err := uform.ModalityFromName(uform.ModalityImage.String())
	if err != nil || modality != uform.ModalityImage {
		t.Fatalf("read back %v, %v", modality, err)
	}
	if _, err := uform.MoodFromName("sleepy"); !errors.Is(err, uform.StatusInvalidArgument) {
		t.Fatalf("an unknown spelling answered %v", err)
	}
}

func TestTapes(t *testing.T) {
	tape := uform.TapeOf([]string{"a cat", "", "a dog"})
	if tape.Entries() != 3 || tape.Scalar() != uform.ScalarU8 {
		t.Fatalf("gathered %d entries of %s", tape.Entries(), tape.Scalar())
	}
	if (uform.Tape{}).Entries() != 0 {
		t.Fatal("the zero tape holds no entries")
	}
}

func TestExecutorsAndArenas(t *testing.T) {
	executor, arena, _ := newFixture(t)
	if err := executor.Drain(uform.MoodSpinning); err != nil {
		t.Fatal(err)
	}
	description, err := arena.Describe()
	if err != nil || description.Residency != uform.ResidencyRam || description.Device != -1 {
		t.Fatalf("a host arena reads %+v, %v", description, err)
	}
	devices := make([]uform.GPUDevice, 4)
	if _, err := uform.GPUDevices(devices); err != nil {
		t.Fatal(err)
	}
	if _, err := uform.ExecutorForVram(nil, uform.GPUClassUnknown, nil, uform.CPUClassUnknown, nil, 0); err == nil {
		t.Fatal("the stub carries no GPU")
	}

	closed, err := uform.ExecutorForRam(nil, uform.CPUClassUnknown, 1)
	if err != nil {
		t.Fatal(err)
	}
	closed.Close()
	closed.Close()
	if err := closed.Drain(uform.MoodParked); !errors.Is(err, uform.StatusUninitialized) {
		t.Fatalf("a closed executor answers %v", err)
	}
	if _, err := uform.NewArena(closed); !errors.Is(err, uform.StatusUninitialized) {
		t.Fatalf("an arena on a closed executor answers %v", err)
	}
}

func TestProgress(t *testing.T) {
	executor, arena, model := newFixture(t)
	reports := uint(0)
	config := &uform.TextEncodeConfig{ProgressSteps: 2, Progress: func(completed, total uint) bool {
		reports++
		return true
	}}
	if _, err := model.On(executor).TextEncode(uform.TapeOf([]string{"a", "b", "c"}), uform.EncodeOutputsEmbeddings, uform.ScalarF32, arena, config); err != nil {
		t.Fatal(err)
	}
	if reports == 0 {
		t.Fatal("the handler heard nothing")
	}
	stop := &uform.TextEncodeConfig{ProgressSteps: 1, Progress: func(completed, total uint) bool { return false }}
	if _, err := model.On(executor).TextEncode(uform.TapeOf([]string{"a"}), uform.EncodeOutputsEmbeddings, uform.ScalarF32, arena, stop); !errors.Is(err, uform.StatusCancelled) {
		t.Fatalf("a stopping handler answers %v", err)
	}
}

func TestCallsAllocateOnlyTheirFrames(t *testing.T) {
	executor, arena, model := newFixture(t)
	texts := uform.TapeOf([]string{"a cat on a mat", "a sunrise over the bay"})
	pixels := uform.BytesTape(make([]byte, 12), []uint{0, 12})
	heights := []uint32{2}
	on, deferred := model.On(executor), model.Deferred(executor)

	for name, verb := range map[string]func(){
		"blocking describe": func() { _, _ = on.Describe() },
		"blocking encode":   func() { _, _ = on.TextEncode(texts, uform.EncodeOutputsEmbeddings, uform.ScalarF32, arena, nil) },
		"refused encode": func() {
			_, _ = on.ImageEncodePixels(pixels, heights, nil, uform.EncodeOutputsEmbeddings, uform.ScalarF32, arena, nil)
		},
	} {
		allocations := testing.AllocsPerRun(100, verb)
		t.Logf("a %s allocates %v times", name, allocations)
		if allocations != 1 {
			t.Errorf("a %s allocated %v times", name, allocations)
		}
	}
	// The holder stands for the Deferred, whose cleanup the runtime registers the same way.
	cleanup := testing.AllocsPerRun(100, func() { runtime.AddCleanup(new([2]uintptr), func(uint) {}, 0) })
	for name, verb := range map[string]func(){
		"deferred describe": func() {
			call, _ := deferred.Describe()
			_, _ = call.Wait()
		},
		"deferred encode": func() {
			call, _ := deferred.TextEncode(texts, uform.EncodeOutputsEmbeddings, uform.ScalarF32, arena, nil)
			_, _ = call.Wait()
		},
	} {
		allocations := testing.AllocsPerRun(100, verb)
		t.Logf("a %s and its Wait allocate %v times, %v of them registering its cleanup", name, allocations, cleanup)
		if allocations != 1+cleanup {
			t.Errorf("a %s allocated %v times, %v of them registering its cleanup", name, allocations, cleanup)
		}
	}
}
