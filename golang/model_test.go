// Tests of the model lifecycle and its receivers, against the stub library.
//
// File: golang/model_test.go
// Author: Ash Vardanian
// Date: November 18, 2026

package uform_test

import (
	"errors"
	"testing"

	uform "github.com/unum-cloud/uform/golang"
)

func TestModelLoads(t *testing.T) {
	if _, err := uform.ModelLoadRam("/nonexistent/uform.safetensors", nil); !errors.Is(err, uform.StatusIO) {
		t.Fatalf("a missing checkpoint answers %v", err)
	}
	if _, err := uform.ModelLoadRamBuffer([]byte("uform"), &uform.ModelLoadRamConfig{Quantization: uform.ScalarF32}); !errors.Is(err, uform.StatusUnsupported) {
		t.Fatalf("f32 blocks answer %v", err)
	}
	if _, err := uform.ModelLoadVramBuffer([]byte("uform"), nil); err == nil {
		t.Fatal("the stub carries no GPU")
	}
	model, err := uform.ModelLoadRamBuffer([]byte("uform"), nil)
	if err != nil {
		t.Fatal(err)
	}
	description, err := model.Describe()
	if err != nil || description.Dimensions == 0 || description.Quantization != uform.ScalarBF16 {
		t.Fatalf("a loaded model reads %+v, %v", description, err)
	}
	model.Close()
	model.Close()
	if _, err := model.Describe(); !errors.Is(err, uform.StatusUninitialized) {
		t.Fatalf("a closed model answers %v", err)
	}
}

func TestModelReceivers(t *testing.T) {
	executor, _, model := newFixture(t)
	blocking, err := model.On(executor).Describe()
	if err != nil {
		t.Fatal(err)
	}
	call, err := model.Deferred(executor).Describe()
	if err != nil {
		t.Fatal(err)
	}
	if status := call.Status(); status != uform.StatusDeferred && status != uform.StatusSuccess {
		t.Fatalf("a submitted call polls %v", status)
	}
	deferred, err := call.Wait()
	if err != nil || deferred != blocking {
		t.Fatalf("a deferred describe reads %+v, %v", deferred, err)
	}
	again, err := call.Wait()
	if err != nil || again != deferred {
		t.Fatal("waiting again must answer the same")
	}
	if err := call.Close(); err != nil {
		t.Fatal(err)
	}
}
