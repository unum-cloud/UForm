// Tests of the training verbs' refusals, against the stub library, which carries no GPU.
//
// File: golang/train_test.go
// Author: Ash Vardanian
// Date: November 18, 2026

package uform_test

import (
	"errors"
	"testing"

	uform "github.com/unum-cloud/uform/golang"
)

func TestTrainingRefusals(t *testing.T) {
	executor, _, _ := newFixture(t)
	if _, err := uform.MeshBootstrap(executor, nil); !errors.Is(err, uform.StatusDevice) {
		t.Fatalf("a mesh over a RAM executor answers %v", err)
	}
	if _, err := uform.TrainerInitVram(executor, &uform.TrainerInitVramConfig{Depth: 2, Hidden: 256, Schedule: uform.Schedule{Steps: 1}}); !errors.Is(err, uform.StatusDevice) {
		t.Fatalf("a trainer over a RAM executor answers %v", err)
	}
	if _, err := (*uform.Mesh)(nil).Describe(); !errors.Is(err, uform.StatusUninitialized) {
		t.Fatalf("a missing mesh answers %v", err)
	}

	trainer := new(uform.Trainer)
	contents := uform.TapeOf([]string{"hi", "there"})
	modalities := []uform.Modality{uform.ModalityText, uform.ModalityText}
	if _, err := trainer.On(executor).Step(modalities, contents, nil); !errors.Is(err, uform.StatusUninitialized) {
		t.Fatalf("a closed trainer answers %v", err)
	}
	// A closed handle refuses before the binding reads the shapes, as every binding orders it.
	if _, err := trainer.On(executor).Step(modalities[:1], contents, nil); !errors.Is(err, uform.StatusUninitialized) {
		t.Fatalf("unpaired modalities on a closed trainer answer %v", err)
	}
	unweighted := &uform.TrainerStepConfig{Relations: uform.Relations{Attract: uform.Edges{Offsets: []uint{0, 1, 2}, IDs: []uint32{1, 0}, Weights: []float32{1}}}}
	if _, err := trainer.Deferred(executor).Step(modalities, contents, unweighted); !errors.Is(err, uform.StatusUninitialized) {
		t.Fatalf("unpaired weights on a closed trainer answer %v", err)
	}
	if err := trainer.On(executor).Save("checkpoint.safetensors", nil); !errors.Is(err, uform.StatusUninitialized) {
		t.Fatalf("saving a closed trainer answers %v", err)
	}
	if uform.ModalityImage.String() != "image" || uform.ObjectiveJoint.String() != "joint" || uform.PairLossFormRanked.String() != "ranked" {
		t.Fatal("names are the library's")
	}
}
