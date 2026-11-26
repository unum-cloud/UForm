// Tests of the forward passes: surfaces, views, refusals and deferred calls.
//
// File: golang/encode_test.go
// Author: Ash Vardanian
// Date: November 18, 2026

package uform_test

import (
	"errors"
	"runtime"
	"testing"

	uform "github.com/unum-cloud/uform/golang"
)

func TestTextEncodeSurfaces(t *testing.T) {
	executor, arena, model := newFixture(t)
	description, err := model.Describe()
	if err != nil {
		t.Fatal(err)
	}
	stats := new(uform.TextEncodeStats)
	encoded, err := model.On(executor).TextEncode(uform.TapeOf([]string{"a cat on a mat", "a sunrise", "a dog"}),
		uform.EncodeOutputsEmbeddings|uform.EncodeOutputsStates, uform.ScalarF32, arena, &uform.TextEncodeConfig{Stats: stats})
	if err != nil {
		t.Fatal(err)
	}
	if encoded.Embeddings.Count() != 3 || encoded.Embeddings.Columns() != description.Dimensions {
		t.Fatalf("embeddings read %d rows of %d", encoded.Embeddings.Count(), encoded.Embeddings.Columns())
	}
	row, err := uform.RowValues[float32](encoded.Embeddings, 2)
	if err != nil || uint(len(row)) != description.Dimensions {
		t.Fatalf("a row reads %d values, %v", len(row), err)
	}
	if _, err := uform.RowValues[float64](encoded.Embeddings, 0); !errors.Is(err, uform.StatusInvalidArgument) {
		t.Fatalf("reading f32 rows as f64 answers %v", err)
	}
	if _, err := uform.RowValues[float32](encoded.Embeddings, 3); !errors.Is(err, uform.StatusInvalidArgument) {
		t.Fatalf("reading past the rows answers %v", err)
	}
	offsets := encoded.States.Offsets()
	if len(offsets) != 4 || offsets[3] != stats.Tokens {
		t.Fatalf("state offsets %v disagree with %d tokens", offsets, stats.Tokens)
	}
	item, err := encoded.States.Item(1)
	if err != nil || item.Count() != offsets[2]-offsets[1] {
		t.Fatalf("item 1 reads %d rows, %v", item.Count(), err)
	}
	if encoded.TokenVectors.Count() != 0 || encoded.TokenVectors.Offsets() != nil {
		t.Fatal("a surface left out stays empty")
	}
	if _, err := model.On(executor).TextEncode(uform.TapeOf([]string{"a"}), uform.EncodeOutputsEmbeddings,
		uform.ScalarF32, arena, nil); err != nil {
		t.Fatal(err)
	}
	if _, err := uform.RowValues[float32](encoded.Embeddings, 0); !errors.Is(err, uform.StatusContended) {
		t.Fatalf("a result read after a later call answers %v", err)
	}
}

func TestEncodeRefusals(t *testing.T) {
	executor, arena, model := newFixture(t)
	on := model.On(executor)
	pixels := uform.BytesTape(make([]byte, 12), []uint{0, 12})
	if _, err := on.ImageEncodePixels(pixels, []uint32{2}, []uint32{2, 2}, uform.EncodeOutputsEmbeddings, uform.ScalarF32, arena, nil); !errors.Is(err, uform.StatusShapeMismatch) {
		t.Fatalf("unpaired shapes answer %v", err)
	}
	if _, err := model.Deferred(executor).ImageEncodePixels(pixels, nil, nil, uform.EncodeOutputsEmbeddings, uform.ScalarF32, arena, nil); !errors.Is(err, uform.StatusShapeMismatch) {
		t.Fatalf("a deferred refusal answers %v before anything is submitted", err)
	}
	texts := uform.TapeOf([]string{"one", "two"})
	if _, err := on.TextChoose(texts, uform.TapeOf([]string{"yn"}), arena, nil); !errors.Is(err, uform.StatusShapeMismatch) {
		t.Fatalf("unpaired choices answer %v", err)
	}
	if _, err := on.TextChoose(texts, uform.TapeOf([]string{"yn", "ABCD"}), arena, nil); !errors.Is(err, uform.StatusUnsupported) {
		t.Fatalf("the stub carries no byte decoder, yet answers %v", err)
	}
	if _, err := on.TextEncode(texts, uform.EncodeOutputsEmbeddings, uform.ScalarF32, nil, nil); !errors.Is(err, uform.StatusInvalidArgument) {
		t.Fatalf("a nil arena answers %v", err)
	}
	closed, err := uform.NewArena(executor)
	if err != nil {
		t.Fatal(err)
	}
	closed.Close()
	if _, err := on.TextEncode(texts, uform.EncodeOutputsEmbeddings, uform.ScalarF32, closed, nil); !errors.Is(err, uform.StatusUninitialized) {
		t.Fatalf("a closed arena answers %v", err)
	}
}

func TestOtherForms(t *testing.T) {
	executor, arena, model := newFixture(t)
	on := model.On(executor)
	files := uform.TapeOf([][]byte{make([]byte, 40), make([]byte, 90)})
	for name, encode := range map[string]func() (uform.Encoded, error){
		"image": func() (uform.Encoded, error) {
			return on.ImageEncode(files, uform.EncodeOutputsEmbeddings, uform.ScalarF32, arena, nil)
		},
		"video": func() (uform.Encoded, error) {
			return on.VideoEncode(files, uform.EncodeOutputsEmbeddings, uform.ScalarF32, arena, nil)
		},
		"audio": func() (uform.Encoded, error) {
			return on.AudioEncode(files, uform.EncodeOutputsEmbeddings, uform.ScalarF32, arena, nil)
		},
		"pixels": func() (uform.Encoded, error) {
			return on.ImageEncodePixels(uform.BytesTape(make([]byte, 12), []uint{0, 12}), []uint32{2}, []uint32{2},
				uform.EncodeOutputsEmbeddings, uform.ScalarF32, arena, nil)
		},
		"samples": func() (uform.Encoded, error) {
			return on.AudioEncodeSamples(uform.SamplesOf([][]float32{make([]float32, 512)}), 16000,
				uform.EncodeOutputsEmbeddings, uform.ScalarF32, arena, nil)
		},
	} {
		encoded, err := encode()
		if err != nil || encoded.Embeddings.Count() == 0 {
			t.Fatalf("%s encodes %d rows, %v", name, encoded.Embeddings.Count(), err)
		}
	}
}

func TestADroppedCallSettlesAfterItsArenaClosed(t *testing.T) {
	executor, arena, model := newFixture(t)
	if _, err := model.Deferred(executor).TextEncode(uform.TapeOf([]string{"a fox"}), uform.EncodeOutputsEmbeddings, uform.ScalarF32, arena, nil); err != nil {
		t.Fatal(err)
	}
	if err := arena.Close(); err != nil {
		t.Fatal(err)
	}
	runtime.GC()
	runtime.GC()
}

func TestDeferredEncode(t *testing.T) {
	executor, arena, model := newFixture(t)
	call, err := model.Deferred(executor).TextEncode(uform.TapeOf([]string{"a cat", "a dog"}), uform.EncodeOutputsEmbeddings, uform.ScalarF32, arena, nil)
	if err != nil {
		t.Fatal(err)
	}
	encoded, err := call.Wait()
	if err != nil || encoded.Embeddings.Count() != 2 {
		t.Fatalf("a deferred encode reads %d rows, %v", encoded.Embeddings.Count(), err)
	}

	// A dropped call is settled by the collector once final, and its arena serves the next call.
	if _, err := model.Deferred(executor).TextEncode(uform.TapeOf([]string{"a fox"}), uform.EncodeOutputsEmbeddings, uform.ScalarF32, arena, nil); err != nil {
		t.Fatal(err)
	}
	runtime.GC()
	if _, err := model.On(executor).TextEncode(uform.TapeOf([]string{"a hen"}), uform.EncodeOutputsEmbeddings, uform.ScalarF32, arena, nil); err != nil {
		t.Fatal(err)
	}
}
