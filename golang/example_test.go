// The Quickstart of README.md, compiled by go test.
//
// File: golang/example_test.go
// Author: Ash Vardanian
// Date: November 18, 2026

package uform_test

import (
	"fmt"
	"log"
	"math"
	"os"

	uform "github.com/unum-cloud/uform/golang"
)

func Example() {
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
