package vanadium_index

import (
	"bytes"
	"encoding/gob"
	"testing"
)

func TestFlatIndexSearch(t *testing.T) {
	index, _ := NewFlatIndex(2)
	index.Add([]float32{1, 2, 3, 4})
	index.Add([]float32{5, 6})

	results, err := index.Search([]float32{1, 2, 3, 4, 5, 6}, 1)
	if err != nil {
		t.Fatalf("error: %v", err)
	}
	for i, result := range results {
		if result[0] != i {
			t.Fatalf("result[%d] = %d, expected %d", i, result[0], i)
		}
	}
}

func TestFlatIndexEncodeDecode(t *testing.T) {
	index, _ := NewFlatIndex(2)
	index.Add([]float32{1, 2, 3, 4})
	index.Add([]float32{5, 6})

	var buf bytes.Buffer
	enc := gob.NewEncoder(&buf)
	err := index.Encode(enc)
	if err != nil {
		t.Fatalf("error: %v", err)
	}

	dec := gob.NewDecoder(&buf)
	index2, err := LoadFlatIndex(dec)
	if err != nil {
		t.Fatalf("error: %v", err)
	}

	if index2.state.NumFeatures != index.state.NumFeatures {
		t.Fatalf("numFeatures mismatch: %d != %d", index2.state.NumFeatures, index.state.NumFeatures)
	}

	numVectors := index.NumVectors()
	if numVectors != index2.NumVectors() {
		t.Fatalf("numVectors mismatch: %d != %d", numVectors, index2.NumVectors())
	}
	numFeatures := index.state.NumFeatures

	for i := range numVectors {
		for j := range numFeatures {
			if index.state.Data[i*numFeatures+j] != index2.state.Data[i*numFeatures+j] {
				t.Fatalf("data mismatch: %v != %v", index.state.Data[i*numFeatures+j], index2.state.Data[i*numFeatures+j])
			}
		}
	}
}
