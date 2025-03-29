package annindex

import (
	"testing"
)

func TestFlatIndexInterface(t *testing.T) {
	var index ANNIndex
	index, _ = NewFlatIndex(2)
	if _, ok := index.(*FlatIndex); !ok {
		t.Fatalf("index is not a FlatIndex")
	}
}

func TestProductQuantizationIndexInterface(t *testing.T) {
	var index ANNIndex
	index, _ = NewProductQuantizationIndex(2, 2, 2)
	if _, ok := index.(*ProductQuantizationIndex); !ok {
		t.Fatalf("index is not a ProductQuantizationIndex")
	}
}
