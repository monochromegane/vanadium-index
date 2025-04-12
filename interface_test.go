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
	index, _ = NewProductQuantizationIndex[uint8](2, 2, 2, WithPQMaxIterations(10), WithPQTolerance(0.001))
	if _, ok := index.(*ProductQuantizationIndex[uint8]); !ok {
		t.Fatalf("index is not a ProductQuantizationIndex")
	}
}

func TestInvertedFileFlatIndexInterface(t *testing.T) {
	var index ANNIndex
	index, _ = NewInvertedFileFlatIndex[uint8](2, 2, WithIVFMaxIterations(10), WithIVFTolerance(0.001))
	if _, ok := index.(*InvertedFileIndex[uint8, uint8]); !ok {
		t.Fatalf("index is not a InvertedFileIndex")
	}
}

func TestInvertedFilePQIndexInterface(t *testing.T) {
	var index ANNIndex
	index, _ = NewInvertedFilePQIndex[uint8, uint8](2, 2, 2, 2, WithIVFMaxIterations(10), WithIVFTolerance(0.001), WithIVFPQIndex(
		WithPQMaxIterations(10), WithPQTolerance(0.001),
	))
	if _, ok := index.(*InvertedFileIndex[uint8, uint8]); !ok {
		t.Fatalf("index is not a InvertedFileIndex")
	}
}
