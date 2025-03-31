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
	index, _ = NewProductQuantizationIndex(2, 2, WithPQNumClusters(2), WithPQNumIterations(10), WithPQTol(0.001))
	if _, ok := index.(*ProductQuantizationIndex); !ok {
		t.Fatalf("index is not a ProductQuantizationIndex")
	}
}

func TestInvertedFileIndexInterface(t *testing.T) {
	var index ANNIndex
	index, _ = NewInvertedFileIndex(2, WithIVFNumClusters(2), WithIVFNumIterations(10), WithIVFTol(0.001))
	if _, ok := index.(*InvertedFileIndex); !ok {
		t.Fatalf("index is not a InvertedFileIndex")
	}
}

func TestInvertedFileIndexWithPQInterface(t *testing.T) {
	var index ANNIndex
	index, _ = NewInvertedFileIndex(2, WithIVFNumClusters(2), WithIVFNumIterations(10), WithIVFTol(0.001), WithIVFPQIndex(
		2, WithPQNumClusters(2), WithPQNumIterations(10), WithPQTol(0.001),
	))
	if _, ok := index.(*InvertedFileIndex); !ok {
		t.Fatalf("index is not a InvertedFileIndex")
	}
}
