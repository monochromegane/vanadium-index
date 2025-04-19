package vanadium_index

import (
	"bytes"
	"encoding/gob"
	"testing"
)

func TestProductQuantizationIndex(t *testing.T) {
	numFeatures := 4
	numSubspaces := 2
	numClusters := uint8(4)
	numIterations := 10
	tol := float32(0.001)
	index, err := newProductQuantizationIndex(numFeatures, numSubspaces, numClusters, WithPQMaxIterations(numIterations), WithPQTolerance(tol))
	if err != nil {
		t.Fatalf("Failed to create index: %v", err)
	}

	data := []float32{
		0.1, 0.2, 0.3, 0.4,
		0.5, 0.6, 0.7, 0.8,
		0.9, 1.0, 1.1, 1.2,
		1.3, 1.4, 1.5, 1.6,
	}

	err = index.Train(data)
	if err != nil {
		t.Fatalf("Failed to train index: %v", err)
	}

	for i := 0; i < len(data); i += 4 {
		err = index.Add(data[i : i+4])
		if err != nil {
			t.Fatalf("Failed to add data: %v", err)
		}
	}

	query := data

	results, err := index.Search(query, 1)
	if err != nil {
		t.Fatalf("Failed to search index: %v", err)
	}

	for i, result := range results {
		if result[0] != i {
			t.Fatalf("result[%d] = %d, expected %d", i, result[0], i)
		}
	}
}

func TestProductQuantizationIndexSaveLoad(t *testing.T) {
	numFeatures := 4
	numSubspaces := 2
	numClusters := uint8(4)
	numIterations := 10
	tol := float32(0.001)
	index, err := newProductQuantizationIndex(numFeatures, numSubspaces, numClusters, WithPQMaxIterations(numIterations), WithPQTolerance(tol))
	if err != nil {
		t.Fatalf("Failed to create index: %v", err)
	}

	data := []float32{
		0.1, 0.2, 0.3, 0.4,
		0.5, 0.6, 0.7, 0.8,
		0.9, 1.0, 1.1, 1.2,
		1.3, 1.4, 1.5, 1.6,
	}

	err = index.Train(data)
	if err != nil {
		t.Fatalf("Failed to train index: %v", err)
	}

	for i := 0; i < len(data); i += 4 {
		err = index.Add(data[i : i+4])
		if err != nil {
			t.Fatalf("Failed to add data: %v", err)
		}
	}

	var buf bytes.Buffer
	enc := gob.NewEncoder(&buf)
	err = index.Save(enc)
	if err != nil {
		t.Fatalf("Failed to save index: %v", err)
	}

	dec := gob.NewDecoder(&buf)
	annIndex, err := LoadIndex(dec)
	if err != nil {
		t.Fatalf("error: %v", err)
	}

	index2, ok := annIndex.(*ProductQuantizationIndex[uint8])
	if !ok {
		t.Fatalf("loaded index is not a ProductQuantizationIndex")
	}

	if index2.state.NumFeatures != index.state.NumFeatures {
		t.Fatalf("numFeatures mismatch: %d != %d", index2.state.NumFeatures, index.state.NumFeatures)
	}

	numVectors := index.NumVectors()
	if numVectors != index2.NumVectors() {
		t.Fatalf("numVectors mismatch: %d != %d", numVectors, index2.NumVectors())
	}

	if index.state.NumSubspaces != index2.state.NumSubspaces {
		t.Fatalf("numSubspaces mismatch: %d != %d", index.state.NumSubspaces, index2.state.NumSubspaces)
	}

	if index.state.NumSubFeatures != index2.state.NumSubFeatures {
		t.Fatalf("numSubFeatures mismatch: %d != %d", index.state.NumSubFeatures, index2.state.NumSubFeatures)
	}

	if index.state.NumClusters != index2.state.NumClusters {
		t.Fatalf("numClusters mismatch: %d != %d", index.state.NumClusters, index2.state.NumClusters)
	}

	if index.state.Config.MaxIterations != index2.state.Config.MaxIterations {
		t.Fatalf("numIterations mismatch: %d != %d", index.state.Config.MaxIterations, index2.state.Config.MaxIterations)
	}

	if index.state.Config.Tolerance != index2.state.Config.Tolerance {
		t.Fatalf("tol mismatch: %f != %f", index.state.Config.Tolerance, index2.state.Config.Tolerance)
	}

	for i := range index.state.Codebooks {
		for j := range index.state.Codebooks[i] {
			for k := range index.state.Codebooks[i][j] {
				if index.state.Codebooks[i][j][k] != index2.state.Codebooks[i][j][k] {
					t.Fatalf("codebook mismatch: %v != %v", index.state.Codebooks[i][j][k], index2.state.Codebooks[i][j][k])
				}
			}
		}
	}

	for i := range index.state.Codes {
		if index.state.Codes[i] != index2.state.Codes[i] {
			t.Fatalf("code mismatch: %v != %v", index.state.Codes[i], index2.state.Codes[i])
		}
	}

	for i := range index.clusters {
		centroids1 := index.clusters[i].Centroids()
		centroids2 := index2.clusters[i].Centroids()
		for j := range centroids1 {
			for k := range centroids1[j] {
				if centroids1[j][k] != centroids2[j][k] {
					t.Fatalf("centroid mismatch: %v != %v", centroids1[j][k], centroids2[j][k])
				}
			}
		}
	}
}
