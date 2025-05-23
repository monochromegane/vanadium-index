package vanadium_index

import (
	"bytes"
	"encoding/gob"
	"testing"
)

func TestInvertedFileIndex(t *testing.T) {
	numFeatures := 4
	numClusters := uint8(4)
	numIterations := 10
	tol := float32(0.001)
	index, err := newInvertedFileFlatIndex(
		numFeatures,
		numClusters,
		WithIVFMaxIterations(numIterations),
		WithIVFTolerance(tol),
	)
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

	results, distances, err := index.Search(query, 1)
	if err != nil {
		t.Fatalf("Failed to search index: %v", err)
	}

	for i, result := range results {
		if result[0] != i {
			t.Fatalf("result[%d] = %d, expected %d", i, result[0], i)
		}
	}

	for i, distance := range distances {
		if distance[0] != 0 {
			t.Fatalf("distance[%d] = %f, expected 0", i, distance[0])
		}
	}
}

func TestInvertedFileIndexSaveLoad(t *testing.T) {
	numFeatures := 4
	numClusters := uint8(4)
	numIterations := 10
	tol := float32(0.001)
	index, err := newInvertedFileFlatIndex(
		numFeatures,
		numClusters,
		WithIVFMaxIterations(numIterations),
		WithIVFTolerance(tol),
	)
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
		t.Fatalf("Failed to encode index: %v", err)
	}

	dec := gob.NewDecoder(&buf)
	annIndex, err := LoadIndex(dec)
	if err != nil {
		t.Fatalf("error: %v", err)
	}
	index2, ok := annIndex.(*InvertedFileIndex[uint8, uint8])
	if !ok {
		t.Fatalf("loaded index is not a InvertedFileIndex")
	}

	if index2.state.NumFeatures != index.state.NumFeatures {
		t.Fatalf("index2.state.NumFeatures = %d, expected %d", index2.state.NumFeatures, index.state.NumFeatures)
	}

	if index2.state.NumClusters != index.state.NumClusters {
		t.Fatalf("index2.state.NumClusters = %d, expected %d", index2.state.NumClusters, index.state.NumClusters)
	}

	if index2.state.IsTrained != index.state.IsTrained {
		t.Fatalf("index2.state.IsTrained = %t, expected %t", index2.state.IsTrained, index.state.IsTrained)
	}

	if index2.state.ShouldTrainIndexes != index.state.ShouldTrainIndexes {
		t.Fatalf("index2.state.ShouldTrainIndexes = %t, expected %t", index2.state.ShouldTrainIndexes, index.state.ShouldTrainIndexes)
	}

	if index2.state.Config.MaxIterations != index.state.Config.MaxIterations {
		t.Fatalf("index2.state.Config.MaxIterations = %d, expected %d", index2.state.Config.MaxIterations, index.state.Config.MaxIterations)
	}

	if index2.state.Config.Tolerance != index.state.Config.Tolerance {
		t.Fatalf("index2.state.Config.Tolerance = %f, expected %f", index2.state.Config.Tolerance, index.state.Config.Tolerance)
	}

	for i := range index.state.Mapping {
		for j := range index.state.Mapping[i] {
			if index2.state.Mapping[i][j] != index.state.Mapping[i][j] {
				t.Fatalf("index2.state.Mapping[%d][%d] = %d, expected %d", i, j, index2.state.Mapping[i][j], index.state.Mapping[i][j])
			}
		}
	}

	centroids1 := index.cluster.Centroids()
	centroids2 := index2.cluster.Centroids()
	for i := range centroids1 {
		for j := range centroids1[i] {
			if centroids1[i][j] != centroids2[i][j] {
				t.Fatalf("centroids mismatch: %v != %v", centroids1[i][j], centroids2[i][j])
			}
		}
	}

	for i := range index.indexes {
		if index.indexes[i].NumVectors() != index2.indexes[i].NumVectors() {
			t.Fatalf("index2.indexes[%d].NumVectors() = %d, expected %d", i, index2.indexes[i].NumVectors(), index.indexes[i].NumVectors())
		}
	}
}

func TestInvertedFileIndexWithPQIndex(t *testing.T) {
	numFeatures := 4
	numIvfClusters := uint8(4)
	numPqClusters := uint8(1)
	numIterations := 10
	tol := float32(0.001)

	numSubspaces := 1
	index, err := newInvertedFilePQIndex(
		numFeatures,
		numIvfClusters,
		numSubspaces,
		numPqClusters,
		WithIVFMaxIterations(numIterations),
		WithIVFTolerance(tol),
		WithIVFPQIndex(
			WithPQMaxIterations(numIterations),
			WithPQTolerance(tol),
		),
	)
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

	results, distances, err := index.Search(query, 1)
	if err != nil {
		t.Fatalf("Failed to search index: %v", err)
	}

	for i, result := range results {
		if result[0] != i {
			t.Fatalf("result[%d] = %d, expected %d", i, result[0], i)
		}
	}

	for i, distance := range distances {
		if distance[0] != 0 {
			t.Fatalf("distance[%d] = %f, expected 0", i, distance[0])
		}
	}
}

func TestInvertedFileIndexWithPQIndexSaveLoad(t *testing.T) {
	numFeatures := 4
	numIvfClusters := uint8(4)
	numPqClusters := uint8(1)
	numIterations := 10
	tol := float32(0.001)

	numSubspaces := 1
	index, err := newInvertedFilePQIndex(
		numFeatures,
		numIvfClusters,
		numSubspaces,
		numPqClusters,
		WithIVFMaxIterations(numIterations),
		WithIVFTolerance(tol),
		WithIVFPQIndex(
			WithPQMaxIterations(numIterations),
			WithPQTolerance(tol),
		),
	)
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
		t.Fatalf("Failed to encode index: %v", err)
	}

	dec := gob.NewDecoder(&buf)
	annIndex, err := LoadIndex(dec)
	if err != nil {
		t.Fatalf("error: %v", err)
	}
	index2, ok := annIndex.(*InvertedFileIndex[uint8, uint8])
	if !ok {
		t.Fatalf("loaded index is not a InvertedFileIndex")
	}

	if index2.state.NumFeatures != index.state.NumFeatures {
		t.Fatalf("index2.state.NumFeatures = %d, expected %d", index2.state.NumFeatures, index.state.NumFeatures)
	}

	if index2.state.NumClusters != index.state.NumClusters {
		t.Fatalf("index2.state.NumClusters = %d, expected %d", index2.state.NumClusters, index.state.NumClusters)
	}

	if index2.state.IsTrained != index.state.IsTrained {
		t.Fatalf("index2.state.IsTrained = %t, expected %t", index2.state.IsTrained, index.state.IsTrained)
	}

	if index2.state.ShouldTrainIndexes != index.state.ShouldTrainIndexes {
		t.Fatalf("index2.state.ShouldTrainIndexes = %t, expected %t", index2.state.ShouldTrainIndexes, index.state.ShouldTrainIndexes)
	}

	if index2.state.Config.MaxIterations != index.state.Config.MaxIterations {
		t.Fatalf("index2.state.Config.MaxIterations = %d, expected %d", index2.state.Config.MaxIterations, index.state.Config.MaxIterations)
	}

	if index2.state.Config.Tolerance != index.state.Config.Tolerance {
		t.Fatalf("index2.state.Config.Tolerance = %f, expected %f", index2.state.Config.Tolerance, index.state.Config.Tolerance)
	}

	for i := range index.state.Mapping {
		for j := range index.state.Mapping[i] {
			if index2.state.Mapping[i][j] != index.state.Mapping[i][j] {
				t.Fatalf("index2.state.Mapping[%d][%d] = %d, expected %d", i, j, index2.state.Mapping[i][j], index.state.Mapping[i][j])
			}
		}
	}

	centroids1 := index.cluster.Centroids()
	centroids2 := index2.cluster.Centroids()
	for i := range centroids1 {
		for j := range centroids1[i] {
			if centroids1[i][j] != centroids2[i][j] {
				t.Fatalf("centroids mismatch: %v != %v", centroids1[i][j], centroids2[i][j])
			}
		}
	}

	for i := range index.indexes {
		if index.indexes[i].NumVectors() != index2.indexes[i].NumVectors() {
			t.Fatalf("index2.indexes[%d].NumVectors() = %d, expected %d", i, index2.indexes[i].NumVectors(), index.indexes[i].NumVectors())
		}
	}
}
