package annindex

import (
	"testing"
)

func TestProductQuantizationIndex(t *testing.T) {
	numFeatures := 4
	numSubspaces := 2
	numClusters := 4
	numIterations := 10
	tol := 0.001
	index, err := NewProductQuantizationIndex(numFeatures, numSubspaces, numClusters, numIterations, tol)
	if err != nil {
		t.Fatalf("Failed to create index: %v", err)
	}

	data := []float64{
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
