package annindex

import (
	"testing"
)

func TestProductQuantizationIndexAdd(t *testing.T) {
	numFeatures := 4
	numSubspaces := 2
	numClusters := 1
	numIterations := 10
	tol := 0.001
	index, err := NewProductQuantizationIndex(numFeatures, numSubspaces, numClusters, numIterations, tol)
	if err != nil {
		t.Fatalf("Failed to create index: %v", err)
	}

	dataSet := [][]float64{
		{
			1, 2, 3, 4,
			5, 6, 7, 8,
		},
		{
			9, 10, 11, 12,
		},
	}

	for _, data := range dataSet {
		err = index.Add(data)
		if err != nil {
			t.Fatalf("Failed to add data: %v", err)
		}
	}

	expectedSubDataset := [][]float64{
		{
			1, 2,
			5, 6,
			9, 10,
		},
		{
			3, 4,
			7, 8,
			11, 12,
		},
	}

	for i := range index.subDataset {
		subDataset := index.subDataset[i]
		if len(subDataset) != len(expectedSubDataset[i]) {
			t.Fatalf("Subdataset length does not match expected value")
		}
		for j := range subDataset {
			if subDataset[j] != expectedSubDataset[i][j] {
				t.Fatalf("Subdataset does not match expected value")
			}
		}
	}
}

func TestProductQuantizationIndexTrain(t *testing.T) {
	numFeatures := 4
	numSubspaces := 2
	numClusters := 1
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

	err = index.Add(data)
	if err != nil {
		t.Fatalf("Failed to add data: %v", err)
	}

	err = index.Train(data)
	if err != nil {
		t.Fatalf("Failed to train index: %v", err)
	}

}
