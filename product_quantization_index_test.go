package annindex

import (
	"testing"
)

func TestProductQuantizationIndexAdd(t *testing.T) {
	numFeatures := 4
	numSubspaces := 2
	numClusters := 1
	index, err := NewProductQuantizationIndex(numFeatures, numSubspaces, numClusters)
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
