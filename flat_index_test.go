package annindex

import (
	"testing"
)

func TestFlatIndexAdd(t *testing.T) {
	index, _ := NewFlatIndex(2)
	index.Add([]float64{1, 2, 3, 4})
	index.Add([]float64{5, 6})

	results, err := index.Search([]float64{1, 2, 3, 4, 5, 6}, 1)
	if err != nil {
		t.Fatalf("error: %v", err)
	}
	for i, result := range results {
		if result[0] != i {
			t.Fatalf("result[%d] = %d, expected %d", i, result[0], i)
		}
	}
}
