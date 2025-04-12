package annindex

import (
	"testing"
)

func TestSmallestK(t *testing.T) {
	h := NewSmallestK(3)
	indices := []int{0, 1, 2}
	values := []float32{0.1, 0.3, 0.2}
	for i, index := range indices {
		h.Push(index, values[i])
	}
	if h.maxHeap.Len() != 3 {
		t.Fatalf("heap length is not 3")
	}

	smallestK := h.SmallestK()
	if smallestK[0].index != 0 {
		t.Fatalf("smallestK[0].index is not 0")
	}
	if smallestK[1].index != 2 {
		t.Fatalf("smallestK[1].index is not 2")
	}
	if smallestK[2].index != 1 {
		t.Fatalf("smallestK[2].index is not 1")
	}
}

func TestSmallestKOverK(t *testing.T) {
	h := NewSmallestK(3)
	indices := []int{0, 1, 2, 3}
	values := []float32{0.1, 0.3, 0.2, 0.0}
	for i, index := range indices {
		h.Push(index, values[i])
	}
	smallestK := h.SmallestK()
	if len(smallestK) != 3 {
		t.Fatalf("smallestK length is not 3")
	}
	if smallestK[0].index != 3 {
		t.Fatalf("smallestK[0].index is not 3")
	}
	if smallestK[1].index != 0 {
		t.Fatalf("smallestK[1].index is not 0")
	}
	if smallestK[2].index != 2 {
		t.Fatalf("smallestK[2].index is not 2")
	}
}
