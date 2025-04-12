package annindex

import (
	"container/heap"
	"sort"
)

type heapItem struct {
	index int
	value float32
}

type MaxHeap []heapItem

func (h MaxHeap) Len() int { return len(h) }

func (h MaxHeap) Less(i, j int) bool {
	return h[i].value > h[j].value
}
func (h MaxHeap) Swap(i, j int) { h[i], h[j] = h[j], h[i] }

func (h *MaxHeap) Push(x any) {
	*h = append(*h, x.(heapItem))
}

func (h *MaxHeap) Pop() any {
	old := *h
	n := len(old)
	item := old[n-1]
	*h = old[0 : n-1]
	return item
}

type SmallestK struct {
	maxHeap *MaxHeap
	k       int
}

func NewSmallestK(k int) *SmallestK {
	h := &MaxHeap{}
	heap.Init(h)
	return &SmallestK{
		maxHeap: h,
		k:       k,
	}
}

func (s *SmallestK) Push(index int, value float32) {
	if s.maxHeap.Len() < s.k {
		heap.Push(s.maxHeap, heapItem{index: index, value: value})
	} else if (*s.maxHeap)[0].value > value {
		heap.Pop(s.maxHeap)
		heap.Push(s.maxHeap, heapItem{index: index, value: value})
	}
}

func (s *SmallestK) SmallestK() []heapItem {
	result := make([]heapItem, s.maxHeap.Len())
	copy(result, *s.maxHeap)
	sort.Slice(result, func(i, j int) bool {
		return result[i].value < result[j].value
	})
	return result
}
