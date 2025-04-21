package vanadium_index

import (
	"encoding/gob"
)

type FlatIndex struct {
	state *FlatIndexState
}

type FlatIndexState struct {
	NumFeatures int
	Data        []float32
}

func newFlatIndex(numFeatures int) (*FlatIndex, error) {
	if numFeatures <= 0 {
		return nil, ErrInvalidNumFeatures
	}
	return &FlatIndex{
		state: &FlatIndexState{
			NumFeatures: numFeatures,
			Data:        make([]float32, 0),
		},
	}, nil
}

func loadFlatIndex(dec *gob.Decoder) (*FlatIndex, error) {
	index := &FlatIndex{}
	err := index.decode(dec)
	if err != nil {
		return nil, err
	}
	return index, nil
}

func (index *FlatIndex) Train(data []float32) error {
	return nil
}

func (index *FlatIndex) Add(data []float32) error {
	if len(data) == 0 {
		return ErrEmptyData
	}

	if len(data)%index.state.NumFeatures != 0 {
		return ErrInvalidDataLength
	}

	index.state.Data = append(index.state.Data, data...)
	return nil
}

func (index *FlatIndex) Search(query []float32, k int) ([][]int, [][]float32, error) {
	if k <= 0 {
		return nil, nil, ErrInvalidK
	}

	if len(query) == 0 {
		return nil, nil, ErrEmptyData
	}

	if len(query)%index.state.NumFeatures != 0 {
		return nil, nil, ErrInvalidDataLength
	}

	N := len(index.state.Data) / index.state.NumFeatures
	numQueries := len(query) / index.state.NumFeatures

	neighbors := make([]*SmallestK, numQueries)
	for q := range numQueries {
		neighbors[q] = NewSmallestK(k)
		for n := range N {
			subData := index.state.Data[n*index.state.NumFeatures : (n+1)*index.state.NumFeatures]
			subQuery := query[q*index.state.NumFeatures : (q+1)*index.state.NumFeatures]
			dist := index.squaredEuclideanDistance(subQuery, subData)
			neighbors[q].Push(n, dist)
		}
	}

	results := make([][]int, numQueries)
	distances := make([][]float32, numQueries)
	for q := range numQueries {
		results[q] = make([]int, k)
		distances[q] = make([]float32, k)
		for i := range k {
			results[q][i] = neighbors[q].SmallestK()[i].index
			distances[q][i] = neighbors[q].SmallestK()[i].value
		}
	}

	return results, distances, nil
}

func (index *FlatIndex) NumVectors() int {
	return len(index.state.Data) / index.state.NumFeatures
}

func (index *FlatIndex) Save(enc *gob.Encoder) error {
	meta := MetaData{
		IndexType: IndexTypeFlat,
		CodeType1: CodeTypeNameNone,
		CodeType2: CodeTypeNameNone,
	}
	err := enc.Encode(meta)
	if err != nil {
		return err
	}
	return index.encode(enc)
}

func (index *FlatIndex) encode(enc *gob.Encoder) error {
	return enc.Encode(index.state)
}

func (index *FlatIndex) decode(dec *gob.Decoder) error {
	index.state = &FlatIndexState{}
	return dec.Decode(index.state)
}

func (index *FlatIndex) squaredEuclideanDistance(x, y []float32) float32 {
	distance := float32(0)
	for i := range x {
		diff := x[i] - y[i]
		distance += diff * diff
	}
	return distance
}
