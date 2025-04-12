package annindex

import (
	"encoding/gob"

	"gonum.org/v1/gonum/mat"
)

type FlatIndex struct {
	state *FlatIndexState
}

type FlatIndexState struct {
	NumFeatures int
	Data        *mat.Dense
	Xnorm       *mat.VecDense
}

func NewFlatIndex(numFeatures int) (*FlatIndex, error) {
	if numFeatures <= 0 {
		return nil, ErrInvalidNumFeatures
	}
	return &FlatIndex{
		state: &FlatIndexState{
			NumFeatures: numFeatures,
		},
	}, nil
}

func LoadFlatIndex(dec *gob.Decoder) (*FlatIndex, error) {
	index := &FlatIndex{}
	err := index.Decode(dec)
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
	data64 := make([]float64, len(data))
	for i, v := range data {
		data64[i] = float64(v)
	}

	if index.state.Data == nil {
		numRows := len(data) / index.state.NumFeatures
		index.state.Data = mat.NewDense(numRows, index.state.NumFeatures, data64)
	} else {
		dataRows, _ := index.state.Data.Dims()
		numRows := len(data) / index.state.NumFeatures
		newData := mat.NewDense(dataRows+numRows, index.state.NumFeatures, nil)
		newData.Stack(index.state.Data, mat.NewDense(numRows, index.state.NumFeatures, data64))
		index.state.Data = newData
	}
	index.state.Xnorm = normVec(index.state.Data)
	return nil
}

func (index *FlatIndex) Search(query []float32, k int) ([][]int, error) {
	if k <= 0 {
		return nil, ErrInvalidK
	}

	if len(query) == 0 {
		return nil, ErrEmptyData
	}

	if len(query)%index.state.NumFeatures != 0 {
		return nil, ErrInvalidDataLength
	}

	query64 := make([]float64, len(query))
	for i, v := range query {
		query64[i] = float64(v)
	}

	N, _ := index.state.Data.Dims()
	numQueries := len(query) / index.state.NumFeatures

	XX := tile(N, numQueries, index.state.Xnorm)
	Q := mat.NewDense(numQueries, index.state.NumFeatures, query64)
	qNorm := normVec(Q)
	QQ := tile(numQueries, N, qNorm)

	XQT := mat.NewDense(N, numQueries, nil)
	XQT.Mul(index.state.Data, Q.T())
	XQT.Scale(-2, XQT)

	XQT.Add(XQT, XX)
	XQT.Add(XQT, QQ.T())

	neighbors := make([]*SmallestK, numQueries)
	for q := range numQueries {
		neighbors[q] = NewSmallestK(k)
	}
	for n := range N {
		for q := range numQueries {
			dist := XQT.At(n, q)
			neighbors[q].Push(n, float32(dist))
		}
	}

	results := make([][]int, numQueries)
	for q := range numQueries {
		results[q] = make([]int, k)
		for i := range k {
			results[q][i] = neighbors[q].SmallestK()[i].index
		}
	}

	return results, nil
}

func (index *FlatIndex) NumVectors() int {
	if index.state.Data == nil {
		return 0
	}
	rows, _ := index.state.Data.Dims()
	return rows
}

func (index *FlatIndex) Encode(enc *gob.Encoder) error {
	return enc.Encode(index.state)
}

func (index *FlatIndex) Decode(dec *gob.Decoder) error {
	index.state = &FlatIndexState{}
	return dec.Decode(index.state)
}

func normVec(X *mat.Dense) *mat.VecDense {
	N, _ := X.Dims()
	normVec := mat.NewVecDense(N, nil)
	for i := range N {
		normVec.SetVec(i, mat.Norm(X.RowView(i), 2))
	}
	normVec.MulElemVec(normVec, normVec)
	return normVec
}

func tile(r, c int, x *mat.VecDense) *mat.Dense {
	X := mat.NewDense(r, c, nil)
	for i := range c {
		X.SetCol(i, x.RawVector().Data)
	}
	return X
}
