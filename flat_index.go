package annindex

import (
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

func (index *FlatIndex) Train(data []float64) error {
	return nil
}

func (index *FlatIndex) Add(data []float64) error {
	if len(data) == 0 {
		return ErrEmptyData
	}

	if len(data)%index.state.NumFeatures != 0 {
		return ErrInvalidDataLength
	}

	if index.state.Data == nil {
		numRows := len(data) / index.state.NumFeatures
		index.state.Data = mat.NewDense(numRows, index.state.NumFeatures, data)
	} else {
		dataRows, _ := index.state.Data.Dims()
		numRows := len(data) / index.state.NumFeatures
		newData := mat.NewDense(dataRows+numRows, index.state.NumFeatures, nil)
		newData.Stack(index.state.Data, mat.NewDense(numRows, index.state.NumFeatures, data))
		index.state.Data = newData
	}
	index.state.Xnorm = normVec(index.state.Data)
	return nil
}

func (index *FlatIndex) Search(query []float64, k int) ([][]int, error) {
	if k <= 0 {
		return nil, ErrInvalidK
	}

	if len(query) == 0 {
		return nil, ErrEmptyData
	}

	if len(query)%index.state.NumFeatures != 0 {
		return nil, ErrInvalidDataLength
	}

	N, _ := index.state.Data.Dims()
	numQueries := len(query) / index.state.NumFeatures

	XX := tile(N, numQueries, index.state.Xnorm)
	Q := mat.NewDense(numQueries, index.state.NumFeatures, query)
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
			neighbors[q].Push(n, dist)
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
