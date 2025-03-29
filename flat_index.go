package annindex

import (
	"gonum.org/v1/gonum/mat"
)

type FlatIndex struct {
	numFeatures int
	data        *mat.Dense
	xNorm       *mat.VecDense
}

func NewFlatIndex(numFeatures int) (*FlatIndex, error) {
	if numFeatures <= 0 {
		return nil, ErrInvalidNumFeatures
	}
	return &FlatIndex{numFeatures: numFeatures}, nil
}

func (index *FlatIndex) Add(data []float64) error {
	if len(data) == 0 {
		return ErrEmptyData
	}

	if len(data)%index.numFeatures != 0 {
		return ErrInvalidDataLength
	}

	if index.data == nil {
		numRows := len(data) / index.numFeatures
		index.data = mat.NewDense(numRows, index.numFeatures, data)
	} else {
		dataRows, _ := index.data.Dims()
		numRows := len(data) / index.numFeatures
		newData := mat.NewDense(dataRows+numRows, index.numFeatures, nil)
		newData.Stack(index.data, mat.NewDense(numRows, index.numFeatures, data))
		index.data = newData
	}
	index.xNorm = normVec(index.data)
	return nil
}

func (index *FlatIndex) Search(query []float64, k int) ([][]int, error) {
	if k <= 0 {
		return nil, ErrInvalidK
	}

	if len(query) == 0 {
		return nil, ErrEmptyData
	}

	if len(query)%index.numFeatures != 0 {
		return nil, ErrInvalidDataLength
	}

	N, _ := index.data.Dims()
	numQueries := len(query) / index.numFeatures

	XX := tile(N, numQueries, index.xNorm)
	Q := mat.NewDense(numQueries, index.numFeatures, query)
	qNorm := normVec(Q)
	QQ := tile(numQueries, N, qNorm)

	XQT := mat.NewDense(N, numQueries, nil)
	XQT.Mul(index.data, Q.T())
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

func (index *FlatIndex) Train(data []float64) error {
	return nil
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
