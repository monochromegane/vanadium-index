package annindex

import (
	"github.com/monochromegane/kmeans"
)

type ProductQuantizationIndex struct {
	numFeatures   int
	numSubspaces  int
	numIterations int
	tol           float64
	subspaces     []kmeans.KMeans
	subDataset    [][]float64
	codebooks     [][][]float64
	codes         [][]int
}

func NewProductQuantizationIndex(numFeatures, numSubspaces, numClusters, numIterations int, tol float64) (*ProductQuantizationIndex, error) {
	if numFeatures <= 0 {
		return nil, ErrInvalidNumFeatures
	}
	if numSubspaces <= 0 || numSubspaces > numFeatures || numFeatures%numSubspaces != 0 {
		return nil, ErrInvalidNumSubspaces
	}
	subspaceSize := numFeatures / numSubspaces
	subspaces := make([]kmeans.KMeans, numSubspaces)
	for i := range numSubspaces {
		var err error
		var subspace kmeans.KMeans
		if subspaceSize > 256 {
			subspace, err = kmeans.NewLinearAlgebraKMeans(numClusters, subspaceSize, kmeans.INIT_KMEANS_PLUS_PLUS)
		} else {
			subspace, err = kmeans.NewNaiveKMeans(numClusters, subspaceSize, kmeans.INIT_RANDOM)
		}
		if err != nil {
			return nil, err
		}
		subspaces[i] = subspace
	}
	return &ProductQuantizationIndex{
		numFeatures:   numFeatures,
		numSubspaces:  numSubspaces,
		numIterations: numIterations,
		tol:           tol,
		subspaces:     subspaces,
		subDataset:    make([][]float64, numSubspaces),
		codebooks:     make([][][]float64, numSubspaces),
		codes:         make([][]int, numSubspaces),
	}, nil
}

func (index *ProductQuantizationIndex) Add(data []float64) error {
	if len(data) == 0 {
		return ErrEmptyData
	}

	if len(data)%index.numFeatures != 0 {
		return ErrInvalidDataLength
	}

	numVectors := len(data) / index.numFeatures
	subspaceSize := index.numFeatures / index.numSubspaces

	for i := range index.numSubspaces {
		for v := range numVectors {
			start := v*index.numFeatures + i*subspaceSize
			end := start + subspaceSize
			subvector := data[start:end]
			index.subDataset[i] = append(index.subDataset[i], subvector...)
		}
	}
	return nil
}

func (index *ProductQuantizationIndex) Search(query []float64, k int) ([][]int, error) {
	return nil, nil
}

func (index *ProductQuantizationIndex) Train(data []float64) error {
	dataSize := len(data) / index.numFeatures

	for i := range index.numSubspaces {
		_, _, err := index.subspaces[i].Train(index.subDataset[i], index.numIterations, index.tol)
		if err != nil {
			return err
		}

		centroids := index.subspaces[i].Centroids()
		index.codebooks[i] = centroids

		code := make([]int, dataSize)
		err = index.subspaces[i].Predict(index.subDataset[i], func(row int, minCol int, minVal float64) error {
			code[row] = minCol
			return nil
		})
		if err != nil {
			return err
		}
		index.codes[i] = code
	}
	return nil
}
