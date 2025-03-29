package annindex

import (
	"github.com/monochromegane/kmeans"
)

type ProductQuantizationIndex struct {
	numFeatures  int
	numSubspaces int
	subspaces    []kmeans.KMeans
}

func NewProductQuantizationIndex(numFeatures, numSubspaces, numClusters int) (*ProductQuantizationIndex, error) {
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
		numFeatures:  numFeatures,
		numSubspaces: numSubspaces,
		subspaces:    subspaces,
	}, nil
}

func (index *ProductQuantizationIndex) Add(data []float64) error {
	return nil
}

func (index *ProductQuantizationIndex) Search(query []float64, k int) ([][]int, error) {
	return nil, nil
}

func (index *ProductQuantizationIndex) Train(data []float64) error {
	return nil
}
