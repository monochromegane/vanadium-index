package annindex

import "github.com/monochromegane/kmeans"

type InvertedFileIndex struct {
	numFeatures    int
	numClusters    int
	numIterations  int
	tol            float64
	cluster        kmeans.KMeans
	indexes        []ANNIndex
	pqNumSubspaces int
	pqOptions      []ProductQuantizationIndexOption
}

func NewInvertedFileIndex(numFeatures int, opts ...InvertedFileIndexOption) (*InvertedFileIndex, error) {
	index := &InvertedFileIndex{
		numFeatures: numFeatures,

		// Default values
		numClusters:   256,
		numIterations: 100,
		tol:           1e-6,
	}
	for _, opt := range opts {
		err := opt(index)
		if err != nil {
			return nil, err
		}
	}

	var err error
	if numFeatures > 256 {
		index.cluster, err = kmeans.NewLinearAlgebraKMeans(index.numClusters, numFeatures, kmeans.INIT_KMEANS_PLUS_PLUS)
	} else {
		index.cluster, err = kmeans.NewNaiveKMeans(index.numClusters, numFeatures, kmeans.INIT_RANDOM)
	}
	if err != nil {
		return nil, err
	}

	index.indexes = make([]ANNIndex, index.numClusters)
	for c := range index.numClusters {
		var err error
		if index.pqOptions == nil {
			index.indexes[c], err = NewFlatIndex(numFeatures)
			if err != nil {
				return nil, err
			}
		} else {
			index.indexes[c], err = NewProductQuantizationIndex(numFeatures, index.pqNumSubspaces, index.pqOptions...)
			if err != nil {
				return nil, err
			}
		}
	}
	return index, nil
}

func (index *InvertedFileIndex) Train(data []float64) error {
	return nil
}

func (index *InvertedFileIndex) Add(data []float64) error {
	return nil
}

func (index *InvertedFileIndex) Search(query []float64, k int) ([][]int, error) {
	return nil, nil
}
