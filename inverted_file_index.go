package annindex

import (
	"runtime"

	"github.com/monochromegane/kmeans"
	"golang.org/x/sync/errgroup"
)

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
	if len(data) == 0 {
		return ErrEmptyData
	}

	if len(data)%index.numFeatures != 0 {
		return ErrInvalidDataLength
	}

	_, _, err := index.cluster.Train(data, index.numIterations, index.tol)
	if err != nil {
		return err
	}

	numVectors := len(data) / index.numFeatures

	code := make([]int, numVectors)
	numElements := make([]int, index.numClusters)
	err = index.cluster.Predict(data, func(row int, minCol int, minVal float64) error {
		code[row] = minCol
		numElements[minCol] += 1
		return nil
	})
	if err != nil {
		return err
	}

	if index.pqOptions == nil {
		return nil
	}

	var eg errgroup.Group
	eg.SetLimit(runtime.NumCPU())

	for c := range index.numClusters {
		eg.Go(func() error {
			countClusterData := 0
			clusterData := make([]float64, numElements[c]*index.numFeatures)
			for v := range numVectors {
				if code[v] != c {
					continue
				}
				start := v * index.numFeatures
				end := start + index.numFeatures
				copy(clusterData[countClusterData*index.numFeatures:], data[start:end])
				countClusterData += 1
			}

			return index.indexes[c].Train(clusterData)
		})
	}
	if err := eg.Wait(); err != nil {
		return err
	}

	return nil
}

func (index *InvertedFileIndex) Add(data []float64) error {
	return nil
}

func (index *InvertedFileIndex) Search(query []float64, k int) ([][]int, error) {
	return nil, nil
}
