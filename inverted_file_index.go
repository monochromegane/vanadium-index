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
	isTrained      bool
	cluster        kmeans.KMeans
	indexes        []ANNIndex
	mapping        [][]int
	pqNumSubspaces int
	pqOptions      []ProductQuantizationIndexOption
}

func NewInvertedFileIndex(numFeatures int, opts ...InvertedFileIndexOption) (*InvertedFileIndex, error) {
	index := &InvertedFileIndex{
		numFeatures: numFeatures,
		isTrained:   false,

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
	index.mapping = make([][]int, index.numClusters)
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
		index.isTrained = true
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
	index.isTrained = true

	return nil
}

func (index *InvertedFileIndex) Add(data []float64) error {
	if len(data) == 0 {
		return ErrEmptyData
	}

	if len(data)%index.numFeatures != 0 {
		return ErrInvalidDataLength
	}

	if !index.isTrained {
		return ErrNotTrained
	}

	ivfRow := index.NumVectors()
	err := index.cluster.Predict(data, func(row int, minCol int, minVal float64) error {
		rowData := data[row*index.numFeatures : (row+1)*index.numFeatures]
		index.mapping[minCol] = append(index.mapping[minCol], ivfRow)
		return index.indexes[minCol].Add(rowData)
	})
	if err != nil {
		return err
	}

	return nil
}

func (index *InvertedFileIndex) Search(query []float64, k int) ([][]int, error) {
	if k <= 0 {
		return nil, ErrInvalidK
	}

	if len(query) == 0 {
		return nil, ErrEmptyData
	}

	if len(query)%index.numFeatures != 0 {
		return nil, ErrInvalidDataLength
	}

	if !index.isTrained {
		return nil, ErrNotTrained
	}

	numQueries := len(query) / index.numFeatures
	results := make([][]int, numQueries)
	err := index.cluster.Predict(query, func(row int, minCol int, minVal float64) error {
		rowQuery := query[row*index.numFeatures : (row+1)*index.numFeatures]
		result, err := index.indexes[minCol].Search(rowQuery, k)
		if err != nil {
			return err
		}
		results[row] = make([]int, len(result[0]))
		for i, r := range result[0] {
			results[row][i] = index.mapping[minCol][r]
		}
		return nil
	})
	if err != nil {
		return nil, err
	}

	return results, nil
}

func (index *InvertedFileIndex) NumVectors() int {
	numVectors := 0
	for _, index := range index.indexes {
		numVectors += index.NumVectors()
	}
	return numVectors
}
