package annindex

import (
	"runtime"

	"github.com/monochromegane/kmeans"
	"golang.org/x/sync/errgroup"
)

type InvertedFileIndex[T1, T2 CodeType] struct {
	numFeatures int
	numClusters T1
	config      *InvertedFileIndexConfig
	isTrained   bool
	cluster     kmeans.KMeans
	indexes     []ANNIndex
	mapping     [][]int
}

type InvertedFileIndexConfig struct {
	numIterations int
	tol           float64
	pqOptions     []ProductQuantizationIndexOption
}

func NewInvertedFileFlatIndex[T CodeType](
	numFeatures int,
	numClusters T,
	opts ...InvertedFileIndexOption,
) (*InvertedFileIndex[T, T], error) {
	index := &InvertedFileIndex[T, T]{
		numFeatures: numFeatures,
		isTrained:   false,
		numClusters: numClusters,

		// Default values
		config: &InvertedFileIndexConfig{
			numIterations: 100,
			tol:           1e-6,
		},
	}
	for _, opt := range opts {
		err := opt(index.config)
		if err != nil {
			return nil, err
		}
	}
	indexBuilder := &subFlatIndexBuilder{}
	return newInvertedFileIndex(index, indexBuilder)
}

func NewInvertedFilePQIndex[T1, T2 CodeType](
	numFeatures int,
	numIvfClusters T1,
	numPqSubspaces int,
	numPqClusters T2,
	opts ...InvertedFileIndexOption,
) (*InvertedFileIndex[T1, T2], error) {
	index := &InvertedFileIndex[T1, T2]{
		numFeatures: numFeatures,
		isTrained:   false,
		numClusters: numIvfClusters,

		// Default values
		config: &InvertedFileIndexConfig{
			numIterations: 100,
			tol:           1e-6,
		},
	}
	for _, opt := range opts {
		err := opt(index.config)
		if err != nil {
			return nil, err
		}
	}
	indexBuilder := &subPQIndexBuilder[T2]{
		numSubspaces: numPqSubspaces,
		numClusters:  numPqClusters,
		opts:         index.config.pqOptions,
	}
	return newInvertedFileIndex(index, indexBuilder)
}

func newInvertedFileIndex[T1, T2 CodeType](
	index *InvertedFileIndex[T1, T2],
	indexBuilder subIndexBuilder,
) (*InvertedFileIndex[T1, T2], error) {
	var err error
	if index.numFeatures > 256 {
		index.cluster, err = kmeans.NewLinearAlgebraKMeans(int(index.numClusters), index.numFeatures, kmeans.INIT_KMEANS_PLUS_PLUS)
	} else {
		index.cluster, err = kmeans.NewNaiveKMeans(int(index.numClusters), index.numFeatures, kmeans.INIT_RANDOM)
	}
	if err != nil {
		return nil, err
	}

	index.indexes = make([]ANNIndex, index.numClusters)
	for c := range int(index.numClusters) {
		subIndex, err := indexBuilder.build(index.numFeatures)
		if err != nil {
			return nil, err
		}
		index.indexes[c] = subIndex
	}
	index.mapping = make([][]int, index.numClusters)
	return index, nil
}

func (index *InvertedFileIndex[T1, T2]) Train(data []float64) error {
	if len(data) == 0 {
		return ErrEmptyData
	}

	if len(data)%index.numFeatures != 0 {
		return ErrInvalidDataLength
	}

	_, _, err := index.cluster.Train(data, index.config.numIterations, index.config.tol)
	if err != nil {
		return err
	}

	numVectors := len(data) / index.numFeatures

	code := make([]T1, numVectors)
	numElements := make([]int, int(index.numClusters))
	err = index.cluster.Predict(data, func(row int, minCol int, minVal float64) error {
		code[row] = T1(minCol)
		numElements[minCol] += 1
		return nil
	})
	if err != nil {
		return err
	}

	if index.config.pqOptions == nil {
		index.isTrained = true
		return nil
	}

	var eg errgroup.Group
	eg.SetLimit(runtime.NumCPU())

	for c := range int(index.numClusters) {
		eg.Go(func() error {
			countClusterData := 0
			clusterData := make([]float64, numElements[c]*index.numFeatures)
			for v := range numVectors {
				if code[v] != T1(c) {
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

func (index *InvertedFileIndex[T1, T2]) Add(data []float64) error {
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
		index.mapping[minCol] = append(index.mapping[minCol], ivfRow+row)
		return index.indexes[minCol].Add(rowData)
	})
	if err != nil {
		return err
	}

	return nil
}

func (index *InvertedFileIndex[T1, T2]) Search(query []float64, k int) ([][]int, error) {
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

func (index *InvertedFileIndex[T1, T2]) NumVectors() int {
	numVectors := 0
	for _, index := range index.indexes {
		numVectors += index.NumVectors()
	}
	return numVectors
}

type subFlatIndexBuilder struct{}

func (b *subFlatIndexBuilder) build(numFeatures int) (ANNIndex, error) {
	return NewFlatIndex(numFeatures)
}

type subPQIndexBuilder[T CodeType] struct {
	numSubspaces int
	numClusters  T
	opts         []ProductQuantizationIndexOption
}

func (b *subPQIndexBuilder[T]) build(numFeatures int) (ANNIndex, error) {
	return NewProductQuantizationIndex(numFeatures, b.numSubspaces, b.numClusters, b.opts...)
}
