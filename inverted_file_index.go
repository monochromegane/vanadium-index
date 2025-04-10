package annindex

import (
	"runtime"

	"github.com/monochromegane/kmeans"
	"golang.org/x/sync/errgroup"
)

type InvertedFileIndex[T1, T2 CodeType] struct {
	state   *InvertedFileIndexState[T1, T2]
	cluster kmeans.KMeans
	indexes []ANNIndex
}

type InvertedFileIndexState[T1, T2 CodeType] struct {
	NumFeatures int
	NumClusters T1
	IsTrained   bool
	Config      *InvertedFileIndexConfig
	Mapping     [][]int
}

type InvertedFileIndexConfig struct {
	NumIterations int
	Tol           float64
	PqOptions     []ProductQuantizationIndexOption
}

func NewInvertedFileFlatIndex[T CodeType](
	numFeatures int,
	numClusters T,
	opts ...InvertedFileIndexOption,
) (*InvertedFileIndex[T, T], error) {
	index := &InvertedFileIndex[T, T]{
		state: &InvertedFileIndexState[T, T]{
			NumFeatures: numFeatures,
			NumClusters: numClusters,
			IsTrained:   false,
			// Default values
			Config: &InvertedFileIndexConfig{
				NumIterations: 100,
				Tol:           1e-6,
			},
		},
	}
	for _, opt := range opts {
		err := opt(index.state.Config)
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
		state: &InvertedFileIndexState[T1, T2]{
			NumFeatures: numFeatures,
			NumClusters: numIvfClusters,
			IsTrained:   false,
			// Default values
			Config: &InvertedFileIndexConfig{
				NumIterations: 100,
				Tol:           1e-6,
			},
		},
	}
	for _, opt := range opts {
		err := opt(index.state.Config)
		if err != nil {
			return nil, err
		}
	}
	indexBuilder := &subPQIndexBuilder[T2]{
		numSubspaces: numPqSubspaces,
		numClusters:  numPqClusters,
		opts:         index.state.Config.PqOptions,
	}
	return newInvertedFileIndex(index, indexBuilder)
}

func newInvertedFileIndex[T1, T2 CodeType](
	index *InvertedFileIndex[T1, T2],
	indexBuilder subIndexBuilder,
) (*InvertedFileIndex[T1, T2], error) {
	var err error
	if index.state.NumFeatures > 256 {
		index.cluster, err = kmeans.NewLinearAlgebraKMeans(int(index.state.NumClusters), index.state.NumFeatures, kmeans.INIT_KMEANS_PLUS_PLUS)
	} else {
		index.cluster, err = kmeans.NewNaiveKMeans(int(index.state.NumClusters), index.state.NumFeatures, kmeans.INIT_RANDOM)
	}
	if err != nil {
		return nil, err
	}

	index.indexes = make([]ANNIndex, index.state.NumClusters)
	for c := range int(index.state.NumClusters) {
		subIndex, err := indexBuilder.build(index.state.NumFeatures)
		if err != nil {
			return nil, err
		}
		index.indexes[c] = subIndex
	}
	index.state.Mapping = make([][]int, index.state.NumClusters)
	return index, nil
}

func (index *InvertedFileIndex[T1, T2]) Train(data []float64) error {
	if len(data) == 0 {
		return ErrEmptyData
	}

	if len(data)%index.state.NumFeatures != 0 {
		return ErrInvalidDataLength
	}

	_, _, err := index.cluster.Train(data, index.state.Config.NumIterations, index.state.Config.Tol)
	if err != nil {
		return err
	}

	numVectors := len(data) / index.state.NumFeatures

	code := make([]T1, numVectors)
	numElements := make([]int, int(index.state.NumClusters))
	err = index.cluster.Predict(data, func(row int, minCol int, minVal float64) error {
		code[row] = T1(minCol)
		numElements[minCol] += 1
		return nil
	})
	if err != nil {
		return err
	}

	if index.state.Config.PqOptions == nil {
		index.state.IsTrained = true
		return nil
	}

	var eg errgroup.Group
	eg.SetLimit(runtime.NumCPU())

	for c := range int(index.state.NumClusters) {
		eg.Go(func() error {
			countClusterData := 0
			clusterData := make([]float64, numElements[c]*index.state.NumFeatures)
			for v := range numVectors {
				if code[v] != T1(c) {
					continue
				}
				start := v * index.state.NumFeatures
				end := start + index.state.NumFeatures
				copy(clusterData[countClusterData*index.state.NumFeatures:], data[start:end])
				countClusterData += 1
			}

			return index.indexes[c].Train(clusterData)
		})
	}
	if err := eg.Wait(); err != nil {
		return err
	}
	index.state.IsTrained = true

	return nil
}

func (index *InvertedFileIndex[T1, T2]) Add(data []float64) error {
	if len(data) == 0 {
		return ErrEmptyData
	}

	if len(data)%index.state.NumFeatures != 0 {
		return ErrInvalidDataLength
	}

	if !index.state.IsTrained {
		return ErrNotTrained
	}

	ivfRow := index.NumVectors()
	err := index.cluster.Predict(data, func(row int, minCol int, minVal float64) error {
		rowData := data[row*index.state.NumFeatures : (row+1)*index.state.NumFeatures]
		index.state.Mapping[minCol] = append(index.state.Mapping[minCol], ivfRow+row)
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

	if len(query)%index.state.NumFeatures != 0 {
		return nil, ErrInvalidDataLength
	}

	if !index.state.IsTrained {
		return nil, ErrNotTrained
	}

	numQueries := len(query) / index.state.NumFeatures
	results := make([][]int, numQueries)
	err := index.cluster.Predict(query, func(row int, minCol int, minVal float64) error {
		rowQuery := query[row*index.state.NumFeatures : (row+1)*index.state.NumFeatures]
		result, err := index.indexes[minCol].Search(rowQuery, k)
		if err != nil {
			return err
		}
		results[row] = make([]int, len(result[0]))
		for i, r := range result[0] {
			results[row][i] = index.state.Mapping[minCol][r]
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
