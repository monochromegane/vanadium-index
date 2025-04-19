package vanadium_index

import (
	"encoding/gob"
	"reflect"
	"runtime"

	"github.com/monochromegane/kmeans"
	"golang.org/x/sync/errgroup"
)

type InvertedFileIndex[T1, T2 CodeType] struct {
	state   *InvertedFileIndexState[T1, T2]
	cluster *kmeans.KMeans
	indexes []ANNIndex
}

type InvertedFileIndexState[T1, T2 CodeType] struct {
	NumFeatures        int
	NumClusters        T1
	IsTrained          bool
	ShouldTrainIndexes bool
	Config             *InvertedFileIndexConfig
	Mapping            [][]int
}

type InvertedFileIndexConfig struct {
	MaxIterations int
	Tolerance     float32
}

func newInvertedFileFlatIndex[T CodeType](
	numFeatures int,
	numClusters T,
	opts ...InvertedFileIndexOption,
) (*InvertedFileIndex[T, T], error) {
	index := &InvertedFileIndex[T, T]{
		state: &InvertedFileIndexState[T, T]{
			NumFeatures:        numFeatures,
			NumClusters:        numClusters,
			IsTrained:          false,
			ShouldTrainIndexes: false,
			// Default values
			Config: &InvertedFileIndexConfig{
				MaxIterations: 100,
				Tolerance:     1e-4,
			},
		},
	}

	for _, opt := range opts {
		err := opt(index.state.Config, nil)
		if err != nil {
			return nil, err
		}
	}
	indexBuilder := &subFlatIndexBuilder{}
	return newInvertedFileIndex(index, indexBuilder)
}

func newInvertedFilePQIndex[T1, T2 CodeType](
	numFeatures int,
	numIvfClusters T1,
	numPqSubspaces int,
	numPqClusters T2,
	opts ...InvertedFileIndexOption,
) (*InvertedFileIndex[T1, T2], error) {
	index := &InvertedFileIndex[T1, T2]{
		state: &InvertedFileIndexState[T1, T2]{
			NumFeatures:        numFeatures,
			NumClusters:        numIvfClusters,
			IsTrained:          false,
			ShouldTrainIndexes: true,
			// Default values
			Config: &InvertedFileIndexConfig{
				MaxIterations: 100,
				Tolerance:     1e-4,
			},
		},
	}

	pqOpts := []ProductQuantizationIndexOption{}
	for _, opt := range opts {
		err := opt(index.state.Config, pqOpts)
		if err != nil {
			return nil, err
		}
	}
	indexBuilder := &subPQIndexBuilder[T2]{
		numSubspaces: numPqSubspaces,
		numClusters:  numPqClusters,
		opts:         pqOpts,
	}
	return newInvertedFileIndex(index, indexBuilder)
}

func loadInvertedFile[T1, T2 CodeType](dec *gob.Decoder) (*InvertedFileIndex[T1, T2], error) {
	index := &InvertedFileIndex[T1, T2]{}
	err := index.decode(dec)
	if err != nil {
		return nil, err
	}
	return index, nil
}

func newInvertedFileIndex[T1, T2 CodeType](
	index *InvertedFileIndex[T1, T2],
	indexBuilder subIndexBuilder,
) (*InvertedFileIndex[T1, T2], error) {
	cluster, err := kmeans.NewKMeans(
		int(index.state.NumClusters),
		index.state.NumFeatures,
		kmeans.WithInitMethod(kmeans.INIT_KMEANS_PLUS_PLUS),
	)
	if err != nil {
		return nil, err
	}
	index.cluster = cluster

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

func (index *InvertedFileIndex[T1, T2]) Train(data []float32) error {
	if len(data) == 0 {
		return ErrEmptyData
	}

	if len(data)%index.state.NumFeatures != 0 {
		return ErrInvalidDataLength
	}

	_, _, err := index.cluster.Train(
		data,
		kmeans.WithMaxIterations(index.state.Config.MaxIterations),
		kmeans.WithTolerance(index.state.Config.Tolerance),
	)
	if err != nil {
		return err
	}

	numVectors := len(data) / index.state.NumFeatures

	code := make([]T1, numVectors)
	numElements := make([]int, int(index.state.NumClusters))
	err = index.cluster.Predict(data, func(row int, minCol int, minVal float32) error {
		code[row] = T1(minCol)
		numElements[minCol] += 1
		return nil
	})
	if err != nil {
		return err
	}

	if !index.state.ShouldTrainIndexes {
		index.state.IsTrained = true
		return nil
	}

	var eg errgroup.Group
	eg.SetLimit(runtime.NumCPU())

	for c := range int(index.state.NumClusters) {
		eg.Go(func() error {
			countClusterData := 0
			clusterData := make([]float32, numElements[c]*index.state.NumFeatures)
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

func (index *InvertedFileIndex[T1, T2]) Add(data []float32) error {
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
	err := index.cluster.Predict(data, func(row int, minCol int, minVal float32) error {
		rowData := data[row*index.state.NumFeatures : (row+1)*index.state.NumFeatures]
		index.state.Mapping[minCol] = append(index.state.Mapping[minCol], ivfRow+row)
		return index.indexes[minCol].Add(rowData)
	})
	if err != nil {
		return err
	}

	return nil
}

func (index *InvertedFileIndex[T1, T2]) Search(query []float32, k int) ([][]int, error) {
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
	err := index.cluster.Predict(query, func(row int, minCol int, minVal float32) error {
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

func (index *InvertedFileIndex[T1, T2]) Save(enc *gob.Encoder) error {
	var t1 T1
	var t2 T2
	meta := MetaData{
		IndexType: IndexTypeIVF,
		CodeType1: CodeTypeName(reflect.TypeOf(t1).String()),
		CodeType2: CodeTypeName(reflect.TypeOf(t2).String()),
	}
	err := enc.Encode(meta)
	if err != nil {
		return err
	}
	return index.encode(enc)
}

func (index *InvertedFileIndex[T1, T2]) encode(enc *gob.Encoder) error {
	err := enc.Encode(index.state)
	if err != nil {
		return err
	}
	err = index.cluster.Encode(enc)
	if err != nil {
		return err
	}
	for _, index := range index.indexes {
		err = index.encode(enc)
		if err != nil {
			return err
		}
	}
	return nil
}

func (index *InvertedFileIndex[T1, T2]) decode(dec *gob.Decoder) error {
	index.state = &InvertedFileIndexState[T1, T2]{
		Config: &InvertedFileIndexConfig{},
	}
	err := dec.Decode(index.state)
	if err != nil {
		return err
	}

	cluster, err := kmeans.LoadKMeans(dec)
	if err != nil {
		return err
	}
	index.cluster = cluster

	index.indexes = make([]ANNIndex, index.state.NumClusters)
	for i := range int(index.state.NumClusters) {
		if index.state.ShouldTrainIndexes {
			index.indexes[i], err = loadProductQuantizationIndex[T2](dec)
		} else {
			index.indexes[i], err = loadFlatIndex(dec)
		}
		if err != nil {
			return err
		}
	}

	return nil
}

type subFlatIndexBuilder struct{}

func (b *subFlatIndexBuilder) build(numFeatures int) (ANNIndex, error) {
	return newFlatIndex(numFeatures)
}

type subPQIndexBuilder[T CodeType] struct {
	numSubspaces int
	numClusters  T
	opts         []ProductQuantizationIndexOption
}

func (b *subPQIndexBuilder[T]) build(numFeatures int) (ANNIndex, error) {
	return newProductQuantizationIndex(numFeatures, b.numSubspaces, b.numClusters, b.opts...)
}
