package vanadium_index

import (
	"encoding/gob"
	"reflect"
	"runtime"

	"github.com/monochromegane/kmeans"
	"golang.org/x/sync/errgroup"
)

type ProductQuantizationIndex[T CodeType] struct {
	state    *ProductQuantizationState[T]
	clusters []*kmeans.KMeans
}

type ProductQuantizationState[T CodeType] struct {
	NumFeatures    int
	NumSubspaces   int
	NumSubFeatures int
	IsTrained      bool
	NumVectors     int
	Config         *ProductQuantizationIndexConfig
	NumClusters    T
	Codebooks      [][][]float32
	Codes          []T
}

type ProductQuantizationIndexConfig struct {
	MaxIterations int
	Tolerance     float32
}

func newProductQuantizationIndex[T CodeType](
	numFeatures, numSubspaces int,
	numClusters T,
	opts ...ProductQuantizationIndexOption,
) (*ProductQuantizationIndex[T], error) {
	if numFeatures <= 0 {
		return nil, ErrInvalidNumFeatures
	}
	if numSubspaces <= 0 || numSubspaces > numFeatures || numFeatures%numSubspaces != 0 {
		return nil, ErrInvalidNumSubspaces
	}
	numSubFeatures := numFeatures / numSubspaces

	if numClusters <= 0 {
		return nil, ErrInvalidNumClusters
	}

	index := &ProductQuantizationIndex[T]{
		state: &ProductQuantizationState[T]{
			NumFeatures:    numFeatures,
			NumSubspaces:   numSubspaces,
			NumSubFeatures: numSubFeatures,
			IsTrained:      false,
			NumClusters:    numClusters,
			Codebooks:      make([][][]float32, numSubspaces),
			Codes:          make([]T, numSubspaces),
			// Default values
			Config: &ProductQuantizationIndexConfig{
				MaxIterations: 100,
				Tolerance:     1e-4,
			},
		},
		clusters: make([]*kmeans.KMeans, numSubspaces),
	}
	for _, opt := range opts {
		err := opt(index.state.Config)
		if err != nil {
			return nil, err
		}
	}

	for i := range index.state.NumSubspaces {
		cluster, err := kmeans.NewKMeans(
			int(index.state.NumClusters),
			numSubFeatures,
			kmeans.WithInitMethod(kmeans.INIT_KMEANS_PLUS_PLUS),
		)
		if err != nil {
			return nil, err
		}
		index.clusters[i] = cluster
	}

	return index, nil
}

func loadProductQuantizationIndex[T CodeType](dec *gob.Decoder) (*ProductQuantizationIndex[T], error) {
	index := &ProductQuantizationIndex[T]{}
	err := index.decode(dec)
	if err != nil {
		return nil, err
	}
	return index, nil
}

func (index *ProductQuantizationIndex[T]) Train(data []float32) error {
	if len(data) == 0 {
		return ErrEmptyData
	}

	if len(data)%index.state.NumFeatures != 0 {
		return ErrInvalidDataLength
	}

	var eg errgroup.Group
	eg.SetLimit(runtime.NumCPU())

	numVectors := len(data) / index.state.NumFeatures

	for i := range index.state.NumSubspaces {
		eg.Go(func() error {
			subData := make([]float32, numVectors*index.state.NumSubFeatures)
			for v := range numVectors {
				start := v*index.state.NumFeatures + i*index.state.NumSubFeatures
				end := start + index.state.NumSubFeatures
				copy(subData[v*index.state.NumSubFeatures:], data[start:end])
			}

			_, _, err := index.clusters[i].Train(
				subData,
				kmeans.WithMaxIterations(index.state.Config.MaxIterations),
				kmeans.WithTolerance(index.state.Config.Tolerance),
			)
			if err != nil {
				return err
			}
			centroids := index.clusters[i].Centroids()
			index.state.Codebooks[i] = centroids
			return nil
		})
	}
	if err := eg.Wait(); err != nil {
		return err
	}

	index.state.IsTrained = true
	return nil
}

func (index *ProductQuantizationIndex[T]) Add(data []float32) error {
	if len(data) == 0 {
		return ErrEmptyData
	}

	if len(data)%index.state.NumFeatures != 0 {
		return ErrInvalidDataLength
	}

	if !index.state.IsTrained {
		return ErrNotTrained
	}

	var eg errgroup.Group
	eg.SetLimit(runtime.NumCPU())

	numVectors := len(data) / index.state.NumFeatures
	oldNumVectors := index.state.NumVectors
	newCodes := make([]T, (oldNumVectors+numVectors)*index.state.NumSubspaces)
	copy(newCodes, index.state.Codes)
	index.state.Codes = newCodes

	for i := range index.state.NumSubspaces {
		eg.Go(func() error {
			subData := make([]float32, numVectors*index.state.NumSubFeatures)
			for v := range numVectors {
				start := v*index.state.NumFeatures + i*index.state.NumSubFeatures
				end := start + index.state.NumSubFeatures
				copy(subData[v*index.state.NumSubFeatures:], data[start:end])
			}

			err := index.clusters[i].Predict(subData, func(row int, minCol int, minVal float32) error {
				index.state.Codes[(oldNumVectors+row)*index.state.NumSubspaces+i] = T(minCol)
				return nil
			})
			if err != nil {
				return err
			}
			return nil
		})
	}
	if err := eg.Wait(); err != nil {
		return err
	}

	index.state.NumVectors += numVectors
	return nil
}

func (index *ProductQuantizationIndex[T]) Search(query []float32, k int) ([][]int, error) {
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

	type distanceItem struct {
		index    int
		distance float32
	}
	batchSize := 1000

	numQueries := len(query) / index.state.NumFeatures
	neighbors := make([]*SmallestK, numQueries)
	for q := range numQueries {
		neighbors[q] = NewSmallestK(k)
		query := query[q*index.state.NumFeatures : (q+1)*index.state.NumFeatures]

		numWorkers := runtime.NumCPU()
		var eg errgroup.Group
		eg.SetLimit(numWorkers)

		distanceTable := make([]float32, index.state.NumSubspaces*int(index.state.NumClusters))
		for m := range index.state.NumSubspaces {
			eg.Go(func() error {
				subQuery := query[m*index.state.NumSubFeatures : (m+1)*index.state.NumSubFeatures]
				offset := m * int(index.state.NumClusters)
				for c := range int(index.state.NumClusters) {
					distanceTable[offset+c] = index.squaredEuclideanDistance(subQuery, index.state.Codebooks[m][c])
				}
				return nil
			})
		}
		if err := eg.Wait(); err != nil {
			return nil, err
		}

		chunkSize := index.state.NumVectors / numWorkers
		if chunkSize == 0 {
			chunkSize = 1
			numWorkers = index.state.NumVectors
		}
		distChan := make(chan []distanceItem, numWorkers)

		for w := range numWorkers {
			start := w * chunkSize
			end := start + chunkSize
			if w == numWorkers-1 {
				end = index.state.NumVectors
			}
			eg.Go(func() error {
				batchDistance := make([]distanceItem, 0, batchSize)
				for n := start; n < end; n++ {
					distance := float32(0)
					for m := range index.state.NumSubspaces {
						code := index.state.Codes[n*index.state.NumSubspaces+m]
						distance += distanceTable[m*int(index.state.NumClusters)+int(code)]
					}
					batchDistance = append(batchDistance, distanceItem{index: n, distance: distance})
					if len(batchDistance) >= batchSize {
						distChan <- batchDistance
						batchDistance = make([]distanceItem, 0, batchSize)
					}
				}
				if len(batchDistance) > 0 {
					distChan <- batchDistance
				}
				return nil
			})
		}
		go func() {
			eg.Wait()
			close(distChan)
		}()

		for batch := range distChan {
			for _, item := range batch {
				neighbors[q].Push(item.index, item.distance)
			}
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

func (index *ProductQuantizationIndex[T]) NumVectors() int {
	return index.state.NumVectors
}

func (index *ProductQuantizationIndex[T]) Save(enc *gob.Encoder) error {
	var t T
	meta := MetaData{
		IndexType: IndexTypePQ,
		CodeType1: CodeTypeName(reflect.TypeOf(t).String()),
		CodeType2: CodeTypeNameNone,
	}
	err := enc.Encode(meta)
	if err != nil {
		return err
	}
	return index.encode(enc)
}

func (index *ProductQuantizationIndex[T]) encode(enc *gob.Encoder) error {
	err := enc.Encode(index.state)
	if err != nil {
		return err
	}
	for _, cluster := range index.clusters {
		err = cluster.Encode(enc)
		if err != nil {
			return err
		}
	}
	return nil
}

func (index *ProductQuantizationIndex[T]) decode(dec *gob.Decoder) error {
	index.state = &ProductQuantizationState[T]{
		Config: &ProductQuantizationIndexConfig{},
	}
	err := dec.Decode(index.state)
	if err != nil {
		return err
	}

	numSubspaces := index.state.NumSubspaces
	clusters := make([]*kmeans.KMeans, numSubspaces)
	for i := range numSubspaces {
		cluster, err := kmeans.LoadKMeans(dec)
		if err != nil {
			return err
		}
		clusters[i] = cluster
	}
	index.clusters = clusters
	return nil
}

func (index *ProductQuantizationIndex[T]) squaredEuclideanDistance(x, y []float32) float32 {
	distance := float32(0)
	for i := range x {
		diff := x[i] - y[i]
		distance += diff * diff
	}
	return distance
}
