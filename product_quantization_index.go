package annindex

import (
	"runtime"

	"github.com/monochromegane/kmeans"
	"golang.org/x/sync/errgroup"
)

type ProductQuantizationIndex[T CodeType] struct {
	numFeatures    int
	numSubspaces   int
	numSubFeatures int
	numClusters    T
	config         *ProductQuantizationIndexConfig
	isTrained      bool
	clusters       []kmeans.KMeans
	codebooks      [][][]float64
	codes          []T
	numVectors     int
}

type ProductQuantizationIndexConfig struct {
	numIterations int
	tol           float64
}

func NewProductQuantizationIndex[T CodeType](
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
		numFeatures:    numFeatures,
		numSubspaces:   numSubspaces,
		numSubFeatures: numSubFeatures,
		isTrained:      false,
		codebooks:      make([][][]float64, numSubspaces),
		codes:          make([]T, numSubspaces),
		clusters:       make([]kmeans.KMeans, numSubspaces),
		numClusters:    numClusters,

		// Default values
		config: &ProductQuantizationIndexConfig{
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

	for i := range index.numSubspaces {
		var err error
		var cluster kmeans.KMeans
		if numSubFeatures > 256 {
			cluster, err = kmeans.NewLinearAlgebraKMeans(int(index.numClusters), numSubFeatures, kmeans.INIT_KMEANS_PLUS_PLUS)
		} else {
			cluster, err = kmeans.NewNaiveKMeans(int(index.numClusters), numSubFeatures, kmeans.INIT_KMEANS_PLUS_PLUS)
		}
		if err != nil {
			return nil, err
		}
		index.clusters[i] = cluster
	}

	return index, nil
}

func (index *ProductQuantizationIndex[T]) Train(data []float64) error {
	if len(data) == 0 {
		return ErrEmptyData
	}

	if len(data)%index.numFeatures != 0 {
		return ErrInvalidDataLength
	}

	var eg errgroup.Group
	eg.SetLimit(runtime.NumCPU())

	numVectors := len(data) / index.numFeatures

	for i := range index.numSubspaces {
		eg.Go(func() error {
			subData := make([]float64, numVectors*index.numSubFeatures)
			for v := range numVectors {
				start := v*index.numFeatures + i*index.numSubFeatures
				end := start + index.numSubFeatures
				copy(subData[v*index.numSubFeatures:], data[start:end])
			}

			_, _, err := index.clusters[i].Train(subData, index.config.numIterations, index.config.tol)
			if err != nil {
				return err
			}
			centroids := index.clusters[i].Centroids()
			index.codebooks[i] = centroids
			return nil
		})
	}
	if err := eg.Wait(); err != nil {
		return err
	}

	index.isTrained = true
	return nil
}

func (index *ProductQuantizationIndex[T]) Add(data []float64) error {
	if len(data) == 0 {
		return ErrEmptyData
	}

	if len(data)%index.numFeatures != 0 {
		return ErrInvalidDataLength
	}

	if !index.isTrained {
		return ErrNotTrained
	}

	var eg errgroup.Group
	eg.SetLimit(runtime.NumCPU())

	numVectors := len(data) / index.numFeatures
	oldNumVectors := index.numVectors
	newCodes := make([]T, (oldNumVectors+numVectors)*index.numSubspaces)
	copy(newCodes, index.codes)
	index.codes = newCodes

	for i := range index.numSubspaces {
		eg.Go(func() error {
			subData := make([]float64, numVectors*index.numSubFeatures)
			for v := range numVectors {
				start := v*index.numFeatures + i*index.numSubFeatures
				end := start + index.numSubFeatures
				copy(subData[v*index.numSubFeatures:], data[start:end])
			}

			err := index.clusters[i].Predict(subData, func(row int, minCol int, minVal float64) error {
				index.codes[(oldNumVectors+row)*index.numSubspaces+i] = T(minCol)
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

	index.numVectors += numVectors
	return nil
}

func (index *ProductQuantizationIndex[T]) Search(query []float64, k int) ([][]int, error) {
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
	neighbors := make([]*SmallestK, numQueries)
	for q := range numQueries {
		neighbors[q] = NewSmallestK(k)
		query := query[q*index.numFeatures : (q+1)*index.numFeatures]

		var eg errgroup.Group
		eg.SetLimit(runtime.NumCPU())

		distanceTables := make([][]float64, index.numSubspaces)
		for m := range index.numSubspaces {
			eg.Go(func() error {
				subQuery := query[m*index.numSubFeatures : (m+1)*index.numSubFeatures]
				distanceTable := make([]float64, int(index.numClusters))
				for c := range int(index.numClusters) {
					distanceTable[c] = index.squaredEuclideanDistance(subQuery, index.codebooks[m][c])
				}
				distanceTables[m] = distanceTable
				return nil
			})
		}
		if err := eg.Wait(); err != nil {
			return nil, err
		}

		for n := range index.numVectors {
			distance := 0.0
			for m := range index.numSubspaces {
				code := index.codes[n*index.numSubspaces+m]
				distance += distanceTables[m][code]
			}
			neighbors[q].Push(n, distance)
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
	return index.numVectors
}

func (index *ProductQuantizationIndex[T]) squaredEuclideanDistance(x, y []float64) float64 {
	distance := 0.0
	for i := range x {
		diff := x[i] - y[i]
		distance += diff * diff
	}
	return distance
}
