package annindex

import (
	"runtime"

	"github.com/monochromegane/kmeans"
	"golang.org/x/sync/errgroup"
)

type ProductQuantizationIndex struct {
	numFeatures    int
	numSubspaces   int
	numSubFeatures int
	numClusters    int
	numIterations  int
	tol            float64
	isTrained      bool
	clusters       []kmeans.KMeans
	codebooks      [][][]float64
	codes          [][]int
}

func NewProductQuantizationIndex(numFeatures, numSubspaces int, opts ...ProductQuantizationIndexOption) (*ProductQuantizationIndex, error) {
	if numFeatures <= 0 {
		return nil, ErrInvalidNumFeatures
	}
	if numSubspaces <= 0 || numSubspaces > numFeatures || numFeatures%numSubspaces != 0 {
		return nil, ErrInvalidNumSubspaces
	}
	numSubFeatures := numFeatures / numSubspaces

	index := &ProductQuantizationIndex{
		numFeatures:    numFeatures,
		numSubspaces:   numSubspaces,
		numSubFeatures: numSubFeatures,
		isTrained:      false,
		codebooks:      make([][][]float64, numSubspaces),
		codes:          make([][]int, numSubspaces),
		clusters:       make([]kmeans.KMeans, numSubspaces),

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

	for i := range index.numSubspaces {
		var err error
		var cluster kmeans.KMeans
		if numSubFeatures > 256 {
			cluster, err = kmeans.NewLinearAlgebraKMeans(index.numClusters, numSubFeatures, kmeans.INIT_KMEANS_PLUS_PLUS)
		} else {
			cluster, err = kmeans.NewNaiveKMeans(index.numClusters, numSubFeatures, kmeans.INIT_RANDOM)
		}
		if err != nil {
			return nil, err
		}
		index.clusters[i] = cluster
	}

	return index, nil
}

func (index *ProductQuantizationIndex) Train(data []float64) error {
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

			_, _, err := index.clusters[i].Train(subData, index.numIterations, index.tol)
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

func (index *ProductQuantizationIndex) Add(data []float64) error {
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

	for i := range index.numSubspaces {
		eg.Go(func() error {
			subData := make([]float64, numVectors*index.numSubFeatures)
			for v := range numVectors {
				start := v*index.numFeatures + i*index.numSubFeatures
				end := start + index.numSubFeatures
				copy(subData[v*index.numSubFeatures:], data[start:end])
			}
			code := make([]int, numVectors)
			err := index.clusters[i].Predict(subData, func(row int, minCol int, minVal float64) error {
				code[row] = minCol
				return nil
			})
			if err != nil {
				return err
			}
			index.codes[i] = append(index.codes[i], code...)
			return nil
		})
	}
	if err := eg.Wait(); err != nil {
		return err
	}

	return nil
}

func (index *ProductQuantizationIndex) Search(query []float64, k int) ([][]int, error) {
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
				distanceTable := make([]float64, index.numClusters)
				for c := range index.numClusters {
					distanceTable[c] = index.squaredEuclideanDistance(subQuery, index.codebooks[m][c])
				}
				distanceTables[m] = distanceTable
				return nil
			})
		}
		if err := eg.Wait(); err != nil {
			return nil, err
		}

		numVectors := len(index.codes[0])
		for n := range numVectors {
			distance := 0.0
			for m := range index.numSubspaces {
				distance += distanceTables[m][index.codes[m][n]]
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

func (index *ProductQuantizationIndex) squaredEuclideanDistance(x, y []float64) float64 {
	distance := 0.0
	for i := range x {
		diff := x[i] - y[i]
		distance += diff * diff
	}
	return distance
}
