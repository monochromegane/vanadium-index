package annindex

import "math"

type IndexBuilder func(*IndexConfig) (ANNIndex, error)

func AsFlat() (IndexBuilder, error) {
	return func(config *IndexConfig) (ANNIndex, error) {
		return NewFlatIndex(config.NumFeatures)
	}, nil
}

func AsPQ(numSubspaces int, numClusters int, opts ...ProductQuantizationIndexOption) (IndexBuilder, error) {
	return func(config *IndexConfig) (ANNIndex, error) {
		if numClusters <= 0 || numClusters > math.MaxUint32 {
			return nil, ErrInvalidNumClusters
		}

		switch {
		case numClusters < 256:
			return NewProductQuantizationIndex(config.NumFeatures, numSubspaces, uint8(numClusters), opts...)
		case numClusters < 65536:
			return NewProductQuantizationIndex(config.NumFeatures, numSubspaces, uint16(numClusters), opts...)
		default:
			return NewProductQuantizationIndex(config.NumFeatures, numSubspaces, uint32(numClusters), opts...)
		}
	}, nil
}

func AsIVFFlat(numClusters int, opts ...InvertedFileIndexOption) (IndexBuilder, error) {
	return func(config *IndexConfig) (ANNIndex, error) {
		if numClusters <= 0 || numClusters > math.MaxUint32 {
			return nil, ErrInvalidNumClusters
		}

		switch {
		case numClusters < 256:
			return NewInvertedFileFlatIndex(config.NumFeatures, uint8(numClusters), opts...)
		case numClusters < 65536:
			return NewInvertedFileFlatIndex(config.NumFeatures, uint16(numClusters), opts...)
		default:
			return NewInvertedFileFlatIndex(config.NumFeatures, uint32(numClusters), opts...)
		}
	}, nil
}

func AsIVFPQ(numClusters int, numSubspaces int, numClustersPerSubspace int, opts ...InvertedFileIndexOption) (IndexBuilder, error) {
	return func(config *IndexConfig) (ANNIndex, error) {
		if numClusters <= 0 || numClustersPerSubspace <= 0 || numClusters > math.MaxUint32 || numClustersPerSubspace > math.MaxUint32 {
			return nil, ErrInvalidNumClusters
		}

		switch {
		case numClusters < 256:
			switch {
			case numClustersPerSubspace < 256:
				return NewInvertedFilePQIndex(config.NumFeatures, uint8(numClusters), numSubspaces, uint8(numClustersPerSubspace), opts...)
			case numClustersPerSubspace < 65536:
				return NewInvertedFilePQIndex(config.NumFeatures, uint8(numClusters), numSubspaces, uint16(numClustersPerSubspace), opts...)
			default:
				return NewInvertedFilePQIndex(config.NumFeatures, uint8(numClusters), numSubspaces, uint32(numClustersPerSubspace), opts...)
			}
		case numClusters < 65536:
			switch {
			case numClustersPerSubspace < 256:
				return NewInvertedFilePQIndex(config.NumFeatures, uint16(numClusters), numSubspaces, uint8(numClustersPerSubspace), opts...)
			case numClustersPerSubspace < 65536:
				return NewInvertedFilePQIndex(config.NumFeatures, uint16(numClusters), numSubspaces, uint16(numClustersPerSubspace), opts...)
			default:
				return NewInvertedFilePQIndex(config.NumFeatures, uint16(numClusters), numSubspaces, uint32(numClustersPerSubspace), opts...)
			}
		default:
			switch {
			case numClustersPerSubspace < 256:
				return NewInvertedFilePQIndex(config.NumFeatures, uint32(numClusters), numSubspaces, uint8(numClustersPerSubspace), opts...)
			case numClustersPerSubspace < 65536:
				return NewInvertedFilePQIndex(config.NumFeatures, uint32(numClusters), numSubspaces, uint16(numClustersPerSubspace), opts...)
			default:
				return NewInvertedFilePQIndex(config.NumFeatures, uint32(numClusters), numSubspaces, uint32(numClustersPerSubspace), opts...)
			}
		}
	}, nil
}

type IndexConfig struct {
	NumFeatures int
}

func NewIndex(numFeatures int, builder IndexBuilder) (ANNIndex, error) {
	config := &IndexConfig{
		NumFeatures: numFeatures,
	}
	return builder(config)
}
