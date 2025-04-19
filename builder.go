package vanadium_index

import "math"

type IndexBuilder func(*IndexConfig) (ANNIndex, error)

func AsFlat() IndexBuilder {
	return func(config *IndexConfig) (ANNIndex, error) {
		return NewFlatIndex(config.NumFeatures)
	}
}

func AsPQ(numSubspaces int, numClusters int, opts ...ProductQuantizationIndexOption) IndexBuilder {
	return func(config *IndexConfig) (ANNIndex, error) {
		if numClusters <= 0 || numClusters > math.MaxUint32 {
			return nil, ErrInvalidNumClusters
		}

		switch {
		case numClusters < math.MaxUint8:
			return NewProductQuantizationIndex(config.NumFeatures, numSubspaces, uint8(numClusters), opts...)
		case numClusters < math.MaxUint16:
			return NewProductQuantizationIndex(config.NumFeatures, numSubspaces, uint16(numClusters), opts...)
		default:
			return NewProductQuantizationIndex(config.NumFeatures, numSubspaces, uint32(numClusters), opts...)
		}
	}
}

func AsIVFFlat(numClusters int, opts ...InvertedFileIndexOption) IndexBuilder {
	return func(config *IndexConfig) (ANNIndex, error) {
		if numClusters <= 0 || numClusters > math.MaxUint32 {
			return nil, ErrInvalidNumClusters
		}

		switch {
		case numClusters < math.MaxUint8:
			return NewInvertedFileFlatIndex(config.NumFeatures, uint8(numClusters), opts...)
		case numClusters < math.MaxUint16:
			return NewInvertedFileFlatIndex(config.NumFeatures, uint16(numClusters), opts...)
		default:
			return NewInvertedFileFlatIndex(config.NumFeatures, uint32(numClusters), opts...)
		}
	}
}

func AsIVFPQ(numClusters int, numSubspaces int, numClustersPerSubspace int, opts ...InvertedFileIndexOption) IndexBuilder {
	return func(config *IndexConfig) (ANNIndex, error) {
		if numClusters <= 0 || numClustersPerSubspace <= 0 || numClusters > math.MaxUint32 || numClustersPerSubspace > math.MaxUint32 {
			return nil, ErrInvalidNumClusters
		}

		switch {
		case numClusters < math.MaxUint8:
			switch {
			case numClustersPerSubspace < math.MaxUint8:
				return NewInvertedFilePQIndex(config.NumFeatures, uint8(numClusters), numSubspaces, uint8(numClustersPerSubspace), opts...)
			case numClustersPerSubspace < math.MaxUint16:
				return NewInvertedFilePQIndex(config.NumFeatures, uint8(numClusters), numSubspaces, uint16(numClustersPerSubspace), opts...)
			default:
				return NewInvertedFilePQIndex(config.NumFeatures, uint8(numClusters), numSubspaces, uint32(numClustersPerSubspace), opts...)
			}
		case numClusters < math.MaxUint16:
			switch {
			case numClustersPerSubspace < math.MaxUint8:
				return NewInvertedFilePQIndex(config.NumFeatures, uint16(numClusters), numSubspaces, uint8(numClustersPerSubspace), opts...)
			case numClustersPerSubspace < math.MaxUint16:
				return NewInvertedFilePQIndex(config.NumFeatures, uint16(numClusters), numSubspaces, uint16(numClustersPerSubspace), opts...)
			default:
				return NewInvertedFilePQIndex(config.NumFeatures, uint16(numClusters), numSubspaces, uint32(numClustersPerSubspace), opts...)
			}
		default:
			switch {
			case numClustersPerSubspace < math.MaxUint8:
				return NewInvertedFilePQIndex(config.NumFeatures, uint32(numClusters), numSubspaces, uint8(numClustersPerSubspace), opts...)
			case numClustersPerSubspace < math.MaxUint16:
				return NewInvertedFilePQIndex(config.NumFeatures, uint32(numClusters), numSubspaces, uint16(numClustersPerSubspace), opts...)
			default:
				return NewInvertedFilePQIndex(config.NumFeatures, uint32(numClusters), numSubspaces, uint32(numClustersPerSubspace), opts...)
			}
		}
	}
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
