package annindex

type InvertedFileIndexOption func(*InvertedFileIndex) error

func WithIVFNumClusters(numClusters int) InvertedFileIndexOption {
	return func(index *InvertedFileIndex) error {
		if numClusters <= 0 {
			return ErrInvalidNumClusters
		}
		index.numClusters = numClusters
		return nil
	}
}

func WithIVFNumIterations(numIterations int) InvertedFileIndexOption {
	return func(index *InvertedFileIndex) error {
		if numIterations <= 0 {
			return ErrInvalidNumIterations
		}
		index.numIterations = numIterations
		return nil
	}
}

func WithIVFTol(tol float64) InvertedFileIndexOption {
	return func(index *InvertedFileIndex) error {
		if tol <= 0 {
			return ErrInvalidTol
		}
		index.tol = tol
		return nil
	}
}

func WithIVFPQIndex(numSubspaces int, opts ...ProductQuantizationIndexOption) InvertedFileIndexOption {
	return func(index *InvertedFileIndex) error {
		index.pqNumSubspaces = numSubspaces
		index.pqOptions = opts
		return nil
	}
}
