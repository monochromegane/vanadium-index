package annindex

type ProductQuantizationIndexOption func(*ProductQuantizationIndex) error

func WithPQNumClusters(numClusters int) ProductQuantizationIndexOption {
	return func(index *ProductQuantizationIndex) error {
		if numClusters <= 0 {
			return ErrInvalidNumClusters
		}
		index.numClusters = numClusters
		return nil
	}
}

func WithPQNumIterations(numIterations int) ProductQuantizationIndexOption {
	return func(index *ProductQuantizationIndex) error {
		if numIterations <= 0 {
			return ErrInvalidNumIterations
		}
		index.numIterations = numIterations
		return nil
	}
}

func WithPQTol(tol float64) ProductQuantizationIndexOption {
	return func(index *ProductQuantizationIndex) error {
		if tol <= 0 {
			return ErrInvalidTol
		}
		index.tol = tol
		return nil
	}
}
