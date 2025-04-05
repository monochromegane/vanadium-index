package annindex

type InvertedFileIndexOption func(*InvertedFileIndexConfig) error

func WithIVFNumIterations(numIterations int) InvertedFileIndexOption {
	return func(config *InvertedFileIndexConfig) error {
		if numIterations <= 0 {
			return ErrInvalidNumIterations
		}
		config.numIterations = numIterations
		return nil
	}
}

func WithIVFTol(tol float64) InvertedFileIndexOption {
	return func(config *InvertedFileIndexConfig) error {
		if tol <= 0 {
			return ErrInvalidTol
		}
		config.tol = tol
		return nil
	}
}

func WithIVFPQIndex(opts ...ProductQuantizationIndexOption) InvertedFileIndexOption {
	return func(config *InvertedFileIndexConfig) error {
		config.pqOptions = opts
		return nil
	}
}
