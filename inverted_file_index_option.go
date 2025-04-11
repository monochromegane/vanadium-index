package annindex

type InvertedFileIndexOption func(*InvertedFileIndexConfig, []ProductQuantizationIndexOption) error

func WithIVFNumIterations(numIterations int) InvertedFileIndexOption {
	return func(config *InvertedFileIndexConfig, _ []ProductQuantizationIndexOption) error {
		if numIterations <= 0 {
			return ErrInvalidNumIterations
		}
		config.NumIterations = numIterations
		return nil
	}
}

func WithIVFTol(tol float64) InvertedFileIndexOption {
	return func(config *InvertedFileIndexConfig, _ []ProductQuantizationIndexOption) error {
		if tol <= 0 {
			return ErrInvalidTol
		}
		config.Tol = tol
		return nil
	}
}

func WithIVFPQIndex(opts ...ProductQuantizationIndexOption) InvertedFileIndexOption {
	return func(config *InvertedFileIndexConfig, pqOpts []ProductQuantizationIndexOption) error {
		if pqOpts == nil {
			return ErrInvalidPQOptions
		}
		pqOpts = append(pqOpts, opts...)
		return nil
	}
}
