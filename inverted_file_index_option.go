package annindex

type InvertedFileIndexOption func(*InvertedFileIndexConfig, []ProductQuantizationIndexOption) error

func WithIVFMaxIterations(maxIterations int) InvertedFileIndexOption {
	return func(config *InvertedFileIndexConfig, _ []ProductQuantizationIndexOption) error {
		if maxIterations <= 0 {
			return ErrInvalidNumIterations
		}
		config.MaxIterations = maxIterations
		return nil
	}
}

func WithIVFTolerance(tol float32) InvertedFileIndexOption {
	return func(config *InvertedFileIndexConfig, _ []ProductQuantizationIndexOption) error {
		if tol <= 0 {
			return ErrInvalidTol
		}
		config.Tolerance = tol
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
