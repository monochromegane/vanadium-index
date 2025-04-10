package annindex

type ProductQuantizationIndexOption func(*ProductQuantizationIndexConfig) error

func WithPQNumIterations(numIterations int) ProductQuantizationIndexOption {
	return func(config *ProductQuantizationIndexConfig) error {
		config.NumIterations = numIterations
		return nil
	}
}

func WithPQTol(tol float64) ProductQuantizationIndexOption {
	return func(config *ProductQuantizationIndexConfig) error {
		config.Tol = tol
		return nil
	}
}
