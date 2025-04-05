package annindex

type ProductQuantizationIndexOption func(*ProductQuantizationIndexConfig) error

func WithPQNumIterations(numIterations int) ProductQuantizationIndexOption {
	return func(config *ProductQuantizationIndexConfig) error {
		config.numIterations = numIterations
		return nil
	}
}

func WithPQTol(tol float64) ProductQuantizationIndexOption {
	return func(config *ProductQuantizationIndexConfig) error {
		config.tol = tol
		return nil
	}
}
