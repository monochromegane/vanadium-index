package vanadium_index

type ProductQuantizationIndexOption func(*ProductQuantizationIndexConfig) error

func WithPQMaxIterations(maxIterations int) ProductQuantizationIndexOption {
	return func(config *ProductQuantizationIndexConfig) error {
		config.MaxIterations = maxIterations
		return nil
	}
}

func WithPQTolerance(tol float32) ProductQuantizationIndexOption {
	return func(config *ProductQuantizationIndexConfig) error {
		config.Tolerance = tol
		return nil
	}
}
