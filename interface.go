package annindex

type ANNIndex interface {
	Add(data []float64) error
	Search(query []float64, k int) ([][]int, error)
	Train(data []float64) error
}
