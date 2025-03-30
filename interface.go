package annindex

type ANNIndex interface {
	Train(data []float64) error
	Add(data []float64) error
	Search(query []float64, k int) ([][]int, error)
}
