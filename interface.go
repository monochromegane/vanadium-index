package vanadium_index

import "encoding/gob"

type ANNIndex interface {
	Train(data []float32) error
	Add(data []float32) error
	Search(query []float32, k int) ([][]int, error)
	NumVectors() int
	Save(enc *gob.Encoder) error

	encode(enc *gob.Encoder) error
	decode(dec *gob.Decoder) error
}

type CodeType interface {
	~uint8 | ~uint16 | ~uint32
}

type subIndexBuilder interface {
	build(numFeatures int) (ANNIndex, error)
}
