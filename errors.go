package vanadium_index

import (
	"fmt"
	"math"
)

var ErrInvalidDataLength = fmt.Errorf("data length must be divisible by the number of features")

var ErrInvalidNumFeatures = fmt.Errorf("number of features must be greater than 0")

var ErrEmptyData = fmt.Errorf("data is empty")

var ErrInvalidK = fmt.Errorf("k must be greater than 0")

var ErrInvalidNumSubspaces = fmt.Errorf("number of subspaces must be greater than 0")

var ErrInvalidNumClusters = fmt.Errorf("number of clusters must be greater than 0 and less than or equal to %d", math.MaxUint32)

var ErrInvalidNumIterations = fmt.Errorf("number of iterations must be greater than 0")

var ErrInvalidTol = fmt.Errorf("tolerance must be greater than 0")

var ErrInvalidPQOptions = fmt.Errorf("pq options can use only in ProductQuantizationIndex")

var ErrNotTrained = fmt.Errorf("index is not trained")
