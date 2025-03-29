package annindex

import "fmt"

var ErrInvalidDataLength = fmt.Errorf("data length must be divisible by the number of features")

var ErrInvalidNumFeatures = fmt.Errorf("number of features must be greater than 0")

var ErrEmptyData = fmt.Errorf("data is empty")

var ErrInvalidK = fmt.Errorf("k must be greater than 0")
