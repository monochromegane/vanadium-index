package vanadium_index

import (
	"encoding/gob"
	"fmt"
)

type IndexType string

const (
	IndexTypeFlat IndexType = "flat"
	IndexTypePQ   IndexType = "pq"
	IndexTypeIVF  IndexType = "ivf"
)

type CodeTypeName string

const (
	CodeTypeNameNone   CodeTypeName = "none"
	CodeTypeNameUint8  CodeTypeName = "uint8"
	CodeTypeNameUint16 CodeTypeName = "uint16"
	CodeTypeNameUint32 CodeTypeName = "uint32"
)

type MetaData struct {
	IndexType IndexType
	CodeType1 CodeTypeName
	CodeType2 CodeTypeName
}

func LoadIndex(dec *gob.Decoder) (ANNIndex, error) {
	var meta MetaData
	err := dec.Decode(&meta)
	if err != nil {
		return nil, err
	}
	switch meta.IndexType {
	case IndexTypeFlat:
		return loadFlatIndex(dec)
	case IndexTypePQ:
		switch meta.CodeType1 {
		case CodeTypeNameUint8:
			return loadProductQuantizationIndex[uint8](dec)
		case CodeTypeNameUint16:
			return loadProductQuantizationIndex[uint16](dec)
		case CodeTypeNameUint32:
			return loadProductQuantizationIndex[uint32](dec)
		default:
			return nil, fmt.Errorf("unknown code type: %s", meta.CodeType1)
		}
	case IndexTypeIVF:
		switch meta.CodeType1 {
		case CodeTypeNameUint8:
			switch meta.CodeType2 {
			case CodeTypeNameUint8:
				return loadInvertedFile[uint8, uint8](dec)
			case CodeTypeNameUint16:
				return loadInvertedFile[uint8, uint16](dec)
			case CodeTypeNameUint32:
				return loadInvertedFile[uint8, uint32](dec)
			default:
				return nil, fmt.Errorf("unknown code type: %s", meta.CodeType2)
			}
		case CodeTypeNameUint16:
			switch meta.CodeType2 {
			case CodeTypeNameUint8:
				return loadInvertedFile[uint16, uint8](dec)
			case CodeTypeNameUint16:
				return loadInvertedFile[uint16, uint16](dec)
			case CodeTypeNameUint32:
				return loadInvertedFile[uint16, uint32](dec)
			default:
				return nil, fmt.Errorf("unknown code type: %s", meta.CodeType2)
			}
		case CodeTypeNameUint32:
			switch meta.CodeType2 {
			case CodeTypeNameUint8:
				return loadInvertedFile[uint32, uint8](dec)
			case CodeTypeNameUint16:
				return loadInvertedFile[uint32, uint16](dec)
			case CodeTypeNameUint32:
				return loadInvertedFile[uint32, uint32](dec)
			default:
				return nil, fmt.Errorf("unknown code type: %s", meta.CodeType2)
			}
		}
		return nil, fmt.Errorf("unknown code type: %s", meta.CodeType1)
	}
	return nil, fmt.Errorf("unknown index type: %s", meta.IndexType)
}
