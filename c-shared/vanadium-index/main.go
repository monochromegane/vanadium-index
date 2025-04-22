package main

/*
#include <stdlib.h>
#include <stdbool.h>
*/
import "C"
import (
	"encoding/gob"
	"os"
	"runtime/cgo"
	"unsafe"

	vanadium "github.com/monochromegane/vanadium-index"
)

//export NewFlatIndex
func NewFlatIndex(handle *C.ulong, errMsg **C.char, numFeatures C.int) C.int {
	annIndex, err := vanadium.NewIndex(int(numFeatures), vanadium.AsFlat())
	if err != nil {
		*errMsg = C.CString(err.Error())
		return 1
	}
	h := cgo.NewHandle(annIndex)
	*handle = C.ulong(h)
	*errMsg = nil
	return 0
}

//export NewPQIndex
func NewPQIndex(handle *C.ulong, errMsg **C.char, numFeatures C.int, numSubspaces C.int, numClusters C.int, maxIterations C.int, tolerance C.float) C.int {
	opts := []vanadium.ProductQuantizationIndexOption{}
	if maxIterations > 0 {
		opts = append(opts, vanadium.WithPQMaxIterations(int(maxIterations)))
	}
	if tolerance > 0 {
		opts = append(opts, vanadium.WithPQTolerance(float32(tolerance)))
	}
	annIndex, err := vanadium.NewIndex(
		int(numFeatures),
		vanadium.AsPQ(
			int(numSubspaces),
			int(numClusters),
			opts...,
		),
	)
	if err != nil {
		*errMsg = C.CString(err.Error())
		return 1
	}
	h := cgo.NewHandle(annIndex)
	*handle = C.ulong(h)
	*errMsg = nil
	return 0
}

//export NewIVFFlatIndex
func NewIVFFlatIndex(handle *C.ulong, errMsg **C.char, numFeatures C.int, numClusters C.int, maxIterations C.int, tolerance C.float) C.int {
	opts := []vanadium.InvertedFileIndexOption{}
	if maxIterations > 0 {
		opts = append(opts, vanadium.WithIVFMaxIterations(int(maxIterations)))
	}
	if tolerance > 0 {
		opts = append(opts, vanadium.WithIVFTolerance(float32(tolerance)))
	}
	annIndex, err := vanadium.NewIndex(int(numFeatures), vanadium.AsIVFFlat(int(numClusters), opts...))
	if err != nil {
		*errMsg = C.CString(err.Error())
		return 1
	}
	h := cgo.NewHandle(annIndex)
	*handle = C.ulong(h)
	*errMsg = nil
	return 0
}

//export NewIVFPQIndex
func NewIVFPQIndex(handle *C.ulong, errMsg **C.char, numFeatures C.int, numClusters C.int, numSubspaces C.int, numClustersPerSubspace C.int,
	maxIterations C.int, tolerance C.float, pqMaxIterations C.int, pqTolerance C.float) C.int {
	opts := []vanadium.InvertedFileIndexOption{}
	if maxIterations > 0 {
		opts = append(opts, vanadium.WithIVFMaxIterations(int(maxIterations)))
	}
	if tolerance > 0 {
		opts = append(opts, vanadium.WithIVFTolerance(float32(tolerance)))
	}
	if pqMaxIterations > 0 {
		opts = append(opts, vanadium.WithIVFPQIndex(vanadium.WithPQMaxIterations(int(pqMaxIterations))))
	}
	if pqTolerance > 0 {
		opts = append(opts, vanadium.WithIVFPQIndex(vanadium.WithPQTolerance(float32(pqTolerance))))
	}
	annIndex, err := vanadium.NewIndex(
		int(numFeatures),
		vanadium.AsIVFPQ(
			int(numClusters),
			int(numSubspaces),
			int(numClustersPerSubspace),
			opts...,
		),
	)
	if err != nil {
		*errMsg = C.CString(err.Error())
		return 1
	}
	h := cgo.NewHandle(annIndex)
	*handle = C.ulong(h)
	*errMsg = nil
	return 0
}

//export FreeIndex
func FreeIndex(handle C.ulong) {
	h := cgo.Handle(handle)
	h.Delete()
}

//export FreeMemory
func FreeMemory(ptr unsafe.Pointer) {
	C.free(ptr)
}

//export Train
func Train(handle C.ulong, errMsg **C.char, data *C.float, dataLength C.int) C.int {
	slice := unsafe.Slice(data, dataLength)
	dataSlice := *(*[]float32)(unsafe.Pointer(&slice))
	annIndex := cgo.Handle(handle).Value().(vanadium.ANNIndex)
	if err := annIndex.Train(dataSlice); err != nil {
		*errMsg = C.CString(err.Error())
		return 1
	}
	*errMsg = nil
	return 0
}

//export Add
func Add(handle C.ulong, errMsg **C.char, keepData C.bool, data *C.float, dataLength C.int) C.int {
	annIndex := cgo.Handle(handle).Value().(vanadium.ANNIndex)
	slice := unsafe.Slice(data, dataLength)
	dataSlice := *(*[]float32)(unsafe.Pointer(&slice))
	if keepData {
		copiedData := make([]float32, dataLength)
		copy(copiedData, dataSlice)
		if err := annIndex.Add(copiedData); err != nil {
			*errMsg = C.CString(err.Error())
			return 1
		}
	} else {
		if err := annIndex.Add(dataSlice); err != nil {
			*errMsg = C.CString(err.Error())
			return 1
		}
	}
	*errMsg = nil
	return 0
}

//export Search
func Search(handle C.ulong, errMsg **C.char, query *C.float, queryLength C.int, k C.int,
	outIndices **C.int, outDistances **C.float, outOffsets *C.int, outLengths *C.int) C.int {
	annIndex := cgo.Handle(handle).Value().(vanadium.ANNIndex)

	slice := unsafe.Slice(query, queryLength)
	querySlice := *(*[]float32)(unsafe.Pointer(&slice))
	resultIndices, resultDistances, err := annIndex.Search(querySlice, int(k))
	if err != nil {
		*errMsg = C.CString(err.Error())
		return 1
	}

	total := 0
	for _, r := range resultIndices {
		total += len(r)
	}

	indices := (*C.int)(C.malloc(C.size_t(total) * C.size_t(C.sizeof_int)))
	distances := (*C.float)(C.malloc(C.size_t(total) * C.size_t(C.sizeof_float)))
	offsets := unsafe.Slice(outOffsets, len(resultIndices))
	lengths := unsafe.Slice(outLengths, len(resultIndices))

	idx := 0
	goIndices := unsafe.Slice(indices, total)
	goDistances := unsafe.Slice(distances, total)
	for i, r := range resultIndices {
		offsets[i] = C.int(idx)
		lengths[i] = C.int(len(r))
		for j, val := range r {
			goIndices[idx] = C.int(val)
			goDistances[idx] = C.float(resultDistances[i][j])
			idx++
		}
	}
	*outIndices = indices
	*outDistances = distances
	*errMsg = nil
	return 0
}

//export NumVectors
func NumVectors(handle C.ulong) C.int {
	annIndex := cgo.Handle(handle).Value().(vanadium.ANNIndex)
	return C.int(annIndex.NumVectors())
}

//export Save
func Save(handle C.ulong, errMsg **C.char, path *C.char) C.int {
	annIndex := cgo.Handle(handle).Value().(vanadium.ANNIndex)
	file, err := os.Create(C.GoString(path))
	if err != nil {
		*errMsg = C.CString(err.Error())
		return 1
	}
	defer file.Close()
	enc := gob.NewEncoder(file)
	if err := annIndex.Save(enc); err != nil {
		*errMsg = C.CString(err.Error())
		return 1
	}
	*errMsg = nil
	return 0
}

//export Load
func Load(handle *C.ulong, errMsg **C.char, path *C.char) C.int {
	file, err := os.Open(C.GoString(path))
	if err != nil {
		*errMsg = C.CString(err.Error())
		return 1
	}
	defer file.Close()
	dec := gob.NewDecoder(file)
	annIndex, err := vanadium.LoadIndex(dec)
	if err != nil {
		*errMsg = C.CString(err.Error())
		return 1
	}
	h := cgo.NewHandle(annIndex)
	*handle = C.ulong(h)
	*errMsg = nil
	return 0
}

func main() {}
