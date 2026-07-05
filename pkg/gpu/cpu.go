package gpu

import (
	"github.com/magomedcoder/gogguf/pkg/ops"
)

// CPUBackend выполняет matmul на CPU через pkg/ops
type CPUBackend struct{}

func (CPUBackend) Name() string {
	return "CPU"
}

func (CPUBackend) MatMulVec(matrix []float32, rows, cols int, vec []float32) ([]float32, error) {
	return ops.MatMulVec(matrix, rows, cols, vec)
}

func (CPUBackend) MatMulVecCached(_ string, matrix []float32, rows, cols int, vec []float32) ([]float32, error) {
	return ops.MatMulVec(matrix, rows, cols, vec)
}

func (CPUBackend) MatMulVecQ8_0Cached(_ string, raw []byte, rows, cols int, vec []float32) ([]float32, error) {
	return ops.MatMulVecQ8_0(raw, rows, cols, vec)
}

func (CPUBackend) Close() error { return nil }
