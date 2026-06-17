package gpu

import (
	"github.com/magomedcoder/gguf.go/pkg/ops"
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

func (CPUBackend) Close() error { return nil }
