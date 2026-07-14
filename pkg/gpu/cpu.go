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

func (CPUBackend) RMSNormInto(dst, x, weight []float32, eps float32) error {
	return ops.RMSNormInto(dst, x, weight, eps)
}

func (CPUBackend) ApplyRoPEHeads(v []float32, nHeads, headDim, pos int, freqBase float32) error {
	ops.ApplyRoPEHeads(v, nHeads, headDim, pos, freqBase)
	return nil
}

func (CPUBackend) ApplyRoPEHeadsNorm(v []float32, nHeads, headDim, pos int, freqBase float32) error {
	ops.ApplyRoPEHeadsNorm(v, nHeads, headDim, pos, freqBase)
	return nil
}

func (CPUBackend) SwiGLUInPlace(gate, up []float32) error {
	ops.SwiGLUInPlace(gate, up)
	return nil
}

func (CPUBackend) AttentionScoresInto(dst, q, k, v, scores []float32, seqLen, nHeads, nKVHeads, headDim int) error {
	return ops.AttentionScoresInto(dst, q, k, v, scores, seqLen, nHeads, nKVHeads, headDim)
}

func (CPUBackend) KVCacheInit(int, int, int, int, int) error {
	return nil
}

func (CPUBackend) KVCacheReset() {}

func (CPUBackend) KVCacheAppend(int, int, []float32, []float32) error {
	return nil
}

func (CPUBackend) AttentionScoresKV(int, []float32, []float32, int, int, int, int) error {
	return ErrKVCacheUnavailable
}

func (CPUBackend) Close() error {
	return nil
}
