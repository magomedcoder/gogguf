package dispatch

import (
	"fmt"

	"github.com/magomedcoder/gogguf/pkg/compute"
	"github.com/magomedcoder/gogguf/pkg/dtype"
	"github.com/magomedcoder/gogguf/pkg/weights"
)

// QuantOps is the CPU quant kernel pair for one GGML dtype (no generics over asm blocks)
type QuantOps struct {
	EmbedInto     func(dst []float32, raw []byte, dim, tokenID int) error
	MatMulVecInto func(raw []byte, rows, cols int, vec, out []float32) error
}

var quantOps = map[dtype.DType]QuantOps{
	dtype.Q8_0: {EmbedInto: compute.EmbeddingQ8_0Into, MatMulVecInto: compute.MatMulVecQ8_0Into},
	dtype.Q4_0: {EmbedInto: compute.EmbeddingQ4_0Into, MatMulVecInto: compute.MatMulVecQ4_0Into},
	dtype.Q4_1: {EmbedInto: compute.EmbeddingQ4_1Into, MatMulVecInto: compute.MatMulVecQ4_1Into},
	dtype.Q5_0: {EmbedInto: compute.EmbeddingQ5_0Into, MatMulVecInto: compute.MatMulVecQ5_0Into},
	dtype.Q5_1: {EmbedInto: compute.EmbeddingQ5_1Into, MatMulVecInto: compute.MatMulVecQ5_1Into},
	dtype.Q4_K: {EmbedInto: compute.EmbeddingQ4_KInto, MatMulVecInto: compute.MatMulVecQ4_KInto},
	dtype.Q5_K: {EmbedInto: compute.EmbeddingQ5_KInto, MatMulVecInto: compute.MatMulVecQ5_KInto},
	dtype.Q6_K: {EmbedInto: compute.EmbeddingQ6_KInto, MatMulVecInto: compute.MatMulVecQ6_KInto},
	dtype.Q2_K: {EmbedInto: compute.EmbeddingQ2_KInto, MatMulVecInto: compute.MatMulVecQ2_KInto},
	dtype.Q3_K: {EmbedInto: compute.EmbeddingQ3_KInto, MatMulVecInto: compute.MatMulVecQ3_KInto},
	dtype.Q8_K: {EmbedInto: compute.EmbeddingQ8_KInto, MatMulVecInto: compute.MatMulVecQ8_KInto},
}

// Ops returns CPU quant kernels for t, if registered
func Ops(t dtype.DType) (QuantOps, bool) {
	op, ok := quantOps[t]

	return op, ok
}

// Dispatch looks up a value in a dtype table (thin generic helper for callers)
func Dispatch[T any](t dtype.DType, table map[dtype.DType]T) (T, error) {
	v, ok := table[t]
	if !ok {
		var zero T
		return zero, fmt.Errorf("dispatch: unsupported dtype %v", t)
	}

	return v, nil
}

// EmbeddingTypedInto dequantizes one embedding row by GGML type.
func EmbeddingTypedInto(t dtype.DType, dst []float32, raw []byte, embd, tokenID int, store *weights.Store, name string) error {
	if op, ok := Ops(t); ok {
		return op.EmbedInto(dst, raw, embd, tokenID)
	}

	if store == nil {
		return fmt.Errorf("dispatch: unsupported type %v for %s", t, name)
	}

	f32, err := store.Floats(name)
	if err != nil {
		return err
	}

	off := tokenID * embd
	copy(dst, f32[off:off+embd])

	return nil
}

// LogitsFromHiddenInto RMSNorm(x) into h, then lm_head matmul into logits
func LogitsFromHiddenInto(store *weights.Store, h, x, norm, logits []float32, lmHead string, vocab, embd int, eps float32) error {
	if err := compute.RMSNormInto(h, x, norm, eps); err != nil {
		return err
	}

	return MatMulRawInto(store, lmHead, vocab, embd, h, logits)
}
