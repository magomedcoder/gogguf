package device

import (
	"fmt"

	"github.com/magomedcoder/gogguf/pkg/dtype"
)

// MatMulIntoBackend is optionally implemented by backends that can write matmul into dst.
type MatMulIntoBackend interface {
	MatMulVecCachedInto(name string, matrix []float32, rows, cols int, vec, dst []float32) error
	MatMulVecQ8_0CachedInto(name string, raw []byte, rows, cols int, vec, dst []float32) error
	MatMulVecQ4_0CachedInto(name string, raw []byte, rows, cols int, vec, dst []float32) error
	MatMulVecQ4_KCachedInto(name string, raw []byte, rows, cols int, vec, dst []float32) error
	MatMulVecQ5_KCachedInto(name string, raw []byte, rows, cols int, vec, dst []float32) error
	MatMulVecQ6_KCachedInto(name string, raw []byte, rows, cols int, vec, dst []float32) error
	MatMulVecQ4_1CachedInto(name string, raw []byte, rows, cols int, vec, dst []float32) error
	MatMulVecQ5_0CachedInto(name string, raw []byte, rows, cols int, vec, dst []float32) error
	MatMulVecQ5_1CachedInto(name string, raw []byte, rows, cols int, vec, dst []float32) error
	MatMulVecQ2_KCachedInto(name string, raw []byte, rows, cols int, vec, dst []float32) error
	MatMulVecQ3_KCachedInto(name string, raw []byte, rows, cols int, vec, dst []float32) error
	MatMulVecQ8_KCachedInto(name string, raw []byte, rows, cols int, vec, dst []float32) error
}

// MatMulMatBackend is optionally implemented for W * X[batch] -> out[batch*rows] (GEMM / batched GEMV).
type MatMulMatBackend interface {
	MatMulMatCachedInto(name string, matrix []float32, rows, cols int, x []float32, batch int, out []float32) error
	MatMulMatQ8_0CachedInto(name string, raw []byte, rows, cols int, x []float32, batch int, out []float32) error
	MatMulMatQ4_0CachedInto(name string, raw []byte, rows, cols int, x []float32, batch int, out []float32) error
	MatMulMatQ4_KCachedInto(name string, raw []byte, rows, cols int, x []float32, batch int, out []float32) error
	MatMulMatQ5_KCachedInto(name string, raw []byte, rows, cols int, x []float32, batch int, out []float32) error
	MatMulMatQ6_KCachedInto(name string, raw []byte, rows, cols int, x []float32, batch int, out []float32) error
	MatMulMatQ4_1CachedInto(name string, raw []byte, rows, cols int, x []float32, batch int, out []float32) error
	MatMulMatQ5_0CachedInto(name string, raw []byte, rows, cols int, x []float32, batch int, out []float32) error
	MatMulMatQ5_1CachedInto(name string, raw []byte, rows, cols int, x []float32, batch int, out []float32) error
	MatMulMatQ2_KCachedInto(name string, raw []byte, rows, cols int, x []float32, batch int, out []float32) error
	MatMulMatQ3_KCachedInto(name string, raw []byte, rows, cols int, x []float32, batch int, out []float32) error
	MatMulMatQ8_KCachedInto(name string, raw []byte, rows, cols int, x []float32, batch int, out []float32) error
}

// MatMulVecCachedInto writes FP32 Cached matmul into dst.
func MatMulVecCachedInto(b Backend, name string, matrix []float32, rows, cols int, vec, dst []float32) error {
	if len(dst) < rows {
		return fmt.Errorf("device: MatMulVecCachedInto dst too short")
	}

	if ib, ok := b.(MatMulIntoBackend); ok {
		return ib.MatMulVecCachedInto(name, matrix, rows, cols, vec, dst)
	}

	got, err := b.MatMulVecCached(name, matrix, rows, cols, vec)
	if err != nil {
		return err
	}

	copy(dst, got)

	return nil
}

// MatMulVecQuantCachedInto dispatches quant Cached matmul into dst.
func MatMulVecQuantCachedInto(b Backend, t dtype.DType, name string, raw []byte, rows, cols int, vec, dst []float32) error {
	if len(dst) < rows {
		return fmt.Errorf("device: MatMulVecQuantCachedInto dst too short")
	}

	if ib, ok := b.(MatMulIntoBackend); ok {
		switch t {
		case dtype.Q8_0:
			return ib.MatMulVecQ8_0CachedInto(name, raw, rows, cols, vec, dst)
		case dtype.Q4_0:
			return ib.MatMulVecQ4_0CachedInto(name, raw, rows, cols, vec, dst)
		case dtype.Q4_K:
			return ib.MatMulVecQ4_KCachedInto(name, raw, rows, cols, vec, dst)
		case dtype.Q5_K:
			return ib.MatMulVecQ5_KCachedInto(name, raw, rows, cols, vec, dst)
		case dtype.Q6_K:
			return ib.MatMulVecQ6_KCachedInto(name, raw, rows, cols, vec, dst)
		case dtype.Q4_1:
			return ib.MatMulVecQ4_1CachedInto(name, raw, rows, cols, vec, dst)
		case dtype.Q5_0:
			return ib.MatMulVecQ5_0CachedInto(name, raw, rows, cols, vec, dst)
		case dtype.Q5_1:
			return ib.MatMulVecQ5_1CachedInto(name, raw, rows, cols, vec, dst)
		case dtype.Q2_K:
			return ib.MatMulVecQ2_KCachedInto(name, raw, rows, cols, vec, dst)
		case dtype.Q3_K:
			return ib.MatMulVecQ3_KCachedInto(name, raw, rows, cols, vec, dst)
		case dtype.Q8_K:
			return ib.MatMulVecQ8_KCachedInto(name, raw, rows, cols, vec, dst)
		}
	}

	var got []float32
	var err error
	switch t {
	case dtype.Q8_0:
		got, err = b.MatMulVecQ8_0Cached(name, raw, rows, cols, vec)
	case dtype.Q4_0:
		got, err = b.MatMulVecQ4_0Cached(name, raw, rows, cols, vec)
	case dtype.Q4_K:
		got, err = b.MatMulVecQ4_KCached(name, raw, rows, cols, vec)
	case dtype.Q5_K:
		got, err = b.MatMulVecQ5_KCached(name, raw, rows, cols, vec)
	case dtype.Q6_K:
		got, err = b.MatMulVecQ6_KCached(name, raw, rows, cols, vec)
	case dtype.Q4_1:
		got, err = b.MatMulVecQ4_1Cached(name, raw, rows, cols, vec)
	case dtype.Q5_0:
		got, err = b.MatMulVecQ5_0Cached(name, raw, rows, cols, vec)
	case dtype.Q5_1:
		got, err = b.MatMulVecQ5_1Cached(name, raw, rows, cols, vec)
	case dtype.Q2_K:
		got, err = b.MatMulVecQ2_KCached(name, raw, rows, cols, vec)
	case dtype.Q3_K:
		got, err = b.MatMulVecQ3_KCached(name, raw, rows, cols, vec)
	case dtype.Q8_K:
		got, err = b.MatMulVecQ8_KCached(name, raw, rows, cols, vec)
	default:
		return fmt.Errorf("device: unsupported quant %v for CachedInto", t)
	}
	if err != nil {
		return err
	}

	copy(dst, got)

	return nil
}

// MatMulMatQuantCachedInto dispatches W * X[batch] for supported quants.
// Fallback: loop MatMulVecQuantCachedInto when backend lacks MatMulMatBackend.
func MatMulMatQuantCachedInto(b Backend, t dtype.DType, name string, raw []byte, rows, cols int, x []float32, batch int, out []float32) error {
	if batch < 1 {
		return fmt.Errorf("device: batch=%d", batch)
	}

	if len(x) < batch*cols || len(out) < batch*rows {
		return fmt.Errorf("device: MatMulMat buffers too short for batch=%d", batch)
	}

	if mb, ok := b.(MatMulMatBackend); ok {
		switch t {
		case dtype.Q8_0:
			return mb.MatMulMatQ8_0CachedInto(name, raw, rows, cols, x, batch, out)
		case dtype.Q4_0:
			return mb.MatMulMatQ4_0CachedInto(name, raw, rows, cols, x, batch, out)
		case dtype.Q4_K:
			return mb.MatMulMatQ4_KCachedInto(name, raw, rows, cols, x, batch, out)
		case dtype.Q5_K:
			return mb.MatMulMatQ5_KCachedInto(name, raw, rows, cols, x, batch, out)
		case dtype.Q6_K:
			return mb.MatMulMatQ6_KCachedInto(name, raw, rows, cols, x, batch, out)
		case dtype.Q4_1:
			return mb.MatMulMatQ4_1CachedInto(name, raw, rows, cols, x, batch, out)
		case dtype.Q5_0:
			return mb.MatMulMatQ5_0CachedInto(name, raw, rows, cols, x, batch, out)
		case dtype.Q5_1:
			return mb.MatMulMatQ5_1CachedInto(name, raw, rows, cols, x, batch, out)
		case dtype.Q2_K:
			return mb.MatMulMatQ2_KCachedInto(name, raw, rows, cols, x, batch, out)
		case dtype.Q3_K:
			return mb.MatMulMatQ3_KCachedInto(name, raw, rows, cols, x, batch, out)
		case dtype.Q8_K:
			return mb.MatMulMatQ8_KCachedInto(name, raw, rows, cols, x, batch, out)
		}
	}

	for i := range batch {
		if err := MatMulVecQuantCachedInto(b, t, name, raw, rows, cols, x[i*cols:(i+1)*cols], out[i*rows:(i+1)*rows]); err != nil {
			return err
		}
	}

	return nil
}

// MatMulMatCachedIntoFP32 runs FP32 W * X[batch] (loop-B if no MatMulMatBackend).
func MatMulMatCachedIntoFP32(b Backend, name string, matrix []float32, rows, cols int, x []float32, batch int, out []float32) error {
	if batch < 1 {
		return fmt.Errorf("device: batch=%d", batch)
	}

	if len(x) < batch*cols || len(out) < batch*rows {
		return fmt.Errorf("device: MatMulMat buffers too short for batch=%d", batch)
	}

	if mb, ok := b.(MatMulMatBackend); ok {
		return mb.MatMulMatCachedInto(name, matrix, rows, cols, x, batch, out)
	}

	for i := range batch {
		if err := MatMulVecCachedInto(b, name, matrix, rows, cols, x[i*cols:(i+1)*cols], out[i*rows:(i+1)*rows]); err != nil {
			return err
		}
	}

	return nil
}

// MatMulVecQuantCached dispatches quant Cached matmul for any FusedQuantSupported type and returns a new slice.
func MatMulVecQuantCached(b Backend, t dtype.DType, name string, raw []byte, rows, cols int, vec []float32) ([]float32, error) {
	out := make([]float32, rows)
	if err := MatMulVecQuantCachedInto(b, t, name, raw, rows, cols, vec, out); err != nil {
		return nil, err
	}

	return out, nil
}
