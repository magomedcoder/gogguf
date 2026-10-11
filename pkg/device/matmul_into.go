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
	default:
		return fmt.Errorf("device: unsupported quant %v for CachedInto", t)
	}
	if err != nil {
		return err
	}

	copy(dst, got)

	return nil
}
