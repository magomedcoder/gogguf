package gpu

import (
	"fmt"

	"github.com/magomedcoder/gogguf/pkg/format"
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
		return fmt.Errorf("gpu: MatMulVecCachedInto dst too short")
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
func MatMulVecQuantCachedInto(b Backend, t format.GGML, name string, raw []byte, rows, cols int, vec, dst []float32) error {
	if len(dst) < rows {
		return fmt.Errorf("gpu: MatMulVecQuantCachedInto dst too short")
	}

	if ib, ok := b.(MatMulIntoBackend); ok {
		switch t {
		case format.GgmlQ8_0:
			return ib.MatMulVecQ8_0CachedInto(name, raw, rows, cols, vec, dst)
		case format.GgmlQ4_0:
			return ib.MatMulVecQ4_0CachedInto(name, raw, rows, cols, vec, dst)
		case format.GgmlQ4_K:
			return ib.MatMulVecQ4_KCachedInto(name, raw, rows, cols, vec, dst)
		case format.GgmlQ5_K:
			return ib.MatMulVecQ5_KCachedInto(name, raw, rows, cols, vec, dst)
		case format.GgmlQ6_K:
			return ib.MatMulVecQ6_KCachedInto(name, raw, rows, cols, vec, dst)
		}
	}

	var got []float32
	var err error
	switch t {
	case format.GgmlQ8_0:
		got, err = b.MatMulVecQ8_0Cached(name, raw, rows, cols, vec)
	case format.GgmlQ4_0:
		got, err = b.MatMulVecQ4_0Cached(name, raw, rows, cols, vec)
	case format.GgmlQ4_K:
		got, err = b.MatMulVecQ4_KCached(name, raw, rows, cols, vec)
	case format.GgmlQ5_K:
		got, err = b.MatMulVecQ5_KCached(name, raw, rows, cols, vec)
	case format.GgmlQ6_K:
		got, err = b.MatMulVecQ6_KCached(name, raw, rows, cols, vec)
	default:
		return fmt.Errorf("gpu: unsupported quant %v for CachedInto", t)
	}
	if err != nil {
		return err
	}

	copy(dst, got)

	return nil
}
