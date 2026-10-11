// Package dispatch provides shared CPU matmul/embed helpers for model architectures.
package dispatch

import (
	"fmt"
	"github.com/magomedcoder/gogguf/pkg/dtype"

	"github.com/magomedcoder/gogguf/pkg/compute"
	"github.com/magomedcoder/gogguf/pkg/weights"
)

// MatMulRawInto multiplies a named weight matrix by vec into out (CPU, all common quants).
func MatMulRawInto(store *weights.Store, name string, rows, cols int, vec, out []float32) error {
	raw, err := store.Raw(name)
	if err != nil {
		return err
	}

	info, err := store.Info(name)
	if err != nil {
		return err
	}

	return MatMulTypedInto(info.Type, raw, rows, cols, vec, out, store, name)
}

// MatMulTypedInto dispatches by GGML type. store/name used only for FP32 fallback via Floats.
func MatMulTypedInto(t dtype.DType, raw []byte, rows, cols int, vec, out []float32, store *weights.Store, name string) error {
	switch t {
	case dtype.Q8_0:
		return compute.MatMulVecQ8_0Into(raw, rows, cols, vec, out)
	case dtype.Q4_0:
		return compute.MatMulVecQ4_0Into(raw, rows, cols, vec, out)
	case dtype.Q4_1:
		return compute.MatMulVecQ4_1Into(raw, rows, cols, vec, out)
	case dtype.Q5_0:
		return compute.MatMulVecQ5_0Into(raw, rows, cols, vec, out)
	case dtype.Q5_1:
		return compute.MatMulVecQ5_1Into(raw, rows, cols, vec, out)
	case dtype.Q4_K:
		return compute.MatMulVecQ4_KInto(raw, rows, cols, vec, out)
	case dtype.Q5_K:
		return compute.MatMulVecQ5_KInto(raw, rows, cols, vec, out)
	case dtype.Q6_K:
		return compute.MatMulVecQ6_KInto(raw, rows, cols, vec, out)
	case dtype.Q2_K:
		return compute.MatMulVecQ2_KInto(raw, rows, cols, vec, out)
	case dtype.Q3_K:
		return compute.MatMulVecQ3_KInto(raw, rows, cols, vec, out)
	case dtype.Q8_K:
		return compute.MatMulVecQ8_KInto(raw, rows, cols, vec, out)
	default:
		if store == nil {
			return fmt.Errorf("dispatch: unsupported type %v for %s", t, name)
		}

		f32, err := store.Floats(name)
		if err != nil {
			return err
		}

		return compute.MatMulVecInto(f32, rows, cols, vec, out)
	}
}

// MatMulRaw allocates and returns matmul result (CPU).
func MatMulRaw(store *weights.Store, name string, rows, cols int, vec []float32) ([]float32, error) {
	out := make([]float32, rows)
	if err := MatMulRawInto(store, name, rows, cols, vec, out); err != nil {
		return nil, err
	}

	return out, nil
}
