// Package dispatch provides shared CPU matmul/embed helpers for model architectures.
package dispatch

import (
	"fmt"

	"github.com/magomedcoder/gogguf/pkg/format"
	"github.com/magomedcoder/gogguf/pkg/ops"
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
func MatMulTypedInto(t format.GGML, raw []byte, rows, cols int, vec, out []float32, store *weights.Store, name string) error {
	switch t {
	case format.GgmlQ8_0:
		return ops.MatMulVecQ8_0Into(raw, rows, cols, vec, out)
	case format.GgmlQ4_0:
		return ops.MatMulVecQ4_0Into(raw, rows, cols, vec, out)
	case format.GgmlQ4_1:
		return ops.MatMulVecQ4_1Into(raw, rows, cols, vec, out)
	case format.GgmlQ5_0:
		return ops.MatMulVecQ5_0Into(raw, rows, cols, vec, out)
	case format.GgmlQ5_1:
		return ops.MatMulVecQ5_1Into(raw, rows, cols, vec, out)
	case format.GgmlQ4_K:
		return ops.MatMulVecQ4_KInto(raw, rows, cols, vec, out)
	case format.GgmlQ5_K:
		return ops.MatMulVecQ5_KInto(raw, rows, cols, vec, out)
	case format.GgmlQ6_K:
		return ops.MatMulVecQ6_KInto(raw, rows, cols, vec, out)
	case format.GgmlQ2_K:
		return ops.MatMulVecQ2_KInto(raw, rows, cols, vec, out)
	case format.GgmlQ3_K:
		return ops.MatMulVecQ3_KInto(raw, rows, cols, vec, out)
	case format.GgmlQ8_K:
		return ops.MatMulVecQ8_KInto(raw, rows, cols, vec, out)
	default:
		if store == nil {
			return fmt.Errorf("dispatch: unsupported type %v for %s", t, name)
		}

		f32, err := store.Floats(name)
		if err != nil {
			return err
		}

		return ops.MatMulVecInto(f32, rows, cols, vec, out)
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
