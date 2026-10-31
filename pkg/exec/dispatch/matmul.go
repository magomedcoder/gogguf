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
	if op, ok := Ops(t); ok {
		return op.MatMulVecInto(raw, rows, cols, vec, out)
	}
	if store == nil {
		return fmt.Errorf("dispatch: unsupported type %v for %s", t, name)
	}

	f32, err := store.Floats(name)
	if err != nil {
		return err
	}

	return compute.MatMulVecInto(f32, rows, cols, vec, out)
}

// MatMulRaw allocates and returns matmul result (CPU).
func MatMulRaw(store *weights.Store, name string, rows, cols int, vec []float32) ([]float32, error) {
	out := make([]float32, rows)
	if err := MatMulRawInto(store, name, rows, cols, vec, out); err != nil {
		return nil, err
	}

	return out, nil
}
