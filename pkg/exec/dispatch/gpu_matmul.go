package dispatch

import (
	"github.com/magomedcoder/gogguf/pkg/device"
	"github.com/magomedcoder/gogguf/pkg/dtype"
	"github.com/magomedcoder/gogguf/pkg/weights"
)

// MatMulGPUInto multiplies a named weight by vec into out on GPU (quant or FP32 CachedInto).
func MatMulGPUInto(b device.Backend, store *weights.Store, name string, rows, cols int, vec, out []float32) error {
	info, err := store.Info(name)
	if err != nil {
		return err
	}

	switch info.Type {
	case dtype.Q8_0,
		dtype.Q4_0,
		dtype.Q4_K,
		dtype.Q5_K,
		dtype.Q6_K:
		raw, err := store.Raw(name)
		if err != nil {
			return err
		}

		return device.MatMulVecQuantCachedInto(b, info.Type, name, raw, rows, cols, vec, out)
	default:
		f32, err := store.Floats(name)
		if err != nil {
			return err
		}

		return device.MatMulVecCachedInto(b, name, f32, rows, cols, vec, out)
	}
}
