package dispatch

import (
	"github.com/magomedcoder/gogguf/pkg/format"
	"github.com/magomedcoder/gogguf/pkg/gpu"
	"github.com/magomedcoder/gogguf/pkg/weights"
)

// MatMulGPUInto multiplies a named weight by vec into out on GPU (quant or FP32 CachedInto).
func MatMulGPUInto(b gpu.Backend, store *weights.Store, name string, rows, cols int, vec, out []float32) error {
	info, err := store.Info(name)
	if err != nil {
		return err
	}

	switch info.Type {
	case format.GgmlQ8_0, format.GgmlQ4_0, format.GgmlQ4_K, format.GgmlQ5_K, format.GgmlQ6_K:
		raw, err := store.Raw(name)
		if err != nil {
			return err
		}

		return gpu.MatMulVecQuantCachedInto(b, info.Type, name, raw, rows, cols, vec, out)
	default:
		f32, err := store.Floats(name)
		if err != nil {
			return err
		}

		return gpu.MatMulVecCachedInto(b, name, f32, rows, cols, vec, out)
	}
}
