package dispatch

import (
	"github.com/magomedcoder/gogguf/pkg/device"
	"github.com/magomedcoder/gogguf/pkg/dtype"
	"github.com/magomedcoder/gogguf/pkg/weights"
)

// FFNSwiGLUMatGPUInto runs fused batch FFN on GPU when available (one HtoD + one DtoH)
func FFNSwiGLUMatGPUInto(b device.Backend, store *weights.Store, gateName, upName, downName string, x, out []float32, embd, ffn, batch int) error {
	info, err := store.Info(gateName)
	if err != nil {
		return err
	}

	switch info.Type {
	case dtype.Q8_0,
		dtype.Q8_1,
		dtype.Q4_0,
		dtype.Q4_1,
		dtype.Q5_0,
		dtype.Q5_1,
		dtype.Q2_K,
		dtype.Q3_K,
		dtype.Q4_K,
		dtype.Q5_K,
		dtype.Q6_K,
		dtype.Q8_K:
		gateRaw, err := store.Raw(gateName)
		if err != nil {
			return err
		}

		upRaw, err := store.Raw(upName)
		if err != nil {
			return err
		}

		downRaw, err := store.Raw(downName)
		if err != nil {
			return err
		}

		return device.FFNSwiGLUMatQuantCached(b, info.Type, gateName, upName, downName, gateRaw, upRaw, downRaw, x, out, embd, ffn, batch)
	default:
		gateW, err := store.Floats(gateName)
		if err != nil {
			return err
		}

		upW, err := store.Floats(upName)
		if err != nil {
			return err
		}

		downW, err := store.Floats(downName)
		if err != nil {
			return err
		}

		return device.FFNSwiGLUMatCachedFP32(b, gateName, upName, downName, gateW, upW, downW, x, out, embd, ffn, batch)
	}
}
