package dispatch

import (
	"fmt"

	"github.com/magomedcoder/gogguf/pkg/compute"
	"github.com/magomedcoder/gogguf/pkg/device"
	"github.com/magomedcoder/gogguf/pkg/dtype"
	"github.com/magomedcoder/gogguf/pkg/weights"
)

// MatMulMatRawInto computes W * X[batch] -> out[batch*rows] on CPU (n_batch prefill).
// x: [batch*cols], out: [batch*rows].
func MatMulMatRawInto(store *weights.Store, name string, rows, cols int, x []float32, batch int, out []float32) error {
	if batch < 1 {
		return fmt.Errorf("dispatch: batch=%d", batch)
	}

	if len(x) < batch*cols || len(out) < batch*rows {
		return fmt.Errorf("dispatch: MatMulMat buffers too short for batch=%d", batch)
	}

	raw, err := store.Raw(name)
	if err != nil {
		return err
	}

	info, err := store.Info(name)
	if err != nil {
		return err
	}

	switch info.Type {
	case dtype.Q8_0:
		return compute.MatMulMatQ8_0Into(raw, rows, cols, x, batch, out)
	case dtype.Q4_0,
		dtype.Q4_1,
		dtype.Q5_0,
		dtype.Q5_1,
		dtype.Q4_K,
		dtype.Q5_K,
		dtype.Q6_K,
		dtype.Q2_K,
		dtype.Q3_K,
		dtype.Q8_K:
		return compute.MatMulMatViaVecInto(batch, rows, cols, x, out, func(vec, dst []float32) error {
			return MatMulTypedInto(info.Type, raw, rows, cols, vec, dst, store, name)
		})
	default:
		f32, err := store.Floats(name)
		if err != nil {
			return err
		}

		return compute.MatMulMatInto(f32, rows, cols, x, batch, out)
	}
}

// MatMulMatGPUInto computes W * X[batch] -> out[batch*rows] on GPU (quant GEMM / batched launches, else FP32).
func MatMulMatGPUInto(b device.Backend, store *weights.Store, name string, rows, cols int, x []float32, batch int, out []float32) error {
	info, err := store.Info(name)
	if err != nil {
		return err
	}

	switch info.Type {
	case dtype.Q8_0,
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
		raw, err := store.Raw(name)
		if err != nil {
			return err
		}

		return device.MatMulMatQuantCachedInto(b, info.Type, name, raw, rows, cols, x, batch, out)
	default:
		f32, err := store.Floats(name)
		if err != nil {
			return err
		}

		return device.MatMulMatCachedIntoFP32(b, name, f32, rows, cols, x, batch, out)
	}
}

// EmbedTokenInto writes the embedding row of tokenID into dst [embd].
func EmbedTokenInto(store *weights.Store, dst []float32, embd, tokenID int) error {
	const name = "token_embd.weight"

	raw, err := store.Raw(name)
	if err != nil {
		return err
	}

	info, err := store.Info(name)
	if err != nil {
		return err
	}

	switch info.Type {
	case dtype.Q8_0:
		return compute.EmbeddingQ8_0Into(dst, raw, embd, tokenID)
	case dtype.Q4_0:
		return compute.EmbeddingQ4_0Into(dst, raw, embd, tokenID)
	case dtype.Q4_1:
		return compute.EmbeddingQ4_1Into(dst, raw, embd, tokenID)
	case dtype.Q5_0:
		return compute.EmbeddingQ5_0Into(dst, raw, embd, tokenID)
	case dtype.Q5_1:
		return compute.EmbeddingQ5_1Into(dst, raw, embd, tokenID)
	case dtype.Q4_K:
		return compute.EmbeddingQ4_KInto(dst, raw, embd, tokenID)
	case dtype.Q5_K:
		return compute.EmbeddingQ5_KInto(dst, raw, embd, tokenID)
	case dtype.Q6_K:
		return compute.EmbeddingQ6_KInto(dst, raw, embd, tokenID)
	case dtype.Q2_K:
		return compute.EmbeddingQ2_KInto(dst, raw, embd, tokenID)
	case dtype.Q3_K:
		return compute.EmbeddingQ3_KInto(dst, raw, embd, tokenID)
	case dtype.Q8_K:
		return compute.EmbeddingQ8_KInto(dst, raw, embd, tokenID)
	default:
		f32, err := store.Floats(name)
		if err != nil {
			return err
		}

		off := tokenID * embd
		copy(dst, f32[off:off+embd])

		return nil
	}
}
