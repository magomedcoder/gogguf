package metal

import (
	"fmt"
	"math"

	"github.com/magomedcoder/gogguf/pkg/dtype"
)

// B3: prefill (-b) / batch GEMM. The contracts mirror pkg/device/cuda (matmul_mat.go, ffn_mat.go, AttentionScoresKVBatch) so pkg/device.MatMulMatBackend, FFNSwiGLUMatBackend and
// KVBatchAttentionBackend work on Metal unchanged:
//   - x is [batch][cols] and out [batch][rows], token b at offset b*cols / b*rows
//   - Q8_0 is the only weight type with a GEMM kernel (matmul_mat_q8_0: ONE launch for W * X[batch]); batch == 1 takes the existing matvec path
//   - every other MatMulMat* entry (F32 and the remaining quants) loops the matching MatMulVec*Cached INSIDE the method, so device.MatMulMatQuantCachedInto (which type-asserts MatMulMatBackend and then calls the per-type method) keeps working for Q4_0 / Q4_K prefill and the unsupported quants keep failing with ErrUnavailable like before
//   - FFNSwiGLUMatQuantCached / AttentionScoresKVBatch only exist for Q8_0 / the device KV cache; callers fall back to matmulBatchInto / the per-query loop on error
//   - the host-side input x is staged once and reused while the same slice (pointer + length, content-checked) feeds consecutive GEMMs (Q -> K -> V, gate -> up), like CUDA skip_vec_htod

// ffnMatArgs is the batch FFN argument bundle handed to the runtime
type ffnMatArgs struct {
	gateName, upName, downName string
	gateRaw, upRaw, downRaw    []byte
	x, out                     []float32
	embd, ffn, batch           int
}

// maxGridThreads bounds batch*rows for matmul_mat_q8_0 (thread_position_in_grid is 32-bit)
const maxGridThreads = math.MaxUint32 - 1024

// checkMatMulMat validates the shared MatMulMat* geometry and buffers (same messages as the CUDA backend)
func checkMatMulMat(op string, rows, cols, batch int, x, out []float32) error {
	if batch < 1 {
		return fmt.Errorf("metal: %s: batch=%d", op, batch)
	}

	if rows < 1 || cols < 1 {
		return fmt.Errorf("metal: %s: rows=%d cols=%d", op, rows, cols)
	}

	if len(x) < batch*cols {
		return fmt.Errorf("metal: %s: x too short for batch=%d", op, batch)
	}

	if len(out) < batch*rows {
		return fmt.Errorf("metal: %s: out too short for batch=%d", op, batch)
	}

	return nil
}

// matMulMatLoop is the loop-B fallback inside MatMulMat*: one matvec per token via mul (a MatMulVec*Cached method), results copied into out.
// Used for F32 and every quant without a GEMM kernel; a matvec ErrUnavailable is returned as is.
func (b *Backend) matMulMatLoop(op string, rows, cols, batch int, x, out []float32, mul func(vec []float32) ([]float32, error)) error {
	if _, err := b.runtime(op); err != nil {
		return err
	}

	if err := checkMatMulMat(op, rows, cols, batch, x, out); err != nil {
		return err
	}

	for i := range batch {
		got, err := mul(x[i*cols : (i+1)*cols])
		if err != nil {
			return err
		}

		copy(out[i*rows:(i+1)*rows], got)
	}

	return nil
}

// MatMulMatCachedInto FP32 W * X[batch]: no GEMM kernel, loops MatMulVecCached
func (b *Backend) MatMulMatCachedInto(name string, matrix []float32, rows, cols int, x []float32, batch int, out []float32) error {
	return b.matMulMatLoop("MatMulMatCachedInto", rows, cols, batch, x, out, func(vec []float32) ([]float32, error) {
		return b.MatMulVecCached(name, matrix, rows, cols, vec)
	})
}

// MatMulMatQ8_0CachedInto W * X[batch] for a Q8_0 matrix: one matmul_mat_q8_0 launch (batch == 1: the matvec kernel).
// The weight is cached by name in the same device cache as the matvec path (decode and prefill share one upload); name == "" uploads for this call only.
func (b *Backend) MatMulMatQ8_0CachedInto(name string, raw []byte, rows, cols int, x []float32, batch int, out []float32) error {
	const op = "MatMulMatQ8_0CachedInto"

	rt, err := b.runtime(op)
	if err != nil {
		return err
	}

	if err := checkMatMulMat(op, rows, cols, batch, x, out); err != nil {
		return err
	}

	if err := checkQuantMatrix(op, dtype.Q8_0, raw, rows, cols); err != nil {
		return err
	}

	if batch == 1 {
		return rt.matMulQ8_0(name, raw, rows, cols, x[:cols], out[:rows])
	}

	if uint64(rows)*uint64(batch) > maxGridThreads {
		return fmt.Errorf("metal: %s: rows=%d batch=%d exceed the launch grid", op, rows, batch)
	}

	return rt.matMulMatQ8_0(name, raw, rows, cols, x[:batch*cols], batch, out[:batch*rows])
}

func (b *Backend) MatMulMatQ8_1CachedInto(name string, raw []byte, rows, cols int, x []float32, batch int, out []float32) error {
	return b.matMulMatLoop("MatMulMatQ8_1CachedInto", rows, cols, batch, x, out, func(vec []float32) ([]float32, error) {
		return b.MatMulVecQ8_1Cached(name, raw, rows, cols, vec)
	})
}

func (b *Backend) MatMulMatQ4_0CachedInto(name string, raw []byte, rows, cols int, x []float32, batch int, out []float32) error {
	return b.matMulMatLoop("MatMulMatQ4_0CachedInto", rows, cols, batch, x, out, func(vec []float32) ([]float32, error) {
		return b.MatMulVecQ4_0Cached(name, raw, rows, cols, vec)
	})
}

func (b *Backend) MatMulMatQ4_KCachedInto(name string, raw []byte, rows, cols int, x []float32, batch int, out []float32) error {
	return b.matMulMatLoop("MatMulMatQ4_KCachedInto", rows, cols, batch, x, out, func(vec []float32) ([]float32, error) {
		return b.MatMulVecQ4_KCached(name, raw, rows, cols, vec)
	})
}

func (b *Backend) MatMulMatQ5_KCachedInto(name string, raw []byte, rows, cols int, x []float32, batch int, out []float32) error {
	return b.matMulMatLoop("MatMulMatQ5_KCachedInto", rows, cols, batch, x, out, func(vec []float32) ([]float32, error) {
		return b.MatMulVecQ5_KCached(name, raw, rows, cols, vec)
	})
}

func (b *Backend) MatMulMatQ6_KCachedInto(name string, raw []byte, rows, cols int, x []float32, batch int, out []float32) error {
	return b.matMulMatLoop("MatMulMatQ6_KCachedInto", rows, cols, batch, x, out, func(vec []float32) ([]float32, error) {
		return b.MatMulVecQ6_KCached(name, raw, rows, cols, vec)
	})
}

func (b *Backend) MatMulMatQ4_1CachedInto(name string, raw []byte, rows, cols int, x []float32, batch int, out []float32) error {
	return b.matMulMatLoop("MatMulMatQ4_1CachedInto", rows, cols, batch, x, out, func(vec []float32) ([]float32, error) {
		return b.MatMulVecQ4_1Cached(name, raw, rows, cols, vec)
	})
}

func (b *Backend) MatMulMatQ5_0CachedInto(name string, raw []byte, rows, cols int, x []float32, batch int, out []float32) error {
	return b.matMulMatLoop("MatMulMatQ5_0CachedInto", rows, cols, batch, x, out, func(vec []float32) ([]float32, error) {
		return b.MatMulVecQ5_0Cached(name, raw, rows, cols, vec)
	})
}

func (b *Backend) MatMulMatQ5_1CachedInto(name string, raw []byte, rows, cols int, x []float32, batch int, out []float32) error {
	return b.matMulMatLoop("MatMulMatQ5_1CachedInto", rows, cols, batch, x, out, func(vec []float32) ([]float32, error) {
		return b.MatMulVecQ5_1Cached(name, raw, rows, cols, vec)
	})
}

func (b *Backend) MatMulMatQ2_KCachedInto(name string, raw []byte, rows, cols int, x []float32, batch int, out []float32) error {
	return b.matMulMatLoop("MatMulMatQ2_KCachedInto", rows, cols, batch, x, out, func(vec []float32) ([]float32, error) {
		return b.MatMulVecQ2_KCached(name, raw, rows, cols, vec)
	})
}

func (b *Backend) MatMulMatQ3_KCachedInto(name string, raw []byte, rows, cols int, x []float32, batch int, out []float32) error {
	return b.matMulMatLoop("MatMulMatQ3_KCachedInto", rows, cols, batch, x, out, func(vec []float32) ([]float32, error) {
		return b.MatMulVecQ3_KCached(name, raw, rows, cols, vec)
	})
}

func (b *Backend) MatMulMatQ8_KCachedInto(name string, raw []byte, rows, cols int, x []float32, batch int, out []float32) error {
	return b.matMulMatLoop("MatMulMatQ8_KCachedInto", rows, cols, batch, x, out, func(vec []float32) ([]float32, error) {
		return b.MatMulVecQ8_KCached(name, raw, rows, cols, vec)
	})
}

// FFNSwiGLUMatCached FP32 batch FFN has no Metal kernel: ErrUnavailable (callers use the per-matmul path)
func (b *Backend) FFNSwiGLUMatCached(_, _, _ string, _, _, _, _, _ []float32, _, _, _ int) error {
	return unavailable("FFNSwiGLUMatCached")
}

// FFNSwiGLUMatQuantCached batch FFN for Q8_0: gate/up GEMM -> SwiGLU -> down GEMM in one command buffer, activations never leave the device.
// x / out are [batch][embd] and may alias (the qwen3 prefill passes the same slice for both).
// Other types return ErrUnavailable (the caller falls back to MatMulMat*).
// batch == 1 is the B2 fused matvec FFN.
func (b *Backend) FFNSwiGLUMatQuantCached(t dtype.DType, gateName, upName, downName string, gateRaw, upRaw, downRaw []byte, x, out []float32, embd, ffn, batch int) error {
	const op = "FFNSwiGLUMatQuantCached"

	rt, err := b.runtime(op)
	if err != nil {
		return err
	}

	if t != dtype.Q8_0 {
		return unsupported(op, fmt.Sprintf("batch FFN supports Q8_0 only, got %s", t))
	}

	if batch < 1 || embd < 1 || ffn < 1 {
		return fmt.Errorf("metal: %s: batch=%d embd=%d ffn=%d", op, batch, embd, ffn)
	}

	if len(x) < batch*embd || len(out) < batch*embd {
		return fmt.Errorf("metal: %s: buffers too short", op)
	}

	if gateName == "" || upName == "" || downName == "" {
		return fmt.Errorf("metal: %s: weight names are required (device weight cache key)", op)
	}

	for _, c := range []struct {
		raw        []byte
		rows, cols int
	}{{gateRaw, ffn, embd}, {upRaw, ffn, embd}, {downRaw, embd, ffn}} {
		if err := checkQuantMatrix(op, t, c.raw, c.rows, c.cols); err != nil {
			return err
		}
	}

	if batch == 1 {
		return b.FFNSwiGLUQuantCached(t, gateName, upName, downName, gateRaw, upRaw, downRaw, x[:embd], out[:embd], embd, ffn)
	}

	if uint64(batch)*uint64(max(ffn, embd)) > maxGridThreads {
		return fmt.Errorf("metal: %s: batch=%d embd=%d ffn=%d exceed the launch grid", op, batch, embd, ffn)
	}

	return rt.ffnSwiGLUMatQ8_0(&ffnMatArgs{
		gateName: gateName,
		upName:   upName,
		downName: downName,
		gateRaw:  gateRaw,
		upRaw:    upRaw,
		downRaw:  downRaw,
		x:        x[:batch*embd],
		out:      out[:batch*embd],
		embd:     embd,
		ffn:      ffn,
		batch:    batch,
	})
}

// AttentionScoresKVBatch causal attention for n queries over the device KV cache: q / dst are [n][nHeads*headDim], query i attends to pastLen+i+1 positions.
// The K/V of all n tokens must already be in the cache (KVCacheAppendN), like the CUDA backend. The n attention launches (one per query row) are encoded into one command buffer: one commit + wait, one Q upload, one result copy.
func (b *Backend) AttentionScoresKVBatch(layer int, dst, q []float32, n, pastLen, nHeads, nKVHeads, headDim int) error {
	const op = "AttentionScoresKVBatch"

	rt, err := b.runtime(op)
	if err != nil {
		return err
	}

	if n < 1 || pastLen < 0 {
		return fmt.Errorf("metal: %s: n=%d pastLen=%d", op, n, pastLen)
	}

	if err := checkAttnDims(op, pastLen+n, nHeads, nKVHeads, headDim); err != nil {
		return err
	}

	qDim := nHeads * headDim
	if len(dst) < n*qDim || len(q) < n*qDim {
		return fmt.Errorf("metal: %s: dst/q too short", op)
	}

	return rt.attentionKVBatch(layer, dst[:n*qDim], q[:n*qDim], n, pastLen, nHeads, nKVHeads, headDim)
}
