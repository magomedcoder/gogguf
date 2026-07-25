package gpu

import (
	"github.com/magomedcoder/gogguf/pkg/ops"
)

// CPUBackend выполняет matmul на CPU через pkg/ops
type CPUBackend struct{}

func (CPUBackend) Name() string {
	return "CPU"
}

func (CPUBackend) MatMulVec(matrix []float32, rows, cols int, vec []float32) ([]float32, error) {
	return ops.MatMulVec(matrix, rows, cols, vec)
}

func (CPUBackend) MatMulVecCached(_ string, matrix []float32, rows, cols int, vec []float32) ([]float32, error) {
	return ops.MatMulVec(matrix, rows, cols, vec)
}

func (CPUBackend) MatMulVecQ8_0Cached(_ string, raw []byte, rows, cols int, vec []float32) ([]float32, error) {
	return ops.MatMulVecQ8_0(raw, rows, cols, vec)
}

func (CPUBackend) RMSNormInto(dst, x, weight []float32, eps float32) error {
	return ops.RMSNormInto(dst, x, weight, eps)
}

func (CPUBackend) ApplyRoPEHeads(v []float32, nHeads, headDim, pos int, freqBase float32) error {
	ops.ApplyRoPEHeads(v, nHeads, headDim, pos, freqBase)
	return nil
}

func (CPUBackend) ApplyRoPEHeadsNorm(v []float32, nHeads, headDim, pos int, freqBase float32) error {
	ops.ApplyRoPEHeadsNorm(v, nHeads, headDim, pos, freqBase)
	return nil
}

func (CPUBackend) SwiGLUInPlace(gate, up []float32) error {
	ops.SwiGLUInPlace(gate, up)
	return nil
}

func (CPUBackend) FFNSwiGLUCached(_, _, _ string, gateW, upW, downW, x, out []float32, embd, ffn int) error {
	gate := make([]float32, ffn)
	if err := ops.MatMulVecInto(gateW, ffn, embd, x, gate); err != nil {
		return err
	}

	up := make([]float32, ffn)
	if err := ops.MatMulVecInto(upW, ffn, embd, x, up); err != nil {
		return err
	}

	ops.SwiGLUInPlace(gate, up)

	return ops.MatMulVecInto(downW, embd, ffn, gate, out)
}

func (CPUBackend) FFNSwiGLUQ8_0Cached(_, _, _ string, gateRaw, upRaw, downRaw []byte, x, out []float32, embd, ffn int) error {
	gate := make([]float32, ffn)
	if err := ops.MatMulVecQ8_0Into(gateRaw, ffn, embd, x, gate); err != nil {
		return err
	}

	up := make([]float32, ffn)
	if err := ops.MatMulVecQ8_0Into(upRaw, ffn, embd, x, up); err != nil {
		return err
	}

	ops.SwiGLUInPlace(gate, up)

	return ops.MatMulVecQ8_0Into(downRaw, embd, ffn, gate, out)
}

func (CPUBackend) AttnFFNResidualCached(_, _, _, _, _ string, woW, ffnNorm, gateW, upW, downW, x, attn []float32, embd, attnDim, ffn int, eps float32) error {
	h := make([]float32, embd)
	if err := ops.MatMulVecInto(woW, embd, attnDim, attn, h); err != nil {
		return err
	}

	ops.AddInPlace(x, h)

	if err := ops.RMSNormInto(h, x, ffnNorm, eps); err != nil {
		return err
	}

	if err := (CPUBackend{}).FFNSwiGLUCached("", "", "", gateW, upW, downW, h, h, embd, ffn); err != nil {
		return err
	}

	ops.AddInPlace(x, h)

	return nil
}

func (CPUBackend) AttnFFNResidualQ8_0Cached(_, _, _, _, _ string, woRaw, gateRaw, upRaw, downRaw []byte, ffnNorm, x, attn []float32, embd, attnDim, ffn int, eps float32) error {
	h := make([]float32, embd)
	if err := ops.MatMulVecQ8_0Into(woRaw, embd, attnDim, attn, h); err != nil {
		return err
	}

	ops.AddInPlace(x, h)

	if err := ops.RMSNormInto(h, x, ffnNorm, eps); err != nil {
		return err
	}

	if err := (CPUBackend{}).FFNSwiGLUQ8_0Cached("", "", "", gateRaw, upRaw, downRaw, h, h, embd, ffn); err != nil {
		return err
	}

	ops.AddInPlace(x, h)

	return nil
}

func (CPUBackend) QKVRoPEAttentionCached(_, _, _, _, _ string, qW, kW, vW, qNorm, kNorm, h, cos, sin, attn, kOut, vOut []float32, embd, nHeads, nKVHeads, headDim, layer, kvPos, seqLen int, eps float32) error {
	_ = cos
	_ = sin
	_ = layer
	_ = kvPos
	_ = seqLen
	q := make([]float32, nHeads*headDim)
	if err := ops.MatMulVecInto(qW, nHeads*headDim, embd, h, q); err != nil {
		return err
	}

	k := make([]float32, nKVHeads*headDim)
	if err := ops.MatMulVecInto(kW, nKVHeads*headDim, embd, h, k); err != nil {
		return err
	}

	if err := ops.MatMulVecInto(vW, nKVHeads*headDim, embd, h, vOut); err != nil {
		return err
	}

	for hi := 0; hi < nHeads; hi++ {
		off := hi * headDim
		if err := ops.RMSNormInto(q[off:off+headDim], q[off:off+headDim], qNorm, eps); err != nil {
			return err
		}
	}

	for hi := 0; hi < nKVHeads; hi++ {
		off := hi * headDim
		if err := ops.RMSNormInto(k[off:off+headDim], k[off:off+headDim], kNorm, eps); err != nil {
			return err
		}
	}

	copy(kOut, k)
	copy(attn, q)

	return nil
}

func (CPUBackend) QKVRoPEAttentionQ8_0Cached(_, _, _, _, _ string, qRaw, kRaw, vRaw []byte, qNorm, kNorm, h, cos, sin, attn, kOut, vOut []float32, embd, nHeads, nKVHeads, headDim, layer, kvPos, seqLen int, eps float32) error {
	_ = cos
	_ = sin
	_ = layer
	_ = kvPos
	_ = seqLen
	q := make([]float32, nHeads*headDim)
	if err := ops.MatMulVecQ8_0Into(qRaw, nHeads*headDim, embd, h, q); err != nil {
		return err
	}

	k := make([]float32, nKVHeads*headDim)
	if err := ops.MatMulVecQ8_0Into(kRaw, nKVHeads*headDim, embd, h, k); err != nil {
		return err
	}

	if err := ops.MatMulVecQ8_0Into(vRaw, nKVHeads*headDim, embd, h, vOut); err != nil {
		return err
	}

	for hi := 0; hi < nHeads; hi++ {
		off := hi * headDim
		if err := ops.RMSNormInto(q[off:off+headDim], q[off:off+headDim], qNorm, eps); err != nil {
			return err
		}
	}

	for hi := 0; hi < nKVHeads; hi++ {
		off := hi * headDim
		if err := ops.RMSNormInto(k[off:off+headDim], k[off:off+headDim], kNorm, eps); err != nil {
			return err
		}
	}

	copy(kOut, k)
	copy(attn, q)

	return nil
}

func (CPUBackend) AttentionScoresInto(dst, q, k, v, scores []float32, seqLen, nHeads, nKVHeads, headDim int) error {
	return ops.AttentionScoresInto(dst, q, k, v, scores, seqLen, nHeads, nKVHeads, headDim)
}

func (CPUBackend) KVCacheInit(int, int, int, int, int) error {
	return nil
}

func (CPUBackend) KVCacheReset() {}

func (CPUBackend) KVCacheAppend(int, int, []float32, []float32) error {
	return nil
}

func (CPUBackend) AttentionScoresKV(int, []float32, []float32, int, int, int, int) error {
	return ErrKVCacheUnavailable
}

func (CPUBackend) Close() error {
	return nil
}
