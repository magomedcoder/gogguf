package gpu

import (
	"fmt"

	"github.com/magomedcoder/gogguf/pkg/format"
	"github.com/magomedcoder/gogguf/pkg/ops"
)

// CPUBackend runs matmul on CPU via pkg/ops
type CPUBackend struct{}

var _ Backend = CPUBackend{}

func (CPUBackend) Name() string {
	return "CPU"
}

// VRAMInfo: CPU backend has no VRAM
func (CPUBackend) VRAMInfo() (used, total uint64, err error) {
	return 0, 0, nil
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

func (CPUBackend) MatMulVecQ4_0Cached(_ string, raw []byte, rows, cols int, vec []float32) ([]float32, error) {
	return ops.MatMulVecQ4_0(raw, rows, cols, vec)
}

func (CPUBackend) MatMulVecQ4_KCached(_ string, raw []byte, rows, cols int, vec []float32) ([]float32, error) {
	return ops.MatMulVecQ4_K(raw, rows, cols, vec)
}

func (CPUBackend) MatMulVecQ5_KCached(_ string, raw []byte, rows, cols int, vec []float32) ([]float32, error) {
	return ops.MatMulVecQ5_K(raw, rows, cols, vec)
}

func (CPUBackend) MatMulVecQ6_KCached(_ string, raw []byte, rows, cols int, vec []float32) ([]float32, error) {
	return ops.MatMulVecQ6_K(raw, rows, cols, vec)
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

	for hi := range nHeads {
		off := hi * headDim
		if err := ops.RMSNormInto(q[off:off+headDim], q[off:off+headDim], qNorm, eps); err != nil {
			return err
		}
	}

	for hi := range nKVHeads {
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

	for hi := range nHeads {
		off := hi * headDim
		if err := ops.RMSNormInto(q[off:off+headDim], q[off:off+headDim], qNorm, eps); err != nil {
			return err
		}
	}

	for hi := range nKVHeads {
		off := hi * headDim
		if err := ops.RMSNormInto(k[off:off+headDim], k[off:off+headDim], kNorm, eps); err != nil {
			return err
		}
	}

	copy(kOut, k)
	copy(attn, q)

	return nil
}

// matMulQuantInto - CPU equivalent of quantized matmul for fused paths
func matMulQuantInto(t format.GGML, raw []byte, rows, cols int, vec, out []float32) error {
	switch t {
	case format.GgmlQ8_0:
		return ops.MatMulVecQ8_0Into(raw, rows, cols, vec, out)
	case format.GgmlQ4_0:
		return ops.MatMulVecQ4_0Into(raw, rows, cols, vec, out)
	case format.GgmlQ4_K:
		return ops.MatMulVecQ4_KInto(raw, rows, cols, vec, out)
	case format.GgmlQ5_K:
		return ops.MatMulVecQ5_KInto(raw, rows, cols, vec, out)
	case format.GgmlQ6_K:
		return ops.MatMulVecQ6_KInto(raw, rows, cols, vec, out)
	default:
		return fmt.Errorf("gpu: type %s not supported on fused path", t)
	}
}

func (CPUBackend) FFNSwiGLUQuantCached(t format.GGML, _, _, _ string, gateRaw, upRaw, downRaw []byte, x, out []float32, embd, ffn int) error {
	gate := make([]float32, ffn)
	if err := matMulQuantInto(t, gateRaw, ffn, embd, x, gate); err != nil {
		return err
	}

	up := make([]float32, ffn)
	if err := matMulQuantInto(t, upRaw, ffn, embd, x, up); err != nil {
		return err
	}

	ops.SwiGLUInPlace(gate, up)

	return matMulQuantInto(t, downRaw, embd, ffn, gate, out)
}

func (CPUBackend) FFNGeGLUQuantCached(t format.GGML, _, _, _ string, gateRaw, upRaw, downRaw []byte, x, out []float32, embd, ffn int) error {
	gate := make([]float32, ffn)
	if err := matMulQuantInto(t, gateRaw, ffn, embd, x, gate); err != nil {
		return err
	}

	up := make([]float32, ffn)
	if err := matMulQuantInto(t, upRaw, ffn, embd, x, up); err != nil {
		return err
	}

	ops.GeGLUInPlace(gate, up)

	return matMulQuantInto(t, downRaw, embd, ffn, gate, out)
}

func (CPUBackend) AttnFFNResidualQuantCached(t format.GGML, _, _, _, _, _ string, woRaw, gateRaw, upRaw, downRaw []byte, ffnNorm, x, attn []float32, embd, attnDim, ffn int, eps float32) error {
	h := make([]float32, embd)
	if err := matMulQuantInto(t, woRaw, embd, attnDim, attn, h); err != nil {
		return err
	}

	ops.AddInPlace(x, h)

	if err := ops.RMSNormInto(h, x, ffnNorm, eps); err != nil {
		return err
	}

	if err := (CPUBackend{}).FFNSwiGLUQuantCached(t, "", "", "", gateRaw, upRaw, downRaw, h, h, embd, ffn); err != nil {
		return err
	}

	ops.AddInPlace(x, h)

	return nil
}

// QKVRoPEAttentionQuantCached: CPU equivalent of QKV+QK-norm (RoPE and attention remain caller's job). empty qNorm/kNorm - layer without QK-norm.
func (CPUBackend) QKVRoPEAttentionQuantCached(t format.GGML, _ RoPEMode, _, _, _, _, _, _ string, qRaw, kRaw, vRaw []byte, qNorm, kNorm, _, h, _, _, attn, kOut, vOut []float32, embd, nHeads, nKVHeads, headDim, _, _, _ int, eps float32) error {
	q := make([]float32, nHeads*headDim)
	if err := matMulQuantInto(t, qRaw, nHeads*headDim, embd, h, q); err != nil {
		return err
	}

	k := make([]float32, nKVHeads*headDim)
	if err := matMulQuantInto(t, kRaw, nKVHeads*headDim, embd, h, k); err != nil {
		return err
	}

	if err := matMulQuantInto(t, vRaw, nKVHeads*headDim, embd, h, vOut); err != nil {
		return err
	}

	if len(qNorm) > 0 {
		for hi := range nHeads {
			off := hi * headDim
			if err := ops.RMSNormInto(q[off:off+headDim], q[off:off+headDim], qNorm, eps); err != nil {
				return err
			}
		}
	}

	if len(kNorm) > 0 {
		for hi := range nKVHeads {
			off := hi * headDim
			if err := ops.RMSNormInto(k[off:off+headDim], k[off:off+headDim], kNorm, eps); err != nil {
				return err
			}
		}
	}

	copy(kOut, k)
	copy(attn, q)

	return nil
}

// HiddenResident: CPU backend keeps hidden in host memory - residency not needed
func (CPUBackend) HiddenResident() bool {
	return false
}

func (CPUBackend) HiddenActive() bool {
	return false
}

func (CPUBackend) HiddenUpload([]float32) error {
	return ErrHiddenUnavailable
}

func (CPUBackend) HiddenDownload([]float32) error {
	return ErrHiddenUnavailable
}

func (CPUBackend) LogitsFromDevice(format.GGML, string, []float32, string, []byte, []float32, []float32, int, int, float32) error {
	return ErrHiddenUnavailable
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

func (CPUBackend) KVCacheAppendN(int, int, []float32, []float32, int) error {
	return nil
}

func (CPUBackend) AttentionScoresKV(int, []float32, []float32, int, int, int, int) error {
	return ErrKVCacheUnavailable
}

func (CPUBackend) Close() error {
	return nil
}
