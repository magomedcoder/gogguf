// Package gpuresid - shared fused GPU layer paths (§5-§6): QKV+RoPE+attention, WO+FFN+residual, FFN, and MoE experts.
// Architectures differ only in weight names, dimensions, and RoPE mode, so logic is not duplicated across model packages.
package gpuresid

import (
	"fmt"

	"github.com/magomedcoder/gogguf/pkg/format"
	"github.com/magomedcoder/gogguf/pkg/gpu"
	"github.com/magomedcoder/gogguf/pkg/ops"
	"github.com/magomedcoder/gogguf/pkg/weights"
)

// Tensors - layer weight names for fused paths.
// QNorm/KNorm empty - layer without QK-norm (Llama / Mistral)
type Tensors struct {
	AttnQ    string
	AttnK    string
	AttnV    string
	AttnOut  string
	FFNGate  string
	FFNUp    string
	FFNDown  string
	QNorm    string
	KNorm    string
	AttnNorm string
	FFNNorm  string
}

// Dims - layer dimensions
type Dims struct {
	Embd     int
	NHeads   int
	NKVHeads int
	HeadDim  int
	FFN      int
	Eps      float32
	RopeBase float32
}

// QDim - attention output size (n_heads * head_dim)
func (d Dims) QDim() int {
	return d.NHeads * d.HeadDim
}

// KVDim - K/V size for one token
func (d Dims) KVDim() int {
	return d.NKVHeads * d.HeadDim
}

// Runner runs fused layer paths on the backend
type Runner struct {
	w    *weights.Store
	g    gpu.Backend
	rope gpu.RoPEMode
	cos  []float32
	sin  []float32
}

// New creates a runner for the model with the given RoPE mode
func New(w *weights.Store, g gpu.Backend, rope gpu.RoPEMode) *Runner {
	return &Runner{w: w, g: g, rope: rope}
}

// QKVAttn: QKV + QK-norm + RoPE + KV append + attention on GPU.
// h == nil - hidden is resident on device: attn_norm from residual (§1), host buffer h is not read
func (r *Runner) QKVAttn(t Tensors, d Dims, qNorm, kNorm, attnNorm, h, attn, kOut, vOut []float32, layer, pos, kvPos, seqLen int) error {
	info, err := r.w.Info(t.AttnQ)
	if err != nil {
		return err
	}

	half := d.HeadDim / 2
	if cap(r.cos) < half {
		r.cos = make([]float32, half)
		r.sin = make([]float32, half)
	}

	cos, sin := r.cos[:half], r.sin[:half]
	ops.RoPECosSin(cos, sin, d.HeadDim, pos, d.RopeBase)

	// §3: supported quants use fused path natively, without host Floats
	if gpu.FusedQuantSupported(info.Type) {
		qRaw, kRaw, vRaw, err := r.raw3(t.AttnQ, t.AttnK, t.AttnV)
		if err != nil {
			return err
		}

		return r.g.QKVRoPEAttentionQuantCached(info.Type, r.rope, t.AttnQ, t.AttnK, t.AttnV, t.QNorm, t.KNorm, t.AttnNorm, qRaw, kRaw, vRaw, qNorm, kNorm, attnNorm, h, cos, sin, attn, kOut, vOut, d.Embd, d.NHeads, d.NKVHeads, d.HeadDim, layer, kvPos, seqLen, d.Eps)
	}

	if h == nil {
		return fmt.Errorf("gpuresid: layer %d: type %s does not support device residency", layer, info.Type)
	}

	// FP32 path exists only for NeoX + QK-norm (Qwen3)
	if r.rope != gpu.RoPENeoX || t.QNorm == "" || t.KNorm == "" {
		return fmt.Errorf("gpuresid: layer %d: type %s does not support fused QKV", layer, info.Type)
	}

	qW, kW, vW, err := r.floats3(t.AttnQ, t.AttnK, t.AttnV)
	if err != nil {
		return err
	}

	return r.g.QKVRoPEAttentionCached(t.AttnQ, t.AttnK, t.AttnV, t.QNorm, t.KNorm, qW, kW, vW, qNorm, kNorm, h, cos, sin, attn, kOut, vOut, d.Embd, d.NHeads, d.NKVHeads, d.HeadDim, layer, kvPos, seqLen, d.Eps)
}

// AttnFFN: WO + residual + ffn_norm + FFN + residual on GPU.
// x == nil - residual stays on device (§1)
func (r *Runner) AttnFFN(t Tensors, d Dims, ffnNorm, x, attn []float32) error {
	info, err := r.w.Info(t.AttnOut)
	if err != nil {
		return err
	}

	if gpu.FusedQuantSupported(info.Type) {
		woRaw, err := r.w.Raw(t.AttnOut)
		if err != nil {
			return err
		}

		gateRaw, upRaw, downRaw, err := r.raw3(t.FFNGate, t.FFNUp, t.FFNDown)
		if err != nil {
			return err
		}

		return r.g.AttnFFNResidualQuantCached(info.Type, t.AttnOut, t.FFNNorm, t.FFNGate, t.FFNUp, t.FFNDown, woRaw, gateRaw, upRaw, downRaw, ffnNorm, x, attn, d.Embd, d.QDim(), d.FFN, d.Eps)
	}

	if x == nil {
		return fmt.Errorf("gpuresid: type %s does not support device residency", info.Type)
	}

	woW, err := r.w.Floats(t.AttnOut)
	if err != nil {
		return err
	}

	gateW, upW, downW, err := r.floats3(t.FFNGate, t.FFNUp, t.FFNDown)
	if err != nil {
		return err
	}

	return r.g.AttnFFNResidualCached(t.AttnOut, t.FFNNorm, t.FFNGate, t.FFNUp, t.FFNDown, woW, ffnNorm, gateW, upW, downW, x, attn, d.Embd, d.QDim(), d.FFN, d.Eps)
}

// FFN: gate/up/down + SwiGLU on GPU
func (r *Runner) FFN(t Tensors, d Dims, x, out []float32) error {
	return r.ffn(false, t.FFNGate, t.FFNUp, t.FFNDown, x, out, d.Embd, d.FFN)
}

// FFNGeGLU: same fused FFN with GeGLU activation (Gemma)
func (r *Runner) FFNGeGLU(t Tensors, d Dims, x, out []float32) error {
	return r.ffn(true, t.FFNGate, t.FFNUp, t.FFNDown, x, out, d.Embd, d.FFN)
}

// FFNNamed - fused SwiGLU-FFN with arbitrary weight names (shared expert MoE)
func (r *Runner) FFNNamed(gateName, upName, downName string, x, out []float32, embd, ffn int) error {
	return r.ffn(false, gateName, upName, downName, x, out, embd, ffn)
}

func (r *Runner) ffn(geglu bool, gateName, upName, downName string, x, out []float32, embd, ffn int) error {
	info, err := r.w.Info(gateName)
	if err != nil {
		return err
	}

	if gpu.FusedQuantSupported(info.Type) {
		gateRaw, upRaw, downRaw, err := r.raw3(gateName, upName, downName)
		if err != nil {
			return err
		}

		if geglu {
			return r.g.FFNGeGLUQuantCached(info.Type, gateName, upName, downName, gateRaw, upRaw, downRaw, x, out, embd, ffn)
		}

		return r.g.FFNSwiGLUQuantCached(info.Type, gateName, upName, downName, gateRaw, upRaw, downRaw, x, out, embd, ffn)
	}

	if geglu {
		return fmt.Errorf("gpuresid: type %s does not support fused GeGLU", info.Type)
	}

	gateW, upW, downW, err := r.floats3(gateName, upName, downName)
	if err != nil {
		return err
	}

	return r.g.FFNSwiGLUCached(gateName, upName, downName, gateW, upW, downW, x, out, embd, ffn)
}

// MatMulInto - single matmul on GPU by weight name (§7: MLA projections, MoE router, lm_head). Result is written to out
func (r *Runner) MatMulInto(name string, rows, cols int, vec, out []float32) error {
	if len(out) < rows {
		return fmt.Errorf("gpuresid: %s: short out: %d < %d", name, len(out), rows)
	}

	info, err := r.w.Info(name)
	if err != nil {
		return err
	}

	var got []float32
	if gpu.FusedQuantSupported(info.Type) {
		raw, err := r.w.Raw(name)
		if err != nil {
			return err
		}

		got, err = r.matMulQuant(info.Type, name, raw, rows, cols, vec)
		if err != nil {
			return err
		}
	} else {
		f32, err := r.w.Floats(name)
		if err != nil {
			return err
		}

		if got, err = r.g.MatMulVecCached(name, f32, rows, cols, vec); err != nil {
			return err
		}
	}

	copy(out[:rows], got[:rows])

	return nil
}

func (r *Runner) matMulQuant(t format.GGML, name string, raw []byte, rows, cols int, vec []float32) ([]float32, error) {
	switch t {
	case format.GgmlQ8_0:
		return r.g.MatMulVecQ8_0Cached(name, raw, rows, cols, vec)
	case format.GgmlQ4_0:
		return r.g.MatMulVecQ4_0Cached(name, raw, rows, cols, vec)
	case format.GgmlQ4_K:
		return r.g.MatMulVecQ4_KCached(name, raw, rows, cols, vec)
	case format.GgmlQ5_K:
		return r.g.MatMulVecQ5_KCached(name, raw, rows, cols, vec)
	case format.GgmlQ6_K:
		return r.g.MatMulVecQ6_KCached(name, raw, rows, cols, vec)
	default:
		return nil, fmt.Errorf("gpuresid: %s: type %s has no GPU matmul", name, t)
	}
}

// ExpertFFN: FFN of one MoE expert on GPU (§6).
// Expert weights are one tensor [n_exp][rows][cols]; only expert slice goes to device as "<tensor>#<expert>", without dequantizing full matrix on host
func (r *Runner) ExpertFFN(gateExps, upExps, downExps string, expert int, x, out []float32, embd, ffn int) error {
	gate, err := r.expertSlice(gateExps, expert, ffn, embd)
	if err != nil {
		return err
	}

	up, err := r.expertSlice(upExps, expert, ffn, embd)
	if err != nil {
		return err
	}

	down, err := r.expertSlice(downExps, expert, embd, ffn)
	if err != nil {
		return err
	}

	if gate.t != up.t || gate.t != down.t {
		return fmt.Errorf("gpuresid: experts of different types: %s/%s/%s", gate.t, up.t, down.t)
	}

	return r.g.FFNSwiGLUQuantCached(gate.t, gate.name, up.name, down.name, gate.raw, up.raw, down.raw, x, out, embd, ffn)
}

// MoESupported reports whether experts can run on GPU without host dequantization
func (r *Runner) MoESupported(names ...string) bool {
	for _, name := range names {
		info, err := r.w.Info(name)
		if err != nil || !gpu.FusedQuantSupported(info.Type) {
			return false
		}
	}

	return true
}

// LogitsDevice: out_norm + lm_head on GPU over resident hidden (§2)
func (r *Runner) LogitsDevice(normName string, norm []float32, headName string, logits []float32, vocab, embd int, eps float32) error {
	info, err := r.w.Info(headName)
	if err != nil {
		return err
	}

	t := info.Type
	var raw []byte
	var f32 []float32
	if gpu.FusedQuantSupported(t) {
		if raw, err = r.w.Raw(headName); err != nil {
			return err
		}
	} else {
		if f32, err = r.w.Floats(headName); err != nil {
			return err
		}

		t = format.GgmlFloat32
	}

	return r.g.LogitsFromDevice(t, normName, norm, headName, raw, f32, logits, vocab, embd, eps)
}

type expertWeights struct {
	name string
	raw  []byte
	t    format.GGML
}

// expertSlice returns raw slice of expert from tensor [n_exp][rows][cols]
func (r *Runner) expertSlice(name string, expert, rows, cols int) (expertWeights, error) {
	info, err := r.w.Info(name)
	if err != nil {
		return expertWeights{}, err
	}

	if !gpu.FusedQuantSupported(info.Type) {
		return expertWeights{}, fmt.Errorf("gpuresid: %s: type %s not supported on fused path", name, info.Type)
	}

	bytesPerExpert, err := QuantBytes(info.Type, rows, cols)
	if err != nil {
		return expertWeights{}, fmt.Errorf("gpuresid: %s: %w", name, err)
	}

	raw, err := r.w.Raw(name)
	if err != nil {
		return expertWeights{}, err
	}

	off := expert * bytesPerExpert
	if off < 0 || off+bytesPerExpert > len(raw) {
		return expertWeights{}, fmt.Errorf("gpuresid: %s: expert %d out of tensor bounds (%d bytes)", name, expert, len(raw))
	}

	return expertWeights{
		name: fmt.Sprintf("%s#%d", name, expert),
		raw:  raw[off : off+bytesPerExpert],
		t:    info.Type,
	}, nil
}

func (r *Runner) raw3(a, b, c string) (ra, rb, rc []byte, err error) {
	if ra, err = r.w.Raw(a); err != nil {
		return nil, nil, nil, err
	}

	if rb, err = r.w.Raw(b); err != nil {
		return nil, nil, nil, err
	}

	if rc, err = r.w.Raw(c); err != nil {
		return nil, nil, nil, err
	}

	return ra, rb, rc, nil
}

func (r *Runner) floats3(a, b, c string) (fa, fb, fc []float32, err error) {
	if fa, err = r.w.Floats(a); err != nil {
		return nil, nil, nil, err
	}

	if fb, err = r.w.Floats(b); err != nil {
		return nil, nil, nil, err
	}

	if fc, err = r.w.Floats(c); err != nil {
		return nil, nil, nil, err
	}

	return fa, fb, fc, nil
}

// LayerQuant returns the common matmul weight type for a layer or format.GgmlFloat32 when types differ (that layer uses the FP32/discrete path)
func LayerQuant(w *weights.Store, names ...string) format.GGML {
	kind := format.GgmlFloat32
	for i, name := range names {
		info, err := w.Info(name)
		if err != nil || (i > 0 && info.Type != kind) {
			return format.GgmlFloat32
		}

		kind = info.Type
	}

	return kind
}
