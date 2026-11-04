package gpuresid

import (
	"fmt"
	"github.com/magomedcoder/gogguf/pkg/dtype"

	"github.com/magomedcoder/gogguf/pkg/compute"
	"github.com/magomedcoder/gogguf/pkg/device"
	"github.com/magomedcoder/gogguf/pkg/weights"
)

// Tensors - layer weight names for fused paths.
// QNorm/KNorm empty - layer without QK-norm (Llama / Mistral)
// PostAttnNorm/PostFFNNorm - Gemma2 post-norms (empty = skip)
// AttnQKV set - Phi-3 fused attn_qkv; Q/K/V are row-slices (AttnQ/K/V may be empty)
// FFNGateUp set - Phi-3 merged ffn_up gate||up; FFNGate/FFNUp may be empty
type Tensors struct {
	AttnQ        string
	AttnK        string
	AttnV        string
	AttnQKV      string
	AttnOut      string
	FFNGate      string
	FFNUp        string
	FFNGateUp    string
	FFNDown      string
	QNorm        string
	KNorm        string
	AttnNorm     string
	FFNNorm      string
	PostAttnNorm string
	PostFFNNorm  string
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

	// Optional partial / LongRoPE (Phi-3).
	// NRot 0 = full HeadDim.
	// RopeScale zero fields = plain RoPECosSin.
	// When NRot < HeadDim, cos/sin beyond NRot/2 are identity-padded (cos=1, sin=0) so the existing rope_heads kernel (half=HeadDim/2) leaves non-rotated pairs unchanged.
	NRot      int
	RopeScale compute.RoPEScale
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
	g    device.Backend
	rope device.RoPEMode
	cos  []float32
	sin  []float32
	tmp  []float32 // scratch for releasing residency after logits-only passes
}

// New creates a runner for the model with the given RoPE mode
func New(w *weights.Store, g device.Backend, rope device.RoPEMode) *Runner {
	return &Runner{w: w, g: g, rope: rope}
}

// QKVAttn QKV + QK-norm + RoPE + KV append + attention on GPU.
// h == nil - hidden is resident on device: attn_norm from residual, host buffer h is not read.
// When d.NRot/d.RopeScale are set, cos/sin come from RoPECosSinScaled (LongRoPE / partial); pairs beyond NRot/2 are identity-padded for the headDim/2 rope_heads launch.
func (r *Runner) QKVAttn(t Tensors, d Dims, qNorm, kNorm, attnNorm, h, attn, kOut, vOut []float32, layer, pos, kvPos, seqLen int) error {
	qName, kName, vName := t.AttnQ, t.AttnK, t.AttnV
	var qRaw, kRaw, vRaw []byte
	var infoType dtype.DType

	if t.AttnQKV != "" {
		q, k, v, err := r.qkvFusedSlices(t.AttnQKV, d.QDim(), d.KVDim())
		if err != nil {
			return err
		}
		qName, kName, vName = q.name, k.name, v.name
		qRaw, kRaw, vRaw = q.raw, k.raw, v.raw
		infoType = q.t
	} else {
		info, err := r.w.Info(t.AttnQ)
		if err != nil {
			return err
		}
		infoType = info.Type
		if device.FusedQuantSupported(infoType) {
			qRaw, kRaw, vRaw, err = r.raw3(t.AttnQ, t.AttnK, t.AttnV)
			if err != nil {
				return err
			}
		}
	}

	cos, sin := r.ropeTables(d, pos)

	// supported quants use fused path natively, without host Floats
	if device.FusedQuantSupported(infoType) {
		return r.g.QKVRoPEAttentionQuantCached(infoType, r.rope, qName, kName, vName, t.QNorm, t.KNorm, t.AttnNorm, qRaw, kRaw, vRaw, qNorm, kNorm, attnNorm, h, cos, sin, attn, kOut, vOut, d.Embd, d.NHeads, d.NKVHeads, d.HeadDim, layer, kvPos, seqLen, d.Eps)
	}

	if h == nil {
		return fmt.Errorf("gpuresid: layer %d: type %s does not support device residency", layer, infoType)
	}

	// FP32 path exists only for NeoX + QK-norm (Qwen3); fused attn_qkv has no FP32 split here
	if t.AttnQKV != "" || r.rope != device.RoPENeoX || t.QNorm == "" || t.KNorm == "" {
		return fmt.Errorf("gpuresid: layer %d: type %s does not support fused QKV", layer, infoType)
	}

	qW, kW, vW, err := r.floats3(t.AttnQ, t.AttnK, t.AttnV)
	if err != nil {
		return err
	}

	return r.g.QKVRoPEAttentionCached(t.AttnQ, t.AttnK, t.AttnV, t.QNorm, t.KNorm, qW, kW, vW, qNorm, kNorm, h, cos, sin, attn, kOut, vOut, d.Embd, d.NHeads, d.NKVHeads, d.HeadDim, layer, kvPos, seqLen, d.Eps)
}

// ropeTables builds cos/sin of length HeadDim/2 for the GPU rope_heads kernel
func (r *Runner) ropeTables(d Dims, pos int) (cos, sin []float32) {
	half := d.HeadDim / 2
	if half < 0 {
		half = 0
	}
	if cap(r.cos) < half {
		r.cos = make([]float32, half)
		r.sin = make([]float32, half)
	}
	cos, sin = r.cos[:half], r.sin[:half]

	nRot := d.NRot
	if nRot <= 0 || nRot > d.HeadDim {
		nRot = d.HeadDim
	}

	if nRot%2 != 0 {
		nRot--
	}

	nHalf := nRot / 2
	if nHalf > half {
		nHalf = half
	}

	if nHalf > 0 {
		compute.RoPECosSinScaled(cos[:nHalf], sin[:nHalf], nRot, pos, d.RopeBase, d.RopeScale)
	}

	// Identity pad so pairs beyond nRot stay unchanged under headDim/2 NeoX rope
	for i := nHalf; i < half; i++ {
		cos[i] = 1
		sin[i] = 0
	}

	return cos, sin
}

// AttnFFN WO + residual + ffn_norm + FFN + residual on GPU.
// x == nil - residual stays on device
func (r *Runner) AttnFFN(t Tensors, d Dims, ffnNorm, x, attn []float32) error {
	return r.attnFFN(false, t, d, ffnNorm, x, attn)
}

// AttnFFNGeGLU like AttnFFN but GeGLU activation (Gemma1; no post-norms)
func (r *Runner) AttnFFNGeGLU(t Tensors, d Dims, ffnNorm, x, attn []float32) error {
	return r.attnFFNGeGLU(t, d, ffnNorm, nil, nil, x, attn)
}

// AttnFFNGeGLUPostNorm like AttnFFNGeGLU with Gemma2 post-attention / post-FFN RMSNorm before residual add.
func (r *Runner) AttnFFNGeGLUPostNorm(t Tensors, d Dims, ffnNorm, postAttn, postFFN, x, attn []float32) error {
	return r.attnFFNGeGLU(t, d, ffnNorm, postAttn, postFFN, x, attn)
}

func (r *Runner) attnFFN(geglu bool, t Tensors, d Dims, ffnNorm, x, attn []float32) error {
	if geglu {
		return r.attnFFNGeGLU(t, d, ffnNorm, nil, nil, x, attn)
	}

	info, err := r.w.Info(t.AttnOut)
	if err != nil {
		return err
	}

	if device.FusedQuantSupported(info.Type) {
		woRaw, err := r.w.Raw(t.AttnOut)
		if err != nil {
			return err
		}

		gateName, upName, gateRaw, upRaw, err := r.ffnGateUpRaw(t, d.FFN)
		if err != nil {
			return err
		}
		downRaw, err := r.w.Raw(t.FFNDown)
		if err != nil {
			return err
		}

		return r.g.AttnFFNResidualQuantCached(info.Type, t.AttnOut, t.FFNNorm, gateName, upName, t.FFNDown, woRaw, gateRaw, upRaw, downRaw, ffnNorm, x, attn, d.Embd, d.QDim(), d.FFN, d.Eps)
	}

	if x == nil || t.FFNGateUp != "" {
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

func (r *Runner) attnFFNGeGLU(t Tensors, d Dims, ffnNorm, postAttn, postFFN, x, attn []float32) error {
	info, err := r.w.Info(t.AttnOut)
	if err != nil {
		return err
	}

	if device.FusedQuantSupported(info.Type) {
		woRaw, err := r.w.Raw(t.AttnOut)
		if err != nil {
			return err
		}

		gateRaw, upRaw, downRaw, err := r.raw3(t.FFNGate, t.FFNUp, t.FFNDown)
		if err != nil {
			return err
		}

		return r.g.AttnFFNResidualGeGLUQuantCached(info.Type, t.AttnOut, t.FFNNorm, t.PostAttnNorm, t.PostFFNNorm, t.FFNGate, t.FFNUp, t.FFNDown, woRaw, gateRaw, upRaw, downRaw, ffnNorm, postAttn, postFFN, x, attn, d.Embd, d.QDim(), d.FFN, d.Eps)
	}

	if x == nil {
		return fmt.Errorf("gpuresid: type %s does not support device residency", info.Type)
	}

	return fmt.Errorf("gpuresid: type %s does not support fused AttnFFN GeGLU", info.Type)
}

// FFN gate/up/down + SwiGLU on GPU
func (r *Runner) FFN(t Tensors, d Dims, x, out []float32) error {
	return r.ffn(false, t.FFNGate, t.FFNUp, t.FFNDown, x, out, d.Embd, d.FFN)
}

// FFNGeGLU same fused FFN with GeGLU activation (Gemma)
func (r *Runner) FFNGeGLU(t Tensors, d Dims, x, out []float32) error {
	return r.ffn(true, t.FFNGate, t.FFNUp, t.FFNDown, x, out, d.Embd, d.FFN)
}

// FFNNamed fused SwiGLU-FFN with arbitrary weight names (shared expert MoE)
func (r *Runner) FFNNamed(gateName, upName, downName string, x, out []float32, embd, ffn int) error {
	return r.ffn(false, gateName, upName, downName, x, out, embd, ffn)
}

func (r *Runner) ffn(geglu bool, gateName, upName, downName string, x, out []float32, embd, ffn int) error {
	info, err := r.w.Info(gateName)
	if err != nil {
		return err
	}

	if device.FusedQuantSupported(info.Type) {
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

// MatMulInto single matmul on GPU by weight name (MLA projections, MoE router, lm_head). Result is written to out
func (r *Runner) MatMulInto(name string, rows, cols int, vec, out []float32) error {
	if len(out) < rows {
		return fmt.Errorf("gpuresid: %s: short out: %d < %d", name, len(out), rows)
	}

	info, err := r.w.Info(name)
	if err != nil {
		return err
	}

	var got []float32
	if device.FusedQuantSupported(info.Type) {
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

func (r *Runner) matMulQuant(t dtype.DType, name string, raw []byte, rows, cols int, vec []float32) ([]float32, error) {
	switch t {
	case dtype.Q8_0:
		return r.g.MatMulVecQ8_0Cached(name, raw, rows, cols, vec)
	case dtype.Q4_0:
		return r.g.MatMulVecQ4_0Cached(name, raw, rows, cols, vec)
	case dtype.Q4_K:
		return r.g.MatMulVecQ4_KCached(name, raw, rows, cols, vec)
	case dtype.Q5_K:
		return r.g.MatMulVecQ5_KCached(name, raw, rows, cols, vec)
	case dtype.Q6_K:
		return r.g.MatMulVecQ6_KCached(name, raw, rows, cols, vec)
	case dtype.Q4_1:
		return r.g.MatMulVecQ4_1Cached(name, raw, rows, cols, vec)
	case dtype.Q5_0:
		return r.g.MatMulVecQ5_0Cached(name, raw, rows, cols, vec)
	case dtype.Q5_1:
		return r.g.MatMulVecQ5_1Cached(name, raw, rows, cols, vec)
	case dtype.Q2_K:
		return r.g.MatMulVecQ2_KCached(name, raw, rows, cols, vec)
	case dtype.Q3_K:
		return r.g.MatMulVecQ3_KCached(name, raw, rows, cols, vec)
	case dtype.Q8_K:
		return r.g.MatMulVecQ8_KCached(name, raw, rows, cols, vec)
	default:
		return nil, fmt.Errorf("gpuresid: %s: type %s has no GPU matmul", name, t)
	}
}

// MoESupported reports whether experts can run on GPU without host dequantization
func (r *Runner) MoESupported(names ...string) bool {
	for _, name := range names {
		info, err := r.w.Info(name)
		if err != nil || !device.FusedQuantSupported(info.Type) {
			return false
		}
	}

	return true
}

// SoftcapLogits applies Gemma2 final-logit softcap on a host logits buffer
func (r *Runner) SoftcapLogits(logits []float32, softcap float32) error {
	return r.g.SoftcapInPlace(logits, softcap)
}

// LogitsDevice out_norm + lm_head on GPU over resident hidden
func (r *Runner) LogitsDevice(normName string, norm []float32, headName string, logits []float32, vocab, embd int, eps float32) error {
	info, err := r.w.Info(headName)
	if err != nil {
		return err
	}

	t := info.Type
	var raw []byte
	var f32 []float32
	if device.FusedQuantSupported(t) {
		if raw, err = r.w.Raw(headName); err != nil {
			return err
		}
	} else {
		if f32, err = r.w.Floats(headName); err != nil {
			return err
		}

		t = dtype.F32
	}

	return r.g.LogitsFromDevice(t, normName, norm, headName, raw, f32, logits, vocab, embd, eps)
}

// LogitsDeviceLayerNorm LayerNorm+bias(resident hidden) + lm_head on GPU (Phi-2)
func (r *Runner) LogitsDeviceLayerNorm(normWName, normBName string, normW, normB []float32, headName string, logits []float32, vocab, embd int, eps float32) error {
	info, err := r.w.Info(headName)
	if err != nil {
		return err
	}

	t := info.Type
	var raw []byte
	var f32 []float32
	if device.FusedQuantSupported(t) {
		if raw, err = r.w.Raw(headName); err != nil {
			return err
		}
	} else {
		if f32, err = r.w.Floats(headName); err != nil {
			return err
		}

		t = dtype.F32
	}

	return r.g.LogitsFromDeviceLayerNorm(t, normWName, normBName, normW, normB, headName, raw, f32, logits, vocab, embd, eps)
}

// LogitsFromHostLayerNorm uploads hidden x then LayerNorm+bias + lm_head on GPU (Phi-2 non-resident path)
func (r *Runner) LogitsFromHostLayerNorm(x []float32, normWName, normBName string, normW, normB []float32, headName string, logits []float32, vocab, embd int, eps float32) error {
	if len(x) < embd {
		return fmt.Errorf("gpuresid: LogitsFromHostLayerNorm: short hidden %d < %d", len(x), embd)
	}

	if err := r.g.HiddenUpload(x[:embd]); err != nil {
		return err
	}

	err := r.LogitsDeviceLayerNorm(normWName, normBName, normW, normB, headName, logits, vocab, embd, eps)
	r.releaseHidden(embd)

	return err
}

type expertWeights struct {
	name string
	raw  []byte
	t    dtype.DType
}

// expertSlice returns raw slice of expert from tensor [n_exp][rows][cols]
func (r *Runner) expertSlice(name string, expert, rows, cols int) (expertWeights, error) {
	return r.rowSlice(name, fmt.Sprintf("%s#%d", name, expert), expert*rows, rows, cols)
}

// qkvFusedSlices splits Phi-3 attn_qkv [qDim+2*kvDim, embd] into Q/K/V row slices.
func (r *Runner) qkvFusedSlices(name string, qDim, kvDim int) (q, k, v expertWeights, err error) {
	info, err := r.w.Info(name)
	if err != nil {
		return expertWeights{}, expertWeights{}, expertWeights{}, err
	}

	if !device.FusedQuantSupported(info.Type) {
		return expertWeights{}, expertWeights{}, expertWeights{}, fmt.Errorf("gpuresid: %s: type %s not supported on fused path", name, info.Type)
	}

	totalRows := qDim + 2*kvDim
	raw, err := r.w.Raw(name)
	if err != nil {
		return expertWeights{}, expertWeights{}, expertWeights{}, err
	}

	if totalRows <= 0 || len(raw)%totalRows != 0 {
		return expertWeights{}, expertWeights{}, expertWeights{}, fmt.Errorf("gpuresid: %s: fused qkv size %d not divisible by %d rows", name, len(raw), totalRows)
	}

	bytesPerRow := len(raw) / totalRows
	qBytes := qDim * bytesPerRow
	kvBytes := kvDim * bytesPerRow
	if qBytes+2*kvBytes != len(raw) {
		return expertWeights{}, expertWeights{}, expertWeights{}, fmt.Errorf("gpuresid: %s: fused qkv layout mismatch", name)
	}

	return expertWeights{name: name + "#q", raw: raw[:qBytes], t: info.Type}, expertWeights{name: name + "#k", raw: raw[qBytes : qBytes+kvBytes], t: info.Type}, expertWeights{name: name + "#v", raw: raw[qBytes+kvBytes:], t: info.Type}, nil
}

func (r *Runner) ffnGateUpRaw(t Tensors, ffn int) (gateName, upName string, gateRaw, upRaw []byte, err error) {
	if t.FFNGateUp != "" {
		gate, up, err := r.gateUpFusedSlices(t.FFNGateUp, ffn)
		if err != nil {
			return "", "", nil, nil, err
		}

		return gate.name, up.name, gate.raw, up.raw, nil
	}

	if gateRaw, err = r.w.Raw(t.FFNGate); err != nil {
		return "", "", nil, nil, err
	}

	if upRaw, err = r.w.Raw(t.FFNUp); err != nil {
		return "", "", nil, nil, err
	}

	return t.FFNGate, t.FFNUp, gateRaw, upRaw, nil
}

func (r *Runner) gateUpFusedSlices(name string, ffn int) (gate, up expertWeights, err error) {
	info, err := r.w.Info(name)
	if err != nil {
		return expertWeights{}, expertWeights{}, err
	}

	if !device.FusedQuantSupported(info.Type) {
		return expertWeights{}, expertWeights{}, fmt.Errorf("gpuresid: %s: type %s not supported on fused path", name, info.Type)
	}

	raw, err := r.w.Raw(name)
	if err != nil {
		return expertWeights{}, expertWeights{}, err
	}

	totalRows := 2 * ffn
	if totalRows <= 0 || len(raw)%totalRows != 0 {
		return expertWeights{}, expertWeights{}, fmt.Errorf("gpuresid: %s: merged ffn_up size %d not divisible by %d rows", name, len(raw), totalRows)
	}
	half := len(raw) / 2
	return expertWeights{name: name + "#gate", raw: raw[:half], t: info.Type}, expertWeights{name: name + "#up", raw: raw[half:], t: info.Type}, nil
}

// rowSlice returns a contiguous row-range slice of a quantized [rows_total][cols] tensor
func (r *Runner) rowSlice(name, sliceName string, row0, rows, cols int) (expertWeights, error) {
	info, err := r.w.Info(name)
	if err != nil {
		return expertWeights{}, err
	}

	if !device.FusedQuantSupported(info.Type) {
		return expertWeights{}, fmt.Errorf("gpuresid: %s: type %s not supported on fused path", name, info.Type)
	}

	raw, err := r.w.Raw(name)
	if err != nil {
		return expertWeights{}, err
	}

	if cols <= 0 {
		// MoE expert path always passes cols; generic callers may omit
		return expertWeights{}, fmt.Errorf("gpuresid: %s: cols required", name)
	}

	bytesPerRowBlock, err := QuantBytes(info.Type, 1, cols)
	if err != nil {
		return expertWeights{}, fmt.Errorf("gpuresid: %s: %w", name, err)
	}

	bytesPerSlice, err := QuantBytes(info.Type, rows, cols)
	if err != nil {
		return expertWeights{}, fmt.Errorf("gpuresid: %s: %w", name, err)
	}

	off := row0 * bytesPerRowBlock
	if off < 0 || off+bytesPerSlice > len(raw) {
		return expertWeights{}, fmt.Errorf("gpuresid: %s: rows [%d,%d) out of tensor bounds (%d bytes)", name, row0, row0+rows, len(raw))
	}

	return expertWeights{
		name: sliceName,
		raw:  raw[off : off+bytesPerSlice],
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

// LayerQuant returns the common matmul weight type for a layer or dtype.F32 when types differ (that layer uses the FP32/discrete path)
func LayerQuant(w *weights.Store, names ...string) dtype.DType {
	kind := dtype.F32
	for i, name := range names {
		info, err := w.Info(name)
		if err != nil || (i > 0 && info.Type != kind) {
			return dtype.F32
		}

		kind = info.Type
	}

	return kind
}
