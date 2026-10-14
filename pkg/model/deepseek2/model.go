package deepseek2

import (
	"fmt"
	"github.com/magomedcoder/gogguf/pkg/dtype"
	"math"

	"github.com/magomedcoder/gogguf/pkg/compute"
	"github.com/magomedcoder/gogguf/pkg/device"
	"github.com/magomedcoder/gogguf/pkg/exec/dispatch"
	"github.com/magomedcoder/gogguf/pkg/exec/gpuresid"
	"github.com/magomedcoder/gogguf/pkg/exec/moe"
	"github.com/magomedcoder/gogguf/pkg/weights"
)

// Model - DeepSeek-V2/V3: absorbed MLA + mixture of experts
type Model struct {
	cfg          Config
	weights      *weights.Store
	dev          device.Backend
	ngl          int
	fused        *gpuresid.Runner
	moeGPU       []bool
	cache        *MLACache
	scratch      scratch
	layerNorms   []layerNorms
	layerTensors []layerTensors
	outNorm      []float32
	lmHeadName   string

	// device MLA (absorbed attention + latent KV cache on GPU); nil = CPU attention
	gpuMaxSeq int
	mlaDev    device.MLABackend
	mlaParams device.MLAParams
	mlaStale  bool // device latent cache missed a token: GPU MLA stays off until ResetCache
}

// Load creates DeepSeek2 from weights. GPU runs MLA projections, FFN, and experts.
// With a device.MLABackend (CUDA) absorbed MLA attention also runs on GPU over a latent KV cache (kvLora+rope K, kvLora V) for the first ngl layers; k_b/v_b are uploaded once as FP32 device buffers.
// CPU attention remains the fallback (error / unsupported backend / seq > gpuMaxSeq).
// MLA / MoE: serial only - n_batch is accepted for API symmetry and forced to 1
func Load(w *weights.Store, g device.Backend, ngl, gpuMaxSeq, nBatch int) (*Model, error) {
	_ = nBatch // n_batch=1: batch prefill is not implemented for absorbed MLA

	cfg, err := ParseConfig(w.Reader())
	if err != nil {
		return nil, err
	}

	layerNorms, outNorm, err := loadNormWeights(w, cfg)
	if err != nil {
		return nil, err
	}

	layerTensors, err := loadLayerTensors(w, cfg)
	if err != nil {
		return nil, err
	}

	lmHeadName, err := resolveLMHeadName(w)
	if err != nil {
		return nil, err
	}

	if ngl > cfg.NumLayers {
		ngl = cfg.NumLayers
	}

	m := &Model{
		cfg:          cfg,
		weights:      w,
		dev:          g,
		ngl:          ngl,
		cache:        NewMLACache(cfg),
		scratch:      newScratch(cfg),
		layerNorms:   layerNorms,
		layerTensors: layerTensors,
		outNorm:      outNorm,
		lmHeadName:   lmHeadName,
		gpuMaxSeq:    gpuMaxSeq,
	}

	m.initFused()
	m.initMLAGPU()

	return m, nil
}

// initFused sets up GPU layer paths. MLA attention uses the dedicated device MLA cache (see initMLAGPU): K=kv_lora+rope, V=kv_lora does not fit the GQA KV-cache (n_kv_heads * head_dim)
//
// MoE (P1 deferred): full device-resident MoE layers need batched expert dispatch (router top-k, then per-expert FFN and accumulation on device) and n_batch>1 support.
// Until then MoE layers keep the per-token ExpertFFN path (n_batch=1, hidden state on host); only out_norm + lm_head may run on GPU (lm_head already runs on GPU through matmulInto here).
func (m *Model) initFused() {
	if m.dev == nil || m.ngl <= 0 {
		return
	}

	m.fused = gpuresid.New(m.weights, m.dev, device.RoPENeoX)
	m.moeGPU = make([]bool, m.cfg.NumLayers)
	for i, lt := range m.layerTensors {
		if !lt.moe {
			continue
		}

		m.moeGPU[i] = m.fused.MoESupported(lt.gateExps, lt.upExps, lt.downExps)
	}
}

// initMLAGPU allocates the device latent KV cache when the backend supports MLA.
// Any failure leaves mlaDev nil (CPU attention).
func (m *Model) initMLAGPU() {
	if m.dev == nil || m.ngl <= 0 {
		return
	}

	absorbed := false
	for _, lt := range m.layerTensors {
		if lt.absorbed {
			absorbed = true
			break
		}
	}

	if !absorbed {
		return
	}

	mb, ok := device.AsMLABackend(m.dev)
	if !ok {
		return
	}

	scale := m.cfg.AttnScale
	if scale == 0 {
		scale = float32(1 / math.Sqrt(float64(m.cfg.qkDim())))
	}

	p := device.MLAParams{
		NHeads: m.cfg.NumHeads,
		Nope:   m.cfg.QKNopeDim,
		KVLora: m.cfg.KVLoraRank,
		Rope:   m.cfg.RopeDim,
		VHead:  m.cfg.VHeadDim,
		Scale:  scale,
	}

	maxSeq := device.CapMaxSeq(m.cfg.ContextLength, m.gpuMaxSeq)
	if err := mb.MLACacheInit(m.ngl, maxSeq, p); err != nil {
		return
	}

	m.mlaDev = mb
	m.mlaParams = p
}

// mlaOnGPU reports whether this layer's MLA attention should be attempted on the device
func (m *Model) mlaOnGPU(layer int) bool {
	return m.mlaDev != nil && !m.mlaStale && m.layerOnGPU(layer)
}

// mlaWeightLoader dequantizes k_b / v_b once for the device upload and drops the host copy (Store.ReleaseRaw: no F32 duplicate stays in RAM on the hot path).
func (m *Model) mlaWeightLoader(name string) ([]float32, error) {
	f, err := m.weights.Floats(name)
	if err != nil {
		return nil, err
	}

	m.weights.ReleaseRaw(name)

	return f, nil
}

// mlaAttnGPU: RoPE q_pe on host, append latent K/V to the device cache, run mla_attn (absorb k_b, attention, v_b projection) into scratch.attn.
// Any error: caller falls back to CPU and sets mlaStale.
func (m *Model) mlaAttnGPU(layer, pos, devPos, seqLen int, lt layerTensors) error {
	cfg := m.cfg
	headK := cfg.QKNopeDim + cfg.RopeDim
	for h := 0; h < cfg.NumHeads; h++ {
		dst := m.scratch.qRope[h*headK : (h+1)*headK]
		copy(dst, m.scratch.qFull[h*headK:(h+1)*headK])
		compute.ApplyRoPEPartialScaled(dst[cfg.QKNopeDim:], pos, cfg.RopeFreqBase, cfg.RopeDim, cfg.ropeScale())
	}

	if err := m.mlaDev.MLACacheAppend(layer, devPos, m.scratch.kCache, m.scratch.vCache); err != nil {
		return err
	}

	return m.mlaDev.MLAAttention(layer, m.scratch.attn, m.scratch.qRope, seqLen, m.mlaParams, lt.kB, lt.vB, m.mlaWeightLoader)
}

// layerOnGPU reports whether the layer runs on GPU
func (m *Model) layerOnGPU(layer int) bool {
	return m.fused != nil && device.LayerOnGPU(layer, m.ngl, m.cfg.NumLayers)
}

func (m *Model) Config() Config {
	return m.cfg
}

func (m *Model) ResetCache() {
	m.cache.Reset()
	m.mlaStale = false
}

func (m *Model) Close() error {
	if m.dev == nil {
		return nil
	}

	err := m.dev.Close()
	m.dev = nil
	m.fused = nil
	m.mlaDev = nil

	return err
}

func (m *Model) EmbeddingDim() int {
	return m.cfg.EmbeddingDim
}

func (m *Model) Forward(tokenIDs []int, startPos int) ([]float32, error) {
	if len(tokenIDs) == 0 {
		return nil, fmt.Errorf("deepseek2: empty input")
	}

	for i, tok := range tokenIDs {
		if err := m.forwardToken(tok, startPos+i); err != nil {
			return nil, err
		}
	}

	if err := m.logitsFromHidden(m.scratch.x); err != nil {
		return nil, err
	}

	return m.scratch.logits, nil
}

func (m *Model) Embed(tokenIDs []int) ([]float32, error) {
	if len(tokenIDs) == 0 {
		return nil, fmt.Errorf("deepseek2: empty input")
	}

	m.ResetCache()
	defer m.ResetCache()
	for i, tok := range tokenIDs {
		if err := m.forwardToken(tok, i); err != nil {
			return nil, err
		}
	}

	if err := compute.RMSNormInto(m.scratch.h, m.scratch.x, m.outNorm, m.cfg.RMSNormEps); err != nil {
		return nil, err
	}

	out := make([]float32, m.cfg.EmbeddingDim)
	copy(out, m.scratch.h)

	return out, nil
}

func (m *Model) forwardToken(tokenID, pos int) error {
	if err := m.embedToken(tokenID); err != nil {
		return err
	}

	for layer := 0; layer < m.cfg.NumLayers; layer++ {
		if err := m.forwardBlock(layer, pos); err != nil {
			return err
		}
	}

	m.cache.Advance()

	return nil
}

func (m *Model) embedToken(tokenID int) error {
	raw, err := m.weights.Raw("token_embd.weight")
	if err != nil {
		return err
	}

	info, err := m.weights.Info("token_embd.weight")
	if err != nil {
		return err
	}

	switch info.Type {
	case dtype.Q8_0:
		return compute.EmbeddingQ8_0Into(m.scratch.x, raw, m.cfg.EmbeddingDim, tokenID)
	case dtype.Q4_0:
		return compute.EmbeddingQ4_0Into(m.scratch.x, raw, m.cfg.EmbeddingDim, tokenID)
	case dtype.Q4_1:
		return compute.EmbeddingQ4_1Into(m.scratch.x, raw, m.cfg.EmbeddingDim, tokenID)
	case dtype.Q5_0:
		return compute.EmbeddingQ5_0Into(m.scratch.x, raw, m.cfg.EmbeddingDim, tokenID)
	case dtype.Q5_1:
		return compute.EmbeddingQ5_1Into(m.scratch.x, raw, m.cfg.EmbeddingDim, tokenID)
	case dtype.Q4_K:
		return compute.EmbeddingQ4_KInto(m.scratch.x, raw, m.cfg.EmbeddingDim, tokenID)
	case dtype.Q5_K:
		return compute.EmbeddingQ5_KInto(m.scratch.x, raw, m.cfg.EmbeddingDim, tokenID)
	case dtype.Q6_K:
		return compute.EmbeddingQ6_KInto(m.scratch.x, raw, m.cfg.EmbeddingDim, tokenID)
	case dtype.Q2_K:
		return compute.EmbeddingQ2_KInto(m.scratch.x, raw, m.cfg.EmbeddingDim, tokenID)
	case dtype.Q3_K:
		return compute.EmbeddingQ3_KInto(m.scratch.x, raw, m.cfg.EmbeddingDim, tokenID)
	case dtype.Q8_K:
		return compute.EmbeddingQ8_KInto(m.scratch.x, raw, m.cfg.EmbeddingDim, tokenID)
	default:
		f32, err := m.weights.Floats("token_embd.weight")
		if err != nil {
			return err
		}

		off := tokenID * m.cfg.EmbeddingDim
		copy(m.scratch.x, f32[off:off+m.cfg.EmbeddingDim])
		return nil
	}
}

func (m *Model) forwardBlock(layer, pos int) error {
	ln := m.layerNorms[layer]
	lt := m.layerTensors[layer]

	if err := compute.RMSNormInto(m.scratch.h, m.scratch.x, ln.attnNorm, m.cfg.RMSNormEps); err != nil {
		return err
	}

	if err := m.mlaAttn(layer, pos, lt, ln); err != nil {
		return err
	}

	compute.AddInPlace(m.scratch.x, m.scratch.tmp) // attn output is in tmp after wo

	if err := compute.RMSNormInto(m.scratch.h, m.scratch.x, ln.ffnNorm, m.cfg.RMSNormEps); err != nil {
		return err
	}

	if lt.moe {
		if err := m.ffnMoE(lt, layer); err != nil {
			return err
		}

		compute.AddInPlace(m.scratch.x, m.scratch.moeAcc)
		return nil
	}

	if err := m.ffnDense(lt, layer); err != nil {
		return err
	}

	compute.AddInPlace(m.scratch.x, m.scratch.tmp)
	return nil
}

// ffnDense - dense layer FFN; on GPU one fused SwiGLU call
func (m *Model) ffnDense(lt layerTensors, layer int) error {
	if m.layerOnGPU(layer) {
		if err := m.fused.FFNNamed(lt.ffnGate, lt.ffnUp, lt.ffnDown, m.scratch.h, m.scratch.tmp, m.cfg.EmbeddingDim, m.cfg.FFNHidden); err == nil {
			return nil
		}
	}

	if err := m.matmulInto(lt.ffnGate, m.cfg.FFNHidden, m.cfg.EmbeddingDim, m.scratch.h, m.scratch.gate, layer); err != nil {
		return err
	}

	if err := m.matmulInto(lt.ffnUp, m.cfg.FFNHidden, m.cfg.EmbeddingDim, m.scratch.h, m.scratch.up, layer); err != nil {
		return err
	}

	compute.SwiGLUInPlace(m.scratch.gate[:m.cfg.FFNHidden], m.scratch.up[:m.cfg.FFNHidden])

	return m.matmulInto(lt.ffnDown, m.cfg.EmbeddingDim, m.cfg.FFNHidden, m.scratch.gate[:m.cfg.FFNHidden], m.scratch.tmp, layer)
}

func (m *Model) mlaAttn(layer, pos int, lt layerTensors, ln layerNorms) error {
	cfg := m.cfg
	headK := cfg.QKNopeDim + cfg.RopeDim

	// Q projection
	if lt.liteQ {
		if err := m.matmulInto(lt.q, cfg.NumHeads*headK, cfg.EmbeddingDim, m.scratch.h, m.scratch.qFull, layer); err != nil {
			return err
		}
	} else {
		if err := m.matmulInto(lt.qA, cfg.QLoraRank, cfg.EmbeddingDim, m.scratch.h, m.scratch.qLora[:cfg.QLoraRank], layer); err != nil {
			return err
		}

		// qLora -> tmp[:qLora] -> qLora (not in-place)
		tmpQ := m.scratch.tmp[:cfg.QLoraRank]
		if err := compute.RMSNormInto(tmpQ, m.scratch.qLora[:cfg.QLoraRank], ln.qANorm, cfg.RMSNormEps); err != nil {
			return err
		}

		copy(m.scratch.qLora[:cfg.QLoraRank], tmpQ)
		if err := m.matmulInto(lt.qB, cfg.NumHeads*headK, cfg.QLoraRank, m.scratch.qLora[:cfg.QLoraRank], m.scratch.qFull, layer); err != nil {
			return err
		}
	}

	// KV compression + k_pe
	if err := m.matmulInto(lt.kvAMQA, cfg.qkDim(), cfg.EmbeddingDim, m.scratch.h, m.scratch.kvPe, layer); err != nil {
		return err
	}

	copy(m.scratch.vCache, m.scratch.kvPe[:cfg.KVLoraRank])

	kPe := m.scratch.kvPe[cfg.KVLoraRank : cfg.KVLoraRank+cfg.RopeDim]
	compute.ApplyRoPEPartialScaled(kPe, pos, cfg.RopeFreqBase, cfg.RopeDim, cfg.ropeScale())
	if err := compute.RMSNormInto(m.scratch.vCache, m.scratch.vCache, ln.kvANorm, cfg.RMSNormEps); err != nil {
		return err
	}

	if !lt.absorbed {
		return m.mlaAttnLegacy(layer, pos, lt)
	}

	// to cache: K = concat(kv_cmpr, k_pe), V = kv_cmpr
	copy(m.scratch.kCache[:cfg.KVLoraRank], m.scratch.vCache)
	copy(m.scratch.kCache[cfg.KVLoraRank:], kPe)
	devPos := m.cache.Len()
	if err := m.cache.Append(layer, m.scratch.kCache, m.scratch.vCache); err != nil {
		return err
	}
	seqLen := devPos + 1

	// GPU: absorb + attention + v_b projection in one launch; CPU path below is the fallback
	if m.mlaOnGPU(layer) {
		if err := m.mlaAttnGPU(layer, pos, devPos, seqLen, lt); err == nil {
			return m.matmulInto(lt.attnOut, cfg.EmbeddingDim, cfg.NumHeads*cfg.VHeadDim, m.scratch.attn, m.scratch.tmp, layer)
		}

		// device latent cache may now miss this token: keep CPU until ResetCache
		m.mlaStale = true
	}

	// absorb q_nope via wk_b; RoPE on q_pe; Q = concat(q_absorbed, q_pe)
	kBW, err := m.weights.Floats(lt.kB)
	if err != nil {
		return err
	}

	vBW, err := m.weights.Floats(lt.vB)
	if err != nil {
		return err
	}

	qk := cfg.qkDim()
	for h := 0; h < cfg.NumHeads; h++ {
		src := m.scratch.qFull[h*headK : (h+1)*headK]
		qNope := src[:cfg.QKNopeDim]
		qPe := m.scratch.qPe[:cfg.RopeDim]
		copy(qPe, src[cfg.QKNopeDim:])
		compute.ApplyRoPEPartialScaled(qPe, pos, cfg.RopeFreqBase, cfg.RopeDim, cfg.ropeScale())

		// wk_b head: {qk_nope, kv_lora}
		wOff := h * cfg.QKNopeDim * cfg.KVLoraRank
		dstAbs := m.scratch.qAbs[h*qk : h*qk+cfg.KVLoraRank]
		if err := compute.MatMulColMajorInto(kBW[wOff:wOff+cfg.QKNopeDim*cfg.KVLoraRank], cfg.QKNopeDim, cfg.KVLoraRank, qNope, dstAbs); err != nil {
			return err
		}

		copy(m.scratch.qAbs[h*qk+cfg.KVLoraRank:h*qk+qk], qPe)
	}

	if err := compute.AttentionMLAAbsorbedInto(
		m.scratch.attn,
		m.scratch.qAbs,
		m.cache.KLayer(layer),
		m.cache.VLayer(layer),
		m.scratch.scores,
		vBW,
		seqLen,
		cfg.NumHeads,
		cfg.KVLoraRank,
		cfg.RopeDim,
		cfg.VHeadDim,
		cfg.AttnScale,
	); err != nil {
		return err
	}

	return m.matmulInto(lt.attnOut, cfg.EmbeddingDim, cfg.NumHeads*cfg.VHeadDim, m.scratch.attn, m.scratch.tmp, layer)
}

func (m *Model) mlaAttnLegacy(layer, pos int, lt layerTensors) error {
	cfg := m.cfg
	headK := cfg.QKNopeDim + cfg.RopeDim
	kvBW, err := m.weights.Floats(lt.kvB)
	if err != nil {
		return err
	}

	// kv_b: {kv_lora, n_head*(qk_nope+v_head)} - expand each head
	perHead := cfg.QKNopeDim + cfg.VHeadDim
	kFull := make([]float32, cfg.NumHeads*headK)
	vFull := make([]float32, cfg.NumHeads*cfg.VHeadDim)

	for h := 0; h < cfg.NumHeads; h++ {
		wOff := h * cfg.KVLoraRank * perHead
		// ggml mul_mat: out[perHead] from W[kv_lora, perHead]
		out := make([]float32, perHead)
		if err := compute.MatMulColMajorInto(kvBW[wOff:wOff+cfg.KVLoraRank*perHead], cfg.KVLoraRank, perHead, m.scratch.vCache, out); err != nil {
			return err
		}

		copy(kFull[h*headK:h*headK+cfg.QKNopeDim], out[:cfg.QKNopeDim])
		// broadcast k_pe across heads
		copy(kFull[h*headK+cfg.QKNopeDim:h*headK+headK], m.scratch.kvPe[cfg.KVLoraRank:])
		copy(vFull[h*cfg.VHeadDim:(h+1)*cfg.VHeadDim], out[cfg.QKNopeDim:])
	}

	for h := 0; h < cfg.NumHeads; h++ {
		compute.ApplyRoPEPartialScaled(m.scratch.qFull[h*headK+cfg.QKNopeDim:h*headK+headK], pos, cfg.RopeFreqBase, cfg.RopeDim, cfg.ropeScale())
	}

	// Legacy path without absorbed MLA is not implemented (needs separate MHA cache).
	_ = kFull
	_ = vFull
	_ = layer
	return fmt.Errorf("deepseek2: legacy attn_kv_b (non-MLA GGUF) not supported yet; need attn_k_b/attn_v_b")
}

func (m *Model) ffnMoE(lt layerTensors, layer int) error {
	embd := m.cfg.EmbeddingDim
	nExp := m.cfg.ExpertCount
	ffn := m.cfg.ExpertFFN

	if err := m.matmulInto(lt.gateInp, nExp, embd, m.scratch.h, m.scratch.router[:nExp], layer); err != nil {
		return err
	}

	if lt.expProbsB != "" {
		bias, err := m.weights.Floats(lt.expProbsB)
		if err != nil {
			return err
		}
		compute.AddBiasInPlace(m.scratch.router[:nExp], bias)
	}

	idxs, weights := moe.TopKSoftmax(m.scratch.router[:nExp], m.cfg.ExpertUsedCount, m.cfg.ExpertWeightsNorm, m.cfg.ExpertWeightScale)
	clear(m.scratch.moeAcc)

	// selected experts run on GPU via tensor slice, without dequantizing full expert matrix on host
	if m.moeExpertsGPU(lt, layer, idxs, weights) {
		return m.sharedExpert(lt, layer)
	}

	gateBuf := m.scratch.gate[:ffn]
	upBuf := m.scratch.up[:ffn]
	for i, ei := range idxs {
		if err := dispatch.MatMulExpertInto(m.weights, lt.gateExps, ei, ffn, embd, m.scratch.h, gateBuf); err != nil {
			return err
		}
		if err := dispatch.MatMulExpertInto(m.weights, lt.upExps, ei, ffn, embd, m.scratch.h, upBuf); err != nil {
			return err
		}
		compute.SwiGLUInPlace(gateBuf, upBuf)
		if err := dispatch.MatMulExpertInto(m.weights, lt.downExps, ei, embd, ffn, gateBuf, m.scratch.tmp); err != nil {
			return err
		}

		w := weights[i]
		for j := range embd {
			m.scratch.moeAcc[j] += m.scratch.tmp[j] * w
		}
	}

	return m.sharedExpert(lt, layer)
}

// moeExpertsGPU runs selected experts on GPU and accumulates them in moeAcc.
// false - layer does not run on GPU (no offload or weight type outside fused paths)
func (m *Model) moeExpertsGPU(lt layerTensors, layer int, idxs []int, weights []float32) bool {
	if !m.layerOnGPU(layer) || layer >= len(m.moeGPU) || !m.moeGPU[layer] {
		return false
	}

	embd := m.cfg.EmbeddingDim
	ffn := m.cfg.ExpertFFN
	for i, ei := range idxs {
		if err := m.fused.ExpertFFN(lt.gateExps, lt.upExps, lt.downExps, ei, m.scratch.h, m.scratch.tmp, embd, ffn); err != nil {
			// partially computed moeAcc is invalid: reset and fall back to CPU
			clear(m.scratch.moeAcc)
			return false
		}

		w := weights[i]
		for j := range embd {
			m.scratch.moeAcc[j] += m.scratch.tmp[j] * w
		}
	}

	return true
}

// sharedExpert adds the layer shared expert to moeAcc (if present)
func (m *Model) sharedExpert(lt layerTensors, layer int) error {
	if lt.gateShexp == "" {
		return nil
	}

	embd := m.cfg.EmbeddingDim
	shared := m.cfg.sharedFFN()
	if m.layerOnGPU(layer) {
		if err := m.fused.FFNNamed(lt.gateShexp, lt.upShexp, lt.downShexp, m.scratch.h, m.scratch.tmp, embd, shared); err == nil {
			compute.AddInPlace(m.scratch.moeAcc, m.scratch.tmp)
			return nil
		}
	}

	sg := m.scratch.gate[:shared]
	su := m.scratch.up[:shared]
	if err := m.matmulInto(lt.gateShexp, shared, embd, m.scratch.h, sg, layer); err != nil {
		return err
	}

	if err := m.matmulInto(lt.upShexp, shared, embd, m.scratch.h, su, layer); err != nil {
		return err
	}

	compute.SwiGLUInPlace(sg, su)
	if err := m.matmulInto(lt.downShexp, embd, shared, sg, m.scratch.tmp, layer); err != nil {
		return err
	}

	compute.AddInPlace(m.scratch.moeAcc, m.scratch.tmp)

	return nil
}

func (m *Model) matmulInto(name string, rows, cols int, vec, out []float32, layer int) error {
	if m.layerOnGPU(layer) {
		if err := m.fused.MatMulInto(name, rows, cols, vec, out); err == nil {
			return nil
		}
	}

	return dispatch.MatMulRawInto(m.weights, name, rows, cols, vec, out)
}

func (m *Model) logitsFromHidden(x []float32) error {
	if err := compute.RMSNormInto(m.scratch.h, x, m.outNorm, m.cfg.RMSNormEps); err != nil {
		return err
	}

	return m.matmulInto(m.lmHeadName, m.cfg.VocabSize, m.cfg.EmbeddingDim, m.scratch.h, m.scratch.logits, m.cfg.NumLayers-1)
}
