package deepseek2

import (
	"fmt"
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
	layerQuant   []bool
	residency    bool
	decode       gpuresid.DecodeFlags
	cache        *MLACache
	mhaCache     *MHACache // legacy attn_kv_b full MHA K/V; nil when all layers absorbed
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

	for _, lt := range layerTensors {
		if !lt.absorbed {
			m.mhaCache = NewMHACache(cfg)
			break
		}
	}

	m.initFused()
	m.initMLAGPU()

	return m, nil
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

// mlaAttnGPU: YaRN/partial RoPE of q_pe on device (rope_heads over packed pe), append latent K/V, run mla_attn into scratch.attn.
// k_pe must already be RoPE'd in kCache. Any error: caller falls back to CPU and sets mlaStale.
func (m *Model) mlaAttnGPU(layer, pos, devPos, seqLen int, lt layerTensors) error {
	cfg := m.cfg
	headK := cfg.QKNopeDim + cfg.RopeDim
	for h := 0; h < cfg.NumHeads; h++ {
		copy(m.scratch.qRope[h*headK:(h+1)*headK], m.scratch.qFull[h*headK:(h+1)*headK])
	}

	if err := m.ropeQPe(m.scratch.qRope, pos); err != nil {
		return err
	}

	if err := m.mlaDev.MLACacheAppend(layer, devPos, m.scratch.kCache, m.scratch.vCache); err != nil {
		return err
	}

	return m.mlaDev.MLAAttention(layer, m.scratch.attn, m.scratch.qRope, seqLen, m.mlaParams, lt.kB, lt.vB, m.mlaWeightLoader)
}

// applyRoPEPE applies YaRN NeoX RoPE to a contiguous pe buffer [nHeads*RopeDim] via device rope_heads when available
func (m *Model) applyRoPEPE(pe []float32, nHeads, pos int) error {
	cfg := m.cfg
	nRot := cfg.RopeDim
	if nHeads <= 0 || nRot <= 0 || len(pe) < nHeads*nRot {
		return fmt.Errorf("deepseek2: RoPE pe buffer too short")
	}

	scale := cfg.ropeScale()
	if m.dev != nil {
		if err := m.dev.ApplyRoPEHeadsScaled(pe, nHeads, nRot, pos, cfg.RopeFreqBase, scale); err == nil {
			return nil
		}
	}

	compute.ApplyRoPEHeadsPartialScaled(pe, nHeads, nRot, nRot, pos, cfg.RopeFreqBase, scale)
	return nil
}

// ropeQPe RoPEs pe dims in-place for heads laid out as [nope|pe] * nHeads (packs pe, device RoPE, unpacks)
func (m *Model) ropeQPe(heads []float32, pos int) error {
	cfg := m.cfg
	headK := cfg.QKNopeDim + cfg.RopeDim
	nRot := cfg.RopeDim
	pack := m.scratch.pePack
	for h := 0; h < cfg.NumHeads; h++ {
		copy(pack[h*nRot:(h+1)*nRot], heads[h*headK+cfg.QKNopeDim:h*headK+headK])
	}

	if err := m.applyRoPEPE(pack, cfg.NumHeads, pos); err != nil {
		return err
	}

	for h := 0; h < cfg.NumHeads; h++ {
		copy(heads[h*headK+cfg.QKNopeDim:h*headK+headK], pack[h*nRot:(h+1)*nRot])
	}

	return nil
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
	if m.mhaCache != nil {
		m.mhaCache.Reset()
	}
	m.decode.Clear()
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

	if err := m.logitsFinish(); err != nil {
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

	if err := m.syncHiddenFromDevice(); err != nil {
		return nil, err
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
	return dispatch.EmbedTokenInto(m.weights, m.scratch.x, m.cfg.EmbeddingDim, tokenID)
}

func (m *Model) forwardBlock(layer, pos int) error {
	ln := m.layerNorms[layer]
	lt := m.layerTensors[layer]

	// MLA projections read host hidden; pull resid back before attention
	if err := m.syncHiddenFromDevice(); err != nil {
		return err
	}

	if err := compute.RMSNormInto(m.scratch.h, m.scratch.x, ln.attnNorm, m.cfg.RMSNormEps); err != nil {
		return err
	}

	if err := m.mlaAttn(layer, pos, lt, ln); err != nil {
		return err
	}

	compute.AddInPlace(m.scratch.x, m.scratch.tmp) // attn output is in tmp after wo

	// Device-resident MoE/dense FFN after MLA (n_batch=1)
	if m.residency && m.layerOnGPU(layer) {
		var err error
		if lt.moe {
			err = m.forwardBlockDeviceMoE(layer, ln, lt)
		} else {
			err = m.forwardBlockDevice(layer, ln, lt)
		}
		if err == nil {
			return nil
		}

		if rerr := m.recoverHiddenFromDevice(layer, err); rerr != nil {
			return rerr
		}
	}

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

func (m *Model) logitsFinish() error {
	if m.decode.ResidDevice && m.decode.LogitsOnGPU && m.fused != nil {
		if err := m.logitsDevice(); err == nil {
			return nil
		}
		m.decode.LogitsOnGPU = false
	}

	if err := m.syncHiddenFromDevice(); err != nil {
		return err
	}

	return m.logitsFromHidden(m.scratch.x)
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
	if err := m.applyRoPEPE(kPe, 1, pos); err != nil {
		return err
	}

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
	if err := m.ropeQPe(m.scratch.qFull, pos); err != nil {
		return err
	}

	for h := 0; h < cfg.NumHeads; h++ {
		src := m.scratch.qFull[h*headK : (h+1)*headK]
		qNope := src[:cfg.QKNopeDim]
		qPe := src[cfg.QKNopeDim:]

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
	if m.mhaCache == nil {
		return fmt.Errorf("deepseek2: legacy attn_kv_b cache not initialized")
	}

	kvBW, err := m.weights.Floats(lt.kvB)
	if err != nil {
		return err
	}

	// kv_b: {kv_lora, n_head*(qk_nope+v_head)} - expand each head
	perHead := cfg.QKNopeDim + cfg.VHeadDim
	kFull := m.scratch.kMHA
	vFull := m.scratch.vMHA
	out := m.scratch.kvExp[:perHead]
	kPe := m.scratch.kvPe[cfg.KVLoraRank : cfg.KVLoraRank+cfg.RopeDim]

	for h := 0; h < cfg.NumHeads; h++ {
		wOff := h * cfg.KVLoraRank * perHead
		// ggml mul_mat: out[perHead] from W[kv_lora, perHead]
		if err := compute.MatMulColMajorInto(kvBW[wOff:wOff+cfg.KVLoraRank*perHead], cfg.KVLoraRank, perHead, m.scratch.vCache, out); err != nil {
			return err
		}

		copy(kFull[h*headK:h*headK+cfg.QKNopeDim], out[:cfg.QKNopeDim])
		// broadcast already-RoPE'd k_pe across heads
		copy(kFull[h*headK+cfg.QKNopeDim:h*headK+headK], kPe)
		copy(vFull[h*cfg.VHeadDim:(h+1)*cfg.VHeadDim], out[cfg.QKNopeDim:])
	}

	if err := m.ropeQPe(m.scratch.qFull, pos); err != nil {
		return err
	}

	kvPos := m.cache.Len()
	if err := m.mhaCache.Append(layer, kFull, vFull); err != nil {
		return err
	}
	seqLen := kvPos + 1

	scale := cfg.AttnScale
	if scale == 0 {
		scale = float32(1 / math.Sqrt(float64(headK)))
	}

	if err := compute.AttentionScoresHVInto(
		m.scratch.attn,
		m.scratch.qFull,
		m.mhaCache.KLayer(layer),
		m.mhaCache.VLayer(layer),
		m.scratch.scores,
		seqLen,
		cfg.NumHeads,
		cfg.NumHeads,
		headK,
		cfg.VHeadDim,
		scale,
	); err != nil {
		return err
	}

	return m.matmulInto(lt.attnOut, cfg.EmbeddingDim, cfg.NumHeads*cfg.VHeadDim, m.scratch.attn, m.scratch.tmp, layer)
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

	if err := moe.ExpertsSwiGLUInto(
		m.weights, lt.gateExps, lt.upExps, lt.downExps,
		idxs, weights, m.scratch.h,
		m.scratch.gate[:ffn], m.scratch.up[:ffn], m.scratch.tmp, m.scratch.moeAcc,
		embd, ffn,
	); err != nil {
		return err
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
