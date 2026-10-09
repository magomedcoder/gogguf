package deepseek2

import (
	"fmt"

	"github.com/magomedcoder/gogguf/pkg/format"
	"github.com/magomedcoder/gogguf/pkg/gpu"
	"github.com/magomedcoder/gogguf/pkg/model/gpuresid"
	"github.com/magomedcoder/gogguf/pkg/model/moe"
	"github.com/magomedcoder/gogguf/pkg/ops"
	"github.com/magomedcoder/gogguf/pkg/weights"
)

// Model - DeepSeek-V2/V3: absorbed MLA + mixture of experts
type Model struct {
	cfg          Config
	weights      *weights.Store
	gpu          gpu.Backend
	ngl          int
	fused        *gpuresid.Runner
	moeGPU       []bool
	cache        *MLACache
	scratch      scratch
	layerNorms   []layerNorms
	layerTensors []layerTensors
	outNorm      []float32
	lmHeadName   string
}

// Load creates DeepSeek2 from weights. GPU runs MLA projections, FFN, and experts (§7),
// MLA attention itself stays on CPU: its cache layout differs from GPU KV-cache
func Load(w *weights.Store, g gpu.Backend, ngl, _ int) (*Model, error) {
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
		gpu:          g,
		ngl:          ngl,
		cache:        NewMLACache(cfg),
		scratch:      newScratch(cfg),
		layerNorms:   layerNorms,
		layerTensors: layerTensors,
		outNorm:      outNorm,
		lmHeadName:   lmHeadName,
	}

	m.initFused()

	return m, nil
}

// initFused sets up GPU layer paths (§7). MLA attention stays on CPU: absorbed wk_b/wv_b are column-major, MLA cache is K=kv_lora+rope, V=kv_lora, which GPU KV-cache (n_kv_heads * head_dim) does not represent
func (m *Model) initFused() {
	if m.gpu == nil || m.ngl <= 0 {
		return
	}

	m.fused = gpuresid.New(m.weights, m.gpu, gpu.RoPENeoX)
	m.moeGPU = make([]bool, m.cfg.NumLayers)
	for i, lt := range m.layerTensors {
		if !lt.moe {
			continue
		}

		m.moeGPU[i] = m.fused.MoESupported(lt.gateExps, lt.upExps, lt.downExps)
	}
}

// layerOnGPU reports whether the layer runs on GPU
func (m *Model) layerOnGPU(layer int) bool {
	return m.fused != nil && gpu.LayerOnGPU(layer, m.ngl, m.cfg.NumLayers)
}

func (m *Model) Config() Config {
	return m.cfg
}

func (m *Model) ResetCache() {
	m.cache.Reset()
}

func (m *Model) Close() error {
	if m.gpu == nil {
		return nil
	}

	err := m.gpu.Close()
	m.gpu = nil
	m.fused = nil

	return err
}

func (m *Model) EmbeddingDim() int {
	return m.cfg.EmbeddingDim
}

func (m *Model) Forward(tokenIDs []int, startPos int) ([]float32, error) {
	if len(tokenIDs) == 0 {
		return nil, fmt.Errorf("deepseek2: пустой ввод")
	}

	for i, tok := range tokenIDs {
		if err := m.forwardToken(tok, startPos+i); err != nil {
			return nil, err
		}
	}

	if err := m.logitsFromHidden(m.scratch.x); err != nil {
		return nil, err
	}

	copy(m.scratch.out, m.scratch.logits)
	return m.scratch.out, nil
}

func (m *Model) Embed(tokenIDs []int) ([]float32, error) {
	if len(tokenIDs) == 0 {
		return nil, fmt.Errorf("deepseek2: пустой ввод")
	}

	m.ResetCache()
	defer m.ResetCache()
	for i, tok := range tokenIDs {
		if err := m.forwardToken(tok, i); err != nil {
			return nil, err
		}
	}

	if err := ops.RMSNormInto(m.scratch.h, m.scratch.x, m.outNorm, m.cfg.RMSNormEps); err != nil {
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
	case format.GgmlQ8_0:
		return ops.EmbeddingQ8_0Into(m.scratch.x, raw, m.cfg.EmbeddingDim, tokenID)
	case format.GgmlQ4_0:
		return ops.EmbeddingQ4_0Into(m.scratch.x, raw, m.cfg.EmbeddingDim, tokenID)
	case format.GgmlQ4_1:
		return ops.EmbeddingQ4_1Into(m.scratch.x, raw, m.cfg.EmbeddingDim, tokenID)
	case format.GgmlQ5_0:
		return ops.EmbeddingQ5_0Into(m.scratch.x, raw, m.cfg.EmbeddingDim, tokenID)
	case format.GgmlQ5_1:
		return ops.EmbeddingQ5_1Into(m.scratch.x, raw, m.cfg.EmbeddingDim, tokenID)
	case format.GgmlQ4_K:
		return ops.EmbeddingQ4_KInto(m.scratch.x, raw, m.cfg.EmbeddingDim, tokenID)
	case format.GgmlQ5_K:
		return ops.EmbeddingQ5_KInto(m.scratch.x, raw, m.cfg.EmbeddingDim, tokenID)
	case format.GgmlQ6_K:
		return ops.EmbeddingQ6_KInto(m.scratch.x, raw, m.cfg.EmbeddingDim, tokenID)
	case format.GgmlQ2_K:
		return ops.EmbeddingQ2_KInto(m.scratch.x, raw, m.cfg.EmbeddingDim, tokenID)
	case format.GgmlQ3_K:
		return ops.EmbeddingQ3_KInto(m.scratch.x, raw, m.cfg.EmbeddingDim, tokenID)
	case format.GgmlQ8_K:
		return ops.EmbeddingQ8_KInto(m.scratch.x, raw, m.cfg.EmbeddingDim, tokenID)
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

	if err := ops.RMSNormInto(m.scratch.h, m.scratch.x, ln.attnNorm, m.cfg.RMSNormEps); err != nil {
		return err
	}

	if err := m.mlaAttn(layer, pos, lt, ln); err != nil {
		return err
	}

	ops.AddInPlace(m.scratch.x, m.scratch.tmp) // attn output is in tmp after wo

	if err := ops.RMSNormInto(m.scratch.h, m.scratch.x, ln.ffnNorm, m.cfg.RMSNormEps); err != nil {
		return err
	}

	if lt.moe {
		if err := m.ffnMoE(lt, layer); err != nil {
			return err
		}

		ops.AddInPlace(m.scratch.x, m.scratch.moeAcc)
		return nil
	}

	if err := m.ffnDense(lt, layer); err != nil {
		return err
	}

	ops.AddInPlace(m.scratch.x, m.scratch.tmp)
	return nil
}

// ffnDense - dense layer FFN; on GPU one fused SwiGLU call (§7)
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

	ops.SwiGLUInPlace(m.scratch.gate[:m.cfg.FFNHidden], m.scratch.up[:m.cfg.FFNHidden])

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
		if err := ops.RMSNormInto(tmpQ, m.scratch.qLora[:cfg.QLoraRank], ln.qANorm, cfg.RMSNormEps); err != nil {
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
	ops.ApplyRoPEPartialScaled(kPe, pos, cfg.RopeFreqBase, cfg.RopeDim, cfg.ropeScale())
	if err := ops.RMSNormInto(m.scratch.vCache, m.scratch.vCache, ln.kvANorm, cfg.RMSNormEps); err != nil {
		return err
	}

	if !lt.absorbed {
		return m.mlaAttnLegacy(layer, pos, lt)
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
		qPe := append([]float32(nil), src[cfg.QKNopeDim:]...)
		ops.ApplyRoPEPartialScaled(qPe, pos, cfg.RopeFreqBase, cfg.RopeDim, cfg.ropeScale())

		// wk_b head: {qk_nope, kv_lora}
		wOff := h * cfg.QKNopeDim * cfg.KVLoraRank
		dstAbs := m.scratch.qAbs[h*qk : h*qk+cfg.KVLoraRank]
		if err := ops.MatMulColMajorInto(kBW[wOff:wOff+cfg.QKNopeDim*cfg.KVLoraRank], cfg.QKNopeDim, cfg.KVLoraRank, qNope, dstAbs); err != nil {
			return err
		}

		copy(m.scratch.qAbs[h*qk+cfg.KVLoraRank:h*qk+qk], qPe)
	}

	// to cache: K = concat(kv_cmpr, k_pe), V = kv_cmpr
	copy(m.scratch.kCache[:cfg.KVLoraRank], m.scratch.vCache)
	copy(m.scratch.kCache[cfg.KVLoraRank:], kPe)
	m.cache.Append(layer, m.scratch.kCache, m.scratch.vCache)

	seqLen := m.cache.Len() + 1
	if err := ops.AttentionMLAAbsorbedInto(
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
		if err := ops.MatMulColMajorInto(kvBW[wOff:wOff+cfg.KVLoraRank*perHead], cfg.KVLoraRank, perHead, m.scratch.vCache, out); err != nil {
			return err
		}

		copy(kFull[h*headK:h*headK+cfg.QKNopeDim], out[:cfg.QKNopeDim])
		// broadcast k_pe across heads
		copy(kFull[h*headK+cfg.QKNopeDim:h*headK+headK], m.scratch.kvPe[cfg.KVLoraRank:])
		copy(vFull[h*cfg.VHeadDim:(h+1)*cfg.VHeadDim], out[cfg.QKNopeDim:])
	}

	// RoPE on the Q tail
	q := append([]float32(nil), m.scratch.qFull...)
	for h := 0; h < cfg.NumHeads; h++ {
		ops.ApplyRoPEPartialScaled(q[h*headK+cfg.QKNopeDim:h*headK+headK], pos, cfg.RopeFreqBase, cfg.RopeDim, cfg.ropeScale())
	}

	// Legacy path without absorbed MLA is not implemented (needs separate MHA cache).
	_ = q
	_ = kFull
	_ = vFull
	_ = layer
	return fmt.Errorf("deepseek2: устаревший attn_kv_b (non-MLA GGUF) пока не поддерживается; нужен attn_k_b/attn_v_b")
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
		ops.AddBiasInPlace(m.scratch.router[:nExp], bias)
	}

	idxs, weights := moe.TopKSoftmax(m.scratch.router[:nExp], m.cfg.ExpertUsedCount, m.cfg.ExpertWeightsNorm, m.cfg.ExpertWeightScale)
	clear(m.scratch.moeAcc)

	// §6: selected experts run on GPU via tensor slice, without dequantizing full expert matrix on host
	if m.moeExpertsGPU(lt, layer, idxs, weights) {
		return m.sharedExpert(lt, layer)
	}

	gateW, err := m.weights.Floats(lt.gateExps)
	if err != nil {
		return err
	}

	upW, err := m.weights.Floats(lt.upExps)
	if err != nil {
		return err
	}

	downW, err := m.weights.Floats(lt.downExps)
	if err != nil {
		return err
	}

	expertElems := ffn * embd
	gateBuf := m.scratch.gate[:ffn]
	upBuf := m.scratch.up[:ffn]
	for i, ei := range idxs {
		gOff := ei * expertElems
		if err := ops.MatMulVecInto(gateW[gOff:gOff+expertElems], ffn, embd, m.scratch.h, gateBuf); err != nil {
			return err
		}

		if err := ops.MatMulVecInto(upW[gOff:gOff+expertElems], ffn, embd, m.scratch.h, upBuf); err != nil {
			return err
		}

		ops.SwiGLUInPlace(gateBuf, upBuf)
		if err := ops.MatMulVecInto(downW[gOff:gOff+expertElems], embd, ffn, gateBuf, m.scratch.tmp); err != nil {
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
			ops.AddInPlace(m.scratch.moeAcc, m.scratch.tmp)
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

	ops.SwiGLUInPlace(sg, su)
	if err := m.matmulInto(lt.downShexp, embd, shared, sg, m.scratch.tmp, layer); err != nil {
		return err
	}

	ops.AddInPlace(m.scratch.moeAcc, m.scratch.tmp)

	return nil
}

func (m *Model) matmulInto(name string, rows, cols int, vec, out []float32, layer int) error {
	if m.layerOnGPU(layer) {
		if err := m.fused.MatMulInto(name, rows, cols, vec, out); err == nil {
			return nil
		}
	}

	raw, err := m.weights.Raw(name)
	if err != nil {
		return err
	}

	info, err := m.weights.Info(name)
	if err != nil {
		return err
	}

	switch info.Type {
	case format.GgmlQ8_0:
		return ops.MatMulVecQ8_0Into(raw, rows, cols, vec, out)
	case format.GgmlQ4_0:
		return ops.MatMulVecQ4_0Into(raw, rows, cols, vec, out)
	case format.GgmlQ4_1:
		return ops.MatMulVecQ4_1Into(raw, rows, cols, vec, out)
	case format.GgmlQ5_0:
		return ops.MatMulVecQ5_0Into(raw, rows, cols, vec, out)
	case format.GgmlQ5_1:
		return ops.MatMulVecQ5_1Into(raw, rows, cols, vec, out)
	case format.GgmlQ4_K:
		return ops.MatMulVecQ4_KInto(raw, rows, cols, vec, out)
	case format.GgmlQ5_K:
		return ops.MatMulVecQ5_KInto(raw, rows, cols, vec, out)
	case format.GgmlQ6_K:
		return ops.MatMulVecQ6_KInto(raw, rows, cols, vec, out)
	case format.GgmlQ2_K:
		return ops.MatMulVecQ2_KInto(raw, rows, cols, vec, out)
	case format.GgmlQ3_K:
		return ops.MatMulVecQ3_KInto(raw, rows, cols, vec, out)
	case format.GgmlQ8_K:
		return ops.MatMulVecQ8_KInto(raw, rows, cols, vec, out)
	default:
		f32, err := m.weights.Floats(name)
		if err != nil {
			return err
		}

		return ops.MatMulVecInto(f32, rows, cols, vec, out)
	}
}

func (m *Model) logitsFromHidden(x []float32) error {
	if err := ops.RMSNormInto(m.scratch.h, x, m.outNorm, m.cfg.RMSNormEps); err != nil {
		return err
	}

	return m.matmulInto(m.lmHeadName, m.cfg.VocabSize, m.cfg.EmbeddingDim, m.scratch.h, m.scratch.logits, m.cfg.NumLayers-1)
}
