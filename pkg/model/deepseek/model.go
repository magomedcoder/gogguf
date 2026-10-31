package deepseek

import (
	"fmt"
	"github.com/magomedcoder/gogguf/pkg/dtype"

	"github.com/magomedcoder/gogguf/pkg/compute"
	"github.com/magomedcoder/gogguf/pkg/device"
	"github.com/magomedcoder/gogguf/pkg/exec/dispatch"
	"github.com/magomedcoder/gogguf/pkg/exec/gpuresid"
	"github.com/magomedcoder/gogguf/pkg/exec/moe"
	"github.com/magomedcoder/gogguf/pkg/weights"
)

// Model - DeepSeek / DeepSeek-MoE (RMSNorm, Llama RoPE, SwiGLU; dense lead + MoE)
type Model struct {
	cfg          Config
	weights      *weights.Store
	cache        *KVCache
	dev          device.Backend
	ngl          int
	gpuMaxSeq    int
	nBatch       int              // chunk prefill size (1 = per token); dense models only
	fused        *gpuresid.Runner // fused GPU paths: FFN, MoE experts
	moeGPU       []bool           // layer can run experts on GPU
	scratch      scratch
	layerNorms   []layerNorms
	layerTensors []layerTensors
	outNorm      []float32
	lmHeadName   string
	logitsOnGPU  bool // out_norm + lm_head on GPU from the final host hidden state
}

// Load creates DeepSeek from weights
func Load(w *weights.Store, g device.Backend, ngl, gpuMaxSeq, nBatch int) (*Model, error) {
	cfg, err := ParseConfig(w.Reader())
	if err != nil {
		return nil, err
	}

	if ngl > cfg.NumLayers {
		ngl = cfg.NumLayers
	}

	nBatch = min(max(nBatch, 1), 512)

	// MoE layers (and the dense lead of MoE models): serial only for now (n_batch=1)
	if cfg.ExpertCount > 0 {
		nBatch = 1
	}

	layerNorms, outNorm, err := loadNormWeights(w, cfg.NumLayers)
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

	m := &Model{
		cfg:          cfg,
		weights:      w,
		cache:        NewKVCache(cfg),
		dev:          g,
		ngl:          ngl,
		gpuMaxSeq:    gpuMaxSeq,
		nBatch:       nBatch,
		scratch:      newScratch(cfg, nBatch),
		layerNorms:   layerNorms,
		layerTensors: layerTensors,
		outNorm:      outNorm,
		lmHeadName:   lmHeadName,
	}

	if err := m.initGPUKVCache(); err != nil {
		return nil, err
	}

	m.initFused()

	return m, nil
}

// initFused sets up GPU layer paths. DeepSeek v1: RoPE NORM, experts via ExpertFFN.
//
// MoE (P1 deferred): full device-resident MoE layers need batched expert dispatch (router top-k, then per-expert FFN and accumulation on device) and n_batch>1 support.
// Until then MoE layers keep the per-token ExpertFFN path (n_batch=1, hidden state on host); only out_norm + lm_head may run on GPU via gpuresid.LogitsFromHost when ngl covers every layer.
func (m *Model) initFused() {
	if m.dev == nil || m.ngl <= 0 {
		return
	}

	m.fused = gpuresid.New(m.weights, m.dev, device.RoPENorm)
	m.logitsOnGPU = gpuresid.CanLogitsFromDevice(m.dev, m.ngl, m.cfg.NumLayers) && m.fused.HeadOnGPU(m.lmHeadName)
	m.moeGPU = make([]bool, m.cfg.NumLayers)
	for i, lt := range m.layerTensors {
		if !lt.moe {
			continue
		}

		m.moeGPU[i] = m.fused.MoESupported(lt.gateExps, lt.upExps, lt.downExps)
	}
}

func (m *Model) layerOnGPU(layer int) bool {
	return m.fused != nil && device.LayerOnGPU(layer, m.ngl, m.cfg.NumLayers)
}

func (m *Model) initGPUKVCache() error {
	if m.dev == nil || m.ngl <= 0 {
		return nil
	}

	kvDim := m.cfg.NumKVHeads * m.cfg.HeadDim
	maxSeq := device.CapMaxSeq(m.cfg.ContextLength, m.gpuMaxSeq)

	return m.dev.KVCacheInit(m.ngl, maxSeq, kvDim, m.cfg.NumHeads, m.cfg.HeadDim)
}

// Config returns the model configuration
func (m *Model) Config() Config {
	return m.cfg
}

// ResetCache clears the KV-cache
func (m *Model) ResetCache() {
	m.cache.Reset()
	if m.dev != nil {
		m.dev.KVCacheReset()
	}
}

// Close releases GPU resources held by the model
func (m *Model) Close() error {
	if m.dev == nil {
		return nil
	}
	err := m.dev.Close()
	m.dev = nil
	m.fused = nil
	return err
}

// Forward runs forward pass for tokenIDs starting at startPos
// Returns logits for the last token [vocabSize].
// The slice is the model's internal scratch buffer (no copy): the caller must not retain or modify it across the next Forward / Embed call
func (m *Model) Forward(tokenIDs []int, startPos int) ([]float32, error) {
	if len(tokenIDs) == 0 {
		return nil, fmt.Errorf("deepseek: empty input")
	}

	// n_batch>1 (dense, ExpertCount==0): multi-token prefill; MoE - serial
	if m.nBatch > 1 && m.cfg.ExpertCount == 0 && len(tokenIDs) > 1 {
		for i := 0; i < len(tokenIDs); {
			end := min(i+m.nBatch, len(tokenIDs))

			needLogits := end == len(tokenIDs)
			if err := m.forwardBatch(tokenIDs[i:end], startPos+i, needLogits); err != nil {
				return nil, err
			}
			i = end
		}
	} else {
		for i, tok := range tokenIDs {
			if err := m.forwardToken(tok, startPos+i); err != nil {
				return nil, err
			}
		}

		if err := m.logitsFromLast(m.scratch.x); err != nil {
			return nil, err
		}
	}

	return m.scratch.logits, nil
}

// EmbeddingDim returns the hidden state dimension
func (m *Model) EmbeddingDim() int {
	return m.cfg.EmbeddingDim
}

// Embed - last-token RMSNorm(hidden) before lm_head (clears KV)
func (m *Model) Embed(tokenIDs []int) ([]float32, error) {
	if len(tokenIDs) == 0 {
		return nil, fmt.Errorf("deepseek: empty input")
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
	return dispatch.EmbedTokenInto(m.weights, m.scratch.x, m.cfg.EmbeddingDim, tokenID)
}

func (m *Model) forwardBlock(layer int, pos int) error {
	ln := m.layerNorms[layer]
	lt := m.layerTensors[layer]

	if err := compute.RMSNormInto(m.scratch.h, m.scratch.x, ln.attnNorm, m.cfg.RMSNormEps); err != nil {
		return err
	}

	if err := m.matmulInto(lt.attnQ, m.cfg.NumHeads*m.cfg.HeadDim, m.cfg.EmbeddingDim, m.scratch.h, m.scratch.q, layer); err != nil {
		return err
	}

	if err := m.matmulInto(lt.attnK, m.cfg.NumKVHeads*m.cfg.HeadDim, m.cfg.EmbeddingDim, m.scratch.h, m.scratch.k, layer); err != nil {
		return err
	}

	if err := m.matmulInto(lt.attnV, m.cfg.NumKVHeads*m.cfg.HeadDim, m.cfg.EmbeddingDim, m.scratch.h, m.scratch.v, layer); err != nil {
		return err
	}

	compute.ApplyRoPEHeadsNorm(m.scratch.q, m.cfg.NumHeads, m.cfg.HeadDim, pos, m.cfg.RopeFreqBase)
	compute.ApplyRoPEHeadsNorm(m.scratch.k, m.cfg.NumKVHeads, m.cfg.HeadDim, pos, m.cfg.RopeFreqBase)

	kvPos := m.cache.Len()
	if err := m.cache.Append(layer, m.scratch.k, m.scratch.v); err != nil {
		return err
	}

	if m.dev != nil && device.LayerOnGPU(layer, m.ngl, m.cfg.NumLayers) {
		_ = m.dev.KVCacheAppend(layer, kvPos, m.scratch.k, m.scratch.v)
	}

	seqLen := m.cache.Len() + 1
	if err := m.attentionScoresInto(m.scratch.attn, m.scratch.q, m.cache.KLayer(layer), m.cache.VLayer(layer), m.scratch.scores, seqLen, layer); err != nil {
		return err
	}

	if err := m.matmulInto(lt.attnOut, m.cfg.EmbeddingDim, m.cfg.NumHeads*m.cfg.HeadDim, m.scratch.attn, m.scratch.h, layer); err != nil {
		return err
	}
	compute.AddInPlace(m.scratch.x, m.scratch.h)

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

	if m.layerOnGPU(layer) {
		if err := m.fused.FFNNamed(lt.ffnGate, lt.ffnUp, lt.ffnDown, m.scratch.h, m.scratch.h, m.cfg.EmbeddingDim, m.cfg.FFNHidden); err == nil {
			compute.AddInPlace(m.scratch.x, m.scratch.h)
			return nil
		}
	}

	if m.dev != nil && device.LayerOnGPU(layer, m.ngl, m.cfg.NumLayers) {
		if err := m.ffnGPU(lt, m.scratch.h, m.scratch.h); err == nil {
			compute.AddInPlace(m.scratch.x, m.scratch.h)
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
	if err := m.matmulInto(lt.ffnDown, m.cfg.EmbeddingDim, m.cfg.FFNHidden, m.scratch.gate[:m.cfg.FFNHidden], m.scratch.h, layer); err != nil {
		return err
	}
	compute.AddInPlace(m.scratch.x, m.scratch.h)

	return nil
}

func (m *Model) ffnMoE(lt layerTensors, layer int) error {
	embd := m.cfg.EmbeddingDim
	nExp := m.cfg.ExpertCount
	ffn := m.cfg.ExpertFFN

	// router matmul on GPU via MatMulInto (same quant cache)
	if err := m.matmulInto(lt.gateInp, nExp, embd, m.scratch.h, m.scratch.router[:nExp], layer); err != nil {
		return err
	}

	idxs, weights := moe.TopKSoftmax(m.scratch.router[:nExp], m.cfg.ExpertUsedCount, false, m.cfg.ExpertWeightScale)

	clear(m.scratch.moeAcc)

	// selected experts - fused FFN on GPU from raw weight slice
	if m.moeExpertsGPU(lt, layer, idxs, weights) {
		return m.sharedExpert(lt, layer)
	}

	if err := moe.ExpertsSwiGLUInto(
		m.weights,
		lt.gateExps,
		lt.upExps,
		lt.downExps,
		idxs, weights,
		m.scratch.h,
		m.scratch.gate[:ffn],
		m.scratch.up[:ffn],
		m.scratch.tmp,
		m.scratch.moeAcc,
		embd,
		ffn,
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

// sharedExpert adds layer shared expert into moeAcc (if present)
func (m *Model) sharedExpert(lt layerTensors, layer int) error {
	if m.cfg.ExpertShared <= 0 || lt.gateShexp == "" {
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

func (m *Model) ffnGPU(lt layerTensors, x, out []float32) error {
	info, err := m.weights.Info(lt.ffnGate)
	if err != nil {
		return err
	}

	embd, ffn := m.cfg.EmbeddingDim, m.cfg.FFNHidden
	if info.Type == dtype.Q8_0 {
		gateRaw, err := m.weights.Raw(lt.ffnGate)
		if err != nil {
			return err
		}

		upRaw, err := m.weights.Raw(lt.ffnUp)
		if err != nil {
			return err
		}

		downRaw, err := m.weights.Raw(lt.ffnDown)
		if err != nil {
			return err
		}

		return m.dev.FFNSwiGLUQ8_0Cached(lt.ffnGate, lt.ffnUp, lt.ffnDown, gateRaw, upRaw, downRaw, x, out, embd, ffn)
	}

	gateW, err := m.weights.Floats(lt.ffnGate)
	if err != nil {
		return err
	}

	upW, err := m.weights.Floats(lt.ffnUp)
	if err != nil {
		return err
	}

	downW, err := m.weights.Floats(lt.ffnDown)
	if err != nil {
		return err
	}

	return m.dev.FFNSwiGLUCached(lt.ffnGate, lt.ffnUp, lt.ffnDown, gateW, upW, downW, x, out, embd, ffn)
}

func (m *Model) matmulInto(name string, rows, cols int, vec, out []float32, layer int) error {
	if m.layerOnGPU(layer) {
		if err := m.fused.MatMulInto(name, rows, cols, vec, out); err == nil {
			return nil
		}
	}

	if m.dev != nil && device.LayerOnGPU(layer, m.ngl, m.cfg.NumLayers) {
		return dispatch.MatMulGPUInto(m.dev, m.weights, name, rows, cols, vec, out)
	}

	return dispatch.MatMulRawInto(m.weights, name, rows, cols, vec, out)
}

// logitsFromLast: lm_head on GPU (hidden uploaded once) when enabled, else on host
func (m *Model) logitsFromLast(x []float32) error {
	if m.logitsOnGPU && m.fused != nil {
		err := m.fused.LogitsFromHost(x, "output_norm.weight", m.outNorm, m.lmHeadName, m.scratch.logits, m.cfg.VocabSize, m.cfg.EmbeddingDim, m.cfg.RMSNormEps)
		if err == nil {
			return nil
		}

		m.logitsOnGPU = false
	}

	return m.logitsFromHidden(x)
}

func (m *Model) logitsFromHidden(x []float32) error {
	return dispatch.LogitsFromHiddenInto(m.weights, m.scratch.h, x, m.outNorm, m.scratch.logits, m.lmHeadName, m.cfg.VocabSize, m.cfg.EmbeddingDim, m.cfg.RMSNormEps)
}

func (m *Model) attentionScoresInto(dst, q, k, v, scores []float32, seqLen, layer int) error {
	if m.dev != nil && device.LayerOnGPU(layer, m.ngl, m.cfg.NumLayers) {
		if err := m.dev.AttentionScoresKV(layer, dst, q, seqLen, m.cfg.NumHeads, m.cfg.NumKVHeads, m.cfg.HeadDim); err == nil {
			return nil
		}

		if err := m.dev.AttentionScoresInto(dst, q, k, v, scores, seqLen, m.cfg.NumHeads, m.cfg.NumKVHeads, m.cfg.HeadDim); err == nil {
			return nil
		}
	}

	return compute.AttentionScoresInto(dst, q, k, v, scores, seqLen, m.cfg.NumHeads, m.cfg.NumKVHeads, m.cfg.HeadDim)
}
