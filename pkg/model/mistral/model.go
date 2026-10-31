package mistral

import (
	"fmt"

	"github.com/magomedcoder/gogguf/pkg/compute"
	"github.com/magomedcoder/gogguf/pkg/device"
	"github.com/magomedcoder/gogguf/pkg/exec/dispatch"
	"github.com/magomedcoder/gogguf/pkg/exec/gpuresid"
	"github.com/magomedcoder/gogguf/pkg/exec/moe"
	"github.com/magomedcoder/gogguf/pkg/format"
	"github.com/magomedcoder/gogguf/pkg/weights"
)

// Model - Mistral transformer (NeoX RoPE, GQA, SwiGLU)
type Model struct {
	cfg          Config
	weights      *weights.Store
	cache        *KVCache
	dev          device.Backend
	ngl          int
	gpuMaxSeq    int
	nBatch       int // chunk prefill size (1 = per token); decode always 1
	scratch      scratch
	layerNorms   []layerNorms
	layerTensors []layerTensors
	outNorm      []float32
	lmHeadName   string
	debug        *DebugHooks

	fused      *gpuresid.Runner   // fused GPU layer paths
	fusedDims  gpuresid.Dims      // dimensions for fused paths
	gpuLayers  []gpuresid.Tensors // layer weight names for fused paths
	layerQuant []bool             // layer fully in supported fused quant
	moeGPU     bool               // MoE experts computed on GPU

	residency   bool // layers can keep hidden state on device
	residDevice bool // hidden state is on device; host buffer x is stale
	logitsOnGPU bool // out_norm + lm_head computed on GPU
	gpuKVStale  bool // GPU KV-cache incomplete: attention on CPU only
	cpuKVHollow bool // CPU KV slots bumped without K/V data (residency decode)
}

// Load creates Mistral from weights (mistral.* prefix)
func Load(w *weights.Store, g device.Backend, ngl, gpuMaxSeq, nBatch int) (*Model, error) {
	return loadWithConfig(w, g, ngl, gpuMaxSeq, nBatch, ParseConfig)
}

// LoadLlamaMeta creates Mistral from GGUF with general.architecture=llama (TheBloke, convert.py)
func LoadLlamaMeta(w *weights.Store, g device.Backend, ngl, gpuMaxSeq, nBatch int) (*Model, error) {
	return loadWithConfig(w, g, ngl, gpuMaxSeq, nBatch, ParseConfigLlama)
}

// LoadQwen2 creates a model with qwen2.* prefix (NeoX RoPE, no QK-norm; Distill-Qwen, etc.)
func LoadQwen2(w *weights.Store, g device.Backend, ngl, gpuMaxSeq, nBatch int) (*Model, error) {
	return loadWithConfig(w, g, ngl, gpuMaxSeq, nBatch, ParseConfigQwen2)
}

// LoadQwen2MoE creates Qwen2-MoE (qwen2moe.*, gated shared expert)
func LoadQwen2MoE(w *weights.Store, g device.Backend, ngl, gpuMaxSeq, nBatch int) (*Model, error) {
	return loadWithConfig(w, g, ngl, gpuMaxSeq, nBatch, ParseConfigQwen2MoE)
}

func loadWithConfig(w *weights.Store, g device.Backend, ngl, gpuMaxSeq, nBatch int, parse func(*format.Reader) (Config, error)) (*Model, error) {
	cfg, err := parse(w.Reader())
	if err != nil {
		return nil, err
	}

	if ngl > cfg.NumLayers {
		ngl = cfg.NumLayers
	}

	nBatch = min(max(nBatch, 1), 512)

	// MoE: serial only for now (n_batch=1)
	if cfg.isMoE() {
		nBatch = 1
	}

	// SWA: batch attention does not apply the sliding window - serial only
	if cfg.SlidingWindow > 0 && cfg.SlidingWindow < cfg.ContextLength {
		nBatch = 1
	}

	layerNorms, outNorm, err := loadNormWeights(w, cfg.NumLayers)
	if err != nil {
		return nil, err
	}

	lmHeadName, err := resolveLMHeadName(w)
	if err != nil {
		return nil, err
	}

	layerTensors, err := loadLayerTensors(w, cfg)
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
	m.initResidency()

	return m, nil
}

// ensureKVRoom reclaims oldest KV when the sliding-window cache is full.
func (m *Model) ensureKVRoom(n int) {
	if m.cfg.SlidingWindow <= 0 {
		return
	}

	if dropped := m.cache.EnsureRoom(n); dropped > 0 && m.dev != nil {
		m.gpuKVStale = true
	}
}

func (m *Model) initGPUKVCache() error {
	if m.dev == nil || m.ngl <= 0 {
		return nil
	}

	kvDim := m.cfg.NumKVHeads * m.cfg.HeadDim
	maxSeq := device.CapMaxSeq(kvMaxSeq(m.cfg), m.gpuMaxSeq)

	return m.dev.KVCacheInit(m.ngl, maxSeq, kvDim, m.cfg.NumHeads, m.cfg.HeadDim)
}

// Config returns the model configuration
func (m *Model) Config() Config {
	return m.cfg
}

// ResetCache clears the KV-cache
func (m *Model) ResetCache() {
	m.cache.Reset()
	m.residDevice = false
	m.gpuKVStale = false
	m.cpuKVHollow = false
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

	return err
}

// SetDebugHooks enables callbacks for step-by-step forward pass debugging
func (m *Model) SetDebugHooks(h *DebugHooks) {
	m.debug = h
}

// Forward runs forward pass for tokenIDs starting at startPos
// Returns logits for the last token [vocabSize]. The slice is the model's
// internal scratch buffer (no copy): the caller must not retain or modify it
// across the next Forward / Embed call
func (m *Model) Forward(tokenIDs []int, startPos int) ([]float32, error) {
	if len(tokenIDs) == 0 {
		return nil, fmt.Errorf("mistral: empty input")
	}

	// n_batch>1: multi-token prefill (chunk matmul on GPU with -ngl, else CPU GEMM). MoE/SWA/debug - serial.
	if m.nBatch > 1 && m.debug == nil && len(tokenIDs) > 1 {
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
			pos := startPos + i
			last := i == len(tokenIDs)-1
			if err := m.forwardToken(tok, pos, last); err != nil {
				return nil, err
			}
		}

		if err := m.logitsFinish(); err != nil {
			return nil, err
		}
	}

	if m.debug != nil && m.debug.OnLogits != nil {
		m.debug.OnLogits(m.scratch.logits)
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
		return nil, fmt.Errorf("mistral: empty input")
	}
	m.ResetCache()
	defer m.ResetCache()

	for i, tok := range tokenIDs {
		if err := m.forwardToken(tok, i, false); err != nil {
			return nil, err
		}
	}

	// Embed reads hidden on host: with residency, sync it from the device first
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

func (m *Model) forwardToken(tokenID, pos int, debug bool) error {
	m.ensureKVRoom(1)

	if err := m.embedToken(tokenID); err != nil {
		return err
	}

	if debug && m.debug != nil && m.debug.OnEmbed != nil {
		m.debug.OnEmbed(m.scratch.x)
	}

	// single HtoD hidden state per token; layers then run on device
	m.uploadHidden()

	for layer := 0; layer < m.cfg.NumLayers; layer++ {
		if err := m.forwardBlock(layer, pos); err != nil {
			return err
		}

		if debug && m.debug != nil {
			if m.debug.OnLayer != nil {
				m.debug.OnLayer(layer, m.scratch.x)
			}

			if m.debug.OnLayerLogits != nil {
				if err := m.logitsFromHidden(m.scratch.x); err != nil {
					return err
				}

				m.debug.OnLayerLogits(layer, m.scratch.logits)
			}
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
	onGPU := m.dev != nil && device.LayerOnGPU(layer, m.ngl, m.cfg.NumLayers)
	usedQKVResidency := false

	// entire layer on device - attn_norm, QKV, attention, WO, FFN, and both residuals
	if m.residDevice {
		err := m.forwardBlockDevice(layer, pos, ln)
		if err == nil {
			return nil
		}

		if m.cpuKVHollow {
			m.ResetCache()
			return fmt.Errorf("mistral: layer %d residency failed without CPU KV mirror: %w", layer, err)
		}

		if rerr := m.recoverHiddenFromDevice(layer, err); rerr != nil {
			return rerr
		}
	}

	if err := m.rmsNormInto(m.scratch.h, m.scratch.x, ln.attnNorm, layer); err != nil {
		return err
	}

	// fused QKV attends over full GPU KV-cache: SWA window and stale cache are incompatible
	if onGPU && gpuresid.QKVResidencyEnabled() && !m.gpuKVStale && m.gpuAttnFull(m.cache.Len()+1) {
		if err := m.qkvAttnGPU(layer, pos, ln, m.scratch.h, m.scratch.k, m.scratch.v, m.scratch.attn); err == nil {
			if err := m.cache.Append(layer, m.scratch.k, m.scratch.v); err != nil {
				return err
			}

			usedQKVResidency = true
		}
	}

	if !usedQKVResidency {
		if err := m.matmulInto(lt.attnQ, m.cfg.NumHeads*m.cfg.HeadDim, m.cfg.EmbeddingDim, m.scratch.h, m.scratch.q, layer); err != nil {
			return err
		}

		if err := m.matmulInto(lt.attnK, m.cfg.NumKVHeads*m.cfg.HeadDim, m.cfg.EmbeddingDim, m.scratch.h, m.scratch.k, layer); err != nil {
			return err
		}

		if err := m.matmulInto(lt.attnV, m.cfg.NumKVHeads*m.cfg.HeadDim, m.cfg.EmbeddingDim, m.scratch.h, m.scratch.v, layer); err != nil {
			return err
		}

		m.applyRoPEHeads(m.scratch.q, m.cfg.NumHeads, pos, layer)
		m.applyRoPEHeads(m.scratch.k, m.cfg.NumKVHeads, pos, layer)

		kvPos := m.cache.Len()
		if err := m.cache.Append(layer, m.scratch.k, m.scratch.v); err != nil {
			return err
		}

		if onGPU && !m.gpuKVStale {
			if err := m.dev.KVCacheAppend(layer, kvPos, m.scratch.k, m.scratch.v); err != nil {
				m.gpuKVStale = true
			}
		}

		seqLen := m.cache.Len() + 1
		k, v := m.attentionKV(layer, seqLen)

		if err := m.attentionScoresInto(m.scratch.attn, m.scratch.q, k, v, m.scratch.scores, seqLen, layer); err != nil {
			return err
		}
	}

	// WO+RMSNorm+FFN residency: MoE layers skip this; experts compute their FFN
	if !lt.moe && onGPU && gpuresid.AttnFFNResidencyEnabled() {
		if err := m.attnFFNGPU(layer, ln, m.scratch.x, m.scratch.attn); err == nil {
			return nil
		}
	}

	if err := m.matmulInto(lt.attnOut, m.cfg.EmbeddingDim, m.cfg.NumHeads*m.cfg.HeadDim, m.scratch.attn, m.scratch.h, layer); err != nil {
		return err
	}
	compute.AddInPlace(m.scratch.x, m.scratch.h)

	if err := m.rmsNormInto(m.scratch.h, m.scratch.x, ln.ffnNorm, layer); err != nil {
		return err
	}

	if lt.moe {
		if err := m.ffnMoE(lt, layer); err != nil {
			return err
		}

		compute.AddInPlace(m.scratch.x, m.scratch.moeAcc)
		return nil
	}

	if onGPU {
		if err := m.ffnGPU(layer, m.scratch.h, m.scratch.h); err == nil {
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
	m.swigluInPlace(m.scratch.gate, m.scratch.up, layer)

	if err := m.matmulInto(lt.ffnDown, m.cfg.EmbeddingDim, m.cfg.FFNHidden, m.scratch.gate, m.scratch.h, layer); err != nil {
		return err
	}

	compute.AddInPlace(m.scratch.x, m.scratch.h)
	return nil
}

func (m *Model) ffnMoE(lt layerTensors, layer int) error {
	embd := m.cfg.EmbeddingDim
	nExp := m.cfg.ExpertCount
	ffn := m.cfg.expertFFN()

	if err := m.matmulInto(lt.gateInp, nExp, embd, m.scratch.h, m.scratch.router[:nExp], layer); err != nil {
		return err
	}

	idxs, weights := moe.TopKSoftmax(m.scratch.router[:nExp], m.cfg.ExpertUsedCount, m.cfg.MoENormWeights, m.cfg.ExpertWeightScale)

	clear(m.scratch.moeAcc)

	// selected experts - fused FFN on GPU from expert raw weight slice
	if m.moeGPU && device.LayerOnGPU(layer, m.ngl, m.cfg.NumLayers) {
		if err := m.moeExpertsGPU(lt, idxs, weights, ffn); err == nil {
			return m.sharedExpert(lt, layer)
		}

		m.moeGPU = false
		clear(m.scratch.moeAcc)
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

// moeExpertsGPU runs selected experts as fused-FFN on GPU and accumulates weighted sum
func (m *Model) moeExpertsGPU(lt layerTensors, idxs []int, weights []float32, ffn int) error {
	embd := m.cfg.EmbeddingDim
	for i, ei := range idxs {
		if err := m.fused.ExpertFFN(lt.gateExps, lt.upExps, lt.downExps, ei, m.scratch.h, m.scratch.tmp, embd, ffn); err != nil {
			return err
		}

		w := weights[i]
		for j := range embd {
			m.scratch.moeAcc[j] += m.scratch.tmp[j] * w
		}
	}

	return nil
}

// sharedExpert adds shared expert (DeepSeek / Qwen2-MoE) to expert sum
func (m *Model) sharedExpert(lt layerTensors, layer int) error {
	if lt.gateShexp == "" {
		return nil
	}

	embd := m.cfg.EmbeddingDim
	shared := m.cfg.SharedFFN
	if shared <= 0 {
		shared = m.cfg.FFNHidden
	}

	gateScale := float32(1)
	if lt.gateInpShexp != "" {
		var gateInp [1]float32
		if err := m.matmulInto(lt.gateInpShexp, 1, embd, m.scratch.h, gateInp[:], layer); err != nil {
			return err
		}

		gateScale = siluDiv(gateInp[0])
	}

	// shared expert uses same fused FFN on same resident weight buffer
	done := false
	if m.fused != nil && device.LayerOnGPU(layer, m.ngl, m.cfg.NumLayers) {
		if err := m.fused.FFNNamed(lt.gateShexp, lt.upShexp, lt.downShexp, m.scratch.h, m.scratch.tmp, embd, shared); err == nil {
			done = true
		}
	}

	if !done {
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
	}

	if gateScale != 1 {
		compute.ScaleInPlace(m.scratch.tmp, gateScale)
	}

	compute.AddInPlace(m.scratch.moeAcc, m.scratch.tmp)

	return nil
}

func siluDiv(x float32) float32 {
	if x == 0 {
		return 0.5
	}
	return compute.SiLU(x) / x
}

func (m *Model) ffnGPU(layer int, x, out []float32) error {
	return m.fused.FFN(m.gpuLayers[layer], m.fusedDims, x, out)
}

// attentionKV returns K/V for attention respecting the sliding window
func (m *Model) attentionKV(layer, seqLen int) (k, v []float32) {
	k = m.cache.KLayer(layer)
	v = m.cache.VLayer(layer)

	w := m.cfg.SlidingWindow
	if w <= 0 || seqLen <= w {
		return k, v
	}

	kvDim := m.cfg.NumKVHeads * m.cfg.HeadDim
	skip := (seqLen - 1 - w) * kvDim
	if skip <= 0 || skip >= len(k) {
		return k, v
	}

	return k[skip:], v[skip:]
}

func (m *Model) matmulInto(name string, rows, cols int, vec, out []float32, layer int) error {
	if m.dev != nil && device.LayerOnGPU(layer, m.ngl, m.cfg.NumLayers) {
		return dispatch.MatMulGPUInto(m.dev, m.weights, name, rows, cols, vec, out)
	}

	return dispatch.MatMulRawInto(m.weights, name, rows, cols, vec, out)
}

func (m *Model) logitsFinish() error {
	// hidden on device - out_norm + lm_head on GPU too; only logits go to host
	if m.residDevice && m.logitsOnGPU {
		if err := m.logitsDevice(); err == nil {
			return nil
		}

		m.logitsOnGPU = false
	}

	if err := m.syncHiddenFromDevice(); err != nil {
		return err
	}

	// hidden on host (MoE): lm_head on GPU after one upload
	if m.logitsOnGPU && m.cfg.isMoE() {
		if err := m.logitsFromHost(); err == nil {
			return nil
		}

		m.logitsOnGPU = false
	}

	return m.logitsFromHidden(m.scratch.x)
}

func (m *Model) logitsFromHidden(x []float32) error {
	return dispatch.LogitsFromHiddenInto(m.weights, m.scratch.h, x, m.outNorm, m.scratch.logits, m.lmHeadName, m.cfg.VocabSize, m.cfg.EmbeddingDim, m.cfg.RMSNormEps)
}

func (m *Model) rmsNormInto(dst, x, weight []float32, layer int) error {
	_ = layer
	return compute.RMSNormInto(dst, x, weight, m.cfg.RMSNormEps)
}

func (m *Model) applyRoPEHeads(v []float32, nHeads, pos, layer int) {
	_ = layer
	compute.ApplyRoPEHeads(v, nHeads, m.cfg.HeadDim, pos, m.cfg.RopeFreqBase)
}

func (m *Model) swigluInPlace(gate, up []float32, layer int) {
	if m.dev != nil && device.LayerOnGPU(layer, m.ngl, m.cfg.NumLayers) {
		if err := m.dev.SwiGLUInPlace(gate, up); err == nil {
			return
		}
	}
	compute.SwiGLUInPlace(gate, up)
}

func (m *Model) attentionScoresInto(dst, q, k, v, scores []float32, seqLen, layer int) error {
	w := m.cfg.SlidingWindow
	effectiveLen := seqLen
	if w > 0 && seqLen > w {
		effectiveLen = w
	}

	if m.dev != nil && device.LayerOnGPU(layer, m.ngl, m.cfg.NumLayers) {
		// gpuKVStale: some tokens missed GPU KV-cache - attention uses CPU mirror only
		if effectiveLen == seqLen && !m.gpuKVStale {
			if err := m.dev.AttentionScoresKV(layer, dst, q, seqLen, m.cfg.NumHeads, m.cfg.NumKVHeads, m.cfg.HeadDim); err == nil {
				return nil
			}
		}

		if err := m.dev.AttentionScoresInto(dst, q, k, v, scores, effectiveLen, m.cfg.NumHeads, m.cfg.NumKVHeads, m.cfg.HeadDim); err == nil {
			return nil
		}
	}

	return compute.AttentionScoresInto(dst, q, k, v, scores, effectiveLen, m.cfg.NumHeads, m.cfg.NumKVHeads, m.cfg.HeadDim)
}
