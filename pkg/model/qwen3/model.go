package qwen3

import (
	"fmt"
	"strings"

	"github.com/magomedcoder/gogguf/pkg/compute"
	"github.com/magomedcoder/gogguf/pkg/device"
	"github.com/magomedcoder/gogguf/pkg/dtype"
	"github.com/magomedcoder/gogguf/pkg/exec/dispatch"
	"github.com/magomedcoder/gogguf/pkg/exec/gpuresid"
	"github.com/magomedcoder/gogguf/pkg/exec/moe"
	"github.com/magomedcoder/gogguf/pkg/format"
	"github.com/magomedcoder/gogguf/pkg/weights"
)

// Model - Qwen3 transformer
type Model struct {
	cfg          Config
	weights      *weights.Store
	cache        *KVCache
	dev          device.Backend
	ngl          int
	gpuMaxSeq    int
	nBatch       int // chunk prefill size (1 = per token); decode always 1
	debug        *DebugHooks
	scratch      scratch
	layerNorms   []layerNorms
	layerTensors []layerTensors
	outNorm      []float32
	lmHeadName   string

	layerQuant  []dtype.DType // matmul weight type per layer for fused paths
	residency   bool          // layers can keep hidden state on device
	residDevice bool          // hidden state is on device; host buffer x is stale
	logitsOnGPU bool          // out_norm + lm_head computed on GPU
	gpuKVStale  bool          // GPU KV-cache incomplete: attention on CPU only
	cpuKVHollow bool          // CPU KV slots bumped without K/V data (residency decode)

	fused     *gpuresid.Runner   // fused GPU layer paths
	fusedDims gpuresid.Dims      // dimensions for fused paths
	gpuLayers []gpuresid.Tensors // layer weight names for fused paths
	moeGPU    bool               // MoE experts computed on GPU
}

// Load creates Qwen3 from weights
func Load(w *weights.Store, g device.Backend, ngl, gpuMaxSeq, nBatch int) (*Model, error) {
	return loadWithConfig(w, g, ngl, gpuMaxSeq, nBatch, ParseConfig)
}

// LoadMoE creates Qwen3-MoE (qwen3moe.*, QK-norm + MoE)
func LoadMoE(w *weights.Store, g device.Backend, ngl, gpuMaxSeq, nBatch int) (*Model, error) {
	return loadWithConfig(w, g, ngl, gpuMaxSeq, nBatch, ParseConfigMoE)
}

func loadWithConfig(w *weights.Store, g device.Backend, ngl, gpuMaxSeq, nBatch int, parse func(*format.Reader) (Config, error)) (*Model, error) {
	cfg, err := parse(w.Reader())
	if err != nil {
		return nil, err
	}

	if ngl > cfg.NumLayers {
		ngl = cfg.NumLayers
	}

	if nBatch < 1 {
		nBatch = 1
	}

	if nBatch > 512 {
		nBatch = 512
	}

	// MoE: serial only for now (n_batch=1)
	if cfg.ExpertCount > 0 {
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

	m.layerQuant = resolveLayerQuants(w, layerTensors)
	m.initFused()
	m.initResidency()

	return m, nil
}

// resolveLayerQuants returns matmul weight type per layer; dtype.F32 means no unified quant type - that layer uses FP32/discrete path
func resolveLayerQuants(w *weights.Store, lts []layerTensors) []dtype.DType {
	kinds := make([]dtype.DType, len(lts))
	for i, lt := range lts {
		names := []string{lt.attnQ, lt.attnK, lt.attnV, lt.attnOut}
		if !lt.moe {
			names = append(names, lt.ffnGate, lt.ffnUp, lt.ffnDown)
		}

		kinds[i] = gpuresid.LayerQuant(w, names...)
	}

	return kinds
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
// Returns logits for the last token [vocabSize]
func (m *Model) Forward(tokenIDs []int, startPos int) ([]float32, error) {
	if len(tokenIDs) == 0 {
		return nil, fmt.Errorf("qwen3: empty input")
	}

	// n_batch>1: multi-token prefill; with -ngl chunk matmul/attn on GPU (loop MatMulVec*Cached + AttentionScoresKV), else CPU GEMM. MoE/debug - serial.
	useBatch := m.nBatch > 1 && m.cfg.ExpertCount == 0 && m.debug == nil && len(tokenIDs) > 1
	if useBatch {
		for i := 0; i < len(tokenIDs); {
			end := min(i+m.nBatch, len(tokenIDs))

			needLogits := end == len(tokenIDs)
			if err := m.forwardBatch(tokenIDs[i:end], startPos+i, needLogits); err != nil {
				return nil, err
			}
			i = end
		}
	} else {
		var err error
		for i, tok := range tokenIDs {
			pos := startPos + i
			last := i == len(tokenIDs)-1
			if err = m.forwardToken(tok, pos, last); err != nil {
				return nil, err
			}
		}

		if err = m.logitsFinish(); err != nil {
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
		return nil, fmt.Errorf("qwen3: empty input")
	}
	m.ResetCache()
	defer m.ResetCache()

	embd := m.cfg.EmbeddingDim
	for i, tok := range tokenIDs {
		if err := m.forwardToken(tok, i, false); err != nil {
			return nil, err
		}
	}

	// Embed reads hidden on host: with residency, sync it from the device first
	if err := m.syncHiddenFromDevice(); err != nil {
		return nil, err
	}

	x := m.scratch.x[:embd]
	h := m.scratch.h[:embd]
	if err := compute.RMSNormInto(h, x, m.outNorm, m.cfg.RMSNormEps); err != nil {
		return nil, err
	}

	out := make([]float32, embd)
	copy(out, h)
	return out, nil
}

func (m *Model) forwardToken(tokenID, pos int, debug bool) error {
	// Prefer GPU gather into d_resid (no host emb HtoD); else CPU embed + HiddenUpload.
	gathered := false
	if m.residency && m.debug == nil && !m.gpuKVStale && m.dev != nil {
		if raw, err := m.weights.Raw("token_embd.weight"); err == nil {
			if info, err := m.weights.Info("token_embd.weight"); err == nil {
				if err := device.TryEmbedGather(m.dev, "token_embd.weight", raw, info.Type, tokenID, m.cfg.VocabSize, m.cfg.EmbeddingDim); err == nil {
					m.residDevice = true
					gathered = true
				}
			}
		}
	}
	if !gathered {
		if err := m.embedToken(tokenID); err != nil {
			return err
		}
	}

	if debug && m.debug != nil && m.debug.OnEmbed != nil {
		if gathered {
			_ = m.syncHiddenFromDevice()
		}
		m.debug.OnEmbed(m.scratch.x[:m.cfg.EmbeddingDim])
	}

	// single HtoD hidden state per token when gather missed; layers then run on device.
	// Debug hooks read host buffer x, so residency is disabled with them.
	// gpuKVStale: fused QKV attends using GPU KV-cache, which is incomplete.
	if !gathered && m.residency && m.debug == nil && !m.gpuKVStale {
		if err := m.dev.HiddenUpload(m.scratch.x[:m.cfg.EmbeddingDim]); err != nil {
			m.residency = false
		} else {
			m.residDevice = true
		}
	}

	for layer := 0; layer < m.cfg.NumLayers; layer++ {
		if err := m.forwardBlock(layer, pos); err != nil {
			return err
		}

		if debug && m.debug != nil {
			if m.debug.OnLayer != nil {
				m.debug.OnLayer(layer, m.scratch.x[:m.cfg.EmbeddingDim])
			}

			if m.debug.OnLayerLogits != nil {
				if err := m.logitsFromHidden(m.scratch.x[:m.cfg.EmbeddingDim]); err != nil {
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
	return m.embedTokenInto(tokenID, m.scratch.x[:m.cfg.EmbeddingDim])
}

func (m *Model) forwardBlock(layer int, pos int) error {
	ln := m.layerNorms[layer]
	lt := m.layerTensors[layer]
	usedQKVResidency := false

	embd := m.cfg.EmbeddingDim
	qDim := m.cfg.NumHeads * m.cfg.HeadDim
	kvDim := m.cfg.NumKVHeads * m.cfg.HeadDim
	ffn := m.cfg.FFNHidden
	x := m.scratch.x[:embd]
	h := m.scratch.h[:embd]
	q := m.scratch.q[:qDim]
	k := m.scratch.k[:kvDim]
	v := m.scratch.v[:kvDim]
	attn := m.scratch.attn[:qDim]
	gate := m.scratch.gate[:ffn]
	up := m.scratch.up[:ffn]

	// entire layer on device - attn_norm, QKV, attention, WO, FFN, and both residuals.
	// host x is not read or updated.
	if m.residDevice {
		err := m.forwardBlockDevice(layer, pos, ln)
		if err == nil {
			return nil
		}

		// Hollow CPU mirror: cannot fall back to host attention; reset and fail.
		if m.cpuKVHollow {
			m.ResetCache()
			return fmt.Errorf("qwen3: layer %d residency failed without CPU KV mirror: %w", layer, err)
		}

		// Device KV append failure (-6): mark stale and try rebuild from CPU mirror.
		if strings.Contains(err.Error(), "code -6") || strings.Contains(err.Error(), "code -2") {
			m.gpuKVStale = true
			_ = m.tryRecoverGPUKV()
		}

		if rerr := m.recoverHiddenFromDevice(layer, err); rerr != nil {
			return rerr
		}
	}

	if err := m.rmsNormInto(h, x, ln.attnNorm, layer); err != nil {
		return err
	}

	// fused QKV reads attention from GPU KV-cache: when gpuKVStale it is incomplete
	if m.dev != nil && device.LayerOnGPU(layer, m.ngl, m.cfg.NumLayers) && qkvResidencyEnabled() && !m.gpuKVStale {
		if err := m.qkvAttnGPU(layer, pos, ln, h, k, v, attn); err == nil {
			if err := m.cache.Append(layer, k, v); err != nil {
				return err
			}
			usedQKVResidency = true
		}
	}

	if !usedQKVResidency {
		if err := m.matmulInto(lt.attnQ, qDim, embd, h, q, layer); err != nil {
			return err
		}

		if err := m.matmulInto(lt.attnK, kvDim, embd, h, k, layer); err != nil {
			return err
		}

		if err := m.matmulInto(lt.attnV, kvDim, embd, h, v, layer); err != nil {
			return err
		}

		if err := m.normHeadsInto(q, ln.qNorm, m.cfg.NumHeads, layer); err != nil {
			return err
		}

		if err := m.normHeadsInto(k, ln.kNorm, m.cfg.NumKVHeads, layer); err != nil {
			return err
		}

		m.applyRoPEHeads(q, m.cfg.NumHeads, pos, layer)
		m.applyRoPEHeads(k, m.cfg.NumKVHeads, pos, layer)

		kvPos := m.cache.Len()
		if err := m.cache.Append(layer, k, v); err != nil {
			return err
		}

		// token missed GPU KV-cache - disable GPU attention until ResetCache, else fused QKV/AttentionScoresKV use incomplete cache
		if m.dev != nil && device.LayerOnGPU(layer, m.ngl, m.cfg.NumLayers) && !m.gpuKVStale {
			if err := m.dev.KVCacheAppend(layer, kvPos, k, v); err != nil {
				m.gpuKVStale = true
				_ = m.tryRecoverGPUKV()
			}
		}

		seqLen := m.cache.Len() + 1

		if err := m.attentionScoresInto(attn, q, m.cache.KLayer(layer), m.cache.VLayer(layer), m.scratch.scores, seqLen, layer); err != nil {
			return err
		}
	}
	// WO+RMSNorm+FFN residency (LoadOptions.AttnFFNResidency).
	// MoE layers skip this: the fused path assumes dense FFN, replaced by experts
	if !lt.moe && m.dev != nil && device.LayerOnGPU(layer, m.ngl, m.cfg.NumLayers) && attnFFNResidencyEnabled() {
		if err := m.attnFFNGPU(layer, ln, x, attn); err == nil {
			return nil
		}
	}

	if err := m.matmulInto(lt.attnOut, embd, qDim, attn, h, layer); err != nil {
		return err
	}
	compute.AddInPlace(x, h)

	if err := m.rmsNormInto(h, x, ln.ffnNorm, layer); err != nil {
		return err
	}

	if lt.moe {
		if err := m.ffnMoE(lt, layer); err != nil {
			return err
		}

		compute.AddInPlace(x, m.scratch.moeAcc)
		return nil
	}

	if m.dev != nil && device.LayerOnGPU(layer, m.ngl, m.cfg.NumLayers) {
		if err := m.ffnGPU(layer, h, h); err == nil {
			compute.AddInPlace(x, h)
			return nil
		}
	}

	if err := m.matmulInto(lt.ffnGate, ffn, embd, h, gate, layer); err != nil {
		return err
	}

	if err := m.matmulInto(lt.ffnUp, ffn, embd, h, up, layer); err != nil {
		return err
	}
	m.swigluInPlace(gate, up, layer)

	if err := m.matmulInto(lt.ffnDown, embd, ffn, gate, h, layer); err != nil {
		return err
	}

	compute.AddInPlace(x, h)
	return nil
}

func (m *Model) ffnMoE(lt layerTensors, layer int) error {
	embd := m.cfg.EmbeddingDim
	nExp := m.cfg.ExpertCount
	ffn := m.cfg.expertFFN()
	h := m.scratch.h[:embd]

	if err := m.matmulInto(lt.gateInp, nExp, embd, h, m.scratch.router[:nExp], layer); err != nil {
		return err
	}

	idxs, weights := moe.TopKSoftmax(m.scratch.router[:nExp], m.cfg.ExpertUsedCount, m.cfg.MoENormWeights, m.cfg.ExpertWeightScale)

	clear(m.scratch.moeAcc)

	// selected experts on GPU; expert weights cached on device as raw tensor slice, no full expert matrix dequant on host
	if m.moeGPU && device.LayerOnGPU(layer, m.ngl, m.cfg.NumLayers) {
		if err := m.moeExpertsGPU(lt, idxs, weights, h, ffn); err == nil {
			return nil
		}

		// expert did not run on GPU (VRAM / weight type): CPU only from here
		m.moeGPU = false
		clear(m.scratch.moeAcc)
	}

	return moe.ExpertsSwiGLUInto(
		m.weights, lt.gateExps, lt.upExps, lt.downExps,
		idxs, weights, h,
		m.scratch.gate[:ffn], m.scratch.up[:ffn], m.scratch.tmp, m.scratch.moeAcc,
		embd, ffn,
	)
}

// moeExpertsGPU runs selected experts as fused-FFN on GPU and accumulates weighted sum
func (m *Model) moeExpertsGPU(lt layerTensors, idxs []int, weights []float32, h []float32, ffn int) error {
	embd := m.cfg.EmbeddingDim
	for i, ei := range idxs {
		if err := m.fused.ExpertFFN(lt.gateExps, lt.upExps, lt.downExps, ei, h, m.scratch.tmp[:embd], embd, ffn); err != nil {
			return err
		}

		w := weights[i]
		for j := range embd {
			m.scratch.moeAcc[j] += m.scratch.tmp[j] * w
		}
	}

	return nil
}

func (m *Model) matmulInto(name string, rows, cols int, vec, out []float32, layer int) error {
	if m.dev != nil && device.LayerOnGPU(layer, m.ngl, m.cfg.NumLayers) {
		return dispatch.MatMulGPUInto(m.dev, m.weights, name, rows, cols, vec, out)
	}

	return dispatch.MatMulRawInto(m.weights, name, rows, cols, vec, out)
}

func (m *Model) logits() error {
	// hidden on device - out_norm + lm_head on GPU too; only logits go to host
	if m.residDevice && m.logitsOnGPU {
		if err := m.logitsDevice(); err == nil {
			return nil
		}

		// lm_head on GPU failed (weight type / VRAM): continue on host
		m.logitsOnGPU = false
	}

	if err := m.syncHiddenFromDevice(); err != nil {
		return err
	}

	// hidden on host (MoE): lm_head on GPU after one upload
	if m.logitsOnGPU && m.cfg.ExpertCount > 0 && m.fused != nil {
		err := m.fused.LogitsFromHost(m.scratch.x, "output_norm.weight", m.outNorm, m.lmHeadName, m.scratch.logits, m.cfg.VocabSize, m.cfg.EmbeddingDim, m.cfg.RMSNormEps)
		if err == nil {
			return nil
		}

		m.logitsOnGPU = false
	}

	return m.logitsFromHidden(m.scratch.x[:m.cfg.EmbeddingDim])
}

func (m *Model) logitsFromHidden(x []float32) error {
	h := m.scratch.h[:len(x)]
	return dispatch.LogitsFromHiddenInto(m.weights, h, x, m.outNorm, m.scratch.logits, m.lmHeadName, m.cfg.VocabSize, m.cfg.EmbeddingDim, m.cfg.RMSNormEps)
}

func (m *Model) emitLogitsDebug() {
	if m.debug != nil && m.debug.OnLogits != nil {
		m.debug.OnLogits(m.scratch.logits)
	}
}

func (m *Model) logitsFinish() error {
	if err := m.logits(); err != nil {
		return err
	}
	m.emitLogitsDebug()

	return nil
}

func (m *Model) rmsNormInto(dst, x, weight []float32, layer int) error {
	// RMSNorm on GPU is not worthwhile over PCIe (small buffer): CPU only.
	_ = layer
	return compute.RMSNormInto(dst, x, weight, m.cfg.RMSNormEps)
}

func (m *Model) applyRoPEHeads(v []float32, nHeads, pos, layer int) {
	// RoPE on GPU: HtoD/DtoH cost exceeds decode benefit; CPU is faster for the residency path.
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
	if m.dev != nil && device.LayerOnGPU(layer, m.ngl, m.cfg.NumLayers) {
		// gpuKVStale: some tokens missed GPU KV-cache - attention uses CPU cache only
		if !m.gpuKVStale {
			if err := m.dev.AttentionScoresKV(layer, dst, q, seqLen, m.cfg.NumHeads, m.cfg.NumKVHeads, m.cfg.HeadDim); err == nil {
				return nil
			}
		}

		if err := m.dev.AttentionScoresInto(dst, q, k, v, scores, seqLen, m.cfg.NumHeads, m.cfg.NumKVHeads, m.cfg.HeadDim); err == nil {
			return nil
		}
	}

	return compute.AttentionScoresInto(dst, q, k, v, scores, seqLen, m.cfg.NumHeads, m.cfg.NumKVHeads, m.cfg.HeadDim)
}

func (m *Model) normHeadsInto(v []float32, weight []float32, nHeads, layer int) error {
	if len(weight) != m.cfg.HeadDim {
		return fmt.Errorf("qwen3: len(weight)=%d, head_dim=%d", len(weight), m.cfg.HeadDim)
	}

	for h := range nHeads {
		off := h * m.cfg.HeadDim
		if err := m.rmsNormInto(v[off:off+m.cfg.HeadDim], v[off:off+m.cfg.HeadDim], weight, layer); err != nil {
			return err
		}
	}

	return nil
}
