package gemma

import (
	"fmt"

	"github.com/magomedcoder/gogguf/pkg/compute"
	"github.com/magomedcoder/gogguf/pkg/device"
	"github.com/magomedcoder/gogguf/pkg/exec/dispatch"
	"github.com/magomedcoder/gogguf/pkg/exec/gpuresid"
	"github.com/magomedcoder/gogguf/pkg/format"
	"github.com/magomedcoder/gogguf/pkg/weights"
)

// Model - Gemma / Gemma2 (RMSNorm, NeoX RoPE, GeGLU; gemma2: post-norm, softcap, SWA)
type Model struct {
	cfg          Config
	weights      *weights.Store
	cache        *KVCache
	dev          device.Backend
	ngl          int
	gpuMaxSeq    int
	nBatch       int // chunk prefill size (1 = per token)
	scratch      scratch
	layerNorms   []layerNorms
	layerTensors []layerTensors
	outNorm      []float32
	lmHeadName   string

	fused      *gpuresid.Runner   // fused GPU layer paths
	fusedDims  gpuresid.Dims      // dimensions for fused paths
	gpuLayers  []gpuresid.Tensors // layer weight names for fused paths
	layerQuant []bool             // layer fully in supported fused quant

	residency   bool // fused QKV+attention decode (Gemma 1 only; hidden stays on host, see initResidency)
	logitsOnGPU bool // out_norm + lm_head computed on GPU from the final hidden state
	gpuKVStale  bool // GPU KV-cache incomplete: attention on CPU only
}

// LoadGemma creates Gemma 1
func LoadGemma(w *weights.Store, g device.Backend, ngl, gpuMaxSeq, nBatch int) (*Model, error) {
	return load(w, g, ngl, gpuMaxSeq, nBatch, ParseConfigGemma)
}

// LoadGemma2 creates Gemma 2
func LoadGemma2(w *weights.Store, g device.Backend, ngl, gpuMaxSeq, nBatch int) (*Model, error) {
	return load(w, g, ngl, gpuMaxSeq, nBatch, ParseConfigGemma2)
}

func load(w *weights.Store, g device.Backend, ngl, gpuMaxSeq, nBatch int, parse func(*format.Reader) (Config, error)) (*Model, error) {
	cfg, err := parse(w.Reader())
	if err != nil {
		return nil, err
	}

	if ngl > cfg.NumLayers {
		ngl = cfg.NumLayers
	}

	nBatch = min(max(nBatch, 1), 512)

	layerNorms, outNorm, err := loadNormWeights(w, cfg)
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
		layerTensors: loadLayerTensors(cfg.NumLayers),
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

// prepareSWAAppend reclaims oldest KV on sliding-window layers so Append fits the window.
func (m *Model) prepareSWAAppend(layer, n int) {
	if !m.cfg.IsSWALayer(layer) || m.cfg.SlidingWindow <= 0 || n < 1 {
		return
	}

	w := m.cfg.SlidingWindow
	have := m.cache.LayerLen(layer)
	if have+n <= w {
		return
	}

	m.cache.DropOldestLayer(layer, have+n-w)
	if m.dev != nil {
		m.gpuKVStale = true
	}
}

func (m *Model) initGPUKVCache() error {
	if m.dev == nil || m.ngl <= 0 {
		return nil
	}

	kvDim := m.cfg.NumKVHeads * m.cfg.HeadDim
	maxSeq := device.CapMaxSeq(m.cfg.ContextLength, m.gpuMaxSeq)
	return m.dev.KVCacheInit(m.ngl, maxSeq, kvDim, m.cfg.NumHeads, m.cfg.HeadDim)
}

// Config returns the configuration
func (m *Model) Config() Config { return m.cfg }

// ResetCache clears the KV-cache
func (m *Model) ResetCache() {
	m.cache.Reset()
	m.gpuKVStale = false
	if m.dev != nil {
		m.dev.KVCacheReset()
	}
}

// Close releases GPU resources
func (m *Model) Close() error {
	if m.dev == nil {
		return nil
	}

	err := m.dev.Close()
	m.dev = nil
	return err
}

// Forward runs forward pass for tokenIDs starting at startPos Returns logits for the last token [vocabSize].
// The slice is the model's internal scratch buffer (no copy): the caller must not retain or modify it across the next Forward / Embed call
func (m *Model) Forward(tokenIDs []int, startPos int) ([]float32, error) {
	if len(tokenIDs) == 0 {
		return nil, fmt.Errorf("gemma: empty input")
	}

	// n_batch>1: multi-token prefill (chunk matmul on GPU with -ngl, else CPU GEMM)
	if m.nBatch > 1 && len(tokenIDs) > 1 {
		for i := 0; i < len(tokenIDs); {
			end := min(i+m.nBatch, len(tokenIDs))

			needLogits := end == len(tokenIDs)
			if err := m.forwardBatch(tokenIDs[i:end], startPos+i, needLogits); err != nil {
				return nil, err
			}
			i = end
		}
	} else {
		for i, id := range tokenIDs {
			pos := startPos + i
			if err := m.embed(id); err != nil {
				return nil, err
			}

			for layer := range m.cfg.NumLayers {
				if err := m.forwardBlock(layer, pos); err != nil {
					return nil, fmt.Errorf("gemma: layer %d pos %d: %w", layer, pos, err)
				}
			}
			m.cache.Advance()
		}

		if err := m.logitsFinish(); err != nil {
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
		return nil, fmt.Errorf("gemma: empty input")
	}

	m.ResetCache()
	defer m.ResetCache()

	for i, id := range tokenIDs {
		if err := m.embed(id); err != nil {
			return nil, err
		}

		for layer := range m.cfg.NumLayers {
			if err := m.forwardBlock(layer, i); err != nil {
				return nil, err
			}
		}

		m.cache.Advance()
	}
	if err := compute.RMSNormInto(m.scratch.h, m.scratch.x, m.outNorm, m.cfg.RMSNormEps); err != nil {
		return nil, err
	}

	out := make([]float32, m.cfg.EmbeddingDim)
	copy(out, m.scratch.h)
	return out, nil
}

func (m *Model) embed(tokenID int) error {
	if err := m.embedRaw(tokenID); err != nil {
		return err
	}

	compute.ScaleInPlace(m.scratch.x, m.cfg.EmbedScale)
	return nil
}

func (m *Model) embedRaw(tokenID int) error {
	return dispatch.EmbedTokenInto(m.weights, m.scratch.x, m.cfg.EmbeddingDim, tokenID)
}

func (m *Model) forwardBlock(layer int, pos int) error {
	ln := m.layerNorms[layer]
	lt := m.layerTensors[layer]

	if err := compute.RMSNormInto(m.scratch.h, m.scratch.x, ln.attnNorm, m.cfg.RMSNormEps); err != nil {
		return err
	}

	qDim := m.cfg.NumHeads * m.cfg.HeadDim
	kvDim := m.cfg.NumKVHeads * m.cfg.HeadDim
	onGPU := m.dev != nil && device.LayerOnGPU(layer, m.ngl, m.cfg.NumLayers)
	usedQKVResidency := false

	m.prepareSWAAppend(layer, 1)

	// Gemma 1 only (m.residency): fused QKV+RoPE+attention reads the GPU KV-cache, so it must be complete
	if m.residency && onGPU && !m.gpuKVStale {
		if err := m.qkvAttnGPU(layer, pos, ln, m.scratch.h, m.scratch.k, m.scratch.v, m.scratch.attn); err == nil {
			if err := m.cache.Append(layer, m.scratch.k, m.scratch.v); err != nil {
				return err
			}
			usedQKVResidency = true
		}
	}

	if !usedQKVResidency {
		if err := m.matmulInto(lt.attnQ, qDim, m.cfg.EmbeddingDim, m.scratch.h, m.scratch.q, layer); err != nil {
			return err
		}

		if err := m.matmulInto(lt.attnK, kvDim, m.cfg.EmbeddingDim, m.scratch.h, m.scratch.k, layer); err != nil {
			return err
		}

		if err := m.matmulInto(lt.attnV, kvDim, m.cfg.EmbeddingDim, m.scratch.h, m.scratch.v, layer); err != nil {
			return err
		}

		base := m.cfg.ropeBase(layer)
		compute.ApplyRoPEHeads(m.scratch.q, m.cfg.NumHeads, m.cfg.HeadDim, pos, base)
		compute.ApplyRoPEHeads(m.scratch.k, m.cfg.NumKVHeads, m.cfg.HeadDim, pos, base)

		kvPos := m.cache.Len()
		if err := m.cache.Append(layer, m.scratch.k, m.scratch.v); err != nil {
			return err
		}

		// token missed GPU KV-cache - disable GPU attention until ResetCache
		if onGPU && !m.gpuKVStale {
			if err := m.dev.KVCacheAppend(layer, kvPos, m.scratch.k, m.scratch.v); err != nil {
				m.gpuKVStale = true
			}
		}

		seqLen := m.cache.Len() + 1
		k, v, attnSeq := m.attentionKV(layer, seqLen)

		if err := m.attentionScoresInto(m.scratch.attn, m.scratch.q, k, v, m.scratch.scores, attnSeq, layer); err != nil {
			return err
		}
	}

	if err := m.matmulInto(lt.attnOut, m.cfg.EmbeddingDim, qDim, m.scratch.attn, m.scratch.h, layer); err != nil {
		return err
	}

	if m.cfg.Variant == VariantGemma2 {
		if err := compute.RMSNormInto(m.scratch.h, m.scratch.h, ln.attnPostNorm, m.cfg.RMSNormEps); err != nil {
			return err
		}
	}
	compute.AddInPlace(m.scratch.x, m.scratch.h)

	if err := compute.RMSNormInto(m.scratch.h, m.scratch.x, ln.ffnNorm, m.cfg.RMSNormEps); err != nil {
		return err
	}

	if err := m.ffnInto(layer, lt, m.scratch.h, m.scratch.h); err != nil {
		return err
	}

	if m.cfg.Variant == VariantGemma2 {
		if err := compute.RMSNormInto(m.scratch.h, m.scratch.h, ln.ffnPostNorm, m.cfg.RMSNormEps); err != nil {
			return err
		}
	}
	compute.AddInPlace(m.scratch.x, m.scratch.h)
	return nil
}

// ffnInto: gate/up + GeGLU + down. On GPU - one fused call, otherwise piecewise
func (m *Model) ffnInto(layer int, lt layerTensors, x, out []float32) error {
	if m.fused != nil && device.LayerOnGPU(layer, m.ngl, m.cfg.NumLayers) {
		if err := m.fused.FFNGeGLU(m.gpuLayers[layer], m.fusedDims, x, out); err == nil {
			return nil
		}
	}

	if err := m.matmulInto(lt.ffnGate, m.cfg.FFNHidden, m.cfg.EmbeddingDim, x, m.scratch.gate, layer); err != nil {
		return err
	}

	if err := m.matmulInto(lt.ffnUp, m.cfg.FFNHidden, m.cfg.EmbeddingDim, x, m.scratch.up, layer); err != nil {
		return err
	}
	compute.GeGLUInPlace(m.scratch.gate, m.scratch.up)

	return m.matmulInto(lt.ffnDown, m.cfg.EmbeddingDim, m.cfg.FFNHidden, m.scratch.gate, out, layer)
}

func (m *Model) attentionKV(layer, seqLen int) (k, v []float32, attnSeq int) {
	k = m.cache.KLayer(layer)
	v = m.cache.VLayer(layer)
	attnSeq = seqLen
	if !m.cfg.IsSWALayer(layer) {
		return k, v, attnSeq
	}

	kvDim := m.cfg.NumKVHeads * m.cfg.HeadDim
	stored := 0
	if kvDim > 0 {
		stored = len(k) / kvDim
	}

	// After DropOldestLayer reclaim, stored tokens are already the window.
	if stored > 0 && stored <= m.cfg.SlidingWindow {
		return k, v, stored
	}

	w := m.cfg.SlidingWindow
	if w <= 0 || seqLen <= w {
		return k, v, attnSeq
	}

	skip := (seqLen - 1 - w) * kvDim
	if skip <= 0 || skip >= len(k) {
		return k, v, attnSeq
	}

	k, v = k[skip:], v[skip:]
	return k, v, len(k) / kvDim
}

func (m *Model) attentionScoresInto(dst, q, k, v, scores []float32, seqLen, layer int) error {
	softcap := m.cfg.AttnLogitSoftcap
	swaClip := m.cfg.IsSWALayer(layer) && m.cfg.SlidingWindow > 0
	if softcap == 0 && !swaClip && !m.gpuKVStale && m.dev != nil && device.LayerOnGPU(layer, m.ngl, m.cfg.NumLayers) {
		if err := m.dev.AttentionScoresKV(layer, dst, q, seqLen, m.cfg.NumHeads, m.cfg.NumKVHeads, m.cfg.HeadDim); err == nil {
			return nil
		}
	}

	return compute.AttentionScoresIntoSoftcap(dst, q, k, v, scores, seqLen, m.cfg.NumHeads, m.cfg.NumKVHeads, m.cfg.HeadDim, softcap)
}

func (m *Model) matmulInto(name string, rows, cols int, vec, out []float32, layer int) error {
	if m.dev != nil && device.LayerOnGPU(layer, m.ngl, m.cfg.NumLayers) {
		return dispatch.MatMulGPUInto(m.dev, m.weights, name, rows, cols, vec, out)
	}

	return dispatch.MatMulRawInto(m.weights, name, rows, cols, vec, out)
}

func (m *Model) logitsFinish() error {
	return m.logitsFromLast(m.scratch.x)
}

// logitsFromLast: lm_head on GPU (hidden uploaded once) when enabled, else on host
func (m *Model) logitsFromLast(x []float32) error {
	if m.logitsOnGPU && m.fused != nil {
		if err := m.logitsDevice(x); err == nil {
			return nil
		}

		// lm_head on GPU failed (weight type / VRAM): continue on host
		m.recoverLogits()
	}

	return m.logitsFromHidden(x)
}

// logitsFromHidden: out_norm + lm_head for hidden state x [embd]
func (m *Model) logitsFromHidden(x []float32) error {
	if err := dispatch.LogitsFromHiddenInto(m.weights, m.scratch.h, x, m.outNorm, m.scratch.logits, m.lmHeadName, m.cfg.VocabSize, m.cfg.EmbeddingDim, m.cfg.RMSNormEps); err != nil {
		return err
	}

	compute.SoftcapInPlace(m.scratch.logits, m.cfg.FinalLogitSoftcap)
	return nil
}
