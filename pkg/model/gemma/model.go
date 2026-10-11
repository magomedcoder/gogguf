package gemma

import (
	"fmt"
	"github.com/magomedcoder/gogguf/pkg/dtype"

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
	scratch      scratch
	layerNorms   []layerNorms
	layerTensors []layerTensors
	outNorm      []float32
	lmHeadName   string

	fused     *gpuresid.Runner   // fused GPU layer paths (§5)
	fusedDims gpuresid.Dims      // dimensions for fused FFN
	gpuLayers []gpuresid.Tensors // layer weight names for fused paths
}

// LoadGemma creates Gemma 1
func LoadGemma(w *weights.Store, g device.Backend, ngl, gpuMaxSeq int) (*Model, error) {
	return load(w, g, ngl, gpuMaxSeq, ParseConfigGemma)
}

// LoadGemma2 creates Gemma 2
func LoadGemma2(w *weights.Store, g device.Backend, ngl, gpuMaxSeq int) (*Model, error) {
	return load(w, g, ngl, gpuMaxSeq, ParseConfigGemma2)
}

func load(w *weights.Store, g device.Backend, ngl, gpuMaxSeq int, parse func(*format.Reader) (Config, error)) (*Model, error) {
	cfg, err := parse(w.Reader())
	if err != nil {
		return nil, err
	}

	if ngl > cfg.NumLayers {
		ngl = cfg.NumLayers
	}

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
		scratch:      newScratch(cfg),
		layerNorms:   layerNorms,
		layerTensors: loadLayerTensors(cfg.NumLayers),
		outNorm:      outNorm,
		lmHeadName:   lmHeadName,
	}
	if err := m.initGPUKVCache(); err != nil {
		return nil, err
	}

	m.initFused()

	return m, nil
}

// initFused sets up fused GPU layer FFN (§5). Gemma: GeGLU instead of SwiGLU; residency not enabled yet - blocked by post-norm, softcap, and SWA
func (m *Model) initFused() {
	if m.dev == nil {
		return
	}

	m.fused = gpuresid.New(m.weights, m.dev, device.RoPENeoX)
	m.fusedDims = gpuresid.Dims{
		Embd:     m.cfg.EmbeddingDim,
		NHeads:   m.cfg.NumHeads,
		NKVHeads: m.cfg.NumKVHeads,
		HeadDim:  m.cfg.HeadDim,
		FFN:      m.cfg.FFNHidden,
		Eps:      m.cfg.RMSNormEps,
	}

	m.gpuLayers = make([]gpuresid.Tensors, m.cfg.NumLayers)
	for i, lt := range m.layerTensors {
		m.gpuLayers[i] = gpuresid.Tensors{
			FFNGate: lt.ffnGate,
			FFNUp:   lt.ffnUp,
			FFNDown: lt.ffnDown,
		}
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

// Forward runs the forward pass
func (m *Model) Forward(tokenIDs []int, startPos int) ([]float32, error) {
	if len(tokenIDs) == 0 {
		return nil, fmt.Errorf("gemma: empty input")
	}

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

	copy(m.scratch.out, m.scratch.logits)
	return m.scratch.out, nil
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

func (m *Model) forwardBlock(layer int, pos int) error {
	ln := m.layerNorms[layer]
	lt := m.layerTensors[layer]

	if err := compute.RMSNormInto(m.scratch.h, m.scratch.x, ln.attnNorm, m.cfg.RMSNormEps); err != nil {
		return err
	}

	qDim := m.cfg.NumHeads * m.cfg.HeadDim
	kvDim := m.cfg.NumKVHeads * m.cfg.HeadDim
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
	if m.dev != nil && device.LayerOnGPU(layer, m.ngl, m.cfg.NumLayers) {
		_ = m.dev.KVCacheAppend(layer, kvPos, m.scratch.k, m.scratch.v)
	}

	seqLen := m.cache.Len() + 1
	k, v, attnSeq := m.attentionKV(layer, seqLen)

	if err := m.attentionScoresInto(m.scratch.attn, m.scratch.q, k, v, m.scratch.scores, attnSeq, layer); err != nil {
		return err
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

// ffnInto: gate/up + GeGLU + down. On GPU - one fused call (§5), otherwise piecewise
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

	w := m.cfg.SlidingWindow
	if w <= 0 || seqLen <= w {
		return k, v, attnSeq
	}

	kvDim := m.cfg.NumKVHeads * m.cfg.HeadDim
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
	if softcap == 0 && !swaClip && m.dev != nil && device.LayerOnGPU(layer, m.ngl, m.cfg.NumLayers) {
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

func (m *Model) matmulGPU(name string, rows, cols int, vec []float32) ([]float32, error) {
	info, err := m.weights.Info(name)
	if err != nil {
		return nil, err
	}

	raw, err := m.weights.Raw(name)
	if err != nil && info.Type != dtype.F32 && info.Type != dtype.F16 {
		return nil, err
	}

	switch info.Type {
	case dtype.Q8_0:
		return m.dev.MatMulVecQ8_0Cached(name, raw, rows, cols, vec)
	case dtype.Q4_0:
		return m.dev.MatMulVecQ4_0Cached(name, raw, rows, cols, vec)
	case dtype.Q4_K:
		return m.dev.MatMulVecQ4_KCached(name, raw, rows, cols, vec)
	case dtype.Q5_K:
		return m.dev.MatMulVecQ5_KCached(name, raw, rows, cols, vec)
	case dtype.Q6_K:
		return m.dev.MatMulVecQ6_KCached(name, raw, rows, cols, vec)
	default:
		f32, err := m.weights.Floats(name)
		if err != nil {
			return nil, err
		}

		return m.dev.MatMulVecCached(name, f32, rows, cols, vec)
	}
}

func (m *Model) logitsFinish() error {
	if err := compute.RMSNormInto(m.scratch.h, m.scratch.x, m.outNorm, m.cfg.RMSNormEps); err != nil {
		return err
	}

	name := m.lmHeadName
	raw, err := m.weights.Raw(name)
	if err != nil {
		return err
	}

	info, err := m.weights.Info(name)
	if err != nil {
		return err
	}

	var matErr error
	switch info.Type {
	case dtype.Q8_0:
		matErr = compute.MatMulVecQ8_0Into(raw, m.cfg.VocabSize, m.cfg.EmbeddingDim, m.scratch.h, m.scratch.logits)
	case dtype.Q4_0:
		matErr = compute.MatMulVecQ4_0Into(raw, m.cfg.VocabSize, m.cfg.EmbeddingDim, m.scratch.h, m.scratch.logits)
	case dtype.Q4_1:
		matErr = compute.MatMulVecQ4_1Into(raw, m.cfg.VocabSize, m.cfg.EmbeddingDim, m.scratch.h, m.scratch.logits)
	case dtype.Q5_0:
		matErr = compute.MatMulVecQ5_0Into(raw, m.cfg.VocabSize, m.cfg.EmbeddingDim, m.scratch.h, m.scratch.logits)
	case dtype.Q5_1:
		matErr = compute.MatMulVecQ5_1Into(raw, m.cfg.VocabSize, m.cfg.EmbeddingDim, m.scratch.h, m.scratch.logits)
	case dtype.Q4_K:
		matErr = compute.MatMulVecQ4_KInto(raw, m.cfg.VocabSize, m.cfg.EmbeddingDim, m.scratch.h, m.scratch.logits)
	case dtype.Q5_K:
		matErr = compute.MatMulVecQ5_KInto(raw, m.cfg.VocabSize, m.cfg.EmbeddingDim, m.scratch.h, m.scratch.logits)
	case dtype.Q6_K:
		matErr = compute.MatMulVecQ6_KInto(raw, m.cfg.VocabSize, m.cfg.EmbeddingDim, m.scratch.h, m.scratch.logits)
	case dtype.Q2_K:
		matErr = compute.MatMulVecQ2_KInto(raw, m.cfg.VocabSize, m.cfg.EmbeddingDim, m.scratch.h, m.scratch.logits)
	case dtype.Q3_K:
		matErr = compute.MatMulVecQ3_KInto(raw, m.cfg.VocabSize, m.cfg.EmbeddingDim, m.scratch.h, m.scratch.logits)
	case dtype.Q8_K:
		matErr = compute.MatMulVecQ8_KInto(raw, m.cfg.VocabSize, m.cfg.EmbeddingDim, m.scratch.h, m.scratch.logits)
	default:
		f32, err := m.weights.Floats(name)
		if err != nil {
			return err
		}

		matErr = compute.MatMulVecInto(f32, m.cfg.VocabSize, m.cfg.EmbeddingDim, m.scratch.h, m.scratch.logits)
	}
	if matErr != nil {
		return matErr
	}

	compute.SoftcapInPlace(m.scratch.logits, m.cfg.FinalLogitSoftcap)
	return nil
}
