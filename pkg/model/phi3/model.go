package phi3

import (
	"fmt"

	"github.com/magomedcoder/gogguf/pkg/compute"
	"github.com/magomedcoder/gogguf/pkg/device"
	"github.com/magomedcoder/gogguf/pkg/exec/dispatch"
	"github.com/magomedcoder/gogguf/pkg/exec/gpuresid"
	"github.com/magomedcoder/gogguf/pkg/weights"
)

// Model - Phi-3 / Phi-3.5 (RMSNorm, NeoX partial RoPE, SwiGLU; fused QKV/FFN)
type Model struct {
	cfg              Config
	weights          *weights.Store
	cache            *KVCache
	dev              device.Backend
	ngl              int
	gpuMaxSeq        int
	nBatch           int // chunk prefill size (1 = per token)
	scratch          scratch
	layerNorms       []layerNorms
	layerTensors     []layerTensors
	outNorm          []float32
	lmHeadName       string
	ropeFactorsShort []float32
	ropeFactorsLong  []float32
	fused            *gpuresid.Runner // fused GPU layer FFN
	ffnGPU           []bool           // layer FFN runs fused on GPU (separate gate/up only)
	logitsOnGPU      bool             // out_norm + lm_head computed on GPU from the final hidden state
}

// Load creates Phi-3 from GGUF weights
func Load(w *weights.Store, g device.Backend, ngl, gpuMaxSeq, nBatch int) (*Model, error) {
	cfg, err := ParseConfig(w.Reader())
	if err != nil {
		return nil, err
	}

	if ngl > cfg.NumLayers {
		ngl = cfg.NumLayers
	}

	nBatch = min(max(nBatch, 1), 512)

	layerNorms, outNorm, err := loadNormWeights(w, cfg.NumLayers)
	if err != nil {
		return nil, err
	}

	layerTensors, err := loadLayerTensors(w, cfg.NumLayers)
	if err != nil {
		return nil, err
	}

	lmHeadName, err := resolveLMHeadName(w)
	if err != nil {
		return nil, err
	}

	ropeShort, ropeLong, err := loadRopeFactors(w)
	if err != nil {
		return nil, err
	}

	m := &Model{
		cfg:              cfg,
		weights:          w,
		cache:            NewKVCache(cfg),
		dev:              g,
		ngl:              ngl,
		gpuMaxSeq:        gpuMaxSeq,
		nBatch:           nBatch,
		scratch:          newScratch(cfg, nBatch),
		layerNorms:       layerNorms,
		layerTensors:     layerTensors,
		outNorm:          outNorm,
		lmHeadName:       lmHeadName,
		ropeFactorsShort: ropeShort,
		ropeFactorsLong:  ropeLong,
	}

	if err := m.initGPUKVCache(); err != nil {
		return nil, err
	}

	// fused FFN only for layers with separate ffn_gate/ffn_up: merged ffn_up [embd, 2*ffn] would need a separate kernel
	m.initFused()
	m.initResidency()

	return m, nil
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

	return err
}

// Forward runs forward pass for tokenIDs starting at startPos
// Returns logits for the last token [vocabSize]. The slice is the model's
// internal scratch buffer (no copy): the caller must not retain or modify it
// across the next Forward / Embed call
func (m *Model) Forward(tokenIDs []int, startPos int) ([]float32, error) {
	if len(tokenIDs) == 0 {
		return nil, fmt.Errorf("phi3: empty input")
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
					return nil, fmt.Errorf("phi3: layer %d pos %d: %w", layer, pos, err)
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
		return nil, fmt.Errorf("phi3: empty input")
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
	return dispatch.EmbedTokenInto(m.weights, m.scratch.x, m.cfg.EmbeddingDim, tokenID)
}

func (m *Model) forwardBlock(layer int, pos int) error {
	ln := m.layerNorms[layer]
	lt := m.layerTensors[layer]

	if err := compute.RMSNormInto(m.scratch.h, m.scratch.x, ln.attnNorm, m.cfg.RMSNormEps); err != nil {
		return err
	}

	if err := m.projectQKV(lt, layer); err != nil {
		return err
	}

	scale := m.ropeScale()
	compute.ApplyRoPEHeadsPartialScaled(m.scratch.q, m.cfg.NumHeads, m.cfg.HeadDim, m.cfg.RopeDim, pos, m.cfg.RopeFreqBase, scale)
	compute.ApplyRoPEHeadsPartialScaled(m.scratch.k, m.cfg.NumKVHeads, m.cfg.HeadDim, m.cfg.RopeDim, pos, m.cfg.RopeFreqBase, scale)

	kvPos := m.cache.Len()
	if err := m.cache.Append(layer, m.scratch.k, m.scratch.v); err != nil {
		return err
	}

	if m.dev != nil && device.LayerOnGPU(layer, m.ngl, m.cfg.NumLayers) {
		_ = m.dev.KVCacheAppend(layer, kvPos, m.scratch.k, m.scratch.v)
	}

	seqLen := m.cache.Len() + 1
	k := m.cache.KLayer(layer)
	v := m.cache.VLayer(layer)

	if err := m.attentionScoresInto(m.scratch.attn, m.scratch.q, k, v, m.scratch.scores, seqLen, layer); err != nil {
		return err
	}

	if err := m.matmulInto(lt.attnOut, m.cfg.EmbeddingDim, m.cfg.NumHeads*m.cfg.HeadDim, m.scratch.attn, m.scratch.h, layer); err != nil {
		return err
	}
	compute.AddInPlace(m.scratch.x, m.scratch.h)

	if err := compute.RMSNormInto(m.scratch.h, m.scratch.x, ln.ffnNorm, m.cfg.RMSNormEps); err != nil {
		return err
	}

	if err := m.ffn(lt, layer); err != nil {
		return err
	}
	compute.AddInPlace(m.scratch.x, m.scratch.h)

	return nil
}

// ropeScale picks short/long factors like llama.cpp get_rope_factors
func (m *Model) ropeScale() compute.RoPEScale {
	s := compute.RoPEScale{
		FreqScale:  m.cfg.RopeFreqScale,
		AttnFactor: m.cfg.RopeAttnFactor,
	}
	orig := m.cfg.OrigCtxLen
	if orig <= 0 {
		orig = m.cfg.ContextLength
	}

	if m.cfg.ContextLength > orig && len(m.ropeFactorsLong) > 0 {
		s.Factors = m.ropeFactorsLong
		return s
	}

	if len(m.ropeFactorsShort) > 0 {
		s.Factors = m.ropeFactorsShort
		return s
	}

	if len(m.ropeFactorsLong) > 0 {
		s.Factors = m.ropeFactorsLong
	}

	return s
}

func (m *Model) projectQKV(lt layerTensors, layer int) error {
	qDim := m.cfg.NumHeads * m.cfg.HeadDim
	kvDim := m.cfg.NumKVHeads * m.cfg.HeadDim

	if lt.fusedQKV {
		if err := m.matmulInto(lt.attnQKV, qDim+2*kvDim, m.cfg.EmbeddingDim, m.scratch.h, m.scratch.qkv, layer); err != nil {
			return err
		}
		copy(m.scratch.q, m.scratch.qkv[:qDim])
		copy(m.scratch.k, m.scratch.qkv[qDim:qDim+kvDim])
		copy(m.scratch.v, m.scratch.qkv[qDim+kvDim:qDim+2*kvDim])

		return nil
	}

	if err := m.matmulInto(lt.attnQ, qDim, m.cfg.EmbeddingDim, m.scratch.h, m.scratch.q, layer); err != nil {
		return err
	}

	if err := m.matmulInto(lt.attnK, kvDim, m.cfg.EmbeddingDim, m.scratch.h, m.scratch.k, layer); err != nil {
		return err
	}

	return m.matmulInto(lt.attnV, kvDim, m.cfg.EmbeddingDim, m.scratch.h, m.scratch.v, layer)
}

func (m *Model) ffn(lt layerTensors, layer int) error {
	ffn := m.cfg.FFNHidden
	if lt.fusedFFN {
		// ffn_up: [embd, 2*ffn] -> gate||up; SwiGLU; ffn_down
		if err := m.matmulInto(lt.ffnUp, 2*ffn, m.cfg.EmbeddingDim, m.scratch.h, m.scratch.gateUp, layer); err != nil {
			return err
		}
		copy(m.scratch.gate, m.scratch.gateUp[:ffn])
		copy(m.scratch.up, m.scratch.gateUp[ffn:])
		compute.SwiGLUInPlace(m.scratch.gate, m.scratch.up)

		return m.matmulInto(lt.ffnDown, m.cfg.EmbeddingDim, ffn, m.scratch.gate, m.scratch.h, layer)
	}

	// separate gate/up go through one fused SwiGLU call on GPU
	if m.fused != nil && layer < len(m.ffnGPU) && m.ffnGPU[layer] {
		if err := m.fused.FFNNamed(lt.ffnGate, lt.ffnUp, lt.ffnDown, m.scratch.h, m.scratch.h, m.cfg.EmbeddingDim, ffn); err == nil {
			return nil
		}
	}

	if err := m.matmulInto(lt.ffnGate, ffn, m.cfg.EmbeddingDim, m.scratch.h, m.scratch.gate, layer); err != nil {
		return err
	}

	if err := m.matmulInto(lt.ffnUp, ffn, m.cfg.EmbeddingDim, m.scratch.h, m.scratch.up, layer); err != nil {
		return err
	}
	compute.SwiGLUInPlace(m.scratch.gate, m.scratch.up)

	return m.matmulInto(lt.ffnDown, m.cfg.EmbeddingDim, ffn, m.scratch.gate, m.scratch.h, layer)
}

func (m *Model) attentionScoresInto(dst, q, k, v, scores []float32, seqLen, layer int) error {
	if m.dev != nil && device.LayerOnGPU(layer, m.ngl, m.cfg.NumLayers) {
		if err := m.dev.AttentionScoresKV(layer, dst, q, seqLen, m.cfg.NumHeads, m.cfg.NumKVHeads, m.cfg.HeadDim); err == nil {
			return nil
		}
	}

	return compute.AttentionScoresInto(dst, q, k, v, scores, seqLen, m.cfg.NumHeads, m.cfg.NumKVHeads, m.cfg.HeadDim)
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

// logitsFromHidden: out_norm + lm_head for hidden state x [embd]
func (m *Model) logitsFromHidden(x []float32) error {
	return dispatch.LogitsFromHiddenInto(m.weights, m.scratch.h, x, m.outNorm, m.scratch.logits, m.lmHeadName, m.cfg.VocabSize, m.cfg.EmbeddingDim, m.cfg.RMSNormEps)
}
