package phi2

import (
	"fmt"

	"github.com/magomedcoder/gogguf/pkg/compute"
	"github.com/magomedcoder/gogguf/pkg/device"
	"github.com/magomedcoder/gogguf/pkg/exec/dispatch"
	"github.com/magomedcoder/gogguf/pkg/exec/gpuresid"
	"github.com/magomedcoder/gogguf/pkg/weights"
)

// Model - Phi-2 (LayerNorm+bias, NeoX partial RoPE, parallel residual, GELU FFN)
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
	layerBias    []layerBias
	outNorm      outNorms
	lmHeadName   string
	lmHeadBias   []float32

	fused       *gpuresid.Runner // lm_head on GPU (see initResidency)
	logitsOnGPU bool             // lm_head matmul runs on GPU after host LayerNorm
}

type layerBias struct {
	qkv     []float32 // fused or nil
	q, k, v []float32
	wo      []float32
	ffnUp   []float32
	ffnDn   []float32
}

// Load creates Phi-2 from GGUF weights
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

	layerBias, err := loadLayerBiases(w, layerTensors)
	if err != nil {
		return nil, err
	}

	lmHeadName, err := resolveLMHeadName(w)
	if err != nil {
		return nil, err
	}

	lmHeadBias, err := optionalFloats(w, "output.bias")
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
		layerBias:    layerBias,
		outNorm:      outNorm,
		lmHeadName:   lmHeadName,
		lmHeadBias:   lmHeadBias,
	}

	if err := m.initGPUKVCache(); err != nil {
		return nil, err
	}

	m.initResidency()

	return m, nil
}

func loadLayerBiases(w *weights.Store, layers []layerTensors) ([]layerBias, error) {
	out := make([]layerBias, len(layers))
	for i, lt := range layers {
		var err error
		b := layerBias{}
		if lt.fusedQKV {
			if b.qkv, err = optionalFloats(w, lt.attnQKVb); err != nil {
				return nil, err
			}
		} else {
			if b.q, err = optionalFloats(w, lt.attnQb); err != nil {
				return nil, err
			}

			if b.k, err = optionalFloats(w, lt.attnKb); err != nil {
				return nil, err
			}

			if b.v, err = optionalFloats(w, lt.attnVb); err != nil {
				return nil, err
			}
		}
		if b.wo, err = w.Floats(lt.attnOutB); err != nil {
			return nil, fmt.Errorf("phi2: %s: %w", lt.attnOutB, err)
		}

		if b.ffnUp, err = w.Floats(lt.ffnUpB); err != nil {
			return nil, fmt.Errorf("phi2: %s: %w", lt.ffnUpB, err)
		}

		if b.ffnDn, err = w.Floats(lt.ffnDownB); err != nil {
			return nil, fmt.Errorf("phi2: %s: %w", lt.ffnDownB, err)
		}

		out[i] = b
	}

	return out, nil
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
		return nil, fmt.Errorf("phi2: empty input")
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
					return nil, fmt.Errorf("phi2: layer %d pos %d: %w", layer, pos, err)
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

// Embed - last-token LayerNorm(hidden) before lm_head (clears KV)
func (m *Model) Embed(tokenIDs []int) ([]float32, error) {
	if len(tokenIDs) == 0 {
		return nil, fmt.Errorf("phi2: empty input")
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

	if err := compute.LayerNormInto(m.scratch.h, m.scratch.x, m.outNorm.w, m.outNorm.b, m.cfg.LayerNormEps); err != nil {
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
	lb := m.layerBias[layer]

	// one LayerNorm on residual; attn and FFN in parallel
	if err := compute.LayerNormInto(m.scratch.h, m.scratch.x, ln.attnW, ln.attnB, m.cfg.LayerNormEps); err != nil {
		return err
	}

	if err := m.projectQKV(lt, lb, layer); err != nil {
		return err
	}

	compute.ApplyRoPEHeadsPartial(m.scratch.q, m.cfg.NumHeads, m.cfg.HeadDim, m.cfg.RopeDim, pos, m.cfg.RopeFreqBase)
	compute.ApplyRoPEHeadsPartial(m.scratch.k, m.cfg.NumKVHeads, m.cfg.HeadDim, m.cfg.RopeDim, pos, m.cfg.RopeFreqBase)

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

	if err := m.matmulInto(lt.attnOut, m.cfg.EmbeddingDim, m.cfg.NumHeads*m.cfg.HeadDim, m.scratch.attn, m.scratch.attnOut, layer); err != nil {
		return err
	}
	compute.AddBiasInPlace(m.scratch.attnOut, lb.wo)
	compute.AddInPlace(m.scratch.x, m.scratch.attnOut)

	// FFN from the same normed h (parallel residual)
	if err := m.matmulInto(lt.ffnUp, m.cfg.FFNHidden, m.cfg.EmbeddingDim, m.scratch.h, m.scratch.up, layer); err != nil {
		return err
	}
	compute.AddBiasInPlace(m.scratch.up, lb.ffnUp)
	compute.GELUInPlace(m.scratch.up)

	if err := m.matmulInto(lt.ffnDown, m.cfg.EmbeddingDim, m.cfg.FFNHidden, m.scratch.up, m.scratch.h, layer); err != nil {
		return err
	}
	compute.AddBiasInPlace(m.scratch.h, lb.ffnDn)
	compute.AddInPlace(m.scratch.x, m.scratch.h)

	return nil
}

func (m *Model) projectQKV(lt layerTensors, lb layerBias, layer int) error {
	qDim := m.cfg.NumHeads * m.cfg.HeadDim
	kvDim := m.cfg.NumKVHeads * m.cfg.HeadDim

	if lt.fusedQKV {
		if err := m.matmulInto(lt.attnQKV, qDim+2*kvDim, m.cfg.EmbeddingDim, m.scratch.h, m.scratch.qkv, layer); err != nil {
			return err
		}
		compute.AddBiasInPlace(m.scratch.qkv, lb.qkv)
		copy(m.scratch.q, m.scratch.qkv[:qDim])
		copy(m.scratch.k, m.scratch.qkv[qDim:qDim+kvDim])
		copy(m.scratch.v, m.scratch.qkv[qDim+kvDim:qDim+2*kvDim])
		return nil
	}

	if err := m.matmulInto(lt.attnQ, qDim, m.cfg.EmbeddingDim, m.scratch.h, m.scratch.q, layer); err != nil {
		return err
	}
	compute.AddBiasInPlace(m.scratch.q, lb.q)

	if err := m.matmulInto(lt.attnK, kvDim, m.cfg.EmbeddingDim, m.scratch.h, m.scratch.k, layer); err != nil {
		return err
	}
	compute.AddBiasInPlace(m.scratch.k, lb.k)

	if err := m.matmulInto(lt.attnV, kvDim, m.cfg.EmbeddingDim, m.scratch.h, m.scratch.v, layer); err != nil {
		return err
	}
	compute.AddBiasInPlace(m.scratch.v, lb.v)
	return nil
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
	return m.logitsFromHidden(m.scratch.x)
}

// logitsFromHidden: out_norm + lm_head for hidden state x [embd]
func (m *Model) logitsFromHidden(x []float32) error {
	if err := compute.LayerNormInto(m.scratch.h, x, m.outNorm.w, m.outNorm.b, m.cfg.LayerNormEps); err != nil {
		return err
	}

	if m.lmHeadGPU(m.scratch.h) {
		compute.AddBiasInPlace(m.scratch.logits, m.lmHeadBias)
		return nil
	}

	if err := dispatch.MatMulRawInto(m.weights, m.lmHeadName, m.cfg.VocabSize, m.cfg.EmbeddingDim, m.scratch.h, m.scratch.logits); err != nil {
		return err
	}

	compute.AddBiasInPlace(m.scratch.logits, m.lmHeadBias)
	return nil
}
