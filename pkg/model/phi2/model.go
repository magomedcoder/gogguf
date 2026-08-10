package phi2

import (
	"fmt"

	"github.com/magomedcoder/gogguf/pkg/format"
	"github.com/magomedcoder/gogguf/pkg/gpu"
	"github.com/magomedcoder/gogguf/pkg/ops"
	"github.com/magomedcoder/gogguf/pkg/weights"
)

// Model - Phi-2 (LayerNorm+bias, NeoX partial RoPE, parallel residual, GELU FFN)
type Model struct {
	cfg          Config
	weights      *weights.Store
	cache        *KVCache
	gpu          gpu.Backend
	ngl          int
	gpuMaxSeq    int
	scratch      scratch
	layerNorms   []layerNorms
	layerTensors []layerTensors
	layerBias    []layerBias
	outNorm      outNorms
	lmHeadName   string
	lmHeadBias   []float32
}

type layerBias struct {
	qkv     []float32 // fused or nil
	q, k, v []float32
	wo      []float32
	ffnUp   []float32
	ffnDn   []float32
}

// Load создаёт Phi-2 из весов GGUF
func Load(w *weights.Store, g gpu.Backend, ngl, gpuMaxSeq int) (*Model, error) {
	cfg, err := ParseConfig(w.Reader())
	if err != nil {
		return nil, err
	}

	if ngl > cfg.NumLayers {
		ngl = cfg.NumLayers
	}

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
		gpu:          g,
		ngl:          ngl,
		gpuMaxSeq:    gpuMaxSeq,
		scratch:      newScratch(cfg),
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
	if m.gpu == nil || m.ngl <= 0 {
		return nil
	}

	kvDim := m.cfg.NumKVHeads * m.cfg.HeadDim
	maxSeq := gpu.CapMaxSeq(m.cfg.ContextLength, m.gpuMaxSeq)

	return m.gpu.KVCacheInit(m.ngl, maxSeq, kvDim, m.cfg.NumHeads, m.cfg.HeadDim)
}

// Config возвращает конфигурацию модели
func (m *Model) Config() Config {
	return m.cfg
}

// ResetCache сбрасывает KV-cache
func (m *Model) ResetCache() {
	m.cache.Reset()
	if m.gpu != nil {
		m.gpu.KVCacheReset()
	}
}

// Close освобождает GPU-ресурсы модели
func (m *Model) Close() error {
	if m.gpu == nil {
		return nil
	}
	err := m.gpu.Close()
	m.gpu = nil
	return err
}

// Forward выполняет forward pass для последовательности tokenIDs начиная с startPos
func (m *Model) Forward(tokenIDs []int, startPos int) ([]float32, error) {
	if len(tokenIDs) == 0 {
		return nil, fmt.Errorf("phi2: пустой ввод")
	}

	for i, id := range tokenIDs {
		pos := startPos + i
		if err := m.embed(id); err != nil {
			return nil, err
		}

		for layer := range m.cfg.NumLayers {
			if err := m.forwardBlock(layer, pos); err != nil {
				return nil, fmt.Errorf("phi2: слой %d pos %d: %w", layer, pos, err)
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

func (m *Model) embed(tokenID int) error {
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

func (m *Model) forwardBlock(layer int, pos int) error {
	ln := m.layerNorms[layer]
	lt := m.layerTensors[layer]
	lb := m.layerBias[layer]

	// один LayerNorm на residual; attn и FFN параллельно
	if err := ops.LayerNormInto(m.scratch.h, m.scratch.x, ln.attnW, ln.attnB, m.cfg.LayerNormEps); err != nil {
		return err
	}

	if err := m.projectQKV(lt, lb, layer); err != nil {
		return err
	}

	ops.ApplyRoPEHeadsPartial(m.scratch.q, m.cfg.NumHeads, m.cfg.HeadDim, m.cfg.RopeDim, pos, m.cfg.RopeFreqBase)
	ops.ApplyRoPEHeadsPartial(m.scratch.k, m.cfg.NumKVHeads, m.cfg.HeadDim, m.cfg.RopeDim, pos, m.cfg.RopeFreqBase)

	kvPos := m.cache.Len()
	m.cache.Append(layer, m.scratch.k, m.scratch.v)
	if m.gpu != nil && gpu.LayerOnGPU(layer, m.ngl, m.cfg.NumLayers) {
		_ = m.gpu.KVCacheAppend(layer, kvPos, m.scratch.k, m.scratch.v)
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
	ops.AddBiasInPlace(m.scratch.attnOut, lb.wo)
	ops.AddInPlace(m.scratch.x, m.scratch.attnOut)

	// FFN от того же normed h (parallel residual)
	if err := m.matmulInto(lt.ffnUp, m.cfg.FFNHidden, m.cfg.EmbeddingDim, m.scratch.h, m.scratch.up, layer); err != nil {
		return err
	}
	ops.AddBiasInPlace(m.scratch.up, lb.ffnUp)
	ops.GELUInPlace(m.scratch.up)

	if err := m.matmulInto(lt.ffnDown, m.cfg.EmbeddingDim, m.cfg.FFNHidden, m.scratch.up, m.scratch.h, layer); err != nil {
		return err
	}
	ops.AddBiasInPlace(m.scratch.h, lb.ffnDn)
	ops.AddInPlace(m.scratch.x, m.scratch.h)

	return nil
}

func (m *Model) projectQKV(lt layerTensors, lb layerBias, layer int) error {
	qDim := m.cfg.NumHeads * m.cfg.HeadDim
	kvDim := m.cfg.NumKVHeads * m.cfg.HeadDim

	if lt.fusedQKV {
		if err := m.matmulInto(lt.attnQKV, qDim+2*kvDim, m.cfg.EmbeddingDim, m.scratch.h, m.scratch.qkv, layer); err != nil {
			return err
		}
		ops.AddBiasInPlace(m.scratch.qkv, lb.qkv)
		copy(m.scratch.q, m.scratch.qkv[:qDim])
		copy(m.scratch.k, m.scratch.qkv[qDim:qDim+kvDim])
		copy(m.scratch.v, m.scratch.qkv[qDim+kvDim:qDim+2*kvDim])
		return nil
	}

	if err := m.matmulInto(lt.attnQ, qDim, m.cfg.EmbeddingDim, m.scratch.h, m.scratch.q, layer); err != nil {
		return err
	}
	ops.AddBiasInPlace(m.scratch.q, lb.q)

	if err := m.matmulInto(lt.attnK, kvDim, m.cfg.EmbeddingDim, m.scratch.h, m.scratch.k, layer); err != nil {
		return err
	}
	ops.AddBiasInPlace(m.scratch.k, lb.k)

	if err := m.matmulInto(lt.attnV, kvDim, m.cfg.EmbeddingDim, m.scratch.h, m.scratch.v, layer); err != nil {
		return err
	}
	ops.AddBiasInPlace(m.scratch.v, lb.v)
	return nil
}

func (m *Model) attentionScoresInto(dst, q, k, v, scores []float32, seqLen, layer int) error {
	if m.gpu != nil && gpu.LayerOnGPU(layer, m.ngl, m.cfg.NumLayers) {
		if err := m.gpu.AttentionScoresKV(layer, dst, q, seqLen, m.cfg.NumHeads, m.cfg.NumKVHeads, m.cfg.HeadDim); err == nil {
			return nil
		}
	}

	return ops.AttentionScoresInto(dst, q, k, v, scores, seqLen, m.cfg.NumHeads, m.cfg.NumKVHeads, m.cfg.HeadDim)
}

func (m *Model) matmulInto(name string, rows, cols int, vec, out []float32, layer int) error {
	if m.gpu != nil && gpu.LayerOnGPU(layer, m.ngl, m.cfg.NumLayers) {
		got, err := m.matmulGPU(name, rows, cols, vec)
		if err == nil {
			copy(out, got)
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

func (m *Model) matmulGPU(name string, rows, cols int, vec []float32) ([]float32, error) {
	info, err := m.weights.Info(name)
	if err != nil {
		return nil, err
	}

	raw, err := m.weights.Raw(name)
	if err != nil && info.Type != format.GgmlFloat32 && info.Type != format.GgmlFloat16 {
		return nil, err
	}

	switch info.Type {
	case format.GgmlQ8_0:
		return m.gpu.MatMulVecQ8_0Cached(name, raw, rows, cols, vec)
	case format.GgmlQ4_0:
		return m.gpu.MatMulVecQ4_0Cached(name, raw, rows, cols, vec)
	case format.GgmlQ4_K:
		return m.gpu.MatMulVecQ4_KCached(name, raw, rows, cols, vec)
	case format.GgmlQ5_K:
		return m.gpu.MatMulVecQ5_KCached(name, raw, rows, cols, vec)
	case format.GgmlQ6_K:
		return m.gpu.MatMulVecQ6_KCached(name, raw, rows, cols, vec)
	default:
		f32, err := m.weights.Floats(name)
		if err != nil {
			return nil, err
		}
		return m.gpu.MatMulVecCached(name, f32, rows, cols, vec)
	}
}

func (m *Model) logitsFinish() error {
	if err := ops.LayerNormInto(m.scratch.h, m.scratch.x, m.outNorm.w, m.outNorm.b, m.cfg.LayerNormEps); err != nil {
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
	case format.GgmlQ8_0:
		matErr = ops.MatMulVecQ8_0Into(raw, m.cfg.VocabSize, m.cfg.EmbeddingDim, m.scratch.h, m.scratch.logits)
	case format.GgmlQ4_0:
		matErr = ops.MatMulVecQ4_0Into(raw, m.cfg.VocabSize, m.cfg.EmbeddingDim, m.scratch.h, m.scratch.logits)
	case format.GgmlQ4_1:
		matErr = ops.MatMulVecQ4_1Into(raw, m.cfg.VocabSize, m.cfg.EmbeddingDim, m.scratch.h, m.scratch.logits)
	case format.GgmlQ5_0:
		matErr = ops.MatMulVecQ5_0Into(raw, m.cfg.VocabSize, m.cfg.EmbeddingDim, m.scratch.h, m.scratch.logits)
	case format.GgmlQ5_1:
		matErr = ops.MatMulVecQ5_1Into(raw, m.cfg.VocabSize, m.cfg.EmbeddingDim, m.scratch.h, m.scratch.logits)
	case format.GgmlQ4_K:
		matErr = ops.MatMulVecQ4_KInto(raw, m.cfg.VocabSize, m.cfg.EmbeddingDim, m.scratch.h, m.scratch.logits)
	case format.GgmlQ5_K:
		matErr = ops.MatMulVecQ5_KInto(raw, m.cfg.VocabSize, m.cfg.EmbeddingDim, m.scratch.h, m.scratch.logits)
	case format.GgmlQ6_K:
		matErr = ops.MatMulVecQ6_KInto(raw, m.cfg.VocabSize, m.cfg.EmbeddingDim, m.scratch.h, m.scratch.logits)
	case format.GgmlQ2_K:
		matErr = ops.MatMulVecQ2_KInto(raw, m.cfg.VocabSize, m.cfg.EmbeddingDim, m.scratch.h, m.scratch.logits)
	case format.GgmlQ3_K:
		matErr = ops.MatMulVecQ3_KInto(raw, m.cfg.VocabSize, m.cfg.EmbeddingDim, m.scratch.h, m.scratch.logits)
	case format.GgmlQ8_K:
		matErr = ops.MatMulVecQ8_KInto(raw, m.cfg.VocabSize, m.cfg.EmbeddingDim, m.scratch.h, m.scratch.logits)
	default:
		f32, err := m.weights.Floats(name)
		if err != nil {
			return err
		}
		matErr = ops.MatMulVecInto(f32, m.cfg.VocabSize, m.cfg.EmbeddingDim, m.scratch.h, m.scratch.logits)
	}
	if matErr != nil {
		return matErr
	}

	ops.AddBiasInPlace(m.scratch.logits, m.lmHeadBias)
	return nil
}
