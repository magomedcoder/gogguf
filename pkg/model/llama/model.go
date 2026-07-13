package llama

import (
	"fmt"

	"github.com/magomedcoder/gogguf/pkg/format"
	"github.com/magomedcoder/gogguf/pkg/gpu"
	"github.com/magomedcoder/gogguf/pkg/ops"
	"github.com/magomedcoder/gogguf/pkg/weights"
)

// Model - Llama transformer (Llama 2 / 3)
type Model struct {
	cfg          Config
	weights      *weights.Store
	cache        *KVCache
	gpu          gpu.Backend
	ngl          int
	scratch      scratch
	layerNorms   []layerNorms
	layerTensors []layerTensors
	outNorm      []float32
	lmHeadName   string
	debug        *DebugHooks
}

// Load создаёт Llama из весов
func Load(w *weights.Store, g gpu.Backend, ngl int) (*Model, error) {
	cfg, err := ParseConfig(w.Reader())
	if err != nil {
		return nil, err
	}

	if ngl > cfg.NumLayers {
		return nil, fmt.Errorf("llama: ngl=%d больше числа слоёв %d", ngl, cfg.NumLayers)
	}

	layerNorms, outNorm, err := loadNormWeights(w, cfg.NumLayers)
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
		gpu:          g,
		ngl:          ngl,
		scratch:      newScratch(cfg),
		layerNorms:   layerNorms,
		layerTensors: loadLayerTensors(cfg.NumLayers),
		outNorm:      outNorm,
		lmHeadName:   lmHeadName,
	}

	if err := m.initGPUKVCache(); err != nil {
		return nil, err
	}

	return m, nil
}

func (m *Model) initGPUKVCache() error {
	if m.gpu == nil || m.ngl <= 0 {
		return nil
	}

	kvDim := m.cfg.NumKVHeads * m.cfg.HeadDim
	return m.gpu.KVCacheInit(m.ngl, m.cfg.ContextLength, kvDim, m.cfg.NumHeads, m.cfg.HeadDim)
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

// SetDebugHooks включает колбэки для пошаговой отладки forward pass
func (m *Model) SetDebugHooks(h *DebugHooks) {
	m.debug = h
}

// Forward выполняет forward pass для последовательности tokenIDs начиная с startPos
func (m *Model) Forward(tokenIDs []int, startPos int) ([]float32, error) {
	if len(tokenIDs) == 0 {
		return nil, fmt.Errorf("llama: пустой ввод")
	}

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

	if m.debug != nil && m.debug.OnLogits != nil {
		m.debug.OnLogits(m.scratch.logits)
	}

	out := make([]float32, m.cfg.VocabSize)
	copy(out, m.scratch.logits)
	return out, nil
}

func (m *Model) forwardToken(tokenID, pos int, debug bool) error {
	if err := m.embedToken(tokenID); err != nil {
		return err
	}

	if debug && m.debug != nil && m.debug.OnEmbed != nil {
		m.debug.OnEmbed(m.scratch.x)
	}

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
	case format.GgmlQ4_K:
		return ops.EmbeddingQ4_KInto(m.scratch.x, raw, m.cfg.EmbeddingDim, tokenID)
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

	if err := m.rmsNormInto(m.scratch.h, m.scratch.x, ln.attnNorm, layer); err != nil {
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

	m.applyRoPEHeads(m.scratch.q, m.cfg.NumHeads, pos, layer)
	m.applyRoPEHeads(m.scratch.k, m.cfg.NumKVHeads, pos, layer)

	kvPos := m.cache.Len()
	m.cache.Append(layer, m.scratch.k, m.scratch.v)
	if m.gpu != nil && gpu.LayerOnGPU(layer, m.ngl, m.cfg.NumLayers) {
		_ = m.gpu.KVCacheAppend(layer, kvPos, m.scratch.k, m.scratch.v)
	}

	seqLen := m.cache.Len() + 1

	if err := m.attentionScoresInto(m.scratch.attn, m.scratch.q, m.cache.KLayer(layer), m.cache.VLayer(layer), m.scratch.scores, seqLen, layer); err != nil {
		return err
	}

	if err := m.matmulInto(lt.attnOut, m.cfg.EmbeddingDim, m.cfg.NumHeads*m.cfg.HeadDim, m.scratch.attn, m.scratch.h, layer); err != nil {
		return err
	}
	ops.AddInPlace(m.scratch.x, m.scratch.h)

	if err := m.rmsNormInto(m.scratch.h, m.scratch.x, ln.ffnNorm, layer); err != nil {
		return err
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

	ops.AddInPlace(m.scratch.x, m.scratch.h)
	return nil
}

func (m *Model) matmulInto(name string, rows, cols int, vec, out []float32, layer int) error {
	if m.gpu != nil && gpu.LayerOnGPU(layer, m.ngl, m.cfg.NumLayers) {
		got, err := m.matmulGPU(name, rows, cols, vec)
		if err != nil {
			return err
		}
		copy(out, got)
		return nil
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
	case format.GgmlQ4_K:
		return ops.MatMulVecQ4_KInto(raw, rows, cols, vec, out)
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

	if info.Type == format.GgmlQ8_0 {
		raw, err := m.weights.Raw(name)
		if err != nil {
			return nil, err
		}
		return m.gpu.MatMulVecQ8_0Cached(name, raw, rows, cols, vec)
	}

	f32, err := m.weights.Floats(name)
	if err != nil {
		return nil, err
	}
	return m.gpu.MatMulVecCached(name, f32, rows, cols, vec)
}

func (m *Model) logitsFinish() error {
	if err := m.logitsFromHidden(m.scratch.x); err != nil {
		return err
	}
	return nil
}

func (m *Model) logitsFromHidden(x []float32) error {
	if err := ops.RMSNormInto(m.scratch.h, x, m.outNorm, m.cfg.RMSNormEps); err != nil {
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

	switch info.Type {
	case format.GgmlQ8_0:
		return ops.MatMulVecQ8_0Into(raw, m.cfg.VocabSize, m.cfg.EmbeddingDim, m.scratch.h, m.scratch.logits)
	case format.GgmlQ4_0:
		return ops.MatMulVecQ4_0Into(raw, m.cfg.VocabSize, m.cfg.EmbeddingDim, m.scratch.h, m.scratch.logits)
	case format.GgmlQ4_K:
		return ops.MatMulVecQ4_KInto(raw, m.cfg.VocabSize, m.cfg.EmbeddingDim, m.scratch.h, m.scratch.logits)
	default:
		f32, err := m.weights.Floats(name)
		if err != nil {
			return err
		}
		return ops.MatMulVecInto(f32, m.cfg.VocabSize, m.cfg.EmbeddingDim, m.scratch.h, m.scratch.logits)
	}
}

func (m *Model) rmsNormInto(dst, x, weight []float32, layer int) error {
	if m.gpu != nil && gpu.LayerOnGPU(layer, m.ngl, m.cfg.NumLayers) {
		if err := m.gpu.RMSNormInto(dst, x, weight, m.cfg.RMSNormEps); err == nil {
			return nil
		}
	}
	return ops.RMSNormInto(dst, x, weight, m.cfg.RMSNormEps)
}

func (m *Model) applyRoPEHeads(v []float32, nHeads, pos, layer int) {
	if m.gpu != nil && gpu.LayerOnGPU(layer, m.ngl, m.cfg.NumLayers) {
		if err := m.gpu.ApplyRoPEHeads(v, nHeads, m.cfg.HeadDim, pos, m.cfg.RopeFreqBase); err == nil {
			return
		}
	}
	ops.ApplyRoPEHeadsNorm(v, nHeads, m.cfg.HeadDim, pos, m.cfg.RopeFreqBase)
}

func (m *Model) swigluInPlace(gate, up []float32, layer int) {
	if m.gpu != nil && gpu.LayerOnGPU(layer, m.ngl, m.cfg.NumLayers) {
		if err := m.gpu.SwiGLUInPlace(gate, up); err == nil {
			return
		}
	}
	ops.SwiGLUInPlace(gate, up)
}

func (m *Model) attentionScoresInto(dst, q, k, v, scores []float32, seqLen, layer int) error {
	if m.gpu != nil && gpu.LayerOnGPU(layer, m.ngl, m.cfg.NumLayers) {
		if err := m.gpu.AttentionScoresKV(layer, dst, q, seqLen, m.cfg.NumHeads, m.cfg.NumKVHeads, m.cfg.HeadDim); err == nil {
			return nil
		}

		if err := m.gpu.AttentionScoresInto(dst, q, k, v, scores, seqLen, m.cfg.NumHeads, m.cfg.NumKVHeads, m.cfg.HeadDim); err == nil {
			return nil
		}
	}

	return ops.AttentionScoresInto(dst, q, k, v, scores, seqLen, m.cfg.NumHeads, m.cfg.NumKVHeads, m.cfg.HeadDim)
}
