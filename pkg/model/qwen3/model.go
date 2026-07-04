package qwen3

import (
	"fmt"

	"github.com/magomedcoder/gguf.go/pkg/format"
	"github.com/magomedcoder/gguf.go/pkg/gpu"
	"github.com/magomedcoder/gguf.go/pkg/ops"
	"github.com/magomedcoder/gguf.go/pkg/weights"
)

// Model - Qwen3 transformer
type Model struct {
	cfg     Config
	weights *weights.Store
	cache   *KVCache
	gpu     gpu.Backend
	ngl     int
	debug   *DebugHooks
	scratch scratch
}

// Load создаёт Qwen3 из весов
func Load(w *weights.Store, g gpu.Backend, ngl int) (*Model, error) {
	cfg, err := ParseConfig(w.Reader())
	if err != nil {
		return nil, err
	}

	if ngl > cfg.NumLayers {
		return nil, fmt.Errorf("qwen3: ngl=%d больше числа слоёв %d", ngl, cfg.NumLayers)
	}

	return &Model{
		cfg:     cfg,
		weights: w,
		cache:   NewKVCache(cfg),
		gpu:     g,
		ngl:     ngl,
		scratch: newScratch(cfg),
	}, nil
}

// Config возвращает конфигурацию модели
func (m *Model) Config() Config {
	return m.cfg
}

// ResetCache сбрасывает KV-cache
func (m *Model) ResetCache() {
	m.cache.Reset()
}

// SetDebugHooks включает колбэки для пошаговой отладки forward pass
func (m *Model) SetDebugHooks(h *DebugHooks) {
	m.debug = h
}

// Forward выполняет forward pass для последовательности tokenIDs начиная с startPos
// Возвращает logits для последнего токена [vocabSize]
func (m *Model) Forward(tokenIDs []int, startPos int) ([]float32, error) {
	if len(tokenIDs) == 0 {
		return nil, fmt.Errorf("qwen3: пустой ввод")
	}

	var err error

	for i, tok := range tokenIDs {
		pos := startPos + i
		last := i == len(tokenIDs)-1
		if err = m.forwardToken(tok, pos, last); err != nil {
			return nil, err
		}
	}

	if err = m.logits(); err != nil {
		return nil, err
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

		if debug && m.debug != nil && m.debug.OnLayer != nil {
			m.debug.OnLayer(layer, m.scratch.x)
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
	p := fmt.Sprintf("blk.%d.", layer)

	attnNorm, err := m.weights.Floats(p + "attn_norm.weight")
	if err != nil {
		return err
	}

	if err := ops.RMSNormInto(m.scratch.h, m.scratch.x, attnNorm, m.cfg.RMSNormEps); err != nil {
		return err
	}

	if err := m.matmulInto(p+"attn_q.weight", m.cfg.NumHeads*m.cfg.HeadDim, m.cfg.EmbeddingDim, m.scratch.h, m.scratch.q, layer); err != nil {
		return err
	}

	if err := m.matmulInto(p+"attn_k.weight", m.cfg.NumKVHeads*m.cfg.HeadDim, m.cfg.EmbeddingDim, m.scratch.h, m.scratch.k, layer); err != nil {
		return err
	}

	if err := m.matmulInto(p+"attn_v.weight", m.cfg.NumKVHeads*m.cfg.HeadDim, m.cfg.EmbeddingDim, m.scratch.h, m.scratch.v, layer); err != nil {
		return err
	}

	if err := m.normHeadsInto(m.scratch.q, p+"attn_q_norm.weight", m.cfg.NumHeads); err != nil {
		return err
	}

	if err := m.normHeadsInto(m.scratch.k, p+"attn_k_norm.weight", m.cfg.NumKVHeads); err != nil {
		return err
	}

	applyRoPEHeads(m.scratch.q, m.cfg.NumHeads, m.cfg.HeadDim, pos, m.cfg.RopeFreqBase)
	applyRoPEHeads(m.scratch.k, m.cfg.NumKVHeads, m.cfg.HeadDim, pos, m.cfg.RopeFreqBase)

	m.cache.Append(layer, m.scratch.k, m.scratch.v)
	seqLen := m.cache.Len() + 1

	if err := ops.AttentionScoresInto(m.scratch.attn, m.scratch.q, m.cache.KLayer(layer), m.cache.VLayer(layer), m.scratch.scores, seqLen, m.cfg.NumHeads, m.cfg.NumKVHeads, m.cfg.HeadDim); err != nil {
		return err
	}

	if err := m.matmulInto(p+"attn_output.weight", m.cfg.EmbeddingDim, m.cfg.NumHeads*m.cfg.HeadDim, m.scratch.attn, m.scratch.h, layer); err != nil {
		return err
	}
	ops.AddInPlace(m.scratch.x, m.scratch.h)

	ffnNorm, err := m.weights.Floats(p + "ffn_norm.weight")
	if err != nil {
		return err
	}

	if err := ops.RMSNormInto(m.scratch.h, m.scratch.x, ffnNorm, m.cfg.RMSNormEps); err != nil {
		return err
	}

	if err := m.matmulInto(p+"ffn_gate.weight", m.cfg.FFNHidden, m.cfg.EmbeddingDim, m.scratch.h, m.scratch.gate, layer); err != nil {
		return err
	}

	if err := m.matmulInto(p+"ffn_up.weight", m.cfg.FFNHidden, m.cfg.EmbeddingDim, m.scratch.h, m.scratch.up, layer); err != nil {
		return err
	}
	ops.SwiGLUInPlace(m.scratch.gate, m.scratch.up)

	if err := m.matmulInto(p+"ffn_down.weight", m.cfg.EmbeddingDim, m.cfg.FFNHidden, m.scratch.gate, m.scratch.h, layer); err != nil {
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

func (m *Model) matmul(name string, rows, cols int, vec []float32, layer int) ([]float32, error) {
	if m.gpu != nil && gpu.LayerOnGPU(layer, m.ngl, m.cfg.NumLayers) {
		return m.matmulGPU(name, rows, cols, vec)
	}

	raw, err := m.weights.Raw(name)
	if err != nil {
		return nil, err
	}

	info, err := m.weights.Info(name)
	if err != nil {
		return nil, err
	}

	switch info.Type {
	case format.GgmlQ8_0:
		return ops.MatMulVecQ8_0(raw, rows, cols, vec)
	case format.GgmlQ4_0:
		return ops.MatMulVecQ4_0(raw, rows, cols, vec)
	case format.GgmlQ4_K:
		return ops.MatMulVecQ4_K(raw, rows, cols, vec)
	default:
		f32, err := m.weights.Floats(name)
		if err != nil {
			return nil, err
		}

		return ops.MatMulVec(f32, rows, cols, vec)
	}
}

// matmulGPU выполняет matmul на GPU (Q8_0 без деквантизации, иначе FP32)
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

func (m *Model) logits() error {
	outNorm, err := m.weights.Floats("output_norm.weight")
	if err != nil {
		return err
	}

	if err := ops.RMSNormInto(m.scratch.h, m.scratch.x, outNorm, m.cfg.RMSNormEps); err != nil {
		return err
	}

	name := "output.weight"
	if _, err := m.weights.Info(name); err != nil {
		name = "token_embd.weight"
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
		err = ops.MatMulVecQ8_0Into(raw, m.cfg.VocabSize, m.cfg.EmbeddingDim, m.scratch.h, m.scratch.logits)
	case format.GgmlQ4_0:
		err = ops.MatMulVecQ4_0Into(raw, m.cfg.VocabSize, m.cfg.EmbeddingDim, m.scratch.h, m.scratch.logits)
	case format.GgmlQ4_K:
		err = ops.MatMulVecQ4_KInto(raw, m.cfg.VocabSize, m.cfg.EmbeddingDim, m.scratch.h, m.scratch.logits)
	default:
		f32, err := m.weights.Floats(name)
		if err != nil {
			return err
		}
		err = ops.MatMulVecInto(f32, m.cfg.VocabSize, m.cfg.EmbeddingDim, m.scratch.h, m.scratch.logits)
	}

	if err != nil {
		return err
	}

	if m.debug != nil && m.debug.OnLogits != nil {
		m.debug.OnLogits(m.scratch.logits)
	}

	return nil
}

func applyRoPEHeads(v []float32, nHeads, headDim, pos int, freqBase float32) {
	for h := range nHeads {
		off := h * headDim
		ops.ApplyRoPE(v[off:off+headDim], pos, freqBase)
	}
}

func (m *Model) normHeadsInto(v []float32, weightName string, nHeads int) error {
	weight, err := m.weights.Floats(weightName)
	if err != nil {
		return err
	}

	if len(weight) != m.cfg.HeadDim {
		return fmt.Errorf("qwen3: %s: len=%d, head_dim=%d", weightName, len(weight), m.cfg.HeadDim)
	}

	for h := range nHeads {
		off := h * m.cfg.HeadDim
		if err := ops.RMSNormInto(v[off:off+m.cfg.HeadDim], v[off:off+m.cfg.HeadDim], weight, m.cfg.RMSNormEps); err != nil {
			return err
		}
	}

	return nil
}
