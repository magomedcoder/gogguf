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

	var x []float32
	var err error

	for i, tok := range tokenIDs {
		pos := startPos + i
		last := i == len(tokenIDs)-1
		x, err = m.forwardToken(tok, pos, last)
		if err != nil {
			return nil, err
		}
	}

	return m.logits(x)
}

func (m *Model) forwardToken(tokenID, pos int, debug bool) ([]float32, error) {
	x, err := m.embedToken(tokenID)
	if err != nil {
		return nil, err
	}

	if debug && m.debug != nil && m.debug.OnEmbed != nil {
		m.debug.OnEmbed(x)
	}

	for layer := 0; layer < m.cfg.NumLayers; layer++ {
		x, err = m.forwardBlock(layer, x, pos)
		if err != nil {
			return nil, err
		}

		if debug && m.debug != nil && m.debug.OnLayer != nil {
			m.debug.OnLayer(layer, x)
		}
	}
	m.cache.Advance()

	return x, nil
}

func (m *Model) embedToken(tokenID int) ([]float32, error) {
	raw, err := m.weights.Raw("token_embd.weight")
	if err != nil {
		return nil, err
	}

	info, err := m.weights.Info("token_embd.weight")
	if err != nil {
		return nil, err
	}

	switch info.Type {
	case format.GgmlQ8_0:
		return ops.EmbeddingQ8_0(raw, m.cfg.EmbeddingDim, tokenID)
	case format.GgmlQ4_0:
		return ops.EmbeddingQ4_0(raw, m.cfg.EmbeddingDim, tokenID)
	case format.GgmlQ4_K:
		return ops.EmbeddingQ4_K(raw, m.cfg.EmbeddingDim, tokenID)
	default:
		f32, err := m.weights.Floats("token_embd.weight")
		if err != nil {
			return nil, err
		}
		off := tokenID * m.cfg.EmbeddingDim
		row := f32[off : off+m.cfg.EmbeddingDim]
		out := make([]float32, len(row))
		copy(out, row)
		return out, nil
	}
}

func (m *Model) forwardBlock(layer int, x []float32, pos int) ([]float32, error) {
	p := fmt.Sprintf("blk.%d.", layer)

	attnNorm, err := m.weights.Floats(p + "attn_norm.weight")
	if err != nil {
		return nil, err
	}

	h, err := ops.RMSNorm(x, attnNorm, m.cfg.RMSNormEps)
	if err != nil {
		return nil, err
	}

	q, err := m.matmul(p+"attn_q.weight", m.cfg.NumHeads*m.cfg.HeadDim, m.cfg.EmbeddingDim, h, layer)
	if err != nil {
		return nil, err
	}

	k, err := m.matmul(p+"attn_k.weight", m.cfg.NumKVHeads*m.cfg.HeadDim, m.cfg.EmbeddingDim, h, layer)
	if err != nil {
		return nil, err
	}

	v, err := m.matmul(p+"attn_v.weight", m.cfg.NumKVHeads*m.cfg.HeadDim, m.cfg.EmbeddingDim, h, layer)
	if err != nil {
		return nil, err
	}

	q, err = m.normHeads(q, p+"attn_q_norm.weight", m.cfg.NumHeads)
	if err != nil {
		return nil, err
	}

	k, err = m.normHeads(k, p+"attn_k_norm.weight", m.cfg.NumKVHeads)
	if err != nil {
		return nil, err
	}

	applyRoPEHeads(q, m.cfg.NumHeads, m.cfg.HeadDim, pos, m.cfg.RopeFreqBase)
	applyRoPEHeads(k, m.cfg.NumKVHeads, m.cfg.HeadDim, pos, m.cfg.RopeFreqBase)

	m.cache.Append(layer, k, v)
	seqLen := m.cache.Len() + 1

	attn, err := ops.AttentionScores(q, m.cache.KLayer(layer), m.cache.VLayer(layer), seqLen, m.cfg.NumHeads, m.cfg.NumKVHeads, m.cfg.HeadDim)
	if err != nil {
		return nil, err
	}

	attnOut, err := m.matmul(p+"attn_output.weight", m.cfg.EmbeddingDim, m.cfg.NumHeads*m.cfg.HeadDim, attn, layer)
	if err != nil {
		return nil, err
	}
	x = ops.Add(x, attnOut)

	ffnNorm, err := m.weights.Floats(p + "ffn_norm.weight")
	if err != nil {
		return nil, err
	}

	h, err = ops.RMSNorm(x, ffnNorm, m.cfg.RMSNormEps)
	if err != nil {
		return nil, err
	}

	gate, err := m.matmul(p+"ffn_gate.weight", m.cfg.FFNHidden, m.cfg.EmbeddingDim, h, layer)
	if err != nil {
		return nil, err
	}

	up, err := m.matmul(p+"ffn_up.weight", m.cfg.FFNHidden, m.cfg.EmbeddingDim, h, layer)
	if err != nil {
		return nil, err
	}
	hidden := ops.SwiGLU(gate, up)

	down, err := m.matmul(p+"ffn_down.weight", m.cfg.EmbeddingDim, m.cfg.FFNHidden, hidden, layer)
	if err != nil {
		return nil, err
	}

	x = ops.Add(x, down)

	return x, nil
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

func (m *Model) logits(x []float32) ([]float32, error) {
	outNorm, err := m.weights.Floats("output_norm.weight")
	if err != nil {
		return nil, err
	}

	x, err = ops.RMSNorm(x, outNorm, m.cfg.RMSNormEps)
	if err != nil {
		return nil, err
	}

	// lm_head: output.weight или tied token_embd
	name := "output.weight"
	if _, err := m.weights.Info(name); err != nil {
		name = "token_embd.weight"
	}

	raw, err := m.weights.Raw(name)
	if err != nil {
		return nil, err
	}

	info, err := m.weights.Info(name)
	if err != nil {
		return nil, err
	}

	var out []float32
	switch info.Type {
	case format.GgmlQ8_0:
		out, err = ops.MatMulVecQ8_0(raw, m.cfg.VocabSize, m.cfg.EmbeddingDim, x)
	case format.GgmlQ4_0:
		out, err = ops.MatMulVecQ4_0(raw, m.cfg.VocabSize, m.cfg.EmbeddingDim, x)
	case format.GgmlQ4_K:
		out, err = ops.MatMulVecQ4_K(raw, m.cfg.VocabSize, m.cfg.EmbeddingDim, x)
	default:
		f32, err := m.weights.Floats(name)
		if err != nil {
			return nil, err
		}

		out, err = ops.MatMulVec(f32, m.cfg.VocabSize, m.cfg.EmbeddingDim, x)
	}

	if err != nil {
		return nil, err
	}

	if m.debug != nil && m.debug.OnLogits != nil {
		m.debug.OnLogits(out)
	}

	return out, nil
}

func applyRoPEHeads(v []float32, nHeads, headDim, pos int, freqBase float32) {
	for h := range nHeads {
		off := h * headDim
		ops.ApplyRoPE(v[off:off+headDim], pos, freqBase)
	}
}

func (m *Model) normHeads(v []float32, weightName string, nHeads int) ([]float32, error) {
	weight, err := m.weights.Floats(weightName)
	if err != nil {
		return nil, err
	}

	if len(weight) != m.cfg.HeadDim {
		return nil, fmt.Errorf("qwen3: %s: len=%d, head_dim=%d", weightName, len(weight), m.cfg.HeadDim)
	}

	out := make([]float32, len(v))
	for h := range nHeads {
		off := h * m.cfg.HeadDim
		normed, err := ops.RMSNorm(v[off:off+m.cfg.HeadDim], weight, m.cfg.RMSNormEps)
		if err != nil {
			return nil, err
		}
		copy(out[off:off+m.cfg.HeadDim], normed)
	}

	return out, nil
}
