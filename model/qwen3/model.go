package qwen3

import (
	"fmt"

	"github.com/magomedcoder/gguf.go"
	"github.com/magomedcoder/gguf.go/ops"
	"github.com/magomedcoder/gguf.go/weights"
)

// Model - Qwen3 transformer
type Model struct {
	cfg     Config
	weights *weights.Store
	cache   *KVCache
}

// Load создаёт Qwen3 из весов
func Load(w *weights.Store) (*Model, error) {
	cfg, err := ParseConfig(w.Reader())
	if err != nil {
		return nil, err
	}
	return &Model{
		cfg:     cfg,
		weights: w,
		cache:   NewKVCache(cfg),
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
		x, err = m.forwardToken(tok, pos)
		if err != nil {
			return nil, err
		}
	}

	return m.logits(x)
}

func (m *Model) forwardToken(tokenID, pos int) ([]float32, error) {
	embd, err := m.weights.Raw("token_embd.weight")
	if err != nil {
		return nil, err
	}

	x, err := ops.EmbeddingQ8_0(embd, m.cfg.EmbeddingDim, tokenID)
	if err != nil {
		return nil, err
	}

	for layer := 0; layer < m.cfg.NumLayers; layer++ {
		x, err = m.forwardBlock(layer, x, pos)
		if err != nil {
			return nil, err
		}
	}
	m.cache.Advance()

	return x, nil
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

	q, err := m.matmul(p+"attn_q.weight", m.cfg.NumHeads*m.cfg.HeadDim, m.cfg.EmbeddingDim, h)
	if err != nil {
		return nil, err
	}

	k, err := m.matmul(p+"attn_k.weight", m.cfg.NumKVHeads*m.cfg.HeadDim, m.cfg.EmbeddingDim, h)
	if err != nil {
		return nil, err
	}

	v, err := m.matmul(p+"attn_v.weight", m.cfg.NumKVHeads*m.cfg.HeadDim, m.cfg.EmbeddingDim, h)
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

	attnOut, err := m.matmul(p+"attn_output.weight", m.cfg.EmbeddingDim, m.cfg.EmbeddingDim, attn)
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

	gate, err := m.matmul(p+"ffn_gate.weight", m.cfg.FFNHidden, m.cfg.EmbeddingDim, h)
	if err != nil {
		return nil, err
	}

	up, err := m.matmul(p+"ffn_up.weight", m.cfg.FFNHidden, m.cfg.EmbeddingDim, h)
	if err != nil {
		return nil, err
	}
	hidden := ops.SwiGLU(gate, up)

	down, err := m.matmul(p+"ffn_down.weight", m.cfg.EmbeddingDim, m.cfg.FFNHidden, hidden)
	if err != nil {
		return nil, err
	}

	x = ops.Add(x, down)

	return x, nil
}

func (m *Model) matmul(name string, rows, cols int, vec []float32) ([]float32, error) {
	raw, err := m.weights.Raw(name)
	if err != nil {
		return nil, err
	}

	info, err := m.weights.Info(name)
	if err != nil {
		return nil, err
	}

	switch info.Type {
	case gguf.GgmlQ8_0:
		return ops.MatMulVecQ8_0(raw, rows, cols, vec)
	default:
		f32, err := m.weights.Floats(name)
		if err != nil {
			return nil, err
		}

		return ops.MatMulVec(f32, rows, cols, vec)
	}
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

	switch info.Type {
	case gguf.GgmlQ8_0:
		return ops.MatMulVecQ8_0(raw, m.cfg.VocabSize, m.cfg.EmbeddingDim, x)
	default:
		f32, err := m.weights.Floats(name)
		if err != nil {
			return nil, err
		}

		return ops.MatMulVec(f32, m.cfg.VocabSize, m.cfg.EmbeddingDim, x)
	}
}

func applyRoPEHeads(v []float32, nHeads, headDim, pos int, freqBase float32) {
	for h := 0; h < nHeads; h++ {
		off := h * headDim
		ops.ApplyRoPE(v[off:off+headDim], pos, freqBase)
	}
}
