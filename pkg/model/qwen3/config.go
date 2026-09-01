package qwen3

import (
	"fmt"

	"github.com/magomedcoder/gogguf/pkg/format"
)

// Config - гиперпараметры Qwen3 из метаданных GGUF
type Config struct {
	ContextLength     int
	EmbeddingDim      int
	FFNHidden         int
	NumLayers         int
	NumHeads          int
	NumKVHeads        int
	HeadDim           int
	VocabSize         int
	RMSNormEps        float32
	RopeFreqBase      float32
	ExpertCount       int
	ExpertUsedCount   int
	ExpertFFN         int
	ExpertWeightScale float32
	MoENormWeights    bool
}

// ParseConfigMoE читает qwen3moe.*
func ParseConfigMoE(r *format.Reader) (Config, error) {
	cfg, err := parseConfigWithPrefix(r, "qwen3moe.")
	if err != nil {
		return Config{}, err
	}

	if cfg.ExpertCount <= 0 {
		return Config{}, fmt.Errorf("qwen3moe: expert_count=%d", cfg.ExpertCount)
	}

	cfg.MoENormWeights = true
	if cfg.ExpertFFN <= 0 && cfg.ExpertUsedCount > 0 {
		cfg.ExpertFFN = cfg.FFNHidden / cfg.ExpertUsedCount
	}

	return cfg, nil
}

// ParseConfig читает конфиг из метаданных GGUF
func ParseConfig(r *format.Reader) (Config, error) {
	return parseConfigWithPrefix(r, "qwen3.")
}

func parseConfigWithPrefix(r *format.Reader, prefix string) (Config, error) {
	getInt := func(key string) (int, error) {
		return r.Metadata.Int(prefix + key)
	}

	ctx, err := getInt("context_length")
	if err != nil {
		return Config{}, err
	}

	emb, err := getInt("embedding_length")
	if err != nil {
		return Config{}, err
	}

	ffn, err := getInt("feed_forward_length")
	if err != nil {
		return Config{}, err
	}

	layers, err := getInt("block_count")
	if err != nil {
		return Config{}, err
	}

	heads, err := getInt("attention.head_count")
	if err != nil {
		return Config{}, err
	}

	kvHeads, err := getInt("attention.head_count_kv")
	if err != nil {
		return Config{}, err
	}

	eps := float32(1e-6)
	if v, err := format.MetaValue[float32](r.Metadata, prefix+"attention.layer_norm_rms_epsilon"); err == nil {
		eps = v
	}

	freqBase := float32(1000000)
	if v, err := format.MetaValue[float32](r.Metadata, prefix+"rope.freq_base"); err == nil {
		freqBase = v
	}

	headDim := emb / heads
	if v, err := getInt("attention.key_length"); err == nil && v > 0 {
		headDim = v
	}

	if heads*headDim != emb {
		qOut := heads * headDim
		if qOut <= 0 {
			return Config{}, fmt.Errorf("qwen3: некорректный head_dim=%d", headDim)
		}
		_ = qOut
	}

	nExpert := 0
	if v, err := getInt("expert_count"); err == nil {
		nExpert = v
	}

	nExpertUsed := 0
	if v, err := getInt("expert_used_count"); err == nil {
		nExpertUsed = v
	}

	expertFFN := 0
	if v, err := getInt("expert_feed_forward_length"); err == nil && v > 0 {
		expertFFN = v
	}

	wScale := float32(1)
	if v, err := format.MetaValue[float32](r.Metadata, prefix+"expert_weights_scale"); err == nil && v != 0 {
		wScale = v
	}

	if nExpert > 0 {
		if nExpertUsed <= 0 {
			nExpertUsed = 2
		}

		if nExpertUsed > nExpert {
			return Config{}, fmt.Errorf("qwen3: expert_used_count=%d > expert_count=%d", nExpertUsed, nExpert)
		}
	}

	vocab, err := vocabSize(r, emb)
	if err != nil {
		return Config{}, err
	}

	return Config{
		ContextLength:     ctx,
		EmbeddingDim:      emb,
		FFNHidden:         ffn,
		NumLayers:         layers,
		NumHeads:          heads,
		NumKVHeads:        kvHeads,
		HeadDim:           headDim,
		VocabSize:         vocab,
		RMSNormEps:        eps,
		RopeFreqBase:      freqBase,
		ExpertCount:       nExpert,
		ExpertUsedCount:   nExpertUsed,
		ExpertFFN:         expertFFN,
		ExpertWeightScale: wScale,
	}, nil
}

func (c Config) isMoE() bool {
	return c.ExpertCount > 0
}

func (c Config) expertFFN() int {
	if c.ExpertFFN > 0 {
		return c.ExpertFFN
	}

	return c.FFNHidden
}

func (c Config) maxFFN() int {
	m := c.FFNHidden
	if e := c.expertFFN(); e > m {
		m = e
	}

	return m
}

func vocabSize(r *format.Reader, emb int) (int, error) {
	info, err := r.TensorInfo("token_embd.weight")
	if err != nil {
		return 0, err
	}

	if len(info.Dimensions) != 2 {
		return 0, fmt.Errorf("qwen3: token_embd.weight: ожидается 2D")
	}

	a, b := int(info.Dimensions[0]), int(info.Dimensions[1])
	if a == emb {
		return b, nil
	}

	if b == emb {
		return a, nil
	}

	return 0, fmt.Errorf("qwen3: token_embd.weight %v не содержит embedding_length=%d", info.Dimensions, emb)
}
