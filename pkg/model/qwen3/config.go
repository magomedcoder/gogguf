package qwen3

import (
	"fmt"

	"github.com/magomedcoder/gogguf/pkg/format"
)

// Config - Qwen3 hyperparameters from GGUF metadata
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

// ParseConfigMoE reads qwen3moe.*
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

// ParseConfig reads config from GGUF metadata
func ParseConfig(r *format.Reader) (Config, error) {
	return parseConfigWithPrefix(r, "qwen3.")
}

func parseConfigWithPrefix(r *format.Reader, prefix string) (Config, error) {
	meta := format.PrefixedMeta{Prefix: prefix, MD: r.Metadata}

	ctx, err := meta.Int("context_length")
	if err != nil {
		return Config{}, err
	}

	emb, err := meta.Int("embedding_length")
	if err != nil {
		return Config{}, err
	}

	ffn, err := meta.Int("feed_forward_length")
	if err != nil {
		return Config{}, err
	}

	layers, err := meta.Int("block_count")
	if err != nil {
		return Config{}, err
	}

	heads, err := meta.Int("attention.head_count")
	if err != nil {
		return Config{}, err
	}

	kvHeads, err := meta.Int("attention.head_count_kv")
	if err != nil {
		return Config{}, err
	}

	eps := meta.Float32Or("attention.layer_norm_rms_epsilon", 1e-6)
	freqBase := meta.Float32Or("rope.freq_base", float32(1000000))

	headDim := emb / heads
	if v, err := meta.Int("attention.key_length"); err == nil && v > 0 {
		headDim = v
	}

	if heads*headDim != emb {
		qOut := heads * headDim
		if qOut <= 0 {
			return Config{}, fmt.Errorf("qwen3: invalid head_dim=%d", headDim)
		}
		_ = qOut
	}

	nExpert := meta.IntOr("expert_count", 0)
	nExpertUsed := meta.IntOr("expert_used_count", 0)

	expertFFN := 0
	if v, err := meta.Int("expert_feed_forward_length"); err == nil && v > 0 {
		expertFFN = v
	}

	wScale := meta.Float32Or("expert_weights_scale", 1)
	if wScale == 0 {
		wScale = 1
	}

	if nExpert > 0 {
		if nExpertUsed <= 0 {
			nExpertUsed = 2
		}

		if nExpertUsed > nExpert {
			return Config{}, fmt.Errorf("qwen3: expert_used_count=%d > expert_count=%d", nExpertUsed, nExpert)
		}
	}

	vocab, err := format.VocabSizeFromEmb(r, emb, "qwen3")
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
