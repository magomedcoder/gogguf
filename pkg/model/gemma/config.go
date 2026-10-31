package gemma

import (
	"fmt"
	"math"

	"github.com/magomedcoder/gogguf/pkg/format"
)

// Variant - Gemma family
type Variant int

const (
	VariantGemma Variant = iota
	VariantGemma2
)

// Config - Gemma / Gemma2 hyperparameters from GGUF metadata
type Config struct {
	Variant           Variant
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
	RopeFreqBaseSWA   float32 // gemma2 SWA layers; 0 -> RopeFreqBase
	SlidingWindow     int     // gemma2; 0 = no SWA
	SWAPeriod         int     // gemma2 pattern; 0/2: even=SWA
	AttnLogitSoftcap  float32 // gemma2; 0 = off
	FinalLogitSoftcap float32 // gemma2; 0 = off
	EmbedScale        float32 // sqrt(embd); set during parsing
}

// ParseConfigGemma reads gemma.*
func ParseConfigGemma(r *format.Reader) (Config, error) {
	return parseConfig(r, "gemma.", VariantGemma)
}

// ParseConfigGemma2 reads gemma2.*
func ParseConfigGemma2(r *format.Reader) (Config, error) {
	return parseConfig(r, "gemma2.", VariantGemma2)
}

func parseConfig(r *format.Reader, prefix string, variant Variant) (Config, error) {
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

	kvHeads := heads
	if v, err := meta.Int("attention.head_count_kv"); err == nil && v > 0 {
		kvHeads = v
	}

	eps := meta.Float32Or("attention.layer_norm_rms_epsilon", 1e-6)
	freqBase := meta.Float32Or("rope.freq_base", 10000)

	if heads <= 0 {
		return Config{}, fmt.Errorf("gemma: attention.head_count=%d", heads)
	}

	headDim := emb / heads
	if v, err := meta.Int("attention.key_length"); err == nil && v > 0 {
		headDim = v
	}

	vocab, err := format.VocabSizeFromEmb(r, emb, "gemma")
	if err != nil {
		return Config{}, err
	}

	cfg := Config{
		Variant:       variant,
		ContextLength: ctx,
		EmbeddingDim:  emb,
		FFNHidden:     ffn,
		NumLayers:     layers,
		NumHeads:      heads,
		NumKVHeads:    kvHeads,
		HeadDim:       headDim,
		VocabSize:     vocab,
		RMSNormEps:    eps,
		RopeFreqBase:  freqBase,
		EmbedScale:    float32(math.Sqrt(float64(emb))),
	}

	if variant == VariantGemma2 {
		cfg.SlidingWindow = 4096
		if v, err := meta.Int("attention.sliding_window"); err == nil && v > 0 {
			cfg.SlidingWindow = v
		}

		cfg.SWAPeriod = 2
		if v, err := meta.Int("attention.sliding_window_pattern"); err == nil && v > 0 {
			cfg.SWAPeriod = v
		}

		cfg.RopeFreqBaseSWA = freqBase
		if v := meta.Float32Or("rope.freq_base_swa", 0); v > 0 {
			cfg.RopeFreqBaseSWA = v
		}

		cfg.AttnLogitSoftcap = 50
		if v := meta.Float32Or("attn_logit_softcapping", 0); v > 0 {
			cfg.AttnLogitSoftcap = v
		}

		cfg.FinalLogitSoftcap = 30
		if v := meta.Float32Or("final_logit_softcapping", 0); v > 0 {
			cfg.FinalLogitSoftcap = v
		}
	}

	return cfg, nil
}

// IsSWALayer gemma2, period=2 -> even layers are SWA (like llama.cpp set_swa_pattern)
func (c Config) IsSWALayer(layer int) bool {
	if c.Variant != VariantGemma2 || c.SlidingWindow <= 0 {
		return false
	}

	p := c.SWAPeriod
	if p <= 0 {
		p = 2
	}

	return layer%p < (p - 1)
}

func (c Config) ropeBase(layer int) float32 {
	if c.IsSWALayer(layer) && c.RopeFreqBaseSWA > 0 {
		return c.RopeFreqBaseSWA
	}

	return c.RopeFreqBase
}
