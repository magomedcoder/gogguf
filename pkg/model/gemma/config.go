package gemma

import (
	"fmt"
	"math"

	"github.com/magomedcoder/gogguf/pkg/format"
)

// Variant - семейство Gemma
type Variant int

const (
	VariantGemma Variant = iota
	VariantGemma2
)

// Config - гиперпараметры Gemma / Gemma2 из метаданных GGUF
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
	SlidingWindow     int     // gemma2; 0 = без SWA
	SWAPeriod         int     // gemma2 pattern; 0/2: even=SWA
	AttnLogitSoftcap  float32 // gemma2; 0 = off
	FinalLogitSoftcap float32 // gemma2; 0 = off
	EmbedScale        float32 // sqrt(embd); задаётся при парсинге
}

// ParseConfigGemma читает gemma.*
func ParseConfigGemma(r *format.Reader) (Config, error) {
	return parseConfig(r, "gemma.", VariantGemma)
}

// ParseConfigGemma2 читает gemma2.*
func ParseConfigGemma2(r *format.Reader) (Config, error) {
	return parseConfig(r, "gemma2.", VariantGemma2)
}

func parseConfig(r *format.Reader, prefix string, variant Variant) (Config, error) {
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

	kvHeads := heads
	if v, err := getInt("attention.head_count_kv"); err == nil && v > 0 {
		kvHeads = v
	}

	eps := float32(1e-6)
	if v, err := format.MetaValue[float32](r.Metadata, prefix+"attention.layer_norm_rms_epsilon"); err == nil {
		eps = v
	}

	freqBase := float32(10000)
	if v, err := format.MetaValue[float32](r.Metadata, prefix+"rope.freq_base"); err == nil {
		freqBase = v
	}

	if heads <= 0 {
		return Config{}, fmt.Errorf("gemma: attention.head_count=%d", heads)
	}

	headDim := emb / heads
	if v, err := getInt("attention.key_length"); err == nil && v > 0 {
		headDim = v
	}

	vocab, err := vocabSize(r, emb)
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
		if v, err := getInt("attention.sliding_window"); err == nil && v > 0 {
			cfg.SlidingWindow = v
		}

		cfg.SWAPeriod = 2
		if v, err := getInt("attention.sliding_window_pattern"); err == nil && v > 0 {
			cfg.SWAPeriod = v
		}

		cfg.RopeFreqBaseSWA = freqBase
		if v, err := format.MetaValue[float32](r.Metadata, prefix+"rope.freq_base_swa"); err == nil && v > 0 {
			cfg.RopeFreqBaseSWA = v
		}

		cfg.AttnLogitSoftcap = 50
		if v, err := format.MetaValue[float32](r.Metadata, prefix+"attn_logit_softcapping"); err == nil && v > 0 {
			cfg.AttnLogitSoftcap = v
		}

		cfg.FinalLogitSoftcap = 30
		if v, err := format.MetaValue[float32](r.Metadata, prefix+"final_logit_softcapping"); err == nil && v > 0 {
			cfg.FinalLogitSoftcap = v
		}
	}

	return cfg, nil
}

// IsSWALayer: gemma2, period=2 -> чётные слои SWA (как llama.cpp set_swa_pattern)
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

func vocabSize(r *format.Reader, emb int) (int, error) {
	info, err := r.TensorInfo("token_embd.weight")
	if err != nil {
		return 0, err
	}

	if len(info.Dimensions) != 2 {
		return 0, fmt.Errorf("gemma: token_embd.weight: ожидается 2D")
	}

	a, b := int(info.Dimensions[0]), int(info.Dimensions[1])
	if a == emb {
		return b, nil
	}

	if b == emb {
		return a, nil
	}

	return 0, fmt.Errorf("gemma: token_embd.weight %v не содержит embedding_length=%d", info.Dimensions, emb)
}
