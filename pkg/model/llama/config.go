package llama

import (
	"fmt"

	"github.com/magomedcoder/gogguf/pkg/format"
)

// Config - Llama hyperparameters from GGUF metadata
type Config struct {
	ContextLength int
	EmbeddingDim  int
	FFNHidden     int
	NumLayers     int
	NumHeads      int
	NumKVHeads    int
	HeadDim       int
	VocabSize     int
	RMSNormEps    float32
	RopeFreqBase  float32
}

// ParseConfig reads config from GGUF metadata (llama. prefix)
func ParseConfig(r *format.Reader) (Config, error) {
	meta := format.PrefixedMeta{Prefix: "llama.", MD: r.Metadata}

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

	eps := meta.Float32Or("attention.layer_norm_rms_epsilon", 1e-5)

	headDim := emb / heads
	if heads <= 0 {
		return Config{}, fmt.Errorf("llama: attention.head_count=%d", heads)
	}

	if v, err := meta.Int("attention.key_length"); err == nil && v > 0 {
		headDim = v
	}

	vocab, err := format.VocabSizeFromEmb(r, emb, "llama")
	if err != nil {
		return Config{}, err
	}

	// Llama 2 ≈ 32k vocab / rope 10k; Llama 3 ≈ 128k / 500k - when the key is missing
	freqBase := float32(10000)
	if vocab >= 100000 {
		freqBase = 500000
	}

	freqBase = meta.Float32Or("rope.freq_base", freqBase)

	return Config{
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
	}, nil
}
