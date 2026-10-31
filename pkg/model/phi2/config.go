package phi2

import (
	"fmt"

	"github.com/magomedcoder/gogguf/pkg/format"
)

// Config - Phi-2 hyperparameters from GGUF metadata
type Config struct {
	ContextLength int
	EmbeddingDim  int
	FFNHidden     int
	NumLayers     int
	NumHeads      int
	NumKVHeads    int
	HeadDim       int
	RopeDim       int // partial rotary; 0 -> HeadDim
	VocabSize     int
	LayerNormEps  float32
	RopeFreqBase  float32
}

// ParseConfig reads config from GGUF metadata (phi2. prefix)
func ParseConfig(r *format.Reader) (Config, error) {
	meta := format.PrefixedMeta{Prefix: "phi2.", MD: r.Metadata}

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

	eps := meta.Float32Or("attention.layer_norm_epsilon", 1e-5)
	freqBase := meta.Float32Or("rope.freq_base", 10000)

	if heads <= 0 {
		return Config{}, fmt.Errorf("phi2: attention.head_count=%d", heads)
	}

	headDim := emb / heads
	if v, err := meta.Int("attention.key_length"); err == nil && v > 0 {
		headDim = v
	}

	ropeDim := headDim
	if v, err := meta.Int("rope.dimension_count"); err == nil && v > 0 {
		ropeDim = v
	}

	vocab, err := format.VocabSizeFromEmb(r, emb, "phi2")
	if err != nil {
		return Config{}, err
	}

	return Config{
		ContextLength: ctx,
		EmbeddingDim:  emb,
		FFNHidden:     ffn,
		NumLayers:     layers,
		NumHeads:      heads,
		NumKVHeads:    kvHeads,
		HeadDim:       headDim,
		RopeDim:       ropeDim,
		VocabSize:     vocab,
		LayerNormEps:  eps,
		RopeFreqBase:  freqBase,
	}, nil
}
