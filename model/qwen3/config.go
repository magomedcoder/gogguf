package qwen3

import (
	"fmt"

	"github.com/magomedcoder/gguf.go"
)

// Config - гиперпараметры Qwen3 из метаданных GGUF
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

// ParseConfig читает конфиг из метаданных GGUF
func ParseConfig(r *gguf.Reader) (Config, error) {
	prefix := "qwen3."
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
	if v, err := gguf.MetaValue[float32](r.Metadata, prefix+"attention.layer_norm_rms_epsilon"); err == nil {
		eps = v
	}

	freqBase := float32(1000000)
	if v, err := gguf.MetaValue[float32](r.Metadata, prefix+"rope.freq_base"); err == nil {
		freqBase = v
	}

	headDim := emb / heads
	if headDim*heads != emb {
		return Config{}, fmt.Errorf("qwen3: embedding_length не делится на head_count")
	}

	vocab, err := vocabSize(r, emb)
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
		VocabSize:     vocab,
		RMSNormEps:    eps,
		RopeFreqBase:  freqBase,
	}, nil
}

func vocabSize(r *gguf.Reader, emb int) (int, error) {
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
