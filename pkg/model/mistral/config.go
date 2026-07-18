package mistral

import (
	"fmt"

	"github.com/magomedcoder/gogguf/pkg/format"
)

// Config - гиперпараметры Mistral из метаданных GGUF
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
	SlidingWindow int // 0 = без ограничения (полный KV-cache)
}

// ParseConfig читает конфиг из метаданных GGUF (префикс mistral.)
func ParseConfig(r *format.Reader) (Config, error) {
	return parseConfigWithPrefix(r, "mistral.")
}

// ParseConfigLlama читает конфиг Mistral-модели с префиксом llama.* (TheBloke и convert.py)
func ParseConfigLlama(r *format.Reader) (Config, error) {
	return parseConfigWithPrefix(r, "llama.")
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

	eps := float32(1e-5)
	if v, err := format.MetaValue[float32](r.Metadata, prefix+"attention.layer_norm_rms_epsilon"); err == nil {
		eps = v
	}

	freqBase := float32(1000000)
	if v, err := format.MetaValue[float32](r.Metadata, prefix+"rope.freq_base"); err == nil {
		freqBase = v
	}

	headDim := emb / heads
	if heads <= 0 {
		return Config{}, fmt.Errorf("mistral: attention.head_count=%d", heads)
	}

	if v, err := getInt("attention.key_length"); err == nil && v > 0 {
		headDim = v
	}

	slidingWindow := 0
	if v, err := getInt("attention.sliding_window"); err == nil && v > 0 {
		slidingWindow = v
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
		SlidingWindow: slidingWindow,
	}, nil
}

func vocabSize(r *format.Reader, emb int) (int, error) {
	info, err := r.TensorInfo("token_embd.weight")
	if err != nil {
		return 0, err
	}

	if len(info.Dimensions) != 2 {
		return 0, fmt.Errorf("mistral: token_embd.weight: ожидается 2D")
	}

	a, b := int(info.Dimensions[0]), int(info.Dimensions[1])
	if a == emb {
		return b, nil
	}

	if b == emb {
		return a, nil
	}

	return 0, fmt.Errorf("mistral: token_embd.weight %v не содержит embedding_length=%d", info.Dimensions, emb)
}
