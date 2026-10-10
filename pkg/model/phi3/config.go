package phi3

import (
	"fmt"

	"github.com/magomedcoder/gogguf/pkg/format"
)

// Config - Phi-3 / Phi-3.5 hyperparameters from GGUF metadata
type Config struct {
	ContextLength  int
	EmbeddingDim   int
	FFNHidden      int
	NumLayers      int
	NumHeads       int
	NumKVHeads     int
	HeadDim        int
	RopeDim        int // partial rotary; 0 = HeadDim
	VocabSize      int
	RMSNormEps     float32
	RopeFreqBase   float32
	OrigCtxLen     int     // rope.scaling.original_context_length; 0 -> ContextLength
	RopeFreqScale  float32 // 1/scaling.factor; 0 -> 1
	RopeAttnFactor float32 // rope.scaling.attn_factor; 0 -> 1
}

// ParseConfig reads config from GGUF metadata (phi3. prefix)
func ParseConfig(r *format.Reader) (Config, error) {
	prefix := "phi3."
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

	freqBase := float32(10000)
	if v, err := format.MetaValue[float32](r.Metadata, prefix+"rope.freq_base"); err == nil {
		freqBase = v
	}

	if heads <= 0 {
		return Config{}, fmt.Errorf("phi3: attention.head_count=%d", heads)
	}

	headDim := emb / heads
	if v, err := getInt("attention.key_length"); err == nil && v > 0 {
		headDim = v
	}

	ropeDim := headDim
	if v, err := getInt("rope.dimension_count"); err == nil && v > 0 {
		ropeDim = v
	}

	vocab, err := vocabSize(r, emb)
	if err != nil {
		return Config{}, err
	}

	origCtx := ctx
	if v, err := getInt("rope.scaling.original_context_length"); err == nil && v > 0 {
		origCtx = v
	}

	freqScale := float32(1)
	if v, err := format.MetaValue[float32](r.Metadata, prefix+"rope.scaling.factor"); err == nil && v > 0 {
		freqScale = 1 / v
	}

	if v, err := format.MetaValue[float32](r.Metadata, prefix+"rope.freq_scale"); err == nil && v > 0 {
		freqScale = v
	}

	attnFactor := float32(1)
	if v, err := format.MetaValue[float32](r.Metadata, prefix+"rope.scaling.attn_factor"); err == nil && v > 0 {
		attnFactor = v
	}

	return Config{
		ContextLength:  ctx,
		EmbeddingDim:   emb,
		FFNHidden:      ffn,
		NumLayers:      layers,
		NumHeads:       heads,
		NumKVHeads:     kvHeads,
		HeadDim:        headDim,
		RopeDim:        ropeDim,
		VocabSize:      vocab,
		RMSNormEps:     eps,
		RopeFreqBase:   freqBase,
		OrigCtxLen:     origCtx,
		RopeFreqScale:  freqScale,
		RopeAttnFactor: attnFactor,
	}, nil
}

func vocabSize(r *format.Reader, emb int) (int, error) {
	info, err := r.TensorInfo("token_embd.weight")
	if err != nil {
		return 0, err
	}

	if len(info.Dimensions) != 2 {
		return 0, fmt.Errorf("phi3: token_embd.weight: expected 2D")
	}

	a, b := int(info.Dimensions[0]), int(info.Dimensions[1])
	if a == emb {
		return b, nil
	}

	if b == emb {
		return a, nil
	}

	return 0, fmt.Errorf("phi3: token_embd.weight %v does not contain embedding_length=%d", info.Dimensions, emb)
}
