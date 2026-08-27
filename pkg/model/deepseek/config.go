package deepseek

import (
	"fmt"

	"github.com/magomedcoder/gogguf/pkg/format"
)

// Config - гиперпараметры DeepSeek / DeepSeek-MoE из метаданных GGUF
type Config struct {
	ContextLength     int
	EmbeddingDim      int
	FFNHidden         int // dense lead FFN
	NumLayers         int
	NumHeads          int
	NumKVHeads        int
	HeadDim           int
	VocabSize         int
	RMSNormEps        float32
	RopeFreqBase      float32
	DenseLeadLayers   int     // leading_dense_block_count; слои [0, DenseLead) - dense
	ExpertCount       int     // expert_count
	ExpertUsedCount   int     // expert_used_count (top-k)
	ExpertShared      int     // expert_shared_count
	ExpertFFN         int     // expert_feed_forward_length
	ExpertWeightScale float32 // expert_weights_scale; 0 -> 1
}

// ParseConfig читает deepseek.*
func ParseConfig(r *format.Reader) (Config, error) {
	prefix := "deepseek."
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

	freqBase := float32(10000)
	if v, err := format.MetaValue[float32](r.Metadata, prefix+"rope.freq_base"); err == nil {
		freqBase = v
	}

	if heads <= 0 {
		return Config{}, fmt.Errorf("deepseek: attention.head_count=%d", heads)
	}

	headDim := emb / heads
	if v, err := getInt("attention.key_length"); err == nil && v > 0 {
		headDim = v
	}

	denseLead := 0
	if v, err := getInt("leading_dense_block_count"); err == nil && v >= 0 {
		denseLead = v
	}

	if denseLead > layers {
		return Config{}, fmt.Errorf("deepseek: leading_dense_block_count=%d > block_count=%d", denseLead, layers)
	}

	nExpert := 0
	if v, err := getInt("expert_count"); err == nil {
		nExpert = v
	}

	nExpertUsed := 0
	if v, err := getInt("expert_used_count"); err == nil {
		nExpertUsed = v
	}

	nShared := 0
	if v, err := getInt("expert_shared_count"); err == nil {
		nShared = v
	}

	expertFFN := 0
	if v, err := getInt("expert_feed_forward_length"); err == nil {
		expertFFN = v
	}

	wScale := float32(1)
	if v, err := format.MetaValue[float32](r.Metadata, prefix+"expert_weights_scale"); err == nil && v != 0 {
		wScale = v
	}

	if denseLead < layers {
		if nExpert <= 0 || nExpertUsed <= 0 || expertFFN <= 0 {
			return Config{}, fmt.Errorf("deepseek: MoE слои требуют expert_count/used/ffn (lead=%d layers=%d)", denseLead, layers)
		}

		if nExpertUsed > nExpert {
			return Config{}, fmt.Errorf("deepseek: expert_used_count=%d > expert_count=%d", nExpertUsed, nExpert)
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
		DenseLeadLayers:   denseLead,
		ExpertCount:       nExpert,
		ExpertUsedCount:   nExpertUsed,
		ExpertShared:      nShared,
		ExpertFFN:         expertFFN,
		ExpertWeightScale: wScale,
	}, nil
}

func (c Config) isMoELayer(layer int) bool {
	return layer >= c.DenseLeadLayers && c.ExpertCount > 0
}

func (c Config) sharedFFN() int {
	if c.ExpertShared <= 0 {
		return c.ExpertFFN
	}

	return c.ExpertFFN * c.ExpertShared
}

func (c Config) maxFFN() int {
	m := c.FFNHidden
	if s := c.sharedFFN(); s > m {
		m = s
	}

	if c.ExpertFFN > m {
		m = c.ExpertFFN
	}

	return m
}

func vocabSize(r *format.Reader, emb int) (int, error) {
	info, err := r.TensorInfo("token_embd.weight")
	if err != nil {
		return 0, err
	}

	if len(info.Dimensions) != 2 {
		return 0, fmt.Errorf("deepseek: token_embd.weight: ожидается 2D")
	}

	a, b := int(info.Dimensions[0]), int(info.Dimensions[1])
	if a == emb {
		return b, nil
	}

	if b == emb {
		return a, nil
	}

	return 0, fmt.Errorf("deepseek: token_embd.weight %v не содержит embedding_length=%d", info.Dimensions, emb)
}
