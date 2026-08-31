package mistral

import (
	"fmt"

	"github.com/magomedcoder/gogguf/pkg/format"
)

// Config - гиперпараметры Mistral из метаданных GGUF
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
	SlidingWindow     int // 0 = без ограничения (полный KV-cache)
	ExpertCount       int
	ExpertUsedCount   int
	ExpertFFN         int // expert_feed_forward_length; 0 -> FFNHidden
	SharedFFN         int // expert_shared_feed_forward_length; 0 -> none
	ExpertWeightScale float32
	MoENormWeights    bool // Mixtral renorm top-k; Qwen2MoE/DeepSeek - false
	SharedExpertGate  bool // Qwen2MoE: silu(gate_inp)/gate_inp на shared FFN
}

// ParseConfig читает конфиг из метаданных GGUF (префикс mistral.)
func ParseConfig(r *format.Reader) (Config, error) {
	return parseConfigWithPrefix(r, "mistral.", 1e-5, 1000000)
}

// ParseConfigLlama читает конфиг Mistral-модели с префиксом llama.* (TheBloke и convert.py)
func ParseConfigLlama(r *format.Reader) (Config, error) {
	return parseConfigWithPrefix(r, "llama.", 1e-5, 1000000)
}

// ParseConfigQwen2 читает qwen2.* (DeepSeek distill и др.)
func ParseConfigQwen2(r *format.Reader) (Config, error) {
	return parseConfigWithPrefix(r, "qwen2.", 1e-6, 10000)
}

// ParseConfigQwen2MoE читает qwen2moe.* (MoE + gated shared expert)
func ParseConfigQwen2MoE(r *format.Reader) (Config, error) {
	cfg, err := parseConfigWithPrefix(r, "qwen2moe.", 1e-6, 10000)
	if err != nil {
		return Config{}, err
	}

	if cfg.ExpertCount <= 0 {
		return Config{}, fmt.Errorf("qwen2moe: expert_count=%d", cfg.ExpertCount)
	}

	cfg.MoENormWeights = false
	cfg.SharedExpertGate = true
	if cfg.ExpertFFN <= 0 && cfg.ExpertUsedCount > 0 {
		cfg.ExpertFFN = cfg.FFNHidden / cfg.ExpertUsedCount
	}

	if cfg.SharedFFN <= 0 {
		cfg.SharedFFN = cfg.FFNHidden
	}

	return cfg, nil
}

func parseConfigWithPrefix(r *format.Reader, prefix string, defaultEps, defaultRope float32) (Config, error) {
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

	eps := defaultEps
	if v, err := format.MetaValue[float32](r.Metadata, prefix+"attention.layer_norm_rms_epsilon"); err == nil {
		eps = v
	}

	freqBase := defaultRope
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

	nExpert := 0
	if v, err := getInt("expert_count"); err == nil {
		nExpert = v
	}

	nExpertUsed := 0
	if v, err := getInt("expert_used_count"); err == nil {
		nExpertUsed = v
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
			return Config{}, fmt.Errorf("mistral: expert_used_count=%d > expert_count=%d", nExpertUsed, nExpert)
		}
	}

	expertFFN := 0
	if v, err := getInt("expert_feed_forward_length"); err == nil && v > 0 {
		expertFFN = v
	}

	sharedFFN := 0
	if v, err := getInt("expert_shared_feed_forward_length"); err == nil && v > 0 {
		sharedFFN = v
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
		SlidingWindow:     slidingWindow,
		ExpertCount:       nExpert,
		ExpertUsedCount:   nExpertUsed,
		ExpertFFN:         expertFFN,
		SharedFFN:         sharedFFN,
		ExpertWeightScale: wScale,
		MoENormWeights:    nExpert > 0,
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

	if c.SharedFFN > m {
		m = c.SharedFFN
	}

	return m
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
