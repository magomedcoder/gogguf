package mistral

import (
	"fmt"

	"github.com/magomedcoder/gogguf/pkg/format"
)

// Config - Mistral hyperparameters from GGUF metadata
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
	SlidingWindow     int // 0 = unlimited (full KV-cache)
	ExpertCount       int
	ExpertUsedCount   int
	ExpertFFN         int // expert_feed_forward_length; 0 -> FFNHidden
	SharedFFN         int // expert_shared_feed_forward_length; 0 -> none
	ExpertWeightScale float32
	MoENormWeights    bool // Mixtral renorm top-k; Qwen2MoE/DeepSeek - false
	SharedExpertGate  bool // Qwen2MoE: silu(gate_inp)/gate_inp on shared FFN
}

// defaultMistralSWA is the Mistral 7B / Mixtral sliding window when GGUF omits the key (common for TheBloke llama.*-prefixed exports).
const defaultMistralSWA = 4096

// ParseConfig reads config from GGUF metadata (mistral. prefix)
func ParseConfig(r *format.Reader) (Config, error) {
	return parseConfigWithPrefix(r, "mistral.", 1e-5, 1000000, defaultMistralSWA)
}

// ParseConfigLlama reads Mistral model config with llama.* prefix (TheBloke and convert.py)
func ParseConfigLlama(r *format.Reader) (Config, error) {
	return parseConfigWithPrefix(r, "llama.", 1e-5, 1000000, defaultMistralSWA)
}

// ParseConfigQwen2 reads qwen2.* (DeepSeek distill, etc.)
func ParseConfigQwen2(r *format.Reader) (Config, error) {
	return parseConfigWithPrefix(r, "qwen2.", 1e-6, 10000, 0)
}

// ParseConfigQwen2MoE reads qwen2moe.* (MoE + gated shared expert)
func ParseConfigQwen2MoE(r *format.Reader) (Config, error) {
	cfg, err := parseConfigWithPrefix(r, "qwen2moe.", 1e-6, 10000, 0)
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

func parseConfigWithPrefix(r *format.Reader, prefix string, defaultEps, defaultRope float32, defaultSlidingWindow int) (Config, error) {
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

	eps := meta.Float32Or("attention.layer_norm_rms_epsilon", defaultEps)
	freqBase := meta.Float32Or("rope.freq_base", defaultRope)

	headDim := emb / heads
	if heads <= 0 {
		return Config{}, fmt.Errorf("mistral: attention.head_count=%d", heads)
	}

	if v, err := meta.Int("attention.key_length"); err == nil && v > 0 {
		headDim = v
	}

	slidingWindow := defaultSlidingWindow
	if v, err := meta.Int("attention.sliding_window"); err == nil && v > 0 {
		slidingWindow = v
	}

	nExpert := meta.IntOr("expert_count", 0)
	nExpertUsed := meta.IntOr("expert_used_count", 0)

	wScale := meta.Float32Or("expert_weights_scale", 1)
	if wScale == 0 {
		wScale = 1
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
	if v, err := meta.Int("expert_feed_forward_length"); err == nil && v > 0 {
		expertFFN = v
	}

	sharedFFN := 0
	if v, err := meta.Int("expert_shared_feed_forward_length"); err == nil && v > 0 {
		sharedFFN = v
	}

	vocab, err := format.VocabSizeFromEmb(r, emb, "mistral")
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
