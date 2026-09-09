package deepseek2

import (
	"fmt"
	"math"

	"github.com/magomedcoder/gogguf/pkg/format"
	"github.com/magomedcoder/gogguf/pkg/ops"
)

// Config - DeepSeek-V2/V3 (MLA + MoE) из метаданных deepseek2.*
type Config struct {
	ContextLength     int
	EmbeddingDim      int
	FFNHidden         int
	NumLayers         int
	NumHeads          int
	VocabSize         int
	RMSNormEps        float32
	RopeFreqBase      float32
	RopeDim           int // qk_rope / n_rot
	QKNopeDim         int
	VHeadDim          int // n_embd_head_v_mla
	QLoraRank         int // 0 = lite (attn_q.weight)
	KVLoraRank        int
	DenseLeadLayers   int
	ExpertCount       int
	ExpertUsedCount   int
	ExpertShared      int
	ExpertFFN         int
	ExpertWeightScale float32
	ExpertWeightsNorm bool
	// YaRN-масштабирование RoPE
	RopeFreqScale  float32
	RopeAttnFactor float32
	RopeYarnLogMul float32
	AttnScale      float32 // заранее посчитанный kq_scale
}

// ParseConfig читает deepseek2.*
func ParseConfig(r *format.Reader) (Config, error) {
	prefix := "deepseek2."
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

	if heads <= 0 {
		return Config{}, fmt.Errorf("deepseek2: head_count=%d", heads)
	}

	eps := float32(1e-6)
	if v, err := format.MetaValue[float32](r.Metadata, prefix+"attention.layer_norm_rms_epsilon"); err == nil {
		eps = v
	}

	freqBase := float32(10000)
	if v, err := format.MetaValue[float32](r.Metadata, prefix+"rope.freq_base"); err == nil {
		freqBase = v
	}

	kvLora, err := getInt("attention.kv_lora_rank")
	if err != nil || kvLora <= 0 {
		return Config{}, fmt.Errorf("deepseek2: attention.kv_lora_rank required")
	}

	qLora := 0
	if v, err := getInt("attention.q_lora_rank"); err == nil {
		qLora = v
	}

	ropeDim := 64
	if v, err := getInt("rope.dimension_count"); err == nil && v > 0 {
		ropeDim = v
	}

	headKMLa := 0
	if v, err := getInt("attention.key_length_mla"); err == nil && v > 0 {
		headKMLa = v
	} else if v, err := getInt("attention.key_length"); err == nil && v > 0 {
		headKMLa = v
	}

	if headKMLa <= 0 {
		headKMLa = emb / heads
	}

	qkNope := headKMLa - ropeDim
	if qkNope < 1 {
		return Config{}, fmt.Errorf("deepseek2: qk_nope=%d (head_k=%d rope=%d)", qkNope, headKMLa, ropeDim)
	}

	vHead := qkNope
	if v, err := getInt("attention.value_length_mla"); err == nil && v > 0 {
		vHead = v
	} else if v, err := getInt("attention.value_length"); err == nil && v > 0 {
		vHead = v
	}

	denseLead := 0
	if v, err := getInt("leading_dense_block_count"); err == nil && v >= 0 {
		denseLead = v
	}

	if denseLead > layers {
		return Config{}, fmt.Errorf("deepseek2: dense_lead=%d > layers=%d", denseLead, layers)
	}

	nExpert := 0
	if v, err := getInt("expert_count"); err == nil {
		nExpert = v
	}

	nUsed := 0
	if v, err := getInt("expert_used_count"); err == nil {
		nUsed = v
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

	normW := false
	if v, err := format.MetaValue[bool](r.Metadata, prefix+"expert_weights_norm"); err == nil {
		normW = v
	} else if v, err := getInt("expert_weights_norm"); err == nil {
		normW = v != 0
	}

	if denseLead < layers {
		if nExpert <= 0 || nUsed <= 0 || expertFFN <= 0 {
			return Config{}, fmt.Errorf("deepseek2: MoE требует expert_count/used/ffn")
		}
	}

	freqScale := float32(1)
	attnFactor := float32(1)
	yarnLogMul := float32(0)
	if v, err := format.MetaValue[float32](r.Metadata, prefix+"rope.scaling.factor"); err == nil && v > 0 {
		// GGUF factor = ctx_new/ctx_orig; в ggml обычно freq_scale = 1/factor
		freqScale = 1 / v
	}

	if v, err := format.MetaValue[float32](r.Metadata, prefix+"rope.scaling.freq_scale"); err == nil && v > 0 {
		freqScale = v
	}

	if v, err := format.MetaValue[float32](r.Metadata, prefix+"rope.scaling.attn_factor"); err == nil && v > 0 {
		attnFactor = v
	}

	if v, err := format.MetaValue[float32](r.Metadata, prefix+"rope.scaling.yarn_log_mul"); err == nil {
		// компенсируем множитель скрипта конвертации (TAG_DEEPSEEK2_YARN_LOG_MUL_FIX)
		yarnLogMul = v / 0.1
	}

	attnScale := yarnKQScale(headKMLa, freqScale, attnFactor, yarnLogMul)

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
		VocabSize:         vocab,
		RMSNormEps:        eps,
		RopeFreqBase:      freqBase,
		RopeDim:           ropeDim,
		QKNopeDim:         qkNope,
		VHeadDim:          vHead,
		QLoraRank:         qLora,
		KVLoraRank:        kvLora,
		DenseLeadLayers:   denseLead,
		ExpertCount:       nExpert,
		ExpertUsedCount:   nUsed,
		ExpertShared:      nShared,
		ExpertFFN:         expertFFN,
		ExpertWeightScale: wScale,
		ExpertWeightsNorm: normW,
		RopeFreqScale:     freqScale,
		RopeAttnFactor:    attnFactor,
		RopeYarnLogMul:    yarnLogMul,
		AttnScale:         attnScale,
	}, nil
}

func yarnKQScale(headK int, freqScale, attnFactor, yarnLogMul float32) float32 {
	if freqScale <= 0 {
		freqScale = 1
	}

	if attnFactor <= 0 {
		attnFactor = 1
	}
	// отмена yarn_attn_factor_adjust: attn_factor_org = attn_factor * (1 + 0.1*log(1/freq_scale)) затем mscale = attn_factor_org * (1 + 0.1*yarn_log_mul*log(1/freq_scale))
	inv := float64(1 / freqScale)
	logInv := math.Log(inv)
	attnOrg := float64(attnFactor) * (1 + 0.1*logInv)
	mscale := attnOrg * (1 + 0.1*float64(yarnLogMul)*logInv)
	return float32(mscale * mscale / math.Sqrt(float64(headK)))
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

func (c Config) qkDim() int {
	return c.KVLoraRank + c.RopeDim
}

func (c Config) ropeScale() ops.RoPEScale {
	return ops.RoPEScale{
		FreqScale:  c.RopeFreqScale,
		AttnFactor: c.RopeAttnFactor,
	}
}

func vocabSize(r *format.Reader, emb int) (int, error) {
	info, err := r.TensorInfo("token_embd.weight")
	if err != nil {
		return 0, err
	}

	if len(info.Dimensions) != 2 {
		return 0, fmt.Errorf("deepseek2: token_embd.weight: ожидается 2D")
	}

	a, b := int(info.Dimensions[0]), int(info.Dimensions[1])
	if a == emb {
		return b, nil
	}
	
	if b == emb {
		return a, nil
	}

	return 0, fmt.Errorf("deepseek2: token_embd %v без emb=%d", info.Dimensions, emb)
}
