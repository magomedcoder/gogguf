package deepseek2_test

import (
	"math"
	"testing"

	"github.com/magomedcoder/gogguf/pkg/format"
	"github.com/magomedcoder/gogguf/pkg/model/deepseek2"
)

func TestParseConfigMLA(t *testing.T) {
	r := &format.Reader{
		Metadata: format.Metadata{
			"deepseek2.context_length":                   int32(163840),
			"deepseek2.embedding_length":                 int32(2048),
			"deepseek2.feed_forward_length":              int32(10944),
			"deepseek2.block_count":                      int32(27),
			"deepseek2.attention.head_count":             int32(16),
			"deepseek2.attention.kv_lora_rank":           int32(512),
			"deepseek2.attention.key_length_mla":         int32(192),
			"deepseek2.attention.value_length_mla":       int32(128),
			"deepseek2.rope.dimension_count":             int32(64),
			"deepseek2.attention.layer_norm_rms_epsilon": float32(1e-6),
			"deepseek2.rope.freq_base":                   float32(10000),
			"deepseek2.leading_dense_block_count":        int32(1),
			"deepseek2.expert_count":                     int32(64),
			"deepseek2.expert_used_count":                int32(6),
			"deepseek2.expert_shared_count":              int32(2),
			"deepseek2.expert_feed_forward_length":       int32(1408),
			"deepseek2.rope.scaling.factor":              float32(40),
			"deepseek2.rope.scaling.attn_factor":         float32(1),
			"deepseek2.rope.scaling.yarn_log_mul":        float32(0.1),
		},
		Tensors: []format.TensorInfo{
			{
				Name:       "token_embd.weight",
				Dimensions: []uint64{2048, 102400},
			},
		},
	}

	cfg, err := deepseek2.ParseConfig(r)
	if err != nil {
		t.Fatal(err)
	}

	if cfg.QKNopeDim != 128 { // 192-64
		t.Fatalf("QKNopeDim=%d", cfg.QKNopeDim)
	}

	if cfg.KVLoraRank != 512 || cfg.VHeadDim != 128 {
		t.Fatalf("kv=%d v=%d", cfg.KVLoraRank, cfg.VHeadDim)
	}

	if cfg.QLoraRank != 0 {
		t.Fatalf("for lite expected QLora=0, got %d", cfg.QLoraRank)
	}

	if cfg.DenseLeadLayers != 1 || cfg.ExpertCount != 64 {
		t.Fatalf("lead=%d experts=%d", cfg.DenseLeadLayers, cfg.ExpertCount)
	}

	if cfg.AttnScale <= 0 || math.IsNaN(float64(cfg.AttnScale)) {
		t.Fatalf("AttnScale=%v", cfg.AttnScale)
	}

	// with YaRN factor=40 scale must differ from usual 1/sqrt(192)
	plain := float32(1 / math.Sqrt(192))
	if math.Abs(float64(cfg.AttnScale-plain)) < 1e-6 {
		t.Fatalf("expected YaRN scale ≠ plain, got %v", cfg.AttnScale)
	}
}

func TestParseConfigRequiresKVLora(t *testing.T) {
	r := &format.Reader{
		Metadata: format.Metadata{
			"deepseek2.context_length":            int32(4096),
			"deepseek2.embedding_length":          int32(2048),
			"deepseek2.feed_forward_length":       int32(10944),
			"deepseek2.block_count":               int32(27),
			"deepseek2.attention.head_count":      int32(16),
			"deepseek2.leading_dense_block_count": int32(27),
		},
		Tensors: []format.TensorInfo{
			{
				Name:       "token_embd.weight",
				Dimensions: []uint64{2048, 102400},
			},
		},
	}

	if _, err := deepseek2.ParseConfig(r); err == nil {
		t.Fatal("expected error without kv_lora_rank")
	}
}
