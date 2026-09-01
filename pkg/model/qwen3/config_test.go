package qwen3_test

import (
	"testing"

	"github.com/magomedcoder/gogguf/pkg/format"
	"github.com/magomedcoder/gogguf/pkg/model/qwen3"
)

func TestParseConfigHeadDim(t *testing.T) {
	r := &format.Reader{
		Metadata: format.Metadata{
			"qwen3.context_length":                   int32(4096),
			"qwen3.embedding_length":                 int32(1024),
			"qwen3.feed_forward_length":              int32(3072),
			"qwen3.block_count":                      int32(28),
			"qwen3.attention.head_count":             int32(16),
			"qwen3.attention.head_count_kv":          int32(8),
			"qwen3.attention.key_length":             int32(128),
			"qwen3.attention.layer_norm_rms_epsilon": float32(1e-6),
			"qwen3.rope.freq_base":                   float32(1e6),
		},
		Tensors: []format.TensorInfo{
			{
				Name:       "token_embd.weight",
				Dimensions: []uint64{1024, 151936},
			},
		},
	}

	cfg, err := qwen3.ParseConfig(r)
	if err != nil {
		t.Fatal(err)
	}

	if cfg.HeadDim != 128 {
		t.Fatalf("HeadDim = %d, want 128", cfg.HeadDim)
	}
}

func TestParseConfigMoE(t *testing.T) {
	r := &format.Reader{
		Metadata: format.Metadata{
			"qwen3moe.context_length":             int32(40960),
			"qwen3moe.embedding_length":           int32(2048),
			"qwen3moe.feed_forward_length":        int32(6144),
			"qwen3moe.block_count":                int32(48),
			"qwen3moe.attention.head_count":       int32(32),
			"qwen3moe.attention.head_count_kv":    int32(4),
			"qwen3moe.attention.key_length":       int32(128),
			"qwen3moe.expert_count":               int32(128),
			"qwen3moe.expert_used_count":          int32(8),
			"qwen3moe.expert_feed_forward_length": int32(768),
		},
		Tensors: []format.TensorInfo{
			{
				Name:       "token_embd.weight",
				Dimensions: []uint64{2048, 151936},
			},
		},
	}

	cfg, err := qwen3.ParseConfigMoE(r)
	if err != nil {
		t.Fatal(err)
	}

	if cfg.ExpertCount != 128 || cfg.ExpertUsedCount != 8 {
		t.Fatalf("experts=%d used=%d", cfg.ExpertCount, cfg.ExpertUsedCount)
	}

	if cfg.ExpertFFN != 768 {
		t.Fatalf("ExpertFFN=%d", cfg.ExpertFFN)
	}

	if !cfg.MoENormWeights {
		t.Fatal("MoENormWeights expected true")
	}
}
