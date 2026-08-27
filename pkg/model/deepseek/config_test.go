package deepseek_test

import (
	"testing"

	"github.com/magomedcoder/gogguf/pkg/format"
	"github.com/magomedcoder/gogguf/pkg/model/deepseek"
)

func TestParseConfigMoE(t *testing.T) {
	r := &format.Reader{
		Metadata: format.Metadata{
			"deepseek.context_length":                   int32(4096),
			"deepseek.embedding_length":                 int32(2048),
			"deepseek.feed_forward_length":              int32(10944),
			"deepseek.block_count":                      int32(28),
			"deepseek.attention.head_count":             int32(16),
			"deepseek.attention.head_count_kv":          int32(16),
			"deepseek.attention.layer_norm_rms_epsilon": float32(1e-6),
			"deepseek.rope.freq_base":                   float32(10000),
			"deepseek.leading_dense_block_count":        int32(1),
			"deepseek.expert_count":                     int32(64),
			"deepseek.expert_used_count":                int32(6),
			"deepseek.expert_shared_count":              int32(2),
			"deepseek.expert_feed_forward_length":       int32(1408),
			"deepseek.expert_weights_scale":             float32(1),
		},
		Tensors: []format.TensorInfo{
			{
				Name:       "token_embd.weight",
				Dimensions: []uint64{2048, 102400},
			},
		},
	}

	cfg, err := deepseek.ParseConfig(r)
	if err != nil {
		t.Fatal(err)
	}

	if cfg.DenseLeadLayers != 1 {
		t.Fatalf("DenseLeadLayers=%d", cfg.DenseLeadLayers)
	}

	if cfg.ExpertCount != 64 || cfg.ExpertUsedCount != 6 {
		t.Fatalf("experts=%d used=%d", cfg.ExpertCount, cfg.ExpertUsedCount)
	}

	if cfg.ExpertShared != 2 || cfg.ExpertFFN != 1408 {
		t.Fatalf("shared=%d ffn=%d", cfg.ExpertShared, cfg.ExpertFFN)
	}

	if cfg.HeadDim != 128 {
		t.Fatalf("HeadDim=%d", cfg.HeadDim)
	}

	if cfg.VocabSize != 102400 {
		t.Fatalf("VocabSize=%d", cfg.VocabSize)
	}
}

func TestParseConfigRejectsBadMoE(t *testing.T) {
	r := &format.Reader{
		Metadata: format.Metadata{
			"deepseek.context_length":             int32(4096),
			"deepseek.embedding_length":           int32(2048),
			"deepseek.feed_forward_length":        int32(10944),
			"deepseek.block_count":                int32(28),
			"deepseek.attention.head_count":       int32(16),
			"deepseek.attention.head_count_kv":    int32(16),
			"deepseek.leading_dense_block_count":  int32(1),
			"deepseek.expert_count":               int32(0),
			"deepseek.expert_used_count":          int32(6),
			"deepseek.expert_feed_forward_length": int32(1408),
		},
		Tensors: []format.TensorInfo{
			{
				Name:       "token_embd.weight",
				Dimensions: []uint64{2048, 102400},
			},
		},
	}
	if _, err := deepseek.ParseConfig(r); err == nil {
		t.Fatal("ожидали ошибку при expert_count=0 и MoE-слоях")
	}
}
