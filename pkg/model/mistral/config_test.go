package mistral_test

import (
	"testing"

	"github.com/magomedcoder/gogguf/pkg/format"
	"github.com/magomedcoder/gogguf/pkg/model/mistral"
)

func TestParseConfigMistral7B(t *testing.T) {
	r := &format.Reader{
		Metadata: format.Metadata{
			"mistral.context_length":                   int32(32768),
			"mistral.embedding_length":                 int32(4096),
			"mistral.feed_forward_length":              int32(14336),
			"mistral.block_count":                      int32(32),
			"mistral.attention.head_count":             int32(32),
			"mistral.attention.head_count_kv":          int32(8),
			"mistral.attention.layer_norm_rms_epsilon": float32(1e-5),
			"mistral.rope.freq_base":                   float32(1000000),
			"mistral.attention.sliding_window":         int32(4096),
		},
		Tensors: []format.TensorInfo{
			{
				Name:       "token_embd.weight",
				Dimensions: []uint64{4096, 32000},
			},
		},
	}

	cfg, err := mistral.ParseConfig(r)
	if err != nil {
		t.Fatal(err)
	}

	if cfg.HeadDim != 128 {
		t.Fatalf("HeadDim = %d, ожидали 128", cfg.HeadDim)
	}

	if cfg.RopeFreqBase != 1000000 {
		t.Fatalf("RopeFreqBase = %v, ожидали 1000000", cfg.RopeFreqBase)
	}

	if cfg.SlidingWindow != 4096 {
		t.Fatalf("SlidingWindow = %d, ожидали 4096", cfg.SlidingWindow)
	}
}

func TestParseConfigQwen2Defaults(t *testing.T) {
	r := &format.Reader{
		Metadata: format.Metadata{
			"qwen2.context_length":          int32(32768),
			"qwen2.embedding_length":        int32(3584),
			"qwen2.feed_forward_length":     int32(18944),
			"qwen2.block_count":             int32(28),
			"qwen2.attention.head_count":    int32(28),
			"qwen2.attention.head_count_kv": int32(4),
		},
		Tensors: []format.TensorInfo{
			{
				Name:       "token_embd.weight",
				Dimensions: []uint64{3584, 152064},
			},
		},
	}

	cfg, err := mistral.ParseConfigQwen2(r)
	if err != nil {
		t.Fatal(err)
	}

	if cfg.RopeFreqBase != 10000 {
		t.Fatalf("RopeFreqBase=%v, want 10000 default", cfg.RopeFreqBase)
	}

	if cfg.HeadDim != 128 {
		t.Fatalf("HeadDim=%d", cfg.HeadDim)
	}

	if cfg.VocabSize != 152064 {
		t.Fatalf("VocabSize=%d", cfg.VocabSize)
	}
}

func TestParseConfigLlamaMixtralMoE(t *testing.T) {
	r := &format.Reader{
		Metadata: format.Metadata{
			"llama.context_length":          int32(32768),
			"llama.embedding_length":        int32(4096),
			"llama.feed_forward_length":     int32(14336),
			"llama.block_count":             int32(32),
			"llama.attention.head_count":    int32(32),
			"llama.attention.head_count_kv": int32(8),
			"llama.expert_count":            int32(8),
			"llama.expert_used_count":       int32(2),
		},
		Tensors: []format.TensorInfo{
			{
				Name:       "token_embd.weight",
				Dimensions: []uint64{4096, 32000},
			},
		},
	}

	cfg, err := mistral.ParseConfigLlama(r)
	if err != nil {
		t.Fatal(err)
	}

	if cfg.ExpertCount != 8 || cfg.ExpertUsedCount != 2 {
		t.Fatalf("experts=%d used=%d", cfg.ExpertCount, cfg.ExpertUsedCount)
	}

	if !cfg.MoENormWeights {
		t.Fatal("Mixtral MoE должен renorm веса")
	}
}

func TestParseConfigQwen2MoE(t *testing.T) {
	r := &format.Reader{
		Metadata: format.Metadata{
			"qwen2moe.context_length":                    int32(32768),
			"qwen2moe.embedding_length":                  int32(2048),
			"qwen2moe.feed_forward_length":               int32(5632),
			"qwen2moe.block_count":                       int32(24),
			"qwen2moe.attention.head_count":              int32(16),
			"qwen2moe.attention.head_count_kv":           int32(16),
			"qwen2moe.expert_count":                      int32(60),
			"qwen2moe.expert_used_count":                 int32(4),
			"qwen2moe.expert_feed_forward_length":        int32(1408),
			"qwen2moe.expert_shared_feed_forward_length": int32(5632),
		},
		Tensors: []format.TensorInfo{
			{
				Name:       "token_embd.weight",
				Dimensions: []uint64{2048, 151936},
			},
		},
	}

	cfg, err := mistral.ParseConfigQwen2MoE(r)
	if err != nil {
		t.Fatal(err)
	}

	if cfg.ExpertCount != 60 || cfg.ExpertUsedCount != 4 {
		t.Fatalf("experts=%d used=%d", cfg.ExpertCount, cfg.ExpertUsedCount)
	}

	if cfg.ExpertFFN != 1408 || cfg.SharedFFN != 5632 {
		t.Fatalf("expertFFN=%d sharedFFN=%d", cfg.ExpertFFN, cfg.SharedFFN)
	}

	if cfg.MoENormWeights || !cfg.SharedExpertGate {
		t.Fatalf("norm=%v gate=%v", cfg.MoENormWeights, cfg.SharedExpertGate)
	}
}
