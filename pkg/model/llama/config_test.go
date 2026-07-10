package llama_test

import (
	"testing"

	"github.com/magomedcoder/gogguf/pkg/format"
	"github.com/magomedcoder/gogguf/pkg/model/llama"
)

func TestParseConfigLlama3(t *testing.T) {
	r := &format.Reader{
		Metadata: format.Metadata{
			"llama.context_length":                   int32(8192),
			"llama.embedding_length":                 int32(4096),
			"llama.feed_forward_length":              int32(14336),
			"llama.block_count":                      int32(32),
			"llama.attention.head_count":             int32(32),
			"llama.attention.head_count_kv":          int32(8),
			"llama.attention.layer_norm_rms_epsilon": float32(1e-5),
			"llama.rope.freq_base":                   float32(500000),
		},
		Tensors: []format.TensorInfo{
			{
				Name:       "token_embd.weight",
				Dimensions: []uint64{4096, 128256},
			},
		},
	}

	cfg, err := llama.ParseConfig(r)
	if err != nil {
		t.Fatal(err)
	}

	if cfg.HeadDim != 128 {
		t.Fatalf("HeadDim = %d, want 128", cfg.HeadDim)
	}

	if cfg.RopeFreqBase != 500000 {
		t.Fatalf("RopeFreqBase = %v, want 500000", cfg.RopeFreqBase)
	}
}
