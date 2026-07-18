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
