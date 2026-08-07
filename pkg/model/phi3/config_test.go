package phi3_test

import (
	"testing"

	"github.com/magomedcoder/gogguf/pkg/format"
	"github.com/magomedcoder/gogguf/pkg/model/phi3"
	"github.com/magomedcoder/gogguf/pkg/ops"
)

func TestParseConfigPhi3Mini(t *testing.T) {
	r := &format.Reader{
		Metadata: format.Metadata{
			"phi3.context_length":                   int32(4096),
			"phi3.embedding_length":                 int32(3072),
			"phi3.feed_forward_length":              int32(8192),
			"phi3.block_count":                      int32(32),
			"phi3.attention.head_count":             int32(32),
			"phi3.attention.head_count_kv":          int32(32),
			"phi3.attention.layer_norm_rms_epsilon": float32(1e-5),
			"phi3.rope.freq_base":                   float32(10000),
			"phi3.rope.dimension_count":             int32(96),
		},
		Tensors: []format.TensorInfo{
			{
				Name:       "token_embd.weight",
				Dimensions: []uint64{3072, 32064},
			},
		},
	}

	cfg, err := phi3.ParseConfig(r)
	if err != nil {
		t.Fatal(err)
	}

	if cfg.HeadDim != 96 {
		t.Fatalf("HeadDim = %d, ожидали 96", cfg.HeadDim)
	}

	if cfg.RopeDim != 96 {
		t.Fatalf("RopeDim = %d, ожидали 96", cfg.RopeDim)
	}

	if cfg.VocabSize != 32064 {
		t.Fatalf("VocabSize = %d, ожидали 32064", cfg.VocabSize)
	}
}

func TestApplyRoPEPartialLeavesTail(t *testing.T) {
	head := make([]float32, 8)
	for i := range head {
		head[i] = float32(i + 1)
	}
	tail := head[6]

	ops.ApplyRoPEPartial(head, 1, 10000, 4)
	if head[6] != tail || head[7] != 8 {
		t.Fatalf("хвост головы изменился: %v", head)
	}
}
