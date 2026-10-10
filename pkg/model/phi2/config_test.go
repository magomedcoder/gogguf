package phi2_test

import (
	"math"
	"testing"

	"github.com/magomedcoder/gogguf/pkg/format"
	"github.com/magomedcoder/gogguf/pkg/model/phi2"
	"github.com/magomedcoder/gogguf/pkg/ops"
)

func TestParseConfigPhi2(t *testing.T) {
	r := &format.Reader{
		Metadata: format.Metadata{
			"phi2.context_length":               int32(2048),
			"phi2.embedding_length":             int32(2560),
			"phi2.feed_forward_length":          int32(10240),
			"phi2.block_count":                  int32(32),
			"phi2.attention.head_count":         int32(32),
			"phi2.attention.layer_norm_epsilon": float32(1e-5),
			"phi2.rope.freq_base":               float32(10000),
			"phi2.rope.dimension_count":         int32(32),
		},
		Tensors: []format.TensorInfo{
			{
				Name:       "token_embd.weight",
				Dimensions: []uint64{2560, 51200},
			},
		},
	}

	cfg, err := phi2.ParseConfig(r)
	if err != nil {
		t.Fatal(err)
	}

	if cfg.HeadDim != 80 {
		t.Fatalf("HeadDim = %d, expected 80", cfg.HeadDim)
	}

	if cfg.RopeDim != 32 {
		t.Fatalf("RopeDim = %d, expected 32 (partial 0.4)", cfg.RopeDim)
	}

	if cfg.NumKVHeads != 32 {
		t.Fatalf("NumKVHeads = %d, expected 32", cfg.NumKVHeads)
	}

	if cfg.VocabSize != 51200 {
		t.Fatalf("VocabSize = %d, expected 51200", cfg.VocabSize)
	}
}

func TestLayerNormInto(t *testing.T) {
	x := []float32{1, 2, 3, 4}
	w := []float32{1, 1, 1, 1}
	b := []float32{0, 0, 0, 0}
	dst := make([]float32, 4)

	if err := ops.LayerNormInto(dst, x, w, b, 1e-5); err != nil {
		t.Fatal(err)
	}

	var sum float64
	for _, v := range dst {
		sum += float64(v)
	}

	if math.Abs(sum) > 1e-4 {
		t.Fatalf("mean after LN ≈ 0, sum=%v dst=%v", sum, dst)
	}
}

func TestGELUInPlace(t *testing.T) {
	x := []float32{0, 1, -1}
	ops.GELUInPlace(x)
	if math.Abs(float64(x[0])) > 1e-6 {
		t.Fatalf("GELU(0)=%v", x[0])
	}

	if x[1] <= 0.8 || x[1] >= 1 {
		t.Fatalf("GELU(1)=%v outside expected range", x[1])
	}

	if x[2] >= 0 {
		t.Fatalf("GELU(-1)=%v expected to be < 0", x[2])
	}
}
