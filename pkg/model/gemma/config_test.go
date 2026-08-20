package gemma_test

import (
	"math"
	"testing"

	"github.com/magomedcoder/gogguf/pkg/format"
	"github.com/magomedcoder/gogguf/pkg/model/gemma"
	"github.com/magomedcoder/gogguf/pkg/ops"
)

func TestParseConfigGemma2B(t *testing.T) {
	r := &format.Reader{
		Metadata: format.Metadata{
			"gemma.context_length":                   int32(8192),
			"gemma.embedding_length":                 int32(2048),
			"gemma.feed_forward_length":              int32(16384),
			"gemma.block_count":                      int32(18),
			"gemma.attention.head_count":             int32(8),
			"gemma.attention.head_count_kv":          int32(1),
			"gemma.attention.key_length":             int32(256),
			"gemma.attention.layer_norm_rms_epsilon": float32(1e-6),
			"gemma.rope.freq_base":                   float32(10000),
		},
		Tensors: []format.TensorInfo{
			{Name: "token_embd.weight", Dimensions: []uint64{2048, 256000}},
		},
	}

	cfg, err := gemma.ParseConfigGemma(r)
	if err != nil {
		t.Fatal(err)
	}

	if cfg.HeadDim != 256 {
		t.Fatalf("HeadDim=%d, ожидали 256", cfg.HeadDim)
	}

	if cfg.NumKVHeads != 1 {
		t.Fatalf("NumKVHeads=%d", cfg.NumKVHeads)
	}

	wantScale := float32(math.Sqrt(2048))
	if math.Abs(float64(cfg.EmbedScale-wantScale)) > 1e-5 {
		t.Fatalf("EmbedScale=%v want %v", cfg.EmbedScale, wantScale)
	}

	if cfg.Variant != gemma.VariantGemma {
		t.Fatalf("Variant=%v", cfg.Variant)
	}
}

func TestParseConfigGemma2(t *testing.T) {
	r := &format.Reader{
		Metadata: format.Metadata{
			"gemma2.context_length":                   int32(8192),
			"gemma2.embedding_length":                 int32(2304),
			"gemma2.feed_forward_length":              int32(9216),
			"gemma2.block_count":                      int32(26),
			"gemma2.attention.head_count":             int32(8),
			"gemma2.attention.head_count_kv":          int32(4),
			"gemma2.attention.key_length":             int32(256),
			"gemma2.attention.layer_norm_rms_epsilon": float32(1e-6),
			"gemma2.attention.sliding_window":         int32(4096),
			"gemma2.attn_logit_softcapping":           float32(50),
			"gemma2.final_logit_softcapping":          float32(30),
		},
		Tensors: []format.TensorInfo{
			{
				Name:       "token_embd.weight",
				Dimensions: []uint64{2304, 256000},
			},
		},
	}

	cfg, err := gemma.ParseConfigGemma2(r)
	if err != nil {
		t.Fatal(err)
	}

	if !cfg.IsSWALayer(0) || cfg.IsSWALayer(1) {
		t.Fatalf("SWA pattern: layer0=%v layer1=%v", cfg.IsSWALayer(0), cfg.IsSWALayer(1))
	}

	if cfg.AttnLogitSoftcap != 50 || cfg.FinalLogitSoftcap != 30 {
		t.Fatalf("softcaps attn=%v final=%v", cfg.AttnLogitSoftcap, cfg.FinalLogitSoftcap)
	}
}

func TestGeGLUAndSoftcap(t *testing.T) {
	gate := []float32{1, -1}
	up := []float32{2, 3}
	ops.GeGLUInPlace(gate, up)
	if gate[0] <= 0 || gate[1] >= 0 {
		t.Fatalf("GeGLU unexpected: %v", gate)
	}

	x := []float32{100, -100}
	ops.SoftcapInPlace(x, 30)
	if math.Abs(float64(x[0]-30)) > 0.1 || math.Abs(float64(x[1]+30)) > 0.1 {
		t.Fatalf("Softcap: %v", x)
	}
}
