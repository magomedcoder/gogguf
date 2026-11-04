package deepseek2

import (
	"math"
	"testing"

	"github.com/magomedcoder/gogguf/pkg/compute"
	"github.com/magomedcoder/gogguf/pkg/device"
)

func TestApplyRoPEPEMatchesHostPartialScaled(t *testing.T) {
	cfg := Config{
		NumHeads:       2,
		QKNopeDim:      4,
		RopeDim:        4,
		RopeFreqBase:   10000,
		RopeFreqScale:  0.25,
		RopeAttnFactor: 1.2,
		KVLoraRank:     8,
		VHeadDim:       4,
		EmbeddingDim:   16,
		ContextLength:  8,
		VocabSize:      8,
	}
	m := &Model{
		cfg:     cfg,
		dev:     device.CPUBackend{},
		scratch: newScratch(cfg),
	}

	pos := 3
	pe := []float32{1, 0.5, -0.25, 0.75, 0.1, 0.2, 0.3, 0.4}
	want := make([]float32, len(pe))
	copy(want, pe)
	for h := 0; h < cfg.NumHeads; h++ {
		off := h * cfg.RopeDim
		compute.ApplyRoPEPartialScaled(want[off:off+cfg.RopeDim], pos, cfg.RopeFreqBase, cfg.RopeDim, cfg.ropeScale())
	}

	got := make([]float32, len(pe))
	copy(got, pe)
	if err := m.applyRoPEPE(got, cfg.NumHeads, pos); err != nil {
		t.Fatal(err)
	}

	for i := range want {
		if math.Abs(float64(got[i]-want[i])) > 1e-5 {
			t.Fatalf("[%d] got=%v want=%v", i, got[i], want[i])
		}
	}
}

func TestRopeQPeLeavesNopeUntouched(t *testing.T) {
	cfg := Config{
		NumHeads:       2,
		QKNopeDim:      2,
		RopeDim:        4,
		RopeFreqBase:   10000,
		RopeFreqScale:  1,
		RopeAttnFactor: 1,
		KVLoraRank:     4,
		VHeadDim:       2,
		EmbeddingDim:   8,
		ContextLength:  4,
		VocabSize:      4,
	}
	m := &Model{
		cfg:     cfg,
		scratch: newScratch(cfg),
	}

	headK := cfg.QKNopeDim + cfg.RopeDim
	heads := make([]float32, cfg.NumHeads*headK)
	for i := range heads {
		heads[i] = float32(i + 1)
	}
	nope0 := []float32{heads[0], heads[1]}
	nope1 := []float32{heads[headK], heads[headK+1]}

	if err := m.ropeQPe(heads, 2); err != nil {
		t.Fatal(err)
	}

	if heads[0] != nope0[0] || heads[1] != nope0[1] || heads[headK] != nope1[0] || heads[headK+1] != nope1[1] {
		t.Fatalf("nope dims changed: %v", heads)
	}
}
