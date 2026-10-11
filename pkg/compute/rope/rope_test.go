package rope

import (
	"math"
	"testing"
)

func TestApplyRoPEHeadsMatchesPerHead(t *testing.T) {
	headDim := 4
	nHeads := 3
	v := []float32{1, 0, 1, 0, 0.5, 0.5, 0.5, 0.5, -1, 2, 3, 4}
	want := make([]float32, len(v))
	copy(want, v)

	for h := range nHeads {
		off := h * headDim
		ApplyRoPE(want[off:off+headDim], 2, 10000)
	}

	ApplyRoPEHeads(v, nHeads, headDim, 2, 10000)

	for i := range v {
		if math.Abs(float64(v[i]-want[i])) > 1e-5 {
			t.Fatalf("[%d] heads=%v per-head=%v", i, v[i], want[i])
		}
	}
}

func TestApplyRoPE(t *testing.T) {
	v := []float32{1, 0, 1, 0}
	ApplyRoPE(v, 1, 10000)
	if v[0] == 1 && v[1] == 0 {
		t.Fatalf("RoPE did not change vector: %v", v)
	}
}

func TestApplyRoPEPartialScaledFactors(t *testing.T) {
	base := []float32{1, 0, 0, 1}
	plain := append([]float32(nil), base...)
	scaled := append([]float32(nil), base...)

	ApplyRoPEPartial(plain, 2, 10000, 4)
	ApplyRoPEPartialScaled(scaled, 2, 10000, 4, RoPEScale{
		Factors: []float32{2, 2},
	})

	// factor=2 -> angle half that of plain
	half := []float32{1, 0, 0, 1}
	ApplyRoPEPartial(half, 1, 10000, 4)
	for i := range scaled {
		if math.Abs(float64(scaled[i]-half[i])) > 1e-5 {
			t.Fatalf("[%d] factors=2 pos=2 -> %v, want same as pos=1: %v", i, scaled[i], half[i])
		}
	}

	if math.Abs(float64(scaled[0]-plain[0])) < 1e-4 {
		t.Fatalf("factors must change RoPE: scaled=%v plain=%v", scaled, plain)
	}
}

func TestApplyRoPEPartialScaledAttnFactor(t *testing.T) {
	v := []float32{1, 0, 1, 0}
	ApplyRoPEPartialScaled(v, 1, 10000, 4, RoPEScale{AttnFactor: 2})
	ref := []float32{1, 0, 1, 0}
	ApplyRoPEPartial(ref, 1, 10000, 4)
	for i := range v {
		if math.Abs(float64(v[i]-2*ref[i])) > 1e-4 {
			t.Fatalf("[%d] attn*2: got %v want %v", i, v[i], 2*ref[i])
		}
	}
}
