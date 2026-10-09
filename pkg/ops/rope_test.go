package ops

import (
	"math"
	"testing"
)

func TestSoftmax(t *testing.T) {
	out := Softmax([]float32{1, 2, 3})
	var sum float32
	for _, v := range out {
		sum += v
	}

	if math.Abs(float64(sum-1)) > 1e-5 {
		t.Fatalf("sum = %v", sum)
	}

	if out[2] <= out[0] {
		t.Fatalf("expected out[2] > out[0], got %v", out)
	}
}

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
		t.Fatalf("RoPE не изменил вектор: %v", v)
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
			t.Fatalf("[%d] factors=2 pos=2 -> %v, ожидали как pos=1: %v", i, scaled[i], half[i])
		}
	}

	if math.Abs(float64(scaled[0]-plain[0])) < 1e-4 {
		t.Fatalf("factors должны менять RoPE: scaled=%v plain=%v", scaled, plain)
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

func TestAttentionScoresSingleToken(t *testing.T) {
	headDim := 2
	nHeads := 2
	nKV := 1
	q := []float32{1, 0, 0, 1}
	k := []float32{1, 0}
	v := []float32{2, 3}

	out, err := AttentionScores(q, k, v, 1, nHeads, nKV, headDim)
	if err != nil {
		t.Fatal(err)
	}

	if out[0] != 2 || out[1] != 3 || out[2] != 2 || out[3] != 3 {
		t.Fatalf("got %v", out)
	}
}
