package ops

import (
	"math"
	"testing"
)

func TestDotStrideMatchesNaive(t *testing.T) {
	weights := []float32{0.1, 0.2, 0.3, 0.4}
	v := []float32{
		1, 2,
		3, 4,
		5, 6,
		7, 8,
	}
	vStride := 2

	for i := range 2 {
		got := dotStride(weights, v, i, vStride, len(weights))
		var want float32
		for t := range weights {
			want += weights[t] * v[i+t*vStride]
		}

		if math.Abs(float64(got-want)) > 1e-6 {
			t.Fatalf("dim %d: got %v want %v", i, got, want)
		}
	}
}

func TestVectorMaxMatchesPure(t *testing.T) {
	x := make([]float32, 64)
	for i := range x {
		x[i] = float32(i%17) - 8
	}

	if vectorMax(x) != vectorMaxPure(x) {
		t.Fatalf("vectorMax = %v, pure = %v", vectorMax(x), vectorMaxPure(x))
	}
}

func TestVecScaleInPlaceMatchesPure(t *testing.T) {
	x := make([]float32, 64)
	fast := make([]float32, 64)
	for i := range x {
		x[i] = float32(i%11) - 5
		fast[i] = x[i]
	}

	scale := float32(0.25)
	vecScaleInPlacePure(x, scale)
	vecScaleInPlace(fast, scale)

	for i := range x {
		if math.Abs(float64(x[i]-fast[i])) > 1e-6 {
			t.Fatalf("[%d] pure=%v fast=%v", i, x[i], fast[i])
		}
	}
}

func TestAttentionScoresMultiToken(t *testing.T) {
	headDim := 2
	nHeads := 1
	nKV := 1
	q := []float32{1, 0}
	k := []float32{
		1, 0,
		0, 1,
	}
	v := []float32{
		1, 2,
		3, 4,
	}

	out, err := AttentionScores(q, k, v, 2, nHeads, nKV, headDim)
	if err != nil {
		t.Fatal(err)
	}

	if len(out) != 2 {
		t.Fatalf("len = %d", len(out))
	}

	var sum float32
	for _, v := range out {
		sum += v
	}

	if sum <= 0 {
		t.Fatalf("unexpected output %v", out)
	}
}
