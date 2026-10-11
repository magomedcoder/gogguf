package vec

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
		got := DotStride(weights, v, i, vStride, len(weights))
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

	if Max(x) != maxPure(x) {
		t.Fatalf("MaxFn = %v, pure = %v", Max(x), maxPure(x))
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
	scaleInPlacePure(x, scale)
	ScaleInPlace(fast, scale)

	for i := range x {
		if math.Abs(float64(x[i]-fast[i])) > 1e-6 {
			t.Fatalf("[%d] pure=%v fast=%v", i, x[i], fast[i])
		}
	}
}
