package ops

import (
	"math"
	"testing"
)

func TestDotMatchesPure(t *testing.T) {
	a := make([]float32, 256)
	b := make([]float32, 256)
	for i := range a {
		a[i] = float32(i)*0.01 - 1
		b[i] = float32(i%17) * 0.03
	}

	got := dot(a, b)
	want := dotPure(a, b)
	if math.Abs(float64(got-want)) > 1e-3 {
		t.Fatalf("dot = %v, dotPure = %v", got, want)
	}
}

func TestDotEmpty(t *testing.T) {
	if got := dot(nil, nil); got != 0 {
		t.Fatalf("dot(nil) = %v", got)
	}
}
