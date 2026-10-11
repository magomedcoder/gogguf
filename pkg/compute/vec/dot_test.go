package vec

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

	got := Dot(a, b)
	want := DotPure(a, b)
	if math.Abs(float64(got-want)) > 1e-3 {
		t.Fatalf("Dot = %v, dotPure = %v", got, want)
	}
}

func TestDotEmpty(t *testing.T) {
	if got := Dot(nil, nil); got != 0 {
		t.Fatalf("Dot(nil) = %v", got)
	}
}
