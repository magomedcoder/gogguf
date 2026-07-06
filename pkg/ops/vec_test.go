package ops

import (
	"math"
	"testing"
)

func TestVecMulInPlaceMatchesPure(t *testing.T) {
	a := make([]float32, 128)
	b := make([]float32, 128)
	want := make([]float32, 128)
	fast := make([]float32, 128)

	for i := range a {
		a[i] = float32(i%17) - 8
		b[i] = float32(i%9) * 0.1
		want[i] = a[i]
		fast[i] = a[i]
	}

	vecMulInPlacePure(want, b)
	vecMulInPlace(fast, b)

	for i := range want {
		if math.Abs(float64(want[i]-fast[i])) > 1e-6 {
			t.Fatalf("[%d] pure=%v fast=%v", i, want[i], fast[i])
		}
	}
}

func TestAddInPlaceMatchesPure(t *testing.T) {
	a := make([]float32, 128)
	b := make([]float32, 128)
	want := make([]float32, 128)
	fast := make([]float32, 128)

	for i := range a {
		a[i] = float32(i%17) - 8
		b[i] = float32(i%9) * 0.1
		want[i] = a[i]
		fast[i] = a[i]
	}

	addInPlacePure(want, b)
	addInPlace(fast, b)

	for i := range want {
		if math.Abs(float64(want[i]-fast[i])) > 1e-6 {
			t.Fatalf("[%d] pure=%v fast=%v", i, want[i], fast[i])
		}
	}
}
