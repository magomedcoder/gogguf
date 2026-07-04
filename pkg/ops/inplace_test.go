package ops

import (
	"math"
	"testing"
)

func TestRMSNormIntoMatchesRMSNorm(t *testing.T) {
	x := []float32{1, 2, 3}
	w := []float32{1, 1, 1}
	dst := make([]float32, len(x))

	if err := RMSNormInto(dst, x, w, 1e-5); err != nil {
		t.Fatal(err)
	}

	want, err := RMSNorm(x, w, 1e-5)
	if err != nil {
		t.Fatal(err)
	}

	for i := range dst {
		if math.Abs(float64(dst[i]-want[i])) > 1e-5 {
			t.Fatalf("[%d] dst=%v want=%v", i, dst[i], want[i])
		}
	}
}

func TestSwiGLUInPlace(t *testing.T) {
	gate := []float32{0, 1, -1}
	up := []float32{2, 3, 4}
	want := SwiGLU(append([]float32(nil), gate...), up)

	SwiGLUInPlace(gate, up)
	for i := range gate {
		if math.Abs(float64(gate[i]-want[i])) > 1e-5 {
			t.Fatalf("[%d] got=%v want=%v", i, gate[i], want[i])
		}
	}
}
