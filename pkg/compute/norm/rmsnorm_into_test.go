package norm

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
