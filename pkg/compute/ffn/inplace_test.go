package ffn

import (
	"math"
	"testing"

	"github.com/magomedcoder/gogguf/pkg/compute/vec"
)

func testSwiGLU(gate, up []float32) []float32 {
	out := make([]float32, len(gate))
	copy(out, gate)
	for i := range out {
		out[i] = SiLU(out[i])
	}
	vec.MulInPlace(out, up)

	return out
}

func TestSwiGLUInPlace(t *testing.T) {
	gate := []float32{0, 1, -1}
	up := []float32{2, 3, 4}
	want := testSwiGLU(append([]float32(nil), gate...), up)

	SwiGLUInPlace(gate, up)
	for i := range gate {
		if math.Abs(float64(gate[i]-want[i])) > 1e-5 {
			t.Fatalf("[%d] got=%v want=%v", i, gate[i], want[i])
		}
	}
}

func TestSiluInPlaceMatchesScalar(t *testing.T) {
	for _, n := range []int{1, 7, 8, 9, 31, 64, 1000} {
		x := make([]float32, n)
		for i := range x {
			x[i] = float32(i%41)*0.7 - 14
		}

		// extremes
		if n >= 4 {
			x[0], x[1], x[2], x[3] = 100, -100, 0, -88.5
		}

		got := append([]float32(nil), x...)
		siluInPlace(got)
		for i, v := range x {
			want := SiLU(v)
			if math.Abs(float64(got[i]-want)) > 1e-5*(1+math.Abs(float64(want))) {
				t.Fatalf("n=%d [%d] x=%v got=%v want=%v", n, i, v, got[i], want)
			}
		}
	}
}

func BenchmarkSiluInPlace(b *testing.B) {
	x := make([]float32, 8192)
	for i := range x {
		x[i] = float32(i%41)*0.3 - 6
	}

	b.ResetTimer()
	for range b.N {
		siluInPlace(x)
	}
}
