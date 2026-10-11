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
