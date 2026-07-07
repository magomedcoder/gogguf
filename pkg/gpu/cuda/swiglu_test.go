//go:build cuda

package cuda

import (
	"math"
	"testing"

	"github.com/magomedcoder/gogguf/pkg/ops"
)

func TestSwiGLUGPU(t *testing.T) {
	b, err := Open()
	if err != nil {
		t.Skip("CUDA недоступна:", err)
	}

	if !b.hasSwiGLU {
		t.Skip("CUDA swiglu kernel недоступен")
	}

	defer b.Close()

	gate := []float32{-1.5, 0, 1.2, 2.5, -0.3, 0.7}
	up := []float32{0.5, -1, 2, 0.25, 3, -2}
	wantGate := make([]float32, len(gate))
	copy(wantGate, gate)
	wantUp := make([]float32, len(up))
	copy(wantUp, up)

	ops.SwiGLUInPlace(wantGate, wantUp)

	if err := b.SwiGLUInPlace(gate, up); err != nil {
		t.Fatal(err)
	}

	for i := range wantGate {
		if math.Abs(float64(gate[i]-wantGate[i])) > 1e-3 {
			t.Fatalf("[%d] gpu=%v cpu=%v", i, gate[i], wantGate[i])
		}
	}
}
