//go:build cuda

package cuda

import (
	"math"
	"testing"

	"github.com/magomedcoder/gogguf/pkg/ops"
)

func TestRoPEHeadsGPU(t *testing.T) {
	b, err := Open()
	if err != nil {
		t.Skip("CUDA недоступна:", err)
	}

	if !b.hasRoPE {
		t.Skip("CUDA rope kernel недоступен")
	}

	defer b.Close()

	headDim := 4
	nHeads := 2
	v := []float32{1, 0, 1, 0, 0.5, 0.5, 0.5, 0.5}
	want := make([]float32, len(v))
	copy(want, v)

	ops.ApplyRoPEHeads(want, nHeads, headDim, 3, 10000)

	if err := b.ApplyRoPEHeads(v, nHeads, headDim, 3, 10000); err != nil {
		t.Fatal(err)
	}

	for i := range want {
		if math.Abs(float64(v[i]-want[i])) > 1e-4 {
			t.Fatalf("[%d] gpu=%v cpu=%v", i, v[i], want[i])
		}
	}
}
