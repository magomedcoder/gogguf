package ops

import (
	"math"
	"testing"
)

func TestApplyRoPEHeadsNormVsPerHead(t *testing.T) {
	headDim := 8
	nHeads := 2
	v := []float32{1, 0, 1, 0, 0.5, 0.5, 0.5, 0.5, -1, 2, 3, 4, 0, 0, 0, 0}
	want := make([]float32, len(v))
	copy(want, v)

	for h := range nHeads {
		off := h * headDim
		ApplyRoPENorm(want[off:off+headDim], 2, 10000)
	}

	ApplyRoPEHeadsNorm(v, nHeads, headDim, 2, 10000)

	for i := range v {
		if math.Abs(float64(v[i]-want[i])) > 1e-5 {
			t.Fatalf("[%d] batched=%v per-head=%v", i, v[i], want[i])
		}
	}
}

func TestRoPENormDiffersFromNeoX(t *testing.T) {
	norm := []float32{1, 0, 1, 0, 0.5, 0.5, 0.5, 0.5}
	neox := []float32{1, 0, 1, 0, 0.5, 0.5, 0.5, 0.5}

	ApplyRoPENorm(norm, 3, 500000)
	ApplyRoPE(neox, 3, 500000)

	same := true
	for i := range norm {
		if math.Abs(float64(norm[i]-neox[i])) > 1e-6 {
			same = false
			break
		}
	}

	if same {
		t.Fatal("ожидали различие между Norm и NeoX RoPE")
	}
}
