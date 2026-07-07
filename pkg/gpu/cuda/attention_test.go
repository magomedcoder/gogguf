//go:build cuda

package cuda

import (
	"math"
	"testing"

	"github.com/magomedcoder/gogguf/pkg/ops"
)

func TestAttentionGPU(t *testing.T) {
	b, err := Open()
	if err != nil {
		t.Skip("CUDA недоступна:", err)
	}

	if !b.hasAttn {
		t.Skip("CUDA attention kernels недоступны")
	}

	defer b.Close()

	seqLen := 4
	nHeads := 2
	nKVHeads := 1
	headDim := 4

	q := []float32{1, 0, 0, 1, 0.5, 0.5, 0.5, 0.5}
	k := make([]float32, seqLen*nKVHeads*headDim)
	v := make([]float32, seqLen*nKVHeads*headDim)
	for t := range seqLen {
		off := t * nKVHeads * headDim
		for i := range headDim {
			k[off+i] = float32(t+1) * 0.1
			v[off+i] = float32(i+1) * 0.2
		}
	}

	dst := make([]float32, nHeads*headDim)
	scores := make([]float32, seqLen)
	want := make([]float32, len(dst))

	if err := ops.AttentionScoresInto(want, q, k, v, scores, seqLen, nHeads, nKVHeads, headDim); err != nil {
		t.Fatal(err)
	}

	if err := b.AttentionScoresInto(dst, q, k, v, scores, seqLen, nHeads, nKVHeads, headDim); err != nil {
		t.Fatal(err)
	}

	for i := range want {
		if math.Abs(float64(dst[i]-want[i])) > 1e-4 {
			t.Fatalf("[%d] gpu=%v cpu=%v", i, dst[i], want[i])
		}
	}
}
