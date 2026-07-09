//go:build cuda

package cuda

import (
	"math"
	"testing"

	"github.com/magomedcoder/gogguf/pkg/ops"
)

func TestKVCacheAttention(t *testing.T) {
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
	kvDim := nKVHeads * headDim

	if err := b.KVCacheInit(1, seqLen, kvDim); err != nil {
		t.Fatal(err)
	}

	q := []float32{1, 0, 0, 1, 0.5, 0.5, 0.5, 0.5}
	kHost := make([]float32, seqLen*kvDim)
	vHost := make([]float32, seqLen*kvDim)

	for pos := range seqLen {
		off := pos * kvDim
		for i := range headDim {
			kHost[off+i] = float32(pos+1) * 0.1
			vHost[off+i] = float32(i+1) * 0.2
		}

		if err := b.KVCacheAppend(0, pos, kHost[off:off+kvDim], vHost[off:off+kvDim]); err != nil {
			t.Fatalf("append pos=%d: %v", pos, err)
		}
	}

	dst := make([]float32, nHeads*headDim)
	want := make([]float32, len(dst))
	scores := make([]float32, seqLen)

	if err := ops.AttentionScoresInto(want, q, kHost, vHost, scores, seqLen, nHeads, nKVHeads, headDim); err != nil {
		t.Fatal(err)
	}

	if err := b.AttentionScoresKV(0, dst, q, seqLen, nHeads, nKVHeads, headDim); err != nil {
		t.Fatal(err)
	}

	for i := range want {
		if math.Abs(float64(dst[i]-want[i])) > 1e-4 {
			t.Fatalf("[%d] gpu kv=%v cpu=%v", i, dst[i], want[i])
		}
	}
}
