//go:build cuda

package cuda

import (
	"math"
	"math/rand"
	"testing"

	"github.com/magomedcoder/gogguf/pkg/compute"
)

func flashRandom(n int, rng *rand.Rand) []float32 {
	out := make([]float32, n)
	for i := range out {
		out[i] = rng.Float32()*2 - 1
	}
	return out
}

// TestAttnFlashMatchesCPU: fused kernel (single query + causal batch) vs CPU reference.
// seq spans multiple 256-position chunks; GQA group > 1.
func TestAttnFlashMatchesCPU(t *testing.T) {
	b, err := Open()
	if err != nil {
		t.Skip("CUDA unavailable:", err)
	}
	defer b.Close()

	if !b.hasAttn {
		t.Skip("CUDA attention kernels unavailable")
	}

	if !b.hasAttnFlash {
		t.Skip("attn_flash kernel unavailable")
	}

	cases := []struct {
		name                                string
		past, batch, nHeads, nKVHeads, hdim int
	}{
		{"decode_short", 6, 1, 4, 2, 64},
		{"decode_multichunk", 600, 1, 8, 2, 128},
		{"batch_small", 0, 5, 4, 4, 32},
		{"batch_past_gqa", 300, 7, 8, 2, 64},
		{"hdim_odd", 10, 3, 2, 1, 40},
	}

	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			rng := rand.New(rand.NewSource(1))
			total := tc.past + tc.batch
			kvDim := tc.nKVHeads * tc.hdim
			qDim := tc.nHeads * tc.hdim

			k := flashRandom(total*kvDim, rng)
			v := flashRandom(total*kvDim, rng)
			q := flashRandom(tc.batch*qDim, rng)

			if err := b.KVCacheInit(1, total, kvDim, tc.nHeads, tc.hdim); err != nil {
				t.Fatal(err)
			}

			if err := b.KVCacheAppendN(0, 0, k, v, total); err != nil {
				t.Fatal(err)
			}

			want := make([]float32, tc.batch*qDim)
			scores := make([]float32, total)
			if err := compute.AttentionScoresBatchCausalInto(want, q, k, v, scores, tc.past, tc.batch, tc.nHeads, tc.nKVHeads, tc.hdim); err != nil {
				t.Fatal(err)
			}

			// batched single launch
			got := make([]float32, len(want))
			if err := b.AttentionScoresKVBatch(0, got, q, tc.batch, tc.past, tc.nHeads, tc.nKVHeads, tc.hdim); err != nil {
				t.Fatal(err)
			}

			// per-query path (fused single query kernel)
			one := make([]float32, len(want))
			for i := range tc.batch {
				off := i * qDim
				if err := b.AttentionScoresKV(0, one[off:off+qDim], q[off:off+qDim], tc.past+i+1, tc.nHeads, tc.nKVHeads, tc.hdim); err != nil {
					t.Fatal(err)
				}
			}

			for i := range want {
				if d := math.Abs(float64(got[i] - want[i])); d > 2e-4 {
					t.Fatalf("batch [%d] gpu=%v cpu=%v", i, got[i], want[i])
				}

				if d := math.Abs(float64(one[i] - want[i])); d > 2e-4 {
					t.Fatalf("single [%d] gpu=%v cpu=%v", i, one[i], want[i])
				}
			}
		})
	}
}

// TestAttnFlashFallback: with the fused kernel disabled the legacy qk/softmax/v path still matches.
func TestAttnFlashFallback(t *testing.T) {
	b, err := Open()
	if err != nil {
		t.Skip("CUDA unavailable:", err)
	}
	defer b.Close()

	if !b.hasAttn {
		t.Skip("CUDA attention kernels unavailable")
	}

	b.hasAttnFlash = false
	b.drv.fn_attn_flash = nil

	past, batch, nHeads, nKVHeads, hdim := 20, 3, 4, 2, 32
	total := past + batch
	kvDim := nKVHeads * hdim
	qDim := nHeads * hdim
	rng := rand.New(rand.NewSource(2))
	k := flashRandom(total*kvDim, rng)
	v := flashRandom(total*kvDim, rng)
	q := flashRandom(batch*qDim, rng)

	if err := b.KVCacheInit(1, total, kvDim, nHeads, hdim); err != nil {
		t.Fatal(err)
	}

	if err := b.KVCacheAppendN(0, 0, k, v, total); err != nil {
		t.Fatal(err)
	}

	want := make([]float32, batch*qDim)
	if err := compute.AttentionScoresBatchCausalInto(want, q, k, v, make([]float32, total), past, batch, nHeads, nKVHeads, hdim); err != nil {
		t.Fatal(err)
	}

	got := make([]float32, len(want))
	if err := b.AttentionScoresKVBatch(0, got, q, batch, past, nHeads, nKVHeads, hdim); err != nil {
		t.Fatal(err)
	}

	for i := range want {
		if math.Abs(float64(got[i]-want[i])) > 1e-4 {
			t.Fatalf("[%d] gpu=%v cpu=%v", i, got[i], want[i])
		}
	}
}
