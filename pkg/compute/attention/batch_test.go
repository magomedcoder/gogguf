package attention

import (
	"math"
	"testing"
)

func TestAttentionScoresBatchCausal(t *testing.T) {
	nHeads, nKV, headDim := 1, 1, 2
	past, batch := 1, 2
	q := []float32{
		1, 0,
		0, 1,
	}
	k := []float32{
		1, 0,
		1, 0,
		0, 1,
	}
	v := []float32{
		10, 0,
		20, 0,
		0, 30,
	}

	dst := make([]float32, batch*nHeads*headDim)
	scores := make([]float32, past+batch)
	if err := AttentionScoresBatchCausalInto(dst, q, k, v, scores, past, batch, nHeads, nKV, headDim); err != nil {
		t.Fatal(err)
	}

	if dst[0] < 10 || dst[0] > 20 {
		t.Fatalf("dst b0=%v, want blend [10..20], 0", dst[:2])
	}

	if dst[3] < 15 {
		t.Fatalf("dst b1=%v, want larger contribution on dim1", dst[2:4])
	}
}

func TestAttentionBatchMatchesSerialCausal(t *testing.T) {
	nHeads, nKV, headDim := 2, 1, 4
	past, batch := 2, 3
	total := past + batch
	qDim := nHeads * headDim
	kvDim := nKV * headDim

	q := make([]float32, batch*qDim)
	k := make([]float32, total*kvDim)
	v := make([]float32, total*kvDim)
	for i := range q {
		q[i] = float32(i%7) * 0.1
	}

	for i := range k {
		k[i] = float32((i*3)%11) * 0.05
		v[i] = float32((i*5)%13) * 0.05
	}

	dstB := make([]float32, batch*qDim)
	scores := make([]float32, total)
	if err := AttentionScoresBatchCausalInto(dstB, q, k, v, scores, past, batch, nHeads, nKV, headDim); err != nil {
		t.Fatal(err)
	}

	dst1 := make([]float32, qDim)
	sc1 := make([]float32, total)
	for b := range batch {
		causal := past + b + 1
		if err := AttentionScoresInto(dst1, q[b*qDim:(b+1)*qDim], k[:causal*kvDim], v[:causal*kvDim], sc1, causal, nHeads, nKV, headDim); err != nil {
			t.Fatal(err)
		}

		for i := range qDim {
			if math.Abs(float64(dstB[b*qDim+i]-dst1[i])) > 1e-5 {
				t.Fatalf("b=%d i=%d: batch=%v serial=%v", b, i, dstB[b*qDim+i], dst1[i])
			}
		}
	}
}
