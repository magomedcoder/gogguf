package attention

import (
	"testing"
)

func TestAttentionScoresMultiToken(t *testing.T) {
	headDim := 2
	nHeads := 1
	nKV := 1
	q := []float32{1, 0}
	k := []float32{
		1, 0,
		0, 1,
	}
	v := []float32{
		1, 2,
		3, 4,
	}

	out, err := testAttentionScores(q, k, v, 2, nHeads, nKV, headDim)
	if err != nil {
		t.Fatal(err)
	}

	if len(out) != 2 {
		t.Fatalf("len = %d", len(out))
	}

	var sum float32
	for _, v := range out {
		sum += v
	}

	if sum <= 0 {
		t.Fatalf("unexpected output %v", out)
	}
}
