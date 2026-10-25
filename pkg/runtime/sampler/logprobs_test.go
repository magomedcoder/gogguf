package sampler

import (
	"math"
	"testing"
)

func TestSoftmaxLogprobUniform(t *testing.T) {
	logits := []float32{0, 0, 0, 0}
	lp := SoftmaxLogprob(logits, 1)
	want := -math.Log(4)
	if math.Abs(lp-want) > 1e-9 {
		t.Fatalf("logprob=%v want %v", lp, want)
	}
}

func TestTopLogprobs(t *testing.T) {
	logits := []float32{1, 3, 2, 0}
	top := TopLogprobs(logits, 2)
	if len(top) != 2 || top[0].ID != 1 || top[1].ID != 2 {
		t.Fatalf("top=%v", top)
	}

	if top[0].Logprob < top[1].Logprob {
		t.Fatalf("not sorted: %v", top)
	}
}
