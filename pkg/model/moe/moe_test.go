package moe

import (
	"math"
	"testing"
)

func TestTopKSoftmax(t *testing.T) {
	logits := []float32{1, 3, 2, 0}
	idxs, w := TopKSoftmax(logits, 2, false, 1)
	if len(idxs) != 2 || idxs[0] != 1 || idxs[1] != 2 {
		t.Fatalf("idxs=%v", idxs)
	}

	maxV := float32(3)
	e1 := float32(math.Exp(float64(3 - maxV)))
	e2 := float32(math.Exp(float64(2 - maxV)))
	e0 := float32(math.Exp(float64(1 - maxV)))
	e3 := float32(math.Exp(float64(0 - maxV)))
	sum := e0 + e1 + e2 + e3
	want0, want1 := e1/sum, e2/sum
	if math.Abs(float64(w[0]-want0)) > 1e-5 || math.Abs(float64(w[1]-want1)) > 1e-5 {
		t.Fatalf("weights=%v want %v %v", w, want0, want1)
	}

	_, wn := TopKSoftmax(logits, 2, true, 2)
	s := wn[0] + wn[1]
	if math.Abs(float64(s-2)) > 1e-5 {
		t.Fatalf("norm+scale sum=%v", s)
	}
}
