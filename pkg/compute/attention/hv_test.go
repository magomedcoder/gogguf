package attention

import (
	"math"
	"testing"
)

func TestAttentionScoresHVIntoMatchesSquare(t *testing.T) {
	nHeads, nKV, headDim, seq := 2, 2, 4, 3
	q := make([]float32, nHeads*headDim)
	k := make([]float32, seq*nKV*headDim)
	v := make([]float32, seq*nKV*headDim)
	for i := range q {
		q[i] = float32(i+1) * 0.1
	}
	for i := range k {
		k[i] = float32(i+1) * 0.01
		v[i] = float32(i+1) * 0.02
	}

	scale := float32(1 / math.Sqrt(float64(headDim)))
	want := make([]float32, nHeads*headDim)
	got := make([]float32, nHeads*headDim)
	scores := make([]float32, seq)

	if err := AttentionScoresInto(want, q, k, v, scores, seq, nHeads, nKV, headDim); err != nil {
		t.Fatal(err)
	}

	if err := AttentionScoresHVInto(got, q, k, v, scores, seq, nHeads, nKV, headDim, headDim, scale); err != nil {
		t.Fatal(err)
	}

	for i := range want {
		if math.Abs(float64(got[i]-want[i])) > 1e-5 {
			t.Fatalf("[%d] got=%v want=%v", i, got[i], want[i])
		}
	}
}

func TestAttentionScoresHVIntoSeparateV(t *testing.T) {
	nHeads, kDim, vDim, seq := 1, 4, 2, 2
	q := []float32{1, 0, 0, 0}
	k := []float32{1, 0, 0, 0, 0, 1, 0, 0}
	v := []float32{2, 3, 4, 5}
	dst := make([]float32, vDim)
	scores := make([]float32, seq)

	// scale=1: scores = [1, 0] -> softmax ~ [e/(e+1), 1/(e+1)]
	if err := AttentionScoresHVInto(dst, q, k, v, scores, seq, nHeads, nHeads, kDim, vDim, 1); err != nil {
		t.Fatal(err)
	}

	e := math.Exp(1)
	w0 := float32(e / (e + 1))
	w1 := float32(1 / (e + 1))
	want0 := w0*2 + w1*4
	want1 := w0*3 + w1*5
	if math.Abs(float64(dst[0]-want0)) > 1e-5 || math.Abs(float64(dst[1]-want1)) > 1e-5 {
		t.Fatalf("dst=%v want=[%v %v]", dst, want0, want1)
	}
}
