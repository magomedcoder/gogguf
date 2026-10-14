package device

import "testing"

type recKV struct {
	Backend
	seqLens []int
}

func (r *recKV) AttentionScoresKV(layer int, dst, q []float32, seqLen, nHeads, nKVHeads, headDim int) error {
	r.seqLens = append(r.seqLens, seqLen)
	dst[0] = q[0]
	return nil
}

type recBatch struct {
	recKV
	called bool
}

func (r *recBatch) AttentionScoresKVBatch(layer int, dst, q []float32, n, pastLen, nHeads, nKVHeads, headDim int) error {
	r.called = true
	return nil
}

func TestAttentionScoresKVBatchLoopFallback(t *testing.T) {
	r := &recKV{}
	q := make([]float32, 3*2*4)
	dst := make([]float32, len(q))
	for i := range 3 {
		q[i*8] = float32(i + 1)
	}

	if err := AttentionScoresKVBatch(r, 0, dst, q, 3, 10, 2, 1, 4); err != nil {
		t.Fatal(err)
	}

	if len(r.seqLens) != 3 || r.seqLens[0] != 11 || r.seqLens[1] != 12 || r.seqLens[2] != 13 {
		t.Fatalf("seqLens=%v", r.seqLens)
	}

	if dst[8] != 2 || dst[16] != 3 {
		t.Fatalf("dst=%v", dst)
	}
}

func TestAttentionScoresKVBatchPrefersBatchBackend(t *testing.T) {
	r := &recBatch{}
	if err := AttentionScoresKVBatch(r, 0, make([]float32, 8), make([]float32, 8), 1, 0, 2, 1, 4); err != nil {
		t.Fatal(err)
	}

	if !r.called || len(r.seqLens) != 0 {
		t.Fatalf("batch=%v loop=%v", r.called, r.seqLens)
	}
}
