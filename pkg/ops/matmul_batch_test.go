package ops

import (
	"encoding/binary"
	"math"
	"testing"

	"github.com/magomedcoder/gogguf/pkg/quant"
)

func TestMatMulMatIntoMatchesVec(t *testing.T) {
	rows, cols, batch := 3, 4, 2
	matrix := []float32{
		1, 0, 0, 0,
		0, 1, 0, 0,
		0, 0, 1, 0,
	}
	x := []float32{
		1, 2, 3, 4,
		5, 6, 7, 8,
	}

	out := make([]float32, batch*rows)
	if err := MatMulMatInto(matrix, rows, cols, x, batch, out); err != nil {
		t.Fatal(err)
	}

	want := []float32{1, 2, 3, 5, 6, 7}
	for i := range want {
		if out[i] != want[i] {
			t.Fatalf("out=%v, want %v", out, want)
		}
	}
}

func TestMatMulMatQ8_0MatchesVec(t *testing.T) {
	rows, cols, batch := 2, 32, 2
	raw := make([]byte, rows*quant.BlockQ8_0Size)
	for r := range rows {
		off := r * quant.BlockQ8_0Size
		binary.LittleEndian.PutUint16(raw[off:off+2], 0x3c00) // scale=1
		for i := range quant.QK8_0 {
			raw[off+2+i] = byte(int8(r + 1))
		}
	}

	x := make([]float32, batch*cols)
	for i := range x {
		x[i] = float32(i%5) * 0.25
	}

	outB := make([]float32, batch*rows)
	out1 := make([]float32, rows)
	if err := MatMulMatQ8_0Into(raw, rows, cols, x, batch, outB); err != nil {
		t.Fatal(err)
	}

	for b := range batch {
		if err := MatMulVecQ8_0Into(raw, rows, cols, x[b*cols:(b+1)*cols], out1); err != nil {
			t.Fatal(err)
		}
		for r := range rows {
			if math.Abs(float64(outB[b*rows+r]-out1[r])) > 1e-5 {
				t.Fatalf("b=%d r=%d: batch=%v vec=%v", b, r, outB[b*rows+r], out1[r])
			}
		}
	}
}

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
