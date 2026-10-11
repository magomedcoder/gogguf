package matmul

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
