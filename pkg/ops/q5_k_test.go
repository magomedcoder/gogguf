package ops

import (
	"encoding/binary"
	"math"
	"testing"

	"github.com/magomedcoder/gogguf/pkg/quant"
)

func TestMatMulVecQ5_K(t *testing.T) {
	rows, cols := 2, quant.QK_K
	raw := make([]byte, rows*quant.BlockQ5_KSize)
	vec := make([]float32, cols)
	for i := range vec {
		vec[i] = 1
	}

	for r := range rows {
		block := raw[r*quant.BlockQ5_KSize:]
		binary.LittleEndian.PutUint16(block[0:2], 0x3c00)
		binary.LittleEndian.PutUint16(block[2:4], 0x0000)
		for i := range 4 {
			block[4+i] = 1
			block[8+i] = 0
		}

		qs := block[48:]
		for i := range 128 {
			v := byte((i + r) % 16)
			qs[i] = v | (v << 4)
		}
	}

	got, err := MatMulVecQ5_K(raw, rows, cols, vec)
	if err != nil {
		t.Fatal(err)
	}

	for r := range rows {
		w, err := quant.DequantBlockQ5_K(raw[r*quant.BlockQ5_KSize:])
		if err != nil {
			t.Fatal(err)
		}

		var want float32
		for i := range cols {
			want += w[i]
		}

		if math.Abs(float64(got[r]-want)) > 1e-3 {
			t.Fatalf("row %d: got %v, want %v", r, got[r], want)
		}
	}
}
