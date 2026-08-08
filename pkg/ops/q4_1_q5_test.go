package ops

import (
	"encoding/binary"
	"math"
	"testing"

	"github.com/magomedcoder/gogguf/pkg/quant"
)

func TestMatMulVecQ4_1(t *testing.T) {
	rows, cols := 2, quant.QK4_1
	raw := make([]byte, rows*quant.BlockQ4_1Size)
	vec := make([]float32, cols)
	for i := range vec {
		vec[i] = 1
	}

	for r := range rows {
		block := raw[r*quant.BlockQ4_1Size:]
		binary.LittleEndian.PutUint16(block[0:2], 0x3c00)
		binary.LittleEndian.PutUint16(block[2:4], 0x3800)
		for i := range 16 {
			v := byte((i + r) % 16)
			block[4+i] = v | (v << 4)
		}
	}

	got, err := MatMulVecQ4_1(raw, rows, cols, vec)
	if err != nil {
		t.Fatal(err)
	}

	for r := range rows {
		w, err := quant.DequantBlockQ4_1(raw[r*quant.BlockQ4_1Size:])
		if err != nil {
			t.Fatal(err)
		}

		var want float32
		for i := range cols {
			want += w[i]
		}

		if math.Abs(float64(got[r]-want)) > 1e-3 {
			t.Fatalf("строка %d: получили %v, ожидали %v", r, got[r], want)
		}
	}
}

func TestMatMulVecQ5_0(t *testing.T) {
	rows, cols := 2, quant.QK5_0
	raw := make([]byte, rows*quant.BlockQ5_0Size)
	vec := make([]float32, cols)
	for i := range vec {
		vec[i] = 1
	}

	for r := range rows {
		block := raw[r*quant.BlockQ5_0Size:]
		binary.LittleEndian.PutUint16(block[0:2], 0x3c00)
		binary.LittleEndian.PutUint32(block[2:6], uint32(0x55555555+r))
		for i := range 16 {
			v := byte((i + r) % 16)
			block[6+i] = v | (v << 4)
		}
	}

	got, err := MatMulVecQ5_0(raw, rows, cols, vec)
	if err != nil {
		t.Fatal(err)
	}

	for r := range rows {
		w, err := quant.DequantBlockQ5_0(raw[r*quant.BlockQ5_0Size:])
		if err != nil {
			t.Fatal(err)
		}

		var want float32
		for i := range cols {
			want += w[i]
		}

		if math.Abs(float64(got[r]-want)) > 1e-3 {
			t.Fatalf("строка %d: получили %v, ожидали %v", r, got[r], want)
		}
	}
}

func TestMatMulVecQ5_1(t *testing.T) {
	rows, cols := 2, quant.QK5_1
	raw := make([]byte, rows*quant.BlockQ5_1Size)
	vec := make([]float32, cols)
	for i := range vec {
		vec[i] = 1
	}

	for r := range rows {
		block := raw[r*quant.BlockQ5_1Size:]
		binary.LittleEndian.PutUint16(block[0:2], 0x3c00)
		binary.LittleEndian.PutUint16(block[2:4], 0x3800)
		binary.LittleEndian.PutUint32(block[4:8], uint32(0x55555555+r))
		for i := range 16 {
			v := byte((i + r) % 16)
			block[8+i] = v | (v << 4)
		}
	}

	got, err := MatMulVecQ5_1(raw, rows, cols, vec)
	if err != nil {
		t.Fatal(err)
	}

	for r := range rows {
		w, err := quant.DequantBlockQ5_1(raw[r*quant.BlockQ5_1Size:])
		if err != nil {
			t.Fatal(err)
		}

		var want float32
		for i := range cols {
			want += w[i]
		}

		if math.Abs(float64(got[r]-want)) > 1e-3 {
			t.Fatalf("строка %d: получили %v, ожидали %v", r, got[r], want)
		}
	}
}
