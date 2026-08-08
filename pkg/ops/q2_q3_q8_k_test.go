package ops

import (
	"encoding/binary"
	"math"
	"testing"

	"github.com/magomedcoder/gogguf/pkg/quant"
)

// TestMatMulVecQ2_K сверяет MatMulVecQ2_K с суммой деквантизированного блока (вектор из единиц -> результат = сумма весов строки)
func TestMatMulVecQ2_K(t *testing.T) {
	rows, cols := 2, quant.QK_K
	raw := make([]byte, rows*quant.BlockQ2_KSize)
	vec := make([]float32, cols)
	for i := range vec {
		vec[i] = 1
	}

	for r := range rows {
		block := raw[r*quant.BlockQ2_KSize:]
		// scales[16]: младшие 4 бита - scale, старшие - min
		for i := range 16 {
			block[i] = byte(0x11 + r)
		}

		// qs[64]: 2-битные кванты (0x55 = 01 01 01 01)
		for i := 16; i < 80; i++ {
			block[i] = 0x55
		}
		binary.LittleEndian.PutUint16(block[80:82], 0x3c00) // d = 1.0
		binary.LittleEndian.PutUint16(block[82:84], 0x3800) // dmin = 0.5
	}

	got, err := MatMulVecQ2_K(raw, rows, cols, vec)
	if err != nil {
		t.Fatal(err)
	}

	for r := range rows {
		w, err := quant.DequantBlockQ2_K(raw[r*quant.BlockQ2_KSize:])
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

// TestMatMulVecQ8_K сверяет MatMulVecQ8_K с суммой деквантизированного блока.
func TestMatMulVecQ8_K(t *testing.T) {
	rows, cols := 2, quant.QK_K
	raw := make([]byte, rows*quant.BlockQ8_KSize)
	vec := make([]float32, cols)
	for i := range vec {
		vec[i] = 1
	}

	for r := range rows {
		block := raw[r*quant.BlockQ8_KSize:]
		binary.LittleEndian.PutUint32(block[0:4], math.Float32bits(1)) // d = 1.0
		for i := range quant.QK_K {
			block[4+i] = byte(i%7 + r) // qs[i] как int8
		}
	}

	got, err := MatMulVecQ8_K(raw, rows, cols, vec)
	if err != nil {
		t.Fatal(err)
	}

	for r := range rows {
		w, err := quant.DequantBlockQ8_K(raw[r*quant.BlockQ8_KSize:])
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
