package quant

import (
	"encoding/binary"
	"math"
	"testing"
)

// TestBlockQ2_KSizeMatchesFormat проверяет размер блока Q2_K (llama.cpp: 84 байта)
func TestBlockQ2_KSizeMatchesFormat(t *testing.T) {
	if BlockQ2_KSize != 84 {
		t.Fatalf("BlockQ2_KSize=%d, ожидали 84", BlockQ2_KSize)
	}
}

// TestDequantBlockQ2_K проверяет формулу w = d*scale*q - dmin*min на синтетическом блоке.
func TestDequantBlockQ2_K(t *testing.T) {
	block := make([]byte, BlockQ2_KSize)
	for i := range 16 {
		block[i] = 0x11 // scale=1, min=1
	}
	for i := 16; i < 80; i++ {
		block[i] = 0x55 // qs: 01 01 01 01 -> q=1
	}
	binary.LittleEndian.PutUint16(block[80:82], 0x3c00) // d = 1.0
	binary.LittleEndian.PutUint16(block[82:84], 0x3800) // dmin = 0.5

	out, err := DequantBlockQ2_K(block)
	if err != nil {
		t.Fatal(err)
	}

	// q=1, dl=1*1, ml=0.5*1 -> 1*1 - 0.5 = 0.5
	if math.Abs(float64(out[0]-0.5)) > 1e-5 {
		t.Fatalf("out[0]=%v", out[0])
	}
}

// TestBlockQ3_KSizeMatchesFormat проверяет размер блока Q3_K (llama.cpp: 110 байт).
func TestBlockQ3_KSizeMatchesFormat(t *testing.T) {
	if BlockQ3_KSize != 110 {
		t.Fatalf("BlockQ3_KSize=%d, ожидали 110", BlockQ3_KSize)
	}
}

// TestDequantBlockQ3_K проверяет, что деквант Q3_K даёт конечные значения.
func TestDequantBlockQ3_K(t *testing.T) {
	block := make([]byte, BlockQ3_KSize)
	for i := range 32 {
		block[i] = 0xff // все старшие биты -> без смещения -4
	}

	for i := 32; i < 96; i++ {
		block[i] = 0x55 // младшие 2 бита = 1
	}

	// scales[12]: упакованные 6-битные шкалы (после unpack: scale-32)
	for i := range 12 {
		block[96+i] = 0x21
	}
	binary.LittleEndian.PutUint16(block[108:110], 0x3c00) // d = 1.0

	out, err := DequantBlockQ3_K(block)
	if err != nil {
		t.Fatal(err)
	}

	if math.IsNaN(float64(out[0])) || math.IsInf(float64(out[0]), 0) {
		t.Fatalf("out[0]=%v", out[0])
	}
}

// TestBlockQ8_KSizeMatchesFormat проверяет размер блока Q8_K (llama.cpp: 292 байта).
func TestBlockQ8_KSizeMatchesFormat(t *testing.T) {
	if BlockQ8_KSize != 292 {
		t.Fatalf("BlockQ8_KSize=%d, ожидали 292", BlockQ8_KSize)
	}
}

// TestDequantBlockQ8_K проверяет формулу w = d * qs[i] (qs - int8).
func TestDequantBlockQ8_K(t *testing.T) {
	block := make([]byte, BlockQ8_KSize)
	binary.LittleEndian.PutUint32(block[0:4], math.Float32bits(0.5))
	block[4] = 4    // int8 4 -> 0.5*4 = 2
	block[5] = 0xff // int8 -1 -> -0.5

	out, err := DequantBlockQ8_K(block)
	if err != nil {
		t.Fatal(err)
	}

	if math.Abs(float64(out[0]-2)) > 1e-5 {
		t.Fatalf("out[0]=%v", out[0])
	}

	if math.Abs(float64(out[1]-(-0.5))) > 1e-5 {
		t.Fatalf("out[1]=%v", out[1])
	}
}
