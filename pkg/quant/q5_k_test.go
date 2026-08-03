package quant

import (
	"encoding/binary"
	"math"
	"testing"
)

func TestDequantBlockQ5_K(t *testing.T) {
	block := make([]byte, BlockQ5_KSize)
	binary.LittleEndian.PutUint16(block[0:2], 0x3c00) // d = 1.0
	binary.LittleEndian.PutUint16(block[2:4], 0x3800) // dmin = 0.5
	// группы 0..3: sc=2, m=1
	for i := 0; i < 4; i++ {
		block[4+i] = 2
		block[8+i] = 1
	}

	// qh = 0 -> 5-й бит выключен
	// qs: low nibble = i%16, high = (i*3)%16 для первых 32 байт
	qs := block[48:]
	for i := 0; i < 32; i++ {
		lo := byte(i % 16)
		hi := byte((i * 3) % 16)
		qs[i] = lo | (hi << 4)
	}

	out, err := DequantBlockQ5_K(block)
	if err != nil {
		t.Fatal(err)
	}

	// группа 0: d1=2, m1=0.5; q = low nibble
	for l := 0; l < 32; l++ {
		want := float32(2)*float32(l%16) - 0.5
		if math.Abs(float64(out[l]-want)) > 1e-5 {
			t.Fatalf("out[%d]=%v, ожидали %v", l, out[l], want)
		}
	}

	// группа 1: d2=2, m2=0.5; q = high nibble
	for l := 0; l < 32; l++ {
		want := float32(2)*float32((l*3)%16) - 0.5
		if math.Abs(float64(out[32+l]-want)) > 1e-5 {
			t.Fatalf("out[%d]=%v, ожидали %v", 32+l, out[32+l], want)
		}
	}
}

func TestDequantBlockQ5_KHighBit(t *testing.T) {
	block := make([]byte, BlockQ5_KSize)
	binary.LittleEndian.PutUint16(block[0:2], 0x3c00) // d = 1
	binary.LittleEndian.PutUint16(block[2:4], 0x0000) // dmin = 0
	block[4] = 1                                      // sc группы 0
	block[8] = 0                                      // m группы 0
	block[5] = 1
	block[9] = 0
	// u1=1 для первой группы: выставить бит 0 в qh[0]
	block[16] = 1
	qs := block[48:]
	qs[0] = 0x03 // low=3 -> с high bit: 3+16=19

	out, err := DequantBlockQ5_K(block)
	if err != nil {
		t.Fatal(err)
	}

	want := float32(19)
	if math.Abs(float64(out[0]-want)) > 1e-5 {
		t.Fatalf("out[0]=%v, ожидали %v (5-й бит)", out[0], want)
	}
}

func TestBlockQ5_KSizeMatchesFormat(t *testing.T) {
	if BlockQ5_KSize != 176 {
		t.Fatalf("BlockQ5_KSize=%d, ожидали 176", BlockQ5_KSize)
	}
}
