package quant

import (
	"encoding/binary"
	"math"
	"testing"
)

func TestDequantBlockQ8_1(t *testing.T) {
	var block [BlockQ8_1Size]byte
	binary.LittleEndian.PutUint16(block[0:2], 0x3c00) // fp16 1.0
	binary.LittleEndian.PutUint16(block[2:4], 0)      // unused sum
	block[4] = 2
	block[5] = 0xff // -1

	got, err := DequantBlockQ8_1(block[:])
	if err != nil {
		t.Fatal(err)
	}

	if got[0] != 2 || got[1] != -1 {
		t.Fatalf("got[0]=%v got[1]=%v", got[0], got[1])
	}
}

func TestDotBlockQ8_1(t *testing.T) {
	var block [BlockQ8_1Size]byte
	binary.LittleEndian.PutUint16(block[0:2], 0x3c00)
	for i := range QK8_1 {
		block[4+i] = 1
	}

	x := make([]float32, QK8_1)
	for i := range x {
		x[i] = 2
	}

	dot, err := DotBlockQ8_1(block[:], x)
	if err != nil {
		t.Fatal(err)
	}

	want := float32(QK8_1 * 2)
	if math.Abs(float64(dot-want)) > 1e-5 {
		t.Fatalf("dot=%v want %v", dot, want)
	}
}
