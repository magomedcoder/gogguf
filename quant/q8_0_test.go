package quant

import (
	"encoding/binary"
	"math"
	"testing"
)

func TestFP16ToFP32(t *testing.T) {
	tests := []struct {
		h    uint16
		want float32
	}{
		{0x0000, 0},
		{0x3c00, 1.0},
		{0xbc00, -1.0},
	}
	for _, tt := range tests {
		got := FP16ToFP32(tt.h)
		if math.Abs(float64(got-tt.want)) > 1e-5 {
			t.Errorf("FP16ToFP32(0x%04x) = %v, want %v", tt.h, got, tt.want)
		}
	}
}

func TestDequantBlockQ8_0(t *testing.T) {
	var block [BlockQ8_0Size]byte
	binary.LittleEndian.PutUint16(block[0:2], 0x3c00)
	for i := 0; i < QK8_0; i++ {
		block[2+i] = byte(int8(i + 1))
	}

	got, err := DequantBlockQ8_0(block[:])
	if err != nil {
		t.Fatal(err)
	}

	for i := 0; i < QK8_0; i++ {
		if got[i] != float32(i+1) {
			t.Fatalf("[%d] = %v, want %v", i, got[i], i+1)
		}
	}
}

func TestDotBlockQ8_0(t *testing.T) {
	var block [BlockQ8_0Size]byte
	binary.LittleEndian.PutUint16(block[0:2], 0x3c00)
	for i := 0; i < QK8_0; i++ {
		block[2+i] = 1
	}

	vec := make([]float32, QK8_0)
	for i := range vec {
		vec[i] = 2
	}

	dot, err := DotBlockQ8_0(block[:], vec)
	if err != nil {
		t.Fatal(err)
	}

	if dot != 64 {
		t.Fatalf("dot = %v, want 64", dot)
	}
}

func TestDequantQ8_0PartialBlock(t *testing.T) {
	var block [BlockQ8_0Size]byte
	binary.LittleEndian.PutUint16(block[0:2], 0x3c00)
	block[2] = 7
	block[3] = 3

	out, err := DequantQ8_0(block[:], 2)
	if err != nil {
		t.Fatal(err)
	}

	if len(out) != 2 || out[0] != 7 || out[1] != 3 {
		t.Fatalf("got %v", out)
	}
}

func TestDequantQ8_0TwoRows(t *testing.T) {
	raw := make([]byte, 2*BlockQ8_0Size)
	for r := 0; r < 2; r++ {
		off := r * BlockQ8_0Size
		binary.LittleEndian.PutUint16(raw[off:off+2], 0x3c00)
		for i := 0; i < QK8_0; i++ {
			raw[off+2+i] = byte(int8(r + 1))
		}
	}

	out, err := DequantQ8_0(raw, 64)
	if err != nil {
		t.Fatal(err)
	}

	for i := 0; i < 32; i++ {
		if out[i] != 1 {
			t.Fatalf("row0[%d]=%v", i, out[i])
		}
	}

	for i := 32; i < 64; i++ {
		if out[i] != 2 {
			t.Fatalf("row1[%d]=%v", i, out[i])
		}
	}
}
