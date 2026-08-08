package quant

import (
	"encoding/binary"
	"math"
	"testing"
)

func TestDequantBlockQ4_1(t *testing.T) {
	block := make([]byte, BlockQ4_1Size)
	binary.LittleEndian.PutUint16(block[0:2], 0x3c00) // d = 1.0
	binary.LittleEndian.PutUint16(block[2:4], 0x3800) // m = 0.5
	for i := range 16 {
		block[4+i] = byte(i) | (byte(i+1) << 4)
	}

	out, err := DequantBlockQ4_1(block)
	if err != nil {
		t.Fatal(err)
	}

	if math.Abs(float64(out[0]-(1*0+0.5))) > 1e-5 {
		t.Fatalf("out[0]=%v", out[0])
	}

	if math.Abs(float64(out[1]-(1*1+0.5))) > 1e-5 {
		t.Fatalf("out[1]=%v", out[1])
	}
}

func TestBlockQ4_1SizeMatchesFormat(t *testing.T) {
	if BlockQ4_1Size != 20 {
		t.Fatalf("BlockQ4_1Size=%d, ожидали 20", BlockQ4_1Size)
	}
}

func TestDequantBlockQ5_0(t *testing.T) {
	block := make([]byte, BlockQ5_0Size)
	binary.LittleEndian.PutUint16(block[0:2], 0x3c00) // d = 1.0
	// qh bit0=1 -> q0 high bit; bit16=1 -> q16 high bit (via >>12 & 0x10)
	binary.LittleEndian.PutUint32(block[2:6], (1<<0)|(1<<16))
	block[6] = 0x05 // lo=5 -> 5|16=21; hi=0 -> 0|16=16

	out, err := DequantBlockQ5_0(block)
	if err != nil {
		t.Fatal(err)
	}

	if math.Abs(float64(out[0]-float32(21-16))) > 1e-5 {
		t.Fatalf("out[0]=%v, ожидали 5", out[0])
	}

	if math.Abs(float64(out[16]-float32(16-16))) > 1e-5 {
		t.Fatalf("out[16]=%v, ожидали 0", out[16])
	}
}

func TestBlockQ5_0SizeMatchesFormat(t *testing.T) {
	if BlockQ5_0Size != 22 {
		t.Fatalf("BlockQ5_0Size=%d, ожидали 22", BlockQ5_0Size)
	}
}

func TestDequantBlockQ5_1(t *testing.T) {
	block := make([]byte, BlockQ5_1Size)
	binary.LittleEndian.PutUint16(block[0:2], 0x3c00) // d = 1.0
	binary.LittleEndian.PutUint16(block[2:4], 0x3800) // m = 0.5
	binary.LittleEndian.PutUint32(block[4:8], (1<<0)|(1<<16))
	block[8] = 0x05

	out, err := DequantBlockQ5_1(block)
	if err != nil {
		t.Fatal(err)
	}

	if math.Abs(float64(out[0]-(21*1+0.5))) > 1e-5 {
		t.Fatalf("out[0]=%v", out[0])
	}

	if math.Abs(float64(out[16]-(16*1+0.5))) > 1e-5 {
		t.Fatalf("out[16]=%v", out[16])
	}
}

func TestBlockQ5_1SizeMatchesFormat(t *testing.T) {
	if BlockQ5_1Size != 24 {
		t.Fatalf("BlockQ5_1Size=%d, ожидали 24", BlockQ5_1Size)
	}
}
