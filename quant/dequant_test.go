package quant

import (
	"encoding/binary"
	"math"
	"testing"
)

func TestToFloat32F32(t *testing.T) {
	buf := make([]byte, 8)
	binary.LittleEndian.PutUint32(buf[0:4], math.Float32bits(1.5))
	binary.LittleEndian.PutUint32(buf[4:8], math.Float32bits(-2.25))

	out, err := dequantF32(buf, 2)
	if err != nil {
		t.Fatal(err)
	}

	if out[0] != 1.5 || out[1] != -2.25 {
		t.Fatalf("got %v", out)
	}
}

func TestToFloat32F16(t *testing.T) {
	buf := make([]byte, 4)
	binary.LittleEndian.PutUint16(buf[0:2], 0x3c00)
	binary.LittleEndian.PutUint16(buf[2:4], 0xbc00)

	out, err := dequantF16(buf, 2)
	if err != nil {
		t.Fatal(err)
	}
	
	if out[0] != 1 || out[1] != -1 {
		t.Fatalf("got %v", out)
	}
}
