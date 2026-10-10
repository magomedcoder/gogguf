package quant

import (
	"encoding/binary"
	"fmt"
)

// QK4_0 - number of values in one Q4_0 block
const QK4_0 = 32

// BlockQ4_0Size - Q4_0 block size in bytes (scale fp16 + 16*nibble pairs)
const BlockQ4_0Size = 2 + QK4_0/2

// DequantBlockQ4_0 dequantizes one Q4_0 block to 32 float32
func DequantBlockQ4_0(block []byte) ([QK4_0]float32, error) {
	if len(block) < BlockQ4_0Size {
		return [QK4_0]float32{}, fmt.Errorf("quant: Q4_0 block too short: %d bytes", len(block))
	}

	d := FP16ToFP32(binary.LittleEndian.Uint16(block[0:2]))
	var out [QK4_0]float32
	for i := range QK4_0 {
		q := (block[2+i/2] >> (4 * (i % 2))) & 0x0F
		out[i] = d * float32(int(q)-8)
	}

	return out, nil
}

// DequantQ4_0 dequantizes Q4_0 buffer to n float32
func DequantQ4_0(data []byte, n int) ([]float32, error) {
	out := make([]float32, n)
	if err := DequantQ4_0Into(out, data, n); err != nil {
		return nil, err
	}

	return out, nil
}

// DequantQ4_0Into dequantizes Q4_0 buffer to dst [n]
func DequantQ4_0Into(dst []float32, data []byte, n int) error {
	if n < 0 {
		return fmt.Errorf("quant: n=%d", n)
	}

	if n == 0 {
		return nil
	}

	if len(dst) < n {
		return fmt.Errorf("quant: dst too short")
	}

	want := (n + QK4_0 - 1) / QK4_0 * BlockQ4_0Size
	if len(data) < want {
		return fmt.Errorf("quant: insufficient Q4_0 data: need %d, have %d", want, len(data))
	}

	for i := 0; i < n; i += QK4_0 {
		block, err := DequantBlockQ4_0(data[i/QK4_0*BlockQ4_0Size:])
		if err != nil {
			return err
		}

		copy(dst[i:min(i+QK4_0, n)], block[:min(QK4_0, n-i)])
	}

	return nil
}

// DotBlockQ4_0 - dot product of Q4_0 block with float32 vector slice
func DotBlockQ4_0(block []byte, x []float32) (float32, error) {
	if len(block) < BlockQ4_0Size {
		return 0, fmt.Errorf("quant: Q4_0 block too short")
	}

	if len(x) < QK4_0 {
		return 0, fmt.Errorf("quant: vector shorter than Q4_0")
	}

	d := FP16ToFP32(binary.LittleEndian.Uint16(block[0:2]))
	var sum float32
	for i := range QK4_0 {
		q := (block[2+i/2] >> (4 * (i % 2))) & 0x0F
		sum += d * float32(int(q)-8) * x[i]
	}

	return sum, nil
}
