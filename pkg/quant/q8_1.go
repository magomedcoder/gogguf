package quant

import (
	"encoding/binary"
	"fmt"
)

// QK8_1 - values per Q8_1 block (same as Q8_0)
const QK8_1 = 32

// BlockQ8_1Size - Q8_1 block: fp16 d + fp16 s (precomputed quant sum) + 32*int8
const BlockQ8_1Size = 2 + 2 + QK8_1

// DequantBlockQ8_1 dequantizes one Q8_1 block to 32 float32 (w = d * qs[i]; s unused for FP32 vec)
func DequantBlockQ8_1(block []byte) ([QK8_1]float32, error) {
	if len(block) < BlockQ8_1Size {
		return [QK8_1]float32{}, fmt.Errorf("quant: Q8_1 block too short: %d bytes", len(block))
	}

	d := FP16ToFP32(binary.LittleEndian.Uint16(block[0:2]))
	var out [QK8_1]float32
	for i := range QK8_1 {
		out[i] = d * float32(int8(block[4+i]))
	}

	return out, nil
}

// DequantQ8_1 dequantizes Q8_1 buffer to n float32
func DequantQ8_1(data []byte, n int) ([]float32, error) {
	out := make([]float32, n)
	if err := DequantQ8_1Into(out, data, n); err != nil {
		return nil, err
	}

	return out, nil
}

// DequantQ8_1Into dequantizes Q8_1 buffer to dst [n]
func DequantQ8_1Into(dst []float32, data []byte, n int) error {
	if n < 0 {
		return fmt.Errorf("quant: n=%d", n)
	}

	if n == 0 {
		return nil
	}

	if len(dst) < n {
		return fmt.Errorf("quant: dst too short")
	}

	want := (n + QK8_1 - 1) / QK8_1 * BlockQ8_1Size
	if len(data) < want {
		return fmt.Errorf("quant: insufficient Q8_1 data: need %d, have %d", want, len(data))
	}

	for i := 0; i < n; i += QK8_1 {
		block, err := DequantBlockQ8_1(data[i/QK8_1*BlockQ8_1Size:])
		if err != nil {
			return err
		}
		copy(dst[i:min(i+QK8_1, n)], block[:min(QK8_1, n-i)])
	}

	return nil
}

// DotBlockQ8_1 - dot product of Q8_1 block with float32 vector slice
func DotBlockQ8_1(block []byte, x []float32) (float32, error) {
	if len(block) < BlockQ8_1Size {
		return 0, fmt.Errorf("quant: Q8_1 block too short")
	}

	if len(x) < QK8_1 {
		return 0, fmt.Errorf("quant: vector shorter than Q8_1")
	}

	d := FP16ToFP32(binary.LittleEndian.Uint16(block[0:2]))
	var sum float32
	for i := range QK8_1 {
		sum += d * float32(int8(block[4+i])) * x[i]
	}

	return sum, nil
}
