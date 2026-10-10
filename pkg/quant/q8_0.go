package quant

import (
	"encoding/binary"
	"fmt"
)

// QK8_0 - number of values in one Q8_0 block
const QK8_0 = 32

// BlockQ8_0Size - Q8_0 block size in bytes (scale fp16 + 32*int8)
const BlockQ8_0Size = 2 + QK8_0

// DequantBlockQ8_0 dequantizes one Q8_0 block to 32 float32
func DequantBlockQ8_0(block []byte) ([QK8_0]float32, error) {
	if len(block) < BlockQ8_0Size {
		return [QK8_0]float32{}, fmt.Errorf("quant: Q8_0 block too short: %d bytes", len(block))
	}

	d := FP16ToFP32(binary.LittleEndian.Uint16(block[0:2]))
	var out [QK8_0]float32
	for i := range QK8_0 {
		out[i] = d * float32(int8(block[2+i]))
	}

	return out, nil
}

// DequantQ8_0 dequantizes Q8_0 buffer to n float32
func DequantQ8_0(data []byte, n int) ([]float32, error) {
	out := make([]float32, n)
	if err := DequantQ8_0Into(out, data, n); err != nil {
		return nil, err
	}

	return out, nil
}

// DequantQ8_0Into dequantizes Q8_0 buffer to dst [n]
func DequantQ8_0Into(dst []float32, data []byte, n int) error {
	if n < 0 {
		return fmt.Errorf("quant: n=%d", n)
	}

	if n == 0 {
		return nil
	}

	if len(dst) < n {
		return fmt.Errorf("quant: dst too short")
	}

	want := (n + QK8_0 - 1) / QK8_0 * BlockQ8_0Size
	if len(data) < want {
		return fmt.Errorf("quant: insufficient Q8_0 data: need %d, have %d", want, len(data))
	}

	for i := 0; i < n; i += QK8_0 {
		block, err := DequantBlockQ8_0(data[i/QK8_0*BlockQ8_0Size:])
		if err != nil {
			return err
		}
		copy(dst[i:min(i+QK8_0, n)], block[:min(QK8_0, n-i)])
	}

	return nil
}

// DotBlockQ8_0 - dot product of Q8_0 block with float32 vector slice
func DotBlockQ8_0(block []byte, x []float32) (float32, error) {
	if len(block) < BlockQ8_0Size {
		return 0, fmt.Errorf("quant: Q8_0 block too short")
	}

	if len(x) < QK8_0 {
		return 0, fmt.Errorf("quant: vector shorter than Q8_0")
	}

	return dotBlockQ8_0(block, x), nil
}
