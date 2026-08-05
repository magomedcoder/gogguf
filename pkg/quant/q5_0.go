package quant

import (
	"encoding/binary"
	"fmt"
)

// QK5_0 - число значений в одном блоке Q5_0
const QK5_0 = 32

// BlockQ5_0Size - размер блока Q5_0 (d+qh+qs)
const BlockQ5_0Size = 2 + 4 + QK5_0/2

// DequantBlockQ5_0 деквантизирует один блок Q5_0 в 32 float32
// w[j] = d * (q - 16); q = 5-bit (qs nibble | qh bit)
func DequantBlockQ5_0(block []byte) ([QK5_0]float32, error) {
	if len(block) < BlockQ5_0Size {
		return [QK5_0]float32{}, fmt.Errorf("quant: блок Q5_0 слишком короткий: %d байт", len(block))
	}

	d := FP16ToFP32(binary.LittleEndian.Uint16(block[0:2]))
	qh := binary.LittleEndian.Uint32(block[2:6])
	qs := block[6:22]

	var out [QK5_0]float32
	for j := range QK5_0 / 2 {
		xh0 := ((qh >> uint(j)) << 4) & 0x10
		xh1 := (qh >> uint(j+12)) & 0x10
		q0 := int((uint32(qs[j]&0x0F) | xh0)) - 16
		q1 := int((uint32(qs[j]>>4) | xh1)) - 16
		out[j] = d * float32(q0)
		out[j+16] = d * float32(q1)
	}

	return out, nil
}

// DequantQ5_0 деквантизирует буфер Q5_0 в n float32
func DequantQ5_0(data []byte, n int) ([]float32, error) {
	out := make([]float32, n)
	if err := DequantQ5_0Into(out, data, n); err != nil {
		return nil, err
	}

	return out, nil
}

// DequantQ5_0Into деквантизирует буфер Q5_0 в dst [n]
func DequantQ5_0Into(dst []float32, data []byte, n int) error {
	if n < 0 {
		return fmt.Errorf("quant: n=%d", n)
	}

	if n == 0 {
		return nil
	}

	if len(dst) < n {
		return fmt.Errorf("quant: dst слишком короткий")
	}

	want := (n + QK5_0 - 1) / QK5_0 * BlockQ5_0Size
	if len(data) < want {
		return fmt.Errorf("quant: данных Q5_0 недостаточно: нужно %d, есть %d", want, len(data))
	}

	for i := 0; i < n; i += QK5_0 {
		block, err := DequantBlockQ5_0(data[i/QK5_0*BlockQ5_0Size:])
		if err != nil {
			return err
		}

		copy(dst[i:min(i+QK5_0, n)], block[:min(QK5_0, n-i)])
	}

	return nil
}

// DotBlockQ5_0 - скалярное произведение блока Q5_0 на участок float32-вектора
func DotBlockQ5_0(block []byte, x []float32) (float32, error) {
	w, err := DequantBlockQ5_0(block)
	if err != nil {
		return 0, err
	}

	if len(x) < QK5_0 {
		return 0, fmt.Errorf("quant: вектор короче блока Q5_0")
	}

	var sum float32
	for i := range QK5_0 {
		sum += w[i] * x[i]
	}

	return sum, nil
}
