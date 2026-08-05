package quant

import (
	"encoding/binary"
	"fmt"
)

// QK5_1 - число значений в одном блоке Q5_1
const QK5_1 = 32

// BlockQ5_1Size - размер блока Q5_1 (d+m+qh+qs)
const BlockQ5_1Size = 4 + 4 + QK5_1/2

// DequantBlockQ5_1 деквантизирует один блок Q5_1 в 32 float32
// w[j] = d * q + m; q = 5-bit unsigned
func DequantBlockQ5_1(block []byte) ([QK5_1]float32, error) {
	if len(block) < BlockQ5_1Size {
		return [QK5_1]float32{}, fmt.Errorf("quant: блок Q5_1 слишком короткий: %d байт", len(block))
	}

	d := FP16ToFP32(binary.LittleEndian.Uint16(block[0:2]))
	m := FP16ToFP32(binary.LittleEndian.Uint16(block[2:4]))
	qh := binary.LittleEndian.Uint32(block[4:8])
	qs := block[8:24]

	var out [QK5_1]float32
	for j := range QK5_1 / 2 {
		xh0 := ((qh >> uint(j)) << 4) & 0x10
		xh1 := (qh >> uint(j+12)) & 0x10
		q0 := int(uint32(qs[j]&0x0F) | xh0)
		q1 := int(uint32(qs[j]>>4) | xh1)
		out[j] = d*float32(q0) + m
		out[j+16] = d*float32(q1) + m
	}

	return out, nil
}

// DequantQ5_1 деквантизирует буфер Q5_1 в n float32
func DequantQ5_1(data []byte, n int) ([]float32, error) {
	out := make([]float32, n)
	if err := DequantQ5_1Into(out, data, n); err != nil {
		return nil, err
	}

	return out, nil
}

// DequantQ5_1Into деквантизирует буфер Q5_1 в dst [n]
func DequantQ5_1Into(dst []float32, data []byte, n int) error {
	if n < 0 {
		return fmt.Errorf("quant: n=%d", n)
	}

	if n == 0 {
		return nil
	}

	if len(dst) < n {
		return fmt.Errorf("quant: dst слишком короткий")
	}

	want := (n + QK5_1 - 1) / QK5_1 * BlockQ5_1Size
	if len(data) < want {
		return fmt.Errorf("quant: данных Q5_1 недостаточно: нужно %d, есть %d", want, len(data))
	}

	for i := 0; i < n; i += QK5_1 {
		block, err := DequantBlockQ5_1(data[i/QK5_1*BlockQ5_1Size:])
		if err != nil {
			return err
		}

		copy(dst[i:min(i+QK5_1, n)], block[:min(QK5_1, n-i)])
	}

	return nil
}

// DotBlockQ5_1 - скалярное произведение блока Q5_1 на участок float32-вектора
func DotBlockQ5_1(block []byte, x []float32) (float32, error) {
	w, err := DequantBlockQ5_1(block)
	if err != nil {
		return 0, err
	}

	if len(x) < QK5_1 {
		return 0, fmt.Errorf("quant: вектор короче блока Q5_1")
	}

	var sum float32
	for i := range QK5_1 {
		sum += w[i] * x[i]
	}

	return sum, nil
}
