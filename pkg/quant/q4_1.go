package quant

import (
	"encoding/binary"
	"fmt"
)

// QK4_1 - число значений в одном блоке Q4_1
const QK4_1 = 32

// BlockQ4_1Size - размер блока Q4_1 (d+m+qs)
const BlockQ4_1Size = 4 + QK4_1/2

// DequantBlockQ4_1 деквантизирует один блок Q4_1 в 32 float32
// w[i] = d * q[i] + m
func DequantBlockQ4_1(block []byte) ([QK4_1]float32, error) {
	if len(block) < BlockQ4_1Size {
		return [QK4_1]float32{}, fmt.Errorf("quant: блок Q4_1 слишком короткий: %d байт", len(block))
	}

	d := FP16ToFP32(binary.LittleEndian.Uint16(block[0:2]))
	m := FP16ToFP32(binary.LittleEndian.Uint16(block[2:4]))
	var out [QK4_1]float32
	for i := range QK4_1 {
		q := (block[4+i/2] >> (4 * (i % 2))) & 0x0F
		out[i] = d*float32(q) + m
	}

	return out, nil
}

// DequantQ4_1 деквантизирует буфер Q4_1 в n float32
func DequantQ4_1(data []byte, n int) ([]float32, error) {
	out := make([]float32, n)
	if err := DequantQ4_1Into(out, data, n); err != nil {
		return nil, err
	}

	return out, nil
}

// DequantQ4_1Into деквантизирует буфер Q4_1 в dst [n]
func DequantQ4_1Into(dst []float32, data []byte, n int) error {
	if n < 0 {
		return fmt.Errorf("quant: n=%d", n)
	}

	if n == 0 {
		return nil
	}

	if len(dst) < n {
		return fmt.Errorf("quant: dst слишком короткий")
	}

	want := (n + QK4_1 - 1) / QK4_1 * BlockQ4_1Size
	if len(data) < want {
		return fmt.Errorf("quant: данных Q4_1 недостаточно: нужно %d, есть %d", want, len(data))
	}

	for i := 0; i < n; i += QK4_1 {
		block, err := DequantBlockQ4_1(data[i/QK4_1*BlockQ4_1Size:])
		if err != nil {
			return err
		}

		copy(dst[i:min(i+QK4_1, n)], block[:min(QK4_1, n-i)])
	}

	return nil
}

// DotBlockQ4_1 - скалярное произведение блока Q4_1 на участок float32-вектора
func DotBlockQ4_1(block []byte, x []float32) (float32, error) {
	if len(block) < BlockQ4_1Size {
		return 0, fmt.Errorf("quant: блок Q4_1 слишком короткий")
	}

	if len(x) < QK4_1 {
		return 0, fmt.Errorf("quant: вектор короче блока Q4_1")
	}

	d := FP16ToFP32(binary.LittleEndian.Uint16(block[0:2]))
	m := FP16ToFP32(binary.LittleEndian.Uint16(block[2:4]))
	var sum float32
	for i := range QK4_1 {
		q := (block[4+i/2] >> (4 * (i % 2))) & 0x0F
		sum += (d*float32(q) + m) * x[i]
	}

	return sum, nil
}
