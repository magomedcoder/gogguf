package quant

import (
	"encoding/binary"
	"fmt"
)

// QK_K - число значений в K-quant блоке
const QK_K = 256

// BlockQ4_KSize - размер блока Q4_K в байтах
const BlockQ4_KSize = 144

func getScaleMinK4(j int, scales []byte) (uint8, uint8) {
	if j < 4 {
		return scales[j] & 63, scales[j+4] & 63
	}
	d := (scales[j+4] & 0x0F) | ((scales[j-4] >> 6) << 4)
	m := (scales[j+4] >> 4) | ((scales[j] >> 6) << 4)

	return d, m
}

// DequantBlockQ4_K деквантизирует один блок Q4_K в 256 float32
func DequantBlockQ4_K(block []byte) ([QK_K]float32, error) {
	if len(block) < BlockQ4_KSize {
		return [QK_K]float32{}, fmt.Errorf("quant: блок Q4_K слишком короткий: %d байт", len(block))
	}

	d := FP16ToFP32(binary.LittleEndian.Uint16(block[0:2]))
	dmin := FP16ToFP32(binary.LittleEndian.Uint16(block[2:4]))
	scales := block[4:16]
	q := block[16:144]

	var out [QK_K]float32
	is := 0
	y := 0
	for j := 0; j < QK_K; j += 64 {
		sc, m := getScaleMinK4(is+0, scales)
		d1 := d * float32(sc)
		m1 := dmin * float32(m)
		sc, m = getScaleMinK4(is+1, scales)
		d2 := d * float32(sc)
		m2 := dmin * float32(m)

		for l := range 32 {
			out[y] = d1*float32(q[l]&0x0F) - m1
			y++
		}

		for l := range 32 {
			out[y] = d2*float32(q[l]>>4) - m2
			y++
		}

		q = q[32:]
		is += 2
	}

	return out, nil
}

// DequantQ4_K деквантизирует буфер Q4_K в n float32
func DequantQ4_K(data []byte, n int) ([]float32, error) {
	out := make([]float32, n)
	if err := DequantQ4_KInto(out, data, n); err != nil {
		return nil, err
	}

	return out, nil
}

// DequantQ4_KInto деквантизирует буфер Q4_K в dst [n]
func DequantQ4_KInto(dst []float32, data []byte, n int) error {
	if n < 0 {
		return fmt.Errorf("quant: n=%d", n)
	}

	if n == 0 {
		return nil
	}

	if len(dst) < n {
		return fmt.Errorf("quant: dst слишком короткий")
	}

	want := (n + QK_K - 1) / QK_K * BlockQ4_KSize
	if len(data) < want {
		return fmt.Errorf("quant: данных Q4_K недостаточно: нужно %d, есть %d", want, len(data))
	}

	for i := 0; i < n; i += QK_K {
		block, err := DequantBlockQ4_K(data[i/QK_K*BlockQ4_KSize:])
		if err != nil {
			return err
		}

		copy(dst[i:min(i+QK_K, n)], block[:min(QK_K, n-i)])
	}

	return nil
}

// DotBlockQ4_K - dot product блока Q4_K на 256 float32
func DotBlockQ4_K(block []byte, x []float32) (float32, error) {
	if len(x) < QK_K {
		return 0, fmt.Errorf("quant: вектор короче блока Q4_K")
	}

	w, err := DequantBlockQ4_K(block)
	if err != nil {
		return 0, err
	}

	var sum float32
	for i := range QK_K {
		sum += w[i] * x[i]
	}

	return sum, nil
}
