package quant

import (
	"encoding/binary"
	"fmt"
)

// BlockQ6_KSize - размер блока Q6_K в байтах
const BlockQ6_KSize = 210

// DequantBlockQ6_K деквантизирует один блок Q6_K в 256 float32
func DequantBlockQ6_K(block []byte) ([QK_K]float32, error) {
	if len(block) < BlockQ6_KSize {
		return [QK_K]float32{}, fmt.Errorf("quant: блок Q6_K слишком короткий: %d байт", len(block))
	}

	ql := block[0:128]
	qh := block[128:192]
	scBytes := block[192:208]
	d := FP16ToFP32(binary.LittleEndian.Uint16(block[208:210]))

	var out [QK_K]float32
	y := 0
	qlOff, qhOff, scOff := 0, 0, 0
	for n := 0; n < QK_K; n += 128 {
		for l := range 32 {
			is := l / 16
			q1 := int8((ql[qlOff+l+0]&0xF)|((qh[qhOff+l]>>0)&3)<<4) - 32
			q2 := int8((ql[qlOff+l+32]&0xF)|((qh[qhOff+l]>>2)&3)<<4) - 32
			q3 := int8((ql[qlOff+l+0]>>4)|((qh[qhOff+l]>>4)&3)<<4) - 32
			q4 := int8((ql[qlOff+l+32]>>4)|((qh[qhOff+l]>>6)&3)<<4) - 32
			out[y+l+0] = d * float32(int8(scBytes[scOff+is+0])) * float32(q1)
			out[y+l+32] = d * float32(int8(scBytes[scOff+is+2])) * float32(q2)
			out[y+l+64] = d * float32(int8(scBytes[scOff+is+4])) * float32(q3)
			out[y+l+96] = d * float32(int8(scBytes[scOff+is+6])) * float32(q4)
		}
		y += 128
		qlOff += 64
		qhOff += 32
		scOff += 8
	}

	return out, nil
}

// DequantQ6_KInto деквантизирует буфер Q6_K в dst [n]
func DequantQ6_KInto(dst []float32, data []byte, n int) error {
	if n < 0 {
		return fmt.Errorf("quant: n=%d", n)
	}

	if n == 0 {
		return nil
	}

	if len(dst) < n {
		return fmt.Errorf("quant: dst слишком короткий")
	}

	want := (n + QK_K - 1) / QK_K * BlockQ6_KSize
	if len(data) < want {
		return fmt.Errorf("quant: данных Q6_K недостаточно: нужно %d, есть %d", want, len(data))
	}

	for i := 0; i < n; i += QK_K {
		block, err := DequantBlockQ6_K(data[i/QK_K*BlockQ6_KSize:])
		if err != nil {
			return err
		}

		copy(dst[i:min(i+QK_K, n)], block[:min(QK_K, n-i)])
	}

	return nil
}

// DequantQ6_K деквантизирует буфер Q6_K в n float32
func DequantQ6_K(data []byte, n int) ([]float32, error) {
	out := make([]float32, n)
	if err := DequantQ6_KInto(out, data, n); err != nil {
		return nil, err
	}

	return out, nil
}

// DotBlockQ6_K - dot product блока Q6_K на 256 float32
func DotBlockQ6_K(block []byte, x []float32) (float32, error) {
	if len(x) < QK_K {
		return 0, fmt.Errorf("quant: вектор короче блока Q6_K")
	}

	w, err := DequantBlockQ6_K(block)
	if err != nil {
		return 0, err
	}

	var sum float32
	for i := range QK_K {
		sum += w[i] * x[i]
	}

	return sum, nil
}
