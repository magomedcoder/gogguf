package quant

import (
	"encoding/binary"
	"fmt"
	"math"
)

// BlockQ2_KSize - размер блока Q2_K (scales+qs+d+dmin)
const BlockQ2_KSize = 84

// DequantBlockQ2_K деквантизирует один блок Q2_K в 256 float32
// layout: scales[16], qs[64], d(fp16), dmin(fp16)
func DequantBlockQ2_K(block []byte) ([QK_K]float32, error) {
	if len(block) < BlockQ2_KSize {
		return [QK_K]float32{}, fmt.Errorf("quant: блок Q2_K слишком короткий: %d байт", len(block))
	}

	scales := block[0:16]
	q := block[16:80]
	d := FP16ToFP32(binary.LittleEndian.Uint16(block[80:82]))
	dmin := FP16ToFP32(binary.LittleEndian.Uint16(block[82:84]))

	var out [QK_K]float32
	y := 0
	is := 0
	for n := 0; n < QK_K; n += 128 {
		shift := 0
		for range 4 {
			sc := scales[is]
			is++
			dl := d * float32(sc&0xF)
			ml := dmin * float32(sc>>4)
			for l := range 16 {
				out[y] = dl*float32((q[l]>>shift)&3) - ml
				y++
			}

			sc = scales[is]
			is++
			dl = d * float32(sc&0xF)
			ml = dmin * float32(sc>>4)
			for l := range 16 {
				out[y] = dl*float32((q[l+16]>>shift)&3) - ml
				y++
			}

			shift += 2
		}
		q = q[32:]
	}

	return out, nil
}

// DequantQ2_K деквантизирует буфер Q2_K в n float32
func DequantQ2_K(data []byte, n int) ([]float32, error) {
	out := make([]float32, n)
	if err := DequantQ2_KInto(out, data, n); err != nil {
		return nil, err
	}

	return out, nil
}

// DequantQ2_KInto деквантизирует буфер Q2_K в dst [n]
func DequantQ2_KInto(dst []float32, data []byte, n int) error {
	if n < 0 {
		return fmt.Errorf("quant: n=%d", n)
	}

	if n == 0 {
		return nil
	}

	if len(dst) < n {
		return fmt.Errorf("quant: dst слишком короткий")
	}

	want := (n + QK_K - 1) / QK_K * BlockQ2_KSize
	if len(data) < want {
		return fmt.Errorf("quant: данных Q2_K недостаточно: нужно %d, есть %d", want, len(data))
	}

	for i := 0; i < n; i += QK_K {
		block, err := DequantBlockQ2_K(data[i/QK_K*BlockQ2_KSize:])
		if err != nil {
			return err
		}

		copy(dst[i:min(i+QK_K, n)], block[:min(QK_K, n-i)])
	}

	return nil
}

// DotBlockQ2_K - dot product блока Q2_K на 256 float32
func DotBlockQ2_K(block []byte, x []float32) (float32, error) {
	if len(x) < QK_K {
		return 0, fmt.Errorf("quant: вектор короче блока Q2_K")
	}

	w, err := DequantBlockQ2_K(block)
	if err != nil {
		return 0, err
	}

	var sum float32
	for i := range QK_K {
		sum += w[i] * x[i]
	}

	return sum, nil
}

// BlockQ3_KSize - размер блока Q3_K (hmask+qs+scales+d)
const BlockQ3_KSize = 110

// DequantBlockQ3_K деквантизирует один блок Q3_K в 256 float32
// layout: hmask[32], qs[64], scales[12], d(fp16)
func DequantBlockQ3_K(block []byte) ([QK_K]float32, error) {
	if len(block) < BlockQ3_KSize {
		return [QK_K]float32{}, fmt.Errorf("quant: блок Q3_K слишком короткий: %d байт", len(block))
	}

	hm := block[0:32]
	q := block[32:96]
	scaleBytes := block[96:108]
	dAll := FP16ToFP32(binary.LittleEndian.Uint16(block[108:110]))

	scales := unpackQ3KScales(scaleBytes)

	var out [QK_K]float32
	y := 0
	is := 0
	m := byte(1)
	for n := 0; n < QK_K; n += 128 {
		shift := 0
		for range 4 {
			dl := dAll * float32(int(scales[is])-32)
			is++
			for l := range 16 {
				bit := byte(0)
				if hm[l]&m == 0 {
					bit = 4
				}
				out[y] = dl * float32(int8((q[l]>>shift)&3)-int8(bit))
				y++
			}

			dl = dAll * float32(int(scales[is])-32)
			is++
			for l := range 16 {
				bit := byte(0)
				if hm[l+16]&m == 0 {
					bit = 4
				}
				out[y] = dl * float32(int8((q[l+16]>>shift)&3)-int8(bit))
				y++
			}

			shift += 2
			m <<= 1
		}
		q = q[32:]
	}

	return out, nil
}

func unpackQ3KScales(scales []byte) [16]int8 {
	const kmask1 = 0x03030303
	const kmask2 = 0x0f0f0f0f

	var aux [4]uint32
	aux[0] = binary.LittleEndian.Uint32(scales[0:4])
	aux[1] = binary.LittleEndian.Uint32(scales[4:8])
	aux[2] = binary.LittleEndian.Uint32(scales[8:12])
	tmp := aux[2]
	aux[2] = ((aux[0] >> 4) & kmask2) | (((tmp >> 4) & kmask1) << 4)
	aux[3] = ((aux[1] >> 4) & kmask2) | (((tmp >> 6) & kmask1) << 4)
	aux[0] = (aux[0] & kmask2) | (((tmp >> 0) & kmask1) << 4)
	aux[1] = (aux[1] & kmask2) | (((tmp >> 2) & kmask1) << 4)

	var out [16]int8
	for i := range 4 {
		v := aux[i]
		out[i*4+0] = int8(v)
		out[i*4+1] = int8(v >> 8)
		out[i*4+2] = int8(v >> 16)
		out[i*4+3] = int8(v >> 24)
	}

	return out
}

// DequantQ3_K деквантизирует буфер Q3_K в n float32
func DequantQ3_K(data []byte, n int) ([]float32, error) {
	out := make([]float32, n)
	if err := DequantQ3_KInto(out, data, n); err != nil {
		return nil, err
	}

	return out, nil
}

// DequantQ3_KInto деквантизирует буфер Q3_K в dst [n]
func DequantQ3_KInto(dst []float32, data []byte, n int) error {
	if n < 0 {
		return fmt.Errorf("quant: n=%d", n)
	}

	if n == 0 {
		return nil
	}

	if len(dst) < n {
		return fmt.Errorf("quant: dst слишком короткий")
	}

	want := (n + QK_K - 1) / QK_K * BlockQ3_KSize
	if len(data) < want {
		return fmt.Errorf("quant: данных Q3_K недостаточно: нужно %d, есть %d", want, len(data))
	}

	for i := 0; i < n; i += QK_K {
		block, err := DequantBlockQ3_K(data[i/QK_K*BlockQ3_KSize:])
		if err != nil {
			return err
		}

		copy(dst[i:min(i+QK_K, n)], block[:min(QK_K, n-i)])
	}

	return nil
}

// DotBlockQ3_K - dot product блока Q3_K на 256 float32
func DotBlockQ3_K(block []byte, x []float32) (float32, error) {
	if len(x) < QK_K {
		return 0, fmt.Errorf("quant: вектор короче блока Q3_K")
	}

	w, err := DequantBlockQ3_K(block)
	if err != nil {
		return 0, err
	}

	var sum float32
	for i := range QK_K {
		sum += w[i] * x[i]
	}

	return sum, nil
}

// BlockQ8_KSize - размер блока Q8_K (d float32 + qs + bsums)
const BlockQ8_KSize = 292

// DequantBlockQ8_K деквантизирует один блок Q8_K в 256 float32
// layout: d(float32), qs[256]int8, bsums[16]int16
func DequantBlockQ8_K(block []byte) ([QK_K]float32, error) {
	if len(block) < BlockQ8_KSize {
		return [QK_K]float32{}, fmt.Errorf("quant: блок Q8_K слишком короткий: %d байт", len(block))
	}

	d := math.Float32frombits(binary.LittleEndian.Uint32(block[0:4]))
	qs := block[4 : 4+QK_K]

	var out [QK_K]float32
	for i := range QK_K {
		out[i] = d * float32(int8(qs[i]))
	}

	return out, nil
}

// DequantQ8_K деквантизирует буфер Q8_K в n float32
func DequantQ8_K(data []byte, n int) ([]float32, error) {
	out := make([]float32, n)
	if err := DequantQ8_KInto(out, data, n); err != nil {
		return nil, err
	}

	return out, nil
}

// DequantQ8_KInto деквантизирует буфер Q8_K в dst [n]
func DequantQ8_KInto(dst []float32, data []byte, n int) error {
	if n < 0 {
		return fmt.Errorf("quant: n=%d", n)
	}

	if n == 0 {
		return nil
	}

	if len(dst) < n {
		return fmt.Errorf("quant: dst слишком короткий")
	}

	want := (n + QK_K - 1) / QK_K * BlockQ8_KSize
	if len(data) < want {
		return fmt.Errorf("quant: данных Q8_K недостаточно: нужно %d, есть %d", want, len(data))
	}

	for i := 0; i < n; i += QK_K {
		block, err := DequantBlockQ8_K(data[i/QK_K*BlockQ8_KSize:])
		if err != nil {
			return err
		}

		copy(dst[i:min(i+QK_K, n)], block[:min(QK_K, n-i)])
	}

	return nil
}

// DotBlockQ8_K - dot product блока Q8_K на 256 float32
func DotBlockQ8_K(block []byte, x []float32) (float32, error) {
	if len(x) < QK_K {
		return 0, fmt.Errorf("quant: вектор короче блока Q8_K")
	}

	w, err := DequantBlockQ8_K(block)
	if err != nil {
		return 0, err
	}

	var sum float32
	for i := range QK_K {
		sum += w[i] * x[i]
	}

	return sum, nil
}
