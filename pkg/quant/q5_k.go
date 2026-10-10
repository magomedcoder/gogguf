package quant

import (
	"encoding/binary"
	"fmt"
)

// BlockQ5_KSize - Q5_K block size in bytes (d+dmin+scales+qh+qs)
const BlockQ5_KSize = 176

// DequantBlockQ5_K dequantizes one Q5_K block to 256 float32
func DequantBlockQ5_K(block []byte) ([QK_K]float32, error) {
	if len(block) < BlockQ5_KSize {
		return [QK_K]float32{}, fmt.Errorf("quant: Q5_K block too short: %d bytes", len(block))
	}

	d := FP16ToFP32(binary.LittleEndian.Uint16(block[0:2]))
	dmin := FP16ToFP32(binary.LittleEndian.Uint16(block[2:4]))
	scales := block[4:16]
	qh := block[16:48]
	ql := block[48:176]

	var out [QK_K]float32
	is := 0
	y := 0
	u1, u2 := byte(1), byte(2)
	for j := 0; j < QK_K; j += 64 {
		sc, m := getScaleMinK4(is+0, scales)
		d1 := d * float32(sc)
		m1 := dmin * float32(m)
		sc, m = getScaleMinK4(is+1, scales)
		d2 := d * float32(sc)
		m2 := dmin * float32(m)

		for l := range 32 {
			q := float32(ql[l]&0x0F) + float32(bitu(qh[l]&u1)*16)
			out[y] = d1*q - m1
			y++
		}

		for l := range 32 {
			q := float32(ql[l]>>4) + float32(bitu(qh[l]&u2)*16)
			out[y] = d2*q - m2
			y++
		}

		ql = ql[32:]
		is += 2
		u1 <<= 2
		u2 <<= 2
	}

	return out, nil
}

func bitu(v byte) int {
	if v != 0 {
		return 1
	}

	return 0
}

// DequantQ5_K dequantizes Q5_K buffer to n float32
func DequantQ5_K(data []byte, n int) ([]float32, error) {
	out := make([]float32, n)
	if err := DequantQ5_KInto(out, data, n); err != nil {
		return nil, err
	}

	return out, nil
}

// DequantQ5_KInto dequantizes Q5_K buffer to dst [n]
func DequantQ5_KInto(dst []float32, data []byte, n int) error {
	if n < 0 {
		return fmt.Errorf("quant: n=%d", n)
	}

	if n == 0 {
		return nil
	}

	if len(dst) < n {
		return fmt.Errorf("quant: dst too short")
	}

	want := (n + QK_K - 1) / QK_K * BlockQ5_KSize
	if len(data) < want {
		return fmt.Errorf("quant: insufficient Q5_K data: need %d, have %d", want, len(data))
	}

	for i := 0; i < n; i += QK_K {
		block, err := DequantBlockQ5_K(data[i/QK_K*BlockQ5_KSize:])
		if err != nil {
			return err
		}

		copy(dst[i:min(i+QK_K, n)], block[:min(QK_K, n-i)])
	}

	return nil
}

// DotBlockQ5_K - dot product of Q5_K block with 256 float32
func DotBlockQ5_K(block []byte, x []float32) (float32, error) {
	if len(x) < QK_K {
		return 0, fmt.Errorf("quant: vector shorter than Q5_K")
	}

	w, err := DequantBlockQ5_K(block)
	if err != nil {
		return 0, err
	}

	var sum float32
	for i := range QK_K {
		sum += w[i] * x[i]
	}

	return sum, nil
}
