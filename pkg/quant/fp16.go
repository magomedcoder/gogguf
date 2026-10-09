package quant

import "math"

// FP16ToFP32 converts IEEE 754 half to float32
func FP16ToFP32(h uint16) float32 {
	sign := uint32(h&0x8000) << 16
	exp := (h >> 10) & 0x1f
	mant := uint32(h & 0x3ff)

	switch exp {
	case 0:
		if mant == 0 {
			return math.Float32frombits(sign)
		}
		// subnormal fp16 -> normalize to float32 (like CUDA fp16_to_fp32 / IEEE)
		e := uint32(127 - 15 + 1) // float32 bias − half bias + implicit bit
		for (mant & 0x400) == 0 {
			mant <<= 1
			e--
		}
		mant &= 0x3ff
		return math.Float32frombits(sign | (e << 23) | (mant << 13))
	case 31:
		if mant == 0 {
			if sign != 0 {
				return math.Float32frombits(0xff800000)
			}
			return math.Float32frombits(0x7f800000)
		}
		return math.Float32frombits(sign | 0x7fc00000)
	default:
		return math.Float32frombits(sign | (uint32(exp+112) << 23) | (mant << 13))
	}
}
