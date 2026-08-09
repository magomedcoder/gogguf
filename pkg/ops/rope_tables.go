package ops

import "math"

// RoPECosSin заполняет cos/sin таблицы для ApplyRoPEHeads (len >= headDim/2)
func RoPECosSin(cos, sin []float32, headDim, pos int, freqBase float32) {
	RoPECosSinScaled(cos, sin, headDim, pos, freqBase, RoPEScale{})
}

// RoPECosSinScaled как RoPECosSin с LongRoPE / freq_scale / attn_factor
func RoPECosSinScaled(cos, sin []float32, nRot, pos int, freqBase float32, scale RoPEScale) {
	half := nRot / 2
	if len(cos) < half || len(sin) < half {
		return
	}

	freqScale, attnFactor, factors := scale.normalized()
	n := float64(nRot)
	p := float64(pos)
	fb := float64(freqBase)
	fs := float64(freqScale)
	af := float64(attnFactor)

	for i := range half {
		ff := 1.0
		if i < len(factors) && factors[i] != 0 {
			ff = float64(factors[i])
		}

		theta := fs * p * math.Pow(fb, -2*float64(i)/n) / ff
		cos[i] = float32(math.Cos(theta) * af)
		sin[i] = float32(math.Sin(theta) * af)
	}
}
