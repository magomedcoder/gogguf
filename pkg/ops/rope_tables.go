package ops

import "math"

// RoPECosSin заполняет cos/sin таблицы для ApplyRoPEHeads (len >= headDim/2)
func RoPECosSin(cos, sin []float32, headDim, pos int, freqBase float32) {
	half := headDim / 2
	if len(cos) < half || len(sin) < half {
		return
	}

	n := float64(headDim)
	p := float64(pos)
	fb := float64(freqBase)
	for i := range half {
		theta := p * math.Pow(fb, -2*float64(i)/n)
		cos[i] = float32(math.Cos(theta))
		sin[i] = float32(math.Sin(theta))
	}
}
