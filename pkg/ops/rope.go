package ops

import "math"

// RoPEScale - RoPE scaling (LongRoPE / linear); zero fields = no scaling
type RoPEScale struct {
	FreqScale  float32   // angle multiplier (1/scaling.factor); 0 -> 1
	AttnFactor float32   // mscale for cos/sin; 0 -> 1
	Factors    []float32 // per dim pair (len = nRot/2); nil -> all 1
}

// ApplyRoPE applies rotary positional embedding GPT-NeoX / Qwen style (dim pairs i and i+headDim/2)
func ApplyRoPE(v []float32, pos int, freqBase float32) {
	ApplyRoPEPartial(v, pos, freqBase, len(v))
}

// ApplyRoPEPartial applies NeoX RoPE only to first nRot dimensions of head (rest unchanged)
func ApplyRoPEPartial(v []float32, pos int, freqBase float32, nRot int) {
	ApplyRoPEPartialScaled(v, pos, freqBase, nRot, RoPEScale{})
}

// ApplyRoPEPartialScaled like ApplyRoPEPartial + LongRoPE factors / freq_scale / attn_factor
func ApplyRoPEPartialScaled(v []float32, pos int, freqBase float32, nRot int, scale RoPEScale) {
	if nRot <= 0 || nRot > len(v) {
		nRot = len(v)
	}

	if nRot%2 != 0 {
		nRot--
	}

	half := nRot / 2
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
		cos, sin := math.Cos(theta)*af, math.Sin(theta)*af
		x0, x1 := float64(v[i]), float64(v[half+i])
		v[i] = float32(x0*cos - x1*sin)
		v[half+i] = float32(x0*sin + x1*cos)
	}
}

func (s RoPEScale) normalized() (freqScale, attnFactor float32, factors []float32) {
	freqScale = s.FreqScale
	if freqScale == 0 {
		freqScale = 1
	}

	attnFactor = s.AttnFactor
	if attnFactor == 0 {
		attnFactor = 1
	}

	return freqScale, attnFactor, s.Factors
}
