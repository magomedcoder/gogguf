package ops

import "math"

// RoPEScale - масштабирование RoPE (LongRoPE / linear); нулевые поля = без масштаба
type RoPEScale struct {
	FreqScale  float32   // множитель угла (1/scaling.factor); 0 -> 1
	AttnFactor float32   // mscale для cos/sin; 0 -> 1
	Factors    []float32 // на пару dim (len = nRot/2); nil -> все 1
}

// ApplyRoPE применяет rotary positional embedding в стиле GPT-NeoX / Qwen (пары dim i и i+headDim/2)
func ApplyRoPE(v []float32, pos int, freqBase float32) {
	ApplyRoPEPartial(v, pos, freqBase, len(v))
}

// ApplyRoPEPartial применяет NeoX RoPE только к первым nRot измерениям головы (остальное без изменений)
func ApplyRoPEPartial(v []float32, pos int, freqBase float32, nRot int) {
	ApplyRoPEPartialScaled(v, pos, freqBase, nRot, RoPEScale{})
}

// ApplyRoPEPartialScaled как ApplyRoPEPartial + LongRoPE factors / freq_scale / attn_factor
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
