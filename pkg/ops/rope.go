package ops

import "math"

// ApplyRoPE применяет rotary positional embedding в стиле GPT-NeoX / Qwen (пары dim i и i+headDim/2)
func ApplyRoPE(v []float32, pos int, freqBase float32) {
	ApplyRoPEPartial(v, pos, freqBase, len(v))
}

// ApplyRoPEPartial применяет NeoX RoPE только к первым nRot измерениям головы (остальное без изменений)
func ApplyRoPEPartial(v []float32, pos int, freqBase float32, nRot int) {
	if nRot <= 0 || nRot > len(v) {
		nRot = len(v)
	}

	if nRot%2 != 0 {
		nRot--
	}

	half := nRot / 2
	for i := range half {
		theta := float64(pos) * math.Pow(float64(freqBase), -2*float64(i)/float64(nRot))
		cos, sin := math.Cos(theta), math.Sin(theta)
		x0, x1 := float64(v[i]), float64(v[half+i])
		v[i] = float32(x0*cos - x1*sin)
		v[half+i] = float32(x0*sin + x1*cos)
	}
}
