package ops

import "math"

// ApplyRoPENorm применяет RoPE в стиле llama (пары соседних dim: 0-1, 2-3, ...)
func ApplyRoPENorm(v []float32, pos int, freqBase float32) {
	n := len(v)
	half := n / 2
	for i := range half {
		theta := float64(pos) * math.Pow(float64(freqBase), -2*float64(i)/float64(n))
		cos, sin := math.Cos(theta), math.Sin(theta)
		i0, i1 := 2*i, 2*i+1
		x0, x1 := float64(v[i0]), float64(v[i1])
		v[i0] = float32(x0*cos - x1*sin)
		v[i1] = float32(x0*sin + x1*cos)
	}
}

// ApplyRoPEHeadsNorm применяет Llama RoPE к nHeads головам в v
func ApplyRoPEHeadsNorm(v []float32, nHeads, headDim, pos int, freqBase float32) {
	if nHeads <= 0 || headDim <= 0 {
		return
	}

	half := headDim / 2
	if half > maxRoPEPairs {
		for h := range nHeads {
			off := h * headDim
			ApplyRoPENorm(v[off:off+headDim], pos, freqBase)
		}

		return
	}

	var cosTab [maxRoPEPairs]float32
	var sinTab [maxRoPEPairs]float32

	RoPECosSin(cosTab[:half], sinTab[:half], headDim, pos, freqBase)

	for h := range nHeads {
		base := h * headDim
		for i := range half {
			i0, i1 := base+2*i, base+2*i+1
			x0 := float64(v[i0])
			x1 := float64(v[i1])
			c, s := float64(cosTab[i]), float64(sinTab[i])
			v[i0] = float32(x0*c - x1*s)
			v[i1] = float32(x0*s + x1*c)
		}
	}
}
