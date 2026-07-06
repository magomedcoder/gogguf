package ops

import "math"

const maxRoPEPairs = 128

// ApplyRoPEHeads применяет RoPE к nHeads головам в v; sin/cos вычисляются один раз на позицию
func ApplyRoPEHeads(v []float32, nHeads, headDim, pos int, freqBase float32) {
	if nHeads <= 0 || headDim <= 0 {
		return
	}

	half := headDim / 2
	if half > maxRoPEPairs {
		for h := range nHeads {
			off := h * headDim
			ApplyRoPE(v[off:off+headDim], pos, freqBase)
		}
		return
	}

	var cosTab [maxRoPEPairs]float32
	var sinTab [maxRoPEPairs]float32

	n := float64(headDim)
	p := float64(pos)
	fb := float64(freqBase)
	for i := range half {
		theta := p * math.Pow(fb, -2*float64(i)/n)
		cosTab[i] = float32(math.Cos(theta))
		sinTab[i] = float32(math.Sin(theta))
	}

	for h := range nHeads {
		base := h * headDim
		for i := range half {
			x0 := float64(v[base+i])
			x1 := float64(v[base+half+i])
			c, s := float64(cosTab[i]), float64(sinTab[i])
			v[base+i] = float32(x0*c - x1*s)
			v[base+half+i] = float32(x0*s + x1*c)
		}
	}
}
