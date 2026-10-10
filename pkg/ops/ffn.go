package ops

import "math"

const (
	geluCoefA   = 0.044715
	sqrt2OverPi = 0.7978845608028654 // sqrt(2/π)
)

// GELUInPlace - ggml gelu (tanh approximation) in-place
func GELUInPlace(x []float32) {
	for i, v := range x {
		xf := float64(v)
		x[i] = float32(0.5 * xf * (1 + math.Tanh(sqrt2OverPi*(xf+geluCoefA*xf*xf*xf))))
	}
}

// GeGLUInPlace: gelu(gate) * up -> gate (Gemma FFN)
func GeGLUInPlace(gate, up []float32) {
	GELUInPlace(gate)
	vecMulInPlace(gate, up)
}
