package ops

import "math"

// SwiGLU вычисляет silu(gate) * up поэлементно
func SwiGLU(gate, up []float32) []float32 {
	out := make([]float32, len(gate))
	copy(out, gate)
	for i := range out {
		out[i] = SiLU(out[i])
	}
	vecMulInPlace(out, up)

	return out
}

const (
	geluCoefA   = 0.044715
	sqrt2OverPi = 0.7978845608028654 // sqrt(2/π)
)

// GELUInPlace - ggml gelu (tanh-аппроксимация) in-place
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
