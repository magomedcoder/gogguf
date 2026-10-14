package vec

import "math"

// SiLUInPlaceFn x[i] = x[i] * sigmoid(x[i])
var SiLUInPlaceFn = siluInPlacePure

// SiLUInPlace x[i] = x[i] / (1 + exp(-x[i])) (SIMD approx exp when available)
func SiLUInPlace(x []float32) {
	SiLUInPlaceFn(x)
}

func siluInPlacePure(x []float32) {
	for i, v := range x {
		x[i] = v / (1 + float32(math.Exp(float64(-v))))
	}
}
