package ops

import "math"

// SoftmaxInPlace applies numerically stable softmax to x
func SoftmaxInPlace(x []float32) {
	if len(x) == 0 {
		return
	}

	maxVal := vectorMax(x)

	var sum float64
	for i, v := range x {
		e := math.Exp(float64(v - maxVal))
		x[i] = float32(e)
		sum += e
	}

	vecScaleInPlace(x, float32(1/sum))
}

// Softmax returns softmax(x)
func Softmax(x []float32) []float32 {
	out := make([]float32, len(x))
	copy(out, x)
	SoftmaxInPlace(out)
	return out
}
