package ops

import "math"

// SoftmaxInPlace применяет numerically stable softmax к x
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

// Softmax возвращает softmax(x)
func Softmax(x []float32) []float32 {
	out := make([]float32, len(x))
	copy(out, x)
	SoftmaxInPlace(out)
	return out
}
