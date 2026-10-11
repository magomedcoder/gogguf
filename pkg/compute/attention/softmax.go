package attention

import (
	"github.com/magomedcoder/gogguf/pkg/compute/vec"
	"math"
)

// SoftmaxInPlace applies numerically stable softmax to x
func SoftmaxInPlace(x []float32) {
	if len(x) == 0 {
		return
	}

	maxVal := vec.Max(x)
	sum := softmaxExpInPlace(x, maxVal)
	if sum == 0 {
		return
	}

	vec.ScaleInPlace(x, 1/sum)
}

func softmaxExpInPlace(x []float32, maxVal float32) float32 {
	var sum float32
	n := len(x)
	i := 0
	for ; i+3 < n; i += 4 {
		for j := range 4 {
			e := float32(math.Exp(float64(x[i+j] - maxVal)))
			x[i+j] = e
			sum += e
		}
	}

	for ; i < n; i++ {
		e := float32(math.Exp(float64(x[i] - maxVal)))
		x[i] = e
		sum += e
	}

	return sum
}
