package ops

import (
	"fmt"
	"math"
)

// LayerNormInto: (x - mean) / sqrt(var + eps) * weight + bias -> dst
func LayerNormInto(dst, x, weight, bias []float32, eps float32) error {
	n := len(x)
	if n == 0 || len(dst) != n || len(weight) != n || len(bias) != n {
		return fmt.Errorf("ops: LayerNormInto: несовпадение длин")
	}

	var sum float64
	for _, v := range x {
		sum += float64(v)
	}
	mean := sum / float64(n)

	var sumSq float64
	for _, v := range x {
		d := float64(v) - mean
		sumSq += d * d
	}
	inv := 1.0 / math.Sqrt(sumSq/float64(n)+float64(eps))

	for i := range x {
		dst[i] = float32((float64(x[i])-mean)*inv)*weight[i] + bias[i]
	}

	return nil
}
