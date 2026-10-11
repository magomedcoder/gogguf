package norm

import (
	"fmt"
	"github.com/magomedcoder/gogguf/pkg/compute/vec"
	"math"
)

var rmsnormScaleMul = rmsnormScaleMulPure

func rmsnormScaleMulPure(dst, x, weight []float32, scale float32) {
	for i := range x {
		dst[i] = x[i] * scale * weight[i]
	}
}

func rmsnormInto(dst, x, weight []float32, eps float32) error {
	if len(dst) != len(x) || len(x) != len(weight) {
		return fmt.Errorf("compute: RMSNormInto: length mismatch")
	}

	sumSq := vec.Dot(x, x)
	scale := float32(1) / float32(math.Sqrt(float64(sumSq/float32(len(x))+eps)))
	rmsnormScaleMul(dst, x, weight, scale)
	return nil
}

func rmsnorm(x, weight []float32, eps float32) ([]float32, error) {
	if len(x) != len(weight) {
		return nil, fmt.Errorf("compute: x and weight length mismatch")
	}

	out := make([]float32, len(x))
	if err := rmsnormInto(out, x, weight, eps); err != nil {
		return nil, err
	}

	return out, nil
}

// RMSNormBatchInto - RMSNorm over batch of contiguous vectors [batch*dim]
func RMSNormBatchInto(dst, x, weight []float32, eps float32, batch, dim int) error {
	if batch < 1 || dim < 1 {
		return fmt.Errorf("compute: RMSNormBatchInto: batch=%d dim=%d", batch, dim)
	}

	if len(weight) != dim {
		return fmt.Errorf("compute: RMSNormBatchInto: len(weight)=%d, dim=%d", len(weight), dim)
	}

	if len(dst) < batch*dim || len(x) < batch*dim {
		return fmt.Errorf("compute: RMSNormBatchInto: buffers too short")
	}

	for b := range batch {
		off := b * dim
		if err := rmsnormInto(dst[off:off+dim], x[off:off+dim], weight, eps); err != nil {
			return err
		}
	}

	return nil
}

// RMSNormInto writes RMS normalization to dst
func RMSNormInto(dst, x, weight []float32, eps float32) error {
	return rmsnormInto(dst, x, weight, eps)
}

// RMSNorm applies RMS normalization: x * weight / RMS(x)
func RMSNorm(x, weight []float32, eps float32) ([]float32, error) {
	return rmsnorm(x, weight, eps)
}

// VectorRMS returns sqrt(mean(x²)) - metric for comparing hidden states
func VectorRMS(x []float32) float32 {
	if len(x) == 0 {
		return 0
	}

	var sumSq float64
	for _, v := range x {
		sumSq += float64(v) * float64(v)
	}

	return float32(math.Sqrt(sumSq / float64(len(x))))
}
