package ops

import (
	"fmt"
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
		return fmt.Errorf("ops: RMSNormInto: несовпадение длин")
	}

	sumSq := dot(x, x)
	scale := float32(1) / float32(math.Sqrt(float64(sumSq/float32(len(x))+eps)))
	rmsnormScaleMul(dst, x, weight, scale)
	return nil
}

func rmsnorm(x, weight []float32, eps float32) ([]float32, error) {
	if len(x) != len(weight) {
		return nil, fmt.Errorf("ops: x и weight разной длины")
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
		return fmt.Errorf("ops: RMSNormBatchInto: batch=%d dim=%d", batch, dim)
	}

	if len(weight) != dim {
		return fmt.Errorf("ops: RMSNormBatchInto: len(weight)=%d, dim=%d", len(weight), dim)
	}

	if len(dst) < batch*dim || len(x) < batch*dim {
		return fmt.Errorf("ops: RMSNormBatchInto: буферы слишком короткие")
	}

	for b := range batch {
		off := b * dim
		if err := rmsnormInto(dst[off:off+dim], x[off:off+dim], weight, eps); err != nil {
			return err
		}
	}

	return nil
}
