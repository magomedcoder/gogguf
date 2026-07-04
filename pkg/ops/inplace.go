package ops

import (
	"fmt"
	"math"
)

// RMSNormInto записывает RMS-нормализацию в dst
func RMSNormInto(dst, x, weight []float32, eps float32) error {
	if len(dst) != len(x) || len(x) != len(weight) {
		return fmt.Errorf("ops: RMSNormInto: несовпадение длин")
	}

	sumSq := dot(x, x)
	scale := float32(1) / float32(math.Sqrt(float64(sumSq/float32(len(x))+eps)))

	for i := range x {
		dst[i] = x[i] * scale * weight[i]
	}

	return nil
}

// AddInPlace добавляет b к a поэлементно
func AddInPlace(a, b []float32) {
	for i := range a {
		a[i] += b[i]
	}
}

// SwiGLUInPlace вычисляет silu(gate)*up, результат в gate
func SwiGLUInPlace(gate, up []float32) {
	for i := range gate {
		gate[i] = SiLU(gate[i]) * up[i]
	}
}
