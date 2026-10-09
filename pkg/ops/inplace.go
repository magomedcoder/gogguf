package ops

import "math"

// RMSNormInto writes RMS normalization to dst
func RMSNormInto(dst, x, weight []float32, eps float32) error {
	return rmsnormInto(dst, x, weight, eps)
}

// AddInPlace adds b to a element-wise
func AddInPlace(a, b []float32) {
	addInPlace(a, b)
}

// ScaleInPlace multiplies x by scale in-place
func ScaleInPlace(x []float32, scale float32) {
	vecScaleInPlace(x, scale)
}

// SoftcapInPlace: x = softcap * tanh(x / softcap) (Gemma2 logits)
func SoftcapInPlace(x []float32, softcap float32) {
	if softcap == 0 {
		return
	}

	inv := 1 / float64(softcap)
	sc := float64(softcap)
	for i, v := range x {
		x[i] = float32(sc * math.Tanh(float64(v)*inv))
	}
}

// AddBiasInPlace adds bias to a (same length)
func AddBiasInPlace(a, bias []float32) {
	if len(bias) == 0 {
		return
	}

	addInPlace(a, bias)
}

// SwiGLUInPlace computes silu(gate)*up, result in gate
func SwiGLUInPlace(gate, up []float32) {
	for i := range gate {
		gate[i] = SiLU(gate[i])
	}
	vecMulInPlace(gate, up)
}
