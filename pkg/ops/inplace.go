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
	siluInPlace(gate)
	vecMulInPlace(gate, up)
}

func siluInPlace(x []float32) {
	n := len(x)
	i := 0
	for ; i+3 < n; i += 4 {
		for j := range 4 {
			idx := i + j
			v := x[idx]
			x[idx] = v / (1 + float32(math.Exp(float64(-v))))
		}
	}

	for ; i < n; i++ {
		v := x[i]
		x[i] = v / (1 + float32(math.Exp(float64(-v))))
	}
}
