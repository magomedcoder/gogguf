package ops

import "math"

// RMSNormInto записывает RMS-нормализацию в dst
func RMSNormInto(dst, x, weight []float32, eps float32) error {
	return rmsnormInto(dst, x, weight, eps)
}

// AddInPlace добавляет b к a поэлементно
func AddInPlace(a, b []float32) {
	addInPlace(a, b)
}

// ScaleInPlace умножает x на scale in-place
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

// AddBiasInPlace добавляет bias к a (len совпадают)
func AddBiasInPlace(a, bias []float32) {
	if len(bias) == 0 {
		return
	}

	addInPlace(a, bias)
}

// SwiGLUInPlace вычисляет silu(gate)*up, результат в gate
func SwiGLUInPlace(gate, up []float32) {
	for i := range gate {
		gate[i] = SiLU(gate[i])
	}
	vecMulInPlace(gate, up)
}
