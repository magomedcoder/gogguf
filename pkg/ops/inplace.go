package ops

// RMSNormInto записывает RMS-нормализацию в dst
func RMSNormInto(dst, x, weight []float32, eps float32) error {
	return rmsnormInto(dst, x, weight, eps)
}

// AddInPlace добавляет b к a поэлементно
func AddInPlace(a, b []float32) {
	addInPlace(a, b)
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
