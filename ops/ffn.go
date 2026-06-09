package ops

// SwiGLU вычисляет silu(gate) * up поэлементно
func SwiGLU(gate, up []float32) []float32 {
	out := make([]float32, len(gate))
	for i := range gate {
		out[i] = SiLU(gate[i]) * up[i]
	}

	return out
}
