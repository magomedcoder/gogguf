package ops

// SwiGLU вычисляет silu(gate) * up поэлементно
func SwiGLU(gate, up []float32) []float32 {
	out := make([]float32, len(gate))
	copy(out, gate)
	for i := range out {
		out[i] = SiLU(out[i])
	}
	vecMulInPlace(out, up)

	return out
}
