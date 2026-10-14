package ffn

import (
	"math"

	"github.com/magomedcoder/gogguf/pkg/compute/vec"
)

// AddInPlace adds b to a element-wise
func AddInPlace(a, b []float32) {
	vec.AddInPlace(a, b)
}

// ScaleInPlace multiplies x by scale in-place
func ScaleInPlace(x []float32, scale float32) {
	vec.ScaleInPlace(x, scale)
}

// SoftcapInPlace x = softcap * tanh(x / softcap) (Gemma2 logits)
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

	vec.AddInPlace(a, bias)
}

// SwiGLUInPlace computes silu(gate)*up, result in gate
func SwiGLUInPlace(gate, up []float32) {
	siluInPlace(gate)
	vec.MulInPlace(gate, up)
}

func siluInPlace(x []float32) {
	vec.SiLUInPlace(x)
}

// SiLU x * sigmoid(x)
func SiLU(x float32) float32 {
	return x / (1 + float32(math.Exp(float64(-x))))
}
