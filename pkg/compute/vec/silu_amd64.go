//go:build amd64

package vec

import "math"

// AVX2 SiLU (siluInPlaceAVX2 / siluInPlaceAVX2Asm) is kept for benches but not wired into SiLUInPlaceFn: the Cephes-style expf (~1e-7 rel) drifts past golden/fixture abs tol 1e-4 across many FFN layers (and can flip early lm_head greedy).
// Re-enable once the asm matches math.Exp closely enough.
func siluInPlaceAVX2(x []float32) {
	n := len(x)
	i := 0
	if n >= 8 {
		blocks := n &^ 7
		siluInPlaceAVX2Asm(x, blocks)
		i = blocks
	}

	for ; i < n; i++ {
		v := x[i]
		x[i] = v / (1 + float32(math.Exp(float64(-v))))
	}
}

// siluInPlaceAVX2Asm n is a multiple of 8
//
//go:noescape
func siluInPlaceAVX2Asm(x []float32, n int)
