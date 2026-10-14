//go:build amd64

package vec

import "math"

func init() {
	if HasAVX2() {
		SiLUInPlaceFn = siluInPlaceAVX2
	}
}

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
