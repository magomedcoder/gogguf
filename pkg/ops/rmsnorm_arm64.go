//go:build arm64

package ops

func init() {
	rmsnormScaleMul = rmsnormScaleMulNEON
}

func rmsnormScaleMulNEON(dst, x, weight []float32, scale float32) {
	n := len(x)
	if n != len(dst) || n != len(weight) || n == 0 {
		return
	}

	i := 0
	if n >= 4 {
		blocks := n &^ 3
		rmsnormScaleMulNEONAsm(dst, x, weight, scale, blocks)
		i = blocks
	}

	for ; i < n; i++ {
		dst[i] = x[i] * scale * weight[i]
	}
}

//go:noescape
func rmsnormScaleMulNEONAsm(dst, x, weight []float32, scale float32, n int)
