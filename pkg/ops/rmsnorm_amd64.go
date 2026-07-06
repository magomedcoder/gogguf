//go:build amd64

package ops

func init() {
	if cpuHasAVX2() {
		rmsnormScaleMul = rmsnormScaleMulAVX2
	}
}

func rmsnormScaleMulAVX2(dst, x, weight []float32, scale float32) {
	n := len(x)
	if n != len(dst) || n != len(weight) || n == 0 {
		return
	}

	i := 0
	if n >= 8 {
		blocks := n &^ 7
		rmsnormScaleMulAVX2Asm(dst, x, weight, scale, blocks)
		i = blocks
	}

	for ; i < n; i++ {
		dst[i] = x[i] * scale * weight[i]
	}
}

//go:noescape
func rmsnormScaleMulAVX2Asm(dst, x, weight []float32, scale float32, n int)
