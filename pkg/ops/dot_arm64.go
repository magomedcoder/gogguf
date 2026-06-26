//go:build arm64

package ops

func init() {
	dot = dotNEON
}

func dotNEON(a, b []float32) float32 {
	n := len(a)
	if n != len(b) || n == 0 {
		return 0
	}

	i := 0
	var sum float32
	if n >= 4 {
		blocks := n &^ 3
		sum = dotNEONAsm(a, b, blocks)
		i = blocks
	}

	for ; i < n; i++ {
		sum += a[i] * b[i]
	}

	return sum
}

//go:noescape
func dotNEONAsm(a, b []float32, n int) float32
