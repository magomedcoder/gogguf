//go:build amd64

package ops

func init() {
	if cpuHasAVX2() {
		dot = dotAVX2
	}
}

func dotAVX2(a, b []float32) float32 {
	n := len(a)
	if n != len(b) || n == 0 {
		return 0
	}

	i := 0
	var sum float32
	if n >= 8 {
		blocks := n &^ 7
		sum = dotAVX2Asm(a, b, blocks)
		i = blocks
	}

	for ; i < n; i++ {
		sum += a[i] * b[i]
	}

	return sum
}

//go:noescape
func cpuHasAVX2() bool

//go:noescape
func dotAVX2Asm(a, b []float32, n int) float32
