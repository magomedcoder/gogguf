//go:build amd64

package ops

func init() {
	if cpuHasAVX2() {
		vecMulInPlace = vecMulInPlaceAVX2
		addInPlace = addInPlaceAVX2
	}
}

func vecMulInPlaceAVX2(a, b []float32) {
	n := len(a)
	if n != len(b) || n == 0 {
		return
	}

	i := 0
	if n >= 8 {
		blocks := n &^ 7
		vecMulInPlaceAVX2Asm(a, b, blocks)
		i = blocks
	}

	for ; i < n; i++ {
		a[i] *= b[i]
	}
}

func addInPlaceAVX2(a, b []float32) {
	n := len(a)
	if n != len(b) || n == 0 {
		return
	}

	i := 0
	if n >= 8 {
		blocks := n &^ 7
		addInPlaceAVX2Asm(a, b, blocks)
		i = blocks
	}

	for ; i < n; i++ {
		a[i] += b[i]
	}
}

//go:noescape
func vecMulInPlaceAVX2Asm(a, b []float32, n int)

//go:noescape
func addInPlaceAVX2Asm(a, b []float32, n int)
