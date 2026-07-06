//go:build arm64

package ops

func init() {
	vecMulInPlace = vecMulInPlaceNEON
	addInPlace = addInPlaceNEON
}

func vecMulInPlaceNEON(a, b []float32) {
	n := len(a)
	if n != len(b) || n == 0 {
		return
	}

	i := 0
	if n >= 4 {
		blocks := n &^ 3
		vecMulInPlaceNEONAsm(a, b, blocks)
		i = blocks
	}

	for ; i < n; i++ {
		a[i] *= b[i]
	}
}

func addInPlaceNEON(a, b []float32) {
	n := len(a)
	if n != len(b) || n == 0 {
		return
	}

	i := 0
	if n >= 4 {
		blocks := n &^ 3
		addInPlaceNEONAsm(a, b, blocks)
		i = blocks
	}

	for ; i < n; i++ {
		a[i] += b[i]
	}
}

//go:noescape
func vecMulInPlaceNEONAsm(a, b []float32, n int)

//go:noescape
func addInPlaceNEONAsm(a, b []float32, n int)
