//go:build arm64

package ops

func init() {
	vecMulInPlace = vecMulInPlaceNEON
	addInPlace = addInPlaceNEON
	vectorMax = vectorMaxNEON
	vecScaleInPlace = vecScaleInPlaceNEON
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

func vectorMaxNEON(x []float32) float32 {
	n := len(x)
	if n == 0 {
		return 0
	}

	if n < 4 {
		return vectorMaxPure(x)
	}

	blocks := n &^ 3
	m := vectorMaxNEONAsm(x, blocks)
	for i := blocks; i < n; i++ {
		if x[i] > m {
			m = x[i]
		}
	}

	return m
}

func vecScaleInPlaceNEON(x []float32, scale float32) {
	n := len(x)
	if n == 0 {
		return
	}

	i := 0
	if n >= 4 {
		blocks := n &^ 3
		vecScaleInPlaceNEONAsm(x, scale, blocks)
		i = blocks
	}

	for ; i < n; i++ {
		x[i] *= scale
	}
}

//go:noescape
func vectorMaxNEONAsm(x []float32, n int) float32

//go:noescape
func vecScaleInPlaceNEONAsm(x []float32, scale float32, n int)

//go:noescape
func vecMulInPlaceNEONAsm(a, b []float32, n int)

//go:noescape
func addInPlaceNEONAsm(a, b []float32, n int)
