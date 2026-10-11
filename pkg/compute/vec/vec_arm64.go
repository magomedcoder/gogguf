//go:build arm64

package vec

func init() {
	MulInPlaceFn = mulInPlaceNEON
	AddInPlaceFn = addInPlaceNEON
	MaxFn = maxNEON
	ScaleInPlaceFn = scaleInPlaceNEON
}

func mulInPlaceNEON(a, b []float32) {
	n := len(a)
	if n != len(b) || n == 0 {
		return
	}

	i := 0
	if n >= 4 {
		blocks := n &^ 3
		mulInPlaceNEONAsm(a, b, blocks)
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

func maxNEON(x []float32) float32 {
	n := len(x)
	if n == 0 {
		return 0
	}

	if n < 4 {
		return maxPure(x)
	}

	blocks := n &^ 3
	m := maxNEONAsm(x, blocks)
	for i := blocks; i < n; i++ {
		if x[i] > m {
			m = x[i]
		}
	}

	return m
}

func scaleInPlaceNEON(x []float32, scale float32) {
	n := len(x)
	if n == 0 {
		return
	}

	i := 0
	if n >= 4 {
		blocks := n &^ 3
		scaleInPlaceNEONAsm(x, scale, blocks)
		i = blocks
	}

	for ; i < n; i++ {
		x[i] *= scale
	}
}

//go:noescape
func maxNEONAsm(x []float32, n int) float32

//go:noescape
func scaleInPlaceNEONAsm(x []float32, scale float32, n int)

//go:noescape
func mulInPlaceNEONAsm(a, b []float32, n int)

//go:noescape
func addInPlaceNEONAsm(a, b []float32, n int)
