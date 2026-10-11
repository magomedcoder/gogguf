//go:build amd64

package vec

func init() {
	if HasAVX2() {
		MulInPlaceFn = mulInPlaceAVX2
		AddInPlaceFn = addInPlaceAVX2
		MaxFn = maxAVX2
		ScaleInPlaceFn = scaleInPlaceAVX2
	}
}

func mulInPlaceAVX2(a, b []float32) {
	n := len(a)
	if n != len(b) || n == 0 {
		return
	}

	i := 0
	if n >= 8 {
		blocks := n &^ 7
		mulInPlaceAVX2Asm(a, b, blocks)
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

func maxAVX2(x []float32) float32 {
	n := len(x)
	if n == 0 {
		return 0
	}

	if n < 8 {
		return maxPure(x)
	}

	blocks := n &^ 7
	m := maxAVX2Asm(x, blocks)
	for i := blocks; i < n; i++ {
		if x[i] > m {
			m = x[i]
		}
	}

	return m
}

func scaleInPlaceAVX2(x []float32, scale float32) {
	n := len(x)
	if n == 0 {
		return
	}

	i := 0
	if n >= 8 {
		blocks := n &^ 7
		scaleInPlaceAVX2Asm(x, scale, blocks)
		i = blocks
	}

	for ; i < n; i++ {
		x[i] *= scale
	}
}

//go:noescape
func maxAVX2Asm(x []float32, n int) float32

//go:noescape
func scaleInPlaceAVX2Asm(x []float32, scale float32, n int)

//go:noescape
func mulInPlaceAVX2Asm(a, b []float32, n int)

//go:noescape
func addInPlaceAVX2Asm(a, b []float32, n int)
