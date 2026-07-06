//go:build amd64

package ops

func init() {
	if cpuHasAVX2() {
		vecMulInPlace = vecMulInPlaceAVX2
		addInPlace = addInPlaceAVX2
		vectorMax = vectorMaxAVX2
		vecScaleInPlace = vecScaleInPlaceAVX2
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

func vectorMaxAVX2(x []float32) float32 {
	n := len(x)
	if n == 0 {
		return 0
	}

	if n < 8 {
		return vectorMaxPure(x)
	}

	blocks := n &^ 7
	m := vectorMaxAVX2Asm(x, blocks)
	for i := blocks; i < n; i++ {
		if x[i] > m {
			m = x[i]
		}
	}

	return m
}

func vecScaleInPlaceAVX2(x []float32, scale float32) {
	n := len(x)
	if n == 0 {
		return
	}

	i := 0
	if n >= 8 {
		blocks := n &^ 7
		vecScaleInPlaceAVX2Asm(x, scale, blocks)
		i = blocks
	}

	for ; i < n; i++ {
		x[i] *= scale
	}
}

//go:noescape
func vectorMaxAVX2Asm(x []float32, n int) float32

//go:noescape
func vecScaleInPlaceAVX2Asm(x []float32, scale float32, n int)

//go:noescape
func vecMulInPlaceAVX2Asm(a, b []float32, n int)

//go:noescape
func addInPlaceAVX2Asm(a, b []float32, n int)
