//go:build arm64

package vec

// NEON DotStrideBlock kept for benches; not wired - same fixture drift as amd64 AVX2.

func dotStrideBlockNEON(dst, weights, v []float32, vOff, vStride, n int) {
	d := len(dst)
	if d < 4 || !blockInRange(dst, weights, v, vOff, vStride, n) {
		dotStrideBlockPure(dst, weights, v, vOff, vStride, n)
		return
	}

	i := 0
	if d16 := d &^ 15; d16 > 0 {
		dotStrideBlock16NEONAsm(&dst[0], &weights[0], &v[vOff], vStride, n, d16)
		i = d16
	}

	if d4 := d &^ 3; d4 > i {
		dotStrideBlock4NEONAsm(&dst[i], &weights[0], &v[vOff+i], vStride, n, d4-i)
		i = d4
	}

	for ; i < d; i++ {
		var s float32
		for t := range n {
			s += weights[t] * v[vOff+t*vStride+i]
		}
		dst[i] = s
	}
}

// dotStrideBlock16NEONAsm d is a multiple of 16
//
//go:noescape
func dotStrideBlock16NEONAsm(dst, w, v *float32, stride, n, d int)

// dotStrideBlock4NEONAsm d is a multiple of 4
//
//go:noescape
func dotStrideBlock4NEONAsm(dst, w, v *float32, stride, n, d int)
