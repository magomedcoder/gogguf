//go:build amd64

package vec

// AVX2 DotStride / DotStrideBlock kernels are retained for benches.
// They are not wired into the hot path: fixtures and golden logits were recorded with the pure DotStride loop (pre-SIMD), and the block kernel's FMA order drifts past abs 1e-4.

func dotStrideAVX2(weights []float32, v []float32, vOff, vStride, n int) float32 {
	if n <= 0 || n > len(weights) {
		return 0
	}

	// contiguous v: plain dot
	if vStride == 1 && vOff >= 0 && vOff+n <= len(v) {
		return dotAVX2(weights[:n], v[vOff:vOff+n])
	}

	// AVX2 gather (VGATHERDPS) was benchmarked and is not faster than the scalar unrolled loop (load-port bound); strided callers should use DotStrideBlock, which vectorizes across the contiguous dimension instead
	return dotStridePure(weights, v, vOff, vStride, n)
}

func dotStrideBlockAVX2(dst, weights, v []float32, vOff, vStride, n int) {
	d := len(dst)
	if d < 8 || !blockInRange(dst, weights, v, vOff, vStride, n) {
		dotStrideBlockPure(dst, weights, v, vOff, vStride, n)
		return
	}

	i := 0
	if d32 := d &^ 31; d32 > 0 {
		dotStrideBlock32AVX2Asm(&dst[0], &weights[0], &v[vOff], vStride, n, d32)
		i = d32
	}

	if d8 := d &^ 7; d8 > i {
		dotStrideBlock8AVX2Asm(&dst[i], &weights[0], &v[vOff+i], vStride, n, d8-i)
		i = d8
	}

	for ; i < d; i++ {
		var s float32
		for t := range n {
			s += weights[t] * v[vOff+t*vStride+i]
		}
		dst[i] = s
	}
}

// dotStrideBlock32AVX2Asm d is a multiple of 32
//
//go:noescape
func dotStrideBlock32AVX2Asm(dst, w, v *float32, stride, n, d int)

// dotStrideBlock8AVX2Asm d is a multiple of 8
//
//go:noescape
func dotStrideBlock8AVX2Asm(dst, w, v *float32, stride, n, d int)
