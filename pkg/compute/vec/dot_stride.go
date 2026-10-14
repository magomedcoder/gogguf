package vec

// DotStride sum(weights[t] * v[vOff + t*vStride]) for t in [0,n) (SIMD when available)
var DotStride = dotStridePure

func dotStridePure(weights []float32, v []float32, vOff, vStride, n int) float32 {
	if n <= 0 || n > len(weights) {
		return 0
	}

	var sum float32
	t := 0
	for ; t+3 < n; t += 4 {
		o0 := vOff + t*vStride
		sum += weights[t]*v[o0] + weights[t+1]*v[o0+vStride] + weights[t+2]*v[o0+2*vStride] + weights[t+3]*v[o0+3*vStride]
	}

	for ; t < n; t++ {
		sum += weights[t] * v[vOff+t*vStride]
	}

	return sum
}

// DotStrideBlock dst[i] = sum(weights[t] * v[vOff + t*vStride + i]) for t in [0,n), i in [0,len(dst)) Weighted sum of n rows of v (row stride vStride), SIMD when available
var DotStrideBlock = dotStrideBlockPure

func dotStrideBlockPure(dst, weights, v []float32, vOff, vStride, n int) {
	d := len(dst)
	clear(dst)
	if n <= 0 || d == 0 || n > len(weights) {
		return
	}

	for t := range n {
		w := weights[t]
		row := v[vOff+t*vStride : vOff+t*vStride+d]
		for i, x := range row {
			dst[i] += w * x
		}
	}
}

// blockInRange reports whether the vectorized DotStrideBlock kernels may touch the given memory
func blockInRange(dst, weights, v []float32, vOff, vStride, n int) bool {
	d := len(dst)
	if n <= 0 || d == 0 || n > len(weights) || vStride < d || vOff < 0 {
		return false
	}

	return vOff+(n-1)*vStride+d <= len(v)
}
