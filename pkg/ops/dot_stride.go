package ops

// dotStride: sum(weights[t] * v[vOff + t*vStride]) для t in [0,n)
func dotStride(weights []float32, v []float32, vOff, vStride, n int) float32 {
	if n <= 0 || n > len(weights) {
		return 0
	}

	var sum float32
	off := vOff
	for t := 0; t < n; t++ {
		sum += weights[t] * v[off]
		off += vStride
	}

	return sum
}
