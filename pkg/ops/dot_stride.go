package ops

// dotStride: sum(weights[t] * v[vOff + t*vStride]) for t in [0,n)
func dotStride(weights []float32, v []float32, vOff, vStride, n int) float32 {
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
