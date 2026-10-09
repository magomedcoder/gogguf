package ops

// dot - dot product of two vectors (SIMD when available)
var dot = dotPure

func dotPure(a, b []float32) float32 {
	var sum float32
	for i := range a {
		sum += a[i] * b[i]
	}

	return sum
}
