package ops

// dot - скалярное произведение двух векторов (SIMD при наличии)
var dot = dotPure

func dotPure(a, b []float32) float32 {
	var sum float32
	for i := range a {
		sum += a[i] * b[i]
	}

	return sum
}
