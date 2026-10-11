package vec

import "testing"

func BenchmarkDot(b *testing.B) {
	n := 4096
	a := make([]float32, n)
	bvec := make([]float32, n)
	for i := range a {
		a[i] = 0.001
		bvec[i] = 0.002
	}

	b.ResetTimer()

	for range b.N {
		_ = Dot(a, bvec)
	}
}
