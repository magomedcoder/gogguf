package ops

import "testing"

func BenchmarkMatMulVec(b *testing.B) {
	cols, rows := 1024, 512
	matrix := make([]float32, rows*cols)
	vec := make([]float32, cols)
	for i := range matrix {
		matrix[i] = 0.001
	}

	for i := range vec {
		vec[i] = 0.002
	}

	b.ResetTimer()
	for range b.N {
		_, _ = MatMulVec(matrix, rows, cols, vec)
	}
}

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
		_ = dot(a, bvec)
	}
}
