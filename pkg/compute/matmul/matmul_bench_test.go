package matmul

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
