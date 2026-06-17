//go:build cuda

package cuda

import (
	"testing"

	"github.com/magomedcoder/gguf.go/pkg/ops"
)

func TestMatMulVecGPU(t *testing.T) {
	b, err := Open()
	if err != nil {
		t.Skip("CUDA недоступна:", err)
	}
	defer b.Close()

	matrix := []float32{1, 2, 3, 4}
	vec := []float32{1, 0}
	want, err := ops.MatMulVec(matrix, 2, 2, vec)
	if err != nil {
		t.Fatal(err)
	}

	got, err := b.MatMulVec(matrix, 2, 2, vec)
	if err != nil {
		t.Fatal(err)
	}

	for i := range want {
		if got[i] != want[i] {
			t.Fatalf("строка %d: получили %v, ожидали %v", i, got[i], want[i])
		}
	}
}
