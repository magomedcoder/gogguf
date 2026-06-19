//go:build cuda

package cuda

import (
	"encoding/binary"
	"math"
	"testing"

	"github.com/magomedcoder/gguf.go/pkg/ops"
	"github.com/magomedcoder/gguf.go/pkg/quant"
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

func TestMatMulVecQ8_0GPU(t *testing.T) {
	b, err := Open()
	if err != nil {
		t.Skip("CUDA недоступна:", err)
	}
	defer b.Close()

	// 2x32 Q8_0: row0 = 1..32, row1 = 32..1
	rows, cols := 2, 32
	vec := make([]float32, cols)
	for i := range cols {
		vec[i] = 1
	}

	raw := make([]byte, rows*(cols/quant.QK8_0)*quant.BlockQ8_0Size)
	for r := range rows {
		block := raw[r*quant.BlockQ8_0Size:]
		binary.LittleEndian.PutUint16(block[0:2], 0x3c00) // fp16 1.0
		for i := range quant.QK8_0 {
			v := int8(i + 1)
			if r == 1 {
				v = int8(quant.QK8_0 - i)
			}
			block[2+i] = byte(v)
		}
	}

	want, err := ops.MatMulVecQ8_0(raw, rows, cols, vec)
	if err != nil {
		t.Fatal(err)
	}

	got, err := b.MatMulVecQ8_0Cached("test", raw, rows, cols, vec)
	if err != nil {
		t.Fatal(err)
	}

	for i := range want {
		if math.Abs(float64(got[i]-want[i])) > 1e-3 {
			t.Fatalf("строка %d: получили %v, ожидали %v", i, got[i], want[i])
		}
	}
}
