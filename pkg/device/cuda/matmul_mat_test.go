//go:build cuda

package cuda

import (
	"encoding/binary"
	"math"
	"testing"

	"github.com/magomedcoder/gogguf/pkg/compute"
	"github.com/magomedcoder/gogguf/pkg/quant"
)

func TestMatMulMatF32GPU(t *testing.T) {
	b, err := Open()
	if err != nil {
		t.Skip("CUDA unavailable:", err)
	}
	defer b.Close()

	rows, cols, batch := 5, 32, 8
	matrix := make([]float32, rows*cols)
	for i := range matrix {
		matrix[i] = float32((i%11)+1) * 0.01
	}

	x := make([]float32, batch*cols)
	for i := range x {
		x[i] = float32((i%7)+1) * 0.1
	}

	want := make([]float32, batch*rows)
	for bi := range batch {
		for r := range rows {
			var sum float32
			for c := range cols {
				sum += matrix[r*cols+c] * x[bi*cols+c]
			}

			want[bi*rows+r] = sum
		}
	}

	got := make([]float32, batch*rows)
	if err := b.MatMulMatCachedInto("mat_f32", matrix, rows, cols, x, batch, got); err != nil {
		t.Fatal(err)
	}

	if !b.hasMatF32 {
		t.Fatal("expected matmul_mat_f32 kernel loaded")
	}

	for i := range want {
		if math.Abs(float64(got[i]-want[i])) > 1e-4 {
			t.Fatalf("i=%d got=%v want=%v", i, got[i], want[i])
		}
	}
}

func TestMatMulMatQ8_0GPU(t *testing.T) {
	b, err := Open()
	if err != nil {
		t.Skip("CUDA unavailable:", err)
	}
	defer b.Close()

	rows, cols, batch := 4, 64, 8
	raw := make([]byte, rows*(cols/quant.QK8_0)*quant.BlockQ8_0Size)
	for r := range rows {
		block := raw[r*quant.BlockQ8_0Size:]
		binary.LittleEndian.PutUint16(block[0:2], 0x3c00)
		for i := range quant.QK8_0 {
			block[2+i] = byte(int8(i + 1 + r))
		}
	}

	x := make([]float32, batch*cols)
	for i := range x {
		x[i] = float32((i%7)+1) * 0.1
	}

	want := make([]float32, batch*rows)
	if err := compute.MatMulMatQ8_0Into(raw, rows, cols, x, batch, want); err != nil {
		t.Fatal(err)
	}

	got := make([]float32, batch*rows)
	if err := b.MatMulMatQ8_0CachedInto("mat_q8", raw, rows, cols, x, batch, got); err != nil {
		t.Fatal(err)
	}

	for i := range want {
		if math.Abs(float64(got[i]-want[i])) > 1e-3 {
			t.Fatalf("i=%d got=%v want=%v", i, got[i], want[i])
		}
	}
}
