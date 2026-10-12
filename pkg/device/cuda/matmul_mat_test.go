//go:build cuda

package cuda

import (
	"encoding/binary"
	"math"
	"testing"

	"github.com/magomedcoder/gogguf/pkg/compute"
	"github.com/magomedcoder/gogguf/pkg/quant"
)

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
