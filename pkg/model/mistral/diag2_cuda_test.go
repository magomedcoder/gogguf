//go:build cuda

package mistral

import (
	"fmt"
	"math"
	"testing"

	"github.com/magomedcoder/gogguf/pkg/format"
	"github.com/magomedcoder/gogguf/pkg/gpu"
	"github.com/magomedcoder/gogguf/pkg/ops"
	"github.com/magomedcoder/gogguf/pkg/quant"
	"github.com/magomedcoder/gogguf/pkg/weights"
)

// diagnostics: how many blocks per row break q4_k kernel
func TestDiagQ4KBlocks(t *testing.T) {
	r, err := format.OpenFile(modelPath(t))
	if err != nil {
		t.Fatal(err)
	}

	w := weights.New(r)
	g, err := gpu.OpenCUDA()
	if err != nil {
		t.Skip(err)
	}
	defer g.Close()

	raw, err := w.Raw("blk.0.attn_q.weight")
	if err != nil {
		t.Fatal(err)
	}

	for _, blocks := range []int{1, 2, 4, 16} {
		cols := blocks * quant.QK_K
		rows := 2
		sub := make([]byte, 0, rows*blocks*quant.BlockQ4_KSize)
		for row := 0; row < rows; row++ {
			// source matrix rows have 16 blocks; take first blocks
			off := row * 16 * quant.BlockQ4_KSize
			sub = append(sub, raw[off:off+blocks*quant.BlockQ4_KSize]...)
		}

		vec := make([]float32, cols)
		for i := range vec {
			vec[i] = float32((i%17)-8) * 0.01
		}

		want := make([]float32, rows)
		if err := ops.MatMulVecQ4_KInto(sub, rows, cols, vec, want); err != nil {
			t.Fatal(err)
		}

		got, err := g.MatMulVecQ4_KCached(fmt.Sprintf("diag-q4k-%d", blocks), sub, rows, cols, vec)
		if err != nil {
			t.Fatal(err)
		}

		var worst float64
		for i := range want {
			if d := math.Abs(float64(got[i] - want[i])); d > worst {
				worst = d
			}
		}

		t.Logf("blocks=%d cols=%d: max|diff|=%v cpu=%v gpu=%v", blocks, cols, worst, want, got)
	}
}
