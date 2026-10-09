//go:build cuda

package mistral

import (
	"math"
	"testing"

	"github.com/magomedcoder/gogguf/pkg/format"
	"github.com/magomedcoder/gogguf/pkg/gpu"
	"github.com/magomedcoder/gogguf/pkg/ops"
	"github.com/magomedcoder/gogguf/pkg/weights"
)

// diagnostics: compare GPU matmul kernels to CPU on real layer-0 weights
func TestDiagLayer0Matmuls(t *testing.T) {
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

	for _, name := range []string{
		"blk.0.attn_q.weight",
		"blk.0.attn_k.weight",
		"blk.0.attn_v.weight",
		"blk.0.attn_output.weight",
		"blk.0.ffn_gate.weight",
		"blk.0.ffn_up.weight",
		"blk.0.ffn_down.weight",
	} {
		info, err := w.Info(name)
		if err != nil {
			t.Fatal(err)
		}

		rows := int(info.Dimensions[len(info.Dimensions)-1])
		cols := int(info.Dimensions[0])
		raw, err := w.Raw(name)
		if err != nil {
			t.Fatal(err)
		}

		vec := make([]float32, cols)
		for i := range vec {
			vec[i] = float32((i%17)-8) * 0.01
		}

		want := make([]float32, rows)
		switch info.Type {
		case format.GgmlQ4_K:
			err = ops.MatMulVecQ4_KInto(raw, rows, cols, vec, want)
		case format.GgmlQ5_K:
			err = ops.MatMulVecQ5_KInto(raw, rows, cols, vec, want)
		case format.GgmlQ6_K:
			err = ops.MatMulVecQ6_KInto(raw, rows, cols, vec, want)
		default:
			t.Logf("%s: тип %s пропущен", name, info.Type)
			continue
		}

		if err != nil {
			t.Fatal(err)
		}

		var got []float32
		switch info.Type {
		case format.GgmlQ4_K:
			got, err = g.MatMulVecQ4_KCached(name, raw, rows, cols, vec)
		case format.GgmlQ5_K:
			got, err = g.MatMulVecQ5_KCached(name, raw, rows, cols, vec)
		case format.GgmlQ6_K:
			got, err = g.MatMulVecQ6_KCached(name, raw, rows, cols, vec)
		}

		if err != nil {
			t.Fatal(err)
		}

		// float64 reference from dequantized weights: which is closer - CPU or GPU
		f32, err := w.Floats(name)
		if err != nil {
			t.Fatal(err)
		}

		var worstGPU, worstCPU, maxAbs float64
		for r := 0; r < rows; r++ {
			var acc float64
			row := f32[r*cols : (r+1)*cols]
			for c := 0; c < cols; c++ {
				acc += float64(row[c]) * float64(vec[c])
			}

			if a := math.Abs(acc); a > maxAbs {
				maxAbs = a
			}

			if d := math.Abs(float64(got[r]) - acc); d > worstGPU {
				worstGPU = d
			}

			if d := math.Abs(float64(want[r]) - acc); d > worstCPU {
				worstCPU = d
			}
		}

		var worst float64
		for i := range want {
			if d := math.Abs(float64(got[i] - want[i])); d > worst {
				worst = d
			}
		}

		t.Logf("%s (%s, %dx%d): max|gpu-cpu|=%v max|gpu-f64|=%v max|cpu-f64|=%v max|ref|=%v", name, info.Type, rows, cols, worst, worstGPU, worstCPU, maxAbs)
	}
}
