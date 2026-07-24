//go:build cuda && integration

package integration

import (
	"os"
	"strconv"
	"testing"

	"github.com/magomedcoder/gogguf"
	"github.com/magomedcoder/gogguf/pkg/debug"
)

// TestFullLayersCPUVsGPU сверяет embed + hidden по слоям CPU vs CUDA.
// GPU approx (softmax ex2, residency) накапливает drift к последнему слою (~0.3 abs);
// override: GGUF_GPU_LAYERS_TOL (default 0.35).
func TestFullLayersCPUVsGPU(t *testing.T) {
	model := modelPath(t)

	tol := 0.35
	if v := os.Getenv("GGUF_GPU_LAYERS_TOL"); v != "" {
		f, err := strconv.ParseFloat(v, 64)
		if err != nil {
			t.Fatal(err)
		}
		tol = f
	}

	cpuEng, err := gogguf.Load(model, gogguf.LoadOptions{
		NGL: 0,
	})
	if err != nil {
		t.Fatal(err)
	}

	gpuEng, err := gogguf.Load(model, gogguf.LoadOptions{
		NGL: 999,
	})
	if err != nil {
		t.Skipf("CUDA недоступен: %v", err)
	}

	ctx, err := cpuEng.NewContext()
	if err != nil {
		t.Fatal(err)
	}

	ids, err := ctx.EncodeForInference("Hello")
	if err != nil {
		t.Fatal(err)
	}

	cpu := collectLayerHidden(t, cpuEng, ids)
	gpu := collectLayerHidden(t, gpuEng, ids)

	embed, layers, err := debug.DiffLayers(gpu, cpu, tol)
	if err != nil {
		t.Fatal(err)
	}

	t.Logf("embed: max_abs=%.6g@%d mean=%.6g over=%d", embed.MaxAbs, embed.MaxAbsIdx, embed.MeanAbs, embed.OverTol)

	var worst float64
	var worstI, totalOver int
	for i, st := range layers {
		if st.MaxAbs > worst {
			worst = st.MaxAbs
			worstI = i
		}

		totalOver += st.OverTol
		if st.OverTol > 0 {
			t.Logf("layer[%d]: max_abs=%.6g@%d over=%d", i, st.MaxAbs, st.MaxAbsIdx, st.OverTol)
		}
	}

	t.Logf("layers: n=%d worst_max_abs=%.6g@layer%d total_over=%d (tol=%g)", len(layers), worst, worstI, totalOver, tol)

	if embed.OverTol > 0 || totalOver > 0 {
		t.Fatalf("CPU vs GPU layer hidden: embed_over=%d layers_over=%d (tol=%g)", embed.OverTol, totalOver, tol)
	}
}
