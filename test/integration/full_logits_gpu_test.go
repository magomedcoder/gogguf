//go:build cuda && integration

package integration

import (
	"os"
	"strconv"
	"testing"

	"github.com/magomedcoder/gogguf"
	"github.com/magomedcoder/gogguf/pkg/debug"
)

// TestFullLogitsCPUVsGPU сверяет полный vocab prefill CPU vs CUDA.
// GPU softmax approx (ex2) -> допуск выше raw 1e-4; override: GGUF_GPU_LOGITS_TOL
func TestFullLogitsCPUVsGPU(t *testing.T) {
	model := modelPath(t)

	tol := 1e-2
	if v := os.Getenv("GGUF_GPU_LOGITS_TOL"); v != "" {
		f, err := strconv.ParseFloat(v, 64)
		if err != nil {
			t.Fatal(err)
		}

		tol = f
	}

	cpu, err := forwardPrefill(t, model, "Hello", false, 0)
	if err != nil {
		t.Fatal(err)
	}

	gpu, err := forwardPrefill(t, model, "Hello", false, 999)
	if err != nil {
		t.Skipf("CUDA недоступен: %v", err)
	}

	if len(cpu) != len(gpu) {
		t.Fatalf("vocab cpu=%d gpu=%d", len(cpu), len(gpu))
	}

	aligned := debug.AlignByMax(gpu, cpu)
	st := debug.DiffLogits(aligned, cpu, tol)
	t.Logf("aligned raw: max_abs=%.6g@%d mean=%.6g rmse=%.6g over=%d", st.MaxAbs, st.MaxAbsIdx, st.MeanAbs, st.RMSE, st.OverTol)

	if gogguf.Greedy(cpu) != gogguf.Greedy(gpu) {
		t.Fatalf("greedy cpu=%d gpu=%d", gogguf.Greedy(cpu), gogguf.Greedy(gpu))
	}

	if st.OverTol > 0 {
		// log-softmax часто ближе для sampling
		lpCPU := append([]float32(nil), cpu...)
		lpGPU := append([]float32(nil), gpu...)
		debug.LogSoftmaxInPlace(lpCPU, lpCPU)
		debug.LogSoftmaxInPlace(lpGPU, lpGPU)
		lp := debug.DiffLogits(lpGPU, lpCPU, tol)
		t.Logf("logprob: max_abs=%.6g mean=%.6g over=%d", lp.MaxAbs, lp.MeanAbs, lp.OverTol)
		if lp.OverTol > 0 {
			t.Fatalf("CPU vs GPU logits: aligned max_abs=%.6g over_tol=%d (tol=%g)", st.MaxAbs, st.OverTol, tol)
		}
	}
}

func forwardPrefill(t *testing.T, model, prompt string, chatMode bool, ngl int) ([]float32, error) {
	t.Helper()
	engine, err := gogguf.Load(model, gogguf.LoadOptions{
		NGL: ngl,
	})
	if err != nil {
		return nil, err
	}

	ctx, err := engine.NewContext()
	if err != nil {
		return nil, err
	}

	text := prompt
	if chatMode {
		text, err = gogguf.FormatChatUser(prompt, gogguf.ChatOptions{
			Metadata: engine.Metadata(),
		})
		if err != nil {
			return nil, err
		}
	}

	ids, err := ctx.EncodeForInference(text)
	if err != nil {
		return nil, err
	}

	engine.Model.ResetCache()
	logits, err := engine.Model.Forward(ids, 0)
	if err != nil {
		return nil, err
	}

	out := make([]float32, len(logits))
	copy(out, logits)
	return out, nil
}
