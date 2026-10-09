//go:build cuda

package mistral

import (
	"math"
	"os"
	"path/filepath"
	"testing"

	"github.com/magomedcoder/gogguf/pkg/format"
	"github.com/magomedcoder/gogguf/pkg/gpu"
	"github.com/magomedcoder/gogguf/pkg/weights"
)

// modelPath finds Mistral-7B-Instruct-v0.2-Q4_K_M.gguf or skips the test
func modelPath(t *testing.T) string {
	t.Helper()

	for _, p := range []string{
		filepath.Join("..", "..", "..", "models", "Mistral-7B-Instruct-v0.2-Q4_K_M.gguf"),
		filepath.Join("models", "Mistral-7B-Instruct-v0.2-Q4_K_M.gguf"),
	} {
		if _, err := os.Stat(p); err == nil {
			return p
		}
	}

	t.Skip("нет Mistral-7B-Instruct-v0.2-Q4_K_M.gguf")

	return ""
}

func loadModel(t *testing.T, ngl int) *Model {
	t.Helper()

	r, err := format.OpenFile(modelPath(t))
	if err != nil {
		t.Fatal(err)
	}

	var g gpu.Backend
	if ngl > 0 {
		if g, err = gpu.OpenCUDA(); err != nil {
			t.Skip("CUDA недоступна:", err)
		}
	}

	// GGUF Mistral-7B-Instruct is tagged as general.architecture=llama
	m, err := LoadLlamaMeta(weights.New(r), g, ngl, 0)
	if err != nil {
		if g != nil {
			g.Close()
		}

		t.Skip("offload недоступен:", err)
	}

	return m
}

// §5: fused QKV Mistral (NeoX RoPE, no QK-norm) on Q4_K must match CPU logits. Few layers only: full 7B does not fit 4 GB VRAM
func TestMistralPartialOffloadLogitsParity(t *testing.T) {
	const ngl = 4

	tokens := []int{1, 22557, 28725, 1526, 28808}

	dev := loadModel(t, ngl)
	if dev.fused == nil {
		dev.Close()
		t.Skip("fused-пути недоступны")
	}

	got, err := dev.Forward(tokens, 0)
	if err != nil {
		dev.Close()
		if gpu.IsOutOfMemory(err) {
			t.Skip("не хватило VRAM:", err)
		}
		t.Fatal(err)
	}

	gotLogits := append([]float32(nil), got...)
	dev.Close()

	cpu := loadModel(t, 0)
	defer cpu.Close()

	want, err := cpu.Forward(tokens, 0)
	if err != nil {
		t.Fatal(err)
	}

	var worst float64
	var scale float64
	bestCPU, bestGPU := 0, 0
	for i := range want {
		if d := math.Abs(float64(gotLogits[i] - want[i])); d > worst {
			worst = d
		}

		if a := math.Abs(float64(want[i])); a > scale {
			scale = a
		}

		if want[i] > want[bestCPU] {
			bestCPU = i
		}

		if gotLogits[i] > gotLogits[bestGPU] {
			bestGPU = i
		}
	}

	t.Logf("max|diff|=%v max|logit|=%v argmax cpu=%d gpu=%d", worst, scale, bestCPU, bestGPU)

	if worst > 0.5 {
		t.Fatalf("logits CPU vs GPU расходятся: max|diff|=%v", worst)
	}
}
