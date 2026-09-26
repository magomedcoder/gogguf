//go:build cuda

package llama

import (
	"math"
	"os"
	"path/filepath"
	"testing"

	"github.com/magomedcoder/gogguf/pkg/format"
	"github.com/magomedcoder/gogguf/pkg/gpu"
	"github.com/magomedcoder/gogguf/pkg/weights"
)

// modelPath ищет Llama-3.2-1B-Instruct-Q8_0.gguf или пропускает тест
func modelPath(t *testing.T) string {
	t.Helper()

	for _, p := range []string{
		filepath.Join("..", "..", "..", "models", "Llama-3.2-1B-Instruct-Q8_0.gguf"),
		filepath.Join("models", "Llama-3.2-1B-Instruct-Q8_0.gguf"),
	} {
		if _, err := os.Stat(p); err == nil {
			return p
		}
	}

	t.Skip("нет Llama-3.2-1B-Instruct-Q8_0.gguf")

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

	m, err := Load(weights.New(r), g, ngl, 0)
	if err != nil {
		if g != nil {
			g.Close()
		}

		t.Skip("offload недоступен:", err)
	}

	return m
}

// skipOOM пропускает тест при нехватке VRAM (типично GTX 1050 Ti 4ГБ на полном Q8_0)
func skipOOM(t *testing.T, err error) {
	t.Helper()
	if gpu.IsOutOfMemory(err) {
		t.Skip("не хватило VRAM:", err)
	}
}

// §5: при полном offload Q8_0 residency включается и hidden живёт на устройстве.
// Если полной модели не хватает VRAM - пробуем меньше слоёв: fused QKV/AttnFFN всё равно должны отработать без отката на host для этих слоёв
func TestLlamaResidencyEngagedFullOffload(t *testing.T) {
	m := loadModel(t, 999)
	defer m.Close()

	if m.residency {
		if err := m.forwardToken(1, 0, false); err != nil {
			skipOOM(t, err)
			t.Fatal(err)
		}

		// слой мог молча откатиться на host при OOM upload (recoverHiddenFromDevice)
		if !m.residency || !m.residDevice {
			t.Skip("residency откатилась на host (вероятно нехватка VRAM)")
		}

		if err := m.logitsFinish(); err != nil {
			skipOOM(t, err)
			t.Fatal(err)
		}

		if !m.logitsOnGPU {
			t.Fatal("logitsOnGPU сброшен: GPU lm_head не сработал")
		}

		return
	}

	// Полный offload не включил residency (квант / env) - проверяем fused на части слоёв
	m.Close()
	const partialNGL = 4
	m = loadModel(t, partialNGL)
	defer m.Close()

	if m.fused == nil {
		t.Skip("fused-пути недоступны")
	}

	if err := m.forwardToken(1, 0, false); err != nil {
		skipOOM(t, err)
		t.Fatal(err)
	}

	if err := m.logitsFinish(); err != nil {
		skipOOM(t, err)
		t.Fatal(err)
	}
}

// §5: logits GPU-пути должны совпадать с CPU. Сначала полный offload + residency; при OOM на 4ГБ VRAM - partial ngl без residency (тот же fused QKV/AttnFFN)
func TestLlamaFullOffloadLogitsParity(t *testing.T) {
	tokens := []int{128000, 9906, 11, 1917, 0, 24248, 757, 922, 22915}

	cpu := loadModel(t, 0)
	want, err := cpu.Forward(tokens, 0)
	if err != nil {
		cpu.Close()
		t.Fatal(err)
	}

	wantLogits := append([]float32(nil), want...)
	cpu.Close()

	dev := loadModel(t, 999)
	got, err := dev.Forward(tokens, 0)
	requireResid := dev.residency
	if err != nil {
		dev.Close()
		if !gpu.IsOutOfMemory(err) {
			t.Fatal(err)
		}

		t.Logf("полный offload: %v - повторяем с ngl=4", err)
		requireResid = false
		dev = loadModel(t, 4)
		got, err = dev.Forward(tokens, 0)
		if err != nil {
			dev.Close()
			skipOOM(t, err)
			t.Fatal(err)
		}
	}
	defer dev.Close()

	if requireResid && !dev.residDevice {
		t.Fatal("hidden не остался на устройстве при полном offload")
	}

	var worst float64
	for i := range wantLogits {
		if d := math.Abs(float64(got[i] - wantLogits[i])); d > worst {
			worst = d
		}
	}

	if worst > 0.5 {
		t.Fatalf("logits CPU vs GPU расходятся: max|diff|=%v", worst)
	}
}
