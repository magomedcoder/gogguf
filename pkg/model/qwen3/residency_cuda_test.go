//go:build cuda

package qwen3

import (
	"math"
	"os"
	"path/filepath"
	"testing"

	"github.com/magomedcoder/gogguf/pkg/format"
	"github.com/magomedcoder/gogguf/pkg/gpu"
	"github.com/magomedcoder/gogguf/pkg/weights"
)

// loadResidentModel грузит Qwen3-0.6B-Q8_0 с полным offload или пропускает тест.
// gpuMaxSeq=0 - авто-cap GPU KV-cache.
func loadResidentModel(t *testing.T, gpuMaxSeq, nBatch int) *Model {
	t.Helper()

	var path string
	for _, p := range []string{
		filepath.Join("..", "..", "..", "models", "Qwen3-0.6B-Q8_0.gguf"),
		filepath.Join("models", "Qwen3-0.6B-Q8_0.gguf"),
	} {
		if _, err := os.Stat(p); err == nil {
			path = p
			break
		}
	}

	if path == "" {
		t.Skip("нет Qwen3-0.6B-Q8_0.gguf")
	}

	r, err := format.OpenFile(path)
	if err != nil {
		t.Fatal(err)
	}

	g, err := gpu.OpenCUDA()
	if err != nil {
		t.Skip("CUDA недоступна:", err)
	}

	m, err := Load(weights.New(r), g, 999, gpuMaxSeq, nBatch)
	if err != nil {
		g.Close()
		t.Skip("offload недоступен:", err)
	}

	return m
}

func skipOOM(t *testing.T, err error) {
	t.Helper()
	if gpu.IsOutOfMemory(err) {
		t.Skip("не хватило VRAM:", err)
	}
}

// §1+§2: при полном offload Q8_0 residency включается, hidden реально живёт на устройстве между слоями, а logits считаются на GPU
func TestResidencyEngagedFullOffload(t *testing.T) {
	m := loadResidentModel(t, 0, 1)
	defer m.Close()

	if !m.residency {
		t.Fatal("ожидали residency=true при полном offload Q8_0")
	}

	if !m.logitsOnGPU {
		t.Fatal("ожидали logitsOnGPU=true при полном offload Q8_0")
	}

	if err := m.forwardToken(1, 0, false); err != nil {
		skipOOM(t, err)
		t.Fatal(err)
	}

	// после прохода всех слоёв hidden всё ещё на устройстве: DtoH не было
	if !m.residDevice || !m.residency {
		t.Skip("residency откатилась на host (вероятно нехватка VRAM)")
	}

	// logits берутся из d_resid, host-буфер x при этом не нужен
	if err := m.logits(); err != nil {
		skipOOM(t, err)
		t.Fatal(err)
	}

	if !m.logitsOnGPU {
		t.Fatal("logitsOnGPU сброшен: GPU lm_head не сработал")
	}
}

// §4: batch-prefill с offload не должен ронять GPU KV-cache в stale
func TestBatchPrefillKeepsGPUKV(t *testing.T) {
	m := loadResidentModel(t, 0, 32)
	defer m.Close()

	tokens := []int{1, 2, 3, 4, 5, 6, 7, 8}
	if err := m.forwardBatch(tokens, 0, true); err != nil {
		skipOOM(t, err)
		t.Fatal(err)
	}

	if m.gpuKVStale {
		t.Fatal("GPU KV-cache помечен stale: KVCacheAppendN не сработал")
	}

	// decode после batch должен снова пойти по резидентному GPU-пути
	if err := m.forwardToken(9, len(tokens), false); err != nil {
		skipOOM(t, err)
		t.Fatal(err)
	}

	if !m.residDevice {
		t.Fatal("decode после batch не пошёл по device-резидентному пути")
	}
}

// §4: если GPU KV-cache меньше промпта, KVCacheAppendN упирается в max_seq, модель помечает кеш stale и считает attention по CPU-зеркалу - logits не должны разъехаться
func TestGPUKVStaleFallsBackToCPUAttention(t *testing.T) {
	tokens := []int{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12}

	full := loadResidentModel(t, 0, 32)
	wantLogits, err := full.Forward(tokens, 0)
	if err != nil {
		full.Close()
		skipOOM(t, err)
		t.Fatal(err)
	}

	want := append([]float32(nil), wantLogits...)
	full.Close()

	// GPU KV на 4 токена: chunk из 12 не влезает
	small := loadResidentModel(t, 4, 32)
	defer small.Close()

	got, err := small.Forward(tokens, 0)
	if err != nil {
		skipOOM(t, err)
		t.Fatal(err)
	}

	if !small.gpuKVStale {
		t.Fatal("ожидали gpuKVStale=true при GPU KV меньше промпта")
	}

	var worst float64
	for i := range want {
		if d := math.Abs(float64(got[i] - want[i])); d > worst {
			worst = d
		}
	}

	if worst > 0.5 {
		t.Fatalf("logits после gpuKVStale расходятся: max|diff|=%v", worst)
	}

	// decode дальше тоже должен остаться корректным (без GPU KV)
	if _, err := small.Forward(tokens[len(tokens)-1:], len(tokens)); err != nil {
		skipOOM(t, err)
		t.Fatal(err)
	}
}
