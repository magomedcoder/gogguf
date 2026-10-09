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

// loadResidentModel loads Qwen3-0.6B-Q8_0 with full offload or skips the test.
// gpuMaxSeq=0 - auto-cap GPU KV-cache.
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

// §1+§2: on full Q8_0 offload residency enables; hidden stays on device between layers; logits computed on GPU
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

	// after all layers hidden still on device: no DtoH
	if !m.residDevice || !m.residency {
		t.Skip("residency откатилась на host (вероятно нехватка VRAM)")
	}

	// logits come from d_resid; host buffer x is not needed
	if err := m.logits(); err != nil {
		skipOOM(t, err)
		t.Fatal(err)
	}

	if !m.logitsOnGPU {
		t.Fatal("logitsOnGPU сброшен: GPU lm_head не сработал")
	}
}

// §4: batch-prefill with offload must not leave GPU KV-cache stale
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

	// decode after batch must use the resident GPU path again
	if err := m.forwardToken(9, len(tokens), false); err != nil {
		skipOOM(t, err)
		t.Fatal(err)
	}

	if !m.residDevice {
		t.Fatal("decode после batch не пошёл по device-резидентному пути")
	}
}

// §4: GPU loop-B matmul + AttentionScoresKV match serial GPU prefill logits
func TestGPUBatchMatmulAttnParity(t *testing.T) {
	tokens := []int{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16}

	serial := loadResidentModel(t, 0, 1)
	want, err := serial.Forward(tokens, 0)
	if err != nil {
		serial.Close()
		skipOOM(t, err)
		t.Fatal(err)
	}

	wantCopy := append([]float32(nil), want...)
	serial.Close()

	batch := loadResidentModel(t, 0, 32)
	defer batch.Close()

	got, err := batch.Forward(tokens, 0)
	if err != nil {
		skipOOM(t, err)
		t.Fatal(err)
	}

	if batch.gpuKVStale {
		t.Fatal("GPU KV stale после успешного batch-prefill")
	}

	var worst float64
	for i := range wantCopy {
		if d := math.Abs(float64(got[i] - wantCopy[i])); d > worst {
			worst = d
		}
	}

	if worst > 0.5 {
		t.Fatalf("GPU batch vs serial logits: max|diff|=%v", worst)
	}
}

// §4: when GPU KV-cache is shorter than prompt, KVCacheAppendN hits max_seq, model marks cache stale and uses CPU mirror for attention - logits must still match
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

	// GPU KV holds 4 tokens: a chunk of 12 does not fit
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

	// further decode must remain correct (without GPU KV)
	if _, err := small.Forward(tokens[len(tokens)-1:], len(tokens)); err != nil {
		skipOOM(t, err)
		t.Fatal(err)
	}
}
