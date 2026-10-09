//go:build cuda

package gpu

import (
	"strings"
	"testing"
)

// MultiBackend over one real CUDA device: layer plan, KV, hidden and VRAM should behave like without wrapper (single-card list)
func TestMultiBackendOverRealCUDADevice(t *testing.T) {
	dev, err := OpenCUDADevice(0)
	if err != nil {
		t.Skipf("CUDA недоступна: %v", err)
	}

	m, err := NewMultiBackend([]Backend{dev}, nil)
	if err != nil {
		dev.Close()
		t.Fatal(err)
	}
	defer m.Close()

	if err := m.KVCacheInit(4, 64, 8, 2, 4); err != nil {
		t.Fatalf("KVCacheInit: %v", err)
	}

	if plan := Describe(m); !strings.Contains(plan, "слои 0-3") {
		t.Fatalf("Describe = %q, ожидали диапазон слоёв 0-3", plan)
	}

	x := make([]float32, 16)
	for i := range x {
		x[i] = float32(i) * 0.25
	}

	if err := m.HiddenUpload(x); err != nil {
		t.Fatalf("HiddenUpload: %v", err)
	}

	if !m.HiddenActive() {
		t.Fatal("HiddenActive() = false после HiddenUpload")
	}

	got := make([]float32, len(x))
	if err := m.HiddenDownload(got); err != nil {
		t.Fatalf("HiddenDownload: %v", err)
	}

	for i := range x {
		if got[i] != x[i] {
			t.Fatalf("hidden[%d] = %v, ожидали %v", i, got[i], x[i])
		}
	}

	used, total, err := m.VRAMInfo()
	if err != nil {
		t.Fatalf("VRAMInfo: %v", err)
	}

	if total == 0 || used > total {
		t.Fatalf("VRAMInfo = %d/%d", used, total)
	}
}
