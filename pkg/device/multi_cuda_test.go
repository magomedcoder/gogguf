//go:build cuda

package device

import (
	"strings"
	"testing"
)

// MultiBackend over one real CUDA device: layer plan, KV, hidden and VRAM should behave like without wrapper (single-card list)
func TestMultiBackendOverRealCUDADevice(t *testing.T) {
	dev, err := OpenCUDADevice(0)
	if err != nil {
		t.Skipf("CUDA unavailable: %v", err)
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

	if plan := Describe(m); !strings.Contains(plan, "layers 0-3") {
		t.Fatalf("Describe = %q, expected layer range 0-3", plan)
	}

	x := make([]float32, 16)
	for i := range x {
		x[i] = float32(i) * 0.25
	}

	if err := m.HiddenUpload(x); err != nil {
		t.Fatalf("HiddenUpload: %v", err)
	}

	if !m.HiddenActive() {
		t.Fatal("HiddenActive() = false after HiddenUpload")
	}

	got := make([]float32, len(x))
	if err := m.HiddenDownload(got); err != nil {
		t.Fatalf("HiddenDownload: %v", err)
	}

	for i := range x {
		if got[i] != x[i] {
			t.Fatalf("hidden[%d] = %v, expected %v", i, got[i], x[i])
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

// Real two-GPU P2P: hidden uploaded on GPU0 reaches GPU1 intact through MultiBackend (peer path when topology allows, host fallback otherwise).
// Skipped without two usable CUDA devices.
func TestMultiBackendTwoGPUHiddenMove(t *testing.T) {
	d0, err := OpenCUDADevice(0)
	if err != nil {
		t.Skipf("CUDA device 0 unavailable: %v", err)
	}

	d1, err := OpenCUDADevice(1)
	if err != nil {
		d0.Close()
		t.Skipf("second CUDA device unavailable (multi-GPU test needs 2 GPUs): %v", err)
	}

	m, err := NewMultiBackend([]Backend{d0, d1}, nil)
	if err != nil {
		d0.Close()
		d1.Close()
		t.Fatal(err)
	}
	defer m.Close()

	if err := m.KVCacheInit(4, 64, 8, 2, 4); err != nil {
		t.Fatalf("KVCacheInit: %v", err)
	}

	x := make([]float32, 64)
	for i := range x {
		x[i] = float32(i)*0.5 - 3
	}

	if err := m.HiddenUpload(x); err != nil {
		t.Fatalf("HiddenUpload: %v", err)
	}

	// Layer 2 lives on GPU1: route() moves hidden across the boundary (peer copy or host bounce)
	m.mu.Lock()
	err = m.moveHiddenLocked(1)
	m.mu.Unlock()
	if err != nil {
		t.Fatalf("moveHiddenLocked: %v", err)
	}

	if d0.HiddenActive() || !d1.HiddenActive() {
		t.Fatalf("residency after move: GPU0=%v GPU1=%v, expected false/true", d0.HiddenActive(), d1.HiddenActive())
	}

	got := make([]float32, len(x))
	if err := m.HiddenDownload(got); err != nil {
		t.Fatalf("HiddenDownload: %v", err)
	}

	for i := range x {
		if got[i] != x[i] {
			t.Fatalf("hidden[%d] = %v, expected %v", i, got[i], x[i])
		}
	}

	peer, host := m.TransferStats()
	t.Logf("boundary move: peer=%d host=%d", peer, host)
}
