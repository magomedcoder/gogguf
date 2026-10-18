package device

import (
	"errors"
	"strings"
	"testing"

	"github.com/magomedcoder/gogguf/pkg/device/metal"
)

// Metal stub must implement full Backend so it can replace CUDA when kernels appear
var _ Backend = (*metal.Backend)(nil)

func TestMetalScaffoldCompute(t *testing.T) {
	var b metal.Backend
	if b.Name() == "" {
		t.Fatal("Name() empty")
	}

	if _, _, err := b.VRAMInfo(); err != nil {
		t.Fatalf("VRAMInfo: %v", err)
	}

	if b.HiddenResident() {
		t.Fatal("HiddenResident() = true on stub")
	}

	if _, err := b.MatMulVec(nil, 1, 1, nil); !errors.Is(err, metal.ErrUnavailable) {
		t.Fatalf("MatMulVec = %v, expected ErrUnavailable", err)
	}

	if err := b.KVCacheInit(1, 1, 1, 1, 1); !errors.Is(err, metal.ErrUnavailable) {
		t.Fatalf("KVCacheInit = %v, expected ErrUnavailable", err)
	}

	if err := b.Close(); err != nil {
		t.Fatalf("Close: %v", err)
	}
}

// Error text names the entry point and the planned kernel; Ready stays false until kernels land
func TestMetalScaffoldErrorsAreDescriptive(t *testing.T) {
	var b metal.Backend
	if b.Ready() != metal.KernelsImplemented {
		t.Fatalf("Ready() = %v, KernelsImplemented = %v", b.Ready(), metal.KernelsImplemented)
	}

	_, err := b.MatMulVecQ8_0Cached("w", nil, 1, 1, nil)
	if !errors.Is(err, metal.ErrUnavailable) {
		t.Fatalf("MatMulVecQ8_0Cached = %v, expected ErrUnavailable", err)
	}

	if !strings.Contains(err.Error(), "matmul_vec_q8_0") || !strings.Contains(err.Error(), "MatMulVecQ8_0Cached") {
		t.Fatalf("error %q does not name entry point and planned kernel", err)
	}
}

// Every Backend compute entry point is covered by a planned kernel (keeps PlannedKernels honest)
func TestMetalPlannedKernelsCoverEntries(t *testing.T) {
	if len(metal.PlannedKernels) == 0 {
		t.Fatal("PlannedKernels empty")
	}

	seen := make(map[string]bool)
	for _, k := range metal.PlannedKernels {
		if k.Name == "" || len(k.Entries) == 0 {
			t.Fatalf("bad plan entry %+v", k)
		}

		if seen[k.Name] {
			t.Fatalf("duplicate kernel %q", k.Name)
		}

		seen[k.Name] = true
	}
}
