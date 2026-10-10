package gpu

import (
	"testing"

	"github.com/magomedcoder/gogguf/pkg/gpu/metal"
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

	if _, err := b.MatMulVec(nil, 1, 1, nil); err != metal.ErrUnavailable {
		t.Fatalf("MatMulVec = %v, expected ErrUnavailable", err)
	}

	if err := b.KVCacheInit(1, 1, 1, 1, 1); err != metal.ErrUnavailable {
		t.Fatalf("KVCacheInit = %v, expected ErrUnavailable", err)
	}

	if err := b.Close(); err != nil {
		t.Fatalf("Close: %v", err)
	}
}
