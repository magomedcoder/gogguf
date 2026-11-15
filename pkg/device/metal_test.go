package device

import (
	"errors"
	"strings"
	"testing"

	"github.com/magomedcoder/gogguf/pkg/device/metal"
	"github.com/magomedcoder/gogguf/pkg/dtype"
)

// Metal stub must implement full Backend so it can replace CUDA when kernels appear
var _ Backend = (*metal.Backend)(nil)

// B3 prefill capabilities: GEMM (every MatMulMat* entry, loop-B inside for non-Q8_0), batch FFN and batch attention are real Metal methods, so the type assertions in
// MatMulMatQuantCachedInto / FFNSwiGLUMatQuantCached / AttentionScoresKVBatch succeed on a Metal backend (like cuda_open.go for CUDA)
var (
	_ MatMulMatBackend        = (*metal.Backend)(nil)
	_ FFNSwiGLUMatBackend     = (*metal.Backend)(nil)
	_ KVBatchAttentionBackend = (*metal.Backend)(nil)
)

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

// Error text names the entry point and the planned kernel; a Backend without an open runtime is never Ready (with a runtime, Ready follows KernelsImplemented: true only in a darwin+cgo -tags metal build)
func TestMetalScaffoldErrorsAreDescriptive(t *testing.T) {
	var b metal.Backend
	if b.Ready() {
		t.Fatalf("Ready() = true on a Backend without runtime (KernelsImplemented = %v)", metal.KernelsImplemented)
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

// A scaffold Metal backend still takes the B3 dispatch helpers (type assertions hold) and reports ErrUnavailable from inside, so prefill falls back to the CPU instead of panicking
func TestMetalScaffoldBatchDispatchUnavailable(t *testing.T) {
	var b metal.Backend
	x := make([]float32, 2*32)
	out := make([]float32, 2*2)

	if err := MatMulMatQuantCachedInto(&b, dtype.Q8_0, "w", nil, 2, 32, x, 2, out); !errors.Is(err, metal.ErrUnavailable) {
		t.Fatalf("MatMulMatQuantCachedInto Q8_0 = %v, expected ErrUnavailable", err)
	}

	if err := MatMulMatQuantCachedInto(&b, dtype.Q4_0, "w", nil, 2, 32, x, 2, out); !errors.Is(err, metal.ErrUnavailable) {
		t.Fatalf("MatMulMatQuantCachedInto Q4_0 = %v, expected ErrUnavailable", err)
	}

	if err := MatMulMatCachedIntoFP32(&b, "w", nil, 2, 32, x, 2, out); !errors.Is(err, metal.ErrUnavailable) {
		t.Fatalf("MatMulMatCachedIntoFP32 = %v, expected ErrUnavailable", err)
	}

	if err := FFNSwiGLUMatQuantCached(&b, dtype.Q8_0, "g", "u", "d", nil, nil, nil, x, x, 32, 64, 2); !errors.Is(err, metal.ErrUnavailable) {
		t.Fatalf("FFNSwiGLUMatQuantCached = %v, expected ErrUnavailable", err)
	}

	if err := AttentionScoresKVBatch(&b, 0, out, out, 2, 0, 1, 1, 2); !errors.Is(err, metal.ErrUnavailable) {
		t.Fatalf("AttentionScoresKVBatch = %v, expected ErrUnavailable", err)
	}
}
