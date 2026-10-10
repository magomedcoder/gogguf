//go:build cuda

package qwen3_test

import (
	"math"
	"testing"

	"github.com/magomedcoder/gogguf"
	"github.com/magomedcoder/gogguf/pkg/gpu"
)

// maxAbsDiff - maximum difference between two logit vectors
func maxAbsDiff(t *testing.T, a, b []float32) float64 {
	t.Helper()

	if len(a) != len(b) {
		t.Fatalf("len %d vs %d", len(a), len(b))
	}

	var worst float64
	for i := range a {
		d := math.Abs(float64(a[i] - b[i]))
		if d > worst {
			worst = d
		}
	}

	return worst
}

func skipOOM(t *testing.T, err error) {
	t.Helper()
	if gpu.IsOutOfMemory(err) {
		t.Skip("insufficient VRAM:", err)
	}
}

// §1+§2: full offload with resident hidden and GPU out_norm+lm_head must match CPU-path logits
func TestGPUFullOffloadLogitsParity(t *testing.T) {
	path := testModelPath(t)

	cpuEng, err := gogguf.Load(path, gogguf.LoadOptions{})
	if err != nil {
		t.Fatal(err)
	}
	defer cpuEng.Close()

	tokens, err := cpuEng.Tokenizer().Encode("Hello, world! Tell me about GPU offload.")
	if err != nil {
		t.Fatal(err)
	}

	cpuLogits, err := cpuEng.ForwardTokenIDs(tokens, 0)
	if err != nil {
		t.Fatal(err)
	}

	gpuEng, err := gogguf.Load(path, gogguf.LoadOptions{NGL: 999})
	if err != nil {
		t.Skip("GPU offload unavailable:", err)
	}
	defer gpuEng.Close()

	gpuLogits, err := gpuEng.ForwardTokenIDs(tokens, 0)
	if err != nil {
		skipOOM(t, err)
		t.Fatal(err)
	}

	if d := maxAbsDiff(t, cpuLogits, gpuLogits); d > 0.5 {
		t.Fatalf("CPU vs GPU logits diverge: max|diff|=%v", d)
	}
}

// §4: -b with -ngl is allowed and yields same logits as token-wise GPU prefill
func TestGPUNBatchPrefillParity(t *testing.T) {
	path := testModelPath(t)

	text := "Hello, world! Tell me about GPU offload."

	// run engines sequentially: two full weight copies may not fit in VRAM
	run := func(nBatch int) (prefill, decode []float32) {
		eng, err := gogguf.Load(path, gogguf.LoadOptions{NGL: 999, NBatch: nBatch})
		if err != nil {
			t.Skip("GPU offload unavailable:", err)
		}
		defer eng.Close()

		tokens, err := eng.Tokenizer().Encode(text)
		if err != nil {
			t.Fatal(err)
		}

		prefill, err = eng.ForwardTokenIDs(tokens, 0)
		if err != nil {
			skipOOM(t, err)
			t.Fatal(err)
		}

		// decode after prefill: chunk K/V must be in GPU KV-cache
		decode, err = eng.ForwardTokenIDs(tokens[len(tokens)-1:], len(tokens))
		if err != nil {
			skipOOM(t, err)
			t.Fatal(err)
		}

		return append([]float32(nil), prefill...), append([]float32(nil), decode...)
	}

	serialPrefill, serialDecode := run(1)
	batchPrefill, batchDecode := run(32)

	if d := maxAbsDiff(t, serialPrefill, batchPrefill); d > 0.5 {
		t.Fatalf("prefill logits serial vs batch diverge: max|diff|=%v", d)
	}

	if d := maxAbsDiff(t, serialDecode, batchDecode); d > 0.5 {
		t.Fatalf("decode after batch-prefill diverges: max|diff|=%v", d)
	}
}
