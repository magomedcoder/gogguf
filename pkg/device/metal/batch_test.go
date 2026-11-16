package metal

import (
	"errors"
	"slices"
	"strings"
	"testing"

	"github.com/magomedcoder/gogguf/pkg/compute"
	"github.com/magomedcoder/gogguf/pkg/dtype"
	"github.com/magomedcoder/gogguf/pkg/quant"
)

// B3 (prefill / batch GEMM) on the fake runtime: the same contracts as batch_darwin.go executed with the CPU reference, so Backend validation, the batch==1 / loop-B routing and the ErrUnavailable behaviour are tested on every OS.
// The real-device counterparts live in batch_darwin_test.go.

func (f *fakeGPU) matMulMatQ8_0(_ string, raw []byte, rows, cols int, x []float32, batch int, out []float32) error {
	f.calls = append(f.calls, "matmul_mat_q8_0")
	return compute.MatMulMatQ8_0Into(raw, rows, cols, x, batch, out)
}

func (f *fakeGPU) ffnSwiGLUMatQ8_0(a *ffnMatArgs) error {
	f.calls = append(f.calls, "ffn_mat")

	gate, up := make([]float32, a.batch*a.ffn), make([]float32, a.batch*a.ffn)
	if err := compute.MatMulMatQ8_0Into(a.gateRaw, a.ffn, a.embd, a.x, a.batch, gate); err != nil {
		return err
	}

	if err := compute.MatMulMatQ8_0Into(a.upRaw, a.ffn, a.embd, a.x, a.batch, up); err != nil {
		return err
	}

	compute.SwiGLUInPlace(gate, up)

	out := make([]float32, a.batch*a.embd)
	if err := compute.MatMulMatQ8_0Into(a.downRaw, a.embd, a.ffn, gate, a.batch, out); err != nil {
		return err
	}

	copy(a.out, out) // x and out may alias: the result is only written after the whole chain

	return nil
}

func (f *fakeGPU) attentionKVBatch(layer int, dst, q []float32, n, pastLen, nHeads, nKVHeads, headDim int) error {
	f.calls = append(f.calls, "attn_batch")

	if pastLen+n > f.kvLen[layer] {
		return errors.New("fake: positions not cached")
	}

	total := (pastLen + n) * f.kvDim

	return compute.AttentionScoresBatchCausalInto(dst, q, f.kvK[layer][:total], f.kvV[layer][:total], make([]float32, pastLen+n), pastLen, n, nHeads, nKVHeads, headDim)
}

func countCalls(f *fakeGPU, name string) int {
	n := 0
	for _, c := range f.calls {
		if c == name {
			n++
		}
	}

	return n
}

// MatMulMatQ8_0CachedInto: batch 1 / 8 / 32 match compute.MatMulMatQ8_0Into; only batch >= 2 launches the GEMM kernel, batch == 1 takes the matvec path
func TestBackendMatMulMatQ8_0Parity(t *testing.T) {
	for _, batch := range []int{1, 8, 32} {
		f := &fakeGPU{}
		b := &Backend{
			rt: f,
		}

		rows, cols := 7, 64
		raw := fakeQ8(rows, cols, 700)
		x := detFloats(batch*cols, 701)

		want := make([]float32, batch*rows)
		if err := compute.MatMulMatQ8_0Into(raw, rows, cols, x, batch, want); err != nil {
			t.Fatal(err)
		}

		got := make([]float32, batch*rows)
		if err := b.MatMulMatQ8_0CachedInto("w.q8", raw, rows, cols, x, batch, got); err != nil {
			t.Fatalf("batch=%d: %v", batch, err)
		}

		approxEqual(t, "MatMulMatQ8_0CachedInto", got, want, 1e-5)

		gemm := countCalls(f, "matmul_mat_q8_0")
		if batch == 1 && gemm != 0 {
			t.Fatalf("batch=1 launched the GEMM kernel %d times, want the vec path", gemm)
		}

		if batch > 1 && gemm != 1 {
			t.Fatalf("batch=%d: GEMM launches = %d, want exactly 1 (one launch for W*X[batch])", batch, gemm)
		}
	}
}

func TestBackendMatMulMatQ8_0Contracts(t *testing.T) {
	b := &Backend{
		rt: &fakeGPU{},
	}
	rows, cols, batch := 5, 64, 4
	raw := fakeQ8(rows, cols, 710)
	x := detFloats(batch*cols, 711)
	out := make([]float32, batch*rows)

	for _, tc := range []struct {
		name string
		call func() error
	}{
		{"batch 0", func() error { return b.MatMulMatQ8_0CachedInto("w", raw, rows, cols, x, 0, out) }},
		{"short x", func() error { return b.MatMulMatQ8_0CachedInto("w", raw, rows, cols, x[:batch*cols-1], batch, out) }},
		{"short out", func() error { return b.MatMulMatQ8_0CachedInto("w", raw, rows, cols, x, batch, out[:batch*rows-1]) }},
		{"cols%32", func() error { return b.MatMulMatQ8_0CachedInto("w", raw, rows, cols+1, x, batch, out) }},
		{"short matrix", func() error { return b.MatMulMatQ8_0CachedInto("w", raw[:len(raw)-1], rows, cols, x, batch, out) }},
		{"rows 0", func() error { return b.MatMulMatQ8_0CachedInto("w", raw, 0, cols, x, batch, out) }},
	} {
		if err := tc.call(); err == nil || errors.Is(err, ErrUnavailable) {
			t.Fatalf("%s: err = %v, expected a validation error", tc.name, err)
		}
	}

	var scaffold Backend
	if err := scaffold.MatMulMatQ8_0CachedInto("w", raw, rows, cols, x, batch, out); !errors.Is(err, ErrUnavailable) {
		t.Fatalf("scaffold MatMulMatQ8_0CachedInto = %v, expected ErrUnavailable", err)
	}
}

// F32 and the other quants loop the matvec entry points INSIDE MatMulMat*: Q4_0 / Q4_K / Q5_K / Q6_K / F32 return correct results (no GEMM kernel is launched), unsupported quants are ErrUnavailable
func TestBackendMatMulMatLoopB(t *testing.T) {
	f := &fakeGPU{}
	b := &Backend{
		rt: f,
	}
	const batch = 6

	// F32
	rows, cols := 5, 37
	m, x := detFloats(rows*cols, 720), detFloats(batch*cols, 721)
	want, got := make([]float32, batch*rows), make([]float32, batch*rows)
	if err := compute.MatMulMatInto(m, rows, cols, x, batch, want); err != nil {
		t.Fatal(err)
	}

	if err := b.MatMulMatCachedInto("w.f32", m, rows, cols, x, batch, got); err != nil {
		t.Fatal(err)
	}

	approxEqual(t, "MatMulMatCachedInto", got, want, 1e-5)

	// Q4_0 / Q4_K / Q5_K / Q6_K
	for _, tc := range []struct {
		name      string
		qk, bytes int
		call      func(string, []byte, int, int, []float32, int, []float32) error
		ref       func([]byte, int, int, []float32) ([]float32, error)
	}{
		{"Q4_0", quant.QK4_0, quant.BlockQ4_0Size, b.MatMulMatQ4_0CachedInto, compute.MatMulVecQ4_0},
		{"Q4_K", quant.QK_K, quant.BlockQ4_KSize, b.MatMulMatQ4_KCachedInto, compute.MatMulVecQ4_K},
		{"Q5_K", quant.QK_K, quant.BlockQ5_KSize, b.MatMulMatQ5_KCachedInto, compute.MatMulVecQ5_K},
		{"Q6_K", quant.QK_K, quant.BlockQ6_KSize, b.MatMulMatQ6_KCachedInto, compute.MatMulVecQ6_K},
	} {
		rows, cols := 4, tc.qk*2
		raw := fakeQ4(rows, cols, tc.qk, tc.bytes, 730)
		x := detFloats(batch*cols, 731)

		want := make([]float32, 0, batch*rows)
		for i := range batch {
			v, err := tc.ref(raw, rows, cols, x[i*cols:(i+1)*cols])
			if err != nil {
				t.Fatal(err)
			}

			want = append(want, v...)
		}

		got := make([]float32, batch*rows)
		if err := tc.call("w."+tc.name, raw, rows, cols, x, batch, got); err != nil {
			t.Fatalf("%s: %v", tc.name, err)
		}

		approxEqual(t, tc.name, got, want, 0)

		if err := tc.call("w", raw, rows, cols, x[:batch*cols-1], batch, got); err == nil || errors.Is(err, ErrUnavailable) {
			t.Fatalf("%s: short x: err = %v, expected a validation error", tc.name, err)
		}
	}

	if n := countCalls(f, "matmul_mat_q8_0"); n != 0 {
		t.Fatalf("loop-B entry points launched the Q8_0 GEMM %d times", n)
	}

	// quants without a matvec kernel: ErrUnavailable from inside the method (not a type-assert failure)
	rows, cols = 2, 256
	x = detFloats(batch*cols, 740)
	out := make([]float32, batch*rows)
	for name, call := range map[string]func() error{
		"Q8_1": func() error { return b.MatMulMatQ8_1CachedInto("w", nil, rows, 32, x, batch, out) },
		"Q4_1": func() error { return b.MatMulMatQ4_1CachedInto("w", nil, rows, cols, x, batch, out) },
		"Q5_0": func() error { return b.MatMulMatQ5_0CachedInto("w", nil, rows, cols, x, batch, out) },
		"Q5_1": func() error { return b.MatMulMatQ5_1CachedInto("w", nil, rows, cols, x, batch, out) },
		"Q2_K": func() error { return b.MatMulMatQ2_KCachedInto("w", nil, rows, cols, x, batch, out) },
		"Q3_K": func() error { return b.MatMulMatQ3_KCachedInto("w", nil, rows, cols, x, batch, out) },
		"Q8_K": func() error { return b.MatMulMatQ8_KCachedInto("w", nil, rows, cols, x, batch, out) },
	} {
		if err := call(); !errors.Is(err, ErrUnavailable) {
			t.Fatalf("%s: err = %v, expected ErrUnavailable", name, err)
		}
	}

	var scaffold Backend
	if err := scaffold.MatMulMatCachedInto("w", m, rows, cols, x, batch, out); !errors.Is(err, ErrUnavailable) {
		t.Fatalf("scaffold MatMulMatCachedInto = %v, expected ErrUnavailable", err)
	}

	if err := scaffold.MatMulMatQ4_0CachedInto("w", nil, rows, cols, x, batch, out); !errors.Is(err, ErrUnavailable) {
		t.Fatalf("scaffold MatMulMatQ4_0CachedInto = %v, expected ErrUnavailable", err)
	}
}

// FFNSwiGLUMatQuantCached (Q8_0): gate/up GEMM -> SwiGLU -> down GEMM in one runtime call, x and out may alias; FP32 / other quants are ErrUnavailable
func TestBackendFFNSwiGLUMatQuantCached(t *testing.T) {
	l := newLayerFixture()

	for _, batch := range []int{1, 4, 32} {
		f := &fakeGPU{}
		b := &Backend{rt: f}

		x := detFloats(batch*l.embd, 750)

		gate, up := make([]float32, batch*l.ffn), make([]float32, batch*l.ffn)
		if err := compute.MatMulMatQ8_0Into(l.gate, l.ffn, l.embd, x, batch, gate); err != nil {
			t.Fatal(err)
		}

		if err := compute.MatMulMatQ8_0Into(l.up, l.ffn, l.embd, x, batch, up); err != nil {
			t.Fatal(err)
		}

		compute.SwiGLUInPlace(gate, up)

		want := make([]float32, batch*l.embd)
		if err := compute.MatMulMatQ8_0Into(l.down, l.embd, l.ffn, gate, batch, want); err != nil {
			t.Fatal(err)
		}

		got := make([]float32, batch*l.embd)
		if err := b.FFNSwiGLUMatQuantCached(dtype.Q8_0, "g", "u", "d", l.gate, l.up, l.down, x, got, l.embd, l.ffn, batch); err != nil {
			t.Fatalf("batch=%d: %v", batch, err)
		}

		approxEqual(t, "FFNSwiGLUMatQuantCached", got, want, 1e-4)

		if n := countCalls(f, "ffn_mat"); (batch == 1) == (n == 1) {
			t.Fatalf("batch=%d: ffn_mat launches = %d (batch 1 uses the B2 fused matvec FFN, batch > 1 exactly one batch FFN)", batch, n)
		}

		// in place (x == out), as the qwen3 prefill calls it
		inPlace := slices.Clone(x)
		if err := b.FFNSwiGLUMatQuantCached(dtype.Q8_0, "g", "u", "d", l.gate, l.up, l.down, inPlace, inPlace, l.embd, l.ffn, batch); err != nil {
			t.Fatalf("batch=%d in place: %v", batch, err)
		}

		approxEqual(t, "FFNSwiGLUMatQuantCached in place", inPlace, want, 1e-4)
	}

	b := &Backend{
		rt: &fakeGPU{},
	}
	x := detFloats(2*l.embd, 751)
	out := make([]float32, 2*l.embd)

	if err := b.FFNSwiGLUMatQuantCached(dtype.Q4_0, "g", "u", "d", l.gate, l.up, l.down, x, out, l.embd, l.ffn, 2); !errors.Is(err, ErrUnavailable) {
		t.Fatalf("Q4_0 batch FFN = %v, expected ErrUnavailable", err)
	}

	if err := b.FFNSwiGLUMatQuantCached(dtype.F32, "g", "u", "d", nil, nil, nil, x, out, l.embd, l.ffn, 2); !errors.Is(err, ErrUnavailable) {
		t.Fatalf("F32 batch FFN = %v, expected ErrUnavailable", err)
	}

	if err := b.FFNSwiGLUMatCached("g", "u", "d", nil, nil, nil, x, out, l.embd, l.ffn, 2); !errors.Is(err, ErrUnavailable) {
		t.Fatalf("FFNSwiGLUMatCached (F32) = %v, expected ErrUnavailable", err)
	}

	for name, call := range map[string]func() error{
		"batch 0": func() error {
			return b.FFNSwiGLUMatQuantCached(dtype.Q8_0, "g", "u", "d", l.gate, l.up, l.down, x, out, l.embd, l.ffn, 0)
		},
		"short x": func() error {
			return b.FFNSwiGLUMatQuantCached(dtype.Q8_0, "g", "u", "d", l.gate, l.up, l.down, x[:l.embd], out, l.embd, l.ffn, 2)
		},
		"short out": func() error {
			return b.FFNSwiGLUMatQuantCached(dtype.Q8_0, "g", "u", "d", l.gate, l.up, l.down, x, out[:l.embd], l.embd, l.ffn, 2)
		},
		"no names": func() error {
			return b.FFNSwiGLUMatQuantCached(dtype.Q8_0, "", "u", "d", l.gate, l.up, l.down, x, out, l.embd, l.ffn, 2)
		},
		"short gate": func() error {
			return b.FFNSwiGLUMatQuantCached(dtype.Q8_0, "g", "u", "d", l.gate[:10], l.up, l.down, x, out, l.embd, l.ffn, 2)
		},
		"short down": func() error {
			return b.FFNSwiGLUMatQuantCached(dtype.Q8_0, "g", "u", "d", l.gate, l.up, l.down[:10], x, out, l.embd, l.ffn, 2)
		},
		"embd % 32": func() error {
			return b.FFNSwiGLUMatQuantCached(dtype.Q8_0, "g", "u", "d", l.gate, l.up, l.down, x, out, l.embd+1, l.ffn, 2)
		},
		"zero ffn": func() error {
			return b.FFNSwiGLUMatQuantCached(dtype.Q8_0, "g", "u", "d", l.gate, l.up, l.down, x, out, l.embd, 0, 2)
		},
		"zero embd": func() error {
			return b.FFNSwiGLUMatQuantCached(dtype.Q8_0, "g", "u", "d", l.gate, l.up, l.down, x, out, 0, l.ffn, 2)
		},
		"negative bs": func() error {
			return b.FFNSwiGLUMatQuantCached(dtype.Q8_0, "g", "u", "d", l.gate, l.up, l.down, x, out, l.embd, l.ffn, -1)
		},
	} {
		if err := call(); err == nil || errors.Is(err, ErrUnavailable) {
			t.Fatalf("%s: err = %v, expected a validation error", name, err)
		}
	}

	var scaffold Backend
	if err := scaffold.FFNSwiGLUMatQuantCached(dtype.Q8_0, "g", "u", "d", l.gate, l.up, l.down, x, out, l.embd, l.ffn, 2); !errors.Is(err, ErrUnavailable) {
		t.Fatalf("scaffold FFNSwiGLUMatQuantCached = %v, expected ErrUnavailable", err)
	}
}

// AttentionScoresKVBatch: causal multi-query attention over the device KV cache equals n single-query AttentionScoresKV calls with seqLen = pastLen+i+1
func TestBackendAttentionScoresKVBatch(t *testing.T) {
	const nH, nKV, hd = 4, 2, 16
	kvDim, qDim := nKV*hd, nH*hd

	for _, tc := range []struct{ pastLen, n int }{{0, 1}, {0, 8}, {5, 8}, {9, 32}} {
		f := &fakeGPU{}
		b := &Backend{
			rt: f,
		}

		total := tc.pastLen + tc.n
		if err := b.KVCacheInit(2, total+4, kvDim, nH, hd); err != nil {
			t.Fatal(err)
		}

		k, v := detFloats(total*kvDim, 760), detFloats(total*kvDim, 761)
		if err := b.KVCacheAppendN(1, 0, k, v, total); err != nil {
			t.Fatal(err)
		}

		q := detFloats(tc.n*qDim, 762)

		want := make([]float32, tc.n*qDim)
		for i := range tc.n {
			if err := b.AttentionScoresKV(1, want[i*qDim:(i+1)*qDim], q[i*qDim:(i+1)*qDim], tc.pastLen+i+1, nH, nKV, hd); err != nil {
				t.Fatal(err)
			}
		}

		got := make([]float32, tc.n*qDim)
		if err := b.AttentionScoresKVBatch(1, got, q, tc.n, tc.pastLen, nH, nKV, hd); err != nil {
			t.Fatalf("past=%d n=%d: %v", tc.pastLen, tc.n, err)
		}

		approxEqual(t, "AttentionScoresKVBatch", got, want, 1e-5)

		if n := countCalls(f, "attn_batch"); n != 1 {
			t.Fatalf("attention runtime calls = %d, want 1 (one command buffer for the whole batch)", n)
		}

		// positions of the last query row must be cached: asking for more than was appended is an error, not stale data
		if err := b.AttentionScoresKVBatch(1, got, q, tc.n, tc.pastLen+1, nH, nKV, hd); err == nil {
			t.Fatal("AttentionScoresKVBatch beyond cached positions: expected error")
		}
	}

	b := &Backend{rt: &fakeGPU{}}
	if err := b.KVCacheInit(1, 8, kvDim, nH, hd); err != nil {
		t.Fatal(err)
	}

	dst, q := make([]float32, 2*qDim), make([]float32, 2*qDim)
	for name, call := range map[string]func() error{
		"n 0":        func() error { return b.AttentionScoresKVBatch(0, dst, q, 0, 0, nH, nKV, hd) },
		"negative":   func() error { return b.AttentionScoresKVBatch(0, dst, q, 2, -1, nH, nKV, hd) },
		"gqa":        func() error { return b.AttentionScoresKVBatch(0, dst, q, 2, 0, 3, 2, hd) },
		"zero heads": func() error { return b.AttentionScoresKVBatch(0, dst, q, 2, 0, 0, 2, hd) },
		"short q":    func() error { return b.AttentionScoresKVBatch(0, dst, q[:qDim], 2, 0, nH, nKV, hd) },
		"short dst":  func() error { return b.AttentionScoresKVBatch(0, dst[:qDim], q, 2, 0, nH, nKV, hd) },
		"uncached":   func() error { return b.AttentionScoresKVBatch(0, dst, q, 2, 0, nH, nKV, hd) },
	} {
		if err := call(); err == nil || errors.Is(err, ErrUnavailable) {
			t.Fatalf("%s: err = %v, expected a validation / runtime error", name, err)
		}
	}

	var scaffold Backend
	if err := scaffold.AttentionScoresKVBatch(0, dst, q, 2, 0, nH, nKV, hd); !errors.Is(err, ErrUnavailable) {
		t.Fatalf("scaffold AttentionScoresKVBatch = %v, expected ErrUnavailable", err)
	}
}

// B3 bookkeeping: B3Kernels are planned, the GEMM entry points point at the real methods (the old plan named MatMulVecQ8_0Cached) and the notes document the CUDA-parity design
func TestB3PlannedKernels(t *testing.T) {
	if !slices.Equal(B3Kernels, []string{"matmul_mat_q8_0"}) {
		t.Fatalf("B3Kernels = %v", B3Kernels)
	}

	var plan *KernelPlan
	for i := range PlannedKernels {
		if PlannedKernels[i].Name == "matmul_mat_q8_0" {
			plan = &PlannedKernels[i]
		}
	}

	if plan == nil {
		t.Fatal("matmul_mat_q8_0 missing from PlannedKernels")
	}

	for _, e := range []string{"MatMulMatQ8_0CachedInto", "FFNSwiGLUMatQuantCached"} {
		if !slices.Contains(plan.Entries, e) {
			t.Fatalf("matmul_mat_q8_0 entries %v lack %s", plan.Entries, e)
		}
	}

	if slices.Contains(plan.Entries, "MatMulVecQ8_0Cached") {
		t.Fatal("matmul_mat_q8_0 must not name the matvec entry point")
	}

	for _, want := range []string{"thread-per-(batch,row)", "simdgroup_matrix"} {
		if !strings.Contains(plan.Notes, want) {
			t.Fatalf("matmul_mat_q8_0 notes %q lack %q", plan.Notes, want)
		}
	}

	if got := plannedKernelFor("AttentionScoresKVBatch"); got != "attn_qk_softmax_v" {
		t.Fatalf("AttentionScoresKVBatch planned kernel = %q", got)
	}
}
