//go:build metal && darwin && cgo

package metal

import (
	"slices"
	"testing"

	"github.com/magomedcoder/gogguf/pkg/compute"
	"github.com/magomedcoder/gogguf/pkg/dtype"
	"github.com/magomedcoder/gogguf/pkg/quant"
)

// B3 parity against the CPU reference on a real Metal device (skipped without one, see openTest).
// The Linux-runnable counterparts with a fake runtime live in batch_test.go.

func TestMetalMatMulMatQ8_0Parity(t *testing.T) {
	b := openTest(t)

	for _, tc := range []struct{ rows, cols, batch int }{{1, 32, 2}, {7, 64, 1}, {7, 64, 8}, {70, 96, 32}, {300, 2048, 32}, {129, 256, 33}} {
		raw := makeQ8_0(tc.rows, tc.cols, uint32(tc.rows*17+tc.cols))
		x := detRand(tc.batch*tc.cols, 21)

		want := make([]float32, tc.batch*tc.rows)
		if err := compute.MatMulMatQ8_0Into(raw, tc.rows, tc.cols, x, tc.batch, want); err != nil {
			t.Fatal(err)
		}

		// twice: first call uploads and caches the weight by name, second hits the cache
		for pass := range 2 {
			got := make([]float32, tc.batch*tc.rows)
			if err := b.MatMulMatQ8_0CachedInto("test.mat.q8", raw, tc.rows, tc.cols, x, tc.batch, got); err != nil {
				t.Fatalf("%+v pass %d: %v", tc, pass, err)
			}

			closeTo(t, "MatMulMatQ8_0CachedInto", got, want, 1e-4)
		}

		// uncached (name == ""): temp upload path
		got := make([]float32, tc.batch*tc.rows)
		if err := b.MatMulMatQ8_0CachedInto("", raw, tc.rows, tc.cols, x, tc.batch, got); err != nil {
			t.Fatal(err)
		}

		closeTo(t, "MatMulMatQ8_0CachedInto uncached", got, want, 1e-4)
	}
}

// consecutive GEMMs on the same x (Q -> K -> V) reuse the staged input; rewriting x in place between calls must not return stale results
func TestMetalMatMulMatSkipXReuse(t *testing.T) {
	b := openTest(t)

	const rows, cols, batch = 40, 128, 16
	w1, w2 := makeQ8_0(rows, cols, 31), makeQ8_0(rows, cols, 32)
	x := detRand(batch*cols, 33)

	check := func(label string) {
		t.Helper()

		for i, w := range [][]byte{w1, w2} {
			want := make([]float32, batch*rows)
			if err := compute.MatMulMatQ8_0Into(w, rows, cols, x, batch, want); err != nil {
				t.Fatal(err)
			}

			got := make([]float32, batch*rows)
			if err := b.MatMulMatQ8_0CachedInto([]string{"skip.w1", "skip.w2"}[i], w, rows, cols, x, batch, got); err != nil {
				t.Fatal(err)
			}

			closeTo(t, label, got, want, 1e-4)
		}
	}

	check("same x twice")

	// the same slice rewritten in place (what RMSNormBatchInto does to the prefill scratch between layers)
	copy(x, detRand(batch*cols, 34))
	check("x rewritten in place")

	// out aliasing x
	same := slices.Clone(x[:batch*cols])
	want := make([]float32, batch*rows)
	if err := compute.MatMulMatQ8_0Into(w1, rows, cols, same, batch, want); err != nil {
		t.Fatal(err)
	}

	buf := make([]float32, batch*cols) // out smaller than x is fine, aliasing starts at the same pointer
	copy(buf, same)

	if err := b.MatMulMatQ8_0CachedInto("skip.w1", w1, rows, cols, buf, batch, buf); err != nil {
		t.Fatal(err)
	}

	closeTo(t, "out aliases x", buf[:batch*rows], want, 1e-4)
}

// Q4_0 / Q4_K / F32 prefill keep working through the loop-B entry points of the same Backend
func TestMetalMatMulMatLoopBParity(t *testing.T) {
	b := openTest(t)
	const batch = 5

	rows, cols := 9, 64
	m, x := detRand(rows*cols, 41), detRand(batch*cols, 42)
	want, got := make([]float32, batch*rows), make([]float32, batch*rows)
	if err := compute.MatMulMatInto(m, rows, cols, x, batch, want); err != nil {
		t.Fatal(err)
	}

	if err := b.MatMulMatCachedInto("loop.f32", m, rows, cols, x, batch, got); err != nil {
		t.Fatal(err)
	}

	closeTo(t, "MatMulMatCachedInto", got, want, 1e-4)

	for _, tc := range []struct {
		name      string
		qk, bytes int
		call      func(string, []byte, int, int, []float32, int, []float32) error
		ref       func([]byte, int, int, []float32) ([]float32, error)
	}{
		{"Q4_0", quant.QK4_0, quant.BlockQ4_0Size, b.MatMulMatQ4_0CachedInto, compute.MatMulVecQ4_0},
		{"Q4_K", quant.QK_K, quant.BlockQ4_KSize, b.MatMulMatQ4_KCachedInto, compute.MatMulVecQ4_K},
	} {
		rows, cols := 6, tc.qk*2
		raw := fakeQ4(rows, cols, tc.qk, tc.bytes, 43)
		x := detRand(batch*cols, 44)

		var want []float32
		for i := range batch {
			v, err := tc.ref(raw, rows, cols, x[i*cols:(i+1)*cols])
			if err != nil {
				t.Fatal(err)
			}

			want = append(want, v...)
		}

		got := make([]float32, batch*rows)
		if err := tc.call("loop."+tc.name, raw, rows, cols, x, batch, got); err != nil {
			t.Fatalf("%s: %v", tc.name, err)
		}

		closeTo(t, tc.name, got, want, 1e-4)
	}
}

func TestMetalFFNSwiGLUMatQuantParity(t *testing.T) {
	b := openTest(t)
	l := newLayerFixture()

	for _, batch := range []int{1, 4, 32} {
		x := detRand(batch*l.embd, 51)

		gate, up := make([]float32, batch*l.ffn), make([]float32, batch*l.ffn)
		_ = compute.MatMulMatQ8_0Into(l.gate, l.ffn, l.embd, x, batch, gate)
		_ = compute.MatMulMatQ8_0Into(l.up, l.ffn, l.embd, x, batch, up)
		compute.SwiGLUInPlace(gate, up)

		want := make([]float32, batch*l.embd)
		_ = compute.MatMulMatQ8_0Into(l.down, l.embd, l.ffn, gate, batch, want)

		got := make([]float32, batch*l.embd)
		if err := b.FFNSwiGLUMatQuantCached(dtype.Q8_0, "ffn.gate", "ffn.up", "ffn.down", l.gate, l.up, l.down, x, got, l.embd, l.ffn, batch); err != nil {
			t.Fatalf("batch=%d: %v", batch, err)
		}

		closeTo(t, "FFNSwiGLUMatQuantCached", got, want, 1e-4)

		// in place, as the qwen3 prefill calls it, twice in a row (the second call must not reuse the stale staged input)
		inPlace := slices.Clone(x)
		for pass := range 2 {
			copy(inPlace, x)

			if err := b.FFNSwiGLUMatQuantCached(dtype.Q8_0, "ffn.gate", "ffn.up", "ffn.down", l.gate, l.up, l.down, inPlace, inPlace, l.embd, l.ffn, batch); err != nil {
				t.Fatalf("batch=%d in place pass %d: %v", batch, pass, err)
			}

			closeTo(t, "FFNSwiGLUMatQuantCached in place", inPlace, want, 1e-4)
		}
	}

	// F32 / other quants stay unavailable (the caller uses MatMulMat*)
	x, out := make([]float32, 2*l.embd), make([]float32, 2*l.embd)
	if err := b.FFNSwiGLUMatQuantCached(dtype.Q4_0, "g", "u", "d", l.gate, l.up, l.down, x, out, l.embd, l.ffn, 2); err == nil {
		t.Fatal("Q4_0 batch FFN: expected ErrUnavailable")
	}
}

func TestMetalAttentionScoresKVBatchParity(t *testing.T) {
	b := openTest(t)

	for _, tc := range []struct{ nH, nKV, hd, past, n int }{{4, 4, 16, 0, 1}, {8, 2, 16, 0, 8}, {8, 2, 16, 5, 8}, {16, 8, 128, 20, 32}} {
		kvDim, qDim := tc.nKV*tc.hd, tc.nH*tc.hd
		total := tc.past + tc.n

		k, v := detRand(total*kvDim, 61), detRand(total*kvDim, 62)
		q := detRand(tc.n*qDim, 63)

		want := make([]float32, tc.n*qDim)
		if err := compute.AttentionScoresBatchCausalInto(want, q, k, v, make([]float32, total), tc.past, tc.n, tc.nH, tc.nKV, tc.hd); err != nil {
			t.Fatal(err)
		}

		if err := b.KVCacheInit(1, total+4, kvDim, tc.nH, tc.hd); err != nil {
			t.Fatal(err)
		}

		if err := b.KVCacheAppendN(0, 0, k, v, total); err != nil {
			t.Fatal(err)
		}

		got := make([]float32, tc.n*qDim)
		if err := b.AttentionScoresKVBatch(0, got, q, tc.n, tc.past, tc.nH, tc.nKV, tc.hd); err != nil {
			t.Fatalf("%+v: %v", tc, err)
		}

		closeTo(t, "AttentionScoresKVBatch", got, want, 1e-4)

		// positions beyond the cache are an error (caller falls back to the CPU reference)
		if err := b.AttentionScoresKVBatch(0, got, q, tc.n, tc.past+1, tc.nH, tc.nKV, tc.hd); err == nil {
			t.Fatal("AttentionScoresKVBatch beyond cached positions: expected error")
		}
	}
}
