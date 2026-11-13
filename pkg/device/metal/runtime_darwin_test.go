//go:build metal && darwin && cgo

package metal

import (
	"math"
	"testing"

	"github.com/magomedcoder/gogguf/pkg/compute"
	"github.com/magomedcoder/gogguf/pkg/quant"
)

// openTest opens the real Metal runtime or skips (no GPU, CI VM without Metal)
func openTest(t *testing.T) *Backend {
	t.Helper()

	b, err := Open()
	if err != nil {
		t.Skipf("Metal unavailable: %v", err)
	}

	if !b.Ready() {
		_ = b.Close()
		t.Skip("Metal backend not ready")
	}

	t.Cleanup(func() { _ = b.Close() })

	return b
}

// closeTo compares GPU output to the CPU reference with a tolerance scaled by the largest reference magnitude
func closeTo(t *testing.T, what string, got, want []float32, rel float64) {
	t.Helper()

	if len(got) != len(want) {
		t.Fatalf("%s: len %d, want %d", what, len(got), len(want))
	}

	var maxRef float64
	for _, w := range want {
		maxRef = math.Max(maxRef, math.Abs(float64(w)))
	}

	tol := rel * math.Max(1, maxRef)
	for i := range want {
		if d := math.Abs(float64(got[i] - want[i])); d > tol || math.IsNaN(d) {
			t.Fatalf("%s[%d] = %v, want %v (|diff| %g > tol %g)", what, i, got[i], want[i], d, tol)
		}
	}
}

func detRand(n int, seed uint32) []float32 {
	out := make([]float32, n)
	for i := range out {
		seed = seed*1664525 + 1013904223
		out[i] = float32(int32(seed>>8)%2001-1000) / 1000
	}

	return out
}

// f16Bits converts a normal float32 to IEEE half bits (truncating; test scales are small positive normals)
func f16Bits(v float32) uint16 {
	b := math.Float32bits(v)
	exp := int((b>>23)&0xff) - 127 + 15

	return uint16(b>>16)&0x8000 | uint16(exp)<<10 | uint16((b>>13)&0x3ff)
}

// makeQ8_0 builds rows x cols Q8_0 data in the ggml layout (fp16 scale + 32 int8) that pkg/quant reads
func makeQ8_0(rows, cols int, seed uint32) []byte {
	blocks := rows * cols / quant.QK8_0
	raw := make([]byte, blocks*quant.BlockQ8_0Size)
	rnd := detRand(blocks*(1+quant.QK8_0), seed)

	for b := range blocks {
		off := b * quant.BlockQ8_0Size
		d := f16Bits(0.002 + 0.02*(rnd[b*33]+1)/2)
		raw[off], raw[off+1] = byte(d), byte(d>>8)

		for i := range quant.QK8_0 {
			raw[off+2+i] = byte(int8(rnd[b*33+1+i] * 127))
		}
	}

	return raw
}

func TestMetalMatMulVecQ8_0Parity(t *testing.T) {
	b := openTest(t)

	for _, tc := range []struct{ rows, cols int }{{1, 32}, {7, 64}, {70, 96}, {300, 2048}} {
		raw := makeQ8_0(tc.rows, tc.cols, uint32(tc.rows*31+tc.cols))
		vec := detRand(tc.cols, 11)

		want, err := compute.MatMulVecQ8_0(raw, tc.rows, tc.cols, vec)
		if err != nil {
			t.Fatal(err)
		}

		// twice: first call uploads and caches the weight by name, second hits the cache
		for pass := range 2 {
			got, err := b.MatMulVecQ8_0Cached("test.q8", raw, tc.rows, tc.cols, vec)
			if err != nil {
				t.Fatalf("%dx%d pass %d: %v", tc.rows, tc.cols, pass, err)
			}

			closeTo(t, "MatMulVecQ8_0Cached", got, want, 1e-4)
		}
	}

	if used, total, err := b.VRAMInfo(); err != nil || used == 0 || total == 0 {
		t.Fatalf("VRAMInfo = %d %d %v, expected non-zero after uploads", used, total, err)
	}
}

func TestMetalMatMulVecF32Parity(t *testing.T) {
	b := openTest(t)

	for _, tc := range []struct{ rows, cols int }{{1, 1}, {5, 37}, {64, 64}, {129, 256}} {
		m := detRand(tc.rows*tc.cols, uint32(tc.cols))
		vec := detRand(tc.cols, 5)

		want, err := compute.MatMulVec(m, tc.rows, tc.cols, vec)
		if err != nil {
			t.Fatal(err)
		}

		got, err := b.MatMulVec(m, tc.rows, tc.cols, vec)
		if err != nil {
			t.Fatal(err)
		}

		closeTo(t, "MatMulVec", got, want, 1e-4)

		got, err = b.MatMulVecCached("test.f32", m, tc.rows, tc.cols, vec)
		if err != nil {
			t.Fatal(err)
		}

		closeTo(t, "MatMulVecCached", got, want, 1e-4)
	}
}

func TestMetalRMSNormParity(t *testing.T) {
	b := openTest(t)

	for _, dim := range []int{1, 32, 128, 1000, 2560} {
		x, w := detRand(dim, 21), detRand(dim, 22)
		want := make([]float32, dim)
		if err := compute.RMSNormInto(want, x, w, 1e-6); err != nil {
			t.Fatal(err)
		}

		got := make([]float32, dim)
		if err := b.RMSNormInto(got, x, w, 1e-6); err != nil {
			t.Fatal(err)
		}

		closeTo(t, "RMSNormInto", got, want, 1e-5)

		// in-place (dst aliases x), as the per-head QK-norm does
		inPlace := append([]float32(nil), x...)
		if err := b.RMSNormInto(inPlace, inPlace, w, 1e-6); err != nil {
			t.Fatal(err)
		}

		closeTo(t, "RMSNormInto in-place", inPlace, want, 1e-5)
	}
}

func TestMetalRoPEHeadsParity(t *testing.T) {
	b := openTest(t)

	for _, tc := range []struct{ heads, headDim, pos int }{{1, 2, 0}, {4, 16, 5}, {8, 128, 1234}, {2, 256, 30000}} {
		v := detRand(tc.heads*tc.headDim, uint32(tc.pos)+1)
		want := append([]float32(nil), v...)
		compute.ApplyRoPEHeads(want, tc.heads, tc.headDim, tc.pos, 1000000)

		got := append([]float32(nil), v...)
		if err := b.ApplyRoPEHeads(got, tc.heads, tc.headDim, tc.pos, 1000000); err != nil {
			t.Fatal(err)
		}

		closeTo(t, "ApplyRoPEHeads", got, want, 1e-5)
	}
}

func TestMetalSwiGLUParity(t *testing.T) {
	b := openTest(t)

	for _, n := range []int{1, 255, 256, 257, 9728} {
		gate, up := detRand(n, 31), detRand(n, 32)
		for i := range gate {
			gate[i] *= 8
		}

		want := append([]float32(nil), gate...)
		compute.SwiGLUInPlace(want, up)

		got := append([]float32(nil), gate...)
		if err := b.SwiGLUInPlace(got, up); err != nil {
			t.Fatal(err)
		}

		closeTo(t, "SwiGLUInPlace", got, want, 1e-5)
	}
}

func TestMetalAttentionParity(t *testing.T) {
	b := openTest(t)

	for _, tc := range []struct{ nH, nKV, hd, seq int }{{4, 4, 16, 1}, {8, 2, 16, 37}, {16, 8, 128, 300}, {32, 8, 128, 700}} {
		kvDim := tc.nKV * tc.hd
		q := detRand(tc.nH*tc.hd, 41)
		k, v := detRand(tc.seq*kvDim, 42), detRand(tc.seq*kvDim, 43)

		want := make([]float32, tc.nH*tc.hd)
		if err := compute.AttentionScoresInto(want, q, k, v, make([]float32, tc.seq), tc.seq, tc.nH, tc.nKV, tc.hd); err != nil {
			t.Fatal(err)
		}

		// host k/v path
		got := make([]float32, len(want))
		if err := b.AttentionScoresInto(got, q, k, v, nil, tc.seq, tc.nH, tc.nKV, tc.hd); err != nil {
			t.Fatal(err)
		}

		closeTo(t, "AttentionScoresInto", got, want, 1e-4)

		// device KV path: append token by token on layer 1, the last token in a batch via AppendN
		if err := b.KVCacheInit(2, tc.seq+8, kvDim, tc.nH, tc.hd); err != nil {
			t.Fatal(err)
		}

		single := tc.seq - 1
		for p := range single {
			if err := b.KVCacheAppend(1, p, k[p*kvDim:(p+1)*kvDim], v[p*kvDim:(p+1)*kvDim]); err != nil {
				t.Fatal(err)
			}
		}

		if err := b.KVCacheAppendN(1, single, k[single*kvDim:], v[single*kvDim:], 1); err != nil {
			t.Fatal(err)
		}

		gotKV := make([]float32, len(want))
		if err := b.AttentionScoresKV(1, gotKV, q, tc.seq, tc.nH, tc.nKV, tc.hd); err != nil {
			t.Fatal(err)
		}

		closeTo(t, "AttentionScoresKV", gotKV, want, 1e-4)

		// seqLen beyond cached positions is an error (caller falls back to CPU), not stale data
		if err := b.AttentionScoresKV(1, gotKV, q, tc.seq+1, tc.nH, tc.nKV, tc.hd); err == nil {
			t.Fatal("AttentionScoresKV beyond cached length: expected error")
		}

		// Reset forgets positions
		b.KVCacheReset()

		if err := b.AttentionScoresKV(1, gotKV, q, tc.seq, tc.nH, tc.nKV, tc.hd); err == nil {
			t.Fatal("AttentionScoresKV after KVCacheReset: expected error")
		}
	}
}

func TestMetalKVDropOldestAndRebuild(t *testing.T) {
	b := openTest(t)

	const nH, nKV, hd, seq, drop = 4, 2, 16, 20, 5
	kvDim := nKV * hd
	q := detRand(nH*hd, 51)
	k, v := detRand(seq*kvDim, 52), detRand(seq*kvDim, 53)

	if err := b.KVCacheInit(1, 32, kvDim, nH, hd); err != nil {
		t.Fatal(err)
	}

	if err := b.KVCacheAppendN(0, 0, k, v, seq); err != nil {
		t.Fatal(err)
	}

	if err := b.KVCacheDropOldest(0, drop); err != nil {
		t.Fatal(err)
	}

	want := make([]float32, nH*hd)
	rem := seq - drop
	if err := compute.AttentionScoresInto(want, q, k[drop*kvDim:], v[drop*kvDim:], make([]float32, rem), rem, nH, nKV, hd); err != nil {
		t.Fatal(err)
	}

	got := make([]float32, len(want))
	if err := b.AttentionScoresKV(0, got, q, rem, nH, nKV, hd); err != nil {
		t.Fatal(err)
	}

	closeTo(t, "after DropOldest", got, want, 1e-4)

	if err := b.KVCacheRebuildFromHost(0, k, v, seq); err != nil {
		t.Fatal(err)
	}

	if err := compute.AttentionScoresInto(want, q, k, v, make([]float32, seq), seq, nH, nKV, hd); err != nil {
		t.Fatal(err)
	}

	if err := b.AttentionScoresKV(0, got, q, seq, nH, nKV, hd); err != nil {
		t.Fatal(err)
	}

	closeTo(t, "after RebuildFromHost", got, want, 1e-4)
}

func TestMetalReadyAndName(t *testing.T) {
	b := openTest(t)

	if !KernelsImplemented || !b.Ready() {
		t.Fatalf("KernelsImplemented=%v Ready=%v, expected both true on darwin+metal", KernelsImplemented, b.Ready())
	}

	if b.Name() == "" {
		t.Fatal("empty device name")
	}
}
