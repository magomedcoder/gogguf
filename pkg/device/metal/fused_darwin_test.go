//go:build metal && darwin && cgo

package metal

import (
	"fmt"
	"slices"
	"testing"

	"github.com/magomedcoder/gogguf/pkg/compute"
	"github.com/magomedcoder/gogguf/pkg/dtype"
	"github.com/magomedcoder/gogguf/pkg/quant"
)

// B2 parity against the CPU reference on a real Metal device (skipped without one, see openTest).
// The Linux-runnable counterparts with a fake runtime live in fused_test.go; both share layerFixture.

func openFusedTest(t *testing.T, l *layerFixture) *Backend {
	t.Helper()

	b := openTest(t)
	if !b.HiddenResident() {
		t.Fatal("HiddenResident() = false on an opened darwin+metal runtime")
	}

	if err := b.KVCacheInit(1, 16, l.nKV*l.hd, l.nH, l.hd); err != nil {
		t.Fatal(err)
	}

	return b
}

func TestMetalHiddenUploadDownload(t *testing.T) {
	b := openTest(t)

	if b.HiddenActive() {
		t.Fatal("HiddenActive() before upload")
	}

	for _, n := range []int{1, 64, 2560, 5000} {
		x := detRand(n, uint32(n))
		if err := b.HiddenUpload(x); err != nil {
			t.Fatal(err)
		}

		if !b.HiddenActive() {
			t.Fatal("HiddenActive() = false after HiddenUpload")
		}

		got := make([]float32, n)
		if err := b.HiddenDownload(got); err != nil {
			t.Fatal(err)
		}

		if !slices.Equal(got, x) {
			t.Fatalf("n=%d: hidden changed in a round trip", n)
		}

		if b.HiddenActive() {
			t.Fatal("HiddenActive() = true after HiddenDownload")
		}
	}
}

func TestMetalMatMulVecQ4Parity(t *testing.T) {
	b := openTest(t)

	for _, tc := range []struct {
		name      string
		qk, bytes int
		call      func(string, []byte, int, int, []float32) ([]float32, error)
		ref       func([]byte, int, int, []float32) ([]float32, error)
	}{
		{"Q4_0", quant.QK4_0, quant.BlockQ4_0Size, b.MatMulVecQ4_0Cached, compute.MatMulVecQ4_0},
		{"Q4_K", quant.QK_K, quant.BlockQ4_KSize, b.MatMulVecQ4_KCached, compute.MatMulVecQ4_K},
	} {
		for _, sz := range []struct{ rows, blocks int }{{1, 1}, {7, 2}, {70, 3}, {300, 8}} {
			cols := sz.blocks * tc.qk
			raw := fakeQ4(sz.rows, cols, tc.qk, tc.bytes, uint32(sz.rows*17+cols))
			vec := detRand(cols, 77)

			want, err := tc.ref(raw, sz.rows, cols, vec)
			if err != nil {
				t.Fatal(err)
			}

			for pass := range 2 { // second pass hits the weight cache
				got, err := tc.call("test."+tc.name, raw, sz.rows, cols, vec)
				if err != nil {
					t.Fatalf("%s %dx%d pass %d: %v", tc.name, sz.rows, cols, pass, err)
				}

				closeTo(t, "MatMulVec"+tc.name+"Cached", got, want, 1e-4)
			}
		}
	}
}

// resident decode over several tokens equals the host reference
func TestMetalFusedResidentLayerParity(t *testing.T) {
	l := newLayerFixture()
	b := openFusedTest(t, l)

	var kc, vc []float32
	x := detRand(l.embd, 100)
	for pos := range 4 {
		want := l.reference(x, pos, &kc, &vc)

		if err := b.HiddenUpload(x); err != nil {
			t.Fatal(err)
		}

		if err := l.qkv(b, 0, pos, nil, nil, nil, nil); err != nil {
			t.Fatalf("pos %d qkv: %v", pos, err)
		}

		if !b.HiddenActive() {
			t.Fatal("QKV must keep residency")
		}

		if err := l.attnFFN(b, nil, nil); err != nil {
			t.Fatalf("pos %d attnFFN: %v", pos, err)
		}

		got := make([]float32, l.embd)
		if err := b.HiddenDownload(got); err != nil {
			t.Fatal(err)
		}

		closeTo(t, fmt.Sprintf("resident layer pos %d", pos), got, want, 2e-4)
		x = want
	}

	// attention was consumed: AttnFFN with attn == nil and no fresh QKV must fail cleanly (residency intact)
	if err := b.HiddenUpload(x); err != nil {
		t.Fatal(err)
	}

	if err := l.attnFFN(b, nil, nil); err == nil {
		t.Fatal("AttnFFN without device attention: expected error")
	}

	if !b.HiddenActive() {
		t.Fatal("clean failure must keep residency")
	}
}

func TestMetalFusedHostModeParity(t *testing.T) {
	l := newLayerFixture()
	b := openFusedTest(t, l)

	var kc, vc []float32
	x := detRand(l.embd, 200)
	want := l.reference(x, 0, &kc, &vc)

	h := make([]float32, l.embd)
	if err := compute.RMSNormInto(h, x, l.attnNorm, l.eps); err != nil {
		t.Fatal(err)
	}

	attn := make([]float32, l.nH*l.hd)
	kOut, vOut := make([]float32, l.nKV*l.hd), make([]float32, l.nKV*l.hd)
	if err := l.qkv(b, 0, 0, h, attn, kOut, vOut); err != nil {
		t.Fatal(err)
	}

	closeTo(t, "kOut", kOut, kc, 1e-4)
	closeTo(t, "vOut", vOut, vc, 1e-4)

	got := slices.Clone(x)
	if err := l.attnFFN(b, got, attn); err != nil {
		t.Fatal(err)
	}

	closeTo(t, "host-mode layer", got, want, 2e-4)

	if b.HiddenActive() {
		t.Fatal("host mode must not enable residency")
	}
}

func TestMetalFusedLogitsParity(t *testing.T) {
	b := openTest(t)

	const vocab = 70
	for _, tc := range []struct {
		t    dtype.DType
		embd int
	}{{dtype.Q8_0, 64}, {dtype.F32, 64}, {dtype.F16, 64}, {dtype.Q4_0, 64}, {dtype.Q4_K, 256}} {
		x, norm := detRand(tc.embd, 400), detRand(tc.embd, 401)
		if err := b.HiddenUpload(x); err != nil {
			t.Fatal(err)
		}

		h := make([]float32, tc.embd)
		if err := compute.RMSNormInto(h, x, norm, 1e-6); err != nil {
			t.Fatal(err)
		}

		var raw []byte
		var f32, want []float32
		switch tc.t {
		case dtype.Q8_0:
			raw = makeQ8_0(vocab, tc.embd, 402)
			want, _ = compute.MatMulVecQ8_0(raw, vocab, tc.embd, h)
		case dtype.Q4_0:
			raw = fakeQ4(vocab, tc.embd, quant.QK4_0, quant.BlockQ4_0Size, 403)
			want, _ = compute.MatMulVecQ4_0(raw, vocab, tc.embd, h)
		case dtype.Q4_K:
			raw = fakeQ4(vocab, tc.embd, quant.QK_K, quant.BlockQ4_KSize, 404)
			want, _ = compute.MatMulVecQ4_K(raw, vocab, tc.embd, h)
		default:
			f32 = detRand(vocab*tc.embd, 405)
			want, _ = compute.MatMulVec(f32, vocab, tc.embd, h)
		}

		logits := make([]float32, vocab)
		name := "head." + tc.t.String()
		for pass := range 2 {
			if err := b.LogitsFromDevice(tc.t, "out_norm."+tc.t.String(), norm, name, raw, f32, logits, vocab, tc.embd, 1e-6); err != nil {
				t.Fatalf("%s pass %d: %v", tc.t, pass, err)
			}

			closeTo(t, "LogitsFromDevice "+tc.t.String(), logits, want, 1e-4)
		}

		if !b.HiddenActive() {
			t.Fatal("LogitsFromDevice must keep residency")
		}

		_ = b.HiddenDownload(make([]float32, tc.embd))
	}
}

func TestMetalFFNSwiGLUQuantParity(t *testing.T) {
	l := newLayerFixture()
	b := openFusedTest(t, l)

	x := detRand(l.embd, 500)
	g, u := make([]float32, l.ffn), make([]float32, l.ffn)
	_ = compute.MatMulVecQ8_0Into(l.gate, l.ffn, l.embd, x, g)
	_ = compute.MatMulVecQ8_0Into(l.up, l.ffn, l.embd, x, u)
	compute.SwiGLUInPlace(g, u)

	want := make([]float32, l.embd)
	_ = compute.MatMulVecQ8_0Into(l.down, l.embd, l.ffn, g, want)

	out := make([]float32, l.embd)
	if err := b.FFNSwiGLUQuantCached(dtype.Q8_0, "gate", "up", "down", l.gate, l.up, l.down, x, out, l.embd, l.ffn); err != nil {
		t.Fatal(err)
	}

	closeTo(t, "FFNSwiGLUQuantCached", out, want, 1e-4)
}

// the multi-row rmsnorm launch (rows = nHeads for QK-norm) must keep the plain single-row launch intact; the multi-row case is covered end to end by TestMetalFusedResidentLayerParity
func TestMetalRMSNormSingleRowAfterRowsChange(t *testing.T) {
	b := openTest(t)

	x, w := detRand(128, 61), detRand(128, 62)
	want := make([]float32, 128)
	if err := compute.RMSNormInto(want, x, w, 1e-6); err != nil {
		t.Fatal(err)
	}

	got := make([]float32, 128)
	if err := b.RMSNormInto(got, x, w, 1e-6); err != nil {
		t.Fatal(err)
	}

	closeTo(t, "RMSNormInto", got, want, 1e-5)
}
