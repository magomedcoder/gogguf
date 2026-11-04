//go:build cuda

package cuda

import (
	"math"
	"testing"

	"github.com/magomedcoder/gogguf/pkg/compute"
	"github.com/magomedcoder/gogguf/pkg/dtype"
)

func TestLayerNormBiasGPU(t *testing.T) {
	b, err := Open()
	if err != nil {
		t.Skip("CUDA unavailable:", err)
	}
	defer b.Close()

	if !b.hasLayerNorm {
		t.Skip("CUDA layernorm kernel unavailable")
	}

	x := []float32{1, 2, 3, 4, -1, 0.5}
	weight := []float32{1, 0.5, 1, 2, 1, 1}
	bias := []float32{0.1, -0.2, 0, 0.05, 0, 0.01}
	dst := make([]float32, len(x))
	want := make([]float32, len(x))
	if err := compute.LayerNormInto(want, x, weight, bias, 1e-5); err != nil {
		t.Fatal(err)
	}

	if err := b.LayerNormBiasInto(dst, x, weight, bias, 1e-5); err != nil {
		t.Fatal(err)
	}

	for i := range want {
		if math.Abs(float64(dst[i]-want[i])) > 1e-4 {
			t.Fatalf("element %d: got %v, want %v", i, dst[i], want[i])
		}
	}
}

func TestGELUGPU(t *testing.T) {
	b, err := Open()
	if err != nil {
		t.Skip("CUDA unavailable:", err)
	}
	defer b.Close()

	if !b.hasGELU {
		t.Skip("CUDA gelu kernel unavailable")
	}

	x := []float32{-2, -0.5, 0, 0.25, 1, 3}
	want := append([]float32(nil), x...)
	compute.GELUInPlace(want)

	got := append([]float32(nil), x...)
	if err := b.GELUInPlace(got); err != nil {
		t.Fatal(err)
	}

	for i := range want {
		if math.Abs(float64(got[i]-want[i])) > 2e-4 {
			t.Fatalf("element %d: got %v, want %v", i, got[i], want[i])
		}
	}
}

func TestLogitsFromDeviceLayerNorm(t *testing.T) {
	b, err := Open()
	if err != nil {
		t.Skip(err)
	}
	defer b.Close()

	if !b.hasLayerNorm {
		t.Skip("no layernorm")
	}

	embd, vocab := 32, 64
	eps := float32(1e-5)
	x := make([]float32, embd)
	normW := make([]float32, embd)
	normB := make([]float32, embd)
	for i := range x {
		x[i] = float32(i%7)*0.1 - 0.3
		normW[i] = 1
		normB[i] = float32(i%3) * 0.01
	}

	headRaw, head := makeQ8Matrix(t, vocab, embd, 9)

	normed := make([]float32, embd)
	if err := compute.LayerNormInto(normed, x, normW, normB, eps); err != nil {
		t.Fatal(err)
	}
	want, err := compute.MatMulVec(head, vocab, embd, normed)
	if err != nil {
		t.Fatal(err)
	}

	if err := b.HiddenUpload(x); err != nil {
		t.Fatal(err)
	}

	got := make([]float32, vocab)
	if err := b.LogitsFromDeviceLayerNorm(dtype.Q8_0, "lnw", "lnb", normW, normB, "head", headRaw, nil, got, vocab, embd, eps); err != nil {
		t.Fatal(err)
	}

	for i := range want {
		if math.Abs(float64(got[i]-want[i])) > 3e-3 {
			t.Fatalf("logits[%d]=%v want %v", i, got[i], want[i])
		}
	}
}

func TestLayerNormResidAndResidAdd(t *testing.T) {
	b, err := Open()
	if err != nil {
		t.Skip(err)
	}
	defer b.Close()

	if !b.hasLayerNorm || !b.hasAdd {
		t.Skip("no layernorm/add")
	}

	n := 48
	eps := float32(1e-5)
	x := make([]float32, n)
	w := make([]float32, n)
	bias := make([]float32, n)
	delta := make([]float32, n)
	for i := range x {
		x[i] = float32(i%11)*0.05 - 0.2
		w[i] = 1.1
		bias[i] = 0.02
		delta[i] = float32(i%5) * 0.01
	}

	if err := b.HiddenUpload(x); err != nil {
		t.Fatal(err)
	}

	h := make([]float32, n)
	if err := b.LayerNormResidInto(h, w, bias, "w", "b", eps); err != nil {
		t.Fatal(err)
	}

	wantH := make([]float32, n)
	if err := compute.LayerNormInto(wantH, x, w, bias, eps); err != nil {
		t.Fatal(err)
	}
	for i := range wantH {
		if math.Abs(float64(h[i]-wantH[i])) > 1e-4 {
			t.Fatalf("ln[%d]=%v want %v", i, h[i], wantH[i])
		}
	}

	if err := b.ResidAdd(delta); err != nil {
		t.Fatal(err)
	}

	got := make([]float32, n)
	if err := b.HiddenDownload(got); err != nil {
		t.Fatal(err)
	}
	for i := range x {
		want := x[i] + delta[i]
		if math.Abs(float64(got[i]-want)) > 1e-5 {
			t.Fatalf("resid[%d]=%v want %v", i, got[i], want)
		}
	}
}
