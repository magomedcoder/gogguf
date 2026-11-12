//go:build cuda

package cuda

import (
	"math"
	"testing"

	"github.com/magomedcoder/gogguf/pkg/compute"
	"github.com/magomedcoder/gogguf/pkg/dtype"
)

func TestFFNSwiGLUMatBatchParity(t *testing.T) {
	b, err := Open()
	if err != nil {
		t.Skip("CUDA unavailable:", err)
	}
	defer b.Close()
	if !b.hasSwiGLU {
		t.Skip("SwiGLU unavailable")
	}

	const embd, ffn, batch = 64, 128, 4
	gateW := make([]float32, ffn*embd)
	upW := make([]float32, ffn*embd)
	downW := make([]float32, embd*ffn)
	x := make([]float32, batch*embd)
	for i := range gateW {
		gateW[i] = float32((i%7)-3) * 0.01
	}
	for i := range upW {
		upW[i] = float32((i%5)-2) * 0.02
	}
	for i := range downW {
		downW[i] = float32((i%11)-5) * 0.01
	}
	for i := range x {
		x[i] = float32((i%9)-4) * 0.05
	}

	want := make([]float32, batch*embd)
	gate := make([]float32, batch*ffn)
	up := make([]float32, batch*ffn)
	if err := compute.MatMulMatInto(gateW, ffn, embd, x, batch, gate); err != nil {
		t.Fatal(err)
	}
	if err := compute.MatMulMatInto(upW, ffn, embd, x, batch, up); err != nil {
		t.Fatal(err)
	}
	for i := 0; i < batch; i++ {
		off := i * ffn
		compute.SwiGLUInPlace(gate[off:off+ffn], up[off:off+ffn])
	}
	if err := compute.MatMulMatInto(downW, embd, ffn, gate, batch, want); err != nil {
		t.Fatal(err)
	}

	got := make([]float32, batch*embd)
	if err := b.FFNSwiGLUMatCached("g", "u", "d", gateW, upW, downW, x, got, embd, ffn, batch); err != nil {
		t.Fatal(err)
	}

	var worst float64
	for i := range want {
		if d := math.Abs(float64(got[i] - want[i])); d > worst {
			worst = d
		}
	}
	if worst > 1e-3 {
		t.Fatalf("ffn_mat batch max|diff|=%g", worst)
	}
}

func TestMatMulMatSkipVecReuse(t *testing.T) {
	b, err := Open()
	if err != nil {
		t.Skip("CUDA unavailable:", err)
	}
	defer b.Close()

	const rows, cols, batch = 32, 16, 3
	w1 := make([]float32, rows*cols)
	w2 := make([]float32, rows*cols)
	x := make([]float32, batch*cols)
	for i := range w1 {
		w1[i] = float32(i%5) * 0.1
		w2[i] = float32((i+1)%7) * 0.05
	}
	for i := range x {
		x[i] = float32(i%3) * 0.2
	}

	out1 := make([]float32, batch*rows)
	out2 := make([]float32, batch*rows)
	if err := b.MatMulMatCachedInto("w1", w1, rows, cols, x, batch, out1); err != nil {
		t.Fatal(err)
	}
	// Second call with same host X should reuse d_vec (skip_vec).
	if err := b.MatMulMatCachedInto("w2", w2, rows, cols, x, batch, out2); err != nil {
		t.Fatal(err)
	}

	want1 := make([]float32, batch*rows)
	want2 := make([]float32, batch*rows)
	if err := compute.MatMulMatInto(w1, rows, cols, x, batch, want1); err != nil {
		t.Fatal(err)
	}
	if err := compute.MatMulMatInto(w2, rows, cols, x, batch, want2); err != nil {
		t.Fatal(err)
	}
	for i := range want1 {
		if math.Abs(float64(out1[i]-want1[i])) > 1e-4 || math.Abs(float64(out2[i]-want2[i])) > 1e-4 {
			t.Fatalf("skip_vec matmul mismatch at %d", i)
		}
	}
}

func TestLogitsArgmaxFromDevice(t *testing.T) {
	b, err := Open()
	if err != nil {
		t.Skip("CUDA unavailable:", err)
	}
	defer b.Close()
	if !b.hasLogitsArgmax || !b.hasRMS {
		t.Skip("logits_argmax unavailable")
	}

	const embd, vocab = 32, 64
	norm := make([]float32, embd)
	head := make([]float32, vocab*embd)
	x := make([]float32, embd)
	for i := range norm {
		norm[i] = 1
		x[i] = float32(i%5) * 0.1
	}
	for i := range head {
		head[i] = float32((i%11)-5) * 0.02
	}
	if err := b.HiddenUpload(x); err != nil {
		t.Fatal(err)
	}

	logits := make([]float32, vocab)
	if err := b.LogitsFromDevice(dtype.F32, "n", norm, "h", nil, head, logits, vocab, embd, 1e-6); err != nil {
		t.Fatal(err)
	}
	want := 0
	for i := 1; i < vocab; i++ {
		if logits[i] > logits[want] {
			want = i
		}
	}

	if err := b.HiddenUpload(x); err != nil {
		t.Fatal(err)
	}
	got, _, err := b.LogitsArgmaxFromDevice(dtype.F32, "n", norm, "h", nil, head, vocab, embd, 1e-6)
	if err != nil {
		t.Fatal(err)
	}
	if got != want {
		t.Fatalf("argmax=%d want=%d", got, want)
	}
}
