//go:build cuda

package cuda

import (
	"math"
	"testing"

	"github.com/magomedcoder/gogguf/pkg/ops"
)

func TestAttnFFNResidualGPU(t *testing.T) {
	b, err := Open()
	if err != nil {
		t.Skip(err)
	}
	defer b.Close()

	if !b.hasSwiGLU || !b.hasRMS || !b.hasAdd {
		t.Skip("нет add/rmsnorm/swiglu")
	}

	embd, attnDim, ffn := 8, 16, 12
	eps := float32(1e-6)
	x := make([]float32, embd)
	attn := make([]float32, attnDim)
	wo := make([]float32, embd*attnDim)
	ffnNorm := make([]float32, embd)
	gate := make([]float32, ffn*embd)
	up := make([]float32, ffn*embd)
	down := make([]float32, embd*ffn)
	for i := range x {
		x[i] = float32(i+1) * 0.1
		ffnNorm[i] = 1
	}

	for i := range attn {
		attn[i] = float32((i%5)+1) * 0.05
	}

	for i := range wo {
		wo[i] = float32((i%7)-3) * 0.02
	}

	for i := range gate {
		gate[i] = float32((i%5)-2) * 0.03
		up[i] = float32((i%3)-1) * 0.04
	}

	for i := range down {
		down[i] = float32((i%11)-5) * 0.02
	}

	wantX := append([]float32(nil), x...)
	h := make([]float32, embd)
	if err := ops.MatMulVecInto(wo, embd, attnDim, attn, h); err != nil {
		t.Fatal(err)
	}

	ops.AddInPlace(wantX, h)

	if err := ops.RMSNormInto(h, wantX, ffnNorm, eps); err != nil {
		t.Fatal(err)
	}

	gateV, err := ops.MatMulVec(gate, ffn, embd, h)
	if err != nil {
		t.Fatal(err)
	}

	upV, err := ops.MatMulVec(up, ffn, embd, h)
	if err != nil {
		t.Fatal(err)
	}

	ops.SwiGLUInPlace(gateV, upV)

	downV, err := ops.MatMulVec(down, embd, ffn, gateV)
	if err != nil {
		t.Fatal(err)
	}

	ops.AddInPlace(wantX, downV)

	got := append([]float32(nil), x...)
	if err := b.AttnFFNResidualCached("wo", "norm", "g", "u", "d", wo, ffnNorm, gate, up, down, got, attn, embd, attnDim, ffn, eps); err != nil {
		t.Fatal(err)
	}

	for i := range wantX {
		if math.Abs(float64(got[i]-wantX[i])) > 2e-3 {
			t.Fatalf("x[%d]=%v want %v", i, got[i], wantX[i])
		}
	}
}
