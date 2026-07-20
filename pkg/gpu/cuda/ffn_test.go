//go:build cuda

package cuda

import (
	"math"
	"testing"

	"github.com/magomedcoder/gogguf/pkg/ops"
)

func TestFFNSwiGLUGPU(t *testing.T) {
	b, err := Open()
	if err != nil {
		t.Skip(err)
	}
	defer b.Close()

	if !b.hasSwiGLU {
		t.Skip("нет SwiGLU kernel")
	}

	embd, ffn := 8, 16
	x := make([]float32, embd)
	gateW := make([]float32, ffn*embd)
	upW := make([]float32, ffn*embd)
	downW := make([]float32, embd*ffn)
	for i := range x {
		x[i] = float32(i+1) * 0.1
	}

	for i := range gateW {
		gateW[i] = float32((i%7)-3) * 0.05
		upW[i] = float32((i%5)-2) * 0.04
	}

	for i := range downW {
		downW[i] = float32((i%11)-5) * 0.03
	}

	wantGate, err := ops.MatMulVec(gateW, ffn, embd, x)
	if err != nil {
		t.Fatal(err)
	}

	wantUp, err := ops.MatMulVec(upW, ffn, embd, x)
	if err != nil {
		t.Fatal(err)
	}

	ops.SwiGLUInPlace(wantGate, wantUp)

	want, err := ops.MatMulVec(downW, embd, ffn, wantGate)
	if err != nil {
		t.Fatal(err)
	}

	out := make([]float32, embd)
	if err := b.FFNSwiGLUCached("g", "u", "d", gateW, upW, downW, x, out, embd, ffn); err != nil {
		t.Fatal(err)
	}

	for i := range want {
		if math.Abs(float64(out[i]-want[i])) > 1e-3 {
			t.Fatalf("out[%d]=%v want %v", i, out[i], want[i])
		}
	}

	// повторный вызов (кеш весов)
	out2 := make([]float32, embd)
	if err := b.FFNSwiGLUCached("g", "u", "d", gateW, upW, downW, x, out2, embd, ffn); err != nil {
		t.Fatal(err)
	}

	for i := range want {
		if math.Abs(float64(out2[i]-want[i])) > 1e-3 {
			t.Fatalf("replay out[%d]=%v want %v", i, out2[i], want[i])
		}
	}
}
