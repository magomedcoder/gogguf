//go:build cuda

package cuda

import (
	"math"
	"testing"

	"github.com/magomedcoder/gogguf/pkg/format"
	"github.com/magomedcoder/gogguf/pkg/ops"
)

// §5: fused FFN GeGLU (Gemma) на Q8_0-весах совпадает с CPU-путём
func TestFFNGeGLUQuantQ8(t *testing.T) {
	b, err := Open()
	if err != nil {
		t.Skip(err)
	}
	defer b.Close()

	if !b.hasGeGLU {
		t.Skip("нет geglu kernel")
	}

	embd, ffn := 64, 128
	gateRaw, gate := makeQ8Matrix(t, ffn, embd, 11)
	upRaw, up := makeQ8Matrix(t, ffn, embd, 12)
	downRaw, down := makeQ8Matrix(t, embd, ffn, 13)

	x := make([]float32, embd)
	for i := range x {
		x[i] = float32(i%9)*0.07 - 0.3
	}

	gateV, err := ops.MatMulVec(gate, ffn, embd, x)
	if err != nil {
		t.Fatal(err)
	}

	upV, err := ops.MatMulVec(up, ffn, embd, x)
	if err != nil {
		t.Fatal(err)
	}

	ops.GeGLUInPlace(gateV, upV)
	want, err := ops.MatMulVec(down, embd, ffn, gateV)
	if err != nil {
		t.Fatal(err)
	}

	got := make([]float32, embd)
	if err := b.FFNGeGLUQuantCached(format.GgmlQ8_0, "gg", "gu", "gd", gateRaw, upRaw, downRaw, x, got, embd, ffn); err != nil {
		t.Fatal(err)
	}

	for i := range want {
		if math.Abs(float64(got[i]-want[i])) > 3e-3 {
			t.Fatalf("ffn_geglu[%d]=%v want %v", i, got[i], want[i])
		}
	}
}

// §5: fused QKV в режиме RoPE NORM без QK-norm (Llama / Mistral) совпадает с CPU-путём на тех же деквантованных весах
func TestQKVRoPENormWithoutQKNorm(t *testing.T) {
	b, err := Open()
	if err != nil {
		t.Skip(err)
	}
	defer b.Close()

	if !b.hasAttn || !b.hasRoPENorm || !b.hasRMS {
		t.Skip("нет QKV/rope_norm/rmsnorm kernels")
	}

	embd, nHeads, nKVHeads, headDim := 64, 4, 2, 16
	qDim := nHeads * headDim
	kvDim := nKVHeads * headDim
	pos := 3
	eps := float32(1e-6)
	freqBase := float32(10000)

	if err := b.KVCacheInit(1, 8, kvDim, nHeads, headDim); err != nil {
		t.Skip(err)
	}

	wqRaw, wq := makeQ8Matrix(t, qDim, embd, 21)
	wkRaw, wk := makeQ8Matrix(t, kvDim, embd, 22)
	wvRaw, wv := makeQ8Matrix(t, kvDim, embd, 23)

	h := make([]float32, embd)
	for i := range h {
		h[i] = float32(i%13)*0.04 - 0.25
	}

	half := headDim / 2
	cos := make([]float32, half)
	sin := make([]float32, half)
	ops.RoPECosSin(cos, sin, headDim, pos, freqBase)

	// CPU-эталон: QKV -> RoPE NORM -> attention по единственному токену
	wantQ, err := ops.MatMulVec(wq, qDim, embd, h)
	if err != nil {
		t.Fatal(err)
	}

	wantK, err := ops.MatMulVec(wk, kvDim, embd, h)
	if err != nil {
		t.Fatal(err)
	}

	wantV, err := ops.MatMulVec(wv, kvDim, embd, h)
	if err != nil {
		t.Fatal(err)
	}

	ops.ApplyRoPEHeadsNorm(wantQ, nHeads, headDim, pos, freqBase)
	ops.ApplyRoPEHeadsNorm(wantK, nKVHeads, headDim, pos, freqBase)

	wantAttn := make([]float32, qDim)
	scores := make([]float32, 1)
	if err := ops.AttentionScoresInto(wantAttn, wantQ, wantK, wantV, scores, 1, nHeads, nKVHeads, headDim); err != nil {
		t.Fatal(err)
	}

	gotAttn := make([]float32, qDim)
	gotK := make([]float32, kvDim)
	gotV := make([]float32, kvDim)
	if err := b.QKVRoPEAttentionQuantCached(format.GgmlQ8_0, ops.RoPENorm, "nwq", "nwk", "nwv", "", "", "",
		wqRaw, wkRaw, wvRaw, nil, nil, nil, h, cos, sin,
		gotAttn, gotK, gotV, embd, nHeads, nKVHeads, headDim, 0, 0, 1, eps); err != nil {
		t.Fatal(err)
	}

	for i := range wantK {
		if math.Abs(float64(gotK[i]-wantK[i])) > 3e-3 {
			t.Fatalf("k[%d]=%v want %v", i, gotK[i], wantK[i])
		}

		if math.Abs(float64(gotV[i]-wantV[i])) > 3e-3 {
			t.Fatalf("v[%d]=%v want %v", i, gotV[i], wantV[i])
		}
	}

	for i := range wantAttn {
		if math.Abs(float64(gotAttn[i]-wantAttn[i])) > 3e-3 {
			t.Fatalf("attn[%d]=%v want %v", i, gotAttn[i], wantAttn[i])
		}
	}
}
