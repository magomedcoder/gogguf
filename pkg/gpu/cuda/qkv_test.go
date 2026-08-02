//go:build cuda

package cuda

import (
	"math"
	"testing"

	"github.com/magomedcoder/gogguf/pkg/ops"
)

// TestQKVRoPEAttentionGraphReplay проверяет QKV+RoPE+attn на GPU и replay CUDA Graph.
func TestQKVRoPEAttentionGraphReplay(t *testing.T) {
	b, err := Open()
	if err != nil {
		t.Skip(err)
	}
	defer b.Close()

	if !b.hasAttn || !b.hasRoPE || !b.hasRMS || !b.hasSoftmax {
		t.Skip("нет QKV/RoPE/attn kernels")
	}

	embd := 8
	nHeads := 2
	nKVHeads := 1
	headDim := 4
	qDim := nHeads * headDim
	kvDim := nKVHeads * headDim
	eps := float32(1e-6)
	freqBase := float32(10000)

	if err := b.KVCacheInit(1, 8, kvDim, nHeads, headDim); err != nil {
		t.Fatal(err)
	}

	h := make([]float32, embd)
	for i := range h {
		h[i] = float32(i+1) * 0.05
	}

	wq := make([]float32, qDim*embd)
	wk := make([]float32, kvDim*embd)
	wv := make([]float32, kvDim*embd)
	qNorm := make([]float32, headDim)
	kNorm := make([]float32, headDim)
	for i := range wq {
		wq[i] = float32((i%5)-2) * 0.03
	}

	for i := range wk {
		wk[i] = float32((i%7)-3) * 0.02
		wv[i] = float32((i%3)-1) * 0.04
	}

	for i := range qNorm {
		qNorm[i] = 1
		kNorm[i] = 1
	}

	half := headDim / 2
	cos := make([]float32, half)
	sin := make([]float32, half)

	// runCPU - эталон на CPU: QKV -> RMSNorm -> RoPE -> attention
	runCPU := func(pos, seqLen int) (attn, kTok, vTok []float32) {
		ops.RoPECosSin(cos, sin, headDim, pos, freqBase)
		q := make([]float32, qDim)
		k := make([]float32, kvDim)
		v := make([]float32, kvDim)
		if err := ops.MatMulVecInto(wq, qDim, embd, h, q); err != nil {
			t.Fatal(err)
		}

		if err := ops.MatMulVecInto(wk, kvDim, embd, h, k); err != nil {
			t.Fatal(err)
		}

		if err := ops.MatMulVecInto(wv, kvDim, embd, h, v); err != nil {
			t.Fatal(err)
		}

		for hi := 0; hi < nHeads; hi++ {
			off := hi * headDim
			if err := ops.RMSNormInto(q[off:off+headDim], q[off:off+headDim], qNorm, eps); err != nil {
				t.Fatal(err)
			}
		}

		for hi := 0; hi < nKVHeads; hi++ {
			off := hi * headDim
			if err := ops.RMSNormInto(k[off:off+headDim], k[off:off+headDim], kNorm, eps); err != nil {
				t.Fatal(err)
			}
		}

		ops.ApplyRoPEHeads(q, nHeads, headDim, pos, freqBase)
		ops.ApplyRoPEHeads(k, nKVHeads, headDim, pos, freqBase)

		kCache := make([]float32, seqLen*kvDim)
		vCache := make([]float32, seqLen*kvDim)
		copy(kCache[(seqLen-1)*kvDim:], k)
		copy(vCache[(seqLen-1)*kvDim:], v)
		// Ранние позиции нули: для первого токена важен только текущий
		attn = make([]float32, qDim)
		scores := make([]float32, seqLen)
		if err := ops.AttentionScoresInto(attn, q, kCache, vCache, scores, seqLen, nHeads, nKVHeads, headDim); err != nil {
			t.Fatal(err)
		}

		return attn, k, v
	}

	// Первый токен: capture CUDA Graph
	ops.RoPECosSin(cos, sin, headDim, 0, freqBase)
	wantAttn, _, _ := runCPU(0, 1)
	attn := make([]float32, qDim)
	kOut := make([]float32, kvDim)
	vOut := make([]float32, kvDim)
	if err := b.QKVRoPEAttentionCached("wq", "wk", "wv", "qn", "kn", wq, wk, wv, qNorm, kNorm, h, cos, sin, attn, kOut, vOut, embd, nHeads, nKVHeads, headDim, 0, 0, 1, eps); err != nil {
		t.Fatal(err)
	}

	for i := range wantAttn {
		if math.Abs(float64(attn[i]-wantAttn[i])) > 2e-3 {
			t.Fatalf("токен 0 attn[%d]=%v, ожидали %v", i, attn[i], wantAttn[i])
		}
	}

	// Второй токен: replay graph + seq_len=2
	ops.RoPECosSin(cos, sin, headDim, 1, freqBase)
	wantAttn2, _, _ := runCPU(1, 2)
	// runCPU выше кладёт в KV только текущий токен; собираем полный кеш из двух
	{
		ops.RoPECosSin(cos, sin, headDim, 0, freqBase)
		q0 := make([]float32, qDim)
		k0 := make([]float32, kvDim)
		v0 := make([]float32, kvDim)
		_ = ops.MatMulVecInto(wq, qDim, embd, h, q0)
		_ = ops.MatMulVecInto(wk, kvDim, embd, h, k0)
		_ = ops.MatMulVecInto(wv, kvDim, embd, h, v0)
		for hi := 0; hi < nHeads; hi++ {
			off := hi * headDim
			_ = ops.RMSNormInto(q0[off:off+headDim], q0[off:off+headDim], qNorm, eps)
		}

		for hi := 0; hi < nKVHeads; hi++ {
			off := hi * headDim
			_ = ops.RMSNormInto(k0[off:off+headDim], k0[off:off+headDim], kNorm, eps)
		}

		ops.ApplyRoPEHeads(q0, nHeads, headDim, 0, freqBase)
		ops.ApplyRoPEHeads(k0, nKVHeads, headDim, 0, freqBase)

		ops.RoPECosSin(cos, sin, headDim, 1, freqBase)
		q1 := make([]float32, qDim)
		k1 := make([]float32, kvDim)
		v1 := make([]float32, kvDim)
		_ = ops.MatMulVecInto(wq, qDim, embd, h, q1)
		_ = ops.MatMulVecInto(wk, kvDim, embd, h, k1)
		_ = ops.MatMulVecInto(wv, kvDim, embd, h, v1)
		for hi := 0; hi < nHeads; hi++ {
			off := hi * headDim
			_ = ops.RMSNormInto(q1[off:off+headDim], q1[off:off+headDim], qNorm, eps)
		}

		for hi := 0; hi < nKVHeads; hi++ {
			off := hi * headDim
			_ = ops.RMSNormInto(k1[off:off+headDim], k1[off:off+headDim], kNorm, eps)
		}

		ops.ApplyRoPEHeads(q1, nHeads, headDim, 1, freqBase)
		ops.ApplyRoPEHeads(k1, nKVHeads, headDim, 1, freqBase)

		kCache := append(append([]float32{}, k0...), k1...)
		vCache := append(append([]float32{}, v0...), v1...)
		scores := make([]float32, 2)
		wantAttn2 = make([]float32, qDim)
		if err := ops.AttentionScoresInto(wantAttn2, q1, kCache, vCache, scores, 2, nHeads, nKVHeads, headDim); err != nil {
			t.Fatal(err)
		}
	}

	attn2 := make([]float32, qDim)
	if err := b.QKVRoPEAttentionCached("wq", "wk", "wv", "qn", "kn",
		wq, wk, wv, qNorm, kNorm, h, cos, sin, attn2, kOut, vOut,
		embd, nHeads, nKVHeads, headDim, 0, 1, 2, eps); err != nil {
		t.Fatal(err)
	}

	for i := range wantAttn2 {
		if math.Abs(float64(attn2[i]-wantAttn2[i])) > 2e-3 {
			t.Fatalf("токен 1 attn[%d]=%v, ожидали %v", i, attn2[i], wantAttn2[i])
		}
	}

	if b.hasGraphs && b.matmulPool.layer_graphs == nil {
		t.Fatal("ожидался QKV layer CUDA Graph после replay")
	}

	if b.hasGraphs && b.attnPool.graphs == nil {
		t.Fatal("ожидался attention CUDA Graph по seq_len")
	}
}
