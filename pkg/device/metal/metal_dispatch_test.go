package metal

import (
	"errors"
	"slices"
	"testing"

	"github.com/magomedcoder/gogguf/pkg/compute"
	"github.com/magomedcoder/gogguf/pkg/quant"
)

// fakeGPU implements gpu with the CPU reference so Backend validation and dispatch are tested on every OS (no Metal needed)
type fakeGPU struct {
	calls []string
	kvK   [][]float32
	kvV   [][]float32
	kvLen []int
	kvDim int
}

func (f *fakeGPU) deviceName() string {
	return "Metal (fake)"
}

func (f *fakeGPU) vram() (used, total uint64) {
	return 1, 2
}

func (f *fakeGPU) close() error {
	f.calls = append(f.calls, "close")
	return nil
}

func (f *fakeGPU) kvReset() {
	f.calls = append(f.calls, "kvReset")
}

func (f *fakeGPU) swiglu(gate, up []float32) error {
	compute.SwiGLUInPlace(gate, up)
	return nil
}

func (f *fakeGPU) kvDropOldest(layer, n int) error {
	f.calls = append(f.calls, "kvDropOldest")
	return nil
}

func (f *fakeGPU) rmsNorm(dst, x, w []float32, eps float32) error {
	return compute.RMSNormInto(dst, x, w, eps)
}

func (f *fakeGPU) matMulF32(_ string, m []float32, rows, cols int, vec, dst []float32) error {
	return compute.MatMulVecInto(m, rows, cols, vec, dst)
}

func (f *fakeGPU) matMulQ8_0(_ string, raw []byte, rows, cols int, vec, dst []float32) error {
	return compute.MatMulVecQ8_0Into(raw, rows, cols, vec, dst)
}

func (f *fakeGPU) ropeHeads(v []float32, nHeads, headDim, pos int, base float32) error {
	compute.ApplyRoPEHeads(v, nHeads, headDim, pos, base)
	return nil
}

func (f *fakeGPU) attention(dst, q, k, v []float32, seqLen, nHeads, nKVHeads, headDim int) error {
	return compute.AttentionScoresInto(dst, q, k, v, make([]float32, seqLen), seqLen, nHeads, nKVHeads, headDim)
}

func (f *fakeGPU) kvInit(layers, maxSeq, kvDim, _, _ int) error {
	f.kvDim = kvDim
	f.kvK = make([][]float32, layers)
	f.kvV = make([][]float32, layers)
	f.kvLen = make([]int, layers)
	for i := range layers {
		f.kvK[i] = make([]float32, maxSeq*kvDim)
		f.kvV[i] = make([]float32, maxSeq*kvDim)
	}

	return nil
}

func (f *fakeGPU) kvAppend(layer, pos int, k, v []float32, n int) error {
	copy(f.kvK[layer][pos*f.kvDim:], k)
	copy(f.kvV[layer][pos*f.kvDim:], v)
	f.kvLen[layer] = max(f.kvLen[layer], pos+n)

	return nil
}

func (f *fakeGPU) kvRebuild(layer int, k, v []float32, n int) error {
	return f.kvAppend(layer, 0, k[:n*f.kvDim], v[:n*f.kvDim], n)
}

func (f *fakeGPU) attentionKV(layer int, dst, q []float32, seqLen, nHeads, nKVHeads, headDim int) error {
	return f.attention(dst, q, f.kvK[layer][:seqLen*f.kvDim], f.kvV[layer][:seqLen*f.kvDim], seqLen, nHeads, nKVHeads, headDim)
}

func detFloats(n int, seed uint32) []float32 {
	out := make([]float32, n)
	for i := range out {
		seed = seed*1664525 + 1013904223
		out[i] = float32(int32(seed>>8)%2001-1000) / 1000
	}

	return out
}

// Backend without runtime (scaffold, non-darwin, no metal tag) is never Ready and every B1 entry point stays ErrUnavailable
func TestBackendScaffoldNotReady(t *testing.T) {
	var b Backend
	if b.Ready() {
		t.Fatal("zero Backend Ready() = true")
	}

	if err := b.RMSNormInto(nil, nil, nil, 0); !errors.Is(err, ErrUnavailable) {
		t.Fatalf("RMSNormInto = %v, expected ErrUnavailable", err)
	}

	if err := b.AttentionScoresKV(0, nil, nil, 1, 1, 1, 1); !errors.Is(err, ErrUnavailable) {
		t.Fatalf("AttentionScoresKV = %v, expected ErrUnavailable", err)
	}

	b.KVCacheReset()

	if err := b.Close(); err != nil {
		t.Fatalf("Close: %v", err)
	}

	if !KernelsImplemented && (&Backend{rt: &fakeGPU{}}).Ready() {
		t.Fatal("Ready() = true while KernelsImplemented is false")
	}
}

func TestB1KernelsArePlanned(t *testing.T) {
	for _, name := range B1Kernels {
		if !slices.ContainsFunc(PlannedKernels, func(k KernelPlan) bool { return k.Name == name }) {
			t.Fatalf("B1 kernel %q missing from PlannedKernels", name)
		}
	}
}

// Backend wiring: with a runtime every B1 entry point validates and dispatches (fakeGPU = CPU reference, so results must match exactly)
func TestBackendDispatchWithRuntime(t *testing.T) {
	f := &fakeGPU{}
	b := &Backend{rt: f}

	if b.Name() != "Metal (fake)" {
		t.Fatalf("Name() = %q", b.Name())
	}

	if used, total, err := b.VRAMInfo(); err != nil || used != 1 || total != 2 {
		t.Fatalf("VRAMInfo = %d %d %v", used, total, err)
	}

	// F32 matvec
	rows, cols := 5, 8
	m, vec := detFloats(rows*cols, 1), detFloats(cols, 2)
	want, _ := compute.MatMulVec(m, rows, cols, vec)
	got, err := b.MatMulVecCached("w", m, rows, cols, vec)
	if err != nil || !slices.Equal(got, want) {
		t.Fatalf("MatMulVecCached = %v, %v; want %v", got, err, want)
	}

	if _, err := b.MatMulVec(m[:3], rows, cols, vec); err == nil {
		t.Fatal("MatMulVec: expected short-matrix error")
	}

	// Q8_0 matvec
	rows, cols = 3, 64
	raw := make([]byte, rows*(cols/quant.QK8_0)*quant.BlockQ8_0Size)
	for i := range raw {
		raw[i] = byte(i*7 + 3)
	}

	for blk := 0; blk < len(raw); blk += quant.BlockQ8_0Size { // finite fp16 scale 0x3000 = 0.125
		raw[blk], raw[blk+1] = 0x00, 0x30
	}

	vec = detFloats(cols, 3)
	wantQ, _ := compute.MatMulVecQ8_0(raw, rows, cols, vec)
	gotQ, err := b.MatMulVecQ8_0Cached("q", raw, rows, cols, vec)
	if err != nil || !slices.Equal(gotQ, wantQ) {
		t.Fatalf("MatMulVecQ8_0Cached = %v, %v; want %v", gotQ, err, wantQ)
	}

	if _, err := b.MatMulVecQ8_0Cached("q", raw, rows, 33, vec); err == nil {
		t.Fatal("MatMulVecQ8_0Cached: expected cols%32 error")
	}

	// ops
	x, w := detFloats(16, 4), detFloats(16, 5)
	d1, d2 := make([]float32, 16), make([]float32, 16)
	_ = compute.RMSNormInto(d1, x, w, 1e-6)
	if err := b.RMSNormInto(d2, x, w, 1e-6); err != nil || !slices.Equal(d1, d2) {
		t.Fatalf("RMSNormInto = %v", err)
	}

	if err := b.RMSNormInto(d2, x, w[:4], 1e-6); err == nil {
		t.Fatal("RMSNormInto: expected length mismatch")
	}

	if err := b.ApplyRoPEHeads(x, 2, 7, 1, 10000); err == nil {
		t.Fatal("ApplyRoPEHeads: expected odd headDim error")
	}

	// KV + attention (GQA 4:2, headDim 4)
	const nH, nKV, hd, seq = 4, 2, 4, 6
	kvDim := nKV * hd
	if err := b.KVCacheInit(1, 16, kvDim, nH, hd); err != nil {
		t.Fatal(err)
	}

	k, v, q := detFloats(seq*kvDim, 6), detFloats(seq*kvDim, 7), detFloats(nH*hd, 8)
	if err := b.KVCacheAppendN(0, 0, k, v, seq); err != nil {
		t.Fatal(err)
	}

	if err := b.KVCacheAppendN(0, 0, k[:5], v[:5], 2); err == nil {
		t.Fatal("KVCacheAppendN: expected divisibility error")
	}

	dstKV, dstHost := make([]float32, nH*hd), make([]float32, nH*hd)
	if err := b.AttentionScoresKV(0, dstKV, q, seq, nH, nKV, hd); err != nil {
		t.Fatal(err)
	}

	if err := b.AttentionScoresInto(dstHost, q, k, v, nil, seq, nH, nKV, hd); err != nil {
		t.Fatal(err)
	}

	if !slices.Equal(dstKV, dstHost) {
		t.Fatalf("KV attention %v != host attention %v", dstKV, dstHost)
	}

	if err := b.AttentionScoresKV(0, dstKV[:2], q, seq, nH, nKV, hd); err == nil {
		t.Fatal("AttentionScoresKV: expected short dst error")
	}

	if err := b.AttentionScoresKV(0, dstKV, q, seq, 3, 2, hd); err == nil {
		t.Fatal("AttentionScoresKV: expected GQA divisibility error")
	}

	b.KVCacheReset()

	if err := b.Close(); err != nil {
		t.Fatal(err)
	}

	if b.Ready() {
		t.Fatal("Ready() after Close")
	}

	if want := []string{"kvReset", "close"}; !slices.Equal(f.calls, want) {
		t.Fatalf("runtime calls = %v, want %v", f.calls, want)
	}
}
