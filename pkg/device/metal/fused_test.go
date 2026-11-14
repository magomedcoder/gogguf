package metal

import (
	"errors"
	"fmt"
	"math"
	"testing"

	"github.com/magomedcoder/gogguf/pkg/compute"
	"github.com/magomedcoder/gogguf/pkg/dtype"
	"github.com/magomedcoder/gogguf/pkg/quant"
)

// Fake B2 runtime: the same contracts as runtime_darwin.go / fused_darwin.go executed with the CPU reference, so Backend validation, the residency flag (h == nil / x == nil, dirty failures) and the attention-stays-on-device hand-off are tested on every OS.

func (f *fakeGPU) matMulQuant(t dtype.DType, _ string, raw []byte, rows, cols int, vec, dst []float32) error {
	switch t {
	case dtype.Q8_0:
		return compute.MatMulVecQ8_0Into(raw, rows, cols, vec, dst)
	case dtype.Q4_0:
		return compute.MatMulVecQ4_0Into(raw, rows, cols, vec, dst)
	case dtype.Q4_K:
		return compute.MatMulVecQ4_KInto(raw, rows, cols, vec, dst)
	default:
		return unavailable("matmul " + t.String())
	}
}

func (f *fakeGPU) hiddenReady() bool {
	return !f.noResident
}

func (f *fakeGPU) hiddenUpload(x []float32) error {
	f.calls = append(f.calls, "hiddenUpload")
	f.resid = append(f.resid[:0], x...)
	f.attnDev = nil

	return nil
}

func (f *fakeGPU) hiddenDownload(dst []float32) error {
	f.calls = append(f.calls, "hiddenDownload")
	copy(dst, f.resid)

	return nil
}

func fakeRope(v []float32, nHeads, headDim int, cos, sin []float32) {
	half := headDim / 2
	for h := range nHeads {
		base := h * headDim
		for i := range half {
			x0, x1 := v[base+i], v[base+half+i]
			v[base+i] = x0*cos[i] - x1*sin[i]
			v[base+half+i] = x0*sin[i] + x1*cos[i]
		}
	}
}

func (f *fakeGPU) qkvRoPEAttention(a *qkvArgs, resident bool) error {
	f.calls = append(f.calls, "qkv")
	if f.failQKV {
		return errors.New("fake: qkv failure")
	}

	h := a.h
	if resident {
		h = make([]float32, a.embd)
		if err := compute.RMSNormInto(h, f.resid, a.attnNorm, a.eps); err != nil {
			return err
		}
	}

	qDim, kvDim := a.nHeads*a.headDim, a.nKVHeads*a.headDim
	q, k, v := make([]float32, qDim), make([]float32, kvDim), make([]float32, kvDim)
	if err := compute.MatMulVecQ8_0Into(a.qRaw, qDim, a.embd, h[:a.embd], q); err != nil {
		return err
	}

	if err := compute.MatMulVecQ8_0Into(a.kRaw, kvDim, a.embd, h[:a.embd], k); err != nil {
		return err
	}

	if err := compute.MatMulVecQ8_0Into(a.vRaw, kvDim, a.embd, h[:a.embd], v); err != nil {
		return err
	}

	if a.qNormName != "" {
		for i := range a.nHeads {
			s := q[i*a.headDim : (i+1)*a.headDim]
			_ = compute.RMSNormInto(s, s, a.qNorm[:a.headDim], a.eps)
		}
	}

	if a.kNormName != "" {
		for i := range a.nKVHeads {
			s := k[i*a.headDim : (i+1)*a.headDim]
			_ = compute.RMSNormInto(s, s, a.kNorm[:a.headDim], a.eps)
		}
	}

	fakeRope(q, a.nHeads, a.headDim, a.cos, a.sin)
	fakeRope(k, a.nKVHeads, a.headDim, a.cos, a.sin)

	if err := f.kvAppend(a.layer, a.kvPos, k, v, 1); err != nil {
		return err
	}

	attn := make([]float32, qDim)
	if err := f.attentionKV(a.layer, attn, q, a.seqLen, a.nHeads, a.nKVHeads, a.headDim); err != nil {
		return err
	}

	f.attnDev = attn
	copy(a.attn, attn)
	copy(a.kOut, k)
	copy(a.vOut, v)

	return nil
}

func (f *fakeGPU) ffnRef(gate, up, down []byte, x []float32, embd, ffnDim int) ([]float32, error) {
	g, u := make([]float32, ffnDim), make([]float32, ffnDim)
	if err := compute.MatMulVecQ8_0Into(gate, ffnDim, embd, x, g); err != nil {
		return nil, err
	}

	if err := compute.MatMulVecQ8_0Into(up, ffnDim, embd, x, u); err != nil {
		return nil, err
	}

	compute.SwiGLUInPlace(g, u)

	out := make([]float32, embd)
	if err := compute.MatMulVecQ8_0Into(down, embd, ffnDim, g, out); err != nil {
		return nil, err
	}

	return out, nil
}

func (f *fakeGPU) attnFFNResidual(a *attnFFNArgs, resident bool) (bool, error) {
	f.calls = append(f.calls, "attnFFN")

	attn := a.attn
	if len(attn) < a.attnDim {
		attn = f.attnDev
	}

	f.attnDev = nil
	if len(attn) < a.attnDim {
		return false, errors.New("fake: no attention output on the device")
	}

	if f.failAttnFFN != "" {
		return f.failAttnFFN == "dirty", errors.New("fake: attn+ffn failure")
	}

	resid := f.resid
	if !resident {
		resid = append([]float32(nil), a.x[:a.embd]...)
	}

	wo := make([]float32, a.embd)
	if err := compute.MatMulVecQ8_0Into(a.woRaw, a.embd, a.attnDim, attn[:a.attnDim], wo); err != nil {
		return false, err
	}

	for i := range wo {
		resid[i] += wo[i]
	}

	h := make([]float32, a.embd)
	if err := compute.RMSNormInto(h, resid, a.ffnNorm, a.eps); err != nil {
		return false, err
	}

	d, err := f.ffnRef(a.gateRaw, a.upRaw, a.downRaw, h, a.embd, a.ffn)
	if err != nil {
		return false, err
	}

	for i := range d {
		resid[i] += d[i]
	}

	if !resident {
		copy(a.x, resid)
	}

	return false, nil
}

func (f *fakeGPU) ffnSwiGLU(a *ffnArgs) error {
	d, err := f.ffnRef(a.gateRaw, a.upRaw, a.downRaw, a.x, a.embd, a.ffn)
	if err != nil {
		return err
	}

	copy(a.out, d)

	return nil
}

func (f *fakeGPU) logitsFromResident(a *logitsArgs) error {
	h := make([]float32, a.embd)
	if err := compute.RMSNormInto(h, f.resid, a.norm, a.eps); err != nil {
		return err
	}

	if a.t == dtype.F32 || a.t == dtype.F16 {
		return compute.MatMulVecInto(a.headF32, a.vocab, a.embd, h, a.logits)
	}

	return f.matMulQuant(a.t, a.headName, a.headRaw, a.vocab, a.embd, h, a.logits)
}

// ---- fixtures ----

// fakeQ8 builds rows x cols Q8_0 data with a finite fp16 scale (0x2400 = 2^-6)
func fakeQ8(rows, cols int, seed uint32) []byte {
	blocks := rows * cols / quant.QK8_0
	raw := make([]byte, blocks*quant.BlockQ8_0Size)
	rnd := detFloats(blocks*quant.QK8_0, seed)
	for b := range blocks {
		off := b * quant.BlockQ8_0Size
		raw[off], raw[off+1] = 0x00, 0x24
		for i := range quant.QK8_0 {
			raw[off+2+i] = byte(int8(rnd[b*quant.QK8_0+i] * 100))
		}
	}

	return raw
}

// fakeQ4 builds rows x cols data for a 4-bit quant with the given block size / scale-header bytes: random payload, finite fp16 d (and dmin) in the first 2 / 4 bytes
func fakeQ4(rows, cols, qk, blockBytes int, seed uint32) []byte {
	blocks := rows * cols / qk
	raw := make([]byte, blocks*blockBytes)
	rnd := detFloats(len(raw), seed)
	for i := range raw {
		raw[i] = byte(int32(rnd[i]*127) + 128)
	}

	for b := range blocks {
		off := b * blockBytes
		raw[off], raw[off+1] = 0x00, 0x24 // d = 2^-6
		if blockBytes == quant.BlockQ4_KSize {
			raw[off+2], raw[off+3] = 0x00, 0x20 // dmin = 2^-9
		}
	}

	return raw
}

func approxEqual(t *testing.T, what string, got, want []float32, tol float64) {
	t.Helper()

	if len(got) != len(want) {
		t.Fatalf("%s: len %d, want %d", what, len(got), len(want))
	}

	for i := range want {
		if d := math.Abs(float64(got[i] - want[i])); d > tol || math.IsNaN(d) {
			t.Fatalf("%s[%d] = %v, want %v (|diff| %g > %g)", what, i, got[i], want[i], d, tol)
		}
	}
}

// layerFixture is a tiny Qwen3-like block (GQA 4:2, QK-norm, NeoX RoPE) with Q8_0 weights
type layerFixture struct {
	embd, nH, nKV, hd, ffn int
	eps, base              float32

	q, k, v, wo, gate, up, down []byte
	attnNorm, ffnNorm, qn, kn   []float32
}

func newLayerFixture() *layerFixture {
	l := &layerFixture{
		embd: 64,
		nH:   4,
		nKV:  2,
		hd:   16,
		ffn:  96,
		eps:  1e-6,
		base: 10000,
	}
	qDim, kvDim := l.nH*l.hd, l.nKV*l.hd
	l.q, l.k, l.v = fakeQ8(qDim, l.embd, 1), fakeQ8(kvDim, l.embd, 2), fakeQ8(kvDim, l.embd, 3)
	l.wo = fakeQ8(l.embd, qDim, 4)
	l.gate, l.up, l.down = fakeQ8(l.ffn, l.embd, 5), fakeQ8(l.ffn, l.embd, 6), fakeQ8(l.embd, l.ffn, 7)
	l.attnNorm, l.ffnNorm = detFloats(l.embd, 8), detFloats(l.embd, 9)
	l.qn, l.kn = detFloats(l.hd, 10), detFloats(l.hd, 11)

	return l
}

// reference runs the block on the host (compute package only) and returns the new hidden state; kc/vc are the host KV history (appended)
func (l *layerFixture) reference(x []float32, pos int, kc, vc *[]float32) []float32 {
	qDim, kvDim := l.nH*l.hd, l.nKV*l.hd
	h := make([]float32, l.embd)
	_ = compute.RMSNormInto(h, x, l.attnNorm, l.eps)

	q, k, v := make([]float32, qDim), make([]float32, kvDim), make([]float32, kvDim)
	_ = compute.MatMulVecQ8_0Into(l.q, qDim, l.embd, h, q)
	_ = compute.MatMulVecQ8_0Into(l.k, kvDim, l.embd, h, k)
	_ = compute.MatMulVecQ8_0Into(l.v, kvDim, l.embd, h, v)

	for i := range l.nH {
		_ = compute.RMSNormInto(q[i*l.hd:(i+1)*l.hd], q[i*l.hd:(i+1)*l.hd], l.qn, l.eps)
	}

	for i := range l.nKV {
		_ = compute.RMSNormInto(k[i*l.hd:(i+1)*l.hd], k[i*l.hd:(i+1)*l.hd], l.kn, l.eps)
	}

	compute.ApplyRoPEHeads(q, l.nH, l.hd, pos, l.base)
	compute.ApplyRoPEHeads(k, l.nKV, l.hd, pos, l.base)
	*kc, *vc = append(*kc, k...), append(*vc, v...)

	seq := len(*kc) / kvDim
	attn := make([]float32, qDim)
	_ = compute.AttentionScoresInto(attn, q, *kc, *vc, make([]float32, seq), seq, l.nH, l.nKV, l.hd)

	out := append([]float32(nil), x...)
	wo := make([]float32, l.embd)
	_ = compute.MatMulVecQ8_0Into(l.wo, l.embd, qDim, attn, wo)
	for i := range out {
		out[i] += wo[i]
	}

	hn := make([]float32, l.embd)
	_ = compute.RMSNormInto(hn, out, l.ffnNorm, l.eps)

	g, u := make([]float32, l.ffn), make([]float32, l.ffn)
	_ = compute.MatMulVecQ8_0Into(l.gate, l.ffn, l.embd, hn, g)
	_ = compute.MatMulVecQ8_0Into(l.up, l.ffn, l.embd, hn, u)
	compute.SwiGLUInPlace(g, u)

	d := make([]float32, l.embd)
	_ = compute.MatMulVecQ8_0Into(l.down, l.embd, l.ffn, g, d)
	for i := range out {
		out[i] += d[i]
	}

	return out
}

func (l *layerFixture) cosSin(pos int) (cos, sin []float32) {
	cos, sin = make([]float32, l.hd/2), make([]float32, l.hd/2)
	compute.RoPECosSin(cos, sin, l.hd, pos, l.base)

	return cos, sin
}

// qkv calls the Backend fused QKV; h == nil means "from resident hidden"
func (l *layerFixture) qkv(b *Backend, layer, pos int, h, attn, kOut, vOut []float32) error {
	cos, sin := l.cosSin(pos)

	return b.QKVRoPEAttentionQuantCached(dtype.Q8_0, compute.RoPENeoX, "q", "k", "v", "qn", "kn", "an", l.q, l.k, l.v, l.qn, l.kn, l.attnNorm, h, cos, sin, attn, kOut, vOut, l.embd, l.nH, l.nKV, l.hd, layer, pos, pos+1, l.eps)
}

func (l *layerFixture) attnFFN(b *Backend, x, attn []float32) error {
	return b.AttnFFNResidualQuantCached(dtype.Q8_0, "wo", "fn", "gate", "up", "down", l.wo, l.gate, l.up, l.down, l.ffnNorm, x, attn, l.embd, l.nH*l.hd, l.ffn, l.eps)
}

func newFusedBackend(t *testing.T, l *layerFixture) (*Backend, *fakeGPU) {
	t.Helper()

	f := &fakeGPU{}
	b := &Backend{rt: f}
	if err := b.KVCacheInit(1, 16, l.nKV*l.hd, l.nH, l.hd); err != nil {
		t.Fatal(err)
	}

	return b, f
}

// ---- tests ----

func TestHiddenResidentCapability(t *testing.T) {
	var scaffold Backend
	if scaffold.HiddenResident() || scaffold.HiddenActive() {
		t.Fatal("scaffold Backend: HiddenResident/HiddenActive must be false")
	}

	if err := scaffold.HiddenUpload([]float32{1}); !errors.Is(err, ErrUnavailable) {
		t.Fatalf("scaffold HiddenUpload = %v, expected ErrUnavailable", err)
	}

	if err := scaffold.HiddenDownload(make([]float32, 1)); !errors.Is(err, ErrUnavailable) {
		t.Fatalf("scaffold HiddenDownload = %v, expected ErrUnavailable", err)
	}

	if err := scaffold.LogitsFromDevice(dtype.Q8_0, "n", nil, "h", nil, nil, nil, 1, 1, 0); !errors.Is(err, ErrUnavailable) {
		t.Fatalf("scaffold LogitsFromDevice = %v, expected ErrUnavailable", err)
	}

	// runtime without the residency kernels: capability false, entry points unavailable
	noRes := &Backend{rt: &fakeGPU{noResident: true}}
	if noRes.HiddenResident() {
		t.Fatal("HiddenResident() = true without residency kernels")
	}

	if err := noRes.HiddenUpload([]float32{1}); !errors.Is(err, ErrUnavailable) {
		t.Fatalf("HiddenUpload without kernels = %v, expected ErrUnavailable", err)
	}

	b := &Backend{rt: &fakeGPU{}}
	if !b.HiddenResident() {
		t.Fatal("HiddenResident() = false with a runtime that has the kernels")
	}

	if b.HiddenActive() {
		t.Fatal("HiddenActive() = true before HiddenUpload")
	}

	// kernels planned for residency are in the plan
	for _, name := range B2Kernels {
		if !plannedKernelNamed(name) {
			t.Fatalf("B2 kernel %q missing from PlannedKernels", name)
		}
	}
}

func plannedKernelNamed(name string) bool {
	for _, k := range PlannedKernels {
		if k.Name == name {
			return true
		}
	}

	return false
}

func TestHiddenUploadDownloadClearsResidency(t *testing.T) {
	f := &fakeGPU{}
	b := &Backend{rt: f}

	if err := b.HiddenUpload(nil); err == nil {
		t.Fatal("HiddenUpload(nil): expected error")
	}

	if err := b.HiddenDownload(make([]float32, 4)); err == nil {
		t.Fatal("HiddenDownload without residency: expected error")
	}

	x := detFloats(8, 21)
	if err := b.HiddenUpload(x); err != nil {
		t.Fatal(err)
	}

	if !b.HiddenActive() {
		t.Fatal("HiddenActive() = false after HiddenUpload")
	}

	if err := b.HiddenDownload(nil); err == nil {
		t.Fatal("HiddenDownload(nil): expected error")
	}

	if err := b.HiddenDownload(make([]float32, 9)); err == nil {
		t.Fatal("HiddenDownload longer than the resident hidden: expected error")
	}

	if !b.HiddenActive() {
		t.Fatal("rejected HiddenDownload must keep residency")
	}

	got := make([]float32, 8)
	if err := b.HiddenDownload(got); err != nil {
		t.Fatal(err)
	}

	approxEqual(t, "HiddenDownload", got, x, 0)

	if b.HiddenActive() {
		t.Fatal("HiddenActive() = true after HiddenDownload (download must clear residency)")
	}

	if err := b.HiddenDownload(got); err == nil {
		t.Fatal("second HiddenDownload: expected error")
	}
}

// resident decode (h == nil, x == nil) must equal the host reference over several tokens, and the host x buffer stays untouched
func TestFusedResidentLayerParity(t *testing.T) {
	l := newLayerFixture()
	b, f := newFusedBackend(t, l)

	var kc, vc []float32
	x := detFloats(l.embd, 100)
	for pos := range 3 {
		want := l.reference(x, pos, &kc, &vc)

		if err := b.HiddenUpload(x); err != nil {
			t.Fatal(err)
		}

		if err := l.qkv(b, 0, pos, nil, nil, nil, nil); err != nil {
			t.Fatalf("pos %d qkv: %v", pos, err)
		}

		if !b.HiddenActive() {
			t.Fatal("QKV (read-only on resid) must keep residency")
		}

		if err := l.attnFFN(b, nil, nil); err != nil {
			t.Fatalf("pos %d attnFFN: %v", pos, err)
		}

		got := make([]float32, l.embd)
		if err := b.HiddenDownload(got); err != nil {
			t.Fatal(err)
		}

		approxEqual(t, fmt.Sprintf("resident layer pos %d", pos), got, want, 1e-4)
		x = want
	}

	// attention consumed by AttnFFN: a second AttnFFN without a new QKV has no device attention
	if err := b.HiddenUpload(x); err != nil {
		t.Fatal(err)
	}

	if err := l.attnFFN(b, nil, nil); err == nil {
		t.Fatal("AttnFFN with attn == nil and no preceding QKV: expected error")
	}

	if !b.HiddenActive() {
		t.Fatal("clean failure must keep residency")
	}

	_ = f
}

// host mode (h and x supplied): same result, host x updated, residency never enabled; attn / k / v mirrors are filled
func TestFusedHostLayerParity(t *testing.T) {
	l := newLayerFixture()
	b, _ := newFusedBackend(t, l)

	var kc, vc []float32
	x := detFloats(l.embd, 200)
	want := l.reference(x, 0, &kc, &vc)

	h := make([]float32, l.embd)
	_ = compute.RMSNormInto(h, x, l.attnNorm, l.eps)

	attn := make([]float32, l.nH*l.hd)
	kOut, vOut := make([]float32, l.nKV*l.hd), make([]float32, l.nKV*l.hd)
	if err := l.qkv(b, 0, 0, h, attn, kOut, vOut); err != nil {
		t.Fatal(err)
	}

	approxEqual(t, "kOut mirrors device K", kOut, kc, 1e-5)
	approxEqual(t, "vOut mirrors device V", vOut, vc, 1e-5)

	got := append([]float32(nil), x...)
	if err := l.attnFFN(b, got, attn); err != nil {
		t.Fatal(err)
	}

	approxEqual(t, "host layer", got, want, 1e-4)

	if b.HiddenActive() {
		t.Fatal("host-mode call must not enable residency")
	}

	// host x with active residency: residency is cleared (host buffer is the source of truth)
	if err := b.HiddenUpload(x); err != nil {
		t.Fatal(err)
	}

	if err := l.qkv(b, 0, 1, h, nil, nil, nil); err != nil {
		t.Fatal(err)
	}

	if err := l.attnFFN(b, append([]float32(nil), x...), nil); err != nil {
		t.Fatal(err)
	}

	if b.HiddenActive() {
		t.Fatal("AttnFFN with host x must clear residency")
	}
}

func TestFusedDirtyFailureClearsResidency(t *testing.T) {
	l := newLayerFixture()
	x := detFloats(l.embd, 300)

	for _, tc := range []struct {
		name       string
		fail       string
		wantActive bool
	}{{"resid modified", "dirty", false}, {"resid intact", "clean", true}} {
		t.Run(tc.name, func(t *testing.T) {
			b, f := newFusedBackend(t, l)
			if err := b.HiddenUpload(x); err != nil {
				t.Fatal(err)
			}

			if err := l.qkv(b, 0, 0, nil, nil, nil, nil); err != nil {
				t.Fatal(err)
			}

			f.failAttnFFN = tc.fail
			if err := l.attnFFN(b, nil, nil); err == nil {
				t.Fatal("expected fused failure")
			}

			if b.HiddenActive() != tc.wantActive {
				t.Fatalf("HiddenActive() = %v after %s failure, want %v", b.HiddenActive(), tc.fail, tc.wantActive)
			}

			if tc.wantActive {
				got := make([]float32, l.embd)
				if err := b.HiddenDownload(got); err != nil {
					t.Fatal(err)
				}

				approxEqual(t, "hidden after clean failure", got, x, 0)
			}
		})
	}

	// a QKV failure never touches resid
	b, f := newFusedBackend(t, l)
	_ = b.HiddenUpload(x)
	f.failQKV = true
	if err := l.qkv(b, 0, 0, nil, nil, nil, nil); err == nil || !b.HiddenActive() {
		t.Fatalf("QKV failure: err=%v active=%v, want error and residency kept", err, b.HiddenActive())
	}
}

func TestFusedContractsAndUnavailable(t *testing.T) {
	l := newLayerFixture()
	b, _ := newFusedBackend(t, l)
	cos, sin := l.cosSin(0)

	// h == nil / x == nil without residency is an error, not a silent no-op
	if err := l.qkv(b, 0, 0, nil, nil, nil, nil); err == nil {
		t.Fatal("QKV with h == nil and no residency: expected error")
	}

	if err := l.attnFFN(b, nil, nil); err == nil {
		t.Fatal("AttnFFN with x == nil and no residency: expected error")
	}

	// non-Q8_0 types and non-NeoX RoPE are not fused: ErrUnavailable so callers fall back to the per-op path
	for _, ty := range []dtype.DType{dtype.Q4_0, dtype.Q4_K, dtype.Q6_K, dtype.F32} {
		err := b.QKVRoPEAttentionQuantCached(ty, compute.RoPENeoX, "q", "k", "v", "qn", "kn", "an", l.q, l.k, l.v, l.qn, l.kn, l.attnNorm, make([]float32, l.embd), cos, sin, nil, nil, nil, l.embd, l.nH, l.nKV, l.hd, 0, 0, 1, l.eps)
		if !errors.Is(err, ErrUnavailable) {
			t.Fatalf("QKV %s = %v, expected ErrUnavailable", ty, err)
		}

		err = b.AttnFFNResidualQuantCached(ty, "wo", "fn", "gate", "up", "down", l.wo, l.gate, l.up, l.down, l.ffnNorm, make([]float32, l.embd), nil, l.embd, l.nH*l.hd, l.ffn, l.eps)
		if !errors.Is(err, ErrUnavailable) {
			t.Fatalf("AttnFFN %s = %v, expected ErrUnavailable", ty, err)
		}

		err = b.FFNSwiGLUQuantCached(ty, "gate", "up", "down", l.gate, l.up, l.down, make([]float32, l.embd), make([]float32, l.embd), l.embd, l.ffn)
		if !errors.Is(err, ErrUnavailable) {
			t.Fatalf("FFNSwiGLU %s = %v, expected ErrUnavailable", ty, err)
		}
	}

	err := b.QKVRoPEAttentionQuantCached(dtype.Q8_0, compute.RoPENorm, "q", "k", "v", "qn", "kn", "an", l.q, l.k, l.v, l.qn, l.kn, l.attnNorm, make([]float32, l.embd), cos, sin, nil, nil, nil, l.embd, l.nH, l.nKV, l.hd, 0, 0, 1, l.eps)
	if !errors.Is(err, ErrUnavailable) {
		t.Fatalf("QKV RoPENorm = %v, expected ErrUnavailable", err)
	}

	// other fused entry points stay unavailable
	if err := b.FFNGeGLUQuantCached(dtype.Q8_0, "", "", "", nil, nil, nil, nil, nil, 0, 0); !errors.Is(err, ErrUnavailable) {
		t.Fatalf("FFNGeGLUQuantCached = %v, expected ErrUnavailable", err)
	}

	if err := b.AttnFFNResidualGeGLUQuantCached(dtype.Q8_0, "", "", "", "", "", "", "", nil, nil, nil, nil, nil, nil, nil, nil, nil, 0, 0, 0, 0); !errors.Is(err, ErrUnavailable) {
		t.Fatalf("AttnFFNResidualGeGLUQuantCached = %v, expected ErrUnavailable", err)
	}

	if err := b.ResidAdd(nil); !errors.Is(err, ErrUnavailable) {
		t.Fatalf("ResidAdd = %v, expected ErrUnavailable", err)
	}

	// validation: short weights / buffers
	if err := b.AttnFFNResidualQuantCached(dtype.Q8_0, "wo", "fn", "gate", "up", "down", l.wo[:10], l.gate, l.up, l.down, l.ffnNorm, make([]float32, l.embd), nil, l.embd, l.nH*l.hd, l.ffn, l.eps); err == nil {
		t.Fatal("short WO matrix: expected error")
	}

	if err := b.QKVRoPEAttentionQuantCached(dtype.Q8_0, compute.RoPENeoX, "q", "k", "v", "qn", "kn", "an", l.q, l.k, l.v, l.qn, l.kn, l.attnNorm, make([]float32, l.embd), cos[:2], sin, nil, nil, nil, l.embd, l.nH, l.nKV, l.hd, 0, 0, 1, l.eps); err == nil {
		t.Fatal("short cos: expected error")
	}
}

func TestFusedLogitsFromDevice(t *testing.T) {
	const embd, vocab = 64, 40
	x, norm := detFloats(embd, 400), detFloats(embd, 401)

	b := &Backend{
		rt: &fakeGPU{},
	}
	logits := make([]float32, vocab)

	// requires residency
	if err := b.LogitsFromDevice(dtype.Q8_0, "on", norm, "head", fakeQ8(vocab, embd, 402), nil, logits, vocab, embd, 1e-6); err == nil {
		t.Fatal("LogitsFromDevice without residency: expected error")
	}

	if err := b.HiddenUpload(x); err != nil {
		t.Fatal(err)
	}

	h := make([]float32, embd)
	_ = compute.RMSNormInto(h, x, norm, 1e-6)

	// Q8_0 head
	headQ8 := fakeQ8(vocab, embd, 402)
	want, _ := compute.MatMulVecQ8_0(headQ8, vocab, embd, h)
	if err := b.LogitsFromDevice(dtype.Q8_0, "on", norm, "head", headQ8, nil, logits, vocab, embd, 1e-6); err != nil {
		t.Fatal(err)
	}

	approxEqual(t, "logits Q8_0", logits, want, 1e-4)

	if !b.HiddenActive() {
		t.Fatal("LogitsFromDevice must keep residency (caller downloads / releases)")
	}

	// F32 head
	headF32 := detFloats(vocab*embd, 403)
	wantF32, _ := compute.MatMulVec(headF32, vocab, embd, h)
	if err := b.LogitsFromDevice(dtype.F32, "on", norm, "head32", nil, headF32, logits, vocab, embd, 1e-6); err != nil {
		t.Fatal(err)
	}

	approxEqual(t, "logits F32", logits, wantF32, 1e-4)

	// Q4_0 / Q4_K heads (embd 256 for the K-quant)
	for _, tc := range []struct {
		t         dtype.DType
		qk, bytes int
		embd      int
	}{{dtype.Q4_0, quant.QK4_0, quant.BlockQ4_0Size, 64}, {dtype.Q4_K, quant.QK_K, quant.BlockQ4_KSize, 256}} {
		xs, ns := detFloats(tc.embd, 410), detFloats(tc.embd, 411)
		if err := b.HiddenUpload(xs); err != nil {
			t.Fatal(err)
		}

		raw := fakeQ4(vocab, tc.embd, tc.qk, tc.bytes, 412)
		hs := make([]float32, tc.embd)
		_ = compute.RMSNormInto(hs, xs, ns, 1e-6)

		var wantQ []float32
		if tc.t == dtype.Q4_0 {
			wantQ, _ = compute.MatMulVecQ4_0(raw, vocab, tc.embd, hs)
		} else {
			wantQ, _ = compute.MatMulVecQ4_K(raw, vocab, tc.embd, hs)
		}

		if err := b.LogitsFromDevice(tc.t, "on"+tc.t.String(), ns, "head"+tc.t.String(), raw, nil, logits, vocab, tc.embd, 1e-6); err != nil {
			t.Fatalf("%s: %v", tc.t, err)
		}

		approxEqual(t, "logits "+tc.t.String(), logits, wantQ, 1e-3)
	}

	// embd mismatch with the resident hidden / unsupported head type / short buffers
	if err := b.LogitsFromDevice(dtype.Q8_0, "on", norm, "head", headQ8, nil, logits, vocab, embd, 1e-6); err == nil {
		t.Fatal("embd != resident hidden length: expected error")
	}

	if err := b.LogitsFromDevice(dtype.Q6_K, "on", norm, "head", headQ8, nil, logits, vocab, embd, 1e-6); !errors.Is(err, ErrUnavailable) {
		t.Fatalf("Q6_K head = %v, expected ErrUnavailable", err)
	}

	if err := b.LogitsFromDevice(dtype.Q8_0, "on", norm, "head", headQ8, nil, logits[:3], vocab, embd, 1e-6); err == nil {
		t.Fatal("short logits: expected error")
	}
}

func TestFusedFFNSwiGLUQuantCached(t *testing.T) {
	l := newLayerFixture()
	b, _ := newFusedBackend(t, l)

	x := detFloats(l.embd, 500)
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

	approxEqual(t, "FFNSwiGLUQuantCached", out, want, 1e-4)

	if err := b.FFNSwiGLUQuantCached(dtype.Q8_0, "gate", "up", "down", l.gate, l.up, l.down, nil, out, l.embd, l.ffn); !errors.Is(err, ErrUnavailable) {
		t.Fatalf("x == nil (MoE snapshot) = %v, expected ErrUnavailable", err)
	}

	if err := b.FFNSwiGLUQuantCached(dtype.Q8_0, "gate", "up", "down", l.gate, l.up, l.down, x, out[:3], l.embd, l.ffn); err == nil {
		t.Fatal("short out: expected error")
	}
}

// Wave 3: Q4_0 / Q4_K matvec entry points validate geometry and dispatch to the runtime kernel
func TestBackendQ4MatVec(t *testing.T) {
	b := &Backend{rt: &fakeGPU{}}

	for _, tc := range []struct {
		name      string
		t         dtype.DType
		qk, bytes int
		call      func(string, []byte, int, int, []float32) ([]float32, error)
		ref       func([]byte, int, int, []float32) ([]float32, error)
	}{
		{"Q4_0", dtype.Q4_0, quant.QK4_0, quant.BlockQ4_0Size, b.MatMulVecQ4_0Cached, compute.MatMulVecQ4_0},
		{"Q4_K", dtype.Q4_K, quant.QK_K, quant.BlockQ4_KSize, b.MatMulVecQ4_KCached, compute.MatMulVecQ4_K},
	} {
		rows, cols := 5, tc.qk*2
		raw := fakeQ4(rows, cols, tc.qk, tc.bytes, 600)
		vec := detFloats(cols, 601)

		want, err := tc.ref(raw, rows, cols, vec)
		if err != nil {
			t.Fatal(err)
		}

		got, err := tc.call("w."+tc.name, raw, rows, cols, vec)
		if err != nil {
			t.Fatalf("%s: %v", tc.name, err)
		}

		approxEqual(t, tc.name, got, want, 0)

		if _, err := tc.call("w", raw, rows, cols+1, vec); err == nil {
			t.Fatalf("%s: expected cols%%%d error", tc.name, tc.qk)
		}

		if _, err := tc.call("w", raw[:len(raw)-1], rows, cols, vec); err == nil {
			t.Fatalf("%s: expected short-matrix error", tc.name)
		}

		if _, err := tc.call("w", raw, rows, cols, vec[:cols-1]); err == nil {
			t.Fatalf("%s: expected short-vec error", tc.name)
		}
	}

	var scaffold Backend
	if _, err := scaffold.MatMulVecQ4_0Cached("w", nil, 1, 32, nil); !errors.Is(err, ErrUnavailable) {
		t.Fatalf("scaffold Q4_0 = %v, expected ErrUnavailable", err)
	}

	if _, err := scaffold.MatMulVecQ4_KCached("w", nil, 1, 256, nil); !errors.Is(err, ErrUnavailable) {
		t.Fatalf("scaffold Q4_K = %v, expected ErrUnavailable", err)
	}

	// quants without a kernel stay unavailable
	if _, err := b.MatMulVecQ5_KCached("w", nil, 1, 256, nil); !errors.Is(err, ErrUnavailable) {
		t.Fatalf("Q5_K = %v, expected ErrUnavailable", err)
	}
}
