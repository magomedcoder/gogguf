//go:build cuda

package cuda

import (
	"encoding/binary"
	"math"
	"testing"

	"github.com/magomedcoder/gogguf/pkg/format"
	"github.com/magomedcoder/gogguf/pkg/ops"
	"github.com/magomedcoder/gogguf/pkg/quant"
)

// makeQ8Matrix builds deterministic Q8_0 matrix rows x cols and returns raw bytes with exact dequant values (for CPU reference)
func makeQ8Matrix(t *testing.T, rows, cols, seed int) ([]byte, []float32) {
	t.Helper()

	if cols%quant.QK8_0 != 0 {
		t.Fatalf("cols=%d не кратно %d", cols, quant.QK8_0)
	}

	// exact fp16 scales (2^-7, 2^-6, 2^-5) so CPU reference matches bit-for-bit
	scales := []uint16{0x2000, 0x2400, 0x2800}

	blocks := cols / quant.QK8_0
	raw := make([]byte, rows*blocks*quant.BlockQ8_0Size)
	ref := make([]float32, rows*cols)
	for r := 0; r < rows; r++ {
		for bi := 0; bi < blocks; bi++ {
			off := (r*blocks + bi) * quant.BlockQ8_0Size
			bits := scales[(r+bi+seed)%len(scales)]
			d := quant.FP16ToFP32(bits)
			binary.LittleEndian.PutUint16(raw[off:off+2], bits)
			for i := 0; i < quant.QK8_0; i++ {
				q := int8((r*7+bi*5+i*3+seed)%31 - 15)
				raw[off+2+i] = byte(q)
				ref[r*cols+bi*quant.QK8_0+i] = d * float32(q)
			}
		}
	}

	return raw, ref
}

// §1: HiddenUpload/HiddenDownload - hidden state survives roundtrip via d_resid
func TestHiddenUploadDownloadRoundtrip(t *testing.T) {
	b, err := Open()
	if err != nil {
		t.Skip(err)
	}
	defer b.Close()

	embd := 64
	x := make([]float32, embd)
	for i := range x {
		x[i] = float32(i%13)*0.25 - 1.5
	}

	if err := b.HiddenUpload(x); err != nil {
		t.Fatal(err)
	}

	if !b.HiddenActive() {
		t.Fatal("после HiddenUpload ожидали HiddenActive() == true")
	}

	got := make([]float32, embd)
	if err := b.HiddenDownload(got); err != nil {
		t.Fatal(err)
	}

	for i := range x {
		if got[i] != x[i] {
			t.Fatalf("hidden[%d]=%v want %v", i, got[i], x[i])
		}
	}

	// re-uploading another vector overwrites d_resid
	for i := range x {
		x[i] = -x[i]
	}

	if err := b.HiddenUpload(x); err != nil {
		t.Fatal(err)
	}

	if err := b.HiddenDownload(got); err != nil {
		t.Fatal(err)
	}

	for i := range x {
		if got[i] != x[i] {
			t.Fatalf("повторный hidden[%d]=%v want %v", i, got[i], x[i])
		}
	}
}

// §1+§3: attn+ffn residual with keep-on-device gives same x as host path
func TestAttnFFNResidualDeviceResident(t *testing.T) {
	b, err := Open()
	if err != nil {
		t.Skip(err)
	}
	defer b.Close()

	if !b.hasSwiGLU || !b.hasRMS || !b.hasAdd {
		t.Skip("нет add/rmsnorm/swiglu")
	}

	embd, attnDim, ffn := 32, 32, 64
	eps := float32(1e-6)
	woRaw, wo := makeQ8Matrix(t, embd, attnDim, 1)
	gateRaw, gate := makeQ8Matrix(t, ffn, embd, 2)
	upRaw, up := makeQ8Matrix(t, ffn, embd, 3)
	downRaw, down := makeQ8Matrix(t, embd, ffn, 4)

	x := make([]float32, embd)
	ffnNorm := make([]float32, embd)
	attn := make([]float32, attnDim)
	for i := range x {
		x[i] = float32(i%9)*0.05 - 0.2
		ffnNorm[i] = 1 + float32(i%4)*0.1
	}

	for i := range attn {
		attn[i] = float32((i%5)+1) * 0.05
	}

	// reference on CPU
	want := append([]float32(nil), x...)
	h := make([]float32, embd)
	if err := ops.MatMulVecInto(wo, embd, attnDim, attn, h); err != nil {
		t.Fatal(err)
	}

	ops.AddInPlace(want, h)
	normed := make([]float32, embd)
	if err := ops.RMSNormInto(normed, want, ffnNorm, eps); err != nil {
		t.Fatal(err)
	}

	gateV, err := ops.MatMulVec(gate, ffn, embd, normed)
	if err != nil {
		t.Fatal(err)
	}

	upV, err := ops.MatMulVec(up, ffn, embd, normed)
	if err != nil {
		t.Fatal(err)
	}

	ops.SwiGLUInPlace(gateV, upV)
	downV, err := ops.MatMulVec(down, embd, ffn, gateV)
	if err != nil {
		t.Fatal(err)
	}

	ops.AddInPlace(want, downV)

	// host path: x at input and output lives in host memory
	host := append([]float32(nil), x...)
	if err := b.AttnFFNResidualQuantCached(format.GgmlQ8_0, "wo", "norm", "g", "u", "d",
		woRaw, gateRaw, upRaw, downRaw,
		ffnNorm, host, attn, embd, attnDim, ffn, eps); err != nil {
		t.Fatal(err)
	}

	for i := range want {
		if math.Abs(float64(host[i]-want[i])) > 2e-3 {
			t.Fatalf("host x[%d]=%v want %v", i, host[i], want[i])
		}
	}

	// device-resident: x via HiddenUpload, result stays in d_resid
	if err := b.HiddenUpload(x); err != nil {
		t.Fatal(err)
	}

	if err := b.AttnFFNResidualQuantCached(format.GgmlQ8_0, "wo", "norm", "g", "u", "d",
		woRaw, gateRaw, upRaw, downRaw,
		ffnNorm, nil, attn, embd, attnDim, ffn, eps); err != nil {
		t.Fatal(err)
	}

	if !b.HiddenActive() {
		t.Fatal("после resident residual ожидали HiddenActive() == true")
	}

	got := make([]float32, embd)
	if err := b.HiddenDownload(got); err != nil {
		t.Fatal(err)
	}

	for i := range want {
		if math.Abs(float64(got[i]-want[i])) > 2e-3 {
			t.Fatalf("resident x[%d]=%v want %v", i, got[i], want[i])
		}
	}

	// host call after resident clears residency: host is source of truth again
	host2 := append([]float32(nil), x...)
	if err := b.AttnFFNResidualQuantCached(format.GgmlQ8_0, "wo", "norm", "g", "u", "d",
		woRaw, gateRaw, upRaw, downRaw,
		ffnNorm, host2, attn, embd, attnDim, ffn, eps); err != nil {
		t.Fatal(err)
	}

	if b.HiddenActive() {
		t.Fatal("после host-вызова ожидали HiddenActive() == false")
	}

	for i := range want {
		if math.Abs(float64(host2[i]-want[i])) > 2e-3 {
			t.Fatalf("host2 x[%d]=%v want %v", i, host2[i], want[i])
		}
	}
}

// makeQ4Matrix builds Q4_0 matrix rows x cols: raw + exact dequant values
func makeQ4Matrix(t *testing.T, rows, cols, seed int) ([]byte, []float32) {
	t.Helper()

	if cols%quant.QK4_0 != 0 {
		t.Fatalf("cols=%d не кратно %d", cols, quant.QK4_0)
	}

	scales := []uint16{0x3000, 0x3400, 0x3800} // 0.125, 0.25, 0.5

	blocks := cols / quant.QK4_0
	raw := make([]byte, rows*blocks*quant.BlockQ4_0Size)
	for r := 0; r < rows; r++ {
		for bi := 0; bi < blocks; bi++ {
			off := (r*blocks + bi) * quant.BlockQ4_0Size
			binary.LittleEndian.PutUint16(raw[off:off+2], scales[(r+bi+seed)%len(scales)])
			for i := 0; i < quant.QK4_0/2; i++ {
				lo := byte((r*3 + bi + i + seed) % 16)
				hi := byte((r*5 + bi*2 + i*7 + seed) % 16)
				raw[off+2+i] = lo | (hi << 4)
			}
		}
	}

	ref, err := quant.DequantQ4_0(raw, rows*cols)
	if err != nil {
		t.Fatal(err)
	}

	return raw, ref
}

// §3: fused FFN SwiGLU on Q4_0 weights matches CPU on dequant weights
func TestFFNSwiGLUQuantQ4_0(t *testing.T) {
	b, err := Open()
	if err != nil {
		t.Skip(err)
	}
	defer b.Close()

	if !b.hasSwiGLU || !b.hasQ4 {
		t.Skip("нет swiglu/q4_0 kernels")
	}

	embd, ffn := 32, 64
	gateRaw, gate := makeQ4Matrix(t, ffn, embd, 1)
	upRaw, up := makeQ4Matrix(t, ffn, embd, 2)
	downRaw, down := makeQ4Matrix(t, embd, ffn, 3)

	x := make([]float32, embd)
	for i := range x {
		x[i] = float32(i%7)*0.05 - 0.15
	}

	gateV, err := ops.MatMulVec(gate, ffn, embd, x)
	if err != nil {
		t.Fatal(err)
	}

	upV, err := ops.MatMulVec(up, ffn, embd, x)
	if err != nil {
		t.Fatal(err)
	}

	ops.SwiGLUInPlace(gateV, upV)
	want, err := ops.MatMulVec(down, embd, ffn, gateV)
	if err != nil {
		t.Fatal(err)
	}

	got := make([]float32, embd)
	if err := b.FFNSwiGLUQuantCached(format.GgmlQ4_0, "g4", "u4", "d4", gateRaw, upRaw, downRaw, x, got, embd, ffn); err != nil {
		t.Fatal(err)
	}

	for i := range want {
		if math.Abs(float64(got[i]-want[i])) > 3e-3 {
			t.Fatalf("ffn[%d]=%v want %v", i, got[i], want[i])
		}
	}
}

// §1: QKV with h from d_resid (RMSNorm on device) matches host path with same RMSNorm on CPU
func TestQKVRoPEAttentionFromDevice(t *testing.T) {
	b, err := Open()
	if err != nil {
		t.Skip(err)
	}
	defer b.Close()

	if !b.hasAttn || !b.hasRoPE || !b.hasRMS {
		t.Skip("нет QKV/RoPE/attn kernels")
	}

	embd, nHeads, nKVHeads, headDim := 32, 2, 1, 16
	qDim := nHeads * headDim
	kvDim := nKVHeads * headDim
	eps := float32(1e-6)
	freqBase := float32(10000)

	if err := b.KVCacheInit(1, 8, kvDim, nHeads, headDim); err != nil {
		t.Skip(err)
	}

	wqRaw, _ := makeQ8Matrix(t, qDim, embd, 6)
	wkRaw, _ := makeQ8Matrix(t, kvDim, embd, 7)
	wvRaw, _ := makeQ8Matrix(t, kvDim, embd, 8)

	x := make([]float32, embd)
	attnNorm := make([]float32, embd)
	qNorm := make([]float32, headDim)
	kNorm := make([]float32, headDim)
	for i := range x {
		x[i] = float32(i%11)*0.05 - 0.25
		attnNorm[i] = 1 + float32(i%5)*0.05
	}

	for i := range qNorm {
		qNorm[i] = 1
		kNorm[i] = 1
	}

	half := headDim / 2
	cos := make([]float32, half)
	sin := make([]float32, half)
	ops.RoPECosSin(cos, sin, headDim, 0, freqBase)

	// host path: RMSNorm(x, attnNorm) on CPU, pass h to GPU
	h := make([]float32, embd)
	if err := ops.RMSNormInto(h, x, attnNorm, eps); err != nil {
		t.Fatal(err)
	}

	wantAttn := make([]float32, qDim)
	wantK := make([]float32, kvDim)
	wantV := make([]float32, kvDim)
	if err := b.QKVRoPEAttentionQuantCached(format.GgmlQ8_0, ops.RoPENeoX, "wq", "wk", "wv", "qn", "kn", "an",
		wqRaw, wkRaw, wvRaw, qNorm, kNorm, attnNorm, h, cos, sin,
		wantAttn, wantK, wantV, embd, nHeads, nKVHeads, headDim, 0, 0, 1, eps); err != nil {
		t.Fatal(err)
	}

	// resident path: x in d_resid, GPU computes RMSNorm
	if err := b.HiddenUpload(x); err != nil {
		t.Fatal(err)
	}

	gotAttn := make([]float32, qDim)
	gotK := make([]float32, kvDim)
	gotV := make([]float32, kvDim)
	if err := b.QKVRoPEAttentionQuantCached(format.GgmlQ8_0, ops.RoPENeoX, "wq", "wk", "wv", "qn", "kn", "an",
		wqRaw, wkRaw, wvRaw, qNorm, kNorm, attnNorm, nil, cos, sin,
		gotAttn, gotK, gotV, embd, nHeads, nKVHeads, headDim, 0, 0, 1, eps); err != nil {
		t.Fatal(err)
	}

	// QKV does not change residual, hidden should stay on device
	if !b.HiddenActive() {
		t.Fatal("после resident QKV ожидали HiddenActive() == true")
	}

	for i := range wantAttn {
		if math.Abs(float64(gotAttn[i]-wantAttn[i])) > 2e-3 {
			t.Fatalf("attn[%d]=%v want %v", i, gotAttn[i], wantAttn[i])
		}
	}

	for i := range wantK {
		if math.Abs(float64(gotK[i]-wantK[i])) > 2e-3 {
			t.Fatalf("k[%d]=%v want %v", i, gotK[i], wantK[i])
		}

		if math.Abs(float64(gotV[i]-wantV[i])) > 2e-3 {
			t.Fatalf("v[%d]=%v want %v", i, gotV[i], wantV[i])
		}
	}

	// hidden in d_resid not corrupted
	back := make([]float32, embd)
	if err := b.HiddenDownload(back); err != nil {
		t.Fatal(err)
	}

	for i := range x {
		if back[i] != x[i] {
			t.Fatalf("resid[%d]=%v want %v", i, back[i], x[i])
		}
	}
}

// §2: LogitsFromDevice = rmsnorm(d_resid) + lm_head without downloading hidden to host
func TestLogitsFromDeviceFP32(t *testing.T) {
	b, err := Open()
	if err != nil {
		t.Skip(err)
	}
	defer b.Close()

	if !b.hasRMS {
		t.Skip("нет rmsnorm")
	}

	embd, vocab := 32, 48
	eps := float32(1e-6)
	x := make([]float32, embd)
	norm := make([]float32, embd)
	head := make([]float32, vocab*embd)
	for i := range x {
		x[i] = float32(i%9)*0.1 - 0.4
		norm[i] = 1 + float32(i%3)*0.1
	}

	for i := range head {
		head[i] = float32((i%13)-6) * 0.01
	}

	normed := make([]float32, embd)
	if err := ops.RMSNormInto(normed, x, norm, eps); err != nil {
		t.Fatal(err)
	}

	want, err := ops.MatMulVec(head, vocab, embd, normed)
	if err != nil {
		t.Fatal(err)
	}

	logits := make([]float32, vocab)
	for pass := 0; pass < 2; pass++ {
		if err := b.HiddenUpload(x); err != nil {
			t.Fatal(err)
		}

		for i := range logits {
			logits[i] = 0
		}

		if err := b.LogitsFromDevice(format.GgmlFloat32, "out_norm", norm, "lm_head", nil, head, logits, vocab, embd, eps); err != nil {
			t.Fatal(err)
		}

		for i := range want {
			if math.Abs(float64(logits[i]-want[i])) > 2e-3 {
				t.Fatalf("pass %d: logits[%d]=%v want %v", pass, i, logits[i], want[i])
			}
		}
	}
}

// §2: same path but lm_head quantized as Q8_0
func TestLogitsFromDeviceQ8(t *testing.T) {
	b, err := Open()
	if err != nil {
		t.Skip(err)
	}
	defer b.Close()

	if !b.hasRMS {
		t.Skip("нет rmsnorm")
	}

	embd, vocab := 64, 96
	eps := float32(1e-6)
	headRaw, head := makeQ8Matrix(t, vocab, embd, 5)

	x := make([]float32, embd)
	norm := make([]float32, embd)
	for i := range x {
		x[i] = float32(i%9)*0.1 - 0.4
		norm[i] = 1 + float32(i%3)*0.1
	}

	normed := make([]float32, embd)
	if err := ops.RMSNormInto(normed, x, norm, eps); err != nil {
		t.Fatal(err)
	}

	want, err := ops.MatMulVec(head, vocab, embd, normed)
	if err != nil {
		t.Fatal(err)
	}

	if err := b.HiddenUpload(x); err != nil {
		t.Fatal(err)
	}

	logits := make([]float32, vocab)
	if err := b.LogitsFromDevice(format.GgmlQ8_0, "out_norm2", norm, "lm_head_q8", headRaw, nil, logits, vocab, embd, eps); err != nil {
		t.Fatal(err)
	}

	for i := range want {
		if math.Abs(float64(logits[i]-want[i])) > 2e-3 {
			t.Fatalf("logits[%d]=%v want %v", i, logits[i], want[i])
		}
	}
}

// §4: KVCacheAppendN stores n tokens and attention sees them all
func TestKVCacheAppendN(t *testing.T) {
	b, err := Open()
	if err != nil {
		t.Skip(err)
	}
	defer b.Close()

	if !b.hasAttn {
		t.Skip("нет attn kernels")
	}

	nHeads, nKVHeads, headDim, maxSeq := 2, 1, 4, 16
	kvDim := nKVHeads * headDim
	if err := b.KVCacheInit(1, maxSeq, kvDim, nHeads, headDim); err != nil {
		t.Skip(err)
	}

	// two consecutive chunks: AppendN should pack tightly at pos=0 and pos=n1
	n1, n2 := 3, 4
	seq := n1 + n2
	k := make([]float32, seq*kvDim)
	v := make([]float32, seq*kvDim)
	for i := range k {
		k[i] = float32(i%7)*0.1 - 0.3
		v[i] = float32(i%5)*0.2 - 0.4
	}

	if err := b.KVCacheAppendN(0, 0, k[:n1*kvDim], v[:n1*kvDim], n1); err != nil {
		t.Fatal(err)
	}

	if err := b.KVCacheAppendN(0, n1, k[n1*kvDim:], v[n1*kvDim:], n2); err != nil {
		t.Fatal(err)
	}

	q := make([]float32, nHeads*headDim)
	for i := range q {
		q[i] = float32(i%3)*0.3 + 0.1
	}

	want := make([]float32, nHeads*headDim)
	scores := make([]float32, seq)
	if err := ops.AttentionScoresInto(want, q, k, v, scores, seq, nHeads, nKVHeads, headDim); err != nil {
		t.Fatal(err)
	}

	got := make([]float32, nHeads*headDim)
	if err := b.AttentionScoresKV(0, got, q, seq, nHeads, nKVHeads, headDim); err != nil {
		t.Fatal(err)
	}

	for i := range want {
		if math.Abs(float64(got[i]-want[i])) > 1e-4 {
			t.Fatalf("attn[%d]=%v want %v", i, got[i], want[i])
		}
	}
}
