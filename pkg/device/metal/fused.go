package metal

import (
	"fmt"

	"github.com/magomedcoder/gogguf/pkg/compute"
	"github.com/magomedcoder/gogguf/pkg/dtype"
)

// B2: device-resident hidden state and the fused Q8_0 NeoX decode path (QKV+RoPE+attention, WO+residual+FFN+residual, lm_head).
// The contracts mirror pkg/device/cuda (pkg/exec/gpuresid drives both unchanged):
//   - HiddenUpload puts the hidden state into the device residual buffer, HiddenDownload fetches it and clears residency
//   - QKVRoPEAttentionQuantCached with h == nil reads RMSNorm(resid, attn_norm); the attention output stays on the device for AttnFFNResidualQuantCached
//   - AttnFFNResidualQuantCached with x == nil updates resid in place (host x is not touched); with x it stages x, writes the result back and clears residency
//   - after a fused failure HiddenActive() stays true when resid is intact (caller downloads and falls back to CPU), false when resid may have been modified
//
// Backend owns the residency flag (guarded by Backend.mu); the runtime (runtime_darwin.go) only owns the resid MTLBuffer and runs the kernels.
// Only Q8_0 weights + NeoX RoPE are fused for now: other types / modes return ErrUnavailable and the model falls back to the B1 per-op path.

type qkvArgs struct {
	qName, kName, vName, qNormName, kNormName, attnNormName string
	qRaw, kRaw, vRaw                                        []byte
	qNorm, kNorm, attnNorm, h, cos, sin, attn, kOut, vOut   []float32
	embd, nHeads, nKVHeads, headDim, layer, kvPos, seqLen   int
	eps                                                     float32
}

type attnFFNArgs struct {
	woName, ffnNormName, gateName, upName, downName string
	woRaw, gateRaw, upRaw, downRaw                  []byte
	ffnNorm, x, attn                                []float32
	embd, attnDim, ffn                              int
	eps                                             float32
}

type ffnArgs struct {
	gateName, upName, downName string
	gateRaw, upRaw, downRaw    []byte
	x, out                     []float32
	embd, ffn                  int
}

type logitsArgs struct {
	t               dtype.DType
	normName        string
	norm            []float32
	headName        string
	headRaw         []byte
	headF32, logits []float32
	vocab, embd     int
	eps             float32
}

// unsupported is unavailable(op) with the reason (type / RoPE mode not fused yet); errors.Is(err, ErrUnavailable) holds
func unsupported(op, why string) error {
	return fmt.Errorf("%w: %s: %s; per-op B1 path / -ngl 0 stays available", ErrUnavailable, op, why)
}

// HiddenResident reports that the runtime can keep the residual on device: rmsnorm + add + swiglu + attention + Q8_0 matvec kernels are compiled.
// Unlike Ready() it is false for a Backend without runtime; it does not imply every quant is fused (see fusedQ8).
func (b *Backend) HiddenResident() bool {
	return b != nil && b.rt != nil && b.rt.hiddenReady()
}

// HiddenActive the device residual buffer holds the current hidden state
func (b *Backend) HiddenActive() bool {
	if b == nil {
		return false
	}

	b.mu.Lock()
	defer b.mu.Unlock()

	return b.hidden
}

// clearHiddenLocked drops residency (b.mu held)
func (b *Backend) clearHiddenLocked() {
	b.hidden, b.hiddenLen = false, 0
}

// HiddenUpload writes the hidden state to the device residual buffer and enables residency
func (b *Backend) HiddenUpload(x []float32) error {
	rt, err := b.runtime("HiddenUpload")
	if err != nil {
		return err
	}

	if !rt.hiddenReady() {
		return unavailable("HiddenUpload")
	}

	if len(x) == 0 {
		return fmt.Errorf("metal: HiddenUpload: empty x")
	}

	b.mu.Lock()
	defer b.mu.Unlock()

	if err := rt.hiddenUpload(x); err != nil {
		b.clearHiddenLocked()
		return err
	}

	b.hidden, b.hiddenLen = true, len(x)

	return nil
}

// HiddenDownload fetches the residual into dst and disables residency (also when the copy fails: the caller owns recovery)
func (b *Backend) HiddenDownload(dst []float32) error {
	rt, err := b.runtime("HiddenDownload")
	if err != nil {
		return err
	}

	if len(dst) == 0 {
		return fmt.Errorf("metal: HiddenDownload: empty dst")
	}

	b.mu.Lock()
	defer b.mu.Unlock()

	if !b.hidden {
		return fmt.Errorf("metal: HiddenDownload: residency not active")
	}

	if len(dst) > b.hiddenLen {
		return fmt.Errorf("metal: HiddenDownload: len(dst)=%d > resident hidden %d", len(dst), b.hiddenLen)
	}

	err = rt.hiddenDownload(dst)
	b.clearHiddenLocked()

	return err
}

// fusedRuntime returns the runtime when it can run the fused B2 path
func (b *Backend) fusedRuntime(op string) (gpu, error) {
	rt, err := b.runtime(op)
	if err != nil {
		return nil, err
	}

	if !rt.hiddenReady() {
		return nil, unavailable(op)
	}

	return rt, nil
}

// fusedQ8 gates the fused paths: Q8_0 weights only (other quants return ErrUnavailable until their fused kernels exist)
func fusedQ8(op string, t dtype.DType) error {
	if t != dtype.Q8_0 {
		return unsupported(op, fmt.Sprintf("fused path supports Q8_0 only, got %s", t))
	}

	return nil
}

// QKVRoPEAttentionQuantCached [RMSNorm(resid)] -> Q/K/V matvec -> QK-norm -> RoPE (NeoX) -> KV append -> attention, in one command buffer.
// h == nil with active residency: h = RMSNorm(resid, attnNorm) on device.
// attn / kOut / vOut are optional host mirrors (nil = no copy);
// the attention output always stays on the device for the following AttnFFNResidualQuantCached call.
// Empty qNormName / kNormName: layer without QK-norm.
func (b *Backend) QKVRoPEAttentionQuantCached(t dtype.DType, mode compute.RoPEMode, qName, kName, vName, qNormName, kNormName, attnNormName string, qRaw, kRaw, vRaw []byte, qNorm, kNorm, attnNorm, h, cos, sin, attn, kOut, vOut []float32, embd, nHeads, nKVHeads, headDim, layer, kvPos, seqLen int, eps float32) error {
	const op = "QKVRoPEAttentionQuantCached"

	rt, err := b.fusedRuntime(op)
	if err != nil {
		return err
	}

	if err := fusedQ8(op, t); err != nil {
		return err
	}

	if mode != compute.RoPENeoX {
		return unsupported(op, "fused path supports NeoX RoPE only")
	}

	if embd < 1 || nHeads < 1 || nKVHeads < 1 || headDim < 2 || headDim%2 != 0 || nHeads%nKVHeads != 0 || layer < 0 || kvPos < 0 || seqLen < 1 {
		return fmt.Errorf("metal: %s: invalid dimensions embd=%d nHeads=%d nKVHeads=%d headDim=%d layer=%d kvPos=%d seqLen=%d", op, embd, nHeads, nKVHeads, headDim, layer, kvPos, seqLen)
	}

	if qName == "" || kName == "" || vName == "" {
		return fmt.Errorf("metal: %s: weight names are required (device weight cache key)", op)
	}

	qDim, kvDim := nHeads*headDim, nKVHeads*headDim
	if len(cos) < headDim/2 || len(sin) < headDim/2 {
		return fmt.Errorf("metal: %s: cos/sin too short", op)
	}

	// attn / kOut / vOut are optional host mirrors (nil = skip the copy)
	if (len(attn) > 0 && len(attn) < qDim) || (len(kOut) > 0 && len(kOut) < kvDim) || (len(vOut) > 0 && len(vOut) < kvDim) {
		return fmt.Errorf("metal: %s: attn/k/v buffers too short", op)
	}

	if (qNormName != "" && len(qNorm) < headDim) || (kNormName != "" && len(kNorm) < headDim) {
		return fmt.Errorf("metal: %s: qk-norm weights too short", op)
	}

	if err := checkQuantMatrix(op, t, qRaw, qDim, embd); err != nil {
		return err
	}

	if err := checkQuantMatrix(op, t, kRaw, kvDim, embd); err != nil {
		return err
	}

	if err := checkQuantMatrix(op, t, vRaw, kvDim, embd); err != nil {
		return err
	}

	b.mu.Lock()
	defer b.mu.Unlock()

	// h == nil: hidden comes from the resident residual through RMSNorm on device
	resident := b.hidden && len(h) == 0
	if !resident && len(h) < embd {
		return fmt.Errorf("metal: %s: h too short", op)
	}

	if resident {
		if len(attnNorm) < embd || attnNormName == "" {
			return fmt.Errorf("metal: %s: attn_norm too short", op)
		}

		if b.hiddenLen != embd {
			return fmt.Errorf("metal: %s: resident hidden has %d floats, embd=%d", op, b.hiddenLen, embd)
		}
	}

	return rt.qkvRoPEAttention(&qkvArgs{
		qName: qName, kName: kName, vName: vName, qNormName: qNormName, kNormName: kNormName, attnNormName: attnNormName,
		qRaw: qRaw, kRaw: kRaw, vRaw: vRaw,
		qNorm: qNorm, kNorm: kNorm, attnNorm: attnNorm, h: h, cos: cos, sin: sin, attn: attn, kOut: kOut, vOut: vOut,
		embd: embd, nHeads: nHeads, nKVHeads: nKVHeads, headDim: headDim, layer: layer, kvPos: kvPos, seqLen: seqLen,
		eps: eps,
	}, resident)
}

// AttnFFNResidualQuantCached WO(attn) -> residual add -> ffn_norm -> gate/up -> SwiGLU -> down -> residual add, in one command buffer.
// x == nil with active residency: resid is read and updated in place and host x is not touched.
// Otherwise x is the residual: it is staged on the device, the result is written back into x and residency is cleared.
// attn == nil: the attention output left on the device by QKVRoPEAttentionQuantCached is used.
// A failed call keeps HiddenActive() true unless the residual may already have been modified.
func (b *Backend) AttnFFNResidualQuantCached(t dtype.DType, woName, ffnNormName, gateName, upName, downName string, woRaw, gateRaw, upRaw, downRaw []byte, ffnNorm, x, attn []float32, embd, attnDim, ffn int, eps float32) error {
	const op = "AttnFFNResidualQuantCached"

	rt, err := b.fusedRuntime(op)
	if err != nil {
		return err
	}

	if err := fusedQ8(op, t); err != nil {
		return err
	}

	if embd < 1 || attnDim < 1 || ffn < 1 {
		return fmt.Errorf("metal: %s: invalid dimensions embd=%d attnDim=%d ffn=%d", op, embd, attnDim, ffn)
	}

	if woName == "" || ffnNormName == "" || gateName == "" || upName == "" || downName == "" {
		return fmt.Errorf("metal: %s: weight names are required (device weight cache key)", op)
	}

	if len(ffnNorm) < embd {
		return fmt.Errorf("metal: %s: ffn_norm too short", op)
	}

	if len(attn) > 0 && len(attn) < attnDim {
		return fmt.Errorf("metal: %s: attn too short", op)
	}

	for _, c := range []struct {
		raw        []byte
		rows, cols int
	}{{woRaw, embd, attnDim}, {gateRaw, ffn, embd}, {upRaw, ffn, embd}, {downRaw, embd, ffn}} {
		if err := checkQuantMatrix(op, t, c.raw, c.rows, c.cols); err != nil {
			return err
		}
	}

	b.mu.Lock()
	defer b.mu.Unlock()

	// x == nil: explicit residency request, residual stays on device.
	// Otherwise host x is the source of truth and residency is cleared.
	resident := b.hidden && len(x) == 0
	if !resident {
		if len(x) < embd {
			return fmt.Errorf("metal: %s: x too short", op)
		}

		b.clearHiddenLocked()
	} else if b.hiddenLen != embd {
		return fmt.Errorf("metal: %s: resident hidden has %d floats, embd=%d", op, b.hiddenLen, embd)
	}

	dirty, err := rt.attnFFNResidual(&attnFFNArgs{
		woName:      woName,
		ffnNormName: ffnNormName,
		gateName:    gateName,
		upName:      upName,
		downName:    downName,
		woRaw:       woRaw,
		gateRaw:     gateRaw,
		upRaw:       upRaw,
		downRaw:     downRaw,
		ffnNorm:     ffnNorm,
		x:           x,
		attn:        attn,
		embd:        embd,
		attnDim:     attnDim,
		ffn:         ffn,
		eps:         eps,
	}, resident)
	if err != nil {
		// resid was modified: no valid hidden on the host or the device; HiddenActive() == false tells the caller rollback is impossible
		if resident && dirty {
			b.clearHiddenLocked()
		}

		return err
	}

	return nil
}

// FFNSwiGLUQuantCached gate/up matvec + SwiGLU + down matvec on device for Q8_0; activations never leave the device between the three matvecs.
// x is required (the CUDA MoE-snapshot form x == nil has no Metal counterpart).
func (b *Backend) FFNSwiGLUQuantCached(t dtype.DType, gateName, upName, downName string, gateRaw, upRaw, downRaw []byte, x, out []float32, embd, ffn int) error {
	const op = "FFNSwiGLUQuantCached"

	rt, err := b.fusedRuntime(op)
	if err != nil {
		return err
	}

	if err := fusedQ8(op, t); err != nil {
		return err
	}

	if embd < 1 || ffn < 1 {
		return fmt.Errorf("metal: %s: invalid dimensions embd=%d ffn=%d", op, embd, ffn)
	}

	if len(x) == 0 {
		return unsupported(op, "x == nil (MoE norm snapshot) is not supported")
	}

	if len(x) < embd || len(out) < embd {
		return fmt.Errorf("metal: %s: x/out too short", op)
	}

	if gateName == "" || upName == "" || downName == "" {
		return fmt.Errorf("metal: %s: weight names are required (device weight cache key)", op)
	}

	for _, c := range []struct {
		raw        []byte
		rows, cols int
	}{{gateRaw, ffn, embd}, {upRaw, ffn, embd}, {downRaw, embd, ffn}} {
		if err := checkQuantMatrix(op, t, c.raw, c.rows, c.cols); err != nil {
			return err
		}
	}

	return rt.ffnSwiGLU(&ffnArgs{
		gateName: gateName,
		upName:   upName,
		downName: downName,
		gateRaw:  gateRaw,
		upRaw:    upRaw,
		downRaw:  downRaw,
		x:        x[:embd],
		out:      out[:embd],
		embd:     embd,
		ffn:      ffn,
	})
}

// LogitsFromDevice RMSNorm(resident hidden, output_norm) + lm_head matvec on the device; only the logits are copied to the host.
// Head types: headF32 for F32 / F16, headRaw for Q8_0 / Q4_0 / Q4_K.
// Residency stays active (the caller downloads / releases it).
func (b *Backend) LogitsFromDevice(t dtype.DType, normName string, norm []float32, headName string, headRaw []byte, headF32, logits []float32, vocab, embd int, eps float32) error {
	const op = "LogitsFromDevice"

	rt, err := b.fusedRuntime(op)
	if err != nil {
		return err
	}

	if vocab < 1 || embd < 1 {
		return fmt.Errorf("metal: %s: vocab=%d embd=%d", op, vocab, embd)
	}

	if len(norm) < embd || len(logits) < vocab {
		return fmt.Errorf("metal: %s: buffers too short", op)
	}

	if normName == "" || headName == "" {
		return fmt.Errorf("metal: %s: weight names are required (device weight cache key)", op)
	}

	switch t {
	case dtype.F32, dtype.F16:
		if len(headF32) < vocab*embd {
			return fmt.Errorf("metal: %s: FP32 head too short", op)
		}
	case dtype.Q8_0, dtype.Q4_0, dtype.Q4_K:
		if err := checkQuantMatrix(op, t, headRaw, vocab, embd); err != nil {
			return err
		}
	default:
		return unsupported(op, fmt.Sprintf("lm_head type %s", t))
	}

	b.mu.Lock()
	defer b.mu.Unlock()

	if !b.hidden {
		return fmt.Errorf("metal: %s: residency not active", op)
	}

	if b.hiddenLen != embd {
		return fmt.Errorf("metal: %s: resident hidden has %d floats, embd=%d", op, b.hiddenLen, embd)
	}

	return rt.logitsFromResident(&logitsArgs{
		t:        t,
		normName: normName,
		norm:     norm[:embd],
		headName: headName,
		headRaw:  headRaw,
		headF32:  headF32,
		logits:   logits[:vocab],
		vocab:    vocab,
		embd:     embd,
		eps:      eps,
	})
}
