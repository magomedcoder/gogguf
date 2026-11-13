package metal

import (
	"errors"
	"fmt"

	"github.com/magomedcoder/gogguf/pkg/compute"
	"github.com/magomedcoder/gogguf/pkg/dtype"
	"github.com/magomedcoder/gogguf/pkg/quant"
)

// ErrUnavailable the Metal entry point cannot compute: no opened runtime (scaffold / non-darwin build) or the kernel is not written yet (every compute entry point wraps it, see unavailable)
var ErrUnavailable = errors.New("metal: unavailable (scaffold or kernel not implemented)")

// gpu is the compute runtime behind Backend: runtime_darwin.go (MSL kernels via cgo) is the only implementation.
// Backend validates arguments and dispatches here, so this file stays pure Go and builds everywhere.
// All methods synchronize (command buffer completed) before returning, results are already in the host slices.
type gpu interface {
	deviceName() string
	vram() (used, total uint64)
	// matMulF32 matrix[rows*cols] * vec[cols] -> dst[rows]; name == "" uploads matrix for this call only
	matMulF32(name string, matrix []float32, rows, cols int, vec, dst []float32) error
	// matMulQ8_0 ggml Q8_0 matrix (34-byte blocks) * vec -> dst; name == "" uploads raw for this call only
	matMulQ8_0(name string, raw []byte, rows, cols int, vec, dst []float32) error
	rmsNorm(dst, x, weight []float32, eps float32) error
	ropeHeads(v []float32, nHeads, headDim, pos int, freqBase float32) error
	swiglu(gate, up []float32) error
	// attention over host k/v [seqLen][nKVHeads*headDim]
	attention(dst, q, k, v []float32, seqLen, nHeads, nKVHeads, headDim int) error
	kvInit(layers, maxSeq, kvDim, nHeads, headDim int) error
	kvReset()
	kvAppend(layer, pos int, k, v []float32, n int) error
	kvDropOldest(layer, n int) error
	kvRebuild(layer int, k, v []float32, n int) error
	attentionKV(layer int, dst, q []float32, seqLen, nHeads, nKVHeads, headDim int) error
	close() error
}

// unavailable wraps ErrUnavailable with the entry point name and the planned kernel from PlannedKernels (errors.Is(err, ErrUnavailable) holds)
func unavailable(op string) error {
	if k := plannedKernelFor(op); k != "" {
		return fmt.Errorf("%w: %s needs MSL kernel %q (planned, see metal.PlannedKernels); use -ngl 0", ErrUnavailable, op, k)
	}

	return fmt.Errorf("%w: %s (planned, see metal.PlannedKernels); use -ngl 0", ErrUnavailable, op)
}

// Ready reports whether the backend can run model layers: MSL kernels are built in (KernelsImplemented) and a device runtime is open.
// This is what enables -ngl on darwin in pkg/model.Options.Normalize; a scaffold Backend (no -tags metal, or non-darwin) is never ready.
func (b *Backend) Ready() bool {
	return KernelsImplemented && b != nil && b.rt != nil
}

// Backend - Metal device: B1 decode ops (matvec F32/Q8_0, RMSNorm, RoPE NeoX, SwiGLU, attention, GPU KV) run on the runtime when open;
// everything else (and a Backend without runtime) returns ErrUnavailable so callers fall back to the CPU path
type Backend struct {
	name string
	rt   gpu
}

// Name returns device name, e.g. "Metal Apple M2 Pro" or "Metal (scaffold)"
func (b *Backend) Name() string {
	if b == nil {
		return "Metal (scaffold)"
	}

	if b.rt != nil {
		return b.rt.deviceName()
	}

	if b.name == "" {
		return "Metal (scaffold)"
	}

	return b.name
}

// VRAMInfo: Metal allocations / recommended working set (unified memory); zeros without runtime
func (b *Backend) VRAMInfo() (used, total uint64, err error) {
	if b == nil || b.rt == nil {
		return 0, 0, nil
	}

	used, total = b.rt.vram()

	return used, total, nil
}

// runtime returns the open gpu or ErrUnavailable for op
func (b *Backend) runtime(op string) (gpu, error) {
	if b == nil || b.rt == nil {
		return nil, unavailable(op)
	}

	return b.rt, nil
}

func (b *Backend) MatMulVec(matrix []float32, rows, cols int, vec []float32) ([]float32, error) {
	return b.matMulF32("MatMulVec", "", matrix, rows, cols, vec)
}

func (b *Backend) MatMulVecCached(name string, matrix []float32, rows, cols int, vec []float32) ([]float32, error) {
	return b.matMulF32("MatMulVecCached", name, matrix, rows, cols, vec)
}

func (b *Backend) matMulF32(op, name string, matrix []float32, rows, cols int, vec []float32) ([]float32, error) {
	rt, err := b.runtime(op)
	if err != nil {
		return nil, err
	}

	if rows < 1 || cols < 1 {
		return nil, fmt.Errorf("metal: %s: rows=%d cols=%d", op, rows, cols)
	}

	if len(vec) < cols {
		return nil, fmt.Errorf("metal: %s: len(vec)=%d, cols=%d", op, len(vec), cols)
	}

	if len(matrix) < rows*cols {
		return nil, fmt.Errorf("metal: %s: matrix too short", op)
	}

	dst := make([]float32, rows)
	if err := rt.matMulF32(name, matrix, rows, cols, vec[:cols], dst); err != nil {
		return nil, err
	}

	return dst, nil
}

func (b *Backend) MatMulVecQ8_0Cached(name string, raw []byte, rows, cols int, vec []float32) ([]float32, error) {
	rt, err := b.runtime("MatMulVecQ8_0Cached")
	if err != nil {
		return nil, err
	}

	if rows < 1 || cols < 1 {
		return nil, fmt.Errorf("metal: MatMulVecQ8_0Cached: rows=%d cols=%d", rows, cols)
	}

	if cols%quant.QK8_0 != 0 {
		return nil, fmt.Errorf("metal: MatMulVecQ8_0Cached: cols=%d not divisible by %d", cols, quant.QK8_0)
	}

	if len(vec) < cols {
		return nil, fmt.Errorf("metal: MatMulVecQ8_0Cached: len(vec)=%d, cols=%d", len(vec), cols)
	}

	if len(raw) < rows*(cols/quant.QK8_0)*quant.BlockQ8_0Size {
		return nil, fmt.Errorf("metal: MatMulVecQ8_0Cached: Q8_0 matrix too short")
	}

	dst := make([]float32, rows)
	if err := rt.matMulQ8_0(name, raw, rows, cols, vec[:cols], dst); err != nil {
		return nil, err
	}

	return dst, nil
}

func (b *Backend) MatMulVecQ8_1Cached(string, []byte, int, int, []float32) ([]float32, error) {
	return nil, unavailable("MatMulVecQ8_1Cached")
}

func (b *Backend) MatMulVecQ4_0Cached(string, []byte, int, int, []float32) ([]float32, error) {
	return nil, unavailable("MatMulVecQ4_0Cached")
}

func (b *Backend) MatMulVecQ4_KCached(string, []byte, int, int, []float32) ([]float32, error) {
	return nil, unavailable("MatMulVecQ4_KCached")
}

func (b *Backend) MatMulVecQ5_KCached(string, []byte, int, int, []float32) ([]float32, error) {
	return nil, unavailable("MatMulVecQ5_KCached")
}

func (b *Backend) MatMulVecQ6_KCached(string, []byte, int, int, []float32) ([]float32, error) {
	return nil, unavailable("MatMulVecQ6_KCached")
}

func (b *Backend) MatMulVecQ4_1Cached(string, []byte, int, int, []float32) ([]float32, error) {
	return nil, unavailable("MatMulVecQ4_1Cached")
}

func (b *Backend) MatMulVecQ5_0Cached(string, []byte, int, int, []float32) ([]float32, error) {
	return nil, unavailable("MatMulVecQ5_0Cached")
}

func (b *Backend) MatMulVecQ5_1Cached(string, []byte, int, int, []float32) ([]float32, error) {
	return nil, unavailable("MatMulVecQ5_1Cached")
}

func (b *Backend) MatMulVecQ2_KCached(string, []byte, int, int, []float32) ([]float32, error) {
	return nil, unavailable("MatMulVecQ2_KCached")
}

func (b *Backend) MatMulVecQ3_KCached(string, []byte, int, int, []float32) ([]float32, error) {
	return nil, unavailable("MatMulVecQ3_KCached")
}

func (b *Backend) MatMulVecQ8_KCached(string, []byte, int, int, []float32) ([]float32, error) {
	return nil, unavailable("MatMulVecQ8_KCached")
}

func (b *Backend) RMSNormInto(dst, x, weight []float32, eps float32) error {
	rt, err := b.runtime("RMSNormInto")
	if err != nil {
		return err
	}

	if len(dst) != len(x) || len(x) != len(weight) {
		return fmt.Errorf("metal: RMSNormInto: length mismatch")
	}

	if len(x) == 0 {
		return nil
	}

	return rt.rmsNorm(dst, x, weight, eps)
}

func (b *Backend) LayerNormBiasInto(_, _, _, _ []float32, _ float32) error {
	return unavailable("LayerNormBiasInto")
}

func (b *Backend) GELUInPlace([]float32) error {
	return unavailable("GELUInPlace")
}

// ApplyRoPEHeads applies NeoX RoPE in place (cos/sin tables come from the host float64 reference, the kernel only rotates pairs)
func (b *Backend) ApplyRoPEHeads(v []float32, nHeads, headDim, pos int, freqBase float32) error {
	rt, err := b.runtime("ApplyRoPEHeads")
	if err != nil {
		return err
	}

	if nHeads < 1 || headDim < 2 || headDim%2 != 0 {
		return fmt.Errorf("metal: ApplyRoPEHeads: nHeads=%d headDim=%d", nHeads, headDim)
	}

	if len(v) < nHeads*headDim {
		return fmt.Errorf("metal: ApplyRoPEHeads: len(v)=%d, need %d", len(v), nHeads*headDim)
	}

	return rt.ropeHeads(v[:nHeads*headDim], nHeads, headDim, pos, freqBase)
}

func (b *Backend) ApplyRoPEHeadsScaled([]float32, int, int, int, float32, compute.RoPEScale) error {
	return unavailable("ApplyRoPEHeadsScaled")
}

func (b *Backend) ApplyRoPEHeadsNorm([]float32, int, int, int, float32) error {
	return unavailable("ApplyRoPEHeadsNorm")
}

func (b *Backend) SwiGLUInPlace(gate, up []float32) error {
	rt, err := b.runtime("SwiGLUInPlace")
	if err != nil {
		return err
	}

	if len(up) < len(gate) {
		return fmt.Errorf("metal: SwiGLUInPlace: len(up)=%d < len(gate)=%d", len(up), len(gate))
	}

	if len(gate) == 0 {
		return nil
	}

	return rt.swiglu(gate, up[:len(gate)])
}

// SoftcapInPlace runs on host (logits already DtoH); no Metal kernel required
func (b *Backend) SoftcapInPlace(x []float32, softcap float32) error {
	compute.SoftcapInPlace(x, softcap)
	return nil
}

func (b *Backend) SetAttnSoftcap(float32) {}

func (b *Backend) FFNSwiGLUCached(_, _, _ string, _, _, _, _, _ []float32, _, _ int) error {
	return unavailable("FFNSwiGLUCached")
}

func (b *Backend) FFNSwiGLUQ8_0Cached(_, _, _ string, _, _, _ []byte, _, _ []float32, _, _ int) error {
	return unavailable("FFNSwiGLUQ8_0Cached")
}

func (b *Backend) AttnFFNResidualCached(_, _, _, _, _ string, _, _, _, _, _, _, _ []float32, _, _, _ int, _ float32) error {
	return unavailable("AttnFFNResidualCached")
}

func (b *Backend) AttnFFNResidualQ8_0Cached(_, _, _, _, _ string, _, _, _, _ []byte, _, _, _ []float32, _, _, _ int, _ float32) error {
	return unavailable("AttnFFNResidualQ8_0Cached")
}

func (b *Backend) QKVRoPEAttentionCached(_, _, _, _, _ string, _, _, _, _, _, _, _, _, _, _, _ []float32, _, _, _, _, _, _, _ int, _ float32) error {
	return unavailable("QKVRoPEAttentionCached")
}

func (b *Backend) QKVRoPEAttentionQ8_0Cached(_, _, _, _, _ string, _, _, _ []byte, _, _, _, _, _, _, _, _ []float32, _, _, _, _, _, _, _ int, _ float32) error {
	return unavailable("QKVRoPEAttentionQ8_0Cached")
}

func (b *Backend) FFNSwiGLUQuantCached(dtype.DType, string, string, string, []byte, []byte, []byte, []float32, []float32, int, int) error {
	return unavailable("FFNSwiGLUQuantCached")
}

func (b *Backend) FFNGeGLUQuantCached(dtype.DType, string, string, string, []byte, []byte, []byte, []float32, []float32, int, int) error {
	return unavailable("FFNGeGLUQuantCached")
}

func (b *Backend) AttnFFNResidualQuantCached(_ dtype.DType, _, _, _, _, _ string, _, _, _, _ []byte, _, _, _ []float32, _, _, _ int, _ float32) error {
	return unavailable("AttnFFNResidualQuantCached")
}

func (b *Backend) AttnFFNResidualGeGLUQuantCached(_ dtype.DType, _, _, _, _, _, _, _ string, _, _, _, _ []byte, _, _, _, _, _ []float32, _, _, _ int, _ float32) error {
	return unavailable("AttnFFNResidualGeGLUQuantCached")
}

func (b *Backend) QKVRoPEAttentionQuantCached(_ dtype.DType, _ compute.RoPEMode, _, _, _, _, _, _ string, _, _, _ []byte, _, _, _, _, _, _, _, _, _ []float32, _, _, _, _, _, _, _ int, _ float32) error {
	return unavailable("QKVRoPEAttentionQuantCached")
}

// HiddenResident device-resident hidden state will require Metal kernels
func (b *Backend) HiddenResident() bool {
	return false
}

func (b *Backend) HiddenActive() bool {
	return false
}

func (b *Backend) HiddenUpload([]float32) error {
	return unavailable("HiddenUpload")
}

func (b *Backend) HiddenDownload([]float32) error {
	return unavailable("HiddenDownload")
}

func (b *Backend) LogitsFromDevice(dtype.DType, string, []float32, string, []byte, []float32, []float32, int, int, float32) error {
	return unavailable("LogitsFromDevice")
}

func (b *Backend) LayerNormResidInto([]float32, []float32, []float32, string, string, float32) error {
	return unavailable("LayerNormResidInto")
}

func (b *Backend) ResidAdd([]float32) error {
	return unavailable("ResidAdd")
}

func (b *Backend) LogitsFromDeviceLayerNorm(dtype.DType, string, string, []float32, []float32, string, []byte, []float32, []float32, int, int, float32) error {
	return unavailable("LogitsFromDeviceLayerNorm")
}

// checkAttnDims validates GQA geometry shared by AttentionScoresInto / AttentionScoresKV
func checkAttnDims(op string, seqLen, nHeads, nKVHeads, headDim int) error {
	if seqLen < 1 || nHeads < 1 || nKVHeads < 1 || headDim < 1 {
		return fmt.Errorf("metal: %s: invalid dimensions seqLen=%d nHeads=%d nKVHeads=%d headDim=%d", op, seqLen, nHeads, nKVHeads, headDim)
	}

	if nHeads%nKVHeads != 0 {
		return fmt.Errorf("metal: %s: nHeads=%d not divisible by nKVHeads=%d", op, nHeads, nKVHeads)
	}

	return nil
}

// AttentionScoresInto scaled dot-product attention with GQA over host k/v (scores is unused: softmax weights live in a device scratch buffer)
func (b *Backend) AttentionScoresInto(dst, q, k, v, _ []float32, seqLen, nHeads, nKVHeads, headDim int) error {
	rt, err := b.runtime("AttentionScoresInto")
	if err != nil {
		return err
	}

	if err := checkAttnDims("AttentionScoresInto", seqLen, nHeads, nKVHeads, headDim); err != nil {
		return err
	}

	qDim, kvLen := nHeads*headDim, seqLen*nKVHeads*headDim
	if len(dst) < qDim || len(q) < qDim || len(k) < kvLen || len(v) < kvLen {
		return fmt.Errorf("metal: AttentionScoresInto: dst/q/k/v too short")
	}

	return rt.attention(dst[:qDim], q[:qDim], k[:kvLen], v[:kvLen], seqLen, nHeads, nKVHeads, headDim)
}

// KVCacheInit allocates per-layer K/V MTLBuffers (unified memory) for layers x maxSeq x kvDim
func (b *Backend) KVCacheInit(layers, maxSeq, kvDim, nHeads, headDim int) error {
	rt, err := b.runtime("KVCacheInit")
	if err != nil {
		return err
	}

	if layers < 1 || maxSeq < 1 || kvDim < 1 || nHeads < 1 || headDim < 1 {
		return fmt.Errorf("metal: KVCacheInit: layers=%d maxSeq=%d kvDim=%d nHeads=%d headDim=%d", layers, maxSeq, kvDim, nHeads, headDim)
	}

	return rt.kvInit(layers, maxSeq, kvDim, nHeads, headDim)
}

// KVCacheReset forgets cached positions (buffers stay allocated; positions are overwritten before they are read)
func (b *Backend) KVCacheReset() {
	if b == nil || b.rt == nil {
		return
	}

	b.rt.kvReset()
}

func (b *Backend) KVCacheAppend(layer, pos int, k, v []float32) error {
	rt, err := b.runtime("KVCacheAppend")
	if err != nil {
		return err
	}

	if len(k) == 0 || len(v) == 0 {
		return fmt.Errorf("metal: KVCacheAppend: empty k/v")
	}

	return rt.kvAppend(layer, pos, k, v, 1)
}

func (b *Backend) KVCacheAppendN(layer, pos int, k, v []float32, n int) error {
	rt, err := b.runtime("KVCacheAppendN")
	if err != nil {
		return err
	}

	if n <= 0 {
		return fmt.Errorf("metal: KVCacheAppendN: n=%d", n)
	}

	if len(k) == 0 || len(v) == 0 {
		return fmt.Errorf("metal: KVCacheAppendN: empty k/v")
	}

	if len(k)%n != 0 || len(v)%n != 0 {
		return fmt.Errorf("metal: KVCacheAppendN: len(k)=%d len(v)=%d not divisible by n=%d", len(k), len(v), n)
	}

	return rt.kvAppend(layer, pos, k, v, n)
}

// KVCacheDropOldest shifts device K/V left by n tokens (unified memory: memmove)
func (b *Backend) KVCacheDropOldest(layer, n int) error {
	rt, err := b.runtime("KVCacheDropOldest")
	if err != nil {
		return err
	}

	if n < 1 {
		return nil
	}

	return rt.kvDropOldest(layer, n)
}

// KVCacheRebuildFromHost replaces compact device K/V [0..n) from host buffers
func (b *Backend) KVCacheRebuildFromHost(layer int, k, v []float32, n int) error {
	rt, err := b.runtime("KVCacheRebuildFromHost")
	if err != nil {
		return err
	}

	if n < 1 {
		return fmt.Errorf("metal: KVCacheRebuildFromHost: n=%d", n)
	}

	return rt.kvRebuild(layer, k, v, n)
}

// AttentionScoresKV attention with K/V read from the device KV cache (positions [0, seqLen))
func (b *Backend) AttentionScoresKV(layer int, dst, q []float32, seqLen, nHeads, nKVHeads, headDim int) error {
	rt, err := b.runtime("AttentionScoresKV")
	if err != nil {
		return err
	}

	if err := checkAttnDims("AttentionScoresKV", seqLen, nHeads, nKVHeads, headDim); err != nil {
		return err
	}

	qDim := nHeads * headDim
	if len(dst) < qDim || len(q) < qDim {
		return fmt.Errorf("metal: AttentionScoresKV: dst/q too short")
	}

	return rt.attentionKV(layer, dst[:qDim], q[:qDim], seqLen, nHeads, nKVHeads, headDim)
}

// Close releases the device runtime (weight cache, KV buffers); safe on a scaffold Backend
func (b *Backend) Close() error {
	if b == nil || b.rt == nil {
		return nil
	}

	rt := b.rt
	b.rt = nil

	return rt.close()
}
