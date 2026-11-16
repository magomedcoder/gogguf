package metal

import (
	"errors"
	"fmt"
	"sync"

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
	// matMulQuant same for the other native matvec quants (Q4_0 / Q4_K / Q5_K / Q6_K, pkg/quant layouts); t selects the kernel
	matMulQuant(t dtype.DType, name string, raw []byte, rows, cols int, vec, dst []float32) error
	rmsNorm(dst, x, weight []float32, eps float32) error
	ropeHeads(v []float32, nHeads, headDim, pos int, freqBase float32) error
	// ropeHeadsNorm same for the Llama NORM layout (adjacent pairs 2i, 2i+1)
	ropeHeadsNorm(v []float32, nHeads, headDim, pos int, freqBase float32) error
	swiglu(gate, up []float32) error
	// attention over host k/v [seqLen][nKVHeads*headDim]
	attention(dst, q, k, v []float32, seqLen, nHeads, nKVHeads, headDim int) error
	kvInit(layers, maxSeq, kvDim, nHeads, headDim int) error
	kvReset()
	kvAppend(layer, pos int, k, v []float32, n int) error
	kvDropOldest(layer, n int) error
	kvRebuild(layer int, k, v []float32, n int) error
	attentionKV(layer int, dst, q []float32, seqLen, nHeads, nKVHeads, headDim int) error

	// B2 residency / fused decode (fused.go). The residual lives in a device buffer; Backend owns the "active" flag and the contracts (h == nil, x == nil), the runtime only runs the kernels.
	// hiddenReady reports that the runtime has every kernel residency needs (rmsnorm + add + swiglu + attention + the quant matvecs + both rope layouts)
	hiddenReady() bool
	hiddenUpload(x []float32) error
	hiddenDownload(dst []float32) error
	// qkvRoPEAttention: resident == true reads RMSNorm(resid, attn_norm) instead of a.h; the attention output stays on the device for attnFFNResidual
	qkvRoPEAttention(a *qkvArgs, resident bool) error
	// attnFFNResidual: resident == true works on resid in place; otherwise a.x is the residual (staged in resid, written back on success).
	// dirty reports that resid may have been modified when err != nil (the GPU work was committed and failed)
	attnFFNResidual(a *attnFFNArgs, resident bool) (dirty bool, err error)
	ffnSwiGLU(a *ffnArgs) error
	logitsFromResident(a *logitsArgs) error

	// B3 prefill / batch (batch.go): x is [batch][cols], out [batch][rows], token b at offset b*cols / b*rows. All GPU work of one call is a single command buffer.
	// matMulMatQ8_0: W*X[batch] in ONE matmul_mat_q8_0 launch (batch >= 2; batch == 1 goes through matMulQ8_0). name == "" uploads raw for this call only
	matMulMatQ8_0(name string, raw []byte, rows, cols int, x []float32, batch int, out []float32) error
	// ffnSwiGLUMatQ8_0: gate/up GEMM -> SwiGLU -> down GEMM with the activations on the device; x and out may alias
	ffnSwiGLUMatQ8_0(a *ffnMatArgs) error
	// attentionKVBatch causal multi-query attention over the device KV cache: query i (of n) sees pastLen+i+1 positions
	attentionKVBatch(layer int, dst, q []float32, n, pastLen, nHeads, nKVHeads, headDim int) error
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
// Ready only gates offload (B1 per-op path); the B2 device-resident fused path is advertised separately by HiddenResident (fused.go), which pkg/exec/gpuresid checks on top of it.
func (b *Backend) Ready() bool {
	return KernelsImplemented && b != nil && b.rt != nil
}

// Backend - Metal device: B1 decode ops (matvec F32/Q8_0/Q4_0/Q4_K/Q5_K/Q6_K, RMSNorm, RoPE NeoX + NORM, SwiGLU, attention, GPU KV) and the B2 fused decode path for Q8_0 / Q4_0 / Q4_K / Q5_K / Q6_K weights with NeoX or NORM RoPE (device-resident hidden state, fused QKV / attn+FFN / logits) run on the runtime when open;
// everything else (and a Backend without runtime) returns ErrUnavailable so callers fall back to the B1 per-op path or the CPU
type Backend struct {
	name string
	rt   gpu

	// device-resident hidden state (fused.go); guarded by mu, held across each fused call like the CUDA backend does
	mu        sync.Mutex
	hidden    bool // resid holds the current hidden state
	hiddenLen int  // floats in resid while hidden
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

// VRAMInfo Metal allocations / recommended working set (unified memory); zeros without runtime
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

// quantGeometry returns the block width (values) and block size (bytes) of a Metal matvec quant (pkg/quant layouts)
func quantGeometry(t dtype.DType) (qk, blockBytes int, ok bool) {
	switch t {
	case dtype.Q8_0:
		return quant.QK8_0, quant.BlockQ8_0Size, true
	case dtype.Q4_0:
		return quant.QK4_0, quant.BlockQ4_0Size, true
	case dtype.Q4_K:
		return quant.QK_K, quant.BlockQ4_KSize, true
	case dtype.Q5_K:
		return quant.QK_K, quant.BlockQ5_KSize, true
	case dtype.Q6_K:
		return quant.QK_K, quant.BlockQ6_KSize, true
	default:
		return 0, 0, false
	}
}

// checkQuantMatrix validates a rows x cols matrix of type t held in raw
func checkQuantMatrix(op string, t dtype.DType, raw []byte, rows, cols int) error {
	qk, bs, ok := quantGeometry(t)
	if !ok {
		return unavailable(op)
	}

	if rows < 1 || cols < 1 {
		return fmt.Errorf("metal: %s: rows=%d cols=%d", op, rows, cols)
	}

	if cols%qk != 0 {
		return fmt.Errorf("metal: %s: cols=%d not divisible by %d", op, cols, qk)
	}

	if len(raw) < rows*(cols/qk)*bs {
		return fmt.Errorf("metal: %s: %s matrix too short", op, t)
	}

	return nil
}

// matMulQuantCached is the shared Q4_0 / Q4_K / Q5_K / Q6_K matvec entry (Q8_0 keeps its own wrapper above)
func (b *Backend) matMulQuantCached(op string, t dtype.DType, name string, raw []byte, rows, cols int, vec []float32) ([]float32, error) {
	rt, err := b.runtime(op)
	if err != nil {
		return nil, err
	}

	if err := checkQuantMatrix(op, t, raw, rows, cols); err != nil {
		return nil, err
	}

	if len(vec) < cols {
		return nil, fmt.Errorf("metal: %s: len(vec)=%d, cols=%d", op, len(vec), cols)
	}

	dst := make([]float32, rows)
	if err := rt.matMulQuant(t, name, raw, rows, cols, vec[:cols], dst); err != nil {
		return nil, err
	}

	return dst, nil
}

func (b *Backend) MatMulVecQ8_1Cached(string, []byte, int, int, []float32) ([]float32, error) {
	return nil, unavailable("MatMulVecQ8_1Cached")
}

func (b *Backend) MatMulVecQ4_0Cached(name string, raw []byte, rows, cols int, vec []float32) ([]float32, error) {
	return b.matMulQuantCached("MatMulVecQ4_0Cached", dtype.Q4_0, name, raw, rows, cols, vec)
}

func (b *Backend) MatMulVecQ4_KCached(name string, raw []byte, rows, cols int, vec []float32) ([]float32, error) {
	return b.matMulQuantCached("MatMulVecQ4_KCached", dtype.Q4_K, name, raw, rows, cols, vec)
}

func (b *Backend) MatMulVecQ5_KCached(name string, raw []byte, rows, cols int, vec []float32) ([]float32, error) {
	return b.matMulQuantCached("MatMulVecQ5_KCached", dtype.Q5_K, name, raw, rows, cols, vec)
}

func (b *Backend) MatMulVecQ6_KCached(name string, raw []byte, rows, cols int, vec []float32) ([]float32, error) {
	return b.matMulQuantCached("MatMulVecQ6_KCached", dtype.Q6_K, name, raw, rows, cols, vec)
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

// ApplyRoPEHeadsNorm applies Llama NORM RoPE (adjacent pairs 2i, 2i+1) in place; same host cos/sin table as NeoX, the rope_heads_norm kernel only rotates pairs
func (b *Backend) ApplyRoPEHeadsNorm(v []float32, nHeads, headDim, pos int, freqBase float32) error {
	rt, err := b.runtime("ApplyRoPEHeadsNorm")
	if err != nil {
		return err
	}

	if nHeads < 1 || headDim < 2 || headDim%2 != 0 {
		return fmt.Errorf("metal: ApplyRoPEHeadsNorm: nHeads=%d headDim=%d", nHeads, headDim)
	}

	if len(v) < nHeads*headDim {
		return fmt.Errorf("metal: ApplyRoPEHeadsNorm: len(v)=%d, need %d", len(v), nHeads*headDim)
	}

	return rt.ropeHeadsNorm(v[:nHeads*headDim], nHeads, headDim, pos, freqBase)
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

func (b *Backend) FFNGeGLUQuantCached(dtype.DType, string, string, string, []byte, []byte, []byte, []float32, []float32, int, int) error {
	return unavailable("FFNGeGLUQuantCached")
}

func (b *Backend) AttnFFNResidualGeGLUQuantCached(_ dtype.DType, _, _, _, _, _, _, _ string, _, _, _, _ []byte, _, _, _, _, _ []float32, _, _, _ int, _ float32) error {
	return unavailable("AttnFFNResidualGeGLUQuantCached")
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

	b.mu.Lock()
	b.hidden, b.hiddenLen = false, 0
	b.mu.Unlock()

	rt := b.rt
	b.rt = nil

	return rt.close()
}
