package device

import "github.com/magomedcoder/gogguf/pkg/dtype"

// MoEResidBackend is the optional n_batch=1 MoE path over device-resident residual.
// After QKV (attn in d_vec / skip_attn) or after a host attn residual upload: AttnOutResidualNorm or MoEHiddenNorm snapshots RMSNorm(resid) for router + experts; each expert FFN accumulates scale*out into d_resid without DtoH of embd.
type MoEResidBackend interface {
	// AttnOutResidualNormQuantCached: WO(attn)+resid+= + RMSNorm(resid)->MoE snapshot.
	// x/attn may be nil when resid_on_device / skip_attn_htod are set (after QKVAttn).
	AttnOutResidualNormQuantCached(t dtype.DType, woName, ffnNormName string, woRaw []byte, ffnNorm, x, attn []float32, embd, attnDim int, eps float32) error

	// MoEHiddenNorm: RMSNorm(d_resid)->MoE snapshot (attention residual already applied)
	MoEHiddenNorm(ffnNormName string, ffnNorm []float32, embd int, eps float32) error

	// MatMulVecFromMoEHiddenQuant: router (or any) matmul from MoE snapshot; DtoH out only
	MatMulVecFromMoEHiddenQuant(t dtype.DType, name string, raw []byte, rows, cols int, out []float32) error

	// MatMulVecFromMoEHiddenF32 same for FP32 weight cache
	MatMulVecFromMoEHiddenF32(name string, matrix []float32, rows, cols int, out []float32) error

	// FFNSwiGLUMoEAccumulateQuant: ExpertFFN from MoE snapshot; resid += scale * result
	FFNSwiGLUMoEAccumulateQuant(t dtype.DType, gateName, upName, downName string, gateRaw, upRaw, downRaw []byte, scale float32, embd, ffn int) error

	// ResidualAddScaled: resid += scale * src (host src); fallback helper
	ResidualAddScaled(src []float32, scale float32) error
}

// AsMoEResidBackend returns MoEResidBackend when the backend implements it
func AsMoEResidBackend(b Backend) (MoEResidBackend, bool) {
	m, ok := b.(MoEResidBackend)
	return m, ok
}
