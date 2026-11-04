package gpuresid

import (
	"fmt"

	"github.com/magomedcoder/gogguf/pkg/device"
)

// AttnOutNorm: WO(attn)+resid+= + RMSNorm(resid)->MoE snapshot on device.
// x/attn nil after QKVAttn residency (skip_attn + d_resid).
func (r *Runner) AttnOutNorm(t Tensors, d Dims, ffnNorm, x, attn []float32) error {
	mb, ok := device.AsMoEResidBackend(r.g)
	if !ok {
		return fmt.Errorf("gpuresid: AttnOutNorm: backend lacks MoE resid path")
	}

	info, err := r.w.Info(t.AttnOut)
	if err != nil {
		return err
	}

	if !device.FusedQuantSupported(info.Type) {
		return fmt.Errorf("gpuresid: AttnOutNorm: type %s unsupported", info.Type)
	}

	woRaw, err := r.w.Raw(t.AttnOut)
	if err != nil {
		return err
	}

	return mb.AttnOutResidualNormQuantCached(info.Type, t.AttnOut, t.FFNNorm, woRaw, ffnNorm, x, attn, d.Embd, d.QDim(), d.Eps)
}

// MoEHiddenNorm: RMSNorm(d_resid)->MoE snapshot (attention residual already on device)
func (r *Runner) MoEHiddenNorm(ffnNormName string, ffnNorm []float32, embd int, eps float32) error {
	mb, ok := device.AsMoEResidBackend(r.g)
	if !ok {
		return fmt.Errorf("gpuresid: MoEHiddenNorm: backend lacks MoE resid path")
	}

	return mb.MoEHiddenNorm(ffnNormName, ffnNorm, embd, eps)
}

// MatMulFromMoEHidden: matmul reading MoE snapshot; DtoH out (router logits)
func (r *Runner) MatMulFromMoEHidden(name string, rows, cols int, out []float32) error {
	if len(out) < rows {
		return fmt.Errorf("gpuresid: %s: short out: %d < %d", name, len(out), rows)
	}

	mb, ok := device.AsMoEResidBackend(r.g)
	if !ok {
		return fmt.Errorf("gpuresid: MatMulFromMoEHidden: backend lacks MoE resid path")
	}

	info, err := r.w.Info(name)
	if err != nil {
		return err
	}

	if device.FusedQuantSupported(info.Type) {
		raw, err := r.w.Raw(name)
		if err != nil {
			return err
		}

		return mb.MatMulVecFromMoEHiddenQuant(info.Type, name, raw, rows, cols, out)
	}

	f32, err := r.w.Floats(name)
	if err != nil {
		return err
	}

	return mb.MatMulVecFromMoEHiddenF32(name, f32, rows, cols, out)
}

// ExpertFFNAccumulate: one expert FFN from MoE snapshot; resid += scale * result
func (r *Runner) ExpertFFNAccumulate(gateExps, upExps, downExps string, expert int, scale float32, embd, ffn int) error {
	mb, ok := device.AsMoEResidBackend(r.g)
	if !ok {
		return fmt.Errorf("gpuresid: ExpertFFNAccumulate: backend lacks MoE resid path")
	}

	gate, err := r.expertSlice(gateExps, expert, ffn, embd)
	if err != nil {
		return err
	}

	up, err := r.expertSlice(upExps, expert, ffn, embd)
	if err != nil {
		return err
	}

	down, err := r.expertSlice(downExps, expert, embd, ffn)
	if err != nil {
		return err
	}

	if gate.t != up.t || gate.t != down.t {
		return fmt.Errorf("gpuresid: experts of different types: %s/%s/%s", gate.t, up.t, down.t)
	}

	return mb.FFNSwiGLUMoEAccumulateQuant(gate.t, gate.name, up.name, down.name, gate.raw, up.raw, down.raw, scale, embd, ffn)
}

// FFNNamedAccumulate: named SwiGLU FFN from MoE snapshot; resid += scale * result (shared expert)
func (r *Runner) FFNNamedAccumulate(gateName, upName, downName string, scale float32, embd, ffn int) error {
	mb, ok := device.AsMoEResidBackend(r.g)
	if !ok {
		return fmt.Errorf("gpuresid: FFNNamedAccumulate: backend lacks MoE resid path")
	}

	info, err := r.w.Info(gateName)
	if err != nil {
		return err
	}

	if !device.FusedQuantSupported(info.Type) {
		return fmt.Errorf("gpuresid: FFNNamedAccumulate: type %s unsupported", info.Type)
	}

	gateRaw, upRaw, downRaw, err := r.raw3(gateName, upName, downName)
	if err != nil {
		return err
	}

	return mb.FFNSwiGLUMoEAccumulateQuant(info.Type, gateName, upName, downName, gateRaw, upRaw, downRaw, scale, embd, ffn)
}

// ExpertFFN: FFN of one MoE expert on GPU.
// x==nil reads the MoE snapshot (after AttnOutNorm / MoEHiddenNorm); result still DtoH into out.
// Prefer ExpertFFNAccumulate for resid += scale * expert without embd DtoH.
func (r *Runner) ExpertFFN(gateExps, upExps, downExps string, expert int, x, out []float32, embd, ffn int) error {
	gate, err := r.expertSlice(gateExps, expert, ffn, embd)
	if err != nil {
		return err
	}

	up, err := r.expertSlice(upExps, expert, ffn, embd)
	if err != nil {
		return err
	}

	down, err := r.expertSlice(downExps, expert, embd, ffn)
	if err != nil {
		return err
	}

	if gate.t != up.t || gate.t != down.t {
		return fmt.Errorf("gpuresid: experts of different types: %s/%s/%s", gate.t, up.t, down.t)
	}

	return r.g.FFNSwiGLUQuantCached(gate.t, gate.name, up.name, down.name, gate.raw, up.raw, down.raw, x, out, embd, ffn)
}

// ResidualAddScaled: resid += scale * src on device (host src)
func (r *Runner) ResidualAddScaled(src []float32, scale float32) error {
	mb, ok := device.AsMoEResidBackend(r.g)
	if !ok {
		return fmt.Errorf("gpuresid: ResidualAddScaled: backend lacks MoE resid path")
	}

	return mb.ResidualAddScaled(src, scale)
}

// MoEResidSupported reports whether the backend can run the device-resident MoE path
func MoEResidSupported(g device.Backend) bool {
	_, ok := device.AsMoEResidBackend(g)
	return ok
}
