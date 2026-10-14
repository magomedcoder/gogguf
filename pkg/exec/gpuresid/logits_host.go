package gpuresid

import (
	"fmt"

	"github.com/magomedcoder/gogguf/pkg/device"
	"github.com/magomedcoder/gogguf/pkg/dtype"
)

// CanLogitsFromDevice reports whether the final hidden state of a host-run forward pass may be uploaded for GPU out_norm + lm_head (one HtoD of embd floats, one DtoH of logits).
// Needs a hidden-resident backend (CUDA), residency not disabled, and ngl covering every layer.
// Layer quants and fused layer kernels are not required: layers can run on host
func CanLogitsFromDevice(g device.Backend, ngl, numLayers int) bool {
	if g == nil || !g.HiddenResident() || !HiddenResidencyEnabled() {
		return false
	}

	return AllLayersOffloaded(ngl, numLayers)
}

// HeadOnGPU reports whether the lm_head tensor can run on GPU without a large host FP32 expansion (native quant or FP32 only)
func (r *Runner) HeadOnGPU(headName string) bool {
	info, err := r.w.Info(headName)
	if err != nil {
		return false
	}

	return info.Type == dtype.F32 || device.FusedQuantSupported(info.Type)
}

// LogitsFromHost uploads hidden state x [embd] (host-computed, e.g. after the last layer), computes RMSNorm(x, norm) + lm_head on GPU and copies only logits to host.
// Residency is released afterwards (x is left untouched), so later host-side layers see a normal backend
func (r *Runner) LogitsFromHost(x []float32, normName string, norm []float32, headName string, logits []float32, vocab, embd int, eps float32) error {
	if len(x) < embd {
		return fmt.Errorf("gpuresid: LogitsFromHost: short hidden %d < %d", len(x), embd)
	}

	if err := r.g.HiddenUpload(x[:embd]); err != nil {
		return err
	}

	err := r.LogitsDevice(normName, norm, headName, logits, vocab, embd, eps)
	r.releaseHidden(embd)

	return err
}

// releaseHidden drops device residency after a logits-only pass (hidden is discarded)
func (r *Runner) releaseHidden(embd int) {
	if !r.g.HiddenActive() {
		return
	}

	if cap(r.tmp) < embd {
		r.tmp = make([]float32, embd)
	}

	_ = r.g.HiddenDownload(r.tmp[:embd])
}
