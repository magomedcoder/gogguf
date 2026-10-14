package phi2

import (
	"github.com/magomedcoder/gogguf/pkg/device"
	"github.com/magomedcoder/gogguf/pkg/exec/gpuresid"
)

// initResidency enables the Phi-2 P1 GPU path: the lm_head matmul on GPU when ngl covers every layer and the backend is hidden-resident.
//
// Full device-resident layers are deferred: Phi-2 uses LayerNorm with bias, parallel attention+FFN residual, biased projections and a GELU FFN, none of which the fused RMSNorm/SwiGLU kernels cover.
//
// The final LayerNorm also rules out gpuresid.LogitsFromHost (device RMSNorm + lm_head): RMSNorm(x)*w cannot reproduce (x-mean)/std*w + b.
// So LayerNorm stays on the host and only the lm_head matmul runs on GPU with the normalized vector (same traffic: one HtoD of embd floats, one DtoH of logits); the lm_head bias is added on host
func (m *Model) initResidency() {
	if !gpuresid.CanLogitsFromDevice(m.dev, m.ngl, m.cfg.NumLayers) {
		return
	}

	m.fused = gpuresid.New(m.weights, m.dev, device.RoPENeoX)
	if m.fused.HeadOnGPU(m.lmHeadName) {
		m.logitsOnGPU = true
	}
}

// lmHeadGPU computes logits = lm_head * h on GPU; false means fall back to the host lm_head
func (m *Model) lmHeadGPU(h []float32) bool {
	if !m.logitsOnGPU || m.fused == nil {
		return false
	}

	if err := m.fused.MatMulInto(m.lmHeadName, m.cfg.VocabSize, m.cfg.EmbeddingDim, h, m.scratch.logits); err != nil {
		// weight type / VRAM: continue on host
		m.logitsOnGPU = false
		return false
	}

	return true
}
