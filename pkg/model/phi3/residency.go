package phi3

import (
	"github.com/magomedcoder/gogguf/pkg/device"
	"github.com/magomedcoder/gogguf/pkg/exec/gpuresid"
)

// initFused sets up fused GPU layer paths.
// Only the FFN is fused: layers with separate ffn_gate/ffn_up run one SwiGLU FFNNamed call on GPU (ffnGPU[layer]); merged ffn_up [embd, 2*ffn] stays on the host path
func (m *Model) initFused() {
	if m.dev == nil || m.ngl <= 0 {
		return
	}

	m.fused = gpuresid.New(m.weights, m.dev, device.RoPENeoX)
	m.ffnGPU = make([]bool, m.cfg.NumLayers)
	for i, lt := range m.layerTensors {
		m.ffnGPU[i] = !lt.fusedFFN && device.LayerOnGPU(i, m.ngl, m.cfg.NumLayers)
	}
}

// initResidency enables the Phi-3 P1 GPU path: out_norm + lm_head on GPU from the final hidden state.
//
// Full device-resident layers (m.fused.QKVAttn / AttnFFN) are deferred: they need standard NeoX RoPE over the whole head, but Phi-3 uses partial LongRoPE with short/long factors and attn scale (ApplyRoPEHeadsPartialScaled), attn_qkv is often one fused tensor, and ffn_up may be merged gate||up.
// So the layers stay on the host/per-op GPU path and only logits go through the device (needs ngl >= NumLayers and a hidden-resident backend)
func (m *Model) initResidency() {
	if m.fused == nil {
		return
	}

	if gpuresid.CanLogitsFromDevice(m.dev, m.ngl, m.cfg.NumLayers) && m.fused.HeadOnGPU(m.lmHeadName) {
		m.logitsOnGPU = true
	}
}

// logitsDevice computes RMSNorm(final hidden) + lm_head on GPU: one HtoD of the hidden state, DtoH of logits only
func (m *Model) logitsDevice(x []float32) error {
	return m.fused.LogitsFromHost(x, "output_norm.weight", m.outNorm, m.lmHeadName, m.scratch.logits, m.cfg.VocabSize, m.cfg.EmbeddingDim, m.cfg.RMSNormEps)
}

// logitsFromLast: lm_head on GPU when enabled, else on host
func (m *Model) logitsFromLast(x []float32) error {
	if m.logitsOnGPU && m.fused != nil {
		if err := m.logitsDevice(x); err == nil {
			return nil
		}

		// lm_head on GPU failed (weight type / VRAM): continue on host
		m.logitsOnGPU = false
	}

	return m.logitsFromHidden(x)
}
