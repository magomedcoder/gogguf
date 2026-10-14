package gemma

import (
	"fmt"

	"github.com/magomedcoder/gogguf/pkg/compute"
	"github.com/magomedcoder/gogguf/pkg/device"
	"github.com/magomedcoder/gogguf/pkg/exec/gpuresid"
)

// initFused sets up fused GPU layer paths: QKV+RoPE+attention (NeoX), AttnOut, norms and the GeGLU FFN.
// Filled for both variants: Gemma2 still uses the fused GeGLU FFN per layer, even though it never enables the fused QKV / residency path (post-norm, attention softcap, SWA)
func (m *Model) initFused() {
	if m.dev == nil {
		return
	}

	m.fused = gpuresid.New(m.weights, m.dev, device.RoPENeoX)
	m.fusedDims = gpuresid.Dims{
		Embd:     m.cfg.EmbeddingDim,
		NHeads:   m.cfg.NumHeads,
		NKVHeads: m.cfg.NumKVHeads,
		HeadDim:  m.cfg.HeadDim,
		FFN:      m.cfg.FFNHidden,
		Eps:      m.cfg.RMSNormEps,
		RopeBase: m.cfg.RopeFreqBase,
	}

	m.gpuLayers = make([]gpuresid.Tensors, m.cfg.NumLayers)
	m.layerQuant = make([]bool, m.cfg.NumLayers)
	for i, lt := range m.layerTensors {
		p := fmt.Sprintf("blk.%d.", i)
		m.gpuLayers[i] = gpuresid.Tensors{
			AttnQ:    lt.attnQ,
			AttnK:    lt.attnK,
			AttnV:    lt.attnV,
			AttnOut:  lt.attnOut,
			FFNGate:  lt.ffnGate,
			FFNUp:    lt.ffnUp,
			FFNDown:  lt.ffnDown,
			AttnNorm: p + "attn_norm.weight",
			FFNNorm:  p + "ffn_norm.weight",
		}

		t := gpuresid.LayerQuant(m.weights, lt.attnQ, lt.attnK, lt.attnV, lt.attnOut, lt.ffnGate, lt.ffnUp, lt.ffnDown)
		m.layerQuant[i] = device.FusedQuantSupported(t)
	}
}

// initResidency enables the Gemma P1 GPU paths.
//
// Fused QKV+RoPE+attention (m.residency): VariantGemma only, with every layer offloaded, one supported weight
// quant in every layer, no attention softcap and no SWA. Gemma2 never enables it (post-norms, softcap, SWA).
//
// Hidden state is NOT kept on the device across layers: the fused AttnFFN kernel (WO + residual + ffn_norm +
// FFN + residual) is SwiGLU-only, and Gemma needs GeGLU. So the layer is QKVAttn (fused, host h) + AttnOut +
// fused FFNGeGLU (see ffnInto); the full device-resident layer (forwardBlockDevice) is deferred until a
// GeGLU AttnFFN kernel exists.
//
// Logits on GPU (m.logitsOnGPU): both variants when ngl covers all layers and the backend is hidden-resident
func (m *Model) initResidency() {
	if m.fused == nil {
		return
	}

	if gpuresid.CanLogitsFromDevice(m.dev, m.ngl, m.cfg.NumLayers) && m.fused.HeadOnGPU(m.lmHeadName) {
		m.logitsOnGPU = true
	}

	if m.cfg.Variant != VariantGemma || m.cfg.AttnLogitSoftcap != 0 || m.cfg.SlidingWindow != 0 {
		return
	}

	if !gpuresid.AllLayersOffloaded(m.ngl, m.cfg.NumLayers) || !gpuresid.CanEnableResidency(m.dev, m.layerQuant) {
		return
	}

	m.residency = true
}

// qkvAttnGPU: QKV + RoPE + KV append + attention on GPU. h is the host attn_norm output.
// k/v are copied back for the CPU KV-cache mirror (fallback paths)
func (m *Model) qkvAttnGPU(layer, pos int, ln layerNorms, h, kOut, vOut, attn []float32) error {
	kvPos := m.cache.Len()

	return m.fused.QKVAttn(m.gpuLayers[layer], m.fusedDims, nil, nil, ln.attnNorm, h, attn, kOut, vOut, layer, pos, kvPos, kvPos+1)
}

// logitsDevice computes RMSNorm(final hidden) + lm_head on GPU: one HtoD of the hidden state, DtoH of logits only
func (m *Model) logitsDevice(x []float32) error {
	if err := m.fused.LogitsFromHost(x, "output_norm.weight", m.outNorm, m.lmHeadName, m.scratch.logits, m.cfg.VocabSize, m.cfg.EmbeddingDim, m.cfg.RMSNormEps); err != nil {
		return err
	}

	compute.SoftcapInPlace(m.scratch.logits, m.cfg.FinalLogitSoftcap)

	return nil
}

// recoverLogits disables GPU logits after a failure so the caller falls back to the host lm_head.
// Layers never leave hidden state on the device (see initResidency), so there is nothing to download
func (m *Model) recoverLogits() {
	m.logitsOnGPU = false
}
