package mistral

import (
	"fmt"

	"github.com/magomedcoder/gogguf/pkg/gpu"
	"github.com/magomedcoder/gogguf/pkg/model/gpuresid"
)

// initFused sets up fused GPU layer paths (§5). Mistral / Qwen2: NeoX RoPE, no QK-norm
func (m *Model) initFused() {
	if m.gpu == nil {
		return
	}

	m.fused = gpuresid.New(m.weights, m.gpu, gpu.RoPENeoX)
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
	m.moeGPU = m.cfg.isMoE()
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

		names := []string{lt.attnQ, lt.attnK, lt.attnV, lt.attnOut}
		if !lt.moe {
			names = append(names, lt.ffnGate, lt.ffnUp, lt.ffnDown)
		}

		m.layerQuant[i] = gpu.FusedQuantSupported(gpuresid.LayerQuant(m.weights, names...))

		// §6: experts on GPU only with natively supported quant
		if lt.moe && !m.fused.MoESupported(lt.gateExps, lt.upExps, lt.downExps) {
			m.moeGPU = false
		}
	}
}

// initResidency enables device-resident hidden state (§1, §5).
// MoE layers run on host (experts), so residency is only for dense models
func (m *Model) initResidency() {
	if m.gpu == nil || m.ngl < m.cfg.NumLayers || m.fused == nil || m.cfg.isMoE() {
		return
	}

	// SWA: fused QKV attends over the full GPU KV-cache; it does not apply the window
	if m.cfg.SlidingWindow > 0 && m.cfg.SlidingWindow < m.cfg.ContextLength {
		return
	}

	if !gpuresid.HiddenResidencyEnabled() || !gpuresid.QKVResidencyEnabled() || !gpuresid.AttnFFNResidencyEnabled() {
		return
	}

	if !m.gpu.HiddenResident() {
		return
	}

	for _, ok := range m.layerQuant {
		if !ok {
			return
		}
	}

	m.residency = true
	m.logitsOnGPU = true
}

// gpuAttnFull: full GPU KV-cache attention is correct only when the SWA window covers the full sequence
func (m *Model) gpuAttnFull(seqLen int) bool {
	w := m.cfg.SlidingWindow

	return w <= 0 || seqLen <= w
}

// uploadHidden uploads hidden state before layers (one HtoD per token)
func (m *Model) uploadHidden() {
	if !m.residency || m.debug != nil || m.gpuKVStale || !m.gpuAttnFull(m.cache.Len()+1) {
		return
	}

	if err := m.gpu.HiddenUpload(m.scratch.x); err != nil {
		m.residency = false
		return
	}

	m.residDevice = true
}

// forwardBlockDevice runs the layer on GPU over resident hidden state
func (m *Model) forwardBlockDevice(layer, pos int, ln layerNorms, k, v, attn []float32) error {
	if err := m.qkvAttnGPU(layer, pos, ln, nil, k, v, attn); err != nil {
		return err
	}
	m.cache.Append(layer, k, v)

	return m.attnFFNGPU(layer, ln, nil, attn)
}

// qkvAttnGPU: QKV + RoPE + KV append + attention on GPU.
// h == nil - hidden resident: attn_norm computed on device from residual
func (m *Model) qkvAttnGPU(layer, pos int, ln layerNorms, h, kOut, vOut, attn []float32) error {
	kvPos := m.cache.Len()

	return m.fused.QKVAttn(m.gpuLayers[layer], m.fusedDims, nil, nil, ln.attnNorm, h, attn, kOut, vOut, layer, pos, kvPos, kvPos+1)
}

// attnFFNGPU: WO + residual + ffn_norm + FFN + residual on GPU
func (m *Model) attnFFNGPU(layer int, ln layerNorms, x, attn []float32) error {
	return m.fused.AttnFFN(m.gpuLayers[layer], m.fusedDims, ln.ffnNorm, x, attn)
}

// recoverHiddenFromDevice brings hidden state back to host after a device-layer failure
func (m *Model) recoverHiddenFromDevice(layer int, cause error) error {
	m.residDevice = false
	m.residency = false

	if !m.gpu.HiddenActive() {
		return fmt.Errorf("mistral: layer %d on GPU: %w (hidden state lost)", layer, cause)
	}

	if err := m.gpu.HiddenDownload(m.scratch.x); err != nil {
		return fmt.Errorf("mistral: layer %d on GPU: %w (fallback to host: %v)", layer, cause, err)
	}

	return nil
}

// syncHiddenFromDevice pulls hidden state from device when resident there
func (m *Model) syncHiddenFromDevice() error {
	if !m.residDevice {
		return nil
	}

	m.residDevice = false

	return m.gpu.HiddenDownload(m.scratch.x)
}

// logitsDevice computes RMSNorm(resident hidden) + lm_head on GPU without DtoH hidden (§2)
func (m *Model) logitsDevice() error {
	return m.fused.LogitsDevice("output_norm.weight", m.outNorm, m.lmHeadName, m.scratch.logits, m.cfg.VocabSize, m.cfg.EmbeddingDim, m.cfg.RMSNormEps)
}
