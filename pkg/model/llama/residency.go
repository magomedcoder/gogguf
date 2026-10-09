package llama

import (
	"fmt"

	"github.com/magomedcoder/gogguf/pkg/gpu"
	"github.com/magomedcoder/gogguf/pkg/model/gpuresid"
)

// initFused sets up fused GPU layer paths (§5). Llama: RoPE NORM and no QK-norm
func (m *Model) initFused() {
	if m.gpu == nil {
		return
	}

	m.fused = gpuresid.New(m.weights, m.gpu, gpu.RoPENorm)
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
		m.layerQuant[i] = gpu.FusedQuantSupported(t)
	}
}

// initResidency enables device-resident hidden state (§1, §5): all layers on GPU, fused paths, and one supported weight quant in every layer
func (m *Model) initResidency() {
	if m.gpu == nil || m.ngl < m.cfg.NumLayers || m.fused == nil {
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

// uploadHidden uploads hidden state before layers (one HtoD per token).
// Debug hooks read host buffer x, so residency is not enabled with them
func (m *Model) uploadHidden() {
	if !m.residency || m.debug != nil || m.gpuKVStale {
		return
	}

	if err := m.gpu.HiddenUpload(m.scratch.x); err != nil {
		m.residency = false
		return
	}

	m.residDevice = true
}

// forwardBlockDevice runs the layer on GPU over resident hidden state.
// k/v copied to host only for CPU KV-cache mirror (needed for fallback paths)
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

// attnFFNGPU: WO + residual + ffn_norm + FFN + residual on GPU.
// x == nil - hidden resident on device
func (m *Model) attnFFNGPU(layer int, ln layerNorms, x, attn []float32) error {
	return m.fused.AttnFFN(m.gpuLayers[layer], m.fusedDims, ln.ffnNorm, x, attn)
}

// recoverHiddenFromDevice brings hidden state back to host after a device-layer failure
func (m *Model) recoverHiddenFromDevice(layer int, cause error) error {
	m.residDevice = false
	m.residency = false

	if !m.gpu.HiddenActive() {
		return fmt.Errorf("llama: слой %d на GPU: %w (hidden state потерян)", layer, cause)
	}

	if err := m.gpu.HiddenDownload(m.scratch.x); err != nil {
		return fmt.Errorf("llama: слой %d на GPU: %w (откат на host: %v)", layer, cause, err)
	}

	return nil
}

// syncHiddenFromDevice pulls hidden state from device when resident (for host readers)
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
