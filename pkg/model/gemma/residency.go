package gemma

import (
	"fmt"

	"github.com/magomedcoder/gogguf/pkg/device"
	"github.com/magomedcoder/gogguf/pkg/exec/gpuresid"
)

// initFused sets up fused GPU layer paths: QKV+RoPE+attention (NeoX), AttnOut, norms and the GeGLU FFN.
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
		t := gpuresid.Tensors{
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
		if m.cfg.Variant == VariantGemma2 {
			t.PostAttnNorm = p + "post_attention_norm.weight"
			t.PostFFNNorm = p + "post_ffw_norm.weight"
		}
		m.gpuLayers[i] = t

		qt := gpuresid.LayerQuant(m.weights, lt.attnQ, lt.attnK, lt.attnV, lt.attnOut, lt.ffnGate, lt.ffnUp, lt.ffnDown)
		m.layerQuant[i] = device.FusedQuantSupported(qt)
	}
}

// initResidency enables device-resident hidden state when fused paths and quants allow it.
// Softcap is applied on-device via SetAttnSoftcap; Gemma2 post-norms via AttnFFNGeGLUPostNorm.
// SWA reclaim sets GPUKVStale (same as host path) and falls back out of residency.
func (m *Model) initResidency() {
	if m.fused == nil {
		return
	}

	if gpuresid.CanLogitsFromDevice(m.dev, m.ngl, m.cfg.NumLayers) && m.fused.HeadOnGPU(m.lmHeadName) {
		m.decode.LogitsOnGPU = true
	}

	if !gpuresid.AllLayersOffloaded(m.ngl, m.cfg.NumLayers) || !gpuresid.CanEnableResidency(m.dev, m.layerQuant) {
		return
	}

	m.dev.SetAttnSoftcap(m.cfg.AttnLogitSoftcap)
	m.residency = true
	m.decode.LogitsOnGPU = true
}

// uploadHidden uploads hidden state before layers (one HtoD per token)
func (m *Model) uploadHidden() {
	if !m.residency || m.decode.GPUKVStale {
		return
	}

	if err := m.dev.HiddenUpload(m.scratch.x); err != nil {
		m.residency = false
		return
	}

	m.decode.ResidDevice = true
}

// forwardBlockDevice runs the layer on GPU over resident hidden state
func (m *Model) forwardBlockDevice(layer, pos int, ln layerNorms) error {
	m.prepareSWAAppend(layer, 1)
	if m.decode.GPUKVStale {
		return fmt.Errorf("gemma: GPU KV stale after SWA reclaim")
	}

	kvPos := m.cache.Len()
	seqLen := kvPos + 1
	if m.cfg.IsSWALayer(layer) && m.cfg.SlidingWindow > 0 {
		have := m.cache.LayerLen(layer)
		seqLen = have + 1
		if seqLen > m.cfg.SlidingWindow {
			seqLen = m.cfg.SlidingWindow
		}
	}

	if err := m.qkvAttnGPU(layer, pos, ln, nil, nil, nil, nil, seqLen); err != nil {
		return err
	}

	if err := m.cache.Bump(layer, 1); err != nil {
		return err
	}
	m.decode.CPUKVHollow = true

	return m.attnFFNGPU(layer, ln, nil, nil)
}

// qkvAttnGPU: QKV + RoPE + KV append + attention on GPU.
// h == nil - hidden resident: attn_norm computed on device from residual.
func (m *Model) qkvAttnGPU(layer, pos int, ln layerNorms, h, kOut, vOut, attn []float32, seqLen int) error {
	kvPos := m.cache.Len()
	d := m.fusedDims
	d.RopeBase = m.cfg.ropeBase(layer)
	if seqLen <= 0 {
		seqLen = kvPos + 1
	}

	return m.fused.QKVAttn(m.gpuLayers[layer], d, nil, nil, ln.attnNorm, h, attn, kOut, vOut, layer, pos, kvPos, seqLen)
}

// attnFFNGPU: WO + residual + ffn_norm + GeGLU FFN + residual on GPU (+ Gemma2 post-norms)
func (m *Model) attnFFNGPU(layer int, ln layerNorms, x, attn []float32) error {
	t := m.gpuLayers[layer]
	if m.cfg.Variant == VariantGemma2 {
		return m.fused.AttnFFNGeGLUPostNorm(t, m.fusedDims, ln.ffnNorm, ln.attnPostNorm, ln.ffnPostNorm, x, attn)
	}

	return m.fused.AttnFFNGeGLU(t, m.fusedDims, ln.ffnNorm, x, attn)
}

// tryRecoverGPUKV rebuilds device KV from the CPU mirror without ResetCache
func (m *Model) tryRecoverGPUKV() bool {
	if m.dev == nil || !m.decode.GPUKVStale || m.decode.CPUKVHollow {
		return false
	}

	kvDim := m.cfg.NumKVHeads * m.cfg.HeadDim
	m.dev.KVCacheReset()
	for layer := 0; layer < m.cfg.NumLayers; layer++ {
		if !device.LayerOnGPU(layer, m.ngl, m.cfg.NumLayers) {
			continue
		}

		k := m.cache.KLayer(layer)
		v := m.cache.VLayer(layer)
		if len(k) == 0 || len(v) == 0 || kvDim == 0 {
			continue
		}

		n := len(k) / kvDim
		if n < 1 || len(k) != n*kvDim || len(v) != n*kvDim {
			return false
		}

		// SWA reclaim leaves LayerLen < Len(); absolute kvPos = Len() would be wrong.
		if m.cfg.IsSWALayer(layer) && m.cache.Len() != n {
			return false
		}

		if err := m.dev.KVCacheAppendN(layer, 0, k, v, n); err != nil {
			return false
		}
	}
	m.decode.GPUKVStale = false

	return true
}

// recoverHiddenFromDevice brings hidden state back to host after a device-layer failure
func (m *Model) recoverHiddenFromDevice(layer int, cause error) error {
	m.decode.ResidDevice = false
	m.residency = false

	if !m.dev.HiddenActive() {
		return fmt.Errorf("gemma: layer %d on GPU: %w (hidden state lost)", layer, cause)
	}

	if err := m.dev.HiddenDownload(m.scratch.x); err != nil {
		return fmt.Errorf("gemma: layer %d on GPU: %w (fallback to host: %v)", layer, cause, err)
	}

	return nil
}

// syncHiddenFromDevice pulls hidden state from device when resident (for host readers)
func (m *Model) syncHiddenFromDevice() error {
	if !m.decode.ResidDevice {
		return nil
	}

	m.decode.ResidDevice = false

	return m.dev.HiddenDownload(m.scratch.x)
}

// logitsDevice computes RMSNorm(resident hidden) + lm_head on GPU without DtoH of hidden state
func (m *Model) logitsDevice() error {
	if err := m.fused.LogitsDevice("output_norm.weight", m.outNorm, m.lmHeadName, m.scratch.logits, m.cfg.VocabSize, m.cfg.EmbeddingDim, m.cfg.RMSNormEps); err != nil {
		return err
	}

	return m.fused.SoftcapLogits(m.scratch.logits, m.cfg.FinalLogitSoftcap)
}

// logitsDeviceFromHost: one HtoD of hidden + out_norm + lm_head (non-resident path)
func (m *Model) logitsDeviceFromHost(x []float32) error {
	if err := m.fused.LogitsFromHost(x, "output_norm.weight", m.outNorm, m.lmHeadName, m.scratch.logits, m.cfg.VocabSize, m.cfg.EmbeddingDim, m.cfg.RMSNormEps); err != nil {
		return err
	}

	return m.fused.SoftcapLogits(m.scratch.logits, m.cfg.FinalLogitSoftcap)
}

// recoverLogits disables GPU logits after a failure so the caller falls back to the host lm_head.
func (m *Model) recoverLogits() {
	m.decode.LogitsOnGPU = false
}
