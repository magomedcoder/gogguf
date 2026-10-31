package qwen3

import (
	"fmt"

	"github.com/magomedcoder/gogguf/pkg/device"
	"github.com/magomedcoder/gogguf/pkg/exec/gpuresid"
)

// initFused sets up shared fused GPU path wiring: Qwen3 - NeoX RoPE + QK-norm
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
			QNorm:    p + "attn_q_norm.weight",
			KNorm:    p + "attn_k_norm.weight",
			AttnNorm: p + "attn_norm.weight",
			FFNNorm:  p + "ffn_norm.weight",
		}
	}

	// experts on GPU only if all expert weights are natively quantized
	m.moeGPU = m.cfg.ExpertCount > 0
	for _, lt := range m.layerTensors {
		if lt.moe && !m.fused.MoESupported(lt.gateExps, lt.upExps, lt.downExps) {
			m.moeGPU = false
			break
		}
	}
}

// initResidency enables device-resident hidden state: all layers on GPU required, fused QKV/residual paths and a single supported weight quant across all layers
// MoE (P1 deferred): full device-resident MoE layers need batched expert dispatch (router top-k, then per-expert FFN and accumulation on device) and n_batch>1 support.
// Until then MoE layers keep the per-token ExpertFFN path (n_batch=1, hidden state on host); only out_norm + lm_head may run on GPU via gpuresid.LogitsFromHost when ngl covers every layer.
func (m *Model) initResidency() {
	if m.cfg.ExpertCount > 0 {
		if m.fused != nil && gpuresid.CanLogitsFromDevice(m.dev, m.ngl, m.cfg.NumLayers) && m.fused.HeadOnGPU(m.lmHeadName) {
			m.logitsOnGPU = true
		}

		return
	}

	if !gpuresid.AllLayersOffloaded(m.ngl, m.cfg.NumLayers) {
		return
	}
	ok := make([]bool, len(m.layerQuant))
	for i, t := range m.layerQuant {
		ok[i] = device.FusedQuantSupported(t)
	}
	if !gpuresid.CanEnableResidency(m.dev, ok) {
		return
	}
	m.residency = true
	m.logitsOnGPU = true
}

// forwardBlockDevice runs the layer on GPU over resident hidden state.
// Skips host k/v/attn mirrors (GPU KV + d_vec are source of truth) to cut decode PCIe.
func (m *Model) forwardBlockDevice(layer, pos int, ln layerNorms) error {
	if err := m.qkvAttnGPU(layer, pos, ln, nil, nil, nil, nil); err != nil {
		return err
	}

	if err := m.cache.Bump(layer, 1); err != nil {
		return err
	}
	m.cpuKVHollow = true

	return m.attnFFNGPU(layer, ln, nil, nil)
}

// tryRecoverGPUKV rebuilds device KV from the CPU mirror without ResetCache (session stays).
func (m *Model) tryRecoverGPUKV() bool {
	if m.dev == nil || !m.gpuKVStale || m.cpuKVHollow {
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

		if err := m.dev.KVCacheAppendN(layer, 0, k, v, n); err != nil {
			return false
		}
	}
	m.gpuKVStale = false

	return true
}

// recoverHiddenFromDevice brings hidden state back to host after a device-layer failure.
// If residual was partially updated, valid state exists nowhere - error.
func (m *Model) recoverHiddenFromDevice(layer int, cause error) error {
	m.residDevice = false
	m.residency = false

	if !m.dev.HiddenActive() {
		return fmt.Errorf("qwen3: layer %d on GPU: %w (hidden state lost)", layer, cause)
	}

	if err := m.dev.HiddenDownload(m.scratch.x[:m.cfg.EmbeddingDim]); err != nil {
		return fmt.Errorf("qwen3: layer %d on GPU: %w (fallback to host: %v)", layer, cause, err)
	}

	return nil
}

// syncHiddenFromDevice pulls hidden state from device when resident (for host readers)
func (m *Model) syncHiddenFromDevice() error {
	if !m.residDevice {
		return nil
	}

	m.residDevice = false

	return m.dev.HiddenDownload(m.scratch.x[:m.cfg.EmbeddingDim])
}

func attnFFNResidencyEnabled() bool {
	return gpuresid.AttnFFNResidencyEnabled()
}

// hiddenResidencyEnabled - device-resident hidden state (LoadOptions.HiddenResidency)
func hiddenResidencyEnabled() bool {
	return gpuresid.HiddenResidencyEnabled()
}

func qkvResidencyEnabled() bool {
	return gpuresid.QKVResidencyEnabled()
}

// attnFFNGPU: WO + residual + ffn_norm + FFN + residual on GPU.
// x == nil - hidden resident on device .
func (m *Model) attnFFNGPU(layer int, ln layerNorms, x, attn []float32) error {
	return m.fused.AttnFFN(m.gpuLayers[layer], m.fusedDims, ln.ffnNorm, x, attn)
}

// qkvAttnGPU: QKV + QK-norm + RoPE + KV append + attention on GPU.
// h == nil - hidden resident: attn_norm computed on device from residual.
func (m *Model) qkvAttnGPU(layer, pos int, ln layerNorms, h, kOut, vOut, attn []float32) error {
	kvPos := m.cache.Len()

	return m.fused.QKVAttn(m.gpuLayers[layer], m.fusedDims, ln.qNorm, ln.kNorm, ln.attnNorm, h, attn, kOut, vOut, layer, pos, kvPos, kvPos+1)
}

func (m *Model) ffnGPU(layer int, x, out []float32) error {
	return m.fused.FFN(m.gpuLayers[layer], m.fusedDims, x, out)
}

// logitsDevice computes RMSNorm(resident hidden) + lm_head on GPU without DtoH of hidden state
func (m *Model) logitsDevice() error {
	return m.fused.LogitsDevice("output_norm.weight", m.outNorm, m.lmHeadName, m.scratch.logits, m.cfg.VocabSize, m.cfg.EmbeddingDim, m.cfg.RMSNormEps)
}
