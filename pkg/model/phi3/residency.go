package phi3

import (
	"fmt"

	"github.com/magomedcoder/gogguf/pkg/device"
	"github.com/magomedcoder/gogguf/pkg/exec/gpuresid"
)

// initFused sets up fused GPU layer paths (NeoX RoPE + LongRoPE via Dims.NRot/RopeScale).
//
// Fused attn_qkv and merged ffn_up (gate||up) are exposed to gpuresid as row-slices (AttnQKV / FFNGateUp) so QKVAttn and AttnFFN can run without host dequant.
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
		NRot:     m.cfg.RopeDim,
	}

	m.gpuLayers = make([]gpuresid.Tensors, m.cfg.NumLayers)
	m.layerQuant = make([]bool, m.cfg.NumLayers)
	m.ffnGPU = make([]bool, m.cfg.NumLayers)
	for i, lt := range m.layerTensors {
		p := fmt.Sprintf("blk.%d.", i)
		t := gpuresid.Tensors{
			AttnOut:  lt.attnOut,
			FFNDown:  lt.ffnDown,
			AttnNorm: p + "attn_norm.weight",
			FFNNorm:  p + "ffn_norm.weight",
		}

		names := []string{lt.attnOut, lt.ffnDown}
		if lt.fusedQKV {
			t.AttnQKV = lt.attnQKV
			names = append(names, lt.attnQKV)
		} else {
			t.AttnQ = lt.attnQ
			t.AttnK = lt.attnK
			t.AttnV = lt.attnV
			names = append(names, lt.attnQ, lt.attnK, lt.attnV)
		}

		if lt.fusedFFN {
			t.FFNGateUp = lt.ffnUp
			names = append(names, lt.ffnUp)
		} else {
			t.FFNGate = lt.ffnGate
			t.FFNUp = lt.ffnUp
			names = append(names, lt.ffnGate, lt.ffnUp)
			m.ffnGPU[i] = device.LayerOnGPU(i, m.ngl, m.cfg.NumLayers)
		}
		m.gpuLayers[i] = t

		qt := gpuresid.LayerQuant(m.weights, names...)
		m.layerQuant[i] = device.FusedQuantSupported(qt)
	}
}

// initResidency enables device-resident hidden state when all layers are offloaded and quants support the fused path (including fused attn_qkv / merged ffn_up via row-slices)
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

	m.residency = true
	m.decode.LogitsOnGPU = true
}

// uploadHidden uploads hidden state before layers (one HtoD per token)
func (m *Model) uploadHidden() {
	if !m.residency || m.decode.GPUKVStale {
		return
	}

	if err := m.dev.HiddenUpload(m.scratch.x[:m.cfg.EmbeddingDim]); err != nil {
		m.residency = false
		return
	}
	m.decode.ResidDevice = true
}

// forwardBlockDevice runs the layer on GPU over resident hidden state
func (m *Model) forwardBlockDevice(layer, pos int, ln layerNorms) error {
	dims := m.fusedDims
	dims.RopeScale = m.ropeScale()

	if err := m.fused.QKVAttn(m.gpuLayers[layer], dims, nil, nil, ln.attnNorm, nil, nil, nil, nil, layer, pos, m.cache.Len(), m.cache.Len()+1); err != nil {
		return err
	}

	if err := m.cache.Bump(layer, 1); err != nil {
		return err
	}
	m.decode.CPUKVHollow = true

	return m.fused.AttnFFN(m.gpuLayers[layer], dims, ln.ffnNorm, nil, nil)
}

// qkvAttnGPU: QKV + LongRoPE + KV append + attention on GPU (host h)
func (m *Model) qkvAttnGPU(layer, pos int, ln layerNorms, h, kOut, vOut, attn []float32) error {
	dims := m.fusedDims
	dims.RopeScale = m.ropeScale()
	kvPos := m.cache.Len()

	return m.fused.QKVAttn(m.gpuLayers[layer], dims, nil, nil, ln.attnNorm, h, attn, kOut, vOut, layer, pos, kvPos, kvPos+1)
}

// attnFFNGPU: WO + residual + ffn_norm + FFN + residual on GPU
func (m *Model) attnFFNGPU(layer int, ln layerNorms, x, attn []float32) error {
	return m.fused.AttnFFN(m.gpuLayers[layer], m.fusedDims, ln.ffnNorm, x, attn)
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

		if err := m.dev.KVCacheAppendN(layer, 0, k, v, n); err != nil {
			return false
		}
	}
	m.decode.GPUKVStale = false

	return true
}

// logitsDeviceResident: LogitsDevice over resident residual (no hidden DtoH)
func (m *Model) logitsDeviceResident() error {
	if err := m.fused.LogitsDevice("output_norm.weight", m.outNorm, m.lmHeadName, m.scratch.logits, m.cfg.VocabSize, m.cfg.EmbeddingDim, m.cfg.RMSNormEps); err != nil {
		return err
	}
	m.decode.ResidDevice = false
	return nil
}

// recoverHiddenFromDevice brings hidden state back to host after a device-layer failure
func (m *Model) recoverHiddenFromDevice(layer int, cause error) error {
	m.decode.ResidDevice = false
	m.residency = false

	if !m.dev.HiddenActive() {
		return fmt.Errorf("phi3: layer %d on GPU: %w (hidden state lost)", layer, cause)
	}

	if err := m.dev.HiddenDownload(m.scratch.x[:m.cfg.EmbeddingDim]); err != nil {
		return fmt.Errorf("phi3: layer %d on GPU: %w (fallback to host: %v)", layer, cause, err)
	}

	return nil
}

// syncHiddenFromDevice pulls hidden state from device when resident (for host readers)
func (m *Model) syncHiddenFromDevice() error {
	if !m.decode.ResidDevice {
		return nil
	}

	m.decode.ResidDevice = false

	return m.dev.HiddenDownload(m.scratch.x[:m.cfg.EmbeddingDim])
}

// logitsDevice computes RMSNorm(final hidden) + lm_head on GPU: one HtoD of the hidden state, DtoH of logits only
func (m *Model) logitsDevice(x []float32) error {
	return m.fused.LogitsFromHost(x, "output_norm.weight", m.outNorm, m.lmHeadName, m.scratch.logits, m.cfg.VocabSize, m.cfg.EmbeddingDim, m.cfg.RMSNormEps)
}

// logitsFromLast: lm_head on GPU when enabled, else on host
func (m *Model) logitsFromLast(x []float32) error {
	if m.decode.LogitsOnGPU && m.fused != nil {
		if m.decode.ResidDevice {
			if err := m.fused.LogitsDevice("output_norm.weight", m.outNorm, m.lmHeadName, m.scratch.logits, m.cfg.VocabSize, m.cfg.EmbeddingDim, m.cfg.RMSNormEps); err == nil {
				m.decode.ResidDevice = false
				return nil
			}
			if err := m.syncHiddenFromDevice(); err != nil {
				return err
			}
		}
		if err := m.logitsDevice(x); err == nil {
			return nil
		}

		// lm_head on GPU failed (weight type / VRAM): continue on host
		m.decode.LogitsOnGPU = false
	}

	return m.logitsFromHidden(x)
}
