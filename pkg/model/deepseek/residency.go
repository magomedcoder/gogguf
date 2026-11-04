package deepseek

import (
	"fmt"

	"github.com/magomedcoder/gogguf/pkg/device"
	"github.com/magomedcoder/gogguf/pkg/exec/gpuresid"
	"github.com/magomedcoder/gogguf/pkg/exec/moe"
)

// initFused sets up GPU layer paths. DeepSeek v1: RoPE NORM, experts via ExpertFFN.
// MoE (n_batch=1): QKVAttn + AttnOutNorm + ExpertFFNAccumulate when residency enables.
func (m *Model) initFused() {
	if m.dev == nil || m.ngl <= 0 {
		return
	}

	m.fused = gpuresid.New(m.weights, m.dev, device.RoPENorm)
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
	m.moeGPU = make([]bool, m.cfg.NumLayers)
	moeOK := true
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
		m.layerQuant[i] = device.FusedQuantSupported(gpuresid.LayerQuant(m.weights, names...))

		if lt.moe {
			m.moeGPU[i] = m.fused.MoESupported(lt.gateExps, lt.upExps, lt.downExps)
			if !m.moeGPU[i] {
				moeOK = false
			}
		}
	}

	m.initResidency(moeOK)
}

func (m *Model) initResidency(moeOK bool) {
	if m.fused == nil {
		return
	}

	if m.cfg.ExpertCount > 0 && (!moeOK || !gpuresid.MoEResidSupported(m.dev)) {
		m.decode.LogitsOnGPU = gpuresid.CanLogitsFromDevice(m.dev, m.ngl, m.cfg.NumLayers) && m.fused.HeadOnGPU(m.lmHeadName)
		return
	}

	if !gpuresid.AllLayersOffloaded(m.ngl, m.cfg.NumLayers) {
		m.decode.LogitsOnGPU = gpuresid.CanLogitsFromDevice(m.dev, m.ngl, m.cfg.NumLayers) && m.fused.HeadOnGPU(m.lmHeadName)
		return
	}

	if !gpuresid.CanEnableResidency(m.dev, m.layerQuant) {
		m.decode.LogitsOnGPU = gpuresid.CanLogitsFromDevice(m.dev, m.ngl, m.cfg.NumLayers) && m.fused.HeadOnGPU(m.lmHeadName)
		return
	}

	m.residency = true
	m.decode.LogitsOnGPU = true
}

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

func (m *Model) forwardBlockDevice(layer, pos int, ln layerNorms) error {
	if err := m.qkvAttnGPU(layer, pos, ln, nil, nil, nil, nil); err != nil {
		return err
	}

	if err := m.cache.Bump(layer, 1); err != nil {
		return err
	}
	m.decode.CPUKVHollow = true

	return m.fused.AttnFFN(m.gpuLayers[layer], m.fusedDims, ln.ffnNorm, nil, nil)
}

func (m *Model) forwardBlockDeviceMoE(layer, pos int, ln layerNorms, lt layerTensors) error {
	if err := m.qkvAttnGPU(layer, pos, ln, nil, nil, nil, nil); err != nil {
		return err
	}

	if err := m.cache.Bump(layer, 1); err != nil {
		return err
	}
	m.decode.CPUKVHollow = true

	if err := m.fused.AttnOutNorm(m.gpuLayers[layer], m.fusedDims, ln.ffnNorm, nil, nil); err != nil {
		return err
	}

	embd := m.cfg.EmbeddingDim
	nExp := m.cfg.ExpertCount
	ffn := m.cfg.ExpertFFN
	if err := m.fused.MatMulFromMoEHidden(lt.gateInp, nExp, embd, m.scratch.router[:nExp]); err != nil {
		return err
	}

	idxs, weights := moe.TopKSoftmax(m.scratch.router[:nExp], m.cfg.ExpertUsedCount, false, m.cfg.ExpertWeightScale)
	for i, ei := range idxs {
		if err := m.fused.ExpertFFNAccumulate(lt.gateExps, lt.upExps, lt.downExps, ei, weights[i], embd, ffn); err != nil {
			return err
		}
	}

	if m.cfg.ExpertShared > 0 && lt.gateShexp != "" {
		shared := m.cfg.sharedFFN()
		if err := m.fused.FFNNamedAccumulate(lt.gateShexp, lt.upShexp, lt.downShexp, 1, embd, shared); err != nil {
			return err
		}
	}

	return nil
}

func (m *Model) qkvAttnGPU(layer, pos int, ln layerNorms, h, kOut, vOut, attn []float32) error {
	kvPos := m.cache.Len()

	return m.fused.QKVAttn(m.gpuLayers[layer], m.fusedDims, nil, nil, ln.attnNorm, h, attn, kOut, vOut, layer, pos, kvPos, kvPos+1)
}

func (m *Model) recoverHiddenFromDevice(layer int, cause error) error {
	m.decode.ResidDevice = false
	m.residency = false

	if !m.dev.HiddenActive() {
		return fmt.Errorf("deepseek: layer %d on GPU: %w (hidden state lost)", layer, cause)
	}

	if err := m.dev.HiddenDownload(m.scratch.x); err != nil {
		return fmt.Errorf("deepseek: layer %d on GPU: %w (fallback to host: %v)", layer, cause, err)
	}

	return nil
}

func (m *Model) syncHiddenFromDevice() error {
	if !m.decode.ResidDevice {
		return nil
	}

	m.decode.ResidDevice = false

	return m.dev.HiddenDownload(m.scratch.x)
}

func (m *Model) logitsDevice() error {
	return m.fused.LogitsDevice("output_norm.weight", m.outNorm, m.lmHeadName, m.scratch.logits, m.cfg.VocabSize, m.cfg.EmbeddingDim, m.cfg.RMSNormEps)
}
