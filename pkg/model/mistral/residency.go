package mistral

import (
	"fmt"

	"github.com/magomedcoder/gogguf/pkg/device"
	"github.com/magomedcoder/gogguf/pkg/exec/gpuresid"
	"github.com/magomedcoder/gogguf/pkg/exec/moe"
)

// initFused sets up fused GPU layer paths. Mistral / Qwen2: NeoX RoPE, no QK-norm
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

		m.layerQuant[i] = device.FusedQuantSupported(gpuresid.LayerQuant(m.weights, names...))

		// experts on GPU only with natively supported quant
		if lt.moe && !m.fused.MoESupported(lt.gateExps, lt.upExps, lt.downExps) {
			m.moeGPU = false
		}
	}
}

// initResidency enables device-resident hidden state.
// MoE (n_batch=1): QKVAttn + AttnOutNorm + router DtoH(n_expert) + ExpertFFNAccumulate (+ shared expert).
// PrefillBatchSize stays 1 for MoE; n_batch>1 expert dispatch remains deferred.
func (m *Model) initResidency() {
	if m.fused == nil {
		return
	}

	if m.cfg.isMoE() {
		if !m.moeGPU || !gpuresid.MoEResidSupported(m.dev) {
			m.initMoELogits()
			return
		}
	}

	if !gpuresid.AllLayersOffloaded(m.ngl, m.cfg.NumLayers) {
		return
	}
	// SWA: fused QKV attends over the full GPU KV-cache; it does not apply the window
	if m.cfg.SlidingWindow > 0 && m.cfg.SlidingWindow < m.cfg.ContextLength {
		if m.cfg.isMoE() {
			m.initMoELogits()
		}
		return
	}
	if !gpuresid.CanEnableResidency(m.dev, m.layerQuant) {
		if m.cfg.isMoE() {
			m.initMoELogits()
		}
		return
	}
	m.residency = true
	m.decode.LogitsOnGPU = true
}

// gpuAttnFull: full GPU KV-cache attention is correct only when the SWA window covers the full sequence
func (m *Model) gpuAttnFull(seqLen int) bool {
	w := m.cfg.SlidingWindow

	return w <= 0 || seqLen <= w
}

// uploadHidden uploads hidden state before layers (one HtoD per token)
func (m *Model) uploadHidden() {
	if !m.residency || m.debug != nil || m.decode.GPUKVStale || !m.gpuAttnFull(m.cache.Len()+1) {
		return
	}

	if err := m.dev.HiddenUpload(m.scratch.x); err != nil {
		m.residency = false
		return
	}

	m.decode.ResidDevice = true
}

// forwardBlockDevice runs a dense layer on GPU over resident hidden state.
// Skips host k/v/attn mirrors (GPU KV + d_vec are source of truth) to cut decode PCIe.
func (m *Model) forwardBlockDevice(layer, pos int, ln layerNorms) error {
	if err := m.qkvAttnGPU(layer, pos, ln, nil, nil, nil, nil); err != nil {
		return err
	}

	if err := m.cache.Bump(layer, 1); err != nil {
		return err
	}
	m.decode.CPUKVHollow = true

	return m.attnFFNGPU(layer, ln, nil, nil)
}

// forwardBlockDeviceMoE: QKVAttn resid + WO/norm snapshot + router top-k + ExpertFFNAccumulate (+ shared)
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
	ffn := m.cfg.expertFFN()
	if err := m.fused.MatMulFromMoEHidden(lt.gateInp, nExp, embd, m.scratch.router[:nExp]); err != nil {
		return err
	}

	idxs, weights := moe.TopKSoftmax(m.scratch.router[:nExp], m.cfg.ExpertUsedCount, m.cfg.MoENormWeights, m.cfg.ExpertWeightScale)
	for i, ei := range idxs {
		if err := m.fused.ExpertFFNAccumulate(lt.gateExps, lt.upExps, lt.downExps, ei, weights[i], embd, ffn); err != nil {
			return err
		}
	}

	return m.sharedExpertDevice(lt)
}

// sharedExpertDevice adds the shared expert into d_resid from the MoE snapshot
func (m *Model) sharedExpertDevice(lt layerTensors) error {
	if lt.gateShexp == "" {
		return nil
	}

	embd := m.cfg.EmbeddingDim
	shared := m.cfg.SharedFFN
	if shared <= 0 {
		shared = m.cfg.FFNHidden
	}

	gateScale := float32(1)
	if lt.gateInpShexp != "" {
		var gateInp [1]float32
		if err := m.fused.MatMulFromMoEHidden(lt.gateInpShexp, 1, embd, gateInp[:]); err != nil {
			return err
		}
		gateScale = siluDiv(gateInp[0])
	}

	return m.fused.FFNNamedAccumulate(lt.gateShexp, lt.upShexp, lt.downShexp, gateScale, embd, shared)
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
	m.decode.ResidDevice = false
	m.residency = false

	if !m.dev.HiddenActive() {
		return fmt.Errorf("mistral: layer %d on GPU: %w (hidden state lost)", layer, cause)
	}

	if err := m.dev.HiddenDownload(m.scratch.x); err != nil {
		return fmt.Errorf("mistral: layer %d on GPU: %w (fallback to host: %v)", layer, cause, err)
	}

	return nil
}

// syncHiddenFromDevice pulls hidden state from device when resident there
func (m *Model) syncHiddenFromDevice() error {
	if !m.decode.ResidDevice {
		return nil
	}

	m.decode.ResidDevice = false

	return m.dev.HiddenDownload(m.scratch.x)
}

// logitsDevice computes RMSNorm(resident hidden) + lm_head on GPU without DtoH hidden
func (m *Model) logitsDevice() error {
	return m.fused.LogitsDevice("output_norm.weight", m.outNorm, m.lmHeadName, m.scratch.logits, m.cfg.VocabSize, m.cfg.EmbeddingDim, m.cfg.RMSNormEps)
}

// initMoELogits: MoE layers run on host, but the final hidden state can still go through the GPU lm_head
func (m *Model) initMoELogits() {
	if gpuresid.CanLogitsFromDevice(m.dev, m.ngl, m.cfg.NumLayers) && m.fused.HeadOnGPU(m.lmHeadName) {
		m.decode.LogitsOnGPU = true
	}
}

// logitsFromHost: hidden state is on host (MoE layers): upload once, out_norm + lm_head on GPU
func (m *Model) logitsFromHost() error {
	return m.fused.LogitsFromHost(m.scratch.x, "output_norm.weight", m.outNorm, m.lmHeadName, m.scratch.logits, m.cfg.VocabSize, m.cfg.EmbeddingDim, m.cfg.RMSNormEps)
}
