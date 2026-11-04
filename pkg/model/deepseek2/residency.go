package deepseek2

import (
	"fmt"

	"github.com/magomedcoder/gogguf/pkg/compute"
	"github.com/magomedcoder/gogguf/pkg/device"
	"github.com/magomedcoder/gogguf/pkg/exec/gpuresid"
	"github.com/magomedcoder/gogguf/pkg/exec/moe"
)

// initFused sets up GPU layer paths. MLA attention uses the dedicated device MLA cache.
// MoE (n_batch=1): after host/GPU MLA + WO residual, MoEHiddenNorm + ExpertFFNAccumulate into d_resid.
func (m *Model) initFused() {
	if m.dev == nil || m.ngl <= 0 {
		return
	}

	m.fused = gpuresid.New(m.weights, m.dev, device.RoPENeoX)
	m.moeGPU = make([]bool, m.cfg.NumLayers)
	m.layerQuant = make([]bool, m.cfg.NumLayers)
	moeOK := true
	for i, lt := range m.layerTensors {
		if lt.moe {
			m.moeGPU[i] = m.fused.MoESupported(lt.gateExps, lt.upExps, lt.downExps)
			if !m.moeGPU[i] {
				moeOK = false
			}
			// attn out + router must be fused-quant for resid MoE prep
			names := []string{lt.attnOut, lt.gateInp}
			m.layerQuant[i] = device.FusedQuantSupported(gpuresid.LayerQuant(m.weights, names...))
			continue
		}

		names := []string{lt.attnOut, lt.ffnGate, lt.ffnUp, lt.ffnDown}
		m.layerQuant[i] = device.FusedQuantSupported(gpuresid.LayerQuant(m.weights, names...))
	}

	m.initResidency(moeOK)
}

func (m *Model) initResidency(moeOK bool) {
	if m.fused == nil {
		return
	}

	if m.cfg.ExpertCount > 0 && (!moeOK || !gpuresid.MoEResidSupported(m.dev)) {
		return
	}

	if !gpuresid.AllLayersOffloaded(m.ngl, m.cfg.NumLayers) {
		return
	}

	// MLA keeps attention inputs on host; MoE/dense FFN still use HiddenResident flags
	if !m.dev.HiddenResident() || !gpuresid.HiddenResidencyEnabled() {
		return
	}

	for _, ok := range m.layerQuant {
		if !ok {
			return
		}
	}

	m.residency = true
	m.decode.LogitsOnGPU = gpuresid.CanLogitsFromDevice(m.dev, m.ngl, m.cfg.NumLayers) && m.fused.HeadOnGPU(m.lmHeadName)
}

// forwardBlockDevice: MLA already applied on host into x; run dense FFN into d_resid
func (m *Model) forwardBlockDevice(layer int, ln layerNorms, lt layerTensors) error {
	if err := m.dev.HiddenUpload(m.scratch.x); err != nil {
		return err
	}
	m.decode.ResidDevice = true

	p := fmt.Sprintf("blk.%d.", layer)
	if err := m.fused.MoEHiddenNorm(p+"ffn_norm.weight", ln.ffnNorm, m.cfg.EmbeddingDim, m.cfg.RMSNormEps); err != nil {
		return err
	}

	return m.fused.FFNNamedAccumulate(lt.ffnGate, lt.ffnUp, lt.ffnDown, 1, m.cfg.EmbeddingDim, m.cfg.FFNHidden)
}

// forwardBlockDeviceMoE: MLA already applied; MoE experts accumulate into d_resid
func (m *Model) forwardBlockDeviceMoE(layer int, ln layerNorms, lt layerTensors) error {
	if err := m.dev.HiddenUpload(m.scratch.x); err != nil {
		return err
	}
	m.decode.ResidDevice = true

	p := fmt.Sprintf("blk.%d.", layer)
	embd := m.cfg.EmbeddingDim
	if err := m.fused.MoEHiddenNorm(p+"ffn_norm.weight", ln.ffnNorm, embd, m.cfg.RMSNormEps); err != nil {
		return err
	}

	nExp := m.cfg.ExpertCount
	ffn := m.cfg.ExpertFFN
	if err := m.fused.MatMulFromMoEHidden(lt.gateInp, nExp, embd, m.scratch.router[:nExp]); err != nil {
		return err
	}

	if lt.expProbsB != "" {
		bias, err := m.weights.Floats(lt.expProbsB)
		if err != nil {
			return err
		}
		compute.AddBiasInPlace(m.scratch.router[:nExp], bias)
	}

	idxs, weights := moe.TopKSoftmax(m.scratch.router[:nExp], m.cfg.ExpertUsedCount, m.cfg.ExpertWeightsNorm, m.cfg.ExpertWeightScale)
	for i, ei := range idxs {
		if err := m.fused.ExpertFFNAccumulate(lt.gateExps, lt.upExps, lt.downExps, ei, weights[i], embd, ffn); err != nil {
			return err
		}
	}

	if lt.gateShexp != "" {
		shared := m.cfg.sharedFFN()
		if err := m.fused.FFNNamedAccumulate(lt.gateShexp, lt.upShexp, lt.downShexp, 1, embd, shared); err != nil {
			return err
		}
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

func (m *Model) recoverHiddenFromDevice(layer int, cause error) error {
	m.decode.ResidDevice = false
	m.residency = false

	if !m.dev.HiddenActive() {
		return fmt.Errorf("deepseek2: layer %d on GPU: %w (hidden state lost)", layer, cause)
	}

	if err := m.dev.HiddenDownload(m.scratch.x); err != nil {
		return fmt.Errorf("deepseek2: layer %d on GPU: %w (fallback to host: %v)", layer, cause, err)
	}

	return nil
}

func (m *Model) logitsDevice() error {
	return m.fused.LogitsDevice("output_norm.weight", m.outNorm, m.lmHeadName, m.scratch.logits, m.cfg.VocabSize, m.cfg.EmbeddingDim, m.cfg.RMSNormEps)
}
