package phi2

import (
	"fmt"

	"github.com/magomedcoder/gogguf/pkg/compute"
	"github.com/magomedcoder/gogguf/pkg/device"
	"github.com/magomedcoder/gogguf/pkg/dtype"
	"github.com/magomedcoder/gogguf/pkg/exec/gpuresid"
)

// initResidency enables Phi-2 GPU residency when ngl covers every layer.
//
// Full fused ParallelAttnFFN (LN->QKV->RoPE->attn->WO+bias∥Up->GELU->Down+bias->resid) is not used: Phi-2 needs LayerNorm+bias, parallel residual, projection biases and partial NeoX RoPE.
// Instead: keep residual on device, per-layer device LN + host/GPU matmuls + AttentionScoresKV + GELU FFN + ResidAdd, and LogitsFromDeviceLayerNorm for the final head.
func (m *Model) initResidency() {
	if m.dev == nil {
		return
	}

	m.fused = gpuresid.New(m.weights, m.dev, device.RoPENeoX)

	if gpuresid.CanLogitsFromDevice(m.dev, m.ngl, m.cfg.NumLayers) && m.fused.HeadOnGPU(m.lmHeadName) {
		m.decode.LogitsOnGPU = true
	}

	if !gpuresid.AllLayersOffloaded(m.ngl, m.cfg.NumLayers) || !gpuresid.HiddenResidencyEnabled() {
		return
	}
	if !m.dev.HiddenResident() {
		return
	}

	m.layerQuant = make([]bool, m.cfg.NumLayers)
	allQuant := true
	for i, lt := range m.layerTensors {
		var t dtype.DType
		if lt.fusedQKV {
			t = gpuresid.LayerQuant(m.weights, lt.attnQKV, lt.attnOut, lt.ffnUp, lt.ffnDown)
		} else {
			t = gpuresid.LayerQuant(m.weights, lt.attnQ, lt.attnK, lt.attnV, lt.attnOut, lt.ffnUp, lt.ffnDown)
		}

		ok := device.FusedQuantSupported(t)
		m.layerQuant[i] = ok
		if !ok {
			allQuant = false
		}
	}
	if !allQuant {
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

	if err := m.dev.HiddenUpload(m.scratch.x); err != nil {
		m.residency = false
		return
	}

	m.decode.ResidDevice = true
}

// forwardBlockDevice runs one Phi-2 layer over resident residual: device LN -> QKV/RoPE/attn (host buffers + GPU matmul/KV) -> ResidAdd(attn) -> GELU FFN -> ResidAdd(ffn).
func (m *Model) forwardBlockDevice(layer, pos int) error {
	ln := m.layerNorms[layer]
	lt := m.layerTensors[layer]
	lb := m.layerBias[layer]
	p := fmt.Sprintf("blk.%d.", layer)
	eps := m.cfg.LayerNormEps

	if err := m.dev.LayerNormResidInto(m.scratch.h, ln.attnW, ln.attnB, p+"attn_norm.weight", p+"attn_norm.bias", eps); err != nil {
		return err
	}

	if err := m.projectQKV(lt, lb, layer); err != nil {
		return err
	}

	// Partial NeoX RoPE on host (fused QKV path does not support rope_dim < head_dim)
	compute.ApplyRoPEHeadsPartial(m.scratch.q, m.cfg.NumHeads, m.cfg.HeadDim, m.cfg.RopeDim, pos, m.cfg.RopeFreqBase)
	compute.ApplyRoPEHeadsPartial(m.scratch.k, m.cfg.NumKVHeads, m.cfg.HeadDim, m.cfg.RopeDim, pos, m.cfg.RopeFreqBase)

	kvPos := m.cache.Len()
	if err := m.cache.Bump(layer, 1); err != nil {
		return err
	}
	m.decode.CPUKVHollow = true

	if err := m.dev.KVCacheAppend(layer, kvPos, m.scratch.k, m.scratch.v); err != nil {
		return err
	}

	seqLen := kvPos + 1
	if err := m.dev.AttentionScoresKV(layer, m.scratch.attn, m.scratch.q, seqLen, m.cfg.NumHeads, m.cfg.NumKVHeads, m.cfg.HeadDim); err != nil {
		return err
	}

	if err := m.matmulInto(lt.attnOut, m.cfg.EmbeddingDim, m.cfg.NumHeads*m.cfg.HeadDim, m.scratch.attn, m.scratch.attnOut, layer); err != nil {
		return err
	}

	compute.AddBiasInPlace(m.scratch.attnOut, lb.wo)
	if err := m.dev.ResidAdd(m.scratch.attnOut); err != nil {
		return err
	}

	if err := m.matmulInto(lt.ffnUp, m.cfg.FFNHidden, m.cfg.EmbeddingDim, m.scratch.h, m.scratch.up, layer); err != nil {
		return err
	}

	compute.AddBiasInPlace(m.scratch.up, lb.ffnUp)
	if err := m.dev.GELUInPlace(m.scratch.up); err != nil {
		compute.GELUInPlace(m.scratch.up)
	}

	if err := m.matmulInto(lt.ffnDown, m.cfg.EmbeddingDim, m.cfg.FFNHidden, m.scratch.up, m.scratch.attnOut, layer); err != nil {
		return err
	}
	compute.AddBiasInPlace(m.scratch.attnOut, lb.ffnDn)

	return m.dev.ResidAdd(m.scratch.attnOut)
}

// recoverHiddenFromDevice brings hidden state back to host after a device-layer failure
func (m *Model) recoverHiddenFromDevice(layer int, cause error) error {
	m.decode.ResidDevice = false
	m.residency = false

	if !m.dev.HiddenActive() {
		return fmt.Errorf("phi2: layer %d on GPU: %w (hidden state lost)", layer, cause)
	}

	if err := m.dev.HiddenDownload(m.scratch.x); err != nil {
		return fmt.Errorf("phi2: layer %d on GPU: %w (fallback to host: %v)", layer, cause, err)
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

// logitsDevice: LayerNorm+bias(resident hidden) + lm_head on GPU
func (m *Model) logitsDevice() error {
	return m.fused.LogitsDeviceLayerNorm("output_norm.weight", "output_norm.bias", m.outNorm.w, m.outNorm.b, m.lmHeadName, m.scratch.logits, m.cfg.VocabSize, m.cfg.EmbeddingDim, m.cfg.LayerNormEps)
}

// logitsDeviceFromHost uploads x then LayerNorm+bias + lm_head on GPU
func (m *Model) logitsDeviceFromHost(x []float32) error {
	return m.fused.LogitsFromHostLayerNorm(x, "output_norm.weight", "output_norm.bias", m.outNorm.w, m.outNorm.b, m.lmHeadName, m.scratch.logits, m.cfg.VocabSize, m.cfg.EmbeddingDim, m.cfg.LayerNormEps)
}

// lmHeadGPU computes logits = lm_head * h on GPU; false means fall back to the host lm_head
func (m *Model) lmHeadGPU(h []float32) bool {
	if !m.decode.LogitsOnGPU || m.fused == nil {
		return false
	}

	if err := m.fused.MatMulInto(m.lmHeadName, m.cfg.VocabSize, m.cfg.EmbeddingDim, h, m.scratch.logits); err != nil {
		m.decode.LogitsOnGPU = false
		return false
	}

	return true
}
