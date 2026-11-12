package mistral

import (
	"github.com/magomedcoder/gogguf/pkg/compute"
	"github.com/magomedcoder/gogguf/pkg/device"
	"github.com/magomedcoder/gogguf/pkg/exec/dispatch"
	"github.com/magomedcoder/gogguf/pkg/exec/prefill"
)

// forwardBatchMoESerial - MoE prefill with n_batch>1: loop each token through resid MoE (forwardBlockDeviceMoE / host experts).
func (m *Model) forwardBatchMoESerial(tokenIDs []int, startPos int, needLogits bool) error {
	for i, tok := range tokenIDs {
		if err := m.forwardToken(tok, startPos+i, false); err != nil {
			return err
		}
	}
	if !needLogits {
		return nil
	}
	return m.logitsFinish()
}

// forwardBatch - multi-token prefill (n_batch): one layer pass per chunk (dense / SWA).
// With -ngl matmul uses MatMulMat*Cached; attention is windowed-causal (SWA skip on CPU; full-window GPU AttentionScoresKVBatch when safe).
// SWA reclaim still sets GPUKVStale until A4.
func (m *Model) forwardBatch(tokenIDs []int, startPos int, needLogits bool) error {
	embd := m.cfg.EmbeddingDim
	return prefill.Run("mistral", m.nBatch, m.cfg.NumLayers, startPos, tokenIDs, needLogits,
		m.syncHiddenFromDevice,
		func(b int) error {
			m.ensureKVRoom(b)
			return dispatch.EmbedTokensInto(m.weights, m.scratch.bx[:b*embd], embd, tokenIDs)
		},
		func(layer, startPos, batch int) error {
			return m.forwardBlockBatch(layer, startPos, batch)
		},
		func(b int) {
			m.cache.AdvanceN(b)
		},
		func(b int) error {
			return m.logitsFromHidden(m.scratch.bx[(b-1)*embd : b*embd])
		},
	)
}

func (m *Model) forwardBlockBatch(layer, startPos, batch int) error {
	ln := m.layerNorms[layer]
	lt := m.layerTensors[layer]
	embd := m.cfg.EmbeddingDim
	qDim := m.cfg.NumHeads * m.cfg.HeadDim
	kvDim := m.cfg.NumKVHeads * m.cfg.HeadDim
	ffn := m.cfg.FFNHidden

	x := m.scratch.bx[:batch*embd]
	h := m.scratch.bh[:batch*embd]
	q := m.scratch.bq[:batch*qDim]
	k := m.scratch.bk[:batch*kvDim]
	v := m.scratch.bv[:batch*kvDim]
	attn := m.scratch.battn[:batch*qDim]
	gate := m.scratch.bgate[:batch*ffn]
	up := m.scratch.bup[:batch*ffn]

	if err := compute.RMSNormBatchInto(h, x, ln.attnNorm, m.cfg.RMSNormEps, batch, embd); err != nil {
		return err
	}

	if err := m.matmulBatchInto(lt.attnQ, qDim, embd, h, q, batch, layer); err != nil {
		return err
	}

	if err := m.matmulBatchInto(lt.attnK, kvDim, embd, h, k, batch, layer); err != nil {
		return err
	}

	if err := m.matmulBatchInto(lt.attnV, kvDim, embd, h, v, batch, layer); err != nil {
		return err
	}

	// Mistral / Qwen2 RoPE: NeoX pairs (i, i+head_dim/2)
	for i := range batch {
		pos := startPos + i
		compute.ApplyRoPEHeads(q[i*qDim:(i+1)*qDim], m.cfg.NumHeads, m.cfg.HeadDim, pos, m.cfg.RopeFreqBase)
		compute.ApplyRoPEHeads(k[i*kvDim:(i+1)*kvDim], m.cfg.NumKVHeads, m.cfg.HeadDim, pos, m.cfg.RopeFreqBase)
	}

	pastLen := m.cache.Len()
	if err := m.cache.AppendN(layer, k, v, batch); err != nil {
		return err
	}

	// mirror chunk K/V into GPU KV-cache, else decode after batch-prefill sees incomplete cache
	if m.dev != nil && device.LayerOnGPU(layer, m.ngl, m.cfg.NumLayers) && !m.decode.GPUKVStale {
		if err := m.dev.KVCacheAppendN(layer, pastLen, k, v, batch); err != nil {
			m.decode.GPUKVStale = true
		}
	}

	if err := m.attentionBatchCausalInto(attn, q, layer, pastLen, batch); err != nil {
		return err
	}

	if err := m.matmulBatchInto(lt.attnOut, embd, qDim, attn, h, batch, layer); err != nil {
		return err
	}
	compute.AddInPlace(x, h)

	if err := compute.RMSNormBatchInto(h, x, ln.ffnNorm, m.cfg.RMSNormEps, batch, embd); err != nil {
		return err
	}

	if err := m.matmulBatchInto(lt.ffnGate, ffn, embd, h, gate, batch, layer); err != nil {
		return err
	}

	if err := m.matmulBatchInto(lt.ffnUp, ffn, embd, h, up, batch, layer); err != nil {
		return err
	}
	compute.SwiGLUInPlace(gate, up)

	if err := m.matmulBatchInto(lt.ffnDown, embd, ffn, gate, h, batch, layer); err != nil {
		return err
	}
	compute.AddInPlace(x, h)

	return nil
}

// attentionBatchCausalInto: causal multi-query attention for batch (query i sees past+i+1 keys).
// SWA: same key range as serial attentionKV (skip oldest outside the window).
// GPU AttentionScoresKVBatch only when the window covers the full sequence and KV is not stale.
func (m *Model) attentionBatchCausalInto(attn, q []float32, layer, pastLen, batch int) error {
	qDim := m.cfg.NumHeads * m.cfg.HeadDim
	kvDim := m.cfg.NumKVHeads * m.cfg.HeadDim
	w := m.cfg.SlidingWindow
	swa := w > 0

	maxSeq := pastLen + batch
	gpuOK := m.dev != nil && device.LayerOnGPU(layer, m.ngl, m.cfg.NumLayers) && !m.decode.GPUKVStale
	if gpuOK && (!swa || maxSeq <= w) {
		if err := device.AttentionScoresKVBatch(m.dev, layer, attn, q, batch, pastLen, m.cfg.NumHeads, m.cfg.NumKVHeads, m.cfg.HeadDim); err == nil {
			return nil
		}
	}

	if !swa {
		return compute.AttentionScoresBatchCausalInto(attn, q, m.cache.KLayer(layer), m.cache.VLayer(layer), m.scratch.scores, pastLen, batch, m.cfg.NumHeads, m.cfg.NumKVHeads, m.cfg.HeadDim)
	}

	kLayer := m.cache.KLayer(layer)
	vLayer := m.cache.VLayer(layer)
	for i := range batch {
		seqLen := pastLen + i + 1
		skip := 0
		if seqLen > w {
			skip = seqLen - 1 - w
		}
		off := i * qDim
		if err := compute.AttentionScoresInto(attn[off:off+qDim], q[off:off+qDim], kLayer[skip*kvDim:], vLayer[skip*kvDim:], m.scratch.scores, seqLen-skip, m.cfg.NumHeads, m.cfg.NumKVHeads, m.cfg.HeadDim); err != nil {
			return err
		}
	}

	return nil
}

// matmulBatchInto: W * X[batch] via MatMulMat*Cached on GPU or CPU MatMulMat*.
func (m *Model) matmulBatchInto(name string, rows, cols int, x, out []float32, batch, layer int) error {
	if m.dev != nil && device.LayerOnGPU(layer, m.ngl, m.cfg.NumLayers) {
		if err := dispatch.MatMulMatGPUInto(m.dev, m.weights, name, rows, cols, x, batch, out); err == nil {
			return nil
		}
	}

	return dispatch.MatMulMatRawInto(m.weights, name, rows, cols, x, batch, out)
}
