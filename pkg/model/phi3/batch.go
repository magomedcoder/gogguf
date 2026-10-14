package phi3

import (
	"fmt"

	"github.com/magomedcoder/gogguf/pkg/compute"
	"github.com/magomedcoder/gogguf/pkg/device"
	"github.com/magomedcoder/gogguf/pkg/exec/dispatch"
)

// forwardBatch - multi-token prefill (n_batch): one layer pass per chunk.
// With -ngl matmul uses MatMulMat*Cached; attention - device.AttentionScoresKVBatch over GPU KV (one launch with CUDA attn_flash); otherwise CPU GEMM + CPU causal attn.
func (m *Model) forwardBatch(tokenIDs []int, startPos int, needLogits bool) error {
	b := len(tokenIDs)
	if b == 0 {
		return fmt.Errorf("phi3: empty batch")
	}

	if b > m.nBatch {
		return fmt.Errorf("phi3: batch=%d > n_batch=%d", b, m.nBatch)
	}

	embd := m.cfg.EmbeddingDim
	for i, tok := range tokenIDs {
		if err := dispatch.EmbedTokenInto(m.weights, m.scratch.bx[i*embd:(i+1)*embd], embd, tok); err != nil {
			return err
		}
	}

	for layer := 0; layer < m.cfg.NumLayers; layer++ {
		if err := m.forwardBlockBatch(layer, startPos, b); err != nil {
			return fmt.Errorf("phi3: layer %d pos %d: %w", layer, startPos, err)
		}
	}
	m.cache.AdvanceN(b)

	if !needLogits {
		return nil
	}

	return m.logitsFromLast(m.scratch.bx[(b-1)*embd : b*embd])
}

func (m *Model) forwardBlockBatch(layer, startPos, batch int) error {
	ln := m.layerNorms[layer]
	lt := m.layerTensors[layer]
	embd := m.cfg.EmbeddingDim
	qDim := m.cfg.NumHeads * m.cfg.HeadDim
	kvDim := m.cfg.NumKVHeads * m.cfg.HeadDim

	x := m.scratch.bx[:batch*embd]
	h := m.scratch.bh[:batch*embd]
	q := m.scratch.bq[:batch*qDim]
	k := m.scratch.bk[:batch*kvDim]
	v := m.scratch.bv[:batch*kvDim]
	attn := m.scratch.battn[:batch*qDim]

	if err := compute.RMSNormBatchInto(h, x, ln.attnNorm, m.cfg.RMSNormEps, batch, embd); err != nil {
		return err
	}

	if err := m.projectQKVBatch(lt, layer, h, q, k, v, batch); err != nil {
		return err
	}

	scale := m.ropeScale()
	for i := range batch {
		pos := startPos + i
		compute.ApplyRoPEHeadsPartialScaled(q[i*qDim:(i+1)*qDim], m.cfg.NumHeads, m.cfg.HeadDim, m.cfg.RopeDim, pos, m.cfg.RopeFreqBase, scale)
		compute.ApplyRoPEHeadsPartialScaled(k[i*kvDim:(i+1)*kvDim], m.cfg.NumKVHeads, m.cfg.HeadDim, m.cfg.RopeDim, pos, m.cfg.RopeFreqBase, scale)
	}

	pastLen := m.cache.Len()
	if err := m.cache.AppendN(layer, k, v, batch); err != nil {
		return err
	}

	// mirror chunk K/V into GPU KV-cache (same best-effort policy as the serial path)
	gpuKV := m.dev != nil && device.LayerOnGPU(layer, m.ngl, m.cfg.NumLayers)
	if gpuKV {
		if err := m.dev.KVCacheAppendN(layer, pastLen, k, v, batch); err != nil {
			gpuKV = false
		}
	}

	if err := m.attentionBatchCausalInto(attn, q, layer, pastLen, batch, gpuKV); err != nil {
		return err
	}

	if err := m.matmulBatchInto(lt.attnOut, embd, qDim, attn, h, batch, layer); err != nil {
		return err
	}
	compute.AddInPlace(x, h)

	if err := compute.RMSNormBatchInto(h, x, ln.ffnNorm, m.cfg.RMSNormEps, batch, embd); err != nil {
		return err
	}

	if err := m.ffnBatch(lt, layer, h, batch); err != nil {
		return err
	}
	compute.AddInPlace(x, h)

	return nil
}

// projectQKVBatch: Q/K/V for batch rows; fused QKV is split per row.
func (m *Model) projectQKVBatch(lt layerTensors, layer int, h, q, k, v []float32, batch int) error {
	embd := m.cfg.EmbeddingDim
	qDim := m.cfg.NumHeads * m.cfg.HeadDim
	kvDim := m.cfg.NumKVHeads * m.cfg.HeadDim

	if lt.fusedQKV {
		qkvDim := qDim + 2*kvDim
		qkv := m.scratch.bqkv[:batch*qkvDim]
		if err := m.matmulBatchInto(lt.attnQKV, qkvDim, embd, h, qkv, batch, layer); err != nil {
			return err
		}

		for i := range batch {
			row := qkv[i*qkvDim : (i+1)*qkvDim]
			copy(q[i*qDim:(i+1)*qDim], row[:qDim])
			copy(k[i*kvDim:(i+1)*kvDim], row[qDim:qDim+kvDim])
			copy(v[i*kvDim:(i+1)*kvDim], row[qDim+kvDim:qDim+2*kvDim])
		}

		return nil
	}

	if err := m.matmulBatchInto(lt.attnQ, qDim, embd, h, q, batch, layer); err != nil {
		return err
	}

	if err := m.matmulBatchInto(lt.attnK, kvDim, embd, h, k, batch, layer); err != nil {
		return err
	}

	return m.matmulBatchInto(lt.attnV, kvDim, embd, h, v, batch, layer)
}

// ffnBatch: SwiGLU FFN over batch rows of h, result written back into h.
// Merged ffn_up [embd, 2*ffn] -> gate||up per row; separate gate/up otherwise.
func (m *Model) ffnBatch(lt layerTensors, layer int, h []float32, batch int) error {
	embd := m.cfg.EmbeddingDim
	ffn := m.cfg.FFNHidden
	gate := m.scratch.bgate[:batch*ffn]
	up := m.scratch.bup[:batch*ffn]

	if lt.fusedFFN {
		gateUp := m.scratch.bgateUp[:batch*2*ffn]
		if err := m.matmulBatchInto(lt.ffnUp, 2*ffn, embd, h, gateUp, batch, layer); err != nil {
			return err
		}

		for i := range batch {
			row := gateUp[i*2*ffn : (i+1)*2*ffn]
			copy(gate[i*ffn:(i+1)*ffn], row[:ffn])
			copy(up[i*ffn:(i+1)*ffn], row[ffn:])
		}
	} else {
		if err := m.matmulBatchInto(lt.ffnGate, ffn, embd, h, gate, batch, layer); err != nil {
			return err
		}

		if err := m.matmulBatchInto(lt.ffnUp, ffn, embd, h, up, batch, layer); err != nil {
			return err
		}
	}
	compute.SwiGLUInPlace(gate, up)

	return m.matmulBatchInto(lt.ffnDown, embd, ffn, gate, h, batch, layer)
}

// attentionBatchCausalInto: causal multi-query attention for batch.
// On GPU - device.AttentionScoresKVBatch (queries see past+i+1); else CPU reference.
func (m *Model) attentionBatchCausalInto(attn, q []float32, layer, pastLen, batch int, gpuKV bool) error {
	if gpuKV {
		// single launch (fused attn_flash on CUDA) when the backend supports it, else per-query loop
		if err := device.AttentionScoresKVBatch(m.dev, layer, attn, q, batch, pastLen, m.cfg.NumHeads, m.cfg.NumKVHeads, m.cfg.HeadDim); err == nil {
			return nil
		}
	}

	return compute.AttentionScoresBatchCausalInto(attn, q, m.cache.KLayer(layer), m.cache.VLayer(layer), m.scratch.scores, pastLen, batch, m.cfg.NumHeads, m.cfg.NumKVHeads, m.cfg.HeadDim)
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
