package phi2

import (
	"fmt"

	"github.com/magomedcoder/gogguf/pkg/compute"
	"github.com/magomedcoder/gogguf/pkg/device"
	"github.com/magomedcoder/gogguf/pkg/exec/dispatch"
)

// forwardBatch - multi-token prefill (n_batch): one layer pass per chunk.
// With -ngl matmul uses MatMulMat*Cached; attention - loop AttentionScoresKV over GPU KV; otherwise CPU GEMM + CPU causal attn.
func (m *Model) forwardBatch(tokenIDs []int, startPos int, needLogits bool) error {
	b := len(tokenIDs)
	if b == 0 {
		return fmt.Errorf("phi2: empty batch")
	}

	if b > m.nBatch {
		return fmt.Errorf("phi2: batch=%d > n_batch=%d", b, m.nBatch)
	}

	embd := m.cfg.EmbeddingDim
	for i, tok := range tokenIDs {
		if err := dispatch.EmbedTokenInto(m.weights, m.scratch.bx[i*embd:(i+1)*embd], embd, tok); err != nil {
			return err
		}
	}

	for layer := 0; layer < m.cfg.NumLayers; layer++ {
		if err := m.forwardBlockBatch(layer, startPos, b); err != nil {
			return fmt.Errorf("phi2: layer %d pos %d: %w", layer, startPos, err)
		}
	}
	m.cache.AdvanceN(b)

	if !needLogits {
		return nil
	}

	return m.logitsFromHidden(m.scratch.bx[(b-1)*embd : b*embd])
}

func (m *Model) forwardBlockBatch(layer, startPos, batch int) error {
	ln := m.layerNorms[layer]
	lt := m.layerTensors[layer]
	lb := m.layerBias[layer]
	embd := m.cfg.EmbeddingDim
	qDim := m.cfg.NumHeads * m.cfg.HeadDim
	kvDim := m.cfg.NumKVHeads * m.cfg.HeadDim
	ffn := m.cfg.FFNHidden

	x := m.scratch.bx[:batch*embd]
	h := m.scratch.bh[:batch*embd]
	attnOut := m.scratch.battnOut[:batch*embd]
	q := m.scratch.bq[:batch*qDim]
	k := m.scratch.bk[:batch*kvDim]
	v := m.scratch.bv[:batch*kvDim]
	attn := m.scratch.battn[:batch*qDim]
	up := m.scratch.bup[:batch*ffn]

	// one LayerNorm on residual; attn and FFN in parallel
	for i := range batch {
		off := i * embd
		if err := compute.LayerNormInto(h[off:off+embd], x[off:off+embd], ln.attnW, ln.attnB, m.cfg.LayerNormEps); err != nil {
			return err
		}
	}

	if err := m.projectQKVBatch(lt, lb, layer, h, q, k, v, batch); err != nil {
		return err
	}

	for i := range batch {
		pos := startPos + i
		compute.ApplyRoPEHeadsPartial(q[i*qDim:(i+1)*qDim], m.cfg.NumHeads, m.cfg.HeadDim, m.cfg.RopeDim, pos, m.cfg.RopeFreqBase)
		compute.ApplyRoPEHeadsPartial(k[i*kvDim:(i+1)*kvDim], m.cfg.NumKVHeads, m.cfg.HeadDim, m.cfg.RopeDim, pos, m.cfg.RopeFreqBase)
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

	if err := m.matmulBatchInto(lt.attnOut, embd, qDim, attn, attnOut, batch, layer); err != nil {
		return err
	}

	for i := range batch {
		off := i * embd
		compute.AddBiasInPlace(attnOut[off:off+embd], lb.wo)
	}
	compute.AddInPlace(x, attnOut)

	// FFN from the same normed h (parallel residual)
	if err := m.matmulBatchInto(lt.ffnUp, ffn, embd, h, up, batch, layer); err != nil {
		return err
	}

	for i := range batch {
		off := i * ffn
		compute.AddBiasInPlace(up[off:off+ffn], lb.ffnUp)
	}
	compute.GELUInPlace(up)

	if err := m.matmulBatchInto(lt.ffnDown, embd, ffn, up, h, batch, layer); err != nil {
		return err
	}

	for i := range batch {
		off := i * embd
		compute.AddBiasInPlace(h[off:off+embd], lb.ffnDn)
	}
	compute.AddInPlace(x, h)

	return nil
}

// projectQKVBatch: Q/K/V (+ bias) for batch rows; fused QKV is split per row.
func (m *Model) projectQKVBatch(lt layerTensors, lb layerBias, layer int, h, q, k, v []float32, batch int) error {
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
			compute.AddBiasInPlace(row, lb.qkv)
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

	if err := m.matmulBatchInto(lt.attnV, kvDim, embd, h, v, batch, layer); err != nil {
		return err
	}

	for i := range batch {
		compute.AddBiasInPlace(q[i*qDim:(i+1)*qDim], lb.q)
		compute.AddBiasInPlace(k[i*kvDim:(i+1)*kvDim], lb.k)
		compute.AddBiasInPlace(v[i*kvDim:(i+1)*kvDim], lb.v)
	}

	return nil
}

// attentionBatchCausalInto: causal multi-query attention for batch.
// On GPU - loop AttentionScoresKV with seqLen=past+i+1; else CPU reference.
func (m *Model) attentionBatchCausalInto(attn, q []float32, layer, pastLen, batch int, gpuKV bool) error {
	qDim := m.cfg.NumHeads * m.cfg.HeadDim

	if gpuKV {
		ok := true
		for i := range batch {
			seqLen := pastLen + i + 1
			off := i * qDim
			if err := m.dev.AttentionScoresKV(layer, attn[off:off+qDim], q[off:off+qDim], seqLen, m.cfg.NumHeads, m.cfg.NumKVHeads, m.cfg.HeadDim); err != nil {
				ok = false
				break
			}
		}

		if ok {
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
