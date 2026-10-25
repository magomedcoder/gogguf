package mistral

import (
	"fmt"

	"github.com/magomedcoder/gogguf/pkg/compute"
	"github.com/magomedcoder/gogguf/pkg/device"
	"github.com/magomedcoder/gogguf/pkg/exec/dispatch"
)

// forwardBatch - multi-token prefill (n_batch): one layer pass per chunk.
// With -ngl matmul uses MatMulMat*Cached (quant GEMM / batched device launches); attention - device.AttentionScoresKVBatch over GPU KV (one launch with CUDA attn_flash); otherwise CPU GEMM + CPU causal attn.
func (m *Model) forwardBatch(tokenIDs []int, startPos int, needLogits bool) error {
	b := len(tokenIDs)
	if b == 0 {
		return fmt.Errorf("mistral: empty batch")
	}

	// Prefill chunk uses host scratch; pull any resident decode hidden first.
	if err := m.syncHiddenFromDevice(); err != nil {
		return err
	}

	if b > m.nBatch {
		return fmt.Errorf("mistral: batch=%d > n_batch=%d", b, m.nBatch)
	}

	m.ensureKVRoom(b)

	embd := m.cfg.EmbeddingDim
	for i, tok := range tokenIDs {
		if err := dispatch.EmbedTokenInto(m.weights, m.scratch.bx[i*embd:(i+1)*embd], embd, tok); err != nil {
			return err
		}
	}

	for layer := 0; layer < m.cfg.NumLayers; layer++ {
		if err := m.forwardBlockBatch(layer, startPos, b); err != nil {
			return err
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
	if m.dev != nil && device.LayerOnGPU(layer, m.ngl, m.cfg.NumLayers) && !m.gpuKVStale {
		if err := m.dev.KVCacheAppendN(layer, pastLen, k, v, batch); err != nil {
			m.gpuKVStale = true
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

// attentionBatchCausalInto: causal multi-query attention for batch.
// On GPU - device.AttentionScoresKVBatch (queries see past+i+1); else CPU reference.
func (m *Model) attentionBatchCausalInto(attn, q []float32, layer, pastLen, batch int) error {
	if m.dev != nil && device.LayerOnGPU(layer, m.ngl, m.cfg.NumLayers) && !m.gpuKVStale {
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
