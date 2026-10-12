package deepseek

import (
	"fmt"

	"github.com/magomedcoder/gogguf/pkg/compute"
	"github.com/magomedcoder/gogguf/pkg/device"
	"github.com/magomedcoder/gogguf/pkg/exec/dispatch"
)

// forwardBatch - multi-token prefill (n_batch): one layer pass per chunk.
// With -ngl matmul uses MatMulMat*Cached (quant GEMM / batched device launches); attention - loop AttentionScoresKV over GPU KV; otherwise CPU GEMM + CPU causal attn.
func (m *Model) forwardBatch(tokenIDs []int, startPos int, needLogits bool) error {
	b := len(tokenIDs)
	if b == 0 {
		return fmt.Errorf("deepseek: empty batch")
	}

	if b > m.nBatch {
		return fmt.Errorf("deepseek: batch=%d > n_batch=%d", b, m.nBatch)
	}

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
	ffn := m.cfg.FFNHidden // dense only: lead layers of MoE models never reach the batch path

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

	// DeepSeek v1 RoPE: adjacent pairs (Norm)
	for i := range batch {
		pos := startPos + i
		compute.ApplyRoPEHeadsNorm(q[i*qDim:(i+1)*qDim], m.cfg.NumHeads, m.cfg.HeadDim, pos, m.cfg.RopeFreqBase)
		compute.ApplyRoPEHeadsNorm(k[i*kvDim:(i+1)*kvDim], m.cfg.NumKVHeads, m.cfg.HeadDim, pos, m.cfg.RopeFreqBase)
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
