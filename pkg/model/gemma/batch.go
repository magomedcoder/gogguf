package gemma

import (
	"fmt"

	"github.com/magomedcoder/gogguf/pkg/compute"
	"github.com/magomedcoder/gogguf/pkg/device"
	"github.com/magomedcoder/gogguf/pkg/exec/dispatch"
)

// forwardBatch - multi-token prefill (n_batch): one layer pass per chunk.
// With -ngl matmul uses MatMulMat*Cached; attention is per-query (softcap / SWA aware); FFN - GeGLU.
func (m *Model) forwardBatch(tokenIDs []int, startPos int, needLogits bool) error {
	b := len(tokenIDs)
	if b == 0 {
		return fmt.Errorf("gemma: empty batch")
	}

	if b > m.nBatch {
		return fmt.Errorf("gemma: batch=%d > n_batch=%d", b, m.nBatch)
	}

	embd := m.cfg.EmbeddingDim
	for i, tok := range tokenIDs {
		row := m.scratch.bx[i*embd : (i+1)*embd]
		if err := dispatch.EmbedTokenInto(m.weights, row, embd, tok); err != nil {
			return err
		}
		compute.ScaleInPlace(row, m.cfg.EmbedScale)
	}

	for layer := 0; layer < m.cfg.NumLayers; layer++ {
		if err := m.forwardBlockBatch(layer, startPos, b); err != nil {
			return fmt.Errorf("gemma: layer %d pos %d: %w", layer, startPos, err)
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
	gemma2 := m.cfg.Variant == VariantGemma2

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

	base := m.cfg.ropeBase(layer)
	for i := range batch {
		pos := startPos + i
		compute.ApplyRoPEHeads(q[i*qDim:(i+1)*qDim], m.cfg.NumHeads, m.cfg.HeadDim, pos, base)
		compute.ApplyRoPEHeads(k[i*kvDim:(i+1)*kvDim], m.cfg.NumKVHeads, m.cfg.HeadDim, pos, base)
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

	if gemma2 {
		if err := compute.RMSNormBatchInto(h, h, ln.attnPostNorm, m.cfg.RMSNormEps, batch, embd); err != nil {
			return err
		}
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
	compute.GeGLUInPlace(gate, up)

	if err := m.matmulBatchInto(lt.ffnDown, embd, ffn, gate, h, batch, layer); err != nil {
		return err
	}

	if gemma2 {
		if err := compute.RMSNormBatchInto(h, h, ln.ffnPostNorm, m.cfg.RMSNormEps, batch, embd); err != nil {
			return err
		}
	}
	compute.AddInPlace(x, h)

	return nil
}

// attentionBatchCausalInto: causal multi-query attention for batch (query i sees past+i+1 keys).
// Gemma2: attention softcap and SWA window (same key range as serial attentionKV) applied per query;
// GPU AttentionScoresKV only without softcap / SWA clip.
func (m *Model) attentionBatchCausalInto(attn, q []float32, layer, pastLen, batch int, gpuKV bool) error {
	qDim := m.cfg.NumHeads * m.cfg.HeadDim
	kvDim := m.cfg.NumKVHeads * m.cfg.HeadDim
	softcap := m.cfg.AttnLogitSoftcap
	swa := m.cfg.IsSWALayer(layer) && m.cfg.SlidingWindow > 0

	if gpuKV && softcap == 0 && !swa {
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

	kLayer := m.cache.KLayer(layer)
	vLayer := m.cache.VLayer(layer)
	for i := range batch {
		seqLen := pastLen + i + 1
		skip := 0
		if swa && seqLen > m.cfg.SlidingWindow {
			skip = seqLen - 1 - m.cfg.SlidingWindow
		}

		off := i * qDim
		err := compute.AttentionScoresIntoSoftcap(attn[off:off+qDim], q[off:off+qDim], kLayer[skip*kvDim:], vLayer[skip*kvDim:], m.scratch.scores, seqLen-skip, m.cfg.NumHeads, m.cfg.NumKVHeads, m.cfg.HeadDim, softcap)
		if err != nil {
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
