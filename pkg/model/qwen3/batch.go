package qwen3

import (
	"fmt"
	"github.com/magomedcoder/gogguf/pkg/dtype"

	"github.com/magomedcoder/gogguf/pkg/compute"
	"github.com/magomedcoder/gogguf/pkg/device"
)

// forwardBatch - multi-token prefill (n_batch): one layer pass per chunk.
// With -ngl matmul uses loop-B MatMulVec*Cached (weights resident on GPU), attention -
// loop AttentionScoresKV over GPU KV; otherwise CPU GEMM + CPU causal attn (§4).
func (m *Model) forwardBatch(tokenIDs []int, startPos int, needLogits bool) error {
	b := len(tokenIDs)
	if b == 0 {
		return fmt.Errorf("qwen3: empty batch")
	}

	// batch-path host buffers: resident hidden from device is no longer needed
	if err := m.syncHiddenFromDevice(); err != nil {
		return err
	}

	if b > m.nBatch {
		return fmt.Errorf("qwen3: batch=%d > n_batch=%d", b, m.nBatch)
	}

	embd := m.cfg.EmbeddingDim
	for i, tok := range tokenIDs {
		if err := m.embedTokenInto(tok, m.scratch.x[i*embd:(i+1)*embd]); err != nil {
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

	last := m.scratch.x[(b-1)*embd : b*embd]
	if err := m.logitsFromHidden(last); err != nil {
		return err
	}

	m.emitLogitsDebug()

	// Batch prefill uses host x; restore device-resident hidden for decode when enabled.
	if m.residency && m.dev != nil {
		last := m.scratch.x[(b-1)*embd : b*embd]
		if err := m.dev.HiddenUpload(last); err != nil {
			m.residency = false
		} else {
			m.residDevice = true
		}
	}

	return nil
}

func (m *Model) forwardBlockBatch(layer, startPos, batch int) error {
	ln := m.layerNorms[layer]
	lt := m.layerTensors[layer]
	embd := m.cfg.EmbeddingDim
	qDim := m.cfg.NumHeads * m.cfg.HeadDim
	kvDim := m.cfg.NumKVHeads * m.cfg.HeadDim
	ffn := m.cfg.FFNHidden

	x := m.scratch.x[:batch*embd]
	h := m.scratch.h[:batch*embd]
	q := m.scratch.q[:batch*qDim]
	k := m.scratch.k[:batch*kvDim]
	v := m.scratch.v[:batch*kvDim]
	attn := m.scratch.attn[:batch*qDim]
	gate := m.scratch.gate[:batch*ffn]
	up := m.scratch.up[:batch*ffn]

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

	for i := range batch {
		offQ := i * qDim
		offK := i * kvDim
		if err := m.normHeadsInto(q[offQ:offQ+qDim], ln.qNorm, m.cfg.NumHeads, layer); err != nil {
			return err
		}

		if err := m.normHeadsInto(k[offK:offK+kvDim], ln.kNorm, m.cfg.NumKVHeads, layer); err != nil {
			return err
		}

		pos := startPos + i
		compute.ApplyRoPEHeads(q[offQ:offQ+qDim], m.cfg.NumHeads, m.cfg.HeadDim, pos, m.cfg.RopeFreqBase)
		compute.ApplyRoPEHeads(k[offK:offK+kvDim], m.cfg.NumKVHeads, m.cfg.HeadDim, pos, m.cfg.RopeFreqBase)
	}

	pastLen := m.cache.Len()
	if err := m.cache.AppendN(layer, k, v, batch); err != nil {
		return err
	}

	// §4: mirror chunk K/V into GPU KV-cache, else decode after batch-prefill sees incomplete cache.
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

	for i := range batch {
		off := i * embd
		compute.AddInPlace(x[off:off+embd], h[off:off+embd])
	}

	if err := compute.RMSNormBatchInto(h, x, ln.ffnNorm, m.cfg.RMSNormEps, batch, embd); err != nil {
		return err
	}

	if err := m.matmulBatchInto(lt.ffnGate, ffn, embd, h, gate, batch, layer); err != nil {
		return err
	}

	if err := m.matmulBatchInto(lt.ffnUp, ffn, embd, h, up, batch, layer); err != nil {
		return err
	}

	for i := range batch {
		off := i * ffn
		compute.SwiGLUInPlace(gate[off:off+ffn], up[off:off+ffn])
	}

	if err := m.matmulBatchInto(lt.ffnDown, embd, ffn, gate, h, batch, layer); err != nil {
		return err
	}

	for i := range batch {
		off := i * embd
		compute.AddInPlace(x[off:off+embd], h[off:off+embd])
	}

	return nil
}

// attentionBatchCausalInto: causal multi-query attention for batch.
// On GPU - loop AttentionScoresKV with seqLen=past+i+1; else CPU reference.
func (m *Model) attentionBatchCausalInto(attn, q []float32, layer, pastLen, batch int) error {
	qDim := m.cfg.NumHeads * m.cfg.HeadDim

	if m.dev != nil && device.LayerOnGPU(layer, m.ngl, m.cfg.NumLayers) && !m.gpuKVStale {
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

// matmulBatchInto: W * X[batch].
// With -ngl - loop MatMulVec*Cached (weights already on GPU).
func (m *Model) matmulBatchInto(name string, rows, cols int, x, out []float32, batch, layer int) error {
	if m.dev != nil && device.LayerOnGPU(layer, m.ngl, m.cfg.NumLayers) {
		if err := m.matmulBatchGPU(name, rows, cols, x, out, batch); err == nil {
			return nil
		}
	}

	return m.matmulBatchCPU(name, rows, cols, x, out, batch)
}

func (m *Model) matmulBatchGPU(name string, rows, cols int, x, out []float32, batch int) error {
	for b := range batch {
		dst := out[b*rows : (b+1)*rows]
		if err := m.matmulGPUInto(name, rows, cols, x[b*cols:(b+1)*cols], dst); err != nil {
			return err
		}
	}

	return nil
}

func (m *Model) matmulBatchCPU(name string, rows, cols int, x, out []float32, batch int) error {
	raw, err := m.weights.Raw(name)
	if err != nil {
		return err
	}

	info, err := m.weights.Info(name)
	if err != nil {
		return err
	}

	switch info.Type {
	case dtype.Q8_0:
		return compute.MatMulMatQ8_0Into(raw, rows, cols, x, batch, out)
	case dtype.Q4_0:
		return compute.MatMulMatViaVecInto(batch, rows, cols, x, out, func(vec, dst []float32) error {
			return compute.MatMulVecQ4_0Into(raw, rows, cols, vec, dst)
		})
	case dtype.Q4_1:
		return compute.MatMulMatViaVecInto(batch, rows, cols, x, out, func(vec, dst []float32) error {
			return compute.MatMulVecQ4_1Into(raw, rows, cols, vec, dst)
		})
	case dtype.Q5_0:
		return compute.MatMulMatViaVecInto(batch, rows, cols, x, out, func(vec, dst []float32) error {
			return compute.MatMulVecQ5_0Into(raw, rows, cols, vec, dst)
		})
	case dtype.Q5_1:
		return compute.MatMulMatViaVecInto(batch, rows, cols, x, out, func(vec, dst []float32) error {
			return compute.MatMulVecQ5_1Into(raw, rows, cols, vec, dst)
		})
	case dtype.Q4_K:
		return compute.MatMulMatViaVecInto(batch, rows, cols, x, out, func(vec, dst []float32) error {
			return compute.MatMulVecQ4_KInto(raw, rows, cols, vec, dst)
		})
	case dtype.Q5_K:
		return compute.MatMulMatViaVecInto(batch, rows, cols, x, out, func(vec, dst []float32) error {
			return compute.MatMulVecQ5_KInto(raw, rows, cols, vec, dst)
		})
	case dtype.Q6_K:
		return compute.MatMulMatViaVecInto(batch, rows, cols, x, out, func(vec, dst []float32) error {
			return compute.MatMulVecQ6_KInto(raw, rows, cols, vec, dst)
		})
	case dtype.Q2_K:
		return compute.MatMulMatViaVecInto(batch, rows, cols, x, out, func(vec, dst []float32) error {
			return compute.MatMulVecQ2_KInto(raw, rows, cols, vec, dst)
		})
	case dtype.Q3_K:
		return compute.MatMulMatViaVecInto(batch, rows, cols, x, out, func(vec, dst []float32) error {
			return compute.MatMulVecQ3_KInto(raw, rows, cols, vec, dst)
		})
	case dtype.Q8_K:
		return compute.MatMulMatViaVecInto(batch, rows, cols, x, out, func(vec, dst []float32) error {
			return compute.MatMulVecQ8_KInto(raw, rows, cols, vec, dst)
		})
	default:
		f32, err := m.weights.Floats(name)
		if err != nil {
			return err
		}

		return compute.MatMulMatInto(f32, rows, cols, x, batch, out)
	}
}

func (m *Model) embedTokenInto(tokenID int, dst []float32) error {
	raw, err := m.weights.Raw("token_embd.weight")
	if err != nil {
		return err
	}

	info, err := m.weights.Info("token_embd.weight")
	if err != nil {
		return err
	}

	switch info.Type {
	case dtype.Q8_0:
		return compute.EmbeddingQ8_0Into(dst, raw, m.cfg.EmbeddingDim, tokenID)
	case dtype.Q4_0:
		return compute.EmbeddingQ4_0Into(dst, raw, m.cfg.EmbeddingDim, tokenID)
	case dtype.Q4_1:
		return compute.EmbeddingQ4_1Into(dst, raw, m.cfg.EmbeddingDim, tokenID)
	case dtype.Q5_0:
		return compute.EmbeddingQ5_0Into(dst, raw, m.cfg.EmbeddingDim, tokenID)
	case dtype.Q5_1:
		return compute.EmbeddingQ5_1Into(dst, raw, m.cfg.EmbeddingDim, tokenID)
	case dtype.Q4_K:
		return compute.EmbeddingQ4_KInto(dst, raw, m.cfg.EmbeddingDim, tokenID)
	case dtype.Q5_K:
		return compute.EmbeddingQ5_KInto(dst, raw, m.cfg.EmbeddingDim, tokenID)
	case dtype.Q6_K:
		return compute.EmbeddingQ6_KInto(dst, raw, m.cfg.EmbeddingDim, tokenID)
	case dtype.Q2_K:
		return compute.EmbeddingQ2_KInto(dst, raw, m.cfg.EmbeddingDim, tokenID)
	case dtype.Q3_K:
		return compute.EmbeddingQ3_KInto(dst, raw, m.cfg.EmbeddingDim, tokenID)
	case dtype.Q8_K:
		return compute.EmbeddingQ8_KInto(dst, raw, m.cfg.EmbeddingDim, tokenID)
	default:
		f32, err := m.weights.Floats("token_embd.weight")
		if err != nil {
			return err
		}

		off := tokenID * m.cfg.EmbeddingDim
		copy(dst, f32[off:off+m.cfg.EmbeddingDim])

		return nil
	}
}
