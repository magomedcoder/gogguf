package qwen3

import (
	"fmt"

	"github.com/magomedcoder/gogguf/pkg/format"
	"github.com/magomedcoder/gogguf/pkg/ops"
)

// forwardBatch - true multi-token prefill (n_batch): один проход слоёв на chunk.
// GPU residency для B>1 пока не используется (CPU-путь).
func (m *Model) forwardBatch(tokenIDs []int, startPos int, needLogits bool) error {
	b := len(tokenIDs)
	if b == 0 {
		return fmt.Errorf("qwen3: пустой batch")
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

	if err := ops.RMSNormBatchInto(h, x, ln.attnNorm, m.cfg.RMSNormEps, batch, embd); err != nil {
		return err
	}

	if err := m.matmulBatchInto(lt.attnQ, qDim, embd, h, q, batch); err != nil {
		return err
	}

	if err := m.matmulBatchInto(lt.attnK, kvDim, embd, h, k, batch); err != nil {
		return err
	}

	if err := m.matmulBatchInto(lt.attnV, kvDim, embd, h, v, batch); err != nil {
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
		ops.ApplyRoPEHeads(q[offQ:offQ+qDim], m.cfg.NumHeads, m.cfg.HeadDim, pos, m.cfg.RopeFreqBase)
		ops.ApplyRoPEHeads(k[offK:offK+kvDim], m.cfg.NumKVHeads, m.cfg.HeadDim, pos, m.cfg.RopeFreqBase)
	}

	pastLen := m.cache.Len()
	m.cache.AppendN(layer, k, v, batch)

	if err := ops.AttentionScoresBatchCausalInto(
		attn, q, m.cache.KLayer(layer), m.cache.VLayer(layer), m.scratch.scores,
		pastLen, batch, m.cfg.NumHeads, m.cfg.NumKVHeads, m.cfg.HeadDim,
	); err != nil {
		return err
	}

	if err := m.matmulBatchInto(lt.attnOut, embd, qDim, attn, h, batch); err != nil {
		return err
	}

	for i := range batch {
		off := i * embd
		ops.AddInPlace(x[off:off+embd], h[off:off+embd])
	}

	if err := ops.RMSNormBatchInto(h, x, ln.ffnNorm, m.cfg.RMSNormEps, batch, embd); err != nil {
		return err
	}

	if err := m.matmulBatchInto(lt.ffnGate, ffn, embd, h, gate, batch); err != nil {
		return err
	}

	if err := m.matmulBatchInto(lt.ffnUp, ffn, embd, h, up, batch); err != nil {
		return err
	}

	for i := range batch {
		off := i * ffn
		ops.SwiGLUInPlace(gate[off:off+ffn], up[off:off+ffn])
	}

	if err := m.matmulBatchInto(lt.ffnDown, embd, ffn, gate, h, batch); err != nil {
		return err
	}

	for i := range batch {
		off := i * embd
		ops.AddInPlace(x[off:off+embd], h[off:off+embd])
	}

	return nil
}

func (m *Model) matmulBatchInto(name string, rows, cols int, x, out []float32, batch int) error {
	raw, err := m.weights.Raw(name)
	if err != nil {
		return err
	}

	info, err := m.weights.Info(name)
	if err != nil {
		return err
	}

	switch info.Type {
	case format.GgmlQ8_0:
		return ops.MatMulMatQ8_0Into(raw, rows, cols, x, batch, out)
	case format.GgmlQ4_0:
		return ops.MatMulMatViaVecInto(batch, rows, cols, x, out, func(vec, dst []float32) error {
			return ops.MatMulVecQ4_0Into(raw, rows, cols, vec, dst)
		})
	case format.GgmlQ4_1:
		return ops.MatMulMatViaVecInto(batch, rows, cols, x, out, func(vec, dst []float32) error {
			return ops.MatMulVecQ4_1Into(raw, rows, cols, vec, dst)
		})
	case format.GgmlQ5_0:
		return ops.MatMulMatViaVecInto(batch, rows, cols, x, out, func(vec, dst []float32) error {
			return ops.MatMulVecQ5_0Into(raw, rows, cols, vec, dst)
		})
	case format.GgmlQ5_1:
		return ops.MatMulMatViaVecInto(batch, rows, cols, x, out, func(vec, dst []float32) error {
			return ops.MatMulVecQ5_1Into(raw, rows, cols, vec, dst)
		})
	case format.GgmlQ4_K:
		return ops.MatMulMatViaVecInto(batch, rows, cols, x, out, func(vec, dst []float32) error {
			return ops.MatMulVecQ4_KInto(raw, rows, cols, vec, dst)
		})
	case format.GgmlQ5_K:
		return ops.MatMulMatViaVecInto(batch, rows, cols, x, out, func(vec, dst []float32) error {
			return ops.MatMulVecQ5_KInto(raw, rows, cols, vec, dst)
		})
	case format.GgmlQ6_K:
		return ops.MatMulMatViaVecInto(batch, rows, cols, x, out, func(vec, dst []float32) error {
			return ops.MatMulVecQ6_KInto(raw, rows, cols, vec, dst)
		})
	case format.GgmlQ2_K:
		return ops.MatMulMatViaVecInto(batch, rows, cols, x, out, func(vec, dst []float32) error {
			return ops.MatMulVecQ2_KInto(raw, rows, cols, vec, dst)
		})
	case format.GgmlQ3_K:
		return ops.MatMulMatViaVecInto(batch, rows, cols, x, out, func(vec, dst []float32) error {
			return ops.MatMulVecQ3_KInto(raw, rows, cols, vec, dst)
		})
	case format.GgmlQ8_K:
		return ops.MatMulMatViaVecInto(batch, rows, cols, x, out, func(vec, dst []float32) error {
			return ops.MatMulVecQ8_KInto(raw, rows, cols, vec, dst)
		})
	default:
		f32, err := m.weights.Floats(name)
		if err != nil {
			return err
		}

		return ops.MatMulMatInto(f32, rows, cols, x, batch, out)
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
	case format.GgmlQ8_0:
		return ops.EmbeddingQ8_0Into(dst, raw, m.cfg.EmbeddingDim, tokenID)
	case format.GgmlQ4_0:
		return ops.EmbeddingQ4_0Into(dst, raw, m.cfg.EmbeddingDim, tokenID)
	case format.GgmlQ4_1:
		return ops.EmbeddingQ4_1Into(dst, raw, m.cfg.EmbeddingDim, tokenID)
	case format.GgmlQ5_0:
		return ops.EmbeddingQ5_0Into(dst, raw, m.cfg.EmbeddingDim, tokenID)
	case format.GgmlQ5_1:
		return ops.EmbeddingQ5_1Into(dst, raw, m.cfg.EmbeddingDim, tokenID)
	case format.GgmlQ4_K:
		return ops.EmbeddingQ4_KInto(dst, raw, m.cfg.EmbeddingDim, tokenID)
	case format.GgmlQ5_K:
		return ops.EmbeddingQ5_KInto(dst, raw, m.cfg.EmbeddingDim, tokenID)
	case format.GgmlQ6_K:
		return ops.EmbeddingQ6_KInto(dst, raw, m.cfg.EmbeddingDim, tokenID)
	case format.GgmlQ2_K:
		return ops.EmbeddingQ2_KInto(dst, raw, m.cfg.EmbeddingDim, tokenID)
	case format.GgmlQ3_K:
		return ops.EmbeddingQ3_KInto(dst, raw, m.cfg.EmbeddingDim, tokenID)
	case format.GgmlQ8_K:
		return ops.EmbeddingQ8_KInto(dst, raw, m.cfg.EmbeddingDim, tokenID)
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
