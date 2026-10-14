package device

import "fmt"

// KVBatchAttentionBackend is optionally implemented for causal multi-query attention over the
// device KV cache in a single launch (q/dst are [n][nHeads*headDim]; query i sees pastLen+i+1 positions).
type KVBatchAttentionBackend interface {
	AttentionScoresKVBatch(layer int, dst, q []float32, n, pastLen, nHeads, nKVHeads, headDim int) error
}

// AttentionScoresKVBatch runs causal batch attention on the device KV cache.
// Uses KVBatchAttentionBackend when available, else loops AttentionScoresKV with seqLen=pastLen+i+1.
func AttentionScoresKVBatch(b Backend, layer int, dst, q []float32, n, pastLen, nHeads, nKVHeads, headDim int) error {
	if n < 1 {
		return fmt.Errorf("device: AttentionScoresKVBatch n=%d", n)
	}

	qDim := nHeads * headDim
	if len(dst) < n*qDim || len(q) < n*qDim {
		return fmt.Errorf("device: AttentionScoresKVBatch buffers too short")
	}

	if bb, ok := b.(KVBatchAttentionBackend); ok {
		return bb.AttentionScoresKVBatch(layer, dst, q, n, pastLen, nHeads, nKVHeads, headDim)
	}

	for i := range n {
		off := i * qDim
		if err := b.AttentionScoresKV(layer, dst[off:off+qDim], q[off:off+qDim], pastLen+i+1, nHeads, nKVHeads, headDim); err != nil {
			return err
		}
	}

	return nil
}
