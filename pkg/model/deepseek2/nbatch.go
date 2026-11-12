package deepseek2

// PrefillBatchSize returns the effective n_batch for multi-token prefill.
// DeepSeek2 uses serial per-token MLA+MoE inside each chunk (absorbed MLA is not GEMM-batched across B).
// Legacy attn_kv_b MHA layers use the same serial token loop.
func (m *Model) PrefillBatchSize() int {
	if m == nil || m.nBatch < 1 {
		return 1
	}

	return m.nBatch
}
