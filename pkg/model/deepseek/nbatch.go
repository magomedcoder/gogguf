package deepseek

// PrefillBatchSize returns the effective n_batch used for multi-token prefill.
// Dense: chunk GEMM/attn. MoE: same n_batch with serial per-token resid experts.
func (m *Model) PrefillBatchSize() int {
	if m == nil || m.nBatch < 1 {
		return 1
	}

	return m.nBatch
}
