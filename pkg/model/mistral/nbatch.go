package mistral

// PrefillBatchSize returns the effective n_batch used for multi-token prefill.
// Dense/SWA: chunk GEMM + windowed causal attn (-b>1). MoE: same n_batch with serial resid experts.
// SWA reclaim still marks GPUKVStale until A4.
func (m *Model) PrefillBatchSize() int {
	if m == nil || m.nBatch < 1 {
		return 1
	}

	return m.nBatch
}
