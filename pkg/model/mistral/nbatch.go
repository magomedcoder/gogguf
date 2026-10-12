package mistral

// PrefillBatchSize returns the effective n_batch used for multi-token prefill.
func (m *Model) PrefillBatchSize() int {
	if m == nil || m.nBatch < 1 {
		return 1
	}

	return m.nBatch
}
