package deepseek2

// PrefillBatchSize is always 1.
// MLA attention + MoE experts are kept serial: there is no forwardBatch path (unlike deepseek/llama).
// Raising n_batch would need MLA batch kernels first.
func (m *Model) PrefillBatchSize() int {
	return 1
}
