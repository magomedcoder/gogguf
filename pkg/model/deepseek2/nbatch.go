package deepseek2

// PrefillBatchSize is always 1.
// Blockers for n_batch>1: absorbed MLA batch attention + MoE expert batch dispatch.
// See batch.go.
// Do not raise this without those kernels - host serial loops alone would only re-encode the existing Forward token loop under a false batch API.
func (m *Model) PrefillBatchSize() int {
	return 1
}
