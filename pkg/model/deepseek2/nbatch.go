package deepseek2

// PrefillBatchSize is always 1 (MLA / MoE stay serial)
func (m *Model) PrefillBatchSize() int {
	return 1
}
