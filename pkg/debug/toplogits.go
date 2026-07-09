package debug

// Logit - token id и значение logit
type Logit struct {
	ID    int     `json:"id"`
	Logit float32 `json:"logit"`
}

// TopLogits возвращает top-N logits по убыванию значения
func TopLogits(logits []float32, n int) []Logit {
	if n <= 0 || len(logits) == 0 {
		return nil
	}

	if n > len(logits) {
		n = len(logits)
	}

	indices := make([]int, len(logits))
	for i := range indices {
		indices[i] = i
	}

	for i := 0; i < n; i++ {
		best := i
		for j := i + 1; j < len(indices); j++ {
			if logits[indices[j]] > logits[indices[best]] {
				best = j
			}
		}

		indices[i], indices[best] = indices[best], indices[i]
	}

	out := make([]Logit, n)
	for i := 0; i < n; i++ {
		id := indices[i]
		out[i] = Logit{
			ID:    id,
			Logit: logits[id],
		}
	}

	return out
}
