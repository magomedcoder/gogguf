package sampler

// Greedy выбирает индекс с максимальным logit
func Greedy(logits []float32) int {
	if len(logits) == 0 {
		return -1
	}

	best := 0
	for i := 1; i < len(logits); i++ {
		if logits[i] > logits[best] {
			best = i
		}
	}

	return best
}
