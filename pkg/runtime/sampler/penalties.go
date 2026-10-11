package sampler

// ApplyRepeatPenalty reduces logits of repeated tokens (like llama.cpp)
// penalty == 1 - no change; > 1 - penalty for repeat
func ApplyRepeatPenalty(logits []float32, history []int, penalty float32, lastN int) {
	if penalty == 1 || len(logits) == 0 || len(history) == 0 {
		return
	}

	start := 0
	if lastN > 0 && len(history) > lastN {
		start = len(history) - lastN
	}

	seen := make(map[int]struct{}, lastN)
	for _, id := range history[start:] {
		if id < 0 || id >= len(logits) {
			continue
		}

		if _, ok := seen[id]; ok {
			continue
		}
		seen[id] = struct{}{}

		if logits[id] > 0 {
			logits[id] /= penalty
		} else {
			logits[id] *= penalty
		}
	}
}

// ApplyMinP zeroes logits of tokens with prob < minP * maxProb (after softmax)
func ApplyMinP(logits []float32, minP float32) {
	if minP <= 0 || len(logits) == 0 {
		return
	}

	probs := softmaxInPlace(logits)
	maxProb := 0.0
	for _, p := range probs {
		if p > maxProb {
			maxProb = p
		}
	}

	threshold := float64(minP) * maxProb
	for i, p := range probs {
		if p < threshold {
			logits[i] = -1e10
		}
	}
}
