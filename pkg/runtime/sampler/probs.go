package sampler

import "math"

func maxLogit(logits []float32) float32 {
	if len(logits) == 0 {
		return 0
	}

	max := logits[0]
	for _, v := range logits[1:] {
		if v > max {
			max = v
		}
	}

	return max
}

func softmaxInPlace(logits []float32) []float64 {
	max := maxLogit(logits)
	probs := make([]float64, len(logits))
	sum := 0.0
	for i, v := range logits {
		p := math.Exp(float64(v - max))
		probs[i] = p
		sum += p
	}

	if sum > 0 {
		inv := 1.0 / sum
		for i := range probs {
			probs[i] *= inv
		}
	}

	return probs
}

func applyTemperature(logits []float32, temp float32) {
	if temp <= 0 || temp == 1 {
		return
	}

	inv := 1.0 / float64(temp)
	for i := range logits {
		logits[i] = float32(float64(logits[i]) * inv)
	}
}
